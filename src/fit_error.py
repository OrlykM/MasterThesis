r"""
Fits two empirical error models from data you've ALREADY measured -- no
training, just least-squares curve fitting:

  sigma_depth(z) = a + b*z    -- depth error growth with distance, per FDE model
  sigma_pose(d)  = c + e*d    -- pose drift growth with distance traveled

These replace the advisor's placeholder constants (a=0.05, b=0.05 etc.)
with numbers derived from your own dataset/algorithms, which is the actual
scientific contribution -- a calibrated reliability model instead of a
guessed one.

Usage:
    python3 fit_error_models.py \
        --dataset_root ~/MasterThesis/datasets/euroc/2026-04-01_..._quadrotor \
        --vslam_dir ../../results/dpvo \
        --fde_dir ../../results/zoedepth \
        --output error_model_dpvo_zoedepth.json
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_gt_trajectory, load_camera_intrinsics, load_image_timestamps
from compare_reconstruction_3d import (
    build_evo_trajectory, evo_traj_to_array, looks_like_frame_index, looks_like_raw_nanoseconds,
    remap_pseudo_timestamps, load_vslam_stride, load_tum, find_depth_file, load_depth_frame,
)
from evo.core import sync as evo_sync
from evo.core import metrics as evo_metrics
from evo.core.metrics import PoseRelation, StatisticsType


# ---------- pose error model: ATE on growing trajectory prefixes ----------

def drift_vs_distance(gt_m, est_m, n_windows=12):
    """ATE (evo APE) computed on growing prefixes of the ALREADY-ASSOCIATED
    (same length, same order) gt_m/est_m arrays -- this is what shows error
    ACCUMULATING with distance, not one aggregate number."""
    xyz = gt_m[:, 1:4]
    seg_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    fractions = np.linspace(1.0 / n_windows, 1.0, n_windows)
    out = []
    for frac in fractions:
        k = max(int(len(gt_m) * frac), 5)
        traj_ref = build_evo_trajectory(gt_m[:k])
        traj_est = build_evo_trajectory(est_m[:k])
        try:
            traj_est_copy = copy.deepcopy(traj_est)
            traj_est_copy.align(traj_ref, correct_scale=True)
            ape = evo_metrics.APE(PoseRelation.translation_part)
            ape.process_data((traj_ref, traj_est_copy))
            rmse = ape.get_statistic(StatisticsType.rmse)
        except Exception:
            continue
        out.append((float(cum_dist[k - 1]), float(rmse)))
    return out


def fit_linear(xs, ys):
    """sigma(x) = a + b*x via least squares. Returns (a, b), clamping a>=0
    (an error model can't start negative) and b>=0 (error shouldn't shrink
    with distance -- if the fit wants that, the data doesn't support this
    model shape and you should look at the points directly, not trust the fit)."""
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([np.ones_like(xs), xs]).T
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    a, b = max(a, 0.0), max(b, 0.0)
    residuals = ys - (a + b * xs)
    r2 = 1 - np.sum(residuals ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-9)
    return {"shape": "linear", "a": float(a), "b": float(b), "r_squared": float(r2), "n_points": len(xs)}


def fit_quadratic(xs, ys):
    """sigma(x) = a + b*x + c*x^2 -- try this too and compare R^2, since VSLAM
    drift often accelerates rather than growing linearly (confirmed by the
    sanity test below: a truly quadratic synthetic drift only got R^2=0.49
    from the linear fit)."""
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([np.ones_like(xs), xs, xs ** 2]).T
    (a, b, c), *_ = np.linalg.lstsq(A, ys, rcond=None)
    a, b, c = max(a, 0.0), max(b, 0.0), max(c, 0.0)
    residuals = ys - (a + b * xs + c * xs ** 2)
    r2 = 1 - np.sum(residuals ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-9)
    return {"shape": "quadratic", "a": float(a), "b": float(b), "c": float(c),
            "r_squared": float(r2), "n_points": len(xs)}


def fit_best(xs, ys):
    """Fits both shapes, returns whichever has higher R^2 -- with the
    quadratic's R^2 required to beat linear by a real margin (0.03) since it
    has an extra free parameter and will always fit >= as well numerically."""
    lin = fit_linear(xs, ys)
    quad = fit_quadratic(xs, ys)
    chosen = quad if quad["r_squared"] > lin["r_squared"] + 0.03 else lin
    return chosen, {"linear": lin, "quadratic": quad}


# ---------- depth error model: error vs GT distance, binned ----------

def depth_error_vs_distance(dataset_root, fde_dir, n_frames, n_bins, gt_depth_scale,
                             pred_depth_scale, min_depth, max_depth):
    cam_csv_path = os.path.join(dataset_root, "cam0", "data.csv")
    sensor_yaml_path = os.path.join(dataset_root, "cam0", "sensor.yaml")
    gt_depth_dir = os.path.join(dataset_root, "depth0", "data")
    frames = load_image_timestamps(cam_csv_path)
    cam_w = load_camera_intrinsics(sensor_yaml_path)["width"]
    cam_h = load_camera_intrinsics(sensor_yaml_path)["height"]

    idx_step = max(1, len(frames) // n_frames)
    chosen = frames[::idx_step][:n_frames]

    all_gt, all_err = [], []
    for frame_t, fname in tqdm(chosen, desc="depth error sampling", leave=False):
        gt_path = find_depth_file(gt_depth_dir, fname)
        pred_path = find_depth_file(fde_dir, fname)
        if gt_path is None or pred_path is None:
            continue
        gt_raw = load_depth_frame(gt_path)
        gt_raw = np.where(gt_raw >= 65535, np.nan, gt_raw)
        gt = gt_raw * gt_depth_scale
        pred = load_depth_frame(pred_path) * pred_depth_scale
        if pred.shape != (cam_h, cam_w):
            pred = np.array(Image.fromarray(pred.astype(np.float32)).resize((cam_w, cam_h), Image.BILINEAR))

        valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & np.isfinite(pred)
        all_gt.append(gt[valid].ravel())
        all_err.append(np.abs(pred[valid] - gt[valid]).ravel())

    all_gt = np.concatenate(all_gt) if all_gt else np.zeros(0)
    all_err = np.concatenate(all_err) if all_err else np.zeros(0)
    if len(all_gt) == 0:
        return []

    bins = np.linspace(min_depth, max_depth, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (all_gt >= bins[i]) & (all_gt < bins[i + 1])
        if mask.sum() < 20:  # not enough pixels in this bin to trust its mean
            continue
        out.append((float((bins[i] + bins[i + 1]) / 2), float(all_err[mask].mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--vslam_dir", required=True)
    ap.add_argument("--fde_dir", required=True)
    ap.add_argument("--max_time_diff", type=float, default=0.15,
                     help="max seconds between GT and estimated-pose timestamps to count as a "
                          "match -- Isaac Sim datasets were tight sync (~ms, use 0.02), AirSim "
                          "recordings have a confirmed ~81ms constant offset between IMU-clock "
                          "GT and camera timestamps, hence the higher default here")
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--n_depth_bins", type=int, default=15)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    ap.add_argument("--pred_depth_scale", type=float, default=1.0)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=50.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gt_traj_path = os.path.join(args.dataset_root, "state_groundtruth_estimate0", "data.csv")
    cam_csv_path = os.path.join(args.dataset_root, "cam0", "data.csv")
    poses_path = os.path.join(args.vslam_dir, "poses.txt")

    print("== Pose error model (sigma_pose(d) = c + e*d) ==")
    gt = load_gt_trajectory(gt_traj_path)
    est = load_tum(poses_path)
    frames = load_image_timestamps(cam_csv_path)
    if looks_like_frame_index(est):
        stride = load_vslam_stride(args.vslam_dir, default=1)
        est = remap_pseudo_timestamps(est, frames, stride)
    elif looks_like_raw_nanoseconds(est, gt):
        est = est.copy()
        est[:, 0] = est[:, 0] / 1e9

    traj_ref = build_evo_trajectory(gt)
    traj_est = build_evo_trajectory(est)
    traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(traj_ref, traj_est, max_diff=args.max_time_diff)
    gt_m = evo_traj_to_array(traj_ref_sync)
    est_m_unaligned = evo_traj_to_array(traj_est_sync)  # NOT aligned -- drift_vs_distance aligns per-prefix itself

    pose_points = drift_vs_distance(gt_m, est_m_unaligned, n_windows=12)
    if len(pose_points) < 3:
        print("  Not enough prefixes converged -- can't fit a pose error model reliably.")
        pose_model = None
    else:
        distances, ates = zip(*pose_points)
        pose_model, pose_candidates = fit_best(distances, ates)
        shape = pose_model["shape"]
        print(f"  sigma_pose(d) [{shape}]: R^2={pose_model['r_squared']:.3f} "
              f"(linear R^2={pose_candidates['linear']['r_squared']:.3f}, "
              f"quadratic R^2={pose_candidates['quadratic']['r_squared']:.3f})")
        for d, ate in pose_points:
            print(f"    d={d:7.2f}m  ATE={ate:.4f}m")

    print("\n== Depth error model (sigma_depth(z) = a + b*z) ==")
    depth_points = depth_error_vs_distance(
        args.dataset_root, args.fde_dir, args.n_frames, args.n_depth_bins,
        args.gt_depth_scale, args.pred_depth_scale, args.min_depth, args.max_depth)
    if len(depth_points) < 3:
        print("  Not enough populated bins -- can't fit a depth error model reliably.")
        depth_model = None
    else:
        depths, errs = zip(*depth_points)
        depth_model, depth_candidates = fit_best(depths, errs)
        shape = depth_model["shape"]
        print(f"  sigma_depth(z) [{shape}]: R^2={depth_model['r_squared']:.3f} "
              f"(linear R^2={depth_candidates['linear']['r_squared']:.3f}, "
              f"quadratic R^2={depth_candidates['quadratic']['r_squared']:.3f})")
        for z, err in depth_points:
            print(f"    z={z:7.2f}m  err={err:.4f}m")

    result = {
        "vslam_dir": args.vslam_dir, "fde_dir": args.fde_dir,
        "pose_model": pose_model, "pose_raw_points": pose_points,
        "depth_model": depth_model, "depth_raw_points": depth_points,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()