import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_gt_trajectory, load_camera_intrinsics, load_image_timestamps, pose_row_to_T
from compare_reconstruction_3d import (
    build_evo_trajectory, evo_traj_to_array, looks_like_frame_index, looks_like_raw_nanoseconds,
    remap_pseudo_timestamps, load_vslam_stride, load_tum, find_depth_file, load_depth_frame,
    backproject_depth,
)
from evo.core import sync as evo_sync

VSLAM_ALGOS = ["dpvo", "orbslam3"]
FDE_ALGOS = ["zoedepth", "depthanythingv2", "fastdepth"]


def load_run_config(results_dir, algo):
    path = os.path.join(results_dir, algo, "run_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_trajectory_summary(comparisons_dir, vslam, fde):
    path = os.path.join(comparisons_dir, f"{vslam}_{fde}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def gather_points_for_combo(dataset_root, vslam_dir, fde_dir, n_frames, max_points_per_frame, gt_depth_scale, pred_depth_scale, min_depth, max_depth):
    gt_traj_path = os.path.join(dataset_root, "state_groundtruth_estimate0", "data.csv")
    cam_csv_path = os.path.join(dataset_root, "cam0", "data.csv")
    sensor_yaml_path = os.path.join(dataset_root, "cam0", "sensor.yaml")
    gt_depth_dir = os.path.join(dataset_root, "depth0", "data")
    poses_path = os.path.join(vslam_dir, "poses.txt")

    gt = load_gt_trajectory(gt_traj_path)
    est = load_tum(poses_path)
    frames = load_image_timestamps(cam_csv_path)

    if looks_like_frame_index(est):
        stride = load_vslam_stride(vslam_dir, default=1)
        est = remap_pseudo_timestamps(est, frames, stride)
    elif looks_like_raw_nanoseconds(est, gt):
        est = est.copy()
        est[:, 0] = est[:, 0] / 1e9

    traj_ref = build_evo_trajectory(gt)
    traj_est = build_evo_trajectory(est)
    traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(traj_ref, traj_est, max_diff=0.02)
    traj_est_sync.align(traj_ref_sync, correct_scale=True)

    K = load_camera_intrinsics(sensor_yaml_path)["K"]
    cam_w = load_camera_intrinsics(sensor_yaml_path)["width"]
    cam_h = load_camera_intrinsics(sensor_yaml_path)["height"]
    gt_m = evo_traj_to_array(traj_ref_sync)
    est_m = evo_traj_to_array(traj_est_sync)
    gt_t, est_t = gt_m[:, 0], est_m[:, 0]

    idx_step = max(1, len(frames) // n_frames)
    chosen = frames[::idx_step][:n_frames]
    rng = np.random.default_rng(0)

    gt_pts_all, est_pts_all = [], []
    for frame_t, fname in tqdm(chosen, desc=f"{os.path.basename(vslam_dir)}+{os.path.basename(fde_dir)}", leave=False):
        j_gt = int(np.argmin(np.abs(gt_t - frame_t)))
        j_est = int(np.argmin(np.abs(est_t - frame_t)))
        if abs(gt_t[j_gt] - frame_t) > 0.02 or abs(est_t[j_est] - frame_t) > 0.02:
            continue
        gt_depth_path = find_depth_file(gt_depth_dir, fname)
        pred_depth_path = find_depth_file(fde_dir, fname)
        if gt_depth_path is None or pred_depth_path is None:
            continue

        gt_depth_raw = load_depth_frame(gt_depth_path)
        gt_depth_raw = np.where(gt_depth_raw >= 65535, np.nan, gt_depth_raw)
        gt_depth = gt_depth_raw * gt_depth_scale
        pred_depth = load_depth_frame(pred_depth_path) * pred_depth_scale
        if pred_depth.shape != (cam_h, cam_w):
            pred_depth = np.array(Image.fromarray(pred_depth.astype(np.float32)).resize((cam_w, cam_h), Image.BILINEAR))

        T_wc_gt = pose_row_to_T(gt_m[j_gt])
        gt_pts = backproject_depth(gt_depth, K, T_wc_gt, min_depth=min_depth, max_depth=max_depth,
                                    max_points=max_points_per_frame, rng=rng)
        gt_pts_all.append(gt_pts)

        T_wc_est = pose_row_to_T(est_m[j_est])
        est_pts = backproject_depth(pred_depth, K, T_wc_est, min_depth=min_depth, max_depth=max_depth,
                                     max_points=max_points_per_frame, rng=rng)
        est_pts_all.append(est_pts)

    gt_points = np.concatenate(gt_pts_all, axis=0) if gt_pts_all else np.zeros((0, 3))
    est_points = np.concatenate(est_pts_all, axis=0) if est_pts_all else np.zeros((0, 3))
    return gt_points, est_points


def build_occupancy_grid(points, bounds_min, bounds_max, resolution):
    dims = np.maximum(np.ceil((np.array(bounds_max) - np.array(bounds_min)) / resolution).astype(int), 1)
    grid = np.zeros(dims, dtype=bool)
    if len(points) == 0:
        return grid
    idx = np.floor((points - np.array(bounds_min)) / resolution).astype(int)
    valid = np.all((idx >= 0) & (idx < dims), axis=1)
    idx = idx[valid]
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def occupancy_iou_fpr(gt_grid, pred_grid):
    tp = int(np.logical_and(gt_grid, pred_grid).sum())
    fp = int(np.logical_and(~gt_grid, pred_grid).sum())
    fn = int(np.logical_and(gt_grid, ~pred_grid).sum())
    union = tp + fp + fn
    pred_occupied = int(pred_grid.sum())
    return {
        "occupancy_iou": float(tp / union) if union > 0 else 0.0,
        "false_positive_voxel_rate": float(fp / pred_occupied) if pred_occupied > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--results_dir", default="../../results")
    ap.add_argument("--comparisons_dir", default="../../comparisons")
    ap.add_argument("--voxel_resolution", type=float, default=0.5, help="meters per voxel for the IoU/FPR grid")
    ap.add_argument("--n_frames", type=int, default=60, help="frames to backproject per combo -- denser than the visualizer default since this feeds a metric, not just a picture")
    ap.add_argument("--max_points_per_frame", type=int, default=4000)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    ap.add_argument("--pred_depth_scale", type=float, default=1.0)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=50.0)
    ap.add_argument("--output", default="summary_table.csv")
    args = ap.parse_args()

    print("== Parameters used (from run_config.json) ==")
    all_algos = VSLAM_ALGOS + FDE_ALGOS
    param_rows = []
    for algo in all_algos:
        cfg = load_run_config(args.results_dir, algo)
        if cfg is None:
            print(f"  {algo}: no run_config.json found -- skipped")
            continue
        row = {"algorithm": cfg.get("algorithm", algo), "checkpoint": cfg.get("model_checkpoint", cfg.get("network_checkpoint", "?"))}
        row.update({f"param.{k}": v for k, v in cfg.get("params", {}).items()})
        param_rows.append(row)
        print(f"  {algo}: {row}")

    print("\n== Computing Occupancy IoU / FPR per combo (this backprojects real point clouds, takes a bit) ==")
    combo_rows = []
    for vslam in VSLAM_ALGOS:
        for fde in FDE_ALGOS:
            vslam_dir = os.path.join(args.results_dir, vslam)
            fde_dir = os.path.join(args.results_dir, fde)
            if not os.path.exists(os.path.join(vslam_dir, "poses.txt")) or not os.path.isdir(fde_dir):
                continue

            traj_summary = load_trajectory_summary(args.comparisons_dir, vslam, fde) or {}

            gt_points, est_points = gather_points_for_combo(
                args.dataset_root, vslam_dir, fde_dir, args.n_frames, args.max_points_per_frame,
                args.gt_depth_scale, args.pred_depth_scale, args.min_depth, args.max_depth)

            if len(gt_points) == 0:
                print(f"  {vslam}+{fde}: no points backprojected -- skipping occupancy metrics")
                occ = {"occupancy_iou": None, "false_positive_voxel_rate": None}
            else:
                combined = np.concatenate([gt_points, est_points], axis=0) if len(est_points) else gt_points
                bounds_min = combined.min(axis=0) - 1.0
                bounds_max = combined.max(axis=0) + 1.0
                gt_grid = build_occupancy_grid(gt_points, bounds_min, bounds_max, args.voxel_resolution)
                pred_grid = build_occupancy_grid(est_points, bounds_min, bounds_max, args.voxel_resolution)
                occ = occupancy_iou_fpr(gt_grid, pred_grid)

            row = {
                "vslam": vslam, "fde": fde,
                "ate_rmse_m": traj_summary.get("ate_rmse_m"),
                "rte_rmse_m": traj_summary.get("rte_rmse_m"),
                "rre_rmse_deg": traj_summary.get("rre_rmse_deg"),
                "depth_abs_rel": traj_summary.get("depth_abs_rel_sampled"),
                "occupancy_iou": occ["occupancy_iou"],
                "false_positive_voxel_rate": occ["false_positive_voxel_rate"],
            }
            combo_rows.append(row)
            print(f"  {vslam}+{fde}: IoU={occ['occupancy_iou']}, FPR={occ['false_positive_voxel_rate']}")

    print("\n== Summary table ==")
    headers = ["vslam", "fde", "ate_rmse_m", "rte_rmse_m", "rre_rmse_deg", "depth_abs_rel", "occupancy_iou", "false_positive_voxel_rate"]
    col_w = {h: max(len(h), 10) for h in headers}
    print(" | ".join(h.ljust(col_w[h]) for h in headers))
    for row in combo_rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(" | ".join(fmt(row[h]).ljust(col_w[h]) for h in headers))

    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(combo_rows)
    print(f"\nSaved: {args.output}")

    params_path = args.output.rsplit(".", 1)[0] + "_params.json"
    with open(params_path, "w") as f:
        json.dump(param_rows, f, indent=2)
    print(f"Saved: {params_path}")


if __name__ == "__main__":
    main()