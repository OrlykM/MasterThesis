import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync as evo_sync
from evo.core import metrics as evo_metrics
from evo.core.metrics import PoseRelation, StatisticsType, Unit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_gt_trajectory, load_camera_intrinsics, load_image_timestamps, pose_row_to_T, quat_to_rotmat

def load_tum(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append([float(x) for x in parts[:8]])
    return np.array(rows, dtype=np.float64)


def looks_like_frame_index(est, tolerance=1e-6):
    t = est[:, 0]
    diffs = np.diff(t)
    return np.allclose(diffs, 1.0, atol=tolerance) and abs(t[0]) < tolerance

def looks_like_raw_nanoseconds(est, gt, ratio_tolerance=0.1):
    ratio = np.median(est[:, 0]) / np.median(gt[:, 0])
    return abs(ratio - 1e9) / 1e9 < ratio_tolerance

def remap_pseudo_timestamps(est, frames, stride):
    frame_times = np.array([f[0] for f in frames])
    remapped = est.copy()
    for row_idx in range(len(est)):
        src_frame_idx = int(round(est[row_idx, 0])) * stride
        src_frame_idx = min(src_frame_idx, len(frame_times) - 1)
        remapped[row_idx, 0] = frame_times[src_frame_idx]
    return remapped

def load_vslam_stride(vslam_dir, default=1):
    cfg_path = os.path.join(vslam_dir, "run_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            return int(cfg.get("params", {}).get("stride", default))
        except Exception:
            pass
    return default

def build_evo_trajectory(tum_array):
    t = tum_array[:, 0]
    xyz = tum_array[:, 1:4]
    qx, qy, qz, qw = tum_array[:, 4], tum_array[:, 5], tum_array[:, 6], tum_array[:, 7]
    quat_wxyz = np.stack([qw, qx, qy, qz], axis=1)
    return PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=quat_wxyz, timestamps=t)

def evo_traj_to_array(traj):
    qw, qx, qy, qz = (traj.orientations_quat_wxyz[:, 0], traj.orientations_quat_wxyz[:, 1],
                       traj.orientations_quat_wxyz[:, 2], traj.orientations_quat_wxyz[:, 3])
    return np.column_stack([traj.timestamps, traj.positions_xyz, qx, qy, qz, qw])

def find_depth_file(depth_dir, image_filename):
    stem = os.path.splitext(image_filename)[0]
    for ext in (".npy", ".png", ".tiff", ".tif", ".exr"):
        p = os.path.join(depth_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def load_depth_frame(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    return np.array(Image.open(path)).astype(np.float64)

def backproject_depth(depth, K, T_wc, min_depth=0.1, max_depth=20.0, max_points=None, rng=None):
    h, w = depth.shape
    d = depth.astype(np.float64)
    valid = (d > min_depth) & (d < max_depth) & np.isfinite(d)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    us, vs, d = us[valid], vs[valid], d[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - cx) * d / fx
    y = (vs - cy) * d / fy
    pts_cam = np.stack([x, y, d, np.ones_like(d)], axis=1)
    if max_points is not None and pts_cam.shape[0] > max_points:
        rng = rng or np.random.default_rng()
        idx = rng.choice(pts_cam.shape[0], size=max_points, replace=False)
        pts_cam = pts_cam[idx]
    return (T_wc @ pts_cam.T).T[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--vslam_dir", required=True, help="e.g. results/dpvo (must contain poses.txt)")
    ap.add_argument("--fde_dir", required=True, help="e.g. results/zoedepth (must contain *.npy)")
    ap.add_argument("--monocular", action="store_true", default=True,
                     help="Sim3 (scale-recovering) alignment -- both DPVO and ORB-SLAM3 "
                          "monocular are scale-ambiguous, this is the correct default")
    ap.add_argument("--gt_depth_scale", type=float, default=0.001,
                     help="depth0 raw value -> meters (0.001 for 16-bit PNG, 1px=1mm)")
    ap.add_argument("--pred_depth_scale", type=float, default=1.0,
                     help="predicted .npy raw value -> meters (1.0 -- depth_infer.py already saves meters)")
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=50.0,
                     help="was 20.0 -- too low for aerial scenes where real (non-sentinel) "
                          "depth legitimately reaches tens of meters; the 65535-sentinel "
                          "(sky/no-return) is masked separately regardless of this value")
    ap.add_argument("--n_pointcloud_frames", type=int, default=15,
                     help="frames to backproject for the point clouds -- kept low, this is for "
                          "visual inspection, not the full-density metric computation")
    ap.add_argument("--max_points_per_frame", type=int, default=2000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no_open", action="store_true", help="don't try to auto-open a browser after saving")
    args = ap.parse_args()

    gt_traj_path = os.path.join(args.dataset_root, "state_groundtruth_estimate0", "data.csv")
    cam_csv_path = os.path.join(args.dataset_root, "cam0", "data.csv")
    sensor_yaml_path = os.path.join(args.dataset_root, "cam0", "sensor.yaml")
    gt_depth_dir = os.path.join(args.dataset_root, "depth0", "data")
    poses_path = os.path.join(args.vslam_dir, "poses.txt")

    for p in (gt_traj_path, cam_csv_path, sensor_yaml_path, gt_depth_dir, poses_path, args.fde_dir):
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}")
            sys.exit(1)

    print("== Loading + aligning trajectories ==")
    gt = load_gt_trajectory(gt_traj_path)
    est = load_tum(poses_path)
    frames = load_image_timestamps(cam_csv_path)

    if looks_like_frame_index(est):
        stride = load_vslam_stride(args.vslam_dir, default=1)
        print(f"  NOTE: {poses_path} uses sequential frame indices (0,1,2,...) instead of "
              f"real timestamps -- this is DPVO's default output format for a plain image "
              f"directory. Remapping to real timestamps via cam0/data.csv (stride={stride}, "
              f"read from run_config.json).")
        est = remap_pseudo_timestamps(est, frames, stride)
    elif looks_like_raw_nanoseconds(est, gt):
        print(f"  NOTE: {poses_path} has raw-nanosecond timestamps (echoed back from the "
              f"input times.txt) instead of seconds like GT -- this is ORB-SLAM3's EuRoC-mode "
              f"output behavior. Converting to seconds (dividing by 1e9).")
        est = est.copy()
        est[:, 0] = est[:, 0] / 1e9

    traj_ref = build_evo_trajectory(gt)
    traj_est = build_evo_trajectory(est)
    traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(traj_ref, traj_est, max_diff=0.02)

    r_align, t_align, s_align = traj_est_sync.align(traj_ref_sync, correct_scale=args.monocular)

    ape = evo_metrics.APE(PoseRelation.translation_part)
    ape.process_data((traj_ref_sync, traj_est_sync))
    ate_rmse = ape.get_statistic(StatisticsType.rmse)

    rpe_trans = evo_metrics.RPE(PoseRelation.translation_part, delta=1, delta_unit=Unit.frames)
    rpe_trans.process_data((traj_ref_sync, traj_est_sync))
    rte_rmse = rpe_trans.get_statistic(StatisticsType.rmse)

    rpe_rot = evo_metrics.RPE(PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames)
    rpe_rot.process_data((traj_ref_sync, traj_est_sync))
    rre_rmse = rpe_rot.get_statistic(StatisticsType.rmse)

    print(f"  ATE RMSE: {ate_rmse:.4f} m")
    print(f"  RTE RMSE: {rte_rmse:.4f} m  (frame-to-frame translation)")
    print(f"  RRE RMSE: {rre_rmse:.4f} deg  (frame-to-frame rotation)")
    print(f"  recovered scale: {s_align:.4f}")
    if ate_rmse > 5.0:
        print("  WARNING: ATE RMSE is large -- either real drift (expected, this is your "
              "thesis's whole point) or a units/frame mismatch. Sanity-check before reading "
              "too much into the picture below.")

    K = load_camera_intrinsics(sensor_yaml_path)["K"]
    cam_w = load_camera_intrinsics(sensor_yaml_path)["width"]
    cam_h = load_camera_intrinsics(sensor_yaml_path)["height"]
    gt_m = evo_traj_to_array(traj_ref_sync)
    est_m = evo_traj_to_array(traj_est_sync)  
    gt_t, est_t = gt_m[:, 0], est_m[:, 0]

    idx_step = max(1, len(frames) // args.n_pointcloud_frames)
    chosen = frames[::idx_step][:args.n_pointcloud_frames]
    rng = np.random.default_rng(0)

    print(f"\n== Backprojecting {len(chosen)} sample frames (GT + estimated) ==")
    gt_pts_all, est_pts_all = [], []
    depth_errors = []

    for frame_t, fname in tqdm(chosen, desc="backprojecting"):
        j_gt = int(np.argmin(np.abs(gt_t - frame_t)))
        j_est = int(np.argmin(np.abs(est_t - frame_t)))
        if abs(gt_t[j_gt] - frame_t) > 0.02 or abs(est_t[j_est] - frame_t) > 0.02:
            continue

        gt_depth_path = find_depth_file(gt_depth_dir, fname)
        pred_depth_path = find_depth_file(args.fde_dir, fname)
        if gt_depth_path is None or pred_depth_path is None:
            continue

        gt_depth_raw = load_depth_frame(gt_depth_path)
        gt_depth_raw = np.where(gt_depth_raw >= 65535, np.nan, gt_depth_raw)
        gt_depth = gt_depth_raw * args.gt_depth_scale
        pred_depth = load_depth_frame(pred_depth_path) * args.pred_depth_scale
        if pred_depth.shape != (cam_h, cam_w):
            pred_depth = np.array(Image.fromarray(pred_depth.astype(np.float32))
                                   .resize((cam_w, cam_h), Image.BILINEAR))

        T_wc_gt = pose_row_to_T(gt_m[j_gt])
        gt_pts = backproject_depth(gt_depth, K, T_wc_gt, min_depth=args.min_depth,
                                    max_depth=args.max_depth,
                                    max_points=args.max_points_per_frame, rng=rng)
        gt_pts_all.append(gt_pts)

        T_wc_est = pose_row_to_T(est_m[j_est])
        est_pts = backproject_depth(pred_depth, K, T_wc_est, min_depth=args.min_depth,
                                     max_depth=args.max_depth,
                                     max_points=args.max_points_per_frame, rng=rng)
        est_pts_all.append(est_pts)

        valid = (gt_depth > args.min_depth) & (gt_depth < args.max_depth) & np.isfinite(gt_depth)
        if valid.sum() > 0:
            depth_errors.append(float(np.mean(np.abs(pred_depth[valid] - gt_depth[valid]) / gt_depth[valid])))

    gt_points = np.concatenate(gt_pts_all, axis=0) if gt_pts_all else np.zeros((0, 3))
    est_points = np.concatenate(est_pts_all, axis=0) if est_pts_all else np.zeros((0, 3))
    print(f"  GT points: {len(gt_points)}, estimated points: {len(est_points)}")
    if depth_errors:
        print(f"  Depth AbsRel (sampled frames): {np.mean(depth_errors):.4f}")

    print("\n== Rendering ==")
    import plotly.graph_objects as go
    fig = go.Figure()

    gt_xyz = gt_m[:, 1:4]
    est_xyz_aligned = est_m[:, 1:4] 
    fig.add_trace(go.Scatter3d(x=gt_xyz[:, 0], y=gt_xyz[:, 1], z=gt_xyz[:, 2],
                                mode="lines", line=dict(color="black", width=5), name="GT trajectory"))
    fig.add_trace(go.Scatter3d(x=est_xyz_aligned[:, 0], y=est_xyz_aligned[:, 1], z=est_xyz_aligned[:, 2],
                                mode="lines", line=dict(color="red", width=4), name="Estimated trajectory (aligned)"))
    if len(gt_points) > 0:
        fig.add_trace(go.Scatter3d(x=gt_points[:, 0], y=gt_points[:, 1], z=gt_points[:, 2],
                                    mode="markers", marker=dict(size=1.5, color="gray", opacity=0.5),
                                    name="GT reconstruction"))
    if len(est_points) > 0:
        fig.add_trace(go.Scatter3d(x=est_points[:, 0], y=est_points[:, 1], z=est_points[:, 2],
                                    mode="markers", marker=dict(size=1.5, color="orange", opacity=0.5),
                                    name="Estimated reconstruction (aligned)"))

    combined = np.concatenate([gt_xyz, est_xyz_aligned, gt_points, est_points], axis=0)
    lo, hi = np.percentile(combined, 1, axis=0), np.percentile(combined, 99, axis=0)
    span = np.maximum(hi - lo, 0.5)
    lo, hi = lo - 0.15 * span, hi + 0.15 * span

    vslam_name = os.path.basename(os.path.normpath(args.vslam_dir))
    fde_name = os.path.basename(os.path.normpath(args.fde_dir))
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis=dict(title="X (m)", range=[lo[0], hi[0]]),
                   yaxis=dict(title="Y (m)", range=[lo[1], hi[1]]),
                   zaxis=dict(title="Z (m)", range=[lo[2], hi[2]])),
        title=f"{vslam_name} + {fde_name} -- ATE {ate_rmse:.3f} m, RTE {rte_rmse:.3f} m, RRE {rre_rmse:.2f} deg",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(args.output)
    print(f"Saved: {args.output}")

    if not args.no_open:
        import webbrowser
        try:
            webbrowser.open(f"file://{os.path.abspath(args.output)}")
        except Exception:
            pass
        print("If a browser didn't open automatically: explorer.exe " + args.output)

    summary = {
        "vslam": vslam_name, "fde": fde_name,
        "ate_rmse_m": ate_rmse, "rte_rmse_m": rte_rmse, "rre_rmse_deg": rre_rmse,
        "recovered_scale": s_align,
        "depth_abs_rel_sampled": float(np.mean(depth_errors)) if depth_errors else None,
        "n_frames_sampled": len(chosen),
    }
    summary_path = args.output.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()