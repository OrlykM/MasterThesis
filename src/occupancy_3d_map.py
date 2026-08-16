import argparse
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
)
from evo.core import sync as evo_sync


def eval_error_model(model, x):
    if model is None:
        return np.zeros_like(x) if hasattr(x, "shape") else 0.0
    a, b = model["a"], model["b"]
    if model["shape"] == "quadratic":
        return a + b * x + model["c"] * x ** 2
    return a + b * x


class ProbabilisticVoxelGrid:
    def __init__(self, bounds_min, bounds_max, resolution=0.5, l_occ_base=0.85, l_free_base=-0.4, l_min=-4.0, l_max=4.0):
        self.origin = np.array(bounds_min, dtype=np.float64)
        self.resolution = resolution
        dims = np.ceil((np.array(bounds_max) - self.origin) / resolution).astype(int)
        self.dims = np.maximum(dims, 1)
        self.log_odds = np.zeros(self.dims, dtype=np.float32)
        self.l_occ_base, self.l_free_base = l_occ_base, l_free_base
        self.l_min, self.l_max = l_min, l_max

    def world_to_voxel(self, points):
        return np.floor((points - self.origin) / self.resolution).astype(int)

    def _in_bounds(self, idx):
        return np.all((idx >= 0) & (idx < self.dims), axis=1)

    def insert_weighted(self, points_world, weights, sensor_origin, n_ray_samples=4):
        if points_world.shape[0] == 0:
            return
        occ_idx = self.world_to_voxel(points_world)
        mask = self._in_bounds(occ_idx)
        occ_idx, w = occ_idx[mask], weights[mask]
        if occ_idx.shape[0] > 0:
            flat = np.ravel_multi_index(occ_idx.T, self.dims)
            updated = self.log_odds.reshape(-1)
            np.add.at(updated, flat, self.l_occ_base * w)
            self.log_odds = np.clip(updated, self.l_min, self.l_max).reshape(self.dims)

        pts = points_world[mask]
        w_masked = weights[mask]
        if pts.shape[0] == 0 or n_ray_samples <= 0:
            return
        t = np.linspace(0.05, 0.92, n_ray_samples)
        ray_pts = sensor_origin[None, None, :] + t[:, None, None] * (pts[None, :, :] - sensor_origin[None, None, :])
        ray_w = np.tile(w_masked[None, :], (n_ray_samples, 1)).reshape(-1)
        ray_pts = ray_pts.reshape(-1, 3)
        free_idx = self.world_to_voxel(ray_pts)
        fmask = self._in_bounds(free_idx)
        free_idx, ray_w = free_idx[fmask], ray_w[fmask]
        if free_idx.shape[0] > 0:
            flat = np.ravel_multi_index(free_idx.T, self.dims)
            updated = self.log_odds.reshape(-1)
            np.add.at(updated, flat, self.l_free_base * ray_w)
            self.log_odds = np.clip(updated, self.l_min, self.l_max).reshape(self.dims)

    def apply_decay(self, factor):
        self.log_odds *= (1.0 - factor)

    def occupied_mask(self, threshold=0.0):
        return self.log_odds > threshold

    def probability(self):
        return 1.0 / (1.0 + np.exp(-self.log_odds))


def backproject_with_weights(depth, K, T_wc, sigma_pose_val, depth_model, min_depth, max_depth, max_points=None, rng=None, w_ref=1.0):
    h, w = depth.shape
    d = depth.astype(np.float64)
    valid = (d > min_depth) & (d < max_depth) & np.isfinite(d)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    us, vs, d = us[valid], vs[valid], d[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - cx) * d / fx
    y = (vs - cy) * d / fy
    pts_cam = np.stack([x, y, d, np.ones_like(d)], axis=1)

    sigma_depth_vals = eval_error_model(depth_model, d)
    sigma_total = sigma_depth_vals + sigma_pose_val
    weights = 1.0 / (1.0 + sigma_total / w_ref)

    if max_points is not None and pts_cam.shape[0] > max_points:
        rng = rng or np.random.default_rng()
        idx = rng.choice(pts_cam.shape[0], size=max_points, replace=False)
        pts_cam, weights = pts_cam[idx], weights[idx]

    pts_world = (T_wc @ pts_cam.T).T[:, :3]
    return pts_world, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--vslam_dir", required=True)
    ap.add_argument("--fde_dir", required=True)
    ap.add_argument("--error_model", required=True, help="output of fit_error_models.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--voxel_resolution", type=float, default=0.5)
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--max_points_per_frame", type=int, default=4000)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    ap.add_argument("--pred_depth_scale", type=float, default=1.0)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=50.0)
    ap.add_argument("--w_ref", type=float, default=1.0, help="reference uncertainty (meters) at which weight drops to 0.5 -- tune this against your error model's own scale")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.error_model) as f:
        err = json.load(f)
    pose_model, depth_model = err["pose_model"], err["depth_model"]

    gt_traj_path = os.path.join(args.dataset_root, "state_groundtruth_estimate0", "data.csv")
    cam_csv_path = os.path.join(args.dataset_root, "cam0", "data.csv")
    sensor_yaml_path = os.path.join(args.dataset_root, "cam0", "sensor.yaml")
    fde_dir_full = os.path.join(args.fde_dir)
    poses_path = os.path.join(args.vslam_dir, "poses.txt")

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
    traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(traj_ref, traj_est, max_diff=0.02)
    traj_est_sync.align(traj_ref_sync, correct_scale=True)
    gt_m = evo_traj_to_array(traj_ref_sync)
    est_m = evo_traj_to_array(traj_est_sync)
    gt_t, est_t = gt_m[:, 0], est_m[:, 0]

    K = load_camera_intrinsics(sensor_yaml_path)["K"]
    cam_w = load_camera_intrinsics(sensor_yaml_path)["width"]
    cam_h = load_camera_intrinsics(sensor_yaml_path)["height"]

    xyz = gt_m[:, 1:4]
    seg_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    idx_step = max(1, len(frames) // args.n_frames)
    chosen = frames[::idx_step][:args.n_frames]
    rng = np.random.default_rng(0)

    all_pts, all_weights, all_pts_naive = [], [], []
    for frame_t, fname in tqdm(chosen, desc="fusing"):
        j_gt = int(np.argmin(np.abs(gt_t - frame_t)))
        j_est = int(np.argmin(np.abs(est_t - frame_t)))
        if abs(gt_t[j_gt] - frame_t) > 0.02 or abs(est_t[j_est] - frame_t) > 0.02:
            continue
        pred_path = find_depth_file(fde_dir_full, fname)
        if pred_path is None:
            continue
        pred_depth = load_depth_frame(pred_path) * args.pred_depth_scale
        if pred_depth.shape != (cam_h, cam_w):
            pred_depth = np.array(Image.fromarray(pred_depth.astype(np.float32)).resize((cam_w, cam_h), Image.BILINEAR))

        T_wc = pose_row_to_T(est_m[j_est])
        sigma_pose_val = float(eval_error_model(pose_model, np.array([cum_dist[j_gt]]))[0])

        pts, w = backproject_with_weights(pred_depth, K, T_wc, sigma_pose_val, depth_model,
                                           args.min_depth, args.max_depth,
                                           max_points=args.max_points_per_frame, rng=rng, w_ref=args.w_ref)
        all_pts.append(pts)
        all_weights.append(w)
        all_pts_naive.append((pts, T_wc[:3, 3]))

    points = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    weights = np.concatenate(all_weights, axis=0) if all_weights else np.zeros(0)
    print(f"\nTotal points: {len(points)}, weight range: [{weights.min():.3f}, {weights.max():.3f}]"
          if len(weights) else "No points backprojected.")

    if len(points) == 0:
        print("Nothing to save.")
        return

    bounds_min = points.min(axis=0) - 1.0
    bounds_max = points.max(axis=0) + 1.0

    naive_grid = ProbabilisticVoxelGrid(bounds_min, bounds_max, args.voxel_resolution)
    for pts, origin in all_pts_naive:
        naive_grid.insert_weighted(pts, np.ones(len(pts)), origin)

    weighted_grid = ProbabilisticVoxelGrid(bounds_min, bounds_max, args.voxel_resolution)
    offset = 0
    for pts, origin in all_pts_naive:
        n = len(pts)
        weighted_grid.insert_weighted(pts, weights[offset:offset + n], origin)
        offset += n

    np.savez_compressed(os.path.join(args.output_dir, "naive_grid.npz"), log_odds=naive_grid.log_odds, origin=naive_grid.origin, resolution=naive_grid.resolution)
    np.savez_compressed(os.path.join(args.output_dir, "weighted_grid.npz"), log_odds=weighted_grid.log_odds, origin=weighted_grid.origin, resolution=weighted_grid.resolution)

    print(f"Naive occupied voxels:    {int(naive_grid.occupied_mask().sum())}")
    print(f"Weighted occupied voxels: {int(weighted_grid.occupied_mask().sum())}")
    print(f"Saved: {args.output_dir}/naive_grid.npz, weighted_grid.npz")


if __name__ == "__main__":
    main()