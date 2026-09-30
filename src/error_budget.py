r"""
Where does the occupancy error actually come from? Swaps ONE input at a time
for ground truth and re-measures, so the headline IoU/FPVR can be attributed
to a specific stage instead of argued about.

    GT depth  + GT pose   -> ceiling: what this voxelization can do at all
    GT depth  + EST pose  -> what VSLAM drift alone costs
    FDE depth + GT pose   -> what depth error alone costs
    FDE depth + EST pose  -> the real pipeline

Every row uses the same voxel grid, the same frames and the same reference,
so the four numbers are directly comparable. The gap between rows 1 and 4,
split across rows 2 and 3, says where investing actually pays.

Usage:
    python3 error_budget.py --dataset_root ../datasets/euroc/..._quadrotor \
        --vslam_dir ../results/euroc/dpvo --fde_dir ../results/euroc/zoedepth
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import (load_gt_trajectory, load_camera_intrinsics, load_image_timestamps, pose_row_to_T,
                       estimate_axis_convention_fix)
from compare_reconstruction_3d import (
    build_evo_trajectory, evo_traj_to_array, looks_like_frame_index, looks_like_raw_nanoseconds,
    remap_pseudo_timestamps, load_vslam_stride, load_tum, find_depth_file, load_depth_frame,
)
from evo.core import sync as evo_sync
from occupancy_3d_map import ProbabilisticVoxelGrid, backproject_with_weight_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--vslam_dir", required=True)
    ap.add_argument("--fde_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=60)
    ap.add_argument("--voxel_resolution", type=float, default=1.0)
    ap.add_argument("--max_points_per_frame", type=int, default=4000)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=50.0)
    ap.add_argument("--max_time_diff", type=float, default=0.15)
    ap.add_argument("--axis_fix", choices=["auto", "off"], default="auto")
    args = ap.parse_args()

    gt = load_gt_trajectory(os.path.join(args.dataset_root, "state_groundtruth_estimate0", "data.csv"))
    frames = load_image_timestamps(os.path.join(args.dataset_root, "cam0", "data.csv"))
    intr = load_camera_intrinsics(os.path.join(args.dataset_root, "cam0", "sensor.yaml"))
    K, cam_w, cam_h = intr["K"], intr["width"], intr["height"]
    est = load_tum(os.path.join(args.vslam_dir, "poses.txt"))
    if looks_like_frame_index(est):
        est = remap_pseudo_timestamps(est, frames, load_vslam_stride(args.vslam_dir, default=1))
    elif looks_like_raw_nanoseconds(est, gt):
        est = est.copy()
        est[:, 0] = est[:, 0] / 1e9
    trs, tes = evo_sync.associate_trajectories(build_evo_trajectory(gt), build_evo_trajectory(est),
                                                max_diff=args.max_time_diff)
    tes.align(trs, correct_scale=True)
    gt_m, est_m = evo_traj_to_array(trs), evo_traj_to_array(tes)
    gt_t, est_t = gt_m[:, 0], est_m[:, 0]

    R_axis_fix = np.eye(3)
    if args.axis_fix == "auto":
        R_axis_fix, res_after, res_before = estimate_axis_convention_fix(gt_m, est_m)
        print(f"[axis_fix] {res_before:.2f} deg -> {res_after:.2f} deg after axis-permutation snap")

    rng = np.random.default_rng(0)
    step = max(1, len(frames) // args.n_frames)
    combos = {"GT depth  + GT pose ": [], "GT depth  + EST pose": [],
              "FDE depth + GT pose ": [], "FDE depth + EST pose": []}

    for frame_t, fname in tqdm(frames[::step][:args.n_frames], desc="error budget", leave=False):
        jg = int(np.argmin(np.abs(gt_t - frame_t)))
        je = int(np.argmin(np.abs(est_t - frame_t)))
        if abs(gt_t[jg] - frame_t) > args.max_time_diff or abs(est_t[je] - frame_t) > args.max_time_diff:
            continue
        gp = find_depth_file(os.path.join(args.dataset_root, "depth0", "data"), fname)
        pp = find_depth_file(args.fde_dir, fname)
        if gp is None or pp is None:
            continue
        raw = load_depth_frame(gp)
        d_gt = np.where(raw >= 65535, np.nan, raw) * args.gt_depth_scale
        d_fde = load_depth_frame(pp)
        if d_fde.shape != (cam_h, cam_w):
            d_fde = np.array(Image.fromarray(d_fde.astype(np.float32)).resize((cam_w, cam_h), Image.BILINEAR))
        T_gt, T_est = pose_row_to_T(gt_m[jg]), pose_row_to_T(est_m[je])
        T_gt[:3, :3] = T_gt[:3, :3] @ R_axis_fix.T  # GT is a BODY pose, convert to camera

        for label, (depth, T) in {
            "GT depth  + GT pose ": (d_gt, T_gt),
            "GT depth  + EST pose": (d_gt, T_est),
            "FDE depth + GT pose ": (d_fde, T_gt),
            "FDE depth + EST pose": (d_fde, T_est),
        }.items():
            pts, _ = backproject_with_weight_map(depth, np.ones_like(depth), K, T,
                                                  args.min_depth, args.max_depth,
                                                  max_points=args.max_points_per_frame, rng=rng)
            combos[label].append((pts, T[:3, 3]))

    allp = np.concatenate([p for v in combos.values() for p, _ in v], axis=0)
    bmin, bmax = allp.min(axis=0) - 1.0, allp.max(axis=0) + 1.0

    def build(entries, rays):
        g = ProbabilisticVoxelGrid(bmin, bmax, args.voxel_resolution)
        for pts, origin in entries:
            g.insert_weighted(pts, np.ones(len(pts)), origin, n_ray_samples=rays)
        return g.occupied_mask()

    ref = build(combos["GT depth  + GT pose "], 0)
    total = int(np.prod(ref.shape))
    print(f"\nvoxel={args.voxel_resolution}m  grid={ref.shape}  reference occupied={int(ref.sum())} "
          f"({100*ref.sum()/total:.2f}% of volume)")
    print(f"{'configuration':22s} {'IoU':>8s} {'FPVR':>8s} {'precision':>10s} {'recall':>8s} "
          f"{'vs random':>10s}")
    for label, entries in combos.items():
        m = build(entries, 4)
        tp = int(np.logical_and(ref, m).sum())
        fp = int(np.logical_and(~ref, m).sum())
        fn = int(np.logical_and(ref, ~m).sum())
        n_occ = int(m.sum())
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        prec = tp / n_occ if n_occ else 0.0
        chance = ref.sum() / total
        print(f"{label:22s} {iou:8.4f} {fp/n_occ if n_occ else 0:8.4f} {prec:10.4f} "
              f"{tp/(tp+fn) if (tp+fn) else 0:8.4f} {prec/chance if chance else 0:9.1f}x")


if __name__ == "__main__":
    main()
