r"""
Controlled-noise ablation: does the probabilistic layer actually smooth away
VSLAM+FDE error, or not?

On real FDE output that question can't be answered, because two effects are
tangled: (1) how well the fusion layer suppresses error, and (2) how bad the
off-the-shelf depth network happens to be on this domain. Measured on the
EuRoC quadrotor data, (2) dominates so hard (median 12m between a predicted
occupied voxel and the nearest true surface) that (1) is invisible.

So: feed the SAME pipeline ground-truth depth with a KNOWN error injected,
and vary the KIND of error.

    --error_kind random      per-pixel, zero-mean noise, fresh every frame
    --error_kind systematic  one scale+offset error per frame, smoothly
                             drifting -- i.e. what a mis-calibrated monocular
                             depth network actually produces
    --error_kind both

The prediction the layer's design makes: Bayesian log-odds accumulation
averages out RANDOM error (many independent observations of the same voxel
converge on the truth) but CANNOT fix SYSTEMATIC error (every observation
agrees on the same wrong answer, so accumulation just makes the map more
confidently wrong). This script measures that split directly, which is the
honest way to report what the layer contributes independently of which depth
network is bolted onto it.

Usage:
    python3 noise_ablation.py \
        --dataset_root ../datasets/euroc/2026-04-01_12-13-18_000_quadrotor \
        --error_kind random --depth_noise 0.25 --pose_noise 0.3
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_gt_trajectory, load_camera_intrinsics, load_image_timestamps, pose_row_to_T
from compare_reconstruction_3d import find_depth_file, load_depth_frame
from occupancy_3d_map import ProbabilisticVoxelGrid, backproject_with_weight_map
import depth_calibration as depth_calib


def occ_metrics(gt_mask, pred_mask):
    tp = int(np.logical_and(gt_mask, pred_mask).sum())
    fp = int(np.logical_and(~gt_mask, pred_mask).sum())
    fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
    union = tp + fp + fn
    n_occ = int(pred_mask.sum())
    return {
        "iou": tp / union if union else 0.0,
        "fpvr": fp / n_occ if n_occ else 0.0,
        "precision": tp / n_occ if n_occ else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "n_occ": n_occ,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--n_frames", type=int, default=60)
    ap.add_argument("--max_points_per_frame", type=int, default=4000)
    ap.add_argument("--voxel_resolution", type=float, default=1.0)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=50.0)
    ap.add_argument("--error_kind", choices=["random", "systematic", "both", "none"], default="random")
    ap.add_argument("--depth_noise", type=float, default=0.25,
                     help="relative depth error magnitude (0.25 = 25%% of true depth)")
    ap.add_argument("--pose_noise", type=float, default=0.0,
                     help="per-frame position error (meters, random walk for 'systematic')")
    ap.add_argument("--n_anchors", type=int, default=800,
                     help="synthetic sparse VSLAM landmarks per frame (sampled from TRUE depth, "
                          "with their own small noise -- stands in for triangulated map points)")
    ap.add_argument("--anchor_noise", type=float, default=0.05,
                     help="relative noise on the synthetic landmarks themselves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_splat", action="store_true",
                     help="ablation: adaptive path keeps calibration+confidence weighting but "
                          "inserts each point into a single voxel (no uncertainty splatting)")
    ap.add_argument("--no_calib", action="store_true",
                     help="ablation: adaptive path skips the depth recalibration, keeping only "
                          "confidence weighting (+splatting unless --no_splat)")
    ap.add_argument("--report_sigma", action="store_true",
                     help="print the distribution of the per-point sigma driving splat radius")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cam_csv = os.path.join(args.dataset_root, "cam0", "data.csv")
    sensor_yaml = os.path.join(args.dataset_root, "cam0", "sensor.yaml")
    gt_depth_dir = os.path.join(args.dataset_root, "depth0", "data")
    gt = load_gt_trajectory(os.path.join(args.dataset_root, "state_groundtruth_estimate0", "data.csv"))
    frames = load_image_timestamps(cam_csv)
    intr = load_camera_intrinsics(sensor_yaml)
    K, cam_w, cam_h = intr["K"], intr["width"], intr["height"]
    gt_t = gt[:, 0]

    step = max(1, len(frames) // args.n_frames)
    chosen = frames[::step][:args.n_frames]

    local_calib = depth_calib.LocalCalibrator()
    clean, noisy_naive, noisy_adaptive = [], [], []
    drift = np.zeros(3)
    frame_scale = 1.0

    for frame_t, fname in tqdm(chosen, desc=f"noise={args.error_kind}", leave=False):
        j = int(np.argmin(np.abs(gt_t - frame_t)))
        if abs(gt_t[j] - frame_t) > 0.15:
            continue
        path = find_depth_file(gt_depth_dir, fname)
        if path is None:
            continue
        raw = load_depth_frame(path)
        d_true = np.where(raw >= 65535, np.nan, raw) * args.gt_depth_scale
        T_true = pose_row_to_T(gt[j])

        # ---- inject the error ----
        d_noisy = d_true.copy()
        if args.error_kind in ("random", "both"):
            d_noisy = d_noisy * (1.0 + rng.normal(0.0, args.depth_noise, size=d_noisy.shape))
        if args.error_kind in ("systematic", "both"):
            # one smoothly-drifting scale+offset for the whole frame -- the
            # signature of a mis-calibrated depth network, not of sensor noise
            frame_scale = np.clip(frame_scale + rng.normal(0.0, 0.02), 0.5, 1.8)
            d_noisy = d_noisy * (1.0 + args.depth_noise * (frame_scale - 1.0) / 0.4)

        T_noisy = T_true.copy()
        if args.pose_noise > 0:
            if args.error_kind in ("systematic", "both"):
                drift = drift + rng.normal(0.0, args.pose_noise * 0.1, size=3)  # random walk = drift
                T_noisy[:3, 3] = T_true[:3, 3] + drift
            else:
                T_noisy[:3, 3] = T_true[:3, 3] + rng.normal(0.0, args.pose_noise, size=3)

        # ---- clean reference points (true depth, true pose) ----
        pts_c, _ = backproject_with_weight_map(d_true, np.ones_like(d_true), K, T_true,
                                                args.min_depth, args.max_depth,
                                                max_points=args.max_points_per_frame, rng=rng)
        clean.append((pts_c, T_true[:3, 3]))

        # ---- naive: fuse the noisy data as-is ----
        pts_n, w_n = backproject_with_weight_map(d_noisy, np.ones_like(d_noisy), K, T_noisy,
                                                  args.min_depth, args.max_depth,
                                                  max_points=args.max_points_per_frame, rng=rng)
        noisy_naive.append((pts_n, w_n, T_noisy[:3, 3]))

        # ---- adaptive: synthetic sparse landmarks + LocalCalibrator + splat ----
        valid = np.isfinite(d_true) & (d_true > args.min_depth) & (d_true < args.max_depth)
        vs, us = np.nonzero(valid)
        if len(us) > args.n_anchors:
            sel = rng.choice(len(us), size=args.n_anchors, replace=False)
            us, vs = us[sel], vs[sel]
        if len(us) < 16:
            continue
        az = d_true[vs, us] * (1.0 + rng.normal(0.0, args.anchor_noise, size=len(us)))
        result = local_calib.calibrate_frame(us.astype(float), vs.astype(float), az, d_noisy)
        if result is None:
            continue
        alpha_map, beta_map, conf_map, sigma_map, _support_map = result
        corrected = d_noisy if args.no_calib else np.clip(alpha_map * d_noisy + beta_map, 0.0, None)
        pts_a, w_a, sig_a = backproject_with_weight_map(
            corrected, conf_map, K, T_noisy, args.min_depth, args.max_depth,
            max_points=args.max_points_per_frame, rng=rng, extra_maps=(sigma_map,))
        noisy_adaptive.append((pts_a, w_a, sig_a, T_noisy[:3, 3]))

    all_pts = np.concatenate([p for p, _ in clean] + [p for p, _, _ in noisy_naive]
                              + [p for p, _, _, _ in noisy_adaptive], axis=0)
    bmin, bmax = all_pts.min(axis=0) - 1.0, all_pts.max(axis=0) + 1.0

    gt_grid = ProbabilisticVoxelGrid(bmin, bmax, args.voxel_resolution)
    for pts, origin in clean:
        gt_grid.insert_weighted(pts, np.ones(len(pts)), origin, n_ray_samples=0)
    gt_mask = gt_grid.occupied_mask()

    naive_grid = ProbabilisticVoxelGrid(bmin, bmax, args.voxel_resolution)
    for pts, w, origin in noisy_naive:
        naive_grid.insert_weighted(pts, np.ones(len(pts)), origin, n_ray_samples=4)

    if args.report_sigma:
        all_sig = np.concatenate([s for _, _, s, _ in noisy_adaptive])
        print(f"[sigma] per-point sigma (m): p10={np.percentile(all_sig,10):.2f} "
              f"median={np.median(all_sig):.2f} p90={np.percentile(all_sig,90):.2f} "
              f"-> splat radius in voxels at {args.voxel_resolution}m: "
              f"median={int(np.ceil(1.5*np.median(all_sig)/args.voxel_resolution))}")

    adaptive_grid = ProbabilisticVoxelGrid(
        bmin, bmax, args.voxel_resolution,
        consistency_tau=depth_calib.RunningScale(init=2.0, half_life=30))
    for pts, w, sig, origin in noisy_adaptive:
        adaptive_grid.insert_weighted(pts, w, origin, sigma=None if args.no_splat else sig,
                                       n_ray_samples=4)

    print(f"\n=== error_kind={args.error_kind} depth_noise={args.depth_noise} "
          f"pose_noise={args.pose_noise} voxel={args.voxel_resolution}m "
          f"splat={'off' if args.no_splat else 'on'} calib={'off' if args.no_calib else 'on'} ===")
    print(f"GT (clean) occupied voxels: {int(gt_mask.sum())}")
    for label, grid in (("naive   ", naive_grid), ("adaptive", adaptive_grid)):
        best = None
        for thr in (0.0, 0.43, 0.85, 1.7, 2.55):
            m = occ_metrics(gt_mask, grid.occupied_mask(threshold=thr))
            marker = ""
            if best is None or m["iou"] > best[1]["iou"]:
                best, marker = (thr, m), ""
            print(f"  {label} thr={thr:4.2f}: IoU={m['iou']:.4f}  FPVR={m['fpvr']:.4f}  "
                  f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  occ={m['n_occ']:7d}{marker}")
        print(f"  {label} BEST: thr={best[0]:.2f} IoU={best[1]['iou']:.4f} FPVR={best[1]['fpvr']:.4f}")


if __name__ == "__main__":
    main()
