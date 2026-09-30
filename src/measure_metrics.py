r"""
Computes Occupancy IoU / False Positive Voxel Rate DIRECTLY from the actual
naive_grid.npz / weighted_grid.npz files that occupancy_3d_map.py produces --
not a separate point-cloud recomputation like evaluation.py does. This is
the real "naive vs weighted" comparison your thesis needs: same voxel grid
(same origin, same resolution, same dims) for GT and both predictions, so
IoU/FPR are computed voxel-for-voxel against the exact map you built.

Auto-discovers every maps/<dataset>/<vslam>_<fde>/{naive,weighted}_grid.npz
and writes one consolidated CSV row per combo, matching the columns of the
metrics_naive_vs_weighted.csv template from earlier:
    dataset, vslam, fde, iou_naive, iou_weighted, delta_iou,
    fpr_naive, fpr_weighted, delta_fpr

Usage:
    python3 measure_iou_fpvr_voxel.py --datasets_root ~/MasterThesis/datasets \
        --maps_root ../maps --output ../iou_fpvr_naive_vs_weighted.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import (load_gt_trajectory, load_camera_intrinsics, load_image_timestamps, pose_row_to_T,
                       estimate_axis_convention_fix)
from compare_reconstruction_3d import (find_depth_file, load_depth_frame, backproject_depth,
                                        build_evo_trajectory, evo_traj_to_array, load_tum,
                                        looks_like_frame_index, looks_like_raw_nanoseconds,
                                        remap_pseudo_timestamps, load_vslam_stride)
from evo.core import sync as evo_sync


def axis_fix_for(dataset_root, vslam_dir, max_time_diff=0.15):
    """The GT trajectory in these datasets is a BODY pose, not a camera pose
    (sensor.yaml declares T_BS identity, but DPVO's own landmarks sit 0.35m
    from the true surface once the conversion is applied, versus 8.93m
    without). Backprojecting GT depth through an unconverted GT pose builds a
    reference rotated 120 deg away from the maps being scored against it,
    which alone drove occupancy IoU to ~0. Returns the camera->body rotation
    to apply as R_gt @ C.T, or identity if it can't be established."""
    import gt_frame
    cached = gt_frame.resolve(dataset_root)
    if cached is not None:
        return cached.T
    poses = os.path.join(vslam_dir, "poses.txt")
    if not os.path.exists(poses):
        return np.eye(3)
    gt = load_gt_trajectory(os.path.join(dataset_root, "state_groundtruth_estimate0", "data.csv"))
    frames = load_image_timestamps(os.path.join(dataset_root, "cam0", "data.csv"))
    est = load_tum(poses)
    if looks_like_frame_index(est):
        est = remap_pseudo_timestamps(est, frames, load_vslam_stride(vslam_dir, default=1))
    elif looks_like_raw_nanoseconds(est, gt):
        est = est.copy()
        est[:, 0] = est[:, 0] / 1e9
    try:
        trs, tes = evo_sync.associate_trajectories(build_evo_trajectory(gt), build_evo_trajectory(est),
                                                    max_diff=max_time_diff)
        tes.align(trs, correct_scale=True)
        C, after, before = estimate_axis_convention_fix(evo_traj_to_array(trs), evo_traj_to_array(tes))
    except Exception as exc:
        print(f"  NOTE could not establish the GT frame convention ({exc}) -- assuming camera frame")
        return np.eye(3)
    print(f"  [axis_fix] GT-vs-VSLAM orientation offset {before:.1f} deg -> {after:.1f} deg residual")
    if after > 5.0:
        print(f"  [axis_fix] WARNING residual too large to be a pure convention offset -- this "
              f"VSLAM run's orientations genuinely diverged")
    return C


def plot_naive_vs_weighted(rows, naive_key, weighted_key, metric_title, output_path, higher_is_better):
    """Grouped bars: for every (vslam+fde @ dataset), naive (gray) next to
    weighted (colored) side by side. The weighted bar is highlighted green/red
    for the pair with the best/worst SUMMED weighted value across datasets --
    naive is only the baseline reference, so it stays neutral gray throughout."""
    labels = [f"{r['vslam']}+{r['fde']}\n@{r['dataset']}" for r in rows]
    naive_vals = [r[naive_key] for r in rows]
    weighted_vals = [r[weighted_key] for r in rows]
    pair_keys = [f"{r['vslam']}+{r['fde']}" for r in rows]

    sums = {}
    for pk, v in zip(pair_keys, weighted_vals):
        sums[pk] = sums.get(pk, 0.0) + v
    if higher_is_better:
        best_pair = max(sums, key=sums.get)
        worst_pair = min(sums, key=sums.get)
    else:
        worst_pair = max(sums, key=sums.get)
        best_pair = min(sums, key=sums.get)

    weighted_colors = []
    for pk in pair_keys:
        if pk == worst_pair:
            weighted_colors.append("#D62728")
        elif pk == best_pair:
            weighted_colors.append("#2CA02C")
        else:
            weighted_colors.append("#4C72B0")

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.38
    fig_width = max(10, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars_naive = ax.bar(x - bar_w / 2, naive_vals, bar_w, color="#B0B0B0", label="naive")
    bars_weighted = ax.bar(x + bar_w / 2, weighted_vals, bar_w, color=weighted_colors, label="weighted")

    ax.bar_label(bars_naive, fmt=lambda v: f"{v:.2f}", padding=2, fontsize=7)
    ax.bar_label(bars_weighted, fmt=lambda v: f"{v:.2f}", padding=2, fontsize=7)
    ymax = max(max(naive_vals, default=0), max(weighted_vals, default=0))
    ax.set_ylim(top=ymax * 1.15 if ymax > 0 else 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric_title)
    direction_note = "вище = краще" if higher_is_better else "нижче = краще"
    ax.set_title(f"{metric_title}: naive vs зважений -- усі VSLAM+FDE x датасет ({direction_note})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    best_label = "найкращий зважений (сума)" if higher_is_better else "найкращий зважений (сума)"
    legend_handles = [
        Patch(facecolor="#B0B0B0", label="naive (базова лінія)"),
        Patch(facecolor="#2CA02C", label=f"{best_label}: {best_pair}"),
        Patch(facecolor="#D62728", label=f"найгірший зважений (сума): {worst_pair}"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def build_gt_grid(dataset_root, origin, resolution, dims, n_frames=60, max_points_per_frame=4000,
                   gt_depth_scale=0.001, min_depth=0.1, max_depth=50.0, axis_fix=None,
                   blade_mask_path="../masks/blades_airsim.npz"):
    """Backprojects GT depth through GT poses, voxelized into the EXACT same
    grid (origin/resolution/dims) as the naive/weighted map -- required for
    a voxel-for-voxel comparison to mean anything."""
    gt_traj_path = os.path.join(dataset_root, "state_groundtruth_estimate0", "data.csv")
    cam_csv_path = os.path.join(dataset_root, "cam0", "data.csv")
    sensor_yaml_path = os.path.join(dataset_root, "cam0", "sensor.yaml")
    gt_depth_dir = os.path.join(dataset_root, "depth0", "data")

    gt = load_gt_trajectory(gt_traj_path)
    frames = load_image_timestamps(cam_csv_path)
    K = load_camera_intrinsics(sensor_yaml_path)["K"]

    gt_t = gt[:, 0]
    idx_step = max(1, len(frames) // n_frames)
    chosen = frames[::idx_step][:n_frames]
    rng = np.random.default_rng(0)

    import blade_mask as bm
    blades = None
    if blade_mask_path and os.path.exists(blade_mask_path):
        m = bm.load_mask(blade_mask_path)
        blades = m if m.sum() else None

    grid = np.zeros(dims, dtype=bool)
    for frame_t, fname in tqdm(chosen, desc="GT voxelization", leave=False):
        j_gt = int(np.argmin(np.abs(gt_t - frame_t)))
        gt_depth_path = find_depth_file(gt_depth_dir, fname)
        if gt_depth_path is None:
            continue
        gt_depth_raw = load_depth_frame(gt_depth_path)
        gt_depth_raw = np.where(gt_depth_raw >= 65535, np.nan, gt_depth_raw)
        gt_depth = gt_depth_raw * gt_depth_scale
        if blades is not None:
            gt_depth = np.where(blades, np.nan, gt_depth)

        T_wc_gt = pose_row_to_T(gt[j_gt])
        if axis_fix is not None:
            T_wc_gt[:3, :3] = T_wc_gt[:3, :3] @ axis_fix.T  # GT is a BODY pose
        pts = backproject_depth(gt_depth, K, T_wc_gt, min_depth=min_depth, max_depth=max_depth,
                                 max_points=max_points_per_frame, rng=rng)
        if len(pts) == 0:
            continue
        idx = np.floor((pts - origin) / resolution).astype(int)
        valid = np.all((idx >= 0) & (idx < dims), axis=1)
        idx = idx[valid]
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def iou_fpr(gt_grid, pred_grid):
    tp = int(np.logical_and(gt_grid, pred_grid).sum())
    fp = int(np.logical_and(~gt_grid, pred_grid).sum())
    fn = int(np.logical_and(gt_grid, ~pred_grid).sum())
    union = tp + fp + fn
    pred_occupied = int(pred_grid.sum())
    return {
        "iou": float(tp / union) if union > 0 else 0.0,
        "fpr": float(fp / pred_occupied) if pred_occupied > 0 else 0.0,
        "precision": float(tp / pred_occupied) if pred_occupied > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", required=True, help="parent folder containing your dataset folders (e.g. run1, run2, ...)")
    ap.add_argument("--results_root", default="../results",
                     help="where <dataset>/<vslam>/poses.txt live -- needed to establish the GT "
                          "frame convention (see axis_fix_for)")
    ap.add_argument("--maps_root", required=True, help="e.g. maps/ -- contains maps/<dataset>/<vslam>_<fde>/{naive,weighted}_grid.npz")
    ap.add_argument("--output", default="iou_fpvr_naive_vs_weighted.csv")
    ap.add_argument("--charts_dir", default="iou_fpvr_charts", help="where to save the 2 bar chart PNGs")
    ap.add_argument("--threshold", type=float, default=0.0, help="log-odds threshold for 'occupied' (0.0 = P>0.5)")
    ap.add_argument("--n_frames", type=int, default=60)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=50.0)
    args = ap.parse_args()

    rows = []
    dataset_dirs = sorted(d for d in glob.glob(os.path.join(args.maps_root, "*")) if os.path.isdir(d))
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        dataset_root = os.path.join(args.datasets_root, dataset_name)
        if not os.path.exists(os.path.join(dataset_root, "state_groundtruth_estimate0", "data.csv")):
            # Not directly here -- search at ANY depth below (handles the
            # original Isaac Sim dataset's extra nesting, e.g.
            # datasets/euroc/euroc_runs/<run_name>/ -- two levels deep --
            # instead of a flat datasets/run1/ like the newer AirSim runs).
            candidates = glob.glob(os.path.join(dataset_root, "**", "state_groundtruth_estimate0"), recursive=True)
            if candidates:
                found_root = os.path.dirname(candidates[0])
                print(f"  NOTE: {dataset_root} has no direct state_groundtruth_estimate0/ -- "
                      f"found the real dataset nested at {found_root}, using that instead.")
                dataset_root = found_root
        if not os.path.isdir(dataset_root):
            print(f"SKIP {dataset_name}: no matching folder in --datasets_root")
            continue

        for combo_dir in sorted(d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)):
            combo_name = os.path.basename(combo_dir)
            naive_path = os.path.join(combo_dir, "naive_grid.npz")
            if not os.path.exists(naive_path):
                print(f"SKIP {dataset_name}/{combo_name}: no naive_grid.npz")
                continue

            print(f"== {dataset_name} / {combo_name} ==")
            vslam, fde = (combo_name.split("_", 1) + ["?"])[:2]
            axis_fix = axis_fix_for(dataset_root, os.path.join(args.results_root, dataset_name, vslam))

            naive_data = np.load(naive_path)
            origin, resolution = naive_data["origin"], float(naive_data["resolution"])
            dims = naive_data["log_odds"].shape
            # Every variant is scored against a GT grid built on ITS OWN grid
            # geometry, which matters now that --voxel_resolution auto gives
            # different combos different voxel sizes.
            gt_grid = build_gt_grid(dataset_root, origin, resolution, dims,
                                     n_frames=args.n_frames, gt_depth_scale=args.gt_depth_scale,
                                     min_depth=args.min_depth, max_depth=args.max_depth,
                                     axis_fix=axis_fix)

            row = {"dataset": dataset_name, "vslam": vslam, "fde": fde,
                   "voxel_m": resolution, "gt_occupied": int(gt_grid.sum())}
            for key, fname_npz in (("naive", "naive_grid.npz"), ("weighted", "weighted_grid.npz"),
                                    ("adaptive", "adaptive_grid.npz")):
                path = os.path.join(combo_dir, fname_npz)
                if not os.path.exists(path):
                    row[f"iou_{key}"], row[f"fpr_{key}"] = None, None
                    continue
                data = np.load(path)
                if data["log_odds"].shape != dims or abs(float(data["resolution"]) - resolution) > 1e-9:
                    print(f"  SKIP {fname_npz}: grid geometry differs from naive_grid "
                          f"({data['log_odds'].shape} @ {float(data['resolution']):.2f}m) -- "
                          f"regenerate the whole combo in one run")
                    row[f"iou_{key}"], row[f"fpr_{key}"] = None, None
                    continue
                r = iou_fpr(gt_grid, data["log_odds"] > args.threshold)
                row[f"iou_{key}"], row[f"fpr_{key}"] = r["iou"], r["fpr"]
                row[f"precision_{key}"], row[f"recall_{key}"] = r["precision"], r["recall"]
                print(f"  {key:9s} IoU={r['iou']:.4f}  FPVR={r['fpr']:.4f}  "
                      f"prec={r['precision']:.4f}  rec={r['recall']:.4f}")
            if row["iou_adaptive"] is not None and row["iou_naive"] is not None:
                row["delta_iou"] = row["iou_adaptive"] - row["iou_naive"]
                row["delta_fpr"] = row["fpr_adaptive"] - row["fpr_naive"]
                print(f"  adaptive vs naive: dIoU={row['delta_iou']:+.4f}  dFPVR={row['delta_fpr']:+.4f}")
            else:
                row["delta_iou"], row["delta_fpr"] = None, None
            rows.append(row)

    if not rows:
        print("\nNothing computed -- check --datasets_root / --maps_root paths.")
        return

    import csv
    headers = ["dataset", "vslam", "fde", "voxel_m", "gt_occupied",
               "iou_naive", "iou_weighted", "iou_adaptive", "delta_iou",
               "fpr_naive", "fpr_weighted", "fpr_adaptive", "delta_fpr",
               "precision_naive", "precision_weighted", "precision_adaptive",
               "recall_naive", "recall_weighted", "recall_adaptive"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.output} ({len(rows)} rows)")

    os.makedirs(args.charts_dir, exist_ok=True)
    rows = [r for r in rows if r.get("iou_adaptive") is not None]
    if not rows:
        print("no adaptive grids to chart -- CSV written, charts skipped")
        return
    plot_naive_vs_weighted(rows, "iou_naive", "iou_adaptive", "Occupancy IoU",
                            os.path.join(args.charts_dir, "01_IoU_naive_vs_weighted.png"),
                            higher_is_better=True)
    plot_naive_vs_weighted(rows, "fpr_naive", "fpr_adaptive", "False Positive Voxel Rate (FPVR)",
                            os.path.join(args.charts_dir, "02_FPVR_naive_vs_weighted.png"),
                            higher_is_better=False)


if __name__ == "__main__":
    main()