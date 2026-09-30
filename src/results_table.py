r"""
The 2x2 results table: {baseline, adaptive} x {without, with the learned depth
correction}, measured at FIXED voxel sizes so the numbers are comparable
across combos, with the pipeline's self-chosen voxel size reported alongside
as its own column rather than mixed into the comparison.

Every cell is IoU / FPVR of the THRESHOLDED probabilistic map against a
ground-truth occupancy grid built on that same grid geometry -- and the GT
reference is built from GT poses converted body->camera (these datasets store
GT as body poses; skipping that conversion rotates the reference 120 deg and
drives IoU to ~0, see euroc_io.estimate_axis_convention_fix).

The correction table must come from depth_refine.py trained with THIS dataset
held out (refiners/loo_<dataset>_<fde>.npz), so "with correction" is never
scored on data it was fitted to.

Usage:
    python3 results_table.py --datasets euroc run1 --vslam dpvo \
        --fdes zoedepth depthanythingv2 fastdepth \
        --refiner_dir ../refiners --output ../results_2x2.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import occupancy_3d_map as om
from measure_metrics import build_gt_grid, axis_fix_for, iou_fpr

DATASET_ROOTS = {"euroc": "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor"}


def dataset_root(name):
    return DATASET_ROOTS.get(name, f"../datasets/{name}")


def run_once(dataset, vslam, fde, resolution, refiner, out_dir, n_frames, max_time_diff,
              net=None):
    """Builds naive+adaptive for one configuration and returns the two grids."""
    argv = ["--mode", "dataset",
            "--dataset_root", dataset_root(dataset),
            "--vslam_dir", f"../results/{dataset}/{vslam}",
            "--fde_dir", f"../results/{dataset}/{fde}",
            "--output_dir", out_dir,
            "--voxel_resolution", str(resolution),
            "--n_frames", str(n_frames),
            "--max_time_diff", str(max_time_diff)]
    if refiner:
        argv += ["--depth_refiner", refiner]
    if net:
        argv += ["--depth_net", net]
    args = om.build_arg_parser().parse_args(argv)
    om.run_dataset_mode(args)
    grids = {}
    for key in ("naive", "adaptive"):
        path = os.path.join(out_dir, f"{key}_grid.npz")
        grids[key] = np.load(path) if os.path.exists(path) else None
    return grids, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--vslam", default="dpvo")
    ap.add_argument("--fdes", nargs="+", default=["zoedepth", "depthanythingv2", "fastdepth"])
    ap.add_argument("--resolutions", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--refiner_dir", default="../refiners")
    ap.add_argument("--net_dir", default="../nets")
    ap.add_argument("--work_dir", default="/tmp/results_table")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--max_time_diff", type=float, default=0.15)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--gt_frames", type=int, default=60)
    args = ap.parse_args()
    # checkpoints are named <short>_holdout_<dataset>.pt
    args.net_prefix = {"zoedepth": "zoe", "depthanythingv2": "dav2", "fastdepth": "fast"}

    rows = []
    for dataset in args.datasets:
        root = dataset_root(dataset)
        axis_fix = axis_fix_for(root, f"../results/{dataset}/{args.vslam}", args.max_time_diff)
        for fde in args.fdes:
            refiner = os.path.join(args.refiner_dir, f"loo_{dataset}_{fde}.npz")
            if not os.path.exists(refiner):
                print(f"!! no held-out refiner for {dataset}/{fde} -- skipping that column")
                refiner = None
            net = os.path.join(args.net_dir, f"{args.net_prefix.get(fde, fde)}_holdout_{dataset}.pt")
            if not os.path.exists(net):
                net = None

            # The pipeline's own choice, reported but not used for comparison
            auto_dir = os.path.join(args.work_dir, f"{dataset}_{fde}_auto")
            _, auto_args = run_once(dataset, args.vslam, fde, "auto", refiner, auto_dir,
                                     args.n_frames, args.max_time_diff)
            auto_res = float(auto_args.voxel_resolution)
            auto_max_depth = float(auto_args.max_depth)

            for resolution in args.resolutions:
                row = {"dataset": dataset, "vslam": args.vslam, "fde": fde,
                       "voxel_m": resolution, "auto_voxel_m": round(auto_res, 2),
                       "auto_max_depth_m": round(auto_max_depth, 1)}
                # Three depth sources, each giving both a baseline and an
                # adaptive map -> the five-variant table of stage 3.3:
                #   raw      baseline / adaptive on the FDE depth as it comes
                #   refined  the same two with the depth_refine table
                #   net      the same two with the network
                for label, (ref, nt) in (("raw", (None, None)),
                                          ("refined", (refiner, None)),
                                          ("net", (None, net))):
                    if label == "refined" and ref is None or label == "net" and nt is None:
                        for key in ("naive", "adaptive"):
                            for m in ("iou", "fpvr", "prec", "rec"):
                                row[f"{m}_{key}_{label}"] = None
                        continue
                    out_dir = os.path.join(args.work_dir, f"{dataset}_{fde}_{resolution}_{label}")
                    grids, _ = run_once(dataset, args.vslam, fde, resolution, ref, out_dir,
                                         args.n_frames, args.max_time_diff, net=nt)
                    ref_grid = grids["naive"] if grids["naive"] is not None else grids["adaptive"]
                    if ref_grid is None:
                        continue
                    gt_grid = build_gt_grid(root, ref_grid["origin"], float(ref_grid["resolution"]),
                                             ref_grid["log_odds"].shape, n_frames=args.gt_frames,
                                             max_depth=auto_max_depth, axis_fix=axis_fix)
                    row["gt_occupied"] = int(gt_grid.sum())
                    for key in ("naive", "adaptive"):
                        g = grids[key]
                        if g is None:
                            row[f"iou_{key}_{label}"] = None
                            row[f"fpvr_{key}_{label}"] = None
                            continue
                        m = iou_fpr(gt_grid, g["log_odds"] > args.threshold)
                        row[f"iou_{key}_{label}"] = round(m["iou"], 4)
                        row[f"fpvr_{key}_{label}"] = round(m["fpr"], 4)
                        row[f"prec_{key}_{label}"] = round(m["precision"], 4)
                        row[f"rec_{key}_{label}"] = round(m["recall"], 4)
                rows.append(row)
                print(f"\n== {dataset} / {args.vslam}+{fde} @ {resolution}m "
                      f"(auto would be {auto_res:.2f}m) ==")
                def f(v):
                    return f"{v:.4f}" if isinstance(v, float) else "    -   "
                print(f"{'variant':20s} {'IoU':>8s} {'FPVR':>8s} {'prec':>8s} {'rec':>8s}")
                for key in ("naive", "adaptive"):
                    for label in ("raw", "refined", "net"):
                        name = {"naive": "baseline", "adaptive": "adaptive"}[key]
                        extra = {"raw": "", "refined": " + table", "net": " + net"}[label]
                        print(f"{name + extra:20s} {f(row.get(f'iou_{key}_{label}')):>8s} "
                              f"{f(row.get(f'fpvr_{key}_{label}')):>8s} "
                              f"{f(row.get(f'prec_{key}_{label}')):>8s} "
                              f"{f(row.get(f'rec_{key}_{label}')):>8s}")

    headers = (["dataset", "vslam", "fde", "voxel_m", "auto_voxel_m", "auto_max_depth_m",
                "gt_occupied"]
               + [f"{m}_{k}_{l}" for m in ("iou", "fpvr", "prec", "rec")
                  for k in ("naive", "adaptive") for l in ("raw", "refined", "net")])
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
