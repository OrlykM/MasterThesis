r"""
In-domain depth correction, learned from data instead of assumed.

The error budget (error_budget.py) puts the blame squarely on depth: with
ground-truth poses and ground-truth depth the voxelization reaches IoU 0.907,
with ground-truth poses and FDE depth it drops to 0.106. Pose drift costs 35%,
depth costs 88%. Nothing on the fusion side moves that -- five separate
attempts were measured and rejected (see the flags in occupancy_3d_map.py).
The depth itself has to get better.

These FDE models are out of domain here: trained on indoor rooms / driving,
applied to an aerial scene with a third of every frame being sky. Two failure
modes were MEASURED, and this module learns a correction for exactly those:

  1. Depth-dependent bias -- error grows with range (2.5m at z=1.8m up to 25m
     at z=48m for ZoeDepth).
  2. Sky hallucination -- 24-48% of every frame has no return in ground truth,
     and ZoeDepth reports a within-range surface for 100% of those pixels.

Model: a lookup table over (predicted-depth bin) x (image-row bin), holding
  - the robust median ratio true/predicted, i.e. a multiplicative correction
  - how often ground truth is even VALID there, i.e. how often a surface
    exists at all -- which is what lets sky be rejected from evidence rather
    than guessed at (an earlier attempt to infer sky from sparse-landmark
    sparsity was measured to hurt: IoU 0.0233 -> 0.0088)

Both axes are normalized (depth as a fraction of the model's own observed
range, row as a fraction of image height), so a table trained at one
resolution or scene scale still applies at another.

Protocol: train on some sequences, apply to a HELD-OUT one. Ground truth is
used for training only -- exactly as a depth network's own training would,
and never at inference.

Usage:
    python3 depth_refine.py train --dataset_roots ../datasets/run1 ../datasets/run2 \
        --fde_subdir zoedepth --results_root ../results --output refiner_zoedepth.npz
    python3 depth_refine.py report --model refiner_zoedepth.npz
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_camera_intrinsics, load_image_timestamps
from compare_reconstruction_3d import find_depth_file, load_depth_frame
import blade_mask as bm

N_DEPTH_BINS = 12
N_ROW_BINS = 8
MIN_CELL_SAMPLES = 50


def _depth_edges(max_depth, min_depth):
    """Log-spaced, because the error grows multiplicatively with range."""
    return np.geomspace(max(min_depth, 1e-3), max_depth, N_DEPTH_BINS + 1)


def _cell_indices(pred, rows_frac, edges):
    d_bin = np.clip(np.digitize(pred, edges) - 1, 0, N_DEPTH_BINS - 1)
    r_bin = np.clip((rows_frac * N_ROW_BINS).astype(int), 0, N_ROW_BINS - 1)
    return d_bin, r_bin


def train(args):
    edges = _depth_edges(args.max_depth, args.min_depth)
    blades = bm.load_mask(args.blade_mask) if args.blade_mask and os.path.exists(args.blade_mask) else None
    if blades is not None:
        print(f"  blade mask: excluding {int(blades.sum())} px per frame")
    ratios = [[[] for _ in range(N_ROW_BINS)] for _ in range(N_DEPTH_BINS)]
    n_valid = np.zeros((N_DEPTH_BINS, N_ROW_BINS))
    n_total = np.zeros((N_DEPTH_BINS, N_ROW_BINS))

    for root in args.dataset_roots:
        # A dataset may be nested one level down (datasets/euroc/<run_name>/)
        # while its results sit under the PARENT's name (results/euroc/), so
        # try the folder itself and then its parent before giving up --
        # otherwise that dataset is silently dropped from training.
        run = os.path.basename(os.path.normpath(root))
        fde_dir = os.path.join(args.results_root, run, args.fde_subdir)
        if not os.path.isdir(fde_dir):
            parent = os.path.basename(os.path.dirname(os.path.normpath(root)))
            alt = os.path.join(args.results_root, parent, args.fde_subdir)
            if os.path.isdir(alt):
                run, fde_dir = parent, alt
            else:
                print(f"  {run}: no {args.fde_subdir} results -- skipped")
                continue
        print(f"  {run}: using {fde_dir}")
        frames = load_image_timestamps(os.path.join(root, "cam0", "data.csv"))
        intr = load_camera_intrinsics(os.path.join(root, "cam0", "sensor.yaml"))
        cam_w, cam_h = intr["width"], intr["height"]
        step = max(1, len(frames) // args.frames_per_run)
        rows_frac_full = (np.arange(cam_h)[:, None] / float(cam_h)) * np.ones((1, cam_w))

        for _t, fname in tqdm(frames[::step][:args.frames_per_run], desc=run, leave=False):
            gp = find_depth_file(os.path.join(root, "depth0", "data"), fname)
            pp = find_depth_file(fde_dir, fname)
            if gp is None or pp is None:
                continue
            raw = load_depth_frame(gp)
            gt = np.where(raw >= 65535, np.nan, raw) * args.gt_depth_scale
            pred = load_depth_frame(pp)
            if pred.shape != (cam_h, cam_w):
                pred = np.array(Image.fromarray(pred.astype(np.float32)).resize((cam_w, cam_h), Image.BILINEAR))

            usable_pred = (pred > args.min_depth) & (pred < args.max_depth) & np.isfinite(pred)
            if blades is not None:
                keep = ~bm.load_mask(args.blade_mask, shape=(cam_h, cam_w)) \
                    if blades.shape != (cam_h, cam_w) else ~blades
                usable_pred &= keep
            d_bin, r_bin = _cell_indices(pred, rows_frac_full, edges)
            gt_ok = np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)

            # Validity: of the pixels where the network claims a surface in
            # range, how often is there actually one? This is what learns sky.
            np.add.at(n_total, (d_bin[usable_pred], r_bin[usable_pred]), 1)
            both = usable_pred & gt_ok
            np.add.at(n_valid, (d_bin[both], r_bin[both]), 1)

            # Ratio, subsampled so one frame can't dominate a cell
            db, rb, rt = d_bin[both], r_bin[both], gt[both] / pred[both]
            if rt.size > args.samples_per_frame:
                sel = np.random.default_rng(0).choice(rt.size, args.samples_per_frame, replace=False)
                db, rb, rt = db[sel], rb[sel], rt[sel]
            for i in range(rt.size):
                ratios[db[i]][rb[i]].append(rt[i])

    ratio_med = np.ones((N_DEPTH_BINS, N_ROW_BINS))
    ratio_mad = np.full((N_DEPTH_BINS, N_ROW_BINS), np.nan)
    counts = np.zeros((N_DEPTH_BINS, N_ROW_BINS), dtype=int)
    for i in range(N_DEPTH_BINS):
        for j in range(N_ROW_BINS):
            vals = np.array(ratios[i][j])
            counts[i, j] = vals.size
            if vals.size >= MIN_CELL_SAMPLES:
                ratio_med[i, j] = float(np.median(vals))
                ratio_mad[i, j] = float(np.median(np.abs(vals - ratio_med[i, j])) * 1.4826)

    with np.errstate(invalid="ignore", divide="ignore"):
        validity = np.where(n_total > 0, n_valid / np.maximum(n_total, 1), np.nan)

    np.savez(args.output, edges=edges, ratio=ratio_med, ratio_mad=ratio_mad,
              validity=validity, counts=counts, n_total=n_total,
              trained_on=np.array([os.path.basename(os.path.normpath(r)) for r in args.dataset_roots]),
              fde=args.fde_subdir, min_depth=args.min_depth, max_depth=args.max_depth)
    print(f"\nSaved: {args.output}")
    report_model(args.output)


def load_refiner(path):
    d = np.load(path, allow_pickle=True)
    return {"edges": d["edges"], "ratio": d["ratio"], "validity": d["validity"],
            "counts": d["counts"], "min_depth": float(d["min_depth"]),
            "max_depth": float(d["max_depth"])}


def apply_refiner(model, pred_depth, min_validity=0.5):
    """-> (corrected_depth, keep_mask). Pixels in cells where ground truth
    rarely had a surface at all (sky) come back with keep_mask False, so the
    caller can drop them instead of fusing an invented surface."""
    h, w = pred_depth.shape
    rows_frac = (np.arange(h)[:, None] / float(h)) * np.ones((1, w))
    d_bin, r_bin = _cell_indices(pred_depth, rows_frac, model["edges"])
    ratio = model["ratio"][d_bin, r_bin]
    validity = model["validity"][d_bin, r_bin]
    # Clamp: a cell can clear MIN_CELL_SAMPLES and still be noisy (a couple of
    # cells came out at 4.9 and 0.04). A correction beyond ~4x either way is
    # not a domain bias, it's a badly-sampled bin.
    ratio = np.clip(np.where(np.isfinite(ratio), ratio, 1.0), 0.25, 4.0)
    corrected = pred_depth * ratio
    # A cell with NO training data means "no evidence", not "no surface".
    # Treating NaN as a rejection threw away real image area: masking the
    # propellers out of TRAINING left the lower-corner cells empty, and those
    # same cells were then rejected at inference even on the Isaac Sim dataset,
    # which has no propellers at all.
    keep = ~np.isfinite(validity) | (validity >= min_validity)
    return corrected, keep


def report_model(path):
    m = load_refiner(path)
    edges, ratio, validity, counts = m["edges"], m["ratio"], m["validity"], m["counts"]
    print(f"\nlearned correction  (rows = predicted-depth bin, cols = image row, top -> bottom)")
    print("depth bin        " + "".join(f"{j:>7d}" for j in range(N_ROW_BINS)))
    for i in range(N_DEPTH_BINS):
        cells = "".join(f"{ratio[i, j]:7.2f}" if counts[i, j] >= MIN_CELL_SAMPLES else "      ."
                        for j in range(N_ROW_BINS))
        print(f"{edges[i]:6.1f}-{edges[i+1]:6.1f} {cells}")
    print(f"\nP(a surface actually exists) -- low = the network invents one (sky)")
    print("depth bin        " + "".join(f"{j:>7d}" for j in range(N_ROW_BINS)))
    for i in range(N_DEPTH_BINS):
        cells = "".join(f"{validity[i, j]:7.2f}" if np.isfinite(validity[i, j]) else "      ."
                        for j in range(N_ROW_BINS))
        print(f"{edges[i]:6.1f}-{edges[i+1]:6.1f} {cells}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--dataset_roots", nargs="+", required=True)
    t.add_argument("--results_root", default="../results")
    t.add_argument("--fde_subdir", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--frames_per_run", type=int, default=60)
    t.add_argument("--samples_per_frame", type=int, default=4000)
    t.add_argument("--blade_mask", default="../masks/blades_airsim.npz",
                    help="exclude the drone's own propellers (see blade_mask.py). They sit at "
                         "0.1-0.3 m in the lower corners of every AirSim frame and would otherwise "
                         "teach the table that very close surfaces are normal there")
    t.add_argument("--gt_depth_scale", type=float, default=0.001)
    t.add_argument("--min_depth", type=float, default=0.1)
    t.add_argument("--max_depth", type=float, default=60.0)

    r = sub.add_parser("report")
    r.add_argument("--model", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        report_model(args.model)


if __name__ == "__main__":
    main()
