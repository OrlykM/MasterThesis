r"""
Which rotation turns the ground-truth BODY pose into a CAMERA pose, decided
from the ground truth ALONE.

`state_groundtruth_estimate0` stores body poses while `cam0/sensor.yaml`
declares `T_BS = I`, so backprojecting GT depth through the GT pose unrotated
builds a reference that is wrong by a constant rotation. Symptom recorded in
docs/chat_context.md 7.1: on dataset 1 the drone flies at 0.1-5.8 m while the
mean height of reference points is 16.6 m, and the grey reference cloud visibly
floats above the trajectory.

An earlier estimate compared GT orientations against a VSLAM trajectory, which
only works where that trajectory is sound -- useless on run1..run6, where DPVO
drifts to an ATE of 112-180 m. This module uses the criterion from
check_frames.py instead: the correct rotation is the one under which GT depth
clouds from two NEARBY frames land on top of each other, since both then
describe the same physical surface. That needs no VSLAM at all.

Candidates are the 24 signed axis permutations, so the result is a discrete
convention choice rather than a fitted parameter.

Usage:
    python3 gt_frame.py --dataset_root ../datasets/run1
    python3 gt_frame.py --all                 # every dataset, cached to masks/gt_frame.json
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import (load_gt_trajectory, load_camera_intrinsics, load_image_timestamps,
                       pose_row_to_T, signed_permutation_rotations)

CACHE = "../masks/gt_frame.json"
DATASETS = {"euroc": "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor",
            **{f"run{i}": f"../datasets/run{i}" for i in range(1, 7)}}


def _cloud(dataset_root, gt, fname, K, R_bc, n_points, gt_depth_scale, rng):
    path = os.path.join(dataset_root, "depth0", "data", os.path.splitext(fname)[0] + ".png")
    if not os.path.exists(path):
        return None, None
    raw = np.array(Image.open(path)).astype(np.float64)
    valid = (raw > 100) & (raw < 50000)
    if valid.sum() < 100:
        return None, None
    v, u = np.nonzero(valid)
    z = raw[v, u] * gt_depth_scale
    k = rng.choice(len(z), min(n_points, len(z)), replace=False)
    u, v, z = u[k], v[k], z[k]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts_cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    T = pose_row_to_T(gt)
    # Body pose combined with the candidate camera->body rotation
    return (T[:3, :3] @ R_bc @ pts_cam.T).T + T[:3, 3], T[:3, 3]


def estimate(dataset_root, n_pairs=10, frame_gap=8, n_points=3000, gt_depth_scale=0.001, seed=0):
    """-> (R_bc, table) where table lists every candidate sorted by how well
    adjacent-frame clouds agree under it. R_bc is applied as T_wc = T_wb @ R_bc."""
    gt = load_gt_trajectory(os.path.join(dataset_root, "state_groundtruth_estimate0", "data.csv"))
    frames = load_image_timestamps(os.path.join(dataset_root, "cam0", "data.csv"))
    K = load_camera_intrinsics(os.path.join(dataset_root, "cam0", "sensor.yaml"))["K"]
    gt_t = gt[:, 0]
    rng = np.random.default_rng(seed)

    idx = np.linspace(20, len(frames) - frame_gap - 2, n_pairs).astype(int)
    pairs = []
    for i in idx:
        ta, fa = frames[i]
        tb, fb = frames[i + frame_gap]
        ja = int(np.argmin(np.abs(gt_t - ta)))
        jb = int(np.argmin(np.abs(gt_t - tb)))
        pairs.append((gt[ja], fa, gt[jb], fb))

    results = []
    for R in tqdm(signed_permutation_rotations(), desc=os.path.basename(dataset_root), leave=False):
        dists, heights = [], []
        for ga, fa, gb, fb in pairs:
            A, pa = _cloud(dataset_root, ga, fa, K, R, n_points, gt_depth_scale, rng)
            B, _ = _cloud(dataset_root, gb, fb, K, R, n_points, gt_depth_scale, rng)
            if A is None or B is None:
                continue
            dists.append(float(np.median(cKDTree(A).query(B)[0])))
            heights.append(float(np.median(A[:, 2] - pa[2])))
        if dists:
            results.append({"R": R.astype(int).tolist(),
                            "disagreement_m": float(np.median(dists)),
                            "median_height_above_camera_m": float(np.median(heights))})
    results.sort(key=lambda r: r["disagreement_m"])
    return np.array(results[0]["R"], dtype=float), results


def report(name, results):
    ident = next(r for r in results if r["R"] == np.eye(3).astype(int).tolist())
    best = results[0]
    print(f"\n{name}:")
    print(f"  identity (what the code used to assume): disagreement {ident['disagreement_m']:6.2f} m, "
          f"points {ident['median_height_above_camera_m']:+6.1f} m above the camera")
    print(f"  best candidate:                          disagreement {best['disagreement_m']:6.2f} m, "
          f"points {best['median_height_above_camera_m']:+6.1f} m above the camera")
    print(f"  R_bc = {best['R']}")
    improved = ident["disagreement_m"] / max(best["disagreement_m"], 1e-9)
    print(f"  -> {improved:.1f}x better agreement between adjacent frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--frame_gap", type=int, default=8)
    args = ap.parse_args()

    todo = DATASETS if args.all else {os.path.basename(os.path.normpath(args.dataset_root)): args.dataset_root}
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)
    for name, root in todo.items():
        R, results = estimate(root, n_pairs=args.n_pairs, frame_gap=args.frame_gap)
        report(name, results)
        cache[name] = {"R_bc": R.astype(int).tolist(),
                       "disagreement_m": results[0]["disagreement_m"],
                       "identity_disagreement_m": next(r["disagreement_m"] for r in results
                                                        if r["R"] == np.eye(3).astype(int).tolist())}
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    with open(args.cache, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nSaved: {args.cache}")


def load_cached(name, cache=CACHE):
    """-> R_bc for a dataset, or None if it hasn't been estimated yet."""
    if not os.path.exists(cache):
        return None
    with open(cache) as f:
        entry = json.load(f).get(name)
    return np.array(entry["R_bc"], dtype=float) if entry else None


def resolve(dataset_root, cache=CACHE):
    """-> R_bc for whichever dataset this path belongs to, matching either the
    folder itself or its parent (datasets/euroc/<run_name>/ is registered under
    'euroc'). Returns None when the dataset has not been estimated, so callers
    can fall back rather than silently applying the wrong rotation."""
    name = os.path.basename(os.path.normpath(dataset_root))
    R = load_cached(name, cache)
    if R is None:
        R = load_cached(os.path.basename(os.path.dirname(os.path.normpath(dataset_root))), cache)
    return R


if __name__ == "__main__":
    main()
