r"""
Training cache for the depth-refinement network (stage 1).

One .npz per dataset, written frame by frame so nothing large is ever held in
memory at once (WSL has 10 GB). Everything is stored at 320x240 as float16,
which is about 1.1 MB per frame across all channels.

What goes in, and why:

  grey, grad    Greyscale image and its Sobel magnitude. COLOUR IS EXCLUDED on
                purpose: all six AirSim sequences come from the same park, and
                colour is the fastest route to memorising that park rather than
                learning a correction.
  fde           The depth estimator's own output.
  sparse, mask  Sparse reference depths standing in for VSLAM landmarks. They
                are SYNTHESISED from ground truth at high-gradient pixels (the
                places a feature tracker would actually latch onto), with 2-10%
                noise, 50-800 points per frame and random dropout, so the
                network learns to use whatever sparse support it is handed.
                Real VSLAM points are kept for EVALUATION only (stage 3),
                because DPVO drifts badly on the AirSim sequences and training
                against its errors would teach the network to reproduce them.
  target, valid Ground-truth depth and where it exists: 65535 means no return
                (sky), and the propeller mask from blade_mask.py is removed too.

Scale normalisation (1.2): every depth in a frame -- fde, sparse and target --
is divided by the MEDIAN OF THAT FRAME'S SPARSE DEPTHS and then log-taken:

    x = log(depth / median(sparse))

so the network sees ratios, never the absolute metres of one particular park.
The divisor is stored per frame so predictions can be mapped back.

Usage:
    python3 depth_net_data.py --dataset euroc --fde zoedepth --output ../cache/euroc_zoedepth.npz
    python3 depth_net_data.py --all --fde zoedepth
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

DATASETS = {"euroc": "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor",
            **{f"run{i}": f"../datasets/run{i}" for i in range(1, 7)}}
OUT_H, OUT_W = 240, 320


def _resize(a, order="bilinear"):
    mode = Image.NEAREST if order == "nearest" else Image.BILINEAR
    return np.array(Image.fromarray(a.astype(np.float32)).resize((OUT_W, OUT_H), mode))


def _sobel(grey):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    gx = sum(kx[i, j] * np.roll(np.roll(grey, 1 - i, 0), 1 - j, 1)
             for i in range(3) for j in range(3) if kx[i, j])
    gy = sum(kx[j, i] * np.roll(np.roll(grey, 1 - i, 0), 1 - j, 1)
             for i in range(3) for j in range(3) if kx[j, i])
    return np.sqrt(gx ** 2 + gy ** 2)


def synth_sparse(target, valid, grad, rng, n_min=50, n_max=800,
                  noise_min=0.02, noise_max=0.10, dropout_max=0.3):
    """Sparse reference depths at strong-gradient pixels, the way a feature
    tracker would place them, with per-frame random count, noise and dropout."""
    h, w = target.shape
    cand = valid & (grad > np.percentile(grad[valid], 70)) if valid.any() else valid
    if cand.sum() < n_min:
        cand = valid
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return np.zeros_like(target), np.zeros_like(target, dtype=bool)
    n = int(rng.integers(n_min, n_max + 1))
    n = min(n, idx.size)
    pick = rng.choice(idx, size=n, replace=False)
    keep = rng.random(n) > rng.uniform(0.0, dropout_max)
    pick = pick[keep]
    if pick.size == 0:
        pick = idx[:1]
    noise = rng.uniform(noise_min, noise_max)
    sparse = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    vals = target.ravel()[pick] * (1.0 + rng.normal(0.0, noise, size=pick.size))
    sparse.ravel()[pick] = np.maximum(vals, 1e-3)
    mask.ravel()[pick] = True
    return sparse, mask


def build(dataset, fde, output, target_frames, gt_depth_scale, min_depth, max_depth,
          blade_mask_path, seed):
    root = DATASETS[dataset]
    results_dir = os.path.join("../results", dataset, fde)
    frames = load_image_timestamps(os.path.join(root, "cam0", "data.csv"))
    intr = load_camera_intrinsics(os.path.join(root, "cam0", "sensor.yaml"))
    cam_h, cam_w = intr["height"], intr["width"]
    blades_full = None
    if blade_mask_path and os.path.exists(blade_mask_path):
        m = bm.load_mask(blade_mask_path, shape=(cam_h, cam_w))
        blades_full = m if m.sum() else None

    step = max(1, len(frames) // target_frames)
    chosen = frames[::step][:target_frames]
    rng = np.random.default_rng(seed)

    keys = ["grey", "grad", "fde", "sparse", "smask", "target", "valid"]
    buf = {k: [] for k in keys}
    divisors, names = [], []

    for _t, fname in tqdm(chosen, desc=f"{dataset}/{fde}", leave=False):
        gp = find_depth_file(os.path.join(root, "depth0", "data"), fname)
        pp = find_depth_file(results_dir, fname)
        ip = os.path.join(root, "cam0", "data", fname)
        if gp is None or pp is None or not os.path.exists(ip):
            continue

        raw = load_depth_frame(gp)
        target_full = np.where(raw >= 65535, np.nan, raw) * gt_depth_scale
        valid_full = np.isfinite(target_full) & (target_full > min_depth) & (target_full < max_depth)
        if blades_full is not None:
            valid_full &= ~blades_full
        if valid_full.sum() < 500:
            continue

        img = np.array(Image.open(ip).convert("L")).astype(np.float32) / 255.0
        fde_full = load_depth_frame(pp).astype(np.float32)
        if fde_full.shape != (cam_h, cam_w):
            fde_full = _resize(fde_full)
            fde_full = np.array(Image.fromarray(fde_full).resize((cam_w, cam_h), Image.BILINEAR))

        # Downsample FIRST, then synthesise the sparse points at the output
        # resolution: nearest-downsampling a mask of isolated pixels would throw
        # away about three quarters of them and break the requested 50-800 count.
        grey = _resize(img)
        grad = _resize(_sobel(img))
        target_s = _resize(np.nan_to_num(target_full))
        valid = _resize(valid_full.astype(np.float32), "nearest") > 0.5
        fde_ok_full = np.isfinite(fde_full) & (fde_full > min_depth)
        fde_s = _resize(np.where(fde_ok_full, fde_full, 0.0))
        fde_ok = _resize(fde_ok_full.astype(np.float32), "nearest") > 0.5
        if valid.sum() < 200:
            continue

        sparse, smask = synth_sparse(target_s, valid, grad, rng)
        if smask.sum() < 8:
            continue

        # Scale normalisation: divide by this frame's own sparse median, then log.
        divisor = float(np.median(sparse[smask]))
        if not np.isfinite(divisor) or divisor <= 1e-6:
            continue

        def to_log(a, m):
            out = np.zeros_like(a, dtype=np.float32)
            out[m] = np.log(np.maximum(a[m] / divisor, 1e-6))
            return out

        fde_l = to_log(fde_s, fde_ok)
        sparse_l = to_log(sparse, smask)
        target_l = to_log(target_s, valid)

        for k, v in zip(keys, [grey, grad, fde_l, sparse_l, smask, target_l, valid]):
            buf[k].append(v.astype(np.float16) if v.dtype != bool else v)
        divisors.append(divisor)
        names.append(fname)

    if not divisors:
        raise SystemExit(f"{dataset}/{fde}: nothing usable")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(output,
                        **{k: np.stack(v) for k, v in buf.items()},
                        divisor=np.array(divisors, dtype=np.float32),
                        names=np.array(names),
                        dataset_name=dataset, fde_name=fde, shape=np.array([OUT_H, OUT_W]))
    mb = os.path.getsize(output) / 1e6
    tgt = np.stack(buf["target"]).astype(np.float32)
    val = np.stack(buf["valid"])
    print(f"{dataset}/{fde}: {len(divisors)} frames, {mb:.0f} MB, "
          f"divisor median {np.median(divisors):.2f} m, "
          f"log-target p5/p50/p95 {np.percentile(tgt[val],5):+.2f}/"
          f"{np.percentile(tgt[val],50):+.2f}/{np.percentile(tgt[val],95):+.2f}, "
          f"valid {100*val.mean():.0f}%, sparse pts/frame "
          f"{np.stack(buf['smask']).sum(axis=(1,2)).mean():.0f}")


def to_memmap(cache_dir, fde, datasets):
    """Rewrites each compressed .npz as a directory of uncompressed .npy files.
    Training then opens them with mmap_mode='r' and reads only the frames of the
    current batch: loading all seven datasets at once would be about 7 GB
    uncompressed, against 8 GB of available RAM in this WSL setup."""
    for name in datasets:
        src = os.path.join(cache_dir, f"{name}_{fde}.npz")
        dst = os.path.join(cache_dir, f"{name}_{fde}")
        if not os.path.exists(src):
            continue
        if os.path.isdir(dst) and os.path.exists(os.path.join(dst, "valid.npy")):
            print(f"  {name}: already converted")
            continue
        os.makedirs(dst, exist_ok=True)
        d = np.load(src)
        for k in d.files:
            np.save(os.path.join(dst, f"{k}.npy"), d[k])
        print(f"  {name}: -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to_memmap", action="store_true",
                     help="convert existing .npz caches into .npy directories for memmap reading")
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fde", required=True)
    ap.add_argument("--output")
    ap.add_argument("--cache_dir", default="../cache")
    ap.add_argument("--target_frames", type=int, default=1200)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", type=float, default=60.0)
    ap.add_argument("--blade_mask", default="../masks/blades_airsim.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    todo = list(DATASETS) if args.all else [args.dataset]
    if args.to_memmap:
        to_memmap(args.cache_dir, args.fde, todo)
        return
    for name in todo:
        out = args.output if (args.output and not args.all) else \
            os.path.join(args.cache_dir, f"{name}_{args.fde}.npz")
        build(name, args.fde, out, args.target_frames, args.gt_depth_scale,
              args.min_depth, args.max_depth, args.blade_mask, args.seed)


if __name__ == "__main__":
    main()
