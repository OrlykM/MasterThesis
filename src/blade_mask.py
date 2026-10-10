r"""
Propeller-blade mask for the AirSim datasets.

The lower corners of every AirSim frame show the drone's own propellers at a
ground-truth depth of roughly 0.1-0.3 m. `min_depth = 0.1` does not remove
them, so they enter both the evaluation and any training set as "obstacles"
that are really part of the aircraft.

The mask is derived from the data rather than drawn by hand: a pixel belongs to
the mask when ground-truth depth stays below `--near_m` in at least
`--min_fraction` of frames. A blade occupies the same image region in nearly
every frame, whereas real close-range structure comes and goes, so the
persistence threshold separates the two without any geometric assumption about
where the blades are.

Usage:
    python3 blade_mask.py --dataset_roots ../datasets/run1 ../datasets/run2 ... \
        --output ../masks/blades_airsim.npz
    python3 blade_mask.py report --output ../masks/blades_airsim.npz
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_camera_intrinsics, load_image_timestamps
from compare_reconstruction_3d import find_depth_file, load_depth_frame


def compute(args):
    per_dataset = {}
    shape = None
    for root in args.dataset_roots:
        name = os.path.basename(os.path.normpath(root))
        frames = load_image_timestamps(os.path.join(root, "cam0", "data.csv"))
        intr = load_camera_intrinsics(os.path.join(root, "cam0", "sensor.yaml"))
        h, w = intr["height"], intr["width"]
        if shape is None:
            shape = (h, w)
        elif shape != (h, w):
            print(f"  {name}: {h}x{w} differs from {shape} -- skipped "
                  f"(a shared mask only makes sense for one rig)")
            continue
        near = np.zeros((h, w), dtype=np.int32)
        n = 0
        step = max(1, len(frames) // args.frames_per_run)
        for _t, fname in tqdm(frames[::step][:args.frames_per_run], desc=name, leave=False):
            p = find_depth_file(os.path.join(root, "depth0", "data"), fname)
            if p is None:
                continue
            raw = load_depth_frame(p)
            if raw.shape != (h, w):
                continue
            d = np.where(raw >= 65535, np.nan, raw) * args.gt_depth_scale
            near += ((d > 0) & (d < args.near_m)).astype(np.int32)
            n += 1
        if n:
            per_dataset[name] = near / float(n)
            print(f"  {name}: {n} frames, max persistence {per_dataset[name].max():.2f}")

    if not per_dataset:
        raise SystemExit("no dataset contributed -- check the paths")

    stack = np.stack(list(per_dataset.values()))
    # A blade must persist in EVERY contributing dataset (same airframe), so the
    # minimum across datasets is the right combiner: it rejects a region that is
    # merely close-range in one particular park.
    persistence = stack.min(axis=0)
    mask = persistence >= args.min_fraction

    np.savez_compressed(args.output, mask=mask, persistence=persistence,
                        per_dataset=np.stack(list(per_dataset.values())),
                        names=np.array(list(per_dataset.keys())),
                        near_m=args.near_m, min_fraction=args.min_fraction)
    print(f"\nSaved: {args.output}")
    report(args.output)


def applies_to(path, dataset_name):
    """Is this mask valid for `dataset_name`? The mask describes ONE airframe:
    where that aircraft's own propellers sit in its own camera. It is derived
    from, and only meaningful for, the sequences listed in its `names` field.

    This check exists because of a real mistake: the AirSim mask happens to be
    480x640, the Isaac Sim dataset is 480x640 too, so passing the same file for
    every run silently removed 8274 real pixels (2.69% of the frame, the lower
    corners, where that dataset shows the path and the ground) from a dataset
    whose aircraft has no propellers in view at all.

    A dataset not listed, and not obviously part of the same recording family,
    gets no mask rather than someone else's."""
    try:
        names = [str(x) for x in np.load(path, allow_pickle=True)["names"]]
    except (KeyError, OSError, ValueError):
        return True  # старі файли без поля names -- поводимось як раніше
    return dataset_name in names


def load_mask(path, shape=None, dataset_name=None):
    """-> boolean (H, W) mask of pixels to exclude, resized by nearest
    neighbour if the caller's frames are a different size. With dataset_name
    given, returns an all-False mask when the file does not cover that
    dataset (see applies_to)."""
    d = np.load(path, allow_pickle=True)
    mask = d["mask"]
    if dataset_name is not None and not applies_to(path, dataset_name):
        mask = np.zeros_like(mask, dtype=bool)
    if shape is not None and mask.shape != tuple(shape):
        from PIL import Image
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                        .resize((shape[1], shape[0]), Image.NEAREST)) > 127
    return mask


def report(path):
    d = np.load(path)
    mask, pers = d["mask"], d["persistence"]
    h, w = mask.shape
    print(f"mask {h}x{w}: {int(mask.sum())} px ({100*mask.mean():.2f}% of the frame), "
          f"threshold: GT depth < {float(d['near_m'])} m in >= {float(d['min_fraction']):.0%} of frames")
    print(f"datasets combined (min across them): {', '.join(str(x) for x in d['names'])}")
    rows, cols = 12, 16
    print("\npersistence map (fraction of frames with GT depth below the threshold):")
    for i in range(rows):
        r0, r1 = i * h // rows, (i + 1) * h // rows
        line = "".join(" .:-=+*#@"[min(8, int(pers[r0:r1, j * w // cols:(j + 1) * w // cols].mean() * 9))]
                       for j in range(cols))
        print(f"  {line}")
    q = mask.reshape(2, h // 2, 2, w // 2).mean(axis=(1, 3)) if h % 2 == 0 and w % 2 == 0 else None
    if q is not None:
        print(f"\nshare of each image quadrant masked:  top {q[0,0]:.3f} {q[0,1]:.3f}   "
              f"bottom {q[1,0]:.3f} {q[1,1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="compute", choices=["compute", "report"])
    ap.add_argument("--dataset_roots", nargs="+", default=[f"../datasets/run{i}" for i in range(1, 7)])
    ap.add_argument("--output", default="../masks/blades_airsim.npz")
    ap.add_argument("--frames_per_run", type=int, default=200)
    ap.add_argument("--near_m", type=float, default=0.5)
    ap.add_argument("--min_fraction", type=float, default=0.5)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.mode == "report":
        report(args.output)
    else:
        compute(args)


if __name__ == "__main__":
    main()
