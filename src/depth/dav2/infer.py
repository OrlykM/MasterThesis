"""
Runs Depth-Anything-V2 (metric checkpoint, via transformers) on every frame
in <dataset_root>/cam0/data, saving one float32 .npy depth map (meters) per
frame to <output>, plus a run_config.json documenting exactly what ran.

--relative_depth switches to the plain (non-metric) checkpoint -- output is
then relative/disparity-like and CANNOT be backprojected into metric 3D
without a separate scale-calibration step. Default is metric.
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="/data")
    ap.add_argument("--output", default="/out")
    ap.add_argument("--encoder", default="large", choices=["small", "base", "large"])
    ap.add_argument("--scene", default="outdoor", choices=["indoor", "outdoor"])
    ap.add_argument("--relative_depth", action="store_true")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.relative_depth:
        model_id = f"depth-anything/Depth-Anything-V2-{args.encoder.capitalize()}-hf"
        print("WARNING: --relative_depth gives non-metric output, do not backproject directly.")
    else:
        model_id = f"depth-anything/Depth-Anything-V2-Metric-{args.scene.capitalize()}-{args.encoder.capitalize()}-hf"

    input_dir = os.path.join(args.dataset_root, "cam0", "data")
    os.makedirs(args.output, exist_ok=True)

    device_idx = 0 if args.device == "cuda" else -1
    pipe = pipeline("depth-estimation", model=model_id, device=device_idx)

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"Running {model_id} on {len(files)} frames from {input_dir} ...")
    t0 = time.time()
    for f in tqdm(files):
        img = Image.open(os.path.join(input_dir, f)).convert("RGB")
        d = pipe(img)["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
        np.save(os.path.join(args.output, f.rsplit(".", 1)[0] + ".npy"), d)
    elapsed = time.time() - t0

    config = {
        "algorithm": "Depth-Anything-V2",
        "model_checkpoint": model_id,
        "params": vars(args),
        "n_frames": len(files),
        "elapsed_seconds": round(elapsed, 1),
        "output_units": "relative/disparity-like" if args.relative_depth else "meters (metric checkpoint)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(args.output, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done: {len(files)} depth maps + run_config.json saved to {args.output}")


if __name__ == "__main__":
    main()