"""
Runs real FastDepth (MobileNet encoder + NNConv5 decoder, ICRA 2019
architecture) on every frame in <dataset_root>/cam0/data.

Model input is a fixed 224x224 (matches training) -- frames are resized down
for inference then the output depth map is resized back up to the original
camera resolution before saving, so it lines up pixel-for-pixel with the
camera's actual intrinsics for backprojection. Also writes run_config.json
documenting exactly what ran.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/opt/fastdepth")
from models import FastDepth, FastDepthV2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="/data")
    ap.add_argument("--output", default="/out")
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"],
                     help="v1 = MobileNet backbone (FastDepth_L1_Best.pth), "
                          "v2 = MobileNetV2 backbone (FastDepthV2_L1_Best.pth)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.variant == "v1":
        model = FastDepth(kernel_size=5)
        weights_path = "/opt/fastdepth/Weights/FastDepth_L1_Best.pth"
    else:
        model = FastDepthV2(kernel_size=5)
        weights_path = "/opt/fastdepth/Weights/FastDepthV2_L1_Best.pth"

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if args.device == "cuda":
        model = model.cuda()

    input_dir = os.path.join(args.dataset_root, "cam0", "data")
    os.makedirs(args.output, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"Running FastDepth ({args.variant}) on {len(files)} frames from {input_dir} ...")
    t0 = time.time()
    for f in tqdm(files):
        img = Image.open(os.path.join(input_dir, f)).convert("RGB")
        orig_w, orig_h = img.size
        img_224 = img.resize((224, 224), Image.BILINEAR)
        x = torch.from_numpy(np.array(img_224)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        if args.device == "cuda":
            x = x.cuda()
        with torch.no_grad():
            out = model(x)
        depth_224 = out.squeeze().cpu().numpy()
        depth_full = np.array(Image.fromarray(depth_224).resize((orig_w, orig_h), Image.BILINEAR))
        np.save(os.path.join(args.output, f.rsplit(".", 1)[0] + ".npy"), depth_full.astype(np.float32))
    elapsed = time.time() - t0

    config = {
        "algorithm": "FastDepth",
        "model_checkpoint": weights_path,
        "source_repo": "https://github.com/Hagaik92/FastDepth (modernized fork -- "
                        "original 2019 repo has no working docker/build path)",
        "params": vars(args),
        "n_frames": len(files),
        "elapsed_seconds": round(elapsed, 1),
        "output_units": "meters (trained on NYU Depth v2)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(args.output, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Done: {len(files)} depth maps + run_config.json saved to {args.output}")


if __name__ == "__main__":
    main()