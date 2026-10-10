r"""
Looks at the representation that exists after VSLAM + FDE and after every
correction, but BEFORE it becomes voxels -- the weighted point cloud that
ProbabilisticVoxelGrid.insert_weighted() consumes, plus the per-frame fields
it is built from.

Produced by `occupancy_3d_map.py --dump_intermediate DIR`, which writes
frame_NN.npz (the 2D fields of a few sample frames) and
cloud_before_voxels.npz (every point, its weight and its uncertainty).

Two outputs:
  frames.png    per sample frame, a row of panels: raw FDE depth -> depth after
                the network/table -> depth after the per-tile VSLAM calibration
                -> confidence -> sigma in metres -> rejected pixels, with the
                sparse VSLAM anchors marked on the first panel
  cloud.html    the weighted cloud in 3D, coloured by weight, next to the
                uncorrected baseline cloud for comparison

Usage:
    python3 view_intermediate.py --dump_dir ../maps/euroc_dump
    python3 view_intermediate.py --dump_dir ../maps/euroc_dump --colour sigma
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def single(dump_dir, index, output, dataset_root):
    """One frame, large, with the camera image alongside the fields -- the
    per-frame view of what the pipeline actually hands to the voxel grid."""
    path = os.path.join(dump_dir, f"frame_{index:02d}.npz")
    if not os.path.exists(path):
        print(f"немає {path}")
        return
    d = np.load(path, allow_pickle=True)
    depths = [d[k] for k in ("fde_raw", "depth_after_net_or_table", "depth_calibrated")]
    finite = np.concatenate([x[np.isfinite(x) & (x > 0)].ravel() for x in depths])
    vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0, 1))

    tiles = []
    if dataset_root:
        img_path = os.path.join(dataset_root, "cam0", "data", str(d["frame"]))
        if os.path.exists(img_path):
            from PIL import Image
            tiles.append((np.array(Image.open(img_path).convert("L")), "знімок камери", "gray", None))
    tiles += [(d["fde_raw"], "1. сира глибина FDE, м", "turbo", (vmin, vmax)),
              (d["depth_after_net_or_table"], "2. після мережі / таблиці, м", "turbo", (vmin, vmax)),
              (d["depth_calibrated"], "3. після калібрування по точках VSLAM, м", "turbo", (vmin, vmax)),
              (d["confidence"], "4. вага довіри w (біле = відкинуто)", "viridis", None),
              (d["sigma_m"], "5. невизначеність σ, м", "magma", None),
              (d["dropped"].astype(float), "6. відкинуті пікселі", "gray", (0, 1))]

    ncol = 3
    nrow = int(np.ceil(len(tiles) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (a, title, cmap, lim) in enumerate(tiles):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        a = np.asarray(a, dtype=np.float32)
        kw = dict(vmin=lim[0], vmax=lim[1]) if lim else {}
        im = ax.imshow(np.where(a > 0, a, np.nan) if "глибина" in title else a, cmap=cmap, **kw)
        if "сира глибина" in title and "anchors_uv" in d.files and len(d["anchors_uv"]):
            uv = d["anchors_uv"]
            ax.scatter(uv[:, 0], uv[:, 1], s=3, c="white", alpha=0.8, linewidths=0)
            ax.set_title(f"{title}   +{len(uv)} точок VSLAM (білим)", fontsize=11)
        else:
            ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if cmap != "gray" or lim == (0, 1):
            fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"Кадр {index}: що пайплайн подає у вокселізацію", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=110)
    plt.close(fig)
    print(f"Збережено: {output}")


def panels(dump_dir, output, max_frames):
    files = sorted(glob.glob(os.path.join(dump_dir, "frame_*.npz")))[:max_frames]
    if not files:
        print("немає frame_*.npz -- запусти occupancy_3d_map.py з --dump_intermediate")
        return
    fields = [("fde_raw", "сира глибина FDE", "turbo", "m"),
              ("depth_after_net_or_table", "після мережі/таблиці", "turbo", "m"),
              ("depth_calibrated", "після калібрування VSLAM", "turbo", "m"),
              ("confidence", "вага довіри w", "viridis", ""),
              ("sigma_m", "невизначеність σ", "magma", "m"),
              ("dropped", "відкинуті пікселі", "gray", "")]
    fig, axes = plt.subplots(len(files), len(fields),
                             figsize=(3.1 * len(fields), 2.5 * len(files)), squeeze=False)
    for r, path in enumerate(files):
        d = np.load(path, allow_pickle=True)
        # a shared depth scale across the three depth panels, so the correction
        # is visible as a change in the image rather than in the colour bar
        depths = [d[k] for k, *_ in fields[:3]]
        finite = np.concatenate([x[np.isfinite(x) & (x > 0)].ravel() for x in depths])
        vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0, 1))
        for c, (key, title, cmap, unit) in enumerate(fields):
            ax = axes[r][c]
            a = np.asarray(d[key], dtype=np.float32)
            if key in ("fde_raw", "depth_after_net_or_table", "depth_calibrated"):
                im = ax.imshow(np.where(a > 0, a, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
            elif key == "dropped":
                im = ax.imshow(a, cmap=cmap, vmin=0, vmax=1)
            else:
                im = ax.imshow(a, cmap=cmap)
            if c == 0 and "anchors_uv" in d.files and len(d["anchors_uv"]):
                uv = d["anchors_uv"]
                ax.scatter(uv[:, 0], uv[:, 1], s=1.2, c="white", alpha=0.65, linewidths=0)
                title = f"{title}\n+ {len(uv)} точок VSLAM"
            ax.set_title(title if r == 0 else "", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 or c >= 3:
                cb = fig.colorbar(im, ax=ax, fraction=0.045)
                cb.ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(f"кадр {r}", fontsize=8)
    fig.suptitle("Репрезентація перед вокселізацією: як глибина змінюється на кожному кроці",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=110)
    plt.close(fig)
    print(f"Збережено: {output}  ({len(files)} кадрів)")


def cloud(dump_dir, output, colour, max_points, show_baseline):
    path = os.path.join(dump_dir, "cloud_before_voxels.npz")
    if not os.path.exists(path):
        print("немає cloud_before_voxels.npz")
        return
    d = np.load(path)
    pts, w, sig = d["points"], d["weights"], d["sigma_m"]
    rng = np.random.default_rng(0)
    if len(pts) > max_points:
        k = rng.choice(len(pts), max_points, replace=False)
        pts, w, sig = pts[k], w[k], sig[k]

    val, label, scale = ((w, "вага довіри w", "Viridis") if colour == "weight"
                         else (sig, "невизначеність σ, м", "Magma"))
    traces = []
    if show_baseline and "points_naive" in d.files:
        nb = d["points_naive"]
        if len(nb) > max_points // 2:
            nb = nb[rng.choice(len(nb), max_points // 2, replace=False)]
        traces.append(go.Scatter3d(x=nb[:, 0], y=nb[:, 1], z=nb[:, 2], mode="markers",
                                    marker=dict(size=1.2, color="rgba(150,150,150,0.25)"),
                                    name="без корекції (базова)"))
    traces.append(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=1.8, color=val, colorscale=scale, opacity=0.85,
                    colorbar=dict(title=label)),
        name="після корекції, з вагами"))
    if "origins" in d.files:
        o = d["origins"]
        traces.append(go.Scatter3d(x=o[:, 0], y=o[:, 1], z=o[:, 2], mode="lines+markers",
                                    line=dict(color="lime", width=3),
                                    marker=dict(size=2, color="lime"), name="камера"))

    fig = go.Figure(traces)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 0.5)
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis=dict(title="X, м", range=[lo[0] - 0.1 * span[0], hi[0] + 0.1 * span[0]]),
                   yaxis=dict(title="Y, м", range=[lo[1] - 0.1 * span[1], hi[1] + 0.1 * span[1]]),
                   zaxis=dict(title="Z, м", range=[lo[2] - 0.1 * span[2], hi[2] + 0.1 * span[2]]),
                   bgcolor="rgb(26,26,38)"),
        paper_bgcolor="rgb(26,26,38)", font=dict(color="white"),
        title=f"Зважена хмара перед вокселізацією ({len(pts)} точок, колір = {label})",
        margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html(output)
    print(f"Збережено: {output}  ({len(pts)} точок, w від {w.min():.3f} до {w.max():.3f}, "
          f"σ медіана {np.median(sig):.2f} м)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--colour", choices=["weight", "sigma"], default="weight")
    ap.add_argument("--max_points", type=int, default=120000)
    ap.add_argument("--max_frames", type=int, default=6)
    ap.add_argument("--no_baseline", action="store_true")
    ap.add_argument("--frame", type=int,
                     help="render ONE frame large instead of the grid of all of them")
    ap.add_argument("--dataset_root", help="with --frame: also show the camera image")
    ap.add_argument("--no_cloud", action="store_true", help="skip the 3D cloud")
    args = ap.parse_args()
    if args.frame is not None:
        single(args.dump_dir, args.frame,
               os.path.join(args.dump_dir, f"frame_{args.frame:02d}.png"), args.dataset_root)
        return
    panels(args.dump_dir, os.path.join(args.dump_dir, "frames.png"), args.max_frames)
    if not args.no_cloud:
        cloud(args.dump_dir, os.path.join(args.dump_dir, "cloud.html"), args.colour,
              args.max_points, not args.no_baseline)


if __name__ == "__main__":
    main()
