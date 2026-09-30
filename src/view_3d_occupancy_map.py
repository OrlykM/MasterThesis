r"""
Views a naive_grid.npz / weighted_grid.npz occupancy map in the browser --
no OpenGL, no EGL, no native window (unlike Open3D). Same plotly + .html
approach as the rest of this project's visualizers, so it's immune to the
WSL graphics-stack issues that crash Open3D's GUI window.

Usage:
    python3 view_3d_occupancy_map.py maps/euroc/dpvo_zoedepth/weighted_grid.npz
    python3 view_3d_occupancy_map.py maps/euroc/dpvo_zoedepth/weighted_grid.npz --threshold 0.0 --output map.html
"""
import argparse
import os

import numpy as np
import plotly.graph_objects as go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_file")
    ap.add_argument("--threshold", type=float, default=0.0,
                     help="log-odds threshold for 'occupied' (0.0 = P>0.5)")
    ap.add_argument("--output", default=None, help="default: <map_file>.html next to the .npz")
    ap.add_argument("--max_points", type=int, default=200000,
                     help="subsample if more occupied voxels than this, keeps the browser responsive")
    ap.add_argument("--no_open", action="store_true")
    args = ap.parse_args()

    print(f"Завантаження карти: {args.map_file}...")
    data = np.load(args.map_file)
    log_odds, origin, resolution = data["log_odds"], data["origin"], float(data["resolution"])

    occupied_mask = log_odds > args.threshold
    occupied_idx = np.argwhere(occupied_mask)
    if len(occupied_idx) == 0:
        print("Немає зайнятих вокселів вище цього порогу!")
        return

    log_odds_occ = log_odds[occupied_mask]
    probabilities = 1.0 / (1.0 + np.exp(-log_odds_occ))
    points_world = origin + (occupied_idx * resolution) + (resolution / 2.0)

    n = len(points_world)
    if n > args.max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(n, size=args.max_points, replace=False)
        points_world, probabilities = points_world[keep], probabilities[keep]
        print(f"  {n} зайнятих вокселів -- підвибірка до {args.max_points} для швидкості рендеру")
    else:
        print(f"  {n} зайнятих вокселів")

    fig = go.Figure(go.Scatter3d(
        x=points_world[:, 0], y=points_world[:, 1], z=points_world[:, 2],
        mode="markers",
        marker=dict(
            size=3, color=probabilities, colorscale="Plasma",
            cmin=0.5, cmax=1.0, opacity=0.85,
            colorbar=dict(title="P(occupied)"),
        ),
        name="occupancy",
    ))

    lo, hi = points_world.min(axis=0), points_world.max(axis=0)
    span = np.maximum(hi - lo, 0.5)
    lo, hi = lo - 0.1 * span, hi + 0.1 * span
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X (m)", range=[lo[0], hi[0]]),
            yaxis=dict(title="Y (m)", range=[lo[1], hi[1]]),
            zaxis=dict(title="Z (m)", range=[lo[2], hi[2]]),
            bgcolor="rgb(26,26,38)",
        ),
        paper_bgcolor="rgb(26,26,38)",
        font=dict(color="white"),
        title=f"Occupancy: {os.path.basename(args.map_file)} (threshold={args.threshold})",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    output = args.output or (os.path.splitext(args.map_file)[0] + ".html")
    fig.write_html(output)
    print(f"Збережено: {output}")

    if not args.no_open:
        import webbrowser
        try:
            webbrowser.open(f"file://{os.path.abspath(output)}")
        except Exception:
            pass
        print(f"Якщо браузер не відкрився сам: explorer.exe {output}")


if __name__ == "__main__":
    main()