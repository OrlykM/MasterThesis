r"""
Статичні рисунки для звіту з уже побудованих інтерактивних HTML і воксельних
сіток.

Навіщо окремий модуль. Plotly вміє експортувати PNG сам, але для цього йому
потрібен Chrome, а у WSL тут бракує системних бібліотек і sudo просить пароль.
Тому фігура читається з HTML як дані і перемальовується matplotlib. Це заразом
дає повний контроль над камерою, тобто дозволяє взяти саме вигляд збоку, той,
на якому видно висоту сцени і положення траєкторії.

Що робить:
  maps/<набір>/<vslam>_<fde>/fine/adaptive_grid.npz -> вигляд реконструкції
      збоку, колір за ймовірністю зайнятості, плюс лінія еталонної траєкторії
  maps/<набір>/<vslam>_<fde>/prob2d/*.png           -> один кадр 2D карти
      ймовірності, скопійований поряд
  comparisons/<набір>/<vslam>_<fde>.html            -> вигляд збоку тих самих
      чотирьох шарів, що й у HTML, тобто дві траєкторії і дві хмари

Вихід у docs/images/<набір>/<vslam>_<fde>/.

Використання:
    python3 render_figures.py --all
    python3 render_figures.py --all --azimuth 0 --elevation 8
    python3 render_figures.py --maps_only
"""
import argparse
import base64
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASETS = ["euroc"] + [f"run{i}" for i in range(1, 7)]
DATASET_LABEL = {"euroc": "№1 euroc", "run1": "№2 run1", "run2": "№3 run2", "run3": "№4 run3",
                 "run4": "№5 run4", "run5": "№6 run5", "run6": "№7 run6"}
ROOTS = {"euroc": "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor",
         **{f"run{i}": f"../datasets/run{i}" for i in range(1, 7)}}

# Підписи в comparisons згенеровані англійською, а рисунки йдуть у звіт
NAMES = {"GT trajectory": "еталонна траєкторія",
         "Estimated trajectory (aligned)": "оцінена траєкторія, вирівняна",
         "GT reconstruction": "еталонна хмара точок",
         "Estimated reconstruction (aligned)": "оцінена хмара точок, вирівняна"}


# ---------------------------------------------------------------- читання HTML

def _decode(v):
    """Plotly 6 пакує масиви як {"dtype": "f8", "bdata": "<base64>"}. Списки
    лишає списками."""
    if isinstance(v, dict) and "bdata" in v:
        arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v.get("dtype", "f8")))
        shape = v.get("shape")
        if shape:
            arr = arr.reshape([int(x) for x in str(shape).split(",")])
        return arr
    if isinstance(v, list):
        return np.asarray(v)
    return v


def load_plotly_html(path):
    """-> (traces, layout) з розкодованими масивами, або None."""
    with open(path, encoding="utf-8") as f:
        s = f.read()
    m = re.search(r"Plotly\.(?:newPlot|react)\(", s)
    if m is None:
        return None
    seg = s[m.end():]
    j = seg.index("[")
    dec = json.JSONDecoder()
    traces, k = dec.raw_decode(seg[j:])
    rest = seg[j + k:].lstrip().lstrip(",").lstrip()
    try:
        layout, _ = dec.raw_decode(rest)
    except ValueError:
        layout = {}
    for t in traces:
        for key in ("x", "y", "z"):
            if key in t:
                t[key] = _decode(t[key])
        mk = t.get("marker")
        if isinstance(mk, dict) and "color" in mk:
            mk["color"] = _decode(mk["color"])
    return traces, layout


# ------------------------------------------------------------- спільне креслення

def _equal_box(ax, pts):
    """Межі по справжніх даних, а пропорції коробки задані відношенням
    протяжностей. Кубічна коробка з однаковим радіусом по трьох осях була б
    теж правильною за масштабом, але сцена тут витягнута, тобто 260 на 144 на
    55 м на наборі №1, і в кубі вона займала б малу частину кадру. Так масштаб
    лишається чесним по всіх трьох осях, а кадр заповнений."""
    if len(pts) == 0:
        return
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.04 * span
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_box_aspect(tuple(span / span.max()))


def _crop_white(path, margin=12):
    """Обрізає білі поля по краях. Потрібне тому, що у тривимірних осей
    matplotlib вважає зайнятою всю коробку, навіть коли сцена сплюснута і
    реально займає вузьку смугу посередині, тому ні tight_layout, ні
    bbox_inches цього не прибирають."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    a = np.asarray(img)
    non_white = np.any(a < 250, axis=2)
    rows, cols = np.any(non_white, axis=1), np.any(non_white, axis=0)
    if not rows.any() or not cols.any():
        return
    r0, r1 = np.flatnonzero(rows)[[0, -1]]
    c0, c1 = np.flatnonzero(cols)[[0, -1]]
    img.crop((max(0, c0 - margin), max(0, r0 - margin),
              min(a.shape[1], c1 + margin + 1), min(a.shape[0], r1 + margin + 1))).save(path)


def _finish(ax, fig, title, out_png, elev, azim):
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X, м", fontsize=8)
    ax.set_ylabel("Y, м", fontsize=8)
    ax.set_zlabel("Z, м", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.locator_params(axis="z", nbins=5)
    ax.set_title(title, fontsize=10)
    # tight_layout погано поводиться з тривимірними осями, у яких коробка
    # сплюснута, тому поля задаються явно
    fig.subplots_adjust(left=0.01, right=0.88, top=0.90, bottom=0.03)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _crop_white(out_png)


# ------------------------------------------------------- воксельна сітка -> PNG

def load_grid(npz_path):
    d = np.load(npz_path)
    return d["log_odds"], np.asarray(d["origin"], dtype=float), float(d["resolution"])


def load_trajectory(dataset):
    """Еталонна траєкторія, щоб на рисунку було видно, де летів апарат."""
    root = ROOTS.get(dataset)
    if root is None:
        return None
    csv = os.path.join(root, "state_groundtruth_estimate0", "data.csv")
    if not os.path.exists(csv):
        return None
    try:
        a = np.genfromtxt(csv, delimiter=",", skip_header=1, usecols=(1, 2, 3), comments=None)
        return a[np.isfinite(a).all(axis=1)]
    except Exception:
        return None


def render_grid(npz_path, out_png, title, dataset, elev, azim, max_points, threshold=0.0):
    log_odds, origin, res = load_grid(npz_path)
    idx = np.argwhere(log_odds > threshold)
    if len(idx) == 0:
        return False
    prob = 1.0 / (1.0 + np.exp(-log_odds[tuple(idx.T)]))
    pts = origin + (idx + 0.5) * res
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, prob = pts[sel], prob[sel]

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=prob, cmap="inferno",
                    s=2.0, vmin=0.5, vmax=1.0, linewidths=0, depthshade=False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.10, shrink=0.5)
    cb.set_label("P(зайнято)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    traj = load_trajectory(dataset)
    box = pts
    if traj is not None and len(traj):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#00ff41", linewidth=2.2,
                label="еталонна траєкторія", zorder=10)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.82))
        box = np.vstack([pts, traj])
    _equal_box(ax, box)
    _finish(ax, fig, f"{title}\nвоксель {res:.2f} м, зайнятих {len(idx)}", out_png, elev, azim)
    return True


# ------------------------------------------------- фігура з comparisons -> PNG

def render_html_scatter3d(html_path, out_png, title, elev, azim, max_points):
    loaded = load_plotly_html(html_path)
    if loaded is None:
        return False
    traces, _layout = loaded
    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(0)
    all_pts, drawn = [], 0

    for t in traces:
        if t.get("type") != "scatter3d":
            continue
        x, y, z = (np.asarray(t.get(k, []), dtype=float) for k in ("x", "y", "z"))
        if x.size == 0:
            continue
        name = NAMES.get(t.get("name") or "", t.get("name") or "")
        mode = t.get("mode") or "markers"
        if "lines" in mode:
            col = (t.get("line") or {}).get("color", "black")
            ax.plot(x, y, z, color=col, linewidth=1.8, label=name)
        else:
            mk = t.get("marker") or {}
            col = mk.get("color", "gray")
            if x.size > max_points:
                sel = rng.choice(x.size, max_points, replace=False)
                x, y, z = x[sel], y[sel], z[sel]
                if isinstance(col, np.ndarray) and col.size == len(sel):
                    col = col[sel]
            try:
                alpha = float(mk.get("opacity", 0.55))
            except (TypeError, ValueError):
                alpha = 0.55
            ax.scatter(x, y, z, c=col if not isinstance(col, np.ndarray) else col,
                       s=1.5, alpha=alpha, linewidths=0, label=name, depthshade=False)
        all_pts.append(np.stack([x, y, z], axis=1))
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        return False
    _equal_box(ax, np.vstack(all_pts))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.85))
    _finish(ax, fig, title, out_png, elev, azim)
    return True


# ------------------------------------------------------------------ 2D кадр

def copy_prob_frame(prob_dir, out_png):
    """Один кадр 2D карти ймовірності, з середини польоту, а не з самого
    початку, де камера ще може дивитись у землю або в небо."""
    files = sorted(glob.glob(os.path.join(prob_dir, "prob_*.png")))
    if not files:
        return False
    shutil.copyfile(files[len(files) // 2], out_png)
    return True


# --------------------------------------------------------------------- прогін

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="../docs/images")
    ap.add_argument("--maps_dir", default="../maps")
    ap.add_argument("--comparisons_dir", default="../comparisons")
    ap.add_argument("--elevation", type=float, default=12.0,
                     help="кут над горизонтом. Малий кут це і є вигляд збоку")
    ap.add_argument("--azimuth", type=float, default=-70.0)
    ap.add_argument("--max_points", type=int, default=120000)
    ap.add_argument("--maps_only", action="store_true")
    ap.add_argument("--comparisons_only", action="store_true")
    ap.add_argument("--all", action="store_true", help="лишено для сумісності, нічого не змінює")
    args = ap.parse_args()

    jobs = []
    if not args.comparisons_only:
        for ds in DATASETS:
            for pair_dir in sorted(glob.glob(os.path.join(args.maps_dir, ds, "*"))):
                if not os.path.isdir(pair_dir):
                    continue
                jobs.append(("map", ds, os.path.basename(pair_dir), pair_dir))
    if not args.maps_only:
        for ds in DATASETS:
            for html in sorted(glob.glob(os.path.join(args.comparisons_dir, ds, "*.html"))):
                jobs.append(("cmp", ds, os.path.splitext(os.path.basename(html))[0], html))

    made, skipped = 0, []
    for kind, ds, pair, src in tqdm(jobs, desc="рисунки"):
        out = os.path.join(args.out_dir, ds, pair)
        os.makedirs(out, exist_ok=True)
        label = f"{DATASET_LABEL.get(ds, ds)}, {pair.replace('_', ' + ')}"

        if kind == "map":
            # 3D реконструкція з дрібної сітки, ймовірнісна або базова
            fine = os.path.join(src, "fine")
            grid = None
            for cand, what in ((os.path.join(fine, "adaptive_grid.npz"), "Ймовірнісна карта"),
                                (os.path.join(fine, "naive_grid.npz"), "Базова карта, ORB-SLAM3 не дає розріджених точок")):
                if os.path.exists(cand):
                    grid, what_label = cand, what
                    break
            if grid is not None:
                ok = render_grid(grid, os.path.join(out, "reconstruction_side.png"),
                                  f"{label}. {what_label}, вигляд збоку", ds,
                                  args.elevation, args.azimuth, args.max_points)
                made += int(ok)
                if not ok:
                    skipped.append(f"{ds}/{pair}: у сітці немає зайнятих вокселів")
            else:
                skipped.append(f"{ds}/{pair}: немає fine/*.npz")
            # один 2D кадр
            if copy_prob_frame(os.path.join(src, "prob2d"), os.path.join(out, "prob2d_frame.png")):
                made += 1
            else:
                skipped.append(f"{ds}/{pair}: немає prob2d")
        else:
            ok = render_html_scatter3d(src, os.path.join(out, "comparison_side.png"),
                                        f"{label}. Траєкторії і хмари точок, вигляд збоку",
                                        args.elevation, args.azimuth, args.max_points)
            made += int(ok)
            if not ok:
                skipped.append(f"{ds}/{pair}: не вдалось прочитати {src}")

    print(f"\nЗбережено рисунків: {made}")
    if skipped:
        print(f"Пропущено {len(skipped)}:")
        for s in skipped:
            print(f"  {s}")
    print(f"Каталог: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
