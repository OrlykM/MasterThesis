r"""
Етапи обробки ОДНОГО кадру окремими зображеннями для презентації.

Вісім PNG розміром 640 на 480, по одному на етап, з однаковим оформленням і
спільними шкалами, щоб сусідні етапи можна було порівняти на око. Поряд
stats.txt з числами саме для цього кадру.

Дані беруться з `chosen_frame.npz`, який пише occupancy_3d_map.py з
`--dump_intermediate` і `--dump_frame_name <ім'я кадру>`, а еталонна глибина
читається з самого набору.

Спільні шкали:
    глибина            від 0 до 50 м, turbo
    похибки 03 і 06    від 0 до 20 м, magma, ОДНАКОВІ для обох
    множник s          від 0,5 до 2 з центром у 1, coolwarm
    вага w             від 0 до 1, viridis
    P кадру            від 0,50 до 0,70, RdYlGn_r

Використання:
    python3 presentation_frames.py --dataset euroc
    python3 presentation_frames.py --dataset run1
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_camera_intrinsics
from compare_reconstruction_3d import find_depth_file, load_depth_frame

ROOTS = {"euroc": "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor",
         **{f"run{i}": f"../datasets/run{i}" for i in range(1, 7)}}
LABEL = {"euroc": "№1 euroc", "run1": "№2 run1", "run2": "№3 run2", "run3": "№4 run3",
         "run4": "№5 run4", "run5": "№6 run5", "run6": "№7 run6"}

DEPTH_MAX = 50.0     # шкала глибини
ERR_MAX = 20.0       # спільна шкала похибок для 03 і 06
L_OCC = 0.85         # той самий приріст log-odds, що й у сітці


def _canvas():
    """Рівно 640 на 480 пікселів: 6,4 на 4,8 дюйма при 100 точках на дюйм."""
    return plt.subplots(figsize=(6.4, 4.8), dpi=100)


def _save(fig, out_png):
    fig.savefig(out_png, dpi=100, facecolor="white")
    plt.close(fig)


def panel(arr, out_png, title, cbar_label, cmap, vmin, vmax, bad_colour="white",
          over_colour=None):
    """Одне поле з кольоровою шкалою. Нечислові пікселі показані bad_colour."""
    fig, ax = _canvas()
    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad(bad_colour)
    if over_colour is not None:
        cm.set_over(over_colour)
    im = ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                      extend="max" if over_colour is not None else "neither")
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, out_png)


def panel_image(img, out_png, title):
    fig, ax = _canvas()
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    _save(fig, out_png)


def panel_points(img, u, v, z, out_png, title):
    """Кадр з накладеними опорними точками VSLAM, колір за їхньою глибиною."""
    fig, ax = _canvas()
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    sc = ax.scatter(u, v, c=z, cmap="turbo", vmin=0.0, vmax=DEPTH_MAX,
                    s=14, edgecolors="black", linewidths=0.4)
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title} ({len(u)} точок)", fontsize=11)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, extend="max")
    cb.set_label("глибина точки VSLAM, м", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ROOTS))
    ap.add_argument("--work_dir", default="../presentation_frames",
                     help="тут лежать <набір>/_dump і <набір>/_prob від occupancy_3d_map.py")
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    args = ap.parse_args()

    ds = args.dataset
    base = os.path.join(args.work_dir, ds)
    dump = os.path.join(base, "_dump", "chosen_frame.npz")
    if not os.path.exists(dump):
        raise SystemExit(f"немає {dump}. Спершу запусти occupancy_3d_map.py з "
                         f"--dump_intermediate і --dump_frame_name")
    d = np.load(dump, allow_pickle=True)
    fname = str(d["frame"])
    root = ROOTS[ds]
    label = LABEL.get(ds, ds)

    fde_raw = d["fde_raw"].astype(np.float64)
    calibrated = d["depth_calibrated"].astype(np.float64)
    alpha = d["alpha"].astype(np.float64)
    conf = d["confidence"].astype(np.float64)
    dropped = d["dropped"].astype(bool)
    blades = d["blades"].astype(bool) if "blades" in d.files else np.zeros_like(dropped)
    refine_dropped = d["refine_dropped"].astype(bool) if "refine_dropped" in d.files \
        else np.zeros_like(dropped)
    au, av = d["anchors_uv"][:, 0], d["anchors_uv"][:, 1]
    az = d["anchors_z"]
    S = float(d["residual_scale"]) if "residual_scale" in d.files else float("nan")
    sigma_pose = float(d["sigma_pose"]) if "sigma_pose" in d.files else float("nan")

    img = np.array(Image.open(os.path.join(root, "cam0", "data", fname)).convert("RGB"))

    gp = find_depth_file(os.path.join(root, "depth0", "data"), fname)
    gt = None
    if gp is not None:
        raw = load_depth_frame(gp)
        gt = np.where(raw >= 65535, np.nan, raw) * args.gt_depth_scale
        intr = load_camera_intrinsics(os.path.join(root, "cam0", "sensor.yaml"))
        if gt.shape != (intr["height"], intr["width"]):
            gt = np.array(Image.fromarray(gt.astype(np.float32)).resize(
                (intr["width"], intr["height"]), Image.NEAREST))

    out = base
    os.makedirs(out, exist_ok=True)

    # 01 вхідний кадр
    panel_image(img, os.path.join(out, "01_frame.png"),
                f"{label}. Вхідний кадр камери")

    # 02 сира глибина FDE
    panel(np.where(fde_raw > 0, fde_raw, np.nan), os.path.join(out, "02_fde_raw.png"),
          f"{label}. Сира глибина ZoeDepth", "глибина, м", "turbo", 0.0, DEPTH_MAX,
          over_colour="darkred")

    # 03 і 06 похибки на спільній шкалі
    err_raw = err_cal = None
    if gt is not None:
        err_raw = np.abs(fde_raw - gt)
        err_cal = np.abs(calibrated - gt)
        panel(err_raw, os.path.join(out, "03_err_raw.png"),
              f"{label}. Похибка сирої глибини", "|z − z_ет|, м", "magma", 0.0, ERR_MAX,
              over_colour="lime")
        panel(err_cal, os.path.join(out, "06_err_calib.png"),
              f"{label}. Похибка після таблиці і калібрування", "|z′ − z_ет|, м", "magma",
              0.0, ERR_MAX, over_colour="lime")

    # 04 опорні точки VSLAM поверх кадру
    panel_points(img, au, av, az, os.path.join(out, "04_vslam_points.png"),
                 f"{label}. Опорні точки VSLAM у кадрі")

    # 05 множник калібрування
    panel(alpha, os.path.join(out, "05_scale_s.png"),
          f"{label}. Множник калібрування s по ділянках", "s, безрозмірний",
          "coolwarm", 0.5, 2.0)

    # 07 вага довіри, відкинуті пікселі білі
    w_show = np.where(np.isfinite(conf), conf, np.nan)
    panel(w_show, os.path.join(out, "07_weight_w.png"),
          f"{label}. Вага довіри w (біле = відкинуто)", "w, безрозмірна",
          "viridis", 0.0, 1.0)

    # 08 внесок кадру в карту
    usable = np.isfinite(conf) & np.isfinite(calibrated) & (calibrated > 0.1)
    prob = np.where(usable, 1.0 / (1.0 + np.exp(-L_OCC * np.nan_to_num(conf))), np.nan)
    p_max = 1.0 / (1.0 + np.exp(-L_OCC))
    panel(prob, os.path.join(out, "08_prob_frame.png"),
          f"{label}. Внесок кадру в карту", f"P(зайнято), максимум {p_max:.2f}",
          "RdYlGn_r", 0.5, p_max)

    # ---------------------------------------------------------------- stats.txt
    n = dropped.size
    gt_ok = np.isfinite(gt) if gt is not None else np.ones(dropped.shape, bool)

    def med(a, m):
        v = a[m & np.isfinite(a)]
        return float(np.median(v)) if v.size else float("nan")

    # похибки рахуємо лише там, де піксель реально йде в карту і еталон існує
    scored = (~dropped) & gt_ok
    lines = [
        f"Набір                              {label}",
        f"Кадр                               {fname}",
        f"Пара                               DPVO + ZoeDepth",
        f"Таблиця корекції                   loo_{ds}_zoedepth.npz (навчена без цього набору)",
        "",
        f"Медіанна похибка глибини до        {med(err_raw, scored) if err_raw is not None else float('nan'):8.2f} м",
        f"Медіанна похибка глибини після     {med(err_cal, scored) if err_cal is not None else float('nan'):8.2f} м",
    ]
    if err_raw is not None:
        a, b = med(err_raw, scored), med(err_cal, scored)
        if np.isfinite(a) and a > 0 and np.isfinite(b):
            lines.append(f"Зміна                              {100 * (b - a) / a:+8.1f} %")
    lines += [
        "",
        f"Медіана множника s                 {float(np.median(alpha)):8.3f}",
        f"Розкид s, від 5 до 95 відсотка     {float(np.percentile(alpha, 5)):8.3f} до "
        f"{float(np.percentile(alpha, 95)):.3f}",
        f"Поточне S (residual_scale)         {S:8.2f} м",
        f"Невизначеність пози на цей кадр    {sigma_pose:8.2f} м",
        f"Опорних точок VSLAM у кадрі        {len(au):8d}",
        "",
        f"Відкинуто пікселів, разом          {100 * dropped.mean():8.2f} %",
        f"  таблиця валідності               {100 * (refine_dropped & ~blades).mean():8.2f} %",
        f"     (небо і комірки, де поверхня трапляється рідше ніж у половині випадків)",
        f"  маска лопатей гвинтів            {100 * blades.mean():8.2f} %",
        f"Пікселів іде в карту               {100 * usable.mean():8.2f} %",
        "",
        f"Середня вага w по придатних        {float(np.nanmean(np.where(usable, conf, np.nan))):8.3f}",
        f"Медіанна вага w по придатних       {med(conf, usable):8.3f}",
        f"Середня P(зайнято) з цього кадру   {float(np.nanmean(prob)):8.3f}",
    ]
    with open(os.path.join(out, "stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nЗбережено у {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
