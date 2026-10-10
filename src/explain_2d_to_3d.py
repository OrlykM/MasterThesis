r"""
Ілюстрація для презентації: як двовимірна карта довіри одного кадру стає
тривимірною воксельною картою.

Усе зроблено з РЕАЛЬНОГО прогону. Скрипт не повторює конвеєр, а запускає
`occupancy_3d_map.run_dataset_mode` з необов'язковими гачками, які той приймає
за замовчуванням порожніми, і забирає проміжні величини прямо з нього. Тому
зворотна проєкція, калібрування по плитках, таблиця корекції, маска лопатей,
вето неба і накопичення в логарифмі шансів тут ті самі функції, що й у
справжньому конвеєрі, а не їхні копії.

Два пікселі простежуються наскрізь:
    A  на справжній перешкоді, де вага довіри висока
    B  у ділянці, яку відкинула таблиця корекції, тобто небо

Вихід у presentation_frames/<набір>/2d_to_3d/:
    trace.json                  усі проміжні значення для A і B
    01_prob2d.png               карта P кадру з позначеними пікселями
    02_backprojection.png       піраміда огляду, хмара кадру, промені
    03_voxel_update.png         оновлення вокселів плюс переріз уздовж променя
    04_accumulation.png         як росте L і P вокселя, що містить p_A
    05_local_map.png            фрагмент накопиченої карти проти еталона
    00_pipeline_2d_to_3d.png    збірний рисунок 16 на 9 з підписами
    00_pipeline_2d_to_3d_clean.png   те саме без підписів під панелями
    02_backprojection.html      те саме, що панель b, інтерактивно
    03_voxel_update.html        те саме, що панель c, інтерактивно

Використання:
    python3 explain_2d_to_3d.py
    python3 explain_2d_to_3d.py --n_frames 2600 --frame 1775034831515681460.png
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
import plotly.graph_objects as go
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import occupancy_3d_map as om
from occupancy_3d_map import ProbabilisticVoxelGrid, backproject_with_weight_map
from compare_reconstruction_3d import find_depth_file, load_depth_frame

ROOT_EUROC = "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor"
FRAME_EUROC = "1775034831515681460.png"
L_OCC, L_FREE = 0.85, -0.4
PANEL_SIZE = (16, 12)      # 1600 на 1200 при dpi 100
STEP_TITLES = ["2D карта довіри кадру", "Зворотна проєкція",
               "Оновлення вокселів", "Накопичення по кадрах"]


# ------------------------------------------------------------------ допоміжне

def _cb(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, fontsize=13)
    cb.ax.tick_params(labelsize=11)
    return cb


def _save(fig, path, crop=False, tight=True):
    fig.savefig(path, dpi=100, facecolor="white",
                bbox_inches="tight" if tight else None)
    plt.close(fig)
    if crop:
        img = Image.open(path).convert("RGB")
        a = np.asarray(img)
        nz = np.any(a < 250, axis=2)
        r, c = np.any(nz, axis=1), np.any(nz, axis=0)
        if r.any() and c.any():
            r0, r1 = np.flatnonzero(r)[[0, -1]]
            c0, c1 = np.flatnonzero(c)[[0, -1]]
            img = img.crop((max(0, c0 - 14), max(0, r0 - 14),
                            min(a.shape[1], c1 + 15), min(a.shape[0], r1 + 15)))
        # Рівно 1600 на 1200, як просили: вміст вписується зі збереженням
        # пропорцій, залишок добивається білим.
        tw, th = 1600, 1200
        k = min(tw / img.width, th / img.height)
        img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))),
                         Image.LANCZOS)
        canvas = Image.new("RGB", (tw, th), "white")
        canvas.paste(img, ((tw - img.width) // 2, (th - img.height) // 2))
        canvas.save(path)
    print(f"  {os.path.basename(path)}")


def _frustum(T_wc, K, w, h, dist):
    """Кути зображення, перенесені в простір на відстань dist. Та сама формула
    зворотної проєкції, що й у backproject_with_weight_map."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uv = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
    pts = np.stack([(uv[:, 0] - cx) * dist / fx, (uv[:, 1] - cy) * dist / fy,
                    np.full(len(uv), dist), np.ones(len(uv))], axis=1)
    return (T_wc @ pts.T).T[:, :3]


def _pixel_to_world(u, v, z, K, T_wc):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0])
    return (T_wc @ p_cam)[:3], p_cam[:3]


# ------------------------------------------------------- вибір пікселів A і B

def choose_pixels(ctx, gt_depth, n_extra=18, rng_seed=0):
    """A: висока вага, справжня перешкода, у верхній половині кадру, щоб це було
    дерево чи будівля, а не земля під апаратом. B: піксель, відкинутий таблицею,
    де еталон не має відбиття, тобто небо.

    Еталон тут потрібен ЛИШЕ щоб обрати показовий піксель для рисунка. У самому
    конвеєрі він не бере участі, і в trace.json це зазначено."""
    conf = ctx["confidence"]
    depth = ctx["depth_calibrated"]
    dropped = ctx["dropped"]
    h, w = conf.shape
    rows = np.arange(h)[:, None] * np.ones((1, w))

    usable = np.isfinite(conf) & np.isfinite(depth) & (depth > ctx["min_depth"]) \
        & (depth < ctx["max_depth"])
    real = np.isfinite(gt_depth) & (gt_depth > 3.0) & (gt_depth < 35.0)
    cand_a = usable & real & (rows < 0.62 * h)
    if not cand_a.any():
        cand_a = usable & real
    thr = np.nanpercentile(conf[cand_a], 85)
    strong = cand_a & (conf >= thr)
    vs, us = np.nonzero(strong)
    # з найвпевненіших беремо найвищий у кадрі, тобто найдальший від землі
    k = int(np.argmin(vs))
    a_uv = (int(us[k]), int(vs[k]))

    sky = ~np.isfinite(gt_depth)
    cand_b = dropped & sky & (rows < 0.45 * h)
    if not cand_b.any():
        cand_b = dropped & sky
    vs, us = np.nonzero(cand_b)
    mid = len(us) // 2
    order = np.argsort(us)
    b_uv = (int(us[order[mid]]), int(vs[order[mid]]))

    # 18 рівномірно рознесених придатних пікселів для променів
    extra = []
    ny, nx = 3, 6
    for iy in range(ny):
        for ix in range(nx):
            v0 = int((iy + 0.5) * h / ny)
            u0 = int((ix + 0.5) * w / nx)
            win = 60
            sub = usable[max(0, v0 - win):v0 + win, max(0, u0 - win):u0 + win]
            if not sub.any():
                continue
            svs, sus = np.nonzero(sub)
            d = (svs - min(v0, win)) ** 2 + (sus - min(u0, win)) ** 2
            j = int(np.argmin(d))
            extra.append((int(sus[j] + max(0, u0 - win)), int(svs[j] + max(0, v0 - win))))
    extra = [p for p in extra if p != a_uv][:n_extra]
    return a_uv, b_uv, extra


# ------------------------------------------------------------- простеження A і B

def build_trace(ctx, gt_depth, a_uv, b_uv, grid_res, grid_origin, dl_actual, n_seen):
    def at(arr, uv):
        return float(arr[uv[1], uv[0]]) if arr is not None else None

    out = {"кадр": ctx["fname"], "роздільність_вокселя_м": grid_res,
           "примітка": "еталонна глибина використана лише щоб обрати показові пікселі, "
                       "у конвеєрі вона участі не бере"}
    for name, uv in (("A", a_uv), ("B", b_uv)):
        u, v = uv
        z_fde = at(ctx["fde_raw"], uv)
        z_tab = at(ctx["depth_after_table"], uv)
        z_cal = at(ctx["depth_calibrated"], uv)
        s = at(ctx["alpha"], uv)
        sig_d = at(ctx["sigma_depth"], uv)
        sup = at(ctx["support"], uv)
        S = float(ctx["local_calib"].residual_scale.value)
        conf = at(ctx["confidence"], uv)
        rec = {
            "u": u, "v": v,
            "z_fde_м": z_fde,
            "множник_таблиці_r": (z_tab / z_fde) if (z_fde and z_fde > 1e-6) else None,
            "z_після_таблиці_м": z_tab,
            "множник_калібрування_s": s,
            "z_штрих_м": z_cal,
            "розбіжність_ділянки_r_g_м": (sig_d / (1.0 + sup)) if (sig_d is not None and sup is not None) else None,
            "ширина_ядра_відносна_support": sup,
            "сигма_глибини_м": sig_d,
            "сигма_пози_м": float(ctx["sigma_pose"]),
            "сигма_повна_м": at(ctx["sigma_total"], uv),
            "поточне_S_м": S,
            "вага_w": conf if (conf is not None and np.isfinite(conf)) else None,
            "еталонна_глибина_м": (None if not np.isfinite(gt_depth[v, u]) else float(gt_depth[v, u])),
        }
        if np.isfinite(ctx["confidence"][v, u]) and z_cal and z_cal > ctx["min_depth"]:
            p_world, p_cam = _pixel_to_world(u, v, z_cal, ctx["K"], ctx["T_wc"])
            idx = np.floor((p_world - grid_origin) / grid_res).astype(int)
            rec["точка_в_камері_м"] = [round(float(x), 3) for x in p_cam]
            rec["точка_у_світі_м"] = [round(float(x), 3) for x in p_world]
            rec["індекс_вокселя"] = [int(x) for x in idx]
            rec["ΔL_occ_внесок_цього_кадру"] = round(L_OCC * float(conf), 4)
            if dl_actual is not None and name == "A":
                rec["ΔL_occ_фактична_зміна_у_повному_прогоні"] = round(float(dl_actual), 4)
                if abs(float(dl_actual)) < 1e-6:
                    rec["чому_зміна_нульова"] = ("воксель уже вперся в межу накопичення "
                                                  "L = 4, далі свідчення не додаються")
                rec["кадрів_що_бачили_цей_воксель"] = int(n_seen)
        else:
            reasons = []
            if ctx["blades"] is not None and bool(ctx["blades"][v, u]):
                reasons.append("маска лопатей гвинтів")
            if ctx["refine_keep"] is not None and not bool(ctx["refine_keep"][v, u]):
                reasons.append("таблиця корекції: у цій комірці поверхня трапляється рідше "
                               "ніж у половині випадків, і вето з зображення її не скасувало")
            if not reasons:
                reasons.append("глибина поза робочим діапазоном")
            rec["відкинуто"] = True
            rec["причина"] = "; ".join(reasons)
            rec["у_карту_не_потрапляє"] = True
            if sig_d is not None and np.isfinite(sig_d):
                sig_tot = sig_d + float(ctx["sigma_pose"])
                rec["вага_якби_не_відкинули"] = round(1.0 / (1.0 + sig_tot / S), 4)
                rec["тобто"] = ("навіть без відкидання механізм довіри дав би цьому пікселю "
                                "малу вагу, бо його невизначеність удвічі більша за типову "
                                "для цього прогону")
        out[name] = rec
    return out


# ------------------------------------------------------------------- панель a

def panel_prob2d(ctx, a_uv, b_uv, extra, path):
    conf = ctx["confidence"]
    depth = ctx["depth_calibrated"]
    usable = np.isfinite(conf) & np.isfinite(depth) & (depth > ctx["min_depth"]) \
        & (depth < ctx["max_depth"])
    prob = np.where(usable, 1.0 / (1.0 + np.exp(-L_OCC * np.nan_to_num(conf))), np.nan)
    p_max = 1.0 / (1.0 + np.exp(-L_OCC))

    fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=100)
    cm = matplotlib.colormaps["RdYlGn_r"].copy()
    cm.set_bad("white")
    im = ax.imshow(prob, cmap=cm, vmin=0.5, vmax=p_max, interpolation="nearest")
    for (u, v) in extra:
        ax.plot(u, v, "o", ms=9, mfc="none", mec="black", mew=1.6)
    ax.plot(*a_uv, "o", ms=20, mfc="none", mec="#0040ff", mew=4)
    ax.annotate("A  перешкода, висока вага", a_uv, xytext=(max(12, a_uv[0] - 55), a_uv[1] + 125),
                color="#0040ff", fontsize=17, fontweight="bold",
                arrowprops=dict(color="#0040ff", width=2.5, headwidth=11))
    ax.plot(*b_uv, "x", ms=22, mec="#d000d0", mew=5)
    ax.annotate("B  небо, відкинуто", b_uv, xytext=(b_uv[0] + 55, b_uv[1] + 95),
                color="#d000d0", fontsize=17, fontweight="bold",
                arrowprops=dict(color="#d000d0", width=2.5, headwidth=11))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Карта ймовірності зайнятості одного кадру\nбіле означає, що піксель "
                 "відкинуто і в карту він не йде", fontsize=19)
    _cb(fig, im, ax, f"P(зайнято) з одного кадру, максимум {p_max:.2f}")
    _save(fig, path, crop=True)


# ------------------------------------------------------------------- панель b

def _view(ax, pts, elev=16, azim=-62):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.05 * span
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_box_aspect(tuple(span / span.max()))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X, м", fontsize=12)
    ax.set_ylabel("Y, м", fontsize=12)
    ax.set_zlabel("Z, м", fontsize=12)
    ax.tick_params(labelsize=10)


def panel_backprojection(ctx, cloud, weights, ray_pts, a_world, path, max_points=40000):
    o = ctx["origin"]
    pts, wts = cloud, weights
    if len(pts) > max_points:
        k = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, wts = pts[k], wts[k]

    fig = plt.figure(figsize=PANEL_SIZE, dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=wts, cmap="viridis",
                    vmin=0.0, vmax=1.0, s=2.0, linewidths=0, depthshade=False)

    near = _frustum(ctx["T_wc"], ctx["K"], *ctx["confidence"].shape[::-1], dist=4.0)
    for i in range(4):
        seg = np.stack([o, near[i]])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="black", lw=1.4)
    loop = np.vstack([near, near[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="black", lw=2.0,
            label="піраміда огляду і площина зображення")

    for i, p in enumerate(ray_pts):
        seg = np.stack([o, p])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#505050", lw=1.1, alpha=0.8,
                label=f"промені решти {len(ray_pts)} пікселів" if i == 0 else None)
    seg = np.stack([o, a_world])
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#0040ff", lw=3.4, label="промінь до пікселя A")
    ax.scatter(*a_world, color="#0040ff", s=140, depthshade=False)
    ax.text(a_world[0], a_world[1], a_world[2],
            f"  p_A = ({a_world[0]:.1f}, {a_world[1]:.1f}, {a_world[2]:.1f}) м",
            color="#0040ff", fontsize=15, fontweight="bold")
    ax.scatter(*o, color="red", s=130, depthshade=False, label="центр камери")

    _view(ax, np.vstack([pts, near, [o]]))
    ax.legend(fontsize=13, loc="upper left")
    ax.set_title("Зворотна проєкція кадру в простір\nколір точки це її вага довіри", fontsize=19)
    _cb(fig, sc, ax, "вага довіри w, безрозмірна")
    _save(fig, path, crop=True)


# ------------------------------------------------------------------- панель c

def one_frame_grid(ctx, pts, wts, sig, res, n_ray_samples, k_sigma):
    """Сітка, у яку вставлено ЛИШЕ цей кадр, тією самою функцією, що й у
    конвеєрі. Оскільки сітка починається з нуля, її log_odds і є ΔL цього кадру."""
    allp = np.vstack([pts, ctx["origin"][None, :]])
    lo, hi = allp.min(axis=0) - 2.0, allp.max(axis=0) + 2.0
    g = ProbabilisticVoxelGrid(lo, hi, res)
    g.insert_weighted(pts, wts, ctx["origin"], sigma=sig, n_ray_samples=n_ray_samples,
                      splat_k_sigma=k_sigma, splat_occupied=False)
    return g


def panel_voxel_update(ctx, g1, a_world, a_sigma, path, k_sigma=1.5, n_ray=4):
    """Ліворуч тривимірний вигляд, праворуч ПЕРЕРІЗ справжнього поля ΔL площиною,
    що містить промінь до A і вертикаль. Зріз по осі координат тут не годиться:
    промінь іде навскіс і виходить з такого зрізу вже на другому вокселі."""
    o = ctx["origin"]
    occ = np.argwhere(g1.log_odds > 0)
    free = np.argwhere(g1.log_odds < 0)
    c_occ = g1.origin + (occ + 0.5) * g1.resolution
    c_free = g1.origin + (free + 0.5) * g1.resolution
    dl_occ = g1.log_odds[tuple(occ.T)]
    dl_free = g1.log_odds[tuple(free.T)]

    fig = plt.figure(figsize=(16, 10), dpi=100)

    # ---------------------------------------------------------- тривимірний
    ax = fig.add_axes([0.03, 0.50, 0.46, 0.40], projection="3d")
    if len(c_free):
        ax.scatter(c_free[:, 0], c_free[:, 1], c_free[:, 2], c="#5090d0",
                   s=34, alpha=0.75, linewidths=0, depthshade=False,
                   label=f"вільний простір, ΔL = {L_FREE:+.2f} · w ({len(c_free)} вокселів)")
    if len(c_occ):
        s1 = ax.scatter(c_occ[:, 0], c_occ[:, 1], c_occ[:, 2], c=dl_occ, cmap="autumn_r",
                        s=110, linewidths=0.5, edgecolors="black", depthshade=False,
                        label=f"зайнято ({len(c_occ)} вокселів)")
    seg = np.stack([o, a_world])
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#0040ff", lw=3.0, label="промінь до A")
    ax.scatter(*o, color="red", s=130, depthshade=False, label="центр камери")
    box = np.vstack([c_occ, c_free, [o]]) if len(c_free) else np.vstack([c_occ, [o]])
    _view(ax, box)
    ax.legend(fontsize=11, loc="lower left", bbox_to_anchor=(0.60, 0.02))
    ax.set_title("Які вокселі оновив цей кадр", fontsize=17)
    cax = fig.add_axes([0.50, 0.56, 0.012, 0.30])
    cb = fig.colorbar(s1, cax=cax)
    cb.set_label("ΔL зайнятості = 0,85 · w · w_c", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    # ------------------------------------------------- переріз площиною променя
    ax2 = fig.add_axes([0.05, 0.17, 0.90, 0.27])
    d = a_world - o
    L = float(np.linalg.norm(d))
    e_t = d / max(L, 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    e_h = up - np.dot(up, e_t) * e_t
    if np.linalg.norm(e_h) < 1e-6:
        e_h = np.cross(e_t, np.array([1.0, 0.0, 0.0]))
    e_h /= np.linalg.norm(e_h)

    res = g1.resolution
    # ЦЕНТРИ клітинок задаються першими, і серед них обов'язково є h = 0, тобто
    # рядок, що лежить рівно на промені. Якщо задавати спершу межі, центри
    # падають на пів клітинки вбік, промінь іде по стику, і жодна клітинка не
    # збігається з вокселями, які цей промінь насправді оновив.
    tc = np.arange(-3.0 * res, L + 5.0 * res, res)
    hc = np.arange(-4.0 * res, 5.0 * res, res)
    t_edges = np.concatenate([tc - 0.5 * res, [tc[-1] + 0.5 * res]])
    h_edges = np.concatenate([hc - 0.5 * res, [hc[-1] + 0.5 * res]])
    TT, HH = np.meshgrid(tc, hc, indexing="ij")
    world = o[None, None, :] + TT[..., None] * e_t[None, None, :] + HH[..., None] * e_h[None, None, :]
    idx = np.floor((world - g1.origin) / res).astype(int)
    inb = np.all((idx >= 0) & (idx < np.array(g1.dims)), axis=-1)
    plane = np.zeros(TT.shape, dtype=float)
    ii = idx[inb]
    plane[inb] = g1.log_odds[ii[:, 0], ii[:, 1], ii[:, 2]]

    # Клітинки, яких цей кадр не торкнувся, лишаються сірими. Якби вони малювались
    # кольором нуля, то на розбіжній шкалі це був би білий, і слабкі оновлення
    # завбільшки 0,27 злилися б з тлом.
    cm = matplotlib.colormaps["RdBu_r"].copy()
    cm.set_bad("#e8e8e8")
    shown = np.where(np.abs(plane) > 1e-9, plane, np.nan)
    m = ax2.pcolormesh(t_edges, h_edges, shown.T, cmap=cm, vmin=-0.9, vmax=0.9,
                       edgecolors="#a8a8a8", linewidth=0.5)
    nz = np.argwhere(np.abs(plane) > 1e-9)
    for i, j in nz:
        ax2.text(tc[i], hc[j], "+" if plane[i, j] > 0 else "\u2212", ha="center", va="center",
                 fontsize=18, fontweight="bold", color="black", zorder=7)

    ax2.plot([0.0, L], [0.0, 0.0], color="#0040ff", lw=1.8, alpha=0.75,
             label="промінь до A", zorder=5)
    ax2.plot(0.0, 0.0, "o", color="red", ms=15, label="центр камери", zorder=6)
    ax2.plot(L, 0.0, "o", color="#0040ff", ms=15, label="точка p_A", zorder=6)
    t_stop = min(0.92, max(0.0, (L - k_sigma * a_sigma) / max(L, 1e-9))) * L
    ax2.plot([t_stop, L], [0, 0], color="black", lw=7, alpha=0.25, zorder=4)
    ax2.plot(t_stop, 0.0, "s", color="black", ms=13, zorder=6,
             label=f"кінець вільного простору, запас {k_sigma:g}σ = {k_sigma*a_sigma:.1f} м")
    ax2.set_xlim(t_edges[0], t_edges[-1])
    ax2.set_ylim(h_edges[0], h_edges[-1])
    ax2.set_aspect("equal")
    ax2.set_xlabel("відстань уздовж променя від камери, м", fontsize=14)
    ax2.set_ylabel("відхилення\nвід променя, м", fontsize=13)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4)
    ax2.set_title(f"Переріз поля ΔL площиною променя до A, клітинка {res:.0f} м\n"
                  f"плюс це зайнято, мінус це вільно, {n_ray} відліки вільного простору на "
                  f"промінь", fontsize=16, pad=10)
    cax2 = fig.add_axes([0.32, 0.035, 0.36, 0.015])
    cb2 = fig.colorbar(m, cax=cax2, orientation="horizontal")
    cb2.set_label("ΔL цього кадру, логарифм шансів", fontsize=12)
    cb2.ax.tick_params(labelsize=10)

    fig.suptitle("Оновлення воксельної сітки одним кадром", fontsize=22, y=0.97)
    _save(fig, path, crop=True)


# ------------------------------------------------------------------- панель d

def panel_accumulation(history, frame_idx, path, l_max=4.0):
    L = np.asarray(history, dtype=float)
    k = np.arange(len(L))
    changed = np.concatenate([[True], np.abs(np.diff(L)) > 1e-9])
    seen = k[changed & (np.abs(L) > 1e-9)]

    fig, ax = plt.subplots(figsize=PANEL_SIZE, dpi=100)
    ax.step(k, L, where="post", color="#333333", lw=2.0, label="L, логарифм шансів")
    ax.plot(seen, L[seen], "o", color="#0040ff", ms=9,
            label=f"кадри, що бачили цей воксель ({len(seen)})")
    ax.axhline(l_max, color="red", ls="--", lw=2.0, label=f"межа накопичення L = {l_max:g}")
    ax.axhline(0.0, color="gray", ls=":", lw=1.6, label="поріг P = 0,5, тобто L = 0")
    ax.axvline(frame_idx, color="#0040ff", ls=":", lw=1.8,
               label=f"кадр з рисунків, номер {frame_idx}")
    ax.set_xlabel("номер кадру у прогоні", fontsize=15)
    ax.set_ylabel("L, логарифм шансів", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    P = 1.0 / (1.0 + np.exp(-L))
    ax2.step(k, P, where="post", color="#d06000", lw=1.6, alpha=0.85)
    ax2.set_ylabel("P(зайнято)", fontsize=15, color="#d06000")
    ax2.tick_params(labelsize=12, colors="#d06000")
    ax2.set_ylim(0.0, 1.02)

    # Врізка: самі спостереження займають вузьке вікно кадрів, і без збільшення
    # сходинки зливаються в одну вертикаль.
    if len(seen) > 1:
        lo = max(0, int(seen[0]) - 20)
        hi = min(len(L) - 1, int(seen[-1]) + 20)
        axi = ax.inset_axes([0.08, 0.40, 0.46, 0.50])
        axi.step(k[lo:hi + 1], L[lo:hi + 1], where="post", color="#333333", lw=1.8)
        sin = seen[(seen >= lo) & (seen <= hi)]
        axi.plot(sin, L[sin], "o", color="#0040ff", ms=7)
        axi.axhline(l_max, color="red", ls="--", lw=1.4)
        axi.axvline(frame_idx, color="#0040ff", ls=":", lw=1.4)
        axi.set_xlim(lo, hi)
        axi.grid(alpha=0.3)
        axi.tick_params(labelsize=10)
        axi.set_title(f"збільшено: кадри від {lo} до {hi}", fontsize=12)
        ax.indicate_inset_zoom(axi, edgecolor="#808080")

    ax.legend(fontsize=13, loc="lower right")
    ax.set_title(f"Накопичення свідчень у вокселі, що містить p_A\n"
                 f"кінцеве L = {L[-1]:.2f}, тобто P = {P[-1]:.3f}", fontsize=19)
    _save(fig, path, crop=True)


# ------------------------------------------------------------------- панель e

def panel_local_map(adaptive, gt, a_world, path, half_m=35.0):
    def frag(g):
        if g is None:
            return None, None
        idx = np.argwhere(g.log_odds > 0.0)
        if len(idx) == 0:
            return None, None
        c = g.origin + (idx + 0.5) * g.resolution
        near = np.all(np.abs(c - a_world) <= half_m, axis=1)
        return c[near], 1.0 / (1.0 + np.exp(-g.log_odds[tuple(idx[near].T)]))

    pa, prob = frag(adaptive)
    pg, _ = frag(gt)
    fig = plt.figure(figsize=PANEL_SIZE, dpi=100)
    box = np.vstack([x for x in (pa, pg) if x is not None and len(x)] + [a_world[None, :]])

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    if pa is not None and len(pa):
        sc = ax.scatter(pa[:, 0], pa[:, 1], pa[:, 2], c=prob, cmap="inferno",
                        vmin=0.5, vmax=1.0, s=9, linewidths=0, depthshade=False)
        _cb(fig, sc, ax, "P(зайнято)")
    ax.scatter(*a_world, color="#0040ff", s=160, depthshade=False, label="p_A")
    _view(ax, box)
    ax.legend(fontsize=12, loc="upper left")
    ax.set_title("Накопичена ймовірнісна карта, P > 0,5", fontsize=17)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if pg is not None and len(pg):
        ax2.scatter(pg[:, 0], pg[:, 1], pg[:, 2], color="#808080", s=9, linewidths=0,
                    depthshade=False)
    ax2.scatter(*a_world, color="#0040ff", s=160, depthshade=False, label="p_A")
    _view(ax2, box)
    ax2.legend(fontsize=12, loc="upper left")
    ax2.set_title("Еталонна карта того самого фрагмента", fontsize=17)

    fig.suptitle(f"Фрагмент карти навколо p_A, сторона {2*half_m:.0f} м, воксель "
                 f"{adaptive.resolution:.0f} м", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path, crop=True)


# ------------------------------------------------------------- збірний рисунок

def _pad_to_aspect(img, aspect):
    """Доповнює зображення білим до заданого відношення сторін, щоб усі панелі
    збірного рисунка мали однакову геометрію і ряд виглядав рівним."""
    w, h = img.size
    tw, th = (w, int(round(w / aspect))) if w / h > aspect else (int(round(h * aspect)), h)
    out = Image.new("RGB", (max(tw, w), max(th, h)), "white")
    out.paste(img, ((max(tw, w) - w) // 2, (max(th, h) - h) // 2))
    return out


def combined(panel_paths, captions, path, with_captions=True):
    import textwrap
    imgs = [Image.open(p).convert("RGB") for p in panel_paths]
    # Майже квадратна спільна геометрія. Чотири панелі в один ряд на полотні
    # 16 на 9 при співвідношенні 4 на 3 займали б лише третину висоти, і нижня
    # половина лишалась би порожньою.
    aspect = 1.0
    imgs = [_pad_to_aspect(im, aspect) for im in imgs]

    fig_w, fig_h = 32.0, 18.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    n = len(imgs)
    slot_w = 0.215
    gap = (1.0 - n * slot_w) / (n + 1)
    panel_h_frac = (slot_w * fig_w / aspect) / fig_h
    y0 = 0.52 - panel_h_frac / 2.0

    for i, im in enumerate(imgs):
        x0 = gap + i * (slot_w + gap)
        ax = fig.add_axes([x0, y0, slot_w, panel_h_frac])
        ax.imshow(np.asarray(im))
        ax.axis("off")
        fig.text(x0 + slot_w / 2, y0 + panel_h_frac + 0.025, f"{i + 1}. {STEP_TITLES[i]}",
                 ha="center", va="bottom", fontsize=31, fontweight="bold")
        if with_captions:
            fig.text(x0 + slot_w / 2, y0 - 0.025, "\n".join(textwrap.wrap(captions[i], 42)),
                     ha="center", va="top", fontsize=24)
        if i < n - 1:
            # Довжина стрілки не може перевищити проміжок, інакше вона
            # розвертається назад
            dx = max(0.008, gap - 0.012)
            fig.patches.append(FancyArrow(
                x0 + slot_w + (gap - dx) / 2.0, y0 + panel_h_frac / 2.0, dx, 0.0,
                width=0.005, head_width=0.020, head_length=min(0.008, dx * 0.5),
                length_includes_head=True,
                transform=fig.transFigure, color="#404040", figure=fig))
    fig.suptitle("Від двовимірної карти довіри одного кадру до тривимірної воксельної карти",
                 fontsize=42, y=0.95)
    _save(fig, path, tight=False)


# ----------------------------------------------------------------- інтерактив

def html_backprojection(ctx, cloud, weights, ray_pts, a_world, path, max_points=60000):
    o = ctx["origin"]
    pts, wts = cloud, weights
    if len(pts) > max_points:
        k = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, wts = pts[k], wts[k]
    tr = [go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                       marker=dict(size=1.6, color=wts, colorscale="Viridis", cmin=0, cmax=1,
                                   colorbar=dict(title="вага довіри w")),
                       name="точки кадру")]
    for p in ray_pts:
        tr.append(go.Scatter3d(x=[o[0], p[0]], y=[o[1], p[1]], z=[o[2], p[2]], mode="lines",
                               line=dict(color="gray", width=2), showlegend=False,
                               hoverinfo="skip"))
    tr.append(go.Scatter3d(x=[o[0], a_world[0]], y=[o[1], a_world[1]], z=[o[2], a_world[2]],
                           mode="lines+markers", line=dict(color="#0040ff", width=7),
                           marker=dict(size=4, color="#0040ff"), name="промінь до A"))
    tr.append(go.Scatter3d(x=[o[0]], y=[o[1]], z=[o[2]], mode="markers",
                           marker=dict(size=7, color="red"), name="центр камери"))
    fig = go.Figure(tr)
    fig.update_layout(title="Зворотна проєкція кадру в простір",
                      scene=dict(aspectmode="data", xaxis_title="X, м", yaxis_title="Y, м",
                                 zaxis_title="Z, м"),
                      margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html(path)
    print(f"  {os.path.basename(path)}")


def html_voxel_update(ctx, g1, a_world, path):
    o = ctx["origin"]
    occ = np.argwhere(g1.log_odds > 0)
    free = np.argwhere(g1.log_odds < 0)
    tr = []
    if len(free):
        c = g1.origin + (free + 0.5) * g1.resolution
        tr.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
                               marker=dict(size=2.2, color=g1.log_odds[tuple(free.T)],
                                           colorscale="Blues_r", opacity=0.45),
                               name="вільний простір, ΔL < 0"))
    if len(occ):
        c = g1.origin + (occ + 0.5) * g1.resolution
        tr.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
                               marker=dict(size=5, color=g1.log_odds[tuple(occ.T)],
                                           colorscale="Hot", reversescale=True,
                                           colorbar=dict(title="ΔL зайнятості")),
                               name="зайнято, ΔL > 0"))
    tr.append(go.Scatter3d(x=[o[0], a_world[0]], y=[o[1], a_world[1]], z=[o[2], a_world[2]],
                           mode="lines", line=dict(color="#0040ff", width=7), name="промінь до A"))
    fig = go.Figure(tr)
    fig.update_layout(title="Оновлення вокселів одним кадром, воксель 1 м",
                      scene=dict(aspectmode="data", xaxis_title="X, м", yaxis_title="Y, м",
                                 zaxis_title="Z, м"),
                      margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html(path)
    print(f"  {os.path.basename(path)}")


# ---------------------------------------------------------------------- прогін

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=ROOT_EUROC)
    ap.add_argument("--vslam_dir", default="../results/euroc/dpvo")
    ap.add_argument("--fde_dir", default="../results/euroc/zoedepth")
    ap.add_argument("--refiner", default="../refiners/loo_euroc_zoedepth.npz")
    ap.add_argument("--frame", default=FRAME_EUROC)
    ap.add_argument("--out_dir", default="../presentation_frames/euroc/2d_to_3d")
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--n_frames", type=int, default=600)
    ap.add_argument("--sky_cue", action="store_true", default=True)
    ap.add_argument("--from_cache", action="store_true",
                     help="перемалювати рисунки з _cache.npz попереднього прогону, не запускаючи "
                          "конвеєр заново. Самі величини при цьому ті самі, бо взяті з того прогону")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    argv = ["--mode", "dataset",
            "--dataset_root", args.dataset_root,
            "--vslam_dir", args.vslam_dir,
            "--fde_dir", args.fde_dir,
            "--depth_refiner", args.refiner,
            "--output_dir", os.path.join(args.out_dir, "_grid"),
            "--voxel_resolution", str(args.voxel),
            "--n_frames", str(args.n_frames),
            "--dump_frame_name", args.frame,
            "--gt_compare"]
    if args.sky_cue:
        argv.append("--sky_cue")
    run_args = om.build_arg_parser().parse_args(argv)

    cache_path = os.path.join(args.out_dir, "_cache.npz")
    state = {"ctx": None, "idx": None, "history": [], "voxel": None, "grids": None}

    def on_frame(ctx):
        if ctx["fname"] == args.frame:
            # знімок, бо масиви далі перевикористовуються
            state["ctx"] = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                            for k, v in ctx.items()}
            state["idx"] = ctx["index"]

    def on_grid_frame(name, i, g):
        if name != "adaptive" or state["ctx"] is None:
            return
        if state["voxel"] is None:
            if state.get("a_uv") is None:
                # Цикл по кадрах уже завершено, тобто вага довіри потрібного кадру
                # відома, а накопичення ще не почалось. Саме тут і можна обрати
                # пікселі, не ганяючи конвеєр удруге.
                gp = find_depth_file(os.path.join(args.dataset_root, "depth0", "data"),
                                      args.frame)
                raw = load_depth_frame(gp)
                state["gt_depth"] = np.where(raw >= 65535, np.nan, raw) * 0.001
                a, b, ex = choose_pixels(state["ctx"], state["gt_depth"])
                state["a_uv"], state["b_uv"], state["extra"] = a, b, ex
            a_uv = state["a_uv"]
            z = float(state["ctx"]["depth_calibrated"][a_uv[1], a_uv[0]])
            p, _ = _pixel_to_world(a_uv[0], a_uv[1], z, state["ctx"]["K"], state["ctx"]["T_wc"])
            state["p_a"] = p
            state["voxel"] = np.clip(np.floor((p - g.origin) / g.resolution).astype(int),
                                      0, np.array(g.dims) - 1)
        state["history"].append(float(g.log_odds[tuple(state["voxel"])]))

    def on_grids(d):
        state["grids"] = d

    if args.from_cache and os.path.exists(cache_path):
        print(f"Беру стан з {cache_path}")
        z = np.load(cache_path, allow_pickle=True)
        ctx = {k[4:]: z[k] for k in z.files if k.startswith("ctx_")}
        ctx["fname"] = str(ctx["fname"]); ctx["min_depth"] = float(ctx["min_depth"])
        ctx["max_depth"] = float(ctx["max_depth"]); ctx["sigma_pose"] = float(ctx["sigma_pose"])
        for k in ("blades", "refine_keep"):
            ctx.setdefault(k, None)
        a_uv = tuple(int(x) for x in z["a_uv"]); b_uv = tuple(int(x) for x in z["b_uv"])
        extra = [tuple(int(x) for x in p) for p in z["extra"]]
        hist = z["history"]
        gt_depth = z["gt_depth"]
        adaptive = ProbabilisticVoxelGrid(z["ad_origin"], z["ad_origin"] + np.array(z["ad_dims"]) * float(z["ad_res"]), float(z["ad_res"]))
        adaptive.log_odds = z["ad_log"]
        gt_grid = None
        if "gt_log" in z.files:
            gt_grid = ProbabilisticVoxelGrid(z["ad_origin"], z["ad_origin"] + np.array(z["ad_dims"]) * float(z["ad_res"]), float(z["ad_res"]))
            gt_grid.log_odds = z["gt_log"]
        state["idx"] = int(z["idx"])
        if "n_frames" in z.files:
            args.n_frames = int(z["n_frames"])
        S_cached = float(z["residual_scale"])
        ctx["local_calib"] = type("C", (), {"residual_scale": type("R", (), {"value": S_cached})()})()
        return finish(args, run_args, ctx, gt_depth, a_uv, b_uv, extra, hist, adaptive,
                      gt_grid, state["idx"], cache_path, already_cached=True)

    print(f"Прогін конвеєра, {args.n_frames} кадрів")
    om.run_dataset_mode(run_args, hooks={"frame": on_frame, "grid_frame": on_grid_frame,
                                          "grids": on_grids})
    if state["ctx"] is None:
        raise SystemExit(f"кадр {args.frame} не оброблено")
    ctx = state["ctx"]
    a_uv, b_uv, extra = state["a_uv"], state["b_uv"], state["extra"]
    gt_depth = state["gt_depth"]
    print(f"  A = {a_uv}, B = {b_uv}, додаткових пікселів {len(extra)}")
    grids = state["grids"]
    adaptive, gt_grid = grids["adaptive"], grids["gt"]
    hist = np.asarray(state["history"], dtype=float)
    np.savez_compressed(
        cache_path, a_uv=np.array(a_uv), b_uv=np.array(b_uv),
        extra=np.array(extra), history=hist, gt_depth=gt_depth, idx=state["idx"],
        n_frames=int(args.n_frames),
        residual_scale=float(ctx["local_calib"].residual_scale.value),
        ad_origin=adaptive.origin, ad_dims=np.array(adaptive.dims), ad_res=adaptive.resolution,
        ad_log=adaptive.log_odds,
        **({"gt_log": gt_grid.log_odds} if gt_grid is not None else {}),
        **{f"ctx_{k}": v for k, v in ctx.items()
           if isinstance(v, (np.ndarray, int, float, str))})
    return finish(args, run_args, ctx, gt_depth, a_uv, b_uv, extra, hist, adaptive,
                  gt_grid, state["idx"], cache_path)


def finish(args, run_args, ctx, gt_depth, a_uv, b_uv, extra, hist, adaptive, gt_grid,
           frame_idx, cache_path, already_cached=False):
    # --- величини пікселя A і хмара цього кадру, тією самою функцією ---
    z_a = float(ctx["depth_calibrated"][a_uv[1], a_uv[0]])
    a_world, _ = _pixel_to_world(a_uv[0], a_uv[1], z_a, ctx["K"], ctx["T_wc"])
    a_sigma = float(ctx["sigma_total"][a_uv[1], a_uv[0]])
    cloud, weights, sig = backproject_with_weight_map(
        ctx["depth_calibrated"], ctx["confidence"], ctx["K"], ctx["T_wc"],
        ctx["min_depth"], ctx["max_depth"], max_points=None, extra_maps=(ctx["sigma_total"],))

    ray_pts = []
    for (u, v) in extra:
        z = float(ctx["depth_calibrated"][v, u])
        p, _ = _pixel_to_world(u, v, z, ctx["K"], ctx["T_wc"])
        ray_pts.append(p)
    ray_pts = np.array(ray_pts) if ray_pts else np.zeros((0, 3))

    sel = np.vstack([ray_pts, a_world[None, :]])
    sel_w = np.array([float(ctx["confidence"][v, u]) for (u, v) in extra]
                     + [float(ctx["confidence"][a_uv[1], a_uv[0]])])
    sel_s = np.array([float(ctx["sigma_total"][v, u]) for (u, v) in extra] + [a_sigma])
    g1 = one_frame_grid(ctx, sel, sel_w, sel_s, args.voxel,
                        run_args.n_ray_samples, run_args.splat_k_sigma)

    dl_actual = None
    if frame_idx is not None and 0 < len(hist) > frame_idx:
        prev = hist[frame_idx - 1] if frame_idx > 0 else 0.0
        dl_actual = hist[frame_idx] - prev
    n_seen = int((np.abs(np.diff(np.concatenate([[0.0], hist]))) > 1e-9).sum())

    trace = build_trace(ctx, gt_depth, a_uv, b_uv, adaptive.resolution, adaptive.origin,
                        dl_actual, n_seen)
    trace["кадрів_у_прогоні"] = int(args.n_frames)
    trace["номер_кадру_у_прогоні"] = int(frame_idx)
    trace["кінцеве_L_вокселя"] = round(float(hist[-1]), 4) if len(hist) else None
    trace["кінцеве_P_вокселя"] = round(float(1 / (1 + np.exp(-hist[-1]))), 4) if len(hist) else None
    with open(os.path.join(args.out_dir, "trace.json"), "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    print(f"  trace.json")

    print("Рисунки:")
    p1 = os.path.join(args.out_dir, "01_prob2d.png")
    p2 = os.path.join(args.out_dir, "02_backprojection.png")
    p3 = os.path.join(args.out_dir, "03_voxel_update.png")
    p4 = os.path.join(args.out_dir, "04_accumulation.png")
    p5 = os.path.join(args.out_dir, "05_local_map.png")
    for fn in tqdm([lambda: panel_prob2d(ctx, a_uv, b_uv, extra, p1),
                    lambda: panel_backprojection(ctx, cloud, weights, ray_pts, a_world, p2),
                    lambda: panel_voxel_update(ctx, g1, a_world, a_sigma, p3,
                                                run_args.splat_k_sigma, run_args.n_ray_samples),
                    lambda: panel_accumulation(hist, frame_idx, p4),
                    lambda: panel_local_map(adaptive, gt_grid, a_world, p5)],
                   desc="панелі", leave=False):
        fn()

    A = trace["A"]
    captions = [
        f"піксель A: u = {A['u']}, v = {A['v']}, вага w = {A['вага_w']:.3f}",
        f"z' = {A['z_штрих_м']:.2f} м, p = ({a_world[0]:.1f}, {a_world[1]:.1f}, "
        f"{a_world[2]:.1f}) м",
        f"внесок кадру ΔL = +{A['ΔL_occ_внесок_цього_кадру']:.3f}, "
        f"запас вільного простору 1,5σ = {1.5*a_sigma:.1f} м",
        f"після {n_seen} кадрів, що бачили воксель, P = {trace['кінцеве_P_вокселя']:.3f}",
    ]
    combined([p1, p2, p3, p4], captions,
             os.path.join(args.out_dir, "00_pipeline_2d_to_3d.png"), True)
    combined([p1, p2, p3, p4], captions,
             os.path.join(args.out_dir, "00_pipeline_2d_to_3d_clean.png"), False)

    html_backprojection(ctx, cloud, weights, ray_pts, a_world,
                        os.path.join(args.out_dir, "02_backprojection.html"))
    html_voxel_update(ctx, g1, a_world, os.path.join(args.out_dir, "03_voxel_update.html"))

    print(f"\nГотово: {os.path.abspath(args.out_dir)}")
    print(json.dumps(trace["A"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
