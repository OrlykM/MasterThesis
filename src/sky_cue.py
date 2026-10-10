r"""
Відкидання неба за самим зображенням, без еталона і без навченої таблиці.

Навіщо. Таблиця з `depth_refine.py` вирішує, чи є в пікселі поверхня, за двома
ознаками, за інтервалом глибини і за номером рядка кадру. На рядку горизонту
крона дерева і небо потрапляють в ОДНУ комірку, тому таблиця мусить дати їм
однакову відповідь. Виміряно на наборі №1: у смузі рядків від 160 до 240 справжня
поверхня є у 53 відсотках пікселів, валідність комірок там 0,36, поріг 0,50, і
смуга відрізається цілком разом з деревами. У карту потрапляє 20 відсотків тієї
смуги, тобто дві третини перешкод на горизонті губляться, тоді як земля під
апаратом проходить на 100 відсотків. Для шару безпеки це перевернуто.

Зображення розрізняє їх тривіально. Небо яскраве і без текстури, крона темна і з
сильним градієнтом, і ця різниця лишається в тому самому рядку і на тій самій
глибині.

Правило, без жодного порога, підібраного під набір:

  1. Поріг яскравості береться методом Оцу з гістограми САМОГО кадру.
  2. Поріг градієнта це перцентиль градієнта всередині яскравої частини кадру,
     тобто теж величина цього кадру, а не константа.
  3. З кандидатів лишаються тільки ті зв'язні області, що торкаються верхнього
     краю кадру. Небо завжди звідти і починається, а яскрава пласка дорога внизу
     кадру цієї умови не проходить.
  4. Морфологічне замикання заповнює дрібні дірки, тобто тонкі гілки на тлі неба.

Запобіжники. Якщо область виходить більшою за `max_fraction` кадру або верхній
рядок укритий менш ніж на `min_top_cover`, маска вважається ненадійною і
повертається порожньою. Краще не відкинути нічого, ніж відкинути сцену.

Еталон тут використовується ТІЛЬКИ в режимі перевірки, щоб виміряти якість. Сам
детектор його не бачить.

Використання:
    python3 sky_cue.py eval --datasets euroc run1 run2
    python3 sky_cue.py eval --datasets run3 run4 run5 run6 --frames 40
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_camera_intrinsics, load_image_timestamps
from compare_reconstruction_3d import find_depth_file, load_depth_frame

ROOTS = {"euroc": "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor",
         **{f"run{i}": f"../datasets/run{i}" for i in range(1, 7)}}


def _sobel(grey):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    gx = ndimage.convolve(grey, kx, mode="nearest")
    gy = ndimage.convolve(grey, kx.T, mode="nearest")
    return np.sqrt(gx ** 2 + gy ** 2)


def _otsu(values, bins=256):
    """Поріг Оцу: ділить гістограму так, щоб дисперсія між двома частинами була
    найбільшою. Параметрів не має, тобто підлаштовується під кожен кадр."""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    p = hist / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centres) / np.maximum(w0, 1e-12)
    m1 = (np.sum(p * centres) - np.cumsum(p * centres)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.argmax(between))])


def sky_mask(grey, grad_percentile=55.0, close_px=5, max_fraction=0.85,
             min_top_cover=0.20, top_rows=3, min_fraction=0.01):
    """grey: яскравість кадру від 0 до 1. -> булева маска пікселів неба.

    Порожня маска означає «ознака не спрацювала», а не «неба немає»: викликач
    має трактувати це як відсутність свідчень."""
    grey = np.asarray(grey, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    if grey.max() > 1.5:
        grey = grey / 255.0
    h, w = grey.shape

    grad = _sobel(ndimage.uniform_filter(grey, size=3))
    bright_thr = _otsu(grey.ravel())
    bright = grey > bright_thr
    if bright.sum() < min_fraction * grey.size:
        return np.zeros((h, w), dtype=bool)

    # Поріг градієнта це перцентиль УСЕРЕДИНІ яскравої частини: у кадрі з
    # хмарами небо не ідеально пласке, а в кадрі з чистим небом поріг сам
    # опуститься нижче.
    grad_thr = float(np.percentile(grad[bright], grad_percentile))
    cand = bright & (grad <= grad_thr)

    cand = ndimage.binary_closing(cand, structure=np.ones((close_px, close_px)))
    lab, n = ndimage.label(cand)
    if n == 0:
        return np.zeros((h, w), dtype=bool)

    # Лишаємо лише області, що торкаються верхнього краю. Небо починається
    # звідти, а яскрава доріжка внизу кадру цієї умови не проходить.
    touching = np.unique(lab[:top_rows, :])
    touching = touching[touching > 0]
    if touching.size == 0:
        return np.zeros((h, w), dtype=bool)
    mask = np.isin(lab, touching)
    mask = ndimage.binary_fill_holes(mask)

    frac = mask.mean()
    top_cover = mask[:top_rows, :].mean()
    if frac > max_fraction or top_cover < min_top_cover or frac < min_fraction:
        return np.zeros((h, w), dtype=bool)
    return mask


def sky_mask_from_file(path, **kw):
    return sky_mask(np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0, **kw)


def structure_mask(grey, grad_percentile=70.0, dark_margin=0.0):
    """Протилежний бік тієї самої ознаки: пікселі, які НЕ МОЖУТЬ бути небом.

    Небо яскраве і без текстури, тому темний або текстурований піксель небом не
    є, хай що каже таблиця валідності. Ця маска існує саме для того, щоб
    скасовувати помилки таблиці: та індексована лише рядком і глибиною, тому в
    смузі горизонту вбиває крони дерев разом з небом.

    Пороги знову з самого кадру, Оцу для яскравості і перцентиль для градієнта."""
    grey = np.asarray(grey, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    if grey.max() > 1.5:
        grey = grey / 255.0
    grad = _sobel(ndimage.uniform_filter(grey, size=3))
    thr = _otsu(grey.ravel())
    dark = grey <= (thr - dark_margin)
    textured = grad >= float(np.percentile(grad, grad_percentile))
    return dark | textured


# ------------------------------------------------------------------ перевірка

def evaluate_veto(args):
    """Наскільки добре «це точно не небо» відрізняє поверхню від неба."""
    print(f"{'набір':8s} {'кадрів':>7s} {'небо є':>8s} "
          f"{'поверхні врятовано':>19s} {'неба хибно врятовано':>21s}")
    for ds in args.datasets:
        root = ROOTS[ds]
        frames = load_image_timestamps(os.path.join(root, "cam0", "data.csv"))
        intr = load_camera_intrinsics(os.path.join(root, "cam0", "sensor.yaml"))
        step = max(1, len(frames) // args.frames)
        hit_s = n_s = hit_k = n_k = used = 0
        for _t, fname in tqdm(frames[::step][:args.frames], desc=ds, leave=False):
            gp = find_depth_file(os.path.join(root, "depth0", "data"), fname)
            ip = os.path.join(root, "cam0", "data", fname)
            if gp is None or not os.path.exists(ip):
                continue
            raw = load_depth_frame(gp)
            if raw.shape != (intr["height"], intr["width"]):
                continue
            gt_sky = raw >= 65535
            gt_surface = (~gt_sky) & (raw > 100) & (raw < 50000)
            g = np.array(Image.open(ip).convert("L"), dtype=np.float32) / 255.0
            st = structure_mask(g, grad_percentile=args.grad_percentile)
            hit_s += int((st & gt_surface).sum()); n_s += int(gt_surface.sum())
            hit_k += int((st & gt_sky).sum());     n_k += int(gt_sky.sum())
            used += 1
        if used:
            print(f"{ds:8s} {used:7d} {100*n_k/max(n_k+n_s,1):7.1f}% "
                  f"{100*hit_s/max(n_s,1):18.1f}% {100*hit_k/max(n_k,1):20.1f}%")


def evaluate(args):
    print(f"{'набір':8s} {'кадрів':>7s} {'небо є':>8s} {'знайдено':>9s} "
          f"{'повнота':>8s} {'точність':>9s} {'ВТРАЧЕНО поверхні':>18s}")
    totals = {}
    for ds in args.datasets:
        root = ROOTS[ds]
        frames = load_image_timestamps(os.path.join(root, "cam0", "data.csv"))
        intr = load_camera_intrinsics(os.path.join(root, "cam0", "sensor.yaml"))
        step = max(1, len(frames) // args.frames)
        tp = fp = fn = lost = surf = n_px = n_sky = 0
        used = 0
        for _t, fname in tqdm(frames[::step][:args.frames], desc=ds, leave=False):
            gp = find_depth_file(os.path.join(root, "depth0", "data"), fname)
            ip = os.path.join(root, "cam0", "data", fname)
            if gp is None or not os.path.exists(ip):
                continue
            raw = load_depth_frame(gp)
            if raw.shape != (intr["height"], intr["width"]):
                continue
            gt_sky = raw >= 65535
            gt_surface = (~gt_sky) & (raw > 100) & (raw < 50000)

            m = sky_mask_from_file(ip, grad_percentile=args.grad_percentile,
                                    close_px=args.close_px)
            tp += int((m & gt_sky).sum())
            fp += int((m & ~gt_sky).sum())
            fn += int((~m & gt_sky).sum())
            lost += int((m & gt_surface).sum())
            surf += int(gt_surface.sum())
            n_sky += int(gt_sky.sum())
            n_px += gt_sky.size
            used += 1
        if used == 0:
            continue
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        totals[ds] = (rec, prec, lost / max(surf, 1))
        print(f"{ds:8s} {used:7d} {100*n_sky/max(n_px,1):7.1f}% "
              f"{100*(tp+fp)/max(n_px,1):8.1f}% {100*rec:7.1f}% {100*prec:8.1f}% "
              f"{100*lost/max(surf,1):17.2f}%")
    if totals:
        r = np.mean([v[0] for v in totals.values()])
        p = np.mean([v[1] for v in totals.values()])
        l = np.mean([v[2] for v in totals.values()])
        print(f"\nсередні: повнота {100*r:.1f}%, точність {100*p:.1f}%, "
              f"втрачено поверхні {100*l:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="eval", choices=["eval", "veto"])
    ap.add_argument("--datasets", nargs="+", default=["euroc", "run1", "run2"])
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--grad_percentile", type=float, default=55.0)
    ap.add_argument("--close_px", type=int, default=5)
    args = ap.parse_args()
    (evaluate_veto if args.mode == "veto" else evaluate)(args)


if __name__ == "__main__":
    main()
