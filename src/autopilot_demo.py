r"""
Автопілот на справжніх кадрах: проганяє конвеєр і малює, що він вирішив.

Виходить:
    frame_NNNN.png   кадр, карта довіри, віяло кандидатів, обраний напрямок
    summary.png      швидкість, азимут і запас по кадрах
    commands.csv     усі команди числами

Використання:
    python3 autopilot_demo.py --n_frames 40
    python3 autopilot_demo.py --n_frames 40 --unknown_is_risk
    python3 autopilot_demo.py --n_frames 40 --compare
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autopilot import ReactiveAutopilot, L_OCC
from realtime_map import RealtimeConfidenceMapper
from udp_replay_sender import load_stream

ROOT = "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor"


def draw_frame(path, image, conf, depth, cmd, ap, K, title):
    """Кадр і карта довіри поруч, зверху віяло кандидатів і обраний курс."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=100)

    if image is not None:
        axes[0].imshow(image, cmap="gray")
    axes[0].set_title("кадр камери", fontsize=12)

    cm = matplotlib.colormaps["RdYlGn_r"].copy()
    cm.set_bad("white")
    usable = np.isfinite(conf)
    prob = np.where(usable, 1.0 / (1.0 + np.exp(-L_OCC * np.nan_to_num(conf))), np.nan)
    im = axes[1].imshow(prob, cmap=cm, vmin=0.5, vmax=1 / (1 + np.exp(-L_OCC)),
                        interpolation="nearest")
    axes[1].set_title("карта довіри і вибір напрямку", fontsize=12)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("P(зайнято) з кадру", fontsize=9)

    # віяло кандидатів у пікселях: куди дивиться кожен напрямок
    h, w = conf.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = ap.cand
    ok = d[:, 2] > 1e-6
    uu = fx * d[ok, 0] / d[ok, 2] + cx
    vv = fy * d[ok, 1] / d[ok, 2] + cy
    clear = cmd["_cand_clearance"][ok]
    for ax in (axes[0], axes[1]):
        sc = ax.scatter(uu, vv, c=clear, cmap="viridis", s=26, vmin=0,
                        vmax=max(1.0, float(np.nanmax(clear))), alpha=0.85,
                        edgecolors="black", linewidths=0.4)
    b = cmd["_best"]
    if d[b, 2] > 1e-6:
        bu = fx * d[b, 0] / d[b, 2] + cx
        bv = fy * d[b, 1] / d[b, 2] + cy
        for ax in (axes[0], axes[1]):
            ax.plot([cx, bu], [cy, bv], color="#0040ff", lw=3.0)
            ax.plot(bu, bv, "o", ms=18, mfc="none", mec="#0040ff", mew=3.5)
    cb2 = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
    cb2.set_label("запас у напрямку, м", fontsize=9)

    for ax in axes:
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_xticks([]); ax.set_yticks([])

    colour = "red" if cmd["stop"] else "black"
    fig.suptitle(f"{title}\n{ap.reason(cmd)}", fontsize=13, color=colour)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def summary_plot(rows, path, label=""):
    k = np.array([r["frame"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=100, sharex=True)
    axes[0].plot(k, [r["speed"] for r in rows], color="#0a7", lw=2)
    axes[0].set_ylabel("швидкість, м/с"); axes[0].grid(alpha=0.3)
    axes[1].plot(k, [r["azimuth_deg"] for r in rows], color="#0040ff", lw=2)
    axes[1].axhline(0, color="gray", ls=":")
    axes[1].set_ylabel("азимут, градуси"); axes[1].grid(alpha=0.3)
    axes[2].plot(k, [r["clearance_m"] for r in rows], color="#c40", lw=2)
    axes[2].set_ylabel("запас попереду, м"); axes[2].set_xlabel("кадр")
    axes[2].grid(alpha=0.3)
    stops = k[np.array([r["stop"] for r in rows], dtype=bool)]
    for ax in axes:
        for s in stops:
            ax.axvspan(s - 0.5, s + 0.5, color="red", alpha=0.18)
    fig.suptitle(f"Команди автопілота по кадрах{label}\n"
                 f"червоним позначені кадри зі стопом", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run(args, unknown_is_risk, out_dir, draw=True):
    os.makedirs(out_dir, exist_ok=True)
    K, stream = load_stream(ROOT, args.vslam_dir, args.fde_dir,
                             n_frames=args.n_frames, verbose=False)
    mapper = None
    ap = ReactiveAutopilot(drone_radius=args.drone_radius, max_speed=args.max_speed,
                            stop_distance=args.stop_distance,
                            unknown_is_risk=unknown_is_risk)
    rows = []
    saved = 0
    for fr in tqdm(stream, desc=os.path.basename(out_dir), leave=False):
        if mapper is None:
            c = fr["T_wc"][:3, 3]
            mapper = RealtimeConfidenceMapper(c - 150, c + 150, 1.0,
                                               refiner_path=args.refiner,
                                               use_sky_cue=True, blade_mask_path=None,
                                               max_depth=50.0, build_naive=False)
        res = mapper.process(fr["depth"], fr["T_wc"], K,
                              image=fr["image"], landmarks=fr["landmarks"])
        if res.get("confidence") is None:
            continue
        cmd = ap.step(res["confidence"], res["depth_calibrated"], K, fr["T_wc"])
        rows.append({"frame": len(rows), "name": fr["name"],
                     "speed": cmd["speed"], "azimuth_deg": cmd["azimuth_deg"],
                     "elevation_deg": cmd["elevation_deg"],
                     "clearance_m": cmd["clearance_m"], "risk": cmd["risk"],
                     "stop": cmd["stop"], "blocked_fraction": cmd["blocked_fraction"],
                     "dir_x": cmd["direction"][0], "dir_y": cmd["direction"][1],
                     "dir_z": cmd["direction"][2],
                     "target_x": cmd["target"][0], "target_y": cmd["target"][1],
                     "target_z": cmd["target"][2]})
        if draw and (len(rows) - 1) % max(1, args.n_frames // args.n_images) == 0 \
                and saved < args.n_images:
            draw_frame(os.path.join(out_dir, f"frame_{len(rows)-1:04d}.png"),
                        fr["image"], res["confidence"], res["depth_calibrated"],
                        cmd, ap, K, f"кадр {len(rows)-1}")
            saved += 1

    if rows:
        with open(os.path.join(out_dir, "commands.csv"), "w", newline="",
                  encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        summary_plot(rows, os.path.join(out_dir, "summary.png"),
                     label=", невідоме = ризик" if unknown_is_risk else "")
    return rows


def report(rows, name):
    if not rows:
        print(f"{name}: жодного кадру"); return
    sp = np.array([r["speed"] for r in rows])
    az = np.array([r["azimuth_deg"] for r in rows])
    cl = np.array([r["clearance_m"] for r in rows])
    st = np.array([r["stop"] for r in rows], dtype=bool)
    print(f"{name:26s} кадрів {len(rows):3d}  сер.швидкість {sp.mean():.2f} м/с  "
          f"стопів {int(st.sum()):3d}  |азимут| сер. {np.abs(az).mean():4.1f}°  "
          f"запас медіана {np.median(cl):5.1f} м")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--vslam_dir", default="../results/euroc/dpvo")
    ap_.add_argument("--fde_dir", default="../results/euroc/zoedepth")
    ap_.add_argument("--refiner", default="../refiners/loo_euroc_zoedepth.npz")
    ap_.add_argument("--out_dir", default="../presentation_frames/euroc/autopilot")
    ap_.add_argument("--n_frames", type=int, default=40)
    ap_.add_argument("--n_images", type=int, default=6)
    ap_.add_argument("--drone_radius", type=float, default=0.5)
    ap_.add_argument("--max_speed", type=float, default=3.0)
    ap_.add_argument("--stop_distance", type=float, default=1.5)
    ap_.add_argument("--unknown_is_risk", action="store_true")
    ap_.add_argument("--compare", action="store_true",
                      help="прогнати обидва режими і порівняти поведінку")
    args = ap_.parse_args()

    if args.compare:
        print("Порівняння: як ставлення до невідомого змінює поведінку\n")
        a = run(args, False, os.path.join(args.out_dir, "optimistic"))
        b = run(args, True, os.path.join(args.out_dir, "cautious"), draw=False)
        report(a, "невідоме = порожньо")
        report(b, "невідоме = ризик")
    else:
        rows = run(args, args.unknown_is_risk, args.out_dir)
        report(rows, "автопілот")
    print(f"\nзбережено у {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
