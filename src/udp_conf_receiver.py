r"""
Споживач карти довіри: слухає адресу, на яку `occupancy_3d_map.py --mode udp`
віддає результат, і показує, що саме прилітає.

Це приклад того місця, куди під'єднується автопілот чи шар безпеки. Замість
запису на диск постав сюди свою логіку: `--print_only` нічого не пише і лише
друкує, скільки отримано і з якою затримкою.

Три типи повідомлень:
    conf2d      покадрова 2D карта довіри по пікселях, нечисло означає
                відкинутий піксель
    grid_delta  лише ті вокселі, які змінив цей кадр, тобто індекси і нові
                значення логарифма шансів
    grid        уся сітка, якщо сервер її шле

Воксельна карта тут збирається з приростів: клас `GridAccumulator` тримає
сітку і застосовує до неї кожен приріст. Повна сітка, якщо прилетить,
просто заміняє накопичену.

Використання:
    python3 udp_conf_receiver.py --port 9001 --output_dir ../live
    python3 udp_conf_receiver.py --port 9001 --print_only
    python3 udp_conf_receiver.py --port 9001 --output_dir ../live --save_png_every 10
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from udp_streaming import MapSubscriber

L_OCC = 0.85


class GridAccumulator:
    """Збирає воксельну карту з приростів. Сітка виділяється за першим
    повідомленням, бо саме воно несе межі і розмір вокселя."""

    def __init__(self):
        self.log_odds = None
        self.origin = None
        self.resolution = None

    def apply_delta(self, msg):
        if self.log_odds is None:
            self.log_odds = np.zeros(msg["dims"], dtype=np.float32)
            self.origin = msg["origin"]
            self.resolution = msg["resolution"]
        elif self.log_odds.shape != tuple(msg["dims"]):
            # сервер перезапустився з іншими межами, починаємо заново
            self.log_odds = np.zeros(msg["dims"], dtype=np.float32)
            self.origin = msg["origin"]
            self.resolution = msg["resolution"]
        if len(msg["idx"]):
            i = msg["idx"]
            self.log_odds[i[:, 0], i[:, 1], i[:, 2]] = msg["values"]

    def apply_full(self, msg):
        self.log_odds = msg["log_odds"].astype(np.float32)
        self.origin = msg["origin"]
        self.resolution = msg["resolution"]

    def occupied(self, threshold=0.0):
        return 0 if self.log_odds is None else int((self.log_odds > threshold).sum())

    def save(self, path):
        if self.log_odds is None:
            return
        np.savez_compressed(path, log_odds=self.log_odds, origin=self.origin,
                            resolution=self.resolution)


def save_conf_png(conf, path, timestamp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    usable = np.isfinite(conf)
    prob = np.where(usable, 1.0 / (1.0 + np.exp(-L_OCC * np.nan_to_num(conf))), np.nan)
    p_max = 1.0 / (1.0 + np.exp(-L_OCC))
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    cm = matplotlib.colormaps["RdYlGn_r"].copy()
    cm.set_bad("white")
    im = ax.imshow(prob, cmap=cm, vmin=0.5, vmax=p_max, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Карта довіри з потоку, мітка часу {timestamp:.3f}\n"
                 f"{100*usable.mean():.0f}% пікселів у карту, біле = відкинуто", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(f"P(зайнято) з одного кадру, максимум {p_max:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--output_dir", help="куди писати. Без нього нічого не пишеться")
    ap.add_argument("--print_only", action="store_true")
    ap.add_argument("--save_png_every", type=int, default=0,
                     help="зберігати картинку карти довіри кожні N кадрів")
    ap.add_argument("--save_grid_every", type=int, default=25,
                     help="зберігати накопичену воксельну карту кожні N приростів")
    ap.add_argument("--idle_timeout", type=float, default=None,
                     help="вийти, якщо нічого не приходить стільки секунд")
    ap.add_argument("--max_messages", type=int, default=None)
    args = ap.parse_args()

    out = args.output_dir if (args.output_dir and not args.print_only) else None
    if out:
        os.makedirs(out, exist_ok=True)

    sub = MapSubscriber(args.host, args.port)
    acc = GridAccumulator()
    print(f"[conf_receiver] слухаю {args.host}:{args.port}"
          + (f", пишу в {out}" if out else ", нічого не пишу"))

    n_conf = n_delta = n_grid = 0
    t_first = None
    try:
        for msg in sub.messages(idle_timeout=args.idle_timeout):
            if t_first is None:
                t_first = time.monotonic()
            if msg["type"] == "conf2d":
                n_conf += 1
                conf = msg["confidence"]
                usable = np.isfinite(conf)
                print(f"[conf2d] кадр {msg['frame_id']}: {conf.shape[1]}x{conf.shape[0]}, "
                      f"у карту {100*usable.mean():5.1f}% пікселів, "
                      f"середня вага {np.nanmean(np.where(usable, conf, np.nan)):.3f}")
                if out:
                    np.save(os.path.join(out, "conf_latest.npy"), conf)
                    if args.save_png_every and n_conf % args.save_png_every == 0:
                        save_conf_png(conf, os.path.join(out, f"conf_{msg['frame_id']:05d}.png"),
                                       msg["timestamp"])
            elif msg["type"] == "grid_delta":
                n_delta += 1
                acc.apply_delta(msg)
                print(f"[delta]  кадр {msg['frame_id']}: змінено {len(msg['idx'])} вокселів, "
                      f"зайнятих усього {acc.occupied()}")
                if out and args.save_grid_every and n_delta % args.save_grid_every == 0:
                    acc.save(os.path.join(out, "adaptive_live.npz"))
            elif msg["type"] == "grid":
                n_grid += 1
                acc.apply_full(msg)
                print(f"[grid]   кадр {msg['frame_id']}: повна сітка "
                      f"{acc.log_odds.shape}, зайнятих {acc.occupied()}")
                if out:
                    acc.save(os.path.join(out, "adaptive_live.npz"))

            if args.max_messages and (n_conf + n_delta + n_grid) >= args.max_messages:
                break
    except KeyboardInterrupt:
        print("\n[conf_receiver] зупинено (Ctrl+C)")
    finally:
        sub.close()

    dt = (time.monotonic() - t_first) if t_first else 0.0
    print(f"\n[conf_receiver] отримано: conf2d {n_conf}, приростів {n_delta}, "
          f"повних сіток {n_grid}, за {dt:.1f} с")
    if acc.log_odds is not None:
        print(f"[conf_receiver] накопичена карта: {acc.log_odds.shape}, "
              f"зайнятих вокселів {acc.occupied()}")
        if out:
            acc.save(os.path.join(out, "adaptive_live.npz"))
            print(f"[conf_receiver] збережено {out}/adaptive_live.npz")


if __name__ == "__main__":
    main()
