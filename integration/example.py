r"""
Найкоротший приклад: один кадр туди, карта довіри назад.

Підстав сюди свої масиви замість заглушок і перевір, що повертається. Більше
нічого не треба, решта в README.md.

    python example.py --host 192.168.0.122
    python example.py --host 100.x.y.z --tcp
"""
import argparse
import time

import numpy as np

from client import Client, TcpClient


def my_frame():
    """ЗАМІНИ це своїми даними. Зараз тут проста сцена: стіна за 8 м і підлога."""
    h, w = 480, 640
    fx = fy = 500.0
    cx, cy = w / 2.0, h / 2.0

    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    depth = np.full((h, w), 8.0, dtype=np.float32)          # стіна
    ground = v > cy + 40
    depth[ground] = (1.6 * fy / (v[ground] - cy)).astype(np.float32)   # підлога
    depth = np.clip(depth, 0.3, 50.0)

    grey = np.clip(255 * (1 - depth / 50.0), 0, 255).astype(np.uint8)

    T_wc = np.eye(4)                                        # камера в початку координат
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Розріджені точки VSLAM у СВІТОВИХ координатах. Тут вони зроблені з тієї
    # самої глибини лише для прикладу: у тебе це будуть справжні точки VSLAM.
    rng = np.random.default_rng(0)
    vs = rng.integers(0, h, 300)
    us = rng.integers(0, w, 300)
    z = depth[vs, us].astype(np.float64)
    pts_cam = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z, np.ones_like(z)], axis=1)
    landmarks = (T_wc @ pts_cam.T).T[:, :3].astype(np.float32)

    return depth, T_wc, K, grey, landmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--listen_port", type=int, default=9001)
    ap.add_argument("--tcp", action="store_true")
    args = ap.parse_args()

    c = TcpClient(args.host, args.port) if args.tcp \
        else Client(args.host, args.port, args.listen_port)

    depth, T_wc, K, grey, landmarks = my_frame()
    print(f"шлю кадр {depth.shape[1]}x{depth.shape[0]}, {len(landmarks)} точок VSLAM")
    c.send(0, time.time(), depth, T_wc, K=K, image=grey, landmarks=landmarks)

    for m in c.results(timeout=5.0):
        if m["type"] == "conf2d":
            conf = m["confidence"]
            good = np.isfinite(conf)
            P = 1.0 / (1.0 + np.exp(-0.85 * conf))
            print(f"\nкарта довіри {conf.shape[1]}x{conf.shape[0]}")
            print(f"  у карту йде     {100 * good.mean():.0f}% пікселів")
            print(f"  середня довіра  {np.nanmean(np.where(good, conf, np.nan)):.3f}")
            print(f"  P(зайнято)      від {np.nanmin(P):.3f} до {np.nanmax(P):.3f}")
        elif m["type"] == "grid_delta":
            print(f"\nприріст воксельної карти")
            print(f"  змінено вокселів {len(m['idx'])}")
            print(f"  сітка            {m['dims']}, воксель {m['resolution']} м")
            if len(m["values"]):
                print(f"  логарифм шансів  від {m['values'].min():.2f} "
                      f"до {m['values'].max():.2f}")
    c.close()


if __name__ == "__main__":
    main()
