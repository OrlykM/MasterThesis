r"""
Клієнт: шле кадри на сервер карти довіри і приймає результат назад.

Самодостатній, один файл. Потрібен лише numpy. Копіюй на будь-яку машину і
запускай, репозиторій там не потрібен.

    pip install numpy
    python send_frames_client.py --host 192.168.0.122 --selftest

`--selftest` шле синтетичну сцену, тобто похилу площину з кількома стовпами, і
показує, що повертається. Це перевіряє зв'язок без жодних справжніх даних.

Щоб слати СВОЇ дані, імпортуй звідси `Client` і виклич `send`:

    from send_frames_client import Client
    c = Client("192.168.0.122", 9000, listen_port=9001)
    c.send(frame_id, timestamp, depth, T_wc, K=K, image=grey, landmarks=pts)
    for m in c.results(timeout=2.0):
        print(m["type"], m["frame_id"])

Що саме треба подати:

    depth      (H, W) float32, метри. Нуль або NaN означає «немає значення»
    T_wc       (4, 4) перетворення з камери у світ, метри
    K          (3, 3) матриця камери: fx, fy, cx, cy
    image      (H, W) uint8 яскравість, необов'язково, для вето неба
    landmarks  (N, 3) float32, розріджені точки VSLAM У СВІТОВИХ координатах,
               у тих самих метрах, що й T_wc. БЕЗ НИХ КАРТИ ДОВІРИ НЕ БУДЕ

Пози і точки мають бути в одній системі і в метрах. Сервер VSLAM не запускає і
масштаб не відновлює: на борту це робить інерціальний модуль або висотомір.
"""
import argparse
import socket
import struct
import sys
import time
import zlib

import numpy as np

_TCP_HDR = "!4sBBII"
_TCP_HDR_SIZE = struct.calcsize(_TCP_HDR)

_MAGIC = b"OVM1"
_HDR = "!4sBBIHHI"
_HDR_SIZE = struct.calcsize(_HDR)
MSG_FRAME2, MSG_CONF2D, MSG_GRID_DELTA, MSG_GRID, MSG_COMMAND = 3, 4, 5, 2, 6
_CMD_FMT = "!d 3d 3d d d d B"
F_HAS_K, F_HAS_IMAGE, F_HAS_LANDMARKS = 1, 2, 4
F_HAS_DEPTH, F_HAS_POSE, F_IMAGE_RGB = 8, 16, 32


def _encode_frame2(timestamp, depth=None, T_wc=None, pose_x=0.0, K=None,
                    image=None, landmarks=None):
    flags = 0
    h = w = None
    if depth is not None:
        flags |= F_HAS_DEPTH
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        h, w = depth.shape
    if image is not None:
        flags |= F_HAS_IMAGE
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.ndim == 3:
            flags |= F_IMAGE_RGB
            ih, iw = image.shape[:2]
        else:
            ih, iw = image.shape
        if h is None:
            h, w = ih, iw
        elif (ih, iw) != (h, w):
            raise ValueError(f"зображення {(ih, iw)} не збігається з глибиною {(h, w)}")
    if h is None:
        raise ValueError("кадр має нести або глибину, або зображення")
    if K is not None:
        flags |= F_HAS_K
    if T_wc is not None:
        flags |= F_HAS_POSE
    lm = np.zeros((0, 3), np.float32)
    if landmarks is not None and len(landmarks):
        flags |= F_HAS_LANDMARKS
        lm = np.ascontiguousarray(landmarks, dtype=np.float32).reshape(-1, 3)
    Kf = np.ascontiguousarray(K, np.float64).reshape(9) if K is not None else np.zeros(9)
    Tf = (np.ascontiguousarray(T_wc, np.float64).reshape(16)
          if T_wc is not None else np.eye(4).reshape(16))
    return b"".join([struct.pack("!dBBIIdI", timestamp, flags, 0, h, w, float(pose_x), len(lm)),
                     Kf.tobytes(), Tf.tobytes(),
                     depth.tobytes() if flags & F_HAS_DEPTH else b"",
                     image.tobytes() if flags & F_HAS_IMAGE else b"",
                     lm.tobytes() if flags & F_HAS_LANDMARKS else b""])


def _decode_conf2d(payload):
    fmt = "!dIIBB"
    n = struct.calcsize(fmt)
    ts, h, w, half, flags = struct.unpack(fmt, payload[:n])
    dt = np.float16 if half else np.float32
    conf = np.frombuffer(payload, dt, h * w, n).reshape(h, w).astype(np.float32)
    return {"timestamp": ts, "confidence": conf}


def _decode_grid_delta(payload):
    fmt = "!dd3d3II"
    n = struct.calcsize(fmt)
    v = struct.unpack(fmt, payload[:n])
    cnt = int(v[8])
    idx = np.frombuffer(payload, np.int32, cnt * 3, n).reshape(-1, 3)
    return {"timestamp": v[0], "resolution": v[1], "dims": tuple(int(x) for x in v[5:8]),
            "idx": idx, "values": np.frombuffer(payload, np.float32, cnt, n + cnt * 12)}


def _decode_command(payload):
    v = struct.unpack(_CMD_FMT, payload[:struct.calcsize(_CMD_FMT)])
    return {"timestamp": v[0], "direction": np.array(v[1:4]), "target": np.array(v[4:7]),
            "speed": v[7], "clearance_m": v[8], "azimuth_deg": v[9], "stop": bool(v[10])}


def _recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(min(65536, n - len(buf)))
        if not part:
            return None
        buf += part
    return bytes(buf)


class TcpClient:
    """Для каналу через інтернет. Одне з'єднання несе і кадри, і відповіді,
    тому NAT з боку клієнта не заважає, а втрачені пакети TCP повторює сам."""

    def __init__(self, host, port=9000, timeout=15.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)

    def send(self, frame_id, timestamp, depth=None, T_wc=None, pose_x=0.0, K=None,
             image=None, landmarks=None):
        body = zlib.compress(
            _encode_frame2(timestamp, depth, T_wc, pose_x, K, image, landmarks), 1)
        self.sock.sendall(struct.pack(_TCP_HDR, _MAGIC, MSG_FRAME2, 0,
                                       frame_id & 0xFFFFFFFF, len(body)) + body)

    def results(self, timeout=1.0):
        self.sock.settimeout(timeout)
        try:
            while True:
                head = _recv_exactly(self.sock, _TCP_HDR_SIZE)
                if head is None:
                    return
                _m, mtype, _k, mid, length = struct.unpack(_TCP_HDR, head)
                body = _recv_exactly(self.sock, length)
                if body is None:
                    return
                body = zlib.decompress(body)
                if mtype == MSG_CONF2D:
                    r = _decode_conf2d(body); r["type"] = "conf2d"
                elif mtype == MSG_GRID_DELTA:
                    r = _decode_grid_delta(body); r["type"] = "grid_delta"
                elif mtype == MSG_COMMAND:
                    r = _decode_command(body); r["type"] = "command"
                else:
                    continue
                r["frame_id"] = mid
                yield r
        except (socket.timeout, OSError):
            return
        finally:
            self.sock.settimeout(None)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class Client:
    def __init__(self, host, port=9000, listen_port=9001, chunk_size=1400):
        self.addr = (host, port)
        self.chunk = chunk_size
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tx.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.rx.bind(("0.0.0.0", listen_port))
        self.rx.settimeout(0.3)
        self._pending = {}

    def send(self, frame_id, timestamp, depth=None, T_wc=None, pose_x=0.0, K=None,
             image=None, landmarks=None):
        payload = zlib.compress(
            _encode_frame2(timestamp, depth, T_wc, pose_x, K, image, landmarks), 1)
        n = max(1, -(-len(payload) // self.chunk))
        for seq in range(n):
            part = payload[seq * self.chunk:(seq + 1) * self.chunk]
            self.tx.sendto(struct.pack(_HDR, _MAGIC, MSG_FRAME2, 0,
                                        frame_id & 0xFFFFFFFF, seq, n, len(part)) + part,
                            self.addr)

    def results(self, timeout=1.0):
        """Віддає те, що повернув сервер: conf2d і grid_delta."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                pkt, _ = self.rx.recvfrom(65535)
            except socket.timeout:
                continue
            if len(pkt) < _HDR_SIZE or pkt[:4] != _MAGIC:
                continue
            _m, mtype, _k, mid, seq, nch, clen = struct.unpack(_HDR, pkt[:_HDR_SIZE])
            key = (mtype, mid)
            e = self._pending.setdefault(key, {})
            e[seq] = pkt[_HDR_SIZE:_HDR_SIZE + clen]
            if len(e) < nch:
                continue
            del self._pending[key]
            try:
                body = zlib.decompress(b"".join(e[i] for i in range(nch)))
            except zlib.error:
                continue
            if mtype == MSG_CONF2D:
                r = _decode_conf2d(body); r["type"] = "conf2d"
            elif mtype == MSG_GRID_DELTA:
                r = _decode_grid_delta(body); r["type"] = "grid_delta"
            elif mtype == MSG_COMMAND:
                r = _decode_command(body); r["type"] = "command"
            else:
                continue
            r["frame_id"] = mid
            yield r

    def close(self):
        self.tx.close(); self.rx.close()


def synthetic(i, h=480, w=640, fx=458.0):
    """Похила площина плюс три стовпи. Потрібна лише щоб перевірити звʼязок."""
    cx, cy = w / 2.0, h / 2.0
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ang = (v - cy) / fx
    depth = np.where(ang > 0.02, 2.0 / np.maximum(ang, 1e-3), 60.0).astype(np.float32)
    depth = np.clip(depth, 0.5, 60.0)
    for k, uu in enumerate((150, 320, 500)):
        m = (np.abs(u - uu) < 22) & (v > cy - 120) & (v < cy + 40)
        depth[m] = 12.0 + 4.0 * k
    grey = np.clip(255 * (1.0 - depth / 60.0), 0, 255).astype(np.uint8)
    grey[v < cy - 110] = 245                      # «небо» зверху, світле і гладке
    depth = depth + np.float32(np.random.default_rng(i).normal(0, 0.05, depth.shape))

    T = np.eye(4); T[0, 3] = 0.4 * i              # рух уздовж X
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], float)
    # «точки VSLAM»: справжня глибина у випадкових пікселях, переведена у світ
    rng = np.random.default_rng(1000 + i)
    vs = rng.integers(int(cy), h, 400); us = rng.integers(0, w, 400)
    z = depth[vs, us].astype(np.float64)
    pts_cam = np.stack([(us - cx) * z / fx, (vs - cy) * z / fx, z, np.ones_like(z)], 1)
    lm = (T @ pts_cam.T).T[:, :3].astype(np.float32)
    return depth, T, K, grey, lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="адреса сервера карти довіри")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--listen_port", type=int, default=9001)
    ap.add_argument("--selftest", action="store_true", help="слати синтетичну сцену")
    ap.add_argument("--tcp", action="store_true",
                     help="через інтернет. Одне з'єднання в обидва боки, втрати TCP повторює "
                          "сам. По UDP кадр на 700 КБ при 1 відсотку втрат доходить цілим "
                          "лише у 0,7 відсотка випадків")
    ap.add_argument("--image_only", action="store_true",
                     help="слати ЛИШЕ картинку, без глибини, пози і точок. Тоді все це має "
                          "порахувати приймач своїми VSLAM і FDE")
    ap.add_argument("--scale", type=int, default=1,
                     help="зменшити глибину і зображення у N разів перед відправкою. На "
                          "вузькому каналі це головний важіль")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--rate", type=float, default=4.0)
    args = ap.parse_args()

    if not args.selftest:
        print("Запусти з --selftest, або імпортуй Client і шли свої дані. "
              "Див. опис на початку файлу.")
        return 1

    if args.tcp:
        c = TcpClient(args.host, args.port)
        print(f"TCP з'єднання з {args.host}:{args.port}, відповіді тим самим каналом\n")
    else:
        c = Client(args.host, args.port, args.listen_port)
        print(f"UDP на {args.host}:{args.port}, слухаю відповіді на :{args.listen_port}\n")
    print(f"{'кадр':>4}  {'надіслано':<24}  {'назад':<40}")
    print("-" * 74)
    n_conf = n_delta = n_cmd = 0
    for i in range(args.n):
        depth, T, K, grey, lm = synthetic(i)
        if args.scale > 1:
            k = args.scale
            h2, w2 = (depth.shape[0] // k) * k, (depth.shape[1] // k) * k
            depth = depth[:h2, :w2].reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3)).astype(np.float32)
            grey = grey[:h2, :w2].reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3)).astype(np.uint8)
            K = K.copy(); K[0, 0] /= k; K[1, 1] /= k; K[0, 2] /= k; K[1, 2] /= k
        if args.image_only:
            c.send(i, time.time(), None, None, image=grey)
            sent = f"{grey.shape[1]}x{grey.shape[0]}, ЛИШЕ картинка"
        else:
            c.send(i, time.time(), depth, T, K=K, image=grey, landmarks=lm)
            sent = f"{depth.shape[1]}x{depth.shape[0]}, {len(lm)} точок"
        back = []
        for m in c.results(timeout=max(0.25, 1.0 / args.rate)):
            if m["type"] == "command":
                n_cmd += 1
                back.append("СТОП" if m["stop"] else
                            f"курс {m['azimuth_deg']:+.0f}°, {m['speed']:.1f} м/с")
            elif m["type"] == "conf2d":
                n_conf += 1
                cf = m["confidence"]
                back.append(f"карта довіри: {100*np.isfinite(cf).mean():.0f}% пікселів, "
                            f"w={np.nanmean(np.where(np.isfinite(cf), cf, np.nan)):.2f}")
            else:
                n_delta += 1
                back.append(f"{len(m['idx'])} вокселів")
        print(f"{i:>4}  {sent:<24}  {'; '.join(back) if back else '...'}")
    for m in c.results(timeout=3.0):
        if m["type"] == "conf2d":
            n_conf += 1
        elif m["type"] == "command":
            n_cmd += 1
        else:
            n_delta += 1
    c.close()

    print(f"\nнадіслано {args.n}, повернулось карт довіри {n_conf}, приростів {n_delta}"
          + (f", команд автопілота {n_cmd}" if n_cmd else ""))
    if n_conf:
        print("ЗВʼЯЗОК ПРАЦЮЄ В ОБИДВА БОКИ")
        return 0
    print("Нічого не повернулось. Перевір:\n"
          "  чи запущений сервер і чи досяжна його адреса\n"
          "  чи не блокує брандмауер UDP\n"
          "  чи слухає порт відповідей, заданий у --listen_port")
    return 1


if __name__ == "__main__":
    sys.exit(main())
