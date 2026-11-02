r"""
UDP transport for the occupancy pipeline -- lets occupancy_3d_map.py run in
two extra roles besides "read a pre-recorded dataset":

    FrameSender   -> FrameReceiver   per-frame (depth, pose, ...) input,
                     so frames can arrive live instead of from disk.
    GridSender    -> GridReceiver    per-frame voxel-grid output, so each
                     frame's updated map can be pushed to a DIFFERENT address
                     (e.g. a visualizer or the safety layer) as soon as it's
                     fused, instead of only being written to disk at the end.

Best-effort, fire-and-forget (like RTP) -- no retransmission, no ordering
guarantees beyond what's needed to reassemble one message. That's the right
tradeoff for a live telemetry stream: a dropped/late frame should be skipped,
not stall the whole map. A payload (a depth image or a full voxel grid) is
almost always bigger than one UDP datagram, so every message is zlib-
compressed and split into <= chunk_size fragments that get reassembled by
message id on the receiving end; a message with any fragment still missing
after `stale_after` seconds is dropped.

Wire format, big-endian, one UDP datagram per chunk:
    magic      4s   b"OVM1"
    msg_type   B    1 = FRAME, 2 = GRID
    kind       B    FRAME: unused (0). GRID: 0 = naive, 1 = weighted
    msg_id     I    frame id -- groups chunks and lets a GRID message for the
                    same frame_id/kind overwrite/be matched to its FRAME
    seq        H    chunk index, 0-based
    n_chunks   H    total chunks in this message
    chunk_len  I    length of the chunk payload that follows
    payload    ...  chunk_len bytes (a slice of the zlib-compressed message)
"""
import collections
import socket
import struct
import threading
import time
import zlib

import numpy as np

_MAGIC = b"OVM1"
_HEADER_FMT = "!4sBBIHHI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_DATAGRAM = 65507

MSG_FRAME = 1
MSG_GRID = 2
MSG_FRAME2 = 3      # кадр з зображенням і розрідженими точками VSLAM
MSG_CONF2D = 4      # покадрова 2D карта довіри
MSG_GRID_DELTA = 5  # лише вокселі, які змінив цей кадр
MSG_COMMAND = 6     # команда автопілота на цей кадр
KIND_NAIVE = 0
KIND_WEIGHTED = 1
KIND_ADAPTIVE = 2

# прапорці кадру v2
F_HAS_K = 1 << 0
F_HAS_IMAGE = 1 << 1
F_HAS_LANDMARKS = 1 << 2
F_HAS_DEPTH = 1 << 3      # глибини може не бути: тоді її рахує приймач
F_HAS_POSE = 1 << 4       # пози теж: тоді її рахує VSLAM на боці приймача
F_IMAGE_RGB = 1 << 5      # зображення тривимірне, тобто кольорове


def _make_socket(buf_opt, buf_size):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, buf_opt, buf_size)
    except OSError:
        pass
    return sock


def _send_chunked(sock, addr, msg_type, kind, msg_id, payload, chunk_size):
    compressed = zlib.compress(payload, level=1)
    n_chunks = max(1, -(-len(compressed) // chunk_size))
    for seq in range(n_chunks):
        chunk = compressed[seq * chunk_size:(seq + 1) * chunk_size]
        header = struct.pack(_HEADER_FMT, _MAGIC, msg_type, kind, msg_id & 0xFFFFFFFF,
                              seq, n_chunks, len(chunk))
        sock.sendto(header + chunk, addr)


class _Reassembler:
    """Groups incoming chunks by (msg_type, kind, msg_id) and hands back the
    decompressed payload once every chunk for that key has arrived."""

    def __init__(self, stale_after=5.0):
        self.stale_after = stale_after
        self._pending = {}  # key -> {"chunks": {seq: bytes}, "n_chunks": int, "t0": float}

    def ingest(self, packet):
        if len(packet) < _HEADER_SIZE:
            return None
        magic, msg_type, kind, msg_id, seq, n_chunks, chunk_len = struct.unpack(
            _HEADER_FMT, packet[:_HEADER_SIZE])
        if magic != _MAGIC:
            return None
        chunk = packet[_HEADER_SIZE:_HEADER_SIZE + chunk_len]

        key = (msg_type, kind, msg_id)
        entry = self._pending.get(key)
        if entry is None:
            entry = {"chunks": {}, "n_chunks": n_chunks, "t0": time.monotonic()}
            self._pending[key] = entry
        entry["chunks"][seq] = chunk

        if len(entry["chunks"]) < entry["n_chunks"]:
            return None
        del self._pending[key]
        compressed = b"".join(entry["chunks"][i] for i in range(entry["n_chunks"]))
        try:
            payload = zlib.decompress(compressed)
        except zlib.error:
            return None
        return msg_type, kind, msg_id, payload

    def evict_stale(self):
        now = time.monotonic()
        dead = [k for k, v in self._pending.items() if now - v["t0"] > self.stale_after]
        for k in dead:
            del self._pending[k]


class UDPReceiver:
    """Low-level socket + reassembly loop shared by FrameReceiver/GridReceiver."""

    def __init__(self, host, port, poll_timeout=0.5, stale_after=5.0, recv_buf_size=4 * 1024 * 1024):
        # A voxel grid or depth frame is many chunks fired back-to-back; the
        # OS default recv buffer (often ~200KB) overflows under that burst
        # and silently drops packets, so bump it up best-effort.
        self.sock = _make_socket(socket.SO_RCVBUF, recv_buf_size)
        self.sock.bind((host, port))
        self.sock.settimeout(poll_timeout)
        self._reassembler = _Reassembler(stale_after=stale_after)
        self.last_addr = None
        self._n_unparsed = self._n_foreign = 0
        self._last_warn = 0.0

    def messages(self, idle_timeout=None):
        """Yields (msg_type, kind, msg_id, payload) as messages complete.
        Runs until idle_timeout seconds pass with nothing received at all
        (None або 0 = чекати вічно). Адреса відправника лишається в
        self.last_addr, щоб можна було відповісти туди ж."""
        if not idle_timeout:
            idle_timeout = None
        last_rx = time.monotonic()
        while True:
            try:
                packet, addr = self.sock.recvfrom(_MAX_DATAGRAM)
                self.last_addr = addr
            except socket.timeout:
                self._reassembler.evict_stale()
                if idle_timeout is not None and time.monotonic() - last_rx > idle_timeout:
                    return
                continue
            last_rx = time.monotonic()
            result = self._reassembler.ingest(packet)
            if result is None:
                # Датаграма не з нашого протоколу або ще не зібралась. Перший
                # випадок треба показати: інакше сервер мовчки ковтає сотні
                # мегабайтів, а користувач бачить лише тишу і не розуміє, що
                # відправник говорить іншою мовою.
                self._n_unparsed += 1
                if packet[:4] != _MAGIC:
                    self._n_foreign += 1
                    now = time.monotonic()
                    if self._n_foreign in (1, 50) or now - self._last_warn > 5.0:
                        self._last_warn = now
                        kind = ("схоже на JPEG" if b"\xff\xd8\xff" in packet[:64]
                                else "невідомий формат")
                        print(f"[udp] УВАГА: {self._n_foreign} датаграм не з протоколу "
                              f"OVM1 ({kind}), усі відкинуто. Якщо відправник шле голий "
                              f"JPEG, перезапусти з --raw_jpeg")
                continue
            yield result

    def close(self):
        self.sock.close()


def encode_frame(timestamp, depth, T_wc, pose_x=0.0, K=None):
    """pose_x: the x-value to evaluate the pose error model at (e.g.
    cumulative distance travelled since start/last loop closure) -- NOT a
    sigma itself. This matches occupancy_3d_map.py's dataset mode, which
    evaluates eval_error_model(pose_model, cum_dist) the same way; the
    receiving side (run_udp_mode) does that same evaluation."""
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    T_wc = np.ascontiguousarray(T_wc, dtype=np.float64).reshape(16)
    has_K = K is not None
    K_flat = np.ascontiguousarray(K, dtype=np.float64).reshape(9) if has_K else np.zeros(9)
    h, w = depth.shape
    header = struct.pack("!dBIIdd", timestamp, int(has_K), h, w, float(pose_x), 0.0)
    return header + K_flat.tobytes() + T_wc.tobytes() + depth.tobytes()


def decode_frame(payload):
    fixed_fmt = "!dBIIdd"
    fixed_size = struct.calcsize(fixed_fmt)
    timestamp, has_K, h, w, pose_x, _reserved = struct.unpack(fixed_fmt, payload[:fixed_size])
    off = fixed_size
    K_flat = np.frombuffer(payload, dtype=np.float64, count=9, offset=off).copy()
    off += 9 * 8
    T_wc = np.frombuffer(payload, dtype=np.float64, count=16, offset=off).reshape(4, 4).copy()
    off += 16 * 8
    depth = np.frombuffer(payload, dtype=np.float32, count=h * w, offset=off).reshape(h, w).copy()
    return {
        "timestamp": timestamp,
        "K": K_flat.reshape(3, 3) if has_K else None,
        "T_wc": T_wc,
        "pose_x": pose_x,
        "depth": depth,
    }


def encode_grid(grid):
    log_odds = np.ascontiguousarray(grid.log_odds, dtype=np.float32)
    dims = np.array(log_odds.shape, dtype=np.uint32)
    header = struct.pack("!d3d3I", grid.resolution, *grid.origin.tolist(), *dims.tolist())
    return header + log_odds.tobytes()


def decode_grid(payload):
    fixed_fmt = "!d3d3I"
    fixed_size = struct.calcsize(fixed_fmt)
    vals = struct.unpack(fixed_fmt, payload[:fixed_size])
    resolution = vals[0]
    origin = np.array(vals[1:4], dtype=np.float64)
    dims = tuple(vals[4:7])
    n = dims[0] * dims[1] * dims[2]
    log_odds = np.frombuffer(payload, dtype=np.float32, count=n, offset=fixed_size).reshape(dims).copy()
    return {"resolution": resolution, "origin": origin, "log_odds": log_odds}


# --------------------------------------------------------------- кадр версії 2

def encode_frame2(timestamp, depth=None, T_wc=None, pose_x=0.0, K=None, image=None,
                   landmarks=None):
    """Кадр з усім, що потрібно ЙМОВІРНІСНОМУ шару, а не лише глибиною.

    image      яскравість кадру, uint8, того ж розміру, що й глибина. Потрібна
               ознаці неба з sky_cue.py.
    landmarks  розріджені точки VSLAM (N, 3) у ТІЙ САМІЙ світовій системі, що й
               T_wc. Без них локальне калібрування неможливе, і кадр піде в
               карту без ваг довіри. Перетворення масштабу це справа
               відправника: приймач нічого не вирівнює.
    """
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
    lm = np.zeros((0, 3), dtype=np.float32)
    if landmarks is not None and len(landmarks):
        flags |= F_HAS_LANDMARKS
        lm = np.ascontiguousarray(landmarks, dtype=np.float32).reshape(-1, 3)
    K_flat = np.ascontiguousarray(K, dtype=np.float64).reshape(9) if K is not None else np.zeros(9)
    T_flat = (np.ascontiguousarray(T_wc, dtype=np.float64).reshape(16)
              if T_wc is not None else np.eye(4).reshape(16))

    parts = [struct.pack("!dBBIIdI", timestamp, flags, 0, h, w, float(pose_x), len(lm)),
             K_flat.tobytes(), T_flat.tobytes()]
    if flags & F_HAS_DEPTH:
        parts.append(depth.tobytes())
    if flags & F_HAS_IMAGE:
        parts.append(image.tobytes())
    if flags & F_HAS_LANDMARKS:
        parts.append(lm.tobytes())
    return b"".join(parts)


def decode_frame2(payload):
    fmt = "!dBBIIdI"
    size = struct.calcsize(fmt)
    timestamp, flags, _res, h, w, pose_x, n_lm = struct.unpack(fmt, payload[:size])
    off = size
    K_flat = np.frombuffer(payload, dtype=np.float64, count=9, offset=off).copy(); off += 72
    T_wc = np.frombuffer(payload, dtype=np.float64, count=16, offset=off).reshape(4, 4).copy()
    off += 128
    depth = None
    if flags & F_HAS_DEPTH:
        depth = np.frombuffer(payload, dtype=np.float32, count=h * w,
                               offset=off).reshape(h, w).copy()
        off += h * w * 4
    image = None
    if flags & F_HAS_IMAGE:
        ch = 3 if flags & F_IMAGE_RGB else 1
        image = np.frombuffer(payload, dtype=np.uint8, count=h * w * ch, offset=off)
        image = image.reshape(h, w, 3).copy() if ch == 3 else image.reshape(h, w).copy()
        off += h * w * ch
    landmarks = None
    if flags & F_HAS_LANDMARKS and n_lm:
        landmarks = np.frombuffer(payload, dtype=np.float32, count=n_lm * 3,
                                   offset=off).reshape(-1, 3).copy()
    return {"timestamp": timestamp, "K": K_flat.reshape(3, 3) if flags & F_HAS_K else None,
            "T_wc": T_wc if flags & F_HAS_POSE else None,
            "pose_x": pose_x, "depth": depth,
            "image": image, "landmarks": landmarks}


# ------------------------------------------------- 2D карта довіри як повідомлення

def encode_conf2d(timestamp, confidence, depth=None, half=True):
    """confidence: (H, W) float, нечисло означає відкинутий піксель.
    half: передавати як float16, тобто вдвічі менше байтів. Для величини від 0
    до 1 це близько трьох значущих цифр, чого для карти довіри досить."""
    conf = np.ascontiguousarray(confidence, dtype=np.float16 if half else np.float32)
    h, w = conf.shape
    flags = 1 if depth is not None else 0
    parts = [struct.pack("!dIIBB", timestamp, h, w, 1 if half else 0, flags), conf.tobytes()]
    if depth is not None:
        parts.append(np.ascontiguousarray(depth, dtype=np.float32).tobytes())
    return b"".join(parts)


def decode_conf2d(payload):
    fmt = "!dIIBB"
    size = struct.calcsize(fmt)
    timestamp, h, w, half, flags = struct.unpack(fmt, payload[:size])
    dt = np.float16 if half else np.float32
    off = size
    conf = np.frombuffer(payload, dtype=dt, count=h * w, offset=off).reshape(h, w).astype(np.float32)
    off += h * w * np.dtype(dt).itemsize
    depth = None
    if flags & 1:
        depth = np.frombuffer(payload, dtype=np.float32, count=h * w, offset=off).reshape(h, w).copy()
    return {"timestamp": timestamp, "confidence": conf, "depth": depth}


# ------------------------------------------------- команда автопілота за кадр

# мітка часу, напрямок, цільова точка, швидкість, запас, азимут, ознака стопу.
# Пробіли struct ігнорує, зате видно, де що
_CMD_FMT = "!d 3d 3d d d d B"


def encode_command(timestamp, direction, target, speed, stop, clearance_m=0.0,
                    azimuth_deg=0.0):
    """Те, що шар безпеки віддає автопілоту: куди летіти і як швидко."""
    d = np.ascontiguousarray(direction, dtype=np.float64).reshape(3)
    t = np.ascontiguousarray(target, dtype=np.float64).reshape(3)
    return struct.pack(_CMD_FMT, timestamp, *d.tolist(), *t.tolist(),
                        float(speed), float(clearance_m), float(azimuth_deg),
                        1 if stop else 0)


def decode_command(payload):
    v = struct.unpack(_CMD_FMT, payload[:struct.calcsize(_CMD_FMT)])
    return {"timestamp": v[0],
            "direction": np.array(v[1:4]), "target": np.array(v[4:7]),
            "speed": v[7], "clearance_m": v[8], "azimuth_deg": v[9],
            "stop": bool(v[10])}


# ----------------------------------------------- приріст воксельної карти за кадр

def encode_grid_delta(resolution, origin, dims, idx, values, timestamp=0.0):
    """idx: (N, 3) int32 індекси змінених вокселів, values: (N,) нові log-odds.
    Передавати весь масив щокадру немає сенсу: один кадр чіпає тисячі вокселів
    з мільйонів."""
    idx = np.ascontiguousarray(idx, dtype=np.int32).reshape(-1, 3)
    values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    header = struct.pack("!dd3d3II", timestamp, float(resolution),
                         *[float(x) for x in origin], *[int(x) for x in dims], len(idx))
    return header + idx.tobytes() + values.tobytes()


def decode_grid_delta(payload):
    fmt = "!dd3d3II"
    size = struct.calcsize(fmt)
    vals = struct.unpack(fmt, payload[:size])
    timestamp, resolution = vals[0], vals[1]
    origin = np.array(vals[2:5], dtype=np.float64)
    dims = tuple(int(x) for x in vals[5:8])
    n = int(vals[8])
    off = size
    idx = np.frombuffer(payload, dtype=np.int32, count=n * 3, offset=off).reshape(-1, 3).copy()
    off += n * 12
    values = np.frombuffer(payload, dtype=np.float32, count=n, offset=off).copy()
    return {"timestamp": timestamp, "resolution": resolution, "origin": origin,
            "dims": dims, "idx": idx, "values": values}


class FrameSender:
    """Sends per-frame (depth, pose, ...) input to a occupancy_3d_map.py
    ``--mode udp`` receiver, e.g. from a live camera/VSLAM front-end or from
    udp_replay_sender.py replaying a recorded dataset."""

    def __init__(self, host, port, chunk_size=1400, send_buf_size=4 * 1024 * 1024):
        self.addr = (host, port)
        self.chunk_size = chunk_size
        self.sock = _make_socket(socket.SO_SNDBUF, send_buf_size)

    def send(self, frame_id, timestamp, depth, T_wc, pose_x=0.0, K=None):
        payload = encode_frame(timestamp, depth, T_wc, pose_x=pose_x, K=K)
        _send_chunked(self.sock, self.addr, MSG_FRAME, 0, frame_id, payload, self.chunk_size)

    def send_frame2(self, frame_id, timestamp, depth, T_wc, pose_x=0.0, K=None,
                    image=None, landmarks=None):
        """Кадр з зображенням і розрідженими точками VSLAM. Саме цей варіант
        потрібен, щоб на приймачі побудувалась карта ДОВІРИ, а не просто карта
        зайнятості: без точок немає калібрування, без зображення немає вето
        неба."""
        payload = encode_frame2(timestamp, depth, T_wc, pose_x=pose_x, K=K,
                                 image=image, landmarks=landmarks)
        _send_chunked(self.sock, self.addr, MSG_FRAME2, 0, frame_id, payload, self.chunk_size)

    def close(self):
        self.sock.close()


class FrameReceiver:
    def __init__(self, host, port, poll_timeout=0.5, stale_after=5.0):
        self._rx = UDPReceiver(host, port, poll_timeout=poll_timeout, stale_after=stale_after)

    def frames(self, idle_timeout=None):
        """Віддає і старі кадри (лише глибина і поза), і нові (з зображенням та
        розрідженими точками). Поля `image` і `landmarks` для старих None."""
        for msg_type, _kind, msg_id, payload in self._rx.messages(idle_timeout=idle_timeout):
            if msg_type == MSG_FRAME:
                frame = decode_frame(payload)
                frame["image"] = None
                frame["landmarks"] = None
            elif msg_type == MSG_FRAME2:
                frame = decode_frame2(payload)
            else:
                continue
            frame["frame_id"] = msg_id
            frame["src_addr"] = self._rx.last_addr
            yield frame

    def close(self):
        self._rx.close()


_JPEG_SOI = b"\xff\xd8\xff"


def extract_jpeg(data):
    """Пакет може нести чужий короткий заголовок (номер кадру, часова мітка
    тощо) перед самою картинкою -- шукаємо початок JPEG (SOI-маркер) і беремо
    все від нього, байти перед ним ігноруються."""
    i = data.find(_JPEG_SOI)
    return None if i < 0 else data[i:]


class RawJpegFrameReceiver:
    """Для відправників, які НЕ використовують протокол OVM1 (FrameSender/
    client.py), а просто шлють JPEG-картинку одним UDP-пакетом, можливо зі
    своїм заголовком перед нею. Кожен пакет -- один кадр, без збирання
    шматків: що влізло в один датаграм, те й є кадром.

    Віддає той самий словник, що й FrameReceiver.frames(), з полями
    depth/T_wc/K/landmarks порожніми -- їх рахує my_vslam_and_fde() на
    сервері (чи заглушка --fake), так само, як для кадру "лише картинка",
    надісланого через client.py."""

    def __init__(self, host, port, poll_timeout=0.5, max_queue=1):
        """max_queue -- скільки найсвіжіших кадрів тримати. 1 означає
        "обробляй те, що зняла камера щойно, решту викинь", і це правильна
        умова для шару безпеки.

        Чому взагалі потрібен окремий потік. Якщо читати датаграми в тому
        самому циклі, що й обробляти, вони накопичуються в буфері ядра, а
        recvfrom віддає НАЙСТАРІШУ. При буфері 4 МБ і кадрі 45 КБ це
        дев'яносто кадрів запасу, тобто при 2,5 кадру за секунду команда
        відповідає тому, що камера бачила півхвилини тому. Коли буфер
        заповнюється, ядро починає мовчки викидати нові кадри, і сервер
        назавжди лишається працювати зі старим вікном."""
        self.sock = _make_socket(socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind((host, port))
        self.sock.settimeout(poll_timeout)
        self._next_id = 0
        self._q = collections.deque(maxlen=max(1, int(max_queue)))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._n_rx = 0          # скільки датаграм прийнято
        self._n_dropped = 0     # скільки витиснуто з черги як застарілі
        self._last_rx = time.monotonic()
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()

    def _rx_loop(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(_MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError:
                return
            # Мітку часу ставимо ТУТ, у мить отримання, а не тоді, коли кадр
            # вийде з черги. Інакше очікування в черзі ніде не видно: і
            # звіт про затримку, і мітка у векторі керування показували б
            # нуль навіть при півхвилинному відставанні.
            t_rx = time.time()
            with self._lock:
                self._n_rx += 1
                self._last_rx = time.monotonic()
                if len(self._q) == self._q.maxlen:
                    self._n_dropped += 1          # deque зараз витисне найстаріший
                self._q.append((data, addr, t_rx))

    @property
    def dropped(self):
        """Скільки кадрів витиснуто з черги як застарілі. Це не втрата в
        мережі, а свідомий вибір свіжості замість повноти."""
        return self._n_dropped

    def frames(self, idle_timeout=None):
        import io
        from PIL import Image

        while True:
            with self._lock:
                item = self._q.popleft() if self._q else None
                idle = time.monotonic() - self._last_rx
            if item is None:
                if idle_timeout is not None and idle > idle_timeout:
                    return
                time.sleep(0.002)
                continue
            data, addr, t_rx = item
            jpeg = extract_jpeg(data)
            if jpeg is None:
                continue
            try:
                image = np.array(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            except Exception:
                continue

            frame_id, self._next_id = self._next_id, self._next_id + 1
            yield {"timestamp": t_rx, "K": None, "T_wc": None, "pose_x": 0.0,
                   "depth": None, "image": image, "landmarks": None,
                   "frame_id": frame_id, "src_addr": addr}

    def close(self):
        self._stop.set()
        self.sock.close()


class GridSender:
    """Sends one frame's voxel grid to a separate address than the one frames
    were received from -- a visualizer, logger, or the downstream safety
    layer, independent of wherever the input frames are coming from."""

    def __init__(self, host, port, chunk_size=1400, send_buf_size=4 * 1024 * 1024):
        self.addr = (host, port)
        self.chunk_size = chunk_size
        self.sock = _make_socket(socket.SO_SNDBUF, send_buf_size)

    def send_grid(self, frame_id, grid, kind="weighted"):
        kind_byte = KIND_WEIGHTED if kind == "weighted" else KIND_NAIVE
        payload = encode_grid(grid)
        _send_chunked(self.sock, self.addr, MSG_GRID, kind_byte, frame_id, payload, self.chunk_size)

    def close(self):
        self.sock.close()


class MapPublisher:
    """Усе, що конвеєр віддає назовні в реальному часі, однією адресою:
    покадрову 2D карту довіри, приріст воксельної карти і, за потреби, повну
    сітку. Адреса окремішня від тієї, звідки приходять кадри, тому вхід і вихід
    можуть бути на різних машинах."""

    def __init__(self, host, port, chunk_size=1400, send_buf_size=4 * 1024 * 1024):
        self.addr = (host, port)
        self.chunk_size = chunk_size
        self.sock = _make_socket(socket.SO_SNDBUF, send_buf_size)
        self.bytes_sent = 0

    def _send(self, msg_type, kind, msg_id, payload):
        self.bytes_sent += len(payload)
        _send_chunked(self.sock, self.addr, msg_type, kind, msg_id, payload, self.chunk_size)

    def send_conf2d(self, frame_id, timestamp, confidence, depth=None, half=True):
        self._send(MSG_CONF2D, 0, frame_id,
                   encode_conf2d(timestamp, confidence, depth=depth, half=half))

    def send_grid_delta(self, frame_id, grid, idx, values, timestamp=0.0):
        self._send(MSG_GRID_DELTA, KIND_ADAPTIVE, frame_id,
                   encode_grid_delta(grid.resolution, grid.origin, grid.dims, idx, values,
                                      timestamp=timestamp))

    def send_grid(self, frame_id, grid, kind="adaptive"):
        kind_byte = {"naive": KIND_NAIVE, "weighted": KIND_WEIGHTED}.get(kind, KIND_ADAPTIVE)
        self._send(MSG_GRID, kind_byte, frame_id, encode_grid(grid))

    def send_command(self, frame_id, timestamp, cmd):
        self._send(MSG_COMMAND, 0, frame_id,
                   encode_command(timestamp, cmd["direction"], cmd["target"], cmd["speed"],
                                   cmd["stop"], cmd.get("clearance_m", 0.0),
                                   cmd.get("azimuth_deg", 0.0)))

    def send_raw(self, payload_bytes):
        """Один пакет, без заголовка OVM1, без стиснення і без розбивки на
        шматки -- для приймача, який не розбирає наш протокол і просто
        чекає сирі байти (наприклад, 11 float64 вектора автопілота)."""
        self.sock.sendto(payload_bytes, self.addr)

    def close(self):
        self.sock.close()


class MapSubscriber:
    """Приймає все, що шле MapPublisher. Кожне повідомлення віддається як
    словник з полем `type`: conf2d, grid_delta або grid."""

    def __init__(self, host, port, poll_timeout=0.5, stale_after=5.0):
        self._rx = UDPReceiver(host, port, poll_timeout=poll_timeout, stale_after=stale_after)

    def messages(self, idle_timeout=None):
        for msg_type, kind, msg_id, payload in self._rx.messages(idle_timeout=idle_timeout):
            if msg_type == MSG_CONF2D:
                m = decode_conf2d(payload); m["type"] = "conf2d"
            elif msg_type == MSG_GRID_DELTA:
                m = decode_grid_delta(payload); m["type"] = "grid_delta"
            elif msg_type == MSG_COMMAND:
                m = decode_command(payload); m["type"] = "command"
            elif msg_type == MSG_GRID:
                m = decode_grid(payload); m["type"] = "grid"
                m["kind"] = {KIND_NAIVE: "naive", KIND_WEIGHTED: "weighted"}.get(kind, "adaptive")
            else:
                continue
            m["frame_id"] = msg_id
            yield m

    def close(self):
        self._rx.close()


class GridReceiver:
    """Consumer-side counterpart of GridSender -- e.g. a live viewer or the
    safety layer listening on the "other address" the maps get forwarded to."""

    def __init__(self, host, port, poll_timeout=0.5, stale_after=5.0):
        self._rx = UDPReceiver(host, port, poll_timeout=poll_timeout, stale_after=stale_after)

    def grids(self, idle_timeout=None):
        for msg_type, kind, msg_id, payload in self._rx.messages(idle_timeout=idle_timeout):
            if msg_type != MSG_GRID:
                continue
            grid = decode_grid(payload)
            grid["frame_id"] = msg_id
            grid["kind"] = "weighted" if kind == KIND_WEIGHTED else "naive"
            yield grid

    def close(self):
        self._rx.close()


# ============================================================ транспорт по TCP
#
# Навіщо, якщо вже є UDP. Вимірювання: повідомлення ріжеться на шматки, і
# втрата ОДНОГО губить усе повідомлення, бо повторів немає. При 1% втрат, що
# для інтернету звичайна річ, кадр на 700 КБ доходить цілим у 0,7% випадків, а
# 2D карта довіри у 15%. У локальній мережі втрат практично немає і UDP
# правильний вибір, а через місто він просто не працює.
#
# TCP повторює втрачене сам. Платою є затримка на повторах і те, що одне
# застрягле повідомлення притримує наступні. Для каналу через пів міста це
# прийнятно, для бортового шару безпеки в локальній мережі ні, тому обидва
# транспорти лишаються.
#
# Додаткова зручність: одне з'єднання несе ОБИДВА напрямки. Клієнт сам
# під'єднується до сервера, тому NAT і брандмауер з боку клієнта не заважають,
# і не треба ні окремого порту для відповіді, ні auto, ні ретранслятора.
#
# Кадр на дроті: magic, тип, різновид, номер, довжина, тіло зі стисненням zlib.

_TCP_HDR = "!4sBBII"
_TCP_HDR_SIZE = struct.calcsize(_TCP_HDR)


def _tcp_send(sock, msg_type, kind, msg_id, payload):
    body = zlib.compress(payload, level=1)
    sock.sendall(struct.pack(_TCP_HDR, _MAGIC, msg_type, kind,
                              msg_id & 0xFFFFFFFF, len(body)) + body)


def _recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(min(65536, n - len(buf)))
        if not part:
            return None
        buf += part
    return bytes(buf)


def _tcp_recv(sock):
    head = _recv_exactly(sock, _TCP_HDR_SIZE)
    if head is None:
        return None
    magic, msg_type, kind, msg_id, length = struct.unpack(_TCP_HDR, head)
    if magic != _MAGIC:
        raise ValueError("чужі дані у потоці")
    body = _recv_exactly(sock, length)
    if body is None:
        return None
    return msg_type, kind, msg_id, zlib.decompress(body)


class TcpLink:
    """Одне з'єднання, обидва напрямки. Безпечний для викликів з різних потоків
    запис, бо відповіді шле потік обробки, а читає окремий."""

    def __init__(self, sock, peer):
        self.sock = sock
        self.peer = peer
        self._lock = threading.Lock()
        self.bytes_sent = 0

    def send(self, msg_type, kind, msg_id, payload):
        with self._lock:
            self.bytes_sent += len(payload)
            _tcp_send(self.sock, msg_type, kind, msg_id, payload)

    # ті самі назви, що в MapPublisher, щоб викликач не розрізняв транспорт
    def send_conf2d(self, frame_id, timestamp, confidence, depth=None, half=True):
        self.send(MSG_CONF2D, 0, frame_id,
                  encode_conf2d(timestamp, confidence, depth=depth, half=half))

    def send_grid_delta(self, frame_id, grid, idx, values, timestamp=0.0):
        self.send(MSG_GRID_DELTA, KIND_ADAPTIVE, frame_id,
                  encode_grid_delta(grid.resolution, grid.origin, grid.dims, idx, values,
                                     timestamp=timestamp))

    def send_grid(self, frame_id, grid, kind="adaptive"):
        kind_byte = {"naive": KIND_NAIVE, "weighted": KIND_WEIGHTED}.get(kind, KIND_ADAPTIVE)
        self.send(MSG_GRID, kind_byte, frame_id, encode_grid(grid))

    def send_command(self, frame_id, timestamp, cmd):
        self.send(MSG_COMMAND, 0, frame_id,
                  encode_command(timestamp, cmd["direction"], cmd["target"], cmd["speed"],
                                  cmd["stop"], cmd.get("clearance_m", 0.0),
                                  cmd.get("azimuth_deg", 0.0)))

    def messages(self):
        while True:
            try:
                got = _tcp_recv(self.sock)
            except (OSError, ValueError, zlib.error):
                return
            if got is None:
                return
            yield got

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class TcpFrameServer:
    """Слухає TCP і віддає кадри від усіх під'єднаних. Для кожного кадру
    відомо, яким з'єднанням він прийшов, тому відповідь іде туди ж."""

    def __init__(self, host, port, backlog=4):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(backlog)
        self.sock.settimeout(0.5)
        self._stop = threading.Event()
        self._queue = []
        self._cv = threading.Condition()
        self.links = []

    def _serve(self, link):
        try:
            for msg_type, _kind, msg_id, payload in link.messages():
                if msg_type == MSG_FRAME:
                    fr = decode_frame(payload); fr["image"] = None; fr["landmarks"] = None
                elif msg_type == MSG_FRAME2:
                    fr = decode_frame2(payload)
                else:
                    continue
                fr["frame_id"] = msg_id
                fr["link"] = link
                fr["src_addr"] = link.peer
                with self._cv:
                    self._queue.append(fr)
                    self._cv.notify()
        finally:
            link.close()
            if link in self.links:
                self.links.remove(link)
            print(f"[tcp] з'єднання з {link.peer[0]} закрите")

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, peer = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            link = TcpLink(conn, peer)
            self.links.append(link)
            print(f"[tcp] під'єднався {peer[0]}:{peer[1]}")
            threading.Thread(target=self._serve, args=(link,), daemon=True).start()

    def start(self):
        threading.Thread(target=self._accept_loop, daemon=True).start()
        return self

    def frames(self, idle_timeout=None):
        last = time.monotonic()
        while not self._stop.is_set():
            with self._cv:
                if not self._queue:
                    self._cv.wait(timeout=0.5)
                if self._queue:
                    fr = self._queue.pop(0)
                    last = time.monotonic()
                    yield fr
                    continue
            if idle_timeout and time.monotonic() - last > idle_timeout:
                return

    def close(self):
        self._stop.set()
        for l in list(self.links):
            l.close()
        self.sock.close()


def tcp_connect(host, port, timeout=10.0):
    """Клієнтський бік: одне з'єднання, яким ідуть і кадри, і відповіді."""
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(None)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return TcpLink(s, (host, port))

