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
import socket
import struct
import time
import zlib

import numpy as np

_MAGIC = b"OVM1"
_HEADER_FMT = "!4sBBIHHI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_DATAGRAM = 65507

MSG_FRAME = 1
MSG_GRID = 2
KIND_NAIVE = 0
KIND_WEIGHTED = 1


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

    def messages(self, idle_timeout=None):
        """Yields (msg_type, kind, msg_id, payload) as messages complete.
        Runs until idle_timeout seconds pass with nothing received at all
        (None = run forever, i.e. until the caller stops iterating)."""
        last_rx = time.monotonic()
        while True:
            try:
                packet, _addr = self.sock.recvfrom(_MAX_DATAGRAM)
            except socket.timeout:
                self._reassembler.evict_stale()
                if idle_timeout is not None and time.monotonic() - last_rx > idle_timeout:
                    return
                continue
            last_rx = time.monotonic()
            result = self._reassembler.ingest(packet)
            if result is not None:
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

    def close(self):
        self.sock.close()


class FrameReceiver:
    def __init__(self, host, port, poll_timeout=0.5, stale_after=5.0):
        self._rx = UDPReceiver(host, port, poll_timeout=poll_timeout, stale_after=stale_after)

    def frames(self, idle_timeout=None):
        for msg_type, _kind, msg_id, payload in self._rx.messages(idle_timeout=idle_timeout):
            if msg_type != MSG_FRAME:
                continue
            frame = decode_frame(payload)
            frame["frame_id"] = msg_id
            yield frame

    def close(self):
        self._rx.close()


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
