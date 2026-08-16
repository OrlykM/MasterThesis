import os
import re
import numpy as np


def read_csv_numeric(path, skip_header_lines=1):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            first_field = re.split(r"[,\s]+", line)[0]
            try:
                float(first_field)
            except ValueError:
                continue
            rows.append(line)
    out = []
    for line in rows:
        parts = [p for p in re.split(r"[,\s]+", line) if p != ""]
        out.append(parts)
    return out


def load_gt_trajectory(state_gt_csv_path):
    rows = read_csv_numeric(state_gt_csv_path)
    out = np.zeros((len(rows), 8), dtype=np.float64)
    for i, r in enumerate(rows):
        vals = [float(x) for x in r[:8]]
        t_ns, tx, ty, tz, qw, qx, qy, qz = vals
        out[i] = [t_ns * 1e-9, tx, ty, tz, qx, qy, qz, qw]
    return out


def load_image_timestamps(cam_data_csv_path):
    rows = read_csv_numeric(cam_data_csv_path)
    out = []
    for r in rows:
        t_ns = float(r[0])
        fname = r[1]
        out.append((t_ns * 1e-9, fname))
    return out


def load_camera_intrinsics(sensor_yaml_path):
    with open(sensor_yaml_path, "r") as f:
        text = f.read()

    def _extract_list(key):
        m = re.search(rf"{key}\s*:\s*\[([^\]]+)\]", text)
        if not m:
            return None
        return [float(x.strip()) for x in m.group(1).split(",")]

    intr = _extract_list("intrinsics")
    dist = _extract_list("distortion_coefficients")
    res = _extract_list("resolution")
    if intr is None:
        raise ValueError(f"Could not find 'intrinsics:' in {sensor_yaml_path}")

    fx, fy, cx, cy = intr
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return {
        "K": K,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "distortion": np.array(dist) if dist else None,
        "width": int(res[0]) if res else None,
        "height": int(res[1]) if res else None,
    }


def quat_to_rotmat(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw),     1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx**2 + qy**2)],
    ])


def pose_row_to_T(row):
    _, tx, ty, tz, qx, qy, qz, qw = row
    T = np.eye(4)
    T[:3, :3] = quat_to_rotmat(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T