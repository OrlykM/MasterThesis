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


def signed_permutation_rotations():
    """The 24 rotation matrices that are signed axis permutations -- i.e. every
    way two right-handed frames can disagree about which axis is which."""
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                out.append(M)
    return out


def rotmat_to_quat(R):
    """-> (qx, qy, qz, qw), matching load_gt_trajectory/load_tum column order."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([qx, qy, qz, qw])


def body_to_camera(traj_m, C):
    """Rewrites an Nx8 [t,xyz,quat] trajectory of BODY poses as CAMERA poses,
    given the camera->body rotation C (see estimate_axis_convention_fix).
    Positions are unchanged (T_BS has no translation here); only the
    orientation is re-expressed, which is what relative-pose metrics (RTE/RRE)
    and any backprojection actually depend on."""
    out = traj_m.copy()
    for i in range(len(out)):
        out[i, 4:8] = rotmat_to_quat(pose_row_to_T(traj_m[i])[:3, :3] @ C.T)
    return out


def estimate_axis_convention_fix(gt_m, est_m):
    """A trajectory alignment (Umeyama/Sim3) matches POSITIONS; it cannot
    absorb a disagreement about the CAMERA's own axis convention, because
    that's a right-multiplication on each pose, not a world transform. If the
    VSLAM's camera frame is labelled differently from the dataset's, every
    backprojection is thrown in the wrong direction while ATE still looks
    fine -- measured here as a constant 120 deg offset that collapsed
    occupancy IoU to 0.0005 even with PERFECT depth.

    Returns (C, residual_deg_after, residual_deg_before) with R_gt ~= R_est @ C,
    i.e. C is the CAMERA->BODY rotation. Verified on this data by the only
    arbiter that doesn't beg the question: DPVO's own triangulated landmarks
    sit 0.35m from the true surface when the GT poses are treated as BODY
    poses and converted with C.T, versus 8.93m if taken as camera poses. So
    the ground-truth trajectory here is in the body (FLU) frame despite
    sensor.yaml declaring T_BS identity, and the correction belongs on the
    GT side -- applying it to the VSLAM poses instead makes both sides
    self-consistent but leaves the whole map rotated, and silently puts the
    sparse landmark cloud in a different frame from the cameras.

    C is snapped
    to the nearest of the 24 signed axis permutations, so it is a DISCRETE
    convention choice (one per VSLAM/dataset pair, checked identical for both
    DPVO and ORB-SLAM3 here) rather than a continuously fitted parameter.
    Apply as: R_corrected = R_est @ R_fix.

    A small residual means the offset really was a convention mismatch. A
    large one means the VSLAM's orientations genuinely diverged and no single
    constant will fix them -- the caller should say so rather than silently
    applying a bogus rotation."""
    n = min(len(gt_m), len(est_m))
    rel = np.array([pose_row_to_T(est_m[i])[:3, :3].T @ pose_row_to_T(gt_m[i])[:3, :3]
                    for i in range(n)])

    def median_angle(R):
        cos = np.clip((np.trace(rel @ R.T, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.median(np.degrees(np.arccos(cos))))

    before = median_angle(np.eye(3))
    best = min(signed_permutation_rotations(), key=median_angle)
    return best, median_angle(best), before