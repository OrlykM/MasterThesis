r"""
Local, adaptive metric recalibration of dense monocular depth (FDE output)
against sparse VSLAM landmarks -- the reliability model behind
occupancy_3d_map.py's "adaptive_grid".

This replaces two earlier, dataset-specific designs that both turned out to
be dead ends once tested on real data:

  1. A single GLOBAL affine fit (alpha, beta) per frame, from ALL sparse
     anchors regardless of range. On an elongated corridor trajectory this
     let a handful of near-zero-parallax far points (looking straight down
     the flight path -- a classic monocular-VO degenerate case) dominate the
     fit and make it WORSE than no calibration (measured: RMSE 4.8m vs 0.45m
     once far anchors were excluded). Fixing that meant a hand-picked range
     cutoff (--calib_anchor_max_depth) -- exactly the "tied to one dataset"
     problem we're trying to get away from.
  2. sigma(z) = a + b*z curves fit OFFLINE against ground-truth depth
     (fit_error.py). Needs GT depth, which only exists in simulation --
     never available on a real deployment -- and the fit is frozen to
     whatever error PATTERN that one dataset happened to have.

What this module does instead, with zero ground truth and zero pre-fit
curves:

  - LocalCalibrator: instead of one alpha/beta for the whole frame, fits a
    SEPARATE weighted affine correction per image tile, using only that
    tile's k NEAREST sparse anchors (k a count, not a metric-space radius --
    portable across scene scales) with a kernel bandwidth set BY those same
    neighbors' own spread (so a sparse region automatically gets a wide,
    low-confidence kernel instead of silently reusing a distant, irrelevant
    anchor). A region far from any anchor naturally ends up low-confidence
    -- no hand-set range cutoff needed.
  - RunningScale: an online, exponentially-updated robust scale estimate.
    Everywhere the old code had a fixed reference constant (w_ref=1.0m,
    consistency_tau=2.0) tuned by hand for one dataset, this substitutes a
    value that self-calibrates to whatever residual/uncertainty magnitude
    THIS run is actually producing -- meters for one sensor pair, who knows
    what units for the next one.

Both are dataset-agnostic by construction: nothing here is fit offline
against any one dataset's error samples.
"""
import os
import re

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def load_dpvo_sparse_map(ply_path):
    """Reads DPVO's --save_ply output (ASCII PLY, x y z [r g b ...]) into an
    (N, 3) array, in DPVO's own (unscaled, arbitrary-origin) trajectory frame
    -- the SAME frame poses.txt is in, which is exactly what's needed: no
    alignment to any external frame is applied or required."""
    with open(ply_path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{ply_path}: no end_header found")
            header.append(line)
            if line.strip() == b"end_header":
                break
        header_text = b"".join(header).decode("ascii", errors="ignore")
        if "format ascii" not in header_text:
            raise NotImplementedError(f"{ply_path}: only ASCII PLY is supported")
        m = re.search(r"element vertex (\d+)", header_text)
        if not m:
            raise ValueError(f"{ply_path}: no 'element vertex N' in header")
        n_vertex = int(m.group(1))
        data = np.loadtxt(f, max_rows=n_vertex, usecols=(0, 1, 2))
    return data.reshape(-1, 3)


def load_orbslam3_landmarks(landmarks_path):
    """Reads landmarks.txt (one 'timestamp x y z' row per tracked map point
    per frame, written by the patched mono_euroc.cc -- see
    src/slam/orbslam3/) into (timestamps (M,), points (M,3)), in
    ORB-SLAM3's own trajectory frame (same frame poses.txt is in)."""
    data = np.loadtxt(landmarks_path, ndmin=2)
    return data[:, 0], data[:, 1:4]


class SparseMapAnchors:
    """Whichever sparse VSLAM source is available for a run, behind one
    'give me anchor points near this frame' query -- kept in the VSLAM's own
    raw trajectory frame throughout (no GT/Sim3 alignment: this has to work
    identically on a real flight with no ground truth available)."""

    def __init__(self, points, timestamps=None, time_tol=0.05):
        self.points = points
        self.timestamps = timestamps  # None => global cloud (DPVO); else per-point frame tags (ORB-SLAM3)
        self.time_tol = time_tol
        self._traj_t = None
        self._anchor_pose_idx = None
        self._window = None

    def attach_trajectory(self, traj_timestamps, traj_positions, window_frames=20):
        """Associates every landmark of an UNTAGGED cloud (DPVO's merged ply
        carries no frame labels) with the trajectory pose it sits closest to,
        so for_frame() can return only what belongs to the current part of the
        flight instead of the whole map.

        Without this, a frustum test on the global cloud admits landmarks from
        a completely different stretch of the trajectory that merely happen to
        line up with the optical axis -- which on a straight corridor is a
        large, systematically WRONG population: those are the near-zero-
        parallax points whose depth is barely constrained, and they were
        measured dominating the fit (anchor RMSE 4.8m vs 0.45m once excluded)."""
        if self.timestamps is not None:
            return self
        tree = cKDTree(traj_positions)
        _, nearest = tree.query(self.points)
        self._traj_t = np.asarray(traj_timestamps)
        self._anchor_pose_idx = nearest
        self._window = window_frames
        return self

    def for_frame(self, frame_t):
        if self.timestamps is not None:
            return self.points[np.abs(self.timestamps - frame_t) < self.time_tol]
        if self._anchor_pose_idx is None:
            return self.points
        j = int(np.argmin(np.abs(self._traj_t - frame_t)))
        near = np.abs(self._anchor_pose_idx - j) <= self._window
        return self.points[near]

    @classmethod
    def from_vslam_dir(cls, vslam_dir, time_tol=0.05):
        """Auto-detects landmarks.txt (ORB-SLAM3) or dpvo_sparse_map.ply
        (DPVO) in vslam_dir. Returns None if neither exists -- callers must
        treat that as "depth calibration unavailable for this run", not
        silently fuse uncalibrated depth as if it were corrected."""
        landmarks_path = os.path.join(vslam_dir, "landmarks.txt")
        ply_path = os.path.join(vslam_dir, "dpvo_sparse_map.ply")
        if os.path.exists(landmarks_path):
            ts, pts = load_orbslam3_landmarks(landmarks_path)
            return cls(pts, timestamps=ts, time_tol=time_tol)
        if os.path.exists(ply_path):
            return cls(load_dpvo_sparse_map(ply_path), timestamps=None)
        return None


def sim3_transform(points, r, t, s):
    """Applies a Umeyama Sim(3) (as returned by evo's PosePath3D.align) to
    points -- used ONLY for bringing a finished reconstruction into the
    ground-truth frame for evaluation/plotting. Never used inside the fusion
    pipeline itself: fusion stays entirely in the VSLAM's own raw frame, by
    design, so it needs nothing this function produces to run."""
    points = np.asarray(points, dtype=np.float64)
    return (r @ (s * points).T).T + t


def project_points(points, T_wc, K, w, h, min_depth, max_depth):
    """Projects 3D points into this frame's camera. Returns (u, v, z) for
    the points that land inside the image and within [min_depth, max_depth]."""
    if points.shape[0] == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    T_cw = np.linalg.inv(T_wc)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    pts_cam = (T_cw @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid_z = (z > min_depth) & (z < max_depth)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    safe_z = np.where(valid_z, z, 1.0)
    u = fx * pts_cam[:, 0] / safe_z + cx
    v = fy * pts_cam[:, 1] / safe_z + cy
    valid = valid_z & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    return u[valid], v[valid], z[valid]


def _bilinear_sample(img, u, v):
    h, w = img.shape
    u0 = np.clip(np.floor(u).astype(int), 0, w - 1)
    v0 = np.clip(np.floor(v).astype(int), 0, h - 1)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    du = np.clip(u - u0, 0.0, 1.0)
    dv = np.clip(v - v0, 0.0, 1.0)
    Ia, Ib = img[v0, u0], img[v0, u1]
    Ic, Id = img[v1, u0], img[v1, u1]
    return (Ia * (1 - du) * (1 - dv) + Ib * du * (1 - dv)
            + Ic * (1 - du) * dv + Id * du * dv)


def _upsample(grid, h, w):
    """Nearest-tile-centers -> full-resolution field, via the same PIL
    bilinear resize already used elsewhere in this codebase for depth maps
    -- gives a smoothly-blended field between tile centers for free."""
    return np.array(Image.fromarray(grid.astype(np.float32)).resize((w, h), Image.BILINEAR))


def _weighted_scale(z_pred, z_ref, kernel_w, min_alpha=0.1, max_alpha=10.0):
    """z_ref ~= s * z_pred, with s the KERNEL-WEIGHTED MEDIAN of the per-anchor
    ratios. This is the primary model agreed in docs/chat_context.md (7.2):
    one parameter, no offset, and a median rather than a least-squares fit, so
    it needs no separate outlier rejection at all -- half the anchors in the
    neighbourhood can be wrong before it moves."""
    if len(z_pred) < 1 or np.sum(kernel_w) < 1e-6:
        return None
    ok = z_pred > 1e-6
    if not np.any(ok):
        return None
    s = _weighted_median(z_ref[ok] / z_pred[ok], kernel_w[ok])
    if not (min_alpha < s < max_alpha):
        return None
    resid = z_ref - s * z_pred
    rmse = float(np.sqrt(np.average(resid ** 2, weights=np.maximum(kernel_w, 1e-9))))
    return float(s), 0.0, rmse


def _weighted_affine(z_pred, z_ref, kernel_w, n_iter=2, min_alpha=0.1, max_alpha=10.0,
                      prior=None, prior_strength=0.0):
    """z_ref ~= alpha*z_pred + beta via kernel-weighted least squares, with
    an extra Huber-style reweighting pass so one bad anchor inside an
    otherwise-good neighborhood doesn't swing the local fit.

    prior=(alpha0, beta0) with prior_strength>0 adds a ridge penalty
    prior_strength*((alpha-alpha0)^2+(beta-beta0)^2) to the objective --
    standard empirical-Bayes shrinkage toward a more-trusted estimate
    (LocalCalibrator passes the frame-wide robust fit here). This is also
    what replaces the old ad hoc "slope is ill-determined, fall back to
    shift-only" special case: when a neighborhood's z_pred barely varies,
    the data contributes ~no information about alpha, so the ridge term
    dominates and alpha naturally settles near alpha0 -- the SAME outcome,
    reached by one general mechanism instead of a hand-written branch.

    Returns None if there's not enough weight to fit, or the result lands
    somewhere implausible (alpha outside [min_alpha, max_alpha] -- a sign
    the neighborhood carries no real signal, not that it needs a bigger
    correction)."""
    if len(z_pred) < 2 or np.sum(kernel_w) < 1e-6:
        return None
    w = kernel_w.copy()
    alpha0, beta0 = prior if prior is not None else (1.0, 0.0)
    alpha, beta = alpha0, beta0
    ps = np.sqrt(max(prior_strength, 0.0))

    for _ in range(n_iter):
        sw = np.sqrt(np.maximum(w, 0.0))
        A = np.vstack([z_pred, np.ones_like(z_pred)]).T * sw[:, None]
        b = z_ref * sw
        if ps > 0:
            A = np.vstack([A, [ps, 0.0], [0.0, ps]])
            b = np.concatenate([b, [ps * alpha0, ps * beta0]])
        try:
            (alpha, beta), *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        resid = z_ref - (alpha * z_pred + beta)
        mad = np.median(np.abs(resid)) * 1.4826 + 1e-6
        huber_w = np.minimum(1.0, mad / (np.abs(resid) + 1e-9))
        w = kernel_w * huber_w

    if not (min_alpha < alpha < max_alpha):
        return None
    resid = z_ref - (alpha * z_pred + beta)
    rmse = float(np.sqrt(np.average(resid ** 2, weights=np.maximum(kernel_w, 1e-9))))
    return float(alpha), float(beta), rmse


def _weighted_median(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    valid = weights > 0
    if not np.any(valid):
        return float(np.median(values))
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    idx = min(np.searchsorted(cum, cum[-1] / 2.0), len(values) - 1)
    return float(values[idx])


class RunningScale:
    """Online, exponentially-weighted robust scale estimate -- replaces a
    fixed reference constant (the old w_ref=1.0m, consistency_tau=2.0) with
    one that adapts to whatever magnitude of residual/disagreement THIS run
    is actually producing. half_life is in UPDATES, not seconds or meters --
    a structural "how fast should this adapt" knob, not a value tied to any
    dataset's physical scale."""

    def __init__(self, init=1.0, half_life=30, min_scale=1e-3):
        self.value = float(init)
        self._beta = 0.5 ** (1.0 / half_life)
        self.min_scale = min_scale
        self.n = 0

    def update(self, sample):
        if sample is None or not np.isfinite(sample) or sample <= 0:
            return
        self.n += 1
        self.value = sample if self.n == 1 else self._beta * self.value + (1 - self._beta) * sample
        self.value = max(self.value, self.min_scale)


class MeasuredSensorModel:
    """Derives the log-odds increments from the data instead of assuming them.

    The textbook +0.85/-0.4 is not a neutral choice: it encodes p_hit=0.7 /
    p_free=0.4, i.e. "a measurement is right about 70% of the time". For
    monocular FDE that is far too generous, and it is exactly what makes the
    map overconfident -- a single weak observation already pushes a voxel past
    P>0.5, which is where the false-positive rate comes from.

    The only evidence available without ground truth is self-consistency: how
    often a fresh observation AGREES with what OTHER frames already concluded
    about the same voxel. That is a genuine measurement of this sensor chain's
    reliability on this data, and it is what sets the increments here.

    It measures agreement, not correctness -- a systematically biased sensor
    can agree with itself. So it cannot make a broken depth source good; what
    it does is stop the map claiming more confidence than the observations
    actually support."""

    def __init__(self, init_p=0.7, half_life=20, p_min=0.505, p_max=0.95):
        self.p_occ = RunningScale(init=init_p, half_life=half_life, min_scale=p_min)
        self.p_free = RunningScale(init=1.0 - 0.4, half_life=half_life, min_scale=p_min)
        self.p_min, self.p_max = p_min, p_max
        self.n_occ_samples = 0
        self.n_free_samples = 0

    def _logit(self, scale):
        p = float(np.clip(scale.value, self.p_min, self.p_max))
        return float(np.log(p / (1.0 - p)))

    @property
    def l_occ(self):
        return self._logit(self.p_occ)

    @property
    def l_free(self):
        return -self._logit(self.p_free)

    def observe(self, l_prior, update_sign):
        """l_prior: existing log-odds at the voxels this update touches.
        Only voxels another frame already had an opinion about carry any
        information about agreement, so the undecided ones are ignored."""
        decided = np.abs(l_prior) > 1e-6
        if not np.any(decided):
            return
        agree = float(np.mean(np.sign(l_prior[decided]) == update_sign))
        if update_sign > 0:
            self.p_occ.update(agree)
            self.n_occ_samples += 1
        else:
            self.p_free.update(agree)
            self.n_free_samples += 1

    def summary(self):
        return {"p_occ": round(float(np.clip(self.p_occ.value, self.p_min, self.p_max)), 3),
                "p_free": round(float(np.clip(self.p_free.value, self.p_min, self.p_max)), 3),
                "l_occ": round(self.l_occ, 3), "l_free": round(self.l_free, 3)}


class LocalCalibrator:
    """Produces a dense, smoothly-varying (alpha, beta, confidence) field
    for one frame's depth map from sparse VSLAM anchors -- confidence falls
    off automatically in regions with few/distant/disagreeing anchors,
    without any hand-set range cutoff. Two RunningScale instances make its
    notion of "typical fit error" and "typical anchor spacing" self-relative
    to this run instead of fixed constants."""

    def __init__(self, tile_px=40, k_neighbors=12, min_anchors_total=8, model="affine"):
        self.model = model  # "scale" (z'=s*z, the agreed primary) or "affine"
        self.tile_px = tile_px
        self.k_neighbors = k_neighbors
        self.min_anchors_total = min_anchors_total
        self.residual_scale = RunningScale(init=1.0, half_life=30)
        self.bandwidth_scale = RunningScale(init=1.0, half_life=30)
        self.n_frames_calibrated = 0
        self.n_frames_skipped = 0

    def calibrate_frame(self, anchor_u, anchor_v, anchor_zref, pred_depth):
        """anchor_u/v/zref: sparse anchors already projected into this
        frame (see project_points) with anchor_zref their VSLAM-triangulated
        depth. Returns None ("skip this frame's calibrated points entirely"
        -- consistent with this pipeline's "never fabricate evidence"
        principle) or (alpha_map, beta_map, confidence_map, sigma_map,
        support_map), each (H, W). sigma_map is the same uncertainty
        confidence_map is derived from, left in depth-meters so the caller
        can turn it into a spatial extent. support_map is how sparse the
        local anchor support is relative to this run's own typical spacing
        -- the signal for rejecting regions no VSLAM landmark ever reached."""
        n = len(anchor_u)
        if n < self.min_anchors_total:
            self.n_frames_skipped += 1
            return None

        h, w = pred_depth.shape
        anchor_zpred = _bilinear_sample(pred_depth, anchor_u, anchor_v)
        anchor_uv = np.stack([anchor_u, anchor_v], axis=1)
        tree = cKDTree(anchor_uv)

        k = min(self.k_neighbors, n)
        nx = max(1, round(w / self.tile_px))
        ny = max(1, round(h / self.tile_px))
        centers_u = (np.arange(nx) + 0.5) * (w / nx)
        centers_v = (np.arange(ny) + 0.5) * (h / ny)
        grid_u, grid_v = np.meshgrid(centers_u, centers_v)
        centers = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)

        dists, idxs = tree.query(centers, k=k)
        if k == 1:
            dists, idxs = dists[:, None], idxs[:, None]

        n_tiles = len(centers)
        bandwidth_g = np.maximum(dists[:, -1], 1e-6)  # distance to k-th neighbor = local kernel bandwidth
        kernel_ws = np.exp(-0.5 * (dists / bandwidth_g[:, None]) ** 2)

        # Pass 1: independent per-tile fits, no shrinkage. Used only to build
        # a robust PRIOR from the aggregate of tile-level estimates -- this
        # is what fixes the earlier global-only design's failure mode (a
        # cluster of far, low-parallax anchors tilting a single frame-wide
        # fit): pooled across ALL raw anchors, that cluster's residuals look
        # like normal-ish noise to an IRLS pass; isolated into their OWN
        # tile(s), they can only ever mistilt those tiles, and a weighted
        # MEDIAN across tiles shrugs off a minority of bad ones outright.
        alpha0 = np.ones(n_tiles)
        beta0 = np.zeros(n_tiles)
        neff = np.zeros(n_tiles)
        for i in range(n_tiles):
            if self.model == "scale":
                fit = _weighted_scale(anchor_zpred[idxs[i]], anchor_zref[idxs[i]], kernel_ws[i])
            else:
                fit = _weighted_affine(anchor_zpred[idxs[i]], anchor_zref[idxs[i]], kernel_ws[i])
            neff[i] = kernel_ws[i].sum()
            if fit is not None:
                alpha0[i], beta0[i] = fit[0], fit[1]

        prior = (_weighted_median(alpha0, neff), _weighted_median(beta0, neff))
        # Prior strength is a small pseudo-count, not a metric-space value --
        # a tile with local weight comparable to or above it trusts its own
        # data; one with much less defers almost entirely to the prior.
        prior_strength = max(1.0, self.k_neighbors / 3.0)

        # Pass 2: refit each tile, now shrinking toward that robust prior.
        alpha_g = np.ones(n_tiles)
        beta_g = np.zeros(n_tiles)
        rmse_g = np.full(n_tiles, np.inf)
        for i in range(n_tiles):
            if self.model == "scale":
                # Scale-only has no offset to shrink, and a weighted median is
                # already robust, so pass 2 is a no-op for it: reuse pass 1.
                fit = _weighted_scale(anchor_zpred[idxs[i]], anchor_zref[idxs[i]], kernel_ws[i])
            else:
                fit = _weighted_affine(anchor_zpred[idxs[i]], anchor_zref[idxs[i]], kernel_ws[i],
                                        prior=prior, prior_strength=prior_strength)
            if fit is not None:
                alpha_g[i], beta_g[i], rmse_g[i] = fit

        finite_rmse = rmse_g[np.isfinite(rmse_g)]
        self.residual_scale.update(np.median(finite_rmse) if finite_rmse.size else None)
        self.bandwidth_scale.update(np.median(bandwidth_g))

        # A tile's effective uncertainty grows both with its own fit's
        # residual AND with how far/sparse its supporting anchors were --
        # a tile with a "perfect" fit from anchors twice as far apart as
        # usual is still less trustworthy than the bandwidth alone suggests.
        # This is ALSO the depth-side term of the splat radius the caller
        # feeds to ProbabilisticVoxelGrid: a tile we're unsure about should
        # spread its evidence over a wider neighborhood, not plant a single
        # confidently-placed voxel -- capped (not left at inf) so a
        # completely unsupported tile still gets a finite, if generous,
        # spread instead of poisoning the upsampled field with inf/NaN.
        sigma_cap = 5.0 * self.residual_scale.value
        effective_sigma = rmse_g * (1.0 + bandwidth_g / self.bandwidth_scale.value)
        effective_sigma = np.where(np.isfinite(effective_sigma), np.minimum(effective_sigma, sigma_cap), sigma_cap)
        conf_g = 1.0 / (1.0 + effective_sigma / self.residual_scale.value)

        # How far this tile's nearest anchors are, RELATIVE to what's typical
        # for this run (not an absolute pixel or metre threshold -- that's the
        # whole point of it being portable). >1 means "sparser support than
        # usual here". Sky is the extreme case: it has no trackable features
        # at all, so no VSLAM landmark ever lands there, while the depth
        # network still confidently reports a surface -- measured at 24-48%
        # of every frame on the EuRoC quadrotor data, with ZoeDepth placing a
        # within-range surface in 100% of those pixels. The caller uses this
        # to DROP such pixels rather than merely down-weight them, because at
        # an occupancy read-out of P>0.5 any positive weight still marks the
        # voxel occupied.
        support_g = bandwidth_g / self.bandwidth_scale.value

        self.n_frames_calibrated += 1
        alpha_map = _upsample(alpha_g.reshape(ny, nx), h, w)
        beta_map = _upsample(beta_g.reshape(ny, nx), h, w)
        conf_map = _upsample(conf_g.reshape(ny, nx), h, w)
        sigma_map = _upsample(effective_sigma.reshape(ny, nx), h, w)
        support_map = _upsample(support_g.reshape(ny, nx), h, w)
        return alpha_map, beta_map, conf_map, sigma_map, support_map

    def summary(self):
        return {"n_frames_calibrated": self.n_frames_calibrated, "n_frames_skipped": self.n_frames_skipped,
                "residual_scale": self.residual_scale.value, "bandwidth_scale": self.bandwidth_scale.value}
