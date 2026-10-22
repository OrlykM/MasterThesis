r"""
Probabilistic occupancy fusion for VSLAM + FDE, with an ADAPTIVE reliability
model (adaptive_grid) alongside two earlier, weaker baselines kept for
comparison:

  naive_grid    -- fixed +0.85/-0.4 log-odds regardless of confidence.
  weighted_grid -- LEGACY: sigma(z)=a+b*z / sigma(d)=c+e*d curves fit
                   OFFLINE against ground-truth depth (fit_error.py). Only
                   built if --error_model is given. Kept for comparison, not
                   because it's recommended: it needs GT depth (never
                   available on a real flight) and the fit is frozen to
                   whatever error pattern ONE dataset happened to have --
                   plug in a new VSLAM/FDE pair or a new environment and
                   it's silently wrong.
  adaptive_grid -- the new default. Zero ground truth, zero pre-fit curves:
                   * depth_calibration.LocalCalibrator recalibrates the
                     dense FDE depth per-tile against the VSLAM's own
                     sparse landmarks, with a kernel bandwidth set by the
                     landmarks' own local spacing -- a sparse/degenerate
                     region is automatically low-confidence, no hand-picked
                     range cutoff.
                   * depth_calibration.RunningScale replaces the fixed
                     reference constants (old w_ref, consistency_tau) with
                     an online estimate of whatever residual/disagreement
                     magnitude THIS run is actually producing.
                   Both adapt to whatever VSLAM+FDE+scene combination is
                   running, instead of being tuned for one dataset.

Note on metric scale: monocular VSLAM (no IMU fusion) has NO intrinsic
metric scale -- any system built on it needs SOME external scale reference.
Here that's the one-time Sim(3) fit to ground truth (a stand-in for what
visual-inertial scale recovery would give for real); every dataset-specific
GUESSWORK downstream of that (the error curves, the range cutoffs, the fixed
weighting constants) is what's been replaced. Scale recovery itself is a
separate, orthogonal problem (see mono-inertial VSLAM) -- out of scope here.

Usage:
    python3 occupancy_3d_map.py \
        --dataset_root ~/MasterThesis/datasets/euroc/2026-04-01_..._quadrotor \
        --vslam_dir ../../results/dpvo \
        --fde_dir ../../results/zoedepth \
        --output_dir maps/dpvo_zoedepth
    (--error_model is optional now -- only needed to also get the legacy
    weighted_grid for comparison.)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import (load_gt_trajectory, load_camera_intrinsics, load_image_timestamps, pose_row_to_T,
                       estimate_axis_convention_fix)
from compare_reconstruction_3d import (
    build_evo_trajectory, evo_traj_to_array, looks_like_frame_index, looks_like_raw_nanoseconds,
    remap_pseudo_timestamps, load_vslam_stride, load_tum, find_depth_file, load_depth_frame,
)
from evo.core import sync as evo_sync
from udp_streaming import FrameReceiver, GridSender, MapPublisher, TcpFrameServer
import depth_calibration as depth_calib
import depth_refine
import gt_frame
import blade_mask as bm
import sky_cue


def eval_error_model(model, x):
    """sigma(x) from a fitted model dict (linear or quadratic shape) -- only
    used by the legacy weighted_grid path."""
    if model is None:
        return np.zeros_like(x) if hasattr(x, "shape") else 0.0
    a, b = model["a"], model["b"]
    if model["shape"] == "quadratic":
        return a + b * x + model["c"] * x ** 2
    return a + b * x


_KERNEL_OFFSET_CACHE = {}


def _kernel_offsets(radius):
    """All integer (dx, dy, dz) within `radius` voxels of the origin (a
    ball, not a cube -- keeps the offset count down for larger radii) plus
    their Euclidean distances. Cached: in practice there are only a handful
    of distinct radii in play, since radius is itself derived from a small
    set of similar sigma values seen over a run."""
    radius = int(radius)
    if radius not in _KERNEL_OFFSET_CACHE:
        rng = np.arange(-radius, radius + 1)
        dx, dy, dz = np.meshgrid(rng, rng, rng, indexing="ij")
        offsets = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
        dist = np.linalg.norm(offsets, axis=1)
        keep = dist <= radius + 1e-9
        _KERNEL_OFFSET_CACHE[radius] = (offsets[keep], dist[keep])
    return _KERNEL_OFFSET_CACHE[radius]


class ProbabilisticVoxelGrid:
    """Log-odds occupancy grid where each point's contribution is scaled by
    a per-point confidence weight, plus time-based decay toward unknown --
    both missing from the naive baseline.

    Also applies consistency weighting: an update that CONTRADICTS what the
    grid already confidently believes about a voxel (from earlier passes) is
    suppressed in proportion to how confident that prior belief was; an
    update that CONFIRMS the prior gets full weight, and a fresh/unknown
    voxel (log-odds ~0) is never suppressed either way. This is what gives
    the map "memory" of previous passes over the same physical location.

    consistency_tau may be a plain float (fixed, legacy behavior) OR a
    depth_calibration.RunningScale instance -- when it's the latter, the
    "how confident is confident" threshold self-calibrates from the actual
    magnitude of prior beliefs this run has formed, instead of a constant
    tuned for one dataset's log-odds dynamics."""

    def __init__(self, bounds_min, bounds_max, resolution=0.5,
                 l_occ_base=0.85, l_free_base=-0.4, l_min=-4.0, l_max=4.0,
                 consistency_tau=2.0, use_consistency=True, sensor_model=None):
        self.origin = np.array(bounds_min, dtype=np.float64)
        self.resolution = resolution
        dims = np.ceil((np.array(bounds_max) - self.origin) / resolution).astype(int)
        self.dims = np.maximum(dims, 1)
        self.log_odds = np.zeros(self.dims, dtype=np.float32)
        self.l_occ_base, self.l_free_base = l_occ_base, l_free_base
        self.l_min, self.l_max = l_min, l_max
        self.consistency_tau = consistency_tau
        self.use_consistency = use_consistency
        # When given a depth_calibration.MeasuredSensorModel, the log-odds
        # increments stop being assumed constants and become measurements of
        # how self-consistent THIS sensor chain actually is on THIS data.
        self.sensor_model = sensor_model
        self.last_touched = np.zeros(0, dtype=np.int64)

    @property
    def _l_occ(self):
        return self.sensor_model.l_occ if self.sensor_model is not None else self.l_occ_base

    @property
    def _l_free(self):
        return self.sensor_model.l_free if self.sensor_model is not None else self.l_free_base

    @property
    def _tau(self):
        return self.consistency_tau.value if hasattr(self.consistency_tau, "value") else self.consistency_tau

    def world_to_voxel(self, points):
        return np.floor((points - self.origin) / self.resolution).astype(int)

    def _in_bounds(self, idx):
        return np.all((idx >= 0) & (idx < self.dims), axis=1)

    def _consistency_weight(self, l_prior, update_sign):
        """update_sign: +1 for an occupied update, -1 for a free update.
        Contradicts the prior -> weight shrinks with how confident the prior
        was. Confirms the prior (or prior is ~0, i.e. unknown) -> full weight.
        Also feeds the magnitude of touched (non-unknown) priors back into an
        adaptive tau, if one was given."""
        if hasattr(self.consistency_tau, "update"):
            touched = np.abs(l_prior[np.abs(l_prior) > 1e-6])
            if touched.size:
                self.consistency_tau.update(float(np.median(touched)))
        contradicts = (update_sign * l_prior) < 0
        return np.where(contradicts, 1.0 / (1.0 + np.abs(l_prior) / self._tau), 1.0)

    def insert_weighted(self, points_world, weights, sensor_origin, sigma=None, n_ray_samples=-1,
                         splat_k_sigma=1.5, splat_max_radius_voxels=4, splat_occupied=False,
                         track_changes=False):
        """track_changes: записати у self.last_touched плоскі індекси вокселів,
        яких торкнувся цей виклик. Потрібне лише для потокового режиму, де
        приріст за кадр треба віддати по мережі: інакше його довелося б
        діставати копіюванням і порівнянням усього масиву, а це при сітці на
        27 мільйонів вокселів чотири пересилки по 108 МБ на кадр. За
        замовчуванням вимкнено, тобто поведінка не змінюється."""
        """sigma: optional per-point positional uncertainty (meters, same
        length as points_world). When given, a point's OCCUPIED contribution
        is spread with a Gaussian kernel over a neighborhood of radius
        ceil(splat_k_sigma * sigma / resolution) voxels (capped at
        splat_max_radius_voxels) instead of landing in a single voxel, with
        weights normalized to sum to 1 -- the point injects the SAME total
        evidence, just admits honestly where it might actually be instead of
        planting it with false precision. This is what lets the map's
        OVERLAP with the true surface survive positional uncertainty (VSLAM
        drift, depth noise) that no amount of per-point CONFIDENCE weighting
        alone can fix -- confidence changes how much a point counts, not
        where it's allowed to land.

        Free-space ray-casting deliberately does NOT get this treatment: it
        stays a precise single-cell mark, so uncertainty can only ever make
        the map admit MORE possible occupancy nearby, never erase real
        occupancy by over-confidently clearing space around an uncertain
        reading ("never lie about free space").

        n_ray_samples: -1 (default) traverses each ray densely, one update
        per voxel actually crossed. A positive value keeps the old fixed
        sample count, for ablations."""
        if track_changes:
            self.last_touched = []
        if points_world.shape[0] == 0:
            if track_changes:
                self.last_touched = np.zeros(0, dtype=np.int64)
            return
        # Snapshot taken BEFORE this frame's update -- every point/ray in this
        # same call compares against the same prior, not a moving target as
        # other pixels from this same frame get applied first.
        snapshot = self.log_odds.reshape(-1).copy()
        updated = snapshot.copy()

        occ_idx = self.world_to_voxel(points_world)
        mask = self._in_bounds(occ_idx)
        occ_idx_v, w_occ = occ_idx[mask], weights[mask]
        if occ_idx_v.shape[0] > 0:
            if sigma is None or not splat_occupied:
                flat_occ = np.ravel_multi_index(occ_idx_v.T, self.dims)
                if self.sensor_model is not None:
                    self.sensor_model.observe(snapshot[flat_occ], +1)
                if self.use_consistency:
                    w_occ = w_occ * self._consistency_weight(snapshot[flat_occ], +1)
                np.add.at(updated, flat_occ, self._l_occ * w_occ)
                if track_changes:
                    self.last_touched.append(flat_occ)
            else:
                sigma_vox = np.maximum(sigma[mask], 0.0) / self.resolution
                radius_vox = np.clip(np.ceil(splat_k_sigma * sigma_vox).astype(int), 0, splat_max_radius_voxels)
                for r in np.unique(radius_vox):
                    bucket = radius_vox == r
                    base_idx, base_w = occ_idx_v[bucket], w_occ[bucket]
                    base_sigma_vox = np.maximum(sigma_vox[bucket], 1e-6)
                    offsets, dist = _kernel_offsets(r)
                    kw = np.exp(-0.5 * (dist[None, :] / base_sigma_vox[:, None]) ** 2)
                    kw = kw / np.maximum(kw.sum(axis=1, keepdims=True), 1e-12)
                    all_idx = (base_idx[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
                    all_w = (base_w[:, None] * kw).reshape(-1)
                    inb = self._in_bounds(all_idx)
                    all_idx, all_w = all_idx[inb], all_w[inb]
                    if all_idx.shape[0] == 0:
                        continue
                    flat = np.ravel_multi_index(all_idx.T, self.dims)
                    if self.sensor_model is not None:
                        self.sensor_model.observe(snapshot[flat], +1)
                    if self.use_consistency:
                        all_w = all_w * self._consistency_weight(snapshot[flat], +1)
                    np.add.at(updated, flat, self._l_occ * all_w)
                    if track_changes:
                        self.last_touched.append(flat)

        pts = points_world[mask]
        w_masked = weights[mask]
        if pts.shape[0] > 0 and n_ray_samples != 0:
            # DENSE traversal, not a fixed handful of samples: step along each
            # ray at half a voxel so EVERY voxel it passes through gets its
            # free-space update, the way occupancy mapping actually requires
            # (cf. OctoMap). With the old fixed 4 samples, a 30m ray at 0.5m
            # voxels marked 4 of ~60 traversed voxels free -- ~93% of the
            # space we demonstrably saw through got no evidence at all, which
            # is exactly what let phantom surfaces (an FDE predicting a
            # finite depth for empty sky) survive in the map: nothing ever
            # cleared them when a later frame looked straight through.
            ray_vecs = pts - sensor_origin[None, :]
            ray_lens = np.linalg.norm(ray_vecs, axis=1)
            step = 0.5 * self.resolution
            n_steps = int(np.clip(np.ceil(ray_lens.max() / step), 4, 4096)) if n_ray_samples < 0 \
                else int(n_ray_samples)
            t = np.linspace(0.05, 0.92, n_steps)
            ray_pts = sensor_origin[None, None, :] + t[:, None, None] * ray_vecs[None, :, :]
            n_pts = pts.shape[0]
            point_ids = np.tile(np.arange(n_pts)[None, :], (n_steps, 1)).reshape(-1)
            ray_w = np.tile(w_masked[None, :], (n_steps, 1)).reshape(-1)

            # Stop carving free space k*sigma SHORT of the endpoint when a
            # per-point uncertainty is known. With a fixed 92% cutoff and a
            # depth estimate that's off by several meters, the ray punches
            # straight through the real surface and erases it -- measured as
            # the reason dense traversal HURT on real FDE output (it cleared
            # true surfaces as fast as it cleared phantom ones). A point too
            # uncertain to place at all (k*sigma >= its own range) carves
            # nothing: "never lie about free space", now quantified.
            keep_t = np.ones((n_steps, n_pts), dtype=bool)
            if sigma is not None:
                safe_len = ray_lens - splat_k_sigma * np.maximum(sigma[mask], 0.0)
                t_max = np.clip(safe_len / np.maximum(ray_lens, 1e-9), 0.0, 0.92)
                keep_t = t[:, None] <= t_max[None, :]
            keep_flat = keep_t.reshape(-1)
            ray_pts = ray_pts.reshape(-1, 3)[keep_flat]
            point_ids = point_ids[keep_flat]
            ray_w = ray_w[keep_flat]
            if ray_pts.shape[0] == 0:
                self.log_odds = np.clip(updated, self.l_min, self.l_max).reshape(self.dims)
                if track_changes:
                    self.last_touched = (np.unique(np.concatenate(self.last_touched))
                                          if self.last_touched else np.zeros(0, dtype=np.int64))
                return

            free_idx = self.world_to_voxel(ray_pts)
            fmask = self._in_bounds(free_idx)
            free_idx_v, ray_w, point_ids = free_idx[fmask], ray_w[fmask], point_ids[fmask]
            if free_idx_v.shape[0] > 0:
                flat_free = np.ravel_multi_index(free_idx_v.T, self.dims)
                # Oversampling means one ray can land in the same voxel many
                # times; a ray must contribute ONE free update per voxel it
                # traverses, not one per sample, or short rays would clear
                # space far harder than long ones purely from step aliasing.
                keys = point_ids.astype(np.int64) * int(np.prod(self.dims)) + flat_free
                _, uniq = np.unique(keys, return_index=True)
                flat_free, ray_w = flat_free[uniq], ray_w[uniq]
                if self.sensor_model is not None:
                    self.sensor_model.observe(snapshot[flat_free], -1)
                if self.use_consistency:
                    ray_w = ray_w * self._consistency_weight(snapshot[flat_free], -1)
                np.add.at(updated, flat_free, self._l_free * ray_w)
                if track_changes:
                    self.last_touched.append(flat_free)

        self.log_odds = np.clip(updated, self.l_min, self.l_max).reshape(self.dims)
        if track_changes:
            self.last_touched = (np.unique(np.concatenate(self.last_touched))
                                  if self.last_touched else np.zeros(0, dtype=np.int64))

    def apply_decay(self, factor):
        """Pulls all log-odds toward 0 (unknown) by `factor` in [0,1] -- call
        once per frame/keyframe using a factor derived from your target
        half-life: factor = 1 - 0.5**(dt / half_life). This is what keeps
        the map LOCAL in time: old, no-longer-revisited evidence fades
        instead of accumulating forever, bounding how far a single old
        mistake (or a stretch of drift) can keep influencing the map."""
        self.log_odds *= (1.0 - factor)

    def occupied_mask(self, threshold=0.0):
        return self.log_odds > threshold

    def probability(self):
        """log-odds -> probability, the actual 'ймовірнісна карта' from your
        thesis title, not just a binary occupied/free grid."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))


def backproject_with_weight_map(depth, weight_map, K, T_wc, min_depth, max_depth,
                                 max_points=None, rng=None, extra_maps=()):
    """Shared backprojection core: depth and weight_map are both full (H, W)
    images (weight_map broadcastable, e.g. a scalar-filled array or a
    per-pixel confidence field from depth_calibration.LocalCalibrator).
    extra_maps: additional (H, W) fields (e.g. a sigma map) to subsample
    alongside depth/weight, at the SAME pixels/subsample indices -- returned
    as extra positional arrays after weights, in the given order."""
    h, w = depth.shape
    d = depth.astype(np.float64)
    wm = np.broadcast_to(weight_map, d.shape)
    valid = (d > min_depth) & (d < max_depth) & np.isfinite(d) & np.isfinite(wm)
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    us, vs, d, weights = us[valid], vs[valid], d[valid], wm[valid]
    extras = [np.broadcast_to(m, (h, w))[valid] for m in extra_maps]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - cx) * d / fx
    y = (vs - cy) * d / fy
    pts_cam = np.stack([x, y, d, np.ones_like(d)], axis=1)

    if max_points is not None and pts_cam.shape[0] > max_points:
        rng = rng or np.random.default_rng()
        idx = rng.choice(pts_cam.shape[0], size=max_points, replace=False)
        pts_cam, weights = pts_cam[idx], weights[idx]
        extras = [e[idx] for e in extras]

    pts_world = (T_wc @ pts_cam.T).T[:, :3]
    if extra_maps:
        return (pts_world, weights, *extras)
    return pts_world, weights


def legacy_weight_map(depth, sigma_pose_val, depth_model, w_ref):
    """The OLD reliability model: fixed sigma(z)/sigma(d) curves fit offline
    against ground truth. Kept only for the optional weighted_grid baseline."""
    sigma_total = eval_error_model(depth_model, depth) + sigma_pose_val
    return 1.0 / (1.0 + sigma_total / w_ref)


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dataset", "udp"], default="dataset",
                     help="'dataset' (default): read a pre-recorded EuRoC-style dataset. "
                          "'udp': fuse frames arriving live over UDP instead, see "
                          "FrameReceiver/GridSender in udp_streaming.py -- adaptive_grid isn't "
                          "available here yet, since the wire protocol doesn't carry sparse "
                          "VSLAM landmarks per frame")
    ap.add_argument("--dataset_root", help="required in --mode dataset")
    ap.add_argument("--vslam_dir", help="required in --mode dataset")
    ap.add_argument("--fde_dir", help="required in --mode dataset")
    ap.add_argument("--error_model", help="output of fit_error.py; OPTIONAL -- only builds the "
                                           "legacy weighted_grid baseline for comparison. "
                                           "adaptive_grid needs neither this nor ground truth.")
    ap.add_argument("--output_dir", required=True,
                     help="where naive_grid.npz/adaptive_grid.npz (and weighted_grid.npz if "
                          "--error_model given) get written")
    ap.add_argument("--max_time_diff", type=float, default=0.15,
                     help="max seconds between GT and estimated-pose timestamps to count as a "
                          "match -- AirSim recordings have a confirmed ~81ms GT/camera clock "
                          "offset, Isaac Sim datasets can use the tighter 0.02")
    ap.add_argument("--voxel_resolution", default="auto",
                     help="metres per voxel, 'auto' (default) or 'fine'. 'fine' is the finest "
                          "voxel that fits --max_grid_voxels, floored at --min_voxel: it makes "
                          "the map look like a reconstruction rather than a handful of blocks, "
                          "at the cost of quantising the depth far finer than its own measured "
                          "precision, so its IoU is LOWER. Use 'auto' for the metrics and 'fine' "
                          "for the pictures. 'auto': set to the depth uncertainty "
                          "the run MEASURES for itself (LocalCalibrator.residual_scale), times "
                          "--voxel_per_sigma. Quantising an input 10x finer than its own "
                          "precision is what made the map look like noise: measured IoU rose "
                          "monotonically 0.0128 -> 0.0615 as the voxel grew 0.5m -> 4m on data "
                          "with ~5m depth residual.")
    ap.add_argument("--max_grid_voxels", type=float, default=100e6,
                     help="with --voxel_resolution fine: how many voxels one grid may hold. "
                          "Three grids are built (naive, adaptive, GT reference) at 4 bytes a "
                          "voxel, so 100M is ~1.2 GB in total -- chosen for the 10 GB this WSL "
                          "has, not for any dataset.")
    ap.add_argument("--min_voxel", type=float, default=1.0,
                     help="with --voxel_resolution fine: never go finer than this, whatever the "
                          "budget allows. 1 m is roughly the size of the obstacles this map is "
                          "meant to warn about.")
    ap.add_argument("--n_frames", type=int, default=80, help="--mode dataset only")
    ap.add_argument("--max_points_per_frame", type=int, default=4000)
    ap.add_argument("--gt_depth_scale", type=float, default=0.001)
    ap.add_argument("--pred_depth_scale", type=float, default=1.0)
    ap.add_argument("--min_depth", type=float, default=0.1)
    ap.add_argument("--max_depth", default="auto",
                     help="metres, or 'auto' (default): the triangulation limit z = rel_tol * f "
                          "* B / sigma_pix, i.e. the range at which a stereo baseline of "
                          "--baseline_frames of the camera's own travel still gives "
                          "--rel_depth_tol relative depth accuracy. Everything in it is "
                          "measurable without ground truth. An earlier version took a percentile "
                          "of the landmark cloud instead; that does NOT work, because monocular "
                          "clouds are dominated by near-zero-parallax points whose depth is "
                          "unconstrained, so the cloud reports its own garbage back as 'range'.")
    ap.add_argument("--w_ref", type=float, default=1.0,
                     help="legacy weighted_grid only: reference uncertainty (meters) at which "
                          "weight drops to 0.5")
    ap.add_argument("--consistency_tau", type=float, default=2.0,
                     help="naive/legacy weighted_grid only: fixed prior-resistance threshold. "
                          "adaptive_grid always uses a self-calibrating RunningScale instead")
    ap.add_argument("--no_consistency", action="store_true",
                     help="ablation: disable prior-consistency weighting everywhere")
    ap.add_argument("--no_adaptive_grid", action="store_true",
                     help="skip adaptive_grid entirely (e.g. to only get naive/weighted)")
    ap.add_argument("--calib_model", choices=["scale", "affine"], default="scale",
                     help="per-tile depth correction: 'scale' is z'=s*z with s the kernel-weighted "
                          "MEDIAN ratio (one parameter, robust by construction -- the primary "
                          "model agreed in docs/chat_context.md 7.2); 'affine' is alpha*z+beta "
                          "with Huber IRLS and shrinkage toward a robust frame-level prior, kept "
                          "as the comparison variant.")
    ap.add_argument("--calib_tile_px", type=int, default=40,
                     help="adaptive_grid: image-space tile size (pixels) for local depth "
                          "recalibration -- a structural resolution choice, not a per-dataset "
                          "fit; scales naturally with image resolution")
    ap.add_argument("--calib_k_neighbors", type=int, default=12,
                     help="adaptive_grid: how many nearest sparse VSLAM landmarks each tile's "
                          "local fit uses -- a COUNT, not a metric-space radius, so it stays "
                          "meaningful regardless of scene scale")
    ap.add_argument("--anchor_window_frames", default="off",
                     help="restrict the calibration to landmarks belonging to +/- this many frames "
                          "of the current one (each landmark assigned to its nearest trajectory "
                          "pose) instead of the whole merged DPVO cloud. 'off' (default) uses the "
                          "whole cloud, because windowing was MEASURED not to help: +/-20 frames "
                          "gives IoU 0.013, +/-100 gives 0.120, the whole cloud 0.152, and "
                          "'auto' (max_depth / per-frame travel, i.e. +/-458 here) gives 0.155 -- "
                          "within noise of just using everything. The reason is geometric: a point "
                          "seen now at 50m was triangulated from poses 50m BEHIND, so a tight "
                          "window cuts away exactly the observations that constrain it.")
    ap.add_argument("--calib_min_anchors_total", type=int, default=8,
                     help="adaptive_grid: minimum sparse landmarks visible anywhere in a frame "
                          "before it's skipped entirely (never fabricate a calibration from "
                          "near-nothing)")
    ap.add_argument("--baseline_frames", type=int, default=10,
                     help="with --max_depth auto: how many frames of camera travel count as the "
                          "triangulation baseline (a COUNT -- roughly the VSLAM's own window)")
    ap.add_argument("--rel_depth_tol", type=float, default=0.1,
                     help="with --max_depth auto: the relative depth accuracy that still counts "
                          "as usable (0.1 = 10%%). Dimensionless.")
    ap.add_argument("--sigma_pix", type=float, default=1.0,
                     help="with --max_depth auto: assumed feature localisation noise in pixels "
                          "-- a property of the detector, not of the scene")
    ap.add_argument("--voxel_per_sigma", type=float, default=1.0,
                     help="with --voxel_resolution auto: voxel = this x measured depth "
                          "uncertainty. 1.0 = 'quantise at the noise level', the standard rule; "
                          "it is a dimensionless ratio, so it carries across scene scales.")
    ap.add_argument("--pose_drift_rate", type=float, default=0.005,
                     help="adaptive_grid: assumed monocular-VO drift as a fraction of distance "
                          "traveled (no GT, no loop closure possible on a one-way flight -- "
                          "well-tuned monocular VO/VIO without loop closure typically drifts on "
                          "the order of 0.3-1%% of distance traveled; this is a literature-typical "
                          "engineering default, NOT fit to this dataset's own ground truth, "
                          "unlike the old sigma_pose(d) curve -- it was sanity-checked against "
                          "this run's OWN measured ATE/distance ratio (~0.25%%) only to confirm "
                          "the order of magnitude is reasonable, the same way you'd sanity-check "
                          "a Kalman filter's assumed process noise, not fit a curve to it). Feeds "
                          "the position-uncertainty splat radius below, not a per-dataset calibration.")
    ap.add_argument("--axis_fix", choices=["auto", "off"], default="auto",
                     help="correct a camera-axis CONVENTION mismatch between the VSLAM's poses "
                          "and the dataset's frame. A Sim(3) trajectory alignment matches "
                          "positions only and cannot absorb this, so ATE looks fine while every "
                          "backprojection points the wrong way: measured as a constant 120 deg "
                          "offset here, which alone collapsed occupancy IoU to 0.0005 even with "
                          "PERFECT depth. 'auto' snaps it to the nearest of the 24 signed axis "
                          "permutations (a discrete convention choice, identical for DPVO and "
                          "ORB-SLAM3 on this data) and reports the residual so a genuinely "
                          "diverged trajectory is visible instead of silently patched.")
    ap.add_argument("--max_anchor_gap", type=float, default=0.0,
                     help="adaptive_grid: drop pixels whose nearest sparse VSLAM landmark is "
                          "further than this MULTIPLE of the run's own typical anchor spacing "
                          "(relative, not a pixel/metre constant). 0 = disabled, which is the "
                          "default because it was MEASURED counterproductive: it was meant to "
                          "reject sky (24-48%% of every frame, where the depth network invents a "
                          "surface), but DPVO's sparse cloud projects degenerate far points INTO "
                          "the sky region, so the gate cut real structure faster than phantoms "
                          "(IoU 0.0233 -> 0.0088 as it tightened). Kept for the ORB-SLAM3 "
                          "per-frame landmark source, where anchor support is honest.")
    ap.add_argument("--blade_mask", default="../masks/blades_airsim.npz",
                     nargs="?", const="",
                     help="drop the drone's own propellers (blade_mask.py) from EVERY variant and "
                          "from the GT reference -- they are part of the aircraft, not the scene. "
                          "Empty/absent mask = no effect (e.g. the Isaac Sim dataset has none).")
    ap.add_argument("--depth_net",
                     help="path to a depth_net.py checkpoint. Order per frame: FDE -> network "
                          "(refined depth, sigma_net, p_surface) -> drop p_surface < "
                          "--net_min_surface -> the per-tile VSLAM calibration still runs as a "
                          "safety net -> if the refined depth disagrees with the VSLAM landmarks "
                          "by more than --net_reject_k times the run's own residual scale, the "
                          "frame is fused WITHOUT the network. Applied to EVERY variant so the "
                          "comparison stays fair.")
    ap.add_argument("--net_min_surface", type=float, default=0.5,
                     help="with --depth_net: drop pixels the network says hold no surface")
    ap.add_argument("--net_reject_k", type=float, default=3.0,
                     help="with --depth_net: fall back to the statistical path when the network "
                          "disagrees with the VSLAM landmarks by more than this many residual scales")
    ap.add_argument("--depth_refiner",
                     help="path to a depth_refine.py table (learned on OTHER sequences) that "
                          "corrects this FDE model's in-domain bias and rejects pixels where a "
                          "surface rarely exists at all (sky). Ground truth is used to TRAIN it, "
                          "never at inference -- same status as the depth network's own training.")
    ap.add_argument("--sky_cue", action="store_true",
                     help="скасовувати відкидання таблицею там, де саме ЗОБРАЖЕННЯ каже, що "
                          "піксель небом бути не може, тобто темний або текстурований. Таблиця "
                          "індексована лише інтервалом глибини і номером рядка, тому в смузі "
                          "горизонту крона дерева і небо потрапляють в одну комірку і "
                          "відкидаються разом. Виміряно на наборі №1: у смузі рядків від 160 до "
                          "240 поверхня є у 53%% пікселів, а в карту йшло 20%%. З вето втрата "
                          "поверхні падає від значень від 10,5 до 17,3%% до значень від 0,1 до "
                          "5,9%% на всіх семи наборах, а фантомів неба додається від 2 до 4 "
                          "відсоткових пунктів. Для шару безпеки це правильний обмін: пропущена "
                          "перешкода не повертається ніколи, а фантом стирають промені вільного "
                          "простору з наступних кадрів. Застосовується ОДНАКОВО до базової і до "
                          "ймовірнісної карти. Еталон не потрібен.")
    ap.add_argument("--sky_cue_percentile", type=float, default=70.0,
                     help="з --sky_cue: перцентиль градієнта кадру, вище якого піксель вважається "
                          "текстурованим. Підібрано на наборах №1 до №3, перевірено на №4 до №7. "
                          "Перцентиль, а не абсолютний поріг, тому величина береться з самого "
                          "кадру.")
    ap.add_argument("--refiner_min_validity", type=float, default=0.5,
                     help="with --depth_refiner: drop pixels whose cell had a real surface less "
                          "often than this in training")
    ap.add_argument("--measured_sensor_model", action="store_true",
                     help="replace the assumed +0.85/-0.4 log-odds increments with ones measured "
                          "from how self-consistent this sensor chain proves to be. OFF by "
                          "default because it was MEASURED not to help: a systematically biased "
                          "depth source AGREES WITH ITSELF, so the measured p_occ pegs at its "
                          "0.95 ceiling and the map ends up MORE confident, not less (IoU 0.416 "
                          "-> 0.388 on dav2). Self-consistency cannot see systematic error -- "
                          "the same limit that makes accumulation unable to average such error "
                          "away. Kept because it becomes the right thing to do the moment the "
                          "depth source's errors are actually independent between frames.")
    ap.add_argument("--splat", action="store_true",
                     help="adaptive_grid: spread each point's occupied evidence over a "
                          "neighborhood sized by its uncertainty. OFF by default because the "
                          "controlled-noise ablation (noise_ablation.py) measured it as clearly "
                          "HARMFUL at an occupancy threshold of P>0.5: normalizing the mass over "
                          "N voxels still leaves every one of them above zero log-odds, so it "
                          "acts as pure dilation (IoU 0.42 -> 0.18 on clean input). It only "
                          "starts to make sense paired with a much higher read-out threshold.")
    ap.add_argument("--splat_k_sigma", type=float, default=1.5,
                     help="adaptive_grid: a point's OCCUPIED evidence is spread over a "
                          "neighborhood of ceil(this * sigma / voxel_resolution) voxels instead "
                          "of landing in one voxel -- sigma combines depth-calibration residual "
                          "and pose-drift uncertainty. Free-space rays are never spread (see "
                          "ProbabilisticVoxelGrid.insert_weighted).")
    ap.add_argument("--splat_max_radius_voxels", type=int, default=4,
                     help="adaptive_grid: hard cap on the splat radius above, so a wildly "
                          "uncertain point still costs bounded compute instead of smearing "
                          "across the whole grid")
    ap.add_argument("--n_ray_samples", type=int, default=4,
                     help="free-space ray sampling: N samples per ray, or -1 to traverse every "
                          "voxel the ray crosses. Dense (-1) is what occupancy mapping is "
                          "supposed to do, but it was MEASURED slightly worse here even with "
                          "uncertainty-limited carving, because a depth estimate off by several "
                          "meters makes each extra sample another chance to erase a real "
                          "surface; revisit this once depth error drops.")
    ap.add_argument("--prob_maps", metavar="DIR",
                     help="save the 2D occupancy probability for EVERY processed frame as a PNG: "
                          "P = sigmoid(l_occ * w) per pixel, after all corrections, with rejected "
                          "pixels left blank. One image per frame, nothing else.")
    ap.add_argument("--dump_intermediate", metavar="DIR",
                     help="save the representation that exists AFTER VSLAM + FDE + every "
                          "correction but BEFORE voxelisation: per-frame depth/confidence/sigma "
                          "fields for --dump_frames sample frames, plus the whole weighted point "
                          "cloud. Render it with view_intermediate.py.")
    ap.add_argument("--dump_frames", type=int, default=6,
                     help="with --dump_intermediate: how many sample frames to save 2D fields for")
    ap.add_argument("--dump_frame_name",
                     help="restrict --dump_intermediate AND --prob_maps to this ONE frame, named "
                          "exactly as its file in cam0/data (e.g. 1775034831515681460.png). The "
                          "run itself still processes every sampled frame, so the self-calibrating "
                          "scales are the warmed-up ones this flight actually produced, not a "
                          "first-frame guess -- only the saving is narrowed. Without it the dump "
                          "picks frames by a fixed stride, which cannot land on a chosen frame.")
    ap.add_argument("--gt_compare", action="store_true",
                     help="evaluation-only, --mode dataset: also backproject GT depth (via GT "
                          "poses, dataset_root/depth0/) into a reference grid on the SAME voxel "
                          "bounds, and report occupancy IoU/FPVR of naive/weighted/adaptive "
                          "against it -- the metric that actually reflects the THRESHOLDED "
                          "probabilistic map, unlike evaluation.py's raw per-point IoU which "
                          "never exercises the log-odds fusion (weights, splatting) at all")

    udp = ap.add_argument_group("--mode udp")
    udp.add_argument("--listen_host", default="0.0.0.0", help="address to receive frames on")
    udp.add_argument("--listen_port", type=int, default=9000, help="port to receive frames on")
    udp.add_argument("--bounds_min", type=float, nargs=3, metavar=("X", "Y", "Z"),
                      help="voxel grid lower corner (world coords, meters) -- required for "
                           "--mode udp since the grid must be allocated before any points "
                           "arrive, unlike --mode dataset which sizes it from all points upfront")
    udp.add_argument("--bounds_max", type=float, nargs=3, metavar=("X", "Y", "Z"),
                      help="voxel grid upper corner, see --bounds_min")
    udp.add_argument("--sensor_yaml", help="EuRoC sensor.yaml for camera intrinsics, used when "
                                            "incoming frames don't carry their own K")
    udp.add_argument("--send_host", help="куди віддавати результат. Значення auto означає "
                                          "відповідати тому, хто під'єднався, тобто на адресу "
                                          "джерела кадрів. Інакше: forward each frame's voxel grid to this "
                                          "address -- separate from --listen_host/--listen_port, "
                                          "so input frames and output maps can go to/from "
                                          "different machines")
    udp.add_argument("--send_port", type=int, default=9001)
    udp.add_argument("--send_grid", choices=["naive", "weighted", "both"], default="weighted")
    udp.add_argument("--send_every", type=int, default=1, help="forward the grid every Nth "
                                                                 "received frame, to save bandwidth")
    udp.add_argument("--max_frames", type=int, default=None, help="stop after this many frames "
                                                                    "(default: run until Ctrl+C)")
    udp.add_argument("--idle_timeout", type=float, default=None,
                      help="зупинитись, якщо кадрів немає стільки секунд. 0 або не задано "
                           "означає чекати вічно")
    udp.add_argument("--checkpoint_every", type=int, default=200,
                      help="also write naive_grid.npz/weighted_grid.npz to --output_dir every "
                           "N frames, not just on exit")
    udp.add_argument("--transport", choices=["udp", "tcp"], default="udp",
                      help="udp: швидко, без повторів, правильно для локальної мережі і борту. "
                           "tcp: для каналу через інтернет. Виміряно, що при 1%% втрат кадр на "
                           "700 КБ доходить цілим лише у 0,7%% випадків, бо втрата ОДНОГО "
                           "шматка губить усе повідомлення. TCP повторює сам, а ще несе обидва "
                           "напрямки ОДНИМ з'єднанням, тому клієнт за NAT під'єднується сам і "
                           "ні ретранслятор, ні порт для відповіді не потрібні")
    udp.add_argument("--chunk_size", type=int, default=1400,
                      help="розмір шматка UDP. Через WireGuard або Tailscale брати 1200: "
                           "1400 плюс заголовки плюс оболонка VPN перевищує MTU 1500 і дає "
                           "мовчазні втрати")
    udp.add_argument("--conf_scale", type=int, default=1,
                      help="зменшити 2D карту довіри у N разів перед відправкою. На каналі з "
                           "втратами це головний важіль: карта у чверть роздільності це 47 "
                           "шматків замість 185, тобто при 1%% втрат доходить 62%% замість 16%%")
    udp.add_argument("--rig_name",
                      help="назва апарата для маски гвинтів, наприклад run1. Потрібна лише "
                           "тоді, коли задано --blade_mask: маска описує гвинти ОДНОГО апарата "
                           "і застосовується тільки до тих записів, на яких її міряли. Для "
                           "нового апарата або без маски не потрібна зовсім. У режимі udp "
                           "--dataset_root не потрібен: усі дані приходять по мережі")
    udp.add_argument("--bounds_radius", type=float, default=None,
                      help="замість --bounds_min/--bounds_max виділити сітку кубом зі стороною "
                           "2R навколо ПЕРШОЇ отриманої пози. Для живого польоту це зручніше, "
                           "бо межі сцени наперед невідомі. Сітка лишається нерухомою: якщо "
                           "апарат вийде за неї, точки за межами просто не враховуються")
    udp.add_argument("--publish", nargs="+", default=["conf2d", "delta"],
                      choices=["conf2d", "delta", "grid", "none"],
                      help="що віддавати на --send_host. conf2d це покадрова 2D карта довіри "
                           "по пікселях, delta це лише ті вокселі, які змінив цей кадр, grid це "
                           "вся сітка. Повна сітка на мільйони вокселів щокадру нікому не "
                           "потрібна, тому типово conf2d і delta")
    udp.add_argument("--publish_full_grid_every", type=int, default=0,
                      help="додатково слати повну сітку кожні N кадрів, щоб новий підписник міг "
                           "наздогнати стан. 0 означає ніколи")
    udp.add_argument("--conf_float32", action="store_true",
                      help="слати 2D карту довіри у float32 замість float16. Удвічі більше "
                           "байтів заради значущих цифр, яких для величини від 0 до 1 не треба")
    udp.add_argument("--publish_depth", action="store_true",
                      help="класти у повідомлення conf2d ще й відкалібровану глибину, щоб "
                           "споживач міг сам будувати точки")
    udp.add_argument("--queue_frames", type=int, default=2,
                      help="скільки кадрів тримати в черзі прийому. Обробка одного кадру триває "
                           "сотні мілісекунд, тож при швидшому вхідному потоці черга "
                           "переповнюється і НАЙСТАРІШІ кадри викидаються. 1 означає завжди "
                           "працювати лише з найсвіжішим")
    udp.add_argument("--stats_every", type=int, default=20,
                      help="друкувати стан самоналаштовних величин кожні N кадрів")
    udp.add_argument("--legacy_weighted", action="store_true",
                      help="старий режим: замість ймовірнісного шару будувати weighted_grid за "
                           "кривими з --error_model. Лишено для сумісності, не рекомендовано")
    udp.add_argument("--decay_half_life", type=float, default=None,
                      help="if set, pull log-odds toward unknown once per frame with this "
                           "half-life in seconds (uses each frame's own timestamp gap)")
    return ap


naive_grid_l_occ = 0.85  # same increment ProbabilisticVoxelGrid applies per hit


def save_prob_map(out_dir, fname, conf_map, depth, min_depth, max_depth, l_occ):
    """One PNG per frame: the occupancy probability this frame writes into the
    map, pixel by pixel, after every correction. A pixel that was rejected, or
    whose depth fell outside the usable range, is left blank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = np.isfinite(conf_map) & np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    prob = np.where(usable, 1.0 / (1.0 + np.exp(-l_occ * np.nan_to_num(conf_map))), np.nan)

    # The colour range is the range this quantity CAN take, not [0.5, 1]: a
    # single observation contributes l_occ * w with the weight w in [0, 1], so
    # one frame can never push a voxel past sigmoid(l_occ) ~ 0.70. Scaling to
    # 1.0 made every frame render as one flat colour and hid the whole signal.
    p_max = 1.0 / (1.0 + np.exp(-l_occ))

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    im = ax.imshow(prob, cmap="RdYlGn_r", vmin=0.5, vmax=p_max, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{os.path.splitext(fname)[0]}   P(зайнято) з одного кадру\n"
                 f"{int(usable.sum())} з {usable.size} пікселів у карту "
                 f"({100*usable.mean():.0f}%), біле = відкинуто", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label(f"P(зайнято), максимум з одного кадру {p_max:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"prob_{os.path.splitext(fname)[0]}.png"), dpi=100)
    plt.close(fig)


def grey_for_net(args, fname, cam_h, cam_w):
    """Greyscale frame at the dataset's own resolution, for the network's image
    and gradient channels."""
    path = os.path.join(args.dataset_root, "cam0", "data", fname)
    img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    if img.shape != (cam_h, cam_w):
        img = np.array(Image.fromarray(img).resize((cam_w, cam_h), Image.BILINEAR))
    return img


def run_dataset_mode(args, hooks=None):
    """hooks: необов'язковий словник зворотних викликів для інструментів, що
    хочуть бачити проміжні величини цього ж прогону, не повторюючи його код.
    За замовчуванням None, тобто поведінка не змінюється жодним чином.

        "frame"(ctx)              після того, як для кадру остаточно пораховані
                                  вага довіри і відкалібрована глибина
        "grid_frame"(name, i, g)  після вставки i-го кадру у сітку name
        "grids"(d)                коли всі сітки побудовані

    Викликач нічого не повертає, усі словники передаються лише для читання."""
    hooks = hooks or {}
    hook_frame = hooks.get("frame")
    hook_grid_frame = hooks.get("grid_frame")
    hook_grids = hooks.get("grids")
    os.makedirs(args.output_dir, exist_ok=True)
    if not (args.dataset_root and args.vslam_dir and args.fde_dir):
        raise SystemExit("--mode dataset requires --dataset_root, --vslam_dir and --fde_dir")

    pose_model = depth_model = None
    if args.error_model:
        with open(args.error_model) as f:
            err = json.load(f)
        pose_model, depth_model = err["pose_model"], err["depth_model"]

    gt_traj_path = os.path.join(args.dataset_root, "state_groundtruth_estimate0", "data.csv")
    cam_csv_path = os.path.join(args.dataset_root, "cam0", "data.csv")
    sensor_yaml_path = os.path.join(args.dataset_root, "cam0", "sensor.yaml")
    fde_dir_full = os.path.join(args.fde_dir)
    poses_path = os.path.join(args.vslam_dir, "poses.txt")

    gt = load_gt_trajectory(gt_traj_path)
    est = load_tum(poses_path)
    frames = load_image_timestamps(cam_csv_path)
    if looks_like_frame_index(est):
        stride = load_vslam_stride(args.vslam_dir, default=1)
        est = remap_pseudo_timestamps(est, frames, stride)
    elif looks_like_raw_nanoseconds(est, gt):
        est = est.copy()
        est[:, 0] = est[:, 0] / 1e9

    # The ONE remaining use of ground truth: recovering monocular VSLAM's
    # missing metric scale (a stand-in for what visual-inertial fusion would
    # give on a real flight -- see module docstring). Nothing downstream of
    # this (the reliability weighting, the depth recalibration) touches GT.
    traj_ref = build_evo_trajectory(gt)
    traj_est = build_evo_trajectory(est)
    traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(traj_ref, traj_est, max_diff=args.max_time_diff)
    r_a, t_a, s_a = traj_est_sync.align(traj_ref_sync, correct_scale=True)
    gt_m = evo_traj_to_array(traj_ref_sync)
    est_m = evo_traj_to_array(traj_est_sync)
    gt_t, est_t = gt_m[:, 0], est_m[:, 0]

    R_axis_fix = np.eye(3)
    if args.axis_fix == "auto":
        R_bc_cached = gt_frame.resolve(args.dataset_root)
        if R_bc_cached is not None:
            R_axis_fix = R_bc_cached.T
            print("[axis_fix] using the GT-self-consistency convention from gt_frame.py")
            res_after = res_before = float("nan")
        else:
            R_axis_fix, res_after, res_before = estimate_axis_convention_fix(gt_m, est_m)
            print(f"[axis_fix] GT poses are body-frame: camera<->body offset {res_before:.2f} deg -> "
                  f"{res_after:.2f} deg residual after snapping to the nearest axis permutation "
                  f"(applied to the GT reference, not to the VSLAM poses)")
        if res_after == res_after and res_after > 5.0:
            print(f"[axis_fix] WARNING residual {res_after:.2f} deg is too large to be a pure "
                  f"convention offset -- this VSLAM run's orientations genuinely diverged, so "
                  f"the map will be wrong no matter what the fusion layer does")

    K = load_camera_intrinsics(sensor_yaml_path)["K"]
    cam_w = load_camera_intrinsics(sensor_yaml_path)["width"]
    cam_h = load_camera_intrinsics(sensor_yaml_path)["height"]

    xyz = est_m[:, 1:4]
    seg_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])  # from the ESTIMATE, not GT

    idx_step = max(1, len(frames) // args.n_frames)
    chosen = frames[::idx_step][:args.n_frames]
    if args.dump_frame_name:
        # Кадр, який просили зберегти, майже напевно не збігається з кроком
        # вибірки, тому його додають явно і ставлять у хронологічний порядок:
        # самоналаштовні масштаби оновлюються послідовно, і кадр має побачити
        # той самий стан, що й у звичайному прогоні.
        if not any(f == args.dump_frame_name for _t, f in chosen):
            extra = [(t, f) for t, f in frames if f == args.dump_frame_name]
            if not extra:
                raise SystemExit(f"--dump_frame_name {args.dump_frame_name}: такого кадру немає "
                                 f"в cam0/data.csv")
            chosen = sorted(chosen + extra, key=lambda tf: tf[0])
            print(f"[dump] кадр {args.dump_frame_name} додано у вибірку "
                  f"({len(chosen)} кадрів разом)")
    rng = np.random.default_rng(0)

    blades = None
    if args.blade_mask and os.path.exists(args.blade_mask):
        # Маска описує гвинти ОДНОГО апарата, тому її беруть лише для тих
        # наборів, на яких її й міряли. Інакше вона мовчки викидає справжні
        # пікселі з набору, у якого гвинтів у кадрі немає.
        ds_name = os.path.basename(os.path.normpath(args.dataset_root))
        parent = os.path.basename(os.path.dirname(os.path.normpath(args.dataset_root)))
        name = ds_name if bm.applies_to(args.blade_mask, ds_name) else parent
        blades = bm.load_mask(args.blade_mask, shape=(cam_h, cam_w), dataset_name=name)
        if blades.sum() == 0:
            blades = None
            print(f"[blade_mask] {args.blade_mask} не стосується набору {name} -- маска не "
                  f"застосовується")
        else:
            print(f"[blade_mask] excluding {int(blades.sum())} px/frame from EVERY variant")

    refiner = depth_refine.load_refiner(args.depth_refiner) if args.depth_refiner else None
    if refiner is not None:
        print(f"[refiner] loaded {args.depth_refiner}")

    net_refiner = None
    if args.depth_net:
        import depth_net_infer
        net_refiner = depth_net_infer.NetRefiner(args.depth_net)
        print(f"[depth_net] {net_refiner}")

    anchors = None
    local_calib = None
    if not args.no_adaptive_grid:
        anchors = depth_calib.SparseMapAnchors.from_vslam_dir(args.vslam_dir)
        if anchors is None:
            print(f"[adaptive] no landmarks.txt or dpvo_sparse_map.ply in {args.vslam_dir} -- "
                  f"adaptive_grid will be skipped for this run")
        else:
            anchors.points = depth_calib.sim3_transform(anchors.points, r_a, t_a, s_a)
            local_calib = depth_calib.LocalCalibrator(
                tile_px=args.calib_tile_px, k_neighbors=args.calib_k_neighbors,
                min_anchors_total=args.calib_min_anchors_total, model=args.calib_model)

    # ---- --max_depth auto: from triangulation geometry, not from the cloud ----
    # Depth from triangulation has sigma_z ~ z^2 * sigma_pix / (f * B), so the
    # range where depth is meaningful at all is z_max = rel_tol * f * B /
    # sigma_pix. Everything here is measurable without ground truth: f from
    # the intrinsics, B from how far the camera actually travels across the
    # VSLAM's own window, sigma_pix a feature-localisation constant. Taking a
    # percentile of the landmark cloud instead does NOT work: monocular VO
    # clouds are dominated by near-zero-parallax points whose depth is
    # unconstrained (median 121m, tail past 20km on this 227m-long scene), so
    # the cloud reports its own garbage back as "range".
    if isinstance(args.max_depth, str):
        if args.max_depth != "auto":
            args.max_depth = float(args.max_depth)
        else:
            xyz_est = est_m[:, 1:4]
            per_frame = np.linalg.norm(np.diff(xyz_est, axis=0), axis=1)
            baseline = float(np.median(per_frame)) * args.baseline_frames
            f = float((K[0, 0] + K[1, 1]) / 2.0)
            args.max_depth = max(args.rel_depth_tol * f * baseline / args.sigma_pix, 1.0)
            print(f"[auto] max_depth = {args.max_depth:.1f}m (triangulation limit: "
                  f"{args.rel_depth_tol:.0%} depth accuracy at f={f:.0f}px over a "
                  f"{baseline:.2f}m baseline)")

    # Landmark -> frame association, now that max_depth is a number: a point
    # visible at range R was triangulated from poses up to R behind us.
    if anchors is not None:
        if isinstance(args.anchor_window_frames, str) and args.anchor_window_frames != "off":
            if args.anchor_window_frames == "auto":
                per_frame = float(np.median(np.linalg.norm(np.diff(est_m[:, 1:4], axis=0), axis=1)))
                args.anchor_window_frames = int(np.clip(args.max_depth / max(per_frame, 1e-6),
                                                        20, len(est_m)))
                print(f"[auto] anchor_window = +/-{args.anchor_window_frames} frames "
                      f"({args.max_depth:.0f}m range / {per_frame:.3f}m per frame)")
            else:
                args.anchor_window_frames = int(args.anchor_window_frames)
        if args.anchor_window_frames != "off":
            anchors.attach_trajectory(est_m[:, 0], est_m[:, 1:4],
                                       window_frames=int(args.anchor_window_frames))

    gt_depth_dir = os.path.join(args.dataset_root, "depth0", "data")
    prob_dir = args.prob_maps
    if prob_dir:
        os.makedirs(prob_dir, exist_ok=True)

    dump_dir = args.dump_intermediate
    dumped, dump_every = 0, None
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        dump_every = max(1, len(chosen) // max(1, args.dump_frames))

    all_pts, all_weights, all_origins = [], [], []
    all_pts_w, all_weights_w = [], []
    all_pts_a, all_weights_a, all_sigma_a, all_origins_a = [], [], [], []
    all_pts_gt, all_origins_gt = [], []
    for frame_t, fname in tqdm(chosen, desc="fusing"):
        j_gt = int(np.argmin(np.abs(gt_t - frame_t)))
        j_est = int(np.argmin(np.abs(est_t - frame_t)))
        if abs(gt_t[j_gt] - frame_t) > args.max_time_diff or abs(est_t[j_est] - frame_t) > args.max_time_diff:
            continue
        pred_path = find_depth_file(fde_dir_full, fname)
        if pred_path is None:
            continue
        pred_depth = load_depth_frame(pred_path) * args.pred_depth_scale
        if pred_depth.shape != (cam_h, cam_w):
            pred_depth = np.array(Image.fromarray(pred_depth.astype(np.float32)).resize((cam_w, cam_h), Image.BILINEAR))

        T_wc = pose_row_to_T(est_m[j_est])
        origin = T_wc[:3, 3]

        if args.gt_compare:
            gt_depth_path = find_depth_file(gt_depth_dir, fname)
            if gt_depth_path is not None:
                gt_depth_raw = load_depth_frame(gt_depth_path)
                gt_depth_raw = np.where(gt_depth_raw >= 65535, np.nan, gt_depth_raw)
                gt_depth = gt_depth_raw * args.gt_depth_scale
                if blades is not None:
                    gt_depth = np.where(blades, np.nan, gt_depth)
                T_wc_gt = pose_row_to_T(gt_m[j_gt])
                T_wc_gt[:3, :3] = T_wc_gt[:3, :3] @ R_axis_fix.T  # GT is a BODY pose
                pts_gt, _ = backproject_with_weight_map(gt_depth, np.ones_like(gt_depth), K, T_wc_gt,
                                                         args.min_depth, args.max_depth,
                                                         max_points=args.max_points_per_frame, rng=rng)
                all_pts_gt.append(pts_gt)
                all_origins_gt.append(T_wc_gt[:3, 3])

        raw_fde_depth = pred_depth.copy()

        # The network runs first and, like the table below, before every
        # variant, so switching it on is a clean comparison rather than a
        # change only the adaptive path sees.
        net_sigma = None
        if net_refiner is not None:
            au0, av0, az0 = (np.zeros(0), np.zeros(0), np.zeros(0))
            if anchors is not None:
                au0, av0, az0 = depth_calib.project_points(
                    anchors.for_frame(est_m[j_est, 0]), T_wc, K, cam_w, cam_h,
                    args.min_depth, args.max_depth)
            out = net_refiner.refine(pred_depth, grey_for_net(args, fname, cam_h, cam_w),
                                      au0, av0, az0, args.min_depth)
            if out is not None:
                refined, sigma_m, p_surf = out
                scale = local_calib.residual_scale.value if local_calib is not None else np.nan
                if net_refiner.disagrees_with_anchors(refined, au0, av0, az0, scale,
                                                       args.net_reject_k):
                    pass  # too far from the landmarks -- keep the raw FDE depth
                else:
                    pred_depth = refined
                    net_sigma = sigma_m
                    net_keep = p_surf >= args.net_min_surface
                    pred_depth = np.where(net_keep, pred_depth, np.nan)

        # Applied BEFORE every variant, so "with refiner" vs "without" is a
        # clean 2x2 against naive/adaptive rather than a change only one of
        # them sees.
        refine_keep = None
        if refiner is not None:
            pred_depth, refine_keep = depth_refine.apply_refiner(
                refiner, pred_depth, min_validity=args.refiner_min_validity)
        if args.sky_cue:
            # Вето з зображення стоїть ТУТ, до побудови будь-якого варіанта, тому
            # базова і ймовірнісна карта бачать однаковий набір пікселів.
            not_sky = sky_cue.structure_mask(grey_for_net(args, fname, cam_h, cam_w),
                                              grad_percentile=args.sky_cue_percentile)
            refine_keep = not_sky if refine_keep is None else (refine_keep | not_sky)

        naive_wm = np.ones_like(pred_depth)
        if blades is not None:
            naive_wm = np.where(blades, np.nan, naive_wm)
        if refine_keep is not None:
            naive_wm = np.where(refine_keep, naive_wm, np.nan)
        pts, w = backproject_with_weight_map(pred_depth, naive_wm, K, T_wc,
                                              args.min_depth, args.max_depth,
                                              max_points=args.max_points_per_frame, rng=rng)
        all_pts.append(pts)
        all_weights.append(w)
        all_origins.append(origin)

        if pose_model is not None or depth_model is not None:
            sigma_pose_val = float(eval_error_model(pose_model, np.array([cum_dist[j_est]]))[0])
            wm = legacy_weight_map(pred_depth, sigma_pose_val, depth_model, args.w_ref)
            if blades is not None:
                wm = np.where(blades, np.nan, wm)
            pts_w, w_w = backproject_with_weight_map(pred_depth, wm, K, T_wc, args.min_depth, args.max_depth,
                                                      max_points=args.max_points_per_frame, rng=rng)
            all_pts_w.append(pts_w)
            all_weights_w.append(w_w)

        if local_calib is not None:
            frame_anchors = anchors.for_frame(est_m[j_est, 0])
            au, av, az = depth_calib.project_points(frame_anchors, T_wc, K, cam_w, cam_h,
                                                      args.min_depth, args.max_depth)
            result = local_calib.calibrate_frame(au, av, az, pred_depth)
            if result is not None:
                alpha_map, beta_map, conf_map, sigma_depth_map, support_map = result

                # Every reason to reject a pixel is collected into ONE mask and
                # applied last. Applying them to conf_map as we go would be
                # undone by the network branch below, which recomputes conf from
                # the combined sigma -- that silently cancelled the blade mask
                # and the table's sky rejection whenever --depth_net was on.
                drop = np.zeros(pred_depth.shape, dtype=bool)
                if blades is not None:
                    drop |= blades
                if refine_keep is not None:
                    drop |= ~refine_keep
                if args.max_anchor_gap > 0:
                    drop |= support_map > args.max_anchor_gap

                # Pose-drift term: no GT, no loop closure possible on a
                # one-way flight -- see --pose_drift_rate. Added (not just
                # max'd) with the depth-side sigma since they're independent
                # error sources compounding the same positional uncertainty.
                sigma_pose_val = args.pose_drift_rate * cum_dist[j_est]
                sigma_map = sigma_depth_map + sigma_pose_val
                if net_sigma is not None:
                    # independent error sources on the same quantity
                    sigma_map = np.sqrt(sigma_map ** 2 + net_sigma ** 2)
                    scale = max(local_calib.residual_scale.value, 1e-6)
                    conf_map = 1.0 / (1.0 + sigma_map / scale)

                # NaN weight => backproject_with_weight_map drops the pixel
                # outright. Down-weighting is not enough: at a read-out of
                # P>0.5 any positive weight still marks the voxel occupied.
                conf_map = np.where(drop, np.nan, conf_map)
                corrected = np.clip(alpha_map * pred_depth + beta_map, 0.0, None)

                wanted = args.dump_frame_name is None or fname == args.dump_frame_name
                if prob_dir and wanted:
                    save_prob_map(prob_dir, fname, conf_map, corrected,
                                   args.min_depth, args.max_depth, naive_grid_l_occ)

                if dump_dir and wanted and (args.dump_frame_name is not None
                                             or ((len(all_pts) - 1) % dump_every == 0
                                                 and dumped < args.dump_frames)):
                    np.savez_compressed(
                        os.path.join(dump_dir, "chosen_frame.npz" if args.dump_frame_name
                                      else f"frame_{dumped:02d}.npz"),
                        fde_raw=raw_fde_depth.astype(np.float32),
                        depth_after_net_or_table=pred_depth.astype(np.float32),
                        depth_calibrated=corrected.astype(np.float32),
                        alpha=alpha_map.astype(np.float32), beta=beta_map.astype(np.float32),
                        confidence=conf_map.astype(np.float32), sigma_m=sigma_map.astype(np.float32),
                        dropped=drop, anchors_uv=np.stack([au, av], 1).astype(np.float32),
                        anchors_z=az.astype(np.float32), frame=fname,
                        T_wc=T_wc, K=K,
                        residual_scale=float(local_calib.residual_scale.value),
                        bandwidth_scale=float(local_calib.bandwidth_scale.value),
                        sigma_pose=float(sigma_pose_val),
                        blades=(blades if blades is not None else np.zeros_like(drop)),
                        refine_dropped=(~refine_keep if refine_keep is not None
                                         else np.zeros_like(drop)))
                    dumped += 1
                if hook_frame is not None:
                    hook_frame(dict(
                        fname=fname, frame_t=frame_t, index=len(all_pts_a),
                        fde_raw=raw_fde_depth, depth_after_table=pred_depth,
                        depth_calibrated=corrected,
                        alpha=alpha_map, beta=beta_map, confidence=conf_map,
                        sigma_depth=sigma_depth_map, sigma_total=sigma_map,
                        sigma_pose=sigma_pose_val, support=support_map, dropped=drop,
                        refine_keep=refine_keep, blades=blades,
                        anchors_u=au, anchors_v=av, anchors_z=az,
                        T_wc=T_wc, K=K, origin=origin, local_calib=local_calib,
                        min_depth=args.min_depth, max_depth=args.max_depth))
                pts_a, w_a, sig_a = backproject_with_weight_map(
                    corrected, conf_map, K, T_wc, args.min_depth, args.max_depth,
                    max_points=args.max_points_per_frame, rng=rng, extra_maps=(sigma_map,))
                all_pts_a.append(pts_a)
                all_weights_a.append(w_a)
                all_sigma_a.append(sig_a)
                all_origins_a.append(origin)

    points = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    weights = np.concatenate(all_weights, axis=0) if all_weights else np.zeros(0)
    print(f"\nTotal points: {len(points)}, weight range: [{weights.min():.3f}, {weights.max():.3f}]"
          if len(weights) else "No points backprojected.")

    if local_calib is not None:
        print(f"[adaptive] {local_calib.summary()}")
    if net_refiner is not None:
        print(f"[depth_net] {net_refiner.summary()}")

    if len(points) == 0:
        print("Nothing to save.")
        return

    if dump_dir and all_pts_a:
        np.savez_compressed(
            os.path.join(dump_dir, "cloud_before_voxels.npz"),
            points=np.concatenate(all_pts_a).astype(np.float32),
            weights=np.concatenate(all_weights_a).astype(np.float32),
            sigma_m=np.concatenate(all_sigma_a).astype(np.float32),
            origins=np.stack(all_origins_a).astype(np.float32),
            points_naive=np.concatenate(all_pts).astype(np.float32))
        print(f"[dump] {dumped} frames of 2D fields + the weighted cloud -> {dump_dir}")

    points_a = np.concatenate(all_pts_a, axis=0) if all_pts_a else np.zeros((0, 3))
    weights_a = np.concatenate(all_weights_a, axis=0) if all_weights_a else np.zeros(0)
    bounds_src = np.concatenate([p for p in (points, points_a) if len(p)], axis=0)
    if isinstance(args.voxel_resolution, str):
        if args.voxel_resolution == "fine":
            # The finest voxel this scene fits in the budget. Purely a display
            # choice: nothing here claims the depth is good to that resolution.
            span = bounds_src.max(axis=0) - bounds_src.min(axis=0) + 2.0
            volume = float(np.prod(np.maximum(span, 1.0)))
            args.voxel_resolution = max(args.min_voxel,
                                         (volume / args.max_grid_voxels) ** (1.0 / 3.0))
            print(f"[fine] voxel_resolution = {args.voxel_resolution:.2f}m "
                  f"(finest that keeps a {span.round(0)} m scene under "
                  f"{args.max_grid_voxels/1e6:.0f}M voxels, floor {args.min_voxel} m)")
        elif args.voxel_resolution != "auto":
            args.voxel_resolution = float(args.voxel_resolution)
        elif local_calib is not None and local_calib.n_frames_calibrated > 0:
            args.voxel_resolution = float(args.voxel_per_sigma * local_calib.residual_scale.value)
            print(f"[auto] voxel_resolution = {args.voxel_resolution:.2f}m "
                  f"({args.voxel_per_sigma} x the {local_calib.residual_scale.value:.2f}m depth "
                  f"uncertainty this run measured for itself)")
            span = float(np.max(bounds_src.max(axis=0) - bounds_src.min(axis=0)))
            if args.voxel_resolution > span / 20.0:
                print(f"[auto] WARNING that voxel is coarser than 1/20th of the {span:.0f}m scene "
                      f"-- this depth source is too inaccurate to map this environment at any "
                      f"useful resolution, and no fusion setting will change that")
        else:
            args.voxel_resolution = 0.5
            print("[auto] no calibration to measure uncertainty from -- falling back to 0.5m")

    points_gt = np.concatenate(all_pts_gt, axis=0) if all_pts_gt else np.zeros((0, 3))
    bounds_src = np.concatenate([p for p in (points, points_a, points_gt) if len(p)], axis=0)
    bounds_min = bounds_src.min(axis=0) - 1.0
    bounds_max = bounds_src.max(axis=0) + 1.0

    # ---------- naive baseline: fixed log-odds, no weighting ----------
    naive_grid = ProbabilisticVoxelGrid(bounds_min, bounds_max, args.voxel_resolution)
    for i, (pts, origin) in enumerate(zip(all_pts, all_origins)):
        naive_grid.insert_weighted(pts, np.ones(len(pts)), origin, n_ray_samples=args.n_ray_samples)
        if hook_grid_frame is not None:
            hook_grid_frame("naive", i, naive_grid)

    # ---------- legacy formula-based weighting, only if --error_model given ----------
    weighted_grid = None
    if all_pts_w:
        weighted_grid = ProbabilisticVoxelGrid(
            bounds_min, bounds_max, args.voxel_resolution,
            consistency_tau=args.consistency_tau, use_consistency=not args.no_consistency)
        for pts_w, w_w, origin in zip(all_pts_w, all_weights_w, all_origins):
            weighted_grid.insert_weighted(pts_w, w_w, origin, n_ray_samples=args.n_ray_samples)

    # ---------- adaptive: local depth recalibration + self-calibrating confidence ----------
    adaptive_grid = None
    if len(points_a) > 0:
        sensor_model = depth_calib.MeasuredSensorModel() if args.measured_sensor_model else None
        adaptive_grid = ProbabilisticVoxelGrid(
            bounds_min, bounds_max, args.voxel_resolution,
            consistency_tau=depth_calib.RunningScale(init=args.consistency_tau, half_life=30),
            use_consistency=not args.no_consistency, sensor_model=sensor_model)
        for i, (pts_a, w_a, sig_a, origin) in enumerate(
                zip(all_pts_a, all_weights_a, all_sigma_a, all_origins_a)):
            adaptive_grid.insert_weighted(pts_a, w_a, origin, sigma=sig_a,
                                           n_ray_samples=args.n_ray_samples,
                                           splat_k_sigma=args.splat_k_sigma,
                                           splat_max_radius_voxels=args.splat_max_radius_voxels,
                                           splat_occupied=args.splat)
            if hook_grid_frame is not None:
                hook_grid_frame("adaptive", i, adaptive_grid)

    gt_grid = None
    if args.gt_compare and len(points_gt) > 0:
        # Reference = "which voxels contain a true surface", so no free-space
        # ray-casting here (n_ray_samples=0): GT rays grazing along a wall
        # would otherwise erode the very surfaces the reference is supposed
        # to define, shrinking the ground truth itself by ~2x.
        gt_grid = ProbabilisticVoxelGrid(bounds_min, bounds_max, args.voxel_resolution)
        for pts_gt, origin in zip(all_pts_gt, all_origins_gt):
            gt_grid.insert_weighted(pts_gt, np.ones(len(pts_gt)), origin, n_ray_samples=0)
        gt_mask = gt_grid.occupied_mask()
        print(f"\n[gt_compare] GT occupied voxels: {int(gt_mask.sum())}")
        # Swept over the occupancy threshold, because log_odds>0 (the old
        # single fixed report) is a threshold ANY single observation clears:
        # one glancing low-confidence hit marked a voxel occupied forever.
        # The point of a PROBABILISTIC map is that a voxel re-confirmed from
        # many viewpoints outranks one seen once -- which only shows up if
        # you actually require accumulated evidence.
        thresholds = [0.0, 0.85, 1.7, 2.55, 3.4]
        for label, grid in (("naive", naive_grid), ("weighted (legacy)", weighted_grid), ("adaptive", adaptive_grid)):
            if grid is None:
                continue
            print(f"[gt_compare] {label}:")
            for thr in thresholds:
                pred_mask = grid.occupied_mask(threshold=thr)
                tp = int(np.logical_and(gt_mask, pred_mask).sum())
                fp = int(np.logical_and(~gt_mask, pred_mask).sum())
                fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
                union = tp + fp + fn
                iou = tp / union if union > 0 else 0.0
                n_occ = int(pred_mask.sum())
                fpvr = fp / n_occ if n_occ > 0 else 0.0
                prec = tp / n_occ if n_occ > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                prob = 1.0 / (1.0 + np.exp(-thr))
                print(f"[gt_compare]   thr={thr:4.2f} (P>{prob:.2f}): IoU={iou:.4f} FPVR={fpvr:.4f} "
                      f"prec={prec:.4f} rec={rec:.4f}  occ={n_occ:7d} TP={tp:6d} FP={fp:8d} FN={fn:6d}")
    elif args.gt_compare:
        print("\n[gt_compare] requested but no GT depth frames matched -- skipped")

    if hook_grids is not None:
        hook_grids(dict(naive=naive_grid, weighted=weighted_grid, adaptive=adaptive_grid,
                        gt=gt_grid, bounds_min=bounds_min, bounds_max=bounds_max,
                        resolution=args.voxel_resolution,
                        frames=[o for o in all_origins_a], K=K))
    save_grids(args.output_dir, naive_grid, weighted_grid, adaptive_grid)
    print(f"Naive occupied voxels:    {int(naive_grid.occupied_mask().sum())}")
    if weighted_grid is not None:
        print(f"Weighted occupied voxels: {int(weighted_grid.occupied_mask().sum())} (legacy)")
    if adaptive_grid is not None:
        if adaptive_grid.sensor_model is not None:
            print(f"[sensor_model] measured from self-consistency: "
                  f"{adaptive_grid.sensor_model.summary()} (assumed would be "
                  f"l_occ=0.85, l_free=-0.4)")
        print(f"Adaptive occupied voxels: {int(adaptive_grid.occupied_mask().sum())}")
    print(f"Saved to: {args.output_dir}/")


def save_grids(output_dir, naive_grid, weighted_grid=None, adaptive_grid=None):
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "naive_grid.npz"),
                         log_odds=naive_grid.log_odds, origin=naive_grid.origin, resolution=naive_grid.resolution)
    if weighted_grid is not None:
        np.savez_compressed(os.path.join(output_dir, "weighted_grid.npz"),
                             log_odds=weighted_grid.log_odds, origin=weighted_grid.origin,
                             resolution=weighted_grid.resolution)
    if adaptive_grid is not None:
        np.savez_compressed(os.path.join(output_dir, "adaptive_grid.npz"),
                             log_odds=adaptive_grid.log_odds, origin=adaptive_grid.origin,
                             resolution=adaptive_grid.resolution)


def run_udp_mode(args):
    """Сервер реального часу. Приймає кадри по UDP, будує карту довіри з
    урахуванням усіх попередніх кадрів і віддає результат на окрему адресу.

    Стан між кадрами тримає realtime_map.RealtimeConfidenceMapper: там і
    самоналаштовні масштаби локального калібрування, і накопичена воксельна
    сітка, і поріг узгодженості. Обробка одного кадру там викликає ті самі
    функції, що й офлайновий режим."""
    import realtime_map

    if args.bounds_radius is None and (args.bounds_min is None or args.bounds_max is None):
        raise SystemExit("--mode udp потребує або --bounds_radius R, або пари "
                          "--bounds_min X Y Z і --bounds_max X Y Z: сітку треба виділити "
                          "до того, як прийде перший кадр")
    os.makedirs(args.output_dir, exist_ok=True)

    if isinstance(args.voxel_resolution, str):
        if args.voxel_resolution in ("auto", "fine"):
            raise SystemExit(f"--voxel_resolution {args.voxel_resolution} не працює в режимі "
                              f"udp: обидва способи рахують розмір з УСІХ точок прогону, а в "
                              f"реальному часі їх ще немає. Задай число, наприклад 1.0")
        args.voxel_resolution = float(args.voxel_resolution)
    if isinstance(args.max_depth, str):
        if args.max_depth == "auto":
            raise SystemExit("--max_depth auto не працює в режимі udp: межа рахується з усієї "
                              "траєкторії. Задай число, наприклад 50")
        args.max_depth = float(args.max_depth)

    K_default = load_camera_intrinsics(args.sensor_yaml)["K"] if args.sensor_yaml else None
    # Єдине, для чого тут потрібне ім'я: вирішити, чи стосується цього апарата
    # маска гвинтів. Жодних інших даних з диска потоковий режим не читає, усе
    # інше приходить по мережі.
    rig = args.rig_name or (os.path.basename(os.path.normpath(args.dataset_root))
                             if args.dataset_root else None)

    publish = set(args.publish) - {"none"}
    tcp = args.transport == "tcp"
    if tcp:
        # Одне з'єднання несе обидва напрямки, тому ні окремого порту для
        # відповіді, ні ретранслятора через NAT не потрібно: клієнт сам
        # під'єднується до нас.
        receiver = TcpFrameServer(args.listen_host, args.listen_port).start()
        auto_send = True
    else:
        receiver = FrameReceiver(args.listen_host, args.listen_port)
        auto_send = (args.send_host or "").lower() == "auto"
    pub = (MapPublisher(args.send_host, args.send_port, chunk_size=args.chunk_size)
           if (args.send_host and publish and not auto_send and not tcp) else None)
    print(f"[udp] чекаю кадри на {args.listen_host}:{args.listen_port} "
          f"({'TCP' if tcp else 'UDP'})")
    if pub:
        print(f"[udp] віддаю {', '.join(sorted(publish))} на {args.send_host}:{args.send_port}")
    elif auto_send and tcp:
        print(f"[udp] віддаю {', '.join(sorted(publish))} тим самим з'єднанням")
    elif auto_send:
        print(f"[udp] віддаю {', '.join(sorted(publish))} НАЗАД ТОМУ, хто під'єднався, "
              f"порт {args.send_port}")
    else:
        print("[udp] --send_host не заданий, нічого не віддаю, лише будую карту")

    # Прийом в окремому потоці з чергою на --queue_frames. Один кадр
    # обробляється сотні мілісекунд, і поки він обробляється, нові датаграми
    # просто переповнюють буфер сокета і гинуть мовчки. З чергою втрата стає
    # явною і керованою: беремо НАЙСВІЖІШИЙ кадр, а прострочені викидаємо і
    # рахуємо. Для шару безпеки це правильна політика, бо застаріла карта
    # гірша за пропущений кадр.
    import collections
    import threading

    inbox = collections.deque(maxlen=max(1, args.queue_frames))
    inbox_lock = threading.Lock()
    stop_rx = threading.Event()
    dropped = [0]

    def _rx_loop():
        try:
            for fr in receiver.frames(idle_timeout=args.idle_timeout):
                with inbox_lock:
                    if len(inbox) == inbox.maxlen:
                        dropped[0] += 1
                    inbox.append(fr)
                if stop_rx.is_set():
                    break
        finally:
            stop_rx.set()

    rx_thread = threading.Thread(target=_rx_loop, daemon=True)
    rx_thread.start()

    def _next_frame():
        while True:
            with inbox_lock:
                if inbox:
                    return inbox.popleft()
            if stop_rx.is_set():
                return None
            time.sleep(0.002)

    mapper = None
    n_received = n_with_conf = 0
    prev_timestamp = None
    t_start = time.monotonic()
    proc_times = []

    try:
        while True:
            frame = _next_frame()
            if frame is None:
                break
            K = frame["K"] if frame["K"] is not None else K_default
            if K is None:
                print("[udp] кадр пропущено: немає параметрів камери ні в кадрі, ні в "
                      "--sensor_yaml")
                continue

            if tcp:
                pub = frame.get("link") if publish else None
            elif auto_send and pub is None and frame.get("src_addr") and publish:
                # Відповідаємо туди, звідки прийшов кадр. Для перевірки з іншої
                # машини це знімає потребу знати її адресу наперед.
                host = frame["src_addr"][0]
                pub = MapPublisher(host, args.send_port, chunk_size=args.chunk_size)
                print(f"[udp] джерело {host}, віддаю результат на {host}:{args.send_port}")

            if mapper is None:
                if args.bounds_radius is not None:
                    c = np.asarray(frame["T_wc"], dtype=np.float64)[:3, 3]
                    bmin = c - args.bounds_radius
                    bmax = c + args.bounds_radius
                    print(f"[udp] сітка {2*args.bounds_radius:.0f} м навколо першої пози "
                          f"({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}), воксель "
                          f"{args.voxel_resolution} м")
                else:
                    bmin, bmax = np.array(args.bounds_min), np.array(args.bounds_max)
                mapper = realtime_map.RealtimeConfidenceMapper(
                    bmin, bmax, args.voxel_resolution,
                    refiner_path=args.depth_refiner,
                    refiner_min_validity=args.refiner_min_validity,
                    blade_mask_path=args.blade_mask, blade_dataset_name=rig,
                    use_sky_cue=args.sky_cue, sky_cue_percentile=args.sky_cue_percentile,
                    calib_tile_px=args.calib_tile_px,
                    calib_k_neighbors=args.calib_k_neighbors,
                    calib_min_anchors_total=args.calib_min_anchors_total,
                    calib_model=args.calib_model, max_anchor_gap=args.max_anchor_gap,
                    pose_drift_rate=args.pose_drift_rate,
                    min_depth=args.min_depth, max_depth=args.max_depth,
                    max_points_per_frame=args.max_points_per_frame,
                    n_ray_samples=args.n_ray_samples, splat=args.splat,
                    splat_k_sigma=args.splat_k_sigma,
                    splat_max_radius_voxels=args.splat_max_radius_voxels,
                    consistency_tau=args.consistency_tau,
                    use_consistency=not args.no_consistency)
                print(f"[udp] сітка {'x'.join(str(int(d)) for d in mapper.adaptive_grid.dims)} вокселів")

            t0 = time.monotonic()
            res = mapper.process(frame["depth"], frame["T_wc"], K,
                                  image=frame.get("image"), landmarks=frame.get("landmarks"),
                                  pred_depth_scale=args.pred_depth_scale)
            proc_times.append(time.monotonic() - t0)
            n_received += 1

            if res.get("confidence") is None:
                if n_received <= 3 or n_received % args.stats_every == 0:
                    print(f"[udp] кадр {frame['frame_id']}: {res.get('reason', 'без карти довіри')}")
            else:
                n_with_conf += 1
                if pub:
                    if "conf2d" in publish:
                        conf_out = res["confidence"]
                        if args.conf_scale > 1:
                            k = args.conf_scale
                            h2 = (conf_out.shape[0] // k) * k
                            w2 = (conf_out.shape[1] // k) * k
                            # середнє по блоку, але нечисла лишаються нечислами,
                            # тобто відкинутий окіл лишається відкинутим
                            blk = conf_out[:h2, :w2].reshape(h2 // k, k, w2 // k, k)
                            # Блок, у якому ВСІ пікселі відкинуті, дає нечисло,
                            # і це правильно, тому попередження тут зайве.
                            with np.errstate(invalid="ignore"):
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", RuntimeWarning)
                                    conf_out = np.nanmean(blk, axis=(1, 3))
                        pub.send_conf2d(frame["frame_id"], frame["timestamp"],
                                         conf_out,
                                         depth=(res["depth_calibrated"]
                                                if args.publish_depth and args.conf_scale == 1
                                                else None),
                                         half=not args.conf_float32)
                    if "delta" in publish and len(res["delta_idx"]):
                        pub.send_grid_delta(frame["frame_id"], mapper.adaptive_grid,
                                             res["delta_idx"], res["delta_values"],
                                             timestamp=frame["timestamp"])
                if n_received % args.stats_every == 0:
                    hz = n_received / max(time.monotonic() - t_start, 1e-9)
                    print(f"[udp] кадр {n_received}: S={res['residual_scale']:.2f} м, "
                          f"w сер.={res['mean_weight']:.3f}, у карту "
                          f"{100*res['kept_fraction']:.0f}% пікселів, "
                          f"змінено {len(res['delta_idx'])} вокселів, "
                          f"{1000*np.mean(proc_times[-args.stats_every:]):.0f} мс/кадр, "
                          f"вхід {hz:.1f} Гц")

            if pub and "grid" in publish:
                pub.send_grid(frame["frame_id"], mapper.adaptive_grid, kind="adaptive")
            elif (pub and args.publish_full_grid_every
                  and n_received % args.publish_full_grid_every == 0):
                pub.send_grid(frame["frame_id"], mapper.adaptive_grid, kind="adaptive")

            if args.decay_half_life and prev_timestamp is not None:
                dt = max(0.0, frame["timestamp"] - prev_timestamp)
                factor = 1.0 - 0.5 ** (dt / args.decay_half_life)
                mapper.adaptive_grid.apply_decay(factor)
                if mapper.naive_grid is not None:
                    mapper.naive_grid.apply_decay(factor)
            prev_timestamp = frame["timestamp"]

            if args.checkpoint_every and n_received % args.checkpoint_every == 0:
                save_grids(args.output_dir, mapper.naive_grid, None, mapper.adaptive_grid)

            if args.max_frames and n_received >= args.max_frames:
                print(f"[udp] досягнуто --max_frames {args.max_frames}, зупиняюсь")
                break
    except KeyboardInterrupt:
        print("\n[udp] зупинено (Ctrl+C)")
    finally:
        stop_rx.set()
        receiver.close()
        if pub:
            print(f"[udp] віддано {pub.bytes_sent/1e6:.1f} МБ")
            pub.close()

    if mapper is None:
        print("[udp] жодного кадру не отримано")
        return
    save_grids(args.output_dir, mapper.naive_grid, None, mapper.adaptive_grid)
    print(f"\n[udp] оброблено {n_received} кадрів, з картою довіри {n_with_conf}, "
          f"відкинуто як прострочені {dropped[0]}")
    if proc_times:
        t = np.array(proc_times) * 1000.0
        print(f"[udp] час на кадр: медіана {np.median(t):.0f} мс, 95-й перцентиль "
              f"{np.percentile(t, 95):.0f} мс, тобто межа близько "
              f"{1000.0/max(np.median(t), 1e-9):.1f} Гц")
    for k, v in mapper.summary().items():
        print(f"[udp]   {k}: {v}")
    print(f"[udp] збережено у {args.output_dir}")


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "dataset":
        run_dataset_mode(args)
    else:
        run_udp_mode(args)


if __name__ == "__main__":
    main()
