r"""
Running the trained refinement network inside the mapping pipeline (stage 3.2).

Bridges two mismatches between training and mapping:

  resolution  the network works at 320x240; frames are 640x480, so inputs go
              down and the three output fields come back up.
  sparse input at training time the sparse channel held synthetic points drawn
              from ground truth; here it holds the REAL VSLAM landmarks
              projected into the frame. The scale normalisation is the same
              either way -- divide by the median of whatever sparse depths this
              frame actually has -- which is exactly why the network was
              trained on ratios.

Safety net, in the order the mapping pipeline applies it:

  1. FDE depth -> network -> refined depth, sigma_net, p_surface
  2. drop pixels with p_surface < 0.5 (sky the network recognises)
  3. the existing per-tile VSLAM calibration still runs on the refined depth
  4. if the refined depth disagrees with the VSLAM landmarks by more than
     `k * S` (S = the run's own running residual scale), the frame is fused
     WITHOUT the network -- a bad extrapolation on an unfamiliar scene falls
     back to the statistical path instead of poisoning the map.

Per-pixel weight combines both uncertainties:

    sigma_total = sqrt(sigma_net^2 + sigma_calib^2),   w = 1 / (1 + sigma_total / S)
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NET_H, NET_W = 240, 320


def _to_net(a, mode=Image.BILINEAR):
    return np.array(Image.fromarray(a.astype(np.float32)).resize((NET_W, NET_H), mode))


def _from_net(a, h, w, mode=Image.BILINEAR):
    return np.array(Image.fromarray(a.astype(np.float32)).resize((w, h), mode))


def _sobel(grey):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    gx = sum(kx[i, j] * np.roll(np.roll(grey, 1 - i, 0), 1 - j, 1)
             for i in range(3) for j in range(3) if kx[i, j])
    gy = sum(kx[j, i] * np.roll(np.roll(grey, 1 - i, 0), 1 - j, 1)
             for i in range(3) for j in range(3) if kx[j, i])
    return np.sqrt(gx ** 2 + gy ** 2)


class NetRefiner:
    """Loads a depth_net.py checkpoint and applies it to full-resolution frames."""

    def __init__(self, model_path, device=None, min_anchors=8):
        import torch
        import depth_net
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net, self.ck = depth_net.load_net(model_path, self.device)
        self.predict = depth_net.predict
        self.min_anchors = min_anchors
        self.n_applied = 0
        self.n_skipped_anchors = 0
        self.n_rejected_disagreement = 0

    def __repr__(self):
        return (f"NetRefiner(fde={self.ck['fde']}, holdout={self.ck['holdout']}, "
                f"params={self.ck['params']/1e6:.2f}M, device={self.device})")

    def refine(self, pred_depth, grey, anchor_u, anchor_v, anchor_z, min_depth):
        """-> (refined_depth, sigma_m, p_surface) at the frame's own resolution,
        or None when there is not enough sparse support to normalise by.

        sigma_m is the network's uncertainty converted from log-depth to metres
        (sigma_log * depth), so it can be combined with the calibration residual
        on the same scale."""
        if len(anchor_u) < self.min_anchors:
            self.n_skipped_anchors += 1
            return None
        h, w = pred_depth.shape
        divisor = float(np.median(anchor_z))
        if not np.isfinite(divisor) or divisor <= 1e-6:
            self.n_skipped_anchors += 1
            return None

        sparse_full = np.zeros((h, w), dtype=np.float32)
        smask_full = np.zeros((h, w), dtype=bool)
        uu = np.clip(np.round(anchor_u).astype(int), 0, w - 1)
        vv = np.clip(np.round(anchor_v).astype(int), 0, h - 1)
        sparse_full[vv, uu] = anchor_z
        smask_full[vv, uu] = True

        # Down to the network's grid. The sparse mask is scattered again at the
        # smaller size rather than resampled, so isolated points survive.
        fde_ok = np.isfinite(pred_depth) & (pred_depth > min_depth)
        fde_log = np.zeros((NET_H, NET_W), dtype=np.float32)
        tmp = np.zeros_like(pred_depth, dtype=np.float32)
        tmp[fde_ok] = np.log(np.maximum(pred_depth[fde_ok] / divisor, 1e-6))
        fde_log = _to_net(tmp)

        sx = NET_W / float(w)
        sy = NET_H / float(h)
        su = np.clip((uu * sx).astype(int), 0, NET_W - 1)
        sv = np.clip((vv * sy).astype(int), 0, NET_H - 1)
        sparse_log = np.zeros((NET_H, NET_W), dtype=np.float32)
        smask = np.zeros((NET_H, NET_W), dtype=np.float32)
        sparse_log[sv, su] = np.log(np.maximum(anchor_z / divisor, 1e-6))
        smask[sv, su] = 1.0

        grey_n = _to_net(grey)
        grad_n = _sobel(grey_n)

        delta, sigma_log, p_surf = self.predict(self.net, self.ck, fde_log, sparse_log,
                                                 smask, grey_n, grad_n, self.device)
        delta_f = _from_net(delta, h, w)
        sigma_f = _from_net(sigma_log, h, w)
        p_surf_f = _from_net(p_surf, h, w)

        refined = np.where(fde_ok, pred_depth * np.exp(delta_f), pred_depth)
        self.n_applied += 1
        return refined, np.abs(sigma_f) * np.maximum(refined, 0.0), p_surf_f

    def disagrees_with_anchors(self, refined, anchor_u, anchor_v, anchor_z, scale, k):
        """True when the refined depth is further from the VSLAM landmarks than
        k * scale, in which case the caller should fuse this frame without the
        network."""
        if len(anchor_u) < self.min_anchors or not np.isfinite(scale) or scale <= 0:
            return False
        h, w = refined.shape
        uu = np.clip(np.round(anchor_u).astype(int), 0, w - 1)
        vv = np.clip(np.round(anchor_v).astype(int), 0, h - 1)
        resid = float(np.median(np.abs(refined[vv, uu] - anchor_z)))
        if resid > k * scale:
            self.n_rejected_disagreement += 1
            return True
        return False

    def summary(self):
        return {"applied": self.n_applied,
                "skipped_no_anchors": self.n_skipped_anchors,
                "rejected_disagreement": self.n_rejected_disagreement}
