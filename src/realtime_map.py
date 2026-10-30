r"""
Побудова карти довіри в реальному часі, кадр за кадром, зі станом між кадрами.

Це та сама обробка, що й у `occupancy_3d_map.run_dataset_mode`, винесена в
клас, який живе між кадрами. Усі важкі кроки викликають ТІ САМІ функції, що й
офлайновий конвеєр, а не їхні копії:

    depth_refine.apply_refiner          таблиця корекції глибини
    sky_cue.structure_mask              вето неба з зображення
    blade_mask                          маска гвинтів
    depth_calibration.LocalCalibrator   локальне калібрування по плитках
    occupancy_3d_map.backproject_with_weight_map
    occupancy_3d_map.ProbabilisticVoxelGrid.insert_weighted

Що саме означає «з урахуванням попередніх кадрів». Три окремі механізми, усі
тримаються в цьому класі і переживають кадр:

  1. `LocalCalibrator.residual_scale` і `bandwidth_scale`. Це поточні типові
     похибка підгонки і щільність опорних точок, згладжені експоненційно з
     періодом напівзгасання 30 оновлень. Вага довіри рахується ВІДНОСНО них,
     тобто відносно того, що цей політ сам про себе виміряв. На першому кадрі
     вони ще не прогріті, тому перші кадри навмисно менш певні.
  2. Накопичення в логарифмі шансів. Кожен кадр додає свідчення до того, що
     вже є у вокселі, з обмеженням від мінус 4 до 4.
  3. Вага узгодженості. Спостереження, що суперечить накопиченому стану,
     придушується тим сильніше, чим впевненіший був той стан, а поріг
     «впевненості» теж самоналаштовний.

Чого цей клас НЕ робить. Він не запускає VSLAM і не вирівнює системи координат.
Відправник зобовʼязаний давати пози і розріджені точки в одній світовій системі
та в метрах. Монокулярний VSLAM свого масштабу не має, тож на борту його дає
інерціальний модуль або висотомір, а в офлайновому експерименті він брався з
одноразового вирівнювання Sim(3) до еталона.

Без розріджених точок локальне калібрування неможливе. Такий кадр іде лише в
базову карту, а в ймовірнісну не потрапляє зовсім, і це видно у статистиці як
`frames_no_calib`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_calibration as depth_calib
import depth_refine
import blade_mask as bm
import sky_cue
from occupancy_3d_map import (ProbabilisticVoxelGrid, backproject_with_weight_map,
                               naive_grid_l_occ)


class RealtimeConfidenceMapper:
    """Один кадр усередину, карта довіри і приріст воксельної карти назовні.

    Параметри навмисно збігаються з однойменними прапорцями
    `occupancy_3d_map.py`, щоб офлайн і реальний час давали те саме."""

    def __init__(self, bounds_min, bounds_max, voxel_resolution,
                 refiner_path=None, refiner_min_validity=0.5,
                 blade_mask_path=None, blade_dataset_name=None,
                 use_sky_cue=False, sky_cue_percentile=70.0,
                 calib_tile_px=40, calib_k_neighbors=12, calib_min_anchors_total=8,
                 calib_model="scale", max_anchor_gap=0.0,
                 pose_drift_rate=0.005, min_depth=0.1, max_depth=50.0,
                 max_points_per_frame=4000, n_ray_samples=4,
                 splat=False, splat_k_sigma=1.5, splat_max_radius_voxels=4,
                 consistency_tau=2.0, use_consistency=True,
                 build_naive=True, seed=0):
        self.min_depth, self.max_depth = min_depth, max_depth
        self.max_points_per_frame = max_points_per_frame
        self.n_ray_samples = n_ray_samples
        self.splat = splat
        self.splat_k_sigma = splat_k_sigma
        self.splat_max_radius_voxels = splat_max_radius_voxels
        self.pose_drift_rate = pose_drift_rate
        self.max_anchor_gap = max_anchor_gap
        self.refiner_min_validity = refiner_min_validity
        self.use_sky_cue = use_sky_cue
        self.sky_cue_percentile = sky_cue_percentile
        self.rng = np.random.default_rng(seed)

        self.refiner = depth_refine.load_refiner(refiner_path) if refiner_path else None
        self.blade_mask_path = blade_mask_path
        self.blade_dataset_name = blade_dataset_name
        self._blades = None          # вантажиться на першому кадрі, коли відомий розмір

        self.local_calib = depth_calib.LocalCalibrator(
            tile_px=calib_tile_px, k_neighbors=calib_k_neighbors,
            min_anchors_total=calib_min_anchors_total, model=calib_model)

        self.adaptive_grid = ProbabilisticVoxelGrid(
            bounds_min, bounds_max, voxel_resolution,
            consistency_tau=depth_calib.RunningScale(init=consistency_tau, half_life=30),
            use_consistency=use_consistency)
        self.naive_grid = (ProbabilisticVoxelGrid(bounds_min, bounds_max, voxel_resolution)
                           if build_naive else None)

        self.cum_dist = 0.0
        self._prev_origin = None
        self.n_frames = 0
        self.frames_no_calib = 0
        self.frames_no_landmarks = 0

    # ------------------------------------------------------------------ службове

    def _blade_mask_for(self, shape):
        if self._blades is not None or not self.blade_mask_path:
            return self._blades
        if not os.path.exists(self.blade_mask_path):
            return None
        m = bm.load_mask(self.blade_mask_path, shape=shape,
                          dataset_name=self.blade_dataset_name)
        self._blades = m if m.sum() else None
        return self._blades

    def _advance_distance(self, origin):
        """Пройдений шлях за ОЦІНЕНИМИ позами, як у офлайновому режимі: він
        задає член невизначеності від дрейфу VSLAM."""
        if self._prev_origin is not None:
            self.cum_dist += float(np.linalg.norm(origin - self._prev_origin))
        self._prev_origin = np.array(origin, dtype=np.float64)

    # -------------------------------------------------------------- один кадр

    def _why_no_anchors(self, landmarks, T_wc, K, w, h):
        """Чому жодна точка VSLAM не дійшла до кадру. Кличеться лише тоді,
        коли опорних точок нуль, тому на швидкодію не впливає.

        Розбиває втрати на чотири причини, які виключають одна одну, у тому
        самому порядку, у якому їх відсіює depth_calib.project_points:
        позаду камери, ближче за min_depth, далі за max_depth, за межами
        кадру. Та, у якій майже всі точки, і є справжня причина."""
        p = np.asarray(landmarks, dtype=np.float64)
        n = len(p)
        if n == 0:
            return "точок не було взагалі"
        cam = (np.linalg.inv(T_wc) @ np.concatenate(
            [p, np.ones((n, 1))], axis=1).T).T[:, :3]
        z = cam[:, 2]
        behind = z <= 0
        near = (~behind) & (z <= self.min_depth)
        far = (~behind) & (z >= self.max_depth)
        live = (~behind) & (~near) & (~far)
        out_of_frame = np.zeros(n, dtype=bool)
        if live.any():
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u = fx * cam[live, 0] / z[live] + cx
            v = fy * cam[live, 1] / z[live] + cy
            out_of_frame[live] = (u < 0) | (u > w - 1) | (v < 0) | (v > h - 1)
        parts = [f"позаду камери {int(behind.sum())}",
                 f"ближче за {self.min_depth:g} м {int(near.sum())}",
                 f"далі за {self.max_depth:g} м {int(far.sum())}",
                 f"за межами кадру {int(out_of_frame.sum())}"]
        zq = np.percentile(z, [5, 50, 95]) if n else [0, 0, 0]
        return (f"з {n} точок: " + ", ".join(parts) +
                f". Глибина точок у камері: 5% {zq[0]:.2f}, медіана {zq[1]:.2f}, "
                f"95% {zq[2]:.2f} (одиниці VSLAM, не обов'язково метри)")

    def process(self, depth, T_wc, K, image=None, landmarks=None, pred_depth_scale=1.0):
        """-> словник з картою довіри, відкаліброваною глибиною, точками і
        приростом воксельної карти. Жодних побічних записів на диск."""
        depth = np.asarray(depth, dtype=np.float64) * pred_depth_scale
        T_wc = np.asarray(T_wc, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        origin = T_wc[:3, 3]
        self._advance_distance(origin)
        self.n_frames += 1

        blades = self._blade_mask_for(depth.shape)
        raw_depth = depth.copy()

        # 1. таблиця корекції глибини
        refine_keep = None
        if self.refiner is not None:
            depth, refine_keep = depth_refine.apply_refiner(
                self.refiner, depth, min_validity=self.refiner_min_validity)

        # 2. вето неба з зображення, скасовує помилки таблиці на горизонті
        if self.use_sky_cue and image is not None:
            grey = np.asarray(image, dtype=np.float32)
            if grey.max() > 1.5:
                grey = grey / 255.0
            not_sky = sky_cue.structure_mask(grey, grad_percentile=self.sky_cue_percentile)
            refine_keep = not_sky if refine_keep is None else (refine_keep | not_sky)

        out = {"n_frames": self.n_frames, "cum_dist": self.cum_dist,
               "raw_depth": raw_depth, "depth_after_table": depth,
               "blades": blades, "refine_keep": refine_keep}

        # 3. базова карта: ті самі пікселі, але без ваг і без калібрування
        if self.naive_grid is not None:
            naive_wm = np.ones_like(depth)
            if blades is not None:
                naive_wm = np.where(blades, np.nan, naive_wm)
            if refine_keep is not None:
                naive_wm = np.where(refine_keep, naive_wm, np.nan)
            pts_n, w_n = backproject_with_weight_map(
                depth, naive_wm, K, T_wc, self.min_depth, self.max_depth,
                max_points=self.max_points_per_frame, rng=self.rng)
            self.naive_grid.insert_weighted(pts_n, w_n, origin,
                                             n_ray_samples=self.n_ray_samples)
            out["naive_points"] = len(pts_n)

        # 4. локальне калібрування по розріджених точках VSLAM
        if landmarks is None or len(landmarks) == 0:
            self.frames_no_landmarks += 1
            out["confidence"] = None
            out["reason"] = ("кадр без розріджених точок VSLAM: калібрування і ваги довіри "
                              "неможливі, у ймовірнісну карту кадр не йде")
            return out

        h, w = depth.shape
        au, av, az = depth_calib.project_points(np.asarray(landmarks, dtype=np.float64),
                                                 T_wc, K, w, h, self.min_depth, self.max_depth)
        result = self.local_calib.calibrate_frame(au, av, az, depth)
        if result is None:
            self.frames_no_calib += 1
            out["confidence"] = None
            out["anchors_in_view"] = int(len(au))
            out["reason"] = (f"у кадрі лише {len(au)} опорних точок, менше за мінімум "
                              f"{self.local_calib.min_anchors_total}: калібрування пропущено")
            if len(au) == 0:
                out["reason"] += ". " + self._why_no_anchors(landmarks, T_wc, K, w, h)
            return out

        alpha_map, beta_map, conf_map, sigma_depth_map, support_map = result

        # 5. усі причини відкинути піксель в одну маску, застосовану ОСТАННЬОЮ
        drop = np.zeros(depth.shape, dtype=bool)
        if blades is not None:
            drop |= blades
        if refine_keep is not None:
            drop |= ~refine_keep
        if self.max_anchor_gap > 0:
            drop |= support_map > self.max_anchor_gap

        sigma_pose = self.pose_drift_rate * self.cum_dist
        sigma_map = sigma_depth_map + sigma_pose
        conf_map = np.where(drop, np.nan, conf_map)
        corrected = np.clip(alpha_map * depth + beta_map, 0.0, None)

        # 6. у воксельну карту, з приростом саме цього кадру
        pts_a, w_a, sig_a = backproject_with_weight_map(
            corrected, conf_map, K, T_wc, self.min_depth, self.max_depth,
            max_points=self.max_points_per_frame, rng=self.rng, extra_maps=(sigma_map,))

        # track_changes замість копіювання і порівняння всього масиву: при
        # сітці на десятки мільйонів вокселів різниця між мілісекундами і
        # сотнями мілісекунд на кадр.
        self.adaptive_grid.insert_weighted(
            pts_a, w_a, origin, sigma=sig_a, n_ray_samples=self.n_ray_samples,
            splat_k_sigma=self.splat_k_sigma,
            splat_max_radius_voxels=self.splat_max_radius_voxels,
            splat_occupied=self.splat, track_changes=True)
        touched = self.adaptive_grid.last_touched
        changed = (np.stack(np.unravel_index(touched, self.adaptive_grid.dims), axis=1)
                   if len(touched) else np.zeros((0, 3), dtype=np.int64))

        out.update({
            "confidence": conf_map,
            "depth_calibrated": corrected,
            "alpha": alpha_map, "beta": beta_map,
            "sigma_depth": sigma_depth_map, "sigma_total": sigma_map,
            "sigma_pose": sigma_pose, "support": support_map, "dropped": drop,
            "anchors_in_view": int(len(au)),
            "points": pts_a, "weights": w_a, "sigma": sig_a,
            "delta_idx": changed.astype(np.int32),
            "delta_values": self.adaptive_grid.log_odds[tuple(changed.T)].astype(np.float32)
            if len(changed) else np.zeros(0, dtype=np.float32),
            "residual_scale": float(self.local_calib.residual_scale.value),
            "bandwidth_scale": float(self.local_calib.bandwidth_scale.value),
            "consistency_tau": float(self.adaptive_grid.consistency_tau.value),
            "mean_weight": float(np.nanmean(np.where(np.isfinite(conf_map), conf_map, np.nan)))
            if np.isfinite(conf_map).any() else float("nan"),
            "kept_fraction": float(np.mean(~drop)),
        })
        return out

    def summary(self):
        return {"frames": self.n_frames,
                "frames_no_landmarks": self.frames_no_landmarks,
                "frames_no_calib": self.frames_no_calib,
                "calibrated": self.local_calib.n_frames_calibrated,
                "residual_scale_m": round(float(self.local_calib.residual_scale.value), 3),
                "bandwidth_scale_px": round(float(self.local_calib.bandwidth_scale.value), 2),
                "cum_dist_m": round(self.cum_dist, 2),
                "adaptive_occupied": int(self.adaptive_grid.occupied_mask().sum()),
                "naive_occupied": (int(self.naive_grid.occupied_mask().sum())
                                   if self.naive_grid is not None else None)}
