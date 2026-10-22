# Probabilistic 3D occupancy mapping from VSLAM + monocular depth

Current method, its formulas, its parameters, and the measurements that decided
each choice. Everything below is implemented in `src/` and reproducible with the
commands at the end.

---

## 1. Problem

Given a monocular VSLAM trajectory (DPVO or ORB-SLAM3) and a per-frame dense
depth map from a feed-forward depth estimator (ZoeDepth, Depth-Anything-V2,
FastDepth), build a voxel grid whose cells carry a **probability of being
occupied**, such that the probability is high where obstacles are and low where
they are not — closely enough to ground truth to be usable as a safety layer,
and without being tuned to any one dataset.

The intermediate probabilistic layer is supposed to absorb the errors of both
inputs. Section 7 quantifies how much of that it actually achieves, and where
the limit is.

---

## 2. Coordinate frames — the single largest correction

Ground truth in these datasets stores **body** poses, not camera poses, even
though `cam0/sensor.yaml` declares `T_BS = I`. A Sim(3) trajectory alignment
matches *positions* and therefore cannot absorb this: ATE looks healthy while
every orientation is wrong by a constant rotation.

Let `C` be the camera→body rotation, recovered as

$$C = \arg\min_{P \in \mathcal{P}_{24}} \operatorname{median}_i \angle\!\left(R^{\text{est}\top}_i R^{\text{gt}}_i,\; P\right)$$

where $\mathcal{P}_{24}$ is the set of the 24 signed axis-permutation rotations.
Snapping to that discrete set makes this a *convention choice*, not a fitted
parameter. Measured on EuRoC: the offset is **119.94° → 0.33° residual**, and it
is **identical for DPVO and ORB-SLAM3**, which is what identifies it as a
property of the dataset rather than of an algorithm.

The convention is now established by `gt_frame.py` from the ground truth
ALONE — the rotation under which GT depth clouds from two nearby frames land on
top of each other — so it works on the sequences where the VSLAM trajectory has
diverged too (run1..run6, DPVO ATE 112-180 m). Its answer for EuRoC is exactly
the transpose of what the independent VSLAM-agreement estimate gave, i.e. two
unrelated criteria agree.

Which side is wrong was also settled by an independent arbiter — DPVO's own
triangulated landmarks, which must lie on real surfaces:

| interpretation | median distance, landmark → true surface |
|---|---|
| GT taken as camera poses | 8.93 m |
| GT taken as body poses, converted with $C^\top$ | **0.35 m** (p25 = 0.13 m) |

So the correction belongs on the **GT side**: $R^{\text{cam}}_{\text{gt}} = R_{\text{gt}} C^\top$.
Applying it to the VSLAM poses instead makes both sides self-consistent but
leaves the whole map rotated, and puts the landmark cloud in a different frame
from the cameras.

**Effect on every reported metric:**

| metric | before | after |
|---|---|---|
| ATE (DPVO) | 0.575 m | 0.575 m (unchanged — positions only) |
| RTE (DPVO) | 0.136 m | **0.022 m** |
| RRE (DPVO) | 0.346° | **0.115°** |
| RTE (ORB-SLAM3) | 0.159 m | **0.075 m** |
| RRE (ORB-SLAM3) | 0.369° | **0.134°** |
| occupancy IoU, GT depth + est. pose | 0.0005 | **0.586** |
| occupancy IoU, real pipeline | 0.018 | **0.169** |

RTE/RRE change because a relative pose becomes a conjugation
$\Delta P = C^{-1} \Delta Q\, C$; for pure translation the residual is
$\lVert C^{-1} t - t \rVert$, i.e. a fixed fraction of the per-frame travel,
previously reported as if it were VSLAM error.

Implemented in `euroc_io.estimate_axis_convention_fix` / `body_to_camera`, used
by `occupancy_3d_map.py`, `evaluation.py`, `measure_metrics.py`,
`compare_reconstruction_3d.py`, `error_budget.py`.

---

## 3. Metric scale

Monocular VSLAM has no intrinsic scale. The one remaining use of ground truth
inside the pipeline is a single Umeyama Sim(3) fit to recover it:
$(R_a, t_a, s_a)$, applied as $p_w = R_a (s_a p) + t_a$. On a real platform this
is what visual-inertial fusion would supply. Everything downstream of it is
GT-free.

---

## 4. Self-configuring scene parameters

No range or resolution is given in metres by hand.

**Usable range** from triangulation geometry. Depth uncertainty from
triangulation is $\sigma_z \approx z^2 \sigma_{\text{pix}} / (f B)$, so requiring
a relative accuracy $\tau$ gives

$$z_{\max} = \tau \, f \, B / \sigma_{\text{pix}}, \qquad B = \text{median}_i \lVert t_{i+1}-t_i \rVert \cdot N_B$$

with $f$ from the intrinsics and $B$ from the camera's own measured motion.
Defaults: $\tau = 0.1$, $N_B = 10$ frames, $\sigma_{\text{pix}} = 1$ px.
Measured: **51.6 m on EuRoC** — which independently reproduces the 50 m that had
previously been guessed by hand — and 473 m / 611 m on the much faster, larger
run1 / run5.

A percentile of the landmark cloud was tried first and **rejected**: monocular
clouds carry a heavy tail of near-zero-parallax points (median landmark depth
121 m with a tail past 20 km on a 227 m scene), so the cloud reports its own
garbage back as "range".

**Voxel size** from the uncertainty the run measures for itself:

$$\Delta = \kappa \cdot \widehat{\sigma}_{\text{resid}}, \qquad \kappa = 1.0$$

Quantising an input ~10× finer than its own precision is what made early maps
look like noise: IoU rose monotonically 0.0128 → 0.0615 as the voxel grew
0.5 m → 4 m on data with ~5 m depth residual. A warning fires when
$\Delta > \text{scene span}/20$, which correctly identifies datasets this depth
source cannot map at all.

---

## 5. Local metric recalibration of the depth (`depth_calibration.py`)

Per frame, the dense depth is corrected against the VSLAM's own sparse
landmarks. Landmarks are projected into the frame to give anchors
$(u_k, v_k, z^{\text{ref}}_k)$, and the FDE depth is sampled bilinearly at the
same pixels to give $z^{\text{pred}}_k$.

The image is divided into tiles of `tile_px = 40` px. For tile centre $c$, its
$k = 12$ nearest anchors get Gaussian weights with a **kNN bandwidth**

$$w_k = \exp\!\left(-\tfrac{1}{2}\left(\frac{d_k}{h_c}\right)^{2}\right), \qquad h_c = d_{(k)}$$

i.e. the bandwidth is the distance to the $k$-th neighbour, so sparse regions
widen the kernel automatically and no metric-space radius is ever specified.

**Two passes.** Pass 1 fits each tile independently:

$$(\alpha_c, \beta_c) = \arg\min \sum_k w_k \rho\!\left(z^{\text{ref}}_k - \alpha z^{\text{pred}}_k - \beta\right)$$

with $\rho$ a Huber reweighting (2 IRLS iterations, MAD scale). A robust
frame-level prior is then the weighted median across tiles,
$\bar\alpha = \operatorname{median}_w(\alpha_c)$, $\bar\beta = \operatorname{median}_w(\beta_c)$.
Pass 2 refits every tile with ridge shrinkage toward that prior:

$$(\alpha_c, \beta_c) = \arg\min \sum_k w_k \rho(\cdot) + \lambda\left[(\alpha-\bar\alpha)^2 + (\beta-\bar\beta)^2\right], \qquad \lambda = k/3$$

$\lambda$ is a pseudo-count, so a tile with plenty of its own anchors ignores the
prior and a thin one defers to it. This two-stage form exists because both
extremes were measured to fail: a single global fit is dominated by the
far near-zero-parallax anchors (anchor RMSE 4.8 m vs 0.45 m once they are
excluded — which previously required a hand-picked range cutoff), while
purely local fits lose frame-wide bias where anchors are thin
(Depth-Anything-V2: IoU 0.416 → 0.388 with a naive single global prior,
recovered to 0.583 by the robust median-of-tiles prior).

Per-tile uncertainty and confidence:

$$\sigma_c = \text{rmse}_c\left(1 + \frac{h_c}{\widehat{h}}\right), \qquad
\text{conf}_c = \frac{1}{1 + \sigma_c / \widehat{\sigma}_{\text{resid}}}$$

$\widehat{h}$ and $\widehat{\sigma}_{\text{resid}}$ are **online** robust scale
estimates (`RunningScale`, EMA with a half-life of 30 updates), which is what
replaces the old fixed constants `w_ref = 1.0 m` and `consistency_tau = 2.0`.
Measured self-calibration across scenes without any intervention:
$\widehat{\sigma}_{\text{resid}} =$ 5.19 (EuRoC), 8.49 (run1), 9.29 (run5).

All four fields $(\alpha, \beta, \text{conf}, \sigma)$ are computed per tile and
bilinearly upsampled to full resolution. Corrected depth:
$z' = \max(0,\ \alpha(u,v)\, z + \beta(u,v))$.

---

## 6. Learned in-domain depth correction (`depth_refine.py`)

The local calibration fixes *scale*; it cannot fix a network being out of
domain. Two failure modes were measured:

1. error growing with range (2.5 m at $z$ = 1.8 m up to 25 m at $z$ = 48 m for ZoeDepth);
2. **sky hallucination** — 24–48 % of every frame has no return in GT, and
   ZoeDepth reports an in-range surface for **100 %** of those pixels
   (Depth-Anything-V2: 42–79 %).

A non-parametric table over (log-spaced predicted-depth bin) × (image-row bin),
12 × 8 cells, stores per cell

$$r_{ij} = \operatorname{median}\left(z^{\text{gt}}/z^{\text{pred}}\right) \in [0.25, 4], \qquad
v_{ij} = \Pr\left(\text{a surface exists} \mid \text{network claims one}\right)$$

Inference: $z' = r_{ij} z$, and pixels with $v_{ij} < 0.5$ are **dropped**, not
merely down-weighted (at a read-out of $P>0.5$ any positive weight still marks a
voxel occupied). Both axes are normalised — depth relative to the model's own
range, row as a fraction of image height — so a table transfers across
resolutions and scene scales.

What it learned, unprompted: when ZoeDepth claims a surface at 20–35 m in the
**upper** part of the image, one exists only **5–17 %** of the time; at the
bottom of the image (ground) it is 0.90–1.00 at every range. That is sky,
identified from data. An earlier attempt to infer sky from sparse-landmark
sparsity was measured to *hurt* (IoU 0.0233 → 0.0088) because DPVO's cloud
projects degenerate far points into the sky region.

Ground truth is used to **train** the table and never at inference — the same
status it has in the depth network's own training. All reported numbers use
**leave-one-out** tables: `refiners/loo_<dataset>_<fde>.npz` is fitted on the
other six sequences.

---

## 7. Fusion into log-odds (`occupancy_3d_map.py`)

Standard inverse-sensor-model accumulation, $L \leftarrow \operatorname{clip}(L + \ell w, L_{\min}, L_{\max})$
with $\ell_{\text{occ}} = +0.85$, $\ell_{\text{free}} = -0.4$,
$L \in [-4, 4]$, $P = \sigma(L)$, and three additions:

**Confidence weight.** $w = \text{conf}(u,v)$ from §5 for `adaptive_grid`;
$w \equiv 1$ for `naive_grid`.

**Consistency weighting.** An update that contradicts an already-confident cell
is suppressed:

$$w \leftarrow w \cdot \begin{cases} \dfrac{1}{1 + |L_{\text{prior}}|/\tau} & \text{if } \operatorname{sign}(L_{\text{prior}}) \neq s \\ 1 & \text{otherwise}\end{cases}$$

with $s = \pm 1$ for occupied/free and $\tau$ an online `RunningScale` of the
magnitude of decided cells, not a constant.

**Uncertainty-limited free space.** Rays are sampled and deduplicated to one
update per voxel per ray, and stopped short of the endpoint by the point's own
uncertainty:

$$t_{\max} = \operatorname{clip}\left(\frac{\lVert r \rVert - k_\sigma \sigma}{\lVert r \rVert},\ 0,\ 0.92\right), \qquad k_\sigma = 1.5$$

so a point too uncertain to place carves nothing. Positional uncertainty combines
the depth-side residual with a pose term
$\sigma = \sigma_{\text{depth}}(u,v) + \eta \cdot d_{\text{travelled}}$,
$\eta = 0.005$ (a literature-typical monocular drift rate, sanity-checked
against this run's own measured ~0.25 %, not fitted to it).

### Measured dead ends, kept as off-by-default flags with their numbers

| idea | flag | measured |
|---|---|---|
| Gaussian splatting of occupied evidence | `--splat` | IoU 0.42 → 0.18 on clean input — pure dilation at $P>0.5$ |
| dense ray traversal | `--n_ray_samples -1` | erases true surfaces as fast as phantoms while depth error is metres |
| higher occupancy threshold | `--threshold` sweep | on real data precision *falls* with threshold |
| sky rejection from anchor sparsity | `--max_anchor_gap` | IoU 0.0233 → 0.0088 |
| sensor model measured from self-consistency | `--measured_sensor_model` | $p_{\text{occ}}$ pegs at its 0.95 ceiling — a biased source agrees with itself; IoU 0.416 → 0.388 |
| per-frame landmark window | `--anchor_window_frames` | ±20 frames: 0.013; ±100: 0.120; whole cloud: 0.152; geometric auto (±458): 0.155 |

The recurring reason: the residual error is **systematic**, and neither
accumulation nor self-consistency can see systematic error.

---

## 8. Error budget (`error_budget.py`)

One input at a time replaced by ground truth, EuRoC, 1 m voxels:

| configuration | IoU | FPVR | precision | vs chance |
|---|---|---|---|---|
| GT depth + GT pose | 0.907 | 0.000 | 1.000 | 49× |
| GT depth + est. pose | 0.586 | 0.209 | 0.792 | 39× |
| FDE depth + GT pose | 0.106 | 0.855 | 0.146 | 7.2× |
| FDE depth + est. pose | 0.104 | 0.857 | 0.143 | 7.0× |

Pose drift costs ~35 % of the ceiling; **depth costs ~88 %**. Depth is the
bottleneck, which is what motivates §6.

`noise_ablation.py` isolates the layer's own contribution with controlled,
*known* error injected into GT depth: adaptive beats naive by **+11 %** (random
error), **+14 %** (systematic), **+16 %** (both), with lower FPVR in all three;
on clean input, reading the map at $P>0.93$ gives precision 0.966 / FPVR 0.034.
So the mechanism works — its benefit is simply invisible while the input error
is dominated by out-of-domain depth.

---

## 9. Results, EuRoC held out of the correction table

2×2 at **fixed** voxel sizes, with the pipeline's self-chosen size reported
separately (`results_2x2_euroc.csv`):

### IoU (higher is better)

| voxel | FDE | naive raw | naive refined | adaptive raw | **adaptive refined** | auto voxel |
|---|---|---|---|---|---|---|
| 1.0 m | ZoeDepth | 0.110 | 0.175 | 0.148 | **0.270** | 6.03 m |
| 1.0 m | Depth-Anything-V2 | 0.070 | 0.195 | 0.175 | **0.266** | 5.44 m |
| 1.0 m | FastDepth | 0.001 | 0.086 | 0.192 | **0.233** | 5.40 m |
| 0.5 m | ZoeDepth | 0.067 | 0.118 | 0.111 | **0.173** | 6.03 m |
| 0.5 m | Depth-Anything-V2 | 0.034 | 0.115 | 0.129 | **0.180** | 5.44 m |
| 0.5 m | FastDepth | 0.000 | 0.082 | 0.128 | **0.167** | 5.40 m |

### FPVR (lower is better)

| voxel | FDE | naive raw | naive refined | adaptive raw | **adaptive refined** |
|---|---|---|---|---|---|
| 1.0 m | ZoeDepth | 0.855 | 0.644 | 0.835 | **0.658** |
| 1.0 m | Depth-Anything-V2 | 0.916 | 0.700 | 0.803 | **0.661** |
| 1.0 m | FastDepth | 0.993 | 0.721 | 0.793 | **0.721** |

Both factors help and they compose; the adaptive layer contributes most where
the depth source is weakest (FastDepth: 0.001 → 0.192 from the layer alone).
At the pipeline's self-chosen voxel size the same configuration reaches
IoU 0.583 / FPVR 0.251 (Depth-Anything-V2), i.e. ~64 % of the 0.907 ceiling.

**run1–run6 cannot currently be mapped.** DPVO diverges on them (ATE 112–180 m,
and the orientation residual after axis snapping stays at 6–102°, so the
convention fix correctly refuses to apply); ORB-SLAM3 is usable positionally
(ATE 0.6–4.9 m) but has no landmark export yet, so no adaptive grid is possible.
Recovering those sequences is a VSLAM problem, not a mapping one.

---

## 10. Parameters

Measured or dimensionless — nothing governing the map is a hand-set metre value:

| parameter | default | nature |
|---|---|---|
| `voxel_resolution` | `auto` | = $\kappa\,\widehat{\sigma}_{\text{resid}}$, measured |
| `max_depth` | `auto` | = $\tau f B / \sigma_{\text{pix}}$, measured |
| `voxel_per_sigma` $\kappa$ | 1.0 | dimensionless |
| `rel_depth_tol` $\tau$ | 0.1 | dimensionless |
| `baseline_frames` $N_B$ | 10 | count |
| `sigma_pix` | 1.0 px | detector property |
| `calib_tile_px` | 40 px | image-space, scales with resolution |
| `calib_k_neighbors` $k$ | 12 | count |
| `calib_min_anchors_total` | 8 | count |
| prior strength $\lambda$ | $k/3$ | pseudo-count |
| `RunningScale` half-life | 30 | updates |
| `pose_drift_rate` $\eta$ | 0.005 | dimensionless |
| `splat_k_sigma` $k_\sigma$ | 1.5 | dimensionless |
| `n_ray_samples` | 4 | count |
| `refiner_min_validity` | 0.5 | probability |
| `min_depth` | 0.1 m | universal floor |
| `max_time_diff` | 0.15 s | recording clock offset, not scene |
| `gt_depth_scale` | 0.001 | file format, evaluation only |

---

## 10a. Дивитися на побудовану карту

```bash
cd ~/MasterThesis/src && source ../.venv/bin/activate

# карта (euroc, DPVO + ZoeDepth, з таблицею корекції, навченою БЕЗ euroc)
python3 occupancy_3d_map.py --mode dataset \
    --dataset_root ../datasets/euroc/2026-04-01_12-13-18_000_quadrotor \
    --vslam_dir ../results/euroc/dpvo --fde_dir ../results/euroc/zoedepth \
    --depth_refiner ../refiners/loo_euroc_zoedepth.npz \
    --output_dir ../maps/euroc/dpvo_zoedepth --voxel_resolution 1.0 --gt_compare

# перегляд: тепле = зайнято (колір за P), блакитне = впевнено вільно, зелене = траєкторія
python3 view_3d_occupancy_map.py ../maps/euroc/dpvo_zoedepth/adaptive_grid.npz \
    --show_free --threshold 0.0 \
    --trajectory ../datasets/euroc/2026-04-01_12-13-18_000_quadrotor/state_groundtruth_estimate0/data.csv
```

У WSL відкривати через `explorer.exe maps\euroc\dpvo_zoedepth\adaptive_grid.html`.

Порівняння, які показують внесок кожної частини: `naive_grid.npz` замість `adaptive_grid.npz`
(без ваг довіри), прогін без `--depth_refiner` (без навченої корекції),
`--voxel_resolution auto` (розмір за виміряною похибкою), `--threshold 1.7` у візуалізаторі
(лише P > 0.85, де precision 0.565 замість 0.349).

Приклад на euroc, ZoeDepth, таблиця з виключенням euroc:

| воксель | зайнятих | впевнено вільних | IoU | FPVR | precision | recall |
|---|---|---|---|---|---|---|
| 1.0 м | 31172 | 4401 | 0.2642 | 0.651 | 0.349 | 0.521 |
| 6.85 м (auto) | 748 | 84 | **0.4695** | **0.393** | **0.607** | **0.675** |

---

## 11. Reproducing

```bash
# trajectory metrics + comparison plots (GT re-expressed in the camera frame)
python3 compare_reconstruction_3d.py --dataset_root <ds> --vslam_dir <vslam> \
    --fde_dir <fde> --max_time_diff 0.15 --output <out.html> --no_open

# leave-one-out correction tables
python3 depth_refine.py train --dataset_roots <all but held-out> \
    --results_root ../results --fde_subdir zoedepth --output ../refiners/loo_<ds>_zoedepth.npz

# one map (auto range + auto voxel), with the held-out correction table
python3 occupancy_3d_map.py --mode dataset --dataset_root <ds> --vslam_dir <vslam> \
    --fde_dir <fde> --depth_refiner ../refiners/loo_<ds>_<fde>.npz \
    --output_dir ../maps/<ds>/<combo> --gt_compare

# the 2x2 table at fixed voxel sizes
python3 results_table.py --datasets euroc --vslam dpvo --output ../results_2x2_euroc.csv

# where the error comes from
python3 error_budget.py --dataset_root <ds> --vslam_dir <vslam> --fde_dir <fde>

# the layer's own contribution, controlled error
python3 noise_ablation.py --dataset_root <ds> --error_kind both --depth_noise 0.25 --pose_noise 0.2
```

## 12. Open

1. **Depth is the bottleneck** (88 % of the gap). The table in §6 is a
   non-parametric first step; fine-tuning a depth network on these sequences,
   or replacing single-image depth with multi-view/MVS, is the lever that moves
   0.58 toward 0.9.
2. **IMU** (`imu0/` exists in every dataset, unused): mono-inertial VSLAM would
   remove the last use of ground truth inside the pipeline — metric scale.
3. **ORB-SLAM3 landmark export**: patched (`src/slam/orbslam3/patch_landmarks_export.py`,
   verified against upstream source) but needs a Docker rebuild. It would give
   per-frame anchors instead of one merged cloud, and make the sky gate of §7
   honest.
4. **run1–run6 VSLAM divergence** must be fixed before those sequences can
   contribute map results.


---

# Журнал

## 2026-09-30 — ЕТАП 0: передумови

### Змінено

| файл | що |
|---|---|
| `src/gt_frame.py` | **новий.** Визначає поворот корпус→камера з узгодженості **самого еталона** (хмари глибини сусідніх кадрів мають накладатися), перебором 24 знакових перестановок осей. Не залежить від VSLAM, тому працює й там, де DPVO розійшовся. Кеш `masks/gt_frame.json`, функція `resolve(dataset_root)` |
| `src/blade_mask.py` | **новий.** Маска лопатей з даних: піксель у масці, якщо еталонна глибина < `--near_m` у ≥ `--min_fraction` кадрів, комбінування по наборах через мінімум |
| `src/compare_reconstruction_3d.py` | еталон переводиться в систему камери (`body_to_camera`), джерело повороту — кеш `gt_frame`, оцінка «GT проти VSLAM» лишилась як запас |
| `src/occupancy_3d_map.py` | конвенція з кешу `gt_frame`; у `--gt_compare` додано precision і recall |
| `src/measure_metrics.py` | конвенція з кешу `gt_frame`; додано precision і recall у вивід і CSV |
| `src/results_table.py` | додано precision і recall; фіксовані вокселі 1.0 і 0.5 м, автоматичний розмір окремою колонкою |
| `src/euroc_io.py` | повернуто `rotmat_to_quat`, додано `body_to_camera`; прибрано мертвий код (`estimate_world_rotation_fix`, `apply_world_rotation`) |
| `src/depth_calibration.py` | додано `_weighted_scale` — модель `z' = s·z` з медіанним `s`, як погоджено в `chat_context` 7.2. Новий прапорець `--calib_model {scale,affine}`, дефолт `scale` |
| `docs/chat_context.md` | перенесено з кореня, щоб працювало посилання `@docs/chat_context.md` з `CLAUDE.md` |

### Команди

```bash
python3 gt_frame.py --all
python3 blade_mask.py --output ../masks/blades_airsim.npz
# для кожного з 7 наборів і 6 пар:
python3 compare_reconstruction_3d.py --dataset_root <ds> --vslam_dir <v> --fde_dir <f>     --max_time_diff 0.15 --output ../comparisons/<run>/<v>_<f>.html --no_open
```

### 0.1 Конвенція еталона, визначена з самого еталона

| набір | розбіжність сусідніх кадрів (одинична → найкраща) | точки над камерою | R_bc |
|---|---|---|---|
| euroc | 0.25 → **0.19 м** (1.4×) | +9.9 → **−2.8 м** | `[[0,0,1],[-1,0,0],[0,-1,0]]` |
| run1 | 7.05 → **1.08 м** (6.5×) | +12.6 → **+3.7 м** | `[[0,0,1],[1,0,0],[0,1,0]]` |
| run2 | 7.85 → **1.38 м** (5.7×) | +12.8 → **+3.8 м** | те саме |
| run3 | 9.43 → **0.89 м** (10.5×) | +11.6 → **+4.6 м** | те саме |
| run4 | 8.46 → **1.38 м** (6.1×) | +10.3 → **+4.1 м** | те саме |
| run5 | 7.82 → **1.42 м** (5.5×) | +7.0 → **+2.3 м** | те саме |
| run6 | 7.84 → **1.36 м** (5.8×) | +9.4 → **+3.1 м** | те саме |

Симптом з `chat_context` 7.1 («еталонна хмара висить над траєкторією») зник. Усі шість наборів AirSim дають **однакову** матрицю, euroc — іншу (інший симулятор).

### 0.1 RTE і RRE до та після

| набір | vslam | ATE, м | RTE до | RTE після | змін | RRE до | RRE після | змін |
|---|---|---|---|---|---|---|---|---|
| euroc | dpvo | 0.58 | 0.136 | **0.022** | −84% | 0.346 | **0.115** | −67% |
| euroc | orbslam3 | 1.67 | 0.159 | **0.075** | −53% | 0.369 | **0.134** | −64% |
| run1 | dpvo | 126.93 | 3.569 | 3.044 | −15% | 10.698 | **2.512** | −77% |
| run1 | orbslam3 | 4.37 | 2.221 | **1.034** | −53% | 5.190 | **1.022** | −80% |
| run2 | dpvo | 140.73 | 3.456 | 2.876 | −17% | 10.820 | 10.478 | −3% |
| run2 | orbslam3 | 4.88 | 4.074 | **1.293** | −68% | 11.043 | **1.627** | −85% |
| run3 | dpvo | 179.80 | 3.846 | 3.112 | −19% | 11.977 | **5.749** | −52% |
| run3 | orbslam3 | 0.76 | 1.726 | **0.588** | −66% | 5.269 | 4.574 | −13% |
| run4 | dpvo | 150.38 | 2.971 | **1.881** | −37% | 11.415 | **2.817** | −75% |
| run4 | orbslam3 | 1.59 | 1.841 | **0.254** | −86% | 4.799 | **0.751** | −84% |
| run5 | dpvo | 151.62 | 2.609 | **1.902** | −27% | 8.905 | 9.125 | +2% |
| run5 | orbslam3 | 4.49 | 1.873 | **0.377** | −80% | 3.057 | **0.978** | −68% |
| run6 | dpvo | 112.33 | 4.158 | 3.667 | −12% | 10.241 | **2.542** | −75% |
| run6 | orbslam3 | 0.64 | 1.540 | **0.273** | −82% | 5.593 | **1.741** | −69% |

ATE не змінився ніде (залежить лише від положень). Причина спотворення: відносне переміщення порівнювалось у різних локальних системах, тож помилка складала фіксовану частку кроку між кадрами. Для ORB-SLAM3 покадрова точність була занижена в 4–7 разів. Де DPVO справді розійшовся (run2, run5), RRE майже не змінився — там домінує реальна похибка.

### 0.2 Фіксований воксель і нові метрики

`--gt_compare` тепер друкує IoU, FPVR, precision, recall на п'яти порогах; `results_table.py` рахує при **фіксованих** 1.0 і 0.5 м, автоматичний розмір — окрема колонка. Приклад (euroc, dpvo+zoedepth, воксель 1 м, таблиця корекції з виключенням euroc):

| варіант | поріг | IoU | FPVR | precision | recall |
|---|---|---|---|---|---|
| базова | P>0.50 | 0.1575 | 0.6762 | 0.3238 | 0.2347 |
| базова | P>0.85 | 0.1411 | 0.6409 | 0.3591 | 0.1886 |
| адаптивна | P>0.50 | **0.2491** | 0.6798 | 0.3202 | **0.5287** |
| адаптивна | P>0.85 | 0.2401 | **0.4349** | **0.5651** | 0.2945 |

### 0.3 Наявність кадрів FDE

Усі 7 наборів, усі 3 моделі — глибина є для **всіх** кадрів.

| набір | кадрів | zoe | dav2 | fast |
|---|---|---|---|---|
| euroc | 2593 | 2593 | 2593 | 2593 |
| run1 | 1251 | 1251 | 1251 | 1251 |
| run2 | **1958** | 1958 | 1958 | 1958 |
| run3 | 1118 | 1118 | 1118 | 1118 |
| run4 | 1177 | 1177 | 1177 | 1177 |
| run5 | **2296** | 2296 | 2296 | 2296 |
| run6 | 1165 | 1165 | 1165 | 1165 |

**Розбіжність із `chat_context` 3:** там run2 = 1197 і run5 = 1124 кадри, фактично 1958 і 2296.

### 0.4 Маска лопатей

`masks/blades_airsim.npz`: **8274 px, 2.69 %** кадру 480×640, поріг «еталонна глибина < 0.5 м у ≥ 50 % кадрів», комбіновано по run1–run6 через мінімум. Розподіл: у верхній половині кадру рівно **0 %**, у кожному нижньому квадранті 5.4 % — дві плями в нижніх кутах. euroc (Isaac Sim): максимальна стійкість 0.11, маска **порожня**, тобто лопатей у кадрі немає.

### Відкрите

1. Маску лопатей ще **не підключено** до оцінки і навчання (буде на етапі 1).
2. Карти для run1–run6 побудувати неможливо: DPVO має ATE 112–180 м, ORB-SLAM3 без експорту точок карти (патч `src/slam/orbslam3/patch_landmarks_export.py` готовий, треба перезібрати образ).
3. Порівняння моделей калібрування (euroc, dpvo+zoedepth, воксель 1 м): `scale` краще без таблиці корекції (IoU 0.1674 проти 0.1524), `affine` краще з нею (0.2616 проти 0.2491). Дефолт — `scale`, як погоджено.
4. Вікно якорів «лише поточний кадр» **виміряно шкідливим** (±20 кадрів IoU 0.013, ±100 → 0.120, уся хмара → 0.152, геометричне ±458 → 0.155). Причина геометрична: точку, видиму зараз на 50 м, тріангулювали з поз за 50 м позаду. Лишив прапорцем `--anchor_window_frames`, дефолт `off`. Це **розбіжність із `chat_context` 7.2**, де такий варіант був погоджений як основний.


## 2026-09-30 — маска лопатей у роботі + ЕТАПИ 1–2 (мережа уточнення глибини)

### Маска лопатей підключена

`--blade_mask` (дефолт `../masks/blades_airsim.npz`) застосовується **однаково до всіх варіантів** карти і до еталонної сітки, а також у навчанні таблиць `depth_refine`. LOO-таблиці перетреновані з маскою.

Ефект (euroc відкладений, воксель 1 м, adaptive, `--depth_refiner loo_euroc_*`):

| FDE | IoU без маски | IoU з маскою | FPVR без → з |
|---|---|---|---|
| zoedepth | 0.2491 | **0.2642** | 0.680 → 0.651 |
| depthanythingv2 | 0.2659 | **0.3597** | 0.661 → **0.528** |
| fastdepth | 0.2330 | **0.2474** | 0.721 → 0.656 |

Механізм: таблиці вчаться на AirSim, і лопаті більше не переконують їх, що дуже близька поверхня в нижніх кутах кадру — норма.

### Нові файли

| файл | призначення |
|---|---|
| `src/depth_net_data.py` | кеш для навчання: 320×240, float16, ~0.33 МБ/кадр. Канали: сіре зображення, градієнт Собеля, глибина FDE, розріджені опорні глибини + маска, ціль + валідність. Колір **не** подається (усі AirSim з одного парку). Режим `--to_memmap` переписує `.npz` у теку `.npy` для читання частинами |
| `src/depth_net.py` | U-Net (4 рівні, base=32), навчання, оцінка. Вхід 5 каналів, вихід 3 (Δ, log-дисперсія, «поверхня є»). Ранній зупин за валідацією з **навчальних** наборів, AMP, OneCycleLR, фіксовані зерна |
| `src/depth_net_infer.py` | запуск мережі в конвеєрі карти: масштабування 640×480 ↔ 320×240, реальні точки VSLAM у розрідженому каналі, перевірка розходження з якорями і відкат |

### Нормалізація і функція втрат

Усі глибини кадру діляться на медіану **опорних** глибин цього кадру і логарифмуються: `x = log(depth / median(sparse))`. Дільник зберігається покадрово. Це те, що робить задачу незалежною від абсолютних метрів конкретного парку; під час навчання перевіряється прямо — усі глибини кадру множаться на випадковий коефіцієнт 0.3–3.0 (у лог-просторі це зсув), і задача не має змінитися.

Функція втрат: Лаплас NLL по лог-глибині на валідних пікселях плюс бінарна крос-ентропія для каналу «поверхня є»:

```
L = mean_valid[ |r| / b + log(b) ] + w_bce · BCE(surface, valid)
r = (log_fde + Δ) − log_target,   b = exp(log_var / 2),   w_bce = 0.5
```

Зміни даних: яскравість, контраст, гамма, туман, множення градієнта, множення всіх глибин на 0.3–3.0, випадкове викидання до 50 % опорних точок, додатковий шум на опорних глибинах, горизонтальне віддзеркалення.

### Підключення до карти (`--depth_net`)

Порядок на кадр: FDE → мережа (виправлена глибина, σ_мережі, p_поверхні) → відкинути пікселі з p < `--net_min_surface` (0.5) → наявне калібрування за точками VSLAM як підстраховка → якщо медіанне розходження з якорями > `--net_reject_k` × S (S — власна поточна оцінка залишку), кадр обробляється **без** мережі. Вага пікселя:

```
σ_заг = sqrt(σ_калібрування² + σ_мережі²),   w = 1 / (1 + σ_заг / S)
```

Прапорець діє на всі варіанти карти однаково.

### Кеш (ZoeDepth, усі 7 наборів)

| набір | кадрів | МБ | дільник, м | log-ціль p5/p50/p95 | валідних | опорних точок/кадр |
|---|---|---|---|---|---|---|
| euroc | 1200 | 400 | 11.37 | −3.91/+0.03/+1.39 | 62 % | 355 |
| run1 | 1185 | 462 | 16.41 | −1.01/−0.01/+1.19 | 47 % | 365 |
| run2 | 1198 | 480 | 12.78 | −1.17/−0.03/+1.31 | 53 % | 375 |
| run3 | 1118 | 474 | 9.88 | −1.37/−0.06/+1.32 | 61 % | 365 |
| run4 | 1177 | 478 | 11.60 | −1.33/−0.06/+1.27 | 57 % | 364 |

Разом 3.0 ГБ. Цілі центровані на нулі завдяки нормалізації. Помітна відмінність домену: у euroc p5 = −3.9 (дуже близькі поверхні відносно дільника), у AirSim −1.0…−1.4. Це те, що перевірятиметься прогоном між симуляторами.

### Середовище

PyTorch 2.6.0+cu124, CUDA працює, RTX 4060 Laptop 8 ГБ.

### Відкрите на цей момент

- Перший прогін (euroc відкладений, ZoeDepth) у роботі; числа глибини будуть після нього.
- Кеші для DAv2 і FastDepth ще не побудовані.

---

## 30.09.2026. Карти на всі набори, 2D карти ймовірності, рисунки для звіту

### Що змінено

| файл | зміна |
|---|---|
| `build_all_prob_maps.sh` | новий. 42 прогони, 7 наборів на 2 VSLAM на 3 FDE, авто-воксель, `--prob_maps` у `maps/<набір>/<пара>/prob2d/`, відновлюваний, видаляє тільки свої файли |
| `build_all_fine_maps.sh` | новий. Ті самі 42 прогони з `--voxel_resolution fine --max_depth 50` у `maps/<набір>/<пара>/fine/`, плюс три HTML на прогін |
| `src/occupancy_3d_map.py` | новий режим `--voxel_resolution fine` з `--max_grid_voxels` і `--min_voxel`, виправлена шкала кольору 2D карт, оновлена довідка `--max_depth` |
| `src/render_figures.py` | новий. Рисунки для звіту з HTML і з `.npz` без браузера |
| `docs/pipeline_method.md` | новий. Повний опис конвеєра за кодом, формули (2.1) до (2.33) |
| `docs/images/README.md`, `maps/README.md` | опис структури тек |

### Команди

```bash
./build_all_fine_maps.sh                       # дрібні карти + HTML
./build_all_prob_maps.sh                       # авто-воксель + prob2d
cd src && python3 render_figures.py            # 105 рисунків у docs/images
```

### Числа. Авто-воксель, порівняння базової і ймовірнісної карти при P > 0,5

| набір | FDE | воксель, м | IoU база | IoU ймовір. | FPVR база | FPVR ймовір. |
|---|---|---|---|---|---|---|
| №1 euroc | ZoeDepth | 6,85 | 0,278 | **0,470** | 0,140 | 0,393 |
| №1 euroc | DAv2 | 6,08 | 0,430 | **0,585** | 0,148 | 0,166 |
| №1 euroc | FastDepth | 5,91 | 0,138 | **0,432** | 0,048 | 0,207 |
| №2 run1 | ZoeDepth | 71,9 | 0,275 | 0,248 | 0,359 | 0,723 |
| №3 run2 | ZoeDepth | 75,4 | 0,156 | 0,134 | 0,636 | 0,844 |
| №4 run3 | ZoeDepth | 15,0 | 0,043 | **0,091** | 0,819 | 0,816 |
| №5 run4 | ZoeDepth | 21,8 | 0,097 | **0,127** | 0,712 | 0,810 |
| №6 run5 | ZoeDepth | 84,5 | 0,342 | 0,250 | 0,391 | 0,723 |

Повна таблиця у `maps/all_maps_summary.csv`, дрібні карти у
`maps/all_fine_maps_summary.csv`.

### Що з цього видно

Ймовірнісний шар дає стійке покращення лише на наборі №1, від 1,7 до 3,1 раза
за IoU. На наборах AirSim він або не допомагає, або шкодить, і причина не в
самому шарі. Авто-воксель там виходить від 15 до 85 м, бо `max_depth` за
формулою тріангуляції дає від 473 до 611 м. Це прямий наслідок того, що кадри
записувались з кроком близько 1,5 м замість заданих 0,075 м, тобто база
тріангуляції завелика в десять разів. Поки набори не перезаписані в
покроковому режимі, ці числа характеризують запис, а не метод.

Дрібні карти з вокселем 1 м і радіусом 50 м виглядають реконструкцією, але IoU
на AirSim там від 0,0002 до 0,0035. При похибці глибини близько 10 м воксель
1 м майже нічого не вгадує. На наборі №1 дрібна карта дає IoU від 0,25 до 0,29.

### Дві помилки, знайдені і виправлені

Шкала кольору 2D карт стояла від 0,5 до 1,0, хоча `P = sigmoid(0,85 · w)` при
`w` від 0 до 1 не може перевищити 0,70. Усі кадри виходили однаково зеленими.

Авто-батч робив `rm -rf` на теці пари і зносив разом з нею вкладену `fine/`,
побудовану попереднім батчем. Тепер видаляє лише власні файли.

### Відкрите

1. ORB-SLAM3 не дає розріджених точок, тому для нього ймовірнісна карта не
   будується взагалі, лише базова. Потрібна перезбірка контейнера з експортом
   `landmarks.txt`.
2. Базова карта досі використовує механізм узгодженості зі сталим `τ_c = 2`.
   Треба або вимкнути для чистоти порівняння, або описати у звіті як є.
3. `apply_decay` реалізований, але в циклі не викликається жодного разу.
4. Перезапис наборів AirSim у покроковому режимі.

---

## 01.10.2026. Етапи обробки кадру для презентації, виправлення маски лопатей

### Що змінено

| файл | зміна |
|---|---|
| `src/occupancy_3d_map.py` | новий прапорець `--dump_frame_name`: дамп і 2D карта зберігаються лише для названого кадру, сам кадр примусово додається у вибірку в хронологічному порядку. У дамп додано `residual_scale`, `bandwidth_scale`, `sigma_pose`, `blades`, `refine_dropped` |
| `src/blade_mask.py` | нова `applies_to()`, `load_mask()` приймає `dataset_name` |
| `src/occupancy_3d_map.py` | маска лопатей береться лише для наборів, на яких її міряли |
| `src/presentation_frames.py` | новий. Вісім PNG 640 на 480 на етап обробки плюс `stats.txt` |
| `presentation_frames/README.md` | опис, шкали, команди відтворення |

### Команди

```bash
cd src && python3 occupancy_3d_map.py --mode dataset \
    --dataset_root ../datasets/euroc/2026-04-01_12-13-18_000_quadrotor \
    --vslam_dir ../results/euroc/dpvo --fde_dir ../results/euroc/zoedepth \
    --depth_refiner ../refiners/loo_euroc_zoedepth.npz \
    --output_dir ../presentation_frames/euroc/_grid \
    --dump_intermediate ../presentation_frames/euroc/_dump \
    --prob_maps ../presentation_frames/euroc/_prob \
    --dump_frame_name 1775034831515681460.png --n_frames 80
python3 presentation_frames.py --dataset euroc
```

### Числа по кадрах

| величина | №1 euroc | №2 run1 |
|---|---|---|
| кадр | `1775034831515681460` | `1787595584854525184` |
| медіанна похибка глибини до | 2,72 м | 0,85 м |
| медіанна похибка глибини після | **0,77 м** | **4,83 м** |
| зміна | −71,6 % | +467,1 % |
| медіана множника s | 1,132 | 1,655 |
| розкид s, від 5 до 95 відсотка | від 0,743 до 2,062 | від 0,713 до 5,969 |
| поточне S | 8,00 м | 65,85 м |
| невизначеність пози | 0,26 м | 4,20 м |
| опорних точок VSLAM | 986 | 1703 |
| відкинуто, таблиця валідності | 46,71 % | 42,16 % |
| відкинуто, маска лопатей | 0,00 % | 2,69 % |
| середня вага w | 0,601 | **0,365** |

На №1 калібрування зменшує похибку глибини майже вчетверо. На №2 воно псує
глибину в п'ять з половиною разів, бо розріджені точки DPVO там самі зміщені,
і множник доходить до 6. Але механізм довіри це бачить: S виростає від 8 до
66 м, а середня вага падає від 0,60 до 0,37. Тобто шар не виправляє погані
вхідні дані, але позначає їх як ненадійні.

### Знайдена і виправлена помилка

Маску лопатей пораховано лише на `run1`–`run6`, а застосовувалась вона до всіх
наборів з кадром 480 на 640, тобто й до `euroc`, у якого гвинтів у кадрі немає.
На наборі №1 вона викидала 8274 справжні пікселі, 2,69 відсотка кадру, у нижніх
кутах, де видно доріжку і землю. Тепер маска береться лише для наборів, які
перелічені в полі `names` її файлу.

**Наслідок:** усі числа по набору №1, пораховані до 01.10.2026, трохи занижені.
Для одного кадру медіанна похибка після калібрування змінилась від 0,73 до
0,77 м, частка відкинутих пікселів від 49,41 до 46,71 відсотка. Карти і таблиці
IoU по набору №1 треба перебудувати.

### Відкрите

1. Перебудувати карти набору №1 після виправлення маски.
2. ORB-SLAM3 без `landmarks.txt` дає лише базову карту.
3. Базова карта досі з механізмом узгодженості зі сталим `τ_c = 2`.
4. `apply_decay` не викликається.
5. Перезапис наборів AirSim у покроковому режимі.

---

## 01.10.2026 (друге). Вето неба з зображення

Причина: на кадрі презентації видно, що карта позначає землю під апаратом як
впевнено зайняту, а дерева на горизонті не позначає взагалі.

### Діагноз

Таблиця валідності з `depth_refine.py` індексована лише двома ознаками,
інтервалом глибини і номером рядка кадру. На рядку горизонту крона дерева і
небо потрапляють в одну комірку, тому таблиця мусить дати їм однакову
відповідь. Розкладка кадру euroc по смугах зверху вниз:

| смуга | є поверхня | йшло в карту | втрачалось | медіана GT |
|---|---|---|---|---|
| рядки 80–160 | 17,2 % | 0 % | **100 %** | 23 м |
| рядки 160–240 | 52,9 % | 19,7 % | **66,5 %** | 25 м |
| рядки 240–480 | 97–100 % | 99–100 % | 0 % | від 5,8 до 16 м |

Валідність комірок у смузі горизонту 0,36 при порозі 0,50, тому смуга
відрізалась цілком.

### Що зроблено

| файл | зміна |
|---|---|
| `src/sky_cue.py` | новий. `sky_mask()` і `structure_mask()`, пороги з самого кадру методом Оцу і перцентилем, плюс режими перевірки `eval` і `veto` |
| `src/occupancy_3d_map.py` | прапорці `--sky_cue` і `--sky_cue_percentile`, вето стоїть до побудови будь-якого варіанта |

Правило: піксель, який таблиця відкинула, повертається в карту, якщо саме
зображення каже, що небом він бути не може, тобто темний за порогом Оцу або
текстурований понад 70-й перцентиль градієнта кадру. Еталон не потрібен.

### Числа. Втрата поверхні і фантоми неба, 30 кадрів на набір

| набір | втрачено, таблиця | втрачено, з вето | фантомів, таблиця | фантомів, з вето |
|---|---|---|---|---|
| №1 euroc | 10,53 % | **0,13 %** | 10,99 % | 12,94 % |
| №2 run1 | 11,84 % | **4,33 %** | 26,30 % | 29,03 % |
| №3 run2 | 13,08 % | **4,71 %** | 13,38 % | 17,43 % |
| №4 run3 | 14,65 % | **5,54 %** | 9,24 % | 13,04 % |
| №5 run4 | 11,01 % | **4,01 %** | 10,28 % | 13,35 % |
| №6 run5 | 17,26 % | **5,86 %** | 7,55 % | 10,93 % |
| №7 run6 | 14,08 % | **4,72 %** | 10,68 % | 12,27 % |

Параметр підібрано на №1–№3, перевірено на №4–№7, переноситься без змін.

### Вплив на карту, набір №1, DPVO + ZoeDepth

| поріг зчитування | IoU без вето | IoU з вето | повнота без | повнота з |
|---|---|---|---|---|
| P > 0,50 | 0,4890 | 0,4846 | 0,645 | 0,668 |
| P > 0,70 | 0,4762 | 0,4765 | 0,569 | 0,581 |
| P > 0,85 | 0,4631 | **0,4795** | 0,530 | 0,555 |
| P > 0,93 | 0,4536 | **0,4718** | 0,500 | 0,531 |
| P > 0,97 | 0,4457 | **0,4673** | 0,482 | 0,516 |

На №2 різниця в межах шуму, на №3 вето трохи гірше за IoU. Повнота росте на
всіх трьох порогах і на всіх наборах.

Пояснення: IoU однаково карає пропущену перешкоду і фантом, тому не може
виразити те, заради чого вето й зроблено. Зате на вищих порогах зчитування
вето виграє, і це прямий аргумент на користь ймовірнісної карти: зайві пікселі
неба трапляються раз і з малою вагою, а врятовані крони підтверджуються з
багатьох ракурсів.

### Стан прапорця

`--sky_cue` вимкнений за замовчуванням, як і решта виміряних ідей, тому всі
попередні числа лишаються дійсними. Кадри презентації перегенеровані З вето,
бо саме цю ваду вони й ілюструють. Після вето на euroc втрата поверхні в кадрі
падає з 14,3 до 0,3 відсотка, у смузі рядків 160–240 з 66,5 до 0,7 відсотка.

### Відкрите

1. Вирішити, чи вмикати `--sky_cue` за замовчуванням і перебудувати всі карти.
2. Перебудувати карти набору №1 після виправлення маски лопатей (IoU там уже
   0,4890 замість 0,4695).
3. ORB-SLAM3 без `landmarks.txt`.
4. Базова карта з механізмом узгодженості.
5. `apply_decay` не викликається.
6. Перезапис наборів AirSim у покроковому режимі.

---

## 01.10.2026 (третє). Ілюстрація переходу 2D у 3D для презентації

### Що змінено

| файл | зміна |
|---|---|
| `src/occupancy_3d_map.py` | необов'язковий параметр `hooks` у `run_dataset_mode` з гачками `frame`, `grid_frame`, `grids`. За замовчуванням `None`, поведінка не змінюється |
| `src/explain_2d_to_3d.py` | новий. Вісім рисунків і `trace.json` з реального прогону |
| `presentation_frames/euroc/2d_to_3d/README.md` | опис, числа, команди |

Гачки потрібні були, щоб ілюстрація не повторювала код конвеєра. Скрипт
запускає `run_dataset_mode` один раз і забирає проміжні величини прямо з нього,
тому зворотна проєкція, калібрування, таблиця, маска лопатей, вето неба і
накопичення в логарифмі шансів тут ті самі функції, що в конвеєрі.

### Команди

```bash
cd src && python3 explain_2d_to_3d.py --n_frames 2600   # повний прогін, усі 2592 кадри
cd src && python3 explain_2d_to_3d.py --from_cache      # лише перемалювати рисунки
```

### Числа з trace.json. Набір №1, кадр 1775034831515681460, воксель 1 м

Піксель **A**, крона дерева, u = 86, v = 96:

| величина | значення |
|---|---|
| сира глибина ZoeDepth | 21,63 м |
| множник таблиці корекції r | 0,766 |
| глибина після таблиці | 16,57 м |
| множник калібрування s | 2,110 |
| глибина після калібрування z′ | **34,96 м** |
| еталонна глибина | **34,74 м** |
| розбіжність підгонки ділянки r_g | 2,04 м |
| ширина ядра відносно типової | 0,594 |
| σ глибини, σ пози, σ повна | 3,25 м, 0,26 м, 3,51 м |
| поточне S | 8,01 м |
| вага довіри w | 0,618 |
| точка у світі p_A | (80,3; 26,3; 9,7) м |
| індекс вокселя | (81, 111, 32) |
| внесок кадру ΔL | +0,525 |
| кадрів, що бачили воксель | 18 |
| кінцеве L, P | 4,00 і 0,982 |

Піксель **B**, небо, u = 318, v = 42: відкинутий таблицею корекції, у карту не
йде. σ = 17,1 м проти S = 8,0 м, тобто навіть без відкидання вага була б 0,32.

### Що з цього варте згадки у звіті

Сира глибина ZoeDepth помилялась на 13 м, таблиця корекції зробила ще гірше,
а калібрування по розріджених точках VSLAM вивело на 0,2 м від еталона. Це
найкоротший доказ того, що саме локальне калібрування, а не таблиця, несе
основну роботу.

Воксель набирає L від нуля до межі 4 за 18 кадрів з 2592, тобто за приблизно
140 кадрів польоту, і далі насичується. Через це фактична зміна L на самому
кадрі з рисунків дорівнює нулю: воксель уже вперся в межу. У `trace.json` це
записано окремим полем.

### Відкрите

Без змін до попереднього запису, плюс:
1. Вирішити, чи вмикати `--sky_cue` за замовчуванням.
2. Перебудувати карти після виправлення маски лопатей.

---

## 21.10.2026. Карта довіри по UDP у реальному часі

### Що змінено

| файл | зміна |
|---|---|
| `src/udp_streaming.py` | типи FRAME2 (кадр із зображенням і розрідженими точками), CONF2D (2D карта довіри), GRID_DELTA (лише змінені вокселі). Класи `MapPublisher` і `MapSubscriber`. Старий FRAME і далі приймається |
| `src/realtime_map.py` | новий. `RealtimeConfidenceMapper`: стан між кадрами, обробка одного кадру тими самими функціями, що й офлайн |
| `src/occupancy_3d_map.py` | `run_udp_mode` переписано на ймовірнісний шар замість старих кривих. Прийом у окремому потоці з чергою. Новий `track_changes` у `insert_weighted`, типово вимкнений |
| `src/udp_replay_sender.py` | шле FRAME2 з зображенням і точками DPVO, прапорці `--v1`, `--no_image`, `--no_landmarks` |
| `src/udp_conf_receiver.py` | новий. Споживач: збирає карту з приростів, пише PNG і npz |
| `docs/realtime_udp.md` | новий. Протокол, вимоги до відправника, швидкодія, обмеження |

### Команди

```bash
cd src && source ../.venv/bin/activate
python3 udp_conf_receiver.py --port 9001 --output_dir ../live --save_png_every 10
python3 occupancy_3d_map.py --mode udp --listen_port 9000 \
    --send_host 127.0.0.1 --send_port 9001 \
    --dataset_root ../datasets/euroc/2026-04-01_12-13-18_000_quadrotor \
    --depth_refiner ../refiners/loo_euroc_zoedepth.npz --sky_cue \
    --bounds_radius 60 --voxel_resolution 1.0 --max_depth 50 --output_dir ../live_map
python3 udp_replay_sender.py --dataset_root ../datasets/euroc/...quadrotor \
    --vslam_dir ../results/euroc/dpvo --fde_dir ../results/euroc/zoedepth \
    --host 127.0.0.1 --port 9000 --n_frames 40 --rate 10 --send_intrinsics
```

### Швидкодія, набір №1, воксель 1 м

| радіус карти | сітка | мс на кадр | межа |
|---|---|---|---|
| 150 м | 300³ = 27 млн | 237 | 4,2 Гц |
| **60 м** | **120³ = 1,7 млн** | **66** | **15,2 Гц** |
| 40 м | 80³ = 0,5 млн | 60 | 16,7 Гц |

Розклад 60 мс: калібрування 15,1, таблиця 12,8, вето неба 12,8, зворотна
проєкція 11,4, стиснення 6,9, вставка у сітку близько 3.

Трафік на 15 Гц: 2D карта 3,9 МБ/с (float16 плюс zlib, 259 КБ на кадр),
прирости 0,4 МБ/с. Повна сітка замість приростів коштувала б 104 МБ/с.

### Що перевірено

| що | результат |
|---|---|
| кодування всіх типів, круговий тест | збігається |
| `track_changes` проти повного порівняння масивів | ті самі вокселі, сітки ідентичні |
| **карта в споживача проти карти на сервері**, 40 кадрів | **масиви побайтово ідентичні**, 18743 зайнятих |
| офлайновий режим не зламано | euroc DPVO + ZoeDepth, IoU 0,4890, як і було |
| `--no_landmarks` | 12 кадрів, 0 з картою довіри, причина в лозі |
| `--v1`, старий формат | приймається |
| `--no_image` | 12 кадрів, усі калібровані |
| перевантаження, 60 кадрів у чергу на 1 | 19 оброблено, 41 відкинуто і пораховано |

### Дві помилки, знайдені тестами

`evo` вирівнює траєкторію **на місці**, тому повторний виклик `align()` у
відправнику давав майже одиничне перетворення, і розріджені точки лишались у
сирій системі DPVO. Наслідок: 38 з 40 кадрів без жодної опорної точки у кадрі.
Перетворення тепер береться з першого і єдиного вирівнювання.

Приріст воксельної карти рахувався копіюванням і порівнянням усього масиву:
при 27 млн вокселів це чотири пересилки по 108 МБ на кадр. Замінено на
`track_changes`, який збирає індекси прямо під час вставки.

### Відкрите

1. Сітка нерухома, за апаратом не їде. Рухома локальна карта це окрема робота.
2. При великому радіусі дві третини часу йде на копіювання знімка стану в
   `insert_weighted`. Можна читати попередній стан лише у зачеплених вокселях,
   але це зміна найвідповідальнішої функції, тож потребує такої самої перевірки
   на ідентичність, як робилась для `track_changes`.
3. Решта пунктів з попередніх записів без змін.

### Доповнення: інструменти самоперевірки

| файл | що робить |
|---|---|
| `check_realtime.sh` | автоматична перевірка, 15 пунктів, друкує ТАК або НІ, код виходу 0 означає що все добре. `--full` додає офлайнову регресію |
| `src/udp_demo.py` | жива демонстрація обох напрямків: зводить надіслані кадри і повернуті карти довіри за номером кадру, міряє затримку туди й назад |
| `src/udp_replay_sender.py` | завантаження кадрів винесено у `load_stream()`, бо ним користуються і відправник, і демонстрація |

Виміряно `udp_demo.py` на наборі №1, 12 кадрів, радіус 60 м: 12 з 12 карт
довіри повернулись, затримка туди й назад медіана 98 мс, найгірша 141 мс.
З `--no_landmarks` жодної карти не повертається, і сервер каже чому.

---

## 22.10.2026. Імпровізований автопілот

Реактивний вибір напрямку за покадровою картою зайнятості. Без навчання, без
моделі, лише геометрія і арифметика.

### Файли

| файл | що |
|---|---|
| `src/autopilot.py` | `ReactiveAutopilot`: кадр усередину, напрямок, точка і швидкість назовні |
| `src/autopilot_demo.py` | прогін по справжніх кадрах, картинки і зведений графік |
| `src/udp_streaming.py` | тип повідомлення `MSG_COMMAND` |
| `integration/server_simple.py` | прапорці `--autopilot` і `--cautious` |

### Як вирішує

1. Пікселі кадру стають точками з **імовірністю зайнятості**
   `P = sigmoid(0,85 · w)`, а не просто фактом існування.
2. Віяло кандидатів у межах поля зору, типово 21 на 5 напрямків.
3. На кандидата дві оцінки: **запас** (найближча певна точка в коридорі
   завширшки з апарат) і **ризик** (сума ймовірностей, ближчі важать більше).
4. Вартість = ризик + відхилення від курсу. Перемагає найменша.
5. Швидкість пропорційна запасу, нижче гальмівного шляху нуль і ознака стопу.

### Перевірка на сценах з відомою відповіддю

| сцена | очікувано | вийшло |
|---|---|---|
| порожньо | прямо, повна швидкість | азимут 0°, 3,0 м/с |
| стіна за 1 м | стоп | стоп, запас 0,9 м |
| перешкода ліворуч | відхилення праворуч | +15° |
| перешкода праворуч | відхилення ліворуч | −15° |
| стовп по центру | обхід убік | −25° |
| **той самий стовп, низька довіра** | **слабший маневр** | **0° замість −25°** |

Остання стрічка і є сенс роботи: **ймовірнісний шар напряму змінює поведінку
апарата**. Перешкода, якій конвеєр не довіряє, гальмує слабше за ту, у якій він
упевнений.

### На справжніх кадрах, набір №1, 40 кадрів

| режим | сер. швидкість | стопів | сер. \|азимут\| |
|---|---|---|---|
| невідоме = порожньо | 2,72 м/с | 2 | 3,8° |
| невідоме = ризик (`--cautious`) | 2,65 м/с | 2 | 3,4° |

Перші кадри обережні, бо самоналаштовний масштаб `S` ще не прогрівся, далі
рівний політ уздовж доріжки.

Картинки і `commands.csv` у `presentation_frames/euroc/autopilot/`.

### Відкрите

1. Автопілот дивиться лише на покадрову карту. Накопичена воксельна карта вже
   є поруч, і враховувати її було б природно: те, що вже підтверджене з
   багатьох кадрів, має важити більше за одне спостереження.
2. Немає ніякої динаміки апарата: команда це бажаний напрямок, а не керування.
3. Ціль задається лише як напрямок уперед або явна точка. Маршруту немає.
