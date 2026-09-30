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
