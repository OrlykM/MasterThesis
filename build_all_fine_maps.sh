#!/bin/bash
# «Показові» карти: такі, як maps/euroc_demo_1.0 -- дрібний воксель і HTML,
# на яких карта виглядає реконструкцією, а не десятком кубиків.
#
# Дві відмінності від build_all_prob_maps.sh (обидві однаково для базової,
# ймовірнісної і еталонної карти, тому порівняння лишається чесним):
#
#   --voxel_resolution fine  найдрібніший воксель, що вміщається в пам'ять,
#                            але не дрібніший за 1 м
#   --max_depth <R>          робочий радіус ЛОКАЛЬНОЇ карти. Авто-режим на
#                            наборах AirSim дає від 473 до 611 м, бо крок між
#                            кадрами там 1.5 м і база тріангуляції виходить
#                            15 м. Формально межа правильна, але карта
#                            розмазується на кілометр і воксель авто стає 70 м.
#
# Вихід: maps/<набір>/<vslam>_<fde>/fine/
set -u
cd "$(dirname "$0")/src"
source ../.venv/bin/activate

RADIUS=${RADIUS:-50}
N_FRAMES=${N_FRAMES:-80}
LOGDIR=${LOGDIR:-../logs/build_fine}
mkdir -p "$LOGDIR"
CSV=../maps/all_fine_maps_summary.csv
echo "dataset,vslam,fde,radius_m,voxel_m,grid,gt_occ,naive_occ,naive_iou,naive_fpvr,adaptive_occ,adaptive_iou,adaptive_fpvr,adaptive_prec,adaptive_rec" > "$CSV"

root_of() { case "$1" in euroc) echo "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor";; *) echo "../datasets/$1";; esac; }

# ORB-SLAM3 теж будуємо, але в його теки results/ немає файлу розріджених
# точок (у DPVO це dpvo_sparse_map.ply). Без них немає ні локального
# калібрування глибини, ні ваг довіри, тому для ORB-SLAM3 виходить ЛИШЕ
# базова карта naive_grid, без ймовірнісного шару. Повний ймовірнісний
# варіант зʼявиться після перезбірки контейнера з експортом landmarks.txt.
VSLAMS=${VSLAMS:-"dpvo orbslam3"}
TOTAL=0
for ds in euroc run1 run2 run3 run4 run5 run6; do for v in $VSLAMS; do for f in zoedepth depthanythingv2 fastdepth; do TOTAL=$((TOTAL+1)); done; done; done
DONE=0
for ds in euroc run1 run2 run3 run4 run5 run6; do
  ROOT=$(root_of $ds)
  for vslam in $VSLAMS; do
  for fde in zoedepth depthanythingv2 fastdepth; do
    DONE=$((DONE+1))
    OUT=../maps/$ds/${vslam}_${fde}/fine
    LOG=$LOGDIR/${ds}_${vslam}_${fde}.log
    echo "[$DONE/$TOTAL] $ds  $vslam + $fde"
    if [ -z "${FORCE:-}" ] && [ -f "$OUT/map_with_free.html" -o -f "$OUT/naive_grid.html" ]; then
      echo "        вже є, пропускаю"
    else
    rm -rf "$OUT"; mkdir -p "$OUT"
    python3 occupancy_3d_map.py --mode dataset \
      --dataset_root "$ROOT" \
      --vslam_dir ../results/$ds/$vslam \
      --fde_dir ../results/$ds/$fde \
      --depth_refiner ../refiners/loo_${ds}_${fde}.npz \
      --output_dir "$OUT" \
      --voxel_resolution fine --max_depth $RADIUS \
      --n_frames $N_FRAMES --gt_compare > "$LOG" 2>&1 || echo "        ВПАЛО, див. $LOG"
    fi

    vox=$(grep -oP '\[fine\] voxel_resolution = \K[0-9.]+' "$LOG" | tail -1)
    grid=$(grep -oP 'grid=\K\S+' "$LOG" | tail -1)
    gtocc=$(grep -oP 'GT occupied voxels: \K[0-9]+' "$LOG" | tail -1)
    thr0() { awk -v w="$1" '/^\[gt_compare\] (naive|adaptive|weighted):/{b=$2;sub(":","",b)} b==w && /thr=0.00/{print;exit}' "$LOG"; }
    g() { echo "$1" | grep -oP "(?<![a-z])$2=\s*\K[0-9.]+"; }
    nl=$(thr0 naive); al=$(thr0 adaptive)
    echo "$ds,$vslam,$fde,$RADIUS,$vox,$grid,$gtocc,$(g "$nl" occ),$(g "$nl" IoU),$(g "$nl" FPVR),$(g "$al" occ),$(g "$al" IoU),$(g "$al" FPVR),$(g "$al" prec),$(g "$al" '(?<!p)rec')" >> "$CSV"
    echo "        воксель ${vox:-?} м, зайнятих ${_x:-$(g "$al" occ)}, IoU $(g "$nl" IoU) -> $(g "$al" IoU)"

    # два HTML, як у euroc_demo_1.0: лише зайняті, і зайняті + вільний простір
    TRAJ="$ROOT/state_groundtruth_estimate0/data.csv"
    if [ -f "$OUT/adaptive_grid.npz" ]; then
      python3 view_3d_occupancy_map.py "$OUT/adaptive_grid.npz" --no_open \
        --trajectory "$TRAJ" --output "$OUT/adaptive_grid.html" >> "$LOG" 2>&1
      python3 view_3d_occupancy_map.py "$OUT/adaptive_grid.npz" --no_open --show_free \
        --trajectory "$TRAJ" --output "$OUT/map_with_free.html" >> "$LOG" 2>&1
    fi
    [ -f "$OUT/naive_grid.npz" ] && python3 view_3d_occupancy_map.py "$OUT/naive_grid.npz" \
        --no_open --trajectory "$TRAJ" --output "$OUT/naive_grid.html" >> "$LOG" 2>&1
  done
  done
done
echo; echo "Готово. Таблиця: $CSV"; column -s, -t "$CSV"
