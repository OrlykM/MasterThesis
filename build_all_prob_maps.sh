#!/bin/bash
# Будує ймовірнісну 3D воксельну карту для КОЖНОГО набору і кожної моделі глибини,
# і поруч з кожною картою складає теку prob2d/ з 2D картами ймовірності зайнятості
# (по одному PNG на кадр, P = sigmoid(l_occ * w) по пікселях після всіх корекцій).
#
# Вихід:
#   maps/<набір>/<vslam>_<fde>/naive_grid.npz      базова (ваги = 1, без калібрування)
#   maps/<набір>/<vslam>_<fde>/adaptive_grid.npz   ймовірнісна з вагами довіри
#   maps/<набір>/<vslam>_<fde>/adaptive_grid.html  перегляд у браузері
#   maps/<набір>/<vslam>_<fde>/prob2d/prob_*.png   2D карти ймовірності по кадрах
#   maps/all_maps_summary.csv                      зведена таблиця метрик
#
# Таблиця корекції глибини береться з виключенням цього ж набору (leave-one-out),
# тобто набір, на якому будується карта, у навчанні таблиці не брав участі.
set -u
cd "$(dirname "$0")/src"
source ../.venv/bin/activate

N_FRAMES=${N_FRAMES:-80}
LOGDIR=${LOGDIR:-../logs/build_all}
mkdir -p "$LOGDIR" ../maps
CSV=../maps/all_maps_summary.csv
echo "dataset,vslam,fde,voxel_m,max_depth_m,gt_occ,naive_iou,naive_fpvr,naive_prec,naive_rec,adaptive_iou,adaptive_fpvr,adaptive_prec,adaptive_rec,prob2d_png,note" > "$CSV"

root_of() {
  case "$1" in
    euroc) echo "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor" ;;
    *)     echo "../datasets/$1" ;;
  esac
}

# ORB-SLAM3 у results/ не має файлу розріджених точок
# (landmarks.txt / .ply), а без них немає ні локального калібрування глибини,
# ні ваг довіри, тому ймовірнісна карта для нього не будується взагалі --
# виходить тільки базова. Повернути сюди після перезбірки контейнера
# ORB-SLAM3 з експортом landmarks.txt.
pairs_for() {
  echo "${VSLAMS:-dpvo orbslam3}"
}

TOTAL=0
for ds in euroc run1 run2 run3 run4 run5 run6; do
  for v in $(pairs_for $ds); do for f in zoedepth depthanythingv2 fastdepth; do TOTAL=$((TOTAL+1)); done; done
done
DONE=0

for ds in euroc run1 run2 run3 run4 run5 run6; do
  ROOT=$(root_of $ds)
  for vslam in $(pairs_for $ds); do
    for fde in zoedepth depthanythingv2 fastdepth; do
      DONE=$((DONE+1))
      OUT=../maps/$ds/${vslam}_${fde}
      LOG=$LOGDIR/${ds}_${vslam}_${fde}.log
      echo "[$DONE/$TOTAL] $ds  $vslam + $fde"
      # відновлюваний: готовий прогін не переробляємо, лише перечитуємо його лог
      if [ -z "${FORCE:-}" ] && [ -f "$OUT/naive_grid.npz" ] && \
         { [ "$vslam" = "orbslam3" ] || { [ -f "$OUT/adaptive_grid.npz" ] && \
           [ "$(ls "$OUT/prob2d" 2>/dev/null | wc -l)" -gt 0 ]; }; }; then
        echo "        вже є, пропускаю"
      else
      # Видаляємо ЛИШЕ те, що пише цей скрипт. Простий rm -rf "$OUT" зносив
      # разом з текою і вкладену fine/, яку будує build_all_fine_maps.sh.
      rm -rf "$OUT/prob2d" "$OUT"/*.npz "$OUT"/*.html
      mkdir -p "$OUT/prob2d"
      python3 occupancy_3d_map.py --mode dataset \
        --dataset_root "$ROOT" \
        --vslam_dir ../results/$ds/$vslam \
        --fde_dir ../results/$ds/$fde \
        --depth_refiner ../refiners/loo_${ds}_${fde}.npz \
        --output_dir "$OUT" \
        --prob_maps "$OUT/prob2d" \
        --n_frames $N_FRAMES --gt_compare > "$LOG" 2>&1
      rc=$?
      fi

      note=ok
      [ "${rc:-0}" -ne 0 ] && note="впало (див. $LOG)"; rc=0
      vox=$(grep -oP 'voxel_resolution = \K[0-9.]+' "$LOG" | tail -1)
      mdep=$(grep -oP 'max_depth = \K[0-9.]+' "$LOG" | tail -1)
      gtocc=$(grep -oP 'GT occupied voxels: \K[0-9]+' "$LOG" | tail -1)
      grep -q "coarser than 1/20th" "$LOG" && note="воксель грубший за 1/20 сцени"
      # перший рядок (P>0.50) кожного блоку
      read_thr0() {  # $1 = naive|adaptive
        awk -v want="$1" '/^\[gt_compare\] (naive|adaptive|weighted):/{blk=$2; sub(":","",blk)}
                          blk==want && /thr=0.00/{print; exit}' "$LOG"
      }
      parse() { echo "$1" | grep -oP "(?<![a-z])$2=\K[0-9.]+"; }
      nl=$(read_thr0 naive); al=$(read_thr0 adaptive)
      n_iou=$(parse "$nl" IoU); n_fp=$(parse "$nl" FPVR); n_pr=$(parse "$nl" prec); n_rc=$(parse "$nl" "(?<!p)rec")
      a_iou=$(parse "$al" IoU); a_fp=$(parse "$al" FPVR); a_pr=$(parse "$al" prec); a_rc=$(parse "$al" "(?<!p)rec")
      npng=$(ls "$OUT/prob2d" 2>/dev/null | wc -l)

      echo "$ds,$vslam,$fde,$vox,$mdep,$gtocc,$n_iou,$n_fp,$n_pr,$n_rc,$a_iou,$a_fp,$a_pr,$a_rc,$npng,$note" >> "$CSV"
      echo "        воксель ${vox:-?} м  IoU база ${n_iou:-?} -> ймовірнісна ${a_iou:-?}  FPVR ${n_fp:-?} -> ${a_fp:-?}  2D карт: $npng"

      # HTML для перегляду
      if [ -f "$OUT/adaptive_grid.npz" ]; then
        python3 view_3d_occupancy_map.py "$OUT/adaptive_grid.npz" --no_open --show_free \
          --trajectory "$ROOT/state_groundtruth_estimate0/data.csv" >> "$LOG" 2>&1
      fi
    done
  done
done

echo
echo "Готово. Зведена таблиця: $CSV"
column -s, -t "$CSV"
