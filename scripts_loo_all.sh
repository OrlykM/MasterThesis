#!/bin/bash
# Решта прогонів з виключенням по одному + прогін між симуляторами.
# Запускати ПІСЛЯ перевірки першого прогону.
cd /home/maxorlyk_wsl/MasterThesis/src
source ../.venv/bin/activate
FDE=${1:-zoedepth}
SHORT=${2:-zoe}
for held in run1 run2 run3 run4 run5 run6; do
  out=../nets/${SHORT}_holdout_${held}.pt
  [ -f "$out" ] && { echo "$held: вже є"; continue; }
  echo "=== $held ==="
  python3 depth_net.py train --fde $FDE --holdout $held --output $out --epochs 30 --batch 16
  python3 depth_net.py eval --model $out --metrics_out ${out%.pt}_metrics.json
done
