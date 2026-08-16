#!/bin/bash
# Runs compare_reconstruction_3d.py for all 6 combinations (2 VSLAM x 3 FDE)
# against your results/ folder, saving one .html + one _summary.json per
# combo. Doesn't auto-open 6 browser tabs -- open the ones you want by hand.
#
# Usage: ./compare_all_combos.sh <dataset_root>
set -e

DATASET_ROOT=$1
if [ -z "$DATASET_ROOT" ]; then
    echo "Usage: $0 <dataset_root>"
    exit 1
fi

mkdir -p comparisons
for vslam in dpvo orbslam3; do
    for fde in zoedepth depthanythingv2 fastdepth; do
        if [ ! -f "results/$vslam/poses.txt" ] || [ ! -d "results/$fde" ]; then
            echo "SKIP: $vslam + $fde -- results/$vslam/poses.txt or results/$fde missing"
            continue
        fi
        echo "== $vslam + $fde =="
        python3 compare_reconstruction_3d.py \
            --dataset_root "$DATASET_ROOT" \
            --vslam_dir "results/$vslam" \
            --fde_dir "results/$fde" \
            --output "comparisons/${vslam}_${fde}.html" \
            --no_open
        echo ""
    done
done

echo "All done. Open any result with: explorer.exe comparisons/<vslam>_<fde>.html"
echo "Numeric summaries: comparisons/*_summary.json"
