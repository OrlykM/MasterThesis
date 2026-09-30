#!/bin/bash
# Runs compare_reconstruction_3d.py for all 6 combinations (2 VSLAM x 3 FDE)
# against results/<run_name>/, saving one .html + one _summary.json per
# combo under comparisons/<run_name>/. Doesn't auto-open 6 browser tabs --
# open the ones you want by hand.
#
# <run_name> is taken automatically from dataset_root's own folder name
# (e.g. .../datasets/run1 -> results/run1/<algo>/).
#
# Paths are resolved relative to WHERE THIS SCRIPT FILE LIVES (not your
# current directory when you invoke it) -- assumes results/ and comparisons/
# sit one level up from this script (i.e. this script is in src/, and
# results/ + comparisons/ are siblings of src/, matching where
# docker-compose.yml's default ./results actually lands). Run it from
# anywhere, it'll still find the right folders.
#
# Usage: ./compare_all_combos.sh <dataset_root> [max_time_diff]
#   max_time_diff defaults to 0.15s -- your AirSim recordings have a
#   confirmed ~81ms constant offset between GT (IMU-clock) and camera
#   timestamps, so the old 0.02s default rejects every match. Pass a
#   different value as the 2nd arg if a specific dataset needs it (e.g.
#   your original Isaac Sim dataset was tight enough for 0.02).
set -e

DATASET_ROOT=$1
MAX_TIME_DIFF=${2:-0.15}

if [ -z "$DATASET_ROOT" ]; then
    echo "Usage: $0 <dataset_root> [max_time_diff]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_NAME=$(basename "$DATASET_ROOT")
RESULTS_DIR="$REPO_ROOT/results/$RUN_NAME"
COMPARISONS_DIR="$REPO_ROOT/comparisons/$RUN_NAME"

echo "Script location: $SCRIPT_DIR"
echo "Resolved repo root: $REPO_ROOT"
echo "Looking for results in: $RESULTS_DIR"
echo "max_time_diff: ${MAX_TIME_DIFF}s"

if [ ! -d "$RESULTS_DIR" ]; then
    echo ""
    echo "ERROR: $RESULTS_DIR not found."
    echo "If your actual layout is different, this is the exact path that's wrong --"
    echo "tell me what 'find ~/MasterThesis -maxdepth 3 -type d -name results' shows."
    exit 1
fi

mkdir -p "$COMPARISONS_DIR"
for vslam in dpvo orbslam3; do
    for fde in zoedepth depthanythingv2 fastdepth; do
        if [ ! -f "$RESULTS_DIR/$vslam/poses.txt" ] || [ ! -d "$RESULTS_DIR/$fde" ]; then
            echo "SKIP: $vslam + $fde -- $RESULTS_DIR/$vslam/poses.txt or $RESULTS_DIR/$fde missing"
            continue
        fi
        echo "== [$RUN_NAME] $vslam + $fde =="
        python3 "$SCRIPT_DIR/compare_reconstruction_3d.py" \
            --dataset_root "$DATASET_ROOT" \
            --vslam_dir "$RESULTS_DIR/$vslam" \
            --fde_dir "$RESULTS_DIR/$fde" \
            --output "$COMPARISONS_DIR/${vslam}_${fde}.html" \
            --max_time_diff "$MAX_TIME_DIFF" \
            --no_open
        echo ""
    done
done

echo "All done. Open any result with: explorer.exe $COMPARISONS_DIR/<vslam>_<fde>.html"
echo "Numeric summaries: $COMPARISONS_DIR/*_summary.json"