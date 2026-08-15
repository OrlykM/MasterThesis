#!/bin/bash
# Usage inside the container: entrypoint.sh <dataset_root> <output_dir> [stride] [extra demo.py args...]
# <dataset_root> must contain cam0/data/ and cam0/sensor.yaml.
#
# Anything after [stride] is passed straight through to demo.py -- e.g.
# to enable DPV-SLAM's loop closure or override any config knob:
#   docker compose run --rm dpvo /data /out 1 --opts LOOP_CLOSURE True PATCHES_PER_FRAME 128
set -e

DATASET_ROOT=${1:-/data}
OUTPUT_DIR=${2:-/out}
STRIDE=${3:-1}
shift $(( $# > 3 ? 3 : $# ))
EXTRA_ARGS=("$@")
RUN_NAME="result"

mkdir -p "$OUTPUT_DIR"

# calib.txt from THIS dataset's real intrinsics -- the bundled calib/euroc.txt
# is calibrated for the real ETH EuRoC benchmark cameras, not this one.
/opt/conda/envs/dpvo/bin/python -c "
import re
with open('$DATASET_ROOT/cam0/sensor.yaml') as f:
    text = f.read()
m = re.search(r'intrinsics\s*:\s*\[([^\]]+)\]', text)
vals = [float(x.strip()) for x in m.group(1).split(',')]
print(' '.join(f'{v:.6f}' for v in vals))
" > /DPVO/calib/dataset.txt
CALIB_VALS=$(cat /DPVO/calib/dataset.txt)
echo "calib.txt: $CALIB_VALS"

source /opt/conda/etc/profile.d/conda.sh
conda activate dpvo
cd /DPVO

T0=$(date +%s)
python demo.py \
    --imagedir="$DATASET_ROOT/cam0/data" \
    --calib=calib/dataset.txt \
    --network=/DPVO/dpvo.pth \
    --stride="$STRIDE" \
    --name="$RUN_NAME" \
    --save_trajectory \
    --save_ply \
    "${EXTRA_ARGS[@]}"
ELAPSED=$(( $(date +%s) - T0 ))

cp "/DPVO/saved_trajectories/${RUN_NAME}.txt" "$OUTPUT_DIR/poses.txt"
echo "Saved: $OUTPUT_DIR/poses.txt"

# DPVO's own sparse reconstruction (its internal tracked-patch point cloud --
# NOT the same as the dense FDE-depth-backprojected reconstruction your own
# pipeline builds; this is DPVO's own map, useful as a side comparison).
# Location isn't hardcoded here in case the exact save path differs by
# version -- searched for after the run instead of guessed.
PLY_FILE=$(find /DPVO -name "${RUN_NAME}*.ply" 2>/dev/null | head -1)
if [ -n "$PLY_FILE" ]; then
    cp "$PLY_FILE" "$OUTPUT_DIR/dpvo_sparse_map.ply"
    echo "Saved: $OUTPUT_DIR/dpvo_sparse_map.ply (DPVO's own sparse point cloud)"
else
    echo "NOTE: --save_ply produced no .ply file found under /DPVO -- check the log above for where it went."
fi

# run_config.json -- exact params this run used
N_FRAMES=$(ls "$DATASET_ROOT/cam0/data" | wc -l)
cat > "$OUTPUT_DIR/run_config.json" <<EOF
{
  "algorithm": "DPVO",
  "source_repo": "https://github.com/princeton-vl/DPVO",
  "network_checkpoint": "/DPVO/dpvo.pth",
  "calib": "$CALIB_VALS",
  "params": {
    "stride": $STRIDE,
    "extra_args": "${EXTRA_ARGS[*]}"
  },
  "n_frames_in_dataset": $N_FRAMES,
  "elapsed_seconds": $ELAPSED,
  "timestamp": "$(date -Iseconds)"
}
EOF
echo "Saved: $OUTPUT_DIR/run_config.json"