set -e

DATASET_ROOT=${1:-/data}
OUTPUT_DIR=${2:-/out}
STRIDE=${3:-1}
RUN_NAME="result"

mkdir -p "$OUTPUT_DIR"

/opt/conda/envs/dpvo/bin/python -c "
import re
with open('$DATASET_ROOT/cam0/sensor.yaml') as f:
    text = f.read()
m = re.search(r'intrinsics\s*:\s*\[([^\]]+)\]', text)
vals = [float(x.strip()) for x in m.group(1).split(',')]
print(' '.join(f'{v:.6f}' for v in vals))
" > /DPVO/calib/dataset.txt
echo "calib.txt: $(cat /DPVO/calib/dataset.txt)"

source /opt/conda/etc/profile.d/conda.sh
conda activate dpvo
cd /DPVO

python demo.py \
    --imagedir="$DATASET_ROOT/cam0/data" \
    --calib=calib/dataset.txt \
    --network=/DPVO/dpvo.pth \
    --stride="$STRIDE" \
    --name="$RUN_NAME" \
    --save_trajectory

cp "/DPVO/saved_trajectories/${RUN_NAME}.txt" "$OUTPUT_DIR/poses.txt"
echo "Saved: $OUTPUT_DIR/poses.txt"