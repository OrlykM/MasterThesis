#!/bin/bash
# Usage inside the container: entrypoint.sh <mav0_dir> <output_dir>
# <mav0_dir> is where the dataset is mounted -- docker-compose.yml mounts the
# dataset root directly AT this path (target ".../mav0"), because ORB-SLAM3's
# EuRoC loader expects <sequence_root>/mav0/cam0/data internally and this
# dataset has no mav0/ wrapper of its own.
set -e

MAV0_DIR=${1:-/data/mav0}
OUTPUT_DIR=${2:-/out}
SEQ_ROOT=$(dirname "$MAV0_DIR")
RUN_NAME="result"

mkdir -p "$OUTPUT_DIR"

# times.txt -- header row has NO leading '#' in this dataset, skip it
# unconditionally rather than grep -v '^#' (which would NOT skip it and
# corrupt the first line with the literal word "timestamp").
tail -n +2 "$MAV0_DIR/cam0/data.csv" | cut -d',' -f1 | tr -d ' ' > /tmp/times.txt
echo "times.txt: $(wc -l < /tmp/times.txt) timestamps"

# camera.yaml from THIS dataset's real intrinsics + real fps (rate_hz) --
# not a hardcoded guess.
python3 -c "
import re
with open('$MAV0_DIR/cam0/sensor.yaml') as f:
    text = f.read()
def extract(key):
    m = re.search(rf'{key}\s*:\s*\[([^\]]+)\]', text)
    return [float(x.strip()) for x in m.group(1).split(',')] if m else None
intr = extract('intrinsics')
dist = extract('distortion_coefficients') or [0.0, 0.0, 0.0, 0.0]
res = extract('resolution')
fps_m = re.search(r'rate_hz\s*:\s*([0-9.]+)', text)
fps = float(fps_m.group(1)) if fps_m else 30.0
fx, fy, cx, cy = intr
w, h = int(res[0]), int(res[1])
print(f'''%YAML:1.0
File.version: \"1.0\"
Camera.type: \"PinHole\"
Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}
Camera1.k1: {dist[0]}
Camera1.k2: {dist[1]}
Camera1.p1: {dist[2]}
Camera1.p2: {dist[3]}
Camera.width: {w}
Camera.height: {h}
Camera.fps: {fps}
Camera.RGB: 1
ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7''')
" > /tmp/camera.yaml
echo "camera.yaml generated (fps=$(grep Camera.fps /tmp/camera.yaml))"

cd /ORB_SLAM3
./Examples/Monocular/mono_euroc \
    Vocabulary/ORBvoc.txt /tmp/camera.yaml \
    "$SEQ_ROOT" /tmp/times.txt "$RUN_NAME"

OUT_FILE=$(ls f_${RUN_NAME}*.txt 2>/dev/null | head -1)
if [ -z "$OUT_FILE" ]; then
    echo "ERROR: no trajectory file (f_${RUN_NAME}*.txt) was produced."
    echo "Common causes:"
    echo "  - mono_euroc couldn't initialize (needs enough visual features/parallax"
    echo "    in the first several frames -- a feature-poor start, e.g. mostly sky,"
    echo "    can prevent initialization entirely)"
    echo "  - times.txt row count doesn't match the number of frames in cam0/data/"
    echo "Check the tracking log printed above for the actual reason."
    exit 1
fi
cp "$OUT_FILE" "$OUTPUT_DIR/poses.txt"
echo "Saved: $OUTPUT_DIR/poses.txt"