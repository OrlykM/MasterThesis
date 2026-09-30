#!/bin/bash
# Usage inside the container: entrypoint.sh <mav0_dir> <output_dir>
# <mav0_dir> is where the dataset is mounted -- docker-compose.yml mounts the
# dataset root directly AT this path (target ".../mav0"), because ORB-SLAM3's
# EuRoC loader expects <sequence_root>/mav0/cam0/data internally and this
# dataset has no mav0/ wrapper of its own.
#
# ORB-extractor tuning is overridable via environment variables, e.g.:
#   docker compose run --rm -e ORB_NFEATURES=2000 -e ORB_SCALEFACTOR=1.1 orbslam3 /data/mav0 /out
set -e

MAV0_DIR=${1:-/data/mav0}
OUTPUT_DIR=${2:-/out}
SEQ_ROOT=$(dirname "$MAV0_DIR")
RUN_NAME="result"

# ORB-extractor parameters -- defaults match ORB_SLAM3's own example configs,
# override any of these via -e when you need to tune tracking behavior.
ORB_NFEATURES=${ORB_NFEATURES:-1500}
ORB_SCALEFACTOR=${ORB_SCALEFACTOR:-1.2}
ORB_NLEVELS=${ORB_NLEVELS:-8}
ORB_INITHFAST=${ORB_INITHFAST:-20}
ORB_MINTHFAST=${ORB_MINTHFAST:-7}

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
fps = round(float(fps_m.group(1))) if fps_m else 30
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
ORBextractor.nFeatures: $ORB_NFEATURES
ORBextractor.scaleFactor: $ORB_SCALEFACTOR
ORBextractor.nLevels: $ORB_NLEVELS
ORBextractor.iniThFAST: $ORB_INITHFAST
ORBextractor.minThFAST: $ORB_MINTHFAST

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0''')
" > /tmp/camera.yaml
echo "camera.yaml generated (fps=$(grep Camera.fps /tmp/camera.yaml))"

cd /ORB_SLAM3
T0=$(date +%s)
./Examples/Monocular/mono_euroc \
    Vocabulary/ORBvoc.txt /tmp/camera.yaml \
    "$SEQ_ROOT" /tmp/times.txt "$RUN_NAME"
ELAPSED=$(( $(date +%s) - T0 ))

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
cp /tmp/camera.yaml "$OUTPUT_DIR/camera.yaml"  # full settings file used, for exact reproducibility
echo "Saved: $OUTPUT_DIR/poses.txt"

# landmarks.txt -- per-frame tracked sparse map points (see
# patch_landmarks_export.py), written to CWD (/ORB_SLAM3) by the patched
# mono_euroc.cc. Used by depth_calibration.py to recalibrate FDE depth
# against ORB-SLAM3's own sparse map, same role DPVO's dpvo_sparse_map.ply
# already plays for the dpvo combos.
if [ -f landmarks.txt ]; then
    cp landmarks.txt "$OUTPUT_DIR/landmarks.txt"
    echo "Saved: $OUTPUT_DIR/landmarks.txt ($(wc -l < landmarks.txt) tracked map-point observations)"
else
    echo "NOTE: no landmarks.txt produced -- depth calibration will be unavailable for this run"
fi

N_FRAMES=$(wc -l < /tmp/times.txt)
cat > "$OUTPUT_DIR/run_config.json" <<EOF
{
  "algorithm": "ORB-SLAM3",
  "mode": "monocular",
  "source_repo": "https://github.com/UZ-SLAMLab/ORB_SLAM3",
  "vocabulary": "Vocabulary/ORBvoc.txt",
  "camera_settings_file": "camera.yaml (saved alongside this file)",
  "params": {
    "ORB_NFEATURES": $ORB_NFEATURES,
    "ORB_SCALEFACTOR": $ORB_SCALEFACTOR,
    "ORB_NLEVELS": $ORB_NLEVELS,
    "ORB_INITHFAST": $ORB_INITHFAST,
    "ORB_MINTHFAST": $ORB_MINTHFAST
  },
  "n_frames_in_dataset": $N_FRAMES,
  "elapsed_seconds": $ELAPSED,
  "known_limitation": "ORB-SLAM3 has no built-in FULL map export (confirmed via upstream issue #390). This build patches in per-frame tracked-point export instead (System::GetTrackedMapPoints(), a real existing API) -- see landmarks.txt alongside this file: it's the points matched in each tracked frame, not the complete map, but that's exactly what depth_calibration.py needs for scale calibration.",
  "timestamp": "$(date -Iseconds)"
}
EOF
echo "Saved: $OUTPUT_DIR/run_config.json"
