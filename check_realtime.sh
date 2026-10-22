#!/bin/bash
# Самоперевірка потокового режиму. Нічого не питає, нічого не псує, пише у
# тимчасову теку і сам її прибирає. Кожна перевірка друкує ТАК або НІ, і
# наприкінці підсумок. Код виходу 0 означає, що все пройшло.
#
#   ./check_realtime.sh            усе, крім повільної офлайн-регресії
#   ./check_realtime.sh --full     плюс офлайн-регресія, ще близько хвилини
#   ./check_realtime.sh --keep     не прибирати тимчасову теку
set -u
cd "$(dirname "$0")"
ROOT=$PWD
FULL=0; KEEP=0
for a in "$@"; do [ "$a" = "--full" ] && FULL=1; [ "$a" = "--keep" ] && KEEP=1; done

TMP=$(mktemp -d /tmp/check_rt.XXXXXX)
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
  sleep 0.5
  for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null; done
  [ "$KEEP" = "1" ] && echo "тимчасова тека лишена: $TMP" || rm -rf "$TMP"
}
trap cleanup EXIT

PASS=0; FAIL=0
ok()   { printf '  \033[32mТАК\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mНІ \033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
note() { printf '       %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ----------------------------------------------------------------- 1. оточення
head_ "1. Оточення"
if [ ! -x .venv/bin/python3 ]; then
  bad "немає .venv/bin/python3. Відновити: uv venv --python 3.12 && VIRTUAL_ENV=\$PWD/.venv uv pip install numpy scipy pillow tqdm matplotlib plotly pyyaml evo"
  echo; echo "Далі без оточення немає сенсу."; exit 1
fi
PY=$ROOT/.venv/bin/python3
if $PY -c "import numpy,scipy,PIL,matplotlib,plotly,yaml,tqdm,evo" 2>/dev/null; then
  ok "усі бібліотеки на місці"
else
  bad "бракує бібліотек. VIRTUAL_ENV=\$PWD/.venv uv pip install numpy scipy pillow tqdm matplotlib plotly pyyaml evo"
  exit 1
fi

DS=$(ls -d datasets/euroc/*_quadrotor 2>/dev/null | head -1)
if [ -n "$DS" ] && [ -d "$DS/cam0/data" ]; then ok "набір euroc знайдено: $DS"
else bad "немає datasets/euroc/*_quadrotor"; exit 1; fi
for d in results/euroc/dpvo results/euroc/zoedepth; do
  [ -d "$d" ] && ok "є $d" || { bad "немає $d"; exit 1; }
done
[ -f results/euroc/dpvo/dpvo_sparse_map.ply ] \
  && ok "є розріджені точки DPVO" \
  || bad "немає dpvo_sparse_map.ply: карта довіри не збудується"
[ -f refiners/loo_euroc_zoedepth.npz ] && ok "є таблиця корекції" || bad "немає refiners/loo_euroc_zoedepth.npz"

# ------------------------------------------------------- 2. кодування повідомлень
head_ "2. Кодування і декодування повідомлень"
if $PY - <<'PYEOF' > "$TMP/codec.txt" 2>&1
import sys; sys.path.insert(0, "src")
import numpy as np, udp_streaming as u
d = np.random.rand(48, 64).astype(np.float32) * 20
img = (np.random.rand(48, 64) * 255).astype(np.uint8)
lm = np.random.rand(37, 3).astype(np.float32) * 10
f = u.decode_frame2(u.encode_frame2(1.5, d, np.eye(4), pose_x=3.0, K=np.eye(3), image=img, landmarks=lm))
assert np.allclose(f["depth"], d) and np.array_equal(f["image"], img) and np.allclose(f["landmarks"], lm)
f2 = u.decode_frame2(u.encode_frame2(2.0, d, np.eye(4)))
assert f2["image"] is None and f2["landmarks"] is None
c = np.where(np.random.rand(48, 64) > 0.3, np.random.rand(48, 64), np.nan)
r = u.decode_conf2d(u.encode_conf2d(9.0, c, depth=d))
assert np.array_equal(np.isnan(r["confidence"]), np.isnan(c))
idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
g = u.decode_grid_delta(u.encode_grid_delta(1.0, np.zeros(3), (10, 11, 12), idx,
                                             np.array([0.5, -0.4], np.float32)))
assert np.array_equal(g["idx"], idx) and g["dims"] == (10, 11, 12)
print("ok")
PYEOF
then ok "кадр, карта довіри і приріст кодуються без втрат"
else bad "круговий тест кодування впав"; sed 's/^/       /' "$TMP/codec.txt" | tail -5; fi

# --------------------------------------------- 3. приріст проти повного порівняння
head_ "3. Приріст воксельної карти рахується правильно"
if $PY - <<'PYEOF' > "$TMP/delta.txt" 2>&1
import sys; sys.path.insert(0, "src")
import numpy as np
from occupancy_3d_map import ProbabilisticVoxelGrid
rng = np.random.default_rng(0)
pts = rng.random((500, 3)) * 20; w = rng.random(500); o = np.zeros(3)
a = ProbabilisticVoxelGrid([-5]*3, [25]*3, 1.0)
before = a.log_odds.copy(); a.insert_weighted(pts, w, o, n_ray_samples=4)
ref = np.sort(np.flatnonzero((before != a.log_odds).ravel()))
b = ProbabilisticVoxelGrid([-5]*3, [25]*3, 1.0)
b.insert_weighted(pts, w, o, n_ray_samples=4, track_changes=True)
assert np.array_equal(a.log_odds, b.log_odds), "сітки розійшлись"
assert np.array_equal(ref, np.sort(b.last_touched)), "індекси не збігаються"
print(f"ok {len(ref)}")
PYEOF
then ok "швидкий приріст дає ті самі вокселі, що й повне порівняння ($(cut -d' ' -f2 "$TMP/delta.txt") шт.)"
else bad "приріст не збігається з повним порівнянням"; sed 's/^/       /' "$TMP/delta.txt" | tail -5; fi

# ----------------------------------------------------- 4. наскрізний потік по UDP
head_ "4. Наскрізний потік по UDP"
P_IN=9500; P_OUT=9501
# idle_timeout у приймача коротший за серверний, щоб він завершився і записав
# накопичену карту одразу після кінця потоку, а не через пів хвилини
"$PY" src/udp_conf_receiver.py --port $P_OUT --output_dir "$TMP/live" \
      --idle_timeout 6 > "$TMP/rx.log" 2>&1 &
PIDS+=($!)
sleep 1
"$PY" src/occupancy_3d_map.py --mode udp --listen_port $P_IN \
      --send_host 127.0.0.1 --send_port $P_OUT \
      --dataset_root "$DS" --depth_refiner refiners/loo_euroc_zoedepth.npz --sky_cue \
      --bounds_radius 60 --voxel_resolution 1.0 --max_depth 50 \
      --output_dir "$TMP/srv" --idle_timeout 15 --stats_every 1000 > "$TMP/srv.log" 2>&1 &
PIDS+=($!)
sleep 3
"$PY" src/udp_replay_sender.py --dataset_root "$DS" \
      --vslam_dir results/euroc/dpvo --fde_dir results/euroc/zoedepth \
      --host 127.0.0.1 --port $P_IN --n_frames 30 --rate 5 --send_intrinsics \
      > "$TMP/tx.log" 2>&1
SENT=$(grep -c '^\[replay\] кадр' "$TMP/tx.log")
[ "$SENT" -ge 25 ] && ok "відправлено $SENT кадрів" || bad "відправлено лише $SENT кадрів"

note "чекаю, поки сервер і споживач доробили..."
for _ in $(seq 1 60); do grep -q "збережено у" "$TMP/srv.log" && break; sleep 1; done
# Приймач пише накопичену карту на власному виході, тобто ПІЗНІШЕ за сервер.
# Без цього очікування порівнювати ще немає чого.
for _ in $(seq 1 40); do grep -q "отримано:" "$TMP/rx.log" && break; sleep 1; done

CONF=$(grep -c '^\[conf2d\]' "$TMP/rx.log")
[ "$CONF" -ge 20 ] && ok "споживач отримав $CONF карт довіри" || bad "споживач отримав лише $CONF карт довіри"
CAL=$(grep -oP 'calibrated: \K[0-9]+' "$TMP/srv.log" | tail -1)
[ "${CAL:-0}" -ge 20 ] && ok "кадрів з калібруванням: $CAL" \
  || bad "кадрів з калібруванням лише ${CAL:-0}: перевір розріджені точки"
NOCAL=$(grep -oP 'frames_no_calib: \K[0-9]+' "$TMP/srv.log" | tail -1)
[ "${NOCAL:-1}" = "0" ] && ok "жоден кадр не лишився без калібрування" \
  || bad "без калібрування ${NOCAL} кадрів"
HZ=$(grep -oP 'межа близько \K[0-9.]+' "$TMP/srv.log" | tail -1)
note "швидкодія: ${HZ:-?} Гц, сітка $(grep -oP 'сітка \K[0-9x]+(?= вокселів)' "$TMP/srv.log" | tail -1)"

if [ -f "$TMP/srv/adaptive_grid.npz" ] && [ -f "$TMP/live/adaptive_live.npz" ]; then
  if $PY - "$TMP" <<'PYEOF' > "$TMP/cmp.txt" 2>&1
import sys, numpy as np
t = sys.argv[1]
a = np.load(f"{t}/srv/adaptive_grid.npz"); b = np.load(f"{t}/live/adaptive_live.npz")
assert np.allclose(a["origin"], b["origin"]), "різний початок сітки"
assert np.array_equal(a["log_odds"], b["log_odds"]), "масиви різні"
print(f"ok {int((a['log_odds'] > 0).sum())}")
PYEOF
  then ok "КАРТА В СПОЖИВАЧА ІДЕНТИЧНА СЕРВЕРНІЙ ($(cut -d' ' -f2 "$TMP/cmp.txt") зайнятих вокселів)"
  else bad "карта в споживача НЕ збігається з серверною"; sed 's/^/       /' "$TMP/cmp.txt" | tail -3; fi
else
  bad "немає файлів карти для порівняння"
fi

# ------------------------------------------------- 5. кадри без розріджених точок
head_ "5. Поведінка без розріджених точок VSLAM"
P2=9510
"$PY" src/occupancy_3d_map.py --mode udp --listen_port $P2 \
      --dataset_root "$DS" --depth_refiner refiners/loo_euroc_zoedepth.npz \
      --bounds_radius 40 --voxel_resolution 2.0 --max_depth 50 \
      --output_dir "$TMP/srv2" --idle_timeout 8 --stats_every 1000 --max_frames 10 \
      > "$TMP/srv2.log" 2>&1 &
PIDS+=($!)
sleep 2
"$PY" src/udp_replay_sender.py --dataset_root "$DS" \
      --vslam_dir results/euroc/dpvo --fde_dir results/euroc/zoedepth \
      --host 127.0.0.1 --port $P2 --n_frames 10 --rate 4 --send_intrinsics \
      --no_landmarks > /dev/null 2>&1
for _ in $(seq 1 25); do grep -q "оброблено" "$TMP/srv2.log" && break; sleep 1; done
NOLM=$(grep -oP 'frames_no_landmarks: \K[0-9]+' "$TMP/srv2.log" | tail -1)
if [ "${NOLM:-0}" -ge 5 ] && grep -q "без розріджених точок" "$TMP/srv2.log"; then
  ok "сервер чесно каже, що без точок карти довіри не буде ($NOLM кадрів)"
else
  bad "сервер не повідомив про відсутність точок"
fi

# ------------------------------------------------------- 6. офлайн не зламано
if [ "$FULL" = "1" ]; then
  head_ "6. Офлайновий режим не зламано (повільно)"
  IOU=$(cd src && "$PY" occupancy_3d_map.py --mode dataset \
        --dataset_root "../$DS" --vslam_dir ../results/euroc/dpvo \
        --fde_dir ../results/euroc/zoedepth \
        --depth_refiner ../refiners/loo_euroc_zoedepth.npz \
        --output_dir "$TMP/regress" --n_frames 80 --gt_compare 2>/dev/null \
        | awk '/^\[gt_compare\] adaptive:/{f=1;next} f&&/thr=0.00/{print;exit}' \
        | grep -oP 'IoU=\K[0-9.]+')
  if [ "$IOU" = "0.4890" ]; then ok "IoU 0.4890, точно як до змін"
  else bad "IoU ${IOU:-?}, очікувалось 0.4890"; fi
else
  note ""
  note "Офлайнова регресія пропущена. Щоб і її: ./check_realtime.sh --full"
fi

# ------------------------------------------------------------------- підсумок
head_ "Підсумок"
printf "  пройдено %d, не пройдено %d\n\n" "$PASS" "$FAIL"
if [ "$FAIL" = "0" ]; then
  printf '  \033[32mВсе працює.\033[0m\n'
  exit 0
else
  printf '  \033[31mЄ проблеми, дивись рядки з НІ вище.\033[0m\n'
  exit 1
fi
