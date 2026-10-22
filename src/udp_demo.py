r"""
Жива демонстрація обох напрямків UDP одразу, щоб видно було, що саме працює.

Піднімає всі три ролі і зводить їх в одну таблицю, рядок на кадр:

    хтось під'єднується ДО нас і шле кадри   -->   ми віддаємо карту довіри
    (роль джерела, тут це записаний набір)         (роль сервера, окрема адреса)

Кожен рядок зводиться ЗА НОМЕРОМ КАДРУ, тож видно не просто «щось летить», а
що саме на кадр 7 прилетіла карта довіри саме для кадру 7, і за скільки
мілісекунд. Наприкінці підсумок і присуд.

Сервер піднімається як окремий процес, тобто той самий, що в бою. Джерело і
споживач працюють тут же, бо лише так можна заміряти час відправлення і
отримання за одним годинником.

Використання:
    python3 udp_demo.py
    python3 udp_demo.py --n_frames 30 --rate 8 --radius 60
    python3 udp_demo.py --no_landmarks     # подивитись, як виглядає відмова
"""
import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from udp_replay_sender import load_stream
from udp_streaming import FrameSender, MapSubscriber

ROOT_EUROC = "../datasets/euroc/2026-04-01_12-13-18_000_quadrotor"
G, Y, R, B, DIM, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[2m", "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=ROOT_EUROC)
    ap.add_argument("--vslam_dir", default="../results/euroc/dpvo")
    ap.add_argument("--fde_dir", default="../results/euroc/zoedepth")
    ap.add_argument("--refiner", default="../refiners/loo_euroc_zoedepth.npz")
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--rate", type=float, default=5.0, help="кадрів на секунду")
    ap.add_argument("--radius", type=float, default=60.0, help="радіус карти, м")
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--in_port", type=int, default=9600)
    ap.add_argument("--out_port", type=int, default=9601)
    ap.add_argument("--no_landmarks", action="store_true",
                     help="не слати розріджені точки: видно, як сервер відмовляється "
                          "будувати карту довіри і чесно про це каже")
    ap.add_argument("--no_image", action="store_true")
    ap.add_argument("--keep_server_log", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join("/tmp", f"udp_demo_{os.getpid()}")
    os.makedirs(out_dir, exist_ok=True)
    srv_log_path = os.path.join(out_dir, "server.log")

    print(f"{B}Демонстрація потоку по UDP{OFF}")
    print(f"  джерело  -> 127.0.0.1:{args.in_port}   (кадри: глибина, поза, K, "
          f"зображення, точки VSLAM)")
    print(f"  сервер   -> 127.0.0.1:{args.out_port}  (карта довіри і приріст вокселів)")
    print(f"  карта {2*args.radius:.0f} м, воксель {args.voxel} м\n")

    # ---------------------------------------------------------------- сервер
    cmd = [sys.executable, os.path.join(here, "occupancy_3d_map.py"), "--mode", "udp",
           "--listen_port", str(args.in_port),
           "--send_host", "127.0.0.1", "--send_port", str(args.out_port),
           "--dataset_root", args.dataset_root,
           "--bounds_radius", str(args.radius),
           "--voxel_resolution", str(args.voxel), "--max_depth", "50",
           "--output_dir", os.path.join(out_dir, "srv"),
           "--idle_timeout", "12", "--stats_every", "100000"]
    if os.path.exists(args.refiner):
        cmd += ["--depth_refiner", args.refiner, "--sky_cue"]
    srv_log = open(srv_log_path, "w")
    server = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT, cwd=here)
    print(f"{DIM}сервер піднімається...{OFF}")
    time.sleep(3.0)
    if server.poll() is not None:
        srv_log.close()
        print(f"{R}сервер не запустився:{OFF}")
        print(open(srv_log_path).read()[-1500:])
        return 1

    # -------------------------------------------------------------- споживач
    got = {}
    got_lock = threading.Lock()
    sub = MapSubscriber("0.0.0.0", args.out_port)
    stop = threading.Event()

    def consume():
        try:
            for m in sub.messages(idle_timeout=10):
                with got_lock:
                    e = got.setdefault(m["frame_id"], {})
                    if m["type"] == "conf2d":
                        c = m["confidence"]
                        e["conf_t"] = time.monotonic()
                        e["kept"] = float(np.isfinite(c).mean())
                        e["w"] = float(np.nanmean(np.where(np.isfinite(c), c, np.nan)))
                        e["shape"] = c.shape
                    elif m["type"] == "grid_delta":
                        e["delta_t"] = time.monotonic()
                        e["voxels"] = int(len(m["idx"]))
                if stop.is_set():
                    break
        finally:
            stop.set()

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    # ---------------------------------------------------------------- джерело
    K, stream = load_stream(args.dataset_root, args.vslam_dir, args.fde_dir,
                             n_frames=args.n_frames,
                             with_image=not args.no_image,
                             with_landmarks=not args.no_landmarks)
    sender = FrameSender("127.0.0.1", args.in_port)
    period = 1.0 / args.rate if args.rate > 0 else 0.0

    print(f"\n{B}{'кадр':>4}  {'надіслано':<34}  {'карта довіри назад':<30}  "
          f"{'вокселів':>8}{OFF}")
    print(DIM + "-" * 84 + OFF)

    sent_at, sent_info, n_sent = {}, {}, 0
    for fr in stream:
        fid = fr["frame_id"]
        n_lm = 0 if fr["landmarks"] is None else len(fr["landmarks"])
        sender.send_frame2(fid, fr["timestamp"], fr["depth"], fr["T_wc"],
                            pose_x=fr["pose_x"], K=K,
                            image=fr["image"], landmarks=fr["landmarks"])
        sent_at[fid] = time.monotonic()
        sent_info[fid] = (f"{fr['depth'].shape[1]}x{fr['depth'].shape[0]} глибина, "
                          f"{'img' if fr['image'] is not None else 'без img'}, "
                          f"{n_lm} точок")
        n_sent += 1
        print(f"{fid:>4}  {sent_info[fid]:<34}  {DIM}...{OFF}")
        if period:
            time.sleep(period)

    sender.close()
    print(f"\n{DIM}надіслано {n_sent}, чекаю на відповіді...{OFF}\n")

    # чекаємо, поки споживач затихне
    for _ in range(140):
        if stop.is_set():
            break
        time.sleep(0.1)
    stop.set()
    consumer.join(timeout=12)
    sub.close()
    try:
        server.wait(timeout=25)
    except subprocess.TimeoutExpired:
        server.terminate()
        server.wait(timeout=5)
    srv_log.close()

    # --------------------------------------------------------------- таблиця
    print(f"{B}{'кадр':>4}  {'надіслано':<34}  {'карта довіри назад':<30}  "
          f"{'вокселів':>8}{OFF}")
    print(DIM + "-" * 84 + OFF)
    lat, n_conf, n_delta = [], 0, 0
    for fid in sorted(sent_info):
        e = got.get(fid, {})
        if "conf_t" in e:
            ms = (e["conf_t"] - sent_at[fid]) * 1000.0
            lat.append(ms)
            n_conf += 1
            back = f"{G}{ms:5.0f} мс, {100*e['kept']:4.1f}% пікс, w={e['w']:.2f}{OFF}"
        else:
            back = f"{R}не прийшла{OFF}"
        if "voxels" in e:
            n_delta += 1
            vox = f"{e['voxels']:>8}"
        else:
            vox = f"{DIM}{'0':>8}{OFF}"
        print(f"{fid:>4}  {sent_info[fid]:<34}  {back:<39}  {vox}")

    # --------------------------------------------------------------- підсумок
    srv_text = open(srv_log_path, encoding="utf-8", errors="replace").read()

    def pick(pattern):
        import re
        m = re.findall(pattern, srv_text)
        return m[-1] if m else "?"

    print(f"\n{B}Підсумок{OFF}")
    print(f"  надіслано кадрів            {n_sent}")
    print(f"  повернулось карт довіри     {n_conf}")
    print(f"  повернулось приростів       {n_delta}  "
          f"{DIM}(кадр, що не змінив жодного вокселя, приросту не шле){OFF}")
    if lat:
        a = np.array(lat)
        print(f"  затримка туди й назад       медіана {np.median(a):.0f} мс, "
              f"найгірша {a.max():.0f} мс")
    print(f"  сервер: оброблено           {pick(r'оброблено (\d+) кадрів')}")
    print(f"  сервер: з картою довіри     {pick(r'з картою довіри (\d+)')}")
    print(f"  сервер: відкинуто простр.   {pick(r'відкинуто як прострочені (\d+)')}")
    print(f"  сервер: калібровано         {pick(r'calibrated: (\d+)')}")
    print(f"  сервер: межа швидкодії      {pick(r'межа близько ([0-9.]+) Гц')} Гц")

    ok = True
    print()
    if args.no_landmarks:
        if n_conf == 0 and "без розріджених точок" in srv_text:
            print(f"  {G}ТАК{OFF}  без точок VSLAM сервер не вигадує карту довіри "
                  f"і каже про це прямо")
        else:
            print(f"  {R}НІ {OFF}  очікувалась відмова будувати карту довіри")
            ok = False
    else:
        if n_conf == n_sent:
            print(f"  {G}ТАК{OFF}  на КОЖЕН надісланий кадр повернулась його карта довіри")
        elif n_conf > 0:
            print(f"  {Y}ЧАСТКОВО{OFF}  повернулось {n_conf} з {n_sent}: джерело швидше "
                  f"за обробку, зменш --rate або --radius")
        else:
            print(f"  {R}НІ {OFF}  не повернулось жодної карти довіри")
            ok = False
        if n_delta > 0:
            print(f"  {G}ТАК{OFF}  приріст воксельної карти доходить ({n_delta} повідомлень)")
        else:
            print(f"  {R}НІ {OFF}  приріст воксельної карти не доходить")
            ok = False

    if args.keep_server_log:
        print(f"\n  лог сервера: {srv_log_path}")
    else:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
