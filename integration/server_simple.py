r"""
ПРОСТИЙ СЕРВЕР. Приймає кадри, будує карту довіри, віддає її назад.

===========================================================================
ЩО ВІН РОБИТЬ, ОДНИМ АБЗАЦОМ
===========================================================================

Хтось шле тобі кадри. У кадрі може бути по-різному:

  варіант А   лише картинка              -> ти сам рахуєш глибину і позу
  варіант Б   картинка + глибина + поза  -> береш готове

Сервер розрізняє це сам. Якщо глибини або пози в кадрі немає, він кличе
функцію `my_vslam_and_fde()` нижче. Якщо є, кличе її ТІЛЬКИ по розріджені
точки, бо їх мало хто шле, а без них карти довіри не буде.

Далі він будує карту довіри і шле назад дві речі:
  conf2d      наскільки можна вірити кожному пікселю цього кадру
  grid_delta  які вокселі тривимірної карти цей кадр змінив

===========================================================================
ЩО ТОБІ ТРЕБА ЗРОБИТИ
===========================================================================

Знайди нижче рядок `ТУТ ТВІЙ КОД` і встав туди виклики свого VSLAM і FDE.
Більше нічого міняти не треба.

Запуск:
    python server_simple.py                       приймає на 9000, віддає на 9001
    python server_simple.py --tcp                 через інтернет
    python server_simple.py --fake                без свого коду, на заглушках

`--fake` корисний, щоб спершу переконатись, що зв'язок працює: замість
справжніх VSLAM і FDE підставляються вигадані числа.
"""
import argparse
import os
import sys
import time

import numpy as np

# друкувати одразу, а не коли наповниться буфер: інакше у
# перенаправленому виводі нічого не видно, поки сервер працює
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from autopilot import ReactiveAutopilot, command_to_vector
from realtime_map import RealtimeConfidenceMapper
from udp_streaming import FrameReceiver, MapPublisher, TcpFrameServer, RawJpegFrameReceiver


# ===========================================================================
#  ТУТ ТВІЙ КОД
# ===========================================================================

_zoedepth_pipe = [None]
_vo_step = [0]
_use_gpu = [False]      # виставляється з --gpu у main(), безпечний дефолт -- CPU


def _get_zoedepth(use_gpu):
    """Та сама модель, що й src/depth/zoedepth/infer.py -- завантажується
    раз, при першому кадрі, а не на кожен виклик.

    CPU за замовчуванням: на цій машині CUDA (torch) усередині WSL2
    двічі поклала весь WSL -- dxgkrnl (віртуалізація GPU для WSL2) кидає
    "field-spanning write" у dxgvmb_send_wait_sync_object_gpu і валить VM.
    Це баг драйвера (Windows build 26200 + nvidia 610.57.01, обидва дуже
    свіжі), не нашого коду. --gpu свідомо увімкне ризик ще раз."""
    if _zoedepth_pipe[0] is None:
        import torch
        from transformers import pipeline
        device_idx = 0 if (use_gpu and torch.cuda.is_available()) else -1
        print(f"завантажую ZoeDepth ({'GPU -- ризик краху WSL!' if device_idx == 0 else 'CPU'})...")
        _zoedepth_pipe[0] = pipeline("depth-estimation", model="Intel/zoedepth-nyu-kitti",
                                      device=device_idx)
        print("ZoeDepth завантажено\n")
    return _zoedepth_pipe[0]


def my_vslam_and_fde(image, have_depth, have_pose):
    """Жива глибина (ZoeDepth, та сама модель, що й в офлайн-конвеєрі),
    поза ПОКИ фіктивна -- DPVO/ORB-SLAM3 не перероблені на покадровий
    виклик (вони читають цілу теку кадрів наперед, а не один кадр по UDP).
    Коли буде живий VSLAM, замінити блок `if not have_pose` на його виклик.
    """
    from PIL import Image

    out = {}
    h, w = image.shape[:2]

    if not have_depth:
        pil_img = Image.fromarray(image if image.ndim == 3 else
                                   np.stack([image] * 3, axis=-1)).convert("RGB")
        pred = _get_zoedepth(_use_gpu[0])(pil_img)["predicted_depth"]
        depth = pred.squeeze().cpu().numpy().astype(np.float32)
        if depth.shape != (h, w):
            depth = np.array(Image.fromarray(depth).resize((w, h), Image.BILINEAR))
        out["depth"] = depth

    fx = fy = 0.8 * w
    cx, cy = w / 2.0, h / 2.0
    out["K"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    T = np.eye(4)
    T[0, 3] = 0.3 * _vo_step[0]        # фіктивний рівномірний рух уздовж X
    _vo_step[0] += 1
    if not have_pose:
        out["T_wc"] = T

    d = out.get("depth")
    if d is not None:
        rng = np.random.default_rng(_vo_step[0])
        vs = rng.integers(0, h, 300)
        us = rng.integers(0, w, 300)
        z = d[vs, us].astype(np.float64)
        pc = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z, np.ones_like(z)], axis=1)
        out["landmarks"] = (T @ pc.T).T[:, :3].astype(np.float32)
    return out


# ===========================================================================
#  Заглушка для --fake. Нічого не рахує, просто вигадує правдоподібні числа,
#  щоб можна було перевірити, що кадри ходять туди й назад.
# ===========================================================================

_fake_step = [0]


def fake_vslam_and_fde(image, have_depth, have_pose):
    h, w = image.shape[:2]
    out = {}
    fx = fy = 0.8 * w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    out["K"] = K

    if not have_depth:
        # «глибина»: підлога знизу, стіна попереду
        v = np.arange(h)[:, None] * np.ones((1, w))
        d = np.full((h, w), 9.0, dtype=np.float32)
        low = v > cy + 20
        d[low] = (1.7 * fy / (v[low] - cy)).astype(np.float32)
        out["depth"] = np.clip(d, 0.3, 50.0)

    T = np.eye(4)
    T[0, 3] = 0.3 * _fake_step[0]        # «летимо» вздовж X
    _fake_step[0] += 1
    if not have_pose:
        out["T_wc"] = T

    # «точки VSLAM»: беремо глибину, яку маємо, і переводимо у світ
    d = out.get("depth")
    if d is not None:
        rng = np.random.default_rng(_fake_step[0])
        vs = rng.integers(0, h, 300)
        us = rng.integers(0, w, 300)
        z = d[vs, us].astype(np.float64)
        pc = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z, np.ones_like(z)], axis=1)
        out["landmarks"] = (T @ pc.T).T[:, :3].astype(np.float32)
    return out


# ===========================================================================
#  Далі службове, міняти не треба
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000, help="сюди приходять кадри")
    ap.add_argument("--reply_port", type=int, default=9001, help="туди йде карта")
    ap.add_argument("--tcp", action="store_true",
                     help="через інтернет. Одне з'єднання в обидва боки, втрати "
                          "повторюються. У локальній мережі краще без нього")
    ap.add_argument("--raw_jpeg", action="store_true",
                     help="відправник шле НЕ протокол цього проекту (client.py), а просто "
                          "JPEG-картинку одним UDP-пакетом, можливо зі своїм заголовком "
                          "перед нею. Решта (depth/T_wc/K/landmarks) рахується як для "
                          "кадру 'лише картинка' -- через my_vslam_and_fde() або --fake")
    ap.add_argument("--fake", action="store_true",
                     help="підставити заглушку замість твоїх VSLAM і FDE")
    ap.add_argument("--radius", type=float, default=60.0, help="радіус карти, м")
    ap.add_argument("--voxel", type=float, default=1.0, help="розмір вокселя, м")
    ap.add_argument("--max_depth", type=float, default=50.0)
    ap.add_argument("--no_autopilot", action="store_true",
                     help="не рахувати команду руху. Типово вона рахується і летить назад "
                          "разом з картою як вектор з 11 чисел")
    ap.add_argument("--drone_radius", type=float, default=0.5)
    ap.add_argument("--max_speed", type=float, default=3.0)
    ap.add_argument("--cautious", action="store_true",
                     help="з --autopilot: невідомі пікселі вважати можливою перешкодою, а не "
                          "порожнечею. Обережніше, але повільніше")
    ap.add_argument("--raw_vector", action="store_true",
                     help="приймач команди не розбирає протокол OVM1 -- слати йому замість "
                          "цього один сирий пакет: 11 float64 підряд, порядок як у "
                          "autopilot.COMMAND_FIELDS (timestamp, dir_x..z, target_x..z, speed, "
                          "clearance_m, azimuth_deg, stop). На приймачі: "
                          "np.frombuffer(data, dtype=np.float64)")
    ap.add_argument("--vector_only", action="store_true",
                     help="з --raw_vector: НЕ слати conf2d і grid_delta, лише сам вектор. "
                          "Без цього на той самий порт 9001 ще й далі летить карта довіри "
                          "й приріст вокселів у протоколі OVM1 -- якщо приймач читає з порту "
                          "все підряд як 11 чисел, саме це і виглядає як 'фігня'")
    ap.add_argument("--save_to", default=None,
                     help="куди писати карту, наприклад ../live_map. Без цього не пише")
    ap.add_argument("--show", action="store_true",
                     help="відкрити вікно з останнім отриманим кадром (tkinter, живе у WSL "
                          "через WSLg). Esc/закрити вікно -- зупиняє і сервер")
    ap.add_argument("--gpu", action="store_true",
                     help="рахувати ZoeDepth на GPU замість CPU. На цій машині двічі клало "
                          "весь WSL (баг dxgkrnl, не нашого коду) -- вмикай свідомо, після "
                          "'wsl --update' на Windows-боці, і май на увазі ризик")
    args = ap.parse_args()

    compute = fake_vslam_and_fde if args.fake else my_vslam_and_fde
    _use_gpu[0] = args.gpu
    if args.fake:
        print("УВАГА: --fake, працюють заглушки, а не твої VSLAM і FDE\n")
    elif args.gpu:
        print("УВАГА: --gpu -- на цій машині це вже двічі клало весь WSL\n")

    if args.tcp:
        rx = TcpFrameServer("0.0.0.0", args.port).start()
        print(f"TCP, чекаю з'єднання на порту {args.port}")
        print("відповідь піде тим самим з'єднанням\n")
    elif args.raw_jpeg:
        rx = RawJpegFrameReceiver("0.0.0.0", args.port)
        print(f"UDP (сирий JPEG, без протоколу OVM1), чекаю кадри на порту {args.port}")
        print(f"карту віддаю на порт {args.reply_port} тому, хто під'єднався\n")
    else:
        rx = FrameReceiver("0.0.0.0", args.port)
        print(f"UDP, чекаю кадри на порту {args.port}")
        print(f"карту віддаю на порт {args.reply_port} тому, хто під'єднався\n")

    mapper = None
    pub = None
    pilot = (None if args.no_autopilot else
             ReactiveAutopilot(drone_radius=args.drone_radius, max_speed=args.max_speed,
                                unknown_is_risk=args.cautious))
    if pilot:
        print(f"автопілот увімкнений, радіус апарата {args.drone_radius} м, "
              f"до {args.max_speed} м/с, невідоме = "
              f"{'ризик' if args.cautious else 'порожньо'}\n")
    n = 0
    t0 = time.monotonic()

    view = None
    if args.show:
        from live_view import LiveImageView
        view = LiveImageView("вхідні кадри")

    try:
        for frame in rx.frames():
            if view is not None and view.closed:
                print("вікно закрито, зупиняюсь")
                break

            image = frame.get("image")
            depth = frame.get("depth")
            T_wc = frame.get("T_wc")
            K = frame.get("K")
            landmarks = frame.get("landmarks")

            if view is not None and image is not None:
                view.show(image, status_text=f"кадр {n + 1}")

            if image is None and depth is None:
                print("кадр без зображення і без глибини, пропускаю")
                continue

            # --- чого бракує, те рахуємо самі -----------------------------
            need = (depth is None) or (T_wc is None) or (landmarks is None) or (K is None)
            if need:
                src = image if image is not None else depth
                extra = compute(src, depth is not None, T_wc is not None) or {}
                depth = depth if depth is not None else extra.get("depth")
                T_wc = T_wc if T_wc is not None else extra.get("T_wc")
                K = K if K is not None else extra.get("K")
                landmarks = landmarks if landmarks is not None else extra.get("landmarks")

            if depth is None or T_wc is None or K is None:
                print("після обробки все одно немає глибини, пози або K, пропускаю кадр")
                continue

            # --- сітка виділяється за першою позою ------------------------
            if mapper is None:
                c = np.asarray(T_wc, dtype=np.float64)[:3, 3]
                mapper = RealtimeConfidenceMapper(
                    c - args.radius, c + args.radius, args.voxel,
                    use_sky_cue=True, max_depth=args.max_depth, build_naive=False)
                print(f"карта {2*args.radius:.0f} м навколо ({c[0]:.1f}, {c[1]:.1f}, "
                      f"{c[2]:.1f}), воксель {args.voxel} м, "
                      f"сітка {'x'.join(str(int(d)) for d in mapper.adaptive_grid.dims)}\n")

            # --- вето неба хоче яскравість, не колір ----------------------
            grey = None
            if image is not None:
                grey = image.mean(axis=2).astype(np.uint8) if image.ndim == 3 else image

            res = mapper.process(depth, T_wc, K, image=grey, landmarks=landmarks)
            n += 1

            # --- куди відповідати ------------------------------------------
            if args.tcp:
                pub = frame.get("link")
            elif pub is None and frame.get("src_addr"):
                pub = MapPublisher(frame["src_addr"][0], args.reply_port)
                print(f"відповідаю на {frame['src_addr'][0]}:{args.reply_port}\n")

            if res.get("confidence") is None:
                print(f"кадр {n}: {res.get('reason', 'карти довіри немає')}")
                continue

            if pub is not None and not args.vector_only:
                pub.send_conf2d(frame["frame_id"], frame["timestamp"], res["confidence"])
                if len(res["delta_idx"]):
                    pub.send_grid_delta(frame["frame_id"], mapper.adaptive_grid,
                                         res["delta_idx"], res["delta_values"],
                                         timestamp=frame["timestamp"])

            line = (f"кадр {n:4d}  у карту {100*res['kept_fraction']:3.0f}% пікселів, "
                    f"довіра {res['mean_weight']:.2f}, змінено {len(res['delta_idx']):5d} "
                    f"вокселів, усього зайнято "
                    f"{int(mapper.adaptive_grid.occupied_mask().sum())}")
            if pilot is not None:
                cmd = pilot.step(res["confidence"], res["depth_calibrated"], K, T_wc)
                if pub is not None:
                    if args.raw_vector:
                        vec = command_to_vector(cmd, timestamp=frame["timestamp"])
                        pub.send_raw(vec.astype("<f8").tobytes())
                    else:
                        pub.send_command(frame["frame_id"], frame["timestamp"], cmd)
                line += f"\n          автопілот: {pilot.reason(cmd)}"
            print(line)

            if args.save_to and n % 25 == 0:
                os.makedirs(args.save_to, exist_ok=True)
                np.savez_compressed(os.path.join(args.save_to, "map.npz"),
                                    log_odds=mapper.adaptive_grid.log_odds,
                                    origin=mapper.adaptive_grid.origin,
                                    resolution=mapper.adaptive_grid.resolution)
    except KeyboardInterrupt:
        print("\nзупинено")
    finally:
        rx.close()
        if view is not None:
            view.close()

    if mapper is not None:
        dt = time.monotonic() - t0
        print(f"\nоброблено {n} кадрів за {dt:.0f} с")
        for k, v in mapper.summary().items():
            print(f"  {k}: {v}")
        if args.save_to:
            os.makedirs(args.save_to, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_to, "map.npz"),
                                log_odds=mapper.adaptive_grid.log_odds,
                                origin=mapper.adaptive_grid.origin,
                                resolution=mapper.adaptive_grid.resolution)
            print(f"  карта збережена у {args.save_to}/map.npz")


if __name__ == "__main__":
    main()
