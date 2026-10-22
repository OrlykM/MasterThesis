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
from autopilot import ReactiveAutopilot
from realtime_map import RealtimeConfidenceMapper
from udp_streaming import FrameReceiver, MapPublisher, TcpFrameServer


# ===========================================================================
#  ТУТ ТВІЙ КОД
# ===========================================================================

def my_vslam_and_fde(image, have_depth, have_pose):
    """Викликається на кожен кадр. Поверни те, чого в кадрі НЕ БУЛО.

    На вході:
        image       (H, W) або (H, W, 3) uint8, те, що прислали
        have_depth  True, якщо глибина вже прийшла і рахувати її не треба
        have_pose   True, якщо поза вже прийшла

    Треба повернути словник. Клади в нього лише те, чого бракує:

        {"depth":     (H, W) float32, метри,
         "T_wc":      (4, 4) float64, перетворення камера -> світ, метри,
         "K":         (3, 3) float64, матриця камери,
         "landmarks": (N, 3) float32, розріджені точки VSLAM у СВІТОВИХ
                      координатах, у тих самих метрах, що й T_wc}

    Чого бракує найчастіше: `landmarks`. Це ті точки, за якими твій VSLAM
    рахує своє положення. Саме вони дають масштаб, яким виправляється щільна
    глибина від нейромережі. Без них сервер побудує карту зайнятості, але не
    карту ДОВІРИ.

    Приклад, як це виглядає з DPVO і ZoeDepth:

        out = {}
        if not have_depth:
            out["depth"] = zoedepth_model.infer(image)        # метри
        if not have_pose:
            pose, points = dpvo.track(image)                  # твій VSLAM
            out["T_wc"] = pose
            out["landmarks"] = points
        else:
            out["landmarks"] = dpvo.current_map_points()
        return out
    """
    raise NotImplementedError(
        "Встав сюди свої VSLAM і FDE, або запусти з --fake, щоб спершу "
        "перевірити зв'язок на заглушках")


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
    ap.add_argument("--fake", action="store_true",
                     help="підставити заглушку замість твоїх VSLAM і FDE")
    ap.add_argument("--radius", type=float, default=60.0, help="радіус карти, м")
    ap.add_argument("--voxel", type=float, default=1.0, help="розмір вокселя, м")
    ap.add_argument("--max_depth", type=float, default=50.0)
    ap.add_argument("--autopilot", action="store_true",
                     help="рахувати ще й команду руху: напрямок, цільову точку і швидкість. "
                          "Вона друкується і летить назад разом з картою")
    ap.add_argument("--drone_radius", type=float, default=0.5)
    ap.add_argument("--max_speed", type=float, default=3.0)
    ap.add_argument("--cautious", action="store_true",
                     help="з --autopilot: невідомі пікселі вважати можливою перешкодою, а не "
                          "порожнечею. Обережніше, але повільніше")
    ap.add_argument("--save_to", default=None,
                     help="куди писати карту, наприклад ../live_map. Без цього не пише")
    args = ap.parse_args()

    compute = fake_vslam_and_fde if args.fake else my_vslam_and_fde
    if args.fake:
        print("УВАГА: --fake, працюють заглушки, а не твої VSLAM і FDE\n")

    if args.tcp:
        rx = TcpFrameServer("0.0.0.0", args.port).start()
        print(f"TCP, чекаю з'єднання на порту {args.port}")
        print("відповідь піде тим самим з'єднанням\n")
    else:
        rx = FrameReceiver("0.0.0.0", args.port)
        print(f"UDP, чекаю кадри на порту {args.port}")
        print(f"карту віддаю на порт {args.reply_port} тому, хто під'єднався\n")

    mapper = None
    pub = None
    pilot = (ReactiveAutopilot(drone_radius=args.drone_radius, max_speed=args.max_speed,
                                unknown_is_risk=args.cautious)
             if args.autopilot else None)
    if pilot:
        print(f"автопілот увімкнений, радіус апарата {args.drone_radius} м, "
              f"до {args.max_speed} м/с, невідоме = "
              f"{'ризик' if args.cautious else 'порожньо'}\n")
    n = 0
    t0 = time.monotonic()

    try:
        for frame in rx.frames():
            image = frame.get("image")
            depth = frame.get("depth")
            T_wc = frame.get("T_wc")
            K = frame.get("K")
            landmarks = frame.get("landmarks")

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

            if pub is not None:
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
