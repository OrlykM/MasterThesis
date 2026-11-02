r"""
Живий DPVO -- та сама мережа й той самий клас `dpvo.dpvo.DPVO`, що й
demo.py/entrypoint.sh, але замість "читай цілу теку наперед" приймає кадри
по одному через TCP і одразу віддає позу. Призначено для запуску ВСЕРЕДИНІ
контейнера thesis/dpvo (потрібні lietorch, скомпільовані CUDA-розширення,
dpvo.pth -- їх немає і не повинно бути в звичайному .venv).

Препроцесинг кадру -- зумисне один в один з dpvo/stream.py:image_stream(),
бо мережа тренувалась/перевірялась саме на такому: BGR (не RGB), обрізка до
кратності 16. Відхилення від цього тихо псує якість трекінгу.

Запуск (з host, не зсередини):
    docker run --rm --gpus all -p 8600:8600 \
        -v "$(pwd)/live_dpvo_server.py:/DPVO/live_dpvo_server.py" \
        --entrypoint /opt/conda/envs/dpvo/bin/python \
        thesis/dpvo:latest /DPVO/live_dpvo_server.py

Протокол -- 4-байтна довжина (big-endian) + pickle, в обидва боки. Це НЕ
той самий протокол, що в udp_streaming.py: тут обидва кінці -- наш власний
код на localhost (Docker bridge), тож простий TCP+pickle без накладних
витрат на сумісність доречніший, ніж повний OVM1-протокол.

Кадр, що приймається:
    {"frame_id": int, "image": (H,W,3) uint8 RGB, "K": (3,3) float64}

Відповідь:
    {"frame_id": int, "initialized": bool,
     "T_wc": (4,4) float64 або None,       -- None, поки DPVO "розганяється"
     "landmarks": (N,3) float32 або None}  -- розріджені точки DPVO, світові координати

Перші ~8 кадрів DPVO ще калібрується (init_scale у detect.py), "T_wc" і
"landmarks" будуть None -- це очікувано, не помилка. my_vslam_and_fde() на
боці server_simple.py вже вміє пропускати такі кадри.
"""
import pickle
import socket
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/DPVO")
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.lietorch import SE3

NETWORK = "/DPVO/dpvo.pth"
CONFIG = "/DPVO/config/default.yaml"


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("з'єднання закрито під час читання")
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    (length,) = struct.unpack("!I", recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, length))


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!I", len(data)) + data)


def preprocess(image_rgb, K):
    """RGB (H,W,3) uint8 + K -> (BGR-тензор на GPU, intrinsics-тензор на GPU),
    той самий кроп до кратності 16, що й dpvo/stream.py:image_stream()."""
    bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    h, w, _ = bgr.shape
    bgr = bgr[:h - h % 16, :w - w % 16]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    image_t = torch.from_numpy(bgr.copy()).permute(2, 0, 1).cuda()
    intrinsics_t = torch.from_numpy(intrinsics).cuda()
    return image_t, intrinsics_t


def current_pose_and_landmarks(slam):
    """Поза НАЙОСТАННІШОГО обробленого кадру -- пряме читання з pg.poses_,
    не через slam.terminate() (той ще й гонить глобальну BA/loop closure,
    для одного живого кадру це і дорого, і семантично невірно).

    .inv() -- дзеркало того, що demo.py робить у terminate() перед
    поверненням траєкторії: внутрішнє представлення DPVO інвертоване
    відносно звичного T_wc (камера -> світ), яке чекає наш конвеєр."""
    pose = SE3(slam.pg.poses_[slam.n - 1]).inv()
    T_wc = pose.matrix().detach().cpu().numpy().reshape(4, 4).astype(np.float64)

    # pg.points_ це БУФЕР torch.zeros, який заповнюється лише в update(),
    # тобто вже після ініціалізації. До того slam.m уже рахує латки, і
    # points_[:m] віддає чисті нулі. Якщо їх віддати як опорні точки, на
    # приймачі вони виглядають як хмара в початку координат: усі позаду
    # камери, глибина нуль, жодна не доходить до кадру. Тому поки DPVO не
    # ініціалізувався, чесніше віддати None, ніж нулі.
    if not slam.is_initialized:
        return T_wc, None
    landmarks = slam.pg.points_.detach().cpu().numpy()[:slam.m].astype(np.float32)
    # підстраховка: окремі латки можуть лишатись незаповненими
    nz = np.abs(landmarks).sum(axis=1) > 0
    landmarks = landmarks[nz].astype(np.float32)
    return T_wc, (landmarks if len(landmarks) else None)


def main():
    host, port = "0.0.0.0", 8600
    cfg.merge_from_file(CONFIG)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"живий DPVO слухає на {host}:{port}", flush=True)

    slam = None
    frame_idx = 0

    while True:
        conn, addr = srv.accept()
        # Нове з'єднання означає новий політ, тому стан скидаємо. Інакше
        # DPVO продовжував би стару карту: камера "телепортується" з кінця
        # минулого прогону на початок нового, старі точки лишаються у світі
        # далеко позаду, і жодна з них не потрапляє в кадр.
        if slam is not None:
            print(f"новий клієнт, скидаю стан DPVO (було {slam.n} кадрів, "
                  f"{slam.m} точок)", flush=True)
            slam = None
            frame_idx = 0
        print(f"з'єднання від {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                image_t, intrinsics_t = preprocess(msg["image"], msg["K"])

                if slam is None:
                    _, hh, ww = image_t.shape
                    slam = DPVO(cfg, NETWORK, ht=hh, wd=ww, viz=False)
                    print(f"DPVO ініціалізовано на {ww}x{hh}", flush=True)

                with torch.no_grad():
                    slam(frame_idx, image_t, intrinsics_t)
                frame_idx += 1

                reply = {"frame_id": msg["frame_id"], "initialized": bool(slam.is_initialized),
                          "n": int(slam.n)}
                # Позу віддаємо З ПЕРШОГО кадру, а не чекаємо is_initialized.
                # DPVO тримає оцінку пози у pg.poses_ від самого початку (перші
                # кадри -- від моделі сталої швидкості), а is_initialized у нього
                # перемикається лише коли набереться достатньо паралакса. Чекати
                # на цей прапорець означало б не віддавати НІЧОГО, якщо камера
                # рухається мало -- а для шару безпеки груба поза краща за жодної.
                if slam.n > 0:
                    T_wc, landmarks = current_pose_and_landmarks(slam)
                    reply["T_wc"], reply["landmarks"] = T_wc, landmarks
                else:
                    reply["T_wc"], reply["landmarks"] = None, None

                if slam.n <= 3 or slam.n % 25 == 0:
                    print(f"кадр n={slam.n} initialized={slam.is_initialized} "
                          f"точок={slam.m}", flush=True)

                send_msg(conn, reply)
        except (ConnectionError, OSError) as e:
            print(f"з'єднання {addr} закрито: {e}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
