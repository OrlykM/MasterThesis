r"""
Ретранслятор UDP для WSL2. Запускається НА WINDOWS, не всередині WSL.

Навіщо. WSL2 за замовчуванням сидить за NAT, тому порт, відкритий усередині
WSL, ззовні недосяжний. Звичайний `netsh interface portproxy` тут не поможе:
він уміє лише TCP. Цей скрипт переносить UDP в обидва боки.

    інший ноутбук           Windows (цей скрипт)            WSL
    192.168.0.x                192.168.0.122              172.29.x.x
         │                          │                          │
         │  кадри -> :9000          │   -> 172.29.x.x:9000     │
         ├─────────────────────────►├─────────────────────────►│
         │                          │                          │
         │  <- карта довіри :9001   │   <- :9001               │
         ◄─────────────────────────┤◄─────────────────────────┤

Адресу клієнта скрипт запам'ятовує з першого ж кадру, тому нічого вручну
прописувати не треба.

Запуск у PowerShell або cmd на Windows:

    python wsl_udp_relay.py --wsl-ip 172.29.13.209

Адресу WSL подивитись усередині WSL командою `hostname -I` (перша в списку).
Вона змінюється після перезавантаження.

Якщо Windows спитає про брандмауер, дозволити для приватних мереж.

За замовчуванням ще й відкриває вікно з картинкою з кожного вхідного пакета,
в якому знайшовся JPEG (шукає сигнатуру `FF D8 FF`, байти до неї, якщо є,
просто ігноруються -- це може бути ваш власний короткий заголовок з
номером кадру чи часовою міткою, на пересилання в WSL він не впливає).
Вимкнути вікно: `--no-show`. Вийти і з вікна, і зі скрипта: Esc або `q` у
вікні, або Ctrl+C у консолі.

Потрібен `opencv-python` на Windows-боці (`pip install opencv-python`) лише
для вікна -- без нього пересилання працює так само, просто без картинки.

Зупинка: Ctrl+C.
"""
import argparse
import socket
import sys
import threading
import time

try:
    import cv2
    import numpy as np
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

_JPEG_SOI = b"\xff\xd8\xff"


def human(n):
    """Байти так, щоб 240 пакетів по 88 байтів не виглядали як "0.0 МБ"
    і не сходили за збій. Голий вектор керування це саме 88 байтів, тому
    зворотний потік тут законно мізерний."""
    if n < 1024:
        return f"{n} Б"
    if n < 1024 * 1024:
        return f"{n/1024:.1f} КБ"
    return f"{n/1e6:.1f} МБ"


def extract_jpeg(data):
    """Пакет може нести свій заголовок перед картинкою -- шукаємо початок
    JPEG (SOI-маркер) і беремо все від нього, а не перші N байтів."""
    i = data.find(_JPEG_SOI)
    return None if i < 0 else data[i:]


def pump(name, sock_in, target_getter, counter, verbose):
    while True:
        try:
            data, addr = sock_in.recvfrom(65535)
        except OSError:
            return
        target = target_getter(addr)
        if target is None:
            continue
        try:
            sock_in.sendto(data, target)
        except OSError as e:
            if verbose:
                print(f"[{name}] не вдалось переслати на {target}: {e}")
            continue
        counter[0] += 1
        counter[1] += len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsl-ip", required=True,
                     help="адреса WSL, перша з `hostname -I` всередині WSL")
    ap.add_argument("--in-port", type=int, default=9000,
                     help="порт, на який інший ноутбук шле кадри")
    ap.add_argument("--out-port", type=int, default=9001,
                     help="порт, на який сервер у WSL віддає карту довіри")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-show", action="store_true",
                     help="не відкривати вікно з картинкою, лише пересилати")
    args = ap.parse_args()

    show = (not args.no_show) and _HAVE_CV2
    if not args.no_show and not _HAVE_CV2:
        print("УВАГА: opencv-python не знайдено, вікно з картинкою вимкнене "
              "(`pip install opencv-python`, щоб увімкнути). Пересилання це не чіпає.\n")

    client = [None]        # адреса того, хто шле кадри, з першого пакета
    n_in = [0, 0]
    n_out = [0, 0]
    n_jpeg = [0]
    latest_frame = [None]
    frame_lock = threading.Lock()

    s_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_in.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    s_in.bind(("0.0.0.0", args.in_port))

    s_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_out.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    s_out.bind(("0.0.0.0", args.out_port))

    print(f"Ретранслятор UDP для WSL")
    print(f"  кадри        0.0.0.0:{args.in_port}  ->  {args.wsl_ip}:{args.in_port}")
    print(f"  карта довіри 0.0.0.0:{args.out_port}  ->  назад тому, хто під'єднався")
    print(f"  Ctrl+C щоб зупинити\n")

    def forward_in():
        while True:
            try:
                data, addr = s_in.recvfrom(65535)
            except OSError:
                return
            if client[0] != addr[0]:
                client[0] = addr[0]
                print(f"[вхід] джерело кадрів: {addr[0]}")
            n_in[0] += 1
            n_in[1] += len(data)
            try:
                s_in.sendto(data, (args.wsl_ip, args.in_port))
            except OSError as e:
                print(f"[вхід] не вдалось переслати у WSL: {e}")

            if show:
                jpeg = extract_jpeg(data)
                if jpeg is not None:
                    try:
                        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        img = None
                    if img is not None:
                        n_jpeg[0] += 1
                        with frame_lock:
                            latest_frame[0] = img

    def forward_out():
        while True:
            try:
                data, _addr = s_out.recvfrom(65535)
            except OSError:
                return
            if client[0] is None:
                continue
            n_out[0] += 1
            n_out[1] += len(data)
            try:
                s_out.sendto(data, (client[0], args.out_port))
            except OSError as e:
                print(f"[вихід] не вдалось переслати клієнту: {e}")

    threading.Thread(target=forward_in, daemon=True).start()
    threading.Thread(target=forward_out, daemon=True).start()

    if show:
        cv2.namedWindow("вхідні кадри", cv2.WINDOW_NORMAL)
        print("вікно з картинкою: Esc або 'q' щоб закрити і вийти\n")

    try:
        last = (0, 0)
        last_print = 0.0
        while True:
            if show:
                with frame_lock:
                    img = latest_frame[0]
                if img is not None:
                    cv2.setWindowTitle("вхідні кадри", f"вхідні кадри -- #{n_jpeg[0]}")
                    cv2.imshow("вхідні кадри", img)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):      # Esc або q
                    print("\nзакрито з вікна")
                    break
            else:
                time.sleep(2.0)

            now = time.monotonic()
            if not args.quiet and now - last_print >= 2.0 and (n_in[0], n_out[0]) != last:
                jpeg_part = f", картинок {n_jpeg[0]}" if show else ""
                rate = f"  1 вектор на {n_in[0] // n_out[0]} кадрів" if n_out[0] else ""
                print(f"  туди {n_in[0]} пакетів / {human(n_in[1])}   "
                      f"назад {n_out[0]} пакетів / {human(n_out[1])}{jpeg_part}{rate}")
                last, last_print = (n_in[0], n_out[0]), now
    except KeyboardInterrupt:
        print(f"\nзупинено. Туди {n_in[0]} пакетів, назад {n_out[0]}")
    finally:
        if show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
