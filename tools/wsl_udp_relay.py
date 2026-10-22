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

Зупинка: Ctrl+C.
"""
import argparse
import socket
import sys
import threading
import time


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
    args = ap.parse_args()

    client = [None]        # адреса того, хто шле кадри, з першого пакета
    n_in = [0, 0]
    n_out = [0, 0]

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

    try:
        last = (0, 0)
        while True:
            time.sleep(2.0)
            if not args.quiet and (n_in[0], n_out[0]) != last:
                print(f"  туди {n_in[0]} пакетів / {n_in[1]/1e6:.1f} МБ   "
                      f"назад {n_out[0]} пакетів / {n_out[1]/1e6:.1f} МБ")
                last = (n_in[0], n_out[0])
    except KeyboardInterrupt:
        print(f"\nзупинено. Туди {n_in[0]} пакетів, назад {n_out[0]}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
