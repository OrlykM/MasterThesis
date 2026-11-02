r"""
Клієнт до live_dpvo_server.py (слухає всередині контейнера thesis/dpvo,
--gpus all, порт прокинутий на host). Та сама роль для живої пози, яку
ZoeDepth відіграє для живої глибини -- один виклик на кадр, результат
одразу, без черги кадрів наперед.

Якщо сервіс не піднятий або впав -- ConnectionError/OSError пробивається
назовні як є, виклик my_vslam_and_fde() у server_simple.py сам вирішує, що
з цим робити (там за замовчуванням кадр просто пропускається).
"""
import pickle
import socket
import struct


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("живий DPVO закрив з'єднання")
        buf += chunk
    return bytes(buf)


def _recv_msg(sock):
    (length,) = struct.unpack("!I", _recv_exact(sock, 4))
    return pickle.loads(_recv_exact(sock, length))


def _send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!I", len(data)) + data)


class DpvoClient:
    def __init__(self, host="127.0.0.1", port=8600, connect_timeout=5.0,
                 read_timeout=8.0):
        """read_timeout -- скільки найдовше чекати відповіді на один кадр.

        Без нього сокет чекає вічно, і якщо DPVO спіткнеться (а він ділить
        відеокарту з моделлю глибини), увесь сервер завмирає назавжди:
        кадри далі приходять, але назад не йде нічого, і в лозі ані
        помилки, ані рядка. Саме так це й виглядало -- лічильник
        відповідей застигав, а лічильник кадрів ріс далі.

        Вісім секунд це свідомо багато: при ініціалізації DPVO ганяє
        дванадцять проходів оптимізації поспіль, і це законно довго."""
        self.host, self.port = host, port
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.sock = None
        self.last = {}

    def _ensure_connected(self):
        if self.sock is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.connect_timeout)
            s.connect((self.host, self.port))
            # socket.timeout успадковує OSError, тому track() нижче ловить
            # його разом з розривами і скидає сокет: наступний кадр
            # перепідключиться, а DPVO на новому з'єднанні скине стан
            s.settimeout(self.read_timeout)
            self.sock = s

    def track(self, frame_id, image_rgb, K):
        """Один кадр -> (T_wc або None, landmarks або None, initialized).
        None означає "DPVO ще розганяється" (перші ~8 кадрів), не помилку.

        Після виклику в .last лежить уся відповідь, разом з `n`, тобто
        скільки ключових кадрів DPVO уже набрав. Це єдиний спосіб ззовні
        побачити, чи він узагалі рухається до ініціалізації: вона настає
        рівно на восьмому ключовому кадрі, а кадр стає ключовим лише коли
        мережа бачить щонайменше два пікселі руху."""
        self._ensure_connected()
        try:
            _send_msg(self.sock, {"frame_id": frame_id, "image": image_rgb, "K": K})
            reply = _recv_msg(self.sock)
        except (ConnectionError, OSError):
            self.sock = None       # наступний виклик спробує перепідключитись
            raise
        self.last = reply
        return reply["T_wc"], reply["landmarks"], reply["initialized"]

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
