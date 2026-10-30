r"""
Живе вікно на три панелі: кадр камери, карта довіри цього кадру і
накопичена карта зайнятості згори.

Те саме tkinter + PIL, що й у live_view.py, бо саме ця зв'язка перевірена у
WSLg. Матплотліба в циклі немає навмисно: кольори рахуються таблицею
пошуку по numpy, інакше на кожному кадрі довелось би малювати фігуру і
вікно відставало б від потоку.

    ┌──────────────┬──────────────┬──────────────┐
    │  кадр        │ карта довіри │ зайнятість   │
    │  камери      │ цього кадру  │ згори, з усіх│
    │              │              │ кадрів       │
    └──────────────┴──────────────┴──────────────┘

Середня панель це ЙМОВІРНІСТЬ, яку вносить один кадр: P = сігмоїда(l_occ·w).
Вона за побудовою не може бути вищою за сігмоїду(l_occ), тобто 0,70 при
l_occ = 0,85. Шкала саме така, від 0,50 до 0,70, інакше все виглядало б
однаково зеленим. Білим позначені пікселі, відкинуті з карти.

Права панель це максимум логарифма шансів уздовж вертикальної осі світу, тобто
«чи є хоч щось зайняте в цій колонці». Сірий це невідомо, синій вільно,
червоний зайнято. Біла лінія це пройдений шлях, біле кружальце це поточне
положення.

Не блокує: клич show() у своєму циклі, а не mainloop(). Якщо вікно
закрили, .closed стає True, цикл має це перевірити сам.

    view = LiveMapView()
    view.show(image_rgb, confidence, grid, position)
    if view.closed:
        break
    view.close()
"""
import numpy as np
import tkinter as tk

from PIL import Image, ImageTk

L_OCC = 0.85


def _ramp(stops):
    """Таблиця на 256 кольорів з кількох опорних. stops це [(позиція 0..1,
    (r, g, b)), ...] у порядку зростання позиції."""
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    cs = np.array([s[1] for s in stops], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    out = np.empty((256, 3), dtype=np.uint8)
    for c in range(3):
        out[:, c] = np.interp(t, xs, cs[:, c]).astype(np.uint8)
    return out


# зелений -> жовтий -> червоний: той самий напрямок, що й RdYlGn_r у
# статичних рисунках, щоб презентація і живе вікно не суперечили одне одному
_CONF_LUT = _ramp([(0.0, (26, 150, 65)), (0.5, (255, 255, 191)), (1.0, (215, 25, 28))])
# вільно (синій) -> невідомо (сірий) -> зайнято (червоний)
_OCC_LUT = _ramp([(0.0, (40, 90, 170)), (0.5, (205, 205, 205)), (1.0, (200, 30, 30))])


# Вбудований растровий шрифт PIL не має кирилиці, і підписи виходять
# квадратиками. Беремо перший TTF, який знайдеться, і лише якщо жодного
# немає -- падаємо назад на вбудований.
_FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
_font_cache = {}


def _font(size=13):
    if size not in _font_cache:
        from PIL import ImageFont
        f = None
        for path in _FONT_PATHS:
            try:
                f = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        _font_cache[size] = f if f is not None else ImageFont.load_default()
    return _font_cache[size]


def _panel(label_rgb, title, font_h=20):
    """Підпис над панеллю, щоб не плутати, де що."""
    from PIL import ImageDraw
    h, w = label_rgb.shape[:2]
    out = np.full((h + font_h, w, 3), 24, dtype=np.uint8)
    out[font_h:] = label_rgb
    img = Image.fromarray(out)
    ImageDraw.Draw(img).text((5, 3), title, fill=(235, 235, 235), font=_font(13))
    return np.asarray(img)


def confidence_panel(conf, l_occ=L_OCC):
    """Карта довіри -> RGB. NaN (піксель відкинуто) білим."""
    conf = np.asarray(conf, dtype=np.float64)
    known = np.isfinite(conf)
    p = 1.0 / (1.0 + np.exp(-l_occ * np.where(known, conf, 0.0)))
    p_max = 1.0 / (1.0 + np.exp(-l_occ))
    t = np.clip((p - 0.5) / max(p_max - 0.5, 1e-9), 0.0, 1.0)
    rgb = _CONF_LUT[(t * 255).astype(np.uint8)]
    rgb[~known] = 255
    return rgb


def occupancy_panel(grid, trail=None, up_axis=2, size=360):
    """Накопичена сітка -> вигляд згори. Максимум логарифма шансів уздовж
    вертикальної осі: колонка червона, якщо в ній є хоч один зайнятий
    воксель, і синя, якщо все спостерігалось вільним."""
    lo = grid.log_odds
    top = lo.max(axis=up_axis)                  # (вісь0, вісь1) після проєкції
    lim = float(max(abs(grid.l_min), abs(grid.l_max), 1e-6))
    t = np.clip(top / lim, -1.0, 1.0) * 0.5 + 0.5
    rgb = _OCC_LUT[(t * 255).astype(np.uint8)]

    # пройдений шлях поверх карти
    if trail is not None and len(trail):
        keep = [a for a in range(3) if a != up_axis]
        idx = np.floor((np.asarray(trail) - grid.origin) / grid.resolution).astype(int)
        ok = np.all((idx >= 0) & (idx < grid.dims), axis=1)
        idx = idx[ok]
        if len(idx):
            rgb[idx[:, keep[0]], idx[:, keep[1]]] = (255, 255, 255)
            a, b = idx[-1, keep[0]], idx[-1, keep[1]]
            r = 3
            a0, a1 = max(a - r, 0), min(a + r + 1, rgb.shape[0])
            b0, b1 = max(b - r, 0), min(b + r + 1, rgb.shape[1])
            rgb[a0:a1, b0:b1] = (255, 255, 255)

    # перша вісь угору на екрані, щоб картинка не була дзеркальною
    img = Image.fromarray(np.transpose(rgb, (1, 0, 2))[::-1])
    k = max(1, int(np.ceil(max(img.size) / size)))
    if k > 1:
        img = img.resize((img.width // k, img.height // k), Image.NEAREST)
    elif max(img.size) < size // 2:
        m = size // max(img.size)
        img = img.resize((img.width * m, img.height * m), Image.NEAREST)
    return np.asarray(img)


class LiveMapView:
    def __init__(self, title="карта довіри і зайнятості", panel_w=360, max_trail=4000):
        self.panel_w = panel_w
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root, bg="#181818")
        self.label.pack()
        self.status = tk.Label(self.root, text="", anchor="w")
        self.status.pack(fill="x")
        self._photo = None
        self._closed = False
        self._trail = []
        self._max_trail = max_trail
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed

    def _fit(self, rgb):
        img = Image.fromarray(np.ascontiguousarray(rgb))
        if img.width != self.panel_w:
            img = img.resize((self.panel_w,
                              max(1, int(img.height * self.panel_w / img.width))),
                             Image.BILINEAR)
        return np.asarray(img)

    def show(self, image_rgb=None, confidence=None, grid=None, position=None,
             status_text="", up_axis=2):
        if self._closed:
            return
        if position is not None:
            self._trail.append(np.asarray(position, dtype=np.float64))
            if len(self._trail) > self._max_trail:
                del self._trail[:len(self._trail) - self._max_trail]

        panels = []
        if image_rgb is not None:
            a = np.asarray(image_rgb)
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            panels.append(_panel(self._fit(a.astype(np.uint8)), "кадр камери"))
        if confidence is not None:
            panels.append(_panel(self._fit(confidence_panel(confidence)),
                                 "карта довіри кадру, 0.50 ... 0.70"))
        else:
            blank = np.full((self.panel_w * 3 // 4, self.panel_w, 3), 48, np.uint8)
            panels.append(_panel(blank, "карти довіри ще немає"))
        if grid is not None:
            panels.append(_panel(self._fit(occupancy_panel(grid, self._trail, up_axis)),
                                 "зайнятість згори, усі кадри"))

        h = max(p.shape[0] for p in panels)
        canvas = np.full((h, sum(p.shape[1] for p in panels) + 8 * (len(panels) - 1), 3),
                         24, dtype=np.uint8)
        x = 0
        for p in panels:
            canvas[:p.shape[0], x:x + p.shape[1]] = p
            x += p.shape[1] + 8

        self._photo = ImageTk.PhotoImage(Image.fromarray(canvas))
        self.label.configure(image=self._photo)
        self.status.configure(text=status_text)
        try:
            self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self._closed = True
