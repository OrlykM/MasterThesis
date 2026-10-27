r"""
Живе вікно для одного кадру, що оновлюється на місці -- працює через WSLg
(DISPLAY/X11, перевірено) на звичайному tkinter, без cv2 і без Open3D-стилю
проблем з GUI у WSL.

Навмисно лишається на PIL/Image від початку до кінця -- тобто (H, W, 3)
RGB, той самий порядок каналів, що віддає RawJpegFrameReceiver і
decode_frame2. cv2 за замовчуванням працює в BGR, і якщо десь переплутати
"RGB-масив з нашого протоколу" з "BGR, якого чекає cv2.imshow", то й
виходить той самий ефект "кольори поплили" -- синє й червоне мінялись
місцями. Тут цього джерела помилки просто немає.

Не блокує: виклич show() у своєму циклі, а не mainloop(). Якщо користувач
закрив вікно -- властивість .closed стає True, цикл має це перевірити сам.

Використання:
    view = LiveImageView("вхідні кадри")
    ...
    view.show(image_rgb, status_text=f"кадр {n}")
    if view.closed:
        break
    ...
    view.close()
"""
import tkinter as tk

from PIL import Image, ImageTk


class LiveImageView:
    def __init__(self, title="кадри", max_width=720):
        self.max_width = max_width
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status = tk.Label(self.root, text="", anchor="w")
        self.status.pack(fill="x")
        self._photo = None          # тримати посилання -- інакше Tk забуде картинку
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed

    def show(self, image_rgb, status_text=""):
        """image_rgb: (H, W, 3) або (H, W) uint8, RGB (не BGR)."""
        if self._closed:
            return
        img = Image.fromarray(image_rgb)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if img.width > self.max_width:
            scale = self.max_width / img.width
            img = img.resize((self.max_width, max(1, int(img.height * scale))))
        try:
            self._photo = ImageTk.PhotoImage(img)
            self.label.configure(image=self._photo)
            self.status.configure(text=status_text)
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self):
        if not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True
