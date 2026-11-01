'''
GameWindowCapturor for the wingpu platform.

The bot runs in the VM's script container; the game runs in the Windows guest.
The wingpu SDK gives the WHOLE guest screen (BGR). This capturor:

1. finds the game's client area ("MapleStory Worlds") in the guest screen:
   - fixed:      cfg wingpu.client_rect = [x, y, w, h] (guest pixels)
   - template:   cfg wingpu.title_template (an image of the window's title bar,
                 made by tools/wingpu_calibrate.py) is matched on the guest screen;
                 the client is at a fixed offset below it (wingpu.title_to_client)
   - fullscreen: no window found -> the whole guest screen is the client
   auto (default) tries template, then fullscreen; re-checked every
   wingpu.redetect_interval seconds.
2. crops the client 1:1: no resize, no title band. The bot works on the game's
   real pixels (exact colours) and scales its detectors to the frame size
   (src/utils/resolution.py, MapleStoryAutoBot.on_resolution);
3. maps client coordinates to guest pixels for clicks (window_to_guest = client
   origin + coordinate), and registers that mapping with WingpuKeyboard.
'''
# Standard import
import os
import time
import threading

# Library import
import cv2
import numpy as np

# Local import
from src.utils.logger import logger
from src.input.WingpuVM import get_vm
from src.input.WingpuKeyboard import get_keyboard


class ClientRect:
    '''
    Game client area in guest pixels. The bot frame is this area 1:1.
    '''
    def __init__(self, x, y, w, h, method):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.method = method

    def window_to_guest(self, x, y):
        '''
        Bot frame (client) coordinate -> guest pixel
        '''
        return int(round(self.x + x)), int(round(self.y + y))

    def as_dict(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "method": self.method}

    def __eq__(self, other):
        return isinstance(other, ClientRect) and \
            (self.x, self.y, self.w, self.h) == (other.x, other.y, other.w, other.h)

    def __repr__(self):
        return f"{self.w}x{self.h}+{self.x}+{self.y} ({self.method})"


def detect_client_rect(img, cfg, title_tpl=None):
    '''
    Find the game client area in a guest screen. Returns ClientRect.
    Pure function (no state), used by the capturor and tools/wingpu_check.py.
    '''
    wcfg = cfg.get("wingpu", {}) or {}
    gh, gw = img.shape[:2]
    mode = wcfg.get("detect", "auto")

    # (a) fixed rectangle
    rect = wcfg.get("client_rect")
    if mode == "fixed" or (rect and mode == "auto"):
        if not rect:
            raise ValueError("wingpu.detect is 'fixed' but wingpu.client_rect is empty")
        x, y, w, h = rect
        return ClientRect(x, y, w, h, "fixed")

    # (b) title bar template
    if title_tpl is not None and mode in ("auto", "template"):
        res = cv2.matchTemplate(img, title_tpl, cv2.TM_SQDIFF_NORMED)
        score, _, (tx, ty), _ = cv2.minMaxLoc(res)
        if score <= wcfg.get("title_thres", 0.05):
            dx, dy = wcfg.get("title_to_client", (0, title_tpl.shape[0]))
            cw, ch = wcfg.get("client_size_guest") or (gw - tx - dx, gh - ty - dy)
            x, y = tx + dx, ty + dy
            # clip to the guest screen (window partly off screen)
            w = min(cw, gw - x)
            h = min(ch, gh - y)
            if w > 0 and h > 0:
                return ClientRect(x, y, w, h, f"template({score:.3f})")
        if mode == "template":
            return None

    # (c) the game fills the whole guest screen
    return ClientRect(0, 0, gw, gh, "fullscreen")


class WingpuCapturor:
    '''
    GameWindowCapturor for the wingpu platform (same interface as
    src/input/GameWindowCapturor.py: get_frame(), stop(), fps, window_title).
    '''
    def __init__(self, cfg, test_image_name=None):
        self.cfg = cfg
        self.wcfg = cfg.get("wingpu", {}) or {}
        self.window_title = cfg["game_window"]["title"]
        self.fps_limit = cfg["system"]["fps_limit_window_capturor"]
        self.redetect_interval = self.wcfg.get("redetect_interval", 5.0)
        self.frame = None # bot frame: the game client 1:1, BGR
        self.guest = None # last full guest screen
        self.rect = None # ClientRect
        self.lock = threading.Lock()
        self.is_terminated = False
        self.fps = 0
        self.t_last_run = 0.0
        self.t_last_detect = 0.0
        self.n_frames = 0
        self.t_last_frame = 0.0
        self.capture_control = None

        # Title bar template (optional)
        self.title_tpl = None
        tpl_path = self.wcfg.get("title_template", "")
        if tpl_path and os.path.exists(tpl_path):
            self.title_tpl = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
            logger.info(f"[WingpuCapturor] Title bar template: {tpl_path}")
        elif tpl_path:
            logger.warning(f"[WingpuCapturor] Title bar template not found: {tpl_path}")

        get_keyboard().set_click_mapper(self.window_to_guest)

        # Test image: a saved GUEST screenshot instead of the live VM
        if test_image_name:
            path = test_image_name if os.path.exists(test_image_name) \
                   else f"test/{test_image_name}.png"
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"[WingpuCapturor] Cannot read test image {path}")
            self.process(img)
            logger.info(f"[WingpuCapturor] Test image {path}, client {self.rect}")
            return

        self.vm = get_vm(cfg)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

        # Wait for the first frame (the sender starts the tap on demand)
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 10:
            time.sleep(0.05)
        if self.frame is None:
            logger.warning("[WingpuCapturor] No guest frame yet (is the VM running and "
                           "is its stream sender sharing frames?)")
        logger.info("[WingpuCapturor] Init done")

    def run(self):
        '''
        Capture thread
        '''
        while not self.is_terminated:
            try:
                img = self.vm.frame(wait=1.0, new=True)
            except Exception as e:
                logger.error(f"[WingpuCapturor] frame: {e}")
                img = None
            if img is not None and not isinstance(img, tuple):
                self.process(img)
            self.limit_fps()

    def process(self, img):
        '''
        guest screen -> bot frame
        '''
        now = time.time()
        if self.rect is None or now - self.t_last_detect > self.redetect_interval or \
                img.shape[:2] != (self.guest.shape[:2] if self.guest is not None else None):
            rect = detect_client_rect(img, self.cfg, self.title_tpl)
            self.t_last_detect = now
            if rect is not None and rect != self.rect:
                logger.info(f"[WingpuCapturor] Game client in guest "
                            f"{img.shape[1]}x{img.shape[0]}: {rect}")
                self.rect = rect
            elif rect is None and self.rect is not None:
                logger.warning("[WingpuCapturor] Game window not found; keeping last rect")
        r = self.rect
        if r is None:
            return

        # 1:1 crop; a view when the client is the whole screen (no copy)
        if (r.x, r.y, r.w, r.h) == (0, 0, img.shape[1], img.shape[0]):
            frame = img
        else:
            frame = np.ascontiguousarray(img[r.y:r.y + r.h, r.x:r.x + r.w])

        with self.lock:
            self.guest = img
            self.frame = frame
            self.n_frames += 1
            self.t_last_frame = now

    def get_frame(self):
        '''
        Latest bot frame (BGR, the game client 1:1), or None
        '''
        with self.lock:
            return self.frame

    def get_guest_frame(self):
        '''
        Latest full guest screen (BGR), or None
        '''
        with self.lock:
            return self.guest

    def window_to_guest(self, x, y):
        '''
        Bot frame (client) coordinate -> guest pixel (None if no client rect yet)
        '''
        r = self.rect
        if r is None:
            return None
        return r.window_to_guest(x, y)

    def status(self):
        '''
        For the debug status file
        '''
        return {
            "client_rect": self.rect.as_dict() if self.rect else None,
            "guest_size": list(self.guest.shape[1::-1]) if self.guest is not None else None,
            "frames": self.n_frames,
            "capture_fps": self.fps,
            "frame_age_s": round(time.time() - self.t_last_frame, 3) if self.t_last_frame else None,
        }

    def stop(self):
        '''
        Stop capturing thread
        '''
        self.is_terminated = True
        logger.info("[WingpuCapturor] Terminated")

    def limit_fps(self):
        '''
        Limit FPS
        '''
        target_duration = 1.0 / self.fps_limit
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)
        dt = time.time() - self.t_last_run
        self.fps = round(1.0 / dt) if dt > 0 else 0
        self.t_last_run = time.time()


# Same name as the other platforms' capturors
GameWindowCapturor = WingpuCapturor
