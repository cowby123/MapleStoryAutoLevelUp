'''
Nametag player detection (wingpu).

The player's nametag (dark box with white text under the character) is found
by grayscale template matching (TM_SQDIFF_NORMED):
- local search in a box around the last position while it was seen in the last
  `local_timeout` seconds (cheap),
- otherwise a global search over the whole camera area (first at half size,
  then refined at full size around the best spot).

The template is captured from a live frame (web UI "截取名牌") and saved
outside the code directory, with a sidecar JSON:
    {"scale": <ui scale of the frame it was cut from>,
     "offset": [dx, dy],   # player centre = template top-left + offset (template pixels)
     "frame_wh": [w, h], "rect": [x, y, w, h], "time": "..."}
At another resolution the template and the offset are rescaled by
scale_now / scale_at_capture.
'''
# Standard import
import json
import os
import time

# Library import
import cv2
import numpy as np

# Local import
from src.utils.logger import logger


def default_dir():
    base = "/work" if os.path.isdir("/work") and os.access("/work", os.W_OK) else "."
    return os.path.join(base, "nametag")


def propose_rect(img, player, s):
    '''
    Find the nametag under the player centre `player` (x, y) in a BGR frame.
    The box behind the name is translucent (it takes the colours of what is
    behind it), so the cue is the name itself: white, unsaturated text 7-18 px
    high below the feet. Returns [x, y, w, h] (text box + a margin) or None.
    '''
    if player is None:
        return None
    H, W = img.shape[:2]
    cx, cy = int(player[0]), int(player[1])
    x0, x1 = max(0, cx - int(110 * s)), min(W, cx + int(110 * s))
    y0, y1 = max(0, cy + int(10 * s)), min(H, cy + int(90 * s))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    roi = img[y0:y1, x0:x1].astype(np.int16)
    mn, mx = roi.min(axis=2), roi.max(axis=2)
    white = ((mn > 190) & (mx - mn < 40)).astype(np.uint8) * 255
    raw = white.copy()
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(9 * s)), max(3, int(5 * s))))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    n, _, st, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (6 * s <= h <= 20 * s and 8 * s <= w <= 150 * s):
            continue
        # text, not a solid white thing (a shoe, a cloud): strokes fill 20-60 %
        # of the box, and the box has dark pixels between the strokes
        fill = np.count_nonzero(raw[y:y + h, x:x + w]) / float(w * h)
        if not 0.15 <= fill <= 0.65:
            continue
        if np.count_nonzero(mx[y:y + h, x:x + w] < 90) < 0.15 * w * h:
            continue
        d = abs((x + w / 2) - (cx - x0)) + 0.5 * y
        if best is None or d < best[0]:
            best = (d, (x, y, w, h))
    if best is None:
        return None
    x, y, w, h = best[1]
    mx_, my_ = max(2, int(4 * s)), max(2, int(3 * s))
    X, Y = max(0, x + x0 - mx_), max(0, y + y0 - my_)
    return [int(X), int(Y), int(min(W - X, w + 2 * mx_)), int(min(H - Y, h + 2 * my_))]


def text_feature(img_bgr):
    '''
    The name's white, unsaturated text as a soft 0..1 map. The box behind the
    name is translucent (it shows what is behind it), so matching grey levels
    of the whole box depends on the background; the text does not.
    '''
    b, g, r = cv2.split(img_bgr)
    mn = cv2.min(cv2.min(b, g), r)
    mx = cv2.max(cv2.max(b, g), r)
    _, bright = cv2.threshold(mn, 170, 1, cv2.THRESH_BINARY)
    _, grey = cv2.threshold(cv2.subtract(mx, mn), 49, 1, cv2.THRESH_BINARY_INV)
    white = cv2.bitwise_and(bright, grey).astype(np.float32)
    return cv2.GaussianBlur(white, (3, 3), 0)


class NametagTracker:
    def __init__(self, path, thres=0.12, local_timeout=1.0):
        self.path = path
        self.thres = thres
        self.local_timeout = local_timeout
        self.tpl_src = None     # BGR template as captured
        self.meta = {}
        self.tpl = None         # at the current scale
        self.offset = (0, 0)
        self.scale = 1.0
        self.loc = None         # last top-left in frame coords
        self.t_seen = 0.0
        self.last_score = None
        self.last_mode = None   # "local" | "global"
        self.mtime = None
        self.load()

    @property
    def loaded(self):
        return self.tpl is not None

    def load(self):
        self.tpl_src = None
        self.tpl = None
        if not self.path or not os.path.exists(self.path):
            return False
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"[nametag] cannot read {self.path}")
            return False
        meta = {}
        side = os.path.splitext(self.path)[0] + ".json"
        if os.path.exists(side):
            try:
                with open(side, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError) as e:
                logger.error(f"[nametag] bad {side}: {e}")
        self.tpl_src, self.meta = img, meta
        self.mtime = os.path.getmtime(self.path)
        self.set_scale(self.scale)
        logger.info(f"[nametag] template {self.path} {img.shape[1]}x{img.shape[0]}, meta {meta}")
        return True

    def reload_if_changed(self):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            if self.tpl is not None:
                self.tpl = self.tpl_src = None
            return
        if m != self.mtime:
            self.load()

    def set_scale(self, s):
        self.scale = s
        self.loc = None
        if self.tpl_src is None:
            return
        f = s / float(self.meta.get("scale", s) or s)
        tpl = self.tpl_src
        if abs(f - 1.0) > 1e-3:
            tpl = cv2.resize(tpl, (max(1, round(tpl.shape[1] * f)), max(1, round(tpl.shape[0] * f))),
                             interpolation=cv2.INTER_LINEAR)
        self.tpl = text_feature(tpl)
        off = self.meta.get("offset") or [self.tpl_src.shape[1] / 2, -35 * float(self.meta.get("scale", s))]
        self.offset = (int(round(off[0] * f)), int(round(off[1] * f)))

    def _match(self, img, tpl):
        if img.shape[0] < tpl.shape[0] or img.shape[1] < tpl.shape[1]:
            return None, 1.0
        r = cv2.matchTemplate(img, tpl, cv2.TM_SQDIFF_NORMED)
        v, _, loc, _ = cv2.minMaxLoc(r)
        return loc, float(v)

    def find(self, img):
        '''
        img: camera area (BGR). Returns player centre (x, y) or None.
        '''
        if self.tpl is None:
            return None
        th, tw = self.tpl.shape[:2]
        H, W = img.shape[:2]
        now = time.time()
        if self.loc is not None and now - self.t_seen < self.local_timeout:
            lx, ly = self.loc
            rx, ry = max(40, 3 * tw // 2), max(30, 2 * th)
            x0, y0 = max(0, lx - rx), max(0, ly - ry)
            x1, y1 = min(W, lx + tw + rx), min(H, ly + th + ry)
            loc, v = self._match(text_feature(img[y0:y1, x0:x1]), self.tpl)
            if loc is not None and v <= self.thres:
                return self._hit((loc[0] + x0, loc[1] + y0), v, "local", now)
        # global: half size first, then full size around the best spot
        feat = text_feature(img)
        small = cv2.resize(feat, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        tpl_s = cv2.resize(self.tpl, (max(1, tw // 2), max(1, th // 2)), interpolation=cv2.INTER_AREA)
        loc, v = self._match(small, tpl_s)
        self.last_mode = "global"
        if loc is None:
            self.last_score = v
            return None
        gx, gy = loc[0] * 2, loc[1] * 2
        x0, y0 = max(0, gx - 6), max(0, gy - 6)
        x1, y1 = min(W, gx + tw + 6), min(H, gy + th + 6)
        loc2, v2 = self._match(feat[y0:y1, x0:x1], self.tpl)
        self.last_score = v2
        if loc2 is not None and v2 <= self.thres:
            return self._hit((loc2[0] + x0, loc2[1] + y0), v2, "global", now)
        return None

    def _hit(self, loc, v, mode, now):
        self.loc = (int(loc[0]), int(loc[1]))
        self.t_seen = now
        self.last_score = v
        self.last_mode = mode
        return (self.loc[0] + self.offset[0], self.loc[1] + self.offset[1])

    def status(self):
        return {
            "path": self.path, "loaded": self.loaded,
            "size": list(self.tpl.shape[1::-1]) if self.tpl is not None else None,
            "offset": list(self.offset), "loc": list(self.loc) if self.loc else None,
            "score": round(self.last_score, 3) if self.last_score is not None else None,
            "search": self.last_mode, "meta": self.meta,
        }


def save_template(img_bgr, rect, s, player, out_dir=None, name="nametag"):
    '''
    Cut rect [x, y, w, h] out of a frame and save it with its sidecar JSON.
    player = player centre in that frame (party bar) or None.
    '''
    out_dir = out_dir or default_dir()
    os.makedirs(out_dir, exist_ok=True)
    x, y, w, h = [int(v) for v in rect]
    H, W = img_bgr.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    if w < 8 or h < 6:
        raise ValueError("名牌框太小")
    crop = img_bgr[y:y + h, x:x + w]
    if player is not None:
        offset = [int(player[0]) - x, int(player[1]) - y]
    else:
        offset = [w // 2, -int(round(35 * s))]
    path = os.path.join(out_dir, name + ".png")
    tmp = path + ".tmp.png"
    cv2.imwrite(tmp, crop)
    os.replace(tmp, path)
    meta = {"scale": round(float(s), 5), "offset": offset, "frame_wh": [W, H], "rect": [x, y, w, h],
            "offset_from": "party_bar" if player is not None else "default",
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return path, meta
