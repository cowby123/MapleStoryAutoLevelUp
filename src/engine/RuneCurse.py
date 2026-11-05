'''
Rune curse (Artale / MapleStory Worlds): when a rune has been up for a while
the game shows a purple banner near the top centre:
    「必須解放符文才能解除詛咒!!  詛咒：經驗值獲取、掉落率減少100%效果生效中」
and EXP / drops are 0 until someone solves the rune. The rune itself is often
hidden behind foreground grass and the minimap has no rune marker; the only
cue is a yellow arrow the game draws next to the character, pointing at it.

This module (pure image work, no input):
- CurseDetector: the banner's first text line, matched on its purple text
  (the banner is translucent, the text colour is not); template
  rune/curse_banner_cn.png, cut from a 1366x768 frame (scale 1366/1296) and
  rescaled to the current ui scale.
- find_guide_arrow(): the yellow arrow near the player and its direction.
- FrameCapture: saves frames while the banner is up (option, for fixtures).

Measured 2026-11-03 on maple (/work/frames_rune, 7 frames with the banner;
91 fixture frames without): banner score 0.00-0.03 with, >= 0.6 without;
arrow area 435-606 px, solidity 0.76-0.80, direction -130..-135 deg (down-left,
as seen); yellow blobs on the fixtures (flowers, damage numbers): area <= 181.
'''
# Standard import
import json
import math
import os
import time

# Library import
import cv2
import numpy as np

# Local import
from src.utils.logger import logger

CAPTURE_SCALE = 1366 / 1296  # ui scale of the frame the banner template was cut from


def purple_text(img_bgr):
    '''the banner's purple text as a soft 0..1 map'''
    b, g, r = [c.astype(np.int16) for c in cv2.split(img_bgr)]
    m = ((b >= 150) & (r >= 110) & (g <= 140) & (b - g >= 60)).astype(np.float32)
    return cv2.GaussianBlur(m, (3, 3), 0)


class CurseDetector:
    def __init__(self, path="rune/curse_banner_cn.png", thres=0.3):
        self.path = path
        self.thres = thres
        self.src = cv2.imread(path, cv2.IMREAD_COLOR) if os.path.exists(path) else None
        self.tpl = None
        self.scale = None
        self.last_score = None
        self.last_loc = None
        if self.src is None:
            logger.error(f"[RuneCurse] banner template missing: {path}")

    def set_scale(self, s):
        if self.src is None:
            return
        f = s / CAPTURE_SCALE
        img = self.src if abs(f - 1) < 1e-3 else cv2.resize(
            self.src, (max(1, round(self.src.shape[1] * f)), max(1, round(self.src.shape[0] * f))),
            interpolation=cv2.INTER_LINEAR)
        self.tpl = purple_text(img)
        self.scale = s

    def detect(self, frame, s):
        '''
        True when the curse banner is on screen. Searches the top 45 % of the
        frame (the banner sits near the top centre; position not assumed).
        '''
        if self.src is None:
            return False
        if self.scale != s:
            self.set_scale(s)
        H, W = frame.shape[:2]
        roi = frame[:int(H * 0.45)]
        th, tw = self.tpl.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return False
        feat = purple_text(roi)
        if feat.max() == 0:          # no purple text at all: cheap exit
            self.last_score = 1.0
            return False
        r = cv2.matchTemplate(feat, self.tpl, cv2.TM_SQDIFF_NORMED)
        v, _, loc, _ = cv2.minMaxLoc(r)
        self.last_score, self.last_loc = float(v), loc
        return v <= self.thres


def find_guide_arrow(frame, player, s, ui_y=None):
    '''
    The yellow arrow near the player. Returns None or a dict
      centre (x, y), angle (degrees; 0 = right, 90 = up, -90 = down),
      dir (unit vector, image coords), area, solidity
    '''
    if player is None:
        return None
    H, W = frame.shape[:2]
    ui_y = ui_y or H
    px, py = int(player[0]), int(player[1])
    x0, x1 = max(0, px - int(300 * s)), min(W, px + int(300 * s))
    y0, y1 = max(0, py - int(220 * s)), min(ui_y, py + int(220 * s))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (22, 235, 200), (34, 255, 255))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    k = (s / CAPTURE_SCALE) ** 2
    best = None
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (300 * k <= area <= 1500 * k) or w < 20 * s / CAPTURE_SCALE or h < 18 * s / CAPTURE_SCALE:
            continue
        comp = (lab == i).astype(np.uint8)
        cnt = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        sol = area / max(1.0, cv2.contourArea(cv2.convexHull(cnt)))
        if not 0.6 <= sol <= 0.9:          # an arrow is concave (head + tail)
            continue
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        c = pts.mean(0)
        cov = np.cov((pts - c).T)
        _, vec = np.linalg.eigh(cov)
        ax = vec[:, 1]
        proj = (pts - c) @ ax
        mid = (proj.max() + proj.min()) / 2
        d = ax * (1 if mid < 0 else -1)    # the heavier head is on the centroid's side
        cx, cy = float(c[0] + x0), float(c[1] + y0)
        # the guide arrow is drawn beside / below the character (2026-11-03: 90 px
        # left, 82 below the player position; 2026-11-04: 70-110 px aside, 23-27
        # below); yellow blocks above it and the character's own bow are not it
        if cy < py - 5 * s or abs(cx - px) < 45 * s / CAPTURE_SCALE:
            continue
        # the arrow is drawn on the rune's side of the character: agree with that
        to_arrow = np.array([cx - px, cy - py])
        if np.linalg.norm(to_arrow) > 5 and float(np.dot(d, to_arrow)) < 0:
            d = -d
        dist = abs(cx - px) + abs(cy - py)
        if best is None or dist < best[0]:
            best = (dist, {"centre": (int(cx), int(cy)), "dir": (float(d[0]), float(d[1])),
                           "angle": round(math.degrees(math.atan2(-d[1], d[0])), 1),
                           "area": int(area), "solidity": round(float(sol), 2)})
    return best[1] if best else None


class FrameCapture:
    '''
    Saves frames while the curse banner (or the mini-game) is on screen, for
    fixtures: at most one every `interval` s and `max_files` frames in `out_dir`.
    The folder is re-counted every RECOUNT saves and (at most every 30 s) when
    the cap is reached, so moving files away resumes capturing without a
    restart. rotate=True: at the cap the oldest frames (+ their .json) are
    deleted instead of stopping.
    '''
    RECOUNT = 50

    def __init__(self, out_dir, interval=2.0, max_files=300, rotate=False):
        self.out_dir = out_dir
        self.interval = interval
        self.max_files = max_files
        self.rotate = rotate
        self.t_last = 0.0
        self.count = None
        self.saves = 0
        self.t_recount = 0.0

    def pngs(self):
        return sorted(f for f in os.listdir(self.out_dir) if f.endswith(".png"))

    def recount(self, now):
        self.count = len(self.pngs())
        self.t_recount = now

    def drop_oldest(self, n):
        files = sorted((os.path.getmtime(os.path.join(self.out_dir, f)), f) for f in self.pngs())
        for _, f in files[:max(0, n)]:
            for p in (f, f[:-4] + ".json"):
                try:
                    os.remove(os.path.join(self.out_dir, p))
                except OSError:
                    pass
        self.recount(time.time())

    def maybe_save(self, frame, tag, meta=None):
        now = time.time()
        if now - self.t_last < self.interval:
            return None
        self.t_last = now
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            if self.count is None or self.saves % self.RECOUNT == 0:
                self.recount(now)
            if self.count >= self.max_files:
                if self.rotate:
                    self.drop_oldest(self.count - self.max_files + 1)
                elif now - self.t_recount > 30:
                    self.recount(now)    # files moved away / cap raised?
                if self.count >= self.max_files:
                    return None
            name = time.strftime(f"{tag}_%Y%m%d-%H%M%S") + f"_{int(now * 1000) % 1000:03d}"
            path = os.path.join(self.out_dir, name + ".png")
            cv2.imwrite(path, frame)
            if meta:
                with open(os.path.join(self.out_dir, name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, default=str)
            self.count += 1
            self.saves += 1
            return path
        except OSError as e:
            logger.warning(f"[RuneCurse] capture failed: {e}")
            return None
