'''
The rune arrow mini-game (Artale): a translucent panel in the middle of the
screen with a row of round icons, each a white ring (with a coloured glow)
and a white arrow inside. Pressing the right arrow key makes the icon fade
out, left to right.

read_minigame(frame, s) finds the icons in a band around the screen centre:
white rings of the expected size with a white arrow inside. The arrow's
direction is read from its SHAPE: the arrow mask is compared with the "up"
arrow cut from a real frame (rune/minigame_arrow_up.png) rotated to right /
down / left -- colours differ per direction (down red, right cyan, up yellow
on 2026-11-04) and left was never seen, so colour is not used. Faded
(solved) icons have no white ring any more and are not returned.

Measured on maple 2026-11-04 (8 frames, 1366x768): ring 73x72 px, arrow
~42x42 px, icon centres 417 / 593 / 773 / 948 at y 382, i.e. spacing ~177 px
around the screen centre.
'''
# Standard import
import os

# Library import
import cv2
import numpy as np

CAPTURE_SCALE = 1366 / 1296
DIRS = ("up", "right", "down", "left")      # np.rot90(k=-1) turns up into right
ARROW_CN = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
_UP = None


def _up_mask():
    global _UP
    if _UP is None:
        p = os.path.join("rune", "minigame_arrow_up.png")
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        _UP = (cv2.resize(m, (32, 32), interpolation=cv2.INTER_AREA) > 127) if m is not None else None
    return _UP


def white_mask(img_bgr):
    '''bright, unsaturated pixels (0/1), uint8 OpenCV ops (a few ms for the band)'''
    b, g, r = cv2.split(img_bgr)
    mn = cv2.min(cv2.min(b, g), r)
    mx = cv2.max(cv2.max(b, g), r)
    _, bright = cv2.threshold(mn, 189, 1, cv2.THRESH_BINARY)
    _, grey = cv2.threshold(cv2.subtract(mx, mn), 45, 1, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(bright, grey)


def classify_arrow(mask):
    '''
    mask: binary arrow (any size). Returns (direction, score 0..1, all scores)
    by IoU with the reference up arrow rotated to the four directions
    '''
    up = _up_mask()
    if up is None or mask.size == 0:
        return None, 0.0, {}
    m = cv2.resize(mask.astype(np.uint8) * 255, (32, 32), interpolation=cv2.INTER_AREA) > 127
    scores = {}
    for k, d in enumerate(DIRS):
        ref = np.rot90(up, k=-k)
        inter = np.logical_and(m, ref).sum()
        union = np.logical_or(m, ref).sum()
        scores[d] = inter / union if union else 0.0
    best = max(scores, key=scores.get)
    return best, round(float(scores[best]), 3), {k: round(float(v), 3) for k, v in scores.items()}


def read_minigame(frame, s, min_score=0.55):
    '''
    Active (not yet solved) icons of the mini-game, left to right:
    [{"x", "y", "dir", "score"}], or [] when there is no mini-game.
    '''
    H, W = frame.shape[:2]
    f = s / CAPTURE_SCALE
    hw, hh = int(480 * f), int(130 * f)
    x0, y0 = max(0, W // 2 - hw), max(0, H // 2 - hh)
    band = frame[y0:min(H, H // 2 + hh), x0:min(W, W // 2 + hw)]
    wm = white_mask(band)
    n, lab, st, _ = cv2.connectedComponentsWithStats(wm, connectivity=8)
    ring_lo, ring_hi = 58 * f, 92 * f
    icons = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (ring_lo <= w <= ring_hi and ring_lo <= h <= ring_hi and abs(w - h) <= 10 * f):
            continue
        # a ring: thin (little area for its box), hollow centre
        if area > 0.45 * w * h:
            continue
        cx, cy = x + w // 2, y + h // 2
        # the arrow: white pixels inside the ring, not part of the ring component
        r = int(0.36 * w)
        inner = wm[cy - r:cy + r, cx - r:cx + r].copy()
        inner[lab[cy - r:cy + r, cx - r:cx + r] == i] = 0
        if inner.size == 0 or inner.mean() < 0.15:
            continue
        ys, xs = np.nonzero(inner)
        arrow = inner[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        d, score, _ = classify_arrow(arrow)
        if d is None or score < min_score:
            continue
        icons.append({"x": int(cx + x0), "y": int(cy + y0), "dir": d, "score": score})
    icons.sort(key=lambda c: c["x"])
    # one row: drop icons far from the row's median height
    if icons:
        my = float(np.median([c["y"] for c in icons]))
        icons = [c for c in icons if abs(c["y"] - my) <= 20 * f]
    return icons


def sequence_text(icons, total=None):
    '''「↓ → ↑ →」, with the solved ones counted when total is known'''
    s = " ".join(ARROW_CN[c["dir"]] for c in icons)
    return s
