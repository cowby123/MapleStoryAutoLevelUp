'''
Which character is playing: the character's name in the HUD's bottom-left
box (「法師 / 厭世腿腿」, 「弓箭手 / 腿腿」). A grey snippet of the name row is
saved per profile; the live name row is compared with every saved snippet.
The job line and the level are not used (they change with the character).

Reference 1366x768 (UI scale 1.054): name row x 134-300, y 740-766, bottom
left anchored.
'''
import cv2
import numpy as np

REF_W, REF_H = 1366, 768
NAME_BOX = (134, 740, 300, 766)   # x0, y0, x1, y1 at 1366x768
SHIFT = 3                          # +- px searched (rounding at other sizes)


def name_crop(frame, pad=0):
    '''grey name row of the HUD (pad: extra px around it, for matching)'''
    if frame is None:
        return None
    h, w = frame.shape[:2]
    s = (w / REF_W) if abs(w / h - REF_W / REF_H) < 0.05 else (h / REF_H)
    x0, y0, x1, y1 = NAME_BOX
    X0 = int(round(x0 * s)) - pad
    X1 = int(round(x1 * s)) + pad
    Y0 = int(round(h - (REF_H - y0) * s)) - pad
    Y1 = min(h, int(round(h - (REF_H - y1) * s)) + pad)
    X0, Y0 = max(0, X0), max(0, Y0)
    c = frame[Y0:Y1, X0:X1]
    if c.size == 0:
        return None
    return cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c


def has_text(crop):
    '''a name row has bright text strokes on the dark HUD box'''
    if crop is None or crop.size == 0:
        return False
    bright = np.count_nonzero(crop > 170)
    return 0.02 * crop.size < bright < 0.5 * crop.size


def match(frame, snippet):
    '''SQDIFF_NORMED of the saved snippet in the live name row (0 = same; 1 = nothing)'''
    if snippet is None:
        return 1.0
    live = name_crop(frame, pad=SHIFT)
    if live is None or live.shape[0] < snippet.shape[0] or live.shape[1] < snippet.shape[1]:
        return 1.0
    r = cv2.matchTemplate(live, snippet, cv2.TM_SQDIFF_NORMED)
    return float(np.nan_to_num(r, nan=1.0).min())


def identify(frame, snippets, thres=0.05, margin=0.15):
    '''
    snippets: {profile id: grey snippet}. Returns (id or None, scores): an id
    only when its score <= thres and every other profile is worse by margin
    (no switch on a weak match, an absent HUD or two similar names).
    '''
    scores = {pid: match(frame, sn) for pid, sn in snippets.items() if sn is not None}
    if not scores:
        return None, scores
    crop = name_crop(frame)
    if not has_text(crop):
        return None, scores
    best = min(scores, key=scores.get)
    others = [v for k, v in scores.items() if k != best]
    if scores[best] <= thres and all(v - scores[best] >= margin for v in others):
        return best, scores
    return None, scores
