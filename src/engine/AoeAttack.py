'''
AoE (群體) skill vs single-target attack by how many monsters are in the AoE
skill's area (attack.mode: single | aoe | auto).

Pure helpers (no bot state) so they can be tested with synthetic boxes:
  merge_boxes      one box per monster (template hits + health bars overlap)
  aoe_area         the area's geometry for one facing side
  count_in_area    how many merged monsters are inside it
  AoeCounter       counts over the last window_s (max / median: detection jitter)
'''
import time
from collections import deque

from src.utils.attack_shape import wedge_test, wedge_polygon

SHAPES = ("box", "front", "wedge")


def box_center(b):
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


def merge_boxes(boxes, merge_px):
    '''
    boxes (x, y, w, h) -> one box per monster: a box whose centre is within
    merge_px horizontally and 1.5 * merge_px vertically of a kept box (or that
    overlaps it by half of the smaller one) is the same monster (a template hit
    and its health bar box, two templates on one bear).
    '''
    kept = []
    for b in sorted(boxes, key=lambda b: (b[0], b[1])):
        cx, cy = box_center(b)
        same = False
        for k in kept:
            kx, ky = box_center(k)
            if abs(cx - kx) <= merge_px and abs(cy - ky) <= 1.5 * merge_px:
                same = True
                break
            ix = max(0, min(b[0] + b[2], k[0] + k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[1] + b[3], k[1] + k[3]) - max(b[1], k[1]))
            if ix * iy >= 0.5 * min(b[2] * b[3], k[2] * k[3]) > 0:
                same = True
                break
        if not same:
            kept.append(b)
    return kept


def aoe_area(px, py, facing_left, p):
    '''
    p: dict shape, range_x, range_y (box), start, range, height, dy, slope.
    Returns ("rect", (x0, y0, x1, y1)) or ("wedge", (start, range, slope, dy)).
    box:   range_x x range_y centred on the character (not directional)
    front: from start to start + range in front, height high, centred dy below py
    wedge: like the directional wedge: start..range, half-height d / slope
    '''
    shape = p.get("shape", "box")
    cy = py + p.get("dy", 0)
    if shape == "wedge":
        return "wedge", (p.get("start", 0), p.get("range", 300), max(0.5, float(p.get("slope", 4))), p.get("dy", 0))
    if shape == "front":
        a, b = p.get("start", 0), p.get("start", 0) + p.get("range", 300)
        x0, x1 = (px - b, px - a) if facing_left else (px + a, px + b)
        h = p.get("height", 170)
        return "rect", (x0, cy - h / 2.0, x1, cy + h / 2.0)
    dx, dy = p.get("range_x", 400) / 2.0, p.get("range_y", 170) / 2.0
    return "rect", (px - dx, cy - dy, px + dx, cy + dy)


def in_area(px, py, box, facing_left, p):
    kind, g = aoe_area(px, py, facing_left, p)
    if kind == "wedge":
        return wedge_test(px, py, box, facing_left, *g)[0] == "in"
    cx, cy = box_center(box)
    x0, y0, x1, y1 = g
    return x0 <= cx <= x1 and y0 <= cy <= y1


def count_in_area(px, py, boxes, p):
    '''(left, right) counts; for the non-directional box both are the total'''
    if p.get("shape", "box") == "box":
        n = sum(in_area(px, py, b, False, p) for b in boxes)
        return n, n
    return (sum(in_area(px, py, b, True, p) for b in boxes),
            sum(in_area(px, py, b, False, p) for b in boxes))


def area_polygons(px, py, p):
    '''[(facing_left, [points])] for drawing (one for box)'''
    out = []
    sides = (False,) if p.get("shape", "box") == "box" else (True, False)
    for fl in sides:
        kind, g = aoe_area(px, py, fl, p)
        if kind == "wedge":
            pts = wedge_polygon(px, py, fl, *g)
        else:
            x0, y0, x1, y1 = (int(v) for v in g)
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        out.append((fl, pts))
    return out


class AoeCounter:
    '''
    Counts (left, right) of the last window_s seconds; value() is the max or
    the median of each side (a monster missed in one detection frame, or a
    false hit in one frame, does not flip the decision).
    '''
    def __init__(self):
        self.hist = deque()

    def add(self, left, right, t=None):
        t = time.time() if t is None else t
        self.hist.append((t, left, right))

    def value(self, window_s=0.3, stat="max", t=None):
        t = time.time() if t is None else t
        while len(self.hist) > 1 and t - self.hist[0][0] > window_s:   # the latest always counts
            self.hist.popleft()
        if not self.hist:
            return 0, 0
        ls = sorted(h[1] for h in self.hist)
        rs = sorted(h[2] for h in self.hist)
        if stat == "median":
            return ls[len(ls) // 2], rs[len(rs) // 2]
        return ls[-1], rs[-1]


def aoe_params(cfg):
    '''geometry dict from cfg (scaled values: attack.* lengths, aoe_skill.range_x/y)'''
    a = cfg.get("attack") or {}
    return {"shape": a.get("aoe_shape", "box"),
            "range_x": (cfg.get("aoe_skill") or {}).get("range_x", 400),
            "range_y": (cfg.get("aoe_skill") or {}).get("range_y", 170),
            "start": a.get("aoe_front_start", 0), "range": a.get("aoe_front_range", 300),
            "height": a.get("aoe_height", 170), "dy": a.get("aoe_dy", 0),
            "slope": a.get("aoe_wedge_slope", 4)}
