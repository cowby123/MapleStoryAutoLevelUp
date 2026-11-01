'''
Resolution awareness.

Every template (monster/, nametag/, rune/, misc/) and every pixel number in
config_default.yaml was made on the upstream bot's working frame: the Windows
client 1282x693 resized to 1296x700 (REF_W x REF_H). The bot can now work on the
game's real frame (wingpu: the guest screen 1:1, e.g. 1366x768 or 1920x1080), so
these are rescaled to the actual frame:

- s (ui/world scale): how much bigger the game draws everything than in the
  reference frame. Measured on maple 2026-11-02 (1366x768 fullscreen vs the
  repo's templates): minimap 1.050-1.053, pig sprites 1.065-1.075, HP bar ~1.06,
  party red bar 47x5 px (fits the upstream 5-7 x <=50 rule unscaled).
  Default "auto" = frame width / 1296 (1.054 at 1366); see
  tools/wingpu_detect_test.py --scale to compare choices on saved frames.
- the minimap has its own scale m (route maps and map.png are in minimap
  pixels); MinimapScale measures it by matching the live minimap against
  minimaps/<map>/map.png.

At the reference size (Windows / macOS: always 1296x700) s == 1 and every
function here is the identity.
'''
# Standard import
import copy
import math

# Library import
import cv2
import numpy as np

# Local import
from src.utils.logger import logger

REF_H, REF_W = 700, 1296  # upstream working frame (h, w)
GREEN = (0, 255, 0)       # template background key colour


def compute_scale(h, w, setting="auto"):
    '''
    UI/world scale of a h x w game frame relative to the reference frame.
    setting: "auto" | "width" | "height" | "geometric" | number
    '''
    if isinstance(setting, (int, float)) and not isinstance(setting, bool) and setting > 0:
        return float(setting)
    if setting == "height":
        return h / REF_H
    if setting == "geometric":
        return math.sqrt((h / REF_H) * (w / REF_W))
    return w / REF_W  # "auto" / "width"


def map_point(pt, s, frame_hw, anchor="tl"):
    '''
    A point in reference-frame coordinates -> the actual frame, for a UI element
    anchored to a screen edge/centre. anchor = <x: l|c|r><y: t|c|b> as a 2-char
    string, e.g. "ct" = horizontally centred, top anchored.
    '''
    H, W = frame_hw
    x, y = pt
    ax, ay = anchor[0], anchor[1]
    if ax == "l":
        nx = x * s
    elif ax == "r":
        nx = W - (REF_W - x) * s
    else:
        nx = W / 2 + (x - REF_W / 2) * s
    if ay == "t":
        ny = y * s
    elif ay == "b":
        ny = H - (REF_H - y) * s
    else:
        ny = H / 2 + (y - REF_H / 2) * s
    return (int(round(nx)), int(round(ny)))


def map_box(top_left, bottom_right, s, frame_hw, anchor, margin=0):
    '''
    Search box in reference coords -> actual frame, grown by margin*s px, clipped.
    '''
    H, W = frame_hw
    x0, y0 = map_point(top_left, s, frame_hw, anchor)
    x1, y1 = map_point(bottom_right, s, frame_hw, anchor)
    m = int(round(margin * s))
    return ((max(0, x0 - m), max(0, y0 - m)), (min(W, x1 + m), min(H, y1 + m)))


def sc(v, s):
    '''
    Scale a length (int result, at least 1 when v > 0)
    '''
    r = int(round(v * s))
    return max(1, r) if v > 0 else r


def scale_template(img, s, key=GREEN):
    '''
    Resize a template by s. Colour templates with a green (0,255,0) background
    keep an exact green background (the masks are built from it): the image is
    resized with INTER_LINEAR, the background from an INTER_NEAREST copy.
    '''
    if img is None or abs(s - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    out = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    if img.ndim == 3 and key is not None:
        near = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        bg = np.all(near == key, axis=2)
        if bg.any():
            out[bg] = key
    return out


# (section, key) of config values that are lengths in the reference frame
_LENGTHS = [
    ("directional_attack", "range_x"), ("directional_attack", "range_y"),
    ("aoe_skill", "range_x"), ("aoe_skill", "range_y"),
    ("nametag", "split_width"),
    ("character", "width"), ("character", "height"),
    ("monster_detect", "search_box_margin"),
    ("rune_detect", "box_width"), ("rune_detect", "box_height"),
    ("rune_find", "rune_trigger_distance_x"), ("rune_find", "rune_trigger_distance_y"),
    ("rune_solver", "arrow_box_size"), ("rune_solver", "arrow_box_interval"),
]
_VECTORS = [("party_red_bar", "offset"), ("nametag", "offset")]
# message boxes on the screen: (section, anchor, extra search margin in ref px)
_MSG_BOXES = [
    ("rune_warning_cn", "ct", 12), ("rune_warning_eng", "ct", 12),
    ("rune_enable_msg_cn", "ct", 12), ("rune_enable_msg_eng", "ct", 12),
]
# click targets in upstream WINDOW coordinates (1282x693 client + 59 px title
# bar) and their anchors. NOT verified at other resolutions (needs live input).
_CLICKS = {
    "menu": "rb", "channel": "rb",
    "random_channel": "cc", "random_channel_confirm": "cc", "select_character": "cc",
}
UP_CLIENT_W, UP_CLIENT_H, UP_TITLE = 1282, 693, 59


def snapshot_base(cfg):
    '''
    Copy of the config sections apply_scale() rewrites (call once, after
    load_config's own normalisation).
    '''
    keys = {k for k, _ in _LENGTHS} | {k for k, _ in _VECTORS} | \
        {k for k, _, _ in _MSG_BOXES} | {"monster_detect", "ui_coords", "rune_solver"}
    return {k: copy.deepcopy(cfg[k]) for k in keys if k in cfg}


def apply_scale(cfg, base, s, frame_hw):
    '''
    Write the scaled values into cfg (in place: KeyBoardController, HealthMonitor
    and RuneSolver hold the same dict). base = snapshot_base(cfg).
    '''
    H, W = frame_hw
    for sec, key in _LENGTHS:
        if sec in base and key in base[sec]:
            cfg[sec][key] = sc(base[sec][key], s)
    for sec, key in _VECTORS:
        if sec in base and key in base[sec]:
            cfg[sec][key] = tuple(int(round(v * s)) for v in base[sec][key])
    md = base.get("monster_detect", {})
    if "max_mob_area_trigger" in md:
        cfg["monster_detect"]["max_mob_area_trigger"] = int(round(md["max_mob_area_trigger"] * s * s))
    if "contour_blur" in md:
        b = max(1, int(round(md["contour_blur"] * s)))
        cfg["monster_detect"]["contour_blur"] = b if b % 2 else b + 1
    for sec, anchor, margin in _MSG_BOXES:
        if sec in base:
            tl, br = map_box(base[sec]["top_left"], base[sec]["bottom_right"],
                             s, frame_hw, anchor, margin)
            cfg[sec]["top_left"], cfg[sec]["bottom_right"] = tl, br
    rs = base.get("rune_solver", {})
    if "arrow_box_coord" in rs:
        cfg["rune_solver"]["arrow_box_coord"] = map_point(rs["arrow_box_coord"], s, frame_hw, "cc")
    ui = base.get("ui_coords", {})
    if ui:
        # the bottom UI band (chat + status bars) is bottom anchored
        cfg["ui_coords"]["ui_y_start"] = int(round(H - (REF_H - ui["ui_y_start"]) * s))
        tl, br = map_box(ui["login_button_top_left"], ui["login_button_bottom_right"],
                         s, frame_hw, "cc", 20)
        cfg["ui_coords"]["login_button_top_left"] = tl
        cfg["ui_coords"]["login_button_bottom_right"] = br
        for name, anchor in _CLICKS.items():
            if name not in ui:
                continue
            x, y = ui[name]
            # window coords of the 1282x693 client -> reference frame
            ref = (x * REF_W / UP_CLIENT_W, (y - UP_TITLE) * REF_H / UP_CLIENT_H)
            cfg["ui_coords"][name] = map_point(ref, s, frame_hw, anchor)
    logger.info(f"[resolution] frame {W}x{H}: scale {s:.4f}; attack range_x "
                f"{cfg['directional_attack']['range_x']}, ui_y_start {cfg['ui_coords']['ui_y_start']}")


class MinimapScale:
    '''
    Scale m of the live minimap relative to minimaps/<map>/map.png (the route
    maps' pixel space): live_minimap_px = m * route_px. Measured by matching the
    live minimap, resized by 1/m, against map.png over a range of m.
    '''
    def __init__(self, img_map, thres=0.15):
        self.img_map = img_map
        self.thres = thres
        self.m = None      # measured scale, None until a good match
        self.score = None

    def measure(self, img_minimap, guess=1.0):
        '''
        Returns (m, score) of the best match, sets self.m when score < thres.
        '''
        if self.img_map is None or img_minimap is None:
            return None, None
        mh, mw = img_minimap.shape[:2]
        if mh < 20 or mw < 20:
            return None, None
        # leave out the edges (window frame, player dots move)
        tpl = img_minimap[3:mh - 3, 3:mw - 3]
        best = (1.0, None)

        def score_at(m):
            t = cv2.resize(tpl, (max(1, int(round(tpl.shape[1] / m))),
                                 max(1, int(round(tpl.shape[0] / m)))),
                           interpolation=cv2.INTER_AREA)
            if t.shape[0] > self.img_map.shape[0] or t.shape[1] > self.img_map.shape[1]:
                return 1.0
            r = cv2.matchTemplate(self.img_map, t, cv2.TM_SQDIFF_NORMED)
            return float(cv2.minMaxLoc(r)[0])

        lo, hi = max(0.5, guess * 0.8), guess * 1.25
        for m in np.arange(lo, hi, 0.01):
            v = score_at(m)
            if v < best[0]:
                best = (v, m)
        if best[1] is not None:
            for m in np.arange(best[1] - 0.01, best[1] + 0.0101, 0.0025):
                v = score_at(m)
                if v < best[0]:
                    best = (v, m)
        score, m = best
        if m is not None and score < self.thres:
            self.m, self.score = float(m), score
        return m, score

    def to_route(self, img_minimap):
        '''
        Live minimap -> route-map pixel scale (INTER_AREA), or as is when m ~ 1
        '''
        if self.m is None or abs(self.m - 1.0) < 1e-3:
            return img_minimap
        h, w = img_minimap.shape[:2]
        return cv2.resize(img_minimap, (max(1, int(round(w / self.m))), max(1, int(round(h / self.m)))),
                          interpolation=cv2.INTER_AREA)
