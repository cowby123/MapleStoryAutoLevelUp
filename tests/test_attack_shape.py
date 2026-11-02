'''
Directional attack reach: rect (upstream) vs wedge (MapleStory ranged targeting).

  python tests/test_attack_shape.py
'''
import glob
import logging
import os
import sys

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import wingpu_detect_test as T  # noqa: E402
from src.utils.attack_shape import wedge_test, preset_values, eye_of_amazon, keen_eyes  # noqa: E402

T.logger.set_level(logging.WARNING)
W = dict(start=65, rng=300, slope=4, center_dy=-1)


def box_at(px, py, d, dy, w=40, h=30, left=False):
    '''box whose near edge is d px ahead and whose centre is dy px below the centre line'''
    cy = py + W["center_dy"] + dy
    x = px - d - w if left else px + d
    return (x, int(cy - h / 2), w, h)


def t(box, left=False):
    return wedge_test(500, 400, box, left, W["start"], W["rng"], W["slope"], W["center_dy"])[0]


def test_geometry():
    assert t(box_at(500, 400, 100, 0)) == "in"
    assert t(box_at(500, 400, 100, 0, left=True), left=True) == "in"
    assert t(box_at(500, 400, 100, 0, left=True)) == "behind"
    assert t(box_at(500, 400, 10, 0)) == "close"          # 10-50 px: dead zone
    assert t(box_at(500, 400, 50, 0, w=40)) == "in"       # reaches past 65
    assert t(box_at(500, 400, 310, 0)) == "far"
    # vertical reach grows with distance: 70 px above the centre line
    assert t(box_at(500, 400, 100, -70, w=30, h=20)) == "out"   # far edge 130 -> reach 32
    assert t(box_at(500, 400, 250, -70, w=30, h=20)) == "in"    # far edge 280 -> reach 70
    # a rect of range_y 70 (+-35) never reaches 70 px above
    print("  geometry ok")


def test_presets():
    assert preset_values("archer", 0) == (65, 300)
    assert preset_values("archer", 15) == (65, 300 + 45 + 75)
    assert preset_values("arrow_blow", 1) == (65, 350 + 50)
    assert preset_values("mage") == (50, 300) and preset_values("energy_bolt") == (50, 350)
    assert keen_eyes(14) == 180 and keen_eyes(15) == 200 and preset_values("thief_claw", 15) == (65, 400)
    assert eye_of_amazon(0) == 0 and eye_of_amazon(20) == 120
    print("  presets ok")


def make(shape):
    T.OVERRIDES.clear()
    T.OVERRIDES.update({"directional_attack.shape": shape})
    bot = T.make_bot("wingpu_maple")
    bot.img_frame_debug = np.zeros((768, 1366, 3), np.uint8)
    bot.frame_hw = None
    return bot


def mob(x, y, w=60, h=45):
    return {"name": "pig", "position": (x, y), "size": (h, w), "score": 0.1}


def test_nearest_in_wedge():
    bot = make("wedge")
    bot.scale = 1.0
    bot.loc_player = (600, 400)
    s, r, sl, cdy = bot.wedge_params()
    assert (s, r, sl) == (65, 300, 4), bot.wedge_params()
    # right: one in the dead zone (20 px), one at 120 px, one at 250 px; left: one at 200 px, 60 px up
    close, mid, far = mob(620, 380), mob(720, 380), mob(850, 380)
    up_left = mob(600 - 200 - 60, 400 - 1 - 60 - 22)
    bot.monsters = [close, mid, far, up_left]
    assert bot.get_nearest_monster(is_left=False) is mid      # nearest *inside* the wedge
    assert bot.get_nearest_monster(is_left=True) is up_left   # 60 px up at 200-260 px: inside
    # rect for comparison: range_x 250, range_y 70 -> the high left one is out, the close one is in
    rect = make("rect")
    rect.scale = 1.0
    rect.loc_player = (600, 400)
    rect.monsters = bot.monsters
    assert rect.get_nearest_monster(is_left=True) is None
    assert rect.get_nearest_monster(is_left=False) is close
    # only the dead-zone monster: no attack, step away from it
    bot.monsters = [close]
    assert bot.get_nearest_monster(False) is None and bot.get_nearest_monster(True) is None
    bot.cmd_move_x = "none"
    bot.back_off_from_dead_zone()
    assert bot.cmd_move_x == "left", bot.cmd_move_x
    bot.cfg["directional_attack"]["dead_zone_action"] = "ignore"
    bot.cmd_move_x = "none"
    bot.back_off_from_dead_zone()
    assert bot.cmd_move_x == "none"
    print("  nearest-in-wedge ok: rect picks the dead-zone pig and misses the high one; wedge the opposite")


def test_fixtures_rect_vs_wedge():
    frames = sorted(glob.glob("fixtures/maple2/*.png")) + sorted(glob.glob("fixtures/maple/f20261102_0028*.png"))
    if not frames:
        print("  (no fixtures, skipped)")
        return
    bots = {"rect": make("rect"), "wedge": make("wedge")}
    rows = []
    for f in frames:
        img = cv2.imread(f)
        out = {}
        for name, bot in bots.items():
            bot.capture.set_guest(img)
            bot.t_minimap_ok = bot.t_player_seen = 0
            bot.minimap_rect = None
            bot.run_once()
            if bot.is_lost_paused:
                out[name] = None
                continue
            # decisions on the frame's monsters (no cooldown involved)
            left, right = bot.get_nearest_monster(True), bot.get_nearest_monster(False)
            d = bot.get_attack_direction(left, right)
            bot.cmd_move_x = "none"
            if d is None:
                bot.back_off_from_dead_zone()
            out[name] = (len(bot.monsters), d, bot.cmd_move_x if d is None else None)
        rows.append((os.path.basename(f)[:24], out))
    act = [(n, o) for n, o in rows if o["rect"] is not None]
    same = sum(1 for _, o in act if o["rect"][1] == o["wedge"][1])
    a_r = sum(1 for _, o in act if o["rect"][1])
    a_w = sum(1 for _, o in act if o["wedge"][1])
    back = sum(1 for _, o in act if o["wedge"][2] not in (None, "none"))
    print(f"  fixtures: {len(act)} frames with the player; attack direction: rect {a_r}, wedge {a_w}, "
          f"same decision {same}; wedge back-offs (monster in the dead zone) {back}")
    for n, o in act:
        if o["rect"][1] != o["wedge"][1] or o["wedge"][2] not in (None, "none"):
            print(f"    {n} mobs {o['rect'][0]}: rect -> {o['rect'][1]}, wedge -> {o['wedge'][1]}"
                  + (f" (back off {o['wedge'][2]})" if o["wedge"][2] not in (None, "none") else ""))


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
            n += 1
    print(f"{n} tests passed")
