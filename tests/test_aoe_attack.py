'''
attack.mode 單體 / 群體 / 自動 (src/engine/AoeAttack.py + MapleStoryAutoBot.try_aoe):
synthetic boxes, the decision rules (threshold, cooldown, MP floor, facing,
stand_while_target), and the count on the hand-labelled cloud_balcony frames.

  python tests/test_aoe_attack.py
'''
import glob
import json
import logging
import os
import sys
import time

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import cv2  # noqa: E402
import wingpu_detect_test as T  # noqa: E402
from src.engine.AoeAttack import (AoeCounter, merge_boxes, count_in_area,  # noqa: E402
                                  aoe_params, in_area)

T.logger.set_level(logging.ERROR)


def mob(cx, cy, w=50, h=50, name="bear"):
    return {"name": name, "position": (cx - w // 2, cy - h // 2), "size": (h, w), "score": 0.1}


def test_merge_and_count():
    # one bear: template hit + its health bar box + a second template hit
    boxes = [(100, 100, 50, 50), (95, 110, 70, 46), (104, 98, 50, 52), (300, 100, 50, 50)]
    assert len(merge_boxes(boxes, 30)) == 2
    # two bears side by side (60 px apart) stay two
    assert len(merge_boxes([(100, 100, 50, 50), (160, 100, 50, 50)], 30)) == 2
    p = {"shape": "box", "range_x": 400, "range_y": 170}
    b = [(x - 25, 375, 50, 50) for x in (300, 450, 650, 900)]   # player at 500,400
    assert count_in_area(500, 400, b, p) == (3, 3)              # 900 is outside 300..700
    p = {"shape": "front", "start": 0, "range": 300, "height": 170}
    assert count_in_area(500, 400, b, p) == (2, 1)              # left: 300, 450; right: 650
    p = {"shape": "wedge", "start": 65, "range": 300, "slope": 4, "dy": 0}
    assert in_area(500, 400, (600, 375, 50, 50), False, p)
    assert not in_area(500, 400, (520, 375, 20, 50), False, p)  # dead zone
    assert not in_area(500, 400, (600, 375, 50, 50), True, p)   # behind


def test_counter():
    c = AoeCounter()
    t0 = 1000.0
    for i, n in enumerate([4, 2, 4, 4, 1]):
        c.add(n, 0, t=t0 + 0.05 * i)
    assert c.value(0.3, "max", t=t0 + 0.2) == (4, 0)
    assert c.value(0.3, "median", t=t0 + 0.2) == (4, 0)
    # old values fall out of the window
    c.add(1, 0, t=t0 + 1.0)
    assert c.value(0.3, "max", t=t0 + 1.0) == (1, 0)


def make(mode="auto", **attack):
    T.OVERRIDES.clear()
    T.OVERRIDES["bot.map"] = "cloud_balcony"
    bot = T.make_bot("wingpu_maple")
    T.OVERRIDES.clear()
    bot.on_resolution(768, 1366)
    bot.cfg["attack"]["mode"] = mode
    bot.cfg["attack"].update(attack)
    bot.cfg["key"]["aoe_skill"] = "q"
    bot.loc_player = (500, 400)
    bot.is_on_ladder = False
    bot.cmd_move_x = bot.cmd_move_y = bot.cmd_action = "none"
    bot.dec = {"why": []}
    return bot


def decide(bot, mobs):
    bot.monsters = mobs
    bot.cmd_move_x = bot.cmd_move_y = bot.cmd_action = "none"
    bot.dec = {"why": []}
    bot.update_aoe_count()
    return bot.try_aoe(bot.attack_mode())


def test_decisions():
    four = [mob(x, 400) for x in (380, 450, 560, 640)]
    one = [mob(560, 400)]
    bot = make("auto", aoe_min_count=3, aoe_cooldown=1.0, aoe_count_window_s=0.0)
    assert decide(bot, four) == "fired" and bot.cmd_action == "aoe"
    assert decide(bot, four) == "single"          # cooling down: single meanwhile
    bot.t_last_aoe = 0
    assert decide(bot, one) == "single"           # 1 < 3
    # MP floor
    bot.health_monitor.mp_percent = 10
    bot.cfg["attack"]["aoe_mp_floor"] = 30
    assert decide(bot, four) == "fallback"
    bot.cfg["attack"]["aoe_mp_floor"] = 0
    # 群體 only
    bot = make("aoe", aoe_cooldown=1.0, aoe_count_window_s=0.0)
    assert decide(bot, one) == "fired"
    assert decide(bot, one) == "wait"
    assert decide(bot, [mob(1200, 400)]) == "none"  # outside the area
    # single (default): no AoE at all
    bot = make("single")
    assert bot.attack_mode() == "single"


def test_facing_and_stand():
    left3 = [mob(x, 400) for x in (300, 350, 420)] + [mob(620, 400)]
    bot = make("auto", aoe_shape="front", aoe_front_start=0, aoe_front_range=300,
               aoe_height=170, aoe_count_window_s=0.0)
    bot.cfg["directional_attack"]["stand_while_target"] = True
    assert decide(bot, left3) == "fired"
    assert bot.cmd_move_x == "left" and bot.cmd_move_y == "none"
    assert bot.target_side == "left"


def test_jitter_window():
    four = [mob(x, 400) for x in (380, 450, 560, 640)]
    bot = make("auto", aoe_cooldown=0.0, aoe_count_window_s=0.3, aoe_count_stat="max")
    assert decide(bot, four) == "fired"
    # next frame the detector misses two of them: the window still says 4
    assert decide(bot, four[:2]) == "fired"


def test_explain_text():
    from src.engine.explain import explain
    bot = make("auto")
    rec = {"why": [{"kind": "aoe_attack", "mode": "auto", "n": 4, "need": 3, "key": "q", "side": None}],
           "player": [500, 400], "keys": []}
    assert explain(bot, rec, time.time())[1] == "群體技能：範圍內 4 隻 ≥ 3（Q）"
    rec["why"] = [{"kind": "aoe_count", "reason": "count", "mode": "auto", "n": 1, "need": 3, "key": "q"},
                  {"kind": "attack_target", "dir": "left", "name": "pink_windup_bear", "target": [400, 380, 50, 50]}]
    main = explain(bot, rec, time.time())[1]
    assert main.startswith("單體攻擊：範圍內 1 隻 < 3："), main
    print(f"  {main}")


def test_cloud_fixtures():
    '''
    auto mode on the labelled frames: the bot's count (detected, merged boxes in
    the AoE box around the detected player) vs the count from the labels in the
    same box; and the counting's CPU
    '''
    path = "fixtures/cloud/labels_all.json"
    if not os.path.exists(path):
        print("  (no fixtures, skipped)")
        return
    labels = json.load(open(path))
    T.OVERRIDES.clear()
    T.OVERRIDES["bot.map"] = "cloud_balcony"
    T.OVERRIDES["attack.mode"] = "auto"
    bot = T.make_bot("wingpu_maple")
    T.OVERRIDES.clear()
    p = None
    agree = n = 0
    rows, cost = [], []
    for name, labs in labels.items():
        img = cv2.imread(os.path.join("fixtures/cloud", name))
        if img is None:
            continue
        bot.capture.set_guest(img)
        bot.t_minimap_ok = bot.t_player_seen = 0.0
        bot.minimap_rect = None
        bot.aoe_counter = AoeCounter()   # standalone frame
        bot.run_once()
        if getattr(bot, "aoe_count", None) is None or tuple(bot.loc_player) == (0, 0):
            continue
        p = aoe_params(bot.cfg)
        px, py = bot.loc_player
        inside = [l for l in labs if in_area(px, py, (l[0] - 25, l[1] - 25, 50, 50), False, p)]
        truth = sum(len(l) < 4 for l in inside)          # hard labels (fading, behind cards) may or may not count
        hard = len(inside) - truth
        got = bot.aoe_count["left"]
        t0 = time.perf_counter()
        for _ in range(100):
            bot.update_aoe_count()
        cost.append((time.perf_counter() - t0) / 100)
        rows.append((name[-7:-4], truth, got))
        n += 1
        agree += truth - 1 <= got <= truth + hard + 1
    print(f"  frames {n}: (frame, labels in area, bot count) {rows}")
    print(f"  |count - labels| <= 1 on {agree}/{n}; counting {1e6 * sum(cost) / max(1, len(cost)):.0f} us/frame")
    assert n >= 8 and agree >= 0.75 * n, rows
    assert sum(cost) / len(cost) < 0.001


def test_single_unchanged_on_pigs():
    '''default mode on gon_pig fixtures: never an AoE action'''
    frames = sorted(glob.glob("fixtures/maple2/*.png"))[:20]
    if not frames:
        print("  (no fixtures, skipped)")
        return
    T.OVERRIDES.clear()
    bot = T.make_bot("wingpu_maple")
    for f in frames:
        bot.capture.set_guest(cv2.imread(f))
        bot.t_minimap_ok = bot.t_player_seen = 0.0
        bot.minimap_rect = None
        bot.run_once()
        assert bot.cmd_action != "aoe"
        assert getattr(bot, "aoe_count", None) is None


if __name__ == "__main__":
    for t in (test_merge_and_count, test_counter, test_decisions, test_facing_and_stand,
              test_jitter_window, test_explain_text, test_cloud_fixtures, test_single_unchanged_on_pigs):
        t()
        print(f"{t.__name__}: ok")
