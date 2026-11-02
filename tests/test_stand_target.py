'''
"Walks left-right in place" (maple 2026-11-02 20:47): replay 20 s of the live
decision trace (tests/data/maple_oscillation_trace.json: per frame the
player, the detected monsters and the route's command) through the attack
logic with stand_while_target off (upstream) and on.

  python tests/test_stand_target.py
'''
import json
import logging
import os
import sys

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import wingpu_detect_test as T  # noqa: E402
import src.engine.MapleStoryAutoLevelUp as E  # noqa: E402

T.logger.set_level(logging.ERROR)
TRACE = json.load(open(os.path.join(ROOT, "tests", "data", "maple_oscillation_trace.json")))


class Clock:
    t = 0.0

    @staticmethod
    def time():
        return Clock.t

    @staticmethod
    def sleep(_):
        pass


def replay(stand):
    # the user's settings at the time: wedge, custom range 400, cooldown 0.1
    T.OVERRIDES.clear()
    T.OVERRIDES.update({
        "directional_attack.shape": "wedge", "directional_attack.wedge_range": 400,
        "directional_attack.wedge_start": 65, "directional_attack.cooldown": 0.1,
        "directional_attack.dead_zone_action": "ignore",
        "directional_attack.stand_while_target": stand,
        "monster_detect.interval_frames": 1,
    })
    bot = T.make_bot("wingpu_maple")
    bot.on_resolution(768, 1366)
    bot.img_frame = np.zeros((768, 1366, 3), np.uint8)
    bot.img_frame_debug = bot.img_frame.copy()
    real_time = E.time
    E.time = Clock
    out = []
    try:
        bot.t_last_attack = TRACE[0]["t"] - 1
        for fr in TRACE:
            Clock.t = fr["t"]
            bot.loc_player = tuple(fr["player"])
            bot.is_on_ladder = fr["on_ladder"]
            mobs = [{"name": "pig", "position": (x, y), "size": (h, w), "score": 0.1} for x, y, w, h in fr["mobs"]]
            bot.get_monsters_in_range = lambda *a, m=mobs: list(m)
            bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action = fr["route_cmd"].split()
            bot.dec = {"why": []}
            bot.update_cmd_by_mob_detection()
            out.append((bot.cmd_move_x, bot.cmd_action))
    finally:
        E.time = real_time
    return out


def flips(cmds):
    n, prev = 0, None
    for x, _ in cmds:
        if x in ("left", "right"):
            if prev and x != prev:
                n += 1
            prev = x
    return n


def test_replay():
    recorded = [(fr["kb"].split()[0], fr["kb"].split()[2]) for fr in TRACE]
    off, on = replay(False), replay(True)
    a = lambda c: sum(1 for _, act in c if act == "attack")
    walk = lambda c: sum(1 for x, act in c if x in ("left", "right") and act != "attack")
    print(f"  {len(TRACE)} frames ({TRACE[-1]['t'] - TRACE[0]['t']:.1f} s)")
    print(f"  recorded live:            left/right flips {flips(recorded):3d}, attacks {a(recorded)}, walking frames {walk(recorded)}")
    print(f"  replay, option off:       left/right flips {flips(off):3d}, attacks {a(off)}, walking frames {walk(off)}")
    print(f"  replay, stand_while_target: flips {flips(on):3d}, attacks {a(on)}, walking frames {walk(on)}")
    # the replay with the option off reproduces the live behaviour
    assert abs(flips(off) - flips(recorded)) <= 0.15 * flips(recorded) + 3, (flips(off), flips(recorded))
    assert flips(on) <= flips(off) // 5, (flips(on), flips(off))
    assert a(on) >= 0.8 * a(off), (a(on), a(off))    # it still shoots about as often


if __name__ == "__main__":
    test_replay()
    print("ok   test_replay")
