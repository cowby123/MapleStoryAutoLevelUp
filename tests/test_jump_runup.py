'''
"一直原地跳" (maple 2026-11-03 04:43): the route says "right none jump" at
minimap x 182, the character stands against the ledge's wall and jumps
straight up again and again (x never changes).

1. replay 16 s of the live decision trace (tests/data/maple_jump_stuck_trace.json)
   through route.jump_runup off / on;
2. a small simulation of the ledge (wall at x 182, platform above from 183):
   a jump only gets onto the ledge with speed at take-off.

  python tests/test_jump_runup.py
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

T.logger.set_level(logging.ERROR)
TRACE = json.load(open(os.path.join(ROOT, "tests", "data", "maple_jump_stuck_trace.json")))


def make(on):
    T.OVERRIDES.clear()
    T.OVERRIDES.update({"route.jump_runup": on})
    bot = T.make_bot("wingpu_maple")
    bot.img_frame_debug = np.zeros((768, 1366, 3), np.uint8)
    return bot


def step(bot, t, glob, route_cmd):
    bot.loc_player_global = tuple(glob)
    bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action = route_cmd.split()
    bot.dec = {"why": []}
    bot.jump_runup(now=t)
    return f"{bot.cmd_move_x} {bot.cmd_move_y} {bot.cmd_action}"


def test_replay_trigger():
    stuck = [fr for fr in TRACE if fr["route_cmd"].split()[0] == "right"]
    for on in (False, True):
        bot = make(on)
        out = [step(bot, fr["t"], fr["global"], fr["route_cmd"]) for fr in TRACE]
        back = sum(1 for c in out if c.startswith("left none none"))
        jumps = sum(1 for c in out if c.endswith("jump"))
        n = getattr(bot, "runup_state", {}) or {}
        print(f"  replay ({len(TRACE)} frames, {TRACE[-1]['t'] - TRACE[0]['t']:.0f} s), jump_runup {'on ' if on else 'off'}: "
              f"jump frames {jumps}, walk-back frames {back}, run-ups started {n.get('n', 0)}")
        if not on:
            assert back == 0
        else:
            assert n.get("n", 0) >= 2 and back > 0
    assert len(stuck) > 100


def simulate(on, seconds=12.0, fps=20):
    '''
    Lower step y 101 up to the wall at x 182, platform y 96 from x 183.
    Walking 0.5 px/frame. A jump lasts 0.7 s; the wall blocks until 0.25 s
    into it; the horizontal speed in the air is the speed at take-off.
    Route: x <= 181 "right none none", x 182 "right none jump"; on the
    platform "right none none". (Walking in from the left with speed works
    without the option too -- earlier traces have such successful jumps.)
    '''
    bot = make(on)
    # standing still against the wall (it stopped there: shooting a pig /
    # detection pause) -- the situation of the live trace
    x, y, vx, air, t = 182.0, 101, 0.0, 0.0, 1000.0
    prev_x = x
    for i in range(int(seconds * fps)):
        t += 1.0 / fps
        on_top = y == 96
        route = "right none none" if (on_top or x < 182) else "right none jump"
        cmd = step(bot, t, (int(round(x)), y), route)
        mx, _, act = cmd.split()
        want = 0.5 if mx == "right" else -0.5 if mx == "left" else 0.0
        if air > 0:                                   # in the air: keep take-off speed
            air -= 1.0 / fps
            nx = x + vx
            if y == 101 and nx > 182 and 0.7 - air < 0.25:
                nx = 182                              # still below the ledge: the wall
            x = nx
            if air <= 0:
                y = 96 if x >= 183 else 101
                vx = 0.0
        else:
            if act == "jump":
                vx = x - prev_x if abs(x - prev_x) > 0.01 else 0.0
                air = 0.7
            prev_x = x
            nx = x + want
            if y == 101 and nx > 182:
                nx = 182
            x = nx
        if y == 96:
            return t - 1000.0
    return None


def test_simulated_ledge():
    off, on = simulate(False), simulate(True)
    print(f"  simulated ledge: off -> {'on the ledge after %.1f s' % off if off else 'never gets up (12 s)'}; "
          f"on -> {'on the ledge after %.1f s' % on if on else 'never gets up'}")
    assert off is None and on is not None


if __name__ == "__main__":
    test_replay_trigger()
    print("ok   test_replay_trigger")
    test_simulated_ledge()
    print("ok   test_simulated_ledge")
