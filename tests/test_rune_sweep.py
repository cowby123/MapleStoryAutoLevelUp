'''
Rune final approach (rune_curse.sweep_*): closeness from the guide arrow
(it flips, disappears, or points at the feet while the drop-down does not
move us), movement keys released at once, small left / right taps with the
interact key, back to the long walk when the arrow points away, timeout,
and the mini-game taking over. Synthetic arrow sequences + the 2026-11-06
01:43-01:44 session replayed from the capture metadata.

  python tests/test_rune_sweep.py
'''
import glob
import json
import logging
import math
import os
import sys
import time

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import wingpu_detect_test as T  # noqa: E402
import src.engine.MapleStoryAutoLevelUp as M  # noqa: E402
import src.input.KeyBoardController as K  # noqa: E402

T.logger.set_level(logging.ERROR)
PRESSED = []
K.press_key = lambda key, duration=0.05: PRESSED.append((round(time.time(), 3), key, duration))
ARROW = {"a": None}
M.find_guide_arrow = lambda *a, **k: ARROW["a"]


def arrow(angle):
    if angle is None:
        return None
    r = math.radians(angle)
    return {"centre": (700, 600), "dir": (math.cos(r), -math.sin(r)), "angle": angle, "area": 500, "solidity": 0.8}


def make():
    T.OVERRIDES.clear()
    T.OVERRIDES["rune_curse.enable"] = True
    for k, v in {"sweep_tap_s": 0.01, "sweep_tap_gap": 0.01, "sweep_up_interval": 0.05,
                 "sweep_px_per_tap": 10, "sweep_max_px": 40, "sweep_away_s": 0.3, "sweep_timeout": 2.0,
                 "sweep_down_s": 0.3, "interact_cooldown": 0.8}.items():
        T.OVERRIDES[f"rune_curse.{k}"] = v
    bot = T.make_bot("wingpu_maple")
    T.OVERRIDES.clear()
    bot.on_resolution(768, 1366)
    bot.img_frame = np.zeros((768, 1366, 3), np.uint8)
    bot.loc_player = (650, 606)
    bot.dec = {"why": []}
    now = time.time()
    bot.rune = {"phase": "seek", "t0": now, "t_interact": 0.0, "moved": 0.0, "last_x": 650,
                "t_arrow": now, "arrow": None, "presses": 0}
    PRESSED.clear()
    return bot


def step(bot, angle, dt=0.05, y=None):
    ARROW["a"] = arrow(angle)
    if y is not None:
        bot.loc_player = (bot.loc_player[0], y)
    bot.dec = {"why": []}
    bot.rune_seek(bot.rune_cfg(), bot.rune, time.time())
    time.sleep(dt)
    return [w["kind"] for w in bot.dec["why"]]


def taps():
    return [k for _, k, _ in PRESSED if k in ("left", "right")]


def test_flip_starts_sweep():
    bot = make()
    for _ in range(5):
        step(bot, 0)                         # walking right
    assert bot.kb.cmd == "right none none" and bot.rune.get("sweep") is None
    step(bot, 180)                           # one left read: not yet
    assert bot.rune.get("sweep") is None
    kinds = step(bot, 180)                   # second: walked past it
    assert bot.rune.get("sweep") is not None and "rune_near" in kinds
    assert bot.kb.cmd == "none none none"    # movement keys released at once
    for _ in range(30):
        step(bot, None, dt=0.03)
    t = taps()
    assert t and t[0] == "left", t           # first back toward where we came from
    assert "right" in t and "left" in t
    ups = [k for _, k, _ in PRESSED if k == "up"]
    assert len(ups) >= 5, len(ups)
    times = [tt for tt, k, _ in PRESSED if k == "up"]
    assert min(b - a for a, b in zip(times, times[1:])) >= 0.045   # rate limited
    # the sweep widens: +-10, +-20, +-40 px (10 px per tap)
    assert bot.rune["sweep"]["targets"] == [-1, 1, -2, 2, -4, 4], bot.rune["sweep"]["targets"]
    print(f"  taps {''.join('L' if k == 'left' else 'R' for k in t)}; up x{len(ups)}")


def test_gone_and_away():
    bot = make()
    for _ in range(5):
        step(bot, 180)                       # walking left
    step(bot, None, dt=0.35)                 # arrow gone > 0.3 s
    step(bot, None)
    assert bot.rune.get("sweep") is not None and bot.rune["sweep"]["why"] == "箭頭不見了"
    # the arrow points right, clearly, for longer than sweep_away_s: walk again
    for _ in range(9):
        step(bot, 0, dt=0.05)
    assert bot.rune.get("sweep") is None
    step(bot, 0)
    assert bot.kb.cmd == "right none none", bot.kb.cmd


def test_down_at_feet():
    bot = make()
    for _ in range(3):
        step(bot, -103, dt=0.15, y=606)       # down, but the drop does not move us
    assert bot.rune.get("sweep") is not None and bot.rune["sweep"]["why"] == "箭頭指向腳下"
    assert bot.rune["sweep"]["targets"][0] < 0  # the arrow leans left: left first
    bot = make()
    step(bot, -67, dt=0.15, y=500)
    step(bot, -67, dt=0.15, y=560)            # we did drop: no sweep
    step(bot, -67, dt=0.15, y=606)
    assert bot.rune.get("sweep") is None


def test_timeout():
    bot = make()
    for _ in range(5):
        step(bot, 0)
    step(bot, 180)
    step(bot, 180)
    assert bot.rune.get("sweep") is not None
    t0 = time.time()
    while bot.rune.get("sweep") is not None and time.time() - t0 < 4:
        step(bot, None, dt=0.05)
    assert bot.rune.get("sweep") is None and 1.9 < time.time() - t0 < 3.0


def test_minigame_takes_over():
    bot = make()
    for _ in range(5):
        step(bot, 0)
    step(bot, 180)
    step(bot, 180)
    bot.cursed = True
    bot.minigame = [{"dir": "up"}, {"dir": "left"}]
    bot.mg_total = 2
    bot.t_mg_seen = time.time()
    PRESSED.clear()
    bot.run_rune_curse()
    assert bot.rune["phase"] == "game" and [k for _, k, _ in PRESSED] == ["up"]


def test_replay_0144():
    '''2026-11-06 01:43:52-01:44:20: the arrow pointed down at x 644 for 20 s
    (the bot drop-jumped in place), then the mini-game appeared at x 674'''
    fs = sorted(glob.glob("fixtures/rune_auto/auto/curse_20261106-0143*.json") +
                glob.glob("fixtures/rune_auto/auto/curse_20261106-0144*.json"))
    if not fs:
        print("  (no capture metadata, skipped)")
        return
    bot = make()
    bot.cfg["rune_curse"]["sweep_down_s"] = 1.5
    started = None
    for f in fs:
        d = json.load(open(f, encoding="utf-8"))
        if d.get("phase") != "seek":
            continue
        a = d.get("arrow")
        ARROW["a"] = a
        bot.loc_player = tuple(d["player"])
        bot.dec = {"why": []}
        # frames were 2 s apart: replay them 2 s apart in bot time
        t = bot.rune["t0"] + 2.0 * fs.index(f)
        bot.rune_seek(bot.rune_cfg(), bot.rune, t)
        if bot.rune.get("sweep") is not None and started is None:
            started = os.path.basename(f)[6:-5]
    assert started is not None, "no sweep in the replay"
    print(f"  replay: sweep started at {started} (before: drop-jumping in place until the mini-game at 01:44:20)")


if __name__ == "__main__":
    for t in (test_flip_starts_sweep, test_gone_and_away, test_down_at_feet, test_timeout,
              test_minigame_takes_over, test_replay_0144):
        t()
        print(f"{t.__name__}: ok")
