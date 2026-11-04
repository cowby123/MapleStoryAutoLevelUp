'''
Rune arrow mini-game: reading the icons (real frames, maple 2026-11-04, map
cloud_balcony, the player solving it by hand), mirrored / rotated synthetic
cases for the never-seen 左, and the solver pressing the arrows in order.
Needs fixtures/rune_game/ (62 frames; the last 8 show the mini-game).

  python tests/test_rune_game.py
'''
import glob
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
import numpy as np  # noqa: E402
import wingpu_detect_test as T  # noqa: E402
from src.engine.RuneGame import read_minigame  # noqa: E402
from src.input.WingpuKeyboard import get_keyboard  # noqa: E402

T.logger.set_level(logging.ERROR)
S = 1366 / 1296
GAME = sorted(glob.glob("fixtures/rune_game/*.png"))
# what is on screen in the last 8 frames (checked by eye on the frames)
EXPECT = [["down", "right", "up", "right"]] * 3 + [["right", "up", "right"], ["up", "right"],
                                                   ["right"], ["right"], ["right"]]


def seq(img):
    return [c["dir"] for c in read_minigame(img, S)]


def test_read_real():
    if len(GAME) < 62:
        print("  (no fixtures, skipped)")
        return
    got = [seq(cv2.imread(f)) for f in GAME[-8:]]
    for g, e, f in zip(got, EXPECT, GAME[-8:]):
        assert g == e, (f, g, e)
    print("  8 mini-game frames read as on screen:", [" ".join(d[0] for d in g) for g in got])
    print("  (the 3rd icon of frame 59 is fading out: not counted)")
    negs = GAME[:-8] + sorted(glob.glob("fixtures/maple*/*.png")) + sorted(glob.glob("fixtures/rune/*.png"))
    fp = [f for f in negs if seq(cv2.imread(f))]
    print(f"  no mini-game on {len(negs) - len(fp)}/{len(negs)} other frames")
    assert not fp, fp


def test_left_synthetic():
    if len(GAME) < 62:
        print("  (no fixtures, skipped)")
        return
    img = cv2.imread(GAME[-8])
    mir = cv2.flip(img, 1)
    assert seq(mir) == ["left", "up", "left", "down"], seq(mir)
    # one icon's arrow rotated by 180 degrees in place: right -> left
    icons = read_minigame(img, S)
    x, y = icons[1]["x"], icons[1]["y"]
    r = 30
    patch = img[y - r:y + r, x - r:x + r].copy()
    img2 = img.copy()
    img2[y - r:y + r, x - r:x + r] = cv2.rotate(patch, cv2.ROTATE_180)
    assert seq(img2) == ["down", "left", "up", "right"], seq(img2)
    # rotated 90 degrees: right -> down
    img3 = img.copy()
    img3[y - r:y + r, x - r:x + r] = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    assert seq(img3)[1] == "down", seq(img3)
    print("  mirrored frame: ← ↑ ← ↓; rotated icon: → becomes ← / ↓")


def test_solver():
    if len(GAME) < 62:
        print("  (no fixtures, skipped)")
        return
    T.OVERRIDES.clear()
    T.OVERRIDES.update({"bot.map": "cloud_balcony", "rune_curse.enable": True, "rune_curse.check_interval": 0.0,
                        "rune_curse.capture_frames": False})
    bot = T.make_bot("wingpu_maple")
    kb = get_keyboard()
    kb.configure(bot.cfg, live=False)
    n0 = len(kb.history)
    # the game as the player saw it: each press makes the first icon fade
    frames = [GAME[-8], GAME[-8], GAME[-5], GAME[-5], GAME[-4], GAME[-4], GAME[-3], GAME[-3]]
    mains = []
    for f in frames:
        bot.capture.set_guest(cv2.imread(f))
        bot.t_player_seen = time.time()
        bot.t_mg_check = 0.0
        bot.dec = {"why": []}
        bot.run_once()
        bot.record_decision(0)
        mains.append(bot.now_state["main"])
        time.sleep(0.5)
    keys = [t.split()[1] for _, t in list(kb.history)[n0:] if t.startswith("press Arrow")]
    print(f"  solver pressed (dry run): {keys}")
    print(f"  status e.g. 「{mains[2]}」")
    assert keys == ["ArrowDown", "ArrowRight", "ArrowUp", "ArrowRight"], keys
    assert bot.rune["phase"] == "game"


def test_no_progress_gives_up():
    if len(GAME) < 62:
        print("  (no fixtures, skipped)")
        return
    T.OVERRIDES.clear()
    T.OVERRIDES.update({"bot.map": "cloud_balcony", "rune_curse.enable": True, "rune_curse.check_interval": 0.0,
                        "rune_curse.capture_frames": False, "rune_curse.minigame_retries": 1})
    bot = T.make_bot("wingpu_maple")
    kb = get_keyboard()
    n0 = len(kb.history)
    t_end = time.time() + 4.0
    while time.time() < t_end and bot.rune.get("phase") != "rest":
        bot.capture.set_guest(cv2.imread(GAME[-8]))      # the icons never fade
        bot.t_player_seen = time.time()
        bot.t_mg_check = 0.0
        bot.dec = {"why": []}
        bot.run_once()
        time.sleep(0.3)
    keys = [t.split()[1] for _, t in list(kb.history)[n0:] if t.startswith("press Arrow")]
    print(f"  icons never fade: pressed {keys}, then phase {bot.rune.get('phase')} ({bot.rune.get('why')})")
    assert keys == ["ArrowDown", "ArrowDown"] and bot.rune["phase"] == "rest"


if __name__ == "__main__":
    for name in ("test_read_real", "test_left_synthetic", "test_solver", "test_no_progress_gives_up"):
        globals()[name]()
        print("ok  ", name)
