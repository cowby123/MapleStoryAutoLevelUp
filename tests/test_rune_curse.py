'''
Rune curse: banner detection, the yellow guide arrow, the rune-finding mode
and the frame capture. Needs fixtures/rune/curse_*.png (7 frames with the
banner, maple 2026-11-03) and fixtures/maple*/ (91 without); skipped without.

  python tests/test_rune_curse.py
'''
import glob
import logging
import os
import shutil
import sys
import tempfile
import time

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import cv2  # noqa: E402
import wingpu_detect_test as T  # noqa: E402
from src.engine.RuneCurse import CurseDetector, find_guide_arrow  # noqa: E402
from src.input.WingpuKeyboard import get_keyboard  # noqa: E402

T.logger.set_level(logging.ERROR)
CURSE = sorted(glob.glob("fixtures/rune/curse_*.png"))
CLEAN = sorted(glob.glob("fixtures/maple/*.png")) + sorted(glob.glob("fixtures/maple2/*.png"))
S = 1366 / 1296


def test_banner():
    if not CURSE or not CLEAN:
        print("  (no fixtures, skipped)")
        return
    d = CurseDetector()
    hits = [d.detect(cv2.imread(f), S) for f in CURSE]
    fp = [f for f in CLEAN if d.detect(cv2.imread(f), S)]
    print(f"  banner: {sum(hits)}/{len(CURSE)} curse frames, {len(fp)}/{len(CLEAN)} false hits on clean frames")
    assert all(hits) and not fp
    # other resolutions: the template is rescaled (1920x1080 -> scale 1.48)
    big = cv2.resize(cv2.imread(CURSE[0]), (1920, 1080), interpolation=cv2.INTER_LINEAR)
    assert d.detect(big, 1920 / 1296), d.last_score
    print(f"  banner at 1920x1080 (resized frame): score {d.last_score:.3f}")


def test_arrow():
    if not CURSE:
        print("  (no fixtures, skipped)")
        return
    for f in CURSE:
        a = find_guide_arrow(cv2.imread(f), (856, 436), S)
        assert a is not None and -170 < a["angle"] < -110, (f, a)   # down-left
    a = find_guide_arrow(cv2.imread(CURSE[0]), (856, 436), S)
    print(f"  arrow: {len(CURSE)}/{len(CURSE)} found, e.g. centre {a['centre']}, angle {a['angle']} deg")
    fp = 0
    for f in CLEAN:
        im = cv2.imread(f)
        for p in [(683, 450), (400, 450), (950, 450), (683, 300)]:
            fp += find_guide_arrow(im, p, S) is not None
    print(f"  arrow false hits on {len(CLEAN)} clean frames x 4 player positions: {fp}")
    assert fp == 0


def test_mode():
    if not CURSE or not CLEAN:
        print("  (no fixtures, skipped)")
        return
    tmp = tempfile.mkdtemp()
    try:
        T.OVERRIDES.clear()
        T.OVERRIDES.update({"rune_curse.enable": True, "rune_curse.seek_timeout": 1.0,
                            "rune_curse.check_interval": 0.0, "rune_curse.capture_dir": tmp,
                            "rune_curse.capture_interval": 0.2, "rune_curse.capture_max": 5})
        bot = T.make_bot("wingpu_maple")
        kb = get_keyboard()
        kb.configure(bot.cfg, live=False)
        n0 = len(kb.history)

        def frame(path, erase_arrow=False):
            img = cv2.imread(path)
            if erase_arrow:
                img[480:560, 720:830] = img[380:460, 720:830]   # cover the arrow with scenery
            bot.capture.set_guest(img)
            bot.t_player_seen = time.time()
            bot.dec = {"why": []}
            bot.run_once()
            bot.record_decision(0)
            return bot.now_state["main"]

        main = frame(CURSE[0])
        assert bot.rune["phase"] == "seek" and not bot.is_lost_paused, bot.rune
        assert bot.kb.cmd == "left none none", bot.kb.cmd              # arrow down-left, mostly sideways
        print(f"  seek: kb '{bot.kb.cmd}', status 「{main}」")
        main = frame(CURSE[1], erase_arrow=True)                         # arrow gone ...
        time.sleep(0.35)
        main = frame(CURSE[1], erase_arrow=True)                         # ... for 0.3 s: press up
        presses = [t for _, t in list(kb.history)[n0:] if t.startswith("press ArrowUp")]
        assert presses, list(kb.history)[n0:]
        print(f"  arrow gone -> {presses[0]} (dry run); status 「{main}」")
        time.sleep(1.1)
        main = frame(CURSE[2])                                           # timeout, no safe spot
        main = frame(CURSE[2])
        assert bot.rune["phase"] == "rest" and bot.kb.is_resting, bot.rune
        print(f"  timeout -> {bot.rune['phase']} (no safe spot): 「{main}」")
        frame(CLEAN[45])                                                 # curse gone
        assert bot.rune["phase"] is None and not bot.kb.is_resting
        print("  curse gone -> back to hunting")
        files = [f for f in os.listdir(tmp) if f.endswith(".png")]
        assert 1 <= len(files) <= 5, files
        print(f"  frames saved while cursed: {len(files)} (cap 5)")
        # feature off: no rune mode, capture still works
        T.OVERRIDES["rune_curse.enable"] = False
        bot2 = T.make_bot("wingpu_maple")
        bot2.capture.set_guest(cv2.imread(CURSE[0]))
        bot2.dec = {"why": []}
        bot2.run_once()
        assert bot2.rune["phase"] is None and bot2.cursed
        print("  feature off: banner seen, nothing done (status shows it)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name in ("test_banner", "test_arrow", "test_mode"):
        globals()[name]()
        print("ok  ", name)
