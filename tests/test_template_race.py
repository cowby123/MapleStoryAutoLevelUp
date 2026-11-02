'''
build_templates() from another thread (the web UI saving settings) while the
bot loop matches monsters: no KeyError (2026-11-02: ('pig', 6) in the loop).

  python tests/test_template_race.py
'''
import glob
import logging
import os
import sys
import threading
import time

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import cv2  # noqa: E402
import wingpu_detect_test as T  # noqa: E402

T.logger.set_level(logging.ERROR)


def test_rebuild_while_matching():
    frames = sorted(glob.glob("fixtures/maple2/*.png"))
    if not frames:
        print("  (no fixtures, skipped)")
        return
    T.OVERRIDES.clear()
    bot = T.make_bot("wingpu_maple")
    bot.capture.set_guest(cv2.imread(frames[4]))
    bot.run_once()
    stop = threading.Event()
    rebuilds = [0]

    def rebuild():
        while not stop.is_set():
            bot.build_templates(bot.scale)
            rebuilds[0] += 1

    th = threading.Thread(target=rebuild, daemon=True)
    th.start()
    n, t_end = 0, time.time() + 4
    H, W = bot.img_frame.shape[:2]
    try:
        while time.time() < t_end:
            bot.get_monsters_in_range((0, 200), (W, 650))
            n += 1
    finally:
        stop.set()
        th.join()
    print(f"  {n} monster searches during {rebuilds[0]} template rebuilds: no error")
    assert n > 10 and rebuilds[0] > 10


if __name__ == "__main__":
    test_rebuild_while_matching()
    print("ok   test_rebuild_while_matching")
