'''
monster_detect.mask_erode / color_prefilter_thres per monster: the bears get
eroded masks and a looser prefilter, pigs keep the plain ones; and the windup
bears are found on the hand-labelled cloud_balcony frames (fixtures/cloud).

  python tests/test_mob_per_monster.py
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
import wingpu_mob_eval as E  # noqa: E402

T.logger.set_level(logging.ERROR)


class FakeBot:
    md_per_monster = T.MapleStoryAutoBot.md_per_monster

    def __init__(self, md):
        self.cfg = {"monster_detect": md}


def test_values():
    b = FakeBot({"mask_erode": {"pink_windup_bear": 1}, "color_prefilter_thres": {"default": 0.16, "x": 0.2}})
    assert b.md_per_monster("mask_erode", "pink_windup_bear", 0) == 1
    assert b.md_per_monster("mask_erode", "pig", 0) == 0
    assert b.md_per_monster("color_prefilter_thres", "x", 0.16) == 0.2
    assert b.md_per_monster("color_prefilter_thres", "pig", 0.1) == 0.16
    b = FakeBot({"mask_erode": 2, "color_prefilter_thres": 0.18})
    assert b.md_per_monster("mask_erode", "pig", 0) == 2
    assert b.md_per_monster("color_prefilter_thres", "pig", 0.16) == 0.18
    b = FakeBot({})
    assert b.md_per_monster("mask_erode", "pig", 0) == 0


def test_masks():
    T.OVERRIDES.clear()
    T.OVERRIDES["bot.map"] = "cloud_balcony"
    bot = T.make_bot("wingpu_maple")
    bot.on_resolution(768, 1366)
    T.OVERRIDES["monster_detect.mask_erode"] = 0
    plain = T.make_bot("wingpu_maple")
    plain.on_resolution(768, 1366)
    T.OVERRIDES.clear()
    for key, (_, mk) in bot.monster_color.items():
        # eroded: clearly fewer mask pixels than the plain mask
        n, n0 = np.count_nonzero(mk), np.count_nonzero(plain.monster_color[key][1])
        assert n < 0.9 * n0, (key, n, n0)
    # pigs are not listed: plain masks
    T.OVERRIDES["bot.map"] = "gon_pig"
    pig = T.make_bot("wingpu_maple")
    pig.on_resolution(768, 1366)
    T.OVERRIDES["monster_detect.mask_erode"] = 0
    pig0 = T.make_bot("wingpu_maple")
    pig0.on_resolution(768, 1366)
    T.OVERRIDES.clear()
    for key, (_, mk) in pig.monster_color.items():
        assert np.array_equal(mk, pig0.monster_color[key][1]), key


def test_bears_found():
    path = "fixtures/cloud/labels_all.json"
    if not os.path.exists(path):
        print("  (no fixtures, skipped)")
        return
    T.OVERRIDES.clear()
    T.OVERRIDES["bot.map"] = "cloud_balcony"
    res = E.evaluate(json.load(open(path)), "fixtures/cloud", "wingpu_maple", 30, "full", repeat=1)
    T.OVERRIDES.clear()
    hit, n = (int(v) for v in res["recall"].split()[0].split("/"))
    print(f"  {res}")
    assert hit >= 0.88 * n and res["false_pos"] <= 2, res


if __name__ == "__main__":
    for t in (test_values, test_masks, test_bears_found):
        t()
        print(f"{t.__name__}: ok")
