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


def test_pig_pet_and_portal():
    '''gon_pig 2026-11-05: the pet rabbit (light name box under it) and the
    portal's light pillar are not pigs; the real pigs still are'''
    path = "fixtures/pig2/user_reported_010533.png"
    if not os.path.exists(path):
        print("  (no fixtures, skipped)")
        return
    import cv2
    T.OVERRIDES.clear()
    T.OVERRIDES["bot.map"] = "gon_pig"
    bot = T.make_bot("wingpu_maple")
    T.OVERRIDES.clear()
    bot.on_resolution(768, 1366)
    bot.img_frame = cv2.imread(path)
    assert bot.pet_tag_below({"position": (1058, 426), "size": (48, 71)})       # the rabbit
    assert not bot.pet_tag_below({"position": (788, 488), "size": (48, 71)})    # a pig
    for f, real in (("f20261105_010258_050.png", (1225, 349)), ):
        img = cv2.imread("fixtures/pig2/" + f)
        bot.img_frame, bot.img_frame_debug, bot.loc_player = img, img.copy(), (885, 449)
        mons = bot.get_monsters_in_range((500, 300), (1366, 600))
        assert not any(abs(m["position"][0] - real[0]) < 20 and abs(m["position"][1] - real[1]) < 20 for m in mons), mons
    img = cv2.imread(path)
    bot.img_frame, bot.img_frame_debug, bot.loc_player = img, img.copy(), (983, 452)
    mons = bot.get_monsters_in_range((0, 90), (1366, 673))
    assert any(abs(m["position"][0] - 788) < 10 for m in mons), mons      # the pig on the step


if __name__ == "__main__":
    for t in (test_values, test_masks, test_bears_found, test_pig_pet_and_portal):
        t()
        print(f"{t.__name__}: ok")
