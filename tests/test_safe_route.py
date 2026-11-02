'''
Walking to the safe spot (hunt_cycle rest_action "safe_spot"): the command the
bot gives at several minimap positions, with and without a safe route.

  python tests/test_safe_route.py

Uses the real gon_pig map/routes and the bot's own route lookup
(MapleStoryAutoBot.plan_to_safe -> update_cmd_by_route on route_safe.png).
No VM, no input (dry run).
'''
import logging
import os
import shutil
import sys
import tempfile

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import wingpu_detect_test as T  # noqa: E402

T.logger.set_level(logging.WARNING)

LEFT, RIGHT = (255, 0, 0), (0, 0, 255)           # RGB, route.color_code
DOWN_JUMP, STOP = (127, 255, 0), (0, 255, 127)
SPOT = (60, 96)                                  # route-map pixels, on the ground row


def make_bot(routes_dir):
    T.OVERRIDES.clear()
    T.OVERRIDES.update({
        "hunt_cycle.routes_dir": routes_dir,
        "hunt_cycle.safe_spots": {"gon_pig": list(SPOT)},
    })
    bot = T.make_bot("wingpu_maple")
    bot.img_route = bot.img_routes[0]
    bot.img_frame_debug = np.zeros((768, 1366, 3), np.uint8)
    return bot


def paint_route(routes_dir, h, w):
    '''
    A safe route on gon_pig: along the ground (y 96) everything leads left to
    the spot; under the tree platform (x 95-105, y 78) jump down.
    '''
    img = np.zeros((h, w, 3), np.uint8)          # RGB
    img[96, SPOT[0] + 4:200] = LEFT
    img[96, 20:SPOT[0] - 3] = RIGHT
    img[96, SPOT[0] - 2:SPOT[0] + 3] = STOP
    img[78, 95:106] = DOWN_JUMP
    d = os.path.join(routes_dir, "gon_pig")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "route_safe.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def plan(bot, pos):
    bot.loc_player_global = pos
    return bot.plan_to_safe(bot.safe_spot())


def test_without_safe_route():
    d = tempfile.mkdtemp()
    try:
        bot = make_bot(d)
        assert bot.load_route_safe() is None
        assert plan(bot, (150, 96)) == "left none none"     # same level: walk
        assert plan(bot, (30, 97)) == "right none none"
        assert plan(bot, (61, 96)) == "none none none" and bot.at_safe_spot(SPOT)
        assert plan(bot, (101, 78)) is None                   # other level: no route -> rest in place
        assert bot.safe_plan["how"] is None and bot.safe_plan["dy"] == 18
    finally:
        shutil.rmtree(d)


def test_with_safe_route():
    d = tempfile.mkdtemp()
    try:
        bot = make_bot(d)
        paint_route(d, *bot.img_map.shape[:2])
        assert bot.load_route_safe() is not None
        cases = {
            (150, 96): "left",            # on the painted ground line
            (120, 98): "left",            # 2 px below it (within route.search_range)
            (30, 96): "right",            # left of the spot: comes back right
            (101, 78): "none down jump",  # on the tree platform: jump down
        }
        for pos, want in cases.items():
            cmd = plan(bot, pos)
            got = cmd if " " in want else cmd.split()[0]
            assert got == want, (pos, cmd, bot.safe_plan)
            assert bot.safe_plan["how"] == "route_safe"
        cmd = plan(bot, SPOT)
        assert cmd.split()[0] in ("none", "stop") and bot.at_safe_spot(SPOT), cmd
        # the hunting route is untouched after planning
        assert bot.img_route is bot.img_routes[0]
    finally:
        shutil.rmtree(d)


UP, ROPE_JUMP = (127, 127, 127), (0, 127, 255)
ROPE_X, GROUND_Y, TOP_Y = 150, 96, 58
TOP_SPOT = (172, TOP_Y)


def paint_rope_route(routes_dir, h, w):
    '''
    Safe spot on a platform above a rope that is out of reach from the ground:
      ground (y 96): walk right to the rope        -> blue  "right none none"
      under the rope (x 150, y 94-96)              -> azure "none up jump" (hold up + jump)
      the rope (x 150, y 60-93)                    -> grey  "none up none" (climb)
      the platform on top (y 58, x 151-169)        -> blue  "right none none"
      the spot (x 170-174, y 58)                   -> light green "stop"
    '''
    img = np.zeros((h, w, 3), np.uint8)
    img[GROUND_Y, 100:ROPE_X - 1] = RIGHT
    img[GROUND_Y - 2:GROUND_Y + 1, ROPE_X] = ROPE_JUMP
    img[TOP_Y + 2:GROUND_Y - 2, ROPE_X] = UP
    img[TOP_Y, ROPE_X + 1:TOP_SPOT[0] - 2] = RIGHT
    img[TOP_Y, TOP_SPOT[0] - 2:TOP_SPOT[0] + 3] = STOP
    d = os.path.join(routes_dir, "gon_pig")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "route_safe.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def test_rope_climb():
    d = tempfile.mkdtemp()
    try:
        T.OVERRIDES.clear()
        T.OVERRIDES.update({"hunt_cycle.routes_dir": d,
                            "hunt_cycle.safe_spots": {"gon_pig": list(TOP_SPOT)}})
        bot = T.make_bot("wingpu_maple")
        bot.img_route = bot.img_routes[0]
        bot.img_frame_debug = np.zeros((768, 1366, 3), np.uint8)
        paint_rope_route(d, *bot.img_map.shape[:2])
        steps = [
            # (minimap position, on the rope?, expected command)
            ((120, GROUND_Y), False, "right none none"),     # walking to the rope
            ((ROPE_X, GROUND_Y), False, "none up jump"),     # under it: jump onto it holding up
            ((ROPE_X, 85), True, "none up none"),            # climbing
            ((ROPE_X, 70), True, "none up none"),
            ((ROPE_X, 61), True, "right up none"),           # near the top: keep up, lean right (ladder complement)
            ((160, TOP_Y), False, "right none none"),        # on the platform
        ]
        for pos, on_rope, want in steps:
            bot.is_on_ladder = on_rope
            cmd = plan(bot, pos)
            print(f"  {pos} {'on rope ' if on_rope else ''}-> '{cmd}'")
            assert cmd == want, (pos, cmd, want)
        cmd = plan(bot, TOP_SPOT)
        assert bot.at_safe_spot(TOP_SPOT) and cmd.split()[0] in ("stop", "none"), cmd
        # the keyboard controller turns "none up jump" into: up held + jump pressed
        from src.input.KeyBoardController import KeyBoardController
        from src.input.WingpuKeyboard import get_keyboard
        kb = get_keyboard()
        kb.configure(bot.cfg, live=False)
        n0 = len(kb.history)
        kbc = KeyBoardController(bot.cfg)
        kbc.set_command("none up jump")
        import time
        time.sleep(0.25)
        kbc.is_terminated = True
        time.sleep(0.1)
        acts = [t for _, t in list(kb.history)[n0:]]
        assert any(a.startswith("key_down ArrowUp") for a in acts), acts
        assert any(a.startswith("press AltLeft") for a in acts), acts
        print(f"  keyboard (dry run): {acts[:3]} ...")
    finally:
        shutil.rmtree(d)


def test_fixture_positions():
    '''
    Player positions measured on the saved maple frames (fixtures/maple2, route px)
    '''
    d = tempfile.mkdtemp()
    try:
        bot = make_bot(d)
        paint_route(d, *bot.img_map.shape[:2])
        frames = sorted(T.glob.glob("fixtures/maple2/*.png"))[:20:4]
        if not frames:
            print("  (no fixtures, skipped)")
            return
        for f in frames:
            bot.capture.set_guest(cv2.imread(f))
            bot.t_minimap_ok = bot.t_player_seen = 0
            bot.minimap_rect = None
            bot.run_once()
            pos = tuple(bot.loc_player_global)
            cmd = plan(bot, pos)
            print(f"  {os.path.basename(f)[:24]} player {pos} -> '{cmd}' ({bot.safe_plan['how']}, dx {bot.safe_plan['dx']})")
            assert cmd is not None and cmd.split()[0] in ("left", "none", "stop"), cmd
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
            n += 1
    print(f"{n} tests passed")
