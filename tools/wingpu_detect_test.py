'''
wingpu_detect_test.py - run every detector of the bot over saved game frames
(offline: no VM, no input sent; WingpuKeyboard stays in dry run).

  python tools/wingpu_detect_test.py --frames "fixtures/maple/*.png" [--cfg wingpu_maple]
         [--scale auto|height|geometric|<number>] [--out out_dir] [--limit N]
  python tools/wingpu_detect_test.py --sweep 1.00,1.03,1.054,1.07,1.10 --frames ...

Each frame is a whole guest screen (what /wingpu/frame gives; tools/wingpu_grab.py
saves them). Every frame runs through MapleStoryAutoBot.run_once() as one
standalone frame (detection-lost timers reset), so a frame where the minimap
or the player is missing shows up as "paused".

Prints per frame: minimap rect, player dot (route px), global position + score,
party red bar / player position, monsters, HP/MP/EXP, rune flags, the command
the bot would give, run_once time. Then hit rates and CPU time per frame.
--out writes the annotated debug image of every frame.
'''
# Standard import
import argparse
import glob
import os
import sys
import time

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Library import
import cv2  # noqa: E402

# Local import
from src.utils.logger import logger  # noqa: E402
from src.utils.config_layers import build_cfg  # noqa: E402
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot  # noqa: E402
from src.engine.HealthMonitor import HealthMonitor  # noqa: E402
from src.engine.Profiler import Profiler  # noqa: E402
from src.engine.RuneSolver import RuneSolver  # noqa: E402
from src.input.GameWindowCapturorForWingpu import detect_client_rect  # noqa: E402
from src.input.WingpuKeyboard import get_keyboard  # noqa: E402


class FakeCapture:
    window_title = "MapleStory Worlds"

    def __init__(self, cfg):
        self.cfg = cfg
        self.frame = None
        self.rect = None

    def set_guest(self, img):
        r = detect_client_rect(img, self.cfg)
        self.rect = r
        self.frame = img[r.y:r.y + r.h, r.x:r.x + r.w]

    def get_frame(self):
        return self.frame

    def window_to_guest(self, x, y):
        return self.rect.window_to_guest(x, y)


class FakeKB:
    '''
    Records commands; sends nothing
    '''
    def __init__(self):
        self.is_enable = True
        self.is_hold = False
        self.is_terminated = False
        self.is_need_force_heal = False
        self.t_last_screenshot = 0.0
        self.cmd = "none none none"
        self.cmd_left_right = self.cmd_up_down = self.cmd_action = "none"

    def set_command(self, c):
        self.cmd = c
        self.cmd_left_right, self.cmd_up_down, self.cmd_action = c.split()

    def release_all_key(self):
        pass

    def hold(self, on):
        self.is_hold = bool(on)

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False


class Args:
    disable_viz = False
    disable_control = True
    is_ui = False
    test_image = ""
    init_state = ""
    wingpu_live = False


def make_bot(cfg_name, scale=None):
    cfg = build_cfg(cfg_name, user_override=None)
    if scale is not None:
        cfg.setdefault("wingpu", {})["scale"] = scale
    cfg["profiler"]["enable"] = True
    bot = MapleStoryAutoBot(Args())
    if bot.load_config(cfg) != 0:
        raise SystemExit("load_config failed")
    get_keyboard().configure(bot.cfg, live=False)  # dry run, always
    bot.capture = FakeCapture(bot.cfg)
    bot.kb = FakeKB()
    bot.health_monitor = HealthMonitor(bot.cfg, bot.kb)
    bot.profiler = Profiler(bot.cfg)
    bot.rune_solver = RuneSolver(bot.cfg)
    bot.fsm.set_init_state("hunting")
    return bot


def run(frames, cfg_name, scale=None, out=None, quiet=False):
    bot = make_bot(cfg_name, scale)
    if out:
        os.makedirs(out, exist_ok=True)
    hits = {"minimap": 0, "player_dot": 0, "global_ok": 0, "player": 0, "monsters": 0,
            "hp_bars": 0, "paused": 0}
    cpu = []
    wall = []
    for path in frames:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        bot.capture.set_guest(img)
        # standalone frame: forget the previous frame's tracking
        bot.t_minimap_ok = bot.t_player_seen = 0.0
        bot.minimap_rect = None
        bot.det_counts = {k: 0 for k in bot.det_counts}
        bot.kb.set_command("none none none")
        c0, w0 = time.process_time(), time.perf_counter()
        bot.run_once()
        cpu.append(time.process_time() - c0)
        wall.append(time.perf_counter() - w0)
        hp = bot.health_monitor.get_hp_mp_exp_percent()
        d = bot.det_counts
        mm_ok = d["minimap"] > 0
        dot = d["player_dot"] > 0
        gscore = getattr(bot, "minimap_global_score", None)
        global_ok = dot and gscore is not None and gscore < 0.2
        player = d["player"] > 0
        hits["minimap"] += mm_ok
        hits["player_dot"] += dot
        hits["global_ok"] += global_ok
        hits["player"] += player
        hits["monsters"] += len(bot.monsters) > 0
        hits["hp_bars"] += hp[0] is not None
        hits["paused"] += bot.is_lost_paused
        if not quiet:
            mons = [(m["name"][:6], tuple(int(v) for v in m["position"])) for m in bot.monsters]
            print(f"{os.path.basename(path)[:28]:28s} "
                  f"mm={bot.minimap_rect if mm_ok else None} "
                  f"m={bot.minimap_scale.m if bot.minimap_scale and bot.minimap_scale.m else None} "
                  f"dot={bot.loc_player_minimap if dot else None} "
                  f"glob={bot.loc_player_global}/{round(gscore, 3) if gscore is not None else None} "
                  f"player={tuple(int(v) for v in bot.loc_player) if player else None} "
                  f"mobs={len(bot.monsters)}{mons[:3]} "
                  f"hp/mp/exp={[round(v, 1) if v is not None else None for v in hp]} "
                  f"state={bot.fsm.state.name} "
                  f"cmd='{bot.kb.cmd}'{' PAUSED' if bot.is_lost_paused else ''} "
                  f"{wall[-1] * 1000:.0f}ms")
        if out and bot.img_frame_debug is not None:
            cv2.imwrite(os.path.join(out, os.path.basename(path).rsplit(".", 1)[0] + "_dbg.jpg"),
                        bot.img_frame_debug, [cv2.IMWRITE_JPEG_QUALITY, 85])
    n = max(1, len(cpu))
    summary = {k: f"{v}/{len(cpu)} ({100 * v / n:.0f}%)" for k, v in hits.items()}
    return bot, summary, cpu, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="fixtures/maple/*.png")
    ap.add_argument("--cfg", default="wingpu_maple")
    ap.add_argument("--scale", default=None, help="wingpu.scale override")
    ap.add_argument("--sweep", default="", help="comma separated scales: hit rates per scale")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    frames = sorted(glob.glob(args.frames))
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit(f"no frames match {args.frames}")
    import logging
    logger.set_level(logging.WARNING)

    def parse_scale(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    if args.sweep:
        for v in args.sweep.split(","):
            bot, summary, cpu, wall = run(frames, args.cfg, parse_scale(v), quiet=True)
            print(f"scale {v:>8s} (= {bot.scale:.4f}): {summary}")
        return
    bot, summary, cpu, wall = run(frames, args.cfg, parse_scale(args.scale), args.out or None)
    print()
    print(f"frames {len(cpu)}  frame {bot.frame_hw[1]}x{bot.frame_hw[0]}  ui scale {bot.scale:.4f}  "
          f"minimap scale {bot.minimap_scale.m if bot.minimap_scale else None}")
    for k, v in summary.items():
        print(f"  {k:10s} {v}")
    print(f"run_once: cpu {1000 * sum(cpu) / len(cpu):.1f} ms/frame, wall {1000 * sum(wall) / len(wall):.1f} ms/frame")
    print(bot.profiler.report())
    print(f"input (dry run, recorded only): {get_keyboard().status()['counts']}")


if __name__ == "__main__":
    main()
