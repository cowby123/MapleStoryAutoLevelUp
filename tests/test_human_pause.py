'''
類人保護 (human_pause): scheduler ranges and randomness, hunting-time only,
deferral during risky sequences, the bot holds every key but potions still
fire, and the stuck / no-attack watchdogs do not trip.

  python tests/test_human_pause.py
'''
import logging
import os
import random
import sys
import threading
import time

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import wingpu_detect_test as T  # noqa: E402
from src.engine.HumanPause import HumanPause  # noqa: E402
from src.engine import HealthMonitor as HMmod  # noqa: E402

T.logger.set_level(logging.ERROR)
CFG = {"enable": True, "interval_minutes": [3, 7], "pause_seconds": [3, 5],
       "long_enable": False, "long_chance_percent": 10, "long_seconds": [8, 15]}


def run_until_pause(hp, t, step=0.5, limit=3600):
    '''tick every step seconds (hunting) until a pause starts; returns its start time'''
    end = t + limit
    while t < end:
        if hp.tick(now=t) > 0:
            return t
        t += step
    raise AssertionError("no pause")


def test_ranges_and_randomness():
    hp = HumanPause(CFG, rng=random.Random(1), now=0.0)
    t, gaps, durs = 0.0, [], []
    last_end = 0.0
    for _ in range(200):
        start = run_until_pause(hp, t)
        gaps.append(start - last_end)
        durs.append(hp.dur)
        t = start
        while hp.tick(now=t) > 0:
            t += 0.5
        last_end = t
    assert all(3 * 60 - 1 <= g <= 7 * 60 + 1 for g in gaps), (min(gaps), max(gaps))
    assert all(3 <= d <= 5 for d in durs), (min(durs), max(durs))
    assert len(set(round(g) for g in gaps)) > 50 and len(set(round(d, 1) for d in durs)) > 10   # random
    print(f"  intervals {min(gaps) / 60:.1f}-{max(gaps) / 60:.1f} min, pauses {min(durs):.1f}-{max(durs):.1f} s")


def test_long_pauses():
    cfg = dict(CFG, long_enable=True)
    hp = HumanPause(cfg, rng=random.Random(2), now=0.0)
    longs = [hp.duration_s() for _ in range(5000)]
    n_long = sum(d >= 8 for d in longs)
    assert all(3 <= d <= 5 or 8 <= d <= 15 for d in longs)
    assert 0.07 < n_long / 5000 < 0.13, n_long
    hp = HumanPause(CFG, rng=random.Random(2), now=0.0)   # off: never long
    assert all(d <= 5 for d in (hp.duration_s() for _ in range(2000)))
    print(f"  long pauses {100 * n_long / 5000:.1f}%")


def test_only_hunting_time_counts():
    hp = HumanPause(CFG, rng=random.Random(3), now=0.0)
    due = hp.next_at
    hp.tick(now=10.0)
    # 30 minutes resting (no ticks), then hunting again: not due yet
    hp.tick(now=10.0 + 1800)
    assert hp.next_at > due + 1700 and hp.until is None


def test_deferral():
    hp = HumanPause(CFG, rng=random.Random(4), now=0.0)
    t = hp.next_at
    assert hp.tick(risky=True, now=t) == 0 and hp.deferred
    assert hp.tick(risky=True, now=t + 0.5) == 0
    assert hp.tick(risky=False, now=t + 1.0) > 0 and not hp.deferred


def test_disabled():
    hp = HumanPause(dict(CFG, enable=False), now=0.0)
    assert all(hp.tick(now=t) == 0 for t in range(0, 4000, 1))


def make_bot():
    T.OVERRIDES.clear()
    T.OVERRIDES["bot.map"] = "gon_pig"
    T.OVERRIDES["human_pause.enable"] = True
    T.OVERRIDES["human_pause.interval_minutes"] = [3, 7]
    bot = T.make_bot("wingpu_maple")
    T.OVERRIDES.clear()
    bot.human_pause = HumanPause(bot.cfg["human_pause"], rng=random.Random(5), now=0.0)
    bot.is_on_ladder = False
    bot.cmd_move_x = bot.cmd_move_y = bot.cmd_action = "none"
    bot.loc_player_global = (100, 50)
    bot.dec = {"why": []}
    return bot


def test_bot_pause_and_watchdogs():
    bot = make_bot()
    hp = bot.human_pause
    t = hp.next_at
    bot.t_last_attack = t - 2.0
    # on a rope: deferred
    bot.is_on_ladder = True
    assert bot.run_human_pause(now=t) is False and hp.deferred
    bot.is_on_ladder = False
    bot.runup_state = {"phase": "back"}
    assert bot.run_human_pause(now=t + 0.1) is False
    bot.runup_state = None
    # starts: every key released, nothing sent, buffs / attacks blocked
    assert bot.run_human_pause(now=t + 0.2) is True
    assert bot.kb.is_resting and bot.kb.cmd == "none none none"
    dur = hp.dur
    # the hunt cycle (off) must not undo the hold
    bot.run_hunt_cycle()
    assert bot.kb.is_resting
    # during (frames every 0.25 s): the stuck watchdog follows the clock
    now = t + 0.2
    while True:
        now += 0.25
        if not bot.run_human_pause(now=now):
            break
        assert bot.t_watch_dog == now and bot.kb.is_resting
    # ends: keys back, the no-attack watchdog skips the pause
    assert t + 0.2 + dur <= now <= t + 0.2 + dur + 0.25
    assert not bot.kb.is_resting
    assert abs(bot.t_last_attack - (t - 2.0 + dur)) < 1e-6
    assert bot.t_watch_dog == now
    print(f"  paused {dur:.1f} s; t_last_attack moved by the pause")


def test_curse_cancels():
    bot = make_bot()
    t = bot.human_pause.next_at
    assert bot.run_human_pause(now=t) is True
    bot.cancel_human_pause("curse")
    assert not bot.kb.is_resting and not bot.human_pausing and bot.human_pause.until is None


def test_potions_while_resting():
    '''HealthMonitor drinks while kb.is_resting (normal and force_heal)'''
    for force in (False, True):
        cfg = {"health_monitor": {"fps_limit": 50, "add_hp_percent": 50, "add_mp_percent": 50,
                                  "add_hp_cooldown": 0.1, "add_mp_cooldown": 0.1, "force_heal": force,
                                  "return_home_if_no_potion": False, "return_home_watch_dog_timeout": 5},
               "key": {"add_hp": "delete", "add_mp": "end", "return_home": "home"}}
        kb = T.FakeKB()
        kb.is_resting = True
        hm = HMmod.HealthMonitor(cfg, kb)
        hm.get_hp_mp_exp_percent = lambda: (20.0, 20.0, 50.0)
        pressed = []
        orig = HMmod.press_key
        HMmod.press_key = lambda k, d=0.05: pressed.append(k)
        try:
            th = threading.Thread(target=hm._monitor_loop, daemon=True)
            th.start()
            time.sleep(0.4)
            hm.is_terminated = True
            th.join(2)
        finally:
            HMmod.press_key = orig
        assert "delete" in pressed and "end" in pressed, (force, pressed)
    print("  potions fire while resting (force_heal off and on)")


if __name__ == "__main__":
    for t in (test_ranges_and_randomness, test_long_pauses, test_only_hunting_time_counts, test_deferral,
              test_disabled, test_bot_pause_and_watchdogs, test_curse_cancels, test_potions_while_resting):
        t()
        print(f"{t.__name__}: ok")
