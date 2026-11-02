'''
Unit tests of the hunt/rest scheduler (src/engine/HuntCycle.py).

  python tests/test_hunt_cycle.py        (or: python -m pytest tests/test_hunt_cycle.py)

A fake clock drives the scheduler; short cycles (30 s hunt / 20 s rest).
'''
import datetime
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.engine.HuntCycle import HuntCycle, in_window, _range  # noqa: E402


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def run(self, hc, seconds, step=1.0, **kw):
        '''advance in steps, return the list of (time offset, phase) changes'''
        out = []
        t0 = self.t
        last = hc.phase
        end = self.t + seconds
        while self.t < end:
            self.t += step
            ph = hc.tick(**kw)
            if ph != last:
                out.append((round(self.t - t0), ph))
                last = ph
        return out


def local(h, m):
    return lambda: datetime.datetime(2026, 11, 2, h, m)


def make(cfg, clock, now_local=None, seed=1):
    base = {"enable": True, "hunt_minutes": [0.5, 0.5], "rest_minutes": [1 / 3, 1 / 3]}
    base.update(cfg)
    return HuntCycle(base, clock=clock, rng=random.Random(seed), now_local=now_local or local(12, 0))


def test_disabled():
    c = Clock()
    hc = HuntCycle({"enable": False}, clock=c)
    assert hc.phase == "off" and hc.hunting
    assert c.run(hc, 3600) == []


def test_ranges():
    assert _range([30, 40], None) == (30.0, 40.0)
    assert _range(5, None) == (5.0, 5.0)
    assert _range([40, 30], None) == (40.0, 40.0)     # max below min -> min
    assert _range(None, (1, 2)) == (1.0, 2.0)
    c = Clock()
    hc = make({"hunt_minutes": [1, 3], "rest_minutes": [2, 4]}, c, seed=7)
    hunts, rests = [], []
    for _ in range(200):
        if hc.phase == "hunt":
            hunts.append(hc.t_end - c.t)
        elif hc.phase == "rest":
            rests.append(hc.t_end - c.t)
        c.t = hc.t_end
        hc.tick()
    assert hunts and rests
    assert all(60 <= h <= 180 for h in hunts), (min(hunts), max(hunts))
    assert all(120 <= r <= 240 for r in rests), (min(rests), max(rests))
    assert max(hunts) - min(hunts) > 60     # really random over the range


def test_hunt_rest_cycle_30_20():
    c = Clock()
    hc = make({}, c)
    assert hc.phase == "hunt" and hc.cycle == 1
    changes = c.run(hc, 120)
    assert changes == [(30, "rest"), (50, "hunt"), (80, "rest"), (100, "hunt")], changes
    assert hc.cycle == 3


def test_safe_spot_reached_and_timeout():
    c = Clock()
    hc = make({"rest_action": "safe_spot", "safe_spot_timeout": 15}, c)
    assert c.run(hc, 30) == [(30, "to_safe")]
    # reaches the spot after 5 s -> rest 20 s -> hunt
    assert c.run(hc, 5, at_safe_spot=False) == []
    assert c.run(hc, 1, at_safe_spot=True) == [(1, "rest")]
    assert not hc.fell_back
    assert c.run(hc, 25) == [(20, "hunt")]
    # next time it never gets there (5 s of this hunt already passed):
    # to_safe at 25, rest in place after the 15 s timeout, hunt 20 s later
    changes = c.run(hc, 80, at_safe_spot=False)
    assert changes == [(25, "to_safe"), (40, "rest"), (60, "hunt")], changes
    assert any("沒走到安全點" in e for _, e in hc.events)


def test_go_home_ends_session():
    c = Clock()
    hc = make({"rest_action": "go_home"}, c)
    assert c.run(hc, 31) == [(30, "ended")]
    assert hc.take_go_home() and not hc.take_go_home()   # pressed once
    assert c.run(hc, 600) == []                           # stays ended


def test_session_limit():
    c = Clock()
    hc = make({"session_hours": 100 / 3600}, c)          # 100 s
    changes = c.run(hc, 200)
    assert changes[-1] == (100, "ended"), changes
    assert "已玩" in hc.ended_why


def test_window():
    assert in_window(9 * 60, 17 * 60, 12 * 60)
    assert not in_window(9 * 60, 17 * 60, 8 * 60 + 59)
    assert not in_window(9 * 60, 17 * 60, 17 * 60)       # end exclusive
    assert in_window(22 * 60, 2 * 60, 23 * 60)           # across midnight
    assert in_window(22 * 60, 2 * 60, 60)
    assert not in_window(22 * 60, 2 * 60, 12 * 60)
    assert in_window(None, None, 5)                      # no window
    # outside at start -> wait; window opens -> hunt; closes -> wait again
    c = Clock()
    now = {"v": datetime.datetime(2026, 11, 2, 8, 59)}
    hc = make({"window_start": "09:00", "window_end": "09:02", "hunt_minutes": [10, 10]}, c,
              now_local=lambda: now["v"])
    assert hc.phase == "wait_window"
    now["v"] = datetime.datetime(2026, 11, 2, 9, 0)
    assert hc.tick() == "hunt"
    now["v"] = datetime.datetime(2026, 11, 2, 9, 2)
    assert hc.tick() == "wait_window"


def test_reconfigure():
    c = Clock()
    hc = make({}, c)
    c.t += 10                                            # 10 s into a 30 s hunt
    hc.reconfigure({"enable": True, "hunt_minutes": [1, 1], "rest_minutes": [1 / 3, 1 / 3]})
    assert hc.phase == "hunt" and hc.status()["remaining_s"] == 50.0   # 60 s from the phase start
    hc.reconfigure({"enable": True, "hunt_minutes": [0.1, 0.1], "rest_minutes": [1 / 3, 1 / 3]})
    assert hc.tick() == "rest"                           # 6 s phase already over -> next tick
    hc.reconfigure({"enable": False})
    assert hc.phase == "off" and hc.hunting
    hc.reconfigure({"enable": True, "hunt_minutes": [0.5, 0.5], "rest_minutes": [1 / 3, 1 / 3]})
    assert hc.phase == "hunt" and hc.cycle == 1
    hc2 = make({"rest_action": "safe_spot", "safe_spot_timeout": 100}, c)
    c.t += 30
    assert hc2.tick() == "to_safe"
    hc2.fail_safe_spot("沒有安全點")
    assert hc2.phase == "rest" and hc2.fell_back


def test_status():
    c = Clock()
    hc = make({}, c)
    c.t += 10
    st = hc.status()
    assert st["phase"] == "hunt" and st["phase_cn"] == "練功"
    assert st["remaining_s"] == 20.0 and st["cycle"] == 1


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
            n += 1
    print(f"{n} tests passed")
