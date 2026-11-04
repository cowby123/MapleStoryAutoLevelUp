'''
類人保護 (human_pause): every few minutes of hunting, at a random interval,
stop all keyboard actions for a few random seconds (potions still work).

The scheduler only counts hunting time: tick() is called on hunting frames;
a gap between ticks (resting, walking to the safe spot, paused bot, rune)
moves the next pause later by the same amount. A due pause waits while the
bot is in a risky sequence (rope, jump run-up, ...): tick(risky=True).
'''
import random
import time


def _range(v, default):
    try:
        a, b = float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError):
        a, b = default
    if a > b:
        a, b = b, a
    return a, b


class HumanPause:
    GAP_S = 2.0   # ticks further apart than this: the time in between does not count

    def __init__(self, cfg=None, rng=None, now=None):
        self.rng = rng or random.Random()
        self.cfg = {}
        self.next_at = None
        self.until = None
        self.dur = 0.0
        self.long = False
        self.deferred = False
        self.count = 0
        self.t_last_tick = None
        self.ended = False
        self.reconfigure(cfg or {}, now=now)

    @property
    def enabled(self):
        return bool(self.cfg.get("enable", False))

    def reconfigure(self, cfg, now=None):
        old = dict(self.cfg)
        self.cfg = dict(cfg or {})
        if self.cfg.get("interval_minutes") != old.get("interval_minutes") or self.next_at is None:
            self.schedule(now)

    def interval_s(self):
        a, b = _range(self.cfg.get("interval_minutes"), (3, 7))
        return 60.0 * self.rng.uniform(max(0.1, a), max(0.1, b))

    def duration_s(self):
        if self.cfg.get("long_enable", False) and \
                self.rng.random() < float(self.cfg.get("long_chance_percent", 10)) / 100.0:
            self.long = True
            a, b = _range(self.cfg.get("long_seconds"), (8, 15))
        else:
            self.long = False
            a, b = _range(self.cfg.get("pause_seconds"), (3, 5))
        return self.rng.uniform(max(0.2, a), max(0.2, b))

    def schedule(self, now=None):
        now = time.time() if now is None else now
        self.next_at = now + self.interval_s()

    def tick(self, risky=False, now=None):
        '''
        Call on every hunting frame. Returns the seconds left of the current
        pause (> 0: send nothing but potions), or 0.
        Sets .ended (True on the frame the pause finished) for the caller to
        re-arm its watchdogs.
        '''
        now = time.time() if now is None else now
        self.ended = False
        if not self.enabled:
            self.until = None
            self.t_last_tick = now
            return 0.0
        if self.t_last_tick is not None and now - self.t_last_tick > self.GAP_S:
            gap = now - self.t_last_tick
            if self.next_at is not None:
                self.next_at += gap   # not hunting in between: does not count
            if self.until is not None:
                self.until += gap
        self.t_last_tick = now
        if self.until is not None:
            if now < self.until:
                return self.until - now
            self.until = None
            self.ended = True
            self.schedule(now)
            return 0.0
        if self.next_at is not None and now >= self.next_at:
            if risky:
                self.deferred = True
                return 0.0
            self.deferred = False
            self.dur = self.duration_s()
            self.until = now + self.dur
            self.count += 1
            return self.dur
        return 0.0

    def status(self, now=None):
        now = time.time() if now is None else now
        return {"enabled": self.enabled, "pausing": self.until is not None and now < self.until,
                "left_s": round(max(0.0, self.until - now), 1) if self.until is not None else None,
                "dur_s": round(self.dur, 1), "long": self.long, "deferred": self.deferred,
                "next_in_s": round(max(0.0, self.next_at - now), 0) if self.next_at is not None else None,
                "count": self.count}
