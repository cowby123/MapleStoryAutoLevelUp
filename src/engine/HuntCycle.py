'''
HuntCycle - hunt / rest cycles and session limits (cfg hunt_cycle).

Pure scheduling logic: no input, no image. MapleStoryAutoBot asks it every
frame which phase it is in and acts on that. The clock and the random source
are injectable, so tests can run a whole day in microseconds
(tests/test_hunt_cycle.py).

Phases
  hunt         練功: the normal bot
  to_safe      前往安全點: walking to hunt_cycle.safe_spots[<map>] (rest_action "safe_spot")
  rest         休息: hold still (no moving / attacking / buffs; potions still work)
  wait_window  不在時段內: like rest, until the daily window opens
  ended        已結束: session over (session limit, or after return_home);
               the bot holds still until it is restarted

hunt (random hunt_minutes) -> [to_safe ->] rest (random rest_minutes) -> hunt ...
  to_safe: reached -> rest; not reached within safe_spot_timeout s -> rest in place (logged)
  rest_action "go_home": hunt -> (press return_home once) -> ended
'''
# Standard import
import datetime
import random
import time

PHASE_CN = {
    "hunt": "練功", "to_safe": "前往安全點", "rest": "休息",
    "wait_window": "不在時段內", "ended": "已結束", "off": "未啟用",
}


def _range(v, default):
    '''
    [min, max] minutes from a config value (number or pair); max >= min >= 0
    '''
    if v is None or v == "" or v == []:
        v = default
    if isinstance(v, (int, float)):
        v = (v, v)
    lo, hi = float(v[0]), float(v[1] if len(v) > 1 else v[0])
    lo = max(0.0, lo)
    return lo, max(lo, hi)


def _hhmm(s):
    '''
    "HH:MM" -> minutes after midnight, or None
    '''
    if not s:
        return None
    h, m = str(s).strip().split(":")
    h, m = int(h), int(m)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"bad time {s}")
    return h * 60 + m


def in_window(start, end, minutes):
    '''
    minutes-after-midnight inside [start, end) (crosses midnight when end < start;
    no window when start or end is None or start == end)
    '''
    if start is None or end is None or start == end:
        return True
    if start < end:
        return start <= minutes < end
    return minutes >= start or minutes < end


class HuntCycle:
    def __init__(self, cfg, clock=time.time, rng=None, now_local=None, log=None):
        '''
        cfg: the hunt_cycle config section. clock(): seconds; now_local():
        datetime for the daily window (default: from clock()).
        '''
        self._read(cfg)
        self.clock = clock
        self.rng = rng or random.Random()
        self.now_local = now_local or (lambda: datetime.datetime.fromtimestamp(self.clock()))
        self.log = log or (lambda msg: None)
        self.t_start = self.clock()
        self.phase = "off" if not self.enabled else None
        self.t_phase = self.t_start
        self.t_end = None         # when the current timed phase ends
        self.cycle = 0            # hunt phases started
        self.events = []          # (time, text), last 20
        self.fell_back = False    # the last to_safe timed out
        self.go_home_pending = False
        self.ended_why = ""
        if self.enabled:
            self._enter("hunt" if self._window_open() else "wait_window", "start")

    def _read(self, cfg):
        c = cfg or {}
        self.enabled = bool(c.get("enable", False))
        self.hunt_rng = _range(c.get("hunt_minutes"), (30, 40))
        self.rest_rng = _range(c.get("rest_minutes"), (5, 10))
        self.rest_action = c.get("rest_action", "stop")
        if self.rest_action not in ("stop", "safe_spot", "go_home"):
            self.rest_action = "stop"
        self.safe_timeout = float(c.get("safe_spot_timeout", 120))
        self.session_s = float(c.get("session_hours", 0) or 0) * 3600
        self.win_start = _hhmm(c.get("window_start"))
        self.win_end = _hhmm(c.get("window_end"))

    def reconfigure(self, cfg):
        '''
        New settings while running (web UI). The phase in progress keeps its
        start time; its length is picked again from the new range (or the new
        timeout), so it may end right away. Session time counts from the start.
        '''
        was = self.enabled
        old = (self.hunt_rng, self.rest_rng, self.safe_timeout)
        self._read(cfg)
        now = self.clock()
        if self.enabled and not was:
            self.t_start = now
            self.cycle = 0
            self.ended_why = ""
            self._enter("hunt" if self._window_open() else "wait_window", "啟用")
            return
        if not self.enabled:
            if was:
                self.events = (self.events + [(now, "停用")])[-20:]
            self.phase = "off"
            self.t_end = None
            return
        new = (self.hunt_rng, self.rest_rng, self.safe_timeout)
        if new != old and self.phase in ("hunt", "rest", "to_safe"):
            if self.phase == "hunt" and self.hunt_rng != old[0]:
                self.t_end = self.t_phase + self._pick(self.hunt_rng)
            elif self.phase == "rest" and self.rest_rng != old[1]:
                self.t_end = self.t_phase + self._pick(self.rest_rng)
            elif self.phase == "to_safe":
                self.t_end = self.t_phase + self.safe_timeout
            left = max(0, self.t_end - now)
            self.events = (self.events + [(now, f"設定改了：{PHASE_CN[self.phase]}還剩 {left:.0f} 秒")])[-20:]
            self.log(f"[HuntCycle] reconfigured: {self.phase} ends in {left:.0f}s")

    def fail_safe_spot(self, why):
        '''
        Cannot walk to a safe spot (none set for this map...): rest in place
        '''
        if self.phase == "to_safe":
            self._enter("rest", why)
            self.fell_back = True

    # -------------------------------------------------------------
    def _window_open(self):
        d = self.now_local()
        return in_window(self.win_start, self.win_end, d.hour * 60 + d.minute)

    def _pick(self, rng):
        return self.rng.uniform(rng[0], rng[1]) * 60.0

    def _enter(self, phase, why):
        now = self.clock()
        self.phase = phase
        self.t_phase = now
        self.t_end = None
        if phase == "hunt":
            self.cycle += 1
            self.t_end = now + self._pick(self.hunt_rng)
        elif phase == "rest":
            self.t_end = now + self._pick(self.rest_rng)
        elif phase == "to_safe":
            self.t_end = now + self.safe_timeout
            self.fell_back = False
        text = f"{PHASE_CN.get(phase, phase)}（{why}）" + \
            (f"，{round(self.t_end - now)} 秒" if self.t_end else "")
        self.events = (self.events + [(now, text)])[-20:]
        self.log(f"[HuntCycle] -> {phase} ({why})" + (f", {self.t_end - now:.0f}s" if self.t_end else ""))

    def tick(self, at_safe_spot=False):
        '''
        Advance the schedule. at_safe_spot: the player stands on the safe spot
        (only used in to_safe). Returns the phase.
        '''
        if not self.enabled or self.phase == "ended":
            return self.phase
        now = self.clock()
        if self.session_s > 0 and now - self.t_start >= self.session_s:
            self.ended_why = f"已玩 {self.session_s / 3600:g} 小時"
            self._enter("ended", self.ended_why)
            return self.phase
        if not self._window_open():
            if self.phase != "wait_window":
                self._enter("wait_window", "不在每日時段內")
            return self.phase
        if self.phase == "wait_window":
            self._enter("hunt", "時段開始")
        elif self.phase == "hunt" and now >= self.t_end:
            if self.rest_action == "safe_spot":
                self._enter("to_safe", "練功時間到")
            elif self.rest_action == "go_home":
                self.go_home_pending = True
                self.ended_why = "練功時間到，回城"
                self._enter("ended", self.ended_why)
            else:
                self._enter("rest", "練功時間到")
        elif self.phase == "to_safe":
            if at_safe_spot:
                self._enter("rest", "到達安全點")
            elif now >= self.t_end:
                self.fell_back = True
                self._enter("rest", f"{self.safe_timeout:.0f} 秒內沒走到安全點，原地休息")
        elif self.phase == "rest" and now >= self.t_end:
            self._enter("hunt", "休息結束")
        return self.phase

    def take_go_home(self):
        '''
        True once after rest_action go_home ended the hunt (press return_home now)
        '''
        if self.go_home_pending:
            self.go_home_pending = False
            return True
        return False

    @property
    def hunting(self):
        return not self.enabled or self.phase == "hunt"

    def status(self):
        now = self.clock()
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "phase_cn": PHASE_CN.get(self.phase, self.phase),
            "remaining_s": round(self.t_end - now, 1) if self.t_end else None,
            "cycle": self.cycle,
            "rest_action": self.rest_action,
            "session_elapsed_s": round(now - self.t_start),
            "session_left_s": round(self.session_s - (now - self.t_start)) if self.session_s > 0 else None,
            "window": [self.win_start, self.win_end],
            "fell_back": self.fell_back,
            "ended_why": self.ended_why,
            "events": [f"{datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')} {e}" for t, e in self.events],
        }
