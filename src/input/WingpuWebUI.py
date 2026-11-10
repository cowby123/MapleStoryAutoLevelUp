'''
WingpuWebUI
Settings / control API behind the bot's HTTP server (WingpuControl, cfg
wingpu.http_port). The page is src/input/webui/index.html; it uses only
relative URLs, so it works behind a reverse proxy that strips a path prefix
(wgsd: /app/<vm>/... -> http://<container>:8765/...).

Settings are saved to the user override file (default /work/msbot_config.yaml,
src/utils/config_layers.py), the last config layer. The shipped config/*.yaml
are never written. Applying = restarting the bot process in place (os.execv,
same PID, the container keeps running).

Live input: POST api/live needs {"live": true, "confirm": "LIVE"}. Without
"remember" it lasts until the bot restarts (every start is a dry run);
"remember" writes wingpu.dry_run: false + wingpu.remember_live: true to the
user file.
'''
# Standard import
import copy
import glob
import itertools
import os
import threading
import time

# Library import
import cv2

# Library import
import yaml

# Local import
from src.utils.logger import logger
from src.utils.common import load_yaml, convert_tuples_to_lists
from src.utils.config_layers import default_user_override, build_cfg
from src.utils.profiles import Profiles, split as split_profile
from src.utils.charname import name_crop, has_text, identify
from src.input.WingpuKeyboard import get_keyboard, to_key_code, key_option_groups, KEY_MAP
from src.utils.nametag import propose_rect, save_template, NametagTracker
from src.utils.resolution import apply_scale
import base64
import numpy as np

# Settings the page edits: (section, key, kind, extra)
#   kind: key | keys | num | int | bool | choice | map | routes | floats
FIELDS = [
    ("bot", "map", "map", None),
    ("route", "files", "routes", None),
    ("bot", "mode", "choice", ["normal", "patrol", "aux"]),
    ("bot", "attack", "choice", ["directional", "aoe_skill"]),
    ("key", "directional_attack", "key", None),
    ("key", "aoe_skill", "key", None),
    ("key", "jump", "key", None),
    ("key", "teleport", "key", None),
    ("key", "add_hp", "key", None),
    ("key", "add_mp", "key", None),
    ("key", "party", "key", None),
    ("key", "return_home", "key", None),
    ("buff_skill", "items", "buffs", None),
    ("player_detect", "mode", "choice", ["auto", "party_bar", "nametag"]),
    ("player_detect", "nametag_thres", "num", (0.05, 0.6)),
    ("health_monitor", "enable", "bool", None),
    ("health_monitor", "add_hp_percent", "num", (0, 100)),
    ("health_monitor", "add_mp_percent", "num", (0, 100)),
    ("health_monitor", "add_hp_cooldown", "num", (0, 60)),
    ("health_monitor", "add_mp_cooldown", "num", (0, 60)),
    ("directional_attack", "range_x", "int", (10, 1500)),
    ("directional_attack", "range_y", "int", (10, 800)),
    ("directional_attack", "cooldown", "num", (0, 30)),
    ("directional_attack", "shape", "choice", ["rect", "wedge"]),
    ("directional_attack", "wedge_preset", "choice", ["archer", "arrow_blow", "mage", "energy_bolt", "thief_claw", "custom"]),
    ("directional_attack", "wedge_skill_level", "int", (0, 15)),
    ("directional_attack", "wedge_start", "int", (0, 300)),
    ("directional_attack", "wedge_range", "int", (50, 1000)),
    ("directional_attack", "wedge_slope", "num", (1, 20)),
    ("directional_attack", "wedge_center_height", "int", (0, 100)),
    ("directional_attack", "wedge_feet_offset", "int", (-50, 100)),
    ("directional_attack", "dead_zone_action", "choice", ["ignore", "back_off"]),
    ("directional_attack", "stand_while_target", "bool", None),
    # 單體 / 群體 / 自動 (src/engine/AoeAttack.py)
    ("attack", "mode", "choice", ["single", "aoe", "auto"]),
    ("attack", "aoe_min_count", "int", (1, 30)),
    ("attack", "aoe_cooldown", "num", (0, 120)),
    ("attack", "aoe_shape", "choice", ["box", "front", "wedge"]),
    ("attack", "aoe_front_start", "int", (0, 1000)),
    ("attack", "aoe_front_range", "int", (10, 1500)),
    ("attack", "aoe_height", "int", (10, 800)),
    ("attack", "aoe_dy", "int", (-300, 300)),
    ("attack", "aoe_wedge_slope", "num", (1, 20)),
    ("attack", "aoe_count_window_s", "num", (0, 3)),
    ("attack", "aoe_count_stat", "choice", ["max", "median"]),
    ("route", "jump_runup", "bool", None),
    ("route", "climb_stall_s", "num", (0.3, 2.0)),
    ("route", "climb_retry_walk_s", "num", (0.1, 1.0)),
    ("route", "climb_retry_max", "int", (1, 10)),
    ("wingpu", "profile_autodetect", "bool", None),
    # 類人保護 (src/engine/HumanPause.py)
    ("human_pause", "enable", "bool", None),
    ("human_pause", "interval_minutes", "range", (0.5, 120)),
    ("human_pause", "pause_seconds", "range", (1, 60)),
    ("human_pause", "long_enable", "bool", None),
    ("human_pause", "long_chance_percent", "int", (1, 100)),
    ("human_pause", "long_seconds", "range", (2, 120)),
    ("book_dismiss", "enable", "bool", None),
    ("book_dismiss", "force_in_dry_run", "bool", None),
    ("curse", "stop_while_cursed", "bool", None),
    ("curse", "action", "choice", ["stop", "safe_spot"]),
    ("detection_lost", "minimap_grace_s", "num", (0, 120)),
    ("rune_curse", "enable", "bool", None),
    ("rune_curse", "seek_timeout", "int", (10, 1800)),
    ("rune_curse", "interact_key", "key", None),
    ("rune_curse", "interact_every_px", "int", (5, 500)),
    ("rune_curse", "interact_cooldown", "num", (0.2, 10)),
    ("rune_curse", "resume", "choice", ["curse_gone", "rest_time"]),
    ("rune_curse", "rest_minutes", "range", (0.5, 600)),
    ("rune_curse", "sweep_tap_s", "num", (0.03, 0.5)),
    ("rune_curse", "sweep_tap_gap", "num", (0, 1)),
    ("rune_curse", "sweep_px_per_tap", "num", (1, 60)),
    ("rune_curse", "sweep_max_px", "int", (10, 300)),
    ("rune_curse", "sweep_up_interval", "num", (0.1, 3)),
    ("rune_curse", "sweep_away_s", "num", (0.2, 5)),
    ("rune_curse", "sweep_timeout", "num", (3, 120)),
    ("rune_curse", "minigame_wait", "num", (0.2, 3)),
    ("rune_curse", "minigame_retries", "int", (0, 10)),
    ("rune_curse", "capture_frames", "bool", None),
    ("rune_curse", "capture_interval", "num", (0.5, 60)),
    ("rune_curse", "capture_max", "int", (0, 5000)),
    ("rune_curse", "capture_rotate", "bool", None),
    ("route", "jump_stuck_s", "num", (0.5, 10)),
    ("route", "jump_runup_back_px", "int", (1, 20)),
    ("directional_attack", "target_hold_s", "num", (0, 5)),
    ("directional_attack", "target_commit_s", "num", (0, 5)),
    ("aoe_skill", "range_x", "int", (10, 1500)),
    ("aoe_skill", "range_y", "int", (10, 800)),
    ("monster_detect", "mode", "choice", ["contour_only", "color", "grayscale", "template_free", "yolo"]),
    ("monster_detect", "diff_thres", "num", (0, 1)),
    ("monster_detect", "search_box_margin", "int", (0, 500)),
    ("monster_detect", "search", "choice", ["roi", "full", "attack"]),
    ("monster_detect", "roi_left", "int", (0, 1500)),
    ("monster_detect", "roi_right", "int", (0, 1500)),
    ("monster_detect", "roi_up", "int", (0, 800)),
    ("monster_detect", "roi_down", "int", (0, 800)),
    ("monster_detect", "interval_frames", "int", (1, 10)),
    ("edge_teleport", "enable", "bool", None),
    ("watchdog", "timeout", "num", (1, 600)),
    ("detection_lost", "timeout", "num", (0.5, 60)),
    ("system", "fps_limit_main", "int", (1, 30)),
    # hunt / rest cycles
    ("hunt_cycle", "enable", "bool", None),
    ("hunt_cycle", "hunt_minutes", "range", (0.5, 600)),
    ("hunt_cycle", "rest_minutes", "range", (0.25, 600)),
    ("hunt_cycle", "rest_action", "choice", ["stop", "safe_spot", "go_home"]),
    ("hunt_cycle", "safe_spot_timeout", "int", (10, 1800)),
    ("hunt_cycle", "session_hours", "num", (0, 48)),
    ("hunt_cycle", "session_end_action", "choice", ["stop", "go_home"]),
    ("hunt_cycle", "window_start", "hhmm", None),
    ("hunt_cycle", "window_end", "hhmm", None),
    # every other time setting
    ("watchdog", "last_attack_timeout", "int", (30, 7200)),
    ("scheduled_channel_switching", "interval_seconds", "int", (60, 86400)),
    ("health_monitor", "return_home_watch_dog_timeout", "num", (1, 60)),
    ("health_monitor", "fps_limit", "int", (1, 30)),
    ("buff_skill", "action_cooldown", "num", (0, 10)),
    ("aoe_skill", "cooldown", "num", (0, 30)),
    ("teleport", "cooldown", "num", (0, 30)),
    ("rune_find", "near_rune_duration", "num", (1, 120)),
    ("patrol", "patrol_attack_interval", "num", (0.1, 60)),
    ("player_detect", "nametag_local_timeout", "num", (0.1, 10)),
    ("system", "fps_limit_window_capturor", "int", (1, 30)),
    ("wingpu", "debug_interval", "num", (0.5, 60)),
    ("wingpu", "redetect_interval", "num", (1, 600)),
    ("wingpu", "mjpeg_fps", "int", (1, 30)),
    ("wingpu", "mjpeg_scale", "num", (0.5, 1)),
    # optional behaviours (switches)
    ("rune_find", "enable", "bool", None),
    ("watchdog", "last_attack_enable", "bool", None),
    ("watchdog", "last_attack_timeout_action", "choice", ["change_channel", "go_home"]),
    ("scheduled_channel_switching", "enable", "bool", None),
    ("channel_change", "enable", "bool", None),
    ("health_monitor", "return_home_if_no_potion", "bool", None),
    ("health_monitor", "force_heal", "bool", None),
    ("wingpu", "ensure_party_on_start", "bool", None),
]

# Applied when the bot restarts; everything else applies at once (apply_live)
RESTART_FIELDS = {"bot.map", "route.files", "bot.mode", "bot.attack", "health_monitor.enable"}

# Route colour code -> name in the painter (teleport ones only when a teleport key is set)
COMMAND_CN = {
    "left none none": "左", "right none none": "右",
    "left none jump": "左跳", "right none jump": "右跳",
    "none none jump": "原地跳", "none down jump": "下跳",
    "none up none": "往上爬", "none down none": "往下爬",
    "none up jump": "跳上繩子（按住上＋跳）",
    "stop stop stop": "停", "none none goal": "目標（換下一條路線）",
    "none up teleport": "往上瞬移", "none down teleport": "往下瞬移",
    "left none teleport": "往左瞬移", "right none teleport": "往右瞬移",
}


def _get(cfg, sec, key):
    v = (cfg.get(sec) or {}).get(key)
    if isinstance(v, tuple):
        v = list(v)
    return v


def buff_items(cfg):
    '''
    Buff rows for the editor: buff_skill.items when the web UI wrote them,
    else built from the parallel lists keys / cooldown / cast_at_start / names
    '''
    b = cfg.get("buff_skill") or {}
    items = b.get("items")
    if items:
        return [dict(it) for it in items]
    keys = list(b.get("keys") or [])
    cds = list(b.get("cooldown") or [])
    at = list(b.get("cast_at_start") or [])
    names = list(b.get("names") or [])
    return [{"name": names[i] if i < len(names) else "", "key": k,
             "cooldown": cds[i] if i < len(cds) else 0, "enabled": True,
             "cast_at_start": at[i] if i < len(at) else True} for i, k in enumerate(keys)]


class WebUI:
    '''
    API for the settings page. control = WingpuControl.
    '''
    def __init__(self, control):
        self.control = control
        self.bot = control.bot
        self.path = default_user_override()
        self.lock = threading.Lock()
        # per-character profiles (src/utils/profiles.py); build_cfg migrated the
        # user file already, this covers a start without it
        self.profiles = Profiles(self.path)
        if not self.profiles.exists():
            self.profiles.migrate()
        self.detect = {"t": 0.0, "candidate": None, "count": 0, "scores": {}, "seen": None}
        self.profile_note = None   # (text, time) for 目前狀態
        self.data = load_yaml("config/config_data.yaml")
        self.snapshots = {}  # id -> (frame BGR, player centre or None, scale)
        self.snap_ids = itertools.count(1)

    ###########
    ### GET ###
    ###########
    def maps(self):
        '''
        Maps with a minimaps/<map>/map.png and a monster list in config_data.yaml
        '''
        names = self.data.get("map_mobs_mapping", {})
        out = []
        for d in sorted(glob.glob("minimaps/*/map.png")):
            m = os.path.basename(os.path.dirname(d))
            routes = sorted(os.path.basename(p) for p in glob.glob(f"minimaps/{m}/route*.png")
                            if not p.endswith("route_rest.png"))
            out.append({"id": m, "name": self.data.get("name", {}).get(m) or _cn_name(self.data, m),
                        "routes": routes, "supported": m in names,
                        "monsters": list(names.get(m, []))})
        return out

    def user_override(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError) as e:
                logger.error(f"[WebUI] cannot read {self.path}: {e}")
        return {}

    def config(self):
        cfg = self.control.cfg_unscaled()
        values = {f"{s}.{k}": _get(cfg, s, k) for s, k, _, _ in FIELDS}
        values["buff_skill.items"] = buff_items(cfg)
        kb = get_keyboard()
        return {
            "values": values,
            "fields": [{"name": f"{s}.{k}", "kind": kind,
                        "choices": extra if kind == "choice" else None,
                        "range": extra if kind in ("num", "int", "floats", "range") else None}
                       for s, k, kind, extra in FIELDS],
            "maps": self.maps(),
            # key dropdowns: groups of (value, label) and every accepted name -> code
            "key_options": [{"group": g, "keys": [{"value": v, "label": lab, "code": KEY_MAP[v]} for v, lab in keys]}
                            for g, keys in key_option_groups()],
            "key_codes": KEY_MAP,
            "user_file": self.path,
            "user_override": convert_tuples_to_lists(self.user_override()),
            "live": not kb.dry_run,
            "remember_live": bool((self.user_override().get("wingpu") or {}).get("remember_live")),
            "pending_restart": self.control.pending_restart,
        }

    ############
    ### POST ###
    ############
    def validate(self, values):
        '''
        values: {"section.key": value}. Returns (clean nested dict, errors {name: msg})
        '''
        errors = {}
        clean = {}
        live_input = not get_keyboard().dry_run
        spec = {f"{s}.{k}": (s, k, kind, extra) for s, k, kind, extra in FIELDS}
        maps = {m["id"]: m for m in self.maps()}
        for name, v in values.items():
            if name not in spec:
                errors[name] = "不認得的設定"
                continue
            sec, key, kind, extra = spec[name]
            try:
                if kind == "key":
                    v = "" if v is None else str(v).strip().lower()
                    if v and to_key_code(v) is None:
                        raise ValueError(f"不認得的按鍵「{v}」")
                elif kind == "buffs":
                    rows = []
                    for i, it in enumerate(v or []):
                        key = str(it.get("key", "")).strip().lower()
                        if not key:
                            raise ValueError(f"第 {i + 1} 列沒有按鍵")
                        if to_key_code(key) is None:
                            raise ValueError(f"第 {i + 1} 列：不認得的按鍵「{key}」")
                        try:
                            cd = float(it.get("cooldown"))
                        except (TypeError, ValueError):
                            raise ValueError(f"第 {i + 1} 列：冷卻要是數字") from None
                        if not 1 <= cd <= 36000:
                            raise ValueError(f"第 {i + 1} 列：冷卻範圍 1–36000 秒")
                        rows.append({"name": str(it.get("name", "")).strip()[:24], "key": key,
                                     "cooldown": int(cd) if cd == int(cd) else cd,
                                     "enabled": bool(it.get("enabled", True)),
                                     "cast_at_start": bool(it.get("cast_at_start", True))})
                    on = [r for r in rows if r["enabled"]]
                    # the engine reads the parallel lists; items keeps disabled rows too
                    clean.setdefault("buff_skill", {}).update({
                        "items": rows,
                        "keys": [r["key"] for r in on],
                        "cooldown": [r["cooldown"] for r in on],
                        "cast_at_start": [r["cast_at_start"] for r in on],
                        "names": [r["name"] for r in on],
                    })
                    continue
                elif kind in ("num", "int"):
                    v = float(v)
                    if kind == "int":
                        if v != int(v):
                            raise ValueError("要整數")
                        v = int(v)
                    lo, hi = extra
                    if not lo <= v <= hi:
                        raise ValueError(f"範圍 {lo}–{hi}")
                elif kind == "range":
                    lo, hi = extra
                    a, b = float(v[0]), float(v[1])
                    if not (lo <= a <= hi and lo <= b <= hi):
                        raise ValueError(f"範圍 {lo:g}–{hi:g}")
                    if a > b:
                        raise ValueError("最少不能大於最多")
                    v = [int(a) if a == int(a) else a, int(b) if b == int(b) else b]
                    if live_input and sec == "hunt_cycle" and a < 1:
                        # guard: very short cycles are for dry-run tests only
                        raise ValueError("真的輸入時，最少要 1 分鐘（更短只能在模擬模式測試）")
                elif kind == "hhmm":
                    v = "" if v is None else str(v).strip()
                    if v:
                        hh, _, mm = v.partition(":")
                        if not (hh.isdigit() and mm.isdigit() and 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                            raise ValueError("格式 HH:MM，例如 09:30；空白 = 不限")
                        v = f"{int(hh):02d}:{int(mm):02d}"
                elif kind == "floats":
                    v = [float(x) for x in (v or [])]
                    lo, hi = extra
                    if any(not lo <= x <= hi for x in v):
                        raise ValueError(f"每個值的範圍 {lo}–{hi}")
                elif kind == "bool":
                    if not isinstance(v, bool):
                        raise ValueError("要 true / false")
                elif kind == "choice":
                    if v not in extra:
                        raise ValueError(f"只能是 {', '.join(extra)}")
                elif kind == "map":
                    if v not in maps:
                        raise ValueError("minimaps/ 裡沒有這張地圖")
                    if not maps[v]["supported"]:
                        raise ValueError("config_data.yaml 沒有這張地圖的怪物")
                elif kind == "routes":
                    v = [str(x) for x in (v or [])]
            except (TypeError, ValueError) as e:
                errors[name] = str(e)
                continue
            clean.setdefault(sec, {})[key] = v
        # cross checks
        mode = (clean.get("player_detect") or {}).get("mode")
        if mode in ("nametag", "auto"):
            path = self.nametag_path()
            if not os.path.exists(path):
                errors["player_detect.mode"] = "還沒有名牌模板：先用「截取名牌」截一張，再選名牌 / 自動"
        amode = (clean.get("attack") or {}).get("mode")
        if amode in ("aoe", "auto"):
            k = (clean.get("key") or {}).get("aoe_skill", _get(self.control.cfg_unscaled(), "key", "aoe_skill"))
            if not k:
                errors["attack.mode"] = "群體 / 自動要先設「範圍技鍵」"
        m = (clean.get("bot") or {}).get("map") or _get(self.control.cfg_unscaled(), "bot", "map")
        routes = (clean.get("route") or {}).get("files")
        if routes and m in maps:
            bad = [r for r in routes if r not in maps[m]["routes"]]
            if bad:
                errors["route.files"] = f"{m} 沒有 {', '.join(bad)}"
        return clean, errors

    def save(self, values):
        clean, errors = self.validate(values)
        if errors:
            return {"ok": False, "errors": errors}
        g, p = split_profile(clean)
        with self.lock:
            if g:
                over = self.user_override()
                for sec, kv in g.items():
                    over.setdefault(sec, {})
                    if not isinstance(over[sec], dict):
                        over[sec] = {}
                    over[sec].update(kv)
                self._write(over)
            if p:
                pid = self.profiles.active() or self.profiles.migrate()
                self.profiles.update(pid, p)
        live, restart = self.apply_live(clean)
        if restart:
            self.control.pending_restart = True
        logger.info(f"[WebUI] settings saved to {self.path}: {clean}; live {live}; restart {restart}")
        hc = getattr(self.bot, "hunt_cycle", None)
        return {"ok": True, "saved": clean, "user_file": self.path, "applied_live": live,
                "needs_restart": restart, "hunt_cycle": hc.status() if hc is not None else None}

    def apply_live(self, clean):
        '''
        Put saved values into the running bot. Returns (applied names, names
        that need a bot restart). Lengths are rescaled for the current frame.
        '''
        bot = self.bot
        cfg = bot.cfg
        base = getattr(bot, "cfg_base", None) or {}
        live, restart = [], []
        old_md = copy.deepcopy(cfg.get("monster_detect") or {})
        for sec, kv in clean.items():
            for key, v in kv.items():
                name = f"{sec}.{key}"
                if name in RESTART_FIELDS:
                    restart.append(name)
                    continue
                if sec == "buff_skill" and key in ("keys", "items", "cooldown", "cast_at_start", "names"):
                    continue
                cfg.setdefault(sec, {})[key] = copy.deepcopy(v)
                if sec in base:
                    base[sec][key] = copy.deepcopy(v)
                live.append(name)
        # buffs: same keys in the same order -> live; otherwise the keyboard
        # controller's timers must be rebuilt (restart)
        b = clean.get("buff_skill") or {}
        if "keys" in b:
            if list(b["keys"]) == list(cfg["buff_skill"].get("keys") or []):
                for k in ("cooldown", "cast_at_start", "names", "items"):
                    cfg["buff_skill"][k] = copy.deepcopy(b[k])
                live.append("buff_skill.items")
            else:
                restart.append("buff_skill.items")
        # derived state
        if bot.frame_hw and base:
            apply_scale(cfg, base, bot.scale, bot.frame_hw)
        # templates only when something they depend on changed (every full-form
        # save sends every field)
        tpl_keys = {"mode", "contour_blur", "match_downscale", "color_prefilter", "black_thres"}
        if tpl_keys & set((clean.get("monster_detect") or {}).keys()) and                 any(old_md.get(k) != cfg["monster_detect"].get(k) for k in tpl_keys):
            bot.build_templates(bot.scale)
        kb = bot.kb
        if kb is not None and "key" in clean:
            kb.attack_key = cfg["key"]["aoe_skill"] if cfg["bot"]["attack"] == "aoe_skill" \
                else cfg["key"]["directional_attack"]
        hm = bot.health_monitor
        if hm is not None:
            hm.fps_limit = cfg["health_monitor"]["fps_limit"]
        cap = bot.capture
        if cap is not None:
            if hasattr(cap, "fps_limit"):
                cap.fps_limit = cfg["system"]["fps_limit_window_capturor"]
            if hasattr(cap, "redetect_interval"):
                cap.redetect_interval = float((cfg.get("wingpu") or {}).get("redetect_interval", 5))
        self.control.debug_interval = float((cfg.get("wingpu") or {}).get("debug_interval", 2))
        if "player_detect" in clean and "mode" in clean["player_detect"]:
            # nametag on: load the template; off: drop it (never used)
            if clean["player_detect"]["mode"] in ("nametag", "auto"):
                if getattr(bot, "nametag", None) is None:
                    bot.init_nametag()
            else:
                bot.nametag = None
        tr = getattr(bot, "nametag", None)
        if tr is not None:
            pd = cfg.get("player_detect") or {}
            tr.thres = float(pd.get("nametag_thres", tr.thres))
            tr.local_timeout = float(pd.get("nametag_local_timeout", tr.local_timeout))
        hc = getattr(bot, "hunt_cycle", None)
        if hc is not None and "hunt_cycle" in clean:
            hc.reconfigure(cfg.get("hunt_cycle"))
        hp = getattr(bot, "human_pause", None)
        if hp is not None and "human_pause" in clean:
            hp.reconfigure(cfg.get("human_pause"))
            if not hp.enabled:
                bot.cancel_human_pause("turned off")
        return live, restart

    ##########################
    ### Safe spot / routes ###
    ##########################
    def legend(self):
        cfg = self.bot.cfg
        has_tp = bool(cfg["key"].get("teleport"))
        out = []
        for table in ("color_code", "color_code_up_down"):
            for rgb, cmd in cfg["route"][table].items():
                out.append({"rgb": [int(c) for c in rgb.split(",")], "cmd": cmd,
                            "cn": COMMAND_CN.get(cmd, cmd),
                            "hidden": "teleport" in cmd and not has_tp})
        return out

    def routes(self):
        bot = self.bot
        m = bot.cfg["bot"]["map"]
        h, w = bot.img_map.shape[:2] if bot.img_map is not None else (0, 0)
        names = sorted(os.path.basename(p) for p in glob.glob(f"minimaps/{m}/route*.png")
                       if not p.endswith(("route_rest.png", "route_safe.png")))
        safe_path = bot.route_safe_path()
        return {"map": m, "w": w, "h": h, "routes": names,
                "route_safe": os.path.exists(safe_path), "route_safe_path": safe_path,
                "safe_spot": list(bot.safe_spot()) if bot.safe_spot() else None,
                "player_global": list(bot.loc_player_global),
                "tolerance": [(bot.cfg.get("hunt_cycle") or {}).get("safe_spot_tolerance_x", 3),
                              (bot.cfg.get("hunt_cycle") or {}).get("safe_spot_tolerance_y", 4)],
                "legend": self.legend()}

    def route_png(self, name):
        bot = self.bot
        m = bot.cfg["bot"]["map"]
        if name == "map":
            path = f"minimaps/{m}/map.png"
        elif name == "route_safe":
            path = bot.route_safe_path()
        elif name.startswith("route") and name.endswith(".png") and "/" not in name and "\\" not in name:
            path = f"minimaps/{m}/{name}"
        else:
            return None
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def save_route_safe(self, body):
        '''
        {"png": base64 of an RGBA PNG the size of map.png}: pixels in a route
        colour (alpha > 0) are kept, everything else becomes black
        '''
        bot = self.bot
        if body.get("delete"):
            p = bot.route_safe_path()
            if os.path.exists(p):
                os.replace(p, p + ".bak")
            return {"ok": True, "deleted": True}
        try:
            raw = base64.b64decode(str(body.get("png", "")).split(",")[-1])
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        except (ValueError, TypeError):
            img = None
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            return {"ok": False, "errors": {"route_safe": "圖檔讀不出來"}}
        h, w = bot.img_map.shape[:2]
        if img.shape[:2] != (h, w):
            return {"ok": False, "errors": {"route_safe": f"大小要是 {w}×{h}"}}
        rgb = img[:, :, 2::-1]
        out = np.zeros((h, w, 3), np.uint8)
        counts = {}
        for it in self.legend():
            m = np.all(rgb == it["rgb"], axis=2) & (img[:, :, 3] > 0)
            n = int(m.sum())
            if n:
                out[m] = it["rgb"]
                counts[it["cn"]] = n
        p = bot.route_safe_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp.png"
        cv2.imwrite(tmp, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        os.replace(tmp, p)
        logger.info(f"[WebUI] safe route saved {p}: {counts}")
        return {"ok": True, "path": p, "pixels": counts}

    def save_safe_spot(self, body):
        bot = self.bot
        m = bot.cfg["bot"]["map"]
        with self.lock:
            over = self.user_override()
            hc = over.setdefault("hunt_cycle", {})
            spots = hc.setdefault("safe_spots", {})
            if body.get("clear"):
                spots.pop(m, None)
                val = None
            else:
                h, w = bot.img_map.shape[:2]
                try:
                    x, y = int(body["x"]), int(body["y"])
                except (KeyError, TypeError, ValueError):
                    return {"ok": False, "errors": {"safe_spot": "要 x, y"}}
                if not (0 <= x < w and 0 <= y < h):
                    return {"ok": False, "errors": {"safe_spot": "點在小地圖外面"}}
                spots[m] = val = [x, y]
            self._write(over)
        live = bot.cfg.setdefault("hunt_cycle", {})
        live_spots = dict(live.get("safe_spots") or {})
        if val is None:
            live_spots.pop(m, None)
        else:
            live_spots[m] = val
        live["safe_spots"] = live_spots
        return {"ok": True, "map": m, "safe_spot": val}

    def reset(self):
        '''
        「清除我的設定」: the active profile's settings go (backup .bak); the
        shared file (stream, input, detector, safe spots) stays
        '''
        with self.lock:
            pid = self.profiles.active()
            if pid:
                self.profiles.clear(pid)
        self.control.pending_restart = True
        return {"ok": True, "backup": self.profiles.path(pid) + ".bak" if pid else None}

    ################
    ### Profiles ###
    ################
    def profile_values(self, cfg):
        '''the settings a profile switch compares: every page field + the nametag file'''
        vals = {f"{s}.{k}": _get(cfg, s, k) for s, k, _, _ in FIELDS if not is_global_field(s, k)}
        vals["buff_skill.items"] = buff_items(cfg)
        vals["player_detect.nametag_template"] = _get(cfg, "player_detect", "nametag_template")
        return convert_tuples_to_lists(vals)

    def profiles_status(self):
        prof = self.profiles
        act = prof.active()
        d = dict(self.detect)
        d.pop("t", None)
        return {"active": act, "name": prof.read(act)[0].get("name") if act else None,
                "profiles": prof.list(), "dir": prof.dir,
                "autodetect": bool((self.bot.cfg.get("wingpu") or {}).get("profile_autodetect", True)),
                "detect": d}

    def snippet_png(self, pid):
        p = self.profiles.snippet_path(pid) if pid in self.profiles.ids() else None
        if not p or not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def current_name_crop(self):
        img = getattr(self.bot, "img_frame", None)
        crop = name_crop(img)
        return crop if has_text(crop) else None

    def bind(self, pid):
        crop = self.current_name_crop()
        if crop is None:
            return {"ok": False, "errors": {"profile": "畫面左下角找不到角色名字（遊戲畫面要在前景、沒被擋住）"}}
        # the same character must not be bound to two profiles
        other, scores = identify(self.bot.img_frame, {k: v for k, v in self.profiles.snippets().items() if k != pid})
        if other is not None:
            name = self.profiles.read(other)[0].get("name")
            return {"ok": False, "errors": {"profile": f"這個角色已經綁定在「{name}」"}}
        self.profiles.save_snippet(pid, crop)
        logger.info(f"[Profiles] {pid} bound to the character on screen")
        return {"ok": True}

    def profile_action(self, body):
        act = body.get("action")
        prof = self.profiles
        pid = body.get("id") or prof.active()
        try:
            if act == "create":
                new = prof.create(str(body.get("name") or "").strip()[:24] or None, copy_from=prof.active())
                # its own nametag file
                prof.update(new, {"player_detect": {"nametag_template": os.path.join(prof.dir, f"{new}_nametag.png")}})
                res = {"ok": True, "id": new}
                if body.get("bind"):
                    b = self.bind(new)
                    if not b["ok"]:
                        prof.delete(new)
                        return b
                if body.get("switch", True):
                    res.update(self.switch_profile(new, "新增"))
                return res
            if act == "rename":
                prof.rename(pid, str(body.get("name") or "").strip() or pid)
                return {"ok": True}
            if act == "delete":
                was_active = pid == prof.active()
                prof.delete(pid)
                if was_active:
                    return self.switch_profile(prof.active(), "刪除後")
                return {"ok": True}
            if act == "switch":
                return self.switch_profile(pid, "手動切換")
            if act == "bind":
                return self.bind(pid)
        except ValueError as e:
            return {"ok": False, "errors": {"profile": str(e)}}
        return {"ok": False, "errors": {"profile": f"不認得的動作 {act}"}}

    def switch_profile(self, pid, why, auto=False):
        '''
        Make pid the active profile and apply its settings: live where the page
        applies them live, else a bot restart that keeps running / paused and
        live / dry run as they are. Keys are released first.
        '''
        prof = self.profiles
        if pid not in prof.ids():
            return {"ok": False, "errors": {"profile": "沒有這個設定檔"}}
        bot = self.bot
        control = self.control
        control.release_keys()
        old = self.profile_values(self.control.cfg_unscaled())
        prof.set_active(pid)
        cfg_name = getattr(bot.args, "cfg", None) or "wingpu_maple"
        new_cfg = build_cfg(cfg_name, user_override=self.path)
        new = self.profile_values(new_cfg)
        clean = {}
        for name, v in new.items():
            if name == "buff_skill.items" or old.get(name) == v:
                continue
            sec, key = name.split(".", 1)
            clean.setdefault(sec, {})[key] = copy.deepcopy(v)
        if old.get("buff_skill.items") != new.get("buff_skill.items"):
            b = new_cfg.get("buff_skill") or {}
            clean["buff_skill"] = {k: copy.deepcopy(convert_tuples_to_lists(b.get(k)))
                                   for k in ("items", "keys", "cooldown", "cast_at_start", "names") if k in b}
        nt = (clean.get("player_detect") or {}).pop("nametag_template", None)
        live, restart = self.apply_live(clean) if clean else ([], [])
        if nt is not None:
            bot.cfg.setdefault("player_detect", {})["nametag_template"] = nt
            if getattr(bot, "nametag", None) is not None:
                bot.init_nametag()
        name = prof.read(pid)[0].get("name") or pid
        text = (f"偵測到角色：{name} → 載入設定" if auto else f"切換到「{name}」的設定（{why}）")
        self.profile_note = (text, time.time())
        logger.info(f"[Profiles] -> {pid} ({name}, {why}); live {live}; restart {restart}")
        if restart:
            self.control.restart_keep_state(f"profile {pid}")
            return {"ok": True, "applied_live": live, "restart": restart, "note": text + "：重新啟動 bot（維持執行 / 暫停、真的輸入 / 模擬）"}
        return {"ok": True, "applied_live": live, "restart": [], "note": text + "：已立即套用"}

    def detect_tick(self, now=None):
        '''
        Every ~5 s (WingpuControl.tick): bind a migrated profile to the
        character on screen; with wingpu.profile_autodetect, switch to the
        profile whose saved name matches, after two checks in a row
        '''
        now = time.time() if now is None else now
        prof = self.profiles
        img = getattr(self.bot, "img_frame", None)
        if img is None:
            return
        act = prof.active()
        if act:
            meta, _ = prof.read(act)
            if meta.get("bind_pending") and not os.path.exists(prof.snippet_path(act)):
                if self.current_name_crop() is not None:
                    self.bind(act)
        snippets = prof.snippets()
        pid, scores = identify(img, snippets)
        self.detect.update(t=now, scores={k: round(v, 3) for k, v in scores.items()}, seen=pid)
        if not (self.bot.cfg.get("wingpu") or {}).get("profile_autodetect", True):
            self.detect.update(candidate=None, count=0)
            return
        if pid is None or pid == act:
            self.detect.update(candidate=None, count=0)
            return
        if self.detect.get("candidate") == pid:
            self.detect["count"] += 1
        else:
            self.detect.update(candidate=pid, count=1)
        if self.detect["count"] >= 2:
            self.detect.update(candidate=None, count=0)
            self.switch_profile(pid, "自動偵測", auto=True)

    def _write(self, over):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("# MapleStoryAutoLevelUp user settings (written by the web UI).\n"
                    "# Last config layer; delete this file to go back to the shipped configs.\n")
            yaml.safe_dump(convert_tuples_to_lists(over), f, allow_unicode=True, sort_keys=True)
        os.replace(tmp, self.path)

    def set_live(self, body):
        '''
        {"live": bool, "confirm": "LIVE", "remember": bool}
        '''
        live = bool(body.get("live"))
        remember = bool(body.get("remember"))
        kb = get_keyboard()
        if live and body.get("confirm") != "LIVE":
            return {"ok": False, "errors": {"live": "要確認才能切到真的輸入"}}
        if live:
            kb.configure(self.bot.cfg, live=True)
        else:
            # back to dry run: release what is held while we can still send
            self.control.release_keys()
            kb.configure(self.bot.cfg, live=False)
        with self.lock:
            over = self.user_override()
            w = over.get("wingpu") if isinstance(over.get("wingpu"), dict) else {}
            if live and remember:
                w["dry_run"] = False
                w["remember_live"] = True
            else:
                w.pop("dry_run", None)
                w.pop("remember_live", None)
            if w:
                over["wingpu"] = w
            else:
                over.pop("wingpu", None)
            if over or os.path.exists(self.path):
                self._write(over)
        logger.warning(f"[WebUI] input {'LIVE' if live else 'DRY RUN'}"
                       f"{' (remembered)' if live and remember else ''}")
        return {"ok": True, "live": live, "remember_live": live and remember}


    ###############
    ### Nametag ###
    ###############
    def nametag_path(self):
        tr = getattr(self.bot, "nametag", None)
        if tr is not None:
            return tr.path
        pd = self.bot.cfg.get("player_detect") or {}
        from src.utils.nametag import default_dir
        return pd.get("nametag_template") or os.path.join(default_dir(), "nametag.png")

    def nametag_status(self):
        tr = getattr(self.bot, "nametag", None)
        path = self.nametag_path()
        st = tr.status() if tr is not None else {"loaded": False, "path": path, "meta": None, "size": None}
        st["in_use"] = tr is not None
        st["exists"] = os.path.exists(path)
        st["mtime"] = int(os.path.getmtime(path)) if st["exists"] else None
        if st["exists"] and tr is None:
            img = cv2.imread(path)
            st["size"] = list(img.shape[1::-1]) if img is not None else None
            side = os.path.splitext(path)[0] + ".json"
            if os.path.exists(side):
                import json
                with open(side, "r", encoding="utf-8") as f:
                    st["meta"] = json.load(f)
        return st

    def nametag_snapshot(self):
        '''
        Freeze the frame the bot processed last (and where the party bar put
        the player in it); propose the nametag box under the player.
        '''
        bot = self.bot
        img = bot.img_frame
        if img is None:
            return {"ok": False, "errors": {"snapshot": "還沒有畫面"}}
        img = img.copy()
        player = None
        if getattr(bot, "player_source", None) == "party_bar" and \
                time.time() - getattr(bot, "t_player_seen", 0) < 1.0:
            player = tuple(int(v) for v in bot.loc_player)
        proposal = propose_rect(img, player, bot.scale) if player else None
        sid = next(self.snap_ids)
        self.snapshots = {k: v for k, v in self.snapshots.items() if k > sid - 3}
        self.snapshots[sid] = (img, player, bot.scale)
        return {"ok": True, "id": sid, "w": img.shape[1], "h": img.shape[0],
                "player": list(player) if player else None, "proposal": proposal}

    def snapshot_jpeg(self, sid):
        snap = self.snapshots.get(int(sid))
        if snap is None:
            return None
        ok, buf = cv2.imencode(".jpg", snap[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes() if ok else None

    def nametag_save(self, body):
        try:
            sid = int(body.get("id"))
            rect = [int(v) for v in body.get("rect")]
        except (TypeError, ValueError):
            return {"ok": False, "errors": {"nametag": "要先取畫面、框出名牌"}}
        snap = self.snapshots.get(sid)
        if snap is None:
            return {"ok": False, "errors": {"nametag": "這張畫面已過期，請重新取一張"}}
        img, player, scale = snap
        tr = getattr(self.bot, "nametag", None)
        path = self.nametag_path()
        out_dir, name = os.path.split(path)
        try:
            saved, meta = save_template(img, rect, scale, player, out_dir=out_dir,
                                        name=os.path.splitext(name)[0])
        except ValueError as e:
            return {"ok": False, "errors": {"nametag": str(e)}}
        # check: does the new template find itself in that frame?
        test = NametagTracker(saved, thres=tr.thres if tr is not None else
                              float((self.bot.cfg.get("player_detect") or {}).get("nametag_thres", 0.3)))
        test.set_scale(scale)
        ui_y = min(img.shape[0], self.bot.cfg["ui_coords"]["ui_y_start"])
        p = test.find(img[:ui_y])
        if tr is not None:
            tr.load()  # the bot uses it from the next frame
        logger.info(f"[WebUI] nametag template saved {saved}: {meta}")
        return {"ok": True, "path": saved, "meta": meta,
                "test": {"found": p is not None, "player": list(p) if p else None,
                         "score": test.last_score}}

    def nametag_png(self):
        path = self.nametag_path()
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def nametag_delete(self):
        path = self.nametag_path()
        if os.path.exists(path):
            os.replace(path, path + ".bak")
        tr = getattr(self.bot, "nametag", None)
        if tr is not None:
            tr.reload_if_changed()
        return {"ok": True}


def is_global_field(sec, key):
    from src.utils.profiles import is_global
    return is_global(sec, key)


def _cn_name(data, map_id):
    '''
    The Chinese name in config_data.yaml (the "map" block lists "id: name")
    '''
    for v in data.values():
        if isinstance(v, dict) and isinstance(v.get(map_id), str):
            return v[map_id]
    return map_id


def unscaled(cfg, base):
    '''
    cfg with the resolution-scaled sections replaced by their unscaled values
    '''
    out = copy.deepcopy(cfg)
    for sec, vals in (base or {}).items():
        out[sec] = copy.deepcopy(vals)
    return out
