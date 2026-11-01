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
import os
import threading

# Library import
import yaml

# Local import
from src.utils.logger import logger
from src.utils.common import load_yaml, convert_tuples_to_lists
from src.utils.config_layers import default_user_override
from src.input.WingpuKeyboard import get_keyboard, to_key_code

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
    ("buff_skill", "keys", "keys", None),
    ("buff_skill", "cooldown", "floats", (0, 36000)),
    ("health_monitor", "enable", "bool", None),
    ("health_monitor", "add_hp_percent", "num", (0, 100)),
    ("health_monitor", "add_mp_percent", "num", (0, 100)),
    ("health_monitor", "add_hp_cooldown", "num", (0, 60)),
    ("health_monitor", "add_mp_cooldown", "num", (0, 60)),
    ("directional_attack", "range_x", "int", (10, 1500)),
    ("directional_attack", "range_y", "int", (10, 800)),
    ("directional_attack", "cooldown", "num", (0, 30)),
    ("aoe_skill", "range_x", "int", (10, 1500)),
    ("aoe_skill", "range_y", "int", (10, 800)),
    ("monster_detect", "mode", "choice", ["contour_only", "color", "grayscale", "template_free"]),
    ("monster_detect", "diff_thres", "num", (0, 1)),
    ("monster_detect", "search_box_margin", "int", (0, 500)),
    ("edge_teleport", "enable", "bool", None),
    ("watchdog", "timeout", "num", (1, 600)),
    ("detection_lost", "timeout", "num", (0.5, 60)),
    ("system", "fps_limit_main", "int", (1, 30)),
]


def _get(cfg, sec, key):
    v = (cfg.get(sec) or {}).get(key)
    if isinstance(v, tuple):
        v = list(v)
    return v


class WebUI:
    '''
    API for the settings page. control = WingpuControl.
    '''
    def __init__(self, control):
        self.control = control
        self.bot = control.bot
        self.path = default_user_override()
        self.lock = threading.Lock()
        self.data = load_yaml("config/config_data.yaml")

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
        kb = get_keyboard()
        return {
            "values": values,
            "fields": [{"name": f"{s}.{k}", "kind": kind,
                        "choices": extra if kind == "choice" else None,
                        "range": extra if kind in ("num", "int", "floats") else None}
                       for s, k, kind, extra in FIELDS],
            "maps": self.maps(),
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
                elif kind == "keys":
                    v = [str(x).strip().lower() for x in (v or []) if str(x).strip()]
                    bad = [x for x in v if to_key_code(x) is None]
                    if bad:
                        raise ValueError(f"不認得的按鍵：{', '.join(bad)}")
                elif kind in ("num", "int"):
                    v = float(v)
                    if kind == "int":
                        if v != int(v):
                            raise ValueError("要整數")
                        v = int(v)
                    lo, hi = extra
                    if not lo <= v <= hi:
                        raise ValueError(f"範圍 {lo}–{hi}")
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
        m = (clean.get("bot") or {}).get("map") or _get(self.control.cfg_unscaled(), "bot", "map")
        routes = (clean.get("route") or {}).get("files")
        if routes and m in maps:
            bad = [r for r in routes if r not in maps[m]["routes"]]
            if bad:
                errors["route.files"] = f"{m} 沒有 {', '.join(bad)}"
        cfg = self.control.cfg_unscaled()
        bk = (clean.get("buff_skill") or {}).get("keys", _get(cfg, "buff_skill", "keys") or [])
        bc = (clean.get("buff_skill") or {}).get("cooldown", _get(cfg, "buff_skill", "cooldown") or [])
        if len(bk) != len(bc):
            errors["buff_skill.cooldown"] = f"Buff 按鍵 {len(bk)} 個，冷卻時間 {len(bc)} 個，要一樣多"
        return clean, errors

    def save(self, values):
        clean, errors = self.validate(values)
        if errors:
            return {"ok": False, "errors": errors}
        with self.lock:
            over = self.user_override()
            for sec, kv in clean.items():
                over.setdefault(sec, {})
                if not isinstance(over[sec], dict):
                    over[sec] = {}
                over[sec].update(kv)
            self._write(over)
        self.control.pending_restart = True
        logger.info(f"[WebUI] settings saved to {self.path}: {clean}")
        return {"ok": True, "saved": clean, "user_file": self.path}

    def reset(self):
        '''
        Remove the user file's settings (keeps nothing; shipped configs apply)
        '''
        with self.lock:
            if os.path.exists(self.path):
                os.replace(self.path, self.path + ".bak")
        self.control.pending_restart = True
        return {"ok": True, "backup": self.path + ".bak"}

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
