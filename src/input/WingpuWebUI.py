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
from src.utils.config_layers import default_user_override
from src.input.WingpuKeyboard import get_keyboard, to_key_code
from src.utils.nametag import propose_rect, save_template, NametagTracker

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
    ("aoe_skill", "range_x", "int", (10, 1500)),
    ("aoe_skill", "range_y", "int", (10, 800)),
    ("monster_detect", "mode", "choice", ["contour_only", "color", "grayscale", "template_free"]),
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
]


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


    ###############
    ### Nametag ###
    ###############
    def nametag_status(self):
        tr = getattr(self.bot, "nametag", None)
        st = tr.status() if tr is not None else {"loaded": False}
        st["exists"] = bool(tr is not None and os.path.exists(tr.path))
        st["mtime"] = int(os.path.getmtime(tr.path)) if st["exists"] else None
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
        path = tr.path if tr is not None else None
        out_dir, name = os.path.split(path) if path else (None, "nametag.png")
        try:
            saved, meta = save_template(img, rect, scale, player, out_dir=out_dir,
                                        name=os.path.splitext(name)[0])
        except ValueError as e:
            return {"ok": False, "errors": {"nametag": str(e)}}
        # check: does the new template find itself in that frame?
        test = NametagTracker(saved, thres=tr.thres if tr is not None else 0.3)
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
        tr = getattr(self.bot, "nametag", None)
        if tr is None or not os.path.exists(tr.path):
            return None
        with open(tr.path, "rb") as f:
            return f.read()

    def nametag_delete(self):
        tr = getattr(self.bot, "nametag", None)
        if tr is not None and os.path.exists(tr.path):
            os.replace(tr.path, tr.path + ".bak")
            tr.reload_if_changed()
        return {"ok": True}


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
