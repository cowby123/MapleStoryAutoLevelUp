'''
Per-character settings profiles (wingpu).

  /work/msbot_config.yaml        shared (global) settings + map data
  /work/profiles/<id>.yaml       one character's settings (_profile: name, bound)
  /work/profiles/<id>_name.png   the character's HUD name row (auto-detect)
  /work/profiles/active.txt      the active profile id

Config layers: shipped configs -> msbot_config.yaml -> the active profile.
What is global (GLOBAL below): the bot / stream / input settings (wingpu.*,
system.*), the detector tuning (monster_detect.*, detection_lost.*) and the
map data (hunt_cycle.safe_spots). Everything else is per character: keys,
attack, AoE, buffs, potions, map / routes in use, player detection,
hunt cycle, 類人保護, rune options.
'''
# Standard import
import copy
import os
import re
import shutil
import time

# Library import
import cv2
import yaml

GLOBAL_SECTIONS = ("wingpu", "system", "monster_detect", "detection_lost")
GLOBAL_KEYS = {("hunt_cycle", "safe_spots")}
META = "_profile"


def is_global(sec, key=None):
    return sec in GLOBAL_SECTIONS or (sec, key) in GLOBAL_KEYS


def split(over):
    '''user-override dict -> (global part, profile part)'''
    g, p = {}, {}
    for sec, val in (over or {}).items():
        if sec == META:
            continue
        if not isinstance(val, dict):
            (g if is_global(sec) else p)[sec] = copy.deepcopy(val)
            continue
        for key, v in val.items():
            (g if is_global(sec, key) else p).setdefault(sec, {})[key] = copy.deepcopy(v)
    return g, p


def profiles_dir(user_override_path):
    env = os.environ.get("MSBOT_PROFILES_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(user_override_path)), "profiles")


def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _dump(path, data, header=""):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)
    os.replace(tmp, path)


class Profiles:
    def __init__(self, user_override_path):
        self.global_path = user_override_path
        self.dir = profiles_dir(user_override_path)

    # ---------- files ----------
    def path(self, pid):
        return os.path.join(self.dir, f"{pid}.yaml")

    def snippet_path(self, pid):
        return os.path.join(self.dir, f"{pid}_name.png")

    def ids(self):
        if not os.path.isdir(self.dir):
            return []
        out = [f[:-5] for f in os.listdir(self.dir) if re.fullmatch(r"p\d+\.yaml", f)]
        return sorted(out, key=lambda s: int(s[1:]))

    def exists(self):
        return bool(self.ids())

    def active(self):
        ids = self.ids()
        try:
            with open(os.path.join(self.dir, "active.txt"), "r", encoding="utf-8") as f:
                a = f.read().strip()
            if a in ids:
                return a
        except OSError:
            pass
        return ids[0] if ids else None

    def set_active(self, pid):
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "active.txt"), "w", encoding="utf-8") as f:
            f.write(pid)

    def read(self, pid):
        d = _load(self.path(pid))
        meta = d.pop(META, {}) or {}
        return meta, d

    def write(self, pid, meta, settings):
        os.makedirs(self.dir, exist_ok=True)
        data = copy.deepcopy(settings)
        data[META] = meta
        _dump(self.path(pid), data, "# MapleStoryAutoLevelUp: one character's settings (web UI)\n")

    def settings(self, pid):
        return self.read(pid)[1] if pid else {}

    def list(self):
        act = self.active()
        out = []
        for pid in self.ids():
            meta, _ = self.read(pid)
            out.append({"id": pid, "name": meta.get("name") or pid, "active": pid == act,
                        "bound": os.path.exists(self.snippet_path(pid)),
                        "bind_pending": bool(meta.get("bind_pending"))})
        return out

    # ---------- CRUD ----------
    def new_id(self):
        n = max([int(i[1:]) for i in self.ids()] or [0]) + 1
        return f"p{n}"

    def create(self, name, copy_from=None):
        pid = self.new_id()
        settings = self.settings(copy_from) if copy_from else {}
        self.write(pid, {"name": name or f"角色 {pid[1:]}", "created": time.strftime("%Y-%m-%d %H:%M:%S")}, settings)
        return pid

    def rename(self, pid, name):
        meta, s = self.read(pid)
        meta["name"] = str(name)[:24]
        self.write(pid, meta, s)

    def delete(self, pid):
        ids = self.ids()
        if pid not in ids or len(ids) <= 1:
            raise ValueError("至少要留一個設定檔")
        os.replace(self.path(pid), self.path(pid) + ".deleted")
        if os.path.exists(self.snippet_path(pid)):
            os.replace(self.snippet_path(pid), self.snippet_path(pid) + ".deleted")
        if self.active() == pid or self.active() is None:
            self.set_active([i for i in ids if i != pid][0])

    def update(self, pid, part):
        '''merge a {section: {key: value}} part into a profile'''
        meta, s = self.read(pid)
        for sec, kv in part.items():
            if isinstance(kv, dict):
                if not isinstance(s.get(sec), dict):
                    s[sec] = {}
                s[sec].update(copy.deepcopy(kv))
            else:
                s[sec] = copy.deepcopy(kv)
        self.write(pid, meta, s)

    def clear(self, pid):
        '''「清除我的設定」: the profile's settings go (backup kept), its name / binding stay'''
        meta, s = self.read(pid)
        shutil.copyfile(self.path(pid), self.path(pid) + ".bak")
        self.write(pid, meta, {})

    # ---------- binding ----------
    def save_snippet(self, pid, crop):
        os.makedirs(self.dir, exist_ok=True)
        cv2.imwrite(self.snippet_path(pid), crop)
        meta, s = self.read(pid)
        meta.pop("bind_pending", None)
        meta["bound"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.write(pid, meta, s)

    def snippets(self):
        out = {}
        for pid in self.ids():
            p = self.snippet_path(pid)
            if os.path.exists(p):
                out[pid] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        return out

    # ---------- migration ----------
    def migrate(self, name="角色 1"):
        '''
        First start with profiles: the user file's per-character part becomes
        profile p1 (bound to the character on screen at the first good frame);
        the file keeps only the global part. Backup: <file>.pre-profiles.bak
        Returns the new id or None (nothing to do).
        '''
        if self.exists():
            return None
        over = _load(self.global_path) if os.path.exists(self.global_path) else {}
        g, p = split(over)
        if os.path.exists(self.global_path):
            shutil.copyfile(self.global_path, self.global_path + ".pre-profiles.bak")
            _dump(self.global_path, g, "# MapleStoryAutoLevelUp shared settings (web UI); per-character ones: profiles/\n")
        self.write("p1", {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "bind_pending": True}, p)
        self.set_active("p1")
        return "p1"
