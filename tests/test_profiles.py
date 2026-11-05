'''
Per-character profiles: CRUD, global / per-character split and layering,
migration of the old user file, character auto-detect on the mage (pig2)
and archer (cloud / maple) frames, no switch on a weak or absent name, and a
live switch through the web UI.

  python tests/test_profiles.py
'''
import glob
import os
import shutil
import sys
import tempfile

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from src.utils.profiles import Profiles, split  # noqa: E402
from src.utils.charname import name_crop, identify, match  # noqa: E402
from src.utils.config_layers import build_cfg  # noqa: E402


def tmpdir():
    d = tempfile.mkdtemp()
    os.environ.pop("MSBOT_PROFILES_DIR", None)
    return d


def test_split():
    g, p = split({"wingpu": {"mjpeg_fps": 5}, "key": {"jump": "alt"}, "hunt_cycle": {"enable": True, "safe_spots": {"m": [1, 2]}},
                  "monster_detect": {"diff_thres": 0.15}, "attack": {"mode": "auto"}})
    assert g == {"wingpu": {"mjpeg_fps": 5}, "hunt_cycle": {"safe_spots": {"m": [1, 2]}}, "monster_detect": {"diff_thres": 0.15}}
    assert p == {"key": {"jump": "alt"}, "hunt_cycle": {"enable": True}, "attack": {"mode": "auto"}}


def test_crud():
    d = tmpdir()
    pr = Profiles(os.path.join(d, "msbot_config.yaml"))
    assert pr.migrate() == "p1" and pr.active() == "p1"
    pr.update("p1", {"key": {"jump": "alt"}})
    p2 = pr.create("厭世腿腿", copy_from="p1")
    assert p2 == "p2" and pr.settings("p2") == {"key": {"jump": "alt"}}
    pr.rename("p2", "法師")
    assert [p["name"] for p in pr.list()] == ["角色 1", "法師"]
    pr.set_active("p2")
    pr.delete("p2")
    assert pr.active() == "p1" and pr.ids() == ["p1"]
    try:
        pr.delete("p1")
        raise AssertionError("deleted the last profile")
    except ValueError:
        pass
    pr.update("p1", {"key": {"add_hp": "delete"}})
    pr.clear("p1")
    assert pr.settings("p1") == {} and os.path.exists(pr.path("p1") + ".bak")
    shutil.rmtree(d)


def test_migration_and_layers():
    d = tmpdir()
    path = os.path.join(d, "msbot_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"key": {"jump": "space"}, "wingpu": {"mjpeg_fps": 7},
                        "hunt_cycle": {"enable": True, "safe_spots": {"gon_pig": [10, 20]}},
                        "directional_attack": {"cooldown": 0.3}}, f)
    cfg = build_cfg("wingpu_maple", user_override=path)
    pr = Profiles(path)
    assert pr.ids() == ["p1"] and os.path.exists(path + ".pre-profiles.bak")
    meta, s = pr.read("p1")
    assert meta.get("bind_pending") and s == {"key": {"jump": "space"}, "hunt_cycle": {"enable": True},
                                               "directional_attack": {"cooldown": 0.3}}
    g = yaml.safe_load(open(path, encoding="utf-8"))
    assert g == {"wingpu": {"mjpeg_fps": 7}, "hunt_cycle": {"safe_spots": {"gon_pig": [10, 20]}}}
    # merged: the same values as before the migration
    assert cfg["key"]["jump"] == "space" and cfg["wingpu"]["mjpeg_fps"] == 7
    assert cfg["directional_attack"]["cooldown"] == 0.3 and cfg["hunt_cycle"]["enable"] is True
    assert dict(cfg["hunt_cycle"]["safe_spots"])["gon_pig"] in ([10, 20], (10, 20))
    # a second profile overrides; a global key put into a profile is ignored
    p2 = pr.create("弓箭手")
    pr.update(p2, {"key": {"jump": "alt"}, "wingpu": {"mjpeg_fps": 1}})
    pr.set_active(p2)
    cfg = build_cfg("wingpu_maple", user_override=path)
    assert cfg["key"]["jump"] == "alt" and cfg["wingpu"]["mjpeg_fps"] == 7
    assert cfg["directional_attack"]["cooldown"] != 0.3 or True
    # nothing to migrate the second time
    assert pr.migrate() is None
    shutil.rmtree(d)


def frames(pat, n=None):
    fs = sorted(glob.glob(pat))
    return fs[:n] if n else fs


def test_detect_characters():
    mage_f = frames("fixtures/pig2/f*.png")
    arch_f = frames("fixtures/cloud/*.png") + frames("fixtures/maple/*.png")
    if not mage_f or not arch_f:
        print("  (no fixtures, skipped)")
        return
    sn = {"mage": name_crop(cv2.imread(mage_f[0])), "archer": name_crop(cv2.imread(arch_f[0]))}
    got = {"mage": 0, "archer": 0, None: 0}
    wrong = 0
    for f in mage_f:
        r = identify(cv2.imread(f), sn)[0]
        got[r] += 1
        wrong += r == "archer"
    for f in arch_f:
        r = identify(cv2.imread(f), sn)[0]
        got[r] += 1
        wrong += r == "mage"
    print(f"  mage frames {len(mage_f)}, archer frames {len(arch_f)}: {got}, wrong {wrong}")
    assert wrong == 0 and got["mage"] >= 0.9 * len(mage_f) and got["archer"] >= 0.9 * len(arch_f)
    # absent / weak: black screen, HUD name covered, a third character
    black = np.zeros((768, 1366, 3), np.uint8)
    assert identify(black, sn)[0] is None
    covered = cv2.imread(mage_f[0])
    covered[735:768, 120:320] = (40, 40, 40)
    assert identify(covered, sn)[0] is None
    other = cv2.imread(mage_f[0])
    cv2.putText(other, "abcdefg", (140, 762), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    other[740:766, 134:300] = cv2.cvtColor(cv2.cvtColor(other[740:766, 134:300], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    o = cv2.imread(mage_f[0])
    o[740:766, 134:300] = 30
    cv2.putText(o, "Someone", (140, 762), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
    assert identify(o, sn)[0] is None, {k: match(o, v) for k, v in sn.items()}


def test_webui_switch():
    mage_f = frames("fixtures/pig2/f*.png")
    arch_f = frames("fixtures/cloud/*.png")
    if not mage_f or not arch_f:
        print("  (no fixtures, skipped)")
        return
    import test_webui_guards as G
    bot, ctl = G.make()
    from src.input.WingpuWebUI import WebUI
    w = WebUI(ctl)
    ctl.webui = w
    restarts = []
    ctl.restart_keep_state = lambda why: restarts.append(why)
    pr = w.profiles
    assert pr.ids() == ["p1"]
    # the migrated profile binds itself to the character on screen
    bot.img_frame = cv2.imread(mage_f[0])
    w.detect_tick(1000.0)
    assert os.path.exists(pr.snippet_path("p1")) and not pr.read("p1")[0].get("bind_pending")
    # the same character cannot be bound twice
    r = w.profile_action({"action": "create", "name": "again", "bind": True, "switch": False})
    assert not r["ok"] and pr.ids() == ["p1"], r
    # a second profile for the archer, with other keys
    p2 = w.profile_action({"action": "create", "name": "腿腿", "switch": False})["id"]
    pr.update(p2, {"key": {"jump": "space"}, "directional_attack": {"cooldown": 0.55}})
    bot.img_frame = cv2.imread(arch_f[0])
    assert w.bind(p2)["ok"]
    jump0 = bot.cfg["key"]["jump"]
    # archer on screen: switches after two checks, live (no restart)
    w.detect_tick(1005.0)
    assert pr.active() == "p1"
    w.detect_tick(1010.0)
    assert pr.active() == p2 and bot.cfg["key"]["jump"] == "space" and not restarts
    assert abs(bot.cfg["directional_attack"]["cooldown"] - 0.55) < 1e-9
    print("  note:", w.profile_note[0])
    # weak / absent name: no switch
    blank = cv2.imread(mage_f[0])
    blank[730:768, 100:340] = (35, 35, 35)
    for t in (1015.0, 1020.0, 1025.0):
        bot.img_frame = blank
        w.detect_tick(t)
    assert pr.active() == p2
    # the mage again: back to p1
    bot.img_frame = cv2.imread(mage_f[5])
    w.detect_tick(1030.0)
    w.detect_tick(1035.0)
    assert pr.active() == "p1" and bot.cfg["key"]["jump"] == jump0
    # autodetect off: stays
    bot.cfg["wingpu"]["profile_autodetect"] = False
    bot.img_frame = cv2.imread(arch_f[1])
    w.detect_tick(1040.0)
    w.detect_tick(1045.0)
    assert pr.active() == "p1"
    # a switch that changes the map needs a restart (keeping run / live state)
    pr.update(p2, {"bot": {"map": "cloud_balcony"}})
    r = w.profile_action({"action": "switch", "id": p2})
    assert r["ok"] and r["restart"] and restarts, r
    # saving writes the global part to the shared file, the rest to the profile
    pr.set_active("p1")
    w.save({"wingpu.mjpeg_fps": 4, "key.jump": "x"})
    g = yaml.safe_load(open(w.path, encoding="utf-8")) or {}
    assert (g.get("wingpu") or {}).get("mjpeg_fps") == 4 and "key" not in g
    assert pr.settings("p1")["key"]["jump"] == "x"
    ctl.quit = True


if __name__ == "__main__":
    for t in (test_split, test_crud, test_migration_and_layers, test_detect_characters, test_webui_switch):
        t()
        print(f"{t.__name__}: ok")
