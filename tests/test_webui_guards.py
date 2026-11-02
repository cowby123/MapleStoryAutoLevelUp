'''
Web UI settings: feature switches, defaults, and the live-input guards.
Offline: a bot built from saved frames, the real HTTP server on localhost,
a fake VM (the keyboard's dry_run flag is flipped in-process; nothing reaches
a guest).

  python tests/test_webui_guards.py
'''
import json
import logging
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import cv2  # noqa: E402
import wingpu_detect_test as T  # noqa: E402
from src.input.WingpuKeyboard import get_keyboard  # noqa: E402
from src.utils.config_layers import build_cfg  # noqa: E402

T.logger.set_level(logging.WARNING)
TMP = tempfile.mkdtemp()
os.environ["MSBOT_USER_CONFIG"] = os.path.join(TMP, "msbot_config.yaml")
PORT = 18765


class FakeVM:
    def __init__(self):
        self.msgs = []

    def _call(self, msg):
        self.msgs.append(msg)
        return {"ok": True}


def make():
    T.OVERRIDES.clear()
    T.OVERRIDES.update({
        "player_detect.nametag_template": os.path.join(TMP, "nametag", "nametag.png"),
        "wingpu.debug_dir": os.path.join(TMP, "debug"),
        "wingpu.control_file": os.path.join(TMP, "control.json"),
        "wingpu.http_port": PORT,
    })
    bot = T.make_bot("wingpu_maple")
    bot.nametag = None  # make_bot always builds one; the bot itself only does for nametag/auto
    img = cv2.imread(sorted(T.glob.glob("fixtures/maple2/*.png"))[2])
    bot.capture.set_guest(img)
    bot.run_once()
    from src.input.WingpuControl import WingpuControl
    ctl = WingpuControl(bot, bot.cfg)
    return bot, ctl


def post(path, body, test_header=False):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"X-MSBot-Test": "1"} if test_header else {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_defaults():
    cfg = build_cfg("wingpu_maple", user_override=None)
    assert cfg["player_detect"]["mode"] == "party_bar"
    assert cfg["hunt_cycle"]["enable"] is False and cfg["hunt_cycle"]["rest_action"] == "stop"
    assert cfg["rune_find"]["enable"] is False
    assert cfg["watchdog"]["last_attack_enable"] is False
    assert cfg["scheduled_channel_switching"]["enable"] is False
    assert cfg["channel_change"]["enable"] is False
    assert cfg["health_monitor"]["return_home_if_no_potion"] is False
    assert cfg["wingpu"]["ensure_party_on_start"] is False
    assert cfg["wingpu"]["dry_run"] is True


def test_all(bot_ctl=[]):
    bot, ctl = make()
    kb = get_keyboard()
    kb.vm = FakeVM()
    ui = ctl.webui
    try:
        # party bar only: no nametag tracker, the tool still works
        assert bot.nametag is None
        st = ui.nametag_status()
        assert st["in_use"] is False and st["exists"] is False
        # nametag mode needs a template
        r = ui.save({"player_detect.mode": "nametag"})
        assert not r["ok"] and "截取名牌" in r["errors"]["player_detect.mode"], r
        snap = ui.nametag_snapshot()
        rect = snap["proposal"] or [710, 539, 33, 17]
        assert ui.nametag_save({"id": snap["id"], "rect": rect})["ok"]
        r = ui.save({"player_detect.mode": "auto"})
        assert r["ok"] and bot.nametag is not None and bot.nametag.loaded, r
        r = ui.save({"player_detect.mode": "party_bar"})
        assert r["ok"] and bot.nametag is None
        print("  nametag: off by default, needs a template to switch on, tracker loaded/dropped live")

        # features summary
        feats = {f["id"]: f["on"] for f in ctl.features()}
        assert feats["nametag"] is False and feats["hunt_cycle"] is False and feats["rune"] is False
        assert feats["last_attack"] is False and feats["scheduled_channel"] is False
        print("  features on:", [f["cn"] for f in ctl.features() if f["on"]])

        # dry run: short cycles allowed, test header allowed
        assert kb.dry_run
        r = ui.save({"hunt_cycle.enable": False, "hunt_cycle.hunt_minutes": [0.5, 0.5]})
        assert r["ok"], r
        code, r = post("api/config", {"values": {"hunt_cycle.rest_minutes": [0.5, 1]}}, test_header=True)
        assert code == 200 and r["ok"], (code, r)

        # live input (flag only; FakeVM): short cycles and test requests are refused
        kb.dry_run = False
        r = ui.save({"hunt_cycle.hunt_minutes": [0.5, 0.5]})
        assert not r["ok"] and "1 分鐘" in r["errors"]["hunt_cycle.hunt_minutes"], r
        r = ui.save({"hunt_cycle.hunt_minutes": [1, 2]})
        assert r["ok"], r
        code, r = post("api/config", {"values": {"hunt_cycle.rest_minutes": [5, 10]}}, test_header=True)
        assert code == 403 and "測試" in r["errors"]["test"], (code, r)
        code, r = post("pause", {}, test_header=True)
        assert code == 403
        code, r = post("api/config", {"values": {"hunt_cycle.rest_minutes": [5, 10]}})
        assert code == 200 and r["ok"]            # the user's own saves still work
        print("  live: <1 min cycles refused, X-MSBot-Test requests refused (403), normal saves ok")
        assert kb.vm.msgs == [], kb.vm.msgs        # nothing was sent while testing
    finally:
        kb.dry_run = True
        kb.vm = None
        ctl.stop()


if __name__ == "__main__":
    try:
        test_defaults()
        print("ok   test_defaults")
        test_all()
        print("ok   test_all")
        print("2 tests passed")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
