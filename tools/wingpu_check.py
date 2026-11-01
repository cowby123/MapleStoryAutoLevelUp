'''
wingpu_check.py - offline checks of the wingpu backend (no VM, no input sent).

  WINGPU=1 python tools/wingpu_check.py

1. key mapping: every key used in config/*.yaml and in the code maps to a
   KeyboardEvent.code the guest input path knows
2. capturor: 1:1 crop / window->guest click mapping on synthetic guest
   screens; resolution scaling of config values
3. dry-run guard: every input path (KeyBoardController key_down/key_up/
   press_key/release_all_key, HealthMonitor/RuneSolver press_key,
   click_in_game_window) reaches the VM zero times in dry run, and the right
   messages in live mode (against a fake VM)
4. static scan: every function that calls pyautogui / win32gui / pygetwindow /
   Quartz is either guarded by is_wingpu() or never runs on wingpu
'''
# Standard import
import ast
import glob
import os
import sys
import tempfile

os.environ.setdefault("WINGPU", "1")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Library import
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# Local import
from src.utils.common import load_yaml, override_cfg, is_wingpu  # noqa: E402
from src.input import WingpuKeyboard as WK  # noqa: E402
from src.input.WingpuKeyboard import to_key_code, get_keyboard  # noqa: E402

# Codes the wingpu input path accepts (wingpu repo tools/stream/stream_proto.py QCODE)
QCODES = set("""Escape Enter Space Backspace Tab ShiftLeft ShiftRight ControlLeft ControlRight
AltLeft AltRight MetaLeft MetaRight ContextMenu CapsLock NumLock ScrollLock PrintScreen Pause
ArrowUp ArrowDown ArrowLeft ArrowRight Insert Delete Home End PageUp PageDown Minus Equal
BracketLeft BracketRight Backslash Semicolon Quote Backquote Comma Period Slash IntlBackslash
NumpadAdd NumpadSubtract NumpadMultiply NumpadDivide NumpadEnter NumpadDecimal""".split())
QCODES |= {"Key" + c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
QCODES |= {f"Digit{d}" for d in range(10)} | {f"Numpad{d}" for d in range(10)}
QCODES |= {f"F{f}" for f in range(1, 13)}

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def base_cfg():
    cfg = load_yaml("config/config_default.yaml")
    return override_cfg(cfg, load_yaml("config/config_wingpu.yaml"))


#################
### 1. keys ###
#################
def check_keys():
    print("1. key mapping")
    check(all(v in QCODES for v in WK.KEY_MAP.values()),
          "every KEY_MAP value is a code the guest input path knows")
    used = {}
    for path in sorted(glob.glob("config/config_*.yaml")):
        with open(path, encoding="utf-8") as f:
            c = yaml.safe_load(f) or {}
        for k, v in (c.get("key") or {}).items():
            if v not in (None, ""):
                used.setdefault(str(v), set()).add(f"{os.path.basename(path)}:key.{k}")
        for v in (c.get("buff_skill") or {}).get("keys") or []:
            used.setdefault(str(v), set()).add(f"{os.path.basename(path)}:buff_skill.keys")
    for v in ("left", "right", "up", "down"):  # KeyBoardController, near_rune, RuneSolver
        used.setdefault(v, set()).add("code")
    for v, where in sorted(used.items()):
        code = to_key_code(v)
        check(code in QCODES, f"'{v}' -> {code}  ({len(where)} uses)")
    for name in ("pageup", "pagedown", "insert", "delete", "home", "end", "esc", "enter",
                 "ctrl", "shift", "f1", "f12", "a", "0", "9", "num5", "tab", "backspace"):
        check(to_key_code(name) in QCODES, f"common key '{name}' -> {to_key_code(name)}")
    check(to_key_code("") is None and to_key_code(None) is None, "empty key -> None (skipped)")
    check(to_key_code("ArrowLeft") == "ArrowLeft", "code names pass through")


######################
### 2. capturor ###
######################
def check_capturor():
    print("2. capturor crop / click mapping (1:1, no resize) and resolution scaling")
    from src.input.GameWindowCapturorForWingpu import WingpuCapturor, detect_client_rect
    from src.utils.resolution import compute_scale, apply_scale, snapshot_base, map_point

    cfg = base_cfg()
    tmp = tempfile.mkdtemp()

    # (a) fullscreen 1366x768: the frame IS the guest screen, clicks 1:1
    g = np.zeros((768, 1366, 3), np.uint8)
    g[:, :, 0] = (np.arange(1366) * 255 // 1365)[None, :]
    g[:, :, 1] = (np.arange(768) * 255 // 767)[:, None]
    p = os.path.join(tmp, "full.png")
    cv2.imwrite(p, g)
    cap = WingpuCapturor(cfg, p)
    f = cap.get_frame()
    check(cap.rect.method == "fullscreen", f"fullscreen detected: {cap.rect}")
    check(f.shape == (768, 1366, 3) and np.array_equal(f, g), f"frame = guest screen 1:1 {f.shape}")
    check(cap.window_to_guest(0, 0) == (0, 0) and cap.window_to_guest(683, 384) == (683, 384),
          "client (x, y) -> guest (x, y)")

    # (b) fixed windowed client: exact crop, clicks offset by the client origin
    cfg2 = override_cfg(base_cfg(), {"wingpu": {"client_rect": (300, 200, 1282, 693)}})
    rng = np.random.default_rng(1)
    g2 = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    p2 = os.path.join(tmp, "win.png")
    cv2.imwrite(p2, g2)
    cap2 = WingpuCapturor(cfg2, p2)
    f2 = cap2.get_frame()
    check(cap2.rect.method == "fixed", f"fixed rect {cap2.rect}")
    check(np.array_equal(f2, g2[200:200 + 693, 300:300 + 1282]), "client copied 1:1 (no resize, no title band)")
    check(cap2.window_to_guest(1140, 671) == (1440, 871), "client (1140,671) -> guest (1440,871)")

    # (c) title template finds a moved window
    tpl = g2[200 - 30:200, 300:660].copy()
    cfg3 = override_cfg(base_cfg(), {"wingpu": {"title_to_client": (0, 30),
                                                "client_size_guest": (1282, 693)}})
    g3 = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    g3[150 - 30:150, 500:860] = tpl
    r = detect_client_rect(g3, cfg3, tpl)
    check((r.x, r.y, r.w, r.h) == (500, 150, 1282, 693) and r.method.startswith("template"),
          f"title template -> {r}")
    r = detect_client_rect(rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8), cfg3, tpl)
    check(r.method == "fullscreen", f"no template match -> fallback {r.method}")

    # (d) resolution scaling: identity at the reference size, proportional elsewhere
    check(compute_scale(700, 1296) == 1.0, "scale 1.0 at the reference 1296x700")
    check(abs(compute_scale(768, 1366) - 1366 / 1296) < 1e-9, "auto scale = width / 1296")
    check(map_point((648, 350), 1.0, (700, 1296), "cc") == (648, 350), "map_point identity at s=1")
    c = base_cfg()
    base = snapshot_base(c)
    apply_scale(c, base, 1366 / 1296, (768, 1366))
    check(c["directional_attack"]["range_x"] == round(350 * 1366 / 1296), "range_x scaled")
    check(c["ui_coords"]["ui_y_start"] == round(768 - 90 * 1366 / 1296), "ui_y_start bottom anchored")
    apply_scale(c, base, compute_scale(1080, 1920), (1080, 1920))
    check(c["directional_attack"]["range_x"] == round(350 * 1920 / 1296), "rescaled from the base, not compounded")


####################
### 3. dry run ###
####################
class FakeVM:
    def __init__(self):
        self.msgs = []

    def _call(self, msg):
        self.msgs.append(msg)
        return {"ok": True}


def check_dry_run():
    print("3. dry-run guard")
    check(is_wingpu(), "platform is wingpu (WINGPU=1)")
    from src.input import KeyBoardController as KBC
    from src.engine import HealthMonitor as HM
    from src.engine import RuneSolver as RS
    from src.states import near_rune as NR
    from src.utils.common import click_in_game_window
    from src.input.GameWindowCapturorForWingpu import ClientRect

    cfg = base_cfg()
    kb = get_keyboard()
    fake = FakeVM()
    kb.configure(cfg, live=False)
    kb.vm = fake  # even with a VM handle, dry run must not call it
    kb.set_click_mapper(ClientRect(0, 0, 1920, 1080, "test").window_to_guest)

    def exercise():
        KBC.key_down("left"); KBC.key_down("left"); KBC.key_up("left"); KBC.key_up("left")
        KBC.press_key("z", 0.0); KBC.press_key("", 0.0); KBC.press_key("alt", 0.0)
        HM.press_key("end", 0.0); RS.press_key("up", 0.0); NR.press_key("up", 0.0)
        KBC.key_down("down"); kb.release_all()
        click_in_game_window("MapleStory Worlds", (1140, 730))

    n_hist = len(kb.history)
    exercise()
    check(fake.msgs == [], f"dry run: 0 messages reached the VM ({len(kb.history) - n_hist} actions logged)")
    check(any("click" in s for _, s in kb.history), "dry run: click was logged")

    # live mode against the fake VM: the right messages, deduplicated key state
    kb.dry_run = False
    exercise()
    kb.dry_run = True
    keys = [(m["code"], m["down"]) for m in fake.msgs if m["t"] == "key"]
    check(keys[:2] == [("ArrowLeft", True), ("ArrowLeft", False)],
          f"live: repeated key_down/key_up de-duplicated {keys[:2]}")
    check(("KeyZ", True) in keys and ("AltLeft", True) in keys and ("End", True) in keys,
          "live: z / alt / end mapped")
    check(("ArrowDown", False) in keys, "live: release_all released ArrowDown")
    clicks = [m for m in fake.msgs if m["t"] == "click"]
    check(len(clicks) == 1 and (clicks[0]["x"], clicks[0]["y"]) == (1140, 730),
          f"live: click (1140,730) -> guest {[(c['x'], c['y']) for c in clicks]}")
    check(all(m["t"] in ("key", "click") for m in fake.msgs), "live: only key/click messages")


#######################
### 4. static scan ###
#######################
# functions that call desktop automation APIs but never run on wingpu, and why
NOT_ON_WINGPU = {
    "src/input/KeyBoardController.py:recover_mouse": "only after pyautogui.FailSafeException (non-wingpu branch)",
    "src/utils/common.py:get_window_region_mac": "macOS only (click_in_game_window returns earlier on wingpu)",
    "src/utils/common.py:callback": "nested in get_game_window_title_by_token, which returns first on wingpu",
    "src/input/KeyBoardListener.py:*": "imported only on the non-headless path (main), never on wingpu",
    "src/input/GameWindowCapturor.py:*": "Windows capturor, replaced by WingpuCapturor",
    "src/input/GameWindowCapturorForMac.py:*": "macOS capturor, replaced by WingpuCapturor",
}
APIS = ("pyautogui", "win32gui", "gw", "Quartz", "WindowsCapture", "mss")


def check_static():
    print("4. static scan of src/ for desktop input/capture APIs")
    for path in sorted(glob.glob("src/**/*.py", recursive=True)):
        rel = path.replace(os.sep, "/")
        if "/legacy/" in rel or "/ui/" in rel:
            continue  # legacy code and the PySide6 UI are not used by the headless bot
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = sorted({n.value.id for n in ast.walk(node)
                            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                            and n.value.id in APIS})
            if not calls:
                continue
            key = f"{rel}:{node.name}"
            body = ast.get_source_segment(src, node)
            if "is_wingpu()" in body:
                check(True, f"{key} uses {calls}: guarded by is_wingpu()")
            elif key in NOT_ON_WINGPU or f"{rel}:*" in NOT_ON_WINGPU:
                why = NOT_ON_WINGPU.get(key) or NOT_ON_WINGPU[f"{rel}:*"]
                check(True, f"{key} uses {calls}: {why}")
            elif node.name in ("is_game_window_active",):
                check("is_wingpu()" in body, f"{key}")
            else:
                check(False, f"{key} uses {calls} without an is_wingpu() guard")


if __name__ == "__main__":
    check_keys()
    check_capturor()
    check_dry_run()
    check_static()
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)
