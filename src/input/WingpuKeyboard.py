'''
WingpuKeyboard
Keyboard / mouse output for the wingpu platform: the bot runs in the VM's script
container (Linux) and sends input to the Windows guest through the wingpu SDK
instead of pyautogui.

- Key names are pyautogui names (what config/*.yaml uses: "left", "alt", "z",
  "del", "pageup", ...). KEY_MAP turns them into KeyboardEvent.code names
  ("ArrowLeft", "AltLeft", "KeyZ", "Delete", "PageUp", ...), which is what the
  wingpu input socket understands.
- DRY RUN (default ON): nothing is sent to the guest. Every action the bot WOULD
  send is logged ("[DRY-RUN] ...") and kept in a short history for the debug
  status file. Turn it off only with cfg wingpu.dry_run: false or --wingpu_live.
- Key state is tracked: key_down of a key that is already down and key_up of a
  key that is already up are not re-sent (KeyBoardController calls them on every
  30 Hz loop; the guest only needs the state change).
'''
# Standard import
import collections
import threading
import time

# Local import
from src.utils.logger import logger

#########################
### pyautogui -> code ###
#########################
KEY_MAP = {
    # arrows
    "left": "ArrowLeft", "right": "ArrowRight", "up": "ArrowUp", "down": "ArrowDown",
    # modifiers
    "ctrl": "ControlLeft", "ctrlleft": "ControlLeft", "ctrlright": "ControlRight",
    "control": "ControlLeft",
    "alt": "AltLeft", "altleft": "AltLeft", "altright": "AltRight",
    "option": "AltLeft", "optionleft": "AltLeft", "optionright": "AltRight",
    "shift": "ShiftLeft", "shiftleft": "ShiftLeft", "shiftright": "ShiftRight",
    "win": "MetaLeft", "winleft": "MetaLeft", "winright": "MetaRight",
    "command": "MetaLeft", "apps": "ContextMenu",
    # whitespace / editing
    "space": "Space", " ": "Space",
    "enter": "Enter", "return": "Enter", "\n": "Enter", "\r": "Enter",
    "tab": "Tab", "\t": "Tab",
    "backspace": "Backspace", "\b": "Backspace",
    "esc": "Escape", "escape": "Escape",
    "insert": "Insert", "ins": "Insert",
    "delete": "Delete", "del": "Delete",
    "home": "Home", "end": "End",
    "pageup": "PageUp", "pgup": "PageUp",
    "pagedown": "PageDown", "pgdn": "PageDown",
    # locks / misc
    "capslock": "CapsLock", "numlock": "NumLock", "scrolllock": "ScrollLock",
    "printscreen": "PrintScreen", "prtsc": "PrintScreen", "prtscr": "PrintScreen",
    "prntscrn": "PrintScreen", "print": "PrintScreen", "pause": "Pause",
    # punctuation (US layout)
    "-": "Minus", "=": "Equal", "[": "BracketLeft", "]": "BracketRight",
    "\\": "Backslash", ";": "Semicolon", "'": "Quote", "`": "Backquote",
    ",": "Comma", ".": "Period", "/": "Slash",
    # numpad
    "add": "NumpadAdd", "subtract": "NumpadSubtract", "multiply": "NumpadMultiply",
    "divide": "NumpadDivide", "decimal": "NumpadDecimal",
}
for _c in "abcdefghijklmnopqrstuvwxyz":
    KEY_MAP[_c] = "Key" + _c.upper()
for _d in range(10):
    KEY_MAP[str(_d)] = f"Digit{_d}"
    KEY_MAP[f"num{_d}"] = f"Numpad{_d}"
for _f in range(1, 13):  # the guest input path knows F1..F12
    KEY_MAP[f"f{_f}"] = f"F{_f}"


def key_option_groups():
    '''
    Keys the web UI offers in its key dropdowns, grouped, with friendly names
    (the game's hotbar abbreviations in parentheses). Every value is a KEY_MAP
    name. [(group, [(value, label), ...]), ...]
    '''
    groups = [
        ("字母 A–Z", [(c, c.upper()) for c in "abcdefghijklmnopqrstuvwxyz"]),
        ("數字 0–9", [(str(d), str(d)) for d in range(10)]),
        ("功能鍵 F1–F12", [(f"f{i}", f"F{i}") for i in range(1, 13)]),
        ("控制鍵", [("ctrl", "Ctrl（左）"), ("ctrlright", "Ctrl（右）"), ("shift", "Shift（左）"),
                  ("shiftright", "Shift（右）"), ("alt", "Alt（左）"), ("altright", "Alt（右）"),
                  ("space", "Space（空白鍵）"), ("enter", "Enter"), ("tab", "Tab"), ("esc", "Esc"),
                  ("backspace", "Backspace（退格）")]),
        ("編輯鍵", [("insert", "Insert（Ins）"), ("delete", "Delete（Del）"), ("home", "Home（Hm）"),
                  ("end", "End"), ("pageup", "PageUp（Pup）"), ("pagedown", "PageDown（Pdn）")]),
        ("方向鍵", [("left", "←（左）"), ("right", "→（右）"), ("up", "↑（上）"), ("down", "↓（下）")]),
        ("符號鍵", [("`", "` （波浪號鍵）"), ("-", "-（減號）"), ("=", "=（等號）"), ("[", "["), ("]", "]"),
                  ("\\", "\\（反斜線）"), (";", ";（分號）"), ("'", "'（引號）"), (",", ",（逗號）"),
                  (".", ".（句號）"), ("/", "/（斜線）")]),
        ("數字鍵盤", [(f"num{d}", f"數字鍵盤 {d}") for d in range(10)] +
         [("add", "數字鍵盤 +"), ("subtract", "數字鍵盤 -"), ("multiply", "數字鍵盤 *"),
          ("divide", "數字鍵盤 /"), ("decimal", "數字鍵盤 .")]),
    ]
    return [(g, [(v, lab) for v, lab in keys if v in KEY_MAP]) for g, keys in groups]


def to_key_code(key):
    '''
    pyautogui key name -> KeyboardEvent.code, or None if unknown/empty.
    A KeyboardEvent.code name itself ("ArrowLeft", "KeyZ") is passed through.
    '''
    if key is None:
        return None
    if isinstance(key, int):
        key = str(key)
    if key in KEY_MAP.values():
        return key
    if len(key) == 1:
        return KEY_MAP.get(key.lower(), KEY_MAP.get(key))
    return KEY_MAP.get(key.strip().lower())


class WingpuKeyboard:
    '''
    Input sink for the wingpu platform (process-wide singleton: get_keyboard()).
    '''
    def __init__(self):
        self.dry_run = True # Never send anything until configure(live=True)
        self.cfg = None
        self.vm = None
        self.lock = threading.Lock()
        self.pressed = set() # KeyboardEvent.code currently held down
        self.history = collections.deque(maxlen=200) # (time, text)
        self.counts = collections.Counter()
        self.unknown_keys = set()
        self.click_mapper = None # window coord -> guest coord (set by WingpuCapturor)
        self.blocked_logged = 0.0 # last time "input blocked" was logged

    def configure(self, cfg, live=False):
        '''
        live=True is the only way to turn dry run off.
        '''
        self.cfg = cfg
        self.dry_run = not live
        if self.dry_run:
            logger.info("[WingpuKeyboard] DRY RUN: no key/mouse input will be sent to the guest")
        else:
            logger.warning("[WingpuKeyboard] LIVE: key/mouse input WILL be sent to the guest")
            from src.input.WingpuVM import get_vm  # pylint: disable=import-outside-toplevel
            self.vm = get_vm(cfg)

    def set_click_mapper(self, mapper):
        '''
        mapper(x, y) -> (guest_x, guest_y) or None (client rect unknown)
        '''
        self.click_mapper = mapper

    def _record(self, text):
        self.history.append((time.time(), text))
        self.counts[text.split(" ")[0]] += 1
        if self.dry_run:
            logger.info(f"[DRY-RUN] {text}")
        else:
            logger.debug(f"[WingpuKeyboard] {text}")

    def _send(self, msg):
        '''
        The only place that talks to the guest. Gated by dry_run.
        True = the guest got it (or dry run).
        '''
        if self.dry_run:
            return True
        try:
            self.vm._call(msg)  # pylint: disable=protected-access
            return True
        except Exception as e:
            if str(e).startswith("input blocked"):
                # the manager page's 禁止送輸入 switch: every request fails until it is off; log it now and then
                if time.time() - self.blocked_logged > 30:
                    self.blocked_logged = time.time()
                    logger.warning(f"[WingpuKeyboard] input blocked by the manager page (禁止送輸入): {msg}")
            else:
                logger.error(f"[WingpuKeyboard] input failed {msg}: {e}")
            return False

    def _code(self, key):
        code = to_key_code(key)
        if code is None and key not in ("", None) and key not in self.unknown_keys:
            self.unknown_keys.add(key)
            logger.error(f"[WingpuKeyboard] No KeyboardEvent.code for key '{key}' - ignored")
        return code

    def key_down(self, key):
        code = self._code(key)
        if code is None:
            return
        with self.lock:
            if code in self.pressed:
                return
            self.pressed.add(code)
            self._record(f"key_down {code} ('{key}')")
            if not self._send({"t": "key", "code": code, "down": True}):
                self.pressed.discard(code) # not down in the guest: press it again next time

    def key_up(self, key, force=False):
        code = self._code(key)
        if code is None:
            return
        with self.lock:
            if code not in self.pressed and not force:
                return
            self.pressed.discard(code)
            self._record(f"key_up {code} ('{key}')")
            self._send({"t": "key", "code": code, "down": False})

    def press_key(self, key, duration=0.05):
        code = self._code(key)
        if code is None:
            return
        with self.lock:
            self._record(f"press {code} ('{key}') {duration:.2f}s")
            self.pressed.add(code)
            self._send({"t": "key", "code": code, "down": True})
        time.sleep(duration)
        with self.lock:
            # Like pyautogui: the key is up afterwards, even if it was held before
            self.pressed.discard(code)
            self._send({"t": "key", "code": code, "down": False})

    def release_all(self):
        '''
        Release every key we believe is held.
        '''
        for code in list(self.pressed):
            self.key_up(code)

    def click(self, coord):
        '''
        coord: game window coordinate (title bar included), like
        utils.common.click_in_game_window on Windows.
        '''
        if self.click_mapper is None:
            logger.error("[WingpuKeyboard] click: no capturor (client rect unknown)")
            return
        guest = self.click_mapper(coord[0], coord[1])
        if guest is None:
            logger.error(f"[WingpuKeyboard] click {coord}: client rect unknown - skipped")
            return
        gx, gy = guest
        with self.lock:
            self._record(f"click window{tuple(int(v) for v in coord)} -> guest({gx},{gy})")
            self._send({"t": "click", "x": gx, "y": gy, "button": "left"})

    def status(self, n=20):
        '''
        Summary for the debug status file.
        '''
        now = time.time()
        return {
            "dry_run": self.dry_run,
            "pressed": sorted(self.pressed),
            "counts": dict(self.counts),
            "recent": [f"{now - t:6.1f}s ago  {s}" for t, s in list(self.history)[-n:]],
        }


_keyboard = WingpuKeyboard()


def get_keyboard():
    '''
    Process-wide WingpuKeyboard
    '''
    return _keyboard
