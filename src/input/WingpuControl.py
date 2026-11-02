'''
WingpuControl
Headless control and debug output for the wingpu platform (no pynput hotkeys,
no OpenCV windows in the container):

- control file (default /work/control.json):
      {"run": true}    bot acts (keyboard controller enabled)
      {"run": false}   bot paused: keys released, it only watches
      {"quit": true}   bot exits
  re-read whenever it changes (checked every 0.5 s). With no file the bot starts
  with cfg wingpu.start_running.
- debug output (default /work/debug/, every wingpu.debug_interval seconds):
      latest.png        annotated game frame (the original debug window)
      latest_route.png  route map viz (normal mode)
      latest_guest.jpg  whole guest screen with the detected client rect
      status.json       client rect, player / minimap / monsters / HP-MP,
                        actions the bot sent (or WOULD send in dry run), CPU, fps
- optional HTTP (cfg wingpu.http_port, 0 = off), inside the container network:
      GET /status  GET /debug.png  GET /guest.jpg  GET /mjpeg
      POST /run  POST /pause  POST /quit
'''
# Standard import
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Library import
import cv2
import numpy as np

# Local import
from src.utils.logger import logger
from src.input.WingpuKeyboard import get_keyboard


def json_default(o):
    '''
    json.dump default=: numpy scalars/arrays (detector results are np.int32/float32)
    '''
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def dumps(obj, **kw):
    return json.dumps(obj, default=json_default, ensure_ascii=False, **kw)


def default_out_dir():
    '''
    /work inside the script container, else the current directory
    '''
    return "/work" if os.path.isdir("/work") and os.access("/work", os.W_OK) else "."


class WingpuControl:
    '''
    Control file + debug writer (+ HTTP) for a MapleStoryAutoBot
    '''
    def __init__(self, bot, cfg):
        self.bot = bot
        self.cfg = cfg
        wcfg = cfg.get("wingpu", {}) or {}
        out = default_out_dir()
        self.control_file = wcfg.get("control_file") or os.path.join(out, "control.json")
        self.debug_dir = wcfg.get("debug_dir") or os.path.join(out, "debug")
        self.debug_interval = float(wcfg.get("debug_interval", 2.0))
        self.http_port = int(wcfg.get("http_port", 0) or 0)
        self.debug_png = bool(wcfg.get("debug_png", False))
        self.running = bool(wcfg.get("start_running", True))
        self.quit = False
        self.restart = False # re-exec the bot process (settings applied), see main()
        self.pending_restart = False # settings saved but not applied yet
        self.webui = None
        self.control_mtime = None
        self.t_last_poll = 0.0
        self.t_last_debug = 0.0
        self.cpu_last = (time.time(), sum(os.times()[:2]))
        self.frames_last = 0
        self.status = {}
        self.debug_jpeg = None # latest annotated frame, JPEG bytes (for HTTP)
        self.http = None
        os.makedirs(self.debug_dir, exist_ok=True)

        # Apply the control file (or the default) before the bot acts
        if not self.poll_control(force=True):
            self.apply_run(self.running, "start (no control file)")
        if self.http_port:
            self.start_http()
        logger.info(f"[WingpuControl] control file {self.control_file}, "
                    f"debug {self.debug_dir} every {self.debug_interval}s"
                    + (f", http :{self.http_port}" if self.http_port else ""))

    ###############
    ### Control ###
    ###############
    def apply_run(self, run, why):
        self.running = bool(run)
        kb = self.bot.kb
        if kb is None:
            return
        if self.running:
            kb.enable()
        else:
            kb.disable()
            kb.set_command("none none none")
            kb.release_all_key()
        logger.info(f"[WingpuControl] run={self.running} ({why})")

    def poll_control(self, force=False):
        '''
        Re-read the control file if it changed. Returns True if it exists.
        '''
        try:
            mtime = os.stat(self.control_file).st_mtime
        except FileNotFoundError:
            return False
        if not force and mtime == self.control_mtime:
            return True
        self.control_mtime = mtime
        try:
            with open(self.control_file, "r", encoding="utf-8") as f:
                ctl = json.load(f)
        except (OSError, ValueError) as e:
            logger.error(f"[WingpuControl] bad control file {self.control_file}: {e}")
            return True
        if ctl.get("quit"):
            self.request_quit("control file")
        if "run" in ctl and bool(ctl["run"]) != self.running or force:
            self.apply_run(ctl.get("run", self.running), "control file")
        return True

    def release_keys(self):
        '''
        Release every held key (keyboard controller + the wingpu key state)
        '''
        kb = self.bot.kb
        if kb is not None:
            kb.set_command("none none none")
            kb.release_all_key()
        get_keyboard().release_all()

    def request_restart(self, why):
        logger.info(f"[WingpuControl] restart ({why})")
        self.release_keys()
        self.restart = True
        self.quit = True

    def cfg_unscaled(self):
        '''
        The bot's config with the resolution-scaled values put back to the
        configured (reference-frame) values: what the settings page shows
        '''
        from src.input.WingpuWebUI import unscaled
        return unscaled(self.bot.cfg, getattr(self.bot, "cfg_base", None))

    def request_quit(self, why):
        logger.info(f"[WingpuControl] quit ({why})")
        self.quit = True
        self.bot.terminate_threads()

    def tick(self):
        '''
        Call from the main thread loop (~100 Hz is fine; work is rate limited)
        '''
        now = time.time()
        if now - self.t_last_poll > 0.5:
            self.t_last_poll = now
            self.poll_control()
        if now - self.t_last_debug > self.debug_interval:
            self.t_last_debug = now
            try:
                self.write_debug()
            except Exception as e:
                logger.error(f"[WingpuControl] debug output: {e}")

    #############
    ### Debug ###
    #############
    def collect_status(self):
        bot = self.bot
        now = time.time()
        cpu_t = sum(os.times()[:2])
        t0, c0 = self.cpu_last
        cpu = (cpu_t - c0) / max(now - t0, 1e-6)
        self.cpu_last = (now, cpu_t)
        frames = getattr(bot, "frame_count", 0)
        loop_fps = (frames - self.frames_last) / max(now - t0, 1e-6)
        self.frames_last = frames

        hm = bot.health_monitor
        kb = bot.kb
        mh, mw = bot.img_minimap.shape[:2]
        monsters = [{"name": m.get("name"), "pos": [int(v) for v in m["position"]],
                     "size": [int(v) for v in m["size"]],
                     "score": round(float(m.get("score", 0)), 3)}
                    for m in (bot.monsters or [])]
        status = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "running": self.running,
            "state": bot.fsm.state.name if bot.fsm.state else None,
            "mode": bot.cfg["bot"]["mode"],
            "map": bot.cfg["bot"]["map"],
            "loop_fps": round(loop_fps, 2),
            "process_cpu_cores": round(cpu, 3),
            "capture": bot.capture.status() if hasattr(bot.capture, "status") else None,
            # All coordinates below are in the bot frame = the game client 1:1
            "resolution": {
                "frame_wh": list(bot.frame_hw[::-1]) if getattr(bot, "frame_hw", None) else None,
                "ui_scale": round(getattr(bot, "scale", 1.0), 4),
                "minimap_scale": (round(bot.minimap_scale.m, 4)
                                  if getattr(bot, "minimap_scale", None) is not None
                                  and bot.minimap_scale.m else None),
            },
            "detection": {
                "lost_paused": getattr(bot, "is_lost_paused", False),
                "lost_reason": getattr(bot, "lost_reason", []),
                "minimap_age_s": (round(now - bot.t_minimap_ok, 1) if getattr(bot, "t_minimap_ok", 0) else None),
                "player_age_s": (round(now - bot.t_player_seen, 1) if getattr(bot, "t_player_seen", 0) else None),
                "counts": dict(getattr(bot, "det_counts", {})),
                "hit_rate": {k: round(v / max(1, bot.det_counts.get("frames", 0)), 3)
                             for k, v in getattr(bot, "det_counts", {}).items() if k != "frames"},
            },
            "player": {
                "loc": list(bot.loc_player),
                "party_red_bar": list(bot.loc_party_red_bar),
                "nametag": list(bot.loc_nametag),
                "on_ladder": bot.is_on_ladder,
            },
            "minimap": {
                "loc": list(bot.loc_minimap), "size_wh": [mw, mh],
                "found": (mw, mh) != (10, 10),
                "player_on_minimap": list(bot.loc_player_minimap),
                "player_global": list(bot.loc_player_global),
            },
            "monsters": monsters,
            "hp_mp_exp": {
                "hp": round(hm.hp_percent, 1), "mp": round(hm.mp_percent, 1),
                "exp": round(hm.exp_percent, 1),
                "bars_found": hm.loc_size_bars != [(0, 0, 0, 0)] * 3,
                "bars": [list(b) for b in hm.loc_size_bars],
            } if hm else None,
            "command": {"move_x": bot.cmd_move_x, "move_y": bot.cmd_move_y,
                        "action": bot.cmd_action} ,
            "kb_command": f"{kb.cmd_left_right} {kb.cmd_up_down} {kb.cmd_action}" if kb else None,
            "kb_enabled": kb.is_enable if kb else None,
            "kb_hold": getattr(kb, "is_hold", False) if kb else None,
            "buffs": self.buff_status(),
            "player_source": getattr(bot, "player_source", None),
            "hunt_cycle": dict(bot.hunt_cycle.status(), safe_spot=bot.safe_spot(), plan=bot.safe_plan,
                               resting=getattr(kb, "is_resting", False))
                          if getattr(bot, "hunt_cycle", None) is not None else None,
            "nametag": bot.nametag.status() if getattr(bot, "nametag", None) is not None else None,
            "mob_search": {"mode": bot.cfg["monster_detect"].get("search", "attack"),
                           "box": list(bot.mob_search_box) if getattr(bot, "mob_search_box", None) else None},
            "pending_restart": self.pending_restart,
            "input": get_keyboard().status(),
        }
        self.status = status
        return status

    def buff_status(self):
        '''
        Per enabled buff: seconds until the next cast, seconds since the last
        (dry run: would-be) cast
        '''
        kb = self.bot.kb
        b = self.bot.cfg.get("buff_skill") or {}
        keys, cds = list(b.get("keys") or []), list(b.get("cooldown") or [])
        names = list(b.get("names") or [])
        now = time.time()
        out = []
        for i, k in enumerate(keys):
            t_last = kb.t_last_buff_cast[i] if kb is not None and i < len(kb.t_last_buff_cast) else 0
            logged = getattr(kb, "t_buff_cast_logged", [None] * len(keys))
            t_cast = logged[i] if i < len(logged) else None
            cd = float(cds[i]) if i < len(cds) else 0
            out.append({"name": names[i] if i < len(names) else "", "key": k, "cooldown": cd,
                        "next_in_s": round(max(0.0, cd - (now - t_last)), 1),
                        "last_cast_s_ago": round(now - t_cast, 1) if t_cast else None})
        return out

    def write_debug(self):
        bot = self.bot
        status = self.collect_status()
        tmp = os.path.join(self.debug_dir, ".status.json.tmp")
        text = dumps(status, indent=1)  # serialise first: a failure must not leave a stale file
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, os.path.join(self.debug_dir, "status.json"))

        img = getattr(bot, "img_frame_debug_done", None)
        if img is None:
            img = bot.img_frame_debug
        if img is not None:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.debug_jpeg = buf.tobytes()
                self._write_bytes("latest.jpg", self.debug_jpeg)
            # PNG of the full frame costs ~10x the JPEG; only when asked for
            if self.debug_png:
                self._write_img("latest.png", img)
        if bot.img_route_debug is not None:
            self._write_img("latest_route.png", bot.img_route_debug)
        guest = bot.capture.get_guest_frame() if hasattr(bot.capture, "get_guest_frame") else None
        if guest is not None and bot.capture.rect is not None:
            g = guest.copy()
            r = bot.capture.rect
            cv2.rectangle(g, (r.x, r.y), (r.x + r.w - 1, r.y + r.h - 1), (0, 0, 255), 3)
            org = (r.x + r.w // 3, r.y + 40)  # clear of the minimap (top-left)
            cv2.putText(g, f"client {r}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(g, f"client {r}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
            self._write_img("latest_guest.jpg", g)

    def _write_bytes(self, name, data):
        path = os.path.join(self.debug_dir, name)
        tmp = os.path.join(self.debug_dir, "." + name)
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    def _write_img(self, name, img):
        path = os.path.join(self.debug_dir, name)
        tmp = os.path.join(self.debug_dir, "." + name)
        ext = os.path.splitext(name)[1]
        ok, buf = cv2.imencode(ext, img)
        if ok:
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, path)

    ############
    ### HTTP ###
    ############
    def start_http(self):
        control = self
        from src.input.WingpuWebUI import WebUI
        self.webui = WebUI(self)
        page_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui", "index.html")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code=200):
                self._send(code, "application/json; charset=utf-8", dumps(obj).encode())

            def _file(self, name, ctype):
                p = os.path.join(control.debug_dir, name)
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        self._send(200, ctype, f.read())
                else:
                    self._send(404, "text/plain", b"no frame yet")

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n > 4 * 1024 * 1024:
                    raise ValueError("body too large")
                raw = self.rfile.read(n) if n else b""
                return json.loads(raw.decode("utf-8")) if raw else {}

            def do_GET(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path in ("/", "/index.html"):
                    with open(page_path, "rb") as f:
                        self._send(200, "text/html; charset=utf-8", f.read())
                elif path == "/status":
                    self._json(control.status)
                elif path == "/debug.jpg":
                    jpg = control.debug_jpeg
                    if jpg is None:
                        self._send(404, "text/plain", b"no frame yet")
                    else:
                        self._send(200, "image/jpeg", jpg)
                elif path == "/debug.png":
                    self._file("latest.png" if control.debug_png else "latest.jpg",
                               "image/png" if control.debug_png else "image/jpeg")
                elif path == "/route.png":
                    self._file("latest_route.png", "image/png")
                elif path == "/guest.jpg":
                    self._file("latest_guest.jpg", "image/jpeg")
                elif path == "/api/config":
                    self._json(control.webui.config())
                elif path == "/api/routes":
                    self._json(control.webui.routes())
                elif path == "/api/route_img":
                    q = urlparse(self.path).query
                    name = dict(kv.split("=", 1) for kv in q.split("&") if "=" in kv).get("name", "map")
                    png = control.webui.route_png(name)
                    if png is None:
                        self._send(404, "text/plain", b"no such route")
                    else:
                        self._send(200, "image/png", png)
                elif path == "/api/nametag":
                    self._json(control.webui.nametag_status())
                elif path == "/api/nametag/template.png":
                    png = control.webui.nametag_png()
                    if png is None:
                        self._send(404, "text/plain", b"no template")
                    else:
                        self._send(200, "image/png", png)
                elif path == "/api/nametag/snapshot.jpg":
                    q = urlparse(self.path).query
                    sid = dict(kv.split("=", 1) for kv in q.split("&") if "=" in kv).get("id", "0")
                    jpg = control.webui.snapshot_jpeg(sid) if sid.isdigit() else None
                    if jpg is None:
                        self._send(404, "text/plain", b"no snapshot")
                    else:
                        self._send(200, "image/jpeg", jpg)
                elif path == "/mjpeg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = None
                    try:
                        while not control.quit:
                            jpg = control.debug_jpeg
                            if jpg is not None and jpg is not last:
                                last = jpg
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                                 b"Content-Length: " + str(len(jpg)).encode() +
                                                 b"\r\n\r\n" + jpg + b"\r\n")
                            time.sleep(0.2)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self._send(404, "text/plain", b"not found")

            def do_POST(self):
                path = urlparse(self.path).path.rstrip("/")
                try:
                    body = self._body()
                except (ValueError, UnicodeDecodeError) as e:
                    self._json({"ok": False, "errors": {"body": str(e)}}, 400)
                    return
                if path == "/run":
                    control.apply_run(True, "http")
                elif path == "/pause":
                    control.apply_run(False, "http")
                elif path == "/quit":
                    control.request_quit("http")
                elif path in ("/restart", "/api/restart"):
                    control.request_restart("http")
                elif path == "/api/config":
                    self._json(control.webui.save(body.get("values") or {}))
                    return
                elif path == "/api/config/reset":
                    self._json(control.webui.reset())
                    return
                elif path == "/api/live":
                    self._json(control.webui.set_live(body))
                    return
                elif path == "/api/route_safe":
                    self._json(control.webui.save_route_safe(body))
                    return
                elif path == "/api/safe_spot":
                    self._json(control.webui.save_safe_spot(body))
                    return
                elif path == "/api/nametag/snapshot":
                    self._json(control.webui.nametag_snapshot())
                    return
                elif path == "/api/nametag/save":
                    self._json(control.webui.nametag_save(body))
                    return
                elif path == "/api/nametag/delete":
                    self._json(control.webui.nametag_delete())
                    return
                else:
                    self._send(404, "text/plain", b"POST run, pause, quit, restart, api/config, api/live")
                    return
                self._json({"ok": True, "running": control.running, "restart": control.restart})

        self.http = ThreadingHTTPServer(("0.0.0.0", self.http_port), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def stop(self):
        self.quit = True
        if self.http is not None:
            self.http.shutdown()
