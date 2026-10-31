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

# Library import
import cv2

# Local import
from src.utils.logger import logger
from src.input.WingpuKeyboard import get_keyboard


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
        self.running = bool(wcfg.get("start_running", True))
        self.quit = False
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
            # All coordinates below are in the bot's working frame (1296x700)
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
            "input": get_keyboard().status(),
        }
        self.status = status
        return status

    def write_debug(self):
        bot = self.bot
        status = self.collect_status()
        tmp = os.path.join(self.debug_dir, ".status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=1, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.debug_dir, "status.json"))

        img = bot.img_frame_debug
        if img is not None:
            self._write_img("latest.png", img)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.debug_jpeg = buf.tobytes()
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

            def do_GET(self):
                if self.path.startswith("/status"):
                    self._send(200, "application/json",
                               json.dumps(control.status, ensure_ascii=False).encode())
                elif self.path.startswith("/debug.png"):
                    p = os.path.join(control.debug_dir, "latest.png")
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            self._send(200, "image/png", f.read())
                    else:
                        self._send(404, "text/plain", b"no frame yet")
                elif self.path.startswith("/guest.jpg"):
                    p = os.path.join(control.debug_dir, "latest_guest.jpg")
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            self._send(200, "image/jpeg", f.read())
                    else:
                        self._send(404, "text/plain", b"no frame yet")
                elif self.path.startswith("/mjpeg"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
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
                    self._send(200, "text/html; charset=utf-8",
                               b"<h3>MapleStoryAutoLevelUp (wingpu)</h3><p><a href='status'>status</a> "
                               b"<a href='debug.png'>debug.png</a> <a href='guest.jpg'>guest.jpg</a> "
                               b"<a href='mjpeg'>mjpeg</a></p><img src='mjpeg'>")

            def do_POST(self):
                if self.path.startswith("/run"):
                    control.apply_run(True, "http")
                elif self.path.startswith("/pause"):
                    control.apply_run(False, "http")
                elif self.path.startswith("/quit"):
                    control.request_quit("http")
                else:
                    self._send(404, "text/plain", b"POST /run, /pause or /quit")
                    return
                self._send(200, "application/json", json.dumps({"running": control.running}).encode())

        self.http = ThreadingHTTPServer(("0.0.0.0", self.http_port), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def stop(self):
        self.quit = True
        if self.http is not None:
            self.http.shutdown()
