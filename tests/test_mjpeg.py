'''
Live view (./mjpeg): one producer at wingpu.mjpeg_fps while clients watch,
nothing encoded without clients. Offline: bot from a saved frame, the real
HTTP server on localhost, new "annotated frames" at 20/s from a test thread.

  python tests/test_mjpeg.py
'''
import os
import sys
import threading
import time
import urllib.request

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import test_webui_guards as G  # noqa: E402


def read_stream(url, seconds, out):
    n = 0
    t_end = time.time() + seconds
    with urllib.request.urlopen(url, timeout=5) as r:
        buf = b""
        while time.time() < t_end:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            buf += chunk
            n += buf.count(b"--frame")
            buf = buf[buf.rfind(b"--frame") + 7:] if b"--frame" in buf else buf[-16:]
    out.append(n)


def test_mjpeg_rate_and_idle():
    bot, ctl = G.make()
    bot.cfg["wingpu"]["mjpeg_fps"] = 10
    stop = threading.Event()

    def frames():  # the bot loop publishing a new annotated frame 20x/s
        while not stop.is_set():
            bot.img_frame_debug_done = bot.img_frame.copy()
            time.sleep(0.05)

    threading.Thread(target=frames, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{G.PORT}/mjpeg"
        res = []
        ths = [threading.Thread(target=read_stream, args=(url, 3.0, res)) for _ in range(2)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        enc = ctl.mjpeg_encoded
        print(f"  2 clients for 3 s: frames received {res}, encoded once for both: {enc}, "
              f"encode {1000 * ctl.mjpeg_cpu / max(1, enc):.1f} ms/frame (1366x768)")
        assert all(20 <= n <= 36 for n in res), res        # ~10 fps each
        assert enc <= 36                                   # shared, not per client
        time.sleep(1.0)
        assert ctl.mjpeg_clients == 0
        e0 = ctl.mjpeg_encoded
        time.sleep(1.5)
        assert ctl.mjpeg_encoded == e0, "encoding without clients"
        print("  no clients: no encoding")
        # live change of the rate
        bot.cfg["wingpu"]["mjpeg_fps"] = 4
        res2 = []
        read_stream(url, 3.0, res2)
        print(f"  mjpeg_fps 4: {res2[0]} frames in 3 s")
        assert 8 <= res2[0] <= 15, res2
    finally:
        stop.set()
        ctl.stop()


if __name__ == "__main__":
    try:
        test_mjpeg_rate_and_idle()
        print("ok   test_mjpeg_rate_and_idle")
    finally:
        G.shutil.rmtree(G.TMP, ignore_errors=True)
