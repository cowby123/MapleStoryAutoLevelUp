'''
wingpu_calibrate.py - find / record where the game client is in the guest screen.

Read-only on the game: it only reads frames.

  # 1) see what auto-detect finds (live guest screen, or a saved guest screenshot)
  python tools/wingpu_calibrate.py --show [--image guest.png] [--cfg custom]

  # 2) game in windowed mode: give the client rect once (guest pixels, read it off
  #    the --show / latest_guest.jpg image); this saves the title bar as a template
  #    so auto-detect follows the window when it moves
  python tools/wingpu_calibrate.py --rect X,Y,W,H [--image guest.png] [--title-height 59]

Outputs go to <out>/calibrate/ (out = /work in the container, else .).
'''
# Standard import
import argparse
import os
import sys

# Library import
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Local import
from src.utils.common import load_yaml, override_cfg  # noqa: E402
from src.input.GameWindowCapturorForWingpu import detect_client_rect  # noqa: E402
from src.input.WingpuControl import default_out_dir  # noqa: E402


def load_cfg(name):
    cfg = load_yaml("config/config_default.yaml")
    cfg = override_cfg(cfg, load_yaml("config/config_wingpu.yaml"))
    if name and os.path.exists(f"config/config_{name}.yaml"):
        cfg = override_cfg(cfg, load_yaml(f"config/config_{name}.yaml"))
    return cfg


def grab(image):
    if image:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"cannot read {image}")
        return img
    from src.input.WingpuVM import get_vm  # pylint: disable=import-outside-toplevel
    vm = get_vm(None)
    img = vm.frame(wait=3.0, new=False)
    if img is None:
        sys.exit("no guest frame (VM running? sender sharing frames?)")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="saved guest screenshot instead of the live VM")
    ap.add_argument("--cfg", default="custom")
    ap.add_argument("--show", action="store_true", help="run auto-detect and save an annotated image")
    ap.add_argument("--rect", default="", help="X,Y,W,H of the game client in guest pixels")
    ap.add_argument("--title-height", type=int, default=0,
                    help="title bar height in guest pixels (default: game_window.title_bar_height x scale)")
    ap.add_argument("--title-width", type=int, default=360,
                    help="template width from the window's left edge (icon + title text)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    out = os.path.join(default_out_dir(), "calibrate")
    os.makedirs(out, exist_ok=True)
    img = grab(args.image)
    gh, gw = img.shape[:2]
    cv2.imwrite(os.path.join(out, "guest.png"), img)
    print(f"guest screen {gw}x{gh} -> {out}/guest.png")

    if args.rect:
        x, y, w, h = [int(v) for v in args.rect.split(",")]
        exp_h, exp_w = cfg["game_window"]["size"]
        th = args.title_height or int(round(cfg["game_window"]["title_bar_height"] * h / exp_h))
        if y - th < 0:
            sys.exit(f"no room for a {th}px title bar above y={y}")
        tw = min(args.title_width, w)
        tpl = img[y - th:y, x:x + tw]
        tpl_path = os.path.join(out, "wingpu_title_bar.png")
        cv2.imwrite(tpl_path, tpl)
        cv2.imwrite(os.path.join(out, "client.png"), img[y:y + h, x:x + w])
        print(f"title template {tw}x{th} -> {tpl_path}")
        print("put this in your config (config/config_<cfg>.yaml):\n")
        print("wingpu:")
        print("  detect: \"auto\"")
        print(f"  title_template: \"{tpl_path}\"")
        print(f"  title_to_client: [0, {th}]")
        print(f"  client_size_guest: [{w}, {h}]")
        print(f"# or a fixed rect:  client_rect: [{x}, {y}, {w}, {h}]")
        cfg = override_cfg(cfg, {"wingpu": {"title_template": tpl_path, "title_to_client": (0, th),
                                           "client_size_guest": (w, h)}})
        args.show = True

    if args.show:
        tpl = None
        p = cfg["wingpu"].get("title_template")
        if p and os.path.exists(p):
            tpl = cv2.imread(p, cv2.IMREAD_COLOR)
        r = detect_client_rect(img, cfg, tpl)
        print(f"detected client: {r}")
        g = img.copy()
        if r is not None:
            cv2.rectangle(g, (r.x, r.y), (r.x + r.w - 1, r.y + r.h - 1), (0, 0, 255), 3)
            cv2.putText(g, str(r), (r.x + 10, r.y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out, "detected.png"), g)
        print(f"annotated -> {out}/detected.png")


if __name__ == "__main__":
    main()
