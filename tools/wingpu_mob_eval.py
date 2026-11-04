'''
wingpu_mob_eval.py - monster detection hit rate on hand-labelled frames
(offline, dry run; no VM).

  python tools/wingpu_mob_eval.py --labels fixtures/cloud/labels.json \
      --set bot.map=cloud_balcony [--set monster_detect.diff_thres=0.2] [--region full|roi]

labels.json: {"frame.png": [[cx, cy, kind(, hard)], ...]} in guest pixels
(frames next to labels.json). A label is hit when a detection's box centre is
within --tol px of it (each detection matches one label). "hard" labels (cut off
at the screen edge, behind UI, fading, under hit effects) count as hits when
found but are not misses when missed. A detection near no label is a false
positive. Prints recall, false positives and the CPU time of
get_monsters_in_range per frame for the playfield ("full") or a 1x roi box
around the frame centre ("roi").
'''
import argparse
import json
import os
import sys
import time

os.environ["WINGPU"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

import cv2  # noqa: E402
import yaml  # noqa: E402

import wingpu_detect_test as wdt  # noqa: E402


def evaluate(labels, base, cfg_name, tol, region, repeat=3, verbose=False):
    bot = wdt.make_bot(cfg_name)
    tp = fn = fp = hard_hit = 0
    cpu = []
    per = []
    for name, labs in labels.items():
        img = cv2.imread(os.path.join(base, name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if bot.frame_hw != (h, w):
            bot.on_resolution(h, w)
        bot.img_frame = img
        bot.img_frame_debug = img.copy()
        bot.loc_player = (-1000, -1000)
        y0, y1 = int(90 * h / 768), int(673 * h / 768)
        if region == "full":
            tl, br = (0, y0), (w, y1)
        else:
            tl, br = (w // 2 - 450, y0), (w // 2 + 450, y1)
        best = 1e9
        for _ in range(repeat):
            c0 = time.process_time()
            mons = bot.get_monsters_in_range(tl, br)
            best = min(best, time.process_time() - c0)
        cpu.append(best)
        dets = []
        for m in mons:
            x, y = m["position"]
            mh, mw = m["size"]
            dets.append((x + mw / 2, y + mh / 2, m["name"], m["score"]))
        used = [False] * len(dets)
        miss = []
        for lab in labs:
            cx, cy, kind = lab[:3]
            hard = len(lab) > 3 and lab[3]
            if not (tl[0] <= cx < br[0]):
                continue
            j = next((j for j, d in enumerate(dets) if not used[j]
                      and abs(d[0] - cx) <= tol and abs(d[1] - cy) <= tol), None)
            if j is not None:
                used[j] = True
                if hard:
                    hard_hit += 1
                else:
                    tp += 1
            elif not hard:
                fn += 1
                miss.append((cx, cy, kind))
        # a detection matching a label already taken (two boxes on one bear) is
        # not a false positive either
        fps = []
        for j, d in enumerate(dets):
            if used[j]:
                continue
            if any(abs(d[0] - lab[0]) <= tol and abs(d[1] - lab[1]) <= tol for lab in labs):
                continue
            fps.append((int(d[0]), int(d[1]), d[2], round(float(d[3]), 3)))
        fp += len(fps)
        per.append((name, len(dets), miss, fps))
    n = tp + fn
    res = {"recall": f"{tp}/{n} ({100 * tp / max(1, n):.0f}%)", "hard_hits": hard_hit,
           "false_pos": fp, "cpu_ms": round(1000 * sum(cpu) / max(1, len(cpu)), 1)}
    if verbose:
        for name, nd, miss, fps in per:
            print(f"  {name}: dets={nd} miss={miss} fp={fps}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="fixtures/cloud/labels.json")
    ap.add_argument("--cfg", default="wingpu_maple")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tol", type=int, default=30)
    ap.add_argument("--region", default="full", choices=["full", "roi"])
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=", 1)
        wdt.OVERRIDES[k] = yaml.safe_load(v)
    import logging
    wdt.logger.set_level(logging.WARNING)
    labels = json.load(open(args.labels, encoding="utf-8"))
    print(evaluate(labels, os.path.dirname(args.labels), args.cfg, args.tol, args.region, verbose=args.v))


if __name__ == "__main__":
    main()
