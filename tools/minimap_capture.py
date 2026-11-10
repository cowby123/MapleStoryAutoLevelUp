'''
minimap_capture.py - Capture minimap crops from a running VM via QEMU screendump.

Runs on the host. Takes N screenshots at ~1/s, detects the minimap white border,
crops just the minimap content, and saves each frame as a PNG.

Usage:
    python3 minimap_capture.py [--count 300] [--mon /tmp/qemu-mon-vm103.sock]
                               [--outdir /home/test/minimap_stitch/frames]
'''
import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def screendump(mon_sock, ppm_path="/tmp/_minimap_cap.ppm"):
    cmd = f'echo "screendump {ppm_path}" | socat - UNIX-CONNECT:{mon_sock}'
    subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
    time.sleep(0.3)
    if not os.path.isfile(ppm_path):
        return None
    try:
        img = np.array(Image.open(ppm_path))
        return img
    except Exception:
        return None


def find_minimap(img, min_size=60):
    '''Find the white-bordered minimap rectangle. Returns (x, y, w, h) or None.'''
    white = np.array([255, 255, 255], dtype=np.uint8)
    h_frame, w_frame = img.shape[:2]
    search_h = min(h_frame // 2, 500)
    search_w = min(w_frame // 2, 500)
    roi = img[:search_h, :search_w]

    # Find white horizontal lines (potential top/bottom borders)
    for y in range(roi.shape[0]):
        row = roi[y]
        white_mask = np.all(row == white, axis=1)
        if not np.any(white_mask):
            continue
        # Find runs of white pixels
        changes = np.diff(white_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if white_mask[0]:
            starts = np.concatenate([[0], starts])
        if white_mask[-1]:
            ends = np.concatenate([ends, [len(white_mask)]])
        for s, e in zip(starts, ends):
            run_len = e - s
            if run_len < min_size:
                continue
            # Check if this could be a top border: look for matching bottom border
            x0 = s
            w = run_len
            for y2 in range(y + min_size, min(y + 400, roi.shape[0])):
                if x0 + w <= roi.shape[1] and np.all(roi[y2, x0:x0+w] == white):
                    # Check left and right vertical borders
                    rh = y2 - y + 1
                    if np.all(roi[y:y+rh, x0] == white) and np.all(roi[y:y+rh, x0+w-1] == white):
                        # Found the frame, now get the content bounding box inside
                        content = roi[y+1:y2, x0+1:x0+w-1]
                        mask = np.any(content != white, axis=2)
                        coords = np.argwhere(mask)
                        if len(coords) > 100:
                            cy0, cx0 = coords.min(axis=0)
                            cy1, cx1 = coords.max(axis=0)
                            return (x0 + 1 + cx0, y + 1 + cy0,
                                    cx1 - cx0 + 1, cy1 - cy0 + 1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--mon", default="/tmp/qemu-mon-vm103.sock")
    parser.add_argument("--outdir", default="/home/test/minimap_stitch/frames")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    minimap_rect = None
    saved = 0
    for i in range(args.count):
        t0 = time.time()
        img = screendump(args.mon)
        if img is None:
            print(f"[{i}] screendump failed, skipping")
            time.sleep(args.interval)
            continue

        # Detect minimap location (reuse last if stable)
        if minimap_rect is None or i % 30 == 0:
            r = find_minimap(img)
            if r is not None:
                minimap_rect = r
                print(f"[{i}] minimap at ({r[0]}, {r[1]}) size {r[2]}x{r[3]}")

        if minimap_rect is None:
            print(f"[{i}] minimap not found, skipping")
            time.sleep(args.interval)
            continue

        x, y, w, h = minimap_rect
        crop = img[y:y+h, x:x+w]
        out_path = os.path.join(args.outdir, f"mm_{i:04d}.png")
        Image.fromarray(crop).save(out_path)
        saved += 1

        elapsed = time.time() - t0
        wait = max(0, args.interval - elapsed)
        if i % 20 == 0:
            print(f"[{i}] saved {saved} frames, crop {w}x{h}, took {elapsed:.2f}s")
        time.sleep(wait)

    print(f"Done: {saved} frames saved to {args.outdir}")


if __name__ == "__main__":
    main()
