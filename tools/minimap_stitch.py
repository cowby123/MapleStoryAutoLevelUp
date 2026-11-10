'''
minimap_stitch.py - Stitch minimap crops into a complete map image.

1. Load all minimap frame PNGs from a directory.
2. Determine if the minimap scrolls (frames differ) or is fixed (all same).
3. If fixed: take the pixel-wise median to remove the player dot.
4. If scrolling: align consecutive frames via template matching, build a
   panoramic canvas, then median-blend overlapping regions.

Usage:
    python3 minimap_stitch.py /home/test/minimap_stitch/frames
                              [--out /home/test/minimap_stitch/result.png]
'''
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


def load_frames(frame_dir):
    paths = sorted(glob.glob(os.path.join(frame_dir, "mm_*.png")))
    if not paths:
        print("No frames found", file=sys.stderr)
        sys.exit(1)
    frames = []
    for p in paths:
        img = np.array(Image.open(p))
        frames.append(img)
    print(f"Loaded {len(frames)} frames, shape {frames[0].shape}")
    return frames


def frames_are_static(frames, threshold=5.0):
    '''Check if all frames show the same minimap (no scrolling).'''
    ref = frames[0].astype(float)
    diffs = []
    step = max(1, len(frames) // 20)
    for i in range(0, len(frames), step):
        d = np.mean(np.abs(frames[i].astype(float) - ref))
        diffs.append(d)
    avg_diff = np.mean(diffs)
    print(f"Average frame difference: {avg_diff:.2f} (threshold {threshold})")
    return avg_diff < threshold


def median_blend(frames):
    '''Pixel-wise median of all frames (removes moving player dot).'''
    stack = np.stack(frames, axis=0)
    result = np.median(stack, axis=0).astype(np.uint8)
    return result


def match_offset(ref, cur, search_margin=30):
    '''Find (dx, dy) offset of cur relative to ref using normalized cross-correlation.'''
    # Use grayscale for matching
    ref_g = np.mean(ref, axis=2).astype(np.float32)
    cur_g = np.mean(cur, axis=2).astype(np.float32)

    h, w = ref_g.shape

    # Search region: try offsets from -margin to +margin
    best_score = -1
    best_dx, best_dy = 0, 0

    for dy in range(-search_margin, search_margin + 1):
        for dx in range(-search_margin, search_margin + 1):
            # Overlapping region
            ry0 = max(0, dy)
            ry1 = min(h, h + dy)
            rx0 = max(0, dx)
            rx1 = min(w, w + dx)

            cy0 = max(0, -dy)
            cy1 = cy0 + (ry1 - ry0)
            cx0 = max(0, -dx)
            cx1 = cx0 + (rx1 - rx0)

            if ry1 - ry0 < 20 or rx1 - rx0 < 20:
                continue

            patch_r = ref_g[ry0:ry1, rx0:rx1]
            patch_c = cur_g[cy0:cy1, cx0:cx1]

            if patch_r.shape != patch_c.shape:
                continue

            # Normalized cross-correlation
            r_mean = patch_r - patch_r.mean()
            c_mean = patch_c - patch_c.mean()
            num = np.sum(r_mean * c_mean)
            den = np.sqrt(np.sum(r_mean**2) * np.sum(c_mean**2))
            if den < 1e-6:
                continue
            score = num / den

            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_score


def stitch_scrolling(frames, search_margin=20):
    '''Stitch scrolling minimap frames into a panorama.'''
    # Compute offsets between consecutive frames
    offsets = [(0, 0)]  # first frame at origin
    cum_x, cum_y = 0, 0

    step = max(1, len(frames) // 100)  # sample ~100 pairs
    sampled_indices = list(range(0, len(frames), step))

    print(f"Computing offsets for {len(sampled_indices)} sampled frames...")
    positions = {0: (0, 0)}
    prev_idx = 0
    for idx in sampled_indices[1:]:
        dx, dy, score = match_offset(frames[prev_idx], frames[idx], search_margin)
        cum_x += dx
        cum_y += dy
        positions[idx] = (cum_x, cum_y)
        if score < 0.5:
            print(f"  WARNING: low match score {score:.3f} between frame {prev_idx} and {idx}")
        prev_idx = idx

    if not positions:
        print("No valid offsets computed")
        return median_blend(frames)

    # Find canvas bounds
    all_pos = list(positions.values())
    h, w = frames[0].shape[:2]
    min_x = min(p[0] for p in all_pos)
    min_y = min(p[1] for p in all_pos)
    max_x = max(p[0] for p in all_pos) + w
    max_y = max(p[1] for p in all_pos) + h

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    print(f"Canvas size: {canvas_w}x{canvas_h}")

    if canvas_w > 2000 or canvas_h > 2000:
        print("WARNING: canvas very large, might be matching errors")
        canvas_w = min(canvas_w, 2000)
        canvas_h = min(canvas_h, 2000)

    # Accumulate pixel values with count for median-like blending
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    canvas_count = np.zeros((canvas_h, canvas_w), dtype=np.int32)

    for idx, (ox, oy) in positions.items():
        px = ox - min_x
        py = oy - min_y
        frame = frames[idx]
        fh, fw = frame.shape[:2]
        # Clip to canvas
        x0 = max(0, px)
        y0 = max(0, py)
        x1 = min(canvas_w, px + fw)
        y1 = min(canvas_h, py + fh)
        fx0 = x0 - px
        fy0 = y0 - py
        fx1 = fx0 + (x1 - x0)
        fy1 = fy0 + (y1 - y0)

        canvas_sum[y0:y1, x0:x1] += frame[fy0:fy1, fx0:fx1].astype(np.float64)
        canvas_count[y0:y1, x0:x1] += 1

    # Average (approximation of median for speed)
    mask = canvas_count > 0
    for c in range(3):
        canvas_sum[:, :, c][mask] /= canvas_count[mask]

    result = canvas_sum.astype(np.uint8)

    # Report scroll range
    xs = [p[0] for p in all_pos]
    ys = [p[1] for p in all_pos]
    print(f"Scroll range: x [{min(xs)}, {max(xs)}], y [{min(ys)}, {max(ys)}]")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", help="Directory with mm_NNNN.png frames")
    parser.add_argument("--out", default=None,
                        help="Output path (default: <frame_dir>/../result.png)")
    parser.add_argument("--margin", type=int, default=20,
                        help="Search margin for offset matching (pixels)")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(args.frame_dir.rstrip("/")), "result.png")

    frames = load_frames(args.frame_dir)

    if frames_are_static(frames):
        print("Minimap is STATIC (no scrolling). Using median blend.")
        result = median_blend(frames)
    else:
        print("Minimap SCROLLS. Stitching panorama...")
        result = stitch_scrolling(frames, args.margin)

    Image.fromarray(result).save(args.out)
    print(f"Result saved: {args.out} ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()
