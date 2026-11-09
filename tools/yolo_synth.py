#!/usr/bin/env python3
'''
yolo_synth.py - Generate synthetic YOLO training data for monster detection.

Pastes transparent monster sprites onto real game backgrounds at random
positions, scales, and orientations. Outputs YOLO-format labels.

Usage:
    python tools/yolo_synth.py \
        --backgrounds yolo_data/backgrounds \
        --sprites monster_yolo/dark_stone_golem monster_yolo/mixed_golem \
        --output yolo_data/dataset \
        --count 2000 --val-count 200 --neg-pct 0.1
'''

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Sprite loading
# ---------------------------------------------------------------------------

def load_sprites(sprite_dirs):
    '''Load all RGBA PNGs from the given directories.'''
    sprites = []
    for d in sprite_dirs:
        for ext in ('*.png', '*.PNG'):
            for path in sorted(glob.glob(os.path.join(d, ext))):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None or img.ndim < 3 or img.shape[2] != 4:
                    print(f'skip (not RGBA): {path}')
                    continue
                sprites.append(img)
    if not sprites:
        sys.exit(f'ERROR: no RGBA sprites found in {sprite_dirs}')
    print(f'Loaded {len(sprites)} sprites from {len(sprite_dirs)} dirs')
    return sprites


def load_backgrounds(bg_dir):
    '''Load all background images (BGR, 1366x768 expected).'''
    bgs = []
    for ext in ('*.png', '*.PNG', '*.jpg', '*.jpeg'):
        for path in sorted(glob.glob(os.path.join(bg_dir, ext))):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                bgs.append(img)
    if not bgs:
        sys.exit(f'ERROR: no backgrounds found in {bg_dir}')
    print(f'Loaded {len(bgs)} backgrounds')
    return bgs


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def random_scale(sprite, lo=0.9, hi=1.1):
    '''Scale sprite by a random factor.'''
    factor = random.uniform(lo, hi)
    h, w = sprite.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    return cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)


def random_flip(sprite):
    '''Horizontal flip with 50% probability.'''
    if random.random() < 0.5:
        return cv2.flip(sprite, 1)
    return sprite


def random_rotate(sprite, max_deg=3):
    '''Rotate sprite by a small random angle, keeping alpha.'''
    angle = random.uniform(-max_deg, max_deg)
    if abs(angle) < 0.1:
        return sprite
    h, w = sprite.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Compute new bounding size
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(sprite, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    return rotated


def color_jitter(sprite, brightness_range=0.1, hue_range=5):
    '''Apply slight random brightness and hue shift to the RGB channels.'''
    bgr = sprite[:, :, :3].copy()
    alpha = sprite[:, :, 3]

    # Brightness: scale by random factor
    bfactor = random.uniform(1.0 - brightness_range, 1.0 + brightness_range)
    bgr = np.clip(bgr.astype(np.float32) * bfactor, 0, 255).astype(np.uint8)

    # Hue shift in HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    hue_shift = random.randint(-hue_range, hue_range)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.dstack([bgr, alpha])


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def paste_sprite(canvas, sprite, x, y):
    '''
    Alpha-composite sprite onto canvas at (x, y) = top-left of sprite.
    Returns the actual bounding box (x, y, w, h) clipped to canvas, or None
    if entirely outside.
    '''
    ch, cw = canvas.shape[:2]
    sh, sw = sprite.shape[:2]

    # Clip to canvas bounds
    sx = max(0, -x)
    sy = max(0, -y)
    ex = min(sw, cw - x)
    ey = min(sh, ch - y)
    if ex <= sx or ey <= sy:
        return None  # entirely outside

    cx_start = x + sx
    cy_start = y + sy
    region_w = ex - sx
    region_h = ey - sy

    src = sprite[sy:ey, sx:ex]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    dst = canvas[cy_start:cy_start + region_h, cx_start:cx_start + region_w]

    # Blend: out = alpha * src + (1 - alpha) * dst
    blended = (alpha * src[:, :, :3].astype(np.float32) +
               (1.0 - alpha) * dst.astype(np.float32))
    canvas[cy_start:cy_start + region_h, cx_start:cx_start + region_w] = \
        blended.astype(np.uint8)

    # Return the visible bounding box of the non-transparent pixels
    # (use the alpha mask to find the tight bbox of what was actually drawn)
    mask = (src[:, :, 3] > 10)  # threshold: nearly-transparent pixels ignored
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    bx = int(cx_start + xs.min())
    by = int(cy_start + ys.min())
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    return (bx, by, bw, bh)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_one(backgrounds, sprites, bg_h, bg_w,
                 min_monsters=3, max_monsters=15,
                 y_min=50, y_max=600, is_negative=False):
    '''
    Generate one synthetic training image.
    Returns: (image_bgr, list of (class_id, xc_norm, yc_norm, w_norm, h_norm))
    '''
    bg = random.choice(backgrounds).copy()
    # Resize background if needed (should already be 1366x768)
    if bg.shape[:2] != (bg_h, bg_w):
        bg = cv2.resize(bg, (bg_w, bg_h))

    labels = []
    if is_negative:
        return bg, labels

    n_monsters = random.randint(min_monsters, max_monsters)
    for _ in range(n_monsters):
        sprite = random.choice(sprites).copy()

        # Augment
        sprite = random_scale(sprite, 0.9, 1.1)
        sprite = random_flip(sprite)
        sprite = random_rotate(sprite, max_deg=3)
        sprite = color_jitter(sprite, brightness_range=0.1, hue_range=5)

        sh, sw = sprite.shape[:2]
        # Random position: x anywhere (allow partial off-screen)
        x = random.randint(-sw // 4, bg_w - sw * 3 // 4)
        # y: place sprite bottom on a platform-level (y_min to y_max)
        y_bottom = random.randint(y_min + sh, y_max)
        y = y_bottom - sh

        bbox = paste_sprite(bg, sprite, x, y)
        if bbox is None:
            continue
        bx, by, bw, bh = bbox
        # Skip tiny boxes (< 10x10 after clipping)
        if bw < 10 or bh < 10:
            continue

        # YOLO format: class x_center y_center width height (all normalized)
        xc = (bx + bw / 2) / bg_w
        yc = (by + bh / 2) / bg_h
        wn = bw / bg_w
        hn = bh / bg_h
        # Clamp to [0, 1]
        xc = max(0, min(1, xc))
        yc = max(0, min(1, yc))
        wn = max(0, min(1, wn))
        hn = max(0, min(1, hn))
        labels.append((0, xc, yc, wn, hn))

    return bg, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic YOLO training data for monster detection')
    parser.add_argument('--backgrounds', required=True,
                        help='Directory of background frames (1366x768 PNGs)')
    parser.add_argument('--sprites', nargs='+', required=True,
                        help='Directories of monster sprite PNGs (RGBA)')
    parser.add_argument('--output', required=True,
                        help='Output dataset directory')
    parser.add_argument('--count', type=int, default=2000,
                        help='Number of training images (default: 2000)')
    parser.add_argument('--val-count', type=int, default=200,
                        help='Number of validation images (default: 200)')
    parser.add_argument('--neg-pct', type=float, default=0.1,
                        help='Fraction of negative (no monsters) images (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--img-width', type=int, default=1366,
                        help='Background width (default: 1366)')
    parser.add_argument('--img-height', type=int, default=768,
                        help='Background height (default: 768)')
    parser.add_argument('--y-min', type=int, default=50,
                        help='Minimum y for monster placement (default: 50)')
    parser.add_argument('--y-max', type=int, default=600,
                        help='Maximum y for monster placement (default: 600)')
    parser.add_argument('--min-monsters', type=int, default=3,
                        help='Minimum monsters per image (default: 3)')
    parser.add_argument('--max-monsters', type=int, default=15,
                        help='Maximum monsters per image (default: 15)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    sprites = load_sprites(args.sprites)
    backgrounds = load_backgrounds(args.backgrounds)

    # Create output directories
    out = os.path.abspath(args.output)
    for split in ('train', 'val'):
        os.makedirs(os.path.join(out, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out, 'labels', split), exist_ok=True)

    # Generate images for each split
    for split, total in [('train', args.count), ('val', args.val_count)]:
        n_neg = int(total * args.neg_pct)
        n_pos = total - n_neg
        print(f'\n--- {split}: {n_pos} positive + {n_neg} negative = {total} ---')

        for i in range(total):
            is_neg = (i >= n_pos)
            img, labels = generate_one(
                backgrounds, sprites,
                bg_h=args.img_height, bg_w=args.img_width,
                min_monsters=args.min_monsters,
                max_monsters=args.max_monsters,
                y_min=args.y_min, y_max=args.y_max,
                is_negative=is_neg,
            )

            stem = f'{split}_{i:05d}'
            img_path = os.path.join(out, 'images', split, stem + '.jpg')
            lbl_path = os.path.join(out, 'labels', split, stem + '.txt')

            # Save image (JPG quality 95)
            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save label (empty file for negatives)
            with open(lbl_path, 'w') as f:
                for cls, xc, yc, w, h in labels:
                    f.write(f'{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')

            if (i + 1) % 100 == 0 or i == total - 1:
                print(f'  {split}: {i + 1}/{total}', flush=True)

    # Write dataset.yaml
    yaml_path = os.path.join(out, 'dataset.yaml')
    # Use forward slashes for YOLO compatibility
    train_dir = os.path.join(out, 'images', 'train').replace('\\', '/')
    val_dir = os.path.join(out, 'images', 'val').replace('\\', '/')
    with open(yaml_path, 'w') as f:
        f.write(f'# YOLO dataset config - auto-generated by yolo_synth.py\n')
        f.write(f'path: {out.replace(chr(92), "/")}\n')
        f.write(f'train: images/train\n')
        f.write(f'val: images/val\n')
        f.write(f'\n')
        f.write(f'# Classes\n')
        f.write(f'nc: 1\n')
        f.write(f'names:\n')
        f.write(f'  0: monster\n')
    print(f'\nDataset YAML: {yaml_path}')
    print('Done.')


if __name__ == '__main__':
    main()
