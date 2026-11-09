#!/usr/bin/env python3
'''
gen_dataset_v2.py - Generate improved synthetic YOLO training data.

Uses clean median backgrounds (monsters removed via median filtering)
and heavier augmentation (blur, noise, transparency) to reduce the
domain gap between synthetic sprites and game-rendered monsters.
'''
import cv2
import numpy as np
import os
import glob
import random
import shutil


def paste_sprite(bg, sprite, x, y, scale=1.0, flip=False, angle=0,
                 blur_sigma=0, alpha_mult=1.0):
    '''Paste RGBA sprite onto BGR background with augmentations.'''
    s = sprite.copy()
    if flip:
        s = cv2.flip(s, 1)
    if abs(angle) > 0.1:
        M = cv2.getRotationMatrix2D((s.shape[1] // 2, s.shape[0] // 2), angle, 1.0)
        s = cv2.warpAffine(s, M, (s.shape[1], s.shape[0]), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if abs(scale - 1.0) > 0.01:
        s = cv2.resize(s, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Blur sprite to simulate game rendering (bilinear filtering + anti-aliasing)
    if blur_sigma > 0:
        s[:, :, :3] = cv2.GaussianBlur(s[:, :, :3], (0, 0), blur_sigma)

    h, w = s.shape[:2]
    bh, bw = bg.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + w)
    y2 = min(bh, y + h)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    roi = s[sy1:sy2, sx1:sx2]
    alpha = (roi[:, :, 3].astype(np.float32) / 255.0 * alpha_mult)[:, :, np.newaxis]

    if alpha.max() < 0.01:
        return None

    bg_patch = bg[y1:y2, x1:x2].astype(np.float32)
    fg_patch = roi[:, :, :3].astype(np.float32)
    bg[y1:y2, x1:x2] = (alpha * fg_patch + (1 - alpha) * bg_patch).astype(np.uint8)

    # Compute tight bbox
    alpha_2d = roi[:, :, 3]
    ys, xs = np.where(alpha_2d > 10)
    if len(xs) == 0:
        return None
    bx1 = x1 + int(xs.min())
    by1 = y1 + int(ys.min())
    bx2 = x1 + int(xs.max())
    by2 = y1 + int(ys.max())
    bw_box = bx2 - bx1
    bh_box = by2 - by1
    if bw_box < 10 or bh_box < 10:
        return None
    return (bx1, by1, bw_box, bh_box)


def gen_image(bg, sprites):
    '''Generate one synthetic training image with random monsters.'''
    img = bg.copy()
    fh, fw = img.shape[:2]
    boxes = []
    n_monsters = random.randint(2, 12)

    for _ in range(n_monsters):
        sprite = random.choice(sprites)
        scale = random.uniform(0.95, 1.15)
        flip = random.random() < 0.5
        angle = random.uniform(-3, 3)
        blur_sigma = random.uniform(0.3, 1.5)
        alpha_mult = random.uniform(0.85, 1.0)

        y = random.randint(50, 580) - int(sprite.shape[0] * scale // 2)
        x = random.randint(-50, fw - 50)

        box = paste_sprite(img, sprite, x, y, scale, flip, angle,
                           blur_sigma, alpha_mult)
        if box is not None:
            boxes.append(box)

    # Whole-image augmentation
    if random.random() < 0.5:
        factor = random.uniform(0.85, 1.15)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img, boxes


def main():
    random.seed(42)
    np.random.seed(42)

    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SPRITE_DIRS = [
        os.path.join(REPO, 'monster_yolo', 'dark_stone_golem'),
        os.path.join(REPO, 'monster_yolo', 'mixed_golem'),
    ]
    CLEAN_BG_DIR = os.path.join(REPO, 'yolo_data', 'clean_bg')
    REAL_BG_DIR = os.path.join(REPO, 'yolo_data', 'backgrounds')
    REAL_ANN_DIR = os.path.join(REPO, 'yolo_data', 'real_annotated')
    OUT = os.path.join(REPO, 'yolo_data', 'dataset_v2')

    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUT, 'labels', split), exist_ok=True)

    # Load sprites
    sprites = []
    for d in SPRITE_DIRS:
        for p in sorted(glob.glob(os.path.join(d, '*.png'))):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                sprites.append(img)
    print(f'Loaded {len(sprites)} sprites')

    # Load backgrounds
    clean_bgs = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(CLEAN_BG_DIR, '*.png')))]
    real_bgs = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(REAL_BG_DIR, '*.png')))]
    all_bgs = [b for b in clean_bgs + real_bgs if b is not None]
    print(f'Loaded {len(clean_bgs)} clean + {len(real_bgs)} real backgrounds')

    TRAIN_COUNT = 2000
    VAL_COUNT = 200
    NEG_PCT = 0.08

    for split, count in [('train', TRAIN_COUNT), ('val', VAL_COUNT)]:
        neg_count = int(count * NEG_PCT)
        pos_count = count - neg_count

        for i in range(count):
            # Clean backgrounds 70%, real 30%
            if random.random() < 0.7 and clean_bgs:
                bg = random.choice(clean_bgs).copy()
            else:
                bg = random.choice(all_bgs).copy()

            fh, fw = bg.shape[:2]

            if i < pos_count:
                img, boxes = gen_image(bg, sprites)
            else:
                img, boxes = bg, []

            fname = f'{split}_{i:05d}'
            cv2.imwrite(os.path.join(OUT, 'images', split, f'{fname}.jpg'),
                        img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(OUT, 'labels', split, f'{fname}.txt'), 'w') as f:
                for bx, by, bw, bh in boxes:
                    xc = (bx + bw / 2) / fw
                    yc = (by + bh / 2) / fh
                    nw = bw / fw
                    nh = bh / fh
                    f.write(f'0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n')

            if (i + 1) % 500 == 0:
                print(f'  {split}: {i+1}/{count}')
        print(f'{split}: {count} images done')

    # Copy real annotated frames (health bar labels) into training set
    real_img_dir = os.path.join(REAL_ANN_DIR, 'images', 'train')
    real_lbl_dir = os.path.join(REAL_ANN_DIR, 'labels', 'train')
    real_count = 0
    if os.path.isdir(real_img_dir):
        for img_file in glob.glob(os.path.join(real_img_dir, '*.jpg')):
            fname = os.path.basename(img_file)
            lbl_file = os.path.join(real_lbl_dir, fname.replace('.jpg', '.txt'))
            if os.path.exists(lbl_file):
                shutil.copy2(img_file,
                             os.path.join(OUT, 'images', 'train', f'real_{fname}'))
                shutil.copy2(lbl_file,
                             os.path.join(OUT, 'labels', 'train',
                                          f'real_{fname.replace(".jpg", ".txt")}'))
                real_count += 1
                # Flipped copy
                img = cv2.imread(img_file)
                flipped = cv2.flip(img, 1)
                cv2.imwrite(os.path.join(OUT, 'images', 'train', f'realflip_{fname}'),
                            flipped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                fwidth = img.shape[1]
                with open(lbl_file) as lf:
                    lines = lf.readlines()
                with open(os.path.join(OUT, 'labels', 'train',
                                       f'realflip_{fname.replace(".jpg", ".txt")}'), 'w') as of:
                    for line in lines:
                        parts = line.strip().split()
                        cls = parts[0]
                        xc = float(parts[1])
                        yc = parts[2]
                        w = parts[3]
                        h = parts[4]
                        of.write(f'{cls} {1.0 - xc:.6f} {yc} {w} {h}\n')
                real_count += 1
    print(f'Added {real_count} real annotated frames')

    # Write dataset.yaml
    yaml_path = os.path.join(OUT, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'path: {OUT.replace(chr(92), "/")}\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('nc: 1\n')
        f.write('names:\n')
        f.write('  0: monster\n')

    train_count = len(glob.glob(os.path.join(OUT, 'images', 'train', '*.jpg')))
    val_count = len(glob.glob(os.path.join(OUT, 'images', 'val', '*.jpg')))
    print(f'Dataset v2 ready: {train_count} train, {val_count} val')
    print(f'YAML: {yaml_path}')


if __name__ == '__main__':
    main()
