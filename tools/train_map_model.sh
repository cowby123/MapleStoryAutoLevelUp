#!/bin/bash
# train_map_model.sh - Train a per-map YOLO monster detection model.
#
# Usage:
#   tools/train_map_model.sh <map_name> <monster1> [monster2 ...]
#
# Example:
#   tools/train_map_model.sh ston dark_stone_golem mixed_golem
#
# What it does:
#   1. Downloads monster sprites (if not already present) via mob_maker.py
#   2. Captures background frames from the running VM (or uses existing ones)
#   3. Generates synthetic training data (yolo_synth.py)
#   4. Trains YOLOv8n on the host GPU (ssh to .63)
#   5. Exports to ONNX
#   6. Copies to models/<map_name>.onnx
#
# Requirements:
#   - Host .63 reachable via ssh (test@192.168.66.63)
#   - /home/test/yolo_venv with ultralytics installed on the host
#   - A running VM container (wingpu-ct-<vmid>) with the target map for backgrounds
#   - Python 3 with cv2, numpy on the dev box
#
# Environment variables:
#   YOLO_HOST       SSH target (default: test@192.168.66.63)
#   YOLO_VENV       Python venv on the host (default: /home/test/yolo_venv)
#   YOLO_CONTAINER  Container name for background capture (default: auto-detect from map)
#   YOLO_EPOCHS     Training epochs (default: 50)
#   YOLO_BATCH      Batch size (default: 16)
#   YOLO_DEVICE     Training device (default: 0 = GPU)
#   BG_DIR          Directory of existing background frames (skips capture)
#   TRAIN_COUNT     Number of training images (default: 2000)
#   VAL_COUNT       Number of validation images (default: 200)

set -eu

MAP="${1:?Usage: $0 <map_name> <monster1> [monster2 ...]}"
shift
MONSTERS=("$@")
if [ ${#MONSTERS[@]} -eq 0 ]; then
    echo "ERROR: at least one monster name required" >&2
    exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${YOLO_HOST:-test@192.168.66.63}"
VENV="${YOLO_VENV:-/home/test/yolo_venv}"
EPOCHS="${YOLO_EPOCHS:-50}"
BATCH="${YOLO_BATCH:-16}"
DEVICE="${YOLO_DEVICE:-0}"
TRAIN_COUNT="${TRAIN_COUNT:-2000}"
VAL_COUNT="${VAL_COUNT:-200}"

SPRITE_DIR="$REPO/monster_yolo"
BG_DIR="${BG_DIR:-$REPO/yolo_data/backgrounds_${MAP}}"
DATASET_DIR="$REPO/yolo_data/dataset_${MAP}"
MODEL_OUT="$REPO/models/${MAP}.onnx"

echo "=== Training YOLO model for map: $MAP ==="
echo "    Monsters: ${MONSTERS[*]}"
echo "    Sprites:  $SPRITE_DIR"
echo "    Backgrounds: $BG_DIR"
echo "    Dataset:  $DATASET_DIR"
echo "    Output:   $MODEL_OUT"
echo

# --- Step 1: Download sprites ---
echo "--- Step 1: Download sprites ---"
SPRITE_ARGS=()
for MOB in "${MONSTERS[@]}"; do
    MOB_DIR="$SPRITE_DIR/$MOB"
    SPRITE_ARGS+=("$MOB_DIR")
    if [ -d "$MOB_DIR" ] && [ "$(ls "$MOB_DIR"/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  $MOB: $(ls "$MOB_DIR"/*.png | wc -l) sprites already present"
    else
        echo "  Downloading $MOB sprites..."
        cd "$REPO"
        python -c "
import sys; sys.path.insert(0, '.')
from tools.mob_maker import get_all_mobs, find_mob_id, BASE_URL, DEFAULT_REGION, DEFAULT_VERSION
import requests, zipfile, io, cv2, numpy as np
from pathlib import Path

mobs = get_all_mobs()
mob_id = find_mob_id(mobs, '${MOB}'.replace('_', ' '))
if mob_id is None:
    print(f'ERROR: mob \"${MOB}\" not found in API', file=sys.stderr)
    sys.exit(1)
# Download with transparency (not green bg)
output_dir = Path('$MOB_DIR')
output_dir.mkdir(parents=True, exist_ok=True)
url = f'{BASE_URL}/api/{DEFAULT_REGION}/{DEFAULT_VERSION}/mob/{mob_id}/download'
r = requests.get(url, timeout=60)
r.raise_for_status()
z = zipfile.ZipFile(io.BytesIO(r.content))
idx = 1
for fn in z.namelist():
    if 'die1' in fn.lower(): continue
    data = z.read(fn)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim < 3 or img.shape[2] != 4: continue
    out = output_dir / f'${MOB}_{idx}.png'
    cv2.imwrite(str(out), img)
    idx += 1
print(f'  ${MOB}: {idx-1} sprites downloaded')
"
    fi
done

# --- Step 2: Background frames ---
echo "--- Step 2: Background frames ---"
mkdir -p "$BG_DIR"
EXISTING=$(ls "$BG_DIR"/*.png 2>/dev/null | wc -l)
if [ "$EXISTING" -ge 30 ]; then
    echo "  $EXISTING backgrounds already present, skipping capture"
else
    echo "  Need more backgrounds (have $EXISTING, need 30+)"
    CT="${YOLO_CONTAINER:-}"
    if [ -z "$CT" ]; then
        echo "  No YOLO_CONTAINER set; provide BG_DIR with backgrounds or set YOLO_CONTAINER"
        echo "  Example: YOLO_CONTAINER=wingpu-ct-103 $0 $MAP ${MONSTERS[*]}"
        exit 1
    fi
    NEED=$((40 - EXISTING))
    echo "  Capturing $NEED frames from $CT..."
    ssh "$HOST" "docker exec $CT python3 /work/MapleStoryAutoLevelUp/tools/wingpu_grab.py /work/frames_yolo_${MAP} $NEED 2"
    ssh "$HOST" "docker exec $CT tar cf - -C /work frames_yolo_${MAP}" | tar xf - -C "$(dirname "$BG_DIR")"
    if [ -d "$(dirname "$BG_DIR")/frames_yolo_${MAP}" ]; then
        mv "$(dirname "$BG_DIR")/frames_yolo_${MAP}"/* "$BG_DIR/"
        rmdir "$(dirname "$BG_DIR")/frames_yolo_${MAP}" 2>/dev/null || true
    fi
    echo "  Now have $(ls "$BG_DIR"/*.png 2>/dev/null | wc -l) backgrounds"
fi

# --- Step 3: Synthesize training data ---
echo "--- Step 3: Synthesize training data ---"
cd "$REPO"
python tools/yolo_synth.py \
    --backgrounds "$BG_DIR" \
    --sprites "${SPRITE_ARGS[@]}" \
    --output "$DATASET_DIR" \
    --count "$TRAIN_COUNT" --val-count "$VAL_COUNT" \
    --neg-pct 0.1 --seed 42

# --- Step 4: Upload dataset and train on host GPU ---
echo "--- Step 4: Train on host GPU ---"
REMOTE_DIR="/home/test/yolo_training/${MAP}"
echo "  Uploading dataset to host..."
tar cf - -C "$REPO" "yolo_data/dataset_${MAP}" | ssh "$HOST" "mkdir -p $REMOTE_DIR && tar xf - -C $REMOTE_DIR"

# Update dataset.yaml with host paths
ssh "$HOST" "cat > $REMOTE_DIR/yolo_data/dataset_${MAP}/dataset.yaml << YAML
path: $REMOTE_DIR/yolo_data/dataset_${MAP}
train: images/train
val: images/val
nc: 1
names:
  0: monster
YAML"

echo "  Training YOLOv8n (epochs=$EPOCHS, batch=$BATCH, device=$DEVICE)..."
ssh "$HOST" "cd $REMOTE_DIR && $VENV/bin/yolo detect train \
    model=yolov8n.pt \
    data=$REMOTE_DIR/yolo_data/dataset_${MAP}/dataset.yaml \
    epochs=$EPOCHS imgsz=640 batch=$BATCH workers=4 device=$DEVICE \
    project=$REMOTE_DIR/runs name=${MAP}_monsters exist_ok=True" 2>&1 | tail -20

# --- Step 5: Export to ONNX ---
echo "--- Step 5: Export to ONNX ---"
ssh "$HOST" "$VENV/bin/yolo export \
    model=$REMOTE_DIR/runs/${MAP}_monsters/weights/best.pt \
    format=onnx imgsz=640"

# --- Step 6: Copy model ---
echo "--- Step 6: Copy model to $MODEL_OUT ---"
mkdir -p "$(dirname "$MODEL_OUT")"
scp "$HOST:$REMOTE_DIR/runs/${MAP}_monsters/weights/best.onnx" "$MODEL_OUT"
MODEL_SIZE=$(ls -lh "$MODEL_OUT" | awk '{print $5}')
echo
echo "=== Done! ==="
echo "  Model: $MODEL_OUT ($MODEL_SIZE)"
echo "  Deploy: tools/wingpu_sync.sh && restart the bot with monster_detect.mode: yolo"
