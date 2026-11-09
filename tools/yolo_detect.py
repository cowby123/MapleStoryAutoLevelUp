'''
yolo_detect.py - YOLO v8 nano monster detector (ONNX, CPU only).

Usage as a module:
    from tools.yolo_detect import YoloDetector
    det = YoloDetector("models/ston.onnx")
    det.warmup()
    results = det.detect(frame_bgr)
    # results: [{"name": "monster", "position": (x, y), "size": (h, w), "score": 0.87}, ...]

Standalone test:
    python tools/yolo_detect.py models/ston.onnx frame.png [--show] [--conf 0.25]
'''
import os
import sys
import time
import argparse

import cv2
import numpy as np

# onnxruntime: try GPU providers first, fall back to CPU
try:
    import onnxruntime as ort
except ImportError:
    ort = None


CLASS_NAMES = ["monster"]


def _letterbox(img, new_shape=640):
    '''
    Resize image with unchanged aspect ratio using padding (letterbox).
    Returns (padded_img, ratio, (pad_w, pad_h)).
    '''
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))  # (w, h)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    # Divide padding evenly
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def _nms(boxes, scores, iou_threshold):
    '''
    Non-maximum suppression. boxes: (N, 4) as x1,y1,x2,y2; scores: (N,).
    Returns indices to keep.
    '''
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class YoloDetector:
    '''
    YOLO v8 nano monster detector using ONNX Runtime (CPU).

    The model outputs shape (1, num_classes+4, num_anchors) in YOLO v8 format:
    rows 0-3 = cx, cy, w, h; row 4+ = class confidence (no objectness score
    in v8 -- the class conf IS the final confidence).
    '''

    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45,
                 input_size=640):
        if ort is None:
            raise ImportError("onnxruntime is not installed")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size

        # Prefer CPU; OpenVINO or DirectML if available
        providers = []
        available = ort.get_available_providers()
        for p in ["OpenVINOExecutionProvider", "DmlExecutionProvider",
                   "CPUExecutionProvider"]:
            if p in available:
                providers.append(p)
        if not providers:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2  # keep CPU usage bounded
        self.session = ort.InferenceSession(model_path, opts,
                                            providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def warmup(self):
        '''Run one dummy inference to trigger JIT / graph optimizations.'''
        dummy = np.zeros((1, 3, self.input_size, self.input_size),
                         dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    def detect(self, frame):
        '''
        Detect monsters in a BGR frame (e.g. 1366x768).
        Returns list of dicts:
            {"name": "monster", "position": (x, y), "size": (h, w),
             "score": float}
        where position is the top-left corner in frame pixel coordinates.
        '''
        orig_h, orig_w = frame.shape[:2]

        # --- Preprocess: letterbox + normalize + NCHW ---
        img, ratio, (pad_w, pad_h) = _letterbox(frame, self.input_size)
        img = img[:, :, ::-1]           # BGR -> RGB
        img = img.transpose(2, 0, 1)    # HWC -> CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[np.newaxis, ...]      # add batch dim: (1,3,H,W)

        # --- Inference ---
        outputs = self.session.run(None, {self.input_name: img})
        # YOLO v8 output: (1, 4+num_classes, num_predictions)
        pred = outputs[0]  # shape (1, 5, 8400) for 1 class

        # --- Postprocess ---
        # Transpose to (num_predictions, 4+num_classes)
        pred = pred[0].T  # (8400, 5)

        # Columns: cx, cy, w, h, class0_conf [, class1_conf, ...]
        cx = pred[:, 0]
        cy = pred[:, 1]
        bw = pred[:, 2]
        bh = pred[:, 3]
        # For single class, column 4 is the confidence
        if pred.shape[1] == 5:
            scores = pred[:, 4]
            class_ids = np.zeros(len(pred), dtype=int)
        else:
            # Multi-class: take the max
            class_scores = pred[:, 4:]
            class_ids = class_scores.argmax(axis=1)
            scores = class_scores[np.arange(len(pred)), class_ids]

        # Filter by confidence
        mask = scores > self.conf_thres
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return []

        # Convert cx,cy,w,h -> x1,y1,x2,y2 (still in letterboxed coordinates)
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = _nms(boxes, scores, self.iou_thres)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Scale back to original frame coordinates
        # Undo letterbox: subtract padding, divide by ratio
        boxes[:, 0] = (boxes[:, 0] - pad_w) / ratio
        boxes[:, 1] = (boxes[:, 1] - pad_h) / ratio
        boxes[:, 2] = (boxes[:, 2] - pad_w) / ratio
        boxes[:, 3] = (boxes[:, 3] - pad_h) / ratio

        # Clip to frame
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        # Build result list (same format as the bot's template-matching output)
        results = []
        for i in range(len(boxes)):
            bx1, by1, bx2, by2 = boxes[i]
            w = int(bx2 - bx1)
            h = int(by2 - by1)
            results.append({
                "name": CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else "monster",
                "position": (int(bx1), int(by1)),
                "size": (h, w),
                "score": float(scores[i]),
            })

        return results


# --------------- standalone test ---------------

def main():
    parser = argparse.ArgumentParser(description="YOLO monster detector test")
    parser.add_argument("model", help="Path to ONNX model")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--show", action="store_true", help="Display result")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IOU threshold for NMS (default 0.45)")
    args = parser.parse_args()

    det = YoloDetector(args.model, conf_thres=args.conf, iou_thres=args.iou)
    det.warmup()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    results = det.detect(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Detected {len(results)} monsters in {elapsed_ms:.1f} ms")
    for r in results:
        x, y = r["position"]
        h, w = r["size"]
        print(f"  {r['name']} score={r['score']:.3f} "
              f"pos=({x},{y}) size=({w}x{h})")

    # Draw boxes on image
    vis = frame.copy()
    for r in results:
        x, y = r["position"]
        h, w = r["size"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{r['name']} {r['score']:.2f}"
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    out_path = os.path.splitext(args.image)[0] + "_yolo.png"
    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}")

    if args.show:
        cv2.imshow("YOLO Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
