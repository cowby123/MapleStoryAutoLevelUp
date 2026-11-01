'''
wingpu_grab.py - save guest screens as PNG test fixtures (read only: no input).

  python3 tools/wingpu_grab.py OUT_DIR [COUNT] [INTERVAL_S]

Run inside the VM's script container (or anywhere `import wingpu` works).
The frames are what the bot sees (the whole guest screen); feed them to
tools/wingpu_detect_test.py.
'''
import os
import sys
import time

import cv2
import wingpu


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    os.makedirs(out, exist_ok=True)
    vm = wingpu.connect(fps=5)
    try:
        for i in range(n):
            img = vm.frame(wait=2.0, new=True)
            if img is None:
                print("no frame", flush=True)
            else:
                p = os.path.join(out, time.strftime("f%Y%m%d_%H%M%S") + f"_{i:03d}.png")
                cv2.imwrite(p, img)
                print(p, img.shape, flush=True)
            time.sleep(interval)
    finally:
        vm.close()


if __name__ == "__main__":
    main()
