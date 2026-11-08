'''
Auto-dismiss the monster book (怪物圖鑑) popup. It blocks all controls and
the character dies. The golden corner of the book's frame is matched against
the screen; when found, Esc is pressed (up to max_retries times, one per
check_interval_s).

The template is misc/book_corner.png (the top-left golden border). It scores
0.0 on the book and 0.18+ on every normal frame tested (2026-11-09).
'''
import os
import time

import cv2
import numpy as np

from src.utils.logger import logger

_TPL = None
_TPL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "misc", "book_corner.png")


def _template():
    global _TPL
    if _TPL is None:
        p = _TPL_PATH
        if os.path.exists(p):
            _TPL = cv2.imread(p)
        else:
            logger.warning(f"[BookDismiss] template not found: {p}")
            _TPL = False
    return _TPL if _TPL is not False else None


def detect_book(frame, thres=0.06):
    '''True when the monster book popup is on screen'''
    tpl = _template()
    if tpl is None or frame is None:
        return False
    # The book is in the upper half of the screen, centred horizontally
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.1):int(h * 0.5), int(w * 0.15):int(w * 0.55)]
    if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
        return False
    r = cv2.matchTemplate(roi, tpl, cv2.TM_SQDIFF_NORMED)
    return float(np.nan_to_num(r, nan=1.0).min()) <= thres


class BookDismisser:
    '''
    Checks every check_interval_s frames; when the book is found, presses
    dismiss_key (default Esc) up to max_retries times. Resets after it is
    gone or after a cooldown.
    '''
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.t_last_check = 0.0
        self.t_last_press = 0.0
        self.retries = 0
        self.dismissed = 0        # total count
        self.active = False       # currently trying to dismiss
        self.t_cooldown = 0.0     # don't re-detect right after dismissing

    def reconfigure(self, cfg):
        self.cfg = cfg or {}

    @property
    def enabled(self):
        return bool(self.cfg.get("enable", True))

    def tick(self, frame, press_fn, now=None):
        '''
        Call on every frame (or every Nth). Returns True if the book is up
        and a key was / will be pressed. press_fn(key, duration) does the press
        (the bot decides whether to send it for real or only log it in dry run).
        '''
        now = time.time() if now is None else now
        if not self.enabled:
            self.active = False
            return False
        if now - self.t_cooldown < 1.0:
            return False
        interval = float(self.cfg.get("check_interval_s", 0.5))
        if now - self.t_last_check < interval:
            return self.active
        self.t_last_check = now
        found = detect_book(frame, float(self.cfg.get("thres", 0.06)))
        if not found:
            if self.active:
                logger.info(f"[BookDismiss] book gone after {self.retries} presses")
                self.active = False
                self.retries = 0
                self.t_cooldown = now
            return False
        max_retries = int(self.cfg.get("max_retries", 3))
        if self.retries >= max_retries:
            if not self.active:
                return False
            logger.warning(f"[BookDismiss] book still up after {max_retries} presses, giving up until it's gone")
            return True
        key = self.cfg.get("key", "esc")
        wait = float(self.cfg.get("press_interval_s", 0.4))
        if now - self.t_last_press >= wait:
            press_fn(key, 0.05)
            self.t_last_press = now
            self.retries += 1
            self.dismissed += 1
            self.active = True
            logger.info(f"[BookDismiss] book detected, pressing {key} ({self.retries}/{max_retries})")
        return True
