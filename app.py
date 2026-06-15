import sys
import subprocess
import importlib
from ultralytics import YOLO
_REQUIRED = {
    "cv2":         "opencv-python",
    "numpy":       "numpy",
    "torch":       "torch",
    "torchvision": "torchvision",
    "PIL":         "pillow",
    "ultralytics": "ultralytics",
    "flask":       "flask",
    "pygame":      "pygame",
}

def _auto_install():
    missing = []
    for module, package in _REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"\n[SETUP] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("[SETUP] Installation complete.\n")

_auto_install()

# ──────────────────────────────────────────────────────────────────────────────
#  STANDARD IMPORTS
# ──────────────────────────────────────────────────────────────────────────────

import os
import csv
import time
import math
import argparse
import warnings
import collections
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import threading
import json
import pygame
from flask import Flask, Response, jsonify

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    # Display
    "display_w":      1280,
    "display_h":      720,

    # YOLO (scene context)
    "yolo_model":     "yolov8n.pt",
    "yolo_conf":      0.40,
    "yolo_classes":   [0, 2, 5, 7],          # person, car, bus, truck

    # Fire colour detection (HSV)
    # Range 1: red-orange flames
    "fire_h1_lo":     np.array([0,   120, 120]),
    "fire_h1_hi":     np.array([22,  255, 255]),
    # Range 2: yellow-white core
    "fire_h2_lo":     np.array([22,   80, 180]),
    "fire_h2_hi":     np.array([40,  255, 255]),
    "fire_min_area":  600,                   # px² minimum blob area

    # Smoke detection — tightened to prevent false positives from movement
    "smoke_h_lo":     np.array([0,    0, 140]),   # higher brightness floor (was 100)
    "smoke_h_hi":     np.array([180, 35, 210]),   # lower saturation ceiling (was 50)
    "smoke_min_area": 15000,                       # was 2500 — needs a LARGE blob
    "smoke_diff_thr": 60,                          # was 22  — needs strong motion
    "smoke_blur":     51,                          # was 21  — heavy blur kills micro noise

    # Temporal smoothing — slower rise means brief motion can't spike to CRITICAL
    "smooth_alpha":   0.08,                        # was 0.35 — very slow rise

    # Performance
    "fp16":           True,
    "scale_w":        640,
    "depth_interval": 1,

    # Risk thresholds — all raised so movement noise never reaches CRITICAL
    "crit_thresh":    0.75,                        # was 0.55
    "warn_thresh":    0.55,                        # was 0.30
    "caut_thresh":    0.35,                        # was 0.10

    # Fonts
    "font":           cv2.FONT_HERSHEY_SIMPLEX,
    "font_mono":      cv2.FONT_HERSHEY_DUPLEX,
}

# ──────────────────────────────────────────────────────────────────────────────
#  EMAIL CONFIGURATION  ← fill in your details here
# ──────────────────────────────────────────────────────────────────────────────

EMAIL_CFG = {
    # ── sender (your Gmail) ───────────────────────────────────────────────────
    "enabled":        True,              # set False to disable all email alerts
    "sender_email":   "haripreetham.1111@gmail.com",
    "sender_password":"zeorhhbaxatxbllk",   # Gmail App Password (not your login password)
    "smtp_host":      "smtp.gmail.com",
    "smtp_port":      587,

    # ── recipients ────────────────────────────────────────────────────────────
    "recipients": [
        "haripreethambade@gmail.com",
        # "recipient2@gmail.com",        # add more if needed
    ],

    # ── alert rules ───────────────────────────────────────────────────────────
    # Which risk levels trigger an email
    "alert_on":       ["CRITICAL"],   # remove "WARNING" to only get CRITICAL

    # Minimum seconds between emails (prevents inbox flooding)
    "cooldown_sec":   60,

    # Attach screenshot to email?
    "attach_screenshot": True,
}

# ──────────────────────────────────────────────────────────────────────────────
#  YOLO FIRE MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

FIRE_YOLO_CFG = {
    # Path to your fire-trained YOLOv8 model
    "model_path":    "fire_yolo.pt",

    # Minimum confidence to accept a fire/smoke detection (0.0 - 1.0)
    "conf_thresh":   0.45,

    # Class names in your model — check what your model uses
    # Common ones: "fire", "smoke", "flame"
    # If your model only has one class (fire), just put ["fire"]
    "fire_classes":  ["fire"],
    "smoke_classes": ["smoke"],

    # Run YOLO fire detection every N frames (1 = every frame, 2 = every other)
    # Increase if CPU is too slow
    "interval":      2,

    # Fall back to HSV if YOLO model file not found
    "fallback_hsv":  True,
}

# ──────────────────────────────────────────────────────────────────────────────
#  MULTI-CAMERA CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

MULTI_CAM_CFG = {
    "enabled":  True,          # set False to use single camera (original behaviour)
    "sources": [0,1],        # webcam indices — add more e.g. [0, 1, 2]
    "labels":  ["LIVE-CAM", "FIRE-VIDEO"],   # display name for each camera
}

# ──────────────────────────────────────────────────────────────────────────────
#  SOUND ALARM CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SOUND_CFG = {
    # Set False to disable sound completely
    "enabled":         True,

    # Which risk levels trigger the alarm
    "alarm_on":        ["CRITICAL", "WARNING"],

    # Path to your alarm sound file (.mp3 or .wav)
    # If the file doesn't exist, a beep tone is generated automatically
    "alarm_file":      "alarm.mp3",

    # How many times to repeat the sound per trigger (0 = loop forever until risk clears)
    "loops":           2,

    # Volume: 0.0 (silent) to 1.0 (max)
    "volume":          0.9,

    # Seconds before the alarm can trigger again for the same risk level
    "cooldown_sec":    15,

    # Stop alarm automatically when risk drops back to CLEAR
    "stop_on_clear":   True,
    "sustain_sec":     3,     # ← ADD THIS: fire/smoke must persist for 3 seconds before alarm
}

# Neon colour palette (BGR)
COL = {
    "red":        (  0,  30, 255),
    "orange":     (  0, 130, 255),
    "yellow":     (  0, 215, 255),
    "green":      ( 40, 220,  40),
    "cyan":       (230, 220,   0),
    "blue":       (255, 100,  20),
    "magenta":    (220,   0, 200),
    "white":      (230, 230, 230),
    "dark":       (  8,  10,  14),
    "panel_bg":   ( 14,  16,  22),
    "fire_glow":  (  0,  80, 255),
    "smoke_glow": (160, 160, 180),
    "teal":       (200, 220,  40),
}

# ──────────────────────────────────────────────────────────────────────────────
#  FLASK WEB DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)

# Shared state between detection thread and Flask thread (thread-safe via lock)
_state_lock  = threading.Lock()
_shared = {
    "fire_conf":    0.0,
    "smoke_conf":   0.0,
    "risk":         "CLEAR",
    "fps":          0.0,
    "latency_ms":   0.0,
    "frame_idx":    0,
    "fire_count":   0,
    "smoke_count":  0,
    "scene_count":  0,
    "mode":         "CPU",
    "alert_log":    [],          # list of {"time", "risk", "fire", "smoke"}
}
_frame_lock  = threading.Lock()
_latest_jpeg = None              # latest JPEG bytes for streaming


def _update_state(**kwargs):
    with _state_lock:
        _shared.update(kwargs)


def _push_jpeg(canvas: np.ndarray):
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _frame_lock:
            _latest_jpeg = buf.tobytes()


def _gen_frames():
    """Generator for multipart JPEG stream."""
    while True:
        with _frame_lock:
            frame = _latest_jpeg
        if frame is None:
            import time as _t; _t.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@flask_app.route("/")
def index():
    return Response(open("dashboard.html").read(), mimetype="text/html")


@flask_app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.route("/api/state")
def api_state():
    with _state_lock:
        return jsonify(dict(_shared))


@flask_app.route("/api/alerts")
def api_alerts():
    with _state_lock:
        return jsonify(_shared["alert_log"][-50:])  # last 50 alerts


def start_dashboard(port=5000):
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)       # suppress Flask request logs
    flask_app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


# ──────────────────────────────────────────────────────────────────────────────
#  DEVICE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev  = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"[DEVICE] GPU: {name}  ({vram} MB VRAM)")
        return dev, True
    print("[DEVICE] No GPU found. Running on CPU.")
    return torch.device("cpu"), False

DEVICE, USE_GPU = get_device()
USE_FP16 = USE_GPU and CFG["fp16"]

# ──────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)
        self._last  = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now

    @property
    def fps(self):
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    @property
    def latency_ms(self):
        return (self._times[-1] * 1000.0) if self._times else 0.0


class ExpSmooth:
    """Single-value exponential moving average."""
    def __init__(self, alpha=0.35, init=0.0):
        self.alpha = alpha
        self._v    = init

    def update(self, x):
        self._v = self.alpha * x + (1.0 - self.alpha) * self._v
        return self._v

    @property
    def value(self):
        return self._v


# ──────────────────────────────────────────────────────────────────────────────
#  YOLO SCENE CONTEXT (people / vehicles)
# ──────────────────────────────────────────────────────────────────────────────

class SceneDetector:
    """YOLOv8 wrapper for general scene objects (people, vehicles)."""

    _LABELS = {0: "PERSON", 2: "CAR", 5: "BUS", 7: "TRUCK"}
    _COLORS = {0: COL["magenta"], 2: COL["cyan"],
               5: COL["orange"],  7: COL["teal"]}

    def __init__(self):
        from ultralytics import YOLO
        print(f"[YOLO] Loading {CFG['yolo_model']} ...")
        self.model = YOLO(CFG["yolo_model"])
        self.model.to(DEVICE)
        print("[YOLO] Ready.")

    def run(self, frame: np.ndarray) -> list:
        results = self.model(
            frame,
            conf=CFG["yolo_conf"],
            classes=CFG["yolo_classes"],
            verbose=False,
            half=USE_FP16,
        )
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls  = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "cls":  cls,
                    "conf": conf,
                    "box":  (x1, y1, x2, y2),
                })
        return detections


# ──────────────────────────────────────────────────────────────────────────────
#  FIRE DETECTOR  (HSV colour + morphology)
# ──────────────────────────────────────────────────────────────────────────────

class FireDetector:
    """
    Detects fire and flame using a YOLOv8 model trained specifically on fire.
    Falls back to HSV colour detection if model file is not found.

    Output format is identical to the old FireDetector so nothing else needs
    to change — same blobs list, same confidence score, same mask.
    """

    def __init__(self):
        self._conf_smooth = ExpSmooth(CFG["smooth_alpha"])
        self._model       = None
        self._use_yolo    = False
        self._frame_count = 0
        self._last_blobs  = []
        self._last_conf   = 0.0
        self._last_mask   = None

        model_path = FIRE_YOLO_CFG["model_path"]

        if os.path.exists(model_path):
            try:
                print(f"[FIRE-YOLO] Loading {model_path} ...")
                self._model    = YOLO(model_path)
                self._use_yolo = True
                print(f"[FIRE-YOLO] Ready. Classes: {list(self._model.names.values())}")
            except Exception as e:
                print(f"[FIRE-YOLO] ⚠ Failed to load: {e}")
                if FIRE_YOLO_CFG["fallback_hsv"]:
                    print("[FIRE-YOLO] Falling back to HSV detection.")
        else:
            if FIRE_YOLO_CFG["fallback_hsv"]:
                print(f"[FIRE-YOLO] '{model_path}' not found — using HSV fallback.")
            else:
                print(f"[FIRE-YOLO] ⚠ '{model_path}' not found and fallback disabled.")

        # HSV fallback smoother
        self._static_count  = {}
        self._mask_history  = deque(maxlen=8)

    # ── YOLO-based detection ──────────────────────────────────────────────────

    def _run_yolo(self, frame_bgr):
        h, w   = frame_bgr.shape[:2]
        mask   = np.zeros((h, w), dtype=np.uint8)
        blobs  = []
        total_area = 0

        results = self._model(
            frame_bgr,
            conf=FIRE_YOLO_CFG["conf_thresh"],
            verbose=False,
        )

        fire_names  = [n.lower() for n in FIRE_YOLO_CFG["fire_classes"]]
        smoke_names = [n.lower() for n in FIRE_YOLO_CFG["smoke_classes"]]

        for result in results:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                cls_name   = self._model.names[cls_id].lower()
                confidence = float(box.conf[0])

                # Only process fire/flame classes here
                # (smoke classes handled separately by SmokeDetector)
                if cls_name not in fire_names:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                area = (x2 - x1) * (y2 - y1)

                blobs.append({
                    "box":        (x1, y1, x2, y2),
                    "area":       area,
                    "contour":    None,
                    "confidence": confidence,
                    "source":     "yolo",
                })
                total_area += area

                # Draw filled rect on mask for compatibility
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        raw_conf = min(1.0, total_area / max(1, h * w * 0.05))

        # Also boost confidence from per-box scores
        if blobs:
            max_box_conf = max(b["confidence"] for b in blobs)
            raw_conf     = max(raw_conf, max_box_conf)

        conf = self._conf_smooth.update(raw_conf)
        return blobs, conf, mask

    # ── HSV fallback detection (same as original) ─────────────────────────────

    def _run_hsv(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        m1   = cv2.inRange(hsv, CFG["fire_h1_lo"], CFG["fire_h1_hi"])
        m2   = cv2.inRange(hsv, CFG["fire_h2_lo"], CFG["fire_h2_hi"])
        mask = cv2.bitwise_or(m1, m2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs      = []
        total_area = 0
        h, w       = frame_bgr.shape[:2]
        new_static = {}

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CFG["fire_min_area"]:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            box  = (x, y, x + bw, y + bh)

            roi_v = hsv[y:y+bh, x:x+bw, 2]
            if roi_v.size and roi_v.mean() < 200:
                continue

            # Flicker check
            if len(self._mask_history) >= 2:
                curr    = mask[y:y+bh, x:x+bw]
                prev    = self._mask_history[-1][y:y+bh, x:x+bw]
                if curr.shape == prev.shape and curr.size > 0:
                    changed = np.count_nonzero(cv2.absdiff(curr, prev))
                    flicker = changed / curr.size
                    key = (x//40, y//40, (x+bw)//40, (y+bh)//40)
                    if flicker < 0.04:
                        count = self._static_count.get(key, 0) + 1
                        new_static[key] = count
                        if count > 5:
                            continue
                    else:
                        new_static[key] = 0

            blobs.append({
                "box": box, "area": area,
                "contour": cnt, "source": "hsv"
            })
            total_area += area

        self._static_count = new_static
        self._mask_history.append(mask.copy())

        raw_conf = min(1.0, total_area / max(1, h * w * 0.05))
        conf     = self._conf_smooth.update(raw_conf)
        return blobs, conf, mask

    # ── public run() — same signature as before ───────────────────────────────

    def run(self, frame_bgr: np.ndarray):
        self._frame_count += 1

        # Skip frames based on interval setting (saves CPU)
        if self._use_yolo:
            if self._frame_count % FIRE_YOLO_CFG["interval"] == 0:
                self._last_blobs, self._last_conf, self._last_mask = \
                    self._run_yolo(frame_bgr)
            # Return cached result on skipped frames
            mask = self._last_mask if self._last_mask is not None \
                   else np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            return self._last_blobs, self._last_conf, mask
        else:
            return self._run_hsv(frame_bgr)

# ──────────────────────────────────────────────────────────────────────────────
#  SMOKE DETECTOR  (frame differencing + grey HSV masking)
# ──────────────────────────────────────────────────────────────────────────────

class SmokeDetector:
    """
    Detects smoke using temporal frame differencing combined with
    grey-tone HSV masking to isolate diffuse light-grey/white regions.
    """

    def __init__(self):
        self._prev_gray    = None
        self._conf_smooth  = ExpSmooth(CFG["smooth_alpha"])
        self._bg_sub       = cv2.createBackgroundSubtractorMOG2(
                                 history=800, varThreshold=120,
                                 detectShadows=False)
        self._consec_hits  = 0          # consecutive frames with smoke blobs
        self._CONSEC_MIN   = 8          # must see smoke for 8 frames in a row
        # Share the fire YOLO model for smoke class if available
        self._yolo_model  = None
        self._yolo_ready  = False
        self._frame_count  = 0
        self._last_blobs   = []
        self._last_conf    = 0.0
        self._last_mask    = None
    def set_yolo_model(self, model):
        """Accept the shared YOLO model from FireDetector."""
        self._yolo_model = model
        smoke_names = [n.lower() for n in FIRE_YOLO_CFG["smoke_classes"]]
        model_classes = [v.lower() for v in model.names.values()]
        self._yolo_ready = any(n in model_classes for n in smoke_names)
        if self._yolo_ready:
            print(f"[SMOKE-YOLO] Using YOLO model for smoke detection too.")
    def run(self, frame_bgr: np.ndarray):
        # ── YOLO (primary if model has smoke class) ───────────────────────────
        if self._yolo_ready and self._yolo_model is not None:
            self._frame_count += 1
            if self._frame_count % FIRE_YOLO_CFG["interval"] == 0:
                h, w = frame_bgr.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                blobs = []; total_area = 0
                smoke_names = [n.lower() for n in FIRE_YOLO_CFG["smoke_classes"]]
                results = self._yolo_model(frame_bgr,
                              conf=FIRE_YOLO_CFG["conf_thresh"], verbose=False)
                for result in results:
                    for box in result.boxes:
                        cls_name   = self._yolo_model.names[int(box.cls[0])].lower()
                        confidence = float(box.conf[0])
                        if cls_name not in smoke_names: continue
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                        area = (x2-x1)*(y2-y1)
                        blobs.append({"box":(x1,y1,x2,y2),"area":area,
                                      "contour":None,"confidence":confidence})
                        total_area += area
                        cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)
                raw_conf = min(1.0, total_area/max(1, h*w*0.08))
                if blobs:
                    raw_conf = max(raw_conf, max(b["confidence"] for b in blobs))
                self._last_blobs = blobs
                self._last_conf  = self._conf_smooth.update(raw_conf)
                self._last_mask  = mask
            mask = self._last_mask if self._last_mask is not None \
                   else np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            return self._last_blobs, self._last_conf, mask

        # ── MOG2 fallback ─────────────────────────────────────────────────────
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]

        # Motion mask via MOG2
        fg_mask = self._bg_sub.apply(frame_bgr)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE,
                                   np.ones((9, 9), np.uint8), iterations=2)

        # Colour mask: low-saturation, mid-high brightness (grey/white smoke)
        colour_mask = cv2.inRange(hsv, CFG["smoke_h_lo"], CFG["smoke_h_hi"])

        # Frame differencing (extra motion cue)
        if self._prev_gray is not None:
            diff  = cv2.absdiff(gray, self._prev_gray)
            diff  = cv2.GaussianBlur(diff, (CFG["smoke_blur"], CFG["smoke_blur"]), 0)
            _, diff_mask = cv2.threshold(diff, CFG["smoke_diff_thr"], 255, cv2.THRESH_BINARY)
        else:
            diff_mask = np.zeros_like(gray)
        self._prev_gray = gray.copy()

        # Combined: must be motion AND grey-toned
        combined = cv2.bitwise_and(colour_mask,
                                   cv2.bitwise_or(fg_mask, diff_mask))

        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs = []
        total_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CFG["smoke_min_area"]:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            blobs.append({
                "box":     (x, y, x + bw, y + bh),
                "area":    area,
                "contour": cnt,
            })
            total_area += area

        raw_conf = min(1.0, total_area / max(1, h * w * 0.08))

        # Gate: only feed confidence if smoke appears in N consecutive frames
        if blobs:
            self._consec_hits += 1
        else:
            self._consec_hits = 0   # reset immediately when blobs disappear

        gated_conf = raw_conf if self._consec_hits >= self._CONSEC_MIN else 0.0
        conf = self._conf_smooth.update(gated_conf)

        return blobs, conf, combined


# ──────────────────────────────────────────────────────────────────────────────
#  RISK CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def classify_risk(fire_conf: float, smoke_conf: float) -> str:
    combined = max(fire_conf, smoke_conf * 0.7 + fire_conf * 0.3)
    if combined >= CFG["crit_thresh"]:
        return "CRITICAL"
    elif combined >= CFG["warn_thresh"]:
        return "WARNING"
    elif combined >= CFG["caut_thresh"]:
        return "CAUTION"
    return "CLEAR"

def risk_color(risk: str) -> tuple:
    return {
        "CRITICAL": COL["red"],
        "WARNING":  COL["orange"],
        "CAUTION":  COL["yellow"],
        "CLEAR":    COL["green"],
    }.get(risk, COL["white"])


# ──────────────────────────────────────────────────────────────────────────────
#  SCENE-AWARE BLOB FILTER
# ──────────────────────────────────────────────────────────────────────────────

def _box_overlap_ratio(boxA, boxB) -> float:
    """Return what fraction of boxA overlaps with boxB (0.0 – 1.0)."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / areaA


def filter_blobs_by_scene(blobs: list, scene_dets: list,
                           overlap_thresh: float = 0.35) -> list:
    """
    Remove any smoke/fire blob that overlaps too much with a YOLO-detected
    person or vehicle. A blob whose box overlaps ≥35% with a person box
    is almost certainly the person's body — not smoke or fire.

    Returns the filtered list of blobs (may be empty).
    """
    if not scene_dets or not blobs:
        return blobs

    scene_boxes = [d["box"] for d in scene_dets]
    kept = []
    for blob in blobs:
        dominated = any(
            _box_overlap_ratio(blob["box"], sb) >= overlap_thresh
            for sb in scene_boxes
        )
        if not dominated:
            kept.append(blob)
    return kept


# ──────────────────────────────────────────────────────────────────────────────
#  HUD RENDERER
# ──────────────────────────────────────────────────────────────────────────────

class HUDRenderer:
    """
    Renders all visual elements:
      - Fire / smoke bounding boxes with glow outlines
      - Hazard zone fills
      - Top status bar
      - Bottom telemetry bar
      - Side status panel
      - Animated DANGER / WARNING banners
      - Scene object boxes (people, vehicles)
    """

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self._flash_frame = 0

    # ── low-level helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _alpha_rect(canvas, pt1, pt2, color, alpha,
                    border_col=None, border_t=1):
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2 = min(canvas.shape[1] - 1, pt2[0])
        y2 = min(canvas.shape[0] - 1, pt2[1])
        if x2 <= x1 or y2 <= y1:
            return
        roi     = canvas[y1:y2, x1:x2]
        fill    = np.full_like(roi, color)
        canvas[y1:y2, x1:x2] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)
        if border_col:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_t)

    @staticmethod
    def _text(canvas, txt, pos, scale=0.44, color=COL["white"],
              thickness=1, font=None):
        if font is None:
            font = CFG["font"]
        cv2.putText(canvas, txt, (pos[0]+1, pos[1]+1),
                    font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, txt, pos, font, scale, color,
                    thickness, cv2.LINE_AA)

    def _glow_rect(self, canvas, x1, y1, x2, y2, color, layers=3):
        """Draw a multi-layer glow outline around a rectangle."""
        for i in range(layers, 0, -1):
            alpha = 0.15 * i
            pad   = i * 2
            c_dim = tuple(max(0, int(v * (0.3 + 0.7 * alpha))) for v in color)
            cv2.rectangle(canvas,
                          (x1 - pad, y1 - pad),
                          (x2 + pad, y2 + pad),
                          c_dim, 1 + i, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # ── fire boxes ───────────────────────────────────────────────────────────

    def draw_fire_blobs(self, canvas, blobs, conf):
        for b in blobs:
            x1, y1, x2, y2 = b["box"]
            # Hazard zone fill
            self._alpha_rect(canvas, (x1, y1), (x2, y2),
                             COL["fire_glow"], 0.22)
            # Glow outline
            self._glow_rect(canvas, x1, y1, x2, y2, COL["red"])
            # Corner markers
            cl = 14
            for (px, py), (dx, dy) in [
                ((x1, y1), ( cl,  cl)),
                ((x2, y1), (-cl,  cl)),
                ((x1, y2), ( cl, -cl)),
                ((x2, y2), (-cl, -cl)),
            ]:
                cv2.line(canvas, (px, py), (px + dx, py), COL["red"],  2)
                cv2.line(canvas, (px, py), (px, py + dy), COL["red"],  2)
            # Label
            lbl = f"FIRE  {conf:.0%}"
            tw  = cv2.getTextSize(lbl, CFG["font"], 0.46, 1)[0][0]
            bx  = x1
            self._alpha_rect(canvas, (bx, y1 - 22), (bx + tw + 10, y1),
                             COL["dark"], 0.82, COL["red"])
            self._text(canvas, lbl, (bx + 4, y1 - 6), 0.46, COL["red"])

    # ── smoke boxes ──────────────────────────────────────────────────────────

    def draw_smoke_blobs(self, canvas, blobs, conf):
        for b in blobs:
            x1, y1, x2, y2 = b["box"]
            self._alpha_rect(canvas, (x1, y1), (x2, y2),
                             COL["smoke_glow"], 0.18)
            self._glow_rect(canvas, x1, y1, x2, y2, COL["smoke_glow"])
            lbl = f"SMOKE {conf:.0%}"
            tw  = cv2.getTextSize(lbl, CFG["font"], 0.46, 1)[0][0]
            self._alpha_rect(canvas, (x1, y1 - 22), (x1 + tw + 10, y1),
                             COL["dark"], 0.82, COL["smoke_glow"])
            self._text(canvas, lbl, (x1 + 4, y1 - 6), 0.46, COL["smoke_glow"])

    # ── scene objects (YOLO) ─────────────────────────────────────────────────

    def draw_scene_objects(self, canvas, detections):
        _LABELS = SceneDetector._LABELS
        _COLORS = SceneDetector._COLORS
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cls   = d["cls"]
            conf  = d["conf"]
            color = _COLORS.get(cls, COL["white"])
            label = _LABELS.get(cls, "OBJ")
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            lbl = f"{label} {conf:.0%}"
            self._alpha_rect(canvas, (x1, y1 - 18), (x1 + 110, y1),
                             COL["panel_bg"], 0.75, color)
            self._text(canvas, lbl, (x1 + 3, y1 - 4), 0.36, color)

    # ── top status bar ────────────────────────────────────────────────────────

    def draw_top_bar(self, canvas, fps, latency_ms, frame_idx,
                     fire_conf, smoke_conf, risk, mode_str):
        self._alpha_rect(canvas, (0, 0), (self.w, 32),
                         COL["dark"], 0.90, COL["red"])
        rc = risk_color(risk)
        self._text(canvas, "INDUSTRIAL FIRE & SMOKE DETECTION AI",
                   (8, 21), 0.50, COL["red"], font=CFG["font_mono"])
        self._text(canvas, f"FPS:{fps:5.1f}", (390, 21), 0.42, COL["green"])
        self._text(canvas, f"LAT:{latency_ms:5.1f}ms", (470, 21), 0.42, COL["green"])
        self._text(canvas, f"FRAME:{frame_idx:05d}", (580, 21), 0.42, COL["white"])
        self._text(canvas, f"RISK: {risk}", (700, 21), 0.46, rc)
        mode_x = self.w - 160
        self._text(canvas, mode_str, (mode_x, 21), 0.40, COL["orange"])
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self._text(canvas, ts, (self.w - 168, 21), 0.34, (70, 80, 90))

    # ── side status panel ─────────────────────────────────────────────────────

    def draw_side_panel(self, canvas, fire_conf, smoke_conf,
                        risk, fire_count, smoke_count, scene_count):
        pw, ph = 195, 210
        px = self.w - pw - 4
        py = 36
        self._alpha_rect(canvas, (px, py), (px + pw, py + ph),
                         COL["panel_bg"], 0.85, COL["red"])

        self._text(canvas, "HAZARD STATUS", (px + 8, py + 16),
                   0.42, COL["red"], font=CFG["font_mono"])

        rc = risk_color(risk)
        items = [
            ("FIRE  CONF", f"{fire_conf:5.1%}",
             COL["red"] if fire_conf > CFG["caut_thresh"] else COL["green"]),
            ("SMOKE CONF", f"{smoke_conf:5.1%}",
             COL["orange"] if smoke_conf > CFG["caut_thresh"] else COL["green"]),
            ("RISK LEVEL", risk,  rc),
            ("FIRE  ZONES", str(fire_count),  COL["yellow"]),
            ("SMOKE ZONES", str(smoke_count), COL["smoke_glow"]),
            ("SCENE OBJ",  str(scene_count),  COL["cyan"]),
            ("GPU MODE",   "ON" if USE_GPU else "OFF",
             COL["green"] if USE_GPU else COL["orange"]),
        ]
        y = py + 34
        for label, val, col in items:
            self._text(canvas, label, (px + 8, y), 0.34, (120, 130, 140))
            self._text(canvas, val,   (px + 128, y), 0.38, col)
            y += 24

        # Confidence bars
        for label, conf, col in [
            ("FIRE",  fire_conf,  COL["red"]),
            ("SMOKE", smoke_conf, COL["smoke_glow"]),
        ]:
            self._text(canvas, label, (px + 8, y), 0.32, (100, 110, 120))
            bar_x = px + 60
            bar_y = y - 8
            bar_w = pw - 68
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                          (30, 35, 42), -1)
            fill_w = int(bar_w * conf)
            if fill_w > 0:
                cv2.rectangle(canvas, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + 8), col, -1)
            y += 18

    # ── bottom telemetry bar ──────────────────────────────────────────────────

    def draw_bottom_bar(self, canvas, fire_conf, smoke_conf, risk):
        bar_y = self.h - 24
        self._alpha_rect(canvas, (0, bar_y), (self.w, self.h),
                         COL["dark"], 0.90, COL["red"])
        rc = risk_color(risk)
        items = [
            (f"FIRE: {fire_conf:.0%}",
             COL["red"] if fire_conf > CFG["caut_thresh"] else COL["green"]),
            (f"SMOKE: {smoke_conf:.0%}",
             COL["orange"] if smoke_conf > CFG["caut_thresh"] else COL["green"]),
            (f"ALERT: {risk}", rc),
            (f"MODE: {'FP16 GPU' if USE_FP16 else 'GPU' if USE_GPU else 'CPU'}",
             COL["cyan"]),
        ]
        x = 8
        for txt, col in items:
            self._text(canvas, txt, (x, self.h - 7), 0.36, col)
            x += len(txt) * 8 + 20
        self._text(canvas,
                   "Dev/Creator: Bade Hari Preetham  |  github.com/yourusername",
                   (self.w - 330, self.h - 7), 0.30, (50, 60, 70))

    # ── animated DANGER banner ────────────────────────────────────────────────

    def draw_danger_banner(self, canvas, risk, fire_conf, smoke_conf):
        self._flash_frame += 1
        if risk not in ("CRITICAL", "WARNING"):
            return
        # Flash on/off
        if (self._flash_frame // 12) % 2 == 0 and risk == "CRITICAL":
            return

        bw, bh = 360, 56
        bx = (self.w - bw) // 2
        by = 36
        rc = risk_color(risk)
        self._alpha_rect(canvas, (bx, by), (bx + bw, by + bh),
                         rc, 0.25, rc, 2)
        # Inner border pulse
        pulse = abs(math.sin(self._flash_frame * 0.15))
        border_col = tuple(int(v * (0.5 + 0.5 * pulse)) for v in rc)
        cv2.rectangle(canvas, (bx + 3, by + 3),
                      (bx + bw - 3, by + bh - 3), border_col, 1)

        banner_txt = f"  {risk} -- HAZARD DETECTED  "
        self._text(canvas, banner_txt, (bx + 14, by + 22),
                   0.62, COL["white"], 2)
        sub = f"Fire: {fire_conf:.0%}   Smoke: {smoke_conf:.0%}"
        self._text(canvas, sub, (bx + 60, by + 44), 0.42, rc)

    # ── scan-line overlay (cinematic effect) ──────────────────────────────────

    @staticmethod
    def draw_scanlines(canvas, alpha=0.06):
        h, w = canvas.shape[:2]
        lines = np.zeros((h, w, 3), dtype=np.uint8)
        lines[::3, :] = 30
        cv2.addWeighted(lines, alpha, canvas, 1.0, 0, canvas)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
#  EMAIL ALERTER
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
#  SUSTAINED ALERT GATE
# ──────────────────────────────────────────────────────────────────────────────

class SustainedAlertGate:
    """
    Prevents alert spam by requiring a risk level to be sustained
    for a minimum number of seconds before passing it through.

    CLEAR resets all timers immediately.
    Downgrade (CRITICAL → WARNING) also resets the timer.

    Usage:
        gate = SustainedAlertGate(sustain_sec=3)
        gated_risk = gate.update(raw_risk)   # use gated_risk for alerts
    """

    def __init__(self, sustain_sec: float = 3.0):
        self._sustain    = sustain_sec
        self._first_seen = {}    # risk_level → time first seen this streak
        self._current    = "CLEAR"
        self._confirmed  = "CLEAR"

    def update(self, risk: str) -> str:
        now = time.time()

        if risk == "CLEAR":
            # Reset everything immediately
            self._first_seen = {}
            self._current    = "CLEAR"
            self._confirmed  = "CLEAR"
            return "CLEAR"

        if risk != self._current:
            # Risk level changed — start fresh timer for new level
            self._first_seen = {risk: now}
            self._current    = risk
            # Don't immediately confirm the new level
            return self._confirmed

        # Same risk level — check if it's been sustained long enough
        first = self._first_seen.get(risk, now)
        if (now - first) >= self._sustain:
            self._confirmed = risk   # confirmed! pass it through

        return self._confirmed
    
# ──────────────────────────────────────────────────────────────────────────────
#  SESSION REPORT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class SessionReportGenerator:
    """
    Reads the session's CSV log and generates a self-contained HTML report.
    Call generate() once at the end of the session.
    """

    def __init__(self, log_dir="detection_logs"):
        self._log_dir = log_dir
        self._session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate(self, frame_csv_path: str, alert_csv_path: str,
                 screenshot_dir: str, session_ts: str):
        """Build and save the HTML report. Returns the report file path."""

        # ── read frame CSV ────────────────────────────────────────────────────
        frames = []
        try:
            with open(frame_csv_path, newline="") as f:
                frames = list(csv.DictReader(f))
        except Exception as e:
            print(f"[REPORT] Could not read frame log: {e}")
            return None

        # ── read alert CSV ────────────────────────────────────────────────────
        alerts = []
        try:
            with open(alert_csv_path, newline="") as f:
                alerts = list(csv.DictReader(f))
        except Exception:
            pass

        if not frames:
            print("[REPORT] No frame data — report skipped.")
            return None

        # ── compute summary stats ─────────────────────────────────────────────
        total_frames   = len(frames)
        fire_vals      = [float(r["fire_conf"])  for r in frames]
        smoke_vals     = [float(r["smoke_conf"]) for r in frames]
        risk_counts    = {"CLEAR": 0, "CAUTION": 0, "WARNING": 0, "CRITICAL": 0}
        for r in frames:
            risk_counts[r.get("risk", "CLEAR")] = \
                risk_counts.get(r.get("risk", "CLEAR"), 0) + 1

        max_fire   = max(fire_vals)  if fire_vals  else 0
        max_smoke  = max(smoke_vals) if smoke_vals else 0
        avg_fire   = sum(fire_vals)  / len(fire_vals)  if fire_vals  else 0
        avg_smoke  = sum(smoke_vals) / len(smoke_vals) if smoke_vals else 0

        start_time = frames[0]["timestamp"]  if frames else "--"
        end_time   = frames[-1]["timestamp"] if frames else "--"

        fps_vals = []
        for r in frames:
            try: fps_vals.append(float(r["fps"]))
            except: pass
        avg_fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0

        total_alerts   = len(alerts)
        alert_critical = sum(1 for a in alerts if a.get("risk") == "CRITICAL")
        alert_warning  = sum(1 for a in alerts if a.get("risk") == "WARNING")
        alert_caution  = sum(1 for a in alerts if a.get("risk") == "CAUTION")

        # ── build chart data (sample every 10th frame for smaller HTML) ───────
        step       = max(1, total_frames // 200)
        sampled    = frames[::step]
        chart_labels  = [r["time"] for r in sampled]
        chart_fire    = [round(float(r["fire_conf"])  * 100, 1) for r in sampled]
        chart_smoke   = [round(float(r["smoke_conf"]) * 100, 1) for r in sampled]

        # ── screenshot thumbnails (last 6 CRITICAL ones) ──────────────────────
        crit_shots = [
            a["screenshot"] for a in alerts
            if a.get("risk") == "CRITICAL" and a.get("screenshot")
        ][-6:]

        thumb_html = ""
        for shot in crit_shots:
            shot_path = os.path.join(screenshot_dir, shot)
            if os.path.exists(shot_path):
                import base64
                with open(shot_path, "rb") as img_f:
                    b64 = base64.b64encode(img_f.read()).decode()
                thumb_html += f'''
                <div class="thumb">
                  <img src="data:image/jpeg;base64,{b64}" alt="{shot}"/>
                  <div class="thumb-label">{shot[:19]}</div>
                </div>'''

        # ── alert log rows ────────────────────────────────────────────────────
        alert_rows = ""
        for a in reversed(alerts[-50:]):
            risk    = a.get("risk", "")
            color   = {"CRITICAL":"#ff2d2d","WARNING":"#ff8c00","CAUTION":"#ffd400"}.get(risk,"#aaa")
            alert_rows += f"""
            <tr>
              <td>{a.get('time','')}</td>
              <td style="color:{color};font-weight:600;">{risk}</td>
              <td>{a.get('fire_conf','')}</td>
              <td>{a.get('smoke_conf','')}</td>
              <td>{a.get('fire_zones','')}</td>
              <td>{a.get('smoke_zones','')}</td>
            </tr>"""

        # ── risk distribution bar widths ──────────────────────────────────────
        def pct(n): return round(n / total_frames * 100, 1) if total_frames else 0
        pct_clear    = pct(risk_counts.get("CLEAR",    0))
        pct_caution  = pct(risk_counts.get("CAUTION",  0))
        pct_warning  = pct(risk_counts.get("WARNING",  0))
        pct_critical = pct(risk_counts.get("CRITICAL", 0))

        # ── render HTML ───────────────────────────────────────────────────────
        report_path = os.path.join(
            self._log_dir, f"report_{session_ts}.html")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Detection Report — {session_ts}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',sans-serif;background:#080b10;color:#c9d1d9;padding:24px}}
  h1{{font-size:1.1rem;letter-spacing:.12em;color:#ff2d2d;text-transform:uppercase;margin-bottom:4px}}
  h2{{font-size:.7rem;letter-spacing:.1em;color:#4a5568;text-transform:uppercase;
      margin:28px 0 12px;border-bottom:1px solid #1c2230;padding-bottom:6px}}
  .meta{{font-size:.72rem;color:#4a5568;font-family:monospace;margin-bottom:20px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:8px}}
  .card{{background:#0d1117;border:1px solid #1c2230;border-radius:6px;padding:14px}}
  .card-label{{font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:#4a5568;margin-bottom:6px}}
  .card-value{{font-size:1.5rem;font-weight:700;font-family:monospace}}
  .red{{color:#ff2d2d}} .orange{{color:#ff8c00}} .yellow{{color:#ffd400}}
  .green{{color:#00e676}} .cyan{{color:#00e5ff}} .smoke{{color:#a0aec0}}
  .chart-wrap{{background:#0d1117;border:1px solid #1c2230;border-radius:6px;
               padding:16px;margin-bottom:8px}}
  canvas{{max-height:220px}}
  .risk-bar-wrap{{background:#0d1117;border:1px solid #1c2230;border-radius:6px;padding:16px}}
  .risk-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:.72rem}}
  .risk-row-label{{width:70px;font-family:monospace}}
  .risk-track{{flex:1;background:#1c2230;border-radius:3px;height:8px;overflow:hidden}}
  .risk-fill{{height:100%;border-radius:3px;transition:width .3s}}
  .risk-pct{{width:38px;text-align:right;font-family:monospace;color:#4a5568}}
  table{{width:100%;border-collapse:collapse;font-size:.72rem;font-family:monospace}}
  th{{text-align:left;padding:7px 10px;color:#4a5568;border-bottom:1px solid #1c2230;
      font-size:.6rem;letter-spacing:.08em;text-transform:uppercase}}
  td{{padding:6px 10px;border-bottom:1px solid #0d1117}}
  tr:hover td{{background:rgba(255,255,255,.02)}}
  .thumbs{{display:flex;flex-wrap:wrap;gap:10px;margin-top:4px}}
  .thumb{{border:1px solid #1c2230;border-radius:4px;overflow:hidden;width:180px}}
  .thumb img{{width:100%;display:block}}
  .thumb-label{{font-size:.60rem;font-family:monospace;color:#4a5568;
                padding:4px 6px;background:#0d1117;text-align:center}}
  footer{{margin-top:28px;font-size:.62rem;color:#1c2230;font-family:monospace;text-align:center}}
</style>
</head>
<body>

<h1>&#9632; Industrial Fire &amp; Smoke Detection — Session Report</h1>
<div class="meta">
  Session: {session_ts} &nbsp;|&nbsp;
  Start: {start_time} &nbsp;|&nbsp;
  End: {end_time} &nbsp;|&nbsp;
  Dev: Bade Hari Preetham
</div>

<h2>Session Overview</h2>
<div class="cards">
  <div class="card">
    <div class="card-label">Total Frames</div>
    <div class="card-value cyan">{total_frames:,}</div>
  </div>
  <div class="card">
    <div class="card-label">Avg FPS</div>
    <div class="card-value cyan">{avg_fps:.1f}</div>
  </div>
  <div class="card">
    <div class="card-label">Total Alerts</div>
    <div class="card-value {'red' if total_alerts else 'green'}">{total_alerts}</div>
  </div>
  <div class="card">
    <div class="card-label">Critical</div>
    <div class="card-value red">{alert_critical}</div>
  </div>
  <div class="card">
    <div class="card-label">Warning</div>
    <div class="card-value orange">{alert_warning}</div>
  </div>
  <div class="card">
    <div class="card-label">Caution</div>
    <div class="card-value yellow">{alert_caution}</div>
  </div>
  <div class="card">
    <div class="card-label">Max Fire</div>
    <div class="card-value red">{max_fire:.0%}</div>
  </div>
  <div class="card">
    <div class="card-label">Max Smoke</div>
    <div class="card-value smoke">{max_smoke:.0%}</div>
  </div>
  <div class="card">
    <div class="card-label">Avg Fire</div>
    <div class="card-value red">{avg_fire:.1%}</div>
  </div>
  <div class="card">
    <div class="card-label">Avg Smoke</div>
    <div class="card-value smoke">{avg_smoke:.1%}</div>
  </div>
</div>

<h2>Confidence Trend</h2>
<div class="chart-wrap">
  <canvas id="trendChart"></canvas>
</div>

<h2>Risk Distribution</h2>
<div class="risk-bar-wrap">
  <div class="risk-row">
    <span class="risk-row-label green">CLEAR</span>
    <div class="risk-track"><div class="risk-fill" style="width:{pct_clear}%;background:#00e676"></div></div>
    <span class="risk-pct">{pct_clear}%</span>
  </div>
  <div class="risk-row">
    <span class="risk-row-label yellow">CAUTION</span>
    <div class="risk-track"><div class="risk-fill" style="width:{pct_caution}%;background:#ffd400"></div></div>
    <span class="risk-pct">{pct_caution}%</span>
  </div>
  <div class="risk-row">
    <span class="risk-row-label orange">WARNING</span>
    <div class="risk-track"><div class="risk-fill" style="width:{pct_warning}%;background:#ff8c00"></div></div>
    <span class="risk-pct">{pct_warning}%</span>
  </div>
  <div class="risk-row">
    <span class="risk-row-label red">CRITICAL</span>
    <div class="risk-track"><div class="risk-fill" style="width:{pct_critical}%;background:#ff2d2d"></div></div>
    <span class="risk-pct">{pct_critical}%</span>
  </div>
</div>

{'<h2>Critical Screenshots</h2><div class="thumbs">' + thumb_html + '</div>' if thumb_html else ''}

<h2>Alert Log (last 50)</h2>
{'<p style="color:#4a5568;font-size:.72rem;font-family:monospace;padding:10px 0">No alerts this session.</p>' if not alerts else f'''
<table>
  <thead><tr>
    <th>Time</th><th>Risk</th>
    <th>Fire</th><th>Smoke</th>
    <th>Fire Zones</th><th>Smoke Zones</th>
  </tr></thead>
  <tbody>{alert_rows}</tbody>
</table>'''}

<footer>Bade Hari Preetham &nbsp;|&nbsp; Industrial Fire &amp; Smoke Detection System &nbsp;|&nbsp; {session_ts}</footer>

<script>
const ctx = document.getElementById('trendChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {chart_labels},
    datasets: [
      {{
        label: 'Fire %',
        data: {chart_fire},
        borderColor: '#ff2d2d',
        backgroundColor: 'rgba(255,45,45,0.05)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      }},
      {{
        label: 'Smoke %',
        data: {chart_smoke},
        borderColor: '#a0aec0',
        backgroundColor: 'rgba(160,174,192,0.05)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 11 }} }} }},
      tooltip: {{ mode: 'index', intersect: false }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#4a5568', maxTicksLimit: 10, font: {{ size: 10 }} }},
        grid:  {{ color: '#1c2230' }}
      }},
      y: {{
        min: 0, max: 100,
        ticks: {{ color: '#4a5568', font: {{ size: 10 }},
                  callback: v => v + '%' }},
        grid: {{ color: '#1c2230' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[REPORT] ✅ Report saved: {report_path}")
        return report_path
# ──────────────────────────────────────────────────────────────────────────────
#  SOUND ALARMER
# ──────────────────────────────────────────────────────────────────────────────

class SoundAlarmer:
    """
    Plays an alarm sound when risk level hits WARNING or CRITICAL.
    - Uses pygame.mixer for reliable cross-platform audio.
    - If no alarm.mp3 is found, generates a beep tone automatically.
    - Runs non-blocking: sound plays in background, never slows the video loop.
    - Stops automatically when risk returns to CLEAR (if stop_on_clear=True).
    - Respects per-level cooldown so it doesn't loop every single frame.
    """

    def __init__(self):
        self._enabled   = SOUND_CFG["enabled"]
        self._playing   = False
        self._last_play = {}        # risk → last play timestamp
        self._lock      = threading.Lock()
        self._ready     = False

        if not self._enabled:
            print("[SOUND] Sound alarm disabled.")
            return

        try:
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.init()

            alarm_path = SOUND_CFG["alarm_file"]

            if os.path.exists(alarm_path):
                pygame.mixer.music.load(alarm_path)
                print(f"[SOUND] Loaded alarm file: {alarm_path}")
            else:
                # No file found — generate a beep tone and save it
                print(f"[SOUND] '{alarm_path}' not found — generating beep tone...")
                beep_path = self._generate_beep()
                pygame.mixer.music.load(beep_path)
                print(f"[SOUND] Generated beep: {beep_path}")

            pygame.mixer.music.set_volume(SOUND_CFG["volume"])
            self._ready = True
            print(f"[SOUND] Alarm ready  | triggers on: {SOUND_CFG['alarm_on']}"
                  f"  | cooldown: {SOUND_CFG['cooldown_sec']}s"
                  f"  | volume: {SOUND_CFG['volume']}")

        except Exception as e:
            print(f"[SOUND] ⚠ pygame init failed: {e} — sound disabled.")
            self._enabled = False

    # ── generate a beep WAV if no alarm file exists ───────────────────────────

    def _generate_beep(self) -> str:
        """
        Creates a pulsing beep tone using numpy and saves it as alarm_beep.wav.
        No external files needed — works out of the box.
        """
        import wave, struct
        sample_rate = 44100
        duration    = 0.8       # seconds per beep
        freq        = 880       # Hz (high A note — hard to miss)
        n_samples   = int(sample_rate * duration)

        # Generate sine wave with fade-in/out envelope to avoid clicking
        t       = np.linspace(0, duration, n_samples, endpoint=False)
        wave_   = np.sin(2 * np.pi * freq * t)
        envelope= np.ones(n_samples)
        fade    = int(sample_rate * 0.05)       # 50ms fade
        envelope[:fade]  = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        samples = (wave_ * envelope * 32767).astype(np.int16)

        # Repeat 3 times with a short silence gap
        silence   = np.zeros(int(sample_rate * 0.2), dtype=np.int16)
        full_wave = np.concatenate([samples, silence, samples, silence, samples])

        beep_path = "alarm_beep.wav"
        with wave.open(beep_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(full_wave.tobytes())

        return beep_path

    # ── public call from main loop ────────────────────────────────────────────

    def notify(self, risk: str):
        """Call every frame with current risk level."""
        if not self._enabled or not self._ready:
            return

        # Stop alarm if risk cleared
        if risk == "CLEAR" and SOUND_CFG["stop_on_clear"]:
            self._stop()
            return

        if risk not in SOUND_CFG["alarm_on"]:
            return

        with self._lock:
            last = self._last_play.get(risk, 0)
            if time.time() - last < SOUND_CFG["cooldown_sec"]:
                return          # still in cooldown
            self._last_play[risk] = time.time()

        # Play in background thread — never blocks video loop
        t = threading.Thread(target=self._play, daemon=True)
        t.start()

    def _play(self):
        try:
            with self._lock:
                if self._playing:
                    pygame.mixer.music.stop()
            pygame.mixer.music.play(loops=SOUND_CFG["loops"] - 1)
            self._playing = True
            print(f"[SOUND] 🔊 Alarm playing...")
        except Exception as e:
            print(f"[SOUND] Play error: {e}")

    def _stop(self):
        try:
            if self._playing and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                self._playing = False
        except Exception:
            pass

    def close(self):
        """Clean shutdown."""
        self._stop()
        try:
            pygame.mixer.quit()
        except Exception:
            pass


class EmailAlerter:
    """
    Sends email alerts when risk level crosses configured thresholds.
    Runs each send in a background thread so it never blocks the video loop.
    Respects a per-level cooldown to avoid inbox flooding.
    """

    def __init__(self):
        self._enabled       = EMAIL_CFG["enabled"]
        self._last_sent     = {}          # risk_level → last sent timestamp
        self._lock          = threading.Lock()

        if self._enabled:
            print(f"[EMAIL] Alerts enabled → {EMAIL_CFG['recipients']}")
            print(f"[EMAIL] Triggers on    : {EMAIL_CFG['alert_on']}")
            print(f"[EMAIL] Cooldown       : {EMAIL_CFG['cooldown_sec']}s")
        else:
            print("[EMAIL] Email alerts disabled (set EMAIL_CFG['enabled']=True to enable)")

    # ── public call from main loop ────────────────────────────────────────────

    def notify(self, risk, fire_conf, smoke_conf,
               fire_zones, smoke_zones, frame_idx, screenshot_path=None):
        """Call every frame. Internally handles cooldown + threading."""

        if not self._enabled:
            return
        if risk not in EMAIL_CFG["alert_on"]:
            return

        with self._lock:
            last = self._last_sent.get(risk, 0)
            if time.time() - last < EMAIL_CFG["cooldown_sec"]:
                return                    # still in cooldown
            self._last_sent[risk] = time.time()

        # Send in background so video loop isn't blocked
        t = threading.Thread(
            target=self._send,
            args=(risk, fire_conf, smoke_conf, fire_zones,
                  smoke_zones, frame_idx, screenshot_path),
            daemon=True,
        )
        t.start()

    # ── internal send ─────────────────────────────────────────────────────────

    def _send(self, risk, fire_conf, smoke_conf,
              fire_zones, smoke_zones, frame_idx, screenshot_path):
        try:
            now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            subject  = f"[{risk}] 🔥 Fire & Smoke Detection Alert — {now_str}"

            # ── HTML email body ───────────────────────────────────────────────
            risk_color = {"CRITICAL": "#ff2d2d", "WARNING": "#ff8c00",
                          "CAUTION":  "#ffd400"}.get(risk, "#ffffff")

            html = f"""
<html><body style="font-family:Arial,sans-serif; background:#0d1117; color:#c9d1d9; padding:20px;">

  <div style="max-width:520px; margin:0 auto; border:1px solid #30363d; border-radius:8px; overflow:hidden;">

    <!-- Header -->
    <div style="background:{risk_color}22; border-bottom:2px solid {risk_color}; padding:18px 24px;">
      <h2 style="margin:0; color:{risk_color}; font-size:20px; letter-spacing:2px;">
        ⚠ {risk} — HAZARD DETECTED
      </h2>
      <p style="margin:4px 0 0; color:#8b949e; font-size:12px;">{now_str}</p>
    </div>

    <!-- Stats -->
    <div style="padding:20px 24px;">
      <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 0; color:#8b949e;">Risk Level</td>
          <td style="padding:10px 0; color:{risk_color}; font-weight:bold;">{risk}</td>
        </tr>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 0; color:#8b949e;">Fire Confidence</td>
          <td style="padding:10px 0; color:#ff2d2d; font-weight:bold;">{fire_conf:.1%}</td>
        </tr>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 0; color:#8b949e;">Smoke Confidence</td>
          <td style="padding:10px 0; color:#a0aec0; font-weight:bold;">{smoke_conf:.1%}</td>
        </tr>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 0; color:#8b949e;">Fire Zones</td>
          <td style="padding:10px 0;">{fire_zones}</td>
        </tr>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 0; color:#8b949e;">Smoke Zones</td>
          <td style="padding:10px 0;">{smoke_zones}</td>
        </tr>
        <tr>
          <td style="padding:10px 0; color:#8b949e;">Frame</td>
          <td style="padding:10px 0;">#{frame_idx:05d}</td>
        </tr>
      </table>
    </div>

    {"<div style='padding:0 24px 16px; color:#8b949e; font-size:12px;'>📎 Screenshot attached.</div>" if screenshot_path else ""}

    <!-- Footer -->
    <div style="background:#161b22; padding:12px 24px; border-top:1px solid #21262d;">
      <p style="margin:0; color:#484f58; font-size:11px;">
        Industrial Fire &amp; Smoke Detection System &nbsp;|&nbsp; Bade Hari Preetham
      </p>
    </div>

  </div>
</body></html>
"""
            # ── build message ─────────────────────────────────────────────────
            msg = MIMEMultipart("alternative")
            msg["From"]    = EMAIL_CFG["sender_email"]
            msg["To"]      = ", ".join(EMAIL_CFG["recipients"])
            msg["Subject"] = subject

            msg.attach(MIMEText(html, "html"))

            # ── attach screenshot ─────────────────────────────────────────────
            if EMAIL_CFG["attach_screenshot"] and screenshot_path and \
               os.path.exists(screenshot_path):
                with open(screenshot_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={os.path.basename(screenshot_path)}"
                    )
                    msg.attach(part)

            # ── send via SMTP ─────────────────────────────────────────────────
            context = ssl.create_default_context()
            with smtplib.SMTP(EMAIL_CFG["smtp_host"],
                              EMAIL_CFG["smtp_port"]) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(EMAIL_CFG["sender_email"],
                             EMAIL_CFG["sender_password"])
                server.sendmail(
                    EMAIL_CFG["sender_email"],
                    EMAIL_CFG["recipients"],
                    msg.as_string(),
                )

            print(f"[EMAIL] ✅ Sent {risk} alert to {EMAIL_CFG['recipients']}")

        except smtplib.SMTPAuthenticationError:
            print("[EMAIL] ❌ Auth failed — check sender_email and sender_password in EMAIL_CFG")
        except smtplib.SMTPException as e:
            print(f"[EMAIL] ❌ SMTP error: {e}")
        except Exception as e:
            print(f"[EMAIL] ❌ Unexpected error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  DETECTION LOGGER
# ──────────────────────────────────────────────────────────────────────────────

class DetectionLogger:
    """
    Writes two CSV files every run:
      1. detection_log_YYYYMMDD_HHMMSS.csv  — every frame (full data)
      2. alert_log_YYYYMMDD_HHMMSS.csv      — only CAUTION/WARNING/CRITICAL events
                                              + saves a screenshot for each alert

    Also keeps in-memory stats and prints a summary on exit.
    """

    # Only log a new alert row if risk level changed OR
    # this many seconds have passed since last alert row (avoids spam)
    ALERT_COOLDOWN_SEC = 3

    FRAME_FIELDS = [
        "timestamp", "date", "time", "frame",
        "fire_conf", "smoke_conf", "risk",
        "fire_zones", "smoke_zones", "scene_objects",
        "fps", "latency_ms", "mode",
    ]

    ALERT_FIELDS = [
        "timestamp", "date", "time", "frame",
        "risk", "fire_conf", "smoke_conf",
        "fire_zones", "smoke_zones", "scene_objects",
        "screenshot",
    ]

    def __init__(self, log_dir="detection_logs"):
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "screenshots"), exist_ok=True)

        self.log_dir    = log_dir
        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # File paths
        self.frame_csv_path = os.path.join(
            log_dir, f"detection_log_{self.session_ts}.csv")
        self.alert_csv_path = os.path.join(
            log_dir, f"alert_log_{self.session_ts}.csv")
        self.screenshot_dir = os.path.join(log_dir, "screenshots")

        # Open CSV writers
        self._frame_f  = open(self.frame_csv_path, "w", newline="")
        self._alert_f  = open(self.alert_csv_path, "w", newline="")

        self._frame_w  = csv.DictWriter(self._frame_f, fieldnames=self.FRAME_FIELDS)
        self._alert_w  = csv.DictWriter(self._alert_f, fieldnames=self.ALERT_FIELDS)

        self._frame_w.writeheader()
        self._alert_w.writeheader()

        # In-memory stats
        self._total_frames     = 0
        self._alert_counts     = {"CAUTION": 0, "WARNING": 0, "CRITICAL": 0}
        self._max_fire_conf    = 0.0
        self._max_smoke_conf   = 0.0
        self._last_alert_time  = 0.0
        self._last_risk        = "CLEAR"

        print(f"[LOG] Detection log  : {self.frame_csv_path}")
        print(f"[LOG] Alert log      : {self.alert_csv_path}")
        print(f"[LOG] Screenshots    : {self.screenshot_dir}/")

    # ── per-frame call ────────────────────────────────────────────────────────

    def log_frame(self, frame_idx, fire_conf, smoke_conf, risk,
                  fire_zones, smoke_zones, scene_objects,
                  fps, latency_ms, mode, canvas=None):

        self._total_frames  += 1
        self._max_fire_conf  = max(self._max_fire_conf,  fire_conf)
        self._max_smoke_conf = max(self._max_smoke_conf, smoke_conf)

        now = datetime.now()
        ts  = now.strftime("%Y-%m-%d %H:%M:%S")

        # ── always write frame row ────────────────────────────────────────────
        self._frame_w.writerow({
            "timestamp":    ts,
            "date":         now.strftime("%Y-%m-%d"),
            "time":         now.strftime("%H:%M:%S"),
            "frame":        frame_idx,
            "fire_conf":    f"{fire_conf:.4f}",
            "smoke_conf":   f"{smoke_conf:.4f}",
            "risk":         risk,
            "fire_zones":   fire_zones,
            "smoke_zones":  smoke_zones,
            "scene_objects":scene_objects,
            "fps":          f"{fps:.1f}",
            "latency_ms":   f"{latency_ms:.1f}",
            "mode":         mode,
        })

        # ── alert row: only on risk change or after cooldown ──────────────────
        if risk in ("CAUTION", "WARNING", "CRITICAL"):
            now_t   = time.time()
            changed = (risk != self._last_risk)
            cooled  = (now_t - self._last_alert_time) >= self.ALERT_COOLDOWN_SEC

            if changed or cooled:
                self._alert_counts[risk] += 1
                self._last_alert_time     = now_t
                self._last_risk           = risk

                # Save screenshot
                shot_name = ""
                if canvas is not None:
                    shot_name = (
                        f"{risk}_{now.strftime('%Y%m%d_%H%M%S')}"
                        f"_f{frame_idx:05d}.jpg"
                    )
                    shot_path = os.path.join(self.screenshot_dir, shot_name)
                    cv2.imwrite(shot_path, canvas)

                self._alert_w.writerow({
                    "timestamp":    ts,
                    "date":         now.strftime("%Y-%m-%d"),
                    "time":         now.strftime("%H:%M:%S"),
                    "frame":        frame_idx,
                    "risk":         risk,
                    "fire_conf":    f"{fire_conf:.2%}",
                    "smoke_conf":   f"{smoke_conf:.2%}",
                    "fire_zones":   fire_zones,
                    "smoke_zones":  smoke_zones,
                    "scene_objects":scene_objects,
                    "screenshot":   shot_name,
                })

                # Flush alert file immediately so it's readable even mid-run
                self._alert_f.flush()

                level_sym = {"CAUTION": "⚠", "WARNING": "🔶", "CRITICAL": "🔴"}
                print(f"\n[ALERT] {level_sym.get(risk,'!')} {risk} — "
                      f"Fire:{fire_conf:.0%}  Smoke:{smoke_conf:.0%}  "
                      f"Frame:{frame_idx:05d}"
                      + (f"  → {shot_name}" if shot_name else ""))

        else:
            self._last_risk = "CLEAR"

    # ── call this on exit ─────────────────────────────────────────────────────

    def close(self):
        self._frame_f.flush()
        self._alert_f.flush()
        self._frame_f.close()
        self._alert_f.close()
        self._print_summary()

    def _print_summary(self):
        total_alerts = sum(self._alert_counts.values())
        print("\n" + "=" * 44)
        print("  SESSION SUMMARY")
        print("=" * 44)
        print(f"  Frames processed : {self._total_frames}")
        print(f"  Max fire conf    : {self._max_fire_conf:.1%}")
        print(f"  Max smoke conf   : {self._max_smoke_conf:.1%}")
        print(f"  Total alerts     : {total_alerts}")
        for level, count in self._alert_counts.items():
            if count:
                print(f"    {level:<10} : {count}")
        print(f"  Log files saved in: {self.log_dir}/")
        print("=" * 44)

# ──────────────────────────────────────────────────────────────────────────────
#  CAMERA WORKER  (runs each camera in its own thread)
# ──────────────────────────────────────────────────────────────────────────────

class CameraWorker:
    """
    Runs one camera feed in a background thread.
    Exposes latest frame, blobs, confidences, and risk level
    via thread-safe properties.
    """

    def __init__(self, source, label, use_yolo_fire=True, yolo_model=None):
        self.label       = label
        self.source      = source
        self._lock       = threading.Lock()
        self._running    = False

        # Per-camera detectors
        self.fire_det  = FireDetector()
        self.smoke_det = SmokeDetector()
        if self.fire_det._use_yolo and yolo_model is not None:
            self.smoke_det.set_yolo_model(yolo_model)
        elif self.fire_det._use_yolo:
            self.smoke_det.set_yolo_model(self.fire_det._model)

        # Shared state (read by main thread)
        self._canvas      = None
        self._fire_conf   = 0.0
        self._smoke_conf  = 0.0
        self._risk        = "CLEAR"
        self._fire_blobs  = []
        self._smoke_blobs = []
        self._fps         = 0.0
        self._frame_idx   = 0

        self._hud = HUDRenderer(CFG["display_w"] // 2, CFG["display_h"])
        self._hud.PANEL_W = 160   # smaller panel for half-width view
        self._fps_ctr = FPSCounter()

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[{self.label}] Camera thread started — source: {self.source}")

    def stop(self):
        self._running = False

    def get_state(self):
        with self._lock:
            return {
                "canvas":      self._canvas,
                "fire_conf":   self._fire_conf,
                "smoke_conf":  self._smoke_conf,
                "risk":        self._risk,
                "fire_blobs":  self._fire_blobs,
                "smoke_blobs": self._smoke_blobs,
                "fps":         self._fps,
                "frame_idx":   self._frame_idx,
            }

    def _loop(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[{self.label}] ❌ Cannot open source: {self.source}")
            return

        W = CFG["display_w"] // 2
        H = CFG["display_h"]

        while self._running:
            ret, raw = cap.read()
            if not ret:
                print(f"[{self.label}] Feed ended or lost.")
                break

            frame = cv2.resize(raw, (W, H))

            fire_blobs,  fire_conf,  _ = self.fire_det.run(frame)
            smoke_blobs, smoke_conf, _ = self.smoke_det.run(frame)
            risk = classify_risk(fire_conf, smoke_conf)

            # Draw HUD on this camera's frame
            canvas = frame.copy()
            self._hud.draw_smoke_blobs(canvas, smoke_blobs, smoke_conf)
            self._hud.draw_fire_blobs(canvas,  fire_blobs,  fire_conf)
            self._hud.draw_scanlines(canvas)
            self._hud.draw_danger_banner(canvas, risk, fire_conf, smoke_conf)
            self._hud.draw_top_bar(canvas, self._fps_ctr.fps, 0,
                                   self._frame_idx, fire_conf, smoke_conf,
                                   risk, self.label)
            self._hud.draw_bottom_bar(canvas, fire_conf, smoke_conf, risk)

            # Camera label overlay
            cv2.putText(canvas, self.label, (8, H - 32),
                        CFG["font_mono"], 0.55, COL["cyan"], 1, cv2.LINE_AA)

            self._fps_ctr.tick()

            with self._lock:
                self._canvas      = canvas.copy()
                self._fire_conf   = fire_conf
                self._smoke_conf  = smoke_conf
                self._risk        = risk
                self._fire_blobs  = fire_blobs
                self._smoke_blobs = smoke_blobs
                self._fps         = self._fps_ctr.fps
                self._frame_idx  += 1

        cap.release()
        print(f"[{self.label}] Camera thread stopped.")
def run_multi_camera(args, det_logger, report_generator,
                     alert_gate, sound_alarmer, email_alerter):
    """Main loop for multi-camera mode."""

    sources = MULTI_CAM_CFG["sources"]
    labels  = MULTI_CAM_CFG["labels"]

    # Start all camera workers
    workers = []
    for src, lbl in zip(sources, labels):
        w = CameraWorker(source=src, label=lbl)
        w.start()
        workers.append(w)

    DW = CFG["display_w"]
    DH = CFG["display_h"]
    frame_idx = 0
    paused    = False

    print(f"[MULTI-CAM] Monitoring {len(workers)} cameras.")
    print(f"[MULTI-CAM] Press Q=quit  P=pause  S=screenshot\n")

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"),
        25.0, (DW, DH)
    )

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("p"):
                paused = not paused
                print(f"\n[MULTI-CAM] {'Paused' if paused else 'Resumed'}")
            if paused:
                cv2.waitKey(30); continue

            frame_idx += 1

            # Collect frames from each worker
            states  = [w.get_state() for w in workers]
            frames  = [s["canvas"] for s in states]

            # If any camera not ready yet, skip
            if any(f is None for f in frames):
                time.sleep(0.03); continue

            # Stitch side by side
            combined = np.hstack(frames)

            # Combined risk = worst of all cameras
            all_risks = [s["risk"] for s in states]
            priority  = ["CRITICAL", "WARNING", "CAUTION", "CLEAR"]
            top_risk  = next((r for r in priority if r in all_risks), "CLEAR")
            top_fire  = max(s["fire_conf"]  for s in states)
            top_smoke = max(s["smoke_conf"] for s in states)

            gated_risk = alert_gate.update(top_risk)

            # Draw divider line between cameras
            mid = DW // 2
            cv2.line(combined, (mid, 0), (mid, DH), COL["cyan"], 1)

            # Alerts
            sound_alarmer.notify(gated_risk)

            shot_name = (f"{top_risk}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                         f"_f{frame_idx:05d}.jpg")
            shot_path = os.path.join("detection_logs", "screenshots", shot_name)

            det_logger.log_frame(
                frame_idx=frame_idx, fire_conf=top_fire,
                smoke_conf=top_smoke, risk=gated_risk,
                fire_zones=sum(len(s["fire_blobs"])  for s in states),
                smoke_zones=sum(len(s["smoke_blobs"]) for s in states),
                scene_objects=0, fps=states[0]["fps"],
                latency_ms=0, mode="CPU", canvas=combined,
            )

            email_alerter.notify(
                risk=gated_risk, fire_conf=top_fire,
                smoke_conf=top_smoke,
                fire_zones=sum(len(s["fire_blobs"])  for s in states),
                smoke_zones=sum(len(s["smoke_blobs"]) for s in states),
                frame_idx=frame_idx,
                screenshot_path=shot_path if os.path.exists(shot_path) else None,
            )

            # Push to dashboard
            _push_jpeg(combined)
            _update_state(
                fire_conf=top_fire, smoke_conf=top_smoke, risk=top_risk,
                fps=states[0]["fps"], latency_ms=0, frame_idx=frame_idx,
                fire_count=sum(len(s["fire_blobs"])  for s in states),
                smoke_count=sum(len(s["smoke_blobs"]) for s in states),
                scene_count=0, mode="MULTI-CAM",
            )

            writer.write(combined)
            cv2.imshow(
                "Industrial Fire & Smoke — Multi-Camera  |  Q=Quit  P=Pause",
                combined
            )

            if frame_idx % 30 == 0:
                print(f"\r[MULTI-CAM] Frame:{frame_idx:05d}"
                      f"  Risk:{top_risk:<8}"
                      f"  Fire:{top_fire:.0%}"
                      f"  Smoke:{top_smoke:.0%}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[MULTI-CAM] Interrupted.")
    finally:
        for w in workers: w.stop()
        writer.release()
        cv2.destroyAllWindows()
        det_logger.close()
        sound_alarmer.close()
        report_generator.generate(
            frame_csv_path=det_logger.frame_csv_path,
            alert_csv_path=det_logger.alert_csv_path,
            screenshot_dir=det_logger.screenshot_dir,
            session_ts=det_logger.session_ts,
        )
def print_startup_banner():
    print("\n" + "=" * 44)
    print("  INDUSTRIAL FIRE & SMOKE DETECTION AI")
    print("=" * 44)
    print(f"  Dev/Creator : Bade Hari Preetham")
    print("=" * 44)
    print("\n[INIT] Loading AI models...")
    print("[INIT] Initializing safety monitoring...")
    print("[INIT] Starting real-time hazard detection...\n")


def open_source(src_arg: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(src_arg) if src_arg.isdigit() else src_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {src_arg}")
    return cap


def build_writer(cap: cv2.VideoCapture, path: str,
                 dw: int, dh: int) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                           fps, (dw, dh))


def main():
    parser = argparse.ArgumentParser(
        description="Industrial Fire & Smoke Detection System -- Bade Hari Preetham"
    )
    parser.add_argument("source", nargs="?", default="0",
                        help="Video file, webcam index, or RTSP URL")
    parser.add_argument("--output", default="output_detected.mp4")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Skip YOLO scene detection (faster)")
    args = parser.parse_args()

    print_startup_banner()

    # Start web dashboard in background thread
    dash_thread = threading.Thread(target=start_dashboard, args=(5000,), daemon=True)
    dash_thread.start()
    print("[DASHBOARD] Web dashboard running at http://localhost:5000\n")

    # Init detection logger
    det_logger    = DetectionLogger(log_dir="detection_logs")
    report_generator = SessionReportGenerator(log_dir="detection_logs")
    alert_gate    = SustainedAlertGate(sustain_sec=3.0)   # ← ADD THIS

    # Init sound alarmer
    sound_alarmer = SoundAlarmer()

    # Init email alerter
    email_alerter = EmailAlerter()
    # ── Multi-camera mode ─────────────────────────────────────────────────────
    if MULTI_CAM_CFG["enabled"]:
        run_multi_camera(args, det_logger, report_generator,
                         alert_gate, sound_alarmer, email_alerter)
        return
    # ── Single camera mode (original) ────────────────────────────────────────
    fire_det  = FireDetector()
    smoke_det = SmokeDetector()
    if fire_det._use_yolo:                              # ← ADD
        smoke_det.set_yolo_model(fire_det._model)       # ← ADD
    scene_det = None if args.no_yolo else SceneDetector()

    DW = CFG["display_w"]
    DH = CFG["display_h"]

    hud = HUDRenderer(DW, DH)
    fps_ctr = FPSCounter()

    cap    = open_source(args.source)
    writer = build_writer(cap, args.output, DW, DH)

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mode_str  = "GPU FP16" if USE_FP16 else ("GPU" if USE_GPU else "CPU")

    print(f"[PIPELINE] Source   : {args.source}")
    print(f"[PIPELINE] FPS      : {src_fps:.1f}")
    print(f"[PIPELINE] Frames   : {total_f if total_f > 0 else 'stream'}")
    print(f"[PIPELINE] Mode     : {mode_str}")
    print(f"[PIPELINE] Output   : {args.output}")
    print(f"[PIPELINE] Controls : Q=quit  P=pause  S=screenshot\n")

    frame_idx  = 0
    paused     = False
    scene_dets = []
    canvas_ref = None          # for screenshot

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                print(f"\n[PIPELINE] {'Paused' if paused else 'Resumed'}")
            if key == ord("s") and canvas_ref is not None:
                sname = f"screenshot_{frame_idx:05d}.jpg"
                cv2.imwrite(sname, canvas_ref)
                print(f"\n[PIPELINE] Screenshot saved: {sname}")

            if paused:
                cv2.waitKey(30)
                continue

            ret, raw = cap.read()
            if not ret:
                break

            frame_idx += 1
            t0 = time.perf_counter()

            frame = cv2.resize(raw, (DW, DH))

            # ── detections ────────────────────────────────────────────────────
            fire_blobs,  fire_conf,  _fm = fire_det.run(frame)
            smoke_blobs, smoke_conf, _sm = smoke_det.run(frame)

            # YOLO scene objects (every 2nd frame for speed)
            if scene_det is not None and frame_idx % 2 == 0:
                scene_dets = scene_det.run(frame)

            # ── scene-aware filter ────────────────────────────────────────────
            # Remove any blob that heavily overlaps a detected person/vehicle.
            # This stops cream walls + moving person from being called smoke.
            if scene_dets:
                smoke_blobs = filter_blobs_by_scene(smoke_blobs, scene_dets)
                # fire_blobs  = filter_blobs_by_scene(fire_blobs,  scene_dets)

                # Recompute confidence from surviving blobs only
                DH2, DW2 = frame.shape[:2]
                frame_area = max(1, DH2 * DW2)

                smoke_area = sum(b["area"] for b in smoke_blobs)
                raw_smoke  = min(1.0, smoke_area / (frame_area * 0.08))
                smoke_conf = smoke_det._conf_smooth.update(raw_smoke) \
                             if smoke_blobs else smoke_det._conf_smooth.update(0.0)

                fire_area  = sum(b["area"] for b in fire_blobs)
                raw_fire   = min(1.0, fire_area  / (frame_area * 0.05))
                fire_conf  = fire_det._conf_smooth.update(raw_fire) \
                             if fire_blobs  else fire_det._conf_smooth.update(0.0)

            risk = classify_risk(fire_conf, smoke_conf)
            gated_risk    = alert_gate.update(risk)   
            # ── compose canvas ────────────────────────────────────────────────
            canvas = frame.copy()

            # Subtle vignette darkening at edges
            vign = np.ones((DH, DW), np.float32)
            cx, cy = DW // 2, DH // 2
            Y, X = np.ogrid[:DH, :DW]
            dist_map = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
            vign = np.clip(1.0 - dist_map * 0.45, 0.55, 1.0)
            for c in range(3):
                canvas[:, :, c] = (canvas[:, :, c] * vign).astype(np.uint8)

            # Draw detections
            hud.draw_smoke_blobs(canvas, smoke_blobs, smoke_conf)
            hud.draw_fire_blobs(canvas, fire_blobs,   fire_conf)
            hud.draw_scene_objects(canvas, scene_dets)

            # Cinematic scanlines
            hud.draw_scanlines(canvas)

            # HUD layers
            fps_ctr.tick()
            lat_ms = (time.perf_counter() - t0) * 1000.0

            hud.draw_danger_banner(canvas, risk, fire_conf, smoke_conf)
            hud.draw_top_bar(canvas, fps_ctr.fps, lat_ms, frame_idx,
                             fire_conf, smoke_conf, risk, mode_str)
            hud.draw_side_panel(canvas, fire_conf, smoke_conf, risk,
                                len(fire_blobs), len(smoke_blobs), len(scene_dets))
            hud.draw_bottom_bar(canvas, fire_conf, smoke_conf, risk)

            canvas_ref = canvas

            # ── push to web dashboard ─────────────────────────────────────────
            _push_jpeg(canvas)
            _update_state(
                fire_conf   = fire_conf,
                smoke_conf  = smoke_conf,
                risk        = risk,
                fps         = fps_ctr.fps,
                latency_ms  = lat_ms,
                frame_idx   = frame_idx,
                fire_count  = len(fire_blobs),
                smoke_count = len(smoke_blobs),
                scene_count = len(scene_dets),
                mode        = mode_str,
            )
            # Log alert events when risk is not CLEAR
            if risk != "CLEAR":
                with _state_lock:
                    log_entry = {
                        "time":  datetime.now().strftime("%H:%M:%S"),
                        "risk":  risk,
                        "fire":  f"{fire_conf:.0%}",
                        "smoke": f"{smoke_conf:.0%}",
                    }
                    _shared["alert_log"].append(log_entry)
                    if len(_shared["alert_log"]) > 200:
                        _shared["alert_log"] = _shared["alert_log"][-200:]

            # ── write to CSV detection log ────────────────────────────────────
            det_logger.log_frame(
                frame_idx    = frame_idx,
                fire_conf    = fire_conf,
                smoke_conf   = smoke_conf,
                risk         = gated_risk,
                fire_zones   = len(fire_blobs),
                smoke_zones  = len(smoke_blobs),
                scene_objects= len(scene_dets),
                fps          = fps_ctr.fps,
                latency_ms   = lat_ms,
                mode         = mode_str,
                canvas       = canvas,       # passed for auto-screenshot on alerts
            )

            # ── email alert ───────────────────────────────────────────────────
            # Build screenshot path (matches what DetectionLogger saves)
            shot_name = (
                f"{risk}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_f{frame_idx:05d}.jpg"
            )
            shot_path = os.path.join(
                "detection_logs", "screenshots", shot_name)
            email_alerter.notify(
                risk           = gated_risk,
                fire_conf      = fire_conf,
                smoke_conf     = smoke_conf,
                fire_zones     = len(fire_blobs),
                smoke_zones    = len(smoke_blobs),
                frame_idx      = frame_idx,
                screenshot_path= shot_path if os.path.exists(shot_path) else None,
            )

            # ── sound alarm ───────────────────────────────────────────────────
            sound_alarmer.notify(gated_risk)
            writer.write(canvas)
            cv2.imshow(
                "Industrial Fire & Smoke Detection AI  |  Q=Quit  P=Pause  S=Screenshot",
                canvas
            )

            if frame_idx % 30 == 0:
                pct = (frame_idx / total_f * 100) if total_f > 0 else 0
                print(f"\r[PIPELINE] Frame {frame_idx:05d}"
                      f"  FPS:{fps_ctr.fps:5.1f}"
                      f"  Fire:{fire_conf:.0%}"
                      f"  Smoke:{smoke_conf:.0%}"
                      f"  Risk:{risk:<8}"
                      f"  {pct:5.1f}%", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[PIPELINE] Interrupted.")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        det_logger.close()
        sound_alarmer.close()
        print(f"\n\n[PIPELINE] Done. Output: {args.output}")
        print(f"[PIPELINE] Frames processed: {frame_idx}")
        report_generator.generate(                          # ← ADD THESE 6 LINES
            frame_csv_path = det_logger.frame_csv_path,
            alert_csv_path = det_logger.alert_csv_path,
            screenshot_dir = det_logger.screenshot_dir,
            session_ts     = det_logger.session_ts,
        )

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()