"""Quick dataset analysis: image sizes, face detection rate per class (50-image sample)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2

CLASSES = ["heart", "oblong", "oval", "round", "square"]
SAMPLE = 50
DATA_ROOT = Path("data/raw")

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    return len(faces) > 0


def analyze_class(cls):
    imgs = sorted((DATA_ROOT / cls).glob("*.jpg"))
    sample = random.sample(imgs, min(SAMPLE, len(imgs)))

    detected = 0
    widths, heights = [], []
    unreadable = 0

    for p in sample:
        bgr = cv2.imread(str(p))
        if bgr is None:
            unreadable += 1
            continue
        h, w = bgr.shape[:2]
        widths.append(w)
        heights.append(h)
        if detect_face(bgr):
            detected += 1

    readable = len(sample) - unreadable
    det_rate = detected / readable * 100 if readable else 0
    avg_w = sum(widths) / len(widths) if widths else 0
    avg_h = sum(heights) / len(heights) if heights else 0

    print(f"\n[{cls}]  total={len(imgs)}  sampled={len(sample)}  unreadable={unreadable}")
    print(f"  face detected: {detected}/{readable} ({det_rate:.1f}%)")
    print(f"  avg size: {avg_w:.0f}x{avg_h:.0f}px")
    return det_rate


random.seed(42)
print("Analyzing dataset (50-image sample per class)...\n")

rates = {}
for cls in CLASSES:
    rates[cls] = analyze_class(cls)

overall = sum(rates.values()) / len(rates)
print(f"\n--- Overall face detection rate: {overall:.1f}% ---")
if overall < 80:
    print("WARNING: Low detection rate. D2/D3/D4 will fall back to centre-crop for many images.")
elif overall < 90:
    print("OK: Moderate detection rate. Check which classes are weakest.")
else:
    print("GOOD: High detection rate. Preprocessing should work well.")
