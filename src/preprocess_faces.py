"""Precompute face-cropped and face-aligned dataset variants (run once each).

    python -m src.preprocess_faces --mode crop  --data-root data/raw --out data/preprocessed/crop
    python -m src.preprocess_faces --mode align --data-root data/raw --out data/preprocessed/align

Uses MediaPipe FaceMesh. Output mirrors the raw class/filename structure so the
split csv can remap onto it. Images with no detected face fall back to a centre
square crop and are logged to ``<out>/skipped.csv`` so the count is auditable.

The crop margin intentionally keeps the jawline and forehead/hairline — the very
contour that defines face shape — rather than tightly cropping to the face.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["heart", "oblong", "oval", "round", "square"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# FaceMesh landmark indices (outer eye corners + nose tip) used for alignment.
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263
NOSE_TIP = 1


def _make_facemesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5,
    )


def _landmarks(face_mesh, img_rgb):
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = img_rgb.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def _bbox(pts, w, h, margin=0.25):
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - margin * bw))
    y0 = max(0, int(y0 - margin * bh))
    x1 = min(w, int(x1 + margin * bw))
    y1 = min(h, int(y1 + margin * bh))
    return x0, y0, x1, y1


def _centre_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def _align_and_points(img_rgb, pts):
    """Rotate so the eyes are horizontal; return (aligned_img, transformed_pts)."""
    right_eye, left_eye = pts[RIGHT_EYE_OUTER], pts[LEFT_EYE_OUTER]
    angle = np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
    h, w = img_rgb.shape[:2]
    centre = (float(pts[NOSE_TIP][0]), float(pts[NOSE_TIP][1]))
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    aligned = cv2.warpAffine(img_rgb, M, (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    pts_aligned = (M @ pts_h.T).T
    return aligned, pts_aligned


def process_image(face_mesh, src_path, mode):
    """Return an RGB ndarray for the requested mode, or None on read failure."""
    bgr = cv2.imread(str(src_path))
    if bgr is None:
        return None, "unreadable"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pts = _landmarks(face_mesh, rgb)
    if pts is None:
        return _centre_square(rgb), "no_face_centre_crop"
    if mode == "align":
        rgb, pts = _align_and_points(rgb, pts)
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = _bbox(pts, w, h)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return _centre_square(rgb), "empty_bbox_centre_crop"
    return crop, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["crop", "align"])
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_root, out_root = Path(args.data_root), Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    face_mesh = _make_facemesh()
    skipped = []
    total = 0

    for label in CLASSES:
        cls_dir = data_root / label
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        out_cls = out_root / label
        out_cls.mkdir(parents=True, exist_ok=True)
        for src in sorted(cls_dir.rglob("*")):
            if src.suffix.lower() not in IMG_EXTS:
                continue
            total += 1
            img_rgb, status = process_image(face_mesh, src, args.mode)
            if img_rgb is None:
                skipped.append((str(src), status))
                continue
            if status != "ok":
                skipped.append((str(src), status))
            out_path = out_cls / src.name          # keep original filename for remap
            cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"[{args.mode}] {label}: done")

    face_mesh.close()
    if skipped:
        with (out_root / "skipped.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "status"])
            w.writerows(skipped)
    print(f"[{args.mode}] processed {total} images, {len(skipped)} fell back / failed.")


if __name__ == "__main__":
    main()
