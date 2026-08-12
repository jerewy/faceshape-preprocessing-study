"""Preprocessing version (D1-D4) -> torchvision transform + data source.

The face-crop / alignment heavy lifting is precomputed once by
``src/preprocess_faces.py`` into ``data/preprocessed/{crop,align}/``. Here we
only choose which source folder to read and which transform to apply.
"""
from __future__ import annotations

from pathlib import Path

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Which precomputed source each version reads from. "raw" -> original images.
SOURCE_BY_VERSION = {
    "D1": "resize",  # resize only — reads a precomputed downscale of raw (fast loading)
    "D2": "crop",    # face crop + resize
    "D3": "crop",    # face crop + augmentation
    "D4": "align",   # face crop + alignment + augmentation
    # Alignment ablation. Same image sources as D3/D4, but the augmentation drops
    # the +/-15 deg rotation that otherwise re-introduces the very tilt alignment
    # removes. D4n vs D3n therefore isolates in-plane alignment, which D4 vs D3
    # cannot (the rotation/alignment confound reported as a paper limitation).
    "D3n": "crop",   # face crop + augmentation without rotation
    "D4n": "align",  # face crop + alignment + augmentation without rotation
}
AUGMENT_VERSIONS = {"D3", "D4", "D3n", "D4n"}
# Augmented versions whose regime omits RandomRotation.
NO_ROTATION_VERSIONS = {"D3n", "D4n"}


def build_transform(version, split, img_size=224):
    """Augmentation only on the train split of D3/D4(n); val/test always plain."""
    if split == "train" and version in AUGMENT_VERSIONS:
        # Op order is kept identical to the original D3/D4 regime so that those
        # runs consume the RNG stream exactly as before and stay reproducible.
        ops = [transforms.RandomHorizontalFlip(p=0.5)]
        if version not in NO_ROTATION_VERSIONS:
            ops.append(transforms.RandomRotation(degrees=15))
        ops += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        ]
    else:
        ops = [transforms.Resize((img_size, img_size))]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def source_root_for(version, raw_root, preprocessed_root):
    """Return the folder to read images from, or ``None`` to use raw split paths."""
    src = SOURCE_BY_VERSION[version]
    if src == "raw":
        return None
    return Path(preprocessed_root) / src
