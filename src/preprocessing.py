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
}
AUGMENT_VERSIONS = {"D3", "D4"}


def build_transform(version, split, img_size=224):
    """Augmentation only on the train split of D3/D4; val/test always plain."""
    if split == "train" and version in AUGMENT_VERSIONS:
        ops = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
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
