"""timm model factory for the four architectures in the comparison."""
from __future__ import annotations

import timm

# Comparison key -> timm model id. All ImageNet-pretrained; timm replaces the
# classifier head automatically for num_classes.
TIMM_IDS = {
    "resnet50": "resnet50",
    "efficientnetv2s": "tf_efficientnetv2_s",
    "mobilenetv3small": "mobilenetv3_small_100",   # app deployment target
    "swint": "swin_tiny_patch4_window7_224",
}


def build_model(name, num_classes=5, pretrained=True):
    if name not in TIMM_IDS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(TIMM_IDS)}")
    return timm.create_model(TIMM_IDS[name], pretrained=pretrained, num_classes=num_classes)


def set_backbone_trainable(model, trainable):
    """Freeze/unfreeze everything except the classifier head."""
    head_param_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        if id(p) not in head_param_ids:
            p.requires_grad = trainable
    return model
