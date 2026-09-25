"""Car damage instance segmentation - Mask R-CNN R-101-DC5.

    from cardamage import build_predictor, DAMAGE_CLASSES_EN

    predictor = build_predictor("weights/car_damage_r101_dc5.pth")
    outputs = predictor(image_bgr)          # standard Detectron2 output dict
"""

from .config import (
    DAMAGE_CLASSES_EN,
    DAMAGE_CLASSES_VI,
    build_metadata,
    build_predictor,
    setup_cfg,
)

__version__ = "0.1.0"
__all__ = [
    "build_predictor",
    "setup_cfg",
    "build_metadata",
    "DAMAGE_CLASSES_VI",
    "DAMAGE_CLASSES_EN",
]
