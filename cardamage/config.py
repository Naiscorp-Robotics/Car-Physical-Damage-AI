"""Build a Detectron2 predictor for the car damage model.

Ported from the production service, reduced to the single model this repository
releases. Every knob that used to be hardcoded in the service - weight path,
score threshold, device - is a parameter here, and the architecture itself comes
from ``configs/damage_mask_rcnn_R_101_DC5_3x.yaml`` rather than from Python.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, "damage_mask_rcnn_R_101_DC5_3x.yaml")
LABELS_JSON = os.path.join(CONFIG_DIR, "labels.json")


def _load_labels() -> tuple[List[str], List[str]]:
    with open(LABELS_JSON, encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    classes.sort(key=lambda c: c["id"])
    return [c["vi"] for c in classes], [c["en"] for c in classes]


DAMAGE_CLASSES_VI, DAMAGE_CLASSES_EN = _load_labels()


def setup_cfg(
    weights: str,
    config_file: str = DEFAULT_CONFIG,
    score_thresh: Optional[float] = None,
    device: Optional[str] = None,
):
    """Return a frozen Detectron2 ``CfgNode`` for the damage model.

    :param weights: path to the checkpoint (see the GitHub release assets)
    :param config_file: architecture config; defaults to the released one
    :param score_thresh: overrides ``MODEL.ROI_HEADS.SCORE_THRESH_TEST``
    :param device: ``"cuda"``, ``"cuda:0"``, ``"cpu"``; defaults to ``$DEVICE``
                   and falls back to CUDA when available
    """
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights

    if score_thresh is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    if device is None:
        device = os.environ.get("DEVICE")
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    cfg.freeze()
    return cfg


def build_metadata(language: str = "vi"):
    """Metadata carrying the damage class names, for the Detectron2 visualizer."""
    from detectron2.data.catalog import Metadata

    if language not in ("vi", "en"):
        raise ValueError(f"language must be 'vi' or 'en', got {language!r}")
    meta = Metadata()
    meta.set(thing_classes=DAMAGE_CLASSES_VI if language == "vi" else DAMAGE_CLASSES_EN)
    return meta


def build_predictor(
    weights: str,
    config_file: str = DEFAULT_CONFIG,
    score_thresh: Optional[float] = None,
    device: Optional[str] = None,
):
    """Convenience wrapper returning a ready ``DefaultPredictor``.

    The predictor takes a BGR ``numpy`` array (OpenCV order) and returns the
    standard Detectron2 dict with an ``"instances"`` key.
    """
    from detectron2.engine import DefaultPredictor

    return DefaultPredictor(setup_cfg(weights, config_file, score_thresh, device))
