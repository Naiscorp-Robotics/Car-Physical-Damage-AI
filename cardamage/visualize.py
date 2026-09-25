"""Draw predictions onto an image.

Ported from the production service's ``visualDamage.py``. The car-part variant
was dropped (that model is not released here), the three remaining styles were
given English names, and ``draw_masks`` no longer double-draws: the original
called ``draw_instance_predictions`` and then ``overlay_instances`` on the same
canvas, painting every mask twice at different alphas.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import build_metadata


def _visualizer(image: np.ndarray, metadata, scale: float):
    from detectron2.utils.visualizer import ColorMode, Visualizer

    if metadata is None:
        metadata = build_metadata()
    return Visualizer(image, metadata=metadata, scale=scale, instance_mode=ColorMode.SEGMENTATION)


def draw_masks_and_boxes(
    image: np.ndarray,
    outputs,
    metadata=None,
    scale: float = 1.0,
    as_array: bool = True,
):
    """Masks, boxes, class names and scores - the production default.

    :param image: RGB array, as the Detectron2 visualizer expects
    :param outputs: prediction dict with an ``"instances"`` key
    :param metadata: class-name metadata; defaults to the Vietnamese labels
    :param as_array: return an RGB ``ndarray`` (default) instead of a ``VisImage``
    """
    vis = _visualizer(image, metadata, scale)
    out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image() if as_array else out


def draw_masks(
    image: np.ndarray,
    outputs,
    metadata=None,
    scale: float = 1.0,
    alpha: float = 0.3,
    as_array: bool = True,
):
    """Translucent masks only - no boxes, no labels. Reads best on dense damage."""
    vis = _visualizer(image, metadata, scale)
    instances = outputs["instances"].to("cpu")
    out = vis.overlay_instances(masks=instances.pred_masks, alpha=alpha)
    return out.get_image() if as_array else out


def draw_boxes(
    image: np.ndarray,
    outputs,
    metadata=None,
    scale: float = 1.0,
    as_array: bool = True,
):
    """Bounding boxes with labels, no mask fill."""
    vis = _visualizer(image, metadata, scale)
    instances = outputs["instances"].to("cpu")
    out = vis.overlay_instances(
        boxes=instances.pred_boxes,
        labels=_labels(instances, vis.metadata),
    )
    return out.get_image() if as_array else out


def _labels(instances, metadata) -> Optional[list]:
    if not instances.has("pred_classes"):
        return None
    names = metadata.get("thing_classes")
    classes = instances.pred_classes.tolist()
    if not instances.has("scores"):
        return [names[i] for i in classes]
    return [f"{names[i]} {s:.0%}" for i, s in zip(classes, instances.scores.tolist())]
