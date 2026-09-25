"""Measure the damaged regions a prediction found.

Ported from the production service's ``mathLocations/AreasSegmentation.py``.
Two things were corrected on the way, both noted inline: masks with several
disconnected parts are no longer silently truncated to their first contour, and
the pixel-to-centimetre helper was removed rather than kept wrong.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def mask_contours(outputs) -> Tuple[List[np.ndarray], List[int]]:
    """Extract the outline of every predicted mask.

    :param outputs: a Detectron2 prediction dict (the ``"instances"`` key)
    :returns: ``(contours, class_ids)``. ``contours[i]`` is a list of
        ``(N, 1, 2)`` point arrays - a mask split across several disconnected
        blobs, which happens often with scratches, yields more than one.

    The original production code kept only ``contour[0]``, so the area of any
    damage broken into several pieces was under-reported.
    """
    instances = outputs["instances"].to("cpu")
    class_ids = instances.pred_classes.tolist()

    contours = []
    for pred_mask in instances.pred_masks:
        mask = pred_mask.numpy().astype("uint8")
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.append(list(found))
    return contours, class_ids


def polygon_area(x: Sequence[float], y: Sequence[float]) -> float:
    """Area of a polygon from its vertices, by the shoelace formula."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def split_xy(contour: np.ndarray) -> Tuple[List[int], List[int]]:
    """Split an OpenCV ``(N, 1, 2)`` contour into separate x and y lists."""
    points = np.asarray(contour).reshape(-1, 2)
    return points[:, 0].tolist(), points[:, 1].tolist()


def instance_areas(outputs) -> List[float]:
    """Area in pixels of each predicted instance, summed over its blobs."""
    contours, _ = mask_contours(outputs)
    areas = []
    for blobs in contours:
        areas.append(sum(polygon_area(*split_xy(blob)) for blob in blobs))
    return areas


def area_by_class(outputs, class_names: Sequence[str]) -> Dict[str, float]:
    """Total damaged pixel area per class name.

    The production code left this as an empty stub; it is implemented here
    because it is the number an inspection workflow actually wants.
    """
    contours, class_ids = mask_contours(outputs)
    totals: Dict[str, float] = {}
    for blobs, class_id in zip(contours, class_ids):
        area = sum(polygon_area(*split_xy(blob)) for blob in blobs)
        totals[class_names[class_id]] = totals.get(class_names[class_id], 0.0) + area
    return totals


def area_fraction(outputs, image_shape: Sequence[int]) -> List[float]:
    """Each instance's area as a fraction of the whole image.

    This replaces the old ``PixtoCm()`` helper, which multiplied a *pixel area*
    by 0.0264583333 - the px-to-mm factor for a length at 96 DPI. That is wrong
    twice over: an area conversion needs the factor squared, and the real scale
    depends on camera distance and focal length, which a single photo does not
    carry. A fraction of the frame is the honest scale-free measure; converting
    to real-world area needs a reference object of known size in the shot.
    """
    height, width = image_shape[0], image_shape[1]
    denominator = float(height * width)
    return [area / denominator for area in instance_areas(outputs)]
