"""Image encode/decode helpers used by the demo CLI and the serving layer.

Ported from the production service's ``utils/util.py``, keeping only what this
repository actually needs. Three things changed on the way:

* ``np.fromstring`` became ``np.frombuffer`` - the former was removed in NumPy 2;
* the ``type=1|2`` integer switch became two explicitly named functions;
* the 200-line vendored copy of ``PIL.Image.convert`` was dropped, since Pillow
  provides it.
"""
from __future__ import annotations

import base64
import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

_DATA_URI_PREFIX = re.compile(r"^data:image/.+;base64,")


def _strip_data_uri(img_base64: str) -> bytes:
    return base64.b64decode(_DATA_URI_PREFIX.sub("", img_base64))


def base64_to_bgr(img_base64: str) -> np.ndarray:
    """Decode a base64 string (with or without data-URI prefix) to a BGR array.

    BGR is what Detectron2 predictors expect.
    """
    data = np.frombuffer(_strip_data_uri(img_base64), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image from base64 payload")
    return image


def base64_to_pil(img_base64: str) -> Image.Image:
    """Decode a base64 string to a PIL image (RGB)."""
    return Image.open(BytesIO(_strip_data_uri(img_base64))).convert("RGB")


def bytes_to_bgr(raw: bytes) -> np.ndarray:
    """Decode raw image bytes (an uploaded file) to a BGR array."""
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image from bytes")
    return image


def rgb_to_base64(img_rgb: np.ndarray) -> str:
    """Encode an RGB array as a PNG data URI."""
    buffered = BytesIO()
    Image.fromarray(img_rgb.astype("uint8"), "RGB").save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def estimate_blur(image: np.ndarray, threshold: int = 100) -> tuple[float, bool]:
    """Variance of the Laplacian, the cheap blur check used in production.

    :returns: ``(score, is_blurry)`` - lower score means blurrier. The default
        threshold of 100 was tuned on phone photos of cars; it is a heuristic,
        not a calibrated measure.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = float(np.var(cv2.Laplacian(image, cv2.CV_64F)))
    return score, bool(score < threshold)
