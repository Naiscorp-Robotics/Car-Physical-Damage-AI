"""Minimal HTTP service around the car damage model.

    pip install -r serve/requirements.txt
    CAR_DAMAGE_WEIGHTS=weights/car_damage_r101_dc5.pth \
        uvicorn serve.app:app --host 0.0.0.0 --port 8000

    GET  /health                      model id, device, threshold
    POST /detect                      multipart 'file' -> JSON list of damages
    POST /detect/image                multipart 'file' -> PNG with masks drawn

The production service this was taken from exposed nine endpoints across five
models and accepted base64 JSON only. This is the one endpoint that belongs to
the released model, with file upload added and every internal path removed.
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from PIL import Image  # noqa: E402

from cardamage.config import DEFAULT_CONFIG, build_metadata, setup_cfg  # noqa: E402
from cardamage.geometry import instance_areas  # noqa: E402
from cardamage.imageio import bytes_to_bgr  # noqa: E402
from cardamage.visualize import draw_masks_and_boxes  # noqa: E402

WEIGHTS = os.environ.get("CAR_DAMAGE_WEIGHTS", "weights/car_damage_r101_dc5.pth")
CONFIG_FILE = os.environ.get("CAR_DAMAGE_CONFIG", DEFAULT_CONFIG)
SCORE_THRESH = float(os.environ.get("SCORE_THRESH", "0.7"))
DEVICE = os.environ.get("DEVICE")
# Guard against decompression-bomb style uploads; 40 MP is well past any phone.
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 40_000_000))

app = FastAPI(title="Car Damage Detection", version="0.1.0")

_state: dict = {}


@app.on_event("startup")
def _load_model() -> None:
    from detectron2.engine import DefaultPredictor

    if not os.path.isfile(WEIGHTS):
        raise RuntimeError(
            f"checkpoint not found: {WEIGHTS}. Download it from the GitHub "
            f"release and set $CAR_DAMAGE_WEIGHTS."
        )
    cfg = setup_cfg(WEIGHTS, CONFIG_FILE, SCORE_THRESH, DEVICE)
    _state["cfg"] = cfg
    _state["predictor"] = DefaultPredictor(cfg)
    _state["metadata_vi"] = build_metadata("vi")
    _state["metadata_en"] = build_metadata("en")


def _read(file_bytes: bytes):
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        image = bytes_to_bgr(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"image exceeds {MAX_PIXELS} pixels")
    return image


@app.get("/health")
def health() -> dict:
    cfg = _state.get("cfg")
    return {
        "status": "ok" if cfg is not None else "loading",
        "weights": WEIGHTS,
        "device": cfg.MODEL.DEVICE if cfg else None,
        "score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST if cfg else None,
        "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES if cfg else None,
    }


@app.post("/detect")
def detect(
    file: UploadFile = File(...),
    language: str = Query("vi", pattern="^(vi|en)$"),
) -> dict:
    """Return one entry per damaged region found."""
    image = _read(file.file.read())
    outputs = _state["predictor"](image)
    metadata = _state[f"metadata_{language}"]
    names = metadata.get("thing_classes")

    instances = outputs["instances"].to("cpu")
    areas = instance_areas(outputs)
    height, width = image.shape[:2]

    damages = [
        {
            "label": names[int(c)],
            "class_id": int(c),
            "score": round(float(s), 4),
            "box": [round(float(v), 1) for v in b.tolist()],
            "mask_area_px": round(float(a), 1),
            "mask_area_fraction": round(float(a) / (height * width), 6),
        }
        for c, s, b, a in zip(
            instances.pred_classes, instances.scores, instances.pred_boxes, areas
        )
    ]
    return {"count": len(damages), "image_size": [width, height], "damages": damages}


@app.post("/detect/image")
def detect_image(
    file: UploadFile = File(...),
    language: str = Query("vi", pattern="^(vi|en)$"),
) -> Response:
    """Return the input image with masks, boxes and labels drawn on it."""
    image = _read(file.file.read())
    outputs = _state["predictor"](image)
    overlay = draw_masks_and_boxes(
        image[:, :, ::-1], outputs, metadata=_state[f"metadata_{language}"]
    )
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
