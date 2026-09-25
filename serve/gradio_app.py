#!/usr/bin/env python3
"""Gradio web demo for the car damage Mask R-CNN.

    pip install -r serve/requirements.txt
    # Detectron2 must already be installed (see README)

    CAR_DAMAGE_WEIGHTS=weights/car_damage_r101_dc5.pth \
        python serve/gradio_app.py

Open http://127.0.0.1:7860 — bind address/port via $HOST / $PORT.
"""
from __future__ import annotations

import os
import socket
import sys

import gradio as gr
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cardamage.config import (  # noqa: E402
    DAMAGE_CLASSES_EN,
    DAMAGE_CLASSES_VI,
    build_metadata,
    build_predictor,
)
from cardamage.geometry import instance_areas  # noqa: E402
from cardamage.visualize import draw_masks_and_boxes  # noqa: E402

WEIGHTS = os.environ.get("CAR_DAMAGE_WEIGHTS", os.path.join(ROOT, "weights", "car_damage_r101_dc5.pth"))
SCORE_THRESH_DEFAULT = float(os.environ.get("SCORE_THRESH", "0.7"))
# Load once at the slider floor so raising the slider only filters, never reloads 765 MB.
SCORE_FLOOR = 0.30
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7862")


def _free_port(preferred: int, host: str = "0.0.0.0", span: int = 20) -> int:
    """Return preferred if free, else the next free TCP port in [preferred, preferred+span)."""
    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host if host != "0.0.0.0" else "", port))
            except OSError:
                continue
            return port
    raise SystemExit(
        f"No free port in {preferred}-{preferred + span - 1}. "
        "Set PORT=... to an unused port."
    )

if not os.path.isfile(WEIGHTS):
    raise SystemExit(
        f"Checkpoint not found: {WEIGHTS}\n"
        "Run:  python scripts/fetch_weights.py\n"
        "Or set CAR_DAMAGE_WEIGHTS to the .pth path."
    )

predictor = build_predictor(WEIGHTS, score_thresh=SCORE_FLOOR)
_metadata = build_metadata("en")


def _examples() -> list:
    demo_dir = os.path.join(ROOT, "assets", "demo")
    names = [
        "01_multi_damage.jpg",
        "02_tear_scratch.jpg",
        "04_crease.jpg",
        "05_missed_at_default.jpg",
    ]
    rows = []
    for name in names:
        path = os.path.join(demo_dir, name)
        if os.path.isfile(path):
            rows.append([path, SCORE_THRESH_DEFAULT])
    return rows


def detect(image: Image.Image, score_thresh: float):
    if image is None:
        return None, [], "No image uploaded."

    image_rgb = np.asarray(image.convert("RGB"))
    image_bgr = image_rgb[:, :, ::-1].copy()
    outputs = predictor(image_bgr)

    instances = outputs["instances"].to("cpu")
    keep = instances.scores >= float(score_thresh)
    instances = instances[keep]
    outputs = {"instances": instances}

    areas = instance_areas(outputs)
    overlay = draw_masks_and_boxes(image_rgb, outputs, metadata=_metadata)

    rows = [
        [
            DAMAGE_CLASSES_EN[int(c)],
            round(float(s), 4),
            int(a),
            f"({b[0]:.0f}, {b[1]:.0f}) → ({b[2]:.0f}, {b[3]:.0f})",
        ]
        for c, s, b, a in zip(
            instances.pred_classes, instances.scores, instances.pred_boxes.tensor, areas
        )
    ]

    if not rows:
        note = (
            f"No damage detected at threshold {score_thresh:.2f}. "
            "Lower the threshold to see lower-confidence regions."
        )
    else:
        note = f"{len(rows)} damage region(s) at threshold {score_thresh:.2f}."
    return overlay, rows, note


LABELS_MD = "\n".join(
    f"| {i} | {vi} | {en} |"
    for i, (vi, en) in enumerate(zip(DAMAGE_CLASSES_VI, DAMAGE_CLASSES_EN))
)

with gr.Blocks(title="Car Damage Detection") as demo:
    gr.Markdown(
        "# Car physical damage detection\n"
        "Mask R-CNN R-101-DC5 — segments 7 damage types from a vehicle photo.\n\n"
        "> Assists human inspection. Do **not** use it to auto-approve insurance "
        "claims without a human adjuster reviewing the result."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Vehicle image")
            thresh = gr.Slider(
                0.30,
                0.95,
                value=SCORE_THRESH_DEFAULT,
                step=0.05,
                label="Score threshold",
                info="Default 0.70 — production threshold",
            )
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            out_img = gr.Image(type="numpy", label="Result")
            note = gr.Markdown()
            out_tbl = gr.Dataframe(
                headers=["Label", "Score", "Area (px)", "Bounding box"],
                label="Details",
                wrap=True,
            )

    gr.Markdown(
        f"### 7 damage classes\n\n| id | Vietnamese | English |\n|---:|---|---|\n{LABELS_MD}"
    )

    btn.click(detect, [inp, thresh], [out_img, out_tbl, note])
    examples = _examples()
    if examples:
        gr.Examples(examples, inputs=[inp, thresh], label="Example images")

if __name__ == "__main__":
    port = _free_port(PORT, HOST)
    if port != PORT:
        print(f"Port {PORT} is busy — using {port} instead.", flush=True)
    demo.launch(server_name=HOST, server_port=port)
