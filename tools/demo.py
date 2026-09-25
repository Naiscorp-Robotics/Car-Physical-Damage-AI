#!/usr/bin/env python3
"""Run the car damage model on images, a folder, a video or a webcam.

    python tools/demo.py --weights weights/car_damage_r101_dc5.pth \
                         --input photo.jpg --output out.jpg

    python tools/demo.py --weights weights/car_damage_r101_dc5.pth \
                         --input 'photos/*.jpg' --output results/

    python tools/demo.py --weights weights/car_damage_r101_dc5.pth \
                         --video clip.mp4 --output clip_damage.mp4

    python tools/demo.py --weights weights/car_damage_r101_dc5.pth --webcam

Adapted from the Detectron2 demo script that shipped with the production
service. The architecture now comes from ``configs/`` instead of being patched
onto a COCO config in Python, and ``--json`` was added so results can be
inspected without looking at pictures.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import tqdm  # noqa: E402

from cardamage.config import DEFAULT_CONFIG, build_metadata, setup_cfg  # noqa: E402
from cardamage.predictor import VisualizationDemo  # noqa: E402

WINDOW_NAME = "car damage"


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True, help="path to the checkpoint")
    ap.add_argument("--config-file", default=DEFAULT_CONFIG, help="architecture config")
    ap.add_argument("--input", nargs="+", help="image file(s) or a glob pattern")
    ap.add_argument("--video", help="path to a video file")
    ap.add_argument("--webcam", action="store_true", help="read from the default camera")
    ap.add_argument("--output", help="output file, or a directory for multiple inputs")
    ap.add_argument("--json", help="write the structured results to this JSON file")
    ap.add_argument("--threshold", type=float, default=None,
                    help="score threshold (default: the value in the config, 0.7)")
    ap.add_argument("--device", default=None, help="cuda, cuda:0 or cpu")
    ap.add_argument("--language", default="vi", choices=["vi", "en"], help="label language")
    return ap


def summarise(predictions, metadata) -> list:
    instances = predictions["instances"].to("cpu")
    names = metadata.get("thing_classes")
    return [
        {
            "label": names[int(c)],
            "class_id": int(c),
            "score": round(float(s), 4),
            "box": [round(float(v), 1) for v in b.tolist()],
        }
        for c, s, b in zip(instances.pred_classes, instances.scores, instances.pred_boxes)
    ]


def main() -> int:
    args = get_parser().parse_args()
    if not (args.input or args.video or args.webcam):
        get_parser().error("one of --input, --video or --webcam is required")

    from detectron2.data.detection_utils import read_image
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    cfg = setup_cfg(args.weights, args.config_file, args.threshold, args.device)
    metadata = build_metadata(args.language)
    demo = VisualizationDemo(cfg, metadata=metadata)
    logger.info(f"device={cfg.MODEL.DEVICE} threshold={cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")

    results = {}

    if args.input:
        paths = args.input
        if len(paths) == 1 and any(ch in paths[0] for ch in "*?["):
            paths = sorted(glob.glob(os.path.expanduser(paths[0])))
        if not paths:
            logger.error("no input images matched")
            return 1
        if len(paths) > 1 and args.output and not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

        for path in tqdm.tqdm(paths, disable=len(paths) == 1):
            image = read_image(path, format="BGR")
            start = time.time()
            predictions, visualised = demo.run_on_image(image)
            found = summarise(predictions, metadata)
            results[os.path.basename(path)] = found
            logger.info(f"{path}: {len(found)} damage(s) in {time.time() - start:.2f}s")

            if args.output:
                out_path = (os.path.join(args.output, os.path.basename(path))
                            if os.path.isdir(args.output) else args.output)
                visualised.save(out_path)
            elif not args.json:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualised.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break

    elif args.webcam:
        cam = cv2.VideoCapture(0)
        for frame in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    else:
        video = cv2.VideoCapture(args.video)
        if not video.isOpened():
            logger.error(f"cannot open {args.video}")
            return 1
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        writer = None
        if args.output:
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height), True)
        for frame in tqdm.tqdm(demo.run_on_video(video), total=total):
            if writer is not None:
                writer.write(frame)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) == 27:
                    break
        video.release()
        if writer is not None:
            writer.release()
        else:
            cv2.destroyAllWindows()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
