#!/usr/bin/env python3
"""Score a checkpoint on a labelled split with COCOEvaluator.

    python tools/evaluate.py --weights weights/car_damage_r101_dc5.pth \
                             --dataset-root datasets/car_damage

Reports bbox and segm AP/AP50/AP75/APs/APm/APl, plus per-class AP.

This is the script that fills in the results table in docs/EVALUATION.md. It
needs a val split with COCO annotations; the released checkpoint was trained on
an internal dataset that is not published, so the numbers are not reproducible
from this repository alone - see docs/DATASET.md.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardamage.config import DEFAULT_CONFIG, setup_cfg  # noqa: E402
from datasets.register import register_car_damage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config-file", default=DEFAULT_CONFIG)
    ap.add_argument("--dataset-root", default=os.environ.get("CAR_DAMAGE_ROOT",
                                                             "datasets/car_damage"))
    ap.add_argument("--dataset", default="car_damage_val", help="registered split to score")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default="output/eval")
    ap.add_argument("--score-thresh", type=float, default=0.05,
                    help="keep this low for AP; the 0.7 production default "
                         "truncates the precision-recall curve and depresses AP")
    args = ap.parse_args()

    import torch
    from detectron2.data import build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.utils.logger import setup_logger

    setup_logger()
    registered = register_car_damage(args.dataset_root, strict=False)
    if args.dataset not in registered:
        print(f"dataset {args.dataset!r} is not available under {args.dataset_root!r}; "
              f"registered: {registered or '(none)'}", file=sys.stderr)
        return 1

    cfg = setup_cfg(args.weights, args.config_file, args.score_thresh, args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    evaluator = COCOEvaluator(args.dataset, output_dir=args.output_dir)
    loader = build_detection_test_loader(cfg, args.dataset)
    with torch.no_grad():
        results = inference_on_dataset(model, loader, evaluator)

    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
