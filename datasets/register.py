"""Register the car damage dataset with Detectron2 before training or evaluating.

Ported from the production service's ``utils/registerData.py``. That file could
not run on Linux at all - its first import was ``from msilib.schema import
Error``, a Windows-only standard library module - and the dataset paths were
hardcoded relative to the current working directory.

Expected layout (see datasets/README.md):

    datasets/car_damage/
      train/
        coco_train.json
        *.jpg
      val/
        coco_val.json
        *.jpg

Use it from a training script:

    from datasets.register import register_car_damage
    register_car_damage("datasets/car_damage")

or point ``$CAR_DAMAGE_ROOT`` at the directory and let ``tools/train_net.py``
call it for you.
"""
from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_ROOT = os.environ.get("CAR_DAMAGE_ROOT", "datasets/car_damage")

SPLITS = {
    "car_damage_train": ("train", "coco_train.json"),
    "car_damage_val": ("val", "coco_val.json"),
}


def register_car_damage(root: str = DEFAULT_ROOT, strict: bool = True) -> list:
    """Register the train and val splits. Returns the names actually registered.

    :param root: dataset root holding ``train/`` and ``val/``
    :param strict: raise if a split is missing; set False to register whichever
        splits are present (useful when you only have a test set)
    """
    from detectron2.data.datasets import register_coco_instances

    registered = []
    for name, (subdir, ann_name) in SPLITS.items():
        image_dir = os.path.join(root, subdir)
        ann_file = os.path.join(image_dir, ann_name)
        if not os.path.isfile(ann_file):
            if strict:
                raise FileNotFoundError(
                    f"{ann_file} not found. Generate it with "
                    f"`python -m datasets.via2coco.convert --help`."
                )
            continue
        register_coco_instances(name, {}, ann_file, image_dir)
        registered.append(name)
    return registered


def summarise(root: str = DEFAULT_ROOT) -> None:
    """Print image/instance/class counts per split - a quick sanity check."""
    for name, (subdir, ann_name) in SPLITS.items():
        ann_file = os.path.join(root, subdir, ann_name)
        if not os.path.isfile(ann_file):
            print(f"{name:18s} missing ({ann_file})")
            continue
        with open(ann_file, encoding="utf-8") as f:
            coco = json.load(f)
        per_class: dict = {}
        names = {c["id"]: c["name"] for c in coco["categories"]}
        for ann in coco["annotations"]:
            key = names.get(ann["category_id"], ann["category_id"])
            per_class[key] = per_class.get(key, 0) + 1
        print(f"{name:18s} {len(coco['images']):5d} images  "
              f"{len(coco['annotations']):5d} instances  "
              f"{len(coco['categories'])} classes")
        for key, count in sorted(per_class.items(), key=lambda kv: -kv[1]):
            print(f"    {key:20s} {count:5d}")


def main(argv: Optional[list] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Register and inspect the car damage dataset.")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="dataset root directory")
    args = ap.parse_args(argv)
    summarise(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
