# datasets/

Tooling to turn VGG Image Annotator polygons into a COCO instance-segmentation
dataset Detectron2 can train on. The dataset itself is not distributed — see
[../docs/DATASET.md](../docs/DATASET.md) for why, and for the full walkthrough.

| Path | |
|---|---|
| `via2coco/convert.py` | VIA → COCO, plus an 80/20 train/val split. CLI entry point. |
| `via2coco/getArea.py` | Polygon area (shoelace), used for the COCO `area` field. |
| `via2coco/merger.py` | Merge two VIA exports when several people annotated separately. |
| `register.py` | Register the splits with Detectron2's `DatasetCatalog`; also prints per-class counts. |

## Quick use

```bash
# convert
python -m datasets.via2coco.convert \
    --via-json  raw/via_all.json \
    --image-dir raw/images \
    --out-dir   datasets/car_damage

# inspect what came out
python datasets/register.py --root datasets/car_damage
```

## Expected layout

```
datasets/car_damage/
├── train/
│   ├── coco_train.json
│   └── *.jpg
└── val/
    ├── coco_val.json
    └── *.jpg
```

`$CAR_DAMAGE_ROOT` points at the root; `tools/train_net.py` and
`tools/evaluate.py` read it. Everything under `datasets/car_damage/` is
gitignored — image data does not belong in this repository.
