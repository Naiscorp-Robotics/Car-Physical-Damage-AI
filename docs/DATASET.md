# Dataset

## The training data is not published

The released checkpoint was trained on an internal dataset of vehicle inspection
photographs. Those images are customer property and contain licence plates,
faces and other identifying detail, so neither the images nor the annotations
are distributed with this repository.

What *is* published is the whole pipeline that turned raw annotations into
something Detectron2 can train on — enough to reproduce the process on your own
photos.

## Annotation

Damage regions were drawn as polygons in the
[VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) (VIA), one
polygon per damaged region, each tagged with a `class` attribute holding one of
the seven damage names.

A VIA export looks like this — a dict keyed by `filename+filesize`:

```json
{
  "photo_001.jpg123456": {
    "name": "photo_001.jpg",
    "regions": [
      {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [412, 455, 470, 430],
          "all_points_y": [230, 225, 268, 275]
        },
        "class": "Trầy sơn"
      }
    ]
  }
}
```

The converter reads `region["class"]` for the label and
`region["shape_attributes"]` for the polygon. Only polygon regions carry a
usable segmentation mask — rectangles and circles produce degenerate masks.

## Conversion

```bash
python -m datasets.via2coco.convert \
    --via-json  raw/via_all.json \
    --image-dir raw/images \
    --out-dir   datasets/car_damage \
    [--move]
```

The script does three things:

1. **Splits** the annotated images 80/20 into train and val, copying (or, with
   `--move`, moving) the image files into the split directories.
2. **Derives the category list** from every `class` value seen across the whole
   annotation file, so both splits share one consistent category numbering.
3. **Converts** each split to COCO instance-segmentation JSON — `images`,
   `annotations` with polygon `segmentation`, `bbox` and `area`, and
   `categories`.

Polygon areas come from `datasets/via2coco/getArea.py` (shoelace formula).
`datasets/via2coco/merger.py` combines two VIA exports if several people
annotated separately.

> COCO category ids start at 1; Detectron2 remaps them to contiguous 0-based
> class ids internally. The ids in `configs/labels.json` are the **model's** ids
> (0–6), which is what predictions carry.

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

Point `$CAR_DAMAGE_ROOT` at `datasets/car_damage`, or pass `--root`. Both
`tools/train_net.py` and `tools/evaluate.py` register the splits from there.

## Checking what you built

```bash
python datasets/register.py --root datasets/car_damage
```

prints per-split image, instance and class counts, and per-class instance
counts. Two things to look at before spending GPU hours:

* **Category count must be 7.** A typo in one `class` value creates an eighth
  category, and training will silently learn it.
* **Class balance.** Damage datasets skew hard — scratches and dents dominate,
  punctures and broken lamps are rare. Check that the rare classes have enough
  instances in *both* splits, or val AP for those classes will be noise.

You can also look at the registered data directly:

```bash
python tools/visualize_data.py \
    --config-file configs/damage_mask_rcnn_R_101_DC5_3x.yaml \
    --source annotation --output-dir output/vis
```

This renders the annotations as Detectron2 sees them, which catches
polygon-order and coordinate-space mistakes that no count will show.
