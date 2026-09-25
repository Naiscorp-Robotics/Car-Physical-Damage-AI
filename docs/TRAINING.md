# Training

## What is known about the released run

The schedule comes from `configs/`, and the stopping point from the checkpoint
itself. **No training logs, loss curves or `metrics.json` survive** — there is no
`output/` directory anywhere in the source project. What follows is the recipe,
not a reconstruction of the run.

| | |
|---|---|
| Framework | Detectron2 v0.6 |
| Backbone init | `detectron2://ImageNetPretrained/MSRA/R-101.pkl` |
| Images per batch | 16 |
| Base LR | 0.02 |
| LR schedule | `WarmupMultiStepLR`, ×0.1 at 210,000 and 250,000 |
| Scheduled iterations | 270,000 |
| **Iterations reached** | **59,999** |
| ROI proposals per image | 128 |
| Train-time short side | randomly one of 640, 672, 704, 736, 768, 800 |

At batch 16, 59,999 iterations is roughly 960,000 images seen — a large number of
epochs over a small dataset, but only 22% of the planned LR schedule. The
learning rate never reached either decay step, so the released weights come from
the flat, high-LR part of the curve. A checkpoint taken before the first LR drop
is normally still improving.

A second checkpoint at iteration 74,999 exists in the source project. It was
never evaluated and never deployed, so it is not released; picking between them
requires a labelled split, which is the same thing blocking
[EVALUATION.md](EVALUATION.md).

## Reproducing it

### 1. Build the dataset

Convert VIA annotations to COCO and split them:

```bash
python -m datasets.via2coco.convert \
    --via-json  raw/via_all.json \
    --image-dir raw/images \
    --out-dir   datasets/car_damage
```

This writes `datasets/car_damage/{train,val}/coco_{train,val}.json` alongside the
images. Check what came out:

```bash
python datasets/register.py --root datasets/car_damage
```

See [DATASET.md](DATASET.md) for the expected format.

### 2. Train

```bash
export CAR_DAMAGE_ROOT=datasets/car_damage

python tools/train_net.py \
    --config-file configs/damage_mask_rcnn_R_101_DC5_3x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR output/damage_r101_dc5
```

`tools/train_net.py` calls `register_car_damage()` during setup, so the dataset
names in the config (`car_damage_train`, `car_damage_val`) resolve.

Batch 16 across 8 GPUs is 2 images per GPU. On fewer GPUs, scale both the batch
size and the learning rate by the same factor — the linear scaling rule:

```bash
python tools/train_net.py \
    --config-file configs/damage_mask_rcnn_R_101_DC5_3x.yaml \
    --num-gpus 2 \
    SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005 \
    OUTPUT_DIR output/damage_r101_dc5
```

Resume an interrupted run with `--resume`; it picks up `last_checkpoint` in
`OUTPUT_DIR`.

### 3. Watch it

```bash
tensorboard --logdir output/damage_r101_dc5
```

Detectron2 writes `metrics.json` and TensorBoard events into `OUTPUT_DIR`.
**Keep them.** Their absence for the released run is why this document cannot
tell you what the loss actually did.

### 4. Publish a checkpoint

Training checkpoints carry optimizer state that inference never touches, and it
is about half the file:

```bash
python tools/strip_optimizer.py \
    output/damage_r101_dc5/model_0059999.pth \
    weights/car_damage_r101_dc5.pth
# 1527 MB -> 765 MB
```

## An alternative entry point

`tools/plain_train_net.py` is the same training run written as an explicit loop
rather than through `DefaultTrainer`. It is slower to configure and easier to
read; use it when you want to see where the hooks, the LR schedule and the
evaluation actually fire.

## If you are training on your own data

The class list lives in `configs/labels.json` and the count in
`configs/damage_mask_rcnn_R_101_DC5_3x.yaml` (`ROI_HEADS.NUM_CLASSES`). Change
both together — a mismatch produces a checkpoint whose `cls_score` shape does not
match the labels, and nothing will warn you at inference time.
