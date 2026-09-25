# Architecture

Everything below is read off the released checkpoint and the two config files in
`configs/`, not from memory of what was intended.

## What the checkpoint contains

```
tensors      546
parameters   191,111,222
iteration    59,999
```

Two tensor shapes fix the head sizes:

| Tensor | Shape | What it means |
|---|---|---|
| `roi_heads.box_predictor.cls_score.weight` | `(8, 1024)` | 7 damage classes + 1 background |
| `roi_heads.mask_head.predictor.weight` | `(7, 256, 1, 1)` | one mask channel per class, no background |

And one absence fixes the backbone:

| Tensor | Present? | |
|---|---|---|
| `backbone.res5.*` | yes | the C5 stage is part of the backbone |
| `backbone.fpn_lateral2` … `fpn_lateral5` | **no** | there is no feature pyramid |

## Dilated-C5, not FPN

`configs/Base-RCNN-DilatedC5.yaml`:

```yaml
RESNETS:
  OUT_FEATURES: ["res5"]
  RES5_DILATION: 2
RPN:
  IN_FEATURES: ["res5"]
ROI_HEADS:
  IN_FEATURES: ["res5"]
```

A DC5 backbone produces **one** feature map at stride 16, rather than the four
maps at strides 4–32 that an FPN gives. The `res5` stage keeps stride 16 instead
of dropping to 32, and recovers the lost receptive field with dilation 2.

The trade-off this makes, and why it suits car damage:

* **Cost.** RPN and both ROI heads run on one map instead of four. The backbone
  is heavier per-pixel because `res5` runs at twice the usual resolution, but the
  head work is a quarter as much.
* **Scale range.** An FPN earns its keep when object sizes span a wide range and
  small objects need a fine map. Damage regions in inspection photos cluster in
  the middle of the scale range — the photographer frames the damage. A single
  stride-16 map covers that band.
* **Mask detail.** Masks are predicted at 28×28 and pasted back regardless of the
  backbone, so the pyramid buys less here than the detection literature suggests.

The cost shows up on very small damage. A hairline scratch a few pixels wide has
little signal left at stride 16, and this is the regime where the model misses
most often.

## Heads

From the same base config:

| Component | Setting |
|---|---|
| ROI box head | `FastRCNNConvFCHead`, `NUM_FC: 2`, pooler 7×7 → 2 × FC-1024 |
| ROI mask head | `MaskRCNNConvUpsampleHead`, `NUM_CONV: 4`, pooler 14×14 → conv-256 ×4 → deconv → 28×28 |
| RPN at test | `PRE_NMS_TOPK_TEST: 6000`, `POST_NMS_TOPK_TEST: 1000` |
| ROI batch during training | 128 proposals per image |

## Inference defaults

From `configs/damage_mask_rcnn_R_101_DC5_3x.yaml`:

| | |
|---|---|
| Input | BGR, shortest edge 800, longest edge 1333 (Detectron2 defaults) |
| Score threshold | 0.7 |
| Classes | 7 |

The 0.7 threshold is a production choice favouring precision. It is the wrong
threshold for computing AP — see [EVALUATION.md](EVALUATION.md) — and it is high
enough to lose real damage, see [LIMITATIONS.md](LIMITATIONS.md).

## Verifying these numbers yourself

```bash
python tools/analyze_model.py \
    --tasks parameter \
    --config-file configs/damage_mask_rcnn_R_101_DC5_3x.yaml \
    MODEL.WEIGHTS weights/car_damage_r101_dc5.pth
```

The parameter count printed should be 191,111,222.
