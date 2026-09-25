# Evaluation

## There are no published numbers

This repository ships no AP table, because it ships no labelled test split. The
dataset is internal ([DATASET.md](DATASET.md)) and the source project retained
no evaluation output — no `metrics.json`, no `output/` directory, no
`coco_instances_results.json`.

Quoting numbers that nobody can recompute from this repository would be worse
than quoting none. The protocol below is what `tools/evaluate.py` runs; fill the
table in when you have a split.

## Running it

```bash
python tools/evaluate.py \
    --weights weights/car_damage_r101_dc5.pth \
    --dataset-root datasets/car_damage \
    --dataset car_damage_val
```

This builds the model, loads the checkpoint and runs Detectron2's
`COCOEvaluator`, reporting both `bbox` and `segm` metrics:

| Metric | Meaning |
|---|---|
| `AP` | mean AP over IoU 0.50:0.95 — the headline number |
| `AP50`, `AP75` | AP at a single IoU threshold |
| `APs`, `APm`, `APl` | AP split by object area (<32², 32²–96², >96² px) |
| per-class `AP` | printed as a separate table |

For instance segmentation, **`segm` is the number that matters**. `bbox` tells
you whether the detector found the region; `segm` tells you whether the mask
actually traced the damage.

## Use a low score threshold

`tools/evaluate.py` defaults to `--score-thresh 0.05`, not the 0.7 used in
production. This is not a mistake.

AP integrates precision over the full recall range. A high threshold discards
low-confidence detections before the evaluator ever sees them, truncating the
precision–recall curve and depressing AP for reasons that have nothing to do
with model quality. Score once at 0.05 to get a comparable AP, then pick your
operating threshold separately from the precision/recall trade-off you want.

## Results table

Fill this in from the evaluator output:

| Split | Images | Instances | `segm` AP | AP50 | AP75 | `bbox` AP |
|---|---:|---:|---:|---:|---:|---:|
| `car_damage_val` | — | — | — | — | — | — |

And per class:

| Class | Instances | `segm` AP |
|---|---:|---:|
| Móp lõm / Dent | — | — |
| Trầy sơn / Paint scratch | — | — |
| Rách / Tear | — | — |
| Mất bộ phận / Missing part | — | — |
| Thủng / Puncture | — | — |
| Bể đèn / Broken lamp | — | — |
| Vỡ kính / Broken glass | — | — |

Report the two checkpoints separately if you evaluate both — the released
iteration 59,999 and the unevaluated iteration 74,999
([TRAINING.md](TRAINING.md)).

## Reading the mistakes, not just the score

`COCOEvaluator` writes `coco_instances_results.json` into `--output-dir`. Render
it against the ground truth:

```bash
python tools/visualize_json_results.py \
    --input output/eval/coco_instances_results.json \
    --output output/eval/vis \
    --dataset car_damage_val \
    --conf-threshold 0.5
```

For a damage model the interesting failures are structural, and AP hides them:

* one long scratch predicted as three separate instances, or vice versa;
* the right region found under the wrong class — dent vs. crease vs. scratch is
  a genuinely blurry boundary that annotators disagree on too;
* a correct box with a mask that covers the whole panel instead of the damage.

## Latency

Measure it on your own hardware rather than trusting someone else's number:

```bash
python tools/benchmark.py \
    --config-file configs/damage_mask_rcnn_R_101_DC5_3x.yaml \
    --task eval \
    MODEL.WEIGHTS weights/car_damage_r101_dc5.pth
```

`--task eval` times the full inference path; `--task data` isolates the loader,
which is worth checking before concluding the model is slow.
