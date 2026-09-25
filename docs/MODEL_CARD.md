# Model card — Car Damage Mask R-CNN (R-101-DC5)

## Overview

| | |
|---|---|
| Task | Instance segmentation |
| Domain | Physical damage on cars, from inspection photographs |
| Classes | 7 (see below) |
| Architecture | Mask R-CNN, ResNet-101 Dilated-C5 backbone |
| Framework | Detectron2 v0.6 |
| Parameters | 191,111,222 |
| Checkpoint | iteration 59,999 of 270,000 scheduled |
| Released file | `car_damage_r101_dc5.pth`, 765 MB, fp32 |
| Code licence | Apache-2.0 |
| Weights licence | see [Licence](#licence) below |

## Classes

| id | Vietnamese | English |
|---:|---|---|
| 0 | Móp lõm | Dent |
| 1 | Trầy sơn | Paint scratch |
| 2 | Rách | Tear |
| 3 | Mất bộ phận | Missing part |
| 4 | Thủng | Puncture |
| 5 | Bể đèn | Broken lamp |
| 6 | Vỡ kính | Broken glass |

Background is handled by Detectron2 and is not one of the seven.

## Intended use

Decision support for vehicle inspection — insurance claims, rental check-in and
check-out, resale condition reports. The model highlights damaged regions for a
human to review.

**Out of scope:** automated claim approval or denial without human review;
estimating repair cost; measuring damage in physical units; any use where a
missed or hallucinated detection carries direct financial or legal consequence
without a person in the loop.

## Inputs and outputs

Input: a single RGB photograph. Detectron2 resizes the shortest side to 800 px,
capping the longest at 1333 px.

Output, per detected region:

| Field | |
|---|---|
| `pred_boxes` | xyxy in original image coordinates |
| `pred_masks` | boolean mask at original resolution |
| `pred_classes` | 0–6 |
| `scores` | confidence in [0, 1] |

Default score threshold 0.7.

## Training

| | |
|---|---|
| Initialisation | ImageNet-pretrained ResNet-101 (`detectron2://ImageNetPretrained/MSRA/R-101.pkl`) |
| Images per batch | 16 |
| Base learning rate | 0.02, `WarmupMultiStepLR`, ×0.1 at 210k and 250k |
| Scheduled iterations | 270,000 |
| Iterations reached | **59,999** |
| Train-time short side | random from {640, 672, 704, 736, 768, 800} |

Full recipe and reproduction commands: [TRAINING.md](TRAINING.md).

No training logs survive from the released run — no loss curve, no
`metrics.json`.

## Training data

An internal dataset of vehicle inspection photographs collected in Vietnam,
annotated as polygons in VGG Image Annotator and converted to COCO
instance-segmentation format.

**The dataset is not published.** The images are customer property and contain
licence plates, faces and other identifying detail. The conversion pipeline is
published in full ([DATASET.md](DATASET.md)) so the process can be reproduced on
other data.

## Evaluation

**No accuracy metrics are published.** No labelled test split is distributed
with this repository and no evaluation output survives from the original
training run, so AP cannot currently be computed or verified by anyone.

`tools/evaluate.py` implements the protocol and runs as soon as a labelled split
exists. See [EVALUATION.md](EVALUATION.md).

## Limitations

Summarised here, in full in [LIMITATIONS.md](LIMITATIONS.md):

* an intermediate checkpoint, 22% through its schedule;
* accuracy unmeasured;
* the 0.7 default threshold discards real damage;
* dent / scratch / tear boundaries are ambiguous to human annotators too;
* instance counts are unstable — measure area, not instance count;
* no tracking, so video detections flicker frame to frame;
* small damage is hard at the backbone's stride 16;
* trained on Vietnamese phone photography of passenger cars;
* no conversion from pixel area to physical units, by design.

## Ethical considerations

The model sits in a workflow that decides how much money a person receives after
damage to their vehicle. Two failure modes have direct consequences:

* a **missed** damage under-values a claim against the claimant;
* a **hallucinated** damage supports a claim that should not be paid, or blames
  a renter for damage they did not cause.

Neither rate has been measured. A human inspector must review the output, and
the model's confidence should not be presented to a claimant as a certainty.

Training images depict private property and often contain licence plates and
bystanders. Anything derived from this pipeline should keep that data out of
public artefacts.

## Licence

**Code** — everything in this repository: Apache-2.0 ([LICENSE](../LICENSE)).
It derives from Detectron2, also Apache-2.0; see [NOTICE](../NOTICE) for the
attribution the licence requires.

**Weights** — `car_damage_r101_dc5.pth`, distributed through this repository's
releases: provided as-is, with no warranty of any kind. No performance guarantee
is made or implied; the model is an intermediate checkpoint with no published
accuracy measurement. Evaluate it on your own data before relying on it, and do
not use it to settle an insurance claim without a human adjuster reviewing the
result.

## Citation

See [CITATION.cff](../CITATION.cff).
