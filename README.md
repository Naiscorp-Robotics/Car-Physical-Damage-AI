# Car Damage Detection

Instance segmentation of **seven kinds of physical damage on cars**, for insurance,
rental and resale inspection. Given a photo, the model returns a pixel mask, a
bounding box, a damage class and a confidence score for every damaged region it
finds.

This repository holds the full pipeline the model was built with — dataset
conversion, training config, training entry point, evaluation, inference and
serving.


## Damage classes

| id | Vietnamese | English |
|---:|---|---|
| 0 | Móp lõm | Dent |
| 1 | Trầy sơn | Paint scratch |
| 2 | Rách | Tear |
| 3 | Mất bộ phận | Missing part |
| 4 | Thủng | Puncture |
| 5 | Bể đèn | Broken lamp |
| 6 | Vỡ kính | Broken glass |

## Install

Detectron2 is not on PyPI, so it installs separately, after torch:

```bash
git clone https://github.com/Naiscorp-Robotics/Car-Physical-Damage-AI
cd Car-Physical-Damage-AI
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
```

Then download `car_damage_r101_dc5.pth` from 
https://huggingface.co/Naiscorp/car-damage-detection into `weights/`.

## Quick start

```bash
python tools/demo.py \
    --weights weights/car_damage_r101_dc5.pth \
    --input car.jpg --output car_damage.jpg --json car_damage.json
```

From Python:

```python
import cv2
from cardamage import build_predictor, DAMAGE_CLASSES_EN
from cardamage.visualize import draw_masks_and_boxes

predictor = build_predictor("weights/car_damage_r101_dc5.pth")

image = cv2.imread("car.jpg")            # BGR, as Detectron2 expects
outputs = predictor(image)

instances = outputs["instances"].to("cpu")
for class_id, score in zip(instances.pred_classes, instances.scores):
    print(f"{DAMAGE_CLASSES_EN[class_id]:14s} {score:.3f}")

overlay = draw_masks_and_boxes(image[:, :, ::-1], outputs)   # RGB array
cv2.imwrite("car_damage.jpg", overlay[:, :, ::-1])
```

## Architecture

Measured directly from the released checkpoint:

| | |
|---|---|
| Meta architecture | Mask R-CNN (`GeneralizedRCNN`) |
| Backbone | ResNet-101, **Dilated-C5** — `res5` dilation 2, single feature map, stride 16 |
| Region proposals | `StandardRPNHead` on `res5` |
| ROI heads | `StandardROIHeads`, 7×7 pooler → 2 × FC-1024 |
| Mask head | 14×14 pooler → 4 × conv-256 → deconv, 28×28 output |
| Parameters | **191,111,222** across 546 tensors |
| Checkpoint size | 765 MB (fp32, optimizer state stripped) |

Why DC5 and not FPN: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How the model was built

```
   VIA annotations              datasets/via2coco/convert.py
          ↓                     → COCO instance-segmentation JSON + train/val split
   registered dataset           datasets/register.py
          ↓
   training                     tools/train_net.py
          ↓                     configs/damage_mask_rcnn_R_101_DC5_3x.yaml
   checkpoint (1527 MB)         batch 16, LR 0.02, 59,999 iterations reached
          ↓
   published weights (765 MB)   tools/strip_optimizer.py
          ↓
   inference / serving          tools/demo.py · serve/app.py · tools/export_model.py
```

| Step | Document |
|---|---|
| Data format and annotation | [docs/DATASET.md](docs/DATASET.md) |
| Reproducing the training run | [docs/TRAINING.md](docs/TRAINING.md) |
| Architecture rationale | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Scoring a checkpoint | [docs/EVALUATION.md](docs/EVALUATION.md) |
| TorchScript / ONNX export | [docs/EXPORT.md](docs/EXPORT.md) |
| Running it as a service | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Known limitations | [docs/LIMITATIONS.md](docs/LIMITATIONS.md) |
| Full model card | [docs/MODEL_CARD.md](docs/MODEL_CARD.md) |

## Results

**No benchmark numbers are published yet.** The dataset this model was trained on
is internal and is not distributed with the repository, and no labelled test
split is available here, so AP cannot be recomputed. `tools/evaluate.py` is
ready to run the moment a labelled split exists:

```bash
python tools/evaluate.py --weights weights/car_damage_r101_dc5.pth \
                         --dataset-root datasets/car_damage
```

Publishing a results table with numbers that cannot be reproduced from this
repository would be worse than publishing none. See
[docs/EVALUATION.md](docs/EVALUATION.md) for the protocol that will be used.

## Serve it

### Web demo (Gradio)

```bash
pip install -r serve/requirements.txt
python scripts/fetch_weights.py --link   # or download the Release asset into weights/

mkdir -p ~/tmp/gradio
GRADIO_TEMP_DIR=~/tmp/gradio python serve/gradio_app.py
# http://127.0.0.1:7860  — upload a photo or click an example under assets/demo/
```

### HTTP API (FastAPI)

```bash
pip install -r serve/requirements.txt
CAR_DAMAGE_WEIGHTS=weights/car_damage_r101_dc5.pth \
    uvicorn serve.app:app --host 0.0.0.0 --port 8000

curl -s -F "file=@car.jpg" localhost:8000/detect | jq
curl -s -o out.png -F "file=@car.jpg" "localhost:8000/detect/image?language=en"
```

Or with Docker — see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Publishing this repo

Push to GitHub yourself. Do not commit `weights/*.pth` (~765 MB) — attach them as
GitHub Release assets instead. Review demo images for PII before going public.
See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for weight licence terms.

## Intended use

A **decision-support** tool for vehicle inspection. It is not calibrated for
automated claim decisions, and it should not be used to settle an insurance
claim without a human adjuster reviewing the output.

## Licence

Source code: **Apache-2.0** ([LICENSE](LICENSE)). It derives from Detectron2,
also Apache-2.0 — see [NOTICE](NOTICE) for the attribution the licence requires.

Model weights are distributed under separate terms stated in
[docs/MODEL_CARD.md](docs/MODEL_CARD.md).
