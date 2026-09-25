# Deployment

Four ways to run the model, in order of effort.

## 1. The CLI

For batch work and one-off checks, no server needed:

```bash
python tools/demo.py \
    --weights weights/car_damage_r101_dc5.pth \
    --input 'inspection_photos/*.jpg' \
    --output results/ \
    --json results/damages.json
```

Also takes `--video clip.mp4` and `--webcam`. `--threshold` overrides the 0.7
default, `--language en` switches the drawn labels to English.

## 2. Gradio web demo

Browser UI (English) with a threshold slider and example images from
`assets/demo/`.

```bash
pip install -r serve/requirements.txt
python scripts/fetch_weights.py --link

python serve/gradio_app.py
# http://127.0.0.1:7860
```

| Variable | Default | |
|---|---|---|
| `CAR_DAMAGE_WEIGHTS` | `weights/car_damage_r101_dc5.pth` | checkpoint |
| `SCORE_THRESH` | `0.7` | initial slider value |
| `HOST` / `PORT` | `0.0.0.0` / `7860` | bind address |

The model loads once at startup with an internal floor of 0.30; moving the
slider only filters scores (no 765 MB reload per click).

Do not expose this on a public IP without auth, a reverse proxy, and rate limiting.

## 3. HTTP service

```bash
pip install -r serve/requirements.txt

CAR_DAMAGE_WEIGHTS=weights/car_damage_r101_dc5.pth \
    uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

| Endpoint | In | Out |
|---|---|---|
| `GET /health` | — | status, device, threshold, class count |
| `POST /detect` | multipart `file`, `?language=vi\|en` | JSON list of damages |
| `POST /detect/image` | same | PNG with masks and labels drawn |

```bash
curl -s localhost:8000/health
curl -s -F "file=@car.jpg" localhost:8000/detect
curl -s -o out.png -F "file=@car.jpg" "localhost:8000/detect/image?language=en"
```

Each damage in the JSON response carries:

```json
{
  "label": "Paint scratch",
  "class_id": 1,
  "score": 0.9312,
  "box": [412.0, 230.1, 470.4, 275.8],
  "mask_area_px": 1834.5,
  "mask_area_fraction": 0.002
}
```

(Default API `language` is Vietnamese; pass `?language=en` for English labels as above.)

`mask_area_fraction` is the share of the frame the damage covers. There is
deliberately no conversion to cm² — see the note in `cardamage/geometry.py`:
without a reference object of known size in the photo, pixel area does not
determine physical area.

Environment variables:

| Variable | Default | |
|---|---|---|
| `CAR_DAMAGE_WEIGHTS` | `weights/car_damage_r101_dc5.pth` | checkpoint path |
| `CAR_DAMAGE_CONFIG` | `configs/damage_mask_rcnn_R_101_DC5_3x.yaml` | architecture |
| `SCORE_THRESH` | `0.7` | detection threshold |
| `DEVICE` | auto | `cuda`, `cuda:0`, `cpu` |
| `MAX_PIXELS` | `40000000` | uploads above this get HTTP 413 |

Empty or undecodable uploads return HTTP 400.

### Sizing it

The model loads once at startup and is held in memory for the process lifetime.
Two things to plan for:

* **One process holds one model.** Running `uvicorn --workers N` loads the
  weights N times, in N separate GPU contexts. On a single GPU that usually
  means running one worker and queueing, not N workers.
* **Measure the footprint on your own hardware** before sizing a container.
  `nvidia-smi` during a request gives the real number; it depends on input
  resolution, which for this model is whatever Detectron2's resize produces
  from your photos.

CPU inference works and is what you get automatically without a GPU, at roughly
an order of magnitude more latency.

## 4. Docker

```bash
docker build -f serve/Dockerfile -t car-damage .

docker run --gpus all -p 8000:8000 \
    -v $PWD/weights:/weights \
    -e CAR_DAMAGE_WEIGHTS=/weights/car_damage_r101_dc5.pth \
    car-damage
```

The 765 MB checkpoint is **not** baked into the image — mount it. Baking it in
makes every layer push and pull carry it.

The image builds Detectron2 from source, pinned to `v0.6`, which is a slow layer
but a reproducible one. The base is `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`;
match the CUDA tag to your driver.

## Choosing

| Need | Use |
|---|---|
| Score a folder of photos once | CLI |
| Click-through demo in the browser | Gradio |
| Integrate with an existing backend | HTTP service |
| Ship to someone else's infrastructure | Docker |
| No Python at inference time | TorchScript export, [EXPORT.md](EXPORT.md) |
