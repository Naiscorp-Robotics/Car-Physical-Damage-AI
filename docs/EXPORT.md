# Export

## Stripping the checkpoint first

A Detectron2 training checkpoint carries three top-level keys: `model`,
`iteration` and `trainer` (optimizer moments, scheduler state, hook state).
Inference reads only `model`.

```bash
python tools/strip_optimizer.py \
    output/damage_r101_dc5/model_0059999.pth \
    weights/car_damage_r101_dc5.pth
```

```
dropped keys : trainer
tensors      : 546
parameters   : 191,111,222
iteration    : 59999
size         : 1527 MB -> 765 MB
```

Do this before uploading anything. Beyond halving the file, the optimizer state
is a training artefact nobody downloading a model needs.

## TorchScript and ONNX

```bash
python tools/export_model.py \
    --config-file configs/damage_mask_rcnn_R_101_DC5_3x.yaml \
    --format torchscript \
    --export-method tracing \
    --sample-image assets/demo/sample.jpg \
    --output output/export \
    MODEL.WEIGHTS weights/car_damage_r101_dc5.pth \
    MODEL.DEVICE cuda
```

`--format` takes `torchscript`, `onnx` or `caffe2_tracing`. `--export-method`
takes `tracing` or `scripting`; **tracing is the one to use here** — Mask R-CNN's
control flow does not script cleanly.

### What tracing costs you

A traced graph records the operations that ran for one specific input. Three
consequences follow, and all three bite this model:

* **Input size is baked in.** The traced graph carries the resize behaviour that
  fired for `--sample-image`. Feed it a differently shaped image and you get
  either wrong output or a shape error. Export at the resolution you will serve.
* **Detection count is dynamic, tracing is not.** NMS returns a variable number
  of boxes. Detectron2 works around this with `TracingAdapter`, which flattens
  the output into plain tensors — the exported model returns tensors, not an
  `Instances` object, and you reassemble the fields yourself.
* **Postprocessing is outside the graph.** Mask pasting back to the original
  resolution (`paste_masks_in_image`) does not survive tracing intact. The
  exported model gives you 28×28 mask logits per detection; pasting is your job
  on the other side.

Because of this, the exported artefact is genuinely lower-level than the Python
predictor. If you do not need it, `serve/app.py` on plain PyTorch is simpler and
gives identical results.

### Verifying an export

`tools/export_model.py` accepts `--run-eval`, which scores the exported model
with `COCOEvaluator` on `DATASETS.TEST` right after exporting:

```bash
python tools/export_model.py ... --run-eval
```

Compare that against `tools/evaluate.py` on the PyTorch model. They should agree
to within floating-point noise; a real gap means the tracing assumptions above
were violated. (This needs a labelled split — see [EVALUATION.md](EVALUATION.md).)

A C++ sample consumer is not shipped in this repository. Load the `.ts` file with
LibTorch in your own binary if you need inference outside Python.
