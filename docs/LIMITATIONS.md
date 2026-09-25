# Limitations

Read this before putting the model in front of anything that matters.

## The checkpoint is 22% of the way through training

Iteration 59,999 of a scheduled 270,000. The learning rate never reached either
decay step (210,000 and 250,000), so these weights come from the flat, high-LR
part of the schedule — the region where a model is usually still improving.
It is released because it is what ran in production, not because it was chosen
as the best checkpoint.

## No accuracy has been measured

There is no labelled test split in this repository and no evaluation output
survives from the original run, so nobody can currently tell you the AP. See
[EVALUATION.md](EVALUATION.md).

This is the single biggest gap. Treat every claim about the model's quality as
unverified until you score it on your own data.

## The 0.7 default threshold loses real damage

The production threshold favours precision. Damage that a person sees
immediately can score below it and vanish from the output entirely. Lowering to
0.5 recovers some of it at the cost of false positives.

There is no threshold that is right for every use. Pick yours from the
precision/recall trade-off your workflow can absorb: a screening tool that flags
photos for human review wants recall; an auto-approval path wants precision, and
probably should not exist for this model at all.

## Class boundaries are genuinely ambiguous

`Móp lõm` (dent), `Trầy sơn` (scratch) and `Rách` (tear) shade into each other,
and a crease with paint damage is legitimately two classes at once. Human
annotators disagree on these, which puts a ceiling on what the model can learn
and makes per-class AP partly a measure of annotation consistency.

Where the class matters more than the region, treat the model's label as a
suggestion.

## Instance boundaries are unstable

A long scratch may come back as one instance or as several, depending on
lighting and angle. Any downstream logic that counts damages — "three or more
means total loss" — will be unstable for this reason. Count damaged *area* or
damaged *panels* rather than instances.

## Video is frame-independent

The model has no tracking. Each frame is inferred from scratch, so detections
flicker: a damage present in one frame can be absent in the next, and instance
identity is not preserved across frames. `tools/demo.py --video` reflects this
honestly rather than smoothing it.

Stabilising video output needs a real tracker with a motion model (SORT,
ByteTrack). Naive IoU-based smoothing works only for a static camera and inflates
counts badly when the camera pans.

## Small damage at stride 16

The DC5 backbone produces a single stride-16 feature map
([ARCHITECTURE.md](ARCHITECTURE.md)). Hairline scratches a few pixels wide have
little signal left at that stride, and this is where the model misses most.
Photograph small damage closer rather than expecting the model to find it in a
wide shot.

## Domain

Training photos were vehicle inspection shots taken in Vietnam: phone cameras,
mixed lighting, mostly passenger cars. Expect degradation on anything far from
that — studio photography, drone shots, heavy vehicles, snow, night.

## No physical measurement

The API returns mask area in pixels and as a fraction of the frame. It does not
return cm², and it should not: a single photo carries no scale. Physical
measurement needs a reference object of known size in frame, or a calibrated
camera rig.

## Not for automated decisions

This is decision support for a human inspector. It is not calibrated for
automated claim approval or denial, and using it that way would put an
unevaluated, under-trained model between a person and their money.
