#!/usr/bin/env python3
"""Drop optimizer state from a Detectron2 checkpoint before publishing it.

A checkpoint saved mid-training carries a ``trainer`` key holding the optimizer
moments, the scheduler and the hook state. None of it is needed for inference,
and it is roughly half the file:

    model_0059999.pth   1527 MB  ->  765 MB

Usage:

    python tools/strip_optimizer.py IN.pth OUT.pth
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="training checkpoint")
    ap.add_argument("dst", help="where to write the inference-only checkpoint")
    ap.add_argument("--force", action="store_true", help="overwrite dst if it exists")
    args = ap.parse_args()

    if os.path.exists(args.dst) and not args.force:
        print(f"refusing to overwrite {args.dst} (pass --force)", file=sys.stderr)
        return 1

    import torch

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        print(f"{args.src}: no 'model' key - is this a Detectron2 checkpoint?", file=sys.stderr)
        return 1

    kept = {"model": ckpt["model"], "__author__": "Detectron2 Model Zoo"}
    if "iteration" in ckpt:
        kept["iteration"] = ckpt["iteration"]

    dropped = sorted(set(ckpt) - set(kept))
    torch.save(kept, args.dst)

    src_mb = os.path.getsize(args.src) / 1e6
    dst_mb = os.path.getsize(args.dst) / 1e6
    params = sum(v.numel() for v in kept["model"].values() if hasattr(v, "numel"))
    print(f"dropped keys : {', '.join(dropped) or '(none)'}")
    print(f"tensors      : {len(kept['model'])}")
    print(f"parameters   : {params:,}")
    print(f"iteration    : {kept.get('iteration', 'unknown')}")
    print(f"size         : {src_mb:.0f} MB -> {dst_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
