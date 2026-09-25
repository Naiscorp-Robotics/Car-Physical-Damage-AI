#!/usr/bin/env python3
"""Fetch the released checkpoint into weights/ (not committed to git).

Priority:
  1. Copy/symlink from a local path you pass with --from
  2. Copy from sibling release/hub or release/artifacts if present
  3. Download from Hugging Face (needs `hf` / huggingface_hub + access)

    python scripts/fetch_weights.py
    python scripts/fetch_weights.py --from /path/to/model.safetensors
    python scripts/fetch_weights.py --hf Naiscorp/car-damage-maskrcnn-r101-dc5

Does NOT push anything to GitHub.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(ROOT, "weights")
DEST_NAME = "car_damage_r101_dc5.pth"

# Sibling trees under physical_damage/ that already hold the same checkpoint.
LOCAL_CANDIDATES = [
    os.path.join(ROOT, "..", "release", "hub", "CarDamage_R101_DC5.pth"),
    os.path.join(ROOT, "..", "release", "artifacts", "damage_weights.pth"),
    os.path.join(ROOT, "..", "release", "hub", "model.safetensors"),
]


def _copy(src: str, dest: str, link: bool) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.lexists(dest):
        os.remove(dest)
    if link:
        os.symlink(os.path.abspath(src), dest)
        print(f"symlinked {dest} -> {src}")
    else:
        print(f"copying {src} -> {dest} ...")
        shutil.copy2(src, dest)
        print(f"done ({os.path.getsize(dest)} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", help="local checkpoint path (.pth or .safetensors)")
    ap.add_argument(
        "--hf",
        default="Naiscorp/car-damage-maskrcnn-r101-dc5",
        help="Hugging Face model id (used only if no local file found)",
    )
    ap.add_argument(
        "--hf-file",
        default="CarDamage_R101_DC5.pth",
        help="filename inside the HF repo",
    )
    ap.add_argument(
        "--link",
        action="store_true",
        help="symlink instead of copy (saves ~765 MB disk)",
    )
    ap.add_argument(
        "--dest-name",
        default=DEST_NAME,
        help="filename under weights/ (Detectron2 demo expects .pth)",
    )
    args = ap.parse_args()

    dest = os.path.join(DEST_DIR, args.dest_name)
    if os.path.isfile(dest) and not os.path.islink(dest):
        print(f"already present: {dest} ({os.path.getsize(dest)} bytes)")
        return 0

    src = args.src
    if not src:
        for candidate in LOCAL_CANDIDATES:
            if os.path.isfile(candidate):
                src = candidate
                print(f"found local checkpoint: {src}")
                break

    if src:
        if src.endswith(".safetensors"):
            print(
                "WARNING: Detectron2 demo/serve expect a .pth checkpoint.\n"
                "  .safetensors is for the pure-PyTorch cardamage package / HF Hub.\n"
                "  Prefer CarDamage_R101_DC5.pth or damage_weights.pth."
            )
        _copy(src, dest, link=args.link)
        return 0

    print(f"no local checkpoint; downloading {args.hf_file} from {args.hf} ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("install huggingface_hub first:  pip install huggingface_hub", file=sys.stderr)
        return 1

    try:
        path = hf_hub_download(args.hf, args.hf_file)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print(
            "If the repo is private/gated, run `hf auth login` first,\n"
            "or pass --from /path/to/car_damage_r101_dc5.pth",
            file=sys.stderr,
        )
        return 1

    _copy(path, dest, link=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
