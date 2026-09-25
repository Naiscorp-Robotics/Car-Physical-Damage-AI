---
name: Bug report
about: Something does not work as documented
labels: bug
---

**What happened**

<!-- and what you expected instead -->

**How to reproduce**

```bash
# the exact command
```

**Environment**

Results depend on the versions below — please fill all of them in:

| | |
|---|---|
| OS | |
| Python | |
| torch | `python -c "import torch; print(torch.__version__)"` |
| detectron2 | `python -c "import detectron2; print(detectron2.__version__)"` |
| CUDA / driver | `nvidia-smi` first line, or "CPU only" |
| Checkpoint | release tag, or "trained my own" |

**Output**

<!-- full traceback, or the JSON the model returned -->
