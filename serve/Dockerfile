# Build from the repository root:
#     docker build -f serve/Dockerfile -t car-damage .
#
# The checkpoint is not baked into the image - mount it at run time:
#     docker run --gpus all -p 8000:8000 \
#         -v $PWD/weights:/weights \
#         -e CAR_DAMAGE_WEIGHTS=/weights/car_damage_r101_dc5.pth \
#         car-damage
#
# The production Dockerfile this replaces pinned python 3.7 / torch 1.11+cu113
# and installed Detectron2 from an unpinned master, so it stopped reproducing.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt serve/requirements.txt ./serve/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r serve/requirements.txt \
    && pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

COPY cardamage/ ./cardamage/
COPY configs/ ./configs/
COPY serve/ ./serve/

EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
