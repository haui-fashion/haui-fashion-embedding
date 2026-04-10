FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface

ARG MODEL_NAME=intfloat/multilingual-e5-base
ENV MODEL_NAME=${MODEL_NAME}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Preload model during build so container startup does not need to download weights.
RUN python - <<'PY'
import os
from sentence_transformers import SentenceTransformer

model_name = os.environ["MODEL_NAME"]
cache_dir = os.environ.get("HF_HOME")
SentenceTransformer(model_name, device="cpu", cache_folder=cache_dir)
print(f"Preloaded model '{model_name}' into '{cache_dir}'")
PY

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
