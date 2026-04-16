from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    normalize: bool | None = None

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        cleaned = [item.strip() for item in values if item and item.strip()]
        if not cleaned:
            raise ValueError("texts must contain at least one non-empty string")
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "64"))
        if len(cleaned) > max_batch_size:
            raise ValueError(f"batch size exceeded MAX_BATCH_SIZE={max_batch_size}")
        return cleaned


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


model: SentenceTransformer | None = None
model_name = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base")
device = os.getenv("DEVICE", "cpu")
default_normalize = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
local_files_only = os.getenv("LOCAL_FILES_ONLY", "false").lower() == "true"
cache_folder = os.getenv("HF_HOME")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("embedding-service")

EMBEDDING_REQUESTS_TOTAL = Counter(
    "embedding_requests_total",
    "Total embedding requests",
    ["endpoint", "status"],
)
EMBEDDING_REQUEST_DURATION_SECONDS = Histogram(
    "embedding_request_duration_seconds",
    "Embedding request duration in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def log_event(event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "service": "embedding",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    log_event("embedding_model_loading", model=model_name, device=device)
    model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=cache_folder,
        local_files_only=local_files_only,
    )
    log_event("embedding_model_ready", model=model_name, device=device)
    yield


app = FastAPI(title="Embedding Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    ready = "ok" if model is not None else "loading"
    return {"status": ready, "model": model_name, "device": device}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _embed_texts(payload: EmbedRequest, input_type: str) -> EmbedResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    normalize = default_normalize if payload.normalize is None else payload.normalize
    prefixed_texts = [f"{input_type}: {text}" for text in payload.texts]

    vectors = model.encode(
        prefixed_texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if isinstance(vectors, np.ndarray) and vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    embeddings = vectors.tolist() if isinstance(vectors, np.ndarray) else [list(v) for v in vectors]
    dimension = len(embeddings[0]) if embeddings else 0

    log_event(
        "embedding_generated",
        input_type=input_type,
        batch_size=len(payload.texts),
        dimension=dimension,
        normalize=normalize,
    )

    return EmbedResponse(model=model_name, dimension=dimension, embeddings=embeddings)


@app.post("/query", response_model=EmbedResponse)
def embed_query(payload: EmbedRequest) -> EmbedResponse:
    started_at = time.perf_counter()
    try:
        response = _embed_texts(payload, "query")
        EMBEDDING_REQUESTS_TOTAL.labels(endpoint="query", status="success").inc()
        return response
    except Exception:
        EMBEDDING_REQUESTS_TOTAL.labels(endpoint="query", status="error").inc()
        raise
    finally:
        EMBEDDING_REQUEST_DURATION_SECONDS.labels(endpoint="query").observe(
            time.perf_counter() - started_at
        )


@app.post("/passage", response_model=EmbedResponse)
def embed_passage(payload: EmbedRequest) -> EmbedResponse:
    started_at = time.perf_counter()
    try:
        response = _embed_texts(payload, "passage")
        EMBEDDING_REQUESTS_TOTAL.labels(endpoint="passage", status="success").inc()
        return response
    except Exception:
        EMBEDDING_REQUESTS_TOTAL.labels(endpoint="passage", status="error").inc()
        raise
    finally:
        EMBEDDING_REQUEST_DURATION_SECONDS.labels(endpoint="passage").observe(
            time.perf_counter() - started_at
        )
