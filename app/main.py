from __future__ import annotations

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = SentenceTransformer(model_name, device=device)
    yield


app = FastAPI(title="Embedding Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    ready = "ok" if model is not None else "loading"
    return {"status": ready, "model": model_name, "device": device}


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

    return EmbedResponse(model=model_name, dimension=dimension, embeddings=embeddings)


@app.post("/query", response_model=EmbedResponse)
def embed_query(payload: EmbedRequest) -> EmbedResponse:
    return _embed_texts(payload, "query")


@app.post("/passage", response_model=EmbedResponse)
def embed_passage(payload: EmbedRequest) -> EmbedResponse:
    return _embed_texts(payload, "passage")
