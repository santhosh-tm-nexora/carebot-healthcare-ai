"""FastAPI serving layer for the CareBot summarization pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .ner import load_nlp
from .pipeline import run
from .schema import ClinicalNote


class SummarizeRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=20_000)


def _load_model():
    model_path = Path(os.getenv("CAREBOT_MODEL_PATH", "models/experiments/ner_config_2"))
    return load_nlp(str(model_path)) if model_path.exists() else load_nlp()


nlp = _load_model()
app = FastAPI(title="CareBot Healthcare AI", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize", response_model=ClinicalNote)
def summarize(request: SummarizeRequest) -> ClinicalNote:
    return run(request.transcript, nlp=nlp)
