"""FastAPI serving layer for the CareBot summarization pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .ner import load_nlp
from .pipeline import run
from .realtime import handle_realtime_session
from .schema import ClinicalNote


class SummarizeRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=20_000)


def _load_model():
    model_path = Path(os.getenv("CAREBOT_MODEL_PATH", "models/experiments/ner_config_2"))
    return load_nlp(str(model_path)) if model_path.exists() else load_nlp()


nlp = _load_model()
app = FastAPI(
    title="CareBot Healthcare AI",
    version="0.1.0",
    docs_url="/",
    redoc_url=None,
)

@app.get("/app", include_in_schema=False)
def realtime_app() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/realtime/status")
def realtime_status() -> dict[str, bool | str]:
    configured = bool(os.getenv("GROQ_API_KEY"))
    return {
        "ready": configured,
        "message": (
            "Realtime audio transcription ready"
            if configured
            else "GROQ_API_KEY is required for realtime audio transcription"
        ),
    }

@app.post("/summarize", response_model=ClinicalNote)
def summarize(request: SummarizeRequest) -> ClinicalNote:
    return run(request.transcript, nlp=nlp)


@app.websocket("/ws/realtime")
async def realtime(websocket: WebSocket) -> None:
    await handle_realtime_session(websocket, nlp)
