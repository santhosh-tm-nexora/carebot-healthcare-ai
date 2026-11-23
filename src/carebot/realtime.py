"""Realtime transcript session handling for the doctor voice workflow."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .ner import extract
from .preprocess import normalize
from pydantic import ValidationError

from .schema import ClinicalNote
from .summarize import summarize, template_summary

MAX_TRANSCRIPT_LENGTH = 20_000


def _record_database_path() -> Path:
    path = Path(os.getenv("CAREBOT_RECORD_DB", "data/confirmed_notes.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def persist_confirmed_note(note: ClinicalNote, corrected_transcript: str | None) -> str:
    record_id = str(uuid4())
    connection = sqlite3.connect(_record_database_path())
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS confirmed_notes (
                record_id TEXT PRIMARY KEY,
                confirmed_at TEXT NOT NULL,
                clinical_note_json TEXT NOT NULL,
                transcript TEXT
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(confirmed_notes)").fetchall()
        }
        if "transcript" not in columns:
            connection.execute(
                "ALTER TABLE confirmed_notes ADD COLUMN transcript TEXT"
            )
        connection.execute(
            "INSERT INTO confirmed_notes(record_id, confirmed_at, clinical_note_json, transcript) "
            "VALUES (?, ?, ?, ?)",
            (
                record_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(note.model_dump()),
                corrected_transcript,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return record_id


def transcribe_audio(audio: bytes) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for realtime audio transcription")

    if not audio:
        raise RuntimeError("No audio data was received")

    from openai import OpenAI

    suffix = ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
        audio_file.write(audio)
        audio_path = audio_file.name

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        with open(audio_path, "rb") as audio_stream:
            response = client.audio.transcriptions.create(
                model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3"),
                file=(os.path.basename(audio_path), audio_stream),
                response_format="text",
            )
        if hasattr(response, "text"):
            return response.text.strip()
        if isinstance(response, str):
            return response.strip()
        raise RuntimeError("Groq returned no transcription text")
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def _note_for_transcript(text: str, nlp: Any, final: bool) -> ClinicalNote:
    clean = normalize(text)
    entities = extract(nlp, clean)
    summary = summarize(clean, entities) if final else template_summary(entities)
    return ClinicalNote(**entities, summary=summary)


def reprocess_corrected_transcript(text: str, nlp: Any) -> ClinicalNote:
    """Re-run the full text pipeline after clinician correction."""
    clean = normalize(text)
    entities = extract(nlp, clean)
    clinical_note = ClinicalNote(**entities, summary=summarize(clean, entities))
    return ClinicalNote.model_validate(clinical_note)


async def handle_realtime_session(websocket: WebSocket, nlp: Any) -> None:
    """Receive speech-provider transcript events and emit live clinical notes."""
    await websocket.accept()
    audio_chunks: list[bytes] = []
    text_transcript = ""
    try:
        while True:
            event = await websocket.receive()
            if event["type"] == "websocket.disconnect":
                return

            if "bytes" in event:
                audio_chunks.append(event["bytes"])
                continue

            if "text" not in event:
                continue

            import json

            message = json.loads(event["text"])
            message_type = message.get("type")

            if message_type == "start":
                audio_chunks = []
                text_transcript = ""
                await websocket.send_json(
                    {"type": "ready", "message": "Realtime clinical note session ready"}
                )
                continue

            if message_type == "transcribe_interim":
                if not audio_chunks:
                    continue
                try:
                    text_transcript = await asyncio.to_thread(
                        transcribe_audio,
                        b"".join(audio_chunks),
                    )
                    if len(text_transcript) > MAX_TRANSCRIPT_LENGTH:
                        raise RuntimeError("Transcript exceeds the 20,000 character limit")
                    note = _note_for_transcript(text_transcript, nlp, False)
                    await websocket.send_json(
                        {
                            "type": "clinical_note",
                            "final": False,
                            "transcript": text_transcript,
                            "clinical_note": note.model_dump(),
                        }
                    )
                except Exception as error:
                    await websocket.send_json(
                        {"type": "error", "message": f"Interim transcription failed: {error}"}
                    )
                continue

            if message_type == "reprocess":
                text = message.get("transcript")
                if not isinstance(text, str) or not text.strip():
                    await websocket.send_json(
                        {"type": "error", "message": "Corrected transcript cannot be empty"}
                    )
                    continue
                if len(text) > MAX_TRANSCRIPT_LENGTH:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Corrected transcript exceeds the 20,000 character limit",
                        }
                    )
                    continue
                try:
                    await websocket.send_json(
                        {"type": "stage", "stage": "ner", "message": "Reprocessing corrected transcript"}
                    )
                    await websocket.send_json(
                        {"type": "stage", "stage": "llm", "message": "Regenerating entity-constrained summary"}
                    )
                    note = await asyncio.to_thread(
                        reprocess_corrected_transcript,
                        text,
                        nlp,
                    )
                    await websocket.send_json(
                        {"type": "stage", "stage": "validation", "message": "Validating corrected clinical note"}
                    )
                    await websocket.send_json(
                        {
                            "type": "clinical_note",
                            "final": True,
                            "reprocessed": True,
                            "transcript": text,
                            "clinical_note": note.model_dump(),
                        }
                    )
                except Exception as error:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Corrected transcript reprocessing failed: {error}",
                        }
                    )
                continue

            if message_type == "stop":
                try:
                    await websocket.send_json(
                        {"type": "stage", "stage": "transcription", "message": "Transcribing final audio"}
                    )
                    text_transcript = await asyncio.to_thread(
                        transcribe_audio,
                        b"".join(audio_chunks),
                    )
                    if len(text_transcript) > MAX_TRANSCRIPT_LENGTH:
                        raise RuntimeError("Transcript exceeds the 20,000 character limit")
                    await websocket.send_json(
                        {
                            "type": "stage",
                            "stage": "ner",
                            "message": "Extracting entities",
                            "transcript": text_transcript,
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "stage",
                            "stage": "llm",
                            "message": "Generating entity-constrained summary",
                        }
                    )
                    note = _note_for_transcript(text_transcript, nlp, True)
                    await websocket.send_json(
                        {
                            "type": "stage",
                            "stage": "validation",
                            "message": "Validating structured clinical note",
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "clinical_note",
                            "final": True,
                            "transcript": text_transcript,
                            "clinical_note": note.model_dump(),
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "stage",
                            "stage": "review",
                            "message": "Review and confirm the clinical note",
                        }
                    )
                except Exception as error:
                    await websocket.send_json(
                        {"type": "error", "message": f"Audio transcription failed: {error}"}
                    )
                continue

            if message_type == "transcript":
                text = message.get("text")
                final = message.get("final", False)
                if not isinstance(text, str) or not text.strip():
                    await websocket.send_json(
                        {"type": "error", "message": "Transcript text cannot be empty"}
                    )
                    continue
                if len(text) > MAX_TRANSCRIPT_LENGTH:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Transcript exceeds the 20,000 character limit",
                        }
                    )
                    continue

                text_transcript = text
                note = _note_for_transcript(text, nlp, bool(final))
                await websocket.send_json(
                    {
                        "type": "clinical_note",
                        "final": bool(final),
                        "transcript": text,
                        "clinical_note": note.model_dump(),
                    }
                )
                continue

            if message_type == "confirm":
                note = message.get("clinical_note")
                corrected_transcript = message.get("transcript")
                if not isinstance(note, dict):
                    await websocket.send_json(
                        {"type": "error", "message": "A clinical note is required for confirmation"}
                    )
                    continue
                if corrected_transcript is not None:
                    if not isinstance(corrected_transcript, str) or not corrected_transcript.strip():
                        await websocket.send_json(
                            {"type": "error", "message": "Corrected transcript cannot be empty"}
                        )
                        continue
                    if len(corrected_transcript) > MAX_TRANSCRIPT_LENGTH:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "Corrected transcript exceeds the 20,000 character limit",
                            }
                        )
                        continue
                try:
                    if isinstance(corrected_transcript, str):
                        validated_note = await asyncio.to_thread(
                            reprocess_corrected_transcript,
                            corrected_transcript,
                            nlp,
                        )
                    else:
                        validated_note = ClinicalNote.model_validate(note)
                except ValidationError as error:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Clinical note validation failed",
                            "details": error.errors(include_url=False),
                        }
                    )
                    continue
                record_id = await asyncio.to_thread(
                    persist_confirmed_note,
                    validated_note,
                    corrected_transcript,
                )
                await websocket.send_json(
                    {
                        "type": "confirmed",
                        "message": "Clinical note reviewed and confirmed by clinician",
                        "clinical_note": validated_note.model_dump(),
                        "transcript_corrected": corrected_transcript is not None,
                        "record_id": record_id,
                    }
                )
                continue

            await websocket.send_json(
                {"type": "error", "message": f"Unsupported realtime message type: {message_type}"}
            )
    except WebSocketDisconnect:
        return
