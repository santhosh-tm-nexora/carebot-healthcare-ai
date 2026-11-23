from fastapi.testclient import TestClient
import json
import os
import tempfile

from carebot.api import app


client = TestClient(app)


def test_swagger_ui_is_available_at_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/openapi.json" in response.text


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_returns_clinical_note():
    response = client.post(
        "/summarize",
        json={
            "transcript": (
                "Pt c/o headache. Diagnosis: migraine. "
                "Start paracetamol 500 mg twice daily. Follow-up in 2 weeks."
            )
        },
    )
    assert response.status_code == 200
    assert response.json()["diagnosis"] == ["migraine"]
    assert response.json()["medications"][0]["name"] == "paracetamol"


def test_summarize_rejects_empty_transcript():
    response = client.post("/summarize", json={"transcript": ""})
    assert response.status_code == 422


def test_realtime_app_is_available():
    response = client.get("/app")
    assert response.status_code == 200
    assert "Start microphone" in response.text


def test_realtime_transcript_returns_live_note():
    with client.websocket_connect("/ws/realtime") as websocket:
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_json(
            {
                "type": "transcript",
                "text": "Diagnosis: migraine. Start paracetamol 500 mg.",
                "final": True,
            }
        )
        message = websocket.receive_json()
        assert message["type"] == "clinical_note"
        assert message["final"] is True
        assert message["clinical_note"]["diagnosis"] == ["migraine"]
        assert message["clinical_note"]["medications"][0]["name"] == "paracetamol"


def test_realtime_rejects_unknown_message_type():
    with client.websocket_connect("/ws/realtime") as websocket:
        websocket.send_json({"type": "unknown"})
        message = websocket.receive_json()
        assert message["type"] == "error"


def test_realtime_app_uses_server_audio_pipeline():
    response = client.get("/app")
    assert response.status_code == 200
    assert "MediaRecorder" in response.text
    assert "getUserMedia" in response.text
    assert "/ws/realtime" in response.text


def test_realtime_status_reports_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.get("/realtime/status")
    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_realtime_audio_stop_transcribes_and_extracts(monkeypatch):
    monkeypatch.setattr(
        "carebot.realtime.transcribe_audio",
        lambda audio: (
            "Diagnosis: migraine. Start paracetamol 500 mg twice daily. "
            "Follow-up in 2 weeks."
        ),
    )
    with client.websocket_connect("/ws/realtime") as websocket:
        websocket.send_json({"type": "start"})
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(b"fake-webm-audio")
        websocket.send_json({"type": "stop"})
        message = websocket.receive_json()
        stages = []
        while message["type"] == "stage":
            stages.append(message)
            message = websocket.receive_json()
        assert any(item.get("stage") == "transcription" for item in stages)
        assert any(item.get("stage") == "ner" for item in stages)
        assert any(item.get("stage") == "llm" for item in stages)
        assert any(item.get("stage") == "validation" for item in stages)
        assert message["type"] == "clinical_note"
        assert message["final"] is True
        assert message["clinical_note"]["diagnosis"] == ["migraine"]
        assert message["clinical_note"]["medications"][0]["dosage"] == "500 mg twice daily"


def test_realtime_confirm_requires_valid_clinical_note():
    with client.websocket_connect("/ws/realtime") as websocket:
        websocket.send_json({"type": "confirm", "clinical_note": {"diagnosis": "migraine"}})
        message = websocket.receive_json()
        assert message["type"] == "error"
        assert message["message"] == "Clinical note validation failed"


def test_reprocess_corrected_transcript(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    with client.websocket_connect("/ws/realtime") as websocket:
        websocket.send_json(
            {
                "type": "reprocess",
                "transcript": (
                    "Diagnosis: asthma. Start metformin 500 mg twice daily. "
                    "Follow-up in 1 week."
                ),
            }
        )
        message = websocket.receive_json()
        stages = []
        while message["type"] == "stage":
            stages.append(message)
            message = websocket.receive_json()
        assert any(item.get("stage") == "ner" for item in stages)
        assert any(item.get("stage") == "llm" for item in stages)
        assert any(item.get("stage") == "validation" for item in stages)
        assert message["type"] == "clinical_note"
        assert message["reprocessed"] is True
        assert message["clinical_note"]["diagnosis"] == ["asthma"]
        assert message["clinical_note"]["medications"][0]["name"] == "metformin"
        assert message["clinical_note"]["follow_up"]


def test_realtime_confirm_persists_reviewed_note(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database_path = os.path.join(directory, "confirmed.sqlite3")
        monkeypatch.setenv("CAREBOT_RECORD_DB", database_path)
        note = {
            "diagnosis": ["migraine"],
            "medications": [{"name": "paracetamol", "dosage": "500 mg twice daily"}],
            "follow_up": ["2 weeks"],
            "summary": "Migraine with paracetamol 500 mg twice daily and follow-up in 2 weeks.",
        }
        with client.websocket_connect("/ws/realtime") as websocket:
            websocket.send_json({
                "type": "confirm",
                "transcript": "Diagnosis: migraine. Follow-up in 2 weeks.",
                "clinical_note": note,
            })
            message = websocket.receive_json()
            assert message["type"] == "confirmed"
            assert message["record_id"]

        import sqlite3

        connection = sqlite3.connect(database_path)
        try:
            row = connection.execute(
                "SELECT clinical_note_json FROM confirmed_notes"
            ).fetchone()
        finally:
            connection.close()
        assert json.loads(row[0])["diagnosis"] == ["migraine"]
