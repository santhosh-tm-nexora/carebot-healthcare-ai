from fastapi.testclient import TestClient

from carebot.api import app


client = TestClient(app)


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
