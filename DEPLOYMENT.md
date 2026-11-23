# CareBot Healthcare AI Deployment Guide

## Purpose

CareBot is a synthetic-data prototype. It is not approved for autonomous
clinical diagnosis, prescribing, triage, or treatment decisions.

## Prerequisites

- Python 3.10 or newer.
- Windows PowerShell commands below can be adapted for Linux/macOS.
- Optional Groq credentials for cloud summarization and speech-to-text.

## Local Installation

```powershell
cd C:\Users\santh\Downloads\carebot
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
```

## Environment Variables

| Variable | Required | Purpose |
|---|---:|---|
| `GROQ_API_KEY` | Optional | Enables Groq summarization/STT |
| `GROQ_MODEL` | Optional | Selects the Groq summary model |
| `GROQ_STT_MODEL` | Optional | Selects the Groq speech-to-text model |
| `CAREBOT_MODEL_PATH` | Optional | Loads a trained spaCy model directory |

Do not commit API keys or secrets. Set them only in the local shell or an
approved secret-management system.

Example PowerShell configuration:

```powershell
$env:GROQ_API_KEY = "set-in-local-secret-manager"
$env:GROQ_MODEL = "approved-model-name"
$env:GROQ_STT_MODEL = "whisper-large-v3"
$env:CAREBOT_MODEL_PATH = "C:\path\to\models\ner"
```

Without Groq configuration or when a provider request fails, CareBot uses a
deterministic local template summary. Without `CAREBOT_MODEL_PATH`, the service
uses the rule-based entity baseline.

## Start the API

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn carebot.api:app --host 127.0.0.1 --port 8000
```

Available interfaces:

- API/Swagger: http://127.0.0.1:8000/
- Realtime UI: http://127.0.0.1:8000/app
- Health: http://127.0.0.1:8000/health
- Text extraction: `POST /summarize`
- Realtime transcription: `WS /ws/realtime`

## Smoke Test

```powershell
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/summarize `
  -H "Content-Type: application/json" `
  -d '{\"transcript\":\"Diagnosis: migraine. Start paracetamol 500 mg. Follow-up in 2 weeks.\"}'
```

Expected behavior:

- Health returns a successful readiness response.
- Empty and oversized transcripts are rejected with Pydantic/HTTP 422
  validation.
- Structured output contains diagnosis, medication, dosage, follow-up, and
  summary fields.

## Realtime Voice Deployment

1. Open the `/app` page in a supported browser.
2. Allow microphone permission.
3. Select **Start microphone**, speak, then select **Stop**.
4. Review the transcript and structured note.
5. A clinician must select **Confirm note**.

Microphone access generally requires HTTPS outside localhost. Local
development may permit microphone access on `127.0.0.1`.

The browser sends audio chunks over the WebSocket endpoint. The server sends
only the approved text workflow through Groq where configured.

## Test and Validate

```powershell
$env:PYTHONPATH = "src"
pytest -q
python scripts/load_test.py
python scripts/demo.py
```

GitHub Actions runs `pytest -q` for pushes and pull requests using Python
3.12. See `.github/workflows/ci.yml`.

## Network and Production Considerations

- Use HTTPS/TLS and a real reverse proxy.
- Require authentication and authorization.
- Apply request-size and rate limits.
- Restrict CORS to approved origins.
- Use a secret manager rather than shell variables or source files.
- Keep dependencies pinned and run vulnerability/upgrade reviews.
- Run behind a dedicated service account with least privilege.
- Use health checks and readiness checks without exposing secrets.

## Logging and Privacy

Do not log raw transcripts or identified patient data. Recommended log fields:

- Timestamp and request ID.
- Endpoint.
- Status code.
- Validation outcome.
- Extraction failure type.
- Latency.
- Model/version identifier.
- Groq fallback reason with no transcript or secret.

In a clinical system, define a data-retention schedule and obtain appropriate
privacy, security, legal, and ethics approval before real patient use.

## Monitoring

Monitor:

- Availability and p95/p99 latency.
- HTTP 4xx/5xx and WebSocket failure rates.
- Pydantic validation failures.
- Missing or malformed entities.
- Groq request failures and template-fallback rates.
- CPU, memory, worker saturation, and queue delays.
- Model and dependency versions.
- Clinician correction and override rates.

Alert on latency or error-rate regressions and on an unexpected increase in
fallback or failed extractions.

## Rollback

1. Record the deployed commit and model dependency versions.
2. Stop traffic or route to the last known-good deployment.
3. Restore the previous application artifact and model directory.
4. Re-run health, smoke, and test commands.
5. Review logs and preserve an audit trail.

## Clinical Safety Boundary

The application is a prototype. Every clinical note requires qualified
clinician review. It must not replace documentation review, diagnosis,
prescribing, medication reconciliation, triage, or emergency care.
