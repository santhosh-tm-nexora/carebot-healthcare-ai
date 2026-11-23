# CareBot Deployment Guide

## Local deployment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pip install -e .
python -m uvicorn carebot.api:app --host 127.0.0.1 --port 8000
```

Test the service:

```bash
start http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/summarize ^
  -H "Content-Type: application/json" ^
  -d "{\"transcript\":\"Diagnosis: migraine. Start paracetamol 500 mg. Follow-up in 2 weeks.\"}"
```

The Swagger API documentation is served at `http://127.0.0.1:8000/`.

Open `http://127.0.0.1:8000/app` for the realtime doctor voice workflow.
Allow microphone access, select **Start microphone**, speak, then select
**Stop**. The browser sends audio chunks to the FastAPI `/ws/realtime`
WebSocket; Groq Whisper transcribes the recording, spaCy extracts entities,
and Groq generates the final summary. Review the result and select
**Confirm note**. Set `GROQ_API_KEY` and optionally `GROQ_STT_MODEL`.
Use HTTPS when deploying outside localhost.

Set `CAREBOT_MODEL_PATH` to a spaCy model directory to select a trained model.
If the path is absent, the service uses the rule-based baseline.

## Operational safeguards

Do not send real patient data to this prototype. Keep logs free of raw
transcripts, restrict access to the API, pin dependencies, and review outputs
by a qualified clinician. The service is not approved for autonomous diagnosis
or treatment decisions.
