# CareBot Deployment Guide

## Local deployment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=src
uvicorn carebot.api:app --host 127.0.0.1 --port 8000
```

Test the service:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/summarize ^
  -H "Content-Type: application/json" ^
  -d "{\"transcript\":\"Diagnosis: migraine. Start paracetamol 500 mg. Follow-up in 2 weeks.\"}"
```

Set `CAREBOT_MODEL_PATH` to a spaCy model directory to select a trained model.
If the path is absent, the service uses the rule-based baseline.

## Operational safeguards

Do not send real patient data to this prototype. Keep logs free of raw
transcripts, restrict access to the API, pin dependencies, and review outputs
by a qualified clinician. The service is not approved for autonomous diagnosis
or treatment decisions.
