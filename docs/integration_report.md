# Week 9 Integration and Testing

## Serving layer

CareBot exposes a FastAPI application with:

- `GET /health` for service readiness
- `POST /summarize` accepting a transcript and returning validated
  `ClinicalNote` JSON
- Pydantic input validation returning HTTP 422 for an empty or oversized
  transcript
- One NER model loaded when `carebot.api` starts. Set
  `CAREBOT_MODEL_PATH` to select a trained spaCy model; the rule baseline is
  used when the ignored local model artifact is not present.

Start the service with:

```bash
set PYTHONPATH=src
uvicorn carebot.api:app --host 127.0.0.1 --port 8000
```

## Integration and load testing

```bash
set PYTHONPATH=src
pytest -q
python scripts/load_test.py
```

The load test sends 50 concurrent requests through the ASGI application and
reports successful requests, wall time, throughput, mean latency, and p95
latency. The measured output is recorded below from the local run.

| Requests | Concurrency | Successful | Throughput (req/s) | Mean latency (s) | P95 latency (s) |
|---:|---:|---:|---:|---:|---:|
| 50 | 50 | 50 | 131.08 | 0.2849 | 0.3183 |

These are local in-process measurements, not production network or clinical
deployment benchmarks.
