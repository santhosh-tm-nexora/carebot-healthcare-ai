# CareBot Healthcare AI: Clinical Note Summarization

Pipeline: doctor voice-transcript text -> structured JSON (diagnosis, medications, follow-up) + summary paragraph.
**No real patient data is used**, only the provided synthetic data packs.

## Architecture
```
transcript -> preprocess -> NER (spaCy; rules now, custom model later) -> LLM summary (entity-constrained) -> Pydantic-validated JSON
```

## Quick start
```bash
pip install -r requirements.txt
export PYTHONPATH=src
echo "Pt c/o fever. Diagnosis: viral fever. Paracetamol 500 mg twice daily. Follow-up in 1 week." | python -m carebot.pipeline
pytest
```
No LLM provider was evaluated in the recorded experiments because no API key
was configured. The pipeline therefore uses its deterministic template
summarizer. Gemini support is not enabled by this repository version.

## Week 3 data pipeline

Copy the supplied synthetic CSV to `data/raw/aiml_healthcare_symptoms.csv`.
It contains tabular symptom/vital rows, not clinical transcripts. Generate
fully synthetic notes and train the baseline NER model:

```bash
set PYTHONPATH=src
python scripts/generate_data.py data/raw/aiml_healthcare_symptoms.csv data/processed
python -m carebot.train_ner data/processed/train_notes.jsonl models/ner
python scripts/run_eval.py models/ner data/processed/test_notes.jsonl
```

The generator creates exactly 30 test notes. Medication, dosage, and
follow-up entities come from CareBot's synthetic templates, not from the
provided data. See [docs/baseline_report.md](docs/baseline_report.md) for
limitations and the reproducible results table.

## Weeks 6, 9, and 12

Week 6 trains three NER configurations and logs parameters, per-label F1,
macro F1, JSON validity, and model artifacts to MLflow. It also compares plain
and entity-constrained template summarization:

```bash
set PYTHONPATH=src
python scripts/run_experiments.py data/processed/train_notes.jsonl data/processed/test_notes.jsonl models/experiments
```

Week 9 provides the FastAPI service, integration tests, and a 50-request
concurrent benchmark:

```bash
set PYTHONPATH=src
uvicorn carebot.api:app --host 127.0.0.1 --port 8000
pytest -q
python scripts/load_test.py
```

Week 12 provides the model card, bias audit, deployment guide, reproducible
demo, and GitHub Actions CI:

```bash
set PYTHONPATH=src
python scripts/demo.py
```

## Layout
`src/carebot/` code · `data/raw/` provided data packs · `data/processed/` synthetic notes · `tests/` · `docs/` · `.github/workflows/ci.yml`

## Roadmap
| Week | Milestone |
|---|---|
| 1 | Research + architecture, repo skeleton (this commit) |
| 3 | Data pipeline + baseline (rules) evaluated on test cases |
| 6 | Three MLflow NER configurations and summarization experiment |
| 9 | FastAPI integration, tests, and 50-request load test |
| 12 | Model card, bias audit, deployment guide, demo script, and CI |
