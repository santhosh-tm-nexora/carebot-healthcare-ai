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
Set `ANTHROPIC_API_KEY` to enable the LLM summary; otherwise a template summary is used.

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

## Layout
`src/carebot/` code · `data/raw/` provided data packs (place files here) · `tests/` · `docs/` · `notebooks/`

## Roadmap
| Week | Milestone |
|---|---|
| 1 | Research + architecture, repo skeleton (this commit) |
| 3 | Data pipeline + baseline (rules) evaluated on test cases |
| 6 | Custom spaCy NER trained, LLM prompt experiments |
| 9 | Integration + testing, error analysis |
| 12 | Final demo + model card |
