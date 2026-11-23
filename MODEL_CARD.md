# CareBot Healthcare AI Model Card

## Model Details

- **Project:** CareBot Healthcare AI
- **Task:** Process a clinical transcript and produce diagnosis, medication,
  dosage, and follow-up entities plus a patient-facing clinical summary.
- **Version:** 0.1.0
- **Core components:** spaCy NER, deterministic summarization, optional Groq
  API summarization/STT, Pydantic validation, and FastAPI.
- **Repository:** https://github.com/santhosh-tm-nexora/carebot-healthcare-ai

CareBot has two serving modes:

1. A rule-based entity baseline.
2. A trained spaCy NER model selected with `CAREBOT_MODEL_PATH`.

The Groq summary stage is optional. If `GROQ_API_KEY` or `GROQ_MODEL` is
missing, or a provider request fails, CareBot uses a deterministic template
summarizer.

## Intended Use

CareBot is an educational and software-integration prototype for demonstrating
medical NLP, validation, API testing, load testing, and clinician review.

### Permitted uses

- Testing pipeline behavior on synthetic transcripts.
- Demonstrating entity extraction and structured JSON generation.
- Evaluating deployment, logging, and clinician-confirmation workflows.

### Out-of-scope uses

- Diagnosis or differential diagnosis.
- Medication prescribing or dosing decisions.
- Triage or emergency decision support.
- Replacing a qualified clinician.
- Processing real patient data without an approved privacy and governance
  review.

## Data

The provided data pack is a synthetic tabular symptom/vital dataset containing
800 rows across eight balanced diagnosis classes. It is not a doctor-transcript
dataset.

The project generator creates 300 fully synthetic notes, including 30 notes in
the test split. Diagnosis, medication, dosage, and follow-up text are generated
from templates. Demographic fields in the generated notes are used only for
subgroup reporting and are not evidence of real-world population coverage.

No real patient data was used to train or evaluate the reported prototype
results.

## Evaluation

### Named-entity recognition

The test set contains 30 synthetic notes.

| System | Diagnosis F1 | Medication F1 | Dosage F1 | Follow-up F1 | JSON validity |
|---|---:|---:|---:|---:|---:|
| Rule baseline | 0.3784 | 0.8000 | 0.5897 | 1.0000 | 1.0000 |
| Trained spaCy NER | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The Week 6 experiment trained three configurations on the same synthetic
test split. All three reached 1.0000 macro F1. Configuration 1 was selected
because it used only 2 epochs and the lowest dropout (0.1).

### Summarization

The plain and entity-constrained template fallback variants both measured
1.0000 faithfulness. This comparison did not evaluate an external Groq model
because no API key or model was configured during the recorded experiment.

### Integration and load testing

The local in-process ASGI benchmark sent 50 concurrent requests:

| Successful | Throughput | Mean latency | P95 latency |
|---:|---:|---:|---:|
| 50/50 | 131.08 req/s | 0.2849 s | 0.3183 s |

These values are local benchmark measurements, not production network or
clinical performance benchmarks.

## Limitations and Risks

- Perfect synthetic-test scores may not generalize to real transcripts.
- Training and test notes share controlled template wording.
- The source diagnosis labels include hypertension rows with normal blood
  pressure and are not clinically authoritative.
- Performance on accents, jargon, negation, misspellings, unseen medications,
  and complex clinical narratives has not been established.
- Groq-generated text requires hallucination and safety review.
- The system is not calibrated for clinical decisions.
- Confirmed-note storage is out of scope for the current prototype.

## Bias and Fairness

See [BIAS_AUDIT.md](BIAS_AUDIT.md). The current audit is a limited synthetic
subgroup check, not a clinical fairness validation.

| Attribute | Groups | Entity F1 |
|---|---|---:|
| Gender | F and M | 1.0000 in both groups |
| Age band | <30, 30-49, 50-69, 70+ | 1.0000 in all groups |

## Safety and Monitoring

- Require qualified-clinician review of every output.
- Do not store raw transcripts or sensitive patient data in prototype logs.
- Log validation failures and latency without personal identifiers.
- Monitor extraction failures and provider fallback use by entity type.
- Pin model and dependency versions.
- Use least-privilege access and HTTPS outside localhost.
- Stop or escalate when transcription, extraction, or summary validation is
  uncertain.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for setup and operating instructions.

## Reproduction

```powershell
python -m pip install -e .
python scripts/run_experiments.py data/processed/train_notes.jsonl data/processed/test_notes.jsonl models/experiments
pytest -q
python scripts/load_test.py
python scripts/demo.py
```
