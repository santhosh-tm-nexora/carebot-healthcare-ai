# CareBot Healthcare AI Model Card

## Model details

CareBot is a prototype clinical-note processing pipeline. It normalizes a
transcript, extracts diagnosis, medication, dosage, and follow-up entities
with spaCy, and returns validated `ClinicalNote` JSON with a short summary.
The serving layer is FastAPI.

Groq is the LLM provider for the entity-constrained summary stage. The
provider is configured with the `GROQ_API_KEY` and `GROQ_MODEL` environment
variables; missing configuration or request failures use the deterministic
template summary instead.

## Intended use

This project is for internship demonstration, software evaluation, and
research on synthetic clinical NLP. It is not a diagnostic tool, treatment
recommendation system, or replacement for a qualified clinician.

## Data

The supplied dataset is a synthetic tabular symptom/vital dataset with 800
rows and eight balanced diagnosis labels. It contains no doctor transcripts,
medications, or follow-up fields. CareBot therefore generates approximately
300 synthetic notes from deterministic templates; medication, dosage, and
follow-up values are template-generated. No real patient data is used.

## Evaluation

On the 30-note synthetic test split:

| Metric | Rule baseline | Trained spaCy NER |
|---|---:|---:|
| Diagnosis F1 | 0.3784 | 1.0000 |
| Medication F1 | 0.8000 | 1.0000 |
| Dosage F1 | 0.5897 | 1.0000 |
| Follow-up F1 | 1.0000 | 1.0000 |
| JSON validity | 1.0000 | 1.0000 |

The Week 6 experiments also measured 1.0000 macro F1 for each of three
configurations on this same synthetic split. Week 9's local in-process
benchmark handled 50 concurrent requests with 131.08 requests/second
throughput, 0.2849 seconds mean latency, and 0.3183 seconds p95 latency.

## Limitations and risks

The perfect trained-model score reflects templated train/test data and does
not demonstrate clinical generalization. The source labels include
hypertension rows with normal blood pressure, so they should not be treated as
clinically authoritative. The pipeline has not been validated on real
transcripts, accents, clinical abbreviations beyond the implemented rules,
negation, or unseen medication names.

## Bias and safety audit

The trained NER was evaluated on the 30-note test split after grouping notes
by the demographic attributes embedded in each generated transcript. Entity
F1 is the macro average across `DIAGNOSIS`, `MEDICATION`, `DOSAGE`, and
`FOLLOWUP`.

| Attribute | Group | Notes | Entity F1 |
|---|---|---:|---:|
| Gender | F | 14 | 1.0000 |
| Gender | M | 16 | 1.0000 |
| Age band | <30 | 5 | 1.0000 |
| Age band | 30-49 | 7 | 1.0000 |
| Age band | 50-69 | 15 | 1.0000 |
| Age band | 70+ | 3 | 1.0000 |

All groups score 1.0000 in this run, which is expected because the notes are
template-generated and the entity spans follow the same controlled phrasing
used for training. This is not evidence of fairness or performance on real
populations. The source diagnosis classes are balanced at 100 rows each, but
gender, age, and other feature distributions were not audited against a real
clinical population. Before any clinical use, the system would require
de-identified representative notes, clinician review, subgroup precision and
recall analysis, calibration, privacy controls, and a human approval workflow.

## Deployment and monitoring

See [deployment_guide.md](deployment_guide.md). Operators should log request
latency and validation failures without storing identifiable transcripts,
monitor extraction failures by entity type, pin model versions, and provide a
clinician escalation path.
