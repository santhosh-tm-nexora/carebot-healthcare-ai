# Week 3 Baseline Report

## Status

The Week 3 pipeline is implemented, but this report is intentionally not
filled with invented scores. The supplied CSV was used from
`C:\Users\santh\Downloads\aiml_healthcare_symptoms (1).csv`.

## Reproducible run

```bash
set PYTHONPATH=src
python scripts/generate_data.py data/raw/aiml_healthcare_symptoms.csv data/processed
python -m carebot.train_ner data/processed/train_notes.jsonl models/ner 10
python scripts/run_eval.py models/ner data/processed/test_notes.jsonl
```

The generator created 300 fully synthetic notes, with exactly 30 notes in
`test_notes.jsonl`. Character spans are generated from the rendered text and
cover diagnosis, medication, dosage, and follow-up. The trained model was run
for 10 epochs.

## Data and limitations

The provided CSV contains synthetic tabular symptom/vital rows, not doctor
transcripts. Medications, dosages, and follow-up text are generated from
CareBot templates and are not present in the source data. The source labels
also include hypertension rows with normal blood pressure, so model results
must not be interpreted as clinical performance.

## Results

Populate the rule-baseline and trained-NER tables from the command output
after the raw CSV is available. The command records per-label precision,
recall, F1, and JSON validity rate for both systems.

| System | DIAGNOSIS F1 | MEDICATION F1 | DOSAGE F1 | FOLLOWUP F1 | JSON validity |
|---|---:|---:|---:|---:|---:|
| Rule baseline | 0.3784 | 0.8000 | 0.5897 | 0.0000 | 1.0000 |
| Trained spaCy NER | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Example outputs and failure analysis

Examples from the generated test set include Type2Diabetes with metformin,
Anaemia with ferrous sulfate, and Healthy with no medication. The rule
baseline misses diagnosis variants not in its starter vocabulary and can
mistake glucose units for medication dosage. The trained model achieved
perfect scores on this synthetic test split, which is not evidence of clinical
generalization.
