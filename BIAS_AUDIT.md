# CareBot Healthcare AI Bias Audit

## Scope

This audit evaluates whether the extracted synthetic test notes show subgroup
differences in entity-F1 performance. It covers gender and age bands encoded
in generated transcripts.

This is a documentation and engineering audit, not a clinical bias validation.
No real patients, protected demographic data, or representative clinical
population is used.

## Data and Method

The source data is synthetic tabular symptom/vital data containing 800 rows
and eight diagnosis labels, with 100 rows per label. CareBot generated 300
fully synthetic notes and used 30 notes for testing.

Generated notes are divided into demographic subgroups and evaluated with the
trained spaCy NER model. Entity F1 is macro-averaged across:

- `DIAGNOSIS`
- `MEDICATION`
- `DOSAGE`
- `FOLLOWUP`

## Synthetic Subgroup Results

### Gender

| Group | Notes | Entity F1 |
|---|---:|---:|
| F | 14 | 1.0000 |
| M | 16 | 1.0000 |

### Age band

| Group | Notes | Entity F1 |
|---|---:|---:|
| <30 | 5 | 1.0000 |
| 30-49 | 7 | 1.0000 |
| 50-69 | 15 | 1.0000 |
| 70+ | 3 | 1.0000 |

## Findings

No measured difference was observed between the tested synthetic subgroups.
All subgroup scores were 1.0000.

This result is an expected artifact of template-generated notes: the entity
phrasing and span positions follow controlled patterns shared across generated
demographic groups. It does not establish fairness or equal performance for:

- Real populations.
- Different dialects or languages.
- Different writing styles.
- Clinician and patient speech patterns.
- Real-world medication names and abbreviations.
- Complex or ambiguous clinical narratives.
- Historically underrepresented communities.

## Data Limitations

- The supplied CSV is tabular and does not contain clinical transcripts.
- Medication, dosage, and follow-up values are template-generated.
- Gender and age distributions are not representative of a real clinical
  population.
- The source diagnosis labels include clinically implausible entries.
- Small subgroup sizes produce unstable estimates.
- Accuracy alone does not measure harm, safety, or downstream clinical impact.

## Metrics That Should Be Added Before Clinical Evaluation

1. Per-class and per-entity precision, recall, F1, and confidence intervals.
2. False-negative rates for high-risk diagnoses and medications.
3. Calibration and abstention analysis.
4. Transcript language, accent, and readability coverage.
5. Race, ethnicity, disability, socioeconomic status, and language where
   legally and ethically collected.
6. User-experience errors and clinician override rates.
7. Model behavior across multiple real, de-identified clinical sites.

## Risk Controls

- Do not deploy as autonomous or unsupervised clinical software.
- Require clinician review and correction.
- Do not infer sensitive attributes that are not explicitly and lawfully
  available.
- Remove or minimize raw transcripts and identifiers.
- Use role-based access and audit logs.
- Track fallback-to-template rates and provider failures.
- Re-run subgroup evaluation after model, prompt, or data changes.

## Re-test Procedure

Use a legally and ethically approved, representative, de-identified evaluation
set. Split it by intended clinical population and report:

```text
precision, recall, F1, false negatives, calibration, JSON validity,
summary faithfulness, subgroup confidence intervals
```
Require clinician review and independent statistical analysis before deciding
whether performance differences are acceptable.
