# Week 6 Experiments

## Reproduction

```bash
set PYTHONPATH=src
python scripts/run_experiments.py data/processed/train_notes.jsonl data/processed/test_notes.jsonl models/experiments
```

The runner logs three NER configurations and two summarization variants to
the local MLflow experiment `carebot-week6`. Each NER run logs epochs, dropout,
per-label F1, macro F1, JSON validity, and its model artifact. The
summarization runs compare a plain prompt with an entity-constrained variant.

## Results

The generated [experiments_results.json](experiments_results.json) is the
source of truth for the run table below. Results use the fully synthetic test
notes and must not be interpreted as clinical performance.

| Run | Configuration / variant | Macro F1 | Faithfulness |
|---|---|---:|---:|
| NER config 1 | 2 epochs, dropout 0.1 | 1.0000 | n/a |
| NER config 2 | 3 epochs, dropout 0.2 | 1.0000 | n/a |
| NER config 3 | 4 epochs, dropout 0.4 | 1.0000 | n/a |
| Plain summary | unconstrained template fallback | n/a | 1.0000 |
| Entity-constrained summary | extracted-entity template fallback | n/a | 1.0000 |

No Anthropic or Gemini API key was available for this experiment. Therefore
the summarization comparison uses the repository's template fallback; no LLM
provider was evaluated. The perfect NER scores are limited to synthetic
generated notes.

All three NER configurations tied at 1.0000 macro F1. Config 1 is selected as
the best configuration because it achieves the tied best score with the fewest
epochs (2) and lowest dropout (0.1), reducing training cost without a
measured accuracy trade-off.
