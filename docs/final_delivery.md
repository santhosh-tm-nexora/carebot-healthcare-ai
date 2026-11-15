# Week 12 Final Delivery

## Included deliverables

- [Model card](model_card.md), including data, metrics, limitations, bias
  audit, and safety guidance
- [Deployment guide](deployment_guide.md)
- Reproducible [demo script](../scripts/demo.py)
- FastAPI service, integration tests, and Week 9 load-test evidence
- MLflow experiment results and Week 6 evidence
- GitHub Actions CI running the test suite on pushes and pull requests

## Final validation

Run:

```bash
set PYTHONPATH=src
pytest -q
python scripts/demo.py
python scripts/load_test.py
```

The final demo should show the transcript, structured JSON, summary, API
health response, and the recorded test/load-test outputs. Record the demo
screen and submit the resulting video separately where the milestone form
requests it.

All reported metrics are from synthetic data or local in-process testing.
They are not clinical validation results.
