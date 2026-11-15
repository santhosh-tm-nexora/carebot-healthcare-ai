"""Reproducible five-minute demo walkthrough for the final submission."""
from __future__ import annotations

import json
import time

from carebot.pipeline import run


def main() -> None:
    transcript = (
        "Pt c/o headache for 3 days. Diagnosis: migraine. "
        "Start paracetamol 500 mg twice daily. Follow-up in 2 weeks."
    )
    print("1. Input transcript")
    print(transcript)
    print("\n2. Running preprocessing, NER, summarization, and validation...")
    started = time.perf_counter()
    result = run(transcript)
    elapsed = time.perf_counter() - started
    print(f"\n3. Structured output ({elapsed:.3f}s)")
    print(json.dumps(result.model_dump(), indent=2))
    print("\n4. API and testing evidence")
    print("Run: pytest -q")
    print("Run: python scripts/load_test.py")
    print("The project uses synthetic data only; see docs/model_card.md.")


if __name__ == "__main__":
    main()
