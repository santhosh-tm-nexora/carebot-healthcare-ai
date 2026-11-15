"""Week 6 MLflow experiments for NER and summarization."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import mlflow
import spacy

from .data_gen import read_jsonl
from .evaluate import evaluate_model
from .ner import load_nlp
from .summarize import template_summary
from .train_ner import train


def _terms(note: dict) -> set[str]:
    terms = {note["diagnosis"].lower()}
    for start, end, label in note["entities"]:
        if label in {"MEDICATION", "DOSAGE", "DIAGNOSIS"}:
            terms.add(note["text"][start:end].lower())
    return terms


def faithfulness(summary: str, note: dict) -> bool:
    """Return false when a clinical term in the summary is absent from the source."""
    clinical_terms = re.findall(
        r"\b(?:metformin|amlodipine|oseltamivir|amoxicillin|ferrous sulfate|"
        r"omeprazole|paracetamol|\d+\s*(?:mg|mcg|g|ml)|"
        r"influenza|hypertension|type2diabetes|pneumonia|anaemia|"
        r"gastroenteritis|migraine|healthy)\b",
        summary.lower(),
    )
    source = note["text"].lower()
    return all(term in source for term in clinical_terms)


def summarize_variant(note: dict, constrained: bool) -> str:
    entities = {
        "diagnosis": [note["diagnosis"]],
        "medications": [
            {"name": note["text"][start:end], "dosage": None}
            for start, end, label in note["entities"]
            if label == "MEDICATION"
        ],
        "follow_up": [],
    }
    if constrained:
        return template_summary(entities)
    return f"Clinical note: {note['text']}"


def run_ner_experiments(train_path: str | Path, test_path: str | Path, output_dir: str | Path) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    notes = read_jsonl(test_path)
    results = []
    for index, config in enumerate(
        (
            {"epochs": 2, "dropout": 0.1},
            {"epochs": 3, "dropout": 0.2},
            {"epochs": 4, "dropout": 0.4},
        ),
        start=1,
    ):
        with mlflow.start_run(run_name=f"ner_config_{index}"):
            model_path = output / f"ner_config_{index}"
            train(train_path, model_path, **config, seed=0)
            scores = evaluate_model(spacy.load(model_path), notes)
            mlflow.log_params(config)
            mlflow.log_metric("macro_f1", sum(v["f1"] for v in scores["entities"].values()) / 4)
            for label, metrics in scores["entities"].items():
                mlflow.log_metric(f"{label.lower()}_f1", metrics["f1"])
            mlflow.log_metric("json_validity_rate", scores["json_validity_rate"])
            mlflow.log_artifacts(str(model_path), artifact_path="model")
            results.append({"run": f"ner_config_{index}", **config, **scores})
    return results


def run_summary_experiments(test_path: str | Path) -> list[dict]:
    notes = read_jsonl(test_path)
    results = []
    for constrained in (False, True):
        name = "summary_entity_constrained" if constrained else "summary_plain"
        with mlflow.start_run(run_name=name):
            faithful = [
                faithfulness(summarize_variant(note, constrained), note) for note in notes
            ]
            rate = sum(faithful) / len(faithful)
            mlflow.log_param("prompt_variant", name)
            mlflow.log_param("summarizer", "template_fallback")
            mlflow.log_metric("faithfulness_rate", rate)
            results.append({"run": name, "faithfulness_rate": rate, "summarizer": "template_fallback"})
    return results


def run(train_path: str | Path, test_path: str | Path, output_dir: str | Path) -> dict:
    mlflow.set_experiment("carebot-week6")
    with tempfile.TemporaryDirectory():
        return {
            "ner": run_ner_experiments(train_path, test_path, output_dir),
            "summarization": run_summary_experiments(test_path),
        }
