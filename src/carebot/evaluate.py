"""Metrics: entity-level P/R/F1 per label, and JSON validity rate."""
from collections import defaultdict
from pathlib import Path
from pydantic import ValidationError
from .schema import ClinicalNote
from .data_gen import read_jsonl
from .ner import DOSAGE_RE, FOLLOWUP_SPAN_RE


def prf(pred: set, gold: set):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def entity_scores(pred_notes: list[dict], gold_notes: list[dict]) -> dict:
    """Each note: {"diagnosis": [...], "medications": [{"name":..}], ...}. Exact match on lowercase names."""
    pred, gold = defaultdict(set), defaultdict(set)
    for i, (p, g) in enumerate(zip(pred_notes, gold_notes)):
        for key, getter in (("diagnosis", lambda x: x), ("medications", lambda x: x["name"])):
            pred[key] |= {(i, getter(x).lower()) for x in p.get(key, [])}
            gold[key] |= {(i, getter(x).lower()) for x in g.get(key, [])}
    return {k: dict(zip(("precision", "recall", "f1"), prf(pred[k], gold[k]))) for k in ("diagnosis", "medications")}


def json_validity_rate(outputs: list[dict]) -> float:
    ok = 0
    for o in outputs:
        try:
            ClinicalNote(**o)
            ok += 1
        except ValidationError:
            pass
    return ok / len(outputs) if outputs else 0.0


def span_scores(pred_notes: list[dict], gold_notes: list[dict]) -> dict[str, dict[str, float]]:
    """Score exact character spans by label."""
    pred, gold = defaultdict(set), defaultdict(set)
    for index, (pred_note, gold_note) in enumerate(zip(pred_notes, gold_notes)):
        for start, end, label in pred_note.get("entities", []):
            pred[label].add((index, start, end))
        for start, end, label in gold_note.get("entities", []):
            gold[label].add((index, start, end))
    labels = sorted(set(pred) | set(gold))
    return {label: dict(zip(("precision", "recall", "f1"), prf(pred[label], gold[label]))) for label in labels}


def evaluate_model(nlp, notes: list[dict], include_rule_regex: bool = False) -> dict:
    predictions = []
    outputs = []
    for note in notes:
        doc = nlp(note["text"])
        entities = [[e.start_char, e.end_char, e.label_] for e in doc.ents]
        if include_rule_regex:
            entities.extend(
                [m.start(), m.end(), "DOSAGE"] for m in DOSAGE_RE.finditer(note["text"])
            )
            entities.extend(
                [m.start(), m.end(), "FOLLOWUP"] for m in FOLLOWUP_SPAN_RE.finditer(note["text"])
            )
        predictions.append({"entities": entities})
        diagnosis = [e.text.lower() for e in doc.ents if e.label_ == "DIAGNOSIS"]
        medications = [
            {
                "name": e.text.lower(),
                "dosage": (
                    DOSAGE_RE.search(note["text"][e.end_char : e.end_char + 40]).group(0)
                    if include_rule_regex and DOSAGE_RE.search(note["text"][e.end_char : e.end_char + 40])
                    else None
                ),
            }
            for e in doc.ents
            if e.label_ == "MEDICATION"
        ]
        follow_up = [e.text for e in doc.ents if e.label_ == "FOLLOWUP"]
        outputs.append(
            {
                "diagnosis": diagnosis,
                "medications": medications,
                "follow_up": follow_up,
                "summary": "",
            }
        )
    return {
        "entities": span_scores(predictions, notes),
        "json_validity_rate": json_validity_rate(outputs),
    }


def evaluate_jsonl(model_path: str | Path, notes_path: str | Path) -> dict:
    import spacy
    return evaluate_model(spacy.load(model_path), read_jsonl(notes_path))
