"""Generate deterministic synthetic clinical notes from the supplied CSV rows."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


MEDICATIONS = {
    "Influenza": ("oseltamivir", "75 mg twice daily"),
    "Hypertension": ("amlodipine", "5 mg once daily"),
    "Type2Diabetes": ("metformin", "500 mg twice daily"),
    "Pneumonia": ("amoxicillin", "500 mg three times daily"),
    "Healthy": ("", ""),
    "Anaemia": ("ferrous sulfate", "200 mg once daily"),
    "Gastroenteritis": ("omeprazole", "20 mg once daily"),
    "Migraine": ("paracetamol", "500 mg twice daily"),
}

TEMPLATES = (
    "Pt {age}{gender} c/o symptoms. Vitals: BP {systolic}/{diastolic}, "
    "SpO2 {spo2}%, glucose {glucose} mg/dL. Diagnosis: {diagnosis}. "
    "{medication_sentence} F/u in {follow_up}.",
    "Patient aged {age}, {gender}, reports symptoms. Observations are BP "
    "{systolic}/{diastolic}, SpO2 {spo2}% and glucose {glucose} mg/dL. "
    "Assessment: {diagnosis}. {medication_sentence} Review after {follow_up}.",
    "C/o symptoms in a {age}-year-old {gender}. BP {systolic}/{diastolic}; "
    "SpO2 {spo2}%. The diagnosis is {diagnosis}. {medication_sentence} "
    "Return for follow-up in {follow_up}.",
)


def load_rows(path: str | Path) -> list[dict[str, str]]:
    """Load rows and require the diagnosis column used by the generator."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Diagnosis" not in rows[0]:
        raise ValueError("CSV must contain at least one row with a Diagnosis column")
    return rows


def _span(text: str, value: str, label: str, start: int = 0) -> list[Any]:
    position = text.index(value, start)
    return [position, position + len(value), label]


def render_note(row: dict[str, str], rng: random.Random) -> dict[str, Any]:
    diagnosis = row["Diagnosis"].strip()
    medication, dosage = MEDICATIONS.get(diagnosis, ("", ""))
    medication_sentence = (
        f"Start {medication} {dosage}." if medication else "No medication is required."
    )
    follow_up = rng.choice(("1 week", "2 weeks", "1 month"))
    values = {
        "age": row.get("Age", "45"),
        "gender": row.get("Gender", "adult"),
        "systolic": row.get("BP_Systolic", "120"),
        "diastolic": row.get("BP_Diastolic", "80"),
        "spo2": row.get("SpO2", "98"),
        "glucose": row.get("Glucose_mg_dL", "90"),
        "diagnosis": diagnosis,
        "medication_sentence": medication_sentence,
        "follow_up": follow_up,
    }
    text = rng.choice(TEMPLATES).format(**values)
    entities = [_span(text, diagnosis, "DIAGNOSIS")]
    if medication:
        entities.append(_span(text, medication, "MEDICATION"))
        entities.append(_span(text, dosage, "DOSAGE"))
    entities.append(_span(text, follow_up, "FOLLOWUP"))
    return {"text": text, "entities": entities, "diagnosis": diagnosis}


def generate(
    csv_path: str | Path,
    output_dir: str | Path,
    total: int = 300,
    test_size: int = 30,
    seed: int = 0,
) -> dict[str, int]:
    """Generate train/validation/test JSONL files and return their sizes."""
    if total <= test_size or test_size < 1:
        raise ValueError("total must be greater than test_size and test_size must be positive")
    rows = load_rows(csv_path)
    rng = random.Random(seed)
    notes = [render_note(rng.choice(rows), rng) for _ in range(total)]
    rng.shuffle(notes)
    splits = {
        "test_notes.jsonl": notes[:test_size],
        "val_notes.jsonl": notes[test_size : test_size + max(1, total // 10)],
        "train_notes.jsonl": notes[test_size + max(1, total // 10) :],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, records in splits.items():
        with (destination / filename).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return {name.removesuffix(".jsonl"): len(records) for name, records in splits.items()}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
