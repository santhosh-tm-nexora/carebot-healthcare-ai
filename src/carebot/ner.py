"""NER stage.

Baseline: rule-based spaCy EntityRuler (works with no training data).
Upgrade path: train a custom model with train_ner.py, then pass its path to load_nlp().
"""
import re
import spacy

DIAGNOSES = ["hypertension", "diabetes mellitus", "type 2 diabetes", "asthma", "migraine",
             "viral fever", "gastritis", "anemia", "bronchitis", "urinary tract infection"]
MEDICATIONS = ["metformin", "amlodipine", "paracetamol", "azithromycin", "omeprazole",
               "salbutamol", "ibuprofen", "amoxicillin", "cetirizine", "atorvastatin"]

FOLLOWUP_RE = re.compile(r"[^.]*\b(follow-up|f/u|review|return|come back|revisit)\b[^.]*\.?", re.I)
FOLLOWUP_SPAN_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:day|days|week|weeks|month|months)\b(?:\s+(?:continuously|only))?",
    re.I,
)
FOLLOWUP_CONTEXT_RE = re.compile(
    r"\b(?:follow[- ]?up|f/u|review|return|come back|revisit)"
    r"(?:\s+(?:in|after|within|for))?"
    r"\s*(?:of\s+)?"
    r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:day|days|week|weeks|month|months)"
    r"|(?:tomorrow|next\s+(?:week|month)|tonight))\b",
    re.I,
)
DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?|tablets?|capsules?|puffs?|drops?|%)"
    r"(?:\s+(?:once|twice|thrice)\s+(?:daily|a\s+day))?",
    re.I,
)
TOPICAL_MEDICATION_RE = re.compile(
    r"\b([a-z][a-z0-9-]{2,})(?:\s+cream|\s+ointment|\s+gel|\s+solution|\s+lotion)",
    re.I,
)
SPOKEN_SCHEDULE_RE = re.compile(
    r"(?:once|twice|thrice|three\s+times|three times)\s+(?:daily|a\s+day|per\s+day)"
    r"(?:\s+(?:morning|night|evening)|\s+morning\s+and\s+night)?",
    re.I,
)
WRITTEN_FREQUENCY_RE = re.compile(
    r"\b(?:three\s+times|thrice|twice|once)\s+(?:a\s+day|daily|per\s+day)\b",
    re.I,
)
GENERIC_FORM_RE = re.compile(
    r"\b([a-z][a-z0-9-]{2,})\s+(cream|ointment|gel|lotion|solution|spray|drops|inhaler)\b",
    re.I,
)
PRESCRIPTION_CONTEXT_RE = re.compile(
    r"\b(?:take|start|prescribe(?:d)?|use|apply)\s+"
    r"(?:a\s+|an\s+|the\s+)?([a-z][a-z0-9-]{2,})"
    r"(?:\s+(cream|ointment|gel|lotion|solution|spray|drops|inhaler|tablet|tablets|capsule|capsules))?",
    re.I,
)
DOSAGE_FOLLOWING_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?|tablets?|capsules?|puffs?|drops?|%))"
    r"(?:\s+(?:once|twice|thrice|three\s+times)\s+(?:daily|a\s+day|per\s+day))?",
    re.I,
)


def load_nlp(model_path: str | None = None):
    if model_path:
        return spacy.load(model_path)
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(
        [{"label": "DIAGNOSIS", "pattern": [{"LOWER": w} for w in d.split()]} for d in DIAGNOSES]
        + [{"label": "MEDICATION", "pattern": [{"LOWER": m}]} for m in MEDICATIONS]
    )
    return nlp


def extract(nlp, text: str) -> dict:
    doc = nlp(text)
    diagnosis, meds = [], []
    seen_medications = set()
    for ent in doc.ents:
        if ent.label_ == "DIAGNOSIS" and ent.text.lower() not in diagnosis:
            diagnosis.append(ent.text.lower())
        elif ent.label_ == "MEDICATION":
            tail = text[ent.end_char: ent.end_char + 40]
            m = DOSAGE_RE.search(tail)
            schedule = SPOKEN_SCHEDULE_RE.search(tail)
            dosage = m.group(0) if m else schedule.group(0) if schedule else None
            name = ent.text.lower()
            seen_medications.add(name)
            meds.append({"name": name, "dosage": dosage})

    lower_text = text.lower()
    shared_schedule = WRITTEN_FREQUENCY_RE.search(lower_text)
    shared_dosage = shared_schedule.group(0).strip() if shared_schedule else None
    for medication in MEDICATIONS:
        if re.search(rf"\b{re.escape(medication)}\b", lower_text):
            if medication not in seen_medications:
                seen_medications.add(medication)
                meds.append({"name": medication, "dosage": shared_dosage})

    # Unknown medications are accepted only when they are explicitly introduced
    # with a prescription context or a recognizable pharmaceutical form.
    for pattern in (GENERIC_FORM_RE, PRESCRIPTION_CONTEXT_RE):
        for match in pattern.finditer(text):
            name = match.group(1).lower()
            if name in seen_medications or name in {"follow", "patient", "doctor"}:
                continue
            form = None
            if pattern is GENERIC_FORM_RE:
                form = match.group(2).lower()
            elif len(match.groups()) > 1 and match.group(2):
                form = match.group(2).lower()

            tail = text[match.end(): match.end() + 60]
            numeric_dose = DOSAGE_FOLLOWING_RE.search(tail)
            spoken_schedule = WRITTEN_FREQUENCY_RE.search(tail) or WRITTEN_FREQUENCY_RE.search(
                text[max(0, match.start() - 20): match.end() + 80]
            )
            dosage_parts = []
            if numeric_dose:
                dosage_parts.append(numeric_dose.group(0).strip())
            if spoken_schedule:
                dosage_parts.append(spoken_schedule.group(0).strip())
            if form:
                dosage_parts.insert(0, form)

            seen_medications.add(name)
            meds.append({
                "name": name,
                "dosage": ", ".join(dosage_parts) if dosage_parts else None,
            })

    for match in TOPICAL_MEDICATION_RE.finditer(text):
        name = match.group(1).lower()
        if name in seen_medications:
            continue
        tail = text[match.end(): match.end() + 50]
        schedule = SPOKEN_SCHEDULE_RE.search(tail)
        dosage = (match.group(0).split()[1] if len(match.group(0).split()) > 1 else None)
        if schedule:
            dosage = f"{match.group(0).split()[-1]}, {schedule.group(0)}"
        seen_medications.add(name)
        meds.append({"name": name, "dosage": dosage})

    follow_up = []
    for match in FOLLOWUP_RE.finditer(text):
        value = match.group(0).strip()
        if value and value not in follow_up:
            follow_up.append(value)
    for match in FOLLOWUP_CONTEXT_RE.finditer(text):
        value = match.group(0).strip()
        if value not in follow_up:
            follow_up.append(value)
    for match in FOLLOWUP_SPAN_RE.finditer(text):
        value = match.group(0).strip()
        if value not in follow_up:
            follow_up.append(value)
    return {"diagnosis": diagnosis, "medications": meds, "follow_up": follow_up}
