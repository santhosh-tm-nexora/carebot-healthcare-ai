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
FOLLOWUP_SPAN_RE = re.compile(r"\b\d+\s+(?:day|days|week|weeks|month|months)\b", re.I)
DOSAGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml)\b(?:\s(?:once|twice|thrice)\sdaily)?", re.I)


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
    for ent in doc.ents:
        if ent.label_ == "DIAGNOSIS" and ent.text.lower() not in diagnosis:
            diagnosis.append(ent.text.lower())
        elif ent.label_ == "MEDICATION":
            tail = text[ent.end_char: ent.end_char + 40]
            m = DOSAGE_RE.search(tail)
            meds.append({"name": ent.text.lower(), "dosage": m.group(0) if m else None})
    follow_up = [m.group(0).strip() for m in FOLLOWUP_RE.finditer(text)]
    return {"diagnosis": diagnosis, "medications": meds, "follow_up": follow_up}
