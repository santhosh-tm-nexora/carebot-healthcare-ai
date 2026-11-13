"""Light, meaning-preserving cleanup of transcript text."""
import re

ABBREVIATIONS = {
    r"\bpt\b": "patient",
    r"\bc/o\b": "complains of",
    r"\bf/u\b": "follow-up",
    r"\bhtn\b": "hypertension",
    r"\bdm\b": "diabetes mellitus",
    r"\bbd\b": "twice daily",
    r"\bod\b": "once daily",
}


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    for pat, repl in ABBREVIATIONS.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text
