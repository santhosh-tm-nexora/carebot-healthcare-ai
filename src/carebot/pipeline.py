"""End-to-end: transcript text -> validated ClinicalNote JSON."""
import sys
from .preprocess import normalize
from .ner import load_nlp, extract
from .summarize import summarize
from .schema import ClinicalNote


def run(text: str, nlp=None) -> ClinicalNote:
    nlp = nlp or load_nlp()
    clean = normalize(text)
    ents = extract(nlp, clean)
    return ClinicalNote(**ents, summary=summarize(clean, ents))


if __name__ == "__main__":
    print(run(sys.stdin.read()).model_dump_json(indent=2))
