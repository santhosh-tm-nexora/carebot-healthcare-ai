"""Evaluate a trained model: python scripts/run_eval.py models/ner data/processed/test_notes.jsonl."""
import json
import sys

import spacy

from carebot.data_gen import read_jsonl
from carebot.evaluate import evaluate_model
from carebot.ner import load_nlp


if __name__ == "__main__":
    notes = read_jsonl(sys.argv[2])
    print(json.dumps({
        "rule_baseline": evaluate_model(load_nlp(), notes, include_rule_regex=True),
        "trained_ner": evaluate_model(spacy.load(sys.argv[1]), notes),
    }, indent=2))
