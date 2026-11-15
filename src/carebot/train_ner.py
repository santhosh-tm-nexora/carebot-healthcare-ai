"""Train a custom spaCy NER on generated JSONL notes."""
import json
import random
import sys
from pathlib import Path
import spacy
from spacy.training import Example


def load_samples(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train(path, out_dir, epochs=30, dropout=0.2, seed=0):
    data = load_samples(path)
    random.seed(seed)
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data, dev_data = data[:split], data[split:]
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for sample in train_data:
        for _, _, label in sample["entities"]:
            ner.add_label(label)
    opt = nlp.initialize(
        get_examples=lambda: [
            Example.from_dict(nlp.make_doc(s["text"]), {"entities": s["entities"]})
            for s in train_data
        ]
    )
    for ep in range(epochs):
        random.shuffle(train_data)
        losses = {}
        for s in train_data:
            ex = Example.from_dict(nlp.make_doc(s["text"]), {"entities": s["entities"]})
            nlp.update([ex], drop=dropout, sgd=opt, losses=losses)
        print(f"epoch {ep + 1}: loss {losses.get('ner', 0.0):.2f}")
    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out_dir)
    print(f"saved to {out_dir}; held-out notes: {len(dev_data)}")


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
