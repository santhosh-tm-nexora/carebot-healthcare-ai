from carebot.data_gen import render_note
from carebot.evaluate import span_scores


def test_generated_spans_align():
    note = render_note(
        {
            "Age": "42",
            "Gender": "F",
            "BP_Systolic": "120",
            "BP_Diastolic": "80",
            "SpO2": "98",
            "Glucose_mg_dL": "90",
            "Diagnosis": "Migraine",
        },
        __import__("random").Random(0),
    )
    for start, end, label in note["entities"]:
        assert note["text"][start:end]
        assert label in {"DIAGNOSIS", "MEDICATION", "DOSAGE", "FOLLOWUP"}


def test_span_scores_match_exact_spans():
    gold = [{"entities": [[0, 7, "DIAGNOSIS"]]}]
    assert span_scores(gold, gold)["DIAGNOSIS"]["f1"] == 1.0
