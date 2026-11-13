from carebot.pipeline import run
from carebot.evaluate import json_validity_rate, entity_scores

NOTE = ("Pt c/o headache for 3 days. Diagnosis: migraine. Start paracetamol 500 mg twice daily. "
        "Follow-up in 2 weeks.")


def test_extraction():
    out = run(NOTE)
    assert "migraine" in out.diagnosis
    assert out.medications[0].name == "paracetamol"
    assert out.medications[0].dosage.startswith("500 mg")
    assert out.follow_up and out.summary


def test_validity_and_scores():
    out = run(NOTE).model_dump()
    assert json_validity_rate([out]) == 1.0
    s = entity_scores([out], [out])
    assert s["diagnosis"]["f1"] == 1.0
