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


def test_conjoined_medications_and_spoken_frequency():
    out = run(
        "Take amoxicillin and paracetamol three times a day "
        "up to follow three weeks"
    )
    names = {med.name for med in out.medications}
    assert names == {"amoxicillin", "paracetamol"}
    assert all(med.dosage == "three times a day" for med in out.medications)
    assert "three weeks" in out.follow_up


def test_unknown_topical_medication_is_extracted_explicitly():
    out = run(
        "Apply luliconazole cream twice daily morning and night "
        "for two weeks."
    )
    assert out.medications
    medication = out.medications[0]
    assert medication.name == "luliconazole"
    assert medication.dosage
    assert "cream" in medication.dosage


def test_follow_up_context_variants():
    cases = (
        ("Review in two weeks.", "Review in two weeks"),
        ("Follow-up after 10 days.", "Follow-up after 10 days"),
        ("Return within one month.", "Return within one month"),
    )
    for transcript, expected in cases:
        follow_up = run(transcript).follow_up
        assert any(expected.lower() in value.lower() for value in follow_up)
