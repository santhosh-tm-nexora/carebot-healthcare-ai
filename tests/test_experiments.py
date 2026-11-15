from carebot.experiments import faithfulness


def test_faithfulness_rejects_unseen_medication():
    note = {
        "text": "Diagnosis: Migraine. Start paracetamol 500 mg.",
        "diagnosis": "Migraine",
        "entities": [[11, 18, "DIAGNOSIS"], [26, 36, "MEDICATION"], [37, 44, "DOSAGE"]],
    }
    assert faithfulness("Diagnosis: Migraine. Take amoxicillin 500 mg.", note) is False


def test_faithfulness_accepts_source_terms():
    note = {
        "text": "Diagnosis: Migraine. Start paracetamol 500 mg.",
        "diagnosis": "Migraine",
        "entities": [[11, 18, "DIAGNOSIS"], [26, 36, "MEDICATION"], [37, 44, "DOSAGE"]],
    }
    assert faithfulness("Diagnosis: Migraine. Take paracetamol 500 mg.", note) is True
