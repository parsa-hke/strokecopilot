"""notes/generate.py on synthetic clinical rows (no patient data)."""

import json

import pandas as pd
import pytest

from strokecopilot.data import clinical_columns
from strokecopilot.nlp.notes import generate as g
from strokecopilot.nlp.verifier import rules

SENTINELS = {"Door to groin": "07:41:00", "Door to first series": "08:13:00", "Time of intervention": "09:27:00",
             "Door to recanalization": "10:59:00", "TICI postinterventional": "2c", "NIHSS 24h": "37",
             "mRS 24h": "5", "NIHSS discharge": "33", "mRS discharge": "4", "mRS 3 months": "5", "Center": "2",
             "mRS at admission": "5"}


def row(**over):
    base = {"Sex": "F", "Age": "77.6", "Atrial fibrillation": "1", "Hypertension": "1", "Diabetes": "0",
            "Hyperlipidemia": "99", "Anticoagulation": "0", "Lipid lowering drugs": "1", "PAIs": "1",
            "Glucose": "133.0", "Leucocytes": "8.1", "CRP": "0.4", "INR": "1.04", "Wake-up": "0", "In-House": "0",
            "Referral": "0", "Onset to door": "01:12:00.431000", "Alert to door": None, "NIHSS at admission": "14",
            "mRS premorbid": "1", "Door to imaging": "00:23:00", **SENTINELS}
    base.update(over)
    return base


def test_every_workbook_column_is_tagged():
    for col in clinical_columns.BASELINE_COLUMNS + clinical_columns.OUTCOME_COLUMNS:
        clinical_columns.tag_of(col)


def test_untagged_column_raises():
    with pytest.raises(KeyError):
        g.allowed_fields(row(**{"Thrombolysis given": "1"}))


def test_nothing_forbidden_reaches_the_note():
    assert not set(g.allowed_fields(row())) & set(clinical_columns.FORBIDDEN_FOR_NOTES)
    note, sheet = g.note_for("case-a", row())
    text = note + json.dumps([vars(f) for f in sheet])
    for value in ("07:41", "08:13", "09:27", "10:59", "2c", "37", "33"):  # sentinels unique to forbidden columns
        assert value not in text, value
    for word in ("groin", "tici", "recanal", "angiograph", "thrombectomy", "discharge", "24 h", "centre", "center"):
        assert word not in note.lower(), word


def test_known_onset_note_and_facts():
    note, sheet = g.note_for("case-a", row())
    assert sheet.get("age").value == 77 and sheet.get("sex").value == "female"
    assert sheet.get("onset_to_imaging").value == 95  # 72 min 26 s + 23 min
    assert sheet.get("nihss").value == 14 and sheet.get("premorbid_mrs").value == 1
    assert sheet.get("anticoagulant").value is False and sheet.get("antiplatelet").value is True
    assert sheet.get("glucose").value == 7.4 and sheet.get("glucose").unit == "mmol/L"
    assert sheet.get("inr").value == 1.04
    assert sheet.get("atrial_fibrillation").value is True and sheet.get("diabetes").value is False
    assert sheet.get("hyperlipidaemia") is None  # 99 = unknown: the note is silent
    assert "1 h 35 min" in note
    sheet.check_against(note)


def test_missing_rule_inputs_are_not_recorded():
    note, sheet = g.note_for("case-b", row(**{"Onset to door": None, "INR": None, "mRS premorbid": None,
                                              "NIHSS at admission": None, "Anticoagulation": None}))
    for key in ("onset_to_imaging", "inr", "premorbid_mrs", "nihss", "anticoagulant"):
        assert sheet.get(key).type == "not_recorded", key
    assert "Time of symptom onset: not recorded." in note


def test_wake_up_has_no_last_known_well():
    note, sheet = g.note_for("case-c", row(**{"Wake-up": "1", "Onset to door": None, "Alert to door": "02:00:00"}))
    assert sheet.get("wake_up_stroke").value is True and sheet.get("onset_witnessed").value is False
    assert sheet.get("lkw_to_imaging").type == "not_recorded"
    assert sheet.get("found_to_imaging").value == 143
    assert sheet.get("onset_to_imaging") is None


def test_in_house_and_referral():
    _, sheet = g.note_for("case-d", row(**{"In-House": "1"}))
    assert sheet.get("referral").value == "in-hospital stroke"
    _, sheet = g.note_for("case-d", row(**{"Referral": "1"}))
    assert sheet.get("referral").value == "transfer from an external clinic"


def test_onset_without_door_to_imaging():
    _, sheet = g.note_for("case-e", row(**{"Door to imaging": None}))
    assert sheet.get("onset_to_door").value == 72 and sheet.get("door_to_imaging").type == "not_recorded"
    assert sheet.get("onset_to_imaging") is None


def test_durations():
    assert g.minutes("01:12:00") == 72 and g.minutes("1 day, 02:00:00") == 1560 and g.minutes(None) is None
    assert round(g.minutes("00:10:30.250000"), 2) == 10.5
    with pytest.raises(ValueError):
        g.minutes("about an hour")


def test_same_note_every_run_and_some_variation():
    assert g.note_for("case-a", row()) [0] == g.note_for("case-a", row())[0]
    texts = {g.note_for(f"case-{i}", row())[0] for i in range(30)}
    assert len(texts) > 1


def test_rulebook_reads_generated_facts():
    vision = {"haemorrhage": {"present": False, "confidence": 0.05}, "core_ml": 12.0, "penumbra_ml": 95.0,
              "mismatch_ratio": 7.9, "mismatch_volume_ml": 83.0, "occlusion_site": "M1", "aspects": None}
    _, sheet = g.note_for("case-a", row())
    case = rules.case_from(vision, sheet)
    assert case.minutes == 95 and case.time_basis == "onset_to_imaging" and case.nihss == 14 and case.mrs == 1
    assert case.age == 77 and case.anticoagulant is False and case.inr == 1.04
    verdicts = rules.decide(case)
    assert verdicts["thrombolysis"].value == rules.YES


def test_generate_writes_files_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(g.splits, "check_ids", lambda cases, folds=None: None)
    pd.DataFrame([{"case": "c1", **row()}, {"case": "c2", **row(**{"INR": None})}]).to_csv(tmp_path / "clinical.csv", index=False)
    counts = g.generate(tmp_path, ["c1", "c2"], tmp_path / "notes", log=lambda m: None)
    assert counts["notes"] == 2 and counts["not_recorded"]["inr"] == 1 and counts["onset_known"] == 2
    assert (tmp_path / "notes" / "c1.txt").read_text().startswith("ED STROKE TRIAGE NOTE")
    facts = json.loads((tmp_path / "notes" / "c2.facts.json").read_text())["facts"]
    assert any(f["key"] == "inr" and f["type"] == "not_recorded" for f in facts)
    assert "c1" not in json.dumps(json.loads((tmp_path / "notes" / "manifest.json").read_text()))


def test_generate_refuses_heldout(tmp_path):
    from strokecopilot.data import splits
    folds = {"heldout": ["h1"], "folds": {"0": ["c1"]}}
    with pytest.raises(Exception):
        g.generate(tmp_path, ["h1"], tmp_path / "notes", folds=folds, log=lambda m: None)


def test_main_reads_only_clinical_from_the_tarball(tmp_path, monkeypatch, capsys):
    import hashlib
    import tarfile
    slim = tmp_path / "isles24_slim_v1"
    slim.mkdir()
    pd.DataFrame([{"case": "c1", **row()}, {"case": "h1", **row()}]).to_csv(slim / "clinical.csv", index=False)
    (slim / "manifest.json").write_text(json.dumps({"cases": ["c1", "h1"]}))
    (slim / "c1").mkdir()
    (slim / "c1" / "ncct.nii.gz").write_bytes(b"image")
    tar = tmp_path / "slim.tar"
    with tarfile.open(tar, "w") as t:
        t.add(slim, arcname="isles24_slim_v1")
    folds = {"heldout": ["h1"], "folds": {"0": ["c1"]}}
    monkeypatch.setattr(g, "SLIM_SHA256", hashlib.sha256(tar.read_bytes()).hexdigest())
    monkeypatch.setattr(g, "EXPECTED_DEVELOPMENT", 1)
    monkeypatch.setattr(g.splits, "load_folds", lambda: folds)
    monkeypatch.setattr(g.splits, "development_ids", lambda f=None: ["c1"])
    monkeypatch.setattr(g.splits, "check_ids", lambda cases, f=None: None)
    monkeypatch.setattr(g.splits, "heldout_allowed", lambda: False)
    g.main(["--tar", str(tar), "--out", str(tmp_path / "out")])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["c1.facts.json", "c1.txt", "manifest.json"]
    assert "c1" not in capsys.readouterr().out
