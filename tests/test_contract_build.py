"""contract/build.py on synthetic run tables (placeholder IDs, no data)."""

import json

import pytest

pd = pytest.importorskip("pandas")

from strokecopilot.contract import build  # noqa: E402
from strokecopilot.contract.validate import load_vision_output  # noqa: E402
from strokecopilot.data.splits import HeldoutError  # noqa: E402
from strokecopilot.vision.perfusion import confidence as conf  # noqa: E402

CASES = [f"sub-stroke99{i:02d}" for i in range(10)]
FOLDS = {"heldout": ["sub-stroke9990"], "folds": {str(k): CASES[2 * k:2 * k + 2] for k in range(5)}}
OK = {"occlusion_reason": "ok", "occlusion_site": "ICA", "occlusion_site_share": 0.9, "occlusion_distance_mm": 2.0}
EVIDENCE = {"reason": "ok", "site": "ICA", "site_share": 0.9, "distance_mm": 2.0}  # volumes.occlusion_evidence keys


def _write_runs(root, cases=CASES, core=None, haem=None, evidence=None):
    core = core or {c: 10.0 + i for i, c in enumerate(cases)}
    haem = haem or {c: 0.05 for c in cases}
    evidence = evidence or {}
    fold = {c: k for k, members in FOLDS["folds"].items() for c in members}
    fold.update({c: 0 for c in cases if c not in fold})
    for k, name in build.NNUNET_RUNS.items():
        (root / name).mkdir(parents=True)
        pd.DataFrame([{"case": c, "fold": k, "pred_ml": core[c]} for c in cases if int(fold[c]) == k]
                     ).reindex(columns=["case", "fold", "pred_ml"]).to_csv(root / name / "per_case.csv", index=False)
    (root / build.THRESHOLD_RUN).mkdir()
    rows = []
    for c in cases:
        for variant in ("within_tmax6", "unrestricted"):
            rows.append({"variant": variant, "case": c, "fold": int(fold[c]), "contract_penumbra_ml": 80.0,
                         **{**OK, **evidence.get(c, {})}})
    pd.DataFrame(rows).to_csv(root / build.THRESHOLD_RUN / "per_case.csv", index=False)
    (root / build.HAEMORRHAGE_RUN).mkdir()
    pd.DataFrame([{"case": c, "error": None, build.HAEMORRHAGE_COLUMN: haem[c]} for c in cases]
                 ).to_csv(root / build.HAEMORRHAGE_RUN / "per_case.csv", index=False)
    (root / build.CONFIDENCE_RUN).mkdir()
    pd.DataFrame([{"case": c, "fold": int(fold[c]), "confidence_core_distance": 0.6, "confidence_penumbra": 0.9}
                  for c in cases]).to_csv(root / build.CONFIDENCE_RUN / "per_case.csv", index=False)
    return root


def test_build_case_fields_and_validation():
    obj = build.build_case("sub-stroke9900", "test", 18.04, 95.0, EVIDENCE, 0.2, {"core": 0.61234, "penumbra": 0.9})
    assert obj["core_ml"] == 18.0 and obj["penumbra_ml"] == 95.0 and obj["mismatch_volume_ml"] == 77.0
    assert obj["mismatch_ratio"] == 5.28 and obj["occlusion_site"] == "ICA" and obj["aspects"] is None
    assert obj["haemorrhage"] == {"present": True, "confidence": 0.2}  # 0.2 >= 0.175 (DEC-28)
    assert obj["confidence"] == {"core": 0.6123, "penumbra": 0.9} and obj["warnings"] == []
    low = build.build_case("sub-stroke9900", "test", 0.4, 30.0, {**EVIDENCE, "reason": "empty_mask"}, 0.17, {"core": 0.9, "penumbra": 0.8})
    assert low["haemorrhage"]["present"] is False and low["occlusion_site"] == "unknown"
    assert low["mismatch_ratio"] is None and low["warnings"] == [conf.NEAR_EMPTY_WARNING]
    with pytest.raises(ValueError, match="probability"):
        build.build_case("sub-stroke9900", "test", 5.0, 30.0, EVIDENCE, 1.5, {"core": 0.9, "penumbra": 0.8})


def test_build_all_writes_valid_files_and_counts(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_HELDOUT", raising=False)
    runs = _write_runs(tmp_path / "runs", core={c: (0.3 if i == 0 else 10.0 + i) for i, c in enumerate(CASES)},
                       haem={c: (0.9 if i < 3 else 0.01) for i, c in enumerate(CASES)},
                       evidence={CASES[1]: {"occlusion_site": "MCA"}, CASES[2]: {"occlusion_distance_mm": 12.0}})
    summary = build.build_all(runs, tmp_path / "out", "test", folds=FOLDS, expected=10)
    files = sorted((tmp_path / "out").glob("*.json"))
    assert len(files) == 10 and all(load_vision_output(f) for f in files)
    assert summary["cases"] == 10 and summary["haemorrhage_present"] == 3
    assert summary["occlusion_site"] == {"ICA": 8, "unknown": 2}
    assert summary["warnings"] == {conf.NEAR_EMPTY_WARNING: 1} and summary["null"]["mismatch_ratio"] == 1
    assert not any(c in json.dumps(summary) for c in CASES)


def test_load_sources_refuses_mismatches_and_heldout(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_HELDOUT", raising=False)
    runs = _write_runs(tmp_path / "a")
    with pytest.raises(ValueError, match="expected 129"):
        build.load_sources(runs, folds=FOLDS)
    runs = _write_runs(tmp_path / "b")
    table = pd.read_csv(runs / build.HAEMORRHAGE_RUN / "per_case.csv").iloc[1:]
    table.to_csv(runs / build.HAEMORRHAGE_RUN / "per_case.csv", index=False)
    with pytest.raises(ValueError, match="cases that the nnU-Net runs do not"):
        build.load_sources(runs, folds=FOLDS, expected=10)
    runs = _write_runs(tmp_path / "c", cases=CASES + ["sub-stroke9990"])
    with pytest.raises(HeldoutError):
        build.load_sources(runs, folds=FOLDS, expected=11)
    runs = _write_runs(tmp_path / "d")
    table = pd.read_csv(runs / build.HAEMORRHAGE_RUN / "per_case.csv")
    table["error"] = table["error"].astype(object)
    table.loc[0, "error"] = "failed"
    table.to_csv(runs / build.HAEMORRHAGE_RUN / "per_case.csv", index=False)
    with pytest.raises(ValueError, match="failed"):
        build.load_sources(runs, folds=FOLDS, expected=10)


def test_ground_truth_case_and_build(tmp_path, monkeypatch):
    gt = build.build_ground_truth_case("sub-stroke9900", "gt", 101.04, 140.0, EVIDENCE)
    assert gt["core_ml"] == 101.0 and gt["haemorrhage"] == {"present": False, "confidence": 0.0}
    assert gt["confidence"] == {"core": 1.0, "penumbra": 1.0} and gt["warnings"] == [] and gt["occlusion_site"] == "ICA"
    monkeypatch.delenv("ALLOW_HELDOUT", raising=False)
    runs = _write_runs(tmp_path / "runs")
    for k, name in build.NNUNET_RUNS.items():  # add reference volumes to the fake fold tables
        table = pd.read_csv(runs / name / "per_case.csv")
        table["ref_ml"] = 0.5 if k == 0 else 30.0
        table.to_csv(runs / name / "per_case.csv", index=False)
    summary = build.build_ground_truth_all(runs, tmp_path / "gt", "gt", folds=FOLDS, expected=10)
    assert summary["cases"] == 10 and summary["haemorrhage_present"] == 0 and summary["warnings"] == {}
    assert summary["null"]["mismatch_ratio"] == 2  # the two 0.5 mL references
    assert len(list((tmp_path / "gt").glob("*.json"))) == 10
