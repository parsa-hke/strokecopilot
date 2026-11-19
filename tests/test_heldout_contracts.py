"""The held-out contract path (contract/heldout.py) on synthetic data: fake checkpoints, fake maps and fake IDs."""

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
nib = pytest.importorskip("nibabel")
pytest.importorskip("scipy")

from strokecopilot.contract import heldout as ho  # noqa: E402
from strokecopilot.contract.validate import validate_vision_output  # noqa: E402
from strokecopilot.data.isles24 import Case  # noqa: E402
from strokecopilot.data.splits import HeldoutError  # noqa: E402
from strokecopilot.vision.haemorrhage.aggregate import Calibrator  # noqa: E402
from strokecopilot.vision.perfusion import nnunet_wrapper as nw  # noqa: E402
from strokecopilot.vision.perfusion import thresholds  # noqa: E402

CASES = [f"sub-stroke99{i:02d}" for i in range(12)]
FOLDS = {"heldout": CASES[10:], "folds": {str(k): CASES[2 * k:2 * k + 2] for k in range(5)}}
PLANS = {"dataset_name": nw.DATASET_NAME, "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
         "foreground_intensity_properties_per_channel": {"0": {"mean": 40.0, "std": 12.5}},
         "configurations": {"3d_fullres": {"patch_size": [96, 128, 112], "spacing": [1.5, 1.5, 1.5]}}}
DATASET_JSON = {"channel_names": nw.CHANNEL_NAMES, "numTraining": 129}
MAPPING = {"features": ["log_cutoff_distance"], "fill": [1.43], "mean": [1.43], "std": [1.07], "weights": [-0.91],
           "intercept": -0.83, "n": 129, "positives": 43, "l2": 1.0}
SHAPE, SPACING = (40, 30, 20), 2.0  # 125 voxels per mL


def saved(fold, plans=PLANS, epoch=300, trainer=nw.TRAINER):
    return {"init_args": {"fold": fold, "configuration": "3d_fullres", "plans": json.loads(json.dumps(plans)),
                          "dataset_json": DATASET_JSON}, "trainer_name": trainer, "epoch": epoch, "mirroring": (0, 1, 2)}


def five(**kw):
    return {k: saved(k, **kw) for k in ho.FOLDS}


def synthetic_case(case_id="sub-stroke9900", infarct_voxels=(slice(24, 30), slice(8, 14), slice(6, 12))):
    """Brain on both sides; Tmax 10 s and low CBF on the high-x side (the stroke); an occlusion near an ICA label."""
    ncct = np.full(SHAPE, -1000.0, dtype=np.float32)
    cbf = np.zeros(SHAPE, dtype=np.float32)
    tmax = np.full(SHAPE, -30.0, dtype=np.float32)
    brain = (slice(4, 36), slice(4, 26), slice(4, 16))
    ncct[brain], cbf[brain], tmax[brain] = 30.0, 50.0, 1.0
    tmax[24:34, 8:20, 6:14] = 10.0
    cbf[24:30, 8:14, 6:12] = 5.0
    infarct = np.zeros(SHAPE, bool)
    infarct[infarct_voxels] = True
    cow = np.zeros(SHAPE, np.uint8)
    cow[20:22, 10:12, 8:10] = 4  # ICA label
    occlusion = np.zeros(SHAPE, bool)
    occlusion[20:22, 10:12, 8:10] = True
    arrays = {"ncct": ncct, "cbf": cbf, "tmax": tmax, "occlusion": occlusion, "cow": cow, "infarct": infarct}
    return Case(id=case_id, arrays=arrays, spacing_mm=SPACING, affine=np.eye(4), clinical={},
                native_volumes={"infarct_ml": float(infarct.sum()) / 125.0, "tmax6_ml": 10 * 12 * 8 / 125})


# ------------------------------------------------------------------------------- Settings from the checkpoints


def test_differences_treat_tuples_as_lists_and_name_the_paths():
    assert ho.differences({"a": (1, 2), "b": {"c": 1.0}}, {"a": [1, 2], "b": {"c": 1.0}}) == []
    assert ho.differences({"a": [1, 2], "b": {"c": 1}}, {"a": [1, 3], "b": {"d": 1}}) == \
        ["a[1]", "b.c (only in first)", "b.d (only in second)"]


def test_five_agreeing_folds_pass():
    debug = {k: {"fold": k, "plans": PLANS, "dataset_json": DATASET_JSON} for k in ho.FOLDS}
    recorded = {k: PLANS["configurations"]["3d_fullres"] for k in ho.FOLDS}
    assert ho.check_folds(five(), debug, recorded) == []


def test_disagreeing_folds_are_named():
    other = json.loads(json.dumps(PLANS))
    other["configurations"]["3d_fullres"]["patch_size"] = [64, 64, 64]
    folds = five()
    folds[3] = saved(3, plans=other)
    folds[1] = saved(2)  # the wrong fold in fold 1's place
    folds[4] = saved(4, epoch=150, trainer="nnUNetTrainer")
    problems = ho.check_folds(folds, debug={0: {"fold": 0, "plans": other, "dataset_json": DATASET_JSON}})
    text = "; ".join(problems)
    assert "fold 1's checkpoint says fold 2" in text and "fold 3's plans differ" in text
    assert "epoch 150" in text and "trained by nnUNetTrainer" in text and "fold 0's checkpoint and debug.json disagree" in text


def test_debug_plans_parse_nnunets_python_reprs(tmp_path):
    path = tmp_path / "debug.json"
    path.write_text(json.dumps({"plans_manager": repr(PLANS), "configuration_manager": repr(PLANS["configurations"]["3d_fullres"]),
                                "dataset_json": repr(DATASET_JSON), "fold": "2"}))
    d = ho.debug_plans(path)
    assert d["fold"] == 2 and ho.differences(d["plans"], PLANS) == []


def test_assemble_results_writes_plans_and_every_fold(tmp_path):
    checkpoints = {}
    for k in ho.FOLDS:
        p = tmp_path / f"ck{k}.pth"
        p.write_bytes(b"weights")
        checkpoints[k] = p
    summary = ho.assemble_results(checkpoints, tmp_path / "results", saved=five())
    trainer = tmp_path / "results" / nw.DATASET_NAME / ho.TRAINER_DIR
    assert json.loads((trainer / "plans.json").read_text()) == PLANS
    assert json.loads((trainer / "dataset.json").read_text()) == DATASET_JSON
    assert all((trainer / f"fold_{k}" / ho.CHECKPOINT).read_bytes() == b"weights" for k in ho.FOLDS)
    assert summary["folds"] == list(ho.FOLDS) and len(summary["plans_sha256"]) == 64


def test_assemble_results_refuses_disagreeing_folds(tmp_path):
    folds = five()
    folds[2] = saved(3)
    with pytest.raises(ValueError, match="do not agree"):
        ho.assemble_results({k: tmp_path / "x" for k in ho.FOLDS}, tmp_path / "r", saved=folds)


def test_plan_differences_look_only_at_what_inference_uses():
    rebuilt = json.loads(json.dumps(PLANS))
    rebuilt["dataset_name"] = "elsewhere"  # not compared
    assert ho.plan_differences(PLANS, rebuilt) == []
    rebuilt["foreground_intensity_properties_per_channel"]["0"]["mean"] = 41.0
    rebuilt["configurations"]["3d_fullres"]["spacing"] = [1.0, 1.5, 1.5]
    assert set(ho.plan_differences(PLANS, rebuilt)) == {"foreground_intensity_properties_per_channel.0.mean",
                                                         "configurations.3d_fullres.spacing[0]"}


def test_predict_command_ensembles_all_five_final_checkpoints():
    cmd = ho.predict_command("in", "out")
    f = cmd.index("-f")
    assert cmd[f + 1:f + 6] == ["0", "1", "2", "3", "4"] and cmd[f + 6] == "-chk"
    assert cmd[cmd.index("-chk") + 1] == "checkpoint_final.pth" and cmd[cmd.index("-tr") + 1] == nw.TRAINER
    assert "--disable_tta" not in cmd  # mirroring on, as in training's validation
    single = ho.predict_command("in", "out", folds=(3,))
    assert single[single.index("-f") + 1:single.index("-chk")] == ["3"]


# ------------------------------------------------------------------------------- Each field, as the runs did


def test_vision_a_inputs_match_exp1s_baseline_row():
    case = synthetic_case()
    pred = np.zeros(SHAPE, bool)
    pred[24:29, 8:14, 6:12] = True
    a = ho.vision_a_inputs(case, pred)
    row = thresholds.baseline_row(case, fold=0)  # EXP-1's own function
    assert a["penumbra_ml"] == row["contract_penumbra_ml"]
    assert a["threshold_core_ml"] == pytest.approx(row["pred_ml"])
    assert a["evidence"] == {k[len("occlusion_"):]: v for k, v in row.items() if k.startswith("occlusion_")
                             and k[len("occlusion_"):] in a["evidence"]}
    assert a["core_ml"] == pytest.approx(pred.sum() / 125.0)
    assert a["scores"]["ref_ml"] == pytest.approx(36 * 6 / 125.0) and 0 < a["scores"]["dice"] < 1


def test_case_inputs_build_two_valid_contracts():
    case = synthetic_case()
    pred = case.arrays["infarct"].copy()
    inputs = ho.case_inputs(case, pred, MAPPING, haem_conf=0.30)
    predicted, truth = ho.build_pair(case.id, inputs, "test-version", "test-version (ground truth)")
    validate_vision_output(predicted)
    validate_vision_output(truth)
    assert predicted["haemorrhage"] == {"present": True, "confidence": 0.3}  # 0.30 >= DEC-28's 0.175
    assert truth["haemorrhage"] == {"present": False, "confidence": 0.0}
    assert truth["core_ml"] == round(case.native_volumes["infarct_ml"], 1)
    assert predicted["occlusion_site"] == truth["occlusion_site"] == "ICA"
    assert 0 < predicted["confidence"]["core"] < 1 and truth["confidence"] == {"core": 1.0, "penumbra": 1.0}


def test_near_empty_core_gets_dec27s_warning():
    case = synthetic_case()
    inputs = ho.case_inputs(case, np.zeros(SHAPE, bool), MAPPING, haem_conf=0.05)
    predicted, _ = ho.build_pair(case.id, inputs, "v", "v")
    assert predicted["core_ml"] == 0.0 and predicted["warnings"] and predicted["haemorrhage"]["present"] is False


def test_missing_threshold_core_is_allowed_only_for_the_distance_mapping(monkeypatch):
    case = synthetic_case()
    real = thresholds.threshold_masks
    monkeypatch.setattr(thresholds, "threshold_masks", lambda *a, **k: {**real(*a, **k), "core": None})
    assert ho.case_inputs(case, case.arrays["infarct"], MAPPING, 0.05)["threshold_core_ml"] is None
    with pytest.raises(ValueError, match="needs it"):
        ho.case_inputs(case, case.arrays["infarct"], {**MAPPING, "features": ["log_disagreement"]}, 0.05)


def test_haemorrhage_confidence_uses_exp16s_slabs_and_the_calibration():
    hu = np.full((64, 64, 20), -1000.0)
    yy, xx = np.mgrid[:64, :64]
    for z in range(20):
        hu[..., z][np.hypot(yy - 32, xx - 32) < 25] = 35.0
    img = nib.Nifti1Image(hu, np.diag([3.5, 3.5, 2.5, 1.0]))
    seen = {}

    def predict(images):
        seen["n"] = len(images)
        p = np.full(len(images), 0.05)
        p[:2] = 0.9
        return p

    cal = Calibrator(method="topk", k=2, temperature=1.0, bias=0.0, threshold=0.175, kind="platt")
    assert ho.haemorrhage_confidence(img, predict, cal) == pytest.approx(0.9)
    assert seen["n"] == 10  # 20 slices of 2.5 mm -> 10 slabs of 5 mm


# ------------------------------------------------------------------------------- Case sets and outputs


def test_case_sets_follow_the_mode(monkeypatch):
    monkeypatch.delenv("ALLOW_HELDOUT", raising=False)
    ho.check_case_set(CASES[:4], "development", FOLDS)
    with pytest.raises(HeldoutError):
        ho.check_case_set(CASES[:4] + CASES[10:11], "development", FOLDS)
    with pytest.raises(HeldoutError):
        ho.check_case_set(CASES[10:], "heldout", FOLDS)
    monkeypatch.setenv("ALLOW_HELDOUT", "1")
    with pytest.raises(ValueError, match="2 cases given"):  # the fake split has 2, the real one 20
        ho.check_case_set(CASES[10:], "heldout", FOLDS)
    monkeypatch.setattr(ho, "EXPECTED_HELDOUT", 2)
    ho.check_case_set(CASES[10:], "heldout", FOLDS)
    with pytest.raises(ValueError):
        ho.check_case_set(CASES[10:11], "heldout", FOLDS)
    with pytest.raises(ValueError, match="Unknown mode"):
        ho.check_case_set(CASES[:2], "test", FOLDS)


def test_compare_with_reference_counts_fields_without_ids():
    base = {"haemorrhage": {"present": False, "confidence": 0.1}, "core_ml": 10.0, "penumbra_ml": 50.0,
            "mismatch_ratio": 5.0, "mismatch_volume_ml": 40.0, "occlusion_site": "ICA", "aspects": None, "warnings": []}
    a, b = CASES[0], CASES[1]
    new = {a: {**base, "haemorrhage": {"present": False, "confidence": 0.1002}},
           b: {**base, "core_ml": 10.4, "mismatch_ratio": 4.81, "mismatch_volume_ml": 39.6}}
    old = {a: dict(base), b: dict(base)}
    out = ho.compare_with_reference(new, old)
    assert out["cases"] == 2 and out["mismatched"]["core_ml"] == 1 and out["mismatched"]["penumbra_ml"] == 0
    assert out["largest_difference"]["core_ml"] == 0.4 and out["largest_difference"]["haemorrhage_confidence"] == 0.0002
    assert out["mismatched"]["haemorrhage.present"] == 0  # a confidence moved in the 4th decimal; the call did not
    assert a not in json.dumps(out) and b not in json.dumps(out)


def test_mask_agreement():
    a = np.zeros((4, 4), bool)
    b = a.copy()
    assert ho.mask_agreement(a, b) == {"differing_voxels": 0, "dice": 1.0}
    a[0, :2], b[0, 1:3] = True, True
    assert ho.mask_agreement(a, b) == {"differing_voxels": 2, "dice": 0.5}


def test_write_contracts_never_overwrites(tmp_path):
    ho.write_contracts({"x": {"k": 1}}, tmp_path / "out")
    with pytest.raises(FileExistsError):
        ho.write_contracts({"x": {"k": 2}}, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "x.json").read_text()) == {"k": 1}


def test_on_drive():
    assert ho.on_drive("/content/drive/MyDrive/StrokeCopilot/x")
    assert not ho.on_drive("/tmp/x")
