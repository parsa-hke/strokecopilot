"""Assemble vision_output.json for the 129 development cases (C7; plan §6, §7). No model runs here: each field is
read from a finished run, joined by case, and every object is validated before it is written.

Where each field comes from (development folds only; held-out cases are refused):
- core_ml: nnU-Net's validation prediction from the fold model that did not train on the case (DEC-25), the
  `pred_ml` column of the five fold runs (EXP-14, EXP-18 to EXP-21).
- penumbra_ml: the Tmax > 6 s volume, core included (DEC-7, DEC-26), EXP-1's `contract_penumbra_ml`
  (variant within_tmax6; the penumbra does not depend on the variant).
- mismatch_ratio, mismatch_volume_ml: derived from the rounded volumes (volumes.contract_volumes).
- occlusion_site: the contract rule (volumes.occlusion_site: ICA or basilar within 5 mm, else unknown) applied to
  EXP-1's occlusion evidence columns.
- haemorrhage: EXP-16's calibrated scan confidence on 5 mm slabs (`slab_5mm_confidence`, EXP-6's aggregation and
  Platt calibration); present = confidence >= 0.175 (DEC-28, infer.CONTRACT_THRESHOLD).
- confidence: the cross-fitted values of EXP-32 (DEC-27, confidence.development_confidence).
- aspects: null; no model produces it (DEC-9).
- warnings: confidence.core_warnings (DEC-27: a core under 1 mL).

Ground-truth contracts (the ground-truth arm of DEC-11; build_ground_truth_all): the same cases with what vision would
report if it were right, so the ground-truth vs predicted arms isolate the effect of vision errors:
- core_ml: the reference final infarct volume (`ref_ml`, from native_volumes.csv through the nnU-Net tables). It is
  final infarct, not acute core (plan §8.1), the only label ISLES'24 has.
- haemorrhage: {present: false, confidence: 0.0}. ISLES'24 has no bleed labels; these patients went on to
  reperfusion therapy, so they are presumed bleed-free (as in EXP-16).
- penumbra_ml, occlusion_site: the same as the predicted contracts (no penumbra label; the site already comes from
  the reference occlusion mask).
- confidence: {core: 1.0, penumbra: 1.0}; warnings: [] (nothing is estimated); aspects: null (DEC-9).

Usage: python -m strokecopilot.contract.build <Drive runs folder> <output folder> [--model-version TEXT] [--ground-truth]
Writes one <case>.json per case into the output folder (case IDs: Drive only) and prints counts only.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from strokecopilot.contract.validate import validate_vision_output
from strokecopilot.data.splits import check_ids
from strokecopilot.vision.haemorrhage.infer import CONTRACT_THRESHOLD
from strokecopilot.vision.perfusion import confidence as conf
from strokecopilot.vision.perfusion import volumes

SCHEMA_VERSION = "0.1"
NNUNET_RUNS = {0: "EXP-14_nnunet_fold0", 1: "EXP-18_nnunet_fold1", 2: "EXP-19_nnunet_fold2", 3: "EXP-20_nnunet_fold3",
               4: "EXP-21_nnunet_fold4"}
THRESHOLD_RUN = "EXP-1_thresholds_baseline"
THRESHOLD_VARIANT = "within_tmax6"
HAEMORRHAGE_RUN = "EXP-16_visionB_isles_check"
HAEMORRHAGE_COLUMN = "slab_5mm_confidence"
CONFIDENCE_RUN = "EXP-32_visionA_confidence_dec27"
EXPECTED_CASES = 129


def build_case(case_id, model_version, core_ml, penumbra_ml, evidence, haemorrhage_confidence, confidence):
    """One validated vision_output dict. evidence: volumes.occlusion_evidence keys (reason, site, site_share,
    distance_mm); confidence: {"core", "penumbra"}. Raises ContractError if the result does not validate."""
    if haemorrhage_confidence is None or not 0.0 <= float(haemorrhage_confidence) <= 1.0:
        raise ValueError("The haemorrhage confidence must be a probability.")
    site, _ = volumes.occlusion_site(evidence)
    obj = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "model_version": model_version,
        "haemorrhage": {"present": bool(float(haemorrhage_confidence) >= CONTRACT_THRESHOLD),
                        "confidence": round(float(haemorrhage_confidence), 4)},
        **volumes.contract_volumes(core_ml, penumbra_ml),
        "occlusion_site": site,
        "aspects": None,
        "confidence": {"core": round(float(confidence["core"]), 4), "penumbra": round(float(confidence["penumbra"]), 4)},
        "warnings": conf.core_warnings(core_ml),
    }
    validate_vision_output(obj)
    return obj


def _evidence(row):
    def num(key):
        value = row.get(key)
        return None if value is None or pd.isna(value) else float(value)

    reason = row.get("occlusion_reason")
    return {"reason": reason if isinstance(reason, str) else "missing", "site": row.get("occlusion_site"),
            "site_share": num("occlusion_site_share"), "distance_mm": num("occlusion_distance_mm")}


def load_sources(runs_dir, folds=None, expected=EXPECTED_CASES):
    """{case: inputs for build_case} from the finished runs. Refuses repeated cases, a case missing from any source,
    a fold that differs between sources, and held-out cases (splits.check_ids). Messages carry counts only."""
    runs = Path(runs_dir)
    nnunet = []
    for fold, name in NNUNET_RUNS.items():
        table = pd.read_csv(runs / name / "per_case.csv")
        if set(table["fold"]) != {fold}:
            raise ValueError(f"{name} holds folds {sorted(set(table['fold']))}, expected {fold}.")
        nnunet.append(table[["case", "fold", "pred_ml"]])
    nnunet = pd.concat(nnunet, ignore_index=True)
    threshold = pd.read_csv(runs / THRESHOLD_RUN / "per_case.csv")
    threshold = threshold[threshold["variant"] == THRESHOLD_VARIANT]
    haemorrhage = pd.read_csv(runs / HAEMORRHAGE_RUN / "per_case.csv")
    if haemorrhage["error"].notna().any():
        raise ValueError(f"{int(haemorrhage['error'].notna().sum())} Vision B scans failed in {HAEMORRHAGE_RUN}.")
    tables = {"nnU-Net": nnunet, THRESHOLD_RUN: threshold, HAEMORRHAGE_RUN: haemorrhage}
    for name, table in tables.items():
        if table["case"].duplicated().any():
            raise ValueError(f"{name} repeats {int(table['case'].duplicated().sum())} cases.")
    cases = set(nnunet["case"])
    for name, table in tables.items():
        if set(table["case"]) != cases:
            raise ValueError(f"{name} has {len(set(table['case']) ^ cases)} cases that the nnU-Net runs do not, or the reverse.")
    if expected is not None and len(cases) != expected:
        raise ValueError(f"{len(cases)} cases, expected {expected}.")
    check_ids(sorted(cases), folds)
    fold_of = dict(zip(nnunet["case"], nnunet["fold"]))
    if any(fold_of[c] != f for c, f in zip(threshold["case"], threshold["fold"])):
        raise ValueError("The nnU-Net and threshold runs disagree on a case's fold.")
    confidence = conf.development_confidence(runs / CONFIDENCE_RUN / "per_case.csv", folds=folds)
    if set(confidence) != cases:
        raise ValueError(f"{CONFIDENCE_RUN} has {len(set(confidence) ^ cases)} cases that the nnU-Net runs do not, or the reverse.")
    core = dict(zip(nnunet["case"], nnunet["pred_ml"]))
    thr = threshold.set_index("case")
    haem = dict(zip(haemorrhage["case"], haemorrhage[HAEMORRHAGE_COLUMN]))
    return {case: {"core_ml": float(core[case]), "penumbra_ml": float(thr.at[case, "contract_penumbra_ml"]),
                   "evidence": _evidence(thr.loc[case].to_dict()), "haemorrhage_confidence": float(haem[case]),
                   "confidence": confidence[case], "fold": int(fold_of[case])}
            for case in sorted(cases)}


def summarise(outputs):
    """Counts only, no case IDs."""
    values = list(outputs.values())
    def count(key):
        return sum(v[key] is None for v in values)
    return {
        "cases": len(values),
        "haemorrhage_present": sum(v["haemorrhage"]["present"] for v in values),
        "occlusion_site": dict(sorted(Counter(v["occlusion_site"] for v in values).items())),
        "null": {k: count(k) for k in ("core_ml", "penumbra_ml", "mismatch_ratio", "mismatch_volume_ml", "aspects")},
        "negative_mismatch_volume": sum((v["mismatch_volume_ml"] or 0) < 0 for v in values),
        "warnings": dict(Counter(w for v in values for w in v["warnings"])),
        "core_ml_median": float(pd.Series([v["core_ml"] for v in values]).median()),
        "penumbra_ml_median": float(pd.Series([v["penumbra_ml"] for v in values]).median()),
        "confidence_core_median": float(pd.Series([v["confidence"]["core"] for v in values]).median()),
    }


def build_ground_truth_case(case_id, model_version, ref_ml, penumbra_ml, evidence):
    """One validated ground-truth vision_output dict (see the module docstring)."""
    site, _ = volumes.occlusion_site(evidence)
    obj = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "model_version": model_version,
        "haemorrhage": {"present": False, "confidence": 0.0},
        **volumes.contract_volumes(ref_ml, penumbra_ml),
        "occlusion_site": site,
        "aspects": None,
        "confidence": {"core": 1.0, "penumbra": 1.0},
        "warnings": [],
    }
    validate_vision_output(obj)
    return obj


def load_reference_volumes(runs_dir):
    """{case: ref_ml} from the nnU-Net fold tables (native reference volumes); refuses a case listed twice."""
    refs = {}
    for name in NNUNET_RUNS.values():
        table = pd.read_csv(Path(runs_dir) / name / "per_case.csv")
        for case, ref in zip(table["case"], table["ref_ml"]):
            if case in refs:
                raise ValueError("A case appears in two nnU-Net fold tables.")
            refs[case] = float(ref)
    return refs


def build_ground_truth_all(runs_dir, out_dir, model_version, folds=None, expected=EXPECTED_CASES):
    """Ground-truth contracts for every development case, validated and written like build_all; returns summarise()."""
    sources = load_sources(runs_dir, folds, expected)
    refs = load_reference_volumes(runs_dir)
    outputs = {case: build_ground_truth_case(case, model_version, refs[case], s["penumbra_ml"], s["evidence"])
               for case, s in sources.items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for case, obj in outputs.items():
        (out / f"{case}.json").write_text(json.dumps(obj, indent=1) + "\n")
    return summarise(outputs)


def build_all(runs_dir, out_dir, model_version, folds=None, expected=EXPECTED_CASES):
    """Build, validate and write every development case; returns summarise(). Nothing is written unless all build."""
    sources = load_sources(runs_dir, folds, expected)
    outputs = {case: build_case(case, model_version, s["core_ml"], s["penumbra_ml"], s["evidence"],
                                s["haemorrhage_confidence"], s["confidence"]) for case, s in sources.items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for case, obj in outputs.items():
        (out / f"{case}.json").write_text(json.dumps(obj, indent=1) + "\n")
    return summarise(outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--ground-truth", action="store_true", help="build the ground-truth contracts instead")
    args = parser.parse_args(argv)
    builder = build_ground_truth_all if args.ground_truth else build_all
    print(json.dumps(builder(args.runs_dir, args.out_dir, args.model_version), indent=1))


if __name__ == "__main__":
    main()
