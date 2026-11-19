"""The Space's precomputed bank (DEC-32 §4): every answer the page can show without a live call.

- Presets: the 5 fixtures and the 150 synthetic scenarios, each with the saved, verified answers of DEC-24's four models
  (EXP-37, rulebook arms, rescored with points 19-20) and DEC-30's pair (EXP-47). Sonnet's answer is the fallback for a
  failed live run (`builder.nearest` over the presets' features).
- Real development cases by alias (`D-01`...): EXP-37's 20 sampled cases, predicted and ground-truth vision arms, with
  the case ID replaced everywhere by the alias. ISLES'24 is CC BY-NC-SA 4.0 (DEC-32 §6); IDs never appear.
- Held-out cases are added after Phase 4's analyses, the same way (aliases `H-01`...).

Layout of the output folder (built, never committed: it holds real-case notes):
    index.json            one row per case: id, kind, title, the rules' decisions, the models present, builder inputs
    cases/<id>.json       the case (contract and note per arm) and each model's answer as `live.view` shapes it
`scan` refuses the folder if any case ID, RSNA ID or key pattern appears in it.
"""

import json
import re
from pathlib import Path

from strokecopilot.demo import builder, live
from strokecopilot.nlp.verifier import rules
from strokecopilot.nlp.verifier import scenarios as sc
from strokecopilot.pipeline import compare
from strokecopilot.pipeline import end_to_end as e2e

MODELS = {"claude-sonnet-5-5": "EXP-37", "qwen3.5-27b": "EXP-37", "gemma-3-27b": "EXP-37", "medgemma-27b-text": "EXP-37",
          "gemma-3-4b": "EXP-47", "gemma-3-4b-tuned": "EXP-47"}
FALLBACK_MODEL = "claude-sonnet-5-5"
PRESET_ARMS = {"fixture": "fixtures_rulebook", "scenario": "scenarios_rulebook"}
REAL_ARMS = {"predicted": "real_predicted", "groundtruth": "real_groundtruth"}
KEY_PATTERNS = (re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"), re.compile(r"\bhf_[A-Za-z0-9]{20,}"), re.compile(r"\bID_[0-9a-f]{6,}"))


def _cell(roots, model, arm):
    folder = Path(roots[MODELS[model]]) / model / arm
    return folder if (folder / "config.json").exists() else None


def _arm_view(case, folder):
    """One model's saved answer for one case, shaped by live.view, or a record of why there is none."""
    metrics_file = compare._metrics_file(folder)
    traces_dir = folder / compare._traces_dir_for(metrics_file.name)
    if not traces_dir.exists():
        traces_dir = folder / "traces"
    output, trace_file = folder / "outputs" / f"{case['id']}.json", traces_dir / f"{case['id']}.json"
    if not output.exists() or not trace_file.exists():
        row = next((r for r in json.loads(metrics_file.read_text())["per_case"] if r["case"] == case["id"]), {})
        return {"available": False, "reason": "format failure" if (row.get("error") or "").startswith("format")
                else "no verified answer", "detail": (row.get("error") or "")[:160]}
    rec = json.loads(output.read_text())
    saved = json.loads(trace_file.read_text())
    v = live.view(case, rec, saved["claims"], saved["traces"])
    return {"available": True, **{k: v[k] for k in ("recommendation", "claims", "decisions", "omissions",
                                                    "uncertainty_omissions", "counts")}}


def preset_inputs(case, tags=None):
    """The builder inputs a preset corresponds to, so the page can load it into the controls. Approximate (hidden
    fields such as the imaging clock differ), and None when the preset lies outside the builder's bounds."""
    c = rules.case_from(case["vision"], case["facts"])
    facts = case["facts"]
    tags = tags or {}
    contra = tags.get("contraindication") or "none"
    deficit = (facts.get("deficits").value if facts.get("deficits") else None)
    raw = {"minutes": int(round(c.minutes)) if c.minutes is not None else None,
           "onset": tags.get("onset") or ("wake_up" if c.wake_up else "witnessed" if c.time_basis == "onset_to_imaging" else "unwitnessed"),
           "age": int(c.age) if c.age is not None else None, "sex": facts.get("sex").value if facts.get("sex") else None,
           "nihss": int(c.nihss) if c.nihss is not None else None, "mrs": int(c.mrs) if c.mrs is not None else None,
           "core_ml": c.core, "penumbra_ml": c.penumbra, "site": case["vision"]["occlusion_site"],
           "aspects": case["vision"]["aspects"], "haemorrhage": bool(case["vision"]["haemorrhage"]["present"]),
           "contraindication": "none" if contra == "haemorrhage" else contra,
           "deficit": sc.DEFICITS.index(deficit) if deficit in sc.DEFICITS else 0,
           "antiplatelet": bool(facts.get("antiplatelet").value) if facts.get("antiplatelet") else False,
           "inr": c.inr if contra == "warfarin_inr" else None}
    if c.sleep_mid_minutes is not None and raw["onset"] == "wake_up":
        raw["sleep_minutes"] = 420
    raw = {k: v for k, v in raw.items() if v is not None}
    try:
        return builder.validate(raw)
    except builder.InputError:
        return None


def _summary(expected):
    return {d: {"value": v["value"], "reason": v["reason"]} for d, v in expected.items()}


def presets(roots, models=MODELS):
    """[(index row, case file)] for the fixtures and scenarios."""
    scenario_tags = {s.id: s.tags for s in sc.generate()}
    out = []
    for kind, source in (("fixture", "fixtures"), ("scenario", "scenarios")):
        for case in e2e.load_cases(source):
            expected = {d: v.to_dict() for d, v in rules.decide(rules.case_from(case["vision"], case["facts"])).items()}
            case = {**case, "expected_detail": expected}
            answers = {}
            for model in models:
                folder = _cell(roots, model, PRESET_ARMS[kind])
                if folder is not None:
                    answers[model] = _arm_view(case, folder)
            title = case["id"].split("_", 1)[1].replace("_", " ") if kind == "fixture" else case["id"].replace("-", " ")
            row = {"id": case["id"], "kind": kind, "title": title, "rules": _summary(expected),
                   "models": sorted(m for m, a in answers.items() if a["available"]),
                   "inputs": preset_inputs(case, scenario_tags.get(case["id"])),
                   "features": builder.features(case["vision"], case["facts"], expected)}
            file = {"id": case["id"], "kind": kind, "banner": live.BANNER, "rules": expected,
                    "arms": {"rulebook": {"vision": case["vision"], "note": case["note"], "answers": answers}}}
            out.append((row, file))
    return out


def _alias_everything(obj, case_id, alias):
    return json.loads(json.dumps(obj).replace(case_id, alias))


def development_cases(roots, vision_dirs, notes_dir, ids, models=MODELS):
    """[(index row, case file)] for real development cases by alias; vision_dirs: {"predicted": dir, "groundtruth": dir}."""
    from strokecopilot.pipeline import heldout_analysis as ha

    aliases = ha.aliases_for(ids, "D")
    out = []
    for case_id in sorted(ids):
        alias = aliases[case_id]
        arms, rules_by_arm = {}, {}
        for arm, cell_arm in REAL_ARMS.items():
            case = e2e.real_case(case_id, vision_dirs[arm], notes_dir)
            expected = {d: v.to_dict() for d, v in rules.decide(rules.case_from(case["vision"], case["facts"])).items()}
            case = {**case, "expected_detail": expected}
            answers = {}
            for model in models:
                folder = _cell(roots, model, cell_arm)
                if folder is not None:
                    answers[model] = _arm_view(case, folder)
            arms[arm] = {"vision": case["vision"], "note": case["note"], "rules": expected, "answers": answers}
            rules_by_arm[arm] = _summary(expected)
        row = {"id": alias, "kind": "development", "title": f"Real case {alias}", "rules": rules_by_arm["groundtruth"],
               "rules_predicted": rules_by_arm["predicted"],
               "models": sorted({m for a in arms.values() for m, x in a["answers"].items() if x["available"]})}
        file = {"id": alias, "kind": "development", "banner": live.BANNER, "arms": arms}
        out.append((_alias_everything(row, case_id, alias), _alias_everything(file, case_id, alias)))
    return out


def write(entries, out_dir):
    """index.json and cases/<id>.json. Refuses to write into a folder that already has an index."""
    out = Path(out_dir)
    if (out / "index.json").exists():
        raise FileExistsError(f"{out} already holds a bank; build into a new folder.")
    (out / "cases").mkdir(parents=True, exist_ok=True)
    index = []
    for row, file in entries:
        (out / "cases" / f"{row['id']}.json").write_text(json.dumps(file, separators=(",", ":"), default=str))
        index.append(row)
    (out / "index.json").write_text(json.dumps({"banner": live.BANNER, "cases": index}, separators=(",", ":"), default=str))
    return out


def passages():
    """The rulebook passages the page shows when a claim cites one: our own words (DEC-18), with their pointers."""
    from strokecopilot.nlp.corpus import ingest

    return {pid: {k: p.get(k) for k in ("source", "section", "pointer", "treatment", "text")} for pid, p in ingest.load().items()}


def write_extras(out_dir, chart_data):
    """charts.json (demo.charts) and passages.json beside the bank."""
    out = Path(out_dir)
    (out / "charts.json").write_text(json.dumps(chart_data, separators=(",", ":"), default=str))
    (out / "passages.json").write_text(json.dumps(passages(), separators=(",", ":")))
    return out


def scan(folder, case_ids=()):
    """Problems found in a built folder: case IDs, RSNA IDs or key patterns in any file. Empty means clean."""
    problems = []
    ids = [i for i in case_ids if i]
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if any(i in text for i in ids):
            problems.append(f"{path.name}: a case ID")
        if any(p.search(text) for p in KEY_PATTERNS):
            problems.append(f"{path.name}: an ID or key pattern")
    return problems


def main(argv=None):
    """Build the bank from the Mac's runs/ and Drive's artifacts (development cases only before Phase 4's results).

    python -m strokecopilot.demo.bank --out demo/space/build/bank --drive "<Drive>/StrokeCopilot"
    """
    import argparse
    import os

    from strokecopilot.data import splits

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--drive", default=None, help="the StrokeCopilot folder on Drive (default: config.local_drive_root)")
    parser.add_argument("--exp37", default="runs/EXP-37_paid_runs")
    parser.add_argument("--exp47", default="runs/EXP-47_small_reasoner")
    args = parser.parse_args(argv)
    from strokecopilot.config import local_drive_root

    drive = Path(os.path.expanduser(args.drive)) if args.drive else local_drive_root()
    if drive is None:
        raise ValueError("No Drive folder found; pass --drive or set STROKECOPILOT_DRIVE.")
    args.drive = str(drive)
    art = drive / "artifacts"
    roots = {"EXP-37": args.exp37, "EXP-47": args.exp47}
    entries = presets(roots)
    real = sorted(r["case"] for r in json.loads((Path(args.exp37) / FALLBACK_MODEL / "real_predicted" / "metrics_rescore.json")
                                                .read_text())["per_case"])
    splits.check_ids(real)
    entries += development_cases(roots, {"predicted": art / "vision_outputs_v1", "groundtruth": art / "vision_outputs_gt_v1"},
                                 art / "isles24_notes_v1", real)
    out = write(entries, args.out)
    from strokecopilot.demo import charts

    runs = Path(os.path.expanduser(args.drive)) / "runs"
    write_extras(out, charts.charts({
        "exp48": "runs/EXP-48_small_reasoner_report/rescore_points_19_20/report.json",
        "exp42": "runs/EXP-42_c9_table/rescore_points_19_20/report.json",
        "exp41": "runs/EXP-41_labelling/labels/agreement_final.json",
        "exp49": "runs/EXP-49_visionB_false_alarms/metrics.json",
        "exp50": runs / "EXP-50_visionB_isles_false_alarms" / "metrics.json",
        "exp60": "runs/EXP-60_visionB_rsna_test/metrics.json",
        "exp63": "runs/EXP-63_d3_dev_check/report.json"}))
    problems = scan(out, splits.development_ids())
    if problems:
        raise ValueError(f"The bank failed the leak scan ({len(problems)} files). Delete {out} and check.")
    print(f"OK: bank with {len(entries)} cases in {out}; leak scan clean.")


if __name__ == "__main__":
    main()
