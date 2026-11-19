"""Phase 4's held-out commands from the Mac (DEC-31 §4, amended 19 Nov 2026: Phase 4 code paths may set ALLOW_HELDOUT=1,
per command). Every held-out step goes through this one module, so Parsa's permission rule can name exactly it:

    ALLOW_HELDOUT=1 .venv/bin/python -m strokecopilot.pipeline.phase4 check

Commands print counts only. Each checks the split by sha256, refuses to run without the flag, and checks its own output
for case IDs before printing.

- check: the 20 held-out contracts and ground-truth contracts validate, each note passes its fact check, and what the
  rules say on each (counts). Free.
- status / advance / import: DEC-31 D2's reasoner runs on the held-out arms, with EXP-37's machinery (batched.run_batched:
  Sonnet, the extractor and the judge through the Batch API; Qwen through OpenRouter; the four open models answered on
  Colab, notebooks 53 and 55 in `heldout` mode, then imported here). Six models x two vision arms x 20 cases, plus
  Sonnet's core x0.8 and x1.2 arms (D3's 40 extra requests). Prompt v3 with the rulebook, as frozen (DEC-31 §1).
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from strokecopilot.config import local_drive_root
from strokecopilot.contract.validate import validate_vision_output
from strokecopilot.data import splits
from strokecopilot.pipeline import batched
from strokecopilot.pipeline import end_to_end as e2e
from strokecopilot.pipeline import heldout_analysis as ha
from strokecopilot.pipeline import paid_runs as pr

FOLDS_SHA256 = "023813f2dba930b7521653f1549b1fa1369b8bd29b63658175c5cea58410d291"
EXPECTED_HELDOUT = 20
DIRS = {"predicted": "vision_outputs_heldout_v1", "groundtruth": "vision_outputs_gt_heldout_v1",
        "notes": "isles24_notes_heldout_v1"}


RUN = "EXP-65_heldout_runs"
MODELS = tuple(pr.MODELS) + tuple(pr.SMALL_MODELS)  # DEC-24's four and DEC-30's pair
ARMS = {"heldout_predicted": {"vision": "predicted"}, "heldout_groundtruth": {"vision": "groundtruth"}}
SONNET_SENSITIVITY = {f"heldout_core_x{f:g}": {"vision": "predicted", "core_factor": f} for f in ha.FACTORS}


class Phase4Error(RuntimeError):
    pass


def arms_for(model):
    return {**ARMS, **(SONNET_SENSITIVITY if model == pr.SONNET else {})}


def cells(models=MODELS):
    return [(m, a) for m in models for a in arms_for(m)]


def arm_cases(model, arm, dirs, ids):
    """The cases of one held-out cell. The sensitivity arms are the predicted cases with core_ml scaled
    (heldout_analysis.perturbed_cases: mismatch and the near-empty warning rederived, schema-validated)."""
    spec = arms_for(model)[arm]
    cases = e2e.load_cases("real", ids=ids, vision_dir=dirs[spec["vision"]], notes_dir=dirs["notes"])
    if "core_factor" in spec:
        cases = ha.perturbed_cases(cases, factors=(spec["core_factor"],))
    return cases


def cell_dir(out_root, model, arm):
    return Path(out_root) / model.replace("/", "_") / arm


def _requests(cases):
    from strokecopilot.nlp.reasoner import prompt as rp

    system, schema = rp.system_prompt(pr.PROMPT, rulebook=True), rp.api_schema()
    return {batched.request_key(system, rp.user_message(c["vision"], c["note"])): (system, rp.user_message(c["vision"], c["note"]), schema)
            for c in cases}


def answered(model, arm, dirs, ids, out_root):
    """(answers saved, requests) for one cell."""
    wanted = set(_requests(arm_cases(model, arm, dirs, ids)))
    saved = batched.load_answers(cell_dir(out_root, model, arm) / batched.ANSWERS_FILE, pr.answer_model(model))
    return len(wanted & set(saved)), len(wanted)


def reasoner_pass(models, dirs, ids, out_root, llm_for, workers=pr.WORKERS, log=print):
    """Colab (notebooks 53 and 55, `heldout` mode): only the reasoner for open models served next to the notebook,
    each answer saved to Drive as it arrives. Returns {(model, arm): requests answered in this call}."""
    done = {}
    for model, arm in cells(models):
        requests = _requests(arm_cases(model, arm, dirs, ids))
        out = cell_dir(out_root, model, arm)
        out.mkdir(parents=True, exist_ok=True)
        path = out / batched.ANSWERS_FILE
        before = len(set(requests) & set(batched.load_answers(path, pr.answer_model(model))))
        answers = batched.gather(requests, llm_for(model), log=lambda *_: None, answers_path=path, workers=workers)
        done[(model, arm)] = len(requests) - before
        log(f"{model} / {arm}: {sum('reply' in r for r in answers.values())} of {len(requests)} valid ({before} saved before).")
    return done


def import_answers(src_root, dst_root, models):
    """Colab's answers (on Drive) into the Mac's run folder; only new keys are added. {(model, arm): added}."""
    added = {}
    for model, arm in cells(models):
        rel = Path(model.replace("/", "_")) / arm / batched.ANSWERS_FILE
        src, dst = Path(src_root) / rel, Path(dst_root) / rel
        if not src.exists():
            continue
        name = pr.answer_model(model)
        theirs, mine = batched.load_answers(src, name), batched.load_answers(dst, name)
        new = {k: v for k, v in theirs.items() if k not in mine}
        if new:
            dst.parent.mkdir(parents=True, exist_ok=True)
            batched._save_answers(dst, name, new)
        added[(model, arm)] = len(new)
    return added


def advance(model, arm, dirs, ids, out_root, reasoner, extractor, judge, retriever, wait=False, log=print):
    """One cell forward through EXP-37's passes. {"done": summary} or {"waiting": role, ...}."""
    cases = arm_cases(model, arm, dirs, ids)
    spec = arms_for(model)[arm]
    result = batched.run_batched(cases, reasoner, extractor, judge, retriever, cell_dir(out_root, model, arm), log=log,
                                 source="heldout", version=pr.PROMPT, reasoner_workers=pr.WORKERS, wait=wait, rulebook=True,
                                 extra_config={"phase4": {"experiment": RUN.split("_")[0], "model": model, "arm": arm,
                                                          "cases": len(cases), **spec}})
    if "waiting" in result:
        return result
    return {"done": result["summary"], "cost_usd_estimate": result["cost_usd_estimate"]}


def needs_colab(model):
    return model != pr.SONNET and model not in pr.om.API_ROUTES


def status(models, dirs, ids, out_root):
    rows = []
    for model, arm in cells(models):
        out = cell_dir(out_root, model, arm)
        saved, total = answered(model, arm, dirs, ids, out_root)
        row = {"model": model, "arm": arm, "answers": f"{saved}/{total}", "state": "not started"}
        if (out / "metrics.json").exists():
            row["state"] = "done"
        elif (out / batched.LEDGER_FILE).exists():
            ledger = json.loads((out / batched.LEDGER_FILE).read_text())["batches"]
            open_ = [s.split(":")[0] for s, e in ledger.items() if not e["collected"]]
            row["state"] = f"waiting: {open_[-1]} batch" if open_ else "between stages"
        elif saved:
            row["state"] = "reasoner answers saved"
        rows.append(row)
    return rows


def advance_all(models, dirs, ids, out_root, wait=False, log=print, llm_for=pr.reasoner_llm, verifier=None, only_arms=None):
    extractor, judge = verifier or pr.verifier_llms()
    retriever = pr.make_retriever()
    rows = []
    for model, arm in cells(models):
        if only_arms and arm not in only_arms:
            continue
        if needs_colab(model):
            saved, total = answered(model, arm, dirs, ids, out_root)
            if saved < total:
                rows.append({"model": model, "arm": arm, "state": f"needs Colab ({saved}/{total} answers)"})
                continue
        result = advance(model, arm, dirs, ids, out_root, llm_for(model), extractor, judge, retriever, wait=wait, log=log)
        rows.append({"model": model, "arm": arm,
                     "state": "done" if "done" in result else f"waiting: {result['waiting']} batch"})
    return rows


def heldout_cases(folds=None):
    """The held-out IDs, after the split's sha256 and the flag are checked. Never printed."""
    if hashlib.sha256(splits.FOLDS_PATH.read_bytes()).hexdigest() != FOLDS_SHA256:
        raise Phase4Error("The installed split is not the fixed one. Stop and tell Parsa.")
    if not splits.heldout_allowed():
        raise Phase4Error("Held-out commands need ALLOW_HELDOUT=1 on the command (DEC-31 §4).")
    ids = splits.heldout_ids(folds)
    if len(ids) != EXPECTED_HELDOUT:
        raise Phase4Error(f"{len(ids)} held-out cases, expected {EXPECTED_HELDOUT}.")
    return sorted(ids)


def artifact_dirs(drive=None):
    drive = Path(drive) if drive else local_drive_root()
    if drive is None:
        raise Phase4Error("No Drive folder found; pass --drive or set STROKECOPILOT_DRIVE.")
    return {k: drive / "artifacts" / v for k, v in DIRS.items()}


def check(dirs, ids):
    """Counts: contracts that validate (and name their file), notes that pass their fact check, and the rules'
    decisions on the ground-truth contracts (D3's truth) and on the predicted ones."""
    out = {"cases": len(ids), "valid": {}}
    contracts = {}
    for arm in ("predicted", "groundtruth"):
        contracts[arm] = ha.load_contracts(dirs[arm], ids)  # refuses a case_id that differs from its file name
        for obj in contracts[arm].values():
            validate_vision_output(obj)
        out["valid"][arm] = len(contracts[arm])
    facts = ha.load_facts(dirs["notes"], ids)  # each note checked against its fact sheet
    out["valid"]["notes"] = len(facts)
    word = {True: "yes", False: "no", None: "undetermined"}
    for arm in ("groundtruth", "predicted"):
        decided = [ha.decide(contracts[arm][c], facts[c]) for c in ids]
        out[f"rules_on_{arm}"] = {d: dict(Counter(word[x[d]] for x in decided)) for d in ha.DECISIONS}
    truth = {c: ha.decide(contracts["groundtruth"][c], facts[c]) for c in ids}
    on_pred = {c: ha.decide(contracts["predicted"][c], facts[c]) for c in ids}
    out["rules_on_predicted_vs_truth"] = ha.correctness({c: {"answer": on_pred[c]} for c in ids}, truth)
    ha.check_no_ids(out, ids)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 4's held-out commands (DEC-31 §4). Counts only.")
    parser.add_argument("command", choices=("check", "status", "advance", "import"))
    parser.add_argument("--drive", help="the StrokeCopilot folder on Drive (default: config.local_drive_root)")
    parser.add_argument("--out", default=f"runs/{RUN}")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--arms", help="advance: only these arms (comma-separated)")
    parser.add_argument("--from", dest="src", help="import: the run folder Colab wrote on Drive (default: Drive runs/RUN)")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    ids = heldout_cases()
    dirs = artifact_dirs(args.drive)
    models = [m for m in args.models.split(",") if m]
    if args.command == "check":
        result = check(dirs, ids)
    elif args.command == "status":
        result = status(models, dirs, ids, args.out)
    elif args.command == "advance":
        result = advance_all(models, dirs, ids, args.out, wait=args.wait,
                             only_arms=set(args.arms.split(",")) if args.arms else None, log=lambda *_: None)
    else:
        src = args.src or str(dirs["notes"].parent.parent / "runs" / RUN)
        result = [{"model": m, "arm": a, "added": n} for (m, a), n in import_answers(src, args.out, models).items()]
    ha.check_no_ids(result, ids)
    print(json.dumps(result, indent=1))
    print(f"OK: {args.command} on {len(ids)} held-out cases.")


if __name__ == "__main__":
    main()
