"""Phase 5 task F (DEC-38): EXP-41's 100 human-labelled claims re-judged by a judge from none of the reasoners' families.

The verifier is unchanged except for its LLM judge: `openai/gpt-6-luna` through OpenRouter, with `judge_v3`, τ 0.6 and
reasoning effort medium (Haiku ran at medium). Each labelled claim's case is rescored in a copy of its cell with an
**empty judge cache**, so no Haiku verdict is reused; the code checks (numbers, notes, citations) and the retrieval
cache are the same. Then, per claim: the new judge's verdict, Haiku's verdict (the rescored one the report uses), and
the human consensus.

Read-outs: each judge's agreement with the human consensus (unweighted and weighted by the sampling strata, as EXP-41),
per reasoner model; agreement and Cohen's κ between the two judges. **Must be said with it:** metric points 19 and 20
were adjusted on these same labels (EXP-41), which favours whichever judge the metric was tuned with.

Before any call, `check_haiku` must reproduce EXP-41's reported verifier-human agreement from the saved traces.
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

LABELS = Path("runs/EXP-41_labelling")
JUDGE_MODEL = "openai/gpt-6-luna"
PRICE = {"input": 0.10, "output": 0.50}  # USD per million tokens, OpenRouter list price read 20 Nov 2026
REPORTED_AGREEMENT = 0.61  # EXP-41 verifier-human agreement on all 100, rescored (points 19-20)
VISION_DIRS = {"real_predicted": "vision_outputs_v1", "real_groundtruth": "vision_outputs_gt_v1"}


def labelled_rows():
    from strokecopilot.nlp.verifier import adjudicate as adj
    return adj.read_labels(LABELS / "labels" / "final.csv", LABELS / "key.csv")


def claim_verdict(cell, case, claim_id, traces_dir):
    from strokecopilot.nlp.verifier import adjudicate as adj
    saved = json.loads((Path(cell) / traces_dir / f"{case}.json").read_text())
    trace = next((t for t in saved["traces"] if t["claim_id"] == claim_id), None)
    if trace is None:
        raise KeyError(f"claim {claim_id} not in {traces_dir}/{case}.json")
    return adj.verdict_of(trace.get("subtype_name") if trace.get("subtype") is not None else None)


def haiku_verdicts(rows):
    """{sample_id: verdict} from each cell's newest scoring (traces_rescore), as the report uses."""
    from strokecopilot.pipeline import compare as cp
    out = {}
    for r in rows:
        name = cp._metrics_file(Path(r["run_folder"])).name
        out[r["sample_id"]] = claim_verdict(r["run_folder"], r["case"], r["claim_id"], cp._traces_dir_for(name))
    return out


def agreement(rows, verdicts):
    """Agreement with the human consensus (flagged = error), unweighted and weighted, overall and per model."""
    from strokecopilot.nlp.verifier import adjudicate as adj

    def agree(sel):
        sel = [r for r in sel if adj.consensus_of(r) is not None]
        if not sel:
            return {"n": 0, "agreement": None, "agreement_weighted": None}
        hits = [(adj.consensus_of(r) == "error") == (verdicts[r["sample_id"]] != "clean") for r in sel]
        w = [r["weight"] for r in sel]
        return {"n": len(sel), "agreement": sum(hits) / len(sel),
                "agreement_weighted": sum(x for x, h in zip(w, hits) if h) / sum(w)}
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    return {"all": agree(rows), "per_model": {m: agree(v) for m, v in sorted(by.items())}}


def judge_vs_judge(rows, a, b):
    from strokecopilot.nlp.verifier import adjudicate as adj
    flag_a = [a[r["sample_id"]] != "clean" for r in rows]
    flag_b = [b[r["sample_id"]] != "clean" for r in rows]
    same = sum(x == y for x, y in zip(flag_a, flag_b))
    return {"n": len(rows), "agreement": same / len(rows),
            "kappa": adj.cohen_kappa(["error" if x else "no_error" for x in flag_a], ["error" if x else "no_error" for x in flag_b]),
            "flagged_by_a": sum(flag_a), "flagged_by_b": sum(flag_b),
            "same_verdict_3way": sum(a[r["sample_id"]] == b[r["sample_id"]] for r in rows)}


def check_haiku(rows, tol=0.005):
    got = agreement(rows, haiku_verdicts(rows))["all"]["agreement"]
    if abs(got - REPORTED_AGREEMENT) > tol:
        raise ValueError(f"Haiku's verdicts give agreement {got:.3f}, not EXP-41's {REPORTED_AGREEMENT}. Stop and check.")
    return got


def make_llm():
    from strokecopilot.nlp.reasoner.providers import VLLMJSON
    return VLLMJSON(JUDGE_MODEL, base_url="https://openrouter.ai/api/v1", key_name="OPENROUTER_API_KEY",
                    max_tokens=16000, extra_body={"reasoning": {"effort": "medium"}}, provider="openrouter",
                    billing="per token (API)")


def _cases_for(arm, ids, drive):
    from strokecopilot.pipeline import end_to_end as e2e
    if arm in VISION_DIRS:
        return e2e.load_cases("real", ids=ids, vision_dir=drive / "artifacts" / VISION_DIRS[arm],
                              notes_dir=drive / "artifacts" / "isles24_notes_v1")
    return None


def rejudge(rows, work, llm, drive, limit=None, log=print):
    """Copy each labelled case into a fresh cell (empty judge cache) and rescore it with `llm` as the judge."""
    from strokecopilot.pipeline import end_to_end as e2e
    work = Path(work)
    by_cell = defaultdict(set)
    for r in rows:
        by_cell[r["run_folder"]].add(r["case"])
    pairs = sorted((cell, case) for cell, cases in by_cell.items() for case in cases)
    if limit:
        keep = set(pairs[:limit])
        by_cell = defaultdict(set)
        for cell, case in keep:
            by_cell[cell].add(case)
    for cell, cases in sorted(by_cell.items()):
        src = Path(cell)
        dst = work / src.parent.name / src.name
        for sub in ("calls", "outputs", "traces"):
            (dst / sub).mkdir(parents=True, exist_ok=True)
            for case in cases:
                if (src / sub / f"{case}.json").exists() and not (dst / sub / f"{case}.json").exists():
                    shutil.copyfile(src / sub / f"{case}.json", dst / sub / f"{case}.json")
        for name in ("config.json", "retrieval_cache.json"):
            if not (dst / name).exists():
                shutil.copyfile(src / name, dst / name)
        e2e.rescore(dst, judge_llm=llm, cases=_cases_for(src.name, sorted(cases), drive),
                    metrics_name="metrics_rejudge.json", flagged_name="flagged_rejudge.json", log=lambda *a: None)
        log(f"{src.parent.name}/{src.name}: {len(cases)} cases rescored")
    return by_cell


def new_verdicts(rows, work, done):
    out = {}
    for r in rows:
        cell = Path(r["run_folder"])
        if r["case"] in done.get(r["run_folder"], set()):
            out[r["sample_id"]] = claim_verdict(Path(work) / cell.parent.name / cell.name, r["case"], r["claim_id"],
                                                "traces_rejudge")
    return out


def cost(usage):
    return (usage.get("input_tokens", 0) * PRICE["input"] + usage.get("output_tokens", 0) * PRICE["output"]) / 1e6


def main(argv=None):
    import argparse

    from strokecopilot.config import local_drive_root
    ap = argparse.ArgumentParser(description="Task F: re-judge EXP-41's claims with a non-Claude judge")
    ap.add_argument("--out", default="runs/EXP-81_rejudge")
    ap.add_argument("--limit", type=int, default=None, help="dummy: only the first N case-cells")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    rows = labelled_rows()
    haiku_agreement = check_haiku(rows)
    llm = make_llm()
    done = rejudge(rows, out / "cells", llm, local_drive_root(), limit=args.limit)
    new = new_verdicts(rows, out / "cells", done)
    sel = [r for r in rows if r["sample_id"] in new]
    haiku = haiku_verdicts(sel)
    result = {"experiment": "EXP-81", "judge": JUDGE_MODEL, "claims": len(sel), "haiku_check": haiku_agreement,
              "usage": dict(llm.usage), "cost_usd_estimate": round(cost(llm.usage), 4),
              "new_judge_vs_humans": agreement(sel, new), "haiku_vs_humans": agreement(sel, haiku),
              "judge_vs_judge": judge_vs_judge(sel, haiku, new),
              "note": "Metric points 19 and 20 were adjusted on these labels (EXP-41)."}
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result[k] for k in ("claims", "cost_usd_estimate", "judge_vs_judge")}, indent=1))
    print("new judge vs humans:", json.dumps(result["new_judge_vs_humans"]["all"]),
          "| Haiku vs humans:", json.dumps(result["haiku_vs_humans"]["all"]))


if __name__ == "__main__":
    main()
