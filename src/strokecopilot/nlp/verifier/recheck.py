"""Re-check the verifier after a change (EXP-23, Phase 3 fixes): the EXP-11 seeded-error bar and the EXP-12 fixture
rescore, once per judge prompt, from the saved claims and caches.

The judge answers only the verdicts the caches lack (new candidate passages, or a new prompt version, which is part
of the cache key); everything else is replayed. Writes, in each source folder, `metrics_<tag>.json` (and for EXP-12
`flagged_<tag>.json`), and in `out_dir` one `metrics.json` comparing the prompts.
"""

import json
from pathlib import Path

from strokecopilot.nlp.verifier import run as vr
from strokecopilot.nlp.verifier import trace as tr

JUDGE_NAME = "claude-haiku-5-5"  # EXP-11's cache name for the DEC-20 judge


def seeded_summary(metrics, tau=tr.TAU, judge_name=JUDGE_NAME):
    """The EXP-11 bar at τ: strict recall per sub-type, its minimum, and the false-flag rate on clean outputs."""
    grid = metrics["judges"][judge_name]["by_tau"][str(tau)]
    recall = {name: r["strict"] for name, r in grid["recall"].items()}
    return {"recall": recall, "min_recall": min(recall.values()), "false_flag_rate": grid["false_flag_rate_claims"],
            "meets_bar": grid["meets_bar"], "missed": {n: r["missed"] for n, r in grid["recall"].items() if r["missed"]},
            "judge_calls": metrics["judges"][judge_name]["judge_calls"],
            "cost_usd_estimate": metrics["judges"][judge_name]["cost_usd_estimate"]}


def fixture_summary(metrics):
    s = metrics["summary"]
    pick = lambda k: {key: s[k][key] for key in ("value", "numerator", "denominator")}
    return {"claims": s["claims"], "hallucination_rate": pick("hallucination_rate"),
            "content_hallucination_rate": pick("content_hallucination_rate"),
            "critical_hallucination_rate": pick("critical_hallucination_rate"),
            "source_attribution_accuracy": pick("source_attribution_accuracy"),
            "recommendation_correctness": pick("recommendation_correctness"),
            "subtypes": {k: v["count"] for k, v in s["subtype_breakdown"].items() if v["count"]},
            "judge_calls": metrics.get("judge_calls", 0), "cost_usd_estimate": metrics["cost_usd_estimate"]["judge"]}


def recheck(exp11_dir, exp12_dir, out_dir, judge_factory, prompts=("judge_v2", "judge_v3"), retriever=None, log=print,
            extra=None):
    """judge_factory() -> a fresh judge llm (so each prompt's usage and cost are counted apart)."""
    from strokecopilot.pipeline import end_to_end as e2e  # imported here: end_to_end imports this package

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"prompts": list(prompts), "tau": tr.TAU, **(extra or {}), "by_prompt": {}}
    for prompt in prompts:
        tag = f"phase3_{prompt}"
        llm = judge_factory()
        seeded = vr.rescore(exp11_dir, [JUDGE_NAME], judge_llms={JUDGE_NAME: llm}, judge_prompt=prompt,
                            metrics_name=f"metrics_{tag}.json", retriever=retriever)
        llm = judge_factory()
        fixtures = e2e.rescore(exp12_dir, retriever=retriever, judge_llm=llm, judge_prompt=prompt, log=lambda *a: None,
                               metrics_name=f"metrics_{tag}.json", flagged_name=f"flagged_{tag}.json")
        result["by_prompt"][prompt] = {"exp11": seeded_summary(seeded), "exp12": fixture_summary(fixtures)}
        b, f = result["by_prompt"][prompt]["exp11"], result["by_prompt"][prompt]["exp12"]
        log(f"{prompt}: EXP-11 min recall {b['min_recall']:.2f}, false flags {b['false_flag_rate']:.3f}, bar "
            f"{'met' if b['meets_bar'] else 'NOT met'}; EXP-12 {f['hallucination_rate']['numerator']} of {f['claims']} claims "
            f"flagged (rate {f['hallucination_rate']['value']:.3f}), critical {f['critical_hallucination_rate']['value']:.3f}; "
            f"judge calls {b['judge_calls'] + f['judge_calls']}.")
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    return result
