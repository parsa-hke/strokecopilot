"""End to end in fixture mode: fixture -> reasoner -> verifier -> hallucination rate. Plan §7 Phase 2 acceptance.

Each fixture's hand-written vision JSON and triage note go to the reasoner (prompts/reasoner_v1.md with the whole
rulebook, DEC-19). Its recommendation goes to the verifier (claim extractor, code checks, retrieval and the LLM judge
with τ, DEC-20). The traces are scored with nlp/verifier/metrics.py. Criticality and omissions need the rulebook in
code (scenarios.py), so their rates stay None for now.

Phase 4 adds the real-case mode (case_id -> vision models -> vision_output.json -> the same chain).
"""

import json
from pathlib import Path

from strokecopilot.config import FIXTURES_DIR
from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.notes.facts import fixture_note
from strokecopilot.nlp.reasoner import prompt as rp
from strokecopilot.nlp.reasoner import run as rr
from strokecopilot.nlp.reasoner.providers import RefusalError, cost_usd
from strokecopilot.nlp.verifier import metrics as vm
from strokecopilot.nlp.verifier import run as vr
from strokecopilot.nlp.verifier import trace as tr


def _usage(llm):
    return dict(getattr(llm, "usage", {}) or {})


def run_fixtures(reasoner, extractor, judge, retriever, out_dir, stems=None, tau=tr.TAU, log=print, extra_config=None):
    """Writes config.json, metrics.json, outputs/<stem>.json, traces/<stem>.json and calls/<stem>.json; returns the
    metrics. reasoner, extractor and judge are llm(system, user, schema) callables (providers.ClaudeJSON)."""
    out = Path(out_dir)
    for sub in ("outputs", "traces", "calls"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    stems = stems or rr.fixture_stems()
    system = rp.system_prompt()
    expected = json.loads(rr.EXPECTED.read_text())["decisions"]
    corpus = ingest.load()
    cached_extractor = vr.CachedExtractor(extractor, vr.JsonCache(out / "extraction_cache.json"))
    cached_judge = vr.CachedJudge(tr.Judge(judge), vr.JsonCache(out / "judge_cache.json"))
    cached_retriever = vr.CachedRetriever(retriever, vr.JsonCache(out / "retrieval_cache.json"))
    describe = lambda llm: getattr(llm, "describe", lambda: {"model": getattr(llm, "model", "unknown")})()
    config = {"mode": "fixtures", "fixtures": stems, "reasoner": describe(reasoner), "prompt_version": rp.REASONER_PROMPT,
              "extractor": describe(extractor), "judge": describe(judge), "judge_prompt": tr.JUDGE_PROMPT, "tau": tau,
              "top_k": tr.TOP_K, "corpus_sha256": ingest.version(), "corpus_draft": ingest.unchecked(ingest.read()),
              **(extra_config or {})}
    (out / "config.json").write_text(json.dumps(config, indent=1))

    cases, rows = [], []
    for stem in stems:
        vision = json.loads((Path(FIXTURES_DIR) / f"{stem}.json").read_text())
        note, _ = fixture_note(stem)
        rec, record = rr.recommend(vision, note, reasoner, system=system)
        (out / "calls" / f"{stem}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        check = rr.decision_check(rec, expected[stem])
        judged = [ok for _, _, ok in check.values() if ok is not None]
        row = {"fixture": stem, "decisions": check, "error": record["error"]}
        if rec is None:
            cases.append({"case_id": vision["case_id"], "format_failure": True, "traces": [],
                          "decisions_correct": None, "decisions_determined": None})
            rows.append(row)
            log(f"{stem}: reasoner failed ({record['error'][:120]})")
            continue
        (out / "outputs" / f"{stem}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        inputs = vr.fixture_inputs(stem, cached_judge, cached_retriever, tau, corpus)
        try:
            claims, traces = vr.verify(rec, inputs, cached_extractor)
        except (RefusalError, RuntimeError, json.JSONDecodeError, KeyError) as e:
            row["error"] = f"verifier failed: {str(e)[:300]}"
            rows.append(row)
            log(f"{stem}: verifier failed ({str(e)[:120]})")
            continue
        traces = [t.to_dict() for t in traces]
        (out / "traces" / f"{stem}.json").write_text(json.dumps(
            {"claims": [c.to_dict() for c in claims], "traces": traces}, indent=1, ensure_ascii=False))
        flagged = [{"claim_id": t["claim_id"], "subtype": t["subtype_name"], "reason": t["reason"]} for t in traces if t["subtype"] is not None]
        cases.append({"case_id": vision["case_id"], "format_failure": False, "traces": traces,
                      "omissions": None, "uncertainty_omissions": None,
                      "decisions_correct": sum(judged), "decisions_determined": len(judged)})
        row.update(claims=len(traces), hallucinated=len(flagged), flagged=flagged)
        rows.append(row)
        log(f"{stem}: {len(flagged)} of {len(traces)} claims flagged; decisions "
            + ", ".join(f"{d} {'yes' if got else 'no'}" for d, (got, _, _) in check.items()))

    summary = vm.summarise(cases) if cases else None
    costs = {name: cost_usd(getattr(llm, "model", ""), _usage(llm)) for name, llm in
             (("reasoner", reasoner), ("extractor", extractor), ("judge", judge))}
    metrics = {"summary": summary, "per_fixture": rows,
               "usage": {"reasoner": _usage(reasoner), "extractor": _usage(extractor), "judge": _usage(judge)},
               "cost_usd_estimate": costs, "judge_calls": cached_judge.calls, "extractor_calls": cached_extractor.calls}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    return metrics
