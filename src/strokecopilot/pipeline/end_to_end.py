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


def flagged_claims(stem, claims, traces):
    """A compact record of every flagged claim, to read without the full trace files."""
    by_id = {c["id"] if isinstance(c, dict) else c.id: (c if isinstance(c, dict) else c.to_dict()) for c in claims}
    out = []
    for t in traces:
        if t["subtype"] is None:
            continue
        c = by_id.get(t["claim_id"], {})
        sources = [s.get("field") or s.get("id") or (s.get("quote") or "")[:40] for s in c.get("declared_sources", [])]
        out.append({"fixture": stem, "claim_id": t["claim_id"], "subtype": t["subtype_name"], "reason": t["reason"],
                    "kind": c.get("kind"), "subject": c.get("subject"), "value": c.get("value"), "unit": c.get("unit"),
                    "applies": c.get("applies_to_patient"), "text": (c.get("text") or "")[:240], "sources": sources})
    return out


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
        flagged = flagged_claims(stem, claims, traces)
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
    (out / "flagged.json").write_text(json.dumps([f for r in rows for f in r.get("flagged", [])], indent=1, ensure_ascii=False))
    return metrics


def rescore(out_dir, retriever=None, tau=tr.TAU, metrics_name="metrics_rescore.json", log=print):
    """Re-score a finished fixture run with the current verifier code: the saved reasoner outputs and model claims,
    the regex and citation claims rebuilt, judge verdicts from the cache only (no API calls). `retriever` answers
    retrievals the cache does not hold. Writes metrics_rescore.json and flagged_rescore.json."""
    out = Path(out_dir)
    expected = json.loads(rr.EXPECTED.read_text())["decisions"]
    corpus = ingest.load()
    judge = vr.CachedJudge(None, vr.JsonCache(out / "judge_cache.json"), offline=True)
    cached_retriever = vr.CachedRetriever(retriever, vr.JsonCache(out / "retrieval_cache.json"), offline=retriever is None)
    cases, rows, flagged = [], [], []
    for call in sorted((out / "calls").glob("*.json")):
        stem = call.stem
        output, saved = out / "outputs" / f"{stem}.json", out / "traces" / f"{stem}.json"
        if not output.exists():
            cases.append({"case_id": stem, "format_failure": True, "traces": []})
            rows.append({"fixture": stem, "error": json.loads(call.read_text())["error"]})
            continue
        rec = json.loads(output.read_text())
        claims = vr.refresh_claims(rec, json.loads(saved.read_text())["claims"])
        traces = [t.to_dict() for t in tr.trace_all(claims, vr.fixture_inputs(stem, judge, cached_retriever, tau, corpus))]
        check = rr.decision_check(rec, expected[stem])
        judged = [ok for _, _, ok in check.values() if ok is not None]
        mine = flagged_claims(stem, claims, traces)
        flagged += mine
        cases.append({"case_id": rec["case_id"], "format_failure": False, "traces": traces, "omissions": None,
                      "uncertainty_omissions": None, "decisions_correct": sum(judged), "decisions_determined": len(judged)})
        rows.append({"fixture": stem, "claims": len(traces), "hallucinated": len(mine), "flagged": mine, "decisions": check})
        log(f"{stem}: {len(mine)} of {len(traces)} claims flagged")
    metrics = {"rescored": True, "tau": tau, "summary": vm.summarise(cases) if cases else None, "per_fixture": rows,
               "cost_usd_estimate": {"reasoner": 0.0, "extractor": 0.0, "judge": 0.0}}
    (out / metrics_name).write_text(json.dumps(metrics, indent=1, default=str))
    (out / "flagged_rescore.json").write_text(json.dumps(flagged, indent=1, ensure_ascii=False))
    return metrics
