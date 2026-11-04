"""End to end: case -> reasoner -> verifier -> metrics. Plan §7 Phase 2 acceptance; Phase 3 evaluation.

Two sources of cases, run the same way:
- `fixtures`: the five hand-written fixtures, with the hand-checked expected decisions;
- `scenarios`: the synthetic set from nlp/verifier/scenarios.py (DEC-11), with decisions from the rulebook in code.

Each case's vision JSON and triage note go to the reasoner (prompts/reasoner_v1.md, whole rulebook, DEC-19). Its
recommendation goes to the verifier (claim extractor, code checks, retrieval and the LLM judge with τ, DEC-20). The
rulebook in code (nlp/verifier/rules.py) then marks each hallucination critical or not and counts omissions, and
nlp/verifier/metrics.py scores the lot.

Phase 4 adds real cases (case_id -> vision models -> vision_output.json -> the same chain).
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
from strokecopilot.nlp.verifier import rules
from strokecopilot.nlp.verifier import run as vr
from strokecopilot.nlp.verifier import scenarios as sc
from strokecopilot.nlp.verifier import trace as tr

WORD_TO_BOOL = {rules.YES: True, rules.NO: False, rules.UNDETERMINED: None}


def _usage(llm):
    return dict(getattr(llm, "usage", {}) or {})


def load_cases(source="fixtures", ids=None):
    """[{"id", "vision", "note", "facts", "expected"}]; expected maps each decision to True, False or None (either)."""
    if source == "fixtures":
        expected = json.loads(rr.EXPECTED.read_text())["decisions"]
        out = []
        for stem in rr.fixture_stems():
            note, facts = fixture_note(stem)
            out.append({"id": stem, "vision": json.loads((Path(FIXTURES_DIR) / f"{stem}.json").read_text()),
                        "note": note, "facts": facts, "expected": expected[stem]})
    elif source == "scenarios":
        out = [{"id": s.id, "vision": s.vision, "note": s.note, "facts": s.facts,
                "expected": {d: WORD_TO_BOOL[v["value"]] for d, v in s.expected.items()}} for s in sc.generate()]
    else:
        raise ValueError(f"Unknown source {source!r}")
    return [c for c in out if ids is None or c["id"] in ids]


def flagged_claims(case_id, claims, traces):
    """A compact record of every flagged claim, to read without the full trace files."""
    by_id = {c["id"] if isinstance(c, dict) else c.id: (c if isinstance(c, dict) else c.to_dict()) for c in claims}
    out = []
    for t in traces:
        if t["subtype"] is None:
            continue
        c = by_id.get(t["claim_id"], {})
        sources = [s.get("field") or s.get("id") or (s.get("quote") or "")[:40] for s in c.get("declared_sources", [])]
        out.append({"case": case_id, "claim_id": t["claim_id"], "subtype": t["subtype_name"], "critical": t.get("critical"),
                    "reason": t["reason"], "kind": c.get("kind"), "subject": c.get("subject"), "value": c.get("value"),
                    "unit": c.get("unit"), "applies": c.get("applies_to_patient"), "text": (c.get("text") or "")[:240],
                    "sources": sources})
    return out


def _inputs(case, judge, retriever, tau, corpus):
    return tr.Inputs(vision=case["vision"], note=case["note"], facts=case["facts"], corpus=corpus, judge=judge,
                     retriever=retriever, tau=tau)


def _score(case, rec, claims, traces):
    """The metrics record for one verified output, with criticality and omissions from the rulebook in code."""
    found, missing, _ = rules.annotate(rec, claims, traces, case["vision"], case["facts"])
    check = rr.decision_check(rec, case["expected"])
    judged = [ok for _, _, ok in check.values() if ok is not None]
    record = {"case_id": case["vision"]["case_id"], "format_failure": False, "traces": traces,
              "omissions": len(found), "uncertainty_omissions": len(missing),
              "decisions_correct": sum(judged), "decisions_determined": len(judged)}
    return record, check, found, missing


def run_cases(cases, reasoner, extractor, judge, retriever, out_dir, tau=tr.TAU, log=print, extra_config=None, source=None):
    """Writes config.json, metrics.json, flagged.json, and outputs/, traces/ and calls/ per case; returns the metrics.
    reasoner, extractor and judge are llm(system, user, schema) callables (providers.ClaudeJSON)."""
    out = Path(out_dir)
    for sub in ("outputs", "traces", "calls"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    system = rp.system_prompt()
    corpus = ingest.load()
    cached_extractor = vr.CachedExtractor(extractor, vr.JsonCache(out / "extraction_cache.json"))
    cached_judge = vr.CachedJudge(tr.Judge(judge), vr.JsonCache(out / "judge_cache.json"))
    cached_retriever = vr.CachedRetriever(retriever, vr.JsonCache(out / "retrieval_cache.json"))
    describe = lambda llm: getattr(llm, "describe", lambda: {"model": getattr(llm, "model", "unknown")})()
    config = {"source": source, "cases": [c["id"] for c in cases], "reasoner": describe(reasoner),
              "prompt_version": rp.REASONER_PROMPT, "extractor": describe(extractor), "judge": describe(judge),
              "judge_prompt": tr.JUDGE_PROMPT, "tau": tau, "top_k": tr.TOP_K, "corpus_sha256": ingest.version(),
              "corpus_draft": ingest.unchecked(ingest.read()), **(extra_config or {})}
    if source == "scenarios":
        config["scenarios"] = sc.manifest(sc.generate())
    (out / "config.json").write_text(json.dumps(config, indent=1))

    records, rows = [], []
    for case in cases:
        rec, call = rr.recommend(case["vision"], case["note"], reasoner, system=system)
        (out / "calls" / f"{case['id']}.json").write_text(json.dumps(call, indent=1, ensure_ascii=False))
        row = {"case": case["id"], "error": call["error"]}
        if rec is None:
            records.append({"case_id": case["vision"]["case_id"], "format_failure": True, "traces": []})
            rows.append(row)
            log(f"{case['id']}: reasoner failed ({call['error'][:120]})")
            continue
        (out / "outputs" / f"{case['id']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        try:
            claims, traces = vr.verify(rec, _inputs(case, cached_judge, cached_retriever, tau, corpus), cached_extractor)
        except (RefusalError, RuntimeError, json.JSONDecodeError, KeyError) as e:
            row["error"] = f"verifier failed: {str(e)[:300]}"
            rows.append(row)
            log(f"{case['id']}: verifier failed ({str(e)[:120]})")
            continue
        traces = [t.to_dict() for t in traces]
        record, check, found, missing = _score(case, rec, claims, traces)
        (out / "traces" / f"{case['id']}.json").write_text(json.dumps(
            {"claims": [c.to_dict() for c in claims], "traces": traces}, indent=1, ensure_ascii=False))
        flagged = flagged_claims(case["id"], claims, traces)
        records.append(record)
        row.update(claims=len(traces), hallucinated=len(flagged), flagged=flagged, decisions=check, omissions=found,
                   uncertainty_omissions=missing)
        rows.append(row)
        log(f"{case['id']}: {len(flagged)} of {len(traces)} claims flagged; decisions "
            + ", ".join(f"{d} {'yes' if got else 'no'}" for d, (got, _, _) in check.items())
            + (f"; omissions: {'; '.join(found)}" if found else ""))

    costs = {name: cost_usd(getattr(llm, "model", ""), _usage(llm)) for name, llm in
             (("reasoner", reasoner), ("extractor", extractor), ("judge", judge))}
    metrics = {"summary": vm.summarise(records) if records else None, "per_case": rows,
               "usage": {"reasoner": _usage(reasoner), "extractor": _usage(extractor), "judge": _usage(judge)},
               "cost_usd_estimate": costs, "judge_calls": cached_judge.calls, "extractor_calls": cached_extractor.calls}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    (out / "flagged.json").write_text(json.dumps([f for r in rows for f in r.get("flagged", [])], indent=1, ensure_ascii=False))
    return metrics


def run_fixtures(reasoner, extractor, judge, retriever, out_dir, stems=None, tau=tr.TAU, log=print, extra_config=None):
    """The Phase 2 acceptance run on the five fixtures (EXP-12)."""
    return run_cases(load_cases("fixtures", stems), reasoner, extractor, judge, retriever, out_dir, tau, log,
                     extra_config, source="fixtures")


def rescore(out_dir, retriever=None, tau=tr.TAU, metrics_name="metrics_rescore.json", log=print, source=None):
    """Re-score a finished run with the current verifier and rulebook code: the saved reasoner outputs and model
    claims, the regex and citation claims rebuilt, judge verdicts from the cache only (no API calls). `retriever`
    answers retrievals the cache does not hold. Writes metrics_rescore.json and flagged_rescore.json."""
    out = Path(out_dir)
    config = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    source = source or config.get("source") or config.get("mode") or "fixtures"
    cases = {c["id"]: c for c in load_cases(source)}
    corpus = ingest.load()
    judge = vr.CachedJudge(None, vr.JsonCache(out / "judge_cache.json"), offline=True)
    cached_retriever = vr.CachedRetriever(retriever, vr.JsonCache(out / "retrieval_cache.json"), offline=retriever is None)
    records, rows, flagged = [], [], []
    for call in sorted((out / "calls").glob("*.json")):
        case = cases[call.stem]
        output, saved = out / "outputs" / f"{call.stem}.json", out / "traces" / f"{call.stem}.json"
        if not output.exists():
            records.append({"case_id": case["vision"]["case_id"], "format_failure": True, "traces": []})
            rows.append({"case": call.stem, "error": json.loads(call.read_text())["error"]})
            continue
        if not saved.exists():
            rows.append({"case": call.stem, "error": "verifier did not finish in the original run"})
            continue
        rec = json.loads(output.read_text())
        claims = vr.refresh_claims(rec, json.loads(saved.read_text())["claims"])
        traces = [t.to_dict() for t in tr.trace_all(claims, _inputs(case, judge, cached_retriever, tau, corpus))]
        record, check, found, missing = _score(case, rec, claims, traces)
        mine = flagged_claims(call.stem, claims, traces)
        flagged += mine
        records.append(record)
        rows.append({"case": call.stem, "claims": len(traces), "hallucinated": len(mine), "flagged": mine, "decisions": check,
                     "omissions": found, "uncertainty_omissions": missing})
        log(f"{call.stem}: {len(mine)} of {len(traces)} claims flagged" + (f"; omissions: {'; '.join(found)}" if found else ""))
    metrics = {"rescored": True, "source": source, "tau": tau, "summary": vm.summarise(records) if records else None,
               "per_case": rows, "cost_usd_estimate": {"reasoner": 0.0, "extractor": 0.0, "judge": 0.0}}
    (out / metrics_name).write_text(json.dumps(metrics, indent=1, default=str))
    (out / "flagged_rescore.json").write_text(json.dumps(flagged, indent=1, ensure_ascii=False))
    return metrics
