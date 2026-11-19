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
from strokecopilot.nlp.notes.facts import FactSheet, fixture_note
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


def real_case(case_id, vision_dir, notes_dir):
    """One real development case: its contract (predicted or ground-truth, whichever folder is given), its generated
    triage note and fact sheet (artifacts/isles24_notes_v1). Expected decisions come from the rule checks on these
    inputs (rules.decide), as for the scenarios: the reasoner is judged on the inputs it was given, so the two vision
    arms can expect different decisions for the same patient, which is what the arm comparison measures."""
    vision = json.loads((Path(vision_dir) / f"{case_id}.json").read_text())
    if vision.get("case_id") != case_id:
        raise ValueError("A contract's case_id does not match its file name.")
    note = (Path(notes_dir) / f"{case_id}.txt").read_text()
    facts = FactSheet.from_json(Path(notes_dir) / f"{case_id}.facts.json")
    facts.check_against(note)
    verdicts = rules.decide(rules.case_from(vision, facts))
    return {"id": case_id, "vision": vision, "note": note, "facts": facts,
            "expected": {d: WORD_TO_BOOL[v.value] for d, v in verdicts.items()}}


def sample_real_ids(vision_dir, n=20, seed=20261115, folds=None):
    """n development case IDs, the same number from each fold (DEC-24: 20 real cases), seeded, sorted. Refuses
    held-out cases (splits.check_ids). The sample is the same for every model and both vision arms."""
    import random

    from strokecopilot.data.splits import check_ids, load_folds

    ids = sorted(p.stem for p in Path(vision_dir).glob("*.json"))
    check_ids(ids, folds)
    folds = folds or load_folds()
    by_fold = {k: sorted(set(v) & set(ids)) for k, v in folds["folds"].items()}
    if n % len(by_fold):
        raise ValueError(f"n={n} does not split evenly over {len(by_fold)} folds.")
    rng = random.Random(seed)
    return sorted(c for k in sorted(by_fold) for c in rng.sample(by_fold[k], n // len(by_fold)))


def load_cases(source="fixtures", ids=None, vision_dir=None, notes_dir=None, folds=None):
    """[{"id", "vision", "note", "facts", "expected"}]; expected maps each decision to True, False or None (either).

    source "real": development cases from vision_dir (artifacts/vision_outputs_v1 for predicted vision,
    artifacts/vision_outputs_gt_v1 for ground truth) and notes_dir (artifacts/isles24_notes_v1); `ids` is required
    (sample_real_ids) and held-out cases are refused."""
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
    elif source == "real":
        from strokecopilot.data.splits import check_ids

        if not ids or vision_dir is None or notes_dir is None:
            raise ValueError("Real cases need ids, vision_dir and notes_dir.")
        check_ids(sorted(ids), folds)
        out = [real_case(case_id, vision_dir, notes_dir) for case_id in sorted(ids)]
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


def _inputs(case, judge, retriever, tau, corpus, rec=None, rulebook=True):
    return tr.Inputs(vision=case["vision"], note=case["note"], facts=case["facts"], corpus=corpus, judge=judge,
                     retriever=retriever, tau=tau, decision_passages=tr.decision_passages(rec) if rec else [],
                     rulebook=rulebook)


def _score(case, rec, claims, traces):
    """The metrics record for one verified output, with criticality and omissions from the rulebook in code."""
    found, missing, _ = rules.annotate(rec, claims, traces, case["vision"], case["facts"])
    check = rr.decision_check(rec, case["expected"])
    judged = [ok for _, _, ok in check.values() if ok is not None]
    record = {"case_id": case["vision"]["case_id"], "format_failure": False, "traces": traces,
              "omissions": len(found), "uncertainty_omissions": len(missing),
              "decisions_correct": sum(judged), "decisions_determined": len(judged)}
    return record, check, found, missing


def run_cases(cases, reasoner, extractor, judge, retriever, out_dir, tau=tr.TAU, log=print, extra_config=None, source=None,
              judge_prompt=tr.JUDGE_PROMPT, version=rp.REASONER_PROMPT, rulebook=True):
    """Writes config.json, metrics.json, flagged.json, and outputs/, traces/ and calls/ per case; returns the metrics.
    reasoner, extractor and judge are llm(system, user, schema) callables (providers.ClaudeJSON). rulebook False is
    the without-rulebook arm (DEC-29): the no-rulebook prompt, and the verifier's placeholder rule."""
    out = Path(out_dir)
    for sub in ("outputs", "traces", "calls"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    system = rp.system_prompt(version, rulebook=rulebook)
    corpus = ingest.load()
    cached_extractor = vr.CachedExtractor(extractor, vr.JsonCache(out / "extraction_cache.json"))
    cached_judge = vr.CachedJudge(tr.Judge(judge, judge_prompt), vr.JsonCache(out / "judge_cache.json"), version=judge_prompt)
    cached_retriever = vr.CachedRetriever(retriever, vr.JsonCache(out / "retrieval_cache.json"))
    describe = lambda llm: getattr(llm, "describe", lambda: {"model": getattr(llm, "model", "unknown")})()
    config = {"source": source, "cases": [c["id"] for c in cases], "reasoner": describe(reasoner),
              "prompt_version": version, "rulebook": rulebook, "extractor": describe(extractor), "judge": describe(judge),
              "judge_prompt": judge_prompt, "tau": tau, "top_k": tr.TOP_K, "corpus_sha256": ingest.version(),
              "corpus_draft": ingest.unchecked(ingest.read()), **(extra_config or {})}
    if source == "scenarios":
        config["scenarios"] = sc.manifest(sc.generate())
    (out / "config.json").write_text(json.dumps(config, indent=1))

    records, rows = [], []
    for case in cases:
        rec, call = rr.recommend(case["vision"], case["note"], reasoner, version=version, system=system)
        (out / "calls" / f"{case['id']}.json").write_text(json.dumps(call, indent=1, ensure_ascii=False))
        row = {"case": case["id"], "error": call["error"], "unwrapped_list": bool(call.get("unwrapped_list")),
               "stripped_ids": call.get("stripped_ids", 0)}
        if rec is None:
            records.append({"case_id": case["vision"]["case_id"], "format_failure": True, "traces": []})
            rows.append(row)
            log(f"{case['id']}: reasoner failed ({call['error'][:120]})")
            continue
        (out / "outputs" / f"{case['id']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        try:
            claims, traces = vr.verify(rec, _inputs(case, cached_judge, cached_retriever, tau, corpus, rec, rulebook),
                                       cached_extractor)
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

    # A batched llm (pipeline/batched.py) carries price_factor 0.5: the Message Batches API is half price.
    costs = {name: (lambda c, f: None if c is None else c * f)(cost_usd(getattr(llm, "model", ""), _usage(llm)),
                                                                getattr(llm, "price_factor", 1.0))
             for name, llm in (("reasoner", reasoner), ("extractor", extractor), ("judge", judge))}
    metrics = {"summary": vm.summarise(records) if records else None, "per_case": rows,
               "usage": {"reasoner": _usage(reasoner), "extractor": _usage(extractor), "judge": _usage(judge)},
               "cost_usd_estimate": costs, "judge_calls": cached_judge.calls, "extractor_calls": cached_extractor.calls,
               "unwrapped_lists": sum(r["unwrapped_list"] for r in rows),  # answers scored from a one-object list
               "stripped_ids": sum(r["stripped_ids"] for r in rows)}  # passage IDs with their [brackets] removed
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    (out / "flagged.json").write_text(json.dumps([f for r in rows for f in r.get("flagged", [])], indent=1, ensure_ascii=False))
    return metrics


def reverify(out_dir, extractor, judge, retriever=None, tau=tr.TAU, log=print, source="fixtures", judge_prompt=tr.JUDGE_PROMPT):
    """Verify a finished run's saved reasoner outputs again with another claim extractor (EXP-24): fresh extraction,
    tracing with the current code, and the judge for verdicts the cache lacks. `out_dir` holds the run's outputs/ and
    calls/ (a copy, so the original keeps its own claims). Writes metrics_reverify.json and flagged_reverify.json."""
    out = Path(out_dir)
    cases = {c["id"]: c for c in load_cases(source)}
    corpus = ingest.load()
    cached_extractor = vr.CachedExtractor(extractor, vr.JsonCache(out / "extraction_cache.json"))
    cached_judge = vr.CachedJudge(tr.Judge(judge, judge_prompt), vr.JsonCache(out / "judge_cache.json"), version=judge_prompt)
    cached_retriever = vr.CachedRetriever(retriever, vr.JsonCache(out / "retrieval_cache.json"))
    (out / "traces_reverify").mkdir(exist_ok=True)
    records, rows, flagged = [], [], []
    for output in sorted((out / "outputs").glob("*.json")):
        case, rec = cases[output.stem], json.loads(output.read_text())
        claims, traces = vr.verify(rec, _inputs(case, cached_judge, cached_retriever, tau, corpus, rec), cached_extractor)
        traces = [t.to_dict() for t in traces]
        (out / "traces_reverify" / output.name).write_text(json.dumps(
            {"claims": [c.to_dict() for c in claims], "traces": traces}, indent=1, ensure_ascii=False))
        record, check, found, missing = _score(case, rec, claims, traces)
        mine = flagged_claims(output.stem, claims, traces)
        flagged += mine
        records.append(record)
        rows.append({"case": output.stem, "claims": len(traces), "hallucinated": len(mine), "flagged": mine, "decisions": check,
                     "omissions": found, "uncertainty_omissions": missing})
        log(f"{output.stem}: {len(mine)} of {len(traces)} claims flagged")
    costs = {name: cost_usd(getattr(llm, "model", ""), _usage(llm)) for name, llm in (("extractor", extractor), ("judge", judge))}
    metrics = {"reverified": True, "source": source, "tau": tau, "judge_prompt": judge_prompt,
               "extractor": getattr(extractor, "model", None), "summary": vm.summarise(records) if records else None,
               "per_case": rows, "cost_usd_estimate": {"reasoner": 0.0, **costs},
               "judge_calls": cached_judge.calls, "extractor_calls": cached_extractor.calls}
    (out / "metrics_reverify.json").write_text(json.dumps(metrics, indent=1, default=str))
    (out / "flagged_reverify.json").write_text(json.dumps(flagged, indent=1, ensure_ascii=False))
    return metrics


def run_fixtures(reasoner, extractor, judge, retriever, out_dir, stems=None, tau=tr.TAU, log=print, extra_config=None):
    """The Phase 2 acceptance run on the five fixtures (EXP-12)."""
    return run_cases(load_cases("fixtures", stems), reasoner, extractor, judge, retriever, out_dir, tau, log,
                     extra_config, source="fixtures")


def traces_dir_for(metrics_name):
    """The traces folder that goes with a metrics file: traces/ for metrics.json, traces_rescore/ for
    metrics_rescore.json, traces_<x>/ for metrics_<x>.json."""
    stem = Path(metrics_name).stem
    return "traces" if stem == "metrics" else "traces_" + stem.removeprefix("metrics_")


def rescore(out_dir, retriever=None, tau=tr.TAU, metrics_name="metrics_rescore.json", log=print, source=None,
            judge_llm=None, judge_prompt=None, flagged_name="flagged_rescore.json", cases=None):
    """Re-score a finished run with the current verifier and rulebook code: the saved reasoner outputs and model
    claims, the regex and citation claims rebuilt. Judge verdicts come from the cache only (no API calls), unless
    `judge_llm` is given: then the judge answers only the verdicts the cache lacks (new candidates, or a new
    `judge_prompt`, whose version is part of the cache key). `retriever` answers retrievals the cache does not hold.
    Writes metrics_rescore.json and flagged_rescore.json (or the names given), and the new traces in traces_rescore/
    (named after the metrics file), so compare.case_records reads each case's claims with the scoring they belong to."""
    out = Path(out_dir)
    trace_dir = out / traces_dir_for(metrics_name)
    trace_dir.mkdir(exist_ok=True)
    config = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    source = source or config.get("source") or config.get("mode") or "fixtures"
    # The run's own arm and judge prompt (DEC-29: the without-rulebook arm scores its placeholder differently).
    rulebook = config.get("rulebook", True)
    judge_prompt = judge_prompt or config.get("judge_prompt") or tr.JUDGE_PROMPT
    # Real cases need their contracts and notes: pass them as `cases` ({id: case} or a list).
    cases = {c["id"]: c for c in (cases.values() if isinstance(cases, dict) else cases)} if cases is not None \
        else {c["id"]: c for c in load_cases(source)}
    corpus = ingest.load()
    judge = vr.CachedJudge(tr.Judge(judge_llm, judge_prompt) if judge_llm else None, vr.JsonCache(out / "judge_cache.json"),
                           offline=judge_llm is None, version=judge_prompt)
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
        traces = [t.to_dict() for t in tr.trace_all(claims, _inputs(case, judge, cached_retriever, tau, corpus, rec, rulebook))]
        record, check, found, missing = _score(case, rec, claims, traces)  # marks criticality on the traces
        (trace_dir / f"{call.stem}.json").write_text(json.dumps(
            {"claims": [c.to_dict() for c in claims], "traces": traces}, indent=1, ensure_ascii=False))
        mine = flagged_claims(call.stem, claims, traces)
        flagged += mine
        records.append(record)
        rows.append({"case": call.stem, "claims": len(traces), "hallucinated": len(mine), "flagged": mine, "decisions": check,
                     "omissions": found, "uncertainty_omissions": missing})
        log(f"{call.stem}: {len(mine)} of {len(traces)} claims flagged" + (f"; omissions: {'; '.join(found)}" if found else ""))
    judge_cost = cost_usd(getattr(judge_llm, "model", ""), _usage(judge_llm)) if judge_llm else 0.0
    metrics = {"rescored": True, "source": source, "rulebook": rulebook, "tau": tau, "judge_prompt": judge_prompt,
               "judge_calls": judge.calls,
               "summary": vm.summarise(records) if records else None, "per_case": rows,
               "cost_usd_estimate": {"reasoner": 0.0, "extractor": 0.0, "judge": judge_cost}}
    (out / metrics_name).write_text(json.dumps(metrics, indent=1, default=str))
    (out / flagged_name).write_text(json.dumps(flagged, indent=1, ensure_ascii=False))
    return metrics
