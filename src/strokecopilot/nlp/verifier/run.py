"""Verifier runs: extract -> trace, with caches, and the seeded-error bake-off. Plan §8.4;
docs/HALLUCINATION_METRIC.md §3, §7, §8.

Caches make the bake-off cheap and resumable:
- extraction: one LLM call per distinct output, keyed by the output's JSON;
- judge: one verdict per (claim text, passage, applied, case), so the same unchanged claim in 14 seeded copies of a
  gold output is judged once, and a smaller candidate set later reuses the verdicts;
- retrieval: the top-5 passages and scores per claim text.

The run judges every top-5 retrieved passage (τ = -1). τ is then set offline by replaying the traces from the caches
for each τ in a grid, which needs no API calls.
"""

import hashlib
import json
from pathlib import Path

from strokecopilot.config import FIXTURES_DIR
from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.notes.facts import fixture_note
from strokecopilot.nlp.reasoner.providers import RefusalError, cost_usd
from strokecopilot.nlp.verifier import claims as cl
from strokecopilot.nlp.verifier import seeded_errors as se
from strokecopilot.nlp.verifier import trace as tr

TAU_GRID = (-1.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


class JsonCache:
    """A dict saved to one JSON file after every new entry, so a stopped run resumes."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.data = json.loads(self.path.read_text()) if self.path and self.path.exists() else {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        if self.path:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False))

    def __len__(self):
        return len(self.data)


class CachedExtractor:
    """Wraps an extractor llm(system, user, schema); identical requests to the same model are answered from the cache.

    The model is part of the key (since 12 Nov 2026), so a run with another extractor never reuses these claims."""

    def __init__(self, llm, cache):
        self.llm, self.cache, self.calls = llm, cache, 0

    def __call__(self, system, user, schema):
        model = getattr(self.llm, "model", None)
        key = _hash([system, user]) if model in (None, "claude-sonnet-5-5") else _hash([model, system, user])
        hit = self.cache.get(key)
        if hit is None:
            hit = self.llm(system, user, schema)
            self.calls += 1
            self.cache.put(key, hit)
        return hit


class CachedJudge:
    """Wraps a tr.Judge. Verdicts are cached per passage; only passages without a verdict go to the model.

    With `offline=True` it never calls the model and a missing verdict is an error (used by the τ replay)."""

    def __init__(self, judge, cache, offline=False, version=tr.JUDGE_PROMPT):
        self.judge, self.cache, self.offline, self.calls, self.version = judge, cache, offline, 0, version

    def key(self, claim_text, pid, case_inputs, applied):
        # The prompt version is part of the key, so a changed judge prompt never reuses old verdicts.
        return _hash([getattr(self.judge, "version", None) or self.version, claim_text, pid, case_inputs, bool(applied)])

    def __call__(self, claim_text, passages, case_inputs, applied):
        keys = {p["id"]: self.key(claim_text, p["id"], case_inputs, applied) for p in passages}
        missing = [p for p in passages if self.cache.get(keys[p["id"]]) is None]
        if missing:
            if self.offline:
                raise KeyError(f"No cached verdict for {[p['id'] for p in missing]} on {claim_text[:60]!r}")
            got = self.judge(claim_text, missing, case_inputs, applied)
            self.calls += 1
            for p in missing:  # a passage the judge skipped counts as not addressed
                self.cache.put(keys[p["id"]], list(got.get(p["id"], ("not_addressed", None))))
        return {pid: tuple(self.cache.get(k)) for pid, k in keys.items()}


class CachedRetriever:
    def __init__(self, retriever, cache, offline=False):
        self.retriever, self.cache, self.offline = retriever, cache, offline

    def __call__(self, text, k):
        key = _hash([text, k])
        hit = self.cache.get(key)
        if hit is None:
            if self.offline:
                raise KeyError(f"No cached retrieval for {text[:60]!r}")
            hit = [[pid, float(score)] for pid, score in self.retriever(text, k)]
            self.cache.put(key, hit)
        return [tuple(x) for x in hit]


def fixture_inputs(stem, judge=None, retriever=None, tau=-1.0, corpus=None, rec=None):
    note, sheet = fixture_note(stem)
    vision = json.loads((Path(FIXTURES_DIR) / f"{stem}.json").read_text())
    return tr.Inputs(vision=vision, note=note, facts=sheet, corpus=corpus or ingest.load(), judge=judge,
                     retriever=retriever, tau=tau, decision_passages=tr.decision_passages(rec) if rec else [])


def verify(rec, inputs, extractor):
    """(claims, traces) for one recommendation."""
    claims = cl.extract(rec, extractor)
    inputs.decision_passages = tr.decision_passages(rec)
    return claims, tr.trace_all(claims, inputs)


def run_seeded(extractor_llm, judges, retriever, out_dir, seeds=None, stems=None, log=print, metrics_name="metrics.json"):
    """The bake-off: every gold output and every seed, traced once per judge model.

    judges: {name: llm callable for tr.Judge}. Writes caches, traces/<judge>/<id>.json and metrics.json, and returns
    the metrics: for each judge, the seeded-error score at every τ in TAU_GRID, the calls made and the usage."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = ingest.load()
    extractor = CachedExtractor(extractor_llm, JsonCache(out / "extraction_cache.json"))
    retrieval = JsonCache(out / "retrieval_cache.json")
    items = _items(stems, seeds)
    metrics = {"outputs": len(items), "judges": {}}
    for name, judge_llm in judges.items():
        cache = JsonCache(out / f"judge_{name}_cache.json")
        judge = CachedJudge(tr.Judge(judge_llm), cache)
        live_retriever = CachedRetriever(retriever, retrieval)
        failures, done = [], {}
        (out / "traces" / name).mkdir(parents=True, exist_ok=True)
        for n, item in enumerate(items, 1):
            inputs = fixture_inputs(item["fixture"], judge, live_retriever, -1.0, corpus)
            try:
                claims, traces = verify(item["rec"], inputs, extractor)
            except (RefusalError, RuntimeError, json.JSONDecodeError, KeyError) as e:
                failures.append({"id": item["id"], "error": str(e)[:300]})
                log(f"{name} {item['id']}: FAILED ({str(e)[:120]})")
                continue
            done[item["id"]] = [c.to_dict() for c in claims]
            (out / "traces" / name / f"{item['id']}.json").write_text(json.dumps(
                {"claims": done[item["id"]], "traces": [t.to_dict() for t in traces]}, indent=1, ensure_ascii=False))
            if n % 10 == 0 or n == len(items):
                log(f"{name}: {n}/{len(items)} outputs; judge calls {judge.calls}, extractor calls {extractor.calls}")
        grid = replay(items, done, out, name, corpus, retriever)
        usage = dict(getattr(judge_llm, "usage", {}) or {})
        metrics["judges"][name] = {"failures": failures, "judge_calls": judge.calls, "verdicts_cached": len(cache),
                                   "usage": usage, "cost_usd_estimate": cost_usd(getattr(judge_llm, "model", name), usage),
                                   "by_tau": grid}
    ex_usage = dict(getattr(extractor_llm, "usage", {}) or {})
    metrics["extractor"] = {"calls": extractor.calls, "usage": ex_usage,
                            "cost_usd_estimate": cost_usd(getattr(extractor_llm, "model", ""), ex_usage)}
    (out / metrics_name).write_text(json.dumps(metrics, indent=1))
    return metrics


def refresh_claims(rec, saved):
    """The model's extracted claims as saved, plus the regex and citation claims rebuilt with the current code."""
    llm, _ = cl.restatements([cl.Claim(**c) for c in saved if c.get("origin", "llm") == "llm"], rec)
    return llm + cl.regex_claims(rec, llm) + cl.citation_claims(rec)


def replay(items, claims_by_id, out_dir, judge_name, corpus=None, retriever=None, judge=None):
    """Seeded-error scores at every τ in TAU_GRID. Judge verdicts come from the cache only, unless `judge` (a
    CachedJudge that may call the model for missing verdicts) is given; retrieval comes from the cache, or from
    `retriever` (no API) for a text not retrieved before."""
    out = Path(out_dir)
    corpus = corpus or ingest.load()
    judge = judge or CachedJudge(None, JsonCache(out / f"judge_{judge_name}_cache.json"), offline=True)
    retriever = CachedRetriever(retriever, JsonCache(out / "retrieval_cache.json"), offline=retriever is None)
    grid = {}
    for tau in TAU_GRID:
        seeds, clean = [], []
        for item in items:
            if item["id"] not in claims_by_id:
                continue
            claims = refresh_claims(item["rec"], claims_by_id[item["id"]])
            traces = [t.to_dict() for t in tr.trace_all(claims, fixture_inputs(item["fixture"], judge, retriever, tau, corpus, item["rec"]))]
            if item["subtype"] is None:
                clean.append({"fixture": item["fixture"], "traces": traces})
            else:
                seeds.append({"id": item["id"], "subtype": item["subtype"], "label": item["label"], "traces": traces})
        grid[str(tau)] = se.score(seeds, clean)
    return grid


def _items(stems=None, seeds=None):
    items = [{"id": f"{s[:4]}-gold", "fixture": s, "subtype": None, "rec": se.load_gold(s), "label": None}
             for s in (stems or se.gold_stems())]
    return items + [x for x in se.seeded_set(seeds) if stems is None or x["fixture"] in stems]


def rescore(out_dir, judge_names, stems=None, seeds=None, metrics_name="metrics_rescore.json", retriever=None,
            judge_llms=None, judge_prompt=tr.JUDGE_PROMPT):
    """Re-score an earlier run with the current tracing code, from its saved model claims and the caches.

    With no `judge_llms` it makes no API calls: a claim that needs a verdict that is not cached raises KeyError.
    `judge_llms` ({name: llm}) lets the judge answer only the verdicts the cache lacks, for example new candidate
    passages or a new `judge_prompt` (the version is part of the cache key). `retriever` (the local embedding model)
    answers retrievals the cache does not hold."""
    out = Path(out_dir)
    corpus = ingest.load()
    items = _items(stems, seeds)
    metrics = {"outputs": len(items), "rescored": True, "judge_prompt": judge_prompt, "judges": {}}
    for name in judge_names:
        claims_by_id = {}
        for item in items:
            path = out / "traces" / name / f"{item['id']}.json"
            if path.exists():
                claims_by_id[item["id"]] = json.loads(path.read_text())["claims"]
        llm = (judge_llms or {}).get(name)
        cache = JsonCache(out / f"judge_{name}_cache.json")
        judge = CachedJudge(tr.Judge(llm, judge_prompt) if llm else None, cache, offline=llm is None, version=judge_prompt)
        grid = replay(items, claims_by_id, out, name, corpus, retriever, judge)
        usage = dict(getattr(llm, "usage", {}) or {}) if llm else {}
        metrics["judges"][name] = {"failures": [], "judge_calls": judge.calls, "verdicts_cached": len(cache), "usage": usage,
                                   "cost_usd_estimate": cost_usd(getattr(llm, "model", name), usage) if llm else 0.0,
                                   "outputs_rescored": len(claims_by_id), "by_tau": grid}
    metrics["extractor"] = {"calls": 0, "usage": {}, "cost_usd_estimate": 0.0}
    (out / metrics_name).write_text(json.dumps(metrics, indent=1))
    return metrics
