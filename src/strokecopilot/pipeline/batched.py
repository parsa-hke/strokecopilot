"""Paid runs in two passes, so the Claude calls can go through the Message Batches API (DEC-24, amended 12 Nov 2026).

1. Reasoner: every case's request is answered first, by a ClaudeBatch (Sonnet, half price) or by calling any other llm
   directly (the open models on a pod). The answers are kept by request hash.
2. Claim extraction: the outputs that parse are known now, so their extraction requests go to the extractor as one
   batch. The replies go into the run folder's extraction cache under the key run_cases' CachedExtractor looks up.
3. run_cases runs as usual, with the reasoner and extractor answered from those results: the same files, metrics and
   checks as a direct run, and only the judge (cheap Haiku calls, one per claim) is called live.

Costs: a batched llm carries PRICE_FACTOR 0.5, which run_cases applies to the estimate.
"""

import json
from pathlib import Path

from strokecopilot.nlp.reasoner import prompt as rp
from strokecopilot.nlp.reasoner import run as rr
from strokecopilot.nlp.reasoner.providers import RefusalError
from strokecopilot.nlp.verifier import claims as cl
from strokecopilot.nlp.verifier import run as vr
from strokecopilot.pipeline import end_to_end as e2e


def request_key(system, user):
    """The key a request is stored under: the one CachedExtractor uses for Sonnet, 16 hex characters."""
    return vr._hash([system, user])


class Prefilled:
    """An llm(system, user, schema) answered from results gathered beforehand. It keeps the real model's name and
    describe(), its usage and its price factor, so run_cases records and costs the run as it really ran."""

    def __init__(self, results, source):
        self.results, self.source = results, source
        self.model = getattr(source, "model", "unknown")
        self.usage = getattr(source, "usage", {})
        self.price_factor = getattr(source, "PRICE_FACTOR", 1.0)

    def describe(self):
        describe = getattr(self.source, "describe", None)
        return describe() if describe else {"model": self.model}

    def __call__(self, system, user, schema):
        result = self.results.get(request_key(system, user))
        if result is None:
            raise RuntimeError("No answer gathered for this request.")
        if "reply" in result:
            return result["reply"]
        if result.get("refusal"):
            raise RefusalError(result["error"])
        raise RuntimeError(result["error"])


def gather(requests, llm, log=print):
    """{key: result} for {key: (system, user, schema)}: one batch if the llm has run(), else one call per request."""
    if not requests:
        return {}
    if hasattr(llm, "run"):
        return llm.run(requests, log=log)
    out = {}
    for key, req in requests.items():
        try:
            out[key] = {"reply": llm(*req)}
        except RefusalError as e:
            out[key] = {"error": str(e), "refusal": True}
        except (RuntimeError, json.JSONDecodeError) as e:
            out[key] = {"error": str(e), "refusal": False}
    return out


def run_batched(cases, reasoner, extractor, judge, retriever, out_dir, log=print, source=None, version=rp.REASONER_PROMPT,
                extra_config=None, **run_kwargs):
    """The two passes, then run_cases. `reasoner` and `extractor` may be ClaudeBatch (batched) or any llm (direct)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    system = rp.system_prompt(version)
    schema = rp.api_schema()
    requests = {request_key(system, rp.user_message(c["vision"], c["note"])): (system, rp.user_message(c["vision"], c["note"]), schema)
                for c in cases}
    reasoner_answers = Prefilled(gather(requests, reasoner, log), reasoner)
    log(f"Reasoner: {sum('reply' in r for r in reasoner_answers.results.values())} of {len(requests)} answered.")

    cache = vr.JsonCache(out / "extraction_cache.json")
    ex_system = cl.system_prompt()
    ex_requests = {}
    for case in cases:
        rec, _ = rr.recommend(case["vision"], case["note"], reasoner_answers, version=version, system=system)
        if rec is None:
            continue
        user = cl.user_message(rec)
        key = request_key(ex_system, user)
        if cache.get(key) is None:
            ex_requests[key] = (ex_system, user, cl.EXTRACTOR_SCHEMA)
    for key, result in gather(ex_requests, extractor, log).items():
        if "reply" in result:
            cache.put(key, result["reply"])
    log(f"Extraction: {len(ex_requests)} requested; the cache now holds {len(cache)}.")
    extractor_answers = Prefilled({k: {"reply": v} for k, v in cache.data.items()}, extractor)
    config = {"batched": {"reasoner": hasattr(reasoner, "run"), "extractor": hasattr(extractor, "run"),
                          "batch_ids": [*getattr(reasoner, "batch_ids", []), *getattr(extractor, "batch_ids", [])]},
              **(extra_config or {})}
    return e2e.run_cases(cases, reasoner_answers, extractor_answers, judge, retriever, out, log=log, extra_config=config,
                         source=source, **run_kwargs)
