"""Paid runs in two passes, so the Claude calls can go through the Message Batches API (DEC-24, amended 12 Nov 2026).

1. Reasoner: every case's request is answered first, by a ClaudeBatch (Sonnet, half price) or by calling any other llm
   directly (the open models on a pod). The answers are kept by request hash.
2. Claim extraction: the outputs that parse are known now, so their extraction requests go to the extractor as one
   batch. The replies go into the run folder's extraction cache under the key run_cases' CachedExtractor looks up.
3. run_cases runs as usual, with the reasoner and extractor answered from those results: the same files, metrics and
   checks as a direct run, and only the judge (cheap Haiku calls, one per claim) is called live.

Costs: a batched llm carries PRICE_FACTOR 0.5, which run_cases applies to the estimate.

Resume: a directly called reasoner (an open model on a pod) answers one request at a time, for hours. Each answer is
appended to out_dir/reasoner_answers.jsonl as it arrives, and a rerun into the same out_dir asks only for the rest. A
batch llm is unchanged: its results arrive together.
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


ANSWERS_FILE = "reasoner_answers.jsonl"


def load_answers(path, model):
    """{key: result} saved by an earlier gather() into `path`; {} if there is none. A last line cut off by a crash is
    skipped. Answers from a different model stop the run: a resumed run must be one model's answers."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") != model:
            raise RuntimeError(f"{path} holds answers from {row.get('model')!r}, not {model!r}. Use a new out_dir.")
        out[row["key"]] = row["result"]
    return out


def gather(requests, llm, log=print, answers_path=None):
    """{key: result} for {key: (system, user, schema)}: one batch if the llm has run(), else one call per request.

    With answers_path, a directly called llm's answers are appended there one line each as they arrive, and keys
    already saved are not asked again, so a crashed run resumes. A batch llm ignores answers_path."""
    if not requests:
        return {}
    if hasattr(llm, "run"):
        return llm.run(requests, log=log)
    model = getattr(llm, "model", "unknown")
    saved = load_answers(answers_path, model) if answers_path else {}
    out = {key: saved[key] for key in requests if key in saved}
    if out:
        log(f"Resuming: {len(out)} of {len(requests)} answers already saved in {Path(answers_path).name}.")
    sink = open(answers_path, "a") if answers_path else None
    if sink and sink.tell() and not Path(answers_path).read_text().endswith("\n"):
        sink.write("\n")  # end a line cut off by a crash, so the next answer starts on its own line
    try:
        for key, req in requests.items():
            if key in out:
                continue
            try:
                out[key] = {"reply": llm(*req)}
            except RefusalError as e:
                out[key] = {"error": str(e), "refusal": True}
            except (RuntimeError, json.JSONDecodeError) as e:
                out[key] = {"error": str(e), "refusal": False}
            if sink:
                sink.write(json.dumps({"key": key, "model": model, "result": out[key]}) + "\n")
                sink.flush()
    finally:
        if sink:
            sink.close()
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
    answers_path = out / ANSWERS_FILE
    resumed = 0
    if not hasattr(reasoner, "run"):
        resumed = len(set(requests) & set(load_answers(answers_path, getattr(reasoner, "model", "unknown"))))
    reasoner_answers = Prefilled(gather(requests, reasoner, log, answers_path=answers_path), reasoner)
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
    if resumed:
        # The reasoner's usage counts only this session's calls; these answers came from an earlier one.
        config["resumed_reasoner_answers"] = resumed
    return e2e.run_cases(cases, reasoner_answers, extractor_answers, judge, retriever, out, log=log, extra_config=config,
                         source=source, **run_kwargs)
