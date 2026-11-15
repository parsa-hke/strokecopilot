"""Paid runs in two passes, so the Claude calls can go through the Message Batches API (DEC-24, amended 12 Nov 2026).

1. Reasoner: every case's request is answered first, by a ClaudeBatch (Sonnet, half price) or by calling any other llm
   directly (the open models on a pod). The answers are kept by request hash.
2. Claim extraction: the outputs that parse are known now, so their extraction requests go to the extractor as one
   batch. The replies go into the run folder's extraction cache under the key run_cases' CachedExtractor looks up.
3. run_cases runs as usual, with the reasoner and extractor answered from those results: the same files, metrics and
   checks as a direct run, and only the judge (cheap Haiku calls, one per claim) is called live.

Costs: a batched llm carries PRICE_FACTOR 0.5, which run_cases applies to the estimate.

Resume: a directly called reasoner (an open model on a pod or an API) answers request by request, for hours. Each
answer is appended to out_dir/reasoner_answers.jsonl as it arrives, and a rerun into the same out_dir asks only for
the rest. A batch llm is unchanged: its results arrive together.

Judge (DEC-24, amended 16 Nov 2026): a batch judge is answered in rounds too. collect_judge_requests() walks every
claim exactly as the live path does and records each judge request the live path would send next, byte for byte (the
same system prompt, user message and schema, sent only for passages without a verdict). The requests go out as one
batch per round; a second round exists only where a claim's next request depends on an earlier one. run_cases then
runs with the judge answered from those results. A direct judge (ClaudeJSON) is still called live, as before.

Closing the laptop: with wait=False, run_batched submits whatever batch is due and returns {"waiting": ...} instead of
polling. The batch runs at Anthropic. Calling run_batched again with the same out_dir fetches the results if the batch
has ended and moves on to the next stage. out_dir/batches.json records each batch by role and request set, so a batch
is never submitted twice, and every answer is saved in out_dir (<role>_answers.jsonl), so nothing is paid for twice.
Usage is added up in batches.json across sessions, so the cost estimate covers the whole run.

Concurrency (C8 prep): `workers` sends that many direct requests at once (a thread pool), so vLLM can batch them on
the GPU and an API answers in parallel; at 1 the requests go one at a time, as before. Answers are written by the
calling thread as each finishes, so a crash loses at most the requests in flight.
"""

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.reasoner import prompt as rp
from strokecopilot.nlp.reasoner import run as rr
from strokecopilot.nlp.reasoner.providers import RefusalError
from strokecopilot.nlp.verifier import claims as cl
from strokecopilot.nlp.verifier import run as vr
from strokecopilot.nlp.verifier import trace as tr
from strokecopilot.pipeline import end_to_end as e2e


def request_key(system, user):
    """The key a request is stored under: the one CachedExtractor uses for Sonnet, 16 hex characters."""
    return vr._hash([system, user])


class Prefilled:
    """An llm(system, user, schema) answered from results gathered beforehand. It keeps the real model's name and
    describe(), its usage and its price factor, so run_cases records and costs the run as it really ran."""

    def __init__(self, results, source, usage=None):
        self.results, self.source = results, source
        self.model = getattr(source, "model", "unknown")
        self.usage = usage if usage is not None else getattr(source, "usage", {})  # usage: the ledger's, all sessions
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


def _answer(llm, req):
    try:
        return {"reply": llm(*req)}
    except RefusalError as e:
        return {"error": str(e), "refusal": True}
    except (RuntimeError, json.JSONDecodeError) as e:
        return {"error": str(e), "refusal": False}


def gather(requests, llm, log=print, answers_path=None, workers=1):
    """{key: result} for {key: (system, user, schema)}: one batch if the llm has run(), else one call per request,
    up to `workers` at a time.

    With answers_path, a directly called llm's answers are appended there one line each as they arrive, and keys
    already saved are not asked again, so a crashed run resumes. A batch llm ignores answers_path and workers. An
    error the llm does not turn into a result (the server unreachable, a 4xx) stops the run, as before; requests not
    yet started are cancelled."""
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
    def save(key):
        if sink:
            sink.write(json.dumps({"key": key, "model": model, "result": out[key]}) + "\n")
            sink.flush()

    todo = [(key, req) for key, req in requests.items() if key not in out]
    try:
        if workers <= 1:
            for key, req in todo:
                out[key] = _answer(llm, req)
                save(key)
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                futures = {pool.submit(_answer, llm, req): key for key, req in todo}
                for future in as_completed(futures):
                    key = futures[future]
                    out[key] = future.result()
                    save(key)
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    finally:
        if sink:
            sink.close()
    return out


ANSWER_FILES = {"reasoner": ANSWERS_FILE, "extractor": "extractor_answers.jsonl", "judge": "judge_answers.jsonl"}
LEDGER_FILE = "batches.json"
MAX_JUDGE_ROUNDS = 6


class Ledger:
    """out_dir/batches.json: every batch submitted for this run ({"role:hash of its request keys": id, model, count,
    time, collected}) and each role's usage added up over sessions."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"batches": {}, "usage": {}}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=1))

    def add_usage(self, role, before, after):
        total = self.data["usage"].setdefault(role, {})
        for name, value in after.items():
            if isinstance(value, (int, float)):
                total[name] = total.get(name, 0) + value - before.get(name, 0)
        self.save()

    def usage(self, role):
        return self.data["usage"].get(role, {})

    def batch_ids(self):
        return [entry["id"] for entry in self.data["batches"].values()]


def _save_answers(path, model, results):
    with open(path, "a") as sink:
        for key, result in results.items():
            sink.write(json.dumps({"key": key, "model": model, "result": result}) + "\n")


def gather_staged(role, requests, llm, out_dir, ledger, log=print, wait=True, workers=1):
    """{key: result} for every request, answered from out_dir's saved answers where they exist; or None when a batch
    for the rest is submitted but has not ended and wait is False.

    - An llm with submit() and collect() (ClaudeBatch): one batch for the unanswered requests, recorded in the ledger
      under the role and the hash of its keys, so a later call finds it instead of submitting again.
    - An llm with only run() (a blocking batch): called once for the unanswered requests.
    - Any other llm: gather(), up to `workers` requests at a time, each answer saved as it arrives.
    """
    if not requests:
        return {}
    path = Path(out_dir) / ANSWER_FILES[role]
    model = getattr(llm, "model", "unknown")
    saved = load_answers(path, model)
    todo = {key: req for key, req in requests.items() if key not in saved}
    if todo:
        before = dict(getattr(llm, "usage", {}))
        try:
            if hasattr(llm, "submit") and hasattr(llm, "collect"):
                stage = f"{role}:{vr._hash(sorted(todo))}"
                entry = ledger.data["batches"].get(stage)
                if entry is None:
                    entry = {"id": llm.submit(todo, log=log), "model": model, "requests": len(todo), "collected": False,
                             "submitted_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
                    ledger.data["batches"][stage] = entry
                    ledger.save()
                got = llm.collect(entry["id"], list(todo), log=log, wait=wait)
                if got is None:
                    return None
                _save_answers(path, model, got)
                entry["collected"] = True
                ledger.save()
            elif hasattr(llm, "run"):
                got = llm.run(todo, log=log)
                _save_answers(path, model, got)
            else:
                got = gather(todo, llm, log, answers_path=path, workers=workers)
        finally:
            ledger.add_usage(role, before, getattr(llm, "usage", {}))
        saved.update(got)
    return {key: saved[key] for key in requests}


class Pending(Exception):
    """A judge request whose answer has not been gathered yet. Not a RuntimeError, so nothing records it as a failure."""


class Recording(Prefilled):
    """A Prefilled llm that records a request it has no answer for (in `wanted`) and raises Pending."""

    def __init__(self, results, source):
        super().__init__(results, source)
        self.wanted = {}

    def __call__(self, system, user, schema):
        key = request_key(system, user)
        if key not in self.results:
            self.wanted[key] = (system, user, schema)
            raise Pending(key)
        return super().__call__(system, user, schema)


class CollectingJudge(vr.CachedJudge):
    """CachedJudge for a collection round. The live path sends a claim's passages that have no verdict yet. Here a
    passage whose request is recorded but not answered counts as on its way: it is left out of later requests, as its
    verdict would be cached by then in a live run, and a claim that needs it waits for the next round (Pending)."""

    def __init__(self, judge, cache, version):
        super().__init__(judge, cache, version=version)
        self.inflight = set()

    def __call__(self, claim_text, passages, case_inputs, applied):
        keys = {p["id"]: self.key(claim_text, p["id"], case_inputs, applied) for p in passages}
        missing = [p for p in passages if self.cache.get(keys[p["id"]]) is None]
        send = [p for p in missing if keys[p["id"]] not in self.inflight]
        if send:
            try:
                got = self.judge(claim_text, send, case_inputs, applied)
            except Pending:
                self.inflight.update(keys[p["id"]] for p in send)
                raise
            self.calls += 1
            for p in send:
                self.cache.put(keys[p["id"]], list(got.get(p["id"], ("not_addressed", None))))
        if len(send) < len(missing):
            raise Pending("a passage's verdict is on its way")
        return {pid: tuple(self.cache.get(k)) for pid, k in keys.items()}


def collect_judge_requests(cases, reasoner_answers, extractor_answers, judge_results, judge, retriever, out_dir,
                           tau=tr.TAU, judge_prompt=tr.JUDGE_PROMPT, version=rp.REASONER_PROMPT):
    """{key: (system, user, schema)}: every judge request the live path would send next, given the answers so far.

    Each case is walked as run_cases walks it: the same recommendation, the same claims, the same candidates from the
    same retrieval cache, claim by claim in order. A claim that reaches an unanswered request is left for the next
    round. A case whose extraction or judging fails stops where the live path would stop."""
    out = Path(out_dir)
    system = rp.system_prompt(version)
    corpus = ingest.load()
    recorder = Recording(judge_results, judge)
    # A copy of the saved verdicts, in memory only: run_cases fills the real judge cache, and counts its calls, as in
    # a live run.
    verdicts = vr.JsonCache()
    verdicts.data = dict(vr.JsonCache(out / "judge_cache.json").data)
    cached_judge = CollectingJudge(tr.Judge(recorder, judge_prompt), verdicts, judge_prompt)
    cached_retriever = vr.CachedRetriever(retriever, vr.JsonCache(out / "retrieval_cache.json"))
    cached_extractor = vr.CachedExtractor(extractor_answers, vr.JsonCache(out / "extraction_cache.json"))
    for case in cases:
        rec, _ = rr.recommend(case["vision"], case["note"], reasoner_answers, version=version, system=system)
        if rec is None:
            continue
        try:
            claims = cl.extract(rec, cached_extractor)
        except (RefusalError, RuntimeError, json.JSONDecodeError, KeyError):
            continue  # run_cases records this case as a verifier failure
        inputs = e2e._inputs(case, cached_judge, cached_retriever, tau, corpus, rec)
        for claim in claims:
            try:
                tr.trace_claim(claim, inputs)
            except Pending:
                continue
            except (RefusalError, RuntimeError, json.JSONDecodeError, KeyError):
                break  # the live path stops this case here too
    return recorder.wanted


def _waiting(role, ledger, log):
    pending = [e for stage, e in ledger.data["batches"].items() if stage.startswith(role + ":") and not e["collected"]]
    note = {"waiting": role, "batches": [e["id"] for e in pending], "requests": sum(e["requests"] for e in pending)}
    log(f"Waiting for the {role} batch ({note['requests']} requests). Run again later to continue.")
    return note


def run_batched(cases, reasoner, extractor, judge, retriever, out_dir, log=print, source=None, version=rp.REASONER_PROMPT,
                extra_config=None, reasoner_workers=1, wait=True, **run_kwargs):
    """The passes, then run_cases. `reasoner`, `extractor` and `judge` may each be a batch llm (ClaudeBatch) or any
    direct llm; reasoner_workers is how many direct reasoner requests go at once (gather's workers).

    Returns run_cases' metrics. With wait=False it never polls: if a batch is still running it returns
    {"waiting": role, "batches": [...], "requests": n}, and the next call with the same out_dir carries on."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / LEDGER_FILE)
    system = rp.system_prompt(version)
    schema = rp.api_schema()
    requests = {request_key(system, rp.user_message(c["vision"], c["note"])): (system, rp.user_message(c["vision"], c["note"]), schema)
                for c in cases}
    resumed = 0
    if not hasattr(reasoner, "run"):
        resumed = len(set(requests) & set(load_answers(out / ANSWERS_FILE, getattr(reasoner, "model", "unknown"))))
    answers = gather_staged("reasoner", requests, reasoner, out, ledger, log, wait, reasoner_workers)
    if answers is None:
        return _waiting("reasoner", ledger, log)
    reasoner_answers = Prefilled(answers, reasoner, usage=ledger.usage("reasoner"))
    log(f"Reasoner: {sum('reply' in r for r in answers.values())} of {len(requests)} answered.")

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
    extracted = gather_staged("extractor", ex_requests, extractor, out, ledger, log, wait)
    if extracted is None:
        return _waiting("extractor", ledger, log)
    for key, result in extracted.items():
        if "reply" in result:
            cache.put(key, result["reply"])
    log(f"Extraction: {len(ex_requests)} requested; the cache now holds {len(cache)}.")
    extractor_answers = Prefilled({k: {"reply": v} for k, v in cache.data.items()}, extractor, usage=ledger.usage("extractor"))

    judge_llm, judge_rounds = judge, 0
    if hasattr(judge, "run"):  # a batch judge: gathered in rounds; a direct judge is called live by run_cases
        tau, judge_prompt = run_kwargs.get("tau", tr.TAU), run_kwargs.get("judge_prompt", tr.JUDGE_PROMPT)
        judge_results = load_answers(out / ANSWER_FILES["judge"], getattr(judge, "model", "unknown"))
        while True:
            wanted = collect_judge_requests(cases, reasoner_answers, extractor_answers, judge_results, judge, retriever,
                                            out, tau, judge_prompt, version)
            if not wanted:
                break
            judge_rounds += 1
            if judge_rounds > MAX_JUDGE_ROUNDS:
                raise RuntimeError(f"The judge still has unanswered requests after {MAX_JUDGE_ROUNDS} rounds.")
            got = gather_staged("judge", wanted, judge, out, ledger, log, wait)
            if got is None:
                return _waiting("judge", ledger, log)
            judge_results.update(got)
            log(f"Judge round {judge_rounds}: {len(wanted)} requests answered.")
        judge_llm = Prefilled(judge_results, judge, usage=ledger.usage("judge"))

    config = {"batched": {"reasoner": hasattr(reasoner, "run"), "extractor": hasattr(extractor, "run"),
                          "judge": hasattr(judge, "run"), "reasoner_workers": reasoner_workers,
                          "batch_ids": list(dict.fromkeys([*ledger.batch_ids(), *getattr(reasoner, "batch_ids", []),
                                                           *getattr(extractor, "batch_ids", []),
                                                           *getattr(judge, "batch_ids", [])]))},
              **(extra_config or {})}
    if resumed:
        config["resumed_reasoner_answers"] = resumed  # answers saved by an earlier session; usage covers all sessions
    return e2e.run_cases(cases, reasoner_answers, extractor_answers, judge_llm, retriever, out, log=log, extra_config=config,
                         source=source, **run_kwargs)
