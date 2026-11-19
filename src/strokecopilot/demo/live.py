"""One live run for the Space (DEC-32 §4): the builder's case -> Sonnet 5.5 with prompt v3 and the rulebook -> the
verifier (Sonnet extractor, Haiku judge_v3, τ 0.6) -> the rule checks, with each step reported as it happens.

The models and settings are EXP-37's (`paid_runs`: Sonnet effort high, max_tokens 20,000; extractor Sonnet effort medium;
judge Haiku effort medium, max_tokens 8,000), sent live instead of batched. Each call has its own timeout, the SDK
retries 429, 5xx and connection errors twice, and the whole run has a deadline. Any failure raises `LiveFailure` with
the step it happened in, and the caller falls back to the nearest precomputed case. A live answer is never counted in any
result (DEC-32).

`view` turns a verified recommendation into what the page shows. Live runs and the precomputed bank use the same view,
so the page draws them the same way.
"""

import time

from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.reasoner import prompt as rp
from strokecopilot.nlp.reasoner import providers
from strokecopilot.nlp.reasoner import run as rr
from strokecopilot.nlp.verifier import claims as cl
from strokecopilot.nlp.verifier import trace as tr
from strokecopilot.pipeline import end_to_end as e2e
from strokecopilot.pipeline import paid_runs as pr

PROMPT = pr.PROMPT  # reasoner_v3
TIMEOUTS_S = {"reasoner": 150.0, "extractor": 120.0, "judge": 45.0}
DEADLINE_S = 300.0
MAX_RETRIES = 2
STEPS = ("note", "rules", "reasoner", "extractor", "judge", "score")
BANNER = "Research prototype — not for clinical use."


class LiveFailure(RuntimeError):
    def __init__(self, step, reason):
        super().__init__(f"{step}: {reason}")
        self.step, self.reason = step, reason


def client(timeout_s=None):
    """The Anthropic client for live runs. The key comes from the environment (the Space's secret, DEC-32 §8)."""
    import anthropic

    return anthropic.Anthropic(api_key=providers.api_key(), max_retries=MAX_RETRIES, timeout=timeout_s or DEADLINE_S)


def llms(api_client, timeouts=TIMEOUTS_S):
    """Fresh callables per run (so their usage is this run's), sharing one client."""
    return {
        "reasoner": providers.ClaudeJSON(pr.SONNET, effort=pr.SONNET_EFFORT, max_tokens=rr.REASONER_MAX_TOKENS,
                                         client=api_client.with_options(timeout=timeouts["reasoner"])),
        "extractor": providers.ClaudeJSON(cl.EXTRACTOR_MODEL, effort=cl.EXTRACTOR_EFFORT,
                                          client=api_client.with_options(timeout=timeouts["extractor"])),
        "judge": providers.ClaudeJSON(tr.JUDGE_MODEL, effort=tr.JUDGE_EFFORT, max_tokens=pr.JUDGE_MAX_TOKENS,
                                      client=api_client.with_options(timeout=timeouts["judge"])),
    }


def cost_of(models):
    """{part: USD} at live prices from each callable's usage, and the total."""
    out = {name: round(providers.cost_usd(llm.model, llm.usage) or 0.0, 5) for name, llm in models.items()}
    out["total"] = round(sum(out.values()), 5)
    return out


def usage_of(models):
    return {name: dict(llm.usage) for name, llm in models.items()}


class _CountingJudge:
    """The judge callable, reporting each call to the progress callback."""

    def __init__(self, llm, emit):
        self.llm, self.emit, self.calls = llm, emit, 0
        self.model = llm.model

    def __call__(self, system, user, schema):
        self.calls += 1
        self.emit("judge", "running", {"calls": self.calls})
        return self.llm(system, user, schema)


def view(case, rec, claims, traces):
    """What the page shows for one verified recommendation: the case, both decisions, every claim with its sources and
    verdict, and the rule checks. Claims and traces are dicts (Claim.to_dict, Trace.to_dict)."""
    by_id = {c["id"]: c for c in claims}
    shown = []
    for t in traces:
        c = by_id.get(t["claim_id"], {})
        shown.append({"id": t["claim_id"], "field": t["field"], "entry": c.get("entry"), "text": c.get("text"),
                      "kind": t.get("kind"), "sources": c.get("declared_sources", []), "used": t.get("used", []),
                      "supported_passages": t.get("supported_passages", []), "flagged": t["subtype"] is not None,
                      "subtype": t.get("subtype_name"), "critical": t.get("critical"), "reason": t.get("reason")})
    record, check, omissions, missing = e2e._score(case, rec, claims, traces)
    return {"banner": BANNER, "vision": case["vision"], "note": case["note"],
            "expected": case.get("expected_detail") or case["expected"],
            "recommendation": rec, "claims": shown,
            "decisions": {d: {"recommended": got, "rulebook": want, "agrees": ok} for d, (got, want, ok) in check.items()},
            "omissions": omissions, "uncertainty_omissions": missing,
            "counts": {"claims": len(shown), "flagged": sum(s["flagged"] for s in shown),
                       "critical": sum(s["critical"] is True for s in shown)}}


def run(built, models, retriever, emit=lambda *a: None, deadline_s=DEADLINE_S, clock=time.monotonic):
    """One live run of a builder.Built case. emit(step, status, detail) reports progress. Returns
    {"view", "cost", "usage", "seconds"}; raises LiveFailure on any failure or when the deadline passes."""
    started, seconds = clock(), {}

    def check_deadline(step):
        if clock() - started > deadline_s:
            raise LiveFailure(step, f"the run took longer than {deadline_s:.0f} s")

    case = {"id": built.vision["case_id"], "vision": built.vision, "note": built.note, "facts": built.facts,
            "expected": {d: e2e.WORD_TO_BOOL[v["value"]] for d, v in built.expected.items()},
            "expected_detail": built.expected}
    emit("note", "done", {"note": built.note})
    emit("rules", "done", {"expected": built.expected})

    t = clock()
    emit("reasoner", "running", {})
    system = rp.system_prompt(PROMPT, rulebook=True)
    try:
        rec, call = rr.recommend(built.vision, built.note, models["reasoner"], version=PROMPT, system=system)
    except Exception as e:  # timeouts and API errors after the SDK's retries
        raise LiveFailure("reasoner", type(e).__name__) from e
    seconds["reasoner"] = round(clock() - t, 1)
    if rec is None:
        raise LiveFailure("reasoner", (call.get("error") or "no answer")[:120])
    emit("reasoner", "done", {"recommendation": rec, "seconds": seconds["reasoner"]})
    check_deadline("reasoner")

    t = clock()
    emit("extractor", "running", {})
    judge = _CountingJudge(models["judge"], emit)
    inputs = e2e._inputs(case, tr.Judge(judge, tr.JUDGE_PROMPT), retriever, tr.TAU, ingest.load(), rec)
    try:
        claims = cl.extract(rec, models["extractor"])
        seconds["extractor"] = round(clock() - t, 1)
        emit("extractor", "done", {"claims": len(claims), "seconds": seconds["extractor"]})
        check_deadline("extractor")
        t = clock()
        inputs.decision_passages = tr.decision_passages(rec)
        traces = [x.to_dict() for x in tr.trace_all(claims, inputs)]
    except LiveFailure:
        raise
    except Exception as e:
        raise LiveFailure("judge" if "extractor" in seconds else "extractor", type(e).__name__) from e
    seconds["judge"] = round(clock() - t, 1)
    emit("judge", "done", {"calls": judge.calls, "seconds": seconds["judge"]})
    check_deadline("judge")

    shown = view(case, rec, [c.to_dict() for c in claims], traces)
    seconds["total"] = round(clock() - started, 1)
    emit("score", "done", {"counts": shown["counts"]})
    return {"view": shown, "cost": cost_of(models), "usage": usage_of(models), "seconds": seconds}

