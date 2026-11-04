"""Case -> recommendation.json, and the fixture run that logs every prompt and reply. Plan §8.3, §6.4.

`llm` is any callable `llm(system, user, schema) -> dict` (providers.ClaudeJSON, or a fake in tests). The model fills
the decisions, rationale and uncertainties; this module adds case_id, model and prompt_version and validates the whole
object against schemas/recommendation.json. An output that fails validation is a format failure: it is recorded,
never repaired.
"""

import hashlib
import json
import time
from pathlib import Path

from strokecopilot.config import FIXTURES_DIR
from strokecopilot.contract.validate import ContractError, validate_recommendation, validate_vision_output
from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.notes.facts import fixture_note
from strokecopilot.nlp.reasoner import prompt as rp
from strokecopilot.nlp.reasoner.providers import RefusalError, cost_usd

EXPECTED = Path(FIXTURES_DIR) / "expected_rules.json"
REASONER_MAX_TOKENS = 20000  # thinking counts against it; above ~21K the SDK wants streaming


def model_name(llm):
    describe = getattr(llm, "describe", None)
    return describe()["model"] if describe else getattr(llm, "model", "unknown")


def recommend(vision, note, llm, version=rp.REASONER_PROMPT, system=None):
    """(recommendation or None, call record). The record holds the user message, the raw reply and any error."""
    validate_vision_output(vision)
    system = system if system is not None else rp.system_prompt(version)
    user = rp.user_message(vision, note)
    record = {"case_id": vision["case_id"], "user": user, "reply": None, "error": None}
    started = time.perf_counter()
    try:
        reply = llm(system, user, rp.api_schema())
    except RefusalError as e:
        record.update(error=f"refusal: {e}")
        return None, record
    except (json.JSONDecodeError, RuntimeError) as e:
        record.update(error=f"no valid JSON: {e}")
        return None, record
    finally:
        record["seconds"] = round(time.perf_counter() - started, 2)
    record["reply"] = reply
    rec = {"case_id": vision["case_id"], "model": model_name(llm), "prompt_version": version, **reply}
    try:
        validate_recommendation(rec)
    except ContractError as e:
        record["error"] = f"format failure: {e}"
        return None, record
    return rec, record


def fixture_stems():
    return sorted(p.stem for p in Path(FIXTURES_DIR).glob("fx*.json"))


def _usage_delta(before, after):
    return {k: after[k] - before.get(k, 0) for k in after} if after else None


def decision_check(rec, expected):
    """{decision: (recommended, expected, agrees)}; expected None means either answer is acceptable."""
    out = {}
    for decision, want in expected.items():
        got = rec[decision]["recommend"] if rec else None
        out[decision] = (got, want, None if want is None else got == want)
    return out


def run_fixtures(llm, out_dir, stems=None, repeats=1, version=rp.REASONER_PROMPT, extra_config=None, log=print):
    """Run the reasoner on fixtures and write the run folder:

    config.json, metrics.json, system_prompt.md, outputs/<stem>_r<n>.json (valid recommendations) and
    calls/<stem>_r<n>.json (user message, raw reply, error, usage) for every call."""
    out = Path(out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    (out / "calls").mkdir(exist_ok=True)
    stems = stems or fixture_stems()
    system = rp.system_prompt(version)
    (out / "system_prompt.md").write_text(system, encoding="utf-8")
    expected = json.loads(EXPECTED.read_text())["decisions"]
    model = model_name(llm)
    config = {
        "model": getattr(llm, "describe", lambda: {"model": model})(),
        "prompt_version": version,
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "corpus_sha256": ingest.version(),
        "corpus_unchecked": ingest.unchecked(ingest.read()),
        "fixtures": stems,
        "repeats": repeats,
        **(extra_config or {}),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    rows, failures = [], []
    for stem in stems:
        vision = json.loads((Path(FIXTURES_DIR) / f"{stem}.json").read_text())
        note, _ = fixture_note(stem)
        for r in range(1, repeats + 1):
            before = dict(getattr(llm, "usage", {}) or {})
            rec, record = recommend(vision, note, llm, version, system)
            record["usage"] = _usage_delta(before, getattr(llm, "usage", None))
            name = f"{stem}_r{r}.json"
            (out / "calls" / name).write_text(json.dumps(record, indent=2, ensure_ascii=False))
            if rec is None:
                failures.append({"fixture": stem, "repeat": r, "error": record["error"][:300]})
                log(f"{stem} r{r}: FAILED ({record['error'][:120]})")
            else:
                (out / "outputs" / name).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            check = decision_check(rec, expected[stem])
            rows.append({"fixture": stem, "repeat": r, "seconds": record["seconds"], "decisions": check,
                         "rationale_entries": len(rec["rationale"]) if rec else None})
            if rec is not None:
                log(f"{stem} r{r}: " + ", ".join(
                    f"{d} {'yes' if got else 'no'} (expected {'either' if want is None else 'yes' if want else 'no'})"
                    for d, (got, want, _) in check.items()) + f"; {len(rec['rationale'])} rationale entries; {record['seconds']} s")

    usage = dict(getattr(llm, "usage", {}) or {})
    judged = [ok for row in rows for (_, _, ok) in row["decisions"].values() if ok is not None]
    metrics = {
        "calls": len(rows),
        "format_failures": len(failures),
        "failures": failures,
        "decisions_agreeing": sum(judged),
        "decisions_judged": len(judged),
        "per_call": rows,
        "usage": usage,
        "cost_usd_estimate": cost_usd(model, usage),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
