"""Case -> recommendation.json, and the fixture run that logs every prompt and reply. Plan §8.3, §6.4.

`llm` is any callable `llm(system, user, schema) -> dict` (providers.ClaudeJSON, or a fake in tests). The model fills
the decisions, rationale and uncertainties; this module adds case_id, model and prompt_version and validates the whole
object against schemas/recommendation.json. An output that fails validation is a format failure: it is recorded,
never repaired.
"""

import copy
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


def _bare_id(value):
    """'[AHA2026-T8-ich]' -> 'AHA2026-T8-ich'; anything else unchanged."""
    if isinstance(value, str) and len(value) > 2 and value[0] == "[" and value[-1] == "]" and "[" not in value[1:-1]:
        return value[1:-1]
    return value


def _strip_id_brackets(rec):
    """Parsa, 16 Nov 2026 (EXP-37): one pair of square brackets around a passage ID is removed, in each decision's
    rule_ids and in rationale sources of type passage, for every model; returns how many were removed. The prompt's
    rulebook shows each ID as [ID], and MedGemma copied the brackets into most citations. An ID that still names no
    passage after this stays an error. Uncertainty text is not touched (the prompt asks for [ID] there)."""
    count = 0
    for decision in ("thrombolysis", "thrombectomy"):
        part = rec.get(decision)
        if isinstance(part, dict) and isinstance(part.get("rule_ids"), list):
            bare = [_bare_id(i) for i in part["rule_ids"]]
            count += sum(a != b for a, b in zip(part["rule_ids"], bare))
            part["rule_ids"] = bare
    for entry in rec.get("rationale") or []:
        for source in (entry.get("sources") or []) if isinstance(entry, dict) else []:
            if isinstance(source, dict) and source.get("type") == "passage" and "id" in source:
                bare = _bare_id(source["id"])
                count += bare != source["id"]
                source["id"] = bare
    return count


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
    reply = copy.deepcopy(reply)  # never change the caller's reply in place (EXP-37: a shared saved answer was stripped
    # by the first pass, so the scoring pass counted 0 stripped IDs)
    if isinstance(reply, list) and len(reply) == 1 and isinstance(reply[0], dict):
        # Parsa, 16 Nov 2026 (EXP-37): a list holding exactly one object is scored as that object, for every model,
        # and counted (record["unwrapped_list"], metrics "unwrapped_lists"). Qwen through OpenRouter (Alibaba) wraps
        # many answers this way although the request asks for a strict JSON object: the provider's enforcement, not
        # the answer's content. Any other non-object reply is still a format failure.
        reply = reply[0]
        record["unwrapped_list"] = True
    if not isinstance(reply, dict):
        record["error"] = f"format failure: the reply is a JSON {type(reply).__name__}, not an object"
        return None, record
    rec = {"case_id": vision["case_id"], "model": model_name(llm), "prompt_version": version, **reply}
    stripped = _strip_id_brackets(rec)
    if stripped:
        record["stripped_ids"] = stripped
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
