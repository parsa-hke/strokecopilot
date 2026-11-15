"""The paid model runs of DEC-24 (C8): which cases each model answers in which arm, and one resumable step per cell.

A cell is one model in one arm, with its own folder (<out_root>/<model>/<arm>/) holding the answers, the batch ledger
and, when finished, the same files as any end-to-end run. advance() moves a cell as far as it can without waiting
(pipeline/batched.run_batched with wait=False): it submits the batch that is due, or fetches one that has ended, and
says what it is waiting for. Calling it again later carries on, so the laptop can close between calls.

Arms (DEC-24's matrix, per model):
- fixtures_rulebook: the 5 fixtures.
- scenarios_rulebook: the 150 synthetic scenarios.
- real_predicted, real_groundtruth: 20 development cases (4 per fold, seeded) with the predicted contracts (EXP-33) and
  with the ground-truth contracts (EXP-34): arm 2 of plan §8.4.
- fixtures_norulebook, scenarios_norulebook (75: every second scenario, ordered by decision path), real_norulebook: arm 1's
  other half. NOT READY: what the reasoner is told and how its uncitable rule claims are scored is Parsa's decision,
  pending; advance() refuses these arms until then.
- real_predicted_repeat2 and _repeat3: the same 20 cases again, for Claude Sonnet 5.5 only (run-to-run variation).

Models: Claude Sonnet 5.5 through the Batch API; Qwen3.5-27B through OpenRouter; Gemma 3 27B and MedGemma 27B text on
the rented GPU through an SSH tunnel (open_models.py). The claim extractor (Sonnet) and the judge (Haiku) go through the
Batch API for every model, with the settings every earlier run used.
"""

from dataclasses import dataclass
from pathlib import Path

from strokecopilot.nlp.corpus import index
from strokecopilot.nlp.reasoner import open_models as om
from strokecopilot.nlp.reasoner import prompt as rp
from strokecopilot.nlp.reasoner import providers
from strokecopilot.nlp.reasoner import run as rr
from strokecopilot.nlp.verifier import claims as cl
from strokecopilot.nlp.verifier import trace as tr
from strokecopilot.pipeline import batched
from strokecopilot.pipeline import end_to_end as e2e

SONNET = "claude-sonnet-5-5"
SONNET_EFFORT = "high"  # as in EXP-9 and EXP-12
JUDGE_MAX_TOKENS = 8000  # as in every run since EXP-10
MODELS = (SONNET, "qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text")
FALLBACK_PAIR = ("qwen2.5-32b", "baichuan-m2-32b")  # replaces the Gemma pair if Parsa rules it out (DEC-24)
REAL_CASES = 20
WORKERS = 8  # direct reasoner requests sent at once (OpenRouter, vLLM)

ARMS = {
    "fixtures_rulebook": {"source": "fixtures", "rulebook": True},
    "scenarios_rulebook": {"source": "scenarios", "rulebook": True},
    "real_predicted": {"source": "real", "vision": "predicted", "rulebook": True},
    "real_groundtruth": {"source": "real", "vision": "groundtruth", "rulebook": True},
    "fixtures_norulebook": {"source": "fixtures", "rulebook": False},
    "scenarios_norulebook": {"source": "scenarios", "rulebook": False, "half": True},
    "real_norulebook": {"source": "real", "vision": "predicted", "rulebook": False},
}
SONNET_REPEATS = {name: {"source": "real", "vision": "predicted", "rulebook": True, "repeat_of": "real_predicted"}
                  for name in ("real_predicted_repeat2", "real_predicted_repeat3")}


class NotReady(RuntimeError):
    """An arm that cannot run yet."""


@dataclass(frozen=True)
class Paths:
    predicted: Path  # artifacts/vision_outputs_v1 (EXP-33)
    groundtruth: Path  # artifacts/vision_outputs_gt_v1 (EXP-34)
    notes: Path  # artifacts/isles24_notes_v1
    out_root: Path  # runs/<EXP>_paid_runs


def arms_for(model):
    return {**ARMS, **(SONNET_REPEATS if model == SONNET else {})}


def cells(models=MODELS, ready_only=True):
    """[(model, arm)] in run order; without the arms that are not ready unless ready_only is False."""
    return [(model, arm) for model in models for arm, spec in arms_for(model).items() if spec["rulebook"] or not ready_only]


def half_scenarios(cases):
    """Half the scenarios (75 of 150, DEC-24): sorted by decision path (the pair of expected decisions) and then by ID,
    every second one is kept, so each path keeps about half its scenarios. Deterministic."""
    ordered = sorted(cases, key=lambda c: (str(c["expected"]["thrombolysis"]), str(c["expected"]["thrombectomy"]), c["id"]))
    return sorted(ordered[::2], key=lambda c: c["id"])


def arm_cases(model, arm, paths, folds=None):
    spec = arms_for(model)[arm]
    if spec["source"] == "real":
        ids = e2e.sample_real_ids(paths.predicted, n=REAL_CASES, folds=folds)
        vision_dir = paths.groundtruth if spec["vision"] == "groundtruth" else paths.predicted
        return e2e.load_cases("real", ids=ids, vision_dir=vision_dir, notes_dir=paths.notes, folds=folds)
    cases = e2e.load_cases(spec["source"])
    return half_scenarios(cases) if spec.get("half") else cases


def reasoner_llm(model, port=8000):
    """The reasoner for a model: a batch for Sonnet, the OpenRouter route for a model that has one, else the vLLM
    server behind the SSH tunnel on localhost:<port>."""
    if model == SONNET:
        return providers.ClaudeBatch(SONNET, effort=SONNET_EFFORT, max_tokens=rr.REASONER_MAX_TOKENS)
    if model in om.API_ROUTES:
        return om.api_llm(model)
    entry = om.MODELS[model]
    return providers.VLLMJSON(entry.name, base_url=f"http://localhost:{int(port)}/v1", quantization=entry.quantization,
                              merge_system=bool(entry.merge_system))


def verifier_llms():
    """(extractor, judge) as batches, with the settings of every earlier run (EXP-12, EXP-23)."""
    return (providers.ClaudeBatch(cl.EXTRACTOR_MODEL, effort=cl.EXTRACTOR_EFFORT),
            providers.ClaudeBatch(tr.JUDGE_MODEL, effort=tr.JUDGE_EFFORT, max_tokens=JUDGE_MAX_TOKENS))


def cell_dir(paths, model, arm):
    return Path(paths.out_root) / model.replace("/", "_") / arm


def advance(model, arm, paths, reasoner, extractor, judge, retriever, version=rp.REASONER_PROMPT, wait=False, log=print,
            folds=None, extra_config=None):
    """Move one cell forward. Returns {"done": metrics summary} or {"waiting": role, ...} (batched.run_batched)."""
    spec = arms_for(model)[arm]
    if not spec["rulebook"]:
        raise NotReady(f"{arm}: the without-rulebook arm is not defined yet (Parsa's decision).")
    cases = arm_cases(model, arm, paths, folds)
    out = cell_dir(paths, model, arm)
    result = batched.run_batched(
        cases, reasoner, extractor, judge, retriever, out, log=log, source=spec["source"], version=version,
        reasoner_workers=WORKERS, wait=wait,
        extra_config={"paid_run": {"model": model, "arm": arm, "cases": len(cases), **spec}, **(extra_config or {})})
    if "waiting" in result:
        return result
    return {"done": result["summary"], "cost_usd_estimate": result["cost_usd_estimate"]}


def status(models, paths):
    """One row per ready cell from the files on disk, with no model or network: done, the stage it waits for, or not
    started. For the monitor and for planning the next sitting."""
    import json

    rows = []
    for model, arm in cells(models):
        out = cell_dir(paths, model, arm)
        row = {"model": model, "arm": arm, "state": "not started"}
        ledger = out / batched.LEDGER_FILE
        if (out / "metrics.json").exists():
            row["state"] = "done"
        elif ledger.exists():
            open_batches = [stage.split(":")[0] for stage, e in json.loads(ledger.read_text())["batches"].items()
                            if not e["collected"]]
            row["state"] = f"waiting: {open_batches[-1]} batch" if open_batches else "between stages"
        elif (out / batched.ANSWERS_FILE).exists():
            row["state"] = "reasoner answers saved"
        rows.append(row)
    return rows


def make_retriever():
    """The verifier's retriever, as in every run: BAAI/bge-base-en-v1.5 over the corpus (loaded once)."""
    return index.Index.from_corpus().search
