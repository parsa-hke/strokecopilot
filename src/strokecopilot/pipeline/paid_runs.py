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


# ---------------------------------------------------------------------------- The pod models (served through a tunnel)

def answer_model(model):
    """The model name a cell's reasoner answers are saved under (the llm's own `model`)."""
    return om.API_ROUTES[model].model_id if model in om.API_ROUTES else model


def reasoner_answered(model, arm, paths, version=rp.REASONER_PROMPT, folds=None):
    """True when every reasoner request of the cell already has a saved answer, so the cell needs no reasoner call."""
    system = rp.system_prompt(version)
    wanted = {batched.request_key(system, rp.user_message(c["vision"], c["note"])) for c in arm_cases(model, arm, paths, folds)}
    saved = batched.load_answers(cell_dir(paths, model, arm) / batched.ANSWERS_FILE, answer_model(model))
    return wanted <= set(saved)


def wait_ready(port, model, timeout_s=3600, every_s=15, log=print, fetch=None, sleep=None):
    """Wait until the vLLM server behind the tunnel lists `model` (the first start downloads the weights)."""
    import json
    import time
    import urllib.error
    import urllib.request

    def default_fetch():
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/v1/models", timeout=10) as response:
            return json.loads(response.read())

    fetch, sleep, started = fetch or default_fetch, sleep or time.sleep, time.time()
    while True:
        try:
            if any(m.get("id") == model for m in fetch().get("data", [])):
                log(f"{model} is being served ({time.time() - started:.0f} s after start).")
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError):
            pass
        if time.time() - started > timeout_s:
            raise TimeoutError(f"{model} was not being served after {timeout_s} s.")
        sleep(every_s)


class PodOps:
    """What the driver does to the pod, in one place so tests can replace it."""

    def __init__(self, exp):
        from strokecopilot.compute import pod, remote

        self.exp, self.pod, self.remote = exp, pod, remote
        self.state = pod.load_state(exp)
        if not self.state or self.state.get("destroyed") or "serve" not in self.state:
            raise RuntimeError(f"{exp} has no serve pod up. Create it first with scripts/pod up.")

    def tunnel(self, port):
        return self.pod.Tunnel(self.state, port)

    def start(self, argv):
        return self.pod.remote_sh(self.state, self.remote.start_serve(argv)).strip()

    def stop(self):
        return self.pod.remote_sh(self.state, self.remote.STOP_SERVE).strip()

    def log_tail(self):
        return self.pod.remote_sh(self.state, self.remote.SERVE_LOG)

    def down(self, log=print):
        self.pod.down(self.exp, log=log)


def run_pod_models(exp, paths, retriever, version=rp.REASONER_PROMPT, log=print, port=8000, ops=None, verifier=None,
                   llm_for=reasoner_llm, ready=wait_ready, only_arms=None, folds=None):
    """The reasoner pass of the pod models: for each model of the serve run, start vLLM, wait for it, answer every
    cell that still lacks answers (each saved on the Mac as it arrives, WORKERS at a time), and stop the server. The
    pod is destroyed at the end whatever happens, with the server's last log lines kept in the cell folder's parent.
    Returns {(model, arm): advance()'s result}; the extraction batches of finished cells are submitted on the way."""
    from strokecopilot.compute import jobs

    serve = jobs.get(exp)["serve"]
    ops = ops or PodOps(exp)
    extractor, judge = verifier or verifier_llms()
    results = {}
    try:
        with ops.tunnel(port):
            for model in serve["models"]:
                todo = [arm for _, arm in cells([model]) if (only_arms is None or arm in only_arms)
                        and not reasoner_answered(model, arm, paths, version, folds)]
                if not todo:
                    log(f"{model}: every cell already has its reasoner answers.")
                    continue
                argv = om.serve_command(model, port, allow_gated=serve["allow_gated"], allow_medgemma=serve["allow_medgemma"])
                log(f"{model}: {ops.start(argv)}")
                try:
                    ready(port, model, log=log)
                    for arm in todo:
                        results[(model, arm)] = advance(model, arm, paths, llm_for(model, port), extractor, judge, retriever,
                                                        version, log=log, folds=folds)
                finally:
                    tail = ops.log_tail()
                    out = Path(paths.out_root) / model
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "serve_log_tail.txt").write_text(tail or "")
                    log(f"{model}: {ops.stop()}")
    finally:
        ops.down(log=log)
    return results


def smoke(model, llm, version=rp.REASONER_PROMPT, stem="fx02_thrombectomy_large_mismatch"):
    """One reasoner request (a fixture) to a model: did it return a valid recommendation, how long, how many tokens,
    and what a full run of N requests would cost at that size. Numbers only."""
    case = next(c for c in e2e.load_cases("fixtures") if c["id"] == stem)
    rec, call = rr.recommend(case["vision"], case["note"], llm, version=version)
    usage = dict(getattr(llm, "usage", {}))
    cost = providers.cost_usd(getattr(llm, "model", ""), usage)
    return {"model": model, "valid": rec is not None, "error": call["error"], "seconds": call.get("seconds"), "usage": usage,
            "cost_usd": cost, "projected_cost_usd": {n: None if cost is None else round(cost * n, 2) for n in (195, 295)}}


def advance_all(models, paths, retriever, version=rp.REASONER_PROMPT, wait=False, log=print, llm_for=reasoner_llm, verifier=None,
                only_arms=None):
    """Move every ready cell of `models` one step. A pod model's cell is skipped until its reasoner answers are saved
    (run_pod_models). Returns [{model, arm, state}]."""
    extractor, judge = verifier or verifier_llms()
    rows = []
    for model, arm in cells(models):
        if only_arms is not None and arm not in only_arms:
            continue
        needs_pod = model != SONNET and model not in om.API_ROUTES
        if needs_pod and not reasoner_answered(model, arm, paths, version):
            rows.append({"model": model, "arm": arm, "state": "needs the pod"})
            continue
        result = advance(model, arm, paths, llm_for(model), extractor, judge, retriever, version, wait=wait, log=log)
        rows.append({"model": model, "arm": arm, "state": "done" if "done" in result else f"waiting: {result['waiting']} batch"})
    return rows


def main(argv=None):
    import argparse
    import json

    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="The paid model runs (C8): status, advance, serve, smoke.")
    parser.add_argument("command", choices=("status", "advance", "serve", "smoke"))
    parser.add_argument("--out", required=True, help="the run folder, e.g. runs/EXP-37_paid_runs")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--prompt", default=rp.REASONER_PROMPT)
    parser.add_argument("--exp", help="serve: the pod's run, e.g. EXP-37")
    parser.add_argument("--wait", action="store_true", help="advance: poll each batch instead of returning")
    parser.add_argument("--artifacts", default=str(repo / "data" / "local" / "artifacts"))
    args = parser.parse_args(argv)
    art = Path(args.artifacts)
    paths = Paths(art / "vision_outputs_v1", art / "vision_outputs_gt_v1", art / "isles24_notes_v1", Path(args.out))
    models = [m for m in args.models.split(",") if m]
    if args.command == "status":
        rows = status(models, paths)
    elif args.command == "advance":
        rows = advance_all(models, paths, make_retriever(), args.prompt, wait=args.wait)
    elif args.command == "serve":
        results = run_pod_models(args.exp, paths, make_retriever(), args.prompt)
        rows = [{"model": m, "arm": a, "state": "done" if "done" in r else f"waiting: {r['waiting']} batch"}
                for (m, a), r in results.items()]
    else:
        rows = [smoke(m, reasoner_llm(m), args.prompt) for m in models]
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
