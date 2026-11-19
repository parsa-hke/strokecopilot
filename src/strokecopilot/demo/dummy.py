"""DEC-32 §7's dummy: five live runs of the scenario builder from the Mac, to measure each step's cost and latency before
the Space is built out. Paid (about $0.50, estimate), so it runs only when Parsa starts it, with the Space's dedicated
key in his shell (ANTHROPIC_API_KEY). Never with a key in code or chat.

    python -m strokecopilot.demo.dummy --out runs/EXP-64_space_live_dummy [--cold-wait 330]

Order: run 1 is cold (the 5-minute prompt cache is empty), runs 2-4 follow at once (warm), then it waits --cold-wait
seconds (the cache expires) and run 5 is cold again. The inputs are synthetic, so nothing here holds a case ID.
"""

import argparse
import json
import time
from pathlib import Path

from strokecopilot.demo import builder, live

# Five builder inputs covering different routes (thrombectomy late window, thrombolysis early, a bleed, a large core,
# a wake-up stroke).
INPUTS = (
    {"minutes": 540, "onset": "unwitnessed", "age": 64, "nihss": 16, "mrs": 0, "core_ml": 18.0, "penumbra_ml": 95.0,
     "site": "M1", "aspects": 8},
    {"minutes": 150, "onset": "witnessed", "age": 58, "nihss": 8, "mrs": 0, "core_ml": 2.0, "penumbra_ml": 30.0,
     "site": "M2", "aspects": 9},
    {"minutes": 120, "onset": "witnessed", "age": 77, "nihss": 12, "mrs": 1, "core_ml": 5.0, "penumbra_ml": 40.0,
     "site": "unknown", "haemorrhage": True},
    {"minutes": 360, "onset": "witnessed", "age": 71, "nihss": 18, "mrs": 0, "core_ml": 110.0, "penumbra_ml": 120.0,
     "site": "M1", "aspects": 4},
    {"minutes": 600, "onset": "wake_up", "sleep_minutes": 360, "age": 82, "nihss": 11, "mrs": 0, "core_ml": 15.0,
     "penumbra_ml": 60.0, "site": "M1", "aspects": None},
)


def summarise(runs):
    ok = [r for r in runs if r.get("ok")]
    def spread(values):
        return {"min": round(min(values), 2), "median": round(sorted(values)[len(values) // 2], 2), "max": round(max(values), 2)} if values else None
    out = {"runs": len(runs), "succeeded": len(ok), "failed": [r.get("failure") for r in runs if not r.get("ok")]}
    for label, group in (("cold", [r for r in ok if r["cache"] == "cold"]), ("warm", [r for r in ok if r["cache"] == "warm"])):
        out[label] = {"runs": len(group), "cost_usd": spread([r["cost"]["total"] for r in group]),
                      "seconds_total": spread([r["seconds"]["total"] for r in group]),
                      **{f"seconds_{s}": spread([r["seconds"][s] for r in group]) for s in ("reasoner", "extractor", "judge")},
                      **{f"cost_{p}": spread([r["cost"][p] for r in group]) for p in ("reasoner", "extractor", "judge")}}
    out["total_cost_usd"] = round(sum(r["cost"]["total"] for r in ok), 4)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="DEC-32 dummy: five live builder runs, cost and latency per step.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cold-wait", type=float, default=330.0, help="seconds to wait before run 5 (cache expiry)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"{out} already holds a finished dummy.")
    (out / "views").mkdir(parents=True, exist_ok=True)
    from strokecopilot.nlp.corpus import index

    retriever = index.Index.from_corpus().search
    api = live.client()
    runs = []
    for i, raw in enumerate(INPUTS, 1):
        if i == 5 and args.cold_wait:
            print(f"Waiting {args.cold_wait:.0f} s so the prompt cache expires before run 5.", flush=True)
            time.sleep(args.cold_wait)
        built = builder.build(raw)
        models = live.llms(api)
        cache = "cold" if i in (1, 5) else "warm"
        events = []
        started = time.time()
        try:
            result = live.run(built, models, retriever, emit=lambda s, st, d: events.append([round(time.time() - started, 1), s, st]))
            runs.append({"run": i, "ok": True, "cache": cache, "cost": result["cost"], "seconds": result["seconds"],
                         "usage": result["usage"], "counts": result["view"]["counts"],
                         "decisions": result["view"]["decisions"], "events": events})
            (out / "views" / f"run{i}.json").write_text(json.dumps(result["view"], indent=1, default=str))
            print(f"run {i} ({cache}): {result['seconds']['total']} s, ${result['cost']['total']:.4f}, "
                  f"{result['view']['counts']['flagged']}/{result['view']['counts']['claims']} claims flagged", flush=True)
        except live.LiveFailure as e:
            runs.append({"run": i, "ok": False, "cache": cache, "failure": str(e), "cost": live.cost_of(models),
                         "usage": live.usage_of(models), "events": events})
            print(f"run {i} ({cache}) FAILED at {e.step}: {e.reason}; cost so far ${live.cost_of(models)['total']:.4f}", flush=True)
    metrics = {"experiment": out.name.split("_")[0], "summary": summarise(runs), "runs": runs,
               "settings": {"timeouts_s": live.TIMEOUTS_S, "deadline_s": live.DEADLINE_S, "max_retries": live.MAX_RETRIES,
                            "prompt": live.PROMPT}}
    metrics["summary"]["total_cost_usd_including_failures"] = round(sum(r["cost"]["total"] for r in runs), 4)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    print(json.dumps(metrics["summary"], indent=1))
    print(f"OK: dummy finished, {metrics['summary']['succeeded']}/5 runs, total ${metrics['summary']['total_cost_usd_including_failures']:.4f}.")


if __name__ == "__main__":
    main()
