"""DEC-32 §5's load test: many visitors at once, every one must get a result, with no server error.

Two modes:
- `--fake` (default when no --url): starts the real server in this process on a free port, with a fake live runner that
  is slow and fails on a schedule (no API call, no key), and the real bank.
- `--url https://...`: a running server, e.g. the Space before it is made public, with live runs switched off
  (LIVE_ENABLED=0), so nothing is spent.

    python -m strokecopilot.demo.loadtest --bank demo/space/build/bank --visitors 50
Each visitor posts builder inputs (spread over the bounds) and reads the event stream to its result. The report gives
counts by result source, HTTP errors, and seconds to the result (median, worst).
"""

import argparse
import asyncio
import json
import random
import socket
import statistics
import threading
import time


def _inputs(rng):
    from strokecopilot.demo import builder

    site = rng.choice(["ICA", "M1", "M2", "basilar", "none", "unknown"])
    core = round(rng.uniform(0, 120) * 2) / 2
    onset = rng.choice(["witnessed", "unwitnessed", "wake_up"])
    raw = {"minutes": rng.randrange(300 if onset == "wake_up" else 60, 1500, 10), "onset": onset,
           "age": rng.randrange(50, 91), "nihss": rng.randrange(2, 26), "mrs": rng.randrange(0, 4),
           "core_ml": core, "penumbra_ml": min(250.0, core + round(rng.uniform(0, 120) * 2) / 2), "site": site,
           "aspects": None if site == "basilar" else rng.choice([None, 3, 6, 8, 10]),
           "haemorrhage": rng.random() < 0.1}
    builder.validate(raw)
    return raw


async def visitor(client, base, rng, i, spread_s=0.0):
    if spread_s:
        await asyncio.sleep(rng.uniform(0, spread_s))
    started = time.monotonic()
    r = await client.post(f"{base}/api/run", json={"inputs": _inputs(rng)}, headers={"x-forwarded-for": f"10.0.{i // 250}.{i % 250}"})
    if r.status_code != 200:
        return {"error": f"POST {r.status_code}"}
    run_id = r.json()["run_id"]
    result = None
    async with client.stream("GET", f"{base}/api/run/{run_id}/events") as s:
        if s.status_code != 200:
            return {"error": f"events {s.status_code}"}
        async for line in s.aiter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("type") == "result":
                    result = event
    if result is None:
        return {"error": "stream ended without a result"}
    return {"source": result["source"], "reason": result.get("reason"), "seconds": round(time.monotonic() - started, 2)}


async def run_load(base, visitors, seed=20261119, timeout_s=400, spread_s=0.0):
    import httpx

    rng = random.Random(seed)
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        health = (await client.get(f"{base}/healthz")).json()
        results = await asyncio.gather(*(visitor(client, base, random.Random(rng.random()), i, spread_s) for i in range(visitors)))
    ok = [r for r in results if "error" not in r]
    secs = [r["seconds"] for r in ok]
    by_source = {}
    for r in ok:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    reasons = {}
    for r in ok:
        if r.get("reason"):
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {"visitors": visitors, "results": len(ok), "errors": [r["error"] for r in results if "error" in r],
            "by_source": by_source, "fallback_reasons": reasons,
            "seconds": {"median": statistics.median(secs) if secs else None, "max": max(secs) if secs else None},
            "health_before": {k: health.get(k) for k in ("bank_cases", "live_enabled", "key_present")}}


def fake_runner(delay_s=(2.0, 6.0), fail_every=4, seed=1):
    """A live runner that takes 2-6 s and fails every 4th run at a rotating step, through the real live.run path with
    fake models (gold answers), so the server, limits and fallback are exercised end to end."""
    from strokecopilot.demo import live
    from strokecopilot.nlp.corpus import ingest
    from strokecopilot.nlp.reasoner import gold

    rng, lock, count = random.Random(seed), threading.Lock(), [0]
    corpus = ingest.load()

    class Fake:
        def __init__(self, role, built, model):
            self.role, self.built, self.model = role, built, model
            self.usage = {"requests": 0, "input_tokens": 2000, "output_tokens": 3000, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0}

        def __call__(self, system, user, schema):
            self.usage["requests"] += 1
            if self.role == "reasoner":
                return gold.gold_answer(self.built.vision, self.built.facts, corpus)
            if self.role == "extractor":
                return {"claims": []}
            return {"verdicts": [{"id": p["id"], "verdict": "supported", "scope_fits": True, "reason": "fake"}
                                 for p in json.loads(user)["passages"]]}

    def retriever(text, k=5):
        return [(pid, 0.9) for pid in list(corpus)[:k]]

    def run(built, emit):
        with lock:
            count[0] += 1
            n = count[0]
            wait = rng.uniform(*delay_s)
        time.sleep(wait)
        models = {"reasoner": Fake("reasoner", built, "claude-sonnet-5-5"), "extractor": Fake("extractor", built, "claude-sonnet-5-5"),
                  "judge": Fake("judge", built, "claude-haiku-5-5")}
        if fail_every and n % fail_every == 0:
            e = live.LiveFailure(("reasoner", "extractor", "judge")[n % 3], "fake failure")
            e.cost = 0.02
            raise e
        return live.run(built, models, retriever, emit=emit)
    return run


def serve_in_thread(bank_dir, runner, per_hour=1000, per_day=1000, cap=1000.0, slots=3):
    import uvicorn

    from strokecopilot.demo import limits, server

    state = server.State(bank_dir, runner=runner, live_enabled=True, key_present=True,
                         rate=limits.RateLimiter(per_hour, per_day), spend=limits.SpendCap(cap), slots=limits.Slots(slots))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(server.create_app(state), host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", srv, state


def main(argv=None):
    parser = argparse.ArgumentParser(description="DEC-32 load test.")
    parser.add_argument("--bank", default="demo/space/build/bank")
    parser.add_argument("--url", help="a running server; live runs must be off there")
    parser.add_argument("--visitors", type=int, default=50)
    parser.add_argument("--spread", type=float, default=0.0, help="arrivals spread over this many seconds (0: all at once)")
    args = parser.parse_args(argv)
    if args.url:
        report = asyncio.run(run_load(args.url.rstrip("/"), args.visitors, spread_s=args.spread))
    else:
        base, srv, state = serve_in_thread(args.bank, fake_runner())
        report = asyncio.run(run_load(base, args.visitors, spread_s=args.spread))
        report["server"] = {"spend": state.spend.status(), "cache_entries": len(state.cache), "slot_free": state.slots.try_acquire()}
        srv.should_exit = True
    print(json.dumps(report, indent=1))
    ok = report["results"] == report["visitors"] and not report["errors"]
    print(("OK: " if ok else "NOT OK: ") + f"{report['results']}/{report['visitors']} visitors got a result; "
          f"errors {len(report['errors'])}; by source {report['by_source']}; median {report['seconds']['median']} s, "
          f"worst {report['seconds']['max']} s.")
    return report


if __name__ == "__main__":
    main()
