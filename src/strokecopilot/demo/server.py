"""The Space's server (DEC-32 §2, §5): static page, precomputed bank, live runs with progress, and fallback.

    POST /api/run              {"inputs": {...}} -> {"run_id"}; validated, then cache, limits, a live slot or fallback
    GET  /api/run/{id}/events  server-sent events: progress per step, then one "result" event
    GET  /api/meta             banner, the builder's bounds, choices and defaults, whether live runs are on
    GET  /api/bank/index       the bank's index
    GET  /api/bank/case/{id}   one case file
    GET  /healthz              health: bank loaded, key present (yes/no), live on, spend today, last success/failure
    GET  /                     the static page

A run always ends with a result: a live view, a cached one, or the nearest precomputed scenario with the reason. The
key is read from the environment only (the Space's secret, DEC-32 §8); nothing logs it. Settings come from Space
variables: LIVE_ENABLED, DAILY_CAP_USD, PER_HOUR, PER_DAY, MAX_CONCURRENT.
"""

import asyncio
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from strokecopilot.demo import bank, builder, limits, live
from strokecopilot.nlp.verifier import scenarios as sc

RUN_TTL_S = 900
WATCHDOG_S = live.DEADLINE_S + 30
HEARTBEAT_S = 10
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
       "connect-src 'self'; font-src 'self'; frame-ancestors https://huggingface.co https://*.hf.space; base-uri 'none'; form-action 'self'")


def env_flag(name, default):
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


class Run:
    def __init__(self, built=None):
        self.built = built
        self.id = uuid.uuid4().hex
        self.created = time.time()
        self.events, self.done = [], False
        self._lock = threading.Lock()

    def emit(self, event):
        with self._lock:
            if not self.done:
                self.events.append({"t": round(time.time() - self.created, 1), **event})

    def finish(self, result):
        with self._lock:
            if not self.done:
                self.events.append({"t": round(time.time() - self.created, 1), "type": "result", **result})
                self.done = True

    def since(self, n):
        with self._lock:
            return self.events[n:], self.done


class State:
    """Everything the app holds: the bank, the limits, the runs. `runner(built, emit)` makes one live run and returns
    live.run's result; on failure it raises live.LiveFailure with a `cost` attribute (USD spent before failing)."""

    def __init__(self, bank_dir, runner=None, live_enabled=None, key_present=None, rate=None, spend=None, slots=None,
                 cache=None):
        self.bank_dir = Path(bank_dir)
        index = json.loads((self.bank_dir / "index.json").read_text())
        self.index = index["cases"]
        self.ids = {row["id"] for row in self.index}
        self.fallback_bank = [row for row in self.index if row.get("features") and bank.FALLBACK_MODEL in row["models"]]
        self.key_present = bool(os.environ.get("ANTHROPIC_API_KEY")) if key_present is None else key_present
        self.live_enabled = (env_flag("LIVE_ENABLED", True) and self.key_present) if live_enabled is None else live_enabled
        self.runner = runner
        self.rate = rate or limits.RateLimiter(int(os.environ.get("PER_HOUR", limits.PER_HOUR)),
                                               int(os.environ.get("PER_DAY", limits.PER_DAY)))
        self.spend = spend or limits.SpendCap(float(os.environ.get("DAILY_CAP_USD", limits.DAILY_CAP_USD)))
        self.slots = slots or limits.Slots(int(os.environ.get("MAX_CONCURRENT", limits.MAX_CONCURRENT)))
        self.cache = cache or limits.ResultCache()
        self.executor = ThreadPoolExecutor(max_workers=self.slots.n)
        self.runs, self.last = {}, {"live_success": None, "live_failure": None}
        self.watchdog_s = WATCHDOG_S
        self._lock = threading.Lock()

    def case_file(self, case_id):
        return json.loads((self.bank_dir / "cases" / f"{case_id}.json").read_text())

    def new_run(self, built=None):
        with self._lock:
            now = time.time()
            for rid in [r for r, run in self.runs.items() if now - run.created > RUN_TTL_S]:
                del self.runs[rid]
            run = Run(built)
            self.runs[run.id] = run
            return run

    def fallback(self, run, built, reason):
        target = builder.features(built.vision, built.facts, built.expected)
        entry, differs = builder.nearest(target, self.fallback_bank)
        file = self.case_file(entry["id"])
        arm = file["arms"]["rulebook"]
        answer = arm["answers"][bank.FALLBACK_MODEL]
        run.finish({"source": "precomputed", "reason": reason, "preset_id": entry["id"], "preset_title": entry["title"],
                    "differs": differs,
                    "message": f"{reason}. Showing the closest precomputed scenario instead, answered by Sonnet 5.5 earlier "
                               "with the same prompt and verifier.",
                    "view": {"banner": live.BANNER, "vision": arm["vision"], "note": arm["note"], "expected": file["rules"],
                             **{k: answer[k] for k in ("recommendation", "claims", "decisions", "omissions",
                                                       "uncertainty_omissions", "counts")}}})

    def start(self, built, visitor):
        run = self.new_run(built)
        run.emit({"type": "progress", "step": "note", "status": "done", "detail": {"note": built.note, "vision": built.vision}})
        run.emit({"type": "progress", "step": "rules", "status": "done", "detail": {"expected": built.expected}})
        cached = self.cache.get(built.key)
        if cached is not None:
            run.finish({"source": "cache", "message": "This combination was run live earlier; shown from the cache.",
                        "view": cached})
            return run
        if not self.live_enabled or self.runner is None:
            self.fallback(run, built, "Live runs are switched off")
            return run
        if not self.slots.try_acquire():
            self.fallback(run, built, "All live slots are busy")
            return run
        ok, why = self.spend.reserve()
        if not ok:
            self.slots.release()
            self.fallback(run, built, why[0].upper() + why[1:])
            return run
        ok, why = self.rate.allow(visitor)
        if not ok:
            self.spend.settle(0.0)
            self.slots.release()
            self.fallback(run, built, "You have reached the limit for live runs (" + why + ")")
            return run
        self.executor.submit(self._work, run, built)
        return run

    def _work(self, run, built):
        cost = 0.0
        try:
            def emit(step, status, detail):
                if step in ("note", "rules"):  # already reported by start(), before any live call
                    return
                run.emit({"type": "progress", "step": step, "status": status, "detail": detail})
            result = self.runner(built, emit)
            cost = result["cost"]["total"]
            self.cache.put(built.key, result["view"])
            self.last["live_success"] = time.time()
            run.finish({"source": "live", "message": "Answered live.", "view": result["view"],
                        "seconds": result["seconds"], "cost_usd": cost})
        except Exception as e:  # LiveFailure, or anything unexpected: the visitor still gets a result
            cost = getattr(e, "cost", 0.0) or 0.0
            self.last["live_failure"] = time.time()
            step = getattr(e, "step", "server")
            self.fallback(run, built, f"The live run stopped at the {step} step")
        finally:
            self.spend.settle(cost)
            self.slots.release()

    def health(self):
        return {"ok": True, "banner": live.BANNER, "bank_cases": len(self.index), "fallback_cases": len(self.fallback_bank),
                "key_present": self.key_present, "live_enabled": self.live_enabled, "spend": self.spend.status(),
                "cache_entries": len(self.cache), "last": self.last}


def default_runner():
    """The real runner: one Anthropic client, the bge retriever, live.run per call."""
    from strokecopilot.nlp.corpus import index

    api = live.client()
    retriever = index.Index.from_corpus().search

    def run(built, emit):
        models = live.llms(api)
        try:
            return live.run(built, models, retriever, emit=emit)
        except Exception as e:
            e.cost = live.cost_of(models)["total"]
            raise
    return run


def meta(state):
    return {"banner": live.BANNER, "bounds": builder.BOUNDS, "choices": builder.CHOICES, "defaults": builder.DEFAULTS,
            "deficits": list(sc.DEFICITS), "live_enabled": state.live_enabled,
            "limits": {"per_hour": state.rate.per_hour, "per_day": state.rate.per_day}}


def create_app(state, static_dir=None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Stroke Decision Copilot", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Research-Prototype"] = "not for clinical use"
        return response

    @app.get("/healthz")
    def healthz():
        return state.health()

    @app.get("/api/meta")
    def api_meta():
        return meta(state)

    @app.get("/api/bank/index")
    def bank_index():
        return FileResponse(state.bank_dir / "index.json", media_type="application/json")

    @app.get("/api/bank/charts")
    def bank_charts():
        return FileResponse(state.bank_dir / "charts.json", media_type="application/json")

    @app.get("/api/bank/passages")
    def bank_passages():
        return FileResponse(state.bank_dir / "passages.json", media_type="application/json")

    @app.get("/api/bank/case/{case_id}")
    def bank_case(case_id: str):
        if not ID_RE.match(case_id) or case_id not in state.ids:
            raise HTTPException(404, "No such case.")
        return FileResponse(state.bank_dir / "cases" / f"{case_id}.json", media_type="application/json")

    @app.post("/api/run")
    async def api_run(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "The request must be JSON.")
        try:
            built = builder.build((body or {}).get("inputs") if isinstance(body, dict) else None)
        except builder.InputError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        forwarded = request.headers.get("x-forwarded-for", "")
        address = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
        run = state.start(built, limits.visitor_id(address))
        return {"run_id": run.id}

    @app.get("/api/run/{run_id}/events")
    async def events(run_id: str):
        run = state.runs.get(run_id) if ID_RE.match(run_id) else None
        if run is None:
            raise HTTPException(404, "No such run.")

        async def stream():
            sent, last_beat, started = 0, time.monotonic(), time.monotonic()
            while True:
                new, done = run.since(sent)
                for event in new:
                    yield f"event: {event.get('type', 'progress')}\ndata: {json.dumps(event, default=str)}\n\n"
                sent += len(new)
                if done:
                    return
                if time.monotonic() - started > state.watchdog_s:
                    state.fallback(run, run.built, "The live run took too long")  # the late live answer is ignored
                    continue
                if time.monotonic() - last_beat > HEARTBEAT_S:
                    last_beat = time.monotonic()
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if static_dir and Path(static_dir).exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app


def main():
    """uvicorn entry for the Space: BANK_DIR and STATIC_DIR from the environment."""
    import uvicorn

    state = State(os.environ.get("BANK_DIR", "bank"))
    if state.live_enabled:
        state.runner = default_runner()
    app = create_app(state, os.environ.get("STATIC_DIR", "static"))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
