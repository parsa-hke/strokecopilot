"""The replay bundle (Phase 5 task I, DEC-37): rebuild the hallucination table for the fixtures and the 150 synthetic
scenarios from saved answers, with no API key.

The bundle holds, for each model and arm that has no real case in it (the fixtures and the scenarios, with and without
the rulebook), what `end_to_end.rescore` reads: the run's config, each case's call record, saved recommendation and saved
claims, and the judge's and retriever's caches; plus `expected.json`, the rates each cell reported (metric points 1-20,
the rescored figures). No real or held-out case, no ISLES'24 or RSNA data, no key: `build` scans for all of these and
refuses the bundle if it finds one.

    python -m strokecopilot.pipeline.replay build --out replay_bundle.tar.gz      # maintainers, from runs/
    python -m strokecopilot.pipeline.replay run --bundle replay_bundle.tar.gz     # anyone: verify, rescore, compare

`run` checks the bundle's sha256 against `BUNDLE_SHA256` below, rescores every cell offline (cached judge verdicts and
retrievals only; a missing one is an error, never an API call), checks each rate against `expected.json`, and prints
the table. The bundle is attached to the submission tag as a GitHub release asset (DEC-37).
"""

import hashlib
import io
import json
import re
import shutil
import tarfile
from pathlib import Path

BUNDLE_NAME = "replay_bundle.tar.gz"
BUNDLE_SHA256 = "44c8d3abb4b6e93613a12de0ccf4cfca0945bdaf100a47b203926119add3b799"  # built 20 Nov 2026 (EXP-37, EXP-47 cells; reproducible)
RELEASE_URL = "https://github.com/parsa-hke/strokecopilot/releases"
ROOTS = {"EXP-37": Path("runs/EXP-37_paid_runs"), "EXP-47": Path("runs/EXP-47_small_reasoner")}
CELLS = {
    "EXP-37": {m: ("fixtures_rulebook", "scenarios_rulebook", "fixtures_norulebook", "scenarios_norulebook")
               for m in ("claude-sonnet-5-5", "qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text")},
    "EXP-47": {m: ("fixtures_rulebook", "scenarios_rulebook") for m in ("gemma-3-4b", "gemma-3-4b-tuned")},
}
KEEP_FILES = ("config.json", "judge_cache.json", "retrieval_cache.json")
KEEP_DIRS = ("calls", "outputs", "traces")
RATES = ("hallucination_rate", "content_hallucination_rate", "critical_hallucination_rate", "omission_rate",
         "recommendation_correctness", "format_failure_rate")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}")
LOCAL_PATH = re.compile(r"/Users/|GoogleDrive|/content/drive")


def cells():
    for exp, models in CELLS.items():
        for model, arms in models.items():
            for arm in arms:
                yield exp, model, arm


def problems_in(folder, case_ids=()):
    """Files under `folder` that hold a case ID, an RSNA ID, a key pattern, an e-mail address or a local path."""
    from strokecopilot.demo import bank
    found = list(bank.scan(folder, case_ids))
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file():
            text = path.read_text(errors="ignore")
            if EMAIL.search(text):
                found.append(f"{path.name}: an e-mail address")
            if LOCAL_PATH.search(text):
                found.append(f"{path.name}: a local path")
    return found


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(out, staging):
    """Assemble, scan and pack the bundle reproducibly (sorted members, fixed times). Returns its sha256."""
    from strokecopilot.data import splits
    from strokecopilot.pipeline import compare as cp
    staging = Path(staging)
    if staging.exists():
        shutil.rmtree(staging)
    expected = {}
    for exp, model, arm in cells():
        src, dst = ROOTS[exp] / model / arm, staging / model / arm
        dst.mkdir(parents=True)
        for name in KEEP_FILES:
            if (src / name).exists():
                shutil.copyfile(src / name, dst / name)
        for name in KEEP_DIRS:
            shutil.copytree(src / name, dst / name)
        summary = json.loads(cp._metrics_file(src).read_text())["summary"]
        expected[f"{model}/{arm}"] = {k: summary[k]["value"] for k in RATES if k in summary}
    (staging / "expected.json").write_text(json.dumps(expected, indent=1, sort_keys=True))
    problems = problems_in(staging, list(splits.development_ids()))
    if problems:
        raise ValueError(f"The bundle failed its scan ({len(problems)}): {problems[:5]}. Nothing was packed.")
    import gzip
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(staging)))
                info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
                info.pax_headers = {}
                with open(path, "rb") as f:
                    tar.addfile(info, f)
    with open(out, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, compresslevel=9, mtime=0) as gz:
        gz.write(raw.getvalue())  # mtime 0 and no name in the gzip header, so the same inputs give the same bytes
    return _sha(out)


def _safe_extract(tar, dest):
    dest = Path(dest).resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)) or not member.isfile():
            raise ValueError(f"Refusing an unsafe member: {member.name}")
    tar.extractall(dest, filter="data")


def run(bundle, workdir, sha256=BUNDLE_SHA256, log=print):
    """Verify, extract, rescore every cell offline and compare with expected.json. Returns (table rows, mismatches)."""
    from strokecopilot.pipeline import compare as cp
    from strokecopilot.pipeline import end_to_end as e2e
    if sha256 is not None and _sha(bundle) != sha256:
        raise ValueError("The bundle's sha256 is not the recorded one. Download it again from the release.")
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    with tarfile.open(bundle, "r:gz") as tar:
        _safe_extract(tar, workdir)
    expected = json.loads((workdir / "expected.json").read_text())
    rows, mismatches = [], []
    for key, want in sorted(expected.items()):
        folder = workdir / key
        e2e.rescore(folder, log=lambda *a: None)  # offline: cached verdicts and retrievals only
        got = json.loads((folder / "metrics_rescore.json").read_text())["summary"]
        row = {"cell": key, **{k: got[k]["value"] for k in want}}
        rows.append(row)
        for k, v in want.items():
            if (v is None) != (got[k]["value"] is None) or (v is not None and abs(v - got[k]["value"]) > 1e-12):
                mismatches.append(f"{key} {k}: expected {v}, got {got[k]['value']}")
        log(f"{key:42s} hallucination {100 * row['hallucination_rate']:5.1f}%  content "
            f"{100 * row['content_hallucination_rate']:5.1f}%  {'ok' if not any(m.startswith(key + ' ') for m in mismatches) else 'MISMATCH'}")
    for model in sorted({k.split('/')[0] for k in expected}):
        folders = [workdir / model / a for a in ("fixtures_rulebook", "scenarios_rulebook")]
        recs = [r for f in folders for r in cp.case_records(f)]
        from strokecopilot.nlp.verifier import metrics as vm
        s = vm.summarise(recs)["hallucination_rate"]
        log(f"{model:20s} fixtures + scenarios, with the rulebook: {100 * s['value']:.1f}% "
            f"[{100 * s['ci95'][0]:.1f}, {100 * s['ci95'][1]:.1f}] over {len(recs)} cases")
    return rows, mismatches


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild the hallucination table from the replay bundle (no API key).")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=BUNDLE_NAME)
    b.add_argument("--staging", default="runs/replay_staging")
    r = sub.add_parser("run")
    r.add_argument("--bundle", default=BUNDLE_NAME)
    r.add_argument("--workdir", default="replay_work")
    r.add_argument("--no-sha-check", action="store_true", help="for a bundle built locally before BUNDLE_SHA256 is set")
    args = ap.parse_args(argv)
    if args.command == "build":
        sha = build(args.out, args.staging)
        print(json.dumps({"bundle": args.out, "sha256": sha, "mb": round(Path(args.out).stat().st_size / 1e6, 1)}))
        return
    _, mismatches = run(args.bundle, args.workdir, sha256=None if args.no_sha_check else BUNDLE_SHA256)
    if BUNDLE_SHA256 is None and not args.no_sha_check:
        print("Note: no recorded sha256 in this version of the code.")
    print("OK: every cell matches its reported rates." if not mismatches else f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches))


if __name__ == "__main__":
    main()
