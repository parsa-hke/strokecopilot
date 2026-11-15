"""Human adjudication kit: sample claims across models (flagged and clean), blind the model identity, export a labelling
sheet with a separate key, and compute agreement statistics (docs/HALLUCINATION_METRIC.md §7-8, plan §7 Phase 3 NLP B).

Design (Phase 3, Benny; review fixes by Parsa, 12 and 15 Nov 2026):
- Every claim the verifier scored is a candidate, not only the flagged ones, so the verifier's misses are measured too.
  The claims come from each run's `traces/<case>.json` (written by pipeline/end_to_end.py, with the claim IDs). They
  are joined to the run's flagged-claims file (`flagged*.json`) by (case, claim ID): a claim listed there takes its
  sub-type and criticality from it, every other claim is clean. The claim text is never used to join, because the
  extractor rewords claims and one output entry gives several atomic claims (`rationale[0].0`, `rationale[0].1`, ...).
- A rescore rebuilds the regex and citation claims and drops restatements with the code of its day, so the claims it
  scored can differ from the ones saved in `traces/`. The kit uses the claim set whose per-case count matches the
  metrics file that goes with the flagged file, and refuses the run if neither does (see `collect_all_claims`).
- A run folder without a flagged-claims file is refused: reading it as "no flags" would call every claim clean.
- Stratified by model x verifier verdict (clean; content error, sub-types 1-5; citation error, 6-7), about 100 claims
  in total, equal allocation across strata, fixed seed. Flagged claims are a few percent of all claims, so a
  proportional sample would hold too few of them for agreement per sub-type. Each sampled claim carries its stratum's
  size and its weight (claims in the stratum / claims sampled from it), so overall agreement can be re-weighted to the
  population of claims.
- Blind: the sheet shows the claim, the field it came from and its declared sources, and an opaque case alias. No
  model, experiment, run folder, case ID or verifier verdict: showing the verifier's sub-type would anchor the labellers.
  The case's inputs (imaging JSON and note) are written per alias, so labellers can check the claim against them.
- Unblinding goes through a separate key file (sample ID -> model, run, case, claim, verifier verdict, weight), kept
  away from the labellers, so agreement can be reported per model (§7: judge bias).
- Two labellers (Lorna and Mohamed) label each row independently as `error` or `no_error`; where they disagree, a short
  discussion fills the `consensus` column. Cohen's kappa between them uses every row both labelled. Agreement of the
  verifier with the consensus is reported overall, per verdict, per sub-type and per model.
- Held-out cases are refused through the split's own check, never by opening the split file.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from strokecopilot.data import splits

LABELS = ("error", "no_error")
# Labels of the first sheets, which held flagged claims only: "verifier_mistake" there meant the claim was fine.
OLD_LABELS = {"real_reasoner_error": "error", "verifier_mistake": "no_error"}
FLAGGED_FILES = ("flagged_rescore.json", "flagged.json")  # newest scoring first
CONTENT_SUBTYPES = ("wrong_number", "invented_finding", "wrong_rule", "misapplied_rule", "faulty_derivation")
CITATION_SUBTYPES = ("uncited_rule", "misattributed_source")
VERDICTS = ("clean", "content_error", "citation_error")
SHEET_FIELDS = ["sample_id", "case_alias", "field", "claim", "declared_sources", "label_A", "label_B", "consensus",
                "note_A", "note_B", "note_consensus"]
KEY_FIELDS = ["sample_id", "model", "experiment", "run_folder", "source", "flagged_file", "case", "claim_id", "verdict",
              "subtype", "critical", "reason", "stratum_size", "stratum_sampled", "weight"]


def _flagged_file(folder, flagged_name=None):
    """The flagged-claims file to read, or None when the folder has none."""
    if flagged_name:
        return folder / flagged_name if (folder / flagged_name).exists() else None
    for name in FLAGGED_FILES:
        if (folder / name).exists():
            return folder / name
    return None


def verdict_of(subtype):
    """The verifier's verdict for a claim: clean (not flagged), content_error (1-5) or citation_error (6-7)."""
    if not subtype:
        return "clean"
    if subtype in CONTENT_SUBTYPES:
        return "content_error"
    if subtype in CITATION_SUBTYPES:
        return "citation_error"
    raise ValueError(f"Unknown sub-type {subtype!r}.")


def _scored_counts(folder, flagged_path):
    """Claims per case in the metrics file written with this flagged file (flagged_X.json -> metrics_X.json), or {}."""
    path = folder / flagged_path.name.replace("flagged", "metrics", 1)
    if not path.exists():
        return {}
    metrics = json.loads(path.read_text())
    rows = metrics.get("per_case") or metrics.get("per_fixture") or []
    return {r.get("case") or r.get("fixture"): r["claims"] for r in rows if r.get("claims") is not None}


def _case_claims(folder, case, flagged_path, counts, claims):
    """The claims the scoring counted for one case: as saved in traces/, or refreshed with the current code (as a
    rescore does). `claims` is "auto", "saved" or "refreshed"; see `collect_all_claims`."""
    saved = json.loads((folder / "traces" / f"{case}.json").read_text())["claims"]
    if claims == "saved":
        return saved

    def refreshed():
        from strokecopilot.nlp.verifier import run as vr  # imported here: only a rescored run needs it
        rec = json.loads((folder / "outputs" / f"{case}.json").read_text())
        return [c.to_dict() for c in vr.refresh_claims(rec, saved)]

    if claims == "refreshed":
        return refreshed()
    if claims != "auto":
        raise ValueError(f"claims must be 'auto', 'saved' or 'refreshed', not {claims!r}.")
    original = flagged_path is None or flagged_path.name == "flagged.json"  # the run's own scoring, not a rescore
    order = [lambda: saved, refreshed] if original else [refreshed, lambda: saved]
    expected = counts.get(case)
    if expected is None:
        return order[0]()
    for build in order:
        candidate = build()
        if len(candidate) == expected:
            return candidate
    raise ValueError(f"{folder.name}: no claim set for one case matches the {expected} claims its scoring counted "
                     f"({flagged_path.name}). It was scored with older code; rescore the run, or pass claims='saved' "
                     "or 'refreshed' knowingly.")


def collect_all_claims(run_folders, flagged_name=None, allow_missing_flagged=False, claims="auto"):
    """Every claim the verifier scored in every run folder, with its verdict, the model and the run it came from.

    Each entry: {model, experiment, run_folder, source, flagged_file, case, claim_id, field, text, sources, verdict,
    subtype, critical, reason}. Claims come from traces/<case>.json and are joined to the flagged file by (case,
    claim ID); a claim the flagged file does not list is clean. `flagged_name` picks the flagged file (for example
    "flagged_phase3_judge_v3.json"); by default the newest scoring. A folder without one raises, unless
    `allow_missing_flagged` (then every claim is clean, and `flagged_file` is None).

    `claims`: "auto" (default) takes, per case, the claim set whose count matches the metrics file of the same scoring
    (saved claims first for flagged.json, refreshed first for a rescore), and raises if neither matches; "saved" or
    "refreshed" forces one. A flagged claim ID missing from the claims raises."""
    out = []
    for folder in map(Path, run_folders):
        config = json.loads((folder / "config.json").read_text())
        flagged_path = _flagged_file(folder, flagged_name)
        if flagged_path is None and not allow_missing_flagged:
            looked = flagged_name or ", ".join(FLAGGED_FILES)
            raise FileNotFoundError(f"No flagged-claims file in {folder} (looked for {looked}). Without it every claim "
                                    "would count as clean; pass allow_missing_flagged=True only if that is meant.")
        flagged = json.loads(flagged_path.read_text()) if flagged_path else []
        cases = sorted(p.stem for p in (folder / "traces").glob("*.json"))
        splits.check_ids(set(cases) | {f["case"] for f in flagged})  # refuses held-out cases before Phase 4
        verdicts = {(f["case"], f["claim_id"]): f for f in flagged}
        missing_traces = {f["case"] for f in flagged} - set(cases)
        if missing_traces:
            raise ValueError(f"{folder.name}: {len(missing_traces)} flagged case(s) have no traces file.")
        counts = _scored_counts(folder, flagged_path) if flagged_path else {}
        run = {"model": (config.get("reasoner") or {}).get("model", "unknown"),
               "experiment": config.get("experiment", folder.name), "run_folder": str(folder),
               "source": config.get("source") or config.get("mode") or "fixtures",
               "flagged_file": flagged_path.name if flagged_path else None}
        joined = set()
        for case in cases:
            for claim in _case_claims(folder, case, flagged_path, counts, claims):
                flag = verdicts.get((case, claim["id"]))
                if flag:
                    joined.add((case, claim["id"]))
                subtype = flag["subtype"] if flag else None
                out.append({**run, "case": case, "claim_id": claim["id"], "field": _field(claim["id"]),
                            "text": claim.get("text", ""), "sources": _declared(claim.get("declared_sources") or []),
                            "verdict": verdict_of(subtype), "subtype": subtype,
                            "critical": flag.get("critical") if flag else None,
                            "reason": flag.get("reason", "") if flag else ""})
        unmatched = set(verdicts) - joined
        if unmatched:
            raise ValueError(f"{folder.name}: {len(unmatched)} flagged claim ID(s) in {flagged_path.name} are not among "
                             "the claims read from traces/; the flagged file and the claims come from different code.")
    return out


def _declared(sources):
    """Declared sources as short strings, as pipeline/end_to_end.flagged_claims writes them."""
    return [s.get("field") or s.get("id") or (s.get("quote") or "")[:40] if isinstance(s, dict) else str(s)
            for s in sources]


def sample_claims(all_claims, n_total=100, seed=20261109):
    """Stratified sample by (model, verdict), about `n_total` claims in all: equal allocation across strata, a stratum
    smaller than its share gives all its claims and the rest goes to the others. Fixed seed; the input is not changed.

    Each sampled entry gains `stratum_size` (claims in its stratum), `stratum_sampled` (claims sampled from it) and
    `weight` (stratum_size / stratum_sampled), the inverse sampling fraction for population-weighted agreement."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for entry in all_claims:
        strata[(entry["model"], entry["verdict"])].append(entry)
    pools = {}
    for key in sorted(strata):
        pool = list(strata[key])
        rng.shuffle(pool)
        pools[key] = pool
    taken = {key: 0 for key in pools}
    remaining = min(n_total, len(all_claims))
    while remaining:
        for key, pool in pools.items():
            if remaining and taken[key] < len(pool):
                taken[key] += 1
                remaining -= 1
    sampled = []
    for key, pool in pools.items():
        size, k = len(pool), taken[key]
        sampled.extend({**e, "stratum_size": size, "stratum_sampled": k, "weight": size / k} for e in pool[:k])
    return sampled


def strata(sampled):
    """The stratum sizes of a sample: {(model, verdict): {"size": claims in the stratum, "sampled": claims drawn}}."""
    out = {}
    for e in sampled:
        out[(e["model"], e["verdict"])] = {"size": e["stratum_size"], "sampled": e["stratum_sampled"]}
    return dict(sorted(out.items()))


def _field(claim_id):
    """'thrombolysis.blocking_factors[0].2' -> 'thrombolysis.blocking_factors'."""
    return claim_id.split("[", 1)[0]


def _sources_text(sources):
    return "; ".join(str(s) for s in sources) if sources else "(none declared)"


def blind_and_export(sampled, sheet_path, key_path, seed=20261109):
    """Write the labellers' sheet and the separate key. Rows are shuffled, sample IDs and case aliases are assigned in
    shuffled order, so neither hints at the model, the case or the verdict. Returns (sheet_path, key_path, aliases),
    where aliases maps (run_folder, case) to its alias. The input list is not changed."""
    rng = random.Random(seed)
    rows = list(sampled)
    rng.shuffle(rows)
    aliases, sheet, key = {}, [], []
    for i, entry in enumerate(rows, 1):
        alias = aliases.setdefault((entry["run_folder"], entry["case"]), f"C{len(aliases) + 1:03d}")
        sample_id = f"S{i:03d}"
        sheet.append({"sample_id": sample_id, "case_alias": alias, "field": _field(entry["claim_id"]),
                      "claim": entry["text"], "declared_sources": _sources_text(entry["sources"]),
                      "label_A": "", "label_B": "", "consensus": "", "note_A": "", "note_B": "", "note_consensus": ""})
        key.append({"sample_id": sample_id, **{k: entry.get(k) for k in KEY_FIELDS if k != "sample_id"}})
    for path, fields, data in ((Path(sheet_path), SHEET_FIELDS, sheet), (Path(key_path), KEY_FIELDS, key)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    return Path(sheet_path), Path(key_path), aliases


def export_case_inputs(aliases, out_dir, load_cases=None):
    """One file per case alias with what the reasoner was given: the imaging JSON (without its case ID and model
    version) and the note. `load_cases(source)` defaults to pipeline.end_to_end.load_cases; the source comes from each
    run's config."""
    if load_cases is None:
        from strokecopilot.pipeline.end_to_end import load_cases  # imported here: end_to_end imports the verifier
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for (run_folder, case), alias in aliases.items():
        config = json.loads((Path(run_folder) / "config.json").read_text())
        source = config.get("source") or config.get("mode") or "fixtures"
        if source not in cache:
            cache[source] = {c["id"]: c for c in load_cases(source)}
        inputs = cache[source][case]
        vision = {k: v for k, v in inputs["vision"].items() if k not in ("case_id", "model_version")}
        text = (f"# Case {alias}\n\n## Imaging findings\n\n```json\n{json.dumps(vision, indent=1)}\n```\n\n"
                f"## Triage note\n\n{inputs['note']}\n")
        (out_dir / f"{alias}.md").write_text(text, encoding="utf-8")
    return out_dir


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa between two labellers: (po - pe) / (1 - pe). None when it is undefined (both labellers used the
    same single category throughout, so pe = 1)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Two label lists must be the same length.")
    if not labels_a:
        raise ValueError("No labels to compare.")
    n = len(labels_a)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    pe = sum(count_a[c] * count_b[c] for c in set(count_a) | set(count_b)) / (n * n)
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def _label(value, column, sample_id):
    value = value.strip()
    value = OLD_LABELS.get(value, value)
    if value and value not in LABELS:
        raise ValueError(f"{column} of {sample_id} must be one of {LABELS} or empty; found {value!r}.")
    return value


def read_labels(sheet_path, key_path):
    """Join the labelled sheet with its key: one row per sample with label_A, label_B and consensus (each "error",
    "no_error" or "" when not given yet; the first sheets' labels are mapped), the model and the verifier verdict."""
    with Path(key_path).open(encoding="utf-8") as f:
        key = {row["sample_id"]: row for row in csv.DictReader(f)}
    rows = []
    with Path(sheet_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["sample_id"]
            if sid not in key:
                raise ValueError(f"Sample {sid} is not in the key.")
            k = dict(key[sid])
            k["verdict"] = k.get("verdict") or verdict_of(k.get("subtype"))
            k["weight"] = float(k["weight"]) if k.get("weight") else 1.0
            rows.append({**k, **{c: _label(row.get(c) or "", c, sid) for c in ("label_A", "label_B", "consensus")}})
    return rows


def consensus_of(row):
    """The consensus column if filled, else the shared label when A and B agree, else None (unresolved)."""
    if row.get("consensus"):
        return row["consensus"]
    if row["label_A"] and row["label_A"] == row["label_B"]:
        return row["label_A"]
    return None


def _agreement(rows):
    """Agreement of the verifier (flagged = error, clean = no_error) with the consensus, unweighted and weighted."""
    if not rows:
        return {"n": 0, "agreement": None, "agreement_weighted": None}
    hits = [(consensus_of(r) == "error") == (r["verdict"] != "clean") for r in rows]
    weights = [r.get("weight", 1.0) for r in rows]
    return {"n": len(rows), "agreement": sum(hits) / len(rows),
            "agreement_weighted": sum(w for w, h in zip(weights, hits) if h) / sum(weights)}


def _grouped(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    return {k: _agreement(v) for k, v in sorted(by.items())}


def summarise_agreement(rows):
    """Kappa between the labellers on every row both labelled, and agreement of the verifier with the consensus on
    every row that has one: overall, per verdict, per sub-type ("clean" for unflagged claims) and per model.

    Unresolved disagreements (A and B differ, no consensus yet) count in kappa and are reported, but are left out of the
    agreement with the verifier. `agreement_weighted` re-weights by the sampling weights (metric §8: the sample is
    stratified, flagged claims over-represented)."""
    both = [r for r in rows if r["label_A"] and r["label_B"]]
    resolved = [r for r in rows if consensus_of(r)]
    unresolved = [r for r in both if r["label_A"] != r["label_B"] and not r.get("consensus")]
    by_model = {}
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in both if r["model"] == model]
        by_model[model] = {**_agreement([r for r in resolved if r["model"] == model]),
                           "kappa": cohen_kappa([r["label_A"] for r in mine], [r["label_B"] for r in mine]) if mine else None,
                           "unresolved": sum(r["model"] == model for r in unresolved)}
    return {
        "n": len(rows),
        "n_labelled_by_both": len(both),
        "kappa": cohen_kappa([r["label_A"] for r in both], [r["label_B"] for r in both]) if both else None,
        "n_with_consensus": len(resolved),
        "unresolved_disagreements": len(unresolved),
        "agreement_with_verifier": {
            "overall": _agreement(resolved),
            "by_verdict": _grouped(resolved, lambda r: r["verdict"]),
            "by_subtype": _grouped(resolved, lambda r: r.get("subtype") or "clean"),
            "by_model": by_model,
        },
    }
