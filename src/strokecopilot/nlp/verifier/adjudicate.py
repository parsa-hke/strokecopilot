"""Human adjudication kit: sample flagged claims across models, blind the model identity, export a labelling sheet
with a separate key, and compute agreement statistics (docs/HALLUCINATION_METRIC.md §8, plan §7 Phase 3 NLP B).

Design (Phase 3, Benny; review fixes by Parsa, 12 Nov 2026):
- Claims come from each run's flagged-claims file (`flagged*.json`, written by pipeline/end_to_end.py), which holds
  the claim text and the sources the model declared.
- Stratified by (model, sub-type) so every model's mix of error types is represented. Fixed seed.
- Blind: the sheet shows the claim, the field it came from and its declared sources, and an opaque case alias. No
  model, experiment, run folder, case ID or verifier verdict: showing the verifier's sub-type would anchor the labellers.
  The case's inputs (imaging JSON and note) are written per alias, so labellers can check the claim against them.
- Unblinding goes through a separate key file (sample ID -> model, run, case, claim, verifier sub-type), kept away from
  the labellers, so agreement can be reported per model (§7: judge bias).
- Two labellers (Lorna and Mohamed) label each row independently as real_reasoner_error or verifier_mistake. Cohen's
  kappa between them is reported, and each one's agreement with the verifier, overall, per sub-type and per model.
- Held-out cases are refused through the split's own check, never by opening the split file.

Open, for Benny (review): the sample is flagged claims only, so agreement with the verifier is its precision and
missed hallucinations are never measured. Metric §8 asks for a sample across claims; see the PR #5 review comment.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from strokecopilot.data import splits

LABELS = ("real_reasoner_error", "verifier_mistake")
FLAGGED_FILES = ("flagged_rescore.json", "flagged.json")  # newest scoring first
SHEET_FIELDS = ["sample_id", "case_alias", "field", "claim", "declared_sources", "label_A", "label_B", "note_A", "note_B"]
KEY_FIELDS = ["sample_id", "model", "experiment", "run_folder", "case", "claim_id", "subtype", "critical", "reason"]


def _flagged_file(folder, flagged_name=None):
    if flagged_name:
        return folder / flagged_name
    for name in FLAGGED_FILES:
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"No flagged-claims file in {folder} (looked for {', '.join(FLAGGED_FILES)}).")


def collect_flagged(run_folders, flagged_name=None):
    """Every flagged claim of every run folder, with the model and run it came from.

    Each entry: {model, experiment, run_folder, source, flagged_file, case, claim_id, subtype, critical, reason, text,
    sources}. `flagged_name` picks the file (for example "flagged_phase3_judge_v3.json"); by default the newest scoring.
    """
    out = []
    for folder in map(Path, run_folders):
        config = json.loads((folder / "config.json").read_text())
        path = _flagged_file(folder, flagged_name)
        flagged = json.loads(path.read_text())
        splits.check_ids({f["case"] for f in flagged})  # refuses held-out cases before Phase 4
        for flag in flagged:
            out.append({
                "model": (config.get("reasoner") or {}).get("model", "unknown"),
                "experiment": config.get("experiment", folder.name),
                "run_folder": str(folder),
                "source": config.get("source") or config.get("mode") or "fixtures",
                "flagged_file": path.name,
                "case": flag["case"],
                "claim_id": flag["claim_id"],
                "subtype": flag["subtype"],
                "critical": flag.get("critical"),
                "reason": flag.get("reason", ""),
                "text": flag.get("text", ""),
                "sources": flag.get("sources", []),
            })
    return out


def sample_claims(flagged, n_per_model=30, seed=20261109):
    """Stratified sample by (model, sub-type): at most n_per_model per model, round-robin across that model's
    sub-types. Fixed seed; the input list is not changed."""
    rng = random.Random(seed)
    by_model = defaultdict(list)
    for entry in flagged:
        by_model[entry["model"]].append(entry)
    sampled = []
    for model in sorted(by_model):
        by_subtype = defaultdict(list)
        for e in by_model[model]:
            by_subtype[e["subtype"]].append(e)
        lists = {k: list(v) for k, v in sorted(by_subtype.items())}
        for v in lists.values():
            rng.shuffle(v)
        picked = []
        while len(picked) < n_per_model and any(lists.values()):
            for subtype in lists:
                if lists[subtype] and len(picked) < n_per_model:
                    picked.append(lists[subtype].pop())
        sampled.extend(picked)
    return sampled


def _field(claim_id):
    """'thrombolysis.blocking_factors[0].2' -> 'thrombolysis.blocking_factors'."""
    return claim_id.split("[", 1)[0]


def _sources_text(sources):
    return "; ".join(str(s) for s in sources) if sources else "(none declared)"


def blind_and_export(sampled, sheet_path, key_path, seed=20261109):
    """Write the labellers' sheet and the separate key. Rows are shuffled, sample IDs and case aliases are assigned in
    shuffled order, so neither hints at the model or the case. Returns (sheet_path, key_path, aliases), where aliases maps
    (run_folder, case) to its alias. The input list is not changed."""
    rng = random.Random(seed)
    rows = list(sampled)
    rng.shuffle(rows)
    aliases, sheet, key = {}, [], []
    for i, entry in enumerate(rows, 1):
        alias = aliases.setdefault((entry["run_folder"], entry["case"]), f"C{len(aliases) + 1:03d}")
        sample_id = f"S{i:03d}"
        sheet.append({"sample_id": sample_id, "case_alias": alias, "field": _field(entry["claim_id"]),
                      "claim": entry["text"], "declared_sources": _sources_text(entry["sources"]),
                      "label_A": "", "label_B": "", "note_A": "", "note_B": ""})
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
        from strokecopilot.pipeline.end_to_end import load_cases  # imported here: end_to_end imports this package
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


def agreement_with_verifier(labels, subtypes=None):
    """Share of sampled claims the labeller also calls a real reasoner error. Every sampled claim was flagged by the
    verifier, so this is the verifier's precision as that labeller sees it. With `subtypes`, also per sub-type."""
    if not labels:
        return None
    overall = sum(label == "real_reasoner_error" for label in labels) / len(labels)
    if subtypes is None:
        return overall
    by = defaultdict(list)
    for label, subtype in zip(labels, subtypes):
        by[subtype].append(label == "real_reasoner_error")
    return {"overall": overall, "by_subtype": {k: {"n": len(v), "agreement": sum(v) / len(v)} for k, v in sorted(by.items())}}


def read_labels(sheet_path, key_path):
    """Join the labelled sheet with its key: one row per sample with both labels, the model and the verifier sub-type."""
    with Path(key_path).open(encoding="utf-8") as f:
        key = {row["sample_id"]: row for row in csv.DictReader(f)}
    with Path(sheet_path).open(encoding="utf-8") as f:
        rows = [{**key[row["sample_id"]], "label_A": row["label_A"].strip(), "label_B": row["label_B"].strip()}
                for row in csv.DictReader(f)]
    bad = {r[k] for r in rows for k in ("label_A", "label_B") if r[k] not in LABELS}
    if bad:
        raise ValueError(f"Labels must be one of {LABELS}; found {sorted(bad)}.")
    return rows


def summarise_agreement(rows):
    """Kappa between the labellers, and each one's agreement with the verifier: overall, per sub-type and per model."""
    a, b = [r["label_A"] for r in rows], [r["label_B"] for r in rows]
    subtypes = [r["subtype"] for r in rows]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    return {
        "n": len(rows),
        "kappa": cohen_kappa(a, b),
        "agreement_with_verifier_A": agreement_with_verifier(a, subtypes),
        "agreement_with_verifier_B": agreement_with_verifier(b, subtypes),
        "by_model": {m: {"n": len(rs), "kappa": cohen_kappa([r["label_A"] for r in rs], [r["label_B"] for r in rs]),
                         "agreement_with_verifier_A": agreement_with_verifier([r["label_A"] for r in rs]),
                         "agreement_with_verifier_B": agreement_with_verifier([r["label_B"] for r in rs])}
                     for m, rs in sorted(by_model.items())},
    }
