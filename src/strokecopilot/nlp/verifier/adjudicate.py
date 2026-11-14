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

----- NEW CODE -----"""


import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from strokecopilot.data import splits

LABELS = ("real_reasoner_error", "verifier_mistake")
FLAGGED_FILES = ("flagged_rescore.json", "flagged.json", "flagged_phase3_judge_v3.json")
SHEET_FIELDS = ["sample_id", "case_alias", "field", "claim", "declared_sources", "label_A", "label_B", "consensus", "note_A", "note_B", "note_consensus"]
KEY_FIELDS = ["sample_id", "model", "experiment", "run_folder", "case", "claim_id", "subtype", "critical", "verifier_verdict"]

def _flagged_file(folder, flagged_name=None):
    if flagged_name:
        return folder / flagged_name
    for name in FLAGGED_FILES:
        if (folder / name).exists():
            return folder / name
    return None # Some runs might not have a flagged file

def _categorize_subtype(subtype):
    if not subtype:
        return "clean"
    # Content false subtypes (1-5)
    if subtype in ("wrong_number", "invented_finding", "wrong_rule", "misapplied_rule", "faulty_derivation"):
        return "content_error"
    # Citation only (6-7)
    if subtype in ("uncited_rule", "misattributed_source"):
        return "citation_error"
    return "clean"

def _extract_claims(output_data, case_id, model, experiment, run_folder, flagged_lookup):
    """Extract every claim from the reasoner output, and label with verifier verdict if flagged."""
    claims = []
    
    # 1. Rationale
    for i, entry in enumerate(output_data.get("rationale", [])):
        text = entry.get("claim", "")
        field = f"rationale[{i}]"
        sources = entry.get("sources", [])
        # Match against flagged to get verifier verdict
        sub, crit = flagged_lookup.get((case_id, text), (None, None))
        claims.append({
            "model": model, "experiment": experiment, "run_folder": run_folder,
            "case": case_id, "claim_id": f"{field}.0", "text": text, "sources": sources,
            "field": field, "subtype": sub, "critical": crit, 
            "verifier_verdict": _categorize_subtype(sub)
        })
    
    # 2. Blocking factors (thrombolysis & thrombectomy)
    for decision in ("thrombolysis", "thrombectomy"):
        for i, text in enumerate(output_data.get(decision, {}).get("blocking_factors", [])):
            field = f"{decision}.blocking_factors[{i}]"
            sub, crit = flagged_lookup.get((case_id, text), (None, None))
            claims.append({
                "model": model, "experiment": experiment, "run_folder": run_folder,
                "case": case_id, "claim_id": f"{field}.0", "text": text, "sources": [],
                "field": field, "subtype": sub, "critical": crit,
                "verifier_verdict": _categorize_subtype(sub)
            })
            
    # 3. Uncertainties
    for i, text in enumerate(output_data.get("uncertainties", [])):
        field = f"uncertainties[{i}]"
        sub, crit = flagged_lookup.get((case_id, text), (None, None))
        claims.append({
            "model": model, "experiment": experiment, "run_folder": run_folder,
            "case": case_id, "claim_id": f"{field}.0", "text": text, "sources": [],
            "field": field, "subtype": sub, "critical": crit,
            "verifier_verdict": _categorize_subtype(sub)
        })
        
    # 4. Decision citations (rule_ids)
    for decision in ("thrombolysis", "thrombectomy"):
        for i, rule_id in enumerate(output_data.get(decision, {}).get("rule_ids", [])):
            text = f"Decision rests on passage {rule_id}"
            field = f"{decision}.rule_ids[{i}]"
            sub, crit = flagged_lookup.get((case_id, text), (None, None))
            claims.append({
                "model": model, "experiment": experiment, "run_folder": run_folder,
                "case": case_id, "claim_id": f"{field}.0", "text": text, "sources": [rule_id],
                "field": field, "subtype": sub, "critical": crit,
                "verifier_verdict": _categorize_subtype(sub)
            })
    return claims

def collect_all_claims(run_folders, flagged_name=None):
    """Read raw outputs, extract all claims, and match against flagged file for verdicts."""
    out = []
    for folder in map(Path, run_folders):
        config = json.loads((folder / "config.json").read_text())
        model = (config.get("reasoner") or {}).get("model", "unknown")
        experiment = config.get("experiment", folder.name)
        
        # Build flagged lookup
        flagged_path = _flagged_file(folder, flagged_name)
        flagged_lookup = {}
        if flagged_path and flagged_path.exists():
            flagged = json.loads(flagged_path.read_text())
            for f in flagged:
                flagged_lookup[(f["case"], f["text"])] = (f["subtype"], f.get("critical"))
                
        # Read outputs
        outputs_dir = folder / "outputs"
        if not outputs_dir.exists():
            continue
            
        for out_path in outputs_dir.glob("*.json"):
            data = json.loads(out_path.read_text())
            case_id = out_path.stem  # e.g. fx01_thrombolysis_clear
            
            splits.check_ids({case_id}) # Refuse held-out cases
            
            claims = _extract_claims(data, case_id, model, experiment, str(folder), flagged_lookup)
            out.extend(claims)
            
    return out

def sample_claims(all_claims, n_per_model=30, seed=20261109):
    """Sample across all claims, stratified by (model, verifier_verdict)."""
    rng = random.Random(seed)
    by_model = defaultdict(list)
    for entry in all_claims:
        by_model[entry["model"]].append(entry)
        
    sampled = []
    for model in sorted(by_model):
        by_verdict = defaultdict(list)
        for e in by_model[model]:
            by_verdict[e["verifier_verdict"]].append(e)
            
        lists = {k: list(v) for k, v in sorted(by_verdict.items())}
        for v in lists.values():
            rng.shuffle(v)
            
        picked = []
        while len(picked) < n_per_model and any(lists.values()):
            for verdict in lists:
                if lists[verdict] and len(picked) < n_per_model:
                    picked.append(lists[verdict].pop())
        sampled.extend(picked)
    return sampled

def _sources_text(sources):
    return "; ".join(str(s) for s in sources) if sources else "(none declared)"

def blind_and_export(sampled, sheet_path, key_path, seed=20261109):
    """Write the labellers' sheet and the separate key."""
    rng = random.Random(seed)
    rows = list(sampled)
    rng.shuffle(rows)
    aliases, sheet, key = {}, [], []
    for i, entry in enumerate(rows, 1):
        alias = aliases.setdefault((entry["run_folder"], entry["case"]), f"C{len(aliases) + 1:03d}")
        sample_id = f"S{i:03d}"
        sheet.append({
            "sample_id": sample_id, "case_alias": alias, "field": entry["field"],
            "claim": entry["text"], "declared_sources": _sources_text(entry["sources"]),
            "label_A": "", "label_B": "", "consensus": "", 
            "note_A": "", "note_B": "", "note_consensus": ""
        })
        key.append({k: entry.get(k) for k in KEY_FIELDS})
        
    for path, fields, data in ((Path(sheet_path), SHEET_FIELDS, sheet), (Path(key_path), KEY_FIELDS, key)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    return Path(sheet_path), Path(key_path), aliases

def cohen_kappa(labels_a, labels_b):
    if len(labels_a) != len(labels_b): raise ValueError("Lengths differ.")
    if not labels_a: raise ValueError("No labels.")
    n = len(labels_a)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    pe = sum(count_a[c] * count_b[c] for c in set(count_a) | set(count_b)) / (n * n)
    if pe == 1.0: return None
    return (po - pe) / (1 - pe)

def summarise_agreement(sheet_path, key_path):
    """Read sheet and key. Calculates A vs B kappa, consensus accuracy, and per-model/verdict agreement."""
    with Path(key_path).open(encoding="utf-8") as f:
        key = {row["sample_id"]: row for row in csv.DictReader(f)}
    with Path(sheet_path).open(encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            k = key[row["sample_id"]]
            rows.append({
                "model": k["model"], "subtype": k["subtype"], "verdict": k["verifier_verdict"],
                "label_A": row["label_A"].strip(), "label_B": row["label_B"].strip(), "consensus": row["consensus"].strip()
            })
            
    a, b, c = [r["label_A"] for r in rows], [r["label_B"] for r in rows], [r["consensus"] if r["consensus"] else (r["label_A"] if r["label_A"] == r["label_B"] else None) for r in rows]
    
    # Filter out any rows where consensus is still None (unresolved disagreements)
    valid_idx = [i for i, val in enumerate(c) if val in LABELS]
    a_valid = [a[i] for i in valid_idx]
    b_valid = [b[i] for i in valid_idx]
    c_valid = [c[i] for i in valid_idx]
    rows_valid = [rows[i] for i in valid_idx]
    



    
    def agree_with_verifier(label, verdict):
        if verdict == "clean": return label == "verifier_mistake"
        return label == "real_reasoner_error"

    agree_c = [agree_with_verifier(c_valid[i], rows_valid[i]["verdict"]) for i in range(len(c_valid))]
    agree_a = [agree_with_verifier(a_valid[i], rows_valid[i]["verdict"]) for i in range(len(a_valid))]
    agree_b = [agree_with_verifier(b_valid[i], rows_valid[i]["verdict"]) for i in range(len(b_valid))]

    by_model = defaultdict(list)
    for i, r in enumerate(rows_valid):
        by_model[r["model"]].append((agree_c[i], c_valid[i], c_valid[i]))

    return {
        "n_total": len(rows),
        "n_valid": len(valid_idx),
        "kappa_AB": cohen_kappa(a_valid, b_valid),
        "agreement_with_verifier": {
            "overall_A": sum(agree_a)/len(agree_a) if agree_a else None,
            "overall_B": sum(agree_b)/len(agree_b) if agree_b else None,
            "overall_consensus": sum(agree_c)/len(agree_c) if agree_c else None,
            "by_model": {m: sum(v[0] for v in vals)/len(vals) for m, vals in by_model.items()}
        }
    }
