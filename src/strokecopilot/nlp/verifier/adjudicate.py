"""Export claim samples for human labelling; agreement statistics. Phase 3."""
"""Human adjudication kit: sample flagged claims across models, blind the model
identity, export a labelling sheet, and compute agreement statistics.

Design decisions (Phase 3, Benny):
- Sample from flagged claims only. Clean claims don't need adjudication: the
  verifier and a human would both agree they're fine.
- Stratify by (model, subtype) so every model's mix of error types is
  represented, and no subtype is over- or under-sampled.
- Blind: the exported sheet has no model name, no experiment ID, no run folder.
- Two labellers (Lorna, Mohamed) label each row independently as
  real_reasoner_error or verifier_mistake. Cohen's kappa between them is
  reported, as is each one's agreement with the verifier's subtype.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


# The two labels the labeller sheet offers. Matches EXP-12_benny_labels.csv.
LABELS = ("real_reasoner_error", "verifier_mistake")


def _subtype_to_label(subtype):
    """The verifier's subtype tells us what it thinks. Map to the two-label scheme."""
    if subtype is None:
        return None
    # Content-false subtypes: 1 wrong_number, 2 invented_finding, 3 wrong_rule,
    # 4 misapplied_rule, 5 faulty_derivation. Citation-only: 6 uncited_rule,
    # 7 misattributed_source. The adjudication question is "was the verifier's
    # flag correct?", so any subtype means "the verifier flagged this as real."
    return "real_reasoner_error"


def collect_flagged(run_folders):
    """Read every run folder's metrics.json, return a flat list of flagged claims.

    Each entry: {model, experiment, subtype, text, reason, sources, run_folder}.
    Only entries with a subtype (i.e. flagged) are kept.
    """
    out = []
    for folder in run_folders:
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        metrics = json.loads((folder / "metrics.json").read_text())
        model = config.get("reasoner", {}).get("model", "unknown")
        experiment = config.get("experiment", folder.name)
        for case in metrics.get("per_fixture", []):
            for flag in case.get("flagged", []):
                out.append({
                    "model": model,
                    "experiment": experiment,
                    "case_id": case["fixture"],
                    "claim_id": f"{case['fixture']}:{flag['claim_id']}",
                    "subtype": flag["subtype"],
                    "reason": flag.get("reason", ""),
                    "run_folder": str(folder),
                })
    return out


def sample_claims(flagged, n_per_model=30, seed=20261109):
    """Stratified sample by (model, subtype). At most n_per_model per model,
    balanced across that model's subtypes. Fixed seed for reproducibility."""
    rng = random.Random(seed)
    by_model = defaultdict(list)
    for entry in flagged:
        by_model[entry["model"]].append(entry)

    sampled = []
    for model, entries in by_model.items():
        by_subtype = defaultdict(list)
        for e in entries:
            by_subtype[e["subtype"]].append(e)
        # Round-robin across subtypes until n_per_model or exhaustion
        subtype_lists = {k: list(v) for k, v in by_subtype.items()}
        for v in subtype_lists.values():
            rng.shuffle(v)
        picked = []
        while len(picked) < n_per_model and any(subtype_lists.values()):
            for subtype in list(subtype_lists.keys()):
                if subtype_lists[subtype] and len(picked) < n_per_model:
                    picked.append(subtype_lists[subtype].pop())
        sampled.extend(picked)
    return sampled


def blind_and_export(sampled, out_path, seed=20261109):
    """Write a CSV with no model identity. Columns:
       sample_id, claim_id, subtype, reason, label_A, label_B
       sample_id is a random shuffle so row order doesn't hint at model."""
    rng = random.Random(seed)
    rng.shuffle(sampled)
    rows = []
    for i, entry in enumerate(sampled):
        rows.append({
            "sample_id": f"S{i+1:03d}",
            "claim_id": entry["claim_id"],
            "subtype": entry["subtype"],
            "reason": entry.get("reason", ""),
            "label_A": "",
            "label_B": "",
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "claim_id", "subtype", "reason", "label_A", "label_B"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa between two labeller columns. Both lists same length."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Two label lists must be same length.")
    if not labels_a:
        raise ValueError("No labels to compare.")
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))
    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n
    # Expected agreement by chance
    pe = 0.0
    for cat in categories:
        p_a = sum(1 for a in labels_a if a == cat) / n
        p_b = sum(1 for b in labels_b if b == cat) / n
        pe += p_a * p_b
    if pe == 1.0:
        return None  # perfect agreement by chance — kappa undefined
    return (po - pe) / (1 - po)


def agreement_with_verifier(labels, subtypes):
    """The verifier's view: every flagged claim is real_reasoner_error.
    So 'agreement' = fraction of claims the human also called real."""
    if not labels:
        return None
    return sum(1 for l in labels if l == "real_reasoner_error") / len(labels)


def summarise_agreement(labels_a, labels_b, subtypes):
    """Return a dict with kappa and each labeller's agreement with the verifier."""
    kappa = cohen_kappa(labels_a, labels_b)
    return {
        "n": len(labels_a),
        "kappa": kappa,
        "agreement_with_verifier_A": agreement_with_verifier(labels_a, subtypes),
        "agreement_with_verifier_B": agreement_with_verifier(labels_b, subtypes),
    }
