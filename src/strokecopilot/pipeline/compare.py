"""Comparison table across models and arms.

Reads saved run folders (each with config.json and metrics.json) and produces
the project's headline table: hallucination rate, critical hallucination rate,
omission rate, per model per source per arm, with CIs when available.

The heavy statistics live in nlp/verifier/metrics.py. This module only
assembles saved results into a table and a figure.

Which metrics file a row reads matters: a finished run keeps its first scoring in
metrics.json, and later rescores with fixed verifier code write metrics_rescore.json
(or a named file). Each row records the file it came from (review fix, 12 Nov 2026).
"""




import json
from pathlib import Path

import numpy as np

METRICS_FILES = ("metrics_rescore.json", "metrics.json")  # newest scoring first
METRICS = [
    "hallucination_rate",
    "critical_hallucination_rate",
    "omission_rate",
]

def _metrics_file(folder, metrics_name=None):
    if metrics_name:
        return folder / metrics_name
    for name in METRICS_FILES:
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"No metrics file in {folder}")

def load_run(run_folder, metrics_name=None):
    folder = Path(run_folder)
    config = json.loads((folder / "config.json").read_text())
    path = _metrics_file(folder, metrics_name)
    metrics = json.loads(path.read_text())
    return {
        "model": (config.get("reasoner") or {}).get("model", "unknown"),
        "source": config.get("source") or config.get("mode") or "unknown",
        "arm": _arm_label(config),
        "metrics_file": path.name,
        "summary": metrics.get("summary") or {},
    }

def _arm_label(config):
    if config.get("arm"):
        return config["arm"]
    if config.get("with_rulebook") is False:
        return "without_rulebook"
    return "with_rulebook"

def build_table(run_folders, metrics_name=None):
    """Builds a pivot table: rows are models, columns are source x arm."""
    runs = [load_run(*f) if isinstance(f, tuple) else load_run(f, metrics_name) for f in run_folders]
    
    # Group data by model (the rows)
    by_model = {}
    for run in runs:
        model = run["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(run)

    table = []
    for model, model_runs in by_model.items():
        row = {"model": model}
        for run in model_runs:
            # The column name: e.g., "fixtures_with_rulebook"
            col_key = f"{run['source']}_{run['arm']}"
            for metric in METRICS:
                m = run["summary"].get(metric, {})
                row[f"{col_key}_{metric}"] = m.get("value")
                row[f"{col_key}_{metric}_ci"] = m.get("ci95")
        table.append(row)
    return table
