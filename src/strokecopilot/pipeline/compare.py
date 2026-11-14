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

#--

def plot_table(table, out_path):
    """Bar chart with CI error bars for the three main metrics."""
    import matplotlib.pyplot as plt

    if not table:
        raise ValueError("Empty table.")
        
    labels = [f"{r['model']}\n({r.get('source', 'unknown')}, {r.get('arm', 'unknown')})" for r in table]
    x = np.arange(len(table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(table)), 5))
    
    for i, metric in enumerate(METRICS):
        values = [r.get(metric) if r.get(metric) is not None else np.nan for r in table]
        errors = _errors_from_ci(table, metric)
        
        bars = ax.bar(x + (i - 1) * width, values, width, label=metric.replace("_", " "),
                      yerr=errors, capsize=3)
                      
        for bar, value in zip(bars, values):
            label = "n/a" if np.isnan(value) else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, 0 if np.isnan(value) else bar.get_height(),
                    label, ha="center", va="bottom", fontsize=8)
                    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_title("Hallucination, critical hallucination, omission")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _errors_from_ci(table, metric):
    """Lower/upper error bars from CIs. Zero where the CI is missing or the value is null."""
    lower = []
    upper = []
    for r in table:
        ci = r.get(f"{metric}_ci")
        value = r.get(metric)
        if ci is None or value is None:
            lower.append(0)
            upper.append(0)
        else:
            lower.append(max(0, value - ci[0]))
            upper.append(max(0, ci[1] - value))
    return [lower, upper]
