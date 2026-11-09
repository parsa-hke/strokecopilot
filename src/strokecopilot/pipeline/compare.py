"""Comparison table across models and arms.

Reads saved run folders (each with config.json and metrics.json) and produces
the project's headline table: hallucination rate, critical hallucination rate,
omission rate, per model per source per arm, with CIs when available.

The heavy statistics live in nlp/verifier/metrics.py. This module only
assembles saved results into a table and a figure.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "hallucination_rate",
    "critical_hallucination_rate",
    "omission_rate",
]


def load_run(run_folder):
    """Read one run folder. Returns dict with config, summary, per_fixture."""
    folder = Path(run_folder)
    config = json.loads((folder / "config.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    return {
        "experiment": config.get("experiment", folder.name),
        "model": config.get("reasoner", {}).get("model", "unknown"),
        "source": config.get("mode", "unknown"),
        "arm": _arm_label(config),
        "summary": metrics.get("summary", {}),
        "per_fixture": metrics.get("per_fixture", []),
        "run_folder": str(folder),
    }


def _arm_label(config):
    """Derive the experimental arm from config. Default is 'with_rulebook' (DEC-19)."""
    if config.get("with_rulebook") is False:
        return "without_rulebook"
    return "with_rulebook"


def build_table(run_folders):
    """Assemble rows from a list of run folders. Each run = one row."""
    rows = [load_run(f) for f in run_folders]
    table = []
    for row in rows:
        entry = {
            "experiment": row["experiment"],
            "model": row["model"],
            "source": row["source"],
            "arm": row["arm"],
            "run_folder": row["run_folder"],
        }
        for metric in METRICS:
            m = row["summary"].get(metric, {})
            entry[metric] = m.get("value")
            entry[f"{metric}_ci"] = m.get("ci95")
        table.append(entry)
    return table


def plot_table(table, out_path):
    """Bar chart with CI error bars for the three main metrics."""
    if not table:
        raise ValueError("Empty table.")
    labels = [f"{r['experiment']}\n{r['model']}\n{r['arm']}" for r in table]
    x = np.arange(len(table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(table)), 5))
    for i, metric in enumerate(METRICS):
        values = [r[metric] if r[metric] is not None else 0 for r in table]
        errors = _errors_from_ci(table, metric)
        bars = ax.bar(x + (i - 1) * width, values, width, label=metric.replace("_", " "),
                      yerr=errors, capsize=3)
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{value:.3f}", ha="center", va="bottom", fontsize=8)
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