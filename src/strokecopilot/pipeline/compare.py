"""Comparison table across models and arms.

Reads saved run folders (each with config.json and metrics.json) and produces
the project's headline table: hallucination rate, critical hallucination rate,
omission rate, per model per source per arm, with CIs when available.

The heavy statistics live in nlp/verifier/metrics.py. This module only
assembles saved results into a table and a figure.

Which metrics file a row reads matters: a finished run keeps its first scoring in
metrics.json, and later rescores with fixed verifier code write metrics_rescore.json
(or a named file). Each row records the file it came from (review fix, 12 Nov 2026).

Two shapes (second review, 15 Nov 2026): `build_table` gives one flat row per run folder, which `plot_table` draws;
`pivot_table` turns it into the headline table, models x (source x arm), and `format_pivot` into text for display.

C9 (17 Nov 2026), for the paid runs (EXP-37): their cells record the arm in config.json (`rulebook`, `paid_run`), so
`_arm_label` separates real cases with predicted and with ground-truth vision and Sonnet's repeats. `case_records`
rebuilds one metrics record per case from a cell's saved files (metrics.json per_case and traces/), so `paired` can
compare two cells on the same cases with metrics.paired_difference (a paired case bootstrap). `c9_report` writes the
whole C9 output.
"""

import csv
import json
from pathlib import Path

import numpy as np

from strokecopilot.nlp.verifier import metrics as vm

METRICS_FILES = ("metrics_rescore.json", "metrics.json")  # newest scoring first


METRICS = [
    "hallucination_rate",
    "critical_hallucination_rate",
    "omission_rate",
]
# Also kept in each flat row (C9): the content rate (sub-types 1-5, the fair comparison with the without-rulebook arm,
# which is not charged for citations: DEC-29), decisions, format failures, attribution and named unknowns.
EXTRA_METRICS = [
    "content_hallucination_rate",
    "recommendation_correctness",
    "format_failure_rate",
    "source_attribution_accuracy",
    "uncertainty_omission_rate",
]


def _metrics_file(folder, metrics_name=None):
    if metrics_name:
        return folder / metrics_name
    for name in METRICS_FILES:
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"No metrics file in {folder} (looked for {', '.join(METRICS_FILES)}).")


def load_run(run_folder, metrics_name=None):
    """Read one run folder: by default its newest scoring (metrics_rescore.json, else metrics.json), or the file named.
    The source is the config's `source` (current runs) or `mode` (older runs)."""
    folder = Path(run_folder)
    config = json.loads((folder / "config.json").read_text())
    path = _metrics_file(folder, metrics_name)
    metrics = json.loads(path.read_text())
    return {
        "experiment": config.get("experiment") or _experiment_from_path(folder),
        "model": (config.get("reasoner") or {}).get("model", "unknown"),
        "source": config.get("source") or config.get("mode") or "unknown",
        "arm": _arm_label(config),
        "metrics_file": path.name,
        "summary": metrics.get("summary") or {},
        "per_case": metrics.get("per_case") or metrics.get("per_fixture", []),
        "run_folder": str(folder),
    }


def _experiment_from_path(folder):
    """EXP-37 for runs/EXP-37_paid_runs/<model>/<arm>; the folder's own name when no parent names an experiment."""
    for part in (folder, *folder.parents):
        if part.name.startswith("EXP-"):
            return part.name.split("_", 1)[0]
    return folder.name


def _arm_label(config):
    """The experimental arm from config. An explicit `arm` wins. Otherwise "with_rulebook" or "without_rulebook" (DEC-19,
    DEC-29; `rulebook` as the paid runs write it, `with_rulebook` as older configs did), and for a paid-run cell on real
    cases the vision it used ("/predicted_vision", "/groundtruth_vision") and a repeat's name ("/repeat2")."""
    if config.get("arm"):
        return config["arm"]
    paid = config.get("paid_run") or {}
    rulebook = config.get("rulebook", config.get("with_rulebook", True))
    label = "with_rulebook" if rulebook is not False else "without_rulebook"
    if paid.get("source") == "real" and paid.get("vision"):
        label += f"/{paid['vision']}_vision"
    if paid.get("repeat_of"):
        label += "/" + paid["arm"].rsplit("_", 1)[-1]
    return label


def build_table(run_folders, metrics_name=None):
    """One row per run folder. An item may be a folder or (folder, metrics file name); `metrics_name` applies to the
    plain folders."""
    rows = [load_run(*f) if isinstance(f, tuple) else load_run(f, metrics_name) for f in run_folders]
    table = []
    for row in rows:
        entry = {
            "experiment": row["experiment"],
            "model": row["model"],
            "source": row["source"],
            "arm": row["arm"],
            "metrics_file": row["metrics_file"],
            "run_folder": row["run_folder"],
        }
        for metric in METRICS + EXTRA_METRICS:
            m = row["summary"].get(metric) or {}
            entry[metric] = m.get("value")
            entry[f"{metric}_ci"] = m.get("ci95")
            entry[f"{metric}_n"] = [m.get("numerator"), m.get("denominator")] if m else None
        entry["cases"] = row["summary"].get("cases")
        entry["claims"] = row["summary"].get("claims")
        table.append(entry)
    return table


def pivot_table(table, metrics=METRICS):
    """The headline shape (PR #5 review): one row per model, one column per (source, arm), each cell holding the three
    rates with their CIs and the experiment and metrics file they came from. Runs of the same model with different
    sources or arms merge into one row; two runs for the same (model, source, arm) raise, since the table would have to
    pick one silently.

    Returns {"columns": [(source, arm), ...], "rows": [{"model": m, "cells": {(source, arm): {...}}}, ...]}; a model
    without a run for a column has no cell there."""
    seen, rows = {}, {}
    for r in table:
        key = (r["model"], r["source"], r["arm"])
        if key in seen:
            raise ValueError(f"Two runs for model {r['model']}, source {r['source']}, arm {r['arm']}: "
                             f"{seen[key]} and {r['experiment']} ({r['metrics_file']}). Keep one per cell.")
        seen[key] = f"{r['experiment']} ({r['metrics_file']})"
        cell = {"experiment": r["experiment"], "metrics_file": r["metrics_file"],
                **{k: r.get(k) for m in metrics for k in (m, f"{m}_ci")}}
        rows.setdefault(r["model"], {"model": r["model"], "cells": {}})["cells"][(r["source"], r["arm"])] = cell
    columns = sorted({(r["source"], r["arm"]) for r in table})
    return {"columns": columns, "rows": [rows[m] for m in sorted(rows)]}


def _rate_text(value, ci, digits):
    if value is None:
        return "n/a"
    text = f"{value:.{digits}f}"
    return f"{text} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else text


def format_pivot(pivot, digits=3, metrics=METRICS):
    """The pivot as text for display: one row per model, {"model": m, "cells": {(source, arm, metric): text}}, where
    text is "0.054 [0.030, 0.080]", "0.054" without a CI, "n/a" for an unmeasured rate and "not run" for a missing
    cell."""
    out = []
    for row in pivot["rows"]:
        cells = {}
        for column in pivot["columns"]:
            cell = row["cells"].get(column)
            for metric in metrics:
                cells[(*column, metric)] = "not run" if cell is None else _rate_text(cell.get(metric), cell.get(f"{metric}_ci"), digits)
        out.append({"model": row["model"], "cells": cells})
    return out


def plot_table(table, out_path):
    """Bar chart with CI error bars for the three main metrics. A rate that was not measured is marked n/a, not drawn
    as 0."""
    import matplotlib.pyplot as plt  # imported here: matplotlib is not a core dependency

    if not table:
        raise ValueError("Empty table.")
    labels = [f"{r['experiment']}\n{r['model']}\n{r['arm']}" for r in table]
    x = np.arange(len(table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(table)), 5))
    for i, metric in enumerate(METRICS):
        values = [r[metric] if r[metric] is not None else np.nan for r in table]
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


# --- C9: per-case records and paired comparisons -----------------------------------------------------------------

def case_records(run_folder, metrics_name=None):
    """One metrics record per case of a finished run (the input of metrics.summarise and metrics.paired_difference),
    rebuilt from its metrics file's per_case rows and traces/<case>.json, which hold each claim's sub-type and
    criticality. A case without traces whose reasoner failed is a format failure; a case the verifier did not finish is
    left out, as the run's own scoring left it out."""
    folder = Path(run_folder)
    rows = json.loads(_metrics_file(folder, metrics_name).read_text()).get("per_case") or []
    records = []
    for row in rows:
        error = row.get("error") or ""
        trace_file = folder / "traces" / f"{row['case']}.json"
        if error.startswith("verifier failed"):
            continue
        if not trace_file.exists():
            if not error:
                raise ValueError(f"{folder.name}: case {row['case']} has neither traces nor an error.")
            records.append({"case_id": row["case"], "format_failure": True, "traces": []})
            continue
        judged = [ok for _, _, ok in (row.get("decisions") or {}).values() if ok is not None]
        records.append({"case_id": row["case"], "format_failure": False,
                        "traces": json.loads(trace_file.read_text())["traces"],
                        "omissions": len(row.get("omissions") or []),
                        "uncertainty_omissions": len(row.get("uncertainty_omissions") or []),
                        "decisions_correct": sum(judged), "decisions_determined": len(judged)})
    return records


def paired(folders_a, folders_b, metric="hallucination_rate", metrics_name=None, keep=None):
    """a minus b on the cases both hold (metrics.paired_difference: a paired case bootstrap). Each side may pool
    several run folders (for example fixtures, scenarios and real cases of one arm); case IDs must not repeat within a
    side. `keep(case_id)`, if given, limits both sides to some cases (for example DEC-30's held-out routes). Returns the
    difference with its CI, both rates, and how many cases each side had and shared."""
    def pool(folders):
        out = {}
        for folder in ([folders] if isinstance(folders, (str, Path)) else folders):
            for record in case_records(folder, metrics_name):
                if keep is not None and not keep(record["case_id"]):
                    continue
                if record["case_id"] in out:
                    raise ValueError(f"Case {record['case_id']} appears twice on one side of a comparison.")
                out[record["case_id"]] = record
        return out
    a, b = pool(folders_a), pool(folders_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("The two sides share no case.")
    result = vm.paired_difference([a[c] for c in common], [b[c] for c in common], metric)
    side = lambda recs: vm.summarise(recs)[metric]["ci95"] if len(recs) >= vm.MIN_CASES_FOR_CI else None
    return {**result, "cases_a": len(a), "cases_b": len(b),
            "ci_a": side([a[c] for c in common]), "ci_b": side([b[c] for c in common])}  # each side, matched cases


# --- C9: the report for the paid runs ---------------------------------------------------------------------------

DISPLAY = {"claude-sonnet-5-5": "Sonnet 5.5", "qwen/qwen3.5-27b": "Qwen3.5-27B", "qwen3.5-27b": "Qwen3.5-27B",
           "gemma-3-27b": "Gemma 3 27B", "medgemma-27b-text": "MedGemma 27B", "gemma-3-4b": "Gemma 3 4B",
           "gemma-3-4b-tuned": "Gemma 3 4B tuned"}
# The rulebook comparison pairs each with-rulebook arm with its without-rulebook twin on the same cases. The
# without-rulebook scenarios are a half (DEC-29), so the pairs hold 5 + 75 + 20 cases.
RULEBOOK_PAIRS = (("fixtures_rulebook", "fixtures_norulebook"), ("scenarios_rulebook", "scenarios_norulebook"),
                  ("real_predicted", "real_norulebook"))
# The headline pool per model: every distinct case once, with the rulebook and predicted vision (5 + 150 + 20).
HEADLINE_ARMS = ("fixtures_rulebook", "scenarios_rulebook", "real_predicted")
SERIES = {"with": "#2a78d6", "without": "#eb6834"}  # categorical slots 1 and 2 (dataviz reference palette, light)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _pooled(folders, metric, keep=None):
    records = [r for f in folders for r in case_records(f) if keep is None or keep(r["case_id"])]
    s = vm.summarise(records)
    return {"value": s[metric]["value"], "ci95": s[metric]["ci95"], "numerator": s[metric]["numerator"],
            "denominator": s[metric]["denominator"], "cases": s["cases"]}


def c9_report(paid_root, out_dir, models=None):
    """The C9 output for the paid runs (plan §7 NLP B: the hallucination table, per model and scenario, with CIs).
    Reads only saved cells under `paid_root` (runs/EXP-37_paid_runs) and writes to `out_dir`:

    - table.csv: one row per cell (model x arm, Sonnet's repeats included), every rate with its 95% case-bootstrap
      interval and its counts;
    - report.json: the pooled headline per model, the paired comparisons, and the repeat spread;
    - report.md: the same as tables for the report;
    - c9_rates.png, c9_vs_sonnet.png: the two figures.

    Paired comparisons (metrics.paired_difference on the cases both sides hold):
    - rulebook: content hallucination rate (sub-types 1-5) with minus without, on matched cases. The content rate,
      because the without-rulebook arm is not charged for citations (DEC-29), so the full rate would favour it;
    - vision: hallucination rate on real cases, predicted minus ground-truth vision (DEC-11);
    - each model minus Sonnet on the headline pool, full and content rates."""
    from strokecopilot.pipeline import paid_runs as pr  # imported here: paid_runs imports the reasoner stack

    root, out = Path(paid_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = list(models or pr.MODELS)
    cell = lambda m, a: str(root / m.replace("/", "_") / a)
    folders = [cell(m, a) for m in models for a in pr.arms_for(m)]
    table = build_table(folders)
    columns = ["experiment", "model", "source", "arm", "cases", "claims"]
    for m in METRICS + EXTRA_METRICS:
        columns += [m, f"{m}_ci", f"{m}_n"]
    with (out / "table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in table:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), list) else r.get(k)) for k in columns})

    report = {"paid_root": str(root), "models": models, "headline": {}, "rulebook": {}, "vision": {}, "vs_sonnet": {},
              "sonnet_repeats": {}}
    for m in models:
        heads = [cell(m, a) for a in HEADLINE_ARMS]
        report["headline"][m] = {k: _pooled(heads, k) for k in ("hallucination_rate", "content_hallucination_rate",
                                                                 "critical_hallucination_rate", "recommendation_correctness",
                                                                 "format_failure_rate", "omission_rate")}
        report["headline"][m]["groundtruth_vision"] = _pooled([cell(m, "real_groundtruth")], "hallucination_rate")
        report["headline"][m]["without_rulebook_content"] = _pooled(
            [cell(m, b) for _, b in RULEBOOK_PAIRS], "content_hallucination_rate")
        report["rulebook"][m] = paired([cell(m, a) for a, _ in RULEBOOK_PAIRS], [cell(m, b) for _, b in RULEBOOK_PAIRS],
                                       "content_hallucination_rate")
        report["vision"][m] = paired(cell(m, "real_predicted"), cell(m, "real_groundtruth"), "hallucination_rate")
    sonnet = pr.SONNET
    for m in models:
        if m == sonnet:
            continue
        report["vs_sonnet"][m] = {k: paired([cell(m, a) for a in HEADLINE_ARMS], [cell(sonnet, a) for a in HEADLINE_ARMS], k)
                                  for k in ("hallucination_rate", "content_hallucination_rate")}
    for arm in ("real_predicted", *pr.SONNET_REPEATS):
        report["sonnet_repeats"][arm] = _pooled([cell(sonnet, arm)], "hallucination_rate")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "report.md").write_text(_report_md(report, table), encoding="utf-8")
    _plot_rates(report, out / "c9_rates.png")
    _plot_vs_sonnet(report, out / "c9_vs_sonnet.png")
    return report


def _pct(m, digits=1):
    if m is None or m.get("value") is None:
        return "n/a"
    ci = m.get("ci95")
    text = f"{100 * m['value']:.{digits}f}%"
    return f"{text} [{100 * ci[0]:.{digits}f}, {100 * ci[1]:.{digits}f}]" if ci else text


def _diff(d, digits=1):
    ci = d.get("ci95")
    text = f"{100 * d['difference']:+.{digits}f} pts"
    return f"{text} [{100 * ci[0]:+.{digits}f}, {100 * ci[1]:+.{digits}f}]" if ci else text


def _report_md(report, table):
    name = lambda m: DISPLAY.get(m, m)
    L = ["# C9: hallucination table (EXP-37)", "",
         "Verifier rates, not yet checked against the human labels (EXP-41). Intervals are 95% percentile bootstraps "
         "over cases (10,000 resamples). Rates pool claims across cases.", "",
         "## 1. Headline: with the rulebook, predicted vision (fixtures, scenarios and real cases; each case once)", "",
         "| Model | Hallucination rate | Content rate (sub-types 1–5) | Critical | Decisions right | Format failures |",
         "|---|---|---|---|---|---|"]
    for m, h in report["headline"].items():
        dec, ff = h["recommendation_correctness"], h["format_failure_rate"]
        L.append(f"| {name(m)} | {_pct(h['hallucination_rate'])} | {_pct(h['content_hallucination_rate'])} | "
                 f"{_pct(h['critical_hallucination_rate'])} | {dec['numerator']} / {dec['denominator']} | "
                 f"{ff['numerator']} / {ff['denominator']} |")
    L += ["", "## 2. Per model and arm (hallucination rate)", "",
          "| Model | Fixtures | Scenarios | Real, predicted vision | Real, ground-truth vision | Fixtures, no rulebook | Scenarios (half), no rulebook | Real, no rulebook |",
          "|---|---|---|---|---|---|---|---|"]
    order = [("fixtures", "with_rulebook"), ("scenarios", "with_rulebook"), ("real", "with_rulebook/predicted_vision"),
             ("real", "with_rulebook/groundtruth_vision"), ("fixtures", "without_rulebook"),
             ("scenarios", "without_rulebook"), ("real", "without_rulebook/predicted_vision")]
    by = {(name(r["model"]), r["source"], r["arm"]): r for r in table}
    for m in report["models"]:
        cells = []
        for src, arm in order:
            r = by.get((name(m), src, arm))
            cells.append(_pct({"value": r["hallucination_rate"], "ci95": r["hallucination_rate_ci"]}) if r else "not run")
        L.append(f"| {name(m)} | " + " | ".join(cells) + " |")
    L += ["", "Without the rulebook, citations are not charged (DEC-29), so these rates count content errors and "
          "uncited-rule content only and are not comparable with the columns to their left; section 3 compares content "
          "rates on matched cases.", "",
          "## 3. Paired comparisons (same cases on both sides)", "",
          "| Model | Rulebook effect: content rate, with minus without | Vision effect: predicted minus ground truth (real cases) | Minus Sonnet: hallucination rate | Minus Sonnet: content rate |",
          "|---|---|---|---|---|"]
    for m in report["models"]:
        vs = report["vs_sonnet"].get(m)
        L.append(f"| {name(m)} | {_diff(report['rulebook'][m])} ({report['rulebook'][m]['cases']} cases) | "
                 f"{_diff(report['vision'][m])} ({report['vision'][m]['cases']} cases) | "
                 f"{_diff(vs['hallucination_rate']) if vs else '—'} | {_diff(vs['content_hallucination_rate']) if vs else '—'} |")
    L += ["", "A negative rulebook effect means fewer content errors with the rulebook. A difference whose interval "
          "excludes 0 is the one to report as a difference.", "",
          "## 4. Run-to-run spread: Sonnet on the same 20 real cases, three runs", "",
          "| Run | Hallucination rate |", "|---|---|"]
    for arm, s in report["sonnet_repeats"].items():
        L.append(f"| {arm} | {_pct(s)} ({s['numerator']} / {s['denominator']}) |")
    return "\n".join(L) + "\n"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _plot_rates(report, path):
    """Per model: content rate with and without the rulebook on matched cases, with 95% intervals (dot and line)."""
    import matplotlib.pyplot as plt

    models = report["models"]
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.75 * len(models)), facecolor=SURFACE)
    _style(ax)
    for i, m in enumerate(models):
        d = report["rulebook"][m]
        y = len(models) - 1 - i
        for key, value, offset in (("with", d["a"], 0.12), ("without", d["b"], -0.12)):
            ax.plot([value * 100], [y + offset], "o", color=SERIES[key], markersize=8, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=f"{key} the rulebook" if i == 0 else None)
        ci_a, ci_b = _arm_ci(report, m)
        for key, ci, offset in (("with", ci_a, 0.12), ("without", ci_b, -0.12)):
            if ci:
                ax.plot([100 * ci[0], 100 * ci[1]], [y + offset] * 2, color=SERIES[key], linewidth=2, zorder=2)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in reversed(models)], color=INK, fontsize=11)
    ax.set_xlabel("Content hallucination rate (%), sub-types 1–5, matched cases, 95% intervals", color=INK_2)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_title("Content errors with and without the rulebook", color=INK, fontsize=12, loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10, labelcolor=INK,
              handletextpad=0.3, columnspacing=1.5, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _arm_ci(report, m):
    """The intervals of each side of the rulebook comparison, from the matched cases' own summaries."""
    return report["rulebook"][m].get("ci_a"), report["rulebook"][m].get("ci_b")


def _plot_vs_sonnet(report, path):
    """Each model minus Sonnet, hallucination rate on the headline pool, paired 95% interval; zero line marked."""
    import matplotlib.pyplot as plt

    models = list(report["vs_sonnet"])
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.7 * len(models)), facecolor=SURFACE)
    _style(ax)
    ax.axvline(0, color=INK_2, linewidth=1)
    for i, m in enumerate(models):
        d = report["vs_sonnet"][m]["hallucination_rate"]
        y = len(models) - 1 - i
        if d["ci95"]:
            ax.plot([100 * d["ci95"][0], 100 * d["ci95"][1]], [y, y], color=SERIES["with"], linewidth=2, zorder=2)
        ax.plot([100 * d["difference"]], [y], "o", color=SERIES["with"], markersize=8, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{100 * d['difference']:+.1f}", (100 * d["difference"], y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in reversed(models)], color=INK, fontsize=11)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel("Hallucination rate minus Sonnet 5.5, percentage points (paired 95% intervals)", color=INK_2)
    ax.set_title("Each model against Sonnet 5.5", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
