"""Phase 5 figures (DEC-34), in the style of `pipeline/figures.py`, written to Drive `artifacts/figures/` next to the
existing ones. The figures README gains or replaces one line per figure; the others are left as they are.
"""

import json
from pathlib import Path

import numpy as np

from strokecopilot.pipeline.figures import GRID, INK, INK_2, SLOTS, _plt, _save

VISION_COLOR, NOTE_COLOR = SLOTS[0], SLOTS[1]  # validated as a pair (dataviz validator, light surface)

LABELS = {"tmax6": "Tmax > 6 s volume", "core": "core volume"}

CAPTIONS = {
    "fig_rule_engine_shap": (
        "EXP-68 (DEC-35, SHAP row 8): exact Shapley values of the rule engine's 13 input groups on the 150 synthetic "
        "scenarios; an absent input is unknown. Bars: mean |Shapley value| for reaching each scenario's decision. Text: "
        "the share carried by vision inputs where the rules say yes. Describes our scenario set, not real patients."),
    "fig_ttests_forest": (
        "EXP-69 (DEC-36): the paired NLP comparisons, mean difference of per-case rates with its 95% t interval (Student's "
        "paired t-test, dot) beside the reported paired case bootstrap of the pooled rate (open circle). The case is the "
        "unit. Holm-adjusted p over the 34 tests run is in the run's report; the data had been seen before the tests "
        "were fixed."),
}

SHORT = {"claude-sonnet-5-5": "Sonnet 5.5", "qwen3.5-27b": "Qwen3.5-27B", "gemma-3-27b": "Gemma 3 27B",
         "medgemma-27b-text": "MedGemma 27B", "gemma-3-4b": "Gemma 3 4B", "gemma-3-4b-tuned": "Gemma 3 4B tuned"}
GROUP = {"T3": "With minus without the rulebook (content rate)", "T4": "Predicted minus ground-truth vision",
         "T5": "Minus Sonnet 5.5", "T6": "Tuned minus base Gemma 3 4B"}


def _forest_label(r):
    head = r["label"].split(":")[0].split(" minus ")[0]
    if r["id"].startswith("T6"):
        return {"T6a": "all 175 cases", "T6b": "held-out routes (51)", "T6c": "critical rate, all"}[r["id"]]
    return SHORT.get(head, head)


def ttest_forest_figure(rows, out):
    """T3-T6 on one axis in percentage points: t interval (dot) and the existing bootstrap interval (open circle)."""
    plt = _plt()
    rows = [r for r in rows if r["id"][:2] in GROUP]
    ys, labels, y = [], [], 0
    heads = []
    for g in GROUP:
        heads.append((y, GROUP[g]))
        y += 1
        for r in [r for r in rows if r["id"].startswith(g)]:
            ys.append((y, r))
            labels.append((y, _forest_label(r)))
            y += 1
        y += 0.4
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * y + 1.0))
    for yy, r in ys:
        lo, hi = (100 * v for v in r["ci"])
        ax.plot([lo, hi], [yy - 0.13] * 2, color=VISION_COLOR, lw=2, solid_capstyle="round")
        ax.plot([100 * r["mean_difference"]], [yy - 0.13], "o", ms=6, color=VISION_COLOR, mec="white", mew=1.2)
        ex = r.get("existing") or {}
        if ex.get("ci95"):
            blo, bhi = (100 * v for v in ex["ci95"])
            ax.plot([blo, bhi], [yy + 0.13] * 2, color=NOTE_COLOR, lw=2, solid_capstyle="round")
            ax.plot([100 * ex["pooled_difference"]], [yy + 0.13], "o", ms=6, mfc="white", mec=NOTE_COLOR, mew=1.5)
        if r["agrees_with_existing"] is False:
            ax.text(max(hi, bhi) + 0.8, yy, "intervals disagree", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks([l[0] for l in labels] + [h[0] for h in heads])
    ax.set_yticklabels([l[1] for l in labels] + [h[1] for h in heads])
    for tick, (_, _) in zip(ax.get_yticklabels()[len(labels):], heads):
        tick.set_fontweight("bold")
        tick.set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.2, -0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("difference, percentage points")
    handles = [plt.Line2D([], [], color=VISION_COLOR, marker="o", lw=2, mec="white"),
               plt.Line2D([], [], color=NOTE_COLOR, marker="o", lw=2, mfc="white", mec=NOTE_COLOR)]
    ax.legend(handles, ["paired t-test: mean of per-case rates, 95% t interval",
                        "reported: pooled rate, 95% paired case bootstrap"], loc="upper right", frameon=False, fontsize=7.5)
    written = []
    for target in out if isinstance(out, (list, tuple)) else (out,):
        Path(target).mkdir(parents=True, exist_ok=True)
        _save(fig, target, "fig_ttests_forest", written)
    plt.close(fig)
    return written[:1]


def update_readme(out, name):
    """Add or replace one figure's line in the figures README."""
    path = Path(out) / "README.md"
    lines = path.read_text().splitlines() if path.exists() else ["# Figures", ""]
    entry = f"- **{name}**: {CAPTIONS[name]}"
    lines = [entry if line.startswith(f"- **{name}**") else line for line in lines]
    if entry not in lines:
        lines.append(entry)
    path.write_text("\n".join(lines) + "\n")


def rule_shap_figure(metrics, out, players_order=None):
    """Two panels (thrombolysis, thrombectomy): mean |phi| per input group, coloured vision or note."""
    from strokecopilot.analysis import rule_shap
    plt = _plt()
    s = metrics["summary"]
    rows = json.loads((Path(metrics["_folder"]) / "per_scenario.json").read_text()) if "_folder" in metrics else None
    mag = {d: {p: float(np.mean([abs(r["players"][d][p]) for r in rows])) for p in rule_shap.PLAYERS}
           for d in rule_shap.DECISIONS}
    order = players_order or sorted(rule_shap.PLAYERS, key=lambda p: mag["thrombolysis"][p] + mag["thrombectomy"][p])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
    for ax, d in zip(axes, rule_shap.DECISIONS):
        vals = [mag[d][p] for p in order]
        colors = [VISION_COLOR if p in rule_shap.VISION else NOTE_COLOR for p in order]
        ax.barh(range(len(order)), vals, height=0.62, color=colors)
        for i in np.argsort(vals)[-3:]:
            ax.text(vals[i] + 0.012, i, f"{vals[i]:.2f}", va="center", fontsize=8, color=INK_2)
        yes = s[d].get("decision yes")
        share = yes["vision_vs_note"]["vision"] / sum(yes["vision_vs_note"].values()) if yes else float("nan")
        ax.set_title(f"{d.capitalize()}\nvision inputs carry {100 * share:.0f}% where the rules\nsay yes "
                     f"({yes['cases'] if yes else 0} scenarios)", fontsize=9, color=INK, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 0.68)
        ax.set_xlabel("mean |Shapley value|")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABELS.get(p, p) for p in order])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (VISION_COLOR, NOTE_COLOR)]
    fig.legend(handles, ["vision input (contract)", "note input"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, -0.04), fontsize=8, labelcolor=INK_2)
    fig.subplots_adjust(wspace=0.12, bottom=0.18)
    written = []
    for target in (out,) if not isinstance(out, (list, tuple)) else out:
        Path(target).mkdir(parents=True, exist_ok=True)
        _save(fig, target, "fig_rule_engine_shap", written)
    plt.close(fig)
    return written[:1]


def main(argv=None):
    import argparse
    from strokecopilot.config import local_drive_root
    ap = argparse.ArgumentParser(description="Phase 5 figures")
    ap.add_argument("--rule-shap", default="runs/EXP-68_rule_shap")
    ap.add_argument("--ttests", default="runs/EXP-69_ttests")
    ap.add_argument("--out", default=None, help="default: <Drive>/StrokeCopilot/artifacts/figures")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else local_drive_root() / "artifacts" / "figures"
    m = json.loads((Path(args.rule_shap) / "metrics.json").read_text())
    m["_folder"] = args.rule_shap
    names = rule_shap_figure(m, [out, args.rule_shap])
    tt = Path(args.ttests) / "metrics.json"
    if tt.exists():
        names += ttest_forest_figure(json.loads(tt.read_text())["rows"], [out, args.ttests])
    for n in names:
        update_readme(out, n)
    print(json.dumps({"written": names, "out": str(out)}))


if __name__ == "__main__":
    main()
