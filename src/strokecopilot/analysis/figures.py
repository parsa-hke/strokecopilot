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
}


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
    ap.add_argument("--out", default=None, help="default: <Drive>/StrokeCopilot/artifacts/figures")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else local_drive_root() / "artifacts" / "figures"
    m = json.loads((Path(args.rule_shap) / "metrics.json").read_text())
    m["_folder"] = args.rule_shap
    names = rule_shap_figure(m, [out, args.rule_shap])
    for n in names:
        update_readme(out, n)
    print(json.dumps({"written": names, "out": str(out)}))


if __name__ == "__main__":
    main()
