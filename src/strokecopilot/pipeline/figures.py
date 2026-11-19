"""The report's figures (plan §7 Phase 4: "Export figures to artifacts/figures/"), drawn from the same numbers the Space
shows (demo.charts), so the report and the page cannot disagree. Numbers only: no case ID, no RSNA image.

    python -m strokecopilot.pipeline.figures --out "<Drive>/StrokeCopilot/artifacts/figures"

Each figure is written as PNG (300 dpi) and SVG, with a README.md that says what each shows and where its numbers come
from. Style: the reference palette's categorical slots in fixed order (one per model), thin marks, hairline grid, one axis
per panel, intervals as whiskers, direct labels only where they carry the point. Held-out figures (EXP-66) are added by
`heldout_figures` once the analysis has run.
"""

import argparse
import json
from pathlib import Path

SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"claude-sonnet-5-5": "Sonnet 5.5", "qwen3.5-27b": "Qwen3.5-27B", "gemma-3-27b": "Gemma 3 27B",
          "medgemma-27b-text": "MedGemma 27B", "gemma-3-4b": "Gemma 3 4B", "gemma-3-4b-tuned": "Gemma 3 4B tuned"}
ORDER = tuple(LABELS)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "figure.dpi": 100, "savefig.bbox": "tight"})
    return plt


def _color(model):
    return SLOTS[ORDER.index(model)] if model in ORDER else INK_2


def _save(fig, out, name, written):
    for ext in ("png", "svg"):
        fig.savefig(Path(out) / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    written.append(name)


def _dots(ax, rows, key, xmax, pct=True):
    """Dot and 95% interval per row (rows top to bottom), value labelled at the whisker's end."""
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        v, (lo, hi) = r[key]["value"], r[key]["ci95"]
        ax.plot([lo, hi], [y, y], color=_color(r["model"]), lw=2, alpha=0.55, solid_capstyle="round")
        ax.plot([v], [y], "o", ms=7, color=_color(r["model"]), mec="white", mew=1.5)
        ax.text(hi + xmax * 0.015, y, f"{100 * v:.1f}%" if pct else f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r["model"]] for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.grid(axis="y", visible=False)


def development_figures(c, out):
    """The figures that need no held-out result. c: demo.charts.charts(). Returns the names written."""
    plt, written = _plt(), []
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. C9: three measures, small multiples, one axis each.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (key, title, xmax) in zip(axes, (("hallucination_rate", "All claims", 0.5),
                                             ("content_hallucination_rate", "False content only", 0.5),
                                             ("critical_hallucination_rate", "Would change a decision", 0.1))):
        _dots(ax, c["c9"]["rows"], key, xmax)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.suptitle("Hallucination rate by model (verifier, 175 cases each, 95% case-bootstrap intervals)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    _save(fig, out, "fig_c9_hallucination_by_model", written)
    plt.close(fig)

    # 2. The rulebook effect: without -> with, paired.
    rows = c["rulebook"]["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["without"], r["with"]], [y, y], color="#d3d2cc", lw=2, zorder=1)
        ax.plot([r["without"]], [y], "o", ms=7, mfc="white", mec=_color(r["model"]), mew=2, zorder=2)
        ax.plot([r["with"]], [y], "o", ms=7, color=_color(r["model"]), mec="white", mew=1.5, zorder=3)
        ax.text(max(r["without"], r["with"]) + 0.006, y, f"{100 * r['without']:.1f}% → {100 * r['with']:.1f}%", va="center",
                fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r["model"]] for r in reversed(rows)])
    ax.set_xlim(0, 0.4)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_title("False-content rate without the rulebook (hollow) and with it (filled), same 100 cases", fontsize=10,
                 color=INK, loc="left")
    _save(fig, out, "fig_rulebook_effect", written)
    plt.close(fig)

    # 3. The human check.
    hc = dict(c["human"])
    order = {"clean": 0, "content_error": 1, "citation_error": 2}
    hc["by_verdict"] = sorted(hc["by_verdict"], key=lambda v: order.get(v["verdict"], 9))
    names = {"clean": "Claims the verifier passed", "content_error": "Its false-content flags", "citation_error": "Its citation flags"}
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    for i, v in enumerate(hc["by_verdict"]):
        y = len(hc["by_verdict"]) - 1 - i
        ax.barh(y, v["agreement"], height=0.55, color=SLOTS[0])
        ax.text(v["agreement"] + 0.01, y, f"{100 * v['agreement']:.0f}% (n {v['n']})", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(hc["by_verdict"])))
    ax.set_yticklabels([names.get(v["verdict"], v["verdict"]) for v in reversed(hc["by_verdict"])])
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"People agree with the verifier (EXP-41, {hc['claims']} claims): κ between people {hc['kappa']:.2f}; "
                 f"{hc['flags_confirmed'][0]}/{hc['flags_confirmed'][1]} flags confirmed", fontsize=10, color=INK, loc="left")
    _save(fig, out, "fig_human_check", written)
    plt.close(fig)

    # 4. The tuned small model: two panels, two axes.
    rows = c["small"]["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.2))
    _dots(a1, [{**r, "rate": r["held_out_rate"]} for r in rows], "rate", 0.5)
    a1.set_title("Hallucination rate, routes never seen in training", fontsize=9, color=INK, loc="left")
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        share = r["thrombectomy"]["yes_caught"] / r["thrombectomy"]["yes"]
        a2.barh(y, share, height=0.55, color=_color(r["model"]))
        a2.text(share + 0.02, y, f"{r['thrombectomy']['yes_caught']}/{r['thrombectomy']['yes']}", va="center", fontsize=8, color=INK_2)
    a2.set_yticks(range(len(rows)))
    a2.set_yticklabels([LABELS[r["model"]] for r in reversed(rows)])
    a2.set_xlim(0, 1.15)
    a2.set_ylim(-0.6, len(rows) - 0.4)
    a2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a2.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    a2.grid(axis="y", visible=False)
    a2.set_title("Thrombectomy 'yes' cases the model also said yes to", fontsize=9, color=INK, loc="left")
    fig.subplots_adjust(wspace=0.55)
    _save(fig, out, "fig_small_reasoner_tradeoff", written)
    plt.close(fig)

    # 5. Vision B on the RSNA test split: reliability, and sensitivity by subtype.
    vb = c["vision_b"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    a1.plot([0, 1], [0, 1], color=INK_2, lw=0.8)
    a1.plot([p["mean_prob"] for p in vb["reliability"]], [p["observed"] for p in vb["reliability"]], "-o", color=SLOTS[0], ms=4, lw=1.6)
    a1.set_xlim(0, 1), a1.set_ylim(0, 1)
    a1.set_xlabel("Predicted probability"), a1.set_ylabel("Observed bleed rate")
    a1.set_title(f"Calibration (ECE {vb['scan_ece']:.3f})", fontsize=9, color=INK, loc="left")
    subs = vb["subtypes"]
    for i, s in enumerate(subs):
        y = len(subs) - 1 - i
        share = s["caught"] / s["bleeds"]
        a2.barh(y, share, height=0.55, color=SLOTS[0])
        a2.text(share + 0.005, y, f"{s['caught']}/{s['bleeds']}", va="center", fontsize=8, color=INK_2)
    a2.set_yticks(range(len(subs)))
    a2.set_yticklabels([s["subtype"] for s in reversed(subs)])
    a2.set_xlim(0.8, 1.04)
    a2.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    a2.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    fig.subplots_adjust(wspace=0.55)
    a2.grid(axis="y", visible=False)
    a2.set_title("Bleeds caught at 0.175, by subtype", fontsize=9, color=INK, loc="left")
    fig.suptitle(f"Vision B on {vb['scans']:,} unseen RSNA scans: AUC {vb['scan_auc']:.3f}, sensitivity {100 * vb['sensitivity']:.1f}%, "
                 f"specificity {100 * vb['specificity']:.1f}% (EXP-60)", x=0.01, ha="left", fontsize=10, color=INK)
    _save(fig, out, "fig_visionB_rsna_test", written)
    plt.close(fig)

    # 6. Vision A: wrong side of each cut-off (development, cross-fitted).
    va = c["vision_a"]
    cuts = list(va["wrong_side"].items())
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    for i, (cut, n) in enumerate(cuts):
        y = len(cuts) - 1 - i
        ax.barh(y, n / va["cases"], height=0.55, color=SLOTS[0])
        ax.text(n / va["cases"] + 0.005, y, f"{n} of {va['cases']}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(cuts)))
    ax.set_yticklabels([f"{c} mL" for c, _ in reversed(cuts)])
    ax.set_xlim(0, 0.35)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Predicted core on the wrong side of a decision cut-off (development; Dice {va['dice']['mean']:.2f})",
                 fontsize=10, color=INK, loc="left")
    _save(fig, out, "fig_visionA_cutoff_crossings", written)
    plt.close(fig)

    # 7. ±20% core: scenarios (where the core can decide) against real development cases.
    s = c["scenario_sensitivity"]
    rows = [("Scenarios, core ×0.8", s["x0.8"]), ("Scenarios, core ×1.2", s["x1.2"])]
    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    for i, (label, r) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, r["changed_share"], height=0.55, color=SLOTS[0])
        ax.text(r["changed_share"] + 0.001, y, f"{r['changed']} of {r['decisions']} ({r['yes_no_flips']} yes/no flips)",
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in reversed(rows)])
    ax.set_xlim(0, 0.08)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_title("Rule decisions that change when the core is off by 20% (real development cases: 0 of 258)", fontsize=10,
                 color=INK, loc="left")
    _save(fig, out, "fig_core_sensitivity_scenarios", written)
    plt.close(fig)
    return written


def heldout_figures(rep, out):
    """The held-out figures from EXP-66's report (phase4 analyse). Returns the names written."""
    plt, written = _plt(), []
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in ORDER if "c9" in rep["models"].get(m, {})]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for i, m in enumerate(models):
        y = len(models) - 1 - i
        dev, held = rep["models"][m]["c9"]["development"]["hallucination_rate"], rep["models"][m]["c9"]["heldout"]["hallucination_rate"]
        for r, dy, filled in ((dev, 0.16, False), (held, -0.16, True)):
            if r["value"] is None:
                continue
            ax.plot(r["ci95"], [y + dy, y + dy], color=_color(m), lw=2, alpha=0.5, solid_capstyle="round")
            ax.plot([r["value"]], [y + dy], "o", ms=6.5, mfc=_color(m) if filled else "white", mec=_color(m), mew=1.8)
        ax.text(max(dev["ci95"][1], held["ci95"][1]) + 0.008, y, f"{100 * held['value']:.1f}% held-out, {100 * dev['value']:.1f}% development",
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LABELS[m] for m in reversed(models)])
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlim(0, 0.75)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.grid(axis="y", visible=False)
    ax.set_title("Realism check: hallucination rate on the 20 held-out cases (filled) and 20 development real cases (hollow),\n"
                 "predicted vision, 95% case-bootstrap intervals (EXP-66, EXP-37, EXP-47)", fontsize=9.5, color=INK, loc="left")
    _save(fig, out, "fig_heldout_realism_check", written)
    plt.close(fig)
    return written


CAPTIONS = {
    "fig_heldout_realism_check": ("DEC-31 D3 (EXP-66): each model's verifier hallucination rate on the 20 held-out cases next "
                                  "to EXP-37's/EXP-47's 20 development real cases, predicted vision, unpaired. Qwen's held-out "
                                  "rate rests on 11 answers (9 format failures)."),
    "fig_c9_hallucination_by_model": ("C9 (EXP-42, EXP-48; metric points 1-20), verifier rates on the same 175 cases per "
                                      "model, 95% case-bootstrap intervals. Upper bounds: people confirmed 18 of 54 flags (EXP-41)."),
    "fig_rulebook_effect": "DEC-29 (EXP-42): false-content rate without and with the rulebook, paired on the same 100 cases.",
    "fig_human_check": "EXP-41: verifier-human agreement by the verifier's verdict, 100 claims, two labellers each plus a third rater.",
    "fig_small_reasoner_tradeoff": ("DEC-30 (EXP-47, EXP-48): Gemma 3 4B prompted and LoRA-tuned. Fewer hallucinations on "
                                    "held-out routes, but far fewer thrombectomy recommendations where the rulebook says yes."),
    "fig_visionB_rsna_test": "EXP-60: EXP-6 at DEC-28's 0.175 on the RSNA test split, opened once (patient-level split).",
    "fig_visionA_cutoff_crossings": "EXP-25/EXP-63: nnU-Net cores on the 129 development cases, each by the fold model that never trained on it.",
    "fig_core_sensitivity_scenarios": "DEC-31 D3 supplement (EXP-63): rule decisions on the 150 synthetic scenarios with the core scaled.",
}


def write_readme(out, names):
    lines = ["# Figures", "", "Generated by `python -m strokecopilot.pipeline.figures` from the runs' reports (numbers only; "
             "no case IDs, no RSNA images). Each figure is a PNG (300 dpi) and an SVG.", ""]
    lines += [f"- **{n}**: {CAPTIONS.get(n, '')}" for n in names]
    (Path(out) / "README.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    from strokecopilot.config import local_drive_root
    from strokecopilot.demo import charts

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="default: <Drive>/StrokeCopilot/artifacts/figures")
    args = parser.parse_args(argv)
    drive = local_drive_root()
    out = Path(args.out) if args.out else drive / "artifacts" / "figures"
    c = charts.charts(charts.default_sources(drive / "runs"))
    names = development_figures(c, out)
    heldout = Path("runs/EXP-66_heldout_analysis/report.json")
    if heldout.exists():
        names += heldout_figures(json.loads(heldout.read_text()), out)
    write_readme(out, names)
    print(json.dumps({"written": names, "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
