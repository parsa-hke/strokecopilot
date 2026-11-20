"""Phase 5 figures (DEC-34), in the style of `pipeline/figures.py`, written to Drive `artifacts/figures/` next to the
existing ones. The figures README gains or replaces one line per figure; the others are left as they are.
"""

import json
from pathlib import Path

import numpy as np

from strokecopilot.pipeline.figures import GRID, INK, INK_2, SLOTS, _plt, _save  # noqa: F401

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
    "fig_confidence_v2_shap": (
        "EXP-70 (DEC-35, SHAP row 3): share of mean |SHAP value| (exact, LinearExplainer) of confidence.core v2's four "
        "signals over the 129 development cases, in log-odds of a large error; in-sample (v2 was fitted on them). The "
        "contract uses the distance alone (DEC-27)."),
    "fig_retrieval_token_shap": (
        "EXP-71 (DEC-35, SHAP row 10): for each of the 12 key fixture passages, the share of its bge similarity to a "
        "gold claim that falls on number tokens (filled) against the share of its tokens that are numbers (open); "
        "Partition SHAP over passage tokens. The rank is the passage's place among all 47 for that claim."),
    "fig_surrogate_flags_shap": (
        "EXP-73 (DEC-35, SHAP row 11): a SURROGATE gradient-boosted model of which claims the verifier flags, from claim "
        "features known before verification (27,472 claims, six models, rulebook arms); out-of-fold AUC 0.92 [0.92, 0.93], "
        "grouped by case. Mean SHAP value (log-odds of being flagged) when a claim has each kind or comes from each model, "
        "with the flag rate. It explains the surrogate, not the verifier."),
    "fig_visionB_pixel_shap": (
        "EXP-72 (DEC-35, SHAP row 6): four of the 68 ISLES'24 development scans Vision B flags at DEC-28 (top slab, no "
        "case ID): brain window, Grad-CAM (Item 2) and |SHAP| by expected gradients, background 50 unflagged top slabs. "
        "Over all 136 top-2 slabs, median share in brain tissue: SHAP 50%, Grad-CAM 76%; bone 26% and 9%. ISLES'24, "
        "CC BY-NC-SA 4.0."),
    "fig_subgroups_vision": (
        "EXP-80 (DEC-37, task E): Vision A's Dice and wrong-side rate and Vision B's bleed-flag rate on the 129 development "
        "cases, by age band, sex, centre and onset-information pattern; case-bootstrap (Dice) and Wilson intervals. "
        "Unadjusted: infarct size differs between groups. Groups under 10 cases are left out."),
    "fig_visionA_channel_shap": (
        "EXP-75 (DEC-35, SHAP row 1): exact Shapley values of nnU-Net's five inputs (NCCT, CBF, CBV, MTT, Tmax with its "
        "mask) on the 129 development cases, each predicted by its validating fold model for all 32 combinations; mean "
        "|value| for the predicted core volume and for Dice. Reliance, not ablation: no model was trained without an input."),
    "fig_reasoner_shap": (
        "EXP-74 (DEC-35, SHAP row 9, amended): exact Shapley values of six input groups on 24 synthetic scenarios (the "
        "bleed fixed), for the rule engine and for Gemma 3 4B prompted and LoRA-tuned on the same 64 versions of each "
        "scenario; mean |value| for reaching the full-input decision."),
    "fig_visionB_window_shap": (
        "EXP-76 and EXP-77 (DEC-35, SHAP row 5): exact Shapley values of the three CT windows on the calibrated scan "
        "probability; mean value by group on RSNA validation and on ISLES'24's 129 development scans. An absent window is "
        "set to its RSNA training mean. Reliance, not ablation."),
    "fig_judge_agreement": (
        "EXP-81 (DEC-38, task F): agreement with the human consensus on EXP-41's 100 claims, per reasoner model, for the "
        "Claude Haiku 5.5 judge and for GPT-6 Luna (no family shared with any reasoner). Metric points 19 and 20 were "
        "adjusted on these labels."),
    "fig_biomedclip": (
        "EXP-82 and EXP-83 (DEC-33, task D): on RSNA validation, scan AUC and specificity at the threshold reaching 96% "
        "sensitivity (in-sample for every model alike) for EXP-6 and for BiomedCLIP as a linear probe and zero-shot, "
        "each scan scored by EXP-6's top-2 aggregation; ISLES'24 flags are of the 129 presumed bleed-free development "
        "scans. Right: per-subtype probes on EXP-6's frozen features beside EXP-6's own 'any' score. Report only; the "
        "RSNA test split was not read."),
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
    for run, draw, name in (("runs/EXP-70_confidence_shap", confidence_shap_figure, "metrics.json"),
                            ("runs/EXP-71_retrieval_shap", retrieval_shap_figure, "per_pair.json"),
                            ("runs/EXP-73_surrogate_flags", surrogate_flags_figure, "metrics.json")):
        if (Path(run) / name).exists():
            names += draw(json.loads((Path(run) / name).read_text()), [out, run])
    sub = Path("runs/EXP-80_subgroups/metrics.json")
    if sub.exists():
        names += subgroups_figure(json.loads(sub.read_text()), [out, sub.parent])
    for run, draw in (("runs/EXP-75_visionA_channel_shap", visiona_channel_figure), ("runs/EXP-74_reasoner_shap", reasoner_shap_figure),
                      ("runs/EXP-81_rejudge", judge_figure)):
        if (Path(run) / "metrics.json").exists():
            names += draw(json.loads((Path(run) / "metrics.json").read_text()), [out, run])
    rsna, isles = Path("runs/EXP-76_visionB_window_shap/metrics.json"), Path("runs/EXP-77_visionB_window_shap_isles/metrics.json")
    if rsna.exists() or isles.exists():
        names += visionb_window_figure(json.loads(rsna.read_text()) if rsna.exists() else None,
                                       json.loads(isles.read_text()) if isles.exists() else None, [out, isles.parent])
    clip, clip_isles = Path("runs/EXP-82_biomedclip/metrics.json"), Path("runs/EXP-83_biomedclip_isles/metrics.json")
    if clip.exists():
        names += biomedclip_figure(json.loads(clip.read_text()),
                                   json.loads(clip_isles.read_text()) if clip_isles.exists() else None, [out, clip.parent])
    pixel = Path("runs/EXP-72_visionB_pixel_shap/pixel_shap_examples.png")
    if pixel.exists():
        import shutil
        shutil.copyfile(pixel, out / "fig_visionB_pixel_shap.png")
        names.append("fig_visionB_pixel_shap")
    tt = Path(args.ttests) / "metrics.json"
    if tt.exists():
        names += ttest_forest_figure(json.loads(tt.read_text())["rows"], [out, args.ttests])
    for n in names:
        update_readme(out, n)
    print(json.dumps({"written": names, "out": str(out)}))



def confidence_shap_figure(metrics, out):
    plt = _plt()
    sig = metrics["signals"]
    order = sorted(sig, key=lambda k: sig[k]["share_of_attribution"])
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    vals = [100 * sig[k]["share_of_attribution"] for k in order]
    ax.barh(range(len(order)), vals, height=0.6, color=VISION_COLOR)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([sig[k]["label"] for k in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of mean |SHAP value| (log-odds of a large error)")
    ax.grid(axis="y", visible=False)
    ax.set_title("confidence.core v2: what moves it (129 development cases)", fontsize=9, color=INK, loc="left")
    return _write(fig, out, "fig_confidence_v2_shap")


def retrieval_shap_figure(rows, out):
    plt = _plt()
    rows = sorted(rows, key=lambda r: r["digit_attribution_share"])
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(rows) + 1.1))
    for i, r in enumerate(rows):
        a, t = 100 * r["digit_attribution_share"], 100 * r["digit_token_share"]
        ax.plot([t, a], [i, i], color=GRID, lw=2, zorder=1)
        ax.plot([t], [i], "o", ms=7, mfc="white", mec=NOTE_COLOR, mew=1.5, zorder=2)
        ax.plot([a], [i], "o", ms=7, color=VISION_COLOR, mec="white", mew=1.2, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['passage']}  (rank {r['rank']})" for r in rows], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("percent")
    handles = [plt.Line2D([], [], ls="", marker="o", color=VISION_COLOR, mec="white"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=NOTE_COLOR, mew=1.5)]
    ax.legend(handles, ["share of the similarity's |SHAP| on number tokens", "share of tokens that are numbers"],
              loc="lower right", frameon=False, fontsize=7.5)
    return _write(fig, out, "fig_retrieval_token_shap")


def _write(fig, out, name):
    written = []
    for target in out if isinstance(out, (list, tuple)) else (out,):
        Path(target).mkdir(parents=True, exist_ok=True)
        _save(fig, target, name, written)
    _plt().close(fig)
    return written[:1]


KIND_WORDS = {"rule": "rule claim", "comparison": "comparison or reason", "patient": "patient fact", "number": "number",
              "citation": "decision citation (rule_ids)", "absence": "something not recorded", "vision": "imaging value"}


def surrogate_flags_figure(metrics, out):
    plt = _plt()
    lv = metrics["explained"]["levels"]
    panels = [("kind", "Claim kind"), ("model", "Model")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1, 1]})
    for ax, (feat, title) in zip(axes, panels):
        items = sorted([(k.split("=", 1)[1], v) for k, v in lv.items() if k.startswith(feat + "=")],
                       key=lambda kv: kv[1]["mean_shap_when_present"])
        names = [KIND_WORDS.get(n, n) if feat == "kind" else SHORT.get(n, n) for n, _ in items]
        vals = [v["mean_shap_when_present"] for _, v in items]
        ax.barh(range(len(items)), vals, height=0.6, color=[NOTE_COLOR if x > 0 else VISION_COLOR for x in vals])
        for i, (_, v) in enumerate(items):
            x = v["mean_shap_when_present"]
            ax.text(x + (0.06 if x >= 0 else -0.06), i, f"{100 * v['flag_rate']:.0f}% flagged", va="center",
                    ha="left" if x >= 0 else "right", fontsize=7, color=INK_2)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(names)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-2.2, 2.6)
        ax.set_xlabel("mean SHAP value, log-odds of a flag")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.suptitle("Surrogate model of the verifier's flags (not the verifier itself)", fontsize=9, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    return _write(fig, out, "fig_surrogate_flags_shap")


def subgroups_figure(metrics, out):
    plt = _plt()
    v = metrics["vision"]
    rows = [(g, lab, d) for g in ("age band", "sex", "centre", "onset") for lab, d in v.get(g, {}).items() if not d["too_small"]]
    panels = [("Vision A Dice", lambda d: (d["dice_mean"], d["dice_ci"])),
              ("Vision A: core on the wrong side of a cut-off", lambda d: (d["wrong_side"] / d["cases"], d["wrong_side_ci"])),
              ("Vision B: bleed flag (all presumed false)", lambda d: (d["bleed_flags"] / d["cases"], d["bleed_flags_ci"]))]
    fig, axes = plt.subplots(1, 3, figsize=(10, 0.32 * len(rows) + 1.3), sharey=True)
    for ax, (title, get) in zip(axes, panels):
        for i, (g, lab, d) in enumerate(rows):
            val, ci = get(d)
            y = len(rows) - 1 - i
            ax.plot(ci, [y, y], color=VISION_COLOR, lw=2, alpha=0.55, solid_capstyle="round")
            ax.plot([val], [y], "o", ms=6, color=VISION_COLOR, mec="white", mew=1.2)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{lab} (n={d['cases']})" for g, lab, d in reversed(rows)], fontsize=8)
    fig.tight_layout()
    return _write(fig, out, "fig_subgroups_vision")


THREE = (SLOTS[0], SLOTS[1], SLOTS[2])  # validated as a set (dataviz validator, light; labels carry the contrast)


def _hbars(ax, names, series, labels, xlabel, title, fmt="{:.2f}"):
    """Grouped horizontal bars: names top to bottom, one bar per series, a value label at each bar's end."""
    n = len(series)
    h = 0.8 / n
    for k, (vals, lab) in enumerate(zip(series, labels)):
        ys = [len(names) - 1 - i + (n - 1 - 2 * k) * h / 2 for i in range(len(names))]
        ax.barh(ys, vals, height=h * 0.92, color=THREE[k % 3], label=lab)
        for y, v in zip(ys, vals):
            if abs(v) < 0.005:
                continue  # no label on an empty bar
            ax.text(v + (0.01 * max(max(s) for s in series) if v >= 0 else -0.01), y, fmt.format(v), va="center",
                    ha="left" if v >= 0 else "right", fontsize=6.5, color=INK_2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def visiona_channel_figure(metrics, out):
    plt = _plt()
    players = metrics["players"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.8), sharey=True)
    for ax, key, xlabel, fmt in ((axes[0], "pred_ml", "mean |Shapley value|, mL", "{:.1f}"),
                                 (axes[1], "dice", "mean |Shapley value|, Dice", "{:.3f}")):
        vals = [metrics[key]["mean_abs"][p] for p in players]
        unit, f = (" mL", "{:.1f}") if key == "pred_ml" else ("", "{:.3f}")
        _hbars(ax, players, [vals], [None], xlabel,
               f"{'Predicted core volume' if key == 'pred_ml' else 'Dice'}\nmean over cases: all inputs "
               f"{f.format(metrics[key]['full_mean'])}{unit}, none {f.format(metrics[key]['none_mean'])}{unit}", fmt)
        ax.set_xlim(0, max(vals) * 1.25 or 1)
    fig.suptitle(f"Vision A: which inputs nnU-Net relies on (development cases: {metrics['complete']})", fontsize=9,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _write(fig, out, "fig_visionA_channel_shap")


def reasoner_shap_figure(metrics, out):
    plt = _plt()
    s = metrics["summary"]
    groups = metrics["groups"]
    base, tuned = "gemma-3-4b", "gemma-3-4b-tuned"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, d in zip(axes, ("thrombolysis", "thrombectomy")):
        series = [[s[base][d]["rules_mean_abs"][g] for g in groups], [s[base][d]["model_mean_abs"][g] for g in groups],
                  [s[tuned][d]["model_mean_abs"][g] for g in groups]]
        _hbars(ax, groups, series, ["rule engine", "Gemma 3 4B", "Gemma 3 4B tuned"], "mean |Shapley value|",
               d.capitalize())
        ax.set_xlim(0, 1.1)
    axes[1].legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.suptitle(f"What the rules and the small reasoners rely on ({s[base]['scenarios_complete']} and "
                 f"{s[tuned]['scenarios_complete']} complete scenarios)", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _write(fig, out, "fig_reasoner_shap")


def visionb_window_figure(rsna, isles, out):
    plt = _plt()
    windows = ("brain", "subdural", "bone")
    panels = []
    if rsna:
        g = rsna["groups"]
        panels.append(("RSNA validation", [k for k in ("bleed", "no bleed", "false positive") if k in g], g))
    if isles:
        g = isles["groups"]
        panels.append(("ISLES'24 development (all presumed bleed-free)", list(g), g))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 2.9), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (title, names, g) in zip(axes, panels):
        series = [[g[n]["mean_phi"][w] for n in names] for w in windows]
        labels = [f"{n} ({g[n]['scans']})" for n in names]
        _hbars(ax, labels, series, list(windows), "mean Shapley value, calibrated probability", title)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_xlim(-0.25, 1.0)
    axes[-1].legend(loc="lower right", frameon=False, fontsize=7.5, title="window", title_fontsize=7.5)
    fig.tight_layout()
    return _write(fig, out, "fig_visionB_window_shap")


def judge_figure(metrics, out):
    plt = _plt()
    names = {"claude-sonnet-5-5": "Sonnet 5.5", "qwen/qwen3.5-27b": "Qwen3.5-27B", "gemma-3-27b": "Gemma 3 27B",
             "medgemma-27b-text": "MedGemma 27B"}
    h, n = metrics["haiku_vs_humans"], metrics["new_judge_vs_humans"]
    models = sorted(h["per_model"])
    rows = [("all claims", h["all"], n["all"])] + [(names.get(m, m), h["per_model"][m], n["per_model"][m]) for m in models]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.0))
    for i, (lab, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([a["agreement"], b["agreement"]], [y + 0.12, y - 0.12], color=GRID, lw=2, zorder=1)
        ax.plot([a["agreement"]], [y + 0.12], "o", ms=7, color=THREE[0], mec="white", mew=1.2, zorder=2,
                label="Claude Haiku 5.5" if i == 0 else None)
        ax.plot([b["agreement"]], [y - 0.12], "o", ms=7, color=THREE[1], mec="white", mew=1.2, zorder=3,
                label="GPT-6 Luna" if i == 0 else None)
        ax.text(1.01, y, f"n={a['n']}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("agreement with the human consensus (flag = error)")
    jj = metrics["judge_vs_judge"]
    ax.set_title(f"Two judges on the same {metrics['claims']} claims: judge-judge agreement {jj['agreement']:.2f}, κ {jj['kappa']:.2f}",
                 fontsize=9, color=INK, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    fig.tight_layout()
    return _write(fig, out, "fig_judge_agreement")


MODEL_WORDS = {"exp6": "EXP-6 ResNet-50", "probe": "BiomedCLIP, linear probe", "zero_shot": "BiomedCLIP, zero-shot"}


def _dots(ax, names, series, labels, xlabel, title, xlim):
    """Dot rows (a truncated axis needs dots, not bars): names top to bottom, one offset dot per series."""
    n = len(series)
    for k, (vals, lab) in enumerate(zip(series, labels)):
        off = (n - 1 - 2 * k) * 0.12
        ys = [len(names) - 1 - i + off for i in range(len(names))]
        keep = [(y, v) for y, v in zip(ys, vals) if v is not None]
        ax.plot([v for _, v in keep], [y for y, _ in keep], "o", ms=6.5, color=THREE[k % 3], mec="white", mew=1.1,
                label=lab, zorder=2 + k, ls="none")
        for y, v in keep:
            ax.text(v + 0.02 * (xlim[1] - xlim[0]), y, f"{v:.3f}", va="center", ha="left", fontsize=6.5, color=INK_2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def biomedclip_figure(metrics, isles, out):
    """Task D (EXP-82, and EXP-83 when present): scan AUC and specificity at 96% sensitivity for the three models on RSNA
    validation; per-subtype scan AUC of the subtype probes beside EXP-6's 'any' score."""
    plt = _plt()
    models = [m for m in ("exp6", "probe", "zero_shot") if m in metrics["models"]]
    subtypes = sorted(metrics.get("subtypes", {}))
    fig, axes = plt.subplots(1, 2 if subtypes else 1, figsize=(9.6 if subtypes else 5.2, 3.3))
    axes = np.atleast_1d(axes)
    names = [MODEL_WORDS[m] for m in models]
    if isles:
        names = [f"{w}\nISLES'24 flags {isles['flagged'][m]}/{isles['scans']}" if m in isles["flagged"] else w
                 for w, m in zip(names, models)]
    vals = [[metrics["models"][m][k] for m in models] for k in ("scan_auc", "specificity")]
    lo = min(v for s in vals for v in s if v is not None)
    _dots(axes[0], names, vals, ["scan AUC", "specificity at 96% sensitivity"], "",
          f"Haemorrhage, RSNA validation ({metrics['val_scans']:,} scans)", (min(0.5, np.floor(lo * 20) / 20), 1.04))
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=7)
    if subtypes:
        s = metrics["subtypes"]
        sv = [[s[t]["probe_scan_auc"] for t in subtypes], [s[t]["exp6_any_scan_auc"] for t in subtypes]]
        lo = min(v for row in sv for v in row if v is not None)
        _dots(axes[1], [f"{t} ({s[t]['scans_positive']})" for t in subtypes], sv,
              ["subtype probe on EXP-6's features", "EXP-6's 'any' score"], "scan AUC for that subtype",
              "Per subtype (positive scans)", (min(0.5, np.floor(lo * 20) / 20), 1.04))
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=7)
    fig.tight_layout()
    return _write(fig, out, "fig_biomedclip")


if __name__ == "__main__":
    main()
