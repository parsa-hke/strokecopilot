"""Report-only (DEC-34): Vision A's core volume as a yes/no call at each decision cut-off, from saved per-case results.

For each cut-off (21, 31, 51, 70 mL) a case is "at or above" when its volume is not below the cut-off, the same rule as
`evaluate.crosses`. Sensitivity is the share of cases whose true volume is at or above the cut-off that the method also
puts there; specificity is the share truly below it that the method keeps below. Three methods are compared on the same
cases: nnU-Net (each case predicted by the fold model that did not train on it), the rCBF < 30% threshold inside
Tmax > 6 s, and a constant "always under" call, which shows what a method earns by predicting little everywhere.

Inputs are the development cases' saved volumes (EXP-27 `per_case.csv`: `ref_ml`, `nnunet_pred_ml`,
`threshold_pred_ml`). Nothing is predicted again, no held-out case is read, and no case ID is kept or written.
"""

import csv
import json
from pathlib import Path

import numpy as np

from strokecopilot.vision.haemorrhage.stats import wilson
from strokecopilot.vision.perfusion.evaluate import DECISION_CUTOFFS_ML

EXPECTED_CASES = 129
COLUMNS = {"reference": "ref_ml", "nnunet": "nnunet_pred_ml", "threshold": "threshold_pred_ml"}
METHODS = ("nnunet", "threshold", "always_under")


def load_volumes(path):
    """Volumes in mL as arrays, in file order, without the case column."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {name: np.array([float(row[column]) for row in rows], dtype=float) for name, column in COLUMNS.items()}


def call(pred_ml, ref_ml, cutoff):
    """Counts and rates for one method at one cut-off. A rate is None when its group is empty."""
    pred_above, ref_above = ~(np.asarray(pred_ml) < cutoff), ~(np.asarray(ref_ml) < cutoff)
    tp, fn = int(np.sum(pred_above & ref_above)), int(np.sum(~pred_above & ref_above))
    tn, fp = int(np.sum(~pred_above & ~ref_above)), int(np.sum(pred_above & ~ref_above))
    out = {"at_or_above": tp + fn, "below": tn + fp, "tp": tp, "fn": fn, "tn": tn, "fp": fp, "wrong_side": fn + fp}
    for name, k, n in (("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp)):
        low, high = wilson(k, n)
        out[name] = k / n if n else None
        out[f"{name}_ci95"] = [low, high]
    return out


def cutoff_table(volumes, cutoffs=DECISION_CUTOFFS_ML):
    """{cut-off: {method: counts and rates}} for nnU-Net, the threshold and the constant "always under" call."""
    ref = volumes["reference"]
    preds = {"nnunet": volumes["nnunet"], "threshold": volumes["threshold"], "always_under": np.zeros(len(ref))}
    return {f"{cutoff:g}": {method: call(preds[method], ref, cutoff) for method in METHODS} for cutoff in cutoffs}


def summary(volumes):
    """Case-level mean and median of the absolute volume error, for the like-for-like row beside held-out cases."""
    out = {"cases": int(len(volumes["reference"]))}
    for method in ("nnunet", "threshold"):
        err = np.abs(volumes[method] - volumes["reference"])
        out[method] = {"abs_error_ml_mean": float(err.mean()), "abs_error_ml_median": float(np.median(err))}
    return out


def scatter_figure(volumes, out, name="fig_visionA_volume_scatter"):
    """Predicted against true volume for nnU-Net on a square-root scale, with the four cut-offs on both axes. A point in
    a shaded band between a cut-off line on one axis and the same line on the other is on the wrong side of it."""
    from strokecopilot.pipeline.figures import GRID, INK_2, SLOTS, _plt, _save

    plt = _plt()
    ref, pred = volumes["reference"], volumes["nnunet"]
    wrong = np.zeros(len(ref), dtype=bool)
    for cutoff in DECISION_CUTOFFS_ML:
        wrong |= (pred < cutoff) != (ref < cutoff)
    top = float(max(ref.max(), pred.max())) * 1.05
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.plot([0, top], [0, top], color=GRID, linewidth=1, zorder=1)
    for cutoff in DECISION_CUTOFFS_ML:
        ax.axvline(cutoff, color=GRID, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        ax.axhline(cutoff, color=GRID, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(ref[~wrong], pred[~wrong], s=14, color=SLOTS[0], alpha=0.75, linewidths=0, zorder=3,
               label=f"right side of every cut-off ({int((~wrong).sum())})")
    ax.scatter(ref[wrong], pred[wrong], s=18, color=SLOTS[1], alpha=0.9, linewidths=0, marker="D", zorder=4,
               label=f"wrong side of at least one ({int(wrong.sum())})")
    ticks = [0, 5, *DECISION_CUTOFFS_ML, 150, 300]
    ticks = [t for t in ticks if t <= top]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("True final infarct volume (mL)")
    ax.set_ylabel("nnU-Net volume (mL)")
    ax.set_title(f"Predicted against true volume, {len(ref)} development cases", color=INK_2, loc="left", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8, handletextpad=0.3)
    written = []
    _save(fig, out, name, written)
    plt.close(fig)
    return written


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Vision A at each cut-off: sensitivity and specificity from saved volumes")
    ap.add_argument("--per-case", required=True, help="EXP-27 per_case.csv (development cases; stays on Drive)")
    ap.add_argument("--out", default="runs/EXP-84_cutoff_accuracy")
    ap.add_argument("--experiment", default="EXP-84")
    ap.add_argument("--commit", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"experiment": args.experiment, "what": __doc__.split("\n")[0], "source": "EXP-27 per_case.csv",
              "cutoffs_ml": list(DECISION_CUTOFFS_ML), "methods": list(METHODS), "git_commit": args.commit,
              "rule": "at or above a cut-off = not below it (evaluate.crosses); Wilson 95% intervals"}
    (out / "config.json").write_text(json.dumps(config, indent=1))

    volumes = load_volumes(args.per_case)
    if len(volumes["reference"]) != EXPECTED_CASES:
        raise SystemExit(f"Expected the {EXPECTED_CASES} development cases, found {len(volumes['reference'])}.")
    result = {**config, "summary": summary(volumes), "cutoffs": cutoff_table(volumes),
              "figures": scatter_figure(volumes, out)}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    for cutoff, row in result["cutoffs"].items():
        print(cutoff, {m: (row[m]["tp"], row[m]["fn"], row[m]["tn"], row[m]["fp"]) for m in METHODS})
    return result


if __name__ == "__main__":
    main()
