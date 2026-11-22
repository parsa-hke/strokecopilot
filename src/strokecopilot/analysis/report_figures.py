"""Report-only (DEC-34): three small read-outs of saved results for the MAAI901 report. Nothing is trained or predicted
again, no held-out case is read, and no case ID is kept or written.

- **Training curve** (`training_curve`, `curve_figure`): nnU-Net's own training logs for one fold, parsed into the
  per-epoch training loss, validation loss and validation pseudo-Dice, with nnU-Net's moving average (0.9 old, 0.1 new),
  so the curve can be drawn at a readable size.
- **Vision B's threshold trade-off** (`tradeoff`, `tradeoff_figure`): for each threshold of EXP-17's saved sweep on
  RSNA validation (cross-fitted sensitivity and specificity), the share of ISLES'24's presumed bleed-free development
  scans that EXP-16's saved calibrated scores put at or above it. The saved runs hold these operating points, not
  per-scan RSNA scores (they stay on Kaggle), so the ROC panel shows the points, not a full curve.
- **Centre gap, post hoc** (`centre_test`): Welch's t-test and a Mann-Whitney test of per-case Dice between the two
  centres on the development cases. It was run after the difference had been seen and sits outside DEC-36's 35 tests.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

from strokecopilot.vision.haemorrhage.stats import wilson

EMA_OLD = 0.9  # nnU-Net's logger: ema = 0.9 * ema + 0.1 * new
_NUMBER = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


# ------------------------------------------------------------------------------------------- training curve

def parse_training_log(text):
    """[(epoch, train_loss, val_loss, pseudo_dice)] from one nnU-Net training log; unfinished epochs are dropped."""
    rows, current = [], None
    for line in text.splitlines():
        m = re.search(r": Epoch (\d+)\s*$", line)
        if m:
            current = {"epoch": int(m.group(1))}
            continue
        if current is None:
            continue
        for key, pattern in (("train", r"train_loss " + _NUMBER), ("val", r"val_loss " + _NUMBER),
                             ("dice", r"Pseudo dice \[(?:np\.float32\()?" + _NUMBER)):
            m = re.search(pattern, line)
            if m:
                current[key] = float(m.group(1))
        if all(k in current for k in ("train", "val", "dice")):
            rows.append((current["epoch"], current["train"], current["val"], current["dice"]))
            current = None
    return rows


def training_curve(log_texts):
    """Epochs from several logs of one run (a resumed run has one log per start), later logs winning on overlap."""
    by_epoch = {}
    for text in log_texts:
        for epoch, train, val, dice in parse_training_log(text):
            by_epoch[epoch] = (train, val, dice)
    epochs = sorted(by_epoch)
    train, val, dice = (np.array([by_epoch[e][i] for e in epochs], dtype=float) for i in range(3))
    ema = np.empty(len(dice))
    for i, value in enumerate(dice):
        ema[i] = value if i == 0 else EMA_OLD * ema[i - 1] + (1 - EMA_OLD) * value
    return {"epoch": np.array(epochs), "train_loss": train, "val_loss": val, "pseudo_dice": dice, "pseudo_dice_ema": ema}


def curve_summary(curve, window=10):
    """Where the validation loss was lowest (centred moving average over `window` epochs) and where the moving-average
    pseudo-Dice peaked, with first- and last-window means, as plain numbers."""
    val, epochs = curve["val_loss"], curve["epoch"]
    kernel = np.ones(window) / window
    smooth = np.convolve(val, kernel, mode="valid")
    best = int(np.argmin(smooth))
    return {"epochs": int(len(epochs)), "window": window,
            "val_loss_lowest_window": [int(epochs[best]), int(epochs[best + window - 1])],
            "val_loss_lowest_window_mean": float(smooth[best]),
            "val_loss_last_window_mean": float(val[-window:].mean()),
            "train_loss_first_window_mean": float(curve["train_loss"][:window].mean()),
            "train_loss_last_window_mean": float(curve["train_loss"][-window:].mean()),
            "pseudo_dice_ema_best_epoch": int(epochs[int(np.argmax(curve["pseudo_dice_ema"]))]),
            "pseudo_dice_ema_best": float(curve["pseudo_dice_ema"].max()),
            "pseudo_dice_ema_last": float(curve["pseudo_dice_ema"][-1])}


def curve_figure(curve, out, name="fig_visionA_training_curve"):
    from strokecopilot.pipeline.figures import GRID, INK_2, SLOTS, _plt, _save

    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot(curve["epoch"], curve["train_loss"], color=SLOTS[0], linewidth=1.3, label="training loss")
    ax.plot(curve["epoch"], curve["val_loss"], color=SLOTS[1], linewidth=1.0, label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Dice + cross-entropy; lower is better)")
    ax.grid(True, color=GRID, linewidth=0.6)
    right = ax.twinx()
    right.plot(curve["epoch"], curve["pseudo_dice_ema"], color=SLOTS[2], linewidth=1.3,
               label="validation Dice estimate (moving average)")
    right.set_ylabel("Validation Dice estimate")
    right.spines["right"].set_visible(True)
    handles = ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], loc="center right", bbox_to_anchor=(1.0, 0.56), frameon=False,
              fontsize=8)
    ax.set_title(f"Fold 0, {len(curve['epoch'])} epochs", color=INK_2, loc="left", fontsize=9)
    written = []
    _save(fig, out, name, written)
    plt.close(fig)
    return written


# ------------------------------------------------------------------------------------------- C9 for a printed page


def c9_print_figure(rows, out, name="fig_c9_print", width_in=6.3):
    """The C9 hallucination figure redrawn for a printed page (MAAI902 report): the same three measures and numbers as
    `pipeline.figures`' wide `fig_c9_hallucination_by_model`, stacked at the page's text width so the labels print at
    8 to 10 pt instead of shrinking. rows: demo.charts' c9 rows (one per model, each rate with `value` and `ci95`)."""
    from strokecopilot.pipeline.figures import INK, _dots, _plt, _save

    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(width_in, 6.4))
    for ax, (key, title, xmax) in zip(axes, (("hallucination_rate", "All flagged claims", 0.5),
                                             ("content_hallucination_rate", "False content only", 0.5),
                                             ("critical_hallucination_rate", "Would change a decision", 0.1))):
        _dots(ax, rows, key, xmax)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.tick_params(labelsize=9)
        for text in ax.texts:
            text.set_fontsize(9)
    fig.tight_layout(h_pad=1.2)
    written = []
    _save(fig, out, name, written)
    plt.close(fig)
    return written


# ------------------------------------------------------------------------------------------- threshold trade-off

def tradeoff(sweep, isles_scores, chosen=0.175):
    """One row per saved RSNA threshold: cross-fitted sensitivity and specificity on RSNA validation, and how many
    ISLES'24 development scans score at or above it. `chosen` marks the row nearest the contract's threshold."""
    scores = np.asarray(isles_scores, dtype=float)
    rows = []
    for item in sorted(sweep, key=lambda r: -r["threshold_calibrated"]):
        t = float(item["threshold_calibrated"])
        flagged = int((scores >= t).sum())
        low, high = wilson(flagged, len(scores))
        rows.append({"target_sensitivity": item["target"], "threshold": t,
                     "rsna_sensitivity": item["cross_fitted"]["sensitivity"],
                     "rsna_specificity": item["cross_fitted"]["specificity"],
                     "isles_flagged": flagged, "isles_scans": int(len(scores)), "isles_rate": flagged / len(scores),
                     "isles_rate_ci95": [low, high]})
    nearest = min(range(len(rows)), key=lambda i: abs(rows[i]["threshold"] - chosen))
    for i, row in enumerate(rows):
        row["chosen"] = i == nearest
    return rows


def tradeoff_figure(rows, test_point, out, name="fig_visionB_threshold_tradeoff"):
    """Left: the saved operating points in ROC space, with the test split's point. Right: ISLES'24 flag rate against
    RSNA sensitivity at the same thresholds."""
    from strokecopilot.pipeline.figures import GRID, INK_2, SLOTS, _plt, _save

    plt = _plt()
    fig, (roc, cost) = plt.subplots(1, 2, figsize=(8.6, 3.9))
    fpr = [1 - r["rsna_specificity"] for r in rows]
    sens = [r["rsna_sensitivity"] for r in rows]
    roc.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    roc.plot(fpr, sens, color=SLOTS[0], linewidth=1.2, marker="o", markersize=4,
             label="RSNA validation, cross-fitted (saved thresholds)")
    star = next(r for r in rows if r["chosen"])
    roc.scatter([1 - star["rsna_specificity"]], [star["rsna_sensitivity"]], s=70, facecolors="none",
                edgecolors=SLOTS[1], linewidths=1.6, zorder=5, label=f"chosen threshold {star['threshold']:.3f}")
    roc.scatter([1 - test_point["specificity"]], [test_point["sensitivity"]], s=40, marker="D", color=SLOTS[2], zorder=6,
                label="RSNA test split, same threshold")
    roc.set_xlim(0, 0.7)
    roc.set_ylim(0.85, 1.0)
    roc.set_xlabel("False-positive rate (1 - specificity)")
    roc.set_ylabel("Sensitivity")
    roc.set_title(f"Operating points (scan AUC {test_point['auc']:.3f} on the test split)", color=INK_2, loc="left",
                  fontsize=9)
    roc.grid(True, color=GRID, linewidth=0.6)
    roc.legend(loc="lower right", frameon=False, fontsize=8)

    rate = [100 * r["isles_rate"] for r in rows]
    low = [100 * (r["isles_rate"] - r["isles_rate_ci95"][0]) for r in rows]
    high = [100 * (r["isles_rate_ci95"][1] - r["isles_rate"]) for r in rows]
    x = [100 * s for s in sens]
    cost.errorbar(x, rate, yerr=[low, high], color=SLOTS[0], linewidth=1.2, marker="o", markersize=4, capsize=2)
    cost.scatter([100 * star["rsna_sensitivity"]], [100 * star["isles_rate"]], s=70, facecolors="none",
                 edgecolors=SLOTS[1], linewidths=1.6, zorder=5)
    for r, xi, yi in zip(rows, x, rate):
        cost.annotate(f"{r['threshold']:.3f}", (xi, yi), textcoords="offset points", xytext=(-4, 6), ha="right",
                      fontsize=7, color=INK_2)
    cost.set_xlabel("RSNA sensitivity, cross-fitted (%)")
    cost.set_ylabel(f"ISLES'24 scans flagged, of {rows[0]['isles_scans']} presumed bleed-free (%)")
    cost.set_ylim(0, 100)
    cost.set_title("What each threshold costs on ISLES'24", color=INK_2, loc="left", fontsize=9)
    cost.grid(True, color=GRID, linewidth=0.6)
    fig.tight_layout()
    written = []
    _save(fig, out, name, written)
    plt.close(fig)
    return written


def load_isles_scores(path, column="slab_5mm_confidence"):
    """Calibrated scan scores from EXP-16's per_case.csv, without the case column; failed rows are skipped."""
    with open(path, newline="") as f:
        return np.array([float(row[column]) for row in csv.DictReader(f) if row.get(column) not in (None, "")])


# ------------------------------------------------------------------------------------------- centre, post hoc

def centre_test(dice_by_centre):
    """Welch's t-test (and Mann-Whitney) of Dice between two centres. `dice_by_centre` maps a centre label to its
    per-case Dice values. Returns numbers only."""
    from strokecopilot.analysis import ttests

    labels = sorted(dice_by_centre)
    if len(labels) != 2:
        raise ValueError("the centre test compares exactly two centres")
    a, b = (np.asarray(dice_by_centre[k], dtype=float) for k in labels)
    row = ttests.welch(a, b, label=f"Dice, centre {labels[0]} minus centre {labels[1]}")
    row.pop("_values", None)
    row["post_hoc"] = True
    row["note"] = ("Run after the difference had been seen; outside DEC-36's family of 35 tests, so no Holm "
                   "adjustment applies and the p-value is unadjusted. Not adjusted for infarct size.")
    return row


# ------------------------------------------------------------------------------------------- command line

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Report-only read-outs of saved results (MAAI901 report)")
    ap.add_argument("what", choices=("curve", "tradeoff", "centre"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--logs", nargs="*", help="curve: nnU-Net training logs of one fold, in order")
    ap.add_argument("--sweep", help="tradeoff: EXP-17 metrics.json")
    ap.add_argument("--isles", help="tradeoff: EXP-16 per_case.csv (stays on Drive)")
    ap.add_argument("--test", help="tradeoff: EXP-60 metrics.json")
    ap.add_argument("--per-case", help="centre: EXP-27 per_case.csv (stays on Drive)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"experiment": args.experiment, "what": args.what, "git_commit": args.commit}
    (out / "config.json").write_text(json.dumps(config, indent=1))

    if args.what == "curve":
        curve = training_curve([Path(p).read_text() for p in args.logs])
        result = {**config, "summary": curve_summary(curve), "figures": curve_figure(curve, out)}
    elif args.what == "tradeoff":
        sweep = json.loads(Path(args.sweep).read_text())["sweep"]
        test = json.loads(Path(args.test).read_text())["test"]["preregistered"]
        point = {"sensitivity": test["sensitivity"], "specificity": test["specificity"], "auc": test["scan_auc"]}
        rows = tradeoff(sweep, load_isles_scores(args.isles))
        result = {**config, "rows": rows, "test_point": point, "figures": tradeoff_figure(rows, point, out)}
    else:
        from strokecopilot.data import isles24, splits

        folds = splits.load_folds()
        dev = list(splits.development_ids(folds))
        root = Path(__file__).resolve().parents[3] / "data" / "local" / isles24.SLIM_NAME
        with open(args.per_case, newline="") as f:
            dice = {row["case"]: float(row["nnunet_dice"]) for row in csv.DictReader(f)}
        groups = {}
        for case in dev:
            centre = isles24.load_case(case, root, names=(), folds=folds).clinical.get("Center")
            groups.setdefault(str(int(float(centre))), []).append(dice[case])
        result = {**config, "cases": len(dev), "test": centre_test(groups)}
    text = json.dumps(result, indent=1, default=float)
    if re.search(r"sub-stroke\d", text):
        raise ValueError("A case ID reached metrics.json. Nothing was saved.")
    (out / "metrics.json").write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k not in ("figures",)}, indent=1, default=float)[:2500])
    return result


if __name__ == "__main__":
    main()
