"""Slice -> scan scores, calibration and the Vision B metrics (plan §8.2). NumPy and pandas only.

A scan's score is the max or the mean of the top k slice probabilities, picked by validation scan
AUC. Training sees balanced batches, so raw probabilities are too high at the natural bleed rate.
Calibration therefore fits a temperature and, by default, a bias on the logit of the scan score
(Platt scaling); temperature alone cannot undo a change in prevalence. Both are fitted, and the one
with the lower validation log loss is kept. The calibrated probability is `haemorrhage.confidence`
(DEC-15). Calibration is judged on the half of validation it was not fitted on, swapping halves, so
the reported error is not in-sample.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TOP_K = (2, 3, 5)
ECE_BINS = 10
TARGET_SENSITIVITY = 0.95
EPS = 1e-6


def roc_auc(labels, scores):
    """Area under the ROC curve by ranks, ties shared. None when only one class is present."""
    labels, scores = np.asarray(labels).astype(bool), np.asarray(scores, dtype=float)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def scan_scores(slices, method="max", k=None, prob="prob"):
    """One row per scan: series_id, label (any slice positive) and score.

    slices needs series_id, any and a probability column. method is "max" or "topk" with k.
    """
    grouped = slices.groupby("series_id")
    if method == "max":
        score = grouped[prob].max()
    elif method == "topk":
        score = grouped[prob].apply(lambda p: float(np.sort(p.to_numpy())[-k:].mean()))
    else:
        raise ValueError(f"Unknown aggregation {method!r}.")
    return pd.DataFrame({"label": grouped["any"].max().astype(int), "score": score}).reset_index()


def candidates(top_k=TOP_K):
    return [("max", None)] + [("topk", k) for k in top_k]


def choose_aggregation(val_slices, top_k=TOP_K):
    """Pick the aggregation with the highest validation scan AUC. Returns (method, k, table)."""
    table = []
    for method, k in candidates(top_k):
        scans = scan_scores(val_slices, method, k)
        table.append({"method": method, "k": k, "scan_auc": roc_auc(scans["label"], scans["score"])})
    best = max(table, key=lambda row: (row["scan_auc"] or 0, row["method"] == "max"))
    return best["method"], best["k"], table


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


@dataclass
class Calibrator:
    """p = sigmoid(logit(score) / temperature + bias), then present = p >= threshold."""

    method: str = "max"
    k: int | None = None
    temperature: float = 1.0
    bias: float = 0.0
    threshold: float = 0.5
    kind: str = "identity"

    def apply(self, scores):
        return _sigmoid(_logit(scores) / self.temperature + self.bias)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def fit_logistic(scores, labels, bias=True, iterations=100):
    """Fit sigmoid(a * logit(score) + b) by Newton steps on log loss. Returns (temperature, bias)."""
    z, y = _logit(scores), np.asarray(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iterations):
        p = _sigmoid(a * z + b)
        w = np.maximum(p * (1 - p), EPS)
        grad = np.array([np.sum((p - y) * z), np.sum(p - y)])
        hess = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        if not bias:
            step = np.array([grad[0] / max(hess[0, 0], EPS), 0.0])
        else:
            step = np.linalg.solve(hess + EPS * np.eye(2), grad)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    a = min(max(a, 1e-3), 1e3)  # a non-positive slope would reverse the ranking; separable data would run away
    return float(1 / a), float(b)


def log_loss(probs, labels):
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(probs, labels, bins=ECE_BINS):
    """Expected calibration error with equal-width bins."""
    return float(sum(row["count"] * abs(row["mean_prob"] - row["observed"]) for row in reliability(probs, labels, bins)) / max(len(probs), 1))


def reliability(probs, labels, bins=ECE_BINS):
    """Per bin: count, mean predicted probability and observed bleed rate. Empty bins are left out."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            rows.append({"bin": b, "low": float(edges[b]), "high": float(edges[b + 1]), "count": int(mask.sum()),
                         "mean_prob": float(probs[mask].mean()), "observed": float(labels[mask].mean())})
    return rows


def threshold_for_sensitivity(probs, labels, target=TARGET_SENSITIVITY):
    """The highest threshold whose sensitivity is at least target, with its sensitivity and specificity."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels).astype(bool)
    positives = np.sort(probs[labels])[::-1]
    if len(positives) == 0:
        raise ValueError("No positive scans to set a threshold on.")
    needed = int(np.ceil(target * len(positives)))
    threshold = float(positives[needed - 1])
    return {"threshold": threshold, **operating_point(probs, labels, threshold)}


def operating_point(probs, labels, threshold):
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels).astype(bool)
    called = probs >= threshold
    return {
        "sensitivity": float(called[labels].mean()) if labels.any() else None,
        "specificity": float((~called[~labels]).mean()) if (~labels).any() else None,
        "called_positive_share": float(called.mean()),
    }


def _half(series_id):
    return int(hashlib.sha256(series_id.encode()).hexdigest(), 16) % 2


def _calibrate(scans, bias):
    if bias is None:
        return 1.0, 0.0
    return fit_logistic(scans["score"], scans["label"], bias=bias)


def calibrate(val_slices, method, k, target_sensitivity=TARGET_SENSITIVITY):
    """Fit the calibrator on validation scans and report out-of-half calibration. Returns (Calibrator, report)."""
    scans = scan_scores(val_slices, method, k)
    halves = scans["series_id"].map(_half)
    report = {"method": method, "k": k, "scans": int(len(scans)), "bleed_rate": round(float(scans["label"].mean()), 4), "kinds": {}}
    for kind, bias in (("identity", None), ("temperature", False), ("platt", True)):
        held_probs = np.empty(len(scans))
        for h in (0, 1):
            fit_on, score_on = scans[halves != h], scans[halves == h]
            temperature, b = _calibrate(fit_on, bias)
            held_probs[(halves == h).to_numpy()] = Calibrator(temperature=temperature, bias=b).apply(score_on["score"])
        report["kinds"][kind] = {"log_loss": round(log_loss(held_probs, scans["label"]), 4), "ece": round(ece(held_probs, scans["label"]), 4)}
    kind = min(("temperature", "platt"), key=lambda name: report["kinds"][name]["log_loss"])
    temperature, b = _calibrate(scans, kind == "platt")
    calibrator = Calibrator(method=method, k=k, temperature=temperature, bias=b, kind=kind)
    probs = calibrator.apply(scans["score"])
    point = threshold_for_sensitivity(probs, scans["label"], target_sensitivity)
    calibrator.threshold = point["threshold"]
    report.update(chosen=kind, temperature=round(temperature, 4), bias=round(b, 4), target_sensitivity=target_sensitivity, operating_point=point,
                  scan_auc=roc_auc(scans["label"], probs), reliability=reliability(probs, scans["label"]))
    return calibrator, report


def evaluate(slices, calibrator):
    """Slice AUC, scan AUC, log loss, ECE and the operating point for one split. Numbers only."""
    scans = scan_scores(slices, calibrator.method, calibrator.k)
    probs = calibrator.apply(scans["score"])
    return {
        "slices": int(len(slices)), "scans": int(len(scans)),
        "slice_bleed_rate": round(float(slices["any"].mean()), 4), "scan_bleed_rate": round(float(scans["label"].mean()), 4),
        "slice_auc": roc_auc(slices["any"], slices["prob"]), "scan_auc": roc_auc(scans["label"], probs),
        "scan_log_loss": round(log_loss(probs, scans["label"]), 4), "scan_ece": round(ece(probs, scans["label"]), 4),
        "threshold": calibrator.threshold, **operating_point(probs, scans["label"], calibrator.threshold),
    }


def reliability_plot(rows, out_png, title="Scan calibration, validation"):
    """Reliability diagram from `reliability` rows. Holds bin numbers only, no images or IDs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax.plot([r["mean_prob"] for r in rows], [r["observed"] for r in rows], marker="o")
    for r in rows:
        ax.annotate(str(r["count"]), (r["mean_prob"], r["observed"]), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed bleed rate", title=title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
