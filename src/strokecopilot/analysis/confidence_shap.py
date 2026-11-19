"""SHAP survey row 3 (DEC-35): which of v2's four signals moves `confidence.core` for each development case.

v2 (EXP-31, Lorna's chosen version before DEC-27) is a logistic regression on standardised signals, so its SHAP values
are exact and closed-form, which is what `shap.LinearExplainer` computes with independent features:

    phi_i(x) = w_i * (z_i(x) - mean over the 129 cases of z_i)

in log-odds of a large error (a core on the wrong side of a cut-off): positive means more risk, so lower confidence.
They sum, with the mean log-odds, to each case's log-odds. The mapping explained is `mapping_v2.json`, fitted on all
129 development cases, so this describes what v2 does, in-sample; the contract uses DEC-27's distance alone, whose SHAP
value is its whole log-odds shift (one input) and is not computed here.
"""

import csv
import json
from pathlib import Path

import numpy as np

RUN = "EXP-31_visionA_confidence"
LABELS = {"at_risk_no_data_share": "missing Tmax near the at-risk area", "log_pred_ml": "predicted core size",
          "log_disagreement": "disagreement with the threshold core", "log_cutoff_distance": "distance to the nearest cut-off"}


def load(drive_runs):
    folder = Path(drive_runs) / RUN
    mapping = json.loads((folder / "mapping_v2.json").read_text())
    rows = list(csv.DictReader(open(folder / "per_case.csv")))
    X = np.array([[float(r[f]) if r[f] not in ("", "nan") else np.nan for f in mapping["features"]] for r in rows])
    y = np.array([r["large_error"] in ("True", "true", "1", "1.0") for r in rows])
    return mapping, X, y


def standardised(mapping, X):
    X = np.where(np.isnan(X), np.asarray(mapping["fill"]), X)
    return (X - np.asarray(mapping["mean"])) / np.asarray(mapping["std"])


def shap_values(mapping, X):
    """(phi (n, k), base log-odds, log-odds per case)."""
    Z = standardised(mapping, X)
    w = np.asarray(mapping["weights"])
    phi = w * (Z - Z.mean(axis=0))
    base = mapping["intercept"] + float(Z.mean(axis=0) @ w)
    return phi, base, mapping["intercept"] + Z @ w


def summarise(mapping, phi, y):
    mag = np.abs(phi).mean(axis=0)
    out = {"signals": {}, "cases": int(len(phi)), "large_errors": int(y.sum())}
    for k, f in enumerate(mapping["features"]):
        out["signals"][f] = {"label": LABELS.get(f, f), "weight": mapping["weights"][k], "mean_abs_phi": float(mag[k]),
                             "share_of_attribution": float(mag[k] / mag.sum()),
                             "mean_phi_large_error": float(phi[y, k].mean()) if y.any() else None,
                             "mean_phi_other": float(phi[~y, k].mean()) if (~y).any() else None}
    return out


def main(argv=None):
    import argparse

    from strokecopilot.config import local_drive_root
    ap = argparse.ArgumentParser(description="SHAP row 3: confidence.core v2")
    ap.add_argument("--out", default="runs/EXP-70_confidence_shap")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    mapping, X, y = load(local_drive_root() / "runs")
    phi, base, logit = shap_values(mapping, X)
    if not np.allclose(phi.sum(axis=1) + base, logit):
        raise ValueError("SHAP values do not add up to the log-odds")
    summary = summarise(mapping, phi, y)
    check = None
    try:
        import shap
        Z = standardised(mapping, X)  # all 129 cases as background (shap's default would sample 100)
        explainer = shap.LinearExplainer((np.asarray(mapping["weights"]), mapping["intercept"]),
                                         shap.maskers.Independent(Z, max_samples=len(Z)))
        check = float(np.abs(explainer.shap_values(standardised(mapping, X)) - phi).max())
    except ImportError:
        pass
    summary["shap_linear_explainer_max_difference"] = check
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
