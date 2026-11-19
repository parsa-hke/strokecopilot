"""SHAP survey row 11 (DEC-35): a SURROGATE model of which claims the verifier flags, explained with SHAP.

This explains a small gradient-boosted model fitted to the verifier's decisions, not the verifier itself. It asks which
claim features, all known before verification, go with being flagged.

- **Claims:** every claim of the rulebook arms with predicted vision (fixtures, 150 scenarios, 20 development real
  cases) for the six models, from the saved traces (rescored, metric points 1-20).
- **Features, before verification only:** model; case type; the field the claim sits in; the claim kind the verifier
  routed it to; the number of declared sources and whether they include a passage, an imaging field or a note quote; the
  `derived` flag; the claim's length in words and how many numbers it holds. Nothing from the verdict (subtype,
  supported passages, reason) is used.
- **Target:** flagged (any sub-type 1-7).
- **Validity first:** 5-fold cross-validation grouped by case (a case's claims never straddle folds), out-of-fold AUC
  with a 95% case-bootstrap interval. If the interval includes 0.5 the SHAP result is not reported, as the survey says.
- **Explained:** the model refitted on all claims, `shap.TreeExplainer`; one-hot columns summed back to their feature.
"""

import json
import re
from pathlib import Path

import numpy as np

from strokecopilot.pipeline import compare as cp

MODELS = ("claude-sonnet-5-5", "qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text", "gemma-3-4b", "gemma-3-4b-tuned")
ARMS = {"fixtures_rulebook": "fixture", "scenarios_rulebook": "scenario", "real_predicted": "real case"}
NUMBER = re.compile(r"\d+(?:\.\d+)?")
CATEGORICAL = ("model", "case type", "field", "kind")
NUMERIC = ("sources", "cites a passage", "cites imaging", "cites the note", "derived", "words", "numbers")
SEED = 20261120


def cell(model, arm):
    root = Path("runs/EXP-47_small_reasoner" if model.startswith("gemma-3-4b") else "runs/EXP-37_paid_runs")
    return root / model / arm


def claim_rows(models=MODELS):
    """One row per claim: features, the target and the case it belongs to (case IDs stay in memory, never saved)."""
    rows = []
    for m in models:
        for arm, kind in ARMS.items():
            folder = cell(m, arm)
            metrics_file = cp._metrics_file(folder)
            traces_dir = folder / cp._traces_dir_for(metrics_file.name)
            if not traces_dir.exists():
                traces_dir = folder / "traces"
            for f in sorted(traces_dir.glob("*.json")):
                saved = json.loads(f.read_text())
                claims = {c.get("id") or c.get("claim_id"): c for c in saved.get("claims", [])}
                for t in saved["traces"]:
                    c = claims.get(t["claim_id"], {})
                    sources = c.get("declared_sources") or c.get("sources") or []
                    types = {s.get("type") for s in sources}
                    text = c.get("text") or ""
                    rows.append({"group": f"{arm}:{f.stem}", "model": m, "case type": kind, "field": t["field"],
                                 "kind": t.get("kind") or "other", "sources": len(sources),
                                 "cites a passage": int("passage" in types), "cites imaging": int("vision" in types),
                                 "cites the note": int("note" in types), "derived": int(bool(c.get("derived"))),
                                 "words": len(text.split()), "numbers": len(NUMBER.findall(text)),
                                 "flagged": int(t.get("subtype") is not None)})
    return rows


def design(rows):
    """(X, y, groups, column names, column -> feature)."""
    cols, owner = [], {}
    for f in CATEGORICAL:
        for v in sorted({r[f] for r in rows}):
            cols.append(f"{f}={v}")
            owner[cols[-1]] = f
    for f in NUMERIC:
        cols.append(f)
        owner[f] = f
    X = np.zeros((len(rows), len(cols)))
    index = {c: i for i, c in enumerate(cols)}
    for i, r in enumerate(rows):
        for f in CATEGORICAL:
            X[i, index[f"{f}={r[f]}"]] = 1
        for f in NUMERIC:
            X[i, index[f]] = r[f]
    y = np.array([r["flagged"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    return X, y, groups, cols, owner


def _model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                      random_state=SEED)


def validate(X, y, groups, n_boot=2000):
    """Out-of-fold AUC (5 folds grouped by case) with a 95% case-bootstrap interval."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        oof[test] = _model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    auc = float(roc_auc_score(y, oof))
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], oof[idx]))
    return {"auc": auc, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "claims": int(len(y)), "flagged": int(y.sum()), "cases": int(len(uniq))}


def explain(X, y, cols, owner):
    import shap
    fitted = _model().fit(X, y)
    sv = np.asarray(shap.TreeExplainer(fitted).shap_values(X))
    features = list(dict.fromkeys(owner.values()))
    grouped = {f: np.abs(sv[:, [i for i, c in enumerate(cols) if owner[c] == f]].sum(axis=1)).mean() for f in features}
    total = sum(grouped.values())
    per_level = {}
    for i, c in enumerate(cols):
        if "=" in c:
            on = X[:, i] == 1
            per_level[c] = {"claims": int(on.sum()), "mean_shap_when_present": float(sv[on, i].mean()) if on.any() else None,
                            "flag_rate": float(y[on].mean()) if on.any() else None}
    return {"share_of_attribution": {f: float(v / total) for f, v in sorted(grouped.items(), key=lambda kv: -kv[1])},
            "levels": per_level, "scale": "log-odds of being flagged"}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="SHAP row 11: surrogate of the verifier's flags")
    ap.add_argument("--out", default="runs/EXP-73_surrogate_flags")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    rows = claim_rows()
    X, y, groups, cols, owner = design(rows)
    result = {"experiment": "EXP-73", "label": "SURROGATE MODEL of the verifier's flags", "validity": validate(X, y, groups)}
    lo = result["validity"]["ci95"][0]
    result["explained"] = explain(X, y, cols, owner) if lo > 0.5 else None
    result["note"] = ("SHAP shown: the surrogate's AUC interval excludes 0.5" if lo > 0.5 else
                      "Not explained: the surrogate's AUC interval includes 0.5, so its SHAP values would mean nothing")
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("validity", "note")}, indent=1))
    if result["explained"]:
        print(json.dumps(result["explained"]["share_of_attribution"], indent=1))


if __name__ == "__main__":
    main()
