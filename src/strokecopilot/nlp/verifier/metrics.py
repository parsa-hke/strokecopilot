"""Hallucination, critical hallucination, omission and attribution rates with case bootstrap CIs.

Definitions are docs/HALLUCINATION_METRIC.md §6. Input is one record per reasoner output:

    {"case_id": str, "format_failure": bool, "traces": [Trace.to_dict(), ...],
     "omissions": int | None, "uncertainty_omissions": int | None,
     "decisions_correct": int | None, "decisions_determined": int | None}

A format failure has no traces and is left out of every claim denominator. Fields that need the
rulebook (scenarios.py) may be None until it exists; their rates are then None.
Claim rates pool claims across cases; the mean of per-case rates is reported next to them.
Intervals are 95% percentile bootstraps that resample cases, given only with at least 10 cases.
"""

import numpy as np

from strokecopilot.nlp.verifier.trace import CONTENT_FALSE, SUBTYPES

N_BOOT = 10_000
SEED = 20261104
MIN_CASES_FOR_CI = 10


def _counts(case):
    traces = [] if case.get("format_failure") else case["traces"]
    subtypes = [t["subtype"] for t in traces]
    critical = [t.get("critical") for t in traces]
    return {
        "outputs": 1,
        "format_failures": int(bool(case.get("format_failure"))),
        "claims": len(traces),
        "hallucinated": sum(s is not None for s in subtypes),
        "content_hallucinated": sum(s in CONTENT_FALSE for s in subtypes),
        "critical": sum(c is True for c in critical),
        "criticality_determined": sum(c is not None for c in critical),
        "content_traceable": sum(s not in CONTENT_FALSE for s in subtypes),
        "clean": sum(s is None for s in subtypes),
        "incomplete": sum(bool(t.get("incomplete")) for t in traces),
        "any_critical": int(any(c is True for c in critical)),
        "omission": None if case.get("omissions") is None else int(case["omissions"] > 0),
        "uncertainty_omission": None if case.get("uncertainty_omissions") is None else int(case["uncertainty_omissions"] > 0),
        "decisions_correct": case.get("decisions_correct"),
        "decisions_determined": case.get("decisions_determined"),
        **{f"subtype_{k}": sum(s == k for s in subtypes) for k in SUBTYPES},
    }


# metric -> (numerator column, denominator column)
RATES = {
    "hallucination_rate": ("hallucinated", "claims"),
    "content_hallucination_rate": ("content_hallucinated", "claims"),
    "critical_hallucination_rate": ("critical", "criticality_determined"),
    "cases_with_critical_hallucination": ("any_critical", "case_with_claims"),
    "omission_rate": ("omission", "omission_known"),
    "uncertainty_omission_rate": ("uncertainty_omission", "uncertainty_omission_known"),
    "source_attribution_accuracy": ("clean", "content_traceable"),
    "recommendation_correctness": ("decisions_correct", "decisions_determined"),
    "format_failure_rate": ("format_failures", "outputs"),
}


def _table(cases):
    rows = [_counts(c) for c in cases]
    columns = {key: np.array([0 if r[key] is None else r[key] for r in rows], dtype=float) for key in rows[0]}
    for key in ("omission", "uncertainty_omission"):
        columns[f"{key}_known"] = np.array([r[key] is not None for r in rows], dtype=float)
    columns["case_with_claims"] = 1 - columns["format_failures"]
    return columns


def _ratio(num, den):
    total = den.sum()
    return None if total == 0 else float(num.sum() / total)


def _boot_ratio(num, den, idx):
    n, d = num[idx].sum(axis=1), den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = n / d
    r = r[np.isfinite(r)]
    return None if len(r) == 0 else [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def summarise(cases, n_boot=N_BOOT, seed=SEED):
    """Every §6 metric for a group of cases: value, 95% CI (with >= 10 cases) and counts."""
    if not cases:
        raise ValueError("No cases to summarise.")
    columns = _table(cases)
    idx = np.random.default_rng(seed).integers(0, len(cases), size=(n_boot, len(cases))) if len(cases) >= MIN_CASES_FOR_CI else None
    out = {"cases": len(cases), "claims": int(columns["claims"].sum())}
    for name, (num, den) in RATES.items():
        value = _ratio(columns[num], columns[den])
        out[name] = {"value": value, "numerator": int(columns[num].sum()), "denominator": int(columns[den].sum()),
                     "ci95": _boot_ratio(columns[num], columns[den], idx) if idx is not None and value is not None else None}
    per_case = columns["hallucinated"][columns["claims"] > 0] / columns["claims"][columns["claims"] > 0]
    out["mean_per_case_hallucination_rate"] = float(per_case.mean()) if len(per_case) else None
    hallucinated = int(columns["hallucinated"].sum())
    out["subtype_breakdown"] = {SUBTYPES[k]: {"count": int(columns[f"subtype_{k}"].sum()),
                                              "share": None if hallucinated == 0 else float(columns[f"subtype_{k}"].sum() / hallucinated)}
                                for k in SUBTYPES}
    out["incomplete_attribution"] = int(columns["incomplete"].sum())
    return out


def paired_difference(cases_a, cases_b, metric="hallucination_rate", n_boot=N_BOOT, seed=SEED):
    """a minus b on the same cases, with a paired case bootstrap. Cases are matched by case_id."""
    by_id = {c["case_id"]: c for c in cases_b}
    common = [c for c in cases_a if c["case_id"] in by_id]
    if len(common) != len(cases_a) or len(common) != len(cases_b):
        raise ValueError("Paired comparison needs the same cases in both groups.")
    a, b = _table(common), _table([by_id[c["case_id"]] for c in common])
    num, den = RATES[metric]
    value_a, value_b = _ratio(a[num], a[den]), _ratio(b[num], b[den])
    out = {"metric": metric, "cases": len(common), "a": value_a, "b": value_b,
           "difference": None if value_a is None or value_b is None else value_a - value_b, "ci95": None}
    if len(common) >= MIN_CASES_FOR_CI and out["difference"] is not None:
        idx = np.random.default_rng(seed).integers(0, len(common), size=(n_boot, len(common)))
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = a[num][idx].sum(1) / a[den][idx].sum(1) - b[num][idx].sum(1) / b[den][idx].sum(1)
        diff = diff[np.isfinite(diff)]
        out["ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    return out
