"""Student's t-tests beside the existing results (DEC-36, `docs/PHASE5_TTESTS.md`): the statistics.

`paired` and `welch` return one report row each: t, df, two-sided p, the mean difference with its 95% t interval, an
effect size (d_z, or Hedges' g), skewness and Shapiro-Wilk, and the rank test beside it (Wilcoxon signed-rank, or
Mann-Whitney U). `holm` adjusts the family's p-values. The case is the unit throughout; the callers give one value per
case, and a pair with a missing side is dropped and counted.
"""

import math

import numpy as np
from scipy import stats

ALPHA = 0.05


def _clean_pairs(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep], int((~keep).sum())


def _shape(x):
    x = np.asarray(x, dtype=float)
    sw = stats.shapiro(x).pvalue if len(x) >= 3 and np.ptp(x) > 0 else float("nan")
    sk = float(stats.skew(x)) if len(x) >= 3 and np.ptp(x) > 0 else float("nan")
    return {"skewness": sk, "shapiro_p": float(sw)}


def paired(a, b, label="", alpha=ALPHA):
    """Paired t-test of a - b, per case."""
    a, b, dropped = _clean_pairs(a, b)
    d = a - b
    n = len(d)
    if n < 2:
        raise ValueError("a paired t-test needs at least 2 pairs")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    df = n - 1
    if sd == 0:
        t, p, half = float("nan"), float("nan"), 0.0
    else:
        res = stats.ttest_rel(a, b)
        t, p = float(res.statistic), float(res.pvalue)
        half = stats.t.ppf(1 - alpha / 2, df) * sd / math.sqrt(n)
    nonzero = d[d != 0]
    w = stats.wilcoxon(nonzero) if len(nonzero) >= 1 else None
    return {"label": label, "form": "paired", "n": n, "dropped": dropped, "mean_a": float(a.mean()),
            "mean_b": float(b.mean()), "mean_difference": mean, "ci": [mean - half, mean + half], "t": t, "df": df,
            "p": p, "effect": "d_z", "effect_size": mean / sd if sd else float("nan"),
            "assumptions": _shape(d),
            "rank_test": {"name": "Wilcoxon signed-rank", "statistic": float(w.statistic) if w else float("nan"),
                          "p": float(w.pvalue) if w else float("nan"), "zero_differences": int(n - len(nonzero))}}


def welch(a, b, label="", alpha=ALPHA):
    """Welch's t-test of mean(a) - mean(b), unpaired."""
    a, b = (np.asarray(x, dtype=float) for x in (a, b))
    dropped = int((~np.isfinite(a)).sum() + (~np.isfinite(b)).sum())
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("Welch's t-test needs at least 2 cases per group")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    mean = float(a.mean() - b.mean())
    if se == 0:
        t, p, df, half = float("nan"), float("nan"), float("nan"), 0.0
    else:
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        res = stats.ttest_ind(a, b, equal_var=False)
        t, p = float(res.statistic), float(res.pvalue)
        half = stats.t.ppf(1 - alpha / 2, df) * se
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)  # Hedges' small-sample correction
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"label": label, "form": "Welch", "n": [na, nb], "dropped": dropped, "mean_a": float(a.mean()),
            "mean_b": float(b.mean()), "mean_difference": mean, "ci": [mean - half, mean + half], "t": t,
            "df": float(df), "p": p, "effect": "Hedges' g",
            "effect_size": j * mean / pooled if pooled else float("nan"),
            "assumptions": {"a": _shape(a), "b": _shape(b)},
            "rank_test": {"name": "Mann-Whitney U", "statistic": float(u.statistic), "p": float(u.pvalue)}}


def holm(pvalues):
    """Holm's step-down adjusted p-values, in the input order. NaN stays NaN and is not counted in the family."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = len(idx)
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def adjust(rows):
    """Add `p_holm` to each row, over the rows as one family."""
    for row, ph in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = float(ph)
    return rows


def agrees_with_bootstrap(row, bootstrap_ci):
    """True when the t interval and the existing bootstrap interval agree on whether 0 is excluded."""
    t_excl = row["ci"][0] > 0 or row["ci"][1] < 0
    b_excl = bootstrap_ci[0] > 0 or bootstrap_ci[1] < 0
    return t_excl == b_excl


def case_rate(per_case_row, numerator="hallucinated", denominator="claims"):
    """One case's rate from a metrics per_case row; NaN when it has no claims (a format failure)."""
    den = per_case_row.get(denominator) or 0
    return per_case_row[numerator] / den if den else float("nan")
