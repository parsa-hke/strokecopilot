"""Candidate scan-score rules for Vision B's ISLES'24 false alarms (Item 2). Fixed before any result is looked at.

EXP-6's scan score is the mean of the top k = 2 slice probabilities. EXP-35 read the 23 scans flagged at 0.175 but not
0.278: 12 were driven by skull-base slabs, and none read as a bleed. Two changes target that, and the set is fixed
here, with its parameter grid, before any candidate is scored:

- "none": EXP-6's score, unchanged (the comparison).
- "gate": slices whose brain fraction (explain.slice_measures) is below `ratio` times the scan's largest brain fraction
  are left out of the top-k. The cut is relative to the scan's own largest, so it does not depend on how the density
  masks behave on each dataset. If fewer than k slices pass, the score falls back to "none": the gate never removes a
  scan's evidence entirely.
- "adjacent": the top k must be neighbouring slices: the score is the largest mean of k consecutive slices, in order
  along the body (RSNA's z_mm; ISLES'24's slab order). A bleed spans neighbouring 5 mm slices more often than two
  isolated hot slices do.
- "both": the largest mean of k consecutive slices that all pass the gate; falls back to "adjacent" when no run of k
  passes.

GATE_RATIOS is the grid for `ratio`. Parameters are chosen on RSNA validation only, cross-fitted on patient halves
as in EXP-17 (docs/ITEM2_FALSE_ALARMS.md has the selection and adoption rule). ISLES'24 has no bleed labels, so its
flag count is read out, never optimised (DEC-28).
"""

import numpy as np

RULES = ("none", "gate", "adjacent", "both")
GATE_RATIOS = (0.25, 0.40, 0.55, 0.70)
K = 2  # EXP-6's aggregation: mean of the top 2


def candidates(ratios=GATE_RATIOS):
    """[(rule, ratio)] in a fixed order; ratio is None for the rules without a gate."""
    return ([("none", None)] + [("gate", r) for r in ratios] + [("adjacent", None)] + [("both", r) for r in ratios])


def _topk(probs, k):
    return float(np.sort(probs)[-k:].mean())


def _best_run(probs, k):
    if len(probs) < k:
        return _topk(probs, len(probs))
    windows = np.lib.stride_tricks.sliding_window_view(probs, k)
    return float(windows.mean(axis=1).max())


def passes_gate(brain_fraction, ratio):
    bf = np.asarray(brain_fraction, dtype=float)
    top = bf.max() if bf.size else 0.0
    return bf >= ratio * top if top > 0 else np.ones_like(bf, dtype=bool)


def scan_score(slice_probs, brain_fraction=None, rule="none", ratio=None, k=K):
    """One scan's score under a rule. slice_probs and brain_fraction are in order along the body."""
    p = np.asarray(slice_probs, dtype=float)
    if p.size == 0:
        raise ValueError("A scan needs at least one slice.")
    k = min(k, p.size)
    if rule == "none":
        return _topk(p, k)
    if rule == "adjacent":
        return _best_run(p, k)
    if rule not in ("gate", "both"):
        raise ValueError(f"Unknown rule {rule!r}; one of {RULES}.")
    if brain_fraction is None or ratio is None:
        raise ValueError(f"Rule {rule!r} needs brain_fraction and ratio.")
    ok = passes_gate(brain_fraction, ratio)
    if len(ok) != p.size:
        raise ValueError("slice_probs and brain_fraction differ in length.")
    if rule == "gate":
        return _topk(p[ok], k) if ok.sum() >= k else _topk(p, k)
    runs = [p[i : i + k].mean() for i in range(p.size - k + 1) if ok[i : i + k].all()]
    return float(max(runs)) if runs else _best_run(p, k)


def scan_table(slices, rule, ratio=None, k=K, prob="prob", order="z_mm", brain="brain_fraction"):
    """One row per scan (series_id, label, score) under a rule, like aggregate.scan_scores. `slices` needs series_id,
    any, the probability, the order column and, for gated rules, the brain fraction."""
    import pandas as pd

    rows = []
    for sid, g in slices.sort_values(order).groupby("series_id", sort=False):
        bf = g[brain].to_numpy() if rule in ("gate", "both") else None
        rows.append({"series_id": sid, "label": int(g["any"].max()),
                     "score": scan_score(g[prob].to_numpy(), bf, rule, ratio, k)})
    return pd.DataFrame(rows)
