"""Item 2's selection on RSNA validation (EXP-49), exactly as docs/ITEM2_FALSE_ALARMS.md fixed it before any score.

Input: one row per validation slice with series_id, patient_id, any, the five subtype labels, z_mm, prob (EXP-6's
best.pt) and brain_fraction (explain.tissue_masks, via measure()). For each of slab_rules.candidates():

1. scan scores (slab_rules.scan_table);
2. the threshold for TARGET sensitivity cross-fitted on patient halves (operating_point.cross_fit), for each of the
   20 seeds of EXP-17; pooled held-out sensitivity and specificity, and sensitivity per bleed subtype, each averaged
   over the seeds; patient-bootstrap intervals at the main seed;
3. eligibility: mean sensitivity no more than SENS_MARGIN below "none"'s, and no subtype more than SUBTYPE_MARGIN
   below "none"'s;
4. the proposal: the eligible candidate with the highest mean specificity ("none" itself when nothing beats it);
5. the proposal's specificity gain over "none": a paired patient bootstrap on the main seed's held-out calls.
   Adoption needs its interval above 0 and Parsa's decision.

Everything returned is counts, rates and thresholds: no slice, scan or patient ID (docs/DATA_USE.md).
"""

import numpy as np
import pandas as pd
import torch

from strokecopilot.vision.haemorrhage import explain as ex
from strokecopilot.vision.haemorrhage import operating_point as op
from strokecopilot.vision.haemorrhage import slab_rules as sr
from strokecopilot.vision.haemorrhage import train

TARGET = 0.96  # DEC-28's target
SUBTYPES = ("epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural")
SENS_MARGIN = 0.01
SUBTYPE_MARGIN = 0.03
REFERENCE_PER_GROUP = 20  # Grad-CAM reference: true positives and false positives at DEC-28's threshold


def label(rule, ratio):
    return rule if ratio is None else f"{rule}({ratio:.2f})"


def scans_for(slices, rule, ratio, k=sr.K):
    """One row per scan under a rule: series_id, patient_id, label, score and one 0/1 column per subtype."""
    table = sr.scan_table(slices, rule, ratio, k)
    per_scan = slices.groupby("series_id").agg(patient_id=("patient_id", "first"),
                                               **{s: (s, "max") for s in SUBTYPES})
    if (slices.groupby("series_id")["patient_id"].nunique() > 1).any():
        raise ValueError("A scan belongs to more than one patient.")
    return table.merge(per_scan.reset_index(), on="series_id", how="left")


def _subtype_sensitivity(called, scans):
    return {s: (float(called[scans[s].to_numpy() == 1].mean()) if (scans[s] == 1).any() else None) for s in SUBTYPES}


def evaluate(scans, target=TARGET, seeds=op.REPEAT_SEEDS, main_seed=op.SEED, bootstrap=op.BOOTSTRAP):
    """Cross-fitted results for one candidate's scan table. Returns (summary, held-out calls at the main seed)."""
    scores, labels, patients = scans["score"].to_numpy(), scans["label"].to_numpy(), scans["patient_id"].to_numpy()
    sens, spec, sub = [], [], {s: [] for s in SUBTYPES}
    main_called = None
    for seed in seeds:
        report, called = op.cross_fit(scores, labels, patients, target, seed)
        sens.append(report["pooled"]["sensitivity"])
        spec.append(report["pooled"]["specificity"])
        for s, v in _subtype_sensitivity(called, scans).items():
            if v is not None:
                sub[s].append(v)
        if seed == main_seed:
            main_called, main_report = called, report
    if main_called is None:
        main_report, main_called = op.cross_fit(scores, labels, patients, target, main_seed)
    boot = op.patient_bootstrap(main_called, labels, patients, bootstrap, main_seed)
    mean = lambda v: op._r(float(np.mean(v))) if v else None  # noqa: E731
    summary = {"seeds": len(seeds), "sensitivity": mean(sens), "specificity": mean(spec),
               "sensitivity_range": [op._r(min(sens)), op._r(max(sens))],
               "specificity_range": [op._r(min(spec)), op._r(max(spec))],
               "subtype_sensitivity": {s: mean(v) for s, v in sub.items()},
               "subtype_bleeds": {s: int((scans[s] == 1).sum()) for s in SUBTYPES},
               "main_seed": {k: main_report["pooled"][k] for k in ("sensitivity", "specificity", "tp", "fn", "tn", "fp")},
               "main_seed_bootstrap": boot}
    return summary, main_called


def eligible(summary, none, sens_margin=SENS_MARGIN, subtype_margin=SUBTYPE_MARGIN):
    """(ok, reasons): the pre-registered non-inferiority on sensitivity, overall and per subtype."""
    reasons = []
    if summary["sensitivity"] < none["sensitivity"] - sens_margin:
        reasons.append(f"sensitivity {summary['sensitivity']:.4f} < {none['sensitivity']:.4f} - {sens_margin}")
    for s in SUBTYPES:
        a, b = summary["subtype_sensitivity"][s], none["subtype_sensitivity"][s]
        if a is not None and b is not None and a < b - subtype_margin:
            reasons.append(f"{s} {a:.4f} < {b:.4f} - {subtype_margin}")
    return not reasons, reasons


def paired_specificity_gain(called_a, called_b, labels, patients, n=op.BOOTSTRAP, seed=op.SEED):
    """Specificity of a minus b on the same scans, with a paired 95% patient-bootstrap interval."""
    frame = pd.DataFrame({"patient": np.asarray(patients), "neg": ~np.asarray(labels).astype(bool),
                          "a": ~np.asarray(called_a).astype(bool), "b": ~np.asarray(called_b).astype(bool)})
    frame["tn_a"], frame["tn_b"] = frame["a"] & frame["neg"], frame["b"] & frame["neg"]
    per = frame.groupby("patient")[["tn_a", "tn_b", "neg"]].sum().to_numpy(dtype=float)
    tn_a, tn_b, neg = per.sum(axis=0)
    weights = np.random.default_rng(seed).multinomial(len(per), np.full(len(per), 1 / len(per)), size=n)
    a, b, d = (weights @ per).T
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (a - b) / d
    diff = diff[np.isfinite(diff)]
    return {"gain": op._r((tn_a - tn_b) / neg), "ci95": [op._r(np.percentile(diff, 2.5)), op._r(np.percentile(diff, 97.5))],
            "replicates": n}


def select(slices, target=TARGET, seeds=op.REPEAT_SEEDS, ratios=sr.GATE_RATIOS, bootstrap=op.BOOTSTRAP, log=print):
    """The whole pre-registered selection. Returns a report with numbers only."""
    missing = {"series_id", "patient_id", "any", "prob", "z_mm", "brain_fraction", *SUBTYPES} - set(slices.columns)
    if missing:
        raise ValueError(f"slices lacks {sorted(missing)}.")
    rows, calls, scans_none = {}, {}, None
    for rule, ratio in sr.candidates(ratios):
        scans = scans_for(slices, rule, ratio)
        if scans_none is None:
            scans_none = scans
        elif not scans["series_id"].equals(scans_none["series_id"]):
            scans = scans.set_index("series_id").loc[scans_none["series_id"]].reset_index()
        summary, called = evaluate(scans, target, seeds, bootstrap=bootstrap)
        name = label(rule, ratio)
        rows[name], calls[name] = {"rule": rule, "ratio": ratio, **summary}, called
        log(f"{name:12s} sensitivity {summary['sensitivity']:.4f}, specificity {summary['specificity']:.4f}")
    none = rows["none"]
    for name, r in rows.items():
        r["eligible"], r["ineligible_because"] = eligible(r, none) if name != "none" else (True, [])
    best = max((n for n, r in rows.items() if r["eligible"]), key=lambda n: (rows[n]["specificity"], n == "none"))
    labels, patients = scans_none["label"].to_numpy(), scans_none["patient_id"].to_numpy()
    gain = None if best == "none" else paired_specificity_gain(calls[best], calls["none"], labels, patients, bootstrap)
    adopt = bool(gain and gain["ci95"][0] > 0)
    return {"target": target, "scans": int(len(scans_none)), "bleed_scans": int(labels.sum()),
            "patients": int(pd.Series(patients).nunique()), "candidates": rows, "proposal": best,
            "specificity_gain": gain, "meets_adoption_rule": adopt,
            "note": "Adoption also needs Parsa's decision; it amends DEC-28 (docs/ITEM2_FALSE_ALARMS.md)."}


# --- inputs on Kaggle ---------------------------------------------------------------------------------------------

def measure(items, readers, workers=4, batch_size=256):
    """Brain fraction and head area for each slice of `items` (shard, image_id, any), read through the same shard
    readers as training. Parallel over the data loader's workers."""
    out = [b for b in train.loader(Measures(items, readers), batch_size=batch_size, workers=workers)]
    values = torch.cat(out).numpy() if out else np.zeros((0, 2))
    return pd.DataFrame({"brain_fraction": values[:, 0], "head_px": values[:, 1]}, index=items.index)


class Measures(train.SliceDataset):
    """Shard slices -> [brain fraction, head pixels] per slice (explain.tissue_masks)."""

    def __getitem__(self, i):
        image, _ = super().__getitem__(i)
        m = ex.tissue_masks(image.numpy())
        head = int(m["head"].sum())
        return torch.tensor([m["brain"].sum() / head if head else 0.0, float(head)])


def heights(slices):
    """Each slice's height in its scan, by the rule of explain.slice_measures, from head_px and z_mm."""
    out = pd.Series(np.nan, index=slices.index)
    for _, g in slices.sort_values("z_mm").groupby("series_id", sort=False):
        heads = g["head_px"].to_numpy(dtype=float)
        present = np.flatnonzero(heads >= 0.10 * heads.max()) if heads.max() > 0 else np.array([], dtype=int)
        lo, hi = (int(present[0]), int(present[-1])) if present.size else (0, 0)
        idx = np.arange(len(g))
        out.loc[g.index] = 0.5 if hi == lo else np.clip((idx - lo) / (hi - lo), 0, 1)
    return out


def reference_slices(slices, calibrator, threshold, n=REFERENCE_PER_GROUP, seed=op.SEED):
    """The Grad-CAM reference: for n true-positive and n false-positive scans at `threshold` (EXP-6's scan score and
    calibration, DEC-28's threshold), drawn by seed, the scan's hottest slice. Returns those slice rows with a
    `group` column ("tp" or "fp"); the rows stay on Kaggle."""
    scans = sr.scan_table(slices, "none")
    called = calibrator.apply(scans["score"].to_numpy()) >= threshold
    rng = np.random.default_rng(seed)
    picked = []
    for group, mask in (("tp", called & (scans["label"] == 1)), ("fp", called & (scans["label"] == 0))):
        ids = scans.loc[mask, "series_id"].to_numpy()
        for sid in rng.choice(ids, size=min(n, len(ids)), replace=False):
            hottest = slices[slices["series_id"] == sid]["prob"].idxmax()
            picked.append((hottest, group))
    rows = slices.loc[[i for i, _ in picked]].copy()
    rows["group"] = [g for _, g in picked]
    return rows


def reference_summary(rows, explained):
    """Medians per group (tp, fp) of the heat shares, brain fraction and height. explained: explain.explain_slices
    rows in the same order as `rows`. Numbers only."""
    frame = pd.DataFrame(explained, index=rows.index).assign(group=rows["group"].to_numpy(),
                                                             brain_fraction=rows["brain_fraction"].to_numpy(),
                                                             height=rows["height"].to_numpy())
    cols = ["prob", "brain", "bone", "other", "outside", "brain_fraction", "height"]
    return {g: {"slices": int(len(f)), **{c: op._r(float(f[c].median())) for c in cols}} for g, f in frame.groupby("group")}
