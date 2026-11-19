"""Item 2 on ISLES'24 (EXP-50): flags under each candidate rule, and Grad-CAM on the flagged scans (docs/ITEM2_FALSE_ALARMS.md).

ISLES'24 has no bleed labels, so nothing here takes part in choosing a rule: EXP-49 chose on RSNA validation, and this
reads out what that choice does on the 129 presumed bleed-free development scans. Each scan goes through EXP-16's
preparation (`isles_check.prepare`: 5 mm slabs, windows, JPEG) and EXP-6's model; then, per slab, the brain fraction and
height (`explain.slice_measures`).

Thresholds:
- "none" uses DEC-28: EXP-6's calibration with `infer.CONTRACT_THRESHOLD`, as the contracts were built (EXP-33), so its
  flags should reproduce EXP-16's 68.
- every other candidate uses the raw threshold EXP-49 exported for it (`false_alarms.full_split`). A run of EXP-49
  made before that export was added (19 Nov) gives "none" only.

The per-case table holds case IDs and stays on Drive; metrics.json holds counts, rates and medians only.
"""

import json
import math

import numpy as np
import pandas as pd

from strokecopilot.data import splits
from strokecopilot.vision.haemorrhage import explain as ex
from strokecopilot.vision.haemorrhage import isles_check as ic
from strokecopilot.vision.haemorrhage import slab_rules as sr
from strokecopilot.vision.haemorrhage.stats import wilson

TOP = 2  # EXP-6's aggregation reads the top 2 slabs; Grad-CAM looks at the same two
SKULL_BASE_RELATIVE_BF = 0.40  # descriptive only: a top slab below 0.40 x its scan's largest brain fraction


def raw_threshold(calibrator):
    """The raw scan score at which `calibrator` reaches its threshold (Platt and temperature are monotonic)."""
    if not calibrator.temperature > 0:
        raise ValueError("Calibration temperature must be positive.")
    p = min(max(calibrator.threshold, 1e-9), 1 - 1e-9)
    z = (math.log(p / (1 - p)) - calibrator.bias) * calibrator.temperature
    return 1 / (1 + math.exp(-z))


def thresholds(contract_calibrator, exp49=None):
    """{name: (rule, ratio, raw threshold)}: "none" from DEC-28, the others from EXP-49's metrics (its "selection"),
    for every candidate that carries a `full_split` export."""
    out = {"none": ("none", None, raw_threshold(contract_calibrator))}
    rows = ((exp49 or {}).get("selection") or {}).get("candidates") or {}
    for name, row in rows.items():
        if name != "none" and row.get("full_split"):
            out[name] = (row["rule"], row["ratio"], float(row["full_split"]["threshold_raw"]))
    return out


def check_aggregation(calibrator):
    """slab_rules scores with EXP-6's top-2 mean; the calibration must use the same aggregation."""
    if (calibrator.method, calibrator.k) != ("topk", sr.K):
        raise ValueError(f"The calibration aggregates by {calibrator.method} k={calibrator.k}, not the top-{sr.K} mean.")


def scan_case(img, predict):
    """Native NCCT -> (uint8 slabs, slab probabilities, per-slab measures, preparation info), as EXP-16 prepared it."""
    images, info = ic.prepare(img)
    probs = np.asarray(predict(images), dtype=float)
    if probs.shape != (len(images),):
        raise ValueError(f"predict returned {probs.shape} for {len(images)} slabs.")
    return images, probs, ex.slice_measures(images), info


def score_rows(case, probs, measures, rules):
    """One table row: each candidate's score and call, and where the scan's top slabs sit."""
    bf = np.array([m["brain_fraction"] for m in measures])
    row = {"case": case, "slabs": int(len(probs)), "error": ""}
    for name, (rule, ratio, threshold) in rules.items():
        score = sr.scan_score(probs, bf if rule in ("gate", "both") else None, rule, ratio)
        row[f"{name}_score"], row[f"{name}_called"] = float(score), bool(score >= threshold)
    top = np.argsort(-probs, kind="stable")[:TOP]
    peak = bf.max() if bf.size and bf.max() > 0 else 1.0
    row["top_slabs"] = ";".join(str(int(i)) for i in top)
    row["top_prob"] = float(probs[top[0]])
    row["top_height"] = float(measures[top[0]]["height"])
    row["top_relative_bf"] = float(bf[top[0]] / peak)
    row["top_adjacent"] = bool(len(top) < 2 or abs(int(top[0]) - int(top[1])) == 1)
    return row


def run(cases, load_image, predict, rules, explain_net=None, device="cpu", progress=None):
    """Score every case; Grad-CAM on the top slabs of each scan flagged under "none" when `explain_net` is given.
    Returns (per-case table with case IDs, Grad-CAM rows without IDs, kept example slabs for the figure)."""
    rows, cams, examples = [], [], []
    for i, case in enumerate(cases, 1):
        try:
            images, probs, measures, _ = scan_case(load_image(case), predict)
            row = score_rows(case, probs, measures, rules)
            if explain_net is not None and row["none_called"]:
                top = [int(s) for s in row["top_slabs"].split(";")]
                explained, maps = ex.explain_slices(explain_net, images[top], device)
                for rank, (s, e, m) in enumerate(zip(top, explained, maps)):
                    rel = measures[s]["brain_fraction"] / max(max(x["brain_fraction"] for x in measures), 1e-9)
                    cams.append({"rank": rank, **e, "relative_bf": rel, "height": measures[s]["height"],
                                 "adjacent_pair": row["top_adjacent"]})
                    if rank == 0:
                        examples.append({"image": images[s], "cam": m, "prob": e["prob"], "height": measures[s]["height"],
                                         "relative_bf": rel, "brain": e["brain"], "bone": e["bone"]})
        except splits.HeldoutError:
            raise  # a held-out case stops the run (it never reaches here: list_cases leaves them out)
        except Exception as error:  # reported, never hidden
            row = {"case": case, "error": f"{type(error).__name__}: {str(error)[:200]}"}
        rows.append(row)
        if progress:
            progress(i, len(cases))
    return pd.DataFrame(rows), cams, examples


def _rate(k, n):
    lo, hi = wilson(k, n)
    return {"flagged": int(k), "scans": int(n), "rate": None if not n else round(k / n, 4),
            "ci95": None if lo is None else [round(lo, 4), round(hi, 4)]}


def mcnemar(a, b):
    """Paired calls a and b (booleans): both, only a, only b, and the exact two-sided McNemar p."""
    from scipy.stats import binomtest

    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return {"both": int((a & b).sum()), "only_first": only_a, "only_second": only_b, "p_exact": round(float(p), 4)}


def summarise(table, cams, rules, proposal=None):
    """Counts, rates, paired tests and medians; no case IDs."""
    ok = table[table["error"] == ""] if "error" in table else table
    ok = ok.reindex(columns=sorted(set(ok.columns) | {f"{n}_called" for n in rules} | {"top_height", "top_relative_bf",
                                                                                         "top_adjacent"}))
    for name in rules:  # a table whose every case failed has no call columns
        ok[f"{name}_called"] = ok[f"{name}_called"].fillna(False).astype(bool)
    out = {"cases": int(len(table)), "scored": int(len(ok)), "failed": int(len(table) - len(ok)), "candidates": {}}
    for name in rules:
        out["candidates"][name] = _rate(int(ok[f"{name}_called"].sum()), len(ok))
        if name != "none":
            out["candidates"][name]["vs_none"] = mcnemar(ok["none_called"], ok[f"{name}_called"])
    out["proposal"] = proposal
    flagged, clear = ok[ok["none_called"]], ok[~ok["none_called"]]
    med = lambda s: None if s.empty else round(float(s.median()), 4)  # noqa: E731
    out["top_slab"] = {group: {"scans": int(len(f)), "height": med(f["top_height"]), "relative_bf": med(f["top_relative_bf"]),
                               "adjacent_pair": None if f.empty else round(float(f["top_adjacent"].mean()), 4),
                               "skull_base_share": None if f.empty else round(float((f["top_relative_bf"] < SKULL_BASE_RELATIVE_BF).mean()), 4)}
                       for group, f in (("flagged", flagged), ("not_flagged", clear))}
    if cams:
        c = pd.DataFrame(cams)
        out["gradcam_flagged"] = {"slabs": int(len(c)), **{k: med(c[k]) for k in ("prob", "brain", "bone", "other", "outside",
                                                                                   "relative_bf", "height")},
                                  "bone_over_brain_share": round(float((c["bone"] > c["brain"]).mean()), 4)}
    return out


def gradcam_figure(examples, out_png, n=8, seed=0):
    """Up to n flagged scans' top slab (brain window) with its Grad-CAM, drawn by seed. Captions hold numbers only."""
    import matplotlib.pyplot as plt

    if not examples:
        return None
    rng = np.random.default_rng(seed)
    picked = [examples[i] for i in sorted(rng.choice(len(examples), size=min(n, len(examples)), replace=False))]
    cols = 4
    rows = math.ceil(len(picked) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.7 * rows), facecolor="#fcfcfb")
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, e in zip(np.atleast_1d(axes).ravel(), picked):
        ax.imshow(e["image"][..., 0], cmap="gray")
        ax.imshow(e["cam"], cmap="inferno", alpha=0.45)
        ax.set_title(f"p {e['prob']:.2f} · height {e['height']:.2f}\nheat: brain {e['brain']:.0%}, bone {e['bone']:.0%}",
                     fontsize=9, color="#0b0b0b")
    fig.suptitle("ISLES'24 scans flagged at DEC-28: top slab and Grad-CAM (presumed bleed-free)", fontsize=11, x=0.01,
                 ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)  # room for the title and for each row's two-line captions
    fig.savefig(out_png, dpi=140, facecolor="#fcfcfb")
    plt.close(fig)
    return out_png


def contract_check(table, expected=68):
    """EXP-16 flagged 68 of 129 at DEC-28 on 5 mm slabs; "none" here should reproduce it."""
    got = int(table.loc[table["error"] == "", "none_called"].sum()) if "none_called" in table else None
    return {"none_flags": got, "expected": expected, "reproduced": got == expected}



DRIVE = "~/Library/CloudStorage/GoogleDrive-parsahosseini.it@gmail.com/My Drive/StrokeCopilot"
SLIM_SHA256 = "b0afd16e4cbfb16a69fb5190083fc3869f5dcc6c92f86f10135b0de1f59f4728"  # as notebook 21 checks it
FOLDS_SHA256 = "023813f2dba930b7521653f1549b1fa1369b8bd29b63658175c5cea58410d291"
EXPECTED_DEVELOPMENT = 129


def main(argv=None):
    """EXP-50 on the Mac: python -m strokecopilot.vision.haemorrhage.isles_false_alarms --out <Drive run folder>
    [--exp49 <EXP-49 metrics.json>]. Writes config.json, metrics.json, per_case.csv (case IDs: the folder must be on
    Drive) and gradcam_flagged.png."""
    import argparse
    import hashlib
    import os
    import time
    from pathlib import Path

    import torch

    from strokecopilot.data import isles24
    from strokecopilot.data.archive_listing import sha256_of
    from strokecopilot.vision.haemorrhage import infer, train

    parser = argparse.ArgumentParser(description="EXP-50: Vision B's candidate rules and Grad-CAM on ISLES'24.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--exp49", help="EXP-49's metrics.json; without it, only 'none' (DEC-28) is read out")
    parser.add_argument("--model-dir", default=f"{DRIVE}/runs/EXP-6_visionB_resnet50")
    parser.add_argument("--slim-tar", default=f"{DRIVE}/slim/isles24_slim_v1.tar")
    parser.add_argument("--experiment", default="EXP-50")
    args = parser.parse_args(argv)
    started = time.time()
    out = Path(os.path.expanduser(args.out))
    if "GoogleDrive" not in str(out.resolve()):
        raise ValueError("--out must be on Drive: per_case.csv holds ISLES'24 case IDs.")
    if (out / "metrics.json").exists():
        raise FileExistsError(f"{out} already has metrics.json.")
    out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(splits.FOLDS_PATH.read_bytes()).hexdigest() != FOLDS_SHA256:
        raise ValueError("The installed split is not the fixed one. Stop and tell Parsa.")
    folds = splits.load_folds()
    local = Path(__file__).resolve().parents[4] / "data" / "local"  # the repo's data/local (never committed)
    root = local / isles24.SLIM_NAME
    if not (root / "manifest.json").exists():
        tar = os.path.expanduser(args.slim_tar)
        if sha256_of(tar) != SLIM_SHA256:
            raise ValueError("The ISLES'24 tarball is not the one the build recorded.")
        root = Path(isles24.fetch_slim(tar, parent=str(local)))
    cases = isles24.list_cases(root, folds)
    if len(cases) != EXPECTED_DEVELOPMENT:
        raise ValueError(f"{len(cases)} development cases, expected {EXPECTED_DEVELOPMENT}.")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = Path(os.path.expanduser(args.model_dir))
    net, state = train.load_best(model_dir / "best.pt", device)
    contract = infer.contract_calibrator(model_dir / "calibration.json")
    check_aggregation(contract)
    exp49 = json.loads(Path(os.path.expanduser(args.exp49)).read_text()) if args.exp49 else None
    rules = thresholds(contract, exp49)
    proposal = ((exp49 or {}).get("selection") or {}).get("proposal")
    config = {"experiment": args.experiment, "task": "Item 2: ISLES'24 read-out of the candidate rules, and Grad-CAM",
              "preregistration": "docs/ITEM2_FALSE_ALARMS.md", "cases": len(cases), "device": str(device),
              "model": {"epoch": state["epoch"], "best_pt_sha256": sha256_of(model_dir / "best.pt")},
              "contract_calibration": vars(contract), "rules": {k: list(v) for k, v in rules.items()},
              "exp49": {"given": bool(exp49), "commit": (exp49 or {}).get("commit"), "proposal": proposal},
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "config.json").write_text(json.dumps(config, indent=1, default=str))

    def predict(images):
        return infer.slice_probs(net, images, device)

    def load(case):
        return isles24.load_native_ncct(case, root, folds)

    def progress(done, total):
        if done % 20 == 0 or done == total:
            print(f"{done}/{total} cases, {time.time() - started:.0f} s", flush=True)

    table, cams, examples = run(cases, load, predict, rules, explain_net=net, device=device, progress=progress)
    metrics = {"experiment": args.experiment, **summarise(table, cams, rules, proposal), "contract_check": contract_check(table),
               "minutes": round((time.time() - started) / 60, 1)}
    text = json.dumps(metrics, indent=1, default=str)
    if ic.ids_in(text, cases + list(folds["heldout"])):
        raise ValueError("A case ID reached metrics.json. Nothing was saved.")
    if set(table["case"]) - set(cases):
        raise ValueError("The table holds a case outside the development set. Nothing was saved.")
    (out / "metrics.json").write_text(text)
    table.to_csv(out / "per_case.csv", index=False)
    gradcam_figure(examples, out / "gradcam_flagged.png")
    print(json.dumps({k: metrics[k] for k in ("scored", "failed", "candidates", "contract_check", "top_slab")}, indent=1))
    print(f"OK: {args.experiment}: {metrics['scored']} scans; none flags {metrics['candidates']['none']['flagged']}"
          + (f"; {proposal} flags {metrics['candidates'][proposal]['flagged']}" if proposal and proposal in rules and proposal != "none" else "")
          + f"; {metrics['minutes']} min.")
    return metrics


if __name__ == "__main__":
    main()
