"""DEC-36's list of t-tests (`docs/PHASE5_TTESTS.md` §3), loaded from the saved runs and run beside the reported results.

Before any t statistic, each comparison that already has a reported bootstrap result is recomputed from the same files
with the same code and seed (`metrics.paired_difference`), and must match it; a mismatch stops the run. Then the
per-case values go to `ttests.paired` or `ttests.welch`, and Holm's correction runs over every row as one family.

T7 reads `heldout_rates.json`, written by `phase4 rates` (the only held-out step; post hoc, allowed by Parsa on 20 Nov
2026). T9 waits for task D. Nothing here reads a held-out case.
"""

import csv
import json
from pathlib import Path

import numpy as np

from strokecopilot.analysis import ttests
from strokecopilot.nlp.verifier import metrics as vm
from strokecopilot.pipeline import compare as cp

EXP37, EXP47 = Path("runs/EXP-37_paid_runs"), Path("runs/EXP-47_small_reasoner")
EXP42_REPORT = Path("runs/EXP-42_c9_table/rescore_points_19_20/report.json")
EXP48_REPORT = Path("runs/EXP-48_small_reasoner_report/rescore_points_19_20/report.json")
SONNET, QWEN, GEMMA, MEDGEMMA = "claude-sonnet-5-5", "qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text"
BASE, TUNED = "gemma-3-4b", "gemma-3-4b-tuned"
BIG, SMALL = (SONNET, QWEN, GEMMA, MEDGEMMA), (BASE, TUNED)
ALL_MODELS = BIG + SMALL
EXP25 = {"dice": (0.104, [0.078, 0.133]), "abs_error_ml": (-1.6, [-4.5, 1.3])}  # DEC-25, nnU-Net minus threshold
TOL = 1e-9


class Mismatch(ValueError):
    """A recomputed bootstrap result differs from the reported one: stop before any t statistic."""


def cell(model, arm):
    return str((EXP47 if model in SMALL else EXP37) / model / arm)


def _pool(folders, keep=None):
    out = {}
    for folder in folders:
        for r in cp.case_records(folder):
            if keep is None or keep(r["case_id"]):
                out[r["case_id"]] = r
    return out


def _check(boot, reported, label):
    same = (abs(boot["difference"] - reported["difference"]) < TOL
            and all(abs(x - y) < TOL for x, y in zip(boot["ci95"], reported["ci95"])))
    if not same:
        raise Mismatch(f"{label}: recomputed {boot['difference']} {boot['ci95']}, reported {reported['difference']} "
                       f"{reported['ci95']}")


def nlp_paired(test_id, label, folders_a, folders_b, metric, reported, source, keep=None):
    a, b = _pool(folders_a, keep), _pool(folders_b, keep)
    common = sorted(set(a) & set(b))
    ra, rb = [a[c] for c in common], [b[c] for c in common]
    boot = vm.paired_difference(ra, rb, metric)
    _check(boot, reported, test_id)
    row = ttests.paired(ttests.case_rates(ra, metric), ttests.case_rates(rb, metric), label)
    row.update(id=test_id, measure=f"per-case {metric.replace('_', ' ')}", existing={
        "source": source, "pooled_difference": boot["difference"], "ci95": boot["ci95"], "kind": "paired case bootstrap"})
    row["agrees_with_existing"] = ttests.agrees_with_bootstrap(row, boot["ci95"])
    return row


def vision_a_rows(per_case_csv):
    """T1, T2: nnU-Net minus the threshold per development case (EXP-27), checked against DEC-25's numbers."""
    rows = list(csv.DictReader(open(per_case_csv)))
    out = []
    for test_id, key, a_col, b_col, digits in (("T1", "dice", "nnunet_dice", "threshold_dice", 3),
                                                ("T2", "abs_error_ml", "nnunet_abs_error_ml", "threshold_abs_error_ml", 1)):
        a = np.array([float(r[a_col]) for r in rows])
        b = np.array([float(r[b_col]) for r in rows])
        reported, ci = EXP25[key]
        if round(float((a - b).mean()), digits) != reported:
            raise Mismatch(f"{test_id}: mean difference {(a - b).mean():.4f} against DEC-25's {reported}")
        row = ttests.paired(a, b, f"nnU-Net minus threshold, {key.replace('_', ' ')}")
        row.update(id=test_id, measure=f"per-case {key.replace('_', ' ')}",
                   existing={"source": "DEC-25 / EXP-25", "pooled_difference": reported, "ci95": ci,
                             "kind": "paired case bootstrap"})
        row["agrees_with_existing"] = ttests.agrees_with_bootstrap(row, ci)
        out.append(row)
    return out, rows


def nlp_rows():
    """T3 to T6, each checked against EXP-42's or EXP-48's reported paired bootstrap."""
    from strokecopilot.pipeline import small_reasoner_report as srr
    r42, r48 = json.loads(EXP42_REPORT.read_text()), json.loads(EXP48_REPORT.read_text())
    rb_a, rb_b = [a for a, _ in cp.RULEBOOK_PAIRS], [b for _, b in cp.RULEBOOK_PAIRS]
    head = lambda m: [cell(m, a) for a in cp.HEADLINE_ARMS]
    out = []
    for k, m in enumerate(BIG):
        out.append(nlp_paired(f"T3{'abcd'[k]}", f"{m}: with minus without the rulebook", [cell(m, a) for a in rb_a],
                              [cell(m, b) for b in rb_b], "content_hallucination_rate", r42["rulebook"][m], "EXP-42 §3"))
    for k, m in enumerate(BIG):
        out.append(nlp_paired(f"T4{'abcd'[k]}", f"{m}: predicted minus ground-truth vision", [cell(m, "real_predicted")],
                              [cell(m, "real_groundtruth")], "hallucination_rate", r42["vision"][m], "EXP-42 §3"))
    for k, m in enumerate((QWEN, GEMMA, MEDGEMMA, BASE, TUNED)):
        reported = (r42["vs_sonnet"][m]["hallucination_rate"] if m in BIG else
                    r48["tuned_minus" if m == TUNED else "base_minus"][SONNET]["all"]["hallucination_rate"])
        out.append(nlp_paired(f"T5{'abcde'[k]}", f"{m} minus Sonnet", head(m), head(SONNET), "hallucination_rate",
                              reported, "EXP-42 §3" if m in BIG else "EXP-48 §5"))
    routes = srr.scenario_routes()
    tb = r48["tuned_minus_base"]
    out.append(nlp_paired("T6a", "tuned minus base Gemma 3 4B, all 175 cases", head(TUNED), head(BASE),
                          "hallucination_rate", tb["all"]["hallucination_rate"], "EXP-48 §1"))
    out.append(nlp_paired("T6b", "tuned minus base, held-out-route scenarios", [cell(TUNED, "scenarios_rulebook")],
                          [cell(BASE, "scenarios_rulebook")], "hallucination_rate", tb["held_out"]["hallucination_rate"],
                          "EXP-48 §1", keep=srr._keep("held_out", routes)))
    out.append(nlp_paired("T6c", "tuned minus base, critical rate, all 175 cases", head(TUNED), head(BASE),
                          "critical_hallucination_rate", tb["all"]["critical_hallucination_rate"], "EXP-48 §2"))
    return out


def _welch_row(test_id, label, a, b, measure, existing=None):
    row = ttests.welch(a, b, label)
    row.update(id=test_id, measure=measure, existing=existing)
    row["agrees_with_existing"] = None
    return row


def heldout_rows(rates_json):
    """T7 (post hoc): development minus held-out per-case rates, Welch, per model. No reported interval to check."""
    held = json.loads(Path(rates_json).read_text())["rates"]
    out = []
    for k, m in enumerate(ALL_MODELS):
        if m not in held:
            continue
        dev = ttests.case_rates(cp.case_records(cell(m, "real_predicted")), "hallucination_rate")
        out.append(_welch_row(f"T7{'abcdef'[k]}", f"{m}: development minus held-out real cases (post hoc)", dev,
                              held[m], "per-case hallucination rate, predicted vision",
                              {"source": "EXP-66 realism check", "kind": "unpaired, by eye"}))
    return out


def subgroup_rows(vision_rows, notes_dir):
    """T8a-d (Vision A by age band and sex, development cases) and T8e-j (scenario hallucination by age band)."""
    from strokecopilot.nlp.verifier import scenarios
    from strokecopilot.pipeline import heldout_analysis as ha
    facts = ha.load_facts(notes_dir, [r["case"] for r in vision_rows])
    val = lambda f, k: None if f.get(k) is None or f.get(k).type == "not_recorded" else f.get(k).value
    age = {c: val(f, "age") for c, f in facts.items()}
    sex = {c: (None if val(f, "sex") is None else str(val(f, "sex")).lower()) for c, f in facts.items()}
    sexes = sorted({s for s in sex.values() if s is not None})
    if len(sexes) != 2:
        raise ValueError(f"expected two sex values in the notes, found {len(sexes)}")
    out = []
    for test_id, key, col in (("T8a", "Dice", "nnunet_dice"), ("T8b", "absolute volume error", "nnunet_abs_error_ml")):
        old = [float(r[col]) for r in vision_rows if age[r["case"]] is not None and age[r["case"]] >= 80]
        young = [float(r[col]) for r in vision_rows if age[r["case"]] is not None and age[r["case"]] < 80]
        out.append(_welch_row(test_id, f"Vision A {key}: under 80 minus 80 and over", young, old, f"per-case {key}"))
    missing = {"age": sum(v is None for v in age.values()), "sex": sum(v is None for v in sex.values())}
    for test_id, key, col in (("T8c", "Dice", "nnunet_dice"), ("T8d", "absolute volume error", "nnunet_abs_error_ml")):
        g = [[float(r[col]) for r in vision_rows if sex[r["case"]] == s] for s in sexes]
        out.append(_welch_row(test_id, f"Vision A {key}: {sexes[0]} minus {sexes[1]}", g[0], g[1], f"per-case {key}"))
    for r in out:
        r["not_recorded"] = missing["age" if r["id"] in ("T8a", "T8b") else "sex"]
    s_age = {s.id: s.facts.get("age").value for s in scenarios.generate()}
    for k, m in enumerate(ALL_MODELS):
        recs = cp.case_records(cell(m, "scenarios_rulebook"))
        rates = dict(zip([r["case_id"] for r in recs], ttests.case_rates(recs, "hallucination_rate")))
        young = [v for c, v in rates.items() if s_age[c] < 80]
        old = [v for c, v in rates.items() if s_age[c] >= 80]
        out.append(_welch_row(f"T8{'efghij'[k]}", f"{m}: scenarios, under 80 minus 80 and over", young, old,
                              "per-case hallucination rate"))
    return out


T9_FILE = Path("runs/EXP-82_biomedclip/t9_brier.json")


def t9_row(path=T9_FILE):
    """T9 (DEC-36): BiomedCLIP's linear probe minus EXP-6, per-patient mean Brier score of the calibrated scan
    probability on RSNA validation, paired (task D writes the two lists in the same patient order, without IDs)."""
    d = json.loads(Path(path).read_text())
    if len(d["probe"]) != len(d["exp6"]):
        raise ValueError("T9: the two Brier lists differ in length")
    row = ttests.paired(d["probe"], d["exp6"], "BiomedCLIP probe minus EXP-6 ResNet-50, Brier per patient")
    row.update(id="T9", measure="per-patient Brier score (calibrated scan probability)",
               existing={"source": "task D (EXP-82)", "kind": "AUCs, no paired interval"})
    row["agreement_with_existing"] = None
    row["agrees_with_existing"] = None
    return row


def run(drive, rates_json=None):
    """Every test that can run now, checked, then Holm over them as one family. Returns the rows."""
    drive = Path(drive)
    vrows_out, vrows = vision_a_rows(drive / "runs" / "EXP-27_visionA_error_followups" / "per_case.csv")
    rows = vrows_out + nlp_rows()
    if rates_json and Path(rates_json).exists():
        rows += heldout_rows(rates_json)
    rows += subgroup_rows(vrows, drive / "artifacts" / "isles24_notes_v1")
    if T9_FILE.exists():
        rows.append(t9_row())
    return ttests.adjust(rows)


def _fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}g}" if abs(x) < 1e-3 else f"{x:.{d}f}"


def markdown(rows, family_size, pending):
    pct = lambda r: "rate" in r["measure"]
    L = ["# EXP-69: Student's t-tests beside the existing results (DEC-36)", "",
         "The data had already been seen; the list was fixed before any t statistic (docs/PHASE5_TTESTS.md). The case is "
         f"the unit. Holm's correction over the {len(rows)} tests run here, of a planned family of {family_size}"
         f"{' (pending: ' + ', '.join(pending) + ')' if pending else ''}. Rates as proportions (0.10 = 10 points).", "",
         "| Test | Comparison | n | Mean difference [95% t CI] | t (df) | p | p (Holm) | Effect | Rank test p | Skew | "
         "Existing result | Agree? |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n = r["n"] if isinstance(r["n"], int) else f"{r['n'][0]} / {r['n'][1]}"
        sk = r["assumptions"].get("skewness") if r["form"] == "paired" else max(
            abs(r["assumptions"]["a"]["skewness"] or 0), abs(r["assumptions"]["b"]["skewness"] or 0))
        ex = r.get("existing") or {}
        exs = (f"{_fmt(ex['pooled_difference'])} [{_fmt(ex['ci95'][0])}, {_fmt(ex['ci95'][1])}]" if ex.get("ci95")
               else ex.get("kind", "—"))
        agree = {True: "yes", False: "**no**", None: "—"}[r["agrees_with_existing"]]
        L.append(f"| {r['id']} | {r['label']} ({r['form']}) | {n} | {_fmt(r['mean_difference'])} "
                 f"[{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}] | {_fmt(r['t'], 2)} ({_fmt(r['df'], 1) if r['form'] == 'Welch' else r['df']}) "
                 f"| {_fmt(r['p'])} | {_fmt(r['p_holm'])} | {r['effect']} {_fmt(r['effect_size'], 2)} "
                 f"| {_fmt(r['rank_test']['p'])} | {_fmt(sk, 2)} | {exs} | {agree} |")
    L += ["", "Paired: Student's paired t-test with Wilcoxon signed-rank beside it, effect d_z. Unpaired: Welch's t-test "
          "with Mann-Whitney U beside it, effect Hedges' g. Skew: of the differences (paired) or the larger of the two "
          "groups (Welch). 'Agree?' asks whether the t interval and the existing bootstrap interval agree on excluding 0; "
          "the existing intervals are of pooled claim rates, the t-tests of mean per-case rates. T7 is post hoc on saved "
          "held-out results."]
    return "\n".join(L) + "\n"


FAMILY = 35  # docs/PHASE5_TTESTS.md: T1-T9


def qq_figure(rows, path):
    """Q-Q plots of the paired differences or each Welch group, one small panel per test (the doc's appendix)."""
    from scipy import stats

    from strokecopilot.pipeline.figures import INK_2, SLOTS, _plt
    plt = _plt()
    n = len(rows)
    cols = 6
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(13, 2.1 * ((n + cols - 1) // cols)))
    for ax, r in zip(axes.flat, rows):
        for k, x in enumerate(r["_values"]):
            x = np.asarray(x)[np.isfinite(x)]
            if len(x) >= 3:
                (osm, osr), (slope, icpt, _) = stats.probplot(x)
                ax.plot(osm, osr, "o", ms=2.5, color=SLOTS[k], alpha=0.8)
                ax.plot(osm, slope * osm + icpt, lw=0.8, color=INK_2)
        ax.set_title(r["id"], fontsize=8, loc="left")
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    import argparse
    import subprocess

    from strokecopilot.config import local_drive_root
    ap = argparse.ArgumentParser(description="DEC-36: the approved t-tests that can run now")
    ap.add_argument("--out", default="runs/EXP-69_ttests")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    rows = run(local_drive_root(), out / "heldout_rates.json")
    done = {r["id"][:2] for r in rows}
    pending = [t for t in ("T7", "T9") if t not in done]
    qq_figure(rows, out / "qq_plots.png")
    (out / "report.md").write_text(markdown(rows, FAMILY, pending))
    clean = [{k: v for k, v in r.items() if k != "_values"} for r in rows]
    (out / "metrics.json").write_text(json.dumps({
        "experiment": "EXP-69", "tests_run": len(rows), "family_planned": FAMILY, "pending": pending,
        "holm_over": "the tests run here", "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                                          text=True).strip(),
        "rows": clean}, indent=1, default=float))
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
