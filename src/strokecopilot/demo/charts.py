"""The Space's chart data (DEC-32 §3): numbers only, gathered from the finished runs' reports, so the page draws the
results without reading any run folder. Every figure here is already in the CHANGELOG and plan; nothing is computed
anew except sums of recorded counts.

    charts(sources) -> dict, written as charts.json beside the bank.
"""

import json
from pathlib import Path

MODEL_ORDER = ("claude-sonnet-5-5", "qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text", "gemma-3-4b", "gemma-3-4b-tuned")
LABELS = {"claude-sonnet-5-5": "Sonnet 5.5", "qwen3.5-27b": "Qwen3.5-27B", "gemma-3-27b": "Gemma 3 27B",
          "medgemma-27b-text": "MedGemma 27B", "gemma-3-4b": "Gemma 3 4B", "gemma-3-4b-tuned": "Gemma 3 4B tuned"}
RATES = ("hallucination_rate", "content_hallucination_rate", "critical_hallucination_rate")


def _read(path):
    return json.loads(Path(path).expanduser().read_text())


def _rate(r):
    return {"value": r["value"], "ci95": r["ci95"], "numerator": r["numerator"], "denominator": r["denominator"]}


def default_sources(drive_runs, repo_runs="runs"):
    """Where each finished report lives: the Mac's runs/ for most, Drive for EXP-50 (its per-case table holds IDs)."""
    r = Path(repo_runs)
    return {"exp48": r / "EXP-48_small_reasoner_report/rescore_points_19_20/report.json",
            "exp42": r / "EXP-42_c9_table/rescore_points_19_20/report.json",
            "exp41": r / "EXP-41_labelling/labels/agreement_final.json",
            "exp49": r / "EXP-49_visionB_false_alarms/metrics.json",
            "exp50": Path(drive_runs) / "EXP-50_visionB_isles_false_alarms" / "metrics.json",
            "exp60": r / "EXP-60_visionB_rsna_test/metrics.json",
            "exp63": r / "EXP-63_d3_dev_check/report.json"}


def charts(sources):
    """sources: {"exp48": report.json (rescored), "exp42": report.json (rescored), "exp41": agreement_final.json,
    "exp49": metrics.json, "exp50": metrics.json, "exp60": metrics.json, "exp63": report.json}."""
    small, c9 = _read(sources["exp48"]), _read(sources["exp42"])
    out = {"models": [{"id": m, "label": LABELS[m], "slot": i + 1} for i, m in enumerate(MODEL_ORDER)]}

    # C9: every model on the same 175 cases (5 fixtures, 150 scenarios, 20 real), rulebook arm, points 1-20.
    out["c9"] = {"cases": 175, "source": "EXP-42 and EXP-48, rescored with metric points 19-20 (verifier rates)",
                 "rows": [{"model": m, **{k: _rate(small["summary"][m]["all"][k]) for k in RATES},
                           "correctness": _rate(small["summary"][m]["all"]["recommendation_correctness"])}
                          for m in MODEL_ORDER]}

    # The rulebook effect (DEC-29): content rate with and without the rulebook on the same 100 cases, paired.
    out["rulebook"] = {"source": "EXP-42 (DEC-29), paired on the same 100 cases (5 fixtures, 75 scenarios, 20 real)",
                       "rows": [{"model": m, "with": r["a"], "without": r["b"], "difference": r["difference"],
                                 "ci95": r["ci95"], "cases": r["cases"]}
                                for m, r in c9["rulebook"].items()]}

    # The human check (EXP-41).
    a = _read(sources["exp41"])["summary"]
    out["human"] = {"source": "EXP-41: 100 claims, four labellers in two pairs and a third rater",
                    "claims": a["n"], "kappa": a["kappa_pooled"], "agreement": a["agreement_rescored"],
                    "agreement_weighted": a["agreement_rescored_weighted"],
                    "flags_confirmed": a["verifier_flags_confirmed"], "human_errors_flagged": a["human_errors_flagged"],
                    "by_verdict": [{"verdict": k, **v} for k, v in a["by_verdict_rescored"].items()]}

    # DEC-30: the tuned 4B against its base. Thrombectomy "yes" caught over the 155 fixtures and scenarios.
    def caught(model, decision):
        groups = small["decisions"][model]
        return {k: sum(groups[g][decision][k] for g in ("held_out", "seen", "fixtures")) for k in ("yes_caught", "yes")}

    out["small"] = {"source": "EXP-47 and EXP-48 (DEC-30); held-out routes never seen in training",
                    "rows": [{"model": m, "held_out_rate": _rate(small["summary"][m]["held_out"]["hallucination_rate"]),
                              "all_rate": _rate(small["summary"][m]["all"]["hallucination_rate"]),
                              "thrombectomy": caught(m, "thrombectomy"), "thrombolysis": caught(m, "thrombolysis")}
                             for m in ("gemma-3-4b", "gemma-3-4b-tuned")]}

    # Item 2 and Vision B.
    e50, e49, e60 = _read(sources["exp50"]), _read(sources["exp49"]), _read(sources["exp60"])
    out["item2"] = {"source": "EXP-49, EXP-50 (pre-registered in docs/ITEM2_FALSE_ALARMS.md)", "scans": e50["cases"],
                    "flags_none": e50["candidates"]["none"]["flagged"], "proposal": e50["proposal"],
                    "flags_proposal": e50["candidates"][e50["proposal"]]["flagged"],
                    "heat_in_brain": e50["gradcam_flagged"]["brain"],
                    "rsna_specificity_none": e49["selection"]["candidates"]["none"]["specificity"]}
    test = e60["test"]
    out["vision_b"] = {"source": "EXP-60: EXP-6 at DEC-28's 0.175 on the RSNA test split, opened once",
                       "scans": test["scans"], "patients": test["patients"],
                       **{k: test["preregistered"][k] for k in ("scan_auc", "scan_auc_ci95", "sensitivity", "sensitivity_ci95",
                                                                "specificity", "specificity_ci95", "scan_ece")},
                       "subtypes": [{"subtype": k, "caught": v["caught"], "bleeds": v["bleeds"]}
                                    for k, v in test["preregistered"]["subtype_sensitivity"].items()],
                       "reliability": [{k: r[k] for k in ("count", "mean_prob", "observed")} for r in test["extras"]["reliability"]],
                       "isles_flags": e50["candidates"]["none"]["flagged"], "isles_scans": e50["cases"]}
    e63 = _read(sources["exp63"])
    out["vision_a"] = {"source": "EXP-25 / EXP-63: nnU-Net on the 129 development cases, each by the fold model that did not "
                                 "train on it", **e63["vision"]}
    out["scenario_sensitivity"] = {k: e63["scenario_sensitivity"][k] for k in ("x0.8", "x1.2")}
    return out


def heldout(report):
    """DEC-32 §3's "after Phase 4" results from EXP-66's report (aliases only, no IDs): correctness against the rules on
    the ground-truth contract, error propagation, the ±20% core check, held-out vision, and the realism check."""
    def pooled(c):
        return {"correct": c["pooled"]["correct"], "scored": c["pooled"]["scored"], "rate": c["pooled"]["rate"],
                "ci95": c["pooled"]["ci95"], "no_answer": c["no_answer"]}

    rows = []
    for m in MODEL_ORDER:
        r = report["models"].get(m)
        if r is None:
            continue
        ep = r["predicted"]["error_propagation"]
        rows.append({"model": m, "predicted": pooled(r["predicted"]["correctness"]),
                     "groundtruth": pooled(r["groundtruth"]["correctness"]) if "groundtruth" in r else None,
                     "errors": ep["table"],
                     "examples": [{k: e[k] for k in ("alias", "decision", "cause", "caught")} for e in ep.get("examples", [])],
                     "rate_heldout": _rate(r["c9"]["heldout"]["hallucination_rate"]),
                     "rate_development": _rate(r["c9"]["development"]["hallucination_rate"]) if r["c9"].get("development") else None,
                     "format_failures": r["c9"]["heldout"]["format_failure_rate"]["numerator"]})
    rules = report["rules_on_predicted_correct"]["pooled"]
    return {"source": f"{report['experiment']}: DEC-31 D3 on the {report['cases']} held-out patients, opened once after "
                      "every model and threshold was frozen",
            "cases": report["cases"], "truth": report["truth"],
            "rules_on_predicted": {k: rules[k] for k in ("correct", "scored", "rate", "ci95")},
            "rows": rows, "rule_sensitivity": {k: report["rule_sensitivity"][k] for k in ("x0.8", "x1.2")},
            "sonnet_sensitivity": report.get("sonnet_sensitivity"), "vision": report["vision"]}
