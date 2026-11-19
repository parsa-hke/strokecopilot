"""The fine-tuned small reasoner's evaluation (DEC-30, Item 4 step 7): Gemma 3 4B prompted against Gemma 3 4B tuned,
with the EXP-37 models alongside.

Reads saved cells only: the small pair under runs/EXP-47_small_reasoner/<model>/<arm>/ and the EXP-37 models under
runs/EXP-37_paid_runs/<model>/<arm>/, all with prompt v3, the rulebook and predicted vision. Case groups:

- held_out: the evaluation scenarios that take at least one of the 4 routes left out of training (51 of 150). DEC-30's
  main result: does the tuned model carry the rulebook to routes it never saw?
- seen: the other 99 scenarios. Their routes were in training (with different values: TRAIN_GRIDS share no decision
  value with the evaluation grids).
- fixtures (5) and real (20 development cases): neither group was used in training, and their routes are not labelled.
- all: the 175 together, as in C9's headline.

A route is held out when one of the scenario's two expected reasons starts with it (scenarios.routes' own rule).

Paired comparisons (compare.paired: a paired case bootstrap on the cases both sides hold): tuned minus base in every
group, and tuned minus each EXP-37 model on the scenario groups and on all. The rates are the verifier's, not yet
checked against the human labels (EXP-41). The gold answers the tuned model learned from are templated, so its claims
may be more formulaic and easier to verify: a lower hallucination rate is reported with that limit (DEC-30).
"""

import json
from pathlib import Path

from strokecopilot.nlp.verifier import scenarios as sc
from strokecopilot.pipeline import compare as cp

BASE, TUNED = "gemma-3-4b", "gemma-3-4b-tuned"
COMPARATORS = ("claude-sonnet-5-5", "qwen3.5-27b", "gemma-3-27b", "medgemma-27b-text")  # EXP-37
GROUPS = ("held_out", "seen", "fixtures", "real", "all")
GROUP_LABELS = {"held_out": "Scenarios, held-out routes", "seen": "Scenarios, seen routes", "fixtures": "Fixtures",
                "real": "Real cases (development, predicted vision)", "all": "All 175"}
SUMMARY_METRICS = ("hallucination_rate", "content_hallucination_rate", "critical_hallucination_rate",
                   "recommendation_correctness", "format_failure_rate", "omission_rate")
PAIRED_METRICS = ("hallucination_rate", "content_hallucination_rate", "critical_hallucination_rate",
                  "recommendation_correctness")
VS_METRICS = ("hallucination_rate", "recommendation_correctness")
TUNED_COLOUR, BASE_COLOUR = cp.SERIES["with"], cp.SERIES["without"]  # categorical slots 1 (blue) and 2 (orange)


def scenario_routes(held_out=None):
    """{scenario id: {"group": "held_out" | "seen", "routes": [key routes it takes]}} for the 150 evaluation scenarios."""
    held = sc.held_out_routes() if held_out is None else list(held_out)
    out = {}
    for s in sc.generate():
        reasons = [s.expected[d]["reason"] for d in s.expected]
        taken = [r for r in sc.KEY_ROUTES if any(x.startswith(r) for x in reasons)]
        out[s.id] = {"group": "held_out" if any(r in held for r in taken) else "seen", "routes": taken}
    return out


def _cells(root, model, group):
    cell = lambda arm: str(Path(root) / model.replace("/", "_") / arm)
    return {"held_out": [cell("scenarios_rulebook")], "seen": [cell("scenarios_rulebook")],
            "fixtures": [cell("fixtures_rulebook")], "real": [cell("real_predicted")],
            "all": [cell(a) for a in cp.HEADLINE_ARMS]}[group]


def _keep(group, routes):
    if group in ("held_out", "seen"):
        return lambda case_id: routes.get(case_id, {}).get("group") == group
    return None


def report(small_root, paid_root, out_dir, comparators=COMPARATORS, held_out=None):
    """Write report.json, report.md and small_reasoner.png to out_dir; returns the report."""
    routes = scenario_routes(held_out)
    root_of = lambda m: small_root if m in (BASE, TUNED) else paid_root
    models = [BASE, TUNED, *comparators]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"small_root": str(small_root), "paid_root": str(paid_root), "models": models,
           "held_out_routes": sc.held_out_routes() if held_out is None else list(held_out),
           "group_sizes": {g: sum(r["group"] == g for r in routes.values()) for g in ("held_out", "seen")},
           "summary": {}, "tuned_minus_base": {}, "tuned_minus": {}, "base_minus": {}, "per_route": {}}
    for m in models:
        rep["summary"][m] = {g: {k: cp._pooled(_cells(root_of(m), m, g), k, _keep(g, routes)) for k in SUMMARY_METRICS}
                             for g in GROUPS}
    pair = lambda a, b, g, k: cp.paired(_cells(root_of(a), a, g), _cells(root_of(b), b, g), k, keep=_keep(g, routes))
    rep["tuned_minus_base"] = {g: {k: pair(TUNED, BASE, g, k) for k in PAIRED_METRICS} for g in GROUPS}
    for c in comparators:
        rep["tuned_minus"][c] = {g: {k: pair(TUNED, c, g, k) for k in VS_METRICS} for g in ("held_out", "seen", "all")}
        rep["base_minus"][c] = {g: {k: pair(BASE, c, g, k) for k in VS_METRICS} for g in ("held_out", "seen", "all")}
    for route in sc.KEY_ROUTES:  # small groups: counts, no intervals below metrics.MIN_CASES_FOR_CI cases
        ids = {cid for cid, r in routes.items() if route in r["routes"]}
        keep = lambda case_id, ids=ids: case_id in ids
        rep["per_route"][route] = {"held_out": route in rep["held_out_routes"], "scenarios": len(ids),
                                   **{m: {k: cp._pooled(_cells(root_of(m), m, "seen"), k, keep)
                                          for k in ("recommendation_correctness", "hallucination_rate")}
                                      for m in (BASE, TUNED)}}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    (out / "report.md").write_text(_markdown(rep), encoding="utf-8")
    _plot(rep, out / "small_reasoner.png")
    return rep


def _count(m):
    return f"{m['numerator']} / {m['denominator']}"


def _markdown(rep):
    name = lambda m: cp.DISPLAY.get(m, m)
    s = rep["summary"]
    L = ["# The fine-tuned small reasoner (DEC-30): Gemma 3 4B prompted and tuned", "",
         f"Held-out routes (never in training): {', '.join(rep['held_out_routes'])}. Scenarios on them: "
         f"{rep['group_sizes']['held_out']}; on seen routes only: {rep['group_sizes']['seen']}. Verifier rates, not yet "
         "checked against the human labels (EXP-41). Intervals are 95% case bootstraps (10,000 resamples); a paired "
         "difference is a paired case bootstrap on the same cases. The gold the tuned model learned from is templated, "
         "so its claims may be easier to verify (DEC-30's stated limit).", "",
         "## 1. Tuned against base, by case group", "",
         "| Group | Cases | Hallucination rate, base | Tuned | Tuned minus base | Decisions right, base | Tuned | Tuned minus base |",
         "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        b, t, d = s[BASE][g], s[TUNED][g], rep["tuned_minus_base"][g]
        L.append(f"| {GROUP_LABELS[g]} | {t['hallucination_rate']['cases']} | {cp._pct(b['hallucination_rate'])} | "
                 f"{cp._pct(t['hallucination_rate'])} | {cp._diff(d['hallucination_rate'])} | "
                 f"{_count(b['recommendation_correctness'])} | {_count(t['recommendation_correctness'])} | "
                 f"{cp._diff(d['recommendation_correctness'])} |")
    L += ["", "## 2. Content and critical errors, tuned minus base", "",
          "| Group | Content rate (sub-types 1–5), base | Tuned | Difference | Critical rate, base | Tuned | Difference |",
          "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        b, t, d = s[BASE][g], s[TUNED][g], rep["tuned_minus_base"][g]
        L.append(f"| {GROUP_LABELS[g]} | {cp._pct(b['content_hallucination_rate'])} | "
                 f"{cp._pct(t['content_hallucination_rate'])} | {cp._diff(d['content_hallucination_rate'])} | "
                 f"{cp._pct(b['critical_hallucination_rate'])} | {cp._pct(t['critical_hallucination_rate'])} | "
                 f"{cp._diff(d['critical_hallucination_rate'])} |")
    L += ["", "## 3. All models, hallucination rate by group (with the rulebook, predicted vision)", "",
          "| Model | " + " | ".join(GROUP_LABELS[g] for g in GROUPS) + " | Format failures (all) |",
          "|---|" + "---|" * (len(GROUPS) + 1)]
    for m in rep["models"]:
        L.append(f"| {name(m)} | " + " | ".join(cp._pct(s[m][g]["hallucination_rate"]) for g in GROUPS)
                 + f" | {_count(s[m]['all']['format_failure_rate'])} |")
    L += ["", "## 4. The small pair against each EXP-37 model (paired, same cases)", "",
          "| Against | Tuned: held-out | Tuned: seen | Tuned: all | Base: held-out | Base: seen | Base: all |",
          "|---|---|---|---|---|---|---|"]
    for c in rep["tuned_minus"]:
        t, b = rep["tuned_minus"][c], rep["base_minus"][c]
        L.append(f"| {name(c)} | " + " | ".join(cp._diff(x[g]["hallucination_rate"]) for x in (t, b)
                                                  for g in ("held_out", "seen", "all")) + " |")
    L += ["", "Hallucination rate minus the other model, percentage points; negative means fewer hallucinations than it.",
          "", "## 5. Per key route (scenarios; small groups, counts only)", "",
          "| Route | Held out | Scenarios | Decisions right, base | Tuned | Hallucination rate, base | Tuned |",
          "|---|---|---|---|---|---|---|"]
    for route, r in rep["per_route"].items():
        if not r["scenarios"]:
            continue
        b, t = r[BASE], r[TUNED]
        L.append(f"| {route} | {'yes' if r['held_out'] else ''} | {r['scenarios']} | "
                 f"{_count(b['recommendation_correctness'])} | {_count(t['recommendation_correctness'])} | "
                 f"{cp._pct(b['hallucination_rate'], 0)} | {cp._pct(t['hallucination_rate'], 0)} |")
    return "\n".join(L) + "\n"


def _plot(rep, path):
    """Two panels, by case group: hallucination rate and decisions right, base and tuned, with 95% intervals."""
    import matplotlib.pyplot as plt

    groups = ("held_out", "seen", "real")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=cp.SURFACE)
    panels = (("hallucination_rate", "Hallucination rate (%)"), ("recommendation_correctness", "Decisions right (%)"))
    for ax, (metric, label) in zip(axes, panels):
        cp._style(ax)
        for i, g in enumerate(groups):
            y = len(groups) - 1 - i
            for model, colour, offset, legend in ((TUNED, TUNED_COLOUR, 0.13, "Gemma 3 4B tuned"),
                                                  (BASE, BASE_COLOUR, -0.13, "Gemma 3 4B prompted")):
                m = rep["summary"][model][g][metric]
                if m["value"] is None:
                    continue
                if m["ci95"]:
                    ax.plot([100 * m["ci95"][0], 100 * m["ci95"][1]], [y + offset] * 2, color=colour, linewidth=2, zorder=2)
                ax.plot([100 * m["value"]], [y + offset], "o", color=colour, markersize=8, zorder=3,
                        markeredgecolor=cp.SURFACE, markeredgewidth=2, label=legend if i == 0 else None)
        ax.set_yticks(range(len(groups)))
        sizes = {g: rep["summary"][TUNED][g]["hallucination_rate"]["cases"] for g in groups}
        ax.set_yticklabels([f"{GROUP_LABELS[g].split(' (')[0]} ({sizes[g]})" for g in reversed(groups)],
                           color=cp.INK, fontsize=10)
        ax.set_ylim(-0.6, len(groups) - 0.4)
        ax.set_xlabel(f"{label}, 95% intervals", color=cp.INK_2)
        if metric == "recommendation_correctness":
            ax.set_xlim(-3, 103)  # room for a dot at 0 or 100
        else:
            ax.set_xlim(left=0)
    axes[1].set_yticklabels([])
    axes[0].set_title("Gemma 3 4B, prompted and tuned (DEC-30)", color=cp.INK, fontsize=12, loc="left", pad=28)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10, labelcolor=cp.INK,
                   handletextpad=0.3, columnspacing=1.5, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=cp.SURFACE)
    plt.close(fig)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="DEC-30's report: the small reasoner, prompted and tuned.")
    parser.add_argument("--small", default="runs/EXP-47_small_reasoner")
    parser.add_argument("--paid", default="runs/EXP-37_paid_runs")
    parser.add_argument("--out", required=True, help="e.g. runs/EXP-48_small_reasoner_report")
    args = parser.parse_args(argv)
    rep = report(args.small, args.paid, args.out)
    print((Path(args.out) / "report.md").read_text())
    return rep


if __name__ == "__main__":
    main()
