"""SHAP survey row 8 (DEC-35): exact Shapley values of the rule engine's inputs, on the synthetic scenarios.

The game, per scenario and per treatment:
- players: 13 input groups (PLAYERS), covering every field of `rules.Case` exactly once;
- an absent player: its fields unknown (None), which is what "not recorded" or a null contract field means to the
  rules. `wake_up` is a flag the rules read only with a time, so it is False when time is absent. The mismatch ratio
  and volume are derived from core and Tmax > 6 s, so they keep their full-input values only when both are present;
- the value of a coalition: 1 if `rules.decide` on it gives the same decision as on the full inputs, else 0.

So a player's value is its share in reaching this decision, and the values sum to 1 minus "the decision with every
input unknown is already this one". "Vision as a whole" is the same game with two players: all vision fields against
all note fields. The scenarios cover the decision routes evenly by design, so the results describe our scenario set,
not the mix of patients in practice.
"""

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from strokecopilot.analysis.shapley import shapley
from strokecopilot.nlp.verifier import rules

DECISIONS = ("thrombolysis", "thrombectomy")

PLAYERS = {
    "bleed": ("haemorrhage",),
    "core": ("core",),
    "tmax6": ("penumbra",),
    "occlusion site": ("site",),
    "ASPECTS": ("aspects",),
    "time": ("minutes", "time_basis", "wake_up", "sleep_mid_minutes"),
    "age": ("age",),
    "NIHSS": ("nihss",),
    "pre-stroke mRS": ("mrs",),
    "INR": ("inr",),
    "anticoagulant": ("anticoagulant",),
    "head trauma or surgery": ("head_trauma", "recent_surgery"),
    "previous or recent bleeding": ("previous_ich", "recent_bleeding"),
}
DERIVED = ("ratio", "mismatch")  # present only when core and tmax6 both are
VISION = ("bleed", "core", "tmax6", "occlusion site", "ASPECTS")
ABSENT = {"wake_up": False}


def check_players():
    """Every Case field belongs to exactly one player, or is derived."""
    fields = [f.name for f in dataclasses.fields(rules.Case)]
    owned = [f for group in PLAYERS.values() for f in group] + list(DERIVED)
    if sorted(owned) != sorted(fields):
        raise ValueError(f"players do not cover rules.Case exactly: {sorted(set(fields) ^ set(owned))}")


def masked(case, present):
    """The Case with only the players in `present` (a set of names) kept; others unknown."""
    changes = {}
    for name, group in PLAYERS.items():
        if name not in present:
            changes.update({f: ABSENT.get(f) for f in group})
    if not {"core", "tmax6"} <= present:
        changes.update({f: None for f in DERIVED})
    return dataclasses.replace(case, **changes)


def _game(case, groups):
    """Value function over `groups` (a list of lists of player names) for both treatments at once."""
    full = rules.decide(case)
    target = np.array([full[d].value for d in DECISIONS])

    def value(mask):
        present = {p for i, g in enumerate(groups) if mask >> i & 1 for p in g}
        got = rules.decide(masked(case, present))
        return (np.array([got[d].value for d in DECISIONS]) == target).astype(float)
    return value, {d: full[d].value for d in DECISIONS}


def explain_case(case):
    """{"decisions", "players": {decision: {player: phi}}, "vision_vs_note": {decision: {"vision", "note"}},
    "baseline_same": {decision: bool}} for one Case."""
    names = list(PLAYERS)
    value, decided = _game(case, [[p] for p in names])
    phi, _, v_none = shapley(len(names), value)
    value2, _ = _game(case, [list(VISION), [p for p in names if p not in VISION]])
    phi2, _, _ = shapley(2, value2)
    return {"decisions": decided,
            "players": {d: {p: float(phi[i, k]) for i, p in enumerate(names)} for k, d in enumerate(DECISIONS)},
            "vision_vs_note": {d: {"vision": float(phi2[0, k]), "note": float(phi2[1, k])}
                               for k, d in enumerate(DECISIONS)},
            "baseline_same": {d: bool(v_none[k]) for k, d in enumerate(DECISIONS)}}


def summarise(rows):
    """Mean Shapley value per player and treatment (overall and by decision), and vision against note."""
    out = {}
    for d in DECISIONS:
        by_value = defaultdict(list)
        for r in rows:
            by_value[r["decisions"][d]].append(r)
        groups = {"all": rows, **{f"decision {k}": v for k, v in sorted(by_value.items())}}
        out[d] = {g: {"cases": len(rs),
                      "baseline_already_this_decision": sum(r["baseline_same"][d] for r in rs),
                      "mean_phi": {p: float(np.mean([r["players"][d][p] for r in rs])) for p in PLAYERS},
                      "mean_abs_phi": {p: float(np.mean([abs(r["players"][d][p]) for r in rs])) for p in PLAYERS},
                      "vision_vs_note": {s: float(np.mean([r["vision_vs_note"][d][s] for r in rs]))
                                         for s in ("vision", "note")}}
                  for g, rs in groups.items()}
    return out


def run(scenarios):
    """Rows (one per scenario, by its own synthetic ID) and the summary."""
    check_players()
    rows = []
    for s in scenarios:
        row = explain_case(rules.case_from(s.vision, s.facts))
        row["scenario"] = s.id
        row["route"] = {d: s.expected[d]["reason"] for d in DECISIONS}
        rows.append(row)
    return rows, summarise(rows)


def main(argv=None):
    import argparse

    from strokecopilot.nlp.verifier import scenarios as sc
    ap = argparse.ArgumentParser(description="SHAP row 8: the rule engine's inputs, exact Shapley, 150 scenarios")
    ap.add_argument("--out", required=True, help="a runs/EXP-*_rule_shap folder, which must hold config.json")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    rows, summary = run(sc.generate())
    (out / "per_scenario.json").write_text(json.dumps(rows, indent=1))
    (out / "metrics.json").write_text(json.dumps({"scenarios": len(rows), "players": list(PLAYERS),
                                                  "summary": summary}, indent=1))
    print(json.dumps({d: summary[d]["all"] for d in DECISIONS}, indent=1))


if __name__ == "__main__":
    main()
