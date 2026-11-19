"""SHAP survey row 9 (DEC-35): which inputs the language models rely on, next to the rule engine on the same inputs.

The game, per scenario, model and treatment:
- **Players, six groups** (amended 20 Nov 2026, Parsa: the bleed stays fixed at its real value, because the frozen
  contract cannot say "unknown" for `haemorrhage.present`; it was seven groups):
  perfusion (core and Tmax > 6 s volumes, with ratio and mismatch), occlusion site, ASPECTS, time (onset or last known
  well), patient (age, NIHSS, pre-stroke mRS), contraindication facts (anticoagulant, INR, head trauma, surgery,
  previous or recent bleeding).
- **An absent group:** its contract fields null (occlusion site "unknown", a valid contract value) and its note facts
  written as not recorded. Every version of a note, the full one included, is rendered from the scenario's spec with
  the first phrasing of every choice, so wording does not change between versions; only the hidden facts do.
- **Value of a coalition:** 1 if the answer equals the answer on the full inputs, else 0. The same for the rules
  (`rules.decide` on the same version's contract and fact sheet), so the two are compared like for like.
- **Cases:** 24 synthetic scenarios: one per key route (16, by seed), then a second for the first 8 routes in order.
- **Answers:** Gemma 3 4B and the tuned 4B on vLLM (notebook 55, `MODE = 'shap'`), prompt v3 with the rulebook, as in
  EXP-47; 24 x 64 = 1,536 requests per model. A version whose answer fails the format leaves that scenario's game
  incomplete; it is reported apart and left out of the summary. Synthetic data only: no real or held-out case.
"""

import dataclasses
import json
import random
from pathlib import Path

import numpy as np

from strokecopilot.analysis.shapley import shapley_from_values
from strokecopilot.nlp.notes import render as nr
from strokecopilot.nlp.notes.facts import Fact, FactSheet
from strokecopilot.nlp.verifier import rules
from strokecopilot.nlp.verifier import scenarios as sc

SEED = 20261120
GROUPS = ("perfusion", "occlusion site", "ASPECTS", "time", "patient", "contraindications")
N_CASES, FIRST = 24, 16
DECISIONS = ("thrombolysis", "thrombectomy")
PATIENT_KEYS = ("age", "nihss", "premorbid_mrs")
CONTRA_SPEC = ("anticoagulant", "antiplatelet", "inr", "head_trauma", "recent_surgery", "previous_ich", "recent_bleeding")
CONTRA_NOT_RECORDED = ("anticoagulant", "inr", "head_trauma", "recent_surgery", "previous_ich", "recent_bleeding")
LABEL = {"age": "Age", "nihss": "NIHSS", "premorbid_mrs": "Pre-stroke mRS"}


class _First:
    """An rng for render() that always takes the first phrasing, so every version of a note is worded alike."""

    def randrange(self, n):
        return 0


def scenarios_with_specs():
    """The 150 scenarios and, for each, the note spec it was rendered from (captured from `scenarios.generate`)."""
    specs = {}
    real = sc.render

    def keep(spec, rng):
        note, facts = real(spec, rng)
        specs[note] = dict(spec)
        return note, facts

    sc.render = keep
    try:
        chosen = sc.generate()
    finally:
        sc.render = real
    return [(s, specs[s.note]) for s in chosen]


def choose(pairs, seed=SEED):
    """24 (scenario, spec) pairs: one per key route by seed, then a second for the first 8 routes."""
    rng = random.Random(seed)
    taken = {s.id: [r for r in sc.KEY_ROUTES if any(v["reason"].startswith(r) for v in s.expected.values())]
             for s, _ in pairs}
    by_id = {s.id: (s, spec) for s, spec in pairs}
    chosen = []
    for rounds in (sc.KEY_ROUTES, sc.KEY_ROUTES[: N_CASES - FIRST]):
        for route in rounds:
            pool = sorted(i for i, rs in taken.items() if route in rs and i not in chosen)
            if pool:
                chosen.append(rng.choice(pool))
    return [by_id[i] for i in chosen[:N_CASES]]


def masked_vision(vision, present):
    v = json.loads(json.dumps(vision))
    if "perfusion" not in present:
        v.update(core_ml=None, penumbra_ml=None, mismatch_ratio=None, mismatch_volume_ml=None)
    if "occlusion site" not in present:
        v["occlusion_site"] = "unknown"
        v["warnings"] = [w for w in v.get("warnings", []) if w != "CTA not acquired"]
    if "ASPECTS" not in present:
        v["aspects"] = None
    return v


def masked_note(spec, present):
    """(note, FactSheet) for a version of the note with the absent groups not recorded, first phrasing throughout."""
    s = dict(spec)
    nr_keys = list(s.get("not_recorded", []))
    if "time" not in present:
        s["minutes"] = None
    if "patient" not in present:
        s["mrs"] = None
    if "contraindications" not in present:
        for k in CONTRA_SPEC:
            s[k] = None
        nr_keys += [k for k in CONTRA_NOT_RECORDED if k not in nr_keys]
    s["not_recorded"] = nr_keys
    note, sheet = nr.render(s, _First())
    if "patient" in present:
        return note, sheet
    # render() always writes age and NIHSS: rewrite their spans as not recorded, and add pre-stroke mRS the same way
    by_key = dict(sheet.facts)
    noun = by_key["sex"].span
    age_old = f"{by_key['age'].span} {noun}."
    age_new = f"Adult {noun}; age not recorded."
    nihss_new = "NIHSS: not recorded."
    mrs_new = "Pre-stroke mRS: not recorded."
    note = note.replace(age_old, age_new, 1).replace(by_key["nihss"].span, nihss_new, 1) + "\n" + mrs_new
    gone = {"age": "age not recorded", "nihss": nihss_new, "premorbid_mrs": mrs_new}
    facts = [f for f in sheet if f.key not in gone]
    facts += [Fact(key=k, type="not_recorded", value=None, span=span) for k, span in gone.items()]
    sheet = FactSheet(facts)
    sheet.check_against(note)
    return note, sheet


def versions(scenario, spec):
    """All 64 versions: [{"mask", "vision", "note", "facts"}], mask bit i = GROUPS[i] present."""
    out = []
    for mask in range(1 << len(GROUPS)):
        present = {g for i, g in enumerate(GROUPS) if mask >> i & 1}
        vision = masked_vision(scenario.vision, present)
        note, facts = masked_note(spec, present)
        out.append({"mask": mask, "vision": vision, "note": note, "facts": facts})
    return out


def rule_values(vs):
    """(64, 2) table: 1 where rules.decide on the version gives the full version's decision."""
    decided = [rules.decide(rules.case_from(v["vision"], v["facts"])) for v in vs]
    full = decided[-1]
    return np.array([[float(d[k].value == full[k].value) for k in DECISIONS] for d in decided]), {
        k: full[k].value for k in DECISIONS}


def model_values(recs):
    """(64, 2) table from the model's recommendations (None for a format failure, which makes the game incomplete)."""
    if any(r is None for r in recs):
        return None, None
    full = recs[-1]
    return np.array([[float(r[k]["recommend"] == full[k]["recommend"]) for k in DECISIONS] for r in recs]), {
        k: full[k]["recommend"] for k in DECISIONS}


def shapley_table(values):
    phi = shapley_from_values(values)  # (6, 2)
    return {d: {g: float(phi[i, k]) for i, g in enumerate(GROUPS)} for k, d in enumerate(DECISIONS)}


# ------------------------------------------------------------------------------- requests and answers

def build_requests(chosen):
    """{key: (system, user, schema)} and {key: (scenario id, mask)} for every version of every chosen scenario."""
    from strokecopilot.nlp.reasoner import prompt as rp
    from strokecopilot.pipeline import batched, paid_runs as pr
    from strokecopilot.contract.validate import validate_vision_output
    system, schema = rp.system_prompt(pr.PROMPT, rulebook=True), rp.api_schema()
    requests, index = {}, {}
    for s, spec in chosen:
        for v in versions(s, spec):
            validate_vision_output(v["vision"])
            user = rp.user_message(v["vision"], v["note"])
            key = batched.request_key(system, user)
            requests[key] = (system, user, schema)
            index[key] = (s.id, v["mask"])
    return requests, index


def answer(model, llm, out_dir, chosen=None, workers=8, limit=None):
    """Colab: answer every version for one model, saving each answer as it arrives (resumable). `limit`: the first n
    chosen scenarios only (the dummy run)."""
    from strokecopilot.pipeline import batched, paid_runs as pr
    chosen = (chosen or choose(scenarios_with_specs()))[:limit]
    requests, _ = build_requests(chosen)
    out = Path(out_dir) / model
    out.mkdir(parents=True, exist_ok=True)
    path = out / batched.ANSWERS_FILE
    before = len(set(requests) & set(batched.load_answers(path, pr.answer_model(model))))
    answers = batched.gather(requests, llm, log=lambda *_: None, answers_path=path, workers=workers)
    return {"model": model, "requests": len(requests), "saved_before": before,
            "valid": sum("reply" in r for r in answers.values())}


class _Saved:
    def __init__(self, answers, model):
        self.answers, self.model = answers, model

    def __call__(self, system, user, schema):
        from strokecopilot.pipeline import batched
        result = self.answers.get(batched.request_key(system, user))
        if result is None or "reply" not in result:
            raise RuntimeError((result or {}).get("error", "no answer"))
        return result["reply"]


def score(run_dir, models, chosen=None, limit=None):
    """Shapley values per scenario for each model and for the rules on the same versions, and a summary."""
    from strokecopilot.nlp.reasoner import run as rr
    from strokecopilot.pipeline import batched, paid_runs as pr
    chosen = (chosen or choose(scenarios_with_specs()))[:limit]
    rows, summary = [], {}
    rule_rows = {}
    for s, spec in chosen:
        vs = versions(s, spec)
        values, full = rule_values(vs)
        rule_rows[s.id] = {"shapley": shapley_table(values), "full": full, "vs": vs}
    for model in models:
        answers = batched.load_answers(Path(run_dir) / model / batched.ANSWERS_FILE, pr.answer_model(model))
        llm = _Saved(answers, pr.answer_model(model))
        for s, _ in chosen:
            vs = rule_rows[s.id]["vs"]
            recs = [rr.recommend(v["vision"], v["note"], llm, version=pr.PROMPT)[0] for v in vs]
            values, full = model_values(recs)
            rows.append({"scenario": s.id, "model": model, "complete": values is not None,
                         "format_failures": sum(r is None for r in recs),
                         "full_answer": full, "shapley": shapley_table(values) if values is not None else None,
                         "rules_full": rule_rows[s.id]["full"], "rules_shapley": rule_rows[s.id]["shapley"]})
    for model in models:
        done = [r for r in rows if r["model"] == model and r["complete"]]
        summary[model] = {"scenarios_complete": len(done), "scenarios": len(chosen),
                          **{d: {"model_mean_abs": {g: float(np.mean([abs(r["shapley"][d][g]) for r in done])) for g in GROUPS}
                                 if done else None,
                                 "rules_mean_abs": {g: float(np.mean([abs(r["rules_shapley"][d][g]) for r in done])) for g in GROUPS}
                                 if done else None} for d in DECISIONS}}
    return rows, summary


def main(argv=None):
    """On the Mac, after notebook 55 (`shap`): score the saved answers from the Drive run folder."""
    import argparse

    from strokecopilot.config import local_drive_root
    ap = argparse.ArgumentParser(description="SHAP row 9: score the 4B pair's answers against the rules")
    ap.add_argument("--run", default=None, help="default: <Drive>/runs/EXP-74_reasoner_shap")
    ap.add_argument("--out", default="runs/EXP-74_reasoner_shap")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    run = Path(args.run) if args.run else local_drive_root() / "runs" / "EXP-74_reasoner_shap"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = score(run, ("gemma-3-4b", "gemma-3-4b-tuned"), limit=args.limit)
    (out / "per_scenario.json").write_text(json.dumps(rows, indent=1))
    (out / "metrics.json").write_text(json.dumps({"experiment": "EXP-74", "groups": list(GROUPS), "summary": summary},
                                                 indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
