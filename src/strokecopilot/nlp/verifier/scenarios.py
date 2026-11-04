"""Synthetic scenarios with rule-derived expected recommendations. Plan §9, DEC-11; docs/HALLUCINATION_METRIC.md §5.

Each scenario is a vision_output.json (valid against the contract), a triage note with its fact sheet (spans verbatim),
and the expected decisions from the rulebook in code (rules.py), with the passages and inputs that decided them.
Scenarios are generated from a fixed seed, so the set is the same on every machine; `manifest()` records a hash.

The pool is drawn at random over the factors the rules turn on (time and its basis, occlusion site, ASPECTS, the
perfusion profile, age, NIHSS, pre-stroke mRS, contraindications), then thinned so that no decision path dominates:
at most `per_path` scenarios share the same pair of decision reasons (`per_undetermined` when either decision is
undetermined, so most scenarios have a determined answer). The set is checked to hold at least MIN_PER_ROUTE scenarios
for each route in KEY_ROUTES. Nothing here uses patient data.
"""

import hashlib
import json
import random
from dataclasses import dataclass

from strokecopilot.contract.schema import derive_mismatch
from strokecopilot.contract.validate import validate_vision_output
from strokecopilot.nlp.notes.render import render
from strokecopilot.nlp.verifier import rules

SEED = 20261105
N = 150

MINUTES = (60, 120, 150, 210, 260, 280, 330, 360, 400, 480, 540, 600, 720, 900, 1100, 1500)
SITES = ("M1", "M1", "M1", "M1", "ICA", "ICA", "M2", "basilar", "none", "unknown")
ASPECTS = (None, None, 2, 4, 7, 8, 9)
# (core mL, total hypoperfused mL): small core and large mismatch, DEFUSE-3 near-miss, moderate, large core, tiny core
PROFILES = ((12.0, 95.0), (15.0, 29.5), (28.0, 60.0), (45.0, 70.0), (85.0, 100.0), (0.6, 22.4))
CONTRAINDICATIONS = (None,) * 14 + ("haemorrhage", "haemorrhage", "warfarin_inr", "apixaban", "head_trauma", "previous_ich", "recent_surgery")
DEFICITS = ("right-sided weakness and slurred speech", "left arm and leg weakness with left-sided neglect",
            "word-finding difficulty and right arm weakness", "dysarthria and left facial droop",
            "vertigo, diplopia and unsteady gait")


@dataclass
class Scenario:
    id: str
    vision: dict
    note: str
    facts: object  # FactSheet
    expected: dict  # {"thrombolysis": Verdict.to_dict(), "thrombectomy": ...}
    tags: dict

    def to_dict(self):
        return {"id": self.id, "vision": self.vision, "note": self.note,
                "facts": [vars(f) for f in self.facts], "expected": self.expected, "tags": self.tags}


def _draw(rng, n):
    minutes = rng.choice(MINUTES)
    onset = rng.choice(("witnessed", "witnessed", "unwitnessed", "wake_up")) if minutes > 270 else rng.choice(("witnessed", "witnessed", "unwitnessed"))
    site = rng.choice(SITES)
    core, penumbra = rng.choice(PROFILES)
    contra = rng.choice(CONTRAINDICATIONS)
    aspects = rng.choice(ASPECTS)
    if site == "basilar":
        aspects = None  # ASPECTS is an anterior-circulation score
    haem = contra == "haemorrhage"
    ratio, mismatch = derive_mismatch(core, penumbra)
    vision = {
        "schema_version": "0.1", "case_id": f"scenario-{n:03d}", "model_version": "synthetic",
        "haemorrhage": {"present": haem, "confidence": round(rng.uniform(0.85, 0.99) if haem else rng.uniform(0.01, 0.10), 2)},
        "core_ml": core, "penumbra_ml": penumbra, "mismatch_ratio": ratio, "mismatch_volume_ml": mismatch,
        "occlusion_site": site, "aspects": aspects,
        "confidence": {"core": round(rng.uniform(0.6, 0.92), 2), "penumbra": round(rng.uniform(0.6, 0.92), 2)},
        "warnings": ["CTA not acquired"] if site == "unknown" else [],
    }
    sex = rng.choice(("male", "female"))
    spec = {
        "age": rng.choice((54, 63, 71, 77, 82, 88)), "sex": sex, "onset": onset, "minutes": minutes,
        "imaging_clock": rng.randrange(8 * 60, 22 * 60), "sleep_minutes": rng.choice((360, 420, 480, 540)),
        "deficits": rng.choice(DEFICITS[:4] if site != "basilar" else DEFICITS[4:]),
        "referral": rng.choice((None, "brought in by ambulance", "transferred from a regional hospital")),
        "nihss": rng.choice((4, 7, 11, 16, 22)), "mrs": rng.choice((0, 0, 1, 1, 2, 3)),
        "medications": rng.choice(([], ["ramipril"], ["metformin", "ramipril"], ["atorvastatin"], ["bisoprolol", "atorvastatin"])),
        "anticoagulant": "warfarin" if contra == "warfarin_inr" else "apixaban" if contra == "apixaban" else False,
        "antiplatelet": rng.choice((False, False, True)) if contra not in ("warfarin_inr", "apixaban") else None,
        "inr": 2.1 if contra == "warfarin_inr" else None,
        "history": rng.choice(([], ["hypertension"], ["hypertension", "type 2 diabetes"], ["hyperlipidaemia"],
                               ["hypertension", "ischaemic heart disease"])),
        "head_trauma": True if contra == "head_trauma" else False,
        "previous_ich": True if contra == "previous_ich" else False,
        "recent_surgery": True if contra == "recent_surgery" else False,
        "bp": (rng.choice((138, 152, 168, 176, 194)), rng.choice((78, 84, 92, 104))),
        "glucose": rng.choice((5.4, 6.1, 6.8, 7.4, 8.9)),
    }
    if spec["anticoagulant"] is False and spec["antiplatelet"] is None:
        spec["antiplatelet"] = False
    return vision, spec, contra


KEY_ROUTES = ("Within 4.5 h with a disabling deficit", "Beyond 4.5 h and eligible for thrombectomy", "Extended window, not eligible",
              "LVO 4.5-24 h without thrombectomy", "Acute intracranial haemorrhage", "INR above 1.7", "Recent head injury",
              "Relative contraindication", "0-6 h, ASPECTS 3-10", "6-24 h, ASPECTS 6 or more", "6-24 h, ASPECTS 3-5",
              "ASPECTS unknown; DAWN group met", "ASPECTS unknown; DEFUSE-3", "M2 occlusion: dominance", "Basilar occlusion",
              "No occlusion is identified")
MIN_PER_ROUTE = 3


def routes(scenarios):
    """{route: count} over both decisions' reasons."""
    return {r: sum(any(s.expected[d]["reason"].startswith(r) for d in s.expected) for s in scenarios) for r in KEY_ROUTES}


def generate(n=N, seed=SEED, per_path=5, per_undetermined=3, pool=20000):
    """n scenarios, deterministic for a seed, with at least MIN_PER_ROUTE for each key route.

    Draws fill the set under the per-path caps; if a key route is still short, further draws that hit it replace a
    scenario from the most common path. Raises if the pool runs out first."""
    rng = random.Random(seed)
    chosen, per = [], {}

    def path_of(verdicts):
        return (verdicts["thrombolysis"]["reason"], verdicts["thrombectomy"]["reason"])

    def short(scenarios):
        return [r for r, count in routes(scenarios).items() if count < MIN_PER_ROUTE]

    for k in range(pool):
        vision, spec, contra = _draw(rng, k + 1)
        validate_vision_output(vision)
        note, facts = render(spec, rng)
        verdicts = {d: v.to_dict() for d, v in rules.decide(rules.case_from(vision, facts)).items()}
        path = path_of(verdicts)
        tags = {"contraindication": contra, "site": vision["occlusion_site"], "onset": spec["onset"],
                "minutes": spec["minutes"], "aspects": vision["aspects"]}
        candidate = Scenario(id="", vision=vision, note=note, facts=facts, expected=verdicts, tags=tags)
        if len(chosen) < n:
            undetermined = rules.UNDETERMINED in (verdicts["thrombolysis"]["value"], verdicts["thrombectomy"]["value"])
            if per.get(path, 0) >= (per_undetermined if undetermined else per_path):
                continue
            chosen.append(candidate)
            per[path] = per.get(path, 0) + 1
            continue
        missing = short(chosen)
        if not missing:
            break
        if not any(any(v["reason"].startswith(r) for v in verdicts.values()) for r in missing):
            continue
        # replace one scenario of the most common path whose removal leaves every route at its count or above
        for victim_path, _ in sorted(per.items(), key=lambda kv: -kv[1]):
            idx = next((i for i, c in enumerate(chosen) if path_of(c.expected) == victim_path
                        and not set(short(chosen[:i] + chosen[i + 1:])) - set(missing)), None)
            if idx is not None:
                per[victim_path] -= 1
                chosen[idx] = candidate
                per[path] = per.get(path, 0) + 1
                break
    if len(chosen) < n or short(chosen):
        raise ValueError(f"{len(chosen)} scenarios, short routes {short(chosen)}, after {pool} draws.")
    for i, c in enumerate(chosen, 1):
        c.id = c.vision["case_id"] = f"scenario-{i:03d}"
    return chosen


def manifest(scenarios):
    """Counts by decision and a hash of the whole set, for the run's config.json."""
    blob = json.dumps([s.to_dict() for s in scenarios], sort_keys=True, ensure_ascii=False).encode()
    counts = {}
    for s in scenarios:
        key = f"thrombolysis {s.expected['thrombolysis']['value']} / thrombectomy {s.expected['thrombectomy']['value']}"
        counts[key] = counts.get(key, 0) + 1
    return {"n": len(scenarios), "seed": SEED, "sha256": hashlib.sha256(blob).hexdigest(), "decisions": dict(sorted(counts.items()))}
