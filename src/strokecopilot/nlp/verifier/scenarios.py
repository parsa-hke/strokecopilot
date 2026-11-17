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
# The value grids _draw picks from. TEST_GRIDS are the evaluation set's own (the 150 scenarios keep their hash: same
# values, same order of random draws). Training sets (Item 4, DEC-30) pass other grids, so no training scenario can
# repeat an evaluation one: a new seed alone would not do, because the grids are short.
TEST_GRIDS = {"minutes": MINUTES, "sites": SITES, "aspects": ASPECTS, "profiles": PROFILES,
              "contraindications": CONTRAINDICATIONS, "ages": (54, 63, 71, 77, 82, 88), "sleep": (360, 420, 480, 540),
              "nihss": (4, 7, 11, 16, 22), "mrs": (0, 0, 1, 1, 2, 3), "inr": 2.1}
# Training grids (17 Nov 2026): every decision-relevant value differs from TEST_GRIDS, with values just either side
# of the rules' cut-offs (4.5, 6, 9, 16 and 24 h; ASPECTS 2/3, 5/6; age 80 and 90; NIHSS 6, 10 and 20; mRS 1/2;
# DAWN cores 21, 31, 51 mL; core 70 mL; EXTEND and DEFUSE-3 ratio and mismatch limits). Sites, contraindication kinds
# and deficits are categories and stay the same.
TRAIN_GRIDS = {"minutes": (45, 100, 140, 200, 255, 265, 300, 350, 375, 420, 510, 560, 690, 860, 950, 1200, 1380, 1560),
               "sites": SITES, "aspects": (None, None, 1, 3, 5, 6, 10),
               "profiles": ((9.0, 88.0), (16.5, 33.0), (24.0, 52.0), (38.0, 66.0), (75.0, 92.0), (0.4, 18.0),
                            (50.0, 140.0), (19.0, 25.5)),
               "contraindications": CONTRAINDICATIONS, "ages": (49, 58, 67, 75, 79, 81, 86, 92),
               "sleep": (390, 450, 510, 570), "nihss": (3, 5, 6, 9, 10, 14, 19, 20, 25), "mrs": (0, 0, 1, 1, 2, 3, 4),
               "inr": 2.4}


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


def _draw(rng, n, grids=TEST_GRIDS):
    g = grids
    minutes = rng.choice(g["minutes"])
    onset = rng.choice(("witnessed", "witnessed", "unwitnessed", "wake_up")) if minutes > 270 else rng.choice(("witnessed", "witnessed", "unwitnessed"))
    site = rng.choice(g["sites"])
    core, penumbra = rng.choice(g["profiles"])
    contra = rng.choice(g["contraindications"])
    aspects = rng.choice(g["aspects"])
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
        "age": rng.choice(g["ages"]), "sex": sex, "onset": onset, "minutes": minutes,
        "imaging_clock": rng.randrange(8 * 60, 22 * 60), "sleep_minutes": rng.choice(g["sleep"]),
        "deficits": rng.choice(DEFICITS[:4] if site != "basilar" else DEFICITS[4:]),
        "referral": rng.choice((None, "brought in by ambulance", "transferred from a regional hospital")),
        "nihss": rng.choice(g["nihss"]), "mrs": rng.choice(g["mrs"]),
        "medications": rng.choice(([], ["ramipril"], ["metformin", "ramipril"], ["atorvastatin"], ["bisoprolol", "atorvastatin"])),
        "anticoagulant": "warfarin" if contra == "warfarin_inr" else "apixaban" if contra == "apixaban" else False,
        "antiplatelet": rng.choice((False, False, True)) if contra not in ("warfarin_inr", "apixaban") else None,
        "inr": g["inr"] if contra == "warfarin_inr" else None,
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


# --- training scenarios for the fine-tuned reasoner (Item 4, DEC-30) ---------------------------------------------

TRAIN_SEED = 20261117
N_HELD_OUT_ROUTES = 4


# Never held out (Parsa, 17 Nov 2026, amending DEC-30): the standard thrombolysis "yes" and the main absolute
# contraindication. Drawn from all 16, the seed held out both, which left extended-window thrombolysis as the only
# "yes" in training; the tuned model would never have seen the routes the system's main use and safety rest on.
CORE_ROUTES = ("Within 4.5 h with a disabling deficit", "Acute intracranial haemorrhage")


def held_out_routes(seed=TRAIN_SEED, k=N_HELD_OUT_ROUTES):
    """The key routes left out of training, chosen by seed before any training (DEC-30) from the routes other than
    CORE_ROUTES, in KEY_ROUTES order."""
    pool = [r for r in KEY_ROUTES if r not in CORE_ROUTES]
    picked = set(random.Random(seed).sample(pool, k))
    return [r for r in KEY_ROUTES if r in picked]


def signature(vision, facts):
    """Every input the rules read, as a tuple: two scenarios with the same signature are the same case to the rules."""
    import dataclasses

    return dataclasses.astuple(rules.case_from(vision, facts))


def generate_training(n, seed=TRAIN_SEED, grids=TRAIN_GRIDS, held_out=None, per_path=40, per_undetermined=20,
                      per_route=120, pool=400_000, exclude=None):
    """n training scenarios ("train-0001", ...), deterministic for a seed. A draw is skipped when either decision's
    reason starts with a held-out route (default: held_out_routes(seed)), when its signature matches an evaluation
    scenario (`exclude`, default the 150 of generate()) or an earlier training one, when its decision path is at its
    cap, or when every key route it takes already has `per_route` scenarios. The route cap stops common routes (a
    relative contraindication) crowding out rare ones (DAWN, DEFUSE-3); it must not reject a draw that adds to a short
    route, because the routes of the two decisions come in pairs (a late thrombectomy "yes" mostly comes with a
    thrombolysis contraindication, once "beyond 4.5 h and eligible" is held out). Raises if the pool runs out first."""
    held_out = held_out_routes(seed) if held_out is None else list(held_out)
    taken = {signature(s.vision, s.facts) for s in (generate() if exclude is None else exclude)}
    rng = random.Random(seed)
    chosen, per, route_count = [], {}, {}
    for k in range(pool):
        vision, spec, contra = _draw(rng, k + 1, grids)
        validate_vision_output(vision)
        note, facts = render(spec, rng)
        verdicts = {d: v.to_dict() for d, v in rules.decide(rules.case_from(vision, facts)).items()}
        if any(v["reason"].startswith(r) for v in verdicts.values() for r in held_out):
            continue
        sig = signature(vision, facts)
        if sig in taken:
            continue
        path = (verdicts["thrombolysis"]["reason"], verdicts["thrombectomy"]["reason"])
        undetermined = rules.UNDETERMINED in (verdicts["thrombolysis"]["value"], verdicts["thrombectomy"]["value"])
        if per.get(path, 0) >= (per_undetermined if undetermined else per_path):
            continue
        hit = [r for r in KEY_ROUTES if any(v["reason"].startswith(r) for v in verdicts.values())]
        if hit and all(route_count.get(r, 0) >= per_route for r in hit):
            continue  # every route it takes is full; a draw that adds to a short route is kept
        taken.add(sig)
        per[path] = per.get(path, 0) + 1
        for r in hit:
            route_count[r] = route_count.get(r, 0) + 1
        tags = {"contraindication": contra, "site": vision["occlusion_site"], "onset": spec["onset"],
                "minutes": spec["minutes"], "aspects": vision["aspects"]}
        chosen.append(Scenario(id="", vision=vision, note=note, facts=facts, expected=verdicts, tags=tags))
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(f"{len(chosen)} training scenarios after {pool} draws, {len(per)} decision paths; "
                         "raise the caps or the pool.")
    for i, c in enumerate(chosen, 1):
        c.id = c.vision["case_id"] = f"train-{i:04d}"
    return chosen
