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


class _Caps:
    """generate_training's skip rules, with their counters: a signature already taken, a decision path at its cap, or
    every key route of the draw full. take() applies them to one draw and counts it when it is kept."""

    def __init__(self, taken, per_path, per_undetermined, per_route):
        self.taken, self.per_path, self.per_undetermined, self.per_route = taken, per_path, per_undetermined, per_route
        self.per, self.route_count = {}, {}

    def take(self, sig, verdicts):
        if sig in self.taken:
            return False
        path = (verdicts["thrombolysis"]["reason"], verdicts["thrombectomy"]["reason"])
        undetermined = rules.UNDETERMINED in (verdicts["thrombolysis"]["value"], verdicts["thrombectomy"]["value"])
        if self.per.get(path, 0) >= (self.per_undetermined if undetermined else self.per_path):
            return False
        hit = routes_taken(verdicts)
        if hit and all(self.route_count.get(r, 0) >= self.per_route for r in hit):
            return False  # every route it takes is full; a draw that adds to a short route is kept
        self.taken.add(sig)
        self.per[path] = self.per.get(path, 0) + 1
        for r in hit:
            self.route_count[r] = self.route_count.get(r, 0) + 1
        return True


def routes_taken(verdicts):
    """The key routes a pair of verdicts takes (either reason starts with the route), in KEY_ROUTES order."""
    return [r for r in KEY_ROUTES if any(v["reason"].startswith(r) for v in verdicts.values())]


def recommended(verdicts):
    """{decision: bool}: the answer a gold reply is trained to give. Only a "yes" verdict is recommend = true;
    "undetermined" is recommend = false, like "no" (gold.gold_answer)."""
    return {d: v["value"] == rules.YES for d, v in verdicts.items()}


def _training_draws(seed, grids, held_out, pool, validate=True):
    """The stream of training draws for a seed, in order: (vision, note, facts, verdicts, tags) for every draw that
    takes no held-out route. The same seed gives the same stream, so it can be read more than once. validate=False
    leaves the contract check to the caller (the balanced set reads the stream four times and checks what it keeps)."""
    rng = random.Random(seed)
    for k in range(pool):
        vision, spec, contra = _draw(rng, k + 1, grids)
        if validate:
            validate_vision_output(vision)
        note, facts = render(spec, rng)
        verdicts = {d: v.to_dict() for d, v in rules.decide(rules.case_from(vision, facts)).items()}
        if any(v["reason"].startswith(r) for v in verdicts.values() for r in held_out):
            continue
        tags = {"contraindication": contra, "site": vision["occlusion_site"], "onset": spec["onset"],
                "minutes": spec["minutes"], "aspects": vision["aspects"]}
        yield vision, note, facts, verdicts, tags


# The decision-balanced training set (EXP-91; Parsa, 22 Nov 2026). A new seed; the held-out routes stay DEC-30's
# four, so a balanced set leaves `held_out` at held_out_routes() whatever its seed (held_out_routes(20261122) would
# draw other routes, DAWN and DEFUSE-3 among them).
BALANCED_SEED = 20261122
# The passes of a balanced set, in order; each reads the whole stream. A pass keeps a draw when (it recommends both
# treatments, it recommends at least one, it takes at most one key route) match the pass; None means either.
_BALANCED_PASSES = ((True, True, True), (None, True, True), (None, True, None), (None, False, None))


def generate_training(n, seed=TRAIN_SEED, grids=TRAIN_GRIDS, held_out=None, per_path=40, per_undetermined=20,
                      per_route=120, pool=400_000, exclude=None, balance=False):
    """n training scenarios ("train-0001", ...), deterministic for a seed. A draw is skipped when either decision's
    reason starts with a held-out route (default: held_out_routes(seed)), when its signature matches an evaluation
    scenario (`exclude`, default the 150 of generate()) or an earlier training one, when its decision path is at its
    cap, or when every key route it takes already has `per_route` scenarios. The route cap stops common routes (a
    relative contraindication) crowding out rare ones (DAWN, DEFUSE-3); it must not reject a draw that adds to a short
    route, because the routes of the two decisions come in pairs (a late thrombectomy "yes" mostly comes with a
    thrombolysis contraindication, once "beyond 4.5 h and eligible" is held out). Raises if the pool runs out first.

    balance=True (EXP-91) gives a decision-balanced set, with DEC-30's held-out routes (held_out_routes()) unless
    `held_out` says otherwise, whatever the seed: for each treatment, as many scenarios whose gold answer is
    recommend = true (a "yes" verdict) as the same skip rules allow, up to n // 2, the rest recommend = false ("no" or
    "undetermined": recommended()). The skip rules and the caps are the ones above, unchanged, with one more: a draw
    is skipped when it recommends a treatment that already has n // 2. Without help the stream is mostly "no", so it
    is read four times with the same counters: first the draws that recommend both treatments and take at most one
    key route, then the other draws that recommend a treatment and take at most one key route, then every other draw
    that recommends a treatment, and last the draws that recommend neither, until there are n. One-route draws go
    first because the route cap skips a draw only when every route it takes is full: a two-route draw can still be
    kept after one of its routes fills, a one-route draw cannot. The scenarios are numbered in the order kept, and
    cap_violations() finds none when it replays the rules above over them in that order. Exact balance is not
    promised: with DEC-30's held-out routes only two thrombolysis "yes" routes can be trained, and the caps hold them
    well below n // 2 (EXP-91 records the counts reached)."""
    held_out = held_out_routes(TRAIN_SEED if balance else seed) if held_out is None else list(held_out)
    taken = {signature(s.vision, s.facts) for s in (generate() if exclude is None else exclude)}
    caps = _Caps(taken, per_path, per_undetermined, per_route)
    chosen = []
    if not balance:
        for vision, note, facts, verdicts, tags in _training_draws(seed, grids, held_out, pool):
            if not caps.take(signature(vision, facts), verdicts):
                continue
            chosen.append(Scenario(id="", vision=vision, note=note, facts=facts, expected=verdicts, tags=tags))
            if len(chosen) == n:
                break
    else:
        target, yes = n // 2, {"thrombolysis": 0, "thrombectomy": 0}
        for want_both, want_any, one_route in _BALANCED_PASSES:
            for vision, note, facts, verdicts, tags in _training_draws(seed, grids, held_out, pool, validate=False):
                if len(chosen) == n or (want_any and all(count >= target for count in yes.values())):
                    break
                rec = recommended(verdicts)
                if any(rec.values()) != want_any or (want_both and not all(rec.values())):
                    continue
                if any(rec[d] and yes[d] >= target for d in rec):
                    continue
                if one_route and len(routes_taken(verdicts)) > 1:
                    continue
                if not caps.take(signature(vision, facts), verdicts):
                    continue
                validate_vision_output(vision)
                for d in rec:
                    yes[d] += rec[d]
                chosen.append(Scenario(id="", vision=vision, note=note, facts=facts, expected=verdicts, tags=tags))
    if len(chosen) < n:
        raise ValueError(f"{len(chosen)} training scenarios after {pool} draws, {len(caps.per)} decision paths; "
                         "raise the caps or the pool.")
    for i, c in enumerate(chosen, 1):
        c.id = c.vision["case_id"] = f"train-{i:04d}"
    return chosen


def cap_violations(scenarios, per_path=40, per_undetermined=20, per_route=120):
    """Replay generate_training's caps over a training set in its own order: the IDs of the scenarios the rules would
    have skipped (a repeated signature, a decision path at its cap, every key route full). [] for any set the
    generator wrote with these caps, balanced or not."""
    caps = _Caps(set(), per_path, per_undetermined, per_route)
    return [s.id for s in scenarios if not caps.take(signature(s.vision, s.facts), s.expected)]


def decision_balance(scenarios):
    """Counts for a training set's record: per treatment the verdicts and the trained answers (recommend true or
    false), and the four (thrombolysis, thrombectomy) answer pairs."""
    out = {}
    for d in ("thrombolysis", "thrombectomy"):
        values = [s.expected[d]["value"] for s in scenarios]
        yes = values.count(rules.YES)
        out[d] = {"yes": yes, "no": values.count(rules.NO), "undetermined": values.count(rules.UNDETERMINED),
                  "recommend_true": yes, "recommend_false": len(values) - yes}
    pairs = [tuple(recommended(s.expected)[d] for d in ("thrombolysis", "thrombectomy")) for s in scenarios]
    out["pairs"] = {name: pairs.count(key) for name, key in (("both", (True, True)), ("thrombolysis_only", (True, False)),
                                                               ("thrombectomy_only", (False, True)), ("neither", (False, False)))}
    return out
