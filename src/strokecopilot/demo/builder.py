"""The Space's scenario builder (DEC-32 §4): structured inputs -> a synthetic case, built the way the 150 evaluation
scenarios were (`scenarios._draw`, `notes.render`), and the nearest precomputed case when a live run cannot be served.

No free text reaches a model: every input is a number in a fixed range or one of a fixed set of choices, checked by
`validate`. Fields the builder does not expose (imaging clock, referral, routine drugs, history, blood pressure, glucose,
the vision confidences) are drawn by a generator seeded from the inputs' hash, so the same inputs always give the same
note and contract, which is what lets the result cache work.

The bounds (`BOUNDS`) are the ranges the evaluated sets cover: the 150 scenarios' grids (`scenarios.TEST_GRIDS`), the
five fixtures, and the development contracts' volumes.
"""

import hashlib
import json
import random
from dataclasses import dataclass

from strokecopilot.contract.validate import validate_vision_output
from strokecopilot.nlp.notes.render import render
from strokecopilot.nlp.verifier import rules
from strokecopilot.nlp.verifier import scenarios as sc
from strokecopilot.vision.perfusion import confidence as conf
from strokecopilot.vision.perfusion import volumes

# name: (low, high, step) for numbers; a tuple of choices otherwise.
BOUNDS = {
    "minutes": (60, 1500, 10),          # onset or last known well to imaging; scenarios 60-1500
    "sleep_minutes": (360, 540, 30),    # wake-up only: time asleep before waking; scenarios 360-540
    "age": (50, 90, 1),                 # scenarios 54-88, fixtures 71-82
    "nihss": (2, 25, 1),                # scenarios 4-22, fixtures 8-18
    "mrs": (0, 3, 1),                   # scenarios 0-3
    "core_ml": (0.0, 150.0, 0.5),       # scenarios 0.6-85, fixtures up to 110, development cores up to about 150
    "penumbra_ml": (0.0, 250.0, 0.5),   # scenarios up to 100, fixtures 120, development Tmax > 6 s up to about 250
    "inr": (1.0, 4.0, 0.1),             # warfarin only; scenarios 2.1
}
CHOICES = {
    "onset": ("witnessed", "unwitnessed", "wake_up"),
    "sex": ("female", "male"),
    "site": ("ICA", "M1", "M2", "basilar", "none", "unknown"),
    "aspects": (None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "contraindication": ("none", "warfarin_inr", "apixaban", "head_trauma", "previous_ich", "recent_surgery"),
    "deficit": tuple(range(len(sc.DEFICITS))),
    "haemorrhage": (False, True),
    "antiplatelet": (False, True),
}
FIELDS = tuple(BOUNDS) + tuple(CHOICES)
DEFAULTS = {"minutes": 300, "sleep_minutes": 420, "age": 71, "nihss": 14, "mrs": 0, "core_ml": 20.0, "penumbra_ml": 90.0,
            "inr": 1.0, "onset": "witnessed", "sex": "female", "site": "M1", "aspects": None, "contraindication": "none",
            "deficit": 0, "haemorrhage": False, "antiplatelet": False}


class InputError(ValueError):
    """An input outside the builder's bounds or choices. The message names the field, never echoes free text."""


def _snap(value, low, high, step):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError("expected a number")
    if not low <= value <= high:
        raise InputError(f"must be between {low} and {high}")
    snapped = round(round((value - low) / step) * step + low, 6)
    return int(snapped) if isinstance(step, int) else float(snapped)


def validate(raw):
    """The builder's inputs, checked and snapped to their steps. Unknown fields are refused; missing ones take
    DEFAULTS. Consistency rules: basilar has no ASPECTS and the posterior deficit; anterior sites an anterior one;
    penumbra (Tmax > 6 s) includes the core (DEC-7); INR only with warfarin; sleep time only for wake-up."""
    if not isinstance(raw, dict):
        raise InputError("inputs must be an object")
    unknown = set(raw) - set(FIELDS)
    if unknown:
        raise InputError(f"unknown field(s): {len(unknown)}")
    out = {}
    for name in FIELDS:
        value = raw.get(name, DEFAULTS[name])
        try:
            if name in BOUNDS:
                out[name] = _snap(value, *BOUNDS[name])
            elif value not in CHOICES[name] or (name in ("aspects", "deficit") and isinstance(value, bool)):
                raise InputError("not one of the allowed choices")
            else:
                out[name] = value
        except InputError as e:
            raise InputError(f"{name}: {e}") from None
    if out["penumbra_ml"] < out["core_ml"]:
        raise InputError("penumbra_ml: the Tmax > 6 s volume includes the core, so it cannot be smaller")
    posterior = len(sc.DEFICITS) - 1
    if out["site"] == "basilar":
        out["aspects"], out["deficit"] = None, posterior
    elif out["deficit"] == posterior:
        out["deficit"] = 0
    if out["contraindication"] != "warfarin_inr":
        out["inr"] = None
    if out["onset"] != "wake_up":
        out["sleep_minutes"] = None
    if out["onset"] == "wake_up" and out["minutes"] <= 270:
        raise InputError("minutes: a wake-up stroke in the scenarios is more than 4.5 h from last known well")
    return out


def key(inputs):
    """The canonical hash of validated inputs: the cache key and the seed of the hidden fields."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


@dataclass
class Built:
    key: str
    inputs: dict
    vision: dict
    note: str
    facts: object
    expected: dict  # {decision: Verdict.to_dict()}


def build(raw):
    """Validated inputs -> the case the reasoner sees (contract and note) and what the rules say, as for a scenario."""
    inputs = validate(raw)
    k = key(inputs)
    rng = random.Random(int(k[:16], 16))
    core, penumbra = inputs["core_ml"], inputs["penumbra_ml"]
    contra = inputs["contraindication"]
    haem = inputs["haemorrhage"]
    vision = {
        "schema_version": "0.1", "case_id": f"builder-{k[:10]}", "model_version": "synthetic (Space scenario builder)",
        "haemorrhage": {"present": haem, "confidence": round(rng.uniform(0.85, 0.99) if haem else rng.uniform(0.01, 0.10), 2)},
        **volumes.contract_volumes(core, penumbra),
        "occlusion_site": inputs["site"], "aspects": inputs["aspects"],
        "confidence": {"core": round(rng.uniform(0.6, 0.92), 2), "penumbra": round(rng.uniform(0.6, 0.92), 2)},
        "warnings": (["CTA not acquired"] if inputs["site"] == "unknown" else []) + conf.core_warnings(core),
    }
    validate_vision_output(vision)
    anticoagulant = "warfarin" if contra == "warfarin_inr" else "apixaban" if contra == "apixaban" else False
    spec = {
        "age": inputs["age"], "sex": inputs["sex"], "onset": inputs["onset"], "minutes": inputs["minutes"],
        "imaging_clock": rng.randrange(8 * 60, 22 * 60), "sleep_minutes": inputs["sleep_minutes"] or 420,
        "deficits": sc.DEFICITS[inputs["deficit"]],
        "referral": rng.choice((None, "brought in by ambulance", "transferred from a regional hospital")),
        "nihss": inputs["nihss"], "mrs": inputs["mrs"],
        "medications": rng.choice(([], ["ramipril"], ["metformin", "ramipril"], ["atorvastatin"], ["bisoprolol", "atorvastatin"])),
        "anticoagulant": anticoagulant,
        "antiplatelet": None if anticoagulant else inputs["antiplatelet"],
        "inr": inputs["inr"],
        "history": rng.choice(([], ["hypertension"], ["hypertension", "type 2 diabetes"], ["hyperlipidaemia"],
                               ["hypertension", "ischaemic heart disease"])),
        "head_trauma": contra == "head_trauma", "previous_ich": contra == "previous_ich",
        "recent_surgery": contra == "recent_surgery",
        "bp": (rng.choice((138, 152, 168, 176, 194)), rng.choice((78, 84, 92, 104))),
        "glucose": rng.choice((5.4, 6.1, 6.8, 7.4, 8.9)),
    }
    if spec["anticoagulant"] is False and spec["antiplatelet"] is None:
        spec["antiplatelet"] = False
    note, facts = render(spec, rng)
    expected = {d: v.to_dict() for d, v in rules.decide(rules.case_from(vision, facts)).items()}
    return Built(key=k, inputs=inputs, vision=vision, note=note, facts=facts, expected=expected)


# ------------------------------------------------------------------------------------------- Nearest precomputed case

NUMERIC = {"minutes": 1440.0, "core": 150.0, "penumbra": 250.0, "age": 40.0, "nihss": 23.0, "mrs": 3.0, "aspects": 10.0}
CATEGORICAL = ("haemorrhage", "site", "wake_up", "anticoagulant", "head_trauma", "recent_surgery", "previous_ich")


def features(vision, facts, expected=None):
    """What matching compares, from the rules' own view of a case (rules.case_from) plus its decisions."""
    case = rules.case_from(vision, facts)
    expected = expected or {d: v.to_dict() for d, v in rules.decide(case).items()}
    return {"values": tuple(expected[d]["value"] for d in ("thrombolysis", "thrombectomy")),
            "reasons": tuple(expected[d]["reason"] for d in ("thrombolysis", "thrombectomy")),
            **{c: getattr(case, c) for c in CATEGORICAL},
            **{n: getattr(case, n) for n in NUMERIC}}


def distance(a, b):
    """Sort key: same decisions first, then same reasons, then same categories, then the numbers (normalised)."""
    def num(name):
        x, y = a[name], b[name]
        if x is None and y is None:
            return 0.0
        if x is None or y is None:
            return 1.0
        return abs(float(x) - float(y)) / NUMERIC[name]

    return (sum(x != y for x, y in zip(a["values"], b["values"])), sum(x != y for x, y in zip(a["reasons"], b["reasons"])),
            sum(a[c] != b[c] for c in CATEGORICAL), round(sum(num(n) for n in NUMERIC), 6))


def _norm(features):
    """Features as compared: lists (as JSON stores them) become tuples."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in features.items()}


def nearest(target, bank):
    """The bank entry closest to target. bank: [{"id", "features", ...}], features fresh or read back from JSON.
    Returns (entry, the inputs that differ)."""
    target = _norm(target)
    best = min(bank, key=lambda e: (distance(target, _norm(e["features"])), e["id"]))
    found = _norm(best["features"])
    differs = [c for c in ("values", *CATEGORICAL) if target[c] != found[c]]
    differs += [n for n in NUMERIC if target[n] != found[n]]
    return best, differs
