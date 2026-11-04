"""Seeded errors: known fabrications planted in clean outputs, to measure the verifier. Plan §8.4;
docs/HALLUCINATION_METRIC.md §8.

The clean outputs are hand-written gold recommendations for the five fixtures (`fixtures/gold/`), checked in code by
tests/test_seeded_errors.py: every number, quote, passage ID and decision. Each seed (`fixtures/seeded_errors.json`)
is one edit to one gold output, labelled with the sub-type and the entry where the verifier should find it. The seeds
are hand-written, two per sub-type per fixture (70 in all).

Scoring:
- recall per sub-type, strict (a claim in the labelled entry gets that sub-type) and lenient (it gets any sub-type);
- false flags: hallucinations the verifier reports on the gold outputs, per claim and per output, plus flags
  outside the labelled entry in seeded outputs (a seed should change nothing else).
"""

import copy
import json
import re
from collections import defaultdict
from pathlib import Path

from strokecopilot.config import FIXTURES_DIR
from strokecopilot.nlp.verifier.trace import SUBTYPES

GOLD_DIR = Path(FIXTURES_DIR) / "gold"
SEEDS = Path(FIXTURES_DIR) / "seeded_errors.json"
_ID = re.compile(r"^(?P<loc>[a-z_.]+\[\d+\])")


def load_gold(stem):
    return json.loads((GOLD_DIR / f"{stem}.json").read_text())


def gold_stems():
    return sorted(p.stem for p in GOLD_DIR.glob("fx*.json"))


def load_seeds(path=SEEDS):
    return json.loads(Path(path).read_text())["seeds"]


def apply(rec, seed):
    """(seeded copy of rec, label {"field", "entry"} with "new" resolved to an index)."""
    out = copy.deepcopy(rec)
    label = dict(seed["label"])
    for edit in seed["edits"]:
        op = edit["op"]
        if op == "replace":
            entry = out["rationale"][edit["entry"]]
            if entry["claim"].count(edit["old"]) != 1:
                raise ValueError(f"{seed['id']}: {edit['old']!r} is not in entry {edit['entry']} exactly once")
            entry["claim"] = entry["claim"].replace(edit["old"], edit["new"])
        elif op == "append":
            out["rationale"].append(copy.deepcopy(edit["value"]))
            if label["entry"] == "new":
                label["entry"] = len(out["rationale"]) - 1
        elif op == "set_entry":
            entry = out["rationale"][edit["entry"]]
            entry["sources"] = copy.deepcopy(edit["sources"])
            if "claim" in edit:
                entry["claim"] = edit["claim"]
        elif op == "rule_id":
            ids = out[edit["decision"]]["rule_ids"]
            if edit["index"] >= len(ids):
                raise ValueError(f"{seed['id']}: {edit['decision']} has no rule_ids[{edit['index']}]")
            ids[edit["index"]] = edit["new"]
        else:
            raise ValueError(f"{seed['id']}: unknown edit {op!r}")
    if out == rec:
        raise ValueError(f"{seed['id']} changes nothing")
    return out, label


def seeded_set(seeds=None):
    """[{id, fixture, subtype, what, rec, label}] for every seed."""
    gold = {stem: load_gold(stem) for stem in gold_stems()}
    out = []
    for seed in seeds or load_seeds():
        rec, label = apply(gold[seed["fixture"]], seed)
        out.append({"id": seed["id"], "fixture": seed["fixture"], "subtype": seed["subtype"], "what": seed["what"],
                    "rec": rec, "label": label})
    return out


def _get(trace, name):
    return trace[name] if isinstance(trace, dict) else getattr(trace, name)


def location(claim_id):
    """'rationale[3].1' -> 'rationale[3]'; 'thrombolysis.rule_ids[0]' -> itself."""
    m = _ID.match(claim_id)
    return m.group("loc") if m else claim_id


def judge_seed(traces, label, subtype):
    """(strict hit, lenient hit, flags outside the labelled entry) for one seeded output's traces."""
    at = f"{label['field']}[{label['entry']}]"
    here = [_get(t, "subtype") for t in traces if location(_get(t, "claim_id")) == at]
    elsewhere = sum(1 for t in traces if location(_get(t, "claim_id")) != at and _get(t, "subtype") is not None)
    return subtype in here, any(s is not None for s in here), elsewhere


def score(seed_results, clean_results):
    """seed_results: [{"id", "subtype", "label", "traces"}]; clean_results: [{"fixture", "traces"}].

    Returns recall per sub-type (strict and lenient, with counts), the false-flag rates on the gold outputs, and
    flags outside the labelled entry."""
    by = defaultdict(lambda: {"n": 0, "strict": 0, "lenient": 0, "missed": []})
    outside = 0
    for r in seed_results:
        strict, lenient, elsewhere = judge_seed(r["traces"], r["label"], r["subtype"])
        row = by[r["subtype"]]
        row["n"] += 1
        row["strict"] += strict
        row["lenient"] += lenient
        if not strict:
            row["missed"].append(r["id"])
        outside += elsewhere
    recall = {}
    for st in sorted(by):
        row = by[st]
        recall[SUBTYPES[st]] = {"subtype": st, "n": row["n"], "strict": row["strict"] / row["n"],
                                "lenient": row["lenient"] / row["n"], "missed": row["missed"]}
    claims = sum(len(r["traces"]) for r in clean_results)
    flagged = sum(1 for r in clean_results for t in r["traces"] if _get(t, "subtype") is not None)
    outputs_flagged = sum(1 for r in clean_results if any(_get(t, "subtype") is not None for t in r["traces"]))
    return {
        "recall": recall,
        "false_flag_rate_claims": flagged / claims if claims else None,
        "clean_claims": claims,
        "clean_outputs_flagged": f"{outputs_flagged}/{len(clean_results)}",
        "flags_outside_label": outside,
        "meets_bar": all(v["strict"] >= 0.80 for v in recall.values()) and claims > 0 and flagged / claims <= 0.10,
    }
