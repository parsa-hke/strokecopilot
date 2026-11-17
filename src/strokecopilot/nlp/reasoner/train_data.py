"""The fine-tuning dataset for the small reasoner (Item 4, DEC-30): training scenarios (scenarios.generate_training)
with their gold answers (gold.gold_answer), as chat examples in prompt v3's own format.

Each example is the request the reasoner gets in evaluation (system prompt v3 with the rulebook; the user message
holding the vision JSON and the note) and the gold reply as the assistant's answer, so the model is trained on exactly
the input it will be scored on. Every gold answer must pass gold.check(); one that does not stops the build, because a
training set with a wrong answer in it would teach that error. The validation slice (for early stopping) is every
tenth example by a fixed shuffle. Nothing here uses patient data: the scenarios are synthetic.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from strokecopilot.nlp.corpus import ingest
from strokecopilot.nlp.reasoner import gold
from strokecopilot.nlp.reasoner import prompt as rp
from strokecopilot.nlp.verifier import scenarios as sc

PROMPT = "reasoner_v3"
VAL_EVERY = 10
SPLIT_SEED = 20261117


def examples(scenarios, version=PROMPT, corpus=None):
    """[{id, route, user, answer}] for the scenarios, each answer a gold reply as compact JSON. Raises if a gold
    answer fails gold.check()."""
    corpus = corpus or ingest.load()
    out = []
    for s in scenarios:
        reply = gold.gold_answer(s.vision, s.facts, corpus)
        problems = gold.check(s.vision, s.facts, s.note, reply, corpus)
        if problems:
            raise ValueError(f"The gold answer for {s.id} fails its checks: {problems[:3]}")
        out.append({"id": s.id, "route": [s.expected["thrombolysis"]["reason"], s.expected["thrombectomy"]["reason"]],
                    "user": rp.user_message(s.vision, s.note), "answer": json.dumps(reply, ensure_ascii=False)})
    return out


def split(items, every=VAL_EVERY, seed=SPLIT_SEED):
    """(train, validation): a fixed shuffle, every `every`-th item to validation."""
    order = list(items)
    random.Random(seed).shuffle(order)
    return [x for i, x in enumerate(order) if i % every], [x for i, x in enumerate(order) if not i % every]


def build(out_dir, n, seed=sc.TRAIN_SEED, held_out=None, version=PROMPT):
    """Write train.jsonl, val.jsonl, system_prompt.md and meta.json to out_dir; returns meta. The system prompt is
    the same for every example, so it is written once and its hash recorded."""
    held_out = sc.held_out_routes(seed) if held_out is None else list(held_out)
    scenarios = sc.generate_training(n, seed=seed, held_out=held_out)
    items = examples(scenarios, version)
    train, val = split(items)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    system = rp.system_prompt(version)
    (out / "system_prompt.md").write_text(system, encoding="utf-8")
    for name, rows in (("train", train), ("val", val)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    chars = [len(system) + len(x["user"]) + len(x["answer"]) for x in items]
    answer_chars = [len(x["answer"]) for x in items]
    meta = {"n": len(items), "train": len(train), "val": len(val), "seed": seed, "split_seed": SPLIT_SEED,
            "prompt_version": version, "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "held_out_routes": held_out, "grids": "scenarios.TRAIN_GRIDS",
            "test_set_sha256": sc.manifest(sc.generate())["sha256"],
            "thrombolysis": dict(Counter(s.expected["thrombolysis"]["value"] for s in scenarios)),
            "thrombectomy": dict(Counter(s.expected["thrombectomy"]["value"] for s in scenarios)),
            "routes": {r: c for r, c in sc.routes(scenarios).items()},
            "chars_per_example": {"mean": round(sum(chars) / len(chars)), "max": max(chars)},
            "answer_chars": {"mean": round(sum(answer_chars) / len(answer_chars)), "max": max(answer_chars)}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
