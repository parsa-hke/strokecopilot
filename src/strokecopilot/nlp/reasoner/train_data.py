"""The fine-tuning dataset for the small reasoner (Item 4, DEC-30): training scenarios (scenarios.generate_training)
with their gold answers (gold.gold_answer), as chat examples in prompt v3's own format.

Each example is the request the reasoner gets in evaluation (system prompt v3 with the rulebook; the user message
holding the vision JSON and the note) and the gold reply as the assistant's answer, so the model is trained on exactly
the input it will be scored on. Every gold answer must pass gold.check(); one that does not stops the build, because a
training set with a wrong answer in it would teach that error. The validation slice (for early stopping) is every
tenth example by a fixed shuffle. Nothing here uses patient data: the scenarios are synthetic.

build() writes EXP-43, the set the tuned model of DEC-30 was trained on. build_balanced() writes EXP-91, the
decision-balanced set of the post-freeze, report-only variant (Parsa, 22 Nov 2026): the same recipe with a new seed
and more answers that recommend a treatment.
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
    return _write(out_dir, scenarios, seed, held_out, version)


def _write(out_dir, scenarios, seed, held_out, version, extra=None):
    """The dataset files for `scenarios` and meta.json (with `extra(train, val)` merged in, if given); returns meta."""
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
    if extra:
        meta.update(extra(train, val))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# --- the decision-balanced set (EXP-91; Parsa, 22 Nov 2026) -------------------------------------------------------

EXP43_N = 1200  # EXP-43's size: build_balanced() regenerates its scenarios, so that the new set repeats none of them
CAPS = {"per_path": 40, "per_undetermined": 20, "per_route": 120}  # generate_training's defaults, as in EXP-43
# Draws per pass. The balanced set reads the stream to its end in the passes that look for a "yes"; the counts it
# reaches are the same at 100,000, 200,000 and 400,000 draws (checked 22 Nov 2026), so the caps decide, not the pool.
BALANCED_POOL = 100_000


def answer_counts(rows):
    """{decision: {"true", "false"}} counted on the answers themselves: what the model is trained to output."""
    out = {d: {"true": 0, "false": 0} for d in ("thrombolysis", "thrombectomy")}
    for row in rows:
        reply = json.loads(row["answer"])
        for d in out:
            out[d]["true" if reply[d]["recommend"] else "false"] += 1
    return out


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_balanced(out_dir, n=1200, seed=sc.BALANCED_SEED, version=PROMPT, pool=BALANCED_POOL):
    """EXP-91: the decision-balanced training set, in the same files as build(). Everything is as in EXP-43 (the
    grids, DEC-30's four held-out routes, the caps, prompt v3's request format, the gold answers and their checks,
    the validation slice) except the seed and the balance (scenarios.generate_training(balance=True)). No scenario
    has the signature of an evaluation scenario or of an EXP-43 training scenario. meta.json adds what was reached:
    the answers the model is trained to give, counted on the written answers, in all examples and in each slice."""
    held_out = sc.held_out_routes()  # DEC-30's four, whatever this set's seed
    evaluation = sc.generate()
    earlier = sc.generate_training(EXP43_N, exclude=evaluation)
    scenarios = sc.generate_training(n, seed=seed, held_out=held_out, exclude=evaluation + earlier, balance=True,
                                     pool=pool, **CAPS)
    signatures = {sc.signature(s.vision, s.facts) for s in scenarios}
    shared = {name: len(signatures & {sc.signature(s.vision, s.facts) for s in other})
              for name, other in (("evaluation_scenarios", evaluation), ("exp43_training_scenarios", earlier))}
    paths = Counter((s.expected["thrombolysis"]["reason"], s.expected["thrombectomy"]["reason"]) for s in scenarios)
    out, by_id, corpus = Path(out_dir), {s.id: s for s in scenarios}, ingest.load()

    def passes(row):  # gold.check() again, on the answer as written to the file
        s = by_id[row["id"]]
        return not gold.check(s.vision, s.facts, s.note, json.loads(row["answer"]), corpus)

    def extra(train, val):
        return {"balanced": True, "balance_target_per_treatment": n // 2, "caps": CAPS, "pool": pool,
                "cap_violations": len(sc.cap_violations(scenarios, **CAPS)),
                "recommend": answer_counts(train + val), "recommend_train": answer_counts(train),
                "recommend_val": answer_counts(val), "answer_pairs": sc.decision_balance(scenarios)["pairs"],
                "decision_paths": len(paths), "largest_decision_path": max(paths.values()),
                "cases_shared_with": shared, "gold_checks": {"examples": len(train) + len(val), "passed": sum(passes(r) for r in train + val)},
                "files_sha256": {name: file_sha256(out / name) for name in ("train.jsonl", "val.jsonl")}}

    return _write(out, scenarios, seed, held_out, version, extra)
