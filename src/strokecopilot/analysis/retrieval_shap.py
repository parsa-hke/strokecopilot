"""SHAP survey row 10 (DEC-35): which tokens of a rulebook passage carry the bge similarity to a claim about it.

EXP-7 found top-k retrieval missed 6 of the 15 key fixture rules, and diagnosed that bge-base ranks the right rule
family first but cannot order rows that differ only in numbers. This asks the model directly: for each key passage, how
much of its similarity to a claim citing it comes from the number tokens ("6", "24", "70", "ASPECTS 3-5") against the
word tokens.

- **Pairs:** the 12 distinct passages behind the 15 key fixture rules (`fixtures/expected_rules.json`), each with the
  first gold rationale claim that cites it (`fixtures/gold/`). EXP-7's own queries were not saved, so the gold claims
  stand in for them; this is stated with the result.
- **Explained:** f(passage text) = cosine(bge(claim, as a query), bge(passage)), over the passage's tokens, with the
  `shap` text masker (Partition explainer; a masked token is replaced by the tokenizer's mask token).
- **Read-out:** each token's Shapley value; the share of the |values| on tokens that hold a digit, against the share of
  tokens that hold one (a ratio under 1 means numbers carry less than their share). Also the right passage's rank for
  its claim among all 47, and the passage that beats it.
"""

import json
import re
from pathlib import Path

import numpy as np

from strokecopilot.config import FIXTURES_DIR

DIGIT = re.compile(r"\d")


def pairs():
    """[(passage id, claim)] for the distinct key passages, in sorted order, each with its first citing gold claim."""
    key = json.loads((FIXTURES_DIR / "expected_rules.json").read_text())["fixtures"]
    ids = sorted({i for v in key.values() for i in v})
    cited = {}
    for f in sorted((FIXTURES_DIR / "gold").glob("*.json")):
        for entry in json.loads(f.read_text())["rationale"]:
            for s in entry["sources"]:
                if s["type"] == "passage":
                    cited.setdefault(s["id"], entry["claim"])
    missing = [i for i in ids if i not in cited]
    if missing:
        raise ValueError(f"no gold claim cites {missing}")
    return [(i, cited[i]) for i in ids]


def digit_share(tokens, values):
    """(share of |values| on tokens with a digit, share of tokens with a digit), over non-empty tokens."""
    keep = [k for k, t in enumerate(tokens) if t.strip()]
    tok = [tokens[k] for k in keep]
    val = np.abs(np.asarray(values, dtype=float)[keep])
    is_num = np.array([bool(DIGIT.search(t)) for t in tok])
    if not len(tok) or val.sum() == 0:
        return float("nan"), float("nan")
    return float(val[is_num].sum() / val.sum()), float(is_num.mean())


def explain(index, max_evals=600):
    """One row per pair: the right passage's rank and score, the passage that beats it, and the token attributions."""
    import shap

    model = index.embed
    model([""], query=False)  # load the model
    tokenizer = model._model.tokenizer
    masker = shap.maskers.Text(tokenizer)
    rows = []
    for pid, claim in pairs():
        q = index.embed([claim], query=True)[0]
        ranked = index.search(claim, k=len(index.ids))
        rank = [i for i, _ in ranked].index(pid) + 1

        def f(texts, q=q):
            return index.embed(list(texts), query=False) @ q

        explainer = shap.Explainer(f, masker)
        sv = explainer([index.corpus[pid]["text"]], max_evals=max_evals)
        tokens, values = list(sv.data[0]), [float(v) for v in sv.values[0]]
        num_share, num_tokens = digit_share(tokens, values)
        top = sorted(zip(tokens, values), key=lambda tv: -abs(tv[1]))[:8]
        rows.append({"passage": pid, "claim": claim, "rank": rank, "score": ranked[rank - 1][1],
                     "beaten_by": [i for i, _ in ranked[:rank - 1]][:3], "base": float(sv.base_values[0]),
                     "digit_attribution_share": num_share, "digit_token_share": num_tokens,
                     "ratio": num_share / num_tokens if num_tokens else float("nan"),
                     "top_tokens": [[t.strip(), v] for t, v in top], "tokens": tokens, "values": values})
    return rows


def summarise(rows):
    ratios = np.array([r["ratio"] for r in rows if np.isfinite(r["ratio"])])
    return {"pairs": len(rows), "right_passage_first": sum(r["rank"] == 1 for r in rows),
            "right_passage_in_top5": sum(r["rank"] <= 5 for r in rows),
            "median_digit_attribution_share": float(np.median([r["digit_attribution_share"] for r in rows])),
            "median_digit_token_share": float(np.median([r["digit_token_share"] for r in rows])),
            "median_ratio": float(np.median(ratios)) if len(ratios) else None,
            "pairs_ratio_below_1": int((ratios < 1).sum())}


def main(argv=None):
    import argparse

    from strokecopilot.nlp.corpus.index import Index
    ap = argparse.ArgumentParser(description="SHAP row 10: retriever token attributions")
    ap.add_argument("--out", default="runs/EXP-71_retrieval_shap")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    rows = explain(Index.from_corpus())
    summary = summarise(rows)
    (out / "per_pair.json").write_text(json.dumps(rows, indent=1))
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(r["passage"], "rank", r["rank"], "digit share", round(r["digit_attribution_share"], 2), "of tokens",
              round(r["digit_token_share"], 2), [t for t, _ in r["top_tokens"][:5]])


if __name__ == "__main__":
    main()
