"""Phase 5 task G (DEC-37): EXP-7's retrieval check repeated with BM25, a BM25 + bge hybrid and a cross-encoder.

EXP-7 (bge-base, two decision queries per fixture, top 8 each) found 9 of the 15 key fixture rules
(`fixtures/expected_rules.json`). DEC-19 then sent the whole rulebook to the reasoner instead. This asks whether
retrieval methods that see the numbers do better. DEC-19 stands whatever the result.

Fixed before any score:
- **Queries:** EXP-7's own were not saved, so they are rebuilt the same way: for each fixture and each treatment, the
  triage note, a line of the imaging values and the question "Is <treatment> indicated?". bge must reproduce EXP-7's
  9 of 15 with them, or the comparison is reported as against the rebuilt baseline.
- **Methods**, all over the 47 passage texts, top 8 per query, a rule found if either decision query of its fixture
  returns it:
  1. `bge`: BAAI/bge-base-en-v1.5 cosine, the query with its instruction (`corpus.index`);
  2. `bm25`: BM25 Okapi (k1 1.5, b 0.75, the `rank-bm25` defaults), tokens lower-cased, words and numbers (decimals
     kept whole, so "4.5" is one token);
  3. `hybrid`: reciprocal rank fusion of 1 and 2 (k = 60, the usual constant; equal weights);
  4. `rerank`: the hybrid's top 20 reordered by the cross-encoder BAAI/bge-reranker-base.
- **Read-outs:** key rules found at top 8 (of 15) and at top 5 and 12; each key rule's best rank (median).
"""

import json
import re
from pathlib import Path

import numpy as np

from strokecopilot.config import FIXTURES_DIR

K, RRF_K, RERANK_POOL = 8, 60, 20
TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
RERANKER = "BAAI/bge-reranker-base"


def tokens(text):
    return TOKEN.findall(text.lower())


def key_rules():
    return json.loads((FIXTURES_DIR / "expected_rules.json").read_text())["fixtures"]


def queries():
    """{fixture: {treatment: query text}}."""
    from strokecopilot.pipeline import end_to_end as e2e
    out = {}
    for case in e2e.load_cases("fixtures"):
        v = case["vision"]
        imaging = (f"Imaging: haemorrhage {'present' if v['haemorrhage']['present'] else 'absent'}; core {v['core_ml']} mL; "
                   f"Tmax > 6 s {v['penumbra_ml']} mL; mismatch ratio {v['mismatch_ratio']}; mismatch volume "
                   f"{v['mismatch_volume_ml']} mL; occlusion {v['occlusion_site']}; ASPECTS {v['aspects']}.")
        out[case["id"]] = {t: f"{case['note']}\n{imaging}\nIs {t} indicated?" for t in ("thrombolysis", "thrombectomy")}
    return out


def rrf(*rankings, k=RRF_K):
    """Reciprocal rank fusion of ranked id lists."""
    score = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking, 1):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + r)
    return [pid for pid, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


class Methods:
    def __init__(self, corpus, embed=None, reranker=None):
        from rank_bm25 import BM25Okapi

        from strokecopilot.nlp.corpus.index import Embedder, Index
        self.ids = list(corpus)
        self.texts = [corpus[i]["text"] for i in self.ids]
        self.dense = Index(corpus, embed or Embedder())
        self.bm25 = BM25Okapi([tokens(t) for t in self.texts])
        self.reranker = reranker

    def bge(self, q):
        return [pid for pid, _ in self.dense.search(q, k=len(self.ids))]

    def bm25_rank(self, q):
        scores = self.bm25.get_scores(tokens(q))
        return [self.ids[i] for i in np.argsort(-scores, kind="stable")]

    def hybrid(self, q):
        return rrf(self.bge(q), self.bm25_rank(q))

    def rerank(self, q):
        pool = self.hybrid(q)[:RERANK_POOL]
        text = dict(zip(self.ids, self.texts))
        scores = self.reranker.predict([(q, text[p]) for p in pool])
        order = [pool[i] for i in np.argsort(-np.asarray(scores), kind="stable")]
        return order + [p for p in self.hybrid(q) if p not in order]


def evaluate(rank_fn, qs, keys, ks=(5, 8, 12)):
    """Key rules found at each k (either decision query of the fixture), and each key rule's best rank."""
    found, best = {k: 0 for k in ks}, []
    for fixture, rules in keys.items():
        rankings = [rank_fn(q) for q in qs[fixture].values()]
        for rule in rules:
            r = min(ranking.index(rule) + 1 for ranking in rankings)
            best.append(r)
            for k in ks:
                found[k] += r <= k
    return {"found": {f"top {k}": found[k] for k in ks}, "of": len(best), "median_best_rank": float(np.median(best)),
            "best_ranks": best}


def main(argv=None):
    import argparse

    from sentence_transformers import CrossEncoder

    from strokecopilot.nlp.corpus import ingest
    ap = argparse.ArgumentParser(description="Task G: retrieval re-check")
    ap.add_argument("--out", default="runs/EXP-79_retrieval_check")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        raise SystemExit("write the run's config.json first (plan §11)")
    m = Methods(ingest.load(), reranker=CrossEncoder(RERANKER))
    qs, keys = queries(), key_rules()
    result = {"experiment": "EXP-79", "exp7_bge_top8": "9/15", "methods": {}}
    for name, fn in (("bge", m.bge), ("bm25", m.bm25_rank), ("hybrid", m.hybrid), ("rerank", m.rerank)):
        result["methods"][name] = evaluate(fn, qs, keys)
    result["bge_reproduces_exp7"] = result["methods"]["bge"]["found"]["top 8"] == 9
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: {"found": v["found"], "median_best_rank": v["median_best_rank"]}
                      for k, v in result["methods"].items()}, indent=1), "\nbge reproduces EXP-7:", result["bge_reproduces_exp7"])


if __name__ == "__main__":
    main()
