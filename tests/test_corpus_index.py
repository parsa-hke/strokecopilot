"""The verifier's candidate search with a deterministic bag-of-words embedder, and the reasoner's rulebook block.

The real model (bge-base) was measured in EXP-7, not here."""

import json
import re
import zlib
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from strokecopilot.config import FIXTURES_DIR  # noqa: E402
from strokecopilot.nlp.corpus import index as ix  # noqa: E402
from strokecopilot.nlp.corpus import ingest  # noqa: E402
from strokecopilot.nlp.verifier import trace as tr  # noqa: E402

EXPECTED_RULES = Path(FIXTURES_DIR) / "expected_rules.json"


def fake_embed(texts, query=False):
    rows = np.zeros((len(texts), 512), dtype=np.float32)
    for r, text in enumerate(texts):
        for w in re.findall(r"[a-z]+", text.lower()):
            rows[r, zlib.crc32(w.encode()) % 512] += 1
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)


@pytest.fixture(scope="module")
def index():
    return ix.Index.from_corpus(embed=fake_embed)


def test_search_ranks_and_scores(index):
    hits = index.search("acute intracranial haemorrhage on CT contraindication", k=3)
    assert len(hits) == 3 and hits[0][0] == "AHA2026-T8-ich"
    assert all(-1.0001 <= s <= 1.0001 for _, s in hits) and hits[0][1] >= hits[-1][1]


def test_index_is_a_verifier_retriever(index):
    seen = []
    inputs = tr.Inputs(vision={}, note="", facts=None, corpus=ingest.load(), retriever=index, tau=2.0,
                       judge=lambda text, passages, case, applied: seen.append([p["id"] for p in passages]) or {})
    # tau above any cosine: only the cited passage is a candidate
    tr.check_rule("Haemorrhage rules out thrombolysis.", ["AHA2026-T8-ich"], False, inputs)
    assert seen == [["AHA2026-T8-ich"]]


def test_prompt_block_holds_every_passage_and_no_bookkeeping():
    block = ingest.prompt_block()
    passages = ingest.read()
    assert all(f"[{p.id}]" in block and p.text in block for p in passages)
    assert "must_have" not in block and "status" not in block and "draft" not in block


def test_expected_rules_exist():
    expected = json.loads(EXPECTED_RULES.read_text())["fixtures"]
    corpus = ingest.load()
    assert set(expected) == {p.stem for p in Path(FIXTURES_DIR).glob("fx*.json")}
    assert all(pid in corpus for ids in expected.values() for pid in ids)
