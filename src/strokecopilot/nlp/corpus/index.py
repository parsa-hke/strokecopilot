"""Passage embeddings for the verifier's candidate passages. Plan §8.3, DEC-19; docs/HALLUCINATION_METRIC.md (c).

The reasoner is not served from here: it gets the whole rulebook (`ingest.prompt_block()`), because top-k retrieval
missed key rules in EXP-7. The verifier still needs candidates for a rule claim: the passages it cites plus the top 5
by similarity above τ. `Index.search(text, k)` has the `retriever(text, k)` signature `trace.Inputs` expects.

Needs: pip install "strokecopilot[retrieval]"
"""

import numpy as np

from strokecopilot.nlp.corpus import ingest

MODEL = "BAAI/bge-base-en-v1.5"
# bge v1.5 recommends this instruction on queries (not passages) for query-to-passage retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """sentence-transformers wrapper returning L2-normalised float32 rows. Loaded on first use."""

    def __init__(self, model=MODEL, device=None):
        self.model_name, self.device, self._model = model, device, None

    def __call__(self, texts, query=False):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        if query:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        out = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)


class Index:
    def __init__(self, corpus, embed):
        """corpus: {id: record with 'text'} (ingest.load()); embed(texts, query=bool) -> normalised rows."""
        self.corpus, self.embed = corpus, embed
        self.ids = list(corpus)
        self.vectors = embed([corpus[i]["text"] for i in self.ids], query=False)

    @classmethod
    def from_corpus(cls, embed=None, path=ingest.PASSAGES):
        return cls(ingest.load(path), embed or Embedder())

    def search(self, text, k=5):
        """[(id, cosine score)] best first."""
        scores = self.embed([text], query=True)[0] @ self.vectors.T
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    __call__ = search
