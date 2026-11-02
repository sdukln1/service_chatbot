from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.cut_for_search(text) if t.strip()]


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        corpus = [tokenize(c["text"]) for c in chunks]
        corpus = [doc if doc else ["空"] for doc in corpus]
        self._bm25 = BM25Okapi(corpus) if chunks else None

    def search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            hits.append({**self._chunks[i], "score": float(scores[i])})
        return hits
