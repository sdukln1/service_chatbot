from __future__ import annotations

import time

from kbqa.config import CONFIG
from kbqa.embedder import Embedder
from kbqa.logger import get_logger
from kbqa.store import VectorStore

logger = get_logger("retriever")


class Retriever:
    """纯向量检索（Phase 0 基线，用于对比实验）。"""

    def __init__(self, embedder=None, store=None):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()

    def retrieve(
        self, question: str, query_embedding: list[float] | None = None
    ) -> list[dict]:
        t_start = time.time()
        t0 = time.time()
        qe = (
            query_embedding
            if query_embedding is not None
            else self.embedder.embed_query(question)
        )
        t_embed = time.time() - t0

        t0 = time.time()
        hits = self.store.search(qe)
        t_search = time.time() - t0

        result = [h for h in hits if h["score"] >= CONFIG.score_threshold]
        logger.info(
            f"[retrieve/vector] embed={t_embed:.2f}s search={t_search:.2f}s "
            f"total={time.time() - t_start:.2f}s 候选={len(hits)} 返回={len(result)}"
        )
        return result

    @staticmethod
    def format_context(hits: list[dict]) -> str:
        if not hits:
            return ""
        blocks = []
        for i, hit in enumerate(hits, 1):
            source = hit["metadata"].get("source", "unknown")
            score = hit["score"]
            blocks.append(f"[{i}] (来源: {source}, 相似度: {score:.2f})\n{hit['text']}")
        return "\n\n".join(blocks)


class HybridRetriever:
    """向量检索 + BM25 关键词检索 → 加权 RRF 融合 → Rerank 精排。"""

    def __init__(self, embedder=None, store=None, reranker=None):
        from kbqa.bm25 import BM25Index

        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.bm25 = BM25Index()
        self._reranker = reranker
        self.rebuild_bm25()

    def rebuild_bm25(self) -> None:
        data = self.store.collection.get(include=["documents", "metadatas"])
        chunks = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
        self.bm25.build(chunks)

    @property
    def reranker(self):
        if self._reranker is None:
            from kbqa.reranker import LLMReranker

            self._reranker = LLMReranker()
        return self._reranker

    def retrieve(
        self, question: str, query_embedding: list[float] | None = None
    ) -> list[dict]:
        t_start = time.time()

        t0 = time.time()
        qe = (
            query_embedding
            if query_embedding is not None
            else self.embedder.embed_query(question)
        )
        t_embed = time.time() - t0

        t0 = time.time()
        vec_hits = self.store.search(qe, CONFIG.hybrid_candidates)
        for h in vec_hits:
            h["vector_score"] = h["score"]
        t_vector = time.time() - t0

        t0 = time.time()
        bm_hits = self.bm25.search(question, CONFIG.hybrid_candidates)
        t_bm25 = time.time() - t0

        t0 = time.time()
        fused = self._rrf_fuse(vec_hits, bm_hits)
        t_rrf = time.time() - t0

        timing = (
            f"embed={t_embed:.2f}s vector={t_vector:.2f}s bm25={t_bm25:.2f}s "
            f"rrf={t_rrf:.2f}s"
        )

        if CONFIG.use_rerank:
            t0 = time.time()
            fused = self.reranker.rerank(question, fused)
            t_rerank = time.time() - t0
            timing += f" rerank={t_rerank:.2f}s"
            result = [h for h in fused if h["score"] >= CONFIG.rerank_threshold]
        else:
            result = fused[: CONFIG.top_k]

        logger.info(
            f"[retrieve/hybrid] {timing} total={time.time() - t_start:.2f}s "
            f"融合候选={len(fused)} 返回={len(result)}"
        )
        return result

    def _rrf_fuse(self, vec_hits: list[dict], bm_hits: list[dict]) -> list[dict]:
        fused: dict[str, dict] = {}

        for rank, hit in enumerate(vec_hits, 1):
            key = hit["text"]
            entry = fused.setdefault(key, {**hit, "rrf": 0.0})
            entry["rrf"] += CONFIG.vector_weight / (CONFIG.rrf_k + rank)

        for rank, hit in enumerate(bm_hits, 1):
            key = hit["text"]
            entry = fused.setdefault(key, {**hit, "rrf": 0.0})
            entry["rrf"] += CONFIG.bm25_weight / (CONFIG.rrf_k + rank)

        results = list(fused.values())
        for r in results:
            r["score"] = r["rrf"]
        results.sort(key=lambda h: h["score"], reverse=True)
        return results
