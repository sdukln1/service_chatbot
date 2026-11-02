from __future__ import annotations

from kbqa.config import CONFIG
from kbqa.logger import get_logger

logger = get_logger("cache")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    """语义缓存：相似问题（cosine >= threshold 且同模型）直接返回缓存答案。"""

    def __init__(self):
        self.threshold = CONFIG.cache_threshold
        self.max_size = CONFIG.cache_max_size
        self._entries: list[dict] = []
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        if self._entries:
            logger.info(f"[cache] 缓存已清空（原 {len(self._entries)} 条，知识库已更新）")
        self._entries.clear()

    def lookup(
        self, question: str, query_embedding: list[float], model: str
    ) -> dict | None:
        if not self._entries:
            self.misses += 1
            return None

        # 缓存跨模型共享：答案由同一知识库上下文生成，内容与模型无关
        best, best_sim = None, 0.0
        for entry in self._entries:
            sim = _cosine(query_embedding, entry["embedding"])
            if sim > best_sim:
                best, best_sim = entry, sim

        if best is not None and best_sim >= self.threshold:
            self.hits += 1
            logger.info(
                f"[cache] 命中 sim={best_sim:.3f} 命中问题={best['question']!r} "
                f"(由模型 {best['model']} 生成, 当前请求模型={model}) "
                f"(hits={self.hits} misses={self.misses})"
            )
            return best
        if best is not None:
            logger.info(
                f"[cache] 未命中 最相似 sim={best_sim:.3f} < 阈值 {self.threshold}"
            )
        else:
            self.misses += 1
        return None

    def put(
        self,
        question: str,
        query_embedding: list[float],
        answer: str,
        sources: str,
        model: str,
    ) -> None:
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)
        self._entries.append(
            {
                "question": question,
                "embedding": query_embedding,
                "answer": answer,
                "sources": sources,
                "model": model,
            }
        )
        logger.info(f"[cache] 已缓存 共{len(self._entries)}条 question={question!r}")
