from __future__ import annotations

import json
import re
import time

from kbqa.config import CONFIG
from kbqa.generator import create_client
from kbqa.logger import get_logger

logger = get_logger("reranker")

RERANK_PROMPT = """你是搜索相关性评估员。给定一个用户查询和若干候选文档，请为每个文档打一个 0~10 的相关性分数（10 表示完全回答了查询，0 表示完全无关）。

评分标准：
- 文档直接包含查询所需的信息：7~10
- 文档与查询主题相关但信息不完整：3~6
- 文档与查询无关：0~2

只输出 JSON，不要输出其他内容，格式：
{{"scores": [{{"id": 1, "score": 8}}, {{"id": 2, "score": 0}}]}}

【用户查询】
{question}

【候选文档】
{documents}"""


class LLMReranker:
    """用 LLM 对候选文档批量打相关性分（单次调用），替代本地 CrossEncoder。"""

    def __init__(self, model: str | None = None):
        from kbqa.generator import create_client

        # rerank 是锦上添花，用 30s 短超时：LLM 繁忙时快速失败回退 RRF 顺序
        self.client = create_client(read_timeout=30.0)
        self.model = model or CONFIG.llm_model

    def rerank(
        self, question: str, hits: list[dict], top_n: int | None = None
    ) -> list[dict]:
        top_n = top_n or CONFIG.rerank_top_n
        if not hits:
            return []

        documents = "\n\n".join(
            f"[{i}] {h['text'][:500]}" for i, h in enumerate(hits, 1)
        )
        prompt = RERANK_PROMPT.format(question=question, documents=documents)

        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content
            scores = self._parse_scores(raw)
            if not scores:
                logger.warning(f"[rerank] 分数解析失败，回退 RRF 顺序 原始输出: {raw!r}")
        except Exception as e:
            logger.error(f"[rerank] LLM 打分失败，回退 RRF 顺序 error={type(e).__name__}: {e}")
            scores = {}
        logger.info(
            f"[rerank] 完成 耗时={time.time() - start:.2f}s 候选={len(hits)} 分数={scores}"
        )

        for i, hit in enumerate(hits, 1):
            hit["score"] = float(scores.get(i, 5.0))

        hits.sort(key=lambda h: h["score"], reverse=True)
        top = hits[:top_n]
        return [h for h in top if h["score"] >= CONFIG.rerank_threshold] or top[:1]

    @staticmethod
    def _parse_scores(text: str) -> dict[int, float]:
        if not text:
            return {}
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
            return {
                int(item["id"]): float(item["score"])
                for item in data.get("scores", [])
                if "id" in item and "score" in item
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}
