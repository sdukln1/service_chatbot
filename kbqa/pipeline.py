from __future__ import annotations

import json
import time

from kbqa.cache import SemanticCache
from kbqa.chunker import chunk_text
from kbqa.config import CONFIG
from kbqa.embedder import Embedder
from kbqa.generator import Generator
from kbqa.loader import load_documents
from kbqa.logger import get_logger
from kbqa.retriever import HybridRetriever, Retriever
from kbqa.store import VectorStore

logger = get_logger("pipeline")


class RAGPipeline:
    def __init__(self):
        start = time.time()
        self.embedder = Embedder()
        self.store = VectorStore()
        if CONFIG.use_hybrid:
            self.retriever = HybridRetriever(self.embedder, self.store)
        else:
            self.retriever = Retriever(self.embedder, self.store)
        self.cache = SemanticCache()
        self.generator = None
        self._agent = None
        from kbqa.observability import Observability

        self.observability = Observability()
        logger.info(
            f"RAGPipeline 初始化完成 耗时={time.time() - start:.2f}s "
            f"模式={'混合检索' if CONFIG.use_hybrid else '纯向量'} "
            f"rerank={'开' if CONFIG.use_rerank else '关'} "
            f"缓存={'开' if CONFIG.cache_enabled else '关'} "
            f"知识库块数={self.store.count()}"
        )

    def _get_generator(self) -> Generator:
        if self.generator is None:
            self.generator = Generator()
        return self.generator

    def ingest(self) -> int:
        documents = load_documents()
        if not documents:
            logger.warning("知识库目录为空，请先在 data/knowledge 中放入文档")
            return 0

        all_chunks = []
        for doc in documents:
            for chunk in chunk_text(doc.text):
                all_chunks.append({"text": chunk, "metadata": doc.metadata})
        logger.info(f"[ingest] 加载 {len(documents)} 个文档，切分出 {len(all_chunks)} 个文本块")

        logger.info(f"[ingest] 使用 {CONFIG.embedding_model} 生成向量...")
        start = time.time()
        embeddings = self.embedder.embed_documents([c["text"] for c in all_chunks])
        logger.info(f"[ingest] 向量生成完成 耗时={time.time() - start:.2f}s")
        added = self.store.add(all_chunks, embeddings)
        logger.info(f"[ingest] 已写入向量库，当前共 {self.store.count()} 个文本块")
        if CONFIG.use_hybrid and hasattr(self.retriever, "rebuild_bm25"):
            self.retriever.rebuild_bm25()
            logger.info("[ingest] BM25 索引已重建")
        self.cache.clear()
        return added

    def query(self, question: str, model: str | None = None) -> dict:
        logger.info(f"[query] 收到提问 model={model or CONFIG.llm_model} question={question!r}")
        start = time.time()
        hits = self.retriever.retrieve(question)
        logger.info(f"[query] 检索完成 耗时={time.time() - start:.2f}s 命中 {len(hits)} 条")
        context = Retriever.format_context(hits)
        generator = self._get_generator()
        answer = generator.generate(question, context, model=model)
        logger.info(f"[query] 全链路完成 总耗时={time.time() - start:.2f}s")
        return {"answer": answer, "sources": hits}

    def stream_query(self, question: str, model: str | None = None, enable_agent: bool | None = None):
        resolved_model = model or CONFIG.llm_model
        logger.info(f"[stream] 收到提问 model={resolved_model} question={question!r}")
        start = time.time()
        use_agent = CONFIG.agent_enabled if enable_agent is None else enable_agent
        trace = self.observability.start_trace(question, resolved_model)

        # 0. 意图判断（Phase 4）：实时查询类走工具 Agent，知识咨询类走 RAG
        if use_agent:
            from kbqa.agent import Agent

            t0 = time.time()
            agent = self._get_agent()
            intent = agent.classify_intent(question, model=model)
            logger.info(f"[stream] 意图判断耗时={time.time() - t0:.2f}s 结果={intent}")
            trace.intent = intent
            trace.timings["intent"] = round(time.time() - t0, 2)
            if intent == "tool_use":
                yield from self._run_agent_stream(question, model, trace)
                return
            if intent == "chitchat":
                answer = "您好，我是智能客服，专注商品退货、保修、支付、物流等问题。请问有什么可以帮您？"
                trace.answer = answer
                self.observability.flush(trace)
                yield answer
                return

        # 1. 语义缓存查询
        query_embedding = None
        if CONFIG.cache_enabled:
            t0 = time.time()
            query_embedding = self.embedder.embed_query(question)
            logger.info(f"[stream] 查询向量化 耗时={time.time() - t0:.2f}s")
            cached = self.cache.lookup(question, query_embedding, resolved_model)
            if cached:
                logger.info(f"[stream] 缓存命中 总耗时={time.time() - start:.2f}s")
                answer = cached["answer"] + "\n" + cached["sources"]
                trace.answer = answer
                trace.metadata["cache_hit"] = True
                self.observability.flush(trace)
                yield answer
                return

        # 2. 检索（分项耗时由 retriever 内部日志输出）
        t0 = time.time()
        hits = self.retriever.retrieve(question, query_embedding=query_embedding)
        retrieval_s = time.time() - t0
        trace.timings["retrieval"] = round(retrieval_s, 2)
        logger.info(
            f"[stream] 检索完成 耗时={retrieval_s:.2f}s 命中 {len(hits)} 条: "
            + str([(h["metadata"].get("source"), round(h["score"], 2)) for h in hits])
        )

        if not hits:
            logger.warning("[stream] 检索无命中，返回兜底话术")
            answer = "抱歉，知识库中暂时没有找到相关信息，建议您联系人工客服。"
            trace.answer = answer
            self.observability.flush(trace)
            yield answer
            return

        context = Retriever.format_context(hits)
        trace.metadata["retrieved_sources"] = [h["metadata"].get("source") for h in hits]

        # 3. LLM 流式生成（失败时兜底降级）
        generator = self._get_generator()
        tokens: list[str] = []
        try:
            for token in generator.stream_generate(question, context, model=model):
                tokens.append(token)
                yield token
        except Exception as e:
            logger.error(f"[stream] 生成失败启用兜底 error={type(e).__name__}: {e}")
            if not tokens:
                trace.answer = CONFIG.fallback_answer
                trace.metadata["fallback"] = True
                self.observability.flush(trace)
                yield CONFIG.fallback_answer
            else:
                trace.answer = "".join(tokens) + "（中断）"
                self.observability.flush(trace)
                yield "\n\n（回答生成中断，请重试）"
            return

        # 4. 写入语义缓存
        answer = "".join(tokens)
        sources_line = self._sources_line(hits)
        trace.answer = answer
        trace.timings["total"] = round(time.time() - start, 2)
        if CONFIG.cache_enabled:
            if query_embedding is None:
                query_embedding = self.embedder.embed_query(question)
            self.cache.put(question, query_embedding, answer, sources_line, resolved_model)
        logger.info(f"[stream] 问答全链路完成 总耗时={time.time() - start:.2f}s")
        self.observability.flush(trace)
        yield "\n" + sources_line

    @staticmethod
    def _sources_line(hits: list[dict]) -> str:
        sources = sorted({h["metadata"].get("source", "unknown") for h in hits})
        return "参考来源: " + "、".join(sources)

    def _get_agent(self):
        if self._agent is None:
            from kbqa.agent import Agent

            self._agent = Agent(generator=self._get_generator())
        return self._agent

    def _run_agent_stream(self, question: str, model: str | None, trace=None):
        """工具类问题：执行 ReAct 循环，以事件形式推送工具调用过程与最终答案。

        产出为两种：SSE 事件字符串（tool_call 事件，以 'data: ' 开头，server 透传）
        和普通答案 token 字符串。
        """
        logger.info(f"[agent-stream] 开始工具调用链路 question={question!r}")
        start = time.time()
        agent = self._get_agent()

        def tool_event(step: int, name: str, summary: str) -> str:
            return (
                "data: "
                + json.dumps(
                    {"type": "tool_call", "content": {"step": step, "name": name, "summary": summary}},
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        try:
            result = agent.run(question, model=model)
        except Exception as e:
            logger.error(f"[agent-stream] 工具链路失败启用兜底 error={type(e).__name__}: {e}")
            if trace is not None:
                trace.answer = CONFIG.fallback_answer
                trace.metadata["fallback"] = True
                trace.timings["total"] = round(time.time() - start, 2)
                self.observability.flush(trace)
            yield CONFIG.fallback_answer
            return

        for i, call in enumerate(result.tool_calls, 1):
            r = call["result"]
            if "message" in r:
                summary = r["message"]
            else:
                item = r.get("item") or r.get("company") or r.get("type", "")
                summary = f"已查到，状态：{r.get('status', '')}；{r.get('latest', '')}".strip("；")
            yield tool_event(i, call["name"], summary)

        if trace is not None:
            trace.answer = result.answer
            trace.tool_calls = [
                {"name": c["name"], "args": c["args"]} for c in result.tool_calls
            ]
            trace.timings["total"] = round(time.time() - start, 2)
            self.observability.flush(trace)

        yield result.answer
        logger.info(f"[agent-stream] 完成 总耗时={time.time() - start:.2f}s")
