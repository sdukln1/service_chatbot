"""LangFuse 可观测性（Phase 5）。

可选开关：config.langfuse_enabled 为 False 或未配置时全部为空操作。
所有 trace 上报失败只打 warning，绝不阻塞问答主链路。

记录内容：每条提问一个 trace（含意图、检索命中、工具调用、答案、分项耗时），
在 LangFuse 控制台可按 trace 查看/检索/统计。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kbqa.config import CONFIG
from kbqa.logger import get_logger

logger = get_logger("langfuse")


@dataclass
class Trace:
    """一次问答的观测数据收集器。"""

    question: str
    model: str
    intent: str | None = None
    answer: str = ""
    tool_calls: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    _trace_obj: object | None = field(default=None, repr=False)
    _start: float = field(default_factory=time.time, repr=False)


class Observability:
    def __init__(self):
        self._client = None
        if CONFIG.langfuse_enabled and CONFIG.langfuse_host:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    host=CONFIG.langfuse_host,
                    public_key=CONFIG.langfuse_public_key,
                    secret_key=CONFIG.langfuse_secret_key,
                )
                logger.info(f"[langfuse] 已启用 host={CONFIG.langfuse_host}")
            except Exception as e:
                logger.warning(f"[langfuse] 初始化失败，本次运行不观测: {type(e).__name__}: {e}")
        else:
            logger.info("[langfuse] 未启用（设置 KBQA_LANGFUSE_ENABLED=true 及 host/keys 开启）")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start_trace(self, question: str, model: str) -> Trace:
        return Trace(question=question, model=model)

    def flush(self, trace: Trace) -> None:
        """问答结束时上报 trace；失败只告警。"""
        if not self.enabled:
            return
        try:
            trace_obj = self._client.trace(
                name="kbqa-chat",
                input={"question": trace.question, "model": trace.model},
                output={"answer": trace.answer},
                metadata={
                    "intent": trace.intent,
                    "tool_calls": trace.tool_calls,
                    "timings": trace.timings,
                    **trace.metadata,
                },
                latency=(time.time() - trace._start),
            )
            self._client.flush()
            logger.info("[langfuse] trace 已上报")
        except Exception as e:
            logger.warning(f"[langfuse] 上报失败（不影响服务）: {type(e).__name__}: {e}")
