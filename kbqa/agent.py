"""Agent 模块：意图判断 + ReAct 循环（Function Calling）。

设计（对应 note/test.md 知识点）：
- 意图判断（Router）：LLM 判断问题是"知识问答"（走 RAG）、"实时查询"（走工具）
  还是"闲聊"。RAG 与 Tool Use 互补。
- ReAct：Thought → Action → Observation 循环，LLM 以结构化文本输出工具调用
  指令，应用层解析并安全执行，把 Observation 回填继续推理，直到 Final Answer。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from kbqa.config import CONFIG
from kbqa.generator import Generator, normalize_model
from kbqa.logger import get_logger
from kbqa.tools import execute, tools_as_prompt

logger = get_logger("agent")

INTENT_PROMPT = """你是客服路由器。判断用户的问题属于哪类：

1. knowledge — 知识咨询类：问政策、规则、流程、方法（如"退货政策""保修多久""怎么开发票"）。答案在知识库文档里，不需要查实时数据。
2. tool_use — 实时查询类：查询用户个人的实时数据（订单状态、售后工单进度、物流轨迹），通常包含具体单号。
3. chitchat — 闲聊或与业务无关的问题（打招呼、问你是谁等）。

只输出 JSON：{"intent": "knowledge" 或 "tool_use" 或 "chitchat", "reason": "一句话理由"}

【用户输入】
{question}"""

REACT_SYSTEM = """你是电商客服 Agent，可以调用工具查询用户的实时数据。严格按以下格式输出，不要输出其他内容：

Thought: 分析当前问题，决定下一步做什么
Action: 工具名
Action Input: {"参数名": "值"}

当你认为信息足够时，输出：
Thought: 已获得足够信息
Final Answer: 给用户的最终回答（简洁口语化，直接给出解决方案）

可用工具：
{tools}"""


@dataclass
class AgentResult:
    answer: str
    intent: str
    tool_calls: list = field(default_factory=list)


def _resp_content(response) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def _parse_action(text: str) -> tuple[str | None, str]:
    """从 LLM 输出中解析 Action 与 Action Input。"""
    if not text:
        return None, ""
    name_match = re.search(r"Action:\s*(\w+)", text)
    if not name_match:
        return None, ""
    input_match = re.search(r"Action Input:\s*(\{.*\})", text, re.DOTALL)
    args = input_match.group(1) if input_match else "{}"
    return name_match.group(1).strip(), args


class Agent:
    def __init__(self, generator: Generator | None = None):
        self.generator = generator or Generator()

    def classify_intent(self, question: str, model: str | None = None) -> str:
        resolved = normalize_model(model)
        try:
            resp = self.generator.client.chat.completions.create(
                model=resolved,
                messages=[
                    {"role": "system", "content": "你是意图分类器，只输出 JSON。"},
                    # 注意用 replace 而非 format：Prompt 中含 JSON 花括号示例
                    {"role": "user", "content": INTENT_PROMPT.replace("{question}", question)},
                ],
                temperature=0,
                max_tokens=64,
            )
            raw = _resp_content(resp)
            logger.info(f"[agent] 意图判断原始输出: {raw!r}")
            intent = _parse_intent(raw)
            if intent:
                logger.info(f"[agent] 意图={intent} question={question!r}")
                return intent
            logger.warning(f"[agent] 意图 JSON 解析失败，默认 knowledge: {raw!r}")
        except Exception as e:
            logger.error(f"[agent] 意图判断失败，默认走知识问答: {type(e).__name__}: {e}")
        return "knowledge"

    def run(self, question: str, model: str | None = None) -> AgentResult:
        """ReAct 循环：Thought → Action → Observation → ... → Final Answer。"""
        resolved = normalize_model(model)
        logger.info(f"[agent] ReAct 开始 model={resolved} question={question!r}")

        messages = [
            {"role": "system", "content": REACT_PROMPT},
            {"role": "user", "content": question},
        ]

        tool_calls: list[dict] = []

        for step in range(1, CONFIG.agent_max_steps + 1):
            t0 = time.time()
            try:
                response = self.generator._create_with_retry(
                    resolved,
                    model=resolved,
                    messages=messages,
                    temperature=0,
                    max_tokens=512,
                )
            except Exception as e:
                logger.error(f"[agent] LLM 调用失败: {type(e).__name__}: {e}")
                raise
            text = _resp_content(response)
            logger.info(f"[agent] 第{step}轮 LLM 耗时={time.time() - t0:.2f}s 输出: {text!r}")

            if "Final Answer" in text:
                answer = _extract_final(text)
                logger.info(f"[agent] ReAct 完成 共{len(tool_calls)}次工具调用")
                return AgentResult(answer=answer, intent="tool_use", tool_calls=tool_calls)
            action_name, action_args = _parse_action(text)
            if action_name:
                result = execute(action_name, _safe_json(action_args))
                tool_calls.append(
                    {"name": action_name, "args": _safe_json(action_args), "result": result}
                )
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Observation: {json.dumps(result, ensure_ascii=False)}\n"
                            "请继续。若已获得足够信息，输出 Final Answer。"
                        ),
                    }
                )
                continue

            if text:
                return AgentResult(answer=text.strip(), intent="tool_use", tool_calls=tool_calls)

        logger.warning("[agent] 达到最大步数，基于已有 Observations 收尾")
        messages.append(
            {
                "role": "user",
                "content": "已达到工具调用上限，请根据已获得的 Observations 直接输出 Final Answer。",
            }
        )
        response = self.generator._create_with_retry(
            resolved, model=resolved, messages=messages, temperature=0.2, max_tokens=512
        )
        return AgentResult(
            answer=_extract_final(_resp_content(response)),
            intent="tool_use",
            tool_calls=tool_calls,
        )


def _extract_final(text: str) -> str:
    """从 LLM 输出中提取 Final Answer；无标记则清理 Thought/Action 行后返回。"""
    if not text:
        return ""
    if "Final Answer:" in text:
        return text.split("Final Answer:", 1)[1].strip()
    # 收尾响应可能仍带 Thought 行，逐行过滤掉推理过程
    lines = [
        line
        for line in text.splitlines()
        if not re.match(r"^(Thought|Action|Action Input):", line.strip())
    ]
    return "\n".join(lines).strip()


REACT_PROMPT = REACT_SYSTEM.replace("{tools}", tools_as_prompt())


def _parse_intent(raw: str) -> str | None:
    """从 LLM 输出解析意图，兼容多种脏输出（引号键、单引号、多余文本）。"""
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        # 兜底：直接在文本里找意图关键词
        for keyword in ("tool_use", "chitchat", "knowledge"):
            if keyword in raw.lower():
                return keyword
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        # 去掉键值上多余引号再试（LLM 偶发输出 {"intent": "tool_use"} 但键带引号）
        cleaned = match.group().replace('""', '"')
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            for keyword in ("tool_use", "chitchat", "knowledge"):
                if keyword in raw:
                    return keyword
            return None
    intent = str(data.get("intent", "")).strip().strip('"').lower()
    return intent if intent in {"knowledge", "tool_use", "chitchat"} else None


def _safe_json(text: str) -> dict:
    try:
        data = json.loads(text) if text and text.strip() else {}
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
