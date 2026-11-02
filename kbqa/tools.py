"""Agent 工具集：每个工具 = JSON Schema 声明 + 本地执行函数。

数据均为内存模拟数据，后续接入真实 API 时只需替换 TOOLS 中对应函数实现，
Schema 与注册机制不变。
"""

from __future__ import annotations

import json
from typing import Callable

from kbqa.logger import get_logger

logger = get_logger("tools")

# ---------------- 模拟数据源 ----------------

MOCK_ORDERS = {
    "20260901001": {
        "status": "已发货",
        "item": "无线降噪耳机 Pro",
        "amount": 899.0,
        "logistics": "顺丰 SF1390000123456，已到上海转运中心，预计明天送达",
        "created": "2026-09-01",
    },
    "20260902002": {
        "status": "已签收",
        "item": "机械键盘 K87",
        "amount": 399.0,
        "logistics": "京东 JDV002123456789，已签收（代收点）",
        "created": "2026-09-02",
    },
    "20260903003": {
        "status": "待发货",
        "item": "智能手表 S6",
        "amount": 1599.0,
        "logistics": "未发货，预计 48 小时内发出",
        "created": "2026-09-03",
    },
}

MOCK_TICKETS = {
    "TK-8801": {
        "type": "退货退款",
        "status": "审核通过，等待用户寄回",
        "created": "2026-09-03",
        "order_id": "20260901001",
        "latest": "审核已通过，退货地址已发送至您的 App 消息，请尽快寄回商品。",
    },
    "TK-8802": {
        "type": "维修申请",
        "status": "维修中",
        "created": "2026-09-04",
        "order_id": "20260902002",
        "latest": "设备已收到，工程师检测为主板故障，维修中，预计 9 月 9 日完成。",
    },
}

MOCK_LOGISTICS = {
    "SF1234567890": {
        "company": "顺丰速运",
        "status": "运输中",
        "trail": ["09-05 20:11 已揽收", "09-06 08:30 离开广州中心", "09-07 06:15 到达上海转运中心"],
    },
    "JDV002123456789": {
        "company": "京东物流",
        "status": "已签收",
        "trail": ["09-06 10:00 商家发货", "09-07 14:20 到达北京站点", "09-07 18:05 已签收（代收点）"],
    },
}


# ---------------- 工具实现 ----------------

def query_order(order_id: str) -> dict:
    order = MOCK_ORDERS.get(order_id.strip())
    if not order:
        return {"found": False, "message": f"未找到订单 {order_id}，请核对订单号（示例：20260901001）"}
    return {"found": True, **order}


def query_ticket(ticket_id: str) -> dict:
    ticket = MOCK_TICKETS.get(ticket_id.upper())
    if not ticket:
        return {"found": False, "message": f"未找到售后工单 {ticket_id}，请核对工单号（示例：TK-8801）"}
    return {"found": True, **ticket}


def query_logistics(tracking_no: str) -> dict:
    info = MOCK_LOGISTICS.get(tracking_no)
    if not info:
        return {"found": False, "message": f"未找到物流单 {tracking_no}，请核对运单号（示例：SF1234567890）"}
    return {"found": True, **info}


# ---------------- 工具注册表 ----------------

TOOLS: list[dict] = [
    {
        "name": "query_order",
        "description": "根据订单号查询用户的订单信息：状态、商品、金额、物流、下单时间。用户询问'我的订单''买到的东西发货了吗'等时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号，如 20260901001"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "query_ticket",
        "description": "根据工单号查询用户的售后工单进度（退货、维修等）。用户询问'我的售后单''退货申请处理得怎么样'等时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "售后工单号，如 TK-8801"}
            },
            "required": ["ticket_id"],
        },
    },
    {
        "name": "query_logistics",
        "description": "根据运单号查询实时物流轨迹。用户询问'包裹到哪了''物流信息'等时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "tracking_no": {"type": "string", "description": "快递运单号，如 SF1234567890"}
            },
            "required": ["tracking_no"],
        },
    },
]

_EXECUTORS: dict[str, Callable] = {
    "query_order": query_order,
    "query_ticket": query_ticket,
    "query_logistics": query_logistics,
}


def execute(name: str, arguments: dict) -> dict:
    """安全执行工具：未知工具/异常都不抛出，统一返回错误信息给 LLM。"""
    safe_args = args_obj_safe(arguments)
    logger.info(f"[tool] 执行 {name} args={json.dumps(safe_args, ensure_ascii=False)}")
    fn = _EXECUTORS.get(name)
    if fn is None:
        return {"error": f"未知工具: {name}"}
    try:
        result = fn(**safe_args)
        logger.info(f"[tool] {name} 结果: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result
    except Exception as e:
        logger.error(f"[tool] {name} 执行异常: {type(e).__name__}: {e}")
        return {"error": f"工具执行失败: {type(e).__name__}: {e}"}


def args_obj_safe(args: dict) -> dict:
    return args if isinstance(args, dict) else {}


def tools_as_prompt() -> str:
    """把工具清单渲染成 ReAct Prompt 用的文本。"""
    lines = []
    for t in TOOLS:
        params = ", ".join(t["parameters"].get("properties", {}).keys())
        lines.append(f"- {t['name']}({params}): {t['description']}")
    return "\n".join(lines)
