from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from kbqa.config import CONFIG
from kbqa.logger import get_logger
from kbqa.pipeline import RAGPipeline

logger = get_logger("server")

app = FastAPI(title="智能客服知识问答")

logger.info("正在启动服务...")
pipeline = RAGPipeline()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "kb_chunks": pipeline.store.count(),
        "llm_model": CONFIG.llm_model,
        "llm_models": CONFIG.llm_models,
        "api_base": CONFIG.api_base,
        "hybrid": CONFIG.use_hybrid,
        "rerank": CONFIG.use_rerank,
        "cache": CONFIG.cache_enabled,
    }


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    model = body.get("model") or None
    if not question:
        return {"error": "question is required"}
    logger.info(f"[api/chat] 收到请求 model={model or CONFIG.llm_model} question={question!r}")

    def event_stream():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        try:
            for item in pipeline.stream_query(
                question, model=model, enable_agent=body.get("enable_agent")
            ):
                if item.startswith("data: "):
                    yield item  # 工具调用事件，直接透传
                else:
                    yield f"data: {json.dumps({'type': 'token', 'content': item}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[api/chat] 处理异常: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'服务异常: {type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/feedback")
async def feedback(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"error": "invalid json body"}
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": body.get("question"),
        "answer": body.get("answer"),
        "model": body.get("model"),
        "rating": body.get("rating"),  # up | down
        "comment": body.get("comment"),
    }
    CONFIG.feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_file = CONFIG.feedback_dir / f"{datetime.now():%Y-%m}.jsonl"
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"[api/feedback] rating={entry['rating']} question={entry['question']!r}")
    return {"ok": True}


# 诊断页（调试用）：/debug 提供前端自检面板。
# 注意：必须先挂载 /debug 再挂载 /，因为 mount("/") 会吞掉其后注册的所有路径
debug_dir = Path(__file__).resolve().parent / "debug"
app.mount("/debug", StaticFiles(directory=str(debug_dir), html=True), name="debug")

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


def main():
    import uvicorn

    uvicorn.run(app, host=CONFIG.server_host, port=CONFIG.server_port)


if __name__ == "__main__":
    main()
