# Miemie-Agent-RAG/app/main.py
import json
import logging
import os
import sys

import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, StreamingResponse
from app.graph.workflow import create_workflow
from app.services.retriever import get_milvus_retriever

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("miemie-rag")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    messages: list[ChatMessage] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    is_uvicorn_cmd = any("uvicorn" in arg.lower() for arg in sys.argv)
    is_reloader_main_process = is_uvicorn_cmd and os.getenv("UVICORN_LOOP") is None

    if not is_reloader_main_process:
        logger.info("正在预热：加载 Embedding 模型与向量库...")
        get_milvus_retriever()
        logger.info("预热完成，服务就绪")
    else:
        logger.debug("Uvicorn reloader 主进程，跳过预热")

    yield
    logger.info("服务正在关闭...")


app = FastAPI(
    title="Miemie-Agent-RAG",
    description="基于 LangGraph + Milvus + DeepSeek 的混合检索增强生成系统",
    version="1.0.0",
    lifespan=lifespan,
)

rag_workflow = create_workflow()


# ── 路由 ──────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root_to_docs():
    return RedirectResponse(url="/docs")


@app.get("/health", summary="健康检查")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat", summary="RAG 问答接口（非流式）", tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """非流式问答，返回完整答案。支持多轮对话。"""
    state_input = {
        "question": request.question,
        "messages": [m.model_dump() for m in (request.messages or [])],
        "context": "",
        "answer": "",
    }
    result = await rag_workflow.ainvoke(state_input)
    return {"answer": result.get("answer", "")}


@app.post("/chat/stream", summary="RAG 流式问答接口（SSE）", tags=["Chat"])
async def chat_stream_endpoint(request: ChatRequest):
    """流式问答，SSE 协议逐 token 推送。支持多轮对话。"""
    async def event_generator():
        state_input = {
            "question": request.question,
            "messages": [m.model_dump() for m in (request.messages or [])],
            "context": "",
            "answer": "",
        }
        prev_len = 0
        async for chunk in rag_workflow.astream(state_input):
            if "generate" in chunk and "answer" in chunk["generate"]:
                full = chunk["generate"]["answer"]
                delta = full[prev_len:]
                prev_len = len(full)
                if delta:
                    yield f"data: {json.dumps({'answer': delta}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    logger.info("正在以自定义配置启动 Uvicorn 服务器")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=60,
        reload=False,
    )
