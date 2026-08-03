# Miemie-Agent-RAG/app/main.py
import json
import logging
import os
import sys
import uuid

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
    thread_id: str | None = None  # 会话 ID，传入已有可恢复历史，不传则自动生成


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


# ── Checkpointer 初始化 ──────────────────────────────
# 模块级变量，在 lifespan 中赋值，供路由函数使用
_rag_workflow = None
_checkpointer_ctx = None  # context manager，用于 shutdown 清理


def _get_workflow():
    """获取已编译的 RAG 工作流（仅在 lifespan 完成后有效）"""
    assert _rag_workflow is not None, "工作流尚未初始化，请等待服务就绪"
    return _rag_workflow


def _init_checkpointer_sqlite():
    """初始化本地 SQLite Checkpointer（开发/单机模式）

    Returns:
        (checkpointer, context_manager) — checkpointer 是 BaseCheckpointSaver 实例，
        context_manager 用于应用关闭时清理资源。
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints.db"
    )
    logger.info("使用 SqliteSaver（本地会话状态: %s）", db_path)
    ctx = SqliteSaver.from_conn_string(db_path)
    return ctx.__enter__(), ctx


async def _init_checkpointer_postgres(database_url: str):
    """初始化 PostgreSQL Checkpointer（生产/多副本模式）

    使用 AsyncPostgresSaver 配合 FastAPI 异步上下文，
    通过集中式 PostgreSQL 实现跨 Pod 的会话状态共享。

    Returns:
        (checkpointer, context_manager)
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    logger.info("使用 AsyncPostgresSaver（集中式会话状态 → %s）",
                _mask_connection_string(database_url))
    ctx = AsyncPostgresSaver.from_conn_string(database_url)
    checkpointer = await ctx.__aenter__()
    await checkpointer.setup()
    return checkpointer, ctx


def _mask_connection_string(url: str) -> str:
    """隐藏数据库连接串中的密码部分，用于日志输出"""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理

    启动阶段：
      1. 按 DATABASE_URL 环境变量选择 Checkpointer（Postgres / SQLite）
      2. 编译并缓存 LangGraph 工作流
      3. 预热 Embedding 模型与向量库

    关闭阶段：
      1. 退出 Checkpointer context manager，释放连接池
    """
    global _rag_workflow, _checkpointer_ctx

    is_uvicorn_cmd = any("uvicorn" in arg.lower() for arg in sys.argv)
    is_reloader_main_process = is_uvicorn_cmd and os.getenv("UVICORN_LOOP") is None

    # ── 1. 初始化 Checkpointer ──
    database_url = os.getenv("DATABASE_URL")
    checkpointer = None
    _checkpointer_ctx = None

    if database_url:
        try:
            checkpointer, _checkpointer_ctx = await _init_checkpointer_postgres(database_url)
        except ImportError:
            logger.warning(
                "DATABASE_URL 已设置但 langgraph-checkpoint-postgres 未安装，"
                "回退到 SqliteSaver"
            )
        except Exception as exc:
            logger.error(
                "PostgresSaver 初始化失败 (%s)，回退到 SqliteSaver", exc
            )

    if checkpointer is None:
        checkpointer, _checkpointer_ctx = _init_checkpointer_sqlite()

    # ── 2. 编译工作流 ──
    _rag_workflow = create_workflow(checkpointer=checkpointer)
    logger.info("LangGraph 工作流编译完成")

    # ── 3. 预热 ──
    if not is_reloader_main_process:
        logger.info("正在预热：加载 Embedding 模型与向量库...")
        get_milvus_retriever()
        logger.info("预热完成，服务就绪")
    else:
        logger.debug("Uvicorn reloader 主进程，跳过预热")

    yield

    # ── 关闭阶段 ──
    logger.info("服务正在关闭...")
    if _checkpointer_ctx is not None:
        try:
            if hasattr(_checkpointer_ctx, "__aexit__"):
                await _checkpointer_ctx.__aexit__(None, None, None)
            else:
                _checkpointer_ctx.__exit__(None, None, None)
            logger.info("Checkpointer 资源已释放")
        except Exception as exc:
            logger.warning("Checkpointer 关闭时出现异常: %s", exc)


app = FastAPI(
    title="Miemie-Agent-RAG",
    description="基于 LangGraph + Milvus + DeepSeek 的混合检索增强生成系统",
    version="1.0.0",
    lifespan=lifespan,
)


# ── 路由 ──────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root_to_docs():
    return RedirectResponse(url="/docs")


@app.get("/health", summary="健康检查")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat", summary="RAG 问答接口（非流式）", tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """非流式问答，返回完整答案。支持多轮对话。

    传入 thread_id 可恢复历史会话；不传则自动生成并返回新的 thread_id。
    """
    workflow = _get_workflow()
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 如果已有 checkpoint，用服务端消息历史覆盖客户端传入的 messages
    checkpoint = workflow.get_state(config)
    checkpoint_messages = (
        checkpoint.values.get("messages", [])
        if checkpoint and checkpoint.values
        else None
    )

    state_input = {
        "question": request.question,
        "messages": checkpoint_messages or [m.model_dump() for m in (request.messages or [])],
        "context": "",
        "answer": "",
    }
    result = await workflow.ainvoke(state_input, config)
    return ChatResponse(answer=result.get("answer", ""), thread_id=thread_id)


@app.post("/chat/stream", summary="RAG 流式问答接口（SSE）", tags=["Chat"])
async def chat_stream_endpoint(request: ChatRequest):
    """流式问答，SSE 协议逐 token 推送。支持多轮对话。

    传入 thread_id 可恢复历史会话；不传则自动生成并返回新的 thread_id。
    """
    workflow = _get_workflow()
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 如果已有 checkpoint，用服务端消息历史
    checkpoint = workflow.get_state(config)
    checkpoint_messages = (
        checkpoint.values.get("messages", [])
        if checkpoint and checkpoint.values
        else None
    )

    async def event_generator():
        state_input = {
            "question": request.question,
            "messages": checkpoint_messages or [m.model_dump() for m in (request.messages or [])],
            "context": "",
            "answer": "",
        }
        prev_len = 0
        async for chunk in workflow.astream(state_input, config):
            if "generate" in chunk and "answer" in chunk["generate"]:
                full = chunk["generate"]["answer"]
                delta = full[prev_len:]
                prev_len = len(full)
                if delta:
                    yield f"data: {json.dumps({'answer': delta}, ensure_ascii=False)}\n\n"
        # 流结束，返回 thread_id
        yield f"data: {json.dumps({'thread_id': thread_id}, ensure_ascii=False)}\n\n"
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
