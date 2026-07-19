# Miemie-Agent-RAG/app/main.py
import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, StreamingResponse
import json
from app.graph.workflow import create_workflow
from app.services.retriever import get_milvus_retriever


# 1. 必须先定义生命周期函数，这样 FastAPI 实例化时才能找到它
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 检查是否为 Uvicorn 重载主进程
    is_uvicorn_cmd = any("uvicorn" in arg.lower() for arg in sys.argv)
    is_reloader_main_process = is_uvicorn_cmd and os.getenv("UVICORN_LOOP") is None

    if not is_reloader_main_process:
        print("\n=== [Miemie-RAG 预热激活] 正在独占装载 Embedding 模型与向量库句柄... ===")
        get_milvus_retriever()
        print("=== [Miemie-RAG 预热激活] 模型全重就绪！FastAPI 正式对外开放！ ===\n")
    else:
        print("[ℹ️ Uvicorn 提示] 主控重载监听器启动，跳过预热。")

    yield
    print("=== [Miemie-RAG] 服务正在安全关闭... ===")


# 2. 现在实例化 FastAPI，lifespan 已经定义好了
app = FastAPI(
    title="Miemie-Agent-RAG 生产级后端服务",
    description="Based on FastAPI + LangGraph + Milvus + DeepSeek",
    version="1.0.0",
    lifespan=lifespan
)

# 3. 初始化工作流实例
rag_workflow = create_workflow()


# 4. 定义数据模型
class ChatRequest(BaseModel):
    question: str


# 5. 定义接口路由 (放在 app 实例化之后)
@app.get("/", include_in_schema=False)
async def root_to_docs():
    return RedirectResponse(url="/docs")


@app.post("/chat/stream", summary="RAG 流式智能问答接口", tags=["Miemie 核心对话 Agent 模块"])
async def chat_stream_endpoint(request: ChatRequest):
    async def event_generator():
        prev_len = 0
        async for chunk in rag_workflow.astream({"question": request.question}):
            if "generate" in chunk and "answer" in chunk["generate"]:
                full = chunk["generate"]["answer"]
                # 只发送增量部分（新增的 token），而非完整累积文本
                delta = full[prev_len:]
                prev_len = len(full)
                if delta:
                    yield f"data: {json.dumps({'answer': delta}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    print("====== [系统启动] 正在以自定义配置启动 Uvicorn 服务器 ======")
    uvicorn.run(
        "app.main:app",         # 目标应用 (注意这里要用字符串格式，才能兼容 reload)
        host="0.0.0.0",         # 绑定所有网卡，防止被本地防火墙/IPv6误杀
        port=8000,              # 固定端口
        timeout_keep_alive=60,  # ⚡ 核心修改：将 Keep-Alive 超时时间延长至 60 秒（甚至 120 秒）
        reload=False            # 压测时强烈建议将 reload 设为 False，能大幅提升并发稳定性
    )

