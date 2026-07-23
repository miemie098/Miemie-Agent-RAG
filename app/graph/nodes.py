# Miemie-Agent-RAG/app/graph/nodes.py
import logging
import os
from typing import TypedDict

from app.services.retriever import get_milvus_retriever
from langchain_openai import ChatOpenAI

logger = logging.getLogger("miemie-rag.nodes")

# 对话历史最多保留的轮数
MAX_HISTORY_TURNS = 5


class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]


# LLM 客户端懒加载单例
_global_llm_instance = None


def _get_llm():
    """延迟初始化 LLM 客户端"""
    global _global_llm_instance
    if _global_llm_instance is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未检测到 DEEPSEEK_API_KEY 环境变量。"
                "请在项目根目录的 .env 文件中配置 API Key。"
            )
        _global_llm_instance = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com",
        )
    return _global_llm_instance


def _build_messages(state: GraphState, system_prompt: str) -> list:
    """
    拼接系统指令 + 历史对话 + 当前问题，构建 LLM 输入消息列表。
    这是多轮对话能力的核心入口：所有生成节点都通过此函数统一构建 prompt。
    """
    messages = [{"role": "system", "content": system_prompt}]

    # 裁剪最近 N 轮历史，防止上下文溢出
    history = state.get("messages", []) or []
    recent = history[-(MAX_HISTORY_TURNS * 2):]  # 每轮 = user + assistant
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 当前问题
    messages.append({"role": "user", "content": state["question"]})
    return messages


def retrieve_node(state: GraphState):
    """从 Milvus 检索相关知识"""
    retriever = get_milvus_retriever()
    docs = retriever.invoke(state["question"])
    context = "\n".join([doc.page_content for doc in docs])
    return {"context": context}


def generate_node(state: GraphState):
    """调用 DeepSeek 生成答案（同步，完整返回）"""
    system_prompt = (
        "你是一个知识助手。请基于以下参考资料回答用户问题。"
        "如果参考资料不足以回答，请如实说明。\n\n"
        f"参考资料:\n{state['context']}"
    )
    llm_messages = _build_messages(state, system_prompt)

    try:
        response = _get_llm().invoke(llm_messages)
        answer = response.content
    except Exception as e:
        logger.error(f"DeepSeek API 调用异常: {e}")
        answer = "【系统提示】大模型服务暂时不可用，请稍后重试。"

    # 将本轮 Q&A 追加到历史消息，供 checkpointer 持久化
    history = list(state.get("messages") or [])
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": answer})
    return {"answer": answer, "messages": history}


async def generate_node_stream(state: GraphState):
    """
    调用 DeepSeek 流式生成答案。
    - astream 模式：调用方通过增量获取每个 token
    - ainvoke 模式：LangGraph 取最后一次 yield 的累积结果
    """
    system_prompt = (
        "你是一个知识助手。请基于以下参考资料回答用户问题。"
        "如果参考资料不足以回答，请如实说明。\n\n"
        f"参考资料:\n{state['context']}"
    )
    llm_messages = _build_messages(state, system_prompt)

    try:
        full_answer = ""
        async for chunk in _get_llm().astream(llm_messages):
            full_answer += chunk.content
            yield {"answer": full_answer}
    except Exception as e:
        logger.error(f"DeepSeek API 流式调用异常: {e}")
        full_answer = "【系统提示】大模型服务暂时不可用，请稍后重试。"
        yield {"answer": full_answer}

    # 流结束后，将本轮 Q&A 追加到消息历史
    history = list(state.get("messages") or [])
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": full_answer})
    yield {"answer": full_answer, "messages": history}
