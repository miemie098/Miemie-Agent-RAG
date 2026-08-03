# Miemie-Agent-RAG/app/graph/workflow.py
import logging
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from app.graph.nodes import (
    GraphState,
    retrieve_node,
    generate_node,
    generate_node_stream,
)

logger = logging.getLogger("miemie-rag.workflow")


def create_workflow(streaming: bool = True, checkpointer=None):
    """
    构建 LangGraph 检索-生成工作流。

    Args:
        streaming: True 使用流式生成节点，False 使用同步节点。
        checkpointer: 可选，LangGraph Checkpointer 实例（如 SqliteSaver / PostgresSaver），
                      用于跨请求持久化对话状态。传入后可通过 thread_id 恢复会话。
                      生产环境推荐 AsyncPostgresSaver + 集中式 PostgreSQL。
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node(
        "generate", generate_node_stream if streaming else generate_node
    )

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=checkpointer)
