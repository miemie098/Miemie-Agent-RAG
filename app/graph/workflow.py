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


def create_workflow(streaming: bool = True):
    """
    构建 LangGraph 检索-生成工作流。

    Args:
        streaming: True 使用流式生成节点，False 使用同步节点。
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node(
        "generate", generate_node_stream if streaming else generate_node
    )

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
