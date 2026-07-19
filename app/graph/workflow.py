from dotenv import load_dotenv
load_dotenv()  # 必须放在最前面，确保节点初始化时能拿到 API Key

from langgraph.graph import StateGraph, END
from app.graph.nodes import GraphState, retrieve_node, generate_node, generate_node_stream

def create_workflow(streaming: bool = True):
    """
    构建 LangGraph 工作流。

    Args:
        streaming: True 使用流式生成节点（token 级输出），
                   False 使用同步生成节点（一次性返回完整答案）。
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node_stream if streaming else generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# 简单的测试运行
if __name__ == "__main__":
    app = create_workflow()
    result = app.astream({"question": "我的项目有哪些技术优势？"})
    print(result["answer"])