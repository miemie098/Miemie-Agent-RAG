from dotenv import load_dotenv
load_dotenv()  # 必须放在最前面，确保节点初始化时能拿到 API Key

from langgraph.graph import StateGraph, END
from app.graph.nodes import GraphState, retrieve_node, generate_node

def create_workflow():
    # 1. 初始化图，传入状态结构
    workflow = StateGraph(GraphState)

    # 2. 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # 3. 设置流转逻辑 (检索 -> 生成 -> 结束)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# 简单的测试运行
if __name__ == "__main__":
    app = create_workflow()
    result = app.astream({"question": "我的项目有哪些技术优势？"})
    print(result["answer"])