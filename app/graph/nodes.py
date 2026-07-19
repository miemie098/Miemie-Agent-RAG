# Miemie-Agent-RAG/app/graph/nodes.py
from typing import TypedDict
from app.services.retriever import get_milvus_retriever
from langchain_openai import ChatOpenAI
import os


# 定义系统的状态流转数据结构
class GraphState(TypedDict):
    question: str
    context: str
    answer: str


# 核心优化1：单例复用，将 LLM 客户端初始化提到全局，高并发下复用底层的连接池
# 避免每个并发请求进来都重复执行 os.getenv 和实例化对象
_global_llm_instance = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base='https://api.deepseek.com'
)


def retrieve_node(state: GraphState):
    """节点1：从 Milvus 检索相关知识"""
    retriever = get_milvus_retriever()
    docs = retriever.invoke(state["question"])
    context = "\n".join([doc.page_content for doc in docs])
    return {"context": context}


def generate_node(state: GraphState):
    """节点2：调用 DeepSeek 生成答案"""
    prompt = f"基于以下知识回答问题:\n{state['context']}\n问题: {state['question']}"

    # 核心优化2：加入大厂级别的异常防御性捕获，防止三方网络 API 故障拖垮整站
    try:
        response = _global_llm_instance.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        print(f"[❌ 生产级报错告警] DeepSeek API 调用发生异常: {str(e)}")
        # 优雅降级返回，不让系统报 500 错误
        return {"answer": "【系统提示】由于当前大模型网络链路抖动，知识大脑暂时无法响应，请稍后再试。"}


async def generate_node_stream(state: GraphState):
    """节点2：异步流式生成器"""
    prompt = f"基于以下知识回答问题:\n{state['context']}\n问题: {state['question']}"

    # 使用 astream 处理流式数据
    async for chunk in _global_llm_instance.astream(prompt):
        yield {"answer": chunk.content}