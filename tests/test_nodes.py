# tests/test_nodes.py — 图节点逻辑单元测试

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.graph.nodes import GraphState


def _make_state(**overrides) -> GraphState:
    """构建测试用的 GraphState，提供合理的默认值"""
    state: GraphState = {
        "question": "测试问题",
        "context": "测试上下文",
        "answer": "",
        "messages": [],
    }
    state.update(overrides)
    return state


class TestGraphState:
    """GraphState 数据结构验证"""

    def test_valid_state_with_messages(self):
        state: GraphState = {
            "question": "什么是 RAG？",
            "context": "RAG 是检索增强生成...",
            "answer": "",
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
            ],
        }
        assert state["question"] == "什么是 RAG？"
        assert len(state["messages"]) == 2

    def test_empty_messages_allowed(self):
        state = _make_state(messages=[])
        assert state["messages"] == []

    def test_state_has_all_keys(self):
        state = _make_state()
        assert set(state.keys()) == {"question", "context", "answer", "messages"}


class TestRetrieveNode:
    """检索节点 — 使用 Mock 隔离外部依赖"""

    @patch("app.graph.nodes.get_milvus_retriever")
    def test_concatenates_docs_with_newline(self, mock_get_retriever):
        from app.graph.nodes import retrieve_node

        doc1 = MagicMock()
        doc1.page_content = "第一段内容"
        doc2 = MagicMock()
        doc2.page_content = "第二段内容"

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc1, doc2]
        mock_get_retriever.return_value = mock_retriever

        state = _make_state()
        result = retrieve_node(state)

        assert result["context"] == "第一段内容\n第二段内容"

    @patch("app.graph.nodes.get_milvus_retriever")
    def test_no_results_returns_empty_context(self, mock_get_retriever):
        from app.graph.nodes import retrieve_node

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        state = _make_state()
        result = retrieve_node(state)

        assert result["context"] == ""


class TestGenerateNode:
    """生成节点 — Mock LLM 调用"""

    @patch("app.graph.nodes._get_llm")
    def test_includes_context_and_question(self, mock_get_llm):
        from app.graph.nodes import generate_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "生成的回答"
        mock_get_llm.return_value = mock_llm

        state = _make_state(
            question="什么是 PagedAttention？",
            context="PagedAttention 由 vLLM 提出...",
        )
        result = generate_node(state)

        assert result["answer"] == "生成的回答"
        call_arg = mock_llm.invoke.call_args[0][0]
        # 消息列表最后一员是 user 消息，包含问题
        last_msg = call_arg[-1]
        assert last_msg["role"] == "user"
        assert "PagedAttention" in last_msg["content"]

    @patch("app.graph.nodes._get_llm")
    def test_includes_chat_history(self, mock_get_llm):
        from app.graph.nodes import generate_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "好的"
        mock_get_llm.return_value = mock_llm

        state = _make_state(
            question="继续刚才的话题",
            context="一些上下文",
            messages=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！"},
            ],
        )
        result = generate_node(state)

        call_arg = mock_llm.invoke.call_args[0][0]
        # 应该有 system, user, assistant, user 四条消息
        assert len(call_arg) == 4
        assert call_arg[0]["role"] == "system"
        assert call_arg[1] == {"role": "user", "content": "你好"}
        assert call_arg[2] == {"role": "assistant", "content": "你好！"}
        assert call_arg[3]["role"] == "user"

    @patch("app.graph.nodes._get_llm")
    def test_graceful_degradation_on_api_error(self, mock_get_llm):
        from app.graph.nodes import generate_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("网络超时")
        mock_get_llm.return_value = mock_llm

        state = _make_state()
        result = generate_node(state)

        assert "answer" in result
        assert "系统提示" in result["answer"]


class TestGenerateNodeStream:
    """流式生成节点 — Mock LLM 流式调用"""

    @staticmethod
    async def _collect(async_gen):
        results = []
        async for item in async_gen:
            results.append(item)
        return results

    @patch("app.graph.nodes._get_llm")
    @pytest.mark.asyncio
    async def test_stream_accumulates_tokens(self, mock_get_llm):
        from app.graph.nodes import generate_node_stream

        async def mock_astream(messages):
            for token in ["你好", "，", "世界", "！"]:
                chunk = MagicMock()
                chunk.content = token
                yield chunk

        mock_llm = MagicMock()
        mock_llm.astream = mock_astream
        mock_get_llm.return_value = mock_llm

        state = _make_state()
        results = await self._collect(generate_node_stream(state))

        assert results[0]["answer"] == "你好"
        assert results[1]["answer"] == "你好，"
        assert results[2]["answer"] == "你好，世界"
        assert results[3]["answer"] == "你好，世界！"

    @patch("app.graph.nodes._get_llm")
    @pytest.mark.asyncio
    async def test_stream_graceful_degradation(self, mock_get_llm):
        from app.graph.nodes import generate_node_stream

        async def mock_astream_error(messages):
            chunk = MagicMock()
            chunk.content = "部分输出"
            yield chunk
            raise Exception("连接中断")

        mock_llm = MagicMock()
        mock_llm.astream = mock_astream_error
        mock_get_llm.return_value = mock_llm

        state = _make_state()
        results = await self._collect(generate_node_stream(state))

        assert "系统提示" in results[-1]["answer"]
