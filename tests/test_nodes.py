# tests/test_nodes.py — 图节点逻辑单元测试

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.graph.nodes import GraphState


class TestGraphState:
    """GraphState 数据结构验证"""

    def test_valid_state(self):
        state: GraphState = {
            "question": "什么是 RAG？",
            "context": "RAG 是检索增强生成...",
            "answer": "",
        }
        assert state["question"] == "什么是 RAG？"
        assert state["context"] == "RAG 是检索增强生成..."

    def test_empty_state_allowed(self):
        state: GraphState = {"question": "", "context": "", "answer": ""}
        assert state["question"] == ""

    def test_state_has_expected_keys(self):
        state: GraphState = {"question": "Q", "context": "", "answer": ""}
        assert set(state.keys()) == {"question", "context", "answer"}


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

        state: GraphState = {"question": "测试", "context": "", "answer": ""}
        result = retrieve_node(state)

        assert result["context"] == "第一段内容\n第二段内容"

    @patch("app.graph.nodes.get_milvus_retriever")
    def test_no_results_returns_empty_context(self, mock_get_retriever):
        from app.graph.nodes import retrieve_node

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        state: GraphState = {"question": "测试", "context": "", "answer": ""}
        result = retrieve_node(state)

        assert result["context"] == ""


class TestGenerateNode:
    """生成节点 — Mock LLM 调用"""

    @patch("app.graph.nodes._get_llm")
    def test_builds_prompt_with_context_and_question(self, mock_get_llm):
        from app.graph.nodes import generate_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "生成的回答"
        mock_get_llm.return_value = mock_llm

        state: GraphState = {
            "question": "什么是 PagedAttention？",
            "context": "PagedAttention 由 vLLM 提出...",
            "answer": "",
        }
        result = generate_node(state)

        assert result["answer"] == "生成的回答"
        call_arg = mock_llm.invoke.call_args[0][0]
        assert "PagedAttention 由 vLLM 提出" in call_arg
        assert "什么是 PagedAttention？" in call_arg

    @patch("app.graph.nodes._get_llm")
    def test_graceful_degradation_on_api_error(self, mock_get_llm):
        from app.graph.nodes import generate_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("网络超时")
        mock_get_llm.return_value = mock_llm

        state: GraphState = {
            "question": "测试",
            "context": "测试上下文",
            "answer": "",
        }
        result = generate_node(state)

        assert "answer" in result
        assert "系统提示" in result["answer"]


class TestGenerateNodeStream:
    """流式生成节点 — Mock LLM 流式调用"""

    @staticmethod
    async def _collect_generator(gen):
        """收集 async generator 的所有 yield 值"""
        results = []
        async for item in gen:
            results.append(item)
        return results

    @patch("app.graph.nodes._get_llm")
    @pytest.mark.asyncio
    async def test_stream_accumulates_tokens(self, mock_get_llm):
        from app.graph.nodes import generate_node_stream

        # Mock astream：模拟逐 token 返回
        async def mock_astream(prompt):
            for token in ["你好", "，", "世界", "！"]:
                chunk = MagicMock()
                chunk.content = token
                yield chunk

        mock_llm = MagicMock()
        mock_llm.astream = mock_astream
        mock_get_llm.return_value = mock_llm

        state: GraphState = {
            "question": "测试问题",
            "context": "测试上下文",
            "answer": "",
        }
        results = await self._collect_generator(generate_node_stream(state))

        # 每次 yield 是累积的完整文本
        assert results[0]["answer"] == "你好"
        assert results[1]["answer"] == "你好，"
        assert results[2]["answer"] == "你好，世界"
        assert results[3]["answer"] == "你好，世界！"
        assert len(results) == 4

    @patch("app.graph.nodes._get_llm")
    @pytest.mark.asyncio
    async def test_stream_graceful_degradation(self, mock_get_llm):
        from app.graph.nodes import generate_node_stream

        async def mock_astream_error(prompt):
            chunk = MagicMock()
            chunk.content = "部分输出"
            yield chunk
            raise Exception("连接中断")

        mock_llm = MagicMock()
        mock_llm.astream = mock_astream_error
        mock_get_llm.return_value = mock_llm

        state: GraphState = {
            "question": "测试",
            "context": "测试上下文",
            "answer": "",
        }
        results = await self._collect_generator(generate_node_stream(state))

        # 降级消息应作为最后一个 yield 返回
        assert "系统提示" in results[-1]["answer"]
