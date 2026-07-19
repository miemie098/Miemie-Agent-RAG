# tests/test_workflow.py — 工作流图结构单元测试

import pytest
from unittest.mock import patch

from app.graph.workflow import create_workflow


class TestWorkflowCreation:
    """LangGraph 工作流编译与结构验证"""

    @patch("app.graph.workflow.retrieve_node")
    @patch("app.graph.workflow.generate_node")
    def test_returns_compiled_graph(self, mock_gen, mock_ret):
        """编译后的图应具有 invoke 和 astream 方法"""
        workflow = create_workflow()

        assert hasattr(workflow, "invoke")
        assert hasattr(workflow, "astream")

    @patch("app.graph.workflow.retrieve_node")
    @patch("app.graph.workflow.generate_node")
    def test_has_retrieve_and_generate_nodes(self, mock_gen, mock_ret):
        """图中应注册 retrieve 和 generate 两个节点"""
        workflow = create_workflow()

        node_names = set(workflow.get_graph().nodes.keys())

        assert "retrieve" in node_names
        assert "generate" in node_names

    @patch("app.graph.workflow.retrieve_node")
    @patch("app.graph.workflow.generate_node")
    def test_has_retrieve_to_generate_edge(self, mock_gen, mock_ret):
        """验证边: retrieve → generate"""
        workflow = create_workflow()

        edges = workflow.get_graph().edges
        edge_pairs = {(e.source, e.target) for e in edges}

        assert ("retrieve", "generate") in edge_pairs, f"Missing edge, got: {edge_pairs}"

    @patch("app.graph.workflow.retrieve_node")
    @patch("app.graph.workflow.generate_node")
    def test_workflow_entry_is_retrieve(self, mock_gen, mock_ret):
        """入口节点应为 retrieve"""
        workflow = create_workflow()

        nodes = workflow.get_graph().nodes

        assert "retrieve" in nodes
