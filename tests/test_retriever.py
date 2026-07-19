# tests/test_retriever.py — 检索器单元测试

import pytest
from unittest.mock import MagicMock, patch

# RRF 是纯算法，不需要任何外部依赖


class TestRRFFusion:
    """RRF 名次融合算法 — 纯逻辑单元测试"""

    @staticmethod
    def _get_rrf_func():
        """获取 _rrf_fusion 方法的引用，绕过重型 __init__"""
        from app.services.retriever import MiemieMilvusRetriever

        # 通过 object.__new__ 创建空壳实例，跳过 __init__
        instance = object.__new__(MiemieMilvusRetriever)
        return instance._rrf_fusion

    def test_basic_fusion(self):
        """两路结果有重叠时，正确融合排序"""
        rrf = self._get_rrf_func()
        vec = ["文档A", "文档B", "文档C"]
        bm25 = ["文档B", "文档C", "文档D"]

        result = rrf(vec, bm25, k=60)

        assert result[0] == "文档B"  # 在两路都出现，排第一
        assert result[1] == "文档C"  # 在两路都出现，排第二
        assert len(result) == 4
        assert set(result) == {"文档A", "文档B", "文档C", "文档D"}

    def test_no_overlap(self):
        """两路结果完全不重叠，应保留全部"""
        rrf = self._get_rrf_func()
        vec = ["A1", "A2", "A3"]
        bm25 = ["B1", "B2", "B3"]

        result = rrf(vec, bm25, k=60)

        assert len(result) == 6
        assert set(result) == {"A1", "A2", "A3", "B1", "B2", "B3"}

    def test_empty_inputs(self):
        """空输入边界条件"""
        rrf = self._get_rrf_func()

        assert rrf([], [], k=60) == []
        assert rrf(["A"], [], k=60) == ["A"]
        assert rrf([], ["B"], k=60) == ["B"]

    def test_single_track_preserves_order(self):
        """只有一路有结果时，保持原始排名"""
        rrf = self._get_rrf_func()
        vec = ["first", "second", "third", "fourth", "fifth"]

        result = rrf(vec, [], k=60)

        assert result == vec

    def test_duplicate_removed(self):
        """同一文本在两路的不同排名位置出现，只保留一个"""
        rrf = self._get_rrf_func()
        vec = ["doc1", "doc2"]
        bm25 = ["doc3", "doc1"]

        result = rrf(vec, bm25, k=60)

        assert result.count("doc1") == 1
        assert len(result) == 3

    def test_k_parameter_smoothes_ranks(self):
        """k 值越大，排名差异的权重越小"""
        rrf = self._get_rrf_func()
        vec = ["A", "B", "C"]
        bm25 = ["B", "C", "A"]

        result_small_k = rrf(vec, bm25, k=1)
        result_large_k = rrf(vec, bm25, k=100)

        # 无论 k 取何值，都应包含所有元素
        assert set(result_small_k) == set(result_large_k) == {"A", "B", "C"}

    def test_large_input(self):
        """较大输入量的正确性冒烟测试"""
        rrf = self._get_rrf_func()
        vec = [f"v_{i}" for i in range(50)]
        bm25 = [f"b_{i}" for i in range(50)]

        result = rrf(vec, bm25, k=60)

        assert len(result) == 100


class TestRetrieverSingleton:
    """检索器单例模式测试"""

    def test_singleton_returns_same_instance(self):
        """连续两次调用应返回同一个实例"""
        import app.services.retriever as mod
        from unittest.mock import patch

        # 重置单例
        mod._global_retriever_instance = None

        with patch.object(mod, "MiemieMilvusRetriever") as MockClass:
            instance1 = mod.get_milvus_retriever()
            instance2 = mod.get_milvus_retriever()

            assert MockClass.call_count == 1
            assert instance1 is instance2
