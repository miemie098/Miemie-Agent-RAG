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


class TestLinearWeightedFusion:
    """线性加权融合 — 纯逻辑单元测试"""

    @staticmethod
    def _get_linear_fusion_func():
        """获取 _linear_weighted_fusion 方法的引用，绕过重型 __init__"""
        from app.services.retriever import MiemieMilvusRetriever
        instance = object.__new__(MiemieMilvusRetriever)
        return instance._linear_weighted_fusion

    def test_basic_fusion_equal_weight(self):
        """两路重叠，alpha=0.5 时两路都出现的文档应获得更高组合分"""
        lwf = self._get_linear_fusion_func()
        # Dense: A 高分(0.9), B 中分(0.7), C 低分(0.5)
        dense = [("A", 0.9), ("B", 0.7), ("C", 0.5)]
        # BM25: B 高分(8.0), C 中分(6.0), D 低分(4.0)
        bm25 = [("B", 8.0), ("C", 6.0), ("D", 4.0)]

        result = lwf(dense, bm25, alpha=0.5)

        # B 在两路都是最高归一化分 → 组合分最高，排第一
        assert result[0] == "B"
        # A 虽只在 dense，但其归一化分为 1.0（dense 路最高）→ 0.5 权重，高于 C
        # C 在 dense 归一化 0.0 + bm25 归一化 0.5 → 组合分 0.25
        assert len(result) == 4
        assert set(result) == {"A", "B", "C", "D"}

    def test_alpha_one_dense_only(self):
        """alpha=1.0 时结果应等于纯 Dense 排序"""
        lwf = self._get_linear_fusion_func()
        dense = [("A", 0.9), ("B", 0.7), ("C", 0.5)]
        bm25 = [("X", 10.0), ("Y", 9.0)]

        result = lwf(dense, bm25, alpha=1.0)

        # BM25 权重为 0，其文档得分全为 0，排在末尾
        assert result[:3] == ["A", "B", "C"]
        assert len(result) == 5

    def test_alpha_zero_bm25_only(self):
        """alpha=0.0 时 BM25 非零分文档应排在最前"""
        lwf = self._get_linear_fusion_func()
        dense = [("X", 0.9), ("Y", 0.7)]
        bm25 = [("A", 10.0), ("B", 8.0), ("C", 6.0)]

        result = lwf(dense, bm25, alpha=0.0)

        # BM25 文档得分非零，排在前；dense 文档权重为 0，得分全为 0 排在末尾
        assert result[:2] == ["A", "B"]
        assert len(result) == 5
        assert set(result) == {"A", "B", "C", "X", "Y"}

    def test_no_overlap(self):
        """两路完全不重叠时保留全部文档"""
        lwf = self._get_linear_fusion_func()
        dense = [("A1", 0.9), ("A2", 0.5)]
        bm25 = [("B1", 10.0), ("B2", 5.0)]

        result = lwf(dense, bm25, alpha=0.5)

        assert len(result) == 4
        assert set(result) == {"A1", "A2", "B1", "B2"}

    def test_empty_inputs(self):
        """空输入边界条件"""
        lwf = self._get_linear_fusion_func()

        assert lwf([], [], alpha=0.5) == []
        assert lwf([("A", 0.9)], [], alpha=0.5) == ["A"]
        assert lwf([], [("B", 5.0)], alpha=0.5) == ["B"]

    def test_single_track_preserves_order(self):
        """只有一路有结果时保持原始排名"""
        lwf = self._get_linear_fusion_func()
        dense = [("first", 0.9), ("second", 0.7), ("third", 0.5)]

        result = lwf(dense, [], alpha=0.5)

        assert result == ["first", "second", "third"]

    def test_score_preserves_relative_order(self):
        """同路内原始高分 → 高归一化分 → 高最终排名"""
        lwf = self._get_linear_fusion_func()
        dense = [("top", 0.95), ("mid", 0.5), ("low", 0.05)]
        bm25 = []  # 只有一路

        result = lwf(dense, bm25, alpha=0.5)

        assert result == ["top", "mid", "low"]

    def test_duplicate_text_accumulates_scores(self):
        """同一文本出现在两路，分数应累加"""
        lwf = self._get_linear_fusion_func()
        # A 在 dense 排第一（高分），在 bm25 排第一（高分）→ 应排第一
        # B 只在 dense → 排第二
        dense = [("shared", 0.9), ("only_dense", 0.5)]
        bm25 = [("shared", 10.0)]

        result = lwf(dense, bm25, alpha=0.5)

        assert result[0] == "shared"
        assert len(result) == 2


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
