# Miemie-Agent-RAG/app/services/retriever.py
import logging
import os

from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

logger = logging.getLogger("miemie-rag.retriever")

_global_retriever_instance = None


class _MockDocument:
    """轻量文档包装，兼容 LangGraph 工作流的 .page_content 属性"""

    def __init__(self, page_content: str):
        self.page_content = page_content


class MiemieMilvusRetriever:
    """多路混合检索 + Cross-Encoder 精排的检索器

    检索管线：Dense 向量搜索 + BM25 稀疏检索 → 线性加权融合 → Cross-Encoder 重排
    默认 fusion_method="linear_weighted", fusion_alpha=0.5
    """

    def __init__(
        self,
        collection_name: str = "miemie_knowledge_base",
        fusion_method: str = "linear_weighted",
        fusion_alpha: float = 0.5,
    ):
        if fusion_method not in ("rrf", "linear_weighted"):
            raise ValueError(
                f"不支持的融合方法: {fusion_method}，可选: rrf, linear_weighted"
            )
        if not 0.0 <= fusion_alpha <= 1.0:
            raise ValueError(f"fusion_alpha 必须在 [0.0, 1.0] 范围内，当前值: {fusion_alpha}")

        self.fusion_method = fusion_method
        self.fusion_alpha = fusion_alpha
        # Milvus 连接
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        db_path = os.path.join(project_root, "milvus.db")

        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.client.load_collection(collection_name=self.collection_name)

        # Embedding 模型
        self.dense_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # BM25 稀疏检索索引
        logger.info("正在构建 BM25 倒排索引...")
        all_docs = self.client.query(
            collection_name=self.collection_name,
            filter="id > 0",
            output_fields=["text"],
        )
        self.corpus = [doc.get("text", "") for doc in all_docs] if all_docs else []
        tokenized_corpus = [list(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 索引就绪，语料数: %d", len(self.corpus))

        # Cross-Encoder 精排模型
        reranker_source = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-large")
        logger.info("装载精排模型: %s", reranker_source)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_source)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_source
        )
        self.rerank_model.eval()

    def _rrf_fusion(
        self, vector_results: list[str], bm25_results: list[str], k: int = 60
    ) -> list[str]:
        """Reciprocal Rank Fusion：合并两路检索结果的排名"""
        score_dict: dict[str, float] = {}
        for rank, text in enumerate(vector_results):
            score_dict[text] = score_dict.get(text, 0.0) + 1.0 / (k + rank + 1)
        for rank, text in enumerate(bm25_results):
            score_dict[text] = score_dict.get(text, 0.0) + 1.0 / (k + rank + 1)

        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [text for text, _ in sorted_items]

    @staticmethod
    def _minmax_normalize(scores: list[float]) -> list[float]:
        """Min-max 归一化：将任意尺度分数映射到 [0, 1]"""
        if not scores:
            return []
        s_min, s_max = min(scores), max(scores)
        if s_max == s_min:
            return [1.0] * len(scores)
        return [(s - s_min) / (s_max - s_min) for s in scores]

    def _linear_weighted_fusion(
        self,
        dense_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        alpha: float = 0.5,
    ) -> list[str]:
        """线性加权融合：归一化后按权重合并两路检索分数

        Args:
            dense_results: [(text, raw_similarity_score), ...]
            bm25_results:  [(text, raw_bm25_score), ...]
            alpha:         Dense 路权重（0 = 纯 BM25，1 = 纯 Dense）

        Returns:
            按加权组合分数降序排列的文本列表
        """
        # 归一化各路的原始分数
        dense_texts = [t for t, _ in dense_results]
        dense_scores = self._minmax_normalize([s for _, s in dense_results])
        bm25_texts = [t for t, _ in bm25_results]
        bm25_scores = self._minmax_normalize([s for _, s in bm25_results])

        combined: dict[str, float] = {}
        for text, norm_s in zip(dense_texts, dense_scores):
            combined[text] = combined.get(text, 0.0) + alpha * norm_s
        for text, norm_s in zip(bm25_texts, bm25_scores):
            combined[text] = combined.get(text, 0.0) + (1 - alpha) * norm_s

        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [text for text, _ in sorted_items]

    def _cross_encoder_rerank(
        self, query: str, candidates: list[str], top_n: int = 3
    ) -> list[str]:
        """Cross-Encoder 精排：对候选集做细粒度语义相关度打分"""
        if not candidates:
            return []

        pairs = [[query, cand] for cand in candidates]
        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            )
            scores = self.rerank_model(**inputs).logits.view(-1).float().tolist()

        ranked = [cand for _, cand in sorted(zip(scores, candidates), reverse=True)]
        return ranked[:top_n]

    def invoke(self, query: str) -> list[_MockDocument]:
        """执行完整检索管线，返回与 LangGraph 兼容的文档列表"""
        # 轨道 A：Dense 向量检索（保留相似度分数）
        query_vector = self.dense_embeddings.embed_query(query)
        dense_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=5,
            output_fields=["text"],
        )
        dense_hits: list[tuple[str, float]] = []
        if dense_results and len(dense_results) > 0:
            for hit in dense_results[0]:
                text = hit.get("entity", {}).get("text", "")
                if text:
                    dense_hits.append((text, float(hit.get("distance", 0.0))))

        # 轨道 B：BM25 稀疏检索（保留 BM25 分数）
        bm25_hits: list[tuple[str, float]] = []
        if self.corpus:
            tokenized_query = list(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
            )[:5]
            bm25_hits = [
                (self.corpus[idx], float(doc_scores[idx])) for idx in top_indices
            ]

        # 轨道 C：融合（按配置的策略分发）
        if self.fusion_method == "linear_weighted":
            fused = self._linear_weighted_fusion(
                dense_hits, bm25_hits, alpha=self.fusion_alpha
            )
        else:
            # RRF 只看排名不看分数，提取纯文本列表
            fused = self._rrf_fusion(
                [t for t, _ in dense_hits],
                [t for t, _ in bm25_hits],
                k=60,
            )

        # 轨道 D：Cross-Encoder 精排
        final = self._cross_encoder_rerank(query, fused, top_n=3)

        return [_MockDocument(text) for text in final]


def get_milvus_retriever(
    collection_name: str = "miemie_knowledge_base",
    fusion_method: str = "linear_weighted",
    fusion_alpha: float = 0.5,
):
    """检索器单例工厂

    首次调用时按给定配置初始化单例；后续调用忽略参数，直接返回已缓存的实例。
    如需切换融合策略，请先调用 reset_retriever_singleton()。
    """
    global _global_retriever_instance
    if _global_retriever_instance is None:
        logger.info("初始化检索器单例（fusion=%s）...", fusion_method)
        _global_retriever_instance = MiemieMilvusRetriever(
            collection_name, fusion_method=fusion_method, fusion_alpha=fusion_alpha
        )
    return _global_retriever_instance


def reset_retriever_singleton():
    """重置检索器单例，供评测/测试在切换融合策略时使用"""
    global _global_retriever_instance
    if _global_retriever_instance is not None:
        logger.info("重置检索器单例...")
    _global_retriever_instance = None
