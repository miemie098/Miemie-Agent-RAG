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

    检索管线：Dense 向量搜索 + BM25 稀疏检索 → RRF 融合 → Cross-Encoder 重排
    """

    def __init__(self, collection_name: str = "miemie_knowledge_base"):
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
        # 轨道 A：Dense 向量检索
        query_vector = self.dense_embeddings.embed_query(query)
        dense_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=5,
            output_fields=["text"],
        )
        vector_ranks: list[str] = []
        if dense_results and len(dense_results) > 0:
            for hit in dense_results[0]:
                text = hit.get("entity", {}).get("text", "")
                if text:
                    vector_ranks.append(text)

        # 轨道 B：BM25 稀疏检索
        bm25_ranks: list[str] = []
        if self.corpus:
            tokenized_query = list(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
            )[:5]
            bm25_ranks = [self.corpus[idx] for idx in top_indices]

        # 轨道 C：RRF 融合
        fused = self._rrf_fusion(vector_ranks, bm25_ranks, k=60)

        # 轨道 D：Cross-Encoder 精排
        final = self._cross_encoder_rerank(query, fused, top_n=3)

        return [_MockDocument(text) for text in final]


def get_milvus_retriever(collection_name: str = "miemie_knowledge_base"):
    """检索器单例工厂"""
    global _global_retriever_instance
    if _global_retriever_instance is None:
        logger.info("初始化检索器单例...")
        _global_retriever_instance = MiemieMilvusRetriever(collection_name)
    return _global_retriever_instance
