# Miemie-Agent-RAG/app/services/retriever.py
import os
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi  # ⚡ 引入本地高性能 BM25 库
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # ⚡ 引入大厂精排依赖
import torch

_global_retriever_instance = None


class MiemieMilvusRetriever:
    """标准大厂生产级：多路并行召回 + 工业级 BGE Cross-Encoder 精排重排检索器"""

    def __init__(self, collection_name="miemie_knowledge_base"):
        # 1. 初始化本地 Milvus 句柄
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        db_absolute_path = os.path.join(project_root, "milvus.db")

        self.client = MilvusClient(db_absolute_path)
        self.collection_name = collection_name
        self.client.load_collection(collection_name=self.collection_name)

        # 2. 密集向量语义搜索轨道 (Dense Track)
        self.dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # 3. ⚡ 升级核心：拉取本地全量文本语料，构建确定性的本地 BM25 稀疏检索轨道
        print("====== [Miemie-RAG 算法激活] 正在同步构建本地确定性 BM25 倒排索引... ======")
        all_docs = self.client.query(
            collection_name=self.collection_name,
            filter="id > 0",
            output_fields=["text"]
        )
        self.corpus = [doc.get("text", "") for doc in all_docs] if all_docs else []
        # 按字符切分或分词，构建 BM25 树
        tokenized_corpus = [list(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"====== [Miemie-RAG 算法激活] BM25 索引构建完毕，共加载 {len(self.corpus)} 条底座语料。 ======")

        # 4. 精排模块：BGE-Reranker-Large (Cross-Encoder)
        # 优先使用环境变量指定的本地路径，否则从 HuggingFace Hub 自动下载
        reranker_source = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-large")
        print(f"====== [Miemie-RAG 算法激活] 正在装载精排模型: {reranker_source} ======")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_source)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_source)
        self.rerank_model.eval()

    def _rrf_fusion(self, vector_results: list, bm25_results: list, k: int = 60) -> list:
        """纯手工打造工业级 RRF 混合检索名次融合算法"""
        rrf_score_dict = {}
        for rank, text in enumerate(vector_results):
            if text not in rrf_score_dict:
                rrf_score_dict[text] = {"text": text, "score": 0.0}
            rrf_score_dict[text]["score"] += 1.0 / (k + (rank + 1))

        for rank, text in enumerate(bm25_results):
            if text not in rrf_score_dict:
                rrf_score_dict[text] = {"text": text, "score": 0.0}
            rrf_score_dict[text]["score"] += 1.0 / (k + (rank + 1))

        sorted_docs = sorted(rrf_score_dict.values(), key=lambda x: x["score"], reverse=True)
        return [item["text"] for item in sorted_docs]

    def _cross_encoder_rerank(self, query: str, candidates: list, top_n: int = 3) -> list:
        """
        ⚡ 缝合大厂核心精排算法：Cross-Encoder 精确语义相关度打分
        """
        if not candidates:
            return []

        # 构建大模型精排输入对：[[问题, 候选文本1], [问题, 候选文本2], ...]
        pairs = [[query, cand] for cand in candidates]

        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            # 计算 Logits 得分
            scores = self.rerank_model(**inputs).logits.view(-1).float().tolist()

        # 将得分与文本对齐，并全量降序排列，切片截取前 Top-N 高密度上下文
        reranked_results = [cand for _, cand in sorted(zip(scores, candidates), reverse=True)]
        return reranked_results[:top_n]

    def invoke(self, query: str):
        # === 轨道 A：密集向量语义检索 (粗筛第一路) ===
        query_dense_vector = self.dense_embeddings.embed_query(query)
        dense_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_dense_vector],
            limit=5,
            output_fields=["text"]
        )
        vector_ranks = []
        if dense_results and len(dense_results) > 0:
            for hit in dense_results[0]:
                text = hit.get("entity", {}).get("text", "")
                if text: vector_ranks.append(text)

        # === 轨道 B：确定性本地 BM25 检索 (粗筛第二路) ===
        bm25_ranks = []
        if self.corpus:
            tokenized_query = list(query)
            # 获取所有语料的 BM25 分数
            doc_scores = self.bm25.get_scores(tokenized_query)
            # 取得分最高的前 5 个索引
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:5]
            bm25_ranks = [self.corpus[idx] for idx in top_indices]

        # === 轨道 C：粗筛混流 (RRF 名次融合) ===
        coarse_fused_texts = self._rrf_fusion(vector_ranks, bm25_ranks, k=60)

        # === 轨道 D：⚡ 工业级终极精排精算 (BGE Cross-Encoder Rerank) ===
        # 将 RRF 融合后的全量粗筛候选集，丢进重排模型进行绝对语义相关度计算
        final_fused_texts = self._cross_encoder_rerank(query, coarse_fused_texts, top_n=3)

        # 兼容上层 LangGraph 工作流
        class MockDocument:
            def __init__(self, page_content):
                self.page_content = page_content

        return [MockDocument(text) for text in final_fused_texts]


def get_milvus_retriever(collection_name="miemie_knowledge_base"):
    global _global_retriever_instance
    if _global_retriever_instance is None:
        print("====== [Miemie-RAG 提示] 初始化全生命周期唯一的混合检索精排单例... ======")
        _global_retriever_instance = MiemieMilvusRetriever(collection_name)
    return _global_retriever_instance