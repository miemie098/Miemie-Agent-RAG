# Miemie-Agent-RAG/ingest.py
import logging
import os
import shutil

from pypdf import PdfReader
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("miemie-rag.ingest")

DB_PATH = "./milvus.db"
DATA_DIR = "./data"


def clean_existing_db():
    """清理旧数据库，避免覆盖异常"""
    if os.path.exists(DB_PATH):
        try:
            if os.path.isdir(DB_PATH):
                shutil.rmtree(DB_PATH)
            else:
                os.remove(DB_PATH)
            logger.info("已清理旧的数据库文件")
        except Exception as e:
            logger.warning("清理旧数据库时遇到进程锁: %s", e)


def create_collection(client: MilvusClient):
    """创建 Milvus 集合，手动声明 Schema"""
    id_field = FieldSchema(
        name="id", dtype=DataType.INT64, is_primary=True, description="primary key"
    )
    vector_field = FieldSchema(
        name="vector", dtype=DataType.FLOAT_VECTOR, dim=768, description="dense vector"
    )
    schema = CollectionSchema(
        fields=[id_field, vector_field],
        enable_dynamic_field=True,
        description="rag knowledge base",
    )
    client.create_collection(collection_name="miemie_knowledge_base", schema=schema)
    logger.info("集合 miemie_knowledge_base 创建成功")


def layout_aware_chunker(pdf_path: str) -> list[str]:
    """PDF 布局感知分块：按段落切分，保留页码来源"""
    reader = PdfReader(pdf_path)
    chunks = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 10]
        for p in paragraphs:
            chunks.append(
                f"[来源: {os.path.basename(pdf_path)} | 第 {page_idx + 1} 页] {p}"
            )
    return chunks


def ingest_pdfs(client: MilvusClient, embeddings: HuggingFaceEmbeddings):
    """遍历 data 目录，解析 PDF 并写入 Milvus"""
    all_chunks: list[str] = []

    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(DATA_DIR, filename)
                logger.info("解析文件: %s", filename)
                try:
                    chunks = layout_aware_chunker(pdf_path)
                    all_chunks.extend(chunks)
                    logger.info("  提取 %d 个文本块", len(chunks))
                except Exception as e:
                    logger.error("解析 %s 失败: %s", filename, e)
    else:
        logger.warning("未找到 data 目录，将使用默认占位数据")

    if not all_chunks:
        logger.warning("未提取到有效文本，启用占位数据")
        all_chunks = [
            "候选人简历：精通Transformer架构，深入理解注意力机制。"
            "擅长大模型微调(SFT)与强化学习(GRPO/DPO)。"
            "具有丰富的RAG系统构建经验，熟悉LangGraph与Milvus向量检索。"
            "曾参与多模态Agent开发，具备高并发系统工程能力。"
        ]

    logger.info("开始向量化 %d 个文本块...", len(all_chunks))
    vectors = embeddings.embed_documents(all_chunks)

    insert_data = []
    for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
        insert_data.append({"id": i + 1, "vector": vector, "text": chunk})

    res = client.insert(collection_name="miemie_knowledge_base", data=insert_data)
    logger.info("批量写入完成，插入 %d 条记录", res["insert_count"])


if __name__ == "__main__":
    clean_existing_db()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    client = MilvusClient(uri="./milvus.db")
    create_collection(client)
    ingest_pdfs(client, embeddings)
