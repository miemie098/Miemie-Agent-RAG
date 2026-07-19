# Miemie-Agent-RAG/ingest.py
import os
import shutil
from pypdf import PdfReader
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema  # ⚡ 最底层原子组件
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "./milvus.db"

# 1. 物理粉碎脏数据防御
if os.path.exists(DB_PATH):
    try:
        if os.path.isdir(DB_PATH):
            shutil.rmtree(DB_PATH)
        else:
            os.remove(DB_PATH)
        print("====== [Miemie-RAG 防御中心] 已物理粉碎残留的旧数据库，阻断覆盖异常！ ======")
    except Exception as e:
        print(f"提示：物理清理遭遇进程锁: {e}")

# 2. 初始化网关与密向量模型
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

client = MilvusClient(uri="./milvus.db")
# 3. 大厂原子重构：手动声明低级 FieldSchema，绝不让协议污染
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary key")
vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768, description="dense vector")

schema = CollectionSchema(fields=[id_field, vector_field], enable_dynamic_field=True, description="rag knowledge base")

# 4. 显式建表
client.create_collection(
    collection_name="miemie_knowledge_base",
    schema=schema
)
print("全新的原生集合显式创建成功！")


# 5. ⚡ 智能文档布局感知解析分块器
# ====== 以下是 ingest.py 的下半部分 ======

# 5. ⚡ 智能文档布局感知解析分块器 (单文件处理逻辑)
def layout_aware_chunker(pdf_path: str):
    reader = PdfReader(pdf_path)
    chunks = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        # 按双换行符粗切分，剔除太短的无效字符
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 10]
        for p in paragraphs:
            # 记录页码来源，保证检索时的溯源能力
            chunks.append(f"[来源文件: {os.path.basename(pdf_path)} | 第 {page_idx + 1} 页] {p}")
    return chunks


# 6. ⚡ 执行批量解析并灌入数据库
print("====== [Miemie-RAG 数据工厂] 开始批量摄入本地文档... ======")
DATA_DIR = "./data"  # 相对路径，指向你的 PDF 文件夹
all_chunks = []

# 遍历 data 文件夹，寻找所有的 pdf 文件
if os.path.exists(DATA_DIR):
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, filename)
            print(f"📄 正在解析文件: {filename} ...")
            try:
                file_chunks = layout_aware_chunker(pdf_path)
                all_chunks.extend(file_chunks)
                print(f"   ✅ 成功提取 {len(file_chunks)} 个文本块。")
            except Exception as e:
                print(f"   ❌ 解析 {filename} 失败: {str(e)}")
else:
    print(f"⚠️ 未找到文件夹 {DATA_DIR}，请确保路径正确。")

# 兜底机制：如果文件夹是空的，或者没有有效的文本，就插入默认数据
if not all_chunks:
    print("⚠️ 警告：未从 data 目录提取到任何有效文本，启用系统兜底简历文本。")
    all_chunks = [
        "候选人简历：精通Transformer架构，深入理解注意力机制。擅长大模型微调(SFT)与强化学习(GRPO/DPO)。具有丰富的RAG系统构建经验，熟悉LangGraph与Milvus向量检索。曾参与多模态Agent开发，具备高并发系统工程能力。"
    ]

# 7. 向量化并批量存入 Milvus
print(f"====== [Miemie-RAG 计算中心] 总计准备向量化 {len(all_chunks)} 个高密度知识块... ======")
# 计算密集向量
vectors = embeddings.embed_documents(all_chunks)

# 组装数据并插入 Milvus
insert_data = []
for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
    insert_data.append({
        "id": i + 1,  # Milvus ID 必须大于 0
        "vector": vector,
        "text": chunk  # 动态字段存储原始文本
    })

res = client.insert(
    collection_name="miemie_knowledge_base",
    data=insert_data
)
print(f"====== [Miemie-RAG 数据中心] 批量落盘完成！成功写入 {res['insert_count']} 条记录到 Milvus Lite。 ======")