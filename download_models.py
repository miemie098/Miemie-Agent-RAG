# Miemie-Agent-RAG 模型预下载工具
# 用法:
#   python download_models.py                          # 下载到默认缓存目录
#   python download_models.py --cache-dir ./models     # 下载到指定目录
#   python download_models.py --source modelscope      # 使用 ModelScope 国内镜像（默认）

import os
import argparse
from modelscope.hub.snapshot_download import snapshot_download

MODEL_NAME = "Xorbits/bge-reranker-large"
HUGGINGFACE_MODEL_NAME = "BAAI/bge-reranker-large"


def main():
    parser = argparse.ArgumentParser(description="预下载 BGE-Reranker-Large 精排模型")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv("MODELS_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "modelscope")),
        help="模型缓存目录 (默认: ~/.cache/modelscope)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="下载源 (默认: modelscope 国内镜像)",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    if args.source == "modelscope":
        print(f"开始从 ModelScope 下载 {MODEL_NAME} ...")
        print(f"缓存目录: {args.cache_dir}")
        model_dir = snapshot_download(MODEL_NAME, cache_dir=args.cache_dir)
    else:
        print(f"开始从 HuggingFace Hub 下载 {HUGGINGFACE_MODEL_NAME} ...")
        print(f"缓存目录: {args.cache_dir}")
        from huggingface_hub import snapshot_download as hf_snapshot_download
        model_dir = hf_snapshot_download(HUGGINGFACE_MODEL_NAME, cache_dir=args.cache_dir)

    print(f"✅ 模型已成功下载到: {model_dir}")
    print()
    print("👉 如需指定本地模型路径，在 .env 中设置:")
    print(f'   RERANKER_MODEL_PATH={model_dir}')


if __name__ == "__main__":
    main()
