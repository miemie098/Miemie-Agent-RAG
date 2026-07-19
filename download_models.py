from modelscope.hub.snapshot_download import snapshot_download

print("开始从国内高速通道下载 BAAI/bge-reranker-large ...")

# 明确指定 cache_dir 参数，这里以 D:\models_cache 为例（你可以自己建一个喜欢的文件夹）
# 注意路径前面的 r 不要漏掉，防止 Windows 路径转义报错
model_dir = snapshot_download('Xorbits/bge-reranker-large', cache_dir=r'D:\models_cache')

print(f"✅ 模型已成功下载到: {model_dir}")