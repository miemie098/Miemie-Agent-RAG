# Miemie-Agent-RAG/test_stream.py
import requests
import json
import sys
import time

url = "http://127.0.0.1:8000/chat/stream"
payload = {"question": "对比 DeepSeek-V3 架构和传统的 Transformer 模型，在注意力机制（Attention）和显存优化策略上有什么核心区别？"}
headers = {"Content-Type": "application/json"}

print("🚀 正在向流式接口发起请求，准备观测 TTFT 延迟...")
print("🤖 助手回答：", end="")

# ===== 核心修改开始 =====
# 创建一个独立的 Session 会话
session = requests.Session()
# 强制剥离操作系统的任何代理环境变量和证书劫持！
session.trust_env = False

try:
    with session.post(
            url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=60
    ) as response:
        # ===== 核心修改结束 =====

        # 下面保留你原本优秀的流式解析代码
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        print("\n\n🏁 [流式传输结束：服务器发出 DONE 信号]")
                        break
                    try:
                        chunk = json.loads(data_str)
                        token = chunk.get("answer", "")
                        print(token, end="")
                        sys.stdout.flush()
                    except json.JSONDecodeError:
                        continue
except Exception as e:
    print(f"\n❌ 请求失败：{e}")