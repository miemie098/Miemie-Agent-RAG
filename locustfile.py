import json
import requests  # 引入 requests 以捕获异常
from locust import HttpUser, task, between

class RAGClusterLoadTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_streaming_rag(self):
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "question": "对比 DeepSeek-V3 架构和传统的 Transformer 模型，在注意力机制（Attention）和显存优化策略上有什么核心区别？"
        }

        with self.client.post(
                "/chat/stream",
                data=json.dumps(payload),
                headers=headers,
                stream=True,
                catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    # 尝试读取所有行
                    for _ in response.iter_lines():
                        pass
                    response.success()
                except requests.exceptions.ChunkedEncodingError as e:
                    # 捕获连接突然中断的异常
                    response.failure(f"流式传输中途断开 (可能被限流或超时): {str(e)[:50]}")
                except Exception as e:
                    # 捕获其他未知异常
                    response.failure(f"读取流时发生未知错误: {str(e)[:50]}")
            else:
                response.failure(f"集群拒绝连接，状态码: {response.status_code}")