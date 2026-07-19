# Miemie-Agent-RAG 网络连通性探针
# 用法: python test.py
# 用途: 剥离所有框架，通过底层 SDK 裸连 DeepSeek 官方网关，判断网络是否可达

import asyncio
import os
from openai import AsyncOpenAI

# 清理系统代理环境变量，防止代理劫持
PROXY_VARS = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]
for var in PROXY_VARS:
    os.environ.pop(var, None)


async def check_network():
    print("🔍 [探针启动] 正在剥离所有框架，通过底层 SDK 裸连 DeepSeek 官方网关...")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ [配置错误] 未检测到环境变量 DEEPSEEK_API_KEY")
        print("👉 请确保项目根目录的 .env 文件中已配置正确的 API Key")
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        max_retries=0,
    )

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "ping"}],
            timeout=15.0,
        )
        print("✅ [探针报告] 物理链路完全畅通！网关返回:", response.choices[0].message.content)
        print("👉 结论：DeepSeek API 连通正常，可以启动服务。")

    except Exception as e:
        print(f"❌ [探针报告] 物理链路被阻断！错误类型: {type(e).__name__}")
        print(f"📄 详细报错: {str(e)}")
        print("👉 请检查:")
        print("   1. DEEPSEEK_API_KEY 是否正确")
        print("   2. 是否需要配置代理 (HTTP_PROXY / HTTPS_PROXY)")
        print("   3. 防火墙是否放行 api.deepseek.com:443")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_network())
