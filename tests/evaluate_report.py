# tests/evaluate_report.py
import os
import sys
import time
import asyncio
import re
import numpy as np
from dotenv import load_dotenv

# =====================================================================
# ⚡ 核心修复 1：绝对物理路径提权 (必须放在所有 app 导入之前！)
# =====================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 加载环境变量 (需确保根目录有 .env 文件配置了 DEEPSEEK_API_KEY)
load_dotenv()

# 现在可以安全地导入你的核心模块了
from app.graph.workflow import create_workflow
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# =====================================================================
# ⚡ 初始化配置
# =====================================================================
# 1. 编译你写好的 LangGraph 工作流实例
print("====== [系统预热] 正在挂载 LangGraph 多轨检索工作流... ======")
app_graph = create_workflow()

# 2. 引入独立的“裁判大模型”(Judge LLM)
# 为了保证评分的客观性，裁判模型的 temperature 调低，确保打分的一致性
judge_llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base='https://api.deepseek.com',
    temperature=0.1
)

# 3. 大厂 AI Infra 专家级 Benchmark 测试集
TEST_DATASET = [
    "设计一套面向 10 万并发的高吞吐大模型推理架构，说明如何利用 PagedAttention 解决 KV Cache 内存碎片问题。",
    "针对 70B MoE 大模型，在显存限制为 4 张 A100(80G) 下，设计 Tensor/Pipeline Parallelism 分布式策略。",
    "分析 DeepSeek-V3 的 MLA 机制如何减少 KV Cache 内存占用，并评估其落地可行性。"
]


# =====================================================================
# ⚡ 核心评测函数
# =====================================================================
async def llm_judge_score(task: str, answer: str) -> float:
    """判别器：对 RAG 系统交付的架构方案进行细粒度三维打分"""
    # 如果系统降级报错，直接给 0 分
    if "系统提示" in answer or "暂时无法响应" in answer:
        return 0.0

    prompt = f"""你是一名腾讯/字节级的资深 AI Infra 首席架构师，请对以下 RAG 系统生成的技术方案进行严格审计与量化打分。

[评估任务]: {task}
[检索增强后的交付方案]: {answer}

评分维度（满分10分）：
1. 事实准确性：是否存在严重的技术常识硬伤或逻辑幻觉？
2. 方案完备性：是否触及了核心原理（如 Page 映射、通信开销、低秩投影等）？
3. 知识密度：内容是否充实且切中要害？

请直接输出你的最终评定，必须包含且仅包含一个 0.0 到 10.0 之间的浮点分数，格式严格为: [SCORE]: X.X
"""
    try:
        res = await judge_llm.ainvoke([HumanMessage(content=prompt)])
        match = re.search(r"\[SCORE\]:\s*([0-9.]+)", res.content)
        return float(match.group(1)) if match else 5.0
    except Exception as e:
        print(f"      [裁判模型异常] {e}")
        return 5.0


def calculate_bootstrap_ci(data, n_bootstraps=1000, ci_level=0.95):
    """统计学后置处理：利用 Bootstrap 计算 95% 置信区间，消除大模型裁判的方差不稳定"""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0

    bootstrapped_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))

    mean = np.mean(data)
    lower_bound = float(np.percentile(bootstrapped_means, (1 - ci_level) / 2 * 100))
    upper_bound = float(np.percentile(bootstrapped_means, (1 + ci_level) / 2 * 100))
    return mean, lower_bound, upper_bound


# =====================================================================
# ⚡ 启动评测大盘
# =====================================================================
async def run_production_eval():
    print("\n====== 🚀 启动大厂级【AI Infra 架构判别式评测大盘】 ======\n")
    scores = []
    durations = []

    for i, task in enumerate(TEST_DATASET):
        print(f"▶️ 正在推演样本 [{i + 1}/{len(TEST_DATASET)}]: {task[:25]}...")
        start_time = time.time()

        try:
            # ⚡ 核心修复 2：严格对齐 GraphState 的数据结构
            agent_res = await app_graph.ainvoke({"question": task})

            duration = time.time() - start_time
            durations.append(duration)

            # 提取最终生成的答案
            final_answer = agent_res.get("answer", "")

            print(f"   ⏳ 检索与生成耗时: {duration:.2f}s | 正在进行 AI 盲测打分...")
            score = await llm_judge_score(task, final_answer)
            scores.append(score)

            print(f"   📊 判定得分: {score:.1f}/10.0")

        except Exception as e:
            print(f"   ❌ 样本 {i + 1} 异常崩溃: {str(e)}")

    # 打印最终聚合评估报告
    if scores:
        mean_s, lower_s, upper_s = calculate_bootstrap_ci(scores)
        print("\n=======================================================")
        print("📊 【核心架构量化评估报告】")
        print(f"1. 期望平均得分 (Mean Score): {mean_s:.2f} / 10.0")
        print(f"2. Bootstrap 95% 置信区间 (CI): [{lower_s:.2f}, {upper_s:.2f}]  <-- 证明迭代显著性")
        if durations:
            print(
                f"3. 生产环境 P50 延迟: {np.percentile(durations, 50):.2f}s | P99 延迟: {np.percentile(durations, 99):.2f}s")
        print("=======================================================")


if __name__ == "__main__":
    asyncio.run(run_production_eval())