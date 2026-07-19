# Miemie-Agent-RAG/tests/evaluate_report.py
import asyncio
import logging
import os
import re
import sys
import time

import numpy as np
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from app.graph.workflow import create_workflow
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("miemie-rag.eval")

# ── 初始化 ──────────────────────────────────────────

logger.info("挂载 LangGraph 工作流...")
app_graph = create_workflow(streaming=False)

judge_llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    temperature=0.1,
)

# ── Benchmark 测试集 ────────────────────────────────

TEST_DATASET = [
    "设计一套面向 10 万并发的高吞吐大模型推理架构，说明如何利用 PagedAttention 解决 KV Cache 内存碎片问题。",
    "针对 70B MoE 大模型，在显存限制为 4 张 A100(80G) 下，设计 Tensor/Pipeline Parallelism 分布式策略。",
    "分析 DeepSeek-V3 的 MLA 机制如何减少 KV Cache 内存占用，并评估其落地可行性。",
]


# ── 评测函数 ────────────────────────────────────────

async def llm_judge_score(task: str, answer: str) -> float:
    """用独立裁判模型对 RAG 回答进行评分（0-10）"""
    if "系统提示" in answer or "暂时无法响应" in answer:
        return 0.0

    prompt = f"""你是一名资深 AI Infra 架构师，请对以下 RAG 系统的回答进行严格评分。

[问题]: {task}
[RAG 回答]: {answer}

评分维度（各占 1/3 权重，满分 10）：
1. 事实准确性：是否存在技术常识错误或幻觉？
2. 方案完备性：是否触及核心原理？
3. 知识密度：内容是否充实且切中要害？

请输出最终评定，格式严格为: [SCORE]: X.X
"""
    try:
        res = await judge_llm.ainvoke([HumanMessage(content=prompt)])
        match = re.search(r"\[SCORE\]:\s*([0-9.]+)", res.content)
        return float(match.group(1)) if match else 5.0
    except Exception as e:
        logger.warning("裁判模型异常: %s", e)
        return 5.0


def calculate_bootstrap_ci(data, n_bootstraps=1000, ci_level=0.95):
    """Bootstrap 法计算均值的置信区间"""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0

    bootstrapped_means = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))

    mean = np.mean(data)
    lower = float(np.percentile(bootstrapped_means, (1 - ci_level) / 2 * 100))
    upper = float(np.percentile(bootstrapped_means, (1 + ci_level) / 2 * 100))
    return mean, lower, upper


# ── 评测主流程 ──────────────────────────────────────

async def run_production_eval():
    logger.info("启动 RAG 质量评测")
    scores = []
    durations = []

    for i, task in enumerate(TEST_DATASET):
        logger.info("评测样本 [%d/%d]: %.30s...", i + 1, len(TEST_DATASET), task)

        start_time = time.time()
        try:
            agent_res = await app_graph.ainvoke({"question": task})
            duration = time.time() - start_time
            durations.append(duration)

            final_answer = agent_res.get("answer", "")
            score = await llm_judge_score(task, final_answer)
            scores.append(score)

            logger.info(
                "  耗时: %.2fs | 得分: %.1f/10.0", duration, score
            )
        except Exception as e:
            logger.error("样本 %d 异常: %s", i + 1, e)

    if scores:
        mean_s, lower_s, upper_s = calculate_bootstrap_ci(scores)
        print("\n" + "=" * 55)
        print("RAG 质量评测报告")
        print(f"  平均得分:        {mean_s:.2f} / 10.0")
        print(f"  Bootstrap 95% CI: [{lower_s:.2f}, {upper_s:.2f}]")
        if durations:
            p50 = np.percentile(durations, 50)
            p99 = np.percentile(durations, 99)
            print(f"  P50 延迟:         {p50:.2f}s")
            print(f"  P99 延迟:         {p99:.2f}s")
        print("=" * 55)


if __name__ == "__main__":
    asyncio.run(run_production_eval())
