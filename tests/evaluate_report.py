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
    # ── 推理架构（PagedAttention、Continuous Batching） ──
    "设计一套面向 10 万并发的高吞吐大模型推理架构，说明如何利用 PagedAttention 解决 KV Cache 内存碎片问题。",
    "vLLM 的 Continuous Batching 是如何工作的？相比传统静态 batching 有哪些优势？",
    # ── 分布式策略（MoE、Tensor/Pipeline Parallelism） ──
    "针对 70B MoE 大模型，在显存限制为 4 张 A100(80G) 下，设计 Tensor/Pipeline Parallelism 分布式策略。",
    # ── 注意力机制（MLA、Flash Attention、KV Cache） ──
    "分析 DeepSeek-V3 的 MLA 机制如何减少 KV Cache 内存占用，并评估其落地可行性。",
    "对比 Flash Attention 和标准 Self-Attention 在显存占用和计算效率上的核心区别。",
    "什么是 KV Cache？在长序列推理时它为什么成为显存瓶颈？有哪些优化手段？",
    # ── 推理加速（Speculative Decoding） ──
    "什么是 Speculative Decoding？草稿模型和目标模型之间如何协同工作来加速推理？",
    # ── 微调与量化（LoRA、GPTQ vs AWQ） ──
    "解释 LoRA 微调的原理。在显存受限的场景下，LoRA 相比 Full Fine-tuning 有哪些优势？",
    "大模型推理中，量化（Quantization）技术如何平衡精度和速度？分析 GPTQ 和 AWQ 两种方案的核心思路。",
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

async def run_comparison_eval():
    """对比评测：RRF vs 线性加权融合（多 alpha 消融）"""
    from app.services.retriever import reset_retriever_singleton, get_milvus_retriever

    configs = [
        ("RRF (k=60)", "rrf", None),
        ("Linear α=0.3 (BM25 偏重)", "linear_weighted", 0.3),
        ("Linear α=0.5 (等权)", "linear_weighted", 0.5),
        ("Linear α=0.7 (Dense 偏重)", "linear_weighted", 0.7),
    ]

    results: dict[str, dict] = {}

    for label, method, alpha in configs:
        # 重置单例，以目标融合策略重建检索器
        reset_retriever_singleton()
        get_milvus_retriever(fusion_method=method, fusion_alpha=alpha or 0.5)

        logger.info("=== 评测融合方法: %s ===", label)

        scores = []
        durations = []
        per_q: list[tuple[str, float, float]] = []

        for i, task in enumerate(TEST_DATASET):
            logger.info(
                "评测样本 [%d/%d] (%s): %.30s...",
                i + 1, len(TEST_DATASET), method, task,
            )

            start_time = time.time()
            try:
                agent_res = await app_graph.ainvoke({"question": task})
                duration = time.time() - start_time
                durations.append(duration)

                final_answer = agent_res.get("answer", "")
                score = await llm_judge_score(task, final_answer)
                scores.append(score)
                per_q.append((task[:40], score, duration))

                logger.info("  耗时: %.2fs | 得分: %.1f/10.0", duration, score)
            except Exception as e:
                logger.error("样本 %d 异常: %s", i + 1, e)
                scores.append(0.0)
                durations.append(0.0)
                per_q.append((task[:40], 0.0, 0.0))

        results[label] = {"scores": scores, "durations": durations, "per_q": per_q}

    # ── 输出对比报告 ──────────────────────────────────
    _print_comparison_report(results)


def _print_comparison_report(results: dict[str, dict]):
    """格式化输出融合方法对比报告"""
    labels = list(results.keys())
    if not labels:
        return

    col_w = 22  # 每列宽度
    sep = "=" * 100

    print("\n" + sep)
    print("RAG 融合方法对比评测报告（含 α 消融实验）".center(98))
    print(sep)

    # ── 逐样本得分 ──
    n_questions = len(results[labels[0]]["per_q"])
    print(f"\n{'逐样本得分 (0-10)':-^100s}")
    header = f"{'#':<4s}"
    for label in labels:
        header += f"{label:^{col_w}s}"
    print(header)
    print("-" * 100)

    for i in range(n_questions):
        row = f"{i+1:<4d}"
        for label in labels:
            _, score, _ = results[label]["per_q"][i]
            row += f"{score:^{col_w}.2f}"
        print(row)

    # ── 聚合统计 ──
    print(f"\n{'聚合统计':-^100s}")

    metrics = [
        ("平均得分", lambda s, d: f"{np.mean(s):.2f}" if s else "N/A"),
        ("Bootstrap 95% CI", lambda s, d: (
            f"[{calculate_bootstrap_ci(s)[1]:.2f}, {calculate_bootstrap_ci(s)[2]:.2f}]"
            if s else "N/A"
        )),
        ("P50 延迟", lambda s, d: f"{np.percentile(d, 50):.1f}s" if d else "N/A"),
        ("P99 延迟", lambda s, d: f"{np.percentile(d, 99):.1f}s" if d else "N/A"),
    ]

    for metric_name, metric_fn in metrics:
        row = f"{metric_name:<20s}"
        for label in labels:
            s = results[label]["scores"]
            d = results[label]["durations"]
            row += f"{metric_fn(s, d):^{col_w}s}"
        print(row)

    # ── 排名 ──
    print(f"\n{'综合排名（按平均得分）':-^100s}")
    ranked = sorted(labels, key=lambda lb: np.mean(results[lb]["scores"]), reverse=True)
    for rank, label in enumerate(ranked, 1):
        mean_s = np.mean(results[label]["scores"])
        marker = "👑" if rank == 1 else "  "
        print(f"  {marker} #{rank}  {label:<35s}  {mean_s:.2f}")

    # ── 结论 ──
    print("\n" + "-" * 100)
    best = ranked[0]
    baseline_label = labels[0]  # RRF
    best_mean = np.mean(results[best]["scores"])
    baseline_mean = np.mean(results[baseline_label]["scores"])
    delta = best_mean - baseline_mean

    # 找到最佳 alpha
    linear_labels = [lb for lb in labels if "Linear" in lb]
    best_alpha = None
    if linear_labels:
        best_alpha = max(linear_labels, key=lambda lb: np.mean(results[lb]["scores"]))

    print(f"最佳方案: {best}（平均 {best_mean:.2f} 分）")
    if best != baseline_label:
        print(f"相对 RRF 基线提升: {delta:+.2f} 分")
    if best_alpha:
        print(f"最佳 α 值: {best_alpha}")
    print(f"测试集规模: n={n_questions}，建议扩充到 30+ 题以获得统计显著性。")
    print(sep + "\n")


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
    import sys
    if "--compare" in sys.argv:
        asyncio.run(run_comparison_eval())
    else:
        asyncio.run(run_production_eval())
