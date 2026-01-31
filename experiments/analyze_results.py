#!/usr/bin/env python3
"""
结果分析脚本 - 汇总评测结果并生成报告

功能:
1. 收集所有模型的评测分数
2. 生成对比表格
3. 绘制可视化图表
4. 生成最终报告
"""

import json
import os
from pathlib import Path
from datetime import datetime

# ============== 配置 ==============

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

MODELS = [
    "base_model",
    "baseline_1k",
    "baseline_3k",
    "baseline_10k",
    "alignjuice_1k",
    "alignjuice_3k",
]

METRICS = ["ceval", "cmmlu"]


# ============== 工具函数 ==============

def load_eval_results(model_name: str) -> dict:
    """加载模型评测结果"""
    result_dir = RESULTS_DIR / model_name

    if not result_dir.exists():
        return {}

    results = {}

    # 尝试加载 lm-eval 格式的结果
    for json_file in result_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # lm-eval 格式
            if "results" in data:
                for task, scores in data["results"].items():
                    task_name = task.split(",")[0] if "," in task else task
                    if "acc" in scores:
                        results[task_name] = scores["acc"] * 100
                    elif "acc_norm" in scores:
                        results[task_name] = scores["acc_norm"] * 100

        except Exception as e:
            print(f"  警告: 无法加载 {json_file}: {e}")

    return results


def load_human_eval_results() -> dict:
    """加载人工评测结果"""
    human_eval_file = RESULTS_DIR / "human_eval.json"

    if not human_eval_file.exists():
        return {}

    with open(human_eval_file, "r") as f:
        return json.load(f)


def generate_comparison_table(all_results: dict) -> str:
    """生成对比表格"""
    lines = []

    # 表头
    header = "| 模型 | 数据量 | C-Eval | CMMLU | 人工评分 | 相对基线 |"
    separator = "|------|--------|--------|-------|----------|----------|"
    lines.append(header)
    lines.append(separator)

    # 获取 baseline_10k 作为基准
    baseline_10k_ceval = all_results.get("baseline_10k", {}).get("ceval", 0)

    # 数据行
    model_info = {
        "base_model": ("基座模型", "-"),
        "baseline_1k": ("Baseline", "1K"),
        "baseline_3k": ("Baseline", "3K"),
        "baseline_10k": ("Baseline", "10K"),
        "alignjuice_1k": ("AlignJuice", "1K"),
        "alignjuice_3k": ("AlignJuice", "3K"),
    }

    for model in MODELS:
        info = model_info.get(model, (model, "-"))
        results = all_results.get(model, {})

        ceval = results.get("ceval", "-")
        cmmlu = results.get("cmmlu", "-")
        human = results.get("human_eval", "-")

        # 计算相对基线
        if isinstance(ceval, (int, float)) and baseline_10k_ceval > 0:
            relative = f"{ceval / baseline_10k_ceval * 100:.1f}%"
        else:
            relative = "-"

        # 格式化数值
        ceval_str = f"{ceval:.1f}" if isinstance(ceval, (int, float)) else str(ceval)
        cmmlu_str = f"{cmmlu:.1f}" if isinstance(cmmlu, (int, float)) else str(cmmlu)
        human_str = f"{human:.2f}" if isinstance(human, (int, float)) else str(human)

        line = f"| {info[0]} | {info[1]} | {ceval_str} | {cmmlu_str} | {human_str} | {relative} |"
        lines.append(line)

    return "\n".join(lines)


def generate_analysis(all_results: dict) -> str:
    """生成分析结论"""
    analysis = []

    # 获取关键数据
    baseline_1k = all_results.get("baseline_1k", {}).get("ceval", 0)
    baseline_10k = all_results.get("baseline_10k", {}).get("ceval", 0)
    alignjuice_1k = all_results.get("alignjuice_1k", {}).get("ceval", 0)
    alignjuice_3k = all_results.get("alignjuice_3k", {}).get("ceval", 0)

    analysis.append("## 关键发现\n")

    # 分析 1: AlignJuice 1K vs Baseline 10K
    if alignjuice_1k and baseline_10k:
        ratio = alignjuice_1k / baseline_10k * 100
        if ratio >= 95:
            analysis.append(f"✅ **数据效率提升验证成功**: AlignJuice 1K 达到 Baseline 10K 的 {ratio:.1f}%")
            analysis.append(f"   - 数据效率提升: **10x** (1K vs 10K)")
        elif ratio >= 90:
            analysis.append(f"⚠️ **接近目标**: AlignJuice 1K 达到 Baseline 10K 的 {ratio:.1f}%")
        else:
            analysis.append(f"❌ **未达目标**: AlignJuice 1K 仅达到 Baseline 10K 的 {ratio:.1f}%")

    # 分析 2: AlignJuice 3K vs Baseline 10K
    if alignjuice_3k and baseline_10k:
        if alignjuice_3k > baseline_10k:
            improvement = (alignjuice_3k - baseline_10k) / baseline_10k * 100
            analysis.append(f"✅ **质量优势验证成功**: AlignJuice 3K 超过 Baseline 10K {improvement:.1f}%")
        else:
            analysis.append(f"⚠️ AlignJuice 3K 未超过 Baseline 10K")

    # 分析 3: 数据量 scaling
    if baseline_1k and baseline_10k:
        scaling = baseline_10k / baseline_1k
        analysis.append(f"\n📊 **Baseline Scaling**: 10K/1K = {scaling:.2f}x 效果提升")

    return "\n".join(analysis)


def generate_report(all_results: dict) -> str:
    """生成完整报告"""
    report = []

    report.append("# AlignJuice 效果验证实验报告\n")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## 实验概述\n")
    report.append("验证核心假设: **AlignJuice 处理后的 1K 数据 ≈ 原始 10K 数据的微调效果**\n")

    report.append("## 评测结果\n")
    report.append(generate_comparison_table(all_results))
    report.append("\n")

    report.append(generate_analysis(all_results))
    report.append("\n")

    report.append("## 结论\n")

    # 根据结果生成结论
    alignjuice_1k = all_results.get("alignjuice_1k", {}).get("ceval", 0)
    baseline_10k = all_results.get("baseline_10k", {}).get("ceval", 0)

    if alignjuice_1k and baseline_10k:
        ratio = alignjuice_1k / baseline_10k * 100
        if ratio >= 95:
            report.append("**实验成功**: AlignJuice 数据处理有效提升了数据效率，")
            report.append("使用 1/10 的数据量即可达到相近的模型效果。\n")
            report.append("\n这证明了 AlignJuice 的核心价值主张：")
            report.append("**高质量数据 > 大量低质量数据**\n")
        else:
            report.append("**实验部分成功**: AlignJuice 处理的数据质量有所提升，")
            report.append("但未完全达到 10x 数据效率的目标。\n")
            report.append("\n建议后续优化方向：")
            report.append("1. 调整去重阈值")
            report.append("2. 优化知识密度评分算法")
            report.append("3. 增加 LLM 合成增强")
    else:
        report.append("**数据不完整**: 请确保所有模型都已完成评测。\n")

    return "\n".join(report)


def try_plot_results(all_results: dict):
    """尝试绘制可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # 准备数据
        models = ["baseline_1k", "baseline_3k", "baseline_10k", "alignjuice_1k", "alignjuice_3k"]
        labels = ["Base 1K", "Base 3K", "Base 10K", "AJ 1K", "AJ 3K"]

        ceval_scores = [all_results.get(m, {}).get("ceval", 0) for m in models]
        cmmlu_scores = [all_results.get(m, {}).get("cmmlu", 0) for m in models]

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, ceval_scores, width, label='C-Eval', color='steelblue')
        bars2 = ax.bar(x + width/2, cmmlu_scores, width, label='CMMLU', color='coral')

        ax.set_xlabel('Model')
        ax.set_ylabel('Score (%)')
        ax.set_title('AlignJuice Effect Validation: Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 100)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "comparison_chart.png", dpi=150)
        print(f"  图表已保存: {RESULTS_DIR / 'comparison_chart.png'}")

    except ImportError:
        print("  提示: 安装 matplotlib 可生成可视化图表")
    except Exception as e:
        print(f"  警告: 无法生成图表: {e}")


# ============== 主函数 ==============

def main():
    print("=" * 60)
    print("AlignJuice 效果验证实验 - 结果分析")
    print("=" * 60)

    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 收集所有结果
    print("\n[1] 收集评测结果")
    all_results = {}

    for model in MODELS:
        print(f"  加载: {model}")
        results = load_eval_results(model)
        if results:
            all_results[model] = results
            print(f"    C-Eval: {results.get('ceval', '-')}")
            print(f"    CMMLU: {results.get('cmmlu', '-')}")
        else:
            print(f"    (无数据)")

    # 加载人工评测结果
    print("\n[2] 加载人工评测结果")
    human_results = load_human_eval_results()
    if human_results:
        for model, score in human_results.items():
            if model in all_results:
                all_results[model]["human_eval"] = score
            else:
                all_results[model] = {"human_eval": score}
        print(f"  已加载 {len(human_results)} 个模型的人工评分")
    else:
        print("  (无人工评测数据)")

    # 生成报告
    print("\n[3] 生成分析报告")
    report = generate_report(all_results)

    report_path = RESULTS_DIR / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  报告已保存: {report_path}")

    # 保存汇总数据
    summary_path = RESULTS_DIR / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  汇总数据已保存: {summary_path}")

    # 尝试生成图表
    print("\n[4] 生成可视化图表")
    try_plot_results(all_results)

    # 打印报告
    print("\n" + "=" * 60)
    print("分析报告预览")
    print("=" * 60)
    print(report)

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {report_path}")
    print(f"  - {summary_path}")
    if (RESULTS_DIR / "comparison_chart.png").exists():
        print(f"  - {RESULTS_DIR / 'comparison_chart.png'}")


if __name__ == "__main__":
    main()
