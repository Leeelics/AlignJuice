#!/usr/bin/env python3
"""
AlignJuice 完整实战测试 - 百科数据处理

包含语义去重，验证完整流程
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignjuice.core.data_container import DataContainer
from alignjuice.operators.dedup.exact import ExactDedup
from alignjuice.operators.dedup.semantic import SemanticDedup
from alignjuice.operators.filter.quality import QualityFilter
from alignjuice.operators.filter.knowledge import KnowledgeFilter
from alignjuice.operators.filter.diversity import DiversityFilter
from alignjuice.stages.s4_sandbox_eval import SandboxEvalStage


def main():
    print("=" * 60)
    print("AlignJuice 完整实战测试 - 百科数据处理")
    print("=" * 60)

    # 1. 加载数据
    input_path = Path(__file__).parent.parent / "data" / "baike_sample_3000.jsonl"
    print(f"\n[Stage 1] 加载数据")
    print(f"  路径: {input_path}")
    data = DataContainer.from_jsonl(input_path)
    print(f"  加载完成: {len(data)} 条")

    # 2. 精确去重
    print(f"\n[Stage 2] 精确去重")
    exact_dedup = ExactDedup(field="instruction")
    data = exact_dedup(data)
    print(f"  结果: {len(data)} 条 (去除 {exact_dedup.metrics['removed_count']} 条精确重复)")

    # 3. 语义去重 (这是关键步骤，数据集有大量语义相似问题)
    print(f"\n[Stage 3] 语义去重 (threshold=0.95)")
    print(f"  正在计算嵌入向量...")
    semantic_dedup = SemanticDedup(
        threshold=0.95,
        field="instruction",
        embedding_backend="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
    )
    data = semantic_dedup(data)
    print(f"  结果: {len(data)} 条 (去除 {semantic_dedup.metrics['removed_count']} 条语义重复)")
    print(f"  去重率: {semantic_dedup.metrics['dedup_rate']:.1%}")

    # 4. 质量过滤
    print(f"\n[Stage 4] 质量过滤 (threshold=0.7)")
    quality_filter = QualityFilter(
        threshold=0.7,
        min_instruction_length=5,
        min_output_length=10,
    )
    data = quality_filter(data)
    print(f"  结果: {len(data)} 条")
    print(f"  平均质量分: {quality_filter.metrics['avg_score']:.2f}")

    # 5. 多样性选择
    print(f"\n[Stage 5] 多样性选择 (target=1500)")
    if len(data) > 1500:
        diversity_filter = DiversityFilter(
            target_count=1500,
            selection_method="maxmin",
        )
        data = diversity_filter(data)
        print(f"  结果: {len(data)} 条")
        print(f"  多样性分数: {diversity_filter.metrics.get('diversity_score', 'N/A')}")
    else:
        print(f"  跳过 (当前 {len(data)} 条 < 1500)")

    # 6. 知识密度筛选
    print(f"\n[Stage 6] 知识密度筛选 (top_k=1000)")
    if len(data) > 1000:
        knowledge_filter = KnowledgeFilter(
            top_k=1000,
            embedding_backend="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
        )
        data = knowledge_filter(data)
        print(f"  结果: {len(data)} 条")
        print(f"  平均知识分: {knowledge_filter.metrics.get('avg_score', 'N/A')}")
    else:
        print(f"  跳过 (当前 {len(data)} 条 <= 1000)")

    # 7. 评估报告
    print(f"\n[Stage 7] 生成评估报告")
    eval_stage = SandboxEvalStage(
        report_path="reports/baike_full_report.html",
        metrics_path="reports/baike_full_metrics.json",
    )
    data = eval_stage.process(data)
    print(f"  报告已生成")

    # 8. 保存结果
    output_path = Path(__file__).parent.parent / "output" / "baike_final_full.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_jsonl(output_path)

    # 最终统计
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)

    print(f"""
处理流程:
  输入:     3000 条
  精确去重: → {3000 - exact_dedup.metrics['removed_count']} 条
  语义去重: → {3000 - exact_dedup.metrics['removed_count'] - semantic_dedup.metrics['removed_count']} 条
  质量过滤: → {quality_filter.metrics['output_count']} 条
  最终输出: → {len(data)} 条

输出文件:
  - output/baike_final_full.jsonl
  - reports/baike_full_report.html
  - reports/baike_full_metrics.json
""")

    # 显示类别分布
    print("类别分布:")
    categories = {}
    for s in data:
        categories[s.category] = categories.get(s.category, 0) + 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # 显示样例
    print(f"\n最终数据样例 (3条):")
    for i, sample in enumerate(list(data)[:3]):
        print(f"\n  [{i+1}] ID: {sample.id} | Category: {sample.category}")
        print(f"      Q: {sample.instruction}")
        print(f"      A: {sample.output[:100]}...")


if __name__ == "__main__":
    main()
