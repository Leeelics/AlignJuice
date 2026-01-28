#!/usr/bin/env python3
"""
AlignJuice 实战测试脚本 - 百科数据处理

简化版流程：不依赖 sentence-transformers，使用基础算子验证流程
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignjuice.core.data_container import DataContainer
from alignjuice.operators.dedup.exact import ExactDedup
from alignjuice.operators.filter.quality import QualityFilter
from alignjuice.stages.s4_sandbox_eval import SandboxEvalStage


def main():
    print("=" * 60)
    print("AlignJuice 实战测试 - 百科数据处理")
    print("=" * 60)

    # 1. 加载数据
    input_path = Path(__file__).parent.parent / "data" / "baike_sample_3000.jsonl"
    print(f"\n[Step 1] 加载数据: {input_path}")
    data = DataContainer.from_jsonl(input_path)
    print(f"  加载完成: {len(data)} 条")
    data.describe()

    # 2. 精确去重
    print(f"\n[Step 2] 精确去重 (field=instruction)")
    exact_dedup = ExactDedup(field="instruction")
    data = exact_dedup(data)
    print(f"  去重后: {len(data)} 条")
    print(f"  去重率: {exact_dedup.metrics['dedup_rate']:.1%}")

    # 3. 质量过滤
    print(f"\n[Step 3] 质量过滤 (threshold=0.7)")
    quality_filter = QualityFilter(
        threshold=0.7,
        min_instruction_length=5,
        min_output_length=10,
    )
    data = quality_filter(data)
    print(f"  过滤后: {len(data)} 条")
    print(f"  平均质量分: {quality_filter.metrics['avg_score']:.2f}")

    # 4. 随机采样到目标数量 (简化版，替代知识过滤)
    target_count = 1000
    print(f"\n[Step 4] 采样到目标数量: {target_count}")
    if len(data) > target_count:
        data = data.sample(target_count, seed=42)
    print(f"  最终数量: {len(data)} 条")

    # 5. 评估报告
    print(f"\n[Step 5] 生成评估报告")
    eval_stage = SandboxEvalStage(
        report_path="reports/baike_quality_report.html",
        metrics_path="reports/baike_metrics.json",
    )
    data = eval_stage.process(data)

    # 6. 保存结果
    output_path = Path(__file__).parent.parent / "output" / "baike_final.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_jsonl(output_path)
    print(f"\n[Step 6] 保存结果: {output_path}")

    # 7. 显示最终统计
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输入: 3000 条")
    print(f"输出: {len(data)} 条")
    print(f"\n报告位置:")
    print(f"  - reports/baike_quality_report.html")
    print(f"  - reports/baike_metrics.json")
    print(f"\n数据位置:")
    print(f"  - {output_path}")

    # 显示样例
    print(f"\n最终数据样例:")
    for i, sample in enumerate(data.sample(3)):
        print(f"\n  [{i+1}] {sample.category}")
        print(f"      Q: {sample.instruction[:50]}...")
        print(f"      A: {sample.output[:50]}...")


if __name__ == "__main__":
    main()
