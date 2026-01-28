#!/usr/bin/env python3
"""
Convert baike dataset to AlignJuice format.

Input format:  {"question": "...", "answer": "..."}
Output format: {"id": "...", "instruction": "...", "input": "", "output": "...", "category": "..."}
"""

import json
import random
import re
import sys
from pathlib import Path


# Category classification keywords
CATEGORY_KEYWORDS = {
    "factual": [
        "是什么", "什么是", "谁是", "哪个", "哪里", "多少", "几", "为什么",
        "历史", "地理", "科学", "技术", "定义", "解释", "介绍", "描述",
        "首都", "人口", "面积", "发明", "发现", "创始", "成立",
    ],
    "reasoning": [
        "如何", "怎么", "怎样", "为什么", "原因", "计算", "分析", "比较",
        "区别", "优缺点", "步骤", "方法", "过程", "原理", "机制",
    ],
    "creative": [
        "写一", "编一", "创作", "想象", "故事", "诗", "歌", "小说",
        "设计", "构思", "发挥", "创意",
    ],
    "daily": [
        "你好", "谢谢", "再见", "喜欢", "感觉", "心情", "天气", "今天",
        "最近", "怎么样", "好吗", "吃", "喝", "玩", "睡", "工作", "生活",
    ],
}


def classify_category(question: str) -> str:
    """Classify question into category based on keywords."""
    question_lower = question.lower()

    # Check each category
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question_lower:
                scores[category] += 1

    # Return category with highest score, default to "daily"
    max_score = max(scores.values())
    if max_score == 0:
        return "daily"

    for category, score in scores.items():
        if score == max_score:
            return category

    return "daily"


def convert_sample(idx: int, raw: dict) -> dict:
    """Convert a single sample to AlignJuice format."""
    question = raw.get("question", "").strip()
    answer = raw.get("answer", "").strip()

    return {
        "id": f"baike_{idx:06d}",
        "instruction": question,
        "input": "",
        "output": answer,
        "category": classify_category(question),
    }


def main():
    # Paths
    input_path = Path.home() / "Downloads" / "data-baike" / "train.jsonl"
    output_path = Path(__file__).parent.parent / "data" / "baike_sample_3000.jsonl"

    # Parameters
    sample_size = 3000
    seed = 42

    print(f"Reading from: {input_path}")
    print(f"Output to: {output_path}")
    print(f"Sample size: {sample_size}")

    # Read all data
    all_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    print(f"Total records: {len(all_data)}")

    # Random sample
    random.seed(seed)
    sampled = random.sample(all_data, min(sample_size, len(all_data)))
    print(f"Sampled: {len(sampled)}")

    # Convert format
    converted = []
    category_counts = {}
    for idx, raw in enumerate(sampled):
        sample = convert_sample(idx, raw)
        converted.append(sample)
        cat = sample["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in converted:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nConversion complete!")
    print(f"Output: {output_path}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(converted) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
