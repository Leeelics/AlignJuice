#!/usr/bin/env python3
"""
数据准备脚本 - 为效果验证实验准备训练数据

生成 5 组数据:
- baseline_1k: 随机采样 1000 条
- baseline_3k: 随机采样 3000 条
- baseline_10k: 随机采样 10000 条
- alignjuice_1k: AlignJuice 处理后 1000 条
- alignjuice_3k: AlignJuice 处理后 3000 条
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alignjuice.core.data_container import DataContainer
from alignjuice.operators.dedup.exact import ExactDedup
from alignjuice.operators.filter.quality import QualityFilter


# ============== 配置 ==============

# 原始数据路径
RAW_DATA_PATH = Path.home() / "Downloads" / "data-baike" / "train.jsonl"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "data"

# 随机种子 (保证可复现)
SEED = 42

# 数据量配置
BASELINE_SIZES = [1000, 3000, 10000]
ALIGNJUICE_SIZES = [1000, 3000]

# AlignJuice 处理需要的原始数据量 (处理后会减少)
ALIGNJUICE_RAW_MULTIPLIER = 5  # 处理 5x 数据得到目标数量


# ============== 工具函数 ==============

def load_raw_data(path: Path, limit: int = None) -> list[dict]:
    """加载原始数据"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def convert_to_sft_format(samples: list[dict], prefix: str = "sample") -> list[dict]:
    """转换为 LLaMA-Factory SFT 格式"""
    sft_data = []
    for i, sample in enumerate(samples):
        # 支持两种输入格式
        if "instruction" in sample:
            instruction = sample["instruction"]
            output = sample["output"]
        else:
            instruction = sample.get("question", "")
            output = sample.get("answer", "")

        sft_data.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })
    return sft_data


def save_sft_data(data: list[dict], path: Path):
    """保存为 LLaMA-Factory 格式 (JSON 数组)"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  保存: {path} ({len(data)} 条)")


def classify_category(question: str) -> str:
    """简单分类"""
    keywords = {
        "factual": ["是什么", "什么是", "谁是", "哪个", "哪里", "多少", "历史", "介绍"],
        "reasoning": ["如何", "怎么", "怎样", "为什么", "原因", "步骤", "方法"],
        "creative": ["写一", "编一", "创作", "想象", "故事"],
    }
    for cat, kws in keywords.items():
        if any(kw in question for kw in kws):
            return cat
    return "daily"


# ============== AlignJuice 处理 ==============

def process_with_alignjuice(raw_data: list[dict], target_count: int) -> list[dict]:
    """使用 AlignJuice 处理数据"""

    # 转换为 AlignJuice 格式
    samples = []
    for i, item in enumerate(raw_data):
        question = item.get("question", "")
        answer = item.get("answer", "")
        samples.append({
            "id": f"raw_{i:06d}",
            "instruction": question,
            "input": "",
            "output": answer,
            "category": classify_category(question),
        })

    # 创建 DataContainer
    data = DataContainer.from_list(samples)
    print(f"    原始数据: {len(data)} 条")

    # Stage 1: 精确去重
    exact_dedup = ExactDedup(field="instruction")
    data = exact_dedup(data)
    print(f"    精确去重后: {len(data)} 条 (-{exact_dedup.metrics['removed_count']})")

    # Stage 2: 质量过滤
    quality_filter = QualityFilter(
        threshold=0.7,
        min_instruction_length=5,
        min_output_length=10,
    )
    data = quality_filter(data)
    print(f"    质量过滤后: {len(data)} 条")

    # Stage 3: TF-IDF 语义去重 (简化版)
    data, removed = semantic_dedup_tfidf(data, threshold=0.85)
    print(f"    语义去重后: {len(data)} 条 (-{removed})")

    # Stage 4: 知识密度排序，选择 Top-K
    scored_samples = score_knowledge_density(data)
    top_samples = [s for _, s in scored_samples[:target_count]]

    print(f"    知识筛选后: {len(top_samples)} 条")

    # 转换回字典格式
    result = []
    for sample in top_samples:
        result.append({
            "question": sample.instruction,
            "answer": sample.output,
        })

    return result


def semantic_dedup_tfidf(data: DataContainer, threshold: float = 0.85):
    """TF-IDF 语义去重"""
    import numpy as np
    from collections import Counter

    def simple_tokenize(text: str) -> list[str]:
        tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
            elif char.isalnum():
                tokens.append(char.lower())
        return tokens

    texts = [s.instruction for s in data]
    tokenized = [simple_tokenize(t) for t in texts]

    # Build vocabulary
    vocab = {}
    doc_freq = Counter()
    for tokens in tokenized:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
            if token not in vocab:
                vocab[token] = len(vocab)

    n_docs = len(texts)
    filtered_vocab = {
        token: idx for token, idx in vocab.items()
        if 2 <= doc_freq[token] < 0.8 * n_docs
    }
    vocab = {token: i for i, token in enumerate(filtered_vocab.keys())}
    vocab_size = len(vocab) if vocab else len(doc_freq)

    if not vocab:
        vocab = {token: i for i, token in enumerate(doc_freq.keys())}

    # Compute TF-IDF
    vectors = np.zeros((len(texts), len(vocab)))
    for i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        for token, count in tf.items():
            if token in vocab:
                idx = vocab[token]
                idf = np.log(n_docs / (doc_freq[token] + 1))
                vectors[i, idx] = count * idf

    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    # Greedy deduplication
    n = len(vectors)
    keep_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep_mask[i]:
            continue
        if i + 1 < n:
            similarities = np.dot(vectors[i], vectors[i + 1:].T)
            duplicate_indices = np.where(similarities >= threshold)[0] + i + 1
            keep_mask[duplicate_indices] = False

    kept_indices = np.where(keep_mask)[0]
    kept_samples = [data[i] for i in kept_indices]
    removed_count = n - len(kept_samples)

    return DataContainer(
        samples=kept_samples,
        provenance=data.provenance + [f"semantic_dedup: {n} -> {len(kept_samples)}"],
    ), removed_count


def score_knowledge_density(data: DataContainer) -> list:
    """知识密度评分"""
    scored_samples = []

    for sample in data:
        output = sample.output
        instruction = sample.instruction

        # Length score
        length_score = min(1.0, len(output) / 200)

        # Unique ratio
        chars = [c for c in output if '\u4e00' <= c <= '\u9fff']
        unique_ratio = len(set(chars)) / max(len(chars), 1)

        # Factual indicators
        factual_score = 0.5
        if any(c.isdigit() for c in output):
            factual_score += 0.2
        if any(word in output for word in ['是', '为', '有', '在', '年', '月', '日']):
            factual_score += 0.1
        if any(word in instruction for word in ['什么', '哪', '谁', '如何', '为什么']):
            factual_score += 0.2

        knowledge_score = (length_score + unique_ratio + min(1.0, factual_score)) / 3
        scored_samples.append((knowledge_score, sample))

    scored_samples.sort(key=lambda x: x[0], reverse=True)
    return scored_samples


# ============== 主函数 ==============

def main():
    print("=" * 60)
    print("AlignJuice 效果验证实验 - 数据准备")
    print("=" * 60)

    # 设置随机种子
    random.seed(SEED)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载原始数据
    print(f"\n[1] 加载原始数据: {RAW_DATA_PATH}")
    all_raw_data = load_raw_data(RAW_DATA_PATH)
    print(f"    总量: {len(all_raw_data)} 条")

    # 打乱数据
    random.shuffle(all_raw_data)

    # ============== 生成 Baseline 数据 ==============
    print(f"\n[2] 生成 Baseline 数据 (随机采样)")

    for size in BASELINE_SIZES:
        print(f"\n  baseline_{size//1000}k:")
        sampled = all_raw_data[:size]
        sft_data = convert_to_sft_format(sampled, f"baseline_{size//1000}k")
        save_sft_data(sft_data, OUTPUT_DIR / f"baseline_{size//1000}k.json")

    # ============== 生成 AlignJuice 数据 ==============
    print(f"\n[3] 生成 AlignJuice 数据 (处理后)")

    for target_size in ALIGNJUICE_SIZES:
        print(f"\n  alignjuice_{target_size//1000}k:")

        # 取足够多的原始数据进行处理
        raw_size = target_size * ALIGNJUICE_RAW_MULTIPLIER
        raw_subset = all_raw_data[:raw_size]

        # AlignJuice 处理
        processed = process_with_alignjuice(raw_subset, target_size)

        # 转换并保存
        sft_data = convert_to_sft_format(processed, f"alignjuice_{target_size//1000}k")
        save_sft_data(sft_data, OUTPUT_DIR / f"alignjuice_{target_size//1000}k.json")

    # ============== 生成数据集注册文件 ==============
    print(f"\n[4] 生成 LLaMA-Factory 数据集注册文件")

    dataset_info = {}
    for size in BASELINE_SIZES:
        name = f"baseline_{size//1000}k"
        dataset_info[name] = {
            "file_name": f"baseline_{size//1000}k.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }

    for size in ALIGNJUICE_SIZES:
        name = f"alignjuice_{size//1000}k"
        dataset_info[name] = {
            "file_name": f"alignjuice_{size//1000}k.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }

    config_dir = PROJECT_ROOT / "experiments" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"  保存: {config_dir / 'dataset_info.json'}")

    # ============== 统计信息 ==============
    print(f"\n[5] 数据统计")
    print("-" * 40)

    for json_file in sorted(OUTPUT_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 计算平均长度
        avg_inst_len = sum(len(d["instruction"]) for d in data) / len(data)
        avg_out_len = sum(len(d["output"]) for d in data) / len(data)

        print(f"  {json_file.stem}:")
        print(f"    数量: {len(data)}")
        print(f"    平均指令长度: {avg_inst_len:.1f}")
        print(f"    平均输出长度: {avg_out_len:.1f}")

    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)
    print(f"\n数据目录: {OUTPUT_DIR}")
    print(f"配置目录: {config_dir}")


if __name__ == "__main__":
    main()
