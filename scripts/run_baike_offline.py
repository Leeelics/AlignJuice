#!/usr/bin/env python3
"""
AlignJuice 实战测试 - 百科数据处理 (离线版)

使用 TF-IDF 替代 sentence-transformers，无需网络下载模型
"""

import sys
from pathlib import Path
import hashlib
import numpy as np
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignjuice.core.data_container import DataContainer, AlignmentSample
from alignjuice.operators.dedup.exact import ExactDedup
from alignjuice.operators.filter.quality import QualityFilter
from alignjuice.stages.s4_sandbox_eval import SandboxEvalStage


def simple_tokenize(text: str) -> list[str]:
    """Simple Chinese tokenizer using character-level + common patterns."""
    # For Chinese, character-level works reasonably well
    tokens = []
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # Chinese character
            tokens.append(char)
        elif char.isalnum():
            tokens.append(char.lower())
    return tokens


def compute_tfidf_vectors(texts: list[str]) -> np.ndarray:
    """Compute simple TF-IDF vectors for texts."""
    # Tokenize all texts
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

    # Filter vocabulary (keep tokens appearing in 2+ docs but < 80% of docs)
    n_docs = len(texts)
    filtered_vocab = {
        token: idx for token, idx in vocab.items()
        if 2 <= doc_freq[token] < 0.8 * n_docs
    }

    # Reindex
    vocab = {token: i for i, token in enumerate(filtered_vocab.keys())}
    vocab_size = len(vocab)

    if vocab_size == 0:
        # Fallback: use all tokens
        vocab = {token: i for i, token in enumerate(doc_freq.keys())}
        vocab_size = len(vocab)

    print(f"    词汇表大小: {vocab_size}")

    # Compute TF-IDF
    vectors = np.zeros((len(texts), vocab_size))
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

    return vectors


def semantic_dedup_tfidf(data: DataContainer, threshold: float = 0.85) -> DataContainer:
    """Semantic deduplication using TF-IDF similarity."""
    print(f"    计算 TF-IDF 向量...")
    texts = [s.instruction for s in data]
    vectors = compute_tfidf_vectors(texts)

    print(f"    查找语义重复...")
    n = len(vectors)
    keep_mask = np.ones(n, dtype=bool)

    # Greedy deduplication
    for i in range(n):
        if not keep_mask[i]:
            continue
        # Compute similarity with remaining samples
        if i + 1 < n:
            similarities = np.dot(vectors[i], vectors[i + 1:].T)
            duplicate_indices = np.where(similarities >= threshold)[0] + i + 1
            keep_mask[duplicate_indices] = False

    kept_indices = np.where(keep_mask)[0]
    kept_samples = [data[i] for i in kept_indices]
    removed_count = n - len(kept_samples)

    return DataContainer(
        samples=kept_samples,
        provenance=data.provenance + [f"semantic_dedup_tfidf: {n} -> {len(kept_samples)}"],
    ), removed_count


def knowledge_score_tfidf(data: DataContainer) -> DataContainer:
    """Score samples by knowledge density using simple heuristics."""
    scored_samples = []

    for sample in data:
        # Simple knowledge scoring based on:
        # 1. Output length
        # 2. Unique character ratio
        # 3. Contains numbers/facts

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
        sample.metadata['knowledge_score'] = knowledge_score
        scored_samples.append((knowledge_score, sample))

    # Sort by score descending
    scored_samples.sort(key=lambda x: x[0], reverse=True)

    return scored_samples


def main():
    print("=" * 60)
    print("AlignJuice 实战测试 - 百科数据处理 (离线版)")
    print("=" * 60)

    # 1. 加载数据
    input_path = Path(__file__).parent.parent / "data" / "baike_sample_3000.jsonl"
    print(f"\n[Stage 1] 加载数据")
    print(f"  路径: {input_path}")
    data = DataContainer.from_jsonl(input_path)
    initial_count = len(data)
    print(f"  加载完成: {initial_count} 条")

    # 2. 精确去重
    print(f"\n[Stage 2] 精确去重")
    exact_dedup = ExactDedup(field="instruction")
    data = exact_dedup(data)
    exact_removed = exact_dedup.metrics['removed_count']
    print(f"  结果: {len(data)} 条 (去除 {exact_removed} 条精确重复)")

    # 3. 语义去重 (TF-IDF)
    print(f"\n[Stage 3] 语义去重 (TF-IDF, threshold=0.85)")
    data, semantic_removed = semantic_dedup_tfidf(data, threshold=0.85)
    print(f"  结果: {len(data)} 条 (去除 {semantic_removed} 条语义重复)")
    print(f"  去重率: {semantic_removed / (initial_count - exact_removed):.1%}")

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

    # 5. 知识密度排序 + 选择 Top-K
    print(f"\n[Stage 5] 知识密度筛选 (top_k=1000)")
    scored_samples = knowledge_score_tfidf(data)

    target_count = min(1000, len(scored_samples))
    top_samples = [s for _, s in scored_samples[:target_count]]
    avg_knowledge = sum(s.metadata['knowledge_score'] for s in top_samples) / len(top_samples)

    data = DataContainer(
        samples=top_samples,
        provenance=data.provenance + [f"knowledge_filter: top {target_count}"],
    )
    print(f"  结果: {len(data)} 条")
    print(f"  平均知识分: {avg_knowledge:.2f}")

    # 6. 评估报告
    print(f"\n[Stage 6] 生成评估报告")
    eval_stage = SandboxEvalStage(
        report_path="reports/baike_offline_report.html",
        metrics_path="reports/baike_offline_metrics.json",
    )
    data = eval_stage.process(data)
    print(f"  报告已生成")

    # 7. 保存结果
    output_path = Path(__file__).parent.parent / "output" / "baike_final_offline.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_jsonl(output_path)

    # 最终统计
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)

    print(f"""
处理流程:
  输入:     {initial_count} 条
  精确去重: → {initial_count - exact_removed} 条 (-{exact_removed})
  语义去重: → {initial_count - exact_removed - semantic_removed} 条 (-{semantic_removed})
  质量过滤: → {quality_filter.metrics['output_count']} 条
  知识筛选: → {len(data)} 条

输出文件:
  - output/baike_final_offline.jsonl
  - reports/baike_offline_report.html
  - reports/baike_offline_metrics.json
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
    print(f"\n最终数据样例 (5条):")
    for i, sample in enumerate(list(data)[:5]):
        ks = sample.metadata.get('knowledge_score', 0)
        qs = sample.metadata.get('quality_score', 0)
        print(f"\n  [{i+1}] {sample.category} | 知识分:{ks:.2f} | 质量分:{qs:.2f}")
        print(f"      Q: {sample.instruction}")
        print(f"      A: {sample.output[:80]}...")


if __name__ == "__main__":
    main()
