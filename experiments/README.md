# AlignJuice 效果验证实验

## 实验目标

验证核心假设：**AlignJuice 处理后的 1K 数据 ≈ 原始 10K 数据的微调效果**

## 实验设计

### 对比组

| 组别 | 数据量 | 处理方式 | 说明 |
|------|--------|---------|------|
| baseline_1k | 1,000 | 随机采样 | 基线对照 |
| baseline_3k | 3,000 | 随机采样 | 基线对照 |
| baseline_10k | 10,000 | 随机采样 | 基线对照 |
| alignjuice_1k | 1,000 | AlignJuice 处理 | 实验组 |
| alignjuice_3k | 3,000 | AlignJuice 处理 | 实验组 |

### 评测指标

- **C-Eval**: 中文综合能力评测
- **CMMLU**: 中文多任务理解
- **人工评测**: 50 条问答盲评

## 目录结构

```
experiments/
├── README.md              # 本文件
├── prepare_data.py        # 数据准备脚本
├── run_training.sh        # 训练启动脚本
├── run_eval.sh            # 评测脚本
├── analyze_results.py     # 结果分析脚本
├── human_eval.py          # 人工评测工具
├── configs/               # LLaMA-Factory 配置
│   ├── dataset_info.json  # 数据集注册
│   ├── baseline_1k.yaml   # 训练配置
│   ├── baseline_3k.yaml
│   ├── baseline_10k.yaml
│   ├── alignjuice_1k.yaml
│   └── alignjuice_3k.yaml
├── data/                  # 实验数据
│   ├── baseline_1k.json
│   ├── baseline_3k.json
│   ├── baseline_10k.json
│   ├── alignjuice_1k.json
│   └── alignjuice_3k.json
├── outputs/               # 模型输出
│   ├── baseline_1k/
│   ├── baseline_3k/
│   ├── baseline_10k/
│   ├── alignjuice_1k/
│   └── alignjuice_3k/
└── results/               # 评测结果
    ├── eval_scores.json
    ├── human_eval.json
    └── final_report.md
```

## 快速开始

### 1. 环境准备

```bash
# 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 安装评测工具
pip install lm-eval opencompass
```

### 2. 数据准备

```bash
cd /path/to/AlignJuice
python experiments/prepare_data.py
```

### 3. 模型训练

```bash
# 训练所有变体 (可并行)
bash experiments/run_training.sh all

# 或单独训练
bash experiments/run_training.sh baseline_1k
bash experiments/run_training.sh alignjuice_1k
```

### 4. 模型评测

```bash
bash experiments/run_eval.sh all
```

### 5. 结果分析

```bash
python experiments/analyze_results.py
```

## 预期结果

```
模型              C-Eval    CMMLU    人工评分
─────────────────────────────────────────────
baseline_1k       45.2      43.1     3.2
baseline_3k       48.5      46.3     3.5
baseline_10k      51.2      49.8     3.8
alignjuice_1k     50.8      48.5     3.9  ← 目标: 接近 baseline_10k
alignjuice_3k     53.1      51.2     4.1  ← 目标: 超过 baseline_10k
```

## 成功标准

- [ ] alignjuice_1k 的 C-Eval 分数 ≥ baseline_10k 的 95%
- [ ] alignjuice_1k 的人工评分 ≥ baseline_10k
- [ ] 数据效率提升 ≥ 10x

## 注意事项

1. 确保所有训练使用相同的超参数
2. 评测时使用相同的 prompt 模板
3. 人工评测采用盲评方式
4. 记录所有随机种子以保证可复现
