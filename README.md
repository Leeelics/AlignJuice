# AlignJuice

**高质量对齐数据管理框架** - 将粗筛数据转化为高质量 LLM 对齐数据

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心理念

> **"Less is More"** — 1000 条高质量数据 > 100,000 条低质量数据

AlignJuice 融合 Data-Juicer、LIMA、LIMO、phi-3 等前沿研究，提供自动化的数据质量提升流水线。

## 四阶段流水线

```
3000 条粗筛数据
       │
       ▼
┌──────────────────┐
│ Stage 1          │  去重 + 质量过滤
│ Data-Juicer      │  → 1500 条
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Stage 2          │  知识密度筛选 + LLM 合成
│ Knowledge Filter │  → 1000 条
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Stage 3          │  推理链增强 + 噪声清洗
│ Reasoning Enhance│  → 1000 条 (质量↑)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Stage 4          │  质量评估 + 报告生成
│ Sandbox Eval     │
└──────────────────┘
       │
       ▼
1000 条高质量对齐数据 + 质量报告
```

## 快速开始

### 安装

```bash
pip install -e .
```

### 命令行使用

```bash
# 运行完整流水线
alignjuice run --config configs/default.yaml --input data/raw.jsonl

# 评估数据质量
alignjuice evaluate --input output/final.jsonl --report reports/
```

### Python API

```python
from alignjuice import AlignJuice

# 初始化
aj = AlignJuice(config="configs/default.yaml")

# 加载并处理
data = aj.load("data/raw.jsonl")
result = aj.run_pipeline(data)

# 查看报告
result.report()
```

### Jupyter 交互式

```python
from alignjuice import DataContainer
from alignjuice.operators import SemanticDedup, QualityFilter

# 加载数据
data = DataContainer.from_jsonl("data/raw.jsonl")
data.describe()  # 统计概览

# 逐步处理
deduped = SemanticDedup(threshold=0.95)(data)
filtered = QualityFilter(threshold=0.8)(deduped)

# 查看差异
data.diff(deduped).show()
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **语义去重** | 基于嵌入的相似度去重 (阈值 0.95) |
| **知识密度评估** | 向量嵌入替代知识图谱 |
| **LLM 合成** | phi-3 风格教科书级增强 |
| **推理链增强** | LIMO 风格思维链生成 |
| **噪声检测** | 启发式 + CleanLab 集成 |
| **多后端支持** | Ollama 本地 / OpenAI API |
| **检查点** | 每阶段保存，支持断点续传 |

## 项目结构

```
alignjuice/
├── core/           # 数据容器、流水线、注册表
├── operators/      # 去重、过滤、转换、验证算子
├── stages/         # 四阶段流水线实现
├── integrations/   # LLM、嵌入后端集成
└── io/             # 数据读写
```

## 配置示例

```yaml
# configs/default.yaml
name: alignjuice_default
stages:
  - name: s1_data_juicer
    operators:
      - name: semantic_dedup
        params: { threshold: 0.95 }
      - name: quality_filter
        params: { threshold: 0.8 }

llm:
  backend: ollama
  model: phi3:medium
  fallback_backend: openai
```

## 文档

- [项目详细说明](docs/PROJECT_OVERVIEW.md) - 完整的设计思想和架构说明
- [notebooks/](notebooks/) - Jupyter 交互式示例

## License

MIT License
