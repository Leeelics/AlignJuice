# AutoDL 快速启动指南

## 环境配置

### 1. 租用实例
- **GPU**: V100-32GB
- **镜像**: PyTorch 2.0 + Python 3.10 (或更高版本)

### 2. 上传代码

```bash
# 方式1: 从 GitHub 克隆 (如果已推送)
cd /root/autodl-tmp
git clone https://github.com/your-username/AlignJuice.git
cd AlignJuice

# 方式2: 使用 AutoDL 文件上传功能
# 将本地 AlignJuice 目录打包上传
```

### 3. 上传数据

将百科数据集上传到 AutoDL:
```bash
# 在 AutoDL 实例上创建目录
mkdir -p ~/Downloads/data-baike

# 使用 AutoDL 文件传输上传 train.jsonl
# 或使用 scp/rsync
```

### 4. 安装依赖

```bash
cd /root/autodl-tmp/AlignJuice

# 安装 AlignJuice
pip install -e .

# 安装 LLaMA-Factory
cd /root/autodl-tmp
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 安装评测工具
pip install lm-eval
```

### 5. 设置环境变量

```bash
# 添加到 ~/.bashrc
export LLAMA_FACTORY_PATH=/root/autodl-tmp/LLaMA-Factory
export BASE_MODEL=Qwen/Qwen2-1.5B
export CUDA_VISIBLE_DEVICES=0

# 使生效
source ~/.bashrc
```

---

## 运行实验

### Step 1: 准备数据 (~5分钟)

```bash
cd /root/autodl-tmp/AlignJuice
python experiments/prepare_data.py
```

预期输出:
```
[1] 加载原始数据: 274148 条
[2] 生成 Baseline 数据
[3] 生成 AlignJuice 数据
...
数据准备完成!
```

### Step 2: 训练模型 (~2小时)

```bash
# 训练所有 5 个模型
bash experiments/run_training.sh all

# 或分开训练 (可以先跑一个测试)
bash experiments/run_training.sh baseline_1k      # ~10分钟
bash experiments/run_training.sh alignjuice_1k    # ~10分钟
bash experiments/run_training.sh baseline_3k      # ~25分钟
bash experiments/run_training.sh alignjuice_3k    # ~25分钟
bash experiments/run_training.sh baseline_10k     # ~1小时
```

### Step 3: 评测模型 (~1小时)

```bash
# 评测所有模型
bash experiments/run_eval.sh all

# 或分开评测
bash experiments/run_eval.sh baseline_1k
bash experiments/run_eval.sh alignjuice_1k
# ...
```

### Step 4: 分析结果

```bash
python experiments/analyze_results.py
```

查看报告:
```bash
cat experiments/results/final_report.md
```

---

## 预计时间和成本

| 阶段 | 时间 | 说明 |
|------|------|------|
| 环境配置 | 30 分钟 | 安装依赖 |
| 数据准备 | 5 分钟 | 生成 5 组数据 |
| 模型训练 | 2 小时 | 5 个模型 |
| 模型评测 | 1 小时 | C-Eval + CMMLU |
| 结果分析 | 5 分钟 | 生成报告 |
| **总计** | **~4 小时** | |

**成本估算**: V100-32GB 约 2-3 元/小时 → 总计约 **10-15 元**

---

## 常见问题

### Q: 显存不足 (OOM)
```bash
# 减小 batch_size
export BATCH_SIZE=4
export GRAD_ACCUM=4
bash experiments/run_training.sh all
```

### Q: 下载模型慢
```bash
# 使用 ModelScope 镜像
export HF_ENDPOINT=https://hf-mirror.com
# 或使用 modelscope 下载
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2-1.5B')"
```

### Q: 训练中断如何恢复
```bash
# LLaMA-Factory 支持断点续训
# 修改配置中的 resume_from_checkpoint 参数
```

---

## 快速验证脚本

一键运行完整实验:

```bash
#!/bin/bash
# quick_run.sh

set -e

echo "=== Step 1: 数据准备 ==="
python experiments/prepare_data.py

echo "=== Step 2: 训练模型 ==="
bash experiments/run_training.sh all

echo "=== Step 3: 评测模型 ==="
bash experiments/run_eval.sh all

echo "=== Step 4: 分析结果 ==="
python experiments/analyze_results.py

echo "=== 完成! ==="
cat experiments/results/final_report.md
```

保存为 `quick_run.sh` 后运行:
```bash
chmod +x quick_run.sh
nohup ./quick_run.sh > experiment.log 2>&1 &
tail -f experiment.log
```
