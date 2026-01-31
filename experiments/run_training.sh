#!/bin/bash
# AlignJuice 效果验证实验 - 训练脚本
# 使用 LLaMA-Factory 进行 LoRA 微调

set -e

# ============== 配置 ==============

# LLaMA-Factory 路径 (需要修改为实际路径)
LLAMA_FACTORY_PATH="${LLAMA_FACTORY_PATH:-$HOME/LLaMA-Factory}"

# 实验目录
EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$EXPERIMENT_DIR/data"
CONFIG_DIR="$EXPERIMENT_DIR/configs"
OUTPUT_DIR="$EXPERIMENT_DIR/outputs"

# 基座模型 (Qwen2-1.5B 在 V100-32GB 上运行良好)
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-1.5B}"

# 训练参数 (针对 V100-32GB 优化)
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"      # V100-32GB 可以用更大 batch
GRAD_ACCUM="${GRAD_ACCUM:-2}"      # 有效 batch = 8*2 = 16
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-64}"

# GPU 配置
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============== 函数 ==============

print_header() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

train_model() {
    local dataset_name=$1
    local output_name=$2

    echo ""
    print_header "训练模型: $output_name"
    echo "数据集: $dataset_name"
    echo "输出目录: $OUTPUT_DIR/$output_name"
    echo ""

    # 创建临时配置文件
    local config_file="$CONFIG_DIR/${output_name}.yaml"

    cat > "$config_file" << EOF
### 自动生成的训练配置: $output_name

### 模型配置
model_name_or_path: $BASE_MODEL
trust_remote_code: true

### 训练方式
stage: sft
do_train: true
finetuning_type: lora

### 数据集配置
dataset_dir: $DATA_DIR
dataset: $dataset_name
template: qwen
cutoff_len: 1024
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### LoRA 配置
lora_target: all
lora_rank: $LORA_RANK
lora_alpha: $((LORA_RANK * 2))
lora_dropout: 0.1

### 训练超参数
output_dir: $OUTPUT_DIR/$output_name
logging_steps: 10
save_steps: 500
save_total_limit: 2
per_device_train_batch_size: $BATCH_SIZE
gradient_accumulation_steps: $GRAD_ACCUM
learning_rate: $LEARNING_RATE
num_train_epochs: $NUM_EPOCHS
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
ddp_timeout: 180000000

### 其他
plot_loss: true
report_to: none
EOF

    echo "配置文件: $config_file"

    # 复制 dataset_info.json 到数据目录
    cp "$CONFIG_DIR/dataset_info.json" "$DATA_DIR/dataset_info.json"

    # 运行训练
    cd "$LLAMA_FACTORY_PATH"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES llamafactory-cli train "$config_file"

    echo ""
    echo "训练完成: $output_name"
    echo "模型保存在: $OUTPUT_DIR/$output_name"
}

# ============== 主逻辑 ==============

print_header "AlignJuice 效果验证实验 - 模型训练"

echo "配置信息:"
echo "  LLaMA-Factory: $LLAMA_FACTORY_PATH"
echo "  基座模型: $BASE_MODEL"
echo "  训练轮数: $NUM_EPOCHS"
echo "  批次大小: $BATCH_SIZE x $GRAD_ACCUM"
echo "  学习率: $LEARNING_RATE"
echo "  LoRA Rank: $LORA_RANK"
echo "  GPU: $CUDA_VISIBLE_DEVICES"

# 检查 LLaMA-Factory
if [ ! -d "$LLAMA_FACTORY_PATH" ]; then
    echo ""
    echo "错误: LLaMA-Factory 未找到: $LLAMA_FACTORY_PATH"
    echo "请设置 LLAMA_FACTORY_PATH 环境变量或安装 LLaMA-Factory:"
    echo "  git clone https://github.com/hiyouga/LLaMA-Factory.git"
    echo "  cd LLaMA-Factory && pip install -e ."
    exit 1
fi

# 检查数据
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR/*.json 2>/dev/null)" ]; then
    echo ""
    echo "错误: 训练数据未找到"
    echo "请先运行数据准备脚本:"
    echo "  python experiments/prepare_data.py"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 解析命令行参数
case "${1:-all}" in
    "baseline_1k")
        train_model "baseline_1k" "baseline_1k"
        ;;
    "baseline_3k")
        train_model "baseline_3k" "baseline_3k"
        ;;
    "baseline_10k")
        train_model "baseline_10k" "baseline_10k"
        ;;
    "alignjuice_1k")
        train_model "alignjuice_1k" "alignjuice_1k"
        ;;
    "alignjuice_3k")
        train_model "alignjuice_3k" "alignjuice_3k"
        ;;
    "baseline")
        train_model "baseline_1k" "baseline_1k"
        train_model "baseline_3k" "baseline_3k"
        train_model "baseline_10k" "baseline_10k"
        ;;
    "alignjuice")
        train_model "alignjuice_1k" "alignjuice_1k"
        train_model "alignjuice_3k" "alignjuice_3k"
        ;;
    "all")
        train_model "baseline_1k" "baseline_1k"
        train_model "baseline_3k" "baseline_3k"
        train_model "baseline_10k" "baseline_10k"
        train_model "alignjuice_1k" "alignjuice_1k"
        train_model "alignjuice_3k" "alignjuice_3k"
        ;;
    *)
        echo "用法: $0 [baseline_1k|baseline_3k|baseline_10k|alignjuice_1k|alignjuice_3k|baseline|alignjuice|all]"
        echo ""
        echo "选项:"
        echo "  baseline_1k    - 训练 baseline 1K 模型"
        echo "  baseline_3k    - 训练 baseline 3K 模型"
        echo "  baseline_10k   - 训练 baseline 10K 模型"
        echo "  alignjuice_1k  - 训练 AlignJuice 1K 模型"
        echo "  alignjuice_3k  - 训练 AlignJuice 3K 模型"
        echo "  baseline       - 训练所有 baseline 模型"
        echo "  alignjuice     - 训练所有 AlignJuice 模型"
        echo "  all            - 训练所有模型 (默认)"
        exit 1
        ;;
esac

print_header "所有训练任务完成!"
echo "模型输出目录: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
