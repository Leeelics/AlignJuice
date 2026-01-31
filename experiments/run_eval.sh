#!/bin/bash
# AlignJuice 效果验证实验 - 评测脚本
# 使用 lm-evaluation-harness 进行自动评测

set -e

# ============== 配置 ==============

# 实验目录
EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$EXPERIMENT_DIR/outputs"
RESULTS_DIR="$EXPERIMENT_DIR/results"

# 基座模型
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-1.5B}"

# 评测任务
EVAL_TASKS="${EVAL_TASKS:-ceval,cmmlu}"

# GPU 配置
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 批次大小
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

# ============== 函数 ==============

print_header() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

eval_model() {
    local model_name=$1
    local model_path="$OUTPUT_DIR/$model_name"

    echo ""
    print_header "评测模型: $model_name"

    # 检查模型是否存在
    if [ ! -d "$model_path" ]; then
        echo "警告: 模型不存在: $model_path"
        echo "跳过评测"
        return
    fi

    local result_file="$RESULTS_DIR/${model_name}_eval.json"

    echo "模型路径: $model_path"
    echo "评测任务: $EVAL_TASKS"
    echo "结果文件: $result_file"
    echo ""

    # 使用 LLaMA-Factory 的评测功能
    # 或者使用 lm-evaluation-harness
    if command -v lm_eval &> /dev/null; then
        echo "使用 lm-evaluation-harness 评测..."

        # 合并 LoRA 权重后评测
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES lm_eval \
            --model hf \
            --model_args pretrained=$BASE_MODEL,peft=$model_path,trust_remote_code=True \
            --tasks $EVAL_TASKS \
            --batch_size $EVAL_BATCH_SIZE \
            --output_path "$RESULTS_DIR/${model_name}" \
            --log_samples

    elif command -v llamafactory-cli &> /dev/null; then
        echo "使用 LLaMA-Factory 评测..."

        # 创建评测配置
        local eval_config="$RESULTS_DIR/${model_name}_eval_config.yaml"
        cat > "$eval_config" << EOF
### 评测配置: $model_name
model_name_or_path: $BASE_MODEL
adapter_name_or_path: $model_path
trust_remote_code: true
template: qwen
task: ceval
split: test
lang: zh
n_shot: 5
batch_size: $EVAL_BATCH_SIZE
EOF

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES llamafactory-cli eval "$eval_config"

    else
        echo "错误: 未找到评测工具"
        echo "请安装 lm-evaluation-harness 或 LLaMA-Factory:"
        echo "  pip install lm-eval"
        echo "  # 或"
        echo "  pip install llamafactory"
        return 1
    fi

    echo ""
    echo "评测完成: $model_name"
}

eval_base_model() {
    echo ""
    print_header "评测基座模型: $BASE_MODEL"

    local result_file="$RESULTS_DIR/base_model_eval.json"

    if command -v lm_eval &> /dev/null; then
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES lm_eval \
            --model hf \
            --model_args pretrained=$BASE_MODEL,trust_remote_code=True \
            --tasks $EVAL_TASKS \
            --batch_size $EVAL_BATCH_SIZE \
            --output_path "$RESULTS_DIR/base_model" \
            --log_samples
    fi

    echo "基座模型评测完成"
}

# ============== 主逻辑 ==============

print_header "AlignJuice 效果验证实验 - 模型评测"

echo "配置信息:"
echo "  基座模型: $BASE_MODEL"
echo "  评测任务: $EVAL_TASKS"
echo "  批次大小: $EVAL_BATCH_SIZE"
echo "  GPU: $CUDA_VISIBLE_DEVICES"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 解析命令行参数
case "${1:-all}" in
    "base")
        eval_base_model
        ;;
    "baseline_1k")
        eval_model "baseline_1k"
        ;;
    "baseline_3k")
        eval_model "baseline_3k"
        ;;
    "baseline_10k")
        eval_model "baseline_10k"
        ;;
    "alignjuice_1k")
        eval_model "alignjuice_1k"
        ;;
    "alignjuice_3k")
        eval_model "alignjuice_3k"
        ;;
    "baseline")
        eval_model "baseline_1k"
        eval_model "baseline_3k"
        eval_model "baseline_10k"
        ;;
    "alignjuice")
        eval_model "alignjuice_1k"
        eval_model "alignjuice_3k"
        ;;
    "all")
        eval_base_model
        eval_model "baseline_1k"
        eval_model "baseline_3k"
        eval_model "baseline_10k"
        eval_model "alignjuice_1k"
        eval_model "alignjuice_3k"
        ;;
    *)
        echo "用法: $0 [base|baseline_1k|baseline_3k|baseline_10k|alignjuice_1k|alignjuice_3k|baseline|alignjuice|all]"
        echo ""
        echo "选项:"
        echo "  base           - 评测基座模型"
        echo "  baseline_1k    - 评测 baseline 1K 模型"
        echo "  baseline_3k    - 评测 baseline 3K 模型"
        echo "  baseline_10k   - 评测 baseline 10K 模型"
        echo "  alignjuice_1k  - 评测 AlignJuice 1K 模型"
        echo "  alignjuice_3k  - 评测 AlignJuice 3K 模型"
        echo "  baseline       - 评测所有 baseline 模型"
        echo "  alignjuice     - 评测所有 AlignJuice 模型"
        echo "  all            - 评测所有模型 (默认)"
        exit 1
        ;;
esac

print_header "评测任务完成!"
echo "结果目录: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
