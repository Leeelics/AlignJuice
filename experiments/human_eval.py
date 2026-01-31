#!/usr/bin/env python3
"""
人工评测工具 - 盲评模型生成质量

功能:
1. 随机抽取测试问题
2. 用各模型生成回答
3. 打乱顺序进行盲评
4. 统计评分结果
"""

import json
import random
import os
from pathlib import Path
from datetime import datetime

# ============== 配置 ==============

EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# 评测问题数量
NUM_QUESTIONS = 50

# 模型列表
MODELS = [
    "baseline_1k",
    "baseline_3k",
    "baseline_10k",
    "alignjuice_1k",
    "alignjuice_3k",
]

# 评分维度
SCORING_DIMENSIONS = {
    "accuracy": "准确性 (事实是否正确)",
    "completeness": "完整性 (回答是否全面)",
    "fluency": "流畅性 (语言是否自然)",
}


# ============== 工具函数 ==============

def load_test_questions(num: int = NUM_QUESTIONS) -> list[dict]:
    """加载测试问题"""
    # 从 alignjuice 数据中抽取 (这些是高质量问题)
    data_file = DATA_DIR / "alignjuice_1k.json"

    if not data_file.exists():
        print(f"错误: 数据文件不存在: {data_file}")
        print("请先运行 prepare_data.py")
        return []

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 随机抽取
    random.seed(42)
    questions = random.sample(data, min(num, len(data)))

    return [{"instruction": q["instruction"], "input": q.get("input", "")} for q in questions]


def generate_responses_placeholder(questions: list[dict]) -> dict:
    """
    生成模型回答的占位函数

    实际使用时需要:
    1. 加载各个微调后的模型
    2. 对每个问题生成回答
    3. 返回 {model_name: [responses]}
    """
    print("\n" + "=" * 50)
    print("注意: 这是占位函数")
    print("实际使用时需要加载模型并生成回答")
    print("=" * 50)

    # 返回示例结构
    responses = {}
    for model in MODELS:
        responses[model] = [
            f"[{model} 的回答占位符 - 问题 {i+1}]"
            for i in range(len(questions))
        ]

    return responses


def generate_responses_with_llamafactory(questions: list[dict]) -> dict:
    """
    使用 LLaMA-Factory 生成回答

    需要安装: pip install llamafactory
    """
    try:
        from llamafactory.chat import ChatModel
    except ImportError:
        print("警告: 未安装 llamafactory，使用占位符")
        return generate_responses_placeholder(questions)

    responses = {}
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2-1.5B")

    for model_name in MODELS:
        print(f"\n生成回答: {model_name}")
        adapter_path = OUTPUT_DIR / model_name

        if not adapter_path.exists():
            print(f"  警告: 模型不存在: {adapter_path}")
            responses[model_name] = ["[模型不存在]"] * len(questions)
            continue

        try:
            # 加载模型
            chat_model = ChatModel(dict(
                model_name_or_path=base_model,
                adapter_name_or_path=str(adapter_path),
                template="qwen",
                finetuning_type="lora",
            ))

            # 生成回答
            model_responses = []
            for i, q in enumerate(questions):
                messages = [{"role": "user", "content": q["instruction"]}]
                response = chat_model.chat(messages)[0].response_text
                model_responses.append(response)

                if (i + 1) % 10 == 0:
                    print(f"  进度: {i+1}/{len(questions)}")

            responses[model_name] = model_responses

        except Exception as e:
            print(f"  错误: {e}")
            responses[model_name] = [f"[生成失败: {e}]"] * len(questions)

    return responses


def create_blind_eval_sheet(questions: list[dict], responses: dict) -> list[dict]:
    """创建盲评表格"""
    eval_sheet = []

    for i, question in enumerate(questions):
        # 收集所有模型的回答
        model_responses = []
        for model in MODELS:
            model_responses.append({
                "model": model,
                "response": responses.get(model, [""])[i] if i < len(responses.get(model, [])) else "",
            })

        # 打乱顺序
        random.shuffle(model_responses)

        # 创建评测项
        eval_item = {
            "question_id": i + 1,
            "instruction": question["instruction"],
            "responses": [
                {
                    "response_id": chr(65 + j),  # A, B, C, D, E
                    "content": r["response"],
                    "_model": r["model"],  # 隐藏字段，评测时不显示
                    "scores": {dim: None for dim in SCORING_DIMENSIONS},
                }
                for j, r in enumerate(model_responses)
            ],
        }
        eval_sheet.append(eval_item)

    return eval_sheet


def print_eval_interface(eval_sheet: list[dict], start_idx: int = 0):
    """打印评测界面"""
    for item in eval_sheet[start_idx:start_idx + 5]:
        print("\n" + "=" * 60)
        print(f"问题 {item['question_id']}: {item['instruction']}")
        print("=" * 60)

        for resp in item["responses"]:
            print(f"\n[{resp['response_id']}] {resp['content'][:200]}...")
            print("-" * 40)

        print("\n请为每个回答打分 (1-5分):")
        for dim, desc in SCORING_DIMENSIONS.items():
            print(f"  {dim}: {desc}")


def calculate_scores(eval_sheet: list[dict]) -> dict:
    """计算各模型的平均分"""
    model_scores = {model: [] for model in MODELS}

    for item in eval_sheet:
        for resp in item["responses"]:
            model = resp["_model"]
            scores = resp["scores"]

            # 计算该回答的平均分
            valid_scores = [s for s in scores.values() if s is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                model_scores[model].append(avg_score)

    # 计算各模型的总平均分
    final_scores = {}
    for model, scores in model_scores.items():
        if scores:
            final_scores[model] = sum(scores) / len(scores)
        else:
            final_scores[model] = None

    return final_scores


def save_eval_results(eval_sheet: list[dict], scores: dict):
    """保存评测结果"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 保存详细评测表
    eval_file = RESULTS_DIR / "human_eval_detail.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_sheet, f, ensure_ascii=False, indent=2)
    print(f"详细评测已保存: {eval_file}")

    # 保存汇总分数
    scores_file = RESULTS_DIR / "human_eval.json"
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"汇总分数已保存: {scores_file}")


# ============== 主函数 ==============

def main():
    print("=" * 60)
    print("AlignJuice 效果验证实验 - 人工评测工具")
    print("=" * 60)

    # 1. 加载测试问题
    print(f"\n[1] 加载测试问题 ({NUM_QUESTIONS} 条)")
    questions = load_test_questions(NUM_QUESTIONS)
    if not questions:
        return
    print(f"  已加载 {len(questions)} 个问题")

    # 2. 生成模型回答
    print(f"\n[2] 生成模型回答")
    use_real_models = input("是否使用真实模型生成回答? (y/n, 默认 n): ").strip().lower() == "y"

    if use_real_models:
        responses = generate_responses_with_llamafactory(questions)
    else:
        responses = generate_responses_placeholder(questions)

    # 3. 创建盲评表格
    print(f"\n[3] 创建盲评表格")
    eval_sheet = create_blind_eval_sheet(questions, responses)
    print(f"  已创建 {len(eval_sheet)} 个评测项")

    # 4. 保存评测表格 (供人工评测使用)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_template_file = RESULTS_DIR / "human_eval_template.json"
    with open(eval_template_file, "w", encoding="utf-8") as f:
        # 移除隐藏的模型信息，供盲评使用
        blind_sheet = []
        for item in eval_sheet:
            blind_item = {
                "question_id": item["question_id"],
                "instruction": item["instruction"],
                "responses": [
                    {
                        "response_id": r["response_id"],
                        "content": r["content"],
                        "scores": r["scores"],
                    }
                    for r in item["responses"]
                ],
            }
            blind_sheet.append(blind_item)
        json.dump(blind_sheet, f, ensure_ascii=False, indent=2)
    print(f"  盲评模板已保存: {eval_template_file}")

    # 5. 显示评测界面示例
    print(f"\n[4] 评测界面示例")
    print_eval_interface(eval_sheet, 0)

    # 6. 说明
    print("\n" + "=" * 60)
    print("人工评测说明")
    print("=" * 60)
    print(f"""
1. 打开评测模板文件: {eval_template_file}

2. 对每个问题的每个回答进行打分 (1-5分):
   - 1分: 很差
   - 2分: 较差
   - 3分: 一般
   - 4分: 较好
   - 5分: 很好

3. 评分维度:
   - accuracy: 准确性 (事实是否正确)
   - completeness: 完整性 (回答是否全面)
   - fluency: 流畅性 (语言是否自然)

4. 完成评测后，将文件保存为: {RESULTS_DIR / 'human_eval_completed.json'}

5. 运行以下命令计算分数:
   python experiments/human_eval.py --calculate
""")

    # 检查是否需要计算分数
    import sys
    if "--calculate" in sys.argv:
        completed_file = RESULTS_DIR / "human_eval_completed.json"
        if completed_file.exists():
            print("\n[计算评测分数]")
            with open(completed_file, "r", encoding="utf-8") as f:
                completed_sheet = json.load(f)

            # 恢复模型信息
            for i, item in enumerate(completed_sheet):
                for j, resp in enumerate(item["responses"]):
                    resp["_model"] = eval_sheet[i]["responses"][j]["_model"]

            scores = calculate_scores(completed_sheet)
            save_eval_results(completed_sheet, scores)

            print("\n评测结果:")
            for model, score in sorted(scores.items(), key=lambda x: x[1] or 0, reverse=True):
                score_str = f"{score:.2f}" if score else "N/A"
                print(f"  {model}: {score_str}")
        else:
            print(f"\n错误: 未找到完成的评测文件: {completed_file}")


if __name__ == "__main__":
    main()
