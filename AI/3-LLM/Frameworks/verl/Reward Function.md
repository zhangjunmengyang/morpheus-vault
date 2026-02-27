---
brief: "verl Reward Function 设计——自定义奖励函数的接口规范和工程实践；rule-based reward（可验证任务）vs LLM-as-judge（开放任务）的接入方式；奖励归一化/clip 的工程细节。"
title: "Reward Function"
type: concept
domain: ai/llm/frameworks/verl
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/frameworks/verl
  - type/concept
---
# Reward Function

> verl 中 reward 的设计和实现。Reward 是 RL 训练的灵魂 — reward 设计好坏直接决定模型训练效果。

## verl 的 Reward 接口

verl 支持多种 reward 来源：

```python
# 1. 基于规则的 reward function（最常用、最可控）
def rule_based_reward(prompt, response, ground_truth):
    """适合有明确标准答案的任务"""
    # 精确匹配
    if extract_answer(response) == ground_truth:
        return 1.0
    return 0.0

# 2. Reward Model（神经网络打分）
class RewardModel:
    def __call__(self, prompt, response):
        inputs = tokenizer(prompt + response, return_tensors="pt")
        score = self.model(**inputs).logits
        return score.item()

# 3. LLM-as-Judge（用另一个 LLM 打分）
def llm_judge_reward(prompt, response):
    judge_prompt = f"""
    请评估以下回答的质量（1-10分）：
    问题: {prompt}
    回答: {response}
    """
    score = judge_model.generate(judge_prompt)
    return parse_score(score) / 10.0

# 4. 代码执行验证（数学/编程任务）
def code_execution_reward(prompt, response):
    code = extract_code(response)
    test_results = sandbox.execute(code, test_cases)
    return sum(test_results) / len(test_results)
```

## 在 verl 中注册自定义 Reward

```python
# verl 的 reward function 通过配置注册
# 在 config.yaml 中指定

# reward:
#   type: "custom"
#   path: "my_rewards.math_reward"

# my_rewards.py
import re

def math_reward(data_item):
    """
    data_item 包含：
    - prompt: 原始问题
    - response: 模型生成的回答
    - ground_truth: 标准答案（来自数据集）
    """
    response = data_item["response"]
    ground_truth = data_item["ground_truth"]
    
    # 提取 \boxed{} 中的答案
    pattern = r'\\boxed\{(.+?)\}'
    match = re.search(pattern, response)
    if match:
        predicted = match.group(1).strip()
        if predicted == ground_truth.strip():
            return 1.0
    return 0.0
```

## Reward 设计的常见模式

### 1. 稀疏 vs 密集 Reward

```python
# 稀疏 reward：只在最后给分
# 优点：简单、不容易 reward hack
# 缺点：学习信号弱，收敛慢
sparse_reward = 1.0 if task_correct else 0.0

# 密集 reward：过程也给分
# 优点：学习信号强，收敛快
# 缺点：容易被 hack
dense_reward = (
    0.3 * has_reasoning_steps +    # 有推理过程
    0.2 * format_correct +          # 格式正确
    0.5 * answer_correct            # 答案正确
)
# ⚠️ 经验: 密集 reward 中间项的权重不要太大
# 否则模型会学会 "写很长的推理过程但答案错误"
```

### 2. 组合 Reward

```python
def composite_reward(prompt, response, ground_truth):
    # 正确性 reward
    correctness = 1.0 if is_correct(response, ground_truth) else 0.0
    
    # 长度惩罚（鼓励简洁）
    length_penalty = -0.001 * max(0, len(response) - 500)
    
    # 格式奖励
    format_bonus = 0.1 if follows_format(response) else 0.0
    
    # KL penalty（通常在 PPO/GRPO 训练器中自动添加）
    # 这里不需要手动加
    
    return correctness + length_penalty + format_bonus
```

### 3. 多任务 Reward

```python
def multi_task_reward(data_item):
    task_type = data_item["task_type"]
    
    if task_type == "math":
        return math_exact_match(data_item)
    elif task_type == "code":
        return code_execution(data_item)
    elif task_type == "general":
        return llm_judge(data_item)
    else:
        raise ValueError(f"Unknown task: {task_type}")
```

## Reward Hacking 防御

模型会学会利用 reward 的漏洞。常见的 hack：

```
1. 重复输出 → 某些 RM 给高分
   对策: 加重复检测惩罚

2. 过长输出 → 累积中间 reward
   对策: 长度惩罚或直接 truncate

3. 格式游戏 → 用特定格式触发 RM 高分
   对策: 规则检查 + RM 多样化

4. 答案抄 prompt → 在 prompt 中找到答案直接搬
   对策: 数据清洗，确保答案不在 prompt 中
```

## 调试 Reward 的方法

```python
# 1. Reward 分布分析
import numpy as np

rewards = [reward_fn(item) for item in eval_set]
print(f"Mean: {np.mean(rewards):.3f}")
print(f"Std:  {np.std(rewards):.3f}")
print(f"Min:  {np.min(rewards):.3f}")
print(f"Max:  {np.max(rewards):.3f}")

# 理想情况：mean ≈ 0.3-0.7, std > 0.1
# 如果 mean ≈ 0 → 任务太难
# 如果 mean ≈ 1 → 任务太简单
# 如果 std ≈ 0 → reward 没有区分度

# 2. 对比好坏样本的 reward
# 确保好回答的 reward > 坏回答的 reward
# 如果不是，说明 reward function 有 bug
```

## 相关

- [[AI/3-LLM/Frameworks/verl/Sandbox Fusion 沙箱|Sandbox Fusion 沙箱]] — 代码执行类 reward
- [[AI/3-LLM/Frameworks/verl/Post-Training 数据准备|Post-Training 数据准备]] — 数据中的 ground truth
- [[AI/3-LLM/Frameworks/verl/性能调优|性能调优]] — reward 计算的性能
- [[AI/2-Agent/Agentic-RL/Agentic-RL-Training-verl|Agentic RL Training]] — Agent 场景的 reward 设计
