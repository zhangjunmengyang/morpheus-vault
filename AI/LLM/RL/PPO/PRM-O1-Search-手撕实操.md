---
title: "PRM & O1 Search 手撕实操"
brief: "Process Reward Model（PRM）完整实现：step-level reward打分、Beam Search与MCTS树搜索集成，O1-style test-time compute scaling，含PRM vs ORM对比，是Agent Agentic RL中credit assignment的实践基础，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, prm, process-reward, mcts, beam-search, test-time-compute, pytorch]
related:
  - "[[AI/LLM/RL/PPO/PPO-手撕实操-MA-RLHF|PPO-手撕实操-MA-RLHF]]"
  - "[[AI/LLM/RL/GRPO/GRPO-手撕实操|GRPO-手撕实操]]"
  - "[[AI/Agent/Agentic-RL/GiGPO-Group-in-Group-Policy-Optimization|GiGPO]]"
  - "[[AI/Agent/Agentic-RL/Long-Horizon-Credit-Assignment-专题综述|Long-Horizon CA专题]]"
---

# PRM + O1 推理搜索 手撕实操 —— MA-RLHF

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

## 一、原理概述

PRM（Process Reward Model）对推理过程的每一步打分，而非仅对最终结果打分（ORM）。结合 Step-Wise SFT 和 PRM 搜索，可以实现 o1 风格的推理能力。

**完整 Pipeline**：
1. **Step-Wise SFT**：训练模型逐步生成推理过程，步骤间用 SEP token 分隔
2. **PRM 训练**：在 SEP token 位置预测 Positive/Negative，判断每步正确性
3. **PRM-Search**：交替进行 Step-Wise 生成和 PRM 验证，搜索最优推理路径
4. **MCTS**：更进一步，用蒙特卡洛树搜索探索推理空间

## 二、核心实现

### 2.1 Process Reward Model 基础

**原理**：PRM 对 CoT 推理的每一步打分。输入为 prompt + 多步推理，输出为每步的正确/错误判断。

```python
class PRM(nn.Module):
    def __init__(self, vocab_size=100, embd_size=128, num_class=2):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, embd_size)
        self.wq = nn.Linear(embd_size, embd_size)
        self.wk = nn.Linear(embd_size, embd_size)
        self.wv = nn.Linear(embd_size, embd_size)
        self.wo = nn.Linear(embd_size, embd_size)
        self.head = nn.Linear(embd_size, 2)  # 二分类：Positive/Negative
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.embd(x)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        attn = q @ k.t() @ v
        last_hidden = self.wo(attn)
        logits = self.head(last_hidden)
        return {'logits': logits, 'logprob': self.log_softmax(logits)}

# PRM 训练：每步的 label 为 0(negative) 或 1(positive)
for data, label in zip(data_list, label_list):
    logprob = model(data[0])['logprob']
    loss = -logprob[-1, label]  # 最后一个 token 的分类 loss
    loss.backward()
```

### 2.2 Step-Wise SFT 数据构建

**格式**：`#SYSTEM:...\n#USER:{question}\n#ASSISTANT:{step1}<SEP>{step2}<SEP>...{stepN}<SEP><EOS>`

```python
SEP_TOKEN = '<|reserved_special_token_1|>'
system_prompt = '#SYSTEM:you should step-wise answer math question.\n'

def format_step_sft(example):
    question = example['prompt']
    steps = example['completion']
    
    prompt_format = system_prompt + '#USER:' + question + '\n#ASSISTANT:'
    prompt_encode = tokenizer.encode(prompt_format, return_tensors='pt')[0]
    prompt_len = prompt_encode.shape[0]
    
    # 每步后加 SEP token，最后加 EOS
    response = ''
    for step in steps:
        response += step + SEP_TOKEN
    response += tokenizer.eos_token
    response_encode = tokenizer.encode(response, return_tensors='pt')[0][1:]
    
    input_ids = torch.cat((prompt_encode, response_encode), dim=0)
    # Label：仅回答部分，左移一位
    labels = torch.clone(input_ids)
    labels[:prompt_len] = -100
    labels = torch.roll(labels, shifts=-1)
    return {'input_ids': input_ids, 'labels': labels}
```

### 2.3 PRM 数据构建与训练

**关键思想**：复用语言模型的 LM Head 做 PRM，用 "Positive"/"Negative" 的 token id 作为分类标签。

```python
# PRM 复用 LM 的 128K 分类头
positive_id = tokenizer('Positive', add_special_tokens=False)['input_ids']  # [36590]
negative_id = tokenizer('Negative', add_special_tokens=False)['input_ids']  # [39589]
label_map = {0: negative_id, 1: positive_id}

def format_step_prm(example):
    # ... 拼接 prompt + response（同 SFT）
    # 在每个 SEP token 位置设置 label
    prompt_label = torch.ones_like(input_ids) * -100
    for place_idx, label in zip(place_indexs, labels):
        prompt_label[place_idx - 1] = label_map[label][0]
    return {'input_ids': input_ids, 'label_ids': prompt_label}
```

**PRM 推理**：对已生成的多步解答，**一次前向**即可并行评估所有 SEP 位置的正确性。

```python
# 推理：找到所有 SEP token 位置，取 Positive/Negative 概率
output = prm_model(input_ids)
logits = output.logits
sep_positions = torch.where(input_ids[0] == SEP_TOKEN_ID)[0]
logits_at_sep = logits[0, sep_positions, :]
logits_binary = logits_at_sep[:, [negative_token_id, positive_token_id]]
probs = F.softmax(logits_binary, dim=1)
predictions = torch.argmax(probs, dim=1)  # 0=Negative, 1=Positive
```

### 2.4 Step-Wise 生成与 PRM 搜索

```python
# Step 1: SFT 模型生成 step-wise 解答
output = model.generate(input_ids, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=0.6)

# Step 2: PRM 评分
model = PeftModel.from_pretrained(model, prm_lora_path)
with torch.no_grad():
    logits = model(input_ids).logits
sep_idx = torch.where(input_ids == SEP_TOKEN_ID)[0]
for step, label, prob in zip(steps, predictions, probs):
    print(f"Step: {step}, Label: {'Correct' if label else 'Wrong'}, Prob: {prob}")
```

### 2.5 MCTS for LLM（五子棋示例）

**AlphaGo-Zero 风格的 MCTS 实现**：

```python
class GomokuNet(nn.Module):
    """Policy + Value Network"""
    def forward(self, x):
        # ... CNN 特征提取
        policy = self.fc2(x4)        # 动作概率分布
        value = torch.tanh(self.fc3(x4))  # 局面价值估计
        return policy, value

class MCTSNode:
    def uct_value(self, c=1.4):
        """UCB 选择公式"""
        q = self.value / self.visits
        u = c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u
    
    def expand(self, policy):
        """一次展开所有合法子节点"""
        for action in valid_actions:
            child_state = self.state.clone()
            child_state.move(action)
            child_node = MCTSNode(child_state, parent=self)
            child_node.prior = policy[0, action_id]
            self.children[action] = child_node
    
    def backpropagate(self, value):
        """回溯更新 Q 值"""
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(-value)  # 对手视角取反

def mcts_move(state, net, num_simulations=1000):
    root = MCTSNode(state)
    for _ in range(num_simulations):
        node = root
        # Selection
        while len(node.children) != 0:
            node = node.select_child()[1]
        # Expansion
        if not node.state.is_terminal():
            policy, _ = net(node.state.board)
            node = node.expand(F.softmax(policy, dim=1))
        # Evaluation
        _, value = net(node.state.board)
        # Backpropagation
        node.backpropagate(value.item())
    return max(root.children.items(), key=lambda x: x[1].visits)[0]
```

**MCTS 训练**：

```python
for episode in range(episodes):
    state = GomokuState()
    states, policies, values, actions = [], [], [], []
    while not state.is_terminal():
        policy, value = net(state.board)
        action = mcts_move(state, net, num_simulations=100)
        state.move(action)
    
    # Loss = Policy CE + Value MSE
    policy_loss = CrossEntropyLoss(pred_policies, actual_actions)
    value_loss = MSE(pred_values, actual_game_results)
    loss = policy_loss + value_loss
```

### 2.6 O1 数据集构建

**数据来源与特点**：
- **PRM800K**：有 step 标注，含错误 step，部分不完整，有反思风格
- **GSM8K**：小学数学，所有 step 正确
- **MATH**：竞赛难题，无 step 分隔（整段解答当一个 step）

```python
# 统一格式：{prompt, completions, labels, is_step, is_end, type}
# Step SFT 数据：过滤掉含错误 step 的数据
# PRM 数据：保留含错误 step 的数据（需要正负样本）

# PRM800K 特殊处理
def filter(dataset):
    for data in dataset:
        # 检查最后一步是否为 Answer
        if '\n\n# Answer\n\n' in data['completions'][-1]:
            data['is_end'] = True
        # 过滤全部正确的数据用于 Step SFT
        if all(data['labels']):
            data_list.append(data)
    # 去重
    # ...
```

## 三、工程实践（配套代码）

> 完整代码见：
> - PRM 训练：`/tmp/ma-rlhf/ma-rlhf/process_reward_model.py`
> - Step 生成 + PRM 搜索：`/tmp/ma-rlhf/ma-rlhf/generate_step_prm.py`
> - 数据处理：`/tmp/ma-rlhf/data/o1_math_dataset_process.ipynb`、`process_prm800k_to_step_sft.ipynb`

### process_reward_model.py 关键架构

```python
# 使用 LoRA 微调 LM Head 做 PRM（仅训练 LM Head 的 LoRA）
peft_config = create_peft_prm_lm_head(True)
model = get_peft_model(model, peft_config)

# 复用 Trainer 做 PRM 训练
trainer = Trainer(model, args=training_args,
                  train_dataset=train_datasets,
                  data_collator=DataCollatorForSFT(tokenizer))
```

### generate_step_prm.py 推理流程

```python
# 1. SFT 模型生成 step-wise 解答
output = model.generate(input_ids, do_sample=True, temperature=0.6)

# 2. 加载 PRM LoRA 适配器
model = PeftModel.from_pretrained(model, prm_lora_path)

# 3. 在 SEP token 位置提取 Positive/Negative 概率
sep_idx = torch.where(input_ids == SEP_TOKEN_ID)[0]
logits_at_sep = logits[0, sep_idx, [negative_id, positive_id]]
probs = F.softmax(logits_at_sep, dim=1)
```

## PRM Search 变体谱系

```
PRM Search（Best-Step Greedy，本实现）
    ↓ 扩展搜索宽度
Beam Search（每步保留 K 条候选，PRM 打分选 Top-K）
    ↓ 增加前瞻
MCTS（Monte Carlo Tree Search）→ 节点价值 = PRM 分 + MC rollout 估计
    ↓ 训练时使用
TSR（Trajectory Search Rollout）→ 把搜索到的好轨迹作为 RL 训练数据
GiGPO → 把"anchor state"的多条轨迹做组内 advantage 估计
```

**TTC（Test-Time Compute）Scaling 的逻辑**：不改模型权重，靠更多推理计算（更多搜索步骤）提升正确率。PRM Search 是最直接的 TTC 实现——问题越难，允许更多步骤重试。

## 四、关键洞察与总结

1. **PRM vs ORM**：PRM 在每步提供反馈，比 ORM（仅终点反馈）更密集——类似 RL 中 dense reward vs sparse reward
2. **复用 LM Head 做 PRM 是巧妙的**：用 "Positive"/"Negative" token 的概率做分类，不需要额外的分类头
3. **SEP token 是关键设计**：将连续推理切分为可评估的步骤
4. **MCTS 与 LLM 的 Gap**：
   - 动作空间：棋类 ~225，LLM ~128K
   - 树深度：棋类 ~100步，LLM token 序列可达数千
   - 采样成本：LLM 模拟到 terminal 成本过高
   - 解决方向：PRM 提供中间反馈，减少模拟深度
5. **数据质量决定上限**：PRM800K 有大量重复和不完整数据，需要仔细过滤
6. **Step SFT + PRM 的互补**：Step SFT 教模型"怎么想"，PRM 教模型"判断对错"——两者结合才能实现有效推理搜索
