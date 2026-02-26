---
title: "O1 PRM 搜索完整实现"
brief: "Process Reward Model + MCTS-O1 搜索完整实现：PRM 逐步打分、MCTS 树搜索（UCT 选择 + 展开 + 回溯）、Best-of-N vs MCTS 效率对比。理解 OpenAI o1/o3 的推理时搜索机制。"
date: 2026-02-25
type: code-practice
source: "https://github.com/dhcode-cpp/MA-RLHF"
tags: [code-practice, llm-engineering, ma-rlhf, rl, prm, mcts, test-time-compute, o1]
related:
  - "[[AI/LLM/RL/PPO/PRM-O1-Search-手撕实操]]"
  - "[[AI/LLM/RL/PPO/RLHF-PPO-完整Pytorch实现]]"
  - "[[AI/LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
---

# O1 PRM 搜索完整实现

> 来源：`ma-rlhf/notebook/o1/` — o1_prm_search.ipynb + process_reward_modeling.ipynb + gomoku-mcts-pytorch.ipynb
> Author: xiaodongguaAIGC

---

## 1. PRM vs ORM：步骤级 vs 结果级奖励

### 1.1 核心区别

| | ORM（Outcome Reward Model） | PRM（Process Reward Model） |
|---|---|---|
| **评估粒度** | 只看最终答案是否正确 | 评估每一步推理是否正确 |
| **反馈信号** | 稀疏：整个 chain 只有一个 reward | 稠密：每一步都有 reward |
| **错误定位** | 无法定位哪一步出错 | 精确定位第一个错误步骤 |
| **训练数据** | 只需 (question, answer, correct?) | 需要 step-level 标注 (step, correct?) |
| **搜索引导** | 只能在最终结果上选择 | 可以在中间步骤上剪枝/分支 |

### 1.2 为什么 PRM 更适合数学推理？

数学推理是**串行依赖**的：第 3 步错了，后面全错。ORM 只能告诉你"最终答案错了"，但 PRM 能说"第 3 步开始就偏了"。这使得 PRM 可以：
1. 在搜索时及早剪枝错误分支
2. 在训练时提供更精确的 credit assignment
3. 与 beam search 结合实现 test-time compute scaling

---

## 2. PRM 训练

### 2.1 Step-Level 标注数据格式

使用 `prm800k` 数据集（OpenAI），每条数据包含：

```
{
  "prompt": [{"content": "数学题目"}],
  "completion": [
    {"content": "步骤1文本"},
    {"content": "步骤2文本"},
    {"content": "步骤3文本"},
  ],
  "labels": [1, 1, 0]  // 1=正确, 0=错误
}
```

### 2.2 数据预处理：Step-Wise 格式

```
#SYSTEM: you should step-wise score the correctness of answer step
#USER: {Question}
#ASSISTANT: {step1}<SEP>{step2}<SEP>...{stepN}<SEP><EOS>
```

每个 `<SEP>` token 的位置对应一个 step 的分类标签（Positive/Negative）。

```python
SEP_TOKEN = '<|reserved_special_token_1|>'
positive_id = tokenizer('Positive', add_special_tokens=False)['input_ids']
negative_id = tokenizer('Negative', add_special_tokens=False)['input_ids']

def format_step_prm(example):
    question = example['prompt']
    steps = example['completion']
    labels = example['labels']

    prompt_format = format_template(prm_system_prompt, question)
    prompt_encode = tokenizer.encode(prompt_format, return_tensors='pt')[0]
    prompt_len = prompt_encode.shape[0]

    response_token_ids = []
    place_indexs = []  # SEP token 的位置索引
    label_idx = []     # 每个 SEP 对应的标签 token id

    for step, label in zip(steps, labels):
        response = step + SEP_TOKEN
        step_ids = tokenizer.encode(response, add_special_tokens=False)
        response_token_ids.extend(step_ids)
        place_indexs.append(len(response_token_ids) + prompt_len)
        label_idx.append(label_map[label][0])  # 0→Negative, 1→Positive

    response_token_ids.extend([tokenizer.eos_token_id])
    input_ids = torch.cat((prompt_encode, torch.tensor(response_token_ids)), dim=0)

    # 只在 SEP token 位置有标签，其余为 -100（忽略）
    prompt_label = torch.ones_like(input_ids) * -100
    place_indexs = [idx - 1 for idx in place_indexs]
    prompt_label[place_indexs] = torch.tensor(label_idx, dtype=torch.long)

    return {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids),
            'label_ids': prompt_label}
```

### 2.3 PRM 模型架构

**关键设计**：PRM 复用语言模型，不需要单独的分类头。

```python
# PRM 复用 SFT 后的语言模型权重
prm_model = AutoModelForCausalLM.from_pretrained(sft_model_path)
```

在 SEP token 位置，取 `Positive` 和 `Negative` 这两个 token 的 logit 做 softmax，得到步骤正确的概率：

```python
# 找到所有 SEP token 的位置
idx = torch.where(input_ids == SEP_TOKEN_ID)[0]

# 一次 forward 并行评估所有步骤
logits = prm_model(input_ids=input_ids.unsqueeze(0),
                   attention_mask=attention_mask.unsqueeze(0)).logits

# 取 SEP 位置的 logits
sep_logits = logits[0, idx, :]

# 只取 Positive/Negative 两个 token 的概率
sep_logits_class = sep_logits[:, [positive_id, negative_id]].squeeze(dim=2)
sep_prob = F.softmax(sep_logits_class, dim=1)
_, pred = torch.max(sep_prob, dim=1)  # 0=Negative, 1=Positive
```

### 2.4 简化 PRM（来自 process_reward_modeling.ipynb）

独立的极简 PRM 实现，展示核心思想：

```python
class PRM(nn.Module):
    def __init__(self, vocab_size=100, embd_size=128, num_class=2):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, embd_size)
        self.wq = nn.Linear(embd_size, embd_size)
        self.wk = nn.Linear(embd_size, embd_size)
        self.wv = nn.Linear(embd_size, embd_size)
        self.wo = nn.Linear(embd_size, embd_size)
        self.head = nn.Linear(embd_size, 2)         # 二分类：正确/错误
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embd(x)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        attn = q @ k.t() @ v
        last_hidden = self.wo(attn)
        logits = self.head(last_hidden)
        logprob = self.log_softmax(logits)
        return {'last_hidden_states': last_hidden, 'logprob': logprob}

# 训练数据：CoT 的每一步累积序列 + 正确/错误标签
# A=prompt, B/D/E=正确步骤, C=错误步骤
data =  [[B, C], [C, D, E], [B, D, E]]
label = [[1, 0], [0, 1, 1], [1, 1, 1]]

# 训练：对最后一个 token 的分类 loss
for data_item, label_item in zip(data_list, label_list):
    logprob = model(data_item[0])['logprob']
    loss = -logprob[-1, label_item]  # 最后位置的 NLL
```

---

## 3. Beam Search + PRM 打分

### 3.1 Step-Wise 生成

SFT 模型按步骤生成，以 `<SEP>` token 为步骤分隔符：

```python
def generate_greedy_step(input, model, max_tokens=10, temperature=0.9,
                         past_key_values=None):
    """生成一个推理步骤，遇到 SEP token 停止"""
    result = []
    for i in range(max_tokens):
        with torch.no_grad():
            output = model(input_ids=input, past_key_values=past_key_values,
                           use_cache=False)
            logits = output.logits[0, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        result.append(next_token.item())
        input = next_token.unsqueeze(dim=0)

        if next_token == SEP_TOKEN_ID:
            return result, output.past_key_values

    result.append(SEP_TOKEN_ID)  # 强制补 SEP
    return result, output.past_key_values
```

### 3.2 PRM 验证函数

```python
def verify_function(input_prm, prm_model, prm_past_key_values=None):
    """对输入序列的最后一个 SEP token 做正确性判断"""
    prm_model.eval()
    with torch.no_grad():
        output = prm_model(input_ids=input_prm,
                           past_key_values=prm_past_key_values,
                           use_cache=True)
        last_logits = output.logits[0, -1, :]

    # Positive vs Negative 的 softmax 概率
    p_positive = torch.exp(last_logits[positive_id]) / (
        torch.exp(last_logits[positive_id]) + torch.exp(last_logits[negative_id]))
    p_negative = 1 - p_positive

    return p_positive > p_negative, p_positive.item(), p_negative.item()
```

### 3.3 PRM-Search：交替生成与验证

```python
def prm_search(input, input_prm, model, prm_model,
               max_step=10, accumulate_max_step=30, temperature=0.9):
    """
    核心搜索循环：
    1. SFT 模型生成一步 → 2. PRM 验证 → 3. 通过则接受，不通过则重试
    """
    i = 0
    acc_max_step = 0

    while i < max_step:
        if input[0, -1] == tokenizer.eos_token_id:
            break  # 生成完毕

        # Step 1: 生成一个推理步骤
        step_idx, new_kv = generate_greedy_step(input=input, model=model)
        step_tensor = torch.tensor(step_idx, dtype=torch.long).unsqueeze(0)

        # Step 2: PRM 验证这一步
        new_input_prm = torch.cat((input_prm, step_tensor), dim=1)
        result, p_pos, p_neg, _ = verify_function(
            input_prm=new_input_prm, prm_model=prm_model)

        # Step 3: 接受或拒绝
        if result:  # PRM 判断正确 → 接受这一步
            input = torch.cat((input, step_tensor), dim=1)
            input_prm = new_input_prm
            i += 1
        # 如果不正确 → 不更新，下次 do_sample 会采样不同的步骤

        acc_max_step += 1
        if acc_max_step > accumulate_max_step:
            break  # 防止无限循环

    return input
```

**关键机制**：
- 生成使用 `do_sample`（temperature sampling），所以同一位置可以采样出不同的步骤
- 被 PRM 拒绝的步骤直接丢弃，不更新 KV cache
- `accumulate_max_step` 限制总重试次数，防止死循环

---

## 4. MCTS 基本框架（来自 gomoku-mcts-pytorch）

### 4.1 为什么看五子棋 MCTS？

五子棋是理解 MCTS 的最佳教学场景：状态离散、规则简单、可以验证终态。Notebook 实现了完整的 AlphaGo Zero 风格 MCTS，包含 policy/value 网络。

### 4.2 四个阶段

```
Selection → Expansion → Simulation → Backpropagation
```

**Selection**：从根节点出发，用 UCB 公式选子节点，直到到达叶子节点。

```python
def select_child(self):
    return max(self.children.items(), key=lambda x: x[1].uct_value())

def uct_value(self, c=1.4):
    if self.visits == 0:
        return float('inf')
    q = self.value / self.visits                               # exploitation
    u = c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)  # exploration
    return q + u
```

**Expansion**：叶子节点展开所有合法动作，用 policy network 赋予先验概率。

```python
def expand(self, policy):
    valid_actions = self.state.get_legal_actions()
    for id in range(len(valid_actions)):
        idx = valid_actions[id].item()
        action = [int(idx / self.state.board_size),
                  int(idx % self.state.board_size)]
        child_state = self.state.clone()
        child_state.move(action)
        child_node = MCTSNode(child_state, self)
        child_node.prior = policy[0, idx]  # policy network 的先验
        self.children[tuple(action)] = child_node
    return max_child_node  # 返回概率最大的子节点
```

**Simulation / Evaluation**：用 value network 估计当前局面的价值（替代随机 rollout）。

```python
if not node.state.is_terminal():
    value = node.state.get_reward()
    if value == 0:  # 未结束，用 value network 估计
        _, value = net(node.state.board)
        value = value.item()
```

**Backpropagation**：将价值沿路径回传，更新所有祖先节点。

```python
def backpropagate(self, value):
    self.visits += 1
    self.value += value
    if self.parent:
        self.parent.backpropagate(-value)  # 对手视角取负
```

### 4.3 Policy + Value Network 联合训练

```python
# Policy loss: CrossEntropy（MCTS 选择的动作作为标签）
policy_loss = CrossEntropyLoss(pred_policies, mcts_actions)

# Value loss: MSE（预测值 vs 真实终局奖励）
value_loss = MSE(pred_values, actual_game_outcome)

# 总 loss
loss = policy_loss + value_loss
```

### 4.4 完整 MCTS 搜索函数

```python
def mcts_move(state, net, num_simulations=1000):
    root = MCTSNode(state)
    for _ in range(num_simulations):
        node = root
        # Selection: UCB 选到叶子
        while len(node.children) != 0:
            node = node.select_child()[1]
        # Expansion
        if len(node.state.get_legal_actions()) != 0 and not node.state.is_terminal():
            policy, _ = net(node.state.board)
            policy = torch.softmax(policy, dim=1)
            node = node.expand(policy)
        # Evaluation + Backprop
        value = node.state.get_reward() or net(node.state.board)[1].item()
        node.backpropagate(value)

    # 选择访问次数最多的动作（最鲁棒）
    return max(root.children.items(), key=lambda x: x[1].visits)[0]
```

---

## 5. Test-Time Compute Scaling

### 5.1 核心思想

在有 **verifier**（PRM 或正确答案检查器）的任务上，推理时投入更多计算 = 更好结果。

**机制**：
- 生成 N 个候选 → PRM 打分 → 选最优 → 准确率随 N 增长
- 或者 beam search：每步保留 top-k → PRM 评分 → 最终选最高分路径
- 或者 PRM-Search（本 notebook）：逐步生成+验证，错了重试

### 5.2 为什么有效？

1. **搜索放大了模型能力**：即使单次生成的准确率只有 30%，搜索 10 次后至少有一次正确的概率是 $1 - 0.7^{10} \approx 97\%$
2. **PRM 提供了可靠的步骤级信号**：不需要等到最终答案才知道对错，中间就能剪枝
3. **计算换质量的 scaling law**：在数学推理任务上，test-time compute 的效果可以类比 train-time compute 的 scaling

### 5.3 适用条件

- 任务必须有 **可验证的正确答案**（数学、代码、逻辑推理）
- 需要一个可靠的 verifier（PRM、单元测试、形式化验证）
- 对延迟不敏感的场景（允许多次采样和验证）

---

## 6. O1 的推断

### 6.1 Chain-of-Thought + PRM 可能是 O1 的核心机制

Notebook 作者总结的 LLM 与 MCTS 的 gap：

1. LLM 的动作空间巨大（128k 词表） vs 棋类游戏有限动作集
2. LLM 的推理链极深（1024+ tokens） vs 棋局深度有限
3. LLM 模拟采样到正确 terminal 的难度大，有效 feedback 太少
4. 关键问题：如何减少模拟成本？如何有效采样正确推理步骤？如何获得准确 feedback？

### 6.2 O1 的可能架构

综合本 notebook 的实现和公开信息推断：

```
O1 ≈ Step-Wise SFT（生成器） + PRM（验证器） + 搜索策略
```

- **生成器**：经过 step-wise SFT 训练，能按步骤生成推理链
- **验证器**：PRM 对每一步打分，提供稠密的 reward signal
- **搜索**：可能是 beam search、best-of-N、或更复杂的 tree search
- **呈现**：对外表现为"思考更久 → 回答更好"的 test-time scaling

---

## 7. 面试考点

### Q1：PRM 和 ORM 分别适合什么场景？

ORM 适合结果可验证但中间过程难以评估的任务（如代码执行、数学最终答案）。PRM 适合推理过程很重要的任务（数学证明、多步推理），它能提供每一步的信号用于搜索剪枝和训练 credit assignment。PRM 的缺点是标注成本高——需要人工标注每一步的正确性，而 ORM 只需最终答案。

### Q2：PRM 如何复用语言模型架构？为什么不用单独的分类头？

Notebook 的巧妙设计：PRM 直接复用 LM 的 128k 分类头，只关注 `Positive` 和 `Negative` 两个 token 对应的 logit。这样：(1) 可以直接从 SFT 模型 warm start；(2) 训练数据格式与 SFT 完全一致（只是标签在 SEP token 位置）；(3) 推理时一次 forward 可以并行评估所有步骤。

### Q3：PRM-Search 中为什么 do_sample 很重要？

如果用 greedy decoding，被 PRM 拒绝后重试会生成完全相同的步骤，陷入死循环。Temperature sampling 引入随机性，使得同一个前缀可以采样出不同的后续步骤，给了搜索"探索"的能力。这与 MCTS 中 UCB 的 exploration 项思想一致。

### Q4：MCTS 的 UCB 公式中 exploration 和 exploitation 分别是什么？

`Q = value / visits`（exploitation）：选平均价值高的节点。`U = c * prior * √(parent.visits) / (1 + visits)`（exploration）：选访问次数少但先验概率高的节点。两者平衡保证既深挖好路径，又不忽略未探索的分支。`prior` 来自 policy network，`c=1.4` 控制探索强度。

### Q5：从 MCTS 到 LLM 搜索，最大的挑战是什么？

三个核心挑战：(1) **动作空间**：棋类 ~361 个合法动作，LLM 128k 词表，无法穷举展开；(2) **深度**：棋局 ~200 步，LLM 推理链可能 1000+ tokens，树的深度导致搜索成本爆炸；(3) **模拟成本**：棋类可以快速模拟到终局，LLM 需要完整生成才能评估，且正确路径稀疏。解决方向：step-level 搜索（而非 token-level）+ PRM 替代完整 rollout + beam search 替代完整 MCTS。
