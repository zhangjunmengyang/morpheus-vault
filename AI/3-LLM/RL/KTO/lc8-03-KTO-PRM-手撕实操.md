---
title: lc8 — KTO + PRM-Search 从零手写
brief: 从零实现 KTO（Kahneman-Tversky Optimization，无需配对数据的偏好优化）和 o1 风格的 PRM 搜索（Process Reward Model + beam search/MCTS）。核心洞察：KTO 用前景理论替代 BT 模型，PRM-Search 把 RL 信号转化为搜索树中的过程奖励。
date: 2026-02-26
type: code-practice
source: https://github.com/dhcode-cpp/MA-RLHF
tags:
  - code-practice
  - llm-engineering
  - ma-rlhf
  - rl-alignment
  - kto
  - prm
  - o1-search
  - lc8
related:
  - "[[AI/3-LLM/RL/KTO/KTO-手撕实操]]"
  - "[[AI/3-LLM/RL/KTO/KTO-完整Notebook实现]]"
  - "[[AI/3-LLM/RL/PPO/PRM-O1-Search-手撕实操]]"
  - "[[AI/3-LLM/MA-RLHF课程/lc8-RL×LLM-MOC]]"
  - "[[RLHF-DPO-2026-技术全景]]"
---

# KTO + PRM-Search 从零手写

> MA-RLHF Batch B / lc8-KTO + o1_prm_search
> Source: `notebook/KTO/KTO.ipynb` + `notebook/o1/o1_prm_search.ipynb`
> Author: xiaodongguaAIGC / dhcode-cpp
> 评分: ★★★★★

---

## TL;DR

两个 notebook 联合精读，覆盖偏好学习和 Test-Time Compute 的关键技术：

- **KTO**：无需成对数据，每条样本只需 (prompt, response, label: good/bad)，基于前景理论建模人类偏好
- **PRM-Search**：步骤级验证模型 + 贪心搜索，o1-style 的完整实现链：step-wise SFT → PRM 训练 → PRM-Guided 生成

---

## Part 1：KTO（Kahneman-Tversky Optimization）

### KTO vs DPO 的核心区别

| 维度 | DPO | KTO |
|------|-----|-----|
| 数据格式 | (prompt, chosen, rejected) 成对 | (prompt, response, label: True/False) 单条 |
| 数学基础 | Bradley-Terry 偏好模型 | Kahneman-Tversky 前景理论 |
| 优势 | 理论简洁 | **数据效率高**（不需要配对），适合有大量单标签数据的场景 |
| z_ref | 无 | 需要估计 KL baseline（batch 内均值） |

### 前景理论的直觉

人类对损失和收益的感受不对称：
- **损失厌恶**：失去 100 元的痛苦 > 获得 100 元的快乐
- KTO 把这个不对称性建模进 loss：好的 response 要"收益足够大"才认为对齐，坏的 response 要"惩罚足够重"

### 数据格式与 Tokenization

```python
kto_dataset_dict = {
    "prompt": ["Hey, hello", "How are you", ...],
    "completion": ["hi nice to meet you", "leave me alone", ...],
    "label": [True, False, ...]  # True=好回复, False=坏回复
}

def process_kto_dataset(example):
    prompt_id = tokenizer('USER:' + example['prompt'] + 'ASSISTANT:')['input_ids']
    completion_id = tokenizer(example['completion'])['input_ids'] + [eos]
    example['input_ids'] = prompt_id + completion_id
    example['label_mask'] = [0] * len(prompt_id) + [1] * len(completion_id)  # 只在response上算loss
    example['attention_mask'] = [1] * len(example['input_ids'])
    return example
```

### per-token log-prob（带 label_mask 归一化）

```python
def get_probs(logits, labels, mask):
    per_token_logps = torch.gather(
        logits.log_softmax(-1),
        dim=2,
        index=labels.unsqueeze(2)
    ).squeeze(2)
    per_token_logps *= mask / mask.sum()    # mask + 归一化（重要！）
    return per_token_logps
```

注意：KTO 中 `get_probs` 不仅 mask 掉 prompt，还做了 `/ mask.sum()` 的归一化（等价于均值而非求和），这与 DPO 的实现略有不同。

### KL Baseline 估计

$$z_{ref} = \mathbb{E}_{x' \sim D}\left[\beta \cdot KL(\pi_\theta(y'|x') \| \pi_{ref}(y'|x'))\right]$$

```python
kl = (probs - ref_probs).mean().detach()    # per-token log-ratio 的均值近似 KL
# detach：kl 只作为 baseline，不参与梯度
```

**关键**：`detach()` 阻断 kl 的梯度路径，保证它只作为常数 baseline。

### Chosen/Rejected 分离

```python
chosen_id   = [k for k, v in enumerate(batch['label']) if v == True]
rejected_id = [k for k, v in enumerate(batch['label']) if v == False]

chosen_probs   = probs[chosen_id, :]
ref_chosen_probs = ref_probs[chosen_id, :]
rejected_probs = probs[rejected_id, :]
ref_rejected_probs = ref_probs[rejected_id, :]

chosen_ratio   = chosen_probs   - ref_chosen_probs    # log(π_θ/π_ref) for chosen
rejected_ratio = rejected_probs - ref_rejected_probs  # log(π_θ/π_ref) for rejected
```

同一个 batch 里，chosen 和 rejected **不需要配对**，只需要分别存在。

### KTO Loss

$$v_{KTO}(x,y;\beta) = \begin{cases}
\sigma(r_{KTO}(x,y) - z_{ref}) & \text{if desirable} \\
\sigma(z_{ref} - r_{KTO}(x,y)) & \text{if undesirable}
\end{cases}$$

$$L_{KTO} = \mathbb{E}[w(y)(1 - v_{KTO}(x,y;\beta))]$$

```python
beta = 0.1
desirable_weight   = 1.33    # λ_D（好样本权重，通常略大）
undesirable_weight = 1.0     # λ_U

# 好回复：希望 r_KTO - z_ref 尽量大（sigmoid 接近 1，loss 接近 0）
chosen_losses   = 1 - F.sigmoid(beta * (chosen_ratio   - kl))   # [n_chosen, T]

# 坏回复：希望 z_ref - r_KTO 尽量大（即 r_KTO 尽量小）
rejected_losses = 1 - F.sigmoid(beta * (rejected_ratio - kl))   # [n_rejected, T]

losses = torch.cat(
    (desirable_weight * chosen_losses,
     undesirable_weight * rejected_losses),
    0,
)
kto_losses = losses.nanmean()
```

### KTO Loss 的直觉

- **好回复**：`chosen_ratio = log(π_θ/π_ref)`，希望它 > KL baseline（z_ref）→ 证明模型相对 ref 更倾向生成好回复
- **坏回复**：`z_ref - rejected_ratio`，希望 rejected_ratio < KL baseline → 模型相对 ref 更不倾向生成坏回复
- 两者用同一个 z_ref 作为分界点，相当于问："这条回复的质量，是高于还是低于模型整体漂移的水平线？"

### desirable_weight > undesirable_weight 的意义

来自前景理论：损失厌恶意味着坏样本的惩罚天然比好样本的奖励"感觉更重"。1.33:1.0 的权重是 KTO 论文建议值，平衡两者的梯度幅度。

---

## Part 2：PRM-Search（o1-style 步骤验证与引导搜索）

### 整体架构

```
Step 1: Step-wise SFT
   训练模型生成带 <SEP> token 标记的步骤链

Step 2: PRM Training（Process Reward Model）
   在 <SEP> 位置预测该步骤 Positive/Negative
   复用 SFT 的 LM head，只取 Positive/Negative 两个 token 的 logits

Step 3: PRM-Search（Greedy Search with Verification）
   交替：SFT 模型生成一步 → PRM 验证 → 通过则接受，否则重试
```

### Step 1：Step-wise SFT 数据格式

```
SYSTEM: 你需要分步骤回答数学问题，步骤用特殊 token 标记
USER: how to solve y=x^2+2x+1
ASSISTANT: {step1}<SEP>{step2}<SEP>...{stepN}<SEP><EOS>
```

```python
SEP_TOKEN = '<|reserved_special_token_1|>'   # 步骤分隔符

def format_step_sft(example):
    # 构建 response: step1 + SEP + step2 + SEP + ... + EOS
    response = ''
    for step in steps:
        response = response + step + SEP_TOKEN
    response = response + tokenizer.eos_token

    # Label: prompt 部分 = -100（ignore），response 部分 = 下一个 token（causal LM）
    prompt_label = torch.clone(input_ids)
    prompt_label[:prompt_len] = -100
    labels = torch.roll(prompt_label, shifts=-1)    # 左移一位 = next token prediction
    return ...
```

**Causal LM 的 label 处理**：label 是 input 左移一位（因为预测的是下一个 token），prompt 部分设为 -100 忽略。

### Step 2：PRM 训练数据格式

PRM 复用 SFT 模型的 LM Head（vocab_size 个类别），**只在 SEP token 位置预测 Positive/Negative**：

```python
positive_id = tokenizer('Positive', add_special_tokens=False)['input_ids']   # [36590]
negative_id = tokenizer('Negative', add_special_tokens=False)['input_ids']   # [39589]

def format_step_prm(example):
    for step, label in zip(steps, labels):
        response = step + SEP_TOKEN
        step_token_ids = tokenizer.encode(response, add_special_tokens=False)
        response_token_ids.extend(step_token_ids)
        place_indexs.append(len(response_token_ids) + prompt_len)
        label_idx.append(label_map[label][0])    # 0→negative_id, 1→positive_id

    # Label: 只在 SEP 位置有 label（Positive/Negative token id），其他 = -100
    prompt_label = torch.ones_like(input_ids) * -100
    place_indexs = [idx - 1 for idx in place_indexs]    # SEP 前一位是分类预测目标
    prompt_label[place_indexs] = torch.tensor(label_idx)
    return ...
```

**PRM 的核心 trick**：不改变模型结构，直接用 LM Head 预测"下一个 token 是 Positive 还是 Negative"。SEP token 作为分界点，SEP 前一个位置的 logits 就是该步骤正确与否的分类。

### Step 2：PRM 并行推理

```python
# 一次 forward，对所有 SEP 位置并行预测
with torch.no_grad():
    logits = prm_model(input_ids=...).logits    # [1, T, V]

# 找所有 SEP token 的位置
idx = torch.where(test_data['input_ids'] == SEP_TOKEN_ID)[0]
sep_logits = logits[0, idx, :]    # [n_steps, V]

# 只取 Positive/Negative 两个类别
sep_logits_class = sep_logits[:, [positive_id, negative_id]].squeeze(dim=2)  # [n_steps, 2]
sep_prob = F.softmax(sep_logits_class, dim=1)
_, pred = torch.max(sep_prob, dim=1)   # 0=negative, 1=positive
```

**高效性**：一次 forward 同时验证所有步骤，不需要逐步验证 N 次。

### Step 3：PRM-Search（Greedy）

```python
def prm_search(input, input_prm, model, prm_model, max_step=10, ...):
    while i < max_step:
        # 1. SFT 模型生成一步（生成到下一个 SEP token 为止）
        step_idx, new_past_kv = generate_greedy_step(input=input, model=model, ...)

        # 2. PRM 验证这一步
        new_input_prm = torch.cat((input_prm, step_idx_tensor), dim=1)
        is_correct, p_pos, p_neg, _ = verify_function(new_input_prm, prm_model)

        # 3. 接受或重试
        if is_correct:    # 验证通过 → 接受，更新序列
            input = new_input_prm
            input_prm = new_input_prm
            i += 1
        # 否则：不更新 i，下次循环重新生成（do_sample=True 会采样不同输出）
```

**搜索逻辑**：
- 生成一步 → PRM 打分 → 通过则接受，不通过则重试（同一位置）
- `acc_max_step` 限制总尝试次数，防止无限循环
- 由于 `do_sample=True`，每次重试会采样不同的步骤

### KV Cache 优化（代码中的 TODO）

```python
# 当前实现：每次 verify 都重新 forward 全部序列（没有复用 KV Cache）
# TODO：prm_past_key_values 变量已预留，但 use_cache 还是 False
# 优化后：只需要 forward 新增的 step tokens，KV Cache 保存历史

def verify_function(input_prm, prm_model, prm_past_key_values=None):
    output = prm_model(
        input_ids=input_prm,
        past_key_values=prm_past_key_values,
        use_cache=True,   # TODO: 实际生产应开启
    )
    last_logits = output.logits[0, -1, :]    # 只看最后一个 token（SEP）
    past_key_values = output.past_key_values  # 保存 KV Cache
```

### 生成函数：每步生成到 SEP

```python
def generate_greedy_step(input, model, max_tokens=10, temperature=0.9, ...):
    for i in range(max_tokens):
        output = model(input_ids=input, past_key_values=past_kv, use_cache=False)
        logits = output.logits[0, -1, :] / temperature
        probs = F.softmax(logits)
        next_token = torch.multinomial(probs, 1)   # temperature sampling
        result.append(next_token.item())
        input = next_token.unsqueeze(0)
        if next_token == SEP_TOKEN_ID:
            return result, past_kv   # 遇到 SEP 停止
    # 超过 max_tokens 仍未遇 SEP → 强制添加 SEP
    result.append(SEP_TOKEN_ID)
    return result, past_kv
```

---

## 知识体系总结

### 偏好优化全谱系（加入 KTO 后）

```
人类反馈数据
  ├── 成对数据 (y_w, y_l)    → DPO / IPO / RLHF-PPO
  └── 单标签数据 (y, True/False) → KTO

数学基础
  ├── Bradley-Terry     → DPO
  ├── 前景理论          → KTO
  └── KL 约束最优解     → IPO
```

### TTC（Test-Time Compute）谱系

```
PRM（Process Reward Model）
  ├── 验证器：一次 forward 并行验证所有步骤
  └── 训练：SEP token 位置预测 Positive/Negative（复用 LM Head）

Search 策略
  ├── Greedy（本 notebook）：每步生成 → 验证 → 接受/重试
  ├── Beam Search：维护 N 条候选，PRM 打分后选 topK
  └── MCTS（单独 notebook）：树搜索，PRM 作为价值函数

o1 / R1 的关键：
  SFT 训练模型生成 <think>...</think> 的推理链
  PRM/ORM 验证步骤/最终答案
  RL（GRPO/PPO）强化正确推理路径
```

---

## 面试高频考点

**Q: KTO 为什么不需要成对数据？**
A: DPO 用 Bradley-Terry 建模"A 比 B 好"的相对偏好，必须成对。KTO 用前景理论建模绝对偏好：每条回复独立地被标记为好/坏，通过 KL baseline（z_ref）区分"高于或低于模型均值水平"。适合有大量单标签标注数据的场景（如真实用户反馈）。

**Q: PRM 是怎么训练的？和 ORM 的区别？**
A: PRM（Process RM）在每个步骤的 SEP token 位置预测该步骤正确与否（Positive/Negative）；ORM（Outcome RM）只在最终答案处打分。PRM 提供更细粒度的信用分配，能早期发现推理错误，但需要步骤级标注数据（如 PRM800K 数据集）。

**Q: PRM-Search 如何复用 KV Cache？**
A: 验证通过的步骤 tokens 已经存在 past_key_values 里，下次 forward 只需要传入新生成的 tokens（past_kv 自动覆盖历史）。代码中 TODO 处：`use_cache=True` + 传入 `prm_past_key_values`。未优化版本每次重新 forward 全序列，成本 O(T²)；优化后 O(T_new)。

**Q: KTO loss 中 desirable_weight > undesirable_weight 的理由？**
A: 前景理论：人对损失的感受比同等收益更强烈（损失厌恶系数约 2.25 倍）。在训练中，好样本奖励信号天然弱于坏样本惩罚信号，1.33:1.0 权重部分补偿这种不对称。

**Q: step-wise SFT 的 label 为什么要 roll(-1)？**
A: 因果语言模型（Causal LM）预测"当前 token 的下一个"。如果 input = [t1, t2, t3, t4]，那么 label = [t2, t3, t4, -100]（最后一位没有下一个）。`torch.roll(prompt_label, shifts=-1)` 就是左移一位，实现 next-token prediction 的 label 对齐。

**Q: 为什么 PRM 训练数据要累积前缀？**
A: 因为步骤是否正确取决于上下文（causal）。第 3 步的对错需要第 1/2 步的信息。用 `[prompt + step_1 + ... + step_i]` 作为 input 才能让 PRM 做 context-aware 评判。

**Q: PRM Search 和 MCTS 的关系？**
A: PRM Search 是贪心版（每步接受/拒绝，不回退多步）。MCTS 更完整：建树、UCB 探索、MC rollout 估值、反向传播节点价值。Beam Search 是中间版。R1/o1 推理时使用的主要是 Best-of-N 变体（多条完整 rollout 选最佳），而非严格的逐步 MCTS。

---

## 关联笔记

- `AI/LLM/MA-RLHF课程/lc8-DPO-IPO-BT-偏好优化从零手写.md` — DPO/IPO/BT（同批次）
- `AI/LLM/MA-RLHF课程/lc8-RLHF-PPO-Pytorch从零手写.md` — PPO 完整 pipeline
- `AI/LLM/Inference/Speculative-Decoding-手撕实操.md` — KV Cache 机制详解
- `AI/LLM/RL/Theory/GRPO-Improvement-Panorama-2026.md` — GRPO 改进（KTO 也有 GRPO-KTO 变体）
- `AI/Agent/Agentic-RL/AgentPRM-Process-Reward-Models-for-LLM-Agents.md` — Agent 领域的 PRM

## See Also

- [[AI/3-LLM/RL/KTO/KTO-手撕实操]] — 同算法手撕实操版（MA-RLHF lc8 Batch A）
- [[AI/3-LLM/RL/KTO/KTO-完整Notebook实现]] — 同算法 Notebook 端到端版
- [[AI/3-LLM/RL/PPO/PRM-O1-Search-手撕实操]] — PRM 搜索 Batch A 版
- [[AI/3-LLM/MA-RLHF课程/lc8-DPO-IPO-BT-偏好优化从零手写]] — 同系列：DPO/IPO 实现
- [[AI/3-LLM/MA-RLHF课程/lc8-RLHF-PPO-Pytorch从零手写]] — 同系列：PPO 完整实现
