# lc8 DPO / IPO 手撕实操

**来源**：MA-RLHF 课程 `notebook/DPO/DPO.ipynb`（27 cells，完整）
**评级**：★★★★★
**标签**：#DPO #IPO #偏好对齐 #实现细节 #过拟合分析

---

## 一、DPO 公式回顾

$$\mathcal{L}_{DPO}(\pi;\pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\left[\log \sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**直觉**：最大化 chosen 相对 rejected 的 log-ratio 差，用 KL 惩罚（β）控制偏离 ref_model 的程度。

---

## 二、数据格式

```python
# 输入格式：prompt_chosen 和 prompt_rejected 共享相同 prompt 前缀
prompt_length = 6
answer_length = 4

# chosen:   [prompt(6个token) | response(4个token)]
prompt_chosen   = torch.tensor([[5, 8, 9, 10, 5, 3,   16, 29, 18, 17]])
prompt_rejected = torch.tensor([[5, 8, 9, 10, 5, 3,   26, 14, 31,  0]])
attention_mask  = torch.tensor([[1, 1, 1, 1,  1, 1,    1,  1,  1,  1]])

# label: 只对 response 部分计算 loss（prompt 部分 mask 掉）
label = torch.tensor([[0, 0, 0, 0, 0, 0,   1, 1, 1, 1]])  # bool mask
```

**关键**：label 是 bool mask，不是 class label。prompt 部分不参与 loss 计算。

---

## 三、Token-Level Log Probability

```python
def get_probs(logits, labels):
    # logits: [B, T, V]  labels: [B, T]
    per_token_logps = torch.gather(
        logits.log_softmax(-1),       # 先 softmax → log，得 [B, T, V]
        dim=2,                         # 在 vocab 维度 gather
        index=labels.unsqueeze(2)      # labels 扩 dim → [B, T, 1]
    ).squeeze(2)                       # 结果 [B, T]
    return per_token_logps
```

**逐步理解**：
1. `log_softmax(-1)`：对每个位置的 32个token score 做 log-softmax
2. `labels.unsqueeze(2)`：把标签扩成 [B, T, 1]，用来 index vocab 维
3. `torch.gather`：在 vocab 维度取出对应 ground truth token 的 log prob
4. `squeeze(2)`：去掉多余维度，得到 per-token log prob

**等价直觉**：position i 生成了 token id=18，就去 logits[B, i, 18] 取那个 log prob。

---

## 四、四路 Forward 计算

```python
# 四路计算：ref/model × chosen/rejected
logits_chosen_ref   = ref_model(**x_chosen).logits    # ref on chosen
logits_rejected_ref = ref_model(**x_rejected).logits  # ref on rejected
logits_chosen       = model(**x_chosen).logits         # θ on chosen
logits_rejected     = model(**x_rejected).logits       # θ on rejected

probs_chosen_ref   = get_probs(logits_chosen_ref,   prompt_chosen)
probs_chosen       = get_probs(logits_chosen,        prompt_chosen)
probs_rejected_ref = get_probs(logits_rejected_ref, prompt_rejected)
probs_rejected     = get_probs(logits_rejected,      prompt_rejected)
```

**为什么要 ref_model no_grad**：
- ref_model 冻结，不参与反向传播
- 实际训练中应该 `with torch.no_grad(): ...`

---

## 五、DPO Loss 计算

```python
beta = 0.1

# log ratio：chosen - rejected
pi_logratios  = probs_chosen       - probs_rejected       # [B, T]
ref_logratios = probs_chosen_ref   - probs_rejected_ref   # [B, T]

# DPO logits = π的相对差 - ref的相对差
logits = pi_logratios - ref_logratios  # 等价于 DPO 公式里 β 前面那坨

# per-token loss，只在 response 位置计算（label mask）
losses = -F.logsigmoid(beta * logits) * label  # [B, T]

# 平均到 token 数
loss = losses.sum(-1) / attention_mask.sum()
```

**注意**：这里有一个 token-level 的展开，把 sequence-level 的 log-ratio 分解到 token 级别再求和。

---

## 六、DPO 过拟合问题（IPO 论文核心）

### 问题根源

BT Model（Bradley-Terry）的优化目标是使 reward 差 → +∞：

$$P(y \succ y') = \sigma(r(y) - r(y')) \rightarrow 1 \Leftrightarrow r(y) - r(y') \rightarrow +\infty$$

这等价于让 π(y') → 0（rejected policy 概率变为0）。

**DPO 的 hack 路径**：
- DPO 把 reward = β * log(π/π_ref) 代入 BT
- 最大化 reward 差的一条捷径：让 π_θ(y_l|x) → 0
- KL 惩罚（β）约束随着偏好越来越确定而**越来越弱**
- 即使真实偏好只是 0.8 vs 0.2，有限数据下 empirical preference 可能变成 1.0 vs 0.0
- **结果**：rejected response 的 log prob 急剧下降，模型退化

### 实验验证

训练曲线观察：
- DPO 训练：π(y') 单调下降趋向 0，无论 β 取 0.1 还是 0.5 结果相同
- **β 不影响最终收敛的 π(y')** —— 这是 BT 假设的内在缺陷

---

## 七、IPO：修复 DPO 的过拟合

### IPO 核心 insight

IPO 放弃 BT 假设，直接回归 log-ratio 差到一个目标常数 τ⁻¹/2：

$$h_\pi(y, y', x) = \log\frac{\pi(y|x)}{\pi_{ref}(y|x)} - \log\frac{\pi(y'|x)}{\pi_{ref}(y'|x)}$$

这和 DPO 的 logits 变量完全一样！差异只在 loss function：

$$\mathcal{L}_{IPO} = \mathbb{E}_{(y_w, y_l, x)\sim D}\left(h_\pi(y_w, y_l, x) - \frac{\tau^{-1}}{2}\right)^2$$

**二次目标**（MSE）代替 log-sigmoid，让 log-ratio 差收敛到一个有限目标值，不再退化到 ±∞。

### IPO β 参数的含义

IPO 的 constant = 1/(β*2) = τ⁻¹/2，所以：
- β 越小 → 目标差越大 → 允许更大的偏好差异
- β 越大 → 目标差越小 → 更保守，更接近 ref_model

**β 现在真的有效地控制收敛的 π(y') 值**（DPO 中 β 对最终 π(y') 无效）。

### IPO 实现

```python
def train_XPO(model, beta, loss_type, epochs, lr, optim_type='SGD'):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for i in range(epochs):
        optimizer.zero_grad()
        
        # 复用相同的 logits 计算（与 DPO 完全相同）
        logits, probs_rejected = get_logits(model, ref_model, x_chosen, x_rejected)
        
        constant = 1.0 / (beta * 2.0)  # IPO 目标值 = τ⁻¹/2
        
        if loss_type == 'DPO':
            losses = -F.logsigmoid(beta * logits) * label
        elif loss_type == 'IPO':
            losses = torch.square(logits - constant) * label  # MSE！
        
        loss = losses.sum(-1) / attention_mask.sum()
        loss.backward()
        optimizer.step()
```

**关键差异**：
- DPO：`-F.logsigmoid(β * logits)`（log-sigmoid，无边界约束）
- IPO：`(logits - constant)²`（MSE，明确的收敛目标）

---

## 八、DPO vs IPO 实验对比

```python
# DPO: β 不影响最终 π(y') 收敛值
pi_4 = train_XPO(model, 0.1, 'DPO', epochs, 0.01)  # π(y') → 0
pi_5 = train_XPO(model, 0.5, 'DPO', epochs, 0.01)  # π(y') → 0 (相同！)

# IPO: β 控制收敛目标，π(y') 不再退化为 0
pi_1 = train_XPO(model, 0.1, 'IPO', epochs, 0.0001)  # 较大常数 → 较大 gap
pi_2 = train_XPO(model, 0.5, 'IPO', epochs, 0.0001)  # 较小常数 → 较小 gap
pi_3 = train_XPO(model, 1.0, 'IPO', epochs, 0.0001)  # 最小常数 → 最小 gap
```

**稳定性注意**：
- IPO 用大 lr（0.01）容易训飞（二次目标梯度大）
- IPO 用 ADAM 比 SGD 更不稳定（ADAM 自适应 lr 反而引入波动）
- IPO 推荐 lr = 0.0001，SGD

---

## 九、三层逐步递进理解

```
BT Model：P(y≻y') = σ(r_y - r_y')
           ↓ 问题：reward差可→+∞，policy退化
           
DPO：把 r = β*log(π/π_ref) 代入 BT，去掉显式 reward model
     ↓ 问题：BT 假设继承，π(y')→0 的 hack 路径依然存在
     
IPO：放弃 BT 假设，直接回归 log-ratio 差到有限目标
     目标：h_π(y,y') = τ⁻¹/2（一个常数）
     loss = (h_π - τ⁻¹/2)²
     ✅ β 有效控制收敛；π(y') 不退化为 0
```

---

## 十、面试必备

**Q：DPO 和 RLHF-PPO 相比的优势和劣势？**
- 优势：无需 RM、无需 RL 训练循环、offline 数据、训练稳定
- 劣势：offline 数据分布漂移（chosen/rejected 来自旧 policy）；BT 过拟合退化

**Q：DPO loss 中 β 的作用？为什么叫 KL penalty？**
- β 控制 π_θ 偏离 π_ref 的程度
- DPO 的 optimal policy 推导中，KL 项自然化为 β 的倒数
- 小 β → 允许更大偏离 → 更 aggressive；大 β → 保守

**Q：IPO 怎么解决 DPO 的过拟合？实现上有什么区别？**
- 放弃 BT 假设，改用 MSE 目标
- `torch.square(logits - constant)` vs `-F.logsigmoid(β * logits)`
- β 现在真正控制 π(y') 的收敛值（DPO 中 β 对收敛终态无效）

**Q：手撕 get_probs 中 torch.gather 的逻辑？**
- logits [B,T,V] → log_softmax → [B,T,V]
- labels [B,T] → unsqueeze → [B,T,1]
- gather(dim=2): 在 V 维取 labels 指定的 token index
- 结果 [B,T,1] → squeeze → [B,T] per-token log prob

**Q：DPO 为什么 label 要 mask prompt 部分？**
- 只优化 response 部分的 log-ratio
- Prompt 是 conditioning context，chosen/rejected 共享相同 prompt，不应计入偏好信号

---

## 十一、与其他方法的关系

| 方法 | 数据格式 | Loss | 核心假设 | 过拟合风险 |
|------|---------|------|---------|----------|
| RLHF-PPO | (x, y) + RM | PPO clipped | 无 | 奖励 hacking |
| DPO | (x, y_w, y_l) | log-sigmoid | Bradley-Terry | π(y')→0 退化 |
| IPO | (x, y_w, y_l) | MSE | 无 BT 假设 | 低（有界收敛目标）|
| KTO | (x, y, label) | 前景理论 | KL baseline | 低 |

**iStar（2509.19199）扩展**：把 trajectory-level DPO 在 step 级别展开，等价于对每步学习 implicit step reward，无需额外标注。
