# Muon 优化器手撕实操

> 来源：`ma-rlhf/notebook/muon/Muon.ipynb` + `Muon-Train.ipynb`
> 参考：[苏剑林：Muon优化器赏析——从向量到矩阵的本质跨越](https://kexue.fm/archives/10592)
> 官方实现：[KellerJordan/Muon](https://github.com/KellerJordan/Muon)

---

## 1. Muon 核心：Nesterov Momentum + 矩阵正交化

### 算法公式

$$\begin{aligned}
M_t &= \beta M_{t-1} + G_t \\
W_t &= W_{t-1} - \eta_t [\text{msign}(M_t) + \lambda W_{t-1}]
\end{aligned}$$

三个关键组件：
1. **动量累积**：$M_t = \beta M_{t-1} + G_t$（标准动量）
2. **msign 正交化**：将动量矩阵投影到最近的正交矩阵
3. **权重衰减**：$\lambda W_{t-1}$

### msign 是什么？

$$\text{msign}(M) = M(M^\top M)^{-1/2} = UV^\top$$

其中 $M = U\Sigma V^\top$ 是 SVD 分解。**msign 就是去掉奇异值，只保留左右奇异向量的乘积**。

几何意义：msign(M) 是 **M 的最优正交近似**：

$$\text{msign}(M) = \arg\min_{O^\top O = I} \|M - O\|_F^2$$

### 特殊情况理解

| M 的形状 | msign(M) 等价于 |
|----------|---------------|
| 标量 1×1 | `sign(m)` — 符号函数 |
| 向量 n×1 | `m / ||m||₂` — L2 归一化 |
| 矩阵 n×m | `U @ V^T` — 最优正交近似 |

---

## 2. 为什么矩阵正交化有效？

**核心洞察**：Adam 等逐元素优化器对每个参数独立缩放梯度，忽略了权重矩阵的整体结构。Muon 通过 msign 将更新方向约束在 **steepest descent manifold** 上。

具体地：
- msign 去掉了奇异值（缩放信息），只保留方向信息
- 等价于在**矩阵 2-范数约束下的梯度下降**：更新方向是使 loss 下降最快的单位正交矩阵
- 忽略缩放杂质，仅逼近低秩正交量 $U_{[:,:r]} V_{[:r,:]}^\top$
- 苏剑林总结：**Muon 比 Adam 少一组缓存变量，显存成本更低**

---

## 3. Newton-Schulz 迭代求解 msign

直接做 SVD 复杂度 $O(nm^2)$（对 7168×18432 的矩阵太贵），用 Newton-Schulz 多项式迭代逼近：

### 数学推导

对 $t^{-1/2}$ 做泰勒展开，保留到 2 阶得到五次多项式：

$$X_{t+1} = aX_t + bX_t(X_t^\top X_t) + cX_t(X_t^\top X_t)^2$$

### 系数求解

标准泰勒系数 $(15/8, -5/4, 3/8)$ 可用，但 Muon 官方通过**数值优化**找到更好的系数。方法是：
1. 采样大量随机矩阵，SVD 得到归一化奇异值分布
2. 将迭代参数化为 $g(x) = x + \kappa x(x^2 - x_1^2)(x^2 - x_2^2)$
3. 用 SGD 最小化 MSE Loss 求解 $(\kappa, x_1, x_2)$
4. 展开得到 $(a, b, c)$

Muon 官方系数：**(3.4445, -4.7750, 2.0315)**，5 次迭代即可收敛。

```python
def newton_schulz_msign(M, steps=5):
    """Newton-Schulz 迭代求 msign(M)"""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M
    
    # 归一化：保证迭代收敛
    if X.dim() == 2:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    for _ in range(steps):
        A = X @ X.mT                          # X^T X
        B = b * A + c * A @ A                  # 合并计算减少矩阵乘
        X = a * X + B @ X                      # 五次多项式更新
    return X
```

---

## 4. 完整实现

### 4.1 Muon 优化器

```python
class Muon:
    """
    Muon 优化器：Nesterov Momentum + Newton-Schulz 正交化
    
    适用于：Attention/FFN 中的 2D 权重矩阵
    不适用于：Embedding、LayerNorm/RMSNorm（1D 参数用 Adam）
    """
    def __init__(self, named_parameters, lr=1e-3, momentum=0.9):
        self.named_parameters = named_parameters
        self.lr = lr
        self.momentum = momentum
        self.M = {}  # 动量缓存（只需一组，比 Adam 少一组 V）
        for name, param in self.named_parameters:
            self.M[name] = torch.zeros_like(param.data, dtype=torch.float32)

    def muon_iter(self, M, steps=5):
        """Newton-Schulz 迭代求 msign"""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = M
        if X.dim() == 2:
            X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X
    
    def muon_update(self, G, M, beta=0.95, ns_steps=5, nesterov=True):
        """Nesterov 动量 + msign"""
        M = beta * M + (1 - beta) * G           # 动量累积
        if nesterov:
            M_ = G * (1 - beta) + M * beta       # Nesterov look-ahead
        else:
            M_ = M
        dW = self.muon_iter(M_, steps=ns_steps)   # 正交化
        return dW

    @torch.no_grad()
    def step(self, named_parameters=None, weight_decay=1e-2):
        self.named_parameters = named_parameters
        for name, param in self.named_parameters:
            if name in self.M:
                if param.grad is None:
                    param.grad = torch.zeros_like(param.data)
                G = param.grad
                M = self.M[name]
                dW = self.muon_update(G, M, beta=self.momentum)
                
                # 权重衰减（decoupled）
                param.data = param.data * (1 - self.lr * weight_decay)
                # 参数更新
                param.data = param.data - self.lr * dW

    def zero_grad(self):
        for name, param in self.named_parameters:
            if name in self.M and param.grad is not None:
                param.grad = torch.zeros_like(param.data)
```

### 4.2 Adam 优化器（对照实现）

```python
class Adam:
    """标准 Adam，用于 Embedding/Norm/LM_Head 等非矩阵参数"""
    def __init__(self, named_parameters, lr=1e-3, betas=(0.9, 0.95), epsilon=1e-10):
        self.lr = lr
        self.t = 0
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
        self.M, self.V = {}, {}
        for name, param in named_parameters:
            self.M[name] = torch.zeros_like(param.data, dtype=torch.float32)
            self.V[name] = torch.zeros_like(param.data, dtype=torch.float32)

    @torch.no_grad()
    def step(self, named_parameters=None, weight_decay=1e-2):
        self.t += 1
        for name, param in named_parameters:
            if name in self.M:
                if param.grad is None:
                    param.grad = torch.zeros_like(param.data)
                self.M[name] = self.beta1 * self.M[name] + (1 - self.beta1) * param.grad
                self.V[name] = self.beta2 * self.V[name] + (1 - self.beta2) * param.grad.pow(2)
                m_hat = self.M[name] / (1 - self.beta1 ** self.t)
                v_hat = self.V[name] / (1 - self.beta2 ** self.t)
                param.data -= self.lr * (m_hat / (v_hat.sqrt() + self.epsilon) 
                                         + weight_decay * param.data)

    def zero_grad(self):
        for name, param in self.named_parameters:
            if name in self.M and param.grad is not None:
                param.grad = torch.zeros_like(param.data)
```

### 4.3 混合训练：Muon + Adam

```python
def SplitParam(named_parameters):
    """按参数类型拆分：矩阵权重用 Muon，其他用 Adam"""
    muon_params = ((n, p) for n, p in named_parameters 
                   if 'Attn' in n or 'SwiGLU' in n)
    adam_params = ((n, p) for n, p in named_parameters 
                   if 'Attn' not in n and 'SwiGLU' not in n)
    return muon_params, adam_params

# 初始化
model = LanguageModel(dim=512)
muon_params, adam_params = SplitParam(model.named_parameters())
optimizer_muon = Muon(muon_params, lr=1e-4)
optimizer_adam = Adam(adam_params, lr=1e-4)

# 训练循环
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(epochs):
    for batch in train_dataloader:
        optimizer_muon.zero_grad()
        optimizer_adam.zero_grad()
        
        logits = model(batch['input_ids'])
        bs, seq_len = batch['input_ids'].shape
        loss = loss_fn(logits.view(bs * seq_len, vocab_size), 
                       batch['label'].view(bs * seq_len))
        loss.backward()
        
        # 分别更新
        optimizer_muon.step(named_parameters=muon_params)
        optimizer_adam.step(named_parameters=adam_params)
```

---

## 5. Muon vs Adam 对比

| 维度 | Muon | Adam |
|------|------|------|
| **缓存变量** | 1 组（动量 M） | 2 组（一阶 M + 二阶 V） |
| **显存** | 更低（少一组） | 更高 |
| **更新方向** | 矩阵正交近似（全局结构） | 逐元素自适应缩放 |
| **适用参数** | 2D 权重矩阵（Attention W_q/k/v/o, FFN W_up/down） | 1D 参数（Embedding, Norm gamma, bias） |
| **计算开销** | Newton-Schulz 迭代（5 次矩阵乘） | 逐元素运算 |
| **数学含义** | 矩阵 2-范数约束下的最速下降 | 逐元素 L2 约束下的自适应下降 |

### DeepSeek V3 中 Muon Split 解决 MLA 不兼容问题

DeepSeek-V3 的 MLA 将 Q/K/V 压缩为低维 latent 再上投影，上投影矩阵 $W_{uk}$ 和 $W_{uv}$ 与 latent 强耦合，不适合整体做 msign 正交化；因此 **Muon Split 将 MLA 的压缩投影和上投影矩阵分开处理**：上投影矩阵按行/列分块后分别做 Newton-Schulz，避免破坏 latent 空间的结构。

---

## 6. 面试考点

### 考点 1：msign 的几何意义是什么？为什么比逐元素 sign 更好？

**答**：msign(M) = UV^T 是 M 在 Frobenius 范数下的**最优正交近似**——去掉了奇异值（缩放杂质），只保留方向信息。逐元素 sign 丢弃了矩阵行列之间的关联结构，而 msign 保持了矩阵的全局正交结构。等价于在矩阵 2-范数约束下做最速下降，更新方向是使 loss 下降最快的正交矩阵。

### 考点 2：Newton-Schulz 迭代为什么用 (3.4445, -4.7750, 2.0315) 而不是泰勒系数 (15/8, -5/4, 3/8)？

**答**：泰勒系数是对 $t^{-1/2}$ 在 $t=1$ 处的局部展开，只在 $t \approx 1$ 时精确。但实际矩阵的奇异值分布并不集中在 1 附近。Muon 通过采样大量随机矩阵得到实际奇异值分布，然后用 SGD 优化 $g(x) = x + \kappa x(x^2-x_1^2)(x^2-x_2^2)$ 的参数，使得 5 次迭代后所有奇异值都尽可能接近 1。这是一种**数据驱动的多项式逼近**，比局部泰勒展开在全局上效果更好。

### 考点 3：为什么 Muon 只适合 2D 权重矩阵，不适合 Embedding 和 Norm？

**答**：
- **Embedding** 是查找表，每行独立对应一个 token，行间不存在需要正交化的结构关系。msign 会强制让 embedding 向量正交化，破坏语义空间
- **RMSNorm/LayerNorm 的 gamma** 是 1D 向量，msign 退化为 L2 归一化 `γ/||γ||`，等价于强制 gamma 在单位球上，丧失缩放自由度
- **Bias** 同理是 1D，msign 无意义
- 只有 Attention/FFN 的 2D 权重矩阵具有行列结构，msign 的正交约束才能提供有意义的归纳偏置
