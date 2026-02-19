---
title: "AI 面试速查手册"
date: 2026-02-20
tags: [面试, 速查, AI]
---

# AI 面试速查手册

> 面试前 30 分钟快速过一遍。每题只保留关键词级提示。

---

## 一、Agent 方向

**本方向核心关键词**：ReAct、Plan-and-Execute、MCP、Multi-Agent、Tool Use、Policy Engine、CUA、Agentic Workflow

### Q1: Agent 和 Chatbot 的本质区别？
- Chatbot 是开环 request-response，Agent 是闭环 goal-directed
- Agent = 自主决策 + 多步执行 + 工具调用 + 反馈循环
- Agentic 是光谱而非二元，生产系统通常在中间位置
- 核心判别：环境交互、适应性、失败恢复能力

### Q2: 什么时候不该用 Agent？
- 任务确定性强 → 用 Workflow Engine（Temporal/Airflow）
- 失败不可逆且后果严重 → 不用 Agent 做 primary actor
- SLA < 200ms → Agent loop 延迟不可接受
- 无法定义 "done" → Agent 会陷入无限循环

### Q3: 主流 Agent 架构对比？
- **ReAct**：Thought-Action-Observation 循环，简单可解释，适合 <5 步
- **Plan-and-Execute**：先规划后执行，适合 5+ 步复杂任务
- **Reflexion**：加自我反思循环，从失败中学习，需成功/失败信号
- **LATS**：MCTS 树搜索，探索性强但成本 10-20x
- **Multi-Agent**：专业化分工，debug 复杂度指数增长

### Q4: Agent 记忆系统怎么设计？
- **Working Memory**：Context window + Scratchpad，受 token 限制
- **Episodic Memory**：执行轨迹 + 向量检索，跨 session 复用
- **Semantic Memory**：长期知识库 + 用户偏好，永久存储
- Context 超 80K 后成功率下降，需分层 pruning

### Q5: MCP 是什么？
- Anthropic 推出的 Agent-Tool 连接标准协议
- 解决 N×M 适配问题，类比 REST 之于 Web
- 基于 JSON-RPC 2.0，提供 Tools/Resources/Prompts/Sampling
- 2025.12 捐给 Linux Foundation AAIF，成为行业标准

### Q6: Agent 安全的核心原则？
- 边界必须是 structural 的，不能是 prompting 的
- 四层防护：Input Guard → Policy Engine → Sandbox → Monitoring
- Indirect Prompt Injection 是最大威胁（恶意指令藏在工具返回中）
- 26.1% 社区 Skill 含安全漏洞（Agent Skills Security 论文）

### Q7: Agent 评测关注什么？
- Pass@1 vs Pass@5 看稳定性 vs 能力
- Cost per success 看生产可行性
- Failure mode 分布定位改进方向
- 主流 Benchmark：SWE-bench / WebArena / GAIA / AgentBench

### Q8: 如何防止 Agent 无限循环？
- 硬步数限制（orchestrator 层强制）
- 循环检测（action hash 或 embedding 相似度）
- Token/cost 预算硬上限
- Wall-clock timeout 兜底

### Q9: Multi-Agent 最大的技术挑战？
- Debug 复杂度：交互是网状的，错误传播难追踪
- 自然语言传精确信息是 lossy channel
- 解法：distributed tracing + 结构化中间消息 + hierarchical 架构

### Q10: 2026 Agent 关键趋势？
- MCP 标准化 → Agent-Tool 即插即用
- CUA（Computer-Use Agent）→ 无 API 系统的自动化
- Environment Scaling → 合成环境训练 Agent
- Agent 治理 → RBAC + audit trail + policy engine

---

## 二、RL 对齐方向

**本方向核心关键词**：RLHF、DPO、GRPO、Reward Hacking、KL Penalty、Bradley-Terry、Process RM、DAPO

### Q1: RLHF 三个阶段及挑战？
- **SFT**：预训练→指令跟随，坑在 chat template 和数据质量
- **RM**：学偏好排序，标注者一致性仅 65-75%，length bias
- **PPO**：4 模型并行，训练不稳定，reward hacking，generation 瓶颈

### Q2: DPO 的核心思想？
- 将 RL 问题转化为 supervised loss，绕过 RM 和 PPO
- RLHF 的 KL-constrained 优化有解析解，反解 reward 代入 BT 模型
- Partition function Z(x) 在做差时消掉
- 优势：2 模型、稳定、简单；劣势：offline、无探索、mode collapse

### Q3: DPO vs RLHF 怎么选？
- 数据充足追求简单稳定 → DPO
- 追求最高性能有计算资源 → Online RL（PPO/GRPO）
- 推理/代码有 verifiable reward → GRPO + RLVR
- RL 优于 DPO 的条件：representation 充分 + 优化够充分

### Q4: GRPO 是什么？和 PPO 区别？
- 去掉 Critic，用组内相对 reward 做 advantage
- 对每 prompt 生成 G 个 response，组内标准化：Â=(r-μ)/σ
- 省掉 Critic → 显存减半；特别适合 sparse/binary reward
- DeepSeek-R1 用 GRPO 从 base model 涌现推理能力

### Q5: Reward Hacking 怎么防？
- 常见形式：length exploitation、sycophancy、formatting tricks
- 检测信号：RM score↑ 但 human eval↓
- 防御：KL penalty + RM ensemble + length normalization + 迭代 retrain RM
- Verifiable rewards（数学/代码）不可 hack

### Q6: KTO vs DPO 区别？
- KTO 只需 binary signal（好/坏），不需 pairwise 配对
- 基于 Kahneman-Tversky 前景理论，loss aversion
- 高风险场景（法律/医疗）更好，标注成本更低

### Q7: DAPO 对比 GRPO 改进？
- Decoupled Clipping：正向/反向不同 clip 范围
- Dynamic Sampling：过滤全对/全错的 prompt group
- Token-Level Loss：防止长 response 主导梯度
- Overlong Reward Shaping：超长回答 soft penalty

### Q8: DeepSeek-R1 如何用纯 RL 涌现推理？
- 起点：671B base model，不做 SFT
- GRPO + rule-based verifiable rewards（答案对/错）
- 涌现 CoT、self-reflection、"aha moment"
- RL 不是"教"推理，是"激发"已有的 latent capability

### Q9: Online DPO 为什么比 Offline 好？
- Distribution matching：数据来自当前 policy，无 off-policy 问题
- 持续探索：每轮有新 response
- 自适应难度：policy 变强后对比对更 challenging
- GRPO 本质上是 Online DPO（group size > 2）

### Q10: Mode Collapse 怎么诊断和修复？
- 诊断：response diversity↓、vocabulary 变窄、温度无效
- 修复：降低 β、增大 batch size、数据多样化、entropy bonus
- GRPO 的 group generation 天然增加多样性

### Q11: Alignment Tax 如何最小化？
- 能力-安全分离：先训能力再做安全对齐
- 混合 pretrain 数据到 alignment 训练中（10-20%）
- LoRA 做对齐只改部分参数
- 细粒度 safety policy，不一刀切拒绝

### Q12: Process RM vs Outcome RM？
- ORM：整个 response 一个分数，标注简单但 signal sparse
- PRM：每个 reasoning step 一个分数，dense signal 但标注贵
- PRM + tree search 在数学推理上显著优于 ORM

---

## 三、推理优化方向

**本方向核心关键词**：Prefill/Decode、Flash Attention、PagedAttention、KV Cache、Speculative Decoding、MoE、Continuous Batching、量化

### Q1: LLM 推理为什么慢？
- Prefill = compute-bound（大矩阵乘法），指标 TTFT
- Decode = memory-bound（每 token 加载全部权重），指标 TPOT
- 70B 模型每 token 加载 140GB 需 70ms，计算仅 0.9ms

### Q2: Flash Attention 核心思想？
- Tiling + Online Softmax + Kernel Fusion
- 不存 N×N attention 矩阵，显存 O(N²)→O(N)
- 加速来自减少 HBM 读写，非减少 FLOPS
- v2 优化 GPU 利用率达 70%+；v3 利用 H100 Hopper 特性

### Q3: PagedAttention vs Flash Attention？
- 互补关系，解决不同问题
- FA 优化 attention 计算速度和显存
- PA 优化 KV Cache 内存管理（解决碎片化）
- PA 借鉴 OS 虚拟内存分页，显存浪费从 60-80% 降至 <4%

### Q4: MQA / GQA / MLA 对比？
- **MQA**：所有 Q head 共享 1 组 KV，压缩 n_heads 倍，质量略降
- **GQA**：Q 分 G 组共享 KV，平衡质量和效率
- **MLA**：低秩压缩 latent vector，DeepSeek-V3 压缩 30×+
- MLA 用 absorption trick 避免显式恢复 K

### Q5: Speculative Decoding 为什么不改变输出分布？
- Rejection sampling 数学保证：q(x)≤p(x) 直接接受
- q(x)>p(x) 以 p(x)/q(x) 概率接受，否则从 max(0,p-q) 重采样
- 加速本质：memory-bound 下验证 K 个 token ≈ 生成 1 个
- Medusa 加 head 做树形 draft；EAGLE 在 hidden state 层做 draft

### Q6: Continuous Batching vs Static Batching？
- Static：等所有请求完成才接新请求，GPU 利用率低
- Continuous：每个 decode iteration 级别调度，立即插入新请求
- 吞吐量提升 2-8×，已成所有 serving 框架标配

### Q7: Chunked Prefill 解决什么问题？
- 长 prompt prefill 独占 GPU，导致 decode 请求延迟尖峰
- 将 prefill 分成小块和 decode 请求混合 batching
- P99 TPOT 改善 3-5×，代价是 TTFT 略增

### Q8: MoE 推理的主要挑战？
- 显存大：所有 expert 常驻即使只激活部分
- 负载不均：popular expert 成瓶颈
- 通信开销：EP 下需要 all-to-all 通信
- Batch 碎片化：路由分散后每 expert 有效 batch 小

### Q9: GPTQ vs AWQ 核心区别？
- GPTQ：逐层逐列量化 + Hessian 误差补偿，数学严谨
- AWQ：识别 ~1% 显著权重做 per-channel scaling 再量化，更简单
- AWQ 硬件更友好速度略快，质量相当

### Q10: Serving 框架怎么选？
- **vLLM**：通用首选，模型最全，最稳定
- **SGLang**：Agent/高并发/结构化输出，prefix caching 最强
- **TensorRT-LLM**：NVIDIA 极致单卡性能，闭源
- **llama.cpp**：Mac 本地/边缘设备，跨平台王者

### Q11: PD 分离的动机和挑战？
- 动机：Prefill(compute-bound) 和 Decode(memory-bound) 特征不同
- 分离后各自独立优化和 scaling
- 挑战：KV Cache 迁移带宽、高速互连需求、调度复杂

### Q12: 如何评估 Serving 系统性能？
- 延迟：TTFT / TPOT / E2E Latency / ITL P99
- 吞吐：Throughput(tokens/s) / Goodput(满足 SLA 的有效吞吐)
- 效率：MFU / Cost per token / Tokens per GPU-hour

---

## 四、多模态方向

**本方向核心关键词**：VLM、ViT、SigLIP、Cross-Attention、Early Fusion、Dynamic Resolution、Native Multimodal、Video-LLM、VLA

### Q1: 多模态三种主流架构？
- **Early Fusion**：视觉+文本 token 拼接共享 self-attention，交互充分但 context 开销大
- **Cross-Attention**：视觉作 KV 注入特定层，效率高但需改 LLM 架构
- **Late Fusion**：各模态独立编码后合并，交互最少
- 当前主流是 Early Fusion（LLaVA/InternVL/Qwen-VL）

### Q2: SigLIP 为什么取代 CLIP？
- CLIP 用 softmax InfoNCE 需全局归一化，跨 GPU 通信大
- SigLIP 用 Sigmoid Loss，每 pair 独立二分类，无需全局归一化
- 可用更大 batch size 不增通信开销，性能全面超 CLIP
- SigLIP2 加入多任务训练 + 区域感知 + 多分辨率

### Q3: Dynamic Resolution 为什么重要？
- 固定分辨率 resize 损害 OCR 能力，破坏宽高比
- 按原始比例分割为 tiles 独立编码 + 低分辨率 thumbnail
- InternVL：Dynamic Resolution + Pixel Unshuffle（token 降 1/4）
- Pixtral：原生分辨率处理，完全不 resize

### Q4: 多模态训练三阶段？
- **Alignment**：冻结 ViT+LLM，只训 Projector，~1M pairs
- **Instruction Tuning**：解冻 LLM 全参数微调，~1-5M 指令
- **Preference Optimization**：DPO/RLHF 减少幻觉提升质量
- 趋势：Native Multimodal 从头联合训练所有模态

### Q5: 多模态幻觉有哪些类型？
- 物体幻觉、属性幻觉、关系幻觉、数量幻觉、OCR 幻觉
- 缓解：DPO 针对幻觉偏好优化 + 提升分辨率 + visual grounding
- 产品层：始终显示 citation 让用户验证

### Q6: 视频理解 vs 图像理解的核心挑战？
- Token 爆炸：1 分钟 30fps = 数十万 tokens
- 时序建模：动作识别/事件因果/状态变化需跨帧推理
- 方案：帧采样 + ViT / Temporal Encoder(STORM) / 超长上下文

### Q7: 原生多模态 vs 模块化多模态？
- 模块化 = 先独立学再对齐，有"桥接瓶颈"
- 原生 = 从头联合训练所有模态 token，无瓶颈
- 模块化：训练成本低、灵活；原生：天花板更高

### Q8: MoE 在多模态中的应用？
- 效率层面：Qwen3-VL 235B 总参仅 22B 激活
- 多模态层面：不同模态可路由到不同专家网络
- 跨模态 token 同时激活多个专家实现融合推理

### Q9: GPT-4o 原生音频 vs 传统 ASR+TTS pipeline？
- Pipeline：延迟高（3-5 模块串联）、丢失语调/情感、错误累积
- Native：端到端 sub-200ms、保留副语言信息、支持打断
- Audio tokens（EnCodec/SoundStream）和 text tokens 统一处理

### Q10: VLA 模型是什么？
- VLM + Action 输出 = Vision-Language-Action
- 输入图像+指令，输出机器人控制指令
- 代表：RT-2 / Octo / OpenVLA
- 多模态走向具身智能的关键一步

### Q11: Pixel Unshuffle 是什么？
- 空间→通道转换：H×W×C → H/2×W/2×4C
- 目的：减少视觉 token 数量到 1/4
- 无损信息变换，配合 MLP Projector 保留关键信息
- InternVL 系列的核心压缩技术

### Q12: 2026 多模态前沿？
- Native Multimodal：消除桥接瓶颈
- Any-to-Any：任意模态组合输入输出
- World Model：从描述到物理理解和未来预测
- Embodied Intelligence：VLM→VLA，机器人大脑

---

## 五、RAG 方向

**本方向核心关键词**：Hybrid Search、Reranking、Chunking、RAGAS、Self-RAG、Graph RAG、HyDE、Embedding、向量数据库

### Q1: 长上下文为什么没杀死 RAG？
- 成本：1M tokens/请求 vs RAG 只检索 2-8K tokens，差 100-500 倍
- 准确度：Lost in the Middle 效应，100K+ 仍存在
- 实时性：RAG 天然支持增量索引
- 权限：检索阶段可做 fine-grained access control
- 可解释：天然提供 citation

### Q2: RAG 四代演进？
- **Naive RAG**：单次检索→拼接→生成，无迭代无排序
- **Advanced RAG**：Query Rewrite + Hybrid Search + Reranking
- **Modular RAG**：可插拔模块，标准化接口，自由组合
- **Agentic RAG**：动态规划检索策略，多轮自适应，Self-RAG/CRAG

### Q3: Dense vs Sparse Retrieval？
- Dense：语义理解强（同义词、隐含语义），弱精确匹配
- Sparse（BM25）：精确关键词匹配强，可解释，零样本泛化好
- 生产首选：Hybrid + RRF 融合

### Q4: Reranking 为什么重要？
- Bi-Encoder 是粗排，无 Query-Doc 交互
- Cross-Encoder 精排：拼接编码全层交互，Hit Rate@5 提升 5-15%
- 生产方案：Bi-Encoder 召回 Top-50/100 → Cross-Encoder 精排 Top-5/10

### Q5: Chunking 策略怎么选？
- 固定长度切分：简单快速但不尊重语义边界
- Recursive Splitting：按段落→句子递归切分，生产最常用
- Semantic Chunking：基于相邻句子 embedding 相似度，精度最高但慢
- Parent-Child：小 chunk 检索，大 chunk 送生成，兼顾精度和上下文

### Q6: Self-RAG 核心创新？
- LLM 输出 Reflection Tokens 自我评估
- [Retrieve]：是否需要检索；[IsRel]：结果是否相关
- [IsSup]：生成是否被 context 支持；[IsUse]：回答是否有用
- Agentic RAG 的理论基础

### Q7: Graph RAG 解决什么问题？
- Multi-hop 推理（A→B→C 多跳关系查询）
- 全局摘要（向量检索只找局部相关 chunk）
- 关系推理（实体间连接）
- 构建成本高，只在确需关系推理时引入

### Q8: RAG 和 Fine-tuning 怎么选？
- RAG 解决知识更新/扩展（外部动态信息）
- Fine-tuning 解决行为调整/格式定制（风格/推理链）
- 最佳实践：RAG + Fine-tuning 结合

### Q9: 向量数据库怎么选？
- <100K → Chroma/LanceDB（嵌入式）
- 100K-10M → pgvector（已有 PG）/ Qdrant（新建）
- 10M-1B → Qdrant / Milvus
- \>1B → Milvus 分布式

### Q10: RAG 幻觉怎么防？
- 检索层：Hybrid Search + Reranking 确保高质量 context
- Prompt 层：明确要求"只基于 context 回答"
- 生成层：Self-RAG/CRAG 自评机制
- 后处理层：NLI 模型验证 answer-context 一致性
- 产品层：始终显示 citation/source

### Q11: HyDE 原理？
- 让 LLM 先生成假设性答案文档
- 用假设文档 embedding 做检索
- 假设文档和真实文档语义分布更接近
- 弥合 Query 和文档的语义鸿沟

### Q12: RAGAS 核心指标？
- **Faithfulness**：回答是否忠实于 context（不需 GT）
- **Answer Relevancy**：回答是否与 Query 相关（不需 GT）
- **Context Precision**：相关文档是否排在前面（需 GT）
- **Context Recall**：GT 信息是否被 context 覆盖（需 GT）

---

## 六、Transformer 架构方向

**本方向核心关键词**：Self-Attention、RoPE、GQA/MLA、Pre-Norm、SwiGLU、MoE、Mamba/SSM、Flash Attention、Ring Attention

### Q1: Self-Attention 为什么除以 √d_k？
- Q·K 点积方差约为 d_k，过大会把 softmax 推入饱和区
- 除以 √d_k 归一化方差到 1，保持梯度有效
- 本质是 temperature scaling，T=√d_k

### Q2: RoPE 为什么优于 Learned PE？
- 自然编码相对位置（旋转操作使 q·k 只依赖 m-n）
- 配合 YaRN 可从 4K 扩展到 128K+
- 无额外参数，不增加 embedding 表大小

### Q3: Pre-Norm vs Post-Norm？
- Pre-Norm：残差路径无非线性，梯度直接回传，训练稳定
- Post-Norm：梯度需穿过 LayerNorm，深层不稳定需 warmup
- 2026 共识：Pre-RMSNorm 是绝对主流

### Q4: SwiGLU 为什么比 ReLU 好？
- 门控机制：两路投影一路做 Swish 门控，选择性过滤
- Swish 平滑非单调，梯度传播更好
- 相同参数预算下 PPL 一致优于 ReLU/GELU
- d_ff 调为 8/3·d 补偿第三个矩阵

### Q5: MoE 负载不均衡怎么解决？
- Auxiliary loss：$L=αN·Σf_i·P_i$，鼓励均匀分配
- Expert Choice：expert 选 token 而非 token 选 expert
- DeepSeek-V3：Bias-based，无 auxiliary loss 干扰
- Shared Expert 保证通用知识基线

### Q6: Mamba vs Transformer 的优劣？
- Mamba 优势：推理 O(1)/step、训练 O(n)、固定隐状态
- Mamba 劣势：精确检索弱、ICL 稍弱、生态不成熟
- 混合架构是工程最优解（如 Jamba 3:1 Mamba:Attention）

### Q7: Flash Attention 改变了计算复杂度吗？
- 不变，仍 O(n²d)，优化的是 IO 复杂度
- 分块在 SRAM 计算 + online softmax + 不存 N×N 矩阵
- 显存 O(n²)→O(n)，速度 2-4×

### Q8: Ring Attention 怎么工作？
- 序列均分到 P 个设备，Q 不动，KV 环形传递
- 每步用当前 KV 块计算 partial attention
- 通信和计算可 overlap，理论支持无限长度

### Q9: DeepSeek-V3 MLA 的 absorption trick？
- KV Cache 只存压缩后的 c_kv
- 将 W_UK 吸收进 Q 投影矩阵
- 直接在压缩空间做 attention，不需展开 K
- 额外 decoupled RoPE keys 解决 RoPE 兼容问题

### Q10: YaRN vs Position Interpolation？
- PI：所有频率维度统一缩放，高频被不必要压缩
- YaRN：NTK-by-parts 分频段处理 + attention temperature
- 高频几乎不动（局部关系），低频大幅缩放（远程关系）
- YaRN 微调数据少、steps 少（400-600）效果更好

### Q11: 2026 架构设计共识？
- 归一化：Pre-RMSNorm
- 激活：SwiGLU (d_ff=8/3·d)
- 位置编码：RoPE + YaRN
- KV 效率：GQA 或 MLA
- 稀疏化：MoE（细粒度 + shared expert）
- 混合架构：SSM-Attention hybrid

### Q12: 为什么混合架构而非纯 SSM 取代 Transformer？
- SSM 精确检索瓶颈（needle in haystack 弱）
- ICL 能力依赖对示例的精确记忆
- FlashAttention 已将 Attention 效率推很高
- 少量 Attention 层兜底检索，大部分 SSM 层省内存

---

## 七、预训练与分布式训练方向

**本方向核心关键词**：数据清洗、去重、Scaling Law、TP/PP/DP、ZeRO/FSDP、BF16、Loss Spike、Curriculum Learning、MoE 训练

### Q1: 预训练数据去重为什么重要？
- 重复导致过拟合、训练效率下降、隐私泄露风险
- MinHash + LSH 近似去重，期望 O(n) 复杂度
- Common Crawl 去重可去 30-50%，加子串去重 60%+

### Q2: CLM vs MLM 为什么选 CLM？
- CLM 训练效率高（100% tokens 参与 loss vs MLM 仅 15%）
- 天然支持自回归生成
- Scaling Laws 在 CLM 上研究更充分
- GPT-3 证明 CLM + few-shot 可替代 MLM + fine-tuning

### Q3: 张量并行的通信分析？
- 每 Transformer 层 forward 2 次 AllReduce（Attention+MLP）
- Backward 同样 2 次，共 4 次/层
- 通信频繁但数据量中等，必须放机内（NVLink）
- TP degree 通常 2/4/8，须整除 num_heads

### Q4: ZeRO-1/2/3 分别切什么？
- ZeRO-1：切优化器状态，通信=DDP
- ZeRO-2：+切梯度，通信=DDP
- ZeRO-3：+切参数，通信≈1.5×DDP（forward 多 AllGather）
- 显存不紧张时 ZeRO-2 throughput 更高

### Q5: 流水线并行的气泡问题？
- 气泡比 = (p-1)/(m+p-1)，p=stages，m=micro-batches
- 增大 m 减小气泡但增加显存
- 1F1B schedule 减少峰值显存
- Zero Bubble PP 拆 backward 为 B+W 接近零气泡

### Q6: BF16 vs FP16？
- BF16：8 位指数同 FP32，范围大，不需 loss scaling
- FP16：5 位指数，范围仅 ±65504，需 GradScaler 防溢出
- 2026 共识：训练用 BF16，关键操作（loss/softmax）用 FP32
- 深度学习对范围比精度更敏感

### Q7: Loss Spike 怎么处理？
- 频繁保存 checkpoint（每 100-500 steps）
- 监控 gradient norm 预警
- 发生后：回滚 checkpoint + 跳过问题数据 + 降低 LR
- 预防：BF16 + gradient clipping=1.0 + 数据质量审计

### Q8: 数据配比怎么定？
- 典型：Web 67% + Code 8-15% + Wiki 4-5% + Books 5-8%
- 代码比例是关键杠杆（提升推理能力）
- DoReMi：用 proxy model + DRO 自动找最优配比
- 高质量数据可重复 2-4 epochs

### Q9: Chinchilla Law 为什么被"违反"？
- Chinchilla 只优化训练成本，没考虑推理成本
- 推理成本远大于训练成本（服务百万用户）
- LLaMA-7B 用 10× Chinchilla-optimal 数据：训练贵但推理便宜
- Inference-aware scaling law = total cost 优化

### Q10: 混合并行怎么设计？（128×H100 训 70B）
- TP=8 机内（NVLink），PP=2 跨机，DP=8 剩余
- DP 维度用 ZeRO-2 节省显存
- Activation Checkpointing 每 2 层 checkpoint
- Micro-batch 16-32 控制 PP 气泡 <5%

### Q11: MoE 负载均衡为什么重要？
- 热点 expert 成计算瓶颈，冷门 expert 退化→恶性循环
- Auxiliary loss: $L=αN·Σf_i·P_i$
- DeepSeek-V3：bias-based 无 auxiliary loss 干扰
- Shared expert 保证基线输出

### Q12: SFT 为什么容易过拟合？怎么缓解？
- 数据量小（几千~几万）vs 模型参数巨大
- 缓解：epochs 2-5 / LR 1e-5~5e-5 / NEFTune 正则化
- LoRA 减少可训练参数 / 早停基于 eval metric
- Loss masking：只对 assistant 回复计算 loss

### Q13: FIM (Fill-in-the-Middle) 是什么？
- 代码补全专用目标：给前缀+后缀，预测中间内容
- 50% FIM rate，不降低左到右生成能力（"免费午餐"）
- SPM 模式：suffix 放前面作为提示

---

## 高频考点 TOP 10

| # | 考点 | 方向 | 核心一句话 |
|---|------|------|-----------|
| 1 | DPO vs RLHF | RL对齐 | DPO 简单稳定但 offline；RLHF 上限更高但工程复杂 |
| 2 | Flash Attention | 推理/架构 | 不改计算复杂度，优化 IO；tiling+online softmax+kernel fusion |
| 3 | RAG vs 长上下文 | RAG | 互补不竞争；RAG 省成本+实时+权限+citation |
| 4 | MoE 架构 | 架构/推理 | 总参数大但激活少；显存换计算；负载均衡是核心挑战 |
| 5 | Agent 安全 | Agent | 边界必须 structural 非 prompting；四层防护模型 |
| 6 | KV Cache 优化 | 推理/架构 | GQA 减 head → MLA 低秩压缩；PagedAttention 解决碎片化 |
| 7 | 多模态训练三阶段 | 多模态 | Alignment→Instruction Tuning→Preference Optimization |
| 8 | GRPO | RL对齐 | 去 Critic 用组内相对 reward；DeepSeek-R1 涌现推理能力 |
| 9 | 分布式并行策略 | 预训练 | TP 机内高频通信；PP 跨机低通信量；DP 机间大梯度同步 |
| 10 | Speculative Decoding | 推理 | Rejection sampling 保证分布等价；验证比生成便宜 |
