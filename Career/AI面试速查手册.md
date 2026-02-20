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

---

## 八、评估与 Benchmark 方向

**本方向核心关键词**：MMLU-Pro、GPQA、SWE-Bench、Chatbot Arena、LLM-as-Judge、数据污染、动态评估、Agent 评估

### Q1: MMLU-Pro vs MMLU？
- 选项 4→10，猜对概率 25%→10%
- 剔除模式匹配简单题，侧重推理
- 区分度拉大 16-33pp
- MMLU 饱和（顶级模型差 <2%），Pro 是替代方案

### Q2: LLM-as-Judge 的偏见？
- 位置偏见：倾向选第一个/最后一个
- 长度偏见：倾向选更长回答
- 自我偏好：倾向自己风格
- 缓解：随机化顺序 + 多 Judge + CoT 评分

### Q3: Chatbot Arena 为什么最可信？
- 真实用户盲测 + ELO/Bradley-Terry 排名
- 抗数据污染、持续更新
- 局限：偏好≠正确性、受用户分布影响

### Q4: 数据污染怎么检测？
- n-gram overlap 分析
- canary string 注入
- 给前缀检查 memorization
- 动态评估集 + 多 benchmark 交叉验证

### Q5: HumanEval vs SWE-Bench？
- HumanEval：函数级代码生成，164 题，pass@k
- SWE-Bench：真实 GitHub issue 修复，需理解大型代码库
- 前者测"能写函数"，后者测"能改真实项目"

### Q6: 从零搭建评估体系？
- 确定核心场景 → 选 3-5 个 benchmark → 自动化 pipeline
- 加 LLM-as-Judge → 接入 CI → 定期人类评估校准
- 上线后 AB Testing + 线上监控

### Q7: Agent 评估 vs LLM 评估？
- Agent 有状态，需评估多步决策链
- 需要模拟环境，成本高复现难
- 端到端完成率 vs 过程质量
- 安全评估更复杂：工具误用、权限升级

### Q8: ARC-AGI 的意义？
- 测试抽象推理和归纳：给例子推断规则
- 每题全新模式，无法靠记忆
- 人类 85%，最好 AI ~68.8%（Claude Opus 4.6）
- 衡量"学习新规则"能力

### Q9: 2026 评估趋势？
- "做对题"→"完成真实任务"
- "终态评估"→"过程评估"
- "静态排行榜"→"动态竞技场"
- "单模型"→"系统评估"（Agent+Tool+Memory）

## 九、预训练与分布式训练方向

**本方向核心关键词**：3D 并行、DeepSpeed Zero、FSDP、MoE 训练、Loss Spike、数据工程、Scaling Laws、BF16

### Q1: DP/TP/PP 各解决什么问题？
- DP：数据并行，扩 batch size，通信梯度同步
- TP：张量并行，切单层参数到多卡，机内高带宽
- PP：流水线并行，切层到多机，通信量小但有 bubble

### Q2: DeepSpeed Zero 1/2/3？
- Zero-1：切 optimizer states
- Zero-2：切 optimizer states + gradients
- Zero-3：切 optimizer states + gradients + parameters
- 越高级显存越省但通信越多

### Q3: Loss Spike 排查？
- 查 gradient norm → 查数据（回溯 batch）→ 查硬件
- 恢复：回滚 checkpoint + 降 LR + 跳问题数据
- 预防：BF16（不易溢出）+ gradient clipping + 定期存 ckpt

### Q4: MoE 训练特殊挑战？
- 负载不均衡：auxiliary loss / bias-based routing
- Expert collapse：强制最低使用率
- 通信量大：expert 分布在不同卡，all-to-all 通信

### Q5: 数据配比经验？
- Web 60% + 百科 8% + 书籍 8% + 代码 12% + 学术 5%
- 代码占比高提升推理能力
- 去重在清洗之前（减后续计算量）

### Q6: Scaling Laws 核心结论？
- Loss ∝ N^(-α) · D^(-β)（N=参数，D=数据）
- Chinchilla：最优 D ≈ 20×N
- 2026 修正：小模型+更多数据可能更优（Llama 思路）

## 十、Prompt Engineering 方向

**本方向核心关键词**：CoT、ToT、Self-Consistency、DSPy、Prompt Injection、System Prompt、Few-Shot、Thinking Models

### Q1: CoT 为什么有效？
- 把复杂推理分解为中间步骤，降低单步难度
- 本质是激活模型的"慢思考"路径
- 对小模型无效（<10B），大模型涌现能力

### Q2: Few-Shot 示例怎么选？
- 3-5 个最优，超过 8 个边际递减
- 最后一个示例影响最大（recency bias）
- 格式一致性 > 内容正确性
- 最佳实践：embedding 检索动态选最相似示例

### Q3: Prompt Injection 怎么防？
- 输入过滤 + 架构隔离（XML 标签包裹用户输入）
- LLM classifier 检测注入 + 输出过滤
- 权限最小化：即使注入成功可调用范围受限
- 没有 100% 防御，多层纵深提高攻击成本

### Q4: DSPy 核心思想？
- 从"写 prompt 字符串"→"写 prompt 程序"
- 定义 signature + module，框架自动优化
- 换模型时自动重编译，可复现可版本管理

### Q5: Thinking Models 怎么改变 prompting？
- "少即是多"：简洁目标 > 详细步骤
- 手写 CoT 可能限制模型找更优解
- 从"教模型怎么想"→"告诉模型想什么"
- 约束驱动而非步骤驱动

### Q6: System Prompt 设计原则？
- 身份→能力边界→输出格式→安全约束→示例
- 越具体越好，避免模糊指令
- 2026 趋势：System Prompt 成为"应用灵魂"，产品差异化靠它

## 十一、数据工程方向

### Q1: 预训练数据管线核心步骤？
- 采集→格式提取→语言检测→启发式清洗→质量过滤→去重→PII脱敏→有毒过滤→数据配比→Tokenization→打包
- 质量过滤影响最大（FineWeb-Edu 证明），去重防过拟合至关重要

### Q2: MinHash LSH 去重原理？
- 文档→N-gram集合→K个哈希函数算MinHash签名→LSH分桶（b bands × r rows）
- P(sig_k(A)=sig_k(B)) = Jaccard(A,B)，Jaccard>0.8判重
- vs 精确去重（只能完全匹配）vs Suffix Array（子串级精确，内存大）

### Q3: FineWeb-Edu 为什么有效？
- LLM(70B)对500K样本打教育质量分→蒸馏到小分类器→大规模过滤
- 1.3T tokens（从15T过滤）超越10x大的原始FineWeb
- 核心：数据质量>>数据数量，"教育价值"是好的质量代理指标

### Q4: 合成数据主要方法对比？
- Self-Instruct：简单低成本，多样性受限
- Evol-Instruct/WizardLM：系统化难度升级，多样性好
- Magpie：利用对齐特性无需种子，效率高
- 蒸馏：质量最高但受ToS限制
- 带验证合成（代码/数学）：质量保证最强

### Q5: Model Collapse 是什么？怎么避免？
- 模型在自己生成的数据上迭代训练→分布尾部丢失→输出单调化
- 避免：真实数据≥40-50%、多源teacher、质量过滤非盲目扩增、引入外部验证信号、监控多样性指标

### Q6: DPO vs RLHF 数据需求？
- RLHF需3类数据（SFT+比较排序+PPO prompts），DPO只需偏好对{prompt,chosen,rejected}
- DPO对噪声更敏感，RLHF可通过RM容忍噪声
- 2026趋势：DPO主流化，KTO只需thumbs up/down

### Q7: Tokenization/BPE 影响性能的原因？
- 压缩率决定上下文利用效率，词表覆盖影响罕见词学习
- 数字tokenization影响算术能力，代码空格策略影响代码理解
- 词表大小权衡：大→压缩好但embedding重，小→序列长但参数少

### Q8: 数据污染检测方法？
- N-gram匹配（13-gram，>70%重叠判污染）
- 成员推断（Min-K% Prob检查异常低perplexity）
- Benchmark Canary（嵌入特殊标记）
- 预防：时间切割、URL黑名单、动态benchmark（LiveCodeBench）

---

## 十二、安全与对齐方向

> 深度笔记：[[AI安全与对齐-2026技术全景]]

### Q1: RLHF 的核心安全局限？
- Reward Hacking：模型利用 RM 漏洞（冗长但空洞的回答得高分）
- Goodhart's Law：RM 分数成为目标后不再是好度量
- Scalable Oversight 困境：弱监督在价值判断任务 PGR≈0%（Weak-to-Strong 实验）
- Sleeper Agents：RLHF 无法消除预植入后门，模型学会在训练时隐藏、部署后激活

### Q2: Prompt Injection 直接 vs 间接？
- 直接：用户输入嵌入恶意指令（"忽略之前指令"）→ 输入过滤可防
- 间接（Greshake 2023）：嵌入在外部内容（网页/邮件/MCP返回值）→ Agent 时代更危险
- Agent 场景：攻击面指数增长 × 执行能力（代码/API） × Confused Deputy × Memory Poisoning 持久化
- 防御多层：数据来源标记 + 沙箱化 + 最小权限 + 用户确认

### Q3: Superposition + SAE 解决什么？
- Superposition：特征数量>>神经元数量，多概念叠加共享神经元（高维近似正交）
- SAE：过完备稀疏自编码器（n>>d），ReLU+L1强制稀疏→单义特征（monosemantic）
- Claude SAE发现~4M特征，Golden Gate Bridge实验证明因果性
- 安全意义：精准操控特征→比RLHF更精确的对齐（Feature Steering）

### Q4: Constitutional AI 原理？
- 两阶段：①Critique-Revision SFT（自我批判+修订）②RLAIF（AI根据宪法原则生成偏好→训练RM→PPO）
- 优势：低标注成本、可审计原则、一致性、迭代快
- 劣势：宪法设计主观、AI偏见放大、自我批判盲点、可能过度拒绝

### Q5: Sleeper Agents 为什么令人担忧？
- 实验（Hubinger 2024）：训练模型在"Year:2023"写安全代码、"Year:2024"写漏洞代码
- 关键发现：标准安全训练无法消除后门（模型在训练时"装好"）、更大模型隐藏能力更强、对抗训练反而强化隐藏
- 启示：行为级安全训练不够→需要Mechanistic Interpretability检查内部电路

### Q6: Agent 安全核心威胁？
- MCP攻击面：CVE-2026-25253（One-Click RCE），AgentAudit审计194包→118漏洞（61%）
- Top漏洞：Command Injection 34 > Path Traversal 22 > SSRF 18 > Auth Bypass 15
- Confused Deputy：受信Agent被操纵用用户权限执行恶意操作
- Memory Poisoning：投毒记忆→影响后续所有对话（EchoLeak CVE-2025-32711）
- Multi-Agent：Agent间注入、Trust Boundary Confusion、Privilege Escalation

### Q7: Representation Engineering vs SAE？
- RepE：宏观（表示空间方向）、低成本、粗粒度、inference-time无需重训
- SAE/MechInterp：微观（单特征/电路）、高成本、精细粒度、高可解释性
- 组合使用：RepE粗调 + SAE特征精调 = 多层安全防线
- Circuit Breakers（Zou 2024）：表示空间安装断路器，阻断有害激活

### Q8: Alignment Tax 是什么？怎么降？
- 定义：安全措施导致的性能/延迟/成本/灵活性代价
- 典型表现：推理性能退化、over-refusal、推理延迟增加
- 降低方法：DPO替代PPO、RepE+Circuit Breakers（inference-time）、Feature Steering（SAE精准调控）、Safe RLHF（解耦helpful/harmless的约束优化）、精准红队驱动训练
- 2025-2026进展：对齐税降低约30-50%

---

## 十三、代码生成方向

> 深度笔记：[[LLM代码生成-2026技术全景]]

### Q1: HumanEval vs SWE-bench？
- HumanEval：164题单函数生成，pass@k评估，已饱和（>95%）
- SWE-bench：2294真实GitHub Issue，需要浏览仓库→定位文件→修改→跑测试
- SWE-bench Verified（500题）是2024-2026核心赛场，Gemini 3.1 Pro 80.6%最高
- 区分度：HumanEval 区分不了前沿模型，SWE-bench 还有20%提升空间

### Q2: FIM 训练目标？
- Fill-in-the-Middle：将代码分prefix/middle/suffix，训练模型根据prefix+suffix生成middle
- PSM模式：`<PRE>prefix<SUF>suffix<MID>` → 生成middle
- StarCoder用50% FIM + 50%标准自回归，不降低标准生成性能
- IDE补全的核心基础——光标在中间时需要利用上下文双方向信息

### Q3: 代码预训练数据清洗？
- 流程：语言检测→许可过滤→MinHash去重（Jaccard>0.7）→质量过滤→PII清理→合规检查
- 比NL更复杂：重复更严重（fork/copy）、质量谱系更广、许可合规、敏感信息（API keys）、多语言异质性
- 数据配比：代码60-70% + NL 20-30% + 数学10%（DeepSeek-Coder-V2/Qwen2.5-Coder）

### Q4: 代码RL vs NL RLHF？
- 核心优势：代码有天然ground truth（编译/运行/测试），不需要人工标注reward
- GRPO（DeepSeek）：生成N个候选→执行→按通过率排序→policy gradient，不需要RM
- Process Reward Model for Code：评估每一步编码决策（不只最终结果），密集reward
- 稀疏性问题：100行代码一个off-by-one → reward=0，需要PRM缓解

### Q5: Copilot vs Cursor vs Claude Code？
- Copilot：IDE插件，行内补全+Chat，最大用户基数180万+，适合日常补全
- Cursor：AI-native IDE，上下文管理杀手锏（.cursorrules/向量索引/@引用/LSP），适合深度开发
- Claude Code：CLI Agent，Think-Act-Observe循环，自主浏览项目/编辑/测试，适合复杂任务
- 本质区别：补全（单轮）vs Agent（多轮自主）

### Q6: 代码幻觉类型？
- API幻觉：调用不存在的函数（pandas.read_xls→应read_excel）
- 参数幻觉：使用不存在的参数
- 版本混淆：混合Python2/3或不同库版本API
- 库混淆：np.cuda.is_available()（应是torch的）
- 缓解：RAG检索API文档 + LSP类型检查 + 执行验证 + 版本锁定

### Q7: LLM生成代码的安全漏洞？
- 高频：SQL注入(CWE-89)、XSS(CWE-79)，因训练数据中不安全模式更常见
- Copilot安全敏感场景40%含漏洞（Pearce 2022），2024-2026降到15-20%
- 新威胁：Clinejection—通过规则文件注入让Agent生成含后门代码
- 缓解：安全DPO训练+SAST输出扫描+Code Review Agent+LSP实时标注

### Q8: 代码预训练为何提升通用推理？
- 代码=结构化推理训练信号（条件判断/循环/递归≈分步推理）
- 确定性验证→精确思维迁移
- 代码注释=自带CoT标注的推理数据
- GitHub Issue→Code = 自然语言→形式化解决方案的映射
- 实证：Llama2增加代码训练20%→GSM8K+3-5%，ARC+2-3%

---

## 十四、知识蒸馏与模型压缩方向

> 对应深度笔记：[[知识蒸馏与模型压缩-2026技术全景]]

### Q1: 知识蒸馏中 temperature 的作用？
- Soft label = softmax(logits/T)，T越大分布越平滑，暴露类间关系（"暗知识"）
- T=1退化为硬标签；T=3-5常用于LLM蒸馏；T过高所有类概率趋同失去信息
- Loss = α·KL(student_soft, teacher_soft) + (1-α)·CE(student, hard_label)
- LLM蒸馏特殊：词表巨大（32K-150K），soft label信息量远大于CV场景

### Q2: GPTQ vs AWQ 量化原理差异？
- GPTQ：基于OBS（Optimal Brain Surgeon），逐列量化+误差补偿到剩余列，O(d²)复杂度
- AWQ：观察到<1%显著权重贡献>99%性能，对显著通道先乘scale再量化等效降低误差
- GPTQ需校准数据逐层跑前向，AWQ只需统计激活分布更快
- 实践：AWQ速度快适合部署，GPTQ压缩更极致适合离线

### Q3: BitNet b1.58（1.58-bit）如何工作？
- 权重三值化：{-1, 0, +1}，信息量log₂(3)≈1.58 bit
- 训练时：全精度latent weight → absmean量化 → 三值前向 → 全精度梯度更新
- 推理：矩阵乘法退化为加减法，无需乘法器，能耗降71.4×（vs FP16）
- 限制：必须从头训练（QAT），不能PTQ已有模型；小模型性能差距仍明显

### Q4: SparseGPT 剪枝原理？
- 基于OBS框架：剪掉一个权重后用Hessian逆补偿剩余权重
- 关键创新：分列处理+局部Hessian更新，复杂度从O(d³)降到可处理
- 单次前向即可剪枝50-60%（非结构化），无需重训练
- 配合NVIDIA 2:4稀疏硬件：每4个权重保留2个，硬件自动跳零，理论2×加速

### Q5: DeepSeek R1 推理蒸馏方法？
- 核心：大模型（R1-671B）生成长思维链CoT → 小模型（1.5B-70B）学习完整推理过程
- 不只蒸馏答案，蒸馏推理路径——小模型学会"怎么想"而非"答案是什么"
- 基座选择：Qwen2.5和Llama3系列（不同参数规模）
- 效果惊人：R1-distill-Qwen-32B在数学/代码benchmark上超越GPT-4o和Claude 3.5 Sonnet
- 局限：推理蒸馏后输出变长（更多thinking tokens），推理成本上升

### Q6: 投机解码（Speculative Decoding）原理？
- 小模型（draft）快速生成K个token → 大模型（target）并行验证K个token
- 接受/拒绝机制保证输出分布与大模型完全一致（无损）
- 加速比取决于接受率α：理论加速~1/(1-α)，实践2-3×
- 变体：Self-Speculative（同模型不同层作draft）、Medusa（多头并行猜测）、Eagle（特征级预测）

### Q7: KV-Cache量化的挑战？
- KV-Cache占内存随序列长度线性增长：1M ctx + 128层 → 数百GB
- Key比Value更难量化（Key有异常值通道，方差大）
- KIVI方案：Key用per-channel量化（处理异常值），Value用per-token量化，分别2-bit/2-bit
- 最近token保留全精度（sliding window），旧token压缩
- 与PagedAttention正交可叠加

### Q8: 蒸馏 vs 量化 vs 剪枝如何组合？
- 典型pipeline：先蒸馏（得到好的小模型）→ 再量化（压缩部署）→ 可选剪枝（进一步瘦身）
- 蒸馏改变模型本身（知识转移），量化改变精度（不改结构），剪枝删权重（改稀疏度）
- 组合原则：蒸馏在前（需要训练），量化在后（PTQ不需训练），剪枝灵活
- 注意压缩复合效应：每层压缩都有损失，组合后可能叠加，需在每步评估

---

## 十五、搜索与推荐系统方向

**本方向核心关键词**：漏斗架构、召回/粗排/精排/重排、双塔/DSSM、ANN/HNSW/FAISS、DIN/DIEN、DeepFM/DCN、多目标/ESMM/MMoE、BM25/LTR、冷启动、在线学习

### Q1: 推荐漏斗各阶段的核心区别？
- 召回（万级，<10ms）：多路策略，重 Recall@K
- 粗排（千级，<5ms/item）：轻量排序，保持排序一致性
- 精排（百级，<10ms/item）：交叉特征，重 AUC/GAUC
- 重排（十级，<20ms）：列表级优化，兼顾多样性

### Q2: 双塔模型为什么不如精排模型？
- 双塔 inference 时 user 和 item 无交互（早期融合 vs 晚期融合）
- 无法建模交叉特征（"这个用户对红色商品的偏好"）
- 但双塔可建索引做 ANN，是召回的必选项
- 精排需要 user×item 交叉，无法预计算 item embedding

### Q3: DIN 的核心创新？
- Attention 建模用户行为与候选 item 的相关性
- 不是平均池化 → 不同候选激活不同的用户兴趣
- 证明"用户兴趣是多峰的"：喜欢手机 ≠ 喜欢手机壳
- DIEN 在此基础上加 GRU 建模兴趣**演化**

### Q4: 多目标优化三大模型？
- ESMM：P(CVR) = P(CTR) × P(CVR|CTR)，解决样本选择偏差
- MMoE：多 Expert + 每 Task 一个 Gate，灵活混合
- PLE：区分 shared/task-specific expert，解决 expert 坍塌
- 目标冲突：Pareto 优化 / GradNorm / 业务公式融合

### Q5: ANN 检索 HNSW 为什么快？
- Skip-list + 小世界图：上层稀疏长距跳，下层稠密精搜
- 从最高层贪心搜索逐层下沉，O(log N)
- vs IVF-PQ：HNSW 内存大但查询快；IVF-PQ 内存省但需 nprobe 调参
- 工业选择：Milvus/Qdrant（分布式）、FAISS（单机性能王）

### Q6: 离线AUC涨了线上CTR没涨，为什么？
- 选择偏差：离线数据只有曝光过的 item
- 特征 skew：训练用离线特征，serving 用实时特征
- 位置偏差：线上有位置效应，离线没模拟
- 分布偏移：AB 实验流量与全量用户分布不同

### Q7: LLM 在推荐中的三种角色？
- **Ranker**：直接排序（RankGPT），精度高但延迟大
- **Feature Extractor**：生成语义增强特征（KAR），离线不影响延迟
- **Generative Retrieval**：DSI/TIGER，将索引存在参数中，query → docID
- 2026 趋势：Agent × 推荐，LLM 主动管理推荐时机和意图澄清

### Q8: 冷启动三种场景的解法？
- 用户冷启动：bandit 探索 / 画像迁移 / LLM 对话问偏好
- 物品冷启动：内容特征替代行为特征 / 元学习 MAML / 流量倾斜
- 系统冷启动：内容特征 + 人工规则 → 种子用户行为 → 协同过滤
