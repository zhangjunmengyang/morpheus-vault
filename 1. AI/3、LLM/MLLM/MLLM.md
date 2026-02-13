---
title: "MLLM"
category: "AI"
tags: [Attention, BERT, CLIP, Diffusion, LLM]
created: "2026-02-13"
updated: "2026-02-13"
---

# MLLM

# 信息源

想往多模态方面去发展：

- https://github.com/Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs
- https://github.com/showlab/Awesome-MLLM-Hallucination
https://github.com/JiaWang2001/Paper_reading

https://github.com/lwpyh/Awesome-MLLM-Reasoning-Collection

# 参考

https://arxiv.org/pdf/2312.16602

# 现状认知

最近，多模态大模型取得重大进展。随着数据集和模型的规模不断扩大，**传统的 MM 模型从头开始训练带来了巨大的计算量**，所以一个合理的方法是利用好现成的训练好的单模态基础模型，尤其是 LLM。这样可以减少多模态训练的费用，提升训练效率。

MM-LLM 利用 LLM为各种 MM 任务提供认知能力。LLM 具有良好的语言生成，zero-shot 和 ICL（上下文学习） 的能力。其他模态的基础模型则提供了高质量的表征。考虑到不同模态的模型是分开训练的，如何将不同模态连接起来，实现协同推理，是核心挑战。

# 发展历程

引用：https://zhuanlan.zhihu.com/p/682893729

![image](assets/QneHdwKS8oh4jhxVvGic58aTnug.png)

1. 最初的发展集中在多模态的内容理解和文本的生成
1. 同时实现多模态的输入和输出工作 MM-LMM，探索特定模态的生成
1. 将 LLM 和外部工具继承进来，实现“any-to-any”的 多模态理解和生成
# 架构

## Modality Encoder：

模态编码器主要是对来自不同模态的输入进行编码，来获得相应的特征：

F_X = ME_X(I_X)

存在各种预训练的编码器来处理不同的模态，模态可以是图像，视频，音频，3D 等。

### 视觉模态

对于图像，一般有四个可选的编码器，NFNet-F6，ViT，CLIP VIT，EVA-CLIP ViT。

- NFNet-F6：是一个无归一化的 ResNet 网络，可以在增强过的数据集上获得 SOTA 的图像识别的性能。
- VIT：采用 transformer 模型，将 image 变成 patch，然后对图像进行处理。然后经过线性投影flatten，然后经过多个 transformer 模块。
- CLIP-VIT：利用大量的文本-图像快，通过对比学习来优化 ViT，将成对的文本图像视为正样本，其他的文本和图像视为负样本。
- EVA-CLIP：对大规模的 CLIP 训练稳定了训练过程和优化过程。
对于视频，可以统一采样 5 帧，进行与图像同样的处理。

### 音频模态

通常使用 C-Former，HuBERT，BEATs 和 Whisper 等进行编码。

- C-Former：使用了 CIF 对齐机制来实现序列的转换，并且使用一个 Transformer 来提取音频特征
- HuBERT：是一个自监督的语音表征徐诶框架，基于 BERT。通过离散hidden units 的mask 预测来实现
- BEAT 是：是一个迭代的音频预训练框架，使用音频 Transformer 来学习双向编码表示
## 输入 Projector：

输出projector 的任务是将其他模态的编码特征F_X与文本特征空间的特征T进行对齐。对齐后的特征作为prompts P_x联通文本特征F_T输入到 LLM Backbone 内。给定 X 模态-text数据集\{I_X,t\},目标是最小化生成损失。

输入 Projecor 可以通MLP 或者多层 MLP 来实现。也有复杂的实现，比如 Cross-Attention，Q-Former，P-Former 等。Cross-Attention 使用一系列的可训练的 query 和编码特征 F_X作为 key 来压缩特征序列到固定的长度。将压缩的表示特征输给 LLM。

## LLM Backbone：

LLM作为核心智能体，MM-LLMs 可以继承一些显着的属性，如零样本泛化（zero-shot）、少样本 ICL、思想链 (CoT) 和指令遵循。 LLM 主干处理来自各种模态的表示，参与有关输入的语义理解、推理和决策。它产生 (1) 直接文本输出 t，以及 (2) 来自其他模式（如果有）的信号token S_x。这些信号token充当指导生成器是否生成 MM 内容的指令，如果是，则指定要生成的内容：

t,S_X = LLM(P_X,F_T)

上式中，其他模态P_X的对齐后的表征，可以认为是软 prompt-tuning，输给 LLM Backbone。发而且一些研究工作引入了 PEFT 的方法，例如 Prefix-tuning，Adapter 和 LoRA。这些 case 里面，希望更少的参数可以被训练，甚至少于 0.1% 的 LLM 的参数参与训练。

通常用到的 LLM 模型有 Flan-T5，ChatGLM，UL2，Qwen，Chinchilla，OPT，PaLM，LLaMA ，LLaMA2 ，Vicuna 等。

## Output Projector：

输出Projector将 LLM 的输出的 token 表征S_X转变成特征H_X，然后输给生成器MG_X。

给定数据X-text数据集\{I_X, t\}，首先将文本t输给 LLM，生成对应的S_X，然后映射得到H_X。模型优化的目标是最小化H_X与MG_X的条件文本之间的距离。

## 模态生成器：

模态生成器MG_X一般用于生成不同的模态来输出。当前的工作一般使用现成的扩大模型（Latent diffusion model），例如 Stable Diffusion用于图像生成，Zeroscope用于视频生成，AudioLDM-2 用于音频生成。

输出 Projector 输出的特征H_x作为条件输入，在去噪的过程中，用于生成 MM 的内容。训练过程中， gt content 首先转换为 latent feature z_0，由预训练好的 VQA 模型。然后噪声\epsilon加到z_0上，获得 noise latent feature z_t,预训练好的 UNet 用于计算条件损失，通过最小化 loss 来优化参数。

# 数据集

![image](assets/XXCadgmIEolKBRx3t0Zcd5IFn3b.png)
