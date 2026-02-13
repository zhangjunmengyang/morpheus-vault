---
title: "PPL 计算 交叉熵损失与 ignore_index"
category: "AI"
tags: [BERT, GPT, LLM, T5, 交叉熵]
created: "2026-02-13"
updated: "2026-02-13"
---

# PPL 计算 交叉熵损失与 ignore_index

本期 code：https://github.com/chunhuizhang/bert_t5_gpt/blob/main/tutorials/gpt2_training_inference_ppl.ipynb

穿插阅读 GPT2源码，几个细节：

-  **train + inference**：gpt 的因果注意力，单向的，意思是后面需要预测的词只能看到前面的词，在 inference 的过程也是根据 0 到 n-1 的 token 去预测下一个 token。
- 由此就可以在训练的时候实现等效的并行训练，实际上就是一个下三角矩阵，可以通过这个矩阵直接把所有的 loss 都算出来，因为都是已知的
- CrossEntropyLoss：labels 起到选择的作用，ignore_index：过滤（-100），PPL 计算的时候会用到。
- 两种方式等效：
- 直接求 loss loss **=** nn**.**CrossEntropyLoss()
- 先 F**.**log_softmax(input, dim**=-**1)，然后取平均，再去掉负号