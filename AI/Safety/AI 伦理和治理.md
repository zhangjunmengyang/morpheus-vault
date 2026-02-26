---
title: "AI 伦理和治理"
brief: "AI 伦理与治理概览：偏见/公平/透明/可问责/隐私五大维度；全球监管框架对比（EU AI Act/美国/中国）；对齐与治理的边界划分"
type: concept
domain: ai/safety
created: "2026-02-13"
updated: "2026-02-22"
tags:
  - ai/safety
  - type/concept
---
# AI 伦理和治理

AI 伦理不是学术空谈 —— 它直接影响产品设计决策、数据使用方式、模型部署策略。作为工程师，理解这些边界是必须的。

## 核心伦理议题

### 1. 偏见与公平性（Bias & Fairness）

模型从数据中学习，数据有偏见，模型就有偏见。这不是 bug，是 feature（特征学习的必然结果）。

常见偏见类型：
- **Selection bias** —— 训练数据不代表真实分布（如只用英文数据训练的模型对中文理解差）
- **Measurement bias** —— 标注标准不一致（不同标注员对「有害内容」理解不同）
- **Historical bias** —— 历史数据反映了过去的不公平（如简历筛选模型歧视女性）

去偏不是简单的数据平衡。一些实践方法：

```python
# 公平性评估的基本框架
def evaluate_fairness(model, test_data, sensitive_attrs):
    results = {}
    for attr in sensitive_attrs:  # e.g., gender, race, age
        groups = test_data.groupby(attr)
        for group_name, group_data in groups:
            preds = model.predict(group_data)
            results[f"{attr}_{group_name}"] = {
                "accuracy": accuracy_score(group_data.labels, preds),
                "positive_rate": preds.mean(),  # demographic parity
                "fpr": false_positive_rate(group_data.labels, preds),  # equalized odds
            }
    return results
```

### 2. 透明性与可解释性

「为什么模型这样判断？」—— 这个问题在医疗、金融、法律领域是合规硬性要求。

- **LIME / SHAP** —— 事后解释，不改模型本身
- **Attention visualization** —— 看 attention 权重，但研究表明 attention ≠ explanation
- **Chain-of-thought** —— LLM 时代的可解释性：让模型输出推理过程

### 3. 隐私

GDPR、中国《个人信息保护法》都对 AI 使用个人数据有严格限制。技术层面：

- **Differential Privacy** —— 在训练过程中加噪声，保证个体数据无法被反推
- **Federated Learning** —— 数据不出本地，只交换模型参数
- **Data Anonymization** —— 去标识化，但要注意 re-identification 攻击

## 治理框架

### 国际层面

- **EU AI Act** —— 按风险等级分类（不可接受/高/有限/最小风险），高风险 AI 需要合规审计
- **NIST AI RMF** —— 美国标准，侧重风险管理流程
- **中国《生成式 AI 管理办法》** —— 要求算法备案、安全评估、内容审核

### 企业层面

实际落地需要：

1. **AI 伦理委员会** —— 跨部门，有否决权
2. **Model Card** —— 每个上线模型都要有文档说明用途、限制、偏见评估
3. **Red Teaming** —— 上线前的对抗性测试
4. **监控与反馈** —— 上线后持续监控异常输出，用户反馈闭环

## 工程师的实践建议

1. **数据集文档化** —— 用 Datasheets for Datasets 框架记录数据来源、标注过程、已知偏见
2. **评估不能只看平均指标** —— 拆分人群看子集表现
3. **设计 kill switch** —— 模型上线后发现问题能快速下线
4. **记录决策** —— 为什么选这个模型、为什么用这些数据、做了什么 trade-off

## 我的看法

AI 伦理最大的挑战不是技术问题，而是**激励不对齐** —— 快速上线 vs 充分评估，商业价值 vs 社会影响。工程师能做的是：把伦理评估嵌入到 CI/CD pipeline 中，让它成为标准流程的一部分，而不是可选项。

## 相关

- [[AI/Safety/AI 安全及隐私保护|AI 安全及隐私保护]]
- [[AI/LLM/Application/Prompt-攻击|prompt 攻击]]
- [[AI/AI 思考|AI 思考]]
