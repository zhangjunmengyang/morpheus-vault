---
title: "UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn   Reinforcement Learning"
category: "论文精读"
tags: [Agent, 强化学习, 微调, 预训练]
created: "2026-02-13"
updated: "2026-02-13"
---

# UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn   Reinforcement Learning

by：字节跳动

论文链接：https://hf.co/papers/2509.02544

PaperScope.ai 解读：https://paperscope.ai/hf/2509.02544

![image](assets/Qct8d4rNookG55xdZ0Xcy4F7nb8.png)

由字节跳动等机构提出了UI-TARS-2，该工作通过系统性训练方法解决了GUI智能体开发中的数据扩展性、多轮强化学习、GUI操作限制和环境稳定性四大挑战。核心贡献包括：1）数据飞轮机制实现模型与训练数据的迭代优化，通过持续预训练、监督微调和多轮RL形成自强化循环；2）构建支持文件系统/终端交互的混合GUI环境，突破纯界面操作限制；3）开发异步状态化环境和流式更新的多轮RL框架，提升长序列训练稳定性；4）建立统一沙盒平台实现跨浏览器/虚拟机/模拟器的百万级rollout。
