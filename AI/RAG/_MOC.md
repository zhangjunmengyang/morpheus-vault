---
title: "检索增强生成 RAG"
type: moc
domain: ai/rag
tags:
  - ai/rag
  - type/reference
---

# 🔍 检索增强生成 RAG

> 从 Naive RAG 到 Advanced RAG，检索策略与向量存储

## 综合全景
- [[AI/RAG/RAG-2026-技术全景|RAG 2026 技术全景]] — 面试武器版，1239行，Naive→Advanced→Modular→Agentic RAG 全谱系

## 核心知识
- [[AI/RAG/Advanced RAG|Advanced RAG 进阶技术]] — GraphRAG / Self-RAG / Corrective RAG 等进阶方案
- [[AI/RAG/RAG 检索策略|RAG 检索策略详解]] — 稠密检索 / 稀疏检索 / 混合检索 / Reranker
- [[AI/RAG/向量数据库选型|向量数据库选型指南]] — Faiss / Milvus / Chroma / Qdrant 对比
- [[AI/RAG/RAG-Anything-Multimodal-RAG-Framework|RAG-Anything]] — Dual-graph 多模态 RAG：跨模态图+文本图融合，DocBench 63.4%，HKU HKUDS（★★★★☆）

## 项目实战
- [[Projects/RAG-System/企业 RAG 系统|企业 RAG 系统]] — 从 0 到 1 搭建企业级 RAG
- [[Projects/RAG-System/如何从 0 到 1 搭建数据库知识助手|从 0 到 1 搭建知识助手]]

## 相关 MOC
- ↑ 上级：[[AI/目录]]
- → 相关：[[AI/LLM/目录]]（Embedding）、[[AI/LLM/Application/Embedding/Embedding|Embedding]]

## 基础与原理
- [[AI/RAG/RAG 原理与架构|RAG 原理与架构]] — Naive RAG 基础原理、三阶段（检索/增强/生成）详解
- [[AI/RAG/RAG vs Fine-tuning|RAG vs Fine-tuning]] — 何时用 RAG、何时用微调的决策指南

## 工程组件
- [[AI/RAG/文档解析|文档解析]] — PDF/HTML/表格的解析策略
- [[AI/RAG/文本分块策略|文本分块策略]] — Fixed-size / Semantic / Sentence-window 分块方案对比
- [[AI/RAG/检索策略|检索策略（完整教程版）]] — 稠密/稀疏/混合检索完整代码实现
- [[AI/RAG/Reranker|Reranker]] — Cross-Encoder / ColBERT / LLM-as-Reranker 重排方案

## 评测
- [[AI/RAG/RAG 评测|RAG 评测]] — RAGAS / TruLens 指标体系
