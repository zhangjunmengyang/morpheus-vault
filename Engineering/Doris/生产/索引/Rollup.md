---
title: "Rollup"
type: concept
domain: engineering/doris/生产/索引
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - engineering/doris/生产/索引
  - type/concept
---
# Rollup

### Rollup索引

通过建表语句创建出来的表为Base表，在Base表上，可以创建多个Rollup表，在物理上是独立存储的。作用：上卷、索引。

- Rollup会占用额外物理存储空间，影响导入速度（导入阶段生成所有rollup），Rollup数据更新与Base表同步。
- 只有聚合和唯一模型的Rollup有上卷作用，冗余模型的Rollup只有调整前缀索引的作用。
- 命中Rollup的必要条件：
- 查询涉及的所有列都在一张Rollup表中。
- Join类型是Inner Join或Left Join。
- 命中多个Rollup，则使用数据量小的和创建更早的。
- count(*)无法命中。