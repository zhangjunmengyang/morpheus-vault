---
title: "蒙特卡洛树搜索 MCTS"
type: concept
domain: ai/llm/rl/fundamentals
created: "2026-02-13"
updated: "2026-02-13"
tags:
  - ai/llm/rl/fundamentals
  - type/concept
---
# 蒙特卡洛树搜索 MCTS

https://zhuanlan.zhihu.com/p/9644482549

蒙特卡洛树搜索算法的核心是：选择与模拟。

蒙特卡洛树搜索算法的主要目标是：给定一个游戏状态来选择最佳的下一步。

多臂老虎机（Multi-Armed Bandits)。

- 纯随机(Random)：每次随机选一个摇臂进行摇动。
（劣势：能算个期望收益，但收益不是最大的。）
- 仅探索（Exploration-only)：每个摇臂摇动T/K次。
相当于每个摇臂摇动的次数都一样。(劣势：浪费次数在收益较差的摇臂上)
- 仅利用（Exploitation-only）： 首先对每个摇臂摇动一次，记录收益。随后对剩余的T-K次机会全部摇收益最大的摇臂。（劣势：摇动一次记录的收益随机性太强，不可靠）
上述 仅探索（Exploration-only) 与 仅利用（Exploitation-only）所带来的困境也称为 [Exploration-Exploitation Dilemma](https%3A%2F%2Fzhida.zhihu.com%2Fsearch%3Fcontent_id%3D250951745%26content_type%3DArticle%26match_order%3D1%26q%3DExploration-Exploitation%2BDilemma%26zhida_source%3Dentity)。

为了克服该困境

其它可以尝试的解决算法为：

 (1)ϵ -[贪心算法](https%3A%2F%2Fzhida.zhihu.com%2Fsearch%3Fcontent_id%3D250951745%26content_type%3DArticle%26match_order%3D1%26q%3D%25E8%25B4%25AA%25E5%25BF%2583%25E7%25AE%2597%25E6%25B3%2595%26zhida_source%3Dentity)：在探索与利用之间进行平衡的搜索算法，具体执行过程为，在第t步，ϵ-贪心算法按照如下机制来选择摇动赌博机：

- 以 1−ϵ 的概率，选择在过去t-1次摇动赌博机所得平均收益最高的摇臂进行摇动；
- 以 ϵ 的概率，随机选择一个摇臂进行摇动。
上述贪心算法的不足为没有充分的考虑到每个摇臂被探索的次数。

(2)[上限置信区间算法](https%3A%2F%2Fzhida.zhihu.com%2Fsearch%3Fcontent_id%3D250951745%26content_type%3DArticle%26match_order%3D1%26q%3D%25E4%25B8%258A%25E9%2599%2590%25E7%25BD%25AE%25E4%25BF%25A1%25E5%258C%25BA%25E9%2597%25B4%25E7%25AE%2597%25E6%25B3%2595%26zhida_source%3Dentity)（Upper Confidence Bounds, UCB）:为每个动作的奖励期望计算一个估计的范围，优先采用估计范围上限较高的动作。(也是蒙特卡洛树用到的算法）

假设每个摇臂的收益的均值为 Q(k) ，估计的偏差为 δ(k) ，则每次根据Q(k)+δ(k) 的上界选择摇臂。
