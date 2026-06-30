# Section 7.5: 自定义修改 + 排行榜 (Your Own Modification + Leaderboard)

## 概述

恭喜你走到这里！在这个部分，你将尝试改进 Transformer 架构，看看你的超参数和架构选择如何与班上其他同学比较。

---

## 排行榜规则

除以下限制外，没有其他限制——你可以自由发挥：

| 限制项 | 要求 |
|--------|------|
| **Runtime** | 提交最多在 **1 块 B200 GPU** 上运行 **45 分钟**。可在提交脚本中使用 SLURM 或 Modal 来强制执行时间限制。 |
| **Data** | 只能使用提供的 **OpenWebText 训练数据集**。 |
| **其他** | 随意。 |

---

## 改进思路

以下是一些可以参考的方向：

### 1. SOTA 开源 LLM 系列

- **Llama 3** [Grattafiori et al., 2024]
- **Qwen 2.5** [Yang et al., 2024]

研究这些模型的架构选择和训练技巧，看看哪些可以应用到你的实现中。

### 2. NanoGPT Speedrun 仓库

- 仓库地址：`github.com/KellerJordan/modded-nanogpt`
- 社区成员发布了很多有趣的 speedrun 预训练修改

**经典修改示例 — Weight Tying（权重绑定）**：

将输入嵌入层 (embedding) 和输出嵌入层 (LM head) 的权重绑定在一起（参考 Vaswani et al. [8] Section 3.4, Chowdhery et al. [16] Section 2）。

> **提示**：如果使用 weight tying，可能需要降低 embedding/LM head 初始化的标准差。

### 3. 建议的测试流程

先在 OWT 的小子集或 TinyStories 上测试这些修改，验证其有效性后，再进行完整的 45 分钟训练。

### 4. 关于可推广性的注意

排行榜中表现好的修改**不一定能推广到更大规模的预训练**。后续课程中的 scaling laws 单元会进一步探讨这个话题。

---

## Problem: Leaderboard (10 B200 hrs) (6 points)

在排行榜规则下训练模型，目标是在 **0.75 B200-hours** 内最小化 validation loss。

### Deliverable

- **最终 validation loss**
- 相应的**学习曲线**（x 轴为 wall-clock time，$\leq$ 45 分钟）
- 做了什么修改的**描述**
- 排行榜至少要达到 naive baseline 的 **5.0 loss**

### 提交方式

提交到排行榜仓库：`github.com/stanford-cs336/assignment1-basics-leaderboard`
