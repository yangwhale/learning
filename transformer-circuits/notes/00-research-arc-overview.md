# Transformer Circuits 研究主线总结

## The Arc: From Black Box to Circuit Diagram (2021–2026)

Six years, one question: **what is a language model actually doing inside?**

### The Journey

```
2021  Framework     给 transformer 画了第一张零件图
        │           发现 residual stream 是共享总线，attention head 是独立电路
        │
2022  Superposition 发现零件不可读的根本原因
        │           模型把 1000 个概念塞进 500 维空间（近似正交 + 稀疏性）
        │           所以单个神经元 = 多个概念的混合投影，看不懂
        │
2022  Induction     发现第一个完整电路
        │           两个 head 跨层组合，实现 "见过 AB...A → 预测 B"
        │           这就是 in-context learning 的机制，不是理解，是模式匹配
        │
2023  SAE           发明了拆解工具
        │           overcomplete autoencoder + L1 稀疏 = 每个 latent 对应一个干净概念
        │           小模型上概念验证成功
        │
2024  Scaling       把工具用到真实大模型上
        │           Claude 3 Sonnet → 3400 万个特征，包括抽象概念和安全相关特征
        │           Feature steering 验证因果性（金门大桥实验）
        │
2025  Circuits      画出完整电路图
        │           Transcoder 分解 MLP 计算 + autodiff 构建 attribution graph
        │           对 Claude 3.5 Haiku 做全面解剖：多步推理、安全拒绝、metacognition
        │
2026  Workspace     触及自我意识边界
                    模型内部特征分 verbalizable / non-verbalizable 两类
                    对应认知科学的 Global Workspace Theory
                    定义了模型自我报告的可信度边界
```

### 核心逻辑链

1. Transformer 的 residual stream 是共享总线 → 可以分解分析
2. 但神经元因为 superposition 是混的 → 分解需要新工具
3. SAE 利用稀疏性拆解 superposition → 得到干净特征
4. 干净特征 + transcoder + autodiff → 完整电路图
5. 电路图 → 可以做机制级 debugging、安全监控、行为干预

### 对 Infra 的启示

| 论文 | Infra 启示 |
|------|-----------|
| Framework | Pipeline parallelism 切点 = 切断 residual stream 通信 |
| Superposition | 量化敏感度取决于每层的 superposition 密度 |
| Induction Heads | KV cache 是 ICL 的物理基础，不能随便裁 |
| SAE | Feature 监控需要额外推理开销（encoder forward pass） |
| Scaling | 安全监控可以从"看输出猜"变成"读内部状态" |
| Circuit Tracing | 模型 debugging 从黑盒变成看电路图 |
| Global Workspace | 自我报告式安全监控有可信度边界 |

## 关键类比速查

- **Residual stream** = 共享黑板 / 公共总线
- **Superposition** = 50 把椅子的教室塞 150 人（方向近似正交 + 稀疏性）
- **SAE** = 100 个独立麦克风分离 100 个人的声音
- **Transcoder vs SAE** = 生产线摄像头 vs 产品照片
- **QK/OV** = 眼睛（看哪里）vs 手（拿什么）
- **Induction head** = 见过 AB...A → 预测 B 的模式匹配器
- **Global Workspace** = CEO 简报 vs 中层管理决策（能汇报的 vs 不能的）

## 还没讲到的 Side Quest 论文

- Softmax Linear Units (2022) — 让更多神经元变得 monosemantic 的激活函数
- Privileged Bases (2023) — Adam 优化器让某些坐标轴变得特殊
- Sparse Crosscoders (2024) — 跨层特征 + 模型 diffing
- Natural Language Autoencoders (2026) — 让模型用自然语言描述自己的内部状态
- Emotion Concepts (2026) — Claude 内部有因果有效的情绪表征
