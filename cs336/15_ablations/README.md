# Section 7.3: 消融实验与架构修改 (Ablations and Architecture Modification)

## 概述

通过修改 Transformer 的各个组件来深入理解其行为。

---

## Ablation 1: Layer Normalization

### Problem: Remove RMSNorm and Train (0.5 B200 hrs) (1 point)

移除模型中所有的 RMSNorm 层后进行训练。

需要回答的问题：
- 在之前找到的最优学习率下会发生什么？
- 能否通过降低学习率来获得训练稳定性？

#### Deliverable

- 移除 RMSNorm 后的学习曲线
- 使用最佳学习率的学习曲线
- 几句话评论 RMSNorm 对训练的影响

---

## Ablation 2: Pre-norm vs Post-norm

### Pre-norm（我们当前的实现）

Pre-norm 是在子层操作**之前**进行归一化（公式 25-26）：

$$z = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x))$$

$$y = z + \text{FFN}(\text{RMSNorm}(z))$$

### Post-norm（原始 Transformer 架构）

Post-norm 是在残差连接**之后**进行归一化（公式 27-28）：

$$z = \text{RMSNorm}(x + \text{MultiHeadSelfAttention}(x))$$

$$y = \text{RMSNorm}(z + \text{FFN}(z))$$

### Problem: Implement Post-norm and Train (0.5 B200 hrs) (1 point)

将模型改为 post-norm 实现并训练。

#### Deliverable

- Post-norm vs Pre-norm 的学习曲线对比

由此可见，layer normalization 不仅对 Transformer 的行为影响很大，归一化的**位置**同样至关重要。

---

## Ablation 3: 位置嵌入 (Position Embeddings)

比较 **RoPE** vs **完全不使用位置嵌入 (NoPE)**。

仅使用 causal mask 的 decoder-only Transformer 理论上可以推断出相对或绝对位置信息（Tsai et al. [28], Kazemnejad et al. [29]）。

### Problem: Implement NoPE (0.5 B200 hrs) (1 point)

移除 RoPE 位置编码，观察其对模型性能的影响。

#### Deliverable

- RoPE vs NoPE 的学习曲线对比

---

## Ablation 4: SwiGLU vs SiLU

按照 Shazeer [20] 的方法，比较 **SwiGLU FFN** 和不带门控的 **SiLU FFN**。

不带门控的 SiLU FFN 定义如下（公式 29）：

$$\text{FFN}_{\text{SiLU}}(x) = W_2 \, \text{SiLU}(W_1 x)$$

**参数量匹配注意事项**：

- SwiGLU 使用 $d_{ff} = \frac{8}{3} \times d_{model}$，有 **3 个**权重矩阵
- SiLU baseline 应设置 $d_{ff} = 4 \times d_{model}$，有 **2 个**权重矩阵

这样两者的参数量近似匹配，确保对比的公平性。

### Problem: SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)

#### Deliverable

- SwiGLU vs SiLU 的学习曲线对比（参数量近似匹配）
- 几句话讨论你的发现

---

## Low-Resource Tip

> **资源有限的在线学生应在 TinyStories 上测试架构修改**
>
> 后续我们将转向更大规模、更有噪声的 web 数据集 (OpenWebText)，在其上进行架构修改实验和排行榜提交。在 OWT 上训练到流畅需要较长时间，建议资源有限的在线学生继续在 TinyStories 上测试这些修改（用 validation loss 来评估效果）。
