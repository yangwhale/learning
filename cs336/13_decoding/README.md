# Section 6: 生成文本 (Generating Text)

## 背景知识

### Softmax

语言模型的输出是最终线性层的输出，称为 **logits**。为了将 logits 转换为归一化的概率分布，我们需要对其应用 **softmax** 函数。

### 解码 (Decoding)

给模型提供一个前缀 token 序列（即 **prompt**），模型会输出词汇表上的概率分布，用于预测下一个 token。然后我们从该分布中采样一个 token，将其追加到输入序列末尾，重复这个过程，直到生成特殊的 `<|endoftext|>` token 或达到最大 token 数。

具体地，解码的一步操作如下：给定输入序列 $x_{1..t}$，返回下一个 token $x_{t+1}$：

$$P(x_{t+1} = i \mid x_{1..t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}$$

其中：

$$v = \text{TransformerLM}(x_{1..t})_t \in \mathbb{R}^{\text{vocab\_size}}$$

### 解码技巧 (Decoder Tricks)

#### 1. Temperature Scaling（公式 23）

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{\text{vocab\_size}} \exp(v_j / \tau)}$$

当 $\tau \to 0$ 时，分布会集中到最大元素上（趋近于 one-hot 向量），使输出更加确定性。

#### 2. Nucleus (Top-p) Sampling（公式 24，Holtzman et al. [27]）

$$P(x_{t+1} = i \mid q) = \begin{cases} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{if } i \in V(p) \\ 0 & \text{otherwise} \end{cases}$$

其中 $V(p)$ 是使得 $\sum_{j \in V(p)} q_j \geq p$ 的**最小**索引集合。

**实现方式**：将词汇表中的 token 按概率从大到小排序，依次选取 token 直到累积概率达到阈值 $p$。只保留这些被选中的 token，将其余 token 的概率置零，然后对保留的 token 重新归一化。

---

## Problem: Decoding (3 points)

实现解码函数，建议支持以下功能：

- 给定用户提供的 prompt（token 序列 $x_{1..t}$），采样生成补全文本，直到生成 `<|endoftext|>` token
- 允许用户控制最大生成 token 数
- 给定 temperature 值 $\tau$，对预测的 next-token 分布应用 temperature scaling
- Top-p sampling

### Deliverable

- 解码函数的代码实现
