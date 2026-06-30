# 3.4.4-3.4.5 Attention 机制

## 3.4.4 Scaled Dot-Product Attention

### Softmax

首先实现 softmax 函数（公式 10）：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}$$

#### 数值稳定性

直接计算 $\exp(v_i)$ 可能会溢出为 `inf`，导致 `inf/inf = NaN`。

**解决方法：** 利用 softmax 对所有输入加常数 $c$ 不变的性质，将每个元素减去最大值，使最大项变为 0：

$$\text{softmax}(v)_i = \text{softmax}(v - \max(v))_i$$

---

### Problem: 实现 Softmax (1 point)

实现一个接受张量和维度 $i$ 的 softmax 函数，在第 $i$ 维上应用 softmax。

**要求：**
- 输出形状与输入相同
- 第 $i$ 维现在是归一化的概率分布（即该维度上的值求和为 1）

**测试：**
- Adapter: `adapters.run_softmax`
- 测试命令：

```bash
uv run pytest -k test_softmax_matches_pytorch
```

---

### Attention 操作

Attention 的计算公式（公式 11）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$ — Query 矩阵
- $K \in \mathbb{R}^{m \times d_k}$ — Key 矩阵
- $V \in \mathbb{R}^{m \times d_v}$ — Value 矩阵

> **注意：** $Q$, $K$, $V$ 都是**输入**，不是可学习参数。

### Masking

使用 mask $M \in \{True, False\}^{n \times m}$：
- $True$ 表示可以 attend（保留）
- $False$ 表示不可 attend（屏蔽）

**实现方式：** 在 pre-softmax 值 $\frac{QK^T}{\sqrt{d_k}}$ 上，对 mask 为 $False$ 的位置加上 $-\infty$。

---

### Problem: 实现 Scaled Dot-Product Attention (5 points)

**输入：**
- Keys 和 Queries 的 shape: `(batch_size, ..., seq_len, d_k)`
- Values 的 shape: `(batch_size, ..., seq_len, d_v)`
- 可选的 boolean mask，shape: `(seq_len, seq_len)`

**输出：**
- Shape: `(batch_size, ..., seq_len, d_v)`

**测试：**
- Adapter: `adapters.run_scaled_dot_product_attention`
- 测试命令：

```bash
# 3 阶张量测试
uv run pytest -k test_scaled_dot_product_attention

# 4 阶张量测试
uv run pytest -k test_4d_scaled_dot_product_attention
```

---

## 3.4.5 Causal Multi-Head Self-Attention

### Multi-Head Attention

Multi-Head Attention 的公式（公式 12-14）：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

$$\text{MultiHeadSelfAttention}(x) = W_O \, \text{MultiHead}(W_Q x, \, W_K x, \, W_V x)$$

### 可学习参数

- $W_Q \in \mathbb{R}^{hd_k \times d_{model}}$ — Query 投影矩阵
- $W_K \in \mathbb{R}^{hd_k \times d_{model}}$ — Key 投影矩阵
- $W_V \in \mathbb{R}^{hd_v \times d_{model}}$ — Value 投影矩阵
- $W_O \in \mathbb{R}^{d_{model} \times hd_v}$ — 输出投影矩阵

**实现技巧：** 将 $W_Q$, $W_K$, $W_V$ 视为沿输出维度为每个头分割的单个矩阵。总共只需要**三次矩阵乘法**（而非每个头单独乘）。

> **脚注 5：** 作为进阶目标，可以尝试将 QKV 三个投影合并为一个单独的权重矩阵，只做一次矩阵乘法。

### Causal Masking

Causal masking 防止模型 attend 到未来的 token——对于位置 $i$，位置 $j > i$ 的 token 不可见。

**实现方式：** 使用 `torch.triu` 或广播的索引比较来构造 causal mask。

### 应用 RoPE

- RoPE 应用于 $Q$ 和 $K$ 向量，**不应用于 $V$**
- Head 维度作为 batch 维度处理
- 每个 head 应用相同的旋转

---

### Problem: 实现 Causal Multi-Head Self-Attention (5 points)

```python
# 参数：
# d_model: int    输入维度
# num_heads: int  注意力头数
# 设 d_k = d_v = d_model / num_heads (遵循 Vaswani et al. [8])
```

**测试：**
- Adapter: `adapters.run_multihead_self_attention`
- 测试命令：

```bash
uv run pytest -k test_multihead_self_attention
```
