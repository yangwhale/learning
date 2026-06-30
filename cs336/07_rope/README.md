# 3.4.3 相对位置嵌入 (Relative Positional Embeddings)

## 背景知识

我们使用 **Rotary Position Embeddings (RoPE)** (Su et al. [21]) 来编码位置信息。

### RoPE 的核心思想

对于位置 $i$ 处的 query token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们应用一个旋转矩阵 $R^i$：

$$q'^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}$$

$R^i$ 将嵌入元素 $q_{2k-1:2k}^{(i)}$ 视为一个 2D 向量，并按角度 $\theta_{i,k}$ 旋转：

$$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$$

其中 $k \in \{1, \ldots, d/2\}$，$\Theta$ 是一个常数。

### 旋转矩阵

$R^i$ 是一个分块对角矩阵，每个对角块为标准的 2D 旋转矩阵：

$$R_k^i = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix}$$

完整的旋转矩阵（公式 9）：

$$R^i = \begin{pmatrix} R_1^i & 0 & \cdots & 0 \\ 0 & R_2^i & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & R_{d/2}^i \end{pmatrix}$$

### 高效实现提示

- **不需要构造完整的 $d \times d$ 矩阵**。可以预计算 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 值。
- 预计算的值可以**跨层复用**（每一层都做相同的旋转）。
- 预计算的值也可以**跨 batch 复用**。
- 可以使用 `self.register_buffer(persistent=False)` 来预计算一个 2D 的 sin/cos buffer。
- 对 **key 向量做完全相同的旋转**。
- **注意：RoPE 层没有可学习参数。**

---

## Problem: 实现 RoPE (2 points)

实现 RoPE 模块。

### `__init__` 方法

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

**参数：**
- `theta: float` — RoPE 的 $\Theta$ 值
- `d_k: int` — query/key 向量的维度
- `max_seq_len: int` — 最大序列长度

### `forward` 方法

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

**参数：**
- `x` — 输入张量，shape 为 `(..., seq_len, d_k)`，可以有任意数量的 batch 维度
- `token_positions` — 位置索引张量，shape 为 `(..., seq_len)`

**返回：**
- 旋转后的张量，shape 与输入 `x` 相同

### 实现要点

使用 `token_positions` 来切片（index into）预计算的 cos/sin 张量，以获取对应位置的旋转角度。

### 测试

- Adapter: `adapters.run_rope`
- 测试命令：

```bash
uv run pytest -k test_rope
```
