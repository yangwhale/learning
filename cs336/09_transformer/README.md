# 3.5 完整的 Transformer 语言模型 + Resource Accounting

## 3.5 The Full Transformer LM

### Transformer Block

组装 Transformer block（参考 Figure 2）。每个 block 包含两个子层：

1. **Multi-Head Attention (MHA)**
2. **SwiGLU Feed-Forward Network (FFN)**

每个子层的结构为：**先 RMSNorm → 主操作 → 残差连接**。

第一个子层的公式（公式 15）：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x))$$

第二个子层同理，将 MultiHeadSelfAttention 替换为 SwiGLU FFN。

---

### Problem: 实现 Transformer Block (3 points)

**参数：**
- `d_model: int` — 模型维度
- `num_heads: int` — 注意力头数
- `d_ff: int` — FFN 中间层维度

**测试：**
- Adapter: `adapters.run_transformer_block`
- 测试命令：

```bash
uv run pytest -k test_transformer_block
```

**Deliverable**: 通过测试的代码。

---

### 完整 Transformer 语言模型

组装完整的语言模型（参考 Figure 1）。架构流程：

1. **Token embedding** — 将 token ID 映射为向量
2. **num_layers 个 Transformer blocks** — 堆叠多层 Transformer block
3. **最终 layer norm** — 对最后一层输出做 RMSNorm
4. **Output embedding (LM head)** — 线性投影回词汇表大小
5. **Logits** — 输出每个位置上词汇表的未归一化分数

---

### Problem: 实现 Transformer LM (3 points)

**额外参数（在 Transformer Block 参数之上）：**
- `vocab_size: int` — 词汇表大小
- `context_length: int` — 上下文长度（最大序列长度）
- `num_layers: int` — Transformer block 层数

**测试：**
- Adapter: `adapters.run_transformer_lm`
- 测试命令：

```bash
uv run pytest -k test_transformer_lm
```

**Deliverable**: 通过测试的 Transformer LM 模块。

---

## Resource Accounting（资源计算）

### FLOP 计算规则

**矩阵乘法 FLOPs：** 矩阵 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$ 的乘积 $AB$ 需要 $2mnp$ FLOPs。

**原因：**
- $(AB)[i,j] = A[i,:] \cdot B[:,j]$
- 每个点积需要 $n$ 次乘法和 $n$ 次加法（共 $2n$ FLOPs）
- 矩阵共有 $m \times p$ 个元素
- 总计 $(2n)(mp) = 2mnp$ FLOPs

---

### Problem: Transformer LM Resource Accounting (5 points)

使用 **GPT-2 XL** 配置：

| 参数 | 值 |
|------|-----|
| `vocab_size` | 50,257 |
| `context_length` | 1,024 |
| `num_layers` | 48 |
| `d_model` | 1,600 |
| `num_heads` | 25 |
| `d_ff` | 4,288（$\lfloor 8/3 \times 1600 \rfloor$ 取最近的 64 的倍数）|

**子问题：**

**(a)** 该模型有多少可训练参数？使用单精度浮点（float32）存储所有参数需要多少内存？

**(b)** 列出前向传播中的所有矩阵乘法及其对应的 FLOPs。假设输入序列长度 = `context_length`。

**(c)** 模型的哪些部分消耗最多 FLOPs？

**(d)** 对比 GPT-2 的四种规模，各组件 FLOPs 的占比如何随模型增大而变化？

| 模型 | 层数 | d_model | 头数 |
|------|------|---------|------|
| GPT-2 Small | 12 | 768 | 12 |
| GPT-2 Medium | 24 | 1,024 | 16 |
| GPT-2 Large | 36 | 1,280 | 20 |
| GPT-2 XL | 48 | 1,600 | 25 |

**(e)** 如果将 GPT-2 XL 的 `context_length` 增加到 16,384，总 FLOPs 和各组件的占比如何变化？
