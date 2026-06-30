# CS336 Assignment 1 - Part 5: RMSNorm

> 对应原始 PDF Section 3.4-3.4.1 (pages 19-20)

## 3.4 Pre-Norm Transformer Block

每个 Transformer block 包含两个子层：**multi-head self-attention** 和 **position-wise feed-forward network**（Vaswani et al. [8]）。

### Post-Norm vs Pre-Norm

原始 Transformer 使用 **post-norm** 架构：在每个子层的输出之后进行 layer normalization。然而，大量后续研究发现，将 layer normalization 从子层输出移到子层输入（即 **pre-norm**），并在最终 Transformer block 之后添加额外的 layer norm，可以显著改善训练稳定性（Nguyen et al. [9], Xiong et al. [10]）。

参见 Figure 2 中的 pre-norm Transformer block 结构：

![Transformer Block](../images/figure2_transformer_block.png)

*Figure 2: Pre-norm Transformer block。归一化层放在 self-attention 和 FFN 子层之前，而不是之后。*

### Pre-Norm 的直觉

在 pre-norm 架构中，存在一条从输入嵌入到最终输出的干净"**残差流**" (residual stream)，中间没有任何归一化操作。这条干净的残差路径被认为有助于改善梯度流，使深层网络更容易训练。

Pre-norm 已成为现代大型语言模型的标准做法，包括 GPT-3、LLaMA、PaLM 等。

---

## 3.4.1 Root Mean Square Layer Normalization (RMSNorm)

原始 Transformer 使用 Layer Normalization (Ba et al. [11])。在本作业中，我们按照 Touvron et al. [12] 的做法，使用 **RMSNorm** (Zhang and Sennrich [13])，它是 LayerNorm 的简化版本，省略了均值中心化步骤。

### 数学定义

给定向量 $\mathbf{a} \in \mathbb{R}^{d_\text{model}}$，RMSNorm 对每个激活值 $a_i$ 进行如下缩放：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(\mathbf{a})} \cdot g_i$$

其中：

$$\text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{d_\text{model}} \sum_{i=1}^{d_\text{model}} a_i^2 + \varepsilon}$$

- $g_i$ 是**可学习的 "gain" 参数**（共 $d_\text{model}$ 个），初始化为 1
- $\varepsilon$ 是一个小常数，用于数值稳定性，通常固定为 $1 \times 10^{-5}$

### 数值精度注意事项

**重要**：在执行平方运算之前，应将输入 **upcast 到 `torch.float32`**，以防止在低精度数据类型（如 `float16` 或 `bfloat16`）下发生溢出。`forward` 方法的整体结构应为：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# 在 float32 精度下执行 RMSNorm
...
result = ...

# 将结果转回原始数据类型
return result.to(in_dtype)
```

---

### Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)

实现 RMSNorm 模块。

```python
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Root Mean Square Layer Normalization.

        Args:
            d_model: int
                模型隐藏维度大小。gain 参数的长度。
            eps: float = 1e-5
                数值稳定性的 epsilon 值，添加到 RMS 的平方根内部。
            device: torch.device | None = None
                参数存储的设备。
            dtype: torch.dtype | None = None
                参数的数据类型。
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对输入执行 RMSNorm。

        Args:
            x: torch.Tensor
                输入张量，shape 为 (batch_size, sequence_length, d_model)。

        Returns:
            torch.Tensor
                归一化后的张量，shape 与输入相同：(batch_size, sequence_length, d_model)。
        """
        ...
```

**实现要点**：

1. **继承 `nn.Module`**，调用父类的 `__init__` 方法
2. **gain 参数**：创建一个 shape 为 `(d_model,)` 的 `nn.Parameter`，初始化为全 1
3. **存储 epsilon**：将 `eps` 存储为实例属性
4. **Upcast + Downcast**：
   - 在 `forward` 中，先将输入 upcast 到 `torch.float32`
   - 执行 RMSNorm 计算
   - 将结果 downcast 回输入的原始 dtype
5. **RMS 计算**：对最后一维（`d_model` 维）求均方根，注意在求平方根时加入 epsilon

**测试方法**：

```bash
# 通过 adapter 运行
adapters.run_rmsnorm

# 运行测试
uv run pytest -k test_rmsnorm
```
