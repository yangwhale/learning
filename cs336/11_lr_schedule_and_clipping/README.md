# 4.4-4.5 学习率调度与梯度裁剪

## 4.4 Learning Rate Scheduling（学习率调度）

使用 **cosine annealing schedule** (H. Touvron et al. [12])。

调度器接受当前步数 $t$ 和相关参数，返回该步的学习率 $\alpha_t$。

### Cosine Annealing Schedule

**参数：**

| 参数 | 含义 |
|------|------|
| $t$ | 当前迭代步数 |
| $\alpha_{max}$ | 最大学习率 |
| $\alpha_{min}$ | 最小（最终）学习率 |
| $T_w$ | Warm-up 迭代数 |
| $T_c$ | Cosine annealing 最终迭代 |

**三个阶段：**

**1. Warm-up 阶段**（$t < T_w$）：

$$\alpha_t = \frac{t}{T_w} \times \alpha_{max}$$

学习率从 0 线性增长到 $\alpha_{max}$。

**2. Cosine Annealing 阶段**（$T_w \le t \le T_c$）：

$$\alpha_t = \alpha_{min} + \frac{1}{2} \left(1 + \cos\left(\frac{t - T_w}{T_c - T_w} \times \pi\right)\right) \times (\alpha_{max} - \alpha_{min})$$

学习率按余弦曲线从 $\alpha_{max}$ 平滑衰减到 $\alpha_{min}$。

**3. Post-annealing 阶段**（$t > T_c$）：

$$\alpha_t = \alpha_{min}$$

学习率保持在最小值不变。

> **脚注 8：** 有时使用 learning rate 上升回来（restarts）的 schedule，以帮助跳出局部最小值。

---

### Problem: 实现 Cosine Learning Rate Schedule with Warmup (1 point)

实现一个函数，接受当前步数和调度参数，返回当前学习率。

**函数签名：**

```python
def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    # t: int          当前迭代步数
    # alpha_max: float 最大学习率
    # alpha_min: float 最小学习率
    # T_w: int         warm-up 迭代数
    # T_c: int         cosine annealing 最终迭代
    # 返回: float      当前步的学习率 α_t
    ...
```

**测试：**
- Adapter: `adapters.get_lr_cosine_schedule`
- 测试命令：

```bash
uv run pytest -k test_get_lr_cosine_schedule
```

---

## 4.5 Gradient Clipping（梯度裁剪）

### 背景

训练过程中偶尔会遇到异常大的梯度，导致训练不稳定（参数更新过大，loss 跳升甚至 diverge）。

**Gradient clipping** 在每次反向传播后、优化器步骤前，对梯度的范数施加一个上限。

### 算法

1. 将所有参数的梯度展平并拼接为一个向量 $g$
2. 计算其 $\ell_2$ 范数 $\|g\|_2$
3. 判断是否需要裁剪：
   - 如果 $\|g\|_2 \le M$（最大范数）：保持梯度不变
   - 如果 $\|g\|_2 > M$：将 $g$ 缩放 $\frac{M}{\|g\|_2 + \varepsilon}$

其中 $\varepsilon$（如 $10^{-6}$）用于数值稳定性。

> **注意：** 裁剪后的梯度范数将刚好略低于 $M$。

---

### Problem: 实现 Gradient Clipping (1 point)

**函数要求：**
- 接受参数列表和最大 $\ell_2$ 范数 $M$
- **原地修改**每个参数的 `.grad` 属性
- 使用 $\varepsilon = 10^{-6}$（PyTorch 默认值）

**测试：**
- Adapter: `adapters.run_gradient_clipping`
- 测试命令：

```bash
uv run pytest -k test_gradient_clipping
```
