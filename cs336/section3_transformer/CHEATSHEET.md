# CS336 Assignment 1 · Section 3：Transformer LM — 上手指南

> 从零搭建完整 Transformer 语言模型的**路线图 + 每步提示 + 预警坑**。
> 理论细节看同目录 `README.md`，这份是"怎么下手 + 注意什么"。
> **状态：⬜ 未开始**（做完后像 section2 一样补成完整笔记）

---

## 目录

- [0. 和 Section 2 最大的不同（必读）](#0-和-section-2-最大的不同必读)
- [1. 文件 & adapter 接线](#1-文件--adapter-接线)
- [2. 构建顺序（乐高式，从零件到整机）](#2-构建顺序乐高式从零件到整机)
- [3. 每个组件：要点 + 提示 + 坑](#3-每个组件要点--提示--坑)
- [4. 测试命令](#4-测试命令)
- [5. 预警坑合集](#5-预警坑合集)
- [6. 需要先补的 PyTorch 基础](#6-需要先补的-pytorch-基础)

---

## 0. 和 Section 2 最大的不同（必读）

### 差异 1：这是 PyTorch，不是纯 Python

Section 2 是纯 Python 字符串/字节操作。Section 3 是**张量运算 + `nn.Module` + 自动求导**。核心技能：

- 张量 shape 变换（`reshape` / `view` / `transpose` / `einsum`）
- broadcasting（广播规则）
- `nn.Module` / `nn.Parameter` / `register_buffer`

### 差异 2：★权重是「注入」的，不是自己初始化的★

这是最容易懵的地方。看 adapter 签名：

```python
def run_linear(d_in, d_out, weights, in_features):   # ← weights 是传进来的！
    ...
```

**测试自己造好权重，通过 adapter 塞给你**，你只负责"用这些权重算出正确输出"。

**两种实现策略**（选一种，全程统一）：

| 策略 | 做法 | 适合 |
|------|------|------|
| **A. 纯函数** | adapter 里直接用 `weights` 做矩阵运算，不建类 | 简单组件（linear/softmax/silu）|
| **B. nn.Module + load** | 写 `nn.Module` 类 → adapter 构造它 → `load_state_dict` 塞权重 → forward | 复杂组件（transformer_lm）|

**推荐混合**：底层零件用 A（快），大模块用 B（清晰）。但 `transformer_lm` 的权重是个 **state_dict**（`{"token_embeddings.weight": ..., "layers.0.attn.q_proj.weight": ...}`），**必须**用 B——你的 Module 内部命名要和这些 key **完全对上**，才能 `load_state_dict` 成功。

### 差异 3：向量是「行向量」

作业约定 $\mathbf{x}$ 是行向量，`Linear` 是 $y = xW^T$，权重 shape 是 `(d_out, d_in)`。别写反。

---

## 1. 文件 & adapter 接线

```bash
cd ~/learning/cs336-assignment1
touch cs336_basics/model.py     # 新建，Transformer 代码写这里
```

adapter 接线（`tests/adapters.py`，每个 `run_*` / `get_*` 删掉 `raise NotImplementedError`）：

```python
# 纯函数策略示例（linear）
def run_linear(d_in, d_out, weights, in_features):
    return in_features @ weights.T      # 行向量: y = x W^T

# nn.Module 策略示例（transformer_lm）
def run_transformer_lm(vocab_size, context_length, d_model, num_layers,
                       num_heads, d_ff, rope_theta, weights, in_indices):
    from cs336_basics.model import TransformerLM
    model = TransformerLM(vocab_size, context_length, d_model, num_layers,
                          num_heads, d_ff, rope_theta)
    model.load_state_dict(weights)      # ← 内部命名必须和 weights 的 key 对齐
    return model(in_indices)
```

---

## 2. 构建顺序（乐高式，从零件到整机）

**严格按依赖顺序做，每层跑绿了再上一层。**

```
第 1 层 · 基础零件（无依赖，先做，最简单）
  ├─ run_linear       线性层 y = x W^T（无 bias）
  ├─ run_embedding    token ID → 向量（查表）
  ├─ run_rmsnorm      RMSNorm 归一化
  ├─ run_silu         SiLU = x·σ(x)
  └─ run_softmax      数值稳定 softmax（先减 max）

第 2 层 · 组合零件（依赖第 1 层）
  ├─ run_swiglu                        SwiGLU FFN（W1/W2/W3）
  ├─ run_rope                          旋转位置编码
  └─ run_scaled_dot_product_attention  缩放点积注意力

第 3 层 · 注意力模块（依赖第 2 层）
  ├─ run_multihead_self_attention            多头自注意力
  └─ run_multihead_self_attention_with_rope  带 RoPE 的多头注意力

第 4 层 · 积木块
  └─ run_transformer_block   一个完整 Transformer 层（pre-norm）

第 5 层 · 整机
  └─ run_transformer_lm      堆 N 层 + embedding + final norm + lm_head
```

**里程碑**：第 1 层全绿 → 说明 PyTorch 张量操作和 adapter 接线通了，后面就是搭积木。

---

## 3. 每个组件：要点 + 提示 + 坑

### 3.1 Linear（1 分）

- 公式：$y = x W^T$，权重 `(d_out, d_in)`，**无 bias**。
- 提示：`in_features @ weights.T`，或 `einsum('...i,oi->...o', x, W)`。
- 坑：别把 `W` 和 `W.T` 搞反；`...` 要能处理任意 batch 维。

### 3.2 Embedding（1 分）

- 本质：查表。token ID → 权重矩阵的第 ID 行。
- 提示：`weights[token_ids]`（PyTorch 高级索引，直接支持任意 shape 的 ID）。
- 坑：输出 shape 是 `token_ids.shape + (d_model,)`。

### 3.3 RMSNorm（1 分）

- 公式：$\frac{x_i}{\sqrt{\frac{1}{d}\sum x_j^2 + \epsilon}} \cdot g_i$
- ★关键坑：**先 upcast 到 float32** 算归一化，算完 **cast 回原 dtype**（数值稳定）。
- 提示：`x.pow(2).mean(dim=-1, keepdim=True)` 求均方；`keepdim=True` 才能广播。
- 坑：只在**最后一维** `d_model` 上归一化，别在 batch/seq 维上。

### 3.4 SiLU（1 分）

- 公式：$\text{SiLU}(x) = x \cdot \sigma(x)$
- 提示：`x * torch.sigmoid(x)`。一行搞定。

### 3.5 Softmax（1 分）

- ★数值稳定：**先减去最大值** 再 exp。`x - x.max(dim, keepdim=True).values`
- 坑：`dim` 参数指定在哪一维归一化；减 max 后不影响结果（softmax 平移不变）。

### 3.6 SwiGLU FFN（2 分）

- 公式：$\text{FFN}(x) = W_2\,(\text{SiLU}(W_1 x) \odot W_3 x)$
- 三个权重：`W1,W3: (d_ff, d_model)`，`W2: (d_model, d_ff)`，无 bias。
- $\odot$ 是逐元素乘。
- 坑：`d_ff` 有时要 round 到 64 的倍数（作业给公式），但测试通常直接给 `d_ff`，按参数用即可。

### 3.7 RoPE（2 分）

- 本质：给 Q、K 按位置**旋转**，把位置信息编码进去。**无可学习参数**。
- 只作用于 **Q 和 K，不作用于 V**。
- $\theta_i = \theta_{\text{base}}^{-2i/d}$，$\theta_{\text{base}}=10000$。
- ★提示：**预计算** cos/sin，用 `register_buffer` 存（不参与梯度）。
- 提示：把 d_model 两两配对，每对做 2×2 旋转。可用"偶数位/奇数位拆分"或"旋转半维"技巧。
- 坑：cos/sin 只依赖**位置**，跨 batch 和 head 复用；接口输入 `(batch, seq, heads, d)`。

### 3.8 Scaled Dot-Product Attention（5 分）

- 公式：$\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + \tilde M\right)V$
- mask：`True`=允许注意，`False`=设 $-\infty$（softmax 后变 0）。
- 提示：`scores = Q @ K.transpose(-2,-1) / sqrt(d_k)`；mask 用 `scores.masked_fill(~mask, float('-inf'))`。
- 坑：除以 $\sqrt{d_k}$ 不是 $\sqrt{d_{model}}$；用你写的 softmax（复用）。

### 3.9 Multi-Head Self-Attention（5 分）

- 每个 head 独立算注意力，最后 concat → 过 $W^O$。
- $d_k = d_{model} / h$。
- ★因果 mask：位置 $i$ 只能看 $j \le i$（下三角）。`torch.tril`。
- 提示：Q/K/V 投影可合并成大矩阵一次算，再 reshape 成 `(batch, heads, seq, d_k)`。
- ★state_dict 命名坑：权重 key 是 `attn.q_proj.weight` 等，且 q_proj 是**所有 head 拼接**的 `(num_heads*d_k, d_model)`。你的 reshape 顺序要和这个拼接顺序一致。
- 带 RoPE 版本：在算 scores **之前**、投影 **之后**，对 Q、K 应用 RoPE。

### 3.10 Transformer Block（3 分）

- **Pre-norm** 结构：
  ```
  h1 = x  + MHA(RMSNorm(x))
  h2 = h1 + FFN(RMSNorm(h1))
  ```
- 坑：是 pre-norm（先 norm 再 sublayer），不是 post-norm；两个残差连接别漏。
- state_dict key：`ln1`/`ln2`（两个 RMSNorm）、`attn`、`ffn`。

### 3.11 Transformer LM（3 分）

- 组装：`Embedding → N×TransformerBlock → 最终RMSNorm → LM Head`
- ★state_dict 命名必须**完全对齐**（这样 `load_state_dict` 才成）：
  ```
  token_embeddings.weight
  layers.{i}.ln1.weight
  layers.{i}.attn.{q,k,v,output}_proj.weight
  layers.{i}.ln2.weight
  layers.{i}.ffn.{w1,w2,w3}.weight
  ln_final.weight
  lm_head.weight
  ```
- 提示：用 `nn.ModuleList` 存 N 个 block，命名自动变成 `layers.0`, `layers.1`...
- 坑：输出是 `(batch, seq, vocab_size)` 的 **logits**（未过 softmax）。

---

## 4. 测试命令

```bash
cd ~/learning/cs336-assignment1

# 逐个组件测（按构建顺序）
uv run pytest -k test_linear -v
uv run pytest -k test_embedding -v
uv run pytest -k test_rmsnorm -v
uv run pytest -k test_silu -v
uv run pytest -k test_softmax -v
uv run pytest -k test_swiglu -v          # 或 test_positionwise_feedforward
uv run pytest -k test_rope -v
uv run pytest -k test_scaled_dot_product_attention -v
uv run pytest -k test_multihead_self_attention -v
uv run pytest -k test_transformer_block -v
uv run pytest -k test_transformer_lm -v

# 整个 model 测试文件
uv run pytest tests/test_model.py -v

# 简洁 + 失败短堆栈
uv run pytest tests/test_model.py -q --tb=short
```

---

## 5. 预警坑合集

> 做的时候回来填实际踩到的，现在是**预判**。

| # | 预判坑 | 提醒 |
|---|--------|------|
| 1 | 权重当参数注入，不是自己 init | adapter 里用传入的 `weights`；大模块 `load_state_dict` |
| 2 | state_dict key 对不上 → load 失败 | 你的 Module 内部命名必须逐字匹配 adapter 文档里的 key |
| 3 | Linear 写成 `x @ W`（漏转置）| 行向量约定：`x @ W.T`，W 是 `(d_out, d_in)` |
| 4 | RMSNorm 没 upcast float32 | 先 `.float()` 算，再 cast 回原 dtype |
| 5 | RMSNorm/softmax 归一化维度错 | 都在**最后一维**，记得 `keepdim=True` |
| 6 | softmax 没减 max | 溢出/NaN，先减 `x.max(dim, keepdim=True)` |
| 7 | attention 除错 scale | 是 $\sqrt{d_k}$ 不是 $\sqrt{d_{model}}$ |
| 8 | 忘记因果 mask | `torch.tril`，位置 i 只看 j≤i |
| 9 | RoPE 用到 V 上 | RoPE **只**作用 Q、K |
| 10 | post-norm 写成 pre-norm | 本作业是 **pre-norm**：`x + f(norm(x))` |
| 11 | multi-head reshape 顺序错 | 和 q_proj 的"按 head 拼接"顺序一致 |
| 12 | 输出过了 softmax | LM 输出是 **logits**（原始分数），别 softmax |

---

## 6. 需要先补的 PyTorch 基础

如果对这些不熟，先花点时间过一遍（后面反复用）：

| 主题 | 关键点 |
|------|--------|
| **张量 shape 操作** | `reshape` / `view` / `transpose` / `permute` / `unsqueeze` |
| **einsum** | `torch.einsum('...id,...jd->...ij', a, b)` 表达任意张量乘 |
| **broadcasting** | 维度对齐规则、`keepdim` 的作用 |
| **nn.Module** | `__init__` 里定义参数，`forward` 里算；`nn.Parameter` vs `register_buffer` |
| **高级索引** | `weights[token_ids]` 直接查表 |
| **masked_fill** | `scores.masked_fill(~mask, float('-inf'))` |

> README.md 里 3.2 节有 einsum 示例，可先看那个热身。

---

## 附：心法（一句话记忆）

- **权重是注入的**——你只管"用给定权重算对"，不管初始化。
- **从零件到整机**——linear/softmax 先绿，再往上搭，别一上来写 transformer_lm。
- **shape 是第一大敌**——每步 print `.shape` 确认，比空想快 10 倍。
- **state_dict 命名对齐**——大模块能否 load 成功，全看命名逐字匹配。
- **行向量 + pre-norm + RoPE只给QK + 输出logits**——四个最容易记反的约定。
