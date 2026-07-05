# CS336 Assignment 1 — 做作业步骤指南

> 本文档是 step-by-step 的做题指导，告诉你「做什么、在哪做、怎么跑测试」。
> 不包含任何解题代码——所有代码你自己手写。

---

## 第零步：环境搭建

### 0.1 克隆原始仓库

```bash
git clone https://github.com/stanford-cs336/assignment1-basics.git
cd assignment1-basics
```

### 0.2 安装依赖（用 uv，不用 pip）

本作业用 [uv](https://docs.astral.sh/uv/) 管理依赖，不用 pip。

```bash
# 安装 uv（如果没装过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在项目根目录同步所有依赖
uv sync
```

`uv sync` 会读取 `pyproject.toml`，自动创建 `.venv` 并安装所有依赖（torch, regex, numpy, pytest 等）。

### 0.3 验证环境

```bash
# 确认 Python >= 3.12
uv run python --version

# 确认 torch 可用
uv run python -c "import torch; print(torch.__version__)"

# 跑一下测试（此时全部 FAIL 是正常的，说明环境没问题）
uv run pytest tests/test_model.py -x --tb=no -q 2>&1 | head -5
```

### 0.4 硬件说明

- **CPU 机器**：全部作业都能跑，Section 7 训练会慢（~30 min）但可行
- **Apple Silicon**：用 `device='mps'`，不支持 `torch.compile()`
- **CUDA GPU**：最快，用 `device='cuda'`
- **没有 GPU 也没关系**：BPE (Section 2) 和 Transformer 实现 (Section 3) 完全不需要 GPU

---

## 第一步：理解项目结构

```
assignment1-basics/
├── cs336_basics/              ← 你写代码的地方（目前几乎是空的）
│   ├── __init__.py
│   └── pretokenization_example.py  ← 提供的并行预分词工具函数
├── tests/
│   ├── adapters.py            ← 胶水层，你必须填写的接口
│   ├── test_train_bpe.py      ← BPE 训练测试
│   ├── test_tokenizer.py      ← BPE 编码/解码测试
│   ├── test_model.py          ← Transformer 模型测试
│   ├── test_nn_utils.py       ← cross-entropy / gradient clipping / LR schedule 测试
│   ├── test_optimizer.py      ← AdamW 测试
│   ├── test_data.py           ← data loader 测试
│   └── test_serialization.py  ← checkpoint 测试
├── pyproject.toml
├── CLAUDE.md / AGENTS.md
└── README.md
```

**核心工作流**：

1. 在 `cs336_basics/` 下创建 `.py` 文件，实现功能
2. 在 `tests/adapters.py` 中填写对应的 adapter 函数，调用你的实现
3. 用 `uv run pytest` 跑测试

---

## 第二步：Section 2 — BPE 分词器（7 题，42 分）

这是分值最高的 section，建议最先做。

### 2.1 书面题 (unicode1, unicode2)

这两道是概念题，不需要写代码。在 Python 解释器里试一下 `chr(0)`、UTF-8 编码等，把答案写到 `writeup.pdf`。

### 2.2 实现 BPE 训练 (train_bpe) — 15 分，核心

**创建文件**：在 `cs336_basics/` 下创建你的 BPE 实现文件（比如 `tokenizer.py`）。

**你需要实现**：一个 `train_bpe()` 函数，接收文本文件路径、目标词汇表大小、特殊 token 列表，返回 (vocab, merges)。

**实现步骤**：
1. 读取文本文件
2. 用 GPT-2 正则表达式做预分词（`regex` 包，不是 `re`）
3. 把每个预分词 chunk 转为 UTF-8 字节序列
4. 统计所有相邻 byte pair 的频率
5. 循环：找最高频 pair → 合并 → 更新频率 → 加入词汇表，直到达到目标大小
6. 返回 vocab 和 merges

**连接 adapter**：编辑 `tests/adapters.py` 中的 `run_train_bpe()` 函数，import 并调用你的实现。

**跑测试**：

```bash
# 跑 BPE 训练测试（含正确性和速度测试）
uv run pytest tests/test_train_bpe.py -v

# 只跑正确性测试
uv run pytest tests/test_train_bpe.py -k "not speed"

# 只跑速度测试（必须 1.5 秒内完成）
uv run pytest tests/test_train_bpe.py -k "speed"
```

**速度测试要点**：朴素实现（每轮重新统计所有 pair）会超时。你需要用增量更新——每次合并后只更新受影响的邻居 pair 的计数。

### 2.3 实现 Tokenizer 类 (tokenizer) — 15 分

**你需要实现**：一个 `Tokenizer` 类，包含 `encode()`、`decode()`、`encode_iterable()`、`from_files()` 方法。

**连接 adapter**：填写 `tests/adapters.py` 中的 `get_tokenizer()` 函数。

**跑测试**：

```bash
uv run pytest tests/test_tokenizer.py -v
```

### 2.4 实验题 (train_bpe_tinystories, train_bpe_expts_owt, tokenizer_experiments)

这几题需要下载数据集并跑实验，结果写到 `writeup.pdf`。

---

## 第三步：Section 3 — Transformer 模型（11 题，27 分）

按**从底层到顶层**的顺序依次实现。每个模块实现后立刻跑对应测试。

### 实现顺序和测试命令

| 顺序 | 模块 | 要点 | 测试命令 |
|------|------|------|---------|
| 1 | SiLU 激活函数 | x * sigmoid(x) | `uv run pytest -k test_silu` |
| 2 | Softmax | 记得减最大值保数值稳定 | `uv run pytest -k test_softmax` |
| 3 | Linear | 无 bias，权重 shape (d_out, d_in) | `uv run pytest -k test_linear` |
| 4 | Embedding | 查表操作 | `uv run pytest -k test_embedding` |
| 5 | RMSNorm | upcast 到 float32 再 cast 回来 | `uv run pytest -k test_rmsnorm` |
| 6 | SwiGLU FFN | 三个权重矩阵 W1, W2, W3 | `uv run pytest -k test_positionwise_feedforward` |
| 7 | RoPE | 预计算 cos/sin，用 register_buffer | `uv run pytest -k test_rope` |
| 8 | Scaled Dot-Product Attention | mask 处 → -inf | `uv run pytest -k test_scaled_dot_product_attention` |
| 9 | Multi-Head Self-Attention (无 RoPE) | 合并 QKV 投影矩阵 | `uv run pytest -k "test_multihead_self_attention and not rope"` |
| 10 | Multi-Head Self-Attention (带 RoPE) | RoPE 只加在 Q 和 K 上 | `uv run pytest -k test_multihead_self_attention_with_rope` |
| 11 | Transformer Block | Pre-norm: Norm → Attn → 残差 → Norm → FFN → 残差 | `uv run pytest -k test_transformer_block` |
| 12 | Transformer LM | Embedding + N 层 Block + 最终 Norm + LM Head | `uv run pytest -k test_transformer_lm` |

**参数初始化**：所有 Linear / Embedding 权重用 `trunc_normal_(mean=0, std=0.02)`，RMSNorm gain 初始化为全 1。

**连接 adapter**：每实现一个模块就去 `tests/adapters.py` 填写对应函数（`run_linear`, `run_embedding`, `run_rmsnorm` 等）。

**全量测试**：

```bash
uv run pytest tests/test_model.py -v
```

### 3.13 资源核算 (transformer_accounting) — 5 分，书面题

用 GPT-2 XL 配置手算参数量、内存、FLOPs。答案写到 `writeup.pdf`。

---

## 第四步：Section 4 — 训练组件（6 题，8 分）

### 实现顺序

| 顺序 | 模块 | 测试命令 |
|------|------|---------|
| 1 | Cross-Entropy Loss | `uv run pytest -k test_cross_entropy` |
| 2 | Gradient Clipping | `uv run pytest -k test_gradient_clipping` |
| 3 | AdamW Optimizer | `uv run pytest -k test_adamw` |
| 4 | Cosine LR Schedule | `uv run pytest -k test_lr_schedule` |

**全量测试**：

```bash
uv run pytest tests/test_nn_utils.py tests/test_optimizer.py -v
```

**书面题**：learning_rate_tuning (1分) 和 adamw_accounting (2分) 的答案写到 `writeup.pdf`。

---

## 第五步：Section 5 — 训练循环（3 题，7 分）

### 实现顺序

| 顺序 | 模块 | 测试命令 |
|------|------|---------|
| 1 | Data Loader (get_batch) | `uv run pytest -k test_get_batch` |
| 2 | Save / Load Checkpoint | `uv run pytest -k test_checkpointing` |
| 3 | Training Loop 脚本 | 没有自动测试，自己运行验证 |

**全量测试**：

```bash
uv run pytest tests/test_data.py tests/test_serialization.py -v
```

**训练脚本**：创建一个独立的训练脚本（比如 `cs336_basics/train.py`），把模型、优化器、数据加载、checkpoint 串起来。这个没有 pytest 测试，你自己跑通就行。

---

## 第六步：Section 6 — 文本生成（1 题，3 分）

实现带 temperature scaling 和 top-p sampling 的解码函数。这个没有 pytest 测试，用训练好的模型生成文本来验证。

---

## 第七步：Section 7 — 实验（10 题，22 分）

这一步需要真正训练模型并记录结果。

### 实验顺序

1. **TinyStories 训练** — 用默认超参数跑，目标 validation loss ≤ 1.45
2. **Batch Size 消融** — 至少 3 个 batch size，固定总 token 数
3. **文本生成** — 展示生成结果
4. **4 个架构消融** — 分别去掉 RMSNorm / 换 Post-norm / 去掉 RoPE / 换 SiLU-only FFN
5. **OpenWebText 实验** — 同架构换数据集
6. **排行榜** — 45 分钟内跑出最低 validation loss

### 实验技巧

- 先在 TinyStories 的小子集上验证代码正确性
- 用 SGD 先跑几步确认 loss 在下降
- 打印梯度范数确认没有梯度爆炸/消失
- 如果 loss 不下降，逐个把你的模块替换为 PyTorch 标准实现来定位 bug

---

## 汇总：文件创建清单

你在 `cs336_basics/` 下大致需要创建这些文件（文件名和组织方式随意）：

| 文件 | 内容 | 对应 Section |
|------|------|-------------|
| `tokenizer.py` | BPE 训练 + Tokenizer 类 | Section 2 |
| `model.py` | Linear / Embedding / RMSNorm / SwiGLU / RoPE / Attention / TransformerBlock / TransformerLM | Section 3 |
| `nn_utils.py` | SiLU / Softmax / CrossEntropy / GradientClipping / LR Schedule | Section 3-4 |
| `optimizer.py` | AdamW | Section 4 |
| `data.py` | get_batch / memmap 数据加载 | Section 5 |
| `checkpoint.py` | save / load checkpoint | Section 5 |
| `decode.py` | temperature + top-p 解码 | Section 6 |
| `train.py` | 训练脚本 | Section 5+7 |

**你也可以把所有东西写到一个文件里，或者拆成更多文件——只要 adapters.py 能 import 到就行。**

---

## 汇总：测试命令速查

```bash
# 全部测试
uv run pytest tests/ -v

# 按 section 测
uv run pytest tests/test_train_bpe.py -v           # Section 2 BPE 训练
uv run pytest tests/test_tokenizer.py -v            # Section 2 Tokenizer
uv run pytest tests/test_model.py -v                # Section 3 全部模型
uv run pytest tests/test_nn_utils.py -v             # Section 4 loss/grad clip/LR
uv run pytest tests/test_optimizer.py -v            # Section 4 AdamW
uv run pytest tests/test_data.py -v                 # Section 5 data loader
uv run pytest tests/test_serialization.py -v        # Section 5 checkpoint

# 单个模块测试（用 -k 过滤）
uv run pytest -k test_silu -v
uv run pytest -k test_rmsnorm -v
uv run pytest -k test_rope -v
# ...以此类推
```

---

## 重要提醒

### 不能用的东西

- `torch.nn` 里的层实现（Linear, Embedding, LayerNorm, Softmax 等）
- `torch.nn.functional` 里的函数
- `torch.optim` 里的优化器（除了 `Optimizer` 基类）
- 唯一例外：`torch.nn.Parameter`、`torch.nn.Module`、`torch.nn.ModuleList` 等容器类

### 可以用的东西

- `torch.tensor`, `torch.matmul`, `torch.exp`, `torch.log` 等基本运算
- `torch.einsum`
- `torch.nn.Parameter` 和 `torch.nn.Module` (容器)
- `torch.nn.init.trunc_normal_` (参数初始化)
- `numpy`, `regex`, `tqdm`, `wandb`
- 任何非 `torch.nn` / `torch.nn.functional` / `torch.optim` 的 PyTorch API

### 建议的做题节奏

1. **Week 1**：Section 2 (BPE) + Section 3 前半 (Linear → RMSNorm → SwiGLU)
2. **Week 2**：Section 3 后半 (RoPE → Attention → Transformer) + Section 4
3. **Week 3**：Section 5 + 6 + 7 (训练和实验)
