# CS336 Assignment 1 - Part 2: BPE 分词器训练

> 对应原始 PDF Section 2.3-2.5 (pages 5-9)

## 2.3 子词分词 (Subword Tokenization)

在字节级表示的基础上，我们希望找到一种分词方案，在**纯字节级分词**和**词级分词**之间取得平衡：

- **字节级分词**的问题：序列过长。每个 Unicode 字符可能需要多个字节表示，导致序列长度大幅增加。更长的序列意味着更高的计算成本（Transformer 的 self-attention 复杂度与序列长度的平方成正比），也使模型更难学习长距离依赖关系。
- **词级分词**的问题：词汇外 (OOV) 问题。对于未在训练数据中出现的词，模型无法处理。此外，词级分词无法捕捉词内的共享子结构（如 "learning" 和 "learned" 共享词根 "learn"）。

**子词分词** (subword tokenization) 是两者的折中方案：将常见的字符序列合并为单个 token，同时保留对罕见序列的字节级表示能力。

我们将实现 **Byte Pair Encoding (BPE)** 分词器。BPE 最初是一种数据压缩算法 (P. Gage [5])，后被 Sennrich et al. [3] 引入到 NLP 中用于子词分词。BPE 的核心思想是迭代地将训练语料中最常见的相邻 token 对合并为新的 token，从而构建子词词汇表。

---

## 2.4 BPE 分词器训练

BPE 分词器的训练过程包含三个主要步骤：

### 步骤 1：词汇表初始化

词汇表 (vocabulary) 是从 bytestring token 到 integer ID 的一一映射。初始词汇表包含 256 个条目，对应 0 到 255 的所有单字节值。

### 步骤 2：预分词 (Pre-tokenization)

预分词将输入文本分割成较小的块，BPE 合并只在每个块内部进行，不会跨块边界。

**原始 BPE** 简单地按空格分割文本。

**现代分词器**（如 GPT-2, Radford et al. [6]）使用基于正则表达式的预分词器，能更好地处理标点符号、缩写等：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

测试预分词器：

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

注意空格是如何被附加到后续词的前面的，而缩写（如 `'ll`）被单独分出。

> **实现提示**：使用 `re.finditer` 而不是 `re.findall`，可以避免将所有预分词的词存储在内存中，改为惰性迭代。

### 步骤 3：计算 BPE 合并

在完成预分词后，迭代执行以下操作直到达到目标词汇表大小：

1. 统计训练语料中所有相邻 token 对的出现频率
2. 选择出现频率最高的 token 对
3. 将该 token 对合并为一个新 token，添加到词汇表中
4. 在训练语料中将所有该 token 对的出现替换为新 token

**平局处理**：当多对 token 具有相同频率时，选择字典序 (lexicographic order) 更大的对：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

### 特殊 Token

某些特殊 token（如 `<|endoftext|>`）必须在分词过程中保留为单个 token，不能被拆分或合并。这些特殊 token 被添加到词汇表中，每个有固定的 token ID。

---

## Example (bpe_example): BPE 训练示例

以下是一个 Sennrich et al. [3] 风格的 BPE 训练过程示例。

**训练语料库**（每个词后面的数字表示该词在语料中出现的频率）：

| 词 | 频率 |
|---|---|
| `l o w` | 5 |
| `l o w e r` | 2 |
| `w i d e s t` | 3 |
| `n e w e s t` | 6 |

**初始词汇表**：`{d, e, i, l, n, o, r, s, t, w}`（所有出现过的字符）

**合并过程**：

**第 1 轮**：统计所有相邻字符对的频率：
- `(e, s)`: 出现 9 次（`w i d e s t` × 3 + `n e w e s t` × 6）
- `(s, t)`: 出现 9 次
- `(e, w)`: 出现 6 次
- `(n, e)`: 出现 6 次
- ...

`(e, s)` 和 `(s, t)` 并列最高频（9 次）。按字典序，`(s, t)` > `(e, s)`，因此选择合并 `(s, t) → st`。

合并后：

| 词 | 频率 |
|---|---|
| `l o w` | 5 |
| `l o w e r` | 2 |
| `w i d e st` | 3 |
| `n e w e st` | 6 |

词汇表：`{d, e, i, l, n, o, r, s, t, w, st}`

**第 2 轮**：最高频对 `(e, st)` 出现 9 次，合并 `(e, st) → est`。

合并后：

| 词 | 频率 |
|---|---|
| `l o w` | 5 |
| `l o w e r` | 2 |
| `w i d est` | 3 |
| `n e w est` | 6 |

词汇表：`{d, e, i, l, n, o, r, s, t, w, st, est}`

**第 3 轮**：最高频对 `(l, o)` 出现 7 次，合并 `(l, o) → lo`。

合并后：

| 词 | 频率 |
|---|---|
| `lo w` | 5 |
| `lo w e r` | 2 |
| `w i d est` | 3 |
| `n e w est` | 6 |

词汇表：`{d, e, i, l, n, o, r, s, t, w, st, est, lo}`

**第 4 轮**：最高频对 `(lo, w)` 出现 7 次，合并 `(lo, w) → low`。

合并后：

| 词 | 频率 |
|---|---|
| `low` | 5 |
| `low e r` | 2 |
| `w i d est` | 3 |
| `n e w est` | 6 |

词汇表：`{d, e, i, l, n, o, r, s, t, w, st, est, lo, low}`

**第 5 轮**：`(n, e)` 和 `(e, w)` 都出现 6 次。按字典序 `(n, e)` > `(e, w)`，合并 `(n, e) → ne`。

合并后：

| 词 | 频率 |
|---|---|
| `low` | 5 |
| `low e r` | 2 |
| `w i d est` | 3 |
| `ne w est` | 6 |

词汇表：`{d, e, i, l, n, o, r, s, t, w, st, est, lo, low, ne}`

**第 6 轮**：`(ne, w)` 出现 6 次，合并 `(ne, w) → new`。

合并后：

| 词 | 频率 |
|---|---|
| `low` | 5 |
| `low e r` | 2 |
| `w i d est` | 3 |
| `new est` | 6 |

词汇表：`{d, e, i, l, n, o, r, s, t, w, st, est, lo, low, ne, new}`

此过程可以继续，直到词汇表达到目标大小。

---

## 2.5 BPE 分词器训练实验

### 实现优化建议

**并行预分词**：使用 Python 的 `multiprocessing` 模块并行处理预分词。将输入文本分成多个 chunk，每个 chunk 由一个进程处理。注意 chunk 边界应在 special token 处分割，以确保 special token 不会被拆分。

**预分词前移除 special tokens**：在对文本进行正则表达式预分词之前，先用 `re.split` 和 `re.escape` 将 special tokens 从文本中分离出来。

**优化合并步骤**：朴素实现中，每次合并后需要重新遍历整个语料库统计 pair 频率。可以通过增量更新计数来优化：只更新受合并影响的 pair 的计数，而不是重新统计所有 pair。

### Low-Resource Tips

**Profiling**：使用 `cProfile` 或 `py-spy` 找出代码中的性能瓶颈。

```bash
# cProfile 使用示例
python -m cProfile -s cumulative your_script.py

# py-spy 使用示例（需要 pip install py-spy）
py-spy top -- python your_script.py
```

**Downscaling**：在调试阶段，使用小数据集（如 TinyStories validation set，约 22,000 个文档）而不是完整数据集。这样可以快速迭代和验证实现的正确性。

---

### Problem (train_bpe): BPE Tokenizer Training (15 points)

实现 BPE 分词器训练函数。

**函数接口**：

```python
def train_bpe(
    input_path: str,
    vocab_size: int = 256,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器。

    Args:
        input_path: str
            训练语料文本文件的路径。
        vocab_size: int
            目标词汇表大小。必须 >= 256。
            默认 256（不进行任何合并）。
        special_tokens: list[str] | None
            特殊 token 列表（如 ["<|endoftext|>"]）。
            这些 token 在分词时保留为整体，不被拆分。
            默认 None。

    Returns:
        vocab: dict[int, bytes]
            词汇表，从 token ID (int) 到 token 值 (bytes) 的映射。
        merges: list[tuple[bytes, bytes]]
            BPE 合并列表，按训练时的合并顺序排列。
            每个元素是一个 (bytes, bytes) 元组，表示被合并的两个 token。
    """
```

**测试方法**：

```bash
# 通过 adapter 运行
adapters.run_train_bpe

# 运行测试
uv run pytest tests/test_train_bpe.py
```

**可选加速**：如果 Python 实现太慢，可以考虑用 C++ 或 Rust 实现核心合并逻辑，然后通过 Python 绑定调用。但纯 Python 实现如果经过适当优化，应该能在合理时间内完成。

**正则引擎兼容性说明**：GPT-2 风格的预分词正则表达式使用了 Unicode 属性（如 `\p{L}`、`\p{N}`），Python 标准库的 `re` 模块不支持这些。推荐使用以下两种方案之一：
- **`regex` 包**：`pip install regex`，API 与 `re` 兼容，支持 Unicode 属性
- **Oniguruma 引擎**：通过 `onigurumacffi` 包使用，GPT-2 原始实现使用此引擎

---

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

**(a)** 在 TinyStories 数据集上训练 BPE 分词器：
- 词汇表大小 (vocab_size): 10,000
- 特殊 token: `["<|endoftext|>"]`
- 将训练好的词汇表和合并列表序列化到磁盘
- 报告：训练时间、峰值内存使用量、最长 token（按字节数）

**资源限制**：
- 时间：<= 30 分钟（不使用 GPU）
- 内存：<= 30 GB RAM

**提示**：
- 使用 `multiprocessing` 并行化预分词
- TinyStories 数据集中的文档以 `<|endoftext|>` 分隔
- 注意对 special tokens 的特殊处理

**Deliverable**: 序列化后的词汇表和合并列表文件，以及训练时间、内存使用、最长 token 的报告。

**(b)** Profile 你的代码，哪部分最耗时？

**Deliverable**: 一到两句话回答。

---

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

**(a)** 在 OpenWebText (OWT) 数据集上训练 BPE 分词器：
- 词汇表大小 (vocab_size): 32,000
- 特殊 token: `["<|endoftext|>"]`
- 将训练好的词汇表和合并列表序列化到磁盘
- 报告最长 token（按字节数）

**资源限制**：
- 时间：<= 12 小时
- 内存：<= 100 GB RAM

**Deliverable**: 序列化后的词汇表和合并列表文件，以及最长 token 的报告。

**(b)** 对比 TinyStories 分词器和 OWT 分词器。它们的词汇表有哪些有趣的差异？

**Deliverable**: 一到两句话回答。
