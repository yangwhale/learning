# BPE 第一个功能跑通 — Cheat Sheet

> 从零到第一个绿色 PASSED 的完整流程。
> 只走通 `train_bpe` 的正确性测试，不做速度优化。

---

## Step 1：创建你的代码文件

在 **Stanford 原始仓库**（不是翻译仓库）里操作：

```bash
cd ~/learing/cs336-assignment1
```

创建文件：

```bash
touch cs336_basics/tokenizer.py
```

现在目录结构变成：

```
cs336-assignment1/
├── cs336_basics/
│   ├── __init__.py
│   ├── pretokenization_example.py
│   └── tokenizer.py              ← 新建的，你写代码的地方
└── tests/
    ├── adapters.py                ← 待会要改的胶水文件
    ├── test_train_bpe.py          ← 你要通过的测试
    └── ...
```

---

## Step 2：在 tokenizer.py 里写函数骨架

打开 `cs336_basics/tokenizer.py`，写入函数签名（这个签名在作业文档里给出了）：

```python
import regex as re


def train_bpe(
    input_path: str,
    vocab_size: int = 256,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器。

    Args:
        input_path: 训练语料文本文件的路径
        vocab_size: 目标词汇表大小（必须 >= 256）
        special_tokens: 特殊 token 列表

    Returns:
        vocab: dict[int, bytes] — token ID → token 字节串
        merges: list[tuple[bytes, bytes]] — 合并列表（按训练顺序）
    """
    # ==========================================
    # 你的实现写在这里
    # ==========================================
    #
    # 大致步骤提示（具体代码你自己写）：
    #
    # 1. 初始化 vocab：0-255 的 256 个单字节
    #    再把 special_tokens 加进去（如果有的话）
    #
    # 2. 读取 input_path 文件内容
    #
    # 3. 用 GPT-2 正则做预分词，把文本切成 chunks
    #    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #    注意：用 regex 包（import regex as re），不是标准库 re
    #
    # 4. 把每个 chunk 转成 UTF-8 字节序列（tuple of bytes）
    #
    # 5. 统计所有相邻 pair 的频率
    #
    # 6. 循环直到 vocab 达到 vocab_size：
    #    - 找频率最高的 pair（平局时取字典序最大的）
    #    - 合并这个 pair，生成新 token
    #    - 加入 vocab 和 merges 列表
    #    - 更新语料中的 token 序列
    #
    # 7. return vocab, merges
    #
    raise NotImplementedError("还没写完！")
```

---

## Step 3：把你的函数连接到 adapter

打开 `tests/adapters.py`，找到 `run_train_bpe` 函数（在文件最底部），把它改成：

```python
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    from cs336_basics.tokenizer import train_bpe
    return train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
```

就这么简单——删掉 `raise NotImplementedError`，换成 import 你的函数然后调用。

---

## Step 4：跑测试看报错

```bash
# 只跑正确性测试，跳过速度测试
uv run pytest tests/test_train_bpe.py -k "test_train_bpe and not speed and not special" -v --tb=short
```

此时你会看到报错，因为 Step 2 里函数体还是 `raise NotImplementedError`。

接下来就是你自己写实现的过程了——把 Step 2 里的 `raise NotImplementedError` 替换成你的代码。

---

## Step 5：写你的实现（这一步你自己来）

回到 `cs336_basics/tokenizer.py`，按 Step 2 注释里的 7 个步骤把代码写出来。

**先用最朴素的方式写**，不考虑速度，只要结果对就行。

**调试技巧**：

```bash
# 跑测试，允许 print 输出
uv run pytest tests/test_train_bpe.py -k "test_train_bpe and not speed and not special" -v -s

# 也可以直接用 Python 交互式测试
uv run python -c "
from cs336_basics.tokenizer import train_bpe
vocab, merges = train_bpe('tests/fixtures/corpus.en', vocab_size=300, special_tokens=['<|endoftext|>'])
print(f'vocab size: {len(vocab)}')
print(f'merges count: {len(merges)}')
print(f'first 5 merges: {merges[:5]}')
"
```

---

## Step 6：看到绿色 PASSED

当你的实现正确后，跑测试会看到：

```
tests/test_train_bpe.py::test_train_bpe PASSED
```

---

## 完整流程图

```
 你的实现文件                     胶水文件                       测试文件
 cs336_basics/tokenizer.py       tests/adapters.py             tests/test_train_bpe.py
┌────────────────────┐          ┌────────────────────┐        ┌────────────────────────┐
│                    │          │                    │        │                        │
│  def train_bpe(    │◀─import──│  def run_train_bpe │◀─调用──│  def test_train_bpe(): │
│    input_path,     │          │    ...             │        │    vocab, merges =     │
│    vocab_size,     │          │    from cs336_...  │        │      run_train_bpe(    │
│    special_tokens  │          │      import ...    │        │        "corpus.en",    │
│  ):                │          │    return train_..│        │        500, ...)       │
│    # 你的代码       │──return──▶│                    │─return─▶│                        │
│    ...             │          │                    │        │    assert merges ==    │
│    return vocab,   │          └────────────────────┘        │      reference_merges  │
│           merges   │                                        │    assert vocab keys   │
└────────────────────┘                                        │      == reference keys │
                                                              └────────────────────────┘
```

---

## 测试文件读的什么数据？

| 测试 | 输入文件 | vocab_size | special_tokens |
|------|---------|------------|----------------|
| test_train_bpe | `tests/fixtures/corpus.en` (133 KB 英文语料) | 500 | `["<\|endoftext\|>"]` |
| test_train_bpe_speed | 同上 | 500 | `["<\|endoftext\|>"]` |
| test_train_bpe_special_tokens | `tests/fixtures/tinystories_sample_5M.txt` | 1000 | `["<\|endoftext\|>"]` |

正确性测试把你的 vocab 和 merges 跟 `tests/fixtures/train-bpe-reference-*` 里的标准答案做对比。

---

## 常见错误排查

| 症状 | 可能原因 |
|------|---------|
| `ModuleNotFoundError: No module named 'cs336_basics.tokenizer'` | 文件没创建在 `cs336_basics/` 目录下，或者文件名拼错了 |
| `ImportError: cannot import name 'train_bpe'` | 函数名拼错了，或者没在 tokenizer.py 里定义 |
| merges 数量不对 | 检查循环终止条件：应该循环 `vocab_size - 256 - len(special_tokens)` 次 |
| merges 顺序不对 | 平局时要选字典序最大的 pair：`max(pairs, key=lambda p: (count[p], p))` |
| vocab keys 不匹配 | special tokens 的 ID 分配方式：先 0-255 字节，然后 special tokens，最后 merges 产生的新 token |
| `regex` 报错 | 确认 `import regex as re`，不是 `import re` |
