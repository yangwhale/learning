# CS336 Assignment 1 · Section 2：BPE 分词器 — 作业笔记

> 从零实现字节级 BPE 分词器的完整过程笔记。
> 包含：算法原理、每一步代码、踩过的坑、测试结果。
> **状态：全部完成，`27 passed, 1 xfailed` ✅**

---

## 目录

- [0. 概览：这部分在做什么](#0-概览这部分在做什么)
- [1. 环境 & 文件结构](#1-环境--文件结构)
- [2. 核心概念（一定要先懂）](#2-核心概念一定要先懂)
- [3. train_bpe：训练 BPE](#3-train_bpe训练-bpe)
- [4. Tokenizer：编码 & 解码](#4-tokenizer编码--解码)
- [5. 完整代码](#5-完整代码)
- [6. 测试](#6-测试)
- [7. 踩坑合集（复习重点）](#7-踩坑合集复习重点)

---

## 0. 概览：这部分在做什么

分词器 = 文本 ↔ 数字（token ID）的双向翻译器。模型只认数字，所以：

```
训练模型前:  文本 --encode--> token IDs --> 喂给模型
模型输出后:  token IDs --decode--> 文本 --> 给人看
```

这一节要实现两大块：

| 块 | 函数/类 | 作用 |
|----|---------|------|
| **训练** | `train_bpe()` | 从语料**学出**一套合并规则（vocab + merges）|
| **使用** | `Tokenizer` 类 | 用学好的规则做 `encode` / `decode` |

---

## 1. 环境 & 文件结构

在 Stanford 原始仓库操作（不是翻译仓库）：

```bash
cd ~/learning/cs336-assignment1
```

```
cs336-assignment1/
├── cs336_basics/
│   ├── __init__.py
│   ├── pretokenization_example.py   ← Stanford 给的分块示例（拷了 find_chunk_boundaries）
│   └── tokenizer.py                 ← 我们写代码的地方
└── tests/
    ├── adapters.py                   ← 胶水层：把测试接到我们的实现
    ├── test_train_bpe.py             ← 训练测试（3 个）
    └── test_tokenizer.py             ← 编解码测试（25 个）
```

**三层调用链**（测试文件不能改，靠 adapter 转接）：

```
test_*.py  →  adapters.py (run_train_bpe / get_tokenizer)  →  tokenizer.py (我们的实现)
```

adapter 改法（删掉 `raise NotImplementedError`，换成 import + 调用）：

```python
# tests/adapters.py
def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    from cs336_basics.tokenizer import train_bpe
    return train_bpe(input_path=str(input_path), vocab_size=vocab_size,
                     special_tokens=special_tokens)

def get_tokenizer(vocab, merges, special_tokens=None):
    from cs336_basics.tokenizer import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)
```

---

## 2. 核心概念（一定要先懂）

### 2.1 字节（byte） vs 整数（int） vs 字节串（bytes）

- **1 byte = 8 bit**，能表示 **0~255** 共 256 个值。
- Python 里 `int`（数值 `65`）和 `bytes`（字节序列 `b'A'`）**是不同类型**，`65 != b'A'`。
- `bytes` 是**容器**，里面装 0~255 的 int：`list(b' t') == [32, 116]`。

**为什么 vocab 用 `dict[int, bytes]`**：token 内容是**字节序列**（可能多字节），单个 int 装不下。

```python
vocab = {
    65:  b'A',            # 单字节 token
    259: b' the',         # 多字节 token = 4 字节 [32,116,104,101]
    300: b'\xe4\xbd\xa0', # "你" = 3 字节 UTF-8
}
```

### 2.2 UTF-8：256 个字节怎么表示 15 万字符？

**变长编码**：一个字符用 1~4 个字节。

| 字符 | 字节数 | 例子 |
|------|--------|------|
| ASCII（英文/数字/标点）| 1 | `A` = `[65]` |
| 拉丁扩展（é, ü）| 2 | `é` = `[195,169]` |
| 中文/日文/韩文 | 3 | `你` = `[228,189,160]` |
| emoji | 4 | `🙃` = `[240,159,153,131]` |

**关键**：BPE 在**字节层面**工作，所以只需 256 个基础 token 就能表示任何语言——多字节字符靠 BPE 逐步合并字节还原。

### 2.3 BPE 核心：每次只合并「最高频的相邻 pair」

- **滑动窗口永远是 2**：每次只合并两个相邻 token，从不一次合 3 个。
- 一个中文字（3 字节）要**两轮**才能合成一个 token：`[b1,b2,b3] → [b1b2, b3] → [b1b2b3]`。
- **铁律**：合并出的新 token 频率 **≤** 任一组成部分。越长越稀有（数学保证）。

### 2.4 分层构建（理解 encode 的关键）

高频组合是逐层往上盖楼：

```
第一层: t + h → th      （th 频率超高，rank 小，先学）
第二层: th + e → the    （the 频率也高，rank 大，后学）
```

- `(th, e)` 这个 pair **只有在 `(t,h)` 合并之后才可能出现**，所以 "the" 天生 rank 比 "th" 大。
- **"the" 建立在 "th" 的肩膀上**——"th" 不是终点，是通往 "the" 的踏板。

---

## 3. train_bpe：训练 BPE

### 3.1 七步算法

```
1. 初始化 vocab：0~255 单字节 + special tokens
2. 读训练文本
3. 用 GPT-2 正则预分词（切成 chunk）
4. 每个 chunk 转 UTF-8 字节序列，统计词频 word_freqs
5. 统计所有相邻 pair 频率（+ 建倒排索引）
6. 循环：找最高频 pair → 合并 → 更新，直到 vocab 达到目标大小
7. 返回 vocab + merges
```

### 3.2 GPT-2 预分词正则（为什么这么复杂）

```python
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

**目的**：把文本切成语义单元（词/数字/标点），同时**保留空格位置**，让 BPE 只在合理边界内合并。

6 个子模式：

| 模式 | 匹配 | 例子 |
|------|------|------|
| `'(?:[sdmt]\|ll\|ve\|re)` | 英文缩写 | `don't` → `don` + `'t` |
| ` ?\p{L}+` | 字母序列（含前导空格）| `Hello world` → `Hello` + ` world` |
| ` ?\p{N}+` | 数字序列 | ` 123` |
| ` ?[^\s\p{L}\p{N}]+` | 标点/符号 | `!!!` |
| `\s+(?!\S)` | 行尾空白 | 保留格式 |
| `\s+` | 兜底空白 | — |

**必须用 `import regex as re`（不是标准库 `re`）**——要支持 `\p{L}` `\p{N}` Unicode 属性。

**中文的坑**：中文没空格，`\p{L}+` 会把整句匹配成一个超长 chunk（`"你好世界"` → `['你好世界']`）。这也是中文模型常用 SentencePiece 的原因。

### 3.3 关键数据结构：word 存 `tuple[bytes, ...]` 不是 `tuple[int, ...]`

**踩过的坑**（合并死循环）：如果 word 存 int（字节值），合并时新 token 的多字节会被拆回单字节。

```python
# ❌ 错：tuple of int
word = tuple(chunk.encode('utf-8'))          # (116, 104, 101)

# ✅ 对：tuple of bytes
word = tuple(bytes([b]) for b in chunk.encode('utf-8'))  # (b't', b'h', b'e')
```

### 3.4 平局规则

频率相同时，选**字典序最大**的 pair：

```python
best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
```

`(count, pair)` 元组比较：先比 count，平了比 pair 本身。**必须这样才能和标准答案一致。**

### 3.5 性能优化：增量更新（O(n²) → 快 ~9x）

**朴素版慢**：每轮都全量重扫语料统计所有 pair。243 轮 × 全语料 = O(n²)。

**核心洞察**：合并 `(A,B)→AB` 后，**只有含 `A B` 的词变了**，凭什么全量重算？

**三个数据结构**（`pair_freqs` 从"每轮重建"变成"建一次 + 增量改"）：

| 结构 | 作用 |
|------|------|
| `word_freqs` | 词 → 频率 |
| `pair_freqs` | pair → 总频率（**跨轮存活**）|
| `pair_to_words` | pair → 含它的词集合（**倒排索引**，快速定位受影响的词）|

**每次合并只更新受影响的词**：撤销旧 pair 贡献 → 生成新词 → 加新 pair 贡献。

```
慢版: 243 × 50000 = 1200 万次操作
快版: 50000 + 243 × 几十 ≈ 6 万次操作   → ~200x 理论差距
实测: 1.93s → 0.22s（纯训练），~9x
```

**增量更新的坑**：
1. 重叠 pair（`a a a` 里 `(a,a)` 出现 2 次）：计数按出现次数减，倒排索引按去重后的 pair 减。
2. 减到 0 的残留：选 best_pair 后加 `<= 0` 兜底 break。

### 3.6 special token 保护（防污染）

**问题**：训练文本里的 `<|endoftext|>` 被正则切碎成 `<|`、`endoftext`、`|>`，碎片被 BPE 合并 → vocab 冒出 `b'<|'` 污染。

**注意区分两件事**：
- **① 注册**：special token 进 vocab（初始化时做，本来就有）
- **② 保护**：预分词前用 `re.split` 把 special token 摘出去，不进 BPE 统计（**要加的**）

```python
# 按长度降序（防短 token 是长 token 前缀时切坏）
sorted_specials = sorted(special_tokens, key=len, reverse=True)
# 捕获组 (…) 让 split 保留分隔符
split_pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
segments = re.split(split_pattern, text)
# special 段跳过，普通段才做 GPT-2 正则预分词
```

### 3.7 分块 + 并行（大文件工程）

**两个不同问题，别混**：

| 问题 | 谁解决 |
|------|--------|
| **内存**（几十 GB 读不进来）| `find_chunk_boundaries` 分块 + `multiprocessing` |
| **污染**（`<\|` 进 vocab）| `re.split` 摘 special token（见 3.6）|

`find_chunk_boundaries` **不删 special token**——它只是找**安全切点**（落在 special token 处，不切坏词/字符），块内 special token 原样保留，仍要靠 3.6 保护。

**并行架构**：
```
find_chunk_boundaries 找切点 → 每段 (start,end) 一个子进程统计 → 合并各块 word_freqs
```

- `_process_chunk` 必须是**模块级函数**（才能被 pickle 传给子进程）。
- 合并词频就是最早讨论的"n 份统计加一起"：`word_freqs[word] += freq`。
- **默认串行**（`num_processes=1`）：小文件用并行反而慢（进程池开销）。
- 无 special token 不能并行（没安全切点）。

**为什么 4 进程只快 ~2x**（Amdahl 定律）：并行只加速**统计阶段**，**合并循环天生串行**（每轮依赖上一轮）。

```
实测（5MB, vocab=1000）: 串行 1.93s → 并行(4) 0.98s, 1.96x
结果逐字节一致（词频累加与顺序无关）
```

**不能 import `pretokenization_example.py`**——它第 53 行 `open(..., "rb")` 的 `...` 会在 import 时报错，所以把 `find_chunk_boundaries` 拷进来了。

---

## 4. Tokenizer：编码 & 解码

### 4.1 三个方法难度对比

| 方法 | 难度 | 本质 |
|------|------|------|
| `from_files` | ⭐ | 文件加载（GPT-2 格式，字节↔unicode 还原）|
| `decode` | ⭐ | 查表 + 拼接 + 一次性 UTF-8 解码 |
| `encode` | ⭐⭐⭐ | **重演 BPE 合并**（核心算法）|

### 4.2 encode 核心：按 rank 顺序，不是「最长优先」

**最常见的误解**："尽量合成最长的能匹配的 token" —— **错！**

encode 的规则：**每轮找当前所有相邻 pair 里 rank 最小的合并，反复直到无规则可用。**

- `rank` = merges 列表下标 = 训练时的学习顺序（数字越小越早学、越优先）。
- 长度变长只是**合并的副产品**，不是**目标**。

**反例（"最长优先" 会算错）**：

```
merges: rank0=(b,c)→bc, rank1=(a,b)→ab   词 [a,b,c]

最长优先: 从左扫，"ab" 更长 → [ab, c]
BPE 实际: (b,c) rank 更小先合 → [a, bc]

结果不同！ b 被 ab 和 bc 争抢，长度分不出胜负时 rank 说了算。
```

**为什么必须认 rank**：模型是用 BPE 切法训练的，encode 必须逐 token 复现训练切法，否则喂给模型没见过的序列。

**encode 完整流程**：
```
文本 → ① re.split 摘 special token → ② 普通段 GPT-2 正则预分词
     → ③ 每个 pretoken 按 rank 顺序合并 → ④ 查 inverse_vocab 得 ID
special token 段：直接映射成它的 ID，不进 BPE
```

### 4.3 decode：拼完再解码

```python
data = b"".join(self.vocab.get(i, replacement) for i in ids)
return data.decode("utf-8", errors="replace")
```

**关键坑**：一个 UTF-8 字符可能跨多个 token，**必须把所有 bytes 拼完再一次性 decode**，逐 token decode 会炸。`errors="replace"` 让非法字节序列替换成 U+FFFD 而不报错。

### 4.4 encode_iterable：内存友好

用**生成器 `yield`**，一次只处理一行/一块，不读全文（测试限制 1MB 内存编码 5MB 文件）。行以 `\n` 结尾，天然安全边界（不切断 UTF-8 字符或 special token）。

### 4.5 重叠 special token

`<|endoftext|>` 和 `<|endoftext|><|endoftext|>` 同时是 special token 时，**长的优先匹配**（`sorted(key=len, reverse=True)` + 正则 alternation 顺序）。

### 4.6 from_files：GPT-2 格式

GPT-2 的 vocab.json/merges.txt 里 token 用"可打印 unicode 字符"表示字节（避免控制字符）。加载时用 `_gpt2_bytes_to_unicode` 的**反向映射**还原真字节。

---

## 5. 完整代码

> 文件：`cs336_basics/tokenizer.py`

```python
"""BPE (Byte-Pair Encoding) 分词器实现"""

import json
import os
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool
from typing import BinaryIO, Iterable, Iterator, Optional


# GPT-2 使用的预分词正则表达式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ============================================================
# 训练辅助函数
# ============================================================

def _get_pairs(word: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    """返回一个词里所有相邻 pair（可能有重复，比如 'a a a' 会有两个 (a,a)）。"""
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def _merge_word(word, pair, new_token):
    """在一个词里把所有 pair 替换成 new_token（左到右贪心，不重叠）。"""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def _find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    """把文件切成若干块，切点落在 split_special_token 处（拷自 Stanford 示例）。"""
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def _count_word_freqs(text, special_tokens):
    """把一段文本转成 word_freqs（含 special token 保护 + GPT-2 正则预分词）。"""
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        split_pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
        segments = re.split(split_pattern, text)
    else:
        segments = [text]
    special_set = set(special_tokens)
    pattern = re.compile(GPT2_SPLIT_PATTERN)
    word_freqs = defaultdict(int)
    for segment in segments:
        if segment in special_set:
            continue
        for chunk in pattern.findall(segment):
            word_tokens = tuple(bytes([b]) for b in chunk.encode('utf-8'))
            word_freqs[word_tokens] += 1
    return word_freqs


def _process_chunk(args):
    """multiprocessing worker：读文件某段 → word_freqs。必须是模块级函数。"""
    input_path, start, end, special_tokens = args
    with open(input_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode('utf-8', errors='ignore')
    return dict(_count_word_freqs(text, special_tokens))


# ============================================================
# 训练主函数
# ============================================================

def train_bpe(input_path, vocab_size=256, special_tokens=None, num_processes=1):
    """训练 BPE 分词器，返回 (vocab, merges)。"""
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
    special_tokens = special_tokens or []

    # Step 1: 初始化词汇表（0-255 单字节 + special tokens）
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    for token in special_tokens:
        vocab[next_token_id] = token.encode('utf-8')
        next_token_id += 1

    # Step 2-4: 读取 + 预分词 + 统计词频
    can_parallelize = num_processes > 1 and len(special_tokens) > 0
    if can_parallelize:
        split_token = special_tokens[0].encode('utf-8')
        with open(input_path, 'rb') as f:
            boundaries = _find_chunk_boundaries(f, num_processes, split_token)
        chunk_args = [(input_path, start, end, special_tokens)
                      for start, end in zip(boundaries[:-1], boundaries[1:])]
        with Pool(num_processes) as pool:
            partial_freqs = pool.map(_process_chunk, chunk_args)
        word_freqs = defaultdict(int)
        for pf in partial_freqs:
            for word, freq in pf.items():
                word_freqs[word] += freq
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        word_freqs = _count_word_freqs(text, special_tokens)

    # Step 5: 初始化 pair 计数 + 倒排索引（只扫一次）
    pair_freqs = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in word_freqs.items():
        for pair in _get_pairs(word):
            pair_freqs[pair] += freq
            pair_to_words[pair].add(word)

    # Step 6: 循环合并，每轮只增量更新受影响的词
    merges = []
    num_merges_needed = vocab_size - len(vocab)
    for _ in range(num_merges_needed):
        if not pair_freqs:
            break
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        if pair_freqs[best_pair] <= 0:
            break
        new_token = best_pair[0] + best_pair[1]
        vocab[next_token_id] = new_token
        merges.append(best_pair)
        next_token_id += 1

        affected_words = list(pair_to_words[best_pair])
        for word in affected_words:
            freq = word_freqs.pop(word)
            old_pairs = _get_pairs(word)
            for pair in old_pairs:
                pair_freqs[pair] -= freq
            for pair in set(old_pairs):
                pair_to_words[pair].discard(word)
            new_word = _merge_word(word, best_pair, new_token)
            new_pairs = _get_pairs(new_word)
            for pair in new_pairs:
                pair_freqs[pair] += freq
            for pair in set(new_pairs):
                pair_to_words[pair].add(new_word)
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

    # Step 7: 返回
    return vocab, merges


# ============================================================
# Tokenizer 类（编解码）
# ============================================================

def _gpt2_bytes_to_unicode():
    """GPT-2 的"字节→可打印 unicode 字符"映射（from_files 加载用）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Tokenizer:
    """BPE Tokenizer —— 文本 ↔ token IDs 的编解码。"""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        # merge 优先级：pair → rank（下标越小越优先）
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        # 确保 special token 都在 vocab 里
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = tok_bytes
        # 反向映射：bytes → ID
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._pat = re.compile(GPT2_SPLIT_PATTERN)
        # special token 切分正则（长度降序，重叠时长的优先）
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pat = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_specials) + ")")
        else:
            self._special_pat = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """从 GPT-2 格式的 vocab.json + merges.txt 加载。"""
        byte_decoder = {c: b for b, c in _gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab = {idx: bytes(byte_decoder[ch] for ch in token_str)
                 for token_str, idx in gpt2_vocab.items()}
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line or len(line.split(" ")) != 2:
                    continue
                a, b = line.split(" ")
                merges.append((bytes(byte_decoder[ch] for ch in a),
                               bytes(byte_decoder[ch] for ch in b)))
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_pretoken(self, pretoken):
        """对单个 pretoken 应用 BPE：每轮找 rank 最小的 pair 合并，直到无规则可用。"""
        parts = [bytes([b]) for b in pretoken.encode("utf-8")]
        while len(parts) >= 2:
            best_rank = None
            best_i = None
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            parts[best_i: best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        return [self.inverse_vocab[p] for p in parts]

    def _encode_yield(self, text):
        """文本 → token IDs 流（生成器，encode/encode_iterable 复用）。"""
        if self._special_pat is not None:
            segments = self._special_pat.split(text)
        else:
            segments = [text]
        special_set = set(self.special_tokens)
        for segment in segments:
            if not segment:
                continue
            if segment in special_set:
                yield self.inverse_vocab[segment.encode("utf-8")]
            else:
                for pretoken in self._pat.findall(segment):
                    yield from self._bpe_encode_pretoken(pretoken)

    def encode(self, text):
        return list(self._encode_yield(text))

    def encode_iterable(self, iterable):
        """惰性编码（文件句柄等），内存友好，逐个 yield ID。"""
        for chunk in iterable:
            yield from self._encode_yield(chunk)

    def decode(self, ids):
        """token IDs → 文本：拼完所有 bytes 再一次性 UTF-8 解码。"""
        replacement = "�".encode("utf-8")
        data = b"".join(self.vocab.get(i, replacement) for i in ids)
        return data.decode("utf-8", errors="replace")
```

---

## 6. 测试

```bash
cd ~/learning/cs336-assignment1

# 全套（BPE 训练 + Tokenizer）
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py -v

# 只训练 / 只编解码
uv run pytest tests/test_train_bpe.py -v
uv run pytest tests/test_tokenizer.py -v

# 简洁输出
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py -q

# 单个测试调试（-s 允许 print）
uv run pytest tests/test_tokenizer.py::test_overlapping_special_tokens -v -s
```

**预期结果：`27 passed, 1 xfailed`**

- 那个 `xfailed`（`test_encode_memory_usage`）是**预期失败**——故意验证非流式 `encode` 会超 1MB 内存，本就该费内存。

### 28 个测试测什么（速查）

| 组 | 测试 | 考点 |
|----|------|------|
| **train_bpe** | `test_train_bpe` | 正确性（vocab/merges 对答案）|
| | `test_train_bpe_speed` | < 1.5s（增量更新）|
| | `test_train_bpe_special_tokens` | special token 不污染 vocab |
| **Tokenizer** | `*_roundtrip` | encode→decode 可逆 |
| | `*_matches_tiktoken` | 结果对齐 OpenAI tiktoken |
| | 输入：空串/单字符/emoji/unicode/ascii/德语/地址/故事 | 层层加码 |
| | `test_overlapping_special_tokens` | 重叠 special 长的优先 |
| | `test_encode_iterable_memory_usage` | 1MB 内存编码 5MB 文件 |

---

## 7. 踩坑合集（复习重点）

| # | 坑 | 根因 | 解法 |
|---|-----|------|------|
| 1 | **合并死循环**（全是 `(b' ', b't')`）| word 存 `tuple[int]`，合并时新 token 被拆回单字节 | 改存 `tuple[bytes]`：`tuple(bytes([b]) for b in ...)` |
| 2 | **speed 测试超时**（1.93s > 1.5s）| 每轮全量重算所有 pair，O(n²) | 增量更新 + 倒排索引，只改受影响的词 |
| 3 | **special token 污染**（vocab 冒出 `b'<\|'`）| 训练文本里 `<\|endoftext\|>` 被正则切碎合并 | 预分词前 `re.split` 摘出 special token |
| 4 | **重叠 special 被拆**（`<\|eot\|><\|eot\|>` 拆成两个）| split/match 按短的先匹配 | `sorted(key=len, reverse=True)` 长的优先 |
| 5 | **encode 用「最长优先」**结果错 | 字符争抢时长度分不出胜负 | 严格按 **rank 顺序**（merges 下标）合并 |
| 6 | **decode 乱码**（unicode 字符）| 逐 token decode，一个字符跨多 token | 拼完所有 bytes 再**一次性** decode，`errors='replace'` |
| 7 | **encode_iterable 内存爆**（5MB > 1MB）| 读全文再编码 | 用生成器 `yield`，逐行处理 |
| 8 | **无法 import pretokenization_example** | 它第 53 行 `open(...)` 的 `...` import 时报错 | 把 `find_chunk_boundaries` 拷进来 |
| 9 | 增量更新 **重叠 pair 计数错** | `a a a` 里 `(a,a)` 出现 2 次 | 计数按出现次数，倒排索引按去重 pair |
| 10 | 增量更新 **选到失效 pair** | 减到 0 的残留计数还在 dict 里 | 选 best_pair 后 `<= 0` 兜底 break |

---

## 附：核心心法（一句话记忆）

- **train**：合到 vocab **够大**就停（主动，看 vocab_size），每轮选**全局最高频** pair。
- **encode**：合到词里**没规则可套**就停（被动），每轮选 **rank 最小**（最早学的）pair。
- **vocab** 编解码都要（bytes 查 ID / ID 查 bytes）；**merges** 只有 encode 要。
- **rank 顺序 = 训练顺序**，encode 照 rank 走 = 重放训练切法 = 和模型训练一致。
- 高频词 → 专属 token；低频词 → 碎片拼。这就是 BPE 的精髓。
