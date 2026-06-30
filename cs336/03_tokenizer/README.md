# CS336 Assignment 1 - Part 3: 分词器编码与解码

> 对应原始 PDF Section 2.6-2.7 (pages 10-12)

## 2.6 BPE 分词器：编码和解码

训练完 BPE 分词器后，我们需要实现两个核心功能：
- **编码 (Encode)**：将文本字符串转换为 token ID 序列
- **解码 (Decode)**：将 token ID 序列转换回文本字符串

### 2.6.1 编码文本

编码过程分为两步：

**Step 1: 预分词 + UTF-8 字节表示**

首先对输入文本进行预分词（使用与训练时相同的正则表达式），然后将每个预分词 token 转换为 UTF-8 字节序列。

**Step 2: 按训练时的合并顺序应用合并**

对每个预分词 token 的字节序列，按照训练时记录的合并顺序依次应用合并操作。每一步检查当前序列中是否存在可合并的 pair，如果存在且该 pair 在合并列表中，则执行合并。

---

### Example (bpe_encoding): BPE 编码示例

假设我们有以下词汇表和合并列表：

**词汇表**：
```
{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'\n',
 7: b'at', 8: b'th', 9: b'the', 10: b' c'}
```

**合并列表**（按训练顺序）：
```
[(b'a', b't'), (b't', b'h'), (b'th', b'e'), (b' ', b'c')]
```

**编码 `'the cat ate'`**：

首先进行预分词（使用 GPT-2 正则表达式）：
```
['the', ' cat', ' ate']
```

对每个预分词 token 应用合并：

**Token `'the'`** → 字节序列 `[b't', b'h', b'e']`：
1. 检查合并列表。第一个可用合并 `(b't', b'h')` 存在于序列中 → 合并为 `[b'th', b'e']`
2. 下一个可用合并 `(b'th', b'e')` 存在于序列中 → 合并为 `[b'the']`
3. 无更多可合并的 pair → 最终结果 `[b'the']` → token ID `[9]`

**Token `' cat'`** → 字节序列 `[b' ', b'c', b'a', b't']`：
1. 检查合并列表。第一个可用合并 `(b'a', b't')` 存在 → 合并为 `[b' ', b'c', b'at']`
2. 下一个可用合并 `(b' ', b'c')` 存在 → 合并为 `[b' c', b'at']`
3. 无更多可合并的 pair → 最终结果 `[b' c', b'at']` → token IDs `[10, 7]`

**Token `' ate'`** → 字节序列 `[b' ', b'a', b't', b'e']`：
1. 检查合并列表。第一个可用合并 `(b'a', b't')` 存在 → 合并为 `[b' ', b'at', b'e']`
2. 无更多可合并的 pair → 最终结果 `[b' ', b'at', b'e']` → token IDs `[0, 7, 3]`

**最终编码结果**：`[9, 10, 7, 0, 7, 3]`

> **关键点**：合并必须按照训练时的顺序应用。即使某个 pair 在序列中存在，也必须等到它在合并列表中的轮次才能执行。

---

### Special Tokens 处理

编码时需要正确处理用户定义的特殊 token（如 `<|endoftext|>`）。在进行正则预分词**之前**，应先将输入文本按 special tokens 分割。Special tokens 直接映射为对应的 token ID，不进行进一步的分词处理。

### Memory Considerations

对于大文件，应该分块处理而不是一次性将整个文件读入内存。需要注意确保 token 不会跨 chunk 边界被拆分。

---

### 2.6.2 解码文本

解码过程相对简单：

1. 对每个 token ID，查找词汇表中对应的字节序列
2. 将所有字节序列拼接成一个完整的字节串
3. 将字节串解码为 Unicode 字符串

对于无效的 Unicode 字节序列，使用 Python 的 `errors='replace'` 参数，将无法解码的字节替换为 Unicode 替换字符 `U+FFFD` (&#xFFFD;)。

```python
>>> b'\x80\x81'.decode("utf-8", errors="replace")
'��'
```

---

### Problem (tokenizer): Implementing the Tokenizer (15 points)

实现 `Tokenizer` 类，包含以下接口：

```python
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        初始化分词器。

        Args:
            vocab: dict[int, bytes]
                词汇表，从 token ID 到 token 值（bytes）的映射。
            merges: list[tuple[bytes, bytes]]
                BPE 合并列表，按训练时的合并顺序排列。
            special_tokens: list[str] | None
                特殊 token 列表。默认 None。
        """
        ...

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        从文件加载分词器。

        Args:
            vocab_filepath: str
                词汇表文件路径。
            merges_filepath: str
                合并列表文件路径。
            special_tokens: list[str] | None
                特殊 token 列表。默认 None。
        """
        ...

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 列表。

        Args:
            text: str  输入文本。

        Returns:
            list[int]  token ID 列表。
        """
        ...

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        惰性地将可迭代的文本编码为 token ID。
        适用于处理大文件时的内存高效编码。

        Args:
            iterable: Iterable[str]  文本的可迭代对象。

        Returns:
            Iterator[int]  token ID 的迭代器。
        """
        ...

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码为文本。

        Args:
            ids: list[int]  token ID 列表。

        Returns:
            str  解码后的文本字符串。
        """
        ...
```

**测试方法**：

```bash
# 通过 adapter 运行
adapters.get_tokenizer

# 运行测试
uv run pytest tests/test_tokenizer.py
```

---

### Problem (tokenizer_experiments): Experiments with Tokenizers (4 points)

**(a)** 从 TinyStories 和 OpenWebText 数据集中各随机抽取 10 个文档，分别用 10K 词汇表分词器（TinyStories 训练）和 32K 词汇表分词器（OWT 训练）对它们编码。计算每种组合的**压缩率** (bytes per token)。

> 压缩率 = 原始文本的字节数 / 编码后的 token 数

**Deliverable**: 一张表格，展示 4 种组合（2 个数据集 x 2 个分词器）的平均压缩率。

**(b)** 用 TinyStories 分词器（10K 词汇表）对 OpenWebText 的样本进行分词，与用 OWT 分词器（32K 词汇表）的结果比较压缩率。有什么差异？为什么？

**Deliverable**: 一到两句话回答。

**(c)** 估算分词器的吞吐量 (bytes/second)。基于此估算，使用你的分词器对 The Pile 数据集（约 825 GB）进行完整分词需要多长时间？

**Deliverable**: 吞吐量估算值和 The Pile 分词时间估算。

**(d)** 用 TinyStories 分词器和 OWT 分词器分别编码对应的训练集和开发集 (validation set)，将编码结果序列化为 token ID 序列。建议将结果保存为 NumPy 的 `uint16` 数组。

为什么 `uint16`（范围 0-65535）对于我们的词汇表大小是合适的？

**Deliverable**: 序列化后的 token 文件 + 一句话解释为什么 `uint16` 合适。
