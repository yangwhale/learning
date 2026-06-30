# CS336 作业 1（基础）：构建 Transformer 语言模型

**版本 26.0.3**

CS336 课程组 | 2026 年春季

---

## 1 作业概述

在本次作业中，你将从零开始构建训练标准 Transformer 语言模型（LM）所需的全部组件，并训练一些模型。

### 你将实现的内容

1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数和 AdamW 优化器（第 4 节）
4. 训练循环，支持模型和优化器状态的序列化与加载（第 5 节）

### 你将运行的实验

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 用训练好的分词器对数据集编码，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 进行文本采样生成和困惑度评估。
5. 在 OpenWebText 上训练模型，并将困惑度提交到排行榜。

### 你可以使用的工具

我们期望你从零开始构建每个组件。特别地，你**不可以**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义，以下例外：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（如 `Module`、`ModuleList`、`Sequential` 等）
- `torch.optim.Optimizer` 基类

你可以使用任何其他 PyTorch 定义。如果不确定某个函数或类是否可以使用，可以在 Slack 上提问。如有疑问，请考虑使用它是否违背了本作业"从零开始"的宗旨。

### 关于 AI 工具的声明

AI 可以完全自主地解决作业的许多部分，但这会使你更难深入理解和学习课程材料。

AI 工具可用于回答高层概念性问题，或提供函数签名和库 API 等低层编程文档。但是，**不允许**使用 AI 工具来实现作业的任何部分。这包括编程 Agent（如 Cursor Agents、Codex、Claude Code）和 AI 自动补全（如 Cursor Tab、GitHub Copilot）。使用 AI agent 时，请确保使用提供的 AGENTS.md 文件。使用聊天机器人时也应包含 prompt。

我们强烈建议你在完成作业时禁用 IDE 中的 AI 自动补全功能。之前的学生反馈，禁用 AI 自动补全有助于更深入地理解材料。

完整 AI 政策请参阅相关文档。

### 代码结构

作业代码和本文档均可在 GitHub 上获取：

```
github.com/stanford-cs336/assignment1-basics
```

请 `git clone` 该仓库。如有更新，我们会通知你，届时可以 `git pull` 获取最新版本。

1. **`cs336_basics/*`**：你在这里编写代码。注意这里没有预置代码——你可以从零开始随意组织。
2. **`adapters.py`**：你的代码必须实现的一组功能接口。对于每个功能（如 scaled dot product attention），填写其实现（如 `run_scaled_dot_product_attention`），只需简单调用你的代码。`adapters.py` 的修改不应包含实质性逻辑，它只是粘合代码。
3. **`test_*.py`**：包含你必须通过的所有测试（如 `test_scaled_dot_product_attention`），这些测试会调用 `adapters.py` 中定义的接口。不要编辑测试文件。

### 如何提交

运行 `make_submission.sh` 来构建提交的 zip 文件。如果有大型数据文件或检查点不想包含在提交的 zip 中，请将它们添加到脚本的排除列表中。

你需要向 Gradescope 提交以下文件：
- **writeup.pdf**：回答所有书面问题。请排版你的答案。
- **code.zip**：包含你编写的所有代码。

要提交到排行榜，请向以下仓库提交 PR：
```
github.com/stanford-cs336/assignment1-basics-leaderboard
```

详细提交说明请参阅排行榜仓库中的 `README.md`。

### 数据集获取

本作业使用两个预处理数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。两个数据集都是单个大型纯文本文件。

如果你在课程注册并有课程计算资源的访问权限，可以在计算指南中找到数据集下载说明。

如果你在家中学习，可以使用 `README.md` 中的命令下载这些文件。

> **低资源提示：入门**
>
> 在整个课程作业中，我们会给出在 GPU 资源较少或没有 GPU 的情况下完成部分作业的建议。例如，我们有时会建议**缩小**数据集或模型规模，或说明如何在 Mac 的集成 GPU 或 CPU 上运行训练代码。你会在蓝色框中看到这些"低资源提示"。即使你是注册的 Stanford 学生并有课程机器的访问权限，这些提示也可能帮助你更快迭代和节省时间，我们建议阅读！

> **低资源提示：在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用课程组的参考实现，我们可以在配备 36 GB RAM 的 Apple M4 Max 芯片上，在 Metal GPU (MPS) 上不到 5 分钟、CPU 上约 30 分钟内训练一个 LM 生成相当流畅的文本。只要你的实现正确且高效，你就能用一台相当新的笔记本电脑训练一个生成简单儿童故事的小型 LM。后续作业中我们会说明在 CPU 或 MPS 上需要做哪些调整。

---

## 2 字节对编码（BPE）分词器

在作业的第一部分，我们将训练和实现一个字节级字节对编码（BPE）分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。具体来说，我们将任意 Unicode 字符串表示为字节序列，并在该字节序列上训练 BPE 分词器。之后，我们将使用该分词器将文本（字符串）编码为 token（整数序列），用于语言建模。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，将字符映射为整数**码点**（code points）。截至 Unicode 17.0（2025年9月发布），该标准在 172 种文字中定义了 159,801 个字符。例如，字符 "s" 的码点为 115（通常记作 `U+0073`，其中 `U+` 是约定前缀，`0073` 是十六进制的 115），字符 "牛" 的码点为 29275。在 Python 中，可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示，`chr()` 函数将整数码点转换为对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

> **题目 (unicode1)：理解 Unicode（1 分）**
>
> (a) `chr(0)` 返回什么 Unicode 字符？
>    **提交物**：一句话回答。
>
> (b) 该字符的字符串表示（`__repr__()`）与其打印表示有何不同？
>    **提交物**：一句话回答。
>
> (c) 当该字符出现在文本中时会发生什么？你可以在 Python 解释器中尝试以下操作：
>    ```python
>    >>> chr(0)
>    >>> print(chr(0))
>    >>> "this is a test" + chr(0) + "string"
>    >>> print("this is a test" + chr(0) + "string")
>    ```
>    **提交物**：一句话回答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不现实的，因为词表会非常大（约 15 万项）且稀疏（很多字符非常罕见）。因此，我们使用 Unicode 编码将 Unicode 字符转换为字节序列。Unicode 标准定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网的主导编码（占所有网页的 98% 以上）。

要在 Python 中将 Unicode 字符串编码为 UTF-8，可以使用 `encode()` 函数。要访问 Python `bytes` 对象的底层字节值，可以对其迭代（如调用 `list()`）。最后，可以使用 `decode()` 函数将 UTF-8 字节串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 129, 130, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 码点转换为字节序列（例如通过 UTF-8 编码），我们本质上是将码点序列（21 位整数，有 159,801 个有效值）转换为字节值序列（范围 0 到 255 的整数）。256 长度的字节词表比码点词表**容易管理得多**。使用字节级分词时，我们无需担心词汇外（OOV）的 token，因为任何输入文本都可以表示为 0 到 255 的整数序列。

> **题目 (unicode2)：Unicode 编码（3 分）**
>
> (a) 相比 UTF-16 或 UTF-32，优先在 UTF-8 编码的字节上训练分词器有哪些原因？
>    **提交物**：一到两句话回答。
>
> (b) 考虑以下（不正确的）函数，其目的是将 UTF-8 字节串解码为 Unicode 字符串。为什么这个函数是错误的？请给出一个产生错误结果的输入字节串示例。
>    ```python
>    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
>        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>    >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
>    'hello'
>    ```
>    **提交物**：一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串示例，以及一句话解释为什么该函数是错误的。
>
> (c) 给出一个不能解码为任何 Unicode 字符的两字节序列。
>    **提交物**：一个示例，附一句话解释。

### 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的 OOV 问题，但将文本分词为字节会产生极长的输入序列。这会减慢模型训练速度，因为一个 10 个单词的句子在词级语言模型中可能只有 10 个 token，但在字符级模型中可能有 50 个或更多 token（取决于单词长度）。处理这些更长的序列需要在模型的每一步进行更多计算。此外，在字节序列上进行语言建模很困难，因为更长的输入序列会在数据中产生更远距离的依赖关系。

子词分词是词级分词器和字节级分词器之间的折中方案。注意字节级分词器的词表有 256 个条目（字节值 0 到 255）。子词分词器用更大的词表来换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在原始文本训练数据中经常出现，将其分配为词表中的一个条目就能把这个 3-token 序列压缩为单个 token。

如何选择要添加到词表中的子词单元？[R. Sennrich et al. [3]] 提出使用字节对编码（BPE; [P. Gage [5]]），这是一种压缩算法，通过迭代地将最频繁的字节对替换（"合并"）为一个新的未使用索引。注意，该算法添加的是子词 token 以最大化输入序列的压缩——如果一个词在输入文本中出现的次数足够多，它就会被表示为单个子词单元。

通过 BPE 构建词表的子词分词器通常被称为 BPE 分词器。在本作业中，我们将实现字节级 BPE 分词器，其中词表项目是字节或合并后的字节序列，这为我们提供了 OOV 处理和可管理输入序列长度的双重优势。构建 BPE 分词器词表的过程被称为"训练" BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器训练过程由三个主要步骤组成。

**词表初始化**

分词器词表是从字节串 token 到整数 ID 的一对一映射。由于我们训练的是字节级 BPE 分词器，初始词表就是所有字节值的集合。因此初始词表大小为 256。

**预分词**

一旦有了词表，原则上你可以计算文本中相邻字节对的出现频率，然后从最频繁的对开始合并。但这计算开销很大，因为每次合并都需要遍历整个语料库。此外，直接在语料库上合并字节可能产生只在标点上不同的 token（如 `dog!` 与 `dog.`）。这些 token 会获得完全不同的 token ID，即使它们在语义上可能高度相似。

为了避免这个问题，我们对语料库进行**预分词**。你可以把这看作是对语料库的粗粒度分词，帮助我们计算字符对出现的频率。例如，单词 `'text'` 可能是一个出现 10 次的预 token。在这种情况下，当我们统计 't' 和 'e' 相邻出现的次数时，我们会看到 'text' 中 't' 和 'e' 相邻，可以将它们的计数加 10，而不是遍历整个语料库。由于我们训练的是字节级 BPE 模型，每个预 token 表示为 UTF-8 字节序列。

[R. Sennrich et al. [3]] 的原始 BPE 实现通过简单地按空格分割（即 `s.split(" ")`）来预分词。该方法仍用于基于 SentencePiece 的分词器（例如 Llama 1 和 Llama 2 分词器）。

大多数现代分词器使用基于正则表达式的预分词器，这是 GPT-2 开创的做法 [A. Radford et al. [6]]。我们将使用一个稍微美观一点的原始正则表达式变体，取自 `github.com/openai/tiktoken/pull/234/files`：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

你可以交互式地用这个预分词器分割一些文本来了解其行为：

```python
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

在实际代码中使用时，你应该用 `re.finditer` 来避免在构建从预 token 到计数的映射时存储所有预 token。

**计算 BPE 合并**

现在我们已经将输入文本转换为预 token，并将每个预 token 表示为 UTF-8 字节序列，我们可以计算 BPE 合并（即训练 BPE 分词器）。在高层来看，BPE 算法迭代地计算每对字节的频率，并找出频率最高的对（"A"，"B"）。然后将这个最频繁对的每次出现**合并**，即替换为一个新的 token "AB"。这个新的合并 token 被添加到词表中；因此，BPE 训练后的最终词表大小等于初始词表大小（在我们的情况下为 256）加上 BPE 合并操作的次数。为了效率，在 BPE 训练期间我们不考虑跨预 token 边界的对。当对的频率出现平局时，通过**优先选择字典序更大的对**来确定性地打破平局。例如，如果对 ("A", "B")、("A", "C")、("B", "ZZ")、("BA", "A") 都具有最高频率，我们会合并 ("BA", "A")：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

**特殊 token**

通常，一些字符串（如 `<|endoftext|>`）用于编码元数据（如文档间的边界）。在编码文本时，通常需要将某些字符串视为"特殊 token"，这些 token 永远不会被分割为多个 token（即始终保持为单个 token，对应单个整数 ID），这样我们就知道语言模型何时停止生成。这些特殊 token 必须添加到词表中，它们有对应的固定 token ID。

[R. Sennrich et al. [3]] 的算法 1 包含了一个低效的 BPE 分词器训练实现（基本遵循上述步骤）。作为第一个练习，实现并测试该函数可能很有帮助。

> **示例 (bpe_example)：BPE 训练示例**
>
> 这里是来自 [R. Sennrich et al. [3]] 的一个风格化示例。考虑以下语料库：
> ```
> low low low low low
> lower lower widest widest widest
> newest newest newest newest newest newest
> ```
> 词表有一个特殊 token `<|endoftext|>`。
>
> **词表**：我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化词表。
>
> **预分词**：为简化并聚焦于合并过程，本例假设预分词简单按空格分割。预分词并计数后，得到频率表：
> `{low: 5, lower: 2, widest: 3, newest: 6}`
>
> **合并**：首先查看每对相邻字节及其频率。对 ('e', 's') 和 ('s', 't') 平局，取字典序更大者 ('s', 't')。第一轮合并后得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。
>
> 如果取 6 次合并，我们得到 ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']，词表元素为 [<|endoftext|>, [...256 字节字符], st, est, ow, low, west, ne]。
>
> 用该词表和合并集，单词 `newest` 会被分词为 [ne, west]。

### 2.5 BPE 分词器训练实验

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。

**并行化预分词**

你会发现预分词步骤是主要瓶颈。你可以使用 Python 内置的 `multiprocessing` 库并行化你的代码来加速。具体而言，我们建议在并行实现预分词时，对语料库进行分块，同时确保分块边界出现在特殊 token 的开头。

**在预分词前移除特殊 token**

在使用正则表达式模式运行预分词之前（使用 `re.finditer`），你应该从语料库（或分块）中剥离所有特殊 token。确保按特殊 token **分割**，这样它们所分隔的文本之间就不会发生合并。特殊 token 定义了训练期间的硬分割边界，但它们本身不应参与合并计数。

**优化合并步骤**

朴素的 BPE 训练实现很慢，因为每次合并都需要迭代所有字节对来找到最频繁的对。然而，每次合并后唯一变化的对计数是那些与合并对重叠的对。因此，可以通过索引所有对的计数并增量更新来改进 BPE 训练速度，而不是显式迭代每对字节来统计频率。这种缓存方法可以带来显著加速，不过我们注意到 BPE 训练的合并部分在 Python 中**不可**并行化。

> **题目 (train_bpe)：BPE 分词器训练（15 分）**
>
> **提交物**：编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应处理（至少）以下输入参数：
>
> - `input_path: str` — BPE 分词器训练数据的文本文件路径
> - `vocab_size: int` — 定义最终词表最大大小的正整数（包括初始字节词表、合并产生的词表项和所有特殊 token）
> - `special_tokens: list[str]` — 要添加到词表的字符串列表。训练时将其视为防止跨越其范围合并的硬边界，但不将其纳入合并统计
>
> **输出**：
> - `vocab: dict[int, bytes]` — 分词器词表，从 int（词表中的 token ID）到 bytes（token 字节）的映射
> - `merges: list[tuple[bytes, bytes]]` — BPE 训练产生的合并列表，按创建顺序排列

> **题目 (train_bpe_tinystories)：在 TinyStories 上训练 BPE（2 分）**
>
> (a) 在 TinyStories 数据集上训练字节级 BPE 分词器，最大词表大小为 10,000。确保将 TinyStories 的 `<|endoftext|>` 特殊 token 添加到词表。序列化生成的词表和合并以供进一步检查。训练花了多少时间和内存？词表中最长的 token 是什么？是否合理？
>    **资源要求**：≤ 30 分钟（无 GPU），≤ 30 GB RAM
>
> (b) 分析你的代码。分词器训练过程中哪个部分耗时最多？

> **题目 (train_bpe_expts_owt)：在 OpenWebText 上训练 BPE（2 分）**
>
> (a) 在 OpenWebText 数据集上训练字节级 BPE 分词器，最大词表大小为 32,000。词表中最长的 token 是什么？是否合理？
>    **资源要求**：≤ 12 小时（无 GPU），≤ 100 GB RAM
>
> (b) 对比你在 TinyStories 和 OpenWebText 上训练得到的分词器。

### 2.6 BPE 分词器：编码和解码

在前一部分中，我们实现了在输入文本上训练 BPE 分词器以获得词表和 BPE 合并列表的函数。现在，我们将实现一个 BPE 分词器，加载提供的词表和合并列表，并用它们将文本编码/解码为 token ID。

#### 2.6.1 编码文本

BPE 编码过程与训练过程类似，主要步骤：

**步骤 1：预分词**。先对序列进行预分词，将每个预 token 表示为 UTF-8 字节序列。

**步骤 2：应用合并**。然后取 BPE 训练中创建的合并序列，**按创建顺序**应用到预 token 上。

> **示例 (bpe_encoding)：BPE 编码示例**
>
> 假设输入字符串为 `'the cat ate'`，词表为 {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}，合并为 [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]。
>
> 首先，预分词器将字符串分割为 ['the', ' cat', ' ate']。然后对每个预 token 应用 BPE 合并。
>
> 第一个预 token 'the' 初始表示为 [b't', b'h', b'e']。应用合并 (b't', b'h') 得到 [b'th', b'e']，再应用 (b'th', b'e') 得到 [b'the']，对应整数序列 [9]。
>
> 对剩余预 token 重复此过程：' cat' 得到 [7, 1, 5]，' ate' 得到 [10, 3]。因此，最终编码结果为 [9, 7, 1, 5, 10, 3]。

**特殊 token**

你的分词器在编码文本时应能正确处理用户定义的特殊 token（在构造分词器时提供）。

**内存考虑**

假设我们想分词一个无法全部装入内存的大型文本文件。为了高效地分词这样的大文件（或任何其他数据流），我们需要将其分成可管理的块并依次处理，使得内存复杂度为常数而非线性于文本大小。在此过程中，我们需要确保 token 不会跨越块边界，否则分词结果会与在内存中分词整个序列不同。

#### 2.6.2 解码文本

要将整数 token ID 序列解码回原始文本，我们只需查找每个 ID 在词表中对应的条目（字节序列），将它们拼接在一起，然后将字节解码为 Unicode 字符串。注意，输入 token ID 不保证能映射为有效的 Unicode 字符串（因为用户可以输入任意整数序列）。如果输入 token ID 无法生成有效的 Unicode 字符串，应使用官方 Unicode 替换字符 `U+FFFD` 替换格式错误的字节。`bytes.decode` 的 `errors` 参数控制 Unicode 解码错误的处理方式，使用 `errors='replace'` 会自动将格式错误的数据替换为替换标记。

> **题目 (tokenizer)：实现分词器（15 分）**
>
> **提交物**：实现一个 `Tokenizer` 类，给定词表和合并列表，将文本编码为整数 ID 并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token。推荐接口：
>
> - `__init__(self, vocab, merges, special_tokens=None)` — 从词表、合并列表和（可选）特殊 token 列表构造分词器
> - `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)` — 类方法，从序列化文件加载构造 Tokenizer
> - `encode(self, text: str) -> list[int]` — 将输入文本编码为 token ID 序列
> - `encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]` — 给定字符串的可迭代对象，返回懒惰地生成 token ID 的生成器（用于内存高效地分词大文件）
> - `decode(self, ids: list[int]) -> str` — 将 token ID 序列解码为文本

### 2.7 实验

> **题目 (tokenizer_experiments)：分词器实验（4 分）**
>
> (a) 从 TinyStories 和 OpenWebText 各采样 10 个文档。使用之前训练的 TinyStories 和 OpenWebText 分词器（词表大小分别为 10K 和 32K），将这些文档编码为整数 ID。每个分词器的压缩比（字节/token）是多少？
>
> (b) 如果用 TinyStories 分词器分词 OpenWebText 样本会怎样？比较压缩比和/或定性描述发生了什么。
>
> (c) 估计你的分词器的吞吐量（如字节/秒）。分词 Pile 数据集（825GB 文本）需要多长时间？
>
> (d) 使用 TinyStories 和 OpenWebText 分词器，将各自的训练和验证数据集编码为整数 token ID 序列。我们建议将 token ID 序列化为数据类型 `uint16` 的 NumPy 数组。为什么 `uint16` 是一个合适的选择？

---

## 3 Transformer 语言模型架构

语言模型接收一批整数 token ID（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），返回一个在词表上的（批次化）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中每个输入 token 对应的预测分布是下一个词的分布。训练时，我们用这些下一词预测来计算实际下一个词与预测下一个词之间的交叉熵损失。推理时生成文本时，我们从最终时间步的预测分布中采样下一个 token（如取最高概率、从分布中采样等），将生成的 token 添加到输入序列，然后重复。

在本作业中，我们将从零开始构建这个 Transformer 语言模型。

### 3.1 Transformer LM

给定 token ID 序列，Transformer 语言模型使用输入 embedding 将 token ID 转换为稠密向量，将嵌入后的 token 通过 `num_layers` 个 Transformer 块，然后应用一个学习到的线性投影（"输出 embedding"或"LM head"）来产生预测的下一 token logits。参见图 1 的示意图。

**Token Embedding**

第一步，Transformer 将（批次化的）token ID 序列**嵌入**为包含 token 身份信息的向量序列。具体来说，token embedding 层接收形状为 `(batch_size, sequence_length)` 的整数 tensor，产生形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

**Pre-norm Transformer 块**

嵌入后，激活值由若干结构相同的神经网络层处理。标准的 decoder-only Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer "块"）组成。每个块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，返回相同形状的输出。每个块通过 self-attention 聚合序列中的信息，并通过前馈网络进行非线性变换。

我们将实现"pre-norm" Transformer 块，它还在最后一个 Transformer 块之后额外添加一个层归一化，以确保输出被正确缩放。归一化之后，使用学习到的线性变换将 Transformer 块的输出转换为预测的下一 token logits。

### 3.2 备注：批处理、Einsum 与高效计算

在 Transformer 中，我们将对很多类似批次的输入执行相同的计算：
- **批次中的元素**：对每个批次元素应用相同的 Transformer forward 操作
- **序列长度**："逐位置"操作如 RMSNorm 和前馈网络在序列的每个位置上独立运行
- **注意力头**：注意力操作在多头注意力中跨注意力头批处理

我们**强烈推荐**在课程中学习和使用 einsum 记法。

#### 3.2.1 数学记法与内存排列

许多机器学习论文使用**行向量**记法，这与 NumPy 和 PyTorch 默认的行优先内存排列很好地匹配。用行向量表示，线性变换写为：

$$y = xW^\top$$

本作业中我们主要使用**列向量**的数学记法。使用 `einsum` 做线性代数运算时，只要正确标记轴，这不会造成问题。

### 3.3 基本构建模块：Linear 和 Embedding 模块

#### 3.3.1 参数初始化

训练神经网络需要仔细初始化模型参数。对于本作业，使用：
- Linear 权重：$\mathcal{N}(\mu=0, \sigma^2=\frac{2}{d_{in}+d_{out}})$，截断于 $[-3\sigma, 3\sigma]$
- Embedding：$\mathcal{N}(\mu=0, \sigma^2=1)$，截断于 $[-3, 3]$
- RMSNorm：1

使用 `torch.nn.init.trunc_normal_` 初始化截断正态权重。

#### 3.3.2 Linear 模块

> **题目 (linear)：实现 linear 模块（1 分）**
>
> **提交物**：实现一个继承自 `torch.nn.Module` 的 `Linear` 类，执行线性变换 $y = Wx$。注意遵循现代 LLM 的做法，不包含 bias 项。
>
> 接口：`__init__(self, in_features, out_features, device=None, dtype=None)`
>
> 注意存储参数为 $W$（不是 $W^\top$），放在 `nn.Parameter` 中。不要使用 `nn.Linear` 或 `nn.functional.linear`。

#### 3.3.3 Embedding 模块

> **题目 (embedding)：实现 embedding 模块（1 分）**
>
> **提交物**：实现一个继承自 `torch.nn.Module` 的 `Embedding` 类，执行 embedding 查找。不要使用 `nn.Embedding` 或 `nn.functional.embedding`。
>
> 接口：`__init__(self, num_embeddings, embedding_dim, device=None, dtype=None)`

### 3.4 Pre-Norm Transformer 块

每个 Transformer 块有两个子层：多头 self-attention 机制和逐位置前馈网络。

我们实现"pre-norm" Transformer 块：在每个子层的输入上先做层归一化，然后进行主要操作（MHA/FF），最后通过残差连接添加回子层输入。

#### 3.4.1 均方根层归一化（RMSNorm）

我们将使用均方根层归一化（RMSNorm）进行层归一化。给定激活向量 $a \in \mathbb{R}^{d_{model}}$，RMSNorm 将每个激活值 $a_i$ 重新缩放：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i$$

其中 $\text{RMS}(a) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}$。$g_i$ 是可学习的"增益"参数（共 `d_model` 个），$\varepsilon$ 是通常固定为 1e-5 的超参数。

注意应将输入上转型为 `torch.float32` 以防止平方时溢出，计算完成后再下转型回原始 dtype。

> **题目 (rmsnorm)：实现 RMSNorm（1 分）**

#### 3.4.2 逐位置前馈网络

我们将实现现代 LLM（如 Llama 3、Qwen 2.5）采用的 "SwiGLU" 激活函数，它将 SiLU（又称 Swish）激活与门控线性单元（GLU）相结合。

SiLU 激活函数定义为：$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}$

SwiGLU 前馈网络定义为：

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1 x) \odot W_3 x)$$

其中 $d_{ff} = \frac{8}{3} d_{model}$，实际实现中取 64 的最近倍数。

> **题目 (positionwise_feedforward)：实现逐位置前馈网络（2 分）**

#### 3.4.3 旋转位置编码（RoPE）

为了向模型注入位置信息，我们将实现旋转位置编码（RoPE）[J. Su et al., 2021]。RoPE 对 query 和 key 向量的成对维度应用旋转，旋转角度 $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$。

旋转矩阵 $R^i$ 是块对角矩阵，每个 $2 \times 2$ 块为旋转矩阵 $R^i_k$。实现时应利用这种结构，无需构造完整的 $d \times d$ 矩阵。

> **题目 (rope)：实现 RoPE（2 分）**
>
> 接口：`__init__(self, theta: float, d_k: int, max_seq_len: int, device=None)`
>
> `forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor`

#### 3.4.4 缩放点积注意力

注意力操作数学定义为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Masking**：支持可选的用户提供的布尔掩码，形状为 `(seq_len, seq_len)`。`True` 表示 query 关注该 key，`False` 表示不关注。对掩码为 `False` 的位置添加 $-\infty$。

> **题目 (softmax)：实现 softmax（1 分）**

> **题目 (scaled_dot_product_attention)：实现缩放点积注意力（5 分）**

#### 3.4.5 因果多头自注意力

多头注意力定义为：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)$$

多头自注意力操作为：

$$\text{MultiHeadSelfAttention}(x) = W_O \text{MultiHead}(W_Q x, W_K x, W_V x)$$

**因果 Masking**：防止模型关注序列中的未来 token。可以使用 `torch.triu` 或广播索引比较来构造因果掩码。

**应用 RoPE**：RoPE 应用于 query 和 key 向量，但不用于 value 向量。

> **题目 (multihead_self_attention)：实现因果多头自注意力（5 分）**

### 3.5 完整 Transformer LM

将 Transformer 块组装起来：embedding → `num_layers` 个 Transformer 块 → 最终层 norm 和 LM head。

> **题目 (transformer_block)：实现 Transformer 块（3 分）**

> **题目 (transformer_lm)：实现 Transformer LM（3 分）**

**资源统计**

> **题目 (transformer_accounting)：Transformer LM 资源统计（5 分）**
>
> (a) 一个 GPT-2 XL 规模的模型（vocab_size=50257, context_length=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4288）有多少可训练参数？用 float32 加载需要多少内存？
>
> (b) 列出完成一次前向传播所需的矩阵乘法及总 FLOPs。
>
> (c) 哪些部分消耗最多 FLOPs？
>
> (d) 对比 GPT-2 small/medium/large 的 FLOPs 分布。
>
> (e) 将 GPT-2 XL 的 context_length 增加到 16,384 时 FLOPs 如何变化？

---

## 4 训练 Transformer LM

### 4.1 交叉熵损失

交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D| \cdot m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} \mid x_{1:i})$$

实现时注意数值稳定性：减去最大值，消去 log 和 exp 的冗余计算。

**困惑度**：$\text{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right)$

> **题目 (cross_entropy)：实现交叉熵（1 分）**

### 4.2 SGD 优化器

SGD 更新规则：$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t)$

> **题目 (learning_rate_tuning)：调节学习率（1 分）**

### 4.3 AdamW

AdamW 是 Adam 的变体，通过解耦权重衰减来改进正则化。算法伪代码：

1. 初始化 $m \leftarrow 0$, $v \leftarrow 0$
2. 对每步 $t = 1, ..., T$：
   - 计算梯度 $g$
   - 调整学习率 $\alpha_t \leftarrow \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$
   - 权重衰减 $\theta \leftarrow \theta - \alpha\lambda\theta$
   - 更新一阶矩 $m \leftarrow \beta_1 m + (1-\beta_1)g$
   - 更新二阶矩 $v \leftarrow \beta_2 v + (1-\beta_2)g^2$
   - 参数更新 $\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v}+\varepsilon}$

> **题目 (adamw)：实现 AdamW（2 分）**

> **题目 (adamw_accounting)：AdamW 训练资源统计（2 分）**
>
> (a) 峰值内存分析（参数 + 激活值 + 梯度 + 优化器状态）
> (b) GPT-2 XL 在 80GB 内存下的最大 batch_size
> (c) AdamW 一步需要多少 FLOPs
> (d) 在 H100 上 50% MFU 训练 GPT-2 XL 400K 步需要多久

### 4.4 学习率调度

实现带 warmup 的余弦退火学习率调度：
- Warmup 阶段（$t < T_w$）：$\alpha_t = \frac{t}{T_w}\alpha_{max}$
- 余弦退火阶段（$T_w \le t \le T_c$）：$\alpha_t = \alpha_{min} + \frac{1}{2}(1+\cos(\frac{t-T_w}{T_c-T_w}\pi))(\alpha_{max}-\alpha_{min})$
- 后退火阶段（$t > T_c$）：$\alpha_t = \alpha_{min}$

> **题目 (learning_rate_schedule)：实现余弦学习率调度（1 分）**

### 4.5 梯度裁剪

给定所有参数的梯度 $g$，计算其 $\ell_2$ 范数 $\|g\|_2$。如果范数小于最大值 $M$ 则保持不变，否则按 $\frac{M}{\|g\|_2 + \varepsilon}$ 缩放。

> **题目 (gradient_clipping)：实现梯度裁剪（1 分）**

---

## 5 训练循环

### 5.1 数据加载器

分词后的数据是一个连续的 token 序列 $x = (x_1, ..., x_n)$。**数据加载器**将其转换为批次流，每个批次包含 $B$ 个长度为 $m$ 的序列，配对相应的下一 token 目标。

训练时使用 `np.memmap` 以内存映射模式加载大型数据集。

> **题目 (data_loading)：实现数据加载（2 分）**

### 5.2 检查点

检查点应包含：模型权重、优化器状态和迭代次数。使用 `torch.save` 和 `torch.load`。

> **题目 (checkpointing)：实现模型检查点（1 分）**

### 5.3 训练循环

> **题目 (training_together)：整合训练（4 分）**
>
> 编写一个训练脚本，支持：
> - 配置和控制各种模型和优化器超参数
> - 使用 `np.memmap` 内存高效加载大型训练和验证数据集
> - 序列化检查点到用户指定路径
> - 定期记录训练和验证性能

---

## 6 文本生成

### Softmax

语言模型输出是最后一层线性层的输出（"logits"），需要通过 softmax 转化为归一化概率分布。

### 解码

文本生成过程：提供前缀 token（"prompt"），模型产生词表上的概率分布来预测下一个 token，然后从该分布中采样。重复此过程直到生成 `<|endoftext|>` token 或达到用户指定的最大 token 数。

**温度缩放**：$\text{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{\text{vocab\_size}} \exp(v_j/\tau)}$

**Top-p 采样**（又称核采样）：截断低概率 token，只从概率之和 $\ge p$ 的最小 token 集合中采样。

> **题目 (decoding)：解码（3 分）**
>
> 实现解码函数，支持：用户 prompt 的补全、最大生成 token 数控制、温度缩放、Top-p 采样。

---

## 7 实验

### 7.1 如何运行实验和提交物

快速、一致地实验并保留记录。确保定期评估验证损失，并记录步数和实际运行时间。

> **题目 (experiment_log)：实验日志（3 分）**

### 7.2 TinyStories

使用 TinyStories 数据集（简单数据，训练快速）开始实验。

**推荐超参数**：
- 词表大小 10000
- 上下文长度 256
- d_model 512
- d_ff 1344
- RoPE theta 10000
- 4 层, 16 头
- 总 token 处理量 327,680,000

> **题目 (learning_rate)：调节学习率（3 分，2 B200 小时）**
>
> (a) 对学习率进行超参数扫描，报告最终损失。
>    **目标**：TinyStories 上验证损失 ≤ 1.45
>
> (b) 研究最佳学习率与发散临界点的关系。

> **题目 (batch_size_experiment)：batch size 变化实验（1 分，1 B200 小时）**

> **题目 (generate)：生成文本（1 分）**

### 7.3 消融实验和架构修改

> **题目 (layer_norm_ablation)：移除 RMSNorm 并训练（1 分）**
>
> **题目 (pre_norm_ablation)：实现 post-norm 并训练（1 分）**
>
> **题目 (no_pos_emb)：实现无位置编码（NoPE）（1 分）**
>
> **题目 (swiglu_ablation)：SwiGLU vs. SiLU 对比（1 分）**

### 7.4 在 OpenWebText 上运行

> **题目 (main_experiment)：在 OWT 上实验（2 分，2 B200 小时）**

### 7.5 自定义修改 + 排行榜

在 B200 上最多 45 分钟运行时间内，尽量降低 OpenWebText 上的验证损失。

**排行榜规则**：
- 运行时间：B200 上最多 45 分钟
- 数据：只能使用提供的 OpenWebText 训练数据集
- 其他：无限制

改进参考：Llama 3、Qwen 2.5、NanoGPT speedrun (github.com/KellerJordan/modded-nanogpt) 等。

> **题目 (leaderboard)：排行榜（6 分，10 B200 小时）**
>
> 目标：在 0.75 B200-hours 内达到优于朴素 baseline（loss 5.0）的验证损失。

---

## 参考文献

1. R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.
2. A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.
3. R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in *Proc. of ACL*, 2016.
4. C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.
5. P. Gage, "A new algorithm for data compression," *C Users Journal*, 1994.
6. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.
7. A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.
8. A. Vaswani et al., "Attention is All you Need," in *Proc. of NeurIPS*, 2017.
9. T. Q. Nguyen and J. Salazar, "Transformers without Tears," in *Proc. of IWSWLT*, 2019.
10. R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in *Proc. of ICML*, 2020.
11. J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.
12. H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.
13. B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in *Proc. of NeurIPS*, 2019.
14. A. Grattafiori et al., "The Llama 3 Herd of Models." 2024.
15. A. Yang et al., "Qwen2.5 Technical Report," *arXiv:2412.15115*, 2024.
16. A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.
17. D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.
18. S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." 2017.
19. Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." 2017.
20. N. Shazeer, "GLU Variants Improve Transformer." 2020.
21. J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
22. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. of ICLR*, 2015.
23. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. of ICLR*, 2019.
24. T. B. Brown et al., "Language Models are Few-Shot Learners," in *Proc. of NeurIPS*, 2020.
25. J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.
26. J. Hoffmann et al., "Training Compute-Optimal Language Models." 2022.
27. A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in *Proc. of ICLR*, 2020.
28. Y.-H. H. Tsai et al., "Transformer Dissection," in *Proc. of EMNLP-IJCNLP*, 2019.
29. A. Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers," in *NeurIPS*, 2023.
