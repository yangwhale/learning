# CS336 Assignment 1 - Part 1: Unicode

> 对应原始 PDF Section 2.1-2.2 (pages 3-5)

## 2.1 Unicode 标准

Unicode 是一种文本编码标准，将字符映射为整数**码点** (code points)。截至 Unicode 17.0（2025 年 9 月发布），标准定义了 159,801 个字符，涵盖 172 种文字。例如，字符 `"s"` 的码点是 115（通常表示为 `U+0073`，其中 `U+` 是常规前缀，`0073` 是 115 的十六进制），字符 `"牛"` 的码点是 29275。在 Python 中，可以用 `ord()` 函数将单个 Unicode 字符转为整数表示，用 `chr()` 函数将整数码点转为对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

### Problem (unicode1): Understanding Unicode (1 point)

**(a)** `chr(0)` 返回什么 Unicode 字符？

**Deliverable**: 一句话回答。

**(b)** 这个字符的字符串表示 (`__repr__()`) 与它的打印表示有何不同？

**Deliverable**: 一句话回答。

**(c)** 当这个字符出现在文本中时会发生什么？用以下代码在 Python 解释器中实验：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**Deliverable**: 一句话回答。

---

## 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不切实际的，因为词汇表太大（约 15 万项），且许多字符非常罕见。我们将使用 Unicode **编码** (encoding)，将 Unicode 字符转换为字节序列。Unicode 标准定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主导编码（超过 98% 的网页使用）。

在 Python 中用 `encode()` 函数将 Unicode 字符串编码为 UTF-8。要访问 Python `bytes` 对象的底层字节值，可以迭代它（如调用 `list()`）。用 `decode()` 函数将 UTF-8 字节串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码字符串的字节值（0 到 255 的整数）
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 129, 130, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 码点转换为字节序列（如通过 UTF-8 编码），我们实质上是将一个码点序列（21 位整数，有 159,801 个有效值）转换为一个字节值序列（0 到 255 的整数）。256 长度的字节词汇表要可管理*得多*。使用字节级分词时，我们不需要担心词汇外 (out-of-vocabulary) token，因为**任何**输入文本都可以表示为 0 到 255 的整数序列。

### Problem (unicode2): Unicode Encodings (3 points)

**(a)** 为什么我们更倾向于在 UTF-8 编码的字节上训练分词器，而不是 UTF-16 或 UTF-32？比较这些编码在不同输入字符串上的输出可能会有帮助。

**Deliverable**: 一到两句话回答。

**(b)** 考虑以下（不正确的）函数，它试图将 UTF-8 字节串解码为 Unicode 字符串。为什么这个函数不正确？请给出一个会产生错误结果的输入字节串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**Deliverable**: 一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串示例，并用一句话解释为什么该函数不正确。

**(c)** 给出一个不能解码为任何 Unicode 字符的两字节序列。

**Deliverable**: 一个例子，附一句话解释。
