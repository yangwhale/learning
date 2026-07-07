"""BPE (Byte-Pair Encoding) 分词器实现

这个模块实现了字节级 BPE 训练算法和 Tokenizer 类。
"""

import json
import os
import regex as re
from collections import defaultdict, Counter
from multiprocessing import Pool
from typing import BinaryIO, Iterable, Iterator, Optional


# GPT-2 使用的预分词正则表达式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _get_pairs(word: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    """返回一个词里所有相邻 pair（可能有重复，比如 'a a a' 会有两个 (a,a)）。"""
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def _merge_word(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    new_token: bytes,
) -> tuple[bytes, ...]:
    """在一个词里把所有 pair 替换成 new_token（左到右贪心，不重叠）。"""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(new_token)  # 命中 pair，用新 token 替换这两个
            i += 2
        else:
            new_word.append(word[i])    # 没命中，原样保留
            i += 1
    return tuple(new_word)


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """把文件切成若干块，切点落在 split_special_token 处（保证不切坏词/字符）。

    拷自 Stanford 的 pretokenization_example.py。返回一串字节偏移量，
    实际块数可能少于 desired_num_chunks（若切点重叠）。
    """
    assert isinstance(split_special_token, bytes), "special token 必须是 bytes"

    # 取文件总字节数
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 先均匀猜切点，再往后挪到最近的 special token 处
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次往后读 4K 找 special token

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":  # 到文件尾还没找到 → 切点设在文件末
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:  # 找到 special token → 切点设在它开头
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 去重排序（重叠的切点会被合并，块数因此可能变少）
    return sorted(set(chunk_boundaries))


def _count_word_freqs(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """把一段文本转成 word_freqs（含 special token 保护 + GPT-2 正则预分词）。

    串行和并行两条路径都调这个，保证逻辑一致。
    """
    # Step 3a: 先按 special token 切开，把它摘出去（防止碎片污染 vocab）
    if special_tokens:
        # 按长度降序：避免短 token 是长 token 前缀时把长的切坏
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        # 捕获组 (…) 让 re.split 保留分隔符，才能识别哪些段是 special token
        split_pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
        segments = re.split(split_pattern, text)
    else:
        segments = [text]

    special_set = set(special_tokens)
    pattern = re.compile(GPT2_SPLIT_PATTERN)

    # Step 3b + 4: 每段独立预分词，special token 段跳过
    word_freqs = defaultdict(int)
    for segment in segments:
        if segment in special_set:
            continue  # special token 段：已在 vocab 注册，不进 BPE 统计
        for chunk in pattern.findall(segment):
            word_tokens = tuple(bytes([b]) for b in chunk.encode('utf-8'))
            word_freqs[word_tokens] += 1
    return word_freqs


def _process_chunk(args: tuple[str, int, int, list[str]]) -> dict[tuple[bytes, ...], int]:
    """multiprocessing worker：读文件的 [start, end) 字节段 → word_freqs。

    必须是模块级函数，才能被 pickle 传给子进程。
    """
    input_path, start, end, special_tokens = args
    with open(input_path, 'rb') as f:
        f.seek(start)                                    # 跳到本块起点
        text = f.read(end - start).decode('utf-8', errors='ignore')  # 只读本块
    return dict(_count_word_freqs(text, special_tokens))


def train_bpe(
    input_path: str,
    vocab_size: int = 256,
    special_tokens: Optional[list[str]] = None,
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 BPE 分词器。

    Args:
        input_path: 训练语料文本文件的路径
        vocab_size: 目标词汇表大小（必须 >= 256）
        special_tokens: 特殊 token 列表（如 ["<|endoftext|>"]）
        num_processes: 并行进程数。默认 1（串行，读全部）。
            > 1 时按 special token 边界分块，多进程并行统计词频——
            用于大文件（几 GB+），小文件用并行反而因进程池开销更慢。
            并行需要至少一个 special token 作为安全切分点。

    Returns:
        vocab: dict[int, bytes] — token ID → token 字节串的映射
            例子: {0: b'\x00', 65: b'A', 257: b' t', 258: b'he', 259: b' the', ...}
            - 单字节 token: 65 → b'A' (1个字节)
            - 多字节 token: 259 → b' the' (4个字节: [32, 116, 104, 101])
            - 用途: 解码必需（ID→字节串→文本），编码时也需要（查字节串对应的ID）

        merges: list[tuple[bytes, bytes]] — 合并操作列表（按训练顺序）
            例子: [(b' ', b't'), (b'h', b'e'), (b' t', b'he'), ...]
            - 每个 tuple 记录了哪两个字节串被合并
            - 用途: 编码必需（按顺序应用合并规则），解码不需要
            - 顺序很重要：必须按训练时的顺序应用

    算法步骤：
        1. 初始化词汇表：256 个单字节 token (0-255) + special tokens
        2. 读取训练文本
        3. 用 GPT-2 正则表达式做预分词
        4. 将每个 chunk 转为 UTF-8 字节序列
        5. 统计所有相邻 byte pair 的频率
        6. 循环合并最高频 pair，直到达到目标词汇表大小
        7. 返回 vocab 和 merges
    """
    # 参数验证
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")

    special_tokens = special_tokens or []

    # ==========================================
    # Step 1: 初始化词汇表
    # ==========================================
    # 0-255: 单字节 tokens
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # 添加 special tokens
    for token in special_tokens:
        vocab[next_token_id] = token.encode('utf-8')
        next_token_id += 1

    # ==========================================
    # Step 2-4: 读取 + 预分词 + 统计词频（含 special token 保护）
    # ==========================================
    # word_freqs: dict[tuple[bytes, ...], int]
    # 每个 word 是一个 tuple of tokens，每个 token 是 bytes（初始时每字节一个 token）
    #
    # 两条路径，结果完全一致（词频累加与顺序无关）：
    #   - 串行（默认）：一次性读全部，适合小文件
    #   - 并行：按 special token 边界分块，多进程统计后合并，适合大文件

    can_parallelize = num_processes > 1 and len(special_tokens) > 0
    if can_parallelize:
        # 并行路径：先找安全切点（落在 special token 处，不会切坏词/字符）
        split_token = special_tokens[0].encode('utf-8')
        with open(input_path, 'rb') as f:
            boundaries = _find_chunk_boundaries(f, num_processes, split_token)

        # 每个 (start, end) 段交给一个子进程独立统计
        chunk_args = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        with Pool(num_processes) as pool:
            partial_freqs = pool.map(_process_chunk, chunk_args)

        # 合并各块的词频（这就是我们之前讨论的"n 份统计加一起"）
        word_freqs = defaultdict(int)
        for pf in partial_freqs:
            for word, freq in pf.items():
                word_freqs[word] += freq
    else:
        # 串行路径：读全部文本，一把统计
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        word_freqs = _count_word_freqs(text, special_tokens)

    # ==========================================
    # Step 5: 初始化 pair 计数 + 倒排索引（只扫一次全部语料）
    # ==========================================
    # pair_freqs:    pair → 全语料里的总频率（跨轮存活，只做增量修改，不重建）
    # pair_to_words: pair → 含这个 pair 的词集合（倒排索引，用于快速定位受影响的词）
    pair_freqs = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in word_freqs.items():
        for pair in _get_pairs(word):
            pair_freqs[pair] += freq
            pair_to_words[pair].add(word)

    # ==========================================
    # Step 6: 循环合并，每轮只增量更新受影响的词
    # ==========================================
    merges = []
    num_merges_needed = vocab_size - len(vocab)

    for _ in range(num_merges_needed):
        if not pair_freqs:
            break

        # 找频率最高的 pair；平局时选字典序最大的（(count, pair) 比较）
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        if pair_freqs[best_pair] <= 0:
            # 只剩下已失效的残留计数，没有真正可合并的 pair
            break

        # 生成新 token，登记到 vocab / merges
        new_token = best_pair[0] + best_pair[1]  # bytes + bytes = bytes
        vocab[next_token_id] = new_token
        merges.append(best_pair)
        next_token_id += 1

        # 只处理含 best_pair 的词（倒排索引查出来的那几个，不碰其他词）
        # 先复制成 list：循环里会改动 pair_to_words[best_pair]，不能边改边遍历
        affected_words = list(pair_to_words[best_pair])

        for word in affected_words:
            freq = word_freqs.pop(word)  # 旧词即将消失，取出频率并移除

            # (1) 撤销旧词的全部 pair 贡献
            old_pairs = _get_pairs(word)
            for pair in old_pairs:
                pair_freqs[pair] -= freq          # 计数按出现次数递减
            for pair in set(old_pairs):
                pair_to_words[pair].discard(word)  # 倒排索引按去重后的 pair 移除

            # (2) 生成合并后的新词
            new_word = _merge_word(word, best_pair, new_token)

            # (3) 加上新词的全部 pair 贡献
            new_pairs = _get_pairs(new_word)
            for pair in new_pairs:
                pair_freqs[pair] += freq
            for pair in set(new_pairs):
                pair_to_words[pair].add(new_word)

            # (4) 新词写回 word_freqs（合并可逆 → 新词唯一，不会撞已有词）
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

    # ==========================================
    # Step 7: 返回结果
    # ==========================================
    return vocab, merges


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """GPT-2 的"字节→可打印 unicode 字符"映射（用于加载 GPT-2 格式的 vocab/merges）。

    GPT-2 把 0-255 每个字节映射到一个可打印 unicode 字符，避免 vocab.json 里
    出现控制字符/空白导致解析问题。from_files 加载时要用它的反向映射还原出真字节。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Tokenizer:
    """BPE Tokenizer —— 文本 ↔ token IDs 的编解码。

    - encode:  文本 → token IDs（按 merge rank 顺序重放 BPE 合并）
    - decode:  token IDs → 文本（查 vocab 拿 bytes，拼接后一次性 UTF-8 解码）
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        """初始化 Tokenizer。

        Args:
            vocab: token ID → bytes 的映射
            merges: BPE 合并规则列表（按训练顺序，list 下标即 rank）
            special_tokens: 特殊 token 列表（永不被切分，整体保留为一个 token）
        """
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        # merge 优先级：pair → rank（下标越小越早学、越优先合并）
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # 确保每个 special token 都在 vocab 里（不在就补一个新 ID）
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = tok_bytes

        # 反向映射：bytes → token ID（编码时查合并后的 token 对应哪个 ID）
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # 预编译 GPT-2 预分词正则
        self._pat = re.compile(GPT2_SPLIT_PATTERN)

        # special token 切分正则：按长度降序，让重叠时长的优先匹配
        # 例如 <|endoftext|><|endoftext|> 要整体匹配，而不是拆成两个 <|endoftext|>
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pat = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
            )
        else:
            self._special_pat = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[list[str]] = None,
    ) -> "Tokenizer":
        """从 GPT-2 格式的 vocab.json + merges.txt 加载 Tokenizer。

        GPT-2 文件里的 token 用"可打印 unicode 字符"表示字节，加载时要用
        _gpt2_bytes_to_unicode 的反向映射还原成真正的 bytes。
        """
        byte_decoder = {c: b for b, c in _gpt2_bytes_to_unicode().items()}

        # vocab.json: {token_str: id} → {id: bytes}
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab = {
            idx: bytes(byte_decoder[ch] for ch in token_str)
            for token_str, idx in gpt2_vocab.items()
        }

        # merges.txt: 每行 "token1 token2" → (bytes1, bytes2)
        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line or len(line.split(" ")) != 2:
                    continue  # 跳过空行和文件头注释
                a, b = line.split(" ")
                merges.append(
                    (bytes(byte_decoder[ch] for ch in a),
                     bytes(byte_decoder[ch] for ch in b))
                )

        return cls(vocab, merges, special_tokens)

    def _bpe_encode_pretoken(self, pretoken: str) -> list[int]:
        """对单个 pretoken 应用 BPE 合并，返回 token IDs。

        核心：每轮找当前所有相邻 pair 里 rank 最小的合并，反复直到无规则可用。
        （不是"最长优先"，是严格按 rank 顺序重放训练时的合并路径。）
        """
        # 起点：每个字节一个 token
        parts = [bytes([b]) for b in pretoken.encode("utf-8")]

        while len(parts) >= 2:
            # 找 rank 最小的可合并 pair
            best_rank = None
            best_i = None
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i

            if best_i is None:
                break  # 没有任何 pair 在 merges 里 → 停

            # 合并 best_i 处的两个 token
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]

        return [self.inverse_vocab[p] for p in parts]

    def _encode_yield(self, text: str) -> Iterator[int]:
        """把一段文本编码成 token IDs 流（内部生成器，供 encode/encode_iterable 复用）。"""
        # 先按 special token 切开（保留它们），特殊段直接映射成 ID，普通段走 BPE
        if self._special_pat is not None:
            segments = self._special_pat.split(text)
        else:
            segments = [text]

        special_set = set(self.special_tokens)
        for segment in segments:
            if not segment:
                continue
            if segment in special_set:
                # special token：整体映射成它的 ID，不进 BPE
                yield self.inverse_vocab[segment.encode("utf-8")]
            else:
                # 普通文本：GPT-2 正则预分词，每个 pretoken 独立跑 BPE
                for pretoken in self._pat.findall(segment):
                    yield from self._bpe_encode_pretoken(pretoken)

    def encode(self, text: str) -> list[int]:
        """将文本编码为 token IDs 列表。"""
        return list(self._encode_yield(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性编码一个可迭代的字符串源（如文件句柄），逐个 yield token ID。

        内存友好：一次只处理一块（一行），不把整个文件读进内存。
        行以 '\\n' 结尾，天然是安全边界（不会切断 UTF-8 字符或 special token）。
        """
        for chunk in iterable:
            yield from self._encode_yield(chunk)

    def decode(self, ids: list[int]) -> str:
        """将 token IDs 解码为文本。"""
        # 先把所有 token 的 bytes 拼起来，再【一次性】UTF-8 解码
        # ——因为一个 UTF-8 字符可能跨多个 token，逐 token 解码会出错
        replacement = "�".encode("utf-8")  # 未知 ID 用替换字符
        data = b"".join(self.vocab.get(i, replacement) for i in ids)
        # errors='replace'：非法字节序列不报错，替换成 U+FFFD
        return data.decode("utf-8", errors="replace")
