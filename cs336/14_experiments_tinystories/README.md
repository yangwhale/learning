# Section 7: 实验 (Experiments) — TinyStories

## 概述

现在把所有东西整合起来，在预训练数据集上训练（小型）语言模型。

## 7.1 如何运行实验和交付物

理解 Transformer 架构各组件背后原理的最佳方式是实际修改它们并亲自运行实验。没有什么能替代动手经验。

为了**快速**实验，我们在小规模模型（约 17M 总参数）和简单数据集（TinyStories）上运行。为了**一致性**，我们系统地消融各组件并变化超参数。为了**记录**，我们提交实验日志和学习曲线。

**重要**：确保定期评估验证集 loss，并记录梯度步数 (gradient steps) 和 wall-clock 时间。可以使用 Weights and Biases 等日志基础设施。

### Problem: Experiment Logging (3 points)

创建实验跟踪基础设施，追踪实验和 loss 曲线（同时相对于梯度步数和 wall-clock 时间绘制）。

#### Deliverable

- 日志基础设施代码
- 实验日志文档

---

## 7.2 TinyStories

使用 **TinyStories**（Eldan et al. [1]）——一个简单的数据集，模型可以在上面快速训练并展现有趣的行为。

### TinyStories 数据示例

> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, "Wow, that is a really amazing vase! Can I buy it?" The shopkeeper smiled and said, "Of course you can. You can take it home and show all your friends how amazing it is!" So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!

---

### 7.2.1 超参数调优

给定默认超参数如下：

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| Vocab size | 10000 | 词汇表大小 |
| Context length | 256 | 上下文长度 |
| d_model | 512 | 模型维度 |
| d_ff | 1344 | FFN 隐层维度（约 $\frac{8}{3} \times$ d_model，且为 64 的倍数） |
| RoPE theta | 10000 | RoPE 位置编码参数 |
| Number of layers | 4 | Transformer 层数 |
| Number of heads | 16 | 注意力头数（约 17M 非嵌入参数） |
| Total tokens processed | 327,680,000 | batch_size $\times$ total steps $\times$ context_length $\approx$ 此值 |

你需要通过试错找到合适的：

- Learning rate（学习率）
- Learning rate warmup（学习率预热）
- AdamW 超参数（$\beta_1$, $\beta_2$, $\varepsilon$）
- Weight decay（权重衰减）

### 7.2.2 整合所有模块

流程：用训练好的 BPE 分词器 → tokenize 训练数据集 → 在训练循环中训练模型。

**重要提示**：正确且高效的实现在 1 块 B200 GPU 上大约需要 **20-30 分钟**完成训练。如果运行时间明显过长，请检查以下可能的瓶颈：

- 数据加载效率
- Checkpointing 频率
- 验证集 loss 计算代码

### 7.2.3 调试模型架构的技巧

- **Overfit 单个 minibatch**：loss 应该能快速降到接近零。如果不能，说明模型实现有 bug。
- **设置断点检查中间张量的 shape**：确保各层输出维度符合预期。
- **监控激活值、权重和梯度的范数**：确保它们既不会爆炸也不会消失。

---

### Problem: Tune the Learning Rate (2 B200 hrs) (3 points)

**(a)** 对学习率进行超参数扫描，报告每个学习率对应的最终 loss。

#### Deliverable

- 多个学习率下的学习曲线
- 搜索策略说明
- 在 TinyStories 上达到 **per-token validation loss $\leq$ 1.45** 的模型

> **Low-Resource Tip: 在 CPU 或 Apple Silicon 上训练几步**
>
> 如果在 cpu/mps 上运行，可以将 total tokens 减少到 **40,000,000**，验证 loss 目标放宽到 **2.00**。
>
> 参考数据：M4 Max + 36GB RAM，batch size 32 $\times$ 5000 steps $\times$ 256 = 40,960,000 tokens：
> - cpu：82 分钟
> - mps：36 分钟
> - step 5000 时 validation loss：1.80
>
> 额外提示：
> - 训练 N 步时，调整 cosine schedule 使其在 step N 处结束
> - **mps 上不要使用 TF32 kernels**，不要设置 `torch.set_float32_matmul_precision('high')`（torch 2.9.0 的 mps 后端会静默使用有 bug 的 kernels，导致训练不稳定）
> - 可以用 `torch.compile` 加速：
>   - cpu 上：`model = torch.compile(model)`
>   - mps 上：`model = torch.compile(model, backend="aot_eager")`（mps 不支持 Inductor 编译，torch 2.9.0）

**(b)** 最佳学习率与发散点 (divergence point) 之间有什么关系？

#### Deliverable

- 包含至少一个发散 (diverge) run 的递增学习率学习曲线
- 收敛速率分析

---

### Problem: Batch Size Variations (1 B200 hr) (1 point)

Batch size 从 1 一直增大到 GPU 内存上限，包含 64 和 128 等常见值。

#### Deliverable

- 不同 batch size 下的学习曲线（注意：每个 batch size 需要重新调优学习率）
- 几句话讨论 batch size 对训练的影响

---

### TinyStories 模型生成示例

> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.

> **Low-Resource Tip: 在 CPU 或 Apple Silicon 上生成文本**
>
> 用 40M tokens 的低资源配置训练后，生成的文本应该类似英语但不如上面的示例流畅。示例输出：
>
> Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.
> "Hey, Polly! Let's go out!" said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!"
> As Sue watched the sky moved, what it was. She

---

### Problem: Generate Text (1 point)

使用解码器和训练好的 checkpoint 生成文本。你可能需要调整 temperature、top-p 等参数以获得最佳效果。

#### Deliverable

- 至少 256 tokens 的生成文本（或到第一个 `<|endoftext|>` 为止）
- 关于流畅度的简要评论
- 影响输出质量的至少两个因素
