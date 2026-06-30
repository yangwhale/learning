# CS336 作业 1（基础）：构建 Transformer 语言模型

**版本 26.0.3 | CS336 教学团队 | Spring 2026**

> 本文档为 Stanford CS336 Assignment 1 的完整中文翻译。每个作业题目被拆分到独立文件夹中，包含完整的背景知识、题目要求、代码示例和公式。

---

## 1. 作业概述

在本次作业中，你将从零开始构建训练标准 Transformer 语言模型 (LM) 所需的所有组件，并训练一些模型。

### 你将实现

1. 字节对编码 (BPE) 分词器（[Section 2](02_bpe_training/)）
2. Transformer 语言模型 (LM)（[Section 3](09_transformer/)）
3. 交叉熵损失函数和 AdamW 优化器（[Section 4](10_loss_and_optimizer/)）
4. 训练循环，支持序列化和加载模型与优化器状态（[Section 5](12_training_loop/)）

### 你将运行

1. 在 TinyStories 数据集上训练 BPE 分词器
2. 用训练好的分词器对数据集编码，转换为整数 ID 序列
3. 在 TinyStories 数据集上训练 Transformer LM
4. 使用训练好的 Transformer LM 生成样本并评估困惑度
5. 在 OpenWebText 上训练模型，提交困惑度到排行榜

### 你可以使用的工具

我们期望你从零构建每个组件。特别地，你**不能**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义，以下情况除外：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（如 `Module`, `ModuleList`, `Sequential` 等）。完整列表见 [pytorch.org/docs/stable/nn.html#containers](https://pytorch.org/docs/stable/nn.html#containers)
- `torch.optim.Optimizer` 基类

你可以使用其他任何 PyTorch 定义。如果不确定某个函数或类是否可用，请在 Slack 上询问。有疑问时，考虑使用它是否违背了本作业"从零构建"的精神。

### 关于 AI 工具的声明

AI 可以完全自主地解决作业的很多部分，但这会让你更难深入参与并从课程材料中学习。

- **允许**：使用 AI 工具回答高层概念性问题，或提供底层编程文档（如函数签名和库 API）
- **不允许**：使用 AI 工具实现作业的任何部分。包括编程 agent（如 Cursor Agents, Codex, Claude Code）和 AI 自动补全（如 Cursor Tab, GitHub Copilot）

我们强烈建议在完成作业时在 IDE 中禁用 AI 自动补全。使用 AI agent 时，请确保它使用提供的 AGENTS.md 文件。使用聊天机器人时也应包含该 prompt。

### 代码结构

作业代码和本文档均可在 GitHub 获取：

```
github.com/stanford-cs336/assignment1-basics
```

请 `git clone` 该仓库。代码结构：

1. **`cs336_basics/*`**：你编写代码的地方。注意里面没有现成代码——你可以随意组织。
2. **`adapters.py`**：定义了你的代码必须具备的功能接口。对每个功能（如 `run_scaled_dot_product_attention`），填写其实现来调用你的代码。`adapters.py` 不应包含任何实质性逻辑，它只是胶水代码。
3. **`test_*.py`**：包含你必须通过的所有测试（如 `test_scaled_dot_product_attention`），会调用 `adapters.py` 中定义的钩子函数。不要编辑测试文件。

### 如何提交

运行 `make_submission.sh` 构建提交 zip 文件。确保将大型数据文件或 checkpoint 添加到排除列表中。

你将向 Gradescope 提交：
- **writeup.pdf**：回答所有书面问题（请排版）
- **code.zip**：包含你写的所有代码

提交到排行榜，向以下仓库提交 PR：
```
github.com/stanford-cs336/assignment1-basics-leaderboard
```

见排行榜仓库的 `README.md` 获取详细提交说明。

### 获取数据集

本作业使用两个预处理好的数据集：
- **TinyStories** [R. Eldan et al., 2023]
- **OpenWebText** [A. Gokaslan et al., 2019]

两个数据集都是单个大纯文本文件。如果你是课程注册学生，可在 compute guide 中找到下载说明。如果你在自行学习，可通过 `README.md` 中的命令下载。

> **低资源提示：初始化**
> 在课程作业讲义中，我们会给出使用较少或没有 GPU 资源完成部分作业的建议。例如，我们有时会建议**缩小**你的数据集或模型大小，或解释如何在 Mac 集成 GPU 或 CPU 上运行训练代码。你会在蓝色框中找到这些"低资源提示"。即使你是有课程机器权限的 Stanford 注册学生，这些提示也可以帮你更快迭代、节省时间，推荐阅读！

> **低资源提示：在 Apple Silicon 或 CPU 上完成 Assignment 1**
> 使用教学团队的参考解决方案代码，我们可以在 Apple M4 Max 芯片（36 GB RAM）上训练 LM 生成合理流畅的文本，在 Metal GPU (MPS) 上不到 5 分钟，在 CPU 上约 30 分钟。只要你有一台相当新的笔记本电脑，且你的实现正确高效，你就能训练出一个小型 LM 来生成简单的、流畅度尚可的儿童故事。后续在作业中我们会解释如果你使用 CPU 或 MPS 需要做哪些改动。

---

## 2. 作业题目导航

本作业共 **38 道题**，总分约 **130 分**。按主题分为 17 个模块，每个模块对应一个独立文件夹：

### Part I: BPE 分词器（Section 2）

| 文件夹 | 题目 | 分值 |
|--------|------|------|
| [01_unicode](01_unicode/) | `unicode1` (1pt) + `unicode2` (3pt) | 4 |
| [02_bpe_training](02_bpe_training/) | `train_bpe` (15pt) + `train_bpe_tinystories` (2pt) + `train_bpe_expts_owt` (2pt) | 19 |
| [03_tokenizer](03_tokenizer/) | `tokenizer` (15pt) + `tokenizer_experiments` (4pt) | 19 |

### Part II: Transformer 架构（Section 3）

| 文件夹 | 题目 | 分值 |
|--------|------|------|
| [04_building_blocks](04_building_blocks/) | `linear` (1pt) + `embedding` (1pt) | 2 |
| [05_rmsnorm](05_rmsnorm/) | `rmsnorm` (1pt) | 1 |
| [06_feedforward](06_feedforward/) | `positionwise_feedforward` (2pt) | 2 |
| [07_rope](07_rope/) | `rope` (2pt) | 2 |
| [08_attention](08_attention/) | `softmax` (1pt) + `scaled_dot_product_attention` (5pt) + `multihead_self_attention` (5pt) | 11 |
| [09_transformer](09_transformer/) | `transformer_block` (3pt) + `transformer_lm` (3pt) + `transformer_accounting` (5pt) | 11 |

### Part III: 训练（Section 4-5）

| 文件夹 | 题目 | 分值 |
|--------|------|------|
| [10_loss_and_optimizer](10_loss_and_optimizer/) | `cross_entropy` (1pt) + `learning_rate_tuning` (1pt) + `adamw` (2pt) + `adamw_accounting` (2pt) | 6 |
| [11_lr_schedule_and_clipping](11_lr_schedule_and_clipping/) | `learning_rate_schedule` (1pt) + `gradient_clipping` (1pt) | 2 |
| [12_training_loop](12_training_loop/) | `data_loading` (2pt) + `checkpointing` (1pt) + `training_together` (4pt) | 7 |

### Part IV: 文本生成（Section 6）

| 文件夹 | 题目 | 分值 |
|--------|------|------|
| [13_decoding](13_decoding/) | `decoding` (3pt) | 3 |

### Part V: 实验（Section 7）

| 文件夹 | 题目 | 分值 |
|--------|------|------|
| [14_experiments_tinystories](14_experiments_tinystories/) | `experiment_log` (3pt) + `learning_rate` (3pt) + `batch_size_experiment` (1pt) + `generate` (1pt) | 8 |
| [15_ablations](15_ablations/) | `layer_norm_ablation` (1pt) + `pre_norm_ablation` (1pt) + `no_pos_emb` (1pt) + `swiglu_ablation` (1pt) | 4 |
| [16_main_experiment](16_main_experiment/) | `main_experiment` (2pt) | 2 |
| [17_leaderboard](17_leaderboard/) | `leaderboard` (6pt) | 6 |

---

## 3. 架构图

### Figure 1: Transformer 语言模型概览

![Transformer LM Overview](images/figure1_transformer_lm.png)

### Figure 2: Pre-norm Transformer Block

![Pre-norm Transformer Block](images/figure2_transformer_block.png)

### Figure 3: SiLU vs ReLU 激活函数对比

![SiLU vs ReLU](images/figure3_silu_relu.png)

---

## 4. 参考文献

1. R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.
2. A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.
3. R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in *Proc. of ACL*, 2016.
4. C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.
5. P. Gage, "A new algorithm for data compression," *C Users Journal*, vol. 12, no. 2, pp. 23–38, Feb. 1994.
6. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.
7. A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.
8. A. Vaswani et al., "Attention is All you Need," in *Proc. of NeurIPS*, 2017.
9. T. Q. Nguyen and J. Salazar, "Transformers without Tears," in *Proc. of IWSWLT*, 2019.
10. R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in *Proc. of ICML*, 2020.
11. J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.
12. H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.
13. B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in *Proc. of NeurIPS*, 2019.
14. A. Grattafiori et al., "The Llama 3 Herd of Models." 2024. https://arxiv.org/abs/2407.21783
15. A. Yang et al., "Qwen2.5 Technical Report," *arXiv:2412.15115*, 2024.
16. A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.
17. D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.
18. S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." 2017. https://arxiv.org/abs/1702.03118
19. Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." 2016. https://arxiv.org/abs/1612.08083
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
