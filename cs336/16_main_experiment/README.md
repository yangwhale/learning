# Section 7.4: 在 OpenWebText 上运行 (Running on OpenWebText)

## 概述

现在转向更标准的预训练数据集。OpenWebText (OWT) 是从 web crawl 创建的数据集，其文本比 TinyStories 更加真实、复杂和多样化。

## OWT 数据示例

> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." With a few clicks here and there, you can find out that Noah Syndergaard's fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game's highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. [...]
>
> "He freaks us out." Harry Pavlidis
>
> [...]

---

## 注意事项

从 TinyStories 切换到 OWT 后，可能需要**重新调优超参数**，特别是：

- Learning rate（学习率）
- Batch size（批次大小）

---

## Problem: Experiment on OWT (2 B200 hrs) (2 points)

使用与 TinyStories 相同的模型架构和总训练迭代数，在 OpenWebText 上训练模型。

### Deliverable

1. **OWT 上的学习曲线**。描述 TinyStories 和 OWT 的 loss 差异——你如何解释这种差异？

2. **OWT 语言模型生成的文本**（格式与 TinyStories 生成文本相同）。文本的流畅度如何？为什么在相同模型架构和计算预算下，OWT 模型的输出质量比 TinyStories 模型更差？
