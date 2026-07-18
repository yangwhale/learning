# 谈谈DeepMind会做算法导论的TransNAR并引出基于SAE-GNN的可组合Transformer猜想

> 作者: zartbot  
> 日期: 2024年6月19日 05:57  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490297&idx=1&sn=7d758e84bdce7ae4f20f031f4ac3f221&chksm=f9960a3bcee1832d58956a286d2bc33ca32c69edfb3a00cdad65aec691100f2cc086649aa1bf#rd

---

大模型在一些算法类任务上表现还是有一定的差距,例如前些日子测试了一下大模型数数都有些数不清楚, 虽然唯一一个数正确的文心一言,也还在后台调用了一个python函数执行. 所以现阶段想用大模型去打OI的比赛还是有很大的难度的,当然对一些八股文那样的算法题还是可以凭记忆拼凑的.

去年底在[《大模型时代的数学基础(4)》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&chksm=f996046acee18d7c687403c557a6e30155ba0c04cca7e897de3126e88a5d3ba3a2c2fe0507bd&scene=21#wechat_redirect)5.3节中谈到LLM+GNN算法

联结主义、符号主义的融合来看，基础模型本来就是一个预层范畴，推理过程可以看作是一个DFS搜索，拿么是否可以将这个FM范畴的态射变成一个图，然后在进一步在图上做推理？或者是基于图结构和范畴论中的态射可组合性以及自然变换等构建态射损失函数来生成数据进行训练？另外总是觉得Attention和Ray Tracing有一些相似的地方，我们是否可以把Attention的向量空间投射到另一个空间，然后通过类似于BVH的算法来降低运算复杂度呢？

前几天DeepMind**有一篇论文《Transformers meet Neural Algorithmic Reasoners》[1]也在讲述如何通过GNN的神经算法推理器和Transformer结合, 实现更加泛化、稳健、准确的LLM推理。

本文将先介绍一下DeepMind的工作, 然后再分析一些不足, 并提供了一个可供选择的改进算法.

## 1. DeepMind TransNAR会做《算法导论》题目

### 1.1 神经网络算法推理器NAR

Petar Veličković是我 从"Geometric Deep Learning"的研究开始就一直关注着的一位科学家, 他在2021年提出了一个算法《Neural Algorithmic Reasoning》[2]概念, 通过对抽象算法来训练一个算法推理器(实际上我们可以将它看成一个高维的潜空间构成的一个处理器神经网络), 构建一个复合算子

其中位编码器, 为解码器, 位一个高维潜空间的处理器网络. 然后再保持参数冻结, 把真实的问题通过另一个函数来训练编码器和解码器

为了评估NAR的工作, DeepMind基于算法导论的一些算法,构建了《The CLRS Algorithmic Reasoning Benchmark》[3]主要涵盖排序、搜索、贪心算法、动态规划、图算法、字符串算法和几何算法, 相关的代码可以访问:https://github.com/google-deepmind/clrs

然后在2022年出现了一篇《A Generalist Neural Algorithmic Learner》[4]的工作, 通过构建一个图神经网络来训练和适配多种算法

![图片](assets/5a0f73a251d6.png)

通常它是一个包含`input`,`hint`,`output`的数据集并用一个图表示, `hint`是一系列算法中间状态的时间序列. 例如下图是一个冒泡排序的图结构

![图片](assets/21077754bd90.png)

图片来自于论文《Neural Algorithmic Reasoning Without Intermediate Supervision》[5], 绿线位Encoder, 红线为Decoder

### 1.2 Transformer+NAR

《Transformers meet Neural Algorithmic Reasoners》构造了一个模型, 通过在LLM旁挂一个NAR实现了大模型的算法推理能力

![图片](assets/4daef820da62.png)

它通过一个预训练的GNN-Based NAR来增强Transformer, 通过输入一个文本描述的问题定义, 和一个基于CLRS-30的图表示算法作为提示, 然后通过cross-attention机制来获取答案

![图片](assets/bc94d29e63c3.png)

Transformer计算

max-MPNN

Cross Attention

### 1.3 TransNAR的限制

TransNAR是一种语言模型，结合了Transformer的语言理解能力与基于预训练图神经网络的神经算法推理器的强大算法推理能力，用于解决以自然语言指定的算法任务。虽然out-of-distribution外的场景性能有所提升,但是它需要同时输入文本和图表示, 这样带来了一些限制. 即便如此,这些工作证明了NAR对改善模型性能是有益的, 未来的研究可以促使这类想法在纯粹的单模态Transformer中得到应用。例如使用知识蒸馏等方法将GNN融合到标准Transformer中来降低对图数据流的依赖.

## 2. SAE-GNN Composable Transformer

实际上来看, NAR是一个基于GNN约束的思维链, 但是TransNAR还需要一个额外的Graph输入作为算法提示, 然后在计算的过程中Token by Token的迭代并通过NAR修改图结构并通过Cross-Attention机制影响Transformer输出的Token.

是否能有一种自包含的机制来将NAR等类似的多任务逻辑推理和运算能力嵌入到Transformer呢?接下来谈谈个人的一些想法.

### 2.1 Composable Transformer

Transformer的架构已经显示出了非常高效的信息压缩能力, 但是过度的压缩使得推理过程中的幻觉和一些计算/逻辑推理类任务还是存在缺陷, 虽然通过一些手段最近一年进步很明显, 但是最近的一些数学高考题来看似乎还是处于学渣水平.

假设一个经过充分训练的Dense Transformer模型**已经有足够的信息压缩在模型内,并将其作为基础模型(Foundation Model, FM)

旁置的稀疏图神经网络构成Adapter, 通过CrossAttention或者Activation**的权重修改来影响, 通过FM模型配合不同的GNN图构建稀疏的可组合性.

如下图所示

![图片](assets/769dea0917da.png)

在训练完base的FM后,然后固化住FM的参数,再来训练GNN. 并且这个GNN并不需要每层都有, 而仅是在靠近开头和结尾的地方抽取两层对residual的值旁路有些update.

接下来引入一个对Composable Transformer的假设. 我们是否可以共享一套FM的Dense参数, 然后通过不同的GNN adapter组合的方式来完成复杂任务?

![图片](assets/96d5a760a82e.png)

这种组合的方式其实也是自然社会多人分工领域知识的组织构建的结果, 也是一种Superposition和Composition视角上的融合

![图片](assets/bb01992a241e.png)

另外关于GNN和LLM结合有一篇Survey可以参考一下《Large Language Models on Graphs: A Comprehensive Survey》[6]

### 2.2 SAE-GNN

接下来有一个问题是, TransNAR需要额外的图输入, 而如何通过自包含和自训练的方式来构建呢? 很自然的盯上了SAE相关的工作.

![图片](assets/ab11ac4dacd1.png)

SAE可以抽取出大量的可解释的概念,无监督学习效果非常不错, 例如概念里对于代码等结构的分析

![图片](assets/ac10470b2b46.png)

Anthropic**还做了一个有趣的实验通过增强激活值影响输出

![图片](assets/f8a1c07acd98.png)

那么这里有一个假设, 通过这些SAE构成的高维潜空间, 在Token之间构建相应的态射,从而构建GNN是否可行? 正如我在前一期《谈谈大模型可解释性》中提及的一个观点

非常关键的一点是, 这些特征/概念的形成是在模型训练中无监督学习生成的. 从范畴论的视角上, 我们可以在图上进行进一步的归纳总结. 并通过交换图的视角来对概念的可组合性进行测试和约束,对错误token的产生以及有毒概念的Token产生通过一个旁路的小规模SAE+GNN模型进行拦截, 或者GNN都不需要, 一个简单的决策树模型可能就够了.

### 2.3 从生物学的视角来看待

预训练的过程为儿童2岁前突触形成的过程.而成年的过程则是一个逐渐的突触裁剪的过程.

![图片](assets/766e622605c7.png)

而现实社会需要各有所长的人再通过组织协同来构建一个更大的智能体, 而突触裁剪某种程度上影响了人的认知. 也即是需要多个突触裁剪后的人通过通信来进行组织协同.

另外一方面从人脑的构造来看, 左右脑的结构表现为左右半脑间的信息交流表现为协调跨半球间的兴奋和抑制性输入。

![图片](assets/8f8ccc28922e.png)

例如在一些在决策行为中需要协调兴奋性和抑制性影响的情况下才能实现, 通过GNN+LLM来构建. 概念本身来自于LLM SAE, 然后再决策路径上通过GNN来进行兴奋性(扩大Activation)或者抑制性(降低Activation)的旁路Adapter实现对next token预测的修改以及上下文的自包含映射.

### 2.4 结论

本文通过分析DeepMind TransNAR提出了一个假设,即利用Sparse AutoEncoder的无监督训练产生的概念,通过不同任务来构建的GNN作为Dense Transformer模型的Adapter,并通过单一Dense模型配合多个不同的GNN的方式来构建一种可组合的SAE-GNN Transformer结构.

对于概念约束和决策型任务以及一些推理性的任务上, GNN+LLM的结合并且配合CPU进行GNN相关的计算+GPU进行Dense LLM推理的方式协同, 共享一个经过充分训练的Dense模型对于显存的占用也相对较小, 然后GNN也可以通过跨越多层旁路注入的方式降低对主路径Dense模型推理速度的影响,并适当的隐藏稀疏计算的延迟.

这种方法看上去和Dense MoE有相似之处, 相当于在每个Expert内部嵌入了一个GNN. 另一方面不同的是,对于Activation的增益或者抑制仅需要在Transformer某一层实施即可,也不必像MoE那样每一层都需要处理.

当然这样的处理方式需要前置依赖是训练一个较好的高维SAE, 但是因为没有这样的开源数据集, 只能成为一个假设或者猜想存在.

另外这样的处理方式虽然和自然人脑的处理有些相似之处, 但对AI基础设施的构建也有一定的影响, 或许更适合GB200/GH200这样的平台进行训练优化(CPU offload sparse computation).

另一个感叹是Petar这些TransNAR的工作在DeepMind也做了快四年了,而国内真不确定有哪个机构能有这样的耐心去潜心研究这些, 估计大概只能PlanB去玩玩了.

参考资料

[1]
Transformers meet Neural Algorithmic Reasoners: https://arxiv.org/pdf/2406.09308
[2]
Neural Algorithmic Reasoning: https://arxiv.org/pdf/2105.02761
[3]
The CLRS Algorithmic Reasoning Benchmark: https://arxiv.org/pdf/2205.15659
[4]
A Generalist Neural Algorithmic Learner: https://arxiv.org/pdf/2209.11142
[5]
Neural Algorithmic Reasoning Without Intermediate Supervision: https://arxiv.org/pdf/2306.13411
[6]
Large Language Models on Graphs: A Comprehensive Survey: https://arxiv.org/abs/2312.02783