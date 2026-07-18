# Infini-Transformer和Mixture-of-Depth

> 作者: zartbot  
> 日期: 2024年4月15日 02:36  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489371&idx=1&sn=f46813c8e45e8b75c630198dff5eef50&chksm=f9960799cee18e8f82cd1157694cdf6188bd0a0f729941f901bb63292455061daa90dee7d293#rd

---

继续来看一些和大模型相关的论文，主要是针对Transformer的计算复杂度和内存占用优化，对于Transformer模型，Context长度为，。Infini-Transformer主要是针对长文本的情况，PnP**即插即用的模型长文本增强能力值得关注，本质上是对无状态的Attention计算添加一个Context依赖的状态函数的做法，但是否真的无限长度这个事情感觉有点博眼球了。

另一方面是MoE基础上构建另一种动态路由机制MoD，可能是未来的一个方向。但是针对MoE/MoD这样的方式是否能在软件框架/通信网络和硬件结构上做一些更动态图的优化呢？例如Pathways这样的框架？

### 1. Infini-Transformer

Infini-Transformer来自于论文《Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention》[1] ,论文称可以使Transformer LLM能够有效地处理无限长输入，同时保持有限的内存占用和计算量。

关键的方法是引入了一种Infini-Attention的注意力技术：

![图片](assets/35be8d18ece5.png)
Infini-attention将压缩记忆融入标准注意力机制，并在单个Transformer块中集成了掩码局部注意力和长期线性注意力机制。对Transformer注意力层的这一微妙但关键的修改，使现有LLM可通过持续预训练和微调自然地扩展到无限长上下文处理。Infini-attention重用了标准注意力计算中所有Key、Value和Query状态，用于长期记忆巩固和检索。通过将注意力的旧KV状态存储在压缩记忆中，而非像标准注意力机制那样丢弃。处理后续序列时，使用注意力查询状态从记忆中检索值。为了计算最终的上下文输出，Infini-attention聚合了从长期记忆中检索到的值和局部注意力上下文。

文中对比了Transformer-XL[2]，它采用分段处理的方式，即Segment-level recurrence，在计算每个segment的时候，缓存上一个segment的信息，把前面segment的信息加入正向传播过程的计算，但是它丢弃了上一个Segment的状态

![图片](assets/12650bc0b7ba.png)

而Infini-Transformer则是将Segment的信息更新到一个压缩内存和线性Attention结构中，本质上可以看作是一个带有状态函数的Transformer-XL

![图片](assets/1cd8f5e8b51f.png)

作者总结了三个贡献：

引入了一种实用且强大的注意力机制——Infini-attention，它具有长期压缩记忆和局部因果注意力，能够高效建模长、短距离上下文依赖。

Infini-attention对标准缩放点积注意力的改动最小，并通过设计支持即插即用的持续预训练和长上下文适应能力。

该方法使Transformer LLM能够以流式处理方式处理极长输入，以有限的内存和计算资源处理无限长上下文。

#### 1.1 Compressive Memory

`读取内存`: 对内存  根据Query向量 采用如下非线性激活函数提取和Normalize()，主要是基于训练稳定性的考虑

`内存更新`: 读取后，对内存按照如下方式更新

同时还受到Delta方法的启发，在更新内存时采用减去提取内存的方法进行增量更新

提取的记忆Attention 和原始的标准Attention计算结果构成一个组合传递到后面的网络

#### 1.2 LLM Continual Pre-training

该模型比较有趣的一个做法是，对于原有模型的Attention结构保留，而采用即插即用的方式增加压缩内存，如下图所示，右边是完全保留的原有模型

![图片](assets/63e44fd5979a.png)

这样就可以拿已有的预训练模型对长文本数据进行训练，这个想法不错。

#### 1.3 一些个人的观点

简单来说，这个模型就是在原Attention的机制上，针对长文本任务，增加了一个状态函数，通过将临时计算的上一个Segment的attention映射到一块内存中，然后再动态的提取出来给下一个Segment计算时和标准的Attention进行加权处理。

创新点在于可以做FineTune，但是Infini这个说法感觉有点哗众取宠了。

### 2. Mixture-of-Depths

另一篇论文是《Mixture-of-Depths: Dynamically allocating compute in transformer-based language models》[3]正如文章摘要所述：基于Transformer的语言模型在整个输入序列中均匀分配FLOPs**（或计算量）。而在这项工作中，证明Transformer可以学会动态地将FLOPs（或计算量）分配给序列中的特定位置，沿着模型深度的不同层优化序列中的分配。

并非所有问题都需要相同的时间或努力来解决。类似地，在语言建模中，不需要为所有Token和序列花费相同的时间或努力来准确做出预测。然而，Transformer模型在前向传播中为每个Token花费相同的计算量。理想情况下，Transformer应通过避免不必要的计算。

本质上是我们可以分析每一层的参数分布，例如论文《Locating and Editing Factual Associations in GPT》[4]实际上token在路径上是有一些因果依赖的
![图片](assets/80df9b02c1f3.png)

但是如果采用Conditional Computing的机制，则需要引入动态图，对于当前的硬件架构来说是不友好的，所以希望构建出一种对GPU当前硬件结构友好的做法构建静态图并且最大化MFU的方式。因此网络必须学习如何通过在每一层为每个Token做出关于从可用预算中在哪里花费计算的决策来动态分配可用计算。因此实现中，总计算量是由用户在训练前定义并保持不变的，而不是网络实时决策的函数。

相对于传统的MoE对模型的MLP进行划分，MoD的方法大致的方法就是另一种路径划分，通过Route选择一部分的参数进行计算。利用路由函数去跳过一些Attention的计算

![图片](assets/f9f9922cc5e4.png)

另一方面它还可以和MoE结合构成MoDE

![图片](assets/cf7a29736e47.png)

从Data Flow Space的角度来看，一个是横向剖分，一个是纵向剖分。但是动态路由机制和算力分配是一个难题，特别是现在很多模型训练是SPMD**的，如何保证整个系统的FLOPS消耗是负载均衡的，这是MoD和MoE模型需要考虑的。而其中最关键的是在路由机制的设计上。

通过随机性来路由Token，类似于层或块“dropout”，虽然算力相对平衡了，但是这会导致性能显著下降。那么作者假设学习路由更为可取，从直观上来讲，网络应该能够学习哪些Token需要更多或更少的处理。因此考虑了两种方案：

`token-choice路由`，路由器为每个Token生成跨越计算路径的概率分布。然后，Token被送到它们偏好的路径——即概率最高的路径——辅助损失(auxiliary balancing loss)确保所有Token不会收敛到同一条路径。token-choice路由可能存在负载平衡问题，因为没有保证Token在可能的路径之间适当分配。

`Expert-choice路由`:每个路径根据Token的偏好选择前k个Token。这确保了完美的负载平衡，因为k个Token保证被送到每个路径。然而，这可能导致某些Token过度或不足处理，因为一些Token可能是多个路径的前k个，或者都不是。

![图片](assets/fda5877acbda.png)

使用token-choice路由（左图）时，Token会被引导至它们选择的计算路径。如果某个路径超过了其容量（例如，本例中超过两个Token），则必须丢弃多余的Token（紫色Token）。最终丢弃的确切Token取决于底层代码的精确实现。例如，通常会优先考虑序列或批次顺序较早的Token。

使用expert-choice路由（中间），每个路径都精确选择k个（本例中为两个）Token，通过在Token的路由器权重上采用top-k机制。这里，如果Token不属于任何给定路径的top-k，则会丢弃它们（橙色Token），甚至有些Token可能会被引导至多个路径（黄色Token）。

作者部署的expert-choice路由（右图）。由于只使用一条路径，所以利用了这样一个隐含知识：如果k小于序列长度，Token就会被丢弃，这样我们就可以将Token从自注意力和MLP计算中路由出去。

作者选择了expert-choice路由：

它消除了对辅助平衡损失(auxiliary balancing loss)的需求。

由于top-k操作取决于路由器权重的大小，这种路由方案允许相对路由权重帮助确定哪些Token最需要块的计算；路由器可以通过适当地设置它们的权重来尝试确保最重要的Token是前k个，这是token-choice路由方案无法做到的。对于特定应用场景，其中一个计算路径本质上是空操作，确保重要的Token远离空操作可能至关重要。

因为只通过两条路径路由，一个单一的top-k操作可以有效地将Token分成两组互斥的集合，一组用于每个计算路径，防止上述的过度或不足处理问题。

参考资料

[1] 
Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention: https://arxiv.org/abs/2404.07143
[2] 
Transformer-XL: https://arxiv.org/abs/1901.02860v3
[3] 
Mixture-of-Depths: Dynamically allocating compute in transformer-based language models: https://arxiv.org/pdf/2404.02258.pdf
[4] 
Locating and Editing Factual Associations in GPT: https://arxiv.org/pdf/2202.05262.pdf