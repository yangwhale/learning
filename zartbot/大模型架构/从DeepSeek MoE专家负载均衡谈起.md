# 从DeepSeek MoE专家负载均衡谈起

> 作者: zartbot  
> 日期: 2025年3月10日 00:15  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493405&idx=1&sn=3768b760428245600f77e2b636c8c4d6&chksm=f995f7dfcee27ec9c7fe6507b6c4404dcef76ed09dbd78f871abd5e8af37bf1b437e062ca851#rd

---

上周中的时候, 同事给我了一份线上DeepSeek-R1推理的Expert激活的数据用来研究一些专家负载均衡的算法, 当然这些线上数据来自于公司内部的请求, 从中观测出前面10层专家基本上是相对均衡的, 而越到后面不均衡程度差异越大. 当时讨论到这个问题时, 是怀疑内部的一些请求是否专注于电商领域而带来的不平衡特性, 于是做了一些研究. 恰好搜到Intel的一篇论文《Semantic Specialization in MoE Appears with Scale: A Study of DeepSeek-R1 Expert Specialization》[1]有一些基于语义的MoE分析专家的专业性相关的问题, 再加上前几天看到某个公众号采访某院长的一个比较有趣的说法:“Dense模型适合toB业务,MoE模型适合toC业务”. 因此做了一些分析, 在此记录下来.

### 1. 专家Overlap分析

从这篇论文的第一个Word-in-Context的实验来看, DeepSeek-R1的前面十层专家Overlap的概率相对于较高, 和线上的一些数据分析是一致的.
![图片](assets/e77f64519f4c.png)

比较特别的是在第十层后,不同语义和相似语义之间的区分度完全显现出来了, 而模型本身因为细粒度MoE(256选8)而产生的区分度也显著降低了, 同时论文还对比了Mistral的两个MoE模型, 它采用8选2的方式, 看来语义间对不同专家的区分度有很大的差距. 这个结论也支撑了DeepSeek逐渐向更加细粒度专家的技术路线的正确性, DeepSeek MoE相关的技术演进以前写过一篇

[《详细谈谈DeepSeek MoE相关的技术发展》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493157&idx=1&sn=51c0e27a347dd3fe1ed868d87f667897&scene=21#wechat_redirect)

当然产生这样的结果有几方面可能的因素:

Shared Expert的重要性, 通过Shared Expert消除了一些专家之间的影响, 使得Routed Expert Overlap的概率降低?

本质上就是Routed Expert数量的影响?

R1强化学习的工作流对于Expert Specialization进一步增强了?

但是值得注意的是另一个问题是, 在模型的后面20层内, 层间的Overlap的差异还是很大的, 并且没有进一步的下降, 这个和我拿到的线上的数据分布也是相似的.

这里引入一个思考, 每一层模型的AlltoAll通信时间实际上是受到分布式部署的带宽和延迟约束的, 因此模型深度过深后将会影响到TPOT, 虽然可以用一些ScaleUP的办法来解决, 但是看看GB200的可靠性和成本, 这种取舍是不太恰当的.另一方面, 看到上图中第40层overlap有明显的抖动, 一方面是模型在后面的层中还可以更加稀疏来进一步降低Overlap, 是否也会有一个类似的ScalingLaw我们在稍微后面的章节来分析.

### 2. SAE分析

这篇论文另一个亮点是基于Sparse Auto Encoder的特征来分析专家的路由模式. 关于SAE以前写过几篇分析

[《谈谈大模型可解释性》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490211&idx=1&sn=544e615e159226a1da662e5c658ca1f6&scene=21#wechat_redirect)

[《谈谈DeepMind会做算法导论的TransNAR并引出基于SAE-GNN的可组合Transformer猜想》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490297&idx=1&sn=7d758e84bdce7ae4f20f031f4ac3f221&scene=21#wechat_redirect)

从论文中SAE的分析来看, 能够得出不同的专家在负责不同的推理以及认知专业化的结论, 这和DeepSeek设计细粒度MoE和专家专业化的初衷是匹配的.

![图片](assets/70e7f192c3c2.png)

其实渣B一直在建议从SAE的角度来分析大模型, 并通过对SAE Activation的约束来作为强化学习工作流的一种手段,

SAE对于概念的可视化解释, Anthropic和OAI都做了相应的可视化展示, 例如Anthropic的多模态对金门大桥的概念

![图片](assets/dffe7020afdb.png)

OAI和Claude都在这方面有了蛮长时间的布局, 而国内相对还是落后了一些.

### 3. 从范畴论的视角看R1

这是一个烂尾很久的专题, 一直想抽一周的时间来好好分析并写一篇笔记, 但是最近几个月不停的在各种项目的死线上挣扎. 先简短的写一些吧. 其实R1的整个训练流程从范畴论的意义上来看:

首先是V3-Base的模型本质上是通过一系列数据集的Pre-train流程构成了一个预层范畴(Presheaf).

R1-Zero是基于V3-Base的Presheaf上来强化了一些Morphism的权重, 而这些权重在MoE模型的底子上使得模型具有了更强的泛化能力.

然后在V3-Base的基础上混合R1-Zero的Coldstart数据和一些General samples来构建最终的R1

![图片](assets/c4c781f26b05.png)

比较好奇的是在整个后训练的过程中, 不知道DeepSeek是否记录了梯度更新的情况, 感觉这个地方配合SAE做一些分析可能会有更多的发现, 个人觉得虽然ORM取得了很好的结果, 而PRM本身还有一些过程上的缺陷, 是否可以在SAE的视角上来看出更多的原因, 并且某种意义上还可以给ORM训练输出一些更加抽象泛化的约束能力.

当然这样也会面临一个比较大的算力的挑战, SAE的算力消耗和RL工作流的整体效率上的一个取舍问题.

### 4. MoE ScalingLaw

本文开头提到了一个比较有趣的说法:“Dense模型适合toB业务,MoE模型适合toC业务”, GPT4是MoE模型吧, 它适合toB还是toC? Llama3是一个Dense模型吧? 它适合toB还是toC? 本质的问题是算力的约束下MoE成为继续提高Scaling的一个必然手段. 当然MoE模型本身的Gating数值稳定性问题和Reasoning模型本身通常设置的温度参数相对较低, 使得模型的幻觉程度有所增加而不太适合一些toB的业务场景.

最近还有一篇《Chain-of-Experts: 释放MoE专家的沟通潜能》[2]挺有意思的, 即通过在同一层的专家之间的互相处理来得到最后的output hidden. 实际上这里又有了一些RNN的味道.  但是这样的机制如果迭代次数多了感觉很难去兼顾训练和推理的效率.

![图片](assets/9909fdfbbbd6.png)

从本文第一节的配图上来看, 似乎某种程度上能够得出和DeepSpeed-MoE[3]中提出的pyramid-MoE相似的结构, 随着模型的层数越来越深, 专家专业化程度越来越高, 相应的专家数量和TopK选择数量也需要对应的提高?

![图片](assets/22340234ac16.png)

其实这也是我最近在考虑的一个问题, MoE的本质是否和HNSW（Hierarchical Navigable Small Word）算法某种程度上有相似性?

![图片](assets/0a71be5bae66.png)

在下面这篇文章中也介绍了一些HNSW/CAGRA GPU加速处理的内容

[《英伟达GB200架构解析3: 从搜广推算法的视角来看待AI基础设施演进》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489194&idx=1&sn=7d3ef77de01d88400f1016e84e5abe91&scene=21#wechat_redirect)

那么借助Grace+Blackwell的架构, 是否还能做出点有趣的东西呢? 大概想到一个增量MoE的算法:

首先按照一个相对细粒度的模型进行训练, 例如256 Routed Experts, TopK=8

例如训练到500B tokens时, 模型逐渐添加一些新的专家在后面若干层

反复训练的过程中把模型逐渐迭代成一个金字塔结构.

最后在PostTraining过程中, 基于SAE或者某些层的MoE路由规则冻结一些Expert的参数或者是在这个基础上做一些KL散度的约束来降低幻觉?

为什么需要Grace呢, 因为某种程度上还是需要CPU侧的更大的内存空间来做一些专家权重的置换. PCIe本身的带宽还是太小了. 当然这样的模型部署时在推理阶段可能还有更多的挑战. 设计模型架构时兼顾推理性能是必须要考虑的一个因素了, 这部分内容暂时还没想明白, 隐约觉得在这样的一个模型下, 顺便把Next Few layer的Expert Prediction/Prefetch做了可能是一条路.

这一点非常认同飞刀老师的一个观点[《李飞飞：AI下半场打“系统战”，大模型预训练将走向寡头化》](https://mp.weixin.qq.com/s?__biz=Mzk2NDQ3MzYwOA==&mid=2247484215&idx=1&sn=667a48e1d8c736473bcaa26b3fa375c6&scene=21#wechat_redirect)

目前，阿里云正在GPU加CPU的异构资源池上做优化。未来，数据库要研发的关键能力是将昂贵的GPU尽可能地省下来做最珍贵的计算和缓存，将次要的计算和缓存推到CPU加内存和存储的三层池化中，让在线推理变得更低成本。

在基础设施和分布式系统的视角来看, 和模型的协同还有更多的工作要做.

参考资料

[1] 
Semantic Specialization in MoE Appears with Scale: A Study of DeepSeek-R1 Expert Specialization: *https://arxiv.org/pdf/2502.10928*
[2] 
Chain-of-Experts: 释放MoE专家的沟通潜能: *https://sandy-server-87f.notion.site/Chain-of-Experts-MoE-1ab9bb750b79801bbfebf01ae9a77b3f*
[3] 
DeepSpeed-MoE: *https://arxiv.org/pdf/2201.05596*