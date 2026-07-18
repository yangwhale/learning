# 猜一猜DeepSeek-V4的模型架构

> 作者: zartbot  
> 日期: 2026年1月18日 05:34  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497388&idx=1&sn=7fb21bc482cfc02f7515fc0689fec56b&chksm=f995e66ecee26f78a9a9d678496ab4958d273fe7748cd21e7d1a905e1213a0638c9dab80606c#rd

---

### TL;DR

最近很多人都在传DeepSeek会在春节前发布新一代的模型. 昨天在飞机上仔细想了一下, 因此准备结合这几年整个DeepSeek的研究路线, 做一个猜测. 首先我们在第一章回顾一下DeepSeek整个研发路径, 从哪些地方可以Scale谈起, 然后再进一步第二章来进行一个推测, 可能一些Attention的结构大家都能猜到DSA + mHC + Engram, 但是我想尝试着去理解背后的理论和Know-How

## 1. 稀疏是整个主旋律

其实从DeepSeek诞生到现在, 我们可以看到针对模型结构的一个很清晰的脉络, 针对不同Block的稀疏处理. 背后的逻辑其实非常简单, **因为内存墙的存在, 计算是可以相对容易的Scale的, 而访存则是很难Scale的**. 我们先来回顾一下整个脉络.

### 1.1 MoE

从MoE开始进行FineGrained Expert处理, 详细的演进可以参考[《详细谈谈DeepSeek MoE相关的技术发展》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493182&idx=2&sn=7a6017161753ae1f984bc85e98d00987&scene=21#wechat_redirect)

DeepSeek-V1 MoE引入Fine-Grained Expert, 并且放置了一个独立的Shared Expert, 并利用辅助损失函数引入了专家的负载均衡和设备的负载均衡,  V1有64个路由专家和2个共享专家, topk=6.

DeepSeek-V2 MoE进一步将路由专家总数目增加到了160个, 共享专家还是2个, topk=6. 相对于V1添加了Devie-Limit Routing和通信负载均衡的辅助损失, 并且添加了一些Token丢弃策略.

DeepSeek-V3 MoE路由专家数量扩展到了256, 共享专家1个, topk=8. 引入了专家分组和无需辅助损失函数的负载均衡, 取消了Device-Limit Routing和Token丢弃的策略. 同时开发了DeepEP的新的通信范式, 并配合Redundancy Expert和EPLB完成负载均衡.

就此, 关于FFN的稀疏化MoE处理告了一个段落.

### 1.2 Attention

那么下一个目标自然就是Attention的稀疏化了. Attention的计算复杂度为 . 而现代大模型推理任务中对Context的长度要求越来越高, 因此首先就会考虑构造Sparse Attention. 这一点不光是计算复杂度的优化, 因为即使模型对某个选项有较高的置信度(logit值较高), 在经过Softmax归一化后, 其最终的概率值也会被"稀释", 导致整个概率分布看起来更平坦.

在[《谈谈DeepSeek Native Sparse Attention》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493234&idx=1&sn=cdca1661864f5ebf21c37e26fc51be10&scene=21#wechat_redirect)中对论文有些分析, 其中的想法也是很朴素的. 就是能否构建一个函数把KV映射到一个低维空间?

其实更早的MLA便是通过压缩KV的方式进行处理, 降低KV访存带宽. 具体可以参考[《从MHA到MLA看Attention优化：谈谈DeepSeek拼多多级的推理价格》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489895&idx=1&sn=fa671523881b446be63c93e25a04899e&scene=21#wechat_redirect),然后在NSA中更进一步.

在此之前其实也有很多Sparse Attention的方案了, 可以参考[《大模型时代的数学基础(4)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&scene=21#wechat_redirect)中的一些介绍, 有些用Hash, 有些用随机, 还有Pooling的, 各种编码方式总有点老的机器学习中的特征工程的味道. DeepSeek在NSA论文中吐槽了一下, 要么是一些不可训练的组件, 要么就是BP阶段效率很低.

而DeepSeek的NSA则非常简单直接

![图片](assets/94d1e10aa3b4.png)

压缩(Compression)的部分尽量获得全部输入的一个摘要

滑动(Sliding)窗口获得最近的context关注的注意力焦点

很精彩的根据top-n选择(Selection)把压缩时导致的很多重要的细粒度信息的损失给补了回来

但是在后期又出现了DSA, 并发布了DeepSeek-V3.2. 关于DSA的介绍在[《学习一下DeepSeek-V3.2》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496212&idx=1&sn=3ff9767a1b93ed8a495d2be614146f2d&scene=21#wechat_redirect). 相对于NSA, DSA做了很多简化, block selection 变成了token wise的selection, 并且用一个小规模的indexer计算top-2048个token. 然后通过继续预训练的方式, 在Full MLA的基础上先预热Indexer, 然后利用KL散度来保持将Indexer输出与主注意力分布对齐.

其实这也是一个处理Sparse Attention的一种常见手段, 就像我在去年愚人节开的一个玩笑一样[《谈谈一个新的MoA模型架构DeepSick-4.1T》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493736&idx=1&sn=8f31324965270f562ca5065fc5c799a8&scene=21#wechat_redirect), 通过self-KL散度来衡量Sparse Attention的效果去尽力对齐Full Attention的分布.

当然这里工业界其实有一些分歧了, 例如Qwen和Kimi选择了Linear Attention. 具体的一些讨论在[《谈谈未来Attention算法的选择, Full, Sparse or Linear ?》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496753&idx=1&sn=b66ffd8d2e977cb4e7e27603ea9a9951&scene=21#wechat_redirect).

## 2. 预测一下DeepSeek-V4的模型架构

其实最近的mHC和Engram的论文已经可以看到一些影子了. 在此之前, 我们假装没读过这些论文, 我还是想尝试着猜测一下背后的逻辑. 进一步Scale如何处理? 堆积更多的参数到MoE? 当然在Engram论文之前其实还有不少这样的观点, 例如更多的Expert数量. 实际上Expert数量更多以后, 训练上会出现更多的dead expert, 通信上的压力也会更大. 因此MoE的Scale其实后面会比较难做出收益了, 然后目标自然盯着了Attention. 另外感觉DeepSeek也有一种tick-tock的处理, 引入V1的MoE后, V2对Attention进行了修改构建了MLA, V3进一步修改了MoE. 似乎到了V4又该修改Attention了.

预计整个Attention的结构会基于 DSA + mHC + Engram 构建.

Sparse Attention某种意义上解决了计算复杂度为 中  相关的问题. 但是我们有一个很自然而然的想法, 如何在不增加计算复杂度的情况下去scale? 例如扩大 的维度, 因此一个直觉的想法是, 构建一个旁路的网络结构, 升维在一个高维度的空间进行一个稀疏处理再降低维度. 但是并不扩展 attention自身的维度. 这个想法其实在2024年读DeepMind TransNAR的论文就有, 想到的[《谈谈DeepMind会做算法导论的TransNAR并引出基于SAE-GNN的可组合Transformer猜想》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490297&idx=1&sn=7d758e84bdce7ae4f20f031f4ac3f221&scene=21#wechat_redirect)

正如这篇文章最后谈到的一点, 我们期望有一个旁路的Sparse运算甚至可以Offload到CPU, 然后跨越多层旁路注入的方式. 结合起来便是mHC + Engram. 其实从微观的角度来看, N-gram本身也是一个稀疏的图.

其实Seed最早做出了一版HC, 同时也有Over-Encoding, 但是在工程落地的细节上有很多问题并没有处理的很好, 特别是训练稳定性和效率的问题.

### 2.1 谈谈mHC

mHC的一些详细分析可以参考[《谈谈DeepSeek mHC》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497138&idx=1&sn=8215a15d8e196d412ab908ec3302c857&scene=21#wechat_redirect),其实mHC中的Manifolds Constraint这词有一点over claim了, 本质上我可能更愿意称其为双随机矩阵约束HC(DS-HC?), 但是我想DeepSeek可能的描述是想说Manifolds Constrains的一些方法或者范式可以用在其它地方, 例如考虑到一个由 个参数所有这些分布构成了一个  维的几何空间构成一个“统计流形”, 对于流形上的两个点 和 在这个统计流形上的测地线距离, 曲率或者是Fisher信息度量相关的约束可能会应用到其它地方, 例如在RL Post-training阶段做一些Off-policy的处理上提升整个后训练的速度, 这个话题后面会单独写一篇文章来分析. 在去年8月写过一篇文章[《大模型时代的数学基础(9)- SDPA和最优传输, 强化学习及信息几何的联系》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494688&idx=1&sn=3d589f6d4be56ee372d5db4f8631b0cc&scene=21#wechat_redirect)

回到mHC上, 基于SK迭代算法的处理是很巧妙但也是很难想到的, 有一些做图形相关的模型的同学可能会对SK算法比较熟悉一点, 在LLM这个领域来看, 除了很早期的Sink Attention以及去年看到了一篇《Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport》. 其实真的比较难有人会想到从Birkhoff多面体的性质来这么优雅的处理HC的训练稳定性的问题.

### 2.2 谈谈Engram

详细的介绍在[《谈谈DeepSeek Engram: Conditional Memory》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497337&idx=1&sn=1f36137c326db6103698139d896cb1cb&scene=21#wechat_redirect)

其实一个最本质的问题是早期DeepMind的 Mixture-of-Depths和Mixture-of-Recursions这类的工作, 他们发现. 主要目的是解决标准 Transformer 模型在计算资源分配上的低效问题. 传统 Transformer 对序列中的每个 token 都施加相同的计算量, 而 MoD 则让模型学会动态地、有选择地将计算资源分配给序列中最重要的 token.

但是Recuisive的Attention虽然在模型参数没变的情况下等效的增加了模型的深度, 但是Engram对于一些计算量需求偏低的Token任务做了更巧妙的处理, 通过Ngram嵌入到上下文中, 更轻量级的增加了复杂token的Attention运算budget.

另一方面Engram做的非常巧妙的一点是, MultiHead的Hash并且仅有复杂度, offload到CPU的处理对推理也非常友好. 另一方面是Ngram嵌入到原来的上下文使用的门控巧妙的利用了Attent机制:

![图片](assets/4bf99aaf8fb4.png)

当然现在的Engram还是一个比较静态的Memory, 不知道未来是否会演进到类似于Google提出的Test Time Learning相关的方式:

![图片](assets/0a4da2e2b471.png)

## 3. 对DeepSeek-V4模型做一个整体的预测

### 3.1 Attention

Attention block从当前公开的资料来看, 基于 DSA + mHC + Engram 构建基本上算是明牌了.

### 3.2 MoE

MoE的结构估计这一代应该没有太大的变化, 更多的是从Engram的视角去扩展Scale.

### 3.3 模型的参数

从Engram的论文来看, 假设总参数能够适配一些小规模的8卡服务器(例如H20/H200)一类的平台, 那么Attention + MoE的参数量应该可以Scale到1.5T左右.一方面是GPU显存容量的约束, 另一方面是算力的约束, 按照激活参数量估计会到70B左右, 这个范围也是一个可以跑的比较快的模型.  然后就是Engram的参数规模, 考虑到CPU的内存容量, 规模应该可以Scale到1T左右. 也就是说累计模型参数规模大约可以做到2T~2.5T.

大致的计算如下, 例如模型的Hidden_size是否会进一步扩展到12K到16K, 因为有DSA的处理, 其实算力的消耗对于dim的增加是可控的. 然后配合mHC n=4应该会带来性能很大的提升.  

而对于模型的层数应该会在还是在60层左右, 太长可能会影响到TPOT.

然后另一方面, 其实在DeepSeek-V3推出的时候, 我就想是否能够推出一个相对小的模型. 例如类似于GPT OSS这样的主干模型在120B附近, 然后Engram也差不多100B的规模? 整体模型规模大概在200B~300B左右, 这样对一些国产卡推理或者说NV的一些相对合规的卡或者小规模的单机~4机部署很友好的模型. 其实这个规模的模型对于很多ToC的业务已经够了, 也可以承载大量的Serving.

其实真的挺期待DS能发一些小尺寸模型的.

### 3.4 关于RL后训练

去年来看, RL相关的进展DeepSeek公开的不多, 主要是GRM的工作

![图片](assets/0b5bbd6ebf1e.png)

针对通用的Reward Model设计上采用了模型自己生成Principal的多维度打分评价体系,并根据这些生成的原则来产生评论, 并基于评论给出最终的Point-Wise打分. 通过这样的Reward Model设计非常巧妙的提高了模型本身的泛化能力.

另一方面是今年年初刷新了一下DeepSeek-R1的论文, 披露了一些当时训练的时候就训推对齐的R3(Rollout Routing Replay).

其实我蛮期待DeepSeek能够在Manifolds Constraint的视角上去处理一些RL相关的问题. 特别来说现在很大程度上为了训练的稳定性和效果还是On-Policy居多, 另一方面昨天听到一个翁家翌的podcast, 其中无意中谈到了大概的迭代时间为几百秒到几千秒.  如果我们不追求完全的训推对齐, 而是采用一些流形约束的办法构建高质量的Off-Policy应该是一个非常promising的路径.