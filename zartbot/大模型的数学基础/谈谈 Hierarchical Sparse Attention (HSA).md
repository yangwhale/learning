# 谈谈 Hierarchical Sparse Attention (HSA)

> 作者: zartbot  
> 日期: 2025年10月6日 08:24  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496275&idx=1&sn=5f8a8d8efff22033d3f2aed8a5844e53&chksm=f995e291cee26b874972f43beda008f3ae486327d4d47fae76fa5adfb273518f39ca0519dc26#rd

---

### TL;DR

昨天下午跑步的时候有一个脑洞, 想到一个很有趣的Sparse Attention的算法, 在此记录一下...

大概整个思考的过程如下:

工业界在DSA之前其实也有很多Sparse Attention的尝试, 在[《大模型时代的数学基础(4)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&scene=21#wechat_redirect)中的2.2节有很多记载.

基于[《大模型时代的数学基础(2)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488528&idx=1&sn=fa49e334201e738e7ddb4258030798b3&scene=21#wechat_redirect)中范畴论的视角, 将Attention作为一个预层范畴来看待, 由Yoneda Embedding, 任何局部小范畴中的对象都可被对应的预层范畴中的元素表示.

前段时间写[《CuTe Layout代数-1: Overview》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496154&idx=1&sn=474a5450c46b86169095d84dd3cfd7dc&scene=21#wechat_redirect)的时候, 谈到过范畴论中的Nerve构造..

然后就是读[《学习一下DeepSeek-V3.2》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496212&idx=1&sn=3ff9767a1b93ed8a495d2be614146f2d&scene=21#wechat_redirect)的时候, DSA实际上是用一个低维度的Dense Attn(但是使用ReLu)做一个草稿构成indexer, 再选择其中K=2048个构建稀疏Attn

前段时间看[《谈谈Transformer的一些演进: UT,MoD,MoR...》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494744&idx=1&sn=20f307c5e0fe7c5c5d62a46d81f48646&scene=21#wechat_redirect), 在Attn block做一些recursive的事情是否可行?

把这些东西串起来似乎就构成一个有趣的Hierarchical Sparse Attention算法了.

### 1. 从Dense Attention谈起

在以前分析一篇论文的文章中[《大模型时代的数学基础(9)- SDPA和最优传输, 强化学习及信息几何的联系》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494688&idx=1&sn=3d589f6d4be56ee372d5db4f8631b0cc&scene=21#wechat_redirect), Scaled-Dot-Product Attention(SDPA)注意力机制的前向计算过程, 完全等价于一个单边的熵最优输运(One-Sided Entropic Optimal Transport, EOT)问题的精确解.

伴随着Coding/Agent任务context长度越来越长, 对于一个Dense Attention运算复杂度为, 同时softmax对于SFU的开销也是很大的. 在B200中SFU的算力相对于H200并没有太大的提升, 直到B300才通过砍其它高精度的算力把它提升上去.

另一方面来自于内存墙的瓶颈越来越大, 从B200开始这样的多Die封装的结构, 在跨Die内存访问时延迟会明显增加, 虽然纸面算力看上去提升了, 但是实际来看当前的很多算法上并没有获得这些算力提升的收益, 很多workload甚至比H200还要差一些.  **实质的问题就是计算本身是很容易ScaleOut的, 而内存访问是很难的.** 特别是针对一些任务需要更好的Token per seconds如何处理?

正是这两个约束下, 我们需要引入稀疏注意力机制, 但是另一方面很大程度上还是需要以来SDPA的softmax进行attention scores的计算.

### 2. DSA

DSA是在MLA的基础上训练的一个Sparse Attention的实现, 引入了Lightning Indexer和Top-K selector实现了稀疏的Token处理.

![图片](assets/079623d983dc.png)

**绿色路径 (DSA部分)**: 生成轻量级的和, 进入Indexer, 经过Top-k选择器, 最终输出一个"地址列表", 告诉主注意力应该去关注哪些历史token.

**主路径 (MLA部分)**: 生成重量级的Query Head和当前的KV .

**融合**: 主路径的查询头, 根据绿色路径提供的"地址列表", 从历史键值对中精确地取出被选中的, 并与当前的一起进行MQA计算.

在DSA中. lightning indexer,计算 Query token  与其之前的token  之间的索引得分 , 这个得分决定了哪些token将被该查询token选中:

其中, 表示Indexer的头的数量;  和  是从query token  派生出来的; 而  是从前序token  派生出来的. 选择ReLU作为激活函数是出于对吞吐量的考虑. 鉴于Inexer的头数量很少, 并且可以用FP8格式实现, 它的计算效率非常显著.

实际上可以把Indexer当作一个通过基于ReLU的scratch dense attention, 然后利用Top-k=2048进行了一个过滤. 但实际上对于某些Token可能2048不够, 而对于一些token可能少数几十个就够了. 那么我们如何构建一个Dynamic Top-K机制来进一步降低内存访问呢?

### 3. 范畴论的视角

米田引理 (Yoneda Lemma) 是范畴论的基石. 它的核心思想是: **一个对象完全由它与范畴中所有其他对象的关系 (即"箭头"或"态射") 所决定.** 换句话说, 你不需要知道一个对象的"内部构造", 只需要知道它如何与外界"交互"(即所有射向其他对象的箭头集合), 就能完全理解它.

因此我们对于一个Token可以看作由它的Attention所决定的. 于是token和attention某种意义上可以构成一个范畴. 这就进一步引出了范畴论中的Nerve[1](神经)构造.

范畴论中, 一个小范畴  的神经(nerve)** 是一个由C的对象和态射构造出的单纯集这个单纯集的几何实现是一个拓扑空间称为范畴C$ 的分类空间. 这些紧密相关的对象可以利用代数拓扑学(最常用的是同伦论)的方法, 为一些我们熟悉且有用的范畴提供信息.

范畴的神经常被用来构造模空间(Moduli spaces)的拓扑版本. 如果  是  的一个对象, 它的模空间应该以某种方式编码所有同构于  的对象, 并记录这些对象之间的各种同构关系. 这可能会变得相当复杂, 特别是当对象有很多非恒等自同构时. 神经提供了一种组合的方式来组织这些数据. 由于单纯集有很好的同伦论, 我们可以提出关于不同同伦群  意义的问题. 我们希望这些问题的答案能为原始范畴  或相关范畴提供有趣的信息.

模空间(Moduli spaces)
模空间 (Moduli space) 是代数几何中的一个核心概念. 它是一个几何空间, 其上的每一点都对应着一个特定类型的代数几何对象的同构类. 模空间的研究旨在为这些几何对象的分类问题提供一个统一的框架, 并通过研究模空间本身的几何和拓扑性质, 来反过来理解这些几何对象的性质.模空间的核心思想是**“为对象建立一个几何目录”**

例如语言中的同义词, 相同含义的内容通过不同国家语言的描述等, 我们都可以看作它们为背后蕴藏的知识本身的同构类. 当对象之间存在复杂的等价关系(同构)时, 模空间本身就成了一个复杂的"空间". 神经构造(Nerve Construction)正是处理这种复杂"等价关系网"的系统性工具.

例如, 对于一个简单的三个Token之间的Attention(态射)如下:

![图片](assets/6d75e1584334.png)

 到  的attention为 ,  到  的attention为 . 而  到  的attention可以看作  的复合 .  对于由  个Token和它的Attention构建的一张态射图上来看, 我们能否进行某种神经构造,避免任意两个token之间的attn运算来构造?

令  为一个小范畴.  的每个**0-单纯形**对应  中的一个对象. 的每个**1-单纯形**对应  中的一个态射 , 即任意一个token 构成一个0-单纯形, 它们之间的Dense Attn作为态射, 构成一个1-单纯形. 如上所示三个token , 现在假设有可复合的态射  和 . 那么我们有它们的复合 . 这个图暗示了我们的做法: 为这个交换三角添加一个**2-单纯形**.  的每个2-单纯形都来自于这样一对可复合的态射.

**一般情况:** (k-单纯形的集合) 由  中 个可复合态射组成的序列构成:

为了完成  作为单纯集的定义, 我们还必须指定面映射(face maps)和退化映射(degeneracy maps).

**面映射** 通过以下方式给出:

当  时, 通过在第  个对象上进行态射复合.

当  或  时, 移除序列的首尾对象(和相应的态射).

 将 k-元组

发送到 (k-1)-元组

**退化映射** 通过在对象  处插入一个单位态射(identity morphism)来给出.

**函子式定义:** 单纯集也可以被看作从  到  (集合范畴)的函子, 其中  是由全序有限集  和保序映射构成的范畴. 我们可以将范畴  的神经描述为函子 :

这个描述使得函子性变得透明. 例如, 小范畴  和  之间的一个函子  会诱导出单纯集的一个映射 . 此外, 两个函子之间的自然变换会诱导出映射之间的同伦.

这部分的描述让我们很容易的想到一个类似的结构, 即Hierarchical Navigable Small World graphs, HNSW.

![图片](assets/27d7635f36cf.png)

**问题**: 我们能否构造一个类似的代数结构, 来对Sparse Attention进行处理? 构造一个层次化的结构, 同时又能够满足高并行计算的能力.

### 4. Hierarchical Sparse Attention(HSA)

另一部分的想法来自于Universal Transformer.将输入序列 (长度为m, 维度为) 的初始词嵌入矩阵通过步迭代, 得到成最终的表征矩阵  .

在每个递归步骤 , 对所有 个位置的表征 进行并行更新, 得到. 其更新公式如下:

是上一步的表征. 同时加入了 , 即位置和时间步编码 (Position and Time-step Embeddings).

类似DSA中的Indexer进行计算, 然后首先取T , 是一个递归的超参数, 可以取 [4, 8, 16...]等值. 然后对取出的K个token通过第二个indexer计算, 第二个indexer可以适当的提高计算的维度或数值精度. 再取  个token. 迭代轮后(例如T=4), 最后按某个阈值进行过滤, 得到每个Token需要计算Attention的index-map. 然后根据这个index-map构建sparse attention.

特别的说, 在这种构造下我们获取的index-map不再是按照topk选择, 而是多次迭代后的结果叠加获得的, 当然在这里面也可以参考NSA, 按照block或者sliding window 构建混合的indexer叠加过程, 然后按照阈值进行过滤, 因此对于每个token选择的其它token计算attention的数量不会是一个固定的超参数Topk=2048, 而是一个动态的选择过程. 最终使得整个Attn计算变得更加稀疏一些?

当然这样的计算也会带来更大的算力开销, 为了平衡这些开销. 是否通过这样的方式模型层数可以减少达到同样的效果, 例如从61层降低为41层? 然后对于每层的MoE增加专家数和激活专家数量? 因为Attn计算的时间变长某种意义上来说也可以有更多的budget去overlap MoE. 但整体上通过降低层数又提升了推理速度?

反正就是一个脑洞,  最后大概就这样吧....

参考资料

[1] 
Nerve in Category Theory: *https://en.wikipedia.org/wiki/Nerve_(category_theory)*