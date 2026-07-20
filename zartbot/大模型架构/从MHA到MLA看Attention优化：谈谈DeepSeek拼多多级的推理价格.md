# 从MHA到MLA看Attention优化：谈谈DeepSeek拼多多级的推理价格

> 作者: zartbot  
> 日期: 2024年5月10日 07:43  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489895&idx=1&sn=fa671523881b446be63c93e25a04899e&chksm=f99609a5cee180b39e18a8c010c1f60b2326be022872850736e55bf0cd2adf2823af371dd272#rd

---

### TL；DR

DeepSeek的工作还是非常elegant的，起初看MLA (Multi-head Latent Attention)觉得在Attention上做一些KV压缩有些担心，但仔细把Technical Report读了一下确实是一项很不错的工作，接下来稍微仔细的把它展开分析一下。

最近一直在思考一个问题，从TOPOS视角来看待多模态模型需要什么样的Attention？当把Attention看作是一个态射，预训练大模型构成一个预层范畴，而对于物理世界的几何空间和语言逻辑统一起来，有一本书值得参考《Sheaves in Geometry and Logic : A First Introduction to Topos Theory》，这是我一开始对在Attention上做压缩持怀疑态度的原因，当然TOPOS相关的内容会在后面[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)这个系列中阐述。

当然从范畴论的视角出发来看MLA是否能够同样的泛化支撑一些时间序列任务/AI4S任务/多模态视频类业务场景，还是持有谨慎怀疑的态度。但是正如在[《算力受限下的大模型发展和AI基础设施建设》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489813&idx=1&sn=d39b0334306d3b220eca935e9b694e84&chksm=f99609d7cee180c162633581494fa07c8dc92bd79df9f50508c7d7d28ddef3e002832c81a248&scene=21#wechat_redirect)总结的，工业界需要这样的算法/硬件/经营ROI上结合考虑的系统性的创新，MLA对于语言文本类模型的ROI提升是有很大价值的。

注：本文只讨论DeepSeek-V2的MLA部分，并重点关注推理成本相关的问题，而对于MoE会结合Snowflake的Dense-MoE再单独写一篇。

本文内容目录，个人偏好是采用自包含(Self-Contain)的方式尽量把整个脉络理清楚，但您也可以直接跳到第五节查看MLA的内容。

```
1. DeepSeek-V2 概述2. Attention概述2.1 Attention的由来2.2 Self-Attention2.3 Multihead Attention3. Attention的难题3.1 Attention的时间复杂度3.2 Attention的空间复杂度：KV Cache4. Attention的优化4.1 硬件视角的优化4.2 算法层面的优化<-可以跳这里看5. 从MHA到MLA5.1 代数的视角看MHA稀疏化5.2 MLA概述6. 总结
```

## 1. DeepSeek-V2 概述

DeepSeek-V2支持128K Context的MoE模型，参数量236B，其中每个Token活跃参数21B，训练数据高达8.1T Tokens但计算量仅为Llama 3 70B的1/5。最关键的一点是其API定价，大概只有GPT-4-Turbo的百分之一

![图片](assets/cf8630789a90.png)

低价的最关键原因是其模型结构上，针对Dense模型训练成本下降了42.5%推理KV Cache开销降低了93%，推理速度超过了50K Tokens/s相对于Dense模型提升了5倍。

![图片](assets/b9e62902367d.png)

DeepSeek-V2模型架构如下

![图片](assets/e713f69a128f.png)

## 2. Attention概述

### 2.1 Attention的由来

在《大模型时代的数学基础(4)》中谈论了Attention的由来，它来自于当今十分普及的双组件（two-component）的框架： 这个框架的出现可以追溯到19世纪90年代的威廉·詹姆斯， 他被认为是“美国心理学之父” 在这个框架中，受试者基于`非自主性提示(nonvolitional cue)`和`自主性提示(volitional cue)`有选择地引导注意力的焦点。自主性的与非自主性的注意力提示解释了人类的注意力的方式.

`非自主性提示(nonvolitional cue)`和`感官输入(Sensory inputs)`可以通过一个Key-Value Map机制构建,我们可以在内存中构建一个数据库 $\mathcal{D}$ 并存放相应的 $(k,v)$ 对，定义如下：

$$\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$$

注意力机制可以通过如下形式化的方法构建：

`自主性提示(volitional cue)`：我们将其定义为一个 $Query$ 张量

`非自主性提示(nonvolitional cue)`：我们将其定义为一个 $Key$ 张量

`感官输入(Sensory inputs)`：和Key有对应关系的 $Value$ 张量

![图片](assets/d58e7ed1a997.png)

注意力汇聚时在神经网络中其实就是一个关于 $Query$ 和 $Key$ 构成注意力分数，然后乘以 $Value$ 得出注意力机制的输出和数据库 $\mathcal{D}$ 的函数

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$$

更一般的来看是 $Query$ 和 $Key$ 构成注意力权重，然后乘以 $Value$ 得出注意力机制的输出

![图片](assets/8b0cce5a3da8.png)

$$\alpha_i = \alpha(q,k_i)$$

然后输出为

$$o = \Sigma_{i=1}^m softmax(\alpha_i)v_i$$

一个简单的 $Query$ 和 $Key$ 构成注意力权重函数为点乘(dot-product)

$$\alpha(q,k) = q \cdot k$$

然后考虑维度增加后的梯度影响，再对注意力权重进行一个缩放，即：

$$\alpha(q,k) =q \cdot k/{\sqrt{d_k}}$$

我们注意到：注意力机制是关于 $query$ 和 $key$ 在某个空间的投射，也有人把它弄的很复杂的去搞一些Hidden Layer，但是都没有前面的点乘机制那样简洁高效。

### 2.2 Self-Attention

而我们注意到Attention机制下，输入的Encoder信息会被处理成为一对对的 $(Key,Value)$ 对，并且存放在内存数据库 $\mathcal{D}$ 中

$$\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$$

即然是一个数据库，我们又希望模型可以基于相同的注意力机制学习到不同的行为，并且能够捕获序列内各种范围的依赖关系，那么允许注意力机制组合使用 $Query，Key，Value$ ，是否更好呢？这里就引出了`Self-Attention`机制

Self-Attention机制，通过对输入 $X_i$ 乘以三个可训练的参数权重矩阵 $W_q,W_k,W_v$ 进行线性变换为 $Query,Key,Value$ 向量.

![图片](assets/e29f6aeb7473.png)

然后同样使用AttentionScore的机制

![图片](assets/51bd0294e5ca.png)

通常了解一个词在一个句子中的含义，需要从多个方面来关注，例如日常用语中的指代消解对应的注意力机制需要一个输入对应多个Attnetion，鉴于这种情况出现了多头注意力机制

### 2.3 Multihead Attention

本质上是构造多个可表子空间,在这些子空间构建多个注意力来替代单个Attention

![图片](assets/ebe826944437.png)

代数结构上很简单

$$head_i = attention(W_q^iQ,W_k^iK,W_v^iV)$$

$$multihead(Q,K,V)=W_o concat(head_1,head_2,...,head_h)$$

## 3. Attention的难题

### 3.1 Attention的时间复杂度

工业界大家都在卷更多的参数和更长的Context Length,此时需要更大计算能力，如何构建一个计算复杂度线性增长的模型是我们急切需要寻找的网络架构，用来替代transformer架构。

![图片](assets/0195d4b5df8d.png)

如图所示，Context长度为 $N$ ，计算复杂度 $\mathcal O(N^2\cdot D_v)$ ，同时考虑全连接层计算复杂度 $\mathcal O(N^2\cdot D_v+N\cdot D_k^2 )$ ，我们期望于把复杂度降低到Sub-Quadratic。

### 3.2 Attention的空间复杂度：KV Cache

另一方面针对推理时的内存访问，我们还需要考虑算法的空间复杂度，通常在计算过程中会通过KV Cache的机制去降低重运算的开销，首先针对输入的一个Prompt序列，为每个Transformer层都生成Key和Value的Cache。后续Token推理过程中把前面Token的Key和Value直接从Cache中读取降低重复计算的复杂度。
![图片](assets/9f1658bdb0e3.png)

但这样一来又是一个计算时间和存储空间的折算问题， 这个问题本质上是在算力成本约束和带宽约束下以及Query延迟约束下的最优化问题。

![图片](assets/45a32b93011c.png)

## 4. Attention的优化

### 4.1 硬件视角的优化

从硬件视角来看，FlashAttention和SoftMax做online normalizer计算，尽量提高算子访问的Locality,另一个方法就是进行量化操作，例如量化到INT8来降低算力的需求。还有就是通过Page Attentiond的方式对实际显存碎片进行整理的方式，或者一些分布式的KV Cache方式，例如阿里Infinite-LLM等。还有就是针对推理过程中的Prefill和Decode阶段的算力和访存需求做切分处理。更详细的内容可以参考

[《大模型时代的数学基础(4)》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&chksm=f996046acee18d7c687403c557a6e30155ba0c04cca7e897de3126e88a5d3ba3a2c2fe0507bd&scene=21#wechat_redirect)

### 4.2 算法层面的优化

从算法上来看，并不是每个Token都需要那么多算力的

![图片](assets/9595bde11326.png)

因此算法层面的优化主要是一些稀疏Attention的尝试，张奇老师在《大规模语言模型：从理论到实践》这个图画的很直观

![图片](assets/e92fe199e7d4.png)

工业界也有一些尝试，例如Funnel-Transformer/Monarch Mixer/Tree-Transformer/BigBird这些工作，但是人为构造的稀疏结构和一些随机结构实际的算法收益并不明确

![图片](assets/29e0fc522330.png)

另一方面Monarch这样的Block Diagonal矩阵算法属于带置换的局部注意力机制，其收益也不太好

## 5. 从MHA到MLA

### 5.1 代数的视角看MHA稀疏化

从代数结构上来看，大模型预训练是一个预层范畴，确定性的Attention稀疏性破坏了态射结构，另一方面对于Word2Vec从表示论(Representation Theory)的视角来看，对于一个有限群 $G$ 的有限维表示 $V$ ,由Maschke定理，如果 $V$ 有非平凡的子表示 $W$ ,则有表示的直和分解 $V=W \oplus U$ ,由于 $V$ 是有限维，则存在如下分解

$$V = V_1 \oplus V_2 \oplus ...\oplus V_k$$

其中每个子表示 $V_i$ 都没有非平凡子表示。同时舒尔引理是群与代数的表示论中一个初等但非常有用的命题，对于一个有限群表示可以通过直和分解成多个不可约表示，那么我们就可以将每个不可约表示定义为一个Block，通过Schur引理，使得模型构造出这样的表示？在MLP层处理就是MoE，而在Attention Layer那么就是Mixture-Of-Depth。

另一个角度来看一个用的更广泛的推理优化就是GQA,GQA采用相对密集的Shared KV方式来处理，等同于对Attention空间的均匀采样

![图片](assets/ed4314b0f44a.png)

### 5.2 MLA概述

简单的说就是通过在训练期间构建一个压缩空间，然后在推理阶段针对Decode的Memory Bound问题，通过Up-projection的运算开销来降低访存开销，而Up-Projection恢复KV的运算在计算Attention Score的时候又可以和其它参数矩阵结合。

在MLA中，对于KV采用了一个共享的参数矩阵 $W^{DKV}$ 做down-projection投射到一个低维空间再由独立的参数矩阵做up-projection

![图片](assets/acbd38369fc6.png)

对于MHA

$$
\left\{
\begin{aligned}
q_t & = & W^Qh_t \\
k_t & = & W^Kh_t \\
v_t & = & W^Vh_t
\end{aligned}
\right.
$$

而对于MLA

$$
\left\{
\begin{aligned}
q_t & = & W^{UQ} & \cdot W^{DQ}h_t \\
k_t & = & W^{UK} & \cdot W^{DKV}h_t \\
v_t & = & W^{UV} & \cdot W^{DKV}h_t
\end{aligned}
\right.
$$

其中针对KV $d_h=128$ , $n_{head}=128$ ，压缩的维度为 $d_c = 4d_h= 512$ ,针对Q也是位了节省训练阶段的activation Memory消耗采用了压缩 $d_c'= 1536$ 

对于推理来说，Prefill阶段本来就是算力密集型的，而Decode阶段是内存密集型的，MLA实际上是通过压缩的Latent_KV Cache再进行了一次 $W^{UK},W^{UV}$ 的乘法运算，通过部分的UP-Projection算力消耗(压缩维度 $d_c \ll d_hn_h$ )在Decode阶段降低了内存占用.

![图片](assets/f3d9b4a1b58b.png)

需要注意的是：从 $C_t^{KV}$ 恢复KV需要计算Up-projection矩阵 $W^{UK},W^{UV}$ ， 在计算Attention Score的时候可以和 $W^{UQ},W^O$ 结合，进行一次离线运算即可，实际上并不需要独立的Up-projection运算开销，这样就很巧妙的在不增加算力消耗，同时又降低了KV Cache的开销。

当然有一个问题是ROPE和Low-Rank压缩不兼容，所以采用了独立的一路来做ROPE然后CONCAT构成Q K矩阵。

### 5.3 从代数的角度看MLA

从Embedding的角度看，选择更大的词表其表现力会更强，因此DeepSeek也选择了100K词表，可以看作是一个 $d_h$ 可表空间内对象数量增加，Attention作为词间态射在大规模预训练数据集更多，而这些词间的部分态射可以通过其它词间态射复合的方式构造，那么也就是说提供了一种可能性，可以构建积范畴的形式

![图片](assets/9d84da0dd964.png)

$Y \rightarrow X_1 \times X_2$ 的态射可以由 $f_1,f_2,\pi_1,\pi_2$ 表示，Embedding构成的范畴可以由若干个范畴的积的形式表示。以表示论(Representation Theory)的视角来看，我们可以构造一系列子表示的直和形式来构建。

再以统计学的视角，类似于主成分分析的方式对 $d_h$ 降维投影到低维空间是可行的。另一方面从生物/心理学的角度来看，KV压缩到Latent Space可以看作是对`非自主性提示(nonvolitional cue)`和`感官输入(Sensory inputs)`的模糊化处理也是符合自然规律的一种做法。

因此对于Attention通过Latent Space降低维度是可行的，但是需要注意的是该空间的构建需要在训练阶段，它会直接影响Embedding矩阵并影响词表嵌入的分布，同时KV共享相同的down-projection矩阵使得KV构建在一个相同的基上也是有必要的。

## 5. 总结

对于MLP层采用MoE基本上已经成为共识的演进方向，对Attention的优化还在进行中，一方面是一些Non-Attention的替代，例如Mamba这类结构化状态空间模型（Structured State Space Models, SSMs)尝试去降低复杂度。或者是MoD这样的模型Early-exit跳过部分Attention计算。

![图片](assets/d03f0f4f0284.png)

MLA的工作则是从系统化的角度来考虑，从算法上来看，采用了大规模的词表后，其Embedding空间降维表示是可行的，另一方面是从系统架构来看，这样的降维方式由于KV重算的up-projection的两个权重矩阵可以和attention score的其它权重公式结合，实际上降低了KV Cache开销但又没有增加算力开销。

从模型效果而言，潜在的问题就是当前推理该模型可能还是需要一些大容量显存的卡，但吞吐可以覆盖掉卡的成本，因为空间本身就压缩了以后，再进一步做一些INT8一类的量化可能导致模型的输出质量下降，这一方面我还是有一些担忧的，还需要后续继续观察一下。对于当前模型的一些打榜分数还是有一些怀疑的，可能还需要更多的时间去验证效果，特别是针对一些toB的严肃的场景来看，可能模型的能力上还是会有一定的问题的。

当然把推理成本降低了这么多以后，搞个LLM-PDD从商业模式上来看还是值得尝试一下的，例如接入到“沉浸式翻译”这些场景，还是很有价值的。