# Pretrain ScalingLaw真的终结了么?

> 作者: zartbot  
> 日期: 2025年1月5日 10:29  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493002&idx=1&sn=736a3bd3e03c8a34a9831d1d3324cd1e&chksm=f995f548cee27c5e7edc6ab33bf51b085f3c53298eb5c7248c644905ca55437fef293ae96be0#rd

---

ilya在neurips 2024关于Pre-training的演讲被广泛传播

![图片](assets/1ad5bd14b8dd.png)

但是Pretrain ScalingLaw真的终结了么? 诚然算力在增长, Data并没有. 既然data作为AI的fossil fuel,如果把大模型比成一个油车, 合成数据 (Synthetic data)似乎就有点生物燃料的味道, 但是这条路上的GPT5尚处在难产阶段...那么大模型的“新能源车”在哪? 或许身在圈中的人并不愿意提及这个话题, 这意味着基于Transformer的“油车大模型”路径的终结, 而新的“电车大模型”的框架似乎还有大量的问题悬而未决...

渣B前年在[《大模型时代的数学基础(4)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&scene=21#wechat_redirect)中介绍了一些路径, 例如RWKV/Mamba/MoE等.. 例如Mamba相对于transformer的一些改进:

![图片](assets/23d145e4f45e.png)

似乎在这个基础上, 缝合的“混动的大模型”产生了, 不得不提一个漫画~

![图片](assets/98d0c5473e8e.png)

那么是否还有新的算法出现呢?例如能够超越Scalinglaw?

![图片](assets/bdcac94b4f18.png)

然而工业界, 特别是几个模型大厂在商业化的压力下似乎根本不愿意冒险, 而另一方面学术界又因为算力资源的限制,很大程度上完全排除在了这场变革之外. 而对于DeepSeek-v3的赞扬除了在Infra上非常细致的工作外, 另一方面就是非常敬佩他们是为数不多真正在Deep Seek的公司, 例如在Attention上做出了MLA, 又在MoE上搞出了Shared-MoE和Auxiliary-Loss-Free Load Balancing Strategy.

当然实话实说, DSv3应该也会有一些缺陷, 对于1shared + 8-of-256的MoE, 可能对于math/code这样的应用是非常友好的, 但是对于一些文学/艺术创作类的应用, 我不确定是否有影响, 所以我们也可以看到DSv3在模型的前三层还是MLP的, 这样的组合其实也蛮巧妙的.

另外最近看到几个比较有趣的工作, 主要是关注在更基础的层面上

### 1. Softmax

第一个是来自于Google Deepmind的《softmax is not enough (for sharp out-of-distribution)》[1]

![图片](assets/f6e9ffadc1c5.png)

这个结论挺有意思的,也就表明softmax从根本上无法在所有可能的输入上维持稳健的推理行为.

![图片](assets/20a89a4e0d7e.png)

然后通过自适应温度的方法来缓解的做法看上去还可以?

![图片](assets/816209c942d6.png)

### 2. Normalized Layer

另一个工作是NV的《nGPT: Normalized Transformer with Representation Learning on the Hypersphere》[2]

通过 SLERP（球面线性插值）将参数约束在一个超球面上, 当然SLERP计算复杂度高用了简单的线性插值近似.

![图片](assets/0be6c59dc99d.png)

通过这样的方法使得模型收敛更快

![图片](assets/435788879129.png)

### 3. Optimizer

另一个值得分析的是当前的AdamW是否还有进一步优化的空间? 最近看到一个Optimizer(https://github.com/KellerJordan/modded-nanogpt, https://github.com/nikhilvyas/modded-nanogpt-SOAP)

![图片](assets/66440e38a643.png)

Half the memory usage of Adam

1.36x faster training

<3% wallclock overhead

在同样的Loss下仅需要更少的Token, 同时也大幅度降低了内存的使用.

![图片](assets/3ecff105289d.png)

另外最近不少朋友都在问渣B为什么不去某个大模型的厂家, 首先从个人经历上来看, 特别是在互联网泡沫时期的经历, 虽然也是1998年就开始做网站当站长, 但是后面2000~2010年的互联网发展的路径来看, 软件/算法以及商业模式上是不成熟的, 而更快速增长的是基础设施的建设. 因此最近几年的重心还是更多的放在基础设施的建设上, 特别是在一些分布式系统的效率和容错上, 更多的在底层算力设施上解决一些更难的问题, 例如MoE的alltoall通信等,同时也在关注算法和Infra的结合, 更多的在infra上提供更好的硬件来支撑模型的训练和推理.

另一方面是在算法上, 从当前的模型架构上说服不了自己, 现阶段的尝试感觉大量的还是属于改良“油车大模型”, 新的“电车大模型”可能还需要更多的数学上的算法提供支撑, 这些理论上的研究还在继续做. 例如图神经网络和代数几何/代数拓扑相关的方向, 最近看到一篇非常有趣的论文Grothendieck Graph Neural Networks Framework: An Algebraic Platform for Crafting Topology-Aware GNNs[3] 后面工作稍微轻松一点了再来解读吧.

最后, 来个暴论, 简单的来看Pretrain scalinglaw的终结以数据不够来搪塞是不太恰当的, 在一些很细节的地方, 例如Transformer Attention block的修改, MLP/MoE的一些发展, 甚至是Softmax函数, LayerNorm和一些Optimizer, 以及参数的约束上还有蛮多的工作可以去做的. 但是这些工作似乎在math intuitive上离AGI还有很长的一段路要走, 例如在Transformer构建的Attention结构之上是否存在一些高阶的范畴? 基于差分的transformer 例如最简单的一个《Differential Transformer》[4], 未来会不会出现一些更复杂的代数结构的Attention处理? Maybe...

参考资料

[1] 
softmax is not enough: https://arxiv.org/html/2410.01104v2
[2] 
nGPT: Normalized Transformer with Representation Learning on the Hypersphere: https://arxiv.org/html/2410.01131v1
[3] 
Grothendieck Graph Neural Networks Framework: An Algebraic Platform for Crafting Topology-Aware GNNs: https://arxiv.org/html/2412.08835v1
[4] 
Differential Transformer: https://arxiv.org/html/2410.05258v1