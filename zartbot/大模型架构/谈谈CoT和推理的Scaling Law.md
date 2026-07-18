# 谈谈CoT和推理的Scaling Law

> 作者: zartbot  
> 日期: 2024年9月21日 14:42  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492399&idx=1&sn=a69eed43f684e776b3e335379efb621a&chksm=f995f3edcee27afbbe0c41f92e1fc89eb7943e7052d1004746ca4896cbe600a8b8d4bb20e1f6#rd

---

草莓几天的热度似乎很快的褪去,  大概只有Google和Meta两家的两个华人在怼CoT**有点意思, 当然还有每次吵架都要凑热闹的Lecun...

一方是DeepMind**在今年五月的论文, 另一方是meta的回复...

![图片](assets/c72065a6f546.png)

对于渣个人的看法就是, 你给我足够多的猴子和打印机以及足够多的时间, 我都可以给你弄一个Monkey Scaling Law出来,并且Almost Surely Sky's the limit.

![图片](assets/28a29fcddf91.png)

一群人尬聊吵架真无趣, 好好看看o1这个号称IMO/IOI**金牌级的选手, 20以内整数运算的能力?

![图片](assets/345be989d262.jpg)

突然有点迷茫, 当所有人在谈Scaling Law的时候, 好像所有人又对算法复杂度,可计算性避而不谈, 反正GPU可以每年性能翻倍么? 或者老板投的钱可以每年翻10倍么?

其实问题的关键是: **体系结构的架构师和算法架构师是完全割裂的...  一个比较有意思的问题, 如何给LLM**装一个ALU呢?** 例如O1-mini已经差不多搞定了9x9以内的乘法, 是否可以通过一些特殊的代数结构来外推呢? 也是一个值得研究的话题.  同时从计算机体系结构来看相应的ISA**和ISA Generation的模型相关的研究, 也挺有趣的.

另一方面基础模型的大小和搜索效率之间会进一步形成一个平衡, 例如13B左右的基础模型配合推理Scaling Law比70B的模型,在相同算力下的搜索空间会大很多倍. 这样也有一个好处是对研究的算力约束在变小, 个人和小规模的机构可以参与到大模型的FineTune相关的算法研究了.

**其实当问题进入推理搜索阶段后, 从Chain演进到Tree再到Graph的路径将会变得无比清晰** GNN结合LLM将成为下一代大模型的基本架构.

那么如何约束和修改基础模型的Generation, 同时又不要影响基础模型本身的预训练知识结构的分布? 通过旁路一个非监督学习的概念生成模块,即Sparse Auto Encoder是一个潜在的研发路径. 也是Anthropic和OAI都在做的一些工作

![图片](assets/5bbca63bf98c.png)

详细内容可以参考[《谈谈大模型可解释性》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490256&idx=2&sn=e25763d3bc3236e5cc22e4baed5702a5&chksm=f9960a12cee18304acafa8fcf866fed5e3528a568a7915e2081f0194331b50e57ece4083cab1&scene=21#wechat_redirect). 下一步就是针对这些由Activation生成的概念, 如何约束或者在概念空间去利用强化学习Verify便是一条路径. 正如我几个月前谈到的一些想法[《谈谈DeepMind会做算法导论的TransNAR并引出基于SAE-GNN的可组合Transformer猜想》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490297&idx=1&sn=7d758e84bdce7ae4f20f031f4ac3f221&chksm=f9960a3bcee1832d58956a286d2bc33ca32c69edfb3a00cdad65aec691100f2cc086649aa1bf&scene=21#wechat_redirect). 当时有一段文字如下:

Transformer的架构已经显示出了非常高效的信息压缩能力, 但是过度的压缩使得推理过程中的幻觉和一些计算/逻辑推理类任务还是存在缺陷, 虽然通过一些手段最近一年进步很明显, 但是最近的一些数学高考题来看似乎还是处于学渣水平.

假设一个经过充分训练的Dense Transformer模型已经有足够的信息压缩在模型内,并将其作为基础模型(Foundation Model, FM)

旁置的稀疏图神经网络构成Adapter, 通过CrossAttention或者Activation的权重修改来影响, 通过FM模型配合不同的GNN图构建稀疏的可组合性.

![图片](assets/77c101e763b1.png)
在训练完base的FM后,然后固化住FM的参数,再来训练GNN. 并且这个GNN并不需要每层都有, 而仅是在靠近开头和结尾的地方抽取两层对residual的值旁路有些update.

接下来引入一个对Composable Transformer的假设. 我们是否可以共享一套FM的Dense参数, 然后通过不同的GNN adapter组合的方式来完成复杂任务?

![图片](assets/0049a8c7d4ab.png)

基于Composable的能力会使得大模型的多任务结合变得更加容易. 其实接下来的一个问题就是如何对Adaptive Composable GNN进行训练, 让它成为一个合格的Verifier. 这也是一个非常值得研究的方向.