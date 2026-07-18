# ScalingLaw终结了么?

> 作者: zartbot  
> 日期: 2024年11月17日 05:04  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492660&idx=1&sn=5fd3c82007fc04ca4124a155063436ac&chksm=f995f4f6cee27de050afe97d36bc21d78cd6df1c2a014f1bff1d3bc3a93b6103400c8aa3d767#rd

---

暴论为先: Inference ScalingLaw的实质是:  高阶范畴视角下的Self-Prompt. 是不是能够Scaling,需要模型厂商尽快的去做SAE并开源数据.从高阶范畴的角度来构建另一个维度的Attention才是整个大模型突破的方向.

大概的故事线是:

The Information报道, OAI的训练和模型进化速度变缓,导致GPT5难产已成定局

然后又有某个人在X上说, 似乎另一家在训练的时候遇到了一个未曾遇到的Huge Wall

Bloomberg随后把范围扩大到OAI,Anthropic,Google, 发现大家都在推迟新模型发布.

但是AI社区内大家还是有很强的信心的, 特别是对推理的ScalingLaw

然后就是Ilya Suskever出来补刀: I told you so

有争议的地方必然有Lecun, 也来I told you so

Anthropic CEO Dario出来押注ScalingLaw还继续存在

Sam Altman则直接嘴硬: there is no wall

突然想到去年元旦时读过的《献给阿尔吉侬的花束》, 它是一本科幻小说, 故事叙述一名心智障碍者查理在接受脑部手术之后，由智能障碍急遽跃升为无人可及的天才，而后又因手术副作用而衰退变回智能障碍的过程。该书特色是以查理的第一人称观点来叙事，遣词用字、思维方式随着他的心智变迁而有显著的差异。

过去的几年, 我们见证了LLM从智障一样的满篇错误的语法,再到遣词造句逐渐的像人类, 然后可以完成很多任务, ChatGPT和GPT4再到能写代码能做奥数题的o1, 如同接收手术后的查理, 从智障到天才只用了2年的时间, 再加上累计数千亿美金的投资入局, 人类对AGI有着无限的乐观...

但是FrontierMath让所有的大模型变回了智障, 同时我们还发现通过大量的数据pretrain的scalingLaw已经接近终结, 虽然Meta在Llama 3.1的模型中说用过量的数据训练,模型还能取得额外的收益. 但是又来了一篇ScalingLaw for Precision, 过度的训练导致模型量化变得更加困难.

似乎这一切又像小说中手术副作用后衰退变回智能障碍的过程。另一方面桥水在第三季度的持仓中, 英伟达 Meta 微软和亚马逊的减持幅度为 27.48%, 16.65%, 20.19%, 46.64%, 然后Oracle和IBM都清仓了...

英伟达的Blackwell系列接近年末了, 具体的微架构还未公开,然后还零星的爆出一些供应链的问题. 特别是ScalingLaw for Precision[1], 本质上是越来越多的数据“有损”压缩到模型参数重, 数值扰动对性能的损害幅度会变得更大. 其实这个影响是巨大的, 例如按照相同的DieSize Blackwell FP16的算力只比Hopper提高了15%左右.

![图片](assets/1b74d1007095.png)

然而如果更大规模的模型需要持续的在FP16或更高的精度训练, Pretrain的ScalingLaw将会遇到一个物理极限的墙, 虽然短期内数据中心规模还有扩展的可能, 然后内存还有HBM4的可演进的趋势.

但最本质的问题还是在算法上, 这样Self Attention with token by token的大模型算法走到了尽头, 虽然Inference ScalingLaw看上去还可以再继续的冲一波, AI社区普遍认为o1是一个等同于GPT-2的时代的产物. 另一方面最近有一个Think Claude的Prompt, 把Claude强化成一个类似于o1的模型, 一些demo的结果看上去也不错, 但是很遗憾还是无法节约9.9 和9.11的问题. 甚至是数`1111111111011111111110有多少个1`

Claude35_sonet2回复如下

让我来数一下：1111111111(0)11111111110

这个二进制数中：前10个数字是1
然后是1个0
接着8个1
最后一个0
总共有18个1
所以这个二进制数中有18个1。

而利用ThinkClaude Prompt回复如下

```
Hmm, let me look at this number sequence carefully: 1111111111011111111110,  This is an interesting counting problem. Let me go through it step by step... I notice it's mostly composed of 1s, with just two 0s in there. The sequence appears to be 21 digits long in total.Let me count the 1s methodically... going from left to right...I see 10 ones, then a zero, then 9 more ones, then another zero at the end.So that's 10 + 9 = 19 ones total.Let me double check by counting again... 1-2-3-4-5-6-7-8-9-10, then 0, then 1-2-3-4-5-6-7-8-9, then 0Yes, confirmed - there are exactly 19 ones in this sequence of digits.
```

这个数字序列中有19个1。

从个人的一些数学直觉来看, 一个400B左右的模型已经可以很好的压缩已有的数据, 但本质上并未有任何的抽象和逻辑推理能力.上周在和同事讨论一些问题的时候, 在给同事讲预训练模型本质上是在构建一个预层范畴, 然后我们可以通过Sparse AutoEncoder来抽取特征, 然后在这些特征的基础上再进一步探索其几何结构, 例如文章《The Geometry of Concepts: Sparse Autoencoder Feature Structure》是一个很不错的开端...通过SAE的点云在投影下出现了明显的几何结构

![图片](assets/794dfb3e3f78.png)

而实际上Inference ScalingLaw的实质是:  高阶范畴视角下的Self-Prompt. 对于未来是否能够让模型自身构造一系列非自然语言的Prompt, 本质上是构建一个更加泛化的方法去控制Self-Attention产生的token, 在一个更高的维度上构建Context去约束模型.

正如在以前的文章所讲的

[《谈谈大模型可解释性》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490211&idx=1&sn=544e615e159226a1da662e5c658ca1f6&chksm=f9960a61cee183773dd3016e19617951bf3343e22aa816d069d4a7733f5fb9b532568f3b6725&scene=21#wechat_redirect)

同理正如我在[《大模型时代的数学基础(2)》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488528&idx=1&sn=fa49e334201e738e7ddb4258030798b3&chksm=f99604d2cee18dc45a78ee39db2f1c493b4e3f4fae6c3a8ef0b04d1aff8590b8a2b259827f74&scene=21#wechat_redirect)中讲的:

Transformer算子的可组合性如何设计?通过范畴论这样顶层的抽象视角会得出不少有价值的答案。当然还有很多范畴论的内容，例如limit/colimit，以及相应约束下的强化学习和基于Hom函子去构造数据，最终来提高大模型的逻辑推理能力，范畴论视角下函数式编程和大模型的融合，这些都是非常值得我们去深思的问题

个人的观点是国内的几个模型大厂, 留一些资源把SAE做了,然后公布出来让更多的人参与去更多的研究, 反反复复的去追OAI**/Claude这些本来就接近极限的路本质上无疑是浪费资源.

从高阶范畴的角度来构建另一个维度的Attention才是整个大模型突破的方向, 而SAE是高阶范畴的开端.

参考资料

[1]
ScalingLaw for Precision: https://arxiv.org/html/2411.04330v1