# 从GPT4o谈谈接下来多模态推理小高潮对AI基础设施的需求

> 作者: zartbot  
> 日期: 2024年5月19日 04:50  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489974&idx=1&sn=f9b2d896804886ffd1c31d23ac5c7dd9&chksm=f9960974cee18062db4a91e076e355cea930f1c15c898cc19729c20df9816f27fc4f898d6b4c#rd

---

随着GPT4o for iPhone 以及Google Project Astra for Android, 大模型推理业务将逐渐进入一个小高潮,再伴随着国内各个大模型厂商逐渐开启的价格战,有些厂商开始谈模型不能看价格要看疗效,又有些厂商没融到几个钱的忽悠说大家不要补贴, 还有给大模型刷火箭的, 但是又有多少人想到免费背后实际上获得的数据价值呢?为了应对九月出现的推理需求激增, 模型和基础设施要做些什么应对? 例如直接实时的语音/视频多模态交互下带来了一些新的需求, 例如在GPT4o之前, 基本上智能语音对话都是 ASR+ LLM**+TTS三个独立的系统串接

![图片](assets/ee133f0fd9a7.png)

三个独立的系统相当于每一步都是store-forward的处理方式, 在中间没有LLM的时候, 一些传统的NLP**模型计算延迟较小时, 并没有太多的用户体验上的差距. 而加上LLM以后还要保证端到端的TTR(Time-to-Response)小于200~300ms就成了一个难题, 这些也对整个推理业务的体系结构带来了冲击, 下面先会分析一下GPT4o,然后分别从计算/网络/存储这三个角度谈谈个人的看法,与任职机构无关.

## 1. GPT4o的一些分析

当前GPT4o的做法如下, 对于文本我们可能已经非常熟悉如何通过词表Embedding转换成向量表示了, 在GPT4o中对词表做了进一步增强到200K, 但是清洗质量似乎...当然污染源应该是中文互联网中的很多你懂的免费视频内容带来的污染:)

![图片](assets/8a31d97052b8.png)

对于音频的Embedding其实也是非常简单的, 通过Mel-Spectrum 然后再通过一个卷积神经网络来构建, 如下图所示:

![图片](assets/2ecca8622885.png)
通过将输入的语音带重叠的滑动窗口进行FFT的到一系列分段的频谱信息,但是人类并不按照线性比例感知频率, 例如人可以轻易的分辨500Hz和1000Hz的差异,但是无法分辨清楚10000Hz和10500Hz的差异.因此1937年 Stevens, Volkmann, and Newmann 提出一种基于音高等距的Mel-Scale运算

然后紧跟着一个CNN再配合Transformer Encoder-Decoder就构成了Whisper一类的ASR系统

![图片](assets/84baaeae4f65.png)

对于图片数据通过像素点配合CNN也已经是很成熟的技术了, 而视频则是进一步切分成一系列Patch编码

![图片](assets/90cf3a793cfa.png)

GPT4o直接把所有的数据通过词表/Mel-Spectrum/Video Patch送入模型,然后通过统一的Transformer构建态射,即GPT4o生成的第一张图片中内涵的P(text,pixels,sound) with one big autoregressive transformer

![图片](assets/d0730164805c.png)

另一方面ilya点赞的这篇论文《The Platonic Representation Hypothesis》[1]也在说这事.

![图片](assets/91978ba2e61d.png)

其本质上我们可以从TOPOS的角度去看待这个问题, 即几何和逻辑的某种意义上的统一, 下面这本书后续会在公众号慢慢写一些读书笔记.

![图片](assets/c027ba50c12b.png)

简单的说在《大模型时代的数学基础(2)》中，我们介绍了一些基本的范畴论知识，对于一个对象A，大模型的预训练过程实际上是通过尽量多的数据来构建A和其它对象的Attention的集合，实际上是,它是一个反变函子，也可记为

`定义` 函子范畴中最重要的一个例子就是预层(presheaf)范畴，记为， Presheaf是C上的一个函子, 上的所有presheaf构成的对象和presheaves之间的自然变换构成态射，这样的范畴被成为预层范畴。本质上大模型的预训练过程实际上就是需要构造一个预层范畴。

而预层范畴是一个Topos，有一篇论文《The Topos of Transformer Networks》[2], 2002年菲尔兹奖得主 Laurent Lafforgue（注：Laurent 2021年加入华为巴黎研究院，参与Topos理论的发展与其潜在应用）的两个演讲《Some possible roles for AI of Grothendieck topos theory》[3]及《Some sketches for a topos-theoretic AI》[4]

关于TOPOS的内容, 最近忙完一个项目后再来详细阐述一下.  接下来我们详细来谈谈对大模型AI基础设施的需求.

## 2. 计算

这两天有一些消息说ilya离开OpenAI是因为Sam要砍他的算力用来经营. 实际上也面临了一个囧境, 现在的很多大模型为了效果越来越大只能在训练卡上推理了, AGI的理想和现实经营的ROI的妥协, 进一步来看未来几个月移动端和价格战的双重影响下会导致GPU算力的流动性紧缩, 本质上就会落到当前很多搜广推系统的做法, 算法对算力的妥协.

那么在供应链紧张的时候, 以及ROI约束下,缓解算力流动性紧缩的方法有哪些呢? GQA/MLA可以改善一些KV-Cache的用量增加推理速度, 这是从模型算法层面上来看. 但是另一方面是否能够为未来几个月到来的流动性紧缺训练部分模型充分利用一些老的GPU卡和CPU来进行并行推理?

在国内普遍算力吃紧的时候,我并不认为去克隆一个GPT4o这样的完全多模态的大模型, 或者说在推理阶段通过Pipeline的方式将Mel-Spectrum/视频Patch/Text Embedding这些计算在端侧计算,或者由通用CPU实例计算来降低GPU的算力开销.

另一方面推理上来看, Intel的SPR/EMR的AMX能力也在增强, 多个CPU实例一起构建并行推理集群在GPU和HBM**吃紧的时候也是一个值得考虑的, 因为首先是要考虑如何能够快速获客提高市场占有率, 例如阿里云已经在通用计算实例中普惠支持了RDMA**,这样对于一些中大规模模型的推理也有了一定程度上的GPU替代能力,特别是针对大KV Cache场景.

![图片](assets/2974b7ba4a07.png)

模型本身的架构上是否能够为这样的推理进行优化? 例如在MoE的实现上采用Dense-MoE,让GPU实例去跑一部分Attention,或者是按Head dispatch到多个CPU实例,然后Experts放置在多个CPU实例上,然后利用Dense-MoE来overlap延迟,或者更进一步尝试采用MoD/MoDE的方式来为CPU推理进行优化,这是很多大模型最近几个月需要首先考虑的事情.

## 3. 网络

网络带来的一个变化是, GPT4o开始的实时语音视频交互给推理长尾抑制带来了极大的挑战,这里带来的几个变数是, 东数西算下在西部构建算力集群可能将无法承接近实时的推理业务,另一方面是RTC**相关的技术和LLM推理融合,就连OpenAI也开始招聘RTC工程师了

![图片](assets/a851cd06cc68.png)

很多人总是在设计协议或者架构时并不会通盘去考虑,只是简单的浮于表面人云亦云,乔布斯老爷子有一段话值得大家去反复理解

![图片](assets/24cf19ed5075.png)

这里我还是推荐你们去看看Ruta这样的协议, 一方面是网络协议的设计上需要很深的应用理解和大量的生产实践, 我在Cisco的时候差不多做了15年的语音视频业务和路由器及路由协议设计才得到的Ruta, 另一方面一些聪明的厂商例如字节已经参考这个协议做了RTC的优化

[《Ruta实战及协议详解》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485165&idx=1&sn=412fcb1dd46dd4ef4384a033b0827256&chksm=f996162fcee19f39ab4c995b1be2676779eb5b647ad26dd7017001b75530dfb9a59e9790b37d&scene=21#wechat_redirect)

另一方面就是前面所述的,需要在云环境中针对通用计算实例部署RDMA的能力, 但是现阶段能够在公有云VPC大规模提供RDMA能力的只有阿里云的eRDMA(第八代实例全地域普遍支持RDMA的能力), 一方面可以通过SMC-R让用户基于标准的TCP Socket无需修改代码就可以降低长尾, 另一方面直接使用eRDMA还对CPU和GPU在推理场景下的GPUDirect-RDMA低延迟交互带来优势,而AWS虽然有SRD但是并不是标准的Verbs生态,兼容性存在很多问题.

## 4. 存储

存储上来看,大容量的QLC SSD被疯抢原因何在? 北美这几个客户对高性能存储的需求显著增加. 一方面是针对多模态大模型的训练和推理对存储I/O的需求进一步增长, 而针对云存储BaM带来的一些变化还需要重视

![图片](assets/42d7cd3d2b83.png)

虽然BaM讲的是本地盘的存储,但在推理场景中推理结果的这些生成的数据本身又是一个很好的训练数据集,因此这里的Interconnect对于存算分离的需求还是会很快的到来

![图片](assets/caefe979bb33.png)

参考资料

[1]
The Platonic Representation Hypothesis: https://arxiv.org/pdf/2405.07987
[2]
The Topos of Transformer Networks: https://arxiv.org/pdf/2403.18415.pdf
[3]
Some possible roles for AI of Grothendieck topos theory: https://www.laurentlafforgue.org/Expose_Lafforgue_topos_AI_ETH_sept_2022.pdf
[4]
Some sketches for a topos-theoretic AI: https://mat.uab.cat/~rubio/bM2L/Lafforgue-bM2L.pdf