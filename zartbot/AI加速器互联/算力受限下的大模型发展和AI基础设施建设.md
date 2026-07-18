# 算力受限下的大模型发展和AI基础设施建设

> 作者: zartbot  
> 日期: 2024年4月28日 11:10  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489813&idx=1&sn=d39b0334306d3b220eca935e9b694e84&chksm=f99609d7cee180c162633581494fa07c8dc92bd79df9f50508c7d7d28ddef3e002832c81a248#rd

---

## 0. 为什么要写这个话题

前面零零星星在商业模式/算法/算力芯片架构/算力芯片互联等几个不同的领域进行了分析，这一系列分析现在大概可以串起来做一个简单的小结了，那就是算力受限下的大模型发展和AI基础设施建设这个话题，本文仅代表个人观点，与任职机构无关。

前面的分析包括了一系列文章，探讨了大模型的数学基础，从范畴论/图神经网络/代数结构的视角去分析大模型的算法演进

[《大模型的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

另一条支线是AI4Science从超算的大量偏微分方程数值解来看。

[《AI4Science科学计算》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3412899523293544464#wechat_redirect)

还有就是从比较容易落地的搜广推业务视角看待

[《谈谈AI落地容易的业务-搜广推》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488831&idx=1&sn=192ac23cf674db05d80576c6eac2200f&chksm=f99605fdcee18ceb926a9f59682c7203cc589305be08d9e3cf289da2f489ab50826654e66b88&scene=21#wechat_redirect)

[‍《英伟达GB200架构解析3: 从搜广推算法的视角来看待AI基础设施演进》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489194&idx=1&sn=7d3ef77de01d88400f1016e84e5abe91&chksm=f9960668cee18f7e1f64a526f19b54190230f2e707a9c8648dd68e84731789930a6206e45fab&scene=21#wechat_redirect)

当然也有从AI云基础设施的经营风险角度的分析

[《从金融的视角谈云计算》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488712&idx=1&sn=c3ae48e1c0b3fe9bebfdf2b6d8d5bc0f&chksm=f996040acee18d1c86d93b9a44ad9de972f57e0c99086e04a7d20a342f9175ade81fb8e1b51d&scene=21#wechat_redirect)

[《英伟达GB200架构解析2:谈谈AI工厂和AI云的技术和商业逻辑》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489113&idx=1&sn=4985219c269cea5f858fca75bcb06bd7&chksm=f996069bcee18f8d4144612b211ccac8a142fe9a12632832d2169e76251a1420195188ebf324&scene=21#wechat_redirect)

还有针对GPU硬件架构演进的分析，1980年的SGI到现在GB200架构

[《GPU架构演化史》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=2538479717163761664#wechat_redirect)

另一方面是互联架构的分析，分别谈论了ScaleOut网络，总结了Mellanox十年的研发历程和整个RoCEv2的技术决策失误

[《RDMA这十年的反思》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3398249338911260673#wechat_redirect)

当然也有ScaleUp网络的分析，包括是否能做以太网替代

[《大规模弹性部署：Google如何管理TPUv4集群》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489642&idx=1&sn=db30c4606db2f181f8f602c8e71abf91&chksm=f99608a8cee181be3af4091697b5bdd994a43a7621185e0616c2abcc5feed8fdad09913db8d6&scene=21#wechat_redirect)

[《英伟达GB200架构解析1: 互联架构和未来演进》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489088&idx=1&sn=67f25cf06a1d9128e2ff534d77089688&chksm=f9960682cee18f94d7d46c4e8dd00101cb76a63a2e660947b0d08914a00bf6f527e82dd7c347&scene=21#wechat_redirect)

[《谈谈基于以太网的GPU Scale-UP网络》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489513&idx=1&sn=840d2d139beb6e9b40ac2a0a2b32689b&chksm=f996072bcee18e3d102d67877934f8c12b0ab1661d5b8dfc11250b22ca3c71bb267c1127e83b&scene=21#wechat_redirect)

并且分析了英伟达多Die GPU的内存子系统的演进

[《英伟达GB200架构解析4: BlackWell多die和Cache一致性相关的分析》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489759&idx=1&sn=2c55ec63d6deaeb39ff7f767896ba853&chksm=f996081dcee1810bd399a0730b65bfde4473f8b06fecfb465b51c817d17bb1cd41a32f46b154&scene=21#wechat_redirect)

当然还包含了多个对大模型本身的算法分析

[《测试一下Llama3，并探讨一下不用MoE的原因》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489573&idx=1&sn=74e089a054e24dc2f666ceb32fd5acd1&chksm=f99608e7cee181f1b47d337a273c971edc38f312dc6620a4c27aa4d48bc74117ffcd8ead5b6a&scene=21#wechat_redirect)

[《Infini-Transformer和Mixture-of-Depth》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489371&idx=1&sn=f46813c8e45e8b75c630198dff5eef50&chksm=f9960799cee18e8f82cd1157694cdf6188bd0a0f729941f901bb63292455061daa90dee7d293&scene=21#wechat_redirect)

[《分析Jamba,BTX,CoE等一些新的大模型架构》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489217&idx=1&sn=b4f787fc485f4dfa51eb180ccc352264&chksm=f9960603cee18f15fb227d26c4b872443ee6f41772d32277fabdbe09b56073f0afcc6c12cb51&scene=21#wechat_redirect)

[《大模型时代的数学基础(5)-谈谈MoE和Mixtral 8x7B》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488705&idx=1&sn=f7d81af3260550ed231471b97a1f6260&chksm=f9960403cee18d15b1f80a4b9a92a928cfb9c1db68c3050560d49809d09c8f61ffe65646c792&scene=21#wechat_redirect)

当前对于算力芯片TPP和PD限制下，国内大模型发展的出路在哪？这个是一个系统性工程，需要我们回答一系列问题，因此这个话题分为几个专题来分别阐述。

![图片](assets/74c3de9e9303.png)

从商业逻辑上看待，大模型的短期ROI到中期持续跟进再到长期技术超越上的技术发展路径是什么？

从算法层面该如何解决当前的算力限制问题，并满足长期演进的技术需求？

低算力芯片大规模互联的并行效率问题

低算力芯片大规模调度和编排能力

这几个问题并不能简单的分领域来看待，而应该系统性的考虑。例如从商业逻辑上来看，各地独立建设的智算中心和云AI基础设施的差异是什么？例如这周Meta这样的独立建设的公司虽然Llama3开源的热度很高，但是因为AI支出大幅增加导致盘后跳水18%，相比另两家公有云服务的微软和Google，供需平衡的持续增长使得财务模型更加稳健，从企业经营管理和风险控制的角度来看，我一直强调算力本身的证券化和流动性是关键，也就是云计算讲究的弹性。

另一方面，从算力约束上看算法的创新，这也是我们需要去突破的，Transformer结构出来了7年了，未来几年算法的演进是什么？主要的算力场景是什么？当前的算力建设如何保证未来五年能够提供足够有效的算力？注意是有效算力，例如当模型架构出现变化或者算力场景出现变化？这些问题没有试错的空间，一个错误的集群建设将会浪费上百亿，从国家层面更要充分的考虑。接下来我们从算法开始谈论这个问题，然后逐渐展开到互联效率和编排，最后再从商业逻辑上分析如何构建一个更低风险敞口的算力集群。

## 1.从商业逻辑上看AI基础设施平台建设

在[《谈谈AI落地难的问题》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488792&idx=1&sn=d41f2715b27c182eb8f8bc48547f7273&chksm=f99605dacee18cccc1468bbee9c3b8a9269e769320d5847c5ceb7ca8d2a44ce86c6e3a66d3c6&scene=21#wechat_redirect)一文中有一部分阐述。AI业务落地难的事实并不会因为大模型的出现而改变，本质上我们需要考虑AI落地业务场景自身的业务价值，例如一开始就是做降本类的业务，势必会严重影响其业务毛利率，逐渐演化成一种拿着锤子找到的钉子的项目制推进方式。

而对于平台的ROI则是一个更难回答的问题，对云服务提供商而言，「弹性和多租」的本质实际上是类似于金融机构的“流动性风险”管理。为了满足用户的算力需求并不出现闲置或者挤兑，是云服务提供商需要考虑的问题。计算弹性的利润本质上是流动性溢价带来的。

如果按照传统IDC方式以项目式谈判达成包年包月的交易方式，等同于金融交易中常见的非标合同OTC场外交易模式，对于云服务提供商而言，这些非标合同交易成本极高，流动性管理非常困难。因此需要以更标准化的算力提供方式和用户自服务的交易方式才能提升云服务提供商的流动性管理能力。

Serverless演进过程中，存算分离下的计算无状态化虽然更容易调度和缩扩容，但也给云基础设施带来了巨大的挑战，例如serverless平台的无状态特性需要支持细粒度操作的存储来支撑需要精细状态共享的应用。另一方面还需要细粒度的任务协作方式，和相对标准的通信原语模式。即

从金融机构的视角来看待云计算，做大规模，公共云优先，标准化算力是提供高质量流动性的必然出路。

[《从金融的视角谈云计算》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488712&idx=1&sn=c3ae48e1c0b3fe9bebfdf2b6d8d5bc0f&chksm=f996040acee18d1c86d93b9a44ad9de972f57e0c99086e04a7d20a342f9175ade81fb8e1b51d&scene=21#wechat_redirect)

从这个视角出发，构建的AI基础设施投资是否单独的针对大模型训练，能否训推一体？是否能够从整个投资的生命周期出发，建设初期用于大模型Pre-train，中后期用于接受多租户的推理/FineTune以及搜广推这些业务模型，顺便又编排一些科学计算任务？

但是这些需求是极其割裂的，你可以看到Google在TPU的构建，Borg调度和Pathways编排，对于弹性的理解和算力的实际有效利用率上高于英伟达，TPUv5e的性能类似于4090/H20正好在美国算力限制的边界上，但是可以做到五万块TPUv5e的线性加速能力

![图片](assets/4b4ce5520454.png)
既能满足超大规模算力的交付，又能满足小算力4卡的交付，是否值得学习呢？以前写过一篇文章介绍，后面还会再详细补一篇基于Pathways的编排框架。

[《大规模弹性部署：Google如何管理TPUv4集群》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489642&idx=1&sn=db30c4606db2f181f8f602c8e71abf91&chksm=f99608a8cee181be3af4091697b5bdd994a43a7621185e0616c2abcc5feed8fdad09913db8d6&scene=21#wechat_redirect)

结论: AI基础设施建设上以弹性为先，这是保证平台商业模式可持续的发展的最根本的需求。当然涉及到具体落地的行业评价和供应链分析这些影响投资决策的这些有争议的内容和观点就略去不谈了。

## 2. 从分布式系统的视角

下图来自于Amin Vahdat(Google Fellow and VP and GM, Machine Learning, Systems, and Cloud AI)在2020年SIGCOMM的Keynote

![图片](assets/6b63b4cf47f2.png)

从早期的人机通信到计算机之间的相互通信，以及传统的机器互通或者业务互通，计算机体系结构的设计主要是以`静态数据`和`计算逻辑`为主。本质上是`撰写代码去处理数据`。当发展到第三代末期的时候，复杂的应用程序已经无法在单台机器上运行了。阿里巴巴诞生在这个年代，而阿里云的诞生本质也就是第四代分布式计算架构的代表。多核心处理器的出现，Spine-Leaf这样的分布式网络架构，VPC技术等等。但是这一代架构又遇到一些问题，一方面是由于交互产生的数据流动需要更加实时的计算，从Hadoop这些离线计算框架，逐渐演进到Lambda架构，Storm，Flink这类的实时大数据计算框架的出现，然后进一步演进到像Doris这样的实时的分析型数据库，可以看出人们对`数据交互实时性`的需求。此时系统互联的延迟需求为100us。

### 2.1 AI时代的紧耦合视角

AI/ML的兴起使得计算范式发生了巨大的变化，分布式系统演进到第五代，人开始向机器寻求洞察力。这种洞察力体现在由机器以数据为中心的计算模式，这种模式并不是简单的去处理数据，而最大的变革是要从数据中抽取能够产生决策的代码，典型代表是在线深度学习(ODL)算法在搜广推业务中落地。

在这个过程中又遇到了摩尔定律的另一堵墙，核数量、Cache size、片上网络和功耗的限制使得多核处理器发展也遇到了瓶颈，“per-socket plateau”是在讨论多核处理器性能优化时，表示单个CPU插槽上的处理性能极限，即使增加更多计算资源，性能也不会再有显著增长，因此GPU/TPU等异构加速器件在数据中心内变得更加重要，带宽需求从200+Gbps 激增到1T+bps，并伴随着大量数据的通信，SmartNIC/DPU等设备逐渐出现。

最终第五代分布式系统需要考虑"Perf/TCO-Service"，即整个系统满足SLA要求时，需要考虑如何低成本交付服务。另一方面在单卡性能限制时，需要考虑采用更加紧耦合的方式来获取更高的性价比和可扩展能力。

![图片](assets/b5704bd95968.png)

在第五代分布式系统中，以 GPU、HBM（高带宽存储器）、高速互联网络为代表的分离式（Disaggregation）服务器架构逐渐取代传统以 CPU 为中心的服务器，人工智能智能体（AI Agent）和大模型成为云计算平台的主流服务，深度学习算法逐渐替代传统服务核心算法。

### 2.2 当前现状：GPU异构松耦合互联

当前针对AI/ML场景，当前`GPU作为CPU的附属以二等公民`的形式存在，GPU异构互联存在三套网络相对松散耦合的组网方式，在这种架构中，存储数据需要通过VPC网络和CPU，并经过PCIe总线搬运到GPU上。同时GPU的跨机通信也需要通过PCIe和RDMA技术连接。数据在三个网络中的通信搬移成本较高。

![图片](assets/bf9aea4fa4a3.png)

## 3.算法的演进

可以注意到大量的开源模型已经在使用MoE了，另一方面Mamba这类SSM的模型出现并且和Transformer融合构建Jamba，在MoE的基础上Google又在演进出了Mixture-of-Depth，还有Meta的Branch-Train-Mix，以及一些参数空间融合和数据空间融合的算法。还有Dense Transformer和MoE结合的Snowflake。当然Meta还继续在走Dense的路线。

另一方面对长文本理解上，Transformer-XL/Infini-Transformer等一系列的变化。面对模型架构即将发生重大变化的时候，哪一条路是正确的呢？

### 3.1 从数学本质来看待大模型

任何一次科学技术的革命，本质上都伴随着新的数学工具的引入，这一次也不例外，但当前大模型并没有用到太多的新的数学工具，因此很多年前就谈到一个观点:

这一次人工智能革命的数学基础是：范畴论/代数拓扑/代数几何这些二十世纪的数学第一登上商用计算的舞台。

更详细的内容在如下这个专题中详细阐述

[《大模型的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

#### 3.1.1 Attention是什么

简单来说，从CNN/RNN模型演进到Transformer模型，核心观念是Attention机制。而Attention的本质在范畴论即是一个态射。它是一个双组件(two-Component)框架，受试者基于`非自主性提示(nonvolitional cue)`和`自主性提示(volitional cue)`有选择地引导注意力的焦点。自主性的与非自主性的注意力提示解释了人类的注意力的方式.

`非自主性提示(nonvolitional cue)`和`感官输入(Sensory inputs)`可以通过一个Key-Value Map机制构建,我们可以在内存中构建一个数据库并存放相应的对，定义如下：

注意力机制可以通过如下形式化的方法构建：

`自主性提示(volitional cue)`：我们将其定义为一个Query张量

`非自主性提示(nonvolitional cue)`：我们将其定义为一个Key张量

`感官输入(Sensory inputs)`：和Key有对应关系的Value张量

![图片](assets/215d97dd868b.png)

注意力汇聚时在神经网络中其实就是一个关于和构成注意力分数，然后乘以得出注意力机制的输出和数据库的函数

更一般的来看是和构成注意力权重，然后乘以得出注意力机制的输出

![图片](assets/36c929046f66.png)

然后输出为

一个简单的和构成注意力权重函数为点乘(dot-product)

然后考虑维度增加后的梯度影响，再对注意力权重进行一个缩放，即：

#### 3.1.2 从范畴论的视角看Attention

从范畴论的视角来看，Attention是态射，而更重要的一点是范畴定义中的可组合(Composable)能力，大模型自身的可组合能力本质上对于训练和推理任务的分布式并行/编译器优化以及各种垂直领域模型的组合都带来了很大的便利性，你会注意到无论是LangChain的工具流或者是SambaNova的CoE模型都是算子可组合性的代表，更详细的内容可以参考

[《大模型时代的数学基础(6)-从word2vec谈谈表示论，组合性，幺半范畴和Dataflow Optics》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488775&idx=1&sn=1793eb897beb71ce4a64c9ab44beee6b&chksm=f99605c5cee18cd3481913d17122bb9da63f6385901c9842e8173186e98040d7f91620a91f95&scene=21#wechat_redirect)

回到范畴论本身，对于一个对象A，大模型的预训练过程实际上是通过尽量多的数据来构建A和其它对象的Attention的集合，实际上是,它是一个反变函子，也可记为,我们注意到 Presheaf是C上的一个函子。本质上大模型的预训练过程实际上就是需要构造一个预层范畴。

而对于大模型的泛化，由Yoneda Embedding，对于一个局部小范畴，每个对象包含一个C上的预层：可表示的预层(representable presheaf),实际上也就构成了一个的函子，这些函子构成预层范畴。Yoneda Lemma 这些函子是完全忠实(Fully faithful)的,即任何局部小范畴中的对象都可被对应的预层范畴中的元素表示

而的函子完全忠实的，那么

于是, 当且仅当它们对应的Hom函子同构。而这个推论来看，我们可以说："对象由它与其他对象之间的关系完全决定"

简单来说Attention的本质是通过预训练发现了关系，并构建预层范畴。而模型生成的是基于关系而确定的对象。更详细的内容可以参考：

[《大模型时代的数学基础(2)》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488528&idx=1&sn=fa49e334201e738e7ddb4258030798b3&chksm=f99604d2cee18dc45a78ee39db2f1c493b4e3f4fae6c3a8ef0b04d1aff8590b8a2b259827f74&scene=21#wechat_redirect)

### 3.2 模型稀疏算法优化

[《大模型时代的数学基础(4)》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&chksm=f996046acee18d7c687403c557a6e30155ba0c04cca7e897de3126e88a5d3ba3a2c2fe0507bd&scene=21#wechat_redirect)

对于Transformer模型的演进有详细的分析，工业界大家都在卷更多的参数和更长的Context Length,此时需要更大计算能力，如何构建一个计算复杂度线性增长的模型是我们急切需要寻找的网络架构，用来替代transformer架构。

![图片](assets/a8c9cd95a0c3.png)

如图所示，Context长度为，计算复杂度，同时考虑全连接层计算复杂度

对于TP/DP/PP这些并行的文章是工程上的实现，各种文章也很多了。而这个系列更侧重于从数学和算法层来考虑架构设计的问题，总体来说大概就是如下几个方向：

降低内存占用和计算复杂度，例如稀疏性算子等

增加transformer block之间的连接

自适应的计算时间(例如early stopping)

递归或者层次化结构

更大规模的架构修改

接下来我们从数学的角度来谈论一下这个问题，首先GQA已经基本上被所有的大模型接受，能够有效的用于降低KV的开销。

![图片](assets/4c0ed6c42bec.png)

接下来几个问题是针对Attention和FFN的算力优化。

#### 3.2.1 Attention稀疏化

张奇老师在《大规模语言模型：从理论到实践》这个图画的很直观

![图片](assets/4237cb0a2cde.png)

但是很抱歉，对于大模型的Attention Layer稀疏化的处理做法，这几种都无法满足需求，主要是过分的强调了确定性，同时又丧失了Fullmesh的能力去构建预层范畴。需要的注意的是Attention本身需要一定的Dense能力才能满足需求，直接稀疏注意力都会产生问题。

因此我们需要关注到数据本身的而采用Mixture-of-Depths的方法

![图片](assets/0a553c1d2622.png)

通过数据的Self-Embedding来产生路由决策绕开一些Attention Block才是关键，而不是简单的稀疏Attention。

![图片](assets/672c0c405a6e.png)

当然还有一些逻辑上能够成立的稀疏Attention Layer的做法，例如和图神经网络结合
![图片](assets/ee9c50c59c34.png)

#### 3.2.2  FFN稀疏化

其实也就是MoE模型了，从结构的角度来看是路由到不同专家

![图片](assets/2cc5b114fc40.png)

但是主要存在两大难题：

Dynamic Routing and Load imbalance

Tradeoff between model quality(trim token) and hardware efficiency(zero padding)

从代数上看，Experts越多则前面路由所造成的负载不平衡情况越弱，但是多个处在不同位置的Token集中在一个专家的概率也会低很多，这样会影响模型的泛化能力。个人还是建议像GPT-4/Mixtral那样选择少量的Experts,同时提高Top-K获得更高的Density，不要过分追求稀疏化才能有更好的表现。而计算负载度和不均衡的问题MegaBlocks的处理还是值得参考的。

从代数的角度来看,MoE计算实际上是对Token进行一次置换群的操作，构成

P为一个进行Token位置置换的稀疏矩阵，实际上也构成了代数上的一个置换群的结构，而我们再来看Monarch矩阵，两者代数结构上是相通的，Monarch矩阵定义如下

其中是Permutation矩阵，是Block Diagonal矩阵：

![图片](assets/01acc328ae60.png)

而在MoE中，是需要对Token进行还原，保证原有的Token顺序输出到下一层。

![图片](assets/f38c1c4d1ffa.png)

对于MoE实现的本质问题是，基于Permutation矩阵后构建的稀疏矩阵乘法如何进行并行

Tutel需要维持每个Expert Capacity相同，采用自适应的方式来处理。MegaBlocks不用EC约束，而直接构造Block based 稀疏矩阵乘法来处理。Google Pathsways则是在模型框架上采用MPMD来构建异步化的Dataflow处理负载不均衡的问题。

当然我们也注意到不用MoE的Llama 3，其实它利用了ScalingLaw的另一方面，即提升数据质量。但这样的模型针对日后的FineTune和量化后的能力是否会下降值得接下来几个月去关注一下。从数学上的直觉来看是会产生影响的。

#### 3.2.3 Attention和FFN结合

另一方面最近Snowflake发布的Dense-MoE 也是一个值得关注的点

![图片](assets/b55845817069.png)

其本质上和MoD+MoE应该是同构的

![图片](assets/2c3ac09cca05.png)

#### 3.3 长文本

针对长文本的问题，，Infini-Transformer作为在Transformer-XL上的一个变体可以关注一下

![图片](assets/b7a5cdf6efe5.png)

另一方面是一系列Attention Free的算子引入，例如SSM based Mamba
![图片](assets/79a630b31963.png)

通过Mamba和Attention交替使用来保存长文本信息

![图片](assets/49ac2e31e1ba.png)

这些问题的处理，从数学上来看，实质就是要构建一个有状态函数并且充分利用递归的算法保存和生成信息。

## 4. 从互联结构分析

英伟达对于需要三套网络松耦合的机制做了很多解释，个人觉得这很苍白

![图片](assets/f2f1ccdb2f19.png)

来捅一捅它的痛处，很简单的一个逻辑，因为有三种不同的协议，所有需要三个不同的网络形态。特别是针对FrontEnd和ScaleOut网络的解释，为什么GB200需要接两个CX8的基础上还要再一个BF3？裸金属的是一部分，而更重要的是需要一个以太网和外界连接...你可以看到AWS无论是GH200还是GB200，都是直接Nitro合并了ScaleOut和FrontEnd网络。另一方面还存在一个悖论是，英伟达有不少的论文在讨论GPU Direct Storage，但是在FrontEnd和ScaleOut之间又生硬的把存储放到了FrontEnd，市场策略上的问题。

还有一个问题是Jensen讲这些东西花了100亿美金，而Jim Keller说Ultra Ethernet花个10亿就能搞定。

![图片](assets/d7d1753cd8ad.png)

当然一方面我们需要回答NVLink这类内存语义的协议会出现什么问题，以及基于这类协议伴生的CXL互联也会走向死胡同。本质上这个问题我在写NetDAM论文的时候就讲清楚过

PCIe作为主机内(Intra-Host)各扩展卡和CPU通信的标准已经存在了接近20年，基于PCIe的直接内存访问DMA也被广泛的用于芯片间的通信. RDMA over Converged Ethernet(RoCE)简单的将DMA操作扩展到了主机间(Inter-Host)通信网络构成Lossless RoCE。但是go-back-N的策略对丢包非常敏感，因此DCQCN这一类基于PFC的可靠传输和拥塞控制机制被开发出来，但是随着网络规模增大及VPC等Overlay网络架构的出现，这样的架构将会带来巨大的延迟和抖动以及死锁。Lossy RoCE被开发出来避免PFC的影响，但是依旧无法大规模部署，存在拥塞控制缺陷。

我们重新审视了`主机内(Intra-Host)`和`主机间(Inter-Host)`通信协议，主机内通信由于延迟可控丢包可控通常采用共享内存(share-memory)的模式，而主机间通信则通常采用消息传递(MPI)的方式，因此两者在设计原则上有根本性的不同：

`拓扑`：主机内通信协议通常是有固定的树状拓扑的，并且设备编址和寻址相对固定(例如PCIe使用的DFS),消息路由相对简单。而主机间通信协议通常是非固定的并且有多路径支持和Overlay支持会使得报文调度更加复杂。当然有一些片上网络总线例如AMBA CHI可以实现多跳通信，但是CHI总线更多的用于片上网络设计，对于跨芯片传输和跨主机有丢包和延迟的以太网传输则不适合。

`延迟`: 主机内通信协议通常只有小于200ns的固定传输延迟，而主机间以太网通常为数个微秒的延迟，并由于包调度和多路径及拥塞控制等原因会带来不确定性.

`丢包`: 主机内通信通常由于仲裁器和Credit Token调度通常不会出现丢包，但是在主机间通信经常由于拥塞或者中间节点失效导致丢包，实现不丢包的以太网代价巨大并且成本过高而且网络利用率和复用率较低.

`一致性`：在主机内通信由于往返延迟非常低，因此通常采用基于MESI一类协议的缓存一致性协议实现共享内存的通信。而在主机间高延迟的情况下实现一致性会非常困难，也带来了编程模式的挑战,另外针对GPU的场景，传统的MESI强一致性根本无法工作。

`保序` : 通常主机内通信为了内存一致性是需要严格保序的，从物理实现上也相对容易，而主机间通信由于多路径和一些网络安全设备调度的因素乱序时常发生。

`传输报文大小` :由于主机内通信实时性、低延迟和一致性的需求下，通常一个flit不会放的太大，大多数协议都最大维持到一个CacheLine(64B/128B)的大小.再大会影响其它设备的实时通信，而且很多协议对于ACK、NACK有严格的时序约束，而以太网通常是1500B甚至9000B的传输。

当然利用Ethernet来做ScaleUP，顺便把FrontEnd一起整合了是否可行？看完下面这两篇你大概自己就能找到答案了。

[《谈谈基于以太网的GPU Scale-UP网络》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489513&idx=1&sn=840d2d139beb6e9b40ac2a0a2b32689b&chksm=f996072bcee18e3d102d67877934f8c12b0ab1661d5b8dfc11250b22ca3c71bb267c1127e83b&scene=21#wechat_redirect)

[《英伟达GB200架构解析4: BlackWell多die和Cache一致性相关的分析》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489759&idx=1&sn=2c55ec63d6deaeb39ff7f767896ba853&chksm=f996081dcee1810bd399a0730b65bfde4473f8b06fecfb465b51c817d17bb1cd41a32f46b154&scene=21#wechat_redirect)

## 5. 从调度编排上分析

英伟达收购Run:AI是一件值得关注的事情，另一方面更值得关注的是Google Pathways框架

![图片](assets/32c79c6904a6.png)

以及Ray的框架，特别是Alpa

![图片](assets/8785d2fcd061.png)

![图片](assets/bb40d386a0e3.png)

当然还有英伟达自己的Legate

![图片](assets/c062296de59a.png)

## 6. 小结

算力受限下的大模型发展和AI基础设施建设是一个系统性的工程，从商业模式上来看弹性多租是降低基础设施投资风险的基本需求，而从算法层面上，有哪些稀疏性的工作可以做有哪些是不能做的，从代数的角度去看待这个问题会变得更加清晰。从芯片层面来看，当单芯片算力受限时，如何实现紧耦合的互联并配合算法及编译器完成一系列编排和优化也是一个非常值得探讨的问题。

大概花了一段时间写了一些文章，目的就是要把这一系列工作串联起来，并不是某个地方追求最优就能达到全局最优的，任何一部分的取舍都是一门艺术，做一些分析，提出一些问题，当然还有很多解法涉密就不多说了，只希望能够尽一点微薄之力，能让我们少走点弯路。