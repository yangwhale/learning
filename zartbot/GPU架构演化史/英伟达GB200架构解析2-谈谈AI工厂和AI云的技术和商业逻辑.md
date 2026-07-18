# 英伟达GB200架构解析2:谈谈AI工厂和AI云的技术和商业逻辑

> 作者: zartbot  
> 日期: 2024年3月31日 14:36  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489194&idx=2&sn=4e0a691845a5d4cd736e808c203a46ed&chksm=f9960668cee18f7e83903097a2296cd652d501205e737190354ae022511d963f0a14079ea291#rd

---

前一篇关于GB200互联架构的解析主要是从技术的视角来看待，在公众号后台收到很多机构的行研提问，由于合规问题都无法回答。但是今天就换个身份以一个基金经理的视角来公开写一下，因此以下观点和本人任职的机构无关也不构成投资建议。

从技术路径上来看，GB200的互联系统大概有三个方式， 英伟达的市场策略是很明晰的，如图所示：

![图片](assets/1aa29ae8bea2.png)

对于生成式的AI云和AI工厂的最大区别在于租户不同，单一任务规模不同，底层技术上也通过支持以太网的Spectrum-X和Infiniband**/NVLINK互联区分，那么这三个解决方案就更加清楚了：

互联方式使用场景NVL72 + InfinibandAI工厂标配NVL576 + InfinibandAI工厂土豪特选NVL72 + Spectrum-X生成式AI云服务

另外又有一个思科和英伟达合作，在GTC**上的Session谈到一个根本性的问题：自建还是租？

![图片](assets/b8851b825091.png)

结论很有趣：两者都会存在，但是需要像Cloud那样的使用体验？云的体验是什么？本质上是一个算力证券化的过程。体验统一了，背后的分歧应该只是商业模式上的，是买还是租，或者采用融资租赁等金融工具来买？而实质的技术上不应该有分歧，如果我们把它定义成AI CLOUD-X，对比如下右图所示，这背后又需要什么样的技术呢？

![图片](assets/ef5cca77bd97.png)

其实很多人没有明白云的价值是什么，基本上还是跟IDC混为一谈。根本的区别是云是一个算力金融机构，流动性(弹性)和安全(多租户)是它的根本。

比尔盖茨曾说过："Banking is neccesary, Banks are not". 王坚博士讲过:"计算，为了无法计算的价值(Computing for Value Beyond Computation)". 本质上博士也是在阐述计算的价值，那么照着写一句:

"Computing is neccesary, Computers are not".  这样来看更有云计算的味道，又有了几分Serverless，Datacenter as a computer的意境。

云服务提供商和金融机构具有太多的共性，具体的内容可以参考以前的一篇文章

[《从金融的视角谈云计算》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488712&idx=1&sn=c3ae48e1c0b3fe9bebfdf2b6d8d5bc0f&chksm=f996040acee18d1c86d93b9a44ad9de972f57e0c99086e04a7d20a342f9175ade81fb8e1b51d&scene=21#wechat_redirect)

## [](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488712&idx=1&sn=c3ae48e1c0b3fe9bebfdf2b6d8d5bc0f&chksm=f996040acee18d1c86d93b9a44ad9de972f57e0c99086e04a7d20a342f9175ade81fb8e1b51d&scene=21#wechat_redirect)1. 从商业逻辑上来看AI工厂和AI云

### 1.1 模型规模和弹性

通常金融机构盈利的主要策略是“借短投长”，对于期限错配和流动性风险的管理是金融机构必须要关注的。而对于GB200这样的算力投资来看，云计算厂商考虑的也是类似的逻辑。一次性长期投资下，如何能够尽量的通过频繁的短期租赁来获取收益？

对于超大规模Foundation模型的训练，也就是AI工厂的范畴看，参与者基本上也就会收敛到20家左右，并且训练都是持续数个月的，对于云来说更多的是考虑一种非弹性的融资租赁模式，盈利模式可能更多的是在换代后的推理收益上，下面以一个例子来解释：

例如微软给OpenAI训练GPT-5用了100,000片H100，这些投资前期可能以很低的毛利供给给OpenAI，因为这些长期包年包月的服务并不具有流动性溢价，特别是在国内有多个智算中心的大背景下。而在两年后B100集群上量后，这100,000片H100将拆分出来用于FineTune和推理等业务。

从整个H100的生命周期而言，从现金流的角度来看，中短期内是一个收益率较低的持续稳定现金流，而中长期则是一个看推理/FineTune市场的收益率相对高一些的弹性售卖的逻辑，但是现金流本身和流动性息息相关和售卖率也相关，因此也可以看到英伟达和云对大模型公司的投资，其实是在进行算力的流动性风险管理, 为整个现金流寻找确定性。

因此从现金流来看，AI云服务提供商和AI工厂有着巨大的差异，因此Build or Rent的逻辑更多的可能还是在每个企业如何衡量自身的现金流上， 例如企业自身还可以通过融资租赁的方式来Build替代Rent from Cloud，背后的成本核算是一个很有趣的话题， 例如学过CFA/FRM**的人基本上都懂债券的Barbell或者Bullet两种策略以及其凸性，那么针对AI场景下的ROI分析可能是一个因子更多的问题， 不光是基础设施投资回报还有模型带来的经营收益回报上，后面会针对投资回报率相对确定的搜广推做一些分析。

### 1.2 弹性视角下的AI云

现阶段AI集群存在三套网络

![图片](assets/8bb2f388ffc9.png)

一个是NVLink这样的Scale-Up网络,另一个是基于RDMA**的东西向Scale-Out网络，还有一套是原有的Front-End 存储/管控和南北向流量，今年GTC上Gilad有个演讲《Entering A New Frontier of AI Networking Innovation》[1],也提到了类似的区分

![图片](assets/15717c807f8f.png)

针对网络中的数据流量分为东西向和南北向，主要差异在计算紧耦合/长尾抖动容忍度低/突发强等，宏观的秒级监控来看平均带宽中等，但是微观上突发

![图片](assets/2b3072e6cbc0.png)

这也是英伟达宣称为什么传统以太网无法解决，必须要Spectrum-X技术，或者UEC联盟/ OCP-Falcon 需要解决的问题。

从云的经营视角来看，更多的是关注弹性部署的问题，除了Azure使用Infiniband外，另外两大云服务提供商AWS/GCP**似乎都没有独立的Scale-Out网络，并且都在传统以太网上通过在其DPU上实现了EFA/Falcon协议来支持，例如AWS部署的GH200 NVL32全部使用了Nitro EFA组网，并没有特别的东西/南北流量区分，而事实上这些技术问题确实可以解决并共存，后面第二章我们将详细分析。

### 1.3 AI云服务提供商选择Scale-Up和Scale-Out的平衡

对于云服务提供商而言除了整合Scale-Out和Front-End网络外，还有一个非常重要的话题是Scale-Up和Scale-Out的平衡，AI工厂可能有一些更极端的逻辑去追求极致的性能和探索更大规模的模型，但是上一代的NVL256和这一代的NVL576基本上没有客户买单，AI云服务提供商依旧选择了NVL72，甚至英伟达的推荐也是。在这次GTC《The Next-Generation DGX Architecture for Generative AI》[2]

它通过576个Blackwell GPU构成一个Building block，并称其为hot-aisle containment closet (HAC)，可以看到其内在的逻辑是考虑到物理约束/功耗限制/液冷散热等情况

![图片](assets/944d4b698114.png)

通常由一组互相备份的分布式制冷单元CDUs (cooling distribution units)，配合16个机柜构成，每列机柜包含8个NVL72计算机柜和8个互联的支持机柜

![图片](assets/5ecc5aaafdc5.png)

通过Infiniband构建的Scale-Out网络拓扑如下：

![图片](assets/3ca9edebb2c2.png)

可以看到在单个Pod内构建的576卡集群Scale-Out网络是1:1收敛的，而多个Pod之间的互联并没有构成1:1收敛。整个系统通过这样的两列Sub-HAC构成：

![图片](assets/4bec1040fb91.png)

最终通过把大量的sub-HAC连接构成一个32,000卡的集群

![图片](assets/87af95c73201.png)

官方的介绍中基本上没看到有NVL576的方案，只有在安费诺的一个和连接器相关的Session中介绍了一部分，例如铜缆转光模块的笼子，不确定NVSwitch上闲置的两根线是否跟这个有关

![图片](assets/66e4d4f83a03.png)

从云服务提供商的视角来看，GB200NVL这样的平台比DGX B200的平台更符合云服务提供商对业务弹性的需求，相对于单机8卡，GB200可以按照单机两卡售卖拆散了售卖，同时又可以组装起来构成一个NVL72的大系统。当然对云服务提供商的ROI还需要更多的定量分析，只是个人大概估算了一下GB200NVL的ROI会远高于B200，因为产品中后期的弹性售卖能力和故障恢复能力远高于B200的平台。

![图片](assets/ad8bc78e1c41.png)

### 1.4 AI云的任务编排和调度

另一方面是网络带来的冲突和拥塞控制上，对于大规模AI工厂，通过对节点的编排可以很好的避免冲突。但是对于多租户的AI云场景下由于弹性售卖的逻辑，多个任务调度编排难度极大，编排不当会导致性能损失：

![图片](assets/9fcf278179db.png)

而这个问题通过特殊的交换机和DPU也可以解决，但是英伟达的方案并不干净，必须要使用`无损` `刚性兑付`的网络

![图片](assets/602ab67157af.png)

当然，通过这些技术可以使得整个云服务提供商对Job编排做到位置无感知，可以更好的提供弹性售卖的能力

![图片](assets/e4cbeda12e10.png)

AI云基础设施建设上和AI工厂最大的区别就在于此，它需要考虑GPU生命周期中后期的弹性售卖逻辑支撑多租户和灵活的资源调度编排能力，和碎片化的资源售卖能力。

关于英伟达Spectrum-X具体内容可以参考

[《谈谈英伟达的SpectrumX以太网RDMA方](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489018&idx=1&sn=25b3df2a17d49681edc0e621049b058f&chksm=f9960538cee18c2ed59729db4a0194fd54b6b99670af13bec8a8eb82b250ccb6b0aa5f07caa7&scene=21#wechat_redirect)[案](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489018&idx=1&sn=25b3df2a17d49681edc0e621049b058f&chksm=f9960538cee18c2ed59729db4a0194fd54b6b99670af13bec8a8eb82b250ccb6b0aa5f07caa7&scene=21#wechat_redirect)》

## 2. 从技术上分析如何实现AI Cloud-X

是否存在一个技术能够完成AI工厂超大规模的Foundation模型训练，又能够完成在生命周期的中后期能够弹性售卖？三套网络看上去是不太合理的

![图片](assets/cbee37f72f46.png)

例如GPU针对SORA类应用要从存储拉数据，或者云上需要弹性售卖时，从Front-End到CPU再PCIe灌入GPU显得有些瓶颈。而针对GB200这类业务，其实Front-End的网络带宽也大大提升了。随着SCALE-UP的能力增强对SCALE-OUT网络的依赖也会越少。这一些问题是值得我们去探讨的。

### 2.1 SCALE-OUT和SCALE-UP合并

这是来自于BRCM投资者交流日的胶片：

![图片](assets/9d3e982fe895.png)

伴随着CPO和1.6T以太网，这样的合并价值可能会凸显出来，这也是英伟达未来一代的演进趋势，构建光互联的系统概念图如下：

![图片](assets/0f9aafca76c4.png)

最终构建一个超大规模的光互联系统

![图片](assets/ea372dab5045.png)

### 2.2 SCALE-OUT和Front-End合并

对于AI这些推理应用落地场景最多的搜广推业务，存在大量的CPU实例和GPU实例的交互。例如Meta的一个数据：

![图片](assets/e13e0ca20ec1.png)

事实上针对实时推荐系统来看，十亿级用户行为的捕获，进入Flink这些系统后，用户行为数据通常需要快速的进入推荐系统进行存储并构建Embedding，而伴随着大模型在线推理业务的部署，RAG/AGENT等对向量数据库的需求上都会要求CPU和GPU系统之间有更大的带宽进行通信。

注：对于大模型用于搜广推和GH200/GB200的优势，我们将在下一篇文章中详细阐述。

Grace-Blackwell这些直接CPU和GPU之间通过C2C互联是一种解法，另一种做法就是Front-End和SCALE-OUT网络合并互联。

例如Google在其A3 H100实例上就是这样的部署方式，任何一个通用CPU计算的VM都可以直接通过FRONT-END连接到起SCALE-OUT网卡，事实上也就证明了GCP已经完成了SCALE-OUT和FRONT-END网络的合并，并且协议也没有采用ROCEv2，而是为了兼容有损的FRONT-END网络采用了GPUDirectTCPX或者未来的Falcon。对于AWS的NVL32和NVL72的系统依旧会采用其Nitro构建的EFASRD。

对于英伟达，Jensen在和投资者交流的会议中还在提及无损网络和AR，以及如何实现Noise Isolation等，但事实上有损网络支撑SCALE-OUT网络已经在工业界落地，只需要一些非常优雅的多路径转发和拥塞控制算法即可，这一点上我们也通过上一代传统交换网络进行了验证，并不需要什么超级以太网的新型交换机支撑。

## 3. 结论

本文从生成式AI云服务经营的视角来分析了云对弹性多租的需求，但同时也存在和AI工厂之间的一系列差异，这些差异在商业模式上和技术路径演进上都存在。对于云服务提供商而言，其未来演进会存在一系列网络合并，无论是博通还是英伟达都有明确的路径，另一条路是AWS和GCP。当然GCP还特殊一点，其AI工厂还有TPU这一条线支撑。

参考资料

[1]
Entering A New Frontier of AI Networking Innovation: https://static.rainfocus.com/nvidia/gtcs24/sess/1707189722732001l46P/FinalPresPDF/S62293a%20-%20Entering%20A%20New%20Frontier%20of%20AI%20Networking%20Innovation_1711040929732001ayMI.pdf
[2]
The Next-Generation DGX Architecture for Generative AI: https://static.rainfocus.com/nvidia/gtcs24/sess/1696188785866001bSLb/FinalPresPDF/S62421_1711139422506001ouGg.pdf