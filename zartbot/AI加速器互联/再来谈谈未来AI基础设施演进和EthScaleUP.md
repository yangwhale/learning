# 再来谈谈未来AI基础设施演进和EthScaleUP

> 作者: zartbot  
> 日期: 2024年11月11日 17:25  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492628&idx=1&sn=2da998d684527b991ea6e13e6acc4270&chksm=f995f4d6cee27dc047d709d3bd722352ca9de633e2e0dd00857e494d4cdf8adc766560d0a40a#rd

---

本文背景来自于这周的几个消息:

当前大模型在FrontierMath评测中全体落败

The Information 的一篇评论和某个lab在更长的训练和更多的数据后遇到一堵未曾预料的墙

TSMC的一些关于7nm的消息, 国产GPU怎么办

那么对于未来模型和基础设施的演进, 我们有那些办法呢? 其实每个人都在盲人摸象, 国内外都极少有从算法开始到各种芯片架构约束,再到全局平衡取舍来协同分析, 做算法的对于GPU的微架构了解不够,只是简单的归纳总结一个ScalingLaw, 做GPU算子通信优化的对于最底层的计算和通信芯片之间的协同和干扰能够解决的办法有限, 而做芯片的通常只会在自己的领域做到最优,而其实从全局的视角有更多的可以让算法平衡取舍的点.....

那么先谈谈算法, 再来谈谈在芯片工艺约束下的互联,以及以太网ScaleUP为什么是必须要做的事情. 渣B一直讲你可以在这里看到未来5~6年的东西, 例如对于Ethernet ScaleUP这事, 在2021年做NetDAM的时候就无比清晰了...但是这里有一个结论:

**如果LD/ST无法Batch化处理, 出于传输效率考虑, Ethernet只使用其Serdes, 并采用NVLink类似的传输**

**如果LD/ST可以Batch化处理, 传输效率不用考虑, 标准以太网甚至采用VXLAN封装和FrontEnd一起完成三网融合都是可行的.**

**LD/ST可以执行Batch化处理, 以1KB为例, 在800Gbps下传输延迟仅增加10ns. **

## 1. 让大模型集体落败的FrontierMath是什么

渣B “数学方面”的能力大概就是Evans**还是可以读懂一些的, 毕竟高中准备物理竞赛的时候就学过数学分析, 然后大学还读了几年数学系.

![图片](assets/2c53dd836743.jpg)
数学方面~
其实渣B一直在搞一个专题

[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

而围绕着整个专题的一个结论就是:

这一次人工智能革命的数学基础是：范畴论/代数拓扑/代数几何这些二十世纪的数学第一登上商用计算的舞台。

然而当前的大模型所用到的数学基本上还停留在100~200年前. 而FrontierMath涵盖了现代数学的大多数主要分支——从数论中计算密集型问题到代数几何和范畴论中的抽象问题，目标是捕捉当代数学的概貌.

![图片](assets/042bd313aebe.png)

而所有的模型全部落败,正确率2%以下. 其实回到问题的根源就在于当前的大模型的数学基础工具出现了缺陷, 如何把当代的这些数学工具引入到计算模型中, 才是最关键紧迫的事情.

最近很长一段时间脑子里一直有一个画面:

《原子核物理》的教科书中查到氚氚反应截面的理论值是15巴，是氘氚反应的3倍，这一数据来自于权威杂志《现代物理评论》，所以那时候所有人都同意用氚氚反应实验, 而于敏严格证明了，所有轻核反应的截面均绝对不可能超过5巴。节约了几年时间和几个亿.

而当代的大模型ScalingLaw, 或许也是同样的事情, 无论Altman怎么说当前的算力就可以实现AGI, 而渣B坚信的是同样会有一个节约几年时间和几个亿甚至几百个亿的算法. 渣B的三流水平肯定不会是那个人, 但至少想找个地方, 能够分享一些大家尚未关注过的东西, 例如nGPT这样的hypersphere representaion的视角其实再进一步就逐渐的可以过渡到代数曲面上了.

## 2. AI基础设施演进

其实无论TSMC和美国的禁令如何, 渣B一直以来的观点就是: 自力更生,丰衣足食. 一直以来都在探讨一个问题: 在没有CoWoS**和HBM的情况下, 7nm的工艺,如何构建有竞争力的AI基础设施.

### 2.1 算法的演进

从算法上, Transformer一类的基础模型架构已经7年没有更新的, 当然在超大规模的数据和算力的加持下, 从范畴论的视角来看:

以范畴论的视角来看，Transformer的attention本身就是在构造morphism，Pretrain的实质是构建一个预层范畴.

更详细的阐述可以参考[《大模型时代的数学基础(2)》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488528&idx=1&sn=fa49e334201e738e7ddb4258030798b3&chksm=f99604d2cee18dc45a78ee39db2f1c493b4e3f4fae6c3a8ef0b04d1aff8590b8a2b259827f74&scene=21#wechat_redirect)

在这个基础上, 通过Sparse AutoEncoder来抽取特征, 然后在这些特征的基础上再进一步探索其几何结构, 例如文章《The Geometry of Concepts: Sparse Autoencoder Feature Structure》是一个很不错的开端...通过SAE的点云在投影下出现了明显的几何结构

![图片](assets/c44f571128a9.png)

所以对于未来的模型演进, 现在看得到的一个方向就是不要忽视GNN一类的图神经网络的算法. 例如下面的结构

![图片](assets/9fa2191bee4f.png)

从数学上这个结构很有可能和高阶范畴相关. 另一方面稀疏化本质上也是在算法上绕开大算力需求和大访存带宽需求的一条路, 而当前的模型稀疏化的路径从个人的数学直觉上来讲,总是觉得不对, 加上了太多的假设和强制约束.

### 2.2 基础设施在外界约束下的演进

单颗芯片的限制使得Chiplet**, I/O Die 各种互联技术的演进更加迫切. 如今UAlink放弃PCIe那套东西开始, Ethernet ScaleUP就成显学了.

![图片](assets/764c6b03b028.png)

但是使用Ethernet还需要一些改造. 首先我们需要明确RDMA**并不适合ScaleUP网络, 在下面这篇文章中已经详细的进行了分析

[《HotChip2024后记: 谈谈加速器互联及ScaleUP为什么不能用RDMA》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492300&idx=1&sn=8a239883c831233e7e06659ec3425ea2&chksm=f995f20ecee27b185a42d09868bdf6cef64df38267489ee386b6d4425e57d2c2a699ada0f9cb&scene=21#wechat_redirect)

而LD/ST这些内存语义简单的over Ethernet是否可行呢?答案也是不行的, 主要是在传输效率上

![图片](assets/0d22d3ac1921.png)

通常LD/ST的大小为GPU L2Cache的block size, 通常为128B, 使用标准的IP/UDP传输的overhead非常大, 传输效率大概只有66.7%. 因此工业界也在做一些探索, 如何节省一些header, 例如把IP/UDP头砍掉, 效率可以提升到79%, 但是离NVLink 88.9%的效率还差10%. 但是在Batch化的LD/ST后基本上效率差异在3%以内.

#### 2.1 两层组网的问题

这一节我们战且不考虑千卡并行做TP/CP/EP的计算上的复杂度, 单纯从网络上来看, 超过72卡规模需要2层组网的复杂性上有很多需要考虑的地方. Nvidia在Hopper这一代有一个PPT版的2层组网的NVL256, 在B200这一代又消失了... 背后的原因是什么? 对于2层组网下LD/ST的可靠性以及失效保护的实现上, 还有整个传输过程中的负载均衡等问题都是非常难解决的.

例如我们组建一个千卡规模的无阻塞的每卡6.4T的ScaleUP网络, 以102.4T交换机来预估, 需要128个ASW(每个ASW下行51.2T可以连接8卡,上行51.2T拆分成64个800G接口), 不考虑对外互联, 需要64个PSW. 累计需要192个交换机. 中间光模块带来的故障率如何评估.

另一个问题是两层组网的路由该如何设计? 是不是说一定需要在OSI模型**中的三层才能路由呢?

#### 2.2 Packet Rate Improvement

考虑到IP/UDP的传输效率问题, 要不我们把IP/UDP头砍掉? 但是一个纯Layer2 Switching的方案在很多数据中心网络的可扩展性的问题, 路由怎么做?  因此在这种决策上出现了另一个方案, 能不能把Ethernet头砍掉,保留IP头的以太网? 这就是渣B经常开玩笑说的EthZ:基于IPv5的ScaleUP...

[《基于EthZ的以太网ScaleUP互联方案》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491597&idx=1&sn=a76b855416fe3ae614c6f4ebd5ea7bbc&chksm=f995f0cfcee279d90372defc9243ff35e46289823f8f49e48d493311427ac89782f57c22ea1c&scene=21#wechat_redirect)

其实它源自于2021年的一个愚人节的玩笑,我一直也很纠结这种没有Ethernet头的以太网还是不是叫以太网?

稍微靠谱一点的做法是UEC中正在做的一个Packet Rate Improvement的工作组, 也是两三年前渣B跟很多Ethernet Switch厂商一直在提的MAC Routing的需求.

大致的考虑就是在交换机的PortLogic中, 保留EtherType字段的位置不改变, Preemble/FCS/IFG也保持不变, 然后通过一个特殊的EtherType和对MAC地址的重定义来实现一些三层的路由的能力, 这样的做法只需要在交换机的Lookup Table的地方一些很小的改动就可以兼容标准以太网.

但是这样的改动其实是没有必要, 您可以看到实际的效率比NVLINK还是差了10%左右, 然后又要改交换机, 按照BRCM的做法PRI的交换机反正就给GPU用的, 卖个天价也不是不可能.

另一方面考虑到在102.4T或者未来204Tbps交换机上, 还要进行150B左右的报文转发, 交换机的Lookup性能对片上SRAM的冲击, 以及一些拥塞控制和管理的开销, 这种做法未来可演进的空间是及其有限的.

**结论: 放弃修改以太网协议来提高PacketRate和传输效率的做法,这条路从物理约束上以及效率上来看根本无法走通, 从商业逻辑上也讲不通**

#### 2.3 Bulk/Batch LD/ST

其实渣B一开始就在谈内存语义要满足Semi-Lattice

`交换律`可以保证数据可以用UnOrder方式提交

`幂等`保证了丢包重传的二意性问题，但是需要注意的是对于Reduce这样的加法操作有副作用时，需要基于事务或者数据的幂等处理，当然我在做NetDAM的时候也解决了。

`结合律`针对细粒度的内存访问，通过结合律编排，提升传输效率

满足这三条,其实就可以完全的利用当前的商用以太网交换机什么都不改,甚至连FrontEnd网络都可以融合进来实现三网合一了. TTPoE由于没有在网计算, 因此满足`幂等`就是简单的重传即可,但是需要对同一个地址的操作进行一些保序的保护.

另一个满足结合律的做法就来自英伟达的论文FinePack, 通过Packetizer和Depacketizer实现

![图片](assets/8bd33fd0ac91.png)

例如我们以TTPoE 按照1KB Batch成一个packet的方式, 如果我们的传输带宽为单个Link 800Gbps, 实际上batch带来的延迟也就10ns左右, 大概和光纤拖长10多米增加的延迟差不多.

夏晶老师这段话好好读一下吧

可能出乎很多人意外 ：） 实际上Load/Store/Atomic如果做成异步DMA的方式，是可以做到无限的Outstanding，只要Memory Bandwidth大于IO Bandwidth，无需流控，可以无限Load Outstanding。

这个逻辑的本质，其实和Zartbot提出的NetDAM很类似，其实，只有获得了Memory的控制权，端侧的IO的能力才能发挥出蛮荒之力。看明白了NetDAM的话，再进一步，就是无限outstanding的Load/Store/Atomic DMA了。

## 3. 问题的本质

问题的本质, 其实Tesla的一个图讲的很清楚了: 当然具体的为什么, 还有一些涉及到GPU微架构的问题就不多谈了, 做网络的人不懂,做计算的人门清. 例如在一个ScaleUP域内,假设有p个处理器, 通信和计算偏斜对MFU的影响, 还有一些涉及Cache和Memory Hierachy的干扰的问题以及内存地址计算的问题就不多说了......

![图片](assets/f4677d6384d8.png)