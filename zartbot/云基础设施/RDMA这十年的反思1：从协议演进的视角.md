# RDMA这十年的反思1：从协议演进的视角

> 作者: zartbot  
> 日期: 2024年4月3日 11:34  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489240&idx=1&sn=53c7512d8551a44834bd405fd38b15dd&chksm=f996061acee18f0c26fb6d3f745dfa717a1f9b41a5f63de139e72acbc00968f4a197a16dd272#rd

---

第一次接触RDMA是2014年在思科通过用户态网卡(Userspace NIC,usNIC)給郑州商品期货交易所测试低延迟交易网络技术，正好RoCEv2也在那一年发布。以下观点仅代表个人，和任职机构无关。

### TL;DR

这是这个系列文章的第一篇，从协议演进的视角来看待。后面还会有一篇从芯片微架构的视角来看Mellanox为什么在早期卷赢了BRCM/Intel等一众厂商，而如今在AI网络中又面临着一系例问题要推出DPU和SuperNIC.

这几天看到郭老师一篇文章《比特分享 | 无损网络、反压及拥塞扩散》[1] ,不由得想起郭老师2016年那篇SIGCOMM《RDMA over Commodity Ethernet at Scale》[2]但是真的at scale了么? 工业界为此折腾了快8年，不停的缝缝补补，却没有发现一开始就弄错了。

然后又说基于`第一性原理`致力于解决这个问题。对于已经解决这个问题的人之一反思RoCEv2这十年，我想借用自己金融风控的身份说一句人人都懂的话：刚性兑付**必定导致流动性风险。换成网工的说法：无损网络必定导致无法承受超大规模组网，对于那些还执迷不悟继续用PFC刚性兑付的厂商，似乎他们自己都没法讲清楚为什么，还偏要在SmartNIC和DPU之间区分出一个新物种。

网络，无非就是三块内容：拓扑，路由和流控，第一性原理只有很简单一句话：Smart Edge, Dumb Core。更为完善的诠释就是[RFC1925]The Twelve Networking Truths.但是同样的错误一次次的反复的在犯，真是有几分无语了，基本上每年都可以举出一些违反十二条军规的反例。反反复复出现的原因也很简单：每年都有一些手都没弄脏照本宣科的学术混子。

本文对RDMA的发展历史做了一个整理，大概的脉络如下图所示，你发现这笔糊涂账记清楚了，第一性原理自然就浮出水面了

![图片](assets/22518b472221.png)

## 1. RDMA/Infiniband的一些历史

### 1.1 RDMA诞生于Kernel-Bypass需求

RDMA诞生于1995年，其中很重要的一个人物是现在AWS的CTO 沃纳·沃格尔斯 (Werner Vogels)，他的论文可以看看《Scalable Cluster Technologies for Mission-Critical Enterprise Computing》[3]，这段历史记录在Cornell的一个课程ppt内《High performance networking Unet and FaRM》[4]，还有一篇记录历史的文章《RDMA [1]: A short history of remote DMA networking》[5]

可以看到早期RDMA的名字是U-Net，即发现Kernel处理数据包的种种缺陷：

![图片](assets/e83261d17da7.png)

最终设计出了RDMA的早期结构：

![图片](assets/9732804f9016.png)

而在1997年出现了基于U-Net接口+ Remote DMA Service的 Virtual Interface Architecture(VIA).

### 1.2 Infiniband

Infiniband诞生和当下AI热潮有些类似。它诞生于互联网爆发初期，也是由于带宽不够用，出现了两个组织 Future I/O和NGIO。NGIO由Intel主导，Sun和Dell加入其中。而Future I/O由Compaq/IBM/HP支持。尽管当时有了PCI-X，但是一些功能强大的计算机也接近了PCI总线极限。InfiniBand 贸易协会 (IBTA) 的成立，其中包括硬件供应商和软件供应商（例如 Microsoft）.

IBTA有一个野心勃勃的计划，同时替代主机内I/O的PCI，机房互联的以太网以及存储这些FiberChannel，同时也准备替代集群互联的例如Myrinet等技术，然后还设想了Composable Disaggregation Server over IB Fabric。

Mellanox也就成立于1999年，致力于开发NGIO产品，而后生产了10Gbps的InfiniBridge产品线

可以看到历史,当转入了处理器能力远大于互联带宽的时候，争议就产生了。随着互联网泡沫破灭，Intel转去玩PCIe，微软也停止了IB开发，转而支持以太网。但如今微软成了几大云里唯一一个玩IB的，又和OpenAI一起想转而支持以太网... 轮回，谁也饶不了谁～

后来随着Myrinet在超算上瞎折腾，Mellanox几乎吃掉了所有超算的生意，并和Voltaire合并。而另一家IB供应商QLogic也被Intel收购了，Myrinet也伴随着被Google收购走到了尽头。但是二十年后，你看看Google Falcon或者Intel曾经的Omnipath都留着一些特殊的印记。

### 1.3 RDMA over Ethernet

#### 1.3.1 iWARP

2002年开始，IETF这群以太网玩的很牛的人就开始定义RDMA over Ethernet的标准Internet Wide Area RDMA Protocol(iWARP)，很明显他们想的问题更大，是要在整个Internet上跑RDMA，因此协议选择上对于乱序/重传/拥塞控制等的考虑，采用了TCP作为可靠传输层。值得注意的是其中DDP这些能够充分帮助RDMA在以太网实现多路径的技术2002年就出现了，2007年SC07 Panda老师就有详细的讨论，而另一边Mellanox到现在都还没做干净，当然这段历史我们稍后再讲。

#### 1.3.2 RoCEv1

2010年的时候，Infiniband和以太网擦出了爱情的火花,RoCEv1诞生，但是这玩意为了低延迟只用了以太网链路层协议，连IP头都没有，似乎忘了自己的祖宗U-Net都还是UDP的了。

Intel当年Diss RoCE v1的视频很有趣《iWarp: The Movie | Intel Business》[6]
![图片](assets/36396b84d6f7.png)

#### 1.3.3 RoCEv2

RoCEv1推了四年搞不定，终于在2014年底想清楚了一点，回到U-Net一开始的时候基于UDP，但是拥塞控制上，由于当年Infiniband想要替代PCI这些主机内总线，必须支持Go-back-N的重传输，并且最终在2016年走上了基于PFC的无损以太网的刚性兑付道路。然后一错就是十年。下一章我们将详细阐述这段历史。

#### 1.3.4 Cisco UserspaceNIC

思科usNIC技术发布在RoCEv2之前，针对RoCEv1思科也Diss过，专门写过一篇博客《HPC in L3》[7],正是在这样的背景下诞生了usNIC的技术：
![图片](assets/ec08d88292e3.png)
它针对当时iWARP实现延迟相对偏大，同期的产品大概在3us左右，因此思科在底层采用了UDP协议并支持Unreliable Datagram语义，同时通过滑动窗口/ACK/重传等机制，延迟降低到1.57us，配合190ns的Cisco 3548低延迟交换机实现了端到端< 2us的延迟。
![图片](assets/7eebeef4b258.png)

详细的技术细节可以参考一个视频《Lawrence Berkeley Lab Nov 2013 talk: Cisco Userspace NIC (usNIC)》[8]

从拥塞控制的角度来看，基于滑动窗口/ACK/重传这些机制和当下的Google Falcon几乎一致了，这实现几乎到了未来两三年UEC和英伟达的终局。但是RoCEv2是怎么误入歧途10年的呢？

## 2. RoCEv2 误入歧途这十年

关于这十年RoCEv2的问题，微软/Google的HPC专家以及Cray(HPE) Slingshot团队和博通交换机芯片团队一起写了一篇论文《Datacenter Ethernet and RDMA: Issues at Hyperscale
》[9],再对比郭老师2016年那篇SIGCOMM《RDMA over Commodity Ethernet at Scale》，在说什么就很清楚了，下面来回顾一下这段历史。

### 2.1 惯性误入刚性兑付

RFC1925网络12条军规里有一条：It is easier to move a problem around (for example, by moving the problem to a different part of the overall network architecture) than it is to solve it.

事实上Mellanox从RoCEv1开始就没有想清楚，只是很简单的想复用其网卡的很多已有设计，简单的换个数据包头，但是2层互联存在一系列问题导致在数据中心根本无法落地，于是很仓促的添加了IP/UDP头构建RoCEv2.正如我前文所讲，网络的问题无非是三块：拓扑，路由和流控。RoCEv2解决了路由的问题，却生搬硬套把主机内通信总线的流控机制简单的扩散到了以太网上。

当在主机内通信时，由于距离短信号完整性好，延迟也低通常很简单的使用了一个Go-Back-N的重传机制。而Mellanox最大的问题就是简单的把这套机制搬出来。

### 2.2 推波助澜的微软

微软很早期就停止了IB的开发，转向支持以太网，RoCEv2出现后，微软对分布式系统做了一个非常出色的工作就是Fast Remote Memory(FaRM)

第一篇论文《FaRMv1: Fast Remote Memory》实现了基于RDMA构建的分布式事务两阶段提交和乐观并发控制(OCC)的方案，利用RDMA单边写消息和环形缓冲区实现了低延迟的消息传递原语，以及无锁读(Lock-Free READ)的算法。

第二篇论文《No compromises: distributed transactions with consistency, availability, and performance》提供了一种维持高性能的分布式事务算法，并在FaRMv1的基础上提供严格的可串行化/持久性和高可用性。

第三篇论文《FaRMv2: Fast General Distributed Transactions with Opacity》中利用RDMA单边操作来协调整全局时钟并根据时间戳排序事务

最后基于FaRM架构实现的一个分布式内存图数据库《A1: A Distributed In-Memory Graph Database》在Bing搜索中落地。

但是在落地的时候，PFC/拥塞树的问题被DSCP-Based PFC暂时的隐藏了，并没有从根源上去解决

RFC1925网络12条军规里有一条：It is always possible to aglutenate multiple separate problems into a single complex interdependent solution. In most cases this is a bad idea.

本质上是一个网卡上拥塞控制的问题，被引入到交换机上做DSCP-Based PFC，现在我们再来回顾《RDMA over Commodity Ethernet at Scale》，真的做到large-scale了么？

![图片](assets/7533e164ab45.png)

正如我所述，在同一时期的Cisco usNIC拥塞控制上比Mellanox好很多，但没有人在意。工业界伴随着25Gbps NIC这一代开始大量采购CX4，其它网卡都逐渐的开始消亡了。

### 2.3 幡然醒悟的IRN

时间到2018年，IRN论文中《Revisiting Network Support for RDMA》[10]描述：

![图片](assets/a7aed6f133e3.png)

过去很多年，遇到很多坑，最后还是想清楚了，RNIC网卡能不能解决刚性兑付的问题呢？论文又拿iWARP出来Diss了一通后，最终还是老老实实的学了TCP的丢包恢复机制，利用选择性重传代替了Go-back-N

![图片](assets/a444808ee1fa.png)

终于放弃刚性兑付的PFC了，但是为什么不参考Cisco usNIC把滑动窗口的CC也加上呢？Rate Based CC在后面又带来了一系列问题。

### 2.4 重入Lossless

时间到了2023年，伴随着大模型的火热，对RDMA带宽的需求激增，而Spectrum-X的宣传又回到了Lossless的老路上

![图片](assets/fac00c099ed6.png)

背后的原因其实很清楚，技术上Packet Spray和Adaptive Routing等技术导致其拥塞控制和2018年的IRN拥塞控制存在冲突，最终放弃了Lossy重回了Lossless的道路。

## 3. RDMA现代化

### 3.1 RoCEv2的一系列问题

微软/Google的HPC专家以及Cray(HPE) Slingshot团队和博通交换机芯片团队一起写了一篇论文《Datacenter Ethernet and RDMA: Issues at Hyperscale
》[11]

对于RoCEv2这十年的批评是十分犀利的，论文阐述如下：

RoCE的核心设计是继承自20年前开发的简单的硬件技术，而在今天的以太网环境中并不是最优的。例如，RoCE使用的Infiniband的简单的传输层实现，大量依赖于有序传输和Go-back-N的重传机制，这需要高可靠的保序网络架构才能保证高效运行。因此，RoCE在没有丢包的保序传输的网络(类似于Infiniband)中表现是最佳的。但传统以太网交换机在缓冲区满时会丢弃数据包，并依赖于端到端的重传机制。为了支持RoCE，"融合的以太网"(Converged Ethernet,CE)引入了优先级流控(Priority Flow Control,PFC)机制实现链路级别的无丢包(Lossless)操作,PFC重新利用了以太网的PAUSE帧。PAUSE帧主要用于在以太网中支持不同传输速率的链路，PFC将PAUSE帧增加了停止(或限制)语义用于对特定优先级流量进行控制，以避免丢包。不幸的是，这一复杂的协议会跨越网络中的不同层并产生干扰，从而降低了一些当今最重要的工作负载的效率。

RoCE的语义、负载均衡和拥塞控制机制都继承自InfiniBand。这意味着所有消息应该按顺序出现在目的地，就像它们是通过静态路由传输的那样，从根本上禁止了许多基于数据包级别的负载均衡机制。对于AI训练工作负载，通常是长时间的流，多路径机制可以极大地提高作业完成时间。

此外，RoCEv2使用基于IP的显式拥塞通知（Explicit Congestion Notification,ECN）的简单拥塞控制机制。当检测到拥塞时，支持ECN的交换机会标记数据包，接收方将该信息传递回发送方，然后发送方会减少其注入速率。在经历一段无拥塞期之后，发送端会自动增加速率。ECN使用二进制标志表示经历的拥塞，缺乏细粒度指示会导致需要多次往返时延（RTT）来确定正确的速率。这种简单的机制与InfiniBand的原始前向和后向显式拥塞通知（FECN/BECN）非常相似。它承诺与其他流量共存，但在实践中很难进行配置。"

最终因为这些缺陷，产生了UltraEthernet的组织，开始对RDMA进行现代化的改造
![图片](assets/ce6282adb6b9.png)

### 3.2 2007年iWARP多路径

最早谈到多路径能力居然是在2007年的一篇论文《Analyzing the Impact of Supporting Out-of-Order Communication on In-order Performance with iWARP》[12]

文章开篇就写到：Due to the growing need to `tolerate network faults` and `congestion` in high-end computing systems,supporting `multiple` network communication `paths` is becoming increasingly important. 也就是当今AI训练网络中非常重要的拥塞和链路失效的问题。解决办法就是采用Weak Ordering的Direct Data Placement实现
![图片](assets/11ea01c934ac.png)

看到这个图明白UltraEthernet讲的Out-of-Order Delivery，In-Order Completion了吧？居然2002年就有的东西现在被称为RDMA现代化，无语。。。。

而这也是Mellanox在15年后做多路径时借鉴的东西，但是对于Re-Order的处理还是有很多缺陷，这些和它的微架构有关，后面一篇文章再来分析

### 3.3 AWS SRD

AWS SRD主要是解决云上超算的问题，一方面是QP爆炸的问题，另一方面是和TCP等其它VPC流量混跑的问题，关于SRD的拥塞控制在论文《A Cloud-Optimized Transport Protocol for Elastic and Scalable HPC》[13]中有介绍，它很好的介绍了多路径转发和端到端拥塞控制的处理方法,当发生拥塞时是换一条路径发呢还是降速呢？对于这个两难的问题，Mellanox选择放弃一个拥塞控制回到PFC Lossless的路上，而AWS的做法聪明很多。

Multipath spraying reduces the load on intermediate switches in the network, but by itself does nothing to alleviate incast congestion problem. Incast is a traffic pattern in which many flows converge on the same interface of a switch, exhausting the buffer space for that interface, resulting in packet drops. It is common in the last-hop switch connected to the receiver in many-to-one communication patterns, but it may happen at other layers as well.

多路径Spraying虽然减轻了中间网络的交换机负载，但是并不能解决incast拥塞，所以端到端的多路径转发和端到端的拥塞控制是两个相互耦合的问题。

Spraying can actually make incast problem worse, as micro-bursts from the same sender, even though originally limited by link bandwidth of the sender, may arrive on different paths simultaneously. Therefore, it is critical  that congestion control for a multipath transport keeps aggregate queueing on all paths to  a minimum.

Spraying可能还会加大incast的影响，因此，对于多路径传输而言，至关重要的是确保在所有路径上的累积排队量维持在最低水平。这一点道出了关键，你不必去精细的测量交换机队列深度，而是尽量保证自己不占用队列

The objective of SRD congestion control is to get a `fair share` of the bandwidth with `minimum in-flight` bytes, preventing queue buildup and preventing packet drops (rather than relying on them for congestion detection). SRD congestion control is somewhat similar to BBR, with additional datacenter multipath considerations. It is based on a `per-connection` dynamic rate limit, combined with an `inflight limit`. The sender adjusts its per-connection transmission rate according to rate estimation as indicated by the timing of incoming acknowledge packets, taking into account also the recent transmit rate and RTT changes. Congestion is detected if the RTT goes up on the majority of paths, or if the estimated rate becomes lower than the transmit rate. This method allows detection of connection-wide congestion affecting all paths, e.g., in case of incast. Congestion on an `individual path` is handled `independently` by `rerouting`.

SRD拥塞控制的目标是在最小化在途字节的前提下公平地分配带宽，防止队列堆积和数据包丢失（而不是依赖数据包丢失来检测拥塞）。SRD拥塞控制在某种程度上类似于BBR，但额外考虑了数据中心环境下的多路径因素。它基于每个连接的动态速率限制，并结合了在途限制。发送方根据接收到的确认包的时间戳进行速率估计，并结合近期的发送速率和往返时延（RTT）变化，来调整每个连接的传输速率。当大部分路径上的RTT升高，或者估计速率低于实际发送速率时，就会检测到拥塞。这种方法能够检测到影响所有路径的全局连接拥塞，例如incast情况。而对于单个路径上的拥塞，则通过重路由独立进行处理。

如果说AWS SRD唯一的一个问题就是和RC生态不兼容，当然它作为第一大云同时国外很多企业都有云上HPC的需求下，客户和软件供应商可以配合改造。但是在国内可能就行不通了，很多软件授权版本较老导致即便上云也要用很多年前的软件版本，Verbs生态兼容的压力远大于AWS。

### 3.4 阿里Solar RDMA

针对存储业务，也有多个QP和对存储块写入非阻塞的需求，传统的RC语义实现带来的一些头阻塞等，于是研发了Solar RDMA协议，公开的论文在《From luna to solar: the evolutions of the compute-to-storage networks in Alibaba cloud》[14]

### 3.5 Google Falcon

首先是对可靠/加密传输层和RDMA语义解耦合，然后利用了成熟的Swift拥塞控制算法，仔细分析一下下面这段话吧

![图片](assets/c4b4c44ac3f6.png)

再想想RFC1925网络12条军规里有一条：It is always possible to aglutenate multiple separate problems into a single complex interdependent solution. In most cases this is a bad idea. 大多数时候这些复杂的依赖都是灾难，Falcon几个亮点的地方是

传输/加密/RDMA语义的解耦合：传输层被简单的抽象成PUSH和PULL两个语义，加密在独立的PSP套件上， RDMA/NVMe作为ULP定义。不同的Header来解耦合。

![图片](assets/8f68b2e400f0.png)

想起一个Mellanox 评价iWARP的图
![图片](assets/bdb4d1fce9b9.png)

其实你就会发现相反，iWARP也是这样定义分层解耦传输和RDMA语义。相对于RoCEv2杂糅在一起，协议设计干净很多，至少RoCEv2后面不也弄上了DDP么？

拥塞控制和数据路径的解耦合，由独立Rate Update Engine负责进行拥塞控制。

![图片](assets/0770a60f9674.png)

但是Google Falcon也是有缺陷的，特别是在多路径能力上，PLB算法针对大模型训练的集合通信并不有效，同时并没有提供像AWS那样的Packet Spray的能力。

### 3.6 UltraEthernet

这几个厂家问题定义的很清楚，但是解法上主要是交换机的Random Packet Spray的能力，另一方面在拥塞控制上并没有太清晰的，有时候我们需要好好想一想是否真的需要依赖交换机的拥塞信号呢？第一性原理：Smart Edge, Dumb Core就在那里，或许这是这个组织在谈论端网协同的概念时，该首先考虑的一个问题。

## 4. 结论

本文回顾了整个RDMA的发展历程， 其中你就会发现RoCE在这上面走了太多的弯路，协议定义上从RoCEv1到RoCEv2一系列的错误，再到PFC的问题困扰了工业界很多年，但反过来看同时期的usNIC已经有解法。虽然Lossy RDMA出来了几年，看上去走上正路了，而后又因为Packet Spray这些AI网络需要的特性又回到Lossless上。

![图片](assets/c0c04ced86d2.png)

另一条路径无论是AWS SRD和Google Falcon的演进则看上去非常干净，少了很多折腾。讲真能很好的了解和贯彻第一性原理的人太少了。总有人喜欢是屎上雕花的创新。

### 4.1 本质的问题是内存语义和消息语义的互通

其实这一系列的根本问题还是在于不同总线协议上的差异，我在NetDAM的论文里详细阐述过：

PCIe作为`主机内(Intra-Host)`各扩展卡和CPU通信的标准已经存在了接近20年，基于PCIe的直接内存访问DMA也被广泛的用于芯片间的通信. RDMA over Converged Ethernet(RoCE)简单的将DMA操作扩展到了`主机间(Inter-Host)`通信网络构成Lossless RoCE。但是go-back-N的策略对丢包非常敏感，因此DCQCN这一类基于PFC的可靠传输和拥塞控制机制被开发出来，但是随着网络规模增大及VPC等Overlay网络架构的出现，这样的架构将会带来巨大的延迟和抖动以及死锁。Lossy RoCE被开发出来避免PFC的影响，但是依旧无法大规模部署，存在拥塞控制缺陷。与此同时，通过一些研究发现，PCIe本身由于RootComplex的设计和驱动的问题，也会带来巨大的延迟，因此GenZ、CCIX、CXL等总线被开发出来用于解决这些问题。

但是我们重新审视了`主机内(Intra-Host)`和`主机间(Inter-Host)`通信协议，主机内通信由于延迟可控丢包可控通常采用共享内存(share-memory)的模式，而主机间通信则通常采用消息传递(MPI)的方式，即`内存语义`和`消息语义`的区别，因此两者在设计原则上有根本性的不同：

拓扑：主机内通信协议通常是有固定的树状拓扑的，并且设备编址和寻址相对固定(例如PCIe使用的DFS),消息路由相对简单。而主机间通信协议通常是非固定的并且有多路径支持和Overlay支持会使得报文调度更加复杂。当然有一些片上网络总线例如AMBA CHI可以实现多跳通信，但是CHI总线更多的用于片上网络设计，对于跨芯片传输和跨主机有丢包和延迟的以太网传输则不适合。

延迟: 主机内通信协议通常只有小于200ns的固定传输延迟，而主机间以太网通常为数个微秒的延迟，并由于包调度和多路径及拥塞控制等原因会带来不确定性.

丢包: 主机内通信通常由于仲裁器和Credit Token调度通常不会出现丢包，但是在主机间通信经常由于拥塞或者中间节点失效导致丢包，实现不丢包的以太网代价巨大并且成本过高而且网络利用率和复用率较低.

一致性：在主机内通信由于往返延迟非常低，因此通常采用基于MESI一类协议的缓存一致性协议实现共享内存的通信。而在主机间高延迟的情况下实现一致性会非常困难，也带来了编程模式的挑战。(注：可以参考OpenMP和OpenMPI在超算中的优劣。)

保序 : 通常主机内通信为了内存一致性是需要严格保序的，从物理实现上也相对容易，虽然PCIe也支持Relax Order但是用处并不是很大。而主机间通信由于多路径和一些网络安全设备调度的因素乱序时常发生。

传输报文大小 :由于主机内通信实时性、低延迟和一致性的需求下，通常一个flit不会放的太大，大多数协议都最大维持到一个CacheLine(64B)的大小.再大会影响其它设备的实时通信，而且很多协议对于ACK、NACK有严格的时序约束，而以太网通常是1500B甚至9000B的传输。

正是这些巨大的差异，很多人只有一个domain的知识就简单的复用到另一个domain带来长达十多年的行业停滞不前，如今还在被这些基本的问题困扰着。至今还有人在谈论是否CXL还可以用在GPU Scale Out互联或者Scale UP互联， UltraEthernet能否做ScaleUP网络？

### 4.2 AI集群需要什么样的网络

其实我们反过来想想AI到底需要什么样的网络？是像HPC那样，追求几个字节消息的延迟呢？显然不是，准确的定义是：在延迟容忍的一个范围内带宽尽量大同时避免长尾的一个问题

那么在这种场景下该如何设计呢？还是回到那句话：Smart Edge, Dumb Core,至于拥塞控制，虽然集合通信是非常Bursty的，但是请记住12条军规里的几句话：

3.With sufficient thrust, pigs fly just fine. However, this is not necessarily a good idea. It is hard to be sure where they are going to land, and it could be dangerous sitting under them as they fly overhead. 如果有足够的推力，猪都可以飞起来。然而，这未必是一个好主意，因为我们很难确定它会在哪儿着陆。当这样的猪飞过头顶是，一定感觉很危险吧~

欲速则不达，其实在很多拥塞控制算法中，很多人都尝试着用更精确的测量队列深度等。为什么不想把Rate based CC换成Window based cc让交换机队列尽量浅就够了呢？而且针对MoE的alltoall和TCP等其它流量混跑还有更多的收益。

5.It is always possible to aglutenate multiple separate problems into a single complex interdependent solution. In most cases this is a bad idea.
对于很多个独立的问题，总会找到一个复杂的相互依赖的解决方案，但通常来说，这是灾难开始的地方。

本质上拥塞控制就是一个标量核的subRTT(10~20us)的快速决策过程，另一方面是需要注意整个系统的分布式拥塞控制的收敛性，在这样快速决策过程中引入任何外部信号或者测量结果都导致系统收敛速度变慢。

6.It is easier to move a problem around (for example, by moving the problem to a different part of the overall network architecture) than it is to solve it.推卸问题总比解决问题简单，例如将问题推到网络架构的别的部分

例如Meta采用Rail-Based组网并且利用一系列路由控制流量工程网络拓扑来避免Hash冲突，而不是从根本上去用多路径的技术解决冲突。

In protocol design, perfection has been reached not when there is nothing left to add, but when there is nothing left to take away.在设计协议时，仅当无法再拿掉什么时才算完美，而不是无法再增加什么

反方向是减的太多为了卷几百个ns的延迟导致的问题，Mellanox在HPC的年代极致的卷延迟，当然这是正确的业务价值取舍。当然真要卷，直接把内存放到网卡上，卷赢Mellanox分分钟的事情

![图片](assets/e085b1ed13de.png)

但是作为一个工业界广泛使用的协议，RoCEv1和RoCEv2带来的一系列问题影响深远，以至于现在还在谈论下面这个问题

![图片](assets/f960c47a13fd.png)

真的需要这样分离么？AWS和Google都没有，那么问题出在什么地方呢？回到RDMA历史演进的那个图，再结合第一性原理，你读完本文应该会心一笑吧：）

参考资料

[1]
比特分享 | 无损网络、反压及拥塞扩散: https://mp.weixin.qq.com/s/pBAH2UGkGbEBA3PWXG90pA
[2]
RDMA over Commodity Ethernet at Scale: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/rdma_sigcomm2016.pdf
[3]
Scalable Cluster Technologies for Mission-Critical Enterprise Computing: https://www.cs.vu.nl/~ast/Theses/vogels-thesis.pdf
[4]
High performance networking Unet and FaRM: http://www.cs.cornell.edu/courses/cs6410/2016fa/slides/24-networked-systems-rdma.pdf
[5]
RDMA [1]: A short history of remote DMA networking: http://thinkingaboutdistributedsystems.blogspot.com/2016/12/rdma-1-short-history-of-remote-dma.html
[6]
iWarp: The Movie | Intel Business: https://www.youtube.com/watch?v=ksXmfZxqMBQ
[7]
HPC in L3: https://blogs.cisco.com/performance/hpc-in-l3
[8]
Lawrence Berkeley Lab Nov 2013 talk: Cisco Userspace NIC (usNIC): https://www.youtube.com/watch?v=ZycqcMEfVo0
[9]
Datacenter Ethernet and RDMA: Issues at Hyperscale: https://arxiv.org/abs/2302.03337
[10]
Revisiting Network Support for RDMA: https://arxiv.org/pdf/1806.08159.pdf
[11]
Datacenter Ethernet and RDMA: Issues at Hyperscale: https://arxiv.org/abs/2302.03337
[12]
Analyzing the Impact of Supporting Out-of-Order Communication on In-order Performance with iWARP: https://web.cels.anl.gov/~thakur/papers/sc07-iwarp.pdf
[13]
A Cloud-Optimized Transport Protocol for Elastic and Scalable HPC: https://assets.amazon.science/a6/34/41496f64421faafa1cbe301c007c/a-cloud-optimized-transport-protocol-for-elastic-and-scalable-hpc.pdf
[14]
From luna to solar: the evolutions of the compute-to-storage networks in Alibaba cloud: https://dl.acm.org/doi/abs/10.1145/3544216.3544238