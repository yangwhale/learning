# 再来谈谈UltraEthernet的设计原则

> 作者: zartbot  
> 日期: 2025年8月17日 00:57  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494666&idx=1&sn=9092a2af0d38717af113be0c74c4f664&chksm=f995fcc8cee275de3cccb2601ea29b0311f6e6484e3bc73448d38700d770244ed50cdca6039d#rd

---

### TL;DR

最近, UEC的大佬们写了一篇论文《Ultra Ethernet’s Design Principles and Architectural Innovations》[1] 谈了一下过去两年多UEC做的一些工作, 包括它的设计原则和架构相关的创新.

今天一边读一边来分析一下.个人的观点是UEC在重新造轮子的时候过于复杂, 和当年的OSI7层模型一样过于理想化, 最终导致整个实现比较重. 实际上有更简单的处理方法, 并且沿用原有的RDMA Verbs软件生态.

### 1. RoCE的缺陷带来的UEC

文章在Introduction这一章讲的比较清楚了.  现在的RoCEv2无损有序的传输, 对于PFC存在拥塞扩散和头阻(Head of Line Blocking, HoLB)的问题, 以及严格保序带来的路径选择/Hash冲突相关的问题, 拥塞控制难以配置的问题, 这使得AMD/BRCM/HPE/Intel/MS准备成了一个新的组织来解决.

![图片](assets/f9ddee87307b.jpg)

锐评: 其实前两个问题, Panda老师在2007年的论文就阐述了如何利用iWARP Direct Data Placement(DDP)来解决. 第三和第四个问题实质性的就是TCP的SACK和Window Based CC. 实际上你如果能够看透这一切, 很简单区分iWARP的实现和iWARP的协议, 并且配合一个恰当的微架构就能很干净的搞定这些问题...

但是工业界又针对TCP和UDP打了太多口水仗... 实际上是把TCP拥塞控制和TCP当前在Linux Kernel的实现混在一起在吵架,  当你发现实质性的需求时, UDP做完SACK和Window Based CC的加法后, 区别只是在IP PROTOCOL上用哪个值...

#### 1.1 UEC Scope

其实UEC这个图的定义比原来的ScaleUP/ScaleOut要好一些.我一直以来都在有这样一个疑问, `带有内存语义的ScaleOut算不算ScaleUP?` 另一个问题是ScaleUP的规模要多大的问题. 实质性的问题就是ScaleUP这个名词腐败导致的.  一方面它让人联想到大型机, 另一方面本来就是多颗芯片互连, UP/Out的边界并没有定义清楚.

![图片](assets/f5c3094c8d76.png)

UEC的这个定义算是比较清楚了, Local Network并且约束了<10m距离和<1us的延迟. 然后Backend Network用于区分Frontend Network.  实际上这样的描述更接近真像了, ScaleOut和ScaleUP如果都支持内存语义, 实际上它们的边界就逐渐模糊了...

上周和deepseek的同学聊了一下, 这周又和BRCM的同学聊了一下. 基本上都在阐述这样一个观点:

对于机柜或者双并柜的场景, 也就是UEC定义的Local Network, 基于Credit Based Flow Control 配合Lossless并辅助LLR这些机制并没有什么问题, 同时通常只有一层交换网, 也没有乱七八糟的ECMP/Hash相关的问题. 因此更重要的是要考虑GPGPU/NPU的内存子系统的设计, 如何降低平均访问延迟使得它能够适配更多的算子需求.

对于跨机柜的互连, 实际上就是UEC定义了Backedn Network, 需要考虑如何打满整个Fabric, 特别是在微秒级的microburst情况下, 另一方面可靠性的问题需要提高到一个更重要的位置, 此时网络因为成本的考虑做一些带收敛比的设计时, 此时两种选择, Lossless并采用Deepbuffer去硬扛, 或者支持Lossy.

Lossy的支持或者说更好的拥塞控制+丢包恢复机制, 对比 DeepBuffer的Lossless, 实际上是用算力去置换芯片Buffer SRAM/或外置DRAM占用的面积和成本, 可惜很多人在微架构设计上, 特别是Nvidia Mellanox长期在CX系列走ASIC路线时, 缺乏这方面的积累...

最终你会发现实质上就是在现有的芯片面积约束下, 如何获得更高的PPA做出选择, 结论是Lossy更优. 其实你会看到接下来UEC的设计原则也是在这条路上.

### 2. UltraEthernet的主要功能

使用临时的数据包传递Context(Packet Delivery Context,PDC)来构建connectionless的传输协议

去除语义层中面向连接的依赖, 例如Buffer addressing/access auth/ error model

支持per-packet的多路径转发(packet spraying), 并避免接收端的重排序开销.

支持in-order和out-of-order传输, 并且同时支持unreliable/reliable方式.

支持Lossy传输, 并结合packet trimming或者其他快速丢包恢复机制.

拥塞控制算法, 快速适应网络拥塞和incast

允许设备厂家实现纯硬件/纯软件或者软硬件结合的产品设计

端到端加密

链路层优化

#### 2.1 ECMP PacketSpraying

其实就是一个老生常谈的问题了, 在传统交换机中, 通常针对数据包头的5-Tuple进行Hash转发, 因此单个QP或连接通常只有一条确定性的路径. 如果出现Hash冲突的情况, 将会导致通信带宽受损

![图片](assets/2fc45ebd8f94.png)

那么实际上我们在转发过程中引入额外的信息熵即可, 这也是论文中2.1节结尾讲述的:

“ UE 的数据包喷洒可以通过更改每个数据包的 EV 来避免这种极化，从而在预期中将数据包均匀分布在所有交换机上。即使发生哈希冲突，它们也会很短，由此产生的不平衡可以在交换机缓冲区中吸收。这导致网络利用率的完全提高，随着时间的推移平均分配流量。如果所有端点均匀喷洒，则数据包喷涂很简单，但如果某些流需要按顺序交付，从而确定性地占用某些路径，则数据包喷涂就更具挑战性。UE 提出了各种可选的负载均衡算法来确定如何为每个数据包设置 EV。寻找最佳的此类方案仍然可供供应商区分和研究。 ”

其实它和拥塞控制紧密的耦合在一起, 如果采用各自打补丁的方式, 最终为了解决这个问题会变得异常复杂, 各种corner case处理不干净, 导致整个协议栈会变得更加厚重. 稍微巧妙的结合一下即可, 非常简单的一个处理就行了, 但是我看了一下今年Sigcomm的一些论文的资料, 以及其它几个大厂的实现, 似乎这一块大家都还是做的太复杂了...

#### 2.2 UltraEthernet Profiles

UEC规范还定义了HPC/AI Full/AI Base三种Profile. 例如针对MPI/OpenSHMEM的优化, 以及一些集合通信库 xCCL等...

说实话这部分内容有些过度设计了, 我并不看好libfabric这类的软件接口, 继续延续RDMA Verbs接口不好么? 有什么值得破坏这个接口去另外构造一个生态呢?

### 3. UltraEthernet架构

整体架构如下:

![图片](assets/52cc8964b15d.png)

主要就是定义了上层语义层(SES), 包传输子层(PDS), 拥塞管理子层(CMS)和安全传输子层(TSS), 然后数据链路层还可以有两个可选的扩展, CBFC和LLR.

#### 3.1 SES传输语义层

我不明白是什么原因选择Libfabric的, 但是UEC论文的解释是: 高效/轻量的设计接口以及将传统网络中的语义(寻址/completion/鉴权/故障处理)和连接的概念分开.

对于RDMA Reliable Connection而言, QP Scale的规模一直是被诟病的一点, 特别是在一些早期的Mellanox网卡实现上,实质问题是QP Context Cache missing后, 要从主机内存中拉一份下来导致了性能下降.

了解到这个实质后, 其实Memory Rich架构的卡(eg BF3)在本地能够缓存足够的QPC后, 已经可以很好的支持超大规模组网了. 另外, 关于QPContext的问题, SRQ和DCT都可以缓解很多问题的, 是否有必要再去动软件生态? 至少从一个云服务提供商的视角来看, 新的软硬件交互接口这事上, 还是持保留意见的, 例如AWS SRD就是一个反面的例子. 标准的RC兼容的生态还是会省掉很多不必要的开发.

3.1.1 Addressing
原来面向连接的QP被改为了JobID和目标进程ID(PIDonFEP)的抽象, 再配合不同的Resource Index表构建.

![图片](assets/c29ae35d647a.png)

文章中Claim了一点"A favorable side-effect of the design without queue-pairs acting as connections is that it enables simpler per transaction failures and error handling instead of the queue error states that add significant complexity in traditional RDMA systems."

确实基于事务的Error Handling还是很不错的一个设计. 但是在一些需要混合Ordering的场景下, 还是有很多东西需要仔细设计的.
3.1.2 Messaging and Matching
在接收方是以RI(Resource Idx)关联接收队列的, 然后支持了一些类似于Tag Matching的机制. 通过数据包携带的initiator ID和Matching Key来进行硬件消息匹配. 但是Matching机制在硬件实现上还是有不少挑战的.
3.1.3 处理大消息
其实RDMA TagMatching中也有这样的处理机制, Rendezvous transport, 只是借助tag matching做了一些扩展, 实际业务上呢,  这些功能的用处并不显著....XD

![图片](assets/c88e4c7097d1.png)
3.1.4 RMA Read/Write
原文如下:“UE’s write is straightforward in that the full address, including Job ID, PIDonFEP, resource index, a target memory key, destination address, and offset is encoded into each packet such that packets can be written out of order into the destination buffer. As a side note, this also applies to send/receive messaging.”

实质性的就在描述20多年前的iWARP Direct DataPlacemnt

![图片](assets/667c588c740a.png)

DDP通过Header中携带的MSN和Message Offset, 本质上是RoCE消息编码的问题, 重新用DDP搞了一下而已.

#### 3.2 PDS

主要就是四种, Reliable UnOrder(RUD), ROD, UUD, RUDI. 最后的RUDI加了一个幂等操作. 文章中有一段话:“Its downsides are that it is tricky to use (users need to ensure consistency across synchronized epochs) and it does not apply to nonidempotent operations such as atomic addition“

其实这里涉及到一个问题, 如果UltraEthernet要做In-network computing, 如何保证Reduce的幂等呢? 其实这些东西几年前我就写过, 要保证Semi-Lattice.

然后Dynamic PDC creation本质上和DCT也差不多

![图片](assets/a25e78462f34.png)

然后就是比较关键的快速丢包检测和PacketSpraying这些功能, 然后引入了SACK. 说实话,整个UEC的设计还是有不少问题的. 设计上还没有找到一个极简的方法. 其实Out-of-order窗口和丢包窗口归因, 再配合RACK和Tail Loss Protection才是正路.

#### 3.3 CMS

拥塞控制, 这是在UEC里还有不少争议的地方. 一方面作为交换机厂商希望能够增加更多的网络上的信号, 做端网协同, 也就是 Network Signal-Based CC(NSCC). 然后另一方面可能更希望用Reciever Credit-Based CC(RCCC).

这里也不展开了, 实际上PDS要和CMS进行协同设计, 这一点可能工业界的同学们还没有完全想清楚

#### 3.4 安全

我先来看看IPSec相关的问题, 主要就是Security Association DB容量做不大.  因此当时思科开发了GETVPN(Group Key)的方式构建. 然后Google的PSP做了一些不错的工作. MACSec呢需要逐跳的去做还是有一些问题.另一方面其实一些PCI Complaince要求了必须要Pair-Wise Key, 这些也是需要考虑的.

另一方面是Anti-Replay需要一些PSN的机制和多路径带来的Out-Of-Order传输需要协同. 这个也需要好好设计一下. Anti-Replay Window的设计还是挺有趣的一个话题.

#### 3.5 LLR和CBFC

其实个人觉得这两个东西没有太大的用处. 拥塞控制做的足够好的情况下, 丢包的概率还是可以很大程度的降低的. 然后配合RACK+TLP即可,

### 4.总结

其实整个协议设计在多个公司构成的联盟内, 有太多的取舍. 导致其协议设计挺重的. 首先如果不改造ULP,直接使用RDMA Verbs接口这些事情能不能做?

其实前面的那些问题, 只需要在RoCE上引入iWARP的DDP机制即可, 增加一个MSN/MO的字段.

然后是拥塞控制和多路径负载均衡, 实际上两者之间有一个非常简单的算法就可以搞定. (专利保密的原因, 还是不能多说....)

最后是关于Lossy和Lossless, Lossy的一些实现上,引入SACK, 并且配合RACK-TLP机制就可以很好的完成丢包快速重传了.

其实这些就是真正的第一性原理出发对RDMA进行现代化改造所必须的. 最近很多RoCE增强的协议出来, 暂时还没看清楚, 让子弹飞一会儿吧....

当然, 还有一个问题是, UEC这么厚重的传输层协议, 在ScaleUP和ScaleOut逐渐融合的时候, 又遇到了困难. 如果是你需要设计一个UEC LocalNetwork + UEC Backend Network融合的架构, 怎么设计呢? 事实上又回到了很多年前定义的一个问题, Intra-Host和Inter-Host总线/传输的本质区别上.

夏Core以前有个图:

![图片](assets/3b4c8b11611d.png)

然后有一段话:

如下是我个人观点，100m ~ DC这个范畴，肯定是Read/Write/Send的空间，而3~10m的Rack内外，应当选择Load/Store/Atomic，至于10~100m之间，则尽可能驱虎逐狼，适者生存。

参考资料

[1] 
Ultra Ethernet’s Design Principles and Architectural Innovations: *https://arxiv.org/html/2508.08906v1*