# 谈谈RDMA和ScaleUP的可靠传输

> 作者: zartbot  
> 日期: 2025年8月31日 06:42  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&chksm=f995ff90cee27686a3522b9502ccaaae353271b413139f97d08e1306f88a9c90c01ed0a0bce2#rd

---

### TL;DR

*本文大概有16000字, 阅读时间估计会很长, 建议耐心读完. 另外本文仅代表个人观点, 与作者所任职的机构无关. 

Hotchip上AMD的RNIC和Intel Mt.Morgan正式支持Google Falcon, Nvidia又在CX8中增加了PSA配合DPA. 发生了一些变化. 另一方面BRCM的SUE也有一个不需要可靠传输的SUE-lite, 其实实际问题都指向了一点:**可靠传输做多少, 可靠传输带来Die面积的占用, 是否坚持Lossless?**

其实这是一个工程上的TradeOff. 但是很多人忘记的Ethernet成功的关键是保持简单, 并提供尽力而为的传输, 由此推导出Smart Edge, Dumb Core的原则. 但是这也有很多难处, 特别是在ScaleUP上, 可靠传输显著占用了GPU的die面积, 这又是另一个TradeOff.然后还有一个问题是延迟的容忍程度, 这和GPU这些芯片本身的微架构是紧密相关的. 带着这些问题, 本文来做一个详细的阐述.

9月8日在北京有一个很有趣的《超节点以太互连技术论坛》, 具体会议信息参考本文末. 在这里我对要讨论的众多话题做一个会前Overview, 主要目的还是希望工业界的各种争议尽快的收敛起来. 本文的目录如下:

```
1. RDMA演进1.1 RDMA的一些历史1.2 关于Lossless1.3 关于Lossy1.4 RoCE的缺陷1.4.1 多路径和Out-of-order处理1.4.2 拥塞控制1.4.3 SACK和丢包重传1.4.4 Google Falcon的分析1.5 RDMA现代化的实质2.Ethernet base ScaleUP2.1 Overview2.2 内存语义的重要性2.3 在网计算2.4 内存语义和事务2.5 传输效率2.6 可靠传输2.6 ScaleUP域的规模2.7 从XPU的微架构谈谈延迟影响2.8 是否需要可靠传输3. ScaleUP的一些量化分析4. 谈谈ScaleUP和ScaleOut融合的方案
```

## 1. RDMA演进

### 1.1 RDMA的一些历史

RDMA诞生于1995年，其中很重要的一个人物是现在AWS的CTO 沃纳·沃格尔斯 (Werner Vogels)，他的论文可以看看《Scalable Cluster Technologies for Mission-Critical Enterprise Computing》[1]，这段历史记录在Cornell的一个课程ppt内《High performance networking Unet and FaRM》[2]，还有一篇记录历史的文章《RDMA [1]: A short history of remote DMA networking》[3]

可以看到早期RDMA的名字是U-Net，即发现Kernel处理数据包的种种缺陷, 最终设计出了RDMA的早期结构：

![图片](assets/eca1d9e96f33.png)

而在1997年出现了基于U-Net接口+ Remote DMA Service的 Virtual Interface Architecture(VIA).  然后就是IBTA成立展开了很多年的Inifiband的发展. IBTA有一个野心勃勃的计划，同时替代主机内I/O的PCI，机房互联的以太网以及存储这些FiberChannel，同时也准备替代集群互联的例如Myrinet等技术，然后还设想了Composable Disaggregation Server over IB Fabric。 其实看到这里和现在的ScaleUP/Unified Bus本质是相通的.

我们再稍微回顾一下其中提到的Myrinet, 早期的超算市场, Myrinet有着非常大的install base. 其网卡结构如下, 卡上有一个SRAM, 一个DMA Engine, 一个LANai处理器和一个Serdes Chip.

![图片](assets/dc90a754f0d8.jpg)

整个网卡的结构如下, LBUS连接SRAM, EBUS连接PCIDMA Engine, 然后LANai Core可以做很多可编程的处理.在Myrinet网卡上LANai芯片运行的MCP控制程序(Myrinet Control Program),Driver, 可以加载特定的MCP程序到网卡上,并给应用程序提供的LIBGM函数库
![图片](assets/8e8fff646ffb.jpg)

但后期Myrinet的管理和研发出现了很多问题, 很快在2004年开始就逐渐消亡, 而正是这时候Inifiband逐渐走向成功, 一度达到Top500 HPC的50%份额

![图片](assets/4b1d2d953d58.png)

### 1.2 关于Lossless

对于早期的HPC应用, 例如一些偏微分方程数值解, 这些应用对网络延迟是非常敏感的, 因此如何降低网络延迟对于一个数百台或者上千台服务器构成的大规模超算集群是非常重要的. 极度低延迟的需求下, 实际上就是应用本身会有大量的同步阻塞, 即消息传输时的Queue Depth=1的情况下, 网络中的inflight报文数量并不会太高.  所以在这些场景里, 基本上不太会出现丢包.

那么在这种情况下构建一个简单的传输协议层, 相对于完善的L4可靠传输是有很大的收益的. Infiniband的实现其实就是简单的把PCIe那一套扩展出来, 紧接着又继续把它搬到了以太网上构成了RoCE.

在此之前其实在存储路径上, 也有类似的一个演进. 当时大量的存储采用Fiber Channel协议构建独立的存储网络(SAN). 然后每个服务器实际上是需要两套网络构建的. 在2008年左右, Cisco开始用Fiber Channel over Ethernet的方式统一服务器接入网络

![图片](assets/5751cf8e04c2.png)

此时, 由于存储还是大量的HDD构建的, 实际上的I/O带宽并没有太大, 因此简单的使用了PFC构建一个无损的以太网还是可行的.  然后紧接着在2010年, RDMA也开始了Over Ethernet的发展, 推出了RoCEv1. 但是很遗憾它是一个纯2层的以太网转发技术, 搞了4年没啥进展后. 在2014年采用了基于UDP传输的方式构建RoCEv2. 同样是采用Lossless的方式, 假设网络上基本不怎么丢包.

在当年存储基于HDD带宽并不是很大, 而HPC/数据库等应用的同步调用的特征使得其inflight并不是很高, 而两者对于延迟都非常敏感时, 选择Lossless尽力把网卡做简单, 构建一个相对简单的可靠传输, 并采用ASIC架构来降低延迟是一个很好的trade-off.

### 1.3 关于Lossy

其实在2018年开始, 逐渐也遇到一些问题, PFC的问题逐渐开始暴露出来, 例如IRN论文《Revisiting Network Support for RDMA》[4]

![图片](assets/09cc7f6a1936.png)

当然PFC的问题, 配合DCQCN还是能够很好的调整一些配置来处理的, 只是配置相对来说更加复杂. 同时Go-back-N的重传机制并不是十分有效.当然现阶段还可以通过一些交换机的Buffer去缓解, 利用更深的buffer配合拥塞控制去处理. **但是这些Buffer本身也带来了芯片上极大的面积开销, 特别是对于交换机Traffic Manager这样需要多个端口Shared Buffer时, 多端口的SRAM带来的开销是非常大的.** 那么此时是否有另外的Trade-off呢?

有一个非常值得我们去思考的问题, 整个系统发生了什么变化导致原来工作的还算不错的系统出现了问题需要修改?

实际上就是新兴的一些GPU应用带来的. 2018年我在思科做AI Infra时, 也意识到了这个问题. GPU本身的微架构warp调度这些以及一些Kernel编程逐渐异步化的处理, 通信和计算的Overlap可以做的很高, 此时整个网络中的inflight会很高, 如何打满整个网络的带宽成为更重要的一个需求.

另一方面存储上, 特别是针对云的多租户存储, 伴随着全闪存(SSD)的存储部署下, 实际上整个系统I/O传输的inflight也增加了, 因此网络中的一些拥塞变得更加显著.

因此出现了对RDMA现代化的需求, 产生了UEC

![图片](assets/2a54d6d9790c.png)

其实还有另一个问题, 针对云上部署时, 通常云采用VPC部署的方式将多个租户隔离, 基于Overlay的网络下, PFC这些Lossless的机制是无法实施的. 对于云服务提供商, 或者说AI Cloud而言, 相对于AI Factory更需要一个Lossy的多租户的环境.

### 1.4 RoCE的缺陷

前一节谈到了RoCE的一些缺陷, 客观来说从2016年到2020年这一段时间来看, RoCEv2 Lossless的路径确实解决了当时HPC和存储的一些问题. 但是也逐渐暴露出一些问题出来.

#### 1.4.1 多路径和Out-of-order处理

首先RDMA Reliable Connection是一个strict order的传输, 要求数据包按序传输, 一旦出现sequence gap就可以判定为丢包触发Go-back-N的重传恢复. 当数据量越来越大, 同时数据流传输时间越来越长时, 网络中的ECMP冲突就成为一个必须要考虑的问题.

![图片](assets/3a812e3c61f4.png)

例如上图所示, 蓝色和紫色的GPU的数据流量很可能被交换机Hash到同一个Link上(Link 1)导致性能减半, 而Link4又完全空闲没有流量. 当然我们可以注意到这种情况只在流量通过两层以上交换网时会存在, 那么能否通过拓扑变更把流量尽力约束到1层内, 降低hash冲突的概率呢? 这就是多轨道(Multi-Rail)的部署方式.

![图片](assets/c7aacc76b1bb.png)

即每个机器的同样Rank的GPU连接到同一个轨道(Rail)的交换机上. 尽力在一个更加扁平的网络上通信. 如果需要跨轨道通信, 例如 Node1 GPU1需要和Node3 GPU K进行通信时, 可以通过NVLink, 先将数据拷贝到Node1 GPU K, 再通过网络中的轨道K发送到Node3 GPU K. 通常我们把这种方式成为PXN.

当然这种方式下一个交换机能够连接的GPU是有限的, 例如总共只有512个/1024个端口, 因此对于数十万卡集群, 又逐渐扩展为Multi-Plane的方式. 例如一个800Gbps的Ethernet端口, 实际传输是采用8个112G Serdes传输的, 那么我们能否将其划分到8个平面呢?

![图片](assets/f1ef7e75feca.png)

Nvidia CX8就在网卡中集成了一个Spectrum-X交换机的Port Logic, 然后就可以单个网卡接多个交换机了.  这样原来一个交换机有512个112G serdes, 单个800G网卡要占用8个, 因此整个交换机最多支持64个GPU互连. 而采用MultiPlane的方式可以通过8个交换机支持512个网卡.

![图片](assets/7e7891daae21.png)

但我们实际上来看, 数据包在CX8的网卡分散到8个发出后, 经过的路径并不相同, 中间每个交换机的buffer用量不同, 到达目的地时, 必定会出现乱序的到达的问题. 同时如何保证8个平面负载均衡, 同样也需要一些特殊的处理. 本质上还是需要协议来支撑乱序的能力. 同时还需要启用Adaptive-Routing的能力. 保证数据包在多个路径上分发.

另一方面的因素是, 大EP并行的兴起, 特别是在推理场景中, 使用PXN也会带来显著的延迟增加, 因此能够直接通过RDMA网卡到不同的Rank的GPU变得非常重要, 例如DeepSeek自身是使用了Infiniband的Adaptive-Routing技术.  同时对于AFD这样的方式, 异构的卡通常放置在不同的集群, 也需要跨越多层交换机网络, 并需要显著的降低网络传输的延迟.**注意:这里的延迟是指实际工作时的E2E传输延迟, 不能简单的以空载情况下网卡转发1Bytes报文的延迟来计算, 更多的是需要降低传输时在网卡和网络队列中的排队延迟**

结论: 无论如何RoCE还是需要支持乱序的处理能力. 并支持交换机多跳转发时的多路径负载均衡.

**其实对于做计算的人而言, 乱序多发射的机制已经非常成熟了, 但网络这一块确实在前些年有很多缺失.** 在RoCE的报文格式中, 因为传输层要求严格保序, 报文设计上相对简单.就是一个简单的First/Middle/Last.

![图片](assets/873f70734d3c.png)

中间的报文并不携带操作远端内存的地址. 因此乱序后将无法知道远端该如何写入.  当然Nvidia也做了一个简单的处理, 是否能够通过拆分成多个独立的WRITE报文, 每个都携带地址, 这样就可以乱序发送了呢? 但是对于一个消息而言, 我们还需要通知对方是否完成了, 因此发送端还需要等前面独立拆分的WRITE报文都确认接收后, 再发送一个WRITE_With_Imm消息确认. 或者通过一个ATOMIC消息更新接受端的Fence flag, 这样实际上还是增加了一个Round-Trip-time. 这种做法对于WRITE可以实现, 但是对于SEND/RECV是有缺陷的.因为RECV端的buffer缓冲区的排布并不是绝对的物理地址. 所以在CX7中, Out-of-order的处理只支持WRITE语义.

当然网络这个圈子也有很懂这一块的人, 例如2002年发布的iWARP, 其中就提到了Direct Data Placement(DDP)的技术.

![图片](assets/610efd4b1f87.png)

DDP在每个数据分段中间添加了一个Message Sequence Number(MSN)和Message Offset(MO), 通过MSN和MO其实就非常容易的在多种语义上支持乱序接收, 并且可以在接受端判断消息是否完成, 然后完成Imm提交或者执行Fence flag更新, 这样相对于RoCE现有的实现还节省了一个Round-Trip-Time.

#### 1.4.2 拥塞控制

下面再来看一下拥塞控制和丢包重传, 在传统的HPC应用中, 数据流的burst-size并不大, 而在新兴的AI应用中数据流持续的时间明显更长. 当我们采用Lossless时, 很大程度上需要交换机和网卡更深的buffer去控制, 但事实上这样也会导致E2E的传输延迟有显著增加.  当我们谈论延迟的时候, 很大程度上我们把空载延迟(静态延迟)和实际的延迟(动态延迟)混淆了. 谈论低延迟这个话题时, 需要考虑的是应用负荷高的时候延迟

![图片](assets/6d000f566684.png)

在传统HPC应用中, 消息并不大, 因此在整个网络中排队的延迟相对较低, 更加看重于静态延迟(Statci Latency). 但是针对AI应用, 高吞吐的时候, Lossless和相对不完善的拥塞控制会引入更高的队列延迟.

![图片](assets/67e0ea8026df.png)

特别的来说, 基于Rate-Based的拥塞控制实际上并不能够很好的控制在网络中的报文数量.

在传统的视角下, Window-Based-CC控制了报文inflight的数量, 而Rate-Based控制的是报文的发送速率. 例如对一些Video Streaming这样的应用, 其发送速率是相对稳定的, 采用Rate-based无可厚非. 但是对于新兴的AI应用, 交换机的buffer演进会越来越慢, 因为越多的端口读写, 构建多端口的SRAM同时演进的时候又要维持相应的BDP(Bandwidth X Delay)容量是非常困难的. 而Rate-based如果需要精确的去控制适配网络中的各种情况, 那么势必需要进行更加复杂的控制和更加准确高效的检测和适配网络的拥塞状况.

正如Nvidia在这次HotChip上有一个图:

![图片](assets/f0de002a686d.png)

左边是传统的RNIC和交换机, Ingress-buffer的量非常高. 实际上的原因是什么呢? 其实很大程度上这些消息的堆积就是拥塞控制不当导致的. 而后面Spectrum-X作出的改进其实就是增加更频繁的Telemtry反馈. 并通过PSA处理这些Telemetry获得更加实时的控制能力.

但是为什么不直接用Window-Based拥塞控制替代呢? 标准的RoCE协议实际上是无法支持的. RoCE为什么不能够支持Window-Based拥塞控制主要有两点协议上的缺陷

数据包和ack报文单独发送,数据包里面没有携带ack信息

Read resp报文是用ack报文封装,实质是数据包,而read resp并没有对应的ack;

如果采用window based拥塞控制,需要ack来驱动,read场景就会导致read resp直接发不出去了,解法只能变成使用定时器加token,又变成rate based了.

结论: 在整个RDMA向超大带宽演进, 和要求Fabric利用率在集合通信场景下完全利用网络带宽时, Rate-based-CC需要更加敏捷的遥测数据(Telemetry)来获取更加实时的网络状态并更好的适配网络的拥塞状况. 这样对于网卡的实现复杂度会进一步提高, 并且拥塞控制调整和配置也会变得更加复杂, 这种情况下选择Window-Based-CC是更加遍适的处理. 所以你会看到AMD STRack/UEC和Google/Intel Falcon都选择了采用Window Based拥塞控制.

#### 1.4.3 SACK和丢包重传

最后是关于丢包重传, 其实在一个带宽没有收敛比, 并且采用Window-Based CC控制的数据中心内, 丢包的概率其实并不会太高. 当然在大量的QP情况下采用Rate-based-CC也需要考虑丢包快速恢复的能力. 另一方面现阶段集群的演进已经逐渐要跨越多个数据中心大楼, 甚至因为一些电力因素需要跨越几十公里的不同可用区. 因此配置一套完善的重传机制是非常重要的.

另一方面, 针对多路径转发后的乱序到达, 如何有效的检测是乱序还是真实发生了丢包也是一个亟待解决的问题. 在这种情况下, 通常使用SACK的方式响应发送端是一个非常有效的处理方式. SACK并不只是用于快速重传, 而是尽快的释放发送端的buffer.

Google在今年Sigcomm的论文《Falcon: A Reliable, Low Latency Hardware Transport》[5]中也谈到, 传统的RDMA Go-back-N/Select-Repeat以及在使用Adpative Routing带来乱序时, 丢包产生的影响:

![图片](assets/09f9488e1cb2.png)

而基于SACK/RACK-TLP的Lossy传输的RDMA, Google Falcon和我们CIPU eRDMA都能够做到5%丢包率下还有90%的Goodput,也就是说完全有效的打满了整个带宽.

结论:  在整个RDMA向超大带宽演进时, 采用SACK处理多路径转发的报文响应, 同时配合RACK-TLP机制一起实现快速丢包恢复是关键.

#### 1.4.4 Google Falcon的分析

Google在Falcon的论文中也指出了这些问题:

First, RoCE does not support modern loss recovery. RoCE initially relied on Go-Back-N style loss recovery, even for single packet losses. Proprietary extensions now enable Selective Repeat (SR),but with significant restrictions. We find that SR is supported for RDMA Writes and Read Responses, but other operations remain limited to Go-Back-N recovery. The SR mechanism, which sends a Negative Acknowledgment for each out-of-order packet, can lead to slow, imprecise recovery and high tail latency (see §6.1.1). And when using SR, loss typically leads to out-of-order packet delivery,which violates IB Verbs semantics.

即RoCE不支持现代化的丢包恢复机制, 依赖于Go-back-N或者是扩展的Selective-Reapt机制存在很多限制, 例如SR只支持WRITE和READ Response, SR会对乱序到达的报文发送NAK, 这样会导致在多路径转发乱序到达时处理产生困难.

Adding state-of-the-art loss recovery like Selective Acknowledgments (SACK) to RoCE is non-trivial while also satisfying IB Verbs ordering semantics. RoCE NICs lack resources like packet buffers. To maintain ordering, a RoCE receiver drops out-of-order packets following a loss. This limits its ability to precisely signal missing packets to the sender. To implement precise loss signaling like SACK, RoCE NICs would require substantial changes, such as adding on-NIC packet buffers on the receive side or modifying the ordering semantics. Thus, for many use cases, Priority Flow Control remains necessary to avoid losses.

另一方面是SACK要引入RoCE还是很困难的, RNIC为了维持顺序, RoCE 接收端在发生丢包后会丢弃后续乱序到达的数据包, 这限制了其向发送端精确报告缺失数据包的能力. 要实现类似 SACK 的精确丢包信令, RNIC需要大量修改, 例如在接收端增加大量的Packet Buffer, 或者修改其顺序语义. 因此这也是很多人包括Nvidia(Mellanox)坚持要使用Lossless的一个原因.

Second, RoCE does not have protocol support for multipathing. This causes complications when it is used with Adaptive Routing at switches to utilize all network paths (as is common in ML backend networks). We find that tolerating the natural reordering of multipath adversely impacts loss recovery. This leads to poor performance (see §6.1.1) so PFC is often used to avoid loss. Reordered packets are also delivered out-of-order, which is problematic for general-purpose use because it violates IB Verbs semantics.

这一条是在说RoCE本身无法支持多路径转发, 当 RoCE 与交换机上的自适应路由(Adaptive-Routing)结合使用以充分利用所有网络路径时, 会引发复杂的问题. 在多路径转发的场景下乱序是不可避免的, 同时实现SACK是有很大价值的, 类似于iWARP DDP的语义并不会显著的在接收端增加Buffer占用.

Third, RoCE does not integrate congestion control with the datapath. Congestion control is implemented as an add-on, relying on out-of-band probes [6] to gather congestion signals. This separation makes its congestion response sluggish.

最后是在讲RoCE并没有将拥塞控制和数据路径集成, 而是采用带外探测(Telemetry)的方式来收集拥塞信号, 这种分离导致其拥塞响应迟缓.

最终的结论是, 我们需要在RoCE上进行改造, 支持Lossy, 支持SACK/RACK-TLP这些重传机制, 支持多路径转发等, 这一点上我们需要清晰的认知到**可靠性和传输效率需要算力来置换**. CX8实质上也是在数据路径上通过PSA增加了算力, 通过更加有效的带外Probe机制来获取更细粒度的拥塞信号, 但是这也使得整个拥塞控制响应反而变慢了.

### 1.5 RDMA现代化的实质

RDMA现代化的改造实质是为了配合AI这些新的应用而产生的. 本质上的需求其实Nvidia自己也很清楚, 2年多前跟我们交流CX8的Roadmap时也非常清楚的告诉我们了路标, 当时我们告诉他们这些功能我们已经完全实现时, 他们居然不信.... 下图是两年前内部汇报的一个表格.

![图片](assets/c97062a86f07.png)

其实后续还有一些CX9/CX10的路标, Nvidia自己也应该清楚我说的都是对的, 你们不也在这么干么, 就不展开了. 事实上对于RDMA现代化的改造本质上就是如下几点:

首先, 需要`支持Out-of-order delivery, in-order completion`. 维持RDMA RC Verbs兼容的基础上直接参考iWARP的Direct Data Placement技术即可. 通过这个技术就可以实现多路径的也非常容易了. 当然历史上也出现过AWS SRD或者UEC采用的Libfabric的方式, 这些方式通过改造应用层语义来支持多路径的能力, 其实对生态的破坏还是很大的. **Google Falcon和阿里云CIPU团队是现阶段唯二能够支持RC兼容的多路径转发. 其实我们上线的时间还比Google早几年, 在Google Falcon对外发布前我们已经上线了.**

其次, 在多路径转发时带来的一些Out-of-Order, 如果继续维持Go-back-N Lossless的方式则需要接收端网卡更大的buffer去硬扛, 从网卡芯片的PPA来看会显著增加网卡芯片面积和功耗, 显然这样的做法是不合时宜的. 因此选择Lossy的做法, 通过DDP配合SACK/RACK-TLP是一个更加能够节省芯片面积和功耗的做法. 然后顺势采用Window-Based拥塞控制, 特别来说直接复用一下Google的Swift即可.

现在整个工业界基本名牌了, 那么我来多说一下一些背后的故事. 2020年我在自己做NetDAM的时候就意识到多路径转发解决hash冲突提高fabric利用率是非常关键的, 但是在NIC硬件在做一个完善的可靠传输协议栈是很复杂的, 过去在思科很多年也有过一些TCP offload的网卡用于高频交易等场景, 但是针对数据密集型应用显然是不合时宜的. 因此当时的做法也是直接通过CPU运行一个DPDK来做所有的拥塞控制. 2022年底加入到CIPU团队的时候, 非常惊奇的发现当时CIPU采用的iWARP协议天然的支持Direct Data Placement, 并且DK.Panda老师在2007年的SC07就写过关于iWARP支持多路径转发的论文《Analyzing the Impact of Supporting Out-of-Order Communication on In-order Performance with iWARP》[6]. 并且它已经完善的支持了SACK这些业务, 那么就很轻松的想到如何设计出一个多路径的拥塞控制算法即可.

而如果是在一个标准的RoCE网卡上演进, 那么系统架构势必进行很大的改动, 这也是Nvidia(Mellanox)还在继续坚持RoCE Lossless的一个原因吧, 另一个原因是在ASIC上实现一套完整的可靠传输代价是非常大的.

当时一个朴素的想法就是, 学习一下MPTCP的方式, 每个QP对应于多个subflow, 每个subflow一个window, 然后采用DWRR的方式调度. 当然这样的方式是完全错误的, 也是很多人容易掉坑里的地方, 好像真有人掉坑里了...

这种做法最大的缺陷是per-subflow state会极大的增加QP Context的用量, 使得访问内存的效率大幅度降低, 整个转发效率变低, 同时QP Scale也会变差. 特别来说当需要支撑128KQP同时并发支持多个subflow时, QPContext缓存的压力会更大. 因此这种做法是完全错误的. **实际上你需要构造一个非常巧妙的Statelss subflow的算法** 从一个很特别的角度来看,你就会发现一个很简单的动态规划算法就可以完成. 等专利公布的时候我再来详细说吧.

![图片](assets/5e966086a7ad.png)

上面这个图就是我们对于RDMA现代化改造的一些认知, 特别是针对云环境, 还需要考虑和TCP的公平共享以及云多租户环境下不同的租户各种其它流量带来的干扰, 大规模部署需要跨AZ带来的RDMA长传, 特别是带收敛比的网络下, 如何处理丢包和拥塞的问题. 另一方面是Over VPC提供多租户能力的重要性, 你可以看到CX8其实也在逐渐的通过PSA支持Overlay的能力, 到CX9和CX10会进一步完善Lossy的支持.

而另一方面, 对于Live Migration的需求, 我们CIPU eRDMA从第一天上线起就完善支持的功能, 如今Nvidia还不支持, 学术界今年的Sigcomm才有第一篇论文《Software-based Live Migration for RDMA》[7], 并且这个方案还无法做到Guest OS完全无感知的迁移.. 我一直在说网络这个行业, 学术界和工业界差距是非常巨大的, 有太多的秘密被藏在各个寡头那里.

所以提一个我们以前已经实现的功能作为RDMA现代化的目标吧, 看看工业界什么时候能追上

集合通信能够保证95%以上的Fabric利用率

丢包率5%的时候仍然能够保证90%的Goodput

无需任何交换机的高级特性, 网卡实现多路径和拥塞控制

超大规模(128K QPs)并支持所有QP开启多路径转发能力.

兼容RDMA RC Verbs, 线下RDMA应用无需修改代码即可直接运行.

Incast 128打1这样的场景, 每个QP之间的带宽差额最大100Kbps.

完全OS无感知的热迁移

完善的RDMA虚拟化支持

## 2.Ethernet base ScaleUP

### 2.1 Overview

对于Ethernet ScaleUP这个领域, 我应该算是整个领域的最早的工作者了吧? 从2020年开始构建NetDAM, 实际上的问题也是处理Memory Wal. DDR的容量够但是带宽不足, 而HBM又有容量的限制.

![图片](assets/bbb4f460065a.png)

当时(2020年)一个很朴素的目标就是需要针对AI应用构建一个数Tbps的带宽和数TBytes的容量. 所以一个很朴素的想法就是能不构建一个内存抽象层, 使用内存语义来构建一个大规模互连系统

![图片](assets/eecf82320f54.png)

你会看到现在谈论的In-network-Computing, Ethernet Based ScaleUP这些在2020年方案就完全做好了,并且在2021年春节前后就和同事david一起完全实现了. 通过这样一个统一的内存抽象层实现了内存语义的处理, 可以说这是全世界最早的一份Ethernet based ScaleUP的工作.

![图片](assets/807596d513ba.png)

差不多整个工作领先工业界也4~5年了吧?  但是它落地的困难性也有几方面的原因, 首先GPU生态上, 过去几年除了Nvidia NVLink外, 其它公司对于ScaleUP的重视并不多, 直到Hopper和GB200的出现把它推向了一个很高的位置, 而AMD至今还没有带交换机的ScaleUP方案, Intel和Microsoft一些卡直接选用了RoCE来做ScaleUP也有一些问题, 实质上是RoCE这样的Message语义对GPU并不友好.  另一方面是GPU互连的标准上一直有很多的争议, 而CXL因为Intel的一些原因进展和商用一直非常缓慢...

但至少我当时坚持内存语义, 使用以太网来做这两点的技术路线选择是正确的吧? 然后偷偷给大家说一句, 2年前BRCM还没有做Ethernet ScaleUP计划时, 我就和BRCM交换机事业部的CTO Mohan聊过Ethernet ScaleUP这事, 后面也和Tomhawk Ultra的架构师Surendra有很多沟通.

### 2.2 内存语义的重要性

其实很多人会说, 把ScaleOut的RDMA扩展到和ScaleUP相似的带宽就行了, 然后继续使用RoCE消息语义行不行? 事实上我们要从GPU本身的微架构来分析, 一方面是前述的RoCE的大量缺陷导致其协议栈同样需要现代化改造, 例如一个消息如果不拆分如何从多个ScaleUP Link负载均衡的发出去? 然后当ScaleUP需要两层以上交换机组网时, 一样会遇到Hash冲突的问题.而内存语义则可以很容易的将LD/ST通过Address interleaving的方式均衡的分散到多个Link上. 实质上也是实现了类似于RDMA现代化改造所需要的Weak Order的语义. 后面会详细展开.

另一方面是RDMA ULP带来的开销, 标准的LD/ST对于计算核来说只需要issue一条指令, 而RDMA ULP你需要构造WQE, 然后写入, 并且敲doorbell带来了很多计算核不必要的指令开销.

同时对于消息完成, 内存语义就是很简单的一个memory fence即可, GPU的计算核去Polling一个固定内存地址的flag即可. 而RDMA ULP通常还需要计算核去Polling CQ队列, 然后对CQE进行处理, 而我们需要注意到的是, GPGPU的架构并不是Cache Coherency的, 而且L1Cache的容量也是非常有限的, 因此对于CQ的处理需要多次访问HBM, 并且每次都是Cache miss的情况下, 整个处理伴随着Warp调度, 例如LD cache miss后调度到其它warp处理, 然后再切换回来, 并且整个GPU的Core通常还是单发射的架构, 因此会出现多次切换, 导致需要数个us的时间.

还需要注意到类似于IBGDA的操作, 保存数据到Local HBM, 然后写WQE本身带来的延迟, 加上Kick doorbell的延迟, 再到ScaleUP的RDMA加载WQE, 然后加载数据再发送到对端, 然后写入, 总体来看延迟是非常大的.

还有一点就是消息语义本身的一些问题, 首先你需要将数据从SMEM flush到HBM后, 才能使用消息语义去DMA到远端的HBM. 然后对一些Allreduce/Allgather的计算也涉及到大量的HBM读写, 这一部分的内存访问将极大的影响, 从而影响到其它算子的计算效率. 并且通常这类的ScaleUP还需要dedicated Core运行通信Kernel, 也很大程度上的影响了整个处理器的效率.

### 2.3 在网计算

Nvidia NVLS(Nvlink Sharp)功能有一篇论文讲述过《An in-network architecture for accelerating shared-memory multiprocessor collectives》[8],实际的原理是在Link MMU上构建一个Multicast Memory Region(MCR)

![图片](assets/d2788fcba158.png)

然后针对它的LD/ST如果是multicast region, 则在交换机上实现广播, Reduce则是Push和Pull两种方式,
![图片](assets/87057b863425.png)

![图片](assets/f1d639d73b88.png)

在交换机上增加了一个向量加法器, 位宽为144bits, 其中包含一个16bits的counter

![图片](assets/e565b7c6cc23.png)

然后通过这个counter来记录一些Reduce操作是否完成.

![图片](assets/5b3d5dd9ef3c.png)

然而在BRCM的方案中并没有选择这样的方式, 因为在早期实现中, 还是受到了一些RoCE ScaleUP的影响, 它通过简单的类比SHARP中的 Target Channel Adapter (TCA), 让GPU通过RoCE和交换机建立QP来实现In-network-Compute的能力.

有一个核心的问题是难以避免的, 在交换机上构建INCA(In-network-computing Architecture)是很困难的, 特别是对于现代这些DeepEP相关的Dispatch/Combine通信来看, 交换机根本无法完成大量的token dispatch/combine-reduce的状态保存, 也没有足够的buffer来缓存临时加总求和的结果.

我当时在设计NetDAM的in-network-compute时, 更多的是从端侧来考虑. 首先肯定还是要支持内存语义, 通过一个Vector Add指令来触发Reduce操作. 当然后面在阿里也做了很多其它的东西, 例如Dispatch上采用BIER Multicast Routing来分发token已经在2023年就有了专利了. 例如构造一个组播BitIndex

![图片](assets/2fd8788e05c5.png)

然后allgather可以映射成如下方式:

![图片](assets/bfb7e9c1f5be.png)

但是实际的业务收益, 还是存在很多需要探讨的地方, 首先是通信和计算的Overlap(例如FSDP)使得传统训练中的AG/AR的操作很大一部分都被隐藏起来, 因此SHARP这类的收益并不会太明显.  对于推理的场景, 特别是DeepEP的场景, SHARP一类的又无法处理. 这就导致了INCA某种程度上成为一个比较鸡肋的功能.

实际上我们需要注意在网计算的核心收益并不是在于如何加速allgather或者reduce的操作时间, 而是降低内存访问对GPU GMEM以及L2 Cache的干扰, 这是实质, 点到为止吧.

### 2.4 内存语义和事务

这里就要谈到GPU本身的内存模型了, 抛开GPU的内存模型和微架构实现来讨论这个问题是毫无意义的. 其实这个问题在以前专门有一篇文章讨论

[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect)

本质上就是对ScaleUP网络需要支持Acquire-Release的语义. 另一方面Nvidia在GTC25的一个Session里面也有一个很好的总结:

![图片](assets/2926dadb9212.png)

事实上这个决策树显示了使用LDGSTS和TMA在什么情况下适用, 然后把可靠传输和内存事务耦合在一起其实还有很大的设计难度. 这部分的内容对于纯网络域的同学还是存在大量的挑战的, 事实上它和计算域的同学如何设计计算核的微架构是紧密相关的.

由于各家有各自的计算核的微架构, 因此对于如何实现内存模型, 这是一个更加开放性的问题, 对于一个好的厨子, 也需要根据食材来做恰当的处理, 此处并没有什么标准的做法, 需要计算的兄弟和网络的兄弟一起协作. 我并不认为现在网络的兄弟们提出的这些东西是完全正确的.

### 2.5 传输效率

对于一个标准的以太网, 特别是一个IP UDP packet的开销是非常大的. 对于NVLink而言, 它的一个FLIT如下所示:

![图片](assets/ea5cb0434f3e.png)

而Ethernet我们做了一些计算

![图片](assets/4879eb0f6682.png)

其实计算域的同学对于以太网的诟病也在于此, 实际上也导致的Ethernet ScaleUP的adoption比较缓慢. 当然也不是没有办法去处理, 一种是动态的packing多个指令. 如果需要packing多个LD/ST成一个较大的Message, 那么硬件上还是需要维持一个timer, 例如100ns或者200ns的一个等待时间. 或者是针对不同目的地XPU上构建队列, 然后轮询的方式发, 自然的利用这些轮询间隔的时间来构成packing. 实际上packing本身会带来一些Fabric上的抖动影响.后面一章详细来展开.

另一个是修改以太网的传输头, 提高效率. 我们来看一个以太网报文, Preamble前导帧的开销, Ethernet Header的开销, FCS和IFG的开销等都是值得优化的地方. 但是, 我们需要提出一个最基础的问题: **什么是以太网ScaleUP?** 本质上需要回答什么是以太网.

用了以太网的Serdes算不算以太网?

修改了以太网头算不算以太网?

实际上一个比较严格的定义是: 以太网Serdes作为物理层, 然后保持MAC头上关于EthType的字段, 而对于Source和Destination MAC Address可以做一些适当灵活的定义, 而针对Preamble这些不修改.

![图片](assets/38476f6a0682.png)

因此你会看到BRCM的SUE中AFH定义也是维持这个规则, 将很多信息压缩到SMAC和DMAC字段中, 同时保证Etype的处理不变

![图片](assets/0223a623a485.png)

如果放开这个严格的约束, 此时我们还有另一种选择, 利用成熟的Ethernet物理层生态, 复用它的物理层, 在MAC子层展开丰富的定义, 例如UALink. 它采用64Bytes的Transaction Layer FLIT, 然后再构成一个640Byte DataLink FLIT

![图片](assets/84b109820dea.png)

但是很抱歉你说这样的技术算以太网么? 毕竟还是用了Ethernet的PHY Layer的呀, 其实这是一个工程上的Tradeoff. 另外再加一个点, 反正UALink的Data Link FLIT都是640Byte了, 不妨碍我用一个标准的Ethernet MAC头承载, 这样从XPU出来到交换机之间的链路就是一个标准以太网了, 这种视角下它算不算Ethernet ScaleUP?

![图片](assets/a9c8e0a11cd9.png)

当然很多网络域的兄弟对UALink的诟病还来自于UALink需要在交换机内拆分DL Flit并进行查表进行多个TL FLit转发. 传统的观念认为交换机无法在51.2Tbps/102.4Tpbs这样高的带宽下维持线速转发. 事实上这方面的约束来自于交换机本身的微架构上, 如果采用Shared Buffer Switch, TM的MMU设计要满足到51.2T/102.4T这样的速率打满LineRate的PPS是非常困难的. 实际上采用PortBased Buffer的设计, 在交换机上为每个UALink构建一个小的Tile based PortLogic.

其实BRCM的Tomhawk Ultra就证明了这一点是可行的, 毕竟51.2Tbps 64B linerate是可以达到的.

![图片](assets/a77780bd7f52.png)

延迟也因为这个微架构及一些非常聪明的处理, 降低到了250ns

![图片](assets/a3f50f7f07f3.png)

另一方面还有一个争议点, 考虑到常见的大模型相关的跨卡Workload, 例如DeepEP这样的EP并行, dispatch的单位为一个Token, 那么传输的size为`<数据位宽> * hidden_dim`, 以DeepSeek-V3为例, 采用FP8 dim为7168, 那么传输的size为7168Bytes. 而考虑到分布式Gemm, 通常也会按照Tile切分, 例如一个32x32 FP8的Tile实际上大小也有1KB.

似乎天然的觉得, 我们可以以这样的尺度的去做一个传输? 好像以太网的效率也不是什么大问题?  然而我们还是需要看这些在GPU内部的SMEM上的分布是如何的, 毕竟ScaleUP的目的是需要将数据直接从SMEM发送到远端的GMEM上.

其实我个人的观点是, 在这一部分计算的同学和网络的同学还是要做一些妥协的.

### 2.6 ScaleUP域的规模

其实国内很多算力的焦虑, 简单的想把ScaleUP域拓展到数千颗GPU, 我个人的认知是这条路可能是错误的, 首先从模型本身的并行策略来看, 并不一定需要这么大的规模. 当然你会举出一个反例, TPU为什么做到了9216卡的规模, 但实际上它用来做真正运行某个算子的最多规模也就512卡

这一点上BRCM定义的SUE也是10bits的地址支持1024卡, UALink也是类似. 其实BRCM交换机事业部的GM Ram也讲的很清楚, 用一层的交换机把这些XPU连起来, 然后做点简单的拥塞控制(CBFC)就够了, 非常简单的一件事情.

个人觉得国内针对芯片算力受到工艺的限制, 只能做到NV的1/5甚至1/10, 然后简单的推导出来需要数倍规模的ScaleUP这个结论是有问题的. 大规模的组网后,通信的延迟是不可避免的, 而在算力芯片本身的微架构上SRAM和算力核的配比上还有很多的Trade-off, 集合通信的时间不一定能够被很好的overlap, 后面一个小节我们在做一些量化的理论推导.

### 2.7 从XPU的微架构谈谈延迟影响

对于很多DataFlow架构的NPU而言, 它们的流水线排布是非常紧凑的, 因此这些XPU对于延迟的容忍度会小很多. 但是即便是对于一些GPGPU而言, 虽然有很明确的Cache层次化结构, 如下图所示:

![图片](assets/f66f0e9cdfaa.png)

然后Warp调度隐藏了访存延迟. 但是考虑这样一个视角, 从GPGPU的微架构来看, 关于现代NVGPU的微架构以前做过一个分析

[《现代NVidia GPU架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494136&idx=1&sn=958bc734efd43a59b03c04a2e65ec408&scene=21#wechat_redirect)

当CUDA Core issue一个STS指令将数据从SMEM拷贝到GMEM时, 特别是某些场景下, 需要同时拷贝到Local GMEM和另一个Remote GMEM时, 从CUDA Core的角度来看, 我们并不知道哪个是Remote哪个是Local, 当Remote Memory的延迟显著高于Local时, outstanding的window很有可能被占满. 当然你会说可以先Write-back到L2Cache, 然后再写到对端去. 那么我们再考虑同样的场景下LDGSTS, 如果CUDA Core同时从Local GMEM以及Remote GMEM Load时, outstanding的窗口也会被占满.

另一方面如果延迟非常大, 其实中间Cache的数据还无法释放, 或者LD的outstanding buffer需要allocate很多等待LD完成, 这些都会显著的影响到SM的设计. 显然在SM内部我们不可能放置更大的SMEM或者L1D Cache, 对于整个芯片我们也需要约束L2 Cache的大小. 实际上就构成了一个Trade-off. 在一颗芯片差不多800mm^2下, 放多少SRAM留多少给TensorCore留多少给CUDA Core的问题. 这是一个很难trade-off的事情. 例如Blackwell为了TensorCore的性能, 增加了TMEM, 同时单芯片内的CUDA Core数量减少了, 虽然理论上按照TC的性能, FLOPS增加了很多, 但在很多实际的workload上, 我们却发现了一些短板, 甚至很多时候不如H200. 看上去H200倒是一个各方面都很平衡的配置.

然后你会继续说, 要不我们使用TMA来做一些异步的操作呢? 实际上也是同样的道理, 延迟大了, 如果同时要做一些可靠重传, 还会进一步的增加buffer的使用. 然后在Die内这些scaleUP的IP还会进一步抢占芯片面积.

### 2.8 是否需要可靠传输

这也是一个非常有趣的话题, 在BRCM最新的SUE Spec中添加了非可靠传输的SUE-Lite方案

![图片](assets/234c76aff763.png)

正如Spec讲的那样, SUE-Lite IP占用的芯片面积会小一半. 知乎上也有一篇文章讨论这事《scale-up LLR+CBFC 是否可以保证永不丢包，不需要端到端重传？》[9]

首先SUE对于可靠传输的描述定义如下, 即便是LLR的情况下, 但也有可能丢包, 例如SDC(Silent Data Corruption)导致CRC校验失败而产生的丢包

SUE provides in-order delivery on each plane. Use of lossless traffic classes and LLR reduce the likelihood of packet loss dramatically, however it is still possible. If SUE determines a packet loss event occurred, it uses Go-Back-N to recover.

但是E2E的Go-Back-N真的对么? 需要注意的一点是在SUE-Lite下并没有提供AXI接口, 而只有一个signaling interface.  另一方面, 在UALink Spec的section 6.6也提到了Link-Level-Replay的机制, 并没有E2E的Go-back-N机制.

在SUE的实现过程中, 需要考虑几个因素. 首先发送端需要多个队列Buffer放置发送到不同Dst XPU的数据, 队列Buffer管理本身会比较复杂, 而且队列管理上也很难有效用完所有的buffer

![图片](assets/ff0647a9f3ca.png)

而UALink 则是在XPU侧直接将TL FLIT放置到TX Buffer中并打包成DL FLIT发送, 只需要维持Link-Level Replay的buffer深度, 通常按照Link-Level延迟看512KB就够了.

E2E Go-back-N通常还需要维持超过1个BDP的带宽, 经验上讲为1.5~2BDP. 那么假设E2E的RTT延迟为2us, ScaleUP带宽为1TB/s那么也需要差不多2MB以上的Buffer, 然后还要spray到多个ScaleUP link上穿越多个交换机转发(MultiPlane), 同时要做到负载均衡没有长尾的影响, 设想这样一个问题, 中间某个交换机由于出现一些SDC(Silent Data Corruption)导致CRC校验失败而产生的丢包, 那么E2E的Go-back-N重传是否会导致一些Memory Order的问题呢? BRCM这事上只是简单的说了有一个Optional的Load Balance

![图片](assets/88b5c157e26c.png)

最后特别的来说, 对于现在比较激进的光互连的ScaleUP而言, 还是有很多问题需要去处理的.

## 3. ScaleUP的一些量化分析

在以前一篇文章中的Section 4.2也引用了Nvidia在GTC2025的一个session

[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect)

首先从Little's Law的角度来看,  在一个平稳系统中, 客户的长期平均数量L等于长期的平均有效到达率乘以平均客户在系统中花费的时间, 一个非常直观的公式.这就是Little's Law. 例如在一个扶梯上, 平均2s到达一个顾客(平均1s到达1/2个客户), 然后坐扶梯需要40s, 则系统可以承受的并发数为20个人

![图片](assets/3d6ed2c9d047.png)

内存访问也是相似的,  我们可以根据内存的带宽和平均访问内存的延迟, 测算出Inflight-Bytes, 可以看到对于Hopper需要32KB的inflight才能把内存带宽打满, 而Blackwell差不多要再翻倍到64KB.

![图片](assets/e6fab74600d6.png)

然后非常简单的一个做法是增加指令并行和数据并行, 但这样会极大的增加寄存器的压力. 当SM通过ScaleUP写入到远端HBM时, 延迟的影响也是非常显著的. 为了打满SMEM可能还需要增加到更大容量, 那么在整个芯片的PPA tradeoff上是有些得不偿失的.

另一个值得考虑的问题是使用Kingman‘s公式来评估, Kingman’s公式在网络中通常通过估计平均等待时间来估计buffer的使用量. 对于一个通用的单服务器队列模型 G/G/1, 其平均队列延迟为:

其中为到达速率, 为服务速率,即平均服务时间的倒数, 为系统的使用率. 为到达时间的变异系数,为服务时间的变异系数. 变异系数CV为标准差除以均值. 然后我们可以根据然后取不同的变异系数做出一个图来.

![图片](assets/3c30facd0296.png)

所以整个Fabric的利用率在某个延迟约束下,可以看到变异系数CV越低时, 利用率越高. 对于DataLink层的报文长度, UALink是固定的640B FLIT, 而SUE很大程度上还要受到不同Dest XPU的队列影响以及变长的size和队列调度的影响. 同时写入对端HBM时, 还要受到对端的HBM和NOC的影响, 因为在通信和计算Overlap的过程中, 对端的算子通常需要打满HBM的带宽. 同时发送端的PE也会产生非常大burst的流量, 然后Link-layer的retry或者信号完整性的问题也会产生各种各样的抖动. 还有Fabric上也会产生很大的incast拥塞, 从而又影响了CV.

## 4. 谈谈ScaleUP和ScaleOut融合的方案

其实在2020年做NetDAM的时候就分析的比较清楚了. Intra-host/NOC和Inter-host的互连总线有着显著的差异. 直接把任何一种技术渗透进入都会带来问题. **以太网带宽大价格又便宜的优势是建立在尽力而为的服务基础上的, 同时伴随着大量的规模效应**. 那么可靠传输本身又会占用大量的Die面积, 直接集成或者合封到XPU上都会带来不少的问题.

其实有一个看上去很朴素的想法, 既然ScaleUP都用了Ethernet直接连到ScaleOut做融合是否就好了呢? 例如ScaleUP做SUE, 然后再进一步ScaleOut上做SUE over IP/UDP? 那么可靠性的问题在端侧I/O Die上解决成本是巨大的, 正如前面第一章所分析的, 在一个巨大的ScaleOut网络中还是需要使用Lossy, 特别是在跨越多跳交换机时多路径负载均衡带来的乱序情况下. 此时I/O Die面积的开销会非常的大, 例如一个完备的可靠传输的RoCE IP无论是CX8/AMD/Falcon都在用算力去置换芯片面积做可靠传输, 目测一个800GB/s的ScaleUP这些die面积的消耗可能要接近到300~400mm^2, 那还做个啥算力了呢?

其实我个人是比较反对ScaleUP和ScaleOut的定义的, 可能更好的定义是Local Network或者In-rack network, 然后再扩展到inter-rack network. In-rack内其实有足够的可靠性保障的, 并且本身也能维持大概1us的RTT, 然后在整个机柜内堆64卡~128卡也是可行的, 散热这些都能做到. 为了芯片面积的占用的考虑, 简单的做个Lossless也无妨.  而inter-rack很大程度上是需要一套完善的可靠传输协议的, 同时本来光纤跨机柜加上一些延迟上的约束, RTT延迟可能会到5us左右.

那么在两个网络之间, 加一块Memory构成一个大坝不就行了么? 正如几年前在NetDAM论文中讲的:

![图片](assets/bd2f9269e003.png)

Therefore, "DAM" is required as the barrier of host to divide the intra-host and inter-host I/O into 2 different segmentations.NetDAM is designed to bridge the intra-host and interhost protocols by directly sharing memory with additional instruction level support for in-memory and in-network computing. With this architecture, CPU or Domain Specific Accelerator or other storage component could directly attach to NetDAM via AXI or CHI or PCIe/CXL bus and share the unified memory pool.

逻辑很简单, 即然可靠传输需要额外的Buffer和算力, 那么能不能把这部分拉远呢? 因为本来Inter-Rack的通信就要5us左右的延迟了, 那么我先在In-rack的地方构建一个I/O Memory Pool作为临时缓冲区, 不就行了? 其实在内网中也有一篇2年前写的很长的文章《云基础设施演进的一些思考》从模型到计算/存储/网络通盘考虑后, 也得出了同样的结论.

![图片](assets/f1035eba3432.png)

首先PE要ST的数据先放到NetDAM的memory里, 然后很简单的encode一个instruction让它转发, 到Inter-Rack的I/O Memory, 然后远端的PE再LD即可, 整体E2E InterRack的延迟增加不了多少.对于PE来看, LD/ST是操作Intra-Rack内部的内存, 有确定性延迟的.

当然在NetDAM上的外挂DDR是可以选配的. 选配这些DDR还有额外的好处, 例如KVCache的Offload, 甚至这个节点还可以通过Inter-Rack网络去存储集群Prefetch KVCache. 甚至是一些embedding table

当然另一方面来看, 实际上你可以把它也当一个Parameter Server使用或者一个通信Kernel Offload的部件, 例如在EP并行Dispatch时, PE可以只发一份token给NetDAM, 让后让NetDAM自己去Dispatch, Combine阶段因为NetDAM当年也做了在网计算的实现, 很容易的可以做到reduction.

其实这样的想法在这次HotChip中, Marvell和Celestial AI都有相关的探讨

![图片](assets/598331113cf0.png)

![图片](assets/61c84fb24604.png)

实际上它和Meta做的NVL72的拓扑也是类似的, 只不过是这个地方使用通用CPU做这些事情PPA上并不占优.

![图片](assets/6bfd4be8139c.png)

其实未来在HBM4的base logical die上也不是不能做一些类似的事情....

其实个人觉得, AMD UALink和BRCM之间别再争吵了, 都各自让一步, UALink的DL FLIT采用标准的以太网MAC层, 因为非标准的传输又会落到BRCM PCIe Switch的Group,  然后BRCM Tomhawk Ultra也针对UALink转发做一些修改. 性能延迟这些口水仗真的毫无意义, 江湖里真的不是打打杀杀, 而是拉帮结派呀..

## 5. 小结

本文分两个部分, 一部分介绍了ScaleOut RDMA的一些进展, 关于Google Falcon其实我们也做了类似的工作并且比它要早两年上线, 而Falcon走了一些弯路, 在Mt.Evans上遇到了很多问题导致了很长时间的延期. Google Falcon的论文, 我过两天再详细做一个分析.

后半部分详细探讨了Ethernet ScaleUP的一些问题, 不知道和下面这个Session的内容了有多少是重叠的, 到时候大家去听听吧.

![图片](assets/1ae90a30d59f.png)

参考资料

[1] 
Scalable Cluster Technologies for Mission-Critical Enterprise Computing: *https://www.cs.vu.nl/~ast/Theses/vogels-thesis.pdf*
[2] 
High performance networking Unet and FaRM: *http://www.cs.cornell.edu/courses/cs6410/2016fa/slides/24-networked-systems-rdma.pdf*
[3] 
RDMA [1]: A short history of remote DMA networking: *http://thinkingaboutdistributedsystems.blogspot.com/2016/12/rdma-1-short-history-of-remote-dma.html*
[4] 
Revisiting Network Support for RDMA: *https://arxiv.org/pdf/1806.08159.pdf*
[5] 
Falcon: A Reliable, Low Latency Hardware Transport: *https://dl.acm.org/doi/10.1145/3718958.3754353*
[6] 
Analyzing the Impact of Supporting Out-of-Order Communication on In-order Performance with iWARP: *https://web.cels.anl.gov/~thakur/papers/sc07-iwarp.pdf*
[7] 
Software-based Live Migration for RDMA: *https://dl.acm.org/doi/pdf/10.1145/3718958.3750487*
[8] 
An in-network architecture for accelerating shared-memory multiprocessor collectives: *https://dl.acm.org/doi/10.1109/ISCA45697.2020.00085*
[9] 
scale-up LLR+CBFC 是否可以保证永不丢包，不需要端到端重传？: *https://zhuanlan.zhihu.com/p/1944488159380500554*