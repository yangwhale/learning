# Hotchip-2025: Day1网络篇

> 作者: zartbot  
> 日期: 2025年8月26日 00:24  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495152&idx=1&sn=75689c0d37784db1d2849465807a8187&chksm=f995fd32cee274242ae13cf3702221c422bab18d222a52615b4cd442cc3cf3f66319857409fd#rd

---

### TL;DR

这个Session主要是Nvidia CX8, AMD Pollara 400G和Intel 400G的Mt.Morgan以及BRCM的TF1也就是现在叫Tomhawk Ultra的交换机芯片. 总体来看, AMD在原来的P4架构上实现了400G NIC同时支持了《STrack: A Reliable Multipath Transport for AI/ML Clusters》[1]的多路径算法. 而Intel也是将Mt.evans升级到了400Gbps的Mt.Morgan, 然后官方公开了Google Falcon的支持. Nvidia CX8只是在原有的架构上小升级了一下增加了PSA, 最后是BRCM的Tomhawk Ultra(TU), 大概讲了几个新功能, 当然涉及它的微架构BRCM自己没讲我也当我自己不知道了, 其实TU过去两年和BRCM的架构师们做了太多的深入的交流...

## 1. Nvidia CX8

首先阐述了推理和训练workload的差别:

![图片](assets/a48a2a1903b5.png)

训练需要每个iteration的同步, Tail-latency的影响很大, 通常是单个超大规模Job的特征以及很少的和外界交互.. 而对于推理业务, KVCache和Agentic对外界有频繁的复杂的交互, 同时为了推理效率的要求通常也需要进行一些按需调度构成分布式集群的处理, 然后对于延迟也更加敏感, 但我们看到延迟需要注意区分静态延迟和动态队列延迟, 特别是在一些incast等情况下的长尾延迟.

CX8的主要一些功能增强, 一方面是多路径带来的Out-of-Order支持, 然后在数据路径上引入了PSA Packet Processor, 但是这样极致的静态延迟优势就没了. 同时片上增加了Memory System构成了Memory Rich架构来解决QP Scale的问题. 最后它还集成了一个PCIe交换机芯片, 网口侧也继承了一个简单的Spectrum-X交换机的pipeline. 然后DOCA上增加了一个Programmable RDMA(PRDMA)的能力. 网卡同时支持IB和以太网.

![图片](assets/4185f767a55b.png)

主要的一些功能增强如下所示:

![图片](assets/05048e158db9.png)

然后花了一个图说4K QP的情况下也能保证打满, 实质上就是Memory Rich架构增加的NIC Memory System带来的, QP Context原来是Cache模式, 容量小, QP多了Cache Miss需要从主机内存里面拉导致的性能下降, 而现在本地内存搞定了..

![图片](assets/41b53e73df24.png)

锐评一下: 这图的意思不就是在说CX7打不满么? 早就给这群人讲过要Memory Rich, 阿里云CIPU 1.0几年前就几十K QP能打满了...

然后就是集成了一个PCIe Switch, 终于解决了GB200在CX7上GDR的问题...

![图片](assets/a0c9bee6c5d6.png)

接下来就是它支持了8个Port Logic, 这样可以构成8个平面. Marketing term是交换机实际连接的GPU数量的Radix更大了, 2层可以构建128K GPU

![图片](assets/7c701ba2b014.png)

然后详细展开了一个示意图, 可以支持8x 112G Serdes 构建8个100G端口, 在这里实现了端侧的Packet Spray然后又把Spectrum-4的一些Telemetry功能集成进来了?

![图片](assets/4a1cd50c8e61.png)

然后是RDMA做了一些创新, Application QoS这些嘛, 老生常谈的问题了.  然后第二层Full RDMA Verbs, 很疑惑吧? 难道以前不支持? 确实是的, 老的卡由于Out-of-Order的一些设计上和协议上的缺陷不支持Send/Recv.  然后详细阐述了PSA的能力, 它也是一个多线程的架构, 支持一些Routing和Telemetry的功能, 同时值得注意的是也支持了VPC这些多租户隔离能力, 同时还支持了加密

![图片](assets/1d3c48b3683f.png)

最后还是DPA, 和CX7一样16核16线程处理Event

![图片](assets/3c1745eeb893.png)

拥塞控制和路由的处理如下图所示, Transport层的Event和路由层的Event都会有DPA处理

![图片](assets/b0ab62d751ea.png)

然后就是一些性能相关的, 例如Job性能之间的干扰可以避免:

![图片](assets/b55f09a51ffb.png)

锐评: 那么原来的干扰来自于什么地方呢? Lossless PFC这些导致交换机Buffer占用, 还有就是因为Buffer的原因, ECN带来的一些控制复杂性

然后第二个case是关于ingress buffer占用的, 实际上我们可以看到前面PSA增强了一些Telemetry的能力, 使得拥塞控制可以做的更加好一点

![图片](assets/ed133ee9bade.png)

锐评: 实话说, ingress buffer的使用本质上是Rate-based CC带来的.  其实Window-based congestion control才是正路, 后面可以看到Intel用的Falcon和AMD的STrack都支持了. 但是标准的RoCE协议是无法支持的.

数据包和ack报文单独发送,数据包里面没有携带ack信息;

Read resp报文是用ack报文封装,实质是数据包,而read resp并没有对应的ack;

window based,需要ack来驱动,read场景就会导致read resp直接发不出去了,解法只能变成使用定时器加token,又变成rate based了

然后就是多平面和Adpative-Routing解决了hash冲突的问题

![图片](assets/967a4870a145.png)

实际上, 当你需要多租户支持的时候, 在VPC中如何支持PFC和Lossless呢? 多路径需要完善的Out-of-order能力, CIPU eRDMA Day-1就支持..最终当你选择时都会走到下面这条路.

![图片](assets/88d59271f25e.png)

对于Nvidia现在的Multi-Plane的做法, 我倒是很明显的反对的, Cable会带来更大的复杂性, 更大量的光模块的用量和功耗也是需要考虑的问题..最后NV做了一个总结:

![图片](assets/d55c1e118c50.png)

前面四个都是多路径可以解决的问题, 最后一个Telemetry, 实际上等会儿你看到Google Falcon Swift-RTT明显是更好的解决方案. 最后一直在提一个目标, 目测CX8还是有好几条无法满足的...

集合通信能够保证95%以上的Fabric利用率

丢包率5%的时候仍然能够保证90%的Goodput

无需任何交换机的高级特性, 网卡实现多路径和拥塞控制

超大规模(128K QPs)并支持所有QP开启多路径转发能力.

兼容RDMA RC Verbs, 线下RDMA应用无需修改代码即可直接运行.

Incast 128打1这样的场景, 每个QP之间的带宽差额最大100Kbps.

## 2. AMD Pollara 400G

整个系统架构还是沿用以前Pensando的P4 Engine, 不过去掉了ARM子系统.

![图片](assets/83ed6d69ec87.png)

以前整理过Pensando的P4 Engine结构[《包处理的艺术(4)-低延迟智能网卡设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485370&idx=1&sn=3b5590ccf58909f2d390df00bfb5d853&scene=21#wechat_redirect)

![图片](assets/315776e5ca71.png)

它是一个多级流水线的架构, 在传统的MAU的基础上增加了一些PC和通用寄存器, 扩展了可编程能力

![图片](assets/c8f5f4807040.png)

![图片](assets/84df8b3cd67f.png)

然后有个VA到PA的地址转换能力, 其实这些功能我们很多年前在做eRDMA虚拟化的时候早就实现了..

![图片](assets/e51462703215.png)

然后Atomic这些能力也是很常见的, 我不知道为什么单独拿出来说.

![图片](assets/2c3d04f994d7.png)

哦, 不对, CX7 Atomic inflight有限制, CX8不确定这块是不是也有问题, 笑而不语...

然后AMD还增加了Pipeline上的Cache一致性..猜猜为啥呢?

![图片](assets/771fa93d9a5f.png)

然后这里也不得不扯多路径和拥塞控制, 要做到95%的利用率, 并且支持快速的丢包恢复

![图片](assets/b4d14f5037b3.png)

AMD也详细阐述了ECMP hash冲突的问题

![图片](assets/682265fe22ce.png)

我也一直在谈这事

![图片](assets/0b014b68934e.png)

实际上的核心解决方案是2002年iWARP的Direct-Data-Placement技术实现了乱序提交和保序完成的能力

![图片](assets/2ef0ed42cdf2.png)

另一方面是incast这些导致的问题

![图片](assets/5e282b177b01.png)

我也在说incast这些导致的Tail Latency...

![图片](assets/d1a90da363d2.png)

解决方案是基于RTT的Window-Based CC

![图片](assets/be81283f4016.png)

然后是Lossless无法用, 必须要硬刚Lossy

![图片](assets/33673cae8aa6.png)

结论就是SACK, RACK-TLP这些方案.

AMD在端侧构建的多路径和路径选择, 和eRDMA是一致的.

![图片](assets/b97fd912ebc1.png)

然后整个处理过程是在P4 engine中实现的

![图片](assets/b0f962010264.png)

![图片](assets/ebe7851d5d33.png)

然后拥塞控制也是典型的RTT based Window, 以及使用了SACK这些进行丢包快速恢复

![图片](assets/c551a1a0bf7d.png)

具体P4实现如下:

![图片](assets/efcaece39c3f.png)

![图片](assets/db755e250e6e.png)

丢包也实现了SACK

![图片](assets/35379f77bfd7.png)

![图片](assets/125b119bd849.png)

![图片](assets/821f614b6bdb.png)

回想起两年前Nvidia网卡的一个VP来阿里云找我们谈CX8/BF4 Roadmap记录的一个表格...如今CIPU2.0连加密都完全支持了...

![图片](assets/10ecc8c0ea26.png)

再来对比一下AMD,实际上就这么点东西, 这才是工业界的正确路径...当年一直跟你们说不信...

## 3. Intel Mt.Mogan

大概就是一个400G版本的Mt.Evans, 24个ARM N2的Core. 并包含一些加密和压缩的能力.

![图片](assets/1beca841fde5.jpg)

报文处理的Pipeline和以前也没啥变化, 固定的流水线, 比AMD差一些

![图片](assets/8e14d0f98bfb.png)

然后Inline crypto支持IPSec和PSP

![图片](assets/65eef280249d.png)

然后有一个固定的Traffic Shaper

![图片](assets/c83db5e8043b.png)

亮点是以前给Google私有定制的Falcon, 自从Google贡献给OCP了以后, 公开可用了. 说实话这玩意也是导致Mt.Evans反复折腾延误了很久的根本原因

![图片](assets/f6aa80862117.png)

然后Intel对IPU做了一些use case的描述, 但是对于云IPU的场景理解还是不够的,就不展开说了...

![图片](assets/686978e337ab.png)

## 4. BRCM Tomhawk Ultra

Ethernet ScaleUP是我一直在推动的一个项目, 从2018年在Cisco做AI Infra, 到2020年意识到需要构建一个更加高效的ScaleUP网络, 最终选择了以太网来构建, 很简单以太网交换机非常容易获得, 因此当时构建了NetDAM, 工业界第一个内存语义的Ethernet ScaleUP(2021年完成的, 领先大概4年吧), 并且它是一个ScaleUP和ScaleOut融合的架构.

![图片](assets/dc8268b61ea2.png)

在BRCM Tomhawk Ultra的设计过程以及SUE的标准过程中,也一直和BRCM进行了深度的交流. 当然有些话BRCM没说我也不说了.

首先BRCM分析了HPC和AI-ScaleUP的workload区别:

![图片](assets/d473570e091a.png)

TU主要在传统的以太网技术上增加了以下功能, 特别来说就是LLR/CBFC/SUE header优化/ INCA这几个.

![图片](assets/e36b61e5a189.png)

LLR支持链路级别的重传:

![图片](assets/235e94a50488.png)

CBFC是一个Credit based流控:

![图片](assets/f8c56545fa32.png)

然后就是某国内友商**全国首个!填补以太网GPU Scale-UP互联协议空白** 的东西,实际上全是BRCM干的SUE AFH

《谈谈以太网GPU Scale-UP的工作EthLink》

![图片](assets/4592c6ee3536.png)

![图片](assets/e3876a5adb61.png)

INCA在网计算采用了类似于SHARP的方案, GPU和交换机建立QP, 然后Aggregation Mgr执行一些控制

![图片](assets/4298c19f73e8.png)

然后TU的微架构有很多变化, 具体不能说, 但是能做到64B小包Line-Rate

![图片](assets/4d6561d544a5.jpg)

延迟也因为这个微架构及一些非常聪明的处理, 降低到了250ns

![图片](assets/6edbfeb357ce.jpg)

最后, BRCM整个产品线, 覆盖很全, 从ScaleUP到ScaleOut, 再到Nvidia说的Scale-across(Region Scale-Out)

![图片](assets/aea0f4caf120.png)

参考资料

[1] 
STrack: A Reliable Multipath Transport for AI/ML Clusters: *https://arxiv.org/html/2407.15266v2#S5*