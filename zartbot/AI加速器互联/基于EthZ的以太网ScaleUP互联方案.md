# 基于EthZ的以太网ScaleUP互联方案

> 作者: zartbot  
> 日期: 2024年8月16日 12:17  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491597&idx=1&sn=a76b855416fe3ae614c6f4ebd5ea7bbc&chksm=f995f0cfcee279d90372defc9243ff35e46289823f8f49e48d493311427ac89782f57c22ea1c#rd

---

好像工业界挺喜欢Eth+"字母/符号"的方式来扩展以太网, 也不知道这些人802.3那几千页的标准认真读过没? 反正我是读完了的. 索性我也来搞个东西叫EthZ, 简称`以太渣`,或者按照某H姓数通厂商对某Z姓厂商的评价, 也叫Eth26, 俗称`二流以太网`, 取个贱名好养活.

**当然本文最精彩的部分在第三章和第四章, 请各位读者耐心看完.**

### 1. 工业界对修改以太网的冲动

主要来源是HPE Cray Slingshot, 然后在UEC中也有Packet Rate Improvement(PRI)的工作组. 主要理由是当一些HPC和AI应用需要承载内存语义时, 从RoCEv2的视角来看, 一个报文需要Eth Header(14B) + IP Header(20B) + UDP(8B)= 42B. 还不算BTH(12B)/RETH(16B)/AETH(4B)的开销. 然后还有以太网CRC 4B, 前导帧8B, IFG 12B带来的开销.

而另一方面NVLink的Header只有16B, PCIe TLP+Seq 18B,如下所示

![图片](assets/78f35714a40f.png)

对于GPU访问而言, CacheLine为128B, 但可以通过`cudaLimitMaxL2FetchGranularity`调整.

![图片](assets/8d01c24c8e34.png)

因此相对于NVLINK/PCIe这些总线, 以太网承载LD/ST效率极低. 这是工业界有一部分人想修改Ethernet的冲动来源.

### 2. EthernetZ协议规范

针对协议设计, 本质上是要在原有以太网的基础上构建一个2层的支持多路径的路由能力, 我们从使用场景上来划分:

`ScaleUP`: 通常组网规模并不是很大,  因此通信跳数很少,  同时内存语义需要header越小越好, 这样传输效率高.

`ScaleOut`:主要是更灵活的组网, 地址空间/路由以及协议融合,多租户隔离等多个header stacking, 拥塞控制和可靠传输等.当然还有一些GDS一类的, KVCache相关的, 和外部存储等器件互联互通的需求.

然后还有一个共同点, 消息头都要有足够的信息熵来提供多路径转发的能力. 根据这些需求, 那么协议里面我们大概就要定下几个必须要有的字段:

字段作用FlowLabel解决多路径信息熵的问题HopLimitTTL,防止环路等情况NextHeader提供Header Stacking的能力Payload Length报文载荷长度Traffic Class传输优先级Address需要不同的域进行灵活的编址

最核心的工作就是如何通过编码把IP/UDP的信息压缩到一个更小的Header中, 那么这一切就对上了渣在2021年写的一篇 [《IPv6- : 基于IPv5的48bits寻址互联网协议》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485564&idx=1&sn=0e40eebc00311795c4de65909a2ec220&chksm=f99618becee191a8b8e1579062e95da872737d70d73954260b74ddacf9bbb600d064e072cb34&scene=21#wechat_redirect) 相关的定义

地址类型长度48bits 倍率以太网MAC48bits1MPLS Label24bits0.5VXLAN VNID24bits0.5IPv4 + TCP/UDP端口32bits + 16bits= 48bits1IPv6 + TCP/UDP端口128bits + 16bits= 144bits3

通过这种方式,我们可以灵活的选择地址空间构造ScaleUP和ScaleOut的访问特性. 整个消息头长度为20B,并且可以完整的嵌入IPv4 UDP相关的信息, 相对于NVLink的16B只扩大了4B.

```
 0                   1                   2                   3 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|Version| Traffic Class |               Flow label              |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|         Payload Length        |  Next Header  |   Hop Limit   |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|                         Source Address                        |+                               +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|                               |                               |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               +|                      Destination Address                      |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

当然为了对标NVLink, 我们还可以进一步进行压缩,例如在Next Header中构造一个compact address family, 采用源目的各16bits, 或者只有目的address的方式来构建. 例如下图只需要12B

```
 0                   1                   2                   3 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|Version| Traffic Class |               Flow label              |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|         Payload Length        |  Next Header  |   Hop Limit   |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|         Source Address        |      Destination Address      |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### 3. 真实情况, 因为前面都是扯淡

如果您对前面谈论的话题表示认同, 很抱歉你被这个愚人的把戏给骗了.当时在思科每年都有愚人节专利活动, 就和IPv9一样属于纯粹的整蛊越能骗到专家越好. `IPv6-: 基于IPv5的48bits寻址互联网协议`本来就是一个愚人节的玩笑发布于2021年4月1日针对的是IPv6+, 只是最近有些人搞一些所谓的创新还对这深表认同的时候, 渣再拿出来调侃一番.

#### 3.1 定义协议是一个很容易的事情

定义一个协议是非常容易的一件事情, 只需要protocol[1]随便输入一行脚本即可, 如下所示.甚至找ChatGPT忽悠一个RFC出来也花不了多大的功夫.

```
#protocol "Version:4,Traffic Class:8,Flow label:20,Payload Length:16,Next Header:8,Hop Limit:8,Source Address:16,Destination Address:16" 0                   1                   2                   3 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|Version| Traffic Class |               Flow label              |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|         Payload Length        |  Next Header  |   Hop Limit   |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+|         Source Address        |      Destination Address      |+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### 3.2 想清楚一个协议是一个很难的事情

`以太网做ScaleUP的根本原因是: 快和便宜. ` 带宽演进上相对于PCIe快一些,同时规模效应带来的其价格和可获得性会好很多. 但是它也有很多缺陷, 例如前面的效率相关的问题. 同时上层要么CPU来控制通信, 要么GDR-Async 通过GPU Cuda Core去Launch一些Kernel来填WQE做通信控制, 以及一些QP Context, 这些都是以太网做ScaleUP的难题.

要想薅以太网的羊毛,就要接受它的不完美. 上面讲的那些不伦不类的改造, 都是属于屎上雕花, 毫无意义

### 4. 正确之路

#### 4.1 应用分析: 是否需要细粒度LD/ST

首先我们要分析, 细粒度的LD/ST主要是什么应用, 什么场景?

首先是在HPC场景中, 主要是大量的偏微分方程数值解相关的问题, 相关的详细内容可以参考如下两篇, 例如采用Jacobi迭代求解, 或者多重网格法求解等.

[《科学智能AI4S-1:偏微分方程数值解》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489355&idx=1&sn=a589f864f29a6638dc3d4db99ea12cbb&chksm=f9960789cee18e9f317683a85fe7f1d5c6390e124732dcb31b10be9d9406cc5ec770a64d180e&scene=21#wechat_redirect)

[《科学智能AI4S-2: 变分法和有限元方法》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489445&idx=1&sn=f2c08311757691474e55b2dcd7dd69f2&chksm=f9960767cee18e71271b34e4021f6a298d74a08fbe7d74761fe874eba285c28b40744a76493b&scene=21#wechat_redirect)

这类应用本质上在网格边界的通信负载来看, 更多的是低延迟的需求而不是高带宽的需求. 然后另一方面是医疗CT影像这一块, Model Based Iterative Reconstruction(MBIR)有大量的细粒度的all-to-all通信. 还有就是一些图神经网络相关的算法会有这些细粒度小size的需求.

针对当下的LLM大模型而言, 由于计算范式已经变成了大规模的张量矩阵运算, 并需要配合TensorCore运算, 可以参考渣以前写过的文章:

[《Tensor-003 TensorCore架构》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491424&idx=1&sn=0fc2110931b27714900e78d73b11a5b5&chksm=f9960fa2cee186b4d569cebcca2a4bbda37923bc404fd079010085e2d80faf97b290503859b6&scene=21#wechat_redirect)

[《Tensor-004 TensorCore编程及优化》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491529&idx=1&sn=12902726d6d9a8f9d66405ac6ea42fa7&chksm=f9960f0bcee1861d921cd1fe9bc6ae92d08682243857b310eeced42b2d6ccb665eac5966c250&scene=21#wechat_redirect)

为了满足TensorCore的计算密度, 整个计算过程已经变成了完全异步的数据拷贝.

**结论**: 如果设计一个系统需要同时兼容一些老的应用, 同时还需要兼顾HPC中的一些应用, 以及图神经网络相关的一些稀疏计算任务, 这些是需要细粒度的LD/ST的.  而对于LLM一类的应用, 异步的bulk DMA是可以接受的.

但是我们需要回答的一个问题是这些应用能否在极细粒度时打满整个网络的带宽? 除了改小传输头开销,还有没有别的办法?

#### 4.2 从GPU实现: 是否需要LD/ST

虽然Warp Specialization在GPU上优化内存访问的事情逐渐成为主流, 并且仔细翻了一下《CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization》[2] 都是2011年的论文了.

这里需要考虑的问题: Ethernet ScaleUP是不是要用RDMA?  显然如果使用RDMA, 要么主控的CPU来做控制,大量控制信令穿越PCIe, 要么就走到了当前GPU Direct-Async的路上, 让Cuda Core去填写WQE是一个非常低效的实现. 因此直接使用RoCE的IP来构建Ethernet ScaleUP这个答案是错误的.

我也一直是这个观点, 我需要连接到Memory总线上, 但是不需要LD/ST穿越整个交换网.

#### 4.3 小消息优化

至于前面说的那些屎上雕花的降低报文开销的优化毫无意义. 而本质上如果真要压缩Header也可以在不动以太网头的基础上进行扩展,通常有几种方式, 主要是**避免交换机MAC/PHY的修改, 最大化的实现通用可获得的能力**

使用MPLS标签构建转发等价类

利用原有的48bits MAC地址编码, 同时根据Ethernet Type做一些处理.

另一种方式就是在224G Serdes上, 这一点修改和实际的链路带宽以及一些老的应用根本就不需要在意这一点点延迟和效率的优化, 延迟增加多少? 带宽真能跑满么?

当然还有一种是我在做NetDAM的时候提到的内存操作要满足结合律.
及英伟达2023年《FinePack: Transparently Improving the Efficiency of Fine-Grained Transfers in Multi-GPU Systems》论文提出的方法. Pack多个LD/ST到一个package里面

![图片](assets/2acaf54d0369.png)

这样在GPU上的开销并不太大, 效率也可以极大的提升

![图片](assets/03fcfb7d73b0.png)

#### 4.4 可靠性优化

实际上以太网在大规模组网上比内存总线有优势的, 可靠性更好. PCIe的BDF在组网拓扑上有难度, 而在CXL上出现了PBR(Port-Based-Routing),但部署规模都相对较小, 可靠性的问题上Lossy实现也相对难一些. 但是在网计算和Lossy重传的问题又要对于语义做到加法幂等的操作, 留给你们自己去想吧. 当然还有一些可靠性工程, 例如Link-Level-Retry这些东西, 有意义么?可能有一些,但没有太大的意义. 当然它也需要在以太网的前导帧里面做一些事情, 然后又遇到一些人不懂标准准备去压缩前导帧和IFG.

### 5. 结语

本文第一章提出问题, 第二章做了一个看似straight-forward的方法来解决问题, 也是工业界有些人正在探索的. 但实际上这完全就是一出闹剧.

第三,第四章从应用范式到实现分析了一下各种利弊. 当你选择以太网的快和便宜以及规模组网的时候, 你也要承担它没有内存语义的缺陷. 同时你又不能傻兮兮的直接RoCE扩展进去.

因此GPU而言,我们还是需要一个支持内存语义交付的界面. 而语义交付上

面临多路径的问题, 需要满足**交换律**,支持乱序提交

面临组网可靠性和Lossy的问题, 需要**幂等**, 特别是一些在网计算的场景

面临传输效率低的问题, 需要pack,即满足**结合律**

本质上就是我在做NetDAM时一直强调的, 内存语义上需要满足Semi-Lattice, 以上三条是业务刚需. 另外夏Core的一段话很少有人懂, 再次强调一下

可能出乎很多人意外 ：） 实际上Load/Store/Atomic如果做成异步DMA的方式，是可以做到无限的Outstanding，只要Memory Bandwidth大于IO Bandwidth，无需流控，可以无限Load Outstanding。

这个逻辑的本质，其实和Zartbot提出的NetDAM很类似，其实，`只有获得了Memory的控制权，端侧的IO的能力才能发挥出蛮荒之力`。看明白了NetDAM的话，再进一步，就是无限outstanding的Load/Store/Atomic DMA了。

![图片](assets/a9f49a431ae5.png)

那么本质上, 我们的需求就变成了:

需要一个内存语义总线

传输上要找一些便宜的高Radix的交换机

能够比较容易的扩展并和以太网ScaleOut互通的能力.

UEC在这一系列事情上, 还在PacketSpray/拥塞控制/Libfabric这些泥潭里爬不出来.

**而UALink或许是一个潜在的很好的选择, 但UALink也需要得到更多的网党的输入. 大概就这样吧.**

参考资料

[1]
protocol: https://github.com/luismartingarcia/protocol
[2]
CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization: https://research.nvidia.com/publication/2011-11_cudadma-optimizing-gpu-memory-bandwidth-warp-specialization