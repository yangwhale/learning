# 谈谈UEC和UALink是否融合以及协议设计相关的问题

> 作者: zartbot  
> 日期: 2024年11月15日 15:52  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492654&idx=1&sn=a94f7cadaafb45b2da07bc9c64db3bcc&chksm=f995f4eccee27dfaf23cc98a5ba0ae06afbddf81f806522a1675c313fd7afa57c5213aff9193#rd

---

Linkedin上看到Juniper的一个Sr.Director Sarada在谈UEC和UAL是否融合[1] 观点挺有趣的.

![图片](assets/4d5515ec1aca.png)

正好最近工业界也对ScaleUp和ScaleOut融合, 特别是Ethernet ScaleUP的场景有很多的讨论...特别是BRCM离开UAL和AWS加入UAL也有一些讨论, 因此来谈一些个人的想法.

### 1. Ethernet-Style Serdes不等同于以太网

UALink早期基于PCIe那一套演进太慢是渣B一直都非常反感的, 而最近UALink开始采用类似于以太网的物理层, 在这一点上Sarada原文的用词是非常准确的'Ethernet-style SerDes', 对于一个协议是否是Ethernet还是要以802.3的定义为准, 使用了像Ethernet那样的Serdes的频点/PAM4/FEC等并不能简单的认为其是Ethernet.

其实UEC和UALink的争议也就在具体的帧格式上...当前对于以太网传输小包效率问题有一些修改, 但对于以太网帧格式的修改我们在后面单独一节来讨论什么样的修改还算以太网协议,以及如何对当前以太网交换机的实现更容易.

### 2. ScaleUP和ScaleOut的本质

其实这是一个非常模糊的边界, 如果从物理服务器的视角来看, 例如IBM的大型机central processing complex(CPC)内部有4个CPU Socket, 然后通过一个System Controller芯片基于X-BUS互联, 再将多个CPC通过A-BUS互联构成一个2层的网络

![图片](assets/d0228dc9e80c.png)

那么X-BUS/A-BUS算都是ScaleUP呢? 当然从一个大型机紧耦合的视角是对的. 但是把一个CPC当成一个物理服务器, 机箱之间的互联算不算ScaleOut呢? 本来对于一个大型机就可以根据业务需求来扩展多个CPC.

这个问题背后的本质Google Fellow Amin讲的挺清楚的

![图片](assets/88482ab3b300.png)

`AI/ML的兴起`使得计算范式发生了巨大的变化，分布式系统演进到第五代，人开始`向机器寻求洞察力`。这种洞察力体现在由机器以数据为中心的计算模式，这种模式并不是简单的去处理数据，而最大的变革是要从数据中抽取能够产生决策的代码，典型代表是在线深度学习(ODL)算法在搜广推业务中落地。在这个过程中又遇到了摩尔定律的另一堵墙，核数量、Cache size、片上网络和功耗的限制使得多核处理器发展也遇到了瓶颈，“per-socket plateau”是在讨论多核处理器性能优化时，表示单个CPU插槽上的处理性能极限，即使增加更多计算资源，性能也不会再有显著增长，因此GPU/TPU等异构加速器件在数据中心内变得更加重要，带宽需求从200+Gbps 激增到1T+bps，并伴随着大量数据的通信，SmartNIC/DPU等设备逐渐出现。最终第五代分布式系统需要考虑"Perf/TCO-Service"，即整个系统满足SLA要求时，需要考虑如何低成本交付服务。另一方面在单卡性能限制时，需要考虑采用更加紧耦合的方式来获取更高的性价比和可扩展能力。

![图片](assets/663741098034.png)

因此ScaleUP和ScaleOut在这样紧耦合的趋势下伴随着Composable Disaggregated基础设施的演化,边界变得模糊了.进一步衍生出一个问题

![图片](assets/b093a0c75e8d.png)

**其实问题的本质在于到底是用Intra-host的协议(PCIe/XLINK)透出, 还是Inter-host的协议(TCP/IP,RDMA)渗入. **

这些问题其实几年前就已经讨论清楚了, 两种协议都存在各自的缺陷, 例如RDMA做ScaleUP带来的各种问题

[《HotChip2024后记: 谈谈加速器互联及ScaleUP为什么不能用RDMA》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492300&idx=1&sn=8a239883c831233e7e06659ec3425ea2&chksm=f995f20ecee27b185a42d09868bdf6cef64df38267489ee386b6d4425e57d2c2a699ada0f9cb&scene=21#wechat_redirect)

最好的办法是让I/O获得内存的控制权,进而桥接两种网络, 这也是NetDAM和TTPoE的核心.

[《DPU新范式: 网络大坝和可编程存内计算》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486644&idx=1&sn=a2a18f661c18bfb96a37d5ac0d1a9653&chksm=f9961c76cee1956091037f97b52d420008c2d575c9ce2478ee12707c1336609882c90fae1a28&scene=21#wechat_redirect)

因此**两者本质的区别是ScaleUP为一个内存语义为主的系统, ScaleOut为一个消息语义的系统**, 对于ScaleUp访问延迟带来的LD/ST outstanding数量及效率问题又涉及到GPU架构, 特别是Warp调度的问题. 那么又引申出另一层定义: ScaleUP是一个延迟敏感的系统, 而ScaleOut为一个带宽敏感的系统. 当然您可能会看到NVLink这些ScaleUP的网络也在追求极致的带宽, 其本质还是需要在大数据量传输的过程中降低传输延迟以便多个GPU分块计算时能够隐藏访存延迟.

### 3. ScaleUP内存语义和延迟需求驱动的效率

通常ScaleUP网络中需要保持相对较小的传输单元和往返延迟, 例如NVLink/XLink/PCIe/CXL等协议的Header通常为16B, Tesla TTP也仅有18B.

![图片](assets/56cc782c3466.png)

然后往返延迟也成了关键影响因素, 因为它直接决定了LD/ST能够容忍的outstanding个数

![图片](assets/53a220ddd3e5.png)

但是这样的编码也引入了一个效率问题

![图片](assets/aa5520844987.png)

而LD/ST这些内存语义简单的over Ethernet是否可行呢?答案也是不行的, 主要是在传输效率上

![图片](assets/78bb6ace28e3.png)

那么通常有两种做法:

如果LD/ST无法Batch化处理, 出于传输效率考虑, Ethernet只使用其Serdes, 并采用NVLink类似的传输

如果LD/ST可以Batch化处理, 传输效率不用考虑, 标准以太网甚至采用VXLAN封装和FrontEnd一起完成三网融合都是可行的.

LD/ST执行Batch化处理时, 以1KB为例, 在800Gbps下传输延迟仅增加10ns. 例子:TTPoE, 看上去好像也还行吧? 但这些又涉及到GPU的微架构, I/O需要占用多少Buffer, 如何Batch如何解决冲突的问题. 例如Nvidia在做MCM这些D2D ScaleUP的论文中都考虑过使用L1.5 Cache和采用FinePack一类的技术.

[《英伟达GB200架构解析4: BlackWell多die和Cache一致性相关的分析》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489759&idx=1&sn=2c55ec63d6deaeb39ff7f767896ba853&chksm=f996081dcee1810bd399a0730b65bfde4473f8b06fecfb465b51c817d17bb1cd41a32f46b154&scene=21#wechat_redirect)

### 4. 以太网协议修改

那么在当前GPU微架构无法快速修改又要基于以太网实现LD/ST语义,同时又要追求效率和低成本这几个目标下 ,自然而然的就会想到要不对以太网帧进行一些编码, 能提升效率么? 我们来看一个以太网帧

![图片](assets/7975a06816b3.png)

哪些字段可以改呢? 在这里很多人都会犯一些错误, 甚至交很多学费.不可以改的字段主要有两个

#### 4.1 Preamble

前导帧7B来自于10Mbps以太网时代, 例如那些很长的同轴电缆. 虽然后续的高速以太网系统采用了更复杂的信号编码机制, 但是工业界为了保证和早期帧格式兼容以及时钟同步的需求,还是保留了这7B.

#### 4.2 EthType

在VLAN出现之前, Cisco也有一个ISL方案, 直接在MAC地址前放置VLAN-ID. 但是最终考虑到整个报文Parser的复用,VLAN Header还是移到了MAC以后, 并且在这个位置定义了TPID用作区分后续不同的Parser

![图片](assets/045b1c4e0b8b.png)

#### 4.3 如何修改以太网

对于新的PAM4信号使用FEC引入的延迟, 工业界的做法是采用LowLatency-FEC并配合Link-Level Retry的机制, 那么这些LLR的信息如何放置? 然后我们再来考虑对于NVLink的16B header 如何映射到Ethernet MAC地址的那12Bytes?

![图片](assets/2f51487f24f2.png)

CRC有25bits 可以省掉直接用以太网的CRC字段, 但是相对于NVLink多了9bits开销.
Header字段包含了 Request Type, Address, flow control, tag id等字段有83bits
DL Header字段包括了 Packet Length, Application id 和ack id等, 共计20bits

Packet Length可以根据以太网的SFD和IPG机制省略, 然后我们把剩下的这些字段映射到MAC地址的那12B中是否够了呢? 保守估计我们可能还需要额外的4B用于交换机的Parser更灵活的进行路由查表.

对于一个交换机而言, 原来的机制是根据EthType判断是否为VLAN等协议然后进行不同的查表, 查表时源MAC用于MAC Learning, 目的MAC用于查询出端口.

而在这种新的提升效率的机制下, 交换机如果不改EthType和Preamble, 修改Parser来查表是一件更容易做的事情, 最后针对IPG再进一步优化一下, 效率就会得到很大的提升.

所以在改协议的时候, 也多看看为什么UEC PRI工作组有那么多顾虑.

#### 4.4 查表效率问题

NVLink Switch的速率为什么没有进一步提升? 由于报文Size很小, 查询的PPS要求是非常高的, 同样基于以太网的交换机,通常要在256B才能达到LineRate, 对于未来102.4T/204.8T的交换芯片如何支持128B的Linerate转发查表,其实也是一个挑战.

#### 4.5 是否真的要修改以太网报文

另一方面有一个非常有趣的观点, 对于HBM的位宽越来越宽, 还有传输速率更高了, NVLink是否也可以逐渐的扩大Payload size? 其实和另一个问题是否Batch LD/ST到1KB? 延迟也就增加10ns左右, 或者反正都是失败者联盟的, 效率也不是那么重要,先白嫖以太网?

本质上这个问题要由GPU的架构师来回答, 改不改L2 Cache的Block Size? 效率是多少? 是否设计针对ScaleUP的独立的L1.5Cache(类似于DDIO, 但请读者思考一下为什么是L1.5而不是L3?), Diesize的占用有多少? 以及在I/O上带来的复杂度占用的DieSize是多少, 对算力的影响有多大?

### 5. ScaleUP的规模需要多大?

例如CXL定义的节点规模为4096, UALink定义的规模为1024. 是否我们需要像ScaleOut那样去做一个更大规模的网络呢?  组建这样大规模的网络存在几个难题, 当交换网为2层的时候, 有大量的负载均衡和可靠性的问题需要考虑, 强如硬伟大也只有在Hopper这一代有一个PPT版本的256卡集群.

曾经也yy过NVL576,但最后发现还是IB+NVL72构成的. 个人认为如果未来三年规模到128卡~256卡, 一层交换机组网就能实现的, 就不需要做过度的设计.

而这些规模的估计考虑到了慢节点带来的处理时间偏斜对性能的影响, 当前模型TP/CP/MP等并行策略上拆分太细后通信延迟无法被计算隐藏而带来的加速比的影响等, 以及矩阵本身例如MHA后的单个Head维度的最大值影响等..

另一个问题就是成本的问题, 在ScaleUP构成两层组网后的成本是非常高的, 是否在ScaleUP域也可以有一定的收敛比? 然后又扯到Hash负载均衡的问题, 拥塞控制的问题, 一大堆复杂的东西又要进一步影响I/O所占用的DieSize...

当然老黄在催HBM4, 在HBM4的Logical Die里面能不能做一些有趣的事情呢? 其实本质上又回到了NetDAM的逻辑, 

参考资料

[1]
Should UEC and UAL Merge?
: https://www.linkedin.com/pulse/should-uec-ual-merge-sharada-yeluri-85mcc/