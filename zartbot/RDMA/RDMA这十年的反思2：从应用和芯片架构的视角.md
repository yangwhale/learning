# RDMA这十年的反思2：从应用和芯片架构的视角

> 作者: zartbot  
> 日期: 2024年4月8日 22:45  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489285&idx=1&sn=3a53f4d177aca0a2a052450fd1a58fe2&chksm=f99607c7cee18ed14e672132ba249a61c8d968d4eeec4a5d086a3529daffcd41097704d7e7e6#rd

---

Mellanox作为一个价值60亿美金被英伟达收购有非常多的成功经验值得学习，当然也因为其芯片架构的约束在新的时代遇到了很多挑战。

本文主要是从应用需求以及芯片架构两个维度来看待RDMA这十年的演进，一方面会分析Mellanox的成功经历和遇到的挑战，以及芯片架构导致在RoCE上反反复复的内在原因，最后探讨为什么在AIGC时代它会遇到很多问题。愚人节过去不久，想起IPv9的一段话:`Those who do not study history, are doomed to repeat it`. 作为一个经历过路由器架构十多年随着业务演进，搞了好多代网络处理器，又和Marvell定制过DPU的打工仔，有必要再谈论一下这个话题， 本文仅代表个人观点，和任职机构无关。

有些内容几年前写过一个文章 [《DPU及网络处理器的历史》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486914&idx=1&sn=37d779d3ab6abf8b2460a826f32d0abb&chksm=f9961d00cee19416bbdd28c8aadf41e6b123cefc53a9b942bf931d695f2193ad6e032a247d01&scene=21#wechat_redirect) 也可以参考一下。

### TL；DR

简单来说就是Mellanox靠着ASIC架构和HPC对极低延迟的需求在很长一段时间卷赢了所有的基于NP微码架构做网卡(Tile Based NIC)的厂商，但正是因为这些架构在存储/云/AI等场景下的缺陷，使得其在有损以太网的拥塞控制难以实施，最终通过收购EzChip和Tilera的技术开始构建基于BlueField的产品线，将架构转向了ASIC FastPath+通用处理器的架构。这也是其它DPU厂商的选择，例如当时在思科和Marvell一起定制的Octeon10，Intel Mt.Evans也是类似的架构，这样的方案是针对HPC有了极低延迟，而针对AI场景拥塞控制又有了足够的算力,同时能支持存储/安全还有一系列其它复杂功能.

![图片](assets/efa822917bd9.png)

所以您也会看到在英伟达ConnectX-8这一代直接定位成了SuperNIC，市场策略上侧重于基于IB继续走Lossless的路线，而以太网在GTC的宣传中似乎完全交给了BlueField3和未来的BlueField4. 另外需要注意的是一些完全微码的NP方案会面临AI场景下拥塞控制算力不够，或者HPC场景延迟过大的问题，灵活性也没有BF3好，高吞吐也因CC算力受限，低延迟又卷不够ASIC的尴尬境地。

从工业界的角度来看，ASIC加速经常性路径，通用CPU增加可编程性通常是这类数据密集型业务的终局。二十年前路由器演进到多业务路由器，从ASIC演进到Network Service Processor，二十年后的今天网卡演进到DPU/SuperNIC/SmartNIC等多业务融合网卡，其微架构的演进也会再走一次这样的路。

本文目录

```
1. 从应用看RDMA的业务需求1.1 Kernel-Bypass,TCP透明替代1.2 HPC1.3 分布式数据库1.4 分布式存储1.5 AI大模型训练1.6 RDMA应用的流量模型2. 延迟为王的时代2.1 延迟的影响因素2.2 Infiniband诞生原因2.3 HPC, Mellanox Infiniband的成功之路2.4 Ethernet的低延迟之争3. 复杂业务带来的架构转型3.1 SmartNIC时代3.2 成也ASIC，败也ASIC3.3 DPA算力置换3.4 自适应路由和多路径转发3.5 RDMA现代化3.6 UEC,另一个Converged Ethernet的故事4. 历史的重复，架构的选择4.1 网络处理器和网卡处理器的异同5. 总结A. 附录,RDMA应用分析A.1 分布式数据库A.1.1 Oracle RACA.1.2 IBM DB2 PureScaleA.1.3 Microsoft FaRMA.1.4 PolarDB 严格一致性读及Serverless架构
```

## 1. 从应用看RDMA的业务需求

正如RFC1925网络十二条军规

It is always something. (corollary). Good, Fast, Cheap: Pick any two (you can't have all three).

对于网络而言，衡量的指标就三个：带宽/延迟/抖动。很多时候抛开应用空谈这些指标大概率的是屎上雕花。对于任何一个网络工程师/架构师，了解应用是基本功。从个人职业经历来看，在思科的时候几乎参与了国内很多行业的网络架构设计，特别是国内几乎所有证券交易所的网络和几乎所有高频交易场景下的延迟优化，并且以这些业务输入来设计路由器架构和转发特性, 所以在谈论RDMA设备的芯片架构之前，先以应用的视角来分析业务需求,本节的一些总结如下

场景极低延迟高带宽INCASTQP数HPCYNNY分布式DBYNNN分布式存储YNYYAI大模型NYYY

注：为了不干扰阅读，本文最后附录有一个更详细的应用分析，对于提到的DEC VAXCluster/Oracle RAC/Microsoft FaRM/HPC-偏微分方程数值解等内容，以后还会做一个分布式系统和并行计算的专题进行解析。

### 1.1 Kernel-Bypass,TCP透明替代

从1995年U-Net开始，RDMA的第一个场景就是Kernel-Bypass。

![图片](assets/7faf783fa5c4.png)

IBM在其大型机上基于RDMA技术实现了SMC-R(Shared Memory Communications over RDMA),SMC是IBM开源到Linux代码中，同时IBM也一同提出了IETF RFC7609，作为描述 SMC-R协议是如何实现的。其次，SMC本身也是一种协议，Linux下为AF_SMC，可以直接在socket中制定使用，没有其他特殊的hack或者 tricky的实现，和TCP等价。用户可以不用修改任何代码，通过SMC-R技术透明加速TCP。
![图片](assets/4bb0a5645469.png)

当前您可以在阿里云上支持eRDMA的虚拟服务器(注：所有8代通用计算实例均支持)上使用SMC-R技术，对于Redis这些典型应用，无需任何代码修改即可透明的提升至少40%+的性能，具体可以参考《最佳实践-使用SMC和ERI透明加速Redis应用》[1]

### 1.2 HPC

主要特征：极低延迟，QP数量

HPC主要是应用于科学研究、生物制药、基因测序、CAD/CAE、气象预报、计算模拟等多个场景。通常利用RDMA来加速MPI通信，降低集合通信延迟和长尾。在超算中很大一部分应用是对连续物理系统，如流体、电磁场等进行分析和模拟，其本质是对于大规模的多参数偏微分方程计算数值解，我们通过有限差分的方法把它转换成求解线性方程组的问题，而传统的Gauss-Seidel迭代算法效率不高，因此出现了多重网格方法进行并行计算。

![图片](assets/ab899e9036f8.png)

多重网格通过在不同尺度下的网格计算，利用稀疏网格方程维度低计算速度更快的原理，在细网格到粗网格的时候施加约束，而在粗网格求解后利用延拓算子插值到细网格的方法。

这类应用通常是伴随着整个计算任务的计算处理器核数而需要大量的RDMA QP互相通信，而通信量通常较小，因此延迟成为关键影响的因素。

### 1.3 分布式数据库

主要特征：极低延迟，单边操作

分布式数据库在处理分布式事务一致性时需要尽可能的降低延迟，早期的Oracle RAC采用了DEC VAXCluster的集群技术，其CI-Port和RDMA的数据结构几乎一致。另一方面在IBM DB2 PureScale上也使用了大型机的Coupling Facility组件，有一个集中式的Global Cache层。到现代的Microsoft FaRM基于环形内存空间构建了单边操作的RPC语义，同时也构建了基于RDMA的一致性事务处理能力，这也成为很多分布式系统课程(eg MIT 6.824)必讲的一节。当然还有阿里云的PolarDB利用RDMA实现严格一致性和Serverless等场景。

由于事务一致性的约束下，因此网络通信延迟成为最关键的路径。工业界的解法通常是利用RDMA单边操作避免在事务的关键路径上引入远端CPU。当然在单边操作下还有一些需要通知远端CPU处理的场景，也逐渐的引入了RDMA语义自身的缺陷和单边执行语义优化等需求。

### 1.4 分布式存储

主要特征：QP数量，Incast长尾延迟

主要用于存算分离的需求，在早期的DEC VAXCluster就出现了基于CI-Port的连接方式和分布式一致性的处理。后期Oracle ExaData也大量使用RDMA处理存储业务，云计算的云盘存储也采用同样的方式。通常计算集群需要通过如下方式连接

![图片](assets/5bbadbc68bb3.png)

Block Server和Chunk Server通常都有多个连接，同时并发的响应多个读写请求。例如数百个计算节点同时对一个BlockServer写入时会产生Incast的问题，进而导致长尾。更详细的内容可以参考阿里云的论文《From Luna to Solar: The Evolutions of the Compute-to-Storage Networks in Alibaba Cloud》[2],以及微软的论文《Empowering Azure Storage with RDMA》[3]。对于很多存储I/O并发，为了避免头阻塞和QP数量两个问题，AWS SRD和阿里云Solar RDMA都采用了RD语义实现，这是针对分布式存储很好的处理方式，后面我们将详细阐述。

### 1.5 AI大模型训练

主要特征：QP数量，Incast长尾延迟, 带宽大高突发

主要是集合通信这些不经意的批量同步(oblivious bulk sync,OBS)的流量，突发很高(Bursty)瞬时可以打满整个链路，另一方面伴随着MoE模型越来越多，All-to-All通信特征对incast的抑制也越来越重要。

![图片](assets/8360940ad72a.png)

但是AI大模型训练的场景下由于消息Size普遍较大，对于网络的静态延迟并不敏感，同时GPU也支持LL128的通信来隐藏延迟。

### 1.6 RDMA应用的流量模型

微软/Google的HPC专家以及Cray(HPE) Slingshot团队和博通交换机芯片团队一起写了一篇论文《Datacenter Ethernet and RDMA: Issues at Hyperscale
》[4]中对RDMA的业务有一些简单的阐述，主要涵盖了数据中心内部的东西向流量，例如HPC、AI训练和分布式推理、存储，以及一般的微服务或函数计算服务(Function as a Service ,FaaS)流量中使用的流量，总结出了三种通信模式：Incast(IN)

Incast流量模型是指多个源向单个目的节点并发发送数据流，由于缺少协同而产生拥塞。当服务偶然的被多个互相之间缺乏协同的客户端请求时产生，通常受并发的源进程数和通信事务的大小影响。这种情况在实际生产网络中会随机出现。例如，100个客户端将一个10KiB的写事务提交给一个存储服务器，所有的客户端都可以按照满带宽发送，因为他们并不知道潜在的拥塞。数据包将很快填满网络缓冲区，从而阻碍其他的流量，并最终导致服务受损而达不到SLA要求。最具有挑战性的incast模式是由小的BDP(Bandwidth Delay product，带宽-延迟乘积)的事务引起的，这使得拥塞控制机制无法在事务完成前获得可靠的信号。不断增长的带宽将越来越多的工作负载推入这个关键领域。

Oblivious bulk synchronous(OBS)

许多HPC和AI训练的工作负载都可以用这种不经意的批量同步模型(oblivious bulk sync,OBS)来表达，其中计算步骤和全局通信步骤交替进行，全局通信用于同步进程。OBS意味着应用程序的通信模式取决于少量的参数(例如通信大小和进程数)，而并不取决于处理的数据。它通常可以在应用程序启动前静态决定的，例如，消息传递接口(MPI)标准中的所有集合通信操作都是不经意(Oblivious)传输的。因此，OBS工作负载可以从算法上避免incast。深度学习中的3D并行就是一个典型的例子。OBS可以通过进程数量/计算时间和每个端点的通信大小来建模。如果计算和通信都很小，则整体工作负载对延迟敏感，这中模式经常出现在HPC和AI推理场景中。在AI分布式训练工作负载是更大的通信量，这通常是带宽敏感的。

Latency-Sensitive(LS)

对于某些工作负载，消息延迟(有时也包括消息速率)扮演着核心角色，其中一些属于OBS类别，但其他工作复杂具有复杂的和数据依赖的消息链，并构成应用程序的关键性能路径。这些应用程序通常是Strong Scaling的工作负载，时间至关重要，必须容忍低效的执行。大规模仿真这类需要严格的完成时间约束的任务落入这一类别，例如油气勘探和天气预报等应用。而一些事务处理或者搜索/推理工作负载也是如此，通常这类业务需要个位数微秒的延迟需求。

## 2. 延迟为王的时代

早期的RDMA应用主要在HPC领域，主要是追求极低延迟。Mellanox基于ASIC的解决方案在这个场景中逐渐的赢了原来HPC互联的垄断者Myrinet，并且也卷赢了一众利用NP架构微码固件实现RDMA的厂商。

### 2.1 延迟的影响因素

首先我们来看看延迟的计算方式，互联延迟通常包含两部分：静态延迟+动态延迟，静态延迟包含网络设备PHY/处理延迟和链路传输延迟，动态延迟则是传输消息大小除以带宽的传输延迟加上传输过程中队列等待和丢包重传等一系列延迟

![图片](assets/e0bcfe9a6a17.png)

注：图片来自于《High Performance Ethernet for Computing and Storage Systems》[5]

简单来说, 当交互时传输的消息较小时, 主要影响来自于静态延迟, 而随着消息的增大传输延迟逐渐增加, 在HPC中有大量的偏微分方程数值解的计算任务中，通常通信的Size仅有几个字节，因此静态延迟成为影响HPC性能的最关键因素，但是随着部署规模增大，拥塞控制不当而导致的队列延迟将成为影响性能的最主要因素。

### 2.2 Infiniband诞生原因

InfiniBand起源于1999年，其主要目的是处理器能力越来越强后PCI总线带宽(10Gb/s)演进受限，然后产生了两个组织Next Generation I/O(NGIO)和Future I/O，NGIO由Intel主导，而Mellanox也成立于这个时候来开发NGIO的技术。最终2002年Intel停止了IB相关的开发,转而支持PCIe，所以您会看到两者之间有很多的相似之处。

![图片](assets/1e17f4798b36.png)

由于要承载大量的细粒度的I/O请求，高带宽低延迟成了该协议设计的第一要素，同时拥塞控制等一众技术上也延续了主机内总线的Lossless+Credit based机制。

类似于PCIe多lane的方式，IB也通过支持多lane的方式提升带宽来匹配处理器和主机间的互联带宽，这在当时以太网还处于1Gbps的时候，带宽优势凸显

![图片](assets/ab15a80d225a.png)

Infiniband 单个lane支持在一对双绞线(4 wire)上支持2.5Gbps的速率，通过4x(16 wire)支持10Gbps，通过12X(48 wire)支持30Gps, 连接器如下图所示：

![图片](assets/4ab54dc2810f.png)

另一方面Infiniband还期望于融合数据中心内PCI/以太网/集群互联和存储FiberChannel多种技术，统一互联

![图片](assets/43e05e9d3784.png)

但是最终演进到后来还是变成了LAN/IB/SAN三张独立的网络，很大程度上的原因就是`Good, Fast, Cheap: Pick any two`,IB的价格还是太高存储和通用计算无法承担, 因此慢慢的出现 FCoE/RoCE这些Converged Ethernet的需求来解决成本的问题。

### 2.3 HPC，Mellanox Infiniband的成功之路

在Inifiband诞生之时,HPC正在从专用网络向Myrinet迁移，Myrinet来源于CalTech的Mosaic超算实验平台和基于USC基于Mosaic构建的AtomicLAN，针对低延迟通信需求进行了优化，其网卡硬件结构如下：

![图片](assets/f0252c799e93.png)

通过在LANai芯片上的Firmware开发，可以对多种集合通信源语进行卸载优化和加速，这也是可编程网卡最早的雏形。

TOP500超算集群部署规模的演进如下图所示，2002～2004年时Myrinet几乎占据了TOP500互联30%的市场份额。

![图片](assets/1ad11462ac3f.png)

但是伴随着10Gbps以太网和Infiniband的多个厂家群起攻之，虽然在2005年也发布了基于10G以太网的Myri-10G，但是生态上的封闭使得其在竞争中逐渐落败。基于Infiniband的Mellanox/Voltaire/Qlogic和一些10GbE以太网厂商逐渐胜出。

2008年思科停售了InfiniBand交换机产品线，2010年Mellanox完成对Voltaire的收购后，QLogic成为唯一一个和Mellanox竞争的IB厂家，但坚持了2年后把整个IB业务卖给了Intel，就此Mellanox成为Infiniband的唯一供应商。虽然后期Intel发布了OmniPath等技术与其竞争，但也很快就败下阵来。

另一方面Mellanox也深耕HPC市场，针对HPC中集合通信的Barrier和Allreduce等常见业务瓶颈，开发了SHARP功能进一步降低集合通信延迟。

回顾这段时间的市场变化，我们可以清晰的看到由于业务相对简单，通过ASIC实现的数据路径降低了静态时延，同时通过与交换机协同构建SHARP等在网计算能力降低了通信量，进一步降低了动态传输延迟，这成为Mellanox在HPC市场胜出的关键因素。

### 2.4 Ethernet的低延迟之争

随着10GbE以太网越来越普及，以及Infiniband成本太高这两方面原因，Mellanox也开始了转向在以太网上支持RDMA。由于延续Infiniband低延迟的需求，RoCEv1仅使用了以太网头，报文格式如下，通过借助DCB/PFC实现2层无损以太网来承载业务，但这样导致在数据中心内部署非常困难最终以失败告终。

![图片](assets/d7856b435b05.png)

工业界开始考虑做一些妥协，增加一些消息长度支持Layer3路由能力降低部署复杂度，在2014年RoCEv2诞生，增加了IP/UDP头，抛弃了IB GRH头，报文格式如下：

![图片](assets/1b86580d27a9.png)

但是由于Mellanox ASIC架构的限制，暂时的延续了无损以太网的部署方式和Go-Back-N的重传方式，拥塞控制依旧。同时期的Cisco usNIC用了一种很聪明的方式，借助于主CPU的算力来实现滑动窗口，但是伴随着超融合的出现，思科UCS刀片服务器销量下滑以及MPLS四人离职创建Pensando的影响，思科的网卡产品线就停滞了。

当然任何架构都有利有弊，同时期的几家公司都采用基于NP微码的架构，例如Chelsio基于微码架构实现了iWARP。Mellanox发布了一份报告《RoCE vs. iWARP Competitive Analysis》[6] 但实际上具体差异和协议无关，只是实现微架构的差异，但是被Mellanox误导为协议差异。我们采用Intel E810测试了RoCEv2和iWARP，小于64Bytes的延迟均为4.5us

![图片](assets/d77f7f154eba.png)

另一个可以佐证这个判断的是在同时期高频交易的盛行，工业界也开始尝试对TCP和UDP构建低延迟转发，例如SolarFlare的Open Onload，通过ASIC实现了TCP Offload Engine，实际测试结果来看，这些硬件ASIC实现的平台延迟比Mellanox CX5更低

![图片](assets/4650224e6c9a.png)

当然在极致追求低时延的场景，还有通过专门的处理逻辑避免数据穿越PCIe，例如卷赢SolarFlare的专门用于高频交易的Exablaze网卡，最低Tick-To-Trade Latency仅34ns

![图片](assets/79e8a4187c02.png)

更多的低延迟交易系统和网卡架构的文章可以参考

[《低延迟交易系统设计》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247483692&idx=1&sn=65ba977a2e7f0a333d318a4d0c2d706c&chksm=f99611eecee198f84c0117720cf058f8c7757c74ca7f2b60280a49880fa6a2a2ef938b1422ed&scene=21#wechat_redirect)

[《包处理的艺术(4)-低延迟智能网卡设计》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485370&idx=1&sn=3b5590ccf58909f2d390df00bfb5d853&chksm=f9961778cee19e6e9b29c6898f2618e067422c69d8a1865ed67c798d764d97ef44bec91196cf&scene=21#wechat_redirect)

网卡的微架构对于数据传输延迟有极大的影响，基于ASIC的Mellanox系列网卡延迟远低于基于NP微码架构的平台。当然任何架构都有利有弊，ASIC虽然极致追求低延迟，但在功能日益复杂的时候，可编程性带来的挑战，而正是芯片架构和RoCE协议本身的追求低延迟的缺陷在未来几年导致了一系列问题，下一章我们将谈到智能网卡出现带来的架构转型。

## 3. 复杂业务带来的架构转型

随着100G以太网到来，并且伴随着对主CPU Offloading的需求增长，网卡的功能越来越复杂，因此出现了一系列转向SmartNIC/DPU/SuperNIC的架构演进。

### 3.1 SmartNIC时代

我们可以注意到在ConnectX-3这一代开始，以太网速率已经跟上PCIe总线演进的速率，另一方面CPU的处理能力成为瓶颈，并且伴随着CPU核数增多和虚拟化的流行，工业界对网卡有了更多的需求, ConnectX-4将OVS Offload到网卡，同时期还有大量的基于CX4+FPGA或者其它FPGA/NP/ManyCore的可编程解决方案

![图片](assets/6ee033620ae3.png)

随着业务需求的变化，网卡微架构也百花齐放，但是这些架构都无法很好的运行各种Offload的任务，例如基于P4编程的Pensando延迟和基于MIPS+Co-processor的Fungible虽然有较好的编程性能，但RDMA中部分对低延迟又要求的业务性能并不好，而且其可编程性也无法很好的覆盖业务场景，似乎很多人都忘了一个最基本的原则：ASIC加速经常性路径，剩下的Exception Path能用通用处理器处理就好。

注：在这个场景下，Mellanox也有值得让人敬佩的地方，构建了Innova产品线，通过CX4/CX5+FPGA来尝试解决一部分业务需求，另一方面是在2016年收购了EzChip，而收购而来的EzChip包含了一系列网络处理器技术和EzChip在2014年收购的Tilera的ARM多核片上网络技术。就此Mellanox基本完成了DPU的布局。而同时期Pensando和Fungible也刚好成立。

### 3.2 成也ASIC，败也ASIC

Mellanox ASIC流控机制一直是基于Rate的，这也是导致整个工业界在探测交换机Buffer上屎上雕花的根本原因。下图是在ConnectX-6中引入的可编程拥塞控制机制：

![图片](assets/79870f87e519.png)

在这套机制下，拥塞控制算法只能控制发包速率，当检测到拥塞时，支持ECN的交换机会标记数据包，接收方将该信息传递回发送方，然后发送方会减少其注入速率。在经历一段无拥塞期之后，发送端会自动增加速率。ECN使用二进制标志表示经历的拥塞，缺乏细粒度指示会导致需要多次往返时延（RTT）来确定正确的速率。虽然这些ECN的感知能够将交换机队列保持较浅，但是针对大模型Bursty的集合通信，以及AlltoAll的incast场景和IOPS密集的场景并不有效。

![图片](assets/833cf8a8c26d.png)

相反，我们考虑一下Window based CC, 针对远端处理能力，以及配合Packet Spray等价于一种pacing机制， 本质上极大的降低了交换机buffer的占用，这才是真正最关键的地方……

对于丢包重传，Go-back-N机制固然简单，背后的逻辑是在Infiniband网络中基于信用实现的无损传输，隐含了丢包可能因为传输误码导致，因此Go-back-N机制重传概率并不高。Go-back-N重传效率的问题只是早期Mellanox网卡实现的问题，而最大问题是不支持多路径或者无序传输，这是对AI训练集群非常重要的一个功能，因此本质上还是需要SACK机制。同样由于硬件架构的限制，在IRN中Mellanox实现了一个简单版本的Selective Repeat来支持Lossy RoCEv2。

### 3.3 DPA算力置换

针对多业务Offload的问题，Mellanox第一版是在CX6里面添加了可编程的CC引擎。直到在BF3上加入Data-Path Accelerator(DPA)来通过算力置换拥塞才一定程度上缓解

![图片](assets/4a8cffcc8001.png)

BF3硬件架构上来看，沿用了CX7的基于ASIC的快速转发路径，DPA子系统旁挂于CX7 ASIC快速转发路径旁，并采用RTOS提供实时的拥塞控制反馈

![图片](assets/a3ef8aad3575.png)

DOCA DPA包含了16个Core累计256个Thread

![图片](assets/a50b90831f0b.png)

但遗憾的是DOCA PCC还是只能基于Rate-Based的方式处理，Window based才是王道啊……哎……

![图片](assets/1e257f8d49f6.png)

### 3.4 自适应路由和多路径转发

由于AI训练场景下的大带宽需求，从ConnectX-5开始Mellanox开始支持Adaptive Routing，支持Out-Of-Order delivery。AR的业务收益Mellanox自己也讲的很清楚

![图片](assets/ada7f22f3680.png)

但是AR功能开启在以太网上将面临一系列问题，ReOrder机制如何实现？丢包重传的逻辑如何构建？当收到一个CNP后是降低速率还是切换路径，也是一个两难的问题，最终Mellanox又回到了Lossless的路上。

事实上考虑多路径的拥塞控制算法实现，很多原有单路径的结论都会被推翻，例如通过Random Packet Spray实际上的Burst影响会降低，而incast的问题通过Window Based CC的机制更容易解决，Swift的机制避免了ECN的问题同时也不需要像Mellanox那样同时在Spectrum4和BlueField3上做额外的Telemetry, 当然拥塞时降速还是选路的问题是很难，Google Falcon的PLB也掉入了这个坑里。这个问题非常难，再提示一点，如果用Dynamic WRR来做多路径又会遇到某个大权重流对某个Link的Burst，同时当Link Failure的时候，如何检测并在其它Link上恢复流量同时又避免拥塞又要打满也是一个难题。通常还有一个难题，人们在追求路径的确定性时又希望要通过使用交换机的Hash算法来算转发路径，如何解耦呢？

### 3.5 RDMA现代化

当然随着RDMA部署到存储等场景，硬件架构又遇到了挑战，例如一个典型的问题就是QP数量多了以后性能下降。

因此很多厂家对其进行了现代化改造，举两个有代表性的例子AWS SRD和阿里Solar RDMA都对Reliable Datagram有支持。AWS的出发点是多路径提高带宽利用率降低长尾延迟，另一方面是针对HPC应用QP爆炸的问题而选择了RD，而阿里云的出发点是Block Server可能会收到来自多个计算节点的请求，这些请求和chunk Server相连的QP上是相互独立的，并不需要RC语义严格的保序。

当然对于QP数量爆炸的问题，实际上是片上QP Context缓存较小，当Cache miss后需要通过PCIe从主机侧拉导致的，后期的Mellanox网卡有了显著改善，并且在ConnectX4开始支持了Dynamically Connected Transport能力，也可以很好的缓解这个问题。

### 3.6 UEC,另一个Converged Ethernet的故事

对于互联的技术Ethernet/NVLINK/Infiniband，英伟达似乎构造了三张网络，加上存储的Backend一共四张网络

![图片](assets/a689f3ebc933.png)

一方面认为东西向流量需要单独的一张ScaleOut网络，另一方面又说南北向需要以太网去接存储。但同时利用GPUDirect-Async技术又在直接细粒度的访问存储网络

![图片](assets/da4599fa23e6.png)

与此同时工业界似乎又开始了一轮新的Converged Ethernet的故事，AWS借助于Nitro EFA/SRD，Google借助于IPU Falcon将FrontEnd和ScaleOut网络整合在了一起。而UEC的出现似乎想一套以太网技术统一三张网络。HPE-Cray基于以太网构建的Slingshot interconnect也在UEC中进行讨论，但是UEC还是没有很好的回答这一系列问题，拥塞控制上还有很多路要走，ScaleUP的内存语义和ScaleOut/FrontEnd的消息语义如何平衡也是一个难题，毕竟距离和拥塞都会产生延迟。

## 4. 历史的重复，架构的选择

RFC1925网络的十二条军规里面有一条：

One size never fits all. 即没有一个方案可以适用所有场景的。

这些都是业务驱动下的芯片架构的变化，历史是不断的重复的，RFC1925网络的十二条军规有两条阐述这个事情：

Every old idea will be proposed again with a different name and a different presentation, regardless of whether it works.每一个旧的想法总会用不同的名字或者方式再次呈现，不管它是否能工作.

而针对RDMA也会再重复的出现一次，从最早期的基于处理器编程的Myrinet网卡，再到基于ASIC的普通网卡，然后到一些基于微码的网络处理器架构的智能网卡，再到DPU的出现，而如今英伟达还新增加了一个SuperNIC的概念，背后其实都是在谈同一个问题：不同的应用下芯片架构的选型。

很长一段时间以前，路由器从早期基于通用处理器，再到互联网泡沫初期疯狂的卷处理能力采用ASIC，然后到微码流水线处理器，到后期逐渐因为业务复杂开始使用网络处理器(NP)，最终演进到多业务路由器采用ASIC硬件Fastpath+通用处理器的Network Service Processor。这一段经历对于网卡的微架构有很大的借鉴意义，其实我们可以看到BlueField系列产品背后的两次收购，一次是EzChip收购Tilera(微码NP转向Service NP)，另一次是Mellanox收购EzChip(SmartNIC转向DPU），到最后英伟达收购Mellanox，这是一条非常清晰的路径。

### 4.1 网络处理器和网卡处理器的异同

从本质上讲，网络中的路由器和网卡是有几乎相似的定位. 从早期的协议转换和高性能I/O需求，到后期QoS隔离/安全/多业务等能力。特别是到了DPU时代，业务极其复杂，夏老师关于DPU的分类和功能是讲的非常清楚的

![图片](assets/fbcdc6a8d1b1.png)

而在这个年代，你也会看到各个DPU厂商背后都有做网络处理器的厂商的影子。[《DPU及网络处理器的历史》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486914&idx=1&sn=37d779d3ab6abf8b2460a826f32d0abb&chksm=f9961d00cee19416bbdd28c8aadf41e6b123cefc53a9b942bf931d695f2193ad6e032a247d01&scene=21#wechat_redirect)里面有个详细的介绍，针对多种架构在不同的时代有不同的取舍

![图片](assets/01fec99fc75f.png)

其实针对这些I/O密集型又有Latency Bound的应用而言，体系结构的选择是完全相同的，简单的来说经历了几个时期

早期基于通用处理器的纯软件转发，同样早期的Myrinet网卡也在采用这种方式对集合通信等操作进行优化

由于处理性能的需求，网络处理器转向基于ASIC的专用架构，例如Juniper早期的转发引擎套件，同样我们可以看到Mellanox ConnectX系列纯ASIC架构的实现

网络处理器随着路由器功能增多，例如FR/ATM/MetroEthernet/MPLS-VPN/BRAS等，开始采用基于微码的网络处理器架构，并采用流水线处理数据，例如Cisco PXF/Intel IXP/EzChip。同样我们可以看到随着100Gb SmartNIC的出现，部分厂商也开始采用基于微码的网络处理器架构，例如Chelsio，Netronome，而Netronome就是继承了当年的Intel IXP网络处理器，而Mellanox也收购了EzChip

随着路由器产品业务越来越多，例如思科的Integrated Service Router和Aggregated Service Router，路由器需要同时支持防火墙/DPI流控/IPS/VOIP/SBC/IPSec/压缩等多种带状态业务，传统微码架构的NP可编程性逐渐力不从心。因此工业界开始转向通过NOC连接大量通用计算核的Network Service Processor，例如Cisco QFP/Cavium Octeon/RMI XLR/Tilera 等，而当网卡演进到SmartNIC/DPU的阶段，你可以发现Mellanox的BlueField ARM子系统有着Tilera的影子，而其它几家NP处理器厂家的员工都逐渐在其它几家公司发挥着重要的作用。

## 5. 总结

本文从应用需求以及芯片架构两个维度来看待RDMA这十年的演进，本质上了解应用的需求，针对应用需求定制架构才是一条正确的路

场景极低延迟高带宽INCASTQP数HPCYNNY分布式DBYNNN分布式存储YNYYAI大模型NYYY

本质上的架构争议是On-Path Processing和Off-Path Processing的区别：

![图片](assets/e180448a6458.png)

On-Path无论是基于微码还是基于P4虽然增加了数据路径的可编程灵活性，都存在延迟无法满足存储和HPC的业务需求，因此无论是博通TrueFlow还是Mellanox ASAP，最终都是一个基于ASIC的FastPath处理经常性路径，而一些通用的计算核来处理异常路径。

当然Mellanox在RoCEv2上从Lossless到Lossy再回到Lossless，很大程度上也是受到了硬件架构的约束，协议栈早期过度关注于低延迟的HPC和存储场景，而在AI大带宽的场景中带来的一系列缺陷罢了，随着其演进到BlueField-3 SuperNIC/DPU Mode和未来架构的演进，这些问题也会逐渐解决。

参考资料

[1]
最佳实践-使用SMC和ERI透明加速Redis应用: https://openanolis.cn/sig/high-perf-network/doc/735934915657042794
[2]
From Luna to Solar: The Evolutions of the Compute-to-Storage Networks in Alibaba Cloud: https://rmiao.github.io/assets/pdf/solar-sigcomm22.pdf
[3]
Empowering Azure Storage with RDMA: https://www.usenix.org/system/files/nsdi23-bai.pdf
[4]
Datacenter Ethernet and RDMA: Issues at Hyperscale: https://arxiv.org/abs/2302.03337
[5]
High Performance Ethernet for Computing and Storage Systems: https://www.ieee802.org/3/ad_hoc/ngrates/public/calls/22_0622_HPE/zhuang_nea_01_220622.pdf
[6]
RoCE vs. iWARP Competitive Analysis: https://network.nvidia.com/pdf/whitepapers/WP_RoCE_vs_iWARP.pdf

## A. 附录,RDMA应用分析

### A.1 分布式数据库

主要特征：极低延迟，单边操作

分布式数据库也是广泛使用RDMA的一种应用，主要用途是在分布式一致性的处理上。下面简述几个有代表性的工作。但是篇幅有限，关于VAXCluster的技术架构和OracleRAC， IBM DB2 PureScale等技术的详细解析后面有空再说。这类应用的主要特征是：

分布式事务一致性需要降低延迟和长尾抖动

通常使用RDMA OneSide单边操作语义构建RPC在关键路径上避免远端CPU参与执行

#### A.1.1 Oracle RAC

在谈论Oracle RAC之前，我们需要了解最早期的产品是基于DEC VAXCluster集群技术实现的。而研发这个VAX集群的有一位大名鼎鼎的专家Gordon Bell，他对于研发这个集群的定义：

![图片](assets/c8d1c5eeca80.png)

Oracle RAC的CacheFusion的分布式锁架构也基本上完全照搬了DEC VAXCluster的分布式锁架构。

在VAXCluster硬件架构中最重要的就是一个基于消息(Message-Oriented)的高速计算机互连总线，也被称作CI Bus，而连接在CI Bus上的网络设备被称为CI Port。

![图片](assets/f5633e923a3e.png)

CI Port负责仲裁/选路和数据传输，并且也可以让VAX主机通过网络引导实现无盘启动并共享后端存储。CI Port设计的初衷有两个：

从分布式节点中尽可能多的Offload通信带来的开销

提供标准的基于消息的软件接口，用于处理器之间的通信和设备访问

CI Port的上层软件系统被称为 VMS System Communications Architecture(SCA)。SCA提供三种通信服务：Datagram/Message/block data Transfer。针对Datagram和RDMA Unreliable Datagram(UD)相同，而Message和RDMA RC下支持的SEND/RECV语义相同， block data则是类似于RDMA的单边READ/WRITE操作直接将内存页按照一定的BlockSize转移到其它节点，并且也同样保证可靠性传输。
CI-Port队列定义也和RDMA有大量的相似性，或者几位RDMA的发明者在构建U-Net的时候也充分的借鉴了DEC VAXCluster架构，Command Q/Response Q/Message Free Q的定义和RDMA中的SQ/RQ/CQ几乎一致

![图片](assets/856f31b1015d.png)

分布式锁服务通常有大量的条件执行和复杂的顺序依赖来维持分布式事务的一致性，而这些事务的处理延迟和长尾抖动都是影响数据库吞吐的关键因素，后期在DEC消失后，Oracle RAC大量采用RDMA技术，并通过RDMA Oneside Operation降低延迟的根本原因。

#### A.1.2 IBM DB2 PureScale

这是IBM的分布式数据库的核心技术，主要是依赖于在IBM大型机中构建的多个处理器协调事务处理器Coupling Facility。它也实现了分布式锁/Cache/List等三种结构用于协调多个处理器上的应用。 CF是大型机的一个专用物理处理器，具有数十GB的内存和特殊通道（CF-Link），以及称为耦合设施控制代码(Coupling Facility Control Code ,CFCC)。

![图片](assets/38a9a4d55899.png)

工业界也有一些利用X86+RDMA来实现Coupling Facility的工作

#### A.1.3 Microsoft FaRM

FaRM是微软基于RDMA实现的一套分布式一致性系统，并在Bing搜索中部署。分布式系统课程基本都会讲述，例如MIT 6.824。整个项目从2014年至今有4篇论文：

第一篇论文《FaRMv1: Fast Remote Memory》实现了基于RDMA构建的分布式事务两阶段提交和乐观并发控制(OCC)的方案，利用RDMA单边写消息和环形缓冲区实现了低延迟的消息传递原语，以及无锁读(Lock-Free READ)的算法。

第二篇论文《No compromises: distributed transactions with consistency, availability, and performance》提供了一种维持高性能的分布式事务算法，并在FaRMv1的基础上提供严格的可串行化/持久性和高可用性。

第三篇论文《FaRMv2: Fast General Distributed Transactions with Opacity》中利用RDMA单边操作来协调整全局时钟并根据时间戳排序事务

最后基于FaRM架构实现的一个分布式内存图数据库《A1: A Distributed In-Memory Graph Database》在Bing搜索中落地。

它采用了RDMA_WRITE来实现了消息传递

![图片](assets/b43db9a7a202.png)

通过构造内存环形队列和特殊的数据结构实现了单边操作的RPC，极大的降低了延迟
![图片](assets/54574707a17a.png)

#### A.1.4 PolarDB 严格一致性读及Serverless架构

PolarDB也是一个利用RDMA优化其性能的商用数据库，通过RDMA降低读写延迟，并实现严格的一致性，论文可以参考《PolarDB-SCC: A Cloud-Native Database Ensuring Low Latency for Strongly Consistent Reads》[1]

![图片](assets/d29b57066b21.png)

![图片](assets/703b4305bd32.png)

同时针对云上Serverless架构，也有一篇论文可以学习《PolarDB Serverless: A Cloud Native Database for Disaggregated Data Centers》[2]

![图片](assets/0f4a9cfc8c0e.png)

参考资料

[1] 
PolarDB-SCC: A Cloud-Native Database Ensuring Low Latency for Strongly Consistent Reads: https://www.vldb.org/pvldb/vol16/p3754-chen.pdf
[2] 
PolarDB Serverless: A Cloud Native Database for Disaggregated Data Centers: https://users.cs.utah.edu/~lifeifei/papers/polardbserverless-sigmod21.pdf