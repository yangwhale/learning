# 英伟达GB200架构解析4: BlackWell多die和Cache一致性相关的分析

> 作者: zartbot  
> 日期: 2024年4月26日 11:23  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489759&idx=1&sn=2c55ec63d6deaeb39ff7f767896ba853&chksm=f996081dcee1810bd399a0730b65bfde4473f8b06fecfb465b51c817d17bb1cd41a32f46b154#rd

---

等了很久GB200的Whitepaper还没有出来，但是不妨碍通过一些论文来研究BlackWell两个Die互联的10TB/s NV-HBI接口，互联带宽和封装主要是TMSC的事情，不多谈了。今天主要关注的是如何形成一个逻辑的大GPU便于编程，Jensen在接受CNBC访谈的时候说这玩意花了100亿美金，但Jim Keller又在怼说花10亿美金用Ultra Ethernet就能搞定，所以单独把GPU Multi-Die(英伟达称为Multi-Chip-Module)的互联和Cache结构拿出来分析一下，关键是`如何构成一个逻辑的GPU并进行内存亲和性和一致性用户无感知的调度`。

英伟达关于Multi-Module GPU的论文非常多，大概画了一个图整理了一下

![图片](assets/22766fc457a9.png)

简单的一个总结

多Die合封的MCM-GPU到NVLink互联的MCM-GPU构成的HMG，针对这些NUMA结构的访存，英伟达做了很多年的优化，从Cache到编译器到运行时，同时进行了大量的软硬件融合的处理。从未来展望来看，英伟达也在尝试解决NVLink对低延迟内存语义和细颗粒度访存的优化，未来极有可能Scale-UP和Scale-Out网络通过消息语义进行合并，FinePack和PROACT的工作似乎让这件事情看到一点光了。

这个系列文章还是以分析英伟达技术实现为主，并且以时间维度来分析整个研发流程， 针对以太网做ScaleUP另外再开一篇，从个人的角度是认同Jim Keller的，通过以太网实现灵活互联的观点是一致的，同时NetDAM和Tenstorrent在通信协议定制上也是相似的，通过通信源语的编码来解决一系列条件执行/路由等算法，降低功耗并且避免一些不必要的Cache层次化结构和不必要的一些Cache开销。

## 1. 2017年 MCM-GPU基于性能扩展的视角

第一篇论文来自于2017年的ISCA《MCM-GPU: Multi-Chip-Module GPUs for Continued Performance Scalability》[1]

摩尔定律增速变缓，因此单个GPU芯片的性能曲线最终会达到饱和，但是对算力的需求还在持续增长，因此需要构建多模块GPU封装的模式，通过构建易于制造的基本GPU模块(GPM)，然后通过高带宽低功耗的Die2Die连接构成一个Multi-Chip-Module的MCM-GPU

构建MCM-GPU当时的动机是： 单芯片SM和DRAM增加受到最大Die Size约束()，另一方面良率带来的产能和成本的问题

![图片](assets/10d00fc5f4f6.png)

另一种替代的方式是PCB上放置多个GPU，例如下图Tesla K80由两块GK210构成

![图片](assets/85d34c29746b.png)

但是这样的系统很难扩展，主要是工作划分、负载均衡和跨板载互连网络数据共享相关的多个未解决挑战。但随着封装技术的成熟可以通过如下的方式构建：

![图片](assets/e4faae700baf.png)

通过这种方式，内存带宽和容量扩大的2倍，而SM数量可以扩大4倍，但是这样会导致片间互访带宽受限，因此在内存分配和CTA调度以及Cache架构上都需要作出优化来降低对互联带宽的依赖。

在构建MCM-GPU时，最简单的做法是构建四个独立的GPU互联，但是又会导致任务分割编排亲和性调度等一系列问题，也会带来负载不均衡的问题。因此英伟达还是希望它能构成一个逻辑GPU，有一个共享的SYS-I/O Die和外界互联。

然后探索了如下的一种结构，每个SM有独立的L1Cache，然后通过XBar连接到L2。XBar可以通过物理地址路由到特定的L2 Cache上，拓扑采用一个环形结构降低互联复杂性。
![图片](assets/0e35912e20e5.png)

L2 Cache是一个内存侧Cache只缓存本地DRAM分区的数据，同时不需要保证L2 Cache的缓存一致性，然后采用了一个集中式的CTA调度器调度任务，这样整个MCM-GPU就像单个芯片那样使用。

因此MCM-GPU的内存系统是一个NUMA架构，对于跨Die的访问延迟影响大概要增加32个cycle，另外带宽需求进行了分析，考虑一个有4个GPM的系统，其总DRAM带宽为4b个单位（例子中为 3TB/s），其中b 单位带宽（例子中为 768 GB/s）由直接连接到每个GPM的本地内存分区提供。假设平均情况下 L2 缓存命中率为∼50%，则每个L2缓存分区将提供2b单位的带宽。在统计上均匀的地址分布场景下，每个内存分区的2b单位带宽将被所有四个GPM平均消耗。扩展到捕获所有内存分区之间的跨GPM通信，结果得到MCM-GPU的总跨GPM带宽需求。需要 4b的链路带宽才能提供4b的总 DRAM 带宽。

![图片](assets/3db0ab72b318.png)

作者对互联带宽在内存密集型和计算密集型两种应用下进行了评估，如上图所示，当时可用的技术是ISSCC 2013年《A 0.54pJ/b 20Gb/s ground-referenced single-ended short-haul serial link in 28nm CMOS for advanced packaging applications》[2]

![图片](assets/8c0b9993e315.png)

要实现3～6TB/s的最优互联还是有问题的，因此针对互联带宽瓶颈的约束进行了一些优化方案

### 1.1 L1.5 Cache

第一种方式就是增加缓存层次，构建一个1.5层的缓存，由GPM内所有SM共享，但是需要区分地址空间，本地的内存空间访问就不需要在L1.5处理了，L1.5和L2最大的区别是L1.5和L1一样有CC能力

![图片](assets/980f95ddf905.png)

针对不同Size的L1.5也进行对比分析：

![图片](assets/85a0ef06ed37.png)

计算密集型应用对缓存的配置敏感度较小，主要是访存密集型应用，与基线相比，8MB L1.5提升了4%，16MB L1.5提升 8%, 32MB是一个Reference 提升了 18.3%但是增大了Die Size和大量晶体管数量， 最终平衡的选择是16MB L1.5

### 1.2 CTA 调度

针对计算任务做分布式的NUMA亲和性调度，降低GPM之间的访问需求，同时又对外保证是一个逻辑的GPU

![图片](assets/7d88508ae9dd.png)

### 1.3 GPM本地访问亲和性的数据分片

主要是考虑CTA访问调度时的本地访问亲和性，构建了First Touch Page映射的策略

![图片](assets/18f6e96ec86b.png)
在FT策略中首次引用Page时，Page映射机制检查引用来自哪个 GPM，并将Page映射到该 GPM 的本地内存分区（Memory Partition,MP）。例如，在图中，Page P0 首次被在 GPM0 上执行的 CTA-X 引用。这导致 P0 被分配到 MP0。随后，Page P1 和 P2 首次被在 GPM1 上执行的 CTA-Y 引用，将这些Page映射到 MP1。接着，Page P3 首次被 CTA-X 引用，将Page映射到 MP0。这种策略导致 DRAM 访问大部分保持本地。无论引用顺序如何，如果Page首次从 GPM0 中的 CTA-X 引用，那么该Page将被映射到 MP0，这将使对该Page的访问保持本地并避免跨 GPM 通信。这种Page放置机制通过扩展当前 GPU 驱动程序功能在软件层实现。这种驱动程序修改对操作系统是透明的，不需要程序员进行任何特殊处理。

通过分布式的CTA调度器策略系统来实现，具有相同索引的 CTA 在内核多次迭代启动时绑定到同一 GPM，从而允许带入 GPM 内存分区的内存页在后续内核启动时继续保持本地。

### 1.4 最终的收益

通过前面三个策略，可以看到在768GB/s互联的MCM-GPU的性能

![图片](assets/7fa8fb94452d.png)

### 1.5 NUMA Aware GPU

另外2017年还有一篇论文《Beyond the Socket: NUMA-Aware GPUs》[3]通过PCIe交换机将多个GPU互联，但是又要让它成为一个Logical GPU，并进行Numa-Aware的调度

![图片](assets/1950bf3347a5.png)

这篇论文作者得出一系列结论：

NUMA GPU中，互联带宽将成为主要的性能限制瓶颈，

Socket之间的连接应该在运行时动态适应性的重新配置以最大化带宽利用率，连接策略必须在每个GPU基础上确定

L1和L2都要Numa Aware并动态调整其缓存策略

正是因为1.2这个分析，后来在V100从早期点到点Hybrid Cube Mesh连接逐渐转向NVLink Switch的方案。另一方面文章讨论了一些详细的设计细节。 如果四个芯片采用内存地址Interleaving的方式来负载均衡，这样会导致有3/4的流量需要经过互联的总线，而另一方面NUMA-aware的CTA 调度又容易引起负载不均衡的问题。

利用在Pascal上新增的Unified Virtual Memory(UVM)可以做一些page migration来提升locality

![图片](assets/44a37b780c76.png)

另一个问题就是互联带宽的约束，针对不同工作负载，单个GPU的通信模式有所不同，作者希望能够动态的分配和重新配置链路方向，来提高互联Fabric的利用率，但是这样会导致每个lane都要实现双工，还不如直接把链路带宽扩大一倍，dieSize面积只增加不到1%

![图片](assets/7bb4e400dd35.png)

最后一个问题是Cache相关的，GPU的Cache和CPU的层次结构有所不同，并不实现严格的硬件一致性，同时对于缓存而言，它既可以位于处理器侧(提供某种形式的一致性），也可以位于内存侧(无需一致性),当前的做法是在L1构建强一致性，而L2偏内存侧无需一致性

![图片](assets/129b96d5a89e.png)

先前一篇论文也谈到了一致性的问题来构建L1.5 Cache扩展支持远程数据的缓存，但是静态比例分配也缺乏灵活性，因此这篇文章提出了动态分配L1/L2的做法， 上图的d算法。另外文章也分析了基于硬件的Cache一致性的一些取舍。

## 2. 2018年 软硬件一体手段提升NUMA性能

MICRO-2018有一篇文章《Combining HW/SW Mechanisms to Improve NUMA Performance of Multi-GPU Systems》[4],作者把这个工作简称为CARVE(Caching Remote Data in Video Memory)，主要原因是当前的NUMA访问内存机制，无论是在CTA调度上还是在Page分配上或者是实现Page Migration都会导致性能下降，而主要原因是GPU LLC容量的问题。因此作者通过在GPU的显存里构建一个专用的区域来缓存最近访问的远程共享数据，仅需要将显存的3%用于CARVE即可消除NUMA带宽瓶颈，其性能与理想NUMA-GPU系统的差距仅为6%

### 2.1 概述

虽然2017年有了First Touch Page Mapping和L1.5 Cache这些解决办法，但是对于很多应用还是存在瓶颈，通过构建共享Page的副本可以增加系统的性能

![图片](assets/05941a785b3c.png)

但是这些共享内存的副本又影响了GPU实际可用的内存，另一方面Page Replication仅限于Read Only的Page，针对Read-Write共享页性能影响非常大，下图显示了不同Workload的访存特性

![图片](assets/b68989fbc8cb.png)

虽然可以通过GPU 片上的LLC来cache，但通常cacheSize不够，因此通过借用本地内存来复制远程共享数据，同时将数据操作的粒度从OS Level的2MB降低到更细粒度的，例如一个128B的Cache line。

### 2.2 CARVE RDC

![图片](assets/a588ff4df41e.png)
CARVE在本地内存中构建了一个独立的区域称为(Remote Data Cache,RDC)来缓存远端内存。CARVE对现有GPU设计所需改动极小。在GPU LLC Miss时，GPU内存控制器确定缺失的内存地址将由本地GPU内存还是远程GPU内存服务。如果请求映射到本地GPU内存，数据将直接从本地内存中获取并发送到LLC。否则，内存控制器首先检查数据是否在RDC中可用。如果在RDC中命中，直接从本地读取，如果在RDC中缺失，缺失数据将从远程GPU内存中检索并返回到LLC。缺失数据也会插入到RDC中。 RDC以CacheLine 128B粒度构建，用DRAM构建Cache参考了AlloyCache论文《Fundamental Latency Trade-offs in Architecting DRAM Caches》[5]

![图片](assets/508264a63bd8.png)

针对有32GB显存的平台进行了评估，利用2GB RDC就可以覆盖，并大大的降低对远端内存的访问

![图片](assets/4033a49ca027.png)

但是CARVE对于随机内存访问由于大量的Cache未命中反而会降低10%的性能。但是针对大多数应用，只要能够有效的保证RDC之间的一致性就能够带来显著的性能提升。

![图片](assets/be752714478e.png)

### 2.3 RDC Coherence

传统的GPU设计通过在两个Kernel的boundary来维持一致性，即每次Kernel调用结束后有一个Flush，在下一个Kernel调用时再从内存里读取更新后的值。类似的可以通过这样的方式来扩展到RDC上，在Kernel调用结束时，更新RDC同时写到远端的内存中，但是invalid数百万条RDC-Cache Line可能需要毫秒级的时间，在每个Kernel Boundary产生毫秒级延迟会对应用程序带来显著的影响，因此期望架构能够支持高效的Invalidation和Write Back RDC cache line

![图片](assets/57fdd1746b13.png)

由于RDC tag/valid/dirty bits都存在GPU显存内，Invalidation RDC-Line读写内存，根据RDC的大小， Invalidation是一个长延迟并且带宽密集型的过程。但是，并不需要invalidation所有的航，而只需要知道RDC数据是否过时，可以通过Epoch Number来实现

![图片](assets/3ee46c6c8490.png)

类似于分布式数据库中的MVCC，上一个Kernel调用时有一个Epoch Version存放在EPCTR中，当同一个Kernel调度时有效，但下一个Kernel加载后将发现该Epoch老旧，需要Invalidation并从远端加载刷新。另一方面对于脏页刷新，也希望有一个高效的在Kernel Boundary将脏行刷新到远端内存并写透

`硬件一致性`在论文《Cache Coherence for GPU Architectures》[6]提出了一种GPU-VI算法，利用这个算法配合来增强RDC, GPU-VI实现了（a）写透缓存和（b）在存储请求时向所有远程缓存广播写失效。但是对每个Store请求发送写失效可能会显著增加系统中的网络流量。一般来说，写失效只对读写共享CacheLine是必需的，而对于私有读写CacheLine可以避免。因此，通过识别读写共享缓存行，可以减少Invalidation的流量。为此，使用每个内存位置的缓存行粒度的内存内共享跟踪器（IMST）动态识别读写共享CacheLine。我们提出一个2位IMST，存储在主节点每个CacheLine的备用ECC空间中。下图显示了IMST跟踪的四种可能状态：unCached、Private、R-Shared和RW-Shared

![图片](assets/105b4cdeb6fc.png)

## 3. 2019 MCM-GPU的能耗分析

HPCA-2019有一篇文章来分析MCM-GPU的能耗《Understanding the Future of Energy Efficiency in Multi-Module GPUs》[7]MCM-GPU成为延续GPU性能增长趋势的关键路径。论文作者提出了一种新的GPU效率指标——EDP（Energy-Delay-Product）缩放效率，以量化MCM-GPU在保持强性能扩展的同时整体能源效率的变化。为了进行深入分析，他们开发了一种新颖的自顶向下GPU能量估算框架。论文也揭示了一种反直觉的现象：增加（而非减少）用于提高各GPM间连接带宽的能量投入，实际上可以最多降低45%的总体GPU能耗，这也为GPU疯狂的开始卷互联带宽提供了某种理论依据。

![图片](assets/bf6c73aff88a.png)

## 4. 2020 MCM-GPU及NVLink多卡互联

HPCA-2020有一篇文章论述了MCM-GPU再进一步通过NVLink ScaleUP《HMG: Extending Cache Coherence Protocols Across Modern Hierarchical Multi-GPU Systems》[8]

![图片](assets/735208ea615d.png)

虽然以往关于GPU缓存一致性可以通过简单的硬件和软件协议来满足需求，但是针对MCM-GPU多卡通过NVLink互联使得整个缓存系统存在更深的层级结构和非均匀特性，这篇文章提出HMG（Hierarchical Multi-GPU）缓存一致性协议，专为前瞻性的多GPU系统设计。HMG在简单性和性能之间实现了平衡：它采用易于实现的类似GPU-VI协议跟踪一致性状态，但采用层次化结构来追踪共享者来避免GPU之间带宽的约束。

2018年的CARVE虽然采用Local DRAM作为Cache，但是没有追踪共享者，而且需要广播来invalidation所有的READ-WRITE-Shared data。从应用的角度来看，RNN/分子动力学/图神经网络都存在细粒度的CTA之间的通信。

CUDA和OpenCL最初都支持粗粒度的BSP模型，这种范式下，同一个CTA内的线程可以共享内存，并通过CTA执行barrier操作，但仅允许起依赖的Kernel调用之间进行CTA间的同步，无法保证Global Memory的任意通信的正确性。但后期编程方式更加灵活高效的支持数据共享，引入了范围的概念:同CTA(.cta),同GPU(.gpu)和系统任意位置(.sys)内的任何其他线程的通信。

但是很多强一致性的内存模型带来了复杂性和延迟，例如MESI不适合于GPU。虽然GPU-VI是一种有效简单的协议，但是针对不同的内存范围(.cta/.gpu/.sys)模型，强制Multi-copy-atomicity(MCA)引入了额外的开销，另外GPU-VI没有考虑层次化结构对应的不同带宽带来的复杂性,如下图所示，

![图片](assets/f0852ab46914.png)

将一致性从MCM-GPU扩展到HMG(Hierarchical Multi-GPU)，即使细颗粒度的硬件GPU-VI也和理想情况存在很大的差距，并且未来的GPU共享的L2Cache会更大，这样会放大粗粒度缓存失效的成本，以及带宽受限带来的从远端GPU重新加载失效数据的成本。同一个GPU上的多个GPM经常会重复访问存储在远端GPU熵的相同地址范围：

![图片](assets/bb9a402b6c2f.png)

因此构建构建HMG作为能够扩展到多个GPU的层次化协议，虽然CPU的层次化缓存一致性已有大量研究。然而，与GPU不同的是，CPU通常强制执行更强的内存模型并且有更严格的延迟要求。因此，像MESI这样的CPU一致性协议跟踪所有权以利用写数据局部性，并引入了一系列复杂的处理机制，这些复杂性对于延迟受限的CPU是合适的。但是针对GPU的内存模型是更加松散的，因此类似于CPU的一致性协议在GPU上是不合适的。

在GPU上，Scoped GPU Memory Model也指导了良好GPU一致性层次的设计，明确将Scope作为编程模型一部分的GPU内存模型仅要求在同步边界处以及仅针对所讨论Scope内其他线程保持一致性。多GPU应用程序中的常见模式是，运行在同一GPU上的CTA或内核首先彼此同步，然后较少频率地与其他GPU上的内核同步。这些模式严重依赖于.gpu Scope相对于.sys Scope的相对效率；虽然一些先前工作得出结论认为单个GPU内Scope是不必要的，但在多GPU环境中最宽和最窄Scope之间的延迟/带宽差距是一个数量级。

虽然有一些工作提议对GPU采用multi-copy-atomicity,MCA内存模型,但Scoped GPU内存模型已经形式化的分析了并不太需要MCA模型，简单来看，MCA会为随后的内存访问造成明显的延迟。大多数CPU通过使用带有许多瞬态状态的复杂一致性协议以及使用乱序执行和推测来隐藏延迟开销来强制执行MCA。而针对GPU，为了减少停滞，GPU-VI在L1和L2缓存中分别增加了3个和12个瞬态状态以及24个和41个一致性状态转换。然而，在多GPU环境中，到远程GPU的往返时间是一个数量级更大的，会给一致性协议隐藏延迟的能力带来显著压力。

论文从传统的非层次化的一致性协议non-hierarchical cache coherence(NHCC)开始讨论并逐渐引入层次化结构。如下所示，在一个多GPU系统中的Cache结构如下图所示

![图片](assets/14f078808de7.png)

图中黄色L2Cache放大来看NHCC的一致性结构

![图片](assets/2d0a95e5604b.png)

与许多协议一样，NHCC在每个GPM内每个L2缓存旁附带一个单独的目录。一致性目录采用传统的组关联结构组织。每个目录项跟踪所有GPM共享者身份以及一致性状态。与GPU-VI类似，每行可处于两种稳定状态之一：有效(Valid,V)或无效(Invalid,I)。然而，与GPU-VI不同的是，NHCC没有瞬态状态，仅对释放操作需要确认。非同步存储（即绝大多数存储）在NHCC中不需要确认。

在NHCC中，仅在两种情况下发送显式一致性维护消息：当不同GPM上的CTA之间存在读写共享时，以及发生目录容量带来的Eviction时。大多数内存访问不产生一致性开销（数据为只读或CTA私有）保证了GPU的峰值吞吐。

![图片](assets/0cb50dfadf1a.png)

在考虑到HMG层次化结构的一致性协议设计主要是将来自各个GPM的多个缓存请求能够在穿越低带宽的GPU间链路之前在单个GPU内部进行合并和/或缓存，从而节省带宽和能耗。HMG由两层组成。第一层专为GPU内缓存设计，而第二层旨在优化GPU间的内存请求路由。对于GPU内层，为每个MCM-GPU内的每个给定地址定义一个GPU Home Node(MCM-GPU-HN)。

MCM-GPU-HN使用NHCC管理GPM间一致性,因此MCM-GPU内部的一致性可以不用查询远端的其它MCM-GPU系统。 另一方面，为每个地址的其中一个GPU主节点定义为System Home Node(SHN)。SHN的选择可以使用任何NUMA页面分配策略，如First Touch Page Mapping、NVIDIA Unified Memory等。

在多个GPU中，目录使用NHCC目录设计的一种层次感知变体来跟踪共享者。具体来说，每个GPU HN将跟踪同一GPU中其他GPM之间的任何共享者。每个SHN将跟踪其他GPU之间的任何共享者，但不跟踪这些其他GPU内部的个别GPM。对于M-GPM、N-GPU系统，每个目录条目因此需要跟踪多达M+N-2个共享者。

![图片](assets/0e7f38d83c2e.png)

## 5. 2020 LADM 基于数据局部性的调度

另一种避免跨GPU访问的方法就是在编译阶段标注亲和性并在运行时根据亲和性调度ThreadBlock，具体可以参考LADM这篇论文《Locality-Centric Data and Threadblock Management for Massive GPUs》[9]

当前HMG构成一个NUMA系统

![图片](assets/46218a809716.png)

NUMA对性能带来的影响如下：

![图片](assets/b573adc83e0b.png)

LADM在编译器和Runtime上解决这个问题

![图片](assets/3f8d20f69639.png)

一方面是在编译阶段分析并标注内存的Locality属性，构建Locality Table

![图片](assets/90ec048404bc.png)

Locality根据行/列等访问内存特性进行分类标记

![图片](assets/d3d8cf4b5fbd.png)

![图片](assets/ae8c4223b6b2.png)

另外针对远程的访问CachePolicy也进行区分RONCE/RTWICE

![图片](assets/2fa2dc3eb4f0.png)

## 6. 2021 COPA-GPU

随着GPU逐渐采用低精度浮点数矩阵运算来提升DL性能，从计算和访存的角度来看，传统的高精度HPC和当前的DL负载存在明显的差异，而融合的GPU设计难度增大，因此英伟达也在探索同一个Die通过不同的封装来优化DL相关的内存访问带宽和容量的问题，Composable On-PAckage GPU(COPA-GPU)的架构诞生出来《GPU Domain Specialization via Composable On-Package Architecture》[10]

![图片](assets/9f47df93bad2.png)

同一个计算Die但是通过不同的封装来解决不同Workload的问题，评估了COPA-GPU在DL训练和推理环境下的性能潜力：

非常大的缓存容量可以显著改善DL推理

显著提高DL训练，需要缓存和DRAM的双重改进。

因此作者提出了一种特定的COPA-GPU设计，分别将单GPU训练和推理性能提升高达31%和35%，同时显著降低了数据中心中Scale-Out GPU训练的成本。

![图片](assets/3139009f9f18.png)

![图片](assets/09cfb7a3c891.png)

## 7. GPU内存池化和压缩

COPA-GPU是在封装上解决这个问题，那么还有没有别的解决问题的办法呢？ 构建Offload 内存池？ISCA-2023有一篇论文《Scaling Infrastructure to Support Multi-Trillion Parameter LLM Training》[11]

![图片](assets/f911236db20a.png)

文章得出一些结论：训练一个拥有100万亿参数的LLM模型需要每个GPU配备1TB的Offload Mem，且双向带宽达到100GB/s。更大规模的LLM模型即使使用了Offload Mem，也需要更大的HBM。当然我们可以看大GraceHopper和GraceBlackWell通过NVLink-C2C连接到ARM子系统的内存中构建Offload Memory,那么另一篇论文就值得参考了《Buddy Compression: Enabling Larger Memory for Deep Learning and HPC Workloads on GPUs》[12]

![图片](assets/94e1c3c62424.png)

其实对于GPU而言，十多年前就有相应的纹理内存压缩技术了，Buddy Compression的目标是通过ScaleUP网络实现内存池化，压缩算法采用了《Bit-Plane Compression: Transforming Data for Better Compression in Many-Core Architectures》[13] Buddy Compression的内存分配理念是，如果一个数据能够压缩的很小，那么放到本地内存，如果即便压缩了也很大导致本地空间占用，那么不如利用大的互联带宽传递到远端的Buddy Storage中。

但是有一个问题，如何找到压缩的数据，因此需要有相应的TLB和元数据处理

![图片](assets/0a539832711a.png)

## 8. 细粒度访存优化

### 8.1 GPS 基于PubSub的多GPU访问

现有的Unified Memory虽然提供了简单的编程模型，但是牺牲了性能，由于由于无法有效利用系统资源，应用程序甚至可能随着GPU数量增加而出现性能下降。因此提出了《GPS: A Global Publish-Subscribe Model for Multi-GPU Memory Management》[14]

这是一种硬件/软件多GPU内存管理技术，通过主动数据传输高效协调GPU间通信。GPS兼具多GPU共享内存的可编程优势与GPU本地内存的性能。为此，GPS自动跟踪每个GPU执行的数据访问，维护每个GPU本地内存中共享区域的物理副本，并将更新推送到所有消费者GPU中的副本。GPS与现有NVIDIA GPU内存一致性模型兼容

![图片](assets/0307e318d291.png)

在NVLink Switch上已经实现了NVLS可以执行针对Store指令的多副本写的操作，因此通过PubSub的方式来主动更新，GPS工作原理如下：

![图片](assets/e233d48b89e9.png)

`GPS Load`：对于由订阅该页的GPU向GPS地址空间发出的Load操作，会在订阅时配置常规GPU页表，将其虚拟地址转换为本地副本的物理地址。因此，GPS Load遵循与常规本地内存加载相同的路径，如图中的R1、R2、R3所示。在不常见的情况下，如果GPU未订阅此特定页，则Load会从远程写队列转发值，或者向其中一个订阅者发起远程请求。

`GPS Store & Atomic`：对GPS页的Store最初按正常存储进行，如图中的W1和W2所示。当线程对地址进行存储操作，该地址在常规TLB中的GPS位被设置，并且存在本地副本时，写操作会携带虚拟地址和物理地址转发至本地副本（W3），确保同一GPU线程后续的本地读取能观察到新写入，这是现有GPU内存模型的要求。这种模式还确保了持有本地副本的L2缓存作为在写入被转发到GPU外部之前，对该地址进行Store操作的同GPU内的Ordering Point。在不常见情况下，即不存在本地副本，使用虚拟地址代替。此外，无论是否存在本地副本，都会将写入与其虚拟地址一起转发至GPS单元，以便复制给远程订阅者（W4、W5、W6）。原子操作遵循与Store相同的行为。

Subscribe机制可以采用手工API的方式，也可以根据程序的访存模式自动订阅，例如：

![图片](assets/d92119989751.png)

### 8.2 PROACT

论文在《Efficient multi-GPU shared memory via automatic optimization of fine-grained transfers》[15]，主要是通过在编译期分析并自动追踪Producer Kernel和consumer Kernel之间的访存请求，然后进行优化

![图片](assets/4117d016489a.png)

相对于原有的方式，隐藏了延迟

![图片](assets/529531f25eac.png)

基于BulkDMA是指在一个Kernel计算完成后，通过cudaMemcpy显式调用，传输完成后启动另一个Kernel进行运算。第二种是P2P GPU访问，在多GPU系统中，每个GPU可以独立的去读取或者写入到远端的GPU而不需要CPU进行同步处理，然而，当从远程内存执行加载时，这些P2P加载经常因GPU之间互连的延迟较长而使线程执行停滞，无法通过多线程隐藏加载延迟。同时直接使用P2P存储通常会导致在GPU之间互连上发出大量小Size的写入导致互连效率低下。第三种方式是基于通信库，如MPI/NCCL/NVSHMEM等提供通用的GPU间通信和同步模式，尽管这些库仍在不断优化，但在执行大型传输时，它们很难避免DMA初始化开销。同时这些库都没有试图智能地聚合细粒度传输以提高互连效率。

另一方面无论是PCIe还是NVLink针对小Size传输的效率都非常低

![图片](assets/6ca9084e73c0.png)

两种互连技术对于大于128字节（常见缓存行大小）的传输都能提供高效率，但对于较小的传输大小，效率急剧下降。由于协议分组开销主导了实际有效吞吐量，且在4字节存储时，传输效率在NVLink上降至8%，在PCIe上降至14%，因此在这些较小粒度下互连效率降低。因此，尽管从隐藏延迟的角度来看，P2P写入可能是传输数据最有效的方式，但必须大幅提高互连效率以提升整体多GPU性能。

![图片](assets/eaff8ff4d8b7.png)

PROACT试图在多GPU系统中结合基于DMA的大规模复制和点对点访问的优势，提供大型传输的互连效率以及SM发起的点对点存储的非阻塞性语义。PROACT通过以下方式提高多GPU系统的性能：

平衡数据传输与GPU计算的Overlap；

最大化写合并的机会；

在时间上平滑互连利用率，确保不浪费带宽；

确保通信发生在足够大的粒度上，从而提高互连效率。

### 8.3 FinePack

在HPCA-2023有一篇论文《FinePack: Transparently Improving the Efficiency of Fine-Grained Transfers in Multi-GPU Systems》[16]主要是针对一些细颗粒度的访存请求导致的低效传输

![图片](assets/969f48eff7d8.png)

不同应用的访问内存Size如下，然后作者提出了一种Pack方式

![图片](assets/073c39ec14b0.png)

在PCIe上的编码格式如下：

![图片](assets/4696dcad008e.png)

FinePack架构如下：

![图片](assets/aad520abc6e8.png)

Remote Write Queue结构：

![图片](assets/628d55cf449f.png)

## 9. 一些个人的分析和展望

多Die合封的MCM-GPU到NVLink互联的MCM-GPU构成的HMG，针对这些NUMA结构的访存，英伟达做了很多年的优化，从Cache到编译器到运行时，同时进行了大量的软硬件融合的处理。从未来展望来看，英伟达也在尝试解决NVLink对低延迟内存语义和细颗粒度访存的优化，未来极有可能Scale-UP和Scale-Out网络通过消息语义进行合并，FinePack和PROACT的工作似乎让这件事情看到一点光了。
而UltraEthernet这帮人还在瞎卷HPC Ethernet这些伪需求，我真不知道这帮人瞎创新前是否能够好好的去学习一下别人的工作，当然这只是最近看的一部分论文，还有很多东西也等待我自己去学习。

参考资料

[1] 
Understanding the Future of Energy Efficiency in Multi-Module GPUs: https://ieeexplore.ieee.org/document/8675192
[2] 
A 0.54pJ/b 20Gb/s ground-referenced single-ended short-haul serial link in 28nm CMOS for advanced packaging applications: https://ieeexplore.ieee.org/document/6487789
[3] 
Beyond the Socket: NUMA-Aware GPUs: https://ieeexplore.ieee.org/document/8686671
[4] 
Combining HW/SW Mechanisms to Improve NUMA Performance of Multi-GPU Systems: https://research.nvidia.com/publication/2018-10_combining-hwsw-mechanisms-improve-numa-performance-multi-gpu-systems
[5] 
Fundamental Latency Trade-offs in Architecting DRAM Caches: https://memlab.ece.gatech.edu/papers/MICRO_2012_1.pdf
[6] 
Cache Coherence for GPU Architectures: https://www.cs.sfu.ca/~ashriram/papers/2013_HPCA_GPUCoherence.pdf
[7] 
Understanding the Future of Energy Efficiency in Multi-Module GPUs: https://ieeexplore.ieee.org/document/8675192
[8] 
HMG: Extending Cache Coherence Protocols Across Modern Hierarchical Multi-GPU Systems: https://ieeexplore.ieee.org/document/9065597
[9] 
Locality-Centric Data and Threadblock Management for Massive GPUs: https://ieeexplore.ieee.org/document/9251964
[10] 
GPU Domain Specialization via Composable On-Package Architecture: https://dl.acm.org/doi/pdf/10.1145/3484505
[11] 
Scaling Infrastructure to Support Multi-Trillion Parameter LLM Training: https://openreview.net/pdf?id=rqn2v1Ltgn0
[12] 
Buddy Compression: Enabling Larger Memory for Deep Learning and HPC Workloads on GPUs: https://arxiv.org/abs/1903.02596
[13] 
Bit-Plane Compression: Transforming Data for Better Compression in Many-Core Architectures: https://ieeexplore.ieee.org/document/7551404
[14] 
GPS: A Global Publish-Subscribe Model for Multi-GPU Memory Management: https://research.nvidia.com/publication/2021-10_gps-global-publish-subscribe-model-multi-gpu-memory-management
[15] 
Efficient multi-GPU shared memory via automatic optimization of fine-grained transfers: https://dl.acm.org/doi/abs/10.1109/ISCA52012.2021.00020
[16] 
FinePack: Transparently Improving the Efficiency of Fine-Grained Transfers in Multi-GPU Systems: https://ieeexplore.ieee.org/document/10070949/