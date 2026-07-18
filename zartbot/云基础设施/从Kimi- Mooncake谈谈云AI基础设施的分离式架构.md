# 从Kimi: Mooncake谈谈云AI基础设施的分离式架构

> 作者: zartbot  
> 日期: 2024年7月2日 23:31  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490335&idx=1&sn=6b9e70fb584026930a81922fc457bae1&chksm=f9960bddcee182cbf5272055e5bd9080ea93a376973c022698e0e5fabd2c389e603ab0a8ed23#rd

---

## TL;DR

月之暗面最近公布的一篇Mooncake的分离式推理系统的Technical Report, 其中 @许欣然老师谈到，公开的目的之一就是推动硬件厂商和云厂商向分离式，乃至未来异构分离的方向演化。因此针对这个问题展开讲讲云AI基础设施的分离式架构. 本文第一章先概述介绍一下Mooncake的工作原理, 第二章再来展开谈谈易购分离架构的演化.

![图片](assets/cc3e1a218cf4.png)

看到中间那层淡黄色的Distributed KVCache Pool想到了2020年的工作NetDAM,如下图所示

![图片](assets/a203f00f121c.png)

如今Disaggregated架构可能会像当年MapReduce那样成为一个新的业界标杆,MapReduce诞生于互联网泡沫时期, 伴随着互联网大数据处理而成长, 而分离式架构或许也会有同样的历程. `新的应用新的数据瓶颈, ScaleUP的大型机再到廉价的ScaleOut分布式集群,再到分离式架构...`

![图片](assets/c39b6893cf3d.png)

当时的Data-Centric和如今的KVCache-Centric...

## 1. 概述Mooncake以KV Cache为中心的分离式推理

对于一个推理系统,通常用户服务的SLA分为两个指标:

首个Token的响应时间(time-to-first-token,TTFT)

Token之间的响应时间(time-between-tokens,TBT)

从设备的角度看,LLM推理过程可以分为Prefill和Decode两个阶段,前一阶段是Compute-Bound后一阶段是Memory-Bound. 以整个集群的视角来看为了提高吞吐量, 一方面需要尽量多的重复使用KVCache以时空折中的方式来降低计算资源的开销.另一方面从每个Batch来看需要提高MFU. 但是从更远的位置开始重用KVCache需要带来额外的计算延长TTFT, 而更大的BatchSize会导致更大的TBT. 为了解决这个问题采用了分离式的架构.

![图片](assets/88a96ae3126c.png)

在Prefill端采用算力更强的GPU,例如H100/H800等. 而在Decoder阶段采用带宽更大算力相对较弱的GPU,例如H20等充分利用HBM带宽来处理Memory-Bound的计算. Mooncake最大的一个创新是针对两阶段的调度处理, 优化目标是Prefill阶段尽量多的复用Cache,Decoder阶段尽量的提高吞吐, 但关键来自于SLO的视角来约束. 当然还要考虑在当前算力紧张时,高峰时期的过载处理和一些优先级的调度能力.

Mooncake采用了一种分离架构,核心是将将GPU集群的CPU、DRAM、SSD和RDMA资源分组，以实现分离的KVCache. 下图展示了KVCacheBlock的存储和传输逻辑

![图片](assets/17201ca942d2.png)

在CPU内存中,KVCache以页的形式存储.根据请求的模式,通过LRU/LFU等请求特征进行动态的缓存处理.在CPU和GPU之间有一个Messenger组件通过RDMA传输.同时还可以为外部用户提供上下文的API, 例如通过外部的OSS这些存储把上下文通过RDMA直接灌入作为Prefetch以及Write-Back.
为了调度这些组件,它实现了一个Conductor的全局调度程序, 下图展示了一个典型的工作流程. 调度程序会选择一对Prefill和Decoder节点,并启动工作流:

![图片](assets/73fed3a3e6ea.png)

KVCache重用: Prefill节点组接收到一个请求,包含原始输入、可重用的前缀缓存块ID以及分配给请求的完整缓存块ID。它根据前缀缓存块ID将前缀缓存从远程CPU内存加载到GPU内存，以启动请求。如果不存在前缀缓存，则跳过此步骤。这种选择平衡了三个目标：尽可能多地重用KVCache、平衡不同Prefill节点的工作负载，并保证TTFT SLO。

增量Prefill: Prefill节点组使用前缀缓存完成Prefill阶段, 并生成增量KVCache存储回CPU内存.如果为缓存的输入Token超过一定阈值,则将Prefill阶段分成多块并以Pipeline的方式执行. 此阈值的选择主要考虑相应的GPU的Compute-Bound

KVCache传输: Messenger服务部署在每个节点上来管理和传输这些Cache. 在相应的推理实例上以独立的进程来运行, 并执行高速的KVCache传输. 同时需要考虑尽量Overlap通信, 采用异步通信降低等待时间.

Decoder阶段: 在Decoder节点CPU DRAM中收到KVCache后, 请求以连续的批处理方式加入到下一批. 同时需要注意在Decoder节点选择时需要考虑负载,并且确保不违反TBT SLO.

1.2是亮点, 另外几个很重要的工作是MulitNode Prefill,即采用CPP(Chunked Pipeline Parallelism)/SP等在Prefill阶段多机多卡并行计算. 以及Layer-Wise Prefill用来Overlap和降低存储/传输等开销. 这一点和以前讲过的Nvidia的PROACT处理类似, 降低Bulk DMA的通信延迟暴露

![图片](assets/877abb550fb0.png)

另一方面是KVCache的调度上花了很多功夫,当然一开始还是有一些Cache颠簸带来的性能抖动

![图片](assets/90abd1205e59.png)

章明星老师的分析是:由于 Prefill 和 Decode 集群负载之间的时间差，如果简单的参考当前 Decode 集群负载去拒绝请求的话会导致 Decode 集群负载被消化的时候 Prefill 没有及时跟上，由此产生了跷跷板效应。因此调度器做了一些基于预测的调度, 并且根据请求预测TBT, 并决定是否要Early Reject.

## 2. 从体系架构的视角看云AI基础设施的分离式架构

### 2.1 KVCache的开销

其实在整个过程中我们发现KVCache的缓存策略等处理逻辑和通信处理都需要CPU上的进程. 数据从GPU拷贝到CPU上,然后通过Messenger拷贝到远端Decoder节点CPU上,再异步加载到GPU上. 从体系结构来看GPU本身无法处理这些通信事务的开销,导致整个系统又有点像TCP那样Userspace到KernelSpace的拷贝...

那么第一个问题来了: 如何实现KV-Cache的灵活管理并且Zero-Copy的能力? 那么是否可以把Oracle RAC中的CacheFusion和IBM PureScale中的Cluster Caching Facility抄一个作业,当然LLM推理过程中基本上没有太多的一致性需求,因为数据都是增量的方式,而且Decoder节点为PrefixKVCache只读. 只是在KV传输和置换的过程中需要一定程度的Fence.

另一方面,我还是持有以前的观点,要把GPU和内存升级为整个系统的一等公民, 也就是华为最近在CloudMatrix提的对等连接. 本质上是一回事.
![图片](assets/c442c83d583f.png)

和华为UBMA相对来看的区别是,NetDAM当时的实现是相对来说弱化了Cache一致性的实现来获得更大规模组网和延迟容忍能力的. 当然华为的CloudMatrix和弹性内存供应也是一项非常不错的技术.

### 2.2 调度

从调度的视角来看, 不同优先级请求带来的Prefill/Decoder的加权调度并且维持整个系统的SLO,同时针对一些付费用户提供更好的SLO. 实际上来看就是类似于进程间调度, 而另一方面Messenger类似于进程间通信. 但是相应的通信带来的中断和进程切换(例如Decoder GPU置换出不需要的KVCache并添加下一个请求的Prefix KVCache等), 实际上在分离式架构上Datacenter as a Computer的OS雏型已经逐渐的开始显现出来了. 这也是我最近一两个星期正在做的一些工作, 正在做一些基于排队论的队列调度算法分析.

从软件系统架构来看Ray的一些架构是否可以进一步复用出来,这是一个值得探索的问题. 当然再细致一些是否还可以加入一些带有更强算力的CPU实例来进行混合调度呢? 例如在decoder阶段10个CPU实例 vs 1个H20 的GPU实例的TBT分析? 我相信在Intel/AMD上做到和H20差不多的浮点运算能力是非常容易的事情, 例如未来的AMD Turin AI. 这样就构成了H100/H800 Prefill, CPU Decode的异构部署.当然这会进一步加重调度的负担...

当然从调度的视角还要考虑异构网络拓扑相关的通信负担和拥塞. 因此在调度的视角来看更加对等的拓扑连接CPU/GPU/Mem也是一个必须要去探索的事情. 另一方面伴随着调度的出现,通信将会和LLM训练网络有很大的不同, 因此拥塞控制的考验将会更大, 当然这些问题已经考虑清楚并解决干净了, 对于其它云厂家可能还要面临一些挑战.

### 2.3 系统

从系统架构来看, 我还是建议基于以太网的ScaleUP架构, 并且把CPU和内存作为一等公民加入到这个ScaleUP网络中, 然后构造出很薄的一个Memory ShimLayer, 这样带来的区别就是ScaleUP网络和ScaleOut以及FrontEnd的融合, 而现在的架构导致三张网络相互独立,自然而然就要出现大量的内存拷贝和搬运成本了.

### 2.4 GPU架构

从GPU架构来看,SIMT做简单的并行性任务可以,但是面对复杂的调度和通信纵然有一些GDA-KI的机制,但是GPU来做KVCache的管理和传输还是太复杂了, 有机会的话,在GPU上像AWS那样添加一些通信协处理器可能是必不可少的, 具体要做什么涉密就不多说了. 

### 2.5 算法

Mooncake的工作和MLA这些实际上是正交的, 我们还可以考虑在算法上进一步的创新. 我并不认为一些量化的方法可以来解决当前的一些带宽和算力的瓶颈, 有些得不偿失的感觉, 而我一直看好的一个方向是Speculative Decoding, 有点类似于传统CPU的分支预测机制.

不过在这之上为了提供客户对高质量答案的需求, 大参数的Dense模型可能并不是出路, 某种意义上需要从现在的Decoder-Only模型再一次回归到Decoder+ Auto-Encoder的方式来处理. 例如GNN-SAE Adapter这样的方案,通过SparseAutoEncoder抽取LatentSpace的Concepts,然后通过GNN来约束Concepts进行更高质量的Token输出.

![图片](assets/80c763301e84.png)

当然这样的算法可能更需要GPU和CPU的紧耦合和通信事务上更好的异步操作来隐藏延迟.

## 3. 后记

本文只是浅显的介绍了几个方面的工作, 一些详细的细节涉密就无法展开了.... 但是这是一个最好的年代, 各个层面都一路狂奔~