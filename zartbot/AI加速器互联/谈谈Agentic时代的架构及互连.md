# 谈谈Agentic时代的架构及互连

> 作者: zartbot  
> 日期: 2026年4月5日 01:38  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498097&idx=1&sn=1aaf645abb0f09f6dc6a50c5714ed54b&chksm=f995e9b3cee260a5bac8e9da52a6fdfb75904507f7aea013355d4581fb22d3e6345b32e86361#rd

---

`本文仅代表作者个人观点, 与作者所任职的机构无关`

English version : https://github.com/zartbot/blog/issues/10

### TL;DR

最近ODCC的会上, 看到腾讯牵头的一个I/O NET工作组的一些介绍还是不错的. 特别是夏博的两页介绍. 原始的照片清晰度很差, 我大概转述一下:
AI网络的三个阶段
第一阶段: `ScaleOut`:以构建单任务万卡/十万卡的`训练`任务集群

第二阶段: `ScaleUp`: 用于高性能`推理`的超节点, 追求低延迟和处理MoE通信.

第三阶段: `Agent Fabric`: 用于Agentic LLM的平台. 如下图所示:

![图片](assets/6d3cec90bb71.png)
Agent Fabric主要的任务:
`内存池化项目`, 使用ScaleUP支撑突破内存墙**

平滑扩展 HBM 内存

统一适配各种存算比**

`ScaleUP IO Die与IP`, 支撑Scale UP产品化落地, 主要包括GPU / CPU / Memory 多类器件接入ScaleUP

`数据中心网络融合项目`, 用于满足Agent多业务的互连网络, 包括:

ScaleOut / FrontEnd / Storage 融合

ScaleUP 和 ScaleOut 融合

`端网协同项目`, 主要是解决高效高质量的多厂家互通的网络系统

多厂家网卡互通

网卡与交换机协同技术

新型存算一体 IO-NIC

`AI网络系统级可靠性项目`

当然这些是来自网党的视角,  其中的一些问题和解法在以往的很多篇文章都解释清楚了的, 这篇文章来对Agent时代的互连进行一个详细的归纳和分析. 我们先从CPU/GPU以及存储的视角独立的进行分析, 最后把它们放在一起看互连, 本文目录如下:

```
1. Agent时代的CPU架构及互连1.1 CPU的两种形态及Vera的一些缺陷1.2 CPU I/O的问题1.3 一些改善的建议2. Agent时代的GPU架构及互连2.1 GPU微架构2.2 GPU互连3. Agent时代的存储架构及互连3.1 存储语义3.2 存储互连4. Agent Fabric4.1 总线协议4.2 系统架构4.2.1 GPU互连系统4.2.2 CPU互连系统4.2.3 AI-NIC和DPU架构5. 总结
```

## 1. Agent时代的CPU架构及互连

相对于传统的LLM训练和推理任务, 最大的变化是Agent Sandbox的引入, 无论是RL训练还是Agent workflow, 通常需要几十万核到数百万核的CPU集群参与.

### 1.1 CPU的两种形态及Vera**的一些缺陷

从workload来看, CPU本身在两种不同场景下的需求是有巨大差异的:

针对在作为GPU控制节点的CPU和在RL训练时承载Agent Sandbox的CPU需要更高的单核的IPS(Instruction Per Second), 同时需要更大的带宽用于KVCache等传输.

针对作为Agent Sandbox的CPU需要更高的密度和更低的功耗, 用于高密部署.

其实这也就把CPU推向了两个极端. 显然基于Nvidia Vera的单一CPU是无法满足的. 针对单核高IPS需求, 我们在[《Inside Nvidia GPU: 谈谈Blackwell的不足并预测一下Rubin的微架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21#wechat_redirect)中的2.3节提到过

虽然在Hopper这一代就引入了NVLink C2C, Grace能够和Hopper或者Blackwell直接NVLink连接. 但是Grace的CPU也有不少的问题.  实际上伴随着Blackwell算力越来越强, 很多Kernel的运行时间降低到微秒级别, 这就产生了一个非常经典的 Killer Microsecond问题. 对于ns级别的问题, 同步等待就行了. 对于ms级别的时间, 上下文切换的代价也不大. 但是当到了 us 级别, 其实对于处理器而言已经有很大的挑战了. 虽然引入了很多异步的编程优化, 但是当前Grace这类的CPU还是面临很多瓶颈. 一方面是Kernel Launch的速度不够快, 虽然可以狡辩通过cuda-graph或者一些persistent kernel的方式来解决. 但并不是所有的workload都满足这个条件.

另一方面是一些Grace微架构的缺陷. 虽然Grace使用了ARM当时最强的Neoverse V2的Core, 它的设计上并没有采用V2所使用的2MB L2Cache, 而是裁剪到了1MB. 相比之下,同样使用V2 Core的AWS Graviton 4 采用了2MB L2Cache. 当前有些客户遇到了GB200上Grace L1 ICache Miss的问题很大程度上与这个有关.

我们注意到在Grace的CPU设计上有很多缺陷, 导致GB系列在很多时候由于CPU的限制出现了一些瓶颈. 而Vera能改善多少, 很难说, 我们来对比一下:

![图片](assets/22ba98fd625d.png)

针对Grace CacheMiss的问题, Instr Decode增加到了10way, L1D Cache也增加到了6way, L2Cache的容量也增加到了2MB, 或许就是NV说的,专门针对Grace的问题提了一下PyTorch Optimized Instruction Buffer.  然后官方的资料显示IPC提升1.5倍, 并且包含分支预测器的改动,
![图片](assets/770f85310c03.png)

但还涉及一些Cache和NOC的问题, Nvidia还需要很大的精力去解决它们.

而另一方面针对高密的场景, Intel Clearwater Forest已经有288核, 双Socket达到576核.  AMD Venice也能达到单路256核, 双Socket 512核. Vera的密度还是有很大差距的.

而对于AMD和Intel都有高性能核(I:P-Core/A:Zen6)和高密度核(I:E-Core/A:Zen6c)的选择, 这一点上Nvidia还有不少问题. 试图使用一颗Vera来解决GPU主控/ Agent沙箱 以及DPU主控(BF4)三个场景, 显然是极不合理的.

### 1.2 CPU I/O的问题

我们以AMD Venice为例, 它基于SP7(LGA-9324)有9,324个pin, 预计支持16 Channel DDR5, 相对于SP5增加了53%的针脚, 这主要是为了支持增加的内存通道数和更高速的 I/O 通道(PCIe 6.0),同时提供更高的供电能力以支持 Zen 6 核心. 并且还有大量的供电和接地的pin要使用.

我们从单个CPU Core的视角来看, 通常有一个经验是1GHz大概需要1GBps的带宽, 对于一些数据密集型的场景甚至需要10GB/s才能匹配1GHz的处理节奏. 那么对于一个256核 3GHz的处理器而言, 例如采用16通道MRDIMM大概有1.6TB/s的内存带宽, 实际上还是有很大的内存墙的影响. 另一方面Socket之间的互连和对外的PCIe互连也需要更大的带宽.

另一个是存储带宽的需求, 例如在AgentSandbox场景中, 为了保证实例快速启动, 以及最近还有一些探讨针对Agent长时间多轮执行还需要执行更多的checkpoint和回滚的能力, 通常需要更高速的存储带宽. 当前Agent Sandbox还没有涉及到一些数据密集型的业务处理, 未来可能还会进一步加大网络/存储带宽的需求.

具体的需求, 我们可以参考一下ASPLOS 2026的一篇论文《AgentCgroup: Understanding and Controlling OS Resources of AI Agents》[1], 我们可以看到几方面的需求:

56-74%的端到端任务时间被容器/Agent初始化和工具调用等OS层面的执行所消耗. 另外Agent的容器镜像平均大小为3.5GB, 从快速启动的视角来看, 虽然有Lazyload的能力, 但是如果整机几百个CPU核同时并发执行2000个Agent容器, 大概的启动镜像对存储的带宽需求也会非常大.  例如需要单机400~800Gbps的存储带宽.

Agent的平均CPU利用率很低(单核标准化后低于14%), 但峰值内存需求可达2-4GB. 这意味着在多租户云环境中, 限制并发实例数量的关键资源是内存, 而非计算能力.

内存使用表现为一个约185MB的稳定框架基线, 叠加由工具调用驱动的、短暂(1-2秒)且剧烈(峰均比高达15.4倍)的资源脉冲.

注: 关于Agent场景下的RL训练, 推理, Agent Sandbox的执行可能还需要更多的协同设计, 例如对于一些coding任务的训练还是需要很多高频率高性能的CPU核, 同时又需要很大的规模部署, 加快Rollout速度, 我们后面会通过其它文章详细进行分析.

那么对于CPU的I/O需求来看, 内存突发的带宽需求很高, 同时叠加着对存储/网络高吞吐的需求. 短暂的峰值资源需求使得整个CPU需要更高的I/O. 但由于Socket之间互联的UPI/XGMI需要占用带宽, DDR需要占用带宽, PCIe需要占用带宽, 还有部分的CXL lane想继续做内存池化, 另一方面和GPU互连还要更大带宽的NVLink C2C这样的接口.

CPU出来的pin便受到很大的约束. 因此对外提供更高速的接口(例如224G Serdes)或许是更好的一个选择, 而PCIe Gen6**才64GT/s, 即便是Gen7也才128GT/s, Gen8落地不知道要到什么时候...

### 1.3 一些改善的建议

实质的问题就是, 割裂的总线协议会对CPU带来很多SKU和不同的互连瓶颈. 我们需要一个统一的总线(在暗示用Huawei的Unified Bus么?), 既能够连接CPU-CPU, 也能够连接CPU-GPU, 同时还能够DPU/NIC. 此时就有几种选择:

`CXL行不行?` 不行, 因为大家都被CXL忽悠瘸了, 搞了那么多年没见多少正经落地的. 同时带宽上和PCIe一样慢慢演进挤牙膏. 同时我们还要考虑GPU侧的情况, 有几个GPU Vendor愿意支持CXL呢?

`NVLink C2C行不行?`这个问题比较有趣, 首先Nvidia自己的Grace和Vera都还有不少问题的, 对于作为GPU控制的CPU还需要更高的IPS, 虽然Nvidia投资了Intel, 可能Intel会提供基于NVLink C2C的X86芯片. 但是网卡呢? SSD这些呢? 工业界是不是要NV的做一套, 非NV的再做一套呢?

`UALink行不行?`, 同样的问题, 整个工业界在不同总线之间站队或者各做一套显然是不合理的.

其实任何一个开放的协议组织能够成功, 总归有一个在市场地位占第一的大哥带队. 当年的PCIe如此, 如今这个总线也是如此. 过去若干年什么Gen-Z / OpenCAPI / CCIX啥的没见一个成功便是这个原因.

剩下一个问题, PCIe都用了这么多年了, 能不能改改继续凑合用, 为啥要搞一个新的总线? 原因是要考虑下面这一系列互连场景

![图片](assets/69a431cf6253.png)

`CPU-CPU`: 首先CPU之间的互连或许每家还是有自己的选择, UPI/XGMI等有根据自身CPU微架构和NOC的很多优化, 没有太多统一多厂商互连的标准总线的需求. 但是我们也观察到了NV针对CPU-CPU和CPU-GPU都采用了相同的NVLink-C2C. 统一的总线也有它更多的灵活性.

`CPU-Memory Pool`: 然后我们考虑一下CPU和内存池以及GPU和内存池的互连, 如果暴露一个通用的接口, 延续CXL的故事进一步构建一个大的内存池, 无论对于GPU节点的模型推理所使用的KVCache, 还是为了解决Agent Sandbox的内存资源脉冲都是有价值的. 从供应链的角度来看自然是希望有能够共享的模组. 这里就有一些约束了, 我们可以看到GPU厂商基本上没有一家会考虑支持CXL, 直接在原来的ScaleUP总线上挂载内存池不行么?

`CPU-DPU`: 另一方面是针对CPU和DPU的互连, 现在以太网侧的带宽演进速率远超PCIe, 并且高密度的CPU(256core~512core)演进也需要更大的带宽. 例如1.6Tbps的NIC当前只有接2个PCIe Gen6x16, 然后在这两个PCIe上还会存在一些Ordering的问题.

`GPU-AI_NIC`: 对于GPU和AI-NIC的互连, 如果采用不同的协议, 那么GPU侧也需要按照某个固定的ScaleUP:ScaleOut配比以及两套语义来进行通信.

如果我们需要灵活的互连配比, 各种芯片更好的方式是构建一个相对统一的总线协议. PCIe的问题是它基于RC的树状拓扑, 虽然在CXL上也逐渐有了Port-Based-Routing这样构建交换网的能力, 但是PCIe自身的Controller IP和相对演进缓慢的Serdes也成了一个问题. 那么一个简单的建议是进行控制路径和数据路径的分离, 把PCIe保留1x lane作为控制链路用于维持软件层面上的兼容性, 然后数据路径上采用更高速的总线.

我们也得到一些消息未来CX10也会选用NVLink C2C这类的接口来避免使用PCIe. 其实我很早就在建议NIC本身需要挂载到ScaleUP总线上, 并认为这是一个更加正确的演进路线.

假设我们这个总线叫New Unified Bus(NUB). 对于CPU-CPU, CPU-GPU, CPU-Memory Pool 以及 CPU-DPU 去构建Cache-coherence的一个总线还是有很多业务价值的. 而对于GPU-GPU 以及GPU-AI_NIC 则无需使用Cache-coherence. 但是这个NUB和huawei的UB有一些不同点是, 为了考虑GPU的DieSize占用, 我们不能在它之上引入一些类似于RDMA的消息语义, 一个很简单的逻辑, 单个GPU的die大概约束在, 而一颗800Gbps的RDMA NIC die差不多要,所以我一直反对在GPU ScaleUP总线上引入消息语义. 个人的观点是在超过Rack范围后, 还是维持RDMA通过消息语义传输, 如下图所示:

![图片](assets/0b757444a996.png)

**结论: 针对Agentic时代, 我们需要给CPU一个相对统一的支持Cache-Coherence的总线, 并且使用更高速的Serdes** 当然这件事还涉及到一些商业上的博弈, 例如AMD的UALink联盟发展情况, Broadcom针对这一段互连的态度. 更重要的是Nvidia是否愿意在这个域内构建一个相对开放的总线组织. 另外还有一些不可忽视的力量, 就是北美的几个HyperScaler的云厂商, AWS / Google / Azure 都有自己的ARM CPU. 他们也有类似的业务诉求.

## 2. Agent时代的GPU架构及互连

### 2.1 GPU微架构

对于Agent时代GPU的架构也会发生一些变化. 显而易见的就是这次GTC发布的基于Groq LPU的方案. 我们需要进一步降低推理的延迟, 提高Token per second. 以前一篇文章也详细阐述过[《Inside Nvidia GPU: 谈谈Blackwell的不足并预测一下Rubin的微架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21#wechat_redirect), 老黄也在NV内部转发过.

一个核心的观点就是SM内的一些warp scheduler需要通过一个标量核暴露出来. 通过它能够更高速的控制mbarrier把TensorCore和TMA以及CUDA Core作为协处理器的方式来运行, 这样对一些MegaKernel的实现(eg. Mirage...)来降低整体的GPU推理延迟是有巨大帮助的.

这里其实有一个很大的误区, 包括和很多NV的同学聊到这个话题时. 按照习惯性的思维, 历史上GPU是一个高吞吐但不那么太在意延迟的并行计算处理器, 因此很多时候衡量的标准是只要所有的计算单元能用满即可. 但是在Agentic LLM时代更多的有了低延迟的需求, 为了保证TPS SLA, GPU不需要按照很高的batch-size去用满计算单元或内存带宽, 而是在一个相对较高的利用率下, 通过Warp调度和指令issue来降低延迟.

### 2.2 GPU互连

我们注意到现在Rubin+LPX还是采用ScaleOut网络进行互连的, 未来也会统一到NVLink上. 另外一个Agent时代带来的变化就是对GPU接入内存池或者外部存储, 平滑扩展 HBM 内存以及统一适配各种存算比有了更高的要求. 当然通过HBM连接一些板载LPDDR也是一条路. 但是考虑到巨大的ScaleUP总线带宽和更灵活的存算配比, 或许将一些内存池接入到ScaleUP总线是一个更好的方案.

然后有一个还存在争议的地方就是ScaleUP和ScaleOut融合, 这里涉及到计算团队和网络团队的一些争议. 计算团队普遍认为ScaleUP网络构建多层交换机即可, 但是多层交换则会面临负载均衡/可靠传输的挑战. 即便是Nvdia发布了基于Orben NVL576或者基于Kyber NVL1152的方案, 个人认为在可靠性方面还存在很大的挑战.

并且从客户的视角来看也需要灵活的配比需求, 也就是说这个第二层的ScaleUP网络存在明显的收敛比, 主要原因是数据处理总有它的Data locality, 更大范围不一定需要1:1的无收敛的网络, 那么相应的拥塞控制/可靠传输, 实质是要靠额外的算力来填补. 具体的分析可以参考[《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect)

## 3. Agent时代的存储架构及互连

对于Agent时代infra最大的一个变数是更高性能存储的需求. 传统的存储通常是在FrontEnd网络中接入. 而DeepSeek DualPath逐渐的将其引入到了ScaleOut网络, ODCC I/O NET工作组也在谈论一个话题, 即ScaleOut / FrontEnd / Storage 融合. 但是这件事情非常难, 至少对于Nvidia这些CX系列的RDMA网卡来看.

### 3.1 存储语义

从存储语义来分析, 通常为两类, Nvidia有一页进行了对比

![图片](assets/bc7048ac3027.png)

对于GPU编程来说, 基于Block可能对GPU的效率有影响, 因为KV可能涉及到太多的block, 占用SM资源.  基于File, 复杂的文件系统在CUDA Core上执行显然也是不太行的. 如果需要GPU Initial Direct Storage(GIDS)最佳的接口还是需要采用NVMe KV cmdset.

### 3.2 存储互连

从传输协议来看, ScaleOut / FrontEnd / Storage 融合的第一个难题就在PCIe总线上, 正如DeepSeek DualPath谈到的, PCIe本身本有很好的QoS机制, 带宽又很低, 因此和ScaleOut融合很容易影响集合通信. 那么在下一代协议设计上, 如果AI-NIC或者SSD能够挂载到ScaleUP总线上, 这样的处理方式会简单很多, 动态的存算配比也可以很容易实现.

## 4. Agent Fabric

我们来回顾一下I/O NET定义的 Agent Fabric 主要的任务:

`内存池化项目`: 我们需要使用ScaleUP支撑突破内存墙, 实现平滑扩展 HBM 内存和统一适配各种存算比

`ScaleUP IO Die与IP`: 支撑Scale UP产品化落地, 主要包括GPU / CPU / Memory 多类器件接入ScaleUP

`数据中心网络融合项目`: 用于满足Agent多业务的互连网络, 包括:ScaleOut / FrontEnd / Storage 融合, ScaleUP 和 ScaleOut 融合

`端网协同项目`: 主要是解决高效高质量的多厂家互通的网络系统, 包括多厂家网卡互通, 网卡与交换机协同技术, 新型存算一体 IO-NIC的设计

`AI网络系统级可靠性项目`

把这些放在一起, 如下所示:

![图片](assets/d3c93339b4ca.png)

### 4.1 总线协议

从拓扑连接来看似乎也没有什么特别的, 但是关键都在一些细节中. 最关键的一点, 也是争议最大的一点是, 总有架构师会想着一个协议统一全部互连,但是One size never fit all. 因此从协议上而言, 需要有很明确的区分才行, 如下图所示:

![图片](assets/c92d461472a7.png)

首先对于内存标准演进, 还是由JEDEC负责就好, 虽然Nvidia在Feynmann会引入Custom HBM, 这部分的内容还包括通过HBM attach LPDDR等技术路线或者一些Optical Memory接口. 本文就不展开分析了.

比较大的一个争议是内存语义和消息语义, 特别是Ethernet based ScaleUP以及对应的ScaleUP + ScaleOut融合的讨论和NVLink/UALink这样的ScaleUP协议之间的对比. 个人的观点是**Chassis Level和Rack Level还需要一个新的统一总线, 而在Rack之间的互连沿用基于Ethernet的消息语义协议(eg. RDMA)**. 为了方便描述, 我们把前者称为Intra-Host Protocol, 后者统称为Inter-Host Protocol.

如果希望一个统一的协议融合ScaleUP(Intra-Host)和ScaleOut(Inter-Host)是很困难的, 这个问题在5年前做NetDAM的时候就分析得比较清楚了, 直接扩展Inter-Host协议来支持内存语义, 较小的Flit size对于可靠传输协议及拥塞控制的代价是巨大的,同时对于IO Die的面积占用也大很多, 在GPU上实现较为困难. 同理直接扩展Intra-Host协议支持经过多跳交换机的组网, 对于可靠传输的实现需求会带来更多的代价.
延迟分析
首先我们来分析延迟的差异:Intra-Host 通信协议通常只有小于200ns的固定传输延迟，而Inter-Host以太网通常为数个微秒的延迟，并由于包调度和多路径及拥塞控制等原因会带来不确定性.

其实这两种思路正好也反映在SUE和UALink两种协议的设计区别上. 在SUE中使用加速器侧的队列区分到不同目的地的GPU流量, 并通过Round Robin调度发送到交换机. 交换机也按照Packet进行转发. 如下图所示:

![图片](assets/29c8997a5fe2.png)

我们注意到在EP的情况下会出现多个GPU向一个GPU发送token, 因此就产生了incast的情况. 由于交换机也是按照Packet调度发送到目的GPU的Egress 队列, 因此某个源GPU发送的数据在这种情况下将带来很大的tail-latency, 例如上图中源GPU-4发送的数据由于Egress队列Buffer已满, 最终产生的长尾延迟可能会接近10us. 对于传统的GPU架构为了更好的利用内存带宽, 即更高的GEMM效率, 通常需要足够大的batch-size才进行Expert FFN计算. 计算延迟可能会到几十个微秒, 然后通过Two-Batch-Overlap Attention计算, 因此这样的长尾延迟影响并不大.

但是在Agentic LLM时代, 我们需要更高的TPS. 例如在采用LPU处理时, Expert的参数在SRAM中, 因此小batch-size也可以很快的完成计算, 计算本身的延迟大概只有2us以内. 那么通过ScaleUP网络的长尾延迟就会产生一个很显著的影响了.

对比UALink, 如下图所示:

![图片](assets/14ad19eaf100.png)

发送端并不区分到不同目的地GPU的数据, 直接按照64B TL Flit打包成一个640B Datalink层Flit即可发送到交换机. 交换机会解析其中的TL Flit并将其发送到Egress队列, Egress队列也是凑满10个TL Flit就可以直接发送到目的GPU. 这样的方式长尾延迟可以控制在2us左右,
可靠传输分析
在一个相对短距离(Rack Level)并仅支持单层交换机网络, 可靠传输的问题是非常简单的. 如果经过多层交换网络, 长距离传输使用光的可靠性, 整体系统的MTBF以及交换网络的负载均衡都需要考虑. 另一方面实际的延迟即便只计算数据在光纤中的延迟都需要超过2us.如果出现packet loss的情况, 如何恢复也会带来很复杂的处理逻辑.
MMU的考虑
前几天也在和AMD的同事交流了一个有趣的场景, 对于超大规模组网的内存语义, 内存地址排布应该如何处理. 其实还需要进一探讨另一个问题, 如何在一个ScaleUP+ScaleOut融合的域内动态的增加/删除节点来维持整体系统的可靠性和集群的动态伸缩弹性? 当然具体的技术细节涉及一些机密就不再展开了.
结论
我们需要在Rack Level构建一套新的总线协议来支持内存语义, 对于Inter-Rack可以沿用基于以太网的一系列类似于RDMA的消息协议. 对于物理层和连接器, 可以统一到OIF CEI-112G/224G上来降低成本并维持快速的演进.

### 4.2 系统架构

#### 4.2.1 GPU互连系统

现在GPU的互连如下图所示:

![图片](assets/217e322dbe9e.png)

当GPU需要更大的存储访问需求时, 存储通常需要FrontEnd网络经过CPU再到GPU, 对于NVidia来看, CPU和GPU之间有NVLink C2C支持. 但是如果将Storage引入更大带宽的ScaleOut网络, 潜在的问题还很多: 首先是PCIe缺乏很好的QoS支持, 存储流量很容易干扰到集合通信. 另一方面是拥塞控制/负载均衡/可靠传输这一类RoCE的老问题. 当然还有新的业务需求, 例如动态的可调整的存算配比需求等. 这也带来的一个新的问题:GPU如何构建灵活的ScaleUP和ScaleOut配比? 当前固定的PCIe和NVlink配比直接约束了这种可能性. 如果采用统一的ScaleUP协议, 将AI-NIC挂载到ScaleUP总线上, 这个问题便可以很容易的解决.

但此时有两种选择, 第一种选择是在GPU上集成RoCE IP, 例如微软/Meta这些公司的加速器, 然后基于Ethernet ScaleUP构建.

![图片](assets/fb0bd2a1eb51.png)

但是我们注意到这样的解决方案会导致加速器内大量的芯片面积被占用, 通常需要额外的NIC I/O Die来进行互连. 但是考虑到功耗和带宽的需求, 这些基于RoCE的IP在可靠传输/负载均衡上也有很大的缺陷. 最终会接增加ScaleUP的延迟. 那么我们为什么不考虑另一种方案,将AI-NIC连接到ScaleUP Switch上.

![图片](assets/beaa6de1e9e3.png)

这样的好处是GPU-GPU之间可以维持原有的ScaleUP总线的低延迟, 而AI-NIC可以根据存算配比放置(例如1:4 / 1:8), 同时Ethernet网络也可以根据实际的存储节点带宽构建带有收敛比的网络.

#### 4.2.2 CPU互连系统

对于CPU节点, 它需要类似于NVLink-C2C的总线去连接GPU. 但是现有的架构基于PCIe连接存储/网络也会遇到PCIe带宽不够的瓶颈, 特别是在未来单颗CPU达到512核的时候, 并且考虑到Agent Sandbox的工作特性, 它会有很大量的内存突发使用, 并且在Multi-Turn Agent执行过程中, 我们可以有足够的时间把一些内存页置换到外部内存池中. 例如Agent Sandbox A在执行完成后, 需要将Tool call的结果传回推理引擎时, 它的内存可以置换到外部存储, 而Agent Sandbox B刚拿到LLM返回的下一段执行代码时, 可以将内存从共享内存池中取回. 因此额外的Mempool或者更极致的采用外部的SSD存储都可以节省内存容量的需求. 因此我们可以构建如下的互连系统:

![图片](assets/892acd414284.png)

#### 4.2.3 AI-NIC和DPU架构

这次ODCC会议也谈到了新型存算一体 IO-NIC的设计, 以及多协议多厂商互通和端网融合的问题.

对于多协议互通, 实质上因为拥塞控制/负载均衡/可靠传输这一类RoCE的老问题, 几年时间过去了并没有得到很好的解决, 也导致了每一家企业都在自定义自己的RoCE协议, 例如OpenAI MRC, Google Falcon, AWS EFA-SRD, 微软和Meta也有自己的协议, 国内也有VeRoCE等等... 但是发现一个很大的问题是, 很多协议过了几年的演进还并不成功, 因此, 这次ODCC会议也有一些Session还在反复讨论这个问题, 例如多厂商互连,交换机和网卡协同, 跨AZ需要超大Buffer的交换机等议题.

正好会议期间某个阿里云的客户基于CIPU eRDMA测试了一个跨数据中心的集合通信(经过9跳交换机网络), 很轻松的跑满了带宽, 这是3年前就已经解决好的问题, NV(Mellanox)却一直没办法根治, 其实根源在于Nvidia CX系列网卡的微架构设计, 而 NV 和 Hyperscaler 普遍基于CX系列微架构设计的这些协议都无法很好处理这些问题.

很简单的一个例子, Nvidia BF4上采用了多个组件, 基于Spectrum的交换机IP, 新的PSA处理器, 已有的DSA处理器, 以及Vera通用CPU来处理存算一体业务. 整体功耗几百瓦. 真的需要这样做么? 实质还是它们RoCE本身协议的缺陷和自身那个演进很多年的RoCE IP带来的缺陷. 同时还需要在交换机上支持PacketSpray,垮AZ还有一些需要更大buffer的交换机芯片等不停的修修补补, 几年了问题都没有得到解决.

## 5. 总结

其实挺期待Nvidia能够带领一系列CPU厂商(Intel/AMD/ARM/Google/AWS)定义一个新的总线, 当然作为CPU的总线还是需要支持一定程度上的Cache Coherence, 或者简单的支持一下类似于CXL的Backward-Invalidation即可. 同时这个协议也可以在Non-Cache Coherence的模式下用于GPU ScaleUP. 当然NVidia也在授权NVLink Fusion这样的方案. 商业逻辑上似乎很难让它授权给AMD, 但是今年年底AWS Trainium 4会同时支持NVLink和UALink, 如果两者性能上没有明显差异, 那么互连总线的标准就会很快向UALink倾斜.

我相信Jensen还记得当年3Dfx坚持Glide API, 最后被 NV/ATI 等一众支持DirectX/OpenGL的公司取代的事情吧. 生态的力量是巨大的, 当总线的标准开始往其它地方倾斜的时候, 又叠加上生态从SIMT CUDA到Tile Based programming转变之时, 甚至是一些类似于Nvidia AVO的工作通过Agent能够优化Kernel性能时, 或许NV的壁垒就再也不存在了...

参考资料

[1] 
AgentCgroup: Understanding and Controlling OS Resources of AI Agents: *https://arxiv.org/pdf/2602.09345v2*