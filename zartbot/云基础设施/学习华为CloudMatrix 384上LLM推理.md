# 学习华为CloudMatrix 384上LLM推理

> 作者: zartbot  
> 日期: 2025年6月17日 23:13  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494266&idx=1&sn=acaf668be2c0b1d49816010eddf8a61c&chksm=f995fab8cee273ae61d732e933e74eecb7c6af5c1d379a202f6585832429010c5052209606b6#rd

---

### TL;DR

上周日,华为和硅基流动联合发表了一篇论文《Serving Large Language Models on Huawei CloudMatrix384》[1], 干货特别多,诚意特别满, 向这两个非常优秀的团队致敬, 同时仔细学习了一下, 记录了一些笔记分享出来.

特别值得注意的是huawei CloudMatrix的架构设计, 体现了华为重塑AI Infra的很多愿景, 并且通过很长一段时间的工程积累将其实现了, 从整体架构上来看, ScaleUP UB架构做的很好

![图片](assets/0c8a970f7bbc.png)

同时下面这个图对于云基础设施是有巨大的价值的, 比起UB-Mesh的拓扑, 可能我会更喜欢这样的架构:

![图片](assets/203b53edd46d.png)

技术路线上和我以前做的NetDAM是类似的, 不过可能是长时间在数通领域的经验, 我更希望实现一种能够平滑从现有数据中心架构花几年时间迁移的方案, 毕竟推倒重来和新建孤岛带来的资源浪费是巨大的. 因此几年前我也在现在任职的公司内部分析过如何从传统数据中心平滑演进到这样的架构, 同时我一直也在猜测并建议NV在Rubin上把ScaleOut加入ScaleUP域, 但是很可惜NV这次展出的Rubin机柜还是没想明白...无力吐槽了...

![图片](assets/3b497253441d.png)

敲重点, NV的ScaleUP和ScaleOut是两个团队, 老黄真能搞定这两个巨大的部门墙么. 即便是搞定了, 在工程上NV并不是最一流的数通公司, 还会碰很多壁....

在这里面不光是UB总线的设计, 还有EMS/EVS很多细节的设计上做的也很好, 存储/内存的融合对KVCache很有好处, 后面详细展开分析一下.
另一方面还有硅基流动和华为一起在软件层上的大量的细致的优化工作, 都是非常值得学习的. 在进行DeepSeek-R1推理时, Prefill性能达到6688 tokens/每卡, Decode性能达到1943 tokens每卡. 非常了不起的成绩了.

原始论文很长有59页, 所以可能会分成两篇来分析, 第一篇主要集中在硬件Infra相关的话题上.

### 1. 概述

#### 1.1 从模型的视角来看

原文Introduction章节写了一些模型的变化, 一方面是MoE这样的模型稀疏化的带来的专家路由和同步的开销, 另一方面是上下文窗口扩展后导致的KV Cache的存储压力. 这些在推理系统中对整个系统的可扩展性/延迟/带宽都带来了很多更加苛刻的要求.

另外还有一些真实世界里的负载不均衡现象, 例如每个请求的输入输出是不平衡的, 甚至还会出现一些高突发的用户查询等, 同时专家的激活是不平衡的, 在这一系列不平衡的分布下, 需要保证严格的延迟和吞吐量的目标. 因此对于整个集群就要有更全面的软硬件协同设计, 智能调度/弹性资源管理/灵活的编排/更紧密的算子融合/通信和计算的细粒度Overlap等..

而根据这一系列需求, 原文的这句话非常认同: it becomes essential to reimagine the design of AI infrastructure from the ground up. 对于AI Infra和传统Infra的区别, 一切都要从最根本的应用需求出发, 然后尽量不要去针对workload本身做过多的垂直优化, 而是建立在更底层的通用抽象上来处理.

#### 1.2 CloudMatrix 384设计原则

针对前面的这些需求, 华为构建了CloudMatrix这样的下一代的AI数据中心架构, 并且强调了这是一个Production的方案. 384个Ascend 910c NPU和192个Kunpeng CPU通过UB超高带宽和超低延迟总线链接. 并且这个UB网络能够在所有计算和内存组件之间直接进行all-to-all的数据交换. 它和传统架构中把GPU挂载二奶位置的层次化汇聚架构相比, 整个CM384中所有节点是对等的."everything can be pooled, treated equally, and combined freely". T

其实实质的是基于一个内存总线, 把所有的资源都拉平, 无论是CPU还是GPU, 或者是一些其它的内存资源/存储资源等, 都拉平到一个大的超节点网络中, 通过统一的一个memory 语义层互连.

其实从云的调度装箱的角度来看, 网络中节点的对等做的越好, 调度域越大同时资源售卖率越高, 碎片率越低...

#### 1.3 CloudMatrix-Infer

然后针对模型推理应用的种种挑战和约束, 在CM384架构下构建了CloudMatrix-Infer架构.

首先, 设计了一个点对点的服务架构, 与传统的PD分离相比, 它把推理系统分解成三个独立的子系统构成3个独立的资源池, 即Prefill/Decode/Caching.

这样的架构和传统的以KV Cache为中心的体系结构相比, 实质是构建了更好的一层紧耦合的`存算分离架构`, 构建了一个分布式内存池, 资源分配的灵活性更高, 而且由于是一个对等的结构, 所有节点之间的内存访问的性能也是相当均衡的. 大大的降低了调度复杂度, 提高了系统整体的资源利用率.

其实是针对CM384大规模节点的ScaleUP构建了大规模的专家并行策略, 例如384卡能够支撑EP320, 并且由于UB ScaleUP网络的大带宽给MoE的Dispatch/Combine操作带来了很大的优势, 特别是支持内存语义的UB总线可以做到更细粒度的内存访问控制.

最后是一些硬件感知的优化, 一些INT8量化/算子的加速/Microbatch等overlap机制, 针对硬件上CUBE/Vector/通信等细致的调优.

### 2. LLM对数据中心基础设施的挑战

前面有些简单的介绍, LLM的发展趋势, 例如参数增加, 通过MoE实现更稀疏的计算, 更长的Context Window等.. 然后总结出了几个对数据中心基础设施的挑战.

Challenge 1: Scaling Communication-Intensive Parallelism (XD并行策略如何实施)

Challenge 2: Maintaining High Utilization under Heterogeneous AI Workloads(CPU/GPU/内存资源动态变配)

Challenge 3: Enabling Converged Execution of AI and Data-Intensive Workloads(和通用计算的数据密集型业务耦合)

Challenge 4: Delivering Memory-class Storage Performance(内存级的存储性能)

首先就是随着模型规模的变大, xD并行策略共存时带来的挑战, 例如TP/EP等, 需要频繁的细粒度的低延迟通信, 原来ScaleUP-8卡机型和ScaleOut两张网的架构对于并行通信的复杂度和互相干扰带来的一些overlap策略和算子放置有很多复杂的工作. 然后是工作负责的多样性, 例如推理Decode阶段是memory bound, 而自动驾驶模型训练等任务涉及大量的CPU端的数据预处理, 所以需要根据工作负载动态的进行CPU和NPU以及内存的配比来满足不同的需求.

然后是AI的Workflow和传统的数据密集型业务结合更加紧密, 同时通用计算的一些数据库/大数据/HPC的workload也逐渐演变纳入AI, 这些数据密集型应用通常需要高吞吐/低延迟和灵活的资源编排能力. 而传统的数据中心基础设施难以承受这些严格的要求, 因此需要一个全新的基础设置.

其实这一段话我不太认同的, 工业界做很多事情吧, 推倒重来是很容易的. 但是长期能被工业界接受的是一个带有各种取舍的平滑迁移的过程. 我以前年轻气盛的时候总觉得人家这个不好那个不好, 推倒重来, 结果很多东西都难以落地. 后来逐渐明白能够用几代的架构慢慢的比较平滑的把用户的workload迁移到新的架构上才是更明智的解法.  例如数据库/大数据/HPC这些workload的需求, 在传统数据中心内能不能做? 当然可以, 还可以比线下一些专用IDC机房做的更好, 然后再慢慢的迁移到新架构上.

最后是对存储系统的一些挑战, 例如PB级别的数据集, TB级别的checkpoint, 还有大量的KVCache缓存和RAG这些, 需要存储有更低的延迟, 更高的IOPS以及更低的长尾延迟..

华为对于这几个挑战定义的还是非常准确的, 这才是design of AI infrastructure from the ground up的方法.

但是具体实现上, 如果换做我, 可能先抛弃极致的性能, 把这些接口的抽象先复用Ethernet跑一把, 保证基础设施上的改动逐步进行.  然后花几年时间慢慢培育内存语义生态, 然后再双栈到UB,最后纯UB的方式来构建, iff UB could replace Eth...

其实我在几年前一直都是这个思路, 在Ethernet上搞ScaleUP, 然后构建一个内存抽象层和互连通信的memif接口包装支持内存语义...

### 3. CloudMatrix架构

为了解决AI Workload的一些新的挑战, 华为提出了CloudMatrix架构来作为下一代AI数据中心的架构, 旨在重塑AI基础设施. 它的愿景是去构建一个统一的紧耦合的计算结构, 来有效的支持现代AI应用的大规模/异构部署下的通信需求.

#### 3.1 Vision

毕竟是一个宏大叙事的东西, 当然先要谈谈愿景咯. CloudMatrix的实质是超越传统的以CPU为中心的分层设计, 然后促进所有异构系统组件之间的直接的高性能通信, 包括NPU/CPU/DRAM/SSD/NIC和DSA等, 在不需要CPU中介的情况.

![图片](assets/86b3bc196e87.png)

实质是一个延续了很多年的Composable Disaggregation架构的问题, 突然想起几年前画的一个图:

![图片](assets/976645c71d30.png)

很可惜Cisco当年压根没意识到这件事情的重要性, 另外Cisco自己也没有CPU和GPU这事注定在那里做不成... 不知道他们最近想明白了没? 作为一个身在局中的人, 真心为华为能够做出UB感到高兴, 讲真是超越国外顶级通信厂商的大作, 而其它厂商在各种部门墙的争斗下变得落后了...

回到正题, 华为基于CloudMatrix定义了AI原生基础设施的四个新范式:
3.1.1 TP/EP的可扩展通信
UB互连支持跨NPU的直接高吞吐的点到点通信, 使得EP/TP组能够扩展到很大的范围, 消除了节点间的通信瓶颈, 并允许大模型有效的分布在超节点上.
3.1.2 异构工作负载的灵活资源组合
由于把所有器件都打平到同一个总线上, CloudMatrix将CPU/NPU/内存分解为独立的资源池, 支持了更细粒度的资源动态分配.
3.1.3 统一的基础设施聚合多种工作负载
实质上是在讲如果又了一个统一的基础设施, 原来通用计算中的数据密集型业务, 例如数据库/大数据/HPC这些也可以在这样一套框架下和LLM这些新型的AI Workload混布, 并更加紧密的交互.
3.1.4 内存池化提供高性能存储
实际上是将连接到CPU节点的内存, 通过UB网络暴露出来构成一个池化的资源, 并提供弹性内存服务(EMS), 同时还可以作为后端存储的前置Cache. 然后将这些用来作为KVCache, 参数加载和Checkpoint等业务使用.

#### 3.2 CM384 overview

CM384的互连架构如下:

![图片](assets/fcbf7e87209d.png)

文中有一句话挺有趣的: CM384‘s UB design is a precursor to the UB-Mesh proposed. 384个NPU和192个CPU通过2级的UB交换机连接, 然后下表列出了一下基本的延迟和带宽:

![图片](assets/af140cab1572.png)

在节点间带宽的性能退化约为3%一下, 节点间延迟增加小于1us. 由于AI工作负载都是带宽密集型的,而不是延迟敏感的, 因此这些边际开销对E2E性能的影响可以忽略不计. 在这种设计下CM384可以构成一个超大规模的紧耦合的逻辑节点, 具有全局可寻址的内存以促进统一的资源池调度和高效的负载编排, 为了支持不同的流量模型和传统数据中心网络兼容, 当前CM384还是包含了3个独立的平面, UB Plane作为ScaleUP, RDMA Plane作为ScaleOut, 还有基于DPU的FrontEnd VPC网络.
3.2.1 UB Plane
UB Plane形成了一个超大规模的Fabric, 直接将384个NPU和192个CPU互连在一个端到端无阻塞的拓扑中, 每个Ascend 910c提供392GB/s的单向带宽, TP/EP和一些模型权重/KVCache的快速加载都可以在这个平面完成.
3.2.2 RDMA Plane
RDMA可以作为进一步跨越CM384超节点和其它超节点以及其它外部RDMA系统进行ScaleOut通信的网络. 采用标准RoCE每个NPU支持400Gbps的单向带宽. 例如可以用作Prefill/Decode节点之间的KVCache传输, 或者其他支持兼容RDMA框架的分布式训练和推理等...
3.2.3 VPC Plane
VPC通过采用华为擎天DPU连接, 带宽为400Gbps, 支持标准以太网, 同时可以选用UBoE协议增强. 主要是一些管理控制平面的连接, 然后就是一些对象存储/块存储/文件存储等数据路径. CloudMatrix的长期愿景是将这个VPC Plane和RDMA Scale Plane融合, 但当前还是分开的.

其实RDMA Plane和VPC Plane融合的愿景上, 华为是非常正确的. 但是在标准的RoCE下要做成这个事情是很有难度的. 而我们在这一丢丢的地方是领先华为的, 阿里云CIPU很早就在VPC上运行了RDMA.现在已经有很大的部署规模了.

#### 3.3 硬件组件

CM384采用的NPU是海思的Ascend 910C, 华为24年的旗舰AI加速器, 采用双Die结构, 两个计算芯片Die2Die连接, 共支持8颗HBM.NPU的双Die架构如下:

![图片](assets/e3e028cb8bdf.png)
3.3.1 Ascend 910c3.3.1.1 计算性能
每个芯片大约支持376TFLOPS的Dense FP16/BF16计算, 每个Package支持752TFLOPS.  每个芯片包含24个AI Cube Core, 针对矩阵和卷积进行了优化. 然后支持48个AI Vector core用于一些Elementwise的操作, 所有引擎支持FP16/BF16和INT8类型, 比较遗憾的是没有支持FP8, 但是论文中说采用INT8量化的精度和原生的FP8硬件相当... Die2Die的带宽为单向270GB/s, 双向540GB/s. 补充一个典型的Davinci架构图

![图片](assets/99be4f2c4ee5.png)
3.3.1.2 内存
支持8个16GB的HBM, 共计内存容量为128GB, 每个die 64GB. 内存带宽为每个die 1.6TB/s, 累计整个package 3.2TB/s
3.3.1.3 网络接口
每个Ascend 910C Die有两个接口, 一个是UB接口,工作在224Gbps, 提供单向196GB/s的带宽. 另一个是RDMA接口, 每个Die提供单向200Gbps的带宽
3.3.2 Ascend 910c节点架构
每个CM384的计算节点采用8个910c NPU和4个鲲鹏CPU组成, 并且通过7个UB交换机连接. 然后只有NPU有RDMA接口连接并汇聚到交换背板中, 对外提供单机3.2Tbps的带宽. 然后CPU节点连接到一个DPU, 并提供对外400Gbps的DPU能力连接到VPC.

![图片](assets/48aa1787ed78.png)

计算节点内部的交换芯片对外提供UB的上行连接, 采用光纤连接到第二层交换机.
3.3.3 UB交换机
UB交换机采用2层组网的结构, 整个超节点占用16个机柜, 12个计算柜, 每个柜子4个Ascend910c计算节点, 余下4个作为交换柜

![图片](assets/9a46a991915a.png)

交换网拓扑和带宽如下图所示:

![图片](assets/d10e31ddec1a.png)

#### 3.4 软件栈

主要是一些CANN的介绍, Driver/Runtime/Lib几层的抽象, 以及如何和AI框架兼容

![图片](assets/2bf43f821f1d.png)

比较关键的是下面这个图, 云部署的基础设施软件架构

![图片](assets/b453d5be543e.png)

MatrixResource: 管理超节点内部的物理资源, 包括一些拓扑感知的调度和计算实例的分配, 由擎天DPU提供.

MatrixLink: 为UB和RDMA提供网络服务, 包括QoS和动态路由等, 管理链路级的配置. 也由DPU提供.

MatrixCompute: 协调CloudMatrix实例的生命周期, 从裸金属供应到自动伸缩和故障恢复. 它协调跨多个物理节点的资源组合以创建紧密耦合的逻辑超级节点实例.

MatrixContainer: 提供基于Kubernetes的容器服务,通过拓扑感知调度增强.

ModelArts: PaaS和MaaS层软件.

#### 3.5 DeepSeek模型的适用性分析
3.5.1 DeepSeek模型在H800上的部署
简要介绍了一下DeepSeek-V3的架构和EP/DP并行策略, DeepSeek的官方博客已经讲的很清楚了, 就不重复了.
3.5.2 CloudMatrix384 和 DeepSeek 模型之间的架构协同作用
主要有几点:
3.5.2.1 高效的MoE Dispatch/Combine
UB整个域有384卡, 因此做EP320甚至更多的Redudancy Expert都可以, 然后UB带宽大,又是细粒度的LD/ST, 效率高很多.
3.5.2.2 内存容量
每张卡128GB, 384卡累计就是49.2TB, 可以存很多KVCache, 支持更长的Context, 并且整个内存资源打平在UB总线上, 细粒度访问也很有效.
3.5.2.3 Context Cache Reuse
DeepSeek报告中KVCache的命中率超过56%, UB相对于传统的架构, CPU内存更大容量的存储空间可以被NPU直接访问, 而不需要复杂的绕行RDMA或者绕过PCIe, 因此通过更高的命中率和更低的访问延迟, 显著的降低了TTFT和避免了NPU算力和内存的开销.
3.5.2.4 量化INT8
说实话INT8和FP8量化的效果到底如何, 我不知道... 但是论文说精度没啥影响咯...

### 4. CloudMatrix-Infer

**由于这一篇是专门偏CM384硬件架构的, 所以这部分简略一点, 后面再单独详细写一个**

为了充分利用CM384的能力, 作者提出了CM-Infer这种全面的LLM服务解决方案, 成为部署大规模MoE模型的一个最佳实践. 下图为主要的优化的点和相关章节的介绍:

![图片](assets/c422298f5df6.png)

底层是基于UB架构的弹性内存服务, 详细内容在4.4介绍.  然后是MaaS的ModelArts框架. 然后4.1讲了一下PDC(Prefill/Decode/Caching)的架构, 4.2和4.3分别阐述了Decode和Prefill的一些优化. 最后是一些INT8的量化算法.

#### 4.1 PDC分离架构

**实质是提供了基于UB总线的存算分离架构, 将CPU/GPU/内存都作为一等公民加入到UB总线** CPU节点也纳入到UB总线, 因此CPU的内存可以作为KVCache的节点存放, 同时可以供Prefill/Decode集群高速访问

![图片](assets/fda934d51b21.png)

Prefill节点由16个Ascend 910c NPU芯片(32个Die)构成一个EP32的并行部署. MLA才用了混合并行策略同时支持了多个microbatch pipeline的overlap.

Decode节点由160个Ascend 910c(320die)构成EP320的并行策略, 然后做了一些算子优化/Overlap和MTP的支持.

有一个值得关注的是Global Scheduler这些调度组件. PDC可以基于工作负载的统计, 细粒度的调整Prefill/Decode节点数量, 提高SLA, 然后还有基于Token边界处对齐请求的调度机制, 允许多个Session同时共同的调度和处理.

#### 4.2 Decode

如下右图所示:

![图片](assets/a190311dd678.png)

开发了Fused Dispatch/Combine 两个Operator. 将所有的AlltoAll通信替换为Send-recv源语. 利用UB Plane的高带宽优势, 直接写入减少通信延迟, 同时在NPU之间通信前的调度阶段进行量化操作. 其次是消除和Dynamic Shape相关的开销, 预先分配所需的内存空间, 实现静态图的执行. 另外通信和计算也被组织成了一个流水线管道, 提高资源利用率和吞吐.

首先, 传统的RDMA这些, 需要先从SMEM拷贝到HBM GMEM, 然后再dispatch或者combine, 如下红色的路径. 而UB可以直接从SMEM(华为Ascend中AIV的UBuffer)写到远端NPU的内存

![图片](assets/7b18fa9fb992.png)

整个流水线的Overlap如下:

![图片](assets/0a55ad0c2f17.png)

然后TBO(Two-Batch Overlap)如下所示:

![图片](assets/445679d13f58.png)

MTP也做了一些优化:

![图片](assets/def989cc3c07.png)

#### 4.3 Prefill

MLA TP+SP混合并行:

![图片](assets/e41e8d177324.png)

Prefill的TBO

![图片](assets/999552a143ab.png)

值得关注的是P和D之间的KVCache传输机制. 为了消除对Decode阶段的影响, 采用RDMA平面传输KVCache,这样KVCache的通信和UB计算总线两个平面独立.

然后还有一些异步的Prefill调度, 首先在Decode目标节点分配KV缓冲区, 然后将Prefill任务路由到低负载节点进行计算, 完成时直接触发RDMA缓冲传输.

#### 4.4 UB驱动的分布式缓存

用CPU节点的内存构建了一个分布式的缓存内存池, 控制路径走VPC, 数据路径走UB

![图片](assets/bc9d1383fdfd.png)

而且注意到由于CPU节点有DPU连到VPC支持存储等云盘和OSS可以做很多到离线存储的优化, 然后模型加载的时候有这个缓存层快很多

![图片](assets/83becf895f22.png)

另外就是KVCache的处理, 快了很多

![图片](assets/f21cf35224af.png)

### 5.  Evaluation

**这个也后面有空了再写一篇补**

### 6. 对未来方向的探讨

#### 6.1 未来CM架构的演进
6.1.1 VPC和RDMA两网融合
现在RDMA ScaleOut和FrontEnd是分离的架构,  但是呢TP/EP/SP带宽密集型的通信阶段主要是在UB上承载, 那么对于DP/PP其实带宽需求并没有那么高, 同时通过一些细粒度的Overlap进行通信延迟隐藏也是可行的. 所以可以在一个AZ粒度上通过VPC来承载.

其实VPC要承载RDMA流量还是有很多难度的, TCP这些流量要混跑如何相互避免影响? 大规模部署下的多路径和有损支持能力, RDMA的RC兼容/QP Scale/拥塞控制/安全/热迁移/热升级/设备虚拟化有一堆工作要做, 至少现在NV的很多解决方案是不干净的或者性能受损的.

但是这条路是正确的, 真男人就该把这些细节问题全解决好.

6.1.2 超大规模的超节点
作者有一些论断, 未来还会继续扩大超节点的规模:

例如越来越长的Context, 越来越多的稀疏的专家(大EP), 基于RAG的一些应用等.. 另一个观点是做大规模了能够提高资源的利用率. 还有就是扩大ScaleUP的规模, 按照2层CLOS架构, 网络上摊销的成本也不会增大, 如下图所示, 交换机可以按需缩放.

![图片](assets/31d61e01d16f.png)

然后就是下一代AI工作负载中, 除了NPU和CPU外, 还可以又一些专门任务处理的DSA, 这些Unit将逐渐融入到AI中, 超大的规模可以提供更加flatten网络, 所有的处理器都作为一等公民互连.

其实基于这个视角, 我们就应该需要一个更加开放标准化的ScaleUP网络, 并且实现不同厂商处理器之间的高速内存语义通信, 因此我个人的观点是特别想保留IP和Ethernet这个大的腰部的基础上来解决问题... 或者华为的想法是做个UB NIC也行? PCIe/CXL/AXI啥的都可以转到UB上? 或者实现UBoE的架构?

6.1.3 CPU节点的Disaggregation/Pooling
如下图所示, 实际上是针对计算资源对CPU和NPU之间进行动态的配比

![图片](assets/d3d07cdba5fd.png)

#### 6.2 未来推理系统的增强

谈到了几点, Decode-Attention的解耦和Offloading, Attention和MoE的分离. 然后基于算子分离的视角来看, 逐渐解耦成微服务架构, 进行硬件感知的调度/Colocation/自适应的扩容等...

### 一些感想

知乎和朋友圈都看见夏Core在叫大家来看这论文呀, 有一个留言挺有意思的:

![图片](assets/2c0020ed605c.png)

从技术上来讲, NV虽然有NVLink更高的带宽, 但是可靠性和可扩展性也受到自身协议的束缚. 同一时期国外的几个顶尖的通信大厂因为自己没有处理器和GPU, 都在互连上踏空了. 而这一步华为真的是遥遥领先了. 同时也要祝贺袁老师的硅基流动团队和CM384的协同优化, 使得性能有了大幅度的提升.

我觉得吧, 中国人都要有一点自信, 凭什么别人能做的我们不能做, 凭什么我们就比别人低一头呢? 其实在很多小的领域国内已经开始领先了, 大的领域例如电车/无人机这些已经大规模领先, 埋头干就行了...

如果说唯一的一点小问题, 那就是UB相对封闭的生态. 另一边BRCM今天又更新了SUE的规范, 越来越开放越来越完善...不知道明天会不会又来一个XXXLink国内首创1.02版...

参考资料

[1] 
Serving Large Language Models on Huawei CloudMatrix384: *https://www.arxiv.org/abs/2506.12708*