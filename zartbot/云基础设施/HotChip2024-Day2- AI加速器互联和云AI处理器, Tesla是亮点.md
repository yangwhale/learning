# HotChip2024-Day2: AI加速器互联和云AI处理器, Tesla是亮点

> 作者: zartbot  
> 日期: 2024年8月27日 23:33  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491930&idx=1&sn=e0e7ad650edfe12d8a4afb869424264b&chksm=f995f198cee2788e23e07ffedfab80da407ffc22a93ebfcb6e748d82bba27e92c48a27ae0359#rd

---

### TL;DR

第二日的亮点也很多,特别是Tesla, Tesla的传输协议TTPoE, 借助`iWARP的TCP拥塞控制机制`和`RoCEv1二层转发`构建了基于以太网Lossy的转发. 他们这次讲的非常清楚,而且还可以FrontEnd和ScaleOut混跑, 唯一就是多路径上还有点问题要处理一下就好了,  这个话题后面会单独再开一篇来详细探讨.

其它几家的概述如下.

Azure Maia 100 和Meta MTIA这两个云上用的AI加速器

AMD Versal继续在推AIE-ML v2

Cerebras WSE-3,谈了谈部署的集群, 然后架构上没啥变化.然后也开始卷推理服务了

Stanford的稀疏张量处理器Onyx,有点意思, 值得后面单独分析一下

Intel谈了一下CPO

Enfabrica也讲了一下它的Fabric,言之无物的感觉.

针对HPC应用的MN-Core 2有点意思,主要是NOC上的bcast和reduce

CPU则是 Ampere, AMD Zen5, 国内的香山RISC-V等几颗处理器, 后面再单独谈

## 1. Tesla TTPoE for Lossy Fabric

这次HC2024最想听的就是Nv Blackwell和Telsa TTPoE了, 最后Nv不谈Blackwell微架构一点诚意都没有, 而且自家BF3+SP4 Lossy支持有问题就扯Lossless,完全是扯淡, 是男人就硬干Lossy.

而Tesla则是诚意满满, 从Session题目就点名了`A new Lossy` Fabric. 开篇就是`第一性原理`陈述问题

![图片](assets/95e3899e3acd.png)

DC Ethernet RDMA有一堆问题, Lossless是垃圾PFC在瞎搞, 然后TCP/IP要过CPU, 内核和软件协议栈开销太大,需要GPU-Direct通信. 然后就缝合了一个Remote DMA over TCP over Ethernet 的缝合怪出来. 这才是第一性原理去解决问题的方式, 而不是天天号称掌握了第一性原理然后又天天到处调研分不清现实是啥. 看看人家Tesla写的多清楚, **TCP got it right- just do it in hardware**

![图片](assets/45b9a19b0dea.jpg)

其实这件事情无论Google的Falcon以及DirectTCP-X早就想明白了的...这是正路啊, 各位同学.

本质上就是硬件处理TCP类似的基于Window CC/SACK/快重传这些东西, 然后网络层可选, 底层物理层和数据链路层还是以太网, 一个字节都没改, 什么以太网交换机都可以用. `我真不知道那些天天想着改以太网报文的人在干嘛`...

![图片](assets/de8137df8eb7.png)

整个事务层和TCP几乎一样, NACK做丢包处理

![图片](assets/5104c21bb71a.png)

然后状态机相对于TCP做了简化

![图片](assets/640f1ff4432a.png)

协议层很有趣, 像RoCEv1那样直接over在Layer2 Ethernet之上, 然后提供了VC的概念

![图片](assets/46eaa756db5d.png)

然后针对Lossy, 像TCP那样搞就行了, 快重传加上, Window based CC.

![图片](assets/d8ce1d55b2a1.png)

然后TX控制, 不要交换机和网络来给控制信号

![图片](assets/fe298e6a5422.png)

想想渣一直给你们灌输的东西, 下面这个业务目标是非常明确的.

采用Spine-Leaf拓扑, 不用任何框式交换机, 不需要DeepBuffer. **如何不利用交换机任何Hash函数信息, 不需要交换机任何特殊配置, 不启用ECN和PFC. 通过网卡算法自动打散流量,并维持交换网97.5%以上的利用率, 对于交换机的buffer需求为队列深度低于3us.** 并能够针对128:1的时候incast时最大流和最小流量之间的带宽差异小于100Kbps, 同时针对任何网络线缆故障, 通信中断无感知, 模型训练收敛时间小于100ms.

可惜就是不听, 有人说我分不清现实与幻想? 那么天天说第一性原理的Tesla也分不清了?

然后**人家Tesla就要用标准的以太网MAC,其它都不用改, 和渣一样要去薅以太网量大的羊毛.**

![图片](assets/c7eb4f5e070a.png)

它的MAC微架构

![图片](assets/9119c423f0d4.png)

然后100G的网卡

![图片](assets/cd4c01a3c2fd.png)

你说这玩意和NetDAM是不是同一个东西?

![图片](assets/aa1c6b3cfd81.png)

![图片](assets/898de5d9b73e.png)

![图片](assets/0c31635c2c48.png)

![图片](assets/8d263f6f1383.png)

好好看看这图吧

![图片](assets/c3de76d6ee63.png)

然后人家说从存储捞数据和训练的Backward Pass的Allreduce可以混跑呀, Front-End和Scale-Out可以融合呀

![图片](assets/14a8f31eaffc.png)

只要用TCP一样的拥塞控制算法, 混跑就行了啊, TCP和TTP一张网~

![图片](assets/9fc32e63691a.png)

**不过我也要Diss一下Tesla, 多跳交换机只能做到80%的Fabric利用率, 我们完全解决了多路径Hash冲突的问题, 轻松97.5%**

![图片](assets/ae9c66125d2f.png)

最后这玩意又要弄到UEC里面去, 估计大家都要把UEC搞黄么? AMD一套方案, 微软一套, 博通一套, Tesla一套, 估计NV还要来一套....有一点当年OpenDayLight的感觉.

![图片](assets/49162baa86e8.png)

## 2. Azure Maia 100

Target应该是OpenAI的推理业务, 然后软件做的很不错, 高密部署很不错, 标准以太网融合ScaleUP和ScaleOut也不错, 但是用RoCE就需要配一个额外的Tile控制器, 和Intel Gaudi3要一个中断管理器一样的缺点.这个后面单独写一篇.

![图片](assets/105ae15d48f7.png)

规格来看, 主要是用来推理的, 算力并不是很大

![图片](assets/d89fdb57bb72.jpg)

片上架构, 有一个Tile控制核TCP, 类似于Nvidia TMA的(TDMA)配上了, 向量和张量引擎都有

![图片](assets/238c2f116cb1.png)

Tensor Core也是16xRx16的, 这个size是合适的, 某些128x128真的太大了,没意思

![图片](assets/715761a41c26.png)

和Hopper TMA一样, 有Tensor加速, 不过NV用内存屏障,而它用信号量来做

片上网络来看, 2D-Mesh, 然后支持数据压缩挺好的, 很大的Scratch Pad都是这类AI处理器的标配

![图片](assets/24375f6af378.png)

互联ScaleUp和ScaleOut标准以太网融合好评, 然后RDMA数据加密了好评,毕竟对OAI的推理业务加密很关键.

![图片](assets/3689bb00f97b.png)

软件SDK有自己的集合通信库MCCL, 然后支持Triton和底层Maia API, 但是用RoCE就需要配一个额外的Tile控制器, 和Intel Gaudi3要一个中断管理器一样的缺点.

![图片](assets/9e8d96718b68.png)

这里详细介绍了一下信号量的控制流程

![图片](assets/303b1527cf5e.png)

然后Overlap计算和通信是常规操作, 在网络上传输量化的数据和DeepSpeed Zero++类似

![图片](assets/1524193d9e9a.png)

生态兼容很好,两行代码从cuda换maia

![图片](assets/ed3a09cdc087.png)

通信库中规中矩

![图片](assets/6fbedba22d0f.png)

## 3. Meta MTIA

先谈业务需求,好评~ 针对搜广推的主业, 模型越来越Dense化,规模越来越大, GenAI/LLM也开始使用.

![图片](assets/a75708d019f5.png)

然后业务目标要优化Perf/TCO和Perf/W, 以及方便支持多个模型和快速开发

![图片](assets/fbc48e96e093.png)

功能上的需求

![图片](assets/93c10376d206.png)

加速器Spec, 没有用HBM,直接用LPDDR5, 成本考虑控制TCO, 功耗90W, FP16 177TFLOPS

![图片](assets/95692bc69f54.jpg)

架构如下, 8x8的2D-Mesh, 16Channel LPDDR5, onchip 256MB SRAM

![图片](assets/ec6f5f8d0e6b.png)

控制处理器为4x RISC-V标量处理器, 支持8M L2Cache, 然后PCIe控制器上还有4MB Descriptor SRAM

![图片](assets/2e0dc399eba8.png)

然后片上网络支持一些组播/广播, 和Hopper的Distribute-Shared-Memory类似

![图片](assets/562879465eea.png)

然后计算Core架构如下, 两个RISC-V处理器,一个标量一个向量,配置了矩阵/特殊函数处理,动态量化, 数据动态搬移等协处理器和一个命令处理器和其它block的PE交互

![图片](assets/fa97376bfc6d.png)

向量采用点积的方式, 支持稀疏矩阵

![图片](assets/11f5eee0c0bd.png)

有384KB的Local的SMEM提高峰值处理能力

![图片](assets/9e609a06d379.png)

动态量化的能力

![图片](assets/897918bafe4e.png)

组播的NOC好评

![图片](assets/740634213af1.png)

支持25GB/s的解压缩能力来支撑Embedding Table

![图片](assets/4e17ba3f24b0.png)

数据在NOC上传输也可以压缩解压缩

![图片](assets/cb066bcb79c9.png)

然后针对推荐系统Embeding的Batch操作和Index aligned DMA/prefetch

![图片](assets/5a5657b75e5d.png)

单卡双芯片

![图片](assets/1752ab11eaf6.png)
整机支持12个模块, 单机柜支持3台, 累积72卡

![图片](assets/297b7c089e72.png)

## 4. AMD Versal

主要还是介绍Xilinx搞的AIE-ML, 但是这东西说实话在AMD收购以后发展的并不是很好, Victor Peng也退休了,今天还有一个单独的演讲, 祝好吧~

![图片](assets/9d9468edf7c8.png)

## 5. Cerebras WSE-3

还是突出自己“大”

![图片](assets/6030bb59da67.jpg)

讲了一些部署的集群

![图片](assets/f91c288c115c.jpg)

然后又强调了一次SwarmX和MemoryX构成大的训练集群, 不就是一个超大规模的参数服务器和优化器么

![图片](assets/dfeb082a593d.png)

然后估计融资的压力也有的, 开始卷推理了,理由是它内存墙撞烂了

![图片](assets/3f8bd105ea10.png)

![图片](assets/a7c1c9c0469f.png)

![图片](assets/e6b943502fe0.png)

然后解释了为什么并Diss了一下NV

![图片](assets/d154c50cab36.png)

并行放置多个模型也容易, 4个WSE-3就够了

![图片](assets/c4a75c604868.png)

然后开始卖推理服务了

![图片](assets/b2619b86ca78.png)

## 6. Intel CPO

![图片](assets/55385a8c4f99.jpg)

有个OCI demo

![图片](assets/ba076e79922b.jpg)

![图片](assets/dbb0249ab469.jpg)

## 7. Enfabrica

统一ScaleUp和ScaleOut通信是个好事, 但是这家公司还是网党的观点和技术路线, 没啥意思.

原来互联是这样的

![图片](assets/4675720a809f.png)

然后存在的问题

![图片](assets/6a60fff94bde.png)

解法是自己搞一个RPC Domain的Fabric

![图片](assets/ae927d6f1c07.png)

芯片架构就是一个以太网+PCIe一起的交换机

![图片](assets/1ee5ed0d6fdd.png)

![图片](assets/8023f91de99c.png)

然后画了一个2层组网524K组网的一个饼

![图片](assets/796bbf493628.jpg)

## 8. MN-Core 2

片上多级的broadcast配合MAB的微架构还是有点意思的

![图片](assets/5b709cf759e3.jpg)

PE的架构如下:

![图片](assets/b9dbccef7ffc.png)

多层IR的设计, 感觉有些复杂?

![图片](assets/432d910e4b4f.jpg)

## 9. Stanford Sparse Tensor Accelerator

业务上GNN和稀疏Transformer的需求

![图片](assets/ed9f41de37f7.png)

E2E的稀疏加速硬件有问题

![图片](assets/a636053c7c88.jpg)

然后Kernel-Level的稀疏加速也有问题,需要可编程

![图片](assets/b4c4ef2251e0.png)

但是可编程的CGRA又主要针对Dense应用

![图片](assets/13debc9f16b0.jpg)

Onyx搞了一个任意稀疏/密集张量计算的抽象表示

![图片](assets/2d3ba3afd257.jpg)

架构是CGRA, 计算和内存节点编排如下

![图片](assets/a15384d75c16.png)

这东西长得有点像神威

![图片](assets/bf0f6a4b10e1.png)

稀疏矩阵采用FiberTree表示

![图片](assets/0ed699ac4074.png)

![图片](assets/0bedf5c0ec7f.png)

![图片](assets/3a0c85d0c6ba.png)

后面还有很多内容,熬了两天夜白天还搬砖有点累了, 后面我们会在Tensor系列专题里面介绍Sparse-Tensor的时候详细介绍这一块的内容.