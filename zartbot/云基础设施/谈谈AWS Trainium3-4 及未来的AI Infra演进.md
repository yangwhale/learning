# 谈谈AWS Trainium3/4 及未来的AI Infra演进

> 作者: zartbot  
> 日期: 2025年12月19日 11:05  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496916&idx=1&sn=38a19ad579eff61c1da44d57fd2cf1df&chksm=f995e416cee26d00f8d8e726304821c806b0981329a1efff016ee904b0cc00d6dfe9df94deb9#rd

---

`本文仅代表作者个人的观点, 与作者任职的机构无关`

### TL;DR

基本上每年的AWS Re:invent都会做一些分析, 例如去年的[《AWS Re:invent GenAI路上快速的追赶者, 详细谈谈Trainium2/3架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492863&idx=1&sn=16930a20429695f8a2c9182a83e97739&scene=21#wechat_redirect). 今年本想偷懒却被很多人惦记着要我来写点分析. 鉴于前面一篇对NV GPU微架构的分析[《Inside Nvidia GPU: 谈谈Blackwell的不足并预测一下Rubin的微架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21#wechat_redirect)都被老黄在内部转发, 那么这次也认真详细分析一下Trainium 3/4这类的架构的优缺点, 然后再结合对比NV的微架构对未来AI Infra演进进行一个详细的阐述.

本文目录如下:

```
1. AWS Trainium 3架构分析1.1 Overview1.2 Trainium3 芯片架构1.3 NeuronCore1.4 ComputeTray架构1.5 内存子系统和ScaleUP互连1.5.1 内存子系统1.5.2 Near-Memory Accumulation1.5.3 ScaleUP互连1.5.4 ScaleUP未来演进1.6 ScaleOut/FrontEnd网络1.7 软件架构2. 未来AI Infra的演进2.1 前置技术背景2.2 从业务的视角分析2.2.1 预训练业务2.2.2 RL后训练2.2.3 Reasoning模型和Agentic2.2.4 小结2.3 算法演进2.3.1 从物理的视角2.3.2 从数学的视角2.3.3 从计算机体系结构的视角2.3.4 Attention2.3.5 MoE2.4 从商业模式的视角分析2.5 从基础设施视角分析2.5.1 算力芯片微架构2.5.2 系统互连架构
```

注: 对Trn3架构比较熟悉的读者可以很快的跳过第一章这个前置环节, 进一步阅读关于未来AI Infra的演进, 其实在2023年底我也在内部写过一个版本《云基础设施演进的一些思考》, 例如那个时候针对MoE/MHA的分析推导出ScaleUP超节点, 并逐渐的影响BRCM投入ETH ScaleUP, 或者是基于Memory语义的ScaleOut等, ScaleOut/FrontEnd/ScaleUP三网融合等, 基本判断都成立了, 而如今势必针对未来几年AI Infra的变化进行一个修正, 同样首先以业务背景驱动, AI接下来的Scaling在什么地方, 然后在算法上进一步分析取舍, 最后综合技术背景给出一个AI Infra的趋势.

![图片](assets/85ee9f89176f.png)

## 1. AWS Trainium 3 架构分析

### 1.1 Overview

**最大亮点是AWS宣称整个Neuron软件栈(NKI)未来几个月会完全开源**

Trainium 3芯片结构上和Trainium 2没有太大差别, 依旧是两个Die和封, 采用了TSMC N3P的工艺, 单芯片可提供2.52PFLOPS FP8的算力, 集成了4颗总计144GB的HBM3e显存, 提供4.9TB/s的带宽.

![图片](assets/f0551f096bb5.jpg)

整个系统来看, 支持了OCP MXFP8/MXFP4格式, 整个ScaleUP域规模扩大到了两机柜并联144卡, 最大的亮点是ScaleUP的带宽翻倍并抛弃了原有的Torus互连, 引入了交换机, 使得AlltoAll的互连带宽获得了很大的提升, 对MoE更友好

![图片](assets/4365dc4beef7.png)

AWS的一个session《AWS re:Invent 2025 - AWS Trn3 UltraServers: Power next-generation enterprise AI performance(AIM3335)》[1]详细介绍了整体的一些架构细节.

总体来看设计思路有一页讲的很清楚, 单纯的Peak FLOPS性能并不重要, 而是需要再实际workload中维持稳定的高效的性能.

![图片](assets/2b7334645408.png)

![图片](assets/b921cfeb91a9.png)

主要包含以下多个手段:

![图片](assets/1a7cd4dfdc5a.png)

降低精度, 即支持OCP的MXFP8/MXFP4格式.

随着TensorEngine上的计算加速, 对于Attention计算的Softmax计算性能也需要加速, 避免TensorEngine因为VectorEngine计算缓慢而出现空闲.(这一点也是B200有缺陷,而B300修复的地方)

支持近内存的加法, 对于集合通信中的Reduce帮助很大, 后面会详细阐述

Tensor Dereference对MoE模型中GroupGEMM有很大的帮助, 无需进行预先的Shuffle处理.

Traffic Shaping, 即在传输/通信过程中采用了流量整形的技术, 避免在多种并行以及PD分离的KVCache传输上不同突发流量带来的干扰, 使得整机能够维持更加稳定可持续的性能表现.

Background Transpose, 矩阵专置操作在现代LLM中很常见, 通过新增的硬件指令在后台对矩阵进行转置处理, 避免了额外的开销

MMCastMode, 暂时没有找到相关的资料, 猜测MM是指的Matrix Multiply? 在矩阵乘法中针对不同维度的张量进行cast, 例如A: shape (batch, m, k), B: shape(k, n) 决定Cast Mode到Result shape (batch, m, n)

内存地址基于一个Hash函数的Spray能力, 相当于在简单的Interleave基础上增加了类似于Swizzle的硬件处理.

接下来, 我们分成各个组件展开详细的分析.

### 1.2 Trainium3 芯片架构

整个芯片来看, 其架构和Trainium 2也是类似的, 如下图所示:

![图片](assets/9d9c2c02eafe.png)

都是采用两个Die及4颗HBM, 计算核心从NeuronCore-V3升级到了NeuronCore-V4, 浮点性能从1,299 FP8 TFLOPS升级到了2,517 MXFP8 TFLOPS接近翻倍.

![图片](assets/855c27ae1c23.png)

显存带宽从Trn2的2.9TB/s升级到了4.9TB/s, 显存容量从Trn2的96GB升级到了144GB, 片内的SRAM容量也从28MB升级到了32MB,累计8个NeuronCore-V4总容量升级到了256MB.

![图片](assets/2b500ed0a05b.png)

ScaleUP从NeuronLink-V3升级到了NeuronLink-V4带宽进一步翻倍.

![图片](assets/260a96ccf955.png)

DMA引擎性能也升级到了匹配HBM带宽4.9TB/s的能力

![图片](assets/38b6493bd055.png)

### 1.3 NeuronCore

微架构上延续了原有的NeuronCore的架构, 相对于Trainium 2的Neuron-V3, SRAM容量扩大到了32MB, 并且支持了Near-Memory Accumulation能力, 在Nvidia上的实现是基于NVSwitch支持MultiMem做NVLS(Nvlink SHARP), 或者是TMA指令来做Reduction. 而在Trainium 3上直接可以在SRAM上做, 这样可以显著的降低对HBM的大带宽冲击.  详细的关于SRAM和互连相关的信息我们将在下一个小节介绍.

![图片](assets/ac4362b4d679.png)

计算引擎还是维持了TensorEngine(用于GEMM/卷积/转置操作), VectorEngine(向量计算), Scalar Engine(标量计算), 以及完全可以用C++编程的通用SIMD(GPSIMD Engine)补全一些特殊算子的能力.
TensorEngine
主要区别是在Tensor Engine上支持了MXFP8/MXFP4, 然后延续了前一代支持的结构化M:N稀疏运算的能力(Structed Sparsity), 支持4:16, 4:12, 4:8, 2:8, 2:4, 1:4, and 1:2等多种模式.
VectorEngine
为了配合TensorEngine支持MXFP8/MXFP4 , VectorEngine支持支持将 BF16/FP16 数据快速量化为 MXFP8 数据格式.  然后最关键的是进一步提升了指数函数的计算能力, 保证在Attention等计算过程中避免TensorEngine等待.

![图片](assets/d477c5763fd8.jpg)

注: 这一点比起同期的Blackwell B200要完善很多, 而Nvidia仅是在B300中才通过降低TensorCore FP64算力的方式换取SM更大的面积来支持SFU, 整体来看SFU性能在B200到B300上的修复会使得Attention的运算性能提升接近20%

ScalarEngine
没有太大的变化
GPSIMD Engine
这一直是Trainium系列的一个创新, 可以在上面执行通用的C/C++编译的代码来扩展DSA架构下可编程灵活性的一些缺陷, 对于一些特殊的算子可以通过GPSIMD进行高效实现, 并且它也可以完全访问SRAM.

### 1.4 ComputeTray架构

整个ComputeTray采用模块化设计, 除了液冷管路以外和Rubin一样是采用Cableless的模块便于安装和运维.

![图片](assets/2d3538af3828.png)

在Trn3的ComputeTray中, 后半部采用了4个Trn3芯片, 背后输出的NeuronLink接口通过CableTray连接到整个Rack, 然后板载了一颗NeuronLink Switch连接了四颗Trn3芯片.

在整个ComputeTray的前半部, 我们可以看到两侧各放置了两块Nitro-V6, 一颗Graviton 4 CPU并配置了12channel DDR. 这些应该都是统一的连接到PCIe Switch上的. 不确定是否也复用了NeuronLink Switch的来连接Graviton CPU和Nitro-V6网卡.

另外前面板上还预留了4个NeuronLink接口用于两个72卡的机柜并联使用.

### 1.5 内存子系统和ScaleUP互连

一个显著的变化是, Trn3放弃了Trn2采用的3D Torus Mesh的拓扑结构, 引入了NeuronLink Switch. 并且在Trn4中还会进一步引入相对开放的NVLink Fusion或者UALink.

#### 1.5.1 内存子系统

**抛开微架构和内存子系统谈互连是毫无意义的**, 从内存层次化结构来看, Trainium和TPU这类的DSA与GPGPU有显著的区别, 如下图所示:

![图片](assets/e1228bf4c677.png)

DSA架构通常是在一个核内配置大量的SRAM(例如Trn3单个NeuronCore有32MB), 并且需要显示的进行内存管理和预取处理. 而GPGPU架构通常延续原来的SIMT抽象, 在原有的SMEM+RF的基础上引入TensorCore, 并为了更好的异步访问内存引入了TMA, 同时考虑到Register Spill的压力, 又在Blackwell中引入了TMEM. 同时为了更好的局部访问内存优化, 又引入了CGA的sub NOC.

其实我们可以看到, 整个内存子系统上NV为了维持CUDA生态变得越来越复杂, 单个SM内的SMEM容量相对较小, 因此从Little's Law的角度来看,访问延迟和带宽都会受到约束, 因此SMEM-to-SMEM访问仅限制在CGA范围内, 然后大量的空间在L2Cache上, 但L2Cache无法手工管理. 因此从其它ScaleUP来的读写需求无法直接进入SM内部的SMEM, 而只能存放于GMEM(HBM)上,并由L2Cache做部分的缓冲.

而Trn3这样的DSA架构, On-Chip SRAM单个NeuronCore为32MB, BDP(Bandwidth Delay Product)会好很多, 因此它还可以支持SRAM通过NeuronCore的直接读写另一颗Trn3. 总体来看在集合通信, 特别来说针对MoE的Dispatch/Combine上, 这样的架构对HBM的压力会显著小于GPGPU的架构.

架构的选择本质上都是各种Trade-Off, 有优点也在其它地方存在缺陷.  例如从内存层次化结构来看相对于NV的架构看上去简洁很多, 但是这样的DSA编程的难度也会大很多, 具体的我们将在软件架构中进行详细分析.

#### 1.5.2 Near-Memory Accumulation

Nvidia可以通过在NVSwitch上实现类似SHARP的功能来加速Reduction操作, 也可以通过TMA的部分指令加速. 而AWS则是直接可以对SRAM进行类似的累加操作, 并且不需要在交换机上去做这些复杂业务. 该功能允许DMA引擎通过单次传输操作，直接对SRAM中已有的数据执行 read-add-write 运算.

其实从微观角度来看, 在NV GPU上Reduction的操作对其他计算Kernel内存访问的干扰和影响是非常大的, 通常这样的Partial SUM Reduction或者EP的Combine操作都需要写回到HBM同时造成L2Cache的污染影响其它计算Kernel的性能.

相比之下, Trn3 在EP的Combine阶段, 应该可以从Expert的那张卡直接Combine加回到本地的SRAM上, 整体E2E的延迟也应该更低, 编程处理也更容易一些.

#### 1.5.3 ScaleUP互连

当前的NeuronLink-V4方案是基于PCIe Gen6修改的, 比前一代基于PCIe Gen5的Trn2 NeuronLink-V3的带宽翻倍. 最大的变化是抛弃了3D Torus Mesh的拓扑结构, 引入了交换机增加AlltoAll的性能. 但是似乎又做的不彻底, 从拓扑来看, 每个Trn3芯片有3条NeuronLink, 分别构成不同的连接

![图片](assets/897af9406559.png)

**ComputeTray Switch**: 在单个ComputeTray内部有一个NeuronLink Switch连接ComputeTray内四个Trn3

**机柜Switch**:  每个机柜72卡都有一条NeuronLink连接到一个机柜内的NeuronLink Switch

**机柜间直连**: 剩下的NeuronLink用于两个机柜直接一一对应的互连

这样复杂的约束主要来自于交换机芯片Radix的限制. 从交换机芯片的角度来看, 首期提供的是基于Asteralabs Scorpio-X 160lane的PCIe Gen6交换芯片, 然后会升级到支持320Lanes的Scorpio-X, 后期还有一个Asteralabs的UALink Switch方案. 如果采用新的320Lanes的Scorpio-X或者支持UALink的交换芯片方案, 可能就会完全实现全部挂接到机柜级的Switch Tray上了. SemiAnalysis做了一个预测可以参考一下:

![图片](assets/367dec4ac11a.png)

然后三种互连的带宽也是不对等的, 其中连接机柜的Rack-Level ScaleUP Switch支持80个Lane, 两个机柜之间同Rank Trn3互连的仅有16 Lane, SemiAnalysis有个内部互连的图

![图片](assets/58f03145a236.png)

考虑实际的并行策略来看最省事的方案是通过Rack—Level Switch实现大EP并行. 然后ComputeTray内部的Switch可以用来承载一些TP并行的需求, 也可以用剩下双并柜两个机柜间的带宽, 在两个机柜上进行TP2的部署.

#### 1.5.4 ScaleUP未来演进

整体来看, AWS ScaleUP从封闭私有的基于PCIe标准定制的NeuronLink逐渐转向开放标准的NVLink Fusion和UALink

![图片](assets/f084527417a6.jpg)

很有趣的是在Trainium 4上AWS同时押注NVLink和UALink, 使用I/O Chiplet方式和计算Die采用UCIe互连, 针对使用NVLink ScaleUP的Trn4可以使用NVLink Fusion Chiplet.

![图片](assets/e02b4c86329e.png)

而针对UALink则更换相应的I/OChiplet即可.

### 1.6 ScaleOut/FrontEnd网络

Trainium 2 ComputeTray采用4颗Trn2芯片, 并使用了8块200Gbps的Nitro v5 构建1.6Tbps的ScaleOut网络

![图片](assets/cad412720cbe.png)

然后有独立的CPU Tray 和200Gbps FrontEnd Nitro, 其中100Gbps用于VPC流量, 80Gbps用于EBS/S3存储. 一个CPU Tray包含2颗Intel SPR处理器, 并连接8个Trn2 Compute Tray, CPU和GPU配比为1:8

在Trainuium3上, 分为Gen1和Gen2 两个版本, Gen1为双机柜构建的64卡配置, 整体结构和Trn2类似, 独立的CPU Tray和Trn3 Compute Tray. Trn3 Compute Tray支持2颗Trn3芯片, 可以配置一张或者两张400Gbps的Nitro v6卡. 而CPU Try和Trn2相同, 采用X86的架构配置独立的FrontEnd Nitro, CPU和GPU配比依旧为1:8

而在Gen2的机柜架构上, CPU和GPU整合在一个同一个ComputeTray内, CPU采用了Graviton 4, CPU和GPU的配比变成了1:4. 整个ComputeTray构建来看, FrontEnd和ScaleOut产生了融合, 并没有独立的FrontEnd Nitro. 是否在这个平台上实现了ScaleOut和FrontEnd的融合?

同时从SemiAnalysis的消息来看Trn3 Gen 72x2的机柜ComputeTray支持两种部署

2个Trn3共享一块Nitro-V6 , 平均每卡带宽200Gbps

2个Trn3独享一块Nitro-V6 , 平均每卡带宽400Gbps

现场展示的还是每个Trn3独立配置的Nitro-V6, 然后可以看到并没有给CPU配置专用的Nitro-V6, 而是让CPU和其它4个Nitro-V6共享带宽, 中间仅有一根绿色的RJ45的Cable用于带外管理.两侧有4根400Gbps的光纤.

![图片](assets/b3e67763b188.png)

实质上走向了ScaleOut和FrontEnd的融合路线, 也就是说Trn3配置的Nitro-V6有通过VPC连接存储的能力.

### 1.7 软件架构

AWS对于软件架构设计上也想的挺清楚了, 和Tilelang类似分为三类开发者:

![图片](assets/cb92fe2866d2.png)

最上层的ML Devlopers可能更关注的是如何调用推理引擎加载已经优化好的模型, 如何与大量的功能强大的第三方库整合, 通常他们仅涉及一些推理框架并加载模型来部署应用.

![图片](assets/12d3333c4a47.jpg)

中间是一些算法研究员, 他们关注的是研发新的模型和操作算子, 并且快速迭代. 他们需要的是快速流畅的开发过程保证新的算法实验能够快速的进行, 而并不需要足够极致的性能. 通常这一层以Native Pytorch/Jax为主.

![图片](assets/1b1d6e3c7557.jpg)

最后是一些性能优化工程师, 在Infra这侧和底层硬件优化紧密结合, 写算子和Profiling等工作, 充分打满整个算力芯片.

![图片](assets/e8cb789262ca.jpg)

AWS针对Trn提供了Neuron Kernel Interface(NKI)进行算子编程, 同时提供了Neuron Explorer作为Profiling的工具.

NKI和最近的CuTile/TileLang类似, 都是Python Based DSL, 然后支持Tile Level的编程, 然后说是这个Compiler会完全的开源出来.

![图片](assets/78b1ef0514d0.png)

另一方面在Re:invent的session里演示了Neuron Explorer, 整体来看完成度很高, 由于很多高频的trace可以存放到SRAM上, 因此它可以支持以硬件指令集的粒度来展现NerounCore的执行过程并达到ns级别的精度, 同时Profiler开启后对性能影响也很小.  另一方面后处理能够把这些Trace串接起来形成一个完整的详细的Device级别/系统级别的性能报告

![图片](assets/3dc33fc64dbd.png)

例如对于一些集合通信的离群值也可以做到很好的可视化:

![图片](assets/9ed979296f46.png)

## 2. 未来AI Infra的演进

### 2.1 前置技术背景

对于GPGPU的路线, 以NV为代表已经进行了详细的分析[《Inside Nvidia GPU: 谈谈Blackwell的不足并预测一下Rubin的微架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21#wechat_redirect). 对于TPU/Trn这类的ASIC架构前面一个章节也有详细分析. 对于ScaleUP和ScaleOut互连也可以参考[《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect).

### 2.2 从业务的视角分析

从业务视角来看, 我们首先需要找准AI下一步Scaling的方向, 如下图所示:

![图片](assets/4f3a8c4bc56e.png)

而整个AI Scaling的方向逐渐转移到了RL后学习的阶段, Reasoning模型及相应的DeepResearch的处理以及Agentic+具身的路线上. 对于RL后训练的工作负载分布来看, Rollout占比更大, 而后面两个也都是以推理为主. AWS Re:invent 也提出了类似的业务视角:

![图片](assets/4abf8691d8c4.png)

AWS得出的结论对AI Infra的需求有4点:

Reasoning模型和Agentic需要更长的Context, 针对Attn block  的复杂度需要Sparse-Attn和Linear-Attn等新的算法, 我们将在下一节详细展开, 另一方面也就是PD分离的需求以及超长Context的KVCache存储的需求.

支持对于MoE模型这类模型繁重的通信需求.

训推一体, 支持预训练/后训练和推理在同一个系统运行.

支持超大规模并发的Agent执行和独立操作的能力, 这里Agent本身的执行需要租户间的隔离并要求AI Infra和通用CPU计算集群更好的互连和弹性.

#### 2.2.1 预训练业务

首先对于预训练阶段, 当前模型参数的规模受制于训练数据量的约束正在放缓, 有一个非常粗略简单的从信息压缩的视角来看, 人类数据的规模在30T Tokens的规模来看, 信息压缩比做到10:1时, 模型总参数规模粗略的估计上限为3T参数, 因此整体模型的参数规模增长逐渐在放缓.

当采用超节点的机型后, 我们可以把TP和EP的流量尽量的放置在ScaleUp域内, ScaleOut的流量仅承载一些PP和DP的流量, 通常这些流量也可以很好的被Overlap, 这就带来了一个问题:

**是否可以适当的缩减ScaleOut带宽,或者直接并入FrontEnd网络?**
正方的观点
从业务的视角来看其实FrontEnd配合一些 Hierarchy 的集合通信和Infra的Overlap并适当的扩大FrontEnd域的带宽应该就能满足需求. 避免使用ScaleOut网络可以将整体的成本下降10%~20%, 电力开销也会相应的降低.

但是FrontEnd承载RDMA业务本身的难度, 特别是和VPC内其它TCP流量和存储流量混跑以及网络Hash冲突/拥塞控制等这一系列问题的处理上, **Nvidia的RoCE网卡是有很大的局限性的**, 这也NV一直在宣传FrontEnd作为N-W流量承载, ScaleOut作为E-W流量承载, 并且存储也需要独立组网的方案.

![图片](assets/a0fe9b53ee56.png)

而事实上工业界能够完全解决这个问题的, 只有三家: AWS SRD, Google Falcon和阿里云的CIPU eRDMA. AWS SRD并不支持标准的RDMA Verbs接口, 而Google Falcon多路径算法上还有缺陷以及无法支持Scale-Across的长传, 事实上能做的就只剩下一家.

我们看到AWS在Trn3 Gen2上已经有了融和FrontEnd和ScaleOut的解决方案, 并且也有相对低带宽版本(每卡200Gbps)的实例提供.
反方的观点
对于算法和Infra的同学来看, 有一个很简单的逻辑, 只要你有这么大的带宽, 那么我一定会想尽办法调度通信尽力打满. 另一个观点是从CSP的视角来看,我们并不希望训练和推理分池, 例如训练和推理比例还是在大于1:5的时候, 省去ScaleOut的成本可能非常吸引人, 但是另一方面有可能导致训练集群开不出来影响整体的售卖率. 另一方面通常会用最新的卡去做训练, 老一点的卡去做推理, 因此ScaleOut的成本摊销上不能简单的以整生命周期的成本摊销, 例如一个生命周期5年的集群, ScaleOut网络的成本摊销到推理上时应该以某种残值计算.

另外对于生命周期早期的新卡溢价相对较高, 如果一个没有ScaleOut的集群这一年的收入和其它竞争对手相比的折扣是否也会导致节省ScaleOut网络的成本优势丧失也许需要一个很好的核算.

注意到从GB200开始, 老黄一直把整个故事的叙事逻辑演进到了推理, 是否未来会出现部署的新卡中超过70%的比例在用于推理? 那么整体的ROI模型又需要重估. 从一些口径分析明年美国的新建卡总体规模已经接近1600万卡, 新卡训推比的分布和一些Neocloud/CSP是否会构建无需ScaleOut的实例可能很快就有答案.

#### 2.2.2 RL后训练

对于RL后训练而言, Rollout的时间占比远高于训练, 因此它是一个非常重推理的业务. 推理本身可能更重要的是EP的通信流量 因此构建超节点某种意义上成为了刚需. 在一些超节点的集群上, 训练参数的更新等都可以在ScaleUP上完成,或者构建一个Hierarchy的集合通信, 理论上省掉ScaleOut网络的E2E性能差异会非常小.

#### 2.2.3 Reasoning模型和Agentic

对于大量的Reasoning模型, 特别是配合进行的一些DeepResearch的任务通常还需要调用外部工具和并发的Agent执行(例如WideResearch). 另一方面大规模的Agent并发执行和多租户服务的要求, 客户很有可能将这些Agent放在自己的VPC内和已有的系统/应用融合, 例如千问+高德+淘宝+飞猪+钉钉等.. 从这一方面来看对FrontEnd高并发的需求也在进一步提升.

另外就是这类workload通常需要更长的context, 因此对PD分离和KV-Cache提出了更高的需求, 层次化的KVCache逐渐会成为刚需.

最后, Agentic也对通用计算提出了更高的弹性需求, 通常一个复杂的Agent任务涉及多步的业务调用, 而对于某些应用延迟有非常严苛的需求. 整个链路上的长尾延迟将对SLA带来巨大的影响. 因此对于Agent执行环境的安全隔离/快速拉起, 以及高并发能力提出了新的挑战. 原来我们一个人可能手工只能同时操作一台电脑/手机, 而在Agentic时代很有可能是一个模型在帮我们同时操作100台电脑完成更复杂的任务. 因此通用计算的规模也会伴随着Agentic业务成熟而成倍的增长. 对于仅建立GPU集群的一些算力中心而言将会在通用计算资源和存储资源池化和弹性供给上带来巨大的挑战.

简单的做过一个估算, 对于一个日活2000w的Agentic的业务可能需要的存储规模为每日写入超过200PB的数据量, 并发的Agent执行环境需要几百万台VM.

#### 2.2.4 小结

总体来看, 对于纯推理平台(Reasoning+Agentic)和RL后训练似乎对ScaleOut这个backend孤岛的需求并不是那么的强烈, 但是对于预训练场景还存在一些业务争议和不确定性. 这部分内容我们搁置争议, 主要来看其它两个方面的问题.

**EP并行**: 基本上是确定的需要支持基于内存语义ScaleUP网络构建的超节点, 唯一不确定的是EP的规模和超节点的规模, 这部分我们在下一节算法演进趋势来详细展开.

**FrontEnd网络需求**: 主要是模型推理的结果需要传输到Agent节点运行, 而Agent大量部署在用户的VPC内.

**PD分离**: EP这类业务在ScaleOut RDMA消息语义上本来就不友好, 即便放在Scaleout和PD混跑也有很大的互相干扰, 而KVCache也有接入存储构建HiCache的需求, 通常会将其放入到FrontEnd网络中, FrontEnd和ScaleOut融合还可以进一步将KVCache传输通过GDA/GDS直接加载到GPU显存内, 这一点可以参考阿里云数据库团队和SGLang构建的HiCache和《阿里云 Tair 基于 3FS 工程化落地 KVCache：企业级部署、高可用运维与性能调优实践》

至于某个吹的很厉害的分离方法, 开局就是EP也有同样的问题. 其实我自己今年年初分析完没收益就无视了, 然后某厂吹了一波, 另外某个厂真跳坑做了, 前几天公开讲测试出来50ms SLO性能下降1%, 100ms TPOT才有19%性能收益, 这个SLO就不是可用级别的, 那就不提了……

### 2.3 算法演进

首先我们需要从算法最基础的那几个点来阐述, 智能是什么? 很简单的一个观点是: **它是一个超高维度空间内按照某种结构压缩的低维流形(low-dimensional manifold)**

#### 2.3.1 从物理的视角

从物理世界的角度, Demis Hassabis有一个假设, 自然界的很多规律并不需要写出显示的方程, 而是可以通过在经典的图灵机上以数据压缩的方式学习出来. 例如AlphaFold预测数月甚至数年才能解析的蛋白质折叠结构. 从蛋白质的理论来看是一个10^300的空间, 全无法穷举或物理模拟, 但自然界中蛋白能在毫秒级自动完成折叠, 事实上就是在超高维度空间内按照某种结构压缩的低维流形.  因此自然的行为模式在高维空间中稀疏分布、结构清晰、路径稳定——它们集中在一种可压缩、可调度的结构空间中, 这样就构成了一个低维的流形.  AlphaFold的本质也是如此, 它用深度神经网络从数据中提取出了低维流形，并在这个结构压缩空间中完成了调度和推理. 无需理解所有的物理机制, 而是通过在流形上掌握了“自然允许你走的那些路径”. 其实具身智能的底层逻辑也是如此.

#### 2.3.2 从数学的视角

继续从数学的角度来看, 其实一直在做这方面的工作, 可以详细展开看看一个专题

[《大模型时代的数学基础》 ](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

简单的一个结论是: **这一次人工智能革命的数学基础是：范畴论/代数拓扑/代数几何这些二十世纪的数学第一登上商用计算的舞台.**

事实上正如前一节的观点, 智能是超高维度空间内按照某种结构压缩的低维流形. 那么相应的代数拓扑/代数几何则是对于结构化压缩可以进行很好描述的工具. 例如代数拓扑通过一些代数不变量来区分和分类不同的拓扑空间, 这样对结构化压缩的低维流形在算法层面描述“自然允许你走的那些路径”, 背后的逻辑是从高维空间构造代数不变量, 然后再给予流形某种结构化的约束.

另一个更基础的研究是在范畴论的基础上展开, 在计算范式上如何构造一个合理的代数系统, 以及在算法研究过程中从代数结构上排除一些不必要的算法探索实验. 那么一个很简单做法就是把Attention看作是范畴论中的态射, 然后预训练的实质是在构造一个Presheaf. 实质结论来自于Yoneda Lemma, 即 你不需要知道一个对象的"内部构造", 只需要知道它如何与外界"交互"(即所有射向其他对象的箭头集合), 就能完全理解它. 这种描述和**超高维度空间内按照某种结构压缩的低维流形(low-dimensional manifold)**的实质是一样的.

然后进一步通过Nerve构造来在一个更高维范畴.  事实上这样就引入了模空间的概念. 模空间 (Moduli space) 是代数几何中的一个核心概念. 它是一个几何空间, 其上的每一点都对应着一个特定类型的代数几何对象的同构类. 模空间的研究旨在为这些几何对象的分类问题提供一个统一的框架, 并通过研究模空间本身的几何和拓扑性质, 来反过来理解这些几何对象的性质.模空间的核心思想是“为对象建立一个几何目录”

例如语言中的同义词, 相同含义的内容通过不同国家语言的描述等, 我们都可以看作它们为背后蕴藏的知识本身的同构类. 当对象之间存在复杂的等价关系(同构)时, 模空间本身就成了一个复杂的"空间". 神经构造(Nerve Construction)正是处理这种复杂"等价关系网"的系统性工具.

#### 2.3.3 从计算机体系结构的视角

从计算机体系结构的视角来看, 算力是很容易Scale的, 而访问内存是很难Scale的. 因此整体算法路径上需要一个合理的可训练的稀疏方式.

另一方面对于低维流形的实质是在经典的图灵机上以数据压缩的方式学习出来, 那么如何对于模型结构本身构成一个高并行的图灵机结构, 其实这个视角上解释了Transformer架构的有效性.

如果说最早的token by token的大模型推理是一个顺序纸带的图灵机. 而Reasoning模型出现本质更像是一个比较完备的图灵机了, 但似乎缺少一些纸带回退和擦除的能力. 这些回退和擦除或许是推理阶段节省复杂度的一个好方法. 毕竟我们解决很多问题的时候并不需要一张无限大的草稿纸, 大模型亦然.

当试图通过大模型架构构造一个自己能够产生代码运行的通用计算机架构, 即token as instruction. 当脑子里补出这图的时候, 就豁然开朗了. 大模型从自回归可能真的要走向自生成Instruction的路了.... 那么构造一个大模型的冯诺伊曼架构大概就如下了:

![图片](assets/dd6c59a1c8f4.png)

即Attn作为图灵机的计算控制, MoE作为存储器. 接下来我们从算法中的两个模块进行分开阐述.

#### 2.3.4 Attention

关于Attention的演进, 前段时间有一个分析[《谈谈未来Attention算法的选择, Full, Sparse or Linear ?》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496753&idx=1&sn=b66ffd8d2e977cb4e7e27603ea9a9951&scene=21#wechat_redirect), 实质性的问题还是继续进行稀疏化的处理. 从自然选择的角度来看, Sparse Attn中按Block的处理对于很多Agent任务拼接context也可以节省掉大量的Prefill的算力, 这方面也是相对Linear有很多优势的.

当然Linear的某些类似于RNN的属性还是很有价值的, 但是不妨碍我们在一些Sparse Attention中来引入一些递归循环的处理, 在访问内存不显著增加的情况下增加一些计算的Scale是可行的.

明年应该会有一些真正的变化在开源模型中展现出来.

#### 2.3.5 MoE

另一个比较重要的并值得关注的问题是MoE的未来规模. 部分以网络为背景的超节点厂商认为需要一卡一专家, 因此对于超节点的规模维持到了一个很大的规模. 但是我们看到NV在Rubin最大规模144卡, Trn3实际的EP并行单柜72卡.. 对于当前256个Expert, Topk=8的常见MoE模型, 通常在GB200上 EP32就够了.

那么一个关键的分析是, Expert数是否会超过256/384到1024/2048 , TopK到64? 答案是否定的. 这一点在[《谈谈ESUN, SUE和UALink》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496512&idx=2&sn=0c10cef05fb1cc4e175f326d62b266e3&scene=21#wechat_redirect)有详细的阐述. 实际的决策流程如下:

![图片](assets/b07ac0740faa.png)

首先从模型训练的角度来看, 专家数进一步提升稀疏性会导致训练崩塌以及后训练任务中训推不一致的问题加剧, 因此单层的专家总数最大可能会维持在512以内.

假设一个极限的稀疏的情况, 专家数目为M=512, K=8, batchsize N=32时, 需要访问的专家数目为202个. 当专家数扩展到1024个更极端的情况下时, 需要访问的专家数为227个. 这样就带来了大量的内存访问瓶颈. 通过EP并行, 例如每卡8个专家, 访问专家参数的带宽已经下降到原来的3%. 此时EP的规模为128卡.

结论: 从并行策略来分析, 满足交换机单层组网的Radix规模, 例如最大512卡即可.实际部署时可能考虑弹性缩扩容的需求, 还会进一步降低ScaleUP的规模. 您可以看到即便是Rubin Ultra的NVL576, Kyber机柜的背面可以看到, 单个ScaleUP域也只有144卡的规模. 并且NV还是在选择铜互连. 同样AWS Trn3服务器, AMD MI450x都在做类似的决策.

即便是Google号称通过OCS连接了一个9216卡的ScaleUP域, 但是它仅是一个调度域, 实际的单个实例的规格也是8x8x8= 512卡. 对于Google而言, 是否未来还会维持3D-Torus的拓扑也是一个变数.

### 2.4 从商业模式的视角分析

对于NeoCloud的这样业务模式存在的风险在[《谈谈GPU云的经营风险和流动性管理》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494332&idx=1&sn=29e146c61e958939d5348d6779b743c6&scene=21#wechat_redirect)中已经有详细的阐述. 实质性的问题是计算资源的弹性供给和流动性管理避免风险. 最近几个月几个NeoCloud和Oracle其实已经逐渐陷入到一个流动性风险中了, 而在未来的一年, 北美账面新建规模为1600万卡, 充足的供应下流动性风险将进一步凸显出来.

对于CSP而言, 弹性的训练和推理业务模式将会在GPU资源池规模足够大时变为刚需. 例如AWS在这次Re:invent上也在讨论Checkpointless和弹性训练《AWS re:Invent 2025 - SageMaker HyperPod: Checkpointless & elastic training for AI models (AIM3338)》[2]

而对于推理业务有更明显的峰谷效应下, MaaS构建更具有弹性的推理平台对于成本管理也具有极大的优势. 而实质性的技术问题就是: 存算分离和多租户隔离.

支持存算分离和多租户隔离的GPU实例, 实质上是完成了一个算力证券化的过程.

### 2.5 从基础设施视角分析

前面几节从应用/算法,再到经营的角度分析并阐述了一系列需求, 最终落地的对Infra的实现上.

#### 2.5.1 算力芯片微架构

其实需要回答的问题是DSA or GPGPU两条路线如何选择.

不妨我们从体系结构的视角来看, 通常的一些解决办法分为4个方向:

提高并行性

降低数值精度

更好的DataLocality

DSL

对于提高并行性而言, 无论是DSA还是GPGPU都在演进到Tensor Core + Vector Core + Scalar Core的架构. 对于数值精度而言, MXFP8/MXFP4/NVFP4等基于block scaling的低精度压缩也都在支持.

其实本质的区别是在最后两点, DataLocality的处理上, DSA需要更底层的去管理内存和排布流水, 对编译器的要求会比GPGPU更高. 而DSL则是在易用性上解决一些Datalocality复杂的问题, 最近几个月都在卷的一个方向, cutlass-dsl/cuda-tile/tilelang等..

前面有一个关于DataLocality的对比, DSA一类的通常就是一块很大的SRAM, 而GPGPU通常有更深的内存层次化结构.

![图片](assets/0f6b54c5e3db.png)

实际来看追求极致性能的Infra工程师可能对DSL的依赖并不大, 有些时候还不如自己内联一些指令方便. 其实在Tile-Based DSL架构下, 受众是配合算法开发的一些Infra同学, 他们有部分的内存管理提高DataLocality的需求, 但又不极致追求性能. 通常一个实验需要跑半个月, 因此算法研究员根本不会给配合他们的Infra同学长达一个月的算子开发时间. 那么给开发的时间周期可能少于几天, DSL便是一个很好的选择, 即便是这些临时的DSL的代码无法打满, 大概获得峰值性能的70%左右也是很有价值的.

那么唯一剩下的区别就是在内存子系统设计了, 实际上针对CUDA编程而言, 伴随着TensorCore的引入, 对于SMEM/TMEM的内存分配管理访问, 即便在Hopper/Blackwell上有更好的异步MBarrier的处理能力, 但整体编程的复杂度和用户心智的影响还是很大的. 特别来说在Blackwell上还存在跨物理Die引入的接近400Cycle的延迟差异, 对于一些较小的Kernel运算也有显著的影响.

DSA架构和GPGPU架构在Tile-Based DSL下, 对于内存管理的复杂度实际上是有显著降低的, 那么事实上DSA和GPGPU的区别在逐渐模糊化, 唯一的区别可能就在一些算力调度上, GPGPU的指令集发射/Warp调度等通常在加速器内部实现, 但也逐渐面临一些问题, 例如TMA/TensorCore的异步指令发射仅需要一个线程, 而通常又不得不调度到CUDA Core上一个Warp执行. 而Warp执行的时候为了保证吞吐, 很多指令发射又带有Stall Control-bits, 具体可以参考[《现代NVidia GPU架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494136&idx=1&sn=958bc734efd43a59b03c04a2e65ec408&scene=21#wechat_redirect), 因此对于低延迟的业务而言存在显著的缺陷.

这也就是RDMA并不是适合GPU微架构的根本原因, 主要包含几个方面: RDMA通信需要构造WQE, WQE构造对于SIMT CUDA Core是低效的并且延迟较大. 对于完成事件的通知机制上, RDMA无法直接通知并更新SMEM中的MBarrier,而是需要CUDA Core polling GMEM, 每次Polling会带来极大的延迟和Warp调度开销. 更进一步Completion Queue的数据结构对GPU处理也是消耗巨大的. 当然RDMA对于DSA架构也有同样的问题.

这也是为什么在MoE这样的EP并行处理上, 需要采用ScaleUP总线的原因. 而实质性的需求是使用对GPU和DSA微架构友好的内存语义.

#### 2.5.2 系统互连架构

先来一个最直接的结论(其实2023年底内部的资料也写过): **GPGPU/DSA这些算力芯片需要在整个系统互连体系中成为一等公民**. 简而言之就是要构建GPU Direct Everthing(GDE).

一个最朴素的观点是, GPU作为一个算力芯片, 它在替代CPU成为新一代的算力中心, 那么它理应成为整个数据中心的一等公民. 紧接着作为一等公民需要直接访问数据中心内的一切资源, 避免额外的开销. 例如KVCache的存储, 直接控制CPU-Agent实例, 直接访问EBS/OSS等云存储处理数据和存储checkpoint等.. 因此GPU Direct Everthing成为刚需. 最后是云的弹性经营的逻辑推导出云要多租安全隔离和存算分离实现弹性调度.

当前的GPU架构来看, 通常由三张网络构建, 如下图所示:

![图片](assets/1e73ecaf60ec.png)

Type1: Front-End Network： 用于VPC互联和访问EBS/OSS存储业务

Type2: Scale-Out Network：基于RDMA网卡的多机扩展互联网络

Type3: Scale-Up Network： 基于NVLink私有总线构建的多GPU机内互联网络

从整个云基础设施和弹性调度的视角来看, GPU在FrontEnd作为加速器模式挂载到CPU下的二等公民存在. 从经营的弹性调度视角来看, GPU需要接入存储并支持多租户的连接. 从业务的视角来看, Agentic/RL这些业务也需要更大的FrontEnd带宽, 以及更长的Context也需要GPU连接到一个 Hierarchy 的 KV-Cache 池中, 并且这个HiCache池也是一个弹性多租的业务模式来进一步提升整体资源的利用率. 同时对于弹性部署还需要更快的模型加载速度, 实际上Nvidia也在建议构建东西向的存储集群
![图片](assets/c80bc9f20d28.png)

实质性的结论也是**GPU需要在接入存储成为一等公民**
关于EP并行
当前的一些变化是, 对于EP一类的流量需要支持内存语义的超节点, ScaleOut承载EP并行还是存在一些限制. 而在超节点的演进过程中, 单个超节点通常包含64~144卡, 实际上我们可以看作是GPU已经在单机柜内完成了基于内存语义的ScaleOut从8卡扩展到了64~144卡. 对于一个GPU而言算力所需要的网络带宽是有一个上限的, 它毕竟不是一个网络交换芯片, 大量的计算/访问内存瓶颈决定了这个上限. 因此在超节点机型上ScaleOut是否还需要这样大的带宽是一个有争议的话题. 与之相反的是排除超节点, 对于那些依旧维持8卡架构的B200/B300服务器, 对ScaleOut的带宽需求随着算力的提升还在增加, 这部分实例ScaleOut是一个刚需.
关于PD分离
从业务的视角上我们还需要考虑PD分离部署下的技术选择, 对于Rubin CPX方案的分析可以参考《详细分析一下Nvidia Rubin CPX》. 它分为两种部署, 一种是直接在NVL144机柜内将Rubin CPX串接在ScaleOut网卡和Vera CPU之间.

![图片](assets/182d7f221e2f.png)

另一种做法是采用VR CPX + VR NVL144 Dual Rack双机柜的部署, 利用ScaleOut网络互连.

官方的两种方案, VR CPX NVL144采用固定配比, 并且丧失了ScaleOut GDR的能力, 而Dual-Rack方案虽然天然的支持xPyD, 但又会导致在ScaleOut上同时进行KVCache传输和EP并行的流量产生干扰的问题, 同时多了很多Vera CPU和CX9网卡. 通常一张CX9网卡的售价应该在2000美金作为, 而一张Rubin CPX的售价估计应该在3000~5000美金附近, 再摊销ScaleOut光纤和交换网络的成本到Rubin CPX,  老黄那句“Buy More, save More”是否成立?

另一种成本核算的视角就是采用FrontEnd承载PD分离的流量, 并引入层级化的KVCache缓存. 但是对于PD分离的流量GPU需要通过PCIeSwitch连接到CPU, 然后再通过CPU直连的DPU传输KVCache, 这样的路径来看, PCIeSwitch有天然的收敛比, 同时CPU的内存子系统也会受到大带宽冲击.

那么接下来FrontEnd和ScaleOut融合直接将GPU作为一等公民是否可以避免这样的问题呢?
FrontEnd和ScaleOut融合
那么一个简单的Trade-off, 对于超节点集群适当的提升FrontEnd带宽能否承载ScaleOut的流量, 实质性的目的是把GPU提升成为一等公民符合业务演进的逻辑.

业务逻辑是非常朴素的, 但是技术挑战是巨大的, 特别来说RDMA/RoCE本身就是一个不完善的可靠传输协议, 对于商用RDMA/RoCE, 当它和其它流量混跑时互相的干扰带来的性能下降是非常明显的, 因此独立构建一个RDMA专用的ScaleOut网络是有业务收益的.

但是这个技术挑战已经被解决了, 无论是AWS SRD还是Google Falcon都有完善的解决方案. 而阿里云CIPU eRDMA更是在所有的8代以上的通用计算实例上基于VPC构建了RDMA的能力. RDMA和TCP混跑甚至和存储流量并网对于我们并没有什么难度. 因此将GPU作为一等公民技术上是成立的.

大概就写这么多吧, 还有一些涉密的就不展开了.....

参考资料

[1] 
AWS re:Invent 2025 - AWS Trn3 UltraServers: Power next-generation enterprise AI performance(AIM3335): *https://www.youtube.com/watch?v=c_1FhdXNUSE&list=PL2yQDdvlhXf-UqnINCmXu-dDZJm_B3bbJ&index=26*
[2] 
AWS re:Invent 2025 - SageMaker HyperPod: Checkpointless & elastic training for AI models (AIM3338): *https://www.youtube.com/watch?v=r9J10L2K0F4&list=PL2yQDdvlhXf-UqnINCmXu-dDZJm_B3bbJ&index=6*