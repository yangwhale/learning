# HotChip2024后记:  谈谈加速器互联及ScaleUP为什么不能用RDMA

> 作者: zartbot  
> 日期: 2024年8月30日 11:47  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492300&idx=1&sn=8a239883c831233e7e06659ec3425ea2&chksm=f995f20ecee27b185a42d09868bdf6cef64df38267489ee386b6d4425e57d2c2a699ada0f9cb#rd

---

### TL;DR

专门写一篇来谈谈HotChip 2024介绍的这些AI加速器的互联系统, 各家在互联上的设计有很多不同

从`传输语义`的视角, 使用Ethernet互联的厂家中, 有RDMA**这种`消息语义`的MAIA 100和Gaudi3, 也有直接`内存语义`的Tesla TTP及用RingBuffer的Testorrent.

从`网络融合`的视角, 有ScaleUP和ScaleOut融合的Microsoft MAIA 100, 也有支持有损FrontEnd和ScaleOut融合的Tesla TTPoE.

其实你仔细看Tesla是完全实现了ScaleUP/ScaleOut/FrontEnd三网融合的技术, 并且支持租户隔离, 这也是我一直以来的技术观点, 并且在NetDAM上和Tesla基本同一时期实现的(都在2020年~2021年期间实现的).

不同的设计带来不同的观点和争议, 但是很多观点都是相对片面的, 例如不讨论AI加速器的微架构谈互联协议, 或者不考虑实际的应用场景和业务特性而简单的把A技术应用到B场景, 例如直接把ScaleOut RoCE带宽做大是否能替代ScaleUP? 本文将针对这些问题进行一个详细的分析.

同时我们需要对基于Ethernet的ScaleUP网络定义做一个严格的区分:

EthScaleUP-Type1: 基于标准以太网实现的类似于RDMA的消息语义互联, 例如Gaudi3, MAIA 100等

EthScaleUP-Type2: 基于标准以太网实现的并支持内存语义互联的协议, 例如Tesla TTPoE, Tenstorrent等, 如果UALink over Ethernet,也算.这条路线当然还包括NetDAM, 后面我会详细介绍TTPoE的MAC实现和NetDAM几乎是完全一致的.

对于修改以太网MAC层和PCS层协议的不在本文讨论范围. **本质上选择以太网就干干净净的去薅它的羊毛**, 例如Tesla TTPoE. 以太网的最大优势是通过DCN的量来摊薄部署成本. 如果真要极致的去修改MAC/PCS, 还不如直接复用Serdes完全放弃以太网自己单独搞一套.

本文的关键是LD/ST是否需要以及如何实施以及Cache一致性的处理. 先说结论: **Cache一致性不需要,但LD/ST是必须的, 并且LD/ST不能直接简单穿越以太网**

## 1. 从AI加速器微架构和片上网络看ScaleUP

对于很多网络工程师来看, RDMA的延迟和NVLink相当, 当我们把RDMA ScaleOut网络的带宽扩大到和NVLink一样后,是否就可以替代ScaleUP了? 对于这个问题, 这里先做一点补充.

### 1.1 从应用生态的视角

从应用的视角来看, NVLink是一个标准的计算域的总线, 支持LD/ST是一个非常自然的过程, 如下所示, 可以看到分配不同卡的内存空间后, 只需要在调用Kernel时将相应的指针传入即可, 对于执行的线程而言,如同访问本地GPU一样.

```
__global__ void SimpleKernel(float *local_mem, float *remote_mem){    const int idx = blockIdx.x * blockDim.x + threadIdx.x;    remote_mem[idx] = local_mem[idx] * 2.0f;}int main() {    uint32_t size = pow(2, 30);    // Memory Copy Size    float *dev_0;    cudaSetDevice(0);    cudaMalloc((void **)&dev_0, size);  // GPU0:内存分配    cudaDeviceEnablePeerAccess(1, 0);   // Enable P2P    float *dev_1;    cudaSetDevice(1);    cudaMalloc((void **)&dev_1, size);  // GPU1:内存分配    cudaDeviceEnablePeerAccess(0, 0);   // Enable P2P    cudaSetDevice(0);    SimpleKernel<<<16, 32>>>(dev_0, dev_1);// 执行GPU0 Kernel    cudaSetDevice(1);    SimpleKernel<<<16, 32>>>(dev_1, dev_0);// 执行GPU1 Kernel    // Clean Up    cudaFree(dev_0);    cudaFree(dev_1);}
```

这是处理器最自然的编程方式, 计算核心在片上网络和内存总线中的地位是对等的. 当需要处理大量数据时, 通常可以调用cudaMemcpyDeviceToDevice的方式进行.

```
  // Init Stream  cudaStream_t stream;  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);  //Do a P2P memcpy  cudaMemcpyAsync(dev_0, dev_1, size, cudaMemcpyDeviceToDevice, stream);  cudaStreamSynchronize(stream);
```

即便是没有NVLink的时代,这些代码就已经大量存在了,并通过PCIe P2P执行卡间的数据访问.

**从生态兼容的角度来看, 这些内存语义是需要保留的.**

### 1.2 RDMA通信方式

关于RDMA, 满多做计算的可能也不太熟悉, 这里简单讲一下. 它并不是字面上那种Remote DMA, 而是需要很多额外的处理. RDMA通信是`基于消息`的, 首先需要定义Memory-Region, 然后创建连接并构建工作队列(WQ), 然后将发送队列(SQ),接收队列(RQ), 并且还有一个完成队列(CQ). 针对不同的目的地设备都需要构建这样的Queue Pair Context. RDMA写一笔数据的过程远比LD/ST复杂

![图片](assets/8c0ce7d4fbfd.png)

(1) 请求端应用程序向QP下发一次Write WR(Work Request)。

(2) WR以WQE的形式被添加至SQ中。

(3) 请求端网卡从SQ中取出WQE，获取Write操作的任务信息。

(4) 请求端网卡根据WQE中的信息，从内存中获取待发送的数据并进行数据封装。

(5) 请求端网卡通过物理链路将数据发送到响应端。

(6) 响应端网卡收到数据后，对数据进行解析和校验，校验通过后将数据写入指定的内存区域中。

(7) 响应端回复确认信息给请求端网卡。

(8) 请求端网卡收到响应端回复的确认信息后，生成CQE并添加至CQ中。

(9) CQE以WC的形式返回给请求端应用程序，通知应用程序任务已完成。

注: 知乎上《RDMA杂谈》[1] 以及H3C的一份简介《什么是RDMA？》[2] 都是不错的材料, 本文这一节也引用了这些内容.

简单的说, 基于RDMA通信需要准备WQE, 通过Doorbell通知网卡, 然后对消息的完成通过中断或者Polling处理

![图片](assets/83e9c042df37.png)

### 1.3 RDMA做ScaleUP的难题

CPU在整个通信的关键路径上, 处理这些WQ/Doorbell和CQE.如下图所示

![图片](assets/6523dbe8ef06.png)

#### 1.3.1 延迟问题

虽然空载的时候测量IB的延迟是非常低的, 但是CPU在关键路径上, 进程间切换和CPU/GPU通知机制会带来较大的抖动, 另一方面在Hopper之前, 通过PCIe写入时, 当CPU收到Completion还会再通过Loopback QP去读一次做Fence还会进一步增加延迟. 如果片面的把ScaleUP用ScaleOut替代, CPU处理负载增高会带来更多的延迟和抖动.

#### 1.3.2 ScaleUP带宽导致CPU过载

当前ScaleOut带宽和message per seconds的处理上, CPU是能够承受的. 但是扩展到ScaleUP带宽扩大10倍. 即便是CPU侧不承载关键的数据路径, 但是准备WQE和处理CQ都会带来大量的指令开销导致CPU过载, 另一方面CPU-GPU的PCIe总线出现过载而产生更高的延迟抖动.

#### 1.3.3 QP爆炸和芯片功耗和面积的考虑

在芯片上集成大量的RDMA网卡带来的芯片面积的开销很大, 直接集成RDMA的AI加速器通常由于受芯片面积的约束,无法缓存大量的QP Context, 因此支持的QP数量相对较少, 而对一些AlltoAll场景是无法很好的承载的, 如果要支持更大的QP数量,势必要增加更多的SRAM给RDMA Die,从芯片设计的角度考虑, 与其用这些芯片面积来搞RDMA还不如多加一堆算力核?

#### 1.3.4 L2 Cache和内存子系统的干扰

GPGPU的缓存结构和CPU有很大的不同, GPGPU通常L1和ScratchPad的非常大,而L2Cache则受限于芯片架构通常小于L1构成一个腰形结构

![图片](assets/62a782b33e22.png)

另一方面我们可以注意到AMD的MI300X, 其ScaleOut支持的RDMA内存为总内存的1/8, 为什么?

![图片](assets/ba7f1ee7465d.png)

实际上的原因就是如果这些要穿越整个Infinity Fabric会带来干扰, 因此做出了一些限制避免干扰

![图片](assets/8448739b379b.jpg)

因此我们需要考虑在此之上执行ScaleUP通信时, L2Cache的瓶颈. 特别是在Transformer这些memory bound和compute bound都几乎打满的算子上. 我们更需要细致的考虑这个问题.

#### 1.3.5 传输效率

从RoCEv2的视角来看, 一个报文需要Eth Header(14B) + IP Header(20B) + UDP(8B)= 42B. 还不算BTH(12B)/RETH(16B)/AETH(4B)的开销. 然后还有以太网CRC 4B, 前导帧8B, IFG 12B带来的开销. 对于片上常见的128B传输效率是非常低下的. 这也是有很多人有动机去改以太网协议的原因.

而看看UEC里BRCM的提案和Tesla TTPoE的实现, 本质上做一个MAC路由就能解决的问题, 非要搞的那么复杂.

#### 1.3.6 乱序提交, Lossy支持, 负载均衡

RDMA里面最关键的问题还是在乱序提交和Lossy, Mellanox现在是无法解决的, Meta呢一如即往的用DeepBuffer交换机来缓解PFC风暴. 各家多轨道组网等等搞了一大堆东西, 还是没触及问题的核心.

大概也只有Tesla TTPoE, Google Falcon 还有我们是完全明白这事该怎么搞的.

### 1.4 RDMA做ScaleUP的缓解办法

当然对于前面的一些问题, 肯定有人会反驳, NV有GDA-KI, Intel Gaudi3和微软MAIA 100 都可以做, 有什么不行的? 我们在这一节来谈谈一些缓解的办法.

#### 1.4.1 GDA-KI

Nvidia为了避免CPU在关键路径上引起的延迟问题, 采用了GPU的Cuda Core来构建WQE和敲Doobell并处理CQE的方式. 这种被称为GPUDirect-Async- Kernel Initiated

![图片](assets/44cda594c1ec.png)

对于Doorbell的问题, 可以一个Warp内聚合, 但是大量的per-core的WQE处理也是一个难事.

![图片](assets/8b9f00c74519.png)

但是即便这样, 我们还需要考虑一些问题: 在LD/ST语义下, Local寄存器和SMEM的数据可以很容易的写到远端处理器上. 而针对RDMA语义来看, 或许只能先写穿到HBM, 然后再敲Doorbell. 当然进一步可以把GPU的Shared Memory去动态注册成MR, 其实这样的实现还要考虑到Warp调度器相关的实现, 以及片上网络的一些拥塞, 感觉挺烦的. 通常我们在计算时会通过异步的方式Overlap内存拷贝和算子计算, 如何将数据快速的灌入SMEM, 然后触发TensorCore计算, 完了以后还有大量Memory bound的elementwise的计算. 在这个过程中混入大量的准备WQE的指令是得不偿失的.

另一方面就是针对Cache的Bank冲突以及如何在多个ScaleUP link上负载均衡的处理都是需要考虑的

#### 1.4.2 Intel Gaudi3

对于Gaudi3这样的平台, 内部是一个VLIW指令的系统, 如果由它来产生WQE指令的效率会产生一些问题.

![图片](assets/888317c7a6f3.jpg)

然后对于它而言, 算子融合可能也有一些问题, 因为GEMM完成后一些element-wise的计算需要近内存的一系列ALU实施

![图片](assets/158f91a6fc04.png)

对于多机并行, 通常需要等待这些NMC(Near Memory Compute)完成后,才能通过ScaleUP link 进行搬运, 因此在它的控制面上需要专用的一个中断管理器/同步管理/调度管理

![图片](assets/21bb9214c1b0.jpg)

所以在看到Intel使用RoCE做ScaleUP时, 实际上是有很大代价的, WQE的填写/CQE的处理/调度器的影响等都有可能影响到片间的通信效率.

#### 1.4.3 微软MAIA 100

它也需要每个Tile上放置一个控制处理器(Tile Control Processor, TCP)

![图片](assets/cc8995696978.png)

然后在通信上也需要这样的处理器协助,通过信号量来实现异步编程的能力

![图片](assets/dc5807207bad.png)

### 1.5 结论: RDMA不适合ScaleUP网络

综上所述, 如果将RDMA用于ScaleUP网络, 将会出现一系列问题:

控制路径在CPU上,随着带宽激增,消息数量到10亿每秒的级别时, 通用处理器将成为瓶颈, 同时带来大量的延迟和抖动.影响性能.

WQE/CQE的处理如果Offload到AI加速器上, 对于GPGPU架构的处理器是可以做到的, 但是对于一些DSA的AI加速器是无法支持的, 都需要额外的控制处理器和相应的中断处理/同步处理等器件, 这样带来了很大的复杂性.

即便是片上有CudaCore或者专用控制处理器, 通信过程中WQE处理带来的指令开销, 大量的Doorbell如何处理, 多个RDMA控制器如何负载均衡也是难题.

对于CQE的处理, 是采用中断的方式还是Polling的方式, 中断的设计会严重影响到Warp调度策略, 而大量的计算核Polling带来的L2Cache访问压力, 也会同样影响性能, 当然局部再搞一些Cache Coherent的片上网络? 似乎也没必要.

传输效率的问题, Eth Header(14B) + IP Header(20B) + UDP(8B)= 42B. 还不算BTH(12B)/RETH(16B)/AETH(4B)的开销. 然后还有以太网CRC 4B, 前导帧8B, IFG 12B带来的开销. 对于片上常见的128B传输效率是非常低下

正如前文分析的, 要么需要GPGPU拿出核来准备WQE带来大量的指令开销, 要么就需要芯片上构建专用的控制处理器, 也会带来芯片面积的开销.

RDMA通信是有状态的, 即需要相应的QP Context, 这些Context也需要专用的Cache, 相当于原有的LD/ST 128B基础上要增加一次QPC读,多次WQE准备写, 网卡多次WQE读, CQE读写和Polling等开销.整个传输效率是受限的.

RDMA ScaleUP对于拥塞控制多路径/Out-Of-Order/Lossy还有一大堆问题需要解决.

## 2. 什么协议适合ScaleUP网络

我们需要有一个直接的连接片上网络的接口, AXI这些总线对于计算域的芯片架构师是熟悉的.

对计算域提供LD/ST/Atomic/DMA支持

我们需要一个高Radix和大带宽并且有足够的低的成本的交换网络, 以太网的规模效应值得我们去薅羊毛

负载均衡/乱序提交能力

支持基于Lossy的组网能力

### 2.1 对处理器要直接连接NOC并支持计算域内存语义

正如第一章分析的, RDMA消息语义对于AI加速器的架构上有很大的依赖, 对于DSA架构的需要独立的控制处理器, 即便是GPGPU也会面临大量的WQE准备/Doorbell和Polling CQE的开销, 同时编程方式会更复杂. 另一方面对于计算域的芯片架构师来看, 直接拉到NOC上能够处理内存语义即可, 为什么要旁路上搞那么多WQE控制呢? 还要进一步考虑保序能力.

### 2.2 以太网的羊毛要薅

对于ScaleUP交换机而言, 我一直以来的观点就是要去薅以太网的羊毛. 交换容量和接口演进速度非常快, 单口1.6Tbps和单交换机102.4T伴随着大规模商用, 有通用的DCN来摊薄成本. 而如果使用专用芯片难度是非常高的,特别是国内的一些GPU厂商来看, 既要做GPU又要做交换芯片大概也就华为可以. 所以薅以太网的羊毛是显而易见的决策路径.

其实你会看到符合1~3这几条的, 正是Tesla TTPoE

![图片](assets/d94945e4229d.png)

NOC上是一个标准的内存语义, 支持64B/cycle的NOC报文,然后TTP MAC可以将其合并成1KB的TTP报文发送.  其实这些内容渣在几年前做NetDAM时就分析清楚的

![图片](assets/40fce3f1e81f.png)

对北向暴露内存语义接口,而南向薅以太网的羊毛.

### 2.3 以太网如何承载内存语义

其实非常简单, 在以太网控制器上加一块内存就行了, 就这么简单.

![图片](assets/7e86ed98e61f.png)

而这个Dumb-NIC真是活生生的打脸各种SmartNIC和DPU, 还有新造的名词SuperNIC. 我还记得当时搞NetDAM的时候, 很多人看不懂, 还在那里说:“不就是网卡上加块内存么?” 确实, 就是这个看似Dumb的事情, 解决了大问题

## [《DPU新范式: 网络大坝和可编程存内计算》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486644&idx=1&sn=a2a18f661c18bfb96a37d5ac0d1a9653&chksm=f9961c76cee1956091037f97b52d420008c2d575c9ce2478ee12707c1336609882c90fae1a28&scene=21#wechat_redirect)

![图片](assets/8c532c4a5c4c.png)

而Tesla TPPoE也有两个版本, 实现了统一的内存访问.一个是带HBM版本的的Dojo Interface Processor(DIP)

![图片](assets/45212621c5ba.png)

它用于存储模型参数/训练数据/激活/梯度等信息, 您可以认为它是将多个HBM分布式的接在了DoJo SOW边缘, 为训练芯片提供大容量的分布式内存池

![图片](assets/08e32d5b079a.png)

而Dumb-NIC是一个100G的用于和其它Mojo主机之间的通信, 可以理解为带HBM的DIP是作为ScaleUP的互联,

![图片](assets/4809af5e67a0.png)

为SOW的2D-Mesh提供一个Z轴, 降低在片上通信的跳数, 同时这些分布式的HBM构成一个极大的内存池, 这也是NetDAM曾做过的东西

![图片](assets/8dd025be4f3a.png)

而Scale-Out则是通过DIP和Dumb-NIC实现的

![图片](assets/6b21d357e38c.png)

最终通过标准的以太网就这么容易的薅了羊毛了.

![图片](assets/e4727cbaacb1.jpg)

这里贴个夏Core的图, 真正的Scale UP/Scale Out/FrontEnd 三网统一的理想方法

![图片](assets/03893f6e88e0.png)

最终Tesla就这么简单的做成了, 注意到承载TTP和TCP/IP融合的Spine-Leaf用了VXLAN和BGP-EVPN, 后面我会讲为什么.

![图片](assets/1fa955a1a339.png)

### 2.4 以太网如何实现内存语义

其实很多年前, 在思科的一系列路由器平台上都广泛的在使用memif的技术, 例如我三年前开源出来的zmemif把Golang的收发性能提升到了20Mpps/12Mpps

[《zMemif: go语言高性能网络库》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486848&idx=1&sn=6822302c918e0bc60eb3763973c40116&chksm=f9961d42cee19454e2fc9018cef2510cf4bd2c49dfc3155204476b5a1aa67e2a28d2725f9ca1&scene=21#wechat_redirect)

其实很简单的一个做法, 共享一块内存出来, 直接以内存语义写数据进去, 然后另外一个Core去Snooping就好, 软件的方案采用了DPDK配合Golang, 直接让golang 用户态内存读写就可以完成发包. 而当有NetDAM网卡的时候, 这些Snooping会全部下沉到网卡上, 而你会发现Tesla的做法是完全相同的.

![图片](assets/9a2ae5975aad.png)

Tesla的TTP-MAC微架构,其实就是一个处理器, 数据进入SRAM的Common buffer, 这个TX buffer在100G的NIC上为1MB, 大约可以缓冲80us, 基本上在2~3个RTT的规模. 然后整个处理器是一个4级流水线的架构, 就是一个简单的死循环,读取TX Buffer, 修改, 然后写入触发TX方向封装数据包传输. 当收到ACK后就free掉这段buffer.

所以到这里看懂夏Core的这段话了没?

可能出乎很多人意外 ：） 实际上Load/Store/Atomic如果做成异步DMA的方式，是可以做到无限的Outstanding，只要Memory Bandwidth大于IO Bandwidth，无需流控，可以无限Load Outstanding。

这个逻辑的本质，其实和Zartbot提出的NetDAM很类似，其实，只有获得了Memory的控制权，端侧的IO的能力才能发挥出蛮荒之力。看明白了NetDAM的话，再进一步，就是无限outstanding的Load/Store/Atomic DMA了。

同时,你会发现能这么玩的不只渣和Tesla 还有IBM大型机的Tellum 2 也是将DPU挂到内存总线上了, 它也是大型机的ScaleUP节点:)

![图片](assets/8edffdccceb2.jpg)

当然作为一个内存语义的总线, DMA也是需要的, 提供HBM2HBM得拷贝能力.  但是这里要强调一个问题, 以太网要支持内存语义, 面临拥塞控制和可靠性及以太网效率问题,那么一定需要支持`Semi-Lattice`的语义.

满足交换律:支持乱序提交. 这一点对于LD/ST语义天然支持

满足结合律:其实就是message pack的能力,例如Tesla单个以太网包1KB, NOC packet 64B. 也不需要去过分的改以太网报文头.

幂等: 对于LD/ST操作幂等是容易的, 但是这一条对于做Allreduce的时候, 加法是需要额外的幂等处理的.

渣比Tesla领先的还有Reduce offload能力, 而这样的Offload可以极大的降低对GPU的L2Cache干扰, 懂的人自然懂.所谓的在网计算收益在这里

![图片](assets/bcce774a8c8b.png)

### 2.5 拥塞控制

我一直嘲笑RDMA这十年瞎搞Lossless, 我以为我怼人已经很过分了. 没想到Tesla更猛... [《RDMA这十年的反思1：从协议演进的视角》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489240&idx=1&sn=53c7512d8551a44834bd405fd38b15dd&chksm=f996061acee18f0c26fb6d3f745dfa717a1f9b41a5f63de139e72acbc00968f4a197a16dd272&scene=21#wechat_redirect)其中的一个图要更新一下了

![图片](assets/386631713a5c.png)

Tesla 讲的很直白, PFC是一坨屎, Lossless也是.

![图片](assets/9c8362b59e79.jpg)

而TCP本身的`Window based CC, SACK, FastRetransmit`才是正确的路, 而不是瞎搞PFC/DCQCN什么的, 唯一的问题就是软件/Kernel交互的影响, 需要一个直接DMA到GPU的类似于TCP的协议栈.
![图片](assets/a9496566a2ae.jpg)

整个TTP事务如下所示:

![图片](assets/3d2323f4b052.jpg)

对于丢包和乱序直接NACK触发重传, 然后硬件直接Open/Close链路,建立连接时候直接两次握手

![图片](assets/4399fe520ac2.jpg)

然后拥塞控制而言, 只是简单的控制在传输时的Outstanding 窗口即可. 然后free掉TX buffer即可接收新的数据了.

### 2.7 协议格式

TTP的做法类似于Mellanox DCT, 最多并发支持512个link, 然后可以通过动态的LRU驱逐

![图片](assets/be66d4f4591a.jpg)

然后整个数据包采用了2层路由机制, 使用了EVPN+VXLAN.后面讲单独介绍.  然后内部定义了24bits的Src/Dest Node Addr和报文长度, 传输层定义了opcode, Virtual Channel, 然后还有一个Epoch版本号应该可以用来做一些Atomic/幂等操作时使用, Congestion/Tx/Rx按照我设计NetDAM时的经验, 应该是用于代表TX/RX buffer和outstanding数量的一个信号, 8bit可以表示1/256的精度.

### 2.8 Lossy支持

![图片](assets/4f6ce74944d3.jpg)

很明确类似于TCP的机制, 可以进行选择重传, 也可以进行快重传. 很巧妙的将滑动窗口和片上SRAM结合, ACK了就释放, 超时了就重传. 窗口机制也就是限制到不超过带宽即可. 关键的优势是: 不需要任何交换机做任何事情, ECN也不需要, PFC也不需要,INT也不需要.

![图片](assets/4fcf2cff31b7.jpg)

### 2.9 多路径

初一看TPPoE是一个基于MAC的二层协议, 那么它是如何实现多路径能力的呢?`EVPN+VXLAN`改VXLAN的Srcport就行了

![图片](assets/c3b693db2f86.jpg)

BGP-EVPN针对MAC提供了L2 ECMP的路由, 然后的事情就非常简单了....当然Traffic Engineering还是需要一点技巧的, 具体可以参考Ruta就行了

[《Ruta实战及协议详解》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485165&idx=1&sn=412fcb1dd46dd4ef4384a033b0827256&chksm=f996162fcee19f39ab4c995b1be2676779eb5b647ad26dd7017001b75530dfb9a59e9790b37d&scene=21#wechat_redirect)

### 2.10  传输效率

然后整个消息直接Over在以太网之上, IP层是可选的. 在单个交换机内是直接基于DstMAC转发, 在效率上有一个折中.

![图片](assets/dc4066fa89cd.jpg)

而跨越多层网络则是通过EVPN+VXLAN, 效率有一定的影响, 但是针对大多数场景报文size已经超过1KB了, overhead影响并不明显. 当然在多路径情况下的带宽利用率还是偏低的, 大概只有70%~80% 这一点是Tesla做的不好的地方.  而我们已经实现了97.5%的利用率.

![图片](assets/40fc0d7263ba.jpg)

### 2.11 拓扑组网

标准的Spine-Leaf, 没有多轨道, 然后ScaleUP/ScaleOut/Frontend完全融合的一张网了, 实现了三网合一非常干净的一套解决方案.然后整个部署规模来看,经过了相对较大Scale验证, 累积320Tbps的 All-Reduce I/O

### 2.12 TCP延迟大么?

很多人会片面的以为基于TCP的拥塞控制会导致大延迟, 实际呢?Tesla给了一个数据1.3us

![图片](assets/e6b72a70ea89.jpg)

这个数据和NetDAM也能对上,接近618ns * 2 ~ 1.3us.

![图片](assets/31987fd8aa91.png)

其实对于TCP, 取决于你怎么舔它, 把它舔硬的恰当好处...完全的TOE肯定不行.

![图片](assets/f43efad99141.png)

## 3. Cache一致性和LD/ST穿越交换网

### 3.1 Cache一致性

对于做计算的对于CC是有执念的, 但是我一直以来不认同在大规模组网上CC的价值. 最近有一篇关于Grace-Hopper的论文《Harnessing Integrated CPU-GPU System Memory for HPC:a first look into Grace Hopper》[3]

![图片](assets/793d9a8877a5.png)

![图片](assets/0ebbe2c8fa23.png)
业务收益上来看, Grace和Hopper之间的Cache一致性仅有在一些HPC业务上, 例如PDE计算中的W-Cycle时有价值, 其它情况下并没有显著的收益.

### 3.2  LD/ST穿越交换网

说实话要么选择自研交换机更极致的协议, 例如NVLink这样的. 要么就在以太网上使用Pack合包. 不要在中间去改一些不伦不类的协议,  而直接穿越要做到Lossy其实非常困难, 而Tesla TTPoE是一个更好的选择. 支持Lossy非常重要!!

## 4. 结语

其实我们对于加速器的互联解决方案已经越来越清晰了, 基于RDMA的ScaleUP方案是错误的, 它面临着众多限制, 对GPU架构也有约束, 甚至对调度也产生了影响. 当然我能理解当前很多基于RoCE互联的加速器厂家确实是没有更好的现成的技术而作出的选择, 但是这条路长期来看一定是有问题的, 最终也会走到内存语义上来.

对于内存语义如何在以太网上传输, 我从4年前就在给所有的人讲NetDAM是怎么搞的, 但是总有人认为我在夹带私货甚至是臆想. 而且看到Tesla在HotChip上作出同样的事情后, 居然魔怔了一样, 认为人家Tesla是在卖PPT... 真不知道第一性原理是什么?

对于拥塞控制, Tesla选择的类似于TCP的做法, 跟我们团队去年做的也惊人的一致, 当然我们做的比他还要优秀很多, 所以我还是在这个领域给各位出一个命题如下, 这件事情是能做成的. **而且这件事情做成了前端网络和ScaleOut融合就是非常轻松的一件事情了.**

采用Spine-Leaf拓扑, 不用任何框式交换机, 不需要DeepBuffer. 如何不利用交换机任何Hash函数信息, 不需要交换机任何特殊配置, 不启用ECN和PFC. 通过网卡算法自动打散流量,并维持交换网97.5%以上的利用率, 对于交换机的buffer需求为队列深度低于3us. 并能够针对128:1的时候incast时最大流和最小流量之间的带宽差异小于100Kbps, 同时针对任何网络线缆故障, 通信中断无感知, 模型训练收敛时间小于100ms.

参考资料

[1]
RDMA杂谈: https://www.zhihu.com/column/c_1231181516811390976
[2]
什么是RDMA？: https://wiki.h3c.com/cn/detail.html?WikiName=RDMA
[3]
Harnessing Integrated CPU-GPU System Memory for HPC:a first look into Grace Hopper: https://arxiv.org/pdf/2407.07850