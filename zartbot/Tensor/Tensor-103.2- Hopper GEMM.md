# Tensor-103.2: Hopper GEMM

> 作者: zartbot  
> 日期: 2025年10月22日 10:01  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496586&idx=1&sn=efa07f56b9a56dbdbb75a480556c7759&chksm=f995e348cee26a5e7984201b774eeb8f1cd56f0538eab50b9d6cc29f6df22239868141b26bd5#rd

---

### TL;DR

这是整个Tensor-103 Gemm系列文章的第二篇, 第一篇介绍了基于CuteDSL的basic gemm kernel实现, 这一篇首先分析了Basic Gemm的一些问题, 然后展开分析Hopper的硬件架构和软件协同如何来解决这些问题的. 最后再用CuteDSL的hopper Gemm例子进行详细的分解分析. 本文目录如下:

```
1. Hopper 软硬件功能演进概述
1.1 Basic Gemm的问题
1.1.1 cp.async
1.1.2 TensorCore
1.2 Hopper架构演进
1.2.1 TMA
1.2.2 CGA
1.2.3 TensorCore & WGMMA
1.2.4 Warp Specialization
2. Hopper新功能详解
2.1 CGA
2.1.1 基本操作
2.1.2 Grid/Cluster Layout
2.2 TMA
2.2.1 TMA架构分析
2.2.2 TMA指令和描述符
2.2.3 TMA地址计算及请求生成
2.2.4 TMA同步机制
2.2.5 TMA组播
2.2.6 TMA Reduce
2.2.7 CuteDSL TMA操作
2.3 TensorCore WGMMA
2.3.1 WGMMA编程概述
2.3.2 Swizzle
2.3.3 CuteDSL WGMMA
2.4 CuteDSL异步编程
2.4.1 PipelineAsync
2.4.2 PipelineTmaAsync
2.4.3 PipelineTmaStore
2.5 小结
3. Hopper DenseGemm
3.1 Host侧函数
3.2 Kernel函数
3.2.1 第一阶段: 初始化和坐标计算
3.2.2 第二阶段: 流水线设置与内存分区
3.2.3 第三阶段: Prologue
3.2.4 第四阶段: MainLoop
3.2.5 第五阶段: Epilogue
```

## 1. Hopper 软硬件功能演进概述

Hopper Gemm的介绍也参考cuteDSL Github的Example  Hopper_DenseGemm[1]

### 1.1 Basic Gemm的问题

#### 1.1.1 cp.async

对于cp.async, 虽然它bypass了L1Cache可以直接存放到SMEM, 并且支持完全异步的内存拷贝来掩盖拷贝延迟. 但是它还是具有几个问题:

cp_size最大只有16B, 拷贝大量数据需要CUDA Core发射大量LD/ST指令, 因此对MIO产生瓶颈.

边界检查需要构建大量的谓词张量(Predication Tensor)占用了寄存器资源, 同时也增加了代码复杂度.

地址计算也需要占用线程和寄存器资源, 浪费了算力.

#### 1.1.2 TensorCore

虽然前面的示例并没有使用TensorCore, 但是即便是使用TensorCore在Ampere的架构中, 仍然需要手工从SMEM中加载数据到RMEM, 增加了一次数据搬运并增加了计算的复杂度. 同时如果TensorCore计算规模进一步扩大, 这样将导致寄存器排布的压力变得非常大.

### 1.2 Hopper架构演进

在[《GPU架构演化史14: Hopper架构详解》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488380&idx=1&sn=bf83d9150f629adbd46016c0a1ba7062&scene=21#wechat_redirect)中有一些关于Hopper架构的分析. 但接下来我们结合GEMM的场景来展开谈一下这些设计.

#### 1.2.1 TMA

首先为了解决cp.async的问题, 我们能否专门做一个针对Tile的张量加速引擎. 只需要把Tile的Layout和源目的坐标构成一个描述符交给这个器件处理即可,TMA支持1D~5D张量, 可以定义特定的BLOCK进行异步的数据加载和存储

![图片](assets/839385eea590.png)

这样的做法可以释放对线程和寄存器的占用, 然后让它异步的进行数据拷贝和地址运算. 这样对于SM的MIO压力也降低了, 发射的指令数也大大降低, 这就是TMA的由来.

![图片](assets/aed0c4fabaef.png)

TMA具体的操作在[《Tensor-003 TensorCore架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491424&idx=1&sn=0fc2110931b27714900e78d73b11a5b5&scene=21#wechat_redirect)中也有一些介绍. 后面我们会在专门的一个小节中详细阐述.

#### 1.2.2 CGA

如下图所示, 在Ampere架构下, 同样的Tile在矩阵乘法计算时会被加载多次, 以两个2X2连续的Tile为例, 如果我们可以在GMEM到SMEM支持一定的组播能力, 则GMEM的内存带宽就可以数倍的节省.

![图片](assets/d8db001723f4.png)

因此我们能否在局部的一些SM之间构造一个SM-to-SM网络呢? 这就是在Hopper中添加一个Thread Block Cluster的架构(也被称为 Cooperative Grid Array, CGA), 将多个SM连在一起构成一个Cluster.

![图片](assets/5f2616a038ea.png)

这样就允许了Cluster内部, SM与SM之间的内存访问, 例如在计算sC1的SM可以将sA1共享给sC2的SM, 并且还支持在CGA内使用TMA异步拷贝内存到其它SM的SMEM中. 这样就构成了一个分布式SMEM架构, 也被称为DSMEM.

![图片](assets/a725b07ec03d.png)

另外TMA也针对CGA提供了组播能力, 例如拷贝时, 将sB1通过组播的方式, 从GMEM读取一份, 但同时放入sC1和sC3的两个SM内.

![图片](assets/6a0773227e9e.png)

#### 1.2.3 TensorCore & WGMMA

然后我们再来看TensorCore, 在Ampere中的TensorCore操作数依旧需要占用寄存器资源, 并且需要用户将SMEM中的数据搬运到RMEM中. 因此在Hopper中针对这个情况也进行了优化, Matrix A和 Matrix B都可以直接放置在SMEM让TensorCore读取, 这样就避免了寄存器资源的占用. 而针对结果的 Matrix D则依旧放入到RMEM中, 这是因为还有很多Epilogue的处理需要CUDA Core接力进行, 例如Attention的Softmax等. 然后在Blackwell上, 还增加了Tensor Memory进一步降低了寄存器的占用, 从而使得WMMA issue 有更低的代价, 具体的内容将在Blackwell那一节更新.

| Arch | Matrix A | Matrix B | Matrix D |
| --- | --- | --- | --- |
| Volta | RF | RF | RF |
| Ampere | RF | RF | RF |
| Hopper | RF/SMEM | SMEM | RF |
| Blackwell | TMEM/SMEM | SMEM | TMEM |

由于TensorCore的操作数在Hopper上也放置到了SMEM. 仅结果矩阵Matrix D占用寄存器资源, 因此在Hopper中可以将一个SM内的4个SubCore组在一起, 4个连续的Warp构成WarpGroup 配合TMA实现4个TensorCore并行的矩阵乘法. WarpGroup MMA还有一个优势可以降低对SMEM的带宽. 如下图所示:

![图片](assets/dd186ad25e8b.png)

我们可以预先加载A-Tile, 然后B-Tile可以在4个Warp内广播的方式执行, 这样B-Tile被TensorCore读取的带宽就降低了4倍, 整个TensorCore运算的效率也显著提高了. 当然它也有一些问题, 例如执行前需要wgmma.fence, 完成时也需要wgmma.wait_group等待, 它需要整个warp group去发射执行指令, 这也是Nvidia最终在Blackwell中引入TMEM的另一个原因, 另一方面它还继续括大了规模, 可以通过2CTA(2SM)执行, 在DSMEM上进行广播加载到TensorCore上. 同时每个warp可以由一个thread独立的提交,并通过SMEM的mbarrier通知完成.

![图片](assets/f032742dbec4.png)

关于Blackwell的TensorCore我们后面一章单独介绍.

#### 1.2.4 Warp Specialization

由于TMA的引入, 以及TensorCore的操作数在Hopper上也放置到了SMEM, 并且两者都提供了异步调用的能力, 因此我们可以把TMA和TensorCore作为SM的加速器使用. 回顾一下SM内的Warp Divergence和Warp调度. 对于一个SM一个周期内最多可以从4个不同的Warp中各发射一条指令, ALU的指令通常较短并且有固定的完成时间, 而TensorCore虽然有固定的完成时间, 但通常更长. 而访问内存的指令通常受缓存命中/缺失, 内存拥塞等影响时间很长也不可控.  GPU通过维持大量的Active Warp来实现延迟隐藏. 当一个Warp因长延迟操作(如访存)而停顿时, Warp调度器可以立即切换到另一个准备就绪的Warp来执行指令.

另一方面, 在Warp Divergence时, 每个Warp的SIMT特性会导致性能下降. 例如下图所示:

![图片](assets/1c71bba6832c.png)

相当于在timeline上存在了大量的空泡影响了效率.

Warp Specialization(WASP)是基于CUDA-DMA[2] 和 Singe编译器[3]的工作而流行起来的. 本质上是通过将Warp内的Divergence转移到Warp之间, 由于不同的Warp在相互独立的context下执行, 当Divergence发生在Warp之间时, 通过Warp Scheduler调度并不会引入额外的成本.

另一方面TensorCore和TMA这些加速器引入都有了异步调用的能力, 因此我们可以通过WASP,使用专门的Warp来发出TMA或Tensor Core矩阵乘法指令. TMA Warp发出复制指令并在数据准备好相乘时通知Tensor Core Warp, 而Tensor Core Warp在数据被消耗后通知TMA Warp, 告知内存已空闲, 可以用于更多复制. 如下图所示:

![图片](assets/f6775f0137f2.png)

然后我们来展开分析一下WASP的性能收益.  首先如上图所示, 对于TMA这些内存拷贝受到Cache Miss或者内存带宽的影响是动态的, 如上图所示, 通过Warp Specialization可以通过Warp Scheduler调度将这些动态延迟隐藏掉, 因此可以最大化TensorCore的利用率.

另一方面TMA Warp由于不再需要计算地址和谓词张量做边界检查, 也不需要占用线程资源, 因此这个Warp的寄存器资源占用量小了很多, 因此可以通过`setmaxnreg`将寄存器资源分配给consumer的MMA warp, 在cuteDSL中可以使用cute.arch.warpgroup_reg_alloc(self.num_regs_mma)修改

![图片](assets/6c85e104ea33.png)

但是需要注意的是, 使用WASP对于程序员如何精巧的编排整个流水线和处理同步的数据依赖也带来了一些挑战. 例如Hopper的FlashAttn, SM在发出TMA和TensorCore GEMM操作的同时, 还需要执行大量的CUDA Core相关的工作, 例如softmax的计算等...

![图片](assets/db9c18d878b2.png)

最后, 对于warp specialization还有一个收益是来自于不同Warp的指令数减少了, 这样在Warp执行时, I-Cache的开销也会减少, 否则在一些复杂的Attn计算过程中需要更大量的I-Cache甚至I-Cache Miss导致性能损失.

关于WASP有几篇不错的文档可以参考Unweaving Warp Specialization[4] , 15-779 Lecture 6:Advanced CUDA Programming:Warp Specialization[5],GPGPU Arch(二) —— 漫谈 Hopper WarpSpecialization Pingpong/Cooperative 设计[6],WASP: Exploiting GPU Pipeline Parallelism with Hardware-Accelerated Automatic Warp Specialization[7]

## 2. Hopper新功能详解

### 2.1 CGA

由于在Hopper GPU内部增加了局部的SM-to-SM的网络, 可以在SM之间共享SMEM构成Distributed SMEM.

![图片](assets/f3a03a2bfda9.png)

在软件层面增加了一个`cluster`的层次化结构, 即 Thread Block(CTA,Block)<--Thread Block Cluster(CGA,Cluster)<---Device(Grid)

![图片](assets/c09c553f3cf1.png)

#### 2.1.1 基本操作

一段使用CGA并获取相应cluster-index的测试代码如下所示:

```
import torchimport cutlassimport cutlass.cute as cuteimport cuda.bindings.driver as cuda@cute.jitdef cluster_demo(    stream: cuda.CUstream):        num_threads = 2    cluster_kernel().launch(grid = (4,6,2),        cluster = (2,2,1),        block = (num_threads,1,1),        stream = stream) @cute.kerneldef cluster_kernel():    tidx, _, _ = cute.arch.thread_idx()    bidx, bidy, bidz = cute.arch.block_idx()    cidx, cidy, _ = cute.arch.cluster_idx()    cdimx, cdimy, _ = cute.arch.cluster_dim()    cluster_id = cidx + cdimx * cidy    cute.printf("tid {} block-id {},{},{} cluster-id {},{} id {}",                 tidx, bidx,bidy,bidz,cidx,cidy,cluster_id)    return    torch_stream = torch.cuda.Stream()stream = cuda.CUstream(torch_stream.cuda_stream)cluster_demo(stream)
```

具体的cluster layout如下图所示:

![图片](assets/a479e529b582.png)

#### 2.1.2 Grid/Cluster Layout

对于在Hopper平台上执行基于TensorCore的WGMMA指令支持的Shape如下:

![图片](assets/fe94f6207775.png)

以BF16为例, DenseGemm K=16, 而这里将约束我们选择 Tile_MN 即 bM 和 bN 的大小. 对于一个WarpGroup而言使用了4个warp即128个线程, 如果我们的 bM x bN 为 128x256 时, 每个线程就需要 256 个寄存器, 而在Hopper上, 单个线程寄存器最大数量为255个,  当一个线程寄存器数量达到上限时, 会产生Register spill即某些不需要使用的Reg将会暂存到Local Memory, 这样会显著降低Kernl的性能.  另一方面WGMMA由于Accum矩阵也在RMEM中, 因此需要所有的线程都参与, 如果Register spill发生在TensorCore和这些Accum矩阵使用的寄存器上, 将会极大的影响TensorCore的运算效率.

因此通常的做法是增加线程的梳理, 让两个Warp协同来执行WGMMA运算, 即加载Tile后, 进一步拆分成2个64 x 256的子块进行运算. 这样就保证在整体寄存器使用量不变的情况下, 每个线程的寄存器数量减半. 我们可以通过定义`atom_layout_mnk`来处理这样的进一步的拆分.

```
from typing import Tuple, Typeimport mathimport torchimport cuda.bindings.driver as cudaimport cutlassimport cutlass.cute as cuteimport cutlass.utils as utilsimport cutlass.utils.hopper_helpers as sm90_utilsfrom cutlass.cute.runtime import from_dlpack@cute.jitdef cluster_layout(    a : cute.Tensor,    b : cute.Tensor,    c : cute.Tensor,):        tile_shape_mn = (128,128) # choice: [(128, 128), (128, 256), (128, 64), (64, 64)]    tile_shape_mnk = (*tile_shape_mn, 1) #K-dim稍后更新        cluster_shape_mn =(2,1) # choice: [(1, 1), (2, 1), (1, 2), (2, 2)]    # 对于较大的tile, 由于单个WarpGroup运算会导致RegisterSpill,     # 因此需要两个warp group进行运算    atom_layout_mnk = (            (2, 1, 1)            if tile_shape_mnk[0] > 64 and tile_shape_mnk[1] > 128            else (1, 1, 1)        )    mma_warp_groups = math.prod(atom_layout_mnk)    num_threads_per_warp_group = 128    threads_per_cta = mma_warp_groups * num_threads_per_warp_group    
```

由于WGMMA指令对Tile计算的规模有特定的形状约束, 因此我们需要进一步计算K-dim大小来决定最后的拆分和Layout. 首先我们获取 三个张量的datatype和Majorness. 然后以M=64, N=bN来构造Tiled_MMA, 通过Tiled_MMA的shape中的k-dim, 下图参考自《Targeting NVIDIA Hopper in MLIR》[8]

![图片](assets/2a2b06aa8ce2.png)

```
    # 获取Tensor的数据类型和Majorness    a_dtype = a.element_type    b_dtype = b.element_type    acc_dtype = c.element_type    a_layout = utils.LayoutEnum.from_tensor(a)    b_layout = utils.LayoutEnum.from_tensor(b)    c_layout = utils.LayoutEnum.from_tensor(c)    # 构造Tiled_MMA    tiled_mma = sm90_utils.make_trivial_tiled_mma(        a_dtype,        b_dtype,        a_layout.sm90_mma_major_mode(),        b_layout.sm90_mma_major_mode(),        acc_dtype,        atom_layout_mnk,            tiler_mn=(64, tile_shape_mnk[1]),    )        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])    mma_inst_tile_k = 4    tile_shape_mnk = (        tile_shape_mnk[0],        tile_shape_mnk[1],        mma_inst_shape_k * mma_inst_tile_k,    )    cute.printf("tile-mma {} tile_shape_mnk{}",tiled_mma.shape_mnk,tile_shape_mnk)#output    tile-mma (64,128,16) tile_shape_mnk(128,128,64)
```

此时已经完成了Tile_MNK的计算. 最后根据  Tensor的Shape计算需要的grid. 例如M=N=K=4096, Batch的维度 L=16 时, C_Tiler=(128,128), 因此gC的Layout为(128,128),(32,32,16). 然后根据cluster_shape计算出grid的维度

```
    c_tiler = (tile_shape_mnk[0], tile_shape_mnk[1])  #c_shape = tile_M,tile_N    gc = cute.zipped_divide(c,tiler=c_tiler)    cluster_shape_mnl = (*cluster_shape_mn, 1) #cluster dimz = 1    clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnl)    grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnl))    cute.printf("gC Layout {} grid {}",gc.layout, grid)#outputgC Layout ((128,128),(32,32,16)):((4096,1),(524288,128,16777216)) grid (32,32,16)
```

### 2.2 TMA

#### 2.2.1 TMA架构分析

TMA的实现细节可以在专利 US20230289292A1[9] 中找到.

实现TMA的最根本原因是: 随着GPU中用于矩阵运算的专用单元(如Tensor Core)性能大幅提升, 数据供给成为了新的瓶颈. 传统的存储器访问方式, 即由处理器核心(CUDA Core)执行加载/存储指令, 涉及复杂的地址计算, 占用宝贵的寄存器资源, 并且在等待数据从主存(Global Memory)传输到片上共享存储器(Shared Memory)的过程中, 计算单元可能处于空闲状态, 从而限制了整体性能和能效. 即使是像`LDGSTS`这样的异步拷贝指令, 仍然需要软件(即运行在处理器核心上的程序)来计算每个数据块的地址, 带来了显著的软件开销和性能损耗.
![图片](assets/285b0fdc7303.png)

为了解决这个问题, Nvidia设计了一个专用的存储器访问硬件电路, 即张量存储器访问单元(Tensor Memory Access Unit, TMAU). 其核心思想是:

**卸载:** 将复杂且耗时的多维数据地址计算和数据移动控制逻辑从通用处理核心(如SM中的CUDA Core)中卸载到专门的硬件单元TMAU中.

**异步执行:** TMAU可以独立于处理器核心异步地执行大规模数据块的传输任务. 处理器核心只需发送一个高级别的请求, 之后就可以继续执行其他计算任务, 从而有效地隐藏了存储器访问延迟.

**抽象与简化:** TMAU能够理解张量这样的多维(1D~5D)数据结构的逻辑布局. 程序员不再需要手动计算复杂的物理内存地址, 而是可以通过更高维度的坐标来请求数据, 极大地简化了编程模型, 降低了开发和调试成本.

**边界检查:** 在原有的数据加载过程中, 通常需要CUDA Core构建谓词张量判断访问内存边界, 而在TMAU中内置了边界检查的能力. 这样也极大的减轻了寄存器的压力.

在体系结构上, 每个SM都会紧密耦合一个TMAU, 这种一对一的配置减少了访问延迟和资源争用.

![图片](assets/69e46866c2ba.png)

如下图所示, TMAU能够访问不同层级的存储器, 包括:

Global Memory等外部存储器, 如HBM/GDDR DRAM, 甚至通过PCIe访问Host Memory

SM内部的共享存储器(Shared Memory, SMEM)

通过内部互连网络实现分布式共享存储器(Distributed Shared Memory, DSMEM)的访问

![图片](assets/f7d14a1dc520.png)

TMAU内部结构如下:

![图片](assets/982c51f1e377.png)

| 编号 | 组件名称 | 功能描述 |
| --- | --- | --- |
| 604 | 存储器输入/输出控制器 | 作为SM和TMAU之间的接口, 接收来自SM的存储器访问请求. |
| 606 | 内部请求队列 | 缓存来自SM的请求. 它可以处理两种类型的请求: 张量请求 (需要描述符) 和 非张量请求(线性数据块). |
| 608 | 描述符高速缓存 | 缓存最近使用的张量描述符. 由于对同一张量的访问通常具有时间局部性, 该缓存能显著降低获取描述符的延迟. 如果缓存未命中, 会向全局存储器预取描述符. |
| 610 | Setup Block | 从请求队列中取出请求, 如果是张量请求, 则从描述符缓存中获取描述符. 它负责解析所有参数(来自描述符和请求本身), 进行正确性检查, 并为后续的请求生成器准备好所有必要的计算参数. |
| 616 | 请求生成器 | 这是TMAU的核心引擎. 它接收Setup Block准备好的参数, 遍历多维张量空间(或线性地址空间), 迭代计算每个子块的全局存储器地址和共享存储器地址, 检查越界条件, 并生成发送到存储器子系统的底层请求. |
| 618 | 响应完成跟踪电路 | 跟踪已发出的每个子请求的状态, 实现了TMAU与SM的异步操作. 当所有子请求都完成时, 它负责触发同步机制. |
| 614 | 通用网络接口控制器 | 负责与GPU内部的存储器互连网络进行通信, 发送请求并接收响应. |
| 620 | GNIC响应处理器 | 负责与GPU内部的存储器互连网络进行通信, 对接收响应进行处理. |

注意: 对于非张量请求, 由于不需要处理描述符,可以直接从606请求队列跳转到616请求生成器

具体工作流程:

**SM发起请求**: SM上的一个线程向其耦合的TMAU发送一个请求, 对于张量请求还包含了对张量描述符的指针和块坐标的信息.

**TMAU生成子请求**: TMAU接收到请求后, 开始异步的工作, 它会解析请求, 并根据数据块的大小和存储器系统的限制(例如L2Cache Line大小),自动地计算出一系列子块的物理地址, 并生成多个底层的存储器访问请求发送给存储器子系统.

**数据传输**: 存储器子系统响应这些子请求, 将数据传输到目标位置. TMAU可以将数据直接写入SM的SMEM, 并绕过了寄存器文件(RF)和L1数据缓存, 避免了资源占用和不必要的缓存污染.

**同步**: TMAU内部有完成跟踪电路. 当所有子请求都完成, 整个数据块被成功传输后, TMAU通过一个同步机制(例如, 更新共享存储器中的一个计数器或屏障(barrier))来通知SM. SM上的线程可以通过检查这个同步对象来确定数据是否准备就绪.

#### 2.2.2 TMA指令和描述符

对于SM issue一条TMA指令, 我们通常希望其指令尽量精简. 但是在对基于张量的TMA时, 通常需要携带张量维度(dimensions), 每个维度的大小(size), 元素大小(element size), 每个维度的步幅(stride)等大量信息, 因此TMA的设计是构建一个描述符(Descriptor)放在GMEM中, 然后仅在TMA指令中携带它的指针. 描述符如下图所示:

![图片](assets/f6a4b9a4c45a.png)

**张量描述符 (Tensor Descriptor):** 这是一个存储在GMEM中的数据结构, 定义了一个张量的静态属性. SM在发起请求时只需提供该描述符的指针.

`参数示例`: 张量维度(dimensions), 每个维度的大小(size), 元素大小(element size), 每个维度的步幅(stride).

**访问描述符 (Access Descriptor):** 通常也包含在张量描述符中, 定义了访问模式的属性.

`参数示例`:  访问块的大小(block/box size), 越界填充值(out-of-bounds value).

**TMAU指令参数 (Instruction Parameters):** 这是由SM在发起请求时直接提供的, 用于指定本次具体要访问哪一个数据块.

`参数示例`:  块的起始**坐标** (e.g., (x,y,z)), 目标SMEM地址, 同步对象地址.

由于TMA中有Descriptor Cache, 因此在CuteDSL中也可以通过如下方式从GMEM中Prefetch Descriptor

```
cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
```

#### 2.2.3 TMA地址计算及请求生成

对于`非张量模式`的地址生成非常简单, SM的请求指令会直接提供:

源地址 (GMEM中的起始地址)

目标地址 (SMEM中的起始地址)

要传输的数据总量

TMA的地址生成逻辑此时退化为一个简单的线性地址递增器. 对于第 i 个数据块(通常是16字节), 地址计算为:

$$Addr_{global} = SourceAddress + i \times BlockSize$$

对于`张量模式`则需要根据张量描述符生成. 首先在Setup Block中分析获取必要的信息:

从**张量描述符**中获取:

张量的基地址 (Base Address).

张量的维度 (Dimensions).

每个维度的总大小 (Tensor Size).

每个维度的步幅 (Tensor Stride).

元素大小 (Element Size).

从**访问描述符** (通常也在张量描述符内) 中获取:

请求的**块大小 (Box Size)**, 即在每个维度上要加载多少个元素.

遍历步幅 (Traversal Stride).

从**SM发来的指令**中获取:

请求块的起始坐标. 这是逻辑位置, 例如 `(x, y, z)`.

然后在请求生成器内部实现了一个硬件状态机, 行为等同于一个N维嵌套循环 (N是张量维度)

$$Addr_{global} = BaseAddr + \sum_{i=0}^{N-1} (coord_i \times stride_i)$$

在内部循环的每次迭代中, TMA都会检查当前计算的逻辑坐标 `coord_i` 是否超出了张量描述符中定义的 `TensorSize`, 即每个维度的大小 (tensorSize[0], tensorSize[1], ...). 如果超出, TMA不会去访问这个无效地址, 而是会标记这个元素为越界, 并在后续写入目标地址时使用预设的填充值 (0或者特殊的NaN).

最后对于生成的地址, TMA不会为每个元素都生成一个独立的访存请求. 它会智能地将地址连续的多个元素合并成一个对存储器子系统(如L2缓存)的请求. 请求的大小通常是L2缓存行的大小(例如128B). 这一步骤最大化了存储器总线的利用率.

最后为了避免访问内存的Bank冲突, 需要采用swizzle的方式生成地址. TMA在将数据写入共享存储器时, 自动执行Swizzle. 这个过程对程序员是半透明的(程序员需要知道数据被Swizzle了, 但不需要自己实现Swizzle).

首先根据基地址和LogicalOffset计算出逻辑地址

$$Addr_{smem\_logical} = BaseAddr_{smem} + LogicalOffset$$

然后应用Swizzle函数来生成物理地址

$$Addr_{smem\_physical} = Swizzle(Addr_{smem\_logical})$$

这个Swizzle函数通常是基于**地址位的异或(XOR)操作**. 硬件实现起来非常快速和廉价.

Example: 一个简化的64KB共享存储器, 32个Bank, 每个Bank是4字节宽
一个地址可以唯一地标识一个4Byte的字.

一个地址的低5位 (address[4:0]) 通常用于决定它属于哪个Bank.

一个地址的高位 (address[...:5]) 用于决定它在那个Bank中的哪一行.
不使用Swizzle的情况:
如果一个Warp的32个线程访问的地址分别是 0, 32, 64, 96, ..., 它们的地址二进制表示的低5位都是`00000`. 这意味着所有32个线程都访问`Bank 0`, 造成了严重的Bank Conflict.
使用Swizzle的情况:
TMA可以应用一个Swizzle函数, 例如:

$$\text{physical\_bank\_id} = \text{logical\_bank\_id}\quad XOR\quad \text{logical\_row\_id}$$

更具体地, 在硬件层面可能是:

$$Addr_{physical}[4:0] = Addr_{logical}[4:0] \oplus Addr_{logical}[9:5]$$

这里, 我们用逻辑地址的高位(行信息)去"扰乱"低位(Bank信息). 现在, 当线程访问逻辑地址0, 32, 64, 96, ...时:

逻辑地址  *0*:  Addr_logical 是 *...00000 00000*. 物理Bank ID是 *00000 XOR 00000 = 0*.

逻辑地址 *32*:  Addr_logical 是 *...00001 00000*. 物理Bank ID是 *00000 XOR 00001 = 1*.

逻辑地址 *64*:  Addr_logical 是 *...00010 00000*. 物理Bank ID是 *00000 XOR 00010 = 2*.

...

原本访问同一个Bank的地址序列, 经过Swizzle后, 被完美地分散到了Bank 0, 1, 2, ... , 31上, 从而消除了Bank Conflicts.

#### 2.2.4 TMA同步机制

在Ampere上, cp.async通常由一个线程提交, 然后这个thread自己commit_group和wait_group. 因此只需要对线程可见就行. 因此最直接的方法就是在scoreboard上做一个barrier即可, 或者在SMEM上做一个spin lock或者信号量机制.通常在prologue阶段批量提交一批cp.async访问, 然后`cp.async.commit_group`, 然后通过`cp.async.wait_group N`等待操作完成, 如下图所示:

![图片](assets/9a17c639e802.png)

而在TMA操作中, 通常是一个线程提交TMA指令, 其它线程也需要得知结果, 甚至在基于DSMEM SMEM->SMEM或者GMEM->multicast的场景下, 还需要做到跨CTA的同步机制. 因此在Hopper中引入了memory barrier的同步机制, 针对不同方向的TMA内存拷贝, 可以支持两种不同的完成机制:

![图片](assets/8237c104ee49.png)

对于SMEM->GMEM, 通常也是一个线程自己issue TMA指令, 自己commit和wait即可. 下面我们以warp specialization的TMA Producer和WGMMA Consumer来展开介绍一下mbarrier机制. 在B站上有一个视频《GPU 计算与编程模型演进：异步计算编程中的吞吐与延迟平衡》[10], 在Hopper上需要两组信号量去表达:

![图片](assets/89952811eca6.png)

当Producer TMA放一些数据进如SMEM,需要通过一个`smem_full` Mbarrier告诉Consumer数据已经准备就绪. 同样当Consumer完成计算后需要一个信号量`sem_empty`去通知Producer Refill数据. 在Hopper开始, Mbarrier结构如下:

![图片](assets/07b06c60b602.png)

它内部包含了一个线程数相关的计数器和传输多少Byte的计数器, 包含期望值和当前完成的计数. 然后会有一个Phase bit通过翻转来表示完成状态. 首先如上图所示, 我们需要对Mbarrier进行初始化, 因为TMA是一个线程在issue指令, 因此Expect Arr_Cnt=1, 而其它值都为0.

稍微展开一下, 其实这里的这个数据结构就对应了Hopper上的对外所讲的Async Transaction Barrier

![图片](assets/b176b3e0b64f.png)

图上左侧的Threads cnt就对应于Mbarrier结构体中橙色的部分, 而右边的Transaction cnt就对应于Mbarrier中蓝色的部分.

然后TMA发起了一条请求, TMA会把这条指令挂载到这个Barrier上, 并告知需要传输16KB数据, 接下来进行一个mbarrier_arrive_expect操作, 此时Actual Arrv_Cnt变成1, 同时更新barrier上的Expect TransBytes.

![图片](assets/435dd6aa214d.png)

然后随着数据不断被传输, TMA的Req Completion Tracking模块会根据GNIC返回的Write ACK更新Actual Trans_Bytes. 此时, 由于Expect和Actual Trans_Bytes还不一致, 因此Mbarrier Phase bit还为0, Consumer则继续阻塞等到Phase bit翻转.

![图片](assets/7a7d56f18df7.png)

等到16KB数据传输完成后, 如下图所示, 此时硬件会根据这个状态翻转Phase bit, 翻转该bit是一个原子操作.

![图片](assets/e8fb179435d5.png)

翻转完成后, Consumer阻塞解除, 开始执行WGMMA等计算的操作.

![图片](assets/f3a851d31619.png)

然后Consumer就开始消费数据, 对于前面所讲的那个Barrier代表着数据已经满了, 可以供Consumer消费了, 因此也被称为`smem_full` Mbarrier. 此时我们需要另一个Mbarrier, 语义上代表这段数据已经消费完了并空闲了, 即`smem_empty` Mbarrier, 此时Producer根据这个empty barrier中的phase阻塞操作. 需要注意的是, 由于WGMMA是由WarpGroup中的128个线程参与计算的 ,因此init_mbarrier(&bar_empty,128), 即在expect arrv_cnt中为128代表这128个线程.

![图片](assets/d69d589d8294.png)

然后consumer等待TensorCore完成计算后, 使用mbarrier_arrive(&bar_empty)更新`smem_empty`中的actual arrv_cnt, 此时Phase依旧为0, Producer仍然处于阻塞状态.

![图片](assets/f900539f7c11.png)

当expect和actual数据一致后, 下一个cycle硬件又会通过原子操作翻转`smem_phase` bit, Producer解除阻塞继续执行数据传输.

![图片](assets/8e9f3563402c.png)

然后接下来进行第二轮迭代时, 由于MBarrier中的Phase=1, 因此在这个时候Producer/Consumer需要记录下一次更新时phase bit翻转是从1到0.

![图片](assets/b57e846741f4.png)

正如1.2.4介绍Warp Specialization, 通过这样两个信号量的方式, 硬件可以去做warp的调度. 比起原来的multi-stages, Producer和Consumer的代码完全解耦后, 通过两个信号量的配合以及异步执行,可以看到整个过程中一个被阻塞, 另一个可以执行 因此TensorCore的Warp可以完全打满, 并不需要在中间插入一些等待TMA操作的代码影响TensorCore的利用率, 同时还可以通过setmaxregn,让Consumer有更多的寄存器资源,避免register spill, 最后这样的分离操作, 使得SM中I-Cache的占用也减小了, 避免了复杂的大规模程序带来的I-Cache miss影响.

![图片](assets/afc2b7f0a4dc.png)

当然这些Phase翻转和cnt/bytes的操作非常复杂, 在Cutlass中进行了一些`pipeline`封装,通过`advance`方法来构建多个stage的循环缓冲区流水线的使用.

![图片](assets/0805935bba6b.png)

同时也针对Producer TMA Warp和Consumer TC WarpGroup封装了一层更加方便使用的API

![图片](assets/9ade4b5f71f3.png)

后面几个小节, 我们将在CuteDSL上再来详细分析这个过程.

#### 2.2.5 TMA 组播

由于在Hopper中添加了SM-to-SM的网络, 构建出了DSMEM, 那么在矩阵运算的时候, 我们就可以只从GMEM拷贝一次到SMEM,然后通过SM-to-SM从一个SM的SMEM拷贝到另一个, 这样就能节省GMEM的带宽, 如下图所示:

![图片](assets/950d15f1bf7d.png)

因此TMA也添加了multicast的能力, 首先从一个SM发出TMA multicast指令, 数据到达写入和barrier更新

![图片](assets/6b0c8242560d.png)

此时, 我们可以在TMA Unit上做一些扩展, 因为操作的指令带有Multicast, 因此可以再触发另一个SMEM-to-SMEM的ST操作, 写入到另一个SM

![图片](assets/1d743c631d9a.png)

#### 2.2.6 TMA Reduce

Reduce sum是Split-K GEMM中必不可少的操作, 而reduce min/max通常也在注意力机制中被使用. 一个普通的reduce操作, 需要将CTA SMEM中的值累积到GMEM张量中的一个Tile, 这将包含一次GMEM读取, 将原始数据加载到CTA的SMEM或者RMEM, 执行xor | and | or | add | min | max | inc | dec操作, 然后再一次写入GMEM.

因此TMA增加了Reduce的支持, 并且仅支持SMEM->GMEM的reduce操作, 但是从公开的专利看, TMAU并没有向量计算单元用于处理reduce的计算操作.同时Nvidia也没有完全公开TMA Reduce的实现, 因此做一些猜测:

为了降低GMEM数据两次穿越NOC到SM的开销, 那么这个reduce操作的向量器件应该在SM的外部, 而不是复用CUDA Core的ALU. 然后如果多个SM需要一起做reduction的时候, 还有多个SM之间协同和计算一致性的问题需要处理, 这会变得非常复杂.因此在TMA Reduce实现时, 做了一些约束, 它的源地址仅支持CTA-Level的SMEM, 目的地址仅支持GMEM.

```
cp.reduce.async.bulk.dst.src.completion_mechanism{.level::cache_hint}.redOp.type               [dstMem], [srcMem], size{, cache-policy}.dst =                  { .global      }.src =                  { .shared::cta }.completion_mechanism = { .bulk_group }.level::cache_hint    = { .L2::cache_hint }.redOp=                 { .and, .or, .xor,                          .add, .inc, .dec,                          .min, .max }.type =                 { .f16, .bf16, .b32, .u32, .s32, .b64, .u64, .s64, .f32, .f64 }
```

然后每个被TMA reduce调用的reduction操作是独立的,并使用relaxed-order处理. 但是我们还需要考虑这些操作设计成原子化(atomic)的操作, 然后操作时地址对齐到16B. 因此猜测在L2Cache连接到XBAR的port上增加了一些Reduction Logic. 这些操作并不会需要很大的算力, 并且由于TMA本身是异步操作, 加上还有其它的SM访问内存, 因此也不需要Reduction Logic达到L2的带宽. 例如我们需要2TB/s的reduction带宽, 处理FP32 Reduction计算时 ,算力需求仅需要500Gflops.

#### 2.2.7 CuteDSL TMA操作

对于CuteDSL中的TMA操作定义如下所示:

![图片](assets/20aac6ab84f0.png)

我们以下面一个简单的例子来熟悉CuteDSL的TMA操作. 大概流程就是先通过zipped_devide 获取gmem_tensor, 然后生成smem layout. 再根据它们创建tma_atom. 在SMEM内存管理上 使用了cuteDSL的结构体功能, 将Mbarrier和SMEM上tensor存储空间在一起构成结构体.

```
from typing import Tuple, Typeimport torchimport cuda.bindings.driver as cudaimport cutlassimport cutlass.cute as cuteimport cutlass.utils as utilsimport cutlass.utils.hopper_helpers as sm90_utilsfrom cutlass.cute.runtime import from_dlpack@cute.jitdef tma_example(    A : cute.Tensor,):        tile_shape_mn = (32,32)        # 使用zipped_divide 进行分块, 并计算需要的block layout    gA = cute.zipped_divide(A, tiler=tile_shape_mn)    print(f"zipped_divide A Layout {gA}")    grid_dim = (*gA.shape[1],1)        a_dtype = A.element_type    a_layout = utils.LayoutEnum.from_tensor(A)        # make_smem_layout_atom函数会根据数据类型和major维度的size 进行swizzle.    # 具体的Swizzle操作在后面一节详细阐述.    a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(        kind=sm90_utils.get_smem_layout_atom(            a_layout,            a_dtype,            major_mode_size=tile_shape_mn[0],        ),        element_type=a_dtype    )        # 根据目标的tile_shape及layout atom生成smem_layout    a_smem_layout = cute.tile_to_shape(        atom = a_smem_layout_atom,        trg_shape=tile_shape_mn,        order=(0,1)    )    cute.printf("atom layout {}\n smem layout {}",a_smem_layout_atom,a_smem_layout)    # atom layout S<2,4,3> o 0 o (8,32):(32,1)     # smem layout S<2,4,3> o 0 o ((8,4),(32,1)):((32,256),(1,0))            tma_copy_size_a = cute.size_in_bytes(        a_dtype, cute.select(a_smem_layout, mode=[0,1])    )    cute.printf("tma cp size {}",tma_copy_size_a)    # tma cp size 2048    # 构建结构体, 包含mbar指针和smem中的数据    buffer_align_bytes = 128    @cute.struct    class SharedStorage:        mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]         sA:  cute.struct.Align[            cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout)],            buffer_align_bytes        ]        smem_size = SharedStorage.size_in_bytes()        # 生成tma_atom和tma_tensor    a_tma_atom, a_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),  #使用TMA        gmem_tensor = A,        smem_layout= cute.select(a_smem_layout, mode=[0,1]),        cta_tiler = tile_shape_mn     )    cute.printf("TMA ATOM ThrID  {} \nTV-Layout\n Src {}\n Dst {}\n TMA Tensor {}",                a_tma_atom.thr_id,                a_tma_atom.layout_src_tv,                a_tma_atom.layout_dst_tv,                a_tma_tensor    )    # TV-Layout    #  Src (1,1024):(0,1)    #  Dst (1,1024):(0,1)    #  TMA Tensor (0,0) o (4096,4096):(1@1,1@0)    kernel(        a_tma_atom,        a_tma_tensor,        a_smem_layout,        tma_copy_size_a,        cute.make_layout(tile_shape_mn),        SharedStorage,    ).launch(        grid = grid_dim,        block = (256,1,1),        smem = smem_size    )@cute.kerneldef kernel(    tma_atom: cute.CopyAtom,    tma_tensor : cute.Tensor,    smem_layout: cute.ComposedLayout,    tma_copy_size_a : int,        cta_tiler: cute.Layout,    SharedStorage: cutlass.Constexpr):    bidx, bidy, _ = cute.arch.block_idx()    tidx, _, _ = cute.arch.thread_idx()    warp_idx = cute.arch.warp_idx()        # 第一个warp group作为producer    is_producer = warp_idx < 4    # 尽早的预取TMA descriptor    if warp_idx == 0:        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)    cute.arch.sync_threads()    # 分配SMEM, 并从结构体中获取MBarrier指针        smem = cutlass.utils.SmemAllocator()    storage = smem.allocate(SharedStorage)    mbar_ptr= storage.mbar_ptr.data_ptr()        # 从结构体中获取Tensor    sA = storage.sA.get_tensor(smem_layout.outer, swizzle=smem_layout.inner)    # 初始化MBarrier    with cute.arch.elect_one():        cute.arch.mbarrier_init(mbar_ptr, cnt=1)    cute.arch.mbarrier_init_fence()    #######################################    #  TMA Producer    #######################################    if is_producer :        warp_idx_in_wg = cute.arch.warp_idx() % 4        if warp_idx_in_wg == 0:                          # 根据block的坐标获取Local Tile            tiled_tma_A = cute.local_tile(                tma_tensor,                        tiler = cta_tiler.shape,                coord = (bidx , bidy)            )                        # 用于warpgroup            sA_grouped = cute.group_modes(sA, 0, 2)            tiled_tma_A_grouped = cute.group_modes(tiled_tma_A, 0, 2)                        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(                    atom=tma_atom,                    cta_coord=0, # 用于CGA, 因为没有启用CGA, 因此设置为0                    cta_layout=cute.make_layout(1), # 没有使用CGA, 因此Layout为(1)                    smem_tensor=sA_grouped,                    gmem_tensor=tiled_tma_A_grouped,            )                        if bidx == 0 and bidy == 0 and tidx == 0:                cute.printf("gA: {}", tiled_tma_A.layout)                cute.printf("gA_grouped: {}", tiled_tma_A_grouped.layout)                cute.printf("sA: {}", sA.layout)                cute.printf("sA_grouped: {}", sA_grouped.layout)                cute.printf("tAgA: {}", tAgA.layout)                cute.printf("tAsA: {}", tAsA.layout)                            # gA: (32,32):(1@1,1@0)            # gA_grouped: ((32,32)):((1@1,1@0))            # sA: ((8,4),(32,1)):((32,256),(1,0))            # sA_grouped: (((8,4),(32,1))):(((32,256),(1,0)))            # tAgA: (((32,32),1)):(((1@0,1@1),0))            # tAsA: ((1024,1)):((1,0))            # 执行Copy            cute.copy(tma_atom, tAgA, tAsA, tma_bar_ptr=mbar_ptr)                        # 更新TMA smem_full barrier expect tx-bytes            with cute.arch.elect_one():                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_copy_size_a)                           if bidx == 0 and bidy == 0 and tidx == 0:                cute.printf("PRODUCER: TMA copy issued.")    #######################################    #  Consumer    #######################################        else:        # 等待smem_full barrier        cute.arch.mbarrier_wait(mbar_ptr, 0)        if tidx == 128 and bidx == 0 and bidy == 0 :            cute.printf("CONSUMER: TMA load finished.")            cute.printf("Tile in SMEM {}",sA)    returnM,N = 4096, 4096a = torch.arange(0.0, M *N, device="cuda", dtype=torch.bfloat16).reshape(M,N)_a = from_dlpack(a, assumed_align=16)tma_example(_a)
```

### 2.3 TensorCore WGMMA

#### 2.3.1 WGMMA编程概述

Hopper的TensorCore操作变得粒度更大了 它需要4个warp(128个线程)组成一个WarpGroup在一个SM内同时调用TensorCore,

![图片](assets/bdb5681ca777.png)

然后从右图可以看到, A操作数可以来自SMEM也可以来自寄存器, 而B需要在SMEM中. 并且四个Warp按照M维度拼接而成, 因此操作数B只能放在SMEM, 通过在TensorCore上广播给四个Warp.  而编程者只需要知道每个Warp在指令(WGMMA_instr)支持 16xNx256bit操作, 然后整个WarpGroup在指令上被拼接成支持 64xNx256bit.

然后Shmem描述符定义了操作数在SMEM中的Layout, 特别关键的是使用了硬件的Swizzle方式, 保证TMA和WGMMA的Pattern一致.

最重要的一点是, WGMMA TensorCore的操作也被改为异步操作了, 使用commit_group和wait_group来检查. 常见的指令顺序如下:

![图片](assets/d0f96f426a36.png)

首选需要一个fence表示数据已经准备就绪, 紧接着沿着M和K方向发送多条wgmma指令. 等待指令发送完成后, 对这些所有的in-flight指令打包完成一个commit, 然后通过wait_group等待完成. 下面是一个具体的流程

![图片](assets/783b625309bd.png)

首先Producer TMA发起读取数据到SMEM的流程, 然后作为Consumer需要等待`smem_full` Barrier, 即图中的紫色块. 然后issue WGMMA指令并commit_group. 接下来wait_group等待WGMMA完成, 然后再检查下一个`smem_full` Barrier是否完成, 并issue下一轮的WGMMA指令. 但我们注意到这样的方式中间TensorCore有一定时间的空闲, 如果需要打满, 我们则需要重新排布流水线, 采用多级流水线的方式, 在WGMMA commit_group和wait_group之间的这一段时间检查下一个slot的SMEM是否完成, 并issue下一轮的WGMMA指令.

![图片](assets/7a9e5a7e1f44.png)

具体的代码如下,实际上是有两个计数器`smem_pipe_release`和`smem_pipe_read`. 初始时两者是相等的, 例如都为0. 在Prologue时,然后首先检查数据是否完成加载(上图中的紫色Wait Smem 0), 如果完成则issue WGMMA(上图中白色的字符的WGMMA操作). 然后`++smem_pipe_read`, 对下一块SMEM是否Ready进行检测(上图中紫色的Wait Smem 1), 如果Ready再issue 第二个stage WGMMA(上图中红色字体的WGMMA). 此时`smem_pipe_read=2`, 而`smem_pipe_release=0`, 然后进入稳定的流水Mainloop. 如上右图所示. 但是在Release memory的时候, 总是release前一个stage的memory.

![图片](assets/98979827441b.png)

#### 2.3.2 Swizzle

首先我们从芯片的物理视角来看, SMEM是一个包含32个bank每个bank 32bits的结构, 例如一个 `float s[64]`的数组存储格式如下:

![图片](assets/824ba1f6bb1c.png)

对于Hopper TensorCore, 它支持Warp-Group Level指令为: `64xNx256bits`, 其中`N=[8,256],step=8`, 对于Warp-Level的TensorCore指令为`16x8x256bits`

![图片](assets/bb2bf95a4635.png)

例如一个BF16的WGMMA Shape为 64xNx16, 以A矩阵为例, 那么它为一个64x16的矩阵, 在M维度由4个Warp拼接而成, 每个Warp执行一个16x16的子矩阵. 这个16x16的矩阵实际上是由2x2个Core Matrix构成的, 每个Core Matrix维度为8x128bits.

在TensorCore运算的过程中, SMEM既要通过TMA从GMEM中读取数据并写入. 又要从SMEM中读取数据到TensoreCore. 因此整个过程中都参与了读写操作. 简单的col-major或者row-major的布局无法兼顾SMEM同时满足读和写无bank conflict的需求, 因此需要Swizzle的操作.

在Ampere中数据需要从GMEM拷贝到SMEM, 然后再加载到RMEM供TensorCore计算, 整个过程需要程序员手工参与到swizzle的处理, 这是一个非常繁琐的过程. 因此在Hopper上进行了重新设计, 操作数TensorCore可以直接从SMEM中读取, Nvidia针对这一块进行了优化, 让 TMA 和 TensorCore 支持一样的Swizzle特征, 仅需要一个smem描述符即可, 程序员并不需要在main loop中感知并处理swizzle, 所有的操作都由硬件处理完成.

但是计算的结果还存在RMEM中, 因此在Epilogue阶段还是需要一些处理. 而Nvidia的文档对于Swizzle的解释似乎并不是那么的完善, 所以这一节对Swizzle详细做一些分析.

首先我们来看TMA指令, 它需要保证src/dst内存地址对齐16Bytes, 并且操作的size也为16B的倍数. 因此对于swizzle的单个块的大小为16B. 例如我们在SMEM中可以分配一个矩阵`__shared__ float4 sA[8][8]`, float4刚好满足16B的原子操作.

按照k-dim连续存储如下左图所示, 图上不同的颜色代表了SMEM中不同的Bank. 当需要按列读写时将发生严重的Bank conflict. 如果我们按照如下右图的方式排布, 我们可以看到无论是读写任何一行和任何一列都没有bank conflict

![图片](assets/4b37140c616d.png)

因此我们需要构造一个对于硬件实现非常简单快速的Swizzle函数来表达如上的Layout.在cutlass swizzle.hpp[11]中描述了这样一个算法, 函数的被表示为`Swizzle<BBits, MBase, SShift>` 具体算法如下:

```
// A generic Swizzle functor/* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx *                               ^--^ MBase is the number of least-sig bits to keep constant *                  ^-^       ^-^     BBits is the number of bits in the mask *                    ^---------^     SShift is the distance to shift the YYY mask *                                       (pos shifts YYY to the right, neg shifts YYY to the left) * * e.g. Given * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx * the result is * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY */
```

`MBase`: num_base, 它表示整个最低地址的多少位保持不变, 在Hopper中由于TMA的操作粒度是16Bytes, 因此Mbase = 4

`BBits`: num_bits, 表示参与运算的高位YYY和低位ZZZ的位宽为多少bits

`SShift`: num_shift, 表示高位和低位的间隔为多少bits

运算采用XOR, 一方面位操作对于硬件是非常高效的, 另一方面它有一个很好的计算可逆的性质, 例如YYYY= 1111 ZZZZ=1011, AAAA = ZZZZ XOR YYYY = 0100, 而我们再对生成的Swizzle地址求逆, 即 YYYY XOR AAAA = 1011 = ZZZZ即可简单的恢复成原地址.

我们以Swizzle<3, 4, 3>为例, 即

```
Swizzle<3,4,3> =>num_bits = 3num_base = 4num_shft = 3 //bit_msk = (1 << num_bits)-1bit_msk = (0b00000000_00000001 << 3) - 1 = 0b00000000_00000111//yyy_msk = bit_msk << (num_base + max(0,num_shft))yyy_msk = 0b00000000_00000111 << (4 + 3) = 0b00000011_10000000// return offset ^ shiftr(input_number & yyy_msk{}, num_shft{})// 例如我们取input_number = 0b00000011_11111111 = 1023//input_number & yyy_msk0b00000011_11111111 & 0b00000011_10000000 = 0b00000011_10000000// (input_number & yyy_msk) >> num_shft0b00000011_10000000 >> 3 =  0b00000000_01110000// return value = input_number ^ ((input_number & yyy_msk) >> num_shft)0b00000011_11111111 ^ 0b00000000_01110000 = 0b00000011_10001111 = 911# 逆运算SMEM->GMEM offsetinput_number = 0b00000011_10001111 = 911// input_number & yyy_msk0b00000011_10001111 & 0b00000011_10000000 = 0b00000011_10000000// (input_number & yyy_msk) >> num_shft0b00000011_10000000 >> 3 = 0b00000000_01110000// return value = input_number ^ ((input_number & yyy_msk) >> num_shft)0b00000011_10001111 ^ 0b00000000_01110000 = 0b00000011_11111111 = 1023
```

在Hopper中支持如下几种Swizzle模式:

![图片](assets/e6d968565bdd.png)

其中涉及Layout的参数如下:

`T= 128 / sizeof-elements-in-bits`: 代表以128 bits(16B)为单位的元素个数.

`m`: 代表在同一行有多少个重复的pattern

`k`: 代表在同一列有多少个重复的pattern

`LBO(leading dimension byte offset)`: 在 K 维度上, 两个相邻core matrix之间的字节距离.

`SBO(stride dimension byte offset)`: 在 M 或 N 维度上, 两个相邻core matrix之间的字节距离.

对于TMA而言, 在构造Tensor Descriptor的时候, 有一项参数为Swizzle. 而在TensorCore中, 可以通过一个64bits的寄存器值来构建矩阵描述符, 并用于指定SMEM中参与矩阵乘加运算的矩阵的属性, 其中包含了关于swizzle模式的描述.

![图片](assets/5b8d0c421372.png)

在Hopper中设置Swizzle的方法如下 ,例如我们在准备TMA和WGMMA操作前, 需要sem_layout, 如下所示:

```
import torchimport cutlassimport cutlass.cute as cuteimport cutlass.utils as utilsimport cutlass.utils.hopper_helpers as sm90_utilsfrom cutlass.cute.runtime import from_dlpack@cute.jitdef swizzle_test(    A : cute.Tensor,):        tile_shape_mn = (16,64)    a_dtype = A.element_type    a_layout = utils.LayoutEnum.from_tensor(A)    a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(        kind=sm90_utils.get_smem_layout_atom(            a_layout,            a_dtype,            major_mode_size=tile_shape_mn[0],        ),        element_type=a_dtype    )    cute.printf("smem layout atom{}",a_smem_layout_atom)        a_smem_layout = cute.tile_to_shape(        atom = a_smem_layout_atom,        trg_shape=tile_shape_mn,        order=(0,1)    )        cute.printf("smem layout {}",a_smem_layout)    M,N = 4096, 4096a = torch.arange(0.0, M *N, device="cuda", dtype=torch.bfloat16).reshape(N,M)_a = from_dlpack(a, assumed_align=16)swizzle_test(_a)# outputsmem layout atomS<1,4,3> o 0 o (8,16):(16,1)smem layout S<1,4,3> o 0 o ((8,2),(16,4)):((16,128),(1,256))
```

在`sm90_utils.get_smem_layout_atom`中会根据major_mode_size计算majaor_mode_bits 并根据连续性的需求返回Swizzle Mode.

```
def get_smem_layout_atom(    layout: LayoutEnum,    element_type: Type[Numeric],    major_mode_size: int,    *,    loc=None,    ip=None,):    assert major_mode_size % 8 == 0    sw128_num_contiguous_bits = 1024    sw64_num_contiguous_bits = 512    sw32_num_contiguous_bits = 256    major_mode_size_bits = major_mode_size * element_type.width    if layout.sm90_mma_major_mode() == OperandMajorMode.MN:        if major_mode_size_bits % sw128_num_contiguous_bits == 0:            return cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128        if major_mode_size_bits % sw64_num_contiguous_bits == 0:            return cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64        if major_mode_size_bits % sw32_num_contiguous_bits == 0:            return cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32        return cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER    if major_mode_size_bits % sw128_num_contiguous_bits == 0:        return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128    if major_mode_size_bits % sw64_num_contiguous_bits == 0:        return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64    if major_mode_size_bits % sw32_num_contiguous_bits == 0:        return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32    return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER
```

在`make_smem_layout_atom`中即会生成相应的swizzle

```
@dsl_user_opdef make_smem_layout_atom(    kind: SmemLayoutAtomKind, element_type: Type[Numeric], *, loc=None, ip=None) -> core.ComposedLayout:    """    Makes a SMEM layout Atom.    This function creates a composed layout in unit of elements consistent with the requested layout    Atom kind and element data type.    :param kind:         The kind of layout Atom    :type kind:          SmemLayoutAtomKind    :param element_type: The element data type to construct the layout for    :type element_type:  Type[Numeric]    :return:             The SMEM layout atom    :rtype:              core.ComposedLayout    """    if not isinstance(element_type, NumericMeta):        raise TypeError(f"element_type must be a Numeric, but got {element_type}")    if kind in (SmemLayoutAtomKind.MN_INTER, SmemLayoutAtomKind.K_INTER):        num_contiguous_bits = 128        sw = core.make_swizzle(0, 4, 3)    elif kind in (SmemLayoutAtomKind.MN_SW32, SmemLayoutAtomKind.K_SW32):        num_contiguous_bits = 256        sw = core.make_swizzle(1, 4, 3)    elif kind in (SmemLayoutAtomKind.MN_SW64, SmemLayoutAtomKind.K_SW64):        num_contiguous_bits = 512        sw = core.make_swizzle(2, 4, 3)    elif kind in (SmemLayoutAtomKind.MN_SW128, SmemLayoutAtomKind.K_SW128):        num_contiguous_bits = 1024        sw = core.make_swizzle(3, 4, 3)    else:        raise ValueError("unrecognized SMEM layout atom kind")    num_contiguous_elems = num_contiguous_bits // element_type.width    if kind in (        SmemLayoutAtomKind.MN_INTER,        SmemLayoutAtomKind.MN_SW32,        SmemLayoutAtomKind.MN_SW64,        SmemLayoutAtomKind.MN_SW128,    ):        # M/N-major layout        return core.make_composed_layout(            sw,            0,            core.make_layout(                (num_contiguous_elems, 8), stride=(1, num_contiguous_elems)            ),            loc=loc,            ip=ip,        )    else:        # K-major layout        return core.make_composed_layout(            sw,            0,            core.make_layout(                (8, num_contiguous_elems), stride=(num_contiguous_elems, 1)            ),            loc=loc,            ip=ip,        )
```

#### 2.3.3 CuteDSL WGMMA

下面我们来解释一下CuteDSL中WGMMA的操作, 为了保证整个例子简单, 我们假设数据已经由TMA放置到了SMEM, 直接调用WGMMA进行运算, 并且这段代码不包含Epilogue处理和数据写回. 也不包含复杂的Consumer和Producer交互(后面一节详细展开), 只是简单的分析WGMMA所需要的SMEM Descriptor, tiled_mma 如何构建, 以及对应的Layout相关的内容. 完整的示例我们将在第三章详细分析.

在CuteDSL中, 调用WGMMA需要预先准备很多数据结构, 例如Tiled_MMA对象, 相应的Layout, SMEM中的描述符等信息, 如下图所示:

![图片](assets/307b3920c371.png)

我们首先来看Host侧的函数和数据准备, 我们需要生成Tiled_MMA对象, 需要知道操作数的数据类型, Tiler的shape等, 在CuteDSL中针对Hopper有一个make_trivial_tiled_mma[12]函数来构建tiled_mma

```
import torchimport cutlassimport cutlass.cute as cuteimport cutlass.torch as cutlass_torchimport cutlass.utils as utilsimport cutlass.utils.hopper_helpers as sm90_utilsfrom cutlass.cute.runtime import from_dlpack@cute.jitdef launch_gemm(    a : cute.Tensor,    b : cute.Tensor,    c : cute.Tensor,):    tile_shape_mnk = (128,128,64)    # 获取Tensor的数据类型    a_dtype = a.element_type    b_dtype = b.element_type    c_dtype = c.element_type    a_layout = utils.LayoutEnum.from_tensor(a)    b_layout = utils.LayoutEnum.from_tensor(b)    c_layout = utils.LayoutEnum.from_tensor(c)        # 创建tiled_mma对象    tiled_mma = sm90_utils.make_trivial_tiled_mma(            a_dtype,            b_dtype,            a_layout.sm90_mma_major_mode(),            b_layout.sm90_mma_major_mode(),            c_dtype,            atom_layout_mnk=(1, 1, 1),            tiler_mn=(64, tile_shape_mnk[1]),    )    
```

在这个函数内部, 默认A的操作数在SHMEM中`OperandSource.SMEM`. 然后回根据A和B的dataType选择MMA_OP, 例如FP16的时候选择MmaF16BF16Op, 最后通过MMA_OP创建MMA_ATOM, 并创建Tiled_MMA对象.

```
def make_trivial_tiled_mma(    a_dtype: Type[Numeric],    b_dtype: Type[Numeric],    a_leading_mode: OperandMajorMode,    b_leading_mode: OperandMajorMode,    acc_dtype: Type[Numeric],    atom_layout_mnk: Tuple[int, int, int],    tiler_mn: Tuple[int, int],    a_source: OperandSource = OperandSource.SMEM,    *) -> cute.TiledMma:    if a_dtype in {Float16, BFloat16}:        mma_op = MmaF16BF16Op(            a_dtype,            acc_dtype,            (*tiler_mn, 16),            a_source,            a_leading_mode,            b_leading_mode,        )    elif a_dtype in {Float8E4M3FN, Float8E5M2} and b_dtype in {        Float8E4M3FN,        Float8E5M2,    }:        mma_op = MmaF8Op(            a_dtype,            b_dtype,            acc_dtype,            (*tiler_mn, 32),            a_source,            a_leading_mode,            b_leading_mode,        )    else:        raise TypeError(f"unsupported a_dtype and b_dtype, got {a_dtype} and {b_dtype}")    return cute.make_tiled_mma(cute.make_mma_atom(mma_op), atom_layout_mnk)
```

但是cute.gemm操作还需要A/B和Accumulator C的Layout. 在Host函数中, 通过如下方式构建, 此时将会携带Swizzle相关的属性.

```
    a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))    a_is_k_major = (        a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K    )    a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]    a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(        kind=sm90_utils.get_smem_layout_atom(            a_layout,            a_dtype,            a_major_mode_size,        ),        element_type=a_dtype,    )    a_smem_layout = cute.tile_to_shape(        atom = a_smem_layout_atom,        trg_shape=(tile_shape_mnk[0], tile_shape_mnk[2]),        order=(0,1) if a_is_k_major else (1,0)    )    print(f"a smem layout {a_smem_layout}  inner {a_smem_layout.inner}  outer {a_smem_layout.outer}")    b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))    b_is_k_major = (        b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K    )    b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(        kind=sm90_utils.get_smem_layout_atom(                b_layout,                b_dtype,                b_major_mode_size,            ),        element_type=b_dtype,    )    b_smem_layout = cute.tile_to_shape(        atom = b_smem_layout_atom,        trg_shape=(tile_shape_mnk[0], tile_shape_mnk[2]),        order=(0, 1) if b_is_k_major else (1, 0),    )    print(f"b smem layout {b_smem_layout} inner {b_smem_layout.inner}  outer {b_smem_layout.outer}")    c_smem_shape = cute.slice_(tile_shape_mnk, (None, None, 0))    c_major_mode_size = tile_shape_mnk[1 if c_layout.is_n_major_c() else 0]    c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(        kind=sm90_utils.get_smem_layout_atom(                c_layout,                c_dtype,                c_major_mode_size,            ),        element_type=c_dtype,    )    c_smem_layout = cute.tile_to_shape(        atom = c_smem_layout_atom,        trg_shape=(tile_shape_mnk[0], tile_shape_mnk[1]),        order=(0, 1) if c_layout.is_n_major_c() else (1, 0),    )     print(f"c smem layout {c_smem_layout} inner {c_smem_layout.inner}  outer {c_smem_layout.outer}")    #outputa smem layout S<3,4,3> o 0 o ((8,16),(64,1)):((64,512),(1,0))  inner S<3,4,3>  outer ((8,16),(64,1)):((64,512),(1,0))b smem layout S<3,4,3> o 0 o ((8,16),(64,1)):((64,512),(1,0)) inner S<3,4,3>  outer ((8,16),(64,1)):((64,512),(1,0))c smem layout S<3,4,3> o 0 o ((8,16),(32,4)):((32,256),(1,4096)) inner S<3,4,3>  outer ((8,16),(32,4)):((32,256),(1,4096))
```

最后计算smem内存占用和构建SMEM的数据结构体, 然后launch Kernel

```
    smem_size = (cute.size_in_bytes(cutlass.BFloat16, a_smem_layout) +                  cute.size_in_bytes(cutlass.BFloat16, b_smem_layout) +                 cute.size_in_bytes(cutlass.Float32, c_smem_layout))    buffer_align_bytes = 1024    @cute.struct    class SharedStorage:        sA: cute.struct.Align[            cute.struct.MemRange[                a_dtype, cute.cosize(a_smem_layout)            ],            buffer_align_bytes,        ]        sB: cute.struct.Align[            cute.struct.MemRange[                b_dtype, cute.cosize(b_smem_layout)            ],            buffer_align_bytes,        ]    gemm_kernel(        a_smem_layout,         b_smem_layout,        c_smem_layout,        tile_shape_mnk,        tiled_mma,        SharedStorage    ).launch(        grid=(1, 1, 1),        block=(128, 1, 1),        smem=smem_size    )
```

在Kernel函数中, 我们传入了A/B/C的在SHMEM中的layout, Tiled_MMA和SMEM中的存储结构体以及Tile的Shape. 首先在SMEM中allocate结构体, 并根据layout中的swizzle定义, 获取出Tensor

```
@cute.kerneldef gemm_kernel(        a_smem_layout: cute.ComposedLayout,         b_smem_layout: cute.ComposedLayout,        c_smem_layout: cute.ComposedLayout,        tile_shape_mnk: tuple [int,int,int],        tiled_mma: cute.TiledMma,        SharedStorage: cutlass.Constexpr    ):    acc_type =  cutlass.Float32    tidx, _, _ = cute.arch.thread_idx()     smem = cutlass.utils.SmemAllocator()    storage = smem.allocate(SharedStorage)    sA = storage.sA.get_tensor(        a_smem_layout.outer, swizzle = a_smem_layout.inner    )    sB = storage.sB.get_tensor(        b_smem_layout.outer, swizzle = c_smem_layout.inner    )    sC_ptr = cute.recast_ptr(        sA.iterator, c_smem_layout.inner, dtype=acc_type    )    sC = cute.make_tensor(sC_ptr, c_smem_layout.outer)
```

然后针对每个thread构建fragment.

```
    thr_mma = tiled_mma.get_slice(tidx)    tCsA = thr_mma.partition_A(sA)    tCsB = thr_mma.partition_B(sB)    tCgC = thr_mma.partition_C(sC)    tCrA = thr_mma.make_fragment_A(tCsA)    tCrB = thr_mma.make_fragment_B(tCsB)    accumulator = cute.make_fragment(tCgC.shape, acc_type)    
```

最后执行cute.gemm, 注意通常在执行前, 需要warpgroup的fence确保数据都已经加载完成. 然后指令发射后, 需要commit_group. 并wait_group(N)等待完成.

```
    cute.nvgpu.warpgroup.fence()    cute.gemm(tiled_mma, accumulator, tCrA, tCrB, accumulator)    if tidx == 0 :        cute.printf("WGMMA Issued")        cute.nvgpu.warpgroup.commit_group()    cute.nvgpu.warpgroup.wait_group(0)    if tidx == 0 :        cute.printf("WGMMA Finished")
```

### 2.4 CuteDSL异步编程

下一步就是要把TMA和TensorCore WGMMA两个异步调用协同在一起, 如下图所示:

![图片](assets/fadf1c36c8ad.png)

然后把SMEM分成了多级流水线, 每一级流水线都有`smem_empty`和`smem_full`两个信号量, 其实smem_full这个词很容易让人误解, 我看最近的cuteDSL文档已经将它改为`smem_ready`了, 这样也挺好的.

但是交互的过程还是非常复杂的, 而且很容易出现处理不对导致的Kernel hang住的问题. CuteDSL对整个交互过程做了一个封装,即pipeline[13] 这一节我们来详细展开一下.

这个框架基于经典的 Producer-Consumer 模型:

**生产者 (Producer)**: 通常是负责从低速的 Global Memory 中加载数据的异步操作. 在现代NVIDIA GPU上, 这可以是`cp.async`指令(使用`PipelineCpAsync`)或更高效的张量内存加速器TMA (使用`PipelineTmaAsync`).

**消费者 (Consumer)**: 通常是执行计算任务的线程, 例如执行矩阵乘法(MMA)的Warp.

**缓冲区 (Buffer)**: 位于高速共享内存(SMEM)中的一块区域, 用于生产者和消费者之间的数据交换.

首先这个代码定义了一个`PipelineAsync`的基类, 然后针对cp.async异步的场景定义了`PipelineCpAsync`, 针对TMA异步内存访问的场景定义了`PipelineTmaAsync`和`PipelineTmaMultiConsumersAsync`, 后者支持多个Consumer能够更好的overlap计算和访问内存的延迟, 但是看代码应该是给blackwell用的, 不知道为什么放在SM_90的文件内. 最后还定义了一个用于epilogue的`PipelineTmaStore`类.

#### 2.4.1 PipelineAsync

`PipelineAsync`是一个通用的流水线类, 其生产者和消费者都是异步线程. 它也作为其他专用流水线类的基类. 在这个类里定义了`smem_full`和`smem_empty`两个Mbarrier的状态机

| 屏障 | 状态 | p.acquire | p.commit | c.wait | c.release |
| --- | --- | --- | --- | --- | --- |
| empty_bar | empty | <返回> | n/a | n/a | - |
| empty_bar | wait | <阻塞> | n/a | n/a | -> empty |
| full_bar | wait | n/a | -> full | <阻塞> | n/a |
| full_bar | full | n/a | - | <返回> | n/a |

这个表格清晰地描述了双屏障同步机制:

`empty_bar`: 用于告知生产者"缓冲区是空的, 你可以写入了".

`full_bar`: 用于告知消费者"缓冲区是满的, 你可以读取了".
工作流程:
**Producer**:

调用 `acquire()`: 在 **empty_bar** 上等待. 如果缓冲区是空的 (`empty`状态), 调用立即返回. 如果消费者还未释放(`wait`状态), 则生产者阻塞.

(生产者执行写入操作)

调用 `commit()`: 在 **full_bar** 上发出一个 `arrive` 信号, 将其状态变为 `full`, 通知消费者数据已准备好.

**消费者**:

调用 `wait()`: 在 **full_bar** 上等待. 如果数据已准备好 (`full`状态), 调用立即返回. 如果生产者还未提交 (`wait`状态), 则消费者阻塞.

(消费者执行读取操作)

调用 `release()`: 在 **empty_bar** 上发出一个 `arrive` 信号, 将其状态变回 `empty`, 通知生产者此缓冲区可以再次使用.

这样的一对Mbarrier被放置在一个环形缓冲区的Slot内, `mbarrier`是Hopper架构引入的关键同步原语. 它是一个存放在共享内存中的对象, 相比于旧有的`barrier.sync`等同步方式, 它提供了更灵活的同步模式, 具体操作流程前面章节已经详细介绍了.

![图片](assets/141ac0a89208.png)

其中:

`X`: 空缓冲区 (初始状态)

`W`: 生产者正在写入 (生产者等待缓冲区变空)

`D`: 数据就绪 (生产者已将数据写入缓冲区)

`R`: 消费者正在读取 (消费者正在消费缓冲区中的数据)

设计成循环缓冲区的原因是, 为了隐藏例如从全局内存加载数据到共享内存的数百个时钟周期的延迟, Kernel被构造成流水线形式. 假设我们有`N`个阶段(`num_stages`), 这意味着我们在共享内存中分配了`N`个缓冲区. *   在时刻`t`, 消费者(计算单元)正在处理第`k`个缓冲区的数据. *   同时, 生产者(内存加载单元)可以异步地将数据加载到第`k+1`个缓冲区. *   当消费者完成第`k`个缓冲区的工作后, 它可以立即开始处理第`k+1`个缓冲区的数据, 因为数据已经准备好了.

这种方式使得计算和数据加载可以并行执行, 从而提高了SM的吞吐量.

它的关键成员方法如下:

`sync_object_full`, `sync_object_empty`: 这两个是核心同步对象. 它们通常是 `MbarrierArray` 的实例, 分别管理着`N`个阶段的 "smem_full" Mbarrier和 "smem_empty" Mbarrier.

`num_stages`: 流水线深度.

`producer_mask`, `consumer_mask`: 用于多CTA协作时, 指定哪些CTA应该参与或接收`arrive`信号, 通常用于CGA范围的同步.

`create()`: 静态工厂方法, 用于创建和初始化一个`PipelineAsync`实例. 它会根据传入的参数创建`sync_object_full`和`sync_object_empty`. 注意 `barrier_storage.align(min_align=8)` 确保了 `mbarrier` 对象在共享内存中的8字节对齐, 这是硬件要求.

`producer_acquire()` / `producer_commit()`: 实现生产者的"获取空缓冲区"和"提交满缓冲区"的逻辑.

`consumer_wait()` / `consumer_release()`: 实现消费者的"等待满缓冲区"和"释放空缓冲区"的逻辑.

`producer_tail()`: 一个重要的收尾函数. 在生产者循环结束后调用, 确保所有流水线阶段都已正确同步. 它通过前进到最后一个使用的缓冲区并对其执行`acquire`操作, 来等待消费者完成所有剩余的工作. 这可以防止内核退出后, 仍然有悬空的`mbarrier`信号, 导致状态不一致.

`make_producer()` / `make_consumer()`: 创建用户友好的`PipelineProducer`和`PipelineConsumer`对象, 简化API调用.

这里稍微展开一下`make_producer()` / `make_consumer()`, 如果没有这两个封装, 我们将内部的状态直接暴露. 如下所示,用户将不得不直接与`PipelineAsync`对象交互, 代码可能会是这样:

```
# 不使用接口类的伪代码pipeline = PipelineAsync.create(...)producer_state = make_pipeline_state(PipelineUserType.Producer, num_stages)consumer_state = make_pipeline_state(PipelineUserType.Consumer, num_stages)# 生产者循环for i in range(...):    pipeline.producer_acquire(producer_state) # 等待    # 在 producer_state.index 指向的缓冲区里写数据    pipeline.producer_commit(producer_state) # 提交    producer_state.advance() # 手动推进状态# 消费者循环for i in range(...):    pipeline.consumer_wait(consumer_state) # 等待    # 在 consumer_state.index 指向的缓冲区里读数据    pipeline.consumer_release(consumer_state) # 释放    consumer_state.advance() # 手动推进状态
```

这种方式存在几个严重问题:

**状态管理暴露**: 用户需要手动创建, 传递和更新`producer_state`和`consumer_state`. 这非常繁琐且容易出错.

**API不直观**: `pipeline.producer_acquire(state)`的调用方式不如`producer.acquire()`直观.

**安全性差**: 在复杂的循环或条件逻辑中, 很容易错误地对一个已经`advance`的状态进行`commit`, 或者忘记`advance`, 导致死锁或数据竞争.

`PipelineProducer`和`PipelineConsumer`类的目的就是解决以上所有问题:

`PipelineAsync`及其子类: 关注流水线如何工作. 它们是底层的同步引擎, 处理`mbarrier`, TMA事务, 跨CTA信令等复杂机制.

`PipelineProducer`/`Consumer`: 关注如何使用流水线. 它们为最终用户提供了角色化的, 状态独立的简单接口.

`PipelineProducer`代表了生产者角色, 一个`PipelineProducer`实例持有三个关键的私有成员:

`__pipeline: PipelineAsync`: 对底层流水线引擎的引用. 所有实际的同步操作都委托给这个对象.

`__state: PipelineState`: **可变的状态对象**. 这是`PipelineProducer`的核心, 它追踪生产者**下一次**将要操作的缓冲区索引(`index`)和同步相位(`phase`). 每次调用`advance()`时, 这个`__state`对象内部的值会更新.

`__group: CooperativeGroup`: 标识参与生产操作的线程组.

生产者的标准工作流是 *Acquire -> (Produce Data) -> Commit -> Advance*.

acquire() -> ImmutableResourceHandle:
调用 self.__pipeline.producer_acquire(self.__state, ...) 来执行**阻塞等待**. 这会等待`__state`所指向的那个缓冲区变为空(即上一个循环的消费者已经释放了它).

一旦等待成功, 它会调用`self.__state.clone()`创建一个当前状态的只读快照.

然后, 它将这个快照和`__pipeline`对象的引用包装成一个`ImmutableResourceHandle`并返回.

**作用**: `acquire`是获取资源的行为. 返回的`handle`证明你拥有了对这个特定缓冲区的使用权.
advance():
调用`self.__state.advance()`.

这会更新`__state`内部的`index`和`phase`, 使其指向环形缓冲区中的下一个阶段.

**作用**: 这个方法改变了`PipelineProducer`对象自身的内部状态. 它为下一次`acquire`做准备.
acquire_and_advance() -> ImmutableResourceHandle:
简单地将`acquire()`和`advance()`合并.

它首先调用`acquire()`获取当前阶段的`handle`.

然后立即调用`advance()`将生产者的内部状态推进到下一阶段.

**作用**: 这是最常用的模式. 生产者获取了阶段`k`的缓冲区后, 就可以立即准备去获取阶段`k+1`了, 将状态提前推进可以更好地组织代码逻辑.
commit(handle: Optional[ImmutableResourceHandle] = None):
如果传入了`handle`, 它会调用`handle.commit()`. 这是推荐的使用方式.

如果`handle`为`None`, 它会直接使用`self.__state`来提交: self.__pipeline.producer_commit(self.__state).

**作用**: 这个方法通知流水线, 生产者已经完成了对缓冲区的数据写入.

注意: ImmutableResourceHandle这个内部类是设计的核心, 提供了至关重要的**安全性**. 当`producer.acquire_and_advance()`被调用时, `producer`对象自身的`__state`已经指向了下一阶段(例如`k+1`), 但返回的`handle`内部的`__immutable_state`仍然指向当前阶段(阶段`k`).

之后, 生产者线程执行数据加载等操作, 当它完成时, 它调用`handle.commit()`. 这个调用使用的是`handle`内部保存的那个 不可变的, 指向阶段`k`的`state`. 这就保证了:

你提交的永远是你当初获取的那个阶段.

你不会因为`producer`对象状态的改变而意外地提交了错误的阶段 (例如, 错误地提交了尚未写入数据的阶段`k+1`).

示例代码:

```
producer = pipeline.make_producer()# 迭代 0handle_k0 = producer.acquire_and_advance() # producer.__state -> 1, handle_k0.__state -> 0# ... (向缓冲区 0 写入数据)handle_k0.commit() # 正确地在流水线阶段 0 上发出 commit 信号# 迭代 1handle_k1 = producer.acquire_and_advance() # producer.__state -> 2, handle_k1.__state -> 1# ... (向缓冲区 1 写入数据)handle_k1.commit() # 正确地在流水线阶段 1 上发出 commit 信号
```

`PipelineConsumer`也包含`__pipeline`, `__state`, 和 `__group`. 这里的`__state`追踪消费者下一个要消费的缓冲区.

消费者的标准工作流是 *Wait -> (Consume Data) -> Release -> Advance*.

wait() -> ImmutableResourceHandle:
调用`self.__pipeline.consumer_wait(self.__state, ...)`来执行**阻塞等待**. 这会等待`__state`所指向的那个缓冲区变满(即生产者已经提交了它).

与生产者一样, 等待成功后, 创建`__state`的只读快照.

返回包含此快照的`ImmutableResourceHandle`.

**作用**: `wait`是等待数据就绪的行为. 返回的`handle`证明了你获得了对这个已填充数据的缓冲区的读取权.
advance():
与生产者一样, 调用`self.__state.advance()`将消费者的内部状态推进到下一阶段.

**作用**: 为下一次`wait`做准备.
wait_and_advance() -> ImmutableResourceHandle:
合并`wait()`和`advance()`.

**作用**: 这是消费者循环中最常用的方法.
release(handle: Optional[ImmutableResourceHandle] = None):
如果传入`handle`, 它会调用`handle.release()`. 这是推荐用法.

`handle.release()`内部会调用`self.get_origin().consumer_release(...)`, 使用的是`handle`内部保存的不可变状态.

**作用**: 这个方法通知流水线, 消费者已经完成了对缓冲区数据的读取, 该缓冲区现在是"空的", 可以被生产者再次使用.

`PipelineCpAsync` 类的实现和Base类基本相似,在此就省略了.

#### 2.4.2 PipelineTmaAsync

接下来我们详细分析一下在Hopper中最常用的 PipelineTmaAsync 类, 如下图所示:

![图片](assets/203b518b7bca.png)

这是为Hopper架构的TMA设计的流水线. 生产者是TMA Load操作.这个类引入了显著的改变, 适配TMA工作方式.
producer_acquire():
此方法与基类有本质区别. 它不仅在`empty_bar`上`wait()`, 还立即在`full_bar`上`arrive()`.这是因为TMA的同步模型是基于事务的. `full_bar`在这里被配置为一个事务屏障. 通过`arrive()`操作来初始化这个事务, 并设置一个期望的字节数(`tx_bytes`). 然后, 发出`tma_load`指令. TMA硬件在异步执行加载时, 会自动对这个`mbarrier`执行"到达"操作, 逐渐增加actual tx_bytes.
producer_commit():
此方法是`pass` (空操作), 因为真正的"提交"动作(即TMA加载数据)是由`tma_load`指令异步触发的. 在调用`producer_acquire`之后就可以继续执行其他任务, 无需再次进行同步提交.
init_empty_barrier_arrive_signal()
它专门用于CGA.当一个Cluster内的多个CTA协同处理一个大的GEMM时, 一个CTA(生产者)完成计算后需要通知另一个CTA(消费者)可以使用某块全局内存区域. 这个函数就是为了高效地完成"缓冲区已空"这一信号的跨CTA传递. 它根据线程ID(`tidx`)和集群的布局(`cta_layout_vmnk`), 策略性地选择一个线程(`is_signalling_thread`)来发送`arrive`信号给目标CTA(`dst_rank`).

`is_same_row_or_col`的判断是为了利用集群内部的通信拓扑, 优先选择同行或同列的CTA进行通信, 因为它们之间可能有更快的物理连接. 这避免了所有线程都发送信号造成的拥堵, 也保证了信号的正确传递.
consumer_release():
它是有条件执行的. 只有被`init_empty_barrier_arrive_signal`选中的"信令线程"(`is_signalling_thread`)才会真正执行`arrive`操作, 通知生产者缓冲区已空.
2.4.3 PipelineTmaStore
用于同步Epilogue中的TMA Store操作, 即将计算结果从共享内存异步写回全局内存. 这是一个只有生产者(执行store的线程)的特殊流水线, 没有消费者.

它不使用`mbarrier`, 而是使用`TmaStoreFence`对象, 这是专为TMA Store设计的同步原语.

`producer_acquire()`: 等待, 确保之前的TMA Store操作已完成, 硬件资源可用.

`producer_commit()`: 到达, 提交一个新的TMA Store操作.

`producer_tail()`: 调用`sync_object_full.tail()`, 这是一个栅栏(fence)操作, 确保在内核退出前, 所有已提交的TMA Store都已经完成, 保证数据的全局可见性.

由于需要结合TMA, 详细的使用代码在下一章介绍.

### 2.5 小结

好了, 到此为止, 这一章详细介绍了CGA, TMA和TensorCore以及相互协同的Pipeline, 就此龙珠凑齐了,可以对cuteDSL中Hopper DenseGemm的代码串在一起分析了.

## 3. Hopper DenseGemm

### 3.1 Host侧函数

在Host函数中, 主要用于Kernel运算时首先需要生成用于WGMMA计算的Tiled_MMA对象, 然后根据MMA指令的K的长度, 配合tiled_mn计算出tiled_mnk. 然后根据SMEM内存的容量计算流水线级数(ab_stage 和epi_stage), 最后生成Tile_A, Tile_B和Tile_Epilogue在SMEM中的layout, 此时会判断使用哪一种Swizzle. 然后针对TMA还需要生成TMA ATOM和Tensor Descriptor, 并且需要考虑根据CGA的Layout是否启用TMA组播的能力. 接下来针对SMEM的多级流水线管理, 生成一个SharedStorage结构题. 最后根据CGA的shape和Tiled_mnk的shape, 计算用于launch kernel的grid参数, 最后调用kernel.

根据tiled_mn 的 shape 判断是否需要2个WarpGroup执行避免register spill, 并生成分析atom_layout_mnk对象.

获取Tensor A/B/C的数据类型及Majorness, 并静态验证数据类型

生成Tiled_MMA对象

调用 *self._setup_attributes()*

检查CTA Tile Shape

创建 Tiled-MMA 对象

根据MMA指令的 Shape_k 更新 tile_shape_mnk 中的 K 维度大小

基于cluster_shape_mn 生成 CTA Layout 在CGA中的layout

基于CGA Layout判断A/B是否要组播TMA,即is_a_mcast / is_b_mcast

计算epi_tile shape

针对SMEM容量计算流水线级数 ab_stage 和epi_stage

基于 *_make_smem_layouts* 函数获取A/B/C的smem_layout

构建tma_atom 和tma_tensor

基于tile_shape_mnk和cluster_shape_mn 计算grid 用于kernel launch

构建SharedStorage 结构体

Launch kernel

其中关于Tiled-MMA对象的处理及tiled_k维度的计算和CGA/Grid Layout已经在2.1.2节详细介绍了, 下面我们主要来关注SMEM相关的Layout计算.

首先我们需要根据SMEM的容量和Tile_MNK评估需要的流水线级数,在cuteDSL example中定义了一个`_compute_stages`函数,计算并返回  操作数和最后 Epilogue所需要的流水线级数, 具体实现逻辑如下:

```
@cute.jitdef tma_stage(    a : cute.Tensor,    b : cute.Tensor,    c : cute.Tensor,):        tile_shape_mnk = (128,128,64)    # 获取Tensor的数据类型    a_dtype = a.element_type    b_dtype = b.element_type    c_dtype = c.element_type    # 针对Epilogue设置固定的stage    epi_stage = 4    # C相关的Epilogue操作复用AB的内存    epi_bytes = 0        #获取SMEM容量并定义占用率    smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")    occupancy = 1    #获取A/B的Tile Shape, 并计算每一级流水线内存占用    a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))    b_shape = cute.slice_(tile_shape_mnk, (0, None, None))    ab_bytes_per_stage = (        cute.size(a_shape) * a_dtype.width // 8        + cute.size(b_shape) * b_dtype.width // 8    )    # 另外还需要考虑Mbarrier相关的helper数据结构占用的空间    mbar_helpers_bytes = 1024    # AB的stages数目如下:    ab_stage = (        smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes    ) // ab_bytes_per_stage    cute.printf("ab_stage {} epi_stage {}",ab_stage, epi_stage)M,N,K,L = 4096, 4096, 4097,16a = torch.randn(L, M, K, device="cuda", dtype=torch.bfloat16).permute(1,2,0)b = torch.randn(L, N, K, device="cuda", dtype=torch.bfloat16).permute(1,2,0)c = torch.zeros(L, M, N, device="cuda", dtype=torch.float32).permute(1,2,0)_a = from_dlpack(a, assumed_align=16)_b = from_dlpack(b, assumed_align=16)_c = from_dlpack(c, assumed_align=16)tma_stage(_a,_b,_c)#outputab_stage 7 epi_stage 4
```

然后我们还需要对CGA的shape进行判断, 是否使用TMA Multicast

```
        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))        self.num_mcast_ctas_a = self.cluster_shape_mn[1]        self.num_mcast_ctas_b = self.cluster_shape_mn[0]        self.is_a_mcast = self.num_mcast_ctas_a > 1        self.is_b_mcast = self.num_mcast_ctas_b > 1
```

在计算SMEM Layout前, 我们还需要考虑epilogue tile的shape. 当Tile较大需要设置, 由于单个WarpGroup运算会导致RegisterSpill, 因此需要两个warp group进行运算, epi_tile需要通过_sm90_compute_tile_shape_or_override 函数进行处理, 控制寄存器的占用.

```
    is_cooperative = self.atom_layout_mnk == (2, 1, 1)    self.epi_tile = self._sm90_compute_tile_shape_or_override(        self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative    )    @staticmethod    def _sm90_compute_tile_shape_or_override(        tile_shape_mnk: tuple[int, int, int],        element_type: type[cutlass.Numeric],        is_cooperative: bool = False,        epi_tile_override: tuple[int, int] | None = None,    ) -> tuple[int, int]:        if epi_tile_override is not None:            return epi_tile_override                if is_cooperative:            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))            return (tile_m, tile_n)        else:            n_perf = 64 if element_type.width == 8 else 32            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))            return (tile_m, tile_n)
```

最后根据stage的数量, 计算在smem中每个stage的Layout, 即调用example中的_make_smem_layouts函数获取`a_smem_layout_staged`, `b_smem_layout_staged`和`epi_smem_layout_staged`.

```
    @staticmethod    def _make_smem_layouts(        tile_shape_mnk: tuple[int, int, int],        epi_tile: tuple[int, int],        a_dtype: type[cutlass.Numeric],        a_layout: utils.LayoutEnum,        b_dtype: type[cutlass.Numeric],        b_layout: utils.LayoutEnum,        ab_stage: int,        c_dtype: type[cutlass.Numeric],        c_layout: utils.LayoutEnum,        epi_stage: int,    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))        a_is_k_major = (            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K        )        b_is_k_major = (            b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K        )        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(            sm90_utils.get_smem_layout_atom(                a_layout,                a_dtype,                a_major_mode_size,            ),            a_dtype,        )        a_smem_layout_staged = cute.tile_to_shape(            a_smem_layout_atom,            cute.append(a_smem_shape, ab_stage),            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),        )        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(            sm90_utils.get_smem_layout_atom(                b_layout,                b_dtype,                b_major_mode_size,            ),            b_dtype,        )        b_smem_layout_staged = cute.tile_to_shape(            b_smem_layout_atom,            cute.append(b_smem_shape, ab_stage),            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),        )        c_smem_shape = epi_tile        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(            sm90_utils.get_smem_layout_atom(                c_layout,                c_dtype,                c_major_mode_size,            ),            c_dtype,        )        epi_smem_layout_staged = cute.tile_to_shape(            c_smem_layout_atom,            cute.append(c_smem_shape, epi_stage),            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),        )        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged
```

然后是生成TMA-ATOM和Tensor Descriptor, 主要是A/B加载和Epi_tile存储, 加载过程中需要判断是否使用了Multicast

```
    def _make_tma_atoms_and_tensors(        tensor: cute.Tensor,        smem_layout_staged: cute.ComposedLayout,        smem_tile: tuple[int, int],        mcast_dim: int,    ) -> tuple[cute.CopyAtom, cute.Tensor]:        op = (            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()            if mcast_dim == 1            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()        )        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(            op,            tensor,            smem_layout,            smem_tile,            num_multicast=mcast_dim,        )        return tma_atom, tma_tensor
```

而存储过程, 需要考虑Epi-tile的shape和tensor-C的shape

```
   def _make_tma_store_atoms_and_tensors(        tensor_c: cute.Tensor,        epi_smem_layout_staged: cute.ComposedLayout,        epi_tile: tuple[int, int],    ) -> tuple[cute.CopyAtom, cute.Tensor]:        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))        c_cta_v_layout = cute.composition(            cute.make_identity_layout(tensor_c.shape), epi_tile        )        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),            tensor_c,            epi_smem_layout,            c_cta_v_layout,        )
```

最后就是在SMEM中的结构体, 如下所示, 首先是针对每个Stage有两个Mbarrier, 结构体中mainloop_pipeline_array_ptr存储了这个多级的Mbarrier指针. 然后就是根据A和B在SMEM中的layout和size构成MemRange.

```
        buffer_align_bytes = 1024                @cute.struct        class SharedStorage:            mainloop_pipeline_array_ptr: cute.struct.MemRange[                cutlass.Int64, self.ab_stage * 2            ]            sA: cute.struct.Align[                cute.struct.MemRange[                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)                ],                self.buffer_align_bytes,            ]            sB: cute.struct.Align[                cute.struct.MemRange[                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)                ],                self.buffer_align_bytes,            ]
```

### 3.2 Kernel函数

#### 3.2.1 第一阶段: 初始化和坐标计算

首先对TMA描述符进行prefetch, 由每个 CTA 的第一个 Warp (warp_idx == 0) 执行, 这样做可以降低第一次发起 TMA 拷贝指令时的延迟.

```
        # 获取warp-idx, make_warp_uniform只是一个compiler hint,         # 表示这个值在同一个Warp内保持不变        warp_idx = cute.arch.warp_idx()        warp_idx = cute.arch.make_warp_uniform(warp_idx)                # Prefetch TMA Descriptor        if warp_idx == 0:            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
```

然后就是获取CTA/Warp/Thread idx, 并计算cluster-id. 比较有意思的是后面这段CTA Swizzle to promote L2 data reuse. 它没有直接使用 `cluster_id` 来映射到 C 矩阵的坐标, 而是通过一个 `s_layout` 进行了Swizzle, 目的是提升 L2 缓存命中率.

默认的线性映射可能导致在物理上相邻的 SM 核心处理逻辑上相邻但物理内存上相距很远的 C 矩阵块. Swizzling 使得逻辑上相邻的集群更有可能被调度到物理上邻近的 SM 上. 当它们访问数据时, 由于数据局部性更好, L2 缓存的命中率会提高, 从而减少了对 GMEM 的带宽需求. `pid_m`, `pid_n` 就是经过 Swizzling 后, 当前 CTA 在整个 GEMM 问题中的逻辑块坐标.

然后接着对CGA中需要TMA Multicast的掩码进行处理, `make_layout_image_mask` 根据当前 CTA 在其集群内的坐标 (`cluster_coord_mnk`) 来生成一个掩码. 后续的TMA copy中会根据这个mask进行处理, 如果当前 CTA 不是多播的"源"(root), 那么这个掩码将会在后续的 TMA `copy` 操作中阻止它从 GMEM 读取数据, 但它仍然会参与接收从其他 CTA 多播过来的数据. 最后获取每个Stage的smem_layout, 并计算需要拷贝的byte size,为TMA操作做准备.

```
        # ///////////////////////////////////////////////////////////////////////////////        # Get mcast mask        # ///////////////////////////////////////////////////////////////////////////////        a_mcast_mask = cute.make_layout_image_mask(            cta_layout_mnk, cluster_coord_mnk, mode=1        )        b_mcast_mask = cute.make_layout_image_mask(            cta_layout_mnk, cluster_coord_mnk, mode=0        )        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))        tma_copy_bytes = cute.size_in_bytes(            self.a_dtype, a_smem_layout        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
```

#### 3.2.2 第二阶段: 流水线设置与内存分区

首先是基于SharedStorage结构体分配内存.

```
        smem = cutlass.utils.SmemAllocator()        storage = smem.allocate(self.shared_storage)
```

然后我就需要通过pipeline这个库来初始化memory barrier. 对于`PipelineTmaAsync`创建需要如下参数

```
    def create(        *,        num_stages: int,        producer_group: CooperativeGroup,        consumer_group: CooperativeGroup,        tx_count: int,        barrier_storage: cute.Pointer = None,        cta_layout_vmnk: Optional[cute.Layout] = None,        tidx: Optional[Int32] = None,        mcast_mode_mn: tuple[int, int] = (1, 1),    ):
```

因此在创建之前, 需要创建producer和consumer的CooperativeGroup, tx_count和num_stages已经在前一章节完成了计算, barrier_storage的指针也可以从SharedStorage结构体中获取.

```
        # 从SharedStorage结构体中获取barrier_storage指针        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()        # 创建Producer CG, 默认只有一个Thread issue TMA, 因此arrive_thr_cnt为默认值=1        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(            pipeline.Agent.Thread        )                # 然后Consumer的arrive_thr_cnt 需要考虑TMA multicast的情况        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1        num_warps = self.threads_per_cta // 32        consumer_arrive_cnt = mcast_size * num_warps        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(            pipeline.Agent.Thread, consumer_arrive_cnt        )                cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))                # 创建PipelineTmaAsync对象        mainloop_pipeline = pipeline.PipelineTmaAsync.create(            barrier_storage=mainloop_pipeline_array_ptr,            num_stages=self.ab_stage,            producer_group=mainloop_pipeline_producer_group,            consumer_group=mainloop_pipeline_consumer_group,            tx_count=tma_copy_bytes,            cta_layout_vmnk=cta_layout_vmnk,        )        # 最后在cluster level确保其中所有 CTA 都已经完成了各自的 mainloop_pipeline 对象的初始化        if cute.size(self.cluster_shape_mn) > 1:            cute.arch.cluster_arrive_relaxed()        
```

注意这里使用`cluster_arrive_relaxed`操作和`cluster_arrive`的区别. 实际上这里我们并没有线程间的数据交换, 只需要针对控制流进行同步, 不需要使用`cluster_arrive`那样保证之前的内存写入对其它CTA可见. 因此可以采用relaxed的方式, 不提供额外的内存同步保证. 这样更快, 延迟更低. 另外cluster.arrive和cluster.wait的区别是. 前者表示, 我(这个线程)已经达到了同步点, 它只是一个签到动作.  而cluster.wait则是一个阻塞操作, 表示我(这个线程)要在这里等待, 直到cluster内所有该签到的线程都已经完成签到动作.

接下来需要创建SMEM中的Tensor对象以及从GMEM中取出需要处理的Tile

```
        sA = storage.sA.get_tensor(            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner        )        sB = storage.sB.get_tensor(            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner        )        sC_ptr = cute.recast_ptr(            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype        )        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)        # 基于tile_coord/shape, 使用local_tile函数对GMEM中的Tensor进行分块,        # 并获取本地需要处理的块.        # (bM, bK, RestK)        gA_mkl = cute.local_tile(            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)        )        # (bN, bK, RestK)        gB_nkl = cute.local_tile(            mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)        )        # (bM, bN)        gC_mnl = cute.local_tile(            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)        )
```

然后我们需要针对MMA操作准备相应的metadata, 这里需要考虑一个问题就是前面我们针对一些较大的Tile需要2个warp group防止RegisterSpill, 因此这里在从tiled_mma中获取thr_mma时, 需要做一个额外的处理

```
        warp_group_idx = cute.arch.make_warp_uniform(            tidx // self.num_threads_per_warp_group        )        # 其中self.mma_warp_groups = math.prod(self.atom_layout_mnk)        warp_group_thread_layout = cute.make_layout(            self.mma_warp_groups, stride=self.num_threads_per_warp_group        )                thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))        tCgC = thr_mma.partition_C(gC_mnl)        tCsA = thr_mma.partition_A(sA)        tCsB = thr_mma.partition_B(sB)        tCrA = tiled_mma.make_fragment_A(tCsA)        tCrB = tiled_mma.make_fragment_B(tCsB)        acc_shape = tCgC.shape        accumulators = cute.make_fragment(acc_shape, self.acc_dtype)        
```

接下来, 我们还要对TMA操作所需要的metadata进行处理, 主要是拷贝操作的源和目的的partition

```
        #  TMA load A partition_S/D        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)        a_cta_crd = cluster_coord_mnk[1]        sA_for_tma_partition = cute.group_modes(sA, 0, 2)        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(            tma_atom_a,            a_cta_crd,            a_cta_layout,            sA_for_tma_partition,            gA_for_tma_partition,        )        # TMA load B partition_S/D        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)        b_cta_crd = cluster_coord_mnk[0]        sB_for_tma_partition = cute.group_modes(sB, 0, 2)        gB_for_tma_partition = cute.group_modes(gB_nkl, 0, 2)        tBsB, tBgB_nkl = cute.nvgpu.cpasync.tma_partition(            tma_atom_b,            b_cta_crd,            b_cta_layout,            sB_for_tma_partition,            gB_for_tma_partition,        )
```

完成操作后, 有一个cluster level的同步, 确保集群内所有 CTA 都完成了流水线和屏障的初始化后, 才一起进入主循环.

```
        # cluster wait for barrier init        if cute.size(self.cluster_shape_mn) > 1:            cute.arch.cluster_wait()        else:            cute.arch.sync_threads()
```

#### 3.2.3 第三阶段: Prologue

下面我们进入了最核心的计算流水线了. 首先我们需要issue 一批TMA copy,把数据加载到SMEM中. 同时也需要对Consumer进行初始化, 等待smem_full并issue WGMMA.

![图片](assets/dcf63649da2f.png)

首先计算总共需要处理的K-Tile的数量, 并和SMEM中的流水线级数(ab_stage)比较, 获得需要prefetch的数据数量

```
        k_tile_cnt = cute.size(gA_mkl, mode=[2])        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)
```

然后在这段example code中, 并没有使用`PipelineProducer`和`PipelineConsumer`类, 而是直接通过make_pipeline_state暴露了状态. 总共有三个状态, 在Prologue阶段大概的流程如下, 需要注意的是在Consumer侧创建两个独立的流水线状态机. 一个用于追踪读取 (wait, consume), 另一个用于追踪释放 (release).

```
# 创建Statemainloop_producer_state = pipeline.make_pipeline_state(    pipeline.PipelineUserType.Producer, self.ab_stage)mainloop_consumer_read_state = pipeline.make_pipeline_state(    pipeline.PipelineUserType.Consumer, self.ab_stage)mainloop_consumer_release_state = pipeline.make_pipeline_state(    pipeline.PipelineUserType.Consumer, self.ab_stage)# 生产者循环for i in range(...):    pipeline.producer_acquire(producer_state) # 等待    # 调用TMA 在 producer_state.index 指向的缓冲区里写数据    pipeline.producer_commit(producer_state) # 提交, 实际上是一个空指令, 真的完成有TMA tx-byte cnt硬件更新    producer_state.advance() # 手动推进状态# 消费者循环for i in range(...):    pipeline.consumer_wait(mainloop_consumer_read_state) # 等待数据Ready    # 基于mainloop_consumer_read_state.index 指向的缓冲区数据    # 提交WGMMA指令, 并执行wg.commit_group    mainloop_consumer_read_state.advance() # 手动推进Read状态
```

在 Prologue 中, 我们只进行 read, 不进行 release, 因为我们希望 SMEM 缓冲区保持被占用的状态, 直到主循环中对应的 release 操作发生. 如果在这里就 release, 会打乱流水线的节奏. 这两个状态机的分离是实现这种延迟释放的关键.

即`mainloop_consumer_release_state`状态会在Mainloop中, Consumer的wg.wait_group确定WGMMA异步操作完成后, 才会调用consumer_release(mainloop_consumer_release_state)操做, 并同时推进mainloop_consumer_read_state 和 mainloop_consumer_release_state.
Prologue TMA
如下所示:

```
        # producer warp        if warp_idx == 0:             # 针对需要进行Prefetch的K-tile数量, 构建循环            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):                # 等待A/B buffer为空                mainloop_pipeline.producer_acquire(mainloop_producer_state)                                # 根据当前的state状态, 设置TMA需要拷贝的GMEM和SMEM切片                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]                tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]                # 执行TMA copy                cute.copy(                    tma_atom_a,                    tAgA_k,                    tAsA_pipe,                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(                        mainloop_producer_state                    ),                    mcast_mask=a_mcast_mask,                )                cute.copy(                    tma_atom_b,                    tBgB_k,                    tBsB_pipe,                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(                        mainloop_producer_state                    ),                    mcast_mask=b_mcast_mask,                )                                # 下面这个commit在PipelineTmaAsync中是一个空操作, 真正的commit由TMA的tx-bytes计数器更新完成.                mainloop_pipeline.producer_commit(mainloop_producer_state)                mainloop_producer_state.advance()  # 向前推进
```
Prologue MMA
代码如下所示, 首先我们先解释一下k_pipe_mmas, 这个变量定义了 Prologue 阶段要执行多少轮 MMA. 在这里, 它被硬编码为 1. 这意味着 Prologue 会消耗掉 Prefetch 准备好的第一个数据块. k_pipe_mmas 在通常与 WGMMA 的 wait_group 机制有关, 代表了 WGMMA 指令流水线的深度, 即可以连续提交多少批 WGMMA 指令而无需等待它们完成. 这里设为 1 是一种相对简单和保守的策略.

```
        k_pipe_mmas = 1        peek_ab_full_status = cutlass.Boolean(1)        if mainloop_consumer_read_state.count < k_tile_cnt:            # consumer_try_wait 是一个非阻塞的屏障检查. 它会立即返回一个布尔值, 表示第一个数据块 (Stage 0) 是否已经准备好.            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(                mainloop_consumer_read_state            )        # 前面这段代码是表示在整个循环前先看一眼是不是full, 避免在后面的consumer_wait中傻等        # 因为在consumer_wait循环阻塞中, 如果提前知道这个status是ready的, 可以节省几个cycle.        # 第一轮计算的时候Tile MMA是非累加模式        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)        num_k_blocks = cute.size(tCrA, mode=[2])                # 这个循环只执行一次 (因为 k_pipe_mmas = 1)        for k_tile in cutlass.range_constexpr(k_pipe_mmas):                        # 等待A/B buffer Ready            mainloop_pipeline.consumer_wait(                mainloop_consumer_read_state, peek_ab_full_status            )            # 插入一个fence, 确保 TMA 对 SMEM 的写入对 WGMMA 单元是可见的            cute.nvgpu.warpgroup.fence()            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):                k_block_coord = (                    None,                    None,                    k_block_idx,                    mainloop_consumer_read_state.index,                )                tCrA_1phase = tCrA[k_block_coord]                tCrB_1phase = tCrB[k_block_coord]                cute.gemm(                    tiled_mma,                    accumulators,                    tCrA_1phase,                    tCrB_1phase,                    accumulators,                )                # 第一轮Tile MMA提交后, 针对后面的轮次, 将累加模式改为True                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)            # 提交WGMMA commit_group()            cute.nvgpu.warpgroup.commit_group()                        # 推进mainloop_consumer_read_state.             mainloop_consumer_read_state.advance()                        # 然后继续进入下一个iter前看一眼mainloop_consumer_read_state是否ready.            peek_ab_full_status = cutlass.Boolean(1)            if mainloop_consumer_read_state.count < k_tile_cnt:                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(                    mainloop_consumer_read_state                )
```

#### 3.2.4 第四阶段: MainLoop

MainLoop是整个 Kernel 中最核心、执行时间最长的部分. 在Prologue建立好流水后, MainLoop接管了流水线的稳定运行阶段, 在 K 维度的每一次迭代中, 同时执行三项任务:

**计算 (Consume)**: 使用 WGMMA 计算当前已准备好的 A 和 B 数据块, 并执行WGMMA commit_group

**释放 (Release)**: 通过WGMMA waitgroup等待上一轮WGMMA运算完成, 然后释放掉上上一轮计算使用过的 SMEM 缓冲区.

**加载 (Produce)**: 在刚被释放的缓冲区中, 异步加载下一批需要计算的 A 和 B 数据块.

整个MainLoop结构如下

```
for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):    ###########################          # Consumer                #     ###########################    # 1. 等待数据就绪    mainloop_pipeline.consumer_wait(        mainloop_consumer_read_state, peek_ab_full_status    )    # 2. Ready后根据mainloop_consumer_read_state.index执行WGMMA, 并commit_group        # 3. 等待之前的WGMMA完成计算    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)        # 4. 释放使用完毕的缓冲区    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)        # 5. 推进状态机    mainloop_consumer_read_state.advance()    mainloop_consumer_release_state.advance()        # 6. 尝试提前检查下下个缓冲区状态, 为下一轮循环做优化    peek_ab_full_status = cutlass.Boolean(1)    if mainloop_consumer_read_state.count < k_tile_cnt:        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(            mainloop_consumer_read_state        )            ###########################          # Producer(仅由Warp 0执行)  #     ###########################    if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:        # 1. 申请空闲缓冲区         mainloop_pipeline.producer_acquire(mainloop_producer_state)                # 2. 发起异步TMA加载, 注意对于GMEM采用count, 而SMEM因为是一个环形缓冲区使用index        tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]        tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]                cute.copy(            tma_atom_a,            tAgA_k,            tAsA_pipe,            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(                mainloop_producer_state            ),            mcast_mask=a_mcast_mask,        )        # 对B做同样的TMA拷贝                # 3. 推进状态机        mainloop_pipeline.producer_commit(mainloop_producer_state) # 这是一个空操作(NOP)        mainloop_producer_state.advance()        
```

对于整个状态机, 假设 ab_stage=3, k_pipe_mmas=1.

![图片](assets/3de53d8a455c.png)

#### 3.2.5 第五阶段: Epilogue

`EPILOG` 部分是 GEMM Kernel 的最后阶段. 当 `MAINLOOP` 完成了所有的 K 维度迭代后, 最终的计算结果  已经累加在每个 Warp Group 的私有累加器寄存器 (`accumulators`)中了.

`EPILOG` 的任务就是将这些分布在几千个寄存器中的结果, 安全、高效地写回到GMEM中正确的位置. 当然在某些计算场景中Epilogue还要负责ReLU/softmax等计算. 这里我们以最简单的情况为例, 这个过程通常分为两个主要步骤:

**RMEM -> SMEM (R2S)**: 例如将累加器中的数据 (通常是 FP32) 转换成目标类型 (如 BP16), 并存储到 SMEM 中.

**SMEM -> GMEM (S2G)**: 使用 TMA 将 SMEM 中的结果数据块异步写回到 GMEM.

```
# /////////////////////////////////////////////////////////////////////////////#  EPILOG# /////////////////////////////////////////////////////////////////////////////# 1. 确保所有在 MAINLOOP 中提交的 WGMMA 计算都已经完成, 累加器中的值是最终结果.cute.nvgpu.warpgroup.wait_group(0)# 这一段等待同步非常重要, 它必须要确保CGA/CTA中所有的线程都完成了计算.# 因为在Epilogue阶段, 会复用MAINLoop中为A/B矩阵分配的SMEM空间来存储C矩阵. # 如果不等所有线程完成, 会导致数据污染.if cute.size(self.cluster_shape_mn) > 1:    cute.arch.cluster_arrive()    cute.arch.cluster_wait()else:    cute.arch.sync_threads()# 2. 定义Copy-Atom和 TiledCopy, 这里R2S使用了StMatrix指令存储8x8x16b的数据到SMEMcopy_atom_r2s = sm90_utils.sm90_get_smem_store_op(...)copy_atom_C = cute.make_copy_atom(...)tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)# 3. 内存分区# thr_copy_r2s: 从 Tiled-Copy 中为当前线程 (tidx) 切分出它自己的任务.thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)# tRS_sD: 对目标 SMEM (sC) 进行分区, 得到当前线程负责写入的 SMEM 视图.注意 sC 是一个带有多级缓冲 (pipelined) 的 SMEM 区域.tRS_sD = thr_copy_r2s.partition_D(sC)# retile 并没有移动数据. 它只是在逻辑上将 accumulators (其布局由 WGMMA 决定) 重新解释成一种新的布局, 这种新布局适合于 Epilogue 的分块拷贝.tRS_rAcc = tiled_copy_r2s.retile(accumulators)# 这段代码在RMEM中分配了一个临时的、小型的中转缓冲区 tRS_rD, # 用于暂存从庞大的累加器 (accumulators) 中拷贝出来的一小块数据, 以便进行后续的类型转换和重排 (reshuffle).# accumulators 的布局由WGMMA决定, 而Epilogue Store 的布局, 由于需要写入SMEM, 通常依赖于stmatrix这样的指令. 两者布局由不同需要解耦.rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))tRS_rD_layout = cute.make_layout(rD_shape[:3])tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.acc_dtype)size_tRS_rD = cute.size(tRS_rD)# 4. 针对SMEM->GMEM的TMA进行处理# 描述了src, 即待拷贝数据在SMEM中的Layout.# 这个函数将 sC 布局的Mode 0 和Mode 1 合并成一个新的模式. 便于后续的拷贝.#  - 转换前: Layout<(M, N, Stage), ...>#  - 转换后: Layout<(M * N, Stage), ...>sepi_for_tma_partition = cute.group_modes(sC, 0, 2)# 描述了dst, 即数据最终要放入GMEM中的LayouttCgC_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)# bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(    tma_atom_c,    0,    cute.make_layout(1),    sepi_for_tma_partition, # 源: (Tile, Stage)    tCgC_for_tma_partition, # 目标: (TileGrid, TileShape))# 5. 异步pipeline采用了PipelineTmaStore类c_producer_group = pipeline.CooperativeGroup(    pipeline.Agent.Thread, self.threads_per_cta, self.threads_per_cta)c_pipeline = pipeline.PipelineTmaStore.create(...)# 6. 循环: R2S 和 S2Gfor epi_idx in cutlass.range_constexpr(epi_tile_num):    # 6a. R2S - Part 1: Accumulator -> Registers    for epi_v in cutlass.range_constexpr(size_tRS_rD):        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]    # 5b. R2S - Part 2: 类型转换    tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self.c_dtype)    acc_vec = tRS_rD.load()    tRS_rD_out.store(acc_vec.to(self.c_dtype))    # 6c. R2S - Part 3: Registers -> Shared Memory    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])    cute.copy(        tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]    )    # 6d. 同步 R2S    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, ...)    cute.arch.barrier()    # 6e. S2G - 由 Warp 0 发起TMA    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)    if warp_idx == 0:        cute.copy(            tma_atom_c,            bSG_sD[(None, epi_buffer)],            bSG_gD[(None, gmem_coord)],        )        c_pipeline.producer_commit()        c_pipeline.producer_acquire()    # 6f. 同步 S2G    cute.arch.barrier()# 7. 流水线收尾if warp_idx == 0:    #  等待最后一次提交的 TMA 存储操作完成.    c_pipeline.producer_tail()
```

参考资料

[1] 
Hopper/DenseGemm: *https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py*
[2] 
CUDA DMA: *https://d1qx31qr3h6wln.cloudfront.net/publications/SC_2011_CUDA_DMA.pdf*
[3] 
Singe编译器: *https://cs.stanford.edu/~sjt/pubs/ppopp14.pdf*
[4] 
Unweaving Warp Specialization: *https://rohany.github.io/blog/warp-specialization/*
[5] 
15-779 Lecture 6:Advanced CUDA Programming:Warp Specialization: *https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/06-warp-specialization.pdf*
[6] 
GPGPU Arch(二) —— 漫谈 Hopper WarpSpecialization Pingpong/Cooperative 设计: *https://zhuanlan.zhihu.com/p/1929932276499722808*
[7] 
WASP: Exploiting GPU Pipeline Parallelism with Hardware-Accelerated Automatic Warp Specialization: *https://www.nealcrago.com/wp-content/uploads/WASP_HPCA2024_preprint.pdf*
[8] 
Targeting NVIDIA Hopper in MLIR: *https://llvm.org/devmtg/2024-03/slides/nvidia-hopper-in-mlir.pdf*
[9] 
TMA Patent: *https://patents.google.com/patent/US20230289292A1/*
[10] 
GPU 计算与编程模型演进：异步计算编程中的吞吐与延迟平衡: *https://www.bilibili.com/video/BV11tMwznEmo/*
[11] 
include/cute/swizzle.hpp: *https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp*
[12] 
make_trivial_tiled_mma: *https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/hopper_helpers.py[#L101](javascript:;)*
[13] 
Cutlass Pipeline: *https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/pipeline/sm90.py*