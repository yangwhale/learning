# Tensor-103.4 Blackwell GEMM

> 作者: zartbot  
> 日期: 2025年11月1日 09:56  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496700&idx=1&sn=bac68c7f30ef8b6ce7ae550963580ecb&chksm=f995e33ecee26a28a9ddff4afaa729667b6ebc797590a43dfee00edc4052c8d83ac03567701a#rd

---

### TL;DR

前一篇介绍了Hopper的GEMM, 比较关键的一点是在Hopper中引入的TMA以及关于MBarrier的处理, 但是从架构上而言, Hopper还是有不少问题的, 例如TensorCore需要整个warpgroup issue MMA指令, 并且同步等待. TMA/MMA/Epilogue等warp的调度也需要相对复杂的处理. 这些问题也是Blackwell改进的地方, 通过引入TMEM使得整个TensorCore不再占用RMEM, 并实现了和TMA一样的完全异步的处理, 只需要一个线程issue指令即可. 然后演进到了双Die的架构, 并且增加了互连的带宽, 支持了NVLinkC2C可以和Grace(未来还有Intel的CPU)直接互连, 同时也扩大了ScaleUP的规模, 支持NVL72.

当然Blackwell也有不少的问题, 我们将在后面的一篇文章详细展开.

本文目录如下:

```
1. Hopper GEMM的问题
  1.1 TensorCore
  1.2 CGA
  1.3 Static Tile Scheduling
2. Blackwell 软硬件功能演进概述
  2.1 Blackwell TensorCore
  2.2 Blackwell内存层次结构
    2.2.1 L2 Cache
    2.2.2 TMEM
  2.3 Preferred Thread Block Clusters
  2.4 Dynamic Tile Scheduling
3. Blackwell异步处理
  3.1 从一个GEMM例子谈起
  3.2 Blackwell Pipeline
    3.2.1 PipelineTmaUmma
    3.2.2 PipelineUmmaAsync
4. Simple GEMM Example
  4.1 Overview
  4.2 SharedStorage 结构体
  4.3 Host侧函数
  4.4 Kernel
    4.4.1 初始化
      4.4.1.1 设置流水线
      4.4.1.2 Tensor Partitioning
      4.4.1.3 Epilogue TMEM copy
    4.4.2 Main loop
    4.4.3 Epilogue
5. GEMM Persistent Kernel
  5.1 Overview
  5.2 初始化参数
  5.3 Host端函数
  5.4 Kernel函数
    5.4.1 初始化阶段
    5.4.2 TMA Warp
    5.4.3 UMMA Warp
    5.4.4 Epilogue Warp
  5.5 性能对比
```

相关的测试结果如下, M,N,K=4096, A/B BF16 Acc FP32. 需要注意的是, 2-CTA UMMA对性能提升至关重要

| 算法实现 | Jetson Thor(TFLOPS) |
| --- | --- |
| CublasLt | 87.50 |
| CuteDSL-1CTA | 53.86 |
| CuteDSL-2CTA | 81.45 |

## 1. Hopper GEMM的问题

### 1.1 TensorCore

在Hopper中, 实现TMA和TensorCore配合的流程如下:

![图片](assets/896de38bde49.png)

简单来说, TMA实现了完全的硬件Offload, Producer准备好描述符并且issue TMA指令后就不用管了, TMA会自动去update MBarrier并翻转Phase bit. 但是TensorCore还没有实现完全异步的操作. 因为Accumulate的结果存在于寄存器中, 因此还是一个同步等待的过程.

另一方面我们注意到了在Hopper上存在大量的寄存器占用的问题. 因此需要`setmaxnreg`将寄存器资源分配给consumer的MMA warp, 并降低TMA Warp的寄存器数量, 另一方面由于单个WarpGroup运算会导致RegisterSpill,因此需要配置MMA_ATOM中的atom_layout使用两个warp group进行运算.

当进一步增加TensorCore的吞吐时, 将会占用更多的寄存器. 另一方面MMA计算(生产者)和后续的尾声(Epilogue)操作(消费者, 如类型转换, 存回全局内存等)被紧密耦合在寄存器上, 不利于流水线深度的优化. 而且要求在两个MMA Group之间overlap Epilogue计算.

![图片](assets/c0a3b9b6b42f.png)

### 1.2 CGA

另一方面, Hopper引入了CGA(aka Thread Block Cluster), 但是要求CGA内的所有Thread Block必须处于一个GPC的范围.

这里稍微再补充一下为什么Nvidia会有CGA / CTA / Thread Block Cluster / GPC / TPC这么多层次化的结构描述的名词.

在[《GPU架构演化史》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=2538479717163761664&scene=173&from_msgid=2247487954&from_itemidx=3&count=3&nolastread=1#wechat_redirect)中有可以看到GPC和TPC的由来. 在2006年, Nvidia推出了G80统一着色器架构的GPU, 也就是CUDA计算的开端.  在此之前, GPU拥有独立的顶点着色器和像素着色器管线. G80将它们统一为可编程的流式处理器(SP).在G80中, TPC (Texture Processing Cluster) 首次作为核心组织单元出现. 当时, 每个TPC包含 2个SM. 每个SM包含8个流处理器(SP, 即CUDA Core的前身).

在2010年设计Fermi架构时, Fermi的设计者们认识到, 要想更容易地扩展GPU规模, 就必须采用更加模块化的设计. 因此引入了GPC (Graphics Processing Cluster)的层次化结构, 一个完整的Fermi GF100芯片由4个GPC构成, 每个GPC包含一个独立的Raster Engine, 然后每个SM包含一个PolyMorph Engine. 这一代TPC的概念被弱化了. 但是基于GPC的架构, 老黄出类拔萃的刀法就诞生, 高端型号可以拥有完整的4个GPC, 中低端型号只有3个或2个GPC.

在2016年发布的Pascal架构中, 基本上形成了完善的GPU-->GPC-->TPC-->SM的层次化结构. 1个GPC包含了5个TPC, 整个GPC内有一个共享的Raster Engine, 每个TPC拥有一个完整的PolyMorph Engine.  当然这一代开始GP104用于图形业务, 每个TPC只有一个SM,  GP104包含完整的Raster Engine和PolyMorph Engine. 而对应于计算业务的GP100则去除了这些图形相关的加速器, 每个TPC包含了2个SM. 在硬件上基于GPC/TPC这样的层次化结构描述也就定型了.

而CTA(Cooperative Thread Array)也被称为Thread Block, CGA(Cooperative Grid Array)也被称为Thread Block Cluster更多的是从软件的视角在描述线程的组织结构.

在Hopper中总共有8个GPC, 每个GPC包含9个TPC, 每个TPC包含2个SM.

![图片](assets/07d7a0e9128d.png)

在SM90中支持一个Cluster最多8个SM, 在SM90a中支持最大的ClusterSize为16个, 如果我们用更大的ClusterSize, 借助于TMA multicast访问内存的效率会大大提高. 但是也容易导致一个Cluster内出现资源浪费的情况, 如下图所示, 假设我们有一个包含6个GPC, 每个GPC有6个SM的GPU. 如果我们采用更大尺寸的4x1的cluster, 则每个GPC会空闲2个SM无法被充分使用, 为了能够被充分使用, 又必须使用更小的2x1的Cluster, 但数据访问效率又受到了影响.

![图片](assets/094a8aa348f6.png)

### 1.3 Static Tile Scheduling

在前一篇[《Tensor-103.3 Hopper Persistent Kernel》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496608&idx=1&sn=1eac0c4b71e7c5251c1ca031cb8e744e&scene=21#wechat_redirect)介绍了Hopper Persistent Kernel的处理, 其中调度方式是按照Tile静态调度的, 当一个SM计算拖慢后, 后续需要这个SM多个Wave来处理剩余的Tile, 而无法分摊到其它SM导致负载不均引起的整体资源利用率降低.

![图片](assets/f2b3264e2861.png)

## 2. Blackwell 软硬件功能演进概述

### 2.1 Blackwell TensorCore

最主要的区别还是解决Hopper TensorCore占用RMEM的问题, 新增加了Tensor Memory, 并仅需要一个线程issue MMA指令即可. 而在Hopper中, 需要整个WarpGroup 4个SM同步并集体一起issue指令.

![图片](assets/0117ffdbe102.png)

因此从访问内存来看, 每个操作数矩阵可以存放的内存位置演进如下:

| Arch | Matrix A | Matrix B | Matrix D |
| --- | --- | --- | --- |
| Volta | RF | RF | RF |
| Ampere | RF | RF | RF |
| Hopper | RF/SMEM | SMEM | RF |
| Blackwell | TMEM/SMEM | SMEM | TMEM |

从TensorCore的演进来看, 一方面是提高同样数值精度的吞吐, 即翻倍了M的维度, 每个Warp的指令为 32xNx256bit, 整个WarpGroup在指令上被拼接成支持 128xNx256bit. 因此可以看到在做一个128 x N x 256bits的MMA操作时, Blackwell的性能实现了翻倍.

![图片](assets/ddf81526046a.png)

同时需要注意到在Blackwell上, 整个调用过程由于不依赖RMEM,变成了完全异步的方式, 并且由硬件更新Mbarrier

![图片](assets/29090f4114ff.png)

这样Epilogue可以通过独立的Warp运行, 并不需要像Hopper那样两个Ping-pong的warp来overlap Epilogue计算.

![图片](assets/6043e698f2e5.png)

具体关于tcgen05的异步操作将在后面单独一节展开, 因为tcgen05的指令还是很复杂的.

另一方面则是提供更丰富的低精度类型, 当然也受到DieSize的约束和TMEM带来的面积占用影响, 在B300中砍掉了不常用的算力, 如下表所示:

| Arch | FP64 | FP16 | INT8 | INT4 | FP8 | MXFP |
| --- | --- | --- | --- | --- | --- | --- |
| Volta | ❌ | ✅ FP16 | ❌ | ❌ | ❌ | ❌ |
| Turing | ❌ | ✅ FP16 | ✅ | ✅ | ❌ | ❌ |
| Ampere | ✅ | ✅ FP16/BF16 | ✅ | ✅ | ❌ | ❌ |
| Hopper | ✅ | ✅ FP16/BF16 | ✅ | ❌ | ⚠️FP8/FP22 | ❌ |
| Blackwell | ✅ | ✅ FP16/BF16 | ✅ | ❌ | ✅ | ✅ MXFP(8/6/4) NVFP4 |
| Blackwell Ultra | ⚠️砍算力 | ✅ FP16/BF16 | ⚠️砍算力 | ❌ | ✅ | ✅ MXFP(8/6/4) NVFP4 |

并且支持了Block Scaling, Scale Factor也可以存放在Tensor Memory中在执行运算的时候被TensorCore读取

![图片](assets/da0479d3d6c8.png)

另外针对TensorCore卷积运算在读取SMEM中的权重矩阵时, 还有一小块Cache用于实现weight-stationary. 并且还有一个64bits的zero-column-mask-desc描述符, 掩码中的 0 位表示矩阵 B 中对应列的值应用于 MMA 运算, 掩码中的 1 位表示 MMA 运算中该列的值必须全部为 0.

最后, 在TensorCore执行时, 指令可以同时控制2个SM. 将两个CTA构成一个Pair, 由Leader CTA的一个线程issue MMA指令, 而A/B/accumulator分布在两个SM上. 进一步增加数据复用的能力.

![图片](assets/8a9e1cd2200e.png)

### 2.2 Blackwell内存层次结构

#### 2.2.1 L2 Cache

在Hopper中L2Cache由2块25MB构成, L2 Partition会带来访问远端L2D延迟的显著增加约200个cycle. 在Blackwell上, 由于B200需要两个Die拼接, 跨Die的内存访问就会引入显著的延迟, 如果Die内还进行L2 Partition则会导致内存访问出现更复杂的问题. 因此在Blackwell上单个Die只有一块65MB的L2Cache. 但即便如此, 跨Die的内存访问依旧会增加大约200ns的延迟导致一些Kernel性能受到影响, 预计后面的CUDA版本会逐步引入一些CTA内存亲和性调度的能力.

#### 2.2.2 TMEM

在Blackwell中最大的变化就是增加了Tensor Memor, 用于解决Hopper中RMEM占用的问题. 它是一个2D的memory寻址架构,每个CTA包含512列和128行, 每个Cell是32bit. 每个Lane 2KB, 地址采用32bits Lane<31:16> Column<15:0>的方式.

![图片](assets/e7e74a1dc020.png)

但是也限制了访问, WarpGroup内不同的warp只能访问固定的Lane, 即每个TensorCore有32lanes 64KB的TMEM空间.

在内存管理上, TMEM需要显式的进行alloc/dealloc处理, 这样从软件抽象的角度来看, 就可以把它当成一个类似于Cache的用法, 在循环过程中可控的根据数据依赖和生命周期去做分配.

数据路径上, TMEM支持通过ld/st命令和RMEM进行数据加载和存储, 而在TMEM和SMEM之间的访问时仅支持SMEM到TMEM的copy. 但是需要注意的是这些数据移动需要按照一个预先定义好的Layout按块处理. 因此Epilogue流程中应将Accumulator的结果拷贝到RMEM中, 再由RMEM回刷到GMEM.

L2Cache没有Partition, 新增加Tensor Memory以及TensorCore的规模扩大也带来了很大的面积占用, 另一方面GPC的规模扩展到了20个SM. 因此在Blackwell中单个Die的SM数量降低到了80个, 这也产生了一些问题, 例如在B200上SFU的性能并没有得到显著的提升, 使得softmax这些需要指数运算的性能出现瓶颈. 直到Blackwell Ultra(B300)才进一步通过砍FP64算力将SFU的性能翻倍来解决.

#### 2.2.3 TMA

在Blackwell上TMA也做了一定的扩展, PTX文档中显示新增了tile::scatter4 / tile::gather4 以及 im2col_w的支持. 另外在《Bringing NVIDIA Blackwell GPU support to LLVM and MLIR》[1] 中还提到了支持 Masked Copy能力, 但PTX里面尚未描述. 不过我在想是否这个是为B200/B300 双Die结构可以做一些亲和性的内存访问来使用? 保证调度的CTA通过mask copy TMA避免跨Die的读取?

### 2.3 Preferred Thread Block Clusters

在Hopper中launch size > 2 的CGA会出现SM资源浪费的情况. 在Blackwell中一个GPC有20个SM正好是4的倍数,可以launch更大的CGA提升效率, 同时它也增加了一个新的功能, 在LaunchKernel的时候, 可以选择cluster包含2个Shape. 在SM无法放下时, 可以用更小的 cluster_size 调度进GPC

![图片](assets/0b4b8a4d7b1f.png)

### 2.4 Dynamic Tile Scheduling

在前面一篇文章中介绍了Hopper的静态Tile调度, 如果某些 SM 的资源不可用, 静态调度器容易出现工作负载不均衡的问题.

![图片](assets/4348822d5890.png)

Persistent Kernel有一个根本性的限制, 在启动时无法实时知道究竟有多少个SM可供使用, 某些SM可能被其它Kernel占用, 导致了负载不均衡的问题. 在Blackwell中引入了Cluster Launch Control(CLC)来支持动态调度.

![图片](assets/7b02597bb65c.png)

有了CLC, Kernel可以像non-persistent kernel那样根据输出的Tile数量构建一个数量相等的Grid, Grid中的坐标定义为ClcID. CLC遵循以下规则:

当有可用硬件资源(如空闲的SM)时, 一个待处理的ClcID会被硬件调度器自动启动, 成为一个新的worker.

一个已经存在的worker可以通过clusterlaunchcontrol.try_cancel指令来查询一个待处理的ClcID, 并接管其工作.

系统保证每一个ClcID要么被规则(1)启动, 要么被规则(2)接管, 不会丢失.

每个worker使用其{blockIdx.x, blockIdx.y, blockIdx.z}坐标作为第一个要处理的输出tile. 之后, 它通过CLC查询来获取后续要处理的tile.

clusterlaunchcontrol.try_cancel指令会返回一个成功信号并附带一个ClcID, 或者一个拒绝信号. 最常见的拒绝原因是所有的ClcID都已经被处理完毕.

CLC的工作粒度是CGA. 例如, 一个2x2的persistent worker cluster(由4个CTA组成)的一次查询会同时消耗4个ClcID.

如下图展示:

![图片](assets/2bb218108dfa.png)

仅有的80个worker被启动. 当一个worker完成其工作后, 它会立即通过CLC请求新的工作, 直到所有400个tile都被处理完毕. 这样, 工作负载被动态地均衡分配到所有可用的SM上.

具体的执行过程如下:

```
  // Persistent loop  do {    // Producer    if (is_producer) {      // Only 1 thread of the entire cluster issues the query.      uint32_t mbarrier_addr = scheduler_pipeline.producer_get_barrier(scheduler_pipe_state_write);      // Wait for clcID buffer to become empty with a flipped phase      scheduler_pipeline.producer_acquire(scheduler_pipe_state_write);      // 异步查询CLC      if (cute::elect_one_sync()) {        Scheduler::issue_clc_query(scheduler_pipe_state_write, mbarrier_addr, shared_storage.clc_response);      }      ++scheduler_pipe_state_write;    }    // Consumers    if (is_consumer) {      int linearCLC = work_tile_info.N_idx * gridDim.x + work_tile_info.M_idx;      // Atomically increment the worker count for the linearCLC by 1.      if (lane_predicate) {        atomicAdd(&d_workerCount[linearCLC], 1);      }    }    // Union of all consumers. Note that the producer here is its own consumer.    if (is_producer || is_consumer) {      scheduler_pipeline.consumer_wait(scheduler_pipe_state);      uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&shared_storage.clc_response[scheduler_pipe_state.index()]);      // 获取work_tile_info      work_tile_info = Scheduler::work_tile_info_from_clc_response(smem_addr);      scheduler_pipeline.consumer_release(scheduler_pipe_state);      ++scheduler_pipe_state;      // Add block offset since the scheduler works at cluster level.       dim3 block_id_in_cluster = cute::block_id_in_cluster();      work_tile_info.M_idx += block_id_in_cluster.x;      work_tile_info.N_idx += block_id_in_cluster.y;      work_tile_info.L_idx += block_id_in_cluster.z;    }  } while (work_tile_info.is_valid_tile);
```

CLC的查询和获取work_tile_info的函数锁调用的指令如下:

```
  CUTLASS_HOST_DEVICE  static void  issue_clc_query(PipelineState<Stages> state, uint32_t mbarrier_addr, CLCResponse* clc_response_ptr) {  #if defined(CUTLASS_ARCH_CLC_ENABLED)      uint32_t result_addr = cute::cast_smem_ptr_to_uint(reinterpret_cast<const void*>(            &clc_response_ptr[state.index()]));      asm volatile(        "{\n\t"        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"         "}\n"        :        : "r"(result_addr), "r"(mbarrier_addr));  #else      CUTLASS_NOT_IMPLEMENTED();  #endif  }  CUTLASS_DEVICE  static WorkTileInfo  work_tile_info_from_clc_response(uint32_t result_addr) {    WorkTileInfo work_tile_info;    uint32_t valid = 0;    #if defined(CUTLASS_ARCH_CLC_ENABLED)      asm volatile(        "{\n"        ".reg .pred p1;\n\t"        ".reg .b128 clc_result;\n\t"        "ld.shared.b128 clc_result, [%4];\n\t"        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"        "selp.u32 %3, 1, 0, p1;\n\t"        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_result;\n\t"        "}\n"        : "=r"(work_tile_info.M_idx), "=r"(work_tile_info.N_idx), "=r"(work_tile_info.L_idx), "=r"(valid)        : "r"(result_addr)        : "memory"      );      cutlass::arch::fence_view_async_shared();    #else      CUTLASS_NOT_IMPLEMENTED();    #endif    work_tile_info.is_valid_tile = (valid == 1);    return 
```

## 3. Blackwell异步处理

在blackwell中, TensorCore已经变成了一个相对独立的DSA了. 从指令集上来看, 有同步指令也有异步指令

| tcgen05.* operation | |
| --- | --- |
| 同步指令 | `.alloc` `.dealloc` `.relinquish_alloc_permit` `.fence::*` `.wait::*` `.commit` |
| 异步指令 | `.mma` `.cp` `.shift` `.ld` `.st` |

另一方面issue指令的粒度(Granularity)也是不同的.

![图片](assets/266d02f31709.png)

因此这一章对整个交互过程做一个详细的分析.

### 3.1 从一个GEMM例子谈起

这一节主要从一个最简单的流水线操作来看各种异步编程下的Mbarrier是如何处理的. 首先需要初始化两个MBarrier(MB0和MB1)

![图片](assets/971591323d71.png)

然后warp0执行TMEM内存分配, 分配规则按照需要占用多少个Column定, 例如这里分配了128个Column. 由于tcgen05.alloc是一个warp-level的同步指令, 然后这里有一个barrier需要阻塞. 然后下一步只需要warp0的一个线程执行TMA load加载A和B到SMEM. 并且需要等待 MB0 完成.

![图片](assets/23a61a343245.png)

当数据已经加载到SMEM后, Warp0的一个线程就可以issue tcgen05.mma指令了.这个指令也是一个线程粒度的异步操作. 然后将这些异步的操作使用tcgen05.commit_arrive(MB1)来追踪完成情况.

![图片](assets/31a121182d67.png)

最后采用一个MB1.try_wait同步调用等待完成. 当计算完成后, 所有的结果已经写到了TMEM中. 此时需要warp-level的tcgen05.ld将数据搬运到RMEM中, 这也是一个异步操作,然后需要执行tcgen05.wait::ld等到完成后, 线程可以将RMEM的数据保存到GMEM中

![图片](assets/249d3888609e.png)

最后完成TMEM的释放tcgen05.dealloc和relinquish_alloc_permit释放allocation锁.

![图片](assets/6b71a3b37dc3.png)

总体来说, 当需要从TMEM 执行内存分配和LD/ST到RMEM时, 他们都是warp-level的操作, 针对一个预先设定的Layout从这个warp对应的TMEM 32lane中读取一个Block(包含 N个 Column), 涉及内存管理的事同步操作, 而LD/ST是异步执行. 等待TMEM LD/ST完成的tcgen05.wait::ld/st 则是一个warp-level的同步阻塞.

针对tcgen05.mma/.cp/.shift操作, 通常是线程粒度的异步操作, 然后将一系列inflight的指令打包进行tcgen05.commit挂载到一个MBarrier上, 待指令执行完成后硬件更新MBarrier. 软件采用mbarrier.try_wait()执行线程粒度的同步阻塞.

### 3.2 Blackwell Pipeline

针对复杂的异步操作, 在CuteDSL中封装了PipelineTmaUmma, PipelineUmmaAsync[2]两个类分别用于 Producer TMA--> Consumer UMMA, 以及Producer UMMA --> Consumer cp.async. 另外还有一个PipelineTmaMultiConsumersAsync[3] 类用于多个Consummer.

#### 3.2.1 PipelineTmaUmma

`PipelineTmaUmma` 是为Blackwell TMA和TensorCore定制的一个Software Pipeline同步的类, 正如类名, 它包含

**生产者 (Producer)**: TMA硬件单元. 它的任务是从HBM中高效地加载矩阵A和B的tile到SMEM中.

**消费者 (Consumer)**: UMMA指令, 由Tensor Core执行. 它的任务是从SMEM中读取A和B的tile, 进行矩阵乘法累加运算.

`PipelineTmaUmma`类的作用就是在这两者之间建立一个高效的同步机制. 它使用一个包含多个缓冲区(stages)的共享内存池, 并通过NVIDIA GPU的`mbarrier`硬件同步原语, 来确保:

UMMA不会去计算一个尚未由TMA填充完毕的缓冲区.

TMA不会去覆盖一个UMMA尚未计算完毕的缓冲区.

类定义如下:

```
@dataclass(frozen=True)class PipelineTmaUmma(PipelineAsync):    """    PipelineTmaUmma is used for TMA producers and UMMA consumers (e.g. Blackwell mainloops).    """    # 用于标识当前线程块(CTA)是否是其协作组(CTA Cluster)中的"领导者"    is_leader_cta: bool         # 枚举类型, 可能的值为`CtaGroup.ONE`或`CtaGroup.TWO`, 用于2SM的MMA    cta_group: cute.nvgpu.tcgen05.CtaGroup
```
create
这是一个静态工厂方法, 是创建 PipelineTmaUmma 实例的唯一入口. 核心流程如下:

**创建同步对象**: 创建两个 mbarrier 封装对象:

`sync_object_full`: 用于TMA通知UMMA "缓冲区已满, 可以计算".

`sync_object_empty`: 用于UMMA通知TMA "缓冲区已空, 可以加载新数据".

**配置事务屏障**: 在创建`sync_object_full`时, 传入了`tx_count`参数.TMA硬件在完成`tx_count`字节的数据传输后, 会自动地硬件更新这个MBarrier.

**计算掩码和角色**: 调用`_compute_mcast_arrival_mask`和`_compute_is_leader_cta`来确定每个CTA在同步过程中需要使用的掩码和扮演的角色.

**初始化等待**: `pipeline_init_wait`是一个同步点, 确保Cluster内所有CTA都完成了初始化, 然后才能开始流水线操作.
_compute_is_leader_cta(cta_layout_vmnk: cute.Layout)
计算当前CTA是否为领导者. 它将物理的线程块ID(`block_idx()`)映射到逻辑的集群坐标(`mma_coord_vmnk`), 然后检查在协作维度(`v`维度, 即坐标的第一个元素)上是否为0.
_compute_mcast_arrival_mask(cta_layout_vmnk: cute.Layout, mcast_mode_mn: tuple[int, int])
TMA支持多播(multicast)功能. 当TMA加载一个数据tile到SMEM时, 该SMEM可以被Cluster内的多个CTA共享和访问. 这样做可以极大地节省内存带宽, 因为一份数据服务于多个计算单元.

当TMA加载的数据准备好后, 它需要通知所有将要使用这份数据的消费者CTA. `mbarrier`的`arrive`操作需要一个掩码(bitmask)来精确指定通知的目标. 掩码的每一位对应Cluster中的一个CTA. `create_tma_multicast_mask` 它根据Cluster的布局, 当前CTA的坐标, 以及多播的维度(`mcast_mode=2`代表M维度, `mcast_mode=1`代表N维度)来生成一个基础掩码. 另外针对2SM场景, 通过`cta_in_cluster_coord_vmnk[0] ^ 1`获取peer坐标, 然后计算Peer的Mask.

最终的掩码是当前CTA和其伙伴CTA的掩码的并集. 这样可以确保当数据到达时, 协作对中的两个CTA都能收到`mbarrier`的通知信号.
producer_acquire(self, state: PipelineState, ...)
生产者(TMA warp)在准备向某个缓冲区加载数据之前调用,获取一个可用的空缓冲区. 首先在smem_empty上等待, 即`self.sync_object_empty.wait(state.index, state.phase)`. 如果UMMA还未释放这个缓冲区, producer线程会在此阻塞. 然后让Leader CTA在TMA指令发出前, 调用smem_full.arrive操作, arrive操作会更新arrive cnt, 但是真正的Pahse翻转需要TMA的Tx-bytes计数器完成. 另外调用arrive时也将multicast_mask设置, 硬件触发时, 就会使用这个预设的掩码通知所有正确的消费者.
producer_commit(self, state: PipelineState)
实际上是一个空指令,  "提交"这一动作已经完全由TMA硬件和事务屏障自动处理了. 软件层面不再需要任何显式的`commit`操作.
consumer_release(self, state: PipelineState)
消费者(UMMA warps)使用完一个缓冲区的数据之后,释放该缓冲区, 告知生产者它可以被再次使用. 具体实现是`self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)`,我们在MbarrierArray类中可以看到, 针对tcgen05.mma的arrive函数中增加了tcgen05.commit用于让TensorCore硬件更新这个mbarrier.

```
    def arrive_tcgen05mma(        self, index: int, mask: Optional[int], cta_group: cute.nvgpu.tcgen05.CtaGroup    ) -> None:        if mask is None:            with cute.arch.elect_one():                cute.nvgpu.tcgen05.commit(self.get_barrier(index))        else:            with cute.arch.elect_one():                cute.nvgpu.tcgen05.commit(self.get_barrier(index), mask, cta_group)
```

通过一个双缓冲(num_stages=2)的例子来完整地理解其工作流程:

**初始化**: 2个SMEM缓冲区 (buf0, buf1). empty_barrier[0]和empty_barrier[1]状态为"可使用", full_barrier[0]和full_barrier[1]状态为"等待".

**第1阶段 (Producer)**:

TMA线程调用`producer_acquire(buf0)`. empty_barrier[0]可使用, 因此不会阻塞

Leader CTA对full_barrier[0]执行arrive, 预设好通知掩码.

TMA线程发出tma_load指令, 开始向buf0加载数据.

**第1阶段 (Consumer) & 第2阶段 (Producer)**:

并行地, UMMA线程调用`consumer_acquire`等待full_barrier[0]. 它会阻塞.

并行地, TMA线程推进到下一阶段, 调用`producer_acquire(buf1)`. empty_barrier[1]可使用, 因此不会阻塞.

Leader CTA对full_barrier[1]执行`arrive`, 预设掩码.

TMA线程发出tma_load指令, 开始向buf1加载数据.

**TMA完成加载buf0**:

TMA硬件自动完成对full_barrier[0]的事务. full_barrier[0]状态翻转.

等待full_barrier[0]的UMMA线程被唤醒, 开始使用buf0中的数据进行计算.

**第1阶段 (Consumer Release) & 第2阶段 (Consumer Wait)**:

UMMA线程完成对buf0的计算后, 调用`consumer_release(buf0)`. 它对empty_barrier[0]执行`arrive`.

与此同时, TMA可能已经完成了对buf1的加载, full_barrier[1]状态翻转, UMMA可以开始计算buf1.

**循环:**

当所有消费者都释放了buf0后, empty_barrier[0]状态翻转, 变为"可使用".

TMA线程在完成后续阶段后, 最终会再次循环到`producer_acquire(buf0)`, 此时它可以立即获取该缓冲区, 开始新一轮的数据加载.
3.2.2 PipelineUmmaAsync
在这个场景中:

**生产者 (Producer)**: UMMA 指令. 它完成了的计算, 生产出了最终的累加器结果Tile.

**消费者 (Consumer)**: 异步线程, 用于Epilogue处理和异步拷贝回GMEM.

类定义如下:

```
@dataclass(frozen=True)class PipelineUmmaAsync(PipelineAsync):    """    PipelineUmmaAsync is used for UMMA producers and AsyncThread consumers (e.g. Blackwell accumulator pipelines).    """    # 与 PipelineTmaUmma 中一样, 表示UMMA操作涉及的CTA的规模(1个或2个CTA).    cta_group: cute.nvgpu.tcgen05.CtaGroup
```
create(...)
首先是Role的定义, 明确生产者是`PipelineOp.TCGen05Mma`(UMMA), 消费者是`PipelineOp.AsyncThread`. 然后 创建`sync_object_full`和`sync_object_empty`. 值得注意的是, 这里为`sync_object_full`创建的`mbarrier`**不使用`tx_count`参数**, 因为生产者UMMA是软件线程, 它的完成是通过显式的`arrive`调用来标志的,

然后是通过调用`_compute_tmem_sync_mask`和`_compute_peer_cta_rank`来配置生产者提交和消费者释放时所需的掩码/目标.
_compute_tmem_sync_mask(cta_layout_vmnk: cute.Layout)
计算一个用于**producer commit** 的同步掩码.当UMMA采用2-CTA协作模式时, 一个结果tile是由两个CTA共同计算产生的. 消费者必须等待**两个**CTA都完成了自己的那部分计算后, 才能开始拷贝整个tile到RMEM. 这个掩码定义了哪些CTA属于同一个生产者协作组. 当组内所有CTA都对`full`屏障执行`arrive`操作后, 屏障才会触发, 通知消费者数据已就绪.
_compute_peer_cta_rank()
计算一个用于**consumer release**的目标rank.消费者在完成数据写回后, 需要通知生产者, 对应的TMEM缓冲区现在空了, 可以用来存放下一个计算结果.这个通知不需要发给生产者组内的所有CTA, 只需要发给它们的Leader即可. Leader收到信号后, 会协调其Pair开始下一轮计算.  `cta_rank_in_cluster // 2 * 2` 实现很简单, 例如, rank 0和1的CTA, 计算结果都是0; rank 2和3的CTA, 计算结果都是2. 这样, 消费者就能精确地将"缓冲区已空"的信号只发送给需要接收这个信号的那个Producer Leader.
producer_commit(self, state: PipelineState)
生产者(UMMA warps)完成一个结果tile的计算之后, 通知消费者, 结果数据已在TMEM中准备就绪. 实现为调用`self.sync_object_full.arrive(state.index, self.producer_mask, self.cta_group)`.它使用`producer_mask`来确保只有当一个生产组内的所有CTA都完成计算后, `full`屏障才会触发. `cta_group`指明了这是来自UMMA协作组的信号.
consumer_release(self, state: PipelineState)
此方法由基类`PipelineAsync`提供, 但行为由`consumer_mask`决定, 实现为:`self.sync_object_empty.arrive(state.index, self.consumer_mask)`.  即消费者对empty barrier执行`arrive`操作. `self.consumer_mask`在这里就是`_compute_peer_cta_rank`计算出的领导者rank. 信号被精确地发送给下一个要使用此缓冲区的UMMA生产者小组的Leader.
producer_tail(self, state: PipelineState)
这是一个非常重要的收尾工作, 考虑Kernel的最后一次迭代. UMMA计算完最后一个tile, 消费者开始将其异步拷贝到RMEM处理. 如果UMMA线程(生产者)此时直接退出Kernel, 那么它所使用的TMEM资源可能会被GPU系统回收. 但此时, 消费者线程可能仍在从这块TMEM中读取数据, 将会导致一个Race Condition.

具体解决方案是`producer_tail`由生产者在退出前调用. 它强制生产者等待消费者完成其最后一次工作. 实现如下:

函数中`is_leader_cta`: 只有Leader CTA需要执行这个等待操作.

`for i in cutlass.range_constexpr(self.num_stages - 1): state.advance()`: 将流水线状态`state`快进到指向最后一次被使用的那个缓冲区.

`self.producer_acquire(state)`: 然后, 生产者对这个最后的缓冲区执行`acquire`操作. 这个操作会等待`empty`屏障, 而这个屏障只有在消费者完成了对该缓冲区的拷贝并执行完Epilogue计算写回GMEM后调用`consumer_release`才会消除阻塞.

以一个双缓冲(`num_stages=2`)的Epilogue为例:

**第1阶段 (Producer)**:

UMMA线程组计算出结果tile, 存入`buf0`.

完成后, 调用`producer_commit(buf0)`. `full_barrier[0]`触发, 通知消费者.

**第1阶段 (Consumer) & 第2阶段 (Producer)**:

并行地, 异步线程(消费者)等待`full_barrier[0]`, 被唤醒后, 开始从`buf0`向RMEM拷贝数据.

并行地, UMMA线程组(生产者)推进到下一阶段, 调用`producer_acquire(buf1)`. 假设`buf1`是空的, 它立即获取该缓冲区, 开始计算下一个结果tile并存入`buf1`.

**第2阶段 (Producer Commit) & 第1阶段 (Consumer Release)**:

UMMA完成对`buf1`的计算, 调用`producer_commit(buf1)`. `full_barrier[1]`触发.

与此同时, 异步线程可能已完成`buf0`的拷贝. 它调用`consumer_release(buf0)`, 这会触发`empty_barrier[0]`, 使得`buf0`对生产者重新可用.

**循环**: 消费者接着开始拷贝`buf1`, 生产者则可以回头去获取`buf0`进行再下一轮的计算. 计算和写回操作在不同的缓冲区上流水进行.

**结束**: 当所有tile都计算完毕, 领导者UMMA线程调用`producer_tail`, 等待最后那个tile被消费者安全处理后, 才最终退出Kernel.

`PipelineUmmaAsync` 与 `PipelineTmaUmma` 互为补充, 共同构成了Blackwell架构下GEMM Kernel的完整数据流管理方案.

3.2.3 PipelineTmaMultiConsumersAsync
用于一个TMA生产者和多个不同类型的消费者, 例如, 一部分是执行MMA计算的warp, 另一部分是其他异步线程(AsyncThread). 这个类进一步扩展了同步的复杂性. 因为有两种消费者, 它们必须都完成工作后, 才能通知生产者缓冲区为空.

`sync_object_empty`被设计为`Composite`类型, 意味着它内部管理着多个消费者组的到达.

`sync_object_empty_umma`和`sync_object_empty_async`是`sync_object_empty`的两个"视图", 分别给两种不同的消费者使用.

`consumer_release()`方法接收一个`op_type`参数, 根据调用者是MMA单元还是异步线程, 来在对应的`mbarrier`上发信号.

当所有类型的消费者都对同一个阶段的`empty_bar`发出了`arrive`信号后, 这个`empty_bar`的状态才会变为`empty`, 从而解除生产者的阻塞.

相应的MBarrier处理逻辑类似, 在此就不赘述了. 后面在介绍FlashAttention一类的算子时详细展开.

## 4. Simple GEMM Example

在cuteDSL中针对Blackwell有很多GEMM kernel实现, 我们先从Tutorial gemm[4]开始介绍. 关于Blockscale_gemm和group_gemm将在后续单独几篇文章介绍.

### 4.1 Overview

这个`tutorial_fp16_gemm_0.py`是一个精炼而深刻的教学示例.  核心代码大概仅有200多行, 但是基本上涵盖了Blackwell的操作. 这个例子固化了一些GEMM Kernel的参数, 如下表所示:

| 参数名 | 值 | 解释 |
| --- | --- | --- |
| `io_dtype` | `cutlass.Float16` | 输入/输出矩阵(A, B, C)的数据类型为FP16 . |
| `acc_dtype` | `cutlass.Float32` | 累加器(Accumulator)的数据类型为FP32. |
| `mma_inst_shape_mnk` | `(128, 256, 16)` | Tcgen05的指令支持的Shape. 这意味着一个硬件指令可以执行一个 128 X 256 X 16 的MMA操作. |
| `mma_tiler_mnk` | `(128, 256, 64)` | CTA级的Tile大小. M=128, N=256, K=64. |
| `threads_per_cta` | `128` | 每个线程块包含128个线程. |
| `ab_stages` | `4` | 为输入矩阵A和B的软件流水线设置的Stages. |
| `acc_stage` | `1` | 为累加器设置的流水线阶段数. |

### 4.2 SharedStorage 结构体

这是一个在共享内存(Shared Memory, SMEM)中定义的结构体, 用于管理内核需要的一些元数据.

```
@cute.structclass SharedStorage:    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]    tmem_holding_buf: cutlass.Int32
```

`ab_mbar_ptr`: 用于管理A和B矩阵加载和UMMA计算流水线的MBarrier指针. `ab_stages * 2`是因为每个阶段都需要 2 个Mbarrier

`acc_mbar_ptr`: 用于管理累加器流水线的MBarrier指针.

`tmem_holding_buf`: 用于TMEM分配.

### 4.3 Host侧函数

首先是构造Tile MMA对象. 如下所示:

```
    # Construct tiled MMA    op = tcgen05.MmaF16BF16Op(        io_dtype,        acc_dtype,        mma_inst_shape_mnk,        tcgen05.CtaGroup.ONE, #使用1-CTA模式        tcgen05.OperandSource.SMEM,        tcgen05.OperandMajorMode.K,        tcgen05.OperandMajorMode.K,    )        # 生成Tiled-MMA    tiled_mma = cute.make_tiled_mma(op)        print(f"tiled_mma    = {cute.pretty_str(tiled_mma)}")# outputtiled_mma    = Tiled MMA  Thr Layout VMNK: (1,1,1,1):(0,0,0,0)  Permutation MNK: (_,_,_)MMA Atom  ThrID:           1:0  Shape MNK:       (128,256,16)  TV Layout A:     (1,(128,16)):(128,(1,128))  TV Layout B:     (1,(256,16)):(256,(1,256))  TV Layout C:     (1,(128,256)):(128,(1,128))  
```

`cute.make_tiled_mma(op)`根据硬件指令Shape和数据类型, 创建一个`TiledMma`对象. 然后是使用`sm100_utils.make_smem_layout_a/b`辅助函数, 用于为A和B矩阵创建最优的SMEM数据布局. 其中已经考虑到TMA和UMMA所需要的Swizzle类型, 避免bank conflict.

```
    # Construct SMEM layouts for A and B    a_smem_layout = sm100_utils.make_smem_layout_a(        tiled_mma,        mma_tiler_mnk,        a.element_type,        ab_stages,    )    b_smem_layout = sm100_utils.make_smem_layout_b(        tiled_mma,        mma_tiler_mnk,        b.element_type,        ab_stages,    )    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])#outputa smem layout    = S<3,4,3> o 0 o ((128,16),1,4,4):((64,1),0,16,8192)b smem layout    = S<3,4,3> o 0 o ((256,16),1,4,4):((64,1),0,16,16384)
```

然后使用`make_tiled_tma_atom_A/B`定义了TMA-ATOM.

```
    cluster_layout_vmnk = cute.tiled_divide(        cute.make_layout((1, 1, 1)),        (tiled_mma.thr_id.shape,),    )    # Construct TMA load atoms    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(        op,        a,        a_smem_layout_one_stage,        mma_tiler_mnk,        tiled_mma,        # cluster_layout_vmnk.shape,    )    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(        op,        b,        b_smem_layout_one_stage,        mma_tiler_mnk,        tiled_mma,        # cluster_layout_vmnk.shape,    )#outputa_tma_atom   = Copy Atom  ThrID:         1:0  TV Layout Src: (1,8192):(0,1)  TV Layout Dst: (1,8192):(0,1)  Value type:    f16b_tma_atom   = Copy Atom  ThrID:         1:0  TV Layout Src: (1,16384):(0,1)  TV Layout Dst: (1,16384):(0,1)  Value type:    f16  a tma tensor (0,0) o (8192,8192):(1@1,1@0)b tma tensor (0,0) o (8192,8192):(1@1,1@0)
```

准备完UMMA和TMA描述符后, 就可以启动Kernel了.

```
    # 基于C的shape ceildiv mma_tiler_mn 计算grid shape: (64,32,1)    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])        # Launch the kernel    kernel(        tiled_mma,        a_tma_atom,        a_tma_tensor,        b_tma_atom,        b_tma_tensor,        c,        a_smem_layout,        b_smem_layout,    ).launch(        grid=grid_shape,        block=(threads_per_cta, 1, 1),    )
```

### 4.4 Kernel

#### 4.4.1 初始化

这个阶段负责初始化资源, 设置流水线, 以及最关键的: 使用CuTe对张量进行Partition. 首先需要获取获取当前线程, Warp和线程块在网格(Grid)中的坐标.

```
    # Current thread/warp/block coordinates    tidx, _, _ = cute.arch.thread_idx()    warp_idx = cute.arch.warp_idx()    warp_idx = cute.arch.make_warp_uniform(warp_idx)    bidx, bidy, _ = cute.arch.block_idx()    mma_coord_mnk = (bidx, bidy, None)
```

然后利用SharedStorage结构体分配内存, 这个示例中并没有把sA和sB放入结构体, 而是直接在smem中allocate_tensor

```
    # Allocate SMEM    smem = cutlass.utils.SmemAllocator()    storage = smem.allocate(SharedStorage)    sA = smem.allocate_tensor(        element_type=io_dtype,        layout=a_smem_layout.outer,        byte_alignment=128,        swizzle=a_smem_layout.inner,    )    sB = smem.allocate_tensor(        element_type=io_dtype,        layout=b_smem_layout.outer,        byte_alignment=128,        swizzle=b_smem_layout.inner,    )
```

在Blackwell中需要关注的是Tensor Memory的分配, 它需要一个NamedBarrier, 它是一个由硬件管理的Barrier, 最多支持16个,  barrier_ids值的范围是0-15. 另外TMEM需要按照列分配, 这里完全分配了512列.

```
    # Allocate all TMEM columns    tmem_alloc_barrier = pipeline.NamedBarrier(        barrier_id=1,        num_threads=threads_per_cta,    )    tmem = utils.TmemAllocator(        storage.tmem_holding_buf,        barrier_for_retrieve=tmem_alloc_barrier,    )    num_tmem_cols = 512    tmem.allocate(num_tmem_cols)
```
4.4.1.1 设置流水线
这一节使用了第三章所用的`PipelineTmaUmma`和`PipelineUmmaAsync`两个类, 并且使用了`make_participants`生成PipelineProducer和PipelineConsumer避免了直接对state的操作.

```
    # Prefetch tma descriptor    if warp_idx == 0:        cpasync.prefetch_descriptor(tma_atom_a)        cpasync.prefetch_descriptor(tma_atom_b)    # Pipeline configuration        # 计算TMA所需要的tx-count    num_tma_copy_bytes = cute.size_in_bytes(        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))        # 用于TMA Producer和UMMA Consumer的Pipeline    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(        num_stages=ab_stages,        # TMA和UMMA都由Thread issue, 因此这里pipeline Agent为Thread, 默认cnt为1.        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),        tx_count=num_tma_copy_bytes,        barrier_storage=storage.ab_mbar_ptr.data_ptr(),    ).make_participants()        # 用于UMMA Producer和Epilogue Consumer的Pipeline    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(        num_stages=acc_stage,        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),        # consumer需要CTA内多个线程, 因此arrive cnt = threads_per_cta        consumer_group=pipeline.CooperativeGroup(            pipeline.Agent.Thread, threads_per_cta        ),        barrier_storage=storage.acc_mbar_ptr.data_ptr(),    ).make_participants()
```
4.4.1.2 Tensor Partitioning
使用CuTe将全局问题分解

```
    # Partition tensors for MMA and make fragments    # (bM, bK, RestK)    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))    # (bN, bK, RestK)    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))    # (bM, bN)    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
```

`gA`, `gB`, `gC`: 使用`cute.local_tile`从全局矩阵`mA_mkl`, `mB_nkl`, `mC_mnl`中"切出"当前线程块(CTA)负责处理的部分.

```
    thr_mma = tiled_mma.get_slice(0)    # (MMA, MMA_M, MMA_K)    tCgA = thr_mma.partition_A(gA)    # (MMA, MMA_N, MMA_K)    tCgB = thr_mma.partition_B(gB)    # (MMA, MMA_M, MMA_N)    tCgC = thr_mma.partition_C(gC)
```

`tCgA`, `tCgB`, `tCgC`: `thr_mma.partition_A/B/C`将CTA的Tile进一步分区, 得到每个线程在计算时所看到的全局内存视图.

```
    # (MMA, MMA_M, MMA_K)    tCrA = tiled_mma.make_fragment_A(sA)    # (MMA, MMA_N, MMA_K)    tCrB = tiled_mma.make_fragment_B(sB)    # (MMA, MMA_M, MMA_N)    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])    # (MMA, MMA_M, MMA_N)    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
```

`tCrA`, `tCrB`: `tiled_mma.make_fragment_A/B`创建了线程的RMEM fragment. 这些是张量核心指令直接操作的数据. `tCr`中的`r`代表`register`.

```
    # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM    # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address    tmem.wait_for_alloc()    tmem_ptr = tmem.retrieve_ptr(acc_dtype)    # Swap the pointer in tCtAcc    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)
```

`tCtAcc`: `tiled_mma.make_fragment_C`创建了用于存放累加结果的**TMEM**片段. `tCt`中的`t`代表`TMEM`. 随后`tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)`将这个逻辑上的片段与前面分配的物理TMEM地址关联起来.

```
    # Partition tensors for TMA; This requires the tensors partitioned for MMA    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(        tma_atom_a,        0,        cute.make_layout(1),        cute.group_modes(sA, 0, 3),        cute.group_modes(tCgA, 0, 3),    )    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(        tma_atom_b,        0,        cute.make_layout(1),        cute.group_modes(sB, 0, 3),        cute.group_modes(tCgB, 0, 3),    )
```

`tAsA`, `tAgA`, `tBsB`, `tBgB`: `cute.nvgpu.cpasync.tma_partition`为TMA数据搬运操作进行分区, 定义了每次TMA copy的源(全局内存, `g`)和目标(共享内存, `s`).
4.4.1.3 Epilogue TMEM copy
针对Epilogue中需要的TMEM加载到RMEM构建TMEM_ATOM并处理相关的layout和分配寄存器资源

```
    subtile_cnt = 4    # (EpiTile)    epi_tiler = (        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),    )    # (EpiTile, NumTiles)    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)    # (EpiTile, NumTiles)    gC_epi = cute.zipped_divide(tCgC, epi_tiler)    # Every thread loads 32x128 bits    tmem_atom = cute.make_copy_atom(        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),        cutlass.Float32,    )    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)    # (TmemCpy,NumTmemCpy,NumTiles)    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)    # (TmemCpy,NumTmemCpy,NumTiles)    tDgC = tmem_thr_copy.partition_D(gC_epi)    # (TmemCpy,NumTmemCpy)    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)    # (TmemCpy,NumTmemCpy)    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)
```

#### 4.4.2 Main loop

MainLoop中的代码也非常简单

```
    if warp_idx == 0:        # 通过acc_producer 等待accumulator一个空的Buffer        acc_empty = acc_producer.acquire_and_advance()                # 注意这里使用了一个cuteDSL的语法糖, 允许定义prefetch_stages.        for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):                        # Issue TMA之前检查smem_empty mbarrier            ab_empty = ab_producer.acquire_and_advance()                        # 拷贝A Tile到SMEM            cute.copy(                tma_atom_a,                tAgA[(None, ab_empty.count)],                tAsA[(None, ab_empty.index)],                tma_bar_ptr=ab_empty.barrier,            )            # 拷贝B Tile到SMEM            cute.copy(                tma_atom_b,                tBgB[(None, ab_empty.count)],                tBsB[(None, ab_empty.index)],                tma_bar_ptr=ab_empty.barrier,            )            # Execute one K-block worth of MMA instructions            ab_full = ab_consumer.wait_and_advance()            num_k_blocks = cute.size(tCrA, mode=[2])            for k_block_idx in cutlass.range_constexpr(num_k_blocks):                k_block_coord = (None, None, k_block_idx, ab_full.index)                cute.gemm(                    tiled_mma,                    tCtAcc,                    tCrA[k_block_coord],                    tCrB[k_block_coord],                    tCtAcc,                )                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)            # 这是一个consumer_release, 内部调用sync_object_empty.arrive时            # 会调用tcgen05.commit            ab_full.release()        # Signal that the accumulator is fully computed        acc_empty.commit()
```

#### 4.4.3 Epilogue

当主循环完成所有K维度的计算后, `tCtAcc` (在TMEM中) 保存了当前CTA Tile的最终FP32结果. Epilogue的任务是将这些结果转换成FP16并写回到全局内存的`C`矩阵中. 首先释放TMEM allocation锁, 然后等待 等待计算流水线完成(acc_full barrier). 确保`tCtAcc`中的数据是最终结果.

```
    # Release TMEM allocation lock    tmem.relinquish_alloc_permit()    # Wait for the accumulator buffer to be full    acc_full = acc_consumer.wait_and_advance()    # TMEM -> RMEM -> GEMM    # Sub-tiling for better instruction-level parallelism    for i in cutlass.range(cute.size(tDtC, mode=[2])):        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)        tCrC.store(tCrAcc.load().to(io_dtype))        cute.autovec_copy(tCrC, tDgC[None, None, i])    acc_full.release()    # Deallocate TMEM    pipeline.sync(barrier_id=1)    tmem.free(tmem_ptr)
```

**TMEM -> RMEM -> GMEM**: 这是一个分阶段的数据移动和转换过程.

`cute.copy(tmem_tiled_copy, tDtC, tCrAcc)`: 使用`tcgen05.Ld32x32bOp`指令, 将TMEM中的FP32累加结果(`tDtC`)拷贝到寄存器(`tCrAcc`). 这里的`tmem_tiled_copy`为初始化阶段定义的Copy-Atom

`tCrC.store(tCrAcc.load().to(io_dtype))`: 在寄存器中, 将FP32数据转换为FP16. `tCrC`是用于存储转换后FP16结果的寄存器Fragment.

`cute.autovec_copy(tCrC, tDgC)`: 将寄存器中的FP16结果(`tCrC`)写回到全局内存(`tDgC`). `autovec_copy`会尝试使用向量化的存储指令来提升带宽利用率.

## 5. GEMM Persistent Kernel

### 5.1 Overview

这是在cutedsl中的dense_gemm_persistent.py[5]例子是一个比较完善的GEMM kernel, 支持以下特性:

利用 TMA 实现高效的内存操作.

利用 Blackwell 的 tcgen05.mma 指令进行 MMA 操作 (包括 2cta mma 指令).

通过 CGA 实现 TMA 多播 (multicast), 以减少 L2 内存流量.

支持 Persistent tile 调度, 以便在 tile 之间更好地重叠内存加载/存储与 mma 操作.

支持 warp specialization, 以避免在主循环的加载和 mma 操作之间进行显式的流水线管理.

支持的输入数据类型: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2)

Mma tile 的 M 维度必须是 64/128 (当 use_2cta_instrs=False 时) 或 128/256 (当 use_2cta_instrs=True 时).

Mma tile 的 N 维度必须在 32-256 范围内, 且步长为 32.

但是现在官方的CuteDSL示例中还没有为其添加DynamicTileScheduler功能, 调度依旧使用StaticTileScheduler.

### 5.2 初始化参数

主要分两部分, 一部分是`__init__`构造函数中的配置一些静态参数. 然后是在`_setup_attributes()`中, 根据获得的具体输入张量信息后, 计算和设置依赖于输入的动态属性.

在`__init__`构造函数中初始化了Kernel的静态配置.关键参数如下:

`acc_dtype`: 累加器数据类型 (e.g., Float32).

`use_2cta_instrs`: 是否使用需要两个 CTA 协作的 MMA 指令. 并定义`cta_group`为`tcgen05.CtaGroup.ONE / TWO`

`mma_tiler_mn`: MMA 计算的基本 tile 尺寸, e.g., (128, 128).

`cluster_shape_mn`: Cluster 的形状, e.g., (2, 1) 表示一个 Cluster 由 2x1=2 个 CTA 组成.

`use_tma_store`: Epilogue 阶段是否使用 TMA 来存储最终结果 C.

它使用了Warp Specialization并定义了如下几类Warp

`epilog_warp_id = (0, 1, 2, 3)`: 0-3号 warp 负责 epilogue.

`mma_warp_id = 4`: 4号 warp 负责 MMA 计算.

`tma_warp_id = 5`: 5号 warp 负责 TMA 数据加载.

并根据Warp数量计算了每个CTA的线程数,  1 (MMA) + 1 (TMA) + 4 (Epilogue) = 6 个 warps, 总计 $6 \times 32 = 192$ 个线程.

```
        self.threads_per_cta = 32 * len(            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)        )
```

然后还定义了一些用于CTA同步的NamedBarrier的bar_id

```
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync        self.epilog_sync_bar_id = 1        self.tmem_alloc_sync_bar_id = 2        self.tmem_dealloc_sync_bar_id = 3
```

并且获取了SMEM容量和定义occupancy参数, 用于计算流水线级数, 设置目标占用率(occupancy)为 1, 即每个 SM 上只运行一个 CTA. 这是为了最大化单个 CTA 可用的 SMEM 和其他资源.

```
        self.occupancy = 1        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
```

然后会根据这两个参数在后面调用`_compute_stages` 函数, 最大化 A/B 缓冲区的级数 (`num_ab_stage`) 以便尽可能地隐藏访存延迟.

在`_setup_attributes`在获得具体输入张量信息后被调用, 用于动态配置.

首先还是构造Tiled_MMA对象, 并动态计算 MMA tile 的 K 维度. 它由硬件 MMA 指令本身的能力 (`mma_inst_shape_k`) 和一个平铺因子 (`mma_inst_tile_k`) 决定.

```
        # Configure tiled mma        tiled_mma = sm100_utils.make_trivial_tiled_mma(            self.a_dtype,            self.a_major_mode,            self.b_major_mode,            self.acc_dtype,            self.cta_group,            self.mma_tiler[:2],        )        # Compute mma/cluster/tile shapes        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])        mma_inst_tile_k = 4        self.mma_tiler = (            self.mma_tiler[0],            self.mma_tiler[1],            mma_inst_shape_k * mma_inst_tile_k,        )
```

然后基于此计算出整个 CTA 处理的 `cta_tile_shape_mnk`.

```
        self.cta_tile_shape_mnk = (            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),            self.mma_tiler[1],            self.mma_tiler[2],        )            
```

接下来创建 `cluster_layout_vmnk`, 它定义了 Cluster 内 CTA 的逻辑排布.

```
        # Compute cluster layout        self.cluster_layout_vmnk = cute.tiled_divide(            cute.make_layout((*self.cluster_shape_mn, 1)),            (tiled_mma.thr_id.shape,),        )# 例如使用cluster_shape_mn=(2,1)时, shape如下:tiled_mma thr_id.shape 2  cluster vmnk ((2),1,1,1):((1),0,0,0)
```

然后就可以根据cluster_layout_vmnk来判断是否要进行multicast

```
        # Compute number of multicast CTAs for A/B        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])        self.is_a_mcast = self.num_mcast_ctas_a > 1        self.is_b_mcast = self.num_mcast_ctas_b > 1
```

然后是计算Epilogue Tile的形状和Layout, 如果Epilogue Tile尺寸比较大, 然后又不想直接从RMEM拷贝到GMEM, 可以在这里先暂存到SMEM, 然后用TMA异步拷贝到GMEM中, 因此也要针对Epilogue Tile构建在SMEM中的Layout

```
        # Compute epilogue subtile        if cutlass.const_expr(self.use_tma_store):            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(                self.cta_tile_shape_mnk,                self.use_2cta_instrs,                self.c_layout,                self.c_dtype,            )        else:            self.epi_tile = self.cta_tile_shape_mnk[:2]        c_smem_layout = None        if cutlass.const_expr(self.use_tma_store):            c_smem_layout = sm100_utils.make_smem_layout_epi(                self.c_dtype, self.c_layout, self.epi_tile, 1            )
```

然后就是调用`_compute_stages`计算流水线级数

```
self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(...)def _compute_stages(...) -> Tuple[int, int, int]:    """Computes the number of stages for A/B/C operands based on heuristics."""    # Default ACC stages    # 设置累加器 (在 TMEM 中) 的流水线深度默认为 2.     # 这意味着 MMA warp 可以计算一个 tile, 同时 Epilogue warp 正在处理上一个 tile 的结果, 形成双缓冲.    num_acc_stage = 2    # Default C stages    # 如果 Epilogue 使用 TMA 存储 C, 则为 C 在 SMEM 中设置 2 级流水线. 否则, C 不经过 SMEM, 不需要流水线.    num_c_stage = 2 if use_tma_store else 0    # Calculate smem layout and size for one stage of A, B, and C with 1-stage    a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(...)    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(...)        # 一个 stage 的 A 和 B buffer 所占用的 SMEM 字节数.    ab_bytes_per_stage = cute.size_in_bytes(...) + cute.size_in_bytes(...)    mbar_helpers_bytes = 1024    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)    c_bytes = c_bytes_per_stage * num_c_stage    # 计算 A/B 缓冲区的流水线深度, 然后这里根据occupancy因子计算内存占用    num_ab_stage = (        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)    ) // ab_bytes_per_stage    # Refine epilogue stages:    # 在计算完 num_ab_stage 后, 由于整数除法的存在, 可能还有一部分 SMEM 未被使用.     # 这个步骤就是把这部分剩余 SMEM 全部利用起来, 增加 C buffer 的深度.     # 这可以为 Epilogue 阶段的 R2S (Register to SMEM) 和 S2G (SMEM to GMEM) 操作提供更好的流水线能力.    if use_tma_store:        num_c_stage += (            smem_capacity            - occupancy * ab_bytes_per_stage * num_ab_stage            - occupancy * (mbar_helpers_bytes + c_bytes)        ) // (occupancy * c_bytes_per_stage)    return num_acc_stage, num_ab_stage, num_c_stage        
```

最后生成在SMEM中的Staged Layout, 在make_smem_layout函数中还包含了相应的swizzle的设置, 避免bank conflict. 然后根据Tiled MMA和accumulate stage计算TMEM需要多少个Column, 用于后面TMEM内存分配,

```
        # Compute A/B/C shared memory layout        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage        )        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage        )        self.c_smem_layout_staged = None        if self.use_tma_store:            self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage            )        # Compute the number of tensor memory allocation columns        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(            tiled_mma, self.mma_tiler, self.num_acc_stage        )
```

### 5.3 Host端函数

`__call__` 为Host端函数, 负责准备所有参数并启动 kernel.

第一步, 获取输入张量 A, B, C 的属性 (数据类型, 布局):

```
        # Setup static attributes before smem/grid/tma computation        self.a_dtype: Type[cutlass.Numeric] = a.element_type        self.b_dtype: Type[cutlass.Numeric] = b.element_type        self.c_dtype: Type[cutlass.Numeric] = c.element_type        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()        self.c_layout = utils.LayoutEnum.from_tensor(c)
```

第二步, 调用 `_setup_attributes()` 完成动态配置. 前面一节已经详细介绍了.

第三步, 配置 TMA-ATOM

`make_tiled_tma_atom_A/B`: 创建用于加载 A 和 B 的 TMA ATOM, 这会生成 TMA 描述符所需的信息. 以矩阵 A 为例,

```
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(            self.cluster_shape_mn, tiled_mma.thr_id        )        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(            a_op,            a,            a_smem_layout,            self.mma_tiler,            tiled_mma,            self.cluster_layout_vmnk.shape,            internal_type=(                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None            ),        )
```

然后计算TMA操作拷贝A/B所需要的tx-bytes:

```
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
```

对于矩阵C, 如果使用了`use_tma_store`创建用于存储 C 的 TMA ATOM.

```
        # Setup TMA store for C        tma_atom_c = None        tma_tensor_c = None        if cutlass.const_expr(self.use_tma_store):            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(                cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile            )        
```

第四步, 计算 Grid 尺寸并生成TileScheduler的参数: 它会考虑问题总大小, cluster 形状以及硬件支持的最大活跃 cluster 数.

```
# Compute grid sizeself.tile_sched_params, grid = self._compute_grid(    c, #基于C Tensor    self.cta_tile_shape_mnk, #考虑CTA Tile Shape    self.cluster_shape_mn,  # 考虑Cluster Shape    max_active_clusters  # 从)
```

其中max_active_cluster参数获取如下, 实际上需要根据硬件的SM和GPC数量以及CGA的Layout决定

```
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(        cluster_shape_mn[0] * cluster_shape_mn[1]    )
```

compute_grid计算如下, 由于使用了Persistent Kernel, 因此需要根据max_active_clusters以及cluster shape来计算, 并计算实际的所需计算的Tile数量, 生成TileScheduler的参数.

```
    @staticmethod    def _compute_grid(...) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:            # 计算需要多少个Tile        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))        gc = cute.zipped_divide(c, tiler=c_shape)        num_ctas_mnl = gc[(0, (None, None, None))].shape        cluster_shape_mnl = (*cluster_shape_mn, 1)                # 根据tile数量和cluster shape构建tile schedule 参数        tile_sched_params = utils.PersistentTileSchedulerParams(            num_ctas_mnl, cluster_shape_mnl        )        grid = utils.StaticPersistentTileScheduler.get_grid_shape(            tile_sched_params, max_active_clusters        )        return tile_sched_params, grid        
```

第六步, Launch Kernel

```
        # Launch the kernel synchronously        self.kernel(            tiled_mma,            tma_atom_a,            tma_tensor_a,            tma_atom_b,            tma_tensor_b,            tma_atom_c,            tma_tensor_c if self.use_tma_store else c,            self.cluster_layout_vmnk,            self.a_smem_layout_staged,            self.b_smem_layout_staged,            self.c_smem_layout_staged,            self.epi_tile,            self.tile_sched_params,            epilogue_op,        ).launch(            grid=grid,            block=[self.threads_per_cta, 1, 1],            cluster=(*self.cluster_shape_mn, 1),            stream=stream,        )
```

### 5.4 Kernel函数

在`kernel(...)`函数中, 主要分为如下几个阶段:

#### 5.4.1 初始化阶段

第一步, 获取线程/warp/block 坐标, 特别是 `cta_rank_in_cluster`.

```
        tidx, _, _ = cute.arch.thread_idx()                bidx, bidy, bidz = cute.arch.block_idx()        warp_idx = cute.arch.warp_idx()        warp_idx = cute.arch.make_warp_uniform(warp_idx)                # 如果tiled_mma.thr_id.shape =2 即需要使用2 CTA        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2                 # 判断是否为Leader CTA        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)        is_leader_cta = mma_tile_coord_v == 0        cta_rank_in_cluster = cute.arch.make_warp_uniform(            cute.arch.block_idx_in_cluster()        )        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(            cta_rank_in_cluster        )
```

然后我们还需要Prefetch TMA的描述符

```
        if warp_idx == self.tma_warp_id:            cpasync.prefetch_descriptor(tma_atom_a)            cpasync.prefetch_descriptor(tma_atom_b)            if cutlass.const_expr(self.use_tma_store):                cpasync.prefetch_descriptor(tma_atom_c)
```

第二步, 构造SharedStorage结构体和分配SMEM

```
        @cute.struct        class SharedStorage:            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]            acc_full_mbar_ptr: cute.struct.MemRange[                cutlass.Int64, self.num_acc_stage * 2            ]            tmem_dealloc_mbar_ptr: cutlass.Int64            tmem_holding_buf: cutlass.Int32        smem = utils.SmemAllocator()        storage = smem.allocate(SharedStorage)
```

第三步, 初始化Pipeline对象, 流水线分为两段, 一段是TMA Producer和UMMA Consumer, 另一段是UMMA Producer和Accumulator Consumer.

`pipeline.PipelineTmaUmma`: 创建 A/B 加载的流水线同步对象 (`ab_producer`, `ab_consumer`).

`pipeline.PipelineUmmaAsync`: 创建累加器计算的流水线同步对象 (`acc_pipeline`).

此时需要注意arrive_cnt的计算, 即num_tma_producer需要考虑组播的情况, 而num_acc_consumer_threads需要考虑到2CTA MMA的情况.

```
        # Initialize mainloop ab_pipeline (barrier) and states        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1        ab_pipeline_consumer_group = pipeline.CooperativeGroup(            pipeline.Agent.Thread, num_tma_producer        )        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),            num_stages=self.num_ab_stage,            producer_group=ab_pipeline_producer_group,            consumer_group=ab_pipeline_consumer_group,            tx_count=self.num_tma_load_bytes,            cta_layout_vmnk=cluster_layout_vmnk,        ).make_participants()        # Initialize acc_pipeline (barrier) and states        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)        num_acc_consumer_threads = len(self.epilog_warp_id) * (            2 if use_2cta_instrs else 1        )        acc_pipeline_consumer_group = pipeline.CooperativeGroup(            pipeline.Agent.Thread, num_acc_consumer_threads        )        acc_pipeline = pipeline.PipelineUmmaAsync.create(            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),            num_stages=self.num_acc_stage,            producer_group=acc_pipeline_producer_group,            consumer_group=acc_pipeline_consumer_group,            cta_layout_vmnk=cluster_layout_vmnk,        )
```

第四步, 创建TMEM内存管理器即相应的NamberBarrier

```
        tmem_alloc_barrier = pipeline.NamedBarrier(            barrier_id=self.tmem_alloc_sync_bar_id,            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),        )        tmem_dealloc_barrier = None        if cutlass.const_expr(not self.use_tma_store):            tmem_dealloc_barrier = pipeline.NamedBarrier(                barrier_id=self.tmem_dealloc_sync_bar_id,                num_threads=32 * len(self.epilog_warp_id),            )        # Tensor memory dealloc barrier init        tmem = utils.TmemAllocator(            storage.tmem_holding_buf,            barrier_for_retrieve=tmem_alloc_barrier,            allocator_warp_id=self.epilog_warp_id[0],            is_two_cta=use_2cta_instrs,            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,        )
```

第五步, 需要完成整个Cluster的同步.确保 Cluster 内所有 CTA 都完成了初始化才开始后续工作.

```
        # Cluster arrive after barrier init        if cute.size(self.cluster_shape_mn) > 1:            cute.arch.cluster_arrive_relaxed()
```

第六步, 设置张量的Layout和张量的Partition, 为后续TMA和UMMA做准备.

首先是分配A和B的SharedMemory, 并根据是否使用2CTA和是否需要mcast设置组播mask.

```
 # (MMA, MMA_M, MMA_K, STAGE)        sA = smem.allocate_tensor(            element_type=self.a_dtype,            layout=a_smem_layout_staged.outer,            byte_alignment=128,            swizzle=a_smem_layout_staged.inner,        )        # (MMA, MMA_N, MMA_K, STAGE)        sB = smem.allocate_tensor(            element_type=self.b_dtype,            layout=b_smem_layout_staged.outer,            byte_alignment=128,            swizzle=b_smem_layout_staged.inner,        )        #        # Compute multicast mask for A/B buffer full        #        a_full_mcast_mask = None        b_full_mcast_mask = None        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):            a_full_mcast_mask = cpasync.create_tma_multicast_mask(                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2            )            b_full_mcast_mask = cpasync.create_tma_multicast_mask(                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1            )
```

然后是构建Local Tile , 使用 `cute.local_tile` 将整个输入/输出矩阵 (A, B, C) 映射成一个由 `self.mma_tiler` 大小的 tile 组成的网格.
cute.local_tile(Tensor, Tile, Rest)
`Tensor`: 原始的全局内存张量 (`mA_mkl`, `mB_nkl`, `mC_mnl`).

`Tile`: 定义了划分用的 "tile" 的形状. 这里使用了 `cute.slice_` 从 `self.mma_tiler` (一个 `(M, N, K)` 元组) 中提取出与当前矩阵相关的维度.

对于 A (`mA_mkl`): 使用 `(None, 0, None)`, 对应 `(M, K)` 维度.

对于 B (`mB_nkl`): 使用 `(0, None, None)`, 对应 `(N, K)` 维度.

对于 C (`mC_mnl`): 使用 `(None, None, 0)`, 对应 `(M, N)` 维度.

`Rest`: 定义了 tile 网格的维度. `(None, None, None)` 表示让 CUTE 自动推断.

生成的结果为以 `gA_mkl`为例, `gB_nkl`, `gC_mnl` 同理.

`gA_mkl`: 这是一个新的 CUTE 张量视图. 它的逻辑形状是 `(bM, bK, RestM, RestK, RestL)`, 其中:

`(bM, bK)`: tile 内部的坐标, 尺寸为 `mma_tiler` 的 M 和 K.

`(RestM, RestK, RestL)`: tile 在整个矩阵网格中的坐标. `RestL` 对应批处理 (batch) 维度.

这一步将全局问题分解成了 tile 级问题. 后续可以通过索引 `(RestM, RestK, RestL)` 来访问任意一个 tile. `PersistentTileScheduler` 就是在 `(RestM, RestN, RestL)` 空间上进行调度.

```
        # (bM, bK, RestM, RestK, RestL)        gA_mkl = cute.local_tile(            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)        )        # (bN, bK, RestN, RestK, RestL)        gB_nkl = cute.local_tile(            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)        )        # (bM, bN, RestM, RestN, RestL)        gC_mnl = cute.local_tile(            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)        )
```

然后是为 TiledMMA 划分 GMEM 张量, 如下所示:

```
        # 从划分好的 `gA_mkl` 中获取 K 维度上的 tile 数量.         # 这将是主循环 `for k_tile in ...` 的迭代次数.        k_tile_cnt = cute.size(gA_mkl, mode=[3])                      #        # Partition global tensor for TiledMMA_A/B/C        #        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)        tCgA = thr_mma.partition_A(gA_mkl)        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)        tCgB = thr_mma.partition_B(gB_nkl)        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)        tCgC = thr_mma.partition_C(gC_mnl)        
```

`tiled_mma` 是描述TensorCore操作的对象.`mma_tile_coord_v` 是当前 CTA 在一个 2CTA 协作组中的 ID (0 或 1).`get_slice` 根据这个 ID 获取当前 CTA 负责的 `tiled_mma` 的那一部分. 下面继续以A为例子, 计算`tCgA`, `tCgB`, `tCgC` 同理.

`partition_A` 方法使用 `thr_mma` 的内部布局信息, 将 tile 级的 GMEM 视图 `gA_mkl` 进一步划分为每个线程的 MMA 操作所需的fragment. 产生的`tCgA`是一个非常复杂的Layout, 其逻辑形状为 `(MMA, MMA_M, MMA_K, RestM, RestK, RestL)`.

`MMA`: 描述了线程如何组成一个 MMA 计算单元.

`(MMA_M, MMA_K)`: 每个线程的 MMA 片段在 tile 内部的 M, K 维度上的形状.

`(RestM, RestK, RestL)`: 继承自 `gA_mkl` 的 tile 网格坐标.

这一步将问题从 tile 级分解到了**Thread-MMA-Fragment**级. `tCgC` 最终会用于 Epilogue 阶段, 以便每个线程都能找到自己负责的 GMEM 输出位置.

接下来为TMA Copy进行Partition, 使用 `cpasync.tma_partition` 为 TMA 异步加载操作准备源 (GMEM) 和目标 (SMEM) 的张量视图.
cpasync.tma_partition(TMA_Atom, Mcast_Coord, Mcast_Layout, Tensor_S, Tensor_D)
`TMA_Atom` (tma_atom_a): 之前创建的 TMA "原子操作", 包含了拷贝的形状, 步长等信息.

`Mcast_Coord` 和 `Mcast_Layout`: 当前 CTA 在多播 (multicast) 组中的坐标和该组的布局. TMA 会利用这些信息来优化对 L2 cache 的访问.

`Tensor_S` (sA): SMEM 张量, 即拷贝的目标.

`Tensor_D` (tCgA): GMEM 张量视图, 即拷贝的源.

生成的结果如下:

`tAsA`: 划分后的 SMEM 视图. 它的逻辑形状是 `((atom_v, rest_v), STAGE)`. `atom_v` 对应 TMA 一次拷贝的最小单元. `STAGE` 对应 SMEM 的流水线深度.

`tAgA`: 划分后的 GMEM 视图. 它的逻辑形状是 `((atom_v, rest_v), RestM, RestK, RestL)`.

这一步是为 TMA 操作准备. 它创建了两个新的张量视图 `tAsA` 和 `tAgA`, 它们的布局与 TMA 硬件单元的工作方式完全匹配. 在主循环中, `cute.copy(tma_atom_a, tAgA[...], tAsA[...])` 就可以非常简洁地描述一次复杂的异步张量拷贝.

```
        # TMA load A partition_S/D        a_cta_layout = cute.make_layout(            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape        )        # ((atom_v, rest_v), STAGE)        # ((atom_v, rest_v), RestM, RestK, RestL)        tAsA, tAgA = cpasync.tma_partition(            tma_atom_a,            block_in_cluster_coord_vmnk[2],            a_cta_layout,            cute.group_modes(sA, 0, 3),            cute.group_modes(tCgA, 0, 3),        )        # TMA load B partition_S/D        b_cta_layout = cute.make_layout(            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape        )        # ((atom_v, rest_v), STAGE)        # ((atom_v, rest_v), RestM, RestK, RestL)        tBsB, tBgB = cpasync.tma_partition(            tma_atom_b,            block_in_cluster_coord_vmnk[1],            b_cta_layout,            cute.group_modes(sB, 0, 3),            cute.group_modes(tCgB, 0, 3),        )
```

最后, 为 TiledMMA 划分 SMEM 和 TMEM, 为Kernel的 MMA 计算 (`cute.gemm`) 准备操作数. 这些操作数是寄存器片段 (register fragments), 它们是对 SMEM 或 TMEM 数据的视图.

```
        # (MMA, MMA_M, MMA_K, STAGE)        tCrA = tiled_mma.make_fragment_A(sA)        # (MMA, MMA_N, MMA_K, STAGE)        tCrB = tiled_mma.make_fragment_B(sB)        # (MMA, MMA_M, MMA_N)        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])        # (MMA, MMA_M, MMA_N, STAGE)        tCtAcc_fake = tiled_mma.make_fragment_C(            cute.append(acc_shape, self.num_acc_stage)        )
```

以tCrA为例,  `make_fragment_A` 是 `TiledMma` 对象的一个方法. 它根据张量核心的内部数据需求, 将 SMEM 张量 `sA` 划分成 MMA 指令可以直接使用的寄存器片段(Register Fragment)视图.结果 `tCrA` 的逻辑形状为 `(MMA, MMA_M, MMA_K, STAGE)`, 与第二部分中的 `tCgA` 类似, 但它指向的是 SMEM 而非 GMEM, 并且包含了 `STAGE` 维度.

`make_fragment_C`: 创建一个指向 TMEM 的累加器 C 的寄存器片段视图. 此时 TMEM 尚未分配, 所以这是一个"伪"张量 (`tCtAcc_fake`), 仅用于获取布局信息. 真正的 TMEM 张量将在 MMA Warp 中分配指针后创建.

这一步完成了从 SMEM/TMEM 到 RMEM (寄存器) 的最后一次逻辑映射. 在 `cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)` 调用中, `tCrA`, `tCrB`, `tCtAcc` 就是这里创建的寄存器片段视图. CUTE 和编译器会确保数据被正确地从 SMEM/TMEM 加载供TensorCore使用, 然后再写回 TMEM.

就此, 初始化过程完成, 再进行一次整个Cluster的同步

```
        if cute.size(self.cluster_shape_mn) > 1:            cute.arch.cluster_wait()        else:            cute.arch.sync_threads()
```

接下来的主循环, 通过不同的Warp Specialization完成.

#### 5.4.2 TMA Warp

首先是TMA Warp, 它通过`while work_tile.is_valid_tile:`循环, 从 `tile_sched` 获取一个工作 tile. 然后根据切片加载数据.

```
        if warp_idx == self.tma_warp_id:                        # 创建持久化调度器, 并获取第一个工作 tile.            tile_sched = utils.StaticPersistentTileScheduler.create(                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()            )            work_tile = tile_sched.initial_work_tile_info()            # Persistent Kernel的主循环, 只要还有 tile 要处理就一直循环.            while work_tile.is_valid_tile:                # Get tile coord from tile scheduler                cur_tile_coord = work_tile.tile_idx                mma_tile_coord_mnl = (                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),                    cur_tile_coord[1],                    cur_tile_coord[2],                )                # 对每个MMA Tile所需计算的数据进行切片.                # ((atom_v, rest_v), RestK)                tAgA_slice = tAgA[                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])                ]                # ((atom_v, rest_v), RestK)                tBgB_slice = tBgB[                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])                ]                # 这里有一个Peek优化, "窥探" 一下是否有空的 SMEM buffer,                 # 这是一个非阻塞操作, 主要是在主循环过程中尽快处理.                ab_producer.reset()                peek_ab_empty_status = ab_producer.try_acquire()                                # TMA加载循环                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):                    # 等待A/B Buffer为空, 这里把前面 "窥探" 传入, 如果已经为空可以省一些cycle                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)                    # TMA load A/B                    cute.copy(                        tma_atom_a,                        tAgA_slice[(None, handle.count)],                        tAsA[(None, handle.index)],                        tma_bar_ptr=handle.barrier,                        mcast_mask=a_full_mcast_mask,                    )                    cute.copy(                        tma_atom_b,                        tBgB_slice[(None, handle.count)],                        tBsB[(None, handle.index)],                        tma_bar_ptr=handle.barrier,                        mcast_mask=b_full_mcast_mask,                    )                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1                    # 继续“窥探”下一个slot buffer是否为空.                    peek_ab_empty_status = cutlass.Boolean(1)                    # 等待下一个slot为空.                    if handle.count + 1 < k_tile_cnt:                        peek_ab_empty_status = ab_producer.try_acquire()                # Tile Scheduler移到下一个Tile处理                tile_sched.advance_to_next_work()                work_tile = tile_sched.get_current_work()            # 最后等待所有数据消费完再退出.            ab_producer.tail()
```

#### 5.4.3 UMMA Warp

`if warp_idx == self.mma_warp_id:`时进行UMMA相关的warp计算. 大概流程如下:

**TMEM 分配:** 等待并从 `TmemAllocator` 获取 TMEM 内存指针.

**持久化循环:** 同样根据 `tile_sched` 进行循环.

**流水线操作:**

`ab_consumer.wait_and_advance()`: 等待 SMEM 中 A/B 数据加载完成.

`acc_pipeline.producer_acquire()`: 等待一个可用的 TMEM buffer (用于存放累加结果).

**MMA 主循环:** 遍历 K 维度. 调用 `cute.gemm(tiled_mma, ...)`执行 `tcgen05.mma` 指令. 注意 `tiled_mma.set(tcgen05.Field.ACCUMULATE, True)`: 第一个 K block 是覆盖写, 之后的都是累加.

**提交结果:**`handle.release()` 释放 SMEM buffer, `acc_pipeline.producer_commit()` 通知 epilogue warp 累加器数据已准备好.

**获取下一个 tile:**`tile_sched.advance_to_next_work()`.

```
        if warp_idx == self.mma_warp_id:                        # TMEM 内存分配, 获取TMEM内存指针            tmem.wait_for_alloc()            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)                        # 构建TMEM中tCAcc的Tensor            # (MMA, MMA_M, MMA_N, STAGE)            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)            # 创建Tile Scheduler            tile_sched = utils.StaticPersistentTileScheduler.create(                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()            )            work_tile = tile_sched.initial_work_tile_info()                        # 获取 UMMA->Epilogue Pipeline 的状态.            acc_producer_state = pipeline.make_pipeline_state(                pipeline.PipelineUserType.Producer, self.num_acc_stage            )            # 持久化的循环            while work_tile.is_valid_tile:                # 从tile scheduler获取的work_tile中抽取Tile坐标                cur_tile_coord = work_tile.tile_idx                mma_tile_coord_mnl = (                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),                    cur_tile_coord[1],                    cur_tile_coord[2],                )                # 根据acc_producer_state.index获取当前这一级流水线的tCtAcc                # (MMA, MMA_M, MMA_N)                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]                # “窥探”一下 AB buffer是否为Full状态                # Peek (try_wait) AB buffer full for k_tile = 0                ab_consumer.reset()                peek_ab_full_status = cutlass.Boolean(1)                if is_leader_cta:                    peek_ab_full_status = ab_consumer.try_wait()                # 作为Leader CTA 需要等待accumulator buffer已经为空再开始运算                if is_leader_cta:                    acc_pipeline.producer_acquire(acc_producer_state)                # 第一次循环ACCUMULATE域为False, 代表结果为覆盖写                # 执行完第一次MMA后改为True, 为累加.                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)                # Mma mainloop                for k_tile in range(k_tile_cnt):                    if is_leader_cta:                        # 等待AB Buffer为Full, 即TMA加载完毕.                         handle = ab_consumer.wait_and_advance(peek_ab_full_status)                        # 获取累积的Kblocks                        num_kblocks = cute.size(tCrA, mode=[2])                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):                            kblk_crd = (None, None, kblk_idx, handle.index)                            cute.gemm(                                tiled_mma,                                tCtAcc,                                tCrA[kblk_crd],                                tCrB[kblk_crd],                                tCtAcc,                            )                            # 第一次循环执行完后, 将ACCUMULATE域改为True, 后续累加                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)                        # 通知 TMA warp, 这个 SMEM buffer 的数据已经用完, 可以被新的数据覆盖了.                        handle.release()                        # 继续窥探下一个Tile是否加载完毕                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1                        peek_ab_full_status = cutlass.Boolean(1)                        if handle.count + 1 < k_tile_cnt:                            peek_ab_full_status = ab_consumer.try_wait()                # 通知 Epilogue warp, 当前 TMEM buffer 中的累加结果已经计算完毕, 可以取走处理了.                if is_leader_cta:                    acc_pipeline.producer_commit(acc_producer_state)                acc_producer_state.advance()                # 从Tile Scheduler获取下一个work tile                tile_sched.advance_to_next_work()                work_tile = tile_sched.get_current_work()            # 尾处里, 确保Epilogue完成所有的TMEM中临时数据的处理            acc_pipeline.producer_tail(acc_producer_state)
```

#### 5.4.4 Epilogue Warp

首先根据Epilogue结果是否需要通过RMEM->SMEM--TMA-->GMEM, 创建sC在SMEM中的buffer

```
        sC = None        if cutlass.const_expr(self.use_tma_store):            # (EPI_TILE_M, EPI_TILE_N, STAGE)            sC = smem.allocate_tensor(                element_type=self.c_dtype,                layout=c_smem_layout_staged.outer,                byte_alignment=128,                swizzle=c_smem_layout_staged.inner,            )
```

然后`if warp_idx < self.mma_warp_id:`条件进入Epilogue Warps, 主要流程如下:

**TMEM 分配:** Epilogue warps 负责管理 TMEM 的分配和释放.

**持久化循环:** 同样基于 `tile_sched`.

**区分两种存储路径:** 这两种将在后续详细展开解释.

`epilogue_tma_store`: 如果使用 TMA 存储. 数据路径: TMEM -> RMEM (寄存器) -> SMEM -> GMEM (通过TMA). 这个路径更复杂, 需要额外的 SMEM 作中转站, 并通过 `epilog_sync_barrier` 进行 warp 间的精细同步.

`epilogue`: 如果不使用 TMA 存储. 数据路径: TMEM -> RMEM -> GMEM (直接存储). 这个路径更简单.

**释放 TMEM buffer**

```
        if warp_idx < self.mma_warp_id:                        # 分配TMEM            tmem.allocate(self.num_tmem_alloc_cols)            # 获取TMEM指针, 并构建TMEM中的accumulator Tensor            tmem.wait_for_alloc()            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)            # (MMA, MMA_M, MMA_N, STAGE)            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)            # 创建持久化调度器            tile_sched = utils.StaticPersistentTileScheduler.create(                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()            )            if cutlass.const_expr(self.use_tma_store):                assert tma_atom_c is not None and sC is not None                #  TMEM -> RMEM -> SMEM ---[TMA]---> GMEM的方式                self.epilogue_tma_store(...)            else:                #  TMEM -> RMEM -> GMEM的方式                self.epilogue(...)            # 释放TMEM内存分配锁和释放Buffer            tmem.relinquish_alloc_permit()            tmem.free(tmem_ptr)
```

使用TMA的Epilogue实现如下, 它使用了三级数据拷贝的流水线:

`T->R`: TMEM 到 RMEM

`R->S`: RMEM 到 SMEM

`S->G`: SMEM 到 GMEM (通过 TMA)

函数签名如下:

```
  @cute.jit   def epilogue_tma_store(       self,       # Epilogue warps 内部的线程/warp 索引.       epi_tidx: cutlass.Int32,        warp_idx: cutlass.Int32,        # 与 MMA Warp 同步用的累加器流水线对象.       acc_pipeline: pipeline.PipelineAsync,       tiled_mma: cute.TiledMma,       tma_atom_c: cute.CopyAtom,       # 输入, 指向 TMEM 中分阶段的累加器 C 的张量.       tCtAcc_base: cute.Tensor,       # 中转站, 指向 SMEM 中为 C 准备的张量.       sC: cute.Tensor,       #  输出, 指向 GMEM 中最终结果 C 的 CUTE 张量视图       tCgC: cute.Tensor,       # Epilogue 处理的子块 (subtile) 形状       epi_tile: cute.Tile,       # Tile Scheduler       tile_sched: utils.StaticPersistentTileScheduler,       # 用于Epilogue处理的函数, 例如可以fusion进一个ReLU       #  epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))       epilogue_op: cutlass.Constexpr,   ) -> None:
```

第一步, 初始化和相关的Tensor Partition

```
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(            epi_tidx, tCtAcc_base, tCgC, epi_tile, self.use_2cta_instrs        )        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)        tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(            tiled_copy_t2r, tTR_rC, epi_tidx, sC        )        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)        tCgC_epi = cute.flat_divide(            tCgC[((None, None), 0, 0, None, None, None)], epi_tile        )        # ((ATOM_V, REST_V), EPI_M, EPI_N)        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(            tma_atom_c,            0,            cute.make_layout(1),            cute.group_modes(sC, 0, 2),            cute.group_modes(tCgC_epi, 0, 2),        )
```

首先调用了辅助函数 `epilog_tmem_copy_and_partition`, 创建一个从 TMEM 加载到寄存器 (RMEM) 的拷贝操作 `tiled_copy_t2r` (对应 `tcgen05.ld` 指令). 同时, 划分 TMEM 张量 `tCtAcc_base` 得到 `tTR_tAcc_base` (TMEM 视图), 和一个空的 RMEM 张量 `tTR_rAcc`用于存放加载结果. 然后创建 `tTR_rC`. 这是一个 RMEM 张量, 用于存放从 `tTR_rAcc` (float32) 转换类型后的 C (例如, float16) 的结果.

接着调用辅助函数 `epilog_smem_copy_and_partition`.创建一个从 RMEM 存储到 SMEM 的拷贝操作 `tiled_copy_r2s`. 同时, 划分 `tTR_rC` 得到 `tRS_rC` (RMEM 视图), 和 SMEM 张量 `sC` 得到 `tRS_sC` (SMEM 视图).

最后,`tCgC_epi = cute.flat_divide(...)`: 将 GMEM 上的 C tile 进一步划分为 `epi_tile` 大小的子块. 并通过`cpasync.tma_partition(...)`: 为最终的 TMA 存储操作 (SMEM -> GMEM) 划分源和目标.

`bSG_sC`: TMA 专用的 SMEM 源视图.

`bSG_gC_partitioned`: TMA 专用的 GMEM 目标视图.

第二步, 流水线和同步对象设置

```
        #  创建 `acc_consumer_state`, 用于管理 Epilogue Warp 作为累加器消费者的流水线状态.        acc_consumer_state = pipeline.make_pipeline_state(            pipeline.PipelineUserType.Consumer, self.num_acc_stage        )                # 创建 `c_pipeline`, 这是一个专门为 TMA 存储设计的流水线对象.         # Epilogue warps 在将数据写入 SMEM 后, 会作为此流水线的生产者.        c_producer_group = pipeline.CooperativeGroup(            pipeline.Agent.Thread,            32 * len(self.epilog_warp_id),        )        c_pipeline = pipeline.PipelineTmaStore.create(            num_stages=self.num_c_stage, producer_group=c_producer_group        )        # 创建 `epilog_sync_barrier`, 这是一个NamedBarrier,         # 参与者是所有的 Epilogue warps (4 个 warps * 32 线程/warp).         epilog_sync_barrier = pipeline.NamedBarrier(            barrier_id=self.epilog_sync_bar_id,            num_threads=32 * len(self.epilog_warp_id),        )
```

第三步, 持久化的循环, 这是 Epilogue 的主工作循环, 与 MMA 和 TMA warp 的循环同步进行.

```
        work_tile = tile_sched.initial_work_tile_info()        while work_tile.is_valid_tile:            # 获取 tile 坐标            cur_tile_coord = work_tile.tile_idx            mma_tile_coord_mnl = (                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),                cur_tile_coord[1],                cur_tile_coord[2],            )            # 基于Tile坐标, 切片 bSG_gC            # ((ATOM_V, REST_V), EPI_M, EPI_N)            bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]            # 根据累加器消费者的流水线状态 acc_consumer_state.index,             # 从 TMEM 基地址中切片出当前要处理的那个累加器 buffer (tTR_tAcc).            # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)            tTR_tAcc = tTR_tAcc_base[                (None, None, None, None, None, acc_consumer_state.index)            ]            # 同步点 1. Epilogue Warp 在这里等待, 直到 MMA Warp 调用 producer_commit,             # 表示 TMEM 中的累加器数据已经准备好了.            acc_pipeline.consumer_wait(acc_consumer_state)                        # 重排张量视图            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))            bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))                        # 一个 GEMM tile 内部可能被划分为更小的 subtile(由 epi_tile 定义).             # 下面这个 for 循环遍历所有的 `subtile`.                        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])            num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt            for subtile_idx in cutlass.range(subtile_cnt):            # ... (见下一节的详细分析) ...                        # 在处理完一个 tile 的所有 subtile 后, 进行一次同步.             # 确保所有 epilogue warps 都完成了当前 tile 的工作.            epilog_sync_barrier.arrive_and_wait()            # Epilogue Warp 通知 MMA Warp: "我已经用完这个 TMEM buffer 了, 你可以覆盖它了".            with cute.arch.elect_one():                acc_pipeline.consumer_release(acc_consumer_state)                        # 更新Accumulator Consumer的流水线状态, 准备接收下一个 TMEM buffer.            acc_consumer_state.advance()            # 通过TileScheduler, 获取下一个work tile.            tile_sched.advance_to_next_work()            work_tile = tile_sched.get_current_work()
```

在subtile中的循环处理如下所示:

```
for subtile_idx in cutlass.range(subtile_cnt):    # 执行 `tcgen05.ld` 指令, 将 TMEM 中的累加器数据加载到寄存器 `tTR_rAcc`.    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)    # 类型转换, 并调用epilogue_op, 例如进行ReLU的操作等.    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))    tRS_rC.store(acc_vec)    # 计算当前 subtile 应该使用哪个 SMEM C-buffer stage.    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage        # 数据拷贝 2 (RMEM -> SMEM). 将寄存器中的最终结果 tRS_rC 存储到 SMEM 的 tRS_sC 中.    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])        # 这是一个关键的同步原语. 它确保了 RMEM 到 SMEM 的写操作在后续 TMA 操作开始之前, 对 TMA 硬件单元是可见的.    cute.arch.fence_proxy(        cute.arch.ProxyKind.async_shared,        space=cute.arch.SharedSpace.shared_cta,    )    epilog_sync_barrier.arrive_and_wait()        # TMA操作只需要一个warp issue    if warp_idx == self.epilog_warp_id[0]:        # 指示 TMA 硬件将 SMEM buffer bSG_sC 中的数据拷贝到 GMEM 位置 bSG_gC.        cute.copy(            tma_atom_c,            bSG_sC[(None, c_buffer)],            bSG_gC[(None, subtile_idx)],        )        # 作为 C 流水线的生产者, 提交一个 "TMA 存储已发起" 的任务        c_pipeline.producer_commit()        # 立即尝试获取下一个 C 流水线的许可.        c_pipeline.producer_acquire()    # 所有 Epilogue warps 在这里再次同步.     # 确保 leader warp 已经发起了 TMA 拷贝, 并且其他 warps 等待它完成,     # 之后才能进入下一个 subtile_idx 的循环, 避免对同一个 SMEM buffer 发生Race condition.    epilog_sync_barrier.arrive_and_wait()
```

最后在整个 `while` 主循环结束后, 调用 `producer_tail` 来等待所有已发起的 TMA 存储操作全部完成. 这是一个终结同步点, 保证在Kernel退出前, 所有 C 矩阵的数据都已正确写入 GMEM.

```
c_pipeline.producer_tail()
```

而不带TMA的epilogue函数结构类似, 就不展开讲述了.

### 5.5 性能对比

在[《Tensor-103.1: Basic GEMM》中](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496512&idx=1&sn=a2eb5dfcabea41ea93fe2d482b07661f&scene=21#wechat_redirect)附录有一个CublasLt的测试代码作为baseline, 这里我们对Example中的main函数进行了一点修改.

```
    exec_time = testing.benchmark(        compiled_gemm,        workspace_generator=generate_tensors,        workspace_count=workspace_count,        stream=current_stream,        warmup_iterations=warmup_iterations,        iterations=iterations,    )+    gflops = 2 * m * n * k * l / exec_time / 1e6+    print(f"{m},{n},{k} {gflops:.4f} TFLOPS")
```

执行如下:

```
# 1-CTApython3 static_persistent.py \--ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32 \--mma_tiler_mn 128,128 --cluster_shape_mn 1,1  \--mnkl 4096,4096,4096,1 \--use_tma_store \--warmup_iterations 1000 --iterations 500 \--skip_ref_check # 2-CTApython3 static_persistent.py \--ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32 \--mma_tiler_mn 256,128 --cluster_shape_mn 2,1  \--mnkl 4096,4096,4096,1 \--use_tma_store --use_2cta_instr \--warmup_iterations 1000 --iterations 100 \--skip_ref_check 
```

可以看到在Blackwell上使用2CTA是有很大收益的, 基本上达到了Cublas性能的93%.

| 算法实现 | Jetson Thor(TFLOPS) |
| --- | --- |
| CublasLt | 87.50 |
| CuteDSL-1CTA | 53.86 |
| CuteDSL-2CTA | 81.45 |

参考资料

[1] 
Bringing NVIDIA Blackwell GPU support to LLVM and MLIR: *https://llvm.org/devmtg/2025-04/slides/technical_talk/ozen_blackwell.pdf*
[2] 
PipelineTmaUmma: *https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/pipeline/sm100.py[#L33](javascript:;)*
[3] 
PipelineTmaMultiConsumersAsync: *https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/pipeline/sm90.py*
[4] 
Tutorial GEMM: *https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py*
[5] 
dense_gemm_persistent.py: *https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py*