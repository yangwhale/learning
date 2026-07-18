# 谈谈GPU的内存模型及互联网络设计

> 作者: zartbot  
> 日期: 2025年4月13日 02:31  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&chksm=f995f981cee27097adf7fc3898f214dc3b1a74b5cd204723f6ef82eefdcdab942f5a260bcc1c#rd

---

### TL;DR

其实很多人在谈论ScaleUP和ScaleOut总线的时候, 更多的是在谈论网络这一块, 少了GPU Memory Model的视角. 另一方面老黄讲的:“先要ScaleUP,然后再ScaleOut”,其实上你可以理解成一个销售的话术, 反问一句NV在ScaleOut上今年GTC有啥拿的出手的东西呢?IB交换机都没了声响, 以太网交换机和网卡在RoCE上还有大量的问题...这是实质性的问题. 抛开这些, 其实本质的问题还是在内存上.

![图片](assets/1630c34e31e4.png)

那么今天就做一些笔记,把Nvidia GPU和内存相关的问题从头到尾好好整理一下. 正好GTC25有一个Session《CUDA Techniques to Maximize Memory Bandwidth and Hide Latency》[1], 还找到一个GTC24的Session《Advanced Performance Optimization in CUDA》[2]

再配合Blackwell Tensor Memory引入而产生的内存模型的变化, 和Tile based IR等一系列因素进行一个汇总分析, 恰逢华为UB发布, UALink 1.0标准也发布了, 最后再来谈谈ScaleOut和ScaleUP的内存模型的需求.顺便会谈一下eRDMA在多路径时对于内存模型的实现并对比标准RC和AWS SRD.

另外GTC25的这个session还有两个非常有价值的话题, 低延迟Cluster同步和如何最大化内存带宽, 也会一并介绍, 本文结构如下:

```
1. 内存一致性模型1.1 从一个基础的例子谈起1.2 顺序一致性1.3 Total Store Order1.4 Relaxed Consistency1.5 Cache一致性和Memory Model的区别2. Nvidia GPU Memory Model2.1 Single Thread2.2 Multi-Thread2.2.1 顺序一致性(SC)2.2.2 Acquire2.2.3 Release2.2.4 Relaxed2.2.5 Scope2.3 Async Thread2.4 Async Proxy3. 利用memory order实现低延迟Cluster同步3.1 Thread Block Cluster编程3.2 Cluster低延迟同步3.3 基于DSMEM的组播4. 最大化内存带宽4.1 回顾内存层次化结构4.2 Little's Law4.3 并行优化及异步访问5. LD/ST指令控制Cache5.1 LD指令5.2 ST指令6. ScaleUP和ScaleOut网络设计探讨6.1 谈谈访问内存的Size6.2 访问内存延迟6.3 Memory Model
```

## 1. 内存一致性模型

一致性的本源来自于冯诺伊曼架构中认为："任何的读操作应当返回最近写入的结果"，但是在分布式系统中或者多核CPU系统中， 由于操作的延迟等因素带来了结果的不可预测性。

### 1.1 从一个基础的例子谈起

upenn有一个Sequential Consistency and TSO教程[3]讲的挺好的. 还有一个SPCL的Memory Model[4]的ppt也不错, 更详细的内容可以读读下面这本书

![图片](assets/e89c98e50f0d.png)

首先我们来看有两个Core的处理器, 一个做Producer,一个做Consumer的情况

![图片](assets/5c3e842a033f.png)

是否r2能够获得Core1产生的新的数据呢? 实际执行的时候, 例如Core1的S1和S2出现了ReOrder的情况, 那么Core2的L1会使得L2更早执行, 从而r2拿到老的数据.

![图片](assets/a58780a75707.png)

其实发生Reorder的情况有几类:

![图片](assets/f34f89b383f8.png)

### 1.2 顺序一致性

最直观的内存一致性模型是顺序一致性(Sequential Consistency,SC), 最早的形式化定义来自于Lamport的论文《How	to Make a Multiprocessor Computer that Correctly Executes Multiprocess Programs. IEEE Transactions on Computers, C-28(9):690–91, Sept. 1979》, 对于多核处理器, 本质上就是在MultiCore执行的时候需要保证程序执行的顺序(Program Order)和单核一致.

![图片](assets/a975874cb968.png)

回到问题的本质,其实就是Program Order和Memory Order在不同的Load/Store组合下的约束, 形式化的定义如下所示:

![图片](assets/f1eebc08aac0.png)

维持SC的实质是, 要么采用单核运行, 要么在内存访问的时候, 通过选择保证内存的顺序访问

![图片](assets/f853bf2ccfb8.png)

即在每个时间步，switch 选择要运行的线程，并完全运行其下一个事件。此模型保留了顺序一致性的规则, 但是它的最大的问题就是灾难性的慢.我们一次只能运行一条指令,因此我们已经失去了多个线程并行运行的大部分好处.

更糟糕的是,我们必须等待每条指令完成,然后才能开始下一条指令:在当前指令的效果对其他所有线程可见之前,不能再运行更多指令.

### 1.3 Total Store Order

对于一个处理器来看, 直接等待内存写入会使得Store操作太慢, 因此通常会设计一个Store buffer用于隐藏延迟, 避免stall. 对于MultiCore的处理器各自拥有独立的Store buffer.

![图片](assets/534807ad62c5.png)

但是在这种情况下, 如上图所示, 两个Core都有可能读到旧的值.

![图片](assets/3cefbc278386.png)

但是这种取舍带来的性能收益是巨大的. 这也就是出现Total Store Order(TSO)的原因, 形式化的定义就是它放弃了Store->Load的保序要求, 支持Store Buffer的设计

![图片](assets/4a9daedf9154.png)

而对于Store->Load可以使用FENCE来解决问题. 其实FENCE的实现也很简单, 例如排空store buffer保证主存Read-Write coherent.

### 1.4 Relaxed Consistency

更进一步, 我们是否能够允许更多的Reorder提高程序执行的并行性? 然后通过Fence(Memory Barrier)来保证程序的执行顺序的正确性.

![图片](assets/26b9cf76ffab.png)

另外工业界对于Relaxed Consistency还有一些定义, Total Store Order正如前讲述放弃了, Partial Store Order(PSO)放弃了和约束. 而一些Relaxed Memory Order(RMO)则完全放弃了四条约束., 其实很多处理器都是支持的, 而GPU本身也是一个Relaxed Consistency的系统.

![图片](assets/90d1edd7ec0e.png)

### 1.5 Cache一致性和Memory Model的区别

其实这是很多人容易搞混的地方, Cache一致性主要是一种如何将Store按需传递到其它处理器的机制,使得写可以按需的在其它处理器可见. 而Memory Model更多的是定义操作传递到其它处理器的顺序边界.

## 2. Nvidia GPU Memory Model

Nvidia GPU的内存层次结构如下图所示, 对于多达数千个Cuda Core而言, 保证内存TSO(Total Store Order)的代价会非常大, 因此在nvidia GPU上实现了Partial Store Order的内存模型.

![图片](assets/e06684bdc398.png)

对于该问题, 不同的架构还有一些细微的区别, 例如英乙己自己讲的内存模型的四种写法:)

![图片](assets/0ba9cdf0bfc4.png)

### 2.1 Single Thread

对于单个线程而言, 相同地址的LD/ST是保序的, 如下所示:

![图片](assets/6984789f1464.jpg)

但是这里有一个例外, 我们先看下面这个程序, 猜猜输出结果, 其实行为是未定义的.

```
#include <iostream>#include <cuda.h>__constant__ int val = 1;__global__ void kernel_constant_sc(){int tid = threadIdx.x + blockDim.x * threadIdx.y;if (tid != 0)  {    printf("Thread %d, val %d\n", tid, val); //load val to Const$  } else {    //remove constant    int *mut_val = const_cast<int *>(&val);    asm volatile("" : "+l"(mut_val));    //store new value    *mut_val = 42;  }}int main(int argc, const char *argv[]){int n = 2;if (argc ==2 ) {    n = strtol(argv[1],NULL,10);  }  kernel_constant_sc<<<1, n >>>();  cudaDeviceSynchronize();return0;}
```

这是由于在SM内有一块Read-Only Cache, 常量会放在这个空间内. 它和L2Cache有独立的数据路径, 因此会导致问题

![图片](assets/408d5d684313.jpg)

修改这类值时会产生未定义的行为.

![图片](assets/5bb960ea7cd1.jpg)

### 2.2 Multi-Thread

我们再来回顾一下,顺序一致性(SC)需要保证如下四条规则

Total Store Order(TSO)为了引入Store Buffer放弃了第四条规则, 并且在单个核内部可以通过Bypass Load的方式读取Write Buffer, 核间则引入Fence的方式.

但是GPU内有大量的CudaCore, 对于数千个核需要对内存操作进行保序将会对指令并行执行和数据并行都带来非常大的性能影响, 因此在GPU内维持TSO会带来非常大的代价. 因此更恰当的做法是支持Relax Order并采用ATOMIC和FENCE的方式处理. 在GPU内, Nvidia支持4种模式

![图片](assets/bb8c264d49aa.jpg)

#### 2.2.1 顺序一致性(SC)

如下图所示, 顺序一致性要求LD/ST都不能移动到某个指定操作的前后. 这种方式虽然非常容易编程,但是性能会很慢.

![图片](assets/3b1906fc42f1.jpg)

具体来看, 我们将如下这段代码产生PTX指令进行分析

```
__global__ void kernel_seq_constant(int* array){  cuda::atomic<int> a;int val;//prior load/storeint before = array[0];array[0] = 3;//atomic load  val = a.load(cuda::std::memory_order_seq_cst);//Later loadint after = array[0];printf("before %d, after %d, val %d",before,after,val);}int main(int argc, const char *argv[]){int *array ;  cudaMalloc(&array,sizeof(int)*4);  kernel_seq_constant<<<1, 2>>>(array);  cudaDeviceSynchronize();return0;}
```

在PTX指令中我们可以看到`fence.sc.sys`阻止了Prior load/store在Atomic后面执行. 同时atomic load采用`ld.acquire`指令阻止了后续的LD/ST指令在该指令之前执行.

```
ld.global.u32  %r3, [%rd3];   //before = array[0]st.global.u32  [%rd3], %r2;  //array[0] =3// begin inline asmfence.sc.sys; //阻止后续LD/ST指令提前执行// end inline asmadd.u64  %rd1, %SP, 0;// begin inline asmld.acquire.sys.b32 %r1,[%rd1];//acquire阻止后续LD/ST指令提前执行// end inline asmld.global.u32  %r4, [%rd3]; //after = array[0]
```

#### 2.2.2 Acquire

当把`val = a.load(cuda::std::memory_order_seq_cst)`改为`val = a.load(cuda::std::memory_order_acquire)`后, 我们查看PTX指令可以发现`fence.sc.sys`被移除了

```
ld.global.u32  %r3, [%rd3];//before = array[0]st.global.u32  [%rd3], %r2;//array[0] =3add.u64  %rd1, %SP, 0;// begin inline asmld.acquire.sys.b32 %r1,[%rd1];//acquire阻止后续LD/ST指令提前执行// end inline asmld.global.u32  %r4, [%rd3]; //after = array[0]
```

此时允许Atomic Load之前的LD/ST指令在atomic之后执行, 但是对于atomic之后的Later Load指令则会继续阻塞.

![图片](assets/7845b2c85de5.jpg)

#### 2.2.3 Release

那么既然有阻塞后面的acquire,是否能有阻塞前面LD/ST,而不阻塞后面的内存模型呢? 那就是Release模式, 如下所示:

![图片](assets/22edab4a5c8b.png)

```
__global__ void kernel_release(int* array){  cuda::atomic<int> a;//Prior LD/STint before = array[0];array[0] = 3;// atomic store.release  a.store(1, cuda::std::memory_order_release);//Later Loadint after = array[0];printf("before %d, after %d",before,after);}
```

观察PTX指令我们可以看到采用了`st.release`指令,该指令可以阻塞前序的LD/ST, 但允许`Later Load`提前执行.

```
ld.global.u32  %r3, [%rd3]; //before = array[0]st.global.u32  [%rd3], %r2; //array[0] = 3mov.u32  %r1, 1;add.u64  %rd1, %SP, 0;// begin inline asmst.release.sys.b32 [%rd1], %r1; //Store.release// end inline asmld.global.u32  %r4, [%rd3]; //Later Load
```

#### 2.2.4 Relaxed

最后就是一个最宽松的Relaxed内存模型, 它前后的LD/ST都可以乱序执行.

![图片](assets/d7d15f03c14c.png)

代码如下:

```
__global__ void kernel_relaxed(int* array){  cuda::atomic<int> a;//Prior LD/STint before = array[0];array[0] = 3;// atomic store.release  a.store(1, cuda::std::memory_order_relaxed);//Later Loadint after = array[0];printf("before %d, after %d",before,after);}PTX:ld.global.u32  %r3, [%rd3]; //before = array[0]st.global.u32  [%rd3], %r2; //array[0] = 3mov.u32  %r1, 1;add.u64  %rd1, %SP, 0;// begin inline asmst.relaxed.sys.b32 [%rd1], %r1; //Store.relaxed// end inline asmld.global.u32  %r4, [%rd3];//Later Load
```

#### 2.2.5 Scope

我们注意到前述的指令中都有`.sys`属性, 实际上它可以根据用户的需求选择不同的范围(scope)进行处理, 在CUDA C++的API中定义了如下几种范围

![图片](assets/22414a6f13bc.jpg)

在PTX中的Scope定义如下:

![图片](assets/7622c0ee8df3.jpg)

回顾一下NV GPU的内存层次结构, Block Scope是在SM内部基于L1 Cache保持一致性

![图片](assets/454926aec6be.jpg)

Cluster Scope则是在Thread Block Cluster Level, 从硬件上来看是在GPC内基于L2Cache维持一致性

![图片](assets/c74ac7359587.jpg)

Device Scope则是整颗GPU芯片上所有的SM基于L2维持一致性

![图片](assets/21f12f6a31c3.jpg)

Sys Scope则是包含了整个系统

![图片](assets/2eb3335128ed.jpg)

先来看一个简单的block_scope的例子,代码如下:

```
#include <iostream>#include <cuda.h>#include <cuda/atomic>#define CUDAASSERT(condition)                         \    if (!(condition))                                 \    {                                                 \        printf("Assertion %s failed!\n", #condition); \    }__device__ void producer(    cuda::atomic_ref<int, cuda::thread_scope_block> val){    val.store(42, cuda::memory_order_relaxed);}__device__ void consumer(    cuda::atomic_ref<int, cuda::thread_scope_block> val){    volatileint  tmp = -1;    while (tmp == -1)    {        tmp = val.load(cuda::memory_order_relaxed);    }    CUDAASSERT(tmp == 42);}__global__ void kernel_scope_test(int *array){    if (blockIdx.x == 0)    {        producer(array[0]);    }    else    {        consumer(array[0]);    }}int main(int argc, const char *argv[]){    int *array;    cudaMalloc(&array, sizeof(int) * 4);    dim3 grid(2, 1);    kernel_scope_test<<<grid, 1>>>(array);    cudaDeviceSynchronize();    return0;}
```

由于定义了`cuda::atomic_ref<int, cuda::thread_scope_block> val`, 查看PTX指令可以看到LD/ST relaxed的scope修饰为`.cta`

```
Consumer: mov.u64  %rd6, %rd15;// begin inline asm ld.relaxed.cta.b32 %r3,[%rd6];// end inline asm    Producer: mov.u64  %rd12, %rd15; mov.u32  %r5, 42;// begin inline asm st.relaxed.cta.b32 [%rd12], %r5;// end inline asm    
```

因此运行结果如下图所示, 另一个block的LD在另一个数据路径上, 没有被阻塞

![图片](assets/20c61fa2f22f.jpg)

然后我们把Scope扩大到device level, 如下所示, 然后可以看到PTX的scope已经变成了`.gpu`

```
__device__ void producer(    cuda::atomic_ref<int, cuda::thread_scope_device> val){    val.store(42, cuda::memory_order_relaxed);}__device__ void consumer(    cuda::atomic_ref<int, cuda::thread_scope_device> val){    int tmp = -1;    while (tmp == -1)    {        tmp = val.load(cuda::memory_order_relaxed);    }    CUDAASSERT(tmp == 42);}PTX:Consumer: mov.u64  %rd6, %rd15;// begin inline asm ld.relaxed.gpu.b32 %r3,[%rd6];// end inline asm setp.eq.s32  %p2, %r3, -1; @%p2 bra  $L__BB0_2;Producer: mov.u64  %rd12, %rd15; mov.u32  %r5, 42;// begin inline asm st.relaxed.gpu.b32 [%rd12], %r5;// end inline asm
```

但是程序并没有像Nvidia GTC25这个ppt那样正常的工作, 为什么呢?

![图片](assets/75ae3eb8e61f.png)

我们注意到即便是Scope为整个GPU了, 但是GTC25这个演讲者可能有一个typo, relaxed order的问题, 当然这并不是关键. 其实一个更加正确的做法是采用flag 并使用release/accquire来处理.

```
#include <iostream>#include <cuda.h>#include <cuda/atomic>#define CUDAASSERT(condition)                         \    if (!(condition))                                 \    {                                                 \        printf("Assertion %s failed!\n", #condition); \    }__device__ void producer(    int &val,    cuda::atomic_ref<int, cuda::thread_scope_device> flag){    val = 42;    flag.store(42, cuda::memory_order_relaxed);}__device__ void consumer(    int &val,    cuda::atomic_ref<int, cuda::thread_scope_device> flag){    while (flag.load(cuda::memory_order_acquire) != -1)    {    }    int tmp = val;    CUDAASSERT(tmp == 42);}__global__ void kernel_scope_test(int *array){    array[0] = 0;    int flag = -1;    __syncthreads();    if (blockIdx.x == 0)    {        producer(array[0], flag);    }    else    {        consumer(array[0], flag);    }}int main(int argc, const char *argv[]){    int *array;    cudaMalloc(&array, sizeof(int) * 4);    dim3 grid(2, 1);    kernel_scope_test<<<grid, 1>>>(array);    cudaDeviceSynchronize();    cudaFree(&array);    return0;}
```

![图片](assets/b4470ded66f3.png)

Relaxed 和Acquire-Release对比如下, Relaxed会更快一些, 当两个线程只需要交换一个值时有用,而Release-Acquire因为涉及flush cache会更慢, 当多个线程要交换多个值时更有用

![图片](assets/a96a3b1e03fc.png)

### 2.3 Async Thread

在Ampere这一代开始引入了Async Thread编程能力,主要是实现了异步的拷贝机制从GMEM LD到SMEM,或者从SMEM ST到GMEM, 避免了寄存器和L1占用同时还增加了处理的大数据量时的吞吐

![图片](assets/3dbe5f7d5dee.png)

![图片](assets/cf7f9262f8c8.png)

详细的编程实现可以参考[《Tensor-004 TensorCore编程及优化》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491529&idx=1&sn=12902726d6d9a8f9d66405ac6ea42fa7&scene=21#wechat_redirect)的内容. 通过它可以Overlap数据加载和计算的延迟,大致的流程如下

![图片](assets/3d21cf2c3736.png)

首先采用异步的Prefetch

![图片](assets/e9808fefaaf8.png)
然后执行计算

![图片](assets/1f63ec1fb518.png)

大致的计算流程如下, 详细的代码可以查看`https://github.com/zartbot/tensorcore_gemm/blob/main/05_pipeline_gmem_to_smem.cu`

```
Async Copy A-Chunk from GMEM-->SMEM(Buffer_1)Async Copy B-Chunk from GMEM-->SMEM(Buffer_1)Wait for Async Copy Completionfor (size_t tile_k = CHUNK_K; tile_k < K_tiles; tile_k += CHUNK_K) {   Swap Buffer_1/Buffer_2 Offset   //异步加载Buffer-2 并同时计算Buffer-1进行Overlap   Async Copy A-Chunk from GMEM-->SMEM(Buffer_2)   Async Copy B-Chunk from GMEM-->SMEM(Buffer_2)      for (size_t k_step = 0; k_step < CHUNK_K; ++k_step){        for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)         {            Load-SMEM(Buffer_1)-to-A_fragment            for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)            {               Load-SMEM(Buffer_1)-to-B_fragment               wmma::mma_sync;  //使用TensorCore计算            }        }    }    Wait for Async Copy Completion}Calculate Last Buffer WarpTileWMMA-Store-to-SMEMStore-SMEM->GMEM
```

在Hopper上还引入了st.async的指令用于DSMEM上的数据存储, 详细的测试代码参见第三章.

![图片](assets/031e98fffb0f.png)

但是这些异步的操作引入产生了一个新的问题, 当出现一条异步的数据路径后, 便会产生Data Race, 因此需要格外的小心.

![图片](assets/f108002999b2.png)

### 2.4 Async Proxy

由于在内存层次化结构中存在多个数据路径, 特别是在Hopper开始引入了TMA,以及在Blackwell中引入的TensorMemory. 因此对于不同的数据路径的Data Race Condition需要更好的管理和抽象, 在Hopper开始引入了Async Proxy, 并通过General Proxy和Async Proxy来区分不同的内存访问路径

![图片](assets/d76ebe2ef9a4.png)

通过这样的区分可以对async proxy的内存操作进行fence

![图片](assets/d8b3f635b862.png)

反过来,对于Async Proxy的操作, 通常有一个memory barrier, general proxy的LD/ST可以wait这个barrier完成

![图片](assets/100ad5efdae5.png)

具体流程如下, 首先需要在SMEM内申请一个mbarrier , 然后一个线程issue TMA指令(UBLKCP), 需要注意的是,拷贝到SMEM的数据必须alignas(16) bytes.

![图片](assets/924403138a64.png)

完成采用completion_tx计数器的方式

![图片](assets/3cafcbfa9001.png)

我们来看一个基于TMA-2D的例子

```
#include <cuda.h>#include <cudaTypedefs.h>#include <cuda/barrier>#include <iostream>#pragma nv_diag_suppress static_var_with_dynamic_initusingbarrier_t = cuda::barrier<cuda::thread_scope_block>;namespace cde = cuda::device::experimental;constexprsize_t GLOBAL_M = 64;constexprsize_t GLOBAL_K = 32;constexprsize_t TILE_M = 8;constexprsize_t TILE_K = 16;inline PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled(){    cudaDriverEntryPointQueryResult driver_status;    void *cuTensorMapEncodeTiled_ptr = nullptr;    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,                                     cudaEnableDefault, &driver_status);    if (driver_status != cudaDriverEntryPointSuccess)        throwstd::runtime_error("driver_status != cudaDriverEntryPointSuccess");    returnreinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);}CUtensorMap make_2d_tma_desc(int32_t *global_address,                             uint64_t global_dim[2], uint64_t stride,                             uint32_t smem_dim[2],                             CUtensorMapSwizzle swizzle){    CUtensorMap tensor_map = {};    uint64_t global_stride[1] = {stride};    uint32_t elem_stride[2] = {1, 1};    auto encode = get_cuTensorMapEncodeTiled();    auto res = encode(        &tensor_map,        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,        2, // rank =2        global_address,        global_dim,        global_stride,        smem_dim,        elem_stride,        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,        swizzle,        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);    assert(res == CUDA_SUCCESS && "make tma descriptor failed.");    return tensor_map;}__global__ void tma_kernel(const __grid_constant__ CUtensorMap tensor_map,                           uint32_t x, uint32_t y){    __shared__ alignas(128) int tile_smem[TILE_M * TILE_K];    __shared__ barrier_t bar;    // 初始化Barrier    if (threadIdx.x == 0)    {        init(&bar, blockDim.x);        // 由于TMA调用路径为async proxy, 需要fence保持可见        // cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);        cde::fence_proxy_async_shared_cta(); // b)    }    __syncthreads();        barrier_t::arrival_token token;    if (threadIdx.x == 0)    {        //执行TMA拷贝        cde::cp_async_bulk_tensor_2d_global_to_shared(tile_smem, &tensor_map, x, y, bar);        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile_smem));    }    else    {        token = bar.arrive();    }    // 做一些其它的事情....    int value = threadIdx.x * 100 + threadIdx.x;    // 等待所有数据到达    bar.wait(std::move(token));    printf("[tma_kernel] threadIdx.x %d arrived\n", threadIdx.x);    for (int i = 0; i < TILE_M * TILE_K; i += blockDim.x)    {        tile_smem[i + threadIdx.x] += value;    }    //需要fence async proxy等待SMEM保存结束    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);    __syncthreads();    if (threadIdx.x == 0)    {        //TMA从SMEM拷贝到GMEM        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y,                                                      tile_smem);        cuda::ptx::cp_async_bulk_commit_group();        //等待所有Group完成        cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());    }    printf("thread %d done\n", threadIdx.x);}int main(int argc, char **argv){    uint64_t global_dim[2] = {GLOBAL_M, GLOBAL_K};    size_t GLOBAL_SIZE = GLOBAL_K * GLOBAL_M;    uint32_t tile_dim[2] = {TILE_M, TILE_K};    int h_data[GLOBAL_SIZE];    for (size_t i = 0; i < GLOBAL_SIZE; ++i)    {        h_data[i] = 1;    }    // Malloc memory on GPU6 and allow P2P    cudaSetDevice(6);    cudaDeviceEnablePeerAccess(7, 0);    int *d_data;    cudaMalloc(&d_data, GLOBAL_SIZE * sizeof(int));    cudaMemcpy(d_data, h_data, GLOBAL_SIZE * sizeof(int), cudaMemcpyHostToDevice);    //使用GPU7跨NVLINK测试TMA    cudaSetDevice(7);    cudaDeviceEnablePeerAccess(6, 0);    CUtensorMap tensor_map = make_2d_tma_desc(        d_data, global_dim, GLOBAL_K * sizeof(int),        tile_dim,        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);    uint32_t coord_x = 16;    uint32_t coord_y = 16;    tma_kernel<<<1, TILE_M * TILE_K>>>(tensor_map, coord_x, coord_y);    cudaDeviceSynchronize();    cudaError_t err = cudaGetLastError();    std::cout << cudaGetErrorString(err) << std::endl;    cudaSetDevice(6);    cudaMemcpy(h_data, d_data, GLOBAL_SIZE * sizeof(int),               cudaMemcpyDeviceToHost);    for (size_t i = 0; i < GLOBAL_M; ++i)    {        for (size_t j = 0; j < GLOBAL_K; ++j)        {            printf("%5d ", h_data[i * GLOBAL_K + j]);        }        printf("\n");    }    cudaFree(d_data);    return0;}
```

需要注意的是在不同的内存异步拷贝时的完成机制是不同的

![图片](assets/618dfab7ded3.png)

另外对于mbarrier, 有些采用的是completion.tx计数的方式, 另外还有一些采用waitgroup的方式.

最后作者进行了一个总结, st.async / red.async/ cp.async 在实现上更早, 数据路径上并不支持async.proxy. 而TMA/TMEM/WGMMA则是支持async proxy.

![图片](assets/34060f2e6a02.png)

## 3. 利用memory order实现低延迟Cluster同步

### 3.1 Thread Block Cluster编程

公开的关于Hopper的Cluster编程资料其实并不多, 在《cuda c programming guide》[5]中有些介绍. 它由Hopper引入的一个新的层次化结构. 在Hopper内构建了局部的SM-to-SM数据路径, 并提供了Distribute Shared Memory (DSMEM)的概念

![图片](assets/c282f3be8b25.png)

从软件界面上来看就是在Grid和Block之间新引入了一层Cluster

![图片](assets/2249f56afccd.png)

一个简单的示例代码如下, 通过在Kernel函数定义`__cluster_dims__(x, y, z)`决定cluster的形状,  并且可以通过cg::this_cluster()函数获得当前cluster的描述符.需要注意的是考虑到可移植性在单个Cluster内仅支持最多8个Thread Block, 但是按照Hopper的DataSheet看, H100有8个GPC 132个SM, 也就是说单个Cluster最多可以支持16个. 而在H20上由于SM被阉割, 测试最多只支持8个.

```
#include <iostream>#include <cuda.h>#include <cuda/atomic>#include <cooperative_groups.h>namespace cg = cooperative_groups;__global__ void __cluster_dims__(4, 2, 1) kernel_cluster_test(){    cg::cluster_group cluster = cg::this_cluster();    unsignedint cluster_block_rank = cluster.block_rank();    printf("ThreadIdx [%d,%d,%d], BlockDIM [%d,%d,%d], BlockIdx [%d,%d,%d] Cluster rank %d dim [%d,%d,%d] idx [%d,%d,%d] GridDim [%d,%d,%d]\n",           threadIdx.x, threadIdx.y, threadIdx.z,           blockDim.x, blockDim.y, blockDim.z,           blockIdx.x, blockIdx.y, blockIdx.z,           cluster.block_rank(), cluster.dim_blocks().x, cluster.dim_blocks().y, cluster.dim_blocks().z,           cluster.block_index().x, cluster.block_index().y, cluster.block_index().z,           gridDim.x, gridDim.y, gridDim.z);}int main(int argc, const char *argv[]){    dim3 grid(4, 8, 1);    dim3 block(4, 4, 4);    kernel_cluster_test<<<grid, block>>>();    cudaError_t err = cudaGetLastError();    std::cout << cudaGetErrorString(err) << std::endl;    cudaDeviceSynchronize();    return0;}
```

当然除了__cluster_dims__还可以通过`cudaLaunchKernelEx`函数在runtime决定clusterdim.

```
__global__ void kernel_cluster_test(int var1, int var2){}int main(){    dim3 grid(4, 8, 1);    dim3 block(4, 4, 4);        cudaLaunchConfig_t config = {0};    config.gridDim = grid;    config.blockDim = block;    cudaLaunchAttribute attribute[1];    attribute[0].id = cudaLaunchAttributeClusterDimension;    attribute[0].val.clusterDim.x = 2;    attribute[0].val.clusterDim.y = 1;    attribute[0].val.clusterDim.z = 1;    config.attrs = attribute;    config.numAttrs = 1;    cudaLaunchKernelEx(&config, kernel_cluster_test, var1, var2);}
```

cluster 还有一些其它的函数在cuda c programming guide中Cluster group[6]章节介绍

![图片](assets/0cafcddd23ad.png)

在Cluster中最重要的场景是使用Distributed Shared Memory, 它在Cluster内部可以实现bypass L2的低延迟的SMEM互相访问, 并支持LD/ST,ATOMIC, async DMA等操作.

![图片](assets/55a8efb9f098.png)

例如下面这个示例

```
#include <cstdio>#include <iostream>#include <cuda/ptx>#include <cuda/barrier>#include <cooperative_groups.h>namespace cg = cooperative_groups;__global__ void __cluster_dims__(8, 1, 1) kernel(){  cg::cluster_group cluster = cg::this_cluster();//申明并初始化SMEM  __shared__ int smem_x[32];  smem_x[threadIdx.x] = blockIdx.x * 10000 + threadIdx.x;//在cluster范围内同步并确保所有的线程都完成共享内存申明和初始化  cluster.sync();int peer_rank = cluster.block_rank() ^1;int *dst_mem = cluster.map_shared_rank(smem_x,peer_rank);  dst_mem[threadIdx.x] += cluster.block_rank() * 100;  cluster.sync();printf("threadIdx %d blockIdx %d clusterRank %d smem: %d\n", threadIdx.x,blockIdx.x,cluster.block_rank(), smem_x[threadIdx.x]);}int main() {  kernel<<<8, 4>>>();  cudaDeviceSynchronize();return0;}#执行结果如下:threadIdx 0 blockIdx 6 clusterRank 6 smem: 60700threadIdx 1 blockIdx 6 clusterRank 6 smem: 60701threadIdx 2 blockIdx 6 clusterRank 6 smem: 60702threadIdx 3 blockIdx 6 clusterRank 6 smem: 60703threadIdx 0 blockIdx 7 clusterRank 7 smem: 70600threadIdx 1 blockIdx 7 clusterRank 7 smem: 70601threadIdx 2 blockIdx 7 clusterRank 7 smem: 70602threadIdx 3 blockIdx 7 clusterRank 7 smem: 70603threadIdx 0 blockIdx 0 clusterRank 0 smem: 100threadIdx 1 blockIdx 0 clusterRank 0 smem: 101threadIdx 2 blockIdx 0 clusterRank 0 smem: 102threadIdx 3 blockIdx 0 clusterRank 0 smem: 103threadIdx 0 blockIdx 1 clusterRank 1 smem: 10000threadIdx 1 blockIdx 1 clusterRank 1 smem: 10001threadIdx 2 blockIdx 1 clusterRank 1 smem: 10002threadIdx 3 blockIdx 1 clusterRank 1 smem: 10003threadIdx 0 blockIdx 2 clusterRank 2 smem: 20300threadIdx 1 blockIdx 2 clusterRank 2 smem: 20301threadIdx 2 blockIdx 2 clusterRank 2 smem: 20302threadIdx 3 blockIdx 2 clusterRank 2 smem: 20303threadIdx 0 blockIdx 3 clusterRank 3 smem: 30200threadIdx 1 blockIdx 3 clusterRank 3 smem: 30201threadIdx 2 blockIdx 3 clusterRank 3 smem: 30202threadIdx 3 blockIdx 3 clusterRank 3 smem: 30203threadIdx 0 blockIdx 4 clusterRank 4 smem: 40500threadIdx 1 blockIdx 4 clusterRank 4 smem: 40501threadIdx 2 blockIdx 4 clusterRank 4 smem: 40502threadIdx 3 blockIdx 4 clusterRank 4 smem: 40503threadIdx 0 blockIdx 5 clusterRank 5 smem: 50400threadIdx 1 blockIdx 5 clusterRank 5 smem: 50401threadIdx 2 blockIdx 5 clusterRank 5 smem: 50402threadIdx 3 blockIdx 5 clusterRank 5 smem: 50403
```

Thread Block Cluster的优势是可以通过SM-to-SM网络交互数据避免将数据存入L2/GMEM.
![图片](assets/ce8829ff4864.png)

### 3.2 Cluster低延迟同步

但是前面的实现Cluster::sync()有性能瓶颈, 它会使得的整个cluster同步, 让LD/ST的数据在cluster中的其它thread可见, 这样会数据穿越L2. 实际上在PTX指令中 cluster::sync()会产生连续两条

```
 barrier.cluster.arrive; barrier.cluster.wait;
```

但是可以通过PTX指令来分离arrive和wait, 并且有选择的采用release/relaxed选择LD/ST的可见性

![图片](assets/f5f272b274fb.png)

例如初始化barrier的时候, 我们可以采用cluster::sync()的方式, 虽然简单, 但是由于穿越L2Cache会比较慢

![图片](assets/21b79d454263.png)

我们可以采用release-acquire的方法

![图片](assets/8e0042308ba2.png)

然后Cluster内SM-SM的通信可以采用异步store的方式, 并等到local mbarrier

![图片](assets/a8395f0b7bad.png)

整个过程代码如下所示:

```
#include <cstdio>#include <cuda/ptx>#include <cuda/barrier>#include <cooperative_groups.h>namespace cg = cooperative_groups;using cuda::ptx::scope_cluster;using cuda::ptx::sem_acquire;using cuda::ptx::sem_relaxed;using cuda::ptx::sem_release;using cuda::ptx::space_cluster;using cuda::ptx::space_shared;namespace ptx{    __device__ __forceinline__ uint32_t __as_ptr_smem(constvoid *__ptr)    {        returnstatic_cast<uint32_t>(__cvta_generic_to_shared(__ptr));    }    __device__ __forceinline__ void mbarrier_init(uint64_t *mbar, const uint32_t count)    {        uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);        asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mbar_ptr), "r"(count) : "memory");    }    __device__ __forceinline__ void fence_mbarrier_init(cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t)    {        asm volatile("fence.mbarrier_init.release.cluster; // 3." : : : "memory");    }    __device__ __forceinline__ void barrier_cluster_arrive(cuda::ptx::sem_relaxed_t)    {        asm volatile("barrier.cluster.arrive.relaxed;" : : :);    }    __device__ __forceinline__ void barrier_cluster_wait(cuda::ptx::sem_acquire_t)    {        asm volatile("barrier.cluster.wait.acquire;" : : : "memory");    }    __device__ __forceinline__ void barrier_cluster_wait()    {        asm volatile("barrier.cluster.wait;" : : : "memory");    }    template <cuda::ptx::dot_scope Scope>    __device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(        cuda::ptx::sem_relaxed_t,        cuda::ptx::scope_t<Scope> __scope,        cuda::ptx::space_shared_t,        uint64_t *__addr,        const uint32_t &__txCount)    {        uint64_t __state;        if constexpr (__scope == cuda::ptx::scope_cta)        {            asm("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 %0, [%1], %2;"                : "=l"(__state)                : "r"(__as_ptr_smem(__addr)), "r"(__txCount)                : "memory");        }        elseifconstexpr (__scope == cuda::ptx::scope_cluster)        {            asm("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 %0, [%1], %2;"                : "=l"(__state)                : "r"(__as_ptr_smem(__addr)), "r"(__txCount)                : "memory");        }        return __state;    }    template <cuda::ptx::dot_scope Scope>    __device__ __forceinline__ bool mbarrier_try_wait(        cuda::ptx::sem_acquire_t, cuda::ptx::scope_t<Scope> __scope, uint64_t *__addr, const uint64_t &__state)    {        uint32_t __waitComplete;        if constexpr (__scope == cuda::ptx::scope_cta)        {            asm("{\n\t .reg .pred P_OUT; \n\t"                "mbarrier.try_wait.acquire.cta.shared::cta.b64         P_OUT, [%1], %2;                        // 6a. \n\t"                "selp.b32 %0, 1, 0, P_OUT; \n"                "}"                : "=r"(__waitComplete)                : "r"(__as_ptr_smem(__addr)), "l"(__state)                : "memory");        }        elseifconstexpr (__scope == cuda::ptx::scope_cluster)        {            asm("{\n\t .reg .pred P_OUT; \n\t"                "mbarrier.try_wait.acquire.cluster.shared::cta.b64         P_OUT, [%1], %2;                        // 6a. \n\t"                "selp.b32 %0, 1, 0, P_OUT; \n"                "}"                : "=r"(__waitComplete)                : "r"(__as_ptr_smem(__addr)), "l"(__state)                : "memory");        }        returnstatic_cast<bool>(__waitComplete);    }}__global__ void __cluster_dims__(8, 1, 1) low_latency_kernel(int iter_num){    cg::cluster_group cluster = cg::this_cluster();    __shared__ int receive_buffer[4];    __shared__ uint64_t bar;    // 初始化barrier    if (threadIdx.x == 0)    {        ptx::mbarrier_init(&bar, blockDim.x);    }    // make barrier visible    ptx::fence_mbarrier_init(sem_release, scope_cluster);    ptx::barrier_cluster_arrive(sem_relaxed);    ptx::barrier_cluster_wait(sem_acquire);    // 获取远端的buffer和barrier地址:    unsignedint peer_rank = cluster.block_rank() ^ 1;    uint64_t *remote_bar = cluster.map_shared_rank(&bar, peer_rank);    int *remote_buffer = cluster.map_shared_rank(&receive_buffer[0], peer_rank);    for (int iter = 0; iter < iter_num; ++iter)    {        cuda::ptx::st_async(remote_buffer, {iter, iter, iter, iter}, remote_bar);        // relaxed        uint64_t token = ptx::mbarrier_arrive_expect_tx(            sem_relaxed,             scope_cluster,            space_shared,            &bar,            sizeof(receive_buffer));        bool ready = false;        while (!ready)        {            //acquire            ready = ptx::mbarrier_try_wait(                sem_acquire,                 scope_cluster,                &bar,                token);        }        ptx::barrier_cluster_arrive(sem_relaxed);        ptx::barrier_cluster_wait();    }}__global__ void __cluster_dims__(8, 1, 1) standard_async_kernel(int iter_num){    cg::cluster_group cluster = cg::this_cluster();    usingbarrier_t = cuda::barrier<cuda::thread_scope_block>;    __shared__ int receive_buffer[4];    __shared__ barrier_t bar;    init(&bar, blockDim.x);    // make barrier visible    cluster.sync();    // 获取远端的buffer和barrier地址:    unsignedint other_block_rank = cluster.block_rank() ^ 1;    uint64_t *remote_bar = cluster.map_shared_rank(cuda::device::barrier_native_handle(bar), other_block_rank);    // int * remote_buffer = cluster.map_shared_rank(&receive_buffer, other_block_rank);    int *remote_buffer = cluster.map_shared_rank(&receive_buffer[0], other_block_rank);    for (int iter = 0; iter < iter_num; ++iter)    {        // Arrive on local barrier:        uint64_t arrival_token;                //sem_release        arrival_token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, cuda::device::barrier_native_handle(bar), sizeof(receive_buffer));        cuda::ptx::st_async(remote_buffer, {iter, iter, iter, iter}, remote_bar);        // Wait on local barrier:        while (!cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cluster, cuda::device::barrier_native_handle(bar), arrival_token))        {        }    }}int main(){    cudaEvent_t start, stop;    cudaEventCreate(&start);    cudaEventCreate(&stop);    int num_iter = 10000;    float time;    cudaEventRecord(start);    low_latency_kernel<<<128, 32>>>(num_iter);    cudaEventRecord(stop);    cudaDeviceSynchronize();    cudaEventElapsedTime(&time, start, stop);    printf("low latency kernel elapsed %f\n", time);    cudaEventRecord(start);    standard_async_kernel<<<128, 32>>>(num_iter);    cudaEventRecord(stop);    cudaDeviceSynchronize();    cudaEventElapsedTime(&time, start, stop);    printf("async kernel elapsed %f\n", time);}
```

通过在async.st的时候采用cluster-scope的relaxed, 以及在local barrier wait时采用acquire

![图片](assets/f02c75d21718.png)

在H20上实际测试和mbarrier_arrive_expect_tx(sem_release)相比, 快了46%

```
low latency kernel elapsed 2.068736async kernel elapsed 3.714880
```

### 3.3 基于DSMEM的组播

在TMA上还增加了Multicast能力, 可以将数据同时加载到多个block, 下面是一个例子, 注意编译的时候有一个提示,Multicast.cluster需要使用`sm_90a/sm_100a/sm_101a`的架构

ptxas /tmp/tmpxft_00017425_00000000-6_03-tma-mcast.ptx, line 82; warning : Advisory: '.multicast::cluster' modifier on instruction 'cp.async.bulk{.tensor}' should be used on .target 'sm_90a/sm_100a/sm_101a' instead of .target 'sm_90' as this feature is expected to have substantially reduced performance on some future architectures

```
#include <cuda.h>#include <cudaTypedefs.h>#include <cooperative_groups.h>#include <cuda/barrier>#include <iostream>#pragma nv_diag_suppress static_var_with_dynamic_initusingbarrier_t = cuda::barrier<cuda::thread_scope_block>;namespace cde = cuda::device::experimental;namespace cg = cooperative_groups;constint ARRAY_SIZE = 512;constint TILE_SIZE = 16;constint CLUSTER_DIM = 8;inline PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled(){    cudaDriverEntryPointQueryResult driver_status;    void *cuTensorMapEncodeTiled_ptr = nullptr;    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,                                     cudaEnableDefault, &driver_status);    if (driver_status != cudaDriverEntryPointSuccess)        throwstd::runtime_error("driver_status != cudaDriverEntryPointSuccess");    returnreinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);}CUtensorMap make_1d_tma_desc(int32_t *global_address,                             uint64_t global_dim,                             uint32_t smem_dim){    CUtensorMap tensor_map = {};    uint64_t global_size[1] = { global_dim};    uint64_t global_stride[1] = {global_dim * sizeof(int)};    uint32_t tile_size[1]= {smem_dim};    uint32_t elem_stride[1] = {1};    auto encode = get_cuTensorMapEncodeTiled();    auto res = encode(        &tensor_map,        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,        1, // rank =1        global_address,        global_size,        global_stride,        tile_size,        elem_stride,        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);    assert(res == CUDA_SUCCESS && "make tma descriptor failed.");    return tensor_map;}__global__ void  __cluster_dims__(CLUSTER_DIM, 1, 1) tma_kernel(const __grid_constant__ CUtensorMap tensor_map,                           uint32_t coord){    __shared__ alignas(16) int tile_smem[TILE_SIZE];    __shared__ barrier_t bar;    cg::cluster_group cluster = cg::this_cluster();unsignedint cluster_rank = cluster.block_rank();    // 初始化Barrier    if (threadIdx.x == 0)    {        init(&bar, blockDim.x);        // 由于TMA调用路径为async proxy, 需要fence保持可见        // cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);        cde::fence_proxy_async_shared_cta(); // b)    }    __syncthreads();    barrier_t::arrival_token token;    if ((cluster_rank == 0 ) && (threadIdx.x == 0))    {        uint16_t ctaMask = 0b10111011;        asm volatile(            "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::"            "complete_tx::bytes.multicast::cluster "            "[%0], [%1, {%2}], [%3], %4;\n"            :            : "r"(static_cast<_CUDA_VSTD::uint32_t>(                  __cvta_generic_to_shared(tile_smem))),              "l"(&tensor_map), "r"(coord),              "r"(static_cast<_CUDA_VSTD::uint32_t>(                  __cvta_generic_to_shared(                      cuda::device::barrier_native_handle(bar)))),              "h"(ctaMask)            : "memory");                token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile_smem));    }    else    {        token = bar.arrive();    }    // Wait for the data to have arrived.    bar.wait(std::move(token));    //printf("[tma_kernel] threadIdx.x %d arrived\n", threadIdx.x);    cluster.sync();    if (threadIdx.x == 0 ) {        printf("cluster %d smem[%d %d %d %d]\n",cluster_rank, tile_smem[0],tile_smem[1],tile_smem[2],tile_smem[3]);    }}int main(int argc, char **argv){    int *h_data = nullptr;    cudaHostAlloc(&h_data, ARRAY_SIZE * sizeof(int), cudaHostAllocMapped);    for (size_t i = 0; i < ARRAY_SIZE; ++i)    {        h_data[i] = i;    }    int *d_data;    cudaHostGetDevicePointer(&d_data, h_data, 0);    CUtensorMap tensor_map = make_1d_tma_desc(        d_data, ARRAY_SIZE, TILE_SIZE);    uint32_t coord = 3 * TILE_SIZE;    tma_kernel<<<CLUSTER_DIM, 32>>>(tensor_map, coord);    cudaDeviceSynchronize();    cudaError_t err = cudaGetLastError();    cudaFree(d_data);    return0;}#Outputcluster 6 smem[0000]cluster 7 smem[48495051]cluster 0 smem[48495051]cluster 1 smem[48495051]cluster 2 smem[0000]cluster 3 smem[48495051]cluster 4 smem[48495051]cluster 5 smem[48495051]
```

## 4. 最大化内存带宽

接下来我们来看GTC25这个session的另一个话题"Maxmizing Memory Bandwidth".

### 4.1 回顾内存层次化结构

Nvidia GPU内有大量的Cuda Core, 伴随着这样的架构内存也具有很深的层次化结构

![图片](assets/fee1ce2e6587.png)

更靠近计算核的缓存能够极大的降低访问延迟, 因此Shared Memory虽然在逐代扩大容量, 但是也几乎接近极限, 例如Hopper和Blackwell的SMEM容量已经没有增长了, 因此更多的是构建局部的片上网络, 让SMEM可以通过这个小范围的网络构建成Distributed SMEM, 也是为了更好的Data Locality避免访问GMEM的延迟.

![图片](assets/154cff35d369.png)

因此正如第三章介绍的, 在Hopper和Blackwell上基于DSMEM的编程和Cluster抽象变得更重要的.

![图片](assets/efa8c9997a09.png)

Blackwell还针对Cluster的混布编排调度做了进一步的优化, 提高利用率

![图片](assets/1bc0c7f798ce.png)

但是受制于片上面积的限制, SM的增速已经放缓, 既要大的SMEM,又要更大的算力, 还要更多的SM本质上是矛盾的, 最后只能砍FP64了....

另一方面虽然内存还在因为HBM3e/HBM4持续增长, 但是在计算和访存上需要调优的难度也在加大

![图片](assets/a60230e3937c.png)

### 4.2 Little's Law

对于一个分布式系统优化其运行效率, 无论是微架构上的访问内存延迟, 还是宏观上的负载均衡, 或者网络传输的拥塞控制等, 通过排队论的视角来分析, 基于统计和随机及异步的方式来解决延迟的问题才是正确的手段. 例如前面讲到的Kingman公式等, 而这里我们来谈论一个更简单的模型, 在一个平稳系统中, 客户的长期平均数量L等于长期的平均有效到达率乘以平均客户在系统中花费的时间, 一个非常直观的公式.这就是Little's Law. 例如在一个扶梯上, 平均2s到达一个顾客(平均1s到达1/2个客户), 然后坐扶梯需要40s, 则系统可以承受的并发数为20个人

![图片](assets/802eff03fe74.png)

其实对于内存访问也是相似的, 我们可以根据内存的带宽和平均访问内存的延迟, 测算出Inflight-Bytes, 可以看到对于Hopper需要32KB的inflight才能把内存带宽打满, 而Blackwell差不多要再翻倍到64KB.

![图片](assets/0b3304df2cda.png)

对于一个简单的Kernel, 我们可以通过单个线程访问内存的次数, 单次访问内存的大小, 整个block内线程的数量, 整个SM内block的数量来得出单个SM的inflight Bytes数量

![图片](assets/bb1a04187112.png)

### 4.3 并行优化及异步访问

增加Inflight Bytes的方法通常只有三类, 指令级并行(ILP),数据级并行(DLP)以及异步内存访问

![图片](assets/b6a46b6c6a11.png)

例如通过UNROLL展开循环增加并发的指令数:

![图片](assets/0072c054a17c.png)

另一方面通过Vector Load增加数据并行(DLP)

![图片](assets/5ca092dac887.png)

![图片](assets/08387c8e24cd.png)

但是增加数据并行和指令并行都会极大的增加寄存器的压力

![图片](assets/5c0bc4621920.png)

因此出现了异步拷贝的方式来避免寄存器的占用

![图片](assets/452d75375b5f.png)

同时异步加载增加了数据拷贝和计算的Overlap

![图片](assets/5bed39fed8e7.png)

![图片](assets/ff95048619aa.png)

同时也实现了Producer-consumer的方式,并采用warp specialization执行

![图片](assets/4787d9f190af.png)

例如下面, 用一部分的threads进行内存拷贝作为Producer

![图片](assets/d1f32c64576f.png)

然后Consumer执行计算时, 再继续异步的通过一些threads进行数据预取.

![图片](assets/aaccb2203406.png)

最后GTC25的session给出了一个加载优化的建议

![图片](assets/b9433da579d4.png)

## 5. LD/ST指令控制Cache

当增加了指令并行和数据并行, 并还有大量的异步拷贝时, 如何降低寄存器和缓存的压力就成为一个必须要考虑的问题了. 在Ampere中引入了Async Copy可以bypass寄存器和L1Cache, 直接从GMEM加载到SMEM. 在Hopper进一步的实现了TMA,降低了对nD矩阵等指令的issue数量. 而在Blackwell中引入了TensorMemory, 主要目的也是在做MMA时降低对寄存器的占用. 而这一章我们来看一下General LD/ST如何控制缓存.

例如在DeepEP中用到的  "ld.global.nc.L1::no_allocate.L2::256B" 和"st.global.L1::no_allocate"等..在DeepEP中有一个文件值得借鉴`https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/utils.cuh`

### 5.1 LD指令

官方的PTX文档中记录了ld指令的多种用法

```
ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{.unified}{, cache-policy};ld{.weak}{.ss}{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{.unified}{, cache-policy};ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];ld.relaxed.scope{.ss}{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};ld.acquire.scope{.ss}{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};ld.mmio.relaxed.sys{.global}.type  d, [a];.ss =                       { .const, .global, .local, .param{::entry, ::func}, .shared{::cta, ::cluster} };.cop =                      { .ca, .cg, .cs, .lu, .cv };.level::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,                              .L1::evict_first, .L1::evict_last, .L1::no_allocate };.level::cache_hint =        { .L2::cache_hint };.level::prefetch_size =     { .L2::64B, .L2::128B, .L2::256B }.scope =                    { .cta, .cluster, .gpu, .sys };.vec =                      { .v2, .v4 };.type =                     { .b8, .b16, .b32, .b64, .b128,                              .u8, .u16, .u32, .u64,                              .s8, .s16, .s32, .s64,                              .f32, .f64 };
```

对于.weak实际上编译后的SASS指令和默认情况是一致的, 都是`LDG.E`, 而对于.volatile则是`LDG.E.STRONG`而releaxed/acquire前面章节已经详细介绍了, .mmio是在PTX8.2中增加的,并且仅在SM_70(Volta)以后的架构上支持,SASS指令为`LDG.E.MMIO.SYS`.

`.cop`是对于性能调优非常有用的一个属性, 用于定义cache操作的策略

![图片](assets/c32ea138fd68.png)

`ca`: 表示需要在所有的层次化缓存中存在,  因此在L1/L2中都会被Cache, 这事默认的行为

`cg`: 表示仅在L2中Cache,而不在L1中被Cache

`cs`: 当一个数据可能仅被访问一次时, 可以使用这种策略, 它会在L1和L2Cache中执行Evict-First的处理,可以在SASS指令中看到增加了EF属性, 例如做一些reduction操作时可以选择.

`lu`: LastUse, 在恢复Spilled Reg和弹出函数栈帧, 该属性可以避免不必要的写入.如果对global address操作则和cs属性相同.

`cv`: 表示不需要缓存.

有一个很有趣的考题, `ld.weak.global.cv`和`ld.volatile.global`的区别是什么? 没区别, 都是LDG.E.STRONG.SYS.`ld.weak.global.cg`和他们的区别呢? SASS指令为:LDG.E.STRONG.GPU

然后我们还可以定义L1Cache驱逐的策略, 是否分配L1Cache等

![图片](assets/01bfd686a1d2.png)

然后还可以定义L2Cache Prefetch的Size等. 最后Scope参数在前述内存模型的时候已经介绍过了.

在SM内部除了L1Cache外, 还有一块Read-Only Memory, 在《CUDA Refresher: The CUDA Programming Model》[7]中有个介绍

![图片](assets/b830c148c2ca.png)

Read-only memory—Each SM has an instruction cache, constant memory,  texture memory and RO cache, which is read-only to kernel code.

可以使用`ld.global.nc`有选择的使用这块Cache. 特别是在一些texture cache size较大, 然后延迟又可以通过足够的并行很好的隐藏时, 可以选择.

```
ld.global{.cop}.nc{.level::cache_hint}{.level::prefetch_size}.type                 d, [a]{, cache-policy};ld.global{.cop}.nc{.level::cache_hint}{.level::prefetch_size}.vec.type             d, [a]{, cache-policy};ld.global.nc{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}.type      d, [a]{, cache-policy};ld.global.nc{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}.vec.type  d, [a]{, cache-policy};.cop  =                     { .ca, .cg, .cs };     // cache operation.level::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,                              .L1::evict_first, .L1::evict_last, .L1::no_allocate};.level::cache_hint =        { .L2::cache_hint };.level::prefetch_size =     { .L2::64B, .L2::128B, .L2::256B }.vec  =                     { .v2, .v4 };.type =                     { .b8, .b16, .b32, .b64, .b128,                              .u8, .u16, .u32, .u64,                              .s8, .s16, .s32, .s64,                              .f32, .f64 };
```

例如采用`ld.global.nc`实际的SASS指令为LDG.E.CONSTANT. 然后我们还可以添加L1Cache不分配,L2进行Prefetch的策略, 即`ld.global.nc.L1::no_allocate.L2::256B`此时的SASS指令为:LDG.E.NA.LTC256B.CONSTANT, 这也是在DeepEP中使用的方式.

小结: 我们可以在一个较大的程序中, 采用灵活的Cache策略, 通过这些策略进一步提升Cache的利用率, 提高程序的效率, 但是这里有着大量的组合, 并且涉及到Memory Order相关的问题, 但是很多组合都为在PTX文档中详细描述,还需要未来做进一步细致的分析.

**顺便讲个小知识, 其实做高频交易这一块的人, 对于Cache策略/Memory Model的面试考察基本上是必考的, 毕竟当你的程序需要在ns级抢时间时, 这是必备的技能, 所以DeepSeek这群人去做这些最极致优化很自然的一件事**

### 5.2 ST指令

Store指令如下所示

```
st{.weak}{.ss}{.cop}{.level::cache_hint}{.vec}.type   [a], b{, cache-policy};st{.weak}{.ss}{.level::eviction_priority}{.level::cache_hint}{.vec}.type                                                      [a], b{, cache-policy};st.volatile{.ss}{.vec}.type                           [a], b;st.relaxed.scope{.ss}{.level::eviction_priority}{.level::cache_hint}{.vec}.type                                                      [a], b{, cache-policy};st.release.scope{.ss}{.level::eviction_priority}{.level::cache_hint}{.vec}.type                                                      [a], b{, cache-policy};st.mmio.relaxed.sys{.global}.type         [a], b;.ss =                       { .global, .local, .param{::func}, .shared{::cta, ::cluster} };.level::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,                              .L1::evict_first, .L1::evict_last, .L1::no_allocate };.level::cache_hint =        { .L2::cache_hint };.cop =                      { .wb, .cg, .cs, .wt };.sem =                      { .relaxed, .release };.scope =                    { .cta, .cluster, .gpu, .sys };.vec =                      { .v2, .v4 };.type =                     { .b8, .b16, .b32, .b64, .b128,                              .u8, .u16, .u32, .u64,                              .s8, .s16, .s32, .s64,                              .f32, .f64 };
```

很多内容已经在ld的章节详细介绍过了. 主要是cache operation policy(COP)有一些需要解释, 主要是采用write-back,还是write-through, 是否在L1/L2缓存, 缓存的时候是否支持evict-first策略

![图片](assets/a78dbca3f16d.png)

这些内容也对优化L1/L2 Cache的占用很有帮助.

另外再补充一个分析寄存器压力和生命周期的方式, 首先采用cuobjdump cubin

```
[root@mem-order ldst]# cuobjdump a.out -xelf allExtracting ELF file    1: a.1.sm_86.cubinExtracting ELF file    2: a.2.sm_86.cubin
```

然后用nvidasm 解析

```
[root@mem-order ldst]# nvdisasm -plr ./a.2.sm_86.cubin 
```

![图片](assets/928a9fcee0b8.png)

## 6. ScaleUP和ScaleOut网络设计探讨

### 6.1 访问内存的Size

其实新的GPU配置第五代TensorCore以后, 都会包含TMEM. 详细内容参考[《Tensor-011 Blackwell TensorCore》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493640&idx=1&sn=98cf818a60b670f0d3d40cbbcec4deef&scene=21#wechat_redirect)TMEM的引入使得MMA的结果可以放置, 并且不占用寄存器

![图片](assets/d9e1f8910260.png)

这样从编译/算子拆分调度的视角来看, 基于Tile based的IR就会变得更加容易了. Cutlass这些Layout代数的抽象,  以及Cutlass Distributed GEMMM相关的工作, 然后现有生态里的Triton. 特别是最近字节的Triton-Distributed[8]这样优秀的工作.

因此从应用的角度来看, 未来在ScaleUP和ScaleOut的最小通信单位要么是按照一个Tile,要么是按照一个Token, 因此消息的Size普遍会大于2KB.

另一方面可以看到GTC25的session给出了一个加载优化的建议

![图片](assets/b41924bf4ba3.png)

**对于片间的ScaleUP和ScaleOut网络, 是否要专门针对小size的访问进行优化呢?**

换一个问法, 如果全部是小消息,例如64B/128B.那么当我们需要组建一个超大规模网络的时候, 用于网络路由/CRC等信息的Header需要多少? 当支持一些特殊拓扑结构时, 通常还需要引入一些源路由头等. 因此小消息的实际网络传输效率便是一个问题.

UALink的协议规定的DataLink Flit长度为640B, 而Transaction Layer Flit为64B.

![图片](assets/c1b290fc6c4e.png)

其实这样的设计对于GPU侧是很简单的就可以把片上网络挂上去, 但是对于UALink的Switch要做高吞吐就有一些难度了

![图片](assets/9fca930055b2.png)

每个Switch需要把DL Flit解开, 然后逐个处理TL Flit. 对于交换机查表转发的压力会蛮大的. 同时它也约束了交换机的设计. 如果采用Shared Buffer Switch, TM的MMU设计要满足到51.2T/102.4T这样的速率打满LineRate的PPS是非常困难的. 那么势必就要换成PortBased Buffer的设计, 在交换机上为每个UALink构建一个小的Tile based PortLogic. 但是拥塞控制又是一个难题, 虽然UALink选择了Credit based的方案

![图片](assets/82e1adf8841b.png)

当然这样的方案搞个千卡互联也不是不行, 带宽短期内也可以做的挺高的. 但是总觉得长期演进上, 还是有一些问题的...当然另一个问题是NVLink/UALink这些构建的`大型机`系统长期演进下(例如5~10年)还是否会存在, 也是一个值得探讨的问题.

总体来看, NVLINK虽然传输效率在极端情况下没有UAL那么高, 但是NVLINK的协议上显得更加干净一些, 交换机卷容量也容易一些

![图片](assets/f0bef7e0da94.png)

其实一张以太网的胶布就够了

![图片](assets/a67908d98682.png)

### 6.2 访问内存延迟

前面有一章是GTC25上以Little's Law来建模的内存访问和in-flight. 似乎觉得总能满负载运行? 但是这样的方式并没有考虑真实的workload的变化, 这是我一直强调从一个系统的Scale建模来看, 必须Little's Law和Kingman公式一起来做.

![图片](assets/a9349d2fd9e6.png)

例如在Kingman公式的视角下, 当利用率接近100%时的延迟变化是这样一条曲线:

![图片](assets/4a033ff2dc6f.png)

**但是很多时候, 我们测试和评估延迟时, 通常只是在看空载的情况.** 当然当前做工程的人可能是看到了一些, 例如NCCL lauch kernel的延迟等. 其实您只看到了用cugraph来降低kernel launch的开销这一面

![图片](assets/5acc464654b9.png)

其实这个问题在去年的8月就有一篇文章[《HotChip2024后记: 谈谈加速器互联及ScaleUP为什么不能用RDMA》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492300&idx=1&sn=8a239883c831233e7e06659ec3425ea2&scene=21#wechat_redirect)讲清楚过, 并且9月我们就找到了一些更干净的解法,申请了一些专利.

![图片](assets/5c35cefcd619.png)

这篇文章还顺便讨论了一下在现有的架构下避免延迟, 当时就建议过IBGDA

![图片](assets/ab03175e037b.png)

其实我在反复阐述Kingman公式的原因是, 很多人在RoCE下跑DeepEP的benchmark挺好的,但是最后E2E的性能提升有问题的根源是什么? 通过网络变异系数的视角来看, 你就会明白为什么DeepSeek要开AR. 从计算服务变异系数的视角下,你就会明白为什么要在LowLatency Kernel上做Hook以及GroupGEMM和为什么要EPLB, 通过尽量的负载均衡降低GEMM计算延迟的抖动.

一个系统能不能够Scaling, 并不是简单的说交换机支持多少Radix,拓扑上理论可以搭建多少卡的问题, 当既要大带宽又要低延迟, 系统接近满载时如何控制整个系统的jitter才是降低延迟的关键, 显然从这个视角上来看, PFC这些东西号称At Scale完全就是胡扯, DCQCN搞了那么复杂的一个模型最基本的东西没搞懂...oops..也难怪NV以后都放弃DCQCN了.

如果每次在片间网络访问的数据量大小为2KB~4KB时, 然后通过异步访问又增加了inflight时, 控制抖动比静态延迟变得更加关键, 其实这是很多人不太清楚的, 毕竟渣B在思科十多年, 从片上网络拥塞到数据中心再到广域网处理过大量的问题, 才会反复强调这一点, 包括在设计eRDMA的拥塞控制算法时也把它作为第一优化目标来处理.

### 6.3 Memory Model

前面讲了这么多内存模型, 那么对于ScaleUP和ScaleOut的内存模型该如何设计呢? 同步的LD/ST,Cache一致性这些顺序一致性的东西肯定会影响inflight-bytes导致带宽无法被充分利用, 另一方面超大规模组网通常会有多层交换机构建, 因此又会出现数据从多个路径转发导致的Data Race的问题. 然后当多个数据包传输时, 网络是否容忍丢包重传, 重传导致的乱序的问题如何处理?

当前工业界的解法也非常奇葩, Credit Based Flow Control期望做到零丢包, 并且需要很低的误码率, 例如IB是1E-15, 而Eth是1E-12. 以太网的人最近又在折腾LowLatency Retrans(LLR)....然后IBTA的Reliable Connection定义是要求STRONG ORDER的, 这又导致引入了Lossless和Goback-N重传,或者在接收端Reorder buffer重排...

另一个极致的做法就是以AWS SRD为代表的, 传输完全不保序, 其它的所有事情丢给软件处理. 而丢给软件处理又会导致通信Kernel大量的指令开销, 浪费算力资源. 另一方面也是从指令开销的角度来看, 实在[《HotChip2024后记: 谈谈加速器互联及ScaleUP为什么不能用RDMA》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492300&idx=1&sn=8a239883c831233e7e06659ec3425ea2&scene=21#wechat_redirect)也讲清楚了为什么不能用RDMA的方式, 本来一条LD/ST的指令会被它折腾着去准备WQE搞一大堆事情, 这样做并不干净, IBGDA只是当前架构限制下的一种妥协, 所以你也看到为什么DeepSeek在论文中会提一些Unified ScaleUP/ScaleOut语义的需求.

其实NV自己在Async Proxy下, Same-Address的ordering都是没有保证的,
![图片](assets/c8182952ead8.png)

本质是也就是说计算侧都没那么严格的Ordering的要求, 传输上真没必要给自己加戏了, 把一些weak order的东西做好就行了. 其实这就是一些在架构上的TradeOff了, 这就是我为什么过去几年一直从代数的角度在提Semi-Lattice语义的原因, 可以看一下这篇3年前的文章[《向上，点亮未来：DPU的若干代数问题》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487512&idx=1&sn=23ca42b52aebb0c8c4014fd4c5dd5942&scene=21#wechat_redirect).

![图片](assets/792a969879e3.png)

A commutative idempotent semi-group,然后又来个Partially ordered set，然后很多通信多路径又要Out-Of-Order才能提高吞吐量，其实本源就在如何理解和定义偏序集，放在通信线路上由于多路径的需求，保序带来了极大的压力。那么是不是可以另辟蹊径，放在内存上呢？前文所有的辩证关系都是为了抛出这一点：

首先我们来看内存的分布，其实它就是以内存地址为序列的一个偏序集(Partially ordered set),对内存上进行的操作如果满足可交换(Commutative)、幂等(idempotent)并且满足半群(Semi-Group)中定义的封闭性和结合律。那么这个对内存的操作就是一个半格(Semi-lattice).而简单的内存读写操作是满足幂等的，至于结合律取决于这个操作里的幺元是什么，也就是说是以消息为原子，还是以Byte为原子进行操作，因为内存例如 Write 和 Read之间操作的地址空间有冲突则不满足结合律了，而消息的语义则很好的隔离开了这两者，所以你也就会看到分布式并行程序设计里常见的Actor模型和CSP模型.

所以只要我们对于消息的语义的内存使用作为幺元，然后把内存操作的地址和指令和消息绑定在一起，那么就能够实现Semi-lattice的代数结构了，进而就解决了大规模通信的难题。交换律(Commutative)决定了Out-of-order可以随便用多路径解决拥塞，幂等(idempotent)决定了丢包可以随便重传，结合律(associative)使得多个操作可以代数上merge好了再传远端。

其实在硬件上实现Semi-Lattice并没有太大的代价. 首先对于交换律,允许一些Relaxed Order的操作即可. 然后结合律的实现其实TMA就是这样的处理方式, 通过一个tensor_map描述符将操作通过结合律整合在一起. 而幂等其实也很好实现, 就是在传输的每个消息上添加一个Seq字段, 然后针对Consumer处理过的ci指针进行fence防止重写就行了, 当然在in-network-computing上有一些需要考虑的东西, 加法的幂等处理还有一些考虑.

其实最终的编程接口就变成了这样, GPU只需要issue一条指令,然后等一个mbarrier的completion_tx计数器就可以了.

```
cde::cp_async_bulk_tensor_2d_global_to_shared(tile_smem, &tensor_map, x, y, bar);token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile_smem));// 做一些其它的事情....// 等待所有数据到达bar.wait(std::move(token));
```

对传输的需求就变成了如果我把整个TMA指令产生的细粒度的LD/ST打包成一个消息, 在消息内可以让数据乱序提交,并且可以通过多个路径转发, 丢包重传其实也没啥大问题, 然后还降低了对网络上BER的需求,  最后消息完成时更新mbarrier不就行了? 恭喜您重新发明了一个二十年前iWARP就搞完的东西Direct Data Placement(DDP).

![图片](assets/6abb1cc95439.png)

但是NV(Mellanox)的人吧, 还是压根不太懂这玩意的, 虽然在AR上实现了一些DDP, 你能支持SEND/RECV的DDP么? 必然不行, 这是RoCEv2协议的缺陷.RoCEv2的协议规定了消息的传输方式只有一个Flag表示First/Middle/Last, 中间的消息其实不包含内存操作的地址的, 因此必须要保序传输. 后期通过将一个消息拆分为多个并且每个消息携带RETH,每个数据包虽然带上了操作的内存地址,但是针对SEND/RECV并无法确定接收缓冲区的绝对地址位置. 同时当你的网卡有2个网口时, 还必须要采用XOR选择其中一个口发送.... 如何在两个网口的情况下实现单个QP负载均衡处理呢?

其实iWARP的DDP定义的很清楚, 一个Msg Seq Number(MSN)字段配合一个Msg offset(MO)字段即可. 这样一个相对的offset地址就解决了问题, 接收端完全不需要ReOrder buffer, 数据来了直接根据offset算一下提交到内存就行了, 然后构造一个很小的消息bitmap等收全了去update一下mbarrier就好, 然后对于消息的幂等防止重写, 我们只需要很简单的对MSN做一些处理, 对于已经提交mbarrier的MSN禁止后续重传的消息到达时写就行了.

这就是eRDMA的实现方式, 非常简单硬件开销也非常小, 也不会因为reorder buffer增加一些不确定的延迟和抖动, 同时又可以充分的利用整个Fabric上多个路径并且能够容忍Fabric局部链路失效的情况.

然后你从这个角度再来对比eRDMA(Weak Order)和NV(Mellanox)的StrongOrder和AWS-SRD的RelaxOrder, 什么是最佳实践自然就有一个很好的判断了...

![图片](assets/cf65377e8f3e.png)

参考资料

[1] 
CUDA Techniques to Maximize Memory Bandwidth and Hide Latency: *https://register.nvidia.com/flow/nvidia/gtcs25/vap/page/vsessioncatalog/session/1727709012449001X6PZ*
[2] 
Advanced Performance Optimization in CUDA: *https://www.nvidia.com/en-us/on-demand/session/gtc24-s62192/*
[3] 
Sequential Consistency and TSO: *https://www.cis.upenn.edu/~devietti/classes/cis601-spring2016/sc_tso.pdf*
[4] 
SPCL_Memory Model: *https://spcl.inf.ethz.ch/Teaching/2019-dphpc/lectures/lecture4-memory-models.pdf*
[5] 
cuda c programming guide: *https://docs.nvidia.com/cuda/cuda-c-programming-guide/*
[6] 
Cluster Group: *https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group-cg*
[7] 
CUDA Refresher: The CUDA Programming Model: *https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/*
[8] 
Triton-distributed: *https://github.com/ByteDance-Seed/Triton-distributed*