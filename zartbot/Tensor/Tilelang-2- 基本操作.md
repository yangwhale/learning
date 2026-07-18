# Tilelang-2: 基本操作

> 作者: zartbot  
> 日期: 2025年10月12日 07:56  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496306&idx=1&sn=a6b7ffdafbd29bd7a93ced1ed4e26bb9&chksm=f995e2b0cee26ba6edf86c18b207b8cf352ef7d787f79d304d07b29afd51c468f9d63f612c00#rd

---

## 1. 从TVM到TileLang

在谈Tilelang的一些基本操作之前, 可能需要回顾一下TVM的演进, 顺着这个脉络来讲讲TileLang.

### 1.1 TVM概述

很多年前在Cisco做一个网络设备上的Telemetry数据分析和基于神经网络推理时, 由于训练得到的是TensorFlow导出的模型, 而推理通常会在同时在一些ARM/Xeon-D这些嵌入式平台的CPU或者Jetson Nano**的GPU上. 因此依葫芦画瓢的玩过很短一段时间的TVM.

TVM正是用来解决硬件和框架的碎片化问题.

**前端框架多样**: PyTorch, TensorFlow, ONNX, JAX...

**后端硬件多样**:

**通用处理器**: Intel/AMD CPU, ARM CPU

**图形处理器**: NVIDIA GPU, AMD GPU

**专用加速器**: Google TPU, Huawei Ascend, FPGA 以及各种AI芯片.

如果没有TVM, 模型部署会是一个N x M的复杂问题: 每个框架都需要为每个硬件编写专门的, 高度优化的代码库, 这成本极高且难以维护.

TVM借鉴了Halide的Compute-schedule解耦的想法, 通常一个算子需要定义一个张量计算的表达式, 即Tensor Expression(TE), 然后再定义它的Schedule, Tensor Expression是一种纯粹的声明式的DSL**, 整体描述上贴近数学公式.

```
A = tvm.te.placeholder((n_val,), name="A")B = tvm.te.placeholder((n_val,), name="B")C = tvm.te.compute((n_val,), lambda i: A[i] + B[i], name="C")# 创建PrimFuncfadd_pf = te.create_prim_func([A, B, C])mod = tvm.IRModule({"vector_add": fadd_pf})
```

而schedule是一个命令式的原语, 用来指导编译器如何优化和执行这个计算, 例如使用s.split(), s.reorder(),s.bind()等, 下面是一个例子:

```
s = te.create_schedule(C.op)# a. 获取 block 的循环(i,) = C.op.axis# 定义每个 GPU Block 中 Thread 的数量num_threads = 256    # a. 将循环 i split成两部分：外层循环 block_idx 和内层循环 thread_idxblock_idx, thread_idx = s[C].split(i, factor=num_threads)    # b. bind循环到 GPU 的线程s[C].bind(block_idx, te.thread_axis("blockIdx.x"))s[C].bind(thread_idx, te.thread_axis("threadIdx.x"))
```

另一方面, 对于一个算子(如矩阵乘法), 有成千上万种实现方式(不同的循环顺序, 分块大小等). 手动选择最优方案几乎不可能. TVM使用Auto-Tuning的方式, 自动地为特定模型和特定硬件找到"最优解".

但是随着模型越来越复杂, TE的表现力出现问题. 另一方面随着GPU架构变得越来越复杂, 以及一系列异步内存访问等能力的引入, 使得Schedule无法精确的编程控制SMEM, TensorCore**的引入也带来了大量的复杂性. 随后TVM演进出了Tensor IR.

Tensor Expression (TE): 是一种**声明式(Declarative)** 的语言. 你只需要像写数学公式一样**描述计算本身**, 而不需要关心如何计算. TensorIR: 是一种**命令式(Imperative)** 且 **可编程(Programmable)** 的IR.它可以更好的定义数据在不同层级的内存(GMEM/SMEM/Local)中的数据移动, 也可以更好的映射到TensorCore的指令, 并且为自动优化打好了基础.

另一方面还引入了Relax用于高层级的图优化, 和TensorIR一起构成了IRModule

![图片](assets/e0f475efbd40.png)

IRModule 是整个TVM中使用的主要数据结构:

tir::PrimFunc 是一种底层程序表示, 包含循环嵌套选择, 多维加载/存储, 线程和向量/张量指令, 通常用于表示算子程序.

relax::Function则是一种相对高层的程序表示, 通常可以对应于一个端到端的模型. 它既可以表示Dataflow Graph, 又可以像编程语言一样, 可以轻松表达if/else和循环.

下面是一个TIR+Relax的例子

```
from tvm.script import ir as Ifrom tvm.script import tir as Tfrom tvm.script import relax as R@I.ir_moduleclass RelaxModuleWithTIR:    @T.prim_func    def relu(x: T.handle, y: T.handle):        n, m = T.int64(), T.int64()        X = T.match_buffer(x, (n, m), "float32")        Y = T.match_buffer(y, (n, m), "float32")        for i, j in T.grid(n, m):            with T.block("relu"):                vi, vj = T.axis.remap("SS", [i, j])                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))    @R.function    def forward(        data: R.Tensor(("n", 784), dtype="float32"),        w0: R.Tensor((128, 784), dtype="float32"),        b0: R.Tensor((128,), dtype="float32"),        w1: R.Tensor((10, 128), dtype="float32"),        b1: R.Tensor((10,), dtype="float32"),    ) -> R.Tensor(("n", 10), dtype="float32"):        n = T.int64()        cls = RelaxModuleWithTIR        with R.dataflow():            lv0 = R.matmul(data, R.permute_dims(w0)) + b0            lv1 = R.call_tir(cls.relu, lv0, R.Tensor((n, 128), dtype="float32"))            lv2 = R.matmul(lv1, R.permute_dims(w1)) + b1            R.output(lv2)        return lv2
```

### 1.2 TileLang

顾名思义, 相对于TVM TensorIR/Relax以元素和循环为中心, TileLang则是以运算时的Tile为中心构建的一个DSL. 程序的基本操作单位不再是 `A[i, j]`，而是一个数据块(Tile)，例如 `A[block_row, block_col]`. 同时也将计算和内存的Layout抽象进行了分离, 并使用了Layout Inference机制. 你只需要在逻辑块上进行计算, 同时针对TensorCore这些专用硬件的tensorize, 以及使用软件pipeline来overlap数据移动和计算隐藏延迟也变得更加容易. 详细的介绍, 可以参考前面一篇

[《Tilelang-1: Introduction》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496243&idx=1&sn=a5c8beb7e4872d13b9b2f495bacc2be1&scene=21#wechat_redirect)

在Tile-Based抽象上, 其实和Nvidia的一篇Cypress论文也不谋而合

[《CUDA-Next: 基于任务的张量计算的DSL?》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494047&idx=1&sn=0fe4f9dbfc473692c733145385740c33&scene=21#wechat_redirect)

## 2. TileLang基本操作

### 2.1 基本结构

一个基本的TileLang程序如下:

```
import torchimport tilelangimport tilelang.language as T@tilelang.jit(  out_idx=[2],  # 表示函数的第三个(起始为0)形参, 即Buffer C是输出. 另外它也支持使用out_idx=[-1]表示最后一个参数是输出  target="cuda" # 表示在Nvidia GPU上编译和运行, 也可以选择 hip, cpu..  )def foo(        M,        N,        K,        BLOCK_N = 32,        dtype="float16",        accum_type = "float",):    @T.prim_func    def main (                    A: T.Tensor((M, K), dtype),        B: T.Tensor((K,N), dtype),        C: T.Tensor((M, N), accum_type)    ):        with T.Kernel(2, T.ceildiv(N,BLOCK_N), threads=(16,4,2)) as (bm, bn):            tidx = T.get_thread_binding(0)             bidy = T.get_block_binding(1)            if (tidx < bidy) and (bn == 1)  :                T.print(tidx,"cond 1")            else:                T.print(bn, "cond 2")    return mainM= 64N= 64K= 64func = foo(M,N,K) #实例化成Pytorch可以调用的函数cuda_source = func.get_kernel_source()print("Generated CUDA kernel:\n", cuda_source) #输出产生的cuda源代码a = torch.randn((M,K), device="cuda", dtype=torch.float16)b = torch.randn((K,N), device="cuda", dtype=torch.float16)c = torch.zeros((M,N) ,device="cuda", dtype=torch.float)c = func(a,b) #执行
```

我们注意到在`with T.Kernel(2, T.ceildiv(N,BLOCK_N), threads=(16,4,2)) as (bm, bn):`中初始化了整个上下文, threads之前的参数定义了需要采用的block维度, 等同于cuda中的grid = dim3(x, y, z). 然后threads也可以定义成多维来描述block的多个dim.

程序中可以使用T.get_thread_binding来获取线程对应的index. 并且在其中还可以执行循环/分支判断/打印等功能...输出的cuda源码如下

```
#include <tl_templates/cuda/gemm.h>#include <tl_templates/cuda/copy.h>#include <tl_templates/cuda/reduce.h>#include <tl_templates/cuda/ldsm.h>#include <tl_templates/cuda/threadblock_swizzle.h>#include <tl_templates/cuda/debug.h>#ifdef ENABLE_BF16#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>#endifextern "C" __global__ void main_kernel();extern "C" __global__ void __launch_bounds__(128, 1) main_kernel() {  if ((((int)threadIdx.x) < ((int)blockIdx.y)) && (((int)blockIdx.y) == 1)) {    debug_print_var("cond 1", ((int)threadIdx.x));  } else {    debug_print_var("cond 2", ((int)blockIdx.y));  }}#define ERROR_BUF_SIZE 1024static char error_buf[ERROR_BUF_SIZE];extern "C" const char* get_last_error() {    return error_buf;}extern "C" int init() {    error_buf[0] = '\0';        return 0;}extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ C, cudaStream_t stream=cudaStreamDefault) { main_kernel<<<dim3(2, 2, 1), dim3(16, 4, 2), 0, stream>>>(); TILELANG_CHECK_LAST_ERROR("main_kernel"); return 0;}#程序运行输出如下:msg='cond 1' BlockIdx=(0, 1, 0), ThreadIdx=(0, 1, 1): dtype=int value=0msg='cond 1' BlockIdx=(0, 1, 0), ThreadIdx=(0, 2, 0): dtype=int value=0msg='cond 1' BlockIdx=(0, 1, 0), ThreadIdx=(0, 3, 0): dtype=int value=0msg='cond 2' BlockIdx=(0, 0, 0), ThreadIdx=(0, 2, 1): dtype=int value=0msg='cond 2' BlockIdx=(0, 0, 0), ThreadIdx=(1, 2, 1): dtype=int value=0msg='cond 2' BlockIdx=(0, 0, 0), ThreadIdx=(2, 2, 1): dtype=int value=0msg='cond 2' BlockIdx=(0, 0, 0), ThreadIdx=(3, 2, 1): dtype=int value=0
```

### 2.2 内存管理

现代GPU具有多个层次化的内存结构(RF, SMEM, TMEM, GMEM)如下图所示:

![图片](assets/d3268be96123.png)

这些内存在现代GPU编程的时候成为影响性能的关键因素, 在多代的GPU架构演进中, 从LDMATRIX, 到cp.async, 再到TMA, 然后在Blackwell上又引入了Tensor Memory, 而Tensor Memory本身的数据访问还有一些约束, 这些硬件架构的变化使得整个数据路径上的复杂度在最近几年增加非常快.

在Tilelang中有如下几种allocate方式

**alloc_shared** : 用于分配SMEM

**alloc_local** : 用于分配RMEM

**alloc_fragment**: 也是用于分配RMEM, TileLang 在编译期间使用布局推断过程来派生布局对象 T.Fragment, 该对象决定如何为每个线程分配相应的寄存器文件

**alloc_var**: 用于分配单个变量

**alloc_tmem**: 用于blackwell的Tensor Memory分配

**alloc_barrier**: 用于分配arrive_count 的barrier

**alloc_reducer**: 针对矩阵运算中的reduce操作, 分配一个reduce操作的buffer

**alloc_descriptor**: 用于wgmma或者utcmma的描述符

然后Tilelang通过`T.copy`来对这些内存进行操作, 整体来看和Cutlass的Tiled Copy是类似的

![图片](assets/e2578b881bf8.png)

TileLang中也进行了类似的封装, 并且可以根据不同的平台调用, 例如在Hopper/Blackwell这些平台还会调用TMA执行内存拷贝, 我们也可以通过PassConfig关闭一些功能, 例如在Hopper上我们不想使用TMA,可以如下所示:

```
@tilelang.jit(  pass_configs={    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,  },  out_idx=[2],)def gemv(M, N, block_M=16, block_N=256):  @T.prim_func  def main(    A: T.Tensor((M, N), "float32"),     B: T.Tensor((M,N), "float32"),     C: T.Tensor((M,N), "float32")    ):    with T.Kernel(T.ceildiv(M, block_M),threads=32) as im:        a_smem = T.alloc_shared((block_M, block_N), "float32")        # copy from GMEM to SMEM        T.copy(A[im * block_M, block_N], a_smem)        a_local = T.alloc_local((block_M, block_N), "float32")        # copy from SMEM to RMEM        T.copy(a_smem, a_local)                a_frag = T.alloc_fragment((block_M, block_N), "float32")        T.copy(a_smem, a_frag)  return mainM,N = 1024,1024func = gemv(M, N)cuda_source = func.get_kernel_source()print("Generated CUDA kernel:\n", cuda_source)
```

此时, 我们可以看到生成的代码如下, 关闭TMA时, 它将使用向量化的float4进行加载, 然后我们可以在这里看到alloc_local和alloc_fragment的区别, alloc_local会根据Tile size分配整个buffer(16x256= 4096), 而fragment会根据Layout推理将这个Tile分布到block中的多个线程, 每个线程负责其中一小块.

```
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(float* __restrict__ A) {  extern __shared__ __align__(1024) float a_smem[];  float a_local[4096];  float a_frag[128];  #pragma unroll  for (int i = 0; i < 32; ++i) {    *(float4*)(a_smem + ((i * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(A + (((((((int)blockIdx.x) * 16384) + ((i >> 1) * 1024)) + ((i & 1) * 128)) + (((int)threadIdx.x) * 4)) + 256));  }  __syncthreads();  #pragma unroll  for (int i_1 = 0; i_1 < 32; ++i_1) {    *(float4*)(a_local + ((i_1 * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(a_smem + ((i_1 * 128) + (((int)threadIdx.x) * 4)));  }  #pragma unroll  for (int i_2 = 0; i_2 < 32; ++i_2) {    *(float4*)(a_frag + (i_2 * 4)) = *(float4*)(a_smem + ((i_2 * 128) + (((int)threadIdx.x) * 4)));  }}
```

如果将PassConfig中的`TL_DISABLE_TMA_LOWER`设置为False,则会调用TMA加载, 生成的代码如下:

```
extern "C" __global__ void __launch_bounds__(160, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc) {  extern __shared__ __align__(1024) float a_smem[];  float a_local[4096];  float a_frag[128];  __shared__ uint64_t mbarrier_mem[1];  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);  if (tl::tl_shuffle_elect<0>()) {    tl::prefetch_tma_descriptor(A_desc);    mbarrier[0].init(1);  }  __syncthreads();  if (32 <= ((int)threadIdx.x)) {    tl::warpgroup_reg_dealloc<24>();    if (tl::tl_shuffle_elect<128>()) {      mbarrier[0].arrive_and_expect_tx(16384);      tl::tma_load(A_desc, mbarrier[0], (&(a_smem[0])), 256, (((int)blockIdx.x) * 16));    }  } else {    tl::warpgroup_reg_alloc<240>();    mbarrier[0].wait(0);    #pragma unroll    for (int i = 0; i < 32; ++i) {      *(float4*)(a_local + ((i * 128) + (((int)threadIdx.x) * 4))) = *(float4*)(a_smem + ((i * 128) + (((int)threadIdx.x) * 4)));    }    #pragma unroll    for (int i_1 = 0; i_1 < 32; ++i_1) {      *(float4*)(a_frag + (i_1 * 4)) = *(float4*)(a_smem + ((i_1 * 128) + (((int)threadIdx.x) * 4)));    }  }}
```

另外我们展开介绍一下reduce buffer的使用, 在介绍[《Tensor-102: GEMV》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496294&idx=1&sn=08a1c463d92fc0ebbdc0de6cf57d5b91&scene=21#wechat_redirect)中, 我们可以看到计算时需要执行block-level reduction. 在TileLang中可以通过Reduce buffer来简化编程.

```
@tilelang.jit(  out_idx=[2],)def gemv(M, N, block_M=16, block_N=32):  @T.prim_func  def main(    a: T.Tensor((M, N), "float32"),     x: T.Tensor(N, "float32"),     o: T.Tensor(M, "float32")    ):    with T.Kernel(T.ceildiv(M, block_M)) as i0_m:      # 分配reduce buffer      o_reducer = T.alloc_reducer(block_M, "float32", replication="all")      T.clear(o_reducer)      for i0_n in T.Pipelined(T.ceildiv(N, block_N), num_stages=2):        a_smem = T.alloc_shared((block_M, block_N), "float32")        T.copy(a[i0_m * block_M, i0_n * block_N], a_smem)                a_frag = T.alloc_fragment((block_M, block_N), "float32")        T.copy(a_smem, a_frag)        x_frag = T.alloc_fragment(block_N, "float32")        T.copy(x[i0_n * block_N], x_frag)        for i1_m, i1_n in T.Parallel(block_M, block_N):          # 将Partial sum累加到reduce buffer          o_reducer[i1_m] += a_frag[i1_m, i1_n] * x_frag[i1_n]       # 执行reduce操作           T.finalize_reducer(o_reducer)            # 将reduce结果拷贝到output的GMEM      T.copy(o_reducer, o[i0_m * block_M])  return mainM,N = 4096,4096func = gemv(M, N)cuda_source = func.get_kernel_source()print("Generated CUDA kernel:\n", cuda_source)
```

### 2.3 调度策略

#### 2.3.1 T.Parallel

以ElementWise Add为例

```
@tilelang.jit(out_idx=[-1], target="cuda")def elementwise_add(    TileM,    TileN,    threads,    M=4096,    N=4096,        in_dtype="float32",    out_dtype="float32",):    @T.prim_func    def main(            A: T.Tensor((M, N), in_dtype),            B: T.Tensor((M, N), in_dtype),            C: T.Tensor((M, N), out_dtype),    ):        with T.Kernel(T.ceildiv(N, TileN), T.ceildiv(M, TileM), threads=threads) as (bx, by):            start_x = bx * TileN            start_y = by * TileM            for (local_y, local_x) in T.Parallel(TileM, TileN):                y = start_y + local_y                x = start_x + local_x                C[y, x] = A[y, x] + B[y, x]    return main
```

在T.Kernel中对矩阵进行了分块, 然后得到了线程块分别在N和M两个维度的TILE idx bx,by并将会自动绑定到blockIdx.x和blockIdx.y上. 然后就是一个T.Parallel的语法糖, 它指明了Tile的 M/N的循环 local_y/local_x可以并行, 编译器也会根据线程数自动做映射并尽量生成高效的并行加载和计算. 另外在这样的For Loop中, 还可以针对一些边界情况进行loop_break, 例如

```
for i in T.Parallel(block_M, block_N):    row_idx = by * block_M + i    col_idx = bx * block_N + j    if row_idx >= M:        T.loop_break()    B[row_idx, col_idx] = A[row_idx, col_idx]
```

#### 2.3.2 T.Pipelined

然后比较关键的是T.Pipelined, 它是一个更高级的软件流水封装, 通常这个软件的流水, 需要将部分的copy从循环中拉出, 然后交替进行copy和GEMM, 最后再完成一个GEMM.

![图片](assets/dfa9ab1444b1.png)

如下图所示:

![图片](assets/87284842d6af.png)

用户只需要给一个stage, 然后就会对代码块的Buffer使用情况进行依赖分析, 自动推导出Pipeline的各种属性.

### 2.4 运算相关的Op

#### 2.4.1 T.gemm

T.gemm也是一个对Tile级别的矩阵乘法算子的封装, 可以根据不同的卡调用不同的实现. 例如我们在Blackwell上并不需要手工去写PTX调用tcgen5mma. 可以很简单通过TileLang完成整个计算:

```
def matmul(    M,    N,    K,    block_M,    block_N,    block_K,    trans_A,    trans_B,    in_dtype,    out_dtype,    accum_dtype,    num_stages,    threads,):    A_shape = (K, M) if trans_A else (M, K)    B_shape = (N, K) if trans_B else (K, N)    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)    @T.prim_func    def main(            A: T.Tensor(A_shape, in_dtype),            B: T.Tensor(B_shape, in_dtype),            C: T.Tensor((M, N), out_dtype),    ):        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):            A_shared = T.alloc_shared(A_shared_shape, in_dtype)            B_shared = T.alloc_shared(B_shared_shape, in_dtype)            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)            mbar = T.alloc_barrier(1)            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)            C_shared = T.alloc_shared((block_M, block_N), out_dtype)            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):                T.copy(A[by * block_M, k * block_K], A_shared)                T.copy(B[bx * block_N, k * block_K], B_shared)                                T.gemm(                    A_shared,                    B_shared,                    C_tmem,                    trans_A,                    trans_B,                    mbar=mbar,                    wg_wait=-1,                    clear_accum=k == 0)                                T.mbarrier_wait_parity(mbar, k % 2)            T.copy(C_tmem, C_local)            T.copy(C_local, C_shared)            T.copy(C_shared, C[by * block_M, bx * block_N])    return main
```

对于不同硬件平台而言,操作的Memory Scope有不同的限制

Arch

Matrix A

MatrixB

MatrixD

Volta
RFRFRF
Ampere
RFRFRF
Hopper
RF/SMEMSMEMRF
Blackwell
TMEM/SMEMSMEMTMEM

因此T.gemm会针对不同的硬件架构准备数据. 正如Lei在《高性能GPU矩阵乘法的一种TileLang实现》[1]中也介绍过, 在Ampere这些平台上, T.gemm将自举人通过cp.async拷贝数据到SMEM, 然后通过LDMATRIX加载到RMEM中, 并执行MMA指令.

```
for ko in T.Pipelined((K // block_K), num_stages=stage):  # Load A into shared memory  for i, k in T.Parallel(block_M, block_K):      A_shared[i, k] = A[by * block_M + i, ko * block_K + k]  # Load B into shared memory  for j, k in T.Parallel(block_N, block_K):      B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]  for ki in T.serial(0, (block_K // micro_size_k)):      # Load A into fragment      mma_emitter.ldmatrix_a(          A_local,          A_shared,          ki,      )      # Load B into fragment      mma_emitter.ldmatrix_b(          B_local,          B_shared,          ki,      )      # Perform Matrix Multiplication      mma_emitter.mma(A_local, B_local, C_local)
```

### 2.5 JIT

对于TileLang定义的一个函数, 可以通过JIT进行即时编译, 并将它缓存到`~/.tilelang/cache/`中. 我们可以通过如下方式生成JIT Kernel

```
func = elementwise_add(M, N, TileM,TileN, "float32","float32", 256)jit_kernel = tilelang.compile(func, out_idx=[-1], target="cuda")
```

并且可以通过如下方式调用profiler测试性能

```
profiler = jit_kernel.get_profiler()avg_time = profiler.do_bench(n_warmup=50,n_repeat=100)
```

### 2.6 AutoTune

对于Kernel中的一些超参数, 我们可以通过AutoTune的方式来获得最好的性能, 它通常需要构造一个config, 然后在autotune中使用这个config进行调优, 整个过程如下所示

```
import itertoolsdef get_config():    BLOCK_M = [1, 2, 4, 8,16, 32, 64, 128]    reduce_threads = [4, 8, 16, 32, 64, 128, 256]    _configs = list(itertools.product(        BLOCK_M,        reduce_threads,    ))    configs = []    for c in _configs : #config中也可以通过一些代码逻辑对参数进行约束        if (c[0] * c[1] <=1024):            configs.append({                "BLOCK_M": c[0],                "reduce_threads": c[1],            })    return configs@tilelang.autotune(    configs= get_config(), #调用getconfig函数.    warmup= 50,    rep = 100,)@tilelang.jit(    out_idx=[-1],    target="auto",)def kernel(    BLOCK_M=None,    reduce_threads = None,):    M = 4096    K = 4096    dtype = "float32"    accum_dtype = "float32"    MAX_TRANSACTION_SIZE_IN_BITS = 128    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits    BLOCK_K = reduce_threads * TILE_K     @T.prim_func    def main(            A: T.Buffer((M, K), dtype),            B: T.Buffer((K,), dtype),            C: T.Buffer((M,), dtype),    ):        #add K dim,threads=[BLOCK_M,reduce_threads]        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=(BLOCK_M, reduce_threads)) as bm:            tm = T.get_thread_binding(0)  # tm = threadIdx.x            tk = T.get_thread_binding(1)  # tk = threadIdx.y            A_local = T.alloc_local((TILE_K,), dtype) #thread local buffer            B_local = T.alloc_local((TILE_K,), dtype)            C_accum =  T.alloc_local((1,), accum_dtype)            C_shared = T.alloc_shared((BLOCK_M,), accum_dtype) #alloc SMEM                        T.clear(C_accum)            for bk in T.serial(T.ceildiv(K,BLOCK_K)):                for k in T.vectorized(TILE_K):                    A_local[k] = A[bm * BLOCK_M + tm, bk * BLOCK_K + tk * TILE_K + k]                    B_local[k] = B[bk * BLOCK_K + tk * TILE_K + k]                                    for k in T.serial(TILE_K):                     C_accum[0] +=  A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)                        #使用TVM thread allreduce            C_reduced = T.alloc_local((1,), accum_dtype)            with T.attr(                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),                    "reduce_scope",                    T.reinterpret(T.uint64(0), dtype="handle"),            ):                T.evaluate(                    T.tvm_thread_allreduce(                        T.uint32(1),                        C_accum[0],                        True,                        C_reduced[0],                        tk,                        dtype="handle",                    ))            C[bm * BLOCK_M + tm] = C_reduced[0]    return mainauto_kernel = kernel()best_config = auto_kernel.configbest_latency = auto_kernel.latencyprint(f"Best Config: {best_config}")gflops =  2 * a.numel()  / (best_latency  / 1000) / 1e9print(f"Average execution time: {best_latency:.4f} ms")print(f"Performance (GFLOPS): {gflops:.4f} GFLOPS")print(f"Effective Memory Bandwidth: {( ( a.numel() +b.numel() +c.numel()) * 4) / (best_latency / 1000) / 1e9:.2f} GB/s")
```

## 3. 总结

总体来看, TileLang在TVM之上针对现代的加速器架构(TensorCore)构建了以Tile为一等公民的DSL, 将Tile的数据准备和计算进行了抽象. 同时在调度策略上构建了很多简便的语法糖, 并且通过Layout推理隐藏了很多复杂的Layout细节, 帮助开发者降低了开发Kernel的复杂度.

参考资料

[1] 
高性能GPU矩阵乘法的一种TileLang实现: *https://zhuanlan.zhihu.com/p/20718641070*