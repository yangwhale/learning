# Tensor-006 AI软硬件交互界面: 可组合的Kernel

> 作者: zartbot  
> 日期: 2024年8月22日 12:56  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491708&idx=1&sn=1fd03181e44f573f6ec1d90d66d93a24&chksm=f995f0becee279a8e2bf99a160c465627dad3ce4eb50a2b1cecef34316c6762c59b5f791ef7e#rd

---

在谈CuTe之前我们先来看看另一个话题, 关于Cutlass 3.x的重构. 英伟达有一个Session讲的挺好的《A Generalized Micro-kernel Abstraction for GPU Linear Algebra》[1]其实本质上的问题是算子的可组合性, 可组合性带来的泛化能力是巨大的. 其实隔壁家AMD也在提Composable Kernel(CK)的概念.

接着Jim Keller也在谈论AI的软硬件交付界面, 而这个界面在时间维度和空间维度上的划分都会构成多个可组合的接口来进行泛化和抽象. 本文从这个视角来谈谈这个问题, 以及介绍一下当前Cutlass3.x的演进.

### 0. 软硬件交付界面: Composable的重要性

可能很多工科的同学并没有学习过抽象代数和范畴论相关的知识, 或许只是听说过函数式编程里单子(Monad)的传说. **A monad is a monoid in the category of endofunctors, what's the problem?**

![图片](assets/61f566733b53.png)

这里再稍微展开一下, 其实更一般来的来说, 从矩阵计算的角度来看构成一个半环(Semi-Ring)的代数结构, 当然标准的GEMM加法构成一个交换群, 因为加法是有逆元的. 而乘法有单位元, 并且加法和乘法都满足结合律.

但是我们注意到一些特殊的情况和计算需求, 对于运算可逆的要求是需要放宽的, 因此构成一个半环的代数结构. 例如在做一些图算法时, 并不是标准的加法/乘法构成的矩阵运算. 例如最短路径计算, 实际上是一种基于Min-Plus的Tropical Semi-Ring. 具体可以参考一下GraphBLAS[2]

但本质上这些运算是满足结合律的, 然后又有单位元和满足计算封闭性, 即构成一个幺半群(Monoid). 更详细的内容在[《大模型的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)里面有几篇文章讨论过

[《大模型时代的数学基础(6)-从word2vec谈谈表示论，组合性，幺半范畴和Dataflow Optics》 ](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488775&idx=1&sn=1793eb897beb71ce4a64c9ab44beee6b&chksm=f99605c5cee18cd3481913d17122bb9da63f6385901c9842e8173186e98040d7f91620a91f95&scene=21#wechat_redirect)

[《大模型时代的数学基础(8)-CDL范畴深度学习》 ](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488968&idx=1&sn=8bfd5194e645f9c37469ca2793bbcee7&chksm=f996050acee18c1c5258b28c8e8211dc9fa662c9a0b28d7174011ddce1c94fce671b6eb5e3f3&scene=21#wechat_redirect)

[《大模型时代的数据智能和数学基础》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489983&idx=1&sn=673047dc5d1b22caeb51b7fe9f2bbb4c&chksm=f996097dcee1806b43a5174ca55ca81ecec52a66c26f9b565a49c51ec9ff85cc428027022eae&scene=21#wechat_redirect)

我们从模型到算力芯片, 以及从算力芯片到模型这两个视角来看待, 无论是从上往下,还是从下往上的抽象/封装/泛化,最终都需要一个可组合的软硬件交付界面, 形象的说,就像乐高那样.

#### 0.1 从模型的角度看

从模型本身来看, 每一层Transformer保证输入输出的张量Shape一致性来看, 也是可组合性的体现. 当然可能这里面还涉及到一些Optics和Lens的概念, 形象的说这样的可组合性代表: MoE, Mamba, Transformer=Jamba

![图片](assets/ec471349e8a5.png)

而这些可组合性构建的Pytorch相关的张量计算抽象. 然后进一步的模型拆分/训练数据拆分等构成策略, 再细化到如何调度到多个GPU上.

回到本质上是一个分布式表示的问题. 在[《谈谈大模型可解释性》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490256&idx=2&sn=e25763d3bc3236e5cc22e4baed5702a5&chksm=f9960a12cee18304acafa8fcf866fed5e3528a568a7915e2081f0194331b50e57ece4083cab1&scene=21#wechat_redirect)中对于分布式的可表示性带来的可组合性需求, 以及后续的一些分析和Composable Transformer的一些猜想.

[《谈谈DeepMind会做算法导论的TransNAR并引出基于SAE-GNN的可组合Transformer猜想》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490297&idx=1&sn=7d758e84bdce7ae4f20f031f4ac3f221&chksm=f9960a3bcee1832d58956a286d2bc33ca32c69edfb3a00cdad65aec691100f2cc086649aa1bf&scene=21#wechat_redirect)

#### 0.2 从芯片的角度看

对于一个矩阵乘法任务来看, 我们可以进行两种划分:

**空间划分**: 例如对一个矩阵分块的过程中, 分块矩阵的内存指针和访存抽象, 要保证整个拆分的可组合性, 这样就可以避免大量重复的计算地址和偏移的代码实现.

**时间划分**: 矩阵之间的数据依赖, 算子融合, 异步内存访问接口,延迟隐藏方式, 寄存器文件/共享内存管理等, 甚至更大规模的算子在多卡之间的拆分和调度, 都需要可组合性.

本质上我们需要一个和芯片架构无关的一个抽象,并且能够衔接上算子融合这些Composable的能力.

#### 0.3 从算子表示的角度看

从早期的APL语言[3], 再到后来的Numpy, pytorch这些张量表示, 本质上都有算子可组合性的实现. 更进一步, 例如采用爱因斯坦求和约定(Einstein Summation Convention,Einsum) API来构建可组合算子.

例如Transformer Q,K,V计算和Attention计算的Einsum表示

```
dim = 12288qvk = nn.Linear(dim, dim * 3, bias=False) #<-利用一个线性层构造W矩阵，后续DP Allreduce好打包一起做qkv = to_qvk(x) #此时乘完后是一个(batchsize, tokens, dim*3)的张量#利用rearrange函数拆分QKV，将最后一个维度dim*3拆分成三个张量q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factorif mask is not None:    assert mask.shape == scaled_dot_prod.shape[1:]    scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)att = torch.softmax(scaled_dot_prod, dim=-1)torch.einsum('b i j , b j d -> b i d', att, v)
```

## 1. Composable算子的难题

### 1.1 芯片的原子能力

然后很遗憾的是至少在最底层的芯片实现上, 没法做到相同的原子能力. 例如NV在不同的GPU架构上都有不同的矩阵乘法单元.

![图片](assets/8634ad948704.jpg)

这是和通用CPU ISA软硬件交付界面最大的差别, 那么整个从算力芯片到算法/模型架构逐渐向上的过程中, 需要找到一个公约数, 构成一个可组合的Kernel能力来隐藏底层实现的差异.

### 1.2 复杂的矩阵分块策略

在整个矩阵乘法的过程中, 从Thread Block Tile到WarpTile再到MMA Tile存在多次划分, 然后又要考虑到硬件的实现细节和bank conflict/内存访问聚合等因素, 因此整个过程中涉及大量复杂的地址计算导致我们并不能够通过简单的标准循环展开来处理

![图片](assets/af2f264791de.jpg)

针对不同的内存访问模式, CUTLASS 2.x做了大量的抽象, 从逻辑上的坐标映射到访问内存offset index, 如何实现分块的数据访问迭代, 以及如何进行线程到数据block的映射, 针对每一代的硬件架构, 都还要定义新的Layout, 然后这些东西也非常难以维护.

![图片](assets/364cb4cef85e.jpg)

### 1.3 数值稳定性难题

其实对于算子可组合和融合还有一个问题就是数值稳定性的难题, 也就是一系列浮点运算带来的误差. 这些问题本身大概只有数学系会开一门课《数值分析》或者也叫《计算方法》的课程才会涉及. 因此在矩阵运算中还有不少的约束需要考虑.

## 2. Cutlass 3.x的抽象

Cutlass 3.x对于整个Kernel的抽象泛化表达能力做了如下假设:

Global Memory的Layout在GPU代际之间是不会变化的.

对于层次化的内存, GPU可以通过嵌套分块的并行策略进行建模

在这个层次化嵌套建模的基础上, 对于最内部循环(MMA Tile)或者WarpTile达到峰值算力有相对固定的大小,而唯一的变化在最外层的M-N-K形状和相应的ThreadBlockTile策略上.

不同数据区域间数据流动, 由开发者管理的软件流水线机制和异步内存访问机制来隐藏延迟.

有一些平台相关依赖的, 例如不同的MMA指令大小, 异步流水线的结构等, 这些可以通过一些参数化的模板来实现.

### 2.1 Cutlass 3.x分层

基于这些抽象假设, 那么就可以做到一个和GPU架构无关的交付界面,如下图所示:

![图片](assets/2f8e5b23ef06.jpg)

针对不同的硬件架构, 构建了一个原子层(Atom Layer),这是执行MMA指令或者COPY指令(SMEM->RF)的描述. 这里正好构成了GEMM半环代数结构和复杂的MMA计算的可组合点.  然后抽象了一个Tiled MMA/Copy层, 主要是针对矩阵分块的描述, 即空间维度(Spatial Tiling)的划分. 它的目的主要是在各种架构的原子算子层之上构建一种和硬件架构无关的抽象层. 紧接着抽象出了一个Collective层, 主要是一些时间维度(Temporal Tiling)上的对不同架构的计算/通信算子编排.  而更上的Kernel层则是对Thread/block放置的抽象, Device层则是主机调用的抽象. 第三章我们会给出一个详细的例子.

### 2.2 张量Layout抽象

通过时间维度和空间维度的划分构成了Cutlass 3.x的基于Tile的编程模型抽象

![图片](assets/e711e7f04d5e.jpg)

但是这些Micro-Kernel之间的交付物需要对Tensor的Layout构建一种代数上的可组合的结构. 然而Cutlass 2.x还是太复杂了, 需要进一步抽象

![图片](assets/bde307c7c3f7.jpg)

因此CuTe Layouts出现, 具体的CuTe Layout代数和层次化Layout相关的内容, 我们在后续的文章中做一个专门的介绍. 例如Logical Product/ Divide等.

![图片](assets/02da4225e81d.jpg)

## 3. Cutlass 3.x GEMM Example

我们以CuTe tutorial sgemm_sm80.cu[4]为例来进行分析.

### 3.1 Overview

GEMM Kernel的函数模板如下

```
template <class ProblemShape, class CtaTiler,          class TA, class AStride, class ASmemLayout, class TiledCopyA,          class TB, class BStride, class BSmemLayout, class TiledCopyB,          class TC, class CStride, class CSmemLayout, class TiledMma,          class Alpha, class Beta>__global__ static__launch_bounds__(decltype(size(TiledMma{}))::value)voidgemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,            Alpha alpha, Beta beta){
```

Alpha/Beta为矩阵乘加法的参数, 如下

#### 3.1.1 ProblemShape

即矩阵乘法的M/N/K的值,定义如下

```
  // Define shapes (dynamic)  auto M = int(m);  auto N = int(n);  auto K = int(k);  auto prob_shape = make_shape(M, N, K);    // (M, N, K)
```

#### 3.1.2  `CtaTiler`

它来自于CuTe的Tiler的概念, 后文会详细阐述. 这里可以理解为如何从ProblemShape拆分出BlockTile的策略, 定义如下

```
  // Define CTA tile sizes (static)  auto bM = Int<128>{};  auto bN = Int<128>{};  auto bK = Int<  8>{};  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
```

#### 3.1.3 TA const* A, TB const* B, TC* C

A, B, C矩阵的数据类型和相应的数据指针

#### 3.1.4 Layout & Stride

AStride, BStride, CStride 和具体的矩阵Column-Major, Row-Major Layout相关.

ASmemLayout, BSmemLayout, CSmemLayout 表示在每个CTA内Shared Memory上的Layout.

注: 对于不带TensorCore的老架构还有ThreadLayout:即AThreadLayout, BThreadLayout, CThreadLayout.

对于矩阵的Layout,Cutlass采用如下符号定义, 对于AB矩阵的Layout不同矩阵乘法存在四种组合

N, Column Major Matrix(Non-Transposed)

T, Row Major Matrix(Transposed)

{N,T} x {N,T} - All combinations, i.e., NN, NT, TN, TT

BLASA MajornessA LayoutB MajornessB LayoutNTM-major`(M,K):(1,ldA)`N-major`(N,K):(1,ldA)`TNK-major`(M,K):(ldA,1)`K-major`(N,K):(ldB,1)`NNM-major`(M,K):(1,ldA)`K-major`(N,K):(ldB,1)`TTK-major`(M,K):(ldA,1)`N-major`(N,K):(1,ldA)`

针对不同的Layout,ldA/ldB/ldC定义如下:

```
  int ldA = 0, ldB = 0, ldC = m;  if (transA == 'N') {    ldA = m;  } else if (transA == 'T') {    ldA = k;  } else {    assert(false);  }  if (transB == 'N') {    ldB = k;  } else if (transB == 'T') {    ldB = n;  } else {    assert(false);  }
```

接下来我们以NT为例(A为Colum Major, B为Row Major Layout). 在`gemm_nt`函数中, 定义Layout如下

```
  // 矩阵Shape定义  auto M = int(m);  auto N = int(n);  auto K = int(k);  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)  //定义GMEM中A/B/C张量的Stride  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)  // Define CTA tile sizes (static)  auto bM = Int<128>{};  auto bN = Int<128>{};  auto bK = Int<  8>{};  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)  auto bP = Int<3>{};  // Pipeline  // ASmemLayout, BSmemLayout, CSmemLayout定义  // Define the smem layouts (static)  auto sA = make_layout(make_shape(bM, bK, bP));             // (m,k,p) -> smem_idx; m-major  auto sB = make_layout(make_shape(bN, bK, bP));             // (n,k,p) -> smem_idx; n-major  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major
```

#### 3.1.5 TileCopy,TileMMA

定义Tile的拷贝和MMA指令, 如下所示

```
  //从Global Memory拷贝到SMEM的TileCopy函数  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 m-major                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 m-major  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},                                    Layout<Shape<_32,_8>>{}, // Thr layout 32x8 n-major                                    Layout<Shape< _4,_1>>{});// Val layout  4x1 n-major  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA
```

`SM80_CP_ASYNC_CACHEALWAYS`实际调用的就是`cp.async.ca`指令, 如下所示, 这种做法并不是最优的性能, 因为它会Cache到L1缓存. 后续我们会实现一个bypass L1的`SM80_CP_ASYNC_CACHEGLOBAL`的实现

```
template <class TS, class TD = TS>struct SM80_CP_ASYNC_CACHEALWAYS{  using SRegisters = TS[1];  using DRegisters = TD[1];  static_assert(sizeof(TS) == sizeof(TD), "cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "cp.async sizeof(TS) is not supported");  CUTE_HOST_DEVICE static void  copy(TS const& gmem_src,       TD      & smem_dst)  {#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)    TS const* gmem_ptr    = &gmem_src;    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"        :: "r"(smem_int_ptr),           "l"(gmem_ptr),           "n"(sizeof(TS)));#else    CUTE_INVALID_CONTROL_PATH("Support for cp.async instructions has not been enabled");#endif  }};
```

TiledCopy构成如下所示:

![图片](assets/e7d0b18e1a94.jpg)

同样TiledMMA用于定义MMA的操作, 我们可以通过如下方式产生Latex Layout

```
#include "cute/tensor.hpp"using namespace cute;int main() {  auto tiled_mma = make_tiled_mma(UniversalFMA<float,float,float>{},                                 Layout<Shape<_16,_16,_1>>{});  print_latex(tiled_mma);   return 0;}#nvcc -arch sm_86 tile_mma.cu -o tile_mma# ./tile_mma > foo.tex# pdflatex foo.tex 
```

![图片](assets/f938c3caab88.png)

TiledMMA构成方式如下:

![图片](assets/5776f35e32ad.jpg)

### 3.2 GEMM Kernel

整个GEMM Kernel的流程如下

创建Tensor Tile和相关的Layout,Shape

```
  //  // Full and Tiled Tensors  //  // Represent the full tensors  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)  // 基于CTA的坐标构建相应的Tile  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)  // Shared memory buffers  __shared__ TA smemA[cosize_v<ASmemLayout>];  __shared__ TB smemB[cosize_v<BSmemLayout>];  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K,PIPE)  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K,PIPE)
```

分块拷贝

```
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)
```

流水线预取

```
  auto K_PIPE_MAX = size<3>(tAsA);  // Total count of tiles  int k_tile_count = size<3>(tAgA);  // Current tile index in gmem to read from  int k_tile_next = 0;  // Start async loads for all pipes but the last  CUTE_UNROLL  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));    cp_async_fence();    --k_tile_count;    if (k_tile_count > 0) { ++k_tile_next; }  }
```

定义MMA的fragment

```
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)  // Allocate registers for pipelining  Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,0));                // (MMA,MMA_M,MMA_K)  Tensor tCrB = thr_mma.make_fragment_B(tCsB(_,_,_,0));                // (MMA,MMA_N,MMA_K)  // Allocate the accumulators -- same size as the projected data  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)  // Clear the accumulators  clear(tCrC);
```

GEMM流水线, 可以参考Tensor-004的相关文章, 流程是类似的. 通过几级流水线将Copy和MMA Overlap起来.

```
  //当前读取的Pipeline Index  int smem_pipe_read  = 0;  // 当前需要写入的Pipeline Index  int smem_pipe_write = K_PIPE_MAX-1;  // 从SMEM获取当前的切片  Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);  Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);  // Size of the register pipeline  auto K_BLOCK_MAX = size<2>(tCrA);  // PREFETCH register pipeline  if (K_BLOCK_MAX > 1) {    // Wait until our first prefetched tile is loaded in    cp_async_wait<K_PIPE_MAX-2>();    __syncthreads();    // Prefetch the first rmem from the first k-tile    copy(tCsA_p(_,_,Int<0>{}), tCrA(_,_,Int<0>{}));    copy(tCsB_p(_,_,Int<0>{}), tCrB(_,_,Int<0>{}));  }  //  // PIPELINED MAIN LOOP  // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions  //           and explicit pipelines in shared memory.  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).  //   Data is computed on registers(b_block).  //  //   This allows all copies and compute to overlap:  //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.  //     Copy from smem->rmem can overlap with compute on rmem.  //  CUTE_NO_UNROLL  while (k_tile_count > -(K_PIPE_MAX-1))  {    CUTE_UNROLL    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)    {      if (k_block == K_BLOCK_MAX - 1)      {        // Slice the smem_pipe_read smem        tCsA_p = tCsA(_,_,_,smem_pipe_read);        tCsB_p = tCsB(_,_,_,smem_pipe_read);        // Commit the smem for smem_pipe_read        cp_async_wait<K_PIPE_MAX-2>();        __syncthreads();      }      // Load A, B shmem->regs for k_block+1      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static      copy(tCsA_p(_,_,k_block_next), tCrA(_,_,k_block_next));      copy(tCsB_p(_,_,k_block_next), tCrB(_,_,k_block_next));      // Copy gmem to smem before computing gemm on each k-pipe      if (k_block == 0)      {        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));        cp_async_fence();        // Advance the gmem tile        --k_tile_count;        if (k_tile_count > 0) { ++k_tile_next; }        // Advance the smem pipe        smem_pipe_write = smem_pipe_read;        ++smem_pipe_read;        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;      }      // Thread-level register gemm for k_block      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);    }
```

Epilogue

该阶段为一个很简单的`axpby(alpha, tCrC, beta, tCgC);`函数.

### 3.3 Cutlass 3.x 总结

Cutlass 3.x通过TiledMMA和TileCopy隐藏了内部不同硬件架构的实现, 通过时空两种micro kernel的划分很好的抽象了工作流,同时通过CuTe Layout代数的灵活性, 构建了一个相对容易的可组合算子框架.

![图片](assets/5fd384dd0268.jpg)

下一篇我们将详细介绍CuTe和相应的CuTe代数.

参考资料

[1] 
A Generalized Micro-kernel Abstraction for GPU Linear Algebra: https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf
[2] 
GraphBLAS: https://www.mit.edu/~kepner/GraphBLAS/GraphBLAS-Math-release.pdf
[3] 
APL编程语言: https://en.wikipedia.org/wiki/APL_(programming_language)
[4] 
cute_sgemm_sm80.cu: https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu