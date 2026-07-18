# Tensor-103.1: Basic GEMM

> 作者: zartbot  
> 日期: 2025年10月18日 00:13  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496512&idx=1&sn=a2eb5dfcabea41ea93fe2d482b07661f&chksm=f995e382cee26a94ac988f8f909de9a01016f75c4f4fa4e0730a32d54b9e54299c1c9ca2f022#rd

---

### TL;DR

这一篇我们将介绍一个非常基础并广泛使用的GEMM算子在CuteDSL和Tilelang上的实现, 并结合Hopper(H20)和Blackwell(Jetson Thor)进行一些针对性的优化. 另外, 由于控制单篇文章篇幅的原因, 这块内容将被分成多篇文章. 此为第一篇介绍基于CuteDSL的Basic Gemm. 第二篇会介绍Hopper GEMM的内容, 第三篇介绍Blackwell GEMM, 第四篇介绍TileLang GEMM, 可能后面还会有几篇介绍一些其它的和GEMM相关的内容. 

```
1 GEMM算法1.1 算法概述1.2 分块矩阵乘法1.3 TensorCore2 CuteDSL2.1 Basic Gemm2.1.1 TiledCopy2.1.2 TiledMMA2.1.3 Kernel2.1.3.1 TileCopy Layout2.1.3.2 Predication Tensor2.1.3.3 Prefetch Prologue2.1.3.4 RMEM 分配及预取2.1.3.5 主循环2.1.3.6 EpilogueRef. CublasLt GEMM
```

## 1. GEMM算法

### 1.1 算法概述

GEMM(GEneral Matrix-Matrix multiplication) 是线性代数库BLAS（Basic Linear Algebra Subprograms）中的一个核心操作. 其标准形式为：

![图片](assets/3dbe20d3783a.png)

**A**: M x K 矩阵

**B**: K x N 矩阵

**C**: M x N 矩阵 (输入和输出)

****: 标量 (常数)

它是科学和工程计算的核心构建块. 在深度学习中

**全连接层 (Fully Connected Layer)** ：其前向传播过程直接就是一个GEMM操作。

**卷积层 (Convolutional Layer)** ：通过`im2col` (image-to-column)的方式,可以将卷积操作高效地转换为一个大规模的GEMM操作.

由于神经网络中绝大多数的计算量都集中在这些层, 因此**GEMM的性能直接决定了整个AI训练和推理的速度**.

在CPU上计算的方法如下:

```
// 假设 A, B, C 都是行主序存储// C = A * B (为简化，令 alpha=1, beta=0)for (int i = 0; i < M; ++i)    for (int j = 0; j < N; ++j)        for (int k = 0; k < K; ++k)             C[i][j] += A[i][k] * B[k][j];}
```

这个算法的计算复杂度是 。

### 1.2 分块矩阵乘法

详细内容可以参考《Tensor-001 矩阵乘法分块乘法概述》, 这里做一个简要描述,  通常我们可以把一个矩阵分成多个块, 例如

我们可以将其划分为 4个块

我们可以记为:

分块后的矩阵记为分块矩阵乘法如下所示:

更一般的来讲, 如下图所示:

![图片](assets/cd8401b6e386.png)

给定一个的矩阵切分为行列

另一个的矩阵切分为行列,

则它们的乘积计算如下:

相应的乘法循环代码如下

```
for (int m = 0; m < M; m += Mtile)                // iterate over M dimension    for (int n = 0; n < N; n += Ntile)            // iterate over N dimension        for (int k = 0; k < K; ++k)            for (int i = 0; i < Mtile; ++i)       // compute one tile                 for (int j = 0; j < Ntile; ++j) {                    int row = m + i;                    int col = n + j;                    C[row][col] += A[row][k] * B[k][col];                }
```

优化的核心思想是**最大化数据复用**, 以克服内存墙. 我们一次性加载一小块数据到快速的本地存储中, 然后尽可能多地使用它, 再丢弃. 这就是**分块(Blocking/Tiling)**. 当处理大规模矩阵时，它提供了几个关键优势：

`内存限制`：对于非常大的矩阵，可能无法一次性将整个矩阵加载到内存中。通过将大矩阵分成较小的块（子矩阵），可以只加载一部分到内存中进行计算，然后交换出其他部分，从而管理有限的内存资源。

`并行计算`：现代处理器和计算架构，如多核CPU、GPU以及分布式系统，都支持并行计算。矩阵分块乘法允许将矩阵乘法任务分解成更小的独立任务，这些任务可以在不同的处理器核心或节点上同时进行，从而加速计算过程。

`缓存优化`：计算机的缓存层次结构意味着访问连续或接近的数据比访问随机分布的数据更快。通过适当地分块矩阵，可以确保计算过程中频繁访问的数据位于缓存中，减少缓存缺失，提高计算效率。

`易于实现`：从编程的角度来看，分块乘法往往更容易理解和实现，尤其是当涉及到并行编程时。它提供了一种直观的方法来划分工作负载和数据。

基于CUDA的矩阵分块算法实现可以参考 《Tensor-002 矩阵乘法优化》

### 1.3 TensorCore

对于一个的矩阵乘法, 计算量为, 访存量为, 计算访存比, 简化问题考虑的情况, 计算访存比为, 因此在数据存储和访问时的复用非常必要. 在一个Warp内, Thread计算时的效率还可以进一步并行提升, 特别是WarpLevel的寄存器文件复用上, 这就是Tensor Core诞生的原因.

第一代TensorCore在Volta架构出现, TensorCore架构也演进了很多代, 从TensorCore的计算数值精度上来看

Arch

FP64

FP16

INT8

INT4

FP8

MXFP

Volta

❌

✅ FP16

❌

❌

❌

❌

Turing

❌

✅ FP16

✅

✅

❌

❌

Ampere

✅

✅ FP16/BF16

✅

✅

❌

❌

Hopper

✅

✅ FP16/BF16

✅

❌

⚠️FP8/FP22

❌

Blackwell

✅

✅ FP16/BF16

✅

❌

✅

✅ MXFP(8/6/4)
NVFP4

Blackwell Ultra

⚠️砍算力

✅ FP16/BF16

⚠️砍算力

❌

✅

✅ MXFP(8/6/4)
NVFP4

从访问内存来看, 每个操作数矩阵可以存放的内存位置如下, 特别来说在Blackwell中还引入了tensor memory

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

指令调用上, Volta上一个warp被拆成了每4个线程一组的QuadPair, 然后同步在整个WARP调用4条指令完成计算. 到Ampere开始变成了一个完整的warp-level的同步调用. 再到Hopper增加了warpgroup-level的异步调用能力, 而在Blackwell上, 由于操作数完全不占用寄存器(全部可以存放在TMEM/SMEM上)则可以实现完全异步的调用.

详细内容可以参考

[《Tensor-003 TensorCore架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491424&idx=1&sn=0fc2110931b27714900e78d73b11a5b5&scene=21#wechat_redirect)

[《Tensor-011 Blackwell TensorCore》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493640&idx=1&sn=98cf818a60b670f0d3d40cbbcec4deef&scene=21#wechat_redirect)

基于TensorCore的CUDA和Cutlass编程可以参考下面两篇文章

[《Tensor-004 TensorCore编程及优化》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491529&idx=1&sn=12902726d6d9a8f9d66405ac6ea42fa7&scene=21#wechat_redirect)

[《Tensor-006 AI软硬件交互界面: 可组合的Kernel》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491708&idx=1&sn=1fd03181e44f573f6ec1d90d66d93a24&scene=21#wechat_redirect)

整体来看, 引入TensorCore后, GEMM的编程变得异常复杂, 下图是来自Cypress论文的一个描述Ampere和Hopper的TensorCore GEMM的流程, 大量的异步内存访问使得整个程序的复杂度非常高

![图片](assets/2593e2c084f0.png)

## 2. CuteDSL

我们首先以Cutlass github上ampere sgemm[1]为例, 对整个基于CutDSL进行一个系统性的梳理, 例如常用的一些API使用. 然后再切入到Hopper和Blackwell的一些特性, 并且也试图通过这段介绍来看到整个Nvidia的设计脉络.  例如首先从Ampere开始, 例如我们可以看到在async copy时针对非规则矩阵需要predicate tensor辅助边界计算, 在Hopper中的TMA开始这些复杂性被消除了. 并且引入了Thread Block Cluster以及一些基于TMA的组播能力. 然后再到Blackwell第五代TensorCore所有操作数都在SMEM/TMEM中, 然后还有2SM等一系列的能力.

### 2.1 Basic Gemm

这一节基于Cutlass github上[ampere sgemm]为例, 它并没有采用TensorCore,而是只采用了CUDA Core进行FMA的计算, 性能是非常差的. 另外Ampere已经进入整个生命末期了, 似乎再来看这一块没有太多的价值?

其实, 在此介绍它的主要目的是通过它了解整个Cutlass的Gemm抽象的整个工作流以及大量的tensor和layout相关的操作API. 熟悉使用Layout工具和Tensor相关的API对熟悉整个CuteDSL是非常有帮助的.

另一方面, 把它作为一个baseline, 然后再介绍Hopper和Blackwell微架构的演进, 也是非常具有价值的. 例如它使用了cp.async, 但是考虑到访问越界的情况, 需要额外的谓词张量(Preidication Tensor)防止越界, 然后在Hopper上通过TMA很干净的解决了. 个人觉得需要了解原来实现的复杂性, 才能进一步了解整个架构的演进过程.

我们继续回到矩阵分块乘法本身, 通常我们以结果  矩阵的分块来排布Thread Block, 然后在每个Thread Block内构建Kernel函数执行.

![图片](assets/5e5df288be85.png)

因此在Kernel launch时, grid =(,, 1), 其中 为原始存放在GMEM中的矩阵  的Shape, 而  则是矩阵分块乘法中, 每个Thread Block (在很多文档中也被称为Cooperative thread array,CTA)需要处理的Tile的Size.

每个CTA都会运行一个相同的Kernel代码来处理数据, 大致的数据流程是:

从GMEM中拷贝Tile到SMEM, 即构成  这两个Tile. 因此需要关于它们的Layout,并准备一个`TileCopy`对象描述如何拷贝.

然后针对CTA, 选择用哪一种一种Tile based矩阵乘法操作, 例如使用CUDA Core的FMA指令或者使用TensorCore的MMA指令. 不同的指令还有不同的`MMA_Layout`, 因此也需要一个封装`TiledMMA`对象来描述.

最后例如矩阵是需要计算时, 还需要在分块的矩阵乘法做完后再做一些处理, 缩放  并加上. 当然还有一些深度学习相关的其它算法, 例如算ReLU等. Cutlass把这些都抽象成了一个`Epilogue`过程. 例如下图所示.

![图片](assets/f3e6a336e7ec.png)

而这些就构成了整个在CTA上计算Kernel的输入参数

```
    @cute.kernel    def kernel(        self,        mA: cute.Tensor,        mB: cute.Tensor,        mC: cute.Tensor,        sA_layout: cute.Layout,        sB_layout: cute.Layout,        tiled_copy_A: cute.TiledCopy,        tiled_copy_B: cute.TiledCopy,        tiled_mma: cute.TiledMma,        epilogue_op: cutlass.Constexpr = lambda x: x,    ):
```

#### 2.1.1 TiledCopy

我们首先来看CTA如何将相关的Tile数据从GMEM拷贝到SMEM.  那么首先我们需要了解`gA`, `gB`在GMEM中的Layout, 按照BLAS的定义,分为如下四种情况:

![图片](assets/c2d91ed53aae.png)

在CuteDSL中可以通过`utils.LayoutEnum.from_tensor()`函数获得相应的Layout.

```
@cute.jitdef major_check(    mA: cute.Tensor):    print(f" shape {mA.shape}:{mA.stride} Layout: {utils.LayoutEnum.from_tensor(mA)}")M,N,K = 2048,1024,512a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)bT = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).permute((1,0))_a = from_dlpack(a, assumed_align=16)_b = from_dlpack(b, assumed_align=16)_bT = from_dlpack(bT, assumed_align=16)major_check(_a)major_check(_b)major_check(_bT)#output shape (2048, 512):(512, 1) Layout: LayoutEnum.ROW_MAJOR # A : torch默认是Row-Major shape (1024, 512):(512, 1) Layout: LayoutEnum.ROW_MAJOR # B shape (512, 1024):(1, 512) Layout: LayoutEnum.COL_MAJOR # B^T: Col-Major
```

然后在可以通过cute.make_layout函数构建sA_Layout和sB_Layout. 需要注意的是, 通常为了隐藏内存访问的延迟, 在计算时采用`num_stages`多级流水线的方式来处理, 因此在分配A和B在SMEM中的Tile时, 还需要增加一个num_stage的维度. 默认的Layout是

在example里还考虑到了访问内存bank-conflict的问题, 通过增加padding的方式来解决

```
        # 如果A和B是K-major的, 则需要考虑增加padding来减少bank conflict        padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0        padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0        sA_layout = cute.make_layout(            (self._bM, self._bK, self._num_stages),            stride=(1, (self._bM + padding_a), self._bK * (self._bM + padding_a)),        )        sB_layout = cute.make_layout(            (self._bN, self._bK, self._num_stages),            stride=(1, (self._bN + padding_b), self._bK * (self._bN + padding_b)),        )
```

CuteDSL Layout可以使用cute-viz[2], 使用如下方式安装使用

```
!pip install -U git+https://github.com/NTT123/cute-viz.gitfrom cute_viz import render_layout_svg, display_layout@cute.jitdef layout_SA():    _bM , _bN , _bK = 128 , 128 , 8    _num_stages = 3    padding_a = 4        sA_layout = cute.make_layout(       (_bM, _bK),       stride=(1, _bM),    )    sA_layout_w_padding = cute.make_layout(       (_bM, _bK),       stride=(1, (_bM + padding_a)),    )    # 对于padding造成的存储空间的占用, 可以通过size_in_bytes函数获取    print(f"Layout w/o padding: {sA_layout} Size: {cute.size_in_bytes(cutlass.Float32,sA_layout)}B")    print(f"Layout w   padding: {sA_layout_w_padding} Size: {cute.size_in_bytes(cutlass.Float32,sA_layout_w_padding)}B")       # save to svg file    render_layout_svg(sA_layout, "layout_wo_padding.svg")    render_layout_svg(sA_layout_w_padding, "layout_w_padding.svg")        #也可以直接在jupyter notebook中调用display_layout直接显示    #display_layout(sA_layout)    layout_SA()## outputLayout w/o padding: (128,8):(1,128) Size: 4096BLayout w   padding: (128,8):(1,132) Size: 4208B  # Padding每一列增加了4个元素的padding,累计8-1列 28个元素,按FP32 4Bytes计算,增加112Bytes
```

![图片](assets/d37e59280805.png)

在Ampere中, 需要并行的使用CTA内每个Thread执行拷贝操作, Cutlass中对于TileCopy的抽象如下:

![图片](assets/cae3dd1c976f.jpg)

它通过对PTX指令(即Copy_Op)和相关的PTX meta-info(Copy_Traits)封装构成了拷贝原子操作, 即Copy_Atom. 然后再根据一些Layout相关的信息构成TileCopy对象. 然后CTA内的每个线程可以通过其线程ID在TileCopy对象中切片出它的线程拷贝对象(Thr_Copy).

在CuteDSL可以通过如下API构建TileCopy对象, 其参数为Copy_Atom, Thread_Layout和Value_Layout

```
    tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA,vA)
```
Copy_Atom
对于Copy_Atom而言, 我们可以查看一下Copy_Op即原始的PTX指令

```
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}                         [dst], [src], cp-size{, src-size}{, cache-policy} ;cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}                         [dst], [src], 16{, src-size}{, cache-policy} ;cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}                         [dst], [src], cp-size{, ignore-src}{, cache-policy} ;cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}                         [dst], [src], 16{, ignore-src}{, cache-policy} ;.level::cache_hint =     { .L2::cache_hint }.level::prefetch_size =  { .L2::64B, .L2::128B, .L2::256B }cp-size =                { 4, 8, 16 }
```

注意到`cp-size`仅支持4B, 8B, 16B. 那么对于每个线程处理FP32数据时, 需要根据内存地址的连续情况考虑, 如果连续则尽量使用向量化的拷贝, 即一次拷贝4个FP32元素构成16B, 否则只拷贝单个元素. 在CuteDSL中构建Copy_Atom如下:

```
    atom_async_copy_A = cute.make_copy_atom(        cute.nvgpu.cpasync.CopyG2SOp(),        mA.element_type,        num_bits_per_copy= mA.element_type.width    )
```
TV_Layout
首先对于Thread_Layout而言, 我们按照一个维度对齐_bK构建, 同时Majorness和globalTensor相同, 以处理Tile_A的thread_Layout_A为例

```
    tA = cute.make_layout(        ( num_threads // _bK , _bK), stride=(_bK,1)    )    vA = cute.make_layout((1,1))
```

而Value_Layout默认采用cp-size = 4B, 即一个线程处理一个Value

```
    vA = cute.make_layout((1,1))
```

当我们发现是连续可以向量化时, `num_vectorized = 4`即 `cp_size=16B`. 因此`tA`和`vA`可以按照如下方式更新

```
    if cutlass.const_expr(utils.LayoutEnum.from_tensor(mA) == utils.LayoutEnum.COL_MAJOR) :        num_vectorized = 4            atom_async_copy_A = cute.make_copy_atom(            cute.nvgpu.cpasync.CopyG2SOp(),            mA.element_type,            num_bits_per_copy= mA.element_type.width * num_vectorized,        )                    major_mode_size = _bM // num_vectorized        tA = cute.make_layout(            (major_mode_size, num_threads // major_mode_size),            stride=(1, major_mode_size),        )        vA = cute.make_layout((num_vectorized, 1))    
```

最后我们就可以通过`make_tiled_copy_tv`得到TileCopy对象了, 下面是一段测试代码, 并可以渲染出相关的Layout

```
num_threads = 256_bM , _bN , _bK = 128 , 128 , 8@cute.jitdef tiled_copy(    mA : cute.Tensor):    tA = cute.make_layout(        ( num_threads // _bK , _bK), stride=(_bK,1)    )    vA = cute.make_layout((1,1))    atom_async_copy_A = cute.make_copy_atom(        cute.nvgpu.cpasync.CopyG2SOp(),        mA.element_type,        num_bits_per_copy= mA.element_type.width    )    if cutlass.const_expr(utils.LayoutEnum.from_tensor(mA) == utils.LayoutEnum.COL_MAJOR) :        num_vectorized = 4            atom_async_copy_A = cute.make_copy_atom(            cute.nvgpu.cpasync.CopyG2SOp(),            mA.element_type,            num_bits_per_copy= mA.element_type.width * num_vectorized,        )                    major_mode_size = _bM // num_vectorized        tA = cute.make_layout(            (major_mode_size, num_threads // major_mode_size),            stride=(1, major_mode_size),        )        vA = cute.make_layout((num_vectorized, 1))        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA,vA)    # render layout    render_layout_svg(tA, "thread_layout_A.svg")        _layout = tiled_copy_A.layout_dst_tv_tiled    display_layout(_layout)    render_layout_svg(_layout, "tile_copy_layout.svg")a = torch.randn(1024, 1024, device="cuda", dtype=torch.float)#a为row_major_a = from_dlpack(a, assumed_align=16)     #a为col_major#_a = from_dlpack(a.T, assumed_align=16)tiled_copy(_a)
```

Thread Layout 如下所示:

![图片](assets/0a69fe2db3aa.png)

#### 2.1.2 TiledMMA

对于TiledMMA对象, 在Cutlass中的抽象如下所示, 同样是由原始的PTX指令构成MMA_Op, 然后相应的meta-info构成MMA_Traits, 最后构成MMA的原子操作(MMA_Atom). 然后根据MMA_ATOM layout构建TiledMMA. 最后每个线程可以通过TiledMMA切片获得线程MMA对象(ThrMMA)

![图片](assets/fa81ebd92607.jpg)

首先, 我们来选择MMA_Atom, 这里的示例中选择了基于CUDA Core SIMT的FMA, 即`cute.nvgpu.MmaUniversalOp(cutlass.Float32)`

![图片](assets/85a6afb2d7b4.png)

它是一个1x1x1的MMA. 然后对于Atoms_Layout, 即每次乘法产生一个结果, 然后对于Thread Layout我们按照TileC的Layout简单构建.

```
        atoms_layout = cute.make_layout(            (self._num_threads // 16, 16, 1), stride=(16, 1, 0)        )        if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):            atoms_layout = cute.make_layout(                (16, self._num_threads // 16, 1), stride=(1, 16, 0)            )
```

最后我们通过make_tiled_mma函数构建TiledMMA对象, 它需要两个参数, 一个是MMA_Op, 一个是Atoms_Layout. 还有一个可选参数`permutation_mnk`. 通过重排使得线程可以连续的读值, 整体构建TiledMMA的测试函数日下:

```
num_threads = 256_bM , _bN , _bK = 128 , 128 , 8@cute.jitdef tiled_mma(    mC : cute.Tensor):    atoms_layout = cute.make_layout(        (num_threads // 16, 16, 1), stride=(16, 1, 0)    )    if cutlass.const_expr(utils.LayoutEnum.from_tensor(mC) == utils.LayoutEnum.COL_MAJOR):        atoms_layout = cute.make_layout(            (16, num_threads // 16, 1), stride=(1, 16, 0)        )    op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)    permutation_tiler_M = cute.make_layout(        (atoms_layout.shape[0], 4), stride=(4, 1)    )    permutation_tiler_N = cute.make_layout(        (atoms_layout.shape[1], 4), stride=(4, 1)    )    tiled_mma = cute.make_tiled_mma(        op,        atoms_layout,        permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),    )        print(f"Atoms layout: {atoms_layout}")    print(f"TiledMMA TV-layout-A: {tiled_mma.tv_layout_A_tiled}")    print(f"TiledMMA TV-layout-C: {tiled_mma.tv_layout_C_tiled}")c = torch.randn(1024, 1024, device="cuda", dtype=torch.float)_c = from_dlpack(c, assumed_align=16)tiled_mma(_c)#outputAtoms layout: (16,16,1):(16,1,0)TiledMMA TV-layout-A: ((16,16),(1,(4,1))):((0,4),(0,(1,0)))TiledMMA TV-layout-C: ((16,16),(1,(4,4))):((256,4),(0,(1,64)))#without Permutation:Atoms layout: (16,16,1):(16,1,0)TiledMMA TV-layout-A: ((16,16),(1,(1,1))):((0,1),(0,(0,0)))TiledMMA TV-layout-C: ((16,16),(1,(1,1))):((16,1),(0,(0,0)))
```

最后在Host函数中, 需要计算Grid和Block用于launch kernel

```
        # grid_dim对C的Shape按照block_M和block_N 划分, 而block_dim 则根据atoms_layout size(aka. num_threads)构建.        # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, 1)        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1        self.kernel(            mA,            mB,            mC,            sA_layout,            sB_layout,            tiled_copy_A,            tiled_copy_B,            tiled_mma,            epilogue_op,        ).launch(            grid=grid_dim,            block=[cute.size(atoms_layout), 1, 1],            stream=stream,        )
```

#### 2.1.3 Kernel

前述两节在host侧的代码基本上处理完了, 然后Kernel所需要参数也准备齐全了,接下来我们来看Kernel的代码. 首先我们获取Thread Index和Block Index, 并根据BlockIndex构造Tile的坐标(Tiler_coord). 并且根据thread index(tidx)从TiledMMA对象中获取切片得到Thr_MMA

```
        # Thread and block indices        tidx, tidy, tidz = cute.arch.thread_idx()        bidx, bidy, bidz = cute.arch.block_idx()        tiler_coord = (bidx, bidy, None)                thr_mma = tiled_mma.get_slice(tidx)
```
2.1.3.1 TileCopy Layout
然后根据tiler_coord和CTA_Tiler, 即  构成的元组来获取local_tile. 注意由于cta_tiler和tiler_coord都是3维的, 因此通过proj元组来确定需要的维度.

```
        # ///////////////////////////////////////////////////////////////////////////////        # Get the appropriate tiles for this thread block.        # gA: (BLK_M, BLK_K, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)        # ///////////////////////////////////////////////////////////////////////////////        gA = cute.local_tile(            mA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1) #select M, K dim, tiler=(128,8)        )        gB = cute.local_tile(            mB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1) #select N, K dim, tiler=(128,8)        )        gC = cute.local_tile(            mC, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None) #select M, N dim, tiler=(128,128)        )
```

例如我们计算时采用M,N,K=4096,4096,4098, 同时可以打印gA, gB, gC

```
        if (tidx, tidy, bidx, bidy ) == (0, 0, 1, 0) :            cute.printf("gA {}",gA)            cute.printf("gB {}",gB)#output          gA raw_ptr(0x0000781f44200400: f32, gmem, align<16>) o (128,8,513):(4098,1,8) =   ( 2.232828, -0.914160, -0.434011, 0.241222, -0.203234, -0.858159, -1.057811, -0.276097, -1.159224, -0.202816, -0.339250, 0.847252, 0.282905, -0.242454, 0.251689, -1.054146, -0.730669, 1.360016, -0.043775, 2.123710, 1.541446, -0.778495, -0.293049, 0.340791, 1.854621, -0.315319, -0.030140, 0.353239, 2.685961, 0.276291, 0.416161, [...] )gB raw_ptr(0x0000781f3e000000: f32, gmem, align<16>) o (128,8,513):(4098,1,8) =   ( -0.107501, 1.378693, -0.589159, -0.628105, -0.784239, -0.085578, -1.613212, 0.500337, 0.196457, -0.388494, 0.130661, 0.322285, -0.098240, 0.765472, -0.916214, 1.665686, -1.892140, 1.230363, 1.912450, -0.432180, -0.306490, -0.753587, 0.533725, -1.154443, -0.092635, 0.996614, 0.718422, -0.614310, 0.130793, -0.219224, 0.902878, [...] )
```

然后我们需要考虑边界情况, K在进行 , 则K这个维度的残差为, 然后通过cute.domain_offset将gA/gB的指针移动按照`-k`的方向移动, 使得第一个Tile是成为一个不规则的Tile.

![图片](assets/471ec6b0940a.png)

```
        residue_k = mA.shape[1] - cutlass.Int32(self._bK) * gA.shape[2] #mA.shape[1] == K        gA = cute.domain_offset((0, residue_k, 0), gA)        gB = cute.domain_offset((0, residue_k, 0), gB)        if (tidx, tidy, bidx, bidy ) == (0, 0, 1, 0) :            cute.printf("domain_offset by residue_k={}", residue_k)            cute.printf("gA {}",gA)            cute.printf("gB {}",gB)      # outputdomain_offset by residue_k=-6gA raw_ptr(0x0000781f442003e8: f32, gmem, align<4>) o (128,8,513):(4098,1,8) =   ( 0.545876, 0.220103, 0.771488, 0.042910, -2.330272, -1.605312, 0.793518, 0.719658, 0.783606, 0.096901, 1.967885, 0.062523, -0.751021, 0.431259, 0.881671, 0.435522, 0.937110, 0.633271, 0.864045, -0.535543, 0.767267, 0.483572, -1.474787, 0.092404, -0.984192, -0.882738, -0.088075, -0.012253, 0.609542, -0.092585, 0.712758, [...] )gB raw_ptr(0x0000781f3dffffe8: f32, gmem, align<4>) o (128,8,513):(4098,1,8) =   ( 0.000000, -1.123452, -0.652423, 0.933533, 0.792622, -0.863053, -0.584279, -0.662824, 0.366895, -0.101855, -1.214358, 0.869608, 0.572380, 0.479916, -0.437270, 0.323538, 0.189797, 0.122727, 0.473485, -1.118110, -0.256566, 0.798130, -3.183375, 0.578726, -2.714239, 1.723931, -1.179997, -1.230503, 0.377926, 2.089392, -0.404093, [...] )
```

但是需要注意, 这里给出的是Tiler_Coord=(1,0)的值, 而如果使用Tiler_Coord=(0,0), 打印offset后的张量会导致越界报错. 而越界的问题, 我们将在后面通过谓词张量(Predicate Tensor)来进行边界判断和处理.

对比原始地址gA= 0x0000781f44200400 , 新的gA地址为0x0000781f442003e8, 向前移动了6个元素(0x18= residue_k * 4B)的长度. 具体的计算看domain_offset函数也可以.

```
@dsl_user_opdef domain_offset(coord: Coord, tensor: Tensor, *, loc=None, ip=None) -> Tensor:    offset = crd2idx(coord, tensor.layout, loc=loc, ip=ip)    if isinstance(tensor.iterator, Pointer):        return make_tensor(tensor.iterator + offset, tensor.layout)    elif is_integer(tensor.iterator) or isinstance(tensor.iterator, tuple):        new_iter = _cute_ir.add_offset(            _pack_int_tuple(tensor.iterator), _pack_int_tuple(offset)        )        return make_tensor(_unpack_x_tuple(new_iter), tensor.layout)    else:        raise ValueError(f"unsupported tensor for domain_offset, got {tensor}")
```

然后我们需要分配共享内存, 以及从TileCopy中切片得到ThrCopy.

```
        smem = cutlass.utils.SmemAllocator()        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)        thr_copy_A = tiled_copy_A.get_slice(tidx)        thr_copy_B = tiled_copy_B.get_slice(tidx)        tAgA = thr_copy_A.partition_S(gA)        tAsA = thr_copy_A.partition_D(sA)        tBgB = thr_copy_B.partition_S(gB)        tBsB = thr_copy_B.partition_D(sB)                if (tidx, tidy, bidx, bidy ) == (0, 0, 1, 1) :            cute.printf("sA Layout {}", sA_layout)            cute.printf("tAgA {}",tAgA)            cute.printf("tAsA {}",tAsA)            cute.printf("sB Layout {}", sB_layout)            cute.printf("tBgB {}",tBgB)            cute.printf("tBsB {}",tBsB)# outputsA Layout (128,8,3):(1,132,1056)tAgA raw_ptr(0x000077b95a2003e8: f32, gmem, align<4>) o ((1,1),4,1,513):((0,0),131136,0,8) = tAsA raw_ptr(0x0000000000000400: f32, smem, align<4>) o ((1,1),4,1,3):((0,0),32,0,1056) = sB Layout (128,8,3):(1,132,1056)tBgB raw_ptr(0x000077b9542003e8: f32, gmem, align<4>) o ((1,1),4,1,513):((0,0),131136,0,8) = tBsB raw_ptr(0x0000000000003570: f32, smem, align<4>) o ((1,1),4,1,3):((0,0),32,0,1056) = 
```

另一方面, 我们将B矩阵设置为Col-major时, Layout为:

```
sA Layout (128,8,3):(1,132,1056)tAgA raw_ptr(0x000074ea3a2003e8: f32, gmem, align<4>) o ((1,1),4,1,513):((0,0),131136,0,8) = tAsA raw_ptr(0x0000000000000400: f32, smem, align<4>) o ((1,1),4,1,3):((0,0),32,0,1056) = sB Layout (128,8,3):(1,128,1024)tBgB raw_ptr(0x000074ea33fe8200: f32, gmem, align<16>) o ((4,1),1,1,513):((1,0),0,0,32768) = tBsB raw_ptr(0x0000000000003570: f32, smem, align<16>) o ((4,1),1,1,3):((1,0),0,0,1024) = 
```

对于Layout如下表所示, 其中A是Row-Major, B是Col-Major, 从Shared-Memory来看, sA_Layout的stride中增加了4B Padding用于缓解bank-conflict, 然后两者都是按照num_stages= PIPE来存放多级流水线的信息.

Layout

sA

(bM,bK, PIPE)

(128,8,3):(1,132,1056)

sB

(bN,bK, PIPE)

(128,8,3):(1,128,1024)

tAgA

(CPY_V, CPY_M, CPY_K, k)

((1,1),4,1,513):((0,0),131136,0,8)

tBgB

(CPY_V, CPY_N, CPY_K, k)

((4,1),1,1,513):((1,0),0,0,32768)

tAsA

(CPY_V, CPY_M, CPY_K, PIPE)

((1,1),4,1,3):((0,0),32,0,1056)

tBsB

(CPY_V, CPY_N, CPY_K, PIPE)

((4,1),1,1,3):((1,0),0,0,1024)

然后Thr_copy 通过原始的TileCopy对象按照Thread idx调用get_slice()得到. 并且通过Thr_copy.partition_S(gA)获得Thread拷贝在原GMEM的Partition Layout, 其中对于RowMajor的矩阵A, CPY_V = (1,1), 而对于Cow-Major矩阵可以采用向量化的cp.async, CPY_V = (4,1), 即async_cp的cp_size = 16B. 然后blockDim= (256,1,1), 因此对于单个Thread 需要copy 整个 _bM * _bK / num_threads = 128 * 8 /256 = 4个值.

cp_size = 4B时, 即CPY_V = (1,1)时, CPY_M,CPY_N = (4,1)

cp_size = 16B时, 即CPY_V = (4,1)时, CPY_M,CPY_N = (1,1)
2.1.3.2 Predication Tensor
正如我们在前文看到的, CuTe local_divide时, 例如我们需要将M,K=4096,4098 按照Tiler= (bM,bK)= (128,8)切分是, 会剩余2个Element, 对于dim-K, Cutlass不会采用512个(128,8)的块加上一个(128,2)的块. 而是整体都对齐到(128,8)的块. 然后和其它CUDA编程类似, 通过一个谓词(Predication)矩阵来处理. 详细文档可以参考《0y Predication》[3]

一般的过程如下:

创建一个和原始数据形状相同的“identity”布局. 即 cute.make_identity_tensor(mA.shape), 然后基于CTA_Tiler和对应的block坐标获取Local Tile.

```
        mcA = cute.make_identity_tensor(mA.shape)        mcB = cute.make_identity_tensor(mB.shape)        cA = cute.local_tile(            mcA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)        )        cB = cute.local_tile(            mcB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)        )        
```

然后同样基于这个identity layout构建domain_offset和thr_copy partition

```
        cA = cute.domain_offset((0, residue_k, 0), cA)        cB = cute.domain_offset((0, residue_k, 0), cB)        # Repeat the partitioning with identity layouts        tAcA = thr_copy_A.partition_S(cA)        tBcB = thr_copy_B.partition_S(cB)        # Allocate predicate tensors for m and n
```

然后在RMEM中make_fragment. 数值类型为Bool. 以TileA矩阵为例. tApA是用于主循环中的M/N边界的检查, 而tApA_residue_k则是针对第一列的边界检查M/N和K都需要检查. 以A矩阵为例, 如下图所示:

![图片](assets/72bc8f812154.png)

首先创建Fragment如下所示:

```
        tApA = cute.make_fragment(            cute.make_layout(                (                    tAsA.shape[0][1],  #CPY_V->rest_v                    cute.size(tAsA, mode=[1]), #CPY_M                    cute.size(tAsA, mode=[2]), #CPY_K                ),                stride=(cute.size(tAsA, mode=[1]), 1, 0),            ),            cutlass.Boolean,        )                # Allocate predicate tensors for m, n and k for residue k-tile        tApA_residue_k = cute.make_fragment(            cute.make_layout(                (                    tAsA.shape[0][1],                    cute.size(tAsA, mode=[1]),                    cute.size(tAsA, mode=[2]),                ),                stride=(                    cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),                    cute.size(tAsA, mode=[2]),                    1,                ),            ),            cutlass.Boolean,        )
```

对于主循环, 采用tApA和tBpB 进行检查, 边界检查只需要检查M(for A)和N(for B)维度, 通过一个bool值, 判断是否小于边界M和N. 这里采用了cute.elem_less进行比较, 当大于等于边界时, 返回值为0, 因此代表这些元素不参与拷贝.

```
        # Set predicates for m/n bounds for mainloop        for rest_v in range(tApA.shape[0]):            for m in range(tApA.shape[1]):                tApA[rest_v, m, 0] = cute.elem_less(                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]                )         for rest_v in range(tBpB.shape[0]):            for n in range(tBpB.shape[1]):                tBpB[rest_v, n, 0] = cute.elem_less(                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
```

而对于tApA_residue_k和tBpB_residue_k则需要完整的检查M/N和K的边界条件

```
        # Set predicates for m/n/k bounds for residue k tile        for rest_v in range(tApA_residue_k.shape[0]):            for m in range(tApA_residue_k.shape[1]):                for k in range(tApA_residue_k.shape[2]):                    coord_A = tAcA[(0, rest_v), m, k, 0]                    tApA_residue_k[rest_v, m, k] = cute.elem_less(                        (coord_A[0], cutlass.Int32(-1)), (mA.shape[0], coord_A[1])                    )        for rest_v in range(tBpB_residue_k.shape[0]):            for n in range(tBpB_residue_k.shape[1]):                for k in range(tBpB_residue_k.shape[2]):                    coord_B = tBcB[(0, rest_v), n, k, 0]                    tBpB_residue_k[rest_v, n, k] = cute.elem_less(                        (coord_B[0], cutlass.Int32(-1)), (mB.shape[0], coord_B[1])                    )
```
2.1.3.3 Prefetch Prologue
首先我们需要通过async.cp提交所有的GMEM-> SMEM的内存拷贝, 第一步是对非规则的那块进行拷贝

```
        k_pipe_max = cute.size(tAsA, mode=[3])        k_tile_count = cute.size(tAgA, mode=[3])        gmem_pipe_read = cutlass.Int32(0)        cute.copy(            tiled_copy_A,            tAgA[None, None, None, gmem_pipe_read],            tAsA[None, None, None, 0],            pred=tApA_residue_k, # 使用tApA_residue_k谓词张量作为条件        )        cute.copy(            tiled_copy_B,            tBgB[None, None, None, gmem_pipe_read],            tBsB[None, None, None, 0],            pred=tBpB_residue_k, # 使用tBpB_residue_k谓词张量作为条件        )                # 每次cp.async都会提交一个commit_group, 这样在wait_group时好检查流水是否完成        cute.arch.cp_async_commit_group()                 # 拷贝完成后增加gmem读流水线计数器        gmem_pipe_read = (            gmem_pipe_read + 1            if gmem_pipe_read + 1 < k_tile_count            else cutlass.Int32(0)        )
```

然后是通过一个循环拷贝剩余的Tile, 注意循环的条件只Prefetch SMEM流水线深度-1的数据到SMEM中.

```
        # Start async loads for 1st k-tile onwards, no k-residue handling needed        for k_tile in range(1, k_pipe_max - 1):            if k_tile < k_tile_count:                cute.copy(                    tiled_copy_A,                    tAgA[None, None, None, gmem_pipe_read],                    tAsA[None, None, None, k_tile],                    pred=tApA, # 谓词张量使用了tApA                )                cute.copy(                    tiled_copy_B,                    tBgB[None, None, None, gmem_pipe_read],                    tBsB[None, None, None, k_tile],                    pred=tBpB, # 谓词张量使用了tBpB                )                            # 拷贝完成后增加gmem读流水线计数器            gmem_pipe_read = (                gmem_pipe_read + 1                if gmem_pipe_read + 1 < k_tile_count                else cutlass.Int32(0)            )                        # 流水线每次提交cp.async后, 提交一个commit_group            cute.arch.cp_async_commit_group()
```

最后, 如果Tile的数量小于流水线的数据, 则代表所有的Tile拷贝都被提交后, 就可以清除谓词张量了

```
        # all tiles have been copied from global memory, so clear the        # predicate tensor        if k_tile_count < k_pipe_max:            for rest_v in range(tApA.shape[0]):                for m in range(tApA.shape[1]):                    tApA[rest_v, m, 0] = cutlass.Boolean(0)            for rest_v in range(tBpB.shape[0]):                for n in range(tBpB.shape[1]):                    tBpB[rest_v, n, 0] = cutlass.Boolean(0)
```
2.1.3.4 RMEM 分配及预取
首先我们需要根据TiledMMA的切片生成Thr_MMA, 然后根据它构建从SMEM加载到RMEM的Layout

```
        thr_mma = tiled_mma.get_slice(tidx)                # ///////////////////////////////////////////////////////////////////////////////        # Define A/B partitioning and C accumulators.        # ///////////////////////////////////////////////////////////////////////////////        tCsA = thr_mma.partition_A(sA)        tCsB = thr_mma.partition_B(sB)        tCgC = thr_mma.partition_C(gC)        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])        tCrC = tiled_mma.make_fragment_C(tCgC)        # Clear the accumulator        tCrC.fill(0.0)        # Current pipe index in smem to read from / write to        smem_pipe_read = cutlass.Int32(0)        smem_pipe_write = cutlass.Int32(k_pipe_max - 1)        tCsA_p = tCsA[None, None, None, smem_pipe_read]        tCsB_p = tCsB[None, None, None, smem_pipe_read]
```

然后是预取第一个K-Tile的数据到寄存器中, 此处调用了cute.autovec_copy可以自动采用向量化的指令加载.

```
        # ///////////////////////////////////////////////////////////////////////////////        # PREFETCH register pipeline        # ///////////////////////////////////////////////////////////////////////////////        k_block_max = cute.size(tCrA, mode=[2])        if k_block_max > 1:            # 此处需要首先等待第一个Tile已经完成加载到SMEM            cute.arch.cp_async_wait_group(k_pipe_max - 2)            cute.arch.barrier()            # Prefetch the first rmem from the first k-tile            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])
```
2.1.3.5 主循环
这里详细展示了整个软件流水线(Software Pipeline)的过程. 对于整个主循环, 首先我们来看GMEM -> SMEM拷贝,  默认的SMEM流水线深度为num_stages = 3, 在前述的SMEM分配时, 根据CTA Tiler的描述分配了3倍大小的空间, 在进入主循环前, 只预取了 num_stages - 1 = 2个缓冲区的数据.  从整体来看这个循环的结构如下:

从GMEM拷贝一个k-tile到SMEM

对这个K-Tile执行GEMM计算

等待下次拷贝完成

`cute.arch.cp_async_wait_group(num_smem_stages - 2)` 这条命令会一直等待,直到未完成的'copy'操作数量小于或等于1. 这种方法的优势在于, 它允许共享内存的生产(即步骤-1)和消费(即步骤-2) Overlap同时进行, 如下图所示

![图片](assets/1c08fc13cd63.png)

然后对于SMEM到寄存器的流水线操作也是类似的, 寄存器流水线生产(加载) i+1 的数据, 消费(计算) i 的数据,然后再生产 i+2 的数据... 值得注意的是, i 和 i+1 不会使用相同的寄存器, 这消除了对同一寄存器的依赖, 从而获得更好的并行性.

整个循环的流程如下所示:

```
for _ in range(k_tile_count):    for k_block in range(k_block_max, unroll_full=True):        #  1. 等待前一个流水线完成        #  2. SMEM->RMEM, 拷贝下一个block        #  3. 如果为内层循环的第一次迭代, 即k_block = 0, 预取下一个Tile-A        #  4. 对k_block执行GEMM        #  5. 如果为内层循环的第一次迭代, 即k_block = 0, 预取下一个Tile-B
```

整个MainLoop的代码如下:

```
        for _ in range(k_tile_count):            for k_block in range(k_block_max, unroll_full=True):                if k_block == k_block_max - 1:                    tCsA_p = tCsA[None, None, None, smem_pipe_read]                    tCsB_p = tCsB[None, None, None, smem_pipe_read]                    cute.arch.cp_async_wait_group(k_pipe_max - 2)                    cute.arch.barrier()                # Load A, B from shared memory to registers for k_block + 1                k_block_next = (k_block + 1) % k_block_max  # static                cute.autovec_copy(                    tCsA_p[None, None, k_block_next],                    tCrA[None, None, k_block_next],                )                cute.autovec_copy(                    tCsB_p[None, None, k_block_next],                    tCrB[None, None, k_block_next],                )                # Fetch next A: To better interleave global memory access and                # compute instructions, we intentionally use the sequence:                # copy A, perform GEMM, then copy B.                if k_block == 0:                    cute.copy(                        tiled_copy_A,                        tAgA[None, None, None, gmem_pipe_read],                        tAsA[None, None, None, smem_pipe_write],                        # Use predicates because the m-mode may be irregular                        pred=tApA,                    )                # Thread-level register gemm for k_block                cute.gemm(                    tiled_mma,                    tCrC,                    tCrA[None, None, k_block],                    tCrB[None, None, k_block],                    tCrC,                )                # Fetch next B and update smem pipeline read/write                if k_block == 0:                    cute.copy(                        tiled_copy_B,                        tBgB[None, None, None, gmem_pipe_read],                        tBsB[None, None, None, smem_pipe_write],                        # Use predicates because the n-mode may be irregular                        pred=tBpB,                    )                    cute.arch.cp_async_commit_group()                    smem_pipe_write = smem_pipe_read                    smem_pipe_read = smem_pipe_read + 1                    if smem_pipe_read == k_pipe_max:                        smem_pipe_read = cutlass.Int32(0)                    # After copying all tiles, we avoid clearing the predicate                    # tensor in the `mainloop` to prevent increasing its                    # instruction count. Instead, we continue copying the                    # first tile, though it won't be used. The 0-th tile is not                    # copied due to its irregular shape, which could lead to                    # illegal memory accesses.                    gmem_pipe_read = (                        gmem_pipe_read + 1                        if gmem_pipe_read + 1 < k_tile_count                        else cutlass.Int32(1)                    )
```
2.1.3.6 Epilogue
最后经过K-Tile迭代, 计算的结果已经在分配好的tCrC寄存器中, 然后我们需要等待前述操作都完成后, 执行epilogue_op

```
        cute.arch.cp_async_wait_group(0)        cute.arch.barrier()        tCrC.store(epilogue_op(tCrC.load()))
```

完成EpilogueOp的计算后, 同样使用谓词张量的方法控制边界条件, 并将数据从RMEM拷贝到GMEM

```
        cC = cute.make_identity_tensor(gC.shape)        tCpC = thr_mma.partition_C(cC)        predC = cute.make_fragment(tCrC.layout, cutlass.Boolean)        residue_m = mC.shape[0] - cutlass.Int32(self._bM) * bidx        residue_n = mC.shape[1] - cutlass.Int32(self._bN) * bidy        for i in range(cute.size(tCrC.shape)):            predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))        numIterM = cute.size(tCrC, mode=[1])        numIterN = cute.size(tCrC, mode=[2])        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)        cute.copy(atom, tCrC, tCgC, pred=predC)        return
```

#### 2.1.4 测试及验证

Cutlass Example中的测试代码相对复杂, 我们只选用Class SGemm的代码, 按照如下方式进行测试

```
import timefrom typing import Tuplefrom functools import partialimport cuda.bindings.driver as cudaimport torchimport cutlassimport cutlass.cute as cuteimport cutlass.utils as utilsfrom cutlass.cute.runtime import from_dlpackdef benchmark(M, N , K , callable, *, num_warmups, num_iterations, dtype_size = 4, accum_dtype_size = 4):    start_event = torch.cuda.Event(enable_timing=True)    end_event = torch.cuda.Event(enable_timing=True)    torch.cuda.synchronize()    for _ in range(num_warmups):        callable()    start_event.record(stream=torch.cuda.current_stream())    for _ in range(num_iterations):        callable()    end_event.record(stream=torch.cuda.current_stream())    torch.cuda.synchronize()    elapsed_time = start_event.elapsed_time(end_event)    avg_time = elapsed_time / num_iterations    gflops =  2 * M * N * K / (avg_time  / 1000) / 1e12    print(f"Average execution time: {avg_time:.4f} ms")    print(f"Performance (GFLOPS): {gflops:.4f} TFLOPS")    # dtype = FP16, accum_dtype =FP32    print(f"Effective Memory Bandwidth: {((M * K + K * N) * dtype_size + M * N * accum_dtype_size) / (avg_time / 1000) / 1e9:.2f} GB/s")class SGemm:# 具体内容请参考Cutlass github上[ampere sgemm]M,N, K = 4096, 4096, 4098a = torch.randn(M, K, device="cuda", dtype=torch.float32)b = torch.randn(K, N, device="cuda", dtype=torch.float32).permute((1,0))c = torch.zeros(M, N, device="cuda", dtype=torch.float32)_a = from_dlpack(a, assumed_align=16)_b = from_dlpack(b, assumed_align=16)_c = from_dlpack(c, assumed_align=16)sgemm = SGemm()sgemm_ = cute.compile(sgemm, _a,_b,_c)sgemm_(_a,_b,_c)# verify correctnesstorch.testing.assert_close(c, a @ b.T ,atol=1e-4, rtol=1.3e-6)   benchmark(M, N , K , partial(sgemm_, _a,_b,_c), num_warmups=50, num_iterations=100)
```

在H20上执行Benchmark结果如下:

```
benchmark(M, N , K , partial(sgemm_, _a,_b,_c), num_warmups=50, num_iterations=100)Average execution time: 5.5716 msPerformance (GFLOPS): 24.6860 TFLOPSEffective Memory Bandwidth: 36.15 GB/s
```

当然这个例子只使用FMA, 并没有使用TensorCore, 性能非常差. 但是通过整个流程, 我们可以观察到cute对于Tensor和Layout的大量操作的技巧, 以及如何使用cp.async和对应的谓词张量, 正是这些复杂的cp.async和谓词张量, 以及为了进一步数据路径上的复用, 在Hopper上引入了TMA, Thread block Cluster(CGA)以及DSMEM的结构. 下一篇文章我们将详细展开这部分内容.

### Ref. CuBLASLt实现GEMM

代码如下:

```
#include <iostream>#include <vector>#include <random>#include <cuda_runtime.h>#include <cublas_v2.h>    // For cublasCreate_v2#include <cublasLt.h>#include <cuda_bf16.h> // For __nv_bfloat16// CUDA 和 cuBLAS 错误检查宏#define CHECK_CUDA(func)                                                       \    do {                                                                       \        cudaError_t err = (func);                                              \        if (err != cudaSuccess) {                                              \            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \                      << ": " << cudaGetErrorString(err) << std::endl;         \            exit(EXIT_FAILURE);                                                \        }                                                                      \    } while (0)#define CHECK_CUBLAS(func)                                                     \    do {                                                                       \        cublasStatus_t status = (func);                                        \        if (status != CUBLAS_STATUS_SUCCESS) {                                 \            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__     \                      << " (code: " << status << ")" << std::endl;             \            exit(EXIT_FAILURE);                                                \        }                                                                      \    } while (0)int main() {    // 1. 定义GEMM参数和性能测试参数    // 为了更好的性能，M, N, K 最好是8的倍数, 尤其是在使用Tensor Cores时    int M = 4096;    int N = 4096;    int K = 4096;    float alpha = 1.0f;    float beta = 0.0f;    int warmup_iterations = 5;    int timing_iterations = 50;    std::cout << "GEMM Configuration: M=" << M << ", N=" << N << ", K=" << K << std::endl;    std::cout << "Data Type: BF16, Compute Type: FP32" << std::endl;    std::cout << "Warm-up Iterations: " << warmup_iterations << std::endl;    std::cout << "Timing Iterations: " << timing_iterations << std::endl;    // 2. 在主机端初始化数据 (使用FP32, 然后转换为BF16)    std::vector<float> h_A_fp32(M * K);    std::vector<float> h_B_fp32(K * N);        std::default_random_engine generator(1234); // Use fixed seed for reproducibility    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);    for (int i = 0; i < M * K; ++i) h_A_fp32[i] = distribution(generator);    for (int i = 0; i < K * N; ++i) h_B_fp32[i] = distribution(generator);    std::vector<__nv_bfloat16> h_A_bf16(M * K);    std::vector<__nv_bfloat16> h_B_bf16(K * N);    std::vector<__nv_bfloat16> h_C_bf16(M * N, __nv_bfloat16(0.0f));    #pragma omp parallel for    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __nv_bfloat16(h_A_fp32[i]);    #pragma omp parallel for    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __nv_bfloat16(h_B_fp32[i]);    // 3. 在设备端分配内存并拷贝数据    __nv_bfloat16 *d_A, *d_B, *d_C;    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16)));    CHECK_CUDA(cudaMemcpy(d_A, h_A_bf16.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));    CHECK_CUDA(cudaMemcpy(d_B, h_B_bf16.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));    CHECK_CUDA(cudaMemcpy(d_C, h_C_bf16.data(), M * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));    // 4. cuBLASLt 工作流程    // 4.1 创建句柄    // 根据您的要求，创建cublasHandle_t。注意：这个句柄不会被cublasLt函数使用。    cublasHandle_t regular_cublas_handle;    CHECK_CUBLAS(cublasCreate_v2(&regular_cublas_handle));    // 创建cublasLt专用的句柄，这才是我们真正要用的。    cublasLtHandle_t ltHandle;    CHECK_CUBLAS(cublasLtCreate(&ltHandle));    // 4.2 创建矩阵运算描述符    cublasLtMatmulDesc_t matmulDesc;    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));    cublasOperation_t op_n = CUBLAS_OP_N; // 不转置    cublasOperation_t op_t = CUBLAS_OP_T;     CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_t, sizeof(op_t)));    // 4.3 创建矩阵布局描述符 (行主序)    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16BF, M, K, K));    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16BF, N, K, K));    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_16BF, M, N, N));    // 4.4 寻找最优算法    cublasLtMatmulPreference_t preference;    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));    size_t workspaceSize = 32 * 1024 * 1024; // 32MB workspace    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));    int returnedResults = 0;    cublasLtMatmulHeuristicResult_t heuristicResult;    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, A_desc, B_desc, C_desc, C_desc, preference, 1, &heuristicResult, &returnedResults));    if (returnedResults == 0) {        std::cerr << "No suitable algorithm found!" << std::endl; return1;    }        // 4.5 分配工作空间    void* workspace = nullptr;    if (heuristicResult.workspaceSize > 0) {        CHECK_CUDA(cudaMalloc(&workspace, heuristicResult.workspaceSize));    }    // 5. 性能测试    // 5.1 预热    std::cout << "\nRunning warm-up..." << std::endl;    for (int i = 0; i < warmup_iterations; ++i) {        CHECK_CUBLAS(cublasLtMatmul(ltHandle, matmulDesc, &alpha, d_A, A_desc, d_B, B_desc,                                     &beta, d_C, C_desc, d_C, C_desc, &heuristicResult.algo,                                    workspace, heuristicResult.workspaceSize, 0));    }    CHECK_CUDA(cudaDeviceSynchronize());    // 5.2 正式计时    std::cout << "Running performance measurement..." << std::endl;    cudaEvent_t start, stop;    CHECK_CUDA(cudaEventCreate(&start));    CHECK_CUDA(cudaEventCreate(&stop));        CHECK_CUDA(cudaEventRecord(start, 0));    for (int i = 0; i < timing_iterations; ++i) {        CHECK_CUBLAS(cublasLtMatmul(ltHandle, matmulDesc, &alpha, d_A, A_desc, d_B, B_desc,                                     &beta, d_C, C_desc, d_C, C_desc, &heuristicResult.algo,                                    workspace, heuristicResult.workspaceSize, 0));    }    CHECK_CUDA(cudaEventRecord(stop, 0));    CHECK_CUDA(cudaEventSynchronize(stop));    float ms_total = 0;    CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));    // 5.3 计算并打印性能    float ms_per_gemm = ms_total / timing_iterations;    double gflops = (2.0 * M * N * K * 1e-9) / (ms_per_gemm / 1000.0);    double tflops = gflops / 1000.0;    std::cout << "\n===== Performance Results =====" << std::endl;    std::cout << "Average time per GEMM: " << ms_per_gemm << " ms" << std::endl;    std::cout << "Achieved TFLOPS: " << tflops << std::endl;    std::cout << "=============================" << std::endl;    // 6. 将最后一次的结果拷贝回主机用于验证    std::vector<__nv_bfloat16> h_C_gpu_result_bf16(M * N);    CHECK_CUDA(cudaMemcpy(h_C_gpu_result_bf16.data(), d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));        // 7. 清理资源    if (workspace) CHECK_CUDA(cudaFree(workspace));    CHECK_CUDA(cudaEventDestroy(start));    CHECK_CUDA(cudaEventDestroy(stop));    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(A_desc));    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(B_desc));    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(C_desc));    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));    CHECK_CUBLAS(cublasLtDestroy(ltHandle));    CHECK_CUBLAS(cublasDestroy_v2(regular_cublas_handle));     CHECK_CUDA(cudaFree(d_A));    CHECK_CUDA(cudaFree(d_B));    CHECK_CUDA(cudaFree(d_C));    return0;}
```

编译执行

```
# nvcc -arch=sm_90a -lcublas -lcublasLt gemm.cu # ./a.out GEMM Configuration: M=4096, N=4096, K=4096Data Type: BF16, Compute Type: FP32Warm-up Iterations: 5Timing Iterations: 50Running warm-up...Running performance measurement...===== Performance Results =====Average time per GEMM: 1.0387 msAchieved TFLOPS: 132.318
```

参考资料

[1] 
ampere sgemm: *https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/sgemm.py*
[2] 
cute-viz: *https://github.com/NTT123/cute-viz*
[3] 
0y_predication: *https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0y_predication.md*