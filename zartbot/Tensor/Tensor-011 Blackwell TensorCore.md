# Tensor-011 Blackwell TensorCore

> 作者: zartbot  
> 日期: 2025年3月26日 11:41  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493640&idx=1&sn=98cf818a60b670f0d3d40cbbcec4deef&chksm=f995f8cacee271dc64fa46bf93649c767e36859e4f36c47cbf7013bba22db2b25d336ba54793#rd

---

### TL;DR

GTC25有一个很好的Session讲述Blackwell TensorCore编程《Programming Blackwell Tensor Cores with CuTe and CUTLASS》[1] 学习了一下, 并做了一些笔记.

总体来看, TMEM的出现使得Accumulated的结果和RMEM分离, 优点是无需像以前那样做wgmma,而只需要一个线程issue mma 然后通过SMEM mbarrier执行异步完成通知. 这样整个流水线编排调度更加灵活.  但是TMEM+TensorCore使得内存的一致性更加复杂, 需要显示的alloc/dealloc/ld/store/copy... 编程的复杂度也挺烦的...

## 1. Overview

Blackwell架构的主要变化如下:

![图片](assets/d14843dfd829.png)

### 1.1 Tensor Memory

Hopper是通过wgmma将4个warp组成group issue `wgmma.mma`指令, 结果返回到寄存器中. 而在Blackwell上新增了Tensor Memory并通过不同的lane分配给了4个warp, 每个warp可以由一个thread独立的提交,并通过SMEM的mbarrier通知完成. 很有趣的是还增加了在Epilogues阶段的异步MMA执行.

![图片](assets/0801d7f4e324.png)

TMEM的好处是对于矩阵A和D都可以存放, 然后有了更多的寄存器和SMEM的空间用于构建更深的pipeline, 同时无需像warpgroup中的4个warp一起等待完成, 只需要一个线程issue MMA指令, 然后可以更好的overlap一些softmax相关的计算.

当然TMEM也有它的坏处, 进一步的破坏了内存一致性, 因此需要显式的去分配(tcgen05.alloc/dealloc),然后也需要显示的ld/st到寄存器或者使用cp指令拷贝到SMEM

![图片](assets/af6a80b78008.png)

### 1.2 2SM TensorCore Execution

在Blackwell上增强了Distributed SMEM(DSMEM)的能力, 使得TensorCore可以同时跨越两个SM执行, 此时参数B可以通过DSMEM广播加载到TensorCore上.

![图片](assets/55154adbb22a.png)

### 1.3 Block-Scale Format

针对FP4/FP8, 增加了block scale factor的支持, 这个好评~

![图片](assets/92203499a889.png)

### 1.4 调度能力

在Hopper中,伴随着支持DSMEM实现了在一个GPC(GPU Processing Cluster)内的SM之间的硬件通信能力, 并抽象出了Thread Block Cluster(TBC)的概念,即Grid-->TBC-->CTA-->Thread的层次化结构,  一个TBC由多个CTA组成, 可以被硬件调度到一个GPC内的多个SM上执行.

![图片](assets/bb783cb6e4b3.png)

但是在Hopper上由于一个Cluster内的CTA必须要调度到一个GPC内运行, 这样就会导致某些情况下出现SM空闲的情况:

![图片](assets/80024eebf288.png)

而在Blackwell上有一些改进就是支持了Preferred TBC, 通过修改调度器, 可以在一个GPC内同时执行两种形状的cluster, 避免了SM闲置.

另一方面Tile调度在Hopper上采用静态编排机制, 当其它Grid在某个SM上执行时, 相应的Tile任务会被延后执行, 导致整个计算出现长尾

![图片](assets/105712733e72.png)

而在blackwell上支持了动态的Tile调度机制, Tile 202可以因为由其它Grid占用后, 更快的调度到下一个SM上执行, 从而避免了长尾延迟.

![图片](assets/eca571b19662.png)

## 2. Blackwell CuTe编程

CuTe采用Op/Traits/Atom/Tile的抽象结构, 例如MMA的结构如下:

![图片](assets/048734be5381.png)

COPY的结构如下:

![图片](assets/2e24c5bb65b0.png)

由于TensorCore在Hopper上结果会存入寄存器中, 因此最终会以Thread粒度处理. 相对于Hopper最大的一个变化是Blackwell针对TensorCore的计算结果可以存储在TMEM上, 因此调度粒度可以到CTA Level.

### 2.1 Basic GEMM

注: 该示例的代码在 https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu

基于Cutlass抽象, MMA Op主要是PTX指令的描述, MMA Traits携带了更多的metadata, 可以看到Hopper和Blackwell的区别, 针对MMA的Partition, 由于TMEM的出现,可以基于CTA Level的描述

![图片](assets/ba9ba3b73b82.png)

需要注意的在MMA Traits(代码位置:/include/cute/atom/mma_traits_sm100.hpp)中指定了A和B操作数使用SMEM而结果C使用TMEM. 然后由MMA Op和MMA Traits构成MMA_ATOM, 并由make_tiled_mma函数调用生成TiledMMA

![图片](assets/4349db73f767.png)

整个GEMM流程如下所示, 首先还是在GMEM创建tensor

![图片](assets/cf48c72b2a1a.png)

然后基于MMA_Tiler进行划分, MMA_Tiler在代码中的注释如下, 同理基于MMA粒度的调度后, 可以使用MMA_Coord粒度的坐标.

```cpp
auto mma_tiler = make_shape(bM, bN, bK);       // (MMA_M, MMA_N, MMA_K)
// In SM90,  the MMAs are CTA-local and perform thread-level partitioning.
// In SM100, the MMAs are Cluster-local and perform CTA-level partitioning.
// Thus, SM90 uses a cta_tiler to extract portions of the Problem for the CTA
//  and SM100 uses a mma_tiler to extract portions of the Problem for the MMA.
//  The MMA's partitioning then yeilds the CTA-local work.
// Construct the MMA grid coordinate from the CTA grid coordinate
auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                                 blockIdx.x / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
                                 blockIdx.y,                                //    MMA-N coordinate
                                 _);                                        //    MMA-K coordinate
```

整个矩阵的Tile划分如下图所示:

![图片](assets/52197920fd1e.png)

然后利用前面生成的TiledMMA进行分块, 需要注意的是由于Blackwell基于CTA level, 因此这里创建ThrMMA对象在example code里面被称为cta_mma.

![图片](assets/6f12b947ba2e.png)

然后就是分配SMEM并创建SMEM Tensor

```
struct SharedStorage{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(16) cute::uint64_t mma_barrier;  // Barrier to track MMA computation on SMEM
  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};
// Allocate SMEM
extern __shared__ char shared_memory[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
// Represent the SMEM buffers for A and B
  Tensor tCsA = shared_storage.tensor_sA();         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCsB = shared_storage.tensor_sB();         // (MmaB, NumMma_M, NumMma_K, Tiles_K)
```

![图片](assets/7f308e9da978.png)

然后就是创建MMA Fragments

```
  // MMA Fragment Allocation
// We allocate "fragments" which are SMEM descriptors that serve as inputs to cute::gemm operations.
// For tcgen05.mma operations:
// - Matrices A and B are sourced from SMEM
// - tCrA and tCrB provide descriptor views of tCsA and tCsB respectively
// - The first mode of each descriptor represents the SMEM for a single MMA operation
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);      // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);      // (MmaB, NumMma_M, NumMma_K, Tiles_K)
// TMEM Allocation
// On SM100 architecture, accumulators are stored exclusively in tensor memory (TMEM).
// ThrMma's make_fragment_C() creates a TMEM tensor with the appropriate layout for the accumulator.
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);    // (MmaC, NumMma_M, NumMma_N)
```

![图片](assets/6dbc38f908ad.png)

需要注意的是, 在Blackwell上异步执行基于SMEM mbarrier, 创建如下:

```
  // Barrier Initialization
uint32_t elect_one_thr  = cute::elect_one_sync();
uint32_t elect_one_warp = (threadIdx.x / 32 == 0);
// Barriers in SMEM initialized by a single thread.
if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */1);
  }
int mma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
  __syncthreads();                // Make sure all threads observe barrier initialization.
```

最后整个两重循环, 基于k_tile和k_block

![图片](assets/d7e21f6b1caf.png)

```
 // Execute a MmaTile_M x MmaTile_N x GEMM_K GEMM
for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
  {
    // Step 2a: Load A and B tiles
    // Using auto-vectorized copy operation:
    // - Utilizes 128 threads for parallel data transfer
    // - Copy operations are distributed efficiently across all threads
    // - CuTe can automatically determine optimal vector width
    cooperative_copy<128>(threadIdx.x, tCgA(_,_,_,k_tile), tCsA); // Load MmaTile_M x MmaTile_K A tile
    cooperative_copy<128>(threadIdx.x, tCgB(_,_,_,k_tile), tCsB); // Load MmaTile_N x MmaTile_K B tile
    // Step 2b: Execute the MMAs for this tile
    // Wait for loads to SMEM to complete with __syncthreads()
    __syncthreads();
    // tcgen05.mma instructions require single-thread execution:
    // - Only one warp performs the MMA-related loop operations
    // - CuTe operations internally manage the single-thread execution of tcgen05.mma and tcgen05.cp
    // - No explicit elect_one_sync region is needed from the user
    if (elect_one_warp) {
      // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // Ensure MMAs are completed, only then we can reuse the A and B SMEM.
      cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
    }
    // Wait MMAs to complete to avoid overwriting the A and B SMEM.
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }
```

最后执行epilogue, 从TMEM中拷贝出来, 并执行axpby等

```
  // Step 3: The Epilogue.
// Create the tiled copy operation for the accumulator (TMEM -> RMEM)
  TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy   thr_t2r_copy   = tiled_t2r_copy.get_slice(threadIdx.x);
  Tensor tDgC = thr_t2r_copy.partition_D(tCgC);                   // (CpyD, NumCpy_M, NumCpy_N)
  Tensor tDrC = make_fragment_like(tDgC);                         // (CpyD, NumCpy_M, NumCpy_N)
// Load C tensor GMEM -> RMEM
  copy(tDgC, tDrC);
  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);               // (CpyS, NumCpy_M, NumCpy_N)
  Tensor tDgD   = thr_t2r_copy.partition_D(tCgD);                 // (CpyD, NumCpy_M, NumCpy_N)
using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));              // (CpyD, NumCpy_M, NumCpy_N)
// Load TMEM -> RMEM
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);
// AXPBY RMEM -> RMEM: tDrC = alpha * tDrAcc + beta * tDrC
  axpby(alpha, tDrAcc, beta, tDrC);
// Store RMEM -> GMEM
  copy(tDrC, tDgD);
```

### 2.2 TMA based GEMM

注: 这个例子对应的源码为/examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu

主要修改是在basic gemm的基础上使用TMA加载Tensor到SMEM. 当然首先要创建TMA descriptor, 这里和Hopper是相同的, 因此复用了TMA Load Op

![图片](assets/464c5980a821.png)

```
  // Create TMA descriptors for A and B matrices
  Copy_Atom tma_atom_A = make_tma_atom(
    SM90_TMA_LOAD{},        // TMA Load Op
    mA,                     // Source GMEM tensor
    sA_layout,              // Destination SMEM layout
    select<0,2>(mma_tiler)  // MK Tiler for TMA operation
  );
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));
  Copy_Atom tma_atom_B = make_tma_atom(
      SM90_TMA_LOAD{},        // TMA Load Op
      mB,                     // Source GMEM tensor
      sB_layout,              // Destination SMEM layout
      select<1,2>(mma_tiler)  // NK Tiler for TMA operation
    );
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));   // (Gemm_N, Gemm_K)
```

![图片](assets/08066374851e.png)

TMA partition的定义如下所示:

```
  auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                    Int<0>{}, Layout<_1>{},
                                    group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
  auto [tBgB, tBsB] = tma_partition(tma_atom_B,
                                    Int<0>{}, Layout<_1>{},
                                    group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));
  // Calculate total bytes that TMA will transfer each tile to track completion
  int tma_transaction_bytes = sizeof(make_tensor_like(tAsA))
                            + sizeof(make_tensor_like(tBsB));
```

然后新增加了一个tma相关的mbarrier

```
  int tma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
```

然后在k_tile循环内采用TMA LOAD

```
  // Execute a MmaTile_M x MmaTile_N x GEMM_K GEMM
for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
  {
    // Step 2a: Load A and B tiles
    // TMA Load Operations:
    // - Execute asynchronous TMA loads with single thread
    // - Set transaction bytes and execute with barrier
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
      copy(tma_atom_A.with(shared_storage.tma_barrier), tAgA(_,k_tile), tAsA); // Load MmaTile_M x MmaTile_K A tile
      copy(tma_atom_B.with(shared_storage.tma_barrier), tBgB(_,k_tile), tBsB); // Load MmaTile_N x MmaTile_K B tile
            // Step 2b: Execute the MMAs for this tile
    // Wait for TMA loads to SMEM to complete
    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;
    }
```

![图片](assets/2eca68d54b1d.png)

### 2.3 MMA.2SM + TMA.2SM

Blackwell支持两个CTA同时执行MMA指令, MMA Op修改为cta_group::2, ThrID = Layout<_2>;

![图片](assets/5522958515df.png)

然后在数据加载上, 需要使用TMA_Multicast支持加载到多个SM

![图片](assets/16906a41b2a4.png)

针对multicast tma加载有一个单独的例子/examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu通过创建mcast_mask来实现

```
// TMA Setup
  //
//   These are TMA partitionings, which have a dedicated custom partitioner.
//   In this example, the TMA multicasts the loads across multiple CTAs.
//   Loads of A are multicasted along the N dimension of the cluster_shape_MNK and
//   Loads of B are multicasted along the M dimension of the cluster_shape_MNK.
//      Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
//   For A tensor: The group_modes<0,3> transforms the (MmaA, NumMma_M, NumMma_K, Tiles_K)-shaped tensor
//      into ((MmaA, NumMma_M, NumMma_K), Tiles_K). The partitioning only pays attention to mode-0, the MMA Tile MK.
//   For B tensor: The group_modes<0,3> transforms the (MmaB, NumMma_M, NumMma_K, Tiles_K)-shaped tensor
//      into ((MmaB, NumMma_M, NumMma_K), Tiles_K). The partitioning only pays attention to mode-0, the MMA Tile NK.
//   Simply put, the TMA will be responsible for everything in mode-0 with a single call to cute::copy.
//   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.
// Each CTA with the same m-coord will load a portion of A
// Each CTA with the same n-coord will load a portion of B
// Multicast behavior for CTA 1,2 in the cluster
//   A multicast            B multicast
//    0  1  2  3             0  1  2  3
// 0  -  -  -  -          0  -  -  X  -
// 1  X  X  X  X          1  -  -  X  -
// 2  -  -  -  -          2  -  -  X  -
// 3  -  -  -  -          3  -  -  X  -
// tma_multicast_mask_A = 0x2222
// tma_multicast_mask_B = 0x0F00
// mma_multicast_mask_C = 0x2F22
// Construct the CTA-in-Cluster coordinate for multicasting
auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
// Project the cluster_layout for tma_A along the N-modes
auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                    get<2>(cta_in_cluster_coord_vmnk),          // The CTA coordinate along N mode of the cluster
                                    make_layout(size<2>(cluster_layout_vmnk)),  // The CTA layout along N mode of the cluster
                                    group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
// Project the cluster_layout for tma_B along the M-modes
auto [tBgB, tBsB] = tma_partition(tma_atom_B,
                                    get<1>(cta_in_cluster_coord_vmnk),          // The CTA coordinate along M mode of the cluster
                                    make_layout(size<1>(cluster_layout_vmnk)),  // The CTA layout along M mode of the cluster
                                    group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));
// Project the cluster_layout and cta_coord along the N-mode to determine the multicast mask for A
uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
// Project the cluster_layout and cta_coord along the M-mode to determine the multicast mask for B
uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
// Project the cluster_layout and cta_coord along the VM + VN-modes to determine the multicast mask for C
uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                              create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
```

然后在加载时, 使用mcast_mask

```
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
      copy(tma_atom_A.with(shared_storage.tma_barrier,tma_mcast_mask_a), tAgA(_,k_tile), tAsA); // Load MmaTile_M x MmaTile_K A tile
      copy(tma_atom_B.with(shared_storage.tma_barrier,tma_mcast_mask_b), tBgB(_,k_tile), tBsB); // Load MmaTile_N x MmaTile_K B tile
```

MMA.2SM+TMA.2SM的示例代码在/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu, 和1SM的区别是MMA_Tiler为2个SM

![图片](assets/0ee77828300c.png)

然后TMA加载这些和以前都是相同的, 知识在issue mma指令的时候选择一个leading CTA执行即可

![图片](assets/4726d41ab481.png)

### 2.4 TMEM操作

TMEM累计有128个lane, 每个warp可以访问32个lane, 然后通过tcgen05.alloc/dealloc管理内存

![图片](assets/0dd79d1f07d9.png)

在软件上通过tcgen05.load/store/copy和RMEM/SMEM交互数据

![图片](assets/23ec3eda4add.png)

例如为了执行在CUDA Core上的Epilogue, 需要从TMEM加载到RMEM, 如下所示:

![图片](assets/ad8319afb9df.png)

### 2.5 TMEM Epilogue

注: 代码在 /examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu

由TiledMMA定义, MMA的accumulation结果存在TMEM中

![图片](assets/c8f1d41193cf.png)

Epilogue阶段需要创建一个TiledCopy, 将TMEM拷贝到RMEM, 然后执行axpby一类的操作

![图片](assets/1dd3c884e558.png)

## 3. Cutlass support for Blackwell

支持多种Kernel的组合

![图片](assets/dfbf629de4bb.png)

整个cutlass的概念如下所示: 最底层的是由Op Traits构成的CuTe Atoms, 然后封装成Tile based Copy和MMA, 再上面是一个collective层, 并由Collective Layer封装构成Kernel Layer

![图片](assets/39aa110dfbbe.png)

构造Kernel的方法如下所示:

![图片](assets/d86747e6e2b7.png)

从Hopper migrate到Blackwell, 只需要将Arch改为SM100, 同时TileShape从CTA改到MMA

![图片](assets/72884d2e43d2.png)

针对Blackwell增加了一些collective

![图片](assets/8be3950c987b.png)

另外Warp Specialization的情况, 由于TMEM存在, 可以在不同的warps中执行, 无需像hopper那样pingpong

![图片](assets/669a0e728a30.png)

参考资料

[1] 
Programming Blackwell Tensor Cores with CuTe and CUTLASS: *https://register.nvidia.com/flow/nvidia/gtcs25/vap/page/vsessioncatalog/session/1727748479221001aI91*