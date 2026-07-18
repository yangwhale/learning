# Tensor-004 TensorCore编程及优化

> 作者: zartbot  
> 日期: 2024年8月9日 23:31  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491529&idx=1&sn=12902726d6d9a8f9d66405ac6ea42fa7&chksm=f9960f0bcee1861d921cd1fe9bc6ae92d08682243857b310eeced42b2d6ccb665eac5966c250#rd

---

TensorCore编程相关的代码可以参考, 本文在这些代码的基础上进行整理, `Credit属于这些代码的作者`

Cuda-Samples[1]中的cudaTensorCoreGemm 代码

知乎:木子知的《Nvidia Tensor Core-CUDA HGEMM优化进阶》[2]

Cutlass v0.1.1[3]

《DEVELOPING CUDA KERNELS TO PUSH TENSOR CORES TO THE ABSOLUTE LIMIT ON NVIDIA A100》[4]

本文主要讲述TensorCore的供数优化相关的优化内容, 相关优化方法的测试对比

KernelGFLOPs/S相对于cuBLAS的性能Cublas90051.0100%Load From GMEM6921.47.6%Hierarchy Load49311.854.7%+ Padding SMEM53842.759.7%+ Async Copy57837.564.2%+ GMEM->SMEM Doublebuffer69233.176.8%+ SMEM->RF DoubleBuffer70111.577.8%+ Multistage with Swizzle91842.1101.9%

相关测试代码可以访问github.com/zartbot/tensorecore_gemm[5]

本文目录如下:

```
0. Recap GEMM Optimization1. TensorCore编程1.1 从一个直接GMEM加载的反例谈起1.2 GEMM的分层结构1.3 Padding缓解Bank冲突1.4 异步拷贝2. 流水线优化2.1 GMEM到SMEM, Double Buffer2.2 SMEM到RF, DoubleBuffer2.3 加深流水线3. 结语
```

因为手上暂时只有一块A10的卡测试, 因此这些TensorCore相关的优化不包含Hopper, 并且Hopper的TMA/WGMMA引入也改变了TensorCore异步编程的方法, 我们将在后续的文章中介绍Cutlass时再来补充相关的内容.

## 0. Recap GEMM Optimization

这一章开始介绍TensorCore之前, 我们先来简要回顾一下Tensor-002中介绍了GEMM优化相关的步骤.  首先是简单的内积矩阵乘法循环带来的低效率访问内存

![图片](assets/bf2e82d16d4b.png)

改为外积的优化, 仅需要把中间维度K提到最外层循环中, 降低AB矩阵的加载次数

![图片](assets/7e2db0fefdac.png)

然后进一步考虑到缓存结构, 尽量能够使用SMEM进行多次计算, 因此产生了矩阵分块乘法

![图片](assets/2e886a60a9c4.png)

紧接着我们可以考虑在分块内部进行线程并行处理

![图片](assets/717b783752c2.png)

因此引入了Thread Block Tile的结构

![图片](assets/a66ce47cc9f4.png)

为了解决一些Bank Conflict相关的问题, 我们再进一步的基于Warp拆分进行并行, 引入了Warp-Level TILE的结构

![图片](assets/76cddcf7a162.png)

最后在Warp内进行线程并行处理

![图片](assets/6e65c310effb.png)

最后整个GEMM层次化的分块和数据加载复用流程如下所示:

![图片](assets/85bda0d0a40e.png)

整个矩阵乘法多级分块的过程用循环表示如下:

![图片](assets/927fce01af5b.png)

## 1. TensorCore编程

使用TensorCore时的流程也是基本一致的, 同样需要分块从GMEM拷贝到SMEM, 然后再拆分成WarpTile拷贝到寄存器上, 只是在原有使用CUDA Core进行计算的GEMM时,换成了采用TensorCore, 如下图所示:

![图片](assets/b01d7ca57069.png)

我们首先基于Cublas来测试一下性能基线,在A10上为90.0TFLOPs

```
#include <stdio.h>#include <stdlib.h>#include "cublas_v2.h"#define M_GLOBAL 4096#define N_GLOBAL 4096#define K_GLOBAL 4096#define ITER 100void launch_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha, half beta){  cublasHandle_t handle;  cublasCreate(&handle);  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,               CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,               CUBLAS_GEMM_DEFAULT_TENSOR_OP);}int main(){    const float alpha = 1.0f;    const float beta = 0.0f;    int dev = 0;    cudaDeviceProp deviceProp;    cudaGetDeviceProperties(&deviceProp, dev);    //testError();    half *d_a, *d_b, *d_c;    cudaMalloc(&d_a, M_GLOBAL * K_GLOBAL * sizeof(half));    cudaMalloc(&d_b, K_GLOBAL * N_GLOBAL * sizeof(half));    cudaMalloc(&d_c, M_GLOBAL * N_GLOBAL * sizeof(half));    cudaEvent_t start, end;    cudaEventCreate(&start);    cudaEventCreate(&end);    cudaEventRecord(start);    for (int i = 0; i < ITER; i++)        launch_gemm(M_GLOBAL, N_GLOBAL, K_GLOBAL, d_a, d_b, d_c, alpha, beta);    cudaEventRecord(end);    cudaEventSynchronize(end);    float msec;    cudaEventElapsedTime(&msec, start, end);    long workload = long(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2 * ITER;    double avg_Gflops = ((double)workload / 1e9) / (double(msec) / 1e3);    printf("Average Performance  %10.1lf Gflops\n", avg_Gflops);    cudaFree(d_a);    cudaFree(d_b);    cudaFree(d_c);}# nvcc -lcublas -arch sm_86 00-cublas.cu -o bin/00# ./bin/00Average Performance     90051.0 Gflops# ncu --set full  -c 5 -o 12 ./bin/12==PROF== Profiling "ampere_h16816gemm_256x128_ldg..." - 0 (1/5): 0%....50%....100% - 37 passes
```

Profiling时看到调用的kernel为`ampere_h16816gemm_256x128_ldg8_stages_32x3_tn`

![图片](assets/d7977170c13d.png)

从访存来看, 调用了async copy(LDGSTS)并且bypass了L1 Cache, 矩阵乘法用了TensorCore(HMMA),矩阵加载使用了ldmatrix(LDSM).

![图片](assets/a1909d20e7f9.png)

### 1.1 从一个直接GMEM加载的反例谈起

为了证明逐级拆分搬运内存并充分利用Data Locality的做法是必须的, 我们先来测试一个反例, 直接利用TensorCore从GMEM加载计算, 分块如下图所示:

![图片](assets/6dca72d786ef.png)

```
#include "mma.h"using namespace nvcuda;#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))#define BLOCK_M 16#define BLOCK_N 16#define BLOCK_K 16#define WARP_SIZE 32using namespace nvcuda;__global__ void naiveBlockKernel(const half *A, const half *B, half *C,                                 size_t M, size_t N, size_t K){    const size_t K_tiles = CEIL_DIV(K, BLOCK_K);    const size_t c_row = blockIdx.y * BLOCK_M;    const size_t c_col = blockIdx.x * BLOCK_N;    if (c_row >= M && c_col >= N)    {        return;    }    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, half> C_frag;    wmma::fill_fragment(C_frag, 0.0);#pragma unroll    for (size_t i = 0; i < K_tiles; ++i)    {        wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::row_major> A_frag;        wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::col_major> B_frag;        wmma::load_matrix_sync(A_frag, A + c_row * K + i * BLOCK_K, K);        wmma::load_matrix_sync(B_frag, B + i * BLOCK_K + c_col * K, K);        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);    }    wmma::store_matrix_sync(C + c_row * N + c_col, C_frag, N, wmma::mem_row_major);}void launch_gemm(int M, int N, int K, half *A, half *B, half *C){    dim3 block(WARP_SIZE);    dim3 grid(CEIL_DIV(N, BLOCK_N), CEIL_DIV(M, BLOCK_M));    naiveBlockKernel<<<grid, block>>>(A, B, C, M, N, K);}# nvcc -arch sm_86 01-native.cu -o bin/01# ./bin/01Naive AveragePerformance      6921.4 Gflops
```

可以看到其峰值处理能力仅7TFLOPs. Profiling结果如下:

![图片](assets/af6a4d5cf1c1.png)

### 1.2 GEMM的分层结构

我们定义一下每个分块的名称和相应的Shape变量名, 如下图所示:

![图片](assets/77ec31b7cba6.png)

对于Thread Block Tile我们简记为BT, 对于Warp级的分块(WARP_TILE), 我们记为WT. 最后TensorCore计算的部分定义为MMA_TILE,相应的Shape 以BT_/WT_/MMA_作为前缀区分

![图片](assets/7abc1d251cad.png)

矩阵乘法的伪代码如下:

```
// Loop1A: 并行计算Thread BLOCK_TILE  Loop1A: for each m, n in M, N with step BT_M, BT_N    Loop1B: for each k in K with step BT_K        Move a chunk of A from GMEM to SMEM (As)        Move a chunk of B from GMEM to SMEM (Bs)                // Loop2A: 并行计算WARP_TILE        Loop2A: for each mm, nn in BT_M, BT_N with step WT_M, WT_N          Loop2B: for each kk in BT_K with step WT_K            Move a chunk of As from SMEM to RMEM (Ar)            Move a chunk of Bs from SMEM to RMEM (Br)            // run mma and accumulate in registers            mma(Ar, Br, accum)
```

对于MMA的Shape, 按照WMMA API通常定义为16x16x16. 对于BT/WT的Shape, 通常针对不同的输入矩阵(GLOBAL M,N,K)和不同硬件平台(SMEM大小/TensorCore的实现)相关. BlockTile_A和BlockTile_B需要加载到SMEM中, 同时要保证计算密度和Warp分布, 因此我们要对A/B的形状进行分类, 在Cutlass `dispatch_policies.h`中的分类为`Small`,`Medium`,`Large`,`Tall`,`Wide`,`Huge`等多种.

我们以一个相对较大的M=N=K=4096的HGEMM(半精度矩阵乘法)为例, 考虑到BT_A / BT_B 要放置到SMEM中, 然后结果矩阵BT_D也需要放置在SMEM,然后对齐拷贝到GMEM. 因此我们可以估计SMEM用量是否满足, 并在Launch Kernel时设置

```
void launch_gemm(size_t M, size_t N, size_t K, half *A, half *B, half *C, half alpha, half beta){    // 获取平台SHMEM SIZE    int dev_id = 0;    cudaDeviceProp dev_prop;    cudaGetDeviceProperties(&dev_prop, dev_id);    size_t SHMEM_SZ =        std::max((BT_M + BT_N) * MMA_SMEM_STRIDE_K * sizeof(half), BT_M * BT_N * sizeof(half));    if (dev_prop.sharedMemPerMultiprocessor > SHMEM_SZ)        cudaFuncSetAttribute(blockGemmKernel,                             cudaFuncAttributeMaxDynamicSharedMemorySize,                             SHMEM_SZ);    dim3 block(BT_THREAD_NUM);    dim3 grid(CEIL_DIV(M, BT_M), CEIL_DIV(N, BT_N));    blockGemmKernel<<<grid, block, SHMEM_SZ>>>(A, B, C, M, N, K);}
```

同时我们还需要考虑到WARP_TILE的SIZE, 通常的划分方式是一个Block_Tile按照2x4=8个WARP划分, 因此我们把BT_SIZE设置到了256x128, WT_SIZE为64x64, 计算时的CHUNK_K也需要根据SMEM用量进行调整,相关的变量宏定义如下:

```
// BlockTile的Shape#define BT_M 256#define BT_N 128// WMMA-TensorCore执行计算的Shape#define MMA_M 16#define MMA_N 16#define MMA_K 16// BlockTile内按照Warp 2x4拆分#define BT_ROW_WT_NUM 2 // BlockTile每一行分为2个WarpTile#define BT_COL_WT_NUM 4 // BlockTile每一列分为4个WarpTile// WarpTile的Shape#define WT_M (BT_M / BT_COL_WT_NUM) // WarpTile M-Axis的元素个数#define WT_N (BT_N / BT_ROW_WT_NUM) // WarpTile N-Axis的元素个数// 每个BlockTile的MMA Tile的数量#define BT_COL_MMA_NUM (BT_M / MMA_M) // BlockTile每一列包含的MMA_TILE的数量#define BT_ROW_MMA_NUM (BT_N / MMA_N) // BlockTile每一行包含的MMA_TILE的数量// 每个WarpTile的MMA Tile的数量#define WT_COL_MMA_NUM (WT_M / MMA_M) // WarpTile每一列包含MMA_TILE的数量#define WT_ROW_MMA_NUM (WT_N / MMA_N) // WarpTile每一行包含MMA_TILE的数量// 一个WARP有32个线程, 一个BlockTile内的线程数为BT_THREAD_NUM#define WARP_SIZE 32#define BT_WARP_NUM (BT_ROW_WT_NUM * BT_COL_WT_NUM)#define BT_THREAD_NUM (WARP_SIZE * BT_WARP_NUM)#define CHUNK_K 2      // 每次处理的MMA_TILE_K的Batch个数#define SKEW_PADDING 0 // 为了解决BankConflict增加的Padding#define MMA_SMEM_STRIDE_K (CHUNK_K * MMA_K + SKEW_PADDING)#define CHUNK_LINE_BYTES (CHUNK_K * MMA_K * sizeof(half))#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)
```

分块计算代码如下:

```
__global__ void blockGemmKernel(half *A, half *B, half *C, size_t M, size_t N, size_t K){    // 矩阵被分块成MMA_Tile的各维度个数    const size_t M_tiles = CEIL_DIV(M, MMA_N);    const size_t N_tiles = CEIL_DIV(N, MMA_M);    const size_t K_tiles = CEIL_DIV(K, MMA_K);    // 根据blockIdx查找计算的MMA_TILE的坐标    const size_t block_tile_i = blockIdx.x * BT_COL_MMA_NUM;    const size_t block_tile_j = blockIdx.y * BT_ROW_MMA_NUM;    // OOB(Out-Of-bound)判断    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles)    {        return;    }    extern __shared__ half shmem[][MMA_SMEM_STRIDE_K];    // warp_id和lane_id定义,对齐PTX相关的文档    const size_t warp_id = threadIdx.x / WARP_SIZE;    const size_t lane_id = threadIdx.x % WARP_SIZE;    // 基于MMA_TILE在WARP_LEVEL初始化C_fragment数组    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, half> C_frag[WT_COL_MMA_NUM][WT_ROW_MMA_NUM];#pragma unroll    for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)    {#pragma unroll        for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)        {            wmma::fill_fragment(C_frag[i][j], 0.0);        }    }    // B为Col-major存储, 因此Offset为Y轴的元素个数BT_M    constexpr size_t shmem_idx_b_off = BT_M;    // This pointer is used to access the C and D matrix tiles this warp computes.    half *shmem_warp_tile_ptr = &shmem[0][0] +                                (warp_id / BT_ROW_WT_NUM) * BT_N * WT_M +                                (warp_id % BT_ROW_WT_NUM) * WT_N;    // This pointer is used to stream the C and D matrices block-wide tile to and    // from shared memory    half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * MMA_M * 2 * BT_N;    // This warp's pointer to the C matrix data to copy memory from to shared    // memory.    const size_t gmem_idx =        (block_tile_i + warp_id * 2) * MMA_M * N + block_tile_j * MMA_N;    half *src_gmem_warp_stream_ptr = &C[gmem_idx];    // 加载AB矩阵的GMEM指针    const half *A_warp_ptr = &A[block_tile_i * MMA_M * K] + BT_M / BT_WARP_NUM * K * warp_id;    const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BT_N / BT_WARP_NUM * K * warp_id;    // 每次迭代的拷贝数据量    constexpr size_t A_smem_iters = BT_M / (CHUNK_COPY_LINES_PER_WARP * BT_WARP_NUM);    constexpr size_t B_smem_iters = BT_N / (CHUNK_COPY_LINES_PER_WARP * BT_WARP_NUM);// Loop for Block_Tile_K#pragma unroll    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K)    {        // 将A矩阵的Chunk从GMEM拷贝到SMEM        size_t A_smem_idx = BT_M / BT_WARP_NUM * warp_id;        int4 *A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +                           (lane_id % CHUNK_COPY_LINE_LANES);        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;#pragma unroll        for (size_t i = 0; i < A_smem_iters; ++i)        {            *((int4 *)&shmem[A_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *A_lane_ptr;            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;        }        // 将B矩阵的Chunk从GMEM拷贝到SMEM        size_t B_smem_idx = shmem_idx_b_off + BT_N / BT_WARP_NUM * warp_id;        int4 *B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +                           (lane_id % CHUNK_COPY_LINE_LANES);        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;#pragma unroll        for (size_t i = 0; i < B_smem_iters; ++i)        {            *((int4 *)&shmem[B_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *B_lane_ptr;            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;        }        // 同步等待完成拷贝        __syncthreads();        // WarpTile计算GEMM, 对加载的CHUNK处理#pragma unroll        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step)        {            wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major>                A_frag[WT_COL_MMA_NUM];            wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major>                B_frag[WT_ROW_MMA_NUM];            // 将A-Fragment从SMEM移动到寄存器#pragma unroll            for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)            {                size_t A_smem_idx = (warp_id / BT_ROW_WT_NUM) * WT_M + i * MMA_M;                const half *A_tile_ptr = &shmem[A_smem_idx][k_step * MMA_K];                wmma::load_matrix_sync(A_frag[i], A_tile_ptr, MMA_K * CHUNK_K);                // 将B-Fragment从SMEM移动到寄存器#pragma unroll                for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)                {                    if (i == 0) // B-Fragment只需加载一次, 并在后期复用                    {                        size_t B_smem_idx = shmem_idx_b_off + (warp_id % BT_ROW_WT_NUM) * WT_N + j * MMA_N;                        const half *B_tile_ptr = &shmem[B_smem_idx][k_step * MMA_K];                        wmma::load_matrix_sync(B_frag[j], B_tile_ptr, MMA_K * CHUNK_K);                    }                    // 执行TensorCore MMA计算                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);                }            }        }        // 完成GEMM计算并同步        __syncthreads();    }    // WMMA-STORE 保存结果C矩阵到SHMEM#pragma unroll    for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)    {#pragma unroll        for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)        {            half *C_tile_ptr = shmem_warp_tile_ptr + i * BT_N * MMA_M + j * MMA_N;            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], BT_N, wmma::mem_row_major);        }    }    __syncthreads();    // 对齐写回到GMEM#pragma unroll    for (size_t i = 0; i < MMA_M; ++i)    {        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =            *((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * BT_N) + lane_id % 16);    }}# nvcc -arch sm_86 02_base_tile.cu -o bin/02; ./bin/02Average Performance     49311.8 Gflops
```

性能提升到49TFLOPs/s, Profiling结果如下

![图片](assets/a3caa101fcd6.png)

可以看到还存在Bank Conflict的情况:
![图片](assets/78effc25ca61.png)

### 1.3 Padding缓解Bank冲突

我们可以在Kernel-02的基础上, 在SHMEM分配时额外再申请8B Padding, 让单个Warp访问SHMEM时处于不同区域, diff如下所示:

```
-- 02_base_tile.cu     2024-08-09 18:09:20.824826781 +0800+++ 03_padding.cu       2024-08-09 18:32:25.068925063 +0800@@ -32,8 +32,9 @@ #define BT_THREAD_NUM (WARP_SIZE * BT_WARP_NUM)  #define CHUNK_K 2      // 每次处理的MMA_TILE_K的Batch个数-#define SKEW_PADDING 0 // 为了解决BankConflict增加的Padding+#define SKEW_PADDING 8 // 为了解决BankConflict增加的Padding #define MMA_SMEM_STRIDE_K (CHUNK_K * MMA_K + SKEW_PADDING)+#define C_SMEM_STRIDE (BT_N + SKEW_PADDING)  #define CHUNK_LINE_BYTES (CHUNK_K * MMA_K * sizeof(half)) #define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))@@ -79,12 +80,12 @@      // This pointer is used to access the C and D matrix tiles this warp computes.     half *shmem_warp_tile_ptr = &shmem[0][0] +-                                (warp_id / BT_ROW_WT_NUM) * BT_N * WT_M ++                                (warp_id / BT_ROW_WT_NUM) * C_SMEM_STRIDE * WT_M +                                 (warp_id % BT_ROW_WT_NUM) * WT_N;      // This pointer is used to stream the C and D matrices block-wide tile to and     // from shared memory-    half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * MMA_M * 2 * BT_N;+    half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * MMA_M * 2 * C_SMEM_STRIDE;      // This warp's pointer to the C matrix data to copy memory from to shared     // memory.@@ -155,7 +156,7 @@                 size_t A_smem_idx = (warp_id / BT_ROW_WT_NUM) * WT_M + i * MMA_M;                 const half *A_tile_ptr = &shmem[A_smem_idx][k_step * MMA_K]; -                wmma::load_matrix_sync(A_frag[i], A_tile_ptr, MMA_K * CHUNK_K);+                wmma::load_matrix_sync(A_frag[i], A_tile_ptr, MMA_SMEM_STRIDE_K);                  // 将B-Fragment从SMEM移动到寄存器 #pragma unroll@@ -166,7 +167,7 @@                         size_t B_smem_idx = shmem_idx_b_off + (warp_id % BT_ROW_WT_NUM) * WT_N + j * MMA_N;                         const half *B_tile_ptr = &shmem[B_smem_idx][k_step * MMA_K]; -                        wmma::load_matrix_sync(B_frag[j], B_tile_ptr, MMA_K * CHUNK_K);+                        wmma::load_matrix_sync(B_frag[j], B_tile_ptr, MMA_SMEM_STRIDE_K);                     }                     // 执行TensorCore MMA计算                     wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);@@ -184,8 +185,8 @@ #pragma unroll         for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)         {-            half *C_tile_ptr = shmem_warp_tile_ptr + i * BT_N * MMA_M + j * MMA_N;-            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], BT_N, wmma::mem_row_major);+            half *C_tile_ptr = shmem_warp_tile_ptr + i * C_SMEM_STRIDE * MMA_M + j * MMA_N;+            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], C_SMEM_STRIDE, wmma::mem_row_major);         }     }     __syncthreads();@@ -195,7 +196,7 @@     for (size_t i = 0; i < MMA_M; ++i)     {         *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =-            *((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * BT_N) + lane_id % 16);+            *((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) + lane_id % 16);     } } @@ -207,7 +208,7 @@     cudaGetDeviceProperties(&dev_prop, dev_id);      size_t SHMEM_SZ =-        std::max((BT_M + BT_N) * MMA_SMEM_STRIDE_K * sizeof(half), BT_M * BT_N * sizeof(half));+        std::max((BT_M + BT_N) * MMA_SMEM_STRIDE_K * sizeof(half), BT_M * C_SMEM_STRIDE * sizeof(half));      if (dev_prop.sharedMemPerMultiprocessor > SHMEM_SZ)         cudaFuncSetAttribute(blockGemmKernel,
```

此时性能提升到 53842.7GLOPS/s, Profiling结果如下, 可以看到仅剩下少量的Store Bank conflict

![图片](assets/6e99aceda2dd.png)

但是我们来注意一下L1Cache的情况, HitRate为零, 但是又有大量的数据Load到L1, 下一步我们将利用异步拷贝来优化.

![图片](assets/3bb1e8a08395.png)

### 1.4 异步拷贝

在Ampere这一代增加了异步拷贝能力, 可以使用cp.async 绕过L1直接写入到SMEM

![图片](assets/07b90f57bd9e.png)

参考Tensor-003的文章, 我们增加异步拷贝的宏如下

```
#define CP_ASYNC_CA(dst, src, Bytes) \    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))#define CP_ASYNC_CG(dst, src, Bytes) \    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
```

主要代码修改是在将BT_A和BT_B加载到SMEM的地方, 需要注意拷贝时要对齐16B

![图片](assets/9d1342267eee.png)

我们定义THREAD_COPY_BYTES 16, 并且采用cp.async.cg来避免使用L1, diff如下

```
--- 03_padding.cu       2024-08-09 18:32:25.068925063 +0800+++ 04_async.cu 2024-08-09 19:07:39.607193053 +0800@@ -41,6 +41,9 @@ #define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES) #define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP) +#define THREAD_COPY_BYTES 16++ __global__ void blockGemmKernel(half *A, half *B, half *C, size_t M, size_t N, size_t K) {     // 矩阵被分块成MMA_Tile的各维度个数@@ -116,7 +119,10 @@ #pragma unroll         for (size_t i = 0; i < A_smem_iters; ++i)         {-            *((int4 *)&shmem[A_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *A_lane_ptr;+            uint32_t A_smem_lane_addr =+                __cvta_generic_to_shared(&shmem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;++            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);              A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);             A_smem_idx += CHUNK_COPY_LINES_PER_WARP;@@ -131,11 +137,16 @@ #pragma unroll         for (size_t i = 0; i < B_smem_iters; ++i)         {-            *((int4 *)&shmem[B_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *B_lane_ptr;+            uint32_t B_smem_lane_addr =+                __cvta_generic_to_shared(&shmem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;++            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);              B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);             B_smem_idx += CHUNK_COPY_LINES_PER_WARP;         }+        CP_ASYNC_COMMIT_GROUP();+        CP_ASYNC_WAIT_GROUP(0);          // 同步等待完成拷贝         __syncthreads();
```

性能可以提升到57837.5GFLOPs/s, Profiling结果如下, 可以看到数据已经直接进入到SMEM了

![图片](assets/403983a3752e.png)

Bank Conflict数量:

![图片](assets/4f2cb1ef2c48.png)

我们注意到GPU的计算和访存利用率都很低

![图片](assets/ed664a850857.png)

## 2. 流水线优化

可以看到当前的计算如下, 数据拷贝和计算并没有Overlap导致实际的计算和访存利用率都很低

```
    // Loop for Block_Tile_K    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K)    {        Copy A-Chunk from GMEM-->SMEM        Copy B-Chunk from GMEM-->SMEM        // WarpTile计算GEMM, 对加载的CHUNK处理        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step)            for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)               //加载A-Fragment               wmma::load_matrix_sync(Afragment)               for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)                    //加载B-Fragment                    wmma::load_matrix_sync(B-frag)                    //使用TensorCore计算                    wmma::mma_sync;    }    // WMMA-STORE 保存结果C矩阵到SHMEM    for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)        for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)            wmma::store_matrix_sync    // Store-SMEM->GMEM
```

在CUDA11以后,我们可以通过异步的方式, 逐批的拷贝内存, 并交替进行计算

### 2.1 GMEM到SMEM, Double Buffer

针对整个流程, 我们可以通过异步的方式加载
![图片](assets/cd3136960fd8.png)

伪代码如下, 详细代码可以参考Kernel-05[6]

```
Async Copy A-Chunk from GMEM-->SMEM(Buffer_1)Async Copy B-Chunk from GMEM-->SMEM(Buffer_1)Wait for Async Copy Completionfor (size_t tile_k = CHUNK_K; tile_k < K_tiles; tile_k += CHUNK_K) {   Swap Buffer_1/Buffer_2 Offset   //异步加载Buffer-2 并同时计算Buffer-1进行Overlap   Async Copy A-Chunk from GMEM-->SMEM(Buffer_2)   Async Copy B-Chunk from GMEM-->SMEM(Buffer_2)      for (size_t k_step = 0; k_step < CHUNK_K; ++k_step){        for (size_t i = 0; i < WT_COL_MMA_NUM; ++i)         {            Load-SMEM(Buffer_1)-to-A_fragment            for (size_t j = 0; j < WT_ROW_MMA_NUM; ++j)            {               Load-SMEM(Buffer_1)-to-B_fragment               wmma::mma_sync;  //使用TensorCore计算            }        }    }    Wait for Async Copy Completion}Calculate Last Buffer WarpTileWMMA-Store-to-SMEMStore-SMEM->GMEM
```

Double Buffer可以将性能提升到69233.1 GFLOPs/s, Profiling结果如下, 计算和访问内存的利用率都显著提高,从L2加载带宽也上升到了750GB/s

![图片](assets/4f4455fbaa4b.png)

![图片](assets/def5bf4050d5.png)

### 2.2 SMEM到RF, DoubleBuffer

在将数据从SMEM加载到寄存器时, 也可以进行Overlap, 具体代码参考Github Kernel-06, 原理如下图所示:

![图片](assets/806eb5c220b3.png)

此时性能提升1TFLOPs到70111.5GFLOPs/s

### 2.3 加深流水线

当我们还有足够的SMEM buffer时, 我们可以进一步加深LD数据的流水线,预取更多的数据从而避免TensorCore等待数据, 进一步隐藏延迟.
前面为了解决BankConflict,我们采取了Padding 8B的方法, 而为了更加有效的利用SMEM, 还可以采用XOR置换的Swizzle方法解决冲突, 如下图所示:

![图片](assets/ed1debe47f70.png)

这一步直接使用了知乎:木子知的mma_async_stage4.cu[7] profiling结果, L2访问带宽接近1TB/s, 性能达到Cublas的102%

![图片](assets/289835983d70.png)

但是我们也注意到和Cublas相比,Cublas峰值带宽660GB/s, 访问内存的总量多了50%(Cublas 1.61GB / This 2.42GB), 系统还有进一步调优的空间.

## 3. 结语

本文是对Tensor-003的补充, 通过一系列调优来分析TensorCore的供数/馈数相关的优化, 同时也将基于层次化矩阵分块的工作流配合TensorCore进行了解释, 为下一篇我们正式开始Cutlass相关的介绍做好了铺垫.

参考资料

[1] 
cudaTensorCoreGemm: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu
[2] 
Nvidia Tensor Core-CUDA HGEMM优化进阶: https://zhuanlan.zhihu.com/p/639297098
[3] 
Cutlass v0.1.1: https://github.com/NVIDIA/cutlass/tree/v0.1.1
[4] 
DEVELOPING CUDA KERNELS TO PUSH TENSOR CORES TO THE ABSOLUTE LIMIT ON NVIDIA A100: https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
[5] 
github.com/zartbot/tensorecore_gemm: https://github.com/zartbot/tensorcore_gemm
[6] 
Kernel-05: https://github.com/zartbot/tensorcore_gemm/blob/main/05_pipeline_gmem_to_smem.cu
[7] 
mma_async_stage4.cu: https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/mma/mma_async_stage4.cu