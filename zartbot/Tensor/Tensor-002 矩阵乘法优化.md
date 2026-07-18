# Tensor-002 矩阵乘法优化

> 作者: zartbot  
> 日期: 2024年7月25日 06:20  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490988&idx=1&sn=d68aaa3c838e0011ad591708b7ade22a&chksm=f9960d6ecee184782618110a4c4b98c6eaf20bc51802cb6c7f934645969ce002b932e0ca41b4#rd

---

这一篇主要谈论的是在SIMT架构下, 不使用TensorCore进行矩阵乘法计算所需要的访存相关的优化. 通过逐步迭代优化来更加深入理解GPU的性能相关的特征和访问内存优化. TensorCore相关的内容会在下一篇介绍.

测试环境为一块A10 GPU,  Driver Version: 550.54.15, CUDA Version: 12.4 . 矩阵M=N=K=4092.

KernelGFLOPs/s相对于cuBLAS的性能0: cuBLAS`14765.9`100.0%1: Naive`588.5`3.9%2: GMEM Coalescing`1165.7`7.9%3: SMEM Caching`2166.7`14.6%4: 1D Blocktiling`6082.0`41.2%5: 2D Blocktiling`11279.0`76.4%6: Vectorized Mem Access`12861.4`87.1%7: WarpTiling`14766.3`100.0%

主要参考了如下资料, 并进行了整理和测试, Credit属于如下这些文章的作者:

Simon Boehm, How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog[1]

马骏 | 旷视 MegEngine 架构师, CUDA 矩阵乘法终极优化指南[2]

nicholaswilde, CUDA SGEMM矩阵乘法优化笔记——从入门到cublas[3]

李少侠, [施工中] CUDA GEMM 理论性能分析与 kernel 优化[4]

LeiMao, CUDA Matrix Multiplication Optimization[5]

有了琦琦的棍子, 深入浅出GPU优化[6]

## 1. cuBLAS基线

我们采用cuBLAS作为性能测试基线, 测试环境是一张A10的推理卡, 测试矩阵规模如下:

```
const int M = 4092;const int K = 4092;const int N = 4092;float alpha = 1.0f;float beta = 0.5f;
```

cuBLAS测试代码如下所示:

```
#include <stdio.h>#include <stdlib.h>#include <cublas_v2.h>#include "util.h"int main(){  cudaError_t cudaStat;  // cudaMalloc status  cublasStatus_t stat;   // cuBLAS functions status  cublasHandle_t handle; // cuBLAS context  stat = cublasCreate(&handle); // initialize CUBLAS context  float *d_a, *d_b, *d_c;  cudaMalloc(&d_a, M * K * sizeof(float));  cudaMalloc(&d_b, K * N * sizeof(float));  cudaMalloc(&d_c, M * N * sizeof(float));  cudaEvent_t start, end;  cudaEventCreate(&start);  cudaEventCreate(&end);  cudaEventRecord(start);  for (int i = 0; i < ITER; i++)    stat = cublasSgemm(handle,                       CUBLAS_OP_N, CUBLAS_OP_N,                       N, M, K,                       &alpha, d_b, N,                       d_a, K, &beta, d_c, N);  cudaEventRecord(end);  cudaEventSynchronize(end);  float msec;  cudaEventElapsedTime(&msec, start, end);  long workload = long(M) * N * K * 2 * ITER;  double avg_Gflops = ((double)workload / 1e9) / (double(msec) / 1e3);  printf("cuBLAS AveragePerformance  %10.1lf Gflops\n", avg_Gflops);  cudaFree(d_a);  cudaFree(d_b);  cudaFree(d_c);  cublasDestroy(handle); // destroy CUBLAS context}
```

## 2. 简单实现

前一章讲述郭, 按照三层的循环结构进行编程. 我们从结果C矩阵来看, 可以编排每个线程负责一个位置的值

### 2.1 线程编排

CUDA通过Grid/Block的方式来组织线程, 如下图所示:

![图片](assets/14a36bf60a63.png)

针对当前任务, 我们可以把Z这个维度定义为1, 以2D方式编排线程, 我们选择一个BLOCK包含`32 * 32`个线程,则总共需要的Grid数量如下所示:

```
// 需要的Grid数量为Ceil(M/32) * ceil(N/32)dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);// 每个BLOCK有32 * 32 = 1024线程dim3 blockDim(32, 32, 1);//调用Kernelsgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

整个乘法工作流程如下, 每个线程负责C中一个位置所需要的内积计算:

![图片](assets/3fe68886d7e8.png)

### 2.2 内积计算

这个块在C中的坐标, 如下所示, 我们需要在线程内根据blockIdx和ThreadIdx推算出
![图片](assets/03184032f6a2.png)

同时我们分配BLOCK的时候, 还有部分BLOCK中的THREAD会越出矩阵的边界(如上图中的红色部分), 因此需要一个判断条件控制执行, 最终代码如下

```
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,                            const float *B, float beta, float *C) {    // 计算线程负责的块在C中的坐标  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // 处理边界条件, 由于Grid划分按照Ceil_DIV,边界的BLOCK中有些线程需要禁止处理超越矩阵边界的数据  if (x < M && y < N) {    float tmp = 0.0;    for (int i = 0; i < K; ++i) {      tmp += A[x * K + i] * B[i * N + y];    }        // C = α*(A@B)+β*C    C[x * N + y] = alpha * tmp + beta * C[x * N + y];  }}
```

但是这种方式获得的588.5GFLOPS,仅有cuBLAS的4%

### 2.3 计算运行时间下界分析

整个矩阵规模为M=K=N=4092

浮点计算量为

计算为FP32, 因此需要读取的数据为

总共需要存储的数据为

累计最小需要268MB访存

A10 GPU从官方文档可知, 其FP32的峰值浮点计算能力为30TFLOPs/s全局内存带宽为768GB/s. 按照峰值计算需要4.5ms,  按照峰值内存带宽需要0.34ms进行内存传输. 因此只要我们传输数据的量小于 10x 268MB则为Compute-Bound的算子.

### 2.4 简单模式访存问题

在一个Kernel中, 我们来看同一个BLOCK中的两个线程ThreadId(0,0)和ThreadId(0,1),如下图所示,它们都将加载B的同一列,但加载不同的A的行:

![图片](assets/6b10672830d5.png)

如果假设完全没有Cache, 则每个线程需要加载个浮点数, 总共有个线程,则累计需要产生548GB的内存访问. 因此我们需要优化Kernel的内存访问模式, 以便将全局内存(GMEM)的访问尽量的合并在一起,降低访问数据量.

## 3. 全局内存合并(GMEM Coalescing)

对于一个GPU, 我们通常将32个相邻的线程构成一个warp, 如果每个线程从全局内存加载FP32数据,如果访问数据的地址是连续的则可以合并成32 * 4B=128B的单个Load事务中. 如下图所示, 矩阵按照行排序时, 如果此时访问某一列将会出现不连续的地址. 对于前一章节的简单实现就会导致大量的32B LD从而影响性能.

![图片](assets/7a9333046eab.png)

一种方式是将B和C矩阵按照列排序的方式存储, 当然我们也可以通过重新编排thread的方式来处理, 如下所示, 我们对thread和block的索引同时修改即可实现.

```
__global__ void gmem_coalescing_gemm(int M, int N, int K, float alpha, const float *A,                           const float *B, float beta, float *C){  //交换矩阵C的X/Y索引  const uint y = blockIdx.x * blockDim.x + threadIdx.x;  const uint x = blockIdx.y * blockDim.y + threadIdx.y;  if (x < M && y < N)  {    float tmp = 0.0;    for (int i = 0; i < K; ++i)    {      tmp += A[x * K + i] * B[i * N + y];    }    // C = α*(A@B)+β*C    C[x * N + y] = alpha * tmp + beta * C[x * N + y];     }}void launch_gemm(int M, int N, int K, float alpha, const float *A,                 const float *B, float beta, float *C){  //交换Grid编排  dim3 gridDim(ceil(N / 32), ceil(M / 32), 1);  dim3 blockDim(32, 32, 1);  gmem_coalescing_gemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}
```

简单交换一下后性能可以提升到1165.7GFlops. 通过Profiling可以看到,在简单实现中访问内存带宽为52.29GB/s

![图片](assets/b8ca6a0d1fa1.png)

而采用全局内存合并访问, 则可以到113.33GB/s

![图片](assets/0ad920335bd2.png)

## 4. SMEM Cache-Blocking

我们来看一下GPU的缓存层次架构, 在每个SM内还有一块Shared Memory(SMEM), 如下图所示:

![图片](assets/3272da711db6.png)

通过李少侠老师的测试代码[7] A10 共享内存的带宽大概在15.6TB/s

```
shared memory accessed: 2097152 byteduration: 19348 cyclesshared memory bandwidth per SM (measured): 108.391151 byte/cycleshared memory bandwidth per SM (theoretical): 128 byte/cyclestandard clock frequency: 1695 MHzSM: 72whole chip shared memory bandwidth (theoretical): 15621.120117 GB/s
```

因此, 我们将从全局内存(GMEM)中加载A和B块到共享内存中, 然后对这两个块执行尽可能多的计算:

![图片](assets/87103094a522.png)

我们将沿着 A 的列和 B 的行移动块，对 C 执行部分求和，直到计算出结果。

```
template <const int CHUNK_SIZE>__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,                                       const float *A, const float *B,                                       float beta, float *C) {  // 矩阵C按照BLOCK划分, cRow和cCol为该线程所在Block对应的Block的行号和列号  const uint cRow = blockIdx.x;  const uint cCol = blockIdx.y;  // 分配共享内存, 共享内存可以被Block内所有的thread访问  __shared__ float As[CHUNK_SIZE * CHUNK_SIZE];  __shared__ float Bs[CHUNK_SIZE * CHUNK_SIZE];  // BLOCK内线程在启动内核的时候分配的blockdim仅有一个维度  // 通过threadIdx找到线程在BLOCK内部对应的行和列  const uint threadCol = threadIdx.x % CHUNK_SIZE;  const uint threadRow = threadIdx.x / CHUNK_SIZE;  // 基于cRow和cCol计算矩阵开始的指针位置  A += cRow * CHUNK_SIZE * K;                    // row=cRow, col=0  B += cCol * CHUNK_SIZE;                        // row=0, col=cCol  C += cRow * CHUNK_SIZE * N + cCol * CHUNK_SIZE; // row=cRow, col=cCol  float tmp = 0.0;  for (int bkIdx = 0; bkIdx < K; bkIdx += CHUNK_SIZE) {    // 每个线程加载A和B的一个元素, 由于threadIdx.x是一个连续的分布    // 因此访问GMEM是可以合并的    As[threadRow * CHUNK_SIZE + threadCol] = A[threadRow * K + threadCol];    Bs[threadRow * CHUNK_SIZE + threadCol] = B[threadRow * N + threadCol];    // 同步等待所有thread完成数据加载    __syncthreads();    //将数据移动到下一个CHUNK    A += CHUNK_SIZE;    B += CHUNK_SIZE * N;    // 进行BLOCK Level的内积计算    for (int dotIdx = 0; dotIdx < CHUNK_SIZE; ++dotIdx) {      tmp += As[threadRow * CHUNK_SIZE + dotIdx] *             Bs[dotIdx * CHUNK_SIZE + threadCol];    }    // 考虑Cache影响, 需要在执行下一次加载时再进行一次同步    __syncthreads();  }  C[threadRow * N + threadCol] =      alpha * tmp + beta * C[threadRow * N + threadCol];}void launch_gemm(int M, int N, int K, float alpha, const float *A,                 const float *B, float beta, float *C){  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));  dim3 blockDim(32 * 32);  sgemm_shared_mem_block<32>      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}
```

测试可以看到性能提升到2166GFLOPS, 同时我们通过Profiling看到访问内存的请求转移到了SMEM

![图片](assets/440c44b28e70.png)

但是其性能比cuBLAS和理论峰值还差距非常大, 因此我们需要进一步进行查看, 主要的指令为LDS

![图片](assets/89e7d331acc3.png)

但是WARP调度来看, 主要是 Stall MIO Throttle

![图片](assets/22b96ce3e2cf.png)

Stall MIO Throttle的含义是: Warp因等待 MIO（内存输入/输出）指令队列而产生的停滞, 对于频繁的访问SMEM场景会导致这种情况发生, 因此我们下一步的目标就是需要去优化减少KERNEL发出的LDS指令. 那么我们就需要每个线程计算多个元素.

## 5. 1D BlockTiling

因此, 我们将扩大CHUNK_SIZE, 利用SMEM缓存`BM*BK + BN*BK = 64*8 + 64*8 = 1024`个浮点数, 如下图所示:

![图片](assets/9d71064dc18f.png)

SMEM加载基本相同, 但是在每个线程内部构建(dotIdx/resIdx)两个循环:

```
// 在寄存器文件内分配allocate thread-local cache for results in registerfilefloat threadResults[TM] = {0.0};// 基于BLOCKTILE的外部循环for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {  // 访问内存和以前相同  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];  __syncthreads();  // 将数据移动到下一个BLOCKTILE  A += BK;  B += BK * N;  //每个线程的计算任务, 分为dotIdx/resIdx两个循环  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {    // 为了复用Bs矩阵, 我们把内积循环放在外层, 并缓存到Btmp中    float Btmp = Bs[dotIdx * BN + threadCol];    for (uint resIdx = 0; resIdx < TM; ++resIdx) {      threadResults[resIdx] +=          As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;    }  }  __syncthreads();}
```

整个Kernel代码如下

```
template <const int BM, const int BN, const int BK, const int TM>__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,                                   const float *A, const float *B, float beta,                                   float *C) {  // 交换BLOCK对应的行列编排,使得B矩阵列访问连续.   const uint cRow = blockIdx.y;  const uint cCol = blockIdx.x;  const int threadCol = threadIdx.x % BN;  const int threadRow = threadIdx.x / BN;  // 分配SMEM  __shared__ float As[BM * BK];  __shared__ float Bs[BK * BN];  // 移动BLOCKTILE的指针  A += cRow * BM * K;  B += cCol * BN;  C += cRow * BM * N + cCol * BN;  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing  const uint innerRowA = threadIdx.x / BK;  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing  const uint innerRowB = threadIdx.x / BN;  // allocate thread-local cache for results in registerfile  float threadResults[TM] = {0.0};  // outer loop over block tiles  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {    // populate the SMEM caches    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];    __syncthreads();    // advance blocktile    A += BK;    B += BK * N;    // calculate per-thread results    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {      // we make the dotproduct loop the outside loop, which facilitates      // reuse of the Bs entry, which we can cache in a tmp var.      float tmpB = Bs[dotIdx * BN + threadCol];      for (uint resIdx = 0; resIdx < TM; ++resIdx) {        threadResults[resIdx] +=            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;      }    }    __syncthreads();  }  // write out the results  for (uint resIdx = 0; resIdx < TM; ++resIdx) {    C[(threadRow * TM + resIdx) * N + threadCol] =        alpha * threadResults[resIdx] +        beta * C[(threadRow * TM + resIdx) * N + threadCol];  }}void launch_gemm(int M, int N, int K, float alpha, const float *A,                 const float *B, float beta, float *C){  const uint BM = 64;  const uint BN = 64;  const uint BK = 8;  const uint TM = 8;  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));  dim3 blockDim((BM * BN) / TM);  sgemm1DBlocktiling<BM, BN, BK, TM>      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}
```

执行可以发现, 性能提升到了6082GFlops, 相对于前一个Kernel提升了3倍, WARP MIO Stall相对于原来22个Cycle也有很大的改善.

![图片](assets/ed8ad3c10abd.png)

同时LDS指令也大幅度减少

![图片](assets/d24a01b2f55a.png)

前一个Kernel中:

GMEM外部循环K/32次 * 2次LOAD

SMEM外部循环K/32次 * CHUNKSIZE(32) * 2次LOAD

每个结果内存访问: GMEM: K/16, SMEM K * 2

而新的1D BlockingTiling, 每个线程计算8个结果

GMEM外部循环K/8次 * 2次LOAD

SMEM外部循环K/8次 * BK * (1+TM),注: BK=8,TM=8, (1+TM)为BLOCK-B一次, BLOCK-A TM次

每个结果内存访问: GMEM: K/32, SMEM K * 9/8

另外我们注意到一个编译器的优化, 对于Bs的SMEM LOAD实现了向量化

```
 for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {      // we make the dotproduct loop the outside loop, which facilitates      // reuse of the Bs entry, which we can cache in a tmp var.      float tmpB = Bs[dotIdx * BN + threadCol];      for (uint resIdx = 0; resIdx < TM; ++resIdx) {        threadResults[resIdx] +=            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;      }  }    LDS     R26, [R35.X4+0x800] // a 32b load from AsLDS.128 R8,  [R2]           // a 128b load from BsLDS.128 R12, [R2+0x20] LDS     R24, [R35.X4+0x900] LDS.128 R20, [R2+0x60] LDS     R36, [R35.X4+0xb00] LDS.128 R16, [R2+0x40] LDS.128 R4,  [R2+0x80] LDS     R38, [R35.X4+0xd00] 
```

## 6. 2D BlockTiling

此时,我们还需要更高的计算强度(Arithmetic Intensity)来减缓Stall. 如下图所示, 我们可以通过一个线程计算多个结果的方式来降低LD/ST

![图片](assets/7328efc7d079.png)

当然我们可能会简单的考虑只增加一个维度,但是它访存的数量会大于2D tile, 如下所示:

![图片](assets/aa1e4b2e156b.png)

我们将结果矩阵C按照Block划分后, 每个Thread负责`TM * TN`个块的数据,如下所示:

![图片](assets/7d7d86cdcf34.png)

相对于1D BlockTiling而言, 内部构成了三个循环(dotIdx/ResIdxM/ResIdxN),如下所示

```
// allocate thread-local cache for results in registerfilefloat threadResults[TM * TN] = {0.0};// register caches for As and Bsfloat regM[TM] = {0.0};float regN[TN] = {0.0};// outer-most loop over block tilesfor (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {  // populate the SMEM caches  for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {    As[(innerRowA + loadOffset) * BK + innerColA] =        A[(innerRowA + loadOffset) * K + innerColA];  }  for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {    Bs[(innerRowB + loadOffset) * BN + innerColB] =        B[(innerRowB + loadOffset) * N + innerColB];  }  __syncthreads();  // advance blocktile  A += BK;     // move BK columns to right  B += BK * N; // move BK rows down  // calculate per-thread results  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {    // load relevant As & Bs entries into registers    for (uint i = 0; i < TM; ++i) {      regM[i] = As[(threadRow * TM + i) * BK + dotIdx];    }    for (uint i = 0; i < TN; ++i) {      regN[i] = Bs[dotIdx * BN + threadCol * TN + i];    }    // perform outer product on register cache, accumulate    // into threadResults    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {      for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {        threadResults[resIdxM * TN + resIdxN] +=            regM[resIdxM] * regN[resIdxN];      }    }  }  __syncthreads();}
```

可以注意到, 由于每个线程现在处理 `TM * TN` 个元素, 因此 `As Tile`和 `Bs Tile`要多次加载.
由于

```
threadCol = threadIdx.x % (BN / TN)threadRow = threadIdx.x / (BN / TN)
```

即相邻线程对应 `Bs Tile` 的不同列线程分块、`As Tile`的同一行线程分块,
dotIdx循环仍然是线程块分片沿 K 维度逐一计算, 即每次 As 处理一列, Bs 处理一行.两个内层的循环将重复使用的线程分片元素加载至寄存器.
最后通过resIdxM/resIdxN循环计算线程块分片的结果. 即依次迭代 As dotIdx 列的`TM`个元素和 Bs dotIdx行的`TN`个元素, 计算总共 TM * TN个值.

由于通过将dotIdx移到最外层减少了SMEM的访问次数, 对于三重循环的结构有一个补充的解释图,如下:

![图片](assets/14dd7b755cff.png)

此时性能达到11279GFLOPs,性能比1D BlockTiling又接近翻倍. 我们通过Profiling看到Warp Stall MIO throttle的现象得到了改善.

![图片](assets/1fbc0b6755f7.png)

LDS数量也大幅度下降

![图片](assets/163d55d9c3b2.png)

## 7. 矢量化SMEM/GMEM访问

前面一个优化通过转置As使得可以通过矢量化的指令(LDS.128)从As加载, 但是我们还可以通过float4的向量数据类型对GMEM的所有LD/ST进行向量化处理.例如处理行读和转置两种模式

```
//向量读reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];    // GMEM到SMEM时同时进行转置float4 tmp =    reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
```

`reinterpret_cast<float4 *>`的目的是显示的通知编译器float* B是128b对齐的, 这将导致32b GMEM LD/ST指令(LDG.E & STG.E)替换成为LDG.E.128 & STG.E.128,而且比手动展开的四个LD更快.

```
template <const int BM, const int BN, const int BK, const int TM, const int TN>__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,                               float *B, float beta, float *C) {  const uint cRow = blockIdx.y;  const uint cCol = blockIdx.x;  // BN/TN are the number of threads to span a column  const int threadCol = threadIdx.x % (BN / TN);  const int threadRow = threadIdx.x / (BN / TN);  // allocate space for the current blocktile in smem  __shared__ float As[BM * BK];  __shared__ float Bs[BK * BN];  // Move blocktile to beginning of A's row and B's column  A += cRow * BM * K;  B += cCol * BN;  C += cRow * BM * N + cCol * BN;  // calculating the indices that this thread will load into SMEM  // we'll load 128bit / 32bit = 4 elements per thread at each step  const uint innerRowA = threadIdx.x / (BK / 4);  const uint innerColA = threadIdx.x % (BK / 4);  const uint innerRowB = threadIdx.x / (BN / 4);  const uint innerColB = threadIdx.x % (BN / 4);  // allocate thread-local cache for results in registerfile  float threadResults[TM * TN] = {0.0};  float regM[TM] = {0.0};  float regN[TN] = {0.0};  // outer-most loop over block tiles  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {        //向量化加载并进行转置操作    float4 tmp =        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];    __syncthreads();    // advance blocktile    A += BK;     // move BK columns to right    B += BK * N; // move BK rows down    // calculate per-thread results    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {      // block into registers      for (uint i = 0; i < TM; ++i) {        regM[i] = As[dotIdx * BM + threadRow * TM + i];      }      for (uint i = 0; i < TN; ++i) {        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];      }      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {          threadResults[resIdxM * TN + resIdxN] +=              regM[resIdxM] * regN[resIdxN];        }      }    }    __syncthreads();  }  // write out the results  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {            // 向量化的加载C      float4 tmp = reinterpret_cast<float4 *>(          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];            // 更新结果到寄存器文件      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;            // 向量化写回      reinterpret_cast<float4 *>(          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =          tmp;    }  }}void launch_gemm(int M, int N, int K, float alpha,  float *A,                  float *B, float beta, float *C){  const uint BK = 8;  const uint TM = 8;  const uint TN = 8;    const uint BM = 128;    const uint BN = 128;    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));    dim3 blockDim((BM * BN) / (TM * TN));    sgemmVectorize<BM, BN, BK, TM, TN>        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}
```

最终优化后的性能为12861.4GFLops,接近cuBLAS实现的87%. 此时Profiling显示还有一些Bank Conflict:

![图片](assets/884b0435c671.png)

![图片](assets/4217e795bab0.png)

这些问题我们将在下一节探讨.

## 8. Bank Conflict

### 8.1 什么是Bank冲突

为了实现并发访问的高内存带宽，共享内存被划分为可以同时访问的同等大小的内存模块(Bank), 因此跨越n个不同内存组所在地址的数据都可以任意的并行加载和存储. 我们以简化的一个4线程+4Bank的Warp做图例, 当每个warp中的线程都以Offset=1连续访问数据时, 正好可以一次读完不出现Bank冲突

![图片](assets/451a869a0875.png)

而当Offset=2时, Thread-0和Thread-2, 以及 Thread-1和Thread-3会访问相同Bank的内存,导致冲突使得访问延迟增加

![图片](assets/9cd62c265327.png)

在一个包含32线程的Warp中, Bank冲突如下, 从左至右分别为Offset=1,2,3

![图片](assets/5f1b210bcdc8.png)

Bank冲突带来的性能影响如下图所示:

![图片](assets/9d64e0191325.png)

解决的办法就是我们尽量对线程访问内存进行亲和性编排, 通常这种技术被称为Swizzle. Swizzle更详细的内容会在CuTe Layout中讲述.

### 8.2 Bank冲突分析

我们来看Thread的编排

```
  const int threadCol = threadIdx.x % (BN / TN);  const int threadRow = threadIdx.x / (BN / TN);
```

`BN/TN = 128/8 = 16`, 因此在一个WARP中的32个线程, 每16个有相同的`threadRow`和不同的`threadCol`.

对于`As`的加载`As[dotIdx * BM + threadRow * TM + i]`, dotIdx和i不变时, 相当于16个线程访问同一个地址, 另外16个线程访问Offset=TM的地址, bank相差8, 由于WARP的广播机制, 不会产生Bank冲突.

对于`Bs`的加载`Bs[dotIdx * BN + threadCol * TN + i]`, 由于`threadCol`不相同,相邻线程访问地址差为BN=8 bank, 因此threadIdx每相差4都会产生一个bank冲突.

Simon Boehm中进行了一个修复[8]做了一个16列8行的转换.

![图片](assets/f13a13fb0e5b.png)

```
-    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =-        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];+    // "linearize" Bs while storing it+    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];+    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;+    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;+    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;+    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;// block into registers      for (uint i = 0; i < TM; ++i) {        regM[i] = As[dotIdx * BM + threadRow * TM + i];      }      for (uint i = 0; i < TN; ++i) {-        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];+        regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];      }
```

修改后, 访问`Bs[(dotIdx * 8 + i) * 16 + threadCol]`在同一个Warp中, 前16个线程读取的元素彼此差1,后16个线程和前16个访问地址相同. 但是需要从GMEM加载Bs时, 但是写入`Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2]`时我们注意到

```
  const uint innerRowB = threadIdx.x / (BN / 4);  const uint innerColB = threadIdx.x % (BN / 4);
```

因此`threadIdx`相差1时, 实际的地址相差了64个地址, 因此还是有Bank冲突. 通过Profiling也可以看到,LD的bank冲突解决了,但是还遗留了ST的bank冲突

![图片](assets/c115f8b32b89.png)

但是相对于前一个Kernel, ST的冲突更高了
![图片](assets/51251835f0dd.png)

作者在A6000上测试性能有提升,但在A10上测试发现性能相对于前一个Kernel还有下降.

## 9. WarpTiling

在前述的Kernel中, 可以看到有三个循环
![图片](assets/af704ccfbdcb.png)

BlockTiling和ThreadTiling显著提升了性能, 但还是存在一些访问内存Bank冲突的问题. 从GPU硬件结构来看Warp是映射到SM上, 由Warp Scheduler进行调度. 共享内存的bank冲突仅发生在同一个warp内的thread. 因此在这个基础上再进行一次Warp Tiling.

![图片](assets/b6d62ed29d63.png)

通过这样的方式, BlockTiling把数据分块放置在不同的SM上执行, WarpTiling可以使得在SM内通过Warp调度器上进行调度. 而ThreadTiling的指令可以在相同的CUDA Core上进行指令级并行执行.

```
// dotIdx loops over contents of SMEMfor (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {  // populate registers for this thread's part of the warptile  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {    for (uint i = 0; i < TM; ++i) {      regM[wSubRowIdx * TM + i] =          As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +             threadRowInWarp * TM + i];    }  }  for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {    for (uint i = 0; i < TN; ++i) {      regN[wSubColIdx * TN + i] =          Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +             threadColInWarp * TN + i];    }  }  // execute warptile matmul. Later this will map well to  // warp-wide matrix instructions, executed on tensor cores.  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {      // calculate per-thread results with register-cache locality      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {          threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +                        (wSubColIdx * TN) + resIdxN] +=              regM[wSubRowIdx * TM + resIdxM] *              regN[wSubColIdx * TN + resIdxN];        }      }    }  }}
```

每个WARP将计算`(WSUBN * WNITER) x (WSUBM * WMITER)`的块, 每个线程计算`WNITER * WMITER`个`TM*TN`的块

其中`WM=32`,`WN=64`表示矩阵C按照Warp分片的大小.warp的编排如下

```
  const uint warpIdx = threadIdx.x / WARPSIZE;   const uint warpCol = warpIdx % (BN / WN);  const uint warpRow = warpIdx / (BN / WN);
```

![图片](assets/04980f5f8fb5.png)

`WNITER=4` , `WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER)`在WarpTile中按照`WMITER`和`WNITER`进行迭代, `WSUBM = WM / WMITER (32/2 = 16)`, `WSUBN = WN / WNITER (64/2 = 32)`表示WARP每次迭代时, M和N维度需要处理的元素数.

```
  // size of the warp subtile  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);  constexpr uint WSUBM = WM / WMITER; // 64/2=32  constexpr uint WSUBN = WN / WNITER; // 32/2=16
```

![图片](assets/ec3bceb28a85.png)

对于每个线程在WARPTile内进行索引

```
  // Placement of the thread in the warp subtile  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4
```

![图片](assets/12fa7834932b.png)

线程内处理函数如下:

```
template <const int BM, const int BN, const int BK, const int rowStrideA,          const int rowStrideB>__device__ void loadFromGmem(int N, int K, const float *A, const float *B,                             float *As, float *Bs, int innerRowA, int innerColA,                             int innerRowB, int innerColB) {  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {    const float4 tmp = reinterpret_cast<const float4 *>(        &A[(innerRowA + offset) * K + innerColA * 4])[0];    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;  }  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {    reinterpret_cast<float4 *>(        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =        reinterpret_cast<const float4 *>(            &B[(innerRowB + offset) * N + innerColB * 4])[0];  }}
```

数据加载时增加了一个Offset循环

```
  const uint innerRowA = threadIdx.x / (BK / 4);  const uint innerColA = threadIdx.x % (BK / 4);  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;  const uint innerRowB = threadIdx.x / (BN / 4);  const uint innerColB = threadIdx.x % (BN / 4);    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);
```

需要注意的是对于GMEM加载, `As`的threadIdx相差1会导致innerColA也相差1, 因此`(innerColA * 4 + 0) * BM` 会导致`4* BM= 512`对应于同一个bank, 因此会产生Bank冲突.

```
template <const int BM, const int BN, const int BK, const int WM, const int WN,          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,          const int TM, const int TN>__device__ voidprocessFromSmem(float *regM, float *regN, float *threadResults, const float *As,                const float *Bs, const uint warpRow, const uint warpCol,                const uint threadRowInWarp, const uint threadColInWarp) {  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {    // populate registers for whole warptile    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {      for (uint i = 0; i < TM; ++i) {        regM[wSubRowIdx * TM + i] =            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +               threadRowInWarp * TM + i];      }    }    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {      for (uint i = 0; i < TN; ++i) {        regN[wSubColIdx * TN + i] =            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +               threadColInWarp * TN + i];      }    }    // execute warptile matmul    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {        // calculate per-thread results        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +                          (wSubColIdx * TN) + resIdxN] +=                regM[wSubRowIdx * TM + resIdxM] *                regN[wSubColIdx * TN + resIdxN];          }        }      }    }  }}
```

整个流程图如下所示:
![图片](assets/1afa59f9b312.png)

```
/* * @tparam BM The threadblock size for M dimension SMEM caching. * @tparam BN The threadblock size for N dimension SMEM caching. * @tparam BK The threadblock size for K dimension SMEM caching. * @tparam WM M dim of continuous tile computed by each warp * @tparam WN N dim of continuous tile computed by each warp * @tparam WMITER The number of subwarp tiling steps in M dimension. * @tparam WNITER The number of subwarp tiling steps in N dimension. * @tparam TM The per-thread tile size for M dimension. * @tparam TN The per-thread tile size for N dimension. */template <const int BM, const int BN, const int BK, const int WM, const int WN,          const int WNITER, const int TM, const int TN, const int NUM_THREADS>__global__ void __launch_bounds__(NUM_THREADS)    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,                    float beta, float *C) {  const uint cRow = blockIdx.y;  const uint cCol = blockIdx.x;  // 在Thread BlockTile中放置Warp  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in  const uint warpCol = warpIdx % (BN / WN);  const uint warpRow = warpIdx / (BN / WN);  // size of the warp subtile  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);  constexpr uint WSUBM = WM / WMITER; // 64/2=32  constexpr uint WSUBN = WN / WNITER; // 32/2=16  // 在Warp SubTile中放置Thread  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4  // 分配SMEM  __shared__ float As[BM * BK];  __shared__ float Bs[BK * BN];  // Move blocktile to beginning of A's row and B's column  A += cRow * BM * K;  B += cCol * BN;  // Move C_ptr to warp's output tile  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;  // calculating the indices that this thread will load into SMEM  // we'll load 128bit / 32bit = 4 elements per thread at each step  const uint innerRowA = threadIdx.x / (BK / 4);  const uint innerColA = threadIdx.x % (BK / 4);  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;  const uint innerRowB = threadIdx.x / (BN / 4);  const uint innerColB = threadIdx.x % (BN / 4);  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);  // allocate thread-local cache for results in registerfile  float threadResults[WMITER * TM * WNITER * TN] = {0.0};    // we cache into registers on the warptile level  float regM[WMITER * TM] = {0.0};  float regN[WNITER * TN] = {0.0};  // outer-most loop over block tiles  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);    __syncthreads();        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,                            threadRowInWarp, threadColInWarp);    A += BK;     // move BK columns to right    B += BK * N; // move BK rows down    __syncthreads();  }  // write out the results  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {      // move C pointer to current warp subtile      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {          // load C vector into registers          float4 tmp = reinterpret_cast<float4 *>(              &C_interim[(threadRowInWarp * TM + resIdxM) * N +                         threadColInWarp * TN + resIdxN])[0];          // perform GEMM update in reg          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +                        wSubColIdx * TN + resIdxN;          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;          // write back          reinterpret_cast<float4 *>(              &C_interim[(threadRowInWarp * TM + resIdxM) * N +                         threadColInWarp * TN + resIdxN])[0] = tmp;        }      }    }  }}void launch_gemm(int M, int N, int K, float alpha, float *A,                 float *B, float beta, float *C){  const uint NUM_THREADS = 128;  const uint BN = 128;  const uint BM = 128;  const uint BK = 16;  const uint WN = 64;  const uint WM = 64;  const uint WNITER = 4;  const uint TN = 4;  const uint TM = 8;  dim3 blockDim(NUM_THREADS);  constexpr uint NUM_WARPS = NUM_THREADS / 32;  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));  sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM,                  TN, NUM_THREADS>      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);}
```

基于WARPTiling的性能在A10上测试结果为14766.3GFlops,已经基本和cuBLAS结果一致了.

## 10. Double Buffering

前一个Kernel中, 数据加载和处理还是同步阻塞的模式

```
  // outer-most loop over block tiles  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {    //加载数据    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);    __syncthreads();        //同步阻塞后处理数据    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,                            threadRowInWarp, threadColInWarp);    A += BK;     // move BK columns to right    B += BK * N; // move BK rows down    __syncthreads();  }
```

因此我们是否可以使用两个缓冲区(double buffering)来交替加载呢?

![图片](assets/c427e2135674.png)

这些内容将在下一节介绍TensorCore时展开.

## 附录. Nsight Compute Profiling工具

例如我们需要对basic_gemm kernel 进行5次profile则可以携带参数`-k <kernel-name> -c <num>`,如下所示

```
# ncu --set full -k basic_gemm -c 5  -o  native ./native==PROF== Connected to process 947606 (/data/cuda/gemm/native)==PROF== Profiling "basic_gemm": 0%....50%....100% - 37 passes==PROF== Profiling "basic_gemm": 0%....50%....100% - 37 passes==PROF== Profiling "basic_gemm": 0%....50%....100% - 37 passes==PROF== Profiling "basic_gemm": 0%....50%....100% - 37 passes==PROF== Profiling "basic_gemm": 0%....50%....100% - 37 passesAveragePerformance     42.7145 Gflops==PROF== Disconnected from process 947606==PROF== Report: /data/cuda/gemm/native.ncu-rep
```

执行完后生成的`native.ncu-rep`文件可以下载到本地打开Nsight-Compute进行分析. 其中包含了GPU内存和计算的吞吐的Roofline分析

![图片](assets/72ce15238eec.png)

访问内存的分析

![图片](assets/ea85da9029b2.png)

相关指令的分析

![图片](assets/d39f7fca7014.png)

调度器和Warp Stall的统计

![图片](assets/e605b69602e1.png)

![图片](assets/001ca8d1b8ff.png)

![图片](assets/14525995fd8f.png)

参考资料

[1] 
How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog: https://siboehm.com/articles/22/CUDA-MMM
[2] 
CUDA 矩阵乘法终极优化指南: https://zhuanlan.zhihu.com/p/410278370
[3] 
CUDA SGEMM矩阵乘法优化笔记——从入门到cublas: https://zhuanlan.zhihu.com/p/518857175
[4] 
[施工中] CUDA GEMM 理论性能分析与 kernel 优化: https://zhuanlan.zhihu.com/p/441146275
[5] 
CUDA Matrix Multiplication Optimization: https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
[6] 
深入浅出GPU优化: https://www.zhihu.com/column/c_1437330196193640448
[7] 
SMEM Bandwidth benchmark: https://github.com/Yinghan-Li/YHs_Sample/blob/master/cuda/microbenchmark/smem_bandwidth.cu
[8] 
kernel-7 for bank conflict(linear: https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/7_kernel_resolve_bank_conflicts.cuh