# Tensor-102: GEMV

> 作者: zartbot  
> 日期: 2025年10月9日 00:26  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496294&idx=1&sn=08a1c463d92fc0ebbdc0de6cf57d5b91&chksm=f995e2a4cee26bb2f827d717dad809721833eba44600a38f1508f55b306dec68c3c15a8c99bb#rd

---

### TL;DR

今天继续来做一些CuteDSL和TileLang GEMV算子相关的学习.

算法实现

Thor GFLOPS

Thor MemBW

H20 GFLOPS

H20 MemBW

Cublas

112.70

225.52 GB/s

1325.4

2652.09 GB/s

cutedsl-naive

42.17

84.39 GB/s

95.03

190.16 GB/s

cutedsl-coalesced

117.79

235.55 GB/s

972.54

1946.03 GB/s

cutedsl-block

128.71

257.56 GB/s

1363.69

2728.73 GB/s

tilelang-naive

19.85

39.73 GB/s

50.19

100.43 GB/s

tilelang-split-k

37.82

75.69 GB/s

677.37

1355.41 GB/s

tilelang-vec

83.42

166.93 GB/s

910.22

1821.33 GB/s

tilelang-vec-auto-tune

88.05

176.19 GB/s

1113.13

2227.36 GB/s

## 1. GEMV算法

### 1.1 算法概述

GEMV (GEneral Matrix-Vector multiplication, 通用矩阵-向量乘法)将要实现的计算是:

其中:

 是一个  的矩阵.

 和  是向量.

 和  是标量.

 可以是矩阵  本身 (不转置), 或者是它的转置 .

在CPU中执行运算的代码如下

```
void gemv_cpu(int M, int N,            float alpha, const float *A, const float *x,            float beta, float *y) {    for (int i = 0; i < M; ++i) {        float sum = 0.0f;        for (int j = 0; j < N; ++j) {            sum += A[i * N + j] * x[j];        }        y[i] = alpha * sum + beta * y[i];    }}
```

### 1.2 CuBLAS baseline

GEMV也是一个memory-bound的算子, 在CuBLAS中提供了cublasSgemv函数进行GEMV运算.

```
#include <iostream>#include <vector>#include <cstdlib>#include <cuda_runtime.h>#include <cublas_v2.h>// CUDA 和 CUBLAS API 调用的错误检查宏#define CHECK_CUDA(call) do { \    cudaError_t err = call; \    if (err != cudaSuccess) { \        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \        exit(EXIT_FAILURE); \    } \} while (0)#define CHECK_CUBLAS(call) do { \    cublasStatus_t status = call; \    if (status != CUBLAS_STATUS_SUCCESS) { \        fprintf(stderr, "CUBLAS Error at %s:%d\n", __FILE__, __LINE__); \        exit(EXIT_FAILURE); \    } \} while (0)int main() {    // 1. 定义矩阵和向量的维度    int M = 4096; // 矩阵 A 的行数    int N = 4096; // 矩阵 A 的列数    std::cout << "Testing cublasSgemv with matrix size " << M << "x" << N << std::endl;    // 定义 GEMV 操作的标量 alpha 和 beta    float alpha = 1.0f;    float beta = 0.0f;    // 2. 在主机端分配和初始化数据    std::vector<float> h_A(M * N);    std::vector<float> h_x(N);    std::vector<float> h_y(M, 0.0f); // GPU 计算结果将存放在这里    std::vector<float> h_y_cpu(M, 0.0f); // CPU 计算结果用于验证    // 使用随机数填充矩阵和向量    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;    for (int i = 0; i < N; ++i) h_x[i] = static_cast<float>(rand()) / RAND_MAX;    // 3. 在设备端分配内存    float *d_A, *d_x, *d_y;    CHECK_CUDA(cudaMalloc((void**)&d_A, M * N * sizeof(float)));    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(float)));    CHECK_CUDA(cudaMalloc((void**)&d_y, M * sizeof(float)));    // 4. 创建 CUBLAS 句柄    cublasHandle_t handle;    CHECK_CUBLAS(cublasCreate(&handle));    // 5. 将数据从主机拷贝到设备    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), M * sizeof(float), cudaMemcpyHostToDevice));    // 6. 性能测试    cudaEvent_t start, stop;    CHECK_CUDA(cudaEventCreate(&start));    CHECK_CUDA(cudaEventCreate(&stop));    // long warmup    for (int i = 0; i < 100; ++i) {        CHECK_CUBLAS(cublasSgemv(handle,       // CUBLAS handle                             CUBLAS_OP_N,  // A 是否转置 (No transpose)                             M,            // 矩阵 A 的行数                             N,            // 矩阵 A 的列数                             &alpha,       // 标量 alpha                             d_A,          // 设备上的矩阵 A                             M,            // A 的 leading dimension (对于列主序是行数 M)                             d_x,          // 设备上的向量 x                             1,            // 向量 x 中元素的步长                             &beta,        // 标量 beta                             d_y,          // 设备上的向量 y (输入/输出)                             1));          // 向量 y 中元素的步长    }    CHECK_CUDA(cudaDeviceSynchronize());    // 正式计时    int iterations = 100;    float total_time = 0.0f;    CHECK_CUDA(cudaEventRecord(start));    for (int i = 0; i < iterations; ++i) {        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_x, 1, &beta, d_y, 1));    }    CHECK_CUDA(cudaEventRecord(stop));    CHECK_CUDA(cudaEventSynchronize(stop)); // 等待所有 GPU 操作完成    CHECK_CUDA(cudaEventElapsedTime(&total_time, start, stop));    // 7. 计算并打印性能结果    float avg_time_ms = total_time / iterations;    float avg_time_s = avg_time_ms / 1000.0f;    // GEMV 的浮点操作数 (FLOPs) 大约是 2 * M * N    double flops = 2.0 * M * N;    // GFLOPS = (FLOPs / 10^9) / (time_in_seconds)    double gflops = (flops / 1e9) / avg_time_s ;    // 计算有效内存带宽    // 总共传输  (M * N + M + N) * sizeof(float) 字节    long long num_elements = (long long) M * N + M + N;    double bytes_transferred = num_elements * sizeof(float);    double bandwidth_gb_s = (bytes_transferred / 1e9) / avg_time_s;    std::cout << "------------------------------------------" << std::endl;    std::cout << "Average execution time: " << avg_time_ms << " ms" << std::endl;    std::cout << "Performance(GFLOPS): " << gflops << " GFLOPS" << std::endl;    std::cout << "Effective Memory Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;    std::cout << "------------------------------------------" << std::endl;    // 将 GPU 计算结果拷贝回主机    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost));    // 8. 清理资源    CHECK_CUDA(cudaFree(d_A));    CHECK_CUDA(cudaFree(d_x));    CHECK_CUDA(cudaFree(d_y));    CHECK_CUDA(cudaEventDestroy(start));    CHECK_CUDA(cudaEventDestroy(stop));    CHECK_CUBLAS(cublasDestroy(handle));    return 0;}
```

在Jetson Thor上测试结果如下:

```
zartbot@zartbot-thor:~$ nvcc -arch=sm_110a -lcublas gemv.cuzartbot@zartbot-thor:~$ ./a.outTesting cublasSgemv with matrix size 4096x4096------------------------------------------Average execution time: 0.297716 msPerformance(GFLOPS): 112.706 GFLOPSEffective Memory Bandwidth: 225.523 GB/s------------------------------------------
```

在H20上运行的结果如下

```
Testing cublasSgemv with matrix size 4096x4096------------------------------------------Average execution time: 0.0253165 msPerformance(GFLOPS): 1325.4 GFLOPSEffective Memory Bandwidth: 2652.09 GB/s------------------------------------------
```

## 2. CuteDSL

### 2.1 Naive

计算如下图所示:

![图片](assets/00d522a2c1af.png)

我们来构建一种最简单的运算,  每个线程负责  矩阵一行, 然后乘加后写入到结果.

```
import torchfrom functools import partialimport cutlassimport cutlass.cute as cutefrom cutlass.cute.runtime import from_dlpack@cute.kerneldef naive_gemv_kernel(    gA: cute.Tensor,    gB: cute.Tensor,    gC: cute.Tensor,):    tidx, _, _ = cute.arch.thread_idx()    bidx, _, _ = cute.arch.block_idx()    bdim, _, _ = cute.arch.block_dim()    row = bidx * bdim + tidx    m, n = gA.shape    acc = cutlass.Float64(0.0)    if (row < m ):        for col in range(4096):            acc += gA[row , col] * gB[col]    gC[row] = acc.to(cutlass.Float32)    @cute.jitdef naive_gemv(    mA: cute.Tensor,    mB: cute.Tensor,    mC: cute.Tensor):    num_threads_per_block = 32    m, n = mA.shape    kernel = naive_gemv_kernel(mA, mB, mC)    kernel.launch(grid=(cute.ceil_div(m,num_threads_per_block ), 1, 1),                  block=(num_threads_per_block, 1, 1))M, K = 4096, 4096a = torch.randn(M, K, device="cuda", dtype=torch.float32)b = torch.randn(K, device="cuda", dtype=torch.float32)c = torch.zeros(M, device="cuda", dtype=torch.float32)a_ = from_dlpack(a, assumed_align=16)b_ = from_dlpack(b)c_ = from_dlpack(c)# Compile kernelnaive_gemv_ = cute.compile(naive_gemv, a_, b_, c_)naive_gemv_(a_, b_, c_)# verify correctnesstorch.testing.assert_close(c,torch.mv(a, b),atol=1e-4, rtol=1.3e-6)   
```

同样我们构建benchmark函数进行性能评估

```
def benchmark(callable, *, num_warmups, num_iterations):    start_event = torch.cuda.Event(enable_timing=True)    end_event = torch.cuda.Event(enable_timing=True)    torch.cuda.synchronize()    for _ in range(num_warmups):        callable()    start_event.record(stream=torch.cuda.current_stream())    for _ in range(num_iterations):        callable()    end_event.record(stream=torch.cuda.current_stream())    torch.cuda.synchronize()    elapsed_time = start_event.elapsed_time(end_event)    avg_time = elapsed_time / num_iterations    gflops =  2* a.numel()  / (avg_time  / 1000) / 1e9    print(f"Average execution time: {avg_time:.4f} ms")    print(f"Performance (GFLOPS): {gflops:.4f} GFLOPS")    print(f"Effective Memory Bandwidth: {((a.numel()+b.numel()+c.numel()) * 4) / (avg_time / 1000) / 1e9:.2f} GB/s")benchmark(partial(naive_gemv_, a_, b_, c_), num_warmups=50, num_iterations=100)
```

在Jetson Thor上的结果如下

```
Average execution time: 0.7956 msPerformance (GFLOPS): 42.1748 GFLOPSEffective Memory Bandwidth: 84.39 GB/s
```

在H20上的结果如下:

```
Average execution time: 0.3531 msPerformance (GFLOPS): 95.0356 GFLOPSEffective Memory Bandwidth: 190.16 GB/s
```

### 2.2 访存合并

我们注意到这种实现的性能是非常差的, 主要原因是每个线程访问的内存空间是不连续的, 导致内存加载无法合并. 对于矩阵  是按照Row-major排列的, 是否能够让多个线程连续访问呢?

![图片](assets/380b19339d1c.png)

此时对于每个线程只会访问部分列的值, 如下所示, 因此在计算完成后, 我们还要针对线程计算的部分和进行reduce

![图片](assets/46f166b5752e.png)

CuteDSL Reduction计算在QuACK中有一个文档Getting Memory-bound Kernels to Speed-of-Light[1]

计算warp内的reduce可以使用warp shuffle的方式执行, 例如下图的butterfly warp reduction

![图片](assets/a61949cd2163.png)

具体实现如下:

```
import mathfrom typing import Callable, Optional@cute.jitdef warp_reduce(    val : cute.Numeric,    op: Callable,    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE):    for i in cutlass.range_constexpr(int(math.log2(width))):            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))    return val
```

整个GEMV Kernel如下所示:

```
@cute.kerneldef coalesced_gemv_kernel(    gA: cute.Tensor,    gB: cute.Tensor,    gC: cute.Tensor,):    tidx, _, _ = cute.arch.thread_idx()    bidx, _, _ = cute.arch.block_idx()    bdim, _, _ = cute.arch.block_dim()    m, n = gA.shape    partial_sum = cutlass.Float32(0.0)    for  i in range( n // bdim +1):        col = i * bdim + tidx        if (col < n):            partial_sum += gA[bidx , col] * gB[col]    # reduce partial_sum    partial_sum = warp_reduce(partial_sum,lambda x,y: x +y )    if (tidx == 0):        gC[bidx] = partial_sum    @cute.jitdef coalesced(    mA: cute.Tensor,    mB: cute.Tensor,    mC: cute.Tensor):    num_threads_per_block = cute.arch.WARP_SIZE    m, n = mA.shape    kernel = coalesced_gemv_kernel(mA, mB, mC)    kernel.launch(grid=(m, 1, 1),                  block=(num_threads_per_block, 1, 1))M, K = 4096, 4096a = torch.randn(M, K, device="cuda", dtype=torch.float32)b = torch.randn(K, device="cuda", dtype=torch.float32)c = torch.zeros(M, device="cuda", dtype=torch.float32)a_ = from_dlpack(a, assumed_align=16)b_ = from_dlpack(b)c_ = from_dlpack(c)# Compile kernelcoalesced_ = cute.compile(coalesced, a_, b_, c_)coalesced_(a_, b_, c_)# verify correctnesstorch.testing.assert_close(c,torch.mv(a, b),atol=1e-4, rtol=1.3e-6)   
```

执行Benchmark

```
benchmark(partial(coalesced_gemv_, a_, b_, c_), num_warmups=50, num_iterations=100)
```

在Jetson Thor上执行, 基本上已经打满内存带宽

```
Average execution time: 0.2850 msPerformance (GFLOPS): 117.7191 GFLOPSEffective Memory Bandwidth: 235.55 GB/s
```

在H20上执行, 可以看到它还有提升的空间

```
Average execution time: 0.0345 msPerformance (GFLOPS): 972.5426 GFLOPSEffective Memory Bandwidth: 1946.03 GB/s
```

### 2.3 向量加载

我们注意到每个线程一次LD/ST还是一个标量, 能否通过向量化的方式进行加载? 同样我们采用一个block处理一行, 但是增加每个block内的线程数量`num_thread_per_block`, 则每个线程需要处理的元素个数为

```
num_thread_per_block = 512num_elements_per_thread = A.shape[1] // num_thread_per_block
```

此时我们继续使用TV_Layout的方式. 对于Thread Layout我们采用(1,num_thread_per_block):(num_thread_per_block,1)的方式布置, 对于Value Layout则是(1,num_elements_per_thread):(num_elements_per_thread,1),因此对矩阵  的Tiling如下:

```
    thr_layout = cute.make_layout((1, num_thread_per_block), stride=(num_thread_per_block, 1))    val_layout = cute.make_layout((1, num_elements_per_thread), stride=( num_elements_per_thread, 1))    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
```

例如M=N=4096, num_thread_per_block = 512时, Tiler: (1, 4096), TV Layout: (512,8):(8,1)即一个block有8个warp, 每个warp处理512个值. 在launch Kernel时, 如下所示:

```
    block_gemv_kernel(        gA, mB, mC, tv_layout    ).launch(        grid=[gA.shape[1][0], gA.shape[1][1], 1],        block=[cute.size(tv_layout, mode=[0]), 1, 1],    )
```

在Kernel内, 我们同样采用block tile Layout 和TV_Layout compose的方式来获得需要处理的A-Tile, 并采用tensor.load()使用向量加载

```
    blk_coord = (None, (bidx,bidy))    # logical coord -> address    blkA = gA[blk_coord]  # (TileM, TileN) -> physical address    tidfrgA = cute.composition(blkA, tv_layout)    thr_coord = (tidx, None)    thrA = tidfrgA[thr_coord]  # (V) -> physical address    a_vec = thrA.load()
```

而对于向量B, 我们直接按照每个线程需要处理的元素个数(等于`tidfrgA.shape[1]`)进行partition

```
    b_tiler = cute.make_layout(tidfrgA.shape[1], stride=1)    blkB = cute.zipped_divide(gB, tiler=b_tiler)    b_vec = blkB[(None,tidx)].load()
```

然后我们就可以对线程内的数据进行计算

```
    thread_sum = 0.0    for i in cutlass.range (tidfrgA.shape[1]):        thread_sum += a_vec[i] * b_vec[i]
```

然后每个线程都有一个`thread_sum`, 我们需要在block-level把它们加总求和. 在前一节我们使用warp shuffle可以在warp-level计算reduce-sum. 而这一次我们也使用通用的方法先在warp-level求和. 然后把每个warp的第一个thread(lane_id == 0)写入到smem中, 然后根据warp数量, 将smem中的数据加载到warp-0的不同thread内, 再进行一次warp-reduce, 最后将sum结果写入到GMEM**

```
    lane_id = cute.arch.lane_idx()    warp_id = cute.arch.warp_idx()    warp_num = bdimx // cute.arch.WARP_SIZE        warp_sum = warp_reduce(thread_sum,lambda x,y: x +y)            smem = cutlass.utils.SmemAllocator()    reduce_buffer = smem.allocate_tensor(        element_type= cutlass.Float32,        layout=cute.make_layout(shape=(16), stride=(1)),        byte_alignment=16,    )    if lane_id == 0 :        reduce_buffer[warp_id] = warp_sum      cute.arch.barrier()    sum = 0.0    if (warp_id == 0):               if (tidx < warp_num):            warp_sum = reduce_buffer[tidx]        else:            warp_sum = 0.0        sum = warp_reduce(warp_sum , lambda x,y : x+y)    if (tidx == 0):        gC[bidx] = sum 
```

最后完整的代码如下:

```
@cute.jitdef warp_reduce(    val : cute.Numeric,    op: Callable,    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE):    for i in cutlass.range_constexpr(int(math.log2(width))):            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))    return val@cute.kerneldef block_gemv_kernel(    gA: cute.Tensor,    gB: cute.Tensor,    gC: cute.Tensor,    tv_layout: cute.Layout):    tidx, _, _ = cute.arch.thread_idx()    bidx, bidy, _ = cute.arch.block_idx()    bdimx, bdimy, _ = cute.arch.block_dim()    lane_id = cute.arch.lane_idx()    warp_id = cute.arch.warp_idx()    warp_num = bdimx // cute.arch.WARP_SIZE    blk_coord = (None, (bidx,bidy))    # logical coord -> address    blkA = gA[blk_coord]  # (TileM, TileN) -> physical address    tidfrgA = cute.composition(blkA, tv_layout)    thr_coord = (tidx, None)    thrA = tidfrgA[thr_coord]  # (V) -> physical address    a_vec = thrA.load()    b_tiler = cute.make_layout(tidfrgA.shape[1], stride=1)    blkB = cute.zipped_divide(gB, tiler=b_tiler)    b_vec = blkB[(None,tidx)].load()        thread_sum = 0.0    for i in cutlass.range (tidfrgA.shape[1]):        thread_sum += a_vec[i] * b_vec[i]    warp_sum = warp_reduce(thread_sum,lambda x,y: x +y)            smem = cutlass.utils.SmemAllocator()    reduce_buffer = smem.allocate_tensor(        element_type= cutlass.Float32,        layout=cute.make_layout(shape=(16), stride=(1)),        byte_alignment=16,    )    if lane_id == 0 :        reduce_buffer[warp_id] = warp_sum        cute.arch.barrier()    sum = 0.0    if (warp_id == 0):               if (tidx < warp_num):            warp_sum = reduce_buffer[tidx]        else:            warp_sum = 0.0        sum = warp_reduce(warp_sum , lambda x,y : x+y)    if (tidx == 0):        gC[bidx] = sum         @cute.jitdef block_gemv(    mA: cute.Tensor,    mB: cute.Tensor,    mC: cute.Tensor,):        num_thread_per_block = 512    num_elements_per_thread = mA.shape[1] // num_thread_per_block        thr_layout = cute.make_layout((1, num_thread_per_block), stride=(num_thread_per_block, 1))    val_layout = cute.make_layout((1, num_elements_per_thread), stride=( num_elements_per_thread, 1))    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)    print(f"Tiler: {tiler_mn}")    print(f"TV Layout: {tv_layout}")    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))    print(f"Tiled Input Tensors:")    print(f"  gA: {gA.type}, shape {gA.shape[1][0]}")    block_gemv_kernel(        gA, mB, mC, tv_layout    ).launch(        grid=[gA.shape[1][0], gA.shape[1][1], 1],        block=[cute.size(tv_layout, mode=[0]), 1, 1],    )    M, N = 4096, 4096a = torch.randn(M, N, device="cuda", dtype=torch.float32)b = torch.randn(N, device="cuda", dtype=torch.float32)c = torch.zeros(M, device="cuda", dtype=torch.float32)a_ = from_dlpack(a, assumed_align=16)b_ = from_dlpack(b, assumed_align=16)c_ = from_dlpack(c)block_gemv_ = cute.compile(block_gemv, a_, b_, c_)block_gemv_(a_, b_, c_)# verify correctnesstorch.testing.assert_close(c,torch.mv(a, b),atol=1e-4, rtol=1.3e-6) 
```

Jetson Thor性能测试结果如下, 已经打满内存带宽

```
benchmark(partial(block_gemv_, a_, b_, c_), num_warmups=50, num_iterations=100)Average execution time: 0.2607 msPerformance (GFLOPS): 128.7154 GFLOPSEffective Memory Bandwidth: 257.56 GB/s
```

H20 性能测试结果如下, 性能已经超过Cublas

```
Average execution time: 0.0246 msPerformance (GFLOPS): 1363.6997 GFLOPSEffective Memory Bandwidth: 2728.73 GB/s
```

## 3. TileLang

Tilelang在官方有一个文档介绍《General Matrix-Vector Multiplication (GEMV)》[2]

### 3.1 Navie

由于TileLang本身具有很好的基于Tile的抽象, 因此我们可以很简单的将矩阵  拆分成`(BLOCK_M, BLOCK_K)`的Tile, 也拆分为`(BLOCK_K)`的块, 然后顺序的进行点乘并求和.

```
import torchimport tilelangimport tilelang.language as Tdef naive_gemv(    M: int,    K: int,    BLOCK_M: int,    BLOCK_K: int,    dtype: str = "float16",    accum_dtype: str = "float",):    @T.prim_func    def main(            A: T.Buffer((M, K), dtype),            B: T.Buffer((K,), dtype),            C: T.Buffer((M,), dtype),    ):        with T.Kernel(T.ceildiv(M, BLOCK_M)) as bm:            tm = T.get_thread_binding(0)  # tm = threadIdx.x            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)            B_shared = T.alloc_shared((BLOCK_K,), dtype)            C_reg = T.alloc_local((1,), accum_dtype)            T.clear(C_reg)            for bk in T.serial(T.ceildiv(K, BLOCK_K)):                for tk in T.serial(BLOCK_K):                    A_shared[tm, tk] = A[bm * BLOCK_M + tm, bk * BLOCK_K + tk]                    B_shared[tk] = B[bk * BLOCK_K + tk]                                    for tk in T.serial(BLOCK_K):                    C_reg[0] +=  A_shared[tm,tk].astype(accum_dtype) *B_shared[tk].astype(accum_dtype)             C[bm * BLOCK_M + tm] = C_reg[0]    return main
```

相对于CuteDSL, Tilelang对于张量分块的数据流描述更加简单直观一些.

然后我们继续使用如下代码进行验证和性能测量

```
M,K = 4096,4096BLOCK_M,BLOCK_K = 128,128func = naive_gemv(M, K, BLOCK_M,BLOCK_K, "float32","float32")jit_kernel = tilelang.compile(func, out_idx=[-1], target="cuda")a = torch.randn(M, K, device="cuda", dtype=torch.float32)b = torch.randn(K, device="cuda", dtype=torch.float32)c = torch.zeros(M, device="cuda", dtype=torch.float32)c = jit_kernel(a,b)# verify correctnesstorch.testing.assert_close(c, torch.mv(a ,b),atol=1e-3, rtol=1.3e-6) profiler = jit_kernel.get_profiler()avg_time = profiler.do_bench()gflops =  2 * a.numel()  / (avg_time  / 1000) / 1e9print(f"Average execution time: {avg_time:.4f} ms")print(f"Performance (GFLOPS): {gflops:.4f} GFLOPS")print(f"Effective Memory Bandwidth: {( ( a.numel() +b.numel() +c.numel()) * 4) / (avg_time / 1000) / 1e9:.2f} GB/s")
```

在Jetson Thor上的性能如下:

```
Average execution time: 1.6899 msPerformance (GFLOPS): 19.8556 GFLOPSEffective Memory Bandwidth: 39.73 GB/s
```

在H20上的性能如下:

```
Average execution time: 0.6685 msPerformance (GFLOPS): 50.1927 GFLOPSEffective Memory Bandwidth: 100.43 GB/s
```

由于naive的代码性能只有峰值的1/20不到, 整体的并行度并不高.

### 3.2 增加并行(Split- K)

我们在K维度进行分块 ,增加并行度.

```
def naive_split_gemv(    M: int,    K: int,    BLOCK_M: int,    BLOCK_K: int,    dtype: str = "float16",    accum_dtype: str = "float",):    @T.prim_func    def main(            A: T.Buffer((M, K), dtype),            B: T.Buffer((K,), dtype),            C: T.Buffer((M,), dtype),    ):        #add K dim,threads=[BLOCK_M,BLOCK_K]        with T.Kernel(T.ceildiv(M, BLOCK_M),threads=(BLOCK_M,BLOCK_K)) as bm:            tm = T.get_thread_binding(0)  # tm = threadIdx.x            tk = T.get_thread_binding(1)  # tk = threadIdx.y            A_local = T.alloc_local((1,), dtype) #thread local buffer            B_local = T.alloc_local((1,), dtype)            C_accum =  T.alloc_local((1,), accum_dtype)            C_shared = T.alloc_shared((BLOCK_M,), accum_dtype) #alloc SMEM            if tk == 0:                C_shared[tm] = 0.0            T.clear(C_accum)            for bk in T.serial(T.ceildiv(K,BLOCK_K)):                A_local[0] = A[bm * BLOCK_M + tm, bk * BLOCK_K + tk]                B_local[0] = B[bk * BLOCK_K + tk]                C_accum[0] +=  A_local[0].astype(accum_dtype) * B_local[0].astype(accum_dtype)            #最后通过atomic累加到smem            T.atomic_add(C_shared[tm], C_accum[0])            C[bm * BLOCK_M + tm] = C_shared[tm]    return main
```

测试并验证:

```
M,K = 4096,4096BLOCK_M,BLOCK_K = 8,32func = naive_split_gemv(M, K, BLOCK_M, BLOCK_K, "float32","float32")jit_kernel = tilelang.compile(func, out_idx=[-1], target="cuda")a = torch.randn(M, K, device="cuda", dtype=torch.float32)b = torch.randn(K, device="cuda", dtype=torch.float32)c = torch.zeros(M, device="cuda", dtype=torch.float32)c = jit_kernel(a,b)# verify correctnesstorch.testing.assert_close(c, torch.mv(a ,b),atol=1e-3, rtol=1.3e-6) 
```

最后进行性能测试:

```
profiler = jit_kernel.get_profiler()avg_time = profiler.do_bench()gflops =  2 * a.numel()  / (avg_time  / 1000) / 1e9print(f"Average execution time: {avg_time:.4f} ms")print(f"Performance (GFLOPS): {gflops:.4f} GFLOPS")print(f"Effective Memory Bandwidth: {( ( a.numel() +b.numel() +c.numel()) * 4) / (avg_time / 1000) / 1e9:.2f} GB/s")
```

在Jetson Thor上的测试结果

```
Average execution time: 0.8871 msPerformance (GFLOPS): 37.8253 GFLOPSEffective Memory Bandwidth: 75.69 GB/s
```

在H20上的测试结果:

```
Average execution time: 0.0495 msPerformance (GFLOPS): 677.3747 GFLOPSEffective Memory Bandwidth: 1355.41 GB/s
```

### 3.3 向量化加载及TVM_thread_allreduce

在TileLang中可以使用`T.vectorized`来使用向量化加载. 并且reduce计算中的atomic也非常影响性能, 在cuteDSL使用了warp suffle的方式进行reduce处理, 而在TileLang中可以使用tvm_thread_allreduce替代.

```
from tilelang import tvm as tvmfrom tvm import DataTypedef split_gemv_vec(    M: int,    K: int,    BLOCK_M: int,    reduce_threads: int,    dtype: str = "float16",    accum_dtype: str = "float",):    MAX_TRANSACTION_SIZE_IN_BITS = 128    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits    BLOCK_K = reduce_threads * TILE_K     @T.prim_func    def main(            A: T.Buffer((M, K), dtype),            B: T.Buffer((K,), dtype),            C: T.Buffer((M,), dtype),    ):        #add K dim,threads=[BLOCK_M,reduce_threads]        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=(BLOCK_M, reduce_threads)) as bm:            tm = T.get_thread_binding(0)  # tm = threadIdx.x            tk = T.get_thread_binding(1)  # tk = threadIdx.y            A_local = T.alloc_local((TILE_K,), dtype) #thread local buffer            B_local = T.alloc_local((TILE_K,), dtype)            C_accum =  T.alloc_local((1,), accum_dtype)            C_shared = T.alloc_shared((BLOCK_M,), accum_dtype) #alloc SMEM                        T.clear(C_accum)            for bk in T.serial(T.ceildiv(K,BLOCK_K)):                for k in T.vectorized(TILE_K):                    A_local[k] = A[bm * BLOCK_M + tm, bk * BLOCK_K + tk * TILE_K + k]                    B_local[k] = B[bk * BLOCK_K + tk * TILE_K + k]                                    for k in T.serial(TILE_K):                     C_accum[0] +=  A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)                        #使用TVM thread allreduce            C_reduced = T.alloc_local((1,), accum_dtype)            with T.attr(                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),                    "reduce_scope",                    T.reinterpret(T.uint64(0), dtype="handle"),            ):                T.evaluate(                    T.tvm_thread_allreduce(                        T.uint32(1),                        C_accum[0],                        True,                        C_reduced[0],                        tk,                        dtype="handle",                    ))            C[bm * BLOCK_M + tm] = C_reduced[0]    return mainM,K = 4096,4096BLOCK_M = 4reduce_threads = 32func = split_gemv_vec(M, K, BLOCK_M, reduce_threads, "float32","float32")jit_kernel = tilelang.compile(func, out_idx=[-1], target="cuda")a = torch.randn(M, K, device="cuda", dtype=torch.float32)b = torch.randn(K, device="cuda", dtype=torch.float32)c = torch.zeros(M, device="cuda", dtype=torch.float32)c = jit_kernel(a,b)# verify correctnesstorch.testing.assert_close(c, torch.mv(a ,b),atol=1e-3, rtol=1.3e-6) 
```

性能测试如下, 在Jetson Thor上性能

```
Average execution time: 0.4022 msPerformance (GFLOPS): 83.4248 GFLOPSEffective Memory Bandwidth: 166.93 GB/s
```

在H20上性能:

```
Average execution time: 0.0369 msPerformance (GFLOPS): 910.2222 GFLOPSEffective Memory Bandwidth: 1821.33 GB/s
```

### 3.4 Auto-Tune

在这个Kernel中有`BLOCK_M`, `reduce_threads`两个超参数, 进行AutoTune**

```
import itertoolsdef ref_program(A, B):    return A @ B.Tdef get_config():    BLOCK_M = [1, 2, 4, 8,16, 32, 64, 128]    reduce_threads = [4, 8, 16, 32, 64, 128, 256]    _configs = list(itertools.product(        BLOCK_M,        reduce_threads,    ))    configs = []    for c in _configs :        if (c[0] * c[1] <=1024):            configs.append({                "BLOCK_M": c[0],                "reduce_threads": c[1],            })    return configsget_config()@tilelang.autotune(    configs= get_config(),    warmup= 50,    rep = 100,)@tilelang.jit(    out_idx=[-1],    target="auto",)def kernel(    BLOCK_M=None,    reduce_threads = None,):    M = 4096    K = 4096    dtype = "float32"    accum_dtype = "float32"    MAX_TRANSACTION_SIZE_IN_BITS = 128    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits    BLOCK_K = reduce_threads * TILE_K     @T.prim_func    def main(            A: T.Buffer((M, K), dtype),            B: T.Buffer((K,), dtype),            C: T.Buffer((M,), dtype),    ):        #add K dim,threads=[BLOCK_M,reduce_threads]        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=(BLOCK_M, reduce_threads)) as bm:            tm = T.get_thread_binding(0)  # tm = threadIdx.x            tk = T.get_thread_binding(1)  # tk = threadIdx.y            A_local = T.alloc_local((TILE_K,), dtype) #thread local buffer            B_local = T.alloc_local((TILE_K,), dtype)            C_accum =  T.alloc_local((1,), accum_dtype)            C_shared = T.alloc_shared((BLOCK_M,), accum_dtype) #alloc SMEM                        T.clear(C_accum)            for bk in T.serial(T.ceildiv(K,BLOCK_K)):                for k in T.vectorized(TILE_K):                    A_local[k] = A[bm * BLOCK_M + tm, bk * BLOCK_K + tk * TILE_K + k]                    B_local[k] = B[bk * BLOCK_K + tk * TILE_K + k]                                    for k in T.serial(TILE_K):                     C_accum[0] +=  A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)                        #使用TVM thread allreduce            C_reduced = T.alloc_local((1,), accum_dtype)            with T.attr(                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),                    "reduce_scope",                    T.reinterpret(T.uint64(0), dtype="handle"),            ):                T.evaluate(                    T.tvm_thread_allreduce(                        T.uint32(1),                        C_accum[0],                        True,                        C_reduced[0],                        tk,                        dtype="handle",                    ))            C[bm * BLOCK_M + tm] = C_reduced[0]    return mainauto_kernel = kernel()best_config = auto_kernel.configbest_latency = auto_kernel.latencyprint(f"Best Config: {best_config}")gflops =  2 * a.numel()  / (best_latency  / 1000) / 1e9print(f"Average execution time: {best_latency:.4f} ms")print(f"Performance (GFLOPS): {gflops:.4f} GFLOPS")print(f"Effective Memory Bandwidth: {( ( a.numel() +b.numel() +c.numel()) * 4) / (best_latency / 1000) / 1e9:.2f} GB/s")
```

在Jetson Thor上最佳性能为:

```
Best Config: {'BLOCK_M': 2, 'reduce_threads': 256}Average execution time: 0.3811 msPerformance (GFLOPS): 88.0519 GFLOPSEffective Memory Bandwidth: 176.19 GB/s
```

在H20上最佳性能:

```
Best Config: {'BLOCK_M': 1, 'reduce_threads': 64}Average execution time: 0.0301 msPerformance (GFLOPS): 1113.1380 GFLOPSEffective Memory Bandwidth: 2227.36 GB/s
```

参考资料

[1] 
Getting Memory-bound Kernels to Speed-of-Light: *https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md*
[2] 
TileLang GEMV: *https://tilelang.com/deeplearning_operators/gemv.html*