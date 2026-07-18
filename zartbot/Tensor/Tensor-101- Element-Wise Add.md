# Tensor-101: Element-Wise Add

> 作者: zartbot  
> 日期: 2025年10月5日 15:42  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496268&idx=1&sn=e00708b7d106ff17fadb0d085904280f&chksm=f995e28ecee26b98289dc196997d153508434a6f64a9c89ca083fd3709d0949091ead5940480#rd

---

### TL:DR

接下来混着CuteDSL和Tilelang一起来折腾吧, 先从一些最basic的算子开始. 大概每一篇的结构就是先谈谈算法本身, 然后CuteDSL和Tilelang的实现, 还有就是一些性能调试相关的内容.  大概从elementwise Add, 到GEMV, 再到GEMM, 然后逐渐再到FlashAttn, GroupGEMM这些..

本文介绍了elementwise add的计算, 并通过cuteDSL和tilelang实现了几种kernel和baseline cublas进行对比, 性能结果如下:

算法实现

Thor GFLOPS

Thor MemBW

H20 GFLOPS

H20 MemBW

Cublas

19.61

235.383 GB/s

262.25

3147.11 GB/s

cutedsl-naive

20.07

240.90 GB/s

179.52

2154.25 GB/s

cutedsl-vec-ld/st

21.48

257.77 GB/s

293.23

3518.77 GB/s

cutedsl-tv-layout

20.92

251.12 GB/s

279.28

3351.40 GB/s

tilelang-naive

16.81

201.80 GB/s

268.17

3218.14 GB/s

tilelang-autotune

20.79

249.49 GB/s

271.65

3259.82 GB/s

注: Thor由于没有调整到最高功耗模式运行, 前面的测试并不准... 即便设置nvidia-smi -pm 1后也有很明显的性能抖动.  复测中针对Thor采用了更长的warmup次数, 上表已经更新...

在cuteDSL中还介绍了TV-Layout相关的知识, 相对于cuteDSL, tilelang更加便捷, autotune功能也非常有用.

## 1. ElementWise Add

### 1.1 算法概述

矩阵的ElementWise Addition计算如下:

**输入**: 输入是两个矩阵,我们称为  和 , 约束是  必须要有相同的shape.

**计算方法**: 计算, 且 都是  的矩阵, 对于任意的  和   :

当然还有另一种计算, 存在标量 ,  ,  为:

如果我们使用CPU计算的代码如下所示:

```
void cpu_geam(int m, int n, float alpha, const float* A, float beta, const float* B, float* C) {    for (int j = 0; j < n; ++j) {        for (int i = 0; i < m; ++i) {            int index = j * m + i;            C[index] = alpha * A[index] + beta * B[index];        }    }}
```

### 1.2 CuBLAS baseline

它在计算上非常简单，它属于Memory-Bound操作. CUBLAS 提供了 cublas<t> geam (GEneral matrix-matrix Addition/Multiplication) 函数，可以非常方便地实现这个功能

```
#include <iostream>#include <vector>#include <cstdlib>#include <cuda_runtime.h>#include <cublas_v2.h>// CUDA 和 CUBLAS API 调用的错误检查宏#define CHECK_CUDA(call) do { \    cudaError_t err = call; \    if (err != cudaSuccess) { \        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \        exit(EXIT_FAILURE); \    } \} while (0)#define CHECK_CUBLAS(call) do { \    cublasStatus_t status = call; \    if (status != CUBLAS_STATUS_SUCCESS) { \        fprintf(stderr, "CUBLAS Error at %s:%d\n", __FILE__, __LINE__); \        exit(EXIT_FAILURE); \    } \} while (0)  int main() {    // 1. 定义矩阵维度    int M = 4096; // 矩阵的行数    int N = 4096; // 矩阵的列数    long long num_elements = (long long)M * N;    std::cout << "Testing cublasSgeam (element-wise add) with matrix size " << M << "x" << N << std::endl;    // 定义 GEAM 操作的标量 alpha 和 beta    // C = alpha * A + beta * B. 对于简单的加法, alpha=1.0, beta=1.0    float alpha = 1.0f;    float beta = 1.0f;    // 2. 在主机端分配和初始化数据    std::vector<float> h_A(num_elements);    std::vector<float> h_B(num_elements);    std::vector<float> h_C_gpu(num_elements); // GPU 计算结果    std::vector<float> h_C_cpu(num_elements); // CPU 计算结果用于验证    // 使用随机数填充矩阵    for (long long i = 0; i < num_elements; i++ ) {        h_A[i] = static_cast<float>(rand()) / RAND_MAX;        h_B[i] = static_cast<float>(rand()) / RAND_MAX;    }    // 3. 在设备端分配内存    float *d_A, *d_B, *d_C;    CHECK_CUDA(cudaMalloc((void**)&d_A, num_elements * sizeof(float)));    CHECK_CUDA(cudaMalloc((void**)&d_B, num_elements * sizeof(float)));    CHECK_CUDA(cudaMalloc((void**)&d_C, num_elements * sizeof(float)));    // 4. 创建 CUBLAS 句柄    cublasHandle_t handle;    CHECK_CUBLAS(cublasCreate(&handle));    // 5. 将数据从主机拷贝到设备    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));    // 6. 性能测试    cudaEvent_t start, stop;    CHECK_CUDA(cudaEventCreate(&start));    CHECK_CUDA(cudaEventCreate(&stop));    // Warm-up    CHECK_CUBLAS(cublasSgeam(handle,                              CUBLAS_OP_N, CUBLAS_OP_N, // 不转置 A 和 B                             M, N,                              &alpha,                              d_A, M,                  // A 和它的 leading dimension                             &beta,                             d_B, M,                  // B 和它的 leading dimension                             d_C, M));                // C 和它的 leading dimension    CHECK_CUDA(cudaDeviceSynchronize());    // 正式计时    int iterations = 100;    float total_time = 0.0f;    CHECK_CUDA(cudaEventRecord(start));    for (int i = 0; i < iterations; ++i) {        CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &alpha, d_A, M, &beta, d_B, M, d_C, M));    }    CHECK_CUDA(cudaEventRecord(stop));    CHECK_CUDA(cudaEventSynchronize(stop)); // 等待所有 GPU 操作完成    CHECK_CUDA(cudaEventElapsedTime(&total_time, start, stop));    // 7. 计算并打印性能结果    float avg_time_ms = total_time / iterations;    float avg_time_s = avg_time_ms / 1000.0f;    // 计算 GFLOPS    // 每个元素有 1 次浮点加法    double flops = (double)num_elements;    double gflops = (flops / 1e9) / avg_time_s;    // 计算有效内存带宽    // 每次操作需要: 读取 A (M*N floats), 读取 B (M*N floats), 写入 C (M*N floats)    // 总共传输 3 * M * N * sizeof(float) 字节    double bytes_transferred = 3.0 * num_elements * sizeof(float);    double bandwidth_gb_s = (bytes_transferred / 1e9) / avg_time_s;    std::cout << "--------------------------------------------------------" << std::endl;    std::cout << "Average execution time: " << avg_time_ms << " ms" << std::endl;    std::cout << "Performance (GFLOPS): " << gflops << " GFLOPS" << std::endl;    std::cout << "Effective Memory Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;    std::cout << "--------------------------------------------------------" << std::endl;    // 将 GPU 计算结果拷贝回主机    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, num_elements * sizeof(float), cudaMemcpyDeviceToHost));        // 9. 清理资源    CHECK_CUDA(cudaFree(d_A));    CHECK_CUDA(cudaFree(d_B));    CHECK_CUDA(cudaFree(d_C));    CHECK_CUDA(cudaEventDestroy(start));    CHECK_CUDA(cudaEventDestroy(stop));    CHECK_CUBLAS(cublasDestroy(handle));    return 0;}
```

在Jetson Thor上测试结果如下:

```
zartbot@zartbot-thor:~$ nvcc -arch=sm_110 -lcublas eleadd.cuzartbot@zartbot-thor:~$ ./a.outTesting cublasSgeam (element-wise add) with matrix size 4096x4096--------------------------------------------------------Average execution time: 0.855316 msPerformance (GFLOPS): 19.6152 GFLOPSEffective Memory Bandwidth: 235.383 GB/s--------------------------------------------------------
```

而在H20上测试结果如下:

```
Testing cublasSgeam (element-wise add) with matrix size 4096x4096--------------------------------------------------------Average execution time: 0.0639718 msPerformance (GFLOPS): 262.259 GFLOPSEffective Memory Bandwidth: 3147.11 GB/s--------------------------------------------------------
```

## 2. Cute-DSL

Cute-DSL在git上有一个notebook[1]

### 2.1 Navie elementwise add

首先import相关的库

```
import osos.environ['CUTE_DSL_ARCH'] = 'sm_101a' #Thor在cuda 13.0改名为SM110, 但是cutedsl-4.2还是基于12.9, 因此需要设置一个环境变量import torchfrom functools import partialimport cutlassimport cutlass.cute as cutefrom cutlass.cute.runtime import from_dlpack
```

Cute-DSL Kernel如下所示, 总体来看和写Cuda code也没啥区别

```
@cute.kerneldef naive_elementwise_add_kernel(    gA: cute.Tensor,    gB: cute.Tensor,    gC: cute.Tensor,):    tidx, _, _ = cute.arch.thread_idx()    bidx, _, _ = cute.arch.block_idx()     bdim, _, _ = cute.arch.block_dim()    thread_idx = bidx * bdim + tidx    # Map thread index to logical index of input tensor    m, n = gA.shape    ni = thread_idx % n    mi = thread_idx // n    # Map logical index to physical address via tensor layout    a_val = gA[mi, ni]    b_val = gB[mi, ni]    # Perform element-wise addition    gC[mi, ni] = a_val + b_val@cute.jitdef naive_elementwise_add(    mA: cute.Tensor,    mB: cute.Tensor,    mC: cute.Tensor):    num_threads_per_block = 256    m, n = mA.shape    kernel = naive_elementwise_add_kernel(mA, mB, mC)    kernel.launch(grid=((m * n) // num_threads_per_block, 1, 1),                  block=(num_threads_per_block, 1, 1))
```

然后是一些测试和验证:

```
M, N = 4096, 4096a = torch.randn(M, N, device="cuda", dtype=torch.float32)b = torch.randn(M, N, device="cuda", dtype=torch.float32)c = torch.zeros(M, N, device="cuda", dtype=torch.float32)a_ = from_dlpack(a, assumed_align=16)b_ = from_dlpack(b, assumed_align=16)c_ = from_dlpack(c, assumed_align=16)# Compile kernelnaive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)naive_elementwise_add_(a_, b_, c_)# verify correctnesstorch.testing.assert_close(c, a + b)
```

下面还有一段benchmark的代码:

```
def benchmark(callable, *, num_warmups, num_iterations):    start_event = torch.cuda.Event(enable_timing=True)    end_event = torch.cuda.Event(enable_timing=True)    torch.cuda.synchronize()    for _ in range(num_warmups):        callable()    start_event.record(stream=torch.cuda.current_stream())    for _ in range(num_iterations):        callable()    end_event.record(stream=torch.cuda.current_stream())    torch.cuda.synchronize()    elapsed_time = start_event.elapsed_time(end_event)    avg_time = elapsed_time / num_iterations    gflops =  a.numel()  / (avg_time  / 1000) / 1e9    print(f"Average execution time: {avg_time:.4f} ms")    print(f"Performance (GFLOPS): {gflops:.4f} GFLOPS")    print(f"Effective Memory Bandwidth: {(3 * a.numel() * 4) / (avg_time / 1000) / 1e9:.2f} GB/s")    benchmark(partial(naive_elementwise_add_, a_, b_, c_), num_warmups=5, num_iterations=100)
```

在Thor上执行

```
Average execution time: 0.8357 msPerformance (GFLOPS): 20.0751 GFLOPSEffective Memory Bandwidth: 240.90 GB/s
```

在H20上执行

```
Average execution time: 0.0935 msPerformance (GFLOPS): 179.5212 GFLOPSEffective Memory Bandwidth: 2154.25 GB/s
```

可以看到和Cublas的结果相比, 内存带宽并没有打满.

### 2.2 向量化的LD/ST

Cute-DSL的Note又一次讲起了Little's Law

其中:

是平均到达系统的服务对象个数

是服务对象的平均到达速率(Bandwidth)

是服务对象在系统中的平均耗时(Latency)

对于Memory-Bound的算子而言:  代表平均inflight的LD/ST操作数量,  代表带宽, 即在Memory和COmpute Unit之间的数据传输速率, 代表延迟, 即内存请求的RTT.

![图片](assets/1c73bfb91085.png)

从前面2.1来看, Memory BW并没有打到极限, 那么简单的方法就是通过软件增加LD/ST操作的数量. 在较新的GPU架构中增加了128bits的向量化LD/ST指令(`ld.global.v4.f32` , `st.global.v4.f32`)利用它们可以增加inflight的数量.

但此时, 每个线程通过向量化的LD/ST需要读取4个内存地址连续的元素, 即构成一个(1,4):(0:1)的layout. 即我们需要把 的矩阵按照(1,4)进行分块, 这里稍微展开讲一下矩阵Layout的除法.

在[《CuTe Layout代数-1: Overview》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496154&idx=1&sn=474a5450c46b86169095d84dd3cfd7dc&scene=21#wechat_redirect)中也有一些介绍, 对于矩阵Partitioning/Tiling的代数运算被称为Logical_divide, 例如我们有一个M=16, N=32的张量, 我们希望把它拆分成tileM=2, tileN=4的块. 那么对于行M将拆分成8块, 对于列32将被拆分成32/4=8块. 在Cutelass中定义了logical_divide, zipped_divide, flat_divide, tiled_divide等多种除法, 实际上大家不要慌, 后面三种都是logical_divide的基础上对于生成的nested tuple中的mode进行重排产生的. 下面有一个测试示例

```
@cute.jitdef layout_test(    mA: cute.Tensor):    tiler = (2,4)    lA = cute.logical_divide(mA, tiler=tiler)    zA = cute.zipped_divide(mA, tiler=tiler)    fA = cute.flat_divide(mA, tiler=tiler)    tA = cute.tiled_divide(mA, tiler=tiler)    print(f"[DSL INFO] Tiled Tensors:")    print(f"[DSL INFO] Tiler: {tiler}")    print(f"[DSL INFO] logical_divide  lA = {lA}")        print(100*'=')    print(f"[DSL INFO] zipped_divide  zA = {zA}")    print(f"[DSL INFO] flatten_divide  fA = {fA}")    print(f"[DSL INFO] tiled_divide  tA = {tA}")M,N= 16,32a = torch.randn(M, N, device="cuda", dtype=torch.float32)print(f"Tensor shape  {a.shape} Stride: {a.stride()}")a_ = from_dlpack(a, assumed_align=16)layout_test(a_)
```

具体输出如下

```
Tensor shape  torch.Size([16, 32]) Stride: (32, 1)[DSL INFO] Tiled Tensors:[DSL INFO] Tiler: (2, 4)[DSL INFO] logical_divide  lA = tensor<ptr<f32, gmem, align<16>> o ((2,8),(4,8)):((32,64),(1,4))>====================================================================================================[DSL INFO] zipped_divide  zA = tensor<ptr<f32, gmem, align<16>> o ((2,4),(8,8)):((32,1),(64,4))>[DSL INFO] flatten_divide  fA = tensor<ptr<f32, gmem, align<16>> o (2,4,8,8):(32,1,64,4)>[DSL INFO] tiled_divide  tA = tensor<ptr<f32, gmem, align<16>> o ((2,4),8,8):((32,1),64,4)>
```

对于logical_divide, 它直接在原有的行和列上构造nested tuple. 然后计算相应的stride.  zipped则是把它进行了一个重排, 把tile shape tuple放在前面, 后一段代表以Tile为单位的shape.  flatten则是直接把整个nested tuple打平成一维. tiled_divide 则是只把tile shape tuple构建成nested.

![图片](assets/64571e41e259.png)

对于需要向量化LD/ST, tiler为(1,4). 对于一个M,N=(4096,4096)的矩阵, zipped_div结果如下

![图片](assets/69245b13dfb4.png)

因此我们构建的代码如下, 此时需要注意, 对于每个thread需要向量化读取的slice, 需要使用mi,ni 索引.

```
@cute.kerneldef vectorized_elementwise_add_kernel(    gA: cute.Tensor,    gB: cute.Tensor,    gC: cute.Tensor,):    tidx, _, _ = cute.arch.thread_idx()    bidx, _, _ = cute.arch.block_idx()    bdim, _, _ = cute.arch.block_dim()    thread_idx = bidx * bdim + tidx    # Map thread index to logical index of input tensor    m, n = gA.shape[1]       # thread-domain    print(f"THread domain m={m} , n={n}") #结果为上图中紫色的(4096,1024)    ni = thread_idx % n    mi = thread_idx // n    # 使用.load()方式构建向量化的加载    a_val = gA[(None, (mi, ni))].load()    b_val = gB[(None, (mi, ni))].load()    print(f"[DSL INFO] sliced gA = {gA[(None, (mi, ni))]}")    print(f"[DSL INFO] sliced gB = {gB[(None, (mi, ni))]}")     # Perform element-wise addition    gC[(None, (mi, ni))] = a_val + b_val@cute.jitdef vectorized_elementwise_add(    mA: cute.Tensor,    mB: cute.Tensor,    mC: cute.Tensor):    threads_per_block = 256    #这里对原始矩阵按照(1,4)做zipped div    gA = cute.zipped_divide(mA, (1, 4))    gB = cute.zipped_divide(mB, (1, 4))    gC = cute.zipped_divide(mC, (1, 4))    print(f"[DSL INFO] Tiled Tensors:")    print(f"[DSL INFO]   gA = {gA}")    print(f"[DSL INFO]   gB = {gB}")    print(f"[DSL INFO]   gC = {gC}")    vectorized_elementwise_add_kernel(gA, gB, gC).launch(        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),        block=(threads_per_block, 1, 1),    )
```

然后执行算法验证和性能测试

```
M,N= 4096,4096a = torch.randn(M, N, device="cuda", dtype=torch.float32)b = torch.randn(M, N, device="cuda", dtype=torch.float32)c = torch.zeros(M, N, device="cuda", dtype=torch.float32)a_ = from_dlpack(a, assumed_align=16)b_ = from_dlpack(b, assumed_align=16)c_ = from_dlpack(c, assumed_align=16)compiled_func = cute.compile(vectorized_elementwise_add, a_, b_, c_)compiled_func(a_, b_, c_)# verify correctnesstorch.testing.assert_close(c, a + b)benchmark(partial(compiled_func, a_, b_, c_), num_warmups=5, num_iterations=100)
```

在Jetson Thor上的测试结果, 相对于Naive的实现, 由于整个芯片内存带宽受限, 整体性能并没有提升.

```
[DSL INFO] Tiled Tensors:[DSL INFO]   gA = tensor<ptr<f32, gmem, align<16>> o ((1,4),(4096,1024)):((0,1),(4096,4))>[DSL INFO]   gB = tensor<ptr<f32, gmem, align<16>> o ((1,4),(4096,1024)):((0,1),(4096,4))>[DSL INFO]   gC = tensor<ptr<f32, gmem, align<16>> o ((1,4),(4096,1024)):((0,1),(4096,4))>THread domain m=4096 , n=1024[DSL INFO] sliced gA = tensor<ptr<f32, gmem, align<16>> o ((1,4)):((0,1))>[DSL INFO] sliced gB = tensor<ptr<f32, gmem, align<16>> o ((1,4)):((0,1))>Average execution time: 0.7810 msPerformance (GFLOPS): 21.4811 GFLOPSEffective Memory Bandwidth: 257.77 GB/s
```

而在H20上Vector LD/ST性能有了显著的提升, 物理带宽极限在4TB/s, 已经可以达到3.5TB/s了.

```
Average execution time: 0.0929 msPerformance (GFLOPS): 180.5381 GFLOPSEffective Memory Bandwidth: 2166.46 GB/s[DSL INFO] Tiled Tensors:[DSL INFO]   gA = tensor<ptr<f32, gmem, align<16>> o ((1,4),(4096,1024)):((0,1),(4096,4))>[DSL INFO]   gB = tensor<ptr<f32, gmem, align<16>> o ((1,4),(4096,1024)):((0,1),(4096,4))>[DSL INFO]   gC = tensor<ptr<f32, gmem, align<16>> o ((1,4),(4096,1024)):((0,1),(4096,4))>THread domain m=4096 , n=1024[DSL INFO] sliced gA = tensor<ptr<f32, gmem, align<16>> o ((1,4)):((0,1))>[DSL INFO] sliced gB = tensor<ptr<f32, gmem, align<16>> o ((1,4)):((0,1))>Average execution time: 0.0572 msPerformance (GFLOPS): 293.2309 GFLOPSEffective Memory Bandwidth: 3518.77 GB/s
```

### 2.3 TV Layout

在2.2中, 我们依旧要手工去处理Thread domain的Layout映射, 例如

```
    mi = thread_idx // n    ni = thread_idx % n        a[(None, (mi, ni))].load()
```

对于更高维的场景, 计算这样的映射更加复杂, 是否有一种更简单的直观的代数运算, 直接得到呢? 对于原始张量  我们可以按照某种方式将其切分为很多个大小为(TileM, TileN)的Tile. 我们将其定义为`tiler_mn`的未知变量, 暂时不考虑其具体实现, 构造的切分如下

```
gA = cute.zipped_divide(mA, tiler_mn) # ((TileM, TileN), (RestM, RestN))
```

然后我们通过每个线程块(Thread block)去处理一个小为(TileM, TileN)的Tile. 通过线程块的索引. 即我们可以通过这个Layout的第二个Mode即(RestM, RestN)进行索引. 即对于一个GPU而言, 我们在其Grid内划分为(RestM, RestN)个block来处理. 也就是我们在launch kernel时 grid的参数设置为`cute.size(gC, mode=[1])`的原因.  简单起见对于Grid内的block我们按照一维排布. 即

```
 grid=[cute.size(gC, mode=[1]), 1, 1]
```

在Kernel中, 我们可以通过block-idx获取出该block对应的子张量, 即`gA[((None, None), bidx)]`, 这会返回一个单个 `(TileM, TileN)` 子张量的线程块局部视图.

```
blk_coord = ((None, None), bidx) blkA = gA[blk_coord] # (TileM, TileN) -> physical address
```

也就是说我们通过Layout blkA 可以得到一个映射函数

现在, 一个线程块拿到了一个 `(TileM, TileN)` 的任务. 接下来就要把这个任务再细分给块内的几百个线程. 从GPU物理结构上来看, 一个SM架构如下:

![图片](assets/ac520cb04c52.jpg)

它包含了4个warp, 每个warp包含32个cuda core, 累计4x32个cuda core. 那么我们就可以把线程按照(4,32):(32:1)的方式构建layout, 即让一个 Warp (32个线程) 的线程按行连续加载数据, 4个不同的warp分别读取不同的行, 这样我们就获得了`thr_layout= cute.make_layout((4, 32), stride=(32, 1))`

注: 有些时候, 我们也可以增加warp的数量, 让warp scheduler调度, 使得一个block内有更多的线程隐藏内存访问延迟, 例如当我们需要这个block有256个线程时可以使用 thr_layout= cute.make_layout((8, 32), stride=(32, 1))

然后对于每个线程需要处理的数据, 我们可以构造一个Value Layout, 例如我们为了保证向量化的LD/ST, 至少一行要有连续的4个数据, 例如`val_layout = cute.make_layout((4, 4), stride=(4, 1))`即一个row-major的值矩阵, 每行有4个连续的值便于向量化LD/ST, 一个线程处理4个不同的行.  当我们定义完了`thr_layout`和`val_layout`后, 我们需要构建整个Tile的Layout, 即需要一种方式把`val_layout` "插入"到`thread_layout`中. 这样的操作就是raked_product.

![图片](assets/238bb4de5b9f.png)

通过这样的操作, 我们就可以获得 `(TileM, TileN)` 中坐标(m',n')到thr_idx,val_idx的映射, 即

事实上, 我们在线程的Kernel运算时, 需要它的逆函数即一个能够根据thr_idx, val_idx的layout映射, 如下所示:

即我们可以构造一个临时的以(thr_size,val_size)为形状的tmp layout, 然后和复合, 最终我们就构造完成了tv_layout函数

```
@cute.jitdef tv_layout():    thr_layout = cute.make_layout((4, 32), stride=(32, 1))    val_layout = cute.make_layout((4, 4), stride=(4, 1))    layout_mn = cute.raked_product(thr_layout,val_layout)    print(f"layout mn->thr_idx,val_idx {layout_mn}")    thr_size = cute.size(thr_layout)    val_size = cute.size(val_layout)    print(f"thrsize: {thr_size} val_size: {val_size}")    # 创建一个代表Tile坐标域的临时布局    tmp = cute.make_layout((thr_size,val_size))        # 通过求逆和组合, 构造从 (thr_idx, val_idx) 到 (M,N) 的映射    layout_tv = cute.composition(        cute.right_inverse(layout_mn), tmp    )    print(f"layout_tv: {layout_tv}")        # 计算Tiler,即(TileM,TileN)    tiler_mn = cute.product_each(layout_mn.shape)    return (tiler_mn,layout_tv)tv_layout()
```

这也是`cute.make_layout_tv`的实现方式, 需要补充的是, 通常这样的TV Layout还需要输出一个(TileM,TileN)的元组用于对整个矩阵进行zipped tile, 即上文函数中的tiler_mn的计算.

对于一个线程通过TV_Layout可以通过, 同时我们注意到通过block-idx可以得到`gA[((None, None), bidx)]`, 它是一个. 因此将两个函数复合即可得到

即在cutDSL中我们可以通过如下方式获得相应的物理地址了, 然后利用TensorSSA做load和加法即可.

```
    tidfrgA = cute.composition(blkA, tv_layout)    tidfrgB = cute.composition(blkB, tv_layout)    tidfrgC = cute.composition(blkC, tv_layout)    thr_coord = (tidx, None)    thrA = tidfrgA[thr_coord]  # (V) -> physical address    thrB = tidfrgB[thr_coord]  # (V) -> physical address    thrC = tidfrgC[thr_coord]  # (V) -> physical address    #TensorSSA load and add    thrC[None] = thrA.load() + thrB.load()
```

这样我们就完成了整个TV_Layout的过程. 综上我们可以构建出一个完整的基于TV_Layout的elementwise_add

```
@cute.kerneldef elementwise_add_kernel(    gA: cute.Tensor,    gB: cute.Tensor,    gC: cute.Tensor,    tv_layout: cute.Layout):    tidx, _, _ = cute.arch.thread_idx()    bidx, _, _ = cute.arch.block_idx()    #--------------------------------    # slice for thread-block level view    #--------------------------------    blk_coord = ((None, None), bidx)    # logical coord -> address    blkA = gA[blk_coord]  # (TileM, TileN) -> physical address    blkB = gB[blk_coord]  # (TileM, TileN) -> physical address    blkC = gC[blk_coord]  # (TileM, TileN) -> physical address    #--------------------------------    # compose for thread-index & value-index to physical mapping    #--------------------------------    # blockA:    (TileM, TileN) -> physical address    # tv_layout: (tid, vid)     -> (TileM, TileN)    # tidfrgA = blkA o tv_layout    # tidfrgA:   (tid, vid) -> physical address    tidfrgA = cute.composition(blkA, tv_layout)    tidfrgB = cute.composition(blkB, tv_layout)    tidfrgC = cute.composition(blkC, tv_layout)    print(f"Composed with TV layout:")    print(f"  tidfrgA: {tidfrgA.type}")    #--------------------------------    # slice for thread-level view    #--------------------------------    # `None` represent slice of the entire per-thread data    thr_coord = (tidx, None)    # slice for threads: vid -> address    thrA = tidfrgA[thr_coord]  # (V) -> physical address    thrB = tidfrgB[thr_coord]  # (V) -> physical address    thrC = tidfrgC[thr_coord]  # (V) -> physical address    print(f"thrA : {thrA}")    thrC[None] = thrA.load() + thrB.load()    @cute.jitdef elementwise_add(    mA: cute.Tensor,    mB: cute.Tensor,    mC: cute.Tensor,):    # mA layout: (M, N):(N, 1)    # TV layout map thread & value index to (16, 256) logical tile    #  - contiguous thread index maps to mode-1 because input layout is contiguous on    #     mode-1 for coalesced load-store    #  - each thread load 8 contiguous element each row and load 4 rows    thr_layout = cute.make_layout((4, 32), stride=(32, 1))    val_layout = cute.make_layout((4, 4), stride=(4, 1))    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)    print(f"Tiler: {tiler_mn}")    print(f"TV Layout: {tv_layout}")    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))    print(f"Tiled Input Tensors:")    print(f"  gA: {gA.type}")    print(f"  gB: {gB.type}")    print(f"  gC: {gC.type}")    print(f" block-size: {cute.size(gC, mode=[1])}, thread-size: {cute.size(tv_layout, mode=[0])}")    # Launch the kernel asynchronously    # Async token(s) can also be specified as dependencies    elementwise_add_kernel(        gA, gB, gC, tv_layout    ).launch(        grid=[cute.size(gC, mode=[1]), 1, 1],        block=[cute.size(tv_layout, mode=[0]), 1, 1],    )
```

然后我们依旧用4096x4096的fp32矩阵进行验证和性能测试

```
M,N = 4096,4096a = torch.randn(M, N, device="cuda", dtype=torch.float32)b = torch.randn(M, N, device="cuda", dtype=torch.float32)c = torch.zeros(M, N, device="cuda", dtype=torch.float32)a_ = from_dlpack(a, assumed_align=16)b_ = from_dlpack(b, assumed_align=16)c_ = from_dlpack(c, assumed_align=16)elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)elementwise_add_(a_, b_, c_)# verify correctnesstorch.testing.assert_close(c, a + b)benchmark(partial(elementwise_add_, a_, b_, c_), num_warmups=5, num_iterations=200)
```

在Jetson Thor上的输出为:

```
Tiler: (16, 128)TV Layout: ((32,4),(4,4)):((64,4),(16,1))Tiled Input Tensors:  gA: !cute.memref<f32, gmem, align<16>, "((16,128),(256,32)):((4096,1),(65536,128))">  gB: !cute.memref<f32, gmem, align<16>, "((16,128),(256,32)):((4096,1),(65536,128))">  gC: !cute.memref<f32, gmem, align<16>, "((16,128),(256,32)):((4096,1),(65536,128))"> block-size: 8192, thread-size: 128Composed with TV layout:  tidfrgA: !cute.memref<f32, gmem, align<16>, "((32,4),(4,4)):((4,16384),(1,4096))">thrA : tensor<ptr<f32, gmem, align<16>> o ((4,4)):((1,4096))>Average execution time: 0.8017 msPerformance (GFLOPS): 20.9269 GFLOPSEffective Memory Bandwidth: 251.12 GB/s
```

在H20上的输出为:

```
Average execution time: 0.0601 msPerformance (GFLOPS): 279.2836 GFLOPSEffective Memory Bandwidth: 3351.40 GB/s
```

稍微展开一下, cuteDSL还增加了一个lambda function的能力, 即当我们需要elementwise的其它操作时, 可以通过一个op定义, 如下所示

```
@cute.kerneldef elementwise_apply_kernel(    op: cutlass.Constexpr,    # lambda function must be const expr to generate code at compile time    gA: cute.Tensor,    gB: cute.Tensor,    gC: cute.Tensor,    tv_layout: cute.Layout):    tidx, _, _ = cute.arch.thread_idx()    bidx, _, _ = cute.arch.block_idx()    blk_coord = ((None, None), bidx)    # logical coord -> address    blkA = gA[blk_coord]  # (TileM, TileN) -> physical address    blkB = gB[blk_coord]  # (TileM, TileN) -> physical address    blkC = gC[blk_coord]  # (TileM, TileN) -> physical address    tidfrgA = cute.composition(blkA, tv_layout)    tidfrgB = cute.composition(blkB, tv_layout)    tidfrgC = cute.composition(blkC, tv_layout)    print(f"Composed with TV layout:")    print(f"  tidfrgA: {tidfrgA.type}")    thr_coord = (tidx, None)    # slice for threads: vid -> address    thrA = tidfrgA[thr_coord]  # (V) -> physical address    thrB = tidfrgB[thr_coord]  # (V) -> physical address    thrC = tidfrgC[thr_coord]  # (V) -> physical address    #--------------------------------    # apply custom operation    #--------------------------------    thrC[None] = op(thrA.load(), thrB.load())@cute.jitdef elementwise_op(    op: cutlass.Constexpr,    mA: cute.Tensor,    mB: cute.Tensor,    mC: cute.Tensor,):    # mA layout: (M, N):(N, 1)    # TV layout map thread & value index to (16, 256) logical tile    #  - contiguous thread index maps to mode-1 because input layout is contiguous on    #     mode-1 for coalesced load-store    #  - each thread load 8 contiguous element each row and load 4 rows    thr_layout = cute.make_layout((4, 32), stride=(32, 1))    val_layout = cute.make_layout((4, 8), stride=(8, 1))    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)    print(f"Tiler: {tiler_mn}")    print(f"TV Layout: {tv_layout}")    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))    print(f"Tiled Input Tensors:")    print(f"  gA: {gA.type}")    print(f"  gB: {gB.type}")    print(f"  gC: {gC.type}")    # Launch the kernel asynchronously    # Async token(s) can also be specified as dependencies    elementwise_apply_kernel(        op, gA, gB, gC, tv_layout    ).launch(        grid=[cute.size(gC, mode=[1]), 1, 1],        block=[cute.size(tv_layout, mode=[0]), 1, 1],    )a = torch.randn(M, N, device="cuda", dtype=torch.float16)b = torch.randn(M, N, device="cuda", dtype=torch.float16)c = torch.zeros(M, N, device="cuda", dtype=torch.float16)a_ = from_dlpack(a, assumed_align=16)b_ = from_dlpack(b, assumed_align=16)c_ = from_dlpack(c, assumed_align=16)from operator import mulelementwise_op(mul, a_, b_, c_)# verify correctnesstorch.testing.assert_close(c, mul(a, b))
```

## 3. TileLang

TileLang官方也提供了一个elementwise操作的介绍《ElementWise Operators》[2]

### 3.1 Naive elementwise add

在Tilelang中, 我们并不需要像CuteDSL那样考虑复杂的Layout, 而只是需要简单的把(TileM,TileN)通过T.Parallel(TIleM, TIleN)并行调度即可.

```
import torchimport tilelangimport tilelang.language as Tdef elementwise_add(    M,    N,    TileM,    TileN,    in_dtype,    out_dtype,    threads,):    @T.prim_func    def main(            A: T.Tensor((M, N), in_dtype),            B: T.Tensor((M, N), in_dtype),            C: T.Tensor((M, N), out_dtype),    ):        with T.Kernel(T.ceildiv(N, TileN), T.ceildiv(M, TileM), threads=threads) as (bx, by):            start_x = bx * TileN            start_y = by * TileM            for (local_y, local_x) in T.Parallel(TileM, TileN):                y = start_y + local_y                x = start_x + local_x                C[y, x] = A[y, x] + B[y, x]    return main
```

Kernel编译和验证代码如下

```
M,N = 4096,4096TileM,TileN = 128,128func = elementwise_add(M, N, TileM,TileN, "float32","float32", 256)jit_kernel = tilelang.compile(func, out_idx=[-1], target="cuda")a = torch.randn(M, N, device="cuda", dtype=torch.float32)b = torch.randn(M, N, device="cuda", dtype=torch.float32)c = torch.zeros(M, N, device="cuda", dtype=torch.float32)c = jit_kernel(a,b)# verify correctnesstorch.testing.assert_close(c, a + b)
```

在Tilelang中内置了profiler, 我们可以通过很简单的do_bench()的方法进行性能测量

```
profiler = jit_kernel.get_profiler()avg_time = profiler.do_bench(n_warmup=5000) #n_warmup=5000 for thorgflops =  a.numel()  / (avg_time  / 1000) / 1e9print(f"Average execution time: {avg_time:.4f} ms")print(f"Performance (GFLOPS): {gflops:.4f} GFLOPS")print(f"Effective Memory Bandwidth: {(3 * a.numel() * 4) / (avg_time / 1000) / 1e9:.2f} GB/s")
```

在Jetson Thor的结果如下:

```
Average execution time: 0.9977 msPerformance (GFLOPS): 16.8165 GFLOPSEffective Memory Bandwidth: 201.80 GB/s
```

在H20上的结果如下:

```
Average execution time: 0.0626 msPerformance (GFLOPS): 268.1780 GFLOPSEffective Memory Bandwidth: 3218.14 GB/s
```

然后我们可以通过如下方式获得生成的cuda kernel源码

```
cuda_source = jit_kernel.get_kernel_source()print("Generated CUDA kernel:\n", cuda_source)
```

输出如下, 可以看到已经使用了float4并执行了Vectorized LD/ST.

```
Generated CUDA kernel: #include <tl_templates/cuda/gemm.h>#include <tl_templates/cuda/copy.h>#include <tl_templates/cuda/reduce.h>#include <tl_templates/cuda/ldsm.h>#include <tl_templates/cuda/threadblock_swizzle.h>#include <tl_templates/cuda/debug.h>#ifdef ENABLE_BF16#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>#endifextern "C" __global__ void main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {  #pragma unroll  for (int i = 0; i < 16; ++i) {    float4 __1;      float4 v_ = *(float4*)(A + (((((((int)blockIdx.y) * 524288) + (i * 32768)) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4)));      float4 v__1 = *(float4*)(B + (((((((int)blockIdx.y) * 524288) + (i * 32768)) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4)));      __1.x = (v_.x+v__1.x);      __1.y = (v_.y+v__1.y);      __1.z = (v_.z+v__1.z);      __1.w = (v_.w+v__1.w);    *(float4*)(C + (((((((int)blockIdx.y) * 524288) + (i * 32768)) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4))) = __1;  }}#define ERROR_BUF_SIZE 1024static char error_buf[ERROR_BUF_SIZE];extern "C" const char* get_last_error() {    return error_buf;}extern "C" int init() {    error_buf[0] = '\0';        return 0;}extern "C" int call(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream=cudaStreamDefault) { main_kernel<<<dim3(32, 32, 1), dim3(256, 1, 1), 0, stream>>>(A, B, C); TILELANG_CHECK_LAST_ERROR("main_kernel"); return 0;}
```

### 3.2 AutoTune

对于这个kernel, 我们有几个超参数 TileM,TileN和num_thread, Tilelang内置了AutoTune功能, 如下所示:

```
import itertoolsdef get_config():    TILE_M = [8, 16, 32, 64, 128, 256]    TILE_N = [8, 16, 32, 64, 128, 256]    N_THREAD = [128, 256, 512]    _configs = list(itertools.product(        TILE_M,        TILE_N,        N_THREAD    ))    configs = [{        "TileM" : c[0],        "TileN" : c[1],        "threads" : c[2],    } for c in _configs]    return configs@tilelang.autotune(    configs= get_config(),    warmup= 5,    rep = 20,)@tilelang.jit(out_idx=[-1], target="cuda")def elementwise_add(    TileM,    TileN,    threads,    M=4096,    N=4096,        in_dtype="float32",    out_dtype="float32",):    @T.prim_func    def main(            A: T.Tensor((M, N), in_dtype),            B: T.Tensor((M, N), in_dtype),            C: T.Tensor((M, N), out_dtype),    ):        with T.Kernel(T.ceildiv(N, TileN), T.ceildiv(M, TileM), threads=threads) as (bx, by):            start_x = bx * TileN            start_y = by * TileM            for (local_y, local_x) in T.Parallel(TileM, TileN):                y = start_y + local_y                x = start_x + local_x                C[y, x] = A[y, x] + B[y, x]    return mainauto_kernel = elementwise_add()
```

然后我们可以从auto_kernel中获得最佳配置

```
best_config = auto_kernel.configbest_latency = auto_kernel.latencyprint(f"Best Config: {best_config}")gflops =  a.numel()  / (best_latency  / 1000) / 1e9print(f"Average execution time: {best_latency:.4f} ms")print(f"Performance (GFLOPS): {gflops:.4f} GFLOPS")print(f"Effective Memory Bandwidth: {(3 * a.numel() * 4) / (best_latency / 1000) / 1e9:.2f} GB/s")
```

在Jetson Thor上的输出, 可以看到Tilelang基本上把物理带宽打满了, 性能也超越了Cublas baseline

```
Best Config: {'TileM': 128, 'TileN': 64, 'threads': 128}Average execution time: 0.8069 msPerformance (GFLOPS): 20.7911 GFLOPSEffective Memory Bandwidth: 249.49 GB/s
```

在H20上的输出

```
Best Config: {'TileM': 32, 'TileN': 64, 'threads': 128}Average execution time: 0.0618 msPerformance (GFLOPS): 271.6518 GFLOPSEffective Memory Bandwidth: 3259.82 GB/s
```

参考资料

[1] 
Tutorial: Elementwise Add Kernel in CuTe DSL: *https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb*
[2] 
TileLang ElementWise Operators: *https://tilelang.com/deeplearning_operators/elementwise.html*