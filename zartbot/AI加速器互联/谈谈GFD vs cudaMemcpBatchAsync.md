# 谈谈GFD vs cudaMemcpBatchAsync

> 作者: zartbot  
> 日期: 2026年5月25日 00:37  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498836&idx=2&sn=fdd971da3a00a3c2d61728a8b909f997&chksm=f995ec96cee26580bff1cb094e1e408873415b90f5e1135fbcb940b9932094b69f35feb51036#rd

---

### TL;DR

前几天写了一个[《Agent写的一个高性能Host-to-Device 传输库》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498446&idx=1&sn=0a9a70a346ff1ba03348ff07d11829e2&scene=21#wechat_redirect), 评论区有读者反馈在CUDA 12.8中增加了一个新的API `cudamemcpBatchAsync`. 本文就来进行详细的测试.  代码已经更新:**https://github.com/zartbot/gfd**

### 1. 什么是cudamemcpBatchAsync

`cudaMemcpyBatchAsync` 是 CUDA 12.8 引入的批量异步内存拷贝 API. 它允许用户通过**一次 API 调用**提交 N 个独立的内存拷贝操作, 避免了逐个调用 `cudaMemcpyAsync` 的 API 开销. 传统方式下, 如果需要将 N 个离散的 CPU 内存块拷贝到 GPU, 必须调用 N 次 `cudaMemcpyAsync`, 每次调用都有约 1-2 微秒的 API 开销. 当 N 很大(如 LLM 推理中的数千个 KV-cache block)时, API 开销成为主要瓶颈. `cudaMemcpyBatchAsync` 将 N 个拷贝请求打包为一次调用, 由驱动统一处理.

![图片](assets/6af03a60920d.png)

```
// API 签名cudaError_t cudaMemcpyBatchAsync(    void* const*           dstArray,       // N 个目标地址数组    const void* const*     srcArray,       // N 个源地址数组    size_t*                sizeArray,      // N 个拷贝大小数组    size_t                 count,          // 条目数 N    cudaMemcpyAttributes*  attrArray,      // 属性数组(描述内存类型)    size_t*                attrIdxArray,   // 每条目对应的属性索引    size_t                 numAttrs,       // 属性数量    cudaStream_t           stream          // CUDA 流);// cudaMemcpyAttributes 结构struct cudaMemcpyAttributes {    cudaMemcpySrcAccessOrder  srcAccessOrder;  // 源内存访问顺序    unsigned int              flags;            // 保留标志(设为 0)};
```

相关参数说明:

![图片](assets/1323f31ecc7d.png)

具体工作原理:

![图片](assets/2c885c6f63df.png)

**核心优势:**  将 N 次 API 调用合并为 1 次, 消除了 N-1 次的用户态→内核态切换开销.

**局限性:** 驱动内部仍然为每个条目生成独立的 CE DMA 命令, CE 硬件顺序执行这 N 个命令——这是性能远低于 GFD** 的根本原因.

### 2. 测试结果
测试环境
**GPU:** NVIDIA RTX PRO 5000 72GB(Blackwell, sm_120)

**PCIe:** Gen5 x16(实际带宽约 53 GB/s)

**CPU:** 256 核, 2 个 NUMA** 节点

**GFD 配置:** 15 个 gather 工作线程, 3 个 CE 通道, 5 倍大页 staging 缓冲区

**布局:** Token 以 2 倍步长分散在锁页 CPU 内存中

**迭代:** 每配置 50 次, 15 次预热

测试代码: `examples/04_benchmark.cu` — 执行 `./gfd_benchmark`

![图片](assets/23db562a0db1.png)

对比分析:

![图片](assets/b2bffff55d0c.png)

两者实现差异如下:

![图片](assets/ddef17f9f4ec.png)

![图片](assets/18e25be1088b.png)

### 3. 多GPU并行测试

多GPU场景测试, **每 GPU 传输量:** 2048 × 4KB = 8 MB(以 2 倍步长分散), 代码`examples/05_multi_gpu_benchmark.cu` — 执行 `./gfd_multi_gpu_benchmark`

![图片](assets/265e5defeb9b.png)

可以发现 cudamemcpBatchAsync 在多卡场景下存在严重的问题.

![图片](assets/b9577eb064f6.png)

主要原因在于, 8 个线程竞争同一把驱动锁, 每个线程持锁约 717 微秒构建 2048 个 CE 命令. 平均每个 GPU 等待 3.5 个 GPU 释放锁 → 等待时间 ≈ 3.5 × 717 = 2510 微秒.

![图片](assets/529ad074876e.png)

为什么GFD能够线性缩放**呢?

![图片](assets/49b584bc4e29.png)

![图片](assets/0d1c15f30609.png)

![图片](assets/030dde9a4632.png)

### 附录 A.  cudamemcpBatchAsync 离散 H2D 批量拷贝

```
#include <cuda_runtime.h>#include <vector>#include <cstdio>int main() {    const int NUM_TOKENS = 2048;    const size_t TOKEN_SIZE = 4096;  // 4KB per token    // 1. 分配 GPU 连续目标缓冲区    char* gpu_buf;    cudaMalloc(&gpu_buf, NUM_TOKENS * TOKEN_SIZE);    // 2. 分配 CPU 锁页源缓冲区(模拟离散 KV-cache)    char* cpu_buf;    cudaMallocHost(&cpu_buf, NUM_TOKENS * TOKEN_SIZE * 2);  // 2x stride 模拟离散    // 填充测试数据    for (int i = 0; i < NUM_TOKENS; i++) {        memset(cpu_buf + i * TOKEN_SIZE * 2, i & 0xFF, TOKEN_SIZE);    }    // 3. 构建批量拷贝参数数组    std::vector<void*> dsts(NUM_TOKENS);    std::vector<const void*> srcs(NUM_TOKENS);    std::vector<size_t> sizes(NUM_TOKENS);    for (int i = 0; i < NUM_TOKENS; i++) {        dsts[i] = gpu_buf + i * TOKEN_SIZE;          // GPU 连续排列        srcs[i] = cpu_buf + i * TOKEN_SIZE * 2;      // CPU 离散(2x stride)        sizes[i] = TOKEN_SIZE;    }    // 4. 设置内存属性    cudaMemcpyAttributes attr = {};    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;  // 源为锁页内存    attr.flags = 0;    // 所有条目共享同一个属性(索引 0)    std::vector<size_t> attrIdxs(NUM_TOKENS, 0);    // 5. 创建流并执行批量拷贝    cudaStream_t stream;    cudaStreamCreate(&stream);    cudaMemcpyBatchAsync(        (void* const*)dsts.data(),        (const void* const*)srcs.data(),        sizes.data(),        NUM_TOKENS,        &attr,        attrIdxs.data(),        1,              // numAttrs = 1(只有一种属性)        stream    );    cudaStreamSynchronize(stream);    // 6. 验证    std::vector<char> verify(TOKEN_SIZE);    cudaMemcpy(verify.data(), gpu_buf, TOKEN_SIZE, cudaMemcpyDeviceToHost);    printf("第一个 token 首字节: 0x%02x (期望 0x00)\n", (unsigned char)verify[0]);    cudaMemcpy(verify.data(), gpu_buf + 100 * TOKEN_SIZE, TOKEN_SIZE, cudaMemcpyDeviceToHost);    printf("第 100 个 token 首字节: 0x%02x (期望 0x64)\n", (unsigned char)verify[0]);    // 7. 清理    cudaStreamDestroy(stream);    cudaFree(gpu_buf);    cudaFreeHost(cpu_buf);    printf("批量拷贝完成: %d 个 token, 每个 %zu 字节\n", NUM_TOKENS, TOKEN_SIZE);    return 0;}
```