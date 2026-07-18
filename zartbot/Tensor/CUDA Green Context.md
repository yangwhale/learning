# CUDA Green Context

> 作者: zartbot  
> 日期: 2025年5月7日 02:12  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494140&idx=1&sn=b55b0805a77193ceb02a8a9ee658c26b&chksm=f995f93ecee2702881aef2d2a9616511fb2b68ffc43e063301fcb4d88fb84a973a65e879db93#rd

---

CUDA Green Context[1]是传统Context的轻量级替代, 对于PaaS/MaaS进行多个模型的推理和一些算子灵活的调用可能有一些优势. 但是官方文档似乎没有example code, 于是今天早上花了一点时间找了张A10的卡测试了一下.

### 1. 传统的Context

我们先构建测试用的几个kernel, 并生成cubin文件供不同的Context加载, kernel测试代码如下:

```
//kernel_1.cu#include <cuda_runtime.h>#include <iostream>const int SIZE_N = 32 *512;__global__ void kernel_1(float *data){    int idx = threadIdx.x + blockIdx.x * blockDim.x;    float v = data[idx];    int i = 0;    for (int j = 0; i < 10240; ++j)    {        i = j % SIZE_N;        v += logf(data[idx+i]);        data[idx+i] = v;    }}__global__ void kernel_2(float *data){    int idx = threadIdx.x + blockIdx.x * blockDim.x;    float v = data[idx];    int i = 0;    for (int j = 0; i < 10240; ++j)    {        i = j % SIZE_N;        v += logf(data[idx+i]);        data[idx+i] = v;    }}__global__ void kernel_3(float *data){    int idx = threadIdx.x + blockIdx.x * blockDim.x;    float v = data[idx];    int i = 0;    for (int j = 0; i < 10240; ++j)    {        i = j % SIZE_N;        v += expf(data[idx+i]);        data[idx+i] = v;    } }__global__ void kernel_4(float *data){    int idx = threadIdx.x + blockIdx.x * blockDim.x;    float v = data[idx];    int i = 0;    for (int j = 0; i < 10240; ++j)    {        i = j % SIZE_N;        v += data[idx+i];        data[idx+i] = v;    }    data[idx] = v;}
```

生成cubin, 并获得函数签名

```
 nvcc -arch=sm_86 -ptx kernel_1.cu -o kernel_1.ptx nvcc -arch=sm_86  kernel_1.ptx -cubin -o kernel_1.cubin  more kernel_1.ptx  | grep ".globl"        // .globl       _Z8kernel_1Pf        // .globl       _Z8kernel_2Pf        // .globl       _Z8kernel_3Pf        // .globl       _Z8kernel_4Pf
```

传统的context使用方式如下:

```
#include <cuda_runtime.h>#include <cuda.h>#include <iostream>const int GRID_SIZE = 32;const int BLOCK_SIZE = 512;const int CTX_NUM = 4;#define CHECK_CUDA(func)                                                  \    {                                                                     \        CUresult status = (func);                                         \        if (status != CUDA_SUCCESS)                                       \        {                                                                 \            std::printf("CUDA API failed at line %d with error:  (%d)\n", \                        __LINE__, status);                                \            return EXIT_FAILURE;                                          \        }                                                                 \    }int main(){    cuInit(0);    CUdevice dev;    cuDeviceGet(&dev, 0);    CUmodule module[CTX_NUM];    CUfunction kernel[CTX_NUM];    CUcontext ctx[CTX_NUM];    CUstream stream[CTX_NUM];    float *data[CTX_NUM];    const char *func_name[CTX_NUM] = {"_Z8kernel_1Pf", "_Z8kernel_2Pf", "_Z8kernel_3Pf", "_Z8kernel_4Pf"};    for (int i = 0; i < CTX_NUM; ++i)    {        //创建Context        cuCtxCreate(&ctx[i], 0, dev);        cuCtxSetCurrent(ctx[i]);        //加载Module        cuModuleLoad(&module[i], "kernel_1.cubin");                //获取函数        CHECK_CUDA(cuModuleGetFunction(&kernel[i], module[i], func_name[i]));                cudaMalloc(&data[i], sizeof(float) * GRID_SIZE * BLOCK_SIZE);                //创建cudastream        cuStreamCreate(&stream[i], CU_STREAM_NON_BLOCKING);    }    cudaEvent_t start, end;    cudaEventCreate(&start);    cudaEventCreate(&end);    cudaEventRecord(start);    for (int i = 0; i < CTX_NUM; ++i)    {        //通过cudastream launch kernel        void *kernelParams[] = {(void *)&data[i]};        cuLaunchKernel((CUfunction)kernel[i], GRID_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, stream[i], kernelParams, 0);    }    for (int i = 0; i < CTX_NUM; ++i)    {        cuStreamSynchronize(stream[i]);    }    cudaEventRecord(end);    cudaEventSynchronize(end);    float msec;    cudaEventElapsedTime(&msec, start, end);    printf("Normal Elapsed: %5.3lf ms\n", msec);}
```

执行时间

```
nvcc -arch=sm_86  -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lcuda  ctx.cuNormal Elapsed: 9.172 ms
```

### 2. Green Context

Green Context创建相对复杂一些, 代码如下:

```
#include <cuda_runtime.h>#include <cuda.h>#include <iostream>const int GRID_SIZE = 32;const int BLOCK_SIZE = 512;const int CTX_NUM = 4;#define CHECK_CUDA(func)                                                  \    {                                                                     \        CUresult status = (func);                                         \        if (status != CUDA_SUCCESS)                                       \        {                                                                 \            std::printf("CUDA API failed at line %d with error:  (%d)\n", \                        __LINE__, status);                                \            return EXIT_FAILURE;                                          \        }                                                                 \    }int main(){    cuInit(0);    CUdevice dev;    cuDeviceGet(&dev, 0);    /*    (1) Start with an initial set of resources, for example via cuDeviceGetDevResource. Only SM type is supported today.    (2) Partition this set of resources by providing them as input to a partition API, for example: cuDevSmResourceSplitByCount.    (3) Finalize the specification of resources by creating a descriptor via cuDevResourceGenerateDesc.    (4) Provision the resources and create a green context via cuGreenCtxCreate.    */    CUdevResource resource;    cuDeviceGetDevResource(dev, &resource, CU_DEV_RESOURCE_TYPE_SM);        //最少占用80%的SM资源    unsigned int minCount;    minCount = (unsigned int)((float)resource.sm.smCount * 0.8f);    unsigned int split_group = CTX_NUM;    //基于SM资源分配    CUdevResource split_resource[CTX_NUM];    cuDevSmResourceSplitByCount(split_resource, &split_group, &resource, 0, CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, minCount);    //创建资源描述符    CUdevResourceDesc split_desc[CTX_NUM];    cuDevResourceGenerateDesc(split_desc, split_resource, split_group);    CUgreenCtx gctx[CTX_NUM];    CUstream gstream[CTX_NUM];    CUmodule module[CTX_NUM];    CUfunction kernel[CTX_NUM];    float *data[CTX_NUM];    const char *func_name[CTX_NUM] = {"_Z8kernel_1Pf", "_Z8kernel_2Pf", "_Z8kernel_3Pf", "_Z8kernel_4Pf"};    for (int i = 0; i < CTX_NUM; ++i)    {        //根据资源描述符创建green context和cuda stream, 并加载Module/Function        cuGreenCtxCreate(&gctx[i], split_desc[i], dev, CU_GREEN_CTX_DEFAULT_STREAM);        cuGreenCtxStreamCreate(&gstream[i], gctx[i], CU_STREAM_NON_BLOCKING, 0);        //SetCurrentContext 需要先将Green context转换为传统Context        CUcontext ctx;        cuCtxFromGreenCtx (&ctx,gctx[i]);        cuCtxSetCurrent(ctx);        //加载Module        cuModuleLoad(&module[i], "kernel_1.cubin");        CHECK_CUDA(cuModuleGetFunction(&kernel[i], module[i], func_name[i]));        cudaMalloc(&data[i], sizeof(float) * GRID_SIZE * BLOCK_SIZE);    }    cudaEvent_t start, end;    cudaEventCreate(&start);    cudaEventCreate(&end);    cudaEventRecord(start);    for (int i = 0; i < CTX_NUM; ++i)    {        //使用green context对应的cuda stream调用kernel        void *kernelParams[] = {(void *)&data[i]};        cuLaunchKernel((CUfunction)kernel[i], GRID_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, gstream[i], kernelParams, 0);    }    for (int i = 0; i < CTX_NUM; ++i)    {        cuStreamSynchronize(gstream[i]);    }    cudaEventRecord(end);    cudaEventSynchronize(end);    float msec;    cudaEventElapsedTime(&msec, start, end);    printf("Green Context Elapsed: %5.3lf ms\n", msec);}
```

测试执行时间:

```
vcc -arch=sm_86  -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lcuda  green_ctx.cu Green Context Elapsed: 3.254 ms
```

相对于Normal Context性能快了3倍...

参考资料

[1] 
CUDA Green Context: *https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html*