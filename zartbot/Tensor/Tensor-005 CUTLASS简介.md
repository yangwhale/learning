# Tensor-005 CUTLASS简介

> 作者: zartbot  
> 日期: 2024年8月20日 10:08  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491671&idx=2&sn=d52a54663b529f1e8498e51e64f88c39&chksm=f995f095cee279838395e69171115a61c034255d38f0a9d571d736170cc26234baca5d381020#rd

---

## 1. CUTLASS计算流程抽象

### 1.1 矩阵分块乘法

在前面一章我们介绍了如何使用TensorCore进行矩阵计算, 通常我们需要按照如下流程逐步分块从GMEM加载矩阵块到SMEM再到寄存器文件,然后进行矩阵乘法计算. 同时为了兼顾访存效率, 还需要在较新的GPU(Ampere/Hopper)上实现多级流水线的异步内存加载.

![图片](assets/db547f83b5bc.png)

### 1.2 Epilogue

前文的一些实现中仅有的矩阵乘法, 并没有考虑到, 而这些计算需要有一个后续的步骤进行修正, 如下所示

![图片](assets/16006a8b9ee0.png)

另一方面有一些矩阵乘法完成后基于单个元素的操作算子可以融合在一起,例如激活函数/数据类型转换等

![图片](assets/47a37ceb83eb.png)

将这些步骤拼凑在一起构成完成的计算流程

![图片](assets/6e2a6bf8ef61.png)

即在完成GMEM后, 还有一段代码在Cuda Core上执行Epilogue代码.

### 1.3 计算流抽象

但是我们注意到随着TensorCore支持的数据类型越来越多,各种混合精度的乘法以及不同的矩阵大小, 然后矩阵的不同Layout以及深度学习模型中各种算子融合的需求使得编程复杂度上升.

![图片](assets/a69652322473.png)

是否能够通过C++的一些模版来降低这些编写代码的复杂度? 我们可以从cutlass v0.1.1最早期的代码中发现这样的端倪. 然后整个计算流程被Cutlass抽象如下:

![图片](assets/36825f9aa92e.png)

CUTLASS的设计理念在官方文档的第一段讲的也非常清楚:

CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA.

It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement cuBLAS and cuDNN. CUTLASS decomposes these "moving parts" into reusable, modular software components abstracted by C++ template classes.

总结来看:

集成了一系列的C++模板抽象, 用于在CUDA的基础上提供高性能的矩阵乘法(GEMM)以及各个层次的相关计算.

分层分解和数据移动这些部件的可重用及模块化的定义

对于TileSize和并行策略可以进行层次化的基于模板参数的调整, 灵活的组合降低了开发难度

支持各种数据精度, 并且为各个平台提供了相应的高吞吐的TensorCore相关的实现.

更详细的内容可以参考官方的两个视频:

《半小时快速入门CUTLASS-基于 CUDA 的多层次稠密线性代数计算原语》[1]

《CUTLASS: Software Primitives for Dense Linear Algebra at All Levels and Scales within CUDA》[2]

## 2. CUTLASS Quickstart

### 2.1 Cutlass安装

在阿里云上开了一台GPU的服务器,Ubuntu 22.04 + Cuda 12.0, cutlass安装过程如下

```
cd /optgit clone https://github.com/NVIDIA/cutlasscd cutlassmkdir build && cd buildcmake .. -DCUTLASS_NVCC_ARCHS=90a    # for hoppercmake .. -DCUTLASS_NVCC_ARCHS=80     # for ampere
```

修改路径`vim ~/.bashrc`并添加

```
export CPLUS_INCLUDE_PATH=/opt/cutlass/include:/opt/cutlass/tools/util/include:$CPLUS_INCLUDE_PATHexport C_INCLUDE_PATH=/opt/cutlass/include:/opt/cutlass_test/cutlass/tools/util/include:$C_INCLUDE_PATH
```

### 2.2 测试Profiler

```
make cutlass_profiler -j12./tools/profiler/cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096=============================  Problem ID: 1        Provider: CUTLASS   OperationKind: gemm       Operation: cutlass_simt_sgemm_128x128_8x2_nn_align1          Status: Success    Verification: ON     Disposition: Passedreference_device: Passed          cuBLAS: Not run           cuDNN: Not run       Arguments: --gemm_kind=universal --m=4352 --n=4096 --k=4096 --A=f32:column --B=f32:column --C=f32:column --D=f32:column  \                  --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1 --raster_order=heuristic  \                  --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --cluster_m=1 --cluster_n=1 --cluster_k=1  \                  --stages=2 --warps_m=4 --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024  \           Bytes: 209715200  bytes           FLOPs: 146064539648  flops           FLOPs/Byte: 696         Runtime: 12.5042  ms          Memory: 15.6198 GiB/s            Math: 11681.3 GFLOP/s
```

## 3.  CUTLASS测试程序示例

我们可以通过一个例子来了解CUTLASS模板的使用方法.

### 3.1 矩阵定义

我们可以通过如下方式定义矩阵的数值类型和Layout方式,以及内存对齐的方式

```
// A matrix configurationusing ElementA = cutlass::half_t;                                       // Element type for A matrix operandusing LayoutA = cutlass::layout::RowMajor;                              // Layout type for A matrix operandconstexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)// B matrix configurationusing ElementB = cutlass::half_t;                                       // Element type for B matrix operandusing LayoutB = cutlass::layout::RowMajor;                              // Layout type for B matrix operandconstexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)// C/D matrix configurationusing ElementC = cutlass::half_t;                                       // Element type for C and D matrix operandsusing LayoutC = cutlass::layout::RowMajor;                              // Layout type for C and D matrix operandsconstexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)
```

### 3.2 矩阵乘法分块的方式

然后我们定义需要用到的GPU架构`ArchTag`, 计算精度`ElementAccumulator`以及矩阵分块的到ThreadBlock Tile , WarpTile, 以及最终使用的矩阵乘法的指令Shape, 同时我们还可以定义整个Global MEM加载到Shared MEM的流水线长度

```
// Multiply-accumulate blocking/pipelining detailsusing ElementAccumulator = cutlass::half_t;                      // Element type for internal accumulationusing ArchTag = cutlass::arch::Sm80;                             // Tag indicating the minimum SM that supports the intended featureusing OperatorClass = cutlass::arch::OpClassTensorOp;            // Operator class tagusing ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>; // Threadblock-level tile size (concept: GemmShape)using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;          // Warp-level tile size (concept: GemmShape)using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // Instruction-level tile size (concept: GemmShape)constexpr int NumStages = 4;                                     // Number of global->shared pipeline stages used in the GEMM mainloop
```

### 3.3 定义Epilogue操作

这里只有一个简单的alpha/beta的线性计算, 因此调用了`cutlass::epilogue::thread::LinearCombination`,即

```
// Epilogue output operatorusing EpilogueOp = cutlass::epilogue::thread::LinearCombination<    ElementC,            // Element type for C and D matrix operands    AlignmentC,          // Memory access granularity of C and D matrix in units of elements    ElementAccumulator,  // Element type from internal accumaccumulation    ElementAccumulator>; // Data type used to compute linear combination
```

### 3.4 定义算子

根据前述的各个组件组成GEMM算子

```
// Classic data-parallel device GEMM implementation typeusing DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<    ElementA, LayoutA,    ElementB, LayoutB,    ElementC, LayoutC,    ElementAccumulator,    OperatorClass,    ArchTag,    ThreadblockShape,    WarpShape,    InstructionShape,    EpilogueOp,    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,    NumStages,    AlignmentA,    AlignmentB>;
```

### 3.5 算子加载参数

对于矩阵乘法的Shape MNK, CUTLASS封装了一个GemmCoord对象

```
  const int length_m = 4096;  const int length_n = 4096;  const int length_k = 4096;  // Create a tuple of problem size for matrix multiplication  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
```

然后我们可以构建整个算子加载的参数对象

```
/// Populates a DeviceGemmBasic::Arguments structure from the given commandline optionstypename DeviceGemmBasic::Arguments args_from_options(    const DeviceGemmBasic &device_gemm,    const cutlass::gemm::GemmCoord problem_size,    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,    cutlass::HostTensor<ElementC, LayoutC> &tensor_d){  return typename DeviceGemmBasic::Arguments(      cutlass::gemm::GemmUniversalMode::kGemm, // universal mode      problem_size,                            // problem_size      1,                                      // batch count / splitk slices      {          // epilogue parameters          ElementAccumulator(1.0f), // alpha          ElementAccumulator(0.0f)  // beta      },      tensor_a.device_data(),       // ptr_A      tensor_b.device_data(),       // ptr_B      tensor_c.device_data(),       // ptr_C      tensor_d.device_data(),       // ptr_D      problem_size.mk().product(),  // batch_stride_A      problem_size.nk().product(),  // batch_stride_B      problem_size.mn().product(),  // batch_stride_C      problem_size.mn().product(),  // batch_stride_D      tensor_a.layout().stride(0),  // stride_a      tensor_b.layout().stride(0),  // stride_b      tensor_c.layout().stride(0),  // stride_c      tensor_d.layout().stride(0)); // stride_d}
```

### 3.6 矩阵参数初始化

张量的初始化,CUTLASS也进行了封装, 例如随机填充和全零填充等. 并通过tensor对象啊sync_device()函数进行拷贝

```
  // Initialize tensors using CUTLASS helper functions  cutlass::HostTensor<ElementA, LayoutA> tensor_a(      problem_size.mk()); // <- Create matrix A with dimensions M x K  cutlass::HostTensor<ElementB, LayoutB> tensor_b(      problem_size.kn()); // <- Create matrix B with dimensions K x N  cutlass::HostTensor<ElementC, LayoutC> tensor_c(      problem_size.mn()); // <- Create matrix C with dimensions M x N  cutlass::HostTensor<ElementC, LayoutC> tensor_d(      problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from  // Fill input and output matrices on host using CUTLASS helper functions  cutlass::reference::host::TensorFillRandomUniform(      tensor_a.host_view(),      1,      ElementA(4),      ElementA(-4),      0); // <- Fill matrix A on host with uniform-distribution random data  cutlass::reference::host::TensorFillRandomUniform(      tensor_b.host_view(),      1,      ElementB(4),      ElementB(-4),      0); // <- Fill matrix B on host with uniform-distribution random data  cutlass::reference::host::TensorFillRandomUniform(      tensor_c.host_view(),      1,      ElementC(4),      ElementC(-4),      0); // <- Fill matrix C on host with uniform-distribution random data  cutlass::reference::host::TensorFill(      tensor_d.host_view()); // <- fill matrix D on host with zeros  // Copy data from host to GPU  tensor_a.sync_device();  tensor_b.sync_device();  tensor_c.sync_device();  tensor_d.sync_device();
```

### 3.7 初始化GEMM算子

如下所示, 首先对算子实例化, 然后基于参数构建workspace,检查算子是否能够支持problem_size, 最后实例化Gemm Kernel

```
  // Instantiate CUTLASS kernel depending on templates  DeviceGemmBasic gemm_op;  auto arguments = args_from_options(gemm_op, problem_size, tensor_a, tensor_b, tensor_c, tensor_d);  // Using the arguments, query for extra workspace required for matrix multiplication computation  size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);  // Allocate workspace memory  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);  // Check the problem size is supported or not  gemm_op.can_implement(arguments);  // Initialize CUTLASS kernel with arguments and workspace pointer  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
```

最后`gemm_op()` 调用执行即可.

## 4. 附录

### 4.1 性能测试函数

最后我们基于这个算子进行性能测试

```
  cudaEvent_t start, end;  float elapsedTime;  cudaEventCreate(&start);  cudaEventCreate(&end);  cudaEventRecord(start);  const int ITER = 100;  for (int i = 0; i < ITER; ++i)  {    gemm_op();  }  cudaEventRecord(end);  cudaEventSynchronize(end);  cudaEventElapsedTime(&elapsedTime, start, end);  double workload = double(problem_size.product()) * 2.0 * double(ITER);  double avg_Gflops = (workload / 1e9) / (double(elapsedTime) / 1e3);  printf("Average Performance  %10.1lf Gflops\n", avg_Gflops);# nvcc -arch sm_86 00_basic_gemm.cu # ./a.out Average Performance     76279.8 Gflops
```

基本的性能达到CuBlas的85%的水平. 整个CUTLASS对底层的GMEM加载到SMEM以及如何调用TensorCore的细节进行了封装,同时针对BankConflict这些问题定义了`cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>`解决.

在CUTLASS 3.0中引入了`Spatial Micro-Kernel`和`Temporal Micro-Kernel`的概念, 非常不错的一个时空划分抽象

![图片](assets/9ee8fc4c8b56.png)

然后针对Tensor Layout也引入了Cute Layout代数

![图片](assets/8d7828a05414.png)

通过Layout代数对与Tensor的时空切分进行了统一的代数描述

![图片](assets/144774c3fd26.png)

我们下一篇将开始详细介绍介绍CuTe及其相关的代数表示.

### 4.2 整个测试代码

懒得commit到github了,直接贴

```
#include "cutlass/cutlass.h"#include "cutlass/gemm/device/gemm_universal.h"#include "cutlass/util/command_line.h"#include "cutlass/util/host_tensor.h"#include "cutlass/util/reference/device/gemm.h"#include "cutlass/util/reference/host/tensor_compare.h"#include "cutlass/util/reference/host/tensor_copy.h"#include "cutlass/util/reference/host/tensor_fill.h"#include "cutlass/util/tensor_view_io.h"//////////////////////////////////////////////////////////////////////////////////////////////////// GEMM kernel configurations (cutlass_tensorop_h16816gemm_128x128_32x4_nn_align8)/////////////////////////////////////////////////////////////////////////////////////////////////// A matrix configurationusing ElementA = cutlass::half_t;                                       // Element type for A matrix operandusing LayoutA = cutlass::layout::RowMajor;                              // Layout type for A matrix operandconstexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)// B matrix configurationusing ElementB = cutlass::half_t;                                       // Element type for B matrix operandusing LayoutB = cutlass::layout::RowMajor;                              // Layout type for B matrix operandconstexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)// C/D matrix configurationusing ElementC = cutlass::half_t;                                       // Element type for C and D matrix operandsusing LayoutC = cutlass::layout::RowMajor;                              // Layout type for C and D matrix operandsconstexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)// Multiply-accumulate blocking/pipelining detailsusing ElementAccumulator = cutlass::half_t;                      // Element type for internal accumulationusing ArchTag = cutlass::arch::Sm80;                             // Tag indicating the minimum SM that supports the intended featureusing OperatorClass = cutlass::arch::OpClassTensorOp;            // Operator class tagusing ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>; // Threadblock-level tile size (concept: GemmShape)using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;          // Warp-level tile size (concept: GemmShape)using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // Instruction-level tile size (concept: GemmShape)constexpr int NumStages = 4;                                     // Number of global->shared pipeline stages used in the GEMM mainloop// Epilogue output operatorusing EpilogueOp = cutlass::epilogue::thread::LinearCombination<    ElementC,            // Element type for C and D matrix operands    AlignmentC,          // Memory access granularity of C and D matrix in units of elements    ElementAccumulator,  // Element type from internal accumaccumulation    ElementAccumulator>; // Data type used to compute linear combination// Classic data-parallel device GEMM implementation typeusing DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<    ElementA, LayoutA,    ElementB, LayoutB,    ElementC, LayoutC,    ElementAccumulator,    OperatorClass,    ArchTag,    ThreadblockShape,    WarpShape,    InstructionShape,    EpilogueOp,    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,    NumStages,    AlignmentA,    AlignmentB>;/// Populates a DeviceGemmBasic::Arguments structure from the given commandline optionstypename DeviceGemmBasic::Arguments args_from_options(    const DeviceGemmBasic &device_gemm,    const cutlass::gemm::GemmCoord problem_size,    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,    cutlass::HostTensor<ElementC, LayoutC> &tensor_d){  return typename DeviceGemmBasic::Arguments(      cutlass::gemm::GemmUniversalMode::kGemm, // universal mode      problem_size,                            // problem_size      1,                                      // batch count / splitk slices      {          // epilogue parameters          ElementAccumulator(1.0f), // alpha          ElementAccumulator(0.0f)  // beta      },      tensor_a.device_data(),       // ptr_A      tensor_b.device_data(),       // ptr_B      tensor_c.device_data(),       // ptr_C      tensor_d.device_data(),       // ptr_D      problem_size.mk().product(),  // batch_stride_A      problem_size.nk().product(),  // batch_stride_B      problem_size.mn().product(),  // batch_stride_C      problem_size.mn().product(),  // batch_stride_D      tensor_a.layout().stride(0),  // stride_a      tensor_b.layout().stride(0),  // stride_b      tensor_c.layout().stride(0),  // stride_c      tensor_d.layout().stride(0)); // stride_d}int main(){  const int length_m = 4096;  const int length_n = 4096;  const int length_k = 4096;  // Create a tuple of problem size for matrix multiplication  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);  // Initialize tensors using CUTLASS helper functions  cutlass::HostTensor<ElementA, LayoutA> tensor_a(      problem_size.mk()); // <- Create matrix A with dimensions M x K  cutlass::HostTensor<ElementB, LayoutB> tensor_b(      problem_size.kn()); // <- Create matrix B with dimensions K x N  cutlass::HostTensor<ElementC, LayoutC> tensor_c(      problem_size.mn()); // <- Create matrix C with dimensions M x N  cutlass::HostTensor<ElementC, LayoutC> tensor_d(      problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from  // Fill input and output matrices on host using CUTLASS helper functions  cutlass::reference::host::TensorFillRandomUniform(      tensor_a.host_view(),      1,      ElementA(4),      ElementA(-4),      0); // <- Fill matrix A on host with uniform-distribution random data  cutlass::reference::host::TensorFillRandomUniform(      tensor_b.host_view(),      1,      ElementB(4),      ElementB(-4),      0); // <- Fill matrix B on host with uniform-distribution random data  cutlass::reference::host::TensorFillRandomUniform(      tensor_c.host_view(),      1,      ElementC(4),      ElementC(-4),      0); // <- Fill matrix C on host with uniform-distribution random data  cutlass::reference::host::TensorFill(      tensor_d.host_view()); // <- fill matrix D on host with zeros  // Copy data from host to GPU  tensor_a.sync_device();  tensor_b.sync_device();  tensor_c.sync_device();  tensor_d.sync_device();  // Instantiate CUTLASS kernel depending on templates  DeviceGemmBasic gemm_op;  auto arguments = args_from_options(gemm_op, problem_size, tensor_a, tensor_b, tensor_c, tensor_d);  // Using the arguments, query for extra workspace required for matrix multiplication computation  size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);  // Allocate workspace memory  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);  // Check the problem size is supported or not  gemm_op.can_implement(arguments);  // Initialize CUTLASS kernel with arguments and workspace pointer  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());  cudaEvent_t start, end;  float elapsedTime;  cudaEventCreate(&start);  cudaEventCreate(&end);  cudaEventRecord(start);  const int ITER = 100;  for (int i = 0; i < ITER; ++i)  {    gemm_op();  }  cudaEventRecord(end);  cudaEventSynchronize(end);  cudaEventElapsedTime(&elapsedTime, start, end);  double workload = double(problem_size.product()) * 2.0 * double(ITER);  double avg_Gflops = (workload / 1e9) / (double(elapsedTime) / 1e3);  printf("Average Performance  %10.1lf Gflops\n", avg_Gflops);}
```

参考资料

[1] 
半小时快速入门CUTLASS-基于 CUDA 的多层次稠密线性代数计算原语: https://www.bilibili.com/video/BV1Qk4y1n7Nd/
[2] 
CUTLASS: Software Primitives for Dense Linear Algebra at All Levels and Scales within CUDA: https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8854/