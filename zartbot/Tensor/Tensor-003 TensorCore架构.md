# Tensor-003  TensorCore架构

> 作者: zartbot  
> 日期: 2024年8月3日 08:38  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491424&idx=1&sn=0fc2110931b27714900e78d73b11a5b5&chksm=f9960fa2cee186b4d569cebcca2a4bbda37923bc404fd079010085e2d80faf97b290503859b6#rd

---

时间回到2016年, Google发布TPU后(已经在内部使用了1年多), 同期NV发布的Pascal架构被虐成狗了. 而Volta这一代的架构规划是在2013年, 应该有可能是在2015年附近得到了TPU的消息整个架构进行了修改, 直到2017年才发布,添加的第一代TensorCore也很匆忙, 也有不少问题.

TensorCore的演进如下:

| Arch | Dtype | TC per SM | TC m,n,k | 稀疏 | 访存 | 支持指令 |
|---|---|---|---|---|---|---|
| Volta(SM70) | FP16 | 8 | 4 x 4 x 4 | No | N/A | mma |
| Turing(SM75) | FP16,INT8,INT4,Binary | 8 | 4 x 4 x 4 | No | ldmatrix | mma,ldmatrix |
| Ampere(SM80) | FP16,BF16,TF32,FP64,INT8,INT4,Binary | 4 | 8 x 4 x 8 | Yes | async Copy | mma,ldmatrix, mma.sp |
| Hopper(SM90) | FP16,BF16,TF32,FP64,INT8,Binary | 4 | 8 x 4 x 16 | Yes | TMA | mma,ldmatrix, mma.sp ,wgmma |

可以看到从最早的FMA指令, 到向量化的DP4A,再到Volta(SM70)的第一代TensorCore,然后Ampere/Hopper都在提高矩阵乘法的规模, 提高计算访存比,同时支持更低精度的数据格式.
![图片](assets/ed338feb8f04.png)

时至今日的TensorCore在SIMT架构下依旧还有很多问题, 我们将在后续的文章中详细分析. 本文的目录如下:

```
1. TensorCore概述1.1 16x16x16矩阵乘法1.2 V100 TensorCore架构1.2.1 HMMA.884实现1.2.2 推测TensorCore架构1.2.3 数据加载2. TensorCore演进2.1 Turing第二代TensorCore2.1.1 HMMA.16882.1.2 LDMATRIX2.2 Ampere第三代TensorCore2.2.1 HMMA.168162.2.2 异步拷贝2.2.3 稀疏矩阵2.3 Hopper第四代TensorCore2.3.1 DSMEM2.3.2 TMA2.3.2.1 cp.async.bulk2.3.2.2 cp.reduce.async.bulk2.3.2.3 cp.async.bulk.prefetch2.3.2.4 基于Tensor的cp.async.buk2.3.2.5 TMA编程2.3.3 WGMMA
```

## 1. Tensor Core概述

对于一个 $M \times N \times K$ 的矩阵乘法, 计算量为 $C= 2\times M \times N \times K \sim \mathcal O(N^3)$, 访存量为 $D = M \times K + K \times N + 2 \times M \times N \sim \mathcal O(N^2)$, 计算访存比 $C/D \sim \mathcal O(N)$, 简化问题考虑 $M=N=K$ 的情况, 计算访存比为 $N/2$, 因此在数据存储和访问时的复用非常必要.

在一个Warp内, Thread计算时的效率还可以进一步并行提升, 特别是WarpLevel的寄存器文件复用上, 这就是Tensor Core诞生的原因. 第一代TensorCore在Volta架构出现

![图片](assets/eb1727895648.png)

正如前文所述, 它是应对Google TPU对原有SIMT架构打的一个补丁. 额外放置了一个类似于脉动阵列的矩阵乘法计算单元

![图片](assets/dee24e87a305.png)

在TensorCore中实现了4x4x4的矩阵乘法

![图片](assets/5f3a2c8de9dd.png)

TensorCore并不是什么新奇的东西, 还记得SGI实现的Geometry Engine么？在1980年的时候就提供了一系列的指令集，包括操作寄存器的LoadMM、MultMM、PushMM、PopMM、SotreMM来处理4x4矩阵运算. NV也是类似的提供了几条指令`ldmatrix`, `stmatrix`, `movmatrix`和`mma` 然后在Cuda中包装成了wmma的API.

![图片](assets/1ab33dd2e3b7.png)
通过`wmma::fragment`定义结构体, 然后`wmma::load_matrix`加载数据,再`wmma::mma_sync`进行矩阵乘法计算, 最后`wmma::store_matrix_sync`保存数据.

![图片](assets/d50c0d7dccc2.png)

需要注意的这些指令是Warp-Level的, 因此在Tensor-Core上需要进行Warp-Level的调度

![图片](assets/f9b6ca65a366.png)

### 1.1 16x16x16矩阵乘法

官方介绍的TensorCore是4x4矩阵乘法, 而暴露的API在CUDA C++层面看是16x16的整个Warp同步的运算, 对应的PTX指令为`wmma.mma.sync.aligned.{alayout}.{blayout}.m16n16k16`

![图片](assets/01b3c18380c7.png)

从Volta的TensorCore来看, 可以一个Cycle完成一次4×4 矩阵乘法加累加 (MACC,matrix-multiply-and-accumulation) 操作，即 D = A × B + C. 对于16x16的矩阵乘法可以拆分为16个4x4子矩阵的分块乘法, 累计需要64次TensorCore的MACC操作,在计算过程中采用同步阻塞的方式完成.

![图片](assets/7e9d02e15952.png)

实际执行的方式我们通过一段测试代码来进行分析

```
#include <cuda_fp16.h>#include <mma.h>using namespace nvcuda;__global__ void test_wmma(half  *C, half *A, half *B){        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_frag;        wmma::load_matrix_sync( a_frag, A, 16 );        wmma::load_matrix_sync( b_frag, B, 16 );        wmma::fill_fragment( acc_frag, 0.0f );                wmma::mma_sync( acc_frag, a_frag, b_frag, acc_frag );        wmma::store_matrix_sync( C, acc_frag, 16, wmma::mem_row_major );}
```

通过编译dump PTX指令可以看到调用的为一个m16n16k16的mma.sync乘法指令

```
nvcc -c -arch sm_70 --ptx  tmp2.cu .visible .entry _Z9test_wmmaP6__halfS0_S0_(	.param .u64 _Z9test_wmmaP6__halfS0_S0__param_0,	.param .u64 _Z9test_wmmaP6__halfS0_S0__param_1,	.param .u64 _Z9test_wmmaP6__halfS0_S0__param_2){	.reg .b16 	%rs<2>;	.reg .f32 	%f<2>;	.reg .b32 	%r<23>;	.reg .b64 	%rd<7>;	ld.param.u64 	%rd1, [_Z9test_wmmaP6__halfS0_S0__param_0];	ld.param.u64 	%rd2, [_Z9test_wmmaP6__halfS0_S0__param_1];	ld.param.u64 	%rd3, [_Z9test_wmmaP6__halfS0_S0__param_2];	cvta.to.global.u64 	%rd4, %rd3;	cvta.to.global.u64 	%rd5, %rd2;	mov.u32 	%r1, 16;	wmma.load.a.sync.aligned.row.m16n16k16.global.f16 	{%r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9}, [%rd5], %r1;	wmma.load.b.sync.aligned.col.m16n16k16.global.f16 	{%r10, %r11, %r12, %r13, %r14, %r15, %r16, %r17}, [%rd4], %r1;	mov.f32 	%f1, 0f00000000;	// begin inline asm	{  cvt.rn.f16.f32 %rs1, %f1;}	// end inline asm	mov.b32 	%r18, {%rs1, %rs1};	cvta.to.global.u64 	%rd6, %rd1;	wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 {%r19, %r20, %r21, %r22}, {%r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9}, {%r10, %r11, %r12, %r13, %r14, %r15, %r16, %r17}, {%r18, %r18, %r18, %r18};	wmma.store.d.sync.aligned.row.m16n16k16.global.f16 	[%rd6], {%r19, %r20, %r21, %r22}, %r1;	ret;}
```

查看SASS可以看到, 在V100上的实际的执行单元为m8n8k4(`HMMA.884`), 它将MMA指令分为4组(SET), 在Accumulator为FP16时每个SET为2个STEP, Accumulator为FP32时则需要4个STEP.

```
 nvcc -c -arch sm_70 tmp2.cu ; cuobjdump -sass tmp2.o | grep HMMA        /*01a0*/                   HMMA.884.F16.F16.STEP0 R20, R12.reuse.ROW, R16.reuse.COL, RZ ;   /* 0x000000100c147236 */        /*01b0*/                   HMMA.884.F16.F16.STEP1 R22, R12.ROW, R16.COL, RZ ;               /* 0x000000100c167236 */        /*01c0*/                   HMMA.884.F16.F16.STEP0 R12, R14.reuse.ROW, R18.reuse.COL, R20 ;  /* 0x000000120e0c7236 */        /*01d0*/                   HMMA.884.F16.F16.STEP1 R14, R14.ROW, R18.COL, R22 ;              /* 0x000000120e0e7236 */        /*01e0*/                   HMMA.884.F16.F16.STEP0 R12, R4.reuse.ROW, R8.reuse.COL, R12 ;    /* 0x00000008040c7236 */        /*01f0*/                   HMMA.884.F16.F16.STEP1 R14, R4.ROW, R8.COL, R14 ;                /* 0x00000008040e7236 */        /*0210*/                   HMMA.884.F16.F16.STEP0 R12, R6.reuse.ROW, R10.reuse.COL, R12 ;   /* 0x0000000a060c7236 */        /*0230*/                   HMMA.884.F16.F16.STEP1 R14, R6.ROW, R10.COL, R14 ;               /* 0x0000000a060e7236 */
```

而在较新的TensorCore中, Turing架构支持了`HMMA.1688`矩阵乘法, 而Ampere和Hopper支持了`HMMA.16816`乘法.

```
//Turingnvcc -c -arch sm_75 tmp2.cu ; cuobjdump -sass tmp2.o | grep HMMA        /*0120*/                   HMMA.1688.F16 R16, R8, R0, RZ ;                            /* 0x000000000810723c */        /*0130*/                   HMMA.1688.F16 R18, R8, R13, RZ ;                           /* 0x0000000d0812723c */        /*0150*/                   HMMA.1688.F16 R16, R10, R12, R16 ;                         /* 0x0000000c0a10723c */        /*0170*/                   HMMA.1688.F16 R18, R10, R14, R18 ;                         /* 0x0000000e0a12723c *///Amperenvcc -c -arch sm_80 tmp2.cu ; cuobjdump -sass tmp2.o | grep HMMA        /*0130*/                   HMMA.16816.F16 R12, R4.reuse, R12, RZ ;                    /* 0x0000000c040c723c */        /*0140*/                   HMMA.16816.F16 R14, R4, R14, RZ ;                          /* 0x0000000e040e723c *///Hoppernvcc -c -arch sm_90 tmp2.cu ; cuobjdump -sass tmp2.o | grep HMMA        /*0160*/                   HMMA.16816.F16 R12, R4, R12, RZ ;                /* 0x0000000c040c723c */        /*0170*/                   HMMA.16816.F16 R14, R4, R14, RZ ;                /* 0x0000000e040e723c */        
```

我们将逐代解析这些TensorCore的变化.

### 1.2 V100 TensorCore架构

在《Modeling Deep Learning Accelerator Enabled GPUs》[1]中对Volta系列TensorCore进行了一些详细的分析.另外还有一篇来自NV的论文《Automatic Kernel Generation for Volta Tensor Cores》[2], 在NV的PTX指令集[3]中也有详细的描述.

#### 1.2.1 HMMA.884实现

对于一个16x16矩阵,平均分配到一个WARP的32个线程中,每个线程需要8个寄存器存放. 一个WARP内的线程id通过`%laneid`寄存器表示,取值范围`[0,31]`. 在V100中每四个线程分为一组(ThreadGroup), 即0−3, 4−7, 8−11, 12−15, 16−19, 20−23, 24−27,28 − 31. 然后将四个一组的线程配对, 在NV的文档中将其称为`Quad Pair(QP)`, 例如QP0包含0-3,16-19, QP1包含4-7,20-23如下表所示:

| %laneid | QP | ThreadGroup |
|---|---|---|
| 0~3 | 0 | 0 |
| 4~7 | 1 | 1 |
| 8~11 | 2 | 2 |
| 12~15 | 3 | 3 |
| 16~19 | 0 | 4 |
| 20~23 | 1 | 5 |
| 24~27 | 2 | 6 |
| 28~31 | 3 | 7 |

HMMA.884指令在QP上加载, 如下所示

![图片](assets/c4532c0240c0.png)

单个Warp-Level实际上执行了4个QP的HMMA.884. 然后利用外积求矩阵乘法的方式, 叠加4个Warp-Level的 HMMA.884(SET0~SET3)即可完成16x16x16的运算

![图片](assets/50dfbed5dd63.png)

我们再来看具体的HMMA.884乘法, 以QP0为例, 如下图所示:

![图片](assets/ccdd13b424a6.png)

在计算时矩阵需要访问两次, 如下图所示

![图片](assets/0c89f317ce42.png)

因此HMMA.884分为两个STEP完成, 最终16x16x16的矩阵乘法被分解为4个SET,每个SET包含2个STEP

![图片](assets/c6cec111aebb.png)

需要注意的是, 如果Accumulator矩阵C/D为FP32时, 需要占用更多的寄存器, 因此需要4个STEP

![图片](assets/d5266e8bfe41.png)

对应的SASS

![图片](assets/04904b04ffbe.png)

#### 1.2.2 推测TensorCore架构

推测的矩阵乘法单元架构如下所示:

![图片](assets/a684ad4a860f.png)

一个SubCore内包含两个TensorCore, 每个负责两个Quad Pair, 在一个Quad Pair内包含两个Thread Group, 每个Thread Group内有四个FEDP(Four Elements Dot Product)单元.具体供数逻辑在论文《Modeling Deep Learning Accelerator Enabled GPUs》中有一张图.
![图片](assets/e447d0a0e7a9.png)

其中Octet的定义和Quad-Pairs是等价的, ThreadGroup0(Laneid 0~3)和ThreadGroup4(Laneid 16~19)分别提供各自的A矩阵操作数, B矩阵则通过Mux然后同时注入到乘法器中实现`数据复用`. C矩阵在Operand Bus3注入并完成加法.

#### 1.2.3 数据加载

对于矩阵乘法的操作数`A`和`B`都是加载为4x4的矩阵. 引用《Modeling Deep Learning Accelerator Enabled GPUs》如下:
![图片](assets/3be86fde16e8.png)

如上图中的❷我们在代码中定义A和B的Row-major和Col-major, 采用连续的两个LDG.E.128加载

![图片](assets/33214afd3c21.png)

```
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;        nvcc -c -arch sm_70 tmp2.cu ; cuobjdump -sass tmp2.o | grep LDG        /*0130*/                   LDG.E.128.SYS R12, [R24] ;                                       /* 0x00000000180c7381 */        /*0140*/                   LDG.E.128.SYS R16, [R26] ;                                       /* 0x000000001a107381 */        /*0150*/                   LDG.E.128.SYS R4, [R24+0x10] ;                                   /* 0x0000100018047381 */        /*0160*/                   LDG.E.128.SYS R8, [R26+0x10] ;                                   /* 0x000010001a087381 */
```

当把A和B改为Col-major和Row-major后, 如上图中的 ❸ 采用4个LDG.E.64加载

![图片](assets/b0344eb24fce.png)

```
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> a_frag;        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;        nvcc -c -arch sm_70 tmp2.cu ; cuobjdump -sass tmp2.o | grep LDG        /*0160*/                   LDG.E.64.SYS R22, [R24] ;                                      /* 0x0000000018167381 */        /*0170*/                   LDG.E.64.SYS R6, [R26] ;                                       /* 0x000000001a067381 */        /*0180*/                   LDG.E.64.SYS R16, [R26+0x80] ;                                 /* 0x000080001a107381 */        /*0190*/                   LDG.E.64.SYS R18, [R24+0x80] ;                                 /* 0x0000800018127381 */        /*01a0*/                   LDG.E.64.SYS R10, [R26+0x100] ;                                /* 0x000100001a0a7381 */        /*01b0*/                   LDG.E.64.SYS R12, [R24+0x100] ;                                /* 0x00010000180c7381 */        /*01c0*/                   LDG.E.64.SYS R2, [R26+0x180] ;                                 /* 0x000180001a027381 */        /*01d0*/                   LDG.E.64.SYS R8, [R24+0x180] ;                                 /* 0x0001800018087381 */
```

## 2. TensorCore演进

对已有的TensorCore(TC)四代进行了整理,如下所示:

| Arch | Dtype | TC per SM | TC m,n,k | 稀疏 | 访存 | 支持指令 |
|---|---|---|---|---|---|---|
| Volta(SM70) | FP16 | 8 | 4 x 4 x 4 | No | N/A | mma |
| Turing(SM75) | FP16,INT8,INT4,Binary | 8 | 4 x 4 x 4 | No | ldmatrix | mma,ldmatrix |
| Ampere(SM80) | FP16,BF16,TF32,FP64,INT8,INT4,Binary | 4 | 8 x 4 x 8 | Yes | async Copy | mma,ldmatrix, mma.sp |
| Hopper(SM90) | FP16,BF16,FP8,TF32,FP64,INT8,Binary | 4 | 8 x 4 x 16 | Yes | TMA | mma,ldmatrix, mma.sp ,wgmma |

TensorCore的四代演进主要包含几个方向:

更低精度的数据格式

更大的计算规模

支持稀疏矩阵

逐渐异步化, 提高TC的供数吞吐

### 2.1 Turing第二代TensorCore

在Turing系列的第二代TensorCore中,主要做了一下几个优化:

支持INT8/INT4/Binary这几种格式

Warp-Level支持`HMMA.1688`

访问内存支持`ldmatrix`指令

Turing SM架构如下图所示, 一个SM内包含4个SubCore, 每个包含2个TensorCore

![图片](assets/026051f4ed15.png)

#### 2.1.1 HMMA.1688

`wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16`在Turing上被展开为四条HMMA.1688指令

```
nvcc -c -arch sm_75 tmp2.cu ; cuobjdump -sass tmp2.o | grep HMMA        /*0120*/                   HMMA.1688.F16 R16, R8, R0, RZ ;                            /* 0x000000000810723c */        /*0130*/                   HMMA.1688.F16 R18, R8, R13, RZ ;                           /* 0x0000000d0812723c */        /*0150*/                   HMMA.1688.F16 R16, R10, R12, R16 ;                         /* 0x0000000c0a10723c */        /*0170*/                   HMMA.1688.F16 R18, R10, R14, R18 ;                         /* 0x0000000e0a12723c */
```

具体执行如下图所示:

![图片](assets/512f86c40d1c.png)

#### 2.1.2 LDMATRIX

传统的LDS指令在单个Thread内执行, 因此只能写到线程内的寄存器. 另一方面SMEM以32bits为单位访问, 直接采用LDS.b16会导致每个thread浪费16b,因此在矩阵转置加载的情况下需要两条LDS.b16并且浪费2个16b的数据. 32个线程并行LD的时候从整个WARP来看指令也很多, 同时在LD时我们需要考虑Bank Conflict的情况.

因此从Turing这一代开始,增加了WARP-Level的PTX指令LDMATRIX

```
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];.shape  = {.m8n8};.num    = {.x1, .x2, .x4};.ss     = {.shared};.type   = {.b16};
```

它仅支持从SMEM(`.shared`)中加载元素为16bits(`.b16`)的8x8(`.m8n8`)的矩阵到寄存器中. `.num`表示需要加载的8x8矩阵数量, 支持`x1,x2,x4`.`.trans`表示是否需要转置,`[p]`表示SHMEM的地址指针, 而`r`表示加载的目的寄存器.

对于常用的组合, 我们定义如下6个宏

```
#define LDMATRIX_X1(R, addr) \    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))#define LDMATRIX_X2(R0, R1, addr) \    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \                 : "r"(addr))#define LDMATRIX_X1T(R, addr) \    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))#define LDMATRIX_X2T(R0, R1, addr) \    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))#define LDMATRIX_X4T(R0, R1, R2, R3, addr)                                                  \    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                   \                 : "r"(addr))
```

`LDMATRIX_X1`表示加载一个8x8 b16矩阵, 然后刚好放置到32个线程的`R0`寄存器中(每个寄存器32bits,正好存放2个16bits),该指令按照连续128bits作为一行读取, 累计需要读取8行, 每一行的起始地址由前8个线程(laneid = 0~7)的寄存器addr提供.  `LDMATRIX_X2` 表示加载2个8x8矩阵, 加载的16行的起始地址由前16个线程(laneid = 0~16)的寄存器addr提供.`LDMATRIX_X4`表示加载4个8x8矩阵, 累计32行, 如下表示所:

![图片](assets/903921ac8f39.png)

![图片](assets/ab36e4dd82d0.png)

在NV的一些资料里也有一张很清楚的图

![图片](assets/78e7ce4d15c4.png)

`LDMATRIX_X1T`,`LDMATRIX_X2T`,`LDMATRIX_X4T`为读取后将8x8矩阵转置存放到寄存器的加载指令.

例如我们构建如下测试程序:

```
#include <stdio.h>#include <stdint.h>#include "cuda_fp16.h"__global__ void TestLDMatrix(void){    const int tid = threadIdx.x;    //在共享内存内构建一个4x16x16的矩阵    __shared__ uint16_t M[4 * 16 * 16];    if (tid == 0)    {        int offset = 0;        for (int i = 0; i < 4; ++i){            for (int j = 0; j < 16; ++j){                for (int k = 0; k < 16; ++k)                {                    M[offset] = static_cast<uint16_t>((i+1) * 10000 + (j+1) * 100 + k+1);                    printf(" %6d",M[offset]);                    offset++;                }                printf("\n");            }             printf("\n");        }    }    __syncthreads();        int offset = tid * 16;    uint32_t addr = __cvta_generic_to_shared(M + offset);    uint32_t frag[4];    //LDMATRIX_X1(frag[0],addr);    LDMATRIX_X4(frag[0], frag[1], frag[2], frag[3], addr);    uint16_t data[4][2];    for (int i = 0; i < 4; ++i)    {        data[i][0] = static_cast<uint16_t>(frag[i] & 0xFFFF);        data[i][1] = static_cast<uint16_t>((frag[i] >> 16) & 0xFFFF);    }    printf("OFFSET %4d  tid: %3d | A | %6d %6d | %6d %6d | %6d %6d | %6d %6d |\n", offset, tid,           int(data[0][0]), int(data[0][1]), int(data[1][0]), int(data[1][1]),           int(data[2][0]), int(data[2][1]), int(data[3][0]), int(data[3][1]));}int main(void){    dim3 gridDim(1, 1, 1);    dim3 blockDim(32, 1, 1);    TestLDMatrix<<<gridDim, blockDim>>>();        cudaDeviceReset();    return 0;}
```

编译需要Arch大于Turing(SM_75), 查看SASS指令为LDSM如下所示

```
nvcc -c -arch sm_75 ldmatrix.cu ; cuobjdump -sass ldmatrix.o | grep LDSM        /*1470*/                   LDSM.16.M88.4 R4, [R3] ;                   /* 0x000000000304783b */
```

共享内存的矩阵为4x16x16, 如下所示:

![图片](assets/0938f45d1972.png)

offset = tid * 16 ,正好stride为一行的长度,输出结果如下

![图片](assets/dadcfa67511c.png)

offset = tid * 32 ,则针对原始矩阵M每隔一行读取,输出结果如下

![图片](assets/dec57f823ea2.png)

当我们把指令换成带转置读取的版本,例如`LDMATRIX_X4T`输出如下:

![图片](assets/1cbc63e7d0cf.png)

### 2.2 Ampere第三代TensorCore

第三代TensorCore的改进幅度更大, 在一个SubCore内整合成单个TensorCore,直接在WARP内32个线程内共享, 因此单个SM TensorCore的数量降低为4个.

![图片](assets/381990df3733.png)

单个Cycle支持8x4x8的矩阵乘法, 取消了Quad Pair和ThreadGroup的概念. 另一方面支持HMMA.16816, 对于16x16x16的矩阵乘法只需要两条指令即可, 另外还支持了稀疏矩阵乘法和异步内存拷贝等特性.

![图片](assets/95c9ce1292a2.png)

另一方面支持了Warp-Level的Reduction, L2Cache的有效管理, 异步内存拷贝和异步Barrier等新功能

![图片](assets/2d3859570c76.png)

#### 2.2.1 HMMA.16816

进一步加大矩阵乘法规模

![图片](assets/f747046bce00.png)

```
nvcc -c -arch sm_80 tmp2.cu ; cuobjdump -sass tmp2.o | grep HMMA        /*0130*/                   HMMA.16816.F16 R12, R4.reuse, R12, RZ ;                    /* 0x0000000c040c723c */        /*0140*/                   HMMA.16816.F16 R14, R4, R14, RZ ;                          /* 0x0000000e040e723c */
```

因此整个矩阵只需要两条指令即可

![图片](assets/c247e6759ad2.png)

TensorCore的运算采用整个WARP-LEVEL执行, 因此不需要原有的ThreadGroup和QuadPair的分块, 整个SubCore的TensorCore也完全合并成一个更大的TensorCore,支持8x4x8的矩阵运算

![图片](assets/903ebf0bda72.png)

#### 2.2.2 异步拷贝

伴随着A100发布的CUDA 11.0最大的更新就是对异步编程的支持. 通过Barrier Object替代了原有的__syncthreads

![图片](assets/03c8e07f904f.png)

如上左图所示, 我们来看一个最基本的向量化的拷贝过程, 也就是说数据通过GMEM到了L1再到寄存器, 最后存放到SMEM

```
__global__ void testcopy(float *x, int N) {    int tid = threadIdx.x;    __shared__ float Tile[32];    *reinterpret_cast<float4*>(&Tile[tid]) = *reinterpret_cast<float4*>(&x[tid*4]);    printf("%f ", Tile[tid]);}# nvcc -arch sm_80 -ptx async_cp.cu    ld.global.v4.u32  {%r5, %r6, %r7, %r8}, [%rd6]; st.shared.v4.u32  [%r4], {%r5, %r6, %r7, %r8};    # nvcc -arch sm_80 -c async_cp.cu ; cuobjdump -sass async_cp.o | grep 128        /*0060*/                   LDG.E.128 R8, [R8.64] ;                  /* 0x0000000408087981 */        /*00e0*/                   STS.128 [R0.X4], R8 ;                    /* 0x0000000800007388 */
```

通过异步内存拷贝,可以将GMEM数据bypass L1/RF直接拷贝到SMEM中, 使得对tensorCore的供数效率更高

![图片](assets/89a1c806567a.png)

异步拷贝指令如下

```
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}                         [dst], [src], cp-size{, src-size}{, cache-policy} ;cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}                         [dst], [src], 16{, src-size}{, cache-policy} ;cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}                         [dst], [src], cp-size{, ignore-src}{, cache-policy} ;cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}                         [dst], [src], 16{, ignore-src}{, cache-policy} ;.level::cache_hint =     { .L2::cache_hint }.level::prefetch_size =  { .L2::64B, .L2::128B, .L2::256B }cp-size =                { 4, 8, 16 }
```

`cp-size`:是一个整形常量, 指定了要复制到目标`dst`的数据大小,只能取值为4/8/16.

`ignore-sorce`:是一个可选的谓词参数, 指定是否应该完全忽略src中的数据, 如果被设置,则直接将零填充到`dst`

`cg` | `ca`: Cache操作符, `cg`表示仅在L2中缓存, `ca`表示需要在包含L1的所有缓存层次中缓存

`prefetch_size`:可以定义在L2中的Prefetch数量和
![图片](assets/68947c9f9a28.png)

异步拷贝代码如下

```
__global__ void testcopy2(float *x, int N) {    int tid = threadIdx.x;    __shared__ float Tile[32];    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"                :: "r"((uint32_t)__cvta_generic_to_shared(&Tile[tid])),                "l"(&x[tid]),                "n"(16)            );    printf("%f ", Tile[tid]);}# nvcc -arch sm_80 -ptx async_cp.cu  cp.async.cg.shared.global [%r1], [%rd1], 16;# nvcc -arch sm_80 -c async_cp.cu ; cuobjdump -sass async_cp.o | grep LDGSTS        /*00d0*/                   LDGSTS.E.BYPASS.128 [R0], [R2.64] ;      /* 0x0000000002007fae */
```

可以看到新增加了一条LDGSTS(Load GMEM, Store SMEM)的指令, bypass了L1/RF. 对于异步执行, 我们可以通过cp.async.wait_group 或者async mbarrier指令来处理状态. 例如

```
// Example of .wait_all:cp.async.ca.shared.global [shrd1], [gbl1], 4;cp.async.cg.shared.global [shrd2], [gbl2], 16;cp.async.wait_all;  // waits for all prior cp.async to complete// Example of .wait_group :cp.async.ca.shared.global [shrd3], [gbl3], 8;cp.async.commit_group;  // End of group 1cp.async.cg.shared.global [shrd4], [gbl4], 16;cp.async.commit_group;  // End of group 2cp.async.cg.shared.global [shrd5], [gbl5], 16;cp.async.commit_group;  // End of group 3cp.async.wait_group 1;  // waits for group 1 and group 2 to complete
```

使用CUDA的接口通过barrier实现如下所示:

```
#include <stdio.h>#include <stdint.h>#include <cuda/barrier>#include <cooperative_groups.h>#include <cooperative_groups/memcpy_async.h>namespace cg = cooperative_groups;__global__ void testcopy2(float *global1, float *global2, int subset_count){    extern __shared__ float shared[];    auto group = cooperative_groups::this_thread_block();    // Create a synchronization object     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;    if (group.thread_rank() == 0)    {        init(&barrier, group.size());    }    group.sync();    for (size_t subset = 0; subset < subset_count; ++subset)    {        cuda::memcpy_async(group, shared,                           &global1[subset * group.size()], sizeof(float) * group.size(), barrier);        cuda::memcpy_async(group, shared + group.size(),                           &global2[subset * group.size()], sizeof(float) * group.size(), barrier);        barrier.arrive_and_wait(); // Wait for all copies to complete        compute(shared);        barrier.arrive_and_wait();    }}# nvcc -c -arch sm_80 async_cp.cu ; cuobjdump -sass async_cp.o | grep LDGSTS        /*02f0*/                   LDGSTS.E [R3+0x10], [R6.64] ;                      /* 0x0001000006037fae */        /*0400*/                   ARRIVES.LDGSTSBAR.64 [URZ] ;                       /* 0x00000000ff0079b0 */        /*04f0*/                   LDGSTS.E [R3+0x10], [R4.64] ;                      /* 0x0001000004037fae */        /*05d0*/                   ARRIVES.LDGSTSBAR.64 [URZ] ;                       /* 0x00000000ff0079b0 */
```

但我们需要考虑一个问题就是实际上执行为Weak Order, 例如如下两个async copy使用相同地址的不同长度内存拷贝, 执行带来的结果是未定义的.

```
  asm volatile(    "{\n"    "cp.async.cg.shared.global [%0], [%1], %2, 8;\n"     "cp.async.cg.shared.global [%0], [%1], %2, 16;\n"    "cp.async.commit_group;\n"    "cp.async.wait_group 0\n;"    "}\n" :: "r"(smem), "l"(&x[tid]), "n"(16)  );
```

异步拷贝的详细内容可以参考《Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture》[4]

#### 2.2.3 稀疏矩阵

新增mma.sp指令, 通过在dot product Engine上支持select bitmap实现稀疏矩阵计算.

![图片](assets/288e245cb880.png)

### 2.3 Hopper第四代TensorCore

主要变化是进一步提升了TensorCore的规格, 支持16x8x4的矩阵乘法. 同时支持了FP8格式.

![图片](assets/35adb21cb0b5.png)

在访问内存时提供了Tensor Memory Accelerator (TMA)进行访问内存优化.  同时还支持了基于一个SM-Level的WARP-Group MMA. 针对异步编程模式, 多个Kernel之间以生产者、消费者的方式通信， SM到SM之间的通信带宽需求也在增加, 因此提供了Distribute Shared Memory (DSMEM)的概念

![图片](assets/12ad7e639ba8.png)

#### 2.3.1 DSMEM

在Hopper以前，CUDA处理问题的规模采用Grid、Block两级调度，Block映射到SM上，但是随着协作组Cooperative Groups的出现和异步编程的支持，多个Kernel之间以生产者、消费者的方式通信， SM到SM之间的通信带宽需求也在增加

![图片](assets/f303f4c83bad.png)

而在Hopper上新增了一个Distribute Shared Memory (DSMEM)的概念，在一个GPC内部的SM有了专用的通信带宽，因此CUDA上增加了一级调度层次：

![图片](assets/0defeaf691fd.png)

![图片](assets/0bb389f479ba.png)

同一个Thread Block内采用PGAS构成分布式共享内存(DSMEM),在一个GPC内部实现多个SM的LD/ST，Atomic，reduce和异步DMA操作都变得非常的简洁，

![图片](assets/14ceee528108.png)

#### 2.3.2 TMA

在A100上虽然可以通过Async.copy的LDGSTS指令 bypass L1 直接从GMEM拷贝到SMEM, 但是还需要CudaCore来计算地址,并发起指令. 而在Hopper这一代提供了Tensor Memory Accelerator (TMA)引擎, 支持1D~5D Tensor的 LD/ST.

![图片](assets/c632d1c43a2d.png)

例如在没有TMA的情况下, Triton GEMM的吞吐只能到910GB/s, LSU上有227K指令

![图片](assets/fceec6140dd5.png)

支持TMA后, GEMM吞吐到了1.45TB/s, TMA引擎指令数降低了10倍,只有19K.

![图片](assets/e5c5fef20fc2.png)

TMA不光支持GMEM到SMEM的拷贝,还支持GPC内SM-to-SM的SMEM拷贝

![图片](assets/7554b46161c9.png)

同时针对1D~5D张量, 可以定义特定的BLOCK进行异步的数据加载和存储

![图片](assets/d76520b5abb6.png)

针对矩阵分块乘法需要持续性的以Block Tile为单位传输, 在Hopper上针对TMA的处理新增了ATB(Async Transaction Barrier)的能力, 如下图所示:

![图片](assets/25837f37135c.png)

因此我们就可以构建更加异步的计算模式Overlap计算和访存:

![图片](assets/f8e5ff8e589d.png)

TMA增加的PTX指令如下:

cp.async.bulk

cp.reduce.async.bulk

cp.async.bulk.prefetch

cp.async.bulk.tensor

cp.reduce.async.bulk.tensor

cp.async.bulk.prefetch.tensor

cp.async.bulk.commit_group

cp.async.bulk.wait_group

tensormap.replace
2.3.2.1 cp.async.bulk
cp.async.bulk是一个非阻塞的批量异步内存拷贝指令, 支持mbarrier和bulk_group两种完成机制, 同时还支持组播能力.

```
//组播cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint}                      [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}.dst =                  { .shared::cluster }.src =                  { .global }.completion_mechanism = { .mbarrier::complete_tx::bytes }.level::cache_hint =    { .L2::cache_hint }.multicast =            { .multicast::cluster  }//mbarrier完成机制cp.async.bulk.dst.src.completion_mechanism [dstMem], [srcMem], size, [mbar].dst =                  { .shared::cluster }.src =                  { .shared::cta }.completion_mechanism = { .mbarrier::complete_tx::bytes }//bulk_group完成机制cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint} [dstMem], [srcMem], size {, cache-policy}.dst =                  { .global }.src =                  { .shared::cta }.completion_mechanism = { .bulk_group }.level::cache_hint =    { .L2::cache_hint }
```

但我们注意到在组播指令中, 源操作数为GMEM, 目的为SMEM. 即在一个GPC内, TMA可以通过从GMEM读取一份, 然后通过DSM在SM-to-SM网络上多写到其它SM内, 降低L2Cache的压力.
而对于通知完成机制, 我们可以看到从GMEM到SMEM以及DSM拷贝时采用mbarrier机制, 而从SMEM到GMEM时采用bulk async group机制.

![图片](assets/af226a78ca74.png)

一些常见的例子

```
// GMEM -> SMEMcp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];// GMEM -> SMEM(Multicast)cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster                                             [dstMem], [srcMem], size, [mbar], ctaMask;// GMEM -> SMEM  with L2Cache Hintcp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint                                             [dstMem], [srcMem], size, [mbar], cache-policy;// SMEM -> Distributed SMEMcp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];// SMEM -> GMEMcp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size;cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint} [dstMem], [srcMem], size, cache-policy;
```
2.3.2.2 cp.reduce.async.bulk
在Ampere架构中增加过一个Warp-Level的针对整型的Reduce算子, 而这里TMA上扩展了矩阵计算中常见的规约(Reduce)操作, 支持多种数据类型(FP16/BF16)和算子, 但仅支持GMEM到SMEM方向或者Cluster内通过DSM的内存拷贝规约, 猜测计算能力复用了CudaCore内的ALU资源, 降低了L1Cache的开销, 直接在RF上进行了计算.

```
cp.reduce.async.bulk.dst.src.completion_mechanism.redOp.type              [dstMem], [srcMem], size, [mbar].dst =                  { .shared::cluster }.src =                  { .shared::cta }.completion_mechanism = { .mbarrier::complete_tx::bytes }.redOp=                 { .and, .or, .xor,                          .add, .inc, .dec,                          .min, .max }.type =                 { .b32, .u32, .s32, .b64, .u64 }cp.reduce.async.bulk.dst.src.completion_mechanism{.level::cache_hint}.redOp.type               [dstMem], [srcMem], size{, cache-policy}.dst =                  { .global      }.src =                  { .shared::cta }.completion_mechanism = { .bulk_group }.level::cache_hint    = { .L2::cache_hint }.redOp=                 { .and, .or, .xor,                          .add, .inc, .dec,                          .min, .max }.type =                 { .f16, .bf16, .b32, .u32, .s32, .b64, .u64, .s64, .f32, .f64 }cp.reduce.async.bulk.dst.src.completion_mechanism{.level::cache_hint}.add.noftz.type               [dstMem], [srcMem], size{, cache-policy}.dst  =                 { .global }.src  =                 { .shared::cta }.completion_mechanism = { .bulk_group }.type =                 { .f16, .bf16 }
```
2.3.2.3 cp.async.bulk.prefetch
提供从HBM到L2的预取能力,增加L2Cache的命中率

```
cp.async.bulk.prefetch.L2.src{.level::cache_hint}   [srcMem], size {, cache-policy}.src =                { .global }.level::cache_hint =  { .L2::cache_hint }
```
2.3.2.4 基于Tensor的cp.async.buk
针对高维矩阵, 我们可以定义一个TensorMap描述符, 更详细的内容,我们将在以后的CuTe Layout相关的章节讲述

```
  // Create the tensor descriptor.  CUresult res = cuTensorMapEncodeTiled(    &tensor_map,                // CUtensorMap *tensorMap,    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,    rank,                       // cuuint32_t tensorRank,    tensor_ptr,                 // void *globalAddress,    size,                       // const cuuint64_t *globalDim,    stride,                     // const cuuint64_t *globalStrides,    box_size,                   // const cuuint32_t *boxDim,    elem_stride,                // const cuuint32_t *elementStrides,    // Interleave patterns can be used to accelerate loading of values that    // are less than 4 bytes long.    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,    // Swizzling can be used to avoid shared memory bank conflicts.    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,    // L2 Promotion can be used to widen the effect of a cache-policy to a wider    // set of L2 cache lines.    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,    // Any element that is outside of bounds will be set to zero by the TMA transfer.    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE    );
```

通过这个描述符和坐标,我们就可以在Tensor中取出相应的Block进行计算,避免对实际地址进行Offset/Stride计算带来的复杂性.

![图片](assets/f5854aebd37d.png)

因此在Tensor相关的TMA指令中, 操作数需要包含`[tensorMap, tensorCoords]` 该指令也支持GMEM到SMEM的拷贝,并支持通过DSM组播的能力. 而同时也支持从SMEM到GMEM的写回能力

```
// global -> shared::cluster:cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.level::cache_hint}                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colOffsets}                                   {, ctaMask} {, cache-policy}.dst =                  { .shared::cluster }.src =                  { .global }.dim =                  { .1d, .2d, .3d, .4d, .5d }.completion_mechanism = { .mbarrier::complete_tx::bytes }.load_mode =            { .tile, .im2col }.level::cache_hint =    { .L2::cache_hint }.multicast =            { .multicast::cluster  }// shared::cta -> global:cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}                                   [tensorMap, tensorCoords], [srcMem] {, cache-policy}.dst =                  { .global }.src =                  { .shared::cta }.dim =                  { .1d, .2d, .3d, .4d, .5d }.completion_mechanism = { .bulk_group }.load_mode =            { .tile, .im2col_no_offs }.level::cache_hint =    { .L2::cache_hint }
```

同时针对卷积运算, 还支持im2col的能力

![图片](assets/c3c3896680da.png)

TMA也同时支持了基于Tensor的DSM规约写到GMEM的能力

```
// shared::cta -> global:cp.reduce.async.bulk.tensor.dim.dst.src.redOp{.load_mode}.completion_mechanism{.level::cache_hint}                                          [tensorMap, tensorCoords], [srcMem] {,cache-policy}.dst =                  { .global }.src =                  { .shared::cta }.dim =                  { .1d, .2d, .3d, .4d, .5d }.completion_mechanism = { .bulk_group }.load_mode =            { .tile, .im2col_no_offs }.redOp =                { .add, .min, .max, .inc, .dec, .and, .or, .xor}
```

以及在GMEM中基于Tensor预取的能力

```
// global -> shared::cluster:cp.async.bulk.prefetch.tensor.dim.L2.src{.load_mode}{.level::cache_hint} [tensorMap, tensorCoords]                                                             {, im2colOffsets } {, cache-policy}.src =                { .global }.dim =                { .1d, .2d, .3d, .4d, .5d }.load_mode =          { .tile, .im2col }.level::cache_hint =  { .L2::cache_hint }
```
2.3.2.5 TMA编程
在CUDA_Samples有一个PR: Add TMA example for Hopper H100 #214[5]作为例子

`1. 初始化cuTensorMap`准备GMEM的数据

```
 std::vector<int> tensor_host(H_global * W_global);  for (int i = 0; i < H_global * W_global; ++i) {    tensor_host[i] = i;  }  // Move it to device  int * tensor = nullptr;  CUDA_CHECK(cudaMalloc(&tensor, H_global * W_global * sizeof(int)));  CUDA_CHECK(cudaMemcpy(tensor, tensor_host.data(), H_global * W_global * sizeof(int), cudaMemcpyHostToDevice));
```

初始化CUtensorMap时指定GMEM的指针`*tensor`和相应的Shape, Stride等参数.

```
  CUtensorMap tma_desc{};  CUtensorMapDataType dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32;  auto rank = 2;  uint64_t size[rank] = {W_global, H_global};  uint64_t stride[rank - 1] = {W_global * sizeof(int)};  uint32_t box_size[rank] = {SMEM_W, SMEM_H};  uint32_t elem_stride[rank] = {1, 1};  CUtensorMapInterleave interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;  CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;  CUtensorMapL2promotion l2_promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;  CUtensorMapFloatOOBfill oob_fill = CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;  // Create the tensor descriptor.  CUresult res = cuTensorMapEncodeTiled(      &tma_desc,    // CUtensorMap *tensorMap,      dtype,        // CUtensorMapDataType tensorDataType,      rank,         // cuuint32_t tensorRank,      tensor,       // void *globalAddress,      size,         // const cuuint64_t *globalDim,      stride,       // const cuuint64_t *globalStrides,      box_size,     // const cuuint32_t *boxDim,      elem_stride,  // const cuuint32_t *elementStrides,      interleave,   // CUtensorMapInterleave interleave,      swizzle,      // CUtensorMapSwizzle swizzle,      l2_promotion, // CUtensorMapL2promotion l2Promotion,      oob_fill      // CUtensorMapFloatOOBfill oobFill);    );
```

`2. 构建TMA操作函数`

```
inline __device__ void cp_async_bulk_tensor_2d(  __mbarrier_t *barrier, void *dst, int access_coord_x, int access_coord_y, const CUtensorMap *tensor_desc){  unsigned smem_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(dst));  unsigned smem_barrier_int_ptr = static_cast<unsigned int>(__cvta_generic_to_shared(barrier));  uint64_t tensor_desc_ptr = reinterpret_cast<uint64_t>(tensor_desc);  asm volatile(    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "    "[%0], [%1, {%2, %3}], [%4];\n"    :    : "r"(smem_int_ptr),      "l"(tensor_desc_ptr),      "r"(access_coord_x),      "r"(access_coord_y),      "r"(smem_barrier_int_ptr)    : "memory");}
```

`3. mbarrier`

```
inline __device__ __mbarrier_token_t barrier_arrive1_tx(__mbarrier_t *barrier, uint32_t expected_tx_count ){  __mbarrier_token_t token;  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"               : "=l"(token)               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(expected_tx_count)               : "memory");  return token;}inline __device__ bool barrier_try_wait_token(__mbarrier_t *barrier, __mbarrier_token_t token){  int __ready;  asm volatile("{\n\t"               ".reg .pred p;\n\t"               "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"               "selp.b32 %0, 1, 0, p;\n\t"               "}"               : "=r"(__ready)               : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))),                 "l"(token)               : "memory");  return __ready;}
```

`4. Kernel函数`

```
template <int H, int W>struct smem_t {  //TMA需要地址128B对齐  struct alignas(128) tensor_buffer {    int data[H][W];    __device__ constexpr int width() {return W;}    __device__ constexpr int height() {return H;}  };  tensor_buffer buffer;  // Put the barrier behind the tensor buffer to prevent 100+ bytes of padding.  __mbarrier_t bar;  __device__ constexpr int buffer_size_in_bytes() {    return sizeof(tensor_buffer::data);  }};__global__ void kernel(const __grid_constant__ CUtensorMap tma_desc, int x_0, int y_0) {  // 申明共享内存  __shared__ smem_t<SMEM_H, SMEM_W> smem;  bool leader = threadIdx.x == 0;  if (leader) {    //初始化barrier    __mbarrier_init(&smem.bar, blockDim.x);  }  __syncthreads();  __mbarrier_token_t token;  // 加载第一个Batch  if (leader) {    // Initiate bulk tensor copy.    cp_async_bulk_tensor_2d(&smem.bar, &smem.buffer.data, x_0, y_0, &tma_desc);    // barrier预期cp_async_bulk_tensor_2d拷贝的数据    token = barrier_arrive1_tx(&smem.bar, smem.buffer_size_in_bytes());  } else {    // 其它thread的tx为0.    token = barrier_arrive1_tx(&smem.bar, 0);  }  while(! barrier_try_wait_token(&smem.bar, token)) { };  if (leader) {    printf("\n\nPrinting tile at coordinates x0 = %d, y0 = %d\n", x_0, y_0);    // Print global x coordinates    printf("global->\t");    for (int x = 0; x < smem.buffer.width(); ++x) {      printf("[%4d] ", x_0 + x);    }    printf("\n");    // Print local x coordinates    printf("local ->\t");    for (int x = 0; x < smem.buffer.width(); ++x) {      printf("[%4d] ", x);    }    printf("\n");    for (int y = 0; y < smem.buffer.height(); ++y) {      // Print global and local y coordinates      printf("[%4d] [%2d]\t", y_0 + y, y);      for (int x = 0; x < smem.buffer.width(); ++x) {        printf(" %4d  ", smem.buffer.data[y][x]);      }      printf("\n");    }        //invalid barrier   __mbarrier_inval(&smem.bar);  }}
```

编译我们可以看到TMA的SASS指令UTMALDG(Load from GMEM, 2D Tensor)

```
nvcc -arch sm_90 -c tma1.cu;cuobjdump -sass tma1.o | grep TMA        /*02c0*/                   UTMALDG.2D [UR8], [UR4] ;                                 /* 0x00000008040075b4 */
```

在CUDA中也提供了类似的实验的API

```
cuda::device::experimental::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
```

由于篇幅有限,更多关于TMA的内容, 我们将在后续的Flash Attention-3相关的算子算法中介绍.

#### 2.3.2 WGMMA

在Hopper中可以将一个SM内的4个SubCore组在一起, 4个连续的Warp构成WarpGroup 配合TMA实现4个TensorCore并行的64xNx16(N=8~256)的矩阵乘法.

![图片](assets/6e404dfc82a6.png)

需要注意的是这是一个仅在Hopper SM90架构上支持的功能

![图片](assets/a9b58b61ac6f.png)

由于需要协同4个Warp内的128个线程, 因此WGMMA也是一个异步的指令

wgmma.mma_async

wgmma.fence

wgmma.commit_group

wgmma.wait_group

对于矩阵乘法, 首先需要将它们加载到SMEM或者寄存器上. 加载完成后需要在warp group level执行`wgmma.fence`操作. 然后执行`wgmma.mma_async`.对于所有的outstanding `wgmma.mma_async`通过`wgmma.commit_group`操作, 并且通过`wgmma.wait_group`等待完成.

需要注意的是CD矩阵必须在寄存器上, shape为(M=64, N), Layout为Row-Major格式
![图片](assets/30255a0d0398.png)

A和B矩阵可以在寄存器上,也可以在SMEM上. 在SMEM上时采用8x8 Core Matrix Layout的方式

![图片](assets/42686781101f.png)

A采用Row-major Zigzag排列

![图片](assets/e159fe7f7cdc.png)

B矩阵采用Col-major Zigzag排列

![图片](assets/4f365e6be5e9.png)

WGMMA指令如下

```
wgmma.mma_async.sync.aligned.shape.dtype.f16.f16  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;wgmma.mma_async.sync.aligned.shape.dtype.f16.f16  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;.shape   = {.m64n8k16, .m64n16k16, .m64n24k16, .m64n32k16,            .m64n40k16, .m64n48k16, .m64n56k16, .m64n64k16,            .m64n72k16, .m64n80k16, .m64n88k16, .m64n96k16,            .m64n104k16, .m64n112k16, .m64n120k16, .m64n128k16,            .m64n136k16, .m64n144k16, .m64n152k16, .m64n160k16,            .m64n168k16, .m648176k16, .m64n184k16, .m64n192k16,            .m64n200k16, .m64n208k16, .m64n216k16, .m64n224k16,            .m64n232k16, .m64n240k16, .m64n248k16, .m64n256k16};.dtype   = {.f16, .f32};
```

其中参数`d`为结果矩阵的寄存器, `a-desc`和`b-desc`为AB矩阵的描述符. `scale-d`表示是否需要`D=A*B+D`加D, imm-scale-a/b为AB矩阵正负的符号立即数, imm-trans-a/b为是否需要AB矩阵转置的立即数.

手上暂时没有Hopper的卡, 那么做一个小实验看看编译器生成的指令, 例如我们需要一个M64N16K16的矩阵乘法, 结果矩阵D需要`M*N=64*16=1024`个寄存器, 对于一个WarpGroup有128个线程, 平均每个线程需要8个寄存器.在Cutelass中`include/cute/arch/mma_sm90_gmma.hpp`有相应的示例`struct SM90_64x16x16_F32F16F16_SS`, 其中SS表示AB都在SMEM, RS表示A在寄存器,B在SMEM

```
#include<cuda.h>__global__ void kernel(float* D, uint64_t desc_a, uint64_t desc_b, const int scaleA, const int scaleB, const int scale_D, const int tnspA,const int tnspB) {     float d[16];     for (int i = 0 ; i < 16 ; ++i ) {       d[i]=0;     }        asm volatile(    "{\n"      ".reg .pred p;\n"      "setp.ne.b32 p, %10, 0;\n"      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"      " %8,"      " %9,"      " p,   1, 1 , 0 , 0; \n"    "}\n"      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])      :  "l"(desc_a),         "l"(desc_b),         "r"(int32_t(scale_D)));        //store to GMEM    for(int i = 0 ; i < 8 ; ++i ) {      D[i] = d[i];    }}
```

可以注意到在编译时WGMMA指令需要SM_90a的架构支持

```
#nvcc -arch sm_90 -c wgmma.cu ptxas /tmp/tmpxft_0014c40f_00000000-6_wgmma.ptx, line 48; error   : Instruction 'wgmma.mma_async with floating point types' not supported on .target 'sm_90'# nvcc -arch sm_90a -c wgmma.cu ; cuobjdump -sass wgmma.o > wgmma.sass
```

![图片](assets/f16800a162cc.png)

当我们需要并行执行多个WGMMA乘法时, 需要在尾部加入commit_group/wait_group, 如下所示:

```
__global__ void kernel(float* D, uint64_t desc_a, uint64_t desc_b, const int scaleA, const int scaleB, int scale_D, const int tnspA,const int tnspB) {     float d[16];     for (int i = 0 ; i < 16 ; ++i ) {       d[i]=0;     }        asm volatile(    "{\n"      ".reg .pred p;\n"      "setp.ne.b32 p, %10, 0;\n"      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"      " %8,"      " %9,"      " p,   1, 1 , 0 , 0; \n"    "}\n"      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])      :  "l"(desc_a),         "l"(desc_b),         "r"(int32_t(scale_D)));        //防止编译器优化    desc_a++;    desc_b++;    scale_D=1;    asm volatile(    "{\n"      ".reg .pred p;\n"      "setp.ne.b32 p, %10, 0;\n"      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"      " %8,"      " %9,"      " p,   1, 1 , 0 , 0; \n"    "}\n"      : "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])      :  "l"(desc_a),         "l"(desc_b),         "r"(int32_t(scale_D)));    asm volatile("wgmma.commit_group.sync.aligned;");    asm volatile("wgmma.wait_group.sync.aligned 0;");             //store to GMEM    for(int i = 0 ; i < 16 ; ++i ) {      D[i] = d[i];    }}
```

此时第一条wgmma指令就没有了记分牌

![图片](assets/cb0d178c8d15.png)

下一篇,我们将来讲讲如何通过TensorCore进行矩阵乘法运算加速, 并通过它逐渐了解Cutlass的编程框架.

参考资料

[1] 
Modeling Deep Learning Accelerator Enabled GPUs: https://arxiv.org/abs/1811.08309
[2] 
Automatic Kernel Generation for Volta Tensor Cores: https://arxiv.org/pdf/2006.12645
[3] 
PTX指令集: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
[4] 
Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture: https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/
[5] 
Add TMA example for Hopper H100 #214: https://github.com/NVIDIA/cuda-samples/pull/214