# GPU架构演化史7: Fermi架构详解

> 作者: zartbot  
> 日期: 2022年8月27日 11:20  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488081&idx=1&sn=efeda52bd9d5233eda36c9c2ec75fdbf&chksm=f9960293cee18b85e264d0fe112218cd57e956d820d856e6ae835dcf1b21263edccf3d2c1db9#rd

---

先来说说Fermi开发时的时间背景， 第一代Tesla G8x 是90nm工艺，所以SM在整个Die中的尺寸相对较小，显存是GDDR3位宽384bits，总带宽86.4GB/s. 第二代Telsa GT200换到了55nm，SM多了不少，显存还是GDDR3，但是位宽到了512bit,总带宽159GB/s所以也足够再去多加一些SM. 那么到了Fermi要做40nm，而且又有GDDR5内存可以用，你怎么出牌？还有2008年那段时间Radeon RV770浮点能力远超GT200，你作为N家的架构师会如何考虑？

*讲架构之前，我们来看看产品需求是什么?*

#### HPC需求

计算精度需要双精度浮点，虽然GT200增加了支持，添加了DPU(双精度浮点单元),但是效率不高，然后就是被人诟病的浮点MAD的精度问题，当然还有如何支持C++/FORTRAN/Python等语言?还有一个问题是科学计算对ECC内存的需求.

#### 系统本身的需求

由于新的芯片有了40nm的工艺可以放更多的CUDA core了，执行更多的线程时如何调度，缓存结构如何设计，上下文交换的速度如何保证，分支如何执行？应用程序上对各级内存的原子操作在SIMT时带来的死锁如何处理？延迟如何隐藏?

#### 图形渲染需求

而另一方面来自算法和软件的需求，2008年nVidia收购AGEIA获得PhysX等技术，如何把原有PPU(Physical Processing Unit)的功能集成到GPGPU? 还有曲面细分(Tessellation)技术,延迟渲染(DefferedShading),物理渲染(Physically Based Rendering)等一系列功能使得DirectX11的渲染管线发生了变化，最终DX11增加了3级用于曲面细分的pipeline，同时针对纹理处理，我们也可以构建一个专门的计算着色器(Compute Shader)执行更多的特效。来实现对纹理进行模糊处理(blur)，再将着色器资源视图(shader resource view)与模糊处理后的纹理相绑定，以作为着色器的输入。

![图片](assets/05e9837e098e.png)

在DirectX11以及CS的帮助下，游戏开发者便可以使用更为复杂的数据结构，并在这些数据结构中运行更多的通用算法。另一方面由于CUDA的影响，计算着色器虽然是一种可编程的着色器，但Direct3D 并没有将它直接归为渲染流水线中的一部分，独立出来可以执行大量的并行计算任务，而不必渲染任何图形。

DX11带来的体系结构要求的巨变，CUDA生态的逐渐成熟，周边器件的性能，芯片制程的影响，这些都是导致Fermi架构变革的历史背景...同时期的AMD从RV770到RV870几乎就是照搬上一代核心，然后再增加了一些支持DX11的功能，而nVidia再一次展现了它过人之处，完全按照DX11的特性重新设计了一代产品.
曲面细分和置换贴图

![图片](assets/ae4d20c83132.png)

很多人会问，直接CPU生成多个三角形多个顶点不就行了么，为什么要在Pipeline中间加三级呢?来自nVidia < DX11 Tessellation >的Session讲的非常清楚, CPU的算力和GPU之间的带宽决定了不能生成过多的三角形，三角形的生成需要动态的LOD(Level of Detail，细节级别)，我们可以在CPU上生成一个粗颗粒度的网格，然后交由GPU去增加局部的细节，例如离视角摄像机位置远的就不需要渲染的那么详细，然后还有一些物理模拟特效，也需要在一个低精度粗颗粒度的多边形上模拟，这样也可以降低大量的计算资源。整个计算的pipeline如下:

![图片](assets/ff4e7275a655.png)

Input Assembly会将顶点装配成三角形，但是到了这一步，首先顶点(Vertex)数据会在Patch Assembly这个地方通过若干(1~32)控制点(control point)封装成一个补丁Patch. 然后我们针对这个Patch通过贝塞尔曲线来构建曲面

Hull Shader 针对每个Patch逐一进行处理，并输出网格的曲面细分因子(Tessellation Factor),它代表了曲面细分阶段Patch完成细分处理后的份数，更高的因子代表需要划分更多的三角形，这里就要受LOD影响了，通常的处理方式是：摄像机和物体的距离、根据屏幕占比的范围，表面粗糙程度等..然后通过Tessellator产生细分好的网格

![图片](assets/56b129349189.png)

细分算法较多，例如以PN三角形为例，它采用贝塞尔曲面的方式，细分的网格中都有新的法向量: 

![图片](assets/766b364e8331.png)

然后Domain Shader执行置换贴图(Displacement Map),它是一种称为微型多边形细分的算法来实现的。它根据贴图的灰度决定高度，然后根据镶嵌所得到的多边形，沿着原先的表面法线方向移动微多边形。接着再为新的多边形确定好新的法线方向。最终细分的网格就构建好了：

![图片](assets/6df07d10f141.png)

另一个更形象的例子:

![图片](assets/7adf20648e43.png)

两者结合

![图片](assets/23ec5b99774d.png)

*可以看到这个新功能对画质的提升是非常明显的,但是代价也非常大，由于产生了大量的新的三角形，对于后期像素着色器压力也大了很多， 另一方面延迟渲染和Compute Shader的引入对于Texture的压力也加大*

### Fermi架构

2010年，Fermi架构的发布解决如上的很多问题. 很多分析Fermi架构的人都是做AI加速芯片的，因此有些图形的细节没有Cover的很好.在此做一些补充 

![图片](assets/a4db653efed9.jpg)

得益于40nm工艺，每个SM有了32个cuda core（4倍于GT200)，由于核心的增加和图形任务中对纹理和计算任务中对数据的存储要求和延迟隐藏的需求，增加了64K的Shared Memory和L1可配置的内存，非常出色。而伴随着大量的核及大量的线程，寄存器文件的数据量也增加到了32,768 x 32-bits， 双精度浮点能力也8倍于GT200，同时Warp调度器增加到2个，具体内容我们后面详细分开来介绍:

![图片](assets/df13b1237471.png)

还记得前面所说的曲面细分的功能么？由于DirectX11的Compute Shader引入和曲面细分后的HullShader及Domain Shader使得数据在核内流转的次数更多，如果采用GT200 Gemotry Controller全局调度，传统的Vertex work和Pixel work Distributor相对固定了计算流水线而且效率更低。曲面细分和置换贴图带来的三角形数量膨胀，使得我们必须把ROP也集成在核中, 最终nvidia把一些Geometry Controller的功能核新添加的Tessellator功能集成到了每个SM上，构成了新的PolyMorph Engine，而把原有的ROP核Z-hull等功能整合成了4个SM共享的Raster Engine

![图片](assets/55472481ca33.png)

然后再针对这样复杂的计算流程配置相应的L1、L2 Cache就可以使得整个数据生命周期都在处理器内执行了，降低了对外部内存的访问压力，所以这也是Fermi引入Cache的原因:

![图片](assets/b6f5d1835eb2.png)

它比起GT200专用的Tex L1和更靠近内存控制器和ROP的Tex L2，它变成了一个更加统一的架构：

![图片](assets/ded57d3d5aa4.png)

当然设计独立的L1缓存还有一个用处就是DirectX11中Compute Shader的引入和同期收购PhysX技术：

![图片](assets/6d3fdc407e63.png)

使用Compute Shader构建的运动模糊特效:

![图片](assets/5f08264b88d0.png)

当然还有早期的光追(Ray Tracing)效果也得到了提升，但真正的基于BVH算法的硬件加速RT Core则要到若干年以后

![图片](assets/bed63f4ecdad.png)

### 第二代PTX ISA

Parallel Thread Execution(PTX) ISA可以理解成CUDA编译过程中的一个IR层，然后会根据不同计算能力的GPU JIT产生真正执行的SASS指令集. PTX是一套精简的RISC指令，第二代PTX指令集的更新主要包括对OpenCL和DirectCompute生态的支持，同时相对于第一代能够保证后期多代的PTX层指令集的稳定性，内存访问指令也可以转换成64-bits的寻址模式，同时针对C++支持了虚函数、函数指针、new、delete等内存操作，同时也把抛出异常支持了.

![图片](assets/98aebe1bf3cb.png)

然后针对Tesla相对独立的内存寻址空间，变成了统一的地址空间:

![图片](assets/8a2e8cc198f7.png)

新增加了FMA指令解决被诟病的浮点计算精度问题:

![图片](assets/43440c526330.png)

为了更好的理解PTX指令，我们构建一个CUDA程序，这是一个在线性代数中非常常见的函数, `__global__`表示这个函数是在GPU上执行的，然后我们可以通过cudaMalloc和cudaMemcpy将需要计算的数据拷贝到显存中，最后执行saxpy<<<nblocks,256>>>，然后将结果返回并验证输出.

```
#include <cuda_runtime.h>#include <stdio.h>__global__ void saxpy(int n, float a, float *x, float *y) {  int i = blockIdx.x * blockDim.x + threadIdx.x;  if (i < n)    y[i] = a * x[i] + y[i];}int main() {  int n = 1024;  int nByte = sizeof(float) * n;  float *h_x = (float *)malloc(nByte);  float *h_y = (float *)malloc(nByte);  float *h_from_cuda = (float *)malloc(nByte);  memset(h_from_cuda, 0, nByte);  // generate data  for (int i = 0; i < n; i++) {    h_x[i] = i;    h_y[i] = i * 10000;  }  float *d_x, *d_y;  int nblocks = (n + 255) / 256;  cudaMalloc(&d_x, n * sizeof(float));  cudaMalloc(&d_y, n * sizeof(float));  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);  cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);  saxpy<<<nblocks, 256>>>(n, 2.0, d_x, d_y);  cudaMemcpy(h_from_cuda, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);  cudaDeviceSynchronize();  for (int i = 0; i < n; i++) {    printf("x= %-12.2f  y = %-12.2f : 2*x+y = %-12.2f delta %-4.2f\n", h_x[i], h_y[i],           h_from_cuda[i], h_from_cuda[i] - 2 * h_x[i] - h_y[i]);  }  cudaFree(d_x);  cudaFree(d_y);  free(h_from_cuda);  free(h_x);  free(h_y);}
```

通过nvcc可以产生PTX代码( nvcc -ptx 01_dim.cu, 通过cuobjdump --dump-ptx亦可 ),`.entry` 代表了函数入口，其中包含了四个核函数输入的参数，剩下的ld/mov/add基本上和其它处理器平台一致。例如通过mov加载了blockIdx blockDim和threadIdx，然后通过`mad`计算了线程的tid的值。由于是SIMT平台，它和其他平台不同的是，它有一个谓词寄存器%p1,用于判断分支情况，并在下一个周期根据p1进行跳转.

```
//file 01_dim.ptx//// Generated by NVIDIA NVVM Compiler//// Compiler Build ID: CL-29618528// Cuda compilation tools, release 11.2, V11.2.152// Based on NVVM 7.0.1//.version 7.2.target sm_52.address_size 64        // .globl       _Z5saxpyifPfS_//函数入口和四个参数定义.visible .entry _Z5saxpyifPfS_(        .param .u32 _Z5saxpyifPfS__param_0,        .param .f32 _Z5saxpyifPfS__param_1,        .param .u64 _Z5saxpyifPfS__param_2,        .param .u64 _Z5saxpyifPfS__param_3){        .reg .pred      %p<2>;        .reg .f32       %f<5>;        .reg .b32       %r<6>;        .reg .b64       %rd<8>;        ld.param.u32    %r2, [_Z5saxpyifPfS__param_0];        ld.param.f32    %f1, [_Z5saxpyifPfS__param_1];        ld.param.u64    %rd1, [_Z5saxpyifPfS__param_2];        ld.param.u64    %rd2, [_Z5saxpyifPfS__param_3];        /*计算tid         int i = blockIdx.x * blockDim.x + threadIdx.x;        */        mov.u32         %r3, %ctaid.x;         mov.u32         %r4, %ntid.x;        mov.u32         %r5, %tid.x;        mad.lo.s32      %r1, %r3, %r4, %r5;   /* if (i < n), p1是一个谓词寄存器*/        setp.ge.s32     %p1, %r1, %r2;        @%p1 bra        LBB0_2;        cvta.to.global.u64      %rd3, %rd2;        cvta.to.global.u64      %rd4, %rd1;        mul.wide.s32    %rd5, %r1, 4;        add.s64         %rd6, %rd4, %rd5;        ld.global.f32   %f2, [%rd6];        add.s64         %rd7, %rd3, %rd5;        ld.global.f32   %f3, [%rd7];        fma.rn.f32      %f4, %f2, %f1, %f3;        st.global.f32   [%rd7], %f4;LBB0_2:        ret;}
```

SASS代码更加底层，可以通过cuobjdump 查看

```
cuobjdump ./a.out  --dump-sass     code for sm_52                Function : _Z5saxpyifPfS_        .headerflags    @"EF_CUDA_SM52 EF_CUDA_PTX_SM(EF_CUDA_SM52)"                                                                                 /* 0x001cfc00e22007f6 */        /*0008*/                   MOV R1, c[0x0][0x20] ;                        /* 0x4c98078000870001 */        /*0010*/                   S2R R0, SR_CTAID.X ;                          /* 0xf0c8000002570000 */        /*0018*/                   S2R R2, SR_TID.X ;                            /* 0xf0c8000002170002 */                                                                                 /* 0x001fd842fec20ff1 */        /*0028*/                   XMAD.MRG R3, R0.reuse, c[0x0] [0x8].H1, RZ ;  /* 0x4f107f8000270003 */        /*0030*/                   XMAD R2, R0.reuse, c[0x0] [0x8], R2 ;         /* 0x4e00010000270002 */        /*0038*/                   XMAD.PSL.CBCC R0, R0.H1, R3.H1, R2 ;          /* 0x5b30011800370000 */                                                                                 /* 0x001ff400fd4007ed */        /*0048*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x140], PT ;  /* 0x4b6d038005070007 */        /*0050*/                   NOP ;                                         /* 0x50b0000000070f00 */        /*0058*/               @P0 EXIT ;                                        /* 0xe30000000000000f */                                                                                 /* 0x081fd800fea207f1 */        /*0068*/                   SHL R2, R0.reuse, 0x2 ;                       /* 0x3848000000270002 */        /*0070*/                   SHR R0, R0, 0x1e ;                            /* 0x3829000001e70000 */        /*0078*/                   IADD R4.CC, R2.reuse, c[0x0][0x148] ;         /* 0x4c10800005270204 */                                                                                 /* 0x001fd800fe0207f2 */        /*0088*/                   IADD.X R5, R0.reuse, c[0x0][0x14c] ;          /* 0x4c10080005370005 */        /*0090*/         {         IADD R2.CC, R2, c[0x0][0x150] ;               /* 0x4c10800005470202 */        /*0098*/                   LDG.E R4, [R4]         }                                                                                 /* 0xeed4200000070404 */                                                                                 /* 0x041fc800f6a007e2 */        /*00a8*/                   IADD.X R3, R0, c[0x0][0x154] ;                /* 0x4c10080005570003 */        /*00b0*/                   LDG.E R6, [R2] ;                              /* 0xeed4200000070206 */        /*00b8*/                   FFMA R0, R4, c[0x0][0x144], R6 ;              /* 0x4980030005170400 */                                                                                 /* 0x001f9000fde007f1 */        /*00c8*/                   STG.E [R2], R0 ;                              /* 0xeedc200000070200 */        /*00d0*/                   NOP ;                                         /* 0x50b0000000070f00 */        /*00d8*/                   NOP ;                                         /* 0x50b0000000070f00 */                                                                                 /* 0x001f8000ffe007ff */        /*00e8*/                   EXIT ;                                        /* 0xe30000000007000f */        /*00f0*/                   BRA 0xf0 ;                                    /* 0xe2400fffff87000f */        /*00f8*/                   NOP;                                          /* 0x50b0000000070f00 */                ..........
```

#### SIMT中的分支执行

我们注意到和SIMD不同的是，它支持了针对数据中边界情况的分支处理更加容易，通常分支情况也不会非常复杂，因此当时设计了一种叫SIMT Stack的机制来执行分支。例如我们有如下一段代码,右边是生成的PTX指令，我们可以注意到源程序中有2个if分支和一个while循环，因此PTX中出现了3个谓词寄存器

![图片](assets/4c32ede611bf.png)

整个并行执行的方式如下， SIMT每个线程共享PC，因此对于分支并行执行，系统创建了一个Reconvergence PC，Next PC和Active Mask的栈结构，简单起见这里只显示了4个线程，每个方框内/xxxx表示它们在这一步的跳转条件判断的真假情况：

![图片](assets/71ed32ce1a2b.png)

例如程序运行到A时，进行判断，有1个需要跳转到F，另外三个线程继续走B， 因此系统就会在这个分支时，将NextPC(B或者F)以及它们最终会汇聚的PC(Reconvergence PC) G的地址加入到表中，并且每个线程根据自己的分支谓词更新相应的Active Mask，下一轮调度的时候，GPU采用深度优先的方式并根据Active Mask在前三个thread上执行B，并产生分支C、D，然后将CD分支情况压入栈，并执行，直到最后汇总到RPC都是G且E后到G无指令时，执行F，完成线程执行的汇聚，接下来继续并行执行。

学术界针对分支如何在SIMT下高效执行产生了大量的研究， 例如我们可以注意到根据相同的RPC，如果active mask没有冲突，还可以进一步采用单次发送多个不同指令的方式执行，例如下图:

![图片](assets/27b0bb57131d.png)

#### Atomic指令支持和SIMT死锁

Fermi中还增加了大量的针对本地内存和全局内存的原子操作，但是原子操作在SIMT Stack的架构上会出现死锁

![图片](assets/69817107b231.png)

而后面彻底解决这个问题的是Stack-less convergence Barrier后面我们在介绍Volta的时候再详细叙述.

### CUDA Core

从微处理器架构来看，Cuda Core是一个SIMT的前端，配合一个SIMD的后端

![图片](assets/cea312496d5b.png)

核心内部的访存延迟该如何隐藏呢?指令如何高效执行呢？nvdia一方面的做法是将核心频率设置为调度器的两倍，因此16个CUDA核心就能满足一个Warp 32个线程执行了，为了让32个核心满载，设计了2个Warp Scheduler，每个时钟周期可以从如下四个计算器件中选择发射两条指令。

![图片](assets/066cfc6dccaa.png)

这样就可以充分的用满内部的资源了来维持峰值的处理能力:

![图片](assets/ea94c13016b0.png)

但是这么多线程这么多指令，还有外部的Cache，如果Cache Miss对流水线也会带来影响，因此引入了Operand Collector的组件，具体内容我们在稍后的内存层次化结构的章节讲述.

### GigaThread Scheduler

官方的白皮书只是简单的介绍了这个调度器， 它作为全局调度器和每个SM中的Warp Scheduler一起构成一个2层的分布式调度器，作为芯片这一级的全局调度器，支持多个Kernel函数并行执行:

![图片](assets/72e12dd2bb32.png)

而另一方面提到Context Switching的速度快了10倍，仅有20~25us，这个本质的原因还是在Cache的引入，对于Scheduler的工作模式和相应的取舍分析的文章并不多，有一个技巧，我们可以通过内联汇编的方式查看调度到的SM ID

```
__global__ void saxpy(int n, float a, float *x, float *y) {    unsigned int ret;  asm("mov .u32 %0, %smid;" : "=r"(ret));  if (threadIdx.x == 0) {    printf("Block ID: %d, SMID: %d\n",blockIdx.x,ret);  }  int i = blockIdx.x * blockDim.x + threadIdx.x;  if (i < n)    y[i] = a * x[i] + y[i];}
```

如何在大量的线程中调度一直是一个非常值得研究的方向，内存的局部性，指令的并行度，算法本身等....在《 General-Purpose Graphics Processor Architecture》一书中也介绍了大量的研究成果，而如今在云渲染和GPU需要完成虚拟化场景支持时，如何更加有效的调度和同时防止寄存器文件和缓存污染。

### Cache & Memory

为了HPC场景，寄存器文件、L1、L2和GDDR5显存都支持了ECC，相对于上一代，多了Cache的层次化结构，每个SM新增了可以动态配置的大小的L1 Cache和Shared Memory，可以根据需要选择48KB L1 + 16KB SharedMem或者16KB L1 +  48KB SharedMem，L2为768KB

![图片](assets/2b00dc2d8e3d.png)

但是我们注意到寄存器文件为32K * 32bit，L1Cache 16KB，2级缓存平均每个SM 48KB，尺寸上呈现出一个倒三角的情况，而我们同时注意到不同的应用对Cache重复读写的需求也是有所不同的，所以针对不同业务的需求基于带宽、空间等进行限速:

![图片](assets/2c2af0644f2a.png)

而另一方面针对寄存器文件访问冲突等降低效率的问题， nvdia在Cuda core中引入了Operand Collector(OC)的概念，每个指令进入寄存器读取阶段到最后被分配到OC，这一段就有非常灵活的调度机制了。

![图片](assets/1379e7a9f77c.png)

OC结构如下：由于每个操作涉及3个源操作数，所以每个单元设置三个条目，每个条目包含四个字段:

![图片](assets/b790612feb9d.png)

一个有效位，由于并非每条指令都包含三个操作数，所以用来指定这位是否有效，然后包含一个寄存器RID为，一个就绪位和一个操作数数据字段。只有当寄存器就绪后才通知调度器issue指令执行.

### Debug Tool

nVidia Nexus debug工具也伴随着Fermi发布了，这也极大的方便了程序开发和性能调优，

![图片](assets/af4695a204b9.png)

最后David Patterson总结了Fermi的十大创新，和3个新的挑战:

![图片](assets/b8d3dc0f797d.png)