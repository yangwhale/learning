# GPU架构演化史6: Telsa CUDA架构详解

> 作者: zartbot  
> 日期: 2022年8月22日 16:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488023&idx=1&sn=2a3d6f808b9d52396edb230ca8ce6e09&chksm=f99602d5cee18bc357812b1f682bb3f45c404da328688d825071de81ed8b69d7c92e1e48e592#rd

---

### GPGPU统一之路

#### 顶点和像素着色器的演进

对于GPU是如何演进到GPGPU的, 这里需要更加详细的记录一下. 1999, GeForce 256在1999年，GeForce256集成了一个固化的32-bit 浮点向量计算引擎用于T&L，以及一个固定功能的整数的像素-片段处理流水线(Pixel-Fragment Pipeline)这便是最早的GPU雏形，某种意义上来看也是对SGI RealityEngine的单芯片实现。而在Geforce 3中将Vertex Shader(VS)赋予了可编程能力，变成了一个可配置的32bits浮点向量计算引擎，同时Pixel Shader(PS)也开始支持32bits浮点固化流水线2002年Radeon 9700第一次将24bits浮点可编程引擎引入PS并率先支持了DX9。而后期GeForce FX系列也引入了32位浮点可编程的PS。

DX8 SM1DX9 SM2DX9 SM3DX10 SM4Vertex Instruction12825651264KPixel  Instruction4+832+6451264KVertex Constants9625625616*4096Pixel Constants83222416*4096Vertex Temps1616164096Pixel Temps212324096Vertex Inputs16161616Pixel Inputs4+28+21032Render Targets1448Vertex Texturesn/an/a4128Pixel Textures816161282D Tex Size

2k * 2k8k * 8kInt Ops---YesLoad Ops---YesDerivatives--YesYesVertex Flow Ctrln/aStaticStatic/DynDynamicPixel Flow Ctrln/an/aStatic/DynDynamic
VS主要是操作一系列顶点,然后是多空间转换,例如我们需要渲染一个3D物件,它自己有一个坐标系我们将其称作Local Space,而将其放置于一个3D空间后还会有缩放旋转平移等操作,所以需要从Local Space映射到Global Space, 然后Global Space和我们需要观察的视角还不同,因此再有一次空间变换到Screen Space,最后再从3D投射到2D坐标系. VS处理完所有的变换后交给后续的几何处理产生Primitive(图元)，以前这些都是固定的流水线，DX10新增了一个处理这一步的Geometry Shader(GS)，然后对图元做光栅化处理生成Fragments交由PS处理像素，像素处理中主要是根据纹理材质映射染色过滤遮挡和插值等。

从计算模型来看，VS需要`低延迟` `高精度` 的数学处理， 而PS主要用于`高延迟` `低精度`的材质过滤， 而且通常PS处理像素级的数据，数据量高于VS，基本上那个年代的GPU都按照3：1的比例配置PS和VS，但是现实中也发现两者的工作负载经常会出现一些不平衡的情况，例如有大块的三角形，VS工作负载相对较低，而如果小的三角形特别多的场景VS就很重。而在DX10中加入了新的GS使得我们更难以一个固定的比例去放置VS和PS，融合成了必然选择。

另一方面我们注意到，为了颜色更加真实，PS也使用了浮点计算引擎，并且针对阴影或者Z-Cull这些功能，也需要让PS获得分支处理的能力，因此DX9在PS中也引入了流程控制，最后GPU的VS和PS演进成如下结构:

![图片](assets/8e7abdba2dd6.png)

正如前文所述，VS需要低延迟，因此VS更像是一个MIMD的处理单元，而PS中数据量并发特别大，通常是采用SIMD实现的，一个Fragment流会同时进入多个Fragment Unit处理， 而Fragment Unit它自身也是由两个Shader Unit构成的一个VLIW处理单元：

![图片](assets/3a84010847f6.png)

而在GeForce 6中引入了PS的Branch，就使得SIMD执行的效率也遇到一些问题，具体在GPU Gems2中有些章节在阐述。而最终也是nVidia在设计CUDA架构时使用SIMT的原因。尽管有如此一些小的效率问题，但是丰富的可编程特性抽象，使得可编程的领域专用语言Shader Language(Cg, GLSL, HLSL)出现了. 同时也就诞生了一系列基于SL的数值算法，例如GPU Gems2 Chapter45中提到进行期权价格计算

```
void init(Stream stockPrice,    Stream strikePrice,    Stream yearsToMaturity,  Stream volatility,    uniform float riskFreeRate,  uniform float numSteps,    float2 offset : DOMAIN0,  out float4 result : RANGE0){  float deltaT = yearsToMaturity.value(offset.x)/numSteps;  float u = exp(volatility.value(offset.x) * sqrt(deltaT));  float price = stockPrice.value(offset.x) *                pow(u, 2 * (offset.y - 0.5) - numSteps);  float value = max(strikePrice.value(offset.x) - price, 0);   result = value;}void iterate(Stream Pu,    Stream Pd,    Stream optval,    float2 offset : DOMAIN0,  float2 offsetplus1 : DOMAIN1,    out float4 result : RANGE0){  float val = (Pu.value(offset.x) * optval.value(offsetplus1) +               Pd.value(offset.x) * optval.value(offset));    result = val;}
```

但是这些DSL执行起来还是很麻烦，接下来Stanford玩了一件大事情

#### Brook for GPU

Stanford在2004年发布了一篇论文< Brook for GPUs: Stream Computing on Graphics Hardware > 通过扩展C并实现了一个BRCC的编译器，利用当时Nvidia GeForce 6800和ATI Radeon X800XT 实现了SGEMV矩阵运算、FFT等五种通用计算的算法，

```
// Original Brook code:kernel void saxpy(float alpha, float4 x<>, float4 y<>,               out float4 result<>) { result = (alpha * x) + y;}//Intermediate Cg code:void saxpy (float alpha, float4 x, float4 y, out float4 result) { result = alpha * x + y;}void main (uniform float alpha : register (c1), uniform _stype _tex_x : register (s0), float2 _tex_x_pos : TEXCOORD0, uniform _stype _tex_y : register (s1), float2 _tex_y_pos : TEXCOORD1, out float4 __output_0 : COLOR0) {  float4 x; float4 y; float4 result;  x = __fetch_float4(_tex_x, _tex_x_pos );  y = __fetch_float4(_tex_y, _tex_y_pos );  saxpy(alpha, x, y, result );  __output_0 = result;}
```

### Tesla 架构

针对前面的一系列问题， Tesla架构设计的初衷主要就是融合VS和PS并构建统一的处理架构，我们注意到在VS中包含了Vector ALU，而PS中则是类似于一个mini VLIW的执行机构， VS和PS都要调用Texture Unit，如何整合两个团队并共享例如Texture Unit这些珍贵的硬件资源，而在这个过程中不乏有很多思考和取舍。最后nVidia设计了一个标量的处理器核，然后通过它构成了一个可扩展的处理器阵列，并发布了Tesla架构的第一代产品GeForce 8800(G80)：

这里需要再强调一下在VS时Vector，PS是VLIW的时候，nVidia选择标量的勇气是CUDA成功的关键之一，而ATI似乎对Vector ALU这种东西念念不忘，继续在VLIW的路上走了三代TeraScale架构，最后才迷途知返发布GCN. 至于另一个维度，关于SIMD的Branch性能的问题， Intel当时觉得自己Branch Prediction牛到没对手，直接把P54c的核弄显卡上， 标量、向量、分支预测啥都有，最终夭折...而nV也赢在SIMT的巧妙

![图片](assets/ee395e8ced30.png)

第一块Telsa架构的GeForce 8800包含了128个标量的流处理器(Streaming-processor,SP)核心, 每8个一组构成16组(Streaming Multiprocessors,SM), 然后每两组SM共享一个Texture Unit构成一个Texture/Processor Cluster(TPC). 外部DRAM内存控制器中包含一个L2缓存和一个固化功能的光栅操作处理器(Raster Operation Processor).

Die结构如下：

![图片](assets/e0fb88e5d895.png)

整个工作流程如下，首先GPU host Interface接收来自主CPU的命令，然后从系统内存中获取数据并实现上下文的交换。Input Assembler 采集几何图元(Geometry primitives)，包括点、线、三角形等，然后获取相关的vertex信息，然后输出给Vertex Work Distiribution模块，这个模块采用Round-Robin的方式将Vertex Work packets发送到不同的TPC处理，TPC执行Vertex Shader 程序，针对DX10也会执行Geometry Shader 程序，执行完成后写到onchip buffer然后通过viewport、clip、setup、raster/zcull等模块转换成像素片段(Pixel Fragment)，此后Pixel work distribution再进行一次调度给TPC进行Pixel Shader处理，此时的调度采用基于像素为止的方式，能够获得更好的内存局部性访问，最后处理完的Fragments 由ROP处理。当然它还有一个专门针对计算任务的调度器Compute work Distribution用于计算任务的处理。

#### TPC

一个TPC中使用两个SM配合一个Texture Unit是为了平衡数学计算和访存的复杂度，而在后面一代GT200上将使用3个SM共享一个Texture Unit. 而每个SM内采用8个标量核也是为了对应Geforce 6的PS的两个Unit刚好单个clock可以最多执行8个op.

![图片](assets/46b217331d31.png)

Geometry Controller
TPC内部包含一个Geometry Controller，我们可以注意到VS可以按照Vertex进行调度，而GS则一定要按照Primitive来调度，所以这里采用了Input Assembler组装好Primitive然后送给TPC处理，TPC首先SIMT在多个核处理Vertex，然后如果需要GS，则由Geometry Controller负责将Primitive再次进入SM处理。
SMC
SMC负责将计算任务拆分打包成Warp并且交由其中一个SM执行，同时它还需要负责协调SM和Texture Unit以及外部资源的获取，具体Warp的调度在后面详细讲述。
Streaming Multiprocessor(SM)
SM是一个统一的图像和计算处理器，可以处理VS、GS、PS或者其它通用的并行计算程序, 一个SM包括8个流处理器核(Streaming processor)和2个特殊函数处理单元(SFU), 在SM内部有一个MT Issue用于把Warp任务拆分成一条条指令分配给SP处理。SM内包含了一个I Cache同时也包含了一个16KB的共享内存。
SFU
主要用于一些超越函数的计算和Vertex属性和Primitive来计算插值函数。每个SFU包含了4个浮点乘法器。
Texture Unit
它通过SMC和ROP单元实现了外部内存的读取、存储和原子操作。数据的Load、Store都由SMC触发，Texture指令输入一个4d的齐次坐标，然后返回一个4d的RGBA颜色结果，而中间针对图形的操作配置有专门的AGU地址生成器和相应的Filter单元. 而ROP则是固定函数的去处理光栅操作，这些操作直接对内存进行处理即可。

![图片](assets/3e9d821a6dc1.png)

Streaming Processor（SP）
SP是一个基本的标量处理单元，可以完成32bits浮点的 add、mul、multiply-add操作，也包含了一个32bits的整型计算单元。

### SIMT

我们可以注意到图形处理的计算任务是非常容易并行拆分的，VS处理单个Vertex，GS处理单个Primitive，PS处理Pixel，同时针对CUDA C程序也可以采用单线程并行的方式来处理，因此需要我们对SM中的每个Thread都有独立的执行状态和执行独立的Code Path，而整个CUDA中最重要的也就是如何调度数百个核，传统的SIMD的方式无法执行相对独立的Code Path，NVidia实现了一种SIMT的处理方式。

32个线程被打包在一次称为一个Warp，这个打包工作是在SMC中完成的，然后交由SM执行，SM Multi Thread Issuer则用于创建、管理、调度和执行这些Thread，也就是说Warp交给它后，它会将其打散成Instruction让SP和SFU将一条指令累计执行32次，

![图片](assets/86b72840cb27.png)

但是每个SM只有8个SP？那么一个指令就需要SP 4个cycle执行，每个cycle执行8个thread，这这个时候就可以利用Dual Issue的方式在两个Cycle内继续发射一条指令同时调度FPU和SFU

![图片](assets/f0397b5d9c71.png)

另一个就是延迟隐藏：

![图片](assets/4c6716122d82.png)

但是这样的处理方式与SIMD有啥区别呢？主要就体现在能够支持分支的情况下工作，所以SIMT之前很多人也把其称为Branch SIMD，每个线程都有自己独立的指令地址和寄存器状态，但是当有分支的时候，WARP会同时执行每个Branch Path，并disable不在那个path上的thread，这些通过一个32bit mask标记，然后再对齐汇总到下一段指令，但这不又有几个SP摸鱼了..

![图片](assets/d23cf0c0e593.png)

当然第二代Tesla GT200还增加了Warp Vote的功能，另外还有Dynamic Warp Formation的处理方式：

![图片](assets/dddfa662714a.png)

### SM Instruction

Tesla执行标量指令，和以往的GPU基于Vector的指令完全不同，主要原因是Shader程序变得更长更加标量化，Vector指令或者使用VLIW无法取得较高的占用率，因此转换成标量处理虽然峰值浮点性能差了一些，但标量执行也为C编译器打下了很好的基础，不用麻烦的去折腾VLIW，仅有Texture还是Vector指令，主要是用于坐标的vector(齐次坐标)和返回的过滤的颜色RGBA都是4元的.

#### PTX ISA

PTX ISA是一个基于寄存器的标量指令集，支持浮点、整数、bit、转换、超越函数、分支循环等流程控制和内存读写以及Texture操作指令。由于Tesla使用的PTX 1.0资料非常少，而nVidia开始宣传PTX也是再Fermi时代才开始的，因此我们将这部分详细的内容留到后面的章节。

#### 访存指令

Texture 指令被用来访问Texture memory，为了支持标准的C/C++语言和通用计算，Tesla SM设计了专用的内存Load、Store指令，初期的内存空间被分为Local、Shared、Global三段，后期的Tesla架构还加入了Atomic操作的支持。

### CUDA并行计算架构

对于数据并行解耦的问题，在MPI中其实已经非常常见了，而CUDA也是借鉴了这套技术，同时配合标准C的编译环境，构建Kernel函数后就可以调度到多个核执行了。而在调度过程中为了和硬件核规模解耦并隐藏内部的TPC、SM、SP数量等，它对计算任务做了如下抽象

![图片](assets/98e5ecaa4df8.png)

用于被CUDA卸载到GPU上执行的线程函数被称为Kernel，它构成一个Grid，然后Grid可以拆分成多个Block，每个Block又可以拆分成线程。Block和Grid都支持按照dim3的结构定义，然后一个block内的多个线程构成一个CTA(Co-operative thread Array)，CTA定义是一个Dim3的结构，也就是说您可以将其按照1D、2D、3D的方式来组织,默认为(1,1,1)如果需要2D那么就改x，y的数量:

![图片](assets/bd97084fa8c0.png)

拆分后的每个线程在执行的时候可以通过BlockIdx和ThreadIdx获得自己的线程ID，然后根据线程ID执行对不同数据的操作,如下所示，Kernel函数利用函数头部增加__ global __ 关键字的方式来声明

![图片](assets/d787b709ce52.png)

然后程序需要显示的去GPU上Malloc内存和执行Host2Device以及Device2Host的memcpy.

### GT200改进

真正取得成功的是第二代Tesla架构GT200，新的65nm工艺可以放入更多的组件，可以明显看到SM的尺寸大了很多，同时核心为止增加了一个全局的调度器

![图片](assets/927074231045.png)

TPC增加到了10个，同时它在TPC内增加了一个SM，三个SM共享SMC和Texture Unit

![图片](assets/c29a1dc1d594.png)

另一个改进是针对超算HPC的业务场景，SP内增加了一个双精度浮点的FMA

![图片](assets/149b1869f4af.png)

统一调度单元
在G80中还有原来的VS、GS和PS调度划分，为了支持GS调度还需要前置一个Input Assembler，任务调度相对零散，而在GT200中变成了一个统一的全局的Thread blocks 调度器
warp vote
第二代Tesla还提供了Warp Vote的功能，可以极大的提高处理能力，可以理解为对Thread之间的信息进行比较投票，可以使用__ any 和__ all来判断一个谓词逻辑，即谓词参数非零时返回1，并在相应的Mask上设置。
Atomic
第二代Tesla还增加了针对Shared memory和Global memory的atomic操作。
寄存器数量
相对于G80X，第二代Tesla将每个SP内的本地寄存器文件的数量翻倍.
提升Dual Issue
在GTX200中还提升了Dual Issue的能力

### 后记

读完Tesla的资料, 想到了同一时期另一款处理器Cisco Quantum Flow Processor,同样在原有的微码架构或者交换ASIC架构下,改用了标准C编程的多线程转发模式, 而与之配套的CPP编程框架也值得称道,无奈一个字都不能透露..

时至今日,网络界还在P4这样类似于nvidia 2002年的Sharding Language这样的东西上瞎折腾,而即便是VPP这样的东西也就是SIMD遇到Branch多的case就傻了,还有一些高端的可编程处理器,笑而不语...几分唏嘘，当DPU开始热起来的时候，又有几分感慨，连nVidia自己都不知道自己在干嘛了，当然也不算是nV，我们还可以继续把他们当作卖螺丝。

### 参考资料

IEEE Micro: NVIDIA TESLA: A UNIFIED GRAPHICS AND COMPUTING ARCHITECTURE

GPU Gems 2

NVIDIA’s GT200: Inside a Parallel Processor https://www.realworldtech.com/gt200/4/

NVIDIA GeForce® GTX 200 GPU Architectural Overview

Brook for GPUs: Stream Computing on Graphics Hardware