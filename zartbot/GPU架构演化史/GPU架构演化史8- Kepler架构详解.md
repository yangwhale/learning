# GPU架构演化史8: Kepler架构详解

> 作者: zartbot  
> 日期: 2022年9月1日 12:53  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488134&idx=1&sn=5991ce5c7378ba6cba5c13b0c8eb226c&chksm=f9960244cee18b52c00f564a11bdfb92f2341409882501336b7461207e53d4cc9d66d7f56b18#rd

---

A100/H100的禁售很大程度上是针对DPU(其实是双精度浮点Unit的意思) ,当然也使得大家需要自力更生搞好HPC了，搞DPU的人也得更加认真的学习GPU并从整个数据中心体系架构从底层XPU到互联网络到编译器、再到应用、算法通盘去考虑问题了.

从Fermi架构来看，主要是为了提升曲面细分的计算性能，所以增加了大量的Cuda Core，调度器设计复杂频率上不去，所以通过双发射和倍频模式来应对曲面细分后新增的大量三角形渲染压力。因此基于Fermi**架构的GTX480问世后就背着电老虎和热得快的标签， 所以当年一直是残血版本的GF100，仅开放了480个CUDA核心. 因为功耗的问题成为设计Kepler架构时需要考虑的主要问题，而在2012年时，支持1.5GHz(等效6Gbps)的GDDR5 相对于Fermi时代900MHz 384bits的显存， 还有Warp Scheduler和CUDA Core倍频的设计，此时又该怎么办呢？而更关键的问题是在28nm工艺下，又可以加入更多的核心了，调度器又该如何设计呢？

### Overview

Kepler**架构发布于2012年4月, 28nm工艺,大多数GeForce 600/700系列显卡均基于此架构, 当然也包括Quadro Kxxxx系列和Tesla K20/K40/K80等系列卡. 鉴于Fermi高功耗的问题，Kepler的设计三原则如下:

![图片](assets/e5529870e493.png)

以GK110为例，它包含7.1B个晶体管，15个新一代的SMX引擎，支持大于1TFLOPS的双精度浮点(FP64)计算能力,内置1.5MB L2缓存，显存支持6Gbps的GDDR5(类似于QDR，实际频率1.5GHz)

![图片](assets/2803c6e261ba.png)

对于新的28nm工艺有了更多的腾挪的空间，那么在性能和功耗之间也该做些平衡去解决Fermi电老虎的问题了:

![图片](assets/0f2f3df66b28.png)

其实频率降低也可以更好的隐藏内存访问延迟，而SMX也就被设计成了一个密度超高的处理器单元， 包含192个单精度CUDA Core(6X Fermi),64个双精度浮点单元及32个特殊函数单元和32个LD/ST单元.Tex引擎也增加到了16个，但是每个核分到的少了，L1 Cache 在GK110上为64KB，到GK210进一步增加到了128KB，而Register File为65536 x 32bits（GK110），GK210增加到了128K x 32bits.

![图片](assets/4a0e63aca127.png)

所以从整体来看，Kepler单芯片核心数为Fermi的3倍，频率从1544MHz降到1006MHz，但最终在更低的功耗下，性能翻倍(3090GFLOPS)，因为工艺带来的取舍，很聪明。

![图片](assets/9f9dbf4613e8.png)

但这也带来了一个问题，核心多了如何能够更好的完成`延迟隐藏`和`调度`？Kepler相对于Fermi改动最大的地方就在调度以及内存、缓存子系统上

### Kepler调度系统

Kepler调度系统也是两级结构，一个是全局的调度器升级为了GMU(Grid Management Unit)，而另一方面是核心的Warp Scheduler的变化

#### Dynamic Parallelism

简而言之就是原来Kernel函数只能由CPU调用GPU执行，GPU在这个模式下当一个协处理器。而支持DP后GPU在一个Kernel函数内可以再调用一个Kernel函数，这样业务的执行就变得更加灵活，甚至还可以玩Kernel函数的递归执行了

![图片](assets/878092ed3d20.png)

当然它最大的用处是在一些超算场景，计算网格分的太大计算精度不高，分的太密又有很多区域浪费，那么针对感兴趣的区域通过DP递归的方式调用更密的网格来算，这样就显得非常高效而聪明了:

![图片](assets/8827cbd01f98.png)

同时这样操作也降低了CPU对整个计算过程的参与，使得整个系统的效率也得以提高。

![图片](assets/57481419ba26.png)

![图片](assets/17deb319993c.png)

#### Hyper-Q

另一个问题是资源的有效利用上，Fermi的GigaThread Scheduler实现了一个work queue对多个Kernel函数的顺序调用，最多可以支持16个Kernel进入一个队列调度。

![图片](assets/15e328511d36.png)

但是我们注意到在每个Kernel执行的过程种，它内部的Warp对资源的利用率是不均衡的

![图片](assets/2c6de0b7a530.png)

例如上图中Stream 1  A、B、C每个计算都不会用完SM的所有资源，

![图片](assets/4c037c2b401c.png)

因此可以看到整个调度似乎资源利用率不充分

![图片](assets/ff0db91b82ed.png)

那么是否可以考虑把一些没有用的资源拿来同时执行P、Q、R，因为任务PQR和ABC之间也没有依赖，所以Hyper-Q就因此得名，相当于构建了多个Work Queue并且还可以定义Queue的优先级，然后任务拆分成Warp后由GMU调度执行

![图片](assets/aae4b952e484.png)

这样我们就可以把多个资源并行的利用起来:

![图片](assets/c7642b3c01ff.png)

所以最终硬件的使用效率提升了很多

![图片](assets/98eab0f1b5bd.png)

#### Grid Management Unit

而实现Hyper-Q和Dynamic Parallelism就需要把整个GPU的调度器改一下了，也就是以前调度器以Thread为主，而现在这一个全局调度器则变成了Grid的管理， 一方面需要对每个Grid的任务拆分并行执行(HyperQ)，同时也要接收执行过程中的Grid产生新的Grid(Dynamic Parallelism). 所以整个调度器的原理如下所示：

![图片](assets/e5f860e98dc8.png)

因此在很多任务中，得到了数倍的性能提升:

![图片](assets/43a45e786208.png)

最终Kepler基本上能够实现非常有效的算力:

![图片](assets/b32980e8dcec.png)

### SMX

SMX相对于SM把Warp Shceduler的个数从2个加到了4个，而且分组调度更加独立，单精度的Cuda Core增加到了192个，双精度浮点增加到了64个，LT/ST和SFU也都增加到了32个，寄存器文件大小也2x(GK110)/4x(GK210)于Fermi

![图片](assets/56be5fb14cb7.png)

#### Warp Scheduler

在Fermi时代有一个针对长延迟操作的寄存器记分牌以及复杂的基于硬件的跨Warp调度的算法，多端口寄存器记分牌需要去track每个寄存器，所以单个thread支持的寄存器个数仅63个，而且整个调度功耗和复杂度都特别高，而工艺有限，因此需要将执行单元2倍的频率才能跟上.

![图片](assets/53f211307c3d.jpg)

但事实上一些寄存器依赖和读写的关系，从编译器就可以获得，那么是否可以把这件事情简化呢？编译器器以在SASS上增加Control-Code的方式来显式告知Write/Read、Wait Dependency Barrier替代原有的复杂的实现，使得调度变得更加简单, 具体可以参考maxas Control Codes[1]

```
--:-:-:-:2      XMAD.PSL.CBCC track0, ldx.H1, xmad_t0.H1, track0; // Stall 6 - 4 = 2--:-:1:-:1      TLD.B.LZ.P loadX0, track0, tex, 0x0, 1D, 0xf; // Set Dep 1--:-:2:-:1      TLD.B.LZ.P loadX4, track4, tex, 0x0, 1D, 0xf; // Set Dep 201:-:-:-:1      STS.128 [writeS + 4x<0*128>], loadX0; // Wait Dep 1--:-:-:-:0      IADD track0, track0, ldx8;02:-:-:-:1      STS.128 [writeS + 4x<4*128>], loadX4; // Wait Dep 2--:-:-:-:0      IADD track4, track4, ldx8;--:-:-:-:5      BAR.SYNC 0;--:-:-:-:1      LOP.XOR writeS, writeS, 4x<16*128>;
```

#### Registers

由于不需要复杂的记分牌机制来管理寄存器了，那么单个thread支持的寄存器数量也扩展到了255个，这也对很多运算的性能提升非常明显。但需要注意的是Kepler的L1 Cache会用于寄存器溢出的处理，而Fermi采用Local RAM用于溢出的寄存器。当然编译时以查看和控制使用的寄存器数量，同时注意内存和L1的配置带来的性能影响。

#### PolyMorph Engine 2.0

在Fermi时代，PolyMorph Engine对于曲面细分业务非常重要，因此GTX580总共配置了16个，而在Kepler进行了重新设计，虽然数量减半了，但是单个时钟周期的性能基本上翻倍了，再加上频率的提升使得整体性能也提升了30%

#### Shuffle 指令

在Kepler中非常重要的一个创新是实现了Shuffle指令，通过Shuffle指令一些比较有规律的WARP内线程间通信变得非常容易的直接可以使用寄存器完成，而在Fermi架构中需要共享内存的方式。它实现了4种数据交换方式:

![图片](assets/33ae67948851.png)

PTX指令如下:

![图片](assets/8cb77c91f3bd.png)

Shuffle指令集可以用来在一个WARP内广播某个值，并且还可以和warp vote一起使用构成更加复杂的逻辑

![图片](assets/8571ba4f0049.png)

而其他的场景混洗也非常有用，例如针对并行Scan这些算法，需要在线程间传递值的函数，写起来就特别爽快，感觉直接就一个寄存器丢到旁边线程去了。

```
if (tid < 32) { temp2 = 0.0f; if (tid < blockDim.x/32)  temp2 = temp[tid]; for (int d=1; d<32; d<<=1) {  temp3 = __shfl_up_sync(-1, temp2,d);  if (tid%32 >= d) temp2 += temp3; } if (tid < blockDim.x/32) temp[tid] = temp2;}
```

当然还有Transpose这样的骚操作

![图片](assets/36bbe165547e.png)

其实对很多Scatter-Gatter类的操作，Shlf都会非常有用.

#### ATOMIC

对于多个Threads之间的并行处理，atomic也是一个非常重要的功能，Kepler进一步增加了原生的64bits对min/max、and、or、xor的原子操作支持，同时性能也有数倍的提高

![图片](assets/114a62866993.png)

所以这也解锁了一些data reduction的应用场景

![图片](assets/248caf820083.png)

### 内存及缓存

Kepler的内存子系统的升级也非常多，首先支持了6Gbps的GDDR5，眼图让nvidia显得特别骄傲，总拿出来给大家看..

![图片](assets/4921b4b752af.jpg)

带宽大了当然是好事，但内存子系统上还有更重要的功能

#### L1 Cache、Shared Memory、RO DCache

和Fermi不同，它的L1Cache会用于寄存器不够用时溢出时的缓存,L1CacheLine为128B

![图片](assets/3ee2b0eecdd8.png)

然后您可以根据实际的情况配置 而在GK210中L1 Cache和SharedMemory的共同可配置大小增加到了128KB，因此可以配置112KB/16KB  32KB/96KB  48KB/80KB等组合

```
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);cudaFuncCachePreferNone:   no preference(default)cudaFuncCachePreferShared:   prefer 48KB shared memory and 16 KB L1 cachecudaFuncCachePreferL1:    prefer 48KB L1 cache and 16 KB shared memorycudaFuncCachePreferEqual:   prefer 32KB L1 cache and 32 KB shared memory
```

另一方面共享内存Bank宽度支持4B/8B调整

```
cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *pConfig);cudaSharedMemBankSizeFourBytecudaSharedMemBankSizeEightByte
```

新增的48KB ReadOnly Data Cache ，在Fermi时代，这个Cache来自于象素引擎，有些专业程序员通过load texture的方式来利用这部分带宽，而在kepler给它变成了一个更加通用的架构，直接可以操作

![图片](assets/be997654594e.png)

采用`__ldg`函数或者声明 `const __restrict_`就可以使用了

```
__global__ void kernel(float* output, float* input) { ... output[idx] += __ldg(&input[idx]); ...}__global__ void kernel(float* output, const float* __restrict__ input) { ... output[idx] += input[idx];}
```

*Read Only Cache* 可以用于那些高带宽需求但是非常分散并且难以对齐的数据访问

#### L2 Cache

Kepler相对于Fermi L2Cache容量也翻倍了，带宽也翻倍了

#### ECC

Kepler还针对Fermi架构开启ECC**后性能下降的问题进行了处理

![图片](assets/45e5770cf626.png)

#### 图形新功能

Direct11的新功能，原来只允许Shader code bind 128个纹理，而现在可以通过Bindless的做法支持到1M纹理,同时还可以降低Draw函数的调用，使得CPU-GPU之间的交互进一步降低

![图片](assets/5536f151e01b.png)

同时还有一些关于Adaptive Vsync/TXAA/FXAA**等功能，就不再多说了

### Boost

功耗降下来了，然后就又开始玩动态超频的招数了，哈哈

### NVDEC/NVENC

增加了H.264的硬件编解码功能，支持4K的处理，而如今云端渲染似乎成了一个VPU的新赛道，nv怎么玩呢？

### GPUDirect

这是RDMA和老黄家绑定在一起, 但是十年以后再来看，对不对，是不是要改，可能都是一个很大的话题了? 当年的初心只是DMA.

![图片](assets/dd94ef16a03e.png)

### Nsight

新发布了调试和性能检测工具nsight,直到现在nvprof还一直用得很爽...

![图片](assets/eff8e235cbcb.png)

当然还有那个时候特别想买的GTX690

![图片](assets/193157f9d16d.jpg)

![图片](assets/955b8d5b7bc9.jpg)

#### Reference

[1]
Maxas Control Codes: *https://github.com/NervanaSystems/maxas/wiki/Control-Codes*