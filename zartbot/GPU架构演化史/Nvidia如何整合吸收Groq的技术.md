# Nvidia如何整合吸收Groq的技术

> 作者: zartbot  
> 日期: 2026年1月7日 03:19  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497157&idx=1&sn=d968b1f17c477ba08adeb92add92fcc7&chksm=f995e507cee26c11bad8d0f8a0352700fda51d7b7c97791062c33732d1294340c577cfce9207#rd

---

### TL;DR

前面一篇文章[《谈谈那个被NV看上值20B的Groq》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497068&idx=1&sn=833aaeb0dd37e9badf3115edac7666b4&scene=21#wechat_redirect)主要是详细阐述Groq的微架构以及互连架构, 最后仅有一小段文字在谈论Nvidia如何吸收Groq技术. 最近 X 上 @XpeaGPU 也在谈论另一个Rumor[1], NV在Feynman这一代会通过3D堆叠的方式集成Groq的技术:

![图片](assets/f8bc60968b96.jpg)

事实上我们需要从根源上来先回答几个问题:

Groq LPU相对于NVidia GPGPU的PPA优势是什么?

指令集上如何整合?

内存及互连上如何整合?

本文详细从这几个维度进行分析, 目录如下:

```
1. Groq LPU PPA优势的来源1.1 数据路径1.1.1 320B Vector1.1.2 片上网络没有了复杂的2D路由1.1.3 Streaming RF1.1.4 状态机1.2 控制路径1.2.1 指令效率1.2.2 调度2. Nvidia如何整合吸收Groq的技术2.1 没有Groq技术的3D Package方案2.2 Groq LPU 3D Stacking的优势2.3 NV整合Groq LPU2.3.1 Hardware视角2.3.2 Software视角
```

## 1. Groq LPU PPA优势的来源

### 1.1 数据路径

#### 1.1.1 320B Vector

在Groq LPU中, 数据路径采用320B的向量作为一等公民存在, 如下图所示:

![图片](assets/b1cd82de362f.png)

320B的构成由 20 个 Superlane, 每个Lane 4个Bank, 每个Bank 4Bytes组成. 在整个片上数据流向为东西方向, 如下图所示:

![图片](assets/e84ebc1bbc57.png)

然后针对NVidia GPGPU, 由于考虑到SIMT执行, SM到Global XBAR位宽为32B, L2Cache的位宽为128B. 在整个内存访问的效率上Groq LPU会更高一些, 并且整个数据路径上位宽是对齐了的.

#### 1.1.2 片上网络没有了复杂的2D路由

所有的数据流向为东西向, 因此整个片上数据流向和整个Dataflow的生命周期管理是相对简单的. 不需要考虑片上2D路由的复杂度, 对于数据潜在的需要南北向流动的地方由SXM进行Shuffle/Transpose. 有一个潜在的好处就是不太需要考虑数据路径上的bank conflict相关的问题, 也不需要处理一些复杂的swizzle. 整个数据链路上LD/ST的位宽也很大

然后很巧妙的是控制的指令流刚好是南北方向广播和流动的, 整个芯片看上去数据流的路由是非常干净的.

![图片](assets/73e1940fdd4b.png)

#### 1.1.3 Streaming RF

相对于GPGPU, 需要在GMEM到SMEM, 然后TMEM和RF之间多次显式倒换内存相比, Groq LPU采用Streaming RF的方式:

![图片](assets/2577fa8d3706.png)

由于是直接结果在RF中, 下一级的Function Unit在下一个cycle即可直接处理, 整体来看内存搬运的功耗和延迟都小了. 整体芯片上便不需要复杂的NOC/内存层次的处理.

#### 1.1.4 状态机

整个在推理的过程中, 通过一个Stream来定义状态机, 从SRAM中加载, 到经过整个MXM/VXM/SXM处理后写回到SRAM构成一个完成的状态变更. 这样有一个好处, 对于中间发生的SDC或者MBE, 相对于GPGPU做状态恢复会容易一些, 从宏观的视角来看, 由于没有太多控制面和数据面的副作用影响, 可以简单的Replay一下整个Stream的指令即可.

另外, 对于故障后的FRU替换, 热迁移也相对容易, 仅需要将SRAM中的220MB数据迁移到另一个芯片即可.

### 1.2 控制路径

#### 1.2.1 指令效率

一个320B的Vector构成了整个芯片的一等公民, 然后在东西方向构成了一个VLIW指令, 并通过在南北方向广播构成的SIMD. 相对于NVidia SIMT也大大的提高了指令的效率, 因此片上不会像NV那样每个Thread都有相应的译码处理/指令发射处理. 而是由统一的Instruction Control Unit处理.

然后整个指令集里, 也针对LD/ST一些高维张量在MEM FU上做了一些处理, 避免重复的地址生成计算.

![图片](assets/af07e584a860.png)

#### 1.2.2 调度

由于整个系统构成的确定性, 指令完成的cycle是完全可确定的, 因此就不需要复杂的调度, 也不需要对一些复杂的异步处理隐藏pipeline. 而在NVidia GPGPU中, 还需要考虑指令的完成时间不确定性, 对于寄存器的排布由于这些不确定性其实也挺难的, 有些情况下Register Spill难以避免.

## 2. Nvidia如何整合吸收Groq的技术

在 X 上 @XpeaGPU 的原文如下, 主要阐述了SRAM的尺寸缩放停滞不前

![图片](assets/27708fac6c6c.png)

然后3D-SRAM堆叠似乎是条路? 事实上和NV GPGPU的架构融合还有更多细节的地方需要考虑, 那么我们分别从软硬件的视角展开叙述一下.

### 2.1 没有Groq技术的3D Stacking方案

首先我们来谈谈假设没有Groq技术的3D方案, 最基本的一个处理是能否通过3D堆叠的方式来构造一个支持2SM的CGA?  Logic-on-Logic 的3D芯片供电/功耗控制/散热都是一个非常难以处理的事情, TSV的穿孔也是一个比较难处理的地方.

那么紧接着类似于AMD这样的X3D, 或者说Memory-on-Logic的方式是否可行?

### 2.2 Groq LPU 3D Stacking的优势

Groq自己的一页资料可以看到, 由于确定性的处理, 对于功耗的优化有很多优势, 可以Cycle粒度评估整个系统的功耗和散热

![图片](assets/99499a6a992a.png)

使得Logic-on-Logic的3D Stacking成为可能

![图片](assets/10e0489b48df.png)

另一方面Groq也可以通过自身来构建3D-SRAM的扩展, 以及通过调整Superlane的位宽来支持不同规格的芯片, 并且支持chiplet甚至IP输出:

![图片](assets/26460925aa47.jpg)

### 2.3 NV整合Groq LPU

#### 2.3.1 Hardware视角

实质性的问题是我们把Groq LPU当作一个Process Near Memory的SRAM, 以这个视角如何整合? 利用Groq LPU确定性的执行, 我们可以将其作为一个增强的TensorCore.

![图片](assets/47a76338561d.jpg)

从数据路径上来看, 可能的选择是采用16个SuperLane 对齐NVLink 256B位宽. 然后LPU Die和Feynman Die之间都需要有SRAM的连接来避免LD/ST的不确定性. 同时也需要额外的在CUDA端 MBarrier 和LPU的 SYNC/NOTIFY 指令的交互处理.

对于NV GPU微架构的一个显著的变化或者需求便是在LPU和GPU HBM的地方需要一个确定性的Buffer空间(DATA FIFO), 避免读写DRAM带来的不确定性.  因此可能需要一个Global的L2Cache的位置开出一段SMEM空间.

![图片](assets/0d3e73e2c945.jpg)

另一方面, 我们考虑到Feynman的封装, 似乎可以利用TSMC的SoW-X, 那么在Groq LPU Die上也可以通过多颗芯片D2D互连

![图片](assets/7ec0b40a75c6.png)

整个芯片估计可以提供Groq LPU SRAM的容量为11GB~16GB. 同时为了保证Groq LPU在芯片内的确定性, 需要在设计时, 充分利用NVLink C2C, 那么同样需要专用的一片SRAM来构建 Data FIFO.

对于跨芯片的(例如两个Feynman SOW的互连), 沿用NVLink Switch其实增加的不确定性是可控的.

#### 2.3.2 Software视角

从软件的视角来看, 首先从算法工程师的角度, 他们已经熟悉numpy/pytorch那样的基于张量的计算范式了, 而且workload来看传统的HPC使用的微分方程数值解一类的应用在SIMT上的优势对于当前的广泛的workload来看占比已经很小, 其实最近一系列的DSL, 包括Nvidia cuTile在内, 都有一个很明确的趋势: Tensor Tile作为一等公民, 即便是一个标量也是一个 tile<T> 的类型. 然后进一步把内存模型memory_ordering_semantics 和 memory_scope 属性引入, 关于内存模型可以参考[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect).

因此, Groq LPU编译器可以生成LPU相关的Kernel, 对于LPU Kernel的数据依赖, 例如需要从GMEM的HBM加载的数据, 进一步由TMA生成. 对于一些难以处理的算子, 再Offload到CUDA Core上.然后整体上CUDA Core通过MBarrier异步和LPU Die进行交互, 根据MBarrier让CUDA Core issue LPU的指令. 或者运算完的数据通过LPU SXM拷贝到CUDA Core的SMEM中. 同时SMEM也可以作为一个Buffer为LPU提供访存的确定性.

假设LPU采用16 superlane对齐256B的位宽, 那么CUDA Core可能需要两个SM来处理数据, 这里CUDA Core可以继续使用原有的CGA来处理.

参考资料

[1] 
Rumor Feynman with LPU: *https://x.com/XpeaGPU/status/2005128578045018500*