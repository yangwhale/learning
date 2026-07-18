# GPU架构演化史11: Volta架构详解

> 作者: zartbot  
> 日期: 2022年9月4日 06:38  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488243&idx=1&sn=25b7eacd13e0daf339462ba5fc6ffdba&chksm=f9960231cee18b27ec3c7cfe384214396e56af4fc79c35996760c3e9f30944236a43c8011caa#rd

---

2016年pascal发布不久, Google便发布了TPU. 虽然有些数据还是有争议的，但是确实给NV带来了一些压力。而Volta架构的开发在2013年就有消息，熬过了4年，终于在2017年发布。它还是老黄家第一个专门为深度学习加速的计算卡。当然它的背后还有和IBM Power9一起搭建的Summit超算.  双精度浮点性能达到7.8TFLOPS，在很多HPC**应用中,性能相对于Pascal提高了1.5倍

![图片](assets/11ad71ede35e.png)

除此之外，NVLink和HBM2的带宽都增加了不少，支持了多进程(MPS**)，还有一个非常重要的创新就是每个Thread有了独立的PC和Call Stack，因此支持了一种更新的线程协作方式Cooperative Groups.而更加关键的是利用Tensor Core和FP16使得深度学习浮点计算能力提升了12倍.

![图片](assets/3e37c1c42b57.png)

### Volta SM

这一代使用了12nm的工艺，相对于Pascal又可以堆料了，调度上还是按照GPC、TPC、SM的方式，GPC还是6个，单个GPC包含了7个TPC，累计14个SM，完整配置的GV100包含84个SM，而在发布的V100加速卡中只支持80个SM

![图片](assets/54d6f32cfc8c.png)

SM的结构变换比较大,原有的CUDA Core被拆开，变成了相对独立的 64个FP32 unit和64个INT32 Unit，32个FP64 Unit 4个像素单元

![图片](assets/308c4e369bda.png)

原有的Pascal单个SM有两组Cluster,每组一个Warp Scheduler带有2个Dispatch Unit，而在Volta上直接把它拆开，增加Warp Scheduler变成了4个独立Processing Cluster，同时也增加了L0 I-Cache，当然这么做的原因是在每个线程上支持了PC使得线程协作更加简洁灵活，具体内容在讲CG时详细叙述。

![图片](assets/e8970f14a8c0.png)

和前几代的详细对比, FP32和FP64性能都有1.5X提升，TDP几乎相同，但Volta还有一项改进就是动态的功率调整可以使其在维持50%TDP时还能获得70%~80%的性能。

![图片](assets/f5c8e6dce528.png)

在Maxwell时代因为功耗问题砍掉L1 Cache和Texture Cache共享，而在这一代又加回来和共享内存一起，并且把Kepler**时代的L1$ 和 共享内存 可配置的能力加了回来,L2 Cache也增加到了6MB，有些时候架构师为了解决问题总会不停的兜兜转转，对和错要站在时间维度来看因果。

![图片](assets/0dd05aa44284.png)

而新加入的Tensor Core是一个比较烧脑的话题，为什么需要FP16和4x4，而这个问题又牵扯到一些线性代数，以及图形卡光追带来的性能影响需要通过DLSS补的问题，当然新的Tenscore也需要一系列指令协作，我们把它放到整个文章的最后面. 其实每一个架构师都很聪明，了解并理解他背后的妥协也很不容易。

#### NVLink2.0 &HBM2

在Volta上支持了NvLink 2.0，单个Channel提升到了25GB/s，总Channel数增加到了6个，所以双向来看整个GPU的带宽增加到了300GB/s

![图片](assets/b647ffc3b573.png)

NVLink2.0还专门针对了Power 9的超算部署，新增了对UVM的 Copy Engine Cache Coherence、Address Translation Service(ATS**)的支持，我们在UVM的部分介绍:

![图片](assets/2423b766753c.png)

而HBM2的带宽也扩大到了900GB/s

![图片](assets/edf25b44a835.png)

### UVM

在Pascal上新增的Unified Virtual Memory主要问题是解决程序员在异构程序开发时的心智负担，但效率是一个问题:

![图片](assets/14f8ba8d73a6.png)

其主要原因是对于PageFault的处理两者相对独立，特别是针对一些深度学习训练的数据可能会出现频繁的换页操作

![图片](assets/6b3f5f0ce4f3.png)

归根结底UVM就是GPU的IOMMU，那么你会怎么办呢？

#### Access Counter

第一个做法就是，有些东西访问不频繁就懒得换页了，直接Remote Access好了，对于需要频繁访问的，添加一个计数器，然后优先换页到本地

![图片](assets/307dc0ad027d.png)

#### NVLink 2.0 UVM

当然针对X86和PCIe总线也就只能这样了，牙膏厂也过了好多年才意识到靠I/O限制多卖CPU的策略失效了，才提出CXL。而这些东西在NVLink 2.0和IBM Power 9合作的时候都支持了，主要就是Cohenrence、Atomics和ATS

![图片](assets/2496e5589e3c.png)

Power 9可以直接Access GPU的内存，并且天然支持了Atomic**，爽不爽?

![图片](assets/32e895e297c9.png)

而更关键的是ATS，地址翻译功能，使得整个Page Table可以索引到不同的物理内存空间：

![图片](assets/5464c33fb80a.png)

不光是对CPU-GPU通信，UVM ATS对多卡集群之间共享内存的操作也变得简单了,既然有Coherence和ATS支持，对于远端GPU显存的访问Cache到本地再放一个Copy也没啥问题了.

![图片](assets/1046201544b3.png)

Nv最后对这个总结的很好,一步一步这么走过来也花了好几年的时间

![图片](assets/041516214be3.png)

### 多进程服务(MPS)

在Pascal时代之间，各个Process是按时间顺序调度了，进程之间完全独立，资源利用并不充分，而Pascal虽然增加了Hyper-Q的调度方式，但是资源并没有隔离

![图片](assets/3b53c4dc3488.png)

而在Volta时代，增加了MPS使得硬件资源完全隔离开了，所以这也是Volta需要拆开Warp Scheduler的一个原因.

![图片](assets/d8ff846c059f.png)

而这样的隔离您可能觉得对训练好像并没有什么卵用，但是考虑到多个模型进行推理的生产部署以及超算中大量的MPI应用,这些功能是不错的,所以很多时候不要单独的以AI应用来看待一些架构的设计.而后面更加灵活的解决这个问题的是MIG以及到了Hopper支持的SRIOV VF隔离.

### 独立的程序计数器(PC)

回到一开始，我们来看第一代的Tesla架构如何实现SIMT的，每个Thread都共享一个PC

![图片](assets/85244ed1f594.png)

然后根据分支的情况构建一个NextPC和Reconvergence PC及Active Mask的表，然后根据这个表执行分支，这样来看分支执行时间较长:需要一个block(指程序花括号内的内容，和调度的Block概念不同)内执行完了，再去执行另一个分支。

![图片](assets/ab89b8bd408b.png)

还有一个问题是ActiveMask的开发非常痛苦，这种bitmap操作很容易搞错出现bug.而且线程间消息通信也是一个麻烦事，例如我们需要通过Atomic去做一些事情就会产生死锁

![图片](assets/efd3ef47fbb6.png)

如果每个线程都有自己独立的PC，然后微观上各自玩各自的，在分支结束后Warp level 同步一下是不是更好呢？

![图片](assets/e789689d8aab.png)

这样两个分支的内多条语句就可以交替执行，效率更高，然后在WarpLevel做sync也容易，对齐所有的PC就好，这样可以保证后面的连续访存继续聚合

![图片](assets/ffa18c383441.png)

当然NV会说以前Atomic在SIMT死锁的问题是因为它们仅支持Lock-Free的算法，而现在随意了

![图片](assets/19989a5d88ec.png)

这不专门还弄了一个带lock的链表插入算法做例子呢

![图片](assets/b578ffba7cad.png)

#### Cooperative Group

独立的PC带来另一个好处是线程组调度更加灵活了，nVidia带来了CG的概念,同时分组聚合都更加简洁,借助C++的模板特性对于资源对象的封装也更加简洁:

![图片](assets/a1e38f1d1a12.png)

然后针对Thread group都实现了一个相同的接口

```
void sync(); // Synchronize the threads in the groupunsigned size(); // Total number of threads in the groupunsigned thread_rank ();// Rank of the calling thread within [0,bool is_valid ();// Whether the group violated any API constraintsAnd additional thread_block specific functions:dim3 group_index ();// 3 dimensional block index within the griddim3 thread_index ();// 3 dimensional thread index within the block
```

大家不用再像以前那样计算ThreadIdx、BlockDim找到tid，更像MPI那样直接拿自己的thread_rank，同时对线程进行更细颗粒度的划分也非常简便，

![图片](assets/7b134cbe7c24.png)

CG配合Dynamic Parallelism在解决一些空间递归细分的算法，例如VXGI中的体素化算法时，会很大程度的降低开发者的心智，不用去思考复杂的ActiveMask了.这是一个我非常喜欢的功能.

另外针对常见的一些并行算法，例如归约它提供了标准的函数模板，使得操作变得更加简单

![图片](assets/51d5f018d27c.png)

同时针对每个tile之间的数据通信，更新了以前Warp shfl的支持，还添加了warp match的操作

![图片](assets/52f35b3a4b14.png)

所以整个完整的Reduce算法就非常容易写了

```
#include <cooperative_groups.h>#include <cooperative_groups/reduce.h>using namespace cooperative_groups;namespace cg = cooperative_groups;typedef double FLOAT;__global__ void ReduceSum(const double *x, double *y){    __shared__ double sdata[256];    /* cooperative_groups */    thread_block g = this_thread_block();    thread_block_tile<32> tile = tiled_partition<32>(g);    unsigned int tid = g.thread_rank();    /* load data */    double beta = x[tid];    /* reduction */    sdata[tid] = cg::reduce(tile, beta, cg::plus<FLOAT>());    /* sync */    g.sync();    /* reduction partitial results using shared mem */    if (tid == 0) {        *y = sdata[0] + sdata[32] + sdata[64] + sdata[96] +            sdata[128] + sdata[160] + sdata[192] + sdata[224];    }}
```

而Coalesced Group解决了原来分支情况下复杂的active mask处理，程序更加简洁统一

![图片](assets/652078583d8d.png)

而MultiGrid Group还进一步解决了多卡训练的资源抽象问题

![图片](assets/0c59c36661f4.png)

### Tensor Core

在深度学习中有大量的数据*参数+偏置的运算，然后通过激活函数到下一层神经网络。而另一方面，在图形处理中也有大量的矩阵计算，例如空间平移旋转等，或者基于颜色RGBA的插值等计算，当然nvdia tensor core的设计过程中不光考虑到深度学习，也考虑到了光追实现后原始分辨率低，抗锯齿这些业务需要借助DL模型去提高分辨率的场景，即DLSS. 那么又考虑到它很多颜色的处理值域到65535就够了，那么用FP16不就正解了么？当然FP16也考虑深度学习过程中的梯度消失的场景，所以尾数多.

第一代Tensor Core设计成了4x4x4.怎么理解这个TensorCore的 MxKxN 呢？我们来复习一下线性代数的矩阵乘法

![图片](assets/0e3b578dfb83.png)

其中:

如果说用CPU来做，可以写成三重循环,但是很遗憾最内侧的循环每次都要重新加载数据

```
#内积矩阵乘法for(int m = 0; m < M; m++) {    for(int n = 0; n < N; n++) {        for(int k = 0; k < K; k++) {            C[m][n] += A[m][k]*B[k][n];        }    }}
```

但是考虑到内存访问的局部性，我们可以更改这三个循环的执行顺序，把K放到最外面，如果内层可以并行通过向量并行计算

```
#外积矩阵乘法for(int k = 0; k < K; k++) {    #parallel    for(int m = 0; m < M; m++) {        for(int n = 0; n < N; n++) {            C[m][n] += A[m][k]*B[k][n];        }    }}
```

当然还有一种做法是脉动阵列，也就是Google TPU的做法,这样便于物理实现。

![图片](assets/db8572774b13.gif)

具体内容可以参考资料 Matrix Multiplication: Inner Product, Outer Product & Systolic Array[1]

当然对于内积、外积、脉动阵列各自的优势，业界自然会有自己的选择。而我们不得不再次提醒nVidia因为有图形的业务以及已有的SIMT架构，内积对于他们来说是一个值得的选择，而这个话题以后有时间专门研究好了再另写一份笔记。

回到Volta的Tensor Core上，它实现了一个4x4x4的Core，并且支持FP16混合精度

![图片](assets/668b024aa1e5.png)

还记得这个系列开头SGI实现的Geometry Engine么？在1980年的时候就提供了一系列的指令集，包括操作寄存器的LoadMM、MultMM、PushMM、PopMM、SotreMM来处理4x4矩阵运算，所以Tensor Core嘛，也就是那么回事，兜兜转转30年又绕回去了。另外你也就明白为什么nVidia在后面若干代Tensor Core上一定要保留4这个维度了吧？以及为什么要很尬尴的保留FP16了，图形业务不得不考虑。你问我证据是啥，后面Turing的时候为了推出GTX1650，砍掉Tensor Core老黄为啥要加一堆FP16？

![图片](assets/9ff6f859dc76.png)

其实图形学VR+ AI然后针对人眼的延迟容忍打时间差，这里面有太多的可以玩的东西，点到为止. 对于tensor Core的内部实现可以参考一篇文章 < Modeling Deep Learning Accelerator Enabled GPUs > 内部有些推测:

![图片](assets/30079f5e65f6.png)

还记得前面的Cooperative Group么，基于Warp Level的同步和线程分组，这里直接就给出了一个非常有用的例子:

![图片](assets/fb23bb88580c.png)

内部Thread分组的猜测如下:

![图片](assets/b07d0ffe2a62.png)

整个计算也非常简单的代码:

![图片](assets/fef032d42cde.png)

而这样一搞的结果:

![图片](assets/d284603e2131.png)

最后一个总结吧:

![图片](assets/01ad80101a62.png)

当然针对Tensor的运算，我一直认为有某种简化的方式，深度学习功耗太高，所以个人而言在闲暇时间慢慢的准备去读一些张量范畴的书，有些问题可能还需要从根源去解决，虽然特别难..

#### Reference

[1]
Matrix Multiplication: Inner Product, Outer Product & Systolic Array: *https://www.adityaagrawal.net/blog/architecture/matrix_multiplication*