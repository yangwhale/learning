# GPU架构演化史10: Pascal架构详解

> 作者: zartbot  
> 日期: 2022年9月3日 12:29  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488195&idx=1&sn=ad3cc222fac42fefc4dc9362f6f3b93b&chksm=f9960201cee18b17e20a0fc7b505b97998527ac6cad65998597c5c7cfb4d353c9faaca4da10c#rd

---

2014~2016这段时间, AI开始逐渐热起来了, 2015年底TensorFlow的发布和2016年AlphaGo战胜李世石都具有划时代的意义,一个大幕也拉开了. 而恰逢那时，好友开始做第四范式，而我自己利用闲暇时间和别人合作完成了一个量化基金的研发。从个人视角来看，对于深度学习的认同度并不是很高，金融对结果的可解释性和确定性的要求决定了它很难有施展的空间。但工业界来看，语音识别、图像识别这些领域对结果要求不那么高的场景似乎可以做点事情了..

回到2014年发布Maxwell以后，新的内存形态也逐渐商用，HBM和HMC的技术路线选择也是一个非常有趣的话题。当年Juniper为某一代Trio选择了HMC，Intel在Xeon Phi上也采用了HMC，确实HMC对于硬件工程师来看很美，但是Xeon Phi早已烟消云散，而Juniper也不得不因为HMC面临某一代断货提前停产的压力。而我们也在设计第三代QFP处理器的时候有同样的争议，最终搁置争议选了DDR4，一方面因为已经够用了，另一方面还是因为价格的原因，即便是地主家的粮很多也要省着用... 而nvidia和AMD选择了HBM. 当然nv在图形卡业务上考虑成本和容量还多用了一代GDDR5X，后面的故事就不多说了，各种网络芯片都在HBM的道路上狂奔，例如Silicon One、LightSpeed，我后面在做NetDAM的时候第一个需求也是要用带HBM的FPGA...我也蛮期待带有HBM的Intel SPR可以做一些很有趣的事情，

nVidia可以在这个时候选择14nm、16nm FinFET工艺了，作为一个架构师该如何选择？应用的需求也出现了分化，对于计算业务来自于HPC对于FP64浮点的需求，Maxwell考虑图形应用和功耗砍掉了很多FP64单元，深度学习训练的蓬勃发展对算力的需求带来了对低精度浮点的需求，同时还有一些多卡之间高带宽通信的需求。当然对于图形业务也提出了更多的需求，多屏幕，曲面屏和VR使得图形渲染上有一些更多的考虑，光追VXGI的算法对于算力的要求还是特别大...而nVidia在Pascal这一代深刻的诠释了：成年人的世界没有选择，都要！

但是客户还是有选择的， 记得后面公司内部我和另一个团队都在做基于AI的网络分析，同事的团队选了一块Tesla P100，而我选择了4块GTX1080Ti, 因为我Pascal时期支持的INT8更好玩，更适合网络高性能高吞吐的推理，而很多网络相关的业务是不需要高精度浮点的。具体关于网络和AI的结合，有些东西就不多说了，现在Cisco也发布了Predicative Network，华为前几年也有一个基于RL的buffer调整算法，今年还有一个NLP类似的配置抽象拿了BestPaper Award，另外Denis代表阿里在SIGCOMM22上的主题演讲也很赞的，期待这一块能有更多的成果。

### Pascal

Pascal架构其实只是一个统一的名字，针对图形和计算产品线划分了两种核，基于计算业务的GP100，也就是Tesla P100卡首先在2016年4月5日发布。而基于GP104的GTX1080在稍后的5月17日发布.

![图片](assets/2c5c66a2eb1c.png)

GP104针对于图形业务，但是相对于Maxwell的SMM，你会发现PolyMorph Engine不见了，这个稍后会讲到，整体而言和Maxwell没啥区别的，寄存器文件数量相同， 共享内存从64KB加到了96KB，GP100针对于计算业务，相对于GM104采用4个Processing Block合封，它只有2个Processing Block封在一起，但是比较起来Instruction Cache更大，共享内存更大，寄存器文件也翻倍了，同时还把Maxwell砍掉的fp64 unit加了回来，从Kepler的CudaCore:DP Unit 的3:1变成了现在的2:1.当然还增加了在一个Cuda核内同时处理两个FP16的能力。

从整体架构来看，GigaThreadEngine的调度器抽象层次又封装了一层，在GP104中，采用一个SM加上一个PolyMorph Engine 3.0构成一个TPC(Texture Processing Clusters),然后5个TPC共享一个光栅化引擎构成一个GPC

![图片](assets/c0b2509d4501.png)

GP104有20个SM，累计2560个Cuda Core，同时得益于16nm工艺，die size反而变小了，伴随着频率的提升，浮点运算能力和显存带宽都接近翻倍，整体功耗稍微大了一些,当然显存出于成本的考虑用了GDDR5x(10Gbps)而没有选择HBM.

![图片](assets/9bc91d8889ce.png)

而GP100主要是应对计算业务，所以PolyMorph Engine及Raster Engine都不需要了，它采用将两个SM构成一个TPC的方式，也是5个TPC构成一个GPC，使得调度器结构上跟GP104近似，但总共的GPC数量增加到了6个，累计60个SM,但是发布的P100只使用了56个SM

![图片](assets/d0d1107800df.png)

在GP100上使用了8个512bit的内存控制器，总共4096 bits连接HBM，内存带宽增加到了720GB/s, 相对于前一代提升了3倍，同时还提供了NVLink多卡互联的技术，相对于以前的K40、M40，Tesla P100运行频率内存带宽都得到了极大的提升，L2 Cache也增加到了4MB，FP32的峰值浮点能力相对于Kepler也翻倍了，FP64更是相对于Kepler提升了4倍。其实针对商用的卡都有自己的Refresh周期的，所以商业角度上来考虑4年一次的更新更加有道理，但需要注意的是GP100整体功耗也大了不少. 另一件事情是Google TPU的发布，从这一点上来看，说Volta是应对TPU仓促发布的，倒不如说P100是一个试水的产品，而Volta内部的Tensor Core和Cooperative Groups怎么看都不像是仓促发布的东西。

![图片](assets/d2241c2b70ea.png)

接下来，我们分Compute、Graph两块来介绍Pascal的新功能.从计算的角度来看，这算是专门针对深度学习构建加速器的元年，以至于Pascal的Whitepaper附录里还有一章< Deep Learning in a Nutshell > 而从图形来看，这也算是VR的元年。

### Compute

在Pascal上补齐FP64的支持是HPC那些应用所必须的，当然还有一些别的创新， 这个图算是一个总结：

![图片](assets/58a9ccf58ecf.png)

FP16提升了浮点运算的性能，NVLINK提升了GPU之间通信的带宽，HBM2提高了显存带宽，Unified Virtual Memory提升了内存寻址空间

![图片](assets/41c33e56562e.png)

#### FP16

面对深度学习对算力的需求，FP16就这样产生了,当然我还觉得FP16对图形业务也有帮助，毕竟值域和精度看上去都挺合适的。

![图片](assets/256069f812de.jpg)

不过我个人不太喜欢FP16，因为相对于BF16虽然精度高一些，但是相应的Fraction位数也多一些，乘法器占用的空间也大一些，感觉得不偿失，而我更喜欢BF16从数学的角度，我就收敛的时候按照两个epsilon做个分支就好，而FP16要考虑太多溢出怎么办的问题，烦死人...另一方面本来转换成FP32更容易，所以根据一些中间计算的delta来确定是否要扩展成32位，硬件上扩展或者截尾的操作也更容易。当然对于浮点还有一些更加有趣的玩法，例如：

![图片](assets/3f13ded1708c.jpg)

这个话题和在图形卡中的纹理压缩都是一个非常值得深入研究的，而对于混合各种精度的乘法器的研究也挺好玩的。回到正题上来，FP16本身也算是一个不错的尝试，也算是为后面Volta探了一下路。另一方面我不确定是否nVidia还有对图形的考虑，有些图形的算法FP16刚好在精度和值域上都能满足要求,例如HDR..当然在FP16支持上nVidia还是区分开了GP100和GP104,刀法果然厉害.

![图片](assets/9c491c0ed932.png)

#### 内存子系统

HBM2没什么好多讲的，新的封装，更大宽的总线，更大的带宽, 但容量有些限制，访问延迟依旧在那里，所以内存访问上有一些技巧。

![图片](assets/080d15d02190.png)

但是有一个更有趣的问题值得我们来关注。GP100有14MB的寄存器文件，而L2Cache只有4MB，再来仔细看一下SIMT，GPU运算上并不是像CPU那么latency critical，所以寄存器文件都可以做成Single-port的，而GPU本身核多频率相对低，然后又有大量的线程可供调度来隐藏访存延迟。以每个Thread需要32个寄存器来看，单个SM支持256KB寄存器文件，那么折算出来就可以支持2048个Thread. 然后HBM内存带宽有720GB/s，那么就可以有2us的延迟容忍。那么基于Instruction level 的抢占需要置换大约20个寄存器，看上去也不难实现了吧，这对后面支持虚拟化和QoS也提供了支持。同时由于寄存器数量这么大，单个Thread可以支持到255个寄存器，算法腾挪的空间也大多了，大量的循环展开也可以直接干了.

当然对于抢占这事在GP100上寥寥几句，而在GP104上就详细介绍了Compute和Graph抢占的能力，具体后面说。

#### NvLink

PCIE慢，所以需要更快的总线互联，我们在第一代QFP设计时就考虑到了多芯片互联，后来发布了ESP100(双芯片互联)和ESP200(4芯片互联).NVLink也类似，私有总线嘛，自己玩的开心就行了，NvLink1.0分为4组共计32根信号线，每组可以提供单向峰值20GBps(双向40GBps)的能力，累计160GB/s

![图片](assets/19ef6a8d849d.png)

4组NVLINK正好组成8卡互联,

![图片](assets/4d41a728f332.png)

然后这种封装形式也就构成了DGX-1系统

![图片](assets/e834641ec4ac.png)

NvLink采用定长的128bit Flit，最大支持18个 flits，即256Bytes的数据通信，报文格式如下:

![图片](assets/58e175ab569e.png)

CRC为25bit，紧接着83bits包含了Transcation Layer的Rtype、地址、流控令牌、Tag等，剩下20bits为DataLink Layer包含ack的id、packet长度application tag等信息。然后有一个可选的Address Extension和Byte Enable Flit（用于Atomic一类的操作）然后接下来是一些DataPayload，没有什么Tricky都是根据自己业务来的，NVLink在IBM Power处理器上支持了一段时间后来也没了，只是最近Grace-Hopper变成了NVlink C2C后面有空再说，但是个人可能对于这块更看好UCIe。整个传输嘛就那么回事：

![图片](assets/d14a8a9214ab.png)

#### Unified Virtual Memory

最早的Tesla是每一块内存都有自己的地址空间，然后到了Fermi把GPU上的地址空间统一了， 这样C和C++的一些指针操作也可以玩了，后来在2011年CUDA 4.0引入了Unified Virtual Address，把CPU和GPU的空间统一了，因此支持了Zero Copy，CPU上固定的内存空间可以直接被GPU访问了。

而UVM的主要目的是尝试着去降低开发人员的心智负担，传统的CUDA程序开发需要程序员显式的拷贝内存，然后执行Kernel函数，虽然通过Overlap可以一定程度上的隐藏延迟，但是开发时也需要操心太多的东西去优化访存带宽。UVM便用来尝试着去解决这个问题

![图片](assets/5c354c319785.png)

支持UVM以后和原来的纯CPU代码执行就没有太多的区别了，否则您需要CPU malloc一次，然后GPU CudaMalloc一次，然后再CudaMemCpy一次，烦死人，而UVM从代码上看更加简洁了:,共享同一个指针，对于原来的程序侵入性也小了很多

![图片](assets/95423dc109b7.png)

UVM支持49bits的VirtualAddress，PageSize 2MB，支持GPU page fault能力，

![图片](assets/479ceadf581f.png)

但是受限于CPU-GPU的带宽，page fault migration大概需要20us~50us的时间，那么这就需要做Prefetch了

![图片](assets/d02f9db022b8.png)

这样的做法对于AI训练不一定有效，因为训练数据本来就可以很好的控制，而似乎对那些传统的支持OpenMPI的超算业务来看，可能会使得他们应用迁移更加容易。

#### Atomic

当然UVM还使得Atomic操作变得可以GPU-GPU(Nvlink保证)和GPU-CPU(PCIe with Software)可行了. UVM在Volta中还进一步增强了一些功能，我们后面再来看

#### CUDA Lambda

另一个可以借助UVM的功能就是在CUDA上实现了C++的匿名函数，原来您可能需要显式的去调用一个Kernel，而现在可以如下:

![图片](assets/e7b1856e14b8.png)

对于有没有CUDA加速的平台，一个条件编译就完事了

```
#ifdef USE_GPU for_each(counting_iterator<int>(0), counting_iterator<int>(n), [=] __device__ (int i) {#else for (int i = 0; i < n; i++) { #endif     z[i] = a * x[i] + y[i]; });
```

### Graph

白皮书很有趣，说以前功耗原因要降频，但是现在16nm FinFET工艺上去了，频率也可以提上来了，核心频率从Maxwell的1.1GHz提升到了1.6GHz，Boost到了1.7GHz，同时内存的频率也提升了，针对图形业务的几个技术如下：

#### 内存压缩和GDDR5X

作为一个架构师并不是一定要什么都用最好，够用就成。Pascal继续优化了内存压缩算法，迭代到了第四代:

![图片](assets/f66f6314f21c.png)

再加上一些Tile Based Cache算法，使得整个图形渲染上都更容易被Cache(下图中紫色的部分是被Cache的部分)

![图片](assets/3391cca97568.png)

所以这样算下来，GDDR5X内存频率提升了40%，再加上压缩带来的20%提升，累计有效带宽提升相对于Maxwell也有1.7倍了，完全够用了，因此在这一代上并没有很激进的去用到HBM.

#### Compute Preempt

其实前面已经讲了Register和Cache大，使得基于指令的抢占可行了。

![图片](assets/86519e3b98d4.png)

当然对外还要包装一下的，这不就成了Dynamic Load Balancing了:

![图片](assets/ac8410f357b2.png)

对于游戏引擎开发而言还取了一个好听的名字叫异步计算(Asynchronous Compute)。这样的好处是对于VR可以通过Compute裁剪，对于PhysX这些物理效果的也可以并行运行。所以降低了延迟:

![图片](assets/f41107f29d7c.png)

#### 并发多投影(SMP)

Simultaneous Multi-Projection Engine也是VR和多屏环绕场景中特别有用的一个功能，它是在PolyMorph Engine中位于Geometry流水线和光栅引擎之间的一个特殊的硬件单元。

![图片](assets/e9a3a1065c94.png)

其实就是一个viewport transform的升级，针对双眼或者多屏幕的FOV做多路输出

![图片](assets/aa3e3beee000.jpg)

![图片](assets/3e10376a0090.png)

针对VR场景则可以降低Geometry Engine的工作负担，只是最后在FOV上多做几次变换就好了

![图片](assets/17ea17387ed6.png)

同时针对VR的特点，还进一步增加了VR Audio

![图片](assets/17a834ff076b.png)

然后采用PhysX更改被渲染模型的物理形态，但是我并不喜欢PhysX的PBR，因为算法的问题

![图片](assets/57ce2d7169d5.png)

#### Decouple Render

针对VR场景，解耦渲染,为VR设备推流提供了便利

![图片](assets/3941d97a974d.png)

#### INT8

对于GP104，我当时连买4块的根本原因就是INT8的支持，这个问题就不多说了，

![图片](assets/8666574478f6.png)