# GPU架构演化史13: Ampere架构详解

> 作者: zartbot  
> 日期: 2022年9月6日 03:45  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488322&idx=1&sn=b4dda78f7e82f4f17c86338df545a736&chksm=f9960380cee18a967807559a4f67a77a82763ee78446635811f97b92986b93f87694834c2bd7#rd

---

2017年发布的Transformer模型，通过使用Attention结构代替了LSTM，使得深度学习在大量NLP类型的任务中大放异彩。而NLP的语料比图像识别的多得多，而2018年底BERT模型的发布使整个工业界开始走向了大数据集大模型的道路，对于个人和小型机构处理这类模型几乎变成了"还不是因为穷",最多只能在一些预训练模型上做点微小的工作, 因此针对这样的背景，到2020年时已经可以用TSMC 7nm工艺了，这个时候发布的Ampere架构，该如何考虑呢？

而这一代还有一个很大的不同，比起Volta和Turing错开一年(按照计算能力来看，都属于7.x)，此时又回到了图形卡GA102和计算卡GA100两条线同时发布的时候，各自又有什么特点呢？

![图片](assets/33af90e82979.png)

第三代的Tensor Core登场, 针对超大模型利用稀疏矩阵优化,进一步增加多卡训练的通信带宽和收购Mellanox,利用MIG完成第一代的虚拟化使得可以给云服务提供商售卖1/N A100卡实例.最终BERT性能提升了3~7倍：

![图片](assets/0b74869d18ff.png)

在HPC领域的各项任务也就接近2倍的提升，也难怪最近A100也跟着被禁了...

![图片](assets/6da48c0973f6.png)

### GA100

完全配置的GA100包含8个GPC，每个GPC8个TPC，然后每个TPC又2个SM，累计128个SM，支持6个HBM Stack累计12个512bit内存控制器，发售的A100减为7个GPC，累计108个SM，支持5个HBM2 Stack，累计10个512bits内存控制器。同时增加到了40MB L2 Cache，并且HBM2带宽扩展到了1.56TB/s,支持第三代NVlink

![图片](assets/840721890401.png)

A100 SM如下所示，包含了第三代tensor Core，前两代Volta、Turing每个SM包含8个Tensor Core每周期可以执行64个FP16/FP32的混合精度FMA. 而在第三代TensorCore中，一个周期可以执行256个FP16/FP32 FMA操作，A100每个SM虽然只有4个Tensor Core但整体性能还是翻倍了。同时它还支持192KB L1D/Shared Memory，它还支持大量的异步操作和L2  Cache管理操作，我们具体会在后续CUDA11的小节详细叙述

![图片](assets/45eed4bc10a9.png)

### Tensor Core

新的Tensor Core完善支持了FP16，BF16，TF32，FP64，INT8，INT4，INT1类型,更大规模的Tensor Core使得单个周期能完成更大的矩阵计算，16x16相对于V100提升了4倍，具体的内容，后面再单独整理一篇GEMM优化的笔记..

![图片](assets/639576b7aac9.png)

当然说到数据类型，nVidia的TF32设计时非常巧妙的，新增的TF32不会遇到FP16那样溢出的问题，同时配合新的BF16和FP32可以实现阶梯精度提升，而对于一个乘法器而言，又节省了芯片面积，蛮有趣的一个设计，而FP16还是要留着，毕竟DLSS和光追后的去噪还是要用嘛.

![图片](assets/66cc342cd921.png)

然后还针对矩阵相对稀疏的特点做了一些优化：

![图片](assets/4ba2ed97289c.png)

整体算力又获得了数倍提升：

![图片](assets/90eac3e02c35.png)

#### 内存子系统

针对数据加载路径也进行了优化,减少了中间过程对L1D和RF的占用

![图片](assets/114706b23bf3.png)

而在这一代上把L2Cache也拆分成了两块，使得带宽也获得了2.3倍的提升，同时延迟更低

![图片](assets/37105e9ad2b2.png)

同时DRAM增加到了6块HBM，频率也提升了38%，并且新增了L2-Residency control的功能，通过压缩降低了容量和带宽的使用，我们后面详细介绍

![图片](assets/c5cc37b1d07f.png)

#### 视频图片解码器

当然为了提高吞吐能力，A100还增加了一个新的NVJPG Decoder用于解码JPEG文件以及NVDEC for DL解码视频文件，这些功能是非常有用的，例如我在2018年做Nimble边缘计算引擎的时候，通过海康的摄像头可以捕获视频或者单帧的JPEG文件，但是受制于Jetson的算力，只能以1s一帧的方式识别

![图片](assets/19f261ef760a.png)

而集成硬件的解码器可以对公共安全监控等场景获得极大的速度提升。

#### NVLink

每个Link的带宽还是单向25GB/s,累计增加到了12个，整体带宽600GB/s较V100翻倍：

![图片](assets/5cf2f1015826.png)

同时由于链路的增多,增加了NvSwitch的支持，同时也支持直接从NVMe等存储介质读取文件，这个功能不光对深度学习训练非常有用，对于图形卡加载复杂材质也是有很大性能提升的。

![图片](assets/ef4208a8d41b.png)

#### MIG

相对于Volta时代的MPS，这一代资源隔离更加彻底，在PCIe上支持了SRIOV，针对于云厂商而言，GPU虚拟化分开售卖这些多用户多租户的场景也有需求，A100最多可以拆分成7个Instance，获得独立的SM、Memory、L2Cache带宽及QoS控制

![图片](assets/5bc460a404ac.png)

它可以按照如下方式灵活切分

![图片](assets/b84ef5fa2cd0.png)

针对云端推理业务，也可以配合K8S获得更加弹性的部署。

### GA102

针对图形业务的GA102又有什么不同呢？

![图片](assets/abaca76dc69c.png)

GPC为7组，每组6个TPC，每个TPC包含2个SM和一个PolyMorph Engine，累计84个SM

![图片](assets/6746a80de937.png)

和GA100相同的是Tensor Core也提升到了第三代， 同时针对CUDA Core，FP32的能力得到了增强

![图片](assets/284ed7340e1b.png)

因此整体的FP32浮点运算能力接近于前一代的3倍

![图片](assets/5c3f93ad3b43.png)

还记得第一代光追和CUDA Core及Tensor Core 无法并行执行的问题么？第二代光追核心实现了并行执行

![图片](assets/a52ab54cf675.png)

同时每个RT Core也把三角形相交的计算引擎翻倍了，使得单个周期可以同时追踪两条光线

![图片](assets/d3ad31b83f77.png)

针对运动模糊的光追也做了进一步的优化

![图片](assets/890eb4bd79be.png)

由于Tensor Core算力增强，DLSS支持到了8K，

![图片](assets/06bb28a0eed2.png)

而HBM的价格和支持PAM4的GDDR6X的出现，也使得nVidia在商用卡上采用GDDR6x，也能获得936GB/s的峰值带宽，而前面提到的DMA直接到存储的能力，也可借助DirectStorage API使得下一代游戏加载时间进一步缩短

![图片](assets/cdc6faf7d450.png)

这一代硬件支持AV1的解码也是一个非常好的功能

### CUDA 11

Ampere这一代针对深度学习，增加了大量的新功能支持：

![图片](assets/4c6273f5b489.png)

#### Warp sync Reduce

相对于前几代需要使用WarpShfl的方式，这一代直接一步完成了Reduce操作的硬件加速

![图片](assets/e4caefd174a2.png)

配合Cooperative Group服用效果更好，更加简单

![图片](assets/ba5b965d5403.png)

#### L2 Cache Residency Control

整个内存子系统的层次化结构伴随着自身带宽和容量的提升，以及相应的工作负载急速增加，也需要考虑进一步做一系列优化：

![图片](assets/54d3f2fc1a03.png)

在A100之前，Kernel函数之间的通信要经过显存

![图片](assets/bc085345eac3.png)

在A100后则可以利用L2Cache,让多个Kernel之间的通信降低对全局内存的访问量

![图片](assets/25e681ed4624.png)

同时还可以对L2Cache进行划分，针对持续性的访问设置特殊的内存区间来降低被驱除的概率

![图片](assets/cb03883021ca.png)

而另一个功能是针对数据进行压缩

![图片](assets/275b1da10738.png)

#### Async

CUDA11的最大变化是对异步编程的支持:

![图片](assets/1e8f07844bed.png)

异步实现也非常简单，通过barrier Arrive和wait来实现替代原来的sync_threads

![图片](assets/9f2eb4d5e674.png)

这样就能实现异步的数据拷贝了:

![图片](assets/6537829320f3.png)

拷贝时还可以根据datasize选择是否bypass L1 Cache

![图片](assets/e3e94d569f54.png)

### 总结

Ampere针对更大的数据模型和更大的训练数据,在Tensor Core和内存子系统乃至芯片互联之间做了大量的优化,同时也在算法和编程环境上进一步优化. 有很多值得学习的地方,当然最近几年显卡价格也被炒的奇高,导致渣手上也没有Ampere的卡,所以GEMM和Tensor Core的详细情况后面有时间了再搞一下.