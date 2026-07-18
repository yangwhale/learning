# GPU架构演化史9: Maxwell架构详解

> 作者: zartbot  
> 日期: 2022年9月2日 13:52  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488157&idx=1&sn=3ad1fb6266e7ecaaf7385bc742f43d1d&chksm=f996025fcee18b49396182202469fbad0186c827710ef8e5a9eb240f879f46572172b840ee4b#rd

---

作为一个曾经在外企干marketing的,想起很多年前也处理过合规的事情，看到老黄家被禁,觉得他们应该会做一个FP64裁剪的版本的A99,H99出来,这样不影响炼丹的销售...当然另一方面，国内很多企业还是要继续埋头苦干才行啊，体系结构上给国内一个蛮好的机会，反正也没法后向兼容了，那还不如大干一场...

2012~2014年这段时间，虽然工艺还是在28nm，但应用产生了更高的需求。一方面是对成像质量，实时的光追渲染一直是大家的梦想，当然最终的硬件实现要到了Turing架构才行，而在2011年nvidia有一篇关于Voxel[1]的光照算法引起了很多相关的研究，整个算法基于Kepler架构实现，虽然比Fermi快了接近30%~58%,但似乎离商用还有一定的距离，算力还是不够。另一方面是Tegra在移动端一直不温不火，然后第一次把Kepler和Tegra整合的K1似乎功耗还是很大， 移动端、主机小型化(SFF主机逐渐流行)和笔记本对GPU的需求也在增加，如何设计一个更低功耗，Mobile First的架构？还有一方面的原因是4K等高分辨率显示器价格快速下降，另一个原因是工业界开始逐渐尝试VR应用，这些都使得渲染负载成倍的上升。

而正是这些需求决定了Maxwell的架构，当然还有一个背景是AlexNet发布于2012年，然后接下去基于kepler的深度学习框架逐渐的在被各个大厂接受，例如Google利用Kepler实现猫脸识别任务以及开源的Caffee发布，而在体系架构中还没有出现太多的关于AI的优化，因此这一代的重心都放在了图形上，解决Voxel全局光照，进一步降低功耗，支持一些高分辨率的渲染，但是工艺还是28nm，要怎么折腾呢？

前几天还在和某司一个同学讨论8K 120fps VR渲染的事情，想了一些算法，例如Tile-Based Rendering，Dynamic Super Resolution这些东西，再回头来看Maxwell的架构，又多了几分感慨。

### Maxwell

2014年，在28nm工艺不变的情况下，nVidia通过架构的改进、设计的优化 Maxwell架构的GTX 750Ti发布了，对比前一代定位相同的GTX 550Ti，性能为前代的2倍，功耗从前代的116W降低到了60W。

![图片](assets/89fba0ed26c5.png)

在kepler架构时，一个SMX内CUDA Core个数为192个，并不是偶数个，然后四个Warp Scheduler全部连接到一个Crossbar然后192个Core全部练上去，多任务下的效率不好调度。因此在Maxwell中又把它打散成小型分布，变成4个独立的Crossbar，每个挂载一个Warp Scheduler、独立的16K个32bits寄存器文件，32个Cuda Core、8个LD/ST，8个SFU。当然DPU(FP64)没有画，实际上它也在这个地方砍了一刀，每个Warp Scheduler只挂载了一个FP64的DPU，资料来源于AnandTech的一篇分析[2]。

所以需要注意的是，Maxwell是一个完全针对移动和图形的GPU，而不是一个Kepler Tesla HPC产品线的后继，低功耗的魔法在于取消一些不太需要用的ECC和FP64计算能力，然后由于Warp Scheduler和核分别连接在4个独立的CrossBar上，因此可以针对实际计算的负载动态的以32个CUDA Core为单位进行调度， 例如在一个SMM内工作量不饱满时关闭三组Warp Scheduler就留一组工作？

![图片](assets/b442b5fa2cff.png)

当然您也会看到Tesla M10这一系列的数据中心卡，而它们更多的是在处理当时那种虚拟桌面瘦终端的图形业务上，而不是用于HPC计算。而Kepler真正在超算的后继者是Pascal，它在Maxwell的基础上又把DPU(FP64)加回来了。当然还有一些改动，Maxwell仅支持96KB的共享内存，而L1Cache功能被挪到去和Texture Cache共享，因为它针对于图形也不需要像HPC那样做过多的复杂的运算，很多图形的运算并不需要太多的寄存器，自然寄存器溢出缓存的概率就低了，当然这样调整带来的影响就是L1Cache和SharedMem动态配置的功能取消了，然后Bank在Kepler上4B和8B可调也被固定成4B了。

从整体来看Maxwell CUDA Core的使用效率高了很多, 通过打散4个WarpScheduler，每个核的性能提升了1.4x，所以从SMX到SMM来看，即便是核少了33%，但是核利用率的提升也把性能补回来了，同时由于功耗的控制，GTX980增加了一倍的SM，从而做到了功耗略微下降但整体性能基本翻倍。

![图片](assets/380c7f513ece.png)

#### PolyMorph Engine 3.0

由于核心加多和性能提升，还有潜在的VR及高分辨率需求，PolyMorph Engine性能也得到了2x提升。除此之外还有一些针对Voxel GI算法的硬件加速，后面会讲到。

#### Memory

内存相对于Kepler 6Gbps提升到了7Gbps， L2 Cache增加到了2048KB，同时像素单元压缩算法也到了第三代，不过具体的细节很少，而且在后面也快速迭代到Pascal的第四代和Turing的第五代压缩算法，基本上没有什么细节。不过这些大致的算法就是，先找一个块中的基色，然后把每个坐标按照基色的index编码，然后有不同的算delta，另一种就是检查一个4x2 或者2x2 block内是否为常量，来压缩。Paltashev, I. Perminov, *TEXTURE COMPRESSION TECHNIQUES*[J], *Scientific Visualization*, 2014 Vol 6, Num 1, pp. 106-146 上有一些概述，知乎上有这文的翻译[3]， 而解压缩反正就是TextureUnit去根据地址索引的，完全可以硬件加速的。

![图片](assets/9f952a040b6c.png)

这样下来整个Maxwell的内存频率提升了1/6，然后内存带宽使用通过压缩算法降低了25%，整体来看性能提升了50%。

### VXGI

在计算机图形学中，实时光线追踪一直就像圣杯那样存在。我们在这个系列的第一章就讲述过这个算法。虽然算法很早就有了，但是光线的相交条件追踪计算量非常大，基本上都是用在电影这些可以离线渲染的场景。针对游戏这类场景中的实时渲染还是因为算法复杂一直没有实现，所以过去很多年都在采用光栅化的算法替代。2011年 Cyril Crassin等人基于Fermi架构实现了一个25~70fps的基于体素(Voxel)的渲染算法[4] ，但在Fermi和Kepler上针对一些复杂场景还是存在性能问题。而这些问题通过在Maxwell上的架构调整和一些硬件Offload解决了。而最终光线追踪技术也成为DirectX12中一个非常重要的技术，更完善的支持要到Turing架构上基于BVH算法的RT Core，到时候我们再详细说.

#### Voxelization

除了论文以外，Cyril Crassin在2012年发布《 OpenGL Insights 》中详细讲述了这一算法。传统的3D图形表示都是采用三角形面，所谓的体素化(Voxelization)就是将空间中的三角面进行实时的3D扫描，相当于是先把空间按照像素块化(体素化,Voxelization)建模，然后将其存入一个3D的数据结构中。

![图片](assets/fccbb2693e45.jpg)

然后将直接光照注入，对于这些小立方块，其实就是一个光栅化的过程

![图片](assets/131f8c521d2f.png)

最后在渲染时，然后再来看这些立方体根据原始三角形的法向量方向构成一个锥形反射来追踪光线(类似于手电筒)，而你肯定会好奇，下图中的Octree是干嘛的呢？

![图片](assets/16013d4038e8.png)

我们来看如果按照均匀的512 x 512 x  512来拆分空间，一共128M个Voxel，即便是每个Voxel只占用32B的存储空间，累积起来整个场景渲染也需要4GB.很自然的一个想法，那些细节不重要的分粗一点，细节重要的分细致一些？基于LOD来做？

![图片](assets/20d334d20037.png)

是否有更自动的算法呢？我们来看场景中势必有一些空旷的区域可以过滤掉，所以这里就引入了Octree的结构

![图片](assets/5d8c6eac5b75.png)

它是一种非常精简的基于指针的数据结构，树根代表整个场景，而每个子节点都代表它的父节点的1/8的空间。

#### Octree-Based Sparse Voxelization

它的算法很简单，以整个场景开始构建，逐步将空间细分，再对非空的子空间继续细分,完成后，利用了Mipmap的原理来对3D Texture进行不同级别的Mipamp的产生

![图片](assets/8a07d5eb1422.png)

整个细分的过程是可以通过多线程完成的，然后分成的小的tile还可以做tile based rendering，或者像素做tile based cache，进一步降低了内存的压力。

![图片](assets/9b6be190a359.png)

这一步既利用了GPU多线程能力，又省了带宽，还降低了整个计算规模，巧妙不？

![图片](assets/b35627a79e9e.png)

#### Conservative Rasterization

当然在光栅化的过程中，需要考虑一个问题就是原有的三角面投射可能会导致部分边没法覆盖的问题，

![图片](assets/e0ad386948c4.png)

虽然可以用MSAA这些抗锯齿的方法去补，但还不如更加保守一点，外面加一圈边套使得全部被覆盖

![图片](assets/83287a7a290b.png)

#### Cone Tracing

这也是算法非常巧妙的一步，当光线射入一个点需要计算反射时，它模拟在这个点点亮一个手电筒的方式来计算反射光源影响的范围，而且根据材质不同还可以定义不同的锥状结构.

![图片](assets/04a8b1fba7ad.png)

首先计算光线直接照射时，或者为自发光材料时，基于法向量和材质计算后续间接光照时的额外光反射的辐射值。总体来说光线注入类似于光栅化。

![图片](assets/15e8d9c2ed45.jpg)

而最终对于Camera来拿到数据时，可以根据Octree的Mipmap来构建，根据Cube为止映射找出相应的值来:

![图片](assets/6b0cfb6551e5.png)

#### Maxwell的加速

对于实时渲染而言，虽然Sparse Voxel Octree +Cone Tracing 在Kepler上的一些Demo场景可以跑到70fps，但是针对游戏这些更复杂的场景性能还是不够的，于是Maxwell考虑到整个算法中对计算量最大的几块做了硬件加速，一个部分就是硬件的Conservative Raster，另一个就是针对三角面投射到体素(Voxel)上做了硬件的多投射加速:

![图片](assets/17e461758792.png)

SVO算法除了论文和OpenGL insight的介绍，还有一篇很好的教程:

Sparse Voxel Octree GI (构建体素)[5]

Sparse Voxel Octree GI (光照篇)[6]

#### Dynamic Super Resolution

除此之外，它玩了一个利用高分辨率渲染然后down sampling到低分辨率的功能，怎么想怎么觉得不对，这不后面DLSS还反过来玩呢。。

![图片](assets/c083bd359695.png)

### 小结

总体来看，Maxwell的体系结构即便是在28nm上也腾挪了很多内容，取消L1Cache和Shader Memory共享的结构，转而让L1和Texture Cache共享，拆分SMX成四个小的32核的集群，增加了核心使用率，同时又可以动态控制功率，然后取消了一些图形卡不需要的双精度浮点引擎，也对ECC做了一些取舍，毕竟这一带卡打的是低功耗核图形的市场。同时对VXGI进行了一系列硬件加速，使得工业界把光追算法引入到了游戏，极大的提升了画质，当然还有一些不完善的地方，也为后面RT Core的出现埋下了伏笔...这一代产品是同时从算法、软件、硬件上配合，给大家带来更好的体验。有很多东西值得我们去学习，反思。

#### Reference

[1]
Interactive Indirect Illumination Using Voxel Cone Tracing: *https://research.nvidia.com/publication/2011-09_interactive-indirect-illumination-using-voxel-cone-tracing*
[2]
GM200 - All Graphics, Hold The Double Precision: *https://www.anandtech.com/show/9059/the-nvidia-geforce-gtx-titan-x-review/2*
[3]
图像块压缩/纹理压缩技术细节: *https://zhuanlan.zhihu.com/p/486903217*
[4]
Interactive Indirect Illumination Using Voxel Cone Tracing: *https://research.nvidia.com/sites/default/files/publications/GIVoxels-pg2011-authors.pdf*
[5]
Sparse Voxel Octree GI (构建体素): *https://zhuanlan.zhihu.com/p/80849630*
[6]
Sparse Voxel Octree GI (光照篇): *https://zhuanlan.zhihu.com/p/83760807*