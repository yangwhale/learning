# GPU架构演化史12: Turing架构详解

> 作者: zartbot  
> 日期: 2022年9月5日 04:23  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488280&idx=1&sn=55bf2338621782b6996e69267aecdd2e&chksm=f99603dacee18acc50b6ae07753bf62fe43253d87b66439e3ff5462e77383ee29493fb7aa101#rd

---

紧接着就是2018年发布的Turing架构了, 敢叫图灵自然是做了一些惊天动地的大事, 所以WhitePaper里也有一句话< Graphics Reinvented >. RT Core的引入配合DirectX12实时光线追踪，Mesh Shader的引入使得画质得到革命性的改变. 而深度学习反哺图形的DLSS也是很赞的技术。

![图片](assets/7466af231a39.png)

### Turing Core

turing这一代还是TSMC 12nm工艺，从图形卡的角度对比前一代Pascal增加了Tensor Core和用于光追的RT Core，然后光栅化的一些管线也有变换

![图片](assets/c827e80b5985.png)

TU102包含了6个GPC, 每个GPC可以包含6个TPC,每个TPC内包含2个SM,

![图片](assets/f34cb08e5482.png)

满配核心数如下所示:

![图片](assets/5ea0df25ee8c.png)

核心内分为4个区域，每个区域有16个FP32 Core，16个INT32 Core 2 个TensorCore，一个Warp Scheduler一个Dispatcher。同时包含L0 I-Cache及64KB寄存器文件，4个区域共享96KB L1 DCache/Shared Memory.同时针对光追业务，每个SM增加了一个RT Core

![图片](assets/30c3bf399594.png)

和Volta相同，WarpScheduler拆开，可配置的L1D&Shared Memory加回来了，而由于Warp Scheduler拆开，单个TPC内的LD/ST UNIT也翻倍了

![图片](assets/1523e1e81e94.png)

需要注意的是图灵核心实现了对核心执行数据通路的重大修改，通常现代作色器工作负载混合了FP算术指令和其它一些简单的指令，例如寻址和获取数据的整数加法及浮点数大小比较等。以前执行这些非浮点指令的时候，浮点数据处理路径处于空闲状态。而在图灵每个CUDA核心旁边增加了第二个并行执行单元使得这些指令能够和浮点运算一起执行。根据实际workload统计可以获得大概36%的性能提升。

![图片](assets/2542d2ab94b2.png)

而整个渲染方式由于光追的引入，也变得不同，它包含了光栅化、光线追踪、着色、AI降噪、深度学习超采样等多个功能构成的混合管线

![图片](assets/e6022a4087b2.png)

具体内容我们后面详细展开，这也是在图形卡上引入Tensor Core 及RT Core的原因，但需要注意的是Tensor Core和RT Core及Cuda Core是不能同时执行的.

![图片](assets/f3a88b7a6f28.png)

由于Volta引入了per thread PC, Turing在指令集上引入了Uniform Datapath 和Uniform Register File

![图片](assets/81f060856ee8.png)

### Tensor Core

在Volta中引入了Tensor Core以后深度学习任务性能获得了飞跃性的提升， Turing核心中为了进一步增加推理能力，带来了INT8和INT4精度模式：

![图片](assets/a73be6e8d45b.png)

而这些支持对推理业务及游戏应用帮助非常大。很多推理业务性能提升了数倍，也为大规模商用提供了足够的算力支撑

![图片](assets/3f33ae269fde.png)

另一个应用是深度学习这些计算业务开始反哺图形业务， 光追效果后的AI画面降噪和Deep Learning Super Sampling (DLSS)来提升画面质量，后面我们会详细叙述。

### RT Core

实时光线追踪算法一直是图形学的圣杯，根据光路可逆性从眼开始倒推，然后求光线和物体的交，然后测算光照强度

![图片](assets/05371e43c86d.png)

而整个算法的运算量非常大，如何找到光线对应的那个三角面？

![图片](assets/9cc9a904b7fb.png)

工业界为了完成实时渲染，在Maxwell时代引入了基于体素Voxel的方法，它是一种自顶而下的空间划分，虽然在PolyMorph Engine中有一些硬件投射加速，但整个运算量还是非常大,

![图片](assets/567a67bd62d6.png)

而另一种算法也在后面几代逐渐引入，也就是Bounding Volume Hierarchical Tree(BVH)树，相对于SVO的空间八叉树等分，它采用了一个树状结构划分空间，效率更高

![图片](assets/5d3905e2a7f1.png)

但是整个计算还是在软件上用Cuda核算的，平均每根光线也需要数千条指令

![图片](assets/9415f8cf1339.png)

而RT Core便是用于硬件加速这些指令的, 累计68个RT Core使得Turing的光追性能提高了10倍。

![图片](assets/99a9ba948e2b.png)

光追的引入，也使得渲染的管线发生了变换:

![图片](assets/c130e7cf521b.png)

当然还有另一个问题，那就是光追产生的噪音

![图片](assets/4a40890c3531.png)

利用神经网络来降噪的算法也就被开发出来，正好还能用上Tensor Core

![图片](assets/b36dc7b28b77.png)

所以明白这一点，也就明白了Nvidia为什么Tensor Core要保留一个4的维度，同时还要使用FP16了吧. Turing即便裁剪掉Tensor Core的GTX1650也要加一些FP16的单元，为啥呀？

![图片](assets/a4c8be2893ef.png)

而另一个基于深度学习的应用是DLSS

![图片](assets/58969c2d421e.png)

### VRS可变码率渲染

![图片](assets/8ed6a8585cb9.png)

### MVR

另一个功能是针对VR场景的多视角渲染

![图片](assets/35137cf698f3.png)

### Mesh Shader

它的动机来自于更加细致的几何细节的需求

![图片](assets/f9e5b7caac66.png)

当然对于不规则的几何体如何高效绘制？三角形曲面细分和置换贴图不一定是最高效的方式。

![图片](assets/cbd32a15288e.png)

还有就是Quadro产品线需要的CAD建模类场景

![图片](assets/00833b3ab94d.png)

虽然传统的管线基于曲面细分以及Compute Shader的置换贴图等功能已经可以提高渲染的细节了，但为什么大动干戈去修改整个渲染管线呢？

![图片](assets/0f17e45437d5.png)

本质原因就是在调度的时候线程映射原有的管线相对独立而固定

![图片](assets/daf5556effe4.png)

例如图元数据分发的时候每一帧都要去重

![图片](assets/727799c58cbd.png)

但是如果我们通过Meshlet的方式呢？本质上它是一个预先生成Batch的模式，

![图片](assets/a0eb548c7e4b.png)

而MeshShader的提出也和新的Volta 和Turing架构中引入的更加灵活的cooperative group的能力有关， 当然也使用了Dynamic Parallesim的能力，本质上从CPU launch的Vertex变成了一系列的Batch的ThreadGroup(Worker Group)

![图片](assets/50e8a90cffa1.png)