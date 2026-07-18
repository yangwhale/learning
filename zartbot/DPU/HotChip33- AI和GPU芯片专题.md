# HotChip33: AI和GPU芯片专题

> 作者: zartbot  
> 日期: 2021年8月25日 17:19  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486352&idx=1&sn=f5aa473f1c43d2ddc3cb0abae5b00349&chksm=f9961b52cee19244a2a052d6b5b926e13a44dcd6422abd8559f90de95e6d2b148e3a3b078ccd#rd

---

HotChip33第二天主要是GPU和AI芯片， Cerebras的第二代Wafer-Scale芯片和整个AI集群很有趣，SambaNova的Software-Driven架构也很好玩，然后Intel的GPU也是值得看看的. 相比这些大佬，国内的Enflame就有点，片间连接还在带外走线....总体来说这届Hotchips在片上网络和封装技术给世界打开了一大扇窗，.其中的Cerebras MemoryX，我们已经有一个容量更大的样机了，嘻嘻~

顺手吐槽一下隔壁家的SIGCOMM，比起芯片也软硬件快速的融合和百花齐放的架构，网络这个行业越来越显得沉闷了，难怪SDN都被判了死刑...

### Cerebras

![图片](assets/a0a96368953f.png)

Cerebras是第一家做WaferScale的AI公司，所以它家的芯片直接叫WSE(Wafer-Scale Engine)，也即是在一整个12英寸的晶圆上制造。

![图片](assets/3d4f17fc4df9.png)

当然还有一个异曲同工的处理器就是前几天Telsa发布的Dojo：

![图片](assets/be4644f958ff.jpg)

Dojo则是采用台积电的封装实现的System-on-Wafer

![图片](assets/be76c9feccf4.png)

![图片](assets/15f41f7f19a9.png)

#### Wafer-Scale的技术难点

这样大的芯片，主要有`良品率`和`供电/散热`两大难题，首先来说比较容易理解的散热问题，例如Dojo的供电/散热结构：

![图片](assets/c5572e5f681d.jpg)

而Cerebras由于芯片规模更大，散热和供电更加紧凑:

![图片](assets/dbe5096da5a0.png)

除了这么小一块引擎，整个机箱大量的空间被用于散热和供电:

![图片](assets/dc0b568a4d70.png)

第二个问题是良品率的问题， 众所周知，由于Wafer应力和灰尘等各种因素，整个Wafer上会有部分的单元会失效，所以die-size越大架构就越高。而针对Wafer-Scale的处理器，如果稍微一个小核坏掉了，那么整个片子就报废了? 当然不是，这里就涉及到片上网络的路由协议了，Cerebras能做WSE的主要原因就是可以通过片上网络构建disjoint path绕开失效的节点，通过源路由协议就可以很好的实现对失效引擎的屏蔽了。例如Telsa也有类似的一个图:

![图片](assets/41f80d258343.jpg)

由于每个Tile都有大量的互联接口，片上网络通常在这种2D-Mesh或者3D-Torus的结构中都会设计self-routing header，例如ARM在新的ARMv9中配合的CHI总线或者富士在超算中实现的6D-Torus结构.

![图片](assets/6d0bb260d8b6.png)

当然这也是我设计Ruta协议[1]时一直提到的，我们需要将片上网络和数据中心网络的路由集成在一起,

![图片](assets/311420256f7c.png)

#### Wafer-Scale的原因

事实上在分布式机器学习中,模型并行和数据集并行已经成为分布式任务分割的趋势:

![图片](assets/7ba7103fa577.png)

当然从`训练数据`流动和`参数同步`数据流动这两个趋势来看，任何一种选择都会带来大量的网络吞吐需求，所以更好的办法是实现一种混合的调度:

![图片](assets/7e0cd72ffffa.png)

即便如此，还是有片间通信的延迟和带宽约束，`布线`和`带宽`的约束决定了`串行`总线，而`Serdes`的`延迟`又是一个值得考虑的地方,所以GraphCore采用了PCB上互联，并构建模型不同层，让训练数据在一个PCB内多块IPU上流动：

![图片](assets/9bab53906d8f.png)

但是这样太复杂了:

![图片](assets/1e2be65986e0.png)

而Wafer-Scale直接在光刻的时候把片间网络构造好则成了跳出传统的网络思维的通信方式，而通信的本质无非是共享内存，直接简单粗暴的解决了问题:

![图片](assets/e9f037fd206d.png)

![图片](assets/aa4c9fc9dfdf.png)

当然如果单机解决不了，那么多机也可以解决问题,它采用每个CS-2机箱12个100GE口连接到MemoryX交换机.

![图片](assets/a1ba1fca41e9.png)

而MemoryX主要就是执行Wafer外的参数同步和梯度更新:

![图片](assets/7daf65b526ec.png)

在通信上做好协同，可以极大的提高训练的效率: 

![图片](assets/947f4db06017.png)

同时针对稀疏矩阵乘法,其实可以通过一些非常巧妙的编码来降低带宽: 

![图片](assets/c7e0aa8db6fd.png)

而MemoryX针对参数和梯度的同步也采用了特殊的做法,Broadcast和Reduce有些特殊的处理，当然我们最近正在做一个东西会比这个算法还快，在此就不多透露了 

![图片](assets/24cd45f9e0fb.png)

### GraphCore

Graphcore也发布了第二代处理器并把整个集群的规模扩展到了512个节点，每个1U的节点通过大量的QSFP28接口互联

![图片](assets/607d0b89d01f.png)

GraphCore的架构上对比WaferScale已经没有太多的新意，但是有几个地方值得注意的是，片内大量的SRAM配合DDR比HBM好：

![图片](assets/ec3932fa078e.png)

另一个是任务编排上和系统同步上：

![图片](assets/10695291c9d8.png)

### Esperanto

![图片](assets/2307bb53ac08.png)

基于RISC-V的核心，然后8个Core构成一个Cluster，然后4x4的Xbar互联构成一个Tile，最后通过一个2D-Mesh连接到外面其它Tile:

![图片](assets/b82c57a91182.png)

然后多个Tile互联成一个大的系统:

![图片](assets/37673ea94e22.png)

然后6个Chip构成一个Blade

![图片](assets/6c6c4ca1f72f.png)

同样软硬件协同的方式，他的编译环境也是非常有趣的:

![图片](assets/98b4d123fb16.png)

### SambaNova

给人眼前一亮的是这个图，通过算法降低通信复杂度: 

![图片](assets/87b7ac0392b5.png)

这个Software-Driven Architecture也比较有趣

![图片](assets/49d66bd3a3dd.png)

PCU是计算单元

![图片](assets/3c3f8ae7c66d.png)

PMU是存储单元

![图片](assets/223e1e94bd68.png)

Switch就不多做解释了，互联的交换机，片上网络单元

![图片](assets/1438a7489e4b.png)

比较有趣的是AG和CU:

![图片](assets/ec6803be33d1.png)

然后尽力的通过软件的优化和调度把整个workflow分布在芯片上:

![图片](assets/a9a7b27949ff.png)

### Enflame

HotChip上看到的国内厂家，确实很不容易，但是这么片外的LARE互联总觉得有点山寨啊..

![图片](assets/fd89290e5071.png)

![图片](assets/2c47f38ae051.png)

### Xilinx

大家都在玩片上网络:

![图片](assets/9023435ffc62.png)

AIE其实就是把Memory和计算核更好的耦合，然后通过片上网络互联，针对AI的计算场景做优化:

![图片](assets/6ae0e947ecb0.png)

### Intel

从AGP的那个年代， Intel企图用i740吊打3Dfx和nVidia开始，樱桃做显卡之心不死已经20年了。而这次的显卡则是更多的瞄准了它无法将大量向量和矩阵引擎弄进CPU的现状，只能新开一条GPU的产品线，顺手做做显卡的业务，本质上 还是一个向量和矩阵计算引擎的融合：

![图片](assets/d1e700545900.png)

然后Render Slice就是4个 Core并配合了`Ray Tracing Uint` 这是很不错的选择:

![图片](assets/e35ece4d3bfa.png)

然后多个Render Slice通过共享一个2级缓存互联: 

![图片](assets/2bd967a99bb6.png)

真正好玩的是正面开始迎战nVidia的A100等高端AI集群的GPU,也即是号称`Ponte Vecchio`的架构

![图片](assets/cf2e60b1e0a4.png)

人家有nvlink，樱桃就搞个

![图片](assets/1b2f9c418d6c.png)

Ponte Vecchio还有一个非常值得关注的是把封装的艺术玩到了极致：

![图片](assets/15082678c8b2.png)

![图片](assets/98dfe391cafe.png)

最终构建成了一个灵活的架构应对DGX

![图片](assets/b44b1e8d8402.png)

#### Reference

[1]
A0001：分布式机器学习的网络优化
: https://mp.weixin.qq.com/s/jxzlGL5ijRxOhrFasy-JyQ?scene=21#wechat_redirect