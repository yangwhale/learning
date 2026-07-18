# Hotchip-2025:Day2 AI芯片和光互连

> 作者: zartbot  
> 日期: 2025年8月27日 01:56  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495376&idx=1&sn=365a20180bc44db352df420619863fdf&chksm=f995fe12cee27704b2efcb3ab2ae5494a085af91eec45c49a4ed21250d3e825523ec324d29be#rd

---

### TL;DR

今天算是非常重磅的一个Session了, 有Google TPUv7, AMD MI350x, Nvidia GB10. 互连上有Celestial的光ScaleUP方案, UCIe光互连, 当然还有华为的UB和NV的Scale-across CPO等.. NV搞CPO其实实质原因还是Lossless决定的需要更高的可靠性, 这一点大家要明白...放弃Lossless其实怎么不可靠都无所谓, 反正丢包率5%都能Goodput 90%...

## 1. AI芯片

### 1.1 Google TPUv7

整个系统ScaleUP的规模扩展到了9216个Ironwood芯片, 然后所有的HBM(累计1.77PB)都可以用内存语义被直接访问. 针对大规模集群训练的稳定性做了很多RAS设计, 同时SparseCore迭代到了第四代并进行了一些特殊设计支持了集合通信Offload.

![图片](assets/00cfa1bd116c.png)

互连结构上还是一个单个机柜内4x4x4 构成的64卡 3D Torus 拓扑,然后机柜之间全部连接到OCS编排任务.

![图片](assets/ee886364d9e9.png)

通过OCS可以在集群内选择任意若干个Node 构成一个Job, 即便是单个Node故障, 隔离爆炸半径也很小.

![图片](assets/ba43b2f1b2eb.png)

TPU内部OCS如何路由和故障隔离, 以及链路故障如何通过DOR绕开可以参考去年的一篇文章:

《大规模弹性部署：Google如何管理TPUv4集群》

SparseCore本来是用于处理搜广推模型中的Embedding Table构建的, 在TPUv4中的架构如下:

![图片](assets/c4adc5e05942.png)

然后第四代的SparseCore上, 算力增加了2.4倍以匹配更高的TPU Matrix/Vector算力, 然后微架构上增加了Scatter-Gather Engine,类似于Nvidia的TMA, 同时内置了一个标量核对于运算的控制更加灵活. 并且这一代增加了集合通信Offload的能力.

![图片](assets/eaded369ca87.png)

Ironwood的架构如下, TensorCore架构没有什么太大的变化, HBM用上了HBM3e
![图片](assets/122888659a7a.png)

对于安全性也做了针对性的设计

![图片](assets/a6204703bfde.png)

整个ComputeTray采用4颗Ironwood构成

![图片](assets/b083c6970d9b.jpg)

整个Rack结构如下, 由16个ComputeTray构成:

![图片](assets/a5772ad56e98.png)

前天谈到800VDC直流供电时,也讲到了模型在训练过程中Iteration之间启停带来的供电稳定性的问题. Google构建了一个软硬件协同的机制平滑这样的抖动.

![图片](assets/c2edab71bec9.png)

然后还有一些针对供电设计:

![图片](assets/39a8d3550106.png)

### 1.2 AMD MI350x

主要的变化就在这一页:

![图片](assets/f35984c37444.png)

然后XCD有些提升

![图片](assets/250db47704c6.jpg)

### 1.3 Nvidia GB10

也就是以前的Project Digital

![图片](assets/3f845e1d081d.png)

内部NOC互连如下:

![图片](assets/e7f3a4b7caf2.png)

![图片](assets/330e7a0ea7d3.png)

然后内置了CX7网卡可以双机互连

![图片](assets/3c6f957b5ebf.png)

## 2. 光互连

### 2.1 Celestial AI Photonic Fabric

主要是有一个光的Interposer可以拉远

![图片](assets/285cd5e6f455.png)

![图片](assets/0204f4f44d97.png)

然后硅光采用了EAM

![图片](assets/fd418852af31.jpg)

主要优势是在同样的Beachfront下带宽更大

![图片](assets/51773acbfe2a.jpg)

对于Chip-to-chip的带宽也可以增加很多

![图片](assets/9a8d26af6193.png)

相对于CPO的优势, 同时作为Fabric Module, 支持2颗HBM并还可以对外扩展DDR内存.
![图片](assets/db124829ee3e.png)

第一代是和HBM/DDR一起构建一个内存子系统

![图片](assets/f2258b65e7f5.jpg)

![图片](assets/a0d5521753e9.jpg)

然后就可以构建成一个很大的Fabric

![图片](assets/7b9d80d21d5a.jpg)

另外, Celestial AI在arxiv上发表了一篇题目为《Photonic Fabric Platform for AI Accelerators》[1]的文章, 然后进行了一些仿真分析

![图片](assets/a799fd04ff66.png)

其实很早IBM做OMI的时候, 我也考虑过类似的问题, 通过拉远来构建更大的内存池, 例如NetDAM在2021年的一篇文章

[《内存池化和分布式AI集群优化》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486729&idx=1&sn=3d9005f846e12630f6ef3cc92d314500&scene=21#wechat_redirect)

![图片](assets/f0e4aa81244b.png)

对于拉远带来的200ns延迟增加还是值得考虑这个Trade-off还是需要考虑的. 是否更大的L2Cache可以隐藏这样的延迟?

### 2.2 Ayar Labs Optical UCIe

先介绍了一些光互连产品线
![图片](assets/3aa921853ca5.jpg)

这次带来的是UCIe 8Tbps的 PHY Optical I/O chiplet

![图片](assets/25ca0a54843a.jpg)

E2E的Link-Test看起来还是很不错的

![图片](assets/03159152b706.jpg)

然后这图挺有意义的, 通过光互连的内存和ScaleUP可以进一步扩展性能

![图片](assets/f416034bf37b.png)

### 2.3 3D Photonic Interposer

主要还是芯片的限制决定了带宽

![图片](assets/e991576fd4cf.png)

Lightmatter的3D光Interposer

![图片](assets/b76ab3ba2c46.png)

然后整个采用光Interposer可以将带宽提升到114Tbps并支持1024个Serdes, 对交换机和GPU都是有帮助的

![图片](assets/3e7402a9aa63.jpg)

![图片](assets/71d3117f2ae1.jpg)

![图片](assets/ec2dcc155a8c.png)

然后它还是采用微环, 支持DWDM

![图片](assets/138bc85f13dc.png)

并且这里有对比了一下EAM, 两种路线后面需要详细分析一下

![图片](assets/c0009bdee8ce.png)

有一些实物图片:

![图片](assets/cad945a32cac.jpg)

考虑到可靠性, 还有一些可配置的连接能力

![图片](assets/ae57a6e8c94d.png)

![图片](assets/dc2cd8140532.jpg)

## 3. 内存

### 3.1 Marvell

大概就是分几块介绍Marvell的产品

![图片](assets/f49dc8a2b070.jpg)

在SRAM上能够获得更高的带宽

![图片](assets/4e767f36cf4d.png)

HBM上通过Die2Die的定制化接口, 可以增加更多的计算使用的面积

![图片](assets/2e5ed882b77e.png)

最后介绍了一下CXL Memory Expander产品线,

![图片](assets/e70d00faad9b.jpg)

采用Near Memory Processor 的架构, 可以做一些运算压缩解压缩等

![图片](assets/f6ee48824507.png)

这样就可以构成一个GPU的内存扩展池

![图片](assets/b3787b32da67.png)

### 3.2 d-Matrix

大概就是一个通过chiplet拼接的一个加速器

![图片](assets/956349690edd.jpg)

![图片](assets/73d0a452d06a.jpg)

然后在PCIe卡上多颗互连

![图片](assets/65252e511351.jpg)

Core架构如下:

![图片](assets/682f06f1d5c9.jpg)

![图片](assets/15c5cbb94d20.jpg)

![图片](assets/a1c3c643876f.png)

Memory子系统:
![图片](assets/921d44225996.jpg)

然后用了类似于Fungible的技术做PCIe拉远构建更大规模集群

![图片](assets/77e50469f19f.jpg)

然后构建了一个大EP的场景

![图片](assets/a67c4baa596b.jpg)

总体来看, 这种架构编程适配新的模型难度还是有点高的, 具体商业落地的价值如何需要考虑.

## 4. 互连

### 4.1 Nvidia Scale-across

今年的宣传改为了Low Jitter, CX8也没有宣传极致的静态延迟了

![图片](assets/aeacfa5bf576.png)

还是Spectrum-X那样的宣传, 优势大的很

![图片](assets/0040364151bc.png)

然后就是大概介绍了一下CPO, 其实现阶段51.2T和102.4T上, 我并不认为有多大的收益.

![图片](assets/c3f32d45425d.png)

但是NV说, 信号更好, 好吧,是的... 我个人是觉得管它怎么样, 能够通就行, 可靠性还是要网卡来处理, 例如AMD和Intel都在搞Lossy RDMA....

![图片](assets/9a30a34293bc.png)

今年的宣传是增加了Scale-Across....

![图片](assets/9393f00102af.png)

好吧,我孤陋寡闻了,eRDMA在云上跨AZ几十公里已经商用好多年了,没见有啥问题... 但是我们来看看NV怎么说的

![图片](assets/dad892a79b7b.png)

说实话, 就是拥塞控制没做好, 又在打补丁而已...然后号称有1.9x NCCL性能提升, 反过来说, 原来的方法带宽利用率只有一半咯?

![图片](assets/f646c435b6d3.png)

### 4.2 华为UB

最后是廖博讲了一下UB-Mesh,基本上就是今年UB-Mesh论文的一个概述吧.以前在一篇文章中写过一些, 后面华为有一个闭门会议也去听了一下, 内部做了一些分析.

[《再来谈谈GPU体系结构及互联》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494005&idx=1&sn=8a62f95dd2219613efc32a0b6bf61301&scene=21#wechat_redirect)

UB的出发点是利用高速serdes统一各种IO, 同时UBoE又能兼容以太网.

![图片](assets/8ee905fe7076.png)

同时期(2021年)我在做NetDAM其实也是一样的思路, 需要一个统一的抽象层, 而这层抽象就是一层内存语义

![图片](assets/a8f96279f2d3.png)

UB-Mesh拓扑结构如下:

![图片](assets/b3ddbeadf31f.png)

然后就是可靠性处理上增加了备份NPU

![图片](assets/ec3c2ab58849.png)

链路上也有LLR增加可靠性

![图片](assets/39881c657d84.png)

同时基于源路由可以绕开故障节点

![图片](assets/f61a8c53d36a.png)

华为在上面实现了源路由机制的

![图片](assets/41ec50ba5ab9.png)

源路由其实是这个场景里必不可少的, 我也在当年设计NetDAM的时候增加了Segment Routing Header, 甚至希望在转发的过程中顺手把集合通信一起做了. 只是从云的视角来看, 我还是希望第一层接上一个交换机, 后面怎么Mesh都无所谓, 这样对于计算资源的弹性多租安全隔离都有好处, 毕竟云的视角还会拆散成多个单卡/四卡/八卡售卖来服务一些更小的模型, 弹性调度的逻辑上还有一次trade-off而已.

其实有了源路由做Topology Agnostic的路由和流量调度也是非常容易的一件事情, 只是范围的问题, 计算资源弹性售卖希望在一定的范围内, 例如一个Pod千卡规模尽量用CLOS, 对于Pod间和DC之间的互联, 那是受到光纤资源的约束, 各种广域网流量工程上再用到源路由...

参考资料

[1] 
Photonic Fabric Platform for AI Accelerators: *https://arxiv.org/pdf/2507.14000*