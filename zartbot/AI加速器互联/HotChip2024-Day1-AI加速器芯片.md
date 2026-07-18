# HotChip2024-Day1:AI加速器芯片

> 作者: zartbot  
> 日期: 2024年8月27日 01:45  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491835&idx=1&sn=8de744afeabca2c1b7faf74aa55d3042&chksm=f995f039cee2792f42e77f885d6602dc2ff765048700104399d7c7f2fe75a2dc877465b6fc8f#rd

---

### TL;DR

HC2024第一天, 基本上非云厂商的AI加速器都露脸了, 可惜Nvidia Blackwell还是吹水为主, 对于微架构只字未提,于是本文把它放最后. AMD MI300X和Intel Gaudi3都有一些介绍, Tenstorrent也有一些详细的内容. 然后有一家FuriosaAI的TCP完全基于Tensor描述和爱因斯坦求和约定表达非常有趣的, BRCM介绍了一些CPO相关的内容. 在CPU上有一些端侧处理器, 例如Intel Xeon6 SoC, Intel Lunar Lake, 高通Oryon CPU. 还有IBM大型机**的新一代处理器Telum 2整合了DPU.

## 0. OpenAI Predictable Scaling and Infrastructure

强调ScaleLaw

![图片](assets/f5c9edf17ff7.png)

然后估计也被集群的稳定性惹毛了大谈稳定性...**渣认为: 宕机迁移, 热迁移, 稳定性,并且通过云的规模效应来解决这些难题才是关键啊**

![图片](assets/69ae53de830c.png)

![图片](assets/ba5fcfae9f33.png)

![图片](assets/6de31185b61e.png)

然后还有Iteration之间的电力抖动这些问题

![图片](assets/4e604b1baef6.png)

## 1. AMD MI300X

MI300X架构如下, 片上通过Infinity Fabric Advanced Package提供两个维度分别6TB/s和4.8TB/s的互联

![图片](assets/3ec558b1a86f.jpg)

CDNA3架构上,支持了FP8运算, 同时支持了矩阵运算的MatrixCore

![图片](assets/07d52ae1f595.png)

然后显存容量和带宽和H100 比了一下:

![图片](assets/388afd316534.png)

MI300X架构的一个比较显著的优点是Chiplet做的早, L2 Cache在每个XCD内有独立的2MB,相对干扰比H100小, 然后InfinityFabric提供了一个更大的L3 Cache**称为Infinity Cache, 容量256MB, 可以对外接8个InfinityFabric接口,其中4个兼容PCIe

![图片](assets/11024a48a0e3.png)

片上缓存结构和内存层次化分布如下图所示:

![图片](assets/c7631c18d003.png)

但是没有看到类似于Hopper TMA的东西, 不过这样的架构具有256MB L3Cache还是有优势的. 然后介绍了一些虚拟化相关的内容.

![图片](assets/2c269151e395.jpg)
互联还是大家都已经熟悉的点到点的Fullmesh的连接

![图片](assets/d8f614bb2bed.png)

然后它也很聪明的主打推理和FineTune, 对标了一下H100的性能

![图片](assets/3a9bbfadb8b0.png)

![图片](assets/ebeaa91c77f0.png)

**渣注:** 其实UALink做好了有了交换机能够ScaleUP到64卡后,这样的架构应该还是很能打的. 而针对国内MI308X可能还有一些Rocm上的算子优化适配的工作做完了, 不确定拿它替代H20来做Decode,让H800/H100做Prefill这样的逻辑是否成立,可能还需要一些测试.

比较遗憾的是MI300A没有提及太多, 其实我蛮喜欢CPU Die和GPU Die通过这种方式互联的.

![图片](assets/013d2929b037.jpg)

## 2. Intel Gaudi 3

基本上和前期的白皮书没有太多新增的东西, 主要是一些Spec

![图片](assets/89aa9aa35c4e.png)

矩阵乘法引起是一个脉动阵列的架构, 但是个人觉得256x256是不是有点太大了?对一些扁平一点的矩阵GEMM效率还是有影响的. 然后比较有亮点的是增加了类似于Hopper TMA的AGU

![图片](assets/e94a5ed01275.png)

计算核还是VLIW的, 只能差评了.

![图片](assets/32a83e9b31df.png)

然后L3 Cache也有, 控制路径上中断管理和同步管理,并且有独立的控制NOC.

![图片](assets/1c5feec7785c.png)

伴随着中断管理和GAU, 也能像Hopper TMA那样做一些异步的Copy和MMA的overlap

![图片](assets/5790504b9704.png)

互联嘛就是RoCE based 24个接口, 21个Fullmesh ScaleUP, 3个ScaleOut, 没有什么特别的新东西. 然后性能数据也是在卷推理

![图片](assets/13aa3dfd7d85.png)

## 3. SambaNova SN40L

片上内存大

![图片](assets/f4772d27583e.png)

然后支持在HBM的基础上, 还可以提供额外的DDR**扩大内存容量, 对于推理还是有一些优势的.

![图片](assets/985ee47a2bd4.png)

Tile的架构和前一代比,前一代是交替在NOC上的

![图片](assets/3e706878a850.png)

而这一代直接把PCU和PMU合并在一起了:

![图片](assets/55d0e3c41888.png)

PCU架构上有一个额外的Broadcast Buffer, 然后整个流水线上可以多级广播来构造为一个脉动阵列, 或者SIMD的向量计算, 因此不需要额外的TensorCore了

![图片](assets/5d12d634b022.png)

PMU也类似的添加了一些AGU的能力, 并可以通过多个PMU组合来存储更大的Tensor, 支持对Tensor的转置等操作.

![图片](assets/b2e8479404bc.png)

互联采用Mesh/Ring的混合结构

![图片](assets/660ca7359afb.jpg)

计算上在强调其Dataflow架构的优势

![图片](assets/5eadf3582ce2.png)
传统GPGPU**的问题

![图片](assets/65007c954840.png)

以及其推理性能的优势, 特别是多模态上

![图片](assets/98db16413189.png)

## 4. Furiosa

造了一个新名词: Tensor Contraction Processor

![图片](assets/800b5b7209e7.png)

PCIe的卡48GB HBM

![图片](assets/2a25e8f229c5.jpg)

基于爱因斯坦求和符号好评

![图片](assets/e8332d31c67c.png)

![图片](assets/3d63ec385979.png)

组播到不同的计算单元

![图片](assets/0075e26450e6.png)

和Cutlass一样抽象了空间/时间编排的能力,来提高并行度

![图片](assets/8fbd9fdde92a.png)

相对于TPU的脉动阵列, 更具有灵活性

![图片](assets/61be1728cd08.png)

整个芯片以Tensor为中心,挺有趣的

![图片](assets/5273da4fe400.jpg)

片上NOC也是一个简单的2D Mesh

![图片](assets/7170ee40e4db.jpg)
然后PCIe交换机挂8卡

![图片](assets/86b9d18e335b.png)

软件支持

![图片](assets/a490c2b7df05.png)

然后支持一些自动编译的能力, 效率就不清楚了

![图片](assets/e094e94bbca2.png)

KVCache相关的PageAttention一类的内存管理也有

![图片](assets/456946ca2041.png)

底层用Einsum真的是好评呀~

![图片](assets/c9aa132022cf.png)

## 5. Tenstorrent

架构如下, 已经谈了很多次了

![图片](assets/983675073840.jpg)

微架构很有趣, 分成了三块, 计算/网络(数据移动)/存储

![图片](assets/d9776db5b9cd.jpg)

两个Core做数据搬运, 三个Core控制计算Kernel

![图片](assets/4caeb71098d4.jpg)

数据搬运核的功能, 强调异步访问内存

![图片](assets/1199edbeafef.jpg)

NOC结构两个Route构建2套2D Mesh

![图片](assets/48672ce93812.jpg)

支持多种数据访问原语

![图片](assets/fc6c5732d553.jpg)

计算核包含一个32x32的矩阵乘法引擎, 大小恰当挺好的

![图片](assets/b2e86c98858a.jpg)

然后向量引擎, 支持一些特殊函数和一些shuffle/sort/elementwise的操作

![图片](assets/e3a39be2e24d.jpg)

Kernel间的通信同步通过一个环形缓冲区完成, 好评, 和我在做NetDAM的时候考虑使用Memif的原因是一致的.

![图片](assets/b8636e6e05c4.jpg)

它的ScaleOut基于标准以太网,现在应该支持任意拓扑了, 但当前说的还是2D/3D Torus, 不确定知否支持Fat-Tree和交换机配合

![图片](assets/345109c298a4.jpg)

单个盒子4x8=32个BlakcHole互联

![图片](assets/8ea3c7322952.jpg)

然后计算/存储/网络搭乐高有点意思啊

![图片](assets/6f3bfd54c94a.jpg)

对于Tenstorrent, 后面准备再单独写一个专题详细分析一下,它的TT-Metalium, TT-MLIR都是挺有趣的项目.

## 6.BRCM CPO

主要是TH5的CPO版本, 以及推ScaleUP的光互联

![图片](assets/08732a49edf1.png)

然后可以512卡高密连接

![图片](assets/64c4e801bc91.png)

然后带宽演进也有吸引力

![图片](assets/5ed84df02093.png)

## 7. SK Hynix AIM

继续在推它的存内计算,GDDR6-AiM

![图片](assets/67331f0d7025.jpg)

如何做MHA**这些场景介绍了一下

![图片](assets/b3421686a757.png)

扯了这么多似乎工业界还没真正用起来, 年年上HC, 三年多了也没看到一个产品

![图片](assets/437b2e62dbc5.jpg)

## 8. IBM Telum 2

大型机下一代的处理器, 应该系统架构上和上一代没什么区别, 双Die合封, 然后一个Drawer 4个Socket, 4个Drawer构成32-Socket的系统, 同样没有处理器互联的System Controller

![图片](assets/d48cbcc8757a.png)

变化是在Telum的基础上增加了DPU, 因为原来Z16上把SC搞掉了, 估计还是有点问题, 换了一个现代点的名字加回来

![图片](assets/3402a5df2ef0.png)

直接接在L2Cache上的, 同时还有一个Bulk data transfer的总线, 并控制了PCIe互联

![图片](assets/1abe8a6eae93.png)

然后我不确定以前的大型机上的Coupling Facility的微码是否会跑到这个DPU上来? 然后内置了一个24TOPS的AI加速器

![图片](assets/52e885cd4c26.png)
同时也提供了一个外置的PCIe协处理器卡进一步扩展AI相关的算力

![图片](assets/9541a108f5c8.jpg)

加速器设计也很显然, 多核和更大的Scratchpad

![图片](assets/0baaf2425c56.jpg)

也是脉动阵列配合一些特殊的标量核

![图片](assets/65a70eaf4399.png)
片上网络有点意思

![图片](assets/21cf03308633.png)

总体来看还是步履蹒跚, 离时代越来越远,也难怪昨天CDL那啥....

## 9. Nvidia Blackwell

吹水Session, Blackwell是一个系统, 晶体管数量咋地, NVIDIA High-Bandwidth Interface (NV-HBI) 10TB/s互联, 隔壁按摩店不也是6TB+ 4.8TB的 IFAP么?

前面一文猜测对于`B100估计有160个SM, 然后和Hopper的WGMMA不同的是, 可能B200在片上网络上构建了一些局部的2D Mesh/Ring的拓扑来做GEMM, 可能还会在Distributed Shared Memory的基础上再扩展一些L1.5 Cache/SMEM, 很期待它们NOC和TensorCore这一块的变化.`

唯一有价值的就是稍微透露了一下TensorCore

![图片](assets/be3c465e29a9.png)

计算规模又翻了一倍, 那么与之配套的Cuda Core以及SM的构成如何? 而Hopper的WGMMA也不演进了, 那么势必就是要多个Tensor Core通过片上网络互联并共享一个L1.5层的Cache/SMEME来构建? 然后总体算力单芯片按FP16算,B100比H100高了15%, 对应H100 144个SM扩展到160个SM也差不多?

![图片](assets/46b304a339b7.png)

然后就是继续推FP4的格式啥的. 没有介绍Blackwell微架构差评~ 然后最后的Summary提到"Full-stack, data center scale platform : GPU, CPU, NVSwitch, DPU, NIC, Spectrum and Quantum switches" 还记得30年前的3Dfx么, 当时那家公司吃独食买了STB, 导致合作伙伴翻船站台成就了NV, 而如今呢?

当然Blackwell整个系统是很复杂的,还是很值得致敬的, 铜互联的信号完整性是一个非常复杂的工作, 然后液冷也做了大量的工作

![图片](assets/0ae24218b24b.png)

![图片](assets/40baae787163.png)
冷却剂可靠性, 对不同材质的腐蚀侵蚀

![图片](assets/eb3adcbc6a14.png)

以及液体里的微生物带来的影响和液冷循环这些工作

![图片](assets/56a877201f50.png)

另外其它端侧处理器就不多说了, 或者有空等明天Zen5架构出来了, 以后有时间再补一个专题.