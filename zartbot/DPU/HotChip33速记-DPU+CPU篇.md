# HotChip33速记:DPU+CPU篇

> 作者: zartbot  
> 日期: 2021年8月23日 22:32  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486278&idx=1&sn=54a34250f33be7813a147db80f87f146&chksm=f9961b84cee192921ee9bc900d38e19b4bc89bdcf3d4eaaedce59c2b2b925330e28b03e91a27#rd

---

HotChip33越来越好玩了,分享一些参会的速记，昨天Tutorial主要是一些封装和MLPerf的东西，就没有写文分享了，今天主要是CPU和IPU就做了一些记录

### Intel IPU

Intel的IPU定义从这个图上看还是中规中矩的，毕竟一个以CPU为主业的公司，肯定不会去考虑大量的Offloa的，所以IPU在樱桃眼里就是一个给CPU打杂的基建民工

![图片](assets/60a92c7c20b9.png)

其实一开始单独看这块IPU，200Gbps说实话保守了一点，因为你们家下一代Xeon跑400Gbps的业务基本上没啥问题了.. 

![图片](assets/274e73136207.png)

核的分布来看中规中矩，Crypto、QoS、P4 Pipeline应有尽有，然后16个ARM N1 Core来做复杂逻辑，也还算不错吧，虽然下一个章节就有ARM N2来打脸...

![图片](assets/0926f0f436e5.png)

从软件生态来看，个人认为可能会被P4的一些语言缺陷所拖累，再敲黑板一次，IPU、DPU这个位置是主机的路由器，不是TOR的延拓!

![图片](assets/dade0b84c940.png)

### nVidia DPU

它家的定义也差不多:

![图片](assets/955a21f63e4f.png)

从吞吐率来看，nVidia BlueField3 400G的定位是完全正确的:

![图片](assets/63588e710aed.png)

DOCA的生态看上去不错:

![图片](assets/abbf553ed8a6.png)

但是缺点也很明显， 和DOCA的编程能力如何？总体来说这样渐进式的架构还是很不错的，最下面有Stateless的 pipeline可编程处理器，中间有大量的DOCA做多线程加速，而控制面用16个A78也凑合，毕竟N1和N2还是很贵的.

![图片](assets/9aeea00e242f.png)

### ARM N2

我们因为有Marvell CN10的NDA，因此很早就了解到了大量的ARM N2的信息，因此我才会说Intel新的IPU和某家还在做的基于N1的差距很大...

![图片](assets/46c3327727b3.png)

![图片](assets/8f746e7a5795.png)

CMC Prefetching也是一个非常有用的功能:

![图片](assets/55967b3ce9ff.png)

N2的单核性能比N1提升了40%,伴随着SVE2指令的支持可以灵活的处理多种业务,当然还有一个非常重要的因素是它的片上网络，这样就可以很容易的添加各种协处理器和加速器以及灵活配置多种类型的总线.

![图片](assets/62887a1e6794.png)

![图片](assets/3c49b0024257.png)

针对多核任务，MPAM可以让内存控制器对不同的PartID采用不同的QoS调度，这样针对一些网络相关的应用场景，例如大象流和大量老鼠流的场景就可以通过一些巧妙的设计规避掉Cache的一些问题

![图片](assets/541f6219a3a8.png)

### AMD Zen3

![图片](assets/68ae7b80b60a.png)

Zen3架构来看，增加了19%的IPC性能：

![图片](assets/e29002a88096.png)

从处理器前端来看， 主要的优化就是降低Fetch和Decode延迟，同时增强分支预测的能力,然后更快的I-Cache reorder，主要是现代程序越来越大

![图片](assets/600fd781f822.png)

然后后端的趋势就是越来越宽

![图片](assets/7030216be912.png)

![图片](assets/f3a8b5624da8.png)

这些内容都没什么太大的亮点，昨天的Packaging讲的3D V-Cache是一个更值得期待的东西:

![图片](assets/4d3b68464b38.png)

![图片](assets/7e2a0c58828d.png)

![图片](assets/f1309328caf9.png)

### IBM Telum

5+GHz 大量的片上缓存,片内总线还是一个环

![图片](assets/322dc791e52d.png)

但是可以构成一个32chips的4-Drawer系统，最大的变化是原来的一个负责Drawer和处理器互联的System Controller没了，从画的线来看，似乎变成了类似于3D-Torus Ring的结构.

![图片](assets/fce16db30538.png)

然后就是增加了AI协处理器，然后可以针对信用卡欺诈等AI模型做出快速的推理，这个的确是银行业需要的功能，而且数据本身加密的需求使得Offload成为不可能，而AML对于银行风控又是非常重要的一环.其它从架构上来看就没啥吸引人的东西了.

### Intel

前几天Intel Architecture Day已经讲的足够多了，对于计算任务的划分，樱桃的抽象就是： 

![图片](assets/7520d3703a7e.png)

然后未来的趋势就是通过微架构或者Chiplet配置不同的封装以满足不同应用的需求： 

![图片](assets/05109bbf91c1.png)

#### P-Core

主要侧重于并行性、降低延迟、提升通用性能，同时还加入了新的AMX来执行矩阵乘法运算：

![图片](assets/0c2ea53c7846.png)

#### E-Core

满足客户从低功耗移动应用到多核微服务的全方位计算需求，没有AMX这些矩阵运算，但是标量但愿更宽，前端译码、Fetch和Re-Order队列更深：

![图片](assets/23bbf80b1726.png)

所以P-Core和E-Core针对单线程和多线程业务模式时，通过灵活的搭配可以获得更高的能效比:

![图片](assets/0a7833559ac7.png)

#### Alder Lake

也就是针对不同的应用场景，使用不同数量的E-Core、P-Core构成：

![图片](assets/56e0f3663620.png)

#### Thread Director

针对不同的应用，如何调度到P-Core或者E-Core呢？这是一个非常好玩的功能：

![图片](assets/b419a2230046.png)

利用ML来调度有几分强化学习的味道, 环境和State来自于EHFI Table

![图片](assets/6cb3340c7339.png)

而Action就是Scheduler：

![图片](assets/ea7a88602d0b.png)

### Intel Xeon

Sapphire Rapids也走上了胶水(chiplet)的道路，倒也蛮好的: 

![图片](assets/5a13b17629aa.png)

值得关注的是这个加速引擎：

![图片](assets/073a8461513a.png)

从微架构上而言，这个小家伙很好的帮助了CPU在I/O密集型应用场合下的内存搬运操作, 同时QAT也整合进去，这样支持400Gbps的加密或者160Gbps的压缩、解压缩。所以我一直跟很多做DPU的说，这些东西迟早都会被收拾进CPU的，你们看这不Intel也跟进了么。

![图片](assets/dd8e11c20f7b.png)

从系统架构来看，伴随着Sapphire Rapids的发售，数据中心网络很有可能会直接跳过200G，400G直接到主机，毕竟PCIe5.0、CXL和Sapphire Rapids本身的一些压缩加解密能力，处理400Gbps业务流也不是什么大问题了.

![图片](assets/5718368064c8.png)

然后针对虚机的环境， SVM也干掉了很多DPU的幻想,特别是那些做容器网络的虚机通信的OVS offload的DPU厂家们...

![图片](assets/d63b04165002.png)

还有一个亮点是HBM！

![图片](assets/69bab420fd29.png)

### Samsung 存内计算

![图片](assets/72605ddb34d9.png)

PIM是一个解决冯诺依曼架构问题的好办法，但是从可编程性来看，这个解决方案太硬件了，例如他们在Xilinx U280上的Evaluation

![图片](assets/11251623539e.png)

问题是考虑了硬件架构，但是没有考虑应用. 过段时间会有一篇论文出来，嘿嘿~