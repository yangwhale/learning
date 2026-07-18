# 从大机的视角看DPU发展

> 作者: zartbot  
> 日期: 2021年8月14日 19:50  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486147&idx=1&sn=e01ed29dc0831bcb3ab2234b3ff3ebaf&chksm=f9961a01cee193174892676e81ea7eba96221c8a320137f4b246f1d484e770bebed420db8cf9#rd

---

### 前言

前几天随手写一个DPU的文时,脑子里忽然闪现出若干年前看IBM Z13介绍视频[1]中提到的I/O专用处理器

后来找了一下这段视频, 大概这段适配4分40秒IBM Fellow Jeff Frey 提到每个I/O都有两颗PowerPC Core提高整机使用效率, 使得整个系统使用率到100%

这些便是今天这文的一个开头,是否从大机的视角来看现在的DPU? 混合云场景中，企业需要更大规模的计算资源，公有云几家无论是Outpost还是AzStack，还是国内几个云基本上按照Rack交付混合云产品，是否我们又在通过X86处理器和GPU构建大型机?  

![图片](assets/701fe61af358.png)

从IBM z13到如今z15的演进过程，或许能给我们在`瞎搞DPU`的时代更多的思考和借鉴

### 久分必合

有些东西就该CPU自己整合进去的，不用你们DPU瞎操心

#### PCIe加速卡->CPU内置

在z13的时代(2015年)和现在X86搞DPU的时候很类似，大量的PCIe加速引擎做各种优化，例如Flash Express加速存储、Crypto Express做加密、zEDC Express做压缩.而到了z14时virtual Flash express直接替代了存储加速，而到了z15的时代，CPU内置的CPACF也基本上把所有的对称加密负载Offload了，Crypto Express只是做一些非对称加密场景.同样zEDC也被集成到了CPU内.

DPU上有多少功能可以被CPU吸收呢?

#### 加解密和压缩协处理器

IBM大机上为了加速处理器的吞吐，尝试着使用各种加速引擎，最终在z15上这些协处理器被整合进了CPU：

![图片](assets/a8ebb6de7d87.png)

很多想在DPU上去做这些事情的，基本上最终是徒劳的，压缩、加解密、Hash这些协处理器，最终因为性能和延迟的要求，都会集中到CPU上去。从数据安全的角度考虑，Fungible和Pensando提出的一些Offload加解密、压缩的场景是相对无效的，数据从CPU绕出来做完再读回去浪费的cycle数还不如自己加个协处理器做了呢..

### DPU的标杆

DPU的核心是在内存子系统和I/O子系统，通信的本质不就是共享内存么? 敢于直击核心问题`非冯诺依曼架构`或者继续`缓解`内存瓶颈才是关键.

#### eDRAM 从存储到系统

在IBM大型机的PU互联芯片上，通常都是放置一大块eDRAM作为L4Cache使用：

![图片](assets/02be01fd26d7.png)

只是这块芯片在不同时期的名字有些不同，在z13中它叫Storage Controller:

![图片](assets/f27d6d2e5c91.png)

![图片](assets/febaf66ab9e7.png)

而在z15系统中，这块芯片的Storage Controller变成了System Controller:

![图片](assets/fa463b4c2907.png)

对于PU之间的互联直接IBM直接选择了使用eDRAM构建L4Cache，而且eDRAM的密度[2]可以参考资料中的文:

![图片](assets/f08755a50e25.png)

处理器间的互联通信的本质其实就是在共享内存，因此当我们做DPU时，如果不去主动触碰内存子系统只是在I/O子系统上瞎搞是毫无意义的，再说以前CPU互联总线有一些客观的问题，但是PCIe5.0和CXL给了行业这个机会.

#### 共享内存

大型机共享内存的处理方式也是值得我们借鉴的，IBM SMC(Shared Memory Communications)支持多种模式，首先是SMC-R, R自然代表的是RDMA.

![图片](assets/80b07c9df072.png)

然后还支持SMC-D 即Internal Shared Memory模式，有点类似于SRIOV一类的处理方式，但是这个模式很好的诠释了通信的本质就是共享内存，而共享内存最好的方式就是通信: 

![图片](assets/999045aa3b06.png)

当然两种模式混合也可以：

![图片](assets/f0e56d43c1ba.png)

#### HiperSockets

注意到上图中一个关于socket的细节，既然Memory都可以共享了，那么直接对系统小改一下，把TCP Socket也玩起来:

![图片](assets/cd23814477af.png)

所以很多云在做Socket上的优化，人家在大机上好多年前就玩过了...

#### OMI

在去年的HotChip上IBM发布了一些关于下一代Power10处理器的消息：

![图片](assets/38d2256aac83.png)

从架构上而言，L3和片上I/O子系统的设计非常有趣, 而更有趣的是PowerAXON总线构建的SMP互联，和OMI内存结构

![图片](assets/3d8ab26b9648.png)

而在Z13~Z15上演进了很多代的SC似乎被更加复杂的直接互联干掉了:

![图片](assets/6d9b47a325da.png)

### 结论

从混合云场景的交付方式和应用的部署方式来看，伴随着云原生架构和容器使用，以及数据处理量的急剧增加。当我们在开始闭门造车的时候，去看看大型机的一些技术和体系架构，以及它自身的演进趋势，相信你会找到属于自己的那个答案

#### Reference

[1]
IBM z13 大型机设计介绍: https://www.bilibili.com/video/BV1fK4y1f7FU
[2]
IBM Doubles Its 14nm eDRAM Density, Adds Hundreds of Megabytes of Cache: https://fuse.wikichip.org/news/3383/ibm-doubles-its-14nm-edram-density-adds-hundreds-of-megabytes-of-cache/