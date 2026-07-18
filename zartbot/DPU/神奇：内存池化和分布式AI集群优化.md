# 神奇：内存池化和分布式AI集群优化

> 作者: zartbot  
> 日期: 2021年10月16日 16:01  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486729&idx=1&sn=3d9005f846e12630f6ef3cc92d314500&chksm=f9961dcbcee194dda2f4358234430745b82665f50ed6ca5d7bcb8d7d7fe0e797cf2406e5899f#rd

---

首先感谢某个不知名的读者帮忙endorse了arxiv.org.过几天我们会把论文传上去.

今天有两个话题，一个是如何使用NetDAM做在网(in-network)和存内(in-mem) MPI-Allreduce计算，顺便谈谈某厂的EFLOPS。另一个话题是大规模内存池的构建

### 前置知识：分布式AI的通信

详细了解可以推荐给大家一本书：

或者去年我总结的详细的文章，以及后面关于AllReduce的实战，

[Ruta for AI：分布式机器学习的网络优化](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484313&idx=4&sn=25add69041d44118c9da667957ebc93b&chksm=f996135bcee19a4ddef76c39dbc5106cd4be4b3c4ace9aa7631e27565b86dc6eee6f516ef7cd&scene=21#wechat_redirect)

[MPI Note[8]: 分布式机器学习AllReduce](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485803&idx=1&sn=8fea4ef0df7f89c3c0ff281b8a5cbd8e&chksm=f99619a9cee190bfe8a038d31b6c44803dd90113e01da78ca52fe71876ffe5c337c4932f2b6b&scene=21#wechat_redirect)

[分布式AI训练：RoCEv2 AllReduce实战](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486299&idx=1&sn=065d89fd99d0e455de81248a182a9bc0&chksm=f9961b99cee1928f721bd0880685b973917a5745a9ae57419dacd8477a924e8898c0f883e4a7&scene=21#wechat_redirect)

分布式机器学习产生的原因很简单， 一方面是可供训练的数据越来越多，另一方面是模型自身的规模越来越大，所以必须要多个机器来搞。RoCE一类的通信协议自然被用到了，这其实也是nVidia要买Mellanox的根本原因，而并行的方法主要有如下两种:
![图片](assets/e386b65e1819.png)
`数据并行`很容易解释，主要是如何存储训练样本，并且在多机器之间传递混淆样本，基本上大家大同小异的都在采用SSD、分布式存储解决这些问题,当然还有`内存池化`的需求.

另一个问题便是`模型并行`,当单个工作节点无法存储时，就需要对模型本身进行分割。当分布式训练每轮迭代完成都需要将参数进行同步，通常是将每个模型对应的参数加总求和再获得平均值，这种通信被称为AllReduce
![图片](assets/9c6da0b12f27.png)

最开始的时候，是采用一个集中式的参数服务器(Parameter Server)构建,但是很快就发现它成了整个集群的瓶颈，然后又有了一些环形拓扑的All-Reduce
![图片](assets/840add147630.png)
而对于nVidia而言，它们极力的扩大NVLink的带宽，同时也快速的迭代NCCL，都是为了解决这个AllReduce的问题，但是这些只在单机或者一个极度紧耦合的集群内部。另一方面主机间的通信，自然就选择了超算中非常常见的RDMA ROCE了。

但是即便如此，AllReduce的延迟还是极大的影响了整个训练集群的规模:
![图片](assets/d6615597d873.png)

Allreduce算法简介可以参考鹅厂总结的：

腾讯机智团队分享--AllReduce算法的前世今生[1]

### 另一个工作：EFLOPS

阿里在`HPCA2020`上发布了一篇论文<EFLOPS: Algorithm and System Co-design for a High Performance Distributed Training Platform>,这也是国内企业第一次在计算机体系结构顶级会议上被收录，该论文系统性介绍了阿里巴巴的高性能AI集群的节点架构、网络架构、和通信算法，并展示了EFLOPS集群为阿里巴巴内部业务带来的价值。

阿里其实也看清楚了这个问题，PCIe的拥塞，内部调度的拥塞，网卡的拥塞：

![图片](assets/47e9a3e0abde.png)

然后解决方案很简单，反正钱多，一个GPU配一个网卡就好，然后网口多了，交换网也改成两套Fat-Tree

![图片](assets/39f3699ee868.png)

阿里的文章中有一个结论

![图片](assets/10987a2cfe80.png)

也就是说即便是用了HDRM，也就40Gbps的带宽了，那我先告诉你们NetDAM的一个结论100Gps轻松跑满，单个Alveo U55N可以跑满200Gbps，赠送一句话: In me tiger sniffs the rose.

### NetDAM实现AllReduce

首先不谈AllReduce的算法和相应的拓扑，在带宽一定的条件下的约束是`通信延迟`和`计算延迟`. 如果采用RoCE，从一台机器读和写都要经过一次PCIe，所以从根源上要解决这个问题就是内存前置,延迟不就下来了么？

![图片](assets/29aca7489512.png)

通信延迟降下来了，我们再来看计算延迟，传统的方式要怎么加：
![图片](assets/f818a4effabb.png)

而在计算域内，CPU嘛，AVX512加咯还能怎么样，带上Cache延迟抖动都不好控制，丢GPU上还要多一次Memory Copy，即便是直接使用GPU-Direct不也要过一次PCIe么？所以你跑不到线速100Gbps很正常

直接在网卡上放置大量ALU，收到包的时候，包还在SRAM buffer中，这个时候ALU根据包头的NetDAM Instruction,可以多个ALU同时去load本地DRAM，然后add到相应的SRAM里。加完以后，整个包改个IP头直接就转发，这样一个9000B的报文可以承载2048个float32，等同于AVX(32*2048)的SIMD-Add，所以我当然比你CPU快咯，而且加的时候没有DRAM的Store，只有最后一跳才会Store，又省了多少?

![图片](assets/3d765e68e3cf.png)

除此之外，针对AI训练的场景，还有很多可以直接通过NetDAM ALU过滤的方法，例如当一个SIMD内部的2048个float32有一半以上的0时，我可以很简单的使用<offset,value>的方式只传输非0值，然后远端SRAM内再对齐就好。

而在AllGather阶段,也就是说算好数据需要再次分发的时候,RoCE的组播似乎只是一个概念上的东西,而NetDAM则可以在这个阶段充分发挥以太网组播或者广播的能力，当然具体的丢包重传，这些都在NetDAM之间就可以完成，FPGA检测到Seq丢失直接产生一个READ报文给源就行了，压根不需要CPU参与，具体内容明天讲拥塞控制的时候详细说。

![图片](assets/823fb9ed7f8c.png)

关键还不止这一点，它还内带了一个Segment Routing头，可以做`链式反应`，就像`原子弹`那样~嘣~~~

![图片](assets/4187811a0a94.png)

链式反应另一个特点就是，打开了通向3D-Torus拓扑的新空间，毕竟连交换机延迟都省了,而且用RingAllreduce跑满带宽还不需要考虑incast，漂亮不？香不香？

![图片](assets/01c4bd76c245.png)

而Google TPU集群为什么要用Torus-Ring，甚至一些超算用6D-Torus，想明白了么？其实就是在扩展性上，Non-Blocking成本很高，而且临时扩大或者缩小集群规模需要添加额外的设备构成FatTree，Incast也不好控制，而Torus虽然是有阻塞的但是可以通过通信模式来避免阻塞。

![图片](assets/5bdb80fc500e.png)

即便是用Fat-Tree的数据中心,我们也给你们准备了Ruta的方案来做流量工程，比起那啥搞什么PortRank，更加简单直观的是哪儿不堵走哪儿~ 拥塞控制，明天给你们安排~

![图片](assets/9ba8d7a0e48a.png)

### NetDAM实现内存池

NetDAM是一个`标准`的`UDP`协议，NetDAM可以独立于主机`单独部署`, 因此可以构成一个非常大规模的内存池：
![图片](assets/be5110197d6c.png)

因此`普通主机` `用户态`不需要任何特殊的开发套件，直接一个UDP Socket就可以控制整个内存集群，爽不爽？

![图片](assets/7f2313370a4f.png)

而当你主机自己有了NetDAM卡了以后，可以玩的更High, 分区全局地址空间（partitioned global address space：PGAS）了解一下, 在这种场景下，我们可以把一个交换机芯片改造成MMU，对外提供一个虚拟的IP地址和UDP端口，然后构成一个大的虚拟化池隐藏内部拓扑。而每个netDAM报文访问的内存地址由交换芯片查表做地址转换到最终的NetDAM。这种情况下，交换机MMU还可以采用Interleave编址来解决内存局部使用过热的问题...

![图片](assets/3647317c468b.png)

继续从分布式AI训练集群来看，对内存池的需求主要是一个是训练数据集的分发和混淆，另一个是参数和梯度的更新。所以这次HotChip中Cerebras提供了一个Memory-X套件：

![图片](assets/a2dcb80cc782.png)

计算任务上，MemoryX还添加了Optimizer

![图片](assets/b1db0a9fc1ea.png)

结论 NetDAM也可以同样的实现这个功能：）

预告....EFLOPS谈完了，我们来谈谈HPCC？当延迟为确定性时，只需要考虑Buffer深度了，那么算法就更简单了：

![图片](assets/5c276f96d34f.png)

#### Reference

[1]
腾讯机智团队分享--AllReduce算法的前世今生: https://zhuanlan.zhihu.com/p/79030485