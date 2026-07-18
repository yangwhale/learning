# 谈谈SDR-RDMA, 所谓的软件定义的可靠性

> 作者: zartbot  
> 日期: 2025年6月8日 23:51  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494211&idx=1&sn=691366cbd0cecd6f13c119300a834fa1&chksm=f995fa81cee2739726c3307203ee75e0a2ecde9af4a4fea016ad2ed05d77b5bd3cba54d92741#rd

---

### TL;DR

今天看到一个月前ETH, NV和微软联合发表的一篇论文SDR-RDMA: Software-Defined Reliability Architecture for Planetary Scale RDMA Communication[1].

大概的叙事逻辑是美国电力紧张, 所以很多时候要跨数据中心传输, 那么在长传链路上高延迟和丢包下如何设计可靠传输协议, 而当前RNIC又有各种硬件ASIC的限制, 无法快速适配新的协议. 于是这群人就想到用DPA构造了一个软件定义的可靠传输...主要用了一个bitmap机制和一些基于纠删码(erasure code,EC)的改进.

而我想吐槽的一点是, 这不就是RNIC或者更指名道姓的说NV的网卡设计架构缺陷和RoCE协议缺陷导致的么? 

跟我念20年前发明的IWARP全称: Internet Wide Area RDMA Protocol... 天然就支持internet wide area的长传, 然后作者们重新发明的bitmap机制, 实际上无非就是iWARP中DDP(Direct Data Placement)的另一种表达方式, 并且协同一些多路径传输能力, **我们在CIPU 1.0/CIPU 2.0上的eRDMA早就实现了的...**

实质性的问题: **我们必须要正面应对Lossy网络**, 很遗憾看到某些厂商在Lossless上来回折腾, 从最早的RoCE基于PFC的Lossless, 再到IRN实现Lossy的Selective Repeat, 再到Spectrum-X回到PFC的路上, 而如今又不得不面对Lossy的链路进一步改进SR.... 某厂折腾了太多的弯路.... **真男人就得正面硬刚Lossy呀...  顺带多一嘴, NVL72的很多问题也来自于Lossless..为啥UB就能成呢?**

### 1. 概述

从论文的Abstract来看, 主要讲的是RDMA对于数据中心间的高效分布式训练至关重要, 但是跨数据中心的毫秒级延迟使得可靠传输设计变得很复杂, 特别是在长传链路上, Selective Repeat的效率很低. 为了在现有的硬件上替代这些设计缺陷, 提出了利用BF3的DPA卸载构建一个基于软件定义的可靠传输协议栈, 通过引入接受接收缓冲区Bitmap的方式构建.

设计SDR的动机很简单:

单个数据中心规模受限于供电, 随着训练资源需求增加, 超大规模的云服务提供商开始探索多个数据中心协同执行单个预训练任务, 然后就需要跨域数据中心的长传链路承担RDMA传输.  然后当前的这些卖螺丝的RNIC(CX7/CX8)为了支持数百Gbps的吞吐, Go-back-N和选择性重传(Selective Repeat,SR)都是ASIC实现的, 在高延迟高丢包的长传链路上效率低下.

因此, 借助于NV网卡上的DPA可编程能力, 实现了如下这种软件定义的可靠传输(SDR):

![图片](assets/f41601b72118.png)

最底层是一个在DPA上软件实现的可靠传输中间件, 将底层的数据包处理细节和上层的可靠传输逻辑解耦. SDR通过在接收端引入了一个“部分消息完成”的API语义, 来实现, 部分消息完成通过一个bitmap表示.

### 2. DC间通信的挑战

论文中说长传链路上存在的丢包概率测量如下:

![图片](assets/88cd9063d2ac.png)

丢包概率基本上在千分之一到百分之五这个区间, 和数据中心内的情况差了2个数量级. 还记得当初RoCE的假设么? 因为很多HPC workload基本上不丢包, 丢包概率小于10e-5, 所以做个简单的Go-back-N就够了... 从第一天偷的懒没想到如今成了大麻烦.

然后阐述了如果要做到RC Verbs兼容的可靠传输, 据作者的认知, Broadcom, Intel, Microsoft, and NVIDIA 仅支持基于重传的RC, 并且通常在ASIC硬件中实现, 然后如果要使用新的可靠传输或者基于EC的可靠传输, ASIC开发周期通常要3~4年. 因此作者相信, 基于软件定义的RDMA可靠传输是一条正路.对,你们终于认识到这点了...

很搞笑的是, 这群人为啥不睁大眼睛看看Google Falcon和阿里云的CIPU eRDMA呢? 这两家都可以支持完全RC兼容的可靠传输, 并且都能够实现在5%的丢包率下维持90%的Goodput.. 当然Google Falcon当前只有200Gbps, 阿里云则可以做到400Gbps. 而且阿里云从第一天就是Software Defined Reliable...

这群作者的认知真的是有问题呀....

然后, 作者谈了一下传输协议设计的挑战. UD在包乱序情况下需要主机CPU或者网卡实现保序. 而UC不会出现乱序的问题, 然后作者基于UC和上层的可靠传输RC之间构建了一个软件定义的轻量级的中间件(SDR-SDK). 每个Chunk都和MTU align.. 那么我传输一个hidden-dim= 7168/5120的token呢?在iBGDA的情况下如何处理呢?

### 3. SDR实现

消息传输的方式如下:

![图片](assets/863260e7af63.png)

在接收端构建一个bitmap, 发送端采用WRITE_with_IMM消息, IMM中携带MessageID和PacketOffset, 然后触发CQE, CQE在接收端的DPA上消费, 并更新bitmap.

![图片](assets/9ba949f12e1e.png)

然后可以在对用户侧一个RC QP的基础上, 在DPA实现多个UC QP轮询来实现多路径的转发能力, 但是你如何保证所有的UC QP都是Disjoint Path呢? 有了Hash冲突RR的算法依旧歇菜咯...

这群人为什么不了解一下什么叫iWarp DDP呢? DDP在20年前就定义清楚了, 消息头包含消息序列号(MSN)和消息偏移(MO), 然后直接就能用呀, 即便是报文经过多路径乱序也可以处理...

![图片](assets/28c68325376c.png)

### 4. EC

这群作者还搞出了一个基于EC的方案

![图片](assets/8215bff68968.png)

Cisco大概在10年前就搞过类似的方案,  作者也提到了其它例如CloudBurst/LoWAR的实现...

![图片](assets/9dfb1a7f781b.png)

说实话应对广域网上的拥塞导致的丢包, EC是没啥用处的...很多时候中间的设备在burst拥塞时有显著的连续丢包, EC反而性能更低损失更大....

### 5. Evaluation

作者有一个阐述很搞笑, 按照大包的MTU来算, 他们的SDR能够LineRate并扩展到3.2Tbps, 想以此证明软件的开销并不大.

![图片](assets/aba08635b679.png)

然后, 在很多场合又很双标的要追求更高的pps来反对基于软件的实现...

### 6. 锐评

再来看看[《RDMA这十年的反思1：从协议演进的视角》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489240&idx=1&sn=53c7512d8551a44834bd405fd38b15dd&scene=21#wechat_redirect)里面的这个图, 一直在说RDMA最近十多年走了很多弯路, 总是没人信... 这不又逐渐走回去了么?

![图片](assets/2343fbb04d6d.png)

Google Falcon前几个月终于上线了, 名字叫Cloud RDMA, 而且是RC兼容的, 但是IBGDA的代码还是需要一些细微的适配, 因为那里面都是针对Mellanox硬件hardcode的.

而AWS SRD现在面临RC不兼容导致IBGDA based DeepEP难以部署, 然后长传就更别提了, 连跨AZ都不支持...

基于PFC的Spectrum-X方案无法应对长传, 这篇论文不就又开始讨论Lossy了么...

而UEC的官方标准一直难产, 看到一堆libfabric的定义, 不知道他们对AWS SRD无法支持iBGDA, 对RC兼容这件事情有了几分敬畏之心呢?....

整个工业界学术界从RoCEv2开始搞PFC绕了很大很大的一个圈子, 最终又回来了. 要软件定义, 要做iwarp类似的DDP的工作, 但凡有一点敬畏之心, 都不会说自己是一个Novel software-defined RDMA stack...

阿里云CIPU eRDMA从第一天就意识到该走这条路, RC兼容, IBGDA也能跑, UCX这些都没问题.  而SDR的工作无非就是重新发现了iWARP DDP而已, 并且也实现了类似的软硬件协同的架构...

另一方面, 作者的这个实现还有一些问题, 基于Rate-Based的CC还存在不少缺陷, 特别是长传上, Window Based CC才是正路呀...

最后问一个问题:  当通过bitmap实现的Selective Repeat和SACK本质又有多少区别呢? 然后再把Window Based CC加上和IWARP的区别在哪里呢? 或许只有一个IP Protocol字段的差异了...

这个图才是正解

![图片](assets/130e2fc2d6dd.png)

实质性的结论是, **可靠性和传输效率需要算力来置换**, SDR好的一方面是终于明白了这一点, 但是很有可能又在下一代网卡实现上跑偏了, 因为软件的算力不是那么容易搞的, 特别是可靠传输和拥塞控制耦合在一起的时候....当然这也不是什么难题, CIPU eRDMA不就搞定了么? 根本就没必要去搞什么多平面, 单个网卡出8个接口这么荒唐的事情...

另外, 隔壁友商又基于NV的RNIC改了改, 发布了一个某字母RDMA over VPC... 不知道上次某开源文件系统40GB/s的性能达到没? 

参考资料

[1] 
SDR-RDMA: Software-Defined Reliability Architecture for Planetary Scale RDMA Communication: *https://arxiv.org/html/2505.05366v2*