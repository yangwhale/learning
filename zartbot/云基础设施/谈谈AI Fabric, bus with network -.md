# 谈谈AI Fabric, bus with network ?

> 作者: zartbot  
> 日期: 2024年7月14日 16:01  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490438&idx=1&sn=b419be65a0d5c015b4ff36f370532a1e&chksm=f9960b44cee182523ad3257b8000f1772bbd78ae7fc03638c2191d8c07d1c1fdc69c15db2423#rd

---

夏Core的《AI fabric is a bus or a network？》[1]这篇文章中很多观点非常正确, 从屁股而言我大概两年前就叛逃网党, 只是中间顺手帮网党解决了一下Lossy和Out-of-order的问题, 另外被计党毒打过了完全认同其中很多观点, 特别是下面这段话是肺腑之言, 价值10亿美金的胶带.

![图片](assets/b5c822a9f748.png)

`这个观点其实和UALink不谋而合, 我以前对UALink的误解是它只是基于PCIe的Phy演进太慢持有反对态度, 但看到Ethernet Integration这个路标就彻底认同了`

![图片](assets/8ec9eaada11a.png)

从生态的角度来看, 原来有大量的代码要跑LD/ST还是要有的. 但是又面临两难的问题,这样的互联交换芯片谁来做呢?毕竟国内还是没有像HW这样的企业,就连AMD也得跪求BRCM搞UALink.

选择Ethernet ScaleUP其实更多的是在考虑`可获得性`的问题, 国内大多数的GPU厂商既要自己流片搞GPU,又要搞XX-Link Switch, 短期内还要快速追赶难度较高, 毕竟不是每个国产厂商都有这个能力的.

从当前的workload来看,大模型的矩阵规模相对较大确实不太需要像传统HPC那样的Stencil一样的细颗粒度的访存, 毕竟买以太网的交换机芯片128Radix 51.2T~ 204T的演进基本上都能看到的. 选择Eth并不是屁股而是另一种妥协. 另外从GPU上来看, 一方面是成熟IP的可获得性, 另一方面由于国内算力密度受限的情况下用一些较大的I/O占片上面积也是一个取舍. 还有反正通信库ScaleOut要搞RDMA,顺手搞一把做个短平快的方案占领市场.

当然完全放弃LD/ST的风险非常大,例如对传统的HPC以及夏Core提到的Sparse矩阵的场景, 以及未来一些GNN场景的影响非常大.

其实使用Ethernet做ScaleUP互联这句话还有一些定语.

![图片](assets/422d4472f174.png)

一方面是ScaleUP连接CPU系统的价值在于我们可以走出一条非NVLink-C2C的路出来, 现实中推理的KVCache管理和基于CPU的Decoding还是有需求的.

[《谈谈大模型推理KVCache加速和内存池化》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490427&idx=1&sn=759b9751469885ac0122943ea61ef2c4&chksm=f9960bb9cee182af8c17966f7dfe562df5bbda109612428f167279489cddea9395bef18151a3&scene=21#wechat_redirect)

另一方面,我也不认同直接在ScaleUP上跑RoCE, 并且一直在很多场合强调NV在搞的GPS/PROACT/FinePACK.

![图片](assets/178a175ced95.png)

GPS对计算核还是一个LD/ST的内存语义, 根据不同GPU之间的Kernel进行订阅,并配合PROACT隐藏延迟

![图片](assets/9b1d49e6c346.png)

最后再通过FinePACK一类的机制打包成一个较大的message发送

![图片](assets/2210e377c36e.png)

其实这也就是夏Core谈到的

![图片](assets/01bd9be51ab3.png)

如果没有前置的这些工作, 直接用RoCE的方案是完全否定的. 特别是还有一系列多路径的调度拥塞控制等复杂逻辑的做法, 当然是明确反对的. 在这个位置, 正如几年前在做NetDAM的时候分析的, 我并不认同RDMA:直接扩展主机内总线.

对于拓扑, 我也一直反对任何异构的非对等拓扑, 虽然短期内解决了很多问题,例如多路径等, 但长期是无法演进的. 至于Lossless嘲讽的是那群做PFC魔改的人, 和夏Core无关,只是那一篇发表的时候刚好和两边的人都谈了, 对DCN和NOC夏core的见解有感而发.

另外, 选择Ethernet并不代表要选择RDMA, 只是一个在什么都没有的时候作出的妥协, 先把业务跑起来. 例如国内大量的推理市场需求下, 通过简单的更大带宽的互联把1T多模态的模型承接下来.

话说计党和网党的争论看样子还会继续下去, 但是我认同夏core说的, 而中国的比例，大家都懂的，做网络的太多了些 ：） 不要因为网党而放弃了LD/ST的路, 另一方面计党很多人为了LD/ST还期望继续用PCIe Switch来做ScaleUP也是一个误区.

例如夏Core和UALink都谈到的

统一的物理层和链路层之上，Load/Store/Atomic或Read/Write/Send不过是基于带宽、距离、Topology的取舍之间的变化罢了。

那么有一个问题来了, RackLevel用Ethernet Phy来做LD/ST的一个小规模ScaleUP总线, 如果这里面是黑猫, 那么再上一级到Pod Level呢,或许就是白猫? 中间自然会出现一个类似于协议转换的东西, 熊猫就在这个地方, 希望也是计网两党能够成功会师的地方.

参考资料

[1]
AI fabric is a bus or a network？: https://zhuanlan.zhihu.com/p/708602042