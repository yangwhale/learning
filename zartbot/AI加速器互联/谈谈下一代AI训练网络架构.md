# 谈谈下一代AI训练网络架构

> 作者: zartbot  
> 日期: 2024年8月12日 15:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491577&idx=1&sn=fe4b2d996720e04c3c952541692edc68&chksm=f9960f3bcee1862d5cd17667783ba7584a5bd2b098eacf08b91d34e2495be33a4558a51ca950#rd

---

看到Meta在CTSW上用了带HBM的Jericho2,buffer深度10ms, 还大言不惭说就此不需要DCQCN了? 反正22us延迟也不是个事情, 成本也不考虑, 连RTSW到CTSW 2倍带宽的事情都干的出来.

![图片](assets/7c8ccebac396.png)

于是我给这群人和他们的信徒们出个招, 而且还在以太网上可以很容易的实现在网计算, 根本不需要什么拥塞控制.  于是我们可以设计下面这种超级CTSW, 提供极致的**Smart Core Dumb Edge**解决方案.

![图片](assets/4d5848613f52.png)

直接把Intel Gaudi3当Jericho2来卖, 整个集群提供3072个200G接口, 并且提供总计高达32TB的buffer,既有以太网的ScaleUP, 还有ScaleUP和ScaleOut的融合, 还可以帮助Intel提升市值. 你们不是特别喜欢在网计算么, 直接把计算节点当网络节点卖不就行了?

![图片](assets/ca43856960d3.jpg)

大概算了一下,如果一颗Gaudi3大概10K USD,平均每端口成本也就1K USD不到呀. 看看Intel现在这个市值和大量Gaudi3的销路问题, 还不如把它当颗带HBM交换芯片卖了? 让他们给个骨折价估计还是愿意的.  接口带宽从Jericho2的每端口500MB直接提高到10GB, 翻了20倍, 整机带宽从Jericho2 576个400G提升到3072个200G, 容量提升了2.6倍.

这样做不光可以随路做Allreduce/Allgather, 让它做个Parameter Server来跑Optimizer都行. 甚至在推理过程中的Prefill-Decoder部署时, 还可以拿Gaudi3做Decoder. 模型参数也可以直接Checkpoint到这整个CTSW上的32TB内存上. 然后MoE也可以Offload一些到这些随路节点算.

整个网络要什么拥塞控制?PFC只在H100和Gaudi3之间一段, 整个集合通信的参数量都能在Gaudi-3上buffer住. 然后32TB做推理的分布式KVCache也够牛了吧?

如果嫌Gaudi-3和HBM贵, 那么又来一个乞丐版的方案, 找一堆X86配一堆400G网卡做到单服务器3.2T也不是不行?

## 后记

实在等不及明年愚人节发了,  当一些人踏上一条路到极端的时候, 那么就从这条技术路线上再往前多走几步, 看看是否荒唐?  当然我们也可以像前文那样编造各种荒唐的理由, 反正不是不用考虑钱的问题么? 包括UEC这群人搞什么Low-Latency Ethernet, 也让我想起几年前愚人节的一个玩笑...

[《IPv6- : 基于IPv5的48bits寻址互联网协议》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485564&idx=1&sn=0e40eebc00311795c4de65909a2ec220&chksm=f99618becee191a8b8e1579062e95da872737d70d73954260b74ddacf9bbb600d064e072cb34&scene=21#wechat_redirect)

祝您这条路上走的快乐~ 

反正我还是坚信Smart Edge Dumb Core这条真理

## [[Sigcomm2024论文解析]谈谈网络研究和工程方法](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491570&idx=1&sn=42f06db3f2dcb8f0244cef5c26f7fb50&chksm=f9960f30cee18626be86cb3f7b751a1bba02fc4fe45a55529240319af99b352de29a4b842b62&scene=21#wechat_redirect)