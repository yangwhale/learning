# AWS Re:invent一场教科书级的“科普”

> 作者: zartbot  
> 日期: 2024年12月3日 16:02  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492734&idx=1&sn=77419a75280c683653b08d6f38cb4a7b&chksm=f995f4bccee27daaf4fa3c0b3ea3ae508be5d6733e83ac53ed9a291120647b0ec6eaaffacae5#rd

---

AWS Re:invent 2024开幕之前惊爆一个大消息, 樱桃CEO Pat老先生被董事会退休了, 回望过去几年本来期待樱桃的GPU能够重返江湖像i740那个年代和NV/AMD(ATI)三国杀的, 可惜了..

而似乎国内媒体对这一次的AWS Re:invent的报道很少, AWS真的在AI时代落后了么? Maybe yes, maybe no.. 在Monday Night Keynote完全是一场盛大的科普活动, 新的东西也就Trainium 2 64卡ScaleUP的整机发布和去年发布的路由协议SIDR在10p10u网络的部署. 但是整场科普秀还是很值得去学习的, 可能更新的东西要等第二天AWS CEO的Keynote吧, 希望有惊喜.

### 云计算的本质

AWS对于其云计算业务的价值追求是从未动摇的. 一直都是: 弹性,安全, 性能, 成本,可靠性和可持续性这六个目标.

![图片](assets/45b0daed8158.png)

这种价值观的主张是云计算一切叙事的源头, 这些东西需要交了无数遍学费才能懂的, 只可惜通常交完学费就跟樱桃的Pat老先生一样被赶走了.

Peter以树根做比喻, 阐述了软硬件一体全栈优化的成功理念

![图片](assets/a994317a1d33.png)

### CPU

主要是介绍了Graviton系列处理围绕着性能和成本的迭代优化. 传统的处理器厂商通常以优化Benchmark跑分为目标, 而Graviton在云上大规模部署采集的真实工作负载的性能数据指导着它的微架构演进, 例如优化目标从Benchmark中的L2/L3 Cache**优化,到实际工作负载的分支预测器的优化.

![图片](assets/27c5709526e5.png)

最近一年Graviton在新增的处理器中占比已经50%...

### 安全

多租户的场景, 云计算的安全是整个业务的根基. 特别是现在攻击手段越来越底层, 整个基础设施从供应链开始到算力交付的全流程安全更值得关注:

![图片](assets/dd034f74d157.png)

![图片](assets/da70f3d1b14b.png)

Nitro**作为可信根构建了全流程的安全验证, 这一点上Azure/Google/阿里云都有类似的布局.

![图片](assets/00ac22a3dda8.png)

有一个更极致的安全措施是Nitro和Graviton之间的PCIe链路都加密了

![图片](assets/f3ef37b40d5c.png)

### 存储

在存储上, 讲了一个Barge服务器失败的故事. 最早做了一个超大规模的服务器, 能够放下288块硬盘, 重量达到了2吨, 部署运维很不方便, 同时这么多7200转的硬盘带来的共振非常容易导致硬盘损坏. 并且单台机器故障导致的数据丢失风险太高, 为了数据安全, 数据放置的算法也非常复杂.

![图片](assets/180bfb3ec86c.png)

解耦每四块盘配一个Nitro来构成一个分布式的存储服务, 在弹性/安全/稳定上都带来了更大的收益

![图片](assets/86c36b19b7c8.png)

### AI

这一节堪称是大模型全周期的科普. 首先是发布了Trainium2

![图片](assets/9cf3e10b97c8.jpg)

然后非常详细的介绍了光刻 Reticle Size约束了单颗芯片的最大Size, 然后又详细介绍了封装, 以及最大封装大概在2.x倍的Reticle Size, 还详细介绍了Interposer

![图片](assets/fbf4c0ec3861.jpg)

然后针对它供电做了哪些优化

![图片](assets/72ddeefb67c1.jpg)

接下来还花了很长的时间来介绍大模型训练的数据并行通信和Global Batch Size的约束, 然后在推导出ScaleOut不行,还需要ScaleUP, 因此发布了Trainium2 Server, 一个64卡的ScaleUP机柜

![图片](assets/f1f04efe1b5c.jpg)

为了可靠性,整个结构非常简单, 前面是8块Nitro卡做ScaleOut网络, 后面就是两块Trainium2

![图片](assets/3d15a22f64b6.jpg)

然后又继续科普了一下微架构, CPU和GPU为什么不行, 特别是GPU SM之间的通信虽然有Distribute SMEM, 但大量的计算还是要通过L2Cache**和HBM倒换.  为了避免这些无效的Cache/内存占用. AWS和Google一样选择了脉动阵列的方式

![图片](assets/e2b020b2492d.jpg)
这样就缓解了内存墙的影响

![图片](assets/5a94f9e2317c.jpg)
生态上开发了一套新的编程框架

![图片](assets/1568b4930d74.jpg)

但是个人觉得似乎不一定是终局, 最近被Hopper的TMA**/WGMMA的一堆像猴子一样的代码折磨中, 总觉得CUDA SIMT在GEMM时代似乎整个工业界都没有一个很好的微架构抽象, 加上国内HBM被禁, 这个话题倒适合整个工业界大家一起探索一下.

然后就讲了一下它的ScaleUP NeuronLink, 也是一个NOC总线协议外扩的

![图片](assets/e47b31c1fcaf.jpg)

最后展示了一下64卡机柜的实物, 看上去更像是一个TPU机柜.

![图片](assets/ea74d658bd0b.jpg)

然后就开始科普推理的Prefill Decoding, 以及如何用Trainium2混合执行两个阶段.

![图片](assets/0cd518ef94f7.jpg)

还不忘diss一下竞争对手, AWS的延迟更低

![图片](assets/6cd7111ddf33.jpg)

最后邀请Anthropic**的联合创始人Tom Brown发布了Project Rainer的超大规模集群(数十万个Trn2)

![图片](assets/008d5729af3d.jpg)

### 网络

阐述了云网络和AI网络的不同

![图片](assets/d9f11b9cbf58.jpg)
发布了10p10u的网络架构, 其物理架构和以往的CLOS组网并无区别,

![图片](assets/a18f5ce70860.jpg)

主要是规模支持10Pbps级别, 延迟10us

![图片](assets/bfaea080e18a.jpg)

然后考虑布线复杂, 做了一些特殊的光纤防止布线错误,同时还有一些在布线时的光路检测小工具.

![图片](assets/e1f3c49ddd4a.jpg)

![图片](assets/232f0f2f5114.jpg)

部署规模来看10p10u也增长非常快, 已经有350条光缆了

![图片](assets/bf7e8f5e9cf5.jpg)

同时AWS对光路的可靠性也做了很多优化, 失效率降低到了千分之一左右

![图片](assets/ab0b0aa6f76d.jpg)

### 路由协议

为了解决大规模的组网, 特别是未来十万卡百万卡集群的组网. 传统的路由协议无论是分布式的还是集中式的都面临很多挑战.

![图片](assets/2c086d2ae730.jpg)

![图片](assets/559b9e254a89.jpg)
于是采用了自研SIDR(Scalable Intent Driven Routing)协议.

![图片](assets/8f264f2077bb.png)

其实这个问题渣B在几年前就预见过了, 并且开发了Ruta路由协议, 基于集中式的策略控制面和分布式的路由决策面来解决这一系列问题, 实现了广域网和数据中心网络统一的流量工程可以访问   [Ruta专题](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3752163729702502404#wechat_redirect)  

[《Ruta实战及协议详解》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485165&idx=1&sn=412fcb1dd46dd4ef4384a033b0827256&scene=21#wechat_redirect)

[《Ruta: 替代VXLAN+BGP-EVPN的数据中心部署场景》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484414&idx=1&sn=6c36bde5399a0cd6fa8016b421dd90d7&scene=21#wechat_redirect)

AWS也是类似的玩法 p10u10带来的超大规模组网

![图片](assets/4af33e5731b3.png)

![图片](assets/5d0b2ad11001.png)

设计了一个混合式的路由协议

![图片](assets/8b724ec8d8d3.jpg)

Ruta也是这样

![图片](assets/ac61d9d646dc.png)

AWS也是同样基于CAP Theorem去讨论控制面和数据面的一致性需求

![图片](assets/0dfda68d4693.png)

这一点我在几年前的一篇文章中有详细的阐述, 控制面负责集中式策略管理,采用CP. 而数据面采用AP, 通过SegmentRouting来解决一些一致性的约束并实现BASE. 这样就在组网规模和意图管理易用性以及全局可靠性几个方面做到了最优.

[《包处理的艺术(2)---如何设计协议》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484550&idx=1&sn=0627d92a24590316a573af70f24cb3f0&scene=21#wechat_redirect)

[《分布式路由协议设计:从复杂系统和脸书故障谈起》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486624&idx=1&sn=de852b2c55c9f4af67f323eb1961c304&scene=21#wechat_redirect)

大概就这样吧... 其它CSP在针对十万卡集群组网时的故障处理和路由快速收敛上, 以及解决Hash冲突看看SIDR吧

当然渣B在设计eRDMA的多路径拥塞控制时已经把这个问题从端侧解决了, 做到了ms级别的收敛, 压根就不在意BGP导致的路由黑洞, 哇哈哈哈, 留给你们挑战困难吧, 时隔一年多了还没人追上来...

![图片](assets/8e5c2d629c11.png)