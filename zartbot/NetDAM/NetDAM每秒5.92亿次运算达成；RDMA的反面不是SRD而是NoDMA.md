# NetDAM每秒5.92亿次运算达成；RDMA的反面不是SRD而是NoDMA

> 作者: zartbot  
> 日期: 2021年12月31日 12:13  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487334&idx=1&sn=f10bed828e54b9a58538910a0a60f626&chksm=f9961fa4cee196b257f3f30ff017d5d767b2f57466c35accd2e98e67ccc5a605cf06799107c9#rd

---

NetDAM实现每秒5.92亿次运算

![图片](assets/36ed97f582a3.jpg)

原来的NetDAM设计包大小为68B，因此单个100Gbps接口只能跑到142Mpps，硬件团队在年末做了一点最后的优化，针对一些特殊的应用构建了一个64B的信号，最终单路实现了148.8Mpps的线速，而4路累加实现了**400G线速和5.92亿次每秒的计算能力**。

![图片](assets/1e94895ac22b.png)

这次NetDAM的变化主要是增加了一个Block RAM的区域用于高速高并发的读写和计算需求。

![图片](assets/a012763608fa.png)

主要使用场景是，例如**多台机器协同时的分布式锁，多机协同的信号量（semaphore）机制，或者是分布式数据库中的单调递增的UUID生成器等，这些对于分布式系统协同的性能提升具有非常重要的意义**，好像很多人还是看不懂：）简单的说就是一个服务器同时要满足4个100G接口收包并同时对一个64bit的内存空间每秒操作6亿(595M）次，然后还要以这样的速率把计算结果发送出去? 纯软件能做到多少？而这块FPGA自身的频率都没这么高怎么做到的？有些问题以后有机会看论文咯~

RDMA的反面是什么？是No DMA
昨天看到一个文章讲RDMA是个宝，AWS就是不鸟。其实很多时候RDMA有两个意思，一个是字面上的远端直接内存访问，另一个是狭义的指Infiniband或者RoCEv2的实现，总体来看SRD是一个很好的工程实践，参考下文：

》[**探秘AWS SRD技术**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487186&idx=1&sn=50dbac0d5dc9c4fe9e1eae4427ed90c7&chksm=f9961e10cee19706d7cacba9c4362d6cecbcf74bc7a40d046ee08e067d91b05d6e69f7488a02&scene=21#wechat_redirect)《

但是，RDMA的反面并不是SRD。而所谓的去RDMA更多的是将这个词替代为RoCE或者IB，即去RoCEv2，去Infiniband，例如AWS SRD的论文也是讲对比Infiniband，而不是RDMA本身

![图片](assets/1f86ea7c6ad3.png)
****

**主机之间的通信的本质是内存的拷贝，这一点上任何的优化bypass处理器bypass Kernel都可以称为RDMA。**只可惜有些人根本没想明白。至于渣说的去RDMA一方面是代指去IB去RoCEv2，另一个本质是在400G以上环境中才会出现的，DMA本身实现机制会出现问题，Jitter会导致DDIO等一系列技术出现问题，从而DMA数据需要写到主内存然后再读出来导致的问题。而100G的网络要去自己造轮子去掉DMA纯属没事找事。

》[**探索400Gbps主机网络**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487010&idx=1&sn=81b9d199299b1f98ec4934ef98879da0&chksm=f9961ee0cee197f6aefddb5d9da54b1367be8423984f3c267f613e38c8181a048d7c98dc9b86&scene=21#wechat_redirect)《

》[**RDMA ? or NoDMA!**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487040&idx=1&sn=9108a09638bb44a89a1926ac6a37476b&chksm=f9961e82cee197945cecc3d4d7b07e40d38ea94ff945848fd752c1cc03f52f8196732c8a126a&scene=21#wechat_redirect)《

还是记住那句话：

网络的本质是承载数据流，内存是数据流在某个时刻的快照，而计算是基于快照信息而产生新的数据流

**而通信的本质其实就是共享内存**，Don't communicate by sharing memory; share memory by communicating. 无论如何，都是需要在远端无感知的情况下去直接访问内存，从这个角度直译出来就是 Remote Direct Memory Access。**只是底层协议上是infiniband、还是以太网、还是iWARP、还是SRD、还是其它？**RoCEv2有很多缺点、iWARP也有、Infiniband也有，SRD可能也会遇到，但是**上层应用原语是不会改变的，只要是以主机XPU为中心的内存，通信的最简化模式都可以看作RDMA**，而RDMA的反面只会存在于如何考虑内存的中心位置应当把内存以处理器为中心？还是以网络为中心？poll mode还是push mode 更新内存？以及在一些存算一体化的器件上，点到为止，真正替代RDMA要解决的问题在下图

![图片](assets/e888eda1726f.png)