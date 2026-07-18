# 来谈谈英伟达的Blackwell

> 作者: zartbot  
> 日期: 2024年3月19日 16:04  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489037&idx=1&sn=41cf2a055e1623edb8e4c9812b0f5f32&chksm=f99606cfcee18fd91ae0c1b775cddecd8cef37c3f8560d69ff6822f319409d0f1f87dfed03c8#rd

---

Blackwell B100/B200的发布，有很多人来问我，先随手写一点吧，等最后架构出来了再写个详细的

![图片](assets/55a09af83413.png)

让我想起英伟达在7年前ISCA2017的一篇论文《MCM-GPU: Multi-Chip-Module GPUs
for Continued Performance Scalability》[1]

![图片](assets/bc1d25d7692e.png)

内部的互联结构如下，这是一个类似于NUMA的结构，论文里也探讨了L1.5 Cache的架构

![图片](assets/9da0bf092511.png)

具体的Blackwell架构等白皮书出来后再做一个详细的分析，对于前几代的架构，可以参考这个系列

[**》--〉GPU架构演化历史[从1980年SGI到2022英伟达H100]**](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=2538479717163761664&token=655423231&lang=zh_CN#wechat_redirect)** 〈--《**

英伟达的整个架构演进是一个小步快走的模式，2017年的Ampere架构中，增加了TensorCore，同时也增加了NVLink switch的能力，然后为了Blackwell这些MCM的架构演进，在A100中也分离了两块L2 Cache

![图片](assets/079e9a76a8a9.png)

从性能对比来看，NVLink带宽继续翻倍，显存也因为dual die可以放8块HBM3e，这样带宽和容量上让AMD的优势瞬间消失了。

![图片](assets/5950789116b5.png)

**抛开FP4这些指标看，就FP8/FP16 Tensorcore的性能来看，两个Die 实际性能提升了2.27倍(2250/990TFLOPS)，也就是说平均每个Die来看只提升了13.5% 看样子后续单芯片算力增长有些乏力了。**

当然最关注的还是GB200，主要是对搜广推这些现实可以落地变现的业务，还有生物制药/材料科学这些AI4Science价值巨大

![图片](assets/4f2e722418e4.png)

GB200可以构成一个NVL72的机架，隐约又看到些IBM Z16这些大型机的味道

![图片](assets/27aa383bdfc7.png)

![图片](assets/3b48960f5cd8.png)

![图片](assets/9ebf02acecaa.png)
大量的NVLink铜缆，布线真有趣，又有一点思科在2003年左右开始搞CRS Multi-Chassis的感觉，是不是过一两年，英伟达还要搞一个Blackwell Chassis 和Fabric Chassis？搞互联系统的人这些事情门清。

![图片](assets/a30b127682f1.png)

NVLink Switch性能也翻了一倍，用上了224G Serdes？

![图片](assets/359659c5d3f7.png)

如果要谈一些感受，看到GB200有一些30年前，SGI Reality Engine的感觉，熟悉的配方

![图片](assets/bc94fff5d51e.png)

然后也是大机柜来堆。。至于CX8的网卡，已经不支持以太网了，单独跟IB一起卖。以太网上还是BF3。

## 结论

对于Transformer这样既有Compute Bound又有Memory Bound的算子，然后内存墙影响越来越大时，出现类似的NUMA结构，然后又开始在互联架构上做文章。但似乎2层NVSwitch堆叠的576卡方案没有一个云厂商为它买单，当然这一次Blackwell里面也增加了RAS，也在可靠性上有所改观，而NVSwitch 2层组网的方案稳定性，我一直持怀疑态度。

而这一次的变革，或许使得整个架构上逐渐又走向分布式系统的架构，或许又会有“飞天5K”这样的故事再一次发生。而另一方面Scale-Out RDMA网络和Scale-Up NVLink网络地位也会因为NVLink Switch越来越强而发生更本性的变化，还记得Meta的论文么？

![图片](assets/124970cf718c.png)

本质上又回到一个经典的问题

![图片](assets/d63b571952da.png)

其实真的能够很好的把这两块吃透的人太少了，以至于很多会议上我都懒得开口多说几句....多层NVLink组网下的Fabric设计上，拓扑结构上，心里有了答案，笑而不语。 

不过又是一个激动人心，又逐渐遇到瓶颈的年代。看到CUDA，有一点点的想起了1997年的Glide 3D，还有买STB的3Dfx。历史还会再次重演么？

当然我坚信的是：算法是最后终结这次变革的唯一稻草，而AGI能够真正到来，一定是有新的数学引入，范畴论为纲，而代数几何/代数拓扑的一些很优雅的结论很可能是降低算力需求，避免幻觉的出路。

》--〉[大模型的数学基础](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)〈--《

参考资料

[1]
MCM-GPU: Multi-Chip-Module GPUs for Continued Performance Scalability: https://research.nvidia.com/sites/default/files/publications/ISCA_2017_MCMGPU.pdf