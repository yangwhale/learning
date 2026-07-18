# 谈谈大模型推理KVCache加速和内存池化

> 作者: zartbot  
> 日期: 2024年7月14日 07:03  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490427&idx=1&sn=759b9751469885ac0122943ea61ef2c4&chksm=f9960bb9cee182af8c17966f7dfe562df5bbda109612428f167279489cddea9395bef18151a3#rd

---

本文主要来谈谈PD(Prefill-Decoding)分离的方法对于整个推理基础设施的影响.

![图片](assets/6719f1523661.png)

从Mooncake的论文以及DeepSeek进一步将用户Context放到SSD上来看, 推理系统构建分布式内存池以及层次化存储的需求越来越强烈, 另一方面随着通用CPU算力的增强, 基本上也达到了L20**相等的算力, 例如AMD未来的Turin AI以及Intel的GNR,以及已有的基于ARM N2架构的Yitan 710等,将这些实例用来做Decode将进一步降低推理成本.

## 1. PD分离推理的瓶颈

### 1.1 KVCache转移路径

从当前的实现来看, KVCache从Prefill实例导出再到注入到Decoding实例, 整个I/O路径需要经历,如下图所示:

![图片](assets/16993be1d1dc.png)

整个路径相对较长, 我们再来看看一个标准的8卡GPU服务器的内部连接

![图片](assets/90cecce28900.png)

在推理过程中, 由于PD集群为异构架构, 训练网络通常是采用同类卡构建的同构集群构建的ScaleOut孤岛, 因此原有的ScaleOut网络中的GPU机尾网卡并没有承载流量, 而只是简单的采用连接CPU机头的网卡进行数据.

### 1.2 PCIe瓶颈

我们注意到由于GPU和CPU之间的连接受PCIe交换机**的一些限制和CPU自身PCIe Lane的限制, 通常每个交换机只有1~2个PCIe Gen5x8的连接接到GPU. 那么大量的数据搬移就必须通过这样一个相对较慢的总线, 同时还要受到CPU子系统的内存带宽影响.

### 1.3 基础设施现状

现阶段很多ScaleOut网络在考虑成本以及避免Hash冲突影响的视角下,采用了Multi-Rail多轨道的组网方式. 通常同型号GPU集群有数百张卡连接在同一个交换机上, 并且不同型号卡集群之间的ScaleOut网络并没有很好的互联带宽. 并且KV-Cache数据在Prefill节点和Decoding节点之间的只能通过200G/400G的FrontEnd网络互联, 相对于原有的ScaleOut网络1.6T/3.2T差距很大.

## 2. 一些解法

### 2.1 软件算法的角度

从带宽来看,主要是Prefetch**隐藏延迟, 细粒度的LD/ST降低Burst, 软件调度降低Burst带宽,以及KVCache量化降低吞吐量这几种做法.

由于FlashAttention3用上了Hopper的TMA和WarpGroupMMA,其Prefill的速度会更快,产生的burst会更高.

为了降低Prefill到Decoding实例间转发数据的Burst, 例如Mooncake采用Layer-wise的LD/ST来降低burst影响, 以及在Decoding节点采用Async Load的方式.也有一些论文在考虑做Prefetch

![图片](assets/4071dc976c9a.png)

上面这图来自于InferGen的论文,主要想借用一下谈Prefetch, 但是文章总想着搞critical KV的方式来降低KVCache Load的量这种做法个人觉得属于屎上雕花, 没有很大的收益,反而增加了整个系统的复杂性和模型推理的效果.

最后从容量来看, Cache Eviction策略和一些调度,以及如何缓存到SSD都是值得去探索的方向.

### 2.2 硬件互联的角度

事实上现阶段的集群互联带来的问题, 方佳瑞博士和章明星老师都有类似的观点

现在分离式架构都是用GPU训练集群做推理，节点内NVLink互联，节点间用IB或ROCE的RDMA互联。这种配置分离式架构完全是浪费，好比李云龙攻打平安县城，章明星称之为富裕仗。

#### 2.2.1 CPU实例加入ScaleOut网络

在组网结构上, 几个月前测试过Google的A3(H100)实例, 很有趣的是它的ScaleOut网络和其它的CPU实例是完全互通的, 那么就有一种简单的想法, 能否进一步分离PD, 把分布式的KVCache通过一系列通用CPU服务器构建, 来构建出一个更大规模的内存池,同时这些机器也可以通过其本地SSD或者云盘甚至是Offload到OSS进行KVCache的存储?

当然这有一定的难度, 通常这些Prefill集群的机器是为训练网络设计的, 组网拓扑针对训练网络的集合通信进行了优化, 针对KVCache的LD/ST这些流量类型,基于标准的RoCE**的拥塞控制能力还存在一些问题. 特别是针对Prefill/Decoding M:N的组网时, 以及Prefill/Decoding节点随着推理业务动态加入/移除集群时.

Google当然得益于它DirectTCP-X和Falcon的技术能力,可以将训练流量和通用CPU进行混跑, 至于其他的技术栈, 例如UEC们甚至是NV还有很多问题没考虑清楚.当然Google也有它自己的一些问题, Falcon还需要进一步演进才能成功.

#### 2.2.2 CPU实例进行Decoding

事实上我们发现Baidu最近在提《国产大模型第一梯队玩家，为什么pick了CPU？》[1]利用CPU推理, 同时我们也注意到阿里也开园了一个DashInfer CPU推理框架. 基于阿里倚天710 CPU也有一些来自ARM的优化《加速基于 Arm Neoverse N2 的大语言模型推理》[2], 而未来AMD Turin AI和Intel GNR的推理能力也会进一步加强.

### 2.3 系统的角度

从系统的角度来看,我们需要一个支持Longest Prefix Match的分布式内存存储和集群的动态任务管理和调度,并且支持异构的调度. 从开源软件的视角,我一直认为Ray+vLLM这样的方式是正解. 基于Ray的分布式内存对象服务可以很好的存储KVCache, 并且做一些亲和性调度增强, 在Decoding节点可以根据集群的资源, 用通用CPU配合vLLM进行计算, 这样这些CPU节点可以进一步做大KVCache Pool容量, 同时推理成本也会逐渐降低.

![图片](assets/72b77c049e3b.png)

个人是反对基于CXL**一类的内存池化的, 第一是本来就有FrontEnd/ScaleOut/ScaleUP三张网络,再进一步引入第四张网络成本受不了,同时CXL Switch一直没有看到很好的vendor以及CacheCoherence在这个场景本来就不是必须要的.

那么显而易见的就是这些内存池以何种方式提供, FrontEnd网络来看带宽显然是不够的, 标准CPU算力+内存构成一个MemoryTray挂载到ScaleUP网络是否可行? ScaleOut网络和通用CPU互联要怎么做? 硬件架构上存在什么约束.. 有些答案就不便展开了.

参考资料

[1] 
国产大模型第一梯队玩家，为什么pick了CPU？: https://mp.weixin.qq.com/s/Il22xfBnsYXrA3jo9q81NA
[2] 
加速基于 Arm Neoverse N2 的大语言模型推理: https://mp.weixin.qq.com/s/HUYA3LrtnDFW8t2MVC2FrQ