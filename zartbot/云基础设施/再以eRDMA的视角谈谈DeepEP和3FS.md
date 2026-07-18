# 再以eRDMA的视角谈谈DeepEP和3FS

> 作者: zartbot  
> 日期: 2025年3月18日 22:15  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493544&idx=2&sn=ee019d6cc347c1b13e08693a4dd69b2b&chksm=f995f76acee27e7c2298074bea57bedc2c0c7dca783c38f678555f7483f2cb99a2e21da5bd85#rd

---

本文仅代表作者个人观点, 和作者任职机构无关

写这一篇据说是渣B太卷了, 影响了很多做推理的同学... 那就闲聊一下,大家都缓缓? 换我去卷另一帮人?

写这个话题主要是在做DeepSeek-V3推理仿真分析建模的时候, 针对EPLB延迟建模时回想起了前年设计eRDMA拥塞控制时就考虑到了MoE的场景. 另一方面是关于3FS看到一篇文章要吐槽一下.

### 1. 关于3FS的写性能

首先来谈谈3FS, 吐个槽....   起因是看到友商有一个文章介绍3FS时, 谈到写的性能...

![图片](assets/b517bba6d9be.png)

但是好像不是这么回事, 毕竟DeepSeek的论文Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning[1]写了能到10GiB/s的.

![图片](assets/9026092d392d.png)

**当然我也能明白一些“自研肯定比开源好”的心态, 但做事情还是要更加严谨一些.** 我们eRDMA团队的同学在阿里云上随便开几台机器, 轻松调整一下配置就能复现10GiB/s, FUSE Client的截屏, 500MiB文件对齐, IO大小4MB对齐, 100个job,每个job 100个文件, 累计10000个文件对齐.

![图片](assets/df43697961e6.png)

![图片](assets/f1ca2f2398da.jpg)

3副本存储节点网络带宽, **请注意这还是在云上VPC环境, 和大量公有云的TCP业务混跑时就能轻松达到线速, 压根就不要搞什么DCQCN和PFC, 跑的好好的...讲真RDMA的拥塞控制被NVidia的一群人带歪了....**

![图片](assets/cfdd084d2f0c.png)

其实工程上没有什么大问题, 都是大量的小问题要去处理干净了, 就能带来质的提升. 刚发布的3FS, 因为磁盘的sectorsize在代码里被定义为了ALIGN_SIZE=512, 导致很多sectorsize为4K的盘构建的3FS存在写入I/O没对齐4K时导致I/O Hang的问题, 已经联系DS的同学进行了修复. 另外测试过程中当写入压力过大时, 存在meta server 文件i/o ulimit的限制和Client到存储节点RPC Timeout timer的修改, 以及Client到存储节点QP数的修改等.

当然我能明白3FS默认的QP数量限制到较小的数值是因为Mellanox存在一些QP-Scale的问题, 这些细节在eRDMA上早就处理干净了, 我们支持128K QP还能同时打开多路径并维持非常高的吞吐, 对于我们而言, 因为在云上要服务大量的客户和单个节点多个虚拟机, 这些硬骨头只能啃掉, 不可能去给客户一个限制QP的版本.

再回到3FS本身, 我觉得工程上的事情吧, 到了终局都是讲取舍, 不用神话DeepSeek,也不要轻易的去贬低. 看到很多做存储的人对3FS的评价并不是很高. 其实我个人的粗浅认知是, 数据的可靠性,不丢不错,全链路的校验和备份等, 这些工程上的细节起码要十年时间才能逐渐踩坑爬出来调稳定. 然后3FS采用了一个很简单Chain based replication, 又是多副本, 没有采用星型的复制和EC来节省存储资源.Meta server的可靠性, 集群的动态缩扩容的方案, 这些从传统的存储的视角来看都是问题.

相反我要说的是, 如果把它作为一个高吞吐的Read Heavy的临时存储, 确实是一个很好的方案. 例如量化交易每天的行情数据快速写入, 各种并发的策略回测. 3FS配合Smallpond在这一块的优势还是很明显的. 毕竟针对接近1PB的数据, 全部放内存又放不下, 通过一个200Gbps~400Gbps的网络I/O快速取数据还是很靠谱的. 至于三副本,我个人认为主要的用途还是加速读的速度,而不是简单的做可靠存储.

对于大模型的KVCache, 其实3FS写入数据还有很多优点, 例如最简单的一个事情就是不需要过MetaServer,文件open后, 后续任何块的写都不需要和meta通信, 根据本地的stripe和全局chaintable直接落盘即可. 我觉得这是一个五脏俱全,小而美的针对特定场景的一个文件系统, 个人反而对一些要提支持星型复制和EC这些东西持反对态度. 工程上最难的问题就是:不要什么...

### 2. 关于DeepEP为什么要用AR和IBGDA

工业界看到LowLatency最大的直觉就是去直接测延迟, 你会发现IBGDA用NVSHMEM比直接用CPU测ib_write_lat --use_cuda延迟还高, 然后开了AR延迟更大. 其实当你建模这个延迟模型时, 就很清楚了... 如果你知道排队论的Kingman公式, 这个问题就好解释了

![图片](assets/78b778023faf.png)

到达速率如果解决好了网络上的拥塞基本上可以把网络带宽用到90%, 处理速率取决于GEMM效率. 整个队列延迟影响因此有两块, 为数据从网络到达速率的变异系数(coefficient of variation), 而 是服务时间的变异系数. IBGDA通过GPU直接操作网卡, 然后通过hook的方式调度计算Kernel的实质是将 是服务时间的变异系数降低.

![图片](assets/9fe274de7d9b.png)

而打开AR本质上是降低 . 然后再宏观的配合EPLB获得更加均匀的到达速率, 然后做负载均衡.

实际上这个问题和网卡的静态延迟什么的半毛钱关系都没有, 开了AR也要增加几个us呢... 网络这个圈子的人现在已经很少有人能够认知到这个问题了... 这也是我为什么一直讲, eRDMA多路径拥塞控制算法设计时, 我们要去考虑incast的情况下, 不光要打满带宽, 流量间的差额足够的小...本质上就是知道kingman公式的影响, 网络上必须要消除 , eRDMA的incast数据如下, 带宽打满时变异系数CV只有0.00x~0.04, 自己算算为多少?

![图片](assets/e4a02853fcea.png)

留给其它厂家找差距去吧, 其它包括Nvidia...(逃跑....

### 3. 关于DeepSeek-R1推理的性能估计

其实渣B是站在网络/互联以及模型本身的架构视角来估计的, 对于芯片设计通常耗时几年, 因此需要在模型结构上对新的算力平台有一些预估, 如何修改模型架构来适配更新的硬件平台. 另一方面是估计上尽量对计算少打一些折扣, 这样对网络上的压力才能充分的暴露出来. 所以结论上的TPS数值会有一些偏高. 后面做一些修正再更新一些详细的数据, 代码也在整理完了以后开源出来...

[《DeepSeek-V3/R1推理效率分析-Blackwell性能估计(V0.4)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493504&idx=1&sn=45d4ceb21a51b013d0acc83a79d495a4&scene=21#wechat_redirect)

参考资料

[1] 
Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning: *https://arxiv.org/abs/2408.14158*