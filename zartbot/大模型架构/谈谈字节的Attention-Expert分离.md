# 谈谈字节的Attention/Expert分离

> 作者: zartbot  
> 日期: 2025年4月5日 01:49  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493791&idx=1&sn=6fae3c920421b478e4d98346e27c9540&chksm=f995f85dcee2714b18c8c97df59dfd04d2b8345d7c50dafdddb14891bfe2144b139729cd8e14#rd

---

看到一篇字节的AE分离(Attn/MoE)的文章《MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism》[1] 挺有趣的. 但是简单的说字节和NV都还需要技术扶贫....

文章有一个非常简单的叙事, Microbatch, 然后M:N的Attn:MoE配比并配合异构算力来降低成本.

![图片](assets/ff91baa84abe.png)

![图片](assets/a5bfab2b035a.png)

其实本质的问题是, 加大BatchSize后,如果按照DeepEP的方式来看, 显存容量和一些低算力卡(H20)在Attn计算上太慢带来约束, 高算力卡(H800)在小的batchsize下Expert的GroupGEMM计算利用率又太低,显存80GB又比较难拉高batchsize,退而求其次只能选择大规模EP(144/320)并行.

字节的同学提供了一个成本视角核算的表格, 并根据模型推理的算力需求和SLO的需求构建了一套约束搜索算法.

![图片](assets/d81e3d69570b.png)

搜索算法倒是很好搞的, 其实很容易的改一下shallowSim就可以算了, 本质上枚举完各种组合下的数据, 然后pandas查表就好

![图片](assets/e3091fc29597.png)

下周有点空了去把GPU参数中增加一个价格参数, 然后再做点性价比的计算就好.

主要难点还是在通信上, 字节把同构的All2All通信变成M:N的Mesh通信,实际上还有很多问题没处理干净. 先来看看字节的说法

![图片](assets/18067d622bf1.png)

首先是通信库的问题, NCCL test为什么比perftest高那么多? 特别是在P99的时候? 其实字节的解释没有抓住最根本的问题, Kingman公式来计算队列延迟才是关键呀.  然后问题定义不清楚的情况下,做了几个优化. 一个是自己搞了一套通信机制

![图片](assets/32b5598877be.png)

但这样弄远没有DeepEP LL-Kernel那样直接用IBGDA干净呀. 然后针对网络上调整了拥塞控制算法和提高了ACK的优先级? What's the problem? 其实更直接的叙事应该是引用Kingman公式, 然后想办法在网络上和计算上降低变异系数.

![图片](assets/bbdf12b7f999.png)

计算上的变异系数控制, DeepEP明显做的更干净, 一个hook函数很快的能够拉起计算就行了, 虽然字节也是类似的优化通过flag来控制.

而网络上,本质就是Mellanox(Nvidia)网卡的设计缺陷, 即便是开了AR还会有几个微秒延迟的上升, 主要是在接收端ReOrder的实现上, DDP的作业抄的不干净...我们在两年前设计eRDMA拥塞控制算法的时候就考虑过AE分离的问题, 因此对接收端的incast情况下的变异系数的考虑远高于带宽利用率, 当然最终的结果是带宽又能打满,变异系数又几乎为0,多路径打开和关闭延迟没区别,甚至开了由于单个QP可以在两个网口上传输延迟更低.

然后在接收端ReOrder设计上规避了RoCE协议的缺陷, 直接用iwarp DDP就很容易解决了呀. 下图eRDMA两年前的benchmark, Nvidia到现在还没追上...

![图片](assets/7205a93e3c70.png)

其实还有一个问题是在字节《MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs》[2]提到的

`Reducing ECMP hashing conflicts`. The conflict probability is reduced as the bandwidth of each uplink is double of that of a downlink. Second, eight 200G NICs on the server is connected to eight different switches in a `multi-rail way`.

ECMP hash冲突的问题, 然后当你需要M:N的AE分离时, Multi-Rail way 如何组网呢? 当然钱多换IB, 稍微钱少一点买SP4+BF3开RoCE的AR,但又是Lossless的...
![图片](assets/0f2630ecd76a.png)

还有一种方案是, 如果给我用BlueField3L的网卡, 可以做一个比较hack的Lossy多路径方案,GPU和BF3L建立一个QP,然后利用BF3的DPA去从多个QP发送, 并且每个DPA Core还需要探测路径上的RTT并更改UDP源端口

![图片](assets/441bfa4d4eb0.png)

但是当M:N部署后, Expert侧会有大量的QP, 而DPA只有那么16个Core和每Core 16个线程, 算力又不够咯. 势必又要在这里引入DCT来解决QPScale的问题, 而eRDMA给用户呈现的128K个QP可以多路径全开,主要是底层完全实现了stateless的subflow, 还有一个根本性的问题是当出现拥塞后, DPA如何做到降速或者是路径切换这个两难的决策问题, 另外接收端如何做到ReOrder buffer free的实现? DPA上不得跨核通信么, 本质的问题是MLNX这群人对内存模型的理解存在很大的问题, 这几天正在写一篇文章, 过几天发出来.

![图片](assets/502308e309c2.png)

最后扯个淡... 一个月前就预测过NV的股价会到70~80的区间

![图片](assets/392a11895aef.png)

如今盘后的价格离进入80这个区间就差2块钱了...

![图片](assets/dfa9d8dd6639.png)

另外前段时间写[《谈谈三万亿的破绽(5): 以基金经理的视角》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493389&idx=1&sn=17fce7abc1e8f58658e26a1c37346508&scene=21#wechat_redirect)时, 还推测过段总的操作, 本质上也是预测价格会回落到75~95的区间, 如今已经实现:)

![图片](assets/c09779afd7d1.png)

参考资料

[1] 
MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism: *https://www.arxiv.org/pdf/2504.02263*
[2] 
MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs: *https://arxiv.org/html/2402.15627v1*