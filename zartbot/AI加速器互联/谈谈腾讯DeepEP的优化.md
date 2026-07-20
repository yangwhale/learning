# 谈谈腾讯DeepEP的优化

> 作者: zartbot  
> 日期: 2025年4月24日 15:37  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494024&idx=1&sn=bf8fe65faa9957eb94638ab8ab20e6b8&chksm=f995f94acee2705c44ec4af7aebc4187c7afadd92ca4d2f99adf3ff00422d60d45fc275b3754#rd

---

渣B很多时候的一些文章负面评价可能会比较多, 主要原因就是有些PR性质的东西确实容易把工业界带偏, 说点实话, 例如老黄天天吹ScaleUP, 还有隔壁一开始测个3.8GiB/s经过几番指正还不差不多只能到峰值带宽的一半... 当然渣B对于好的工作都是要好好欣赏和学习, 例如华为的UB, 字节的TILELINK/COMET等, 现在来谈谈鹅厂最近对DeepEP的优化. 物理链路上差不多有43.5GB了, 从鹅厂的文章《腾讯TRMT助力DeepSeek-MoE模型通信提速30%》[1]来看, 其实还有一些其它的优化, 内部的数据可以到62GB/s(46.5GB/s)差不多接近极限了, 没有完全跑到大概还是很微观上的两个MAC口的轻微不平衡以及Atomic带来的一些小的性能的下降. anyway对这样能够基本上接近物理极限的工作都是非常钦佩的.

![图片](assets/fbcc9d948798.png)

> **Support multi-QP for normal kernels #130** (Merged, LyricZhao merged 9 commits into `main` from `trmt/internode_multi_qp`)
>
> This PR is authored by **Tencent Network Platform Department**. Thanks for the contribution! Now normal kernels have a huge speedup:
>
> | Type | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
> |---|---|---|---|---|
> | Internode | 32 | 44 → 58 GB/s (RDMA) | 32 | 47 → 57 GB/s (RDMA) |
> | Internode | 64 | 46 → 51 GB/s (RDMA) | 64 | 45 → 50 GB/s (RDMA) |
>
> Through in-depth optimization, the following enhancements have been implemented in the internode normal kernels.
>
> - Replacing IBRC with IBGDA
> - Utilizing distinct QPs (Queue Pairs) per channel for parallel data transmission
>
> These improvements not only enhance the robustness of the internode normal kernels in scenarios involving dual-port NICs and RoCE networks but also further elevate communication performance.
>
> NOTES: the bandwidth in the table is algorithm bandwidth, for physical bandwidth, e.g. $58 \times 3/4 = 43.5$ .

其实在2月底刚发布DeepEP的时候, 我就做了一些分析

[《分析一下EP并行和DeepSeek开源的DeepEP代码》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493292&idx=1&sn=7af7db0f3d78f0fb52dc847934c7800e&scene=21#wechat_redirect)

特别是对RoCE支持DeepEP的问题详细谈了一下

![图片](assets/4de978ee0896.png)

当时很多人还不以为然, 然后就是3月中的时候有一个Github的issue:Benchmark test over RoCE network #82[2]

![图片](assets/9429dce2f98f.png)

然后解决了双端口(2x200GE)多QP转发后依旧带宽上不去.

来谈谈鹅厂的优化吧, 最重要的工作就是把IBRC换成了IBGDA, 当然可以说IBRC有一些Kernel launch的问题,在IB上也提升了30%的性能.

![图片](assets/7907e4ab63e7.png)

另一方面, 鹅厂的结果也很好玩, 为什么RoCE的性能会翻倍呢?

![图片](assets/4fe6be6bf799.png)

看了鹅公众号的文章, 整个方案如下:

![图片](assets/c05247a4f2de.png)

TRMT-SHMEM应该是一个分布式共享内存池的实现, TRMT=Tencent Remote Memory Transport, 然后从一些支离破碎的信息, 似乎是想用一些办法来构建分布式一致性, 并且采用IBGDA我不确定是否和腾讯的ETH-X项目还有一点关系(盲猜, 猜对了几个鹅厂小伙伴别打我~)....

![图片](assets/7a4b2bab0c6d.png)

然后它内部考虑到网络的一些拥塞, 通过Hashlib会预先选择一些源端口(src port 预规划)的方式来避免. 但是我挺好奇的, 腾讯的星脉网络也是一个多轨道部署, 照理说Normal Kernel也是走类似于PXN的方式, RDMA流量应该不会跨越轨道吧? 当然,也存在一个问题, 每个网卡有两个物理网口, 是否就是为了通过src port来解决这个问题呢? 正如鹅厂文讲的:

依托TRMT-SHMEM模块构建的全互联通信架构，通过动态QP端口分配算法实现网络流量的智能散列。该技术突破源于团队在超大规模集群组网实践中积累的拓扑感知经验，通过UDP源端口动态规划技术，使`双端口网卡带宽利用率达到理论峰值`。

然后为什么性能能够提升100%呢? 我在以前的文章(例如下面这篇)也提过Nvidia单个QP只能通过XOR选择其中一个物理网卡发出去, 而我们可以做到单QP同时用两个物理口....

[《从3FS性能谈谈数据密集型应用上云的挑战和机会》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493664&idx=1&sn=6fcb9fcae1d07a244df64fde0b172407&scene=21#wechat_redirect)

针对Nvidia这网卡的缺陷, 鹅厂采用了多个QP发送, 但是当存在多个数据路径转发时, 很容易导致乱序到达从而产生一些Data Race的问题. 那么解决问题的方法就是采用多个QP传输... 然后就在这里引入了一个Atomic操作, 本质上是做一个memory fence.

其实嘛, 单QP在多个物理网卡上发送同时做到保序并且还不需要网卡上搞什么ReOrder buffer也是可以做的, NV自己做不了其实就是菜...

然后突然就想到了, 如果AWS SRD要来处理这样的case会如何呢? SRD当时宣称的最大优点就是out-of-order,当然对一些场景有用能够提高吞吐率,降低QP Connection的规模. 当它需要处理ATOMIC的时候, 会不会完全阻塞整个网卡的通信产生头阻(Head of Line Blocking)呢? 这就是一个协议设计不完备的地方....

正好今天有一个朋友还在问我NetDAM的事情, 把当年一个没发布出来的最终版本转给他了. 当时对于内存语义有这样一段描述

![图片](assets/e0fb181c6a9f.png)

即便我要享受Out-of-order的好处,让消息传输时满足交换律, 同时我在协议里还会故意留一个Transaction ID来针对一些strict order的业务处理. 另外针对ECN这个字段, 其实它本质上和CSIG是类似的, 因为它的定义是: ECN: ECN field only present in ACK message to indicate internal pending instruction FIFO queue depth.

然后多路径的转发上采用了Segment Routing的机制, 其实当年我不懂云计算的特点, 也在考虑一些非对称拓扑引入了Source Routing的能力, 和UB其实也是类似的. 但是最终大家也别猜了,eRDMA没有用到Segment Routing没有ECN没有INT没有交换机的PacketSpray也没有PFC..

![图片](assets/5ab86e2712be.png)

并且NetDAM针对Atomic也是支持的, 然后当时和蚂蚁的OB团队有一点合作, 通过Atomic给他们做了一个NetDAM-Seq用于分布式事务的定序发号.

[《NetDAM-Seq:一秒5.68亿次的存算一体全局唯一单调递增ID发生器》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487315&idx=1&sn=5e92765fed256094834eb3fb3e8653dd&scene=21#wechat_redirect)

**吐个槽, 你们可以去测试一下Mellanox的网卡Atomic性能如何...其实这里也可能成为未来的一个瓶颈的,  而NetDAM可以64B小包400G LineRate.**

所以你看呀, 设计一个协议并不是那么简单的去针对某个特殊场景, 做一个完备的协议其实要考虑很多东西.... 这不AWS SRD不管是在iBGDA上还是在Atomic上都是一个很不成功的产品, 虽然过去通过NCCL外挂插件解决了一些问题...试想客户在AWS上买的H100跑不了DeepEP是啥感受呢? 跑不了3FS又是啥感受呢? 生态很重要哈.

话说Google的Falcon终于在H4D一些超算实例上线了, 有空可以去玩玩~ 讲真Falcon的协议设计的挺好的, 就是在Intel IPU上实现出了很多问题...而AWS Nitro的架构未来要卷高带宽估计也会遇到不少的问题....

总结一下吧, 腾讯的DeepEP优化是一个非常好的工作,并且开源出来了. 对于解决Prefill和训练的性能问题还是很棒的, 但是下一个问题来了, 在Decoding阶段的LowLatency Kernel, 如何跨越轨道,如何在大规模EP并行时解决incast的问题, 挺好奇工业界在这个问题上如何能够做出一些创新的工作.

最后, 再吐个槽. 有时候一直在想, NV为啥要那么坚持的用高密度的铜? 当然高密度的光模块的可靠性确实是一个问题, 我记得一般DAC的MTBF大概接近500万小时, 光纤大概是它的1/10 ? , 而且受温度影响还挺大的. 另外就是为啥要用MidPlane呢.. 前几天又翻到一个老图,  看看Cisco设计的机框

![图片](assets/5df8bf55cc45.png)

Cisco有一个做法挺好的就是把网卡这些对外的I/O都挂在后面的扩展板上, 这也跟我以前推测Rubin机柜的想法是一致的

![图片](assets/757e761e5467.png)

其实做通信这些高密硬件的厂商基本上都会选择这个方式, 有点不明白NV在搞啥? 往最简单的想或许就是内部有两套班子, 往复杂点想或许真有一系列的约束和取舍...

参考资料

[1] 
腾讯TRMT助力DeepSeek-MoE模型通信提速30%: *https://mp.weixin.qq.com/s/tke8GmdzecqFLTyBA9CWdg*
[2] 
Benchmark test over RoCE network #82: *https://github.com/deepseek-ai/DeepEP/issues/82*