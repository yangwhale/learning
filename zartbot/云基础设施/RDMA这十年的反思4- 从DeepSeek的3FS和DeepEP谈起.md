# RDMA这十年的反思4: 从DeepSeek的3FS和DeepEP谈起

> 作者: zartbot  
> 日期: 2025年3月7日 00:23  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493380&idx=1&sn=7c49a6cb30cc7b5687d249455e45c056&chksm=f995f7c6cee27ed03b34cbd1cf1c7db3c89141a704e5caaf6d528cb01d02ab680aa4f58efabe#rd

---

`本文仅代表个人观点, 与作者任职的机构无关.`

上周DeepSeek发布了一系列开源项目, 和RDMA相关的有DeepEP和3FS. 关于DeepEP的详细分析可以参考:

[《分析一下EP并行和DeepSeek开源的DeepEP代码》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493292&idx=1&sn=7af7db0f3d78f0fb52dc847934c7800e&scene=21#wechat_redirect)

对于3FS网络通信相关的分析, 可以参考蚂蚁存储团队同学进行的分析:

[《DeepSeek 3FS解读与源码分析（2）：网络通信模块分析》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493353&idx=1&sn=80ad7afcd4360fd389d841ad05222238&scene=21#wechat_redirect)

3FS发布几个小时内,  一行代码都没有修改, 我们就在阿里云上通过eRDMA搭建完成,并开始进行一系列压测

[《基于eRDMA实测DeepSeek开源的3FS》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493320&idx=1&sn=70f97436be6617d75940d8bee66cb1df&scene=21#wechat_redirect)

### 1. RDMA RC Verbs兼容的重要性

而今天看到一个3FS Github上的issue[1], **抱歉EFA不支持RDMA RC Verbs, 安装失败.**

![图片](assets/cd1283253410.png)

这就是渣B一直在强调的, 维持RDMA RC Verbs的生态兼容的重要性. 以前只有NCCL的时候, AWS在EFA上弄个插件还行, 而现在随着DeepEP/3FS, 还有Redis, Spark以及很多分布式数据库, 这些生态都在支持RDMA的时候, 怎么办呢? 很多东西并不是美国人做的就好, EFA SRD和UEC Libfabric估计要支持这些生态, 需要很大的人力投入和逐个版本的适配.

同理, 您如果想在Azure或者Google上试一下3FS, 要么去买支持IB的HPC实例, 要么就得去买GPU实例. 即便是使用IB技术构建带有SSD本地盘的机型, 成本也非常高. 至于Google的Falcon至今还没有上线, GPU实例开始买CX7,然后最近几天爆出在以色列建团队自己做网卡...

阿里云eRDMA成为全球唯一一个在所有地域所有可用区上通用计算实例(从第八代开始)完全免费支持RDMA能力的云服务提供商, 并保持RDMA RC Verbs兼容, 线下Nvidia/Mellanox的应用程序可以轻松迁移到云上. 再划个重点,eRDMA完全免费节省的RDMA网络成本大家可以自己核算一下.

即便是Nvidia的RDMA, 在以太网上开启AdaptiveRouting后, SEND/RECV这些还是无法在多个路径上转发的, 然后又是必须基于Lossless无损的以太网... 

[《谈谈英伟达的Spectrum-X以太网RDMA方案》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489018&idx=1&sn=25b3df2a17d49681edc0e621049b058f&scene=21#wechat_redirect)

eRDMA可以在5%的丢包率下依旧维持90%的Goodput, 而Lossless以太网在千分之一的丢包率会怎么样? 更不要说和TCP混跑以及支持各种虚拟化的能力和热迁移能力了, 丢个图给内行的看看先进性.

![图片](assets/bb14fd5adb9f.png)

几个指标:

集合通信能够保证95%以上的Fabric利用率

丢包率5%的时候仍然能够保证90%的Goodput

无需任何交换机的高级特性, 网卡实现多路径和拥塞控制

超大规模(128K QPs)并支持所有QP开启多路径转发能力.

兼容RDMA RC Verbs, 线下RDMA应用无需修改代码即可直接运行.

Incast 128打1这样的场景, 每个QP之间的带宽差额最大100Kbps.

CIPU 2.0支持400Gbps,未来还会更高:)

想起钱老那句话:“中国人怎么不行啊？外国人能搞的，难道中国人不能搞？” 不光能搞, 还比他们做的好. 我一直讲在DPU这个领域, 我们超越Nvidia是本分, 也希望其它领域大家都能更佳自信一点.

### 2. 从DeepEP谈RDMA多路径相关的问题

DeepEP在实现上要求Prefill和训练的Normal Kernel不能开启Adaptive Routing, 而在Decoding的集群上应对高吞吐必须要开启Adaptive Routing能力, 如果要在RoCE上同时在一个集群内承载Prefill和Decoding集群, 交换机要怎么配置? 拓扑该如何构建?

稍微科普一下这个问题, 如果不支持多路径转发会在交换网中产生Hash冲突,

![图片](assets/129ecf9adc95.png)

正是因为这个问题, 在Decoding高吞吐的情况下必须要开启Adaptive Routing的能力. 但是当数据包经历不同路径转发后会导致乱序...这个难题比较难解决, AWS SRD直接选择了放弃... eRDMA采用了2002年iWARP的Direct-Data-Placement技术实现了乱序提交和保序完成的能力

![图片](assets/afb2ee5ba151.png)

而时至今日, Nvidia(Mellanox)由于RoCE协议定义的缺陷, 仅能在WRITE操作上实现类似的能力, SEND/RECV还无法支持.

另一方面, DCQCN的ECN信号也因为流量混跑和微突发导致错误的信号, 甚至因此产生PFC阻塞和死锁的现象. 在超大规模部署中,靠调整交换机ECN门限几乎是不可能实现的, 特别是在云上这种多个租户共享网络的时候,  而标准RoCE导致的丢包Go-Back-N重传又极大的影响了传输效率.

![图片](assets/a8e5c0ace9a6.png)

假设这是一个支持EP并行的Decoding集群, 任何一个长尾都会导致GPU空等, 最终使得Decoding的吞吐率急剧下降. eRDMA在拥塞控制上进行了特殊的设计, 基于RTT的细粒度滑动窗口控制协议,并支持SACK等选择重传技术. 这样就避免了重传效率,同时也压低了交换机Buffer的使用率.

![图片](assets/dae69e8f0a03.png)

整个eRDMA方案无论是拥塞控制还是多路径转发都不需要借助交换机的辅助(ECN/PFC/INT/PacketSpray/VOQ等都不需要), 因此降低了大量的运维负担,完全做到了用户自服务开启的能力.

RDMA over Ethernet这十年,有太多的反思.回顾了整个RDMA的发展历程， 其中你就会发现RoCE在这上面走了太多的弯路，协议定义上从RoCEv1到RoCEv2一系列的错误，再到PFC的问题困扰了工业界很多年，但反过来看同时期的usNIC已经有解法。虽然Lossy RDMA出来了几年，看上去走上正路了，而后又因为Packet Spray这些AI网络需要的特性又回到Lossless上。

![图片](assets/43fcaceb7534.png)

另一条路径无论是AWS SRD和Google Falcon的演进则看上去非常干净，少了很多折腾, 但是一个不支持RC语义, 一个在商用落地上延误了很多年,并且多路径算法上还有很多问题.

就这样吧,大家都自信一点, 一起加油, 一起进步~

参考资料

[1] 
Attempting to deploy on AWS #106: *https://github.com/deepseek-ai/3FS/issues/106*