# 谈谈OpenAI 发布的MRC

> 作者: zartbot  
> 日期: 2026年5月7日 05:10  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498265&idx=1&sn=c3e7359f2974c173378bad40925e3cea&chksm=f995eadbcee263cdbf96f90cc2b7864c44e34c56db9a8d8cf614fe96484cdc91b554700dac75#rd

---

### TL;DR

其实几个月前就知道OpenAI和Nvidia / AMD / BRCM / Intel在搞一个新的RDMA协议, 昨天见到了OCP**的白皮书《Multipath Reliable Connection (MRC) Specification》[1], 以及OpenAI官网的一个发布blog 《Supercomputer networking to accelerate large scale AI training》[2]. 本来这个问题我们已经完美解决了, 阿里云所有Region所有AZ从第八代计算实例开始就完全支持, 多路径都上线2年多了.... 但看到这个协议还是有那么多缺陷, 起初真是懒得多说话了, 但耐不住一堆人来问, 干脆又得罪人来锐评一下.

总体来说, 这个Multipath Reliable Connection (MRC**) 的名字定义是准确的, 要在RC Verbs生态兼容的情况下做好多路径的能力, 至少这件事情终于想清楚了,  没有像 AWS 那样搞什么 SRD 或者 UEC** 去搞Libfabric的接口, 这正是我几年前就一直在说支持兼容RC Verbs接口的多路径 Lossy RDMA才是正路, 苦口婆心没人听...还记得某些人前几年张嘴就来“ OpenAI RoCE用的好好的, 哪来的问题”, 既然没问题, 那搞新的协议干嘛?

当然对于MRC还是很不满意的, 几年前我们CIPU eRDMA就完全实现的东西, 甚至追溯到20年前iWARP DDP就可以解决所有语义的多路径能力, 时至今日**MRC仅支持 WRITE 和 WRITE_IMM** , 说实话我真的是有些无语了, 看看里面的厂家 OAI / Nvidia / AMD / Microsoft / Broadcom / Intel , 不由得再次说出那句话: 世界是一个巨大的草台班子.

## 1. 从RDMA现代化谈起

对于不熟悉这块知识的读者, 可以看以下以前写的一篇比较好懂的文章[《“漫”谈RDMA现代化》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497377&idx=1&sn=a8645893c61e582d9cb8613b70e65506&chksm=f995e663cee26f75ab3cbfef2b940ce0db5890106367ab4f6869983f9a5a383004078380d7fb&scene=21&cur_album_id=3398249338911260673&search_click_id=#wechat_redirect) 来了解 ScaleOut RDMA, 特别是RoCE遇到了哪些问题. RDMA现代化主要是解决以下两个难题:

![图片](assets/5c196a78aaf3.png)

这也是 MRC 协议覆盖的内容, 也是我一直讲的要支持 RC Verbs 生态兼容, 支持 Lossy RDMA, 支持多路径... 下面这个决策图用了好多遍了, 终于看到这几家想清楚了...

![图片](assets/c6b6964ff772.png)

但是很遗憾的是, 他们并没有完全搞明白该怎么做, 例如还需要依赖交换机**, 语义上也仅支持 WRITE 和 WRITE_IMM, 那么SEND/RECV, READ, ATOMIC呢? 又不是做不了, 还不是能力不行...

## 2. 详细分析MRC

### 2.1 Overview

MRC Spec里写的挺清楚的针对AI/ML 作负载在**标准尽力而为以太网**上提供可靠的, 高有效吞吐量的连接. MRC 扩展了 RC 传输协议, 具备显式的多路径操作, 路径健康监控和故障恢复能力, 从而使端点能够在拥塞, 链路故障和其他网络损伤的情况下维持有效的吞吐量. 在更大的系统栈中, MRC 位于现有协议层与 AI/ML 框架或运行时库之间, 暴露的 API 允许 NIC, 加速器和主机软件利用多个网络路径, 同时在需要时**保留有序**的, 可靠的交付语义.

MRC 解决了诸如网络容量利用率不足, 因**瞬时网络问题**导致的作业失败或停滞, 以及**端点行为与特定网络拓扑**的紧密耦合等问题.

渣注
首先标准尽力而为以太网的定义是精准的, 也就是说不要PFC, 不要Lossless, “是男人就要硬刚Lossy”. 然后要保留有序是指的某些场合下尾了兼容性问题, 需要回退到支持Strict Order的场景, 例如DeepEP要构建happen-before语义, 需要 WRITE 后保序执行ATOMIC, 但是MRC的做法是: “那就回退到原来的标准RC语义”, 这样一来不就出问题了? hash冲突会再次困扰EP流量...

然后瞬时网络问题是指训练的时候遇到的光模块实效/交换机故障等因素, 传统的单路径会在这种情况下导致超时最后中断训练任务. 而多路径可以规避网络的失效路径. **其实这一点我们在24年初就使用CIPU eRDMA验证过,  TOR / Spine / Leaf / 端口随便怎么乱搞中断, 训练只有100ms内的抖动, 很快就能恢复, 训练被各种网络故障折腾了好多次从来没断过.**

最后端点行为与特定网络拓扑的紧密耦合指的是, 过去几年为了解决单路径Hash冲突的问题, 被迫通过多轨道/多平面的方式加大TOR Switch Radix, 通过特定的网络拓扑来规避问题. 但是实质性的问题并没有很好的解决.

简单来说做了以下几方面改进, MRC 连接实现了以下核心能力:

**多路径传输 (Multipath Transmission)**: 使用 ECMP 或源路由, 将请求和可靠性控制数据包 (SACKs) 分布到多个网络路径上, 并支持 ECN 标记.

**可靠性控制 (Reliability Control)**: 使用选择性确认 (SACKs) 和否定性确认 (NACKs) 来确保及时的数据包交付并传达拥塞状态.

**WriteIMM 限制 (WriteIMM Limiting)**: 强制执行用户定义的每个连接中在途的带立即数写操作 (Write-with-Immediate) 的最大数量.

**拥塞控制 (Congestion Control)**: 实现 NSCC, 一个在 UltraEthernet 传输规范 [UESPEC] 中指定的基于发送端的拥塞控制算法.

**选择性重传 (Selective Retransmission)**: 允许基于 SACK/NACK 反馈重传特定的数据包.

### 2.2 如何支持多路径

从转发路径上来看, 包括两种实施方案:

基于熵值的请求包轮换, 和Google PLB类似利用IPv6的额外的flowlabel字段作为路径熵影响交换机ECMP hash.

或者使用SRv6这些source-routing的技术, 让交换机跟着uSID寻址转发.

渣注
嵌入Segment的方式我在2021年做netDAM就实现过, 甚至可以通过Segment作为函数调用的方式实现“进程+管道”的处理方式.. 这没啥特别的. 后面针对Internet也实现过Segment Routing over UDP在全球范围内利用Internet资源实现200ms几乎零丢包的可靠传输. 没啥特别的.

但是后来在阿里的时候, 我给自己了一个新的挑战: 如何不用交换机感知和不利用交换机任何高级特性(INP/ECN/packet spray / packet trim)来实现多路径.  很简单, 本来CSP就是一个多租户网络, 有VXLAN这样的Overlay header, 直接改underlay源 UDP端口就可以作为entropy即可...

说实话, 做一个协议, 特别是拥塞控制类的协议, 尽量少的依赖外部信息, 做到自闭环的控制才是一个有品味的人需要做到的...

### 2.3 为什么语义上MRC仅支持 WRITE/WRITE_IMM

语义层面, MRC在标准的RoCE基础上增加了一个OPcode. 然后通过在每个消息中携带RETH Header来实现Direct Data Placement(DDP)能力.

![图片](assets/d27cf8998c89.png)

但是很遗憾, 它只支持WRITE, WRITE with Imm... SEND/RECV, READ, ATOMIC**都不支持...

渣注
其实很简单的一个问题, RETH里面只包含了绝对的内存地址, 当然无法处理SEND/RECV这些操作了. 但是这个问题在20年前就解决干净了的, 在iWARP的DDP Header中定义了 Message Seq Number(MSN)和Message Offset, 这样就很容易通过相对的Offset去做写入地址处理了...

另一方面是memory ordering的处理, 这也是导致MRC不支持ATOMIC的原因. 通过多路径转发后一定会出现乱序的情况, ATOMIC和它前后的WRITE如何处理保证顺序? 抱歉做不了, 也就是说在训练场景中, 例如DeepEPv2/NCCL-gin这些依赖ATOMIC的处理又会有问题要回退到单路径的模式...

那么采用WRITE_IMM可行么? 也不行, 抱歉它硬件实现上有一个max_wimm_inflight的约束...另外WRITE_IMM对GPU做通信-计算Overlap也不好, CQE的数据结构让GPU消费挺消耗资源的...

### 2.3 拥塞控制

主要是借鉴AMD/BRCM在UEC提的那一套:

增加独立的SACK能力,  SACK 传达数据包接收状态以及为传输和拥塞控制 (CC) 算法提供的网络和响应端拥塞信号.

基于PSN的Window based CC,  响应端可以在运行时调整最大在途数据包的范围.

毕竟Intel在里面, AMD也有以前在Intel搞NDP的, 还有BRCM的也是喜欢NDP, 所以Packet Trim也被包含进来了

还是采用Active probe的机制, 并且可以根据Entropy Probe某个路径, 最后可达性信息还可以通过Port Status mask传递.

渣注
好的地方是终于开始支持Lossy RDMA, 算是走向正路了, 然后拥塞控制也从Rate Based换到了Recv-Driven的CC window. 几年前我就一直反复在说AI的网络需要抛弃PFC+DCQCN... 很高兴看到这几家都达成共识了...

但是实现上又有很多问题没考虑清楚, SACK受硬件设计约束, 直接影响到了整个路径上inflight packet window. 那么也就是说这个协议还是解决不了超大规模集群的跨数据中心ScaleAcross的传输...

然后拥塞控制算法上还是存在缺陷.... RTT+ECN的控制其实挺复杂的, 直接用Swift这样的4个时间戳带内的信号控制 window 不就成了? 非要自己造轮子干嘛?

## 3. 结论

大概就这样吧, 也数落了MRC不少的问题了... 本来我都懒得谈这事了, 几年前就干净解决的问题, 几大厂商如今还在这里折腾, 折腾废了UEC, 又开一个新的MRC坑...然后还是解决不干净, 就是一个很脏的工程补丁... 极度无语...

很大程度上他们都被Nvidia CX系列的微架构束缚着. 我还是老话重谈, 下面几个指标是几年前提出的, 做到了再来说话...几个指标如下:

集合通信能够保证95%以上的Fabric利用率 ✅`MRC应该能实现`

丢包率5%的时候仍然能够保证90%的Goodput  ❓`没有测试结果, 理论上MRC可以做,但是受到Inflight限制`

无需任何交换机的高级特性, 网卡实现多路径和拥塞控制 ❌`MRC需要依赖交换机, 多路径可能要SRv6的能力, 拥塞控制要ECN和NDP`

超大规模(128K QPs)并支持所有QP开启多路径转发能力. ❓`这个和具体网卡微架构实现相关`

兼容RDMA RC Verbs, 线下RDMA应用无需修改代码即可直接运行. ❌`坚持RC兼容是MRC作出的正确选择, 但是也是只支持部分语义算个怎么回事?`

Incast 128打1这样的场景, 每个QP之间的带宽差额最大100Kbps.❓`基于Window Based CC应该可以, 但是实现上还需测试`

感兴趣, 自己读读[《RDMA》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3398249338911260673#wechat_redirect)这个专题, 了解更多的内容..

总体来说, 我还是要对MRC能够迈出在RC Verbs兼容的前提下, 支持多路径转发同时支持Lossy RDMA这件事点赞的. 三年前我们的选择成了工业界的共识, 但愿以后别再又折腾些Lossless的事情了... 然后...剧透一个小事情, 留给NV的时间不多了......

参考资料

[1] 
Multipath Reliable Connection (MRC) Specification: *https://www.opencompute.org/documents/ocp-mrc-1-0-pdf*
[2] 
Supercomputer networking to accelerate large scale AI training: *https://openai.com/index/mrc-supercomputer-networking/*