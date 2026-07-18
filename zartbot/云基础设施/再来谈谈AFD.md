# 再来谈谈AFD

> 作者: zartbot  
> 日期: 2025年8月5日 12:13  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494603&idx=1&sn=30491396656a3658ee2947e31fd2bcf1&chksm=f995fb09cee2721fd08e6f6984a8d9627a230404415cbc33b965397d153ea671965b8f17279f#rd

---

### TL;DR

昨天朱老师在知乎有一个文章把AFD和DeepEP对比了一下, 大概得出了一个BatchSize=64 Context 8K时40%性能收益分析. 当然今天发现文章已经删了... 其实主要是有些 apple vs orange的对比在里面.. 有几个问题交织在一起, Step-3的 Expert数量和Active Expert数量为48/3. DS是256/8. 其实部署的解空间还是有很多不同的, 例如最大的部署集规模等... 另外还有一个文章做了一些量化分析..

《Step3——AE分离的量化解析》[1]

说说我粗浅认知的两方观点:

**正方**: 其实简单点来看有一个直觉的判断:  在Attn和SeqLen相关的有状态的算子, 而FFN是无状态的, 因此seqlen长到有显著差异时, 两者分离是有价值的. 特别是在大家谈到的Agent的场景下. 另一方面是设计内存占用的情况. AFD可以把大部分的FFN的参数Offload出去, 使得Attn Instance有更多的内存来存KVCache也是一个优势...

**反方**: E2E的视角来看, 随着Seq越来越大, BatchSize是也在减小的, 那么FFN本身计算量也在降低, 算子本身耗时也相对在减少(即便MFU打不高). 此时应该有一个极限, 如下图所示, 当ATTN耗时加大后, 实际上MLP在整个timeline的时间占比是下降的, 按照DS的timeline来看, 在4K/8K实际上的性能提升上界大概只有20%左右, 我估计实际的性能在8%~15%左右

![图片](assets/fe236ad19fd4.png)

但实际的情况在更长的SeqLen下, 例如Typical 32K/64K SeqLen for Agent时, 对于纯EP情况下, 特别是大EP部署情况下(EP144/EP320), 从显存容量上看每卡专家参数所占用的显存占比并不是很大, 特别是显存越大的卡, 例如H200对AFD的需求会更小. 从计算延迟上看, Attn耗时占比更高, 例如极限的性能收益在8%~10%左右, 分离的价值并不大.

我觉得从两方的观点来看, AFD的甜点区域可能会在8K~32K, 反而真的Agent用的64K左右的区域面对大EP部署可能还没啥收益.

最后从一些集群建设的视角来看, 如果是像阶跃这样的单一集群, 有收益是挺好的. 但是对于云而言, 做一些事情整体异构的情况下, 如果性能收益在10%左右时, 我个人的观点是尽量简单部署, 凡是能够scale的, 尽量简单一点不要搞一些专用的东西.

### 1. 谈谈AFD论文

其实我个人觉得从论文的写作的角度, 这段话攻击性有点强了, “Some are overly emphasizing”实质上就是说DeepSeek嘛...大家和气生财把蛋糕做大不行么? 非得立个“纯EP有同样的问题”的靶子干嘛呢?

![图片](assets/da1b9077f2b1.png)

第一条再说MLA为了省KVCache导致Computation cost太高, 第二条在说DeepSeek MoE你们搞大EP干嘛, 太稀了不能符合现在硬件的需求,导致MFU打的不高..

其实有没有Overly emphasizing, 我觉得是值得商榷的. MLA本身的优势, 苏老师有很多分析了, 然后K2降低了一半的Head其实也挺好的. 关键是在第二条上. 我还是认为可能大家对DeepSeek说的Expert Specialization有误解...

[《详细谈谈DeepSeek MoE相关的技术发展》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493157&idx=1&sn=51c0e27a347dd3fe1ed868d87f667897&scene=21#wechat_redirect)

有些话在前面一篇讲了, 特别是DeepSeek说的不想让大量的token冲击到专家引起它学到的知识太杂, 这是从算法层面上需要大EP的实质性问题,  而这个问题本身通过范畴论的一些视角也可以解释的很清楚. 因此至少第二点是肯定没有过度强调大EP大稀疏的.

其实这一点在《Step3: Cost-Effective Multimodal Intelligence》[2]他们也承认了这一点:

![图片](assets/e888b63d3e15.png)

其实这就是模型和Infra协同设计的另一个问题, 过分的强调Infra和硬件的效率,反而导致模型掉点. 然后最终的E2E推理性能是否真的有显著的提升?

其实我一直想了解Step-3选择专家3/48的原因是什么, 从Infra上来看, 我能明白Top-K降低会降低通信量, 是好事. 然后更大的Expert MFU也会更好..但是不明白在co-design的时候, 算法团队的反馈是什么?

另一方面虽然一直在强调Agent等场景需要更长的seq, 但似乎Known Issues的第二点来看为了多模态的场景在coding的场景似乎有些踏空了...

![图片](assets/3d1d86e804cb.png)

### 2. AFD的收益

实际上我在今年2月的时候就写过一篇

[《谈谈国产算力支持大模型和MoE/RL算法协同演进方向》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493109&idx=1&sn=7d95a97f69bf20d664042615653a8deb&scene=21#wechat_redirect)

那时候也在谈一些国产算力做AFD的事情, 同时在做ShallowSim仿真的时候也意识到小BatchSize打不满的问题(还引用了申老师的一些测试数据), 因此也想到过类似于AFD的事情, 如下图所示

![图片](assets/e9a7b6d0f1c5.png)

同样还有一个想法是, 既然要做Disaggregation, 那么就需要降低机间的Alltoall通信量, 但是我完全能够理解DeepSeek Expert Specialization需要Fingrain Routed Expert的需求, 因此期望通过一个hierarchy gating的做法既保证足够的Fingrain, 又能够同时避免通信的开销. 实际上我一直认为这是一个可行的方向...

后来经过一些收益评估, 在一些常用的SeqLen下(那个时候Agent还没有那么火), 收益并不明显, 同时那时候带141GB显存的H20也挺多的了, 随着显存容量越大, AE的收益其实是相对减弱的...

但是在一些Agent的场景下, LongContext本来带来的一些开销也挺大的, 从整个算法-系统-硬件的架构上, Agent侧的Context Engineering来节省Context是一个更好的选择. 另一方面实质的问题是 Attn本身的算法问题, 那么算法上是否有别的方法么?

DeepSeek其实已经有了NSA的答案了, 那么在NSA的上下文是否也有这样的问题么? 而且从Agent Context Engineering的角度, 我特别喜欢NSA.

### 3. 谈谈AFD的部署

当我们构建一个异构集群的时候. 节点如何放置的问题. 其实对于我个人而言, 普遍的场景下没有30%~50%的收益很难去做这样的集群部署优化的. 对于云和线下数据中心有本质的区别, 我们不可能为了某一些垂直的场景去做一些特殊的事情.

另一方面, 论文中的A和F实例是同Rack部署的, 并且使用了PFC-Only的部署. 然后朱老师有一个回复

![图片](assets/f5e5f9672f09.png)

事实上, 对于云而言更多的是需要自服务的能力, 我们不可能为了各种workload去调整CC, 希望用户开箱即用, 开机即打满而无需感知拓扑, 是否要同Rack, 跨Pod都无所谓.. 这是作为Infra网络团队需要追求的.. 这不朱老师也会因为时间紧来规避这样的复杂性的问题... 正如UEC对DCQCN的吐槽

![图片](assets/b10dcd88f80f.png)

当然我以前对Lossless PFC/DCQCN确实有很多负面的评价, 不光我说, 看看一段回复《AI fabric is a bus or a network？》[3]

![图片](assets/647ed0374283.png)

其实工业界大家都清楚的事情, 现在我可能会比较客气的来说. 单一租户的用PFC可能也挺好的, DCQCN有个网络团队调一下,除了复杂一点要花时间也没什么大不了的. 其实我并不认识朱老师, 并且朱老师一些观点我还挺认同的, 但是这样有攻击力的说法...我只是笑一笑, 其实对于云来说, 例如阿里云本身也不能算Lossless, 而是算Semi Lossy的处理方式...

正如我在[《RDMA》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3398249338911260673#wechat_redirect)整个专栏谈到的, 技术路径上的问题云和单一数据中心有很大的区别.

如果异构的集群部署, 在同一个RACK下, A和F的配比如何定, 特别是异构的卡的情况下. 然后又如何保证训练和推理公用的情况下, 训练的效率最高? 当然有一些MultiRail的做法如何做M:N ? 其实这都是这篇论文没有回答的.

而我一直以来的观点就是正面应对丢包和Hash冲突这些问题, 以及拥塞控制的问题

![图片](assets/1db4ecc15fea.png)

其实使用Direct Data Placement实现多路径并保持RC Verbs能力全球第一家是我们, Nvidia DDP做AR都是借鉴的我们, 而且CX7的SEND/RECV的DDP还不支持... 土不土?.  其实Mellanox是一个ASIC公司, 因为DPU或者SuperNIC被迫向网络处理器架构转型的过程中跑偏了, 毕竟他们也就买了一个EzChip和Tilera, 好多经典网络处理器的东西都不知道...... 相信我们这些来自BRCM和Cisco的老司机的判断, 我们是有十多年的网络处理器芯片和系统设计能力的.

另一个问题是无法避免的就是incast的问题, 本身由于一些问题带来的队列延迟有多少?

![图片](assets/cb043ecd5926.png)

RDMA在云上的需求就是下面这个图

![图片](assets/00f00d9cea89.png)

正常的部署, 基本上都是同一种卡接近万卡的集群, 然后不同卡异构的时候, 基本上都要跨越Spine交换机, 通常过Spine是否要做一些收敛比呢? 这一系列的实际部署的困难导致A和F通常可能要有5跳网络, RTT延迟接近10us, incast和PFC还会带来显著的延迟, 这样的情况下AFD业务收益还有多少呢?

然后A和F分离以后, 实际上按照总卡数摊销后的平均每卡吞吐的收益是多少, 异构的收益可能还有一些问题的?

### 4. 小结

最后还是客观的来讲, AFD本身是一个很不错的工作, 但是真没必要去喷别人过度强调xxx, 也不至于说“纯EP有同样的问题”, 实际上AFD本身的收益也没有那么大, 当然针对无状态算子/内存的拉远池化本身是一个不错的想法, 在学术上挺好的一篇文章. 但是我上面也提到了一些工程落地的难题.

如果对于一个特性或者部署方式性能收益并不大, 并且需要垂直的改造. 我的观点是:   凡是这种case下能够Scale的就尽量简单一点...没有必要去做太多的垂直优化.

另外一点针对Attention和FFN之间在长序列的巨大差异下,难道不该算法协同去想一些新的算法么, 例如Vision中一些Linear Attn, 或者像DS那样做NSA,都是更加直击问题本身的解法. 就像前面那些网络的问题一样, 也是同样需要直击问题本身... 

参考资料

[1] 
Step3——AE分离的量化解析: *https://zhuanlan.zhihu.com/p/1935657127348793545*
[2] 
Step3: Cost-Effective Multimodal Intelligence: *https://www.stepfun.com/research/zh/step3*
[3] 
AI fabric is a bus or a network？: *https://zhuanlan.zhihu.com/p/708602042*