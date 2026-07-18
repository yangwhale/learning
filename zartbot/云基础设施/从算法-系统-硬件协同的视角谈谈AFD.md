# 从算法-系统-硬件协同的视角谈谈AFD

> 作者: zartbot  
> 日期: 2025年8月3日 12:39  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494588&idx=1&sn=8c5e7effb0894981925e76a59e734663&chksm=f995fb7ecee27268fa9eec67a56502c536883f47ddc472822cae19f173d843c1ace0325e291b#rd

---

### TL;DR

很多朋友来问, 还是写一些文字吧... `本文仅代表个人观点, 不代表作者所任职的机构`

大概看了如下材料:

《Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding》[1]

《Step-3推理系统：从PD分离到AF分离（AFD）[2]

《Llama-5 锐评 “Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding”》[3]

《关于 AI Infra 的一切 | 对谈阶跃星辰联创朱亦博》[4]

总体来看朱老师的一些观点是正确的, 春节期间其实就有一篇文章在讨论国产算力做AE分离的问题, 其实出发点是类似的, 国产卡有国产卡的约束, 国产替代也不是掀桌子全国产化...

[《谈谈国产算力支持大模型和MoE/RL算法协同演进方向》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493109&idx=1&sn=7d95a97f69bf20d664042615653a8deb&scene=21#wechat_redirect)

包括后面在做一些ShallowSim[5]推理仿真分析的时候, 也针对不同的硬件有一些分析, 其中也包括了一些国产卡, 也包括模型本身的一些参数以及在不同的Seq和batch-size, EP规模上的分析...

最后在成本和业务收益上以及系统的复杂性上考虑了一系列问题, 当然还有一些云服务提供商(CSP)和线下IDC机房本质上弹性多租在业务目标上的不同带来的取舍, 异构Disaggregation可能带来更大的复杂性...

### 1. 谈谈协同设计

我非常认同朱老师的说法是, 算法/系统/硬件需要co-design. 但是有些问题本身要辩证的去看. 对于Step-3实质上我比较认同知乎上洋哥的说法: `一言以蔽之：the step-3 paper is all about fitting model and software designs to hardware for performance，而且是各种各样因为export control催生的奇奇怪怪配置的hardware` 不光是因为出口管制的问题, 同时国产算力, 特别是一些NPU本身架构的问题.

实际上, 一个好的协同设计确实挺难的, 既要对算法本身有足够深刻的认知, 又要在Infra上提供足够丰富的优化手段, 同时为了避免fit for hw, 还要对硬件上有足够的掌控能力, 其实这一块是国内外很多infra同学欠缺的, 毕竟对于各种处理器微架构的Tradeoff了解相对较少, 对于硬件本身的约束了解也较少. 因此很多时候从infra提出的需求也是相对片面的. 而Infra的同学通常有自己计算/网络/存储的相对独立的背景和视角, 在此之上协同也有很多问题... 例如计党和网党在ScaleUP上大量的争议...

我个人的一个观点是: AFD可能是`针对国产算力现状下`的一个不错的工程实践.

### 2. EP有同样的问题? 从算法谈谈

听了朱老师的访谈, 大致理解了一个思路. 从他的认知来看MoE是一个降低成本的视角, 本身也没啥问题, 确实很长一段时间从算法的同学来看MoE相对Dense掉点的问题是实实在在存在的.  但是就此推测DeepSeek的一些观点我粗浅的认为是值得商榷的.

从朱老师的视角来看(我的理解, 不对请指正), 大EP及大的top-k会在ScaleOut通信上带来更大的压力, 同时较小的Expert又导致GPU本身的MFU不够高, 那么从系统的视角做一个Trade-off, 在维持相近的激活参数的前提下, 每个专家大一些,总专家数少一些(Step-3 NumOfExpert=48), 选择的专家也少一些(Selected Expert per Token =3), 这样FFN计算的MFU也会显著提高, 从总体来看是有很明确的性能收益的.

这样网络上all-to-all的一些复杂性的问题在小规模部署时,特别是如朱老师提到的F数量不多, 特别是机器数量小于MoE topK时退化到All-Gather和Reduce-Scatter. 然后在网络上也可以很容易的处理. 例如和曦智一起做的dOCS配合等..

我们从另外一个视角来看, 春节后几天详细分析过DeepSeek整个MoE的研发过程

[《详细谈谈DeepSeek MoE相关的技术发展》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493157&idx=1&sn=51c0e27a347dd3fe1ed868d87f667897&scene=21#wechat_redirect)

相对于传统的MoE它有一个很清晰的脉络:Towards Ultimate Expert Specialization. 传统的MoE工作通常专家数较少, 在训练过程中, 专家的专业度容易被过量的token冲击, 形象的理解就是大量的知识使得专家学到的内容很杂, 导致它承载的信息密度受损, 因此专家的专业度(Expert Specialization)出现问题, 基于这个视角提出了细粒度的专家(Fine-Grained Expert Segmentation)以及通过共享专家(Shared Expert Isolation)来吸收一些一些共同的知识, 降低其它专家的参数冗余.

这是DeepSeek Fine-Grained EP的初衷. 然后在训练效率上迭代了2轮AuxLoss, 最后发展到AuxLoss Free的解法, 然后Kimi K2再进一步把EPLB做细致后进一步避免了DSv3的Group Gating.

当然确实是大EP的情况下, 配合更高的TopK对网络带来了很大的冲击, 也导致了不少的问题, 特别是很多在RoCE上运行的系统没有像DS那样用IB-AR. 因此我当时的想法也是说是否能够通过2层Gating的方式, 在机间尽量降低Dispatch的通信量, 然后机内再Dispatch一次维持Expert Specialization. 其实在这里考虑到机内的MFU效率, 换成Dense也就类似于AFD了.

![图片](assets/19a259393f4e.png)

但从算法上的视角来看, 我是认同DeepSeek的Expert Specialization的观点的, 一方面是前面很多文章都在阐述的一个内容, 大模型的预训练过程实际上是在构建一个Presheaf,但是受制于算法和算力的协同影响下, 它并不能够相对精确的刻画GroundTruth, 或者更进一步某些Morphism为True的空间是需要约束的, 用计算机科学工作者更能理解的说法是这样Predict Next Token本身的机制上, 必然导致一些表征偏差, 最近DeepMind有一篇论文《Representation biases: will we achieve complete understanding by analyzing representations?》[6]关于一些神经网络表征的分析来看, 模型会天然的偏爱哪些简单的重复的特征. 而对于一些“困难特征”,则会表征不足.

有一篇来自于AI2Paradigm有趣的文章

[《那帮最懂AI的人，绕不开乔姆斯基：当哥德尔的幽灵缠上Sutton的经验，AGI的终局藏在维特根斯坦的自白里》](https://mp.weixin.qq.com/s?__biz=MzIzNTc1ODg4MQ==&mid=2247489800&idx=1&sn=aed0b14ddaaa1eb65912a427a9969409&scene=21#wechat_redirect)

里面谈到一个话题:

`上半场的维特根斯坦`：他写下《逻辑哲学论》，坚信语言可以像镜子一样，通过完美的逻辑结构，精确地描绘世界。他试图建造一座语言的“水晶宫”，并以为自己成功了。 这就是我们训练AI学数学时的梦想。

`下半场的维特根斯坦`：他推翻了自己，写下《哲学研究》。他终于悟到，语言的意义根本不在于它的逻辑结构，而在于它在具体情境中的`使用（Use）`。一个词的意义，就是它在一个“语言游戏”（Language Game）中的用法。

实际上和范畴论中的Yoneda Lemma是相同的, "对象由它与其他对象之间的关系完全决定".. 另一方面也会引入一些Topos理论的内容, 在此就不展开了...

总体来看, 个人的观点是从算法的层面上来讲, Expert Specialization去更好的捕捉困难特征上是有价值的, 而不是简单的去划分MoE只是为了相对于Dense降低成本.

因此我个人粗浅的认知是, AFD从工程上的选择是一个比较Straightforward的方案, 但是考虑到和算法本身的协同上, 大EP的实质是Expert Specialization,并且在算法和Infra上做了一个恰当的Trade-off.

当然具体的模型性能如何, 毕竟在600B~1T这样的规模, 两个方案对比实验的成本是巨大的...

另一方面的问题是关于MFA和MLA的一些讨论, Step-3 TR中有一个图

![图片](assets/8342ebeca345.png)

从Infra的视角来看, 构建一个Attention block 去更好的匹配自己能够获得计算访存比本身是没什么问题的. 但是从算法上来看关于MLA的一些优势, 苏剑林老师也有了很多很详细的分析了, 同时Kimi K2用MLA通过减半n_heads也可以降低一些计算密度.

MFA本身的论文我没怎么仔细看, 但是从一些上下界估计的视角来看挺有趣的, 或许也是蛮不错的工作.

另外从算法上来看, Muon Optimizer和苏老师最近提的《流形上的最速下降：1. SGD + 超球面》[7] 其实去年这些论文(nGPT)刚出来的时候我也提到过

[《谈谈大模型算法和基础设施的演进...》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492599&idx=1&sn=cdda29f8ca81117fb8c35c865754b91c&scene=21#wechat_redirect)

另一方面最近在做Agent相关的一些工作, 其实本质上我可能更希望一个可信赖的30B左右的模型来做一些基础的工作, 再稍微在一些复杂的任务上引入300B以上的模型, 或许从算法上来看, Agent和大模型本身的协同设计的算法是一个更好玩的方向, 即MoA(The Mixture of Agent)

### 3. 谈谈Infra的视角

对于朱老师反对一些大规模ScaleUP的观点, 我也是非常认同的. ScaleUP实质性的问题确实会让做分布式系统的人一眼看到大型机的叙事, 需要通过更好的算法来避免. 朱老师那个访谈的Session讲了很多关于AI Infra的内容我都是蛮认同的...其实去年DeepSeek-V2发布的时候, 就有过一篇文章来探讨过

[《谈谈AISys架构师的基本素养》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490468&idx=1&sn=d6dc47cb35d61b2be189840f26de75ab&scene=21#wechat_redirect)

其实也是一个很类似的叙事, Infra的人在算法和硬件之间可以做很多事情... 例如朱老师提到的, DS本身来看, 梁总有大量在量化交易上积累的Infra经验. 当然我也有, 毕竟国内几个交易所的网络架构都是我设计的, 然后各个券商机构用的一些极低延迟的网卡基本上也是我们搞的... 量化本身的算法我也做了好多年... 实际上从体系结构来讲, 无非就是更好的Data Locality和更高的并发...

但有一个实质性的问题是在AI Infra时代本质上是需要一个Composable Disaggregation Infra(CDI). 更好的Data Locality受限于硬件资源时, 催生了一些紧耦合的需求, 同时不同的workload特征也伴生了大量的Disaggregation的算法. 实际的问题还是在基础设施层面上的同构/异构的CDI的问题.

从Step-3的论文来看, Section 7.2有一些阐述

![图片](assets/b883ed603ec2.png)

当然另一方面知乎洋哥也做了一些评论, 朱老师也有一个回复

![图片](assets/4a17a1203922.png)

本质的原因: Nvidia的RDMA做的就是烂..这不光是我说的, 看看洋哥关于同Rack和PFC-only transport的评价吧. 另外有一个系列的文章也可以看看

[《RMDA这十年的反思》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3398249338911260673#wechat_redirect)

至于PFC, 可以看看AWS SRD论文的一句话

![图片](assets/5713a00f9bde.png)

简单的说AWS在diss  RoCE at Scale is not at that scale...当然NV的人可以各种狡辩, 例如世界上最大规模的集群就是PFC Lossless的....朱老师肯定也对其中很多观点会持有反对态度...

首先有一个问题, 异构的卡AFD分离后是否要做Topology-Aware的处理? 当前论文的做法是A和F在同一个ToR下. 跨ToR带来的Hash冲突和拥塞控制一直是RDMA本身的一个难题. DS在IB上启用了Adaptive-Routing部分的解决了一些问题... 当然AR本身也会带来一些延迟. 特别是在做一些Atomic处理上进行fence导致增加一个RTT...

而如果采用同Rack异构的部署, 那么大规模集群如何建设呢? 例如H800和L20混布在一起的情况下如何处理? 一些常见的MultiRail部署如何让H800和L20对接? 最终集群建设的时候通常还是会同样的卡放在一个Pod内...跨Pod之间的一些收敛比/延迟如何处理?

实际上这些问题在2023年我设计CIPU2.0的eRDMA拥塞控制算法时就已经处理的很干净了, 跨5跳(ToR-Leaf-Spine-Leaf-Tor)打满是一个非常容易实现的目标...不需要PFC/不需要调整CC/甚至连交换PacketSpray/ECN/INT这些特性都不需要.

并且那个时候就考虑了MoE大EP情况下的A2A场景, 端侧多路径负载均衡只需要一个非常简单的算法就能搞定, 并且增加了Reciever Fence的能力去避免多路径乱序增加的一个RTT...

其实DCQCN过去十年本身来看是一个不错的技术, 难点UEC也提到过“Hard to tune”, 正如朱老师的一个回复也提到没有时间去调...

![图片](assets/90260fcfbbeb.png)

其实特别是针对云服务提供商更多的是需要一个免配置的CC, 用户开箱即用直接打满就好, 这可能是站在CSP视角和朱老师大集群视角上的一些区别吧...

而关于iBGDA, 国产卡支持其实并不难, 至少CIPU eRDMA是完全支持iBGDA的, 并且完全支持多路径转发...

其实洋哥谈到了一个很好的需求: “标准的RDMA RC接口, 支持多路径, 支持Lossy, Topology Agnostic”, 当然在UCCL层也可以做很多软件的工作, 而这些工作实际上直接放到网卡上已经完全可以实现了, 进一步降低了对GPU资源的开销. 这方面期待以后能和UCCL一起做一些合作, 例如BRCM网卡上Atomic的一些问题处理上可以给洋哥一些建议:)

说个小事吧, 2024年初Nvidia的人来找到我们, 也在谈到某些功能特性可以在BF4中支持, 我的答复是: 我们早就做完了, 你还有没有别的能打动我们的功能?

其实下面这个图才是正解, 当然这是站在云服务提供商的视角上的...

![图片](assets/dd7e92ab7512.png)

当然在Infra上还有很多事情要处理, 不光是计算和网络, 还有存储上的一些能力...CIPU2.0的存储IOPS可以秒杀BF3...明确的说是BF3这一类的卡微架构的问题... 实际上这一点很多Infra的同学并没有完整的经历过芯片设计和多代处理器迭代的过程, 对于很多Tradeoff并不是很清楚...在后面一节我会详细阐述.

总结一下就是: 是男人就该硬刚Lossless, Multipath, 避免对带有特殊功能的交换机的需求, 降低交换机Buffer设计难度, 即便是一些国产的25.6T/51.2T Switch也能很容易打满.

### 4. 从硬件的视角谈谈

对于PD分离或者AFD来看, 其实历史上也找得到很多影子, 就拿GPU本身而言, 最早其实也有一个分离的特征, 那就是Vertex Shader和Pixel Shader.分离的代价是一些workload的imbalance,

![图片](assets/f55c1275c56f.png)

最终这两者怎么融合的构成了现代GPGPU的雏形, 在以前的文章中介绍了很多了

[《GPU架构演化史》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=2538479717163761664&scene=173&from_msgid=2247487954&from_itemidx=3&count=3&nolastread=1#wechat_redirect)

其实以前在Cisco的时候, 早期也是针对很多特定的功能构建了很多ASIC进行M:N的分离, 例如Table Lookup采用TCAM一类的其间, 独立的Queueing System等.. Input/Output pipeline disaggregation, 最终我们还是把这些都融合到了一颗处理器内部, 通过TBM PLU替代TCAM...通过xxx替代xxxx...然后从微码执行引擎变到通用C编程, 本质上和NV CUDA是基本上一致的.

其实针对国产GPU chiplet这些, 朱老师的一些描述我不是很认同. 从一个处理器设计的视角来看, 主要的目标还是需要尽量能够满足多种workload的需求, 而不是强假设在某种算法上. 当然Model本身去Fit HW的事情还是要做的, 这一点是认同朱老师的. 但既然是co-design, 进一步的需要从算法的层面到Infra的层面去指导硬件的迭代, 硬件本身也需要去追求在多种workload下更好的性能.

其实在芯片体系结构上超越NV还有很多路要走, 模型和硬件的协同设计有巨大的空间可以去做... 例如Google的TPU就是一个不错的例子. 但是NV本身的技术壁垒有很多很多的细节, 例如在编译器上, 软件接口抽象上, 内部的一些很细节的调度以及Cache的处理上,NOC设计上有很多细节没有吃透前, 简单的评价GPGPU(SIMT) vs NPU是不恰当的.

芯片设计上平衡各种workload的性能才是最关键的:)

当然从经营的角度来看, 还有一个硬件成本, 特别是互连成本的问题在那里....这是逃不掉的, 最后一算账....另一方面, 对于交换机也要保持一个更乐观的心态, 实际上112G Serdes来堆高Radix, 特别是当拥塞控制做好了以后, 不再需要复杂的TM后(很少有Infra的同学明白我在说什么), 做一些事情还是很容易的.. 对于Ring/Torus/xD-Mesh或者K-hop Ring这些异构拓扑针对单一租户大集群maybe有它的收益, 但是作为CSP还是有很多的TradeOff的,云上经营的弹性多租能力才是决定拓扑的关键....

参考资料

[1] 
Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding: *https://arxiv.org/pdf/2507.19427*
[2] 
Step-3推理系统：从PD分离到AF分离（AFD）: *https://zhuanlan.zhihu.com/p/1932920900203807997*
[3] 
Llama-5 锐评 “Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding”: *https://zhuanlan.zhihu.com/p/1935224722515275880*
[4] 
关于 AI Infra 的一切 | 对谈阶跃星辰联创朱亦博: *https://www.xiaoyuzhoufm.com/episode/688cc1cc8e06fe8de7d920cd?s=eyJ1IjoiNjJmZWY2MTdlZGNlNjcxMDRhMjEzMmQzIiwiZCI6MX0%3D*
[5] 
ShallowSim: *https://github.com/zartbot/shallowsim*
[6] 
Representation biases: will we achieve complete understanding by analyzing representations?: *https://arxiv.org/html/2507.22216v1*
[7] 
流形上的最速下降：1. SGD + 超球面: *https://kexue.fm/archives/11196*