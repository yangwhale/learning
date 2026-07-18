# 详细分析一下Nvidia Rubin CPX

> 作者: zartbot  
> 日期: 2025年9月17日 01:25  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496031&idx=1&sn=9acec4b621752d263fb600718e43d54a&chksm=f995e19dcee2688bac86254319377a98879221867228454b810e2f3513eea0e76139cdb275b0#rd

---

### TL;DR

上周 Nvidia 发布了Rubin CPX, 对外的宣传是专为大规模上下文推理而设计的新型 GPU. 看了一些分析文章, 其中有一些分析在对比ASIC, 有些在对比HBM和GDDR7, 然后在Rubin CPX NVL144的Prefill/Decode配比上也有一些争议, 每个人观察的视角都有一些片面, 然后一些数据又是以Jensen Math给出的, SemiAnlysis的报告中也有一些错误,  因此本文对它进行一个详细的分析, 同时对基于Rubin CPX的PD分离架构策略进行一些探索.

### 1. Rubin CPX芯片分析

Rubin CPX是一颗基于Rubin架构但使用GDDR7的芯片. *相对于基于HBM的平台不受CoWoS封装产能的限制出货量可以很高*. 芯片的基本规格如下:

![图片](assets/e1fa81d3cce0.jpg)

N3P工艺, 1 x Reticle Size

NVFP4浮点算力 30PFLOPS

128GB GDDR7内存, 内存带宽2TB/s

互连仅支持PCIe Gen6x16, 支持单卡800Gbps带宽

内置了视频编解码, 但是是否带有光线追踪的RTCore未知

针对Attention计算中Softmax相关的指数计算SFU性能比B300提升了3倍

从芯片规格来看, 整体芯片还是对RTX 5090/6000 pro(GB202)的延续, 估计有192个SM, 内存位宽为8 x 64bit-GDDR7, 如下图所示:

![图片](assets/240cc6121273.jpg)

值得关注的是累计算力达到了30PFLOPS, 相对于Blackwell这一代的GB202(RTX 5090/6000pro), TensorCore的算力提升了10倍. 而30PFLOPS甚至超过了1x Reticle Size的Rubin Die(单颗Rubin在GTC25上宣布为50PFLOPS). 对于单个SM来看, TensorCore和SFU都将占用更大的芯片面积, 是否进一步砍掉一些高精度算力?

### 2. Rubin CPX NVL144

新一代的Vera Rubin架构还是延续GB300这样的Oberon机框结构, 单柜版本为18个ComputeTray和9个SwitchTray

![图片](assets/69a1f905200c.png)

对于单个ComputeTray包含了4颗Rubin GPU(每颗由2个Die封装)和2颗Vera CPU以及8颗CX9.

#### 2.1 Vera Rubin CPX NVL144

这一次在ComputeTray发布了另种变体, 一个是在标准ComputeTray上增加了8颗Rubin CPX构建的Vera Rubin CPX NVL144, 如下图所示:

![图片](assets/fe9474ffb36e.jpg)

这种做法的优点是Rubin CPX直接PCIe总线连接到Vera上, KVCache传输的功耗较小. 但是缺点是Prefill和Decode形成了固定的配比, 并不灵活.

#### 2.2 VR CPX + VR NVL144 Dual Rack

另一种做法是采用两个机框的部署, 在VR CPX机柜中, 只有Vera CPU和Rubin CPX, 而没有Rubin芯片, 因此机柜只有18个ComputeTray, 没有NVLink SwitchTray. 而另一个机柜则是标准版的Vera Rubin NVL144机柜

![图片](assets/1b5066428ace.jpg)

这种做法相当于是分离式的部署, 优点是可以根据自己的需求灵活的实现xPyD的配比, 但是缺点是Vera CPU和CX9网卡的数量翻倍了, 而KVCache传输需要通过RDMA网络传输, 功耗和成本都更高. 下一章会从机内拓扑的角度来谈谈

### 3. Inside Rubin CPX

我们再来进一步分析一下机内的拓扑, 这对后面一个章节分析Prefill-Decode分离策略十分重要.

#### 3.1 机内PCIe拓扑

标准的Vera Rubin NVL144机内PCIe拓扑如下左图所示, 而Vera Rubin CPX NVL144机内PCIe拓扑如下右图所示:

![图片](assets/477cdb4785be.png)

CX9并没有像GTC25 Keynote上公布那样成为一颗1.6Tbps的网卡芯片, 而是继续维持在800Gbps, 主要是在CX8的基础上修正了一些bug和增加了某几个公司的一些功能需求. 因此CX9内置的PCIe Switch依旧约束在48 Lane.

对于标准版的VR NVL144, 可以通过16x连接CPU, 16x连接Rubin, 并剩余16x可以连接NVMe盘, 但是考虑到前面板的空间约束, 单个CX9应该只能放置1块最多2块盘. 这个机型的好处是对于ScaleOut网络依旧可以通过GPU-Direct-RDMA进行通信.

而对于VR CPX NVL144的版本, 需要留一根PCIe Gen6x16给Rubin CPX, 因此判断在CX9+Rubin CPX的子卡上, 断开了PCIe的连接. CX9可以GDR到Rubin CPX, 利用ScaleOut网络执行Prefill的计算. 而CX9无法通过ScaleOut GDR连接到Rubin.

而对于Vera Rubin CPX only的计算板拓扑如下所示:

![图片](assets/d98989f67802.png)

#### 3.2 其它CSP改造的可能性

其实对于其它的CSP更有可能选择Dual-Rack的方案, 但是并不需要官方的CPX Only Rack集成Vera CPU, 而是可以直接通过一个PCIe Switch Box旁至, 并使用PCIe AEC互连, 如下图所示:

![图片](assets/dae7c0b371c3.png)

这样的好处是既可以定制自己的网卡, 例如AWS Nitro, 又可以挂在更多的盘,并且还能维持Rubin GPU的 GDR 能力, 综合成本和功耗也应该小于官方的VR NVL144 + VR CPX only的Dual-Rack方案. 甚至还可以定制在PCIe Switch下挂载多个Rubin CPX芯片.

另一方面还可以像Meta那样, 对于VR NVL144依旧构建Dual-Rack的方案, 采用Vera CPU和Rubin 1:1配比. 然后左右两边并柜放置Prefill的PCIe Box.

这些做法还有可能进一步的去做一些有趣的事情, 涉密就不展开了.

### 4. Prefill-Decode策略

#### 4.1 PD分离概述

Nvidia官方有这样一个描述, Prefill阶段是一个Compute Bound的计算. 对于Coding/Agent一类的LLM场景和视频生成(例如Veo3)/图片编辑(Nano Banana)场景来看, Prefill的长度通常会很长. 通用的GPU(例如Rubin)来看虽然算力/内存/NVLink互连带宽都兼顾了, 但是整体ROI来看并不好.

![图片](assets/83c72e4025a8.png)

因此构建一个Rubin CTX, 放弃对高内存带宽的需求, 使用GDDR7降低成本, 另一方面也放弃对高NVLink互连带宽的需求, 专注于Prefill的场景.

![图片](assets/f9e754661de4.png)

#### 4.2 Rubin CPX 的PD分离策略

我们来探讨几种情况下的Rubin CPX的P-D分离策略.
4.2.1 Rubin CPX ScaleOut Prefill
首先是针对所有的Rubin CPX, 通过ScaleOut RDMA网络构成集群进行Prefill处理, 再通过PCIe将KVCache传输给Vera或者Rubin进行Decode Generation. 我们可以等效的看作每个ComputeTray 8张卡,每张卡800Gbps带宽构建的一个144卡的Prefill集群, 对于长Context而言, Attention计算应该没有太大的问题, 而ScaleOut带宽是否足够支撑后续的MoE 的EP并行?

其实在Dual-Rack的方案中就是这样的, Rubin CPX Only的机柜只能通过RDMA ScaleOut网络通信. 同时又有大量的KVCache也要通过RDMA网络传输到VR NVL144机柜. KVCache传输和EP的干扰也是一个麻烦事, 毕竟和单机柜的VR CPX NVL144相比, RDMA传输KVCache比起机内直接D2H copy带来了一些不确定性.

4.2.2 Rubin CPX with NVLink
然后就是第二个方案, 是否可以借助NVLink的大带宽优势呢? 也就是说Attention计算完了以后, 传递一份Token给Vera, 然后通过NVLink dispatch多份到其它ComputeTray, 然后顺便还可以借助DeepSeek-V3这样的Group方案来做2级的分发, 即按照一个ComputeTray一个Group的方式分配Experts, 然后在Nvlink上减少Dispatch的份数, 避免NVLink上对其它Decoding的任务产生影响(例如Rubin的L2Cache污染/HBM带宽占用等). 这是一个可以探索的方向.

这个方案可以降低一些通信量, 并且把NVLink的一些带宽用起来, 但是Vera CPU并不一定能在PCIe上扛住这么大的带宽, 毕竟8x800Gbps已经达到800GB/s了, 这么大的流量穿越Vera还是有一些潜在的问题的. 那么在3.2节提到的基于PCIe Switch, 让Rubin CPX直接PCIe P2P拷贝到Rubin可能是比官方的VR CPX NVL144更好的方案?

4.2.3 Rubin CPX Attention with Rubin FFN
是否能够借助Rubin来做一些Expert的计算? 但是需要考虑整个Timeline如何去做Overlap. 并且不影响Decode. 当然Decode阶段NVLink带宽和GEMM本身的效率来看也需要攒够更大的Batch提升MFU.

可能这种方案有收益, 但额外的极长的Context在Rubin上计算可能对Decode也带来了负收益, 这些取决于SLA标准如何定义, 然后平台如何取舍.

4.2.4 一些混合调度方案
是否还是在NVL144中配置xPyD的方案, 仅对SeqLen很长的任务Offload到Rubin CPX处理? 这也是一个潜在的可以尝试的调度策略. 因为我们还需要考虑KVcache对显存的占用. Rubin CPX毕竟只有128GB的显存. 例如对于一个256K Seqlen的Prefill最高能到多少并发也需要根据模型计算的.

### 5. 一些商业上的分析

总体来看, Rubin CPX只是原来Hopper那一代L40和Blackwell这一代的RTX6000 pro这条产品线的延续, 然后重新包装成了一个Context GPU的概念并且集成到Rubin NVL 144中搭售. 也就是说Rubin CPX并不是专门为Long Context设计的, 而是恰好适合来做这件事情. 另一方面我们也看到了RTX 6000 pro的销售似乎并不好, 而GB200 NVL72相对于B200的ROI也并不是那么的好, 贵了1.5倍? 然后性能只有少数一些case下才有40%以上的收益. 实际上B200更划算? NVL的故事如何讲呢?

官方的两种方案, VR CPX NVL144采用固定配比, 并且丧失了ScaleOut GDR的能力, 而Dual-Rack方案虽然天然的支持xPyD, 但又会导致在ScaleOut上同时进行KVCache传输和EP并行的流量产生干扰的问题, 同时多了很多Vera CPU和CX9网卡. 老黄那句“Buy More, save More”是否成立?

如果专门For Long Context那么为啥不做一个带两个PCIe Gen6x16的 Rubin CPX呢? 进一步扩大ScaleOut带宽?

当然另外一方面, 模型本身也在尝试MoR和Universal Transformer以及一些Linear Attn的事情. 前面两个对于Attn计算的算力要求会更高, 而后面Linear Attn对算力要求会低不少. 这些Trade-off如何处理呢? 等后面有时间从数学上详细分析一下Linear Attn再说吧.

总体来看, Rubin CPX可能还是蛮有收益的一个尝试, 但并不是SA所说的那样 Another Giant Leap.