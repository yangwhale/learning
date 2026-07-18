# 从ODCC夏季会议再来谈谈ScaleUP的一些进展

> 作者: zartbot  
> 日期: 2025年7月1日 00:07  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494310&idx=1&sn=8f2c1970b0be51e830d5dd85efac59f4&chksm=f995fa64cee273724155cefdea4183a56de60b79c8a9b84c09a29694eef6f747e13749f672a6#rd

---

### TL;DR

上周有一个ODCC夏季会议, 现场看到了一些ETH-X ScaleUP机柜/板卡的实物. 另一方面BRCM也在讲了一些SUE的内容. 而去年热热闹闹另外一些类以太网的ScaleUP方案以及某个国内首创没了声音.

另一方面UALink请来了Board Chair, 来自AMD的Kurtis先生. 然后还有Asteralabs, Synopsis, 还有一些国内的GPU/交换机/Serdes厂商.

两套方案还有一些争议, 主要集中在延迟/带宽/规模上的取舍.. 个人的观点是: 对于ScaleUP的实质性的定义上是存在争议的, 一个是计算域(计党)从总线演进的视角来看, 另一方面是网络域(网党)从互连的视角来看. 从组织结构上来看,以往的主机内的互连都是服务器团队负责, 主机间的互连都是网络团队负责. 表面是两党之争, 实质是这里面有太多的取舍和平衡.

仔细想了一下, 核心问题是两党对于整个AI Infra从应用到算子开发,再到处理器(GPU/CPU)微架构, 最后到大规模互连整个技术栈的理解是有局限性的, 各个团队都想在自己的领域获得局部的最优解, 然而各家并没有一个完全的End-to-End的架构师或者架构师团队来帮助他们达成交易.

因此本文将先介绍一下两种路线在这次ODCC发布的一些信息. 其实对于我而言并没有太多的偏向性, 等两者在2027年左右竞争分出胜负了选一个就行, 或者选一个更加合适的融合路径, 但是为了让这个路径走的更加轻松一点, 我会从应用的视角详细分析一下, 并提供一个让两党逐渐达成交易的潜在路径. 本文目录如下

```
 1. Ethernet based ScaleUP 2. UALink 3. 谈谈ScaleUP交换机设计 4. ScaleUP的需求 4.1 ScaleUP的规模? 4.2 延迟的约束 4.3 ScaleUP的一些误区  5. 从市场竞争的角度分析 5.1 AMD应该加速在CPU上支持UALink 5.2 回顾一些历史: 为什么不是PCIe 5.3 SUE和UALink协同的方案
```

### 1. Ethernet based ScaleUP

从去年开始, 腾讯主导ETH-X到现在快大半年的时间就交付了一套实物, 整个国产供应链在系统层面还是很领先的, 帮助NV构建NVL72的时候积累下来的技术基本都能用上了, 现场有一个立讯的CableTray和机柜的实物, CableTray/液冷这些技术都平滑的用到了Eth-ScaleUP技术上.

![图片](assets/eac540c0cf01.jpg)

现场华擎工程师对于整个机柜的供电/液冷/CableTray/线缆等相关组件进行了详细的解析, 特别是在连接器的可靠性上做了很多改进, 例如一些浮动设计, 紧固件的设计等.可靠性在过去一年应该有大幅度的提升, 而未来448Gbps的高速互连也有相应的路标. 除了CableTray, 中兴也提供了一些正交互连架构的方案. 当然还有一些其它Vendor的基于光互连的方案...例如NPO/LPO/CPO等...

另一方面借助现有的芯片, 基于BRCM TH5和Intel Gaudi3构成了交换板和计算板, 特别注意的是基于BRCM交换机有更好的Radix, 还留出了大量的前面板光模块接口用于2级的互连.

![图片](assets/76d870e50ef7.jpg)

ETH-X当前传输还是基于RDMA的方式构建, 而后面伴随着BRCM的SUE(ScaleUP Ethernet)标准的演进可能会有一些进一步的发展. 实质性的争议还是在Ethernet本身的传输效率上. 标准的Ethernet在传输小数据时有大量的Overhead, 因此BRCM SUE对其进行了一些优化

![图片](assets/ace5f7f6a153.png)

同时针对传输时还有一些动态的packing技术, 通过在IO Die上构造per Destination的队列, 然后发送时通过一个调度器轮询将多个数据打包传递.

![图片](assets/38f4b98296c1.png)

但是这种方式在实际workload下的打包效率及调度延迟不容忽视, 这是现场带来的一些争议焦点.

### 2. UALink

UALink在现场有一些争锋相对的胶片来讨论延迟的问题

UALink

Ethernet

Latency TX+RX

60ns

330ns

Latency Switch

175ns

625ns

Total Latency

235ns

955ns

Serdes BW

212.5Gb/s

212.5Gb/s

MaxBW per lane

NoAuth 190Gb/s 

withAuth 178Gb/s

177Gb/s

BW Efficiency

NoAuth 93.5% 

withAuth 87%

64~89%

Crypto

End-to-End

End-to-End

机密计算

yes

yes

Strong Vendor Support

yes

yes

其实这个数据有一部分问题, BRCM在以太网交换机芯片TF1上可以极大的降低延迟到200多ns. 即便是如此, 在一些传输效率上两者之间还存在一些差距, 这些差距在一些具体的协议设计上.

UALink的想法是在GPU侧尽量维持更简单的实现, 降低I/O组件对芯片DieSize的占用, 对于数据不分Destination都直接打包到一个640Bytes的FLIT中, 然后传输给交换机, 由交换机根据不同的Destination拆包转发, 因此才会有64B TL/640B DL的传输方式.

![图片](assets/071a856f7420.png)

但是这样处理对于交换机有一定的复杂性, 这也是网党的主要攻击点, 毕竟“Smart Edge, Dumb Core”的原则在那里, 交换机设计越简单吞吐和Radix越高.

### 3. 谈谈ScaleUP交换机设计

做计算的同学可能对交换机芯片架构有一些不理解, 在90年代初期开始, 交换机通常是采用PortBased Buffer架构, 然后交换机通过简单的Store-Forward机制进行转发, 如下左图所示:

![图片](assets/7d5d9a289fef.png)

但是随着更高的吞吐带来的拥塞, 逐渐演进到缓冲区池化的Shared Buffer交换机架构, 然后再逐渐加入多队列管理等灵活的缓冲区分配算法和优先级调度额能力, 因此通常也把这个共享Buffer的组件称为Traffic Manager(TM)或者也被称为MMU(Memory Management Unit) , 如上右图所示. 其实对于现代交换机而言, 做大整机吞吐和更高的Radix的瓶颈都来自于TM的设计, 这是各个交换机芯片厂商的看家本领.

但是随着交换机端口数的增加, Serdes速率增加, 多端口读写的TM逐渐在容量增长上放缓, 近代有一些交换机又不得不将TM分成独立的不同区域.

而对于ScaleUP交换机, 延迟是一个非常重要的因素, 本文后面一章会详细分析延迟带来的影响. 历史上为了优化交换机延迟, 也出现了Cut-Through的交换机架构, 传统的Store-forward架构需要将整个Packet接收并存储完成后再进行转发, 而Cut-Through在接收到报文头时即开始处理, 保证在极短的时间内查询到目的端口后, 无需存储到TM中的队列,而直接发送到目的端口逻辑.

而这些Cut-Through的交换机架构实现, 又回到了上图左侧的Port-Based Buffer架构. 流控机制上也逐渐回退到Credit-Based方式(Credit-Based FlowControl, CBFC).

而UALink更加追求低延迟, 在交换机上实现了B2BUA(Back-to-Back UAlink Agent)的架构.

![图片](assets/2aa32ac8a690.png)

看上去挺复杂的, 因为对于网党而言主观的会认为有一个细致的Payload Lookup然后拆包组包的过程. 相对于SUE交换机的实现更加复杂. 但事实上有一些约束和取舍下, 这样做也没有很大的代价. 下面详细阐述一下.

考虑到交换机的Radix和带宽, 要引出大量的Serdes, 那么芯片面积相对需求较大. 而为了极致的低延迟又无法使用TM进行Store-Forward. 那么Port-Based buffer Switch架构则成为一种选择.

报文通过XBAR传输时通常可以有两种选择, 一种是基于变长报文, 直接将报文通过XBAR进行传输. 另一种做法是将报文分割为一个个小的固定长度的单元, 然后通过XBAR传输. 在输出端再重新组合成为完整的变长报文.

使用固定长度的Cell非常使得转发非常简单. 我们可以计算和估计出固定长度信元的转发时间. 并将其成为一个时隙”time slot”, 所有的输入/输出都可以在一个固定的时隙内完成. 而对于变长报文转发则显得十分困难, 尤其是在Scheduler. 由于变长报文转发所需要的时间是无法估计的.

也就是说在交换机XBAR转发时都要切成固定大小的Cell, 还不如直接把UALink的TL FLIT做成Cell大小, 并内部携带自路由的信息. 即在每个TL FLIT中都有Destination ACCID.

另一方面是一个老生常谈的问题, XBAR的缓存和Head-of-Line Blocking的问题. UALink是按照Output Queue(OQ)的方式设计的, 在Output的地方由于目的地都是同一个直接打包成DL就可以转发. 这是一个典型的低延迟高吞吐的设计方法, 但是在一些多打一的情况下, OQ的写入速度要求会非常高, 统计上来看需要2x SpeedUP.

![图片](assets/504d84b9772a.png)

因此退而求其次可以采用Virtual Output Queue(VOQ)的方式配合iSLIP这样的调度算法.

![图片](assets/f2f49a76b138.png)

实际上UALink在incast的情况下还有一个比较好的优点, 当多个加速器向一个加速器发送数据时, 交换机将DL FLIT拆分成多个TL FLIT的时候, 多个TL FLIT可以互相交织的发送到Output队列进行组包构成新的DL FLIT, 因此整个系统的Jitter会小很多...

### 4. ScaleUP的需求

几个争议的问题:  ScaleUP规模到底要多大? 极致的低延迟真的需要么? 国外为什么要使用铜互连, 不采用光互连?

#### 4.1 ScaleUP的规模?

一个很朴素的想法就是国产算力受工艺限制, 单芯片的峰值算力基本上是国外的1/5左右. 因此有一种说法是国产算力通常需要5~10倍的ScaleUP域来获得等效于国外的性能.  例如NVL72按照5~10倍估计接近华为CM384这个值.  但是有一些现实的约束会使得ScaleUP的规模受到限制.

那么从规模上而言, 国产算力对于ScaleUP可扩展性的标准有更高的要求或者是说更多的焦虑. 普遍问到的问题都是UALink怎么才支持1024个Accelerator, UALink当前的标准为什么只有KR没有定义光传输?

其实更多的组件导致系统整体的MTBF的下降, 另一方面还有一些关于光互连和铜互连的争议存在. 当然从互连域上的失效来看, LLR和ScaleUP原生的XPU多个平面转发都能一定程度上缓解, 但个人还是希望有这么一个相对宽松一点的目标:  在系统维持80%以上吞吐的情况下的MTBF需要达到xxxx小时. 这是其中一个约束, 如果不行, 那么就要降低整个系统的ScaleUP规模是否能够获得?

另一个问题是来自于计算本身, 性能并不是完全能够按照ScaleUP的规模增加而线性增加的, 计算时的矩阵拆分细到一定程度, GPU本身的算力也会下降, 因此存在一些边际效应. 另一方面的观点是潜在的会有更大规模EP的需求, 例如EP512等...但是个人有一个直觉就是这样的模型本身Active Expert数量也可能需要等比例放大, 这样也加重了Dispatch/Combine的负担, 从模型结构上来看我觉得短期内并没有更大的EP需求, 而且从DeepSeek早期论文来看, Decode的策略是EP320, 最后公布的也降低到了EP144.

对于ScaleUP的规模除了极致的性能以外, 还有一些弹性部署的需求. 例如EP320相对于EP144可能需要一个更高的BatchSize才能得到更优的收益, 而实际的推理任务中的峰谷效应, 按照EP144的规模进行弹性部署可能整体成本更加低一些, 可靠性也好一些? 所以在推理实际工况和峰值性能之间有一个成本的取舍.

基于这两点分析, 我个人认为超过1000节点的ScaleUP的需求是难以成立的.

#### 4.2 延迟的约束

几个月前有一篇已经讲的很清楚了

[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect)

主要是两方面的问题, 以网党的视角来看, 以常见的通信特征分析, 例如基于64x64的TILE或者一个Token大小都是4KB~8KB, 通常GPU ScaleUP会有多个Lane连接到多个交换平面. 因此传输延迟基本上会达到ns级别. 此时静态延迟的影响将会更加显著, 甚至是RS544这样的FEC都会有显著影响. 网党的做法通常是RS272 lowlatency-FEC配合LLR, 然后再配合CutThrough交换机.

以计党的视角来看, Multi-Stage的GEMM为例, SM内的寄存器文件容量/SMEM/TMEM容量和实际的SM FLOPS这几个因素决定了对延迟的需求

![图片](assets/7b0e4d59fc54.png)

虽然GPU可以支持更大的outstanding, 对于延迟的需求让我想到CPU通用计算里的一个经典的Killer Microsecond问题.

![图片](assets/2e07064e74da.png)

如果延迟维持在ns级别和ms级别都好处理, 而在us级别的处理难度陡增.

GPU也有类似的Killer Microsecond的问题? 大概有一个直观的感觉, 延迟维持在1us左右的ScaleUP对GPU的SM是友好的. 这也是UALink极致追求的. 正是UALink对于<1us的RTT的追求, 同时尽量使用1层交换机组网, 避免2层带来的延迟., 因此其部署规模大概也就1~4个Rack, ScaleUP域规模为10bits 1024卡.

而当延迟大于3us以后似乎和10us以上也没有太大的区别了? 为了克服延迟的影响, 又要加大更大的L2Cache和SMEM似乎这个trade-off下又有一些得不偿失?

另一方面是从整个系统的视角上, 算子同学一定是那种有多大带宽就能通过算法榨干的那种. 因此延迟上我们需要考虑整个系统接近满载的情况, 那么就要考虑到满载时的队列延迟影响, 即Kingman公式如下所示, 接近满载时延迟会陡升.

![图片](assets/bb5bdc9b9b39.png)

从影响因子来看, 我们需要尽量的降低网络到达时间的变异系数和服务时间的变异系数. ODCC夏季大会上沐曦也谈到了类似的观点.

![图片](assets/7e7d13b823c4.png)

网络来看, 如前面分析, UALink Switch基于64B TL FLIT的转发方式, 特别是在一些incast情况下, UALink会稍微领先于SUE的方案, 而以太网Cut-through交换机的方案会领先于Store-forward的交换机方案.

对于以太网延迟, 4年前做NetDAM有一个测试, 经过一个支持CutThrough模式的Innovium(Marvell) Teralynx 7, 交换机延迟大概在200ns, 经过一跳交换机的One-Way Latency大概为610ns, Round-Trip也就1.2us左右.

![图片](assets/e838fc54293d.png)

是否真的值得为这200ns~300ns的延迟放弃通用以太网的兼容性? 这一点我是存疑的..

#### 4.3 ScaleUP的一些误区

很多时候, 我们对大带宽的需求来自业务, 我们将其称为High-Bandwidth Domain(HBD). 但是我们不能将HBD和ScaleUP画上等号. 从整个系统来看还有很多的约束.

ScaleUP这张网追溯起来可能要到3Dfx Voodoo的年代的SLI. 连接CPU的PCI带宽有限, 于是多张显卡开始用私有的总线互连. 然后逐渐演化成了NVLink以及NVSwitch到后来的NVL72...

对于我而言, HBD的定义是一个具有极大带宽的支持内存语义的Domain, 然后内部再划分ScaleUP和ScaleOut. 因此一直有一个疑问: **具有Memory Semantic的ScaleOut算不算Ethernet based ScaleUP?** 事实上Memory Semantics替代RDMA可以节省接近3us的RTT延迟, 使得RTT延迟也接近1us. 跨越2层交换网的RTT延迟估计也就2us. 当然在网络中重载的情况下,例如95%以上利用率时SUE和UAL两者的延迟差距可能会因为转发模式的一些差别有所放大...

某种程度上来说, 这些争议不光是UAL和SUE. 即便是华为的CM384和UB-Mesh也有...ScaleUP和ScaleOut域的边界在哪里?

### 5. 从市场竞争的角度分析

BRCM及其代表的网党优势在交换机, 高速Serdes以及未来的一些光互连, 例如CPO/NPO/LPO这些领域. 但是他们和我当年在Cisco做NetDAM的时候有点类似的尴尬, 自己没有处理器... 而且一些沟通后发现他们整体对于GPU/CPU这些处理器的Memory Order的理解还是有一定的问题的.

UALink及其代表的计党优势在处理器上, 特别是在服务器CPU市场上已经和Intel平分秋色了, 最近Intel也在UALink的官网上发布了一个支持UALink[1]的消息. 另一方面在标准中看到了UALink考虑到了很多安全因素, 例如ACCMMU可以做一些地址映射将物理地址(PA)和VA解耦带来了虚拟化和多租户隔离的能力, 另一方面安全上也支持Auth. 相对来说都是比SUE更加完善的地方.

但对于UALink整个生态而言, 毕竟参与方太多, 兼容性测试还是非常重要的. 但是不同厂家的GPU通过UALink互连的场景是很少的, 当然如果我们做一些Attention和Expert分离, 将Expert的计算完全DSA化还是有一定的价值的.

#### 5.1 AMD应该加速在CPU上支持UALink

我个人的建议是AMD很快的流一片给CPU用的UALink的IO Die出来供大家测试和开发周边的其它芯片.

通常一个公司需要推动一个标准时, 最好从自己最强势的产品入手. 如今EYPC的市场占用率已经接近50%,  然后下一代的CPU Venice单个Socket的内存带宽已经1.6TB/s, 非常好的一个socket用于UALink测试和未来在Agent/KVCache等一系列业务中加入到Scale UP域中. **从业务视角来看, 通用计算CPU及其内存子系统在Agent时代对带宽的需求更高, 详细的业务视角的分析后面单独写一篇.**

个人觉得通过在一些server/workstation乃至家用PC上打开一个UALink的口子, 这将使得UALink的生态快速的繁荣起来, NV现在通过C2C绑定Grace CPU, 而AMD和Intel如果能够提供X86生态上的高带宽GPU互连, 将对很多推理任务提供显著的优势, 例如在2026~2027年, NV GPU连接X86还只有PCIe Gen6x16的时候, 如果有一个单向400GB/s的UALink连接GPU呢? 推理市场的格局将发生很大的变化...

当然很多人要问, 为什么不继续用PCIe呢? Gen6x16也有100GB/s的带宽了?下面一节详细来谈谈.

#### 5.2 回顾一些历史: 为什么不是PCIe

历史上的总线争议也很多, 从早期的PCI无法满足GPU带宽需求, Intel开始搞AGP, 然后再到PCIe. 而AMD的HT总线公开后也被很多处理器采用过. 然后PCIe也做过一些RackLevel的Disaggregation的尝试, 后面便是OpenCAPI/GenZ/.... 很遗憾的是这些东西被当时如日中天的Intel埋没了, 然后当大家统一走向CXL的时候, 生不逢时又遇到Intel自身的一系列问题, PCIe演进太慢, 历史包袱太重了...即便是CXL也受制于PCIe的物理层...

对于UALink替代PCIe, Intel和AMD应该都是很愿意去做的. PCIe控制器在GPU中占用的DieSize也挺大的...

而如今的PCIe就和当年的PCI-X有点类似了, 走到了历史的末路了, 历史上从ISA到PCI的演进的事情再发生一次也不是不行? 其实我们看到AMD在MI400上支持3个PCIe6.0x16提供300GB/s的能力, 还不如干干净净的上一个400GB/s的UALink呢?

#### 5.3 SUE和UALink协同的方案

前天和上海某金融机构的副总聊天, 大概也聊到了这些事情. 圈外人对于这些争议的看法就是:“就你们这群搞IT的人事多, 天天想着吃独食, 都不知道如何好好的合作和珍惜自己的羽毛” 从金融狗的视角来看, 渡让一些利益大家一起搞钱不是更好么...

在1~4个机柜的范围内, 用UALink互连, 将CPU/内存/网卡/GPU都连接到这个总线上, 保证计党以及很多服务器产业链的利益, 同时在这个紧耦合的域内充分利用UALink低延迟的特性. 然后网卡无非是从原来的PCIe->Eth变成了UAL->SUE, 进一步可以通过2层Radix=512的以太网交换机将SUE扩展到十万卡的规模.. 网党的利益也可以保证.

顺带一些内存厂商也可以在UALink上分一杯羹, 例如一些想做PIM/PNM的内存厂家顺势也可以通过UALink接入一些内存池的能力...对于云服务提供商也可以很容易的将存储和传统通用计算服务器纳入到这个总线上.

这个方案唯一受损失的估计是NV比较强势的RDMA, 反正做啥都要得罪它, 还不如干干净净的把它干掉...

![图片](assets/f36132b73359.png)

参考资料

[1] 
Intel support Ualink: *https://ualinkconsortium.org/blog/intel-supports-ualink-for-scale-up-networking-829/*