# 从芯片架构的视角谈谈ScaleUP, smart or dumb

> 作者: zartbot  
> 日期: 2026年2月12日 23:40  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497540&idx=1&sn=2895e0768fee36abdbe3c0f7579e0ac5&chksm=f995e786cee26e90f3be47f9b7f452049977f3efb4b4214f3af3b3829be3f50cdc103c38ffb3#rd

---

`本文仅代表作者个人观点, 与作者任职的机构无关`

### TL:DR

最近看到一些文章和一些国内外GPU厂商以及网络芯片厂商有一些交流, 关于ScaleUP的很多分歧还没有收敛. 其中掺杂了大量的各个公司的利益, 各种意见似乎表面上站在自己的利益上都是正确的, 但整个系统却陷入了一个困境.

例如网党经常会强调的**Smart Edge, Dumb Core** 在ScaleUP场景下是否正确? 作为全球第一个搞Ethernet ScaleUP的人想再来展开说一下, 任何的选择抛开计算芯片微架构和内存模型的前提下进行设计和取舍是有缺陷的.

至于精简以太网头部, 例如BRCM的AFH, 其实在21年完成NetDAM的时候, 我就和国内某个交换机芯片公司提出了MAC字段做一点UDF然后支持LPM**处理的能力.

说说我的个人观点吧, **我会比较认同ESUN这样的链路层摊销成本的方案, 但是不太认同SUE**或者其它某些传输层的解决方案.**

## 1. Smart Edge, Dumb Core? 从 ScaleOut 谈起

先说个题外话, 似乎现在的ScaleOut网络成了Smart Edge, Dumb Core的忠诚性考验, 很有趣的是在ScaleOut场景下似乎在放弃这个信条, 谈论着端网协同** / PacketSpray / AdaptiveRouting / DLB / GLB / 在网计算等一系列**Smart Core**的能力建设. 例如隔壁某厂天天吹的102.4T方案, 看了一下SGLB的论文AlltoAll在80%链路负载时, 带宽利用率仅 50% 都能发顶会论文...

![图片](assets/7c92cd52ecb6.png)

感觉似乎有点忠诚不绝对的样子, 真正的网工需要直面惨淡(Dumb)的Fabric Core!

信奉**Smart Edge, Dumb Core**, 那么能不能做到不使用任何交换机的高级特性(包括INT / PacketSpray), 就连 ECN / PFC 都不用, 纯粹的用端侧网卡解决所有的传输拥塞和多路径Hash冲突等问题?  另外对于长距离传输, 不需要更深的交换机Buffer, 也不需要什么交换机QoS策略, 做到跨可用区把带宽打满? 答案是肯定的, 你做不到只是因为你的算法哥数学学少了点而已, 而我们几年前就做到了, 过了一年看到某个公司快想到了, 在2024年保护性的申请了专利, 如今专利过了18个月公开了似乎都没看到这些公司做出个能用的...挺失望的...

最近国内某几个公司的人还在问我, 对于交换机链路失效如何通过增加交换机功能来通知端侧. 其实只要你端侧的算法得当, 真的不需要.

另一方面, 从当初为了避免Hash冲突改拓扑成Rail-Based部署, 然后伴随着MoE**这些大EP的出现最近似乎又开始折腾多平面CLOS架构还是要硬刚多路径... 这也宣告一些企图用拓扑变更来解决问题的方案似乎走到了终局.

对于ScaleOut, 尽量在端侧解决所有的多路径负载均衡 / 可靠传输 / 拥塞控制遵循Smart Edge Dumb Core的原则有这样几点的取舍:

对于端侧网卡芯片, 例如一个800Gbps的网卡引入PCIe Switch后, 周边的一些Serdes(例如 48 x PCIe Lane和 800G的Ethernet Serdes) 决定了这个芯片的Die面积, 增加一些Smart的功能并不会显著的增加Die.

对于交换机芯片, ScaleOut芯片通常需要有一定深度的Buffer, 多端口读写的Traffic Manager的 MMU 实现难度确实非常高. 但是考虑到一个十万卡/几十万卡甚至百万卡规模的部署. 对于交换芯片更多的是需要它有更多的Radix,更高的吞吐. 例如一颗102.4Tbps的交换芯片期望它能提供1024个112G Serdes或者512个224G Serdes, 更进一步未来拓展到200Tbps的交换芯片.

在这两种约束下, 把多路径负载均衡 / 可靠传输 / 拥塞控制等功能放置在端侧构建Smart Edge Dumb Core是一个必然的选择.

## 2. 谈谈ScaleUP, Dumb Edge Dumb Core ?

对于ScaleUP是否还需要遵循Smart Edge, Dumb Core的原则?

### 2.1 ScaleUP的节点数

首先是对于ScaleUP的规模, 从MoE专家的数量来看以及模型架构中能够使用TP并行的规模来看, ScaleUP的规模在至多256颗GPU节点的规模即可. 特别是在推理场景中考虑故障域隔离的因素和根据workload 弹性伸缩部署的需求, 通常还会进一步压低部署规模.

例如用一个256卡的集群, 对于一个MaaS平台每次扩容256卡, 新扩容的这个超节点可能负载只有5%不到, 使得整个集群的负载例如低于70%. 这个时候, 如果我们全部按照32卡的规模逐步扩容, 虽然32卡的场景下累计8个超节点的满载性能可能只达到256卡超节点的80~90%, 但是弹性的按需扩容会有更大的成本收益.

另一方面, 数千卡的超节点没有意义. 并且为了支持超大规模组网, 交换网需要2层会带来更多的负载均衡和可靠性问题, 更倾向于选择单层交换机组网的结构. 同理BRCM的GM也是在强调单层组网.

![图片](assets/feab50ce9eff.png)

结论
ScaleUP节点数64 / 128  / 256 卡即可.

### 2.2 从芯片的视角来看ScaleUP传输

从芯片的视角来看, 由于ScaleUP节点数的需求决定了其交换机部署时的Radix, 以224G Serdes互连那么交换芯片在Radix=256时的容量51.2Tbps即可. 另一方面我们需要关注一下GPU侧, ScaleUP总线在1.8TB/s时, 单向为900GB/s即 7.2Tbps, 也就是说端侧和交换侧的带宽比为1:7左右.

我们需要在这个场景下做出取舍, 考虑到GPU Die本身的功耗/散热的问题, 以及ScaleUP Die面积占用的问题, 是不是把一些功耗的Budget转移到交换机侧会更好? 这样整体的散热难度也更小一些, 功耗管理也更容易一些? 同时对于端侧由于功能更简单还能置换出来更多的面积来进一步提升算力. 似乎在这种观点下, 应该选择**Dumb Edge, Smart Core**.

但是对于Core Switch侧如何处理, 芯片架构如何设计, Transport层如何构建?

我觉得我在2024年初给BRCM建议的时候没有说清楚,  我只是告诉他们可以去参考Nvidia在2023年的一篇论文《FinePack: Transparently Improving the Efficiency of Fine-Grained Transfers in Multi-GPU Systems》论文提出的方法. Pack多个LD/ST到一个packet里面.

![图片](assets/9cd98629dc53.png)

但是在整个过程中没有详细阐述多个LD/ST跨越交换机时该如何dispatch. 因此SUE的Transport层在Pack多个LD/ST时并没有很好的处理.

实际上这里面隐含了另一个交换芯片架构的问题, 选择哪种Buffer架构的交换芯片. 很多网络域的兄弟对UALink**的诟病还来自于UALink需要在交换机内拆分DL Flit并进行查表进行多个TL FLit转发. 传统的观念认为交换机无法在51.2Tbps/102.4Tpbs这样高的带宽下维持线速转发. 事实上这方面的约束来自于交换机本身的微架构上, 如果采用Shared Buffer Switch, TM的MMU设计要满足到51.2T/102.4T这样的速率打满LineRate的PPS是非常困难的. 但是采用PortBased Buffer的设计并且配合ScaleUP域简单的Accelrator ID lookup是非常容易的.具体来说:
Shared Buffer Switch
通常它会将收到的数据包存入一个集中式的缓冲区, 然后取出Packet Header,解析并查表然后通知出接口转发. 如果需要对Payload进行解析和处理, 对于这块Buffer而言, 需要更高的操作速率, 因为它是一个多端口的SRAM构成的MMU, 多端口的SRAM带来的开销是非常大的, 同时解析和多次查表也需要更高的访问速率. 因此工业界的认知是针对这类交换机处理Payload会对整块芯片的吞吐带来极大的影响.
PortBased Buffer Switch
为了避免了大规模多端口SRAM集中式buffer的占用和带宽/操作速率的开销, 可以使用基于每个端口或者每几个端口构成的PortLogic的Buffer结构, 并且通过Cut-Through的方式进行转发降低了延迟.

实际上我们可以看到Tomhawk Ultra在64B报文可以达到51.2Tbps LineRate

![图片](assets/7966dfbb59d4.png)

延迟也进一步降低到< 250ns, 并且整个系统的抖动是相对较小的

![图片](assets/9a454d905f5f.png)

从芯片的结构来看, 它是一个扁长的Die

![图片](assets/b815037e7cac.png)

实际上它的结构整体有32个独立的PortLogic构成,每个PortLogic上有独立的buffer, 用于缓冲和链路层重传(LLR).

![图片](assets/2dfba80908f0.png)

实际上对于一个UALink的DL FLIT, 通过CutThrough的方式, 一边按照64B TL FLIT逐个处理Payload中的地址lookup, 然后dispatch到Egress的PortLogic, 而Egress PortLogic累计收到640B后就可以打包成一个DL FLIT发送给GPU.

实际上交换芯片来看, 增加这些处理的代价并不大, 并且PortLogic上的Buffer还可以复用, 整体要支持UALink那样的解析TL FLIT转发对于芯片面积增加是不显著的, 同时带宽演进上也是有保证的.

### 2.3 争议的焦点: Cut-Through

事实上正如前一节所讲, Cut-Through和Store-forward的取舍, 更多的是在芯片架构维度上选择Shared Buffer 还是 PortBased Buffer. UALink算不算Cut-Through?

其实对于很多做网络的同学, 刚开始看到UALink的标准时, 都会吐槽怎么把交换机搞的这么复杂, 又要拆包又要组包的, 对交换机的复杂度和延迟都有很大的担忧. 确实要想把一个UALink的Switch做好难度也挺高的. 另一方面即便是要把普通的Ethernet Switch做到51.2Tbps / 102.4Tbps也是难度巨大的, 特别是集中的Traffic Manager的 MMU. 某个厂商尝试着使用两块 ingress TM来处理遇到了不少问题, 另一方面国内某厂采用多die的架构也遇到了一些负载均衡的问题挺难处理的.

实际上要做到51.2Tbps / 102.4Tbps 交换芯片的bar无论是普通的UALink Switch或者是以太网Switch都是非常高的, 特别是要做成单Die结构的时候.

另一方面在很多高端的路由器上, 特别是基于Cell-based forwarding的框式路由器上, 实际上转发也是和UALink类似的, 切包成Cell, 然后带上Self-Routing Header转发到Egress再输出.

### 2.4 延迟的影响

UALink和SUE之间的口水战还有一个维度聚焦在延迟上, 相差100ns左右的延迟到底重不重要? 这个事情取决于算力芯片的微架构, 是像很多DSA那样的一整片200MB的SRAM? 还是像GPGPU那样在SM内部只有256KB的SRAM? 是否允许GPU A SRAM 到 GPU B SRAM的直接LD/ST, 还是一定需要GPU A SRAM 到 GPU B DRAM的 再 GPU B LD到SRAM的路径?

对于算力芯片而言, 增加100ns延迟, 以ScaleUP 1.8TB/s计算则需要增加100KB的SRAM. 对于DSA架构一整片200MB SRAM或许没有什么影响, 但是对于那些GPGPU架构的处理器, 可能还需要评估一下. 或许影响也不大?

其实我更看重Fabric抖动的影响, 对于SUE,由于需要在端侧根据不同的目的地GPU构建队列, 然后通过调度轮询的方式pack成一个数据包传输, 因此在穿越整个交换机的过程中, 报文的长度是可变的, 这样就会带来一些抖动.

例如我们按照1500B计算峰值抖动, 在224G Serdes上带来的额外的抖动为56ns, 而按照UALink 64B FLIT来看, 抖动仅2~3ns. 另外还要引入交换机Fabric的一些调度引起的抖动, 然后在incast的情况下, 可能还有更多的抖动增加.

对于UALink, 实际上incast情况下, 由于内部按照64B FLIT在转发, 接收端口可以在一个DL FLIT中封装来自多个GPU的64B TL FLIT, 整体的抖动会相对于SUE的方式小很多.

### 2.5 在网计算: 另一个Smart Core?

虽然NVidia在NVLink Switch中也增加了NVLS的能力, 但是很显然对于非常重要的MoE Dispatch/Combine是无法支持的. 如果要支持在网计算, 那么MXFP4/NVFP4/FP8等很多block based量化的格式需要支持的, 同时考虑加法的精度问题, 可能还是需要FP32的Accumulator, 这些功能实现上很大程度的还会进一步加大交换机实现的复杂度, 具体的E2E业务收益我还是持怀疑态度的.

## 3. ESUN / SUE / UALink

当然有一点我觉得没必要那么对立的把Ethernet ScaleUP 和 UALink来分析. 从传输层来看与UALink对标的是SUE, 我们需要评估同样在1.8TB/s ScaleUP下, 两者之间在加速器侧的Die面积大小对比, 功耗对比. 以及在交换机的实现复杂度上充分的进行取舍.

至于物理链路层, 无论是SUE还是UALink或者是NVLink, 未来能够统一到ESUN也不是不行. UALink over Ethernet 或者 IFoE.

另一方面在整个生态中, 最近还有一些CXL 4.0做ScaleUP的声音, 实质的问题是我们要回答主流的GPU / CPU厂商谁会支持这个方案? 当然UALink也有一些生态上的缺失, 例如AMD自己的GPU不争气怎么办?

时代的车轮在总线的更替上总会碾压很多次, 例如当年从ISA到PCI, 再到AGP, 再到PCIe, 哪个不是为期十年的更替. PCIe本身已经老了不堪重负了, 虽然还有国内很多GPU厂商希望能有一个PCIe like的ScaleUP总线, 但早已积重难返了...

一个好的生态不是大厂之间的打打杀杀, 而是利益均沾的精诚合作. 附另外两篇相关的文章:

[《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect)

[《谈谈ESUN, SUE和UALink》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496512&idx=2&sn=0c10cef05fb1cc4e175f326d62b266e3&scene=21#wechat_redirect)