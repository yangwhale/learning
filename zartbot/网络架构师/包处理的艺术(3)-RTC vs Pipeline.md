# 包处理的艺术(3)-RTC vs Pipeline

> 作者: zartbot  
> 日期: 2021年2月13日 07:53  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485339&idx=1&sn=62439ad879f26f18c4434e1e51a0bdc3&chksm=f9961759cee19e4f776bf5719ce4c0635b09bbbebdcf7e37a682972e836d6c22e23bd64e2982#rd

---

题记：本文主要介绍一些State-of-the-Art的数据面处理方式，例如思科的Silicon One、LightSpeed、QuantumFlow Processor、UADP，同时也会对比介绍一些其它公司的，例如BRCM、Innovium，当然也不会错过智能网卡芯片，Pensando、Fugible. 当然篇幅有限，关于这些演进的历史会放入包处理的艺术<A1>一类的附录中作为补充，而对于多核RTC的调度、DPDK、VPP、BPF或者Pipeline等处理的详细细节可能会在包处理的艺术(4)中出现，因为这个话题涉及到片上网络、AI加速芯片等一系列并行计算的话题。

**1. 架构概述**

计算机网络的出现自然是晚于计算机本身的，因此世界上的第一台网络路由器的名字就是Advanced Gateway Server(AGS)

![图片](assets/162696e7ad8b.jpg)

它是Motorola 68000系列CPU构建的，因此整个转发平面只能从包进入到查询路由到最终包发出整个流程都在一个处理器里面完成，所以有了RTC的定义，即Run-To-completion。

![图片](assets/f7a868c7588a.png)

上图为该机的CPU板，整机包处理能力大概在7,000~14,000pps。接下来的30年时间神奇的将这个数字增加了一百万倍，但是报文处理的本质骨架还是那么些活：**接收->查表->修改报文->缓冲->转发**, 例如Nick McKeown教授在2001年讲课的PPT：

![图片](assets/e57e93e984dd.png)

而创造这三十年伟大工程奇迹的后面有着大量伟大的成就，通常的处理套路下面这本书总结的很好

正如这本书中所描述的计算机体系结构的8个伟大想法

**1. 面向摩尔定律的设计**, 架构师必须在设计之初预测其设计完成时的工艺水平。而以太网速度最近几年的爆炸式增长已经超越了摩尔定律，新的器件工艺的诞生(例如：Serdes/RAM/Optics/Packaging)也为处理能力提升奠定了基础。

**2. 使用抽象简化设计**，通常的做法是隐藏低层细节以提供给高层一个更简单的模型，网络的分层是一种做法，另一种做法就是包处理艺术<2>中讲到的**同构**，例如用一堆CPU构造成一个大路由器(多线卡分布式转发），然后一系列大路由器构造成一个集群(CRS/JNPR-TX/NE5000)，然后随着硅工艺的进步，又把多个CPU核心弄到一个硅芯片上，从而诞生了片上网络，更进一步直接把多个路由器放置在一个芯片上(Cisco Silicon One)。

**3.加速经常性事件，**例如早期对查表算法的优化，到后期offload或者出现FastPath的做法都是这种思维的体现。

**4.通过并行提高性能，**典型的做法就是多核CPU执行并行转发。

**5.通过流水线提高性能，**这也就是本文重点，流水线和RTC的区别。

**6.通过预测提高性能，**DPDK中有大量的分支预测的帮助函数，您会经常的看到likely/unlikely的处理

**7.储存层次，**cacheline对齐，zero copy等处理方式，或者以VPP为代表的优化I-cache的处理方式，以及核上对于Cache size的调优(例如思科针对企业多业务路由器QuantumFlow 和针对运营商的nxPower的区别)

**8.通过冗余提高可靠性，**单机双引擎，双机双引擎各种热备冷备技术都在用，还有SDWAN中的packet duplication，以及Tofu-D中针对误码率高了以后的双链路同时发送bypass RS-FEC都是这类思维的代表。

**1.1 RTC**

现代的RTC网络处理器架构大多如下所示：

![图片](assets/9855107b1dd6.png)

报文进入后通常会有一个调度分发器(Packet Distributor)，然后调度到相应的报文处理引擎(Packet Processing Engine,PPE)处理，调度算法可以有基于包的，也有基于流的，具体在包处理的艺术(4)中分析。

而PPE的结构基本上都是标准的冯诺依曼架构处理器，有相应的指令内存，数据内存，寄存器，ALU，程序计数器(PC)等结构

![图片](assets/20d6214fae35.png)

通常在RTC的处理结构中，我们可以灵活的使用C等多种高级语言编程，然后编译成一系列指令执行，有更高的灵活度，而每个报文在处理器内部的处理时间会有不同，例如很多简单的软件功能可能几个条指令就处理完，然后就可以转发。这也是Silicon One功耗低的本质原因，后面我们会详细介绍。

**1.2 Pipeline**

一个很形象的比喻就是洗衣服可以分为装入洗衣机、洗衣、叠衣、装柜，当您有大量的衣服要洗时，通常的做法如下：

![图片](assets/b25458c34c12.png)

这就是流水线的思路，我们可以把一个处理器的任务按照不同的软件特性分解，例如MAC查找，IPv4查找，ACL查询、QoS等等，然后每一级做成专用的硬件，当第一个报文查完MAC以后就可以脱离第一级进入到IPv4查询，而MAC查找引擎就可以处理第二个报文了，最早期的流水线结构是通过多个ASIC实现的，典型的就是以Juniper M系列为代表：其转发原理如下所示：

![图片](assets/a0b4399ecd51.png)

PIC卡接收到报文后,** PIC I/O ASIC**对报文进行CRC校验, 如果校验失败则丢弃报文并通知JUNOS的相关接口计数器更改. 如果校验成功则进行Layer2解封装, 将Layer 2 PDU发送到FPC上的**I/O Manager ASIC**.

**I/O Manager ASIC**检测报文的协议类型, 然后在包标志信息内设置一个可能被用于不同服务的标志.然后将报文分割为64-byte的Cell(被称为jcell). 其主要原因是使得内存使用更加高效, 同时定长交换可以优化转发时延.并将每个jcell传递给位于SCB(or SFM)上的分布式**Buffer Manager ASIC.**

**Buffer Manager ASIC**将jcell以轮询(根据PFE进行轮询, 将不同的jcell发送到不同的PFE上的共享缓存中)的方式发送到分布式的共享内存中.

**Buffer Manager ASIC**同时根据jcell中的相关flags提取信息, 例如对单播IPv4报文提取源目的地址,输入接口, UDP/TCP端口等参数, 对于MPLS查看MPLS标签等. 收集完这些信息后生成一个key cell, 并将其转发到**Internet Processor II ASIC**

**Internet Processor II ASIC**对报文进行查询, 并将完成查询的转发决策整理成一个result cell返回给**Buffer Manager ASIC**

此后**Buffer Manager ASIC**将result cell中的信息进行解析, 并通告出口PFE的**I/O Manager ASIC**, 如果是组播流量则通告所有的出口**I/O Manager ASIC**

出口**I/O Manager ASIC** 通过分布式的**Buffer Manager ASIC**从分布式共享内存中读取jcell. 并管理包队列和一些QoS特性等. 最后I/O管理器将这些jcell重组为packet, 并将帧结构转发到出口**PIC I/O ASIC**

出口**PIC I/O ASIC**根据出站媒体介质进行编码, 例如POS编码或者HDLC组帧等操作, 最后从端口上发送出去. 离开路由器进入下一跳.

当然后期随着硅工艺的发展逐渐整合到单一的ASIC中，例如Juniper MX就逐渐将多个芯片整合到一个芯片中：

![图片](assets/5637ef327803.jpg)

最终单芯片多流水线架构就出现了：

![图片](assets/5e4ae6cf157d.png)

**2. RTC vs pipeline**

**2.1 RTC和Pipeline的交替发展**

很多人会在技术的选择中走极端，总要比较RTC和Pipeline谁好，但是很少会站在不同的历史节点上看待这个问题。通常我们对于流水线的定义为傻和快，傻是缺少灵活性的意思，当芯片设计好它支持的处理能力就固定了。而RTC架构通常是灵活但相对较慢的代名词，后期您可以通过软件代码的方式加入新的功能，这是RTC架构的优势。

伴随着第一次互联网泡沫，传统的RTC架构的路由器如7500一类的已经无法满足客户的需求了，自然用pipeline的思想，而且还是多颗ASIC在转发平面上构建Pipeline，但是又随着协议的快速发展，传统的线卡跟不上新的业务需求，通常一颗ASIC开发要花费2~5年的时间，一种做法就是逐个的更换ASIC，针对某些功能升级。过去的数年，你可以看到思科的产品线在这两者之间跳跃融合，所以技术上并没有谁最佳，只是在不同的时间点做出恰逢其时的选择罢了：

![图片](assets/7de9b485e044.png)

从AGS+开始便是RTC，然后到7500在每个线卡上都有RTC的MIPS处理器，然后GSR采用多ASIC的处理方式，而后CRS伴随着片上网络的发展和当年基于Treebitmap路由查询算法的诞生又走到了多核PPE执行RTC的路，并且演进了十年。而另一方面ASR9000早期有基于QFP的完全RTC的线卡，但是相对于城域以太网较为简单的包转发特性需求，定制化的Pipeline处理器Ezchip胜出，直到后期因为业务的复杂性又回到了使用nxPower技术的lightspeed RTC网络处理器。同样核心路由器上使用了一段时间的Pipeline的Jericho以后，又回到了Silicon One的RTC架构。

您只需要记住任何技术在其历史的时间点都有它自己合适的选择，并没有优劣，甚至相互还有融合。

**2.1.1 RTC in pipeline**

当然还有一种做法是把一部分的可执行微码的微核嵌入到Pipeline中，例如Cisco的PXF

![图片](assets/b3089c79e843.png)

当我们看到一块芯片有这么多核时，你应该下意识的想想，它做了什么取舍？和通用处理器相比，这类微核通常是专用指令集，编程上受限制较多大量高级语言用不上，同时片上指令内存(缓存)空间有限(I-Cache),所以可执行的指令相对简单，复杂的功能难以实现。

所以后期伴随着难写的微码和网络协议的蓬勃发展及内存技术的发展，又逐渐的被完全支持C代码编程的RTC网络处理器SPP(CRS-1)和QFP(ASR1K)替代了。

**2.1.2 Pipeline in RTC**

我们通常可以把报文的处理通过C语言代码编译成一系列指令执行，而执行的过程通常是取指、译码、执行、内存访问、回写这几个步骤：

![图片](assets/38d2cfe9e2e6.png)

从微观的角度来看，如果都顺序执行是不是太浪费时间了，第一个指令取完就可以在译码的阶段把下一个指令取出，这就是在RTC的内部融合pipeline的技术，如下图所示：

![图片](assets/9d7c72a6a369.png)

**2.2 流水线空转问题**

我们来看一个很常见的问题，代码里对包处理有If-then-else这些分支逻辑怎么办？当我们在执行beq指令时，并不知道比较结果时相等呢，还是不相等，也就是说下一步我要预读取的指令和上一步有关联。

![图片](assets/f3023f84b83a.png)

简单的做法就是随便乱猜一个预先读，万一对了不就省事了么，但是万一错了呢？大不了下一步清了，再读一个，这样流水线里就会有空泡。所以分支预测就是这么个思路，尽力猜对，所以很多代码里都有likely/unlikely的处理来优化分支预测。

另一个流水线空转的问题在于当一个固定的流水线写死了以后，我不要运行某些功能却又必须要走这一级流水线，如下图所示

![图片](assets/d692d4b84054.png)

当入方向不需要ACL的时候，固化的流水线也必然要走这一级没法跳过，自然整机功率就上去了。而RTC做完了就可以释放报文了，相对处理功耗就要小很多，这也是Cisco Silicon One低功耗的原因之一。

**2.3 流水线固化问题：串接业务板卡**

我们来看另一个很常见的问题，流水线和ASIC相对固定的时代，为了灵活的支持新的业务，自然就出现了所谓的业务板卡，也就是ServiceBlade的概念。例如思科7600上的各种业务板卡，最具代表性的就是ACE和增强型的SAMI卡，ACE内置了两块IXP处理器，而SAMI还额外通过子卡添加了6个RTC的网络处理。

![图片](assets/99bbd8e8db8f.jpg)

Juniper也同样有自己的Service PFE，在固化的流水线中插入了一些灵活度的业务。

![图片](assets/880e8625c35b.jpg)

**3. 可编程流水线的发展**

既然固化的流水线有傻的缺点和快的优点，如下图所示：

![图片](assets/7b41c51b5953.png)

于是就有人做起了脑白金的生意：）通常一个ASIC的开发周期为2~5年，那么如何让芯片变得更聪明呢？不因为功能而升级，而只是在三五年后因为性能不够了再换掉，这样对自己和客户都是好事，既有投资得到了充分的发挥。

![图片](assets/832d67873af2.png)

我现在还在用一个2008年研发的第二代QuantumFlow Processor，十多年过去了，还依旧有Segment Routing、DPI、防火墙、IPSec、VXLAN、SDWAN的功能，这就是可编程的好处，任何硬件都不需要更换，添加新的功能也没有性能的下降。当然这是一个RTC的处理器，我们后面一点的章节和Silicon One一起讲，先来看看交换芯片的可编程演进

**3.1 UADP**

UADP项目主要是应对园区网的快速发展，带宽需求基本上满足了，而更多的是一些灵活的策略、终端漫游和寻址而产生的Overlay、TrustSec、无线有线融合，也就是后面出现的软件定义的接入SDA解决方案。

![图片](assets/80c15fb37a33.png)

UADP的设计就是将原有的固化包解析器和各种固化的查表引擎及报文重写引擎改为可编程的结构，这样就可以灵活的在ASIC上处理了，这样有个好处，同一个ASIC可以为不同的业务分配灵活的表项：

![图片](assets/47a6505b6455.png)

UADP的流水线也相当长，Ingress 17级，Egress 8级，每级都可以同时执行2次查询：

![图片](assets/dc79b4db686a.png)

当然还有一个常见的做法诞生了，也就是Recirculation的处理，以VXLAN报文为例，第一次通过的时候解开隧道头，最后Rewrite阶段重新把内部的内容导入流水线完成报文的处理。

![图片](assets/4d7ae26c9537.png)

不过这并不是一件高效的事情，虽然在已经固化的ASIC上实现了更长的流水线处理结构和一些包处理结构的复用，但是Recirculation也导致了整机吞吐的下降，毕竟设计这个ASIC的时候还没VXLAN呢，至于后来Nexus9000刚出来的时候也有特殊的芯片来处理。

**3.2 Barefoot Torfino**

说到可编程交换芯片不得不提到P4. 网工通常连Python代码写起来都很吃力，更不要说写微码了，好多半道出家的计算机体系结构都没学过，所以P4这样的DSL就诞生了，既然UADP验证了可编程ASIC的成功，那么就再进一步在工具链和底层结构上各自向中间靠拢一步，这就是P4.

![图片](assets/4defc72a66df.png)

和UADP一样，本质上Tofino在初始状态是一个完全协议无关的流水线架构，可以根据您自己的需求定义各种报文处理行为：

![图片](assets/ebc9683dbcb5.png)

用通用处理器写过Parser的都知道，在每次解析报文的时候通常需要消耗大量的指令才能将需要的字段放入到寄存器中，而Barefoot的最大创新就在这里，Parser是一个TCAM和SRAM组合的结构，通过片上的SRAM作为寄存器指定输入的Shift Offset，然后TCAM则用来Match需要的字段，一次性的将结果丢入到SRAM中，最后产生对齐的8b、16b、32b 等等不同长度的值供后续的ALU处理，Barefoot将这样解析好的字段定义为PHV(Packet Header Vector)，一个很形象的说法就是From packet to register

![图片](assets/e9869fb16580.png)

稍微补充一下TCAM，传统的CAM是精确匹配每个bit位是否要match或者不match，所以通常用于做一些需要精确匹配(Exactly Match)的场景，例如MAC地址匹配，而TCAM的做法其实就是有些bit位加上了一个Don't Care的处理，对比如下：

![图片](assets/836acd469da7.png)

所以这也是TCAM为啥叫Ternary Content的原因，其实就是每个bit位有三种状态， 0，1，Don't care，这样不就很容易去做一些匹配了么，特别是路由器那样的Longest Match

![图片](assets/41d575406ff5.jpg)

所以您会看到这东西还是SRAM构建的，那么SRAM的结构通常是由六个晶体管组成，如下所示：

![图片](assets/cb00fa6a2b89.png)

然后随着SRAM技术的发展，基本上片上SRAM的存储密度每隔2年就缩减一半，因此容量快速增加，所以很多事情都是在一定的时期才能诞生的，这也是为什么Barefoot的时间点可以做这样的事情，而更早的UADP做不了的原因。具体关于SRAM的发展及功耗稳定性的限制可以参考下面这本书的第二章

当然另一个问题是SRAM容量毕竟有限，精确匹配的场景消耗的CAM数量很多，那么还有别的什么优化方式呢，自然就是Hash表查询了：

![图片](assets/5abf277568d8.png)

写这一小段的目的是让网工们了解芯片内部的器件，很多东西并不是简单的一个黑盒，接下来回到正题，相对于UADP做多级的Parser，末级Rewrite报文和产生Action，工艺的进步使得Torfino可以在每一级执行操作，执行操作的ALU相对简单一些，而且比起标准的RTC处理器而言，它没有控制逻辑，没有大量的临时寄存器，没有PC，单级Action只能做简单的一些指令操作。

![图片](assets/4cd70833de39.png)

然后Barefoot就把这一系列MAU并行起来构成了单级同时匹配多个字段的引擎：

![图片](assets/a0ed2f6bda4c.png)

同时又把多个这样组合好的引擎构建出一个大型的流水线结构，注意到中间的MAU 0~ MAU n就是这样的流水线结构。

![图片](assets/e02e1d7592c1.png)

但是问题来了，流水线怎么调度呢？前面讲过流水线空转的问题，也就是说当上一级和下一级的Action没有依赖的时候，工作的很好(下左图)，当操作相互依赖时延迟就增加了(中图)，而需要做一些分支处理的操作时耗费的时间就更长了，如下右图：

![图片](assets/bb56e2194949.png)

当然Barefoot也有一些Table Prediction的处理方式，如下图所示:

![图片](assets/fa41932d7cfd.png)

但是这样的处理在很多复杂业务下会存在明显的pipeline消耗过长需要Recirculation的过程，这也是Silicon One需要将它换为RTC的原因，当然我会在包处理的艺术(5)里讲一些报文编码的处理技巧如何避免在协议设计的时候出现if-then-else这样的处理，这也是G-SRv6编码的一个硬伤，很多协议的编码需要同时考虑如何让硬件做的舒服才是最重要的。

**另一方面你需要注意的是，这类芯片通常将片上的存储用于了灵活的报文处理，因此并没有预留大量的空间来承载转发表，因此通常这样的芯片是以交换机的形式存在，因为简单的Exactly Match好做，但没有大量的片上TCAM资源无法支撑数百万级路由的LPM(Longest Prefix Match)。**

![图片](assets/56b405b5956b.png)

**后续Jericho和Silicon One的介绍中你就会明白另一种取舍了。**

**4. RTC众核处理**

固定流水线的架构会面临一些问题，特别是大量路由表项和包缓冲需要在紧缺的片上资源里扣除一大块，因此片上资源非常紧张的时候，又有了另外一些想法。

在2000年左右的时候，Will Eatherton有一篇论文<Tree bitmap: hardware/software IP lookups with incremental updates>,然后通过这样的方法避免了IP路由查询受限于TCAM的高功耗,但是通常需要三次DRAM的访问，总体来说在那个年代还是能适应报文处理的速度的，毕竟要做的只是一块40Gbps的网络处理器，这样的思路就产生了一个叫PLU的结构。

Cisco Packet Processor采用Tensilica的指令集构建的一个非常精简的微核作为Packet Processing Engine，完全支持可编程的结构，当然这个系列一开始出来就针对运营商和企业网市场做了区分，针对运营商的为CRS-1的Silicon Packet Processor，针对企业的为QuantumFlow Processor，当然CRS-3的时候又迭代出来QuantumFlow Array，然后是企业网的QFP-2，紧接着又是CRS-X的nxPower，到后面QFP-3和ASR9000的Lightspeed。有些tick-tock的味道吧。

以QFP为例说一下报文的处理方式，通常大量的PPE会通过一个片上网络连接各种资源，例如前面所述的Tree bitmap查找算法的PLU，然后TCAM主要用于ACL匹配，而DRAM的访问也挂在这个片上网络(Resource Interconnect)。

![图片](assets/d827c7ac8bd9.png)

同一时期片上多核的架构代表还有Sun Sparc的Niagara处理器(正好前几天在实验室里找到一块）

![图片](assets/19cd3f27e60d.png)

相对于通用多核处理器，CPP系列做出了一个取舍，那就是不需要Cache一致性的协议，片内结构如下所示:

![图片](assets/3be7c8648326.png)

你会看到片内每个核有4个线程，每个线程都有自己独立的L1D-Cache，但是L1 I-Cache在四个线程内是共享的，然后片上网络是一个二维的Mesh，L2-Cache比较有意思，只有L2 I-Cache,没有在片上放置L2D Cache(后面简化为L2D$). 这样的好处就是可以降低功耗，并且提升核的数量。所以你会看到处理器的设计上处处都是取舍。

报文的处理采用Run-To-Compile（RTC)的结构，即即便功能再多，一个处理器处理完一个包的所有功能， 然后编程上以C代码为主，内存操作也非常灵活，大概的伪码如下:

```
while (1){    block for wakeup    Fetch a new packet descriptor    Process this packet    {       feature-1... feature-N    }    Dispatch the packet to QFP traffic Manager Queue    Notify traffic manager }
```

早期由于工艺的限制和片上网络的结构，因此采用了两块芯片构成，一块包含PPE的叫Popeye，另一块是做Traffic manager的spinach，后期在QFP-2中随着集成度的提高整合在了一起。

![图片](assets/a06b95c3f3c9.jpg)

后续的 QuantumFlow Array、nXpower、LightSpeed、QFP-3基本上都是这样一个框架，只是在片上网络和核心数上做了提升，QFP-3已经到了惊人的224核心896线程，LightSpeed系列更多。

![图片](assets/47f3adab62ab.png)

而企业网路由芯片和运营商路由芯片的取舍主要是在Cache上，运营商的feature以Stateless转发为主，更多的需要考虑pps，QoS也相对没有企业网那么复杂，因此片上packet buffer和cache相对较少，而企业网通常有复杂的软件功能需求，在QFP设计的时候还是2004年，发布在2008年，谁也没想到后面这个平台能够支持SDA支持SDWAN还能同时桥接ACI并执行大量的复杂的软件功能，一个路由器用十多年我自己也没想到，直到现在我还是在用一个2008年购买的ASR1004机框做一些研究。

![图片](assets/1e0426b06229.jpg)

当然没有任何技术是完美的，这样的结构也带来了一些问题，功耗相对较大，pps数很低，带宽也相对较小，但是将它放置于边缘路由器或者企业网路由器的位置是完全满足需求的，毕竟企业广域网200G的处理能力已经足够了，而多线卡机框的ASR9900也能满足大多数运营商PE路由器的需求。

![图片](assets/14ea1b065e64.png)

关于这一章还有一些更详细的东西，篇幅有限将会放到第四篇中详细叙述，包括如何调度多核，是基于流还是基于包，cache一致性和内存一致性，无锁的设计等等。

**5. 取舍有道**

开始新的一章前对于RTC和Pipeline做一个总结，如下图所示：

![图片](assets/53c5de15ab39.png)

最左边是思科的路由处理芯片系列产品线，芯片上的资源主要用于包转发引起PPE和相应的Cache，同时也分配了大量的资源用于转发表FIB查询，当然路由器相对于交换机通常还有大量的广域网和局域网侧带宽不匹配的情况，因此需要Buffer去吸收来自于LAN侧的报文burst以及QoS在广域网侧做多队列调度，这些都是我们在设计一颗广域网路由器芯片时要考虑的问题，当然更细节的设计是企业网比运营商可能因为一些Stateful的软件特性还需要更多的内存访问优化和更大的cache设计。这样的缺点也是很明显的，就是功耗特别大同时也不够快，很多feature可能需要通过片上网络多次访问memory也会带来一些瓶颈。

中间这个图为固定流水线的交换机芯片，通常功能很死但是特别快，所有的资源都用于报文查询的流水线，当网络协议栈创新没那么快的时候它有它的价值，特别是配合智能网卡构建SmartEdge&DummyCore的架构时，这样的芯片很适合作为DummyCore使用。

最右边为Cisco UADP或者Barefoot Torfino一类的可编程交换机芯片，它把片上的资源再一次划分给了一系列灵活的Match-Action引擎，但是相对来说速度就降下来了一些。

您可以注意到在中图和右图中，QoS、Buffer、FIB这些都是交换芯片常常不需要考虑的，因为在LAN侧传输延迟相对较低，带宽上行通常大于下行并不需要太多的QoS设计，转发表规模也没那么大(当然云数据中心做ToR overlay时还是有很多需求的，所以后面Host overlay的诞生也有其内在的原因）。

**5.1 FastPath&SlowPath**

有一种做法就是把固定流水线的做FastPath，而灵活的业务处理丢到SlowPath上让RTC的处理器来做，例如阿里的XGW FastPath采用了Barefoot，而SlowPath采用了Intel Xeon处理器做RTC

![图片](assets/a8042eb3a4b4.jpg)

FastPath/SlowPath的架构也有一些内在的限制，例如流表的更新速度，FastPath的TCAM容量和SlowPath不匹配的问题。以前我测过某厂的类似架构的路由器，号称100Gbps的吞吐就由于这个不匹配的问题在源目的端口随机的测试下构建一个64K*64K的fullmesh 测试流连1Gbps都打不到，直接被甲方废标。

FastPath和SlowPath的处理通常会采用统计学模型，把经常性的5%~20%的条目放入到FastPath处理，而剩下的放入RTC。Google以前有一篇论文讲到5%的ACL可以匹配接近95%的流量也就是这个原因。

另一个问题是FastPath+SlowPath的架构并不适合很多安全性的feature采用，因为这些通常需要每包都有状态的进行复杂处理。

**5.2 Jericho**

另一种做法就是平衡流水线，相对于Barefoot不那么的可编程，又相对于固定交换机提升路由表的容量和QoS、Buffer。Jericho便是这样的处理方式，早期使用片外TCAM、DRAM和片内配合的方式，然后报文处理采用了ElasticPipe的处理方式。

![图片](assets/aa8d38ed1084.png)

通常的做法就是把Memory的容量和带宽加大，然后每一级都可以灵活的进行查找，但是每一级的查找引擎相对又固定一些，不像Torfino那样的MAU需要大量的SRAM、TCAM。这样就在功耗和芯片尺寸上做了节省给片内放置大buffer和相对较大的路由表条目，可编程能力也有一些，但是没有做到完全的Protocol Independent。总结一下它的性能取舍，如下图所示：

![图片](assets/6b84fa5c3d24.png)

但是这样做也有一个问题，您可以注意到片上固化的一些流水线很有可能会在一些核心设备上无法跑到，这样就导致了它功耗相对较高。然后FIB也没有到满足客户需求的地步，现在BGP路由条目数都一百万条了，而它的FIB在内部定义为两种LPM和LEM，和外部的TCAM。

![图片](assets/62db5b1adccc.png)

当然它采用混合查表的机制

![图片](assets/a089430701bc.png)

默认Profile通告Internet BGP条目直接可以把LPM搞爆掉，如下图所示：

![图片](assets/0903e035adf4.png)

所以我们也采用了一些优化的方式来分配LPM和LEM的使用：

![图片](assets/656f16923aa1.png)

它另一个处理方式就是借助了GDDR5作为片外的buffer使用，这种多芯片组合的方式又一次上演，但是问题就是功耗也上去了，Stats的功能还需要专门的FPGA去给它补全。

![图片](assets/1996ed9821f2.png)

随着封装技术的进步，Jericho2使用了HBM，其实下一代很多ASIC使用HBM都是必然趋势，例如各种GPU都是先行者，因为片上封装使得HBM提供了极大的带宽和容量，可以非常好的用作存放大规模的路由表或者Packet Buffer使用，功耗也更低

![图片](assets/f4b05543f708.png)

**5.3 Cisco Silicon One**

当我们意识到这一系列问题后，自然又有很多好玩的想法冒出来，那么又拿起RTC的微核如何？这样Jericho的pipeline的问题似乎也可以解决了，然后片上网络随着7nm的工艺似乎也可以得到很好的提升？这一次的取舍是什么呢？通常Silicon One作为核心路由器存在，并不需要太多的QoS和Buffer支持，这些留给边缘的多业务PE(ASR9900/ASR1000)来做，另一方面Stats的功能也不需要在核心做太多，这一块功能也可以节省一些，即使要也可以Sflow采样好了丢给一个FPGA在片外去做。

![图片](assets/62c023deee7a.png)

这些取舍做完以后，得到的就是快和低功耗同时也有很好的灵活性和可编程能力。从官方的说法来看列出了如下几点：

 消除了Off-Chip内存(DDR4/GDDR5/TCAM）

采用了片上的HBM做FIB和Buffer，但是更多的用于FIB

灵活的RTC引擎支持6B pps，想想第一台路由器AGS只有7000pps，30年一百万倍。

高级的数据结构和查表算法。这也是得益于HBM巨大的内存带宽和容量，然后查表的过程中如果还使用TBM的方式需要3次DRAM访问，在延迟和容量间取舍，自然新的查表算法有了容量肯定要消除延迟一次查完了呀~具体怎么一次查完的保密，但是就是简单粗暴就好。这个技术也用到了LightSpeed+上。

可扩展的灵活的多Slice结构，其实也是现代芯片技术发展的一个趋势chiplet，一个大的芯片可以有大量的chiplet构成，当然这里面也用到了同构的思想。

**5.3.1 Router on Chip Slices**

现代芯片基本上都采用多slice的结构，这样可以很容易的裁剪出需要的不同规格的产品，然后借鉴同构的思想，我可以将一个路由器设计成一个Slice，然后通过在片上构建Fabric，然后就可以把原来一个CRS-1的模块化多线卡路由器构建成到一个chip上，这就是Silicon One的做法：

![图片](assets/1c503d695a63.png)

然后我们继续迭代，如果拿Silicon One构建一个大的模块化多线卡路由器呢？只需要把一些Slice的微码结构从路由器模式转换为Fabric模式，同一个芯片就可以即做线卡又做Fabric

![图片](assets/ab2584b417ad.png)

所以您看到的结果就是整机功耗直线下降，因为原来一个大框要做的事情，一颗芯片就能办到，特别是8201这种单芯片的平台可以在400W的功耗下提供10Tbps的转发能力，其功耗小于一个咖啡壶。

![图片](assets/f0b6f64eff90.png)

这样一套单机性能相当于原来的CRS-X或者NCS6K的一个多框集群，例如一个CRS-X 2+2集群刚好也12.8Tbps，但功耗要25KW。

另一方面Fabric Mode的Silicon One直接就单芯片充当了原来的Fabric Chassis，而Linecard模式的Silicon One性能直接超过一台原来的模块化路由器，这样单台Cisco 8818构建一个260Tbps系统相对于NCS6000构建的8+2集群功耗也大大的降低。同时空间占用也从原来的几个机架降低到一个机架，一个机架降低到一个U。

**5.3.2 RTC**

**我并没有Cisco Silicon One的NDA，所以也无法访问内部的一些文档，但是当你熟知了计算机体系结构和芯片制造工艺后，你会很容易的做一些猜测，毕竟最终构建这些电路的基本元器件只有那些，猜对了别说我泄密，猜错了别打我~**

这一点上也是Silicon One不会说的太详细的地方，大概公开的资料如下，它相对于pipeline的架构有了更多的灵活性。

![图片](assets/05936975c192.png)

其实你仔细想想，每一块芯片设计的过程中取舍是什么？Memory以何种形态出现，要多大，放在片上还是片外，这些内存墙的因素直接决定了包处理的架构。客户在不同的场合下需要不同的内存访问模型，但是万幸的是无论怎么变你能够选择的内存访问方式就那么多：

![图片](assets/3756ac0a76ce.png)

在高端10Tbps芯片上片外的资源基本上就不用考虑了，更多的取舍在片上，Jericho是一种内存池化的方式构建LPM、LEM然后让不同级的Pipeline去访问。对比固定Pipeline和池化内存的Jericho做法，我们来看另一个话题Silicon One直接大大方方的说我也支持P4：

![图片](assets/1958c44233c3.png)

那么架构上自然你需要去参考Torfino的MAU了：

![图片](assets/0c38acee85ee.png)

Torfino的做法是每一级都要专门的RAM、TCAM做PHV的match并放入寄存器，然后每一级都有一个相对固定的Instruction RAM来执行一些指令。那么可不可以在这个架构上改一下，增加一个Program Counter，并把Instr RAM改大，同时增加新的一个ALU和控制器回路去修改Parser的RAM、TCAM，这样就构成了一个RTC架构了。这样做的一个好处是不需要大量的Match和Action Memory，整块芯片功耗自然下降了不少，而且针对MPLS标签交换这样简单的feature可能几个时钟周期就处理完了转发出去了。你会看到Tofino2也在做同样的事情，增加了VLIW指令的支持，不过它们相对于Silicon One还是没有PC寄存器，因此做一些Branch、JMP的操作还是有些限制的

![图片](assets/6cb28a7d1b4f.png)

那么指令是怎么送入到RTC的核呢？这里其实又回到了当年微码编程的解决方案了，即通过类似于P4的DSL构建转发语义，然后编译成微码。比起直接写微码容易很多，算是编译技术上的一个进步。而报文FIB查询使用HBM，那么同样的以空间换时间的做法就出现了，HBM带宽大但延迟还是DRAM那么多，因此可以同样的FIB表放置多份在HBM中，每个Slices 访问自己的那份不就行了？然后RTC内部通常还有一些L1D$，缓存一些HBM访问到的结果，这样进一步降低了HBM的访问频率。

当然针对交换机的市场，Silicon One直接把HBM裁剪掉，片上的TCAM、SRAM结构就够用了，这样就做到了路由交换一体了。

![图片](assets/e0e1b8810cf5.png)

当然微码结构也有它的问题，所幸的是它的开发过程随着P4等DSL和编译技术的提高变得容易。

但是单个RTC还是不能执行过长的指令集，同时很多Stateful的软件特性例如防火墙、DPI这些在这样的芯片上还是无法完成的，毕竟L1 I-Cache Size有限，片上网络的速度也有限制，同时HBM的Size也无法支持同等的Stateful流，例如一个标准的400G防火墙并发连接数的要求可能要超过50M甚至到100M，12.8Tbps的防火墙需要的HBM容量预估为128GB以上，这是HBM现阶段办不到的。当然所幸的是这些更复杂的软件特性通常是在边缘路由器上解决的，思科还有QFP-3来做这样的事情：）

**6. Smart Edge & Dummy Core**

网络界通常针对分布式系统复杂性的另一种思路就是边缘智能和简单的核心。所以你会看到云计算发展到一定程度后也逐渐开始往边缘计算靠拢，这些都需要科学的发展观来看待。

**6.1 Dummy Core：Innovium**

如果说路由器芯片我最喜欢的是QFP(吃饭的家伙能不爱么...)那么最喜欢的交换芯片就是Innovium了，当然最喜欢的路由交换芯片是Silicon One(甲方爸爸要跪舔~）

![图片](assets/8dc8bdc96f99.jpg)

它的取舍也恰到好处，整机采用相对固定的流水线，唯一的优点就是快轻松25.6Tbps，然后数据中心要的功能基本上都有，特别是Telemetry的功能非常有用，整机转发延迟460ns比其它同类交换芯片降低一半。这些其实对于数据中心或者大型分布式集群的Core就够了。

**6.2 Smart Edge-1：Pensando**

在新一代的架构演进过程中，随着虚拟化和容器技术的出现，TOR overlay逐渐变成了Host Overlay，而传统的TOR交换机也面临复杂的ZeroTrust规则下ACL/TCAM不够用的窘境，那么干脆就把TOR做到网卡上，也就是智能网卡DPU这样的场景。

Pensando的做法其实类似于一个FastPath/SlowPath on Chip的处理方式，下方是一些P4的Slice然后通过片上网络和ARM核心及其它加速协处理器互联。

![图片](assets/81fc9004567a.png)

P4和Barefoot类似的做法，就是一系列SRAM、TCAM构建的Parser以PHV的形式存放header fields、指令和一些metadata，最多可以到8kb，然后它的流水线为8级，Lookup engine可以做到2048bit的查询，Hash或者TCAM查询都可以支持，而相对于Barefoot也有了一些动态指令的处理能力，可以注意到Pensando也在上面放置了Icache和Dcache用于做一些更复杂的操作。

![图片](assets/4098de33b993.png)

针对存储和网卡访问的一些特性，它添加了一些压缩和数据完整性校验的功能，同时还配合ARM做控制面实现了TOE(TCP offload Engine)。但是出其不意的是在它7nm的平台Elba上直接把HBM换成了Off-chip的DDR4、5，可能是考虑的主机PCIe侧只有200Gbps的带宽，所以整个chip也没必要用那么高带宽的内存了

![图片](assets/850bf2de5ebd.png)

pensando的DSA架构本质上就是在边缘做所有的事情，但是它们没有选择RTC，而是采用了在P4 engine上扩展一系列I-Cache和D-Cache的方法来做，这都是趋势所向，在RTC和Pipeline之间取得平衡。

![图片](assets/364219498300.png)

当然除此之外类似的架构还有Netronome采用以前Intel IXP的微码技术演化而来的智能网卡，然后直接兼容eBPF的方式来做并行处理，有机会再跟大家介绍。

**
**

**6.3 Smart Edge-2：Fungible**

当然还有继续走RTC路线的fungible或者AWS的nitro，Fungible构建了一个有192个RTC线程的处理器，并且和一些加速硬件紧耦合，单芯片800Gbps的吞吐，同时也采用了HBM。

![图片](assets/4d714d469379.png)

单个Data Cluster采用6核4线程的结构，并且配合专门的加速协处理器，例如Lookup engine、Security engine等：

![图片](assets/7ee8222351c6.png)

整个编程结构相对干净，因为是RTC的所以完全兼容C代码程序：

![图片](assets/edd8cc77a3ac.png)

但是Fungible的问题和juniper很类似，就是不太会做Marketing，一开始老早就吼DPU，最后却是被nvdia的老黄发扬光大的，真是可惜。你看下面这个ppt，真是不知道它想要干什么，但是这东西本身还是不错的，当然任何架构都有软肋，得用了的人才知道

![图片](assets/59306ff6079d.png)

后面似乎有人点拨了一下，ppt写的好了点，它是以网络为中心的第三个Sockets

![图片](assets/9b07741b80ac.jpg)

**6.4 Smart Edge-3：NanoPU**

其实Nick自己也意识到了P4这个问题，一方面P4社区在给P4加指令集，另一方面又要把P4弄到BPF编译，然后看到Netronome有了BPF的微核构成的智能网卡，最终搞出个NanoPU的东西：

![图片](assets/b2c822cc6543.png)

也就是我去年说的：
 如何把数据包编码使得转发模型能够更好适应异构的硬件平台，编码上降低RTC-Pipeline-RTC这样的复杂切换流程，设计一种完备的DSL并且可以同时支持流量的RTC/Pipeline offload，这才是关键
本质上不要被花哨的定义搞懵逼了，这篇文章也道出了另一个关键的问题：在超过100个核心的ManyCore处理上，片上网络和通信延迟时一个必须要考虑的问题。

这篇文章的解决方法就是“Replacing the software thread scheduler and core-selector with hardware， by bypassing PCIe，main Memory and cache hierarchy completely”

![图片](assets/813ca306ce4e.png)

解决方法就是，别扯什么PCIe了，也别扯RTC、Pipeline了，我直接把RISC-V粘到P4上，是不是又想到了当年7600插各种业务板卡呢，不同的是人家把这玩意堆到一个芯片上了。

**6.5 Smart Edge-4：Tenstorrent**

技术扶贫大师Kim金坷垃去了这家小公司，去年Hotchip上就一个图，也是Packet Manager和计算引擎紧耦合

![图片](assets/9c9bd7a41a07.png)

**7. I/O and Co-packaging**

当你看到这么多的技术发展时，还有一些传输上的变化，例如56Gbps PAM4和Co-Packaged技术，光技术的发展逐渐也会小型化，然后最终芯片光互连和芯片内光通信也会逐渐出现。

![图片](assets/e47fca120864.png)

**8. 总结**

技术的本身没有优劣，你会看到不同的场景又不同的取舍，可以通过pipeline固化加速，也可以通过RTC获得灵活的业务。保持开放的心态去评判，找到适合自己的方式组件整个平台才是一个架构师真正需要的。

![图片](assets/f7101a1cbe3f.png)