# TrioML vs SwitchML: RTC还是Pipeline呢?

> 作者: zartbot  
> 日期: 2022年8月11日 02:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487902&idx=1&sn=088064f846718eecb0a6a4a015df3a61&chksm=f996015ccee1884ab7d76f637947aa05b4dd7d1991a009109a45bfd9104b0d8da5079c626f54#rd

---

### 题记

Sigcomm22 有一篇关于Juniper Trio系列可编程芯片的论文[1] 一方面思科内部的一些同事在找我做竞对分析，另一方面利用可编程网络器件做ML优化的一些同学也在找我。那么下文算是一些读论文的小笔记吧。**给思科的同事敲黑板，你们拿Silicon One赶紧找几家互联网大厂和某个大学一起搞个SiOneML。**

总体来说，TrioML对比SwitchML的优势1.8x，这是必然的，只是过去很多年大家在数据中心关注交换机**时间太久了，加上P4搞的一堆东西，路由这个东西被淡忘了。很高兴能够看到拿NP来做这样的ML等网内加速应用的场景，工作本身也很出色。而这篇文章披露的Juniper的处理器架构基本上可以算20%左右Juniper的市值了，也是很值得去读的， 你也可以看看世界上那些顶尖的路由芯片的设计思路和交换芯片的不同。而另一方面值得您注意的是去对比和思科QuantumFlow处理器的架构区别，以及为什么Juniper走呢那么多年微码的架构，为什么Pradeep Sindhu要去创立Fungible，而且当时Fungible的DPU定义很大程度上是要在大量的L4~L7的软件功能上，特别是安全feature上补全Trio，所以才选择了MIPS多核的架构，选择MIPS指令集本身也和JNPR有一些关系，所以这篇文章后面讨论用Trio做In-network security似乎还是差了那么一点的味道...

### 概览

原文摘要讲，Trio是一个多线程的包处理引擎并且有很好的层次化的高容量的内存系统，这句话是怼给Torfino看的，所以后面会讲fundamentally different from pipeline-based architecture. 然后呢就表明了它可以非常优雅的去处理各种协议和场景，并且能够使得非同构(non-Homogeneous)的包处理速率维持在一个很宽的范围。这句话的有点绕，本质上就是说，有些场景可能包处理只需要几十个指令周期，而有些需要数百个。这样的多业务的情况下，传统的Pipeline处理效率是不高。敲黑板，所以为什么思科的Silicon One选择RTC**? 具体内容么参考下文：

<[**包处理的艺术(3)-RTC vs Pipeline**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485339&idx=1&sn=62439ad879f26f18c4434e1e51a0bdc3&chksm=f9961759cee19e4f776bf5719ce4c0635b09bbbebdcf7e37a682972e836d6c22e23bd64e2982&scene=21#wechat_redirect)>

当然Trio这样的处理器微码架构也不是它独创的，也不是最近才出来的。最早在20年前思科的7200系列处理器就逐渐诞生了，然后还有Intel的IXP也就是现在兜兜转转卖到某国内DPU厂商的那玩意。当然思科的一系列CRS-1和ASR9000以及当时的EzChip基本上都是这个路数。针对常见的运营商级路由器来看，微码空间似乎也够用了。但是思科在自己的企业网市场上遇到了更多的挑战，NGFW这些需要DPI**的场景或者更多的其它灵活业务的场景，单个报文处理通常需要几千个指令，那么微码的处理器直接歇菜了，而且同样来说这玩意开发特别的不人道，即便是后期有一些c-like的编程环境。所以我们在2004年设计Cisco QFP处理器的时候就准备在更低容量下实现更多的功能。当然这个处理器的设计者之一Will后面也去了Juniper参与过Trio后面几代的设计。至于它们之间的一些微妙的区别，可以看另外一文：

**<**[**Trio 6, Express 5 and Silicon One**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487442&idx=1&sn=911c39caa81925c6c1786a9a5c775b9d&chksm=f9961f10cee19606f20a1466320ae7dc1740eb991592db947cd6ba602272aaa554530f6b8f29&scene=21#wechat_redirect)**>**

文章后面做了两个DEMO，一个是大家都要做的分布式机器学习的Allreduce操作，典型的I/O密集和计算密集叠加的任务，直接来吊打Tofino的SwitchML，而另一个是利用Trio的计时器线程来缓解在网计算任务中的一些掉队者的情况。
introduction章节
introduction这个章节一开始就说，尽管你PISA架构很厉害，但 often a poor fit for emerging in-network application，真是泪目啊，终于有人看懂了，渣以前一直说P4跟渣一样的垃圾你们不信吧...而Trio本身，则采用多块处理器和交换矩阵连接构成的一个集群架构，和传统的Pipeline交换机区别如下：

![图片](assets/731e036d6b11.png)

然后它本身采用RTC的方式处理报文，所以带来了一个潜在的优势，对于高速率简单功能的报文和相对低速率功能复杂的报文可以混跑。简单的说就是不会像Pipeline那样导致流水线空转的问题，而Trio系列处理器除了有一系列执行微码的PPE外，更重要的是它的内存子系统，也就是我一直说某豹十几年都没搞明白的地方。

![图片](assets/08ea27bdfb86.png)

而Cisco QFP也是类似的架构， 不过比起EzChip、Trio、Intel IXP用微码的，它是玩的是基于Tensilica定制的内核和指令集，可以用标准的C编译器所以后面若干年在一些基于X86的虚拟化平台上，Cisco这套转发引擎的灵活性和性能又远高于其它基于微码的厂商。另一方面和当年相似的Cavium、RMI厂商比，内存上又有一些特殊的地方，所以基本上豹家前生的RMI我们根本看不上，Cavium要么给我们当协处理器，要么做低端的产品线。

![图片](assets/1d4b97effdd4.png)

当然针对高端的路由器平台，确实吞吐率大于功能时，例如CRS-1的SPP、QuantumFlow Array、NCS、ASR9000这些LightSpeed当然还是会选择微码的方式，所以架构上都是时间、空间、成本的取舍。

当然微码空间的问题也导致了Juniper过去几年在功能上增加非常缓慢，相比于Cisco QFP最疯狂的时候可以一个Release提供上千个软件功能，你可以看到它的整个架构演进这十年是相对缓慢的：

![图片](assets/15959217928d.png)

渣的一些疑问：其实我一直怀疑的第一个问题是成本，MX480的价格比Tofino的价格，毕竟路由器是一个高附加值暴利的行业,整机价格数倍于Tofino,说不定让我多买几张100G网卡，整机速度提升更快。另一个怀疑的地方是Trio的浮点计算能力，当时我们内部也有QFP或者Silicon One的处理器用，而最终在做NetDAM的时候选择自己构建FPGA并使用ALU和HBM来构建分布式的PPE，中间连一个傻快傻快的交换机Fabric. 只不过我们和Trio不同的是，我们把PPE直接扔到了网卡上，因为一方面我们可以构建更多的ALU，甚至实现FP32到BF16的转换，在allreduce过程中利用低精度来降低通信量，另一方面能够通过主机的PCIE、CXL更快的和GPU通信，并在主机侧做aggregation更容易.

### Trio架构

Trio被广泛用于Juniper MX系列路由器，宏观的包处理架构参考Juniper出版过的一本书

![图片](assets/ed63dd52bf16.png)

一个路由器的常见器件如下：

![图片](assets/7514b086cfbc.png)

接口逐渐成为标准的以太网了以后， 原来的POS、ATM这些interface block也就消失了，然后Buffer block、查表block和Qos队列等也逐渐的整合了，所以在第三代Trio开始就成了一块芯片

![图片](assets/a2c76745c257.png)

回到Trio PFE的架构图，PFE是一系列芯片逐渐演进构成的一个转发引擎板，从第一代40Gbps到现在第六代单芯片1.6Tbps，然后低端路由器只配置单个PFE卖，高端的则可以支持多块集群。

![图片](assets/b6b92dd45152.png)

每个PFE里面呢有数百个多线程的PPE(Packet Processing Engine)，然后每个PPE支持数十个线程同时处理报文，而且在同一个PFE里面的PPE可以共享内存和访问其它线程的寄存器。

对于并行包处理，通常的节省内存带宽的做法就是把一个包拆成一个128B的Parcel(Packet-Head)放入Dispatcher中，然后剩下Packet-Tail丢到共享内存子系统里，这样也可以节省大量的主处理器PPE的带宽，然后针对路由器常见的功能会添加专门的HASH计算和包过滤等协处理器，挂载在内存总线上。

例如Filter组件，和PISA就非常类似了，基于微码定义的一个引擎，而本质上在这些微码抽象的过程中有一些函数式编程的思想，即利用Pattern-Matching执行分支的方式，

![图片](assets/2ff6f0a12af5.png)

当然思科QFP相对于Juniper还做了更多的处理是，我们的GPM配合共享内存系统提供更好的全报文可见的处理方式，商业机密就不细谈了。而你看到的Fungible其实就是在这个基础上来解决Trio的访存问题的，打住点到为止。

最后报文在离开路由器的时候会有一个ReOrder Engine，本质上是一个Flowlock，对每个flow保序，但是问题来了，在做一些防火墙、NAT等需要流表的时候，如何避免这个锁，以及内存上的锁，相对无锁的Flow Based Dispatcher又会导致内部PPE workload不一致，点到为止。

#### PPE(Packet Processing Engine)

每个PPE是一个VLIW的多线程微码执行引擎，内含多个ALU、并且可以支持相对复杂的分支操作， 由于线程多所以并不太需要每个线程有多好的性能，但是前面说的基于FlowLock的情况针对ElephantFlow的NAT性能在某些性能敏感的地方确实出现了单线程性能不够的场景，某个客户一朝被蛇咬至今都要叫我们测大象流NAT性能。

每个Trio同时只有一个Datapath Instruction，所以呢，针对一些更加灵活的软件特性，需要函数调用的多指令多软件功能融合上，它面临一些瓶颈，例如我想做完VXLAN后再根据内层的端口做SBC的语音流处理，完了再调用一段NAT的代码。这个时候对于PPE来说就需要栈了，但是Trio并没设计这样的模式，所以您会看到Juniper的防火墙产品线，都要...

然后你就看到它Per-Thread的本地存储空间只有1.25KB，利用指针寄存器或者微码中定义好的地址去访问，然后每个Thread有32个64bits的通用寄存器，用于Load、Store 共享内存的子系统。不过对于TrioML来讲，基本上整个报文中的梯度参数都要传递进PPE来算完丢回去，某种意义上来看不是那么有效了。而前面的那个PacketHead、Tail的划分就显得画蛇添足了。

ALU有两种，Condition ALU 可以处理32bits的数据，而Move ALU主要负责Load、Store等操作，这样针对不同长度的头乃至bit-level的内存访问有更好的灵活性，

而接下来的一段Sequencing logic，微码采用顺序执行，但它可以做复杂的多路的Branching，然后执行相应的target block，最多可以nested up to 8 level up，但是并没栈的结构，所以For Loop和更加灵活的function调用就很难了，所以这也是我们在做QFP的时候强调的是Full C programmable with stacks， 而Trio和Ezchip只能说提供C-Like的功能，具体在后面可编程的那个章节继续说。

然后硬件的Hash引擎，基本上网络处理器都会带，好像IBM最近在新的power上也加上了这个做Offload，基本上每家都有的就不详细说了。

### Shared Memory Subsystem

整个内存子系统的架构如下：

![图片](assets/4797e68cbab6.png)

有些Counter类的需要高速修改，所以访存的方式上需要支持事务内存和一系列的Read-Modify-Write的过程，然后它有片上的内存，通过Crossbar后的访问延迟是70ns左右，而对于Off-Chip的DRAM访问延迟是300~400ns(前几代是HMC，但是因为这个原因最近在大规模的断供，后几代换到了HBM)。然后内存支持一些事务操作，然后Read-Modify-write engine就很适合做MPI Allreduce一类的处理了。这里面还有很多Tricky的事情，豹是想不明白学不会的，毕竟后来Fungible也没怎么想明白，但这个问题想明白了约等于豹20%的估值。

### Trio可编程

可编程环境嘛，Trio只能说C-Like，而我们可以做Fully C programmable，当然这些东西是业务属性决定的，毕竟高端的路由器不需要，转发pps和内存带宽及线程数决定了执行时间就那么点，这个没有好坏而是业务属性的取舍

![图片](assets/d1983dc59644.png)

Expression syntax说支持各种变量类型和结构体，并支持指针数组、condition、函数调用和Goto以及switch... 真是浪费词，不支持Loop就明说嘛，文邹邹的干嘛呢？然后微码么自然有一些约束了，单个指令可以使用4个寄存器和两个本地内存的读及其它两个寄存器或者两个本地内存的写操作。然后支持对于外面的一些器件使用XTXN处理调用，同步异步都支持。本质上这是一种内存访问抽象，访问异构器件的常用方式，所以再去对比一下NetDAM的访存和计算的操作指令。

然后vMX虚拟路由器这个就有点牵强了，毕竟微码的高效率到了X86上是啥样呢？有些显而易见的难题，而思科在企业这些环境中常年使用通用处理器做低端平台，例如Intel Atom或者后来的Broadwell，或者Octeon，乃至一些低端的ARM核都可以玩，所以在指定架构的时候运营商视角和企业网视角是有很大的不同的。

不过微码C-like的编程比起P4还是很容易让人接受的，就是内存会搞的一团糟，大概的Example

```
struct ether_t {  dmac : 48;  smac : 48;  etype : 16;};process_ether :  begin    ir0 = 0;    if ( ether_ptr -> etype == 0 x0800 ) {       goto process_ip ;    }    goto count_dropped ;  end  process_ip :  begin    const ipv4_t * ipv4_addr = ether_ptr + sizeof ( ether_t );    ir0 = 1;    if ( ipv4_addr -> ver == 4 && ipv4_addr ->ihl == 5) {      goto forward_packet ;    }    goto count_dropped ;  end  count_dropped :  begin    const : addr = DROP_CNT_BASE + ir0 * 2;    CounterIncPhys (addr , r_work . pkt_len );    goto drop_packet ;  end  
```

### TrioML

TrioML的报文如下：

![图片](assets/d5d020f13524.png)

还是UDP封装+指令头+数据的方式，然后不就是NetDAM么，12Byte的TrioML Header如下：

```
struct trio_ml_hdr_t { // 12 bytes  job_id : 8; // aggregation job id  block_id : 32; // aggregation block id  age_op : 4; // if the block has aged out  final : 1; // if the block is final block  degraded : 1; // aggregation is partial  : 2; // unused for byte alignment  src_id : 8; // source id of the packet  src_cnt : 8; // number of sources contributing  gen_id : 16; // generation id  : 4; // room to expand grad_cnt  grad_cnt : 12; // number of gradients};
```

报文的设计某种意义上来看还是缺少一些灵活度和抽象。其实就差那么一步去看透事务内存的本质，另一个工作是阿里SIGCOMM里面利用类似的方式做存储，也是UDP+指令头+数据,不过头里面定义了自己的RPC-ID等操作

![图片](assets/c74e3826b686.jpg)

*Abstractness is the price of generality，DPU的基层综合治理考虑的艺术* 直接从代数结构上抽象所有的In-network计算的通用点，然后去DPU上实现是更好的处理方式，例如大量的计算有容错和事务操作的需求及READ、WRITE语义和SEND、RECV语义的转换，如果从代数上来看，就是UDP+指令头+数据本质上你要么把这个操作构成一个群、操作可逆的处理很难时，要么把这个操作构成一个半群并利用幂等来处理也就是SemiLattice了，

![图片](assets/2be367da3ff8.png)

所以某院士在那里讲“数学”是什么数据研究的科学，“算术”及研究算力的技术，顿时觉得有些尴尬了...

Allreduce的内容可以参考：

[**MPI Note[8]: 分布式机器学习AllReduce**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485803&idx=1&sn=8fea4ef0df7f89c3c0ff281b8a5cbd8e&chksm=f99619a9cee190bfe8a038d31b6c44803dd90113e01da78ca52fe71876ffe5c337c4932f2b6b&scene=21#wechat_redirect)

然后它的实现是一个window based，因为毕竟是In-network来算，但是想一下如果把PPE和部分OffChip Mem放到DPU直接在Server上呢？并且通过交换机实现内存池化？

TrioML整个算法逻辑如下：

![图片](assets/933067dc0c60.png)

然后就是Straggler mitigation，这个问题也很好，也就是说主机会受到OS和本地存储的影响，所以在做整个计算的时候有不确定性，这个不确定性的根源来自于PCIE DMA：

![图片](assets/60361638f102.png)

如果把内存前置在网卡上就会带来完全确定性的访问延迟，点到为止

![图片](assets/eed484ce2691.png)

当然TrioML使用了Timer Thread来处理并找到aged block，然后aged out并把partially的梯度更新发送出去。

### 测试

最后就是一些测试结果，当然会比SwitchML好很多咯

![图片](assets/af9b62ad2936.png)

但是背后的本质是什么，MX480和Tofino的成本对比，可能在这个setup下可以直接把主机的网卡全换200G了，然后ResNet VGG这些测试Model Size也就最大500MB，和我们当时做NetDAM 2GB的规模比还是差距很大， 然后MX480的处理线程数机架空间和每个Trio的HBM带来的内存带宽，和SwitchML比似乎有点不在同一起跑线上。

当然这个TrioML的工作是非常出色的，至少在一众完全用pipeline Switch的人群中，说出来RTC其实可以做的更好，处理更加丰富的业务逻辑，这一点是非常好的，但是要泛化出去做Network-telemetry、in-network security、PacketLoss等等场景可能会有问题， 当然做技术研究肯定没问题的，问题再商务成本上。

#### Reference

[1]
Using Trio – Juniper Networks’ Programmable Chipset – for
Emerging In-Network Applications: *https://people.csail.mit.edu/ghobadi/papers/trio_sigcomm_2022.pdf*