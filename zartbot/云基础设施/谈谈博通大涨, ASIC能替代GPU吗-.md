# 谈谈博通大涨, ASIC能替代GPU吗?

> 作者: zartbot  
> 日期: 2024年12月21日 16:53  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492938&idx=1&sn=e1498fecef48bd10588d55e5fa834ae0&chksm=f995f588cee27c9e07a0452902cfe0a793b860cecdc4d4052e4001346a5447c2b454bd8d4898#rd

---

随着AWS发布Trainium2, 规模估算达到50万片, 以及未来Trn3的预期. 又伴随着博通CEO的预期这周暴涨, 似乎一夜间ASIC就要替代GPU了? 以前写过一个专题

[#三万亿的破绽 ](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3682079454764843014)

半年了, 基本上还是在三万亿上徘徊, 此刻恰逢思科成立40周年, 今天想来谈谈历史, 上一次互联网泡沫时期, 思科是如何应对Juniper ASIC的挑战的, 顺便谈谈牧本摆动周期, 或许有些答案就显而易见了.本文目录如下:

1.牧本摆动

2.谈谈思科在1990年代如何应对ASIC的挑战

2.1 先谈谈转发算法的历史 

2.2 思科的“NVLink”

3. Juniper ASIC的挑战者

4. Cisco IOS如同今日NV CUDA

5. 谈谈硬伟大

### 1. 牧本摆动

以前整理过一个过去四十年的[《GPU的架构演化史》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=2538479717163761664)

可以看到清晰的牧本摆动, 从定制渲染管线到可编程的GPU, 再到TensorCore光追一类的架构破坏了CUDA SIMT的抽象. 几年前也写过在网络这一行里的牧本摆动

[“网络编程” 还是 “可编程网络”？](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247483952&idx=1&sn=99baa3057eceb0abc0ec34cefd36ef14&scene=21#wechat_redirect)

牧本摆动背后的本质是差不多每隔十年, 当算法演进停滞的时候, 通过规模来继续发展时就需要向ASIC转移, 然后再过十年, 规模遇到瓶颈又会回到算法的创新上来, 并且伴随着半导体工艺的进步,一些可编程的实现在性瓦比上又占据了优势.

其实这一切唯一没有改变的就是计算机体系结构视角上的一系列降低平均访存延迟, 访存次数和访存功耗(Data Locality)的各种tradeoff.

### 2. 谈谈思科在1990年代如何应对ASIC的挑战

恰逢上周参加了思科成立40周年, 思科中国成立30周年的活动. 勾起了很多回忆.特别是在如今地缘政治如此复杂的时候, 看到思科刚进入中国的广告, 互联网过去的三十年确实是彻底的改变了我们的生活

![图片](assets/31ecbff1b6fd.png)

#### 2.1 先谈谈转发算法的历史

早期思科的路由器架构可以参考《Router Architecture And IOS Internals》[1]

最早的Cisco路由器, 其实就是一个普通的PowerPC CPU的主机, 整个数据包转发都是软件实现的, 所以也被称为进程交换

![图片](assets/5eebeddb8c61.png)

和今天的Linux转发报文类似, 如图所示, 网络收取数据有一个环形缓冲区, 通过中断通知CPU进行处理, 然后CPU访问报文头中的字段, 通过查表知道转发的出接口, 并将其转发到出接口. 后来随着数据量的需求越来越大, 又构建了一个Cache来处理, 加快经常性数据的访问, 也被称为Fast Switching

![图片](assets/6377a8aef703.png)

然后在查表的数据结构上又逐渐演化出了Radix-Tree,  Trie 层次化Trie等算法

![图片](assets/5ea4b4d96730.png)

并且以此为基础构成了Cisco Express Forwarding(CEF)的转发架构. 当算法上趋于稳定, 就开始各种卷带宽了.

#### 2.2 思科的“NVLink”

最早Cisco自定义了一个Cbus(Cisco Bus)它是一个16.67MHz的32位总线. 总共能提供533Mbps的带宽, 于是就出现了如下的基于多路总线的架构.

![图片](assets/66daf3f93fdd.png)
实际上它就和第一代的NVLink类似了, 而后期随着PCI的出现, Cisco也很快的切换到基于PCI总线的架构, 例如7200系列路由器.

![图片](assets/84d04ed3a531.png)

然后单个处理器转发不够用了, 于是构建了一个分布式的处理架构, 即7500型路由器, 类似于早期的Nvidia V100架构, 如下所示, 控制面的RSP类似于今天的X86 CPU

![图片](assets/2e9d958ea7b8.png)

数据面上,通过自定义的一个Cybus(类似于NVLink 2.0), 当时支持1.066Gbps(64bit x 16.67Mhz)的带宽来连接多个分布式的处理器. 而下面的VIP就类似于今天的GPU了, 通过多个加速器并行对数据报文处理,并且这些加速器也是通用可编程架构, 基于Motorola R4600/R4700/R5000 的MIPS处理器.

![图片](assets/bb18bce4b393.png)

然后到了大概1996年的时候, 类似于NVSwitch的交换背板架构出现了, 即Cisco 12000系列路由器

![图片](assets/29806092071e.png)

每个LineCard使用的也是R5000 MIPS处理器, 后期换成了PowerPC的处理器.

### 3. Juniper ASIC的挑战者

1996年Juniper成立的, 它在控制平面利用了成熟的FreeBSD和Intel Pentium X86架构CPU构建, 转发平面完全ASIC化

![图片](assets/a4911a6e7f4c.png)

并且基于此构建的Juniper M40系列路由器在1998年上市, 它的架构如下所示:

![图片](assets/b92d2078752b.png)

通过A-Chip, B-Chip和C-Chip等多个ASIC配合完成了全ASIC的数据转发. 例如:

PIC卡接收到报文后, PIC I/O ASIC对报文进行CRC校验, 如果校验失败则丢弃报文并通知JUNOS的相关接口计数器更改. 如果校验成功则进行Layer2解封装, 将Layer 2 PDU发送到FPC上的I/O Manager ASIC.

I/O Manager ASIC检测报文的协议类型, 然后在包标志信息内设置一个可能被用于不同服务的标志.然后将报文分割为64-byte的Cell(被称为jcell). 其主要原因是使得内存使用更加高效, 同时定长交换可以优化转发时延.并将每个jcell传递给位于SCB(or SFM)上的分布式Buffer Manager ASIC.

Buffer Manager ASIC将jcell以轮询(根据PFE进行轮询, 将不同的jcell发送到不同的PFE上的共享缓存中)的方式发送到分布式的共享内存中.

Buffer Manager ASIC同时根据jcell中的相关flags提取信息, 例如对单播IPv4报文提取源目的地址,输入接口, UDP/TCP端口等参数, 对于MPLS查看MPLS标签等. 收集完这些信息后生成一个key cell, 并将其转发到Internet Processor II ASIC

Internet Processor II ASIC对报文进行查询, 并将完成查询的转发决策整理成一个result cell返回给Buffer Manager ASIC, Internet processor II工作流程如下所示:

![图片](assets/46bcfd405410.png)

此后Buffer Manager ASIC将result cell中的信息进行解析, 并通告出口PFE的I/O Manager ASIC, 如果是组播流量则通告所有的出口I/O Manager ASIC

出口I/O Manager ASIC 通过分布式的Buffer Manager ASIC从分布式共享内存中读取jcell. 并管理包队列和一些QoS特性等. 最后I/O管理器将这些jcell重组为packet, 并将帧结构转发到出口PIC I/O ASIC

出口PIC I/O ASIC根据出站媒体介质进行编码, 例如POS编码或者HDLC组帧等操作, 最后从端口上发送出去. 离开路由器进入下一跳.

### 4.思科IOS如同今日的“CUDA”

思科IOS是一个诞生于1980年代的单线程的操作系统, 并且混合了转发平面. 经过很多年的迭代和用户市场的广泛接受度, 用户使用习惯很难更改. 虽然有Cisco 7500/12000这些多个MIPS CPU构成的系统, 实际上还是多个单线程的IOS构建的分布式系统在运行,  类似于今日的NVL72的Blackwell, 相当于每个ComputeTray都有一个OS.

这样的系统稳定性会出现一些问题, 例如我刚进思科的时候, 就处理过一个非常重大的事故(就是那种四个小时就要给CEO钱伯斯汇报一次的故障), 在圣诞节前,某个欧洲国家Cisco 7500过载一直没有更换, 最终导致主控的IOS死机而分布式板卡的还在给其它节点发送心跳信息,导致该国剩下的其它几台设备全部超载而整网瘫痪.

另一方面维持IOS生态的代价也非常大, 但是基本上的查表算法和TCAM等查表器件逐渐成熟使得Cisco也逐渐向ASIC化转移.

从最早的Engine 0采用纯CPU转发, 到Engine 1是用部分ASIC处理ACL,但是还维持了一个慢路径来支持所有IOS的功能, 似乎有一点类似于Nvidia 现在TensorCore+ SIMT的架构. 然后在Engine 2上构建了路由查表的PLU, 硬件存储Mtrie, 以及TLU执行Table Lookup等功能.

而在广域网边界等需要更多功能的地方, Cisco开发了一个PXF的ASIC, 采用流水线和列SRAM的架构, 如下图所示

![图片](assets/fd5bbd50c912.png)

再到后面,发现各个平台号称都使用IOS, 但是软件功能上的差异和一些配置导致走慢路径的性能影响非常大. 整个硬件团队维护多个ASIC的代价也非常大. 后来就出现了一个比较大的生态的分叉.

在运营商级别功能相对较少, 处理速率较高的场景, 开发了一款基于微码的多核心网络处理器SPP, 并且以此为基础构建了CRS-1这样的大系统. 它是一个2D-Mesh的片上网络, 有点类似于Nvidia近期的某个专利, 然后互联采用了光, 如下图所示

![图片](assets/ea3c44452f27.jpg)

![图片](assets/4ca3aee79037.jpg)

图中的光互联和2DMesh片上网络, 让我想到了NV最近的一些专利
![图片](assets/0e0ac6e5010e.png)

CRS-1可以通过多个Fabric Chassis(类似于NVSwitch机柜)和多个LineCard Chassis(ComputeTray机柜)全光互联

![图片](assets/b06360eee888.png)

或许这也是未来Rubin可能会走到的方案, 初期CRS-1号称最多可以支持8个Fabric Chassis + 72个Linecard Chassis 提供超大规模的集群满足互联网爆炸性的数据流转发需求. 然而最终落地的实际规模大多数在2+16这样的集群. 同样有一个72的数字, 我一直在反复折磨自己, NVL72的可靠性和接受度真的那么好么? 是不是最终NVL36这些才是更靠谱的方案?

另一方面,为了支撑SPP的新的微码处理器和多个Fabric Chassis组网的系统可靠性,同时为了避免IOS单线程的各种故障和性能瓶颈. Cisco在这个时候基于QNX系统推出了IOS XR操作系统..

再到后面, 整个公司发现ASIC还是太多了, 特别是一些企业级和运营商边缘产品还是需要可编程的处理器, 于是开始了QuantumFlow Processor的研发, 整个处理器研发大概花了4年的时间, 完成了一个完全可使用C语言编程的处理器, 它同样也是一个片上网络采用2D-Mesh的架构, 处理器微架构和Nvidia的SM几乎是相似的, 也出现在同一个年代, 同时在Memory总线上挂了很多ASIC, 也和今日的Hopper挂载一些TensorCore和TMA类似.

![图片](assets/522db5238065.png)

其实到了这个平台才完全完成了IOS从单线程操作系统到多核分布式处理平台的改造. 我们把IOS作为一个进程在控制平面的Linux内运行, 然后转发平面把IOS以前的很多C代码逻辑复用卸载到QFP上, 这套系统构成了IOS XE, 虽然它也有很多缺陷,但是能够满足用户从老的IOS上完全平滑的迁移升级上来,并且还具有多业务处理能力.

到了后来Intel X86多核处理器逐渐成熟后, 我么还把在QFP上运行的C代码移植到X86和其它商用网络处理器上(Marvell Octeon).

然后就是基于这套框架, 裁剪一下迭代一代做SPP构成了下一代的CRS-3, 然后再回来修改一下迭代出第二代QFP, 然后又衍生到ASR9000的Lightspeed处理, 再回来做第三代QFP处理器, 在高性能微码处理器和多业务C通用编程处理器构建类似于Tick-tok的路径.. 直到后来又有了Cisco Silicon One的决策, 这一切就停滞了

### 5. 谈谈硬伟大

其实硬伟大也非常尴尬, 培育了10多年的CUDA生态, 有点思科当年IOS的味道, 大家都熟悉了, 这是一个很大的惯性和飞轮在运转着.例如时至今日无论是华为还是其它网络厂商CLI, 甚至是部分SONIC的CLI都习惯性的去兼容Cisco IOS的配置方式.

而当NVidia为了应对TPU的挑战, 推出TensorCore的抽象时, 其实某种意义上又破坏了CUDA SIMT的架构, 可编程性在肉眼可见的下降, 为了喂满TensorCore, 各种异步的ld/st其实挺烦人的, 虽然后面有TMA的抽象可以简化很多地址运算的编写, 或者是Cutlass这样的模版开发也还算方便.

但是最大的根源还是在体系结构上, GPU L2Cache并没有一致性, 因此不同的SM之间的数据传递还是要写透到HBM, 虽然小范围的有一些DSMEM可以通信, 但是很多场合下HBM的带宽和Cache层次化结构加深下的大规模数据搬运带来的功耗问题依旧是有很大的挑战的.

相反那些以脉动阵列为主的架构虽然片上网络这些2D-Mesh/3D-Torus挺烦人的, 但是整体能耗和吞吐的优势还是存在的. 特别是未来两年如果全部像o1/o3那样卷推理, Transformer本身模型的架构没有太大的变化时, 或许这是ASIC逐渐占据市场的一个好的机会. 例如Juniper在1999年抢占了15%的市场份额,并成功上市.到2003年核心路由器市场, Juniper占到31%的营收.

而Nvidia为了维持自己One Giant GPU和CUDA的生态抽象, 在NVL72的路上不停的狂奔着. 但是当前GPU的架构已经演进到了极限了, Blackwell的微架构时至今日都没有公布, 我不知道Nvidia在藏着什么? 同样的FP16的算力增长来看, B系列仅比H系列增加了15%.

我会猜测的是, 从Blackwell这一代起, Nvidia会构建更加分布式的L2Cache, 然后逐渐在Rubin这一代比较优雅的抽象出NUMA Aware的CUDA架构, 当然以前Nvidia已经有很多论文在阐述了.

[《英伟达GB200架构解析4: BlackWell多die和Cache一致性相关的分析》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489759&idx=1&sn=2c55ec63d6deaeb39ff7f767896ba853&scene=21#wechat_redirect)

而从Nvidia的专利来看, Rubin以后会在Processor Die上堆叠分块的内存, 并且片上构成一个2D Mesh的架构

![图片](assets/7499530ce10e.png)

然后就构成了如下的架构

![图片](assets/b860e28968de.png)

实际上这样的核与Tenstorrent/Dojo又有多大的区别呢? 维持CUDA的生态, 是NV的优势也是羁绊....

至于ASIC的生意, 当年做路由查表的TCAM器件的公司都有好多家, 或许如今专门搞点TensorCore的Die通过CXL/UALink这些总线做一些互联的探索就变得非常有趣了, 例如在HBM4的Logical die上做一些有趣的事情. 这些的目的本质上是用在相对确定的推理场景下, 例如在一年后给o1,o3这些LRM续命, 同时又能产生大量高质量的数据的关键.

对于未来基于ASIC的体系架构基本上脑子里已经有了答案, 干就行了...

参考资料

[1]
Router Architecture And IOS Internals: https://www.cisco.com/c/dam/global/fr_ca/assets/presentations/MPLS/Router_Architecture_And_IOS_Internals.pdf