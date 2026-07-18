# [Sigcomm2024论文解析]谈谈网络研究和工程方法

> 作者: zartbot  
> 日期: 2024年8月10日 16:10  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491570&idx=1&sn=42f06db3f2dcb8f0244cef5c26f7fb50&chksm=f9960f30cee18626be86cb3f7b751a1bba02fc4fe45a55529240319af99b352de29a4b842b62#rd

---

虽然工作重心早就不在网络了, 只是偶尔遇到难题的时候帮忙打打杂, 但是也不妨碍周末无聊了读点论文聊个天.

### TL;DR

Sigcomm虽然也号称顶会了, 但是`工业界和学术界的差距是非常大`的, 今天来详细谈谈今年的几篇论文让大家了解一下用网络设备的客户和一些学术界的教授学生在认知和方法上和真正工业界顶级架构师的差距.

然后给大家介绍一下网络研发相关的一些方法论的东西, 简单的说

Smart Edge, Dumb Core

网络问题的解空间就三个: `拓扑`, `路由`, `流控`.

再扩展一点, 好好读读这篇文章  [《重读RFC1925，网络的12条军规》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484538&idx=1&sn=676883937b8f2279094fc2921ea3b77b&chksm=f99614b8cee19dae8bfc8b0f58ee51a075aeabafe618ab2fa35a085b11fd6d4232a5e5a580a6&scene=21#wechat_redirect) 我们会结合这些方法来分析Meta的H100 RoCE网络, Google的分布式SDN,某鹅的广域网等几篇Sigcomm 2024的论文, 基本上这些做法要么方向错了,要么就是几年前就搞完了的工作.

最后结合热点, 布置一个课后作业来谈谈当前自研芯片和拥塞算法都不行的一些大模型厂家, 如何搞点分布式流量工程的路由协议. 本文目录如下:

```
1. Smart Edge, Dumb Core1.1 Meta的SmartCore, Dumb Edge1.2 SmartEdge的做法1.3 谈谈分布式SDN1.4 分布式广域网路由2. 网络问题的解空间2.1 保持生态和维持现有基础设施2.2 解决网络问题的方法论3. 遥测数据处理和可预测网络4. 软硬件协同设计5. 课后作业:AI训练网络的路由协议设计
```

#### 0.1 先谈谈学术界和工业界的差距

**理想赛NV, 现实左江退**: 工业界用网络的客户和设备网络研发(`主要是指具有芯片到协议的全栈研发能力的团队`)的技术差距也是非常巨大的, 仔细读读RFC1925的一段话, 几十年过去了仍然受用.

Some things in life can never be fully appreciated nor understood unless experienced firsthand. Some things in networking can never be fully understood by someone who neither builds commercial networking equipment nor runs an operational network.

即便是满足上面两条, 长期针对网络协议和网络芯片架构进行协同设计的架构师和网络运维相关的工程师的差距也是非常巨大, 例如渣这种在思科搞过最核心的转发芯片相关工作的来看, 真正顶级的工作拿来发论文? 你搞笑么? 有些东西连专利都不会写的.

举个例子, 博通这些51.2T交换机的TM实现是非常难的, 你见过他们Sigcomm上论文透露过半点信息么? 整个高端交换机芯片的核心技术就在这里. 即便是这样的顶级厂家也有一些问题, 测试时也观测到明显的抖动, 只有极少数懂的人才会推测出它内部实现的机制和缺陷.

再举一个反面的例子, 国内某厂家的路由器架构有严重的架构缺陷, 但在几年前的招标测试中也被我直接一个case把标称100Gbps吞吐能力的转发引擎打到1Gbps线速都达不到, 因为这样的受攻击的隐患直接废标. 很多时候国内的这些架构师离顶级厂商的资深架构师差距还是非常巨大的, 总是以为发几篇顶会就可以站在世界之巅了么? 还同行评审?

## 1. Smart Edge, Dumb Core

这是才是网络的第一性原理, 任何事情越往Core实现的复杂度就越高一个数量级. 对于同样一个问题, 边缘节点实现可能只需要10Gbps~100Gbps的处理能力, 而在核心实现需要1Tbps~100Tbps的处理能力.

### 1.1 Meta的SmartCore, Dumb Edge

说一个反例就是Meta今年Sigcomm的论文《RDMA over Ethernet for Distributed AI Training at Meta Scale》[1]

主要的原因不知道是Edge的SmartNIC比较Dumb, 还是其它比较Dumb的原因, 于是他们选择了SmartCore, 一会儿扩大带宽2倍, 一会儿搞流量工程, 最后上了Jericho Deepbuffer还大言不惭说就此不需要DCQCN了?

![图片](assets/a980d2c4337e.png)

接入交换机25.6T,实际上只用了(16+18)* 400G, 差不多容量只用了一半多一点. 而且为了解决hash冲突的问题还考虑1:2的收敛比,好像带宽不要钱的样子. 为了解决拥塞的问题又在核心设备CTSW上采用带HBM的框式交换机, 整个系统的网络成本增加30%以上,然后22us的延迟有不得已去调整通信库, 整个问题没解决干净又要在接入的RTSW上改Hash函数同时还有在通信库上增加QP数量. 整个工作充满了hack.

对于22us延迟和30%+的成本, 我们首先来看看核心侧CTSW的Arista 7800R3架构, 主要是有些读者后台反馈对CTSW架构不熟悉, 我们稍微做点补充. 它是一个框式交换机, 内部结构如下, 由多块Fabric和Linecard构成, 相当于由多个Spine-Leaf交换机构成的组合体.

![图片](assets/e74825920eeb.png)

数据包到达Linecard后, 通过查询路由, 切分成信元(Cell)然后在入方向Linecard上排队, 通过Fabric Card转发后到目的Linecard, 然后重组报文再发送出去.

![图片](assets/af564d98d5ad.png)

通常这类架构在转发的过程中会出现队头阻塞(Head of Line Blocking, HOLB)的情况, 如下左图所示

![图片](assets/3921d4c785d3.png)

因此我们需要在Output的Linecard上放置队列(简称为Output Queue), 但是一个更有效的做法是在Ingress的Linecard上区分队列,构建Virtual Output Queue(VOQ)的方式进行转发来避免HOLB.  Meta所用的Arista 7800R3 规格如下.

![图片](assets/3763405d898f.png)
每个Linecard上采用3块 Broadcom Jericho2, 每块9.6Tbps的吞吐率,实际上4.8Tbps接前端光模块接口, 后端112个56GSerdes连接多块Fabric卡. 每个Jericho2芯片上包含了8GB的HBM2, 并且可以在拥塞时通过这些HBM2来缓冲数据包, `每个400G端口可以提供500MB buffer, 缓冲10ms`

![图片](assets/e5510ac5db4c.png)

于是我们算了一下整个24K卡H100集群, 每个Pod 3072张卡中需要部署18台这样的交换机, 累计buffer为192GB * 18 =3.4TB, 有了这样的缓冲能力, 当然可以大言不惭的说不需要DCQCN了,而且PFC也不会扩散了.

有些事情不够荒诞, 例如还有人在提在网计算, 那么就往这个方向再推向一个极端, 例如中间放一堆NVL72的GB200来替代Jericho2好像也能解决问题? 而且还是Smart Core, 带算力!

### 1.2 SmartEdge的做法

实际上这些带HBM的Jericho芯片主要是用在核心网路由器上的, 针对广域网上流量带宽差距很大的场景(LAN:WAN > 10:1)时需要进行缓冲. `Meta这样的做法本质上还是自身能力不行, 正确的做法是如果网卡上我们能够有效的进行拥塞控制.`

如果拥塞控制算法做的好, 平均每块400Gbps网卡大概只需要8MB左右的Buffer就可以非常干净的把流量控制住,并且通过端侧的多路径转发算法很容易打散流量分布到各个交换机上.`算了一下总Buffer的开销, 3072 * 8MB =24GB, 而Meta的SmartCore大概需要3.4TB.`接近150倍的差距.

当然这件事能不能做好呢? 能!

`真正在DPU上超越英伟达的, 全世界就只有一家`.还不用谈虚拟化和裸金属啥的,就放出一个NV最能打的RDMA, BF3来测测看?

采用Spine-Leaf拓扑, 不用任何框式交换机, 不需要DeepBuffer. 如何不利用交换机任何Hash函数信息, 不需要交换机任何特殊配置, 不启用ECN和PFC. 通过网卡算法自动打散流量,并维持交换网97.5%以上的利用率, 对于交换机的buffer需求为队列深度低于3us. 并能够针对128:1的时候incast时最大流和最小流量之间的带宽差异小于100Kbps, 同时针对任何网络线缆故障, 通信中断无感知, 模型训练收敛时间小于100ms.

这个Case是Smart Edge, Dumb Core的极致. 核心交换机buffer大概只需要几个微秒就够了, 但是BF3这些SmartNIC/SupperNIC并不是那么的Smart, 在选路和丢包的拥塞控制上陷入两难境地,Select Repeat做的也不行, 也就只能跟所有人说, 这玩意卡上搞不定, 要Spectrum-4交换机配合做, 还要开启PFC, 还美其名曰无损网络更高级, 端网协同更牛x?

顺便说一下吧, `BF4的几个研发高管去年来拜访我们交流Roadmap的时候, 我们告诉他们这些全部实现了,起初这群人还不信? 呵呵` 博通在UEC提的那些目标, 每一条我们都已经做完了.

![图片](assets/15ab0e6c00bc.png)

### 1.3 谈谈分布式SDN

其实这也同样是一个Smart Edge,Dumb Core的问题. 这有一个很讽刺的历史过程. 当年Nick号称搞SDN OpenFlow的时候其实是中了一个毒, 当年的背景是在Cisco CRS-1系列上有相应的平台抽象层和硬件抽象层, 然后大量的Linecard共享一个控制平面. 然后Nick就把这些东西想抽象成一个标准, 也就是OpenFlow. 然后性能不行啊, 这哥们不死心后来又折腾了一套P4, 最后十多年过去了, 这些网络设备上的控制面从原来的MIPS换成了Xeon-D, 内存也越来越大, 功能也越来越复杂.

其实路由协议是一个设计的很巧妙的分布式系统, 可惜能操刀搞这些协议的人太少了,以至于现在还在修修补补的搞BGP. 例如在广域网上面对大量的不可靠的链路, Centralized SDN也就出现了问题.

所以Google今年Sigcomm的《A Decentralized SDN Architecture for the WAN》[2]开始谈分布式SDN了. 本质上还是回到了SmartEdge 在设备上提供新的分布式控制器.

![图片](assets/c40c4a6c76c9.png)

实际上这些工作还远没达到我2018年做的Nimble Network的水平, 真正学术界和工业界的差距大概也就是6年左右.

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

Nimble是工业界最早在路由器上构建分布式SDN框架的, 可以看到其架构图如下.它也是在Cisco IOS-XE操作系统上构建了一个分布式控制器, 再对比一下Google这个图, 看懂了吧? 早他6年没说瞎话.

![图片](assets/81f36cee61f5.png)

而基于分布式强化学习流量工程今年也有一篇Sigcomm的论文《RedTE: Mitigating Subsecond Traffic Bursts with Real-time and Distributed Traffic Engineering》[3], 但是真正工业界的实现也是在2018年就有了

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

其实RL最难的就是Reward的计算, 当时采用的延迟测量算法(Application Response Time, ART)是2012年的技术,通过它来区分端侧拥塞和网络拥塞

![图片](assets/639d6805fa7d.png)

而后的Google Swift的拥塞算法本质上是相似的以延迟时间戳作为信号, 包括后面的Falcon也是这样的方式.

![图片](assets/3eb610043f79.png)

于是我们根据这些测量的延迟就可以很容的构建模型进行故障推理和性能打分

![图片](assets/68788eec4cbf.png)

然后再利用SegmentRouting控制路径即可.

### 1.4 分布式广域网路由

某厂今年SIGCOMM也有一篇论文《MegaTE: Extending WAN Traffic Engineering to Millions of Endpoints in Virtualized Cloud》[4]大概说的是传统的流量工程端和网是割裂的, 问题提的也算对

![图片](assets/4c455ac32393.png)

于是搞了一个Segment Routing Over VXLAN over UDP

![图片](assets/16fe2d3a5951.png)

说实话这玩意和我2020年搞的Ruta Disaggregation Routing System不就是一回事么?详细内容可以参考github.com/ruta-io[5], 或者如下专题

[《Ruta:Segment Routing Over UDP》 ](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=1364600286966415361#wechat_redirect)

当时做的就是Application Level的pathSelection

![图片](assets/82c218d1fe12.png)

看看Ruta的RFC, 完全一样的东西, 当然有些话就不挑明了,做事情不加个引用这样好么?

![图片](assets/6e5d92fbd9f6.png)

但是他们的设计还是有很多约束的, 例如穿越NAT,在Internet上是否能跑? 答案是否定的. Ruta才是全球第一个能够在Internet上实现Overlay并可以轻易的和QUIC这些传输层协议融合的路由架构.

![图片](assets/d526768b15bc.png)

就简单一句话, 谁能部署几个云上虚拟转发节点,就可以做到全球零丢包 200ms可达?

![图片](assets/8966b6fbb4a7.png)

并且这个路由协议是普适的用在数据中心内部网络替代BGP-EVPN也是可行的, 有个Demo如下:

[《Ruta实战及协议详解》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485165&idx=1&sn=412fcb1dd46dd4ef4384a033b0827256&chksm=f996162fcee19f39ab4c995b1be2676779eb5b647ad26dd7017001b75530dfb9a59e9790b37d&scene=21#wechat_redirect)

其实所有的本质都是在诠释SmartEdge, DumbCore的原理, 尽量保证中间节点无状态和简单的转发能力, 而把复杂的状态都往边缘处理.

## 2. 网络问题的解空间

**其实网络的问题很简单, 就三大块: 拓扑, 路由, 流控.**

### 2.1 保持生态和维持现有基础设施

很多网络的技术研发时, 要考虑逐渐部署的过程和维持现有基础设施的逐渐演进的过程. 推翻重来的那套实际上是行不通的. 例如早期基于MPLS的Segment Routing是很成功的, 但是SRv6还是存在一系列问题大规模部署非常缓慢.另一方面RDMA本来有很多缺陷, AWS才搞了SRD, 但是生态上的诟病使得用起来很不爽, 为什么就不能保证RDMA Verbs兼容呢? 生态的力量和惯性是非常大的, 所以我们RDMA的实现一直都是保证Verbs兼容, 然后再考虑如何解决RoCE本身的传输效率的问题.

同样在做Ruta这些协议设计的时候, 当然可以很轻松的假设Internet上都能跑IPv6/SRv6,但是uRPF这些问题还是解决不了, 另外还有很多网络只有IPv4. Ruta是唯一一个能够在Internet上穿越NAT实现IPv4和IPv6双栈流量工程的路由协议,这些背后的思考都是要保证基础设施尽量不做推倒重建.

### 2.2 解决网络问题的方法论

考虑一个问题, 我们首先就要考虑如何不影响基础设施的情况下解决问题, 例如`流控`能不能解决问题? 流控又涉及到很多人不太懂一些算法, 于是片面的认为端侧决策能力和拥塞发现能力有缺陷,于是需要中间的网络设备辅助,形成端网融合的方案. 但是网络12条军规里有一句话:

5.It is always possible to aglutenate multiple separate problems into a single complex interdependent solution. In most cases this is a bad idea.`对于很多个独立的问题，总会找到一个复杂的相互依赖的解决方案，但通常来说，这是灾难开始的地方。`

很多人没吃过这些亏,自然不能了解背后的很多故事(事故). 然后当流控问题解决不了的时候, 再考虑是否能够通过修改路由甚至修改路由协议来解决问题. 路由协议的设计上主要是考虑Topology Hidding和Fault Isolation这两块的内容. 最后这些尝试都失败的时候,再来谈网络拓扑的修改. 因为网络改配和建设的成本非常高时间周期也很长. 对于拓扑而言, 虽然一些非对称拓扑对于调度编排,流量工程都有难度, 因此网络拓扑上还是以对称同构的网络会更具业务价值.

## 3. 遥测数据处理和可预测网络

其实也是一个SmartEdge,Dumb Core的原则. SDN带来的集中式控制器部署使得它对于遥测数据的处理带来的很大的压力. 例如Meta那篇Sigcomm论文的有一个作者以前就在Cisco做Tetration的, 主要是把数据包的包头都采集下来存储进行大数据分析, 差不多这样一套东西, 一个小规模的数据中心大概需要1M USD来买服务器.

而渣当时也遇到了Cisco SDWAN解决方案中的大数据分析平台的问题, 整个平台原来采用ElasticSearch构建, 还有部分基于Spark的云平台,  基本上做不到实时搜索的能力, 同时遥测数据处理也非常慢.
对于遥测数据分析, 渣当时搞了一个分布式的流数据引擎, 类似于Flink在网络设备上进行预聚合, 然后在控制器层面通过ClickHouse替代ElasticSearch, 并使用MateralizedView来构建实时报表. 然后再利用Neo4j图数据库对数据依赖进行层次化索引和搜索,使得`整个平台的处理能力提高了1000倍`.

![图片](assets/e6521bd22292.png)

![图片](assets/b2fe5cbd787b.png)

![图片](assets/da549f3f4422.png)

流数据处理引擎以色列团队整合进了Cisco园区网的控制器中, 而Clickhouse的方案也由中国的团队整合到了Cisco SDWAN的控制器中. 当然对于这些数据的分析整理, 进行一些时间序列分析和预测的算法, 渣也和思科的Fellow合作构建了`Predictive Network`

## 4. 软硬件协同设计

其实很多人在协议设计时对硬件架构的理解非常的差, 举个例子, CF在设计SRv6的时候就犯了一个错误, 其实很多报文的处理非常难, 后来又打了一个uSID补丁,而国内在搞的一些gSID还有一些branch condition, 都是属于对于交换芯片和路由芯片的架构理解上不充分带来的问题.

通常设计一个协议的背后,在编码上要尽量考虑如何让网络设备降低处理的指令数, 对于一些复杂的有状态的处理,要尽量将这些状态数据可信的带在数据报文内, 降低访问内存的瓶颈, 举个很好的例子就是Google的PSP去解决SADB的密钥查询规模的问题.

还有一个网络中非常经典的Classification的问题:

![图片](assets/27e50efdfd6a.png)

如何对它们进行一个类似于大模型的Embedding映射到一个线性可分空间?

![图片](assets/01feea938a7f.png)

这就是意图网络, 降低ACL和流表的容量.

## 5. 课后作业:AI训练网络的路由协议设计

最后我们以当下热门的一个网络圈的话题作为结束的作业.

如何在端侧RDMA网卡能力不行的情况下, 解决当前RoCE在AI训练网络中的一些问题.  当然能像我们这样自己做芯片和协议并改进拥塞控制算法的毕竟是少数, 那么按照前面说的,

以前就写过一篇文章 [《包处理的艺术(2)---如何设计协议》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484550&idx=1&sn=0627d92a24590316a573af70f24cb3f0&chksm=f9961444cee19d5251efcac850ee9e3339090fc99cc454c496750b197a514bab7e0f22043003&scene=21#wechat_redirect)开头有这样一段话:

2018年OCP，当时还在阿里的依群总有个Keynote：The Challenge and Opportunity to Build a Transparent Network(大概4:00~6:00的时候)提到过BGP、BFD这些协议已经数十年以前设计的了，但是已经无法适应现有的网络需求。同时还有K姐在19年说过：“SDN这事本身错了”。

协议的设计背后有太多的考虑，网络中要给出一个解决方案非常容易，无非就是编码解码总有一种方法能够处理某些特定的案例，但是最终活下来的都是有很强泛化能力的协议。

事实上当流控无能为力的时候, 针对Hash极化的现象, 工业界要么去折腾拓扑搞多轨道, 要么去搞Hash函数, 例如Meta还把它取名为Enhanced-ECMP Protocol? 真是对协议设计有误解啊, 然后还有一个集中式的TE引擎做流量工程.

本质上Hash极化是ECMP引起的, 那么能不能直接在路由协议上做一些处理来解决这个问题呢? 动态产生一条精细的主机路由不产生ECMP行不行呢?  反正一个集群就几万卡几万条主机路由,操作系统又都是SONIC,随便在上面自己写个路由协议来替代BGP不行么?这才是真正的E-ECMP协议. 例如在这张网里面通过前面所述的分布式控制器和分布式TE, 然后对本地的Telemetry数据做一些处理,结合Segment Routing来改行不行呢?

当然这件事情的前提还是在流控无法解决的前提下考虑的, 但凡流控能解决就不要考虑去动路由, 但凡路由能解决就不要去动拓扑.

其实国内这个圈子的人还是很聪明的, 只是很多时候做事情不得方法, 一方面学术界离工业界很远, 工业界又只是少量几个寡头. 大部分人对于很多先进的技术还处在跟随和使用的阶段, 原创性较低. 工作又想出成绩, 因此通常在屎上雕花去解决一些不重要的问题, 或者是解决问题时考虑不周到导致技术演进出现瓶颈, 或者在错误的技术路线上坚持. 没事拍脑门子搞出来的东西, 大概就是理想`赛NV`, 现实就像`左江退`. 这是我写这篇文章的原因.

参考资料

[1]
RDMA over Ethernet for Distributed AI Training at Meta Scale: https://dl.acm.org/doi/10.1145/3651890.3672233
[2]
A Decentralized SDN Architecture for the WAN: https://research.google/pubs/a-decentralized-sdn-architecture-for-the-wan/
[3]
RedTE: Mitigating Subsecond Traffic Bursts with Real-time and Distributed Traffic Engineering: https://dl.acm.org/doi/10.1145/3651890.3672231
[4]
MegaTE: Extending WAN Traffic Engineering to Millions of Endpoints in Virtualized Cloud: https://dl.acm.org/doi/10.1145/3651890.3672242
[5]
https://github.com/ruta-io/: github.com/ruta-io/