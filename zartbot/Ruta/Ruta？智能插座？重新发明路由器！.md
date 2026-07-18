# Ruta？智能插座？重新发明路由器！

> 作者: zartbot  
> 日期: 2020年8月20日 00:22  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484099&idx=1&sn=8d335fa402c8bee2844d200588885bab&chksm=f9961201cee19b17a9c9e96a54e37754296754231451a2e243be47438f021584652752e326b8#rd

---

Ruta是一个完全云原生的去中心化自组织路由器架构，可以通过容器扩容支持数十Tbps容量和数百万端口密度的超大规模路由器系统。

智能插座是Smart Socket的意思，有些半调侃的意味，做网络的人总喜欢以网络为中心，而做应用的人对于网络的理解就是一个socket罢了。如果说MPLS是一个2.5层的技术，那么SRv6是一个3.5层的技术。而SRoU则是一个4.5层的技术。到4.5层有它必然的业务属性，例如Webex抖音等应用的优化， userspace的管控，软件定义边界SDP的实现。其实马云老师以前讲的很好，云计算的本质就是要让它像电和水一样方便好用。而SRoU的目的便是构建一个智能的插座，智能的水龙头，帮助7层的应用更加灵活快速安全的部署。

但是我们还是先从重新发明路由器说起吧：

![图片](assets/b7d95e9726f0.jpg)

第一台路由器应该算是思科的第一款产品AGS了，而IOS软件则从那个年代一直用到了现在, 在那个年代，它就是Software Defined的，因为一开始它就叫服务器

![图片](assets/0c4ec0a4301c.png)

在很多年前卖路由器是一件非常容易的事情，只需要一张图就行了， 因为讲着不同的协议，所以需要连通， 那个年代是广域网协议百花齐放的年代。

![图片](assets/e05de7cff531.jpg)

当年某司的产品样本特别容易让人懂：

![图片](assets/b2198ea0af98.png)

但是过去了那么些年，路由器越来越难卖了，想了很久才发现根本的原因是路由器的架构已经30多年没有太大的变化了，无非就是入方向做查表，出方向做流控，然后规模大了有线卡和矩阵再到后来矩阵机框做集群，CLOS Fat-Tree搭集群罢了。

![图片](assets/d49fdd8bb097.jpg)

现在我们已经没法用一两句话卖路由器，一方面是无休止的带宽端口密度的竞争。另一方面是我们开始玩一些装饰器，例如集成业务路由器，汇聚业务路由器，智能业务路由器，智能广域网。。。经常被用户怼：“反正广域网和局域网都是以太网，为啥不用一个交换机或者一个防火墙解决问题？” 还有当年有些集成商最喜欢的问题是：“请简述6500/7600和GSR12000的区别”。有时候发现客户还真的想的明白：

![图片](assets/66469d2cb3be.jpg)

所以，为什么不重新发明一次路由器呢？把复杂的VXLAN/BGP-EVPN/SDN访问控制各种烦人的事情全部处理好，设计一个新的路由协议，构造一个新的框架呢？这也是三四年前，一个非常棒的PM同事给我说的：“OSPF/BGP都用了30年了，有没有兴趣做一个新的协议把路由器的生意扳回来？”，所以开始了这个项目：

**正经点，下面这图是Router不是 Pxxn Hub！哪个美工画的图！**

![图片](assets/c6d7af32f5d0.png)

从最根本的地方开始思考，路由器是什么？很多国家的语言发音都是音译的Router，而中文的博大精深又一次体现出来，真得感谢翻译它的人。路有路径之意，由则是“必由”之意。

1. 历史上的“分布式”路由器

路由器的本质就是把不同的网络连接起来，而路由协议本质上就是分布式计算协议使得多个节点相互协作完成互通，分布式一致性和协作整个路由器架构的基石。随着单机性能逐渐达到瓶颈，历史上有大量的负载均衡算法查表优化算法， 也有CRS-1/ TX-Matrix / NE5000e等多种集群架构：

![图片](assets/9e8d57a0d024.jpg)

时至今日数据中心也在用着这样的架构, 一个大的CLOS构成的交换网：

![图片](assets/4da1ae78cb20.png)

历史的架构大家可以参考以前的一个文：

《[“网络编程” 还是 “可编程网络”？](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484028&idx=7&sn=dee70a3294dfc6f6c71c2eca6bb3bdfd&chksm=f99612becee19ba85a342faf755516b99322d9a6873ae5f06d9195551984ab3c6411a02fb721&scene=21#wechat_redirect)》

但是三十多年过去了，却一直没有任何人对路由协议动手，只是几年前听阿里的依群总在某一年的OCP Summit上提起过。。。几个月前写数据中心架构的时候写过过如下一段文字：

BGP-EVPN用了很多年，但是它很多人忽视的地方。为了保证CLOS架构的直接通信, IBGP在CLOS下要RR肯定不好，eBGP又涉及到跨集群的ASPATH防环的问题（ASW和PSW使用不同的ASN,但是导入另一个ASW的时候又会出现AS Path loop detection），然后又人为定一些规则来解决。本质上在数据中心内部Overlay地址通告实际上只有K-V pair的需求，没有防环的需求，所以不得不说MPLS这群家伙真的很厉害，当年Insieme的时候很聪明的在DC内部采用了COOPS协议，也就是一个很简单的K-V pair 同步的协议。

随着下一代网络智能网卡和HostOverlay的部署，如果继续采用BGP通告到路由表中的信息变动会更加频繁，而且BGP Speaker的数据会增加数十倍，整网收敛速度会更加慢。所以至少在接入层和汇聚层采用ETCD等分布式数据库将成为一种趋势，也不用像阿里那样把ARP转路由宣告了， 直接SONIC里启动一个ETCD同步表就行了。SONIC基于ETCD的路由系统如何和K8S的ETCD互通这个也非常容易实现，看看哪个大厂能自己写了开源一个，他们不写我就过几个月忙完另一个AI Fabric的活后再来弄。

似乎大厂们没把它当回事，而几个月后还是靠我自己写完了（不过我自己也算大厂的呀。。。哈哈哈），并顺手写了一个RFC-Draft，具体的协议解析和设计思路我会在第二章详细阐述。

![图片](assets/faed16aa50c1.png)

2. 云网融合的新“路由器”

如今，我们面临的挑战是如何把Internet作为一个Fabric，建立一个多云的大路由器，连接世间万物，智能选择路径，绕开拥塞，维持高可靠性和高弹性。

![图片](assets/80e3f78122bc.png)

Ruta的设计思路很大程度上也源自于StarLink，StarLink通过低轨道卫星和地面站协同组网自组织路由的方式构建互联网，是一种典型的去中心化的路由技术。

那么Ruta的解决方案是在公有云和地面一些MEC节点放置一些路由转发节点，充分利用互联网的带宽，寻找到可用的低延迟线路和空闲线路，也是基于这种考虑在现有的IPv4互联网上实现了区段路由（SR）的数据面，在互联网上通过构建大量的转发点实现去中心化的自组织组网，例如我们最近在阿里云腾讯云全球20个AZ部署了20台Ruta 节点构成的集群，节点自动可以发现一些优质便宜的长途链路，例如腾讯广州到腾讯硅谷，阿里呼和浩特到阿里法兰克福，这样就可以保证全球任何地方200ms内可达，容量上则可以几乎无限制的扩容，这样就构建了一个完全云原生的路由器架构。

而新发明一个路由器无外乎控制面用什么协议，转发面用什么协议。Ruta项目转发面使用了SRoU，控制面则针对SRoU和传统路由的各种缺陷设计了一套分布式KV存储的路由协议。

```
控制面：draft-zartbot-srou-signalling.txt转发面：draft-zartbot-sr-udp.txt
```

**2.1 去中心化的路由控制协议才能成功**

从历史来看，几乎所有中心化的控制协议最后都以失败告终，SDN的故事在当年电话网上就以INAP智能网的方式出现过， 而我刚工作的时候也做过一段时间的IMS/NGN， 这些东西都伴随着复杂的信令流程退出了网络。相反一些OTT的语音视频业务却火的很，例如疫情期间的远程教育却没有VoLTE任何事情。

SDN的成功某种意义上也只是假象，很多时候只是多了一个API外壳包裹的CLI罢了，至于YANG Model一类的东西纯粹是没事给自己找事干。最终你所认为的无所不能的控制器在一定的规模下自己就会崩塌。传统的广域网可以汇聚成千上万个分支节点，而上了SDWAN，没有任何一家可以宣称自己可以管理上万个节点的。而SDN在数据中心没出事只是因为链路相对稳定，收敛快罢了，当你MEC大规模部署，特别微小型DC在网络边缘时，B一致性就成为一个大难题，直接影响到整网的容量。
其实路由协议从第一天起就是一个分布式一致性的算法。那么为什么不用一个好的算法倒推协议呢？

**2.2 云原生的网络架构**

从物理机到虚拟化，然后到容器，云原生再到Serverless。应用程序已经顺利的走完了它的演进流程。而网络的演进从物理机到NFV就戛然而止，虽然最近几年也出现了一些例如Calico和Cilium的项目，本质的原因是做网络的人对计算机体系结构的理解还存在严重的知识缺陷，所幸的是Service-mesh的架构使得一部分应用网关的功能从专有网络设备迁移到了Sidecar上，而HostOverlay的兴起也证明了这一点。但是在端侧，服务器端出现着积极的变化，而用户端却始终非常缓慢。

**2.3 把任何一个接入终端当作线卡，赋予他们选路的权利**

三十年前，我们设计路由器和路由协议的时候，只是因为当时CPU和内存的限制，所以做出了一个取舍：主机无需知道网络的状态，而网络应当尽力而为的传输。时至今日，我经常半开玩笑地说，你的任何一个手机的内存都是你家路由器的好几倍，CPU也牛很多倍，你还要把选路的权利交给它？无奈的只是IP路由协议设计的问题吧。试想你想从上海去北京，你会出门上公交车就给自己头上贴个纸条“去北京”，然后就不管了祈求别人帮你？人类从来都是要靠自己掌握选择的权利的。

![图片](assets/d3aa83cf498b.jpg)

当我们习惯了高德导航滴滴打车的时候，却发现自己的网络数据包居然连选路的权利都没有，真是一种莫大的讽刺。于是心中就默默的出现了这样一个念头，我要构造一个大路由器，任何一个终端都可以成为线卡，获得像高德导航那样的选路的能力，而同时也有一些特殊的业务，获得滴滴打车或者东航随心飞的能力~，基于这一思想，下面这个架构就诞生了。

![图片](assets/1c27b4b47f5a.jpg)

3. Ruta控制面解析（ETCD based EVPN Routing)

3.1 去中心化的控制面

传统的SDN/SDWAN做法一定要有一个控制器，或者一个控制器集群，但是SDN中最常见的一个问题就是Controller placement的问题，也就是说控制器放在哪特别讲究。** 因此以往两地三中心的部署模式中，SDWAN的控制器如何放置如何做冗余一直是个难题。而很多厂商对于分布式控制器的理解仅为横向Scale-Out，而不是完完全全的去中心化。**

因此放的太开了有些控制器数据库一致性会要求50ms RTT的约束，否则不一致出现后会产生脑裂（BrainSplit）。而且有些控制器连接会话存在先有鸡还是先有蛋的问题，Underlay链路都不通如何连接到控制器，带外线路又不存在只能带内的场景。

etcd 是一个分布式一致性键值存储。其主要功能有服务注册与发现、消息发布与订阅、负载均衡、分布式通知与协调、分布式锁、分布式队列、集群监控与leader 选举等。所以我们采用了ETCD来实现分布式一致性，同时伴随着标准的操作接口，降低了传统路由协议编码的复杂度，而且现阶段网络带宽已经足够大了，并没有太大的必要去精心压缩bit位，编码解码带来的汇聚收敛延迟可能比网络传输还大，而利用ETCD的接口，应用层也可以轻松的定义自己的网络SLA，对网络的编程更加容易。

**自组织的实现可以是大量的核心交换机路由器节点构成ETCD集群，而任何一个网络上的设备都可以构成一个Proxy，通过LinkLocal地址relay控制信息。这样又很容易的解决了Controller placement的问题。**

![图片](assets/99d9182cd5bb.jpg)

通过这样的方式，整个网络就可以避免使用控制器了，而策略也可以通过任何一个网管设备接入其中，下发policy KV pair。任何一个设备都不需要本地储存配置，所有的配置都在这个分布式的KV上。

控制信令的安全和传输效率则由TLS和gRPC保障了。

3.2 节点属性

对于整个Ruta系统，主要的节点类型分为：
节点类型
用途
STUN
需要有公网地址或者1：1NAT静态映射的地址。

主要用于帮助其它节点发现公网地址
Fabric
用于SRoU的中继转发节点. 它默认会根据经纬度等一系列信息主动的发起和其它Fabric之间的TWAMP测量协议。
Linecard

SRoU的头端和尾端，可以以多种形态存在，移动端的VPN 软件或者代理，家用路由器，AP，服务器端的智能网卡，应用的Sidecar

这些节点可以根据地理位置信息直接发起对目的节点Linecard的TWAMP延迟测量，也可也hunt就近的Fabric节点并注册到Fabric节点的链路上。
Analytic

主要用于watch stats，分析网络故障，为AIOps提供数据支撑等，它也可以作为分布式智能控制器帮助区域内节点进行策略优化等。通常我会用一些类似于Nvidia Jetson Xavier或者带GPU的大型节点放置于园区中用于做一些模型的推理。

当然在某司我有另一个项目就是直接在路由器和网络设备上运行决策树模型，而神经网络模型通常需要大量的矩阵计算还是offload出来比较好。

主要Ruta对节点的定义，因此对于每个节点我们扩展了一些属性：

**1. SystemName：**这是一个比较直观的名字，类似于ASCII String的RouterID，比RouterID更加直观便捷，谁也不会去想1.2.3.4的RouterID代表什么，取而代之的是一个要求不能重复的String。

**2. SiteID：**这个类似于BGP的ASN，主要是用以做一些site level的策略使用的。

**3.SystemLabel：**用来压缩SID长度或者和MPLS网络对接时的标签，也可以用这个标签映射成VPN客户端的IP地址。它总长24bit，通过分布式锁分配，具体后面讲。

**4.Location：**GPS的经纬度咯， 主要用于一些A*的选路算法做约束，或者Fabric之间做随机互联的时候的权重。大概的节点信息如下图所示，例如有个节点叫

SystemName: TE_Beijing 代表腾讯云的北京节点,  ASN: 101, SystemLabel: 6, Location:[39.97,116.07]

![图片](assets/91d413d0760f.png)

3.2.1 节点标签分配

传统的标签分配通常会有一个标签分配进程来控制分发， 而我们是一个完全分布式的协议，因此用到了ETCD的分布式锁。先利用"/lock/systemlabel"申请分布式锁，原理上和摇号选房差不多，看revision号码，谁小的谁拿到锁先选，选完了释放。然后释放锁之间，节点会获取所有的标签表，并找出最小空闲标签，分配个自己，并使用“/systemlabel/X” 注册到ETCD。

3.2.2 节点注册

节点注册采用如下KV的方式注册，注册前需要获取”/node/role"检测是否有重复的systemname.

```
Key="/node/<role>/<systemName>"Value="SiteID,SystemLabel,Lat,Long"
```

3.3 SRoU-Locator及其路由

那么接下来就是对广域网接口和链路进行抽像了，这一点上借鉴了思科的Viptela的技术，类似于TLOC那样的编址方式构建成了SRoU Locator，简称SRLOC，我们的编码方式如下：

![图片](assets/e80f90dbf9a5.png)

第一行为区分SRLOC传输层的关键字，第二行为扩展信息，主要用于宣告公网地址和带宽信息以及linkstate中的bandwidth usage和选路时的metric，这些信息以如下KV pair的方式更新

```
Key="/service/<systemName>"Value="RLOC1，SRLOC2，SRLOC3...."SRLOC="SystemName|Color|LocalIP:Port|PublicIP:Port|LocalInterface|TXBW|RXBW"
```

3.4 Node KeepAlive

通过借用ETCD lease机制完成，在Prototype中我们采用了两级Lease机制，即节点注册信息这些健康状态的信息我们采用一个短超时的lease，而真正控制路由的信息为了防止大规模网络下的抖动和频繁选路以及大量的数据更新，我们采用了长超时的lease机制。Keepalive机制在节点和ETCD实现。

3.5 Linkstate

BGP LinkState又是特别复杂的东西，消息编码那么烦导致解码也烦，其实讲真没必要在意那么点带宽。所以我们直接采用了gRPC在ETCD里通告链路状态的方式。然后任何一个其它节点都可以watch这些prefix做本地的路径计算，analytic 节点都可以消费这些数据转存到其它数据库中，为AIOps积累长期的数据。当然字段上还必须包含过去一个周期的underlay接口的带宽使用率等数据， 我懒暂时不想写。。。

![图片](assets/175fca9aa7ae.png)

然后可以通过ElasticSearch或者任何其它的软件随便画个图，我承认我是懒，而且数据量不大直接丢elk了，例如我们可以看到腾讯广州到阿里SJC的性能：

![图片](assets/7dac9772b383.png)

更多的信息则是公有云阿里10个AZ和腾讯10个AZ互联的性能，具体的数据下一次发文的时候再来分析就好。这些也是我们Linecard选路算法的基础数据

![图片](assets/1242840825ab.jpg)

大概几个月前我也用另一套SDWAN做过，有很多不满意的地方，最终还是用自己写的TWAMP over SRoU完成了。

![图片](assets/595116a1e9c4.png)

![图片](assets/21d521234ec0.png)

其实这也是一个很好的副产品，通常您也可以直接在数据中心交换机的SONIC上开容器放置，它可以自动帮你完成fullmesh的probe，probe的频率上来看，采用一秒一个probe，100s统计一次的方式，协议算法和TWAMP一样的，计算T1~T4时间差获得单向链接延迟。当然你如果觉得fullmesh麻烦，也可以做类似于BGP AS-PATH那样，仅允许SiteID不同的节点相互probe，这样就恰好匹配了整个DC switch的CLOS拓扑，反正你Spine是一个AS号，Leaf是一个。另一方面我也常用它来监控K8S集群，因为我同时会上报节点CPU和内存的信息

```
Key="/stats/linkstates/SRC_SRLOC->DST_SRLOC"Value="TWAMP measured jitter/delay/loss result and underlay interface load"Key="stats/node/SystemName"Value="CPU,Memory usage"
```

3.6 安全加密

这也是一个不可避免的话题，如果分享密钥。首先所有的控制信令通过TLS连接ETCD，权限分配等都可以依赖于ETCD的安全机制解决，而数据面则需要共享一个对称密钥就好，同时还要设计好rekey机制和密钥作用域机制，传统的IPSecVPN采用了复杂的IKE机制交换密钥，最终影响了整网的性能，而且IPsec随着历史的发展支持了多种加密方式，有些已经不安全了，IPsec协议编码也有很多问题，但是整个行业有些人为了自己的饭碗和懒，总是利旧，该推到的不推到，不该推倒的乱来。

密钥采用在ETCD里面通告KVP的方式分发，作用域分节点/SRLOC/Session，如下所示，两个密钥是在Rekey时发生的，更新一个新的密钥的同时还维持老的密钥一段时间，这样对端发现Key2更新用Key2加密就好，而本端先用Key2尝试，如果Key2解密失败再用Key1解，overlap key的时间放在10分钟以内就好，总归整网会同步的。

```
Key="/key/SystemName" Value="Key1,Key2"Key="/key/socket/SRoU_Locator" Value="Key1,Key2"Key="/key/session/SRC_SRoU_Locator->DST_SRoU_Locator" Value="Key1,Key2"
```

另一方面SRoU初期设计的4.5层智能socket的实现是基于QUIC协议的，即将QUIC协议报文放置于SRoU Payload中进行传递，server 生成一个质数 
![图片](assets/24d776b31710.svg)
 和一个整数 
![图片](assets/50750e1acb3b.svg)
 ，其中 
![图片](assets/2259738d03f4.svg)
 是 
![图片](assets/d65082d41832.svg)
 的一个生成元，同时随机生成一个数 
![图片](assets/bad1f407c5a0.svg)
 作为私钥，并计算出公钥 
![图片](assets/a78802474266.svg)
 = 
![图片](assets/13cefa384792.svg)
，将 
![图片](assets/dc30fa9fbe51.svg)
 三元组打包成 
![图片](assets/8619a4594fec.svg)
 ，等待 client 连接。因此我们也可以将ServerConfig放入ETCD中辅助客户端实现0-RTT。

![图片](assets/6db655525423.jpg)

3.7 Overlay路由

Overlay路由上为了和已有的很多BGP-EVPN设备互操作，Prefix我们采用了EVPN的机制，初期实现了Type-2和Type-5的路由。KV pair编码机制如下：

```
Type-2 EVPN Route Key="/route/2/exportRT/RD/MAC/IP" Value="VNID/SystemName/PolicyTag"Type-5 EVPN RouteKey="/route/5/exportRT/RD/IPPrefix/IPMask" Value="VNID/SystemName/PolicyTag"
```

将Export RT作为key主要是很多节点可以根据自己的import RT去watch prefix，而RD在SRoU中采用的是SystemName+VNID的方式，如果和其它节点BGP互操作可以采用Router-ID的方式替代，而nexthop属性中，主要就是SystemName，类似于SR的Prefix-SID机制。

3.8 选路算法

通常根据我们的测试，源Linecard因为对目的Linecard节点的Probe几乎总是一个最好的选择，即直连连接，这样也会省掉较长的Segment List，仅有在网间拥塞故障后，Linecard需要借助Fabric节点做路径Relay时会压入多个标签栈，通常根据我们的测试最多2~3跳。路径选择的算法上则相对于传统的路由协议有很多优势， 因为通过终端Linecard watch ETCD ”/stats“你可以获得尽量详尽的信息，A*一类的算法也成为可能，毕竟路径满足SLA就好，不管它快不快。所以这是第一次给了终端选择的能力。

同时我们为每个节点都准备了geo-location的信息，这样也可以限制终端的搜索域来加快计算速度。同样即便是不需要Linecard节点，一个简单HTTP/3 QUIC的通信流，也可以Server动态的计算一个最近的CDN节点做Segment Routing，这个我们第五章会讲。

3.9 线卡配置

任何节点的配置都是采用yaml文件的方式，任何节点也可以用于容器中，具体配置如下：

```
role: fabricsiteID: 101systemID: TE_Virginialat: 36.75long: -76.04controller: [etcd1.test.com:443,etcd2.test.com:443,etcd3.test.com:443]srloc: [ INET|1000|1000|10.200.0.7:443|]  //Color | TXBW | RXBW | LocalSocket| PubSocket(Optional)interface: [ 1234|veth0|1.0.7.1/24| ]  //VRF | Local Intf | IP/Subnetroutetarget: [ 1234/1:1/1:1 ]         // VRF | importRT | export RTcert: ./cert/client.pemcertkey: ./cert/client-key.pemcacert: ./cert/ca.pem
```

4. Ruta数据面解析（SR over UDP）

4.1 为什么不是SRv6

IPv6搞了这么多年，大概就属于RFC1925说的第三条了吧：

```
With sufficient thrust, pigs fly just fine. However, this isnot necessarily a good idea. It is hard to be sure where theyare going to land, and it could be dangerous sitting under themas they fly overhead.
```

一个技术花了10多年还没很好的落地，一方面是别人利旧和兼容性的原因，另一方面协议的复杂度也是一个问题。而IPv4和IPv6的争论还会持续下去，AWS一直在疯狂的收购IPv4地址就是一个例子。而对于我们而言，最好的办法就是搁置Layer3的争议，在Layer4上去融合IPv4和IPv6网络这也是RFC1925上的一条军规，任何事情没有加一层overlay解决不了的

![图片](assets/5f9224793038.png)

**应对新****冠****疫情和接下来的经济衰退，很少有国家能拿出来大量的资金升级网络的钱， 因此 IPv6的全球升级可能会被疫情极大的延缓。而上云的速度随着移动办公和在家办公等需求会越来越强烈，在这种基调下才诞生了SRoU。****当然，SRoU并不是一定要在IPv4上使用，它依然可以在IPv6上使用，相对于SRv6，由于Header打在UDP Payload中，还更具灵活性，这样不需要对Kernel的升级，应用程序就可以搞定。**

**搁置IPv4和IPv6的争议，在Layer4上构建IPv4和IPv6的互联互通，并为Internet提供一些新的SR技术进一步的释放带宽。****当然现有很多运营商已经建成了SRv6的网络，我们后面会有一个例子讲如何使用SRoU构建SRv6融合场景**

4.2 SRoU概述

SRoU本质上就是把SRH放入UDP Payload中，同时针对IPv4网络中NAT的问题，实现了NAT穿越和中继节点无状态转发等功能。报文封装格式如下:

![图片](assets/a2442dc9fa9f.png)

报文头中的Magic Number，当用于QUIC传输时，它为0x0，这样正好区分开QUIC的Long/Short Packet Flags，而针对IPSec封装时取值为0xF，并规定IPsec SPI不分配0xFxxxxxx的SPI就好。通过这样的方式就可以在原有的IPsec隧道或者QUIC会话中添加SR能力了。

4.3 SRoU vs VXLAN

VXLAN也可以通过一些方式实现NSH，GPE也能实现针对IP包和以太包的封装支持。但是VXLAN编码方式过于复杂，特别是NSH效率低下，和SR互操作能力很弱，NAT穿越能力也没有，加密传输也没有标准，因此我们选择了放弃VXLAN，而SRv6中的可编程能力也是我们看重的，因此我们选择使用SR技术构建UDP overlay来替代VXLAN。同时针对IPv4的情况，我们也定义了一个48bit的SID：

![图片](assets/be7ce0f595dd.png)

Segment-List相对于SRv6也更加节省，每段针对IPv4仅有48bit（公网IP和Port)的组合，并且我们还提供了和SRv6 cSID兼容的处理方式，可以通过类似于cSID的机制打通IPv4和IPv6网络，做到底层无论用什么传，overlay都能保证统一的策略。

4.4 FlowID字段

协议定义是变长的，在报头也有一个Length字段帮助ASIC定位，但工程实现上可以定义为一个64bit或者128bit flowid，它的用途一方面是用于做基于流的监控和分析，类似于Google Dapper的TraceID，而SID-List则可以作为Parent-ID使用，

另一方面也适合于DetNet的处理，可以进一步控制延迟。而当我们在IPv4公网上架设服务时，也需要一个token机制避免其它人使用您的带宽，或者注入攻击流量，这也是一个接入控制的ID，处理机制和互联网公司的Token机制类似，验证+黑名单数据库即可。

另一方面它可以用作一些智能的BRAS业务。传统的BRAS设备需要与终端建立复杂的通信机制，业务流程上来看极大的影响了BRAS的容量，而终端的光猫需要一些增值带宽业务，PPPoE协议又限制了网络可编程能力，SRv6又要Kernel升级导致很多光猫维护升级代价过大，通过SRoU flowID字段可以在接入网为IPv4网络的时候，构建灵活的Userspace支持的编程能力，通过UDP隧道实现和BRAS的互通，同时可以把SRv6的Binding SID直接放置于FlowID字段中，BRAS行为更加简洁。

4.5 Source Address字段

这个字段用于头端封装自己的公网地址，或者第一跳Fabric节点帮助将接收到的IP包的公网地址和端口压入这个字段。主要目的是让远端的隧道终点溯源同时执行逆向对称路由或者可优化的非对称路由。

第一跳压入还有另一个考虑，在报文中间插入header会是一个非常麻烦的事情，中继Fabric节点如果需要插入源地址信息在UDP头中，如果用DPDK实现则需要把mbuf 拆开，然后insert，或者把很多bytes 前移后移空出一段。而第一跳需要LineCard识别公网也很麻烦，搞STUN代价也有点大，所以第一跳还可以什么都不干，allocate一段source address 空间置0就好，而Fabric节点的第一跳设备可以自动的把源地址源端口拷贝进入这个字段即可。

4.6 SRH

定义和操作方式和SRv6完全一致，只是针对IPv4的UDP overlay场景添加了一个48bit长度的SID支持， IPv6网络中可以采用UDP固定协议端口的方式沿用128bit SID即可。所有的字段要直接拷贝到SRv6 SRH中也非常容易。同时正如前文所述，考虑到IPv4支持可编程的能力，我们将SID中高位为255.0.0.0：xx的整个一段用于执行End.X的可编程处理能力，例如我们的线卡节点接收到一条Type-2EVPN路由，数据包的Segment-List[0]的封装即为

```
255.<VNID_24bits>: End.DT2U(16bits)
```

定义上End.X正好16bit，VNID也刚好24bit，而255.0.0.0/8的地址段正好又没人用，就这样我们在IPv4网络上也可以实现类似的可编程能力。

5. Ruta使用场景

5.1 Mobile SDWAN

Ruta创造性的地方在于将网络节点退回到一个帮助者的位置，而给与终端更多的选择权。例如通常SDWAN的设计以网络为中心，通过边缘路由器帮助用户选路，当然一个接在这种路由器WiFi后的终端也不知道有几条路，而Ruta的做法是让路由器把广域网路径信息告诉给终端，终端你决定好了，我傻快傻快的转发就好了。

![图片](assets/f9a5443e30f8.jpg)

在疫情期间，我们发现在疫情期间大量的用户在家办共滋生了对VPN客户端和SDWAN互操作的需求，希望VPN客户端也有一定的灵活选路（Dynamic Split Tunnel)的能力，也就是业界最近开始忽悠的SASE，而通常VPN采用的是DTLS，而SDWAN采用的是IPSec，两者从传输层和控制层上都需要融合。

Ruta创造性的把VPN客户端抽像成为一整个路由器集群的线卡，成功的解决协议互通的问题，例如我家的工作站通过在全球公有云构建fabric，可以保证我全球各站点稳定的200ms内可达。

5.2 软件定义的边界,Ruta也是一个完全分布式的防火墙

从安全厂商的角度来看，Zero Trust和SDP(Software Defined Premetier)的产生，增加了对终端的策略控制的需求。

![图片](assets/47ba4689bd89.jpg)

而事实上我在设计Ruta时便考虑到了这种情况，也就是以前设计的基于自然语言的网络意图工作提到的Security Label: **>>****[意图网络的语言学思考](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484028&idx=5&sn=c8de8c1fe6fa947ca1ff98c2f6b34d20&chksm=f99612becee19ba8bc32862780f058d7a94c7380aa6104a774e07c717086fce936a3a192df6a&scene=21#wechat_redirect)<<**

![图片](assets/ea5074825140.png)

在SRoU的控制面也专门设计了相应的keyprefix用于提供策略和身份认证：

```
Key="/control/RT/2/SRC_MAC/SRC_IP/DST_MAC/DST_IP" Key="/control/RT/5/SRC_Prefix/SRC_Mask/DST_Prefix/DST_Mask" Value="Action" /"SR Locator list"Access ControlKey="/token/permit/flowid" Key="/token/block/flowid"IdentityKey="/identity/owner/deviceid"value="role,policyprefix"
```

SDP的文章也可以参考：<[取代VPN？再见NAC？论SDP](https://mp.weixin.qq.com/s?__biz=MzAxMDA1NjMwMQ==&mid=2651761975&idx=1&sn=d45f8c1cb501ea308c6c8b097b00e802&scene=21#wechat_redirect) >

5.3 数据中心HostOverlay &CNI

您可能注意到了前面的一张图，我通过这样的Ruta Linecard将阿里和腾讯的20个VPC完全fullmesh的连接起来了，相对于报文的封装和解封装能力，性能和中间节点的操作和VXLAN类似，同时比VXLAN EVPN更容易的地方是不需要两个VNID来区分L3和L2，因为报头的SRH定义了End.DT2U和End.DT4，更加高效。中间节点转发也不需要查表，直接shift sid就好，很容易实现线速，特别是一些软件平台使用BPF实现中继转发非常容易。

而我当时为了测试方便，一开始就使用了Veth作为我CE侧的接口，未来一段时间可以将它扩展一下支持CNI，直接就可以和Cilium和Calico开干了，传输上比Cilium和Calico更加高效稳定。

另一方面随着Userspace对End.DT4支持，可以手机等移动端设备直接构建加密隧道和VPC内部的主机进行通信，避免了传统的Transit Gateway的麻烦设计。

5.4 可编程的CDN：Smart UDP Socket

一开始做Ruta的时候其实叫QUIC-SR，本质上就是想帮一些互联网公司调度广域网流量的研究项目，例如Webex，抖音等应用，它的流量大量的东西向的，并且还要对网红主播做流量优化，网络中的热点又有极大的不确定性。传统的CDN业务无法满足需求。

而QUIC即将成为HTTP/3的标准协议，因此我们有理由对它进行可编程的扩展实现流量工程。HTTP/3 QUIC的介绍可以参考下文***:<[HTTP/3](https://mp.weixin.qq.com/s?__biz=MzAwNjI5MTYyMw==&mid=2651498557&idx=1&sn=c8d28aff047c8e151ea7d11f5c9b8a76&scene=21#wechat_redirect)>***

因此我们最早的SRoU的版本就是基于QUIC扩展的

```
我们基于QUIC-GO提供了一个可编程的Sockethttps://github.com/zartbot/draft-quic-sr/blob/master/prototype/quic_go.diff
```

应用侧可以很容易的找到一个CDN转发流量，即在socket option中添加某个CDN的服务端口地址即可：

```
sidlist:=[]string{1.1.1.1:2345,2.2.2.2:4567}flowid:=[]byte{0x1, 0x2, 0x3} //tokenisDup:=true //request packet duplication(for DetNet case)session.SetQUICSR(sidlist,flowid,isDup )
```

前一个参数为SID-List，后一个参数为FlowID,并且可以定义一个Duplicate Ratio，在高丢包链路的情况下可以指定报文复制比例. 这样给了应用的人很便捷的编程能力。传统的CDN需要源站推送一些资源到CDN节点，而现在直接可以根据北京用户访问的Session，在应用侧配置QUIC-SR的属性，就可以将源站到北京用户的流量引流到北京CDN节点，其它地域也类似，同时考虑到网络边界丢包等情况，还可以设置isDup=true来触发报文自动重发，而远端节点可以根据QUIC本身的Seq执行去重的功能，某种意义上SRoU为实现DetNet也打好了坚实基础。

![图片](assets/9d7884779dcf.jpg)

5.5 Proxy网关模式：用户侧Sidecar

其实这是一种靠近用户侧的SideCar的技术，用来实现车联网随路计算等功能，通过SRoU的寻址能力和QUIC的快速可靠传递能力混合，可以在用户边缘侧构建Linecard和Fabric实现快速的访问，而代理模式则是用户端可以维持TCP会话不变，而通过CDN端Sidecar提供GraphQL或者RestAPI的支持，而后端数据则采用QUIC-SR socket访问其它节点。

![图片](assets/d7a25f16face.png)

5.6 5G+WiFi多归属组网

移动端也有多网络接入的能力，通过SRoU也可以提供多路径负载均衡：

![图片](assets/bd87c2ed2f78.png)

5.7 BAS和商用政企专线

我们也制作了OpenWRT的Linecard版本，初始阶段BAS可以为终端的猫DHCP提供一个地址，并设置ACL仅允许猫访问ETCD，然后通过ETCD分配给光猫不同的FlowID（IPTV/政企专网等业务SID）后，终端可以根据这些FlowID将数据包封装成为一个UDP报文发送到BAS,BAS也接受ETCD控制开放ACL，匹配放行FlowID，并可以将FlowID直接映射成SRv6的Biding SID传递到远端。

而我们同时也制作了基于OpenWRT的AP版本，这样很多传统的设备不需要加载我们的线卡软件，只需要连上WIFI也可以获得SRoU的能力，只是把智能决策交给了第一跳的网络设备。

![图片](assets/8b6d4f3986ea.png)

5.8 网络版滴滴打“车”

网络的拥塞其实一直是运营商之间的带宽限制导致的，其实有些企业就可以通过员工自己家里申请电信联通双宽带，构建SRoU Fabric节点就可以轻松的为企业网络提供网间加速服务了。如同个人网约车提供租车服务一样，本质上一个承载人，一个承载数据， 网络版的滴滴打车不就实现了？凡是有双线的终端用户，只需安装一个特殊的小的路由器便可以盈利。也扩大的网间互通的能力。

5.9 StarLink 路由协议

大家有没有想过StarLink是如何组网和路由的，其实Ruta的项目和StarLink类似，StarLink是在低轨道提供路由服务，卫星自组织的方式构建一个覆盖全球的路由网络。而Ruta则是利用公有云和大量地面中继站的方式提供覆盖全球的低延迟服务。这些都是去中心化的组网模式， 真想有兴趣去探索一下StarLink的路由方式，我也是受到这方面的启发，逐渐利用公有云的资源构造自己的StarLink，**我国如果实现StarLink一类的业务一定也可以用到Ruta这样的路由协议。**

**别问我为啥写这么多，我只是在帮你们技术扶贫而已，看到国内很多做网络的人炒作智能网络未来网络，想定义很多标准。的确是，网络的门槛非常低，谁都可以定义一个封装格式，但是正如RFC1925那样说的：**

**Some things in life can never be fully appreciated nor understood unless experienced firsthand. Somethings in networking can never be fully understood by someone who neither builds commercial networking equipment nor runs an operational network.**

****

****

****

5. 后记

网络本来就是一系列矩阵运算，封装是一次，解封装是一次，加密是一次，解密又一次，流表映射是一次，例如为了方便classification，本质上打Tag的目的就是做一个矩阵的基变换，但是很多人比较蠢编码的时候没做好，使得操作的矩阵没有逆，例如TrustSec和ACI的EPG就是，只标记了Source Group，没有Dst，没有更细颗粒度的维度，也就自然没有逆矩阵了。

![图片](assets/052f129e3349.jpg)

而数据包本身的编码需要大量的思考和工程实践，在某司做全球最复杂的路由已经十年多了，这些思考和多业务的处理慢慢的汇聚成了Ruta，你可以看到它EVPN的影子，也可以看到VXLAN的影子，可以看到语音SBC的影子，更能看到IPsecVPN的影子，还有SR的集成互通，还有一些分布式人工智能，利用线卡软件规划路径安全检测等，这一系列的能力都在过去十年被研发虐被产品虐被客户虐的过程中慢慢的诞生出来，十年磨一剑吧。

等写完全文的时候，回到开头，又加上了如下一段乔布斯老爷子话：

**"Design is a funny word. Some people think design means how it looks. But of course, if you dig deeper, it's really how it works. The design of the Mac wasn't what it looked like, although that was part of it. Primarily, it was how it worked. To design something really well, you have to get it. You have to really grok what it's all about. It takes a passionate commitment to really thoroughly understand something, chew it up, not just quickly swallow it. Most people don't take the time to do that."**

最后以一张思科80年代末期的广告词结尾吧：

![图片](assets/1e5ef4086a95.png)

**不用谢我，我只是技术扶贫**

但是有没有云provider来给我经济扶贫呢 XD~