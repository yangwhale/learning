# SDWAN技术演进趋势

> 作者: zartbot  
> 日期: 2021年2月2日 16:39  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485222&idx=1&sn=5fee3cda857962b6be7991a1014019f5&chksm=f99617e4cee19ef2a7a36ebc73ffe2d953c18d30c2d1e6671a7e301ab14a474e3d0894e1dbdb#rd

---

**题记：最近做了一个Ruta的demo可以参考：
**

https://www.github.com/zartbot/ruta_demo

最近有一篇文章讲**《**[**SDP vs. VPN vs. 零信任，后来者居上？**](https://mp.weixin.qq.com/s?__biz=MzAxMDA1NjMwMQ==&mid=2651768839&idx=1&sn=e363ea1fe4f68fb82fc8e2ce93683915&scene=21#wechat_redirect)**》**

其实普遍的反应都是VPN已经过时了，但是后来者是什么还不清楚，接下来我来分享一些个人的看法。

**1. 网络虚拟化和VPN技术的演进**

稍微列举一些有代表性的技术，让我们逐渐了解一下VPN技术的发展，很多技术例如L2TP**、EoMPLS、VPLS、AToM这些就暂时不提了。

![图片](assets/528975754dcd.png)

VPN技术来看，主要分两大块IPSec VPN和MPLS VPN。

**1.1 IPSec VPN技术演进**

最早的IPSec VPN仅支持点到点的连接，例如思科的crypto-map实现需要配置相应的流量触发加密，GRE over CryptoMap或者Cryptomap over GRE部分解决了匹配加密数据流的ACL**问题，基本上就构成了VPN的雏形，然后构建了SVTI技术。后期借助mGRE实现了针对Hub-Spoke的结构的DMVPN并且采用NHRP有了Spoke互联的特征。然后同时也伴生了远程接入VPN：EasyVPN、DVTI和后期基于SSL的SSL VPN。这个以后有机会逐个给大家讲。

**1.2 MPLS VPN技术演进**

最早是IPSilon提出的IP Switching技术，利用ATM的转发平面，放弃了ATM的复杂控制面，然后构建了以RFC1953 IFMP和RFC1987的GSMP**协议，然后1996年秋天思科发布了Tag Switching技术。IPSwitching技术以数据流驱动转发路径，而Tag Switching则是以拓扑触发标签路径，也就是说**MPLS技术是有拓扑依赖**的，这一点上也为以后的SR的蓬勃发展埋下了伏笔。

MPLS VPN的诞生很大程度上是为了解决传统的点到点线路的缺点构建的，运营商PE和P路由器构成了一个大型的分布式集群并通过VPN Label区分客户流量和进行灵活路由控制。

![图片](assets/c5c84f54dd55.png)

其后思科在DMVPN的Overlay上实现了MPLS转发能力和MPLS VPN over DMVPN的能力，也为日后的Viptela SDWAN实现埋下了伏笔。而另一方面在MPLS主线上，控制面协议出来了BGP-EVPN，因此逐渐有了一些基于MPLS封装的BGP-EVPN实现，同时随着数据中心的发展和VXLAN的出现也有了BGP-EVPN+VXLAN的实现。

Segment Routing横空出世，基于**拓扑无依赖的无环转发(TI-LFA)，以及源路由本身带来的FlexAlgo、SR-TE**等技术极大的提高了骨干网络的利用率和可靠性。于是又逐渐演进出一些基于SR的MPLS-EVPN和SR做underlay vxlan做overlay的数据中心互联解决方案。例如ucloud的工程实现**《**[**基于Segment Routing技术构建新一代骨干网：智能、可靠、可调度（二）**](https://mp.weixin.qq.com/s?__biz=MzUwOTA1NDg4NQ==&mid=2247487990&idx=1&sn=459239580e766dc88ee925adea53dbe4&scene=21#wechat_redirect)**》**

MPLS最大的问题自然是底层传输依赖了，因此随着IPv6的兴起，以及数据面可编程技术的需求出现了SRv6技术，特别是在国内SRv6推动很快。但是SRv6也有其内在的问题，那就是IPv6一个SID要消耗128bit，实在是有些长。所以紧接着又诞生了uSID技术压缩包头。

**1.3 SDWAN 技术演进与融合**

SDWAN最早的需求来自于互联网链路的可靠性可以媲美传统的MPLS VPN专线，但是价格只有专线的1/3或者更少。在Internet上构建一个Overlay网络逐渐成为一种物美价廉的选择。几乎所有的厂商都会用一个链接云、互联网和多个站点的拓扑，我们还是以MEF的为例：

![图片](assets/412b88160c32.png)

本质上SDWAN连接这么多站点又有不同的链路类型，那么我们就需要将每个站点后面挂的资源(网段、主机)提供一种寻址的方式。所以无论是思科还是华为在对于资源的描述上都采用类似的做法：Transport Locator(TLOC)

![图片](assets/54539c8bd17b.png)

当然SDWAN还需要解决一个大问题就是VPN的NAT穿越及加密密钥分发流程，毕竟IKE**建立Fullmesh连接时的RSA计算挺耗资源的，因此思科收购的Viptela做了一个创新，OMP协议源自BGP，但是采用了DTLS**替代TCP，控制器vSmart类似于RR，却增加了更多的策略控制逻辑，同时采用TLOC Route技术把密钥和NAT穿越的问题一起解决了。这种做减法的思维非常出色：

![图片](assets/4c088ea885c1.png)

另一个SDWAN的最佳实践是路由计算一定要分布式进行，集中式控制器计算路由的解决方案通常都会碰到Scale的问题或者信息不一致的问题，或者决策无法下放的问题。分布式决策可以伴随着FlexAlgo玩出很多东西来，所以Viptela SDWAN相对于基于DMVPN+BGP+NHRP的方式提供了更简便灵活的解决方案。

![图片](assets/76bd39c86c6c.png)

而转发平面，本质上又借助了MPLS over IPSec VPN的能力：

![图片](assets/a0729db4b3d9.png)

但是它也面临一些问题，例如策略如同在Overlay上配静态路由，多跳策略配置复杂的问题。

**1.4 SRv6 SDWAN**

SRv6最大的创新其实就在<Locator, Function,Args>的定义上，Endpoint Behaviors丰富了转发平面的功能，并且将很多复杂的由控制面触发的行为转移到了转发面，EVPN的实现也变得非常简单易懂。

```
 End                Endpoint function                    The SRv6 instantiation of a Prefix SID [RFC8402] End.X              Endpoint with Layer-3 cross-connect                    The SRv6 instantiation of an Adj SID [RFC8402] End.T              Endpoint with specific IPv6 table lookup End.DX6            Endpoint with decapsulation and IPv6 cross-connect                    e.g. IPv6-L3VPN (equivalent to per-CE VPN label) End.DX4            Endpoint with decaps and IPv4 cross-connect                    e.g. IPv4-L3VPN (equivalent to per-CE VPN label) End.DT6            Endpoint with decapsulation and IPv6 table lookup                    e.g. IPv6-L3VPN (equivalent to per-VRF VPN label) End.DT4            Endpoint with decapsulation and IPv4 table lookup                    e.g. IPv4-L3VPN (equivalent to per-VRF VPN label) End.DT46           Endpoint with decapsulation and IP table lookup                    e.g. IP-L3VPN (equivalent to per-VRF VPN label) End.DX2            Endpoint with decapsulation and L2 cross-connect                    e.g. L2VPN use-case End.DX2V           Endpoint with decaps and VLAN L2 table lookup                    e.g. EVPN Flexible cross-connect use-case End.DT2U           Endpoint with decaps and unicast MAC L2 table lookup                    e.g. EVPN Bridging unicast use-case End.DT2M           Endpoint with decapsulation and L2 table flooding                    e.g. EVPN Bridging BUM use-case with ESI filtering End.B6.Encaps      Endpoint bound to an SRv6 policy with encapsulation                    SRv6 instantiation of a Binding SID End.B6.Encaps.Red  End.B6.Encaps with reduced SRH                    SRv6 instantiation of a Binding SID End.BM             Endpoint bound to an SR-MPLS Policy                    SRv6 instantiation of an SR-MPLS Binding SID
```

当然SRv6还是遇到了很多问题，这里也希望国内很多厂商和运营商多思考一下，首先就是Underlay必须要是IPv6，这个可能国内基本上都开通了也没啥大问题。

另一个问题就是报头压缩的问题了，对于很多NFV来说，压缩算法复杂一点没啥问题，或者对FPGA而言，做起来也有足够的逻辑资源。虽然验证了一些芯片可以支持超过10层的G-SID标签，但是整体的转发性能影响有多少？**理论上一个cut-through的交换机对于报头的解析Cache是有限的，而G-SID的一些操作很有可能要求交换机做一两次Recycle才能完成操作，这样虽然支持了压缩，而转发芯片的性能被活生生剥掉50%，还不如简单的不压缩,  而且Replace Flavor对于转发平面的影响还是很大的。**

另一个问题是多云部署的时候，例如某些时候一个service-chain需要上云再回来，由于SRv6转发的时候源地址没有变更，那么云中继节点转发出来的时候可能因为URPF的策略丢弃报文。

![图片](assets/f5ffce8a24fe.png)

通常B如果是一个云Provider，其云网络的安全策略一定会阻止这样的行为。

**1.5 未来网络该何去何从？**

我们在此总结一下SOTA(State-of-the-Art)的技术优缺点：

![图片](assets/d0518cb37401.jpg)

**2. 技术融合的尝试**

在这些已有的优秀技术中做取舍真的很难，所以更多的是先融合再改变。一方面就是SDWAN和传统的MPLS网络融合，保护用户投资。

思科SDWAN您可以理解为一个定制化的转发面基于MPLS VPN over IPsec，控制面基于增强BGP协议并配置控制器的解决方案。第一件事要做的就是和传统的MPLS VPN专线融合，即SDWAN的节点充当MPLS VPN的PE和传统的MPLS VPN网络互操作《[运营商SDWAN融合MPLS VPN解决方案](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485095&idx=1&sn=0e24851d162c43f38437fdac68a1c507&chksm=f9961665cee19f73b9787515ceb18b146a9bcd04e9c451e8643909b04effc7faf2963c1232ee&scene=21#wechat_redirect)》。

![图片](assets/cb0c217c5f03.png)

另一种方式就是利用DHCP标记策略和SR-MPLS VPN互通

![图片](assets/d4e237eec214.png)

DSCP标记ucloud有个很好的定义

![图片](assets/91bf25fcbcf3.jpg)

但是这些技术也解决不了一个现实问题，现在的SDWAN在overlay上无法做到TI-LFA、FlexAlgo、SR-TE等功能，似乎最简单直白的方式是借用SR-MPLS的做法，这样都有了？

![图片](assets/b81b4118e989.png)

**
**

**2.1 SR over IP或者SR over UDP？**

注意到了么，上图中在SDWAN内构建了SR，本质上是SR over IPsec,其实就是SR Over IP(RFC8663)的另一种封装? 最根本的问题没有用上SRv6的优势：

![图片](assets/10d13c7ce2fd.png)

所以这也是我们为什么要重新定义一个SRover UDP的协议来替代SR over IP的MPLS标签栈的做法，就是要**用上SRv6的可编程能力，但是同时又要解决包头压缩的问题，并解决IPv4承载的问题，Ruta的数据面封装就是这样诞生的。**

![图片](assets/910930b26d46.png)

解决RFC8663穿越NAT的问题，和同时采用48bits的可编程能力，并添加Optional TLV的支持。同时又满足了IPv4网络传输的能力。携带Source Address:Port字段可以保留原地址，并且逐跳转发时修改源地址，解决SRv6的uRPF问题，该字段现阶段放在Segment-List之前，最终为了优化很多P4交换机芯片的处理会放置到Option TLV中，因为中继交换机根本没有看这个字段的必要，而Flow Tracing ID可能要用于作为一个全局的Hash key在多跳间做Telemetry correlation，要放在前面。

![图片](assets/f07aba5b2862.png)

SID-List消耗来看，我们以下LC1->Fabric1->Fabric2->LC2拓扑进行EVPN部署为例

![图片](assets/4473347bb235.jpg)

SID-LIST: 48bits * 4 = 192bits, 相对于SRv6的实现小了很多

[0]255.<1234>.End.DT4

[1]192.168.99.78:colorX

[2]192.168.99.76:color2

[3]192.168.99.75:color1

当然这个也是借助了IPv4地址的优势，而对于SRv6，我们还可以通过分布式锁提供全局分配24bits标签的能力来实现类似于uSID的转发。

**2.2 控制面演进**

另一个需要注意的事情就是SDWAN的控制面技术，特别是SDP对控制面策略提出了更高的要求，另一方面就是要更好的和应用程序安全授权及零信任框架整合。

![图片](assets/8e2d1ce53495.jpg)

你可以看到SDP的拓扑结构和SDWAN惊人的相似，唯一不同的就是SDP Client是在终端上或者应用的Sidecar上，那么我们如果把SDWAN做到SmartNIC上或者用户的VPN client上，然后SDWAN的控制器和IAM等身份鉴权系统互联，同时和K8S集群的ETCD共享信息不就搞定了么？所以这也是Ruta控制面采用ETCD的原因，相对于BGP，能够更加容易的和IAM、K8S集成：

![图片](assets/5e49a46740d5.png)

而Ruta本身就是基于SDP框架实现的整个系统，有支持App原生通信库QUIC-SR的Linecard模块，有支持移动设备的VPN客户端，也有传统路由器上的支持，也很容易做到SmartNIC上，当然各种ServiceMesh的SideCar也不在话下。

![图片](assets/bdb29065e038.png)

SDP的策略可以通过Optional TLV放入Ruta的SRH中，并且相对于SDP多了很多网络上的SR-TE的能力，进一步提升了整网的可靠性。