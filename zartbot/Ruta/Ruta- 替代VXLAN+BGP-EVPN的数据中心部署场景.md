# Ruta: 替代VXLAN+BGP-EVPN的数据中心部署场景

> 作者: zartbot  
> 日期: 2020年11月1日 11:16  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484414&idx=1&sn=6c36bde5399a0cd6fa8016b421dd90d7&chksm=f996133ccee19a2a56d397b77602faa451efdb28b9f573a452d7ddd6f61e291b415a370bd0d1#rd

---

很多人以为Ruta只适合于广域网，有个朋友今天咨询我数据中心架构，我想说广域网的事情都能解决了还怕局域网无法解决？在几个月前的某文中已经提到:

BGP-EVPN用了很多年，但是它很多人忽视的地方。为了保证CLOS架构的直接通信, IBGP在CLOS下要RR肯定不好，eBGP又涉及到跨集群的ASPATH防环的问题，然后又人为定一些规则来解决。本质上在数据中心内部Overlay地址通告实际上只有K-V pair的需求，没有防环的需求，所以不得不说MPLS这群家伙真的很厉害，COOPS协议做的很好。随着下一代网络智能网卡和HostOverlay的部署，如果继续采用BGP通告到路由表中的信息变动会更加频繁，在接入层和汇聚层采用ETCD等分布式数据库将成为一种趋势，直接SONIC里启动一个ETCD同步表就行了。

zartbot.Net，公众号：zartbot[下一代数据中心架构-1：从农民工看装配式建筑的视角谈起](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484099&idx=6&sn=2fbc017d61288d2bb1cf1634a1d73274&chksm=f9961201cee19b17ec106e020cebd0dcbaa79e1d6a574de7c9492bc3cca5a2f9efb58169342f&token=319600774&lang=zh_CN#rd)

反正双十一期间各个云都封网了，等过段时间把关于Ruta的论文写完了，我把代码开源了大家再一起玩玩，某云也别搞啥集中式BGP了，既然下定决心用白牌和SONIC了，那么路由协议也再升级一下？

Ruta标准的解决方案分为两块，数据面为SR over UDP并支持QUIC和IPSec的扩展，控制面为基于ETCD实现的 EVPN+LinkState+TWAMP。**事实上你也可以采用SRv6作为数据面使用Ruta控制面的方式部署，毕竟现有很多交换机都支持SRv6了。**

从数据面看，数据中心内有逐渐使用SRv6 with end.DT4提供更有效地流量调度来替代VXLAN的需求，而且很多云已经有IPv6的underlay了。Ruta在数据面在UDP协议之上，因此提供了IPv4/IPv6支持，相对于VXLAN也有完善的end.DT4等SRv6可编程能力。

但是从控制面来看，SRv6的开通和动态流量调度可能还需要BGP-LinkState并集成TWAMP来实现链路测量，BGP的负担进一步加重。同时应用侧通常对于网络资源的调度，例如某些关键业务保障等需求对BGP的可编程性也提出了挑战，毕竟应用工程师乱写个BGP-FlowSpec都会导致一个大事故，BGP的消息传导机制决定了故障隔离相对困难。

另一方面是在5G/MEC的场景下和HostOverlay场景下，沿用BGP会出现大量Scale的问题，已经有一些大规模的部署出现事故征兆了，具体内容太敏感就不多说了。

Ruta在协议规范中就定义了Fabric Node和LineCard Node，对应数据中心就是Spine-Leaf而已，对于数据中心是否实施SR，有很多人持怀疑态度，事实上过一些年，51.2T基本上就可宣告单芯片密度上限了，看看现在25.6T的Chip size和功耗，未来一些年Spine交换机虽然容量和接口带宽还可以进一步升级，但是最终会面临端口密度的限制，CLOS拓扑在端口密度的约束下会走到尽头，XD-Torus异构拓扑必定会出现，因此在数据中心内部使用源路由是无法避免的。

另一方面Ruta协议规范上对于路由的通告也使用了标准的和EVPN兼容的方式，同时使用Underlay的TWAMP测量协议来实现Linkstate协议，同时作为BFD和链路质量报告。并且根据数据中心运维的需求，将常用的Telemetry参数（接口带宽/CPU/内存）也整合到了Ruta的控制面。

![图片](assets/80cb380ddf06.png)

有这样一个现成的跟你们都考虑好了的下一代数据中心路由协议，不香么？

**另外预告一个文章<The Art of packet processing>,关于RTC还是Pipeline，准备抽空（估计要到12月了）好好整理一下，其实很简单的几句话，Packet是datastream，内存/buffer是snapshot of datastream，RTC和Pipeline的争论其实就是流计算或者批计算的争论，Fastpath和Slowpath的实现本质上就是microbatch。**

**这些问题想清楚了以后，流批一体才是关键了。其实无论是智能网卡或者是各种可编程交换机，无论是SR还是其它，其实本质的诉求就是对packet assign pane，然后window based batch processing，所以你们看VPP为啥要Vector Processing，但是很多场景VPP干不了，例如DPI/DDoS/Firewall等，这些场景又需要flow based assignment（5 tuple hash）然后RTC，但是FBD遇到大象流PE死了，遇到太多老鼠流Ram死了。最佳实践是什么？**

**可以私下跟我勾兑技术扶贫事宜。**

**
**

BTW，前几日有位网友在后台留言，说某个东西刚回来，你怎么说在你实验室太久了。我回复的时候以为说的是Catalyst 8300/8500 Edge，这个的确是放了一两年了，作为这个产品线的TME我准备这个月底组织一个中文的线上发布会。

是我过了几天才想他可能问的是Q200，可能有一些误解，在此向他道个歉。