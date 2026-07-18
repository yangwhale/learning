# Ruta主机网络:纯Golang应用也能高速收发包

> 作者: zartbot  
> 日期: 2021年9月21日 12:44  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486444&idx=1&sn=445969ab08cde98c084b4c0e3226b206&chksm=f9961b2ecee19238c81586feb11b7bc4f072d15835201d609db66417a63c1b71d60ee2f99590#rd

---

先说答案，不需要什么智能网卡什么DPU，就纯Golang应用能UDP收包29Mpps！这个东西也将成为Ruta Host网络的一部分

### Go原生网络库的问题

在云原生环境下，Go的应用程序越来越多，生态越来越好，主要得益于它的协程和Chan机制使得多核并行编程变得非常容易，因此大量的应用对IO的需求也越来越高，但是原生的udp socket性能大概只能在`1Mpps`左右，对于拥有大量核心的CPU已经构成处理瓶颈了，虽然我们可以通过CGO的方式去做一些奇奇怪怪的事情，或者把Go编译成.so然后再和dpdk结合什么的，这些方式始终用起来不是很优雅。

### 为什么要解决UDP的问题

TCP有专门的TOE但是协议决定了buffer管理等一系列处理特别难，另一方面QUIC**等userspace的通信库越来越多，还有音视频网络中大量的数据流都是UDP。而针对网络应用很多遥测数据也是UDP，特别是INT一类的场景对于UDP socket性能要求都在10Mpps以上，同样金融的行情分析，特别是前段时间给某交易所做的行情延迟监控系统，可能需要将路由器复制给多个会员单位的数据都拿来做延迟监管。

### Google没有DPU怎么搞的？

当然另外一种做法就是DPU、例如使用Xilinx的`QDMA`然后让智能网卡DMA到一个`shmem`而且还可以每个核分配一个`Queue`，但是智能网卡本身的成本大概比普通网卡要增加1000刀，而现代多核处理器拿出两个核来处理也不浪费啊。所以基于这个思路我去看了看Google怎么搞的， 毕竟四大公有云厂商它家朴素得就用普通网卡，然后搞了一个Swift**吊打一众人... 但是这里要说的是我突然发现一个很好玩的东西，`Snap`:a Microkernel Approach to Host Networking

![图片](assets/1d78a025be5d.png)

别被微内核一词混淆了，其结构如下：

![图片](assets/839dbd3586d5.png)

结论很清楚， 利用Kernel Bypass和Many-Core CPUs来实现:

![图片](assets/97cc646182e1.png)

### Ruta HostNetwork实现

以前讲过，Ruta本质是也是一个Host network架构，并且有自己的control-plane注册到etcd实现拥塞控制等业务，而数据平面跟几个合作伙伴的大厂商量了一下，基本上决定直接走DPDK放弃过于笨重的OVS或者VPP。

![图片](assets/4502b6531e15.png)

所以最近在做Ruta主机网络时，顺手加了一个Memory Interface把Go的原生udp socket性能问题解决了。大概架构如下：

![图片](assets/4abda562363b.png)

利用DPDK创建一个vhost-user接口，主要是负责协议相关的处理，例如ARP、IP还有一些带外网管运维的协议等，另外单独创建几个memif 接口，利用主机网卡打开RSS，通过2~4个CPU核接收报文后解析报文并match相应的UDP端口送到多个memif处理，剩下的报文一律送到vhost-user。类似于KNI，但是KNI已经被社区废掉了，而且Kernel依赖真的很烦人...

其实一开始我还想直接shmem来做，但是后来看到memif性能也不错，然后DPDK也内置了，干脆就全部切到memif上了，而在golang侧，我们现在使用的是vpp代码中自带的golang memif库做了一些微小的修改，最终用了16个memif，收包速率接近29Mpps：

![图片](assets/a698153b6c21.jpg)

但是我们也发现可以针对UDP场景和应用耦合的好还可以省掉一次memcpy进行优化，最终的实现目标是仅改udp建立socket的几行代码完成这事。这些代码稍后调试好了再开源出来...

### 后记

当然Ruta HostNetwork的功能不会仅限于对UDP的Offload，针对QUIC等协议或者HostOverlay等协议自动添加一些路由和Overlay header，当然很多说这个地方还可以用BPF来做，没错，但是我后面还会添加crypto的支持，以及基于Marvell CN10的智能网卡卸载，所以最终还是选择了DPDK。Marvell CN10可能还不会用于主机网络，而更多的是Ruta的Fabric节点做中继路由。

另外一个问题小小的剧透一下，大家读过这篇

![图片](assets/3536dd01db17.png)

回到20年前，我们一样也有核心设备路由表不够的问题，MPLS标签转发解决了这个问题。而源路由也可以很好的解耦Overlay和Underlay的问题。VXLAN源路由没有标准，SRv6头太大很多可编程交换机**搞不定， 即便是有头压缩也搞不定。仔细想想，您就会明白Ruta整个架构的设计原理了。

其实很多时候，我们国家在IPv4上受制于人，上IPv6这是必然的趋势。但是在传输网络**上，不一定也要IPv6，这是很多网工遇到的最大问题，理智一点吧。另外，很多大厂都有一个问题，基础架构的人不受待见导致创新缓慢，可以参考昨天的一篇文章，关于企业数字化司库[1]管理和转移定价的问题，各厂的财务和人力资源团队可以看看,通过转移定价来核算成本和激励基础架构团队，同时降低应用部门业务需求。

![图片](assets/8cd371eb6884.png)

#### Reference

[1]
企业数字化司库及云计算流动性管理: https://mp.weixin.qq.com/s/Jk03gs7a0r7bN8W-9PXclQ