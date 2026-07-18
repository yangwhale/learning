# Ruta实战及协议详解

> 作者: zartbot  
> 日期: 2021年2月1日 10:30  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485165&idx=1&sn=412fcb1dd46dd4ef4384a033b0827256&chksm=f996162fcee19f39ab4c995b1be2676779eb5b647ad26dd7017001b75530dfb9a59e9790b37d#rd

---

Ruta协议是一个新一代的网络传输协议，控制面采用ETCD构建，而转发面采用SegmentRouting over UDP实现，协议已经有RFC草案，连接如下：

https://tools.ietf.org/html/draft-zartbot-sr-udp-00

https://tools.ietf.org/html/draft-zartbot-srou-control-00

它是我对于《包处理艺术(2)--如何设计协议》一文的工程实践：

原则1：控制平面采用CP

在SDWAN等复杂场景中，控制平面可达性本身就无法保障，经常会存在Headless的情况，那么Availability自然就可以舍弃了，找个CP实现的系统作为控制平面则是非常自然的取舍，一致性的约束范围也有取舍，不保证链路状态一致性，而仅保证资源归属一致性。

原则2：数据平面采用AP->BASE

数据平面才是真正承载用户业务的地方，选择AP的原因是为了实现BASE(Basically Available, Soft State, Eventual Consistency). 如前文所述，使用SR是必然选择

原则3：同构

zartbot.Net，公众号：zartbot[包处理的艺术(2)---如何设计协议](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484550&idx=1&sn=0627d92a24590316a573af70f24cb3f0&chksm=f9961444cee19d5251efcac850ee9e3339090fc99cc454c496750b197a514bab7e0f22043003#rd)

为了更好的让大家了解Ruta协议的实现过程, 最近构建了一个工程demo实验环境，但是和内部研发的版本相比删除了加密和数据面DPDK**的支持，并限制了性能和转发延迟，主要原因是怕有些人误用，开放出来的原因主要是便于多厂商之间的合作和协议设计交流。

**1. Ruta架构简介**

对于整个Ruta系统，主要的节点类型分为：

![图片](assets/91cd9b5d02d3.png)

**节点类型**

**用途**

**STUN**

需要有公网地址或者1：1NAT静态映射的地址。

主要用于帮助其它节点发现公网地址

**Fabric**

用于SRoU的中继转发节点.  它默认会根据经纬度等一系列信息主动的发起和其它Fabric之间的TWAMP测量协议。

**Linecard**

SRoU的头端和尾端，可以以多种形态存在，移动端的VPN  软件或者代理，家用路由器，AP，服务器端的智能网卡，应用的Sidecar

这些节点可以根据地理位置信息直接发起对目的节点Linecard的TWAMP延迟测量，也可也hunt就近的Fabric节点并注册到Fabric节点的链路上。

**Analytic**

主要用于watch stats，分析网络故障，为AIOps提供数据支撑等，它也可以作为分布式智能控制器帮助区域内节点进行策略优化等。通常我会用一些类似于Nvidia Jetson Xavier或者带GPU的大型节点放置于园区中用于做一些模型的推理。

 

当然在某司我有另一个项目就是直接在路由器和网络设备上运行决策树模型，而神经网络模型通常需要大量的矩阵计算还是offload出来比较好。

它的数据面采用了UDP封装形式，并借鉴了SRv6的优点采用SRH编码构建，相对于MPLS over UDP等方案更具灵活的可编程特性。另一方面考虑到IPv4的部署和互联网多云场景，它内置了NAT穿越感知的能力。由于在UDP Payload中，应用程序也可以很好的利用网络资源，在用户态即可调度流量，App可以内置VPN等业务功能，实现了跨越Internet的SegmentRouting

![图片](assets/4afc09db1acb.png)

整个协议框架吸收了各个已有技术的优点

![图片](assets/6cf2907bfc3c.png)

**2. Ruta验证环境部署**

测试环境为一个典型的数据中心Spine-Leaf架构，当然您要构建SDWAN环境在多云上部署也可以，主要需要9个虚拟机，当然您资源紧张也可以将Ruta进程使用不同端口启用在少量主机上。这次演示的环境采用的操作系统为Unbuntu 20.04，因为整个代码为golang构建，也有ARM、MIPS的环境可以在OpenWRT上运行，但这次演示没有将其包含。

![图片](assets/c5b1ab580bec.png)

**2.1 下载程序及配置TLS证书**

ruta协议和ETCD的连接采用TLS，因此您需要配置相应的证书，我们可以在第一台机器上(192.168.99.71)安装git和golang-cfssl

```
sudo apt install git golang-cfssl
```

下载演示环境, 注意确认整个项目安装在/opt/ruta_demo位置，因为证书路径故意做了hardcode为/opt/ruta_demo/cert.

```
cd /optgit clone https://www.github.com/zartbot/ruta_demo
```

生成证书：

```
cd /opt/ruta_demo./01_generate_cert.sh
```

如果您ETCD地址在部署的过程中需要更改，可以修改如下文件

```
/opt/ruta_demo/cert_cfg/server-csr.json  "hosts": [    "localhost",    "0.0.0.0",    "192.168.99.71",    "192.168.99.72",    "192.168.99.73"  ],
```

证书产生完毕后，将整个/opt/ruta_demo 打包拷贝到其它机器(192.168.99.72~79).然后解压到相同的/opt/ruta_demo位置

```
cd /opttar cvzf ruta.tar.gz ./ruta_demoscp ruta.tar.gz zartbot@192.168.99.72~79:@192.168.99.72~79cd /opttar vzxf ~/ruta.tar.gz
```

**2.2 启用ETCD集群**

我已经为您配置好了启动shell和相应的配置文件，在每台机器上执行即可

```
cd /opt/ruta_demo@ETCD1(192.168.99.72)./start_etcd1.sh@ETCD2(192.168.99.72)./start_etcd2.sh@ETCD3(192.168.99.73)./start_etcd3.sh
```

如果您部署的ETCD集群IP地址不同，具体可以参考/opt/etcd/etcdX.yaml修改即可。

**2.3 启用STUN服务**

STUN服务用于帮助其它节点获取公网IP地址，因此STUN服务器和ETCD集群必须要有公网接入或者保证使用1:1静态地址映射的弹性IP。登录到STUN1服务器(192.168.99.74)，执行如下命令启动STUN服务

```
cd/opt/ruta_demo/stun./stun -c=conf.yaml
```

在conf.yaml中您可以定义STUN服务器的IP地址和连接的ETCD控制器地址，如下所示：

```
role: stunsiteID: 100systemID: 1.0.0.1controller: [192.168.99.71:443]srloc: [ INET|1000|1000|192.168.99.74:5555|192.168.99.74:5555]
```

启动后，STUN服务会按照如下方式注册节点和通告业务能力

![图片](assets/16a422c6b9c5.png)
**1. SystemName：**相对于传统的RouterID，SystemName更加易读，您可以继续采用RouterID编码方式，也可以使用ASCII String命名节点

**2. SiteID：**这个类似于BGP**的ASN，主要是用以做一些site level的策略使用的。

**3.SystemLabel：**用来压缩SID长度或者和MPLS网络对接时的标签，也可以用这个标签映射成VPN客户端的IP地址。它总长24bit，通过分布式锁机制由ETCD分配。完成后还会生成一个/systemIP/<Label>到SystemName的映射，便于后期中继节点通过监听/systemIP获取网络标签表

**4.SRLOC：**该字段采用如下编码方式，由于我们此次环境为全内网环境，因此公网地址也和内网一直，当您在有弹性IP的云环境部署时，记得更改公网地址信息。另外SRLOC在yaml中为列表类型，您可以定义多个SRLOC，在后面Fabric的示例中可以看到。

![图片](assets/bc451a0128ab.png)

上图中ETCD Dump K-V的小工具，可以在如下路径中找到:

```
cd /opt/ruta_demo/ops./listen
```

当STUN节点注册后，你可以看到系统中已有的Key-Value Pairs和后续更新的消息，注意其中/key/<role>/<systemName>为用于节点间通信的加密对称密钥，这样避免了节点间进行频繁的IKE**密钥交互流程。

**2.4 启用Fabric服务**

分别在Fabric1(192.168.99.75)和Fabric2(192.168.99.76)上启用

```
@fabric1cd /opt/ruta_demo/fabric./fabric -c=fabric1.yaml@fabric2cd /opt/ruta_demo/fabric./fabric -c=fabric2.yaml
```

Fabric的配置文件中SRLOC没有公网地址，它将自动从ETCD中获取STUN服务资源，然后和STUN服务器通信获得公网信息：

![图片](assets/4ded39b07600.png)

注册完成后，我们可以通过/opt/ruta_demo/ops/monitor工具看到路由信息。当Fabric2启动后，监测到有其它Fabric节点，它将发起Linkstate Probe，并通告链路状态到ETCD

![图片](assets/7d65c7331629.png)

为了更好的显示路由信息，我们可以通过如下命令查看：

```
cd /opt/ruta_demo/ops./monitor
```

![图片](assets/2adaba7ebfd9.png)

**2.5 启用Linecard服务**

我通过veth-pair和network namespace来模拟Linecard后挂接Docker的场景，并通过它构建EVPN和远端Linecard连通，开启veth脚本：

```
@Linecard1cd /opt/ruta_demo/linecard./veth_lc1.shsudo ./linecard -c=lc1.yaml
```

‍

![图片](assets/3235f16d20b1.png)

在Linecard1上执行ping_gw.sh会从veth0学到终端MAC和IP并通告Type2路由。同时您也可以开启Linecard2

```
@Linecard1cd /opt/ruta_demo/linecard./ping_gw.sh@Linecard2cd /opt/ruta_demo/linecard./veth_lc2.shsudo ./linecard -c=lc2.yaml./ping_gw.sh./ping_lc1.sh
```

完成后整个系统路由表如下：

![图片](assets/fcfe2f8eaa84.png)

**3. Ruta路由流程**

Overlay路由查询采用VPN(type2/type5)路由查询获取Underlay SRLOC，然后根据Linkstate本机可以构建一个FlexAlgo计算到远端的SRLOC路径，并构建转发的SRH

![图片](assets/65c07a034e07.png)

FlexAlgo中可以采用我当年的一个AI智能路由算法，识别应用性能，并主动预测和调整路径：

![图片](assets/fe16cf8c5151.png)

![图片](assets/4e1c60308c14.png)

演示如下：

      
     
       
         
           
             
                                

                 
                   
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
   
 

SRoU Header的定义

![图片](assets/aca216fe7465.png)

3.1 Magic Number

报文头中的Magic Number，当用于QUIC传输时，它为0x0**，这样正好区分开QUIC的Long/Short Packet Flags，而针对IPSec封装时取值为0xF，并规定IPsec SPI不分配0xFxxxxxx的SPI就好。通过这样的方式就可以在原有的IPsec隧道或者QUIC会话中添加SR能力了。

3.2 FlowID字段

协议定义是变长的，在报头也有一个Length字段帮助ASIC**定位，但工程实现上可以定义为一个64bit或者128bit flowid，它的用途一方面是用于做基于流的监控和分析，类似于Google Dapper的TraceID，而SID-List则可以作为Parent-ID使用，

另一方面也适合于DetNet的处理，可以进一步控制延迟。而当我们在IPv4公网上架设服务时，也需要一个token机制避免其它人使用您的带宽，或者注入攻击流量，这也是一个接入控制的ID，处理机制和互联网公司的Token机制类似，验证+黑名单数据库即可。

3.3 Source Address字段

这个字段用于头端封装自己的公网地址，或者第一跳Fabric节点帮助将接收到的IP包的公网地址和端口压入这个字段。主要目的是让远端的隧道终点溯源同时执行逆向对称路由或者可优化的非对称路由。

第一跳压入还有另一个考虑，在报文中间插入header会是一个非常麻烦的事情，中继Fabric节点如果需要插入源地址信息在UDP头中，如果用DPDK实现则需要把mbuf 拆开，然后insert，或者把很多bytes 前移后移空出一段。而第一跳需要LineCard识别公网也很麻烦，搞STUN代价也有点大，所以第一跳还可以什么都不干，allocate一段source address 空间置0就好，而Fabric节点的第一跳设备可以自动的把源地址源端口拷贝进入这个字段即可。

3.4 SRH & Function

定义和操作方式和SRv6完全一致，只是针对IPv4的UDP overlay场景添加了一个48bit长度的SID支持， IPv6网络中可以采用UDP固定协议端口的方式沿用128bit SID即可。所有的字段要直接拷贝到SRv6 SRH中也非常容易。同时正如前文所述，考虑到IPv4支持可编程的能力，我们将SID中高位为255.0.0.0：xx的整个一段用于执行End.X的可编程处理能力，例如我们的线卡节点接收到一条Type-2EVPN路由，数据包的Segment-List[0]的封装即为

![图片](assets/9d6e6e3fc297.png)

定义上End.X正好16bit，VNID也刚好24bit，而255.0.0.0/8的地址段正好又没人用，就这样我们在IPv4网络上也可以实现类似的可编程能力。

3.5 INT Rewrite（draft-version-01会添加）

SRH编码也有缺点，header不好删，那么我们可以采用ReWrite用过的SID来构建对该跳节点的In-Network Telemetry描述。具体可以参考**《[凡是过往，皆为序章: 源路由路径感知转发](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484565&idx=1&sn=e2e23840d9fd1a8fd8ff22e447107707&chksm=f9961457cee19d41d2b1f6d2959bb1dbda57809cd722e2dea6433e1b96370e1e2ec45787b1bc&scene=21#wechat_redirect)》**

**4. Ruta控制策略**

在工具中，我们提供了一些便捷的添加和删除路由的工具/opt/ruta_demo/ops/edit.当然严格来说还需要基于PeerGroup那样或者像Viptela那样提供CentralPolicy和Local Policy。协议草案已经考虑到了，在SRoU的控制面也专门设计了相应的keyprefix用于提供策略和身份认证：

```
Key="/control/RT/2/SRC_MAC/SRC_IP/DST_MAC/DST_IP" Key="/control/RT/5/SRC_Prefix/SRC_Mask/DST_Prefix/DST_Mask" Value="Action" /"SR Locator list"Access ControlKey="/token/permit/flowid" Key="/token/block/flowid"IdentityKey="/identity/owner/deviceid"value="role,policyprefix"
```

当然后期在控制面的Draft-version-01中也会添加一些Sequece Matching & Action的逻辑语义。但是设计的原则离不开以前设计的基于自然语言的网络意图工作提到的Security Label: **>>****[意图网络的语言学思考](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484028&idx=5&sn=c8de8c1fe6fa947ca1ff98c2f6b34d20&chksm=f99612becee19ba8bc32862780f058d7a94c7380aa6104a774e07c717086fce936a3a192df6a&scene=21#wechat_redirect)<<**

![图片](assets/07aa2be403a6.png)

**5. Ruta遥测和AIOps**

在Ops工具中，我们提供了一个analytic用于将LinkStats和NodeStats导出到ElasticSearch：

```
cd /opt/ruta_demo/ops./analytic -c=analytic.yaml -uri=http://elastic:pwd@elasti_ip:9200
```

‍

利用Kibana做一些基础的数据呈现：

![图片](assets/089b7fec46b2.png)

当然真正的AIOps和可预测路由及骨干网BGP分析可以参考另一个文**《****[Ruta：不用花10个亿也能做千眼](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484142&idx=1&sn=2a6533102a81ef169124b57a02b4e83d&chksm=f996122ccee19b3accdb3c4100d0e8b532f482254436f1574ff53b7c3b97e12c1dea7d2fb0f4&scene=21#wechat_redirect)****》**

![图片](assets/4346f65a8d23.png)

**6. 结尾**

**"Design is a funny word. Some people think design means how it looks. But of course, if you dig deeper, it's really how it works. The design of the Mac wasn't what it looked like, although that was part of it. Primarily, it was how it worked. To design something really well, you have to get it. You have to really grok what it's all about. It takes a passionate commitment to really thoroughly understand something, chew it up, not just quickly swallow it. Most people don't take the time to do that."**

任何一个协议都有好有坏，有的可能有相同的转发面行为，而控制面却截然不同，有时候被迫接受了SRv6的很多优点，却少了那么退一步海阔天空的气魄。至于我为什么要研发Ruta，这周有个交流结束了，再把Ruta的研发历程和思维方式分享给大家。