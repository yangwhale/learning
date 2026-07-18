# 运营商SDWAN融合MPLS VPN解决方案

> 作者: zartbot  
> 日期: 2021年1月21日 10:58  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485095&idx=1&sn=0e24851d162c43f38437fdac68a1c507&chksm=f9961665cee19f73b9787515ceb18b146a9bcd04e9c451e8643909b04effc7faf2963c1232ee#rd

---

很多人谈到SDWAN总是把它放在MPLS VPN的对立面上，水火不相容的感觉。但是下面我们要说的一件事是：如何实现SDWAN和MPLS VPN的融合，并采用NFV等技术帮助运营商构建下一代PE路由器资源池的案例。

当然继续以往的风格，前面讲业务场景for BDM，中间讲技术细节for TDM，最后讲技术实施。

**1. 业务场景
**

传统运营商通常会在PE位置放置模块化的大机箱，例如ASR9000、NE40等路由器。但是随着业务弹性需求，这类物理设备投资回报率并不是那么的好。同时随着运营商云网融合需求，NFV一类的设备逐渐开始替代这些传统路由器。但是NFV单机容量有限，池化管理也是一个麻烦事，同时需要管理大量的CPE设备。使用SDWAN构建PE-CE链路或者构建靠近用户的微型卫星PE节点成为一种选择：

![图片](assets/d2fa3345942e.png)

例如**最左边的Cedge作为一个卫星PE放置在楼宇中，它可以是一个很小的设备，也可以是一个较大规模的物理路由器，例如整个wework**办公室共享一个Cedge，然后通过不同的LAN子接口隔离多个租户，Cedge PE作为NFV放置在运营商边缘直接作为MPLS VPN的PE使用。如果需要大规模接入，可以放入多台基于NFV的Cedge PE即可。由于NFV部署软件升级割接都会变得更加方便，例如可以采用灰度发布将部分CE切换到新的PE上。**

这个方案初看没啥意思，因为很多SDWAN解决方案都可以支持多VPN，然后通过类似于MPLS InterAS Option-A的方式，使用子接口和原有的MPLS骨干网的PE互通：

![图片](assets/b4e7895b016a.png)

这种两个盒子的操作当然容易，那么今天要给大家讲的是，**单个盒子作为SDWAN的Edge节点和MPLS VPN PE，并且支持NFV，使用SDWAN控制器批量管理并和业务系统集成。**

**2. 技术实现
**

我们以思科SDWAN为例，使用Catalyst 8000v虚拟路由器实现SDWAN网关和MPLS PE在一台机器上的共存，其中Gi1接口作为SDWAN的隧道接口，Gi2作为连接传统MPLS PE-CE的接口，Gi3作为连接MPLS-Core的接口。

![图片](assets/9f3d56b6b269.png)

由于思科SDWAN实现本质上也是一个和BGP、MPLS-VPN类似的实现方式，所以需要在两个功能共存时配置MPLS的标签空间和per-vrf标签

```
mpls label mode all-vrfs protocol bgp-vpnv4 per-vrfmpls label mode all-vrfs protocol bgp-vpnv6 per-vrfmpls label range 100000 1048575 static 16 99
```

然后MP-BGP的配置也没有特别的地方，只需要在相应的VRF**里重分布OMP路由进入即可

```
router bgp 65322 neighbor 3.3.3.3 remote-as 65322 neighbor 3.3.3.3 update-source Loopback0 address-family ipv4 unicast vrf 201  redistribute connected  redistribute omp  exit-address-family ! address-family vpnv4 unicast  neighbor 3.3.3.3 activate  neighbor 3.3.3.3 send-community both  exit-address-family !!
```

接口配置也就是使用简单的LDP**就可以了

```
interface GigabitEthernet3 mpls ip ip address 10.10.12.3 255.255.255.0
```

OMP**路由里，重分布BGP的VRF路由即可：

```
sdwan omp  address-family ipv4 vrf 201   advertise bgp   advertise connected
```

VRF配置注意和VPNv4的Route-target匹配即可：

```
vrf definition 201 rd 1:201 address-family ipv4 ! route-target export 100:201 route-target import 100:201
```

然后检查一下配置，Ping一下，看看转发表，搞定~

![图片](assets/12f3084ec5d5.png)

嘻嘻~今日份技术扶贫简单~ 到此介绍，后面有一个大的活， SDWAN Policy Framework