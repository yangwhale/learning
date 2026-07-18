# 思科SDWAN探秘(5)--设备上线

> 作者: zartbot  
> 日期: 2021年1月11日 11:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485003&idx=5&sn=97b1f659a74e9760ea02d888f31bce40&chksm=f9961689cee19f9f75c7b5a8a18823dc055999c88c462c3e9f5620a5c6beecc0c880f92cc7e5#rd

---

**1. 云端部署CSR8000v**

****以阿里云为例，点击云服务器ECS，选择“镜像”，点击“导入镜像”

![图片](assets/4dc4db082041.png)

首先来制作CSR8000v的镜像，选择您上传页面的OSS ObjectURL地址然后操作系统选择CentOS，镜像格式为QCOW2，许可证类型为自带许可，点击确定即可

![图片](assets/8064e5ed408c.png)

实例类型我们选择阿里云专门为思科定制的g5ne.large实例，镜像类型选择自定义CSR8000v

![图片](assets/e3ff7988ba25.png)

网络类型也是两个或者多个网卡，一个接公网，另外一个连接多个VPC

![图片](assets/a149ab8c682e.png)

密码也是用镜像预设密码：

![图片](assets/ece226cc56ba.png)

登录VNC，将其转换为SDWAN模式,配置如下：

```
enablecontroller-mode enable
```

![图片](assets/b66d1475e52b.png)

然后使用admin、admin登录，登陆后会要求您更改密码,基本配置如下：

```
config-transactionhostname CSR_AliCloudclock timezone china 8systemsystem-ip 204.0.0.1site-id 130002organization-name CiscoDMZvbond 101.37.116.217ntp server ntp.aliyun.com!interface GigabitEthernet1 ip address dhcp!line vty 0 10 transport input sshcommit
```

然后再安装证书,可以在vmanage的vshell中，直接scp证书文件到CSR8000v的bootflash中：

```
vmanage# vshellvmanage:~$ scp root-ca-chain.pem admin@47.114.54.10:root.pem
```

然后再CSR8000v中安装

```
request platform software sdwan root-cert-chain install bootflash:root.pem
```

配置隧道：

```
interface Tunnel1no shutdownip unnumbered GigabitEthernet1tunnel source GigabitEthernet1tunnel mode sdwanexitsdwaninterface GigabitEthernet1tunnel-interfaceencapsulation ipsec allow-service sshd color public-internetcommit
```

然后在Vmanage中挑选一个C8000v的token就好：

![图片](assets/1264d1c6aa76.png)

在CSR8000v控制台敲入：

```
CSR# request platform software sdwan vedge_cloud activate chassis-number <Chassis-Number> token <token>
```

等待片刻，CSR8000v就上线了

**
**

**2. 部署Catalyst Edge8300**

我来看物理路由器如何开局的，默认它会有PnP开局的方式，但是我们先跳过，毕竟这文是讲原理的，首先打开机器接上串口线，将其转换为SDWAN模式,配置 如下：

```
enablecontroller-mode enable
```

![图片](assets/509e95b07a0a.png)

然后使用admin、admin登录，登陆后会要求您更改密码，然后中止原有的pnp流程，配置如下：

```
Router# pnpa service discovery stop
```

基本配置如下：

```
config-transactionhostname CE8300Aclock timezone china 8ntp server 10.74.5.1systemsystem-ip 203.0.0.1site-id 130001organization-name CiscoDMZvbond 101.37.116.217interface GigabitEthernet0/0/0ip address 10.74.6.133 255.255.255.0no shutip route 0.0.0.0 0.0.0.0 10.74.6.1commit
```

然后再安装证书：

![图片](assets/b59e0ebd60a1.png)

在Vmanage上配置“Administration-->Settings"选择One Touch Provisioning，选择“Enabled”并保存。

![图片](assets/c0adcd11ce11.png)

配置隧道：

```
interface Tunnel1no shutdownip unnumbered GigabitEthernet0/0/0tunnel source GigabitEthernet0/0/0tunnel mode sdwanexitsdwaninterface GigabitEthernet0/0/0tunnel-interfaceencapsulation ipseccolor mplscommit
```

完成后就可以在vmanage ”Configuration=》Devices=》Unclaimed WAN Edges“中看到这个设备了，选择，然后Claim就上线咯~

![图片](assets/60ee22d682a7.png)

**3. 边缘云和SD_Branch部署CSR8000v**

我们前一篇文章中讲述了一种CSR8000v开局方式

今天的主角NFVIS，即完全将一个基于CentOS的操作系统安装在ENCS或者UCS标准服务器平台上，然后以虚拟化的方式构建VNF、VM及容器，并通过支持SDWAN的统一虚拟路由器平台CSR8000v链接到云环境。

zartbot.Net，公众号：zartbot[云网一体，端网融合：思科NFVIS探秘](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484644&idx=1&sn=fc2060ca631ef3d0c1285ff109040992&chksm=f9961426cee19d30556fe2d60327acdd64477b8d9100322dc08120e87ec710678286a5986c34&token=2089433615&lang=zh_CN#rd)

在NFVIS中可以通过如下界面实现一键开局，其中UUID就是上图中的Chassis Number，OTP就是上图中的token，设备就会自动上线。

![图片](assets/ecb1d4d2d049.png)

**4. VMWare部署CSR8000v**

而今天我们针对云环境在vmware上部署一个虚机版本的CSR8000v，选择ova部署，然后也是下一步点到选择不同的Profile，做实验玩选择最小的就好，但是成年人的世界，肯定是最大都要呀~

![图片](assets/e69f17a16cea.png)

存储选择也随意，反正它做转发基本不用硬盘，然后网络分三个接口GigabitEthernet1接Internet隧道服务口，Gi2、Gi3接业务口即可：

![图片](assets/afd5a9e10f1d.jpg)

填写基本的主机名，管理用户名密码以及相应的广域网接口地址即可：

![图片](assets/3c3f35468ce5.png)

然后直接翻到最后填写SDWAN配置，选择一个UUID和OTP对填入即可：

![图片](assets/bd2aeff86979.png)

重新启动后，登录CSR8000v，配置系统相关属性，例如systemip和siteid以及Organization-name：

![图片](assets/54c3ddd176b1.png)

配置IP地址和默认路由：

![图片](assets/7f4ca59f1e74.png)

在CSR上拷贝RootCA证书,然后输入如下命令安装证书即可：

![图片](assets/4986a571548e.png)

```
request platform software sdwan root-cert-chain install bootflash:root-ca-chain.pem
```

最后按照如下方式配置隧道即可：

![图片](assets/7a966266c1ac.png)

如果需要ssh登录则需要按如下方式配置：

![图片](assets/c75c024f54fd.png)

最后配置OTP加入到控制器：

CedgeA#request platform software sdwan vedge_cloud activate chassis-number <chassis-number> token <token>

稍等片刻，设备就上线了：

![图片](assets/cda230edbed0.png)

**5. 配置service vpn**

点击Tools-》SSH TERMINAL可以通过ssh远程登录到设备：

![图片](assets/0f71b32a32fa.png)

然后可以配置VRF和Service接口完成Service VPN的配置：

```
vrf definition 101 rd 1:101 address-family ipv4  exit-address-family !!interface GigabitEthernet0/0/1 no shutdown vrf forwarding 101 ip address 192.168.101.1 255.255.255.0exitinterface Loopback101 no shutdown vrf forwarding 101 ip address 192.168.222.1 255.255.255.255exitsdwan omp   address-family ipv4 vrf 101   advertise connectedcommit
```

远端也同样的方式配置相应的service vpn即可，然后查看VRF路由就可以看到远端的路由了

![图片](assets/6f3643dde484.png)