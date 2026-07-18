# 思科SDWAN探秘(3)--公有云部署

> 作者: zartbot  
> 日期: 2021年1月11日 11:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485003&idx=1&sn=17124ff015621ac828ca44d80711934b&chksm=f9961689cee19f9fb1dc03bfdc09d748b0832aaaf23487e96697ef7c43b192bc5bd5e1f0b7e4#rd

---

先来个下集预告, 我们和Azure及AWS合作开发了多云部署方案：

![图片](assets/7bf198bf6ae2.png)

下一期会给大家介绍配置流程，只需要点一点就可以快速在云端创建CSR8000v，将云端的VPC或者VNET**快速连接到本地VPN环境中。

**这是一条分割线**

除了本地采用VMware**部署SDWAN控制器，思科也提供云端的托管控制器服务，当然您也可以自己在云端部署控制做一些POC实验。我们以阿里云为例，**但请您务必注意，vManage、vBond、vSmart、vEdge的云端部署是不受官方支持的，只是作为实验性质。当然我们也在尽力和多个云进行适配，或者云平台承诺使用标准KVM环境，或者裸金属环境也可以进行有针对性的适配：）**

****

而CSR8000v我们已经完成和阿里云的适配，会有官方的支持，现在正在早期验证性测试阶段，它将于今年三月份伴随20.5版本一起发布，您可以直接使用CSR8000v连接阿里云和本地网络。

**注意NTP**和时区一定要配置正确，否则会在证书签发的过程中出现一系列证书生效时间的问题！**

**1. 上传制作镜像**

**1.1 镜像文件上传**

在阿里云中建立对象存储：

![图片](assets/f22fbe73f377.png)

然后点击“上传文件”，本次实验需要上传的文件为思科官网CCO上下载的KVM**镜像文件，主要包括vManage、vEdge（用于vBond）、vSmart和CSR8000v，这次我们选用的是20.4版本的文件

![图片](assets/a76be0ac06ee.png)

**1.2 制作镜像**

点击云服务器ECS，选择“镜像”，点击“导入镜像”

![图片](assets/b4481309b3a2.png)

首先来制作CSR8000v的镜像，选择您上传页面的OSS ObjectURL地址然后操作系统选择CentOS，镜像格式为QCOW2，许可证类型为自带许可，点击确定即可

![图片](assets/efcadad6fe33.png)

其他系统vManage/vEdge/vSmart也类似的处理，这些系统盘大小配置为40GiB即可.

![图片](assets/a4c2d1ec0218.png)

**2. 镜像安装**

**2.1 创建vManage实例**

我们选择阿里云为我们提供的G5NE实例进行配置，vManage需要选择相对较大的实例类型，具体CPU和内存的需求如下：

![图片](assets/d116adf51da9.png)

当然由于我们是测试性质，选择ecs.g5ne.2xlarge

![图片](assets/7364d0118677.png)

镜像选择我们自定义的vmanage，选择添加一块新的硬盘：

![图片](assets/76b784f7e767.png)

安全组中需要放行22、830、443、8443、12346、12446、12546、12646、12746、12846、12946、13046等端口。当然您也可以全部放行，然后vManage自己前端会有端口访问控制的策略。

![图片](assets/89a8197c45b8.png)

网络中vManage需要配置两块网卡，另一块网卡用于内部VPC的管理通信

![图片](assets/798c64684ed6.png)

密码使用镜像预设密码

![图片](assets/afa53646b50e.png)

最终确认订单创建实例

![图片](assets/21bcd3d8bec9.png)

启动后登录VNC，输入默认用户名密码 admin/admin, 选择第二块硬盘，然后下一步即可

![图片](assets/2a0e9347f08c.png)

完成后，系统会自动重启，登录控制台，默认eth0会接入到第二张网卡，而eth1是有公网接口的网卡，所以我们需要将eth0从vpn0移除，将eth1加入：

```
config tsystemntp server ntp.aliyun.comvpn 0no interface eth0dns 223.5.5.5.5 primarydns 223.6.6.6.6 secondaryinterface eth1  ip dhcp-client  no shutdownvpn 512interface eth0  ip dhcp-client  no shutdown
```

**2.2 创建vSmart实例**

我们选择阿里云为我们提供的G5NE实例进行配置，vSmart实例如下,同样需要两块网卡

![图片](assets/da408802988f.png)

完成后，系统会自动重启，登录控制台，输入默认用户名密码 admin/admin, 然后和vmanage相同，默认eth0会接入到第二张网卡，而eth1是有公网接口的网卡，所以我们需要将eth0从vpn0移除，将eth1加入：

```
config tsystemntp server ntp.aliyun.comvpn 0no interface eth0dns 223.5.5.5.5 primarydns 223.6.6.6.6 secondaryinterface eth1  ip dhcp-client  no shutdownvpn 512interface eth0  ip dhcp-client  no shutdown
```

你可以通过在此将其转换为弹性公共IP，将其地址固定下来：

![图片](assets/1374a29e5916.png)

****

**2.3 创建vBond实例**

vBond采用vEdge的镜像创建，vSmart和vManage可以采用动态IP地址创建，但是建议vBond配置弹性公网IP静态绑定。

![图片](assets/3611d4b5d68a.png)

vBond选择的实例和配置类型如下：

![图片](assets/8f42da5482a6.png)

在弹性公网IP界面点击右侧绑定资源：

![图片](assets/cb888931c52a.png)

选择vBond绑定即可：

![图片](assets/31146344df6c.png)

完成后，系统会自动重启，登录控制台，输入默认用户名密码 admin/admin, 这一次地址对了，ge0/0默认为第一块网卡，有公网地址，eth0默认为第二块网卡，您只需要配置DNS和暂时关闭隧道接口即可

```
config tsystemntp server ntp.aliyun.comvpn 0dns 223.5.5.5.5 primarydns 223.6.6.6.6 secondaryinterface eth1  no tunnel-interface
```

就此，实例创建结束，您可以继续阅读《思科SDWAN探秘(4)--初始化控制器》，CSR8000v在阿里云上的安装详情请见《思科SDWAN探秘(5)--设备上线》