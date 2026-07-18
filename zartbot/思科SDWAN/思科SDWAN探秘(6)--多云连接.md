# 思科SDWAN探秘(6)--多云连接

> 作者: zartbot  
> 日期: 2021年1月12日 12:20  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485055&idx=1&sn=2e95f2923654a035d272dbcc3c0ebec4&chksm=f99616bdcee19fabd2d7d57270525b1d547a112a444df1c2cc7087afe336af62d384aeed4a2e#rd

---

企业上云是刚需，但同时企业为了减少对单个云提供商的依赖，因此多云连接成为新刚需。虽然K8S这样的平台可以解决一定的供应商锁定的问题，但是市场上真正可信的和复杂度相对合理的多云连接解决方案依旧还是空白，直到思科最近发布的一个多云解决方案，前期我们支持了AWS和Azure，和GCP**的合作开发正在进行中。下面我们以Azure多云连接为例，带给大家一种全新的多云连接体验。特别感谢Azure的烽哥提供资源支持本次测试：）

在最近发布的20.4 思科SDWAN解决方案中，我们通过简单的点击鼠标就可以完成云虚拟网络和本地网络及分支网络的连通。当然此文继续维持本号的特点：先回顾历史，然后实战当下，最后展望未来。                             

**1. 架构解析**

**1.1 传统的云连接方法**

多云连接通常的做法是云端放置一台或两台NFV设备，一个接口放置在公网中，另外多个接口连接VPC网络，通过IPsec和其它SDWAN网关公网连接并透传内网流量。

![图片](assets/416ee58b9ead.jpg)

基于多地域AZ的分布式VPC和分布式软件架构逐渐成为解决分布式计算问题的一个办法，问题又来了，如果VPC多了，那么VPC-Peering的数量按照O(N^2)增长，运维复杂度成倍的增加.

![图片](assets/1716e525d21f.png)

为了解决这个问题，云计算厂商又提出了Transit VPC的架构，由思科CSR1000v提供的DMVPN作为Region Hub来提供VPC互联的解决方案。

![图片](assets/5892dfdbbe5e.png)

 也就是以后云计算厂商提出的Transit VPC架构，即构成一个中心化的Transit区域

![图片](assets/b04b44650930.png)

**1.2 新一代多云解决方案**

新一代多云连接方案最关键的一环就是SDWAN控制器利用Cloud API自动构建云端基础架构，完成多云的互通，把复杂性掩盖在控制器内部的自动化流程中，用户只需要“哪儿想通点哪”就行了：）

![图片](assets/63bf52889047.jpg)

**2. 实战多云连接**

**2.1 分发Cloud API权限**

登录portal.azure.com，搜索“订阅”并打开，可以看到您的订阅ID，也就是后文需要用到的subscription-id

![图片](assets/f77818a1bf3e.png)

配置RBAC账户我们通常更愿意使用cloud shell的方式，简洁快速，您可以通过以下方式打开cloud shell

![图片](assets/1ddad93575b5.png)

在Cloudshell中使用如下方式申请资源：

```
az account set --subscription="<subscription-id>"az ad sp create-for-rbac --name="CiscoSDWAN_MultiCloud" --role="Contributor" --scopes="/subscriptions/<your-subscription-id>"
```

![图片](assets/8742d7b27617.png)

打开思科vManage，点击右上角的Cloud onramp for Multicloud

![图片](assets/0325fe78b6f0.png)

在Multicloud界面中，我们就会引导您如何通过简单的四步配置上云

![图片](assets/369c2de3eff3.png)

**2.2 vManage设置云账户**

点击“Associate Cloud Account”，下拉菜单中选择Azure，然后Cloud Account Name自己可随意填写，一定要选择“Use for Cloud Gateway”，后面的内容参考表格中的填法

![图片](assets/e39e2d2103ee.png)

点击下一步配置完成后，直接点击“Cloud Global Setting”配置账户信息

![图片](assets/48134563d991.png)

在界面中点击右上角“Add”按钮，选择软件版本17.4.1855，SKU Scale选择3， IP Subnet Pool是指的在Cloud上创建的与VNET连接的Cloud Gateway节点所需要的IP地址段，需要找一个与已有网络不重叠的/16的地址端，AS号有个限制区间，您可以根据提示随意选择，例如65522

![图片](assets/bdd99af4876f.png)

完成后，它会引导您发现云端的VNET和创建Cloud Gateway

![图片](assets/620f509bb92b.png)

**2.3 同步VNET信息**

在Azure中，我们已经创建了4个虚拟网络VNET,如下所示：

![图片](assets/11a83d7a4bc4.png)

在思科VManage中点击“Discover Host Private networks"即可自动发现，您同时需要在本地为它们添加分组标签，勾选其中某一行，点击Add Tag，为其添加标签

![图片](assets/dd94ce369450.png)

配置完成后，如下所示，例如我们可以给Production和Dev两个VNET分为同一组，分配DEV标签，然后MobileWorker和Test各自分配一个标签：

![图片](assets/7b378cd54631.png)

**2.4 配置Cloud Gateway**

点击菜单中“Configuration-》Template”

![图片](assets/853420abcda4.png)

在Template Type下拉菜单中选择All

![图片](assets/1672ee175c44.png)

您可以看到我们已经为您创建好的Default_Azure_vWAN_C8000v模板，点击右侧“...”选择Attach Device

![图片](assets/18e5aac3bc29.png)

左边列表框选择两台加入， 如果左侧没有，您需要申请CSR8000v授权和安装Serial-File，详情请见《[思科SDWAN探秘(4)--初始化控制器](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485003&idx=4&sn=da6d008a9656322d364154008d000c6e&chksm=f9961689cee19f9f08b8b2ce763abec40b7d04aa2fffff10953b23ad32b2ee91346726f66be4&scene=21#wechat_redirect)》第三章。

![图片](assets/004f9c993a73.png)

点击Attach后，你需要配置一下Cloud Gateway的主机名Site-ID和System-IP，点击右侧"..." Edit Template

![图片](assets/9b752dd53457.png)

配置界面如下：

![图片](assets/ab9dfb469969.png)

完成配置后，需要注意两个Gateway需要使用相同的Site-ID

![图片](assets/39e064edf211.png)

点击下一步，然后点击“Configure Device”即可：

![图片](assets/ecd03f3e5c84.png)

回到Multicloud界面，然后点击“Create Cloud Gateway”

![图片](assets/1de406e4af1f.png)

然后填写您的Cloud Gateway的名称，vWAN名称，下拉菜单中选择您前一步attach到Template中的Device ID即可，点击Add就会自动创建vWAN Hub和Cloud Gateway了

![图片](assets/625d89ed4499.png)

整个创建流程会非常长，请您耐心等待这个页面：

![图片](assets/913d736ddec3.png)

当然您也可以登录Azure的Portal，可以看到vWAN已经创建了。

![图片](assets/d01ada2f4b87.png)

vWAN Hub也在自动创建，并且Vmanager自动会通过CloudAPI在Azure云上配置相应的路由表：

![图片](assets/c13c99a188e4.png)

完成后您会发现第三方提供程序：虚拟网络设备（NVA）已经创建好：

![图片](assets/d91b047c138f.png)

而vManage中也已经加好两台CSR8000v

![图片](assets/f053f2cea911.png)

Multicloud的界面也变为如下样式：

![图片](assets/3433a05963fa.png)

**2.5 配置连接意图：“想通哪儿点哪儿”**

点击上图右下角Intent Management中的Connectivity连接，可以看到一个页面，点击右侧的“Edit”，竖列是您On-Prem的VPN ID， 横列是云端的VNET Tag，点击您需要连通的，点击Save即可

![图片](assets/996e10093f5f.png)

注意: 由于vWAN Hub仅有一张路由表，所以在上图中，点击多个本地VPN匹配是无法工作的，因此重分布到SDWAN OMP中时最好为其创建一个单独的Service VPN，然后通过Leakage的方式将不同的路由通过OMP策略分发到相应的VPN中。或者采用路由通， ACL过滤的方式进行隔离。

创建完成后，它将自动和Azure基础架构建立BGP**邻居，并根据选择的VNET地址段自动分发BGP路由，同时本地的OnPrem路由也通过OMP重分布进入BGP

![图片](assets/9a06b3785167.png)

而本地路由器也能看到远端VNET的路由了

![图片](assets/7ca891fdca2f.png)

**3. 展望未来**

**3.1 Mobile SDWAN**

在VWAN Hub中可以创建P2S**的服务，这样我们的智能终端可以非常容易的连接到vWAN Hub的Pop点，然后通过CSR8000v连接到私有云网络，实现端-网-云的完全连通~

例如我们点击虚拟WAN，选择用户VPN配置，点击“创建用户VPN配置”

![图片](assets/be6e420fce29.png)

根证书产生请您参考Azure的官方文档。然后点击“连接性-中心”，选择集线器（vWAN Hub） 进入点击创建用户网关

![图片](assets/e04386249d70.png)

创建完成后，请耐心等待更新网关，更新完成后点击下载虚拟中心用户VPN配置文件

![图片](assets/5cb2caa9a789.png)

下载文件中选择windows平台直接安装即可

![图片](assets/75817b821233.png)

安装完成后，windows选择VPN就能看到这个Profile了：

![图片](assets/f82deac814ed.png)

点击连接前，您还需要安装自定义的那个根证书：

![图片](assets/02fec7864457.png)

点击连接即可：

![图片](assets/8ce0c392b304.png)

您也可以在您的CSR8000v上查看，P2S VPN的网段已经自动通告进入了：

![图片](assets/e3765f59da15.png)

**3.2 vWAN as Backbone**

您也可以将多个Region的vWAN Hub CSR8000v通过Azure的骨干进行连接：

![图片](assets/af19f16d4564.jpg)

**3.3 其它云**

由于Azure提供了非常棒的技术支持，所以只做了Azure的demo，另外vWAN Hub的确是一个非常不错的设计，而AWS没有账号所以暂且未测试。Google云正在开发中，阿里云和其它云暂时还没有计划，当然针对每个云适配可能也是一个麻烦事，背后其实就是Terraform**来创虚机，然后自动化加入和分发路由而已，我觉得可以和国内多个云合作完善一套可以Call vmanage和云API的DevOps小程序就好了~

其它云需要技术扶贫，我们勾兑一下~