# 云网一体，端网融合：思科NFVIS探秘

> 作者: zartbot  
> 日期: 2021年2月2日 11:00  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485191&idx=2&sn=55294e7421a6125f0c66af2052117e6c&chksm=f99617c5cee19ed3347d81a91e7447f634df980f27de2cc8deb80b0c0a24f5c426515b64b444#rd

---

现有的企业网架构上云已经是一个必然选择，云网通过SDWAN的融合也是大势所趋，然而端侧还存在很多挑战，多设备组网环境下，管理困难，很多升级更换等操作都需要到现场施工:

![图片](assets/bc4d9c8f987e.png)

作为一个老牌的路由器厂商，思科从最早的ISR集成多业务路由器（集成VPN和语音等业务），到后期的ASR多业务聚合路由器（聚合DPI**防火墙、BAS、VPN、SBC、MPLS PE等多种业务）。而在这个云的时代，我们推出了Catalyst Edge 8300、8500及8000v，俨然已经不把自己定位成一个路由器了，因为它顺应云时代的变化，将其定义为云的边缘节点，也拉开了云网端融合的序幕。

![图片](assets/87d63ce4a7dd.png)

在路由器上做虚拟化支持大概也是10年前思科中国研发团队开始的，最早实现了在ASR1000 RP2上运行CallManager和ASA的功能，后期随着转发平面虚拟化诞生了云路由器CSR1000v，针对分支机构计算和网络融合的需求，我们有利用IOS XE系统支持的IOX容器平台，在Catalyst Edge8300上更是扩展到了12核的Intel X86处理，路由功能仅使用了其中4个核心，而针对需要更多计算资源的客户，我们可以采用外挂UCS-E系列微型服务器模块的方式。当然还有今天的主角NFVIS，即完全将一个基于CentOS的操作系统安装在ENCS或者UCS标准服务器平台上，然后以虚拟化的方式构建VNF、VM及容器，并通过支持SDWAN的统一虚拟路由器**平台CSR8000v链接到云环境。

![图片](assets/af7f6755ba3f.png)

因此针对不同客户的需求，我们可以实现基于网络面为主的虚拟化平台Catalyst8300，和基于虚拟化层为主的通用服务器平台NFVIS，Catalyst 8300、8500我会在后面的文章中更新，今天主要来谈谈NFVIS。

**1. NFVIS部署场景
**

NFVIS可以理解成为一个支持标准虚拟化的CentOS系统并携带了思科的大量北向管理工具，可以适配思科DNA-Center、vManage、NSO等控制器，由于标准化的接口您也可以使用Ansbile等工具进行编排和运维：

![图片](assets/6cb7b140f697.png)

针对不同的业务场景，我们可以灵活的支持4Core嵌入式处理器- 56Core双路Xeon Scalable处理器的平台以满足不同用户的需求

![图片](assets/dce5d73a6ad3.png)

网络上由于原生支持Hypervisor**层直接建立IPSec VPN隧道到云端管理平面，可以实现安全的带外远程管理和编排任务。今天我们来看看使用最多的企业网络计算融合系统（Enterprise Network Compute System，ENCS），它有两款，一款是AMD 4-Core的5104，另一款是分别支持Intel 6、8、12核的ENCS5400系列：

![图片](assets/4059f1fbf3a2.png)

**成功案例：**

第一个是运营商的uCPE解决方案，主要是看重SDWAN的组网能力和NFVIS北向接口**的管理自动化能力。

![图片](assets/d5b134daad11.png)

第二个案例是一些金融机构，用NFV自动化并可以快速隔离故障，例如升级时我们可以启用第二个C8000v调试正常了以后把流量切过去然后把原来的直接删掉就行了，这样cloud-native的方式避免了大量ISSU的麻烦境地：

![图片](assets/7f1e799763ac.png)

当然除此之外还有客户采用一台uCPE并配合友商的防火墙的，思科一直开放的心态再一次体现出来了，并不会在自己的平台上限制什么。

![图片](assets/2a2bee97d164.png)

**2. ENCS5400介绍**

ENCS5400是我们主打企业网分支机构场景的通用CPE（uCPE），当然国内有些资深“软路由“用户都会采用Atom D525+vmware ESXi+ Lede+iKuai+NAS构建这样的场景：

![图片](assets/22b19097c158.png)

而作为商用产品线，思科考虑到的则是用户在这个场景中的真实需求是什么，通常随着无线WiFi的部署，分支机构对于有线交换机**的端口数需求相对减少，能够接几个无线AP即可，因此我们配备了8口的PoE以太网供电交换机。

![图片](assets/71267d1c1ab7.png)

另一方面分支机构有一些本地文件共享的NAS的需求，我们有内置的M.2 SSD，也配置了两个外置的硬盘插槽并可以实现RAID。同时针对客户的无线LTE备份或者一些传统E1线路上联需求，我们也提供了NIM插槽用于支持各种广域网模块。它的内部网络链接如下图所示：

![图片](assets/9d9751aea0fa.png)

您可能会注意到基于硬件卸载的VM-VM通信，没错，其实就是SR-IOV，在创建虚机时可以直接指定VF。

![图片](assets/7b852cf12704.png)

3. NFVIS软件架构

反正各位安装的时候都可以看到NFVIS就是基于CentOS的，虚拟化层也和各个公有云差不多，操作界面也类似，但是和其它平台不同的是，针对混合云和端网融合的场景，我们北向管理做了很多工作：

![图片](assets/47b8a5bdf055.png)

您可以注意到，当你新安装一台机器时可以通过PnP实现ZTP开局，另一方面可以通过DNA-Center、vManage及NSO等各个平台进行统一管理（这些演示实验我后面会给大家一个一个的补），当然本地的WebUI也会有介绍和演示，最特别的当属它还配备了一个轻量化的IPSecVPN模块用于虚拟化层链接云端，后期还会针对Azure vWAN等场景提供BGP路由发布的支持。

虚拟化层上，我们也是保证一个完全开放的心态，您可以随心所欲地安装您的各种应用，当然我们在网络上会给您提供SDWAN路由器，ASA/FTD下一代安全防火墙、广域网加速的vWAAS和虚拟化的无线网络控制器等多种VNF，另一方面我们也官方支持Paloalto、checkpoint等多家友商的安全设备满足您安全异构的需求。

![图片](assets/a8df143c7f4a.png)

4. ENCS+NFVIS实战

最鄙视有些厂家只喜欢写软文，是产品就要教会客户怎么用，所以我们进入实战环节。这个ENCS实际上您可以认为它是一个服务器，有标准的BMC模块，也就是思科自己的CIMC：

![图片](assets/7f81efe7d495.png)

当然我们还有基于这套CIMC构建的裸金属云解决方案Intersight，这个后面有空跟大家介绍一下， 前段时间我拿一张8年前的CIMC卡冒充支持裸金属的智能网卡骗了不少资深的同事：

![图片](assets/077380b0aad7.jpg)

**4.1 链接和使用CIMC**

ENCS的接口相对复杂，因此我们先来看看如何连接：

![图片](assets/638a3068d4cc.png)

标记红色的为串口，通常使用串口线连接，我通常喜欢使用一个蓝牙的串口小设备：

![图片](assets/5aca73dfc692.jpg)

登录默认用户名密码为：admin/password.串口波特率和其它网络设备一样：9600，登录后要求您修改密码，然后您可以按如下方式配置CIMC 网口DHCP（Scope cimc、scope network、set dhcp-enabled yes、commit）：

![图片](assets/cd757cb46e3a.png)

完成后连接网线可以通过如下方式查询地址：

![图片](assets/9585dc013a1d.png)

完成后就可以通过网页远程登陆到CIMC管理界面了，然后点击右上方KVM就可以启动远程界面了：

![图片](assets/4612f11ebc4b.jpg)

里面可以使用菜单中的虚拟介质，激活虚拟设备、映射CD/DVD关联NFVIS光盘。

**4.2 安装NFVIS**

当然您也可以采用原始的方法连接VGA显示器、制作U盘引导盘、使用USB HUB同时连接键盘和优盘启动来安装，完全抛弃CIMC。**我们强烈建议您第一次装机的时候使用优盘的方式，因为默认我们会刷新系统的Firmware、更新BIOS、更新CIMC，并且记得在第一次安装的时候时间非常长（大概一个小时以上），千万不要随意强行重启机器。**

安装引导和其他操作系统类似，重启后选择F6进入Boot选择菜单，然后选择您的优盘即可，然后菜单中有一个选项就是“Install NFVIS”，然后安装就会自动进行，在安装过程中，您会看到似乎系统假死，实际上是有些安装进度的日志被重定向到了串口，您可以连接到标有蓝色“Console”的串口上，检查安装进度。**安装完成后，系统会自动重启，然后注意千万不要随意关机，在重启后的CentOS postinstall脚本中我们会自动检查BIOS、CIMC的版本并进行升级，这个时候您能会发现系统会自动重启关机等操作，切记不要进行任何操作，静待系统完成升级，最后安装完成后会显示nfvis：字样**

![图片](assets/def7946b6668.png)

然后您可以使用默认密码 admin、Admin123#登录，登陆完成后会强行要求您更改密码。默认管理接口配置了192.168.1.1/24的地址，您可以有线网卡配置192.168.1.x连接到MGMT CPU接口就可以Web访问了。命令行模式可以通过ssh到管理口或者继续使用串口，广域网IP地址为DHCP动态分配，当然您也可通过config t，然后按如下方式配置广域网接口IP地址：

![图片](assets/e51cee68b410.png)

由于这个配置界面采用了confd，注意和传统的IOS不同，需要使用显示的commit确认配置。配置完成后退出可以通过show system settings看到广域网地址：

![图片](assets/4c6e3f040d02.png)

其实您可以留意一下，本质上它是将GE0-0口桥接到了ovs的wan-br上，而GE0-1口桥接到了wan2-br上，如果您需要配置第二个广域网口则配置wan2就可以了，可以通过如下命令看到ovs的信息：

![图片](assets/d56a3b500a9e.png)

4.3 配置NFVIS和SDWAN**
**

通过WebUI输入前面使用的admin用户名登录：

![图片](assets/161047db33fc.png)

登录后的界面如下：

![图片](assets/166c79735e71.png)

主要菜单集中在左端的VM Life Cycle里，点击“Image Repository”就可以上传image了，和很多公有云操作一致，注意下图右上角红色按钮下面有一个摄像头的标签，里面有录制好的教学视频：）

![图片](assets/58751ee77484.png)

这个image的tar.gz需要一个生成工具，您可以基于一个qcow2映像文件配合一个XML定义生成，思科也配置了相应的生成工具，下一节我们会详细叙述。

这里我们采用思科已经打包好的CSR8000v映像构建SDWAN功能，构建完成后会自动根据内置的XML文件产生3个profile：

![图片](assets/a4bffea0be7d.png)

点击Deploy，将Router拖入下框，并连线，您可以选择连接OVS的网桥或者SR-IOV直接创建一个VF，请注意在此您需要输入VF-ID，并记住这些对应的路由器的接口GigabitEthernetX

![图片](assets/808894ccf725.png)

配置完网络后，再次点击路由器，选择Mode为vMange SDWAN模式，然后SDWAN控制器vmanage上获取空余的uuid和一次性Token（OTP）然后填入右表, 然后根据SDWAN的要求填入SystemIP VBOND地址，VPN0广域网接口地址等参数即可，最后点击“Deploy”

![图片](assets/188603eb531e.png)

点击VM LifeCycle Manage 可以看到虚机的状态，并且可以点击红框显示器图标，打开VNC：

![图片](assets/d31340df883d.jpg)

就此可以看到虚拟路由器Catalyst8000v已经启动了,然后它会自动完成上线并加入到vManage的管理中

![图片](assets/6c3defed7017.png)

唯一需要注意的是需要使用默认用户名密码admin、admin登录，然后改密码：（config-transaction、username admin secret xxxx、commit）

内置网络LAN口是直接trunk接到交换机上的，交换机也可以图形化配置：

![图片](assets/8a5082bdb4ca.png)

5. NFVIS运维

这一部分只是一些简单的演示，因为我觉得思科自己开源出来的ansible的role不算非常好用，接下来有空我会自己再写一套。先拿思科开源的ansible-nfvis用着吧， build image的playbook如下所示：

![图片](assets/da29bef89b66.png)

build好了以后会产生一个ubuntu.tar.gz，然后我们可以用upload_img.yml这个playbook上传，记得上传的时候要打开scp服务：

![图片](assets/ad86f39a6e4a.png)

upload的playbook非常简单:

```
- hosts: nfvis  connection: local  gather_facts: no  roles:    - ansible-nfvis  tasks:   - name: Upload Package     nfvis_package:       host: "{{ ansible_host }}"       user: "{{ ansible_user }}"       password: "{{ ansible_password }}"       file: packages/ubuntu.tar.gz       name: ubuntu       state: present
```

然后我们部署一下？

```
- hosts: nfvis  connection: local  gather_facts: no  roles:    - ansible-nfvis  tasks:   - name: Deploy VM     nfvis_deployment:       host: "{{ ansible_host }}"       user: "{{ ansible_user }}"       password: "{{ ansible_password }}"       name: server       state: present         image: ubuntu       flavor: ubuntu-small       interfaces:         - network: lan-net
```

![图片](assets/77f6b5bf2537.png)

完工以后，我们在NFVIS上就能看到这个虚机了,点击VM Monitoring就可以看到一些虚机的使用情况了。

![图片](assets/e1c3bd74f228.png)

当然日常运维可以用ansible来做：

![图片](assets/312c2fbe4752.png)

当然这些脚本的运维逻辑还有点小问题，如何跟用户应用部门的编排集成还有待加强，这也是我接下来几个月的小作业之一。

今日技术扶贫结束，其实边缘云也好，云网融合也好，端网合一也好，缺的是用心去做，用心的去看用户不方便的地方，然后跟用户一起成长，帮用户偷懒。NFVIS现在通过整合NSO、Netconf有了很强的运维能力，接下来我们还会进一步优化Ansible等开源工具以及其它非思科的编排器，当然也不排除把NFVIS和其它友商服务器智能网卡整合的场景：）

当然针对边缘的K8S和容器服务，我们也会在NFVIS 4.6中（大概明年中期）支持，同时SDWAN的CSR8000v虚拟路由器已经完美的和AWS、Azure整合，正在加班加点和阿里云集成~，未来是云，还是端？还是哪~ 你想，我做~

另一方面作为研究项目，我也会把Ruta作为一个NFV运行在NFVIS上提供更轻量化的SDWAN路由能力~，未来真的可期~