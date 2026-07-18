# 思科SDWAN探秘(4)--初始化控制器

> 作者: zartbot  
> 日期: 2021年1月11日 11:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485003&idx=4&sn=da6d008a9656322d364154008d000c6e&chksm=f9961689cee19f9f08b8b2ce763abec40b7d04aa2fffff10953b23ad32b2ee91346726f66be4#rd

---

1. 初始设置

****1.1 地址规划****

在公有云环境中，vManage、vBond和vSmart地址都在NAT后，如下图所示，因此vManage在添加vBond和vSmart时务必使用公网地址。同时建议vSmart和vBond配置弹性公网IP固定公网地址

![图片](assets/028b285db803.png)

如果是私有云环境，建议前面也放一个防火墙做NAT同时过滤一些不必要的流量和WAF保证vManage Portal的安全：）都采用NAT部署会是一个好习惯。当然您也可以选择在公网接口上关闭HTTPS等业务，使用管理口VPN拨入堡垒机的方式运维。

**1.2 修改vManage数据库密码**

以ssh登陆后，您还需要更改Neo4j**数据库的密码，具体步骤如下,使用如下命令停止nms服务

```
request nms application-server stop
```

查询密码并更改它，然后重启NMS**服务

![图片](assets/279a4c350374.png)

**1.3 SiteID等编址**
网元
Site-ID
SystemIP
VPN 0公网地址
vmanage
100001
1.0.0.1114.55.101.236
vSmart
100101
1.0.0.101
120.27.248.89
vBond100201
1.0.0.201
101.37.116.217

**1.4 vManage配置**

```
conf tsystem system-ip             1.0.0.1 site-id               100001 organization-name     CiscoDMZ clock timezone Asia/Shanghai vbond 101.37.116.217commit建议新配置一个新的netadmin账号system  aaa   user zartbot password <pwd>   group netadmin   commit   
```

**1.5 vSmart配置**

```
conf tsystem system-ip             1.0.0.101 site-id               100101 organization-name     CiscoDMZ clock timezone Asia/Shanghai vbond 101.37.116.217commit
```

**1.6 vBond配置**

```
conf tsystem host-name vBond system-ip             1.0.0.201 site-id               100201 organization-name     CiscoDMZ clock timezone Asia/Shanghai vbond 101.37.116.217 localcommit
```

2. 设置控制器

****2.1 证书配置原因****

由于整个SDWAN控制器采用零信任的方式部署，因此所有的通信需要有签名的证书，这也是安装会感觉到繁琐的原因，特别是有些企业网络团队没有自己的CA服务器**的时候，而很多企业通常也因为第三方的证书服务器怕运维麻烦，或者找应用部门产生证书也是一个跨部门协作的事情比较头疼。

因此通常我们会选择让客户网络部门自建CA服务器的方式。

**2.2 证书服务器安装**

我来教你一个非常简单的处理方式，我通常喜欢使用MobaXterm，因为登录后会自动启用scp服务传输文件，因此您可以ssh登录到vmanage上，然后使用vshell登录到shell模式，我们就可以在这个linux shell中构建证书服务器了，然后按照如下方式进入Openssl并输入一下命令即可构建自签名根证书,注意在第四行的时候，O=CiscoDMZ要和vmanage里面配置的Orgnization-name一致，而OU建议留空，**注意一定要填写Organization属性（O=）该属性丢弃会导致后续的vManage签发的证书无法安装到CSR8000v等虚拟设备上。**

```
vmanage# vshellvmanage:~$ opensslOpenSSL> genrsa -out rootCA.key 4096OpenSSL> req -new -x509 -days 3650 -key rootCA.key -out rootCA.crt -subj "/C=CN/ST=CN/L=Shanghai/O=CiscoDMZ/CN=sdwan.cisco.com/emailAddress=none@cisco.com"OpenSSL> x509 -in rootCA.crt -out root-ca-chain.pem -outform PEM
```

![图片](assets/5b26d43932d4.png)

生成好以后，请注意将rootCA.key、rootCA.crt、root-ca-chain.pem拷贝出来备份。然后https登录到vmanage，选择“Administration->Settings”界面：

![图片](assets/d11d02e3e3f6.png)

选择vBond，点击右侧“Edit”，配置vBond地址，同时你需要检查Organization Name是否配置好，如果没有配置请补上

![图片](assets/93d091ecd499.png)

然后选择“Controller Certificate Authorization”，点击右侧Edit，选择“Enterprise Root Certificate”，然后点击“Select a file”选择您刚下载的“root-a-chain.pem", 上传，然后**切忌不要勾选“Set CSR Properties”！**切忌不要勾选“Set CSR Properties”！****切忌不要勾选“Set CSR Properties”！****如下所示：

![图片](assets/308edce53adf.png)

然后点击Import&Save即可，然后接下来点击“Configuration”->"Devices"，如下图所示，

![图片](assets/f23efcdac8eb.png)

然后点击“Controllers”子页面，选择“Add Controller”添加vBond和vSmart，如下图所示：

![图片](assets/e365ae90f8f9.png)

添加界面如下,记住使用公网地址！！！：

![图片](assets/34011925dbf0.png)

![图片](assets/e924a41fd901.png)

然后点击左侧菜单栏“Configuration”->"Certificates"，并选择Controllers子页，如下图所示：

![图片](assets/981cfdfa0f9b.png)

分别选择右侧每个设备“...”字样按钮，点击后会有下拉菜单，

![图片](assets/cdaa9bf1adba.png)

选择Generate CSR，然后就会弹出一个窗口，然后点击Download即可，并将下载好的CSR根据设备改名为vmanage.csr、vbond.csr、vsmart.csr,

![图片](assets/64a1afa23fc1.png)

然后将他们都通过mobaXterm上传到vmanage vshell开着的那个窗口，如下图所示，直接拖入到左边文件列表窗口就可以上传了：

![图片](assets/30a893446c9a.png)

然后通过如下命令在窗口中签发证书：

```
OpenSSL>x509 -req -days 1000  -CA rootCA.crt -CAkey rootCA.key -set_serial 02 -in vmanage.csr -out vmanage.pemOpenSSL>x509 -req -days 1000  -CA rootCA.crt -CAkey rootCA.key -set_serial 03 -in vsmart.csr -out vsmart.pemOpenSSL>x509 -req -days 1000  -CA rootCA.crt -CAkey rootCA.key -set_serial 04 -in vbond.csr -out vbond.pem
```

![图片](assets/9563b19a2706.png)

然后将这三个文件拷贝出来，回到vmanage webui，然后点击右上角“Install Certificate”，选择下载的pem文件安装即可：

![图片](assets/c2c441e19bfa.png)

安装需要等待一会儿，待显示Success了再安装第二台，以vmanage、vbond、vsmart的顺序安装:

![图片](assets/177f11c566d5.png)

安装完成后，您看到的证书状态就如下图所示了：

![图片](assets/89912521bd6c.png)

最后一步，就是在vmanage、vbond、vsmart的传输接口上配置tunnel-interface即可, 唯一需要注意的是要在vbond上配置encapsulation ipsec,如果您需要公网访问，请在tunnel-interface配置后，添加allow-service sshd，当然建议是不添加：）

```
**Vmanage**vpn 0interface eth1 <-此接口为配置在vpn0中使能的接口名 tunnel-interface   allow-service sshd <-可选commit 
```

```
**vsmart**vpn 0interface eth1 <-此接口为配置在vpn0中使能的接口名 tunnel-interface   allow-service sshd <-可选commit
```

```
**vbond**vpn 0interface ge0/0 tunnel-interface   encapsulation ipsec   allow-service sshd <-可选commit
```

配置完成后，你就可以看到所有的控制器都上线了：

![图片](assets/1e8d01da505e.png)

**3. 安装边缘设备**

**3.1 安装serial文件**

Serial file有两种方式，如果您都是物理设备也不需要PnP**零配置开局，可以手工生成一个csv文件，当然这样的场景较少，我们来看支持PnP的方式，登录到software.cisco.com 选择“Plug and Play Connect”，点击“Controller Profiles”，然后选择“+Add Profile”，如下图示：

![图片](assets/37d83bf416e5.png)

Controller Type选择vBond，然后点击下一步：

![图片](assets/74d4495d3c0c.png)

配置如下，证书选择root-ca-chain.pem上传即可：

![图片](assets/cdfbb8f63faf.png)

然后再点入devices页面，根据您的设备序列号点击”+Add Devices"或者购买的虚机版本点击“+Add Software Devices”，输入相应的信息即可。

![图片](assets/e76ab99b31fe.png)

最后回到Controller Profiles页面，点击“Provisioning File”就会自动下载Serial文件，选择文件版本为18.3 and newer：

![图片](assets/dc4966410320.png)

然后在vmanage中点击"Configuration->Devices"选择“Upload WAN Edge List”，

![图片](assets/205ed392f43a.png)

然后选择刚刚下载的文件，并且勾选“validate the uploaded vEdge list and send to controllers”

![图片](assets/edbe8a22eee8.png)

上传完毕后点击"Configuration->Devices“就可以看到已经支持的设备列表了，这些都有一次性的Token用于OTP开局。

![图片](assets/62e07b2d9d2b.png)

接下来您可以查看《思科SDWAN探秘(5)--设备上线》