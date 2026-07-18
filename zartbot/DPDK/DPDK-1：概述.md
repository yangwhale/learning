# DPDK-1：概述

> 作者: zartbot  
> 日期: 2021年5月19日 15:31  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485848&idx=1&sn=cb3e4ae3a65a40458cbf0f257dc51479&chksm=f996195acee1904c6445aefb6758aa0c8065242ae90aec644349514e2f4e40a941cd8c181d27#rd

---

❝
这个dpdk的系列文章可能要连续更新几十篇, 写DPDK这个系列的原因是, 网上找的资料大多数比较零散, 涉及的dpdk版本也很多,而dpdk本身也在快速的发展中,因此基本上会根据最新的DPDK-21.02或者后期21.05的版本来写一个guide,主要包含一些基本的介绍.
❞
具体的教材其实Intel已经写了很好的一本书了，所以DPDK的基本原理什么的就不多写了，主要后面还是从基本的收发包开始，逐个的写一些常用的API的使用，以实战的方式实现Ruta数据面。

![图片](assets/5be53976f150.jpg)

### 用途 

主要原因是内部一个项目的控制器需要数Mpps**的控制信令,而基于Go的udp socket性能不够,XDP似乎也没有很好的库，基于QAT这些加解密的offload也不行，同时Ruta等项目也需要类似的支持, 还有很多合作伙伴要求在基于Octeon TX2或者未来TX3的智能网卡上实现Ruta，因此DPDK是最好的选择（当然还有针对运营商的P4交换机支持，后面再写一个系列)，而Intel nff-go基本上已经没什么维护了,用nff-go写的一个控制信令发包程序也只能到2Mpps, 因此决定还是用C直接调DPDK来写吧，而Ruta似乎也不需要VPP**很多复杂的框架，某个合作伙伴似乎也不太需要vpp的很多冗杂的功能，当然后期可能还会参考一些vpp的东西。

### 环境设置 

在公司的电脑是一台双路Xeon Platium 8259CL的工作站，然后有XL710和MLX CX5的网卡，另外还QAT的加速卡，当然另一个合作伙伴还有OcteonTX2的需求，这个也准备了一台，这些是用来做Ruta的. 家里是一个nvidia Jetson Xavier配了一个PCIe的多网卡以及一个R4S小路由来做Ruta边缘计算节点场景。同时还有一个vmware的虚机来开发.. 下面以虚机环境为例介绍.

主要要求是打开IOMMU和配置多个网卡

![图片](assets/deeaecb75534.png)

很多演示程序需要4个网卡，同时为了方便，记得在vmx配置文件中将网卡类型改为vmxnet3

```
ethernet4.connectionType = "custom"ethernet4.addressType = "generated"ethernet4.vnet = "VMnet12"ethernet4.displayName = "VMnet12"ethernet4.virtualDev = "vmxnet3"
```

### 安装和编译环境配置 

安装`ubuntu 20.04.2`，然后安装的包如下,基本的编译环境，外加了我比较喜欢用terminator和tmux

```
sudo apt updatesudo apt upgradesudo apt install openssh-server terminator tmux build-essential
```

然后修改grub，打开iommu

```
 sudo vi /etc/default/grub//原来GRUB_CMDLINE_LINUX_DEFAULT="quiet splash" //修改后GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt intel_iommu=on" 
```

最后更新grub，然后重启

```
sudo grub-mkconfig -o /boot/grub/grub.cfgsudo reboot
```

### 编译dpdk 

下载dpdk并解压

```
wget http://fast.dpdk.org/rel/dpdk-20.11.1.tar.xztar xf dpdk-21.02.tar.xzcd dpdk-21.02
```

安装meson和pyelftools

```
sudo apt install  mesonsudo apt install python3-pyelftools
```

编译和安装

```
meson -Dexamples=all buildcd buildninjasudo ninja installsudo ldconfig
```

### 配置hugepages和绑定vfio 

```
sudo modprobe vfio-pcisudo dpdk-hugepages.py --setup 2Gsudo dpdk-devbind.py --statusNetwork devices using kernel driver===================================0000:03:00.0 'VMXNET3 Ethernet Controller 07b0' if=ens160 drv=vmxnet3 unused=vfio-pci *Active*0000:04:00.0 'VMXNET3 Ethernet Controller 07b0' if=ens161 drv=vmxnet3 unused=vfio-pci 0000:0b:00.0 'VMXNET3 Ethernet Controller 07b0' if=ens192 drv=vmxnet3 unused=vfio-pci 0000:13:00.0 'VMXNET3 Ethernet Controller 07b0' if=ens224 drv=vmxnet3 unused=vfio-pci 0000:1b:00.0 'VMXNET3 Ethernet Controller 07b0' if=ens256 drv=vmxnet3 unused=vfio-pci sudo dpdk-devbind.py -b vfio-pci ens161sudo dpdk-devbind.py -b vfio-pci ens192sudo dpdk-devbind.py -b vfio-pci ens224sudo dpdk-devbind.py -b vfio-pci ens256
```

### 测试Helloworld 

```
cd examplessudo ./dpdk-helloworld 
```

```
EAL: Detected 32 lcore(s)EAL: Detected 1 NUMA nodesEAL: Detected static linkage of DPDKEAL: Multi-process socket /var/run/dpdk/rte/mp_socketEAL: Selected IOVA mode 'VA'EAL: No available 1048576 kB hugepages reportedEAL: Probing VFIO support...EAL: VFIO support initializedEAL:   Invalid NUMA socket, default to 0EAL:   Invalid NUMA socket, default to 0EAL:   0000:04:00.0 VFIO group is not viable! Not all devices in IOMMU group bound to VFIO or unboundEAL: Requested device 0000:04:00.0 cannot be usedEAL:   Invalid NUMA socket, default to 0EAL:   using IOMMU type 1 (Type 1)EAL: Ignore mapping IO port bar(3)EAL: Probe PCI driver: net_vmxnet3 (15ad:7b0) device: 0000:0b:00.0 (socket 0)EAL:   Invalid NUMA socket, default to 0EAL: Ignore mapping IO port bar(3)EAL: Probe PCI driver: net_vmxnet3 (15ad:7b0) device: 0000:13:00.0 (socket 0)EAL:   Invalid NUMA socket, default to 0EAL: Ignore mapping IO port bar(3)EAL: Probe PCI driver: net_vmxnet3 (15ad:7b0) device: 0000:1b:00.0 (socket 0)EAL: No legacy callbacks, legacy socket not createdhello from core 1hello from core 2hello from core 3hello from core 4hello from core 5hello from core 6hello from core 7hello from core 8hello from core 9hello from core 10hello from core 11hello from core 12hello from core 13hello from core 14hello from core 15hello from core 16hello from core 17hello from core 18hello from core 19hello from core 20hello from core 21hello from core 22hello from core 23hello from core 24hello from core 25hello from core 26hello from core 27hello from core 28hello from core 29hello from core 30hello from core 31hello from core 0zartbot@zarbotNet:~
```