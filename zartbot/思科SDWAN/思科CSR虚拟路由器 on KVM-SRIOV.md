# 思科CSR虚拟路由器 on KVM/SRIOV

> 作者: zartbot  
> 日期: 2021年2月2日 11:00  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485191&idx=1&sn=6efbb9e19f85e1734d1288334c28fde3&chksm=f99617c5cee19ed35ed077e1d5ce2b53f878dc8ba2f1a24aecd701d0a35d043cfc103076ac99#rd

---

以前我们介绍过思科的KVM平台NFVIS操作系统和ENCS系列硬件平台**《**[**云网一体，端网融合：思科NFVIS探秘**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484748&idx=4&sn=dc31f2bf07b09f826b29b23e7419cd12&chksm=f996158ecee19c98085c6e1d8f901924568b1a3207d023c455cd10559279803fdc6756137a30&scene=21#wechat_redirect)**》**这些平台上原生支持SRIOV可以获得高性能的转发。当然您也可以采用CentOS、Ubuntu等系统自己安装KVM和SRIOV来获得同样的转发能力，并将CSR虚拟路由器和您已有的OpenStack等集群很好的整合在一起。

思科运营商业务团队的同事饶维波最近给客户测试了CSR虚拟路由器并为大家准备了这份在开源平台上的安装KVM，并安装和设置CSR1000v、Catalyst8000v的指南，内容包括SRIOV、KVM性能调优及CSR1000v初始化配置等。

      
     
       
         
           
             
                                

                 
                   
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
   
 

https://github.com/weiborao/CSR1KV-Installation-on-KVM-with-SRIOV

**1.CentOS7 安装及设置**

**1.1 BIOS设置建议**

Configuration

Recommended Setting

Intel Hyper-Threading  Technology

Disabled

Number of Enable Cores

ALL

Execute Disable

Enabled

Intel VT

Enabled

Intel VT-D

Enabled

Intel VT-D coherency support

Enabled

Intel VT-D ATS support

Enabled

CPU Performance

High throughput

Hardware Perfetcher

Disabled

Adjacent Cache Line Prefetcher

Disabled

DCU Streamer Prefetch

Disable

Power Technology

Custom

Enhanced Intel Speedstep  Technology

Disabled

Intel Turbo Boost Technology

Enabled

Processor Power State C6

Disabled

Processor Power State C1  Enhanced

Disabled

Frequency Poor Override

Enabled

P-State Coordination

HW_ALL

Energy Performance

Performance

以上来自CSR1000v的安装指南

**1.2 CentOS 7安装**

在安装的时候请您勾选一下组件

Server with GUI

Virtualization Client

Virtualization Hypervisor

Virtualization Tools

启动完毕以后关闭selinux，重启生效。

```
sed -i 's/SELINUX=enforcing/SELINUX=disabled/g' /etc/selinux/configgetenforce    //结果为：Enforcing（开启状态）disabled(关闭状态)
```

安装完后，SSH登录可能显示中文，可修改 .bash_profile

```
LANG="en_US.UTF-8"export LANG
```

```
source .bash_profile
```

检查CPU信息并关闭防火墙

```
egrep -o '(vmx|svm)' /proc/cpuinfo | sort | uniq注：在生产环境中，需要在服务器连接的交换机以及出口防火墙上做好安全策略。systemctl stop firewalldsystemctl disable firewalld
```

检查sysctl.conf

```
cat /etc/sysctl.confnet.ipv4.ip_forward = 1net.bridge.bridge-nf-call-ip6tables = 0net.bridge.bridge-nf-call-iptables = 0net.bridge.bridge-nf-call-arptables = 0
```

检查KVM组件版本：

```
[root@centos7 ~]# libvirtd -Vlibvirtd (libvirt) 4.5.0[root@centos7 ~]# /usr/libexec/qemu-kvm --versionQEMU emulator version 1.5.3 (qemu-kvm-1.5.3-173.el7_8.3), Copyright (c) 2003-2008 Fabrice Bellard[root@centos7 ~]# virt-manager --version1.5.0[root@centos7 ~]# modinfo kvm-intelfilename:       /lib/modules/3.10.0-1127.18.2.el7.x86_64/kernel/arch/x86/kvm/kvm-intel.ko.xz[root@centos7 ~]# modinfo ixgbevffilename:       /lib/modules/3.10.0-1127.18.2.el7.x86_64/kernel/drivers/net/ethernet/intel/ixgbevf/ixgbevf.ko.xzversion:        4.1.0-k-rh7.7
```

1.3 创建本地Yum源(可选)

备份本地/etc/yum.repos.d 目录下的yum源

```
cd /etc/yum.repos.d/mkdir bakmv C* bak/
```

上传CentOS-7-x86_64-Everything-2009.iso镜像到/opt

```
mkdir -p /media/cdrommount -t iso9660 -o loop /opt/CentOS-7-x86_64-Everything-2009.iso /media/cdrom/mount    //查看挂载信息df -hvi /etc/fstab/opt/CentOS-7-x86_64-Everything-2009.iso    /media/cdrom    iso9660 loop 0 0tail -1 /etc/fstab    //查看是否写入/etc/fstab
```

配置本地yum源

```
cd /etc/yum.repos.d/vi local.repo[local]name=localbaseurl=file:///media/cdrom     //前面的file://是协议,后面的/media/cdrom是光盘挂载点gpgcheck=0    //1使用公钥验证rpm包的正确性,0不验证enabled=1    //1启用yum源,0禁用yum源
```

安装numactl和telnet

```
yum install -y numactl telnet
```

运行virt-manager启动图形化界面。

如果对virsh CLI命令熟悉，可以使用virsh 命令创建虚拟机。

1.4 服务器网卡配置（基于NetworkManager配置）

在终端界面，可以通过nmtui打开图形化界面进行设置；以下使用nmcli进行设置。

```
nmcli connection add con-name eno1 type ethernet autoconnect yes ifname eno1nmcli connection modify eno1 ipv4.method manual ipv4.addresses 10.75.58.43/24 ipv4.gateway 10.75.58.1 ipv4.dns 64.104.123.245nmcli connection up eno1 nmcli connection show eno1ping 10.75.58.1
```

上述命令完成后，在/etc/sysconfig/network-scripts 中会生成网卡的ifcfg配置文件。

```
cat /etc/sysconfig/network-scripts/ifcfg-eno1HWADDR=70:7D:B9:59:5B:AETYPE=EthernetPROXY_METHOD=noneBROWSER_ONLY=noBOOTPROTO=noneIPADDR=10.75.58.43PREFIX=24GATEWAY=10.75.58.1DNS1=64.104.123.245DEFROUTE=yesIPV4_FAILURE_FATAL=noIPV6INIT=yesIPV6_AUTOCONF=yesIPV6_DEFROUTE=yesIPV6_FAILURE_FATAL=noIPV6_ADDR_GEN_MODE=stable-privacyNAME=eno1UUID=2a1c8b39-7f44-321b-a65f-a93e70ab0616ONBOOT=yesAUTOCONNECT_PRIORITY=-999DEVICE=eno1
```

此时，可以将network.service停止和关闭。

```
systemctl stop networksystemctl disable network
```

注意，如果NetworkManager未设置妥当，执行systemctl stop network后，会导致服务器无法管理。

准备开启SRIOV的网卡设置，以eno2为例：

```
nmcli connection add con-name eno2 type ethernet autoconnect yes ifname eno2nmcli connection modify eno2 ethernet.mtu 9216 ipv4.method disablednmcli connection up eno2nmcli connection show eno2ip link show dev eno2
```

注：上述MTU值设置为9216是借鉴自Cisco NFVIS平台，如下：

```
CSP5228-1# show pnic-detail mtuName          MTU=============================eth0-1        9216eth0-2        9216eth1-1        9216eth1-2        9216
```

1.5 配置Linux网桥（可选）

网桥br1 配置示例：

```
nmcli connection add con-name br1 type bridge autoconnect yes ipv4.method disabled ethernet.mtu 9216 ifname br1nmcli connection up br1ip link show dev br1
```

网桥br1的物理网卡配置

```
nmcli connection add con-name eno5 type ethernet autoconnect yes ifname eno5nmcli connection modify eno5 ethernet.mtu 9216 ipv4.method disabled master br1nmcli connection up eno5ip link show dev eno5
```

创建net-br1网络

```
[root@centos7 ~]# cat net-br1.xml<network>  <name>net-br1</name>  <forward mode="bridge"/>  <bridge name="br1"/></network>[root@centos7 ~]# virsh net-define net-br1.xmlNetwork net-br1 defined from net-br1.xml[root@centos7 ~]# virsh net-start net-br1Network net-br1 started[root@centos7 ~]# virsh net-autostart net-br1Network net-br1 marked as autostarted
```

2.配置SR-IOV

![图片](assets/e84a241a8b6f.png)

2.1 检查网卡对SR-IOV的支持，并配置网卡

**lshw -c network -businfo**

```
Bus info          Device      Class          Description========================================================pci@0000:1d:00.0  eno5        network        VIC Ethernet NICpci@0000:1d:00.1  eno6        network        VIC Ethernet NICpci@0000:1d:00.2  eno7        network        VIC Ethernet NICpci@0000:1d:00.3  eno8        network        VIC Ethernet NICpci@0000:3b:00.0  eno1        network        Ethernet Controller 10G X550Tpci@0000:3b:00.1  eno2        network        Ethernet Controller 10G X550T
```

上文中的最后一块网卡的BusInfo 3b:00.1用于以下命令****

**lspci -vv -s 3b:00.1 | grep -A 5 -i SR-IOV**

```
Capabilities: [160 v1] Single Root I/O Virtualization (SR-IOV)    IOVCap:  Migration-, Interrupt Message Number: 000    IOVCtl:  Enable+ Migration- Interrupt- MSE+ ARIHierarchy-    IOVSta:  Migration-    Initial VFs: 64, Total VFs: 64, Number of VFs: 8, Function Dependency Link: 01    VF offset: 128, stride: 2, Device ID: 1565
```

2.2 设置启动参数

vi /etc/default/grub

GRUB_CMDLINE_LINUX="crashkernel=autospectre_v2=retpoline rd.lvm.lv=centos/root rd.lvm.lv=centos/swap rhgb quiet hugepagesz=1G hugepages=32 default_hugepagesz=1G intel_iommu=on iommu=pt isolcpus=1-8,37-44"

注：页面数字不要过大，不然启动失败，如果后续不够，可以在运行时添加。

grub2-mkconfig -o/boot/grub2/grub.cfg

grub2-mkconfig -o/boot/efi/EFI/centos/grub.cfg

iommu=pt 参数是将SRIOV设备支持PCI Passthrough

重启后验证

cat /proc/cmdline |grepintel_iommu=on

dmesg |grep -e DMAR -eIOMMU

dmesg | grep -e DMAR -eIOMMU -e AMD-Vi

**default_hugepagesz=1G hugepagesz=1G hugepages=32参数设置主机在启动时分配32个1GB的内存大页，这些是静态内存大页。 CSR 1000v虚拟机将试用这些静态大页以获得最优性能。**

** ****isolcpus=1-8,37-44参数设置的作用是隔离1-8，37-44的CPU核，使其独立于内核的平衡调度算法，也就是内核本身不会将进程分配到被隔离的CPU。之后我们可将指定的进程CSR 1000v虚拟机绑定到被隔离的CPU上运行，让进程独占CPU，使其实时性可得到一定程度的提高。**

**可参考4.1检查平台的能力这个章节获取主机CPU核的相关信息。**

```
[root@centos7 ~]# cat /proc/cmdline |grepintel_iommu=on
```

BOOT_IMAGE=/vmlinuz-3.10.0-1127.el7.x86_64root=/dev/mapper/centos-root ro crashkernel=auto spectre_v2=retpolinerd.lvm.lv=centos/root rd.lvm.lv=centos/swap rhgb quiet hugepagesz=1G hugepages=32default_hugepagesz=1G intel_iommu=on iommu=pt LANG=en_US.UTF-8

```
[root@centos7 ~]# dmesg |grep -e DMAR -e IOMMU
```

[    0.000000] ACPI: DMAR000000005d6f5d70 00250 (v01 Cisco0 CiscoUCS 00000001 INTL 20091013)

[    0.000000] DMAR:IOMMU enabled

查看隔离的CPU核以及所有的CPU核。

```
[root@centos7 ~]# cat /sys/devices/system/cpu/isolated1-8,37-44[root@centos7 ~]# cat /sys/devices/system/cpu/present0-71
```

2.3 通过nmcli持久化VFs配置

nmcli可以设置网卡的sriov参数，如下：

```
nmcli connection modify eno2 sriov.total-vfs 4
```

还可以设置每一个VF设备的MAC地址，便于管理：
nmcli connection modify eno2 sriov.vfs '0 mac=8E:DF:08:C1:D1:DE trust=true, 1 mac=5A:B9:2F:99:A6:CE trust=true, 2 mac=46:78:69:E3:71:3D trust=true, 3 mac=7E:A7:DB:3B:1B:B3 trust=true'
执行上述命令后：

cat/etc/sysconfig/network-scripts/ifcfg-eno2

TYPE=Ethernet

NAME=eno2

UUID=64ffa204-0158-40c8-ba86-2b7aebf27619

DEVICE=eno2

ONBOOT=yes

MTU=9216

HWADDR=70:7D:B9:59:5B:AF

PROXY_METHOD=none

BROWSER_ONLY=no

IPV6INIT=no

SRIOV_TOTAL_VFS=4

SRIOV_VF0="mac=8E:DF:08:C1:D1:DE trust=true"

SRIOV_VF1="mac=5A:B9:2F:99:A6:CE trust=true"

SRIOV_VF2="mac=46:78:69:E3:71:3D trust=true"

SRIOV_VF3="mac=7E:A7:DB:3B:1B:B3 trust=true

重启后，检查dmesg：

**dmesg | grep -i vf | grep -i eno2**

[   11.953333]ixgbe 0000:3b:00.1 eno2: SR-IOV enabled with 4 VFs

[   12.541801] ixgbe 0000:3b:00.1: setting MAC 8e:df:08:c1:d1:de on VF 0

[   12.541805] ixgbe 0000:3b:00.1: Reload the VFdriver to make this change effective.

[   12.541841] ixgbe 0000:3b:00.1 eno2: VF 0 istrusted

[   12.541846] ixgbe 0000:3b:00.1: setting MAC 5a:b9:2f:99:a6:ce on VF 1

[   12.541850] ixgbe 0000:3b:00.1: Reload the VFdriver to make this change effective.

[   12.541883] ixgbe 0000:3b:00.1 eno2: VF 1 istrusted

[   12.541887] ixgbe 0000:3b:00.1: setting MAC46:78:69:e3:71:3d on VF 2

[   12.541891] ixgbe 0000:3b:00.1: Reload the VFdriver to make this change effective.

[   12.541923] ixgbe 0000:3b:00.1 eno2: VF 2 istrusted

[   12.541928] ixgbe 0000:3b:00.1: setting MAC7e:a7:db:3b:1b:b3 on VF 3

[   12.541932] ixgbe 0000:3b:00.1: Reload the VFdriver to make this change effective.

[   12.541965] ixgbe 0000:3b:00.1 eno2: VF 3 istrusted

2.4 检查VF

```
[root@centos7 ~]# lspci | grep -i Virtual[root@centos7 ~]# ip link show | grep -B2 vf
```

寻找Physical Function和Virtual Function之间的对应关系：

```
[root@centos7 ~]# ls -l /sys/class/net/eno1/device/ | grep virtfn
```

VF被创建后，NetworkManager自动给新的设备创建Connection，可以修改名称，如下：

**nmcli connection**

NAME        UUID                                  TYPE      DEVICE

eno1       2a1c8b39-7f44-321b-a65f-a93e70ab0616 ethernet  eno1

eno2       64ffa204-0158-40c8-ba86-2b7aebf27619 ethernet  eno2

enp59s16f1  19c28a93-aa36-38e6-a556-55a922a0a332  ethernet  enp59s16f1

enp59s16f3  428d9707-1515-3475-b356-7eb229c3f937  ethernet enp59s16f3

enp59s16f5  21a8dd5f-239f-37b2-9b09-24ce0e7413bc  ethernet enp59s16f5

enp59s16f7  0ca0da65-64c4-314d-89a4-4213f9e4f478  ethernet enp59s16f

修改名称：
**nmcli connection modify uuid 19c28a93-aa36-38e6-a556-55a922a0a332 connection.id enp59s16f1**
修改MTU值，并禁用IPv4和IPv6，网卡启动更快

**nmcli connection modify enp59s16f1 ifname enp59s16f1 ipv4.method disabled ipv6.method ignore ethernet.mtu 9216 ethernet.mac-address ""**

**nmcli connection up enp59s16f1**

**nmcli connection show enp59s16f**

上述命令生成的ifcfg配置文件如下：

```
[root@centos7 ~]# cat /etc/sysconfig/network-scripts/ifcfg-enp59s16f1TYPE=EthernetPROXY_METHOD=noneBROWSER_ONLY=noDEFROUTE=yesIPV4_FAILURE_FATAL=noIPV6INIT=noIPV6_AUTOCONF=noIPV6_DEFROUTE=yesIPV6_FAILURE_FATAL=noIPV6_ADDR_GEN_MODE=stable-privacyNAME=enp59s16f1UUID=19c28a93-aa36-38e6-a556-55a922a0a332ONBOOT=yesAUTOCONNECT_PRIORITY=-999DEVICE=enp59s16f1MTU=9216
```

这样，即便系统重启，上述配置依然能生效。

2.5 使用KVM的虚拟网络适配器池

主机上创建一个VF网络设备的资源池，资源池内的设备可以自动地分配给虚拟机使用。

### （1）创建一个xml文件。

[root@centos7 ~]# cat eno2_sriov_pool.xml

<network>

   <name>eno2_sriov_pool</name> <!-- This is thename of the file you created -->

   <forward mode='hostdev' managed='yes'>

     <pf dev='eno2'/> <!-- Use the netdev name of your SR-IOV devices PF here -->

   </forward>

</network>

 （2）根据xml定义一个网络，并设置为自动重启

```
virsh net-define eno2_sriov_pool.xmlvirsh net-start eno2_sriov_poolvirsh net-autostart eno2_sriov_pool
```

[root@centos7 ~]# virsh net-dumpxmleno2_sriov_pool

<networkconnections='1'>

  <name>eno2_sriov_pool</name>

 <uuid>e0842451-0137-4255-8783-305ca27f082d</uuid>

  <forward mode='hostdev' managed='yes'>

    <pf dev='eno2'/>

    <address type='pci' domain='0x0000'bus='0x3b' slot='0x10' function='0x1'/>

    <address type='pci' domain='0x0000'bus='0x3b' slot='0x10' function='0x3'/>

    <address type='pci' domain='0x0000'bus='0x3b' slot='0x10' function='0x5'/>

    <address type='pci' domain='0x0000'bus='0x3b' slot='0x10' function='0x7'/>

  </forward>

</network>

### （3）从网络适配器池中分配网卡给虚拟机

用这种方法添加SRIOV网卡比较简单：

![图片](assets/30e2596ba955.png)

按照如上方法添加网卡，等同于以下xml配置：

```
    <interface type='network'>      <mac address='52:54:00:2d:87:99'/>      <source network='eno2_sriov_pool'/>      <model type='virtio'/>      <address type='pci' domain='0x0000'bus='0x00' slot='0x04' function='0x0'/>    </interface>
```

开机后dumpxml如下：   

```
 <interface type='hostdev'managed='yes'>      <mac address='52:54:00:2d:87:99'/>      <driver name='vfio'/>      <source>        <address type='pci' domain='0x0000'bus='0x3b' slot='0x10' function='0x1'/>      </source>      <model type='virtio'/>      <alias name='hostdev0'/>      <address type='pci' domain='0x0000' bus='0x00' slot='0x0f'function='0x0'/></interface>
```

3.使用Virt-Manager安装CSR1000v

在CentOS 图形界面中，打开Terminal，运行virt-manager，按照以下步骤创建CSR1000v；添加网卡，并选择en2_sriov_pool。

![图片](assets/f4cc4826f457.png)

![图片](assets/afacc9c697e9.png)

![图片](assets/abda713a1799.png)

![图片](assets/362f7e1460c3.png)

注：csr1kv-1的第一个网口选择macvtap Bridge模式，这样就无需创建一个Linux网桥。但是，csr1kv-1启动以后不能通过该接口与Linux主机进行通信，仅能通过该接口访问Linux主机外的网络。

![图片](assets/1ae1dc911cb6.png)

![图片](assets/2861ac80e963.png)

添加完网卡后，点击开始安装，然后就可以关闭虚拟机了。上述操作完成后，virt-manager会在/etc/libvirtd/qemu/目录下创建csr1kv-1.xml。

4.KVM调优配置

KVM的调优比较复杂，主要是NUMA、内存大页、vCPU PIN等，参考资料为Redhat Linux 7 PERFORMANCE TUNING GUIDE。

**4.1检查平台的能力**

```
[root@centos7 ~]# virsh nodeinfoCPU model:           x86_64CPU(s):              72CPU frequency:       999 MHzCPU socket(s):       1Core(s) per socket:  18Thread(s) per core:  2NUMA cell(s):        2Memory size:         263665612 KiB[root@centos7 ~]# virsh capabilities<capabilities>  <host>    <uuid>4e53df1f-5b36-6842-99ee-1369d7c68730</uuid>    <cpu>      <arch>x86_64</arch>      <model>Skylake-Server-IBRS</model>      <vendor>Intel</vendor>      <microcode version='33581318'/>      <counter name='tsc' frequency='2294597000' scaling='yes'/>      <topology sockets='1' cores='18' threads='2'/>……
```

可检查平台的CPU核数、分布，内存的NUMA分布等。

4.2 NUMA调优

```
[root@centos7 ~]# numactl --hardwareavailable: 2 nodes (0-1)node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53node 0 size: 128491 MBnode 0 free: 112157 MBnode 1 cpus: 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71node 1 size: 128994 MBnode 1 free: 124120 MBnode distances:node   0   1  0:  10  21  1:  21  10
```

两颗CPU，每颗CPU各有128GB内存，分别是node 0 和 node 1。

4.3 内存大页HugePage以及透明大页

 cat /proc/meminfo | grep HugePages查看当前系统有多少个大页： 

```
[root@centos7 ~]# cat /proc/meminfo | grep HugeAnonHugePages:   1685504 kBHugePages_Total:      64HugePages_Free:       60HugePages_Rsvd:        0HugePages_Surp:        0Hugepagesize:    1048576 kB
```

在系统运行时修改大页数量：

```
[root@centos7 ~]# cat /sys/devices/system/node/node0/hugepages/hugepages-1048576kB/nr_hugepages16[root@centos7 ~]# echo 32 > /sys/devices/system/node/node0/hugepages/hugepages-1048576kB/nr_hugepages[root@centos7 ~]# echo 32 > /sys/devices/system/node/node1/hugepages/hugepages-1048576kB/nr_hugepages[root@centos7 ~]# numastat -cm | egrep 'Node|Huge'                 Node 0 Node 1  TotalAnonHugePages     10962   1528  12490HugePages_Total   32768  32768  65536HugePages_Free    32768  32768  65536HugePages_Surp        0      0      0
```

4.4 vCPU钉选

设置CPU Affinity的好处是提高CPU缓存效率，避免进程在多个CPU核之间跳跃，切换CPU核均会导致缓存中的数据无效，缓存命中率大幅降低，导致数据获取的开销居高不下，损失性能。

```
virsh vcpuinfo csr1kv-1 可以查看vCPU的分配。
```

4.5 编辑CSR 1000v的XML调优参数

virsh edit csr1kv-1 可以编辑XML的参数，如下：

注：<emulatorpin cpuset='37-44'/> 参数，仅当Hyper-Threating开启时使用；有一些平台并未关闭超线程，例如Cisco专门的NFV平台 CSP。通过virsh chapabilities 查看siblings='1,37'，当core 1设置为vcpupin时，core 37应设置到emulatorpin cpuset中。

以下关于CPU和内存的参数设定建议来自于

https://libvirt.org/formatdomain.html 

https://libvirt.org/kbase/kvm-realtime.html

<domaintype='kvm'>

  <name>csr1kv-1</name>

 <uuid>59581018-6387-49df-ab09-2bcf40fc12ba</uuid>

  <memoryunit='KiB'>8388608</memory>

  <currentMemory unit='KiB'>8388608</currentMemory>

  <memoryBacking>

   <hugepages>

      <pagesize='1048576' unit='KiB'/>

</hugepages>

<locked/>

   <nosharepages/>

 </memoryBacking>

  <vcpuplacement='static'>8</vcpu>

  <cputune>

    <vcpupinvcpu='0' cpuset='1'/>

    <vcpupinvcpu='1' cpuset='2'/>

    <vcpupinvcpu='2' cpuset='3'/>

    <vcpupinvcpu='3' cpuset='4'/>

    <vcpupinvcpu='4' cpuset='5'/>

    <vcpupinvcpu='5' cpuset='6'/>

    <vcpupinvcpu='6' cpuset='7'/>

    <vcpupinvcpu='7' cpuset='8'/>

    <emulatorpincpuset='37-44'/>

  </cputune>

  <numatune>

    <memorymode='strict' nodeset='0'/>

  </numatune>

  <os>

    <type arch='x86_64'machine='pc-i440fx-rhel7.0.0'>hvm</type>

    <boot dev='hd'/>

  </os>

  <features>

    <acpi/>

    <apic/>

  </features>

  <cpu mode='host-passthrough'check='none'/>

  <clock offset='utc'>

    <timer name='rtc'tickpolicy='catchup'/>

    <timer name='pit'tickpolicy='delay'/>

    <timer name='hpet' present='no'/>

  </clock>

 <on_poweroff>destroy</on_poweroff>

  <on_reboot>restart</on_reboot>

  <on_crash>destroy</on_crash>

  <pm>

    <suspend-to-mem enabled='no'/>

    <suspend-to-disk enabled='no'/>

  </pm>

  <devices>

   <emulator>/usr/libexec/qemu-kvm</emulator>

    <disk type='file' device='disk'>

      <driver name='qemu' type='qcow2'/>

      <sourcefile='/home/root/images/csr1000v-universalk9.17.02.01v.qcow2'/>

      <target dev='vda' bus='virtio'/>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x07' function='0x0'/>

    </disk>

    <controller type='usb' index='0'model='ich9-ehci1'>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x05' function='0x7'/>

    </controller>

    <controller type='usb' index='0'model='ich9-uhci1'>

      <master startport='0'/>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x05' function='0x0' multifunction='on'/>

    </controller>

    <controller type='usb' index='0'model='ich9-uhci2'>

      <master startport='2'/>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x05' function='0x1'/>

    </controller>

    <controller type='usb' index='0'model='ich9-uhci3'>

      <master startport='4'/>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x05' function='0x2'/>

    </controller>

    <controller type='pci' index='0'model='pci-root'/>

    <controller type='virtio-serial'index='0'>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x06' function='0x0'/>

    </controller>

    <interface type='direct'>

      <mac address='52:54:00:36:95:f0'/>

      <source dev='eno1' mode='bridge'/>

      <model type='virtio'/>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x03' function='0x0'/>

    </interface>

    <interface type='network'>

      <macaddress='52:54:00:2d:87:99'/>

      <sourcenetwork='eno2_sriov_pool'/>

      <modeltype='virtio'/>

      <addresstype='pci' domain='0x0000' bus='0x00' slot='0x04' function='0x0'/>

   </interface>

    <serial type='pty'>

      <target type='isa-serial' port='0'>

        <model name='isa-serial'/>

      </target>

    </serial>

    <console type='pty'>

      <target type='serial' port='0'/>

    </console>

    <channel type='unix'>

      <target type='virtio'name='org.qemu.guest_agent.0'/>

      <address type='virtio-serial'controller='0' bus='0' port='1'/>

    </channel>

    <channel type='spicevmc'>

      <target type='virtio'name='com.redhat.spice.0'/>

      <address type='virtio-serial'controller='0' bus='0' port='2'/>

    </channel>

    <input type='tablet' bus='usb'>

      <address type='usb' bus='0'port='1'/>

    </input>

    <input type='mouse' bus='ps2'/>

    <input type='keyboard' bus='ps2'/>

    <graphics type='spice'autoport='yes'>

      <listen type='address'/>

      <image compression='off'/>

    </graphics>

    <video>

      <model type='qxl' ram='65536'vram='65536' vgamem='16384' heads='1' primary='yes'/>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x02' function='0x0'/>

    </video>

    <redirdev bus='usb' type='spicevmc'>

      <address type='usb' bus='0'port='2'/>

    </redirdev>

    <redirdev bus='usb' type='spicevmc'>

      <address type='usb' bus='0'port='3'/>

    </redirdev>

    <memballoon model='none'/>

    <rng model='virtio'>

      <backendmodel='random'>/dev/urandom</backend>

      <address type='pci' domain='0x0000'bus='0x00' slot='0x09' function='0x0'/>

    </rng>

  </devices>

</domain>

**上述参数整理自《Cisco CSR 1000v and Cisco ISRv Software Configuration Guide》**

https://www.cisco.com/c/en/us/td/docs/routers/csr1000/software/configuration/b_CSR1000v_Configuration_Guide/b_CSR1000v_Configuration_Guide_chapter_0101.html

完成上述XML文件的编辑后，执行

```
cd /etc/libvirtd/qemuvirsh define csr1kv-1.xmlvirsh start csr1kv-1
```

4.6 在KVM主机上访问CSR1000v的console

在virt-manager创建CSR1000v虚拟机的时候，缺省会添加一个Serial Device。

```
[root@centos7 qemu]# virsh console 20Connected to domain CSR1000v-1Escape character is ^]
```

CSR 1000v 暂时不能通过Console配置，需要通过virt-manager的图形化界面进行初始化配置。vCloud 的Console访问正常。

4.7 检验CSR1000v的调优配置

```
[root@centos7 ~]# virsh listId    Name                           State---------------------------------------------------- 6     csr1kv-1                       running[root@centos7 ~]# virsh vcpuinfo 6VCPU:           0CPU:            1State:          runningCPU time:       34.5sCPU Affinity:   -y----------------------------------------------------------------------VCPU:           1CPU:            2State:          runningCPU time:       32.0sCPU Affinity:   --y---------------------------------------------------------------------VCPU:           2CPU:            3State:          runningCPU time:       23.8sCPU Affinity:   ---y--------------------------------------------------------------------VCPU:           3CPU:            4State:          runningCPU time:       19.8sCPU Affinity:   ----y-------------------------------------------------------------------VCPU:           4CPU:            5State:          runningCPU time:       23.0sCPU Affinity:   -----y------------------------------------------------------------------VCPU:           5CPU:            6State:          runningCPU time:       23.4sCPU Affinity:   ------y-----------------------------------------------------------------VCPU:           6CPU:            7State:          runningCPU time:       28.5sCPU Affinity:   -------y----------------------------------------------------------------VCPU:           7CPU:            8State:          runningCPU time:       28.2sCPU Affinity:   --------y---------------------------------------------------------------
```

以上显示CSR1000v虚拟机的CPU亲和性在1-8核上。

```
[root@centos7 ~]# numastat -c qemu-kvmPer-node process memory usage (in MBs) for PID 46895 (qemu-kvm)         Node 0 Node 1 Total         ------ ------ -----Huge       8192      0  8192Heap        118      0   118Stack         0      0     0Private     144      7   151-------  ------ ------ -----Total      8455      7  8461
```

以上显示，内存主要使用Node 0。

```
[root@centos7 ~]# numastat -vm -p 13428 | grep HugePageAnonHugePages             956.00          494.00         1450.00HugePages_Total         16384.00        16384.00        32768.00HugePages_Free           8192.00        16384.00        24576.00HugePages_Surp              0.00            0.00            0.00
```

4.8 在CSR1000v上检查虚拟网卡

```
csr1kv-1#show platform software vnic-if interface-mapping------------------------------------------------------------- Interface Name        Driver Name         Mac Addr------------------------------------------------------------- GigabitEthernet2       net_ixgbe_vf       5254.002d.8799 GigabitEthernet1       net_virtio         5254.0036.95f0
```

上述驱动名显示为net_ixgbe_vf表明该虚拟网卡是一个SR-IOV池中分配的VF设备。

4.9 Linux 上抓取虚拟网卡的报文（参考）

查找虚拟机的网卡列表

```
[root@centos7 ~]# virsh domiflist 10Interface  Type       Source     Model       MAC-------------------------------------------------------vnet0      network    default    virtio      52:54:00:38:71:4amacvtap0   direct     eno1       virtio      52:54:00:b2:70:90vnet5      bridge     net-br1    virtio      52:54:00:69:93:23
```

抓取vnet5的报文

```
[root@centos7 ~]# tcpdump -i vnet5 -w ping.pcaptcpdump: listening on vnet5, link-type EN10MB (Ethernet), capture size 262144 bytes^C49 packets captured51 packets received by filter0 packets dropped by kernel
```

注：VF如果被分配给虚拟机，那么在Linux主机里，通过ip link则查看不到该设备，无法通过上述办法抓包。

5.CSR1000v的初始化配置及Smart License注册

5.1 CSR1000v的初始配置

部分节略，其他为缺省配置。

CSR1000v-1#show sdwanrunning-config

system

 system-ip             1.1.10.1

 site-id               101

 sp-organization-name  CiscoBJ

 organization-name     CiscoBJ

 vbond 10.75.58.51port 12346

!

hostname CSR1000v-1

username admin privilege15 secret 9 <removed>

ip name-server64.104.123.245

ip route 0.0.0.0 0.0.0.010.75.59.1

 

interfaceGigabitEthernet1

 no shutdown

 arp timeout 1200

 ip address 10.75.59.35 255.255.255.0

 no ip redirects

 ip mtu   1500

 mtu 1500

 negotiation auto

exit

 

interface Tunnel1

 no shutdown

 ip unnumberedGigabitEthernet1

 no ip redirects

 ipv6 unnumberedGigabitEthernet1

 no ipv6 redirects

 tunnel sourceGigabitEthernet1

 tunnel mode sdwan

exit

 

clock timezone CST 8 0

ntp server 10.75.58.1version 4

sdwan

 interfaceGigabitEthernet1

  tunnel-interface

   encapsulationipsec

   allow-servicesshd

  exit

常用命令

show sdwan control local-properties

show sdwan control connections

show sdwan control connection-history

show sdwan running-config

show sdwan bfd sessions

show sdwan omp peers

show sdwan omp routes

5.2 CSR1000v Smart License注册

CSR 1000v默认限速为250Mbps，需要注册Smart License才可解开限速。

```
CSR1000v-2#show platform hardware throughput levelThe current throughput level is 250000 kb/s
```

注册Smart License需要满足以下条件：

1、 CSR 1000v已经注册到vManage，控制面连接正常；

2、 配置 ip http client source-interface GigabitEthernet2

3、 sdwan interface GigabitEthernet2 tunnel-interface allow-service all  ---针对16.12.x版本；在新版本中仅需要allow https

4、 CSR 1000v 可以访问URL：

https://tools.cisco.com/its/service/oddce/services/DDCEService

允许访问114.114.114.114，CSR 1000v 可解析域名tools.cisco.com

替代办法，增加一条命令 ip host tools.cisco.com 72.163.4.38

执行license smart register idtoken xxxxxx 进行注册

show license status 查看注册结果

```
注册完成后，系统解开限速：CSR1000v-1#show platform hardware throughput levelThe current throughput level is 200000000 kb/s
```

5.3 性能和相关限制

【SR-IOV性能】在上述CSR1KV-1安装好后，使用测试仪进行性能测试，测试条件中，设置丢包率为0%，以256字节为例，其性能如下：

Packet Site

SDWAN Performance（Mbps）

CEF Performance (Mbps)

128Byte

800

2843.75

256Byte

1431.26

6500.00

512Byte

2581.26

10578.13

1024Byte

3731.26

15500.00

1400Byte

4306.26

18171.88

![图片](assets/d12cc6ecd16d.png)

注：不同服务器和网卡可能测试结果有区别，上述性能数据仅供参考。
**SRIOV的主要限制是每一个VF设备支持的VLAN数，ixgbevf所支持的最大VLAN数为64；因此，在CSR1KV中对应的虚拟接口配置的活跃子接口数最大为64。**
配置指南中有关于SRIOV子接口限制的说明：

 Cisco CSR 1000v and Cisco ISRv Software Configuration Guide:

SR-IOV (ixgbevf)

Maximum VLANs: The maximum number of VLANs supported on PF is 64. Together, all VFs can have a total of 64 VLANs. (Intel limitation.)

SR-IOV (i40evf)

Maximum VLANs: The maximum number of VLANs supported on PF is 512. Together, all VFs can have a total of 512 VLANs. (Intel limitation.) Per-VF resources are managed by the PF (host) device driver.

附录

1. Virt-Manager设置虚机的CPU模式说明

Libvirt 主要支持三种 CPU mode：

host-passthrough: libvirt 令 KVM 把宿主机的 CPU 指令集全部透传给虚拟机。因此虚拟机能够最大限度的使用宿主机 CPU 指令集，故性能是最好的。但是在热迁移时，它要求目的节点的 CPU 和源节点的一致。

host-model: libvirt 根据当前宿主机 CPU 指令集从配置文件/usr/share/libvirt/cpu_map.xml 选择一种最相配的 CPU 型号。在这种 mode 下，虚拟机的指令集往往比宿主机少，性能相对 host-passthrough 要差一点，但是热迁移时，它允许目的节点 CPU 和源节点的存在一定的差异。

custom: 这种模式下虚拟机 CPU 指令集数最少，故性能相对最差，但是它在热迁移时跨不同型号 CPU 的能力最强。此外，custom 模式下支持用户添加额外的指令集。

三种mode的性能排序是：host-passthrough > host-model >custom

实际性能差异不大：100%>95.84%>94.73%

引自：http://wsfdl.com/openstack/2018/01/02/libvirt_cpu_mode.html

2.有关网卡模式的说明

使用virt-manager创建虚拟机，在添加网卡时，有3中选择，分别是e1000, rtl8139, virtio。

“rtl8139”这个网卡模式是qemu-kvm默认的模拟网卡类型，RTL8139是Realtek半导体公司的一个10/100M网卡系列，是曾经非常流行（当然现在看来有点古老）且兼容性好的网卡，几乎所有的现代操作系统都对RTL8139网卡驱动的提供支持。

“e1000”系列提供Intel e1000系列的网卡模拟，纯的QEMU（非qemu-kvm）默认就是提供Intele1000系列的虚拟网卡。

“virtio” 类型是qemu-kvm对半虚拟化IO（virtio）驱动的支持。

这三个网卡的最大区别(此处指最需要关注的地方)是速度：

rtl8139 10/100Mb/s

e1000 1Gb/s

virtio 10Gb/s

注意virtio是唯一可以达到10Gb/s的。

virtio 是一种 I/O 半虚拟化解决方案，是一套通用I/O 设备虚拟化的程序，是对半虚拟化Hypervisor 中的一组通用 I/O 设备的抽象。提供了一套上层应用与各 Hypervisor 虚拟化设备（KVM，Xen，VMware等）之间的通信框架和编程接口，减少跨平台所带来的兼容性问题，大大提高驱动程序开发效率。

 

3.有关MACVTAP

以下内容来自：

https://www.ibm.com/developerworks/cn/linux/1312_xiawc_linuxvirtnet/index.html

MACVTAP 的实现基于传统的 MACVLAN。和 TAP 设备一样，每一个 MACVTAP 设备拥有一个对应的 Linux 字符设备，并拥有和 TAP 设备一样的 IOCTL 接口，因此能直接被 KVM/Qemu 使用，方便地完成网络数据交换工作。引入 MACVTAP 设备的目标是：简化虚拟化环境中的交换网络，代替传统的 Linux TAP 设备加 Bridge 设备组合，同时支持新的虚拟化网络技术，如 802.1 Qbg。

MACVTAP 设备和 VLAN 设备类似，是以一对多的母子关系出现的。在一个母设备上可以创建多个 MACVTAP 子设备，一个 MACVTAP 设备只有一个母设备，MACVTAP 子设备可以做为母设备，再一次嵌套的创建 MACVTAP 子设备。母子设备之间被隐含的桥接起来，母设备相当于现实世界中的交换机 TRUNK 口。实际上当 MACVTAP 设备被创建并且模式不为 Passthrough 时，内核隐含的创建了 MACVLAN 网络，完成转发功能。MACVTAP 设备有四种工作模式：Bridge、VEPA、Private，Passthrough。

Bridge 模式下，它完成与 Bridge 设备类似功能，数据可以在属于同一个母设备的子设备间交换转发，虚拟机相当于简单接入了一个交换机。当前的 Linux 实现有一个缺陷，此模式下 MACVTAP 子设备无法和 Linux Host 通讯，即虚拟机无法和 Host 通讯。----经验证，属实。

Passthrough 模式下，内核的 MACVLAN 数据处理逻辑被跳过，硬件决定数据如何处理，从而释放了 Host CPU 资源。

![图片](assets/213679bdf092.png)

MACVTAP Passthrough 概念与 PCI Passthrough 概念不同，上图详细解释了两种情况的区别。

PCI Passthrough 针对的是任意 PCI 设备，不一定是网络设备，目的是让 Guest OS 直接使用 Host 上的 PCI 硬件以提高效率。以 X86 平台为例，数据将通过需要硬件支持的 VT-D 技术从 Guest OS 直接传递到 Host 硬件上。这样做固然效率很高，但因为模拟器失去了对虚拟硬件的控制，难以同步不同 Host 上的硬件状态，因此当前在使用 PCI Passthrough 的情况下难以做动态迁移。

MACVTAP Passthrough 仅仅针对 MACVTAP 网络设备，目的是绕过内核里MACVTAP 的部分软件处理过程，转而交给硬件处理。在虚拟化条件下，数据还是会先到达模拟器 I/O 层，再转发到硬件上。这样做效率有损失，但模拟器仍然控制虚拟硬件的状态及数据的走向，可以做动态迁移。

4.SR-IOV简介

如果网卡支持SRIOV，请使用SRIOV PCI Passthrough。

![图片](assets/859075ce7f2f.png)

软件模拟是通过Hypervisor层模拟虚拟网卡，实现与物理设备完全一样的接口，虚拟机操作系统无须修改就能直接驱动虚拟网卡，其最大的缺点是性能相对较差；

网卡直通支持虚拟机绕过Hypervisor层，直接访问物理I/O设备，具有最高的性能，但是在同一时刻物理I/O设备只能被一个虚拟机独享；

SR-IOV是Intel在2007年提出的解决虚拟化网络I/O的硬件技术方案，该技术不仅能够继承网卡直通的高性能优势，而且同时支持物理I/O设备的跨虚拟机共享，具有较好的应用前景。

原文链接：https://blog.csdn.net/lsz137105/article/details/100752930

SR-IOV（Single Root I/O Virtualization）是一个将PCIe设备（如网卡）共享给虚拟机的标准，通过为虚拟机提供独立的内存空间、中断、DMA流，来绕过VMM实现数据访问。

SR-IOV引入了两种PCIe functions：

PF（Physical Function）：包含完整的PCIe功能，包括SR-IOV的扩张能力，该功能用于SR-IOV的配置和管理。

VF（Virtual Function）：包含轻量级的PCIe功能。每一个VF有它自己独享的PCI配置区域，并且可能与其他VF共享着同一个物理资源。

SR-IOV网卡通过将SR-IOV功能集成到物理网卡上，将单一的物理网卡虚拟成多个VF接口，每个VF接口都有单独的虚拟PCIe通道，这些虚拟的PCIe通道共用物理网卡的PCIe通道。每个虚拟机可占用一个或多个VF接口，这样虚拟机就可以直接访问自己的VF接口，而不需要Hypervisor的协调干预，从而大幅提升网络吞吐性能。

5.探索虚拟机进程

每一个客户机就是宿主机中的一个QEMU进程，而一个客户机的多个vCPU就是一个QEMU进程中的多个线程。

```
[root@centos7 ~]# ps -ef | grep qemu
```

qemu      52595     1 99 10:37 ?        01:24:12/usr/libexec/qemu-kvm -name csr1kv-1 -S -machine pc-i440fx-rhel7.0.0,accel=kvm,usb=off,dump-guest-core=off,mem-merge=off -cpu host-m 8192 -mem-prealloc-mem-path /dev/hugepages/libvirt/qemu/7-csr1kv-1 -realtime mlock=on -smp8,sockets=8,cores=1,threads=1 -uuid 59581018-6387-49df-ab09-2bcf40fc12ba-no-user-config -nodefaults -chardevsocket,id=charmonitor,path=/var/lib/libvirt/qemu/domain-7-csr1kv-1/monitor.sock,server,nowait-mon chardev=charmonitor,id=monitor,mode=control -rtc base=utc,driftfix=slew-global kvm-pit.lost_tick_policy=delay -no-hpet -no-shutdown -globalPIIX4_PM.disable_s3=1 -global PIIX4_PM.disable_s4=1 -boot strict=on -devicepiix3-usb-uhci,id=usb,bus=pci.0,addr=0x1.0x2 -devicevirtio-serial-pci,id=virtio-serial0,bus=pci.0,addr=0x6 -drivefile=/home/root/images/csr1000v-universalk9.17.02.01v.qcow2,format=qcow2,if=none,id=drive-virtio-disk0-device virtio-blk-pci,scsi=off,bus=pci.0,addr=0x7,drive=drive-virtio-disk0,id=virtio-disk0,bootindex=1-netdev tap,fd=26,id=hostnet0,vhost=on,vhostfd=28 -device virtio-net-pci,netdev=hostnet0,id=net0,mac=52:54:00:36:95:f0,bus=pci.0,addr=0x3-chardev pty,id=charserial0 -device isa-serial,chardev=charserial0,id=serial0-chardevsocket,id=charchannel0,path=/var/lib/libvirt/qemu/channel/target/domain-7-csr1kv-1/org.qemu.guest_agent.0,server,nowait-devicevirtserialport,bus=virtio-serial0.0,nr=1,chardev=charchannel0,id=channel0,name=org.qemu.guest_agent.0-chardev spicevmc,id=charchannel1,name=vdagent -devicevirtserialport,bus=virtio-serial0.0,nr=2,chardev=charchannel1,id=channel1,name=com.redhat.spice.0-spice port=5900,addr=127.0.0.1,disable-ticketing,image-compression=off,seamless-migration=on-vga qxl -global qxl-vga.ram_size=67108864 -global qxl-vga.vram_size=67108864-global qxl-vga.vgamem_mb=16 -global qxl-vga.max_outputs=1 -devicevfio-pci,host=1d:00.0,id=hostdev1,bus=pci.0,addr=0x8 -devicevfio-pci,host=3b:10.1,id=hostdev0,bus=pci.0,addr=0x4 -msg timestamp=on

使用virsh命令或virt-manager开启虚拟机，是通过调用/usr/libexec/qemu-kvm 并附带虚拟配置的参数，来开启qemu-kvm的进程。可以看到上述的参数是非常复杂的，libvirt提供XML参数进行简化。

**ps -efL | grep qemu **可以列出所有的线程，但是输出篇幅很长，不在此列出；使用pstree 可列出其父进程、线程关系，如下：

![图片](assets/e02c29f89589.png)

virt-top 可查看虚机运行状态和资源利用率：

[root@centos7 ~]# virt-top -1 

![图片](assets/ef717388c6eb.png)