# DPDK-A2:MLX5启用SRIOV

> 作者: zartbot  
> 日期: 2021年5月24日 13:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485868&idx=1&sn=a2aeb42ae18c5d043c87f1cde4b99f41&chksm=f996196ecee190785ec77f5a0071833e1e3b11c5e57843f136a689e9b692347a012c1c29d26d#rd

---

❝
所有示例，都放置在了`github.com/zartbot/learn_dpdk`
❞
### 配置SRIOV 

改到`root`登录

```
su root[root@netdev test]# mst startStarting MST (Mellanox Software Tools) driver setLoading MST PCI module - SuccessLoading MST PCI configuration module - SuccessCreate devicesUnloading MST PCI module (unused) - Success
```

查看device-id

```
[root@netdev test]# mst statusMST modules:------------    MST PCI module is not loaded    MST PCI configuration module loadedMST devices:------------/dev/mst/mt4121_pciconf0         - PCI configuration cycles access.                                   domain:bus:dev.fn=0000:86:00.0 addr.reg=88 data.reg=92 cr_bar.gw_offset=-1                                   Chip revision is: 00
```

此步主要是获取`/dev/mst/mt4121_pciconf0`

然后检查相关的配置

```
[root@netdev test]#  mlxconfig -d /dev/mst/mt4121_pciconf0 q | egrep -e  "SRIOV|NUM_OF_VF|UCTX"         NUM_OF_VFS                          4         SRIOV_EN                            True(1)         UCTX_EN                             False(0)         SRIOV_IB_ROUTING_MODE_P1            LID(1)         SRIOV_IB_ROUTING_MODE_P2            LID(1)
```

需要将NUM_OF_VFS设置为4个、然后Enable `SRIOV_EN`和`UCTX_EN`

```
[root@netdev test]# mlxconfig -d /dev/mst/mt4121_pciconf0 set SRIOV_EN=1 NUM_OF_VFS=4 UCTX_EN=1Device #1:----------Device type:    ConnectX5Name:           MCX516A-CDA_Ax_BxDescription:    ConnectX-5 Ex EN network interface card; 100GbE dual-port QSFP28; PCIe4.0 x16; tall bracket; ROHS R6Device:         /dev/mst/mt4121_pciconf0Configurations:                              Next Boot       New         SRIOV_EN                            True(1)         True(1)         NUM_OF_VFS                          4               4         UCTX_EN                             False(0)        True(1) Apply new Configuration? (y/n) [n] : yApplying... Done!-I- Please reboot machine to load new configurations.
```

配置完成后重启整机

```
root
```

#### 创建VF

查看接口

```
[zartbot@netdev ~]$ sudo  ibdev2netdevmlx5_0 port 1 ==> ens17f0 (Up)mlx5_1 port 1 ==> ens17f1 (Up)
```

查看最多能够支持的VF数量

```
[zartbot@netdev ~]$  cat /sys/class/net/ens17f0/device/sriov_totalvfs4[zartbot@netdev ~]$  cat /sys/class/net/ens17f1/device/sriov_totalvfs4
```

启用VF

```
echo 4 | sudo tee  /sys/class/net/ens17f1/device/sriov_numvfsecho 4 | sudo tee /sys/class/net/ens17f0/device/sriov_numvfs
```

然后lspci就可以看到网卡了

```
[zartbot@netdev ~]$ lspci | grep Mellanox86:00.0 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex]86:00.1 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex]86:00.2 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex Virtual Function]86:00.3 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex Virtual Function]86:00.4 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex Virtual Function]86:00.5 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex Virtual Function]86:00.6 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex Virtual Function]86:00.7 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex Virtual Function]86:01.0 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex Virtual Function]86:01.1 Ethernet controller: Mellanox Technologies MT28800 Family [ConnectX-5 Ex Virtual Function]
```