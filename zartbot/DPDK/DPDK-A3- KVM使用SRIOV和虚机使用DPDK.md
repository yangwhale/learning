# DPDK-A3: KVM使用SRIOV和虚机使用DPDK

> 作者: zartbot  
> 日期: 2021年5月24日 13:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485868&idx=2&sn=4ee77ea694e2eaa9d13878eaf173c8de&chksm=f996196ecee19078dd4f768ba314bf22b5e6280ee4aa2e9f37d0f601aed98ea9c9e94a911eea#rd

---

### 虚拟机基本管理 

如下命令可以修改默认网段

```
sudo virsh net-edit --network default
```

```
<network>  <name>default</name>  <uuid>45ed012c-3933-4f3e-9575-b37bffa21b83</uuid>  <forward mode='nat'/>  <bridge name='virbr0' stp='on' delay='0'/>  <mac address='52:54:00:03:a7:5b'/>  <ip address='192.168.122.1' netmask='255.255.255.0'>    <dhcp>      <range start='192.168.122.2' end='192.168.122.254'/>    </dhcp>  </ip></network>
```

Enable forwarding

```
echo "net.ipv4.ip_forward=1" | sudo tee /etc/sysctl.conf
```

```
[root@netdev vm]# virsh list --all Id   Name     State------------------------ 1    ubuntu   running
```

#### 启动、关闭、删除

```
virsh [start/shutdown/destroy] vm
```

#### 挂起和恢复

```
virsh [suspend|resume] vm
```

#### 开机自启动

```
virsh autostart vm
```

查看信息

```
[root@netdev vm]# virsh dominfo ubuntuId:             1Name:           ubuntuUUID:           dd1e0fee-0f38-4a9a-a515-256bb6a10d16OS Type:        hvmState:          runningCPU(s):         4CPU time:       539.3sMax memory:     8388608 KiBUsed memory:    8388608 KiBPersistent:     yesAutostart:      disableManaged save:   noSecurity model: selinuxSecurity DOI:   0Security label: system_u:system_r:svirt_t:s0:c887,c939 (enforcing)
```

#### 克隆虚机

```
[root@netdev vm]# virt-clone -o ubuntu -n vm001 -f /home/vm/vm001.qcow2Allocating 'vm001.qcow2'                                                                                                             | 100 GB  00:00:08Clone 'vm001' created successfully.
```

克隆完成后， 需要uuidgen

```
[root@netdev vm]# uuidgen10c35319-bd71-4447-aa1a-88207ec42fbf[root@netdev vm]# vim /etc/libvirt/qemu/vm001.xml
```

#### 删除虚机

```
virsh undefine vm
```

删除虚机并删除存储

```
virsh undefine vm --storage /home/vm/vm.qcow2virsh undefine vm --remove-all-storage
```

#### dump xml

```
[root@netdev vm]# virsh dumpxml ubuntu
```

使用root登录vnc，然后启用virt-manager

```
virt-manager
```

![图片](assets/74a281ddf6ea.png)

image

然后创建虚机过程就不多讲了，添加PCIe Device将Mellanox**的VF网卡加入即可.

![图片](assets/f882fa25dc88.png)

image
安装Ubuntu虚机

安装ubuntu 20.04.2，然后安装的包如下,基本的编译环境，外加了我比较喜欢用terminator和tmux

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

### MLX5安装 

```
tar vzxf MLNX_OFED_LINUX-5.3-1.0.0.1-ubuntu20.04-x86_64.tgzcd MLNX_OFED_LINUX-5.3-1.0.0.1-ubuntu20.04-x86_64/sudo ./mlnxofedinstall --upstream-libs  --dpdk --add-kernel-support sudo /etc/init.d/openibd restart
```

### 编译dpdk 

下载dpdk并解压

```
wget http://fast.dpdk.org/rel/dpdk-21.05.tar.xztar xf dpdk-21.05.tar.xzcd dpdk-21.05
```

安装meson和pyelftools

```
sudo apt install  mesonsudo apt install python3-pyelftools
```

编译和安装

```
meson -Dexamples=all buildcd buildninjasudo ninja installsudo ldconfig
```

在VM中测试前文所述发包程序， 注意到这个错误提示，性能不好，只有9Mpps，主要是NUMA**的问题，需要配置CPU亲和性， 另外注意需要完全的修改源代码中的源目的MAC，否则在VF下基本无法转发.

```
EAL:   Invalid NUMA socket, default to 0zartbot@zartbot-KVM:~/test$ sudo ./build/helloworldEAL: Detected 4 lcore(s)EAL: Detected 1 NUMA nodesEAL: Detected shared linkage of DPDKEAL: Multi-process socket /var/run/dpdk/rte/mp_socketEAL: Selected IOVA mode 'PA'EAL: No available 1048576 kB hugepages reportedEAL: VFIO support initializedEAL:   Invalid NUMA socket, default to 0EAL: Probe PCI driver: net_virtio (1af4:1041) device: 0000:01:00.0 (socket 0)eth_virtio_pci_init(): Failed to init PCI deviceEAL: Requested device 0000:01:00.0 cannot be usedEAL:   Invalid NUMA socket, default to 0EAL: Probe PCI driver: mlx5_pci (15b3:101a) device: 0000:06:00.0 (socket 0)EAL:   Invalid NUMA socket, default to 0EAL: Probe PCI driver: mlx5_pci (15b3:101a) device: 0000:07:00.0 (socket 0)TELEMETRY: No legacy callbacks, legacy socket not created*****************************************number of available port: 2initializing port 0...port[0] support RX cheksum offload.port[0] support TX mbuf fast free offload.port[0] support TX IPv4 checksum offload.port[0] support TX UDP checksum offload.port[0] support TX TCP checksum offload.Port[0] MAC: 7a:99:ed:5f:e3:a6initializing port 1...port[1] support RX cheksum offload.port[1] support TX mbuf fast free offload.port[1] support TX IPv4 checksum offload.port[1] support TX UDP checksum offload.port[1] support TX TCP checksum offload.Port[1] MAC: 5a:d8:51:db:17:2dPPS: 2081708PPS: 23912180PPS: 18424908PPS: 8359624PPS: 9097548PPS: 18615392PPS: 8275396PPS: 8288376
```

检查Numa

```
[zartbot@netdev ~]$  sudo numactl --hardwareavailable: 2 nodes (0-1)node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71node 0 size: 176933 MBnode 0 free: 174714 MBnode 1 cpus: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95node 1 size: 182932 MBnode 1 free: 152286 MBnode distances:node   0   1  0:  10  21  1:  21  10
```

检查VCPU分配

```
[zartbot@netdev ~]$ sudo virsh vcpuinfo vm001VCPU:           0CPU:            75State:          runningCPU time:       806.6sCPU Affinity:   yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyVCPU:           1CPU:            41State:          runningCPU time:       792.8sCPU Affinity:   yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyVCPU:           2CPU:            39State:          runningCPU time:       689.6sCPU Affinity:   yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyVCPU:           3CPU:            77State:          runningCPU time:       685.5sCPU Affinity:   yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
```

以下关于CPU和内存的参数设定建议来自于

```
https://libvirt.org/formatdomain.html https://libvirt.org/kbase/kvm-realtime.html
```

注意修改

```
virsh edit vm001
```

```
<domain type='kvm'>  <name>vm001</name>  <uuid>a6125142-bc3b-4453-a010-2c03077e7e09</uuid>  <metadata>    <libosinfo:libosinfo xmlns:libosinfo="http://libosinfo.org/xmlns/libvirt/domain/1.0">      <libosinfo:os id="http://ubuntu.com/ubuntu/20.04"/>    </libosinfo:libosinfo>  </metadata>  <memory unit='KiB'>8388608</memory>  <currentMemory unit='KiB'>8388608</currentMemory>  <vcpu placement='static'>8</vcpu>  <cputune>    <vcpupin vcpu='0' cpuset='1'/>    <vcpupin vcpu='1' cpuset='2'/>    <vcpupin vcpu='2' cpuset='3'/>    <vcpupin vcpu='3' cpuset='4'/>    <vcpupin vcpu='4' cpuset='5'/>    <vcpupin vcpu='5' cpuset='6'/>    <vcpupin vcpu='6' cpuset='7'/>    <vcpupin vcpu='7' cpuset='8'/>    <emulatorpin cpuset='37-44'/>  </cputune>  <numatune>    <memory mode='strict' nodeset='0'/>  </numatune>  <os>    <type arch='x86_64' machine='pc-q35-rhel8.2.0'>hvm</type>    <boot dev='hd'/>  </os>  <features>    <acpi/>    <apic/>    <vmport state='off'/>  </features>  <cpu mode='host-passthrough' check='none'/>
```

然后性能就正常了

```
zartbot@zartbot-KVM:~/test$ sudo ./build/helloworldEAL: Detected 8 lcore(s)EAL: Detected 1 NUMA nodesEAL: Detected shared linkage of DPDKEAL: Multi-process socket /var/run/dpdk/rte/mp_socketEAL: Selected IOVA mode 'PA'EAL: No available 1048576 kB hugepages reportedEAL: VFIO support initializedEAL:   Invalid NUMA socket, default to 0EAL: Probe PCI driver: net_virtio (1af4:1041) device: 0000:01:00.0 (socket 0)eth_virtio_pci_init(): Failed to init PCI deviceEAL: Requested device 0000:01:00.0 cannot be usedEAL:   Invalid NUMA socket, default to 0EAL: Probe PCI driver: mlx5_pci (15b3:101a) device: 0000:06:00.0 (socket 0)EAL:   Invalid NUMA socket, default to 0EAL: Probe PCI driver: mlx5_pci (15b3:101a) device: 0000:07:00.0 (socket 0)TELEMETRY: No legacy callbacks, legacy socket not created*****************************************number of available port: 2initializing port 0...port[0] support RX cheksum offload.port[0] support TX mbuf fast free offload.port[0] support TX IPv4 checksum offload.port[0] support TX UDP checksum offload.port[0] support TX TCP checksum offload.Port[0] MAC: 7a:99:ed:5f:e3:a6initializing port 1...port[1] support RX cheksum offload.port[1] support TX mbuf fast free offload.port[1] support TX IPv4 checksum offload.port[1] support TX UDP checksum offload.port[1] support TX TCP checksum offload.Port[1] MAC: 5a:d8:51:db:17:2dPPS: 30PPS: 29593410PPS: 29587011PPS: 29777849PPS: 29716992PPS: 29693889PPS: 29798295
```