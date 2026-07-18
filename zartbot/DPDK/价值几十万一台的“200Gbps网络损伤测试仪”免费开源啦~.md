# 价值几十万一台的“200Gbps网络损伤测试仪”免费开源啦~

> 作者: zartbot  
> 日期: 2021年11月12日 13:45  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486954&idx=1&sn=47ff88a1f74c9d003214ca48a1814b78&chksm=f9961d28cee1943e6d2b3355d4bced0c03c7e40a68d1893844356db9878ed569e17fd1140971#rd

---

最近需要研究一些可靠传输算法并模拟网络中的延迟和丢包，同事需要测量某个产品的遥测数据精度， 和某运营商研究院准备做一些关于应用SLA**评分的模型，特别是需要5G网络中eCPRI下的延迟仿真，另外和某视频大厂的同学们一起做一些元宇宙数据中心视频质量测试的场景也需要它。当然很多互联网大厂的同学们都需要用到这个东西来做混沌测试，但是商用平台通常数十万一台，因此基于DPDK**自己撸了一个给小伙伴使用，把这个项目命名为zMonkey.

项目地址

**https://github.com/zartbot/zmonkey**

### 网络损伤仪简介

其实这是一个非常小众的领域，早期大多数只有网络设备厂商和一些应用厂商用来模拟广域网、卫星通信、3G、DSL**等延迟而使用的，有一些商业产品，但是大家更多的是选择Linux建立一个bridge后，利用tc中的netem实现,但是这种实现基本上只能做到1Gbps.

```
# eth0 网卡延迟增加100ms tc qdisc add dev eth0 root netem delay 100ms# 报文延迟的时间在 100ms ± 20ms 之间（90ms - 110ms）tc qdisc add dev eth0 root netem delay 100ms 20ms# 发送的报文有 50% 的丢包率tc qdisc change dev eth0 root netem loss 50%# 发送的报文有 0.3% ~ 25% 的丢包率tc qdisc change dev eth0 root netem loss 0.3% 25%
```

然后伴随着高速率25Gbps、100Gbps等场景，开源生态基本无能为了，而很多互联网厂商也开始选择国外的Spirent Attero-100G、XENA Chimera-100G，或者国内Holowan的产品...很多都基于硬件FPGA的方案

![图片](assets/ef11128848ea.png)

但是像我这种穷人，软件就能做的，当然是自己动手丰衣足食了呀。而这里面最关键的还是算法，后面讲架构的时候会说:

### zMonkey使用

如果需要测试100Gbps 100ms以上延迟，那么您必须要重新编译您的DPDK，

[dpdk编译安装连接](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=1876465727990202369#wechat_redirect)

在编译前修改`dpdk-21.08/config/rte_config.h` 把mempool可分配的内存空间加大.

```
/* EAL defines */#define RTE_MAX_HEAPS 32#define RTE_MAX_MEMSEG_LISTS 512 //128#define RTE_MAX_MEMSEG_PER_LIST 32768  //8192#define RTE_MAX_MEM_MB_PER_LIST 131072 //32768#define RTE_MAX_MEMSEG_PER_TYPE 131072 //32768#define RTE_MAX_MEM_MB_PER_TYPE 262144 //65536
```

然后再 meson build、ninja install.

我在测试过程中使用64GB hugepage

```
dpdk-hugepage.py --setup 64G
```

下载zmonkey并编译:

```
git clone https://github.com/zartbot/zmonkeycd zmonkeymake
```

然后可执行文件就放置在`/build`目录内,使用指南如下：

```
sudo ./build/zmonkey -- -hzmonkey [EAL options] -- <Parameters> -f --first_lcore         First lcore used for forwarding thread -n --core_num            Number of lcore used for forwarding -m --mbuf_size           Number of elements in mbuf -r --ring_size           Number of elements in mbuf -c --control_port        Remote control udp port(default:6666)zMonkey chaos config -d --l2r_latency         Left  -> Right Delay time [us] -D --r2l_latency         Right -> Left Delay time [us] -j --l2r_jitter          Left  -> Right Jitter time [us] -J --r2l_jitter          Right -> Left Jitter time [us] -l --l2r_loss            Left  -> Right Loss rate [%%] -L --r2l_loss            Right -> Left Loss rate [%%] -u --l2r_dup             Left  -> Right Duplicate rate [%%] -U --r2l_dup             Right -> Left Duplicate rate [%%]
```

例如您需要16个核来处理100Gbps、100ms延迟仿真和12.34%丢包，可以按照如下方式输入参数

```
sudo ./zmonkey -- --first_lcore 24 --core_num 8  --mbuf_size 2097152 --l2r_latency 100000 --l2r_loss 1234
```

### 性能测试

100G小包双向，可以做到90Mpps，这个性能还是基于Intel Xeon 8259CL只有PCIe3.0的情况实现的，在PCIe4.0的情况下满足真实业务200Gbps的需求毫无问题。 

![图片](assets/0e8671119f9c.png)

当然针对核较少的同学，例如我们只使用4个核，也能做到50Mpps，随便完成100Gbps测试

![图片](assets/d8b8230070db.png)

### 精度测试

延迟误差
我们通过连接一对10Gbps IXIA测量仿真延迟和真实延迟，当我们把仿真延迟设置到123456 us时

![图片](assets/ed2c57a5f9b5.png)

IXIA**测试结果如下：

![图片](assets/5a9a07731274.png)

抖动

延迟设置为876，546 us，抖动100，006 us

![图片](assets/bb61d3f393ae.png)

IXIA测试结果如下，均满足测试要求

![图片](assets/85a4021bb6c0.png)

丢包率
设置丢包率为1.23%

![图片](assets/4dcce87170f0.png)

实测IXIA丢包率：

![图片](assets/f56650fe876d.png)

### 远程控制

这也是小伙伴给我的一个需求，通常需要在网络中多点注入损伤，需要一个集中的控制平台，于是简单的实现了一个UDP socket并附送了一个网络工程师喜欢用的python的example，就一个非常简单的udp发送字符串的程序，字符串为<指令>,<方向>,<值> 的组合。

```
import socketimport randomimport timeudpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)monkeyHost = ('127.0.0.1',6666)#  instr#  1: config latency#  2: config jitter in micro seconds#  3: config packet drop rate in %%(1/10000)#  4: config packet duplication rate in %%(1/10000)# #  direction#  0: left-->right#  1: right-->left# #  value uint64_tinstr = 3direction = 0value = 1234sendData = str(instr) +"," + str(direction) +","+ str(value) udpSocket.sendto(sendData.encode(),monkeyHost)
```

### zMonkey架构

其实就是利用网卡RSS分散到多核处理，这样每个核处理10Mpps就够了，然后在实现的过程中做了几个很简单的骚操作
1. 丢包随机产生
很多人针对这种有上界的问题都是rand()% drop_rate 取模运算,考虑到精度为1/10000，我就用了8192作为分母，然后一个简单的& 就搞定了，然后8192/10000补偿到drop_rate上，这样降低了大量的运算

```
static bool event_flag(uint64_t rate){    return (rte_rand() & 8191) < rate;}
```

而真正的DROP也不需要看到某个event就去free packet，而是简单记个数，然后enqueue到延迟队列时少enqueue几个就好了。
2.时间戳
我把它打在了mbuf的private_data段，为了减少内存的读写，我只是每次batch 32个包时打在第一个包上，简单的将这32个包认为是相同的时间戳，然后延迟队列是一个FIFO，那么出队列的时候只对有时间戳的包进行block截留判断延迟。
3.Prefetch&Branch predition
Egress** 处理队列时的prefetch和分支预测也是常见的优化手段。

结论:觉得喜欢给渣github点个赞~，其实很多东西吧，软件认真写真的能做到很高性能的，有些场合真的不太需要硬件。