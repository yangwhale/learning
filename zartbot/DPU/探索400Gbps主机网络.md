# 探索400Gbps主机网络

> 作者: zartbot  
> 日期: 2021年11月16日 02:15  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487010&idx=1&sn=81b9d199299b1f98ec4934ef98879da0&chksm=f9961ee0cee197f6aefddb5d9da54b1367be8423984f3c267f613e38c8181a048d7c98dc9b86#rd

---

阿里和腾讯都跳过了50G直接上到了100G的云服务器，但是未来元宇宙一类的视频和大规模VR/AR计算似乎会将主机网络很快的推向200GE和400GE,例如Meta(Facebook)最近就开始使用思科Silicon One提供200GE的ToR交换机。

因此，我们做了如下测试来分析PCIe网卡DMA时的内存使用率及未来200Gbps~400Gbps主机网络对内存、Cache的影响，发现有些问题并不是单纯的Offload能解决的，需要另辟蹊径。
测试环境
选了一台双路Intel CascadeLake的服务器
名称Intel Xeon Platinum 8259CL @ 2.50GHz插槽类型LGA3647核心数24线程数48主频2.5 GHz睿频3.5 GHzUPI3x10.4GTs内存6x DDR4-2666发布时间Q1 2020
内存虽然是支持2933的，但是实际频率受CPU的限制降到了DDR4-2666

```
sudo dmidecode -t memoryHandle 0x0011, DMI type 17, 40 bytesMemory Device        Array Handle: 0x0010        Type: DDR4        Type Detail: Synchronous        Speed: 2933 MT/s        Manufacturer: Hynix        Serial Number: *********        Asset Tag: CPU1_DIMM_A1_AssetTag        Part Number: ************        Rank: 2>>>>    Configured Memory Speed: 2666 MT/s  <<<<<        Minimum Voltage: 1.2 V        Maximum Voltage: 1.2 V        Configured Voltage: 1.2 V
```

按照理论值计算，单个Channel一秒支持2666MT,则单Channel带宽为 2666M * 64bits(8Bytes) = `21328MB/s`，而理论6个Channel的峰值带宽为但是这是一个峰值理论带宽 127.968GB/s折合1Tbps，似乎远超以太网的带宽了，但是细节的地方都是魔鬼~

我们使用PCM工具(https://github.com/opcm/pcm)来监控内存的使用率，如下图所示：

![图片](assets/5c97e5b8f68a.png)

### 最大带宽测试

首先我们做了一些顺序和随机读写的测试，期望获得理想情况下的最大持续带宽。测试结果汇总如下
类型带宽(每个Channel)峰值比总计带宽顺序读18934MB88.7%113755.83MB随机读13807MB64.7%83006.05MB顺序读+写R 8307MB W: 8281MB  Sum: 16588MB77.7%99631.16MB随机读+写R 7890MB W: 6947MB  Sum: 14837MB69.5%89127.61MB

从这个角度来看，峰值带宽在应用正常随机读写时只剩下89GB/s(712Gbps)了.

### I/O DMA带宽

对于如何从网卡收包这件事上，发现很多工程师有误解，最常见的误解就是下面左图了。当然这图也是对的，涉及到Intel DDIO的技术，我们稍后来讲.

![图片](assets/a9a6fd89b591.png)

默认的DMA流程是，网卡收到的包直接写进主内存,然后CPU再去主内存里读，下面这个图来自于一篇ATC20'的论文[1]

![图片](assets/a757254cc4dd.png)

DDIO的优点我们后面分析，但是DDIO本身因为NetCAT漏洞[2]的缺陷使得它需要在公有云平台关闭，在Intel平台上可以通过如下方法关闭DDIO[3] 实现传统的DMA:

先后先查看PCIe Device ID

```
zartbot@ruta8:$ lspci -vt | grep 810 +-[0000:ae]-+-00.0-[af-b0]--+-00.0  Intel Corporation Ethernet Controller E810-C for QSFP |           |               \-00.1  Intel Corporation Ethernet Controller E810-C for QSFP
```

然后利用下面这个URL下载的脚本执行Disable就好

https://github.com/pmem/rpma/blob/master/tools/ddio.sh

```
zartbot@ruta8:$ sudo ./ddio.sh -d 0000:ae:00.0 -s disable
```

然后可以通过-q参数查询状态

```
zartbot@ruta8:$ sudo ./ddio.sh -d 0000:ae:00.0 -qzartbot@ruta8:$ echo $?1
```

当返回值为`1`是表示DDIO启用了，而返回值为`0`表示DDIO已经禁用了。

在DDIO禁用后，我们利用zMonkey在转发延迟为0时测试了一下，100Gpbs进入服务器并立即100Gbps发出，此时的内存带宽使用为

![图片](assets/efc5d9ff057a.png)

当然Intel E810网卡自己也不争气，只能跑到83Gbps,而Mellanox CX5依旧可以到100Gbps线速(后面会讲Intel E810 vs Mellanox CX5)

![图片](assets/9a435a816b12.png)

但是从这里可以看出，24GB的总带宽需求已经差不多200Gbps了，而整个内存物理的随机访问峰值带宽也就89GB/s，也就是说基本上就简单的I/O DMA操作就占用了接近25%的的内存带宽资源。从这个角度上来看,400Gbps直接就把整个内存带宽用光了.

### DCA & DDIO

当然Intel的工程师看到这个问题也不傻，很早就提出了DCA的技术，在写DRAM的同时去Prefetch到LLC，而AMD现在都不支持DDIO，如下左图所示

![图片](assets/e3d7990c346b.png)

但是DCA做的不是很干净，于是又在2012年做出了DDIO，如上右图，数据将不会写入DRAM直接放置在LLC中。

于是我们打开了DDIO，利用zMonkey测试100Gbps转发的性能, 此时转发100Gbps系统的内存带宽使用只有1.2GB

![图片](assets/d73abb67a725.png)

而检查CPU，发现L3 Cache HitRate也有99%

```
 Core (SKT) | EXEC | IPC  | FREQ  | AFREQ | L3MISS | L2MISS | L3HIT | L2HIT | L3MPI | L2MPI |   L3OCC |   LMB  |   RMB  | TEMP....  24    1     2.93   2.29   1.28    1.28     836      120 K    0.99    0.97    0.00    0.00      768       16        0     36  25    1     2.86   2.23   1.28    1.28     573      131 K    0.99    0.96    0.00    0.00      672       16        0     38  26    1     2.96   2.31   1.28    1.28     459      119 K    0.99    0.97    0.00    0.00      192       17        0     37  27    1     2.95   2.31   1.28    1.28     608      122 K    0.99    0.97    0.00    0.00      960       14        0     39  28    1     2.95   2.31   1.28    1.28     494      111 K    0.99    0.97    0.00    0.00      480       15        0     36  29    1     2.95   2.31   1.28    1.28     598      120 K    0.99    0.97    0.00    0.00      480       16        0     38  30    1     2.95   2.30   1.28    1.28     735      124 K    0.99    0.97    0.00    0.00      192       16        0     39  31    1     2.96   2.31   1.28    1.28     514      117 K    0.99    0.97    0.00    0.00      576       16        0     37....--------------------------------------------------------------------------------------------------------------- SKT    0     0.00   0.75   0.00    0.51      17 K    996 K    0.92    0.12    0.00    0.00    30336        3        4     51 SKT    1     0.49   2.29   0.21    1.28      23 K   1033 K    0.97    0.97    0.00    0.00    26400      130        0     36--------------------------------------------------------------------------------------------------------------- TOTAL  *     0.25   2.27   0.11    1.25      40 K   2030 K    0.96    0.93    0.00    0.00     N/A     N/A     N/A      N/A Instructions retired:   59 G ; Active cycles:   25 G ; Time (TSC): 2498 Mticks ; C0 (active,non-halted) core residency: 8.63 %
```

看上去DDIO很完美了？这是因为CPU在处理报文的时候非常的简单，而且本来代码就优化的很好，报文只需要很少几个cycle就被DMA out然后free掉了，所以被flush写到内存的概率并不高。而且系统没有其它的应用占用Cache，但是正常的应用通常需要大量的计算，正好zMonkey可以调整报文在Cache里滞留的延迟，我们通过这种测试方法来观测DDIO由于其它I/O密集空间用满被flush出去而带来的Cache Miss开销。
模拟计算延迟Cache Miss包转发速率内存读内存写I:10us<1%93Gbps984MB/s2836MB/sI:50us10%93Gbps1844MB/s4213MB/sI:100us21%93Gbps2855MB/s4881MB/sI:1ms53%93Gbps12020MB/s10770MB/sI:10ms53%88Gbps14692MB/s12134MB/sM:10ms72%93Gbps15148MB/s13028MB/sI:100ms92%84Gbps14266MB/s11638MB/sM:100ms95%93Gbps15484MB/s13219MB/s
I：表示利用Intel E810测试的结果，M：表示利用Mellanox CX5测试的而结果.可以注意到同样的CPU Socket时，代码也相同，但是Intel的虽然Cache miss rate低，但是L2 MISS的counter远高于卖螺丝，同时IPC效率也比卖螺丝低很多，网卡的Driver上还是有一些问题的,所以麻烦一下做智能网卡前，能不能先把智障网卡的一些小问题先处理好了再说？
Intel 10ms CPU使用率
```
Core (SKT) | EXEC | IPC  | FREQ  | AFREQ | L3MISS | L2MISS | L3HIT | L2HIT | L3MPI | L2MPI |   L3OCC |   LMB  |   RMB  | TEMP24    1     2.54   1.98   1.28    1.28    1178 K   2329 K    0.48    0.54    0.00    0.00     1824      257        1     3525    1     2.47   1.93   1.28    1.28    1151 K   2281 K    0.48    0.54    0.00    0.00     1920      276        2     3826    1     2.59   2.03   1.28    1.28    1186 K   2325 K    0.47    0.54    0.00    0.00     2304      275        0     3827    1     2.59   2.03   1.28    1.28    1180 K   2329 K    0.48    0.54    0.00    0.00     2208      304        1     3928    1     2.56   2.00   1.28    1.28    1182 K   2322 K    0.48    0.54    0.00    0.00     2688      263        1     3629    1     2.59   2.02   1.28    1.28    1189 K   2307 K    0.47    0.53    0.00    0.00     1632      285        2     3930    1     2.59   2.03   1.28    1.28    1183 K   2319 K    0.48    0.54    0.00    0.00     2304      241        0     4031    1     2.58   2.02   1.28    1.28    1183 K   2321 K    0.48    0.54    0.00    0.00     2592      288        0     38
```
Mellanox 10ms CPU使用率
```
Core (SKT) | EXEC | IPC  | FREQ  | AFREQ | L3MISS | L2MISS | L3HIT | L2HIT | L3MPI | L2MPI |   L3OCC |   LMB  |   RMB  | TEMP24    1     3.24   2.53   1.28    1.28    1246 K   1802 K    0.27    0.47    0.00    0.00     3072      327        0     3825    1     3.26   2.55   1.28    1.28    1137 K   1661 K    0.28    0.47    0.00    0.00     2112      274        0     4026    1     3.20   2.50   1.28    1.28    1234 K   1774 K    0.27    0.47    0.00    0.00     1824      293        0     4027    1     3.25   2.54   1.28    1.28    1128 K   1654 K    0.29    0.48    0.00    0.00     1536      315        1     4228    1     3.22   2.51   1.28    1.28    1167 K   1685 K    0.28    0.46    0.00    0.00     1920      281        0     3929    1     3.24   2.53   1.28    1.28    1218 K   1757 K    0.28    0.47    0.00    0.00     3264      335        0     4130    1     3.25   2.54   1.28    1.28    1125 K   1645 K    0.28    0.47    0.00    0.00     2016      283        0     4231    1     3.24   2.53   1.28    1.28    1217 K   1756 K    0.28    0.46    0.00    0.00     3552      338        0     40
```
Intel 100ms CPU使用率
```
Core (SKT) | EXEC | IPC  | FREQ  | AFREQ | L3MISS | L2MISS | L3HIT | L2HIT | L3MPI | L2MPI |   L3OCC |   LMB  |   RMB  | TEMP24    1     2.54   1.99   1.28    1.28    2073 K   2313 K    0.08    0.52    0.00    0.00     2016      368        0     3625    1     2.48   1.93   1.28    1.28    2063 K   2290 K    0.08    0.52    0.00    0.00     1440      388        0     3926    1     2.59   2.03   1.28    1.28    2082 K   2316 K    0.08    0.52    0.00    0.00     1536      389        0     3827    1     2.59   2.02   1.28    1.28    2073 K   2306 K    0.08    0.52    0.00    0.00      960      355        1     4028    1     2.57   2.01   1.28    1.28    2071 K   2301 K    0.08    0.52    0.00    0.00     1536      389        0     3729    1     2.59   2.02   1.28    1.28    2070 K   2294 K    0.08    0.52    0.00    0.00      960      407        0     3930    1     2.59   2.02   1.28    1.28    2075 K   2314 K    0.08    0.52    0.00    0.00     1056      376        0     4031    1     2.58   2.02   1.28    1.28    2075 K   2312 K    0.08    0.52    0.00    0.00     2016      387        1     38
```
Mellanox 100ms CPU使用率
```
Core (SKT) | EXEC | IPC  | FREQ  | AFREQ | L3MISS | L2MISS | L3HIT | L2HIT | L3MPI | L2MPI |   L3OCC |   LMB  |   RMB  | TEMP24    1     3.21   2.51   1.28    1.28    1685 K   1855 K    0.05    0.43    0.00    0.00     1440      379        0     3725    1     3.23   2.52   1.28    1.28    1640 K   1791 K    0.05    0.43    0.00    0.00     1440      365        0     4026    1     3.19   2.49   1.28    1.28    1675 K   1829 K    0.05    0.45    0.00    0.00     2112      399        1     3927    1     3.23   2.52   1.28    1.28    1630 K   1780 K    0.05    0.43    0.00    0.00     1440      368        0     4128    1     3.15   2.46   1.28    1.28    1606 K   1758 K    0.05    0.43    0.00    0.00     1824      355        2     3929    1     3.18   2.48   1.28    1.28    1652 K   1806 K    0.05    0.44    0.00    0.00     2208      377        0     4030    1     3.22   2.52   1.28    1.28    1618 K   1767 K    0.05    0.43    0.00    0.00     1440      394        0     4131    1     3.21   2.51   1.28    1.28    1670 K   1824 K    0.05    0.44    0.00    0.00     1248      406        0     39
```

DDIO的其它问题其实在另一篇论文里面也讲的很清楚了，在和一些Cache-Senstive的Application一起玩时会出现Contention而带来大量的CacheMiss

![图片](assets/d359ae906e04.png)

### 结论

所以我们需要强调的一点就是，即便是DDIO这样的技术，对于报文的到达是完全不可控的，而当Cache用满了以后，就会导致DDIO的cache被flush out并写入主内存，然后CPU后期又需要从主内存拿，这样一写一读带来的内存开销是网卡带宽的2倍，即100Gbps接收需要写12.5GB/s到主内存，并读12.5GB/s到CPU，而DDR4-2666x6读写的最大带宽也就100GB/s，也就是说未来如果主机上到200Gbps，光I/O就把主内存吃掉一半的带宽了，而400Gbps光读写做完内存就没了。

当然学术界也有一些研究，例如阿里的一些同学跟我提到的NetDIMM[4]:

![图片](assets/f4874bb0f0e5.png)

正如NetDIMM一开始要解决的问题就是**Low-Latency** Near-Memory Network，关键字是延迟(潜台词是带宽不行，用来做高频交易可以)，也就是他们提到的传统网卡在进行一个标准的Client-Server通信时，需要16个单向的PCIe transaction来处理Request-Response，所以看上去Bypass了PCIe，也是网卡上加内存，并且降低了延迟，但另一方面就是占用了主机的DRAM带宽，**加重了内存墙**。所以**关注延迟本身是牺牲带宽为代价**的。

现在工业界的很多方案来看，即便是RDMA这样的技术被炒的很火，但是都没有实质性的触碰到如何**可控节省主内存开销**的场景。即便是我们未来有8个甚至12个通道的DDR5，但是需要注意的是内存本身是一个并行总线，一个Channel就需要380pin，所以不会说那种很容易就有16Channel的东西，而且从下图来看，其实这些年内存带宽的增速还是很慢的。

![图片](assets/0825f2e2ab40.jpg)

虽然DDR5带宽也快了不少，但是访问延迟也大啊,从DDR4 CL20加到了CL44，但是好在从BL8到了BL16可以拆两个Channel访问了，理论上带宽更大使用率更高：

![图片](assets/c7889b6d317b.png)

但是真的能凑齐么？需要时间验证，即便是一个Channel能够处理30GB，也就刚好能凑齐一个100Gbps的带宽I/O需求，400Gbps直接又把8个Channel的系统内存带宽吃掉一半，而且随机读写还有可能本身要打6折的峰值带宽：

![图片](assets/05064d2fd1f9.png)

当然CPU上内挂一个HBM的SPR也是值得期待的一个技术，但是挂在那里用作I/O还是计算值得商榷，或者如AMD Milan-X那样大力出奇迹直接700MB L3Cache，也很难说对和错，对于软件的要求可能更加搞。另一方面也是DDIO的不可控和对计算任务Cache使用无法感知使得效率并不好也有安全性问题。

那么反过来看待这个问题，如果CPU能够按需的从网卡Prefetch报文如何呢？并且将I/O内存和计算内存隔离呢？

虽然说看似Pool mode 比push mode延迟更大了，但是本来在计算密集的时候DDIO的cache也会被flush到主内存。另一方面来看就是工业界CXL的出现有助于CPU去访问异构的内存，当然现在主要的场景还是各种GPU/TPU的加速卡，而在网卡上放置内存可以带来如下几个好处：

I/O内存和主内存空间的隔离，相互不需要干扰，并且至少省掉了一半的主内存开销

虽然看上去这样子会加大一些CPU访问的延迟，但是报文通过以太网到达的延迟就是数个us，所以增加一些并不会对业务带来影响，反而会因为可控的内存访问提高吞吐率和CPU效率，并缓解了内存墙。

将内存放置在网卡上对一些超算的MPI存内计算有好处，例如把需要共享给其它人的内存主动拷贝到网卡的内存上，供远端访问会大幅降低远端访问延迟
![图片](assets/47007ca3bbd5.png)

对于一些通信密集型应用和存内计算、在网计算业务提供了更好的访问能力，例如RDMA需要多次通过PCIe

![图片](assets/0778e537b0f8.png)

而我们只需要从PCIe读一次
![图片](assets/7ab266dff03e.png)

为虚拟化平台提供免通信协议栈的I/O，例如我们为Golang提供的zmemif
![图片](assets/b9e61617e792.png)

为大规模网络池化和内存+IP地址混合寻址带来了新的空间

![图片](assets/d6567898422f.png)

![图片](assets/3647317c468b.png)

为超大规模异构集群提供了通用的内存抽象层
![图片](assets/c70f3a638894.png)

有些时候，退一步，海阔天空~

#### Reference

[1]
Reexamining Direct Cache Access to Optimize I/O Intensive Applications for Multi-hundred-gigabit Networks: https://www.usenix.org/conference/atc20/presentation/farshin
[2]
NetCat: https://www.bilibili.com/read/cv3579340/
[3]
Disable DDIO: http://pmem.io/rpma/documentation/basic-direct-write-to-pmem.html
[4]
NetDIMM: Low-Latency Near-Memory Network Interface Architecture: https://dl.acm.org/doi/10.1145/3352460.3358278