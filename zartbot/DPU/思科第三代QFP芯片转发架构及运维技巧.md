# 思科第三代QFP芯片转发架构及运维技巧

> 作者: zartbot  
> 日期: 2021年4月26日 11:03  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485782&idx=1&sn=b156eb041c02f262651ec1b7a2d24764&chksm=f9961994cee19082b66050f8808779c916f354bdf5afce1a62e3ca252b1ef0e3be21b1eac0b5#rd

---

❝
前面谈了一下华三的某个网络处理器，正好最近思科发布了完整的第三代QFP产品线的产品，ASR 1000系列的ESP-100X和ESP-200X以及Catalyst Edge 8500系列(不包括8500L).那么就来详细的讲讲这几个平台.
❞
大概15年前, 思科开始了Quantum Flow Processor(QFP)系列芯片的研发,它是思科网络处理器的其中一个分支，主要目的是针对企业网越来越复杂的软件特性和越来越大的性能需求而构建的. 另一个分支是用于CRS系列的SPP、QFA、NxPower等芯片，主打运营商市场高吞吐的Stateless转发。

### QFP设计理念：平衡 

很多厂家的系统平台都是基本转发能力可能接近1Tbps，加一个feature掉20%，再加一个掉20%，多个叠加以后性能就不满足需求了。很多客户也被这样的平台搞怕了，于是标书上不顾自己的业务需求，上来就是要1Tbps的路由器但是真部署的时候广域网带宽只有10Gbps.

同样我们在研发ISR 2900、3900系列路由器时，基本转发的最高性能可以到9Gbps，但是我们从来不会在市场上宣称这是一个万兆路由器，而更多的是考虑到多业务叠加的场景，官方建议的部署场景为15Mbps的广域网线路。

这种多个功能叠加后性能急剧下降，或者广域网像糖葫芦串那样串一堆设备的问题需要专用处理器来解决，因此QFP的整体设计理念并不在意单个软件功能的性能，而是`平衡多种软件功能`(NAT/FW/QoS/DPI/IPsec..)同时调用的性能。其实这样的场景在SASE云部署中更加明显，利用X86多核部署相对于QFP性价比会差很多倍。这个留到后面有空讲SASE的时候详细说。

通常我们会将这个处理器的平台部署在多个位置，从接入到核心，从企业到运营商，如下图所示：

![图片](assets/eb6824df477a.png)

而使用的软件功能会让支持它的人发疯，别问我为啥懂这么多，都是被这万能的盒子逼出来的...关键是这玩意支持这么多功能性能还非常棒，妥妥的瑞士军刀，似乎玩了它才会知道P4的可编程就是一个笑话了..

![图片](assets/b6b3c1e02921.png)

### QFP发展历史 

#### 第一代QFP

第一代QFP在2005年开始设计，是一个40核的并行处理器，它是两块芯片构成的一套，`报文处理芯片`叫Popeye，包含`40个核心160线程`并针对网络处理对内存访问做了优化，同时还内置了大量的硬件加速器件，而`流量管理芯片`叫Spanich提供大量的转发队列。

![图片](assets/5e9441e677ac.png)

第一代QFP伴随着ASR1000在2008年正式发售,主要是满足用户40Gbps以内的广域网需求。后期也裁剪了芯片推出了ASR1001等2.5Gbps低端平台。

#### 第二代QFP

第二代QFP伴随着工艺进步将`流量管理芯片`和`报文处理芯片`整合在一起，并且将处理器内核提升到了64核256线程. 这一代芯片可以通过两块并行的方式提供100Gbps的吞吐能力(ESP100/ASR1002-HX)，也可以四块并行提供200Gbps的处理能力(ESP200),但是第二代QFP处理器还是使用外置的加解密引擎和以太网控制器

![图片](assets/2eb069b5d674.png)

第三代QFP

第三代QFP进一步把核心数提升到了`224核896线程`, 当然某友商肯定一开口就要比核心数了，事实上这些友商芯片的竞争对手是和QFP同源的LightSpeed Plus芯片(3584个线程，用于运营商业务)。核心数相对于其它平台较少的原因在于大量的企业网软件功能需要大量的Cache，而那些数千核的处理通常因为Cache不够根本就无法支持复杂的软件特性。第三代QFP的亮点是`内建了加解密引擎`，内建了以太网控制器，支持MACSec，同样也支持双路互联(ESP200-X)，未来也有4路互联的某平台会发布..

![图片](assets/40fff4f9418b.png)

![图片](assets/a838e8a25a9e.png)

### ASR1000架构 

介绍转发流程之前,我们先来看一下ASR1000的硬件架构,以模块化的ASR1006-X机箱为例:

![图片](assets/4d4c68ce3662.png)

整个系统分为路由处理模块(RP)、转发引擎(ESP)、线卡模块(SIP/MIP/EtherentLC)、接口卡(SPA/EPA).转发架构采用集中式转发,即所有的线卡仅将报文转发至ESP，待ESP处理完后转发给出方向的线卡。

### 转发引擎带宽标定 

国内很多企业总喜欢在标书文件上面做一些处理，例如接口最大带宽、双向带宽、交换带宽等名词。思科QFP处理器则一贯保持最朴实的数据，例如ESP100X标识的就是能够处理的穿越转发引擎的流量为100G(100G进、100G出)，同理ESP200X标识200G进、200G出。虽然L2-SubSystem支持240Gbps的Serdes但是没必要去把用不到的数据给客户。而这个数据通常不是简单的包转发处理能力，而是包含了NAT、FW、DPI、QoS、ACL等多种软件功能同时打开时的处理能力。

### QFP转发架构 

#### 入方向线卡

思科线卡架构分为三种、SIP卡用于接第一代和GSR、7600、CRS共享的SPA接口卡，速率通常在10Gbps以下，支持丰富的广域网线路类型(ATM/POS/E1/E3...)，伴随着MSTP的逐渐普及，又推出了2x10GE+20x1GE或者6x10GE的以太网线卡ELC，后期伴随着每槽位100G的带宽需求又推出了MIP100线卡和对应的2x40GE/1x100GE/10x10GE/18x1GE的以太网接口卡EPA. 转发流程大同小异，我们以SIP卡为例。

![图片](assets/688b3fd36a0c.png)

入方向的包优先级分类

分类可以基于802.1p, IPv4 TOS, IPv6 TC, MPLS EXP

可以配置基于端口或者基于VLAN

入方向包括一个128MB的缓存，基于每个接口各分配了一个高低优先级队列

入方向调度器

默认：所有端口采用加权公平队列

超额带宽所有接口共享

超额带宽权重分配可以基于接口调整

SPA汇聚ASIC从入栈缓冲队列中选择报文并通过使用Interconn ASIC转换为ESI总线将报文传送给ESP

ESP通过控制信令可以实现`回压`操作(BackPressure)， 当ESP繁忙过载时，可以使用这样的方式使得SIP卡上对低优先级队列丢包而保证高优先级队列.

最常用的SIP40结构如下：

![图片](assets/2d33d1db6147.png)

MIP100结构如下，注意到了么，Ezchip的NP5c只能做一个线卡处理器做点最简单的转发优先级控制的事情,可见QFP处理器的复杂程度了吧。

![图片](assets/2d2f00a44d98.png)

入方向可以通过如下命令查看分类器和buffer使用情况

```
ASR1000#show platform hardware int gig0/2/0 plim qos input map  Interface GigabitEthernet0/2/0    Low Latency Queue (High Priority):        IP PREC, 6, 7        IPv6 TC, 46        MPLS EXP, 6, 7ASR1000#show platform hardware port 0/2/0 plim buffer settings Interface 0/2/0RX Low    Buffer Size 2064384 Bytes    Drop Threshold 1020864 Bytes    Fill Status Curr/Max 0 Bytes / 0 BytesTX Low    Interim FIFO Size 48 Cache line    Drop Threshold 35136 Bytes    Fill Status Curr/Max 0 Bytes / 3072 BytesRX High    Buffer Size 2064384 Bytes    Drop Threshold 402624 Bytes    Fill Status Curr/Max 0 Bytes / 0 BytesTX High    Interim FIFO Size 48 Cache line    Drop Threshold 35136 Bytes    Fill Status Curr/Max 0 Bytes / 5120 Bytes
```

#### ESP处理

所有有的软件转发功能都在ESP上完成，转发流程如下 

![图片](assets/7efab7d83267.png)

每个报文都将通过和线卡互联的总线送到`GPM`中保存

转发的软件进程在PPE上运行，并且可以通过访问、操作GPM来处理报文

DST(Distributor)为一个报文调度器，它将报文调度到空闲的PPE上处理
❝
注意：很多调度器采用五元组调度到某一个核心上处理会导致大象流性能很差的问题，但是QFP这里采用完全无状态但完全保序的调度算法将报文调度到空闲的核上处理。通常大象流来自于数据中心互联的场景中，很多安全设备基于流的调度(FlowBasedDistribution,FBD)会极大的影响两端主机的通信性能，而QFP则没有这样的顾虑，这也是很多客户在连接公有云专线(ExpressRoute)时也喜欢使用ASR1000的最大原因，它能够保证单个大象流多达54Mpps的处理能力。4.每个PPE线程按照一个名叫FIA(Feature Invocation Array)的顺序处理报文，这个FIA类似于一个DAG，常见的FIA如下：
❞

![图片](assets/881a966af09e.png)

❝
注: 传统的网络处理器使用固定的流水线结构， 使得在多种业务叠加时， 不同的业务都要经过一个冗长的报文处理流水线， 极大地影响了报文处理延迟和整机吞吐量。当需要增加新的业务时， 通常需要很长时间来修改整个流水线。QuantumFlow处理器引入了Feature Invocation Array的处理机制， 为不同的业务类型提供灵活的流水线结构， 节省了计算资源的消耗，同时降低了整机的报文延迟。这种灵活的流水线结构，使得多种业务可以快速叠加在原有的网络架构下。
❞
5.[可选]如果报文需要加解密处理，PPE会发送指令让加解密协处理器从GPM读取报文，然后进行相应的加解密操作后写回到GPM，在第一代和第二代都需要通过SPI总线传送到专用的加解密芯片处理，而第三代QFP直接在芯片内部就处理了

PPE将报文调度到外部的包缓冲DRAM，并放入特定的队列中，通常QFP会内置数十万个队列帮助客户配置灵活的策略，而这些集中式的队列在MSTP Hub时相对于传统的基于端口的队列非常有效。您可以将大量的队列放置在一个物理接口上(数千个子接口或者数千个IPSec隧道都可以使用这些队列).

有一个硬件的调度器会从包缓冲内抽取报文，然后基于优先级和MQC配置进行调度.调度器有一个很牛逼的功能叫三参数调度器，相对于最大最小值添加了一个新的参数，可以实现对空余带宽的灵活调度(BW Remaining Ratio/Pct).

最后报文通过调度器离开ESP送往出方向的线卡.

整个QFP的内部架构图如下:

![图片](assets/9f893faed99a.png)

出方向线卡

互联ASIC通过总线收到来自ESP的数据包

SPA汇聚ASIC接收到数据包以后将其放入出栈缓存

出栈缓存为8MB也分为64个队列，并可以设置为高低优先级队列

SPA汇聚ASIC从合适的队列中选择报文并传送给SPA总线（优先选择High Queue）

SPA卡可以通过SPI总线FIFO状态对流量进行回压操作

SPA卡将报文传送到网络接口， 完成转发

### 运维技巧 

整个转发流程如下，丢包查看的方式可以按照如下命令执行

![图片](assets/6c2ba0c82f9b.png)

show interface

```
Router#show interface ten0/0/0TenGigabitEthernet0/0/0 is up, line protocol is up  Hardware is SPA-1X10GE-L-V2, address is 00f2.8b3f.0340 (bia 00f2.8b3f.0340)  Description: ***LAN-to-LAB-Core***  MTU 1500 bytes, BW 10000000 Kbit/sec, DLY 10 usec,     reliability 255/255, txload 1/255, rxload 1/255  Encapsulation 802.1Q Virtual LAN, Vlan ID  1., loopback not set  Keepalive not supported  Full Duplex, 10000Mbps, media type is 10GBase-SR/SW  output flow-control is on, input flow-control is on  ARP type: ARPA, ARP Timeout 04:00:00  Last input 00:00:00, output 00:00:00, output hang never  Last clearing of "show interface" counters never  Input queue: 0/375/4351/34 (size/max/drops/flushes); Total output drops: 0  Queueing strategy: fifo  Output queue: 0/40 (size/max)  30 second input rate 7296000 bits/sec, 2711 packets/sec  30 second output rate 4208000 bits/sec, 1987 packets/sec     102356415687 packets input, 21655621881769 bytes, 0 no buffer     Received 5985639189 broadcasts (0 IP multicasts)     0 runts, 0 giants, 0 throttles     1564438414 input errors, 0 CRC, 0 frame, 1564438414 overrun, 0 ignored     0 watchdog, 244001497 multicast, 0 pause input     44575862788 packets output, 25117033740486 bytes, 0 underruns     Output 73031519 broadcasts (0 IP multicasts)     0 output errors, 0 collisions, 2 interface resets     8518649 unknown protocol drops     0 babbles, 0 late collision, 0 deferred     0 lost carrier, 0 no carrier, 0 pause output     0 output buffer failures, 0 output buffers swapped out     5 carrier transitions
```

show plim on subslot

```
Router#show plat hardware subslot 0/0 plim statistics0/0, SPA-1XTENGE-XFP-V2, Online  RX Pkts 100714548725 Bytes 21525386767688  TX Pkts 44576030912 Bytes 25117085344342  RX IPC Pkts 0           Bytes 0  TX IPC Pkts 0           Bytes 0
```

show plim on slot

```
Router#show plat hardware slot 0 plim statistics0/0, SPA-1XTENGE-XFP-V2, Online  RX Pkts 100715090368 Bytes 21525550808764  TX Pkts 44576540800 Bytes 25117238613247  RX IPC Pkts 0           Bytes 0  TX IPC Pkts 0           Bytes 00/1, SPA-2X1GE-V2, Online  RX Pkts 11948221670 Bytes 11711597194035  TX Pkts 62215295241 Bytes 6832603766491  RX IPC Pkts 0           Bytes 0  TX IPC Pkts 0           Bytes 00/2, Empty0/3, Empty
```

查看SIP卡到ESP的Serdes

```
Router#show plat hardware slot 0 serdes statisticsFrom Slot F0-Link A  Pkts  High: 0          Low: 44578355715 Bad: 0          Dropped: 0  Bytes High: 0          Low: 25296112629959 Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 0          Flow ctrl count: 84747956To Slot F0-Link A  Pkts  High: 2633933    Low: 100714374335ESI Bandwidth utilization for last 57 seconds:From Slot F0-Link A                    Avg                                  Peak  Pkts High(pps):   0                                       0  Pkts Low(pps):    9670                                11557  Bytes High(Bps):  0                                       0  Bytes Low(Bps):   2825383                           4497969From Slot F0-Link B  Pkts  High: 0          Low: 62215411350 Bad: 0          Dropped: 0  Bytes High: 0          Low: 7081497518781 Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 0          Flow ctrl count: 981493To Slot F0-Link B  Pkts  High: 7587799    Low: 11940739014ESI Bandwidth utilization for last 57 seconds:From Slot F0-Link B                    Avg                                  Peak  Pkts High(pps):   0                                       0  Pkts Low(pps):    779                                  1983  Bytes High(Bps):  0                                       0  Bytes Low(Bps):   198029                             893076From Slot F1-Link A  Pkts  High: 0          Low: 18823159858 Bad: 0          Dropped: 0  Bytes High: 0          Low: 5077915853312 Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 0          Flow ctrl count: 941318To Slot F1-Link A  Pkts  High: 0          Low: 18823159858ESI Bandwidth utilization for last 57 seconds:From Slot F1-Link A                    Avg                                  Peak  Pkts High(pps):   0                                       0  Pkts Low(pps):    0                                       0  Bytes High(Bps):  0                                       0  Bytes Low(Bps):   0                                       0From Slot F1-Link B  Pkts  High: 0          Low: 9306294300 Bad: 0          Dropped: 0  Bytes High: 0          Low: 14095006517180 Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 0          Flow ctrl count: 942153To Slot F1-Link B  Pkts  High: 0          Low: 9306294300ESI Bandwidth utilization for last 57 seconds:From Slot F1-Link B                    Avg                                  Peak  Pkts High(pps):   0                                       0  Pkts Low(pps):    0                                       0  Bytes High(Bps):  0                                       0  Bytes Low(Bps):   0                                       0
```

5.查看ESP接收方向的Serdes

```
Router#show plat hardware slot f0 serdes statisticsFrom Slot R1-Link A  Pkts  High: 0          Low: 2530       Bad: 0          Dropped: 0  Bytes High: 0          Low: 242880     Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 0          Flow ctrl count: 940600To Slot R1-Link A  Pkts  High: 0          Low: 0ESI Bandwidth utilization for last 105 seconds:From Slot R1-Link A                    Avg                                  Peak  Pkts High(pps):   0                                       0  Pkts Low(pps):    0                                       0  Bytes High(Bps):  0                                       0  Bytes Low(Bps):   0                                       0From Slot R0-Link A  Pkts  High: 90625692   Low: 297807408  Bad: 0          Dropped: 0  Bytes High: 15196982017 Low: 143581015824 Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 0          Flow ctrl count: 1010683To Slot R0-Link A  Pkts  High: 359648115  Low: 193220193ESI Bandwidth utilization for last 105 seconds:From Slot R0-Link A                    Avg                                  Peak  Pkts High(pps):   19                                     30  Pkts Low(pps):    46                                    847  Bytes High(Bps):  3221                                 5966  Bytes Low(Bps):   14572                             1104232From Slot F1-Link A  Pkts  High: 0          Low: 1699472    Bad: 0          Dropped: 0  Bytes High: 0          Low: 61186204   Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 0          Flow ctrl count: 940586To Slot F1-Link A  Pkts  High: 0          Low: 1699470ESI Bandwidth utilization for last 105 seconds:From Slot F1-Link A                    Avg                                  Peak  Pkts High(pps):   0                                       0  Pkts Low(pps):    0                                       0  Bytes High(Bps):  0                                       0  Bytes Low(Bps):   0                                       0From Slot 0-Link A  Pkts  High: 2633946    Low: 100714834657 Bad: 0          Dropped: 0  Bytes High: 200710569  Low: 21928939767741 Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 9405934    Flow ctrl count: 940610To Slot 0-Link A  Pkts  High: 0          Low: 44578787326ESI Bandwidth utilization for last 105 seconds:From Slot 0-Link A                    Avg                                  Peak  Pkts High(pps):   0                                       3  Pkts Low(pps):    10461                               13082  Bytes High(Bps):  33                                    226  Bytes Low(Bps):   3115869                           4946229From Slot 0-Link B  Pkts  High: 7588069    Low: 11940757692 Bad: 0          Dropped: 0  Bytes High: 738918998  Low: 11758678030384 Bad: 0          Dropped: 0  Pkts  Looped: 0          Error: 0  Bytes Looped 0  Qstat count: 245608982  Flow ctrl count: 940610To Slot 0-Link B  Pkts  High: 0          Low: 62215435887ESI Bandwidth utilization for last 105 seconds:From Slot 0-Link B                    Avg                                  Peak  Pkts High(pps):   5                                     126  Pkts Low(pps):    562                                  1656  Bytes High(Bps):  675                                 37314  Bytes Low(Bps):   79596                              794938
```

6.查看QFP的IPM

```
Router#Sh plat hard qfp active bqs 0 ipm  mappingBQS IPM Channel MappingChan   Name                Interface      Port     CFIFO 1     CC3 Low             SPI0           0        1 2     CC3 Hi              SPI0           1        0 3     CC2 Low             SPI0           2        1 4     CC2 Hi              SPI0           3        0 5     CC1 Low             SPI0           4        1 6     CC1 Hi              SPI0           5        0 7     CC0 Low             SPI0           6        1 8     CC0 Hi              SPI0           7        0 9     RP1 Low             SPI0           8        110     RP1 Hi              SPI0           9        011     RP0 Low             SPI0          10        112     RP0 Hi              SPI0          11        013     Peer-FP Low         SPI0          12        314     Peer-FP Hi          SPI0          13        215     Nitrox Low          SPI0          14        116     Nitrox Hi           SPI0          15        017     HT Pkt Low          HT             0        118     HT Pkt Hi           HT             1        019     HT IPC Low          HT             2        320     HT IPC Hi           HT             3        221     CC4 Low             SPI0          16        122     CC4 Hi              SPI0          17        023     CC5 Low             SPI0          18        124     CC5 Hi              SPI0          19        0Router#Sh plat hard qfp active bqs 0 ipm statistics channel allBQS IPM Channel StatisticsChan   GoodPkts  GoodBytes    BadPkts   BadBytes 1 - 0000000000 0000000000 0000000000 0000000000 2 - 0000000000 0000000000 0000000000 0000000000 3 - 0000000000 0000000000 0000000000 0000000000 4 - 0000000000 0000000000 0000000000 0000000000 5 - 0000000000 0000000000 0000000000 0000000000 6 - 0000000000 0000000000 0000000000 0000000000 7 - 1a3ad53e65 1ea38fec308a 0000000000 0000000000 8 - 00009bfa72 003801cddf 0000000000 0000000000 9 - 00000009ee 000003b940 0000000000 000000000010 - 0000000000 0000000000 0000000000 000000000011 - 0011c036db 216e1c74cb 0000000000 000000000012 - 000566da8a 0389d24396 0000000000 000000000013 - 000019ee90 0003a5a09c 0000000000 000000000014 - 0000000000 0000000000 0000000000 000000000015 - 0000000000 0000000000 0000000000 000000000016 - 0000000000 0000000000 0000000000 000000000017 - 0000000000 0000000000 0000000000 000000000018 - 0000000000 0000000000 0000000000 000000000019 - 00069fc009 0856e618b4 0000000000 000000000020 - 0000000000 0000000000 0000000000 000000000021 - 0000000000 0000000000 0000000000 000000000022 - 0000000000 0000000000 0000000000 000000000023 - 0000000000 0000000000 0000000000 000000000024 - 0000000000 0000000000 0000000000 0000000000
```

查看QFP处理时的报文情况

```
Router#Show platform hardware qfp active interface if-name ten0/0/0 statisticsPlatform Handle 8----------------------------------------------------------------Receive Stats                             Packets        Octets----------------------------------------------------------------  Ipv4                                       0               0  Ipv6                                       0               0  Tag                                        0               0  McastIpv4                                  0               0  McastIpv6                                  0               0  FragIpv4                                   0               0  FragIpv6                                   0               0  ReassIpv4                                  0               0  ReassIpv6                                  0               0  Other                                      0               0----------------------------------------------------------------Transmit Stats                            Packets        Octets----------------------------------------------------------------  Ipv4                                       0               0  Ipv6                                       0               0  Tag                                        0               0  McastIpv4                                  0               0  McastIpv6                                  0               0  FragmentsIpv4                              0               0  FragmentsIpv6                              0               0  FragmentedIpv4                             0               0  FragmentedIpv6                             0               0  Other                                      0               0----------------------------------------------------------------Input Drop Stats                          Packets        Octets----------------------------------------------------------------  Ingress Drop stats are not enabled on this interface----------------------------------------------------------------Output Drop Stats                         Packets        Octets----------------------------------------------------------------  The Egress Drop stats are not enabled on this interface----------------------------------------------------------------Drop Stats Summary:note: 1) these drop stats are only updated when PAL         reads the interface stats.      2) the interface stats include the subinterfaceInterface                                       Rx Pkts             Tx Pkts---------------------------------------------------------------------------TenGigabitEthernet0/0/0                      3404432835                  12Router#Show platform hardware qfp active statistics dropLast clearing of QFP drops statistics : never-------------------------------------------------------------------------Global Drop Stats                         Packets                  Octets-------------------------------------------------------------------------BadUidbIdx                                     32                    4016CTSNotEnabled                               10735                 1976862Disabled                                     3099                  579098Erspan                                        360                   23776FirewallInvalidZone                             6                     412ForUs                                      206323                13708544Icmp                                         2011                   93056InjectErr                                      86                   42125IpBadOptions                               175602                19318672IpFormatErr                                 33454                33273543IpFragErr                                       1                    1518IpTtlExceeded                               31517                 1824789Ipv4Acl                                1605098153            122584873970Ipv4Martian                                    18                    1336Ipv4NoAdj                               530942788            267740833430Ipv4NoRoute                                     1                      64Ipv4Null0                                      12                     824Ipv4Unclassified                           380311                33492918Ipv4uRpfStrictFailed                   2748733308           1237701703600NatOut2in                                     173                   14296PuntPerCausePolicerDrops               4158279850            823084780206ReassDrop                                    8946                 1289776ReassNoFragInfo                              9681                14366780ReassTimeout                                10426                  919273TailDrop                               1552030255            307335554422UnconfiguredIpv4Fia                      14035997              1956979874UnconfiguredIpv6Fia                     105672577             11804228706Router#Show platform hardware qfp active infra bqs queue out default interface ten0/0/0Interface: TenGigabitEthernet0/0/0 QFP: 0.0 if_h: 8 Num Queues/Schedules: 1  Queue specifics:    Index 0 (Queue ID:0x9b, Name: TenGigabitEthernet0/0/0)    Software Control Info:      (cache) queue id: 0x0000009b, wred: 0x88b3a992, qlimit (bytes): 32812544      parent_sid: 0x260, debug_name: TenGigabitEthernet0/0/0      sw_flags: 0x08000011, sw_state: 0x00000c01, port_uidb: 245752      orig_min  : 0                   ,      min: 1050000000      min_qos   : 0                   , min_dflt: 0      orig_max  : 0                   ,      max: 0      max_qos   : 0                   , max_dflt: 0      share     : 1      plevel    : 0, priority: 65535      defer_obj_refcnt: 0    Statistics:      tail drops  (bytes): 0                   ,          (packets): 0      total enqs  (bytes): 25119133968420      ,          (packets): 44580725049      queue_depth (bytes): 0      licensed throughput oversubscription drops:                  (bytes): 0                   ,          (packets): 0Router#Show platform hardware qfp active interface all statistics  drop_summary----------------------------------------------------------------Drop Stats Summary:note: 1) these drop stats are only updated when PAL         reads the interface stats.      2) the interface stats include the subinterfaceInterface                                       Rx Pkts             Tx Pkts---------------------------------------------------------------------------Tunnel11                                            269                   0TenGigabitEthernet0/0/0                      3404437687                  12GigabitEthernet0/1/1                                  0           929040342GigabitEthernet0/1/0                          309969023          1904458990Router#Show platform hardware qfp active interface if-name Ten0/0/0 statistics  drop_summaryPlatform Handle 8----------------------------------------------------------------Receive Stats                             Packets        Octets----------------------------------------------------------------  Ipv4                                       0               0  Ipv6                                       0               0  Tag                                        0               0  McastIpv4                                  0               0  McastIpv6                                  0               0  FragIpv4                                   0               0  FragIpv6                                   0               0  ReassIpv4                                  0               0  ReassIpv6                                  0               0  Other                                      8             966----------------------------------------------------------------Transmit Stats                            Packets        Octets----------------------------------------------------------------  Ipv4                                       0               0  Ipv6                                       0               0  Tag                                        0               0  McastIpv4                                  0               0  McastIpv6                                  0               0  FragmentsIpv4                              0               0  FragmentsIpv6                              0               0  FragmentedIpv4                             0               0  FragmentedIpv6                             0               0  Other                                      1             412----------------------------------------------------------------Input Drop Stats                          Packets        Octets----------------------------------------------------------------  Ingress Drop stats are not enabled on this interface----------------------------------------------------------------Output Drop Stats                         Packets        Octets----------------------------------------------------------------  The Egress Drop stats are not enabled on this interface----------------------------------------------------------------Drop Stats Summary:note: 1) these drop stats are only updated when PAL         reads the interface stats.      2) the interface stats include the subinterfaceInterface                                       Rx Pkts             Tx Pkts---------------------------------------------------------------------------TenGigabitEthernet0/0/0                      3404440702                  12
```

查看OPM

```
Router#Sh plat hard qfp active bqs 0 opm  mappingBQS OPM Channel MappingChan     Name                          Interface      LogicalChannel 0       CC0 Low                       SPI0            0 1       CC0 Hi                        SPI0            1 2       CC0B Low                      SPI0            2 3       CC0B Hi                       SPI0            3 4       CC1 Low                       SPI0            4 5       CC1 Hi                        SPI0            5 6       CC1B Low                      SPI0            6 7       CC1B Hi                       SPI0            7 8       CC2 Low                       SPI0            8 9       CC2 Hi                        SPI0            910       CC2B Low                      SPI0           1011       CC2B Hi                       SPI0           1112       CC3 Low                       SPI0           1213       CC3 Hi                        SPI0           1314       CC3B Low                      SPI0           1415       CC3B Hi                       SPI0           1516       CC4 Low                       SPI0           1617       CC4 Hi                        SPI0           1718       CC5 Low                       SPI0           1819       CC5 Hi                        SPI0           1920       RP0 Low                       SPI0           2021       RP0 Hi                        SPI0           2122       RP1 Low                       SPI0           2223       RP1 Hi                        SPI0           2324       Peer-FP Low                   SPI0           2425       Peer-FP Hi                    SPI0           2526       Nitrox Low                    SPI0           2627       Nitrox Hi                     SPI0           2728       HT Pkt Low                    HT              029       HT Pkt Hi                     HT              130       HT IPC Low                    HT              231       HT IPC Hi                     HT              332       Unmapped33       Unmapped34       Unmapped35       Unmapped36       Unmapped37       Unmapped38       HighNormal                    GPM             739       HighPriority                  GPM             640       LowNormal                     GPM            1141       LowPriority                   GPM            1042       InternalTrafficHiChannel      GPM            1243       InternalTrafficLoChannel      GPM            1344       AttnTrafficHiChannel          GPM            1445       MetaPktTrafficChannel         GPM            1546       Unmapped47       Unmapped48       Unmapped49       Unmapped50       Unmapped51       Unmapped52       Unmapped53       Unmapped54       Unmapped55*      Drain Low                     GPM             0 * - indicates the drain mode bit is set for this channelRouter#Sh plat hard qfp active bqs 0 opm statistics channel allBQS OPM Channel StatisticsChan   GoodPkts  GoodBytes    BadPkts   BadBytes 0 - 0a61464543 0217616154 0000000000 0000000000 1 - 0000000000 0000000000 0000000000 0000000000 2 - 0e7c592be0 70cea812ba 0000000000 0000000000 3 - 0000000000 0000000000 0000000000 0000000000 4 - 0000000000 0000000000 0000000000 0000000000 5 - 0000000000 0000000000 0000000000 0000000000 6 - 0000000000 0000000000 0000000000 0000000000 7 - 0000000000 0000000000 0000000000 0000000000 8 - 0000000000 0000000000 0000000000 0000000000 9 - 0000000000 0000000000 0000000000 000000000010 - 0000000000 0000000000 0000000000 000000000011 - 0000000000 0000000000 0000000000 000000000012 - 0000000000 0000000000 0000000000 000000000013 - 0000000000 0000000000 0000000000 000000000014 - 0000000000 0000000000 0000000000 000000000015 - 0000000000 0000000000 0000000000 000000000016 - 0000000000 0000000000 0000000000 000000000017 - 0000000000 0000000000 0000000000 000000000018 - 0000000000 0000000000 0000000000 000000000019 - 0000000000 0000000000 0000000000 000000000020 - 000b846d1b 0ad4934b7a 0000000000 000000000021 - 00156ffb1d 0e21e6d1d8 0000000000 000000000022 - 0000000000 0000000000 0000000000 000000000023 - 0000000000 0000000000 0000000000 000000000024 - 000019ee90 000d88d7c0 0000000000 000000000025 - 0000000000 0000000000 0000000000 000000000026 - 0000000000 0000000000 0000000000 000000000027 - 0000000000 0000000000 0000000000 000000000028 - 0000000000 0000000000 0000000000 000000000029 - 0000000000 0000000000 0000000000 000000000030 - 0006a3c99b 0794486c70 0000000000 000000000031 - 0000000000 0000000000 0000000000 000000000032 - 0000000000 0000000000 0000000000 000000000033 - 0000000000 0000000000 0000000000 000000000034 - 0000000000 0000000000 0000000000 000000000035 - 0000000000 0000000000 0000000000 000000000036 - 0000000000 0000000000 0000000000 000000000037 - 0000000000 0000000000 0000000000 000000000038 - 0000c21648 04347f9d37 0000000000 000000000039 - 0000000000 0000000000 0000000000 000000000040 - 0000000000 0000000000 0000000000 000000000041 - 0000000000 0000000000 0000000000 000000000042 - 00078edc8f 2665629f84 0000000000 000000000043 - 1b1c1b0b3b bc87423bb4 0000000000 000000000044 - 48f8d12e29 fb36e10c7c 0000000000 000000000045 - 0000000000 0000000000 0000000000 000000000046 - 0000000000 0000000000 0000000000 000000000047 - 0000000000 0000000000 0000000000 000000000048 - 0000000000 0000000000 0000000000 000000000049 - 0000000000 0000000000 0000000000 000000000050 - 0000000000 0000000000 0000000000 000000000051 - 0000000000 0000000000 0000000000 000000000052 - 0000000000 0000000000 0000000000 000000000053 - 0000000000 0000000000 0000000000 000000000054 - 0000000000 0000000000 0000000000 000000000055 - 0000000000 0000000000 0000000000 000000000056 - 00005b9114 000ea64a38 0000000000 000000000057 - 0000000000 0000000000 0000000000 000000000058 - 0000000000 0000000000 0000000000 000000000059 - 0000000000 0000000000 0000000000 000000000060 - 0000000000 0000000000 0000000000 0000000000 0-55: OPM Channels56-59: Metapacket/Recycle Pools 0-3   60: Reassembled Packets Sent to QED
```

### 抓包分析 

QFP内置抓包工具可以通过这个功能将报文捕获后导出成pcap格式

```
HQ-InternetGW# monitor capture demo interface gi0/0/0 both match ipv4 10.74.30.0/24 10.74.30.0/24 HQ-InternetGW# monitor capture demo startHQ-InternetGW# show monitor capture demo buffer detailed  ------------------------------------------------------------- #   size   timestamp     source      destination   protocol -------------------------------------------------------------   0    0    0.000000   10.74.30.198     ->  10.74.30.136     TCP  0000:  005056B2 6774503D E5973480 08004500   .PV.gtP=..4...E.  0010:  0034A78E 40003C06 45540A4A 1EC60A4A   .4..@.<.ET.J...J  0020:  1E880405 00507188 395AC7CE 6CD98010   .....Pq.9Z..l...  0030:  03FB9DF8 00000101 080A568B A139A0FE   ..........V..9..HQ-InternetGW#  Monitor capture demo export ftp://a:b@1.1.1.1/demo.pcapHQ-InternetGW#  monitor capture demo stop
```

### Packet-Trace 

还有一个更有用的功能是Packet-Trace，思科的TAC和研发非常喜欢用的工具.例如我要从某个接口捕获某个网段的报文处理情况，可以通过如下方式打开debug

```
HQ-InternetGW# debug platform condition interface gi0/1/0 ipv4 10.74.30.0/24 ingress HQ-InternetGW# debug platform packet-trace packet 128 fia-trace data-size 16384 circular HQ-InternetGW# debug platform packet-trace packet 128 circular fia-trace data-size 16384HQ-InternetGW# debug platform packet-trace copy packet both HQ-InternetGW# debug platform condition start
```

检查debug配置,当然这里面还可以加一些和某个软件特性相关的debug

```
Router#show debugIOSXE Conditional Debug Configs:Conditional Debug Global State: StartConditions                                                                                     Direction----------------------------------------------------------------------------------------------|---------GigabitEthernet0/1/0                     & IPV4 [10.74.30.0/24]                                ingressFeature Condition       Type                     Value-----------------------|------------------------|--------------------------------------------------------Feature      Type           Submode                                                                                       Level------------|-------------|----------------------------------------------------------------------------------------------|----------IOSXE Packet Tracing Configs:debug platform packet-trace packet 128 circular fia-trace data-size 16384debug platform packet-trace copy packet both size 64 L2license policy manager client:  platform software policy_manager_error debugging is onMACSec:  MACsec errors debugging is onPacket Infra debugs:Ip Address                                               Port------------------------------------------------------|----------
```

查看捕获的报文

```
Router#show platform packet-trace summaryPkt   Input                     Output                    State  Reason0     Gi0/1/0                   Te0/0/0.504               FWD1     Gi0/1/0                   Te0/0/0.504               FWD2     Gi0/1/0                   Te0/0/0.504               FWD3     Gi0/1/0                   Te0/0/0.504               FWD4     Gi0/1/0                   Te0/0/0.504               FWD5     Gi0/1/0                   Te0/0/0.504               FWD6     Gi0/1/0                   Te0/0/0.504               FWD7     Gi0/1/0                   Te0/0/0.504               FWD8     Gi0/1/0                   Te0/0/0.504               FWD9     Gi0/1/0                   Te0/0/0.504               FWD
```

停止debug并查看

```
Router#debug platform condition stopRouter#show platform packet-trace packet 2Packet: 2           CBUG ID: 601Summary  Input     : GigabitEthernet0/1/0  Output    : TenGigabitEthernet0/0/0.504  State     : FWD  Timestamp    Start   : 4703535693829076 ns (04/26/2021 10:41:56.811018 UTC)    Stop    : 4703535694077163 ns (04/26/2021 10:41:56.811266 UTC)Path Trace  Feature: IPV4(Input)    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Source      : 1.192.193.158    Destination : 10.74.30.31    Protocol    : 6 (TCP)      SrcPort   : 80      DstPort   : 54466  Feature: DEBUG_COND_INPUT_PKT    Entry       : Input - 0x8a013940    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 77 ns  Feature: IPV4_INPUT_DST_LOOKUP_ISSUE    Entry       : Input - 0x8a0138ac    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 7 ns  Feature: IPV4_INPUT_ARL_SANITY    Entry       : Input - 0x8a013d28    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 38 ns  Feature: IPV4_INPUT_SRC_LOOKUP_ISSUE    Entry       : Input - 0x8a013d30    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 5 ns  Feature: IPV4_INPUT_DST_LOOKUP_CONSUME    Entry       : Input - 0x8a01390c    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 10 ns  Feature: IPV4_INPUT_ACL    Entry       : Input - 0x8a013d58    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 70 ns  Feature: IPV4_INPUT_SRC_LOOKUP_CONSUME    Entry       : Input - 0x8a013d60    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 14 ns  Feature: IPV4_INPUT_FOR_US_MARTIAN    Entry       : Input - 0x8a013910    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 5 ns  Feature: CFT    API                   : cft_handle_pkt    packet capabilities   : 0x0000018c    input vrf_idx         : 0    calling feature       : STILE    direction             : Input    triplet.vrf_idx       : 0    triplet.network_start : 0x0003401a    triplet.triplet_flags : 0x00000000    triplet.counter       : 0    cft_bucket_number     : 35591    cft_l3_payload_size   : 20    cft_pkt_ind_flags     : 0x00000140    cft_pkt_ind_valid     : 0x00009bff    tuple.src_ip          : 1.192.193.158    tuple.dst_ip          : 10.74.30.31    tuple.src_port        : 80    tuple.dst_port        : 54466    tuple.vrfid           : 0    tuple.l4_protocol     : TCP    tuple.l3_protocol     : IPV4    pkt_sb_state          : 0    pkt_sb.num_flows      : 1    pkt_sb.tuple_epoch    : 0    returned cft_error    : 0    returned fid          : 0x90a60070  Feature: NBAR    Packet number in flow: N/A    Classification state: Final    Classification name: binary-over-http    Classification ID: [CANA-L7:431]    Classification source: Unknown    Number of matched sub-classifications: 0    Number of extracted fields: 0    Is PA (split) packet: False    TPH-MQC bitmask value: 0x1    Is optimize packet: False    Is allow packet: False    Source MAC address: 84:B2:61:8F:BA:80    Destination MAC address: 00:F2:8B:3F:03:50  Feature: IPV4_INPUT_STILE_LEGACY    Entry       : Input - 0x8a013d80    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 2450 ns  Feature: DEBUG_COND_APPLICATION_IN    Entry       : Input - 0x8a013944    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 11 ns  Feature: IPV4_INGRESS_MMA_LOOKUP    Entry       : Input - 0x8a013d90    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 1071 ns  Feature: IPV4_INPUT_FME_PROCESS    Entry       : Input - 0x8a013d94    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 1449 ns  Feature: IPV4_INPUT_FNF_AOR_FIRST    Entry       : Input - 0x8a013da0    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 493 ns  Feature: IPV4_INPUT_FNF_FIRST    Entry       : Input - 0x8a013da8    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 1530 ns  Feature: DEBUG_COND_APPLICATION_IN_CLR_TXT    Entry       : Input - 0x8a013948    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 13 ns  Feature: IPV4_INPUT_VFR    Entry       : Input - 0x8a013e14    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 6 ns  Feature: OCE_TRACE(Input)    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Type        : OCE_ADJ_IPV4  Feature: NAT    Direction : OUT to IN    Action    : FWD    FWD-POINT : LOOKUP_FAIL    VRF       :   0  Feature: IPV4_NAT_INPUT_FIA    Entry       : Input - 0x8a013e40    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 2129 ns  Feature: IPV4_INPUT_TCP_ADJUST_MSS    Entry       : Input - 0x8a013e84    Input       : GigabitEthernet0/1/0    Output      : <unknown>    Lapsed time : 25 ns  Feature: IPV4_INPUT_LOOKUP_PROCESS    Entry       : Input - 0x8a0138b4    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 102 ns  Feature: IPV4_INPUT_FNF_AOR_FINAL    Entry       : Input - 0x8a013ec0    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 232 ns  Feature: IPV4_INPUT_FNF_FINAL    Entry       : Input - 0x8a013ec8    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 1795 ns  Feature: IPV4_INPUT_FNF_AOR_RELEASE    Entry       : Input - 0x8a013ecc    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 37 ns  Feature: IPV4_INPUT_IPOPTIONS_PROCESS    Entry       : Input - 0x8a013ed0    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 7 ns  Feature: IPV4_INPUT_GOTO_OUTPUT_FEATURE    Entry       : Input - 0x8a013ef0    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 28 ns  Feature: NAT    Direction : IN to OUT    Action    : FWD    FWD-POINT : NOT_IN2OUT  Feature: ZBFW    Action  : Fwd    Zone-pair name         : N/A    Class-map name         : N/A    Input interface        : GigabitEthernet0/1/0    Egress interface       : TenGigabitEthernet0/0/0.504    Input VPN ID           : 65535    Ouput VPN ID           : 65535    AVC Classification ID  : 0    AVC Classification name: N/A  Feature: IPV4_OUTPUT_INSPECT    Entry       : Output - 0x8a013f0c    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 929 ns  Feature: IPV4_OUTPUT_THREAT_DEFENSE    Entry       : Output - 0x8a013f40    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 26 ns  Feature: IPV4_VFR_REFRAG    Entry       : Output - 0x8a013c90    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 5 ns  Feature: IPV4_OUTPUT_L2_REWRITE    Entry       : Output - 0x8a00c17c    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 81 ns  Feature: NBAR    Packet number in flow: N/A    Classification state: Final    Classification name: binary-over-http    Classification ID: [CANA-L7:431]    Classification source: Unknown    Number of matched sub-classifications: 0    Number of extracted fields: 0    Is PA (split) packet: False    TPH-MQC bitmask value: 0x1    Is optimize packet: False    Is allow packet: False    Source MAC address: 00:F2:8B:3F:03:40    Destination MAC address: 3C:97:0E:27:A4:93  Feature: IPV4_OUTPUT_STILE_LEGACY    Entry       : Output - 0x8a013f8c    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 1136 ns  Feature: IPV4_OUTPUT_FRAG    Entry       : Output - 0x8a013fe4    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 5 ns  Feature: IPV4_OUTPUT_DROP_POLICY    Entry       : Output - 0x8a014074    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 112 ns  Feature: MARMOT_SPA_D_TRANSMIT_PKT    Entry       : Output - 0x8a013af0    Input       : GigabitEthernet0/1/0    Output      : TenGigabitEthernet0/0/0.504    Lapsed time : 1039 nsPacket Copy In  00f28b3f 035084b2 618fba80 08004500 0028b411 40003506 a5f701c0 c19e0a4a  1e1f0050 d4c246d0 1d041a36 01185010 040b6bcd 00000000 00000000Packet Copy Out  3c970e27 a49300f2 8b3f0340 810001f8 08004500 0028b411 40003406 a6f701c0  c19e0a4a 1e1f0050 d4c246d0 1d041a36 01185010 040b6bcd 0000
```