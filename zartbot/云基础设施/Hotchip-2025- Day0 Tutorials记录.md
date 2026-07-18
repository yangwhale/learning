# Hotchip-2025: Day0 Tutorials记录

> 作者: zartbot  
> 日期: 2025年8月25日 09:37  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494795&idx=1&sn=7d61581ee737601bbfd8c5669c15873d&chksm=f995fc49cee2755f8d80719642eb9961de2a1d0fd4bc62272816a266578a3c340b5ee7434a0e#rd

---

又是一年的HC, 今年的Tutorials内容作为开胃菜还是挺丰富的, 早上全部是Datacenter Racks相关的内容, 下午是一些Kernel编程相关的内容. 主要记录一下和硬件相关的上午场, **比较值得关注的是Meta的GB200的定制.**

![图片](assets/2be750a089d1.png)

## 1. How AI workloads shape rack system architecture

这是来自AMD的一个Session, 大概算是过去十多年的一个回顾. 从最早的GTX580**, AlexNet开始, 感觉这个$499是在阴阳NV现在卡越来越贵了...哈哈哈

![图片](assets/911d113fc9e2.png)

然后就是回顾各种并行策略, 与之伴随而来的互连结构变化, 总体来说是在给芯片的同学科普Infra**这几年在干什么

![图片](assets/d3f0a8681b6e.png)

然后就是历史上逐年关于数制的变化, 从FP32到FP4..

![图片](assets/dcc162d14e59.png)

然后芯片的package size越来越大

![图片](assets/7ec10698e47f.png)

ScaleUP domain也在加大

![图片](assets/07d5a81284d8.png)

## 2. Scaling Fabric Technologies for AI Clusters

第二个Session也是AMD讲的, 主要科普一些Scale-UP相关的内容. 列举了一些常见的ScaleUP技术, 不写华为UB差评!

![图片](assets/ca6c734ed5f9.png)

然后大概介绍了一下ScaleUP和ScaleOut的区别

![图片](assets/a30fc7b5e9f4.png)

接着就是科普Switch的Radix和数据是ScaleUP GPU数量和带宽大小的关键影响因子

![图片](assets/20cb2e048581.png)

![图片](assets/044fc2ede20e.png)

然后就是一个典型的一层的ScaleUP网络

![图片](assets/ebaedbcaf791.png)

然后也有一些L2 ScaleUP的探讨

![图片](assets/71d26edbd590.png)

或者在L1交换机**基础上构建L1.5 Mesh

![图片](assets/13415b982d57.png)

总之前两个Session都是以科普为主.

## 3. Liquid cooling with Google Characteristics

Google在TPU上的一些经验, 液冷泵模块化设计等,

![图片](assets/f5d7831ae7f9.png)

然后是第五代准备为1MW的Rack进行设计...

![图片](assets/644c01ea1ed7.png)

## 4. Rearchitected power systems

这是来自微软的一个session, 介绍了800VDC供电.  一方面是单个机柜内ScaleUP需要放更多的GPU, 那么就需要把电源转换器外置

![图片](assets/201a562e6432.png)
于是有了RPD(rack power disaggregation)的处理

![图片](assets/181e1917d644.png)

对于整个供电链路, 从今天的架构需要多个交流变压器

![图片](assets/02ad026a96e2.png)

逐渐演化到800V中压直流(MVDC)供电:

![图片](assets/cc25a8cb0e48.png)

实际上主要是交流供电, 然后末端转直流带来的一些损耗

![图片](assets/a47ee5a90c79.png)

未来如果采用800V固态变压器(SST), 整体的供电损耗回下降不少

![图片](assets/b15e2f8c44e3.png)

其实背后还有一个逻辑, 在大规模训练的时候, 所有GPU几乎每个iteration同步启停, 对于整个供电的压力也很大. 然后还有一些谐波影响/无功功率补偿需要处理.  因此现在通常还需要做一些机柜内的电池(BBU)和电容(CBU)等. 而采用高压直流后, 可以做in-row或者在SST上做谐波和无功功率补偿.

总体来看, 800V MVDC供电可以降低一半的数据中心供电损耗, 还是很有价值的.

## 5. Case study: Nvidia GB200NVL72

Nvidia直接拿去年OCP**的一个PPT来糊弄, 直接差评...不讲了.

## 6. Case study: Meta’s Catalina (NVL72)

这是整场最值得看的一个Session. Meta对GB200机柜做了很多定制, 我们来详细看看. 另外我们可以关注一下7月AWS发布的GB200实例, 看看两者在定制化上的区别.

[《谈谈AWS GB200实例, 顺便谈谈各种自研NPU的困境》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494353&idx=1&sn=9eb27fd95ed3a3d184d4f5cbca35a235&scene=21#wechat_redirect)

Meta整个NVL72系统占用了6个机柜

![图片](assets/df260d84c7dc.jpg)

左边和右边分别有两个液冷散热柜, 和AWS一样, 采用就近的一个液冷泵然后直接风冷的方式构建, 这样一些传统的数据中心就不需要进行大规模液冷改造. 中间也采用了两个NVL36机柜并柜的方式构建.

值得注意的是, 官方的GB200采用一个Grace配两个B200, 总共18个Tray

![图片](assets/577e8cd45664.jpg)

而Meta采用了单个Grace配单个B200, 总共36个ComputeTray

![图片](assets/2b44da891c71.jpg)

官方的解释是能够有更大的内存

![图片](assets/18bc0ef6c734.png)

但实际上Meta还有更多的考虑没有对外公布. 首先我们来看官方标准的基于CX7的GB200拓扑.

![图片](assets/efaada68ace5.png)

每个Grace连接了2个B200芯片.  然后B200芯片通过PCIe x1连接到Grace, 单个Grace连接了2张CX7, 平均每个B200的ScaleOut带宽为400Gbps.  然后两颗Grace之间采用6xClink连接.

那么我们来看GPU-Direct-RDMA的场景, 如果通过PCIe GDR只有一个Gen4 x1的带宽到Blackwell上. 因此我们需要将Memory Region allocate在Grace的内存上. 然后通过NVLINK C2C去访问.

另一方面, 前面做了很多仿真分析, B200系列ScaleOut带宽需要单卡800Gbps才能满足. 因此AWS采用了外挂PCIe交换机的方式, 但是PCIe Switch应该还是一个Gen5的, 虽然为每个GPU提供了800Gbps带宽, 当前只能用到400Gbps.

![图片](assets/f11fef2bf42c.png)

原厂的CX8方案采用内置的PCIe Switch, 但是同样还是维持了一个x1连接到Grace用于PCIe管理B200设备.

![图片](assets/e05ffe6a27bf.png)

但是这个方案也不干净, 数据路径还是通过PCIe到Grace, 然后NVLink C2C到GPU.  因此在NCCL 2.27上还需要做一个特殊的Direct NIC处理.

![图片](assets/eaf51ae105c1.png)

那么我们来看Meta针对GDR的问题是如何处理的.

![图片](assets/1a2d2efb95b4.png)

首先GPU和CPU采用1:1配比, 这样Grace可以同时连接两个Gen5x16的CX7网卡, 提供单个B200 800Gbps的ScaleOut网络的能力. 另一方面B200还是通过一个Gen4x1的PCIe连接到Grace CPU上.  事实上这种方式还是无法支持RDMA网卡实现真正的GDR的. 或者说DMA的请求也是需要经过Grace的PCIe RC处理然后通过NVLink C2C写入到GPU缓存的.

后来我仔细想了一下, 既然Grace和B200之间采用1:1的配比, 其实NIC接到CPU还更好一些,  因为有C2C的存在, 不会存在原来过CPU-RC,然后8卡共享CPU内存带宽带来的瓶颈, 因此所有RDMA相关的操作GPU可以直接操作CPU的MEMORY, 而GPU读写这些RDMA数据还可以给HBM**省了接近200GB/s的带宽.

另一方面我们注意到, 由于Grace和Blackwell采用1:1配比. 因此两颗Grace之间的互连Clink带宽也翻了一倍, 变成了12个lane. 这样两个系统之间的互连内存拷贝带宽也更大了.

总体来看, 在CX8没有完全Ready的时候, Meta采用了CPU和GPU 1:1配比的方式来支持了800Gbps ScaleOut, 同时还扩大了CPU侧的内存容量和带宽.

稍微展开一下, 如果我们脑补一下 Grace+CX7 整体构成一个大号的BF3呢? 实际上你就会发现, 哇哦~ 一颗巨大的DPU一面连接ScaleOut, 另一面通过内存语义连接到ScaleUP了. 然后这个DPU还自带巨大的带宽和容量的内存. 此时你就会发现, 它做了我在2021年搞的NetDAM了...这样的方案对于KVCache这些都带来了巨大的优势.

同时Grace还可以搞点什么大众喜欢的INCA. 跑什么DeepEP的时候, 干脆一些通信算子都可以丢到Grace上来玩, 反正有NVLink C2C.

然后整个Meta Compute机柜架构如下:

![图片](assets/94c98eb1d84a.jpg)

它采用了机柜内构建BBU(电池)的方式, 并且供电采用了冗余设计, 然后ScaleOut光纤采用了一个Patch Pannel跳线, 跳线架上应该有一些冗余的光纤.  机柜内上下放置了2台Wedge400交换机, 主要用于连接FrontEnd网络, 很有意思的是Meta没有在这个地方使用BF3作为FrontEnd, 而是使用CX7和另外配的一块DC-SCM安全管理模块构成. FrontEnd每个Grace的带宽为200Gbps. 互连示意图如下所示:

![图片](assets/9129da23d7d2.jpg)

然后ScaleOut 采用了Disaggregated Scheduled Fabric

![图片](assets/fe684100aa14.jpg)

实际上搜一下Disaggregated Scheduled Fabric就清楚很有可能是用了Cisco Silicon One ?还是就是取了一个类似的名字.  无论如何其实多路径Hash冲突这些问题还是要处理干净的.

Meta还做了一件事情, 针对液冷漏液探测这些, 在每个ComputeTray上的PDB增加了一些连接leak sensor和对外连接到RMC(Rack Management Controller)的接口.

![图片](assets/6d5871792d1d.jpg)

RMC结构如下, GPIO/I2C通过RJ45接口连接, 然后液冷的GPIO也连到这个位置, 同时整个系统的漏液检验的一些外部Sensor也连接过来.

![图片](assets/47e44cd6c3fd.jpg)

最后有一个OCP规范的BMC+TPM的模块做服务器远程管理, meta并没有在这里使用BF3.

![图片](assets/5d4b24c3b1ba.jpg)

## 7. TPU Rack Overview

最后一个Session是Google介绍TPU Rack的. 对于TPU的ICI互连, 去年就有一篇详细的分析如何路由实效如何保护,弹性如何调度等.

[《大规模弹性部署：Google如何管理TPUv4集群》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489642&idx=1&sn=db30c4606db2f181f8f602c8e71abf91&scene=21#wechat_redirect)

这一次Google介绍了Ironwood整个Rack的情况

![图片](assets/9c28b310aa6d.png)

同样还是采用4x4x4构建为一个block

![图片](assets/1d41181b83d0.jpg)

然后连接到OCS光交换机也采用Fiber bundle, 并留了冗余光纤, 机柜侧也采用了跳线架(Patch Panel)

![图片](assets/6afc0a36a48e.jpg)

液冷和UPS 采用每个Row 配置一台的方式

![图片](assets/fd8e84ae0df3.jpg)