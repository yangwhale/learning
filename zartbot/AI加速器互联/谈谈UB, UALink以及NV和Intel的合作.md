# 谈谈UB, UALink以及NV和Intel的合作

> 作者: zartbot  
> 日期: 2025年9月20日 12:00  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496052&idx=1&sn=88cdf74bc65420cd73f8d688fbe98b0c&chksm=f995e1b6cee268a09efcb57adf54d3787081906d6797ecbd42a31ac38d20a3920562be18d8ad#rd

---

### TL;DR

昨天华为超节点所使用的Unified Bus协议开源了, 可以访问https://www.unifiedbus.com/ 查看, 具体协议规范只是大概看了一眼, 昨天听传宁老师讲了一下, 后面再有空详细阅读了Spec再做分析吧.

然后前天借着UB的事情和同事聊了一下一些机内总线的演进, 还在想着Intel和AMD的CPU支持UALink或者类似的高速总线替代PCIe.  结果晚上发现Intel X86 CPU会支持NVLink了... 几分感慨...

![图片](assets/9c1a55e72e5c.png)

## 1. 为什么CPU要高速总线

其实很简单的来说UB相对于其它DomainSpecific的ScaleUP相比, 最大的特点是将CPU/GPU/Memory/Storage/DPU/DSA等众多节点加入到Unified Bus中

![图片](assets/536a94cc76cb.png)

而NVLink上和Intel的合作, 也是把CPU/内存加入到ScaleUP域中. 那么取最大公约数, 我们来谈谈为什么CPU需要一个高速总线.

### 1.1 从CPU FSB和北桥谈起

我们先从大概30年前说起, 当时的CPU和内存控制器是分离的两颗芯片, 内存挂载在北桥上, 然后通过一个私有的总线连接低速的南桥, 并逐渐在南桥上连接硬盘和PCI总线扩展外设.

![图片](assets/ae322746e359.png)

然后伴随着3D显卡的发展, PCI带宽不够, 而GPU需要更大的内存访问带宽, 因此逐渐演进出了AGP总线. 而后逐渐在2001年出现了PCIe总线替代了AGP总线. 再到后面发现CPU FSB和北桥的带宽以及内存访问延迟的问题不能满足CPU的需求, 逐渐又将内存控制器和PCIe总线集成到CPU内部, 就此北桥消失了

![图片](assets/4bd58590dfb8.png)

实际上当你看到UB时, 很直观的一个感受就是, 难道北桥芯片又回来了? 另一方面我们可以看到AMD Zen系列的演进, 中间的IO Die也可以算作是北桥的回归

![图片](assets/859db0f63d71.png)

其实NV对Enfabrica的收购也是如此的逻辑

![图片](assets/e6bf72fd84b0.png)

这个逻辑在2020年做NetDAM的时候, 也很清晰的分析过.

![图片](assets/9a0035d014be.png)

当然这里不得不佩服一下华为的执行力和全栈的能力了, 其实当时在Cisco做这事也是因为内部的一些复杂的组织结构和自身没有CPU/GPU的产品线, 所以最终就换了工作跑路了. 当然在计算机体系结构中引入新的总线难度也是非常大的, 涉及大量的厂商之间的协调, 各种路线的争议. 昨天还在调侃ScaleUP大概有6~7套方案, ScaleOut也有各种各样的RDMA方案, 解决这些争议的估计只能靠时间了... 当然云的规模化效应和定制化能力可能会使得这些事情发生的稍微快一点, 这也是我当时换工作选择云的一个原因.

### 1.2 CPU为什么需要高速总线

历史上是因为高速带宽和低延迟的需求, 将北桥中的内存控制器集成到了CPU中, 而另一方面PCIe的RC也集成进入, 演进下Intel获得了更多的控制权.  想要更多的内存? 对不起, 多买CPU, 想要更多的PCIe? 对不起, 买更多的CPU. 甚至在PCIe的带宽演进中扮演了很负面的角色, 长期停留在PCIe 3.0上...当然这些恶果也在反噬Intel.

当CPU集成越来越多的核后, 也面临着平均每核的内存带宽和容量停滞不前, 然后I/O带宽演进过慢的问题. 虽然后面演进过CXL, 但是其带宽还是受到PCIe的约束演进缓慢. 即便是未来有PCIe 7.0 / 8.0 和ScaleUP这些总线还是在面积/带宽比上有显著差距, 另一方面协议上还有太多的复杂度和后向兼容能力, 使得无法快速的扩大带宽.

从芯片的角度来看, DRAM Channel占用了大量的I/O, 也约束着进一步扩展其它外设的I/O. PCIe本身把一些控制和数据通路耦合在一起也有很大的复杂度. **因此一个很朴素的逻辑, 是否能够有一个高速的数据总线拉出来?**

稍微展开一下CPU接入高速总线(NVLink/UALink/UB)的价值. 实质是内存墙的另一种转移方式. 对于KVCache而言, 需要更大的存储空间和更高的IOPS及带宽. 传统的存储域无论是SSD还是HDD要构建满足IOPS的需求并控制成本的考虑, 内存介质本身可能是一个更好的选择.

另一方面是从模型演进的角度来看, 我始终觉得大模型后续还会进一步演化出千人千面的能力和大量任务的在线学习能力, 推荐系统中的Embedding Table会以某种形式重新进入到大模型中.

除了模型以外, 大量的Agent的并行执行, 对于CPU核的密度和IO的需求也会越来越高. 很朴素的一个逻辑是以前任务可能一个人操作一台电脑, 而在Agentic时代, 一个任务可以更高的在数百台电脑上并行执行.

### 1.3 为什么不是CXL

CXL这些年的发展很慢, 虽然Intel和AMD的CPU都支持了, 但是在AI的浪潮下, GPU并不需要很多复杂的Cache一致性相关的功能, 同时CXL/PCIe控制器集成在GPU上Die面积占用太大, 因此需要一个兼顾GPU的方案. 有一些面积的Trade-off其实很早就谈过

[《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect)

## 2. 谈谈一些争议

### 2.1 不就是一个Transport么, 哪来那么多争议?

无论国内还是国外, ScaleUP的方案都有各个的选择, 其实对于做计算的人来看有一个很朴素的想法: 不就是一个transport么, 你们哪来那么多事, 加那么多戏导致整个方案不收敛, 选择一个还要考虑后续演进和兼容性问题, 烦死了..

其实我想说的是, 大概的想法大家都是一致的, 在具体实现过程中有大量的工程细节上的Trade-off, 这些才是争议的实质. 历史上Infiniband不也想通过一个协议统一各种互连么? 真要把细节看清楚的人并不多, 而其中又有大量的手没沾过泥的人在其中搅局.

另一方面, 每个厂商都有自己的利益诉求, 而这些厂商背后的leader们也有自己的技术偏好, 使得这件事变得更难了....

### 2.2 One size never fit all

工程上总有若干个理由, 想让一个协议统一全部, 其实这种思维也太理想化了. 其实很早在做NetDAM的时候就分析过, Inter-Host和Intra-Host有着巨大的差异.

![图片](assets/f67bcf3ed76f.png)

Therefore, "DAM" is required as the barrier of host to divide the intra-host and inter-host I/O into 2 different segmentations.NetDAM is designed to bridge the intra-host and interhost protocols by directly sharing memory with additional instruction level support for in-memory and in-network computing. With this architecture, CPU or Domain Specific Accelerator or other storage component could directly attach to NetDAM via AXI or CHI or PCIe/CXL bus and share the unified memory pool.

其实你看到NV刚收购的Enfabrica也是同样的逻辑.

![图片](assets/951c73f2ff12.png)

但是这件事情一直会pending在ScaleUP总线的标准统一上. Enfabrica上岸有了NVLink加持会很好玩·

### 2.3 生态兼容和后向兼容

另一方面是整个数据中心的改造并不是搞一个新总线然后全部推翻了重建, 它是一个漫长的平滑迁移的过程. 从ISA总线到PCI,再到AGP, 再到PCIe这是一个很漫长的过程. 而另一个极端是CXL过分的在意复用PCIe的链路层和物理层反而给自己挖了一个大坑.

## 3. 谈谈一些可实现的路径

### 3.1 数据(UALink/NVLink)和控制(PCIe)分离

其实不妨来看这样一个观点, 以Grace-Blackwell为例, 实际上Blackwell用了一个PCIe x1的Link连接到Grace做很多控制链路上的事情, 然后NVLink C2C做一些数据链路. 好处是原来的驱动和OS的改动都会非常小. 另一方面这样的分离对于UALink和NVLink卷带宽也变得更加容易.

一个很朴素的想法就是在XXLink上构建一个高速的IO-Switch在Rack-Level实现ScaleUP, PCIe作为一个控制路径不承载数据. 同时再构建一个PCIe to XXLink的Bridge即可.

![图片](assets/38d4a34f308c.png)

然后在协议实现上尽量简单为主, 同时考虑到Rack—Level的可靠性可以得到足够的保障, 并不需要复杂的可靠传输技术支撑,同时仅支持一跳的Switch即可, 并不需要在这个域内扩大规模.

然后对于Inter-Rack的互连, 光的可靠性和整个系统的复杂性等问题以及连接到其它存储池节点等再叠加一些可靠传输的能力.

### 3.2 NVlink的一些变数

Nvidia 和 Intel 合作, 似乎为这些带来了一些变数. 虽然有 NVLink Fusion, 但Nvidia的开放程度还是存疑. 另一方面对于Intel投入到其它更开放的高速总线(eg. UALink)也带来了一些障碍. 当然最好的结局是Nvidia 开放 NVLink, 然后第三方公司基于 NVLink 构建 switch 和其它基于此总线的设备. 但是这条路不一定走的通.

### 3.3 “失败者联盟”的机会

“失败者联盟”本来是一个戏称, 咸鱼时常都有翻身的时候. 对于AMD而言充分利用好X86生态, 拉上一众朋友利用生态的力量或许才是翻身的关键. 充分利用BRCM和Marvell在Serdes上的优势, 以及BRCM定制ASIC的生态, 再拉上一众内存厂商的朋友们才是破局的关键.