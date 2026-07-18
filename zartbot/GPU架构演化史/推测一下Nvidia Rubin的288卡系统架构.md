# 推测一下Nvidia Rubin的288卡系统架构

> 作者: zartbot  
> 日期: 2024年12月8日 15:37  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492912&idx=1&sn=7a6dddb36e0644182a57ecf0f48354ab&chksm=f995f5f2cee27ce4f7f5fe726d6f72c710368f59f36d750a4c498eaac22fba23826ba201cb80#rd

---

### TL;DR

对于一个长期搞电信级设备的渣B, 来谈谈Nvidia下一代Rubin如何能够做到288卡的ScaleUP互联.其实在介绍Blackwell的架构时已经写过一篇文章:

[《英伟达GB200架构解析1: 互联架构和未来演进》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489088&idx=1&sn=67f25cf06a1d9128e2ff534d77089688&scene=21#wechat_redirect)

通常在高密度互联的设计时, 都会采用无中背板的结构, 将业务节点和交换节点正交放置. 例如思科在互联网泡沫后期发布的CRS-1, 几乎所有的电信级的设备都采用这样的无中背板正交放置的结构, 散热通风容易,故障率也更低. 其实这些技术在超算领域也非常成熟了,例如Cray的机柜互联结构

![图片](assets/5c58c2a76f02.png)

所有的连接器都在交换板或者计算板上, 因此就无需现在NVL72的铜互联背板了. 其实NVL72的铜互联背板很有可能对NV Blackwell整个系列是一个很大的灾难, 系统可靠性会非常差, 即便是现在说它的一些问题已经解决了, 但是个人觉得它在上线运行部署的过程中肯定还会遇到更多的稳定性的问题, 这是大多数电信设备厂商以前都遇到过的中背板失效导致整机柜故障难以更换的痛点. 只能说Nvidia在这一代NVL72上还是缺少这方面经验.

NVL72的构型更像是Cisco在互联网泡沫前期(1996年)发布的GSR12000系列路由器, 中置的交换矩阵, 上下放置业务卡, 通过一个背板连接. 而在2004年发布的CRS-1就改为正交放置的架构了. 正是这些历史的经验和教训, 渣B断定Rubin会走向无背板无CableTray正交直接互联的架构.

![图片](assets/43778671fc35.png)

接下来我们从散热和供电谈起, 最后再谈互联来推测一下整个Rubin的互联方案

### 散热和供电

上个月HPE Cray发布了一个机柜224卡Blackwell的方案, 因此散热和供电上应该是可以解决的.

![图片](assets/0033e838547d.png)

它的计算板采用了NVL4的GB200, 如下图所示:

![图片](assets/a32715f905d1.png)

HPE Cray的机柜(cabinet)其实分两种, 一种是高密部署的双宽的机柜(如下左图), 另一种是标准的机柜(如下右图)

![图片](assets/1e8905ab65b5.png)

而放置224卡的应该是这种双宽的机柜, 双宽机柜采用中间竖置的供电模块,并在机柜背部构建直流馈电母线, 然后散热液冷的冷管在两面都会布置, 并采用多个机柜共享一个CDU的方案.

![图片](assets/5e088267b21f.png)

供电模块如下图所示:

![图片](assets/dc98a672c40f.png)

实际上个人觉得Rubin的288卡架构上, 应该是在NVL72的基础上密度翻倍,并采用双机柜并柜的方式部署, 当前的供电和散热以及接插件的密度来看,不太可能在标准单机柜上构建.

### 互联架构

正如文章开头所讲, CableTray的架构存在大量的稳定性的缺陷, 在电信级的设备厂商20年前就淘汰了,转而采用正交无背板的架构, 我推测Rubin这一代ComputeTray和SwitchTray采用正交的方式构建, 下面是一个Cray的示意图,

![图片](assets/edf618219d2d.png)

ComputeTray和SwitchTray通过铜接插件直接互联, 并且机柜上还有专用的锁紧连接器的板卡锁扣. 单个ComputeTray由于机柜宽度为两个标准机柜, 因此可以很容易的放置下8颗Rubin和2个CPU

![图片](assets/75cc276dbba6.png)

但是渣B推测ScaleOut的网卡不会通过PCIe接入到这个板上, FrontEnd的网卡则有可能会放置一两块. 这样刚好ComputeTray 占用36个U就可以放下288个Rubin. 然后在SwitchTray会放置下一代的CX9/10的网卡, 整机架构如下图所示:

![图片](assets/e74cbe2d7d6e.png)

这样的架构有很多好处, 计算板和交换板都可以配置N+1的冗余, 出现故障后直接整个板子热插拔拆掉即可, 性能下降仅为1/N. 可运维的难度比NVL72小很多. 另一方面在ScaleUP域内构建2层交换机会有大量的挑战, 这些可靠性和性能相关的问题是无解的, 因此渣B断定Rubin还是会走一层交换的架构.

而对于直接将ScaleOut网卡接入到ScaleUP域内,同时保留一层ScaleUP的交换网, 则是最佳的选择, 这是渣B几年前做NetDAM时就完全探索清楚的路径.

### 对于非NV的架构

非NV体系还在谈论UEC和UALink相关的问题以及如何通过Ethernet构建ScaleUP上有一些两者需要融合的探讨. 其实对于ScaleUP/ScaleOut/FrontEnd三网融合的观点, 渣B一直在坚持, 这也是从可演进架构上探讨的, 例如AWS的ScaleOut和FrontEnd是融合的, 即便是在GB200也会使用Nitro, 详细的分析可以参考文章

[《AWS Re:Invent 从AWS CTO演讲的教训看AI云基础设施架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492892&idx=1&sn=f8bb07b46960398c4a411ea6fc4e8efb&scene=21#wechat_redirect)

![图片](assets/ed645b16435b.jpg)

![图片](assets/79269ddbbcaa.jpg)

对于FrontEnd和ScaleOut融合, 当前工业界唯一能够实现RDMA RC兼容的并且大规模商用的也只有一家

[《从Mooncake分离式大模型推理架构谈谈RDMA at Scale》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492691&idx=1&sn=584daa6901215ec87af037e997f8421e&scene=21#wechat_redirect)

技术扶贫到此为止, 再说渣B就要返贫了... 过两年这个猜测就可以验证清楚, 或者NV家的转给老黄看看, 我猜的对不对? 

反正我就要Diss当前NVL72设计上压根就是不懂高密互联CableTray的设计各种失误才搞出来了那么多麻烦事, 可能也是Time-To-Market压力下, Blackwell和架构师两个有一个能跑的TradeOff吧.