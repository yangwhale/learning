# 谈谈阿里的AI Infra磐久128卡超节点

> 作者: zartbot  
> 日期: 2025年9月25日 00:01  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496169&idx=1&sn=42da5bafd9fd1fe98902afef5a3ea42b&chksm=f995e12bcee2683d54e08e182268aec2fc2ce72a16ba8f218b891210b6eeb56ced33afb49847#rd

---

云栖大会现场最靓的硬件仔就是这个超节点了

![图片](assets/8eeef706c97d.jpg)

单机柜128卡实物展出, 和国内外的超节点Cable-Tray的方案相比, 只能说一句遥遥领先了....

### 1. 无中背板的正交架构

其实去年底在Rubin架构还没发布的时候, 就写过一篇文章来推测架构

[《推测一下Nvidia Rubin的288卡系统架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492912&idx=1&sn=7a6dddb36e0644182a57ecf0f48354ab&scene=21#wechat_redirect)

但很遗憾, Rubin还是继承了GB300的基于CableTray的架构. 而Rubin-Ultra的机框又使用的是中背板的架构

![图片](assets/b942457bb36e.jpg)

并且也没有使用正交的方式.

![图片](assets/dd2ccc1cf3a5.png)

其实在数通领域使用无中背板的正交架构已经是一个非常成熟的技术了. 例如Cisco从后背板(类似于CableTray)到中背板(类似于Rubin Ultra)再到无中背板的正交架构大概演进了10年的时间, 最后从可靠性/可运维性/可扩展性等多个角度, 这种架构已经成为主流.

其实在2022年[《一些数据中心形态的变化》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488436&idx=1&sn=a08eccaf36dcf1d85b4b40885cdd4cf6&scene=21#wechat_redirect)中也介绍过Cisco的服务器节点. 机框是一个无中背板的架构

![图片](assets/b7261079f4e4.png)

然后采用正交设计连接交换和计算板

![图片](assets/9ed18585255a.png)

对于Nvidia的遗憾, 不妨碍我们自己在阿里做一套方案.

首先相对于CableTray的方案节省了一个连接器, 实际插损会降低很多, 单个连接器插损的容错性也会高很多.

Compute-Tray采用横放的结构, 支持单个ComputeTray 4颗GPU

![图片](assets/a82c641290db.jpg)

而Switch-Tray采用竖放的结构

![图片](assets/f8eebfdc5f7f.jpg)

这样整个机柜本身没有任何CableTray, 同时整个系统针对不同的GPU厂商也可以很好的适配不同的连接器, 无论是N卡A卡或者是若干国产卡, 都可以很容的通过一个通用的机柜结构去适配.

然后CPU板并没有和GPU板在一起, 而是使用了两个标准机柜宽度的配置, 并使用PCIe Cable连接.

![图片](assets/19d8fa4f1f99.jpg)

其中GPU板的宽度超过的一个机柜的宽度, 更容易适配各种卡的结构. CPU板这一块相对来说窄一些. 这样的架构好处是, 我们可以对CPU和GPU进行动态配比.

而背部的交换板竖着放置

![图片](assets/be7efb605b04.jpg)

然后液冷相关的组件放置在最下方

![图片](assets/584697b266cd.jpg)

整个机柜维护性会比CableTray方案好很多. 当CableTray连接器故障时, 可能整个机柜都要停机, 然后更换需要拆卸大量的组建. 而这种无中背板的结构直接更换某个故障的SwitchTray或者Compute-Tray即可, 甚至可以做到不停机的热更换.  拆解和更换速度及可靠性远高于中背板或者CableTray的架构.

### 2. ScaleUP规模

整个机柜实际上是由两套64卡的超节点组成的

![图片](assets/229e058b4b71.jpg)

其实不妨碍我们从交换背板上再拉出光纤构建更大规模的两层交换机构成的ScaleUP. 但实际业务上真的需要光的ScaleUP构建千卡的规模么? 使用光意味着GPU本身的IO需要对于光的闪断等情况做大量的容错和可靠传输处理, 这会使得这些IO相关的IP占用更大的芯片面积.

在文章 [《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect) 详细阐述了这个取舍. 而且从很多并行分析的仿真和测试结论来看, 64卡的规模足够承载很多模型了. 另一方面超大规模的ScaleUP一定是需要光传输的, 可靠性的考虑需要通过GPU Die的面积来置换. 这个Trade-off是否值得?

然后这套方案也不妨碍我们去做back-to-back的连接, 把两个64的机框组合成128卡的规模.

### 3. 协议选择

其实ScaleUP的协议上, 工业界是不收敛的. 当然这套系统,即ALink System本质上是支持各种协议的. 因为连接器本身是可以通过更换SwitchTray和ComputeTray.因为没有任何CableTray和中背板的存在, 整个机柜并不需要严格约束于某个协议.

当然根据[《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect)的阐述, 可能最佳的还是做单跳的ScaleUP并采用Lossless的方式构建. 同时兼顾GPU上I/O的面积效率. 而且BRCM和AMD已经达成一些一致后. 不妨碍我们去做一些UALink Over Ethernet的事情.例如各自让一步, UALink的DL FLIT采用标准的以太网MAC层.

### 4. 小结

总体来看, 这个架构可以兼容多种卡的ScaleUP设计, 灵活支持多种不同的协议. 可靠性和可运维性都非常不错. 欢迎大家有机会到现场看看实物.