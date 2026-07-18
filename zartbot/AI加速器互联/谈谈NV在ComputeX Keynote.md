# 谈谈NV在ComputeX Keynote

> 作者: zartbot  
> 日期: 2025年5月20日 14:06  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494176&idx=1&sn=93e99cf4cee3652bc7a43581e2711503&chksm=f995fae2cee273f40485991b6956c4be64aee40f39676bf2f3da156aa5356a1785d87d50dee3#rd

---

老黄开始花了大量的时间阐述AI Factory以及从CHIP->System->DC->Infra的演进..

![图片](assets/011e5708ce73.png)

### 1. NVLink Fusion

可能最受关注的还是NVLink Fusion... 在AI Factory内支持定制化的ASIC?

![图片](assets/9e37bf4f23a6.jpg)

参与的厂家有好几个, 联发科本来就有合作做了GB10, 而Fujitsu则是A64FX超算的下一代CPU有需求. Qualcomm本来就是要回归服务器CPU市场. Alchip(世芯)/Marvell则是有一些ASIC定制的业务..而AsteraLabs则是PCIe/CXL/UAL/NVLink,反正能搞的都搞....

![图片](assets/9418a97b1d42.jpg)

其实, 仔细想想似乎这事还挺难弄的. 定制化的ASIC要接入CUDA生态, 然后Blackwell这些卡软件/内存访问上要和ASIC互通... 咋搞? 而CPU接入似乎会更加麻烦...另一个问题是, 有什么业务场景需要NV的GPU混合接一堆ASIC, CUDA生态上如何构建一些DSL, 交互的内存接口/软件生态是什么样的?

如果退而求其次, 采用NVLink构建非NV GPU/CPU的互联系统,  那么相当于卖一个NVL72的交换机框, 对比DGX-B200这类的8卡平台, NVL72的定价增益如何? 或者说这些基于NVL72的交换机框本质上是用来摊销GB200-NVL72成本用的? 然后再进一步, 其它的ASIC是否真的需要这样的互联?  所以交互上一定是一个内存接口, 该怎么设计呢? 反正我有答案就是不说... 就简单的想吃一下瓜看看 NV和BRCM在这个市场竞争一下...

其实, 我脑子里想到的是Cisco大概也在互联网泡沫的顶峰, 在核心路由器产品线(GSR)和一些交换机产品线(6500)上也是搞同样的事情. 当时Cisco也有很大带宽的交换网络, 类似于NVL72这样的机柜, 可以有多个插槽插各种Linecard.. 当时也衍生出了一堆各种各样的Service Blade, 例如ACE/SAMI/Roddick这一堆东西...如下图是一个做Wireless的SAMI卡(Service and Application Module)...里面堆了一堆处理器...

![图片](assets/9ecdb8a27b6d.png)

个人感觉最大的可能性还是挂一些内存上去, 例如Samsung/Micron这些厂商来做一些内存扩展是可行的.. 然后另一个场景是NV自己或者Asterlabs搞一些NVLINK-PCIe/CXL的桥接芯片啥的慢慢去蚕食PCIe的一些生态...

举个例子吧, 如果我在国外的大厂, 大概率会把下图这事实现了..趁着NV自己部门墙边界的约束下...

![图片](assets/f91aa934158c.png)

但是在国内, 由于中美关系的不确定性, 很大程度上不会去考虑NVLink这样的方案. 特别是国内的GPU厂商来看...

### 2. NV的企业网布局

当AI Factory的故事过度依赖于CSP时, 作为一个做过Marketing的人, 卖基础设施设备的厂商总归会盯着企业私有云的生意去做, 而这次老黄谈到的内容都涉及到这几个方面, 一个统一的架构覆盖各个价格段的场景.

![图片](assets/18683528a6f1.png)

虽然老黄在极力的推荐DGX Spark(GB10)的小盒子, 但是我个人并不太看好这个生意. 而更多的关注于DGX Station的场景. Spark不是说不好, 只是在那个价格段, 有什么场景需要桌面放一个小盒子. 而云上可以按需使用租到算力更好的平台了. 而DGX Station则是一个比较适合云上租赁模式的产品, 提供足够的弹性部署的能力.

![图片](assets/b00d7beee280.png)

谈到企业网部署, 这次详细展示的RTX Pro Server倒是一个很不错的选择. 单个机框8张RTX 6000Pro

![图片](assets/b3ae81eb77af.png)

然后比较巧妙的用了4个CX8构成的主板, 提供3.2Tbps(8x400GE)的ScaleOut带宽.

![图片](assets/5fca6b07c3fc.png)

RTX 6000Pro的规格如下, 比5090强10%以上

![图片](assets/f5a7eac2a6ba.png)

整体的性能看上去也比H100强了不少, 不过老黄这图的数据好像有点问题, ISL=128K/OSL 4K, 实际上远超了实际的workload....

![图片](assets/613980d1d307.png)

按照DeepSeek ISL=4K,OSL=1K仿真的结果如下, 大概每卡可以做到3000 Tokens/s, 并没有图上H100 4倍的性能差异.

![图片](assets/cb81da784f6d.png)

当然还有一个不得不提的问题, DeepEP在这样的部署下, 如何搞呢? CX8 RoCE的部署下, 没有了NVLink就没有了PXN, 多平面/多轨道的部署要怎么处理呢?  所以我一直坚持的一个观点是, 在网络这个领域, 除非是完全没有其它办法解决了, 千万不要动拓扑... 一时的收益可能带来后续很多麻烦... 所以面对什么Hash冲突拥塞控制多路径负载均衡的问题,还是要干干净净的去直接面对问题...

不过总体来看, 老黄有一个故事挺打动人的, 就是下面这个图. 突然觉得有点像带AI的Oracle Exdata那样的柜子了...

![图片](assets/66e27827a82c.png)

其实在这个图上已经显示出了存算分离的架构, 感觉这个机型就非常适合CSP部署提供租赁和弹性时分复用的逻辑了...

### 3. NV的运营商布局

似乎老黄还在很卖力的推销6G AI-RAN的场景... 5G很多运营商投资回报率都还偏低的情况下, 6G要多久才能成熟呢?

![图片](assets/1fd874f2928b.png)