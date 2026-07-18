# 谈谈以太网GPU Scale-UP的工作EthLink

> 作者: zartbot  
> 日期: 2025年4月30日 11:09  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494110&idx=2&sn=c837699b7c3cff65122fcc1e17f805fd&chksm=f995f91ccee2700a87a87f0706ee27ba5ff35faca52406a7c1411e9880a12c3e578a40a2ff8b#rd

---

`本文仅代表个人观点, 与作者任职的机构无关`

昨天隔壁友商发布了一个**“全国首个!填补以太网GPU Scale-UP互联协议空白”** 的白皮书.仔细想了一下, 腾讯的ETH-X是不是首个呢? 有一个观点是早期的ETH-X是采用RDMA语义, 没有内存语义? 而友商这个有内存语义, 但是最近几个月ETH-X的演进我不太清楚, 不知道这个首个的立论能否成立? 

另外至于为什么不是全球首个呢? 偷偷的告诉你, 因为博通发布了一个《Scale Up Ethernet Framework Spec》[1]. 简单的把供应商的方案拿来改改包装一下就能填补空白, 这个话题值得商榷了...

Ethernet ScaleUP这个话题, 或者更狭义的定义在Eth上进行内存语义传递, 要谈全球首个的工作应该是2021年的NetDAM以及同时期的Tesla TTPoE的工作, 当然那时候我代表思科, 所以不算全国首个? 再宽泛一些应该也包含NanoPU这样直接以太网进寄存器的工作. 针对使用标准以太网来做ScaleUP是我过去几年一直都在推进的工作, 并且一直和博通在一起做了很多讨论和深入分析.

![图片](assets/d4f8d71676d2.png)

比较好的一点是, 友商接受了通用以太网的概念, 既然要想要以太网的广泛可获得能力, 那么不改动以太网头才是真正的以太网. 其实不改动以太网头有一个更狭义的定义, 即从交换芯片的Parser逻辑出发, EthType字段的位置和长度不能修改. 当然我能够理解某些ScaleUP协议即要大规模组网能力又要兼顾传输效率同时还想要薅以太网便宜的羊毛, 于是对以太网头进行了很多修改和舍弃.

关于这个的讨论可以参考我差不多一年前的一篇文章

[《基于EthZ的以太网ScaleUP互联方案》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491597&idx=1&sn=a76b855416fe3ae614c6f4ebd5ea7bbc&scene=21#wechat_redirect)

友商定义的OEFH Header如下所示:

![图片](assets/34ee502e0ace.jpg)

我们可以再来看看博通的Spec的定义:

![图片](assets/52809dfb2c81.jpg)

原来全国首个可以这么玩,  `sed s/AIFH/OEFH/g` ? 下次我也定制一个ZBFH.  至于后面的LowLatency FEC(RS-272), LinkLevel Retry(LLR), Credit Based Flow Control(CBFC)这些, 你会发现完全是一样的.

友商定义的协议栈:

![图片](assets/c59099ad3c06.png)

再来看看博通的版本, 多一个提示Scale-Up-Endpoint = SUE. 然后相对于博通少了PFC.

![图片](assets/1c43de5109a5.png)

然后比博通方案多的一个是一些ScaleUP的RDMA语义, 然后您可以仔细想一下, 这个和IB或者RoCEv1又有什么区别呢? 感觉友商这个协议就是博通的Spec和腾讯的ETH-X的杂交体.  一年多前我就讨论过这个问题, 为什么RDMA不适合做Eth-ScaleUP

[《HotChip2024后记: 谈谈加速器互联及ScaleUP为什么不能用RDMA》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492300&idx=1&sn=8a239883c831233e7e06659ec3425ea2&scene=21#wechat_redirect)

好了, 我不装了摊牌, 有些哑谜因为涉密原因没有多说, 例如如何做msg pack只是很简单的和BRCM进行了沟通, 然后针对乱序情况下的一些Memory Model的处理, 也没沟通过. 于是针对TensorCore等DSA器件的异步内存访问相关的工作在博通的Spec里也是一片空白:) 

另外在下面这个文章打了一个哑谜, 当然被夏Core戳穿了, 其实意思就是脱离XPU本身的内存模型和微架构来谈互联是不行的, 

[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect)

![图片](assets/b5c987878419.png)

然后具体怎么做又是另一个哑谜了, 这不夏Core也出了一道题:)

![图片](assets/d5ed8ed68085.png)

最后说点感想吧, 其实当你承认了以太网的便宜和通用, 那么就不要嫌弃, 世间万物早就标记好了它的价格, 没有太多的可以投机取巧的地方.

要么不接受靠自己, 例如华为的UB. 要么就接受这个糟糠之妻,踏踏实实的过自己的小日子, 少点折腾...即要便宜又要高性能还有低延迟高可靠, 这些叠加起来其实无形的只是把复杂性转移到了其它地方...

另外, 国内差不多已经有五/六套ScaleUP方案了吧? 这么卷干嘛...

参考资料

[1] 
Scale Up Ethernet Framework Spec: *https://docs.broadcom.com/doc/scale-up-ethernet-framework*