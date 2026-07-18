# 再来谈谈ScaleUP网络

> 作者: zartbot  
> 日期: 2024年10月28日 16:20  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492581&idx=1&sn=1463a252c9df89830a60569c19a6157e&chksm=f995f327cee27a316e19678aed4ead30cf1fcdc2ab0511a9d3b1f810bdde492873f609e51e95#rd

---

今天有个同事发现一个好玩的事情, UALink里面没了BRCM的Logo

![图片](assets/5409aeb0ad3b.png)

然后晚上又和几个朋友聊了一下.... 其实一开始按摩店去找BRCM做PCIe交换机的人搞UALink就是一个错误, 也是渣B早期一直Diss UALink的地方, PCIe有太多的限制和沉重的包袱了...

走以太网的路, 我举双手双脚支持, 再加上里面基本上定制加速器的主流玩家都在了, 应该成功的概率比其它方案高一丢丢吧?

![图片](assets/bef67f8bcacf.png)

[《谈谈基于以太网的GPU Scale-UP网络》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489513&idx=1&sn=840d2d139beb6e9b40ac2a0a2b32689b&chksm=f996072bcee18e3d102d67877934f8c12b0ab1661d5b8dfc11250b22ca3c71bb267c1127e83b&scene=21#wechat_redirect)

[《HotChip2024后记: 谈谈加速器互联及ScaleUP为什么不能用RDMA》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492300&idx=1&sn=8a239883c831233e7e06659ec3425ea2&chksm=f995f20ecee27b185a42d09868bdf6cef64df38267489ee386b6d4425e57d2c2a699ada0f9cb&scene=21#wechat_redirect)

最近还听到很多公司对ScaleUP还是没想清楚, 例如GPU的NOC**上能不能把128B往上提一下到1KB? 或许渣B只能对这类人说:“请您去好好读一下《计算机体系结构:量化方法》的第二章还有附录B? ”..你以为128B到1KB就是改个MTU**那么简单么?

不要怪渣B喜欢diss人, 忠言逆耳啊, 当年那些人做DPU**的人不听渣B的话, 如今沦落到啥下场? 要是当初老老实实的抄个NetDAM, 你看看现在GPU的I/O Die的生意不就做的风生水起了么?

[《DPU新范式: 网络大坝和可编程存内计算》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486644&idx=1&sn=a2a18f661c18bfb96a37d5ac0d1a9653&chksm=f9961c76cee1956091037f97b52d420008c2d575c9ce2478ee12707c1336609882c90fae1a28&scene=21#wechat_redirect)

其实你翻开几年前NetDAM的论文, 把ScaleUP(Intra-Host)总线如何桥接到ScaleOut(Inter-Host)总线的问题分析的那么清楚

![图片](assets/abd09022de13.png)

无论是拓扑/延迟/丢包(无损)/Cache一致性/多路径/FlitSize, 都讲的清楚了. 解法也特么简单到谁都会玩, 就网卡上挂一块内存就行了

![图片](assets/fe0d080f8698.png)

这样ScaleUP和ScaleOut就统一了, UALink最终也要走到这里

![图片](assets/dd2e885cce06.png)

然后你再看看Tesla的TTPoE, DumbNIC, 大道至简, 有时候做人不要太聪明就好了...

![图片](assets/a263b56a6a39.png)

![图片](assets/7385022a8868.png)

**除了Tesla的DumbNIC, 还有某个知名大厂也准备做类似的事情了...**

至于有人谈Optical Memory I/O, 想想看当我们有了一个CPO的Switch, 然后如下图这样?

![图片](assets/66315f89e1bd.png)

这是三四年前的图了, 如今只需要把switch的带宽换成102Tbps, 内存容量换成10TB, 然后接口带宽换成800G~1.6Tbps即可. 

对大模型你会收获什么? Disaggregation-HBM,甚至你都不需要HBM, 不需要CoWoS, 一大堆GDDR7不香么?

对于CPO或者Optical NOC稍微多说几句, 对于交换机而言, 如果要做高Radix,例如做到1024个112G Serdes的102T交换机, 整个封装的问题如何解决? 一些翘曲和板上焊接的可靠性良率如何保障? 如果要Radix做到2048呢? 其实很多事情不是单纯的光或者铜 , 多从其它角度考虑一下物理约束不行么?

至于前面一篇谈到NV 5万亿的市值, 说实话看到这个价位的大概有前几个月的但斌总, 当然还有老黄了, 毕竟最近一段时间高薪快速扩张了很多人, 再加上Blackwell订单的预期, 不做市值管理到这个数到时候就是万丈深渊了.

至于渣B反正看不懂也懒得预测, 不过我们还得警惕几个事情, 大选和日元套利交易**的终结等, 很有可能这些都会出一些黑天鹅的事件...不多说了,点到为止...