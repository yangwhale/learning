# 推荐一本《Streaming Systems》的书

> 作者: zartbot  
> 日期: 2024年11月3日 16:43  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492608&idx=1&sn=4e0ac2d9a7271210bd4508e4965447ff&chksm=f995f4c2cee27dd402c7af8c0480e075b2164b56612f47c60929fecbc8edb47fe1547b555213#rd

---

最近有一本书《Streaming System》由Flink团队的陈守元老师翻译成中文了, 英文版前面好几年渣B都一直在推荐, 而对于如今ScaleUP上的一些争议,这本书也有很大的参考价值.

渣B是在2018年的时候给Cisco构建一个分布式的数据平台Nimble时, 对于大量的流数据分析例如各种交换机的Hardware Telemetry, Yang Telemetry, 路由器的Netflow/IPFIX**和防火墙的日志等... 对于这些无穷无尽的数据流, 通常需要对其进行某一列的数据在某一个时间段构成一个向量进行分析和特征抽取, 并打分评估网络服务质量或者是否有安全违规. 正好最近在Linkedin上看到下面这图, 引起了共鸣.

![图片](assets/4b130ae31d09.png)

当时就是一个很偶然的机会搜索到了该书作者的streaming processing 101这样的blog, 然后就读了这本书. 然后也了解到了Flink这样的项目, 只是当时在Cisco的很多嵌入式平台上是ARM架构**, Flink的java runtime相对较重, 而且那个时候我记得Flink SQL也没有, datatype也有一定的局限性, 于是参考《Streaming Systems》这本书自己构建了一个基于Golang的系统Nimble.

对于流式数据的处理, 特别是Window如何构建, 例如TumblingWindow或者Slide Window或者SessionWindow, 以及Event如何Trigger, 按照WaterMark或者ProcessingTime或者基于SessionWindow的onMerge的处理.  Evictor如何设计等内容...当时一边阅读一边设计Nimble的草稿纸...

![图片](assets/4ea8b44e27ef.jpg)

最后这套系统在一些嵌入式平台上也可以很好的支持1M records /s的处理能力, 然后里面还串入了基于scikit-learn决策树相关算法的算子和基于Tensorflow的推理算子, 最后完成了一套能够在网络边缘进行实时数据分析聚合和推理的AI Infra, 最后拿了Cisco的CEO大奖. 后来这套流式数据处理引擎也给了以色列的数据平台团队整合到了Cisco园区网**的DNA-Center中.

![图片](assets/664ab7cf5f7c.png)

![图片](assets/d1a3b89cd9f5.png)

后来在设计NetDAM的时候, 很大程度上也是受Streaming systems的影响. 从I/O节点来看, LD/ST是源源不断的Streaming, 本质上ScaleUP网络的问题, 其实是殊途同归的. Streaming 然后搞个窗口, 再DMA**到远端, 然后恢复成Streaming, 然后这个窗口又很好的解决了网络拥塞的问题,  正如几年前的一篇文章

[《香农和图灵的边界：溯源DPU的价值》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486987&idx=1&sn=e4ea1c3ddae5b59fd586d3aca5b0f536&chksm=f9961ec9cee197dfd6fd921479c35e014d0ee0df08ff8f6ef09bde99770eeabf99d68128af86&scene=21#wechat_redirect)

网络的本质是承载数据流，而内存是数据流在某个时刻的快照，而计算是基于快照信息而产生新的数据流。

所以只需要在I/O节点上加一块内存, 这个数据流就跑起来了,  其实渣B也不太明白,为什么如此简单的一个问题会在工业界产生那么大的争议. 再看看TTPoE的实现, 不也是在践行这个streaming systems的逻辑么?

最后绕回来, 推荐一下这本书, 顺便贴一下军华老师的推荐

![图片](assets/3660205f1e82.jpg)