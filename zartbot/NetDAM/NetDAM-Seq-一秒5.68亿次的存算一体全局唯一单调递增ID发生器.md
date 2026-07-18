# NetDAM-Seq:一秒5.68亿次的存算一体全局唯一单调递增ID发生器

> 作者: zartbot  
> 日期: 2021年12月28日 16:42  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487315&idx=1&sn=5e92765fed256094834eb3fb3e8653dd&chksm=f9961f91cee196871057d3b81765e5c6dd92d1b44b2d9cac1b40914adb9eebffb2712f5baf78#rd

---

这个项目是一个NetDAM的衍生项目，思科中国研发中心的David Peng团队干了几乎所有的活，而渣就是那个让开发非常讨厌的PM+测试。不知道业界最好单机成绩是多少，反正用X86快不了。我们基于一个400G的FPGA实现了一个小包线速的单机系统(4x100G,每接口142MQps，总计568MQps)~给分布式数据库行业做了一点微小的贡献，实现了一秒`0.42薇`（5.68亿)的成绩~
什么是Sequencer
在分布式系统中，全局唯一并且满足单调递增的ID产生系统(Sequencer，序列发生器)是一个非常重要的组件。例如大家都在双十一期间抢购商品，如何判断两个订单的先后次序，最简单的办法就是先到Sequencer上去取一个票，然后票号比大小

![图片](assets/c44eac7d7e7d.png)

这个组件对于整个系统的吞吐量有着至关重要的作用，通常需要满足`高可用`、`低延迟`、`高QPS`的需求，而低延迟进一步则是降低处理抖动，使得整个数据库集群业务处理量也相对稳定。

常见的系统实现方式为某些基于数据库的自增主键的方式，但是水平扩展特别困难，另一种做法就是基于Redis一类的本身单线程能够保证原子性，因此采用INCR操作，然后初始时为N台Redis构建不同的初始值然后以N为步长递增，例如有5台机器是，它们各自产生的ID如下：

```
A：1, 6, 11, 16, 21B：2, 7, 12, 17, 22C：3, 8, 13, 18, 23D：4, 9, 14, 19, 24E：5, 10, 15, 20, 25
```

但是这种方式单机的会存在严重的Contention，因此性能也不会太高，然后至于Twitter开源出来的Snowflake算法，也就一秒钟产生几十万个自增可排序的ID.

![图片](assets/20498272290e.png)

看到这里大家就会想到用RDMA**优化呀，2016 USENIX Annual Technical Conference上讲到的< Design Guidelines for High Performance RDMA Systems>[1]最高性能也就122MQps

![图片](assets/76a0e7841261.png)

NetDAM-Seq存算一体化序列生成器
NetDAM这种存算一体化架构不就非常天然的适合干这活么？
![图片](assets/2d55d9aeacad.png)
对外又是标准的UDP接口，谁都可以调用
![图片](assets/ef75ad0c44f3.png)

针对Sequencer这个场景，我们直接创建一些FPGA上的BRAM再复用原来的netDAM包处理的逻辑就可以实现。最开始是一个月前去蚂蚁金服技术交流的时候谈到的，然后回来先定了一个小目标，一个多亿每秒反正比RDMA那个ATC16快就好。结果后来硬件的团队说可以直接实现你4个100G接口线速的梦想，那么累加起来就是`0.42个薇`了~
于是我们选择了一块Xilinx VU37P的开发板，因为就它有400G的接口(估计换成新的Versal干到600G都没啥问题)，

![图片](assets/d5a3fb93a2c7.jpg)

然后通过NetDAM协议编码实现了客户端UDP发包，NetDAM-Seq回复带64bit Seq的ACK包，整个包大小都规定到64B，就这样实现了每秒5.68亿的处理能力。

一些其它的随笔
昨天朋友圈看到有一位同事把Cisco Press的书扔掉了，而我又翻了一下日历，在七年前的今日我也把所有思科出版社的书送给同事们了。

![图片](assets/2e40f979e09d.jpg)

有几分感慨，而正是这一份对过去命令行和传统路由协议**的告别，才使得渣过去几年做出了好多好玩的东西。2015年ThinCPE，基本上成了现在国内大多数SDWAN的原型系统，2018年Nimble这个分布式的OLAP系统成了思科创新大赛的头奖，2020年的Ruta已经在某个客户那里落地，明年还会迎来一个运营商的客户，而2021年的NetDAM则很快的会成为存算议题体系结构中必须要迈出的那一步。

2022会有什么好玩的事情发生呢？

#### Reference

[1]
Design Guidelines for High Performance RDMA Systems: https://www.usenix.org/conference/atc16/technical-sessions/presentation/kalia