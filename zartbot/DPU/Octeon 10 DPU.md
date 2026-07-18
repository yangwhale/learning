# Octeon 10 DPU

> 作者: zartbot  
> 日期: 2021年6月29日 17:09  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486016&idx=1&sn=9743e363cbe5b345251a80fc4741dd24&chksm=f9961a82cee19394d549be2943aaccb75e281fc20a60b7b70d888b2c7c5abd294f88c6079395#rd

---

❝
大概有这个DPU的NDA快一年半了,现在Marvell自己都写新闻稿了,那么就来说说对它的一些评价吧，国内有些要入局SmartNIC或许从它开始是一个不错的选择，毕竟Pensando和Fungible都是雷声大雨点小..
❞
当然别人家的NDA不该说的我还是不会说的,顺手吐个槽，总体来说Marvell、Xilinx SDNet、nVidia Morpheus给NDA都很快, 但是Tofino樱桃控制的好严格嘛...到现在都不给我，搞得Ruta都不想支持Tofino了，本来想构造一个很好的开源生态的帮各家都带点货的，等了你们两个月了...

Octeon 10其实一开始根本就没有DPU的概念，但是就个人感觉而言，它有可能是一个非常出色的NPU，首先各种性能中规中矩的400Gbps平台，on-chip的AI推理也有，存储方面也可以到20M IOPS，加解密400Gbps,只是因为DPU的概念包装而成的...

![图片](assets/3a098d2385b6.png)

背后的架构来自于Arm最新的Neoverse 2 处理器

![图片](assets/7e4b49bb2440.png)

![图片](assets/acc49e83cb09.png)

整个芯片采用TSMC 5nm工艺，同时散热上采用了fanless设计，这样再很多场景下都会非常有用

![图片](assets/4ba40b0e764d.png)

另外一个我个人非常期待的功能就是整合的AI推理引擎，例如我在思科做的AIOps项目[Nimble Engine](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484301&idx=1&sn=1d304267d9322d4dd59bafe6827b059b&scene=21#wechat_redirect) 整个计算系统对AI推理的要求非常高， 首个阶段在边缘尝试过RK3399pro，也尝试过黄教主的Jetson Xavier和Jetson Nano, 内置一个INT8、FP16的推理引擎还是很不错的，只是Marvell自己估计想不出什么应用场景，也就随便列了几个，还是要我们这些甲方的人来处理...当然黄教主的Morphus也是：）

![图片](assets/06f434d7cb79.png)

回到整个架构上，最值得注意的是ARM的CoreLink CI-700、NI-700对于核间设备访问的优化,同时Marvell也自己做了很多优化，我就不多说了。然后还有一个非常不错的事情就是DPDK ARM相关的maintainer就是Marvell，而且整个Octeon 10 对VPP的支持花了蛮多力气的

![图片](assets/c100f0362dcf.png)

但是整个布局上对于16x 56G的serdes 8xPCIe 5.0Controller都是非常不错的选择， 功耗也控制的非常好。

![图片](assets/f9870b85612f.png)

然后至于内存，虽然没有HBM，但是最高端的支持12个DDR5，总体来说带宽完全够用了

![图片](assets/69c127d529c2.png)

很期待今年下半年拿到这款样卡后把Ruta跑到它上面~

当然按照传统的思维，有人会说，Cavium这些传统的网络处理器早就被X86蚕食光了，但是Neoverse 2的片上网络真的强，功耗也能满足SmartNIC需求，开发周期也短，很多操作RTC可以完成性能也够，也能实现智能网卡的最智障的需求，为什么不选它呢？如果说不好的可能是上一代不支持DPDK的老的SDK，现在DPDK跑的可香了，最近就在拿上一代的CN96先跑着~

另一方面樱桃和Xilinx也需要多考虑一下生态了，如何把P4、SDNet这些东西让软件工程师能够更快的上手才是最关键的事情，点到为止吧...