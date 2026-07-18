# NetDAM-CXL？

> 作者: zartbot  
> 日期: 2022年2月13日 16:05  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487470&idx=1&sn=d89672c9cdcab05bbbfd0702f70b1b4e&chksm=f9961f2ccee1963a64c12fa8b313077678f111106cccb5b07ced369cebabf3d8d9222e8f36cd#rd

---

夏老师的文章一直很好玩, 居然渣这种还能被翻牌... 

https://zhuanlan.zhihu.com/p/466870704

NetDAM一开始就考虑用CXL的，还有P4 Switch做MMU，**只不过一个悲惨的故事是樱桃不给NDA，SPR的CPU也不给，**Xilinx挺好的给CXL，但是苏妈家的CPU还是要等呀...

樱桃为啥就盯着Barefoot的生意忘了CPU和GPU的大业了呢？和Cisco和BRCM一起把NetDAM搞起来，然后再一起去看看卖螺丝在400G上是否支持RDMA而带来骑虎难下不是很好玩么？

》[探索400Gbps主机网络](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487010&idx=1&sn=81b9d199299b1f98ec4934ef98879da0&chksm=f9961ee0cee197f6aefddb5d9da54b1367be8423984f3c267f613e38c8181a048d7c98dc9b86&scene=21#wechat_redirect)《

哎，我一直在把一个主板的图放了很多遍了，就是很多人不懂...还是夏老师看到明白讲的清楚....感觉以后要听妈妈的话，不跟差生玩了...

![图片](assets/b3c3086a9475.png)

另外说一句话，Intel的这次红利可以使得大家很容易做成一个IBM z15那样的大型机，这也是我前几天发的某个文想说的

》[**金融的计算**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487416&idx=1&sn=ac3aa0ba4e5bd68d0bd3092cc623e8ce&chksm=f9961f7acee1966cd29f07d6abf657d032cb58dc7f08456f28de80fb3ea85796207e04856942&scene=21#wechat_redirect)《

[》**从大机的视角看DPU发展**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486147&idx=1&sn=e01ed29dc0831bcb3ab2234b3ff3ebaf&chksm=f9961a01cee193174892676e81ea7eba96221c8a320137f4b246f1d484e770bebed420db8cf9&scene=21#wechat_redirect)《

对于CC有啥用？十几年前思科设计QFP的时候连L2-DCache都不要了。然后仔细看前几天引用的SplitKernel的操作系统论文,网络连接，无缓存一致性:

![图片](assets/2cb81e06c57e.jpg)

但是问题又来了，DMA本身不得不考虑一致性，以致于RDMA不得不考虑一致性，所以导致延迟，导致jitter，导致incast，还要HPCC、Swift去测延迟搞另一个CC(Congestion Control).NetDAM解耦两端固定延迟，jitter ns级不香么？

![图片](assets/dc0234689b45.png)

延迟也600ns左右，比RoCEv2 800ns快多少自己数~而且基于SRAM的NetDAM还可以进一步将延迟降到430ns，抖动降到几乎为0(Fixed pipeline不需要CC），有这样的网络还搞什么Swift，随便把队列深度带到回复的ACK里，直接当OQ用了...有了这个东西，注意下图的三和四：

![图片](assets/1fdfa86383e6.png)

至于夏老师提到的transactional memory在其它ACC上，NetDAM本身是指令集和数据混合放在数据包中的本质上就是想让NetDAM做一个subsystem集成在他们上面，例如上图的第三个和第四个Case, 

而可能画图有个关于memif的误解，其实针对Transactional memory的场景或者所谓的I/O密集的场景走的下图的RQ和CQ，而它们本质上就是CXL承载的，而下面的memif只是处理普通以太网报文，毕竟在云和超算的场景中网卡还是要处理一些以太网通信的,特别那些乱七八糟的legacy协议,比如TCP.....hahaha。。。总归要考虑上层生态的问题嘛，例如那些Java和golang的程序员搞云原生的....

![图片](assets/66fa4f77122a.png)

另外而netDAM上面的ALU还可以干很多好玩的事情，数据裁剪这些，矩阵转置这些，甚至直接把三星的PIM封装上去，香么？所以本质上是借助CXL在主机侧和以太网底层之间构建一个内存Shim layer，

记得去年还有人提CXL over Ethernet，我还专门发了一文,把自己陷入CC的坑里了吧.

>[RDMA、CXL和以太网](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486515&idx=1&sn=bfeda0ba01c6206b9112283b247f8032&chksm=f9961cf1cee195e7d2eee6a85030837ff46179ba5adc380e49be636adb279873f3a22a619153&scene=21#wechat_redirect)<

但是仔细想就是多了一个内存的封装层，这样操作系统就不需要几千外行代码的driver了呀：）所以最后就变成了这样一个好玩的东西,

![图片](assets/00339aabce40.png)

提到Driver，夏老师讲得对就差一个操作系统了，这也是NetDAM-Seq实现了分布式的锁的原因，你仔细想想，无论是SpinLock还是Semaphore或者RWLock都可以通过上面的ALU+SRAM很容易的实现，可以做到600Mpps，延迟嘛只比跨NUMA高一些，但是请注意一个文章FFWD(libffwd)

![图片](assets/93e1cfc2484a.png)

主机内CXL，主机间Ethernet，锁在NetDAM上做delegation不香么？至于一个真正的分布式操作系统， 内存子系统有了，同步的锁有了，调度器也容易，文件存储也好办，memif把网络协议栈也清理干净了。完工~

![图片](assets/07dcb735bfcb.jpg)

而相对于RDMA，除了400Gps NIC的问题外，还有一个问题是Segment Routing，解决拥塞最简单的办法是选路。另外超算中对于不同的计算业务拓扑需求是不同的， 大数据一类的数据库业务树状就可以了，深度学习这些矩阵类的Torus是标配，例如Google TPU、还有日本的 Fukagu 6D-Torus,而针对一些流体力学计算 FFT这类的， Butterfly的拓扑更优，所以Ruta+netDAM一起食用才更舒服~

对，这样道出了渣最近在继续研究的一件事情，如何通过公有云实现Exascale的超算。最近在看一篇关于美国Exascale HPC的论文[1], Slingshot基本上是正在建设的三大Eflops超算的根基,第一页讲的很精彩

![图片](assets/5f65760f69a8.png)

但是点到为止了~~ 喵喵~~

至于NetDAM,我觉得如陈老所说，运气罢了。只是恰当的时间，恰当的需求赶上了，DMA本身有问题，CXL的放开，云上Exascale超算，以及数据库讲存算分离而计算又说存算一体。赶上了一个七十年一遇的好运气罢了。

![图片](assets/13e3ed14b98d.jpg)

#### Reference

[1]
An In-Depth Analysis of the Slingshot Interconnect: https://arxiv.org/pdf/2008.08886.pdf