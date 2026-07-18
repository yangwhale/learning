# RDMA、CXL和以太网

> 作者: zartbot  
> 日期: 2021年9月28日 16:21  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486515&idx=1&sn=bfeda0ba01c6206b9112283b247f8032&chksm=f9961cf1cee195e7d2eee6a85030837ff46179ba5adc380e49be636adb279873f3a22a619153#rd

---

RDMA over Ethernet成为一种解决方案后， CXL over Ethernet看上去是一个很straight-forward的方案为什么就错了呢？ 原因就在RoCE本身就是错误的,而错误的本源来自于PFC的实现，死锁倒是小事，overlay怎么做？ RoCE over VPC或者任何Lossy over VPC技术怎么实现？

计算机技术上很多问题都可以添加Overlay解决，每次谈到这个都会想起下面这个图
![图片](assets/296c7301bb0f.jpg)

但是Overlay太多也是问题，至于有个号称以PCIe为中心的公司，想想苹果为啥要做UMA吧?

![图片](assets/0c2d4b0781b5.png)

另一个问题是为什么我一定要建议厂商用Arm N2，本质在Cache上：

![图片](assets/7aa333c77f09.png)

另一方面是在片上网络:
![图片](assets/fd7434d033e4.png)

而对于IBM，为什么System Controller没了？这个SC L4Cache节点和Fungible所述的DPU位置几乎重合:
![图片](assets/7efc2c8d9cae.png)

![图片](assets/e46bb06f3745.png)

反而是在Cache上做文章？
![图片](assets/5d9201bca5d9.png)

Memory Pooling的使用场景是什么? 做DPU的有几个去好好看过MPI的? 下面这个系列可以帮你上手：[Zartbot MPI教程](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=1840824582203572225#wechat_redirect)

CXL over Ethernet不成功的关键在于延迟，延迟、容量、带宽三者不可兼得。
![图片](assets/2c3f409e065b.png)

所以在体系结构上的Trade-Off 不光是硬件的艺术， 也是软件的艺术。内存要同时获得容量和带宽，只能拉远池化，但是拉远池化后延迟谁都搞不定，而Coherence又会进一步劣化，唯一的取舍就很明显了。

CXL适合于什么？在一个Rack内的多处理器(不仅是CPU还包含XPU)的互联，本质上和IBM Drawer间互联一样，距离受限的:
![图片](assets/c5154cf7d537.png)

最终DPU还是要做Socket的TurnKey交付:
![图片](assets/ccf449e7c40c.png)

而针对内存，OMI可能是一种趋势
![图片](assets/a9edd4004fdc.png)

CXL over Ethernet的本质是内存串行化:
![图片](assets/ea40138caca6.png)

At the end of the day

![图片](assets/50a200c7132f.jpg)