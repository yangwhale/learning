# 谈一谈ASPLOS'24 google的CC-NIC

> 作者: zartbot  
> 日期: 2024年5月13日 12:12  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489928&idx=1&sn=3f33add23dd31fd7b260518f1201b231&chksm=f996094acee1805c7f2c4b3eb78384918d36e35dab0ea4985281a5cb51be2096a807fad66db7#rd

---

最近看到一篇ASPLOS'24的论文，来自Google的工作《CC-NIC: a Cache-Coherent Interface to the NIC》[1] 勾起了一些当时做NetDAM的回忆，解决的问题和方法让我想起几年前夏晶老师的一段话《聊一聊CXL》[2]如下

表面上，你看完论文可以说NetDAM不就是网卡加个独立内存么？实际上啊

重剑无锋，大巧不工！！

我从我的视角交叉佐证一下它存在的逻辑

服务器CPU的内存是DIMM，为什么有DIMM? 实际上是CPU需要的DRAM silicon size太大平摊放不下，所以采用了Z轴折叠的方式，用3D空间换容量来的。但是NIC这东西，需要的是独立、小容量的DRAM，用宝贵的空间换来的DIMM来做NIC的DRAM，天然就是亏的。只要能独立定义的DRAM诉求，最好就不要和DIMM合一。你喜欢喝鸡汤，他喜欢吃牛肉，如果目标都清晰了，何必乱炖在一起?

CPU的内存通道数量最多就是8通道了，这个不是不能做更多（也有12通道的CPU存在），而是当DDR通道超过8通道之后，无论是基板、PCB的代价就急转而上，走入一种成本你几乎无法承受的空间。如果按DDR5 5200速率算，8通道DDR带宽是5.2*8*8=330GB。而一个400Gbps的NIC，其实按我经验的算法，外部带宽折合到DDR带宽是有一个1.5X放大倍数的，简单来说要跑满RX和TX双向，需要400*2*1.5=150GB，NIC吃一半，CPU还玩个屁。这里，如果放弃DIMM用贴片DRAM，在NIC card上满足小容量150GB这个诉求是容易达成的。

很多人没认真算一下400Gbps对DDR访问意味着什么。光看TX，意味着每1ns要均匀获得～512bit数据，如果按DDR访问延迟200ns及访问粒度64B，需要200 outstanding request，并且能均匀地应答响应。这是DDR难以做到的。

其实要解决400Gbps NIC问题，就只有两条路：

NIC与CPU进一步紧耦合，直接基于TB带宽级别的cache交互数据。

NIC与CPU进一步分离，各自有独立的内存空间。

NetDAM选择了第二条路，而这条路上，如果要让操作系统或者说系统协议栈能够直接操控NIC内存，最佳的搭配就是CXL。

PS：第一个开放DOCA_malloc库的DPU公司也许将成为最终的获胜者。

而这篇论文虽然是说CC-Interface，实际上算是一个Mem-NIC的库，只是樱桃还不争气，最终这个工作只能通过UPI模拟，CXL长路漫漫。

但考虑到GPU Scale-Out网络吞吐越来越高的时候，PCIe带来的局限性还是值得好好来看一下这个问题的，特别是国内一些想用PCIe做Scale-UP的GPU厂商。

文章第二章节详细的讨论了PCIe的延迟和性能，2.1讲述了当前网卡的结构，包括Pkt Buffer/Descriptor/Head-Tail ptr，然后2.2讲了PCIe的互联延迟，MMIO的延迟和吞吐。

![图片](assets/1ce11013c42f.png)

![图片](assets/2f2c6400fc8d.png)

然后总结了三个问题：

constraints on the host-NIC interface.We identify three issues:

Since PCIe is not a coherent interconnect, local data structure updates must be communicated or signaled with explicit PCIe transactions.

PCIe operations incur high latency, so reducing the number of interconnect traversals is critical to achieving lowlatency packet transmissions (§2.2).

Data and metadata writes over PCIe are expensive for the CPU in terms of both throughput and high-latency stalls

结论是NIC要改，然后搞了CC-NIC

![图片](assets/e7ca7238564e.png)

API如下:

```
int ccnic_buf_alloc(struct ccnic_pool *pool,    struct ccnic_buf **bufs, unsigned count);void ccnic_buf_free(struct ccnic_pool *pool,    struct ccnic_buf **bufs, unsigned count);int ccnic_tx_burst(int txq_index,    struct ccnic_buf **bufs, unsigned count);int ccnic_rx_burst(int rxq_index,    struct ccnic_buf **bufs, unsigned count);
```

实际上在DPDK中已经有思科贡献的现成的Memif PMD结构和相应的Metadata结构，这个在设计NetDAM的时候就直接复用了

![图片](assets/2fcfcccf78cb.png)

樱桃这个样子，CXL一直没啥进展，直到AI时代自己的CPU地位都不保了。当下考虑到国产GPU的ScaleUP连接，是否会会心一笑呢？SHARP这些随路计算的功能，几年前都做好了。

参考资料

[1]
CC-NIC: a Cache-Coherent Interface to the NIC: https://dl.acm.org/doi/10.1145/3617232.3624868
[2]
聊一聊CXL: https://zhuanlan.zhihu.com/p/466870704