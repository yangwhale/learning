# 高带宽处理器, 又在基于PNM卷访存密集型算子?

> 作者: zartbot  
> 日期: 2025年4月26日 11:58  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494058&idx=1&sn=a840110f4f9ec40e031f680413572afe&chksm=f995f968cee2707ec42bf0356cafbe5960bde8b876d51d30aecf8d115768aadb56cb91cc850a#rd

---

说实话对于内存厂商对于NV拿着HBM**高溢价一直都有些不爽, 总是要捣腾一些事, 最近推理越来越重要了, 于是又把Processing Near Memory(PNM)的概念重新包装了一下搞了一个《HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing》[1], 然后通过GPU+HPU**的异构来解决一些问题.

好像去年的HotChip他们就在搞AIM

![图片](assets/2440d83d8b52.png)

以及如何做MHA**

![图片](assets/76d073fa1119.png)

论文很简单, 针对推理中的访存密集型算子**做Offload, 其实这个图还没画全, 再把MoE加上去可能更好?

![图片](assets/c54babb2e13e.png)

主要的工作就是把以前的PNM/AIM迭代了一下, 然后增加了一些DSA, 通过PCIe和GPU连一起

![图片](assets/59f2f286cb16.png)

然后整个HPU的规格如下, 无论是Prototype的规格, 或者是Production的HPU, 感觉很难的一点是既然HBM都用了,为啥不做大点直接变成一颗GPU呢?

![图片](assets/7b6b72ee6000.png)

反正封装要支持4900GB/s的带宽和144GB容量的HBM, 总归要一个很大的计算Die吧?或者说一个很扁的Die能省多少成本呢?  Prototype用了Xilinx的Alveo U55C, 把Attn block Offload了.

![图片](assets/8ea761e0e1f3.png)

我记得当时我做NetDAM的时候, Xilinx给我提供了一个U50N的版本(似乎没对外卖过...), 2x200GE, 也是带460GB/s带宽的HBM. 55C是一个计算卡,然后HBM容量增加了?  而最新的一些Altera的FPGA 似乎还支持800G+1TB/s带宽的HBM. 其实如果能够广泛低成本获得, 通过它们来做Attn和EP并行似乎也是一个不错的选择呢? 然后看算力规划似乎也就39TFLOPS@FP16, 想不出SK Hynix做这个的意义? 从一些数据上来看, 这种方式好像还是有一些受益的, 但是BatchSize太大的情况又要受到TPS-SLO的约束.

![图片](assets/620cf81c9994.png)

其实这是一笔成本账... 买额外4块HPU配一个L40s**是否划算的问题... 或者说干脆买一块L40s配一堆国产GPU做HPU是否可行呢? 没算过...

参考资料

[1] 
HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing: *https://arxiv.org/abs/2504.16112*