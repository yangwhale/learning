# 关于Blackwell架构的一些更正

> 作者: zartbot  
> 日期: 2025年1月27日 15:24  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493075&idx=1&sn=ad556024e6f763e83327a45d121b9d02&chksm=f995f511cee27c07561d7ac2c3c1b8312319cd66885735f17327dae270a95b43b440387303ed#rd

---

仔细看了一下, 整个B200应该是每个Die有80个SM, 两个Die一起160个SM

![图片](assets/8f04efdb79f9.jpg)

具体标注如下

![图片](assets/cf3b70a4c74c.png)

也就是说对比Hopper 144个SM, L2Cache的量翻倍了(60MB->126MB). Hopper DieSize为814mm^2, 我原以为Blackwell单个Die也差不多800mm^2,N4P工艺比Hopper的N4晶体管密度增加就6%左右, 应该SM数量和Hopper差不多呀?看样子单个Die来对比,L2Cache是差不多的63MB(Blackwell) vs 60MB(Hopper), 但是SM少了很多, 当然TensorCore Gen5对于每个SM增加了256KB的Tensor Memory占用的diesize应该还是很大的. 然后TensorCore本身的占用的面积也应该加大了很多, 使得整个SM相对于Hopper大了不少. 所以单个800mm^2的die上只有80个SM? 对比Hopper的die, 每个SM小了很多.

![图片](assets/3688d0004b5e.jpg)

当然为什么RTX 5090还有192个SM,而且Diesize只有750mm^2 ?仔细看了一下, 5090的L1Cache/SMEM只有100K,为B200 SM的一半. 另一方面MCM两个Die互联,对于L2和NOC的开销还是很大的. L2D多端口读写带来的die面积占用也会大不少.

Tensor Memory的加入和L2Cache相对于每个SM翻倍, 总体来看可能对真实训练过程中的MFU提升还是有很大帮助的. 当然TensorCore中也增加了很多复杂的功能, 特别是既要兼顾已有的FP32/FP16, 又要在FP8和FP6/FP4上干活, 还要针对FP4做一些ScaleFactor, 难度确实很大. 但这样做单个SM的面积来看, CUDA Core占比已经很小了...

另外,晚上想到一个好玩的算法. 一本数学书, 要是没有了显然/易证如下所示:

![图片](assets/345d3f8878f8.png)

那么是否在训练一些Math相关的任务时, 模仿BERT故意Mask掉其中一些步骤会有一些什么样的好处呢? 或者通过模型自生成的把中间的易证这些自动补全呢? 只是临时的一个脑洞而已...