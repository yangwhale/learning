# Blackwell TensorCore架构

> 作者: zartbot  
> 日期: 2025年1月25日 16:00  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493056&idx=1&sn=1c6025f97df16a3b9576746b7944538e&chksm=f995f502cee27c145677e91761ebec0d59cde82d4562c463070f2c72efb3751567bac934c266#rd

---

昨天写了一些关于Blackwell内存子系统的微架构分析

[《一些关于Blackwell微架构的分析》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493041&idx=1&sn=97009afc89e8a031d7bfd63a2cc9a56f&scene=21#wechat_redirect)

今天继续谈谈TensorCore的变化, 在之前的文章中,我们介绍过前几代的TensorCore架构

[《Tensor-003 TensorCore架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491424&idx=1&sn=0fc2110931b27714900e78d73b11a5b5&scene=21#wechat_redirect)

第一代到第四代TensorCore的数据访问路径都是来自于SMEM和RF, 在Blackwell这一代增加了专用的TensorMemory. 它是一个2D的memory寻址架构,每个CTA包含512列和128航, 每个Cell是32bit. 地址采用32bits Lane<31:16> Column<15:0>的方式. 但是只有A和D矩阵可以放TMEM.

![图片](assets/393e73aa4da2.png)

然后继续使用了Hopper中的WGMMA(SM_90a--> SM_100a)的方式, 并且WarpGroup内分别限制访问不同的lane

![图片](assets/d5b4b8022633.png)

总体来看新的Tensor Memory为每个TensorCore增加了2KB x 32 64KB的内存, 同时SMEM的压力也会小一些. 然后TensorMemory增加了一些内存管理指令, 并且有一些约束.首先它只能在CTA内单个warp进行allocate. 然后分配的单位是32列, 并且当一个列被分配后所有的128lane都会被分配. 看上去内存分配还是不很灵活的.

然后LD/ST指令的memory描述符做了一些定义, 相当于原来LDMATRIX指令的一个针对TMEM的扩展

![图片](assets/344495d19fb2.png)

MMA操作增加了一个Zero-Column Mask描述符, 这样在做矩阵乘法的时候针对0较多的矩阵还是可以节省不少计算消耗的.

然后增加了CTA pair的支持.

![图片](assets/5d457e34ab28.png)

新的第五代TensorCore的操作符如下所示:

![图片](assets/20f0854beb02.png)

其中增加了Shift操作,对卷积的支持应该也方便些.

异步操作中可以构建一系列的流水线

![图片](assets/9d3dc8bf512f.png)

针对FP4和FP6的更新, 可以将其Decompression成8bit value.TMEM copy指令支持精度decompression和multicast的能力, 这对矩阵乘法还是有点好处的.

```
tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;.cta_group = { .cta_group::1, .cta_group::2 }.src_fmt = { .b6x16_p32 , .b4x16_p64 }.dst_fmt = { .b8x16 }.shape = { .128x256b, .4x256b, .128x128b, .64x128b**, .32x128b*** }.multicast = { .warpx2::02_13** , .warpx2::01_23**, .warpx4*** }
```

MMA指令中, 操作数内存的定义如下, 以D=A * B +D为例

A为一个MxK矩阵, 可以存放在TMEM也可以在SMEM

B是一个KxN矩阵, 可以在当前CTA的SMEM中, 也可以在Peer CTA的SMEM中

D矩阵为MxN, 存放在TMEM中.

然后DeepSeek-v3中有一些关于未来硬件的量化和Transpose的需求, 在第五代TensorCore中增加了block-scale的能力即

(A * scale_A) * (B * scale_B) + D

![图片](assets/a5ecd311114d.png)

同时因为TMEM为一个2D结构, Transpose似乎也容易一些了

![图片](assets/4a6749dd3209.png)

然后TensorCore内有一个special的collector buffer, 对于未改变的矩阵仅加载一次, 指令上采用activation stationary和weight stationary定义.

![图片](assets/6266f676bf5a.png)

对于Sparse矩阵乘法, 或许这也是一个可以挖掘的点. FP32支持1:2  FP16/FP8支持2:4, 然后FP4支持4:8, 这样就有很多灵活性了

![图片](assets/60c212070ddc.png)

是否在Attention或者MoE中, 通过一个激活函数故意构造一个4:8的表示? 可能是一个潜在的模型架构和算力协同的优化方向.

总体来看, TensorCore在Blackwell上改动还是蛮期待的, 至少Infra上有很多可以尝试的东西了, 但是到真正用到模型上需要多久呢? Hopper这一代最大的问题就是没有家用的卡支持导致TMA和TensorCore的很多特性没法被普通玩家一起来玩. 到了一代的末期才逐渐被DeepSeek把FP8用起来,  期待RTX50xx/GB10这样的平台可以给更多的开发者调试和优化GEMM kernel的空间.

当然, 读PTX的文档还发现一些小功能很有趣, 涉密就不多说了, 嘻嘻嘻嘻~~~