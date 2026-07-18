# 谈谈AI的软硬件交付界面

> 作者: zartbot  
> 日期: 2024年8月20日 10:08  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491671&idx=1&sn=ccc0722adb6a975cbe868d3de9c68493&chksm=f995f095cee279833ed9111d80c7498328fc06b462527373b9f51e1e05955ea89403346fabc3#rd

---

### TL;DR

想谈谈这个话题有几个原因:

#### 0.1 知识屏障

大多数做并行计算开发的业务方都是一些数学/物理/生物统计/材料等专业毕业的算法博士, 他们对数学算法本身有很深的了解,但是对于GPU的微架构了解相对较少, 并且代码能力相对较弱. 当然也有极少数OI竞赛和数学/物理竞赛多修的选手, 但毕竟是少数. 然后能够从代码到芯片架构相关的设计就更少了. 最近几年工作和学习的一些经历搬了不少的砖.

[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)这个专题在从数学上探讨模型架构和算法的演进

[《GPU架构演化分析》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=2538479717163761664&scene=173#wechat_redirect) 这个专题从1980年开始,分析了整个GPU架构演化的过程.

[《Tensor计算》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3557619493198151684&scene=173#wechat_redirect) 这个专题分析了CUDA相关的张量并行计算开发

[《AI加速器互联》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3596388845010403333#wechat_redirect)这个专题分析加速器ScaleUP和ScaleOut互联相关的讨论

[《云基础设施》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3289258526057463810&scene=173#wechat_redirect)这个专题从云计算的视角来看待AI基础设施, 有一些基于金融流动性视角来看待云计算弹性和算力证券化相关的话题.

本质上交付界面还是要回到**从算力上如何调优适配算法, 算法上如何改造最大化算力, 从互联上如何扩展算力,**这个问题值得探讨.

#### 0.2 异构计算的同构表达

这是一个很拗口的一个词, 异构代表了专用, 而另一方面对于人类思维编程方式的抽象, 我们也期望一套专用系统**能够有更好的泛化能力去支撑更多的应用, 本质上我们是在异构器件上去追求某种意义上的同构表达. 前段时间看了一下Jim Keller在61DAC Keynote[1], 再加上今年HotChips**上各种AI加速器百花齐放,国产化GPU层出不穷的时代, 讨论一下这个话题更有意义.

### 1.  The Good Old days

Jim Keller讲到, 过去**软件和硬件之间并不需要太多的交流**, 有一套标准的指令集和内存交付界面即可.
![图片](assets/9dde1e6a23df.png)

有了这个标准的交付界面, 软件工程师可以尽情的发挥优化算法, 硬件工程师也可以不断的优化. 但是从标量时代进入到向量时代, SIMD自动向量化这条路似乎一直做的不好. Jim Keller在这个Keynote中谈到** SIMT is a genius abstraction**. 确实也是, 它通过大量的线程,把向量化的操作通过多个Thread转换成了人类心智可以接受的标量的编程.

### 2. AI时代的问题

随着计算规模的扩大, 如今的深度学习已经需要大量的张量运算, 而CUDA**在张量时代也出现了一系列抽象困难的难题.

#### 2.1 指令集扩展不可行

而国内还有很多人片面的认为, 从标量指令集扩展到向量指令集,然后扩展到张量指令集来构建通用深度学习指令集. 是不是搞个TensorCore,弄点MMA的指令集就好了?  但事实上这样的泛化路径走不下去了. 一方面是矩阵计算本身的计算访存比需要有大量的Data Locality的处理. 因此在数据搬运上需要特殊的流程, 并对算子进行时空切分以及大量在算法上的时空折中.

![图片](assets/a5256d500a6a.png)

本质的问题还是Cache策略上的问题, 标量的预取分支预测非常成熟了, 而SIMT伴随着更大的寄存器文件和Warp**调度隐藏延迟似乎也能把吞吐率打上去. 而到了张量时代, 矩阵乘法和TensorCore相关的访存冲突, 寄存器/SMEM重用,以及后续Element-Wise的操作, 例如Softmax/LayerNorm等算子融合.这些问题都涉及到GPU微架构的取舍.

#### 2.2 标准算子作为交付界面

那么是否能够拿一些标准的算子作为交付界面呢? 对于国产GPU的生态, 很多专家又开始呼吁要去做国产平台的算子库. 特别是很多人片面的认为大模型架构基于Transformer算子已经收敛的这个错误认知上. 从算法的角度的来看, Transformer确实是非常优秀的, 其效率非常高同时榨干了GPU的计算和访存, 伴随着Flash-Attention这类的结合硬件的算子优化几乎做到了极致. `CUDA的壁垒其实变向的成为了另一种“人工”智能, 基本上写一个算子的Kernel大概耗时两三周. `

另一方面,  从代数的角度来看, 模型的算法并没有收敛. Transformer的效率问题还需要解决. 然后从范畴论的角度来看,Transformer的本质其实是在构建一系列态射并构成预层范畴, 那么是否可以最终构成一个稀疏表示来降低计算复杂度呢? MoE其实某种意义上来说是在做这方面的尝试,但是还有很多问题待解决.

那么这条路可能也是在快速变化中, 交付界面依旧无法确定.

#### 2.3 互联架构和模型并行策略的变化

从局部的GEMM访存优化,再扩展一级出来自然就到了多个芯片协同的分布式集群训练. 无论是DP/TP/CP并行策略的本质还是矩阵拆分, 数据的维度/模型矩阵的维度或者是SeqLength的维度, 本质上是相通的. 只是在处理Forward和backward上有一些其它的通信和计算的Overlap策略罢了.
这些灵活的策略又要跨越两个不同的网络(ScaleUP NvLink)和(ScaleOut RDMA),可惜这两套网络都存在一系列自身的缺陷.

#### 2.4 从系统的视角: 金字塔结构

Jim Keller从指令数的视角来看待这个问题非常有意义

![图片](assets/150bfa4ca353.png)

本质上回到了一个更加广泛的问题: 如何在分布式系统中调度算子派发指令并保证更好的Data Locality. 从系统的角度来看待这个问题就对了.

### 3. Contract

答案是什么? 我也不知道, 因为不懂的领域太多了.

开个无厘头的玩笑, 既然没有满足软硬件交付界面的合约, 那么就去找足够满足合约的人不就行了? 这样非常“人工”智能. 当然这条路也是错的.本质上我们可能要回到张量代数上来考虑这个问题, 还有很多东西值得我们探索.

现阶段可能值得讨论的有两点:

#### 3.1 SM微架构

其实从[《Tensor-003 TensorCore架构》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491424&idx=1&sn=0fc2110931b27714900e78d73b11a5b5&chksm=f9960fa2cee186b4d569cebcca2a4bbda37923bc404fd079010085e2d80faf97b290503859b6&scene=21#wechat_redirect)中我们已经看到, TensorCore的计算密度太高已经出现问题. 而更大的脉动阵列构成的TensorCore或许对于些扁平的矩阵(M>>K)也有问题. 英伟达在Hopper上搞了一个临时的WGMMA的胶布. 然后从GMEM到SMEM到RF都完全异步化, 本质上对CUDA的编程难度比起10年前大幅度的提高了.

我估计在B200中, 这样的难度还会进一步提高. 因为维持CUDA编程框架下, 提高TensorCore的算力那么只能有几点:进一步扩大Distributed Shared Memory的范围, 增加到更多的SM. 势必会出现SM互联网络的变化, 2D-Mesh或者Ring的局部拓扑会出现. 另一方面这些SM之间的异步操作可能还会伴随更多的L1.5Cache/SMEM出现.

但是是否还有更好的微架构? CUDA兼容or不兼容, DSA or GPGPU? 最终还是算法上的代数抽象决定的.

#### 3.2 互联的架构

至于Jim Keller后面讲的一些关于Tenstorrent的架构, 从成本的角度抛弃HBM**, 从互联的角度用以太网胶布, 可能都是对的.

![图片](assets/02457de2f0a5.png)

但是问题在哪 ? 在拓扑上

![图片](assets/5b1d8d6ab286.png)

2D-Mesh的拓扑带来了编译和算子放置的一系列问题. 怎么解? 也是一个非常困难的问题.或许也是一个非常简单的问题, 因为对称才是美.

大概就这样吧, 划个水, 读小黄书去~~

![图片](assets/1c0963b48637.jpg)

参考资料

[1] 
Jim Keller：使用RISC-V构建AI —— 61DAC Keynote: https://www.bilibili.com/video/BV19u8zeNER8/