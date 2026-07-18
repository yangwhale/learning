# 关于3D Stacking DRAM GPU的一些想法

> 作者: zartbot  
> 日期: 2025年6月14日 08:40  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494218&idx=1&sn=004c339866ae3d96fc19f0f40fbf2bde&chksm=f995fa88cee2739edae24afad2b9e2b7f56aa3d2655b06332bb555f1d7bd1546218762fb25ff#rd

---

最近看到国内某家做3D DRAM 加速器的一个PR, 其实还有好几家朋友的公司也在探索这条路.  硬伟大似乎也有一个PPT的板本

![图片](assets/b052451da323.png)

然后DeepSeek的一个论文也提到了这个事情:

![图片](assets/e93d617886de.png)

这几天有一个脑洞大开的想法, 记录一下...

### 1. 3D DRAM GPU的优势与问题

3D DRAM的好处是DRAM可以垂直的放在GPU的逻辑Die上方, 内存容量和带宽上可以提升数倍, 针对MoE一类的模型, 特别是推理时的Memory Bound的算子是有很大优势的.

但是它也有一个坏处, 数据的放置是一个难题. 特别是数据要跨Die通信的时候...如下左图所示

![图片](assets/91b618d98a86.png)

其实通常的做法都是尽量苦一苦算子和编译的兄弟们, 做更多的亲和性调度, 或者是苦一下硬件的兄弟, 为这样的2D-Mesh的片上网络架构增加一个z轴, 避免片内的拥塞.

### 2. NDP Trade-off

因此有, 如上右图.  我们能否在3D Stacking DRAM的基础上, 针对编程的易用性和极致的内存带宽两者之间做出一点取舍? 将一半的3D DRAM换成带SRAM的一些通信协处理器的逻辑Die, 然后NDP(Near Data Processor)之间再构建一个NOC?

这样的做法其实也是在一些PIM方案之外的另一种取舍. 以更好的编程易用性的视角出发的.

### 3. 为什么要NDP

如前所属, 单纯的构建一个新的一层的NOC, 降低通信的直径简化编程是有一些好处的. 但是进一步来看, 构建一些带有SRAM的处理逻辑, 执行一些异步的数据访问和调度可能收益更大.

举几个常见的例子吧:

首先, GPU逻辑Die内的SRAM容量是有限的, 穿越多跳2D-Mesh的片上网络通信还会占用一部分. 然后矩阵计算有大量的bcast操作和reduce操作. 新增的NDP某种意义上是扩大了GPU的SRAM空间, 同时在多个GPU Logical Die之间共享数据. 有点类似于NV GPU的Distributed SMEM的做法, 但是有一个专用的NDP Die来处理.

另一方, NDP本身可以用于构建一些TMA的处理逻辑以及其它一些通信Kernel的卸载, 简化GPU本身计算Kernel的复杂度.

同时还有一些ElementWise的处理, 例如Softmax/Norm/量化/Reduction也可以在NDP上更容易的编程. 例如DeepEP的code不占用SM而在NDP上运行, 并且能够在另一个NDP NOC上快速访问.

### 4. 从易用性考虑

不知道从编译器的视角能不能在基于NDP这样的Hybrid 3D Stacking DRAM的架构上构建一个逻辑上比较uniform的memory access layer. 降低访问内存的复杂度...感觉是用一些DSL可以实现的...

周末的一个脑洞, 欢迎大家批评, 大概就这样吧...