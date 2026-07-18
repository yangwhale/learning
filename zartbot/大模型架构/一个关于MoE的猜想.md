# 一个关于MoE的猜想

> 作者: zartbot  
> 日期: 2025年1月13日 17:17  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493019&idx=1&sn=25a87af51b9077d50b40685d6987ca71&chksm=f995f559cee27c4fe7e63beebd50e5bf643c6391ec863a2450a3af53132f19c563e723973dd3#rd

---

### TL;DR

这是一个关于如何演进MoE模型的猜想. 主要是在MoE Routing的基础上再套一层, 构建The Mixure of Expert Group(MoEG), 另一方面是在BIS一些新规出来后,探讨如何进一步用更低的算力,更松耦合的模型架构来适配.

### 先从MoE谈起

MoE的整个计算过程如下图所示：

![图片](assets/62c45485169d.png)

从代数的角度来看,MoE计算实际上是对Token进行一次置换群的操作，构成

P为一个进行Token位置置换的稀疏矩阵，实际上也构成了代数上的一个置换群的结构, 而我们再来看Monarch矩阵，两者代数结构上是相通的，Monarch矩阵定义如下

其中是Permutation矩阵，是Block Diagonal矩阵：

![图片](assets/b1b9c5651820.png)

而在MoE中，是需要对Token进行还原，保证原有的Token顺序输出到下一层。

![图片](assets/d0298e3a7967.png)

对于MoE实现的本质问题是，基于Permutation矩阵后构建的稀疏矩阵乘法如何进行并行

然而MoE有一个天然的缺点, 就是Permutation后的矩阵是一个Block Diagonal.另一方面,BigBird把稀疏性玩到花了，随机Attention，然后又是滑动窗口，再加上Global Attention，好处是这样的稀疏性是有理论保证的，坏处是随机性带来的影响和计算效率的问题.

![图片](assets/711d7b83bc4c.png)

### 从范畴论的视角看MoE

对于一个局部小范畴，每个对象包含一个C上的预层：可表示的预层(representable presheaf),实际上也就构成了一个的函子，这些函子构成预层范畴。Yoneda Lemma 这些函子是完全忠实(Fully faithful)的,即任何局部小范畴中的对象都可被对应的预层范畴中的元素表示

**问题** 这不正是我们对基础大模型泛化的要求么? 大模型的预训练的本质不就是构建预层范畴么?

另一方面

而的函子完全忠实的，那么

于是, 当且仅当它们对应的Hom函子同构。而这个推论来看，我们可以说："对象由它与其他对象之间的关系完全决定"

然而MoE的Block Diagonal矩阵其实本质上是破坏了这样的结构, 使得一些态射被忽视了. 

所以期望的方式是构造2级的Routing Gate, 使得本来Attention里面携带的信息通过两个Gate找到矩阵中(x,y)对应的某个Expert,或者多个expert.

![图片](assets/d535ef02b8c8.png)

Maybe,还可以cross MoEGroup做一些连接. 然后第一层Routing function某种意义上变成了一个Multicast to multiple rows of Experts, 第二层Routing Function Dispatch to some collumn. 然后就构成了

![图片](assets/00e9d6e2274e.png)

似乎这样又natively构成了一个态射图的结果, 然后对于通信而言,似乎也有不少可以优化的方法.

只是深夜突发奇想, 把它记录下来....