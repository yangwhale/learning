# CuTe Layout代数-1: Overview

> 作者: zartbot  
> 日期: 2025年9月23日 23:29  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496154&idx=1&sn=474a5450c46b86169095d84dd3cfd7dc&chksm=f995e118cee2680e986627723e9b4c0c512a874be1147b7919cc5650888af5329d044106295d#rd

---

### TL;DR

Cutlass中最难的一部分内容就是从3.0开始引入的CuTe tensor和Layout抽象, 遗憾的是官方的文档并不完善, 同时也没很详细的给出一些背后的设计逻辑, 只有一些案例教大家如何使用. 以前整理过一篇关于CuTe Layout的文档, 但是测试过程中反复的编译修改消耗了大量的时间, 并且也没有完整的探索和讲述背后的设计逻辑.

这次准备基于cuteDSL做一个全面的探索, 并且从代数的角度分析一下和一些其它的Layout算法进行一个对比(例如前段时间分析的[《学习一下Linear Layout》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496013&idx=1&sn=010d59e5b3916712cfc0164fa063b81c&scene=21#wechat_redirect)). 正当在写这篇文章的时候, 又惊喜的发现colfax写一篇非常棒的文章《Categorical Foundations for CuTe Layouts》[1]. 那么就基于这个这个文档做一个详细的分析吧. 后面再逐渐把这篇文章附带的150页的PDF逐个章节详细分析.

但是呢, 很多相关的同学都没有完整的学过范畴论相关的内容, 因此我会在此做一些补充. 另外还有一个专题从范畴论及其它代数视角看大模型可以参考

[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

本文目录如下:

```
1. 为什么需要Layout代数1.1 从Tensor的内存存储谈起1.2 从矩阵分块乘法的视角来看1.3 一些直观的观察2. 抽象代数和范畴论简述2.1 抽象代数2.1.1 群(Group)2.1.2 环(Ring)2.1.3 域(Field)2.1.4 总结与层级关系2.2 范畴论2.2.1 范畴的定义2.2.2 态射(Morphism)2.2.3 图2.3 Functor(函子)2.3.1 函子定义2.3.2 共变/反变函子2.3.3 Faithful/Full2.3.4 遗忘/自由函子2.3.5 Hom函子2.4 自然变换(Natural transformation)2.4.1 定义2.4.2 可表函子(Representable Functor)2.4.3 函子范畴(Functor Category)2.4.4 Presheaf预层2.5 米田引理2.5.1 Yoneda Lemma2.5.2 Yoneda Embedding2.6 泛构造 (Universal Construction)2.6.1 拉回 (Pullback)2.6.2 推出 (Pushout)3. Layout代数3.1 为什么Layout重要3.2 代数抽象3.3 Tractable Layouts3.4 Tuple-范畴3.5 Layout函数和Realization函子3.5.1 余字典序同构3.5.2 实现函子(Realization Functor)4. Layout操作4.1 合并 (Coalesce)4.2 补 (Complement)4.3 复合 (Composition)4.4 一些展开5. 嵌套布局和复合算法5.1 嵌套布局与嵌套元组态射5.2 复合算法 (The composition algorithm)5.3 逻辑除5.4 小结6. 与算子理论的联系6.1 什么是算子6.2 将布局理论嵌入算子理论6.3 一些额外的分析7. 结尾
```

## 1. 为什么需要Layout代数

### 1.1 从Tensor的内存存储谈起

对于一个Tensor, 从数据结构上来看它是一个多维数组. 对于Tensor中某个元素的逻辑坐标(Logical coord)和实际内存中的地址构成一个一一对应的映射. 而这个映射结构的基本单元就是一个Layout. 实际上一个Layout就是一个Layout函数 $f_L$

$$\text{Layout Function }: f_L(\text{ logical coord } ) \rightarrow \text{ physical address offset}$$

一个`Layout`由形状(Shape, S)和步长(Stride, D) 两部分组成:

**形状 (Shape, S):** 一个整数元组, 描述了数据的**逻辑维度**. 例如, 一个4x8的矩阵, 其Shape就是(4, 8). Shape定义了数据的逻辑结构.

**步长 (Stride, D):** 一个与Shape维度匹配的整数元组, 描述了在每个逻辑维度上移动一个单位, 需要在物理内存中"跳跃"多少个元素.

因此我们可以定义一个Layout为S和D构成的对象`L = S : D`. 下面是一个简单的例子. 对于一个4x8的矩阵, 我们在将其存储到内存中时通常有两种方式:

对于行主序(row-major)的4x8矩阵, Stride是 (8, 1), 逻辑坐标 $(i, j)$ 对应的内存偏移量为 $8i+j$

![图片](assets/4f9fc8eeb856.png)

对于列主序(col-major)的4x8矩阵, Stride是 (1, 4). 逻辑坐标 $(i, j)$ 对应的内存偏移量为 $i+4j$

![图片](assets/9f97c286f323.png)

因此我们就可以得到两种Layout:

$$L^{row}=S:D = (4,8):(8,1)$$

$$L^{col}=S:D = (4,8):(1,4)$$

同时, 对于逻辑索引到实际物理地址的Offset函数 $f_L$ 无论是哪种布局, 都是 Coord 和 Stride 的点积:

$$f_L ( coord ) = coord \cdot L_D = row * \text{stride\_row} + col * \text{stride\_col}$$

### 1.2 从矩阵分块乘法的视角来看

正如[《Tensor-001 矩阵乘法分块乘法概述》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490988&idx=2&sn=ad84861d6c7ef538027f03edfbe5cea3&scene=21#wechat_redirect)中介绍的, 矩阵分块乘法是一个非常常见的优化策略. 如下图所示:

![图片](assets/fb11e27a42b1.png)

而在分块的内部进行块乘法时,访存顺序变为A列优先/B行优先的方式, 因此矩阵的Layout变成了一种Z字排列, 如下所示:

![图片](assets/5826fa0ea9dc.png)

我们如何使用Layout函数来描述这种情况呢? 是否能够和前一节的表示方法结合成一个统一的描述? 其实我们仔细观察可以看到它相当于在矩阵内部某个维度再进行一次Layout, 因此我们采用嵌套(Nested)的方法来描述.

例如我们对于B这种Layout可以定义`Shape = (4, (2,4))`, `Stride = (2, (1,8))`

$$L^{tile1}=S:D = (4, (2,4)):(2, (1,8))$$

![图片](assets/5b47bbb7c986.png)

直观的看, 这是在将4x8的矩阵拆分为4 x (2x4)的块, 但是这种拆分后Stride是如何计算的呢? 看图观察然后掰手指数?

更进一步, 如果我们需要如下图所示的Layout, 该如何技术Shape和Stride?

![图片](assets/6577ccbf8f8f.png)

它相当于在原有的4x8 Layout的基础上, 针对两个维度都做了嵌入, `Shape= ((2,2),(2,4))` `Stride = ((1,4), (2,8))`

$$L^{tile2}=S:D = ((2,2),(2,4)):((1,4), (2,8))$$

我们可以通过如上这种画图的方式来观察在不同Shape下的Stride, 人工处理通常会带来大量的错误, 例如offset算错导致访问内存越界等问题. 有没有更巧妙的办法呢? 以及多层的逻辑坐标如何处理呢?

### 1.3 一些直观的观察

1.2中对4x8矩阵的分块, 是否能够直观的去定义一个“除法”运算呢? 例如以一个4x8的矩阵为例, 将它拆分为若干个2x4的子矩阵构成, 相当于4x8的矩阵`除以` 2x4的矩阵得到一个由2x4子矩阵铺满的4x1的新矩阵.  而第二个例子则是`除以` 2x2的矩阵, 得到一个由2x2子矩阵铺满的2x4的新矩阵. `铺满`这个并不严谨的词, 似乎又意味着某种意义上的`整除`. 例如我们如果用4x8的矩阵则无法`整除`一个2x3的矩阵.

然后我们再来看Stride, 我们以为例, 对于原始的一个矩阵, 我们是否可以定义一个`除法`, 使得

那么其中的 $T= S:D = (2,2):(x,y)$ 中的Stride又该如何表示呢? 当我们把 $L^{col}$ 中的第一个2x2的Tile的Stride作为 T 的Stride, 即 $T=(2,2):(1,4)$ 则可以构造出如下的Layout

![图片](assets/9d1e79daa5fc.png)

在cuteDSL中的测试如下:

```
import cutlassimport cutlass.cute as cute@cute.jitdef logical_divide_example():    """    Demonstrates logical divide    """    # Define the original layout    layout = cute.make_layout((4, 8), stride=(1, 4))         # Define the tiler    tiler = cute.make_layout((2,2), stride=(1,4))         # Apply logical divide    result = cute.logical_divide(layout, tiler=tiler)        # Print results    print(">>> Layout:", layout)    print(">>> Tiler :", tiler)    print(">>> Logical Divide Result:", result)cutlass.cuda.initialize_cuda_context()logical_divide_example()>>> Layout: (4,8):(1,4)>>> Tiler : (2,2):(1,4)>>> Logical Divide Result: ((2,2),(2,4)):((1,4),(2,8))
```

似乎在这里面蕴含着某种运算规则, 但这种规则又和以往的一些四则运算有所不同. 这就是我们需要引入**抽象代数** 和 **范畴论** 的根本原因.

正如我们在前一篇文章[《学习一下Linear Layout》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496013&idx=1&sn=010d59e5b3916712cfc0164fa063b81c&scene=21#wechat_redirect)中介绍抽象代数那一章. 你手上现在有一些的玩具, 需要创造一些玩法, 能够满足产生各种Layout的需求, 这就是我们为什么要引入Layout代数的原因.

## 2. 抽象代数和范畴论简述

### 2.1 抽象代数

这里主要是想写成CS背景同学能看懂的语言, 因此并没有那么形式化的严谨的描述.

假设你有一些`玩具`(这就是代数里的元素集合), 和一些`玩法` (这就是运算, 比如加法, 乘法). 抽象代数就是研究 "这套玩具" 配上 "这几种玩法" 能达到什么样 "和谐" 程度的学科. "和谐" 程度从低到高, 就分别是 群, 环, 域.

#### 2.1.1 群(Group)

群就像一个青铜玩家, 虽然有一套玩具, 但只有`一种`固定的玩法. 例如玩乐高时“拼接”玩具. 通常我们简单的把这种固定的玩法称为“加法”.

它满足以下四个基本规则, 形成一个`自给自足, 可逆`的小团体:

`不产生新玩具 (封闭性)`: 任意两个玩具拿来 "拼接", 得到的还是这套玩具里的一种, 不会冒出个奇怪的新东西.

`例子`: 整数的加法. 任何两个整数相加, 结果还是整数.

`拼接顺序不重要 (结合律)`: 你有三个玩具A, B, C. 先把A和B拼起来, 再拼C; 和先把B和C拼起来, 再拼A, 结果是一样的.

`例子`: $(2+3)+4$ 和 $2+(3+4)$ 结果一样.

`有个"啥也不干"的玩具 (单位元)`: 玩具里有个特殊的叫 "空气" 的玩具. 任何玩具和 "空气" 拼接, 等于它自己没动.

`例子`: 整数加法里的 `0`. 任何数加0, 还是它自己.

`每个玩具都有个"拆散"搭档 (逆元)`: 对每个玩具A, 你总能找到另一个玩具B, 它俩一拼接, 就变回了那个 "空气" 玩具. 这意味着所有操作都是`可逆的`.

`例子`: 对整数 $5$ 来说, 它的 "拆散" 搭档是 $-5$, 因为 $5 + (-5) = 0$.

如果这个玩法还满足"A和B拼接"跟"B和A拼接"结果一样 (交换律), 那这个群就更和谐了, 叫 "阿贝尔群" (Abelian Group), 也称为交换群.

#### 2.1.2 环(Ring)

环相当于在群这个青铜玩家的基础上学会的第二种玩法, 成为一个白银玩家. 你的玩具不仅可以 "拼接" (加法), 还学会了第二种玩法, 比如 "复制" (我们简单的将其称之为乘法), 满足:

在"拼接"玩法下是个完美的和谐的群 (阿贝尔群): 满足上面群的所有规则, 并且还是可交换的.

"复制"玩法也基本守规矩:

两个玩具 "复制", 结果还在社区里 (乘法封闭性).

连续 "复制" 的顺序不重要 (乘法结合律).

两种玩法能和谐相处 (分配律):  "复制" 可以分配到 "拼接" 上. 比如, 你先把A和B拼起来, 再整体复制3遍; 这和你先把A复制3遍, B复制3遍, 再把两堆拼起来, 结果是一样的.

`例子`: $3 \times (4+5)$ 等于 $(3 \times 4) + (3 \times 5)$.

但是环的"不完美"之处在于: 环里的 "复制" (乘法) 玩法通常不要求每个玩具都有 "拆散" 搭档. 也就是说, **环不保证除法能做**.

`例子`: 整数. 你可以做 $3 \times 2 = 6$, 但没法在整数世界里做 $3 \div 2$.

#### 2.1.3 域(Field)

相当于是一个钻石玩家, 在群(青铜)和环(白银)玩家的基础上, 把玩具的玩法掌握得炉火纯青. 一个 "域" 就是一个极度完美的 "环", 它对 "复制" (乘法) 玩法提出了更高的要求:

它首先是个环: 加, 减, 乘法都能做, 分配律也成立.

"复制"玩法也达到了几乎完美的和谐:

"A复制B" 和 "B复制A" 结果一样 (乘法交换律).

除了那个叫"空气"(0)的特殊玩具外, 每个玩具都有一个"反向复制"的搭档 (乘法逆元). 比如, 对玩具A, 总能找到一个玩具B, 它俩一"复制", 就变回了那个代表"一份"的玩具(乘法单位元, 也就是`1`). 这个"反向复制"就是除法!

"域"是一个加, 减, 乘, 除 (除了除以0) 都能自由进行的完美系统.提供了我们最熟悉的四则运算.

`例子`: 有理数(分数), 实数都是域. 在实数里, 你可以做 $3 \div 2 = 1.5$.

`反例`: 整数不是域, 因为除法行不通.

#### 2.1.4 总结与层级关系

**群 (Group)**:  `{玩具 + 1种可逆玩法}`  (比如: 整数 + 加法)   `->` 保证了`加法和减法`.

**环 (Ring)**:  `{群 + 第2种玩法 + 两种玩法通过分配律联动}`  (比如: 整数 + 加法和乘法) `->` 在群的基础上, 增加了`乘法`.

**域 (Field)**:  `{完美的环 + 第2种玩法也可逆}` (比如: 实数 + 加法和乘法)  `->` 在环的基础上, 增加了*`除法` (除0之外).

越往上, 结构越"完美", 限制越多, 但能做的事情也越多. 抽象代数就是通过这种方式, 精确地描述不同数学世界的"能力边界".

### 2.2 范畴论

对于范畴论的基础知识介绍可以参考

[《大模型时代的数学基础(2)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488528&idx=1&sn=fa49e334201e738e7ddb4258030798b3&scene=21#wechat_redirect)

[《大模型时代的数学基础(6)-从word2vec谈谈表示论，组合性，幺半范畴和Dataflow Optics》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488775&idx=1&sn=1793eb897beb71ce4a64c9ab44beee6b&scene=21#wechat_redirect)

引用Emily Riehl在Category Theory in Context中开篇的那一段来描述什么是范畴论：

阿蒂亚将数学描述为“类比的科学”。在这一领域，范畴论的视野是数学的类比。范畴论提供了一种跨学科的数学语言，旨在勾勒出一般现象，这使得思想可以从一个研究领域转移到另一个领域。范畴论的观点可以作为一个简化的抽象概念，它将那些出于形式原因成立的命题与那些需要特定数学学科的技术来证明的命题隔离开。微妙的视角转变使得数学内容可以用一种对考虑的对象种类相对漠不关心的语言来描述。范畴论的方法不是直接对对象进行刻画，而是强调同一通用类型的对象之间的变换。

范畴论是数学的一个跨学科的领域，它采用了一种新的视角来理解数学现象。与数学的大多数其他分支不同，范畴论对被考虑的对象本身不太感兴趣。相反，它专注于同一类型对象之间以及不同类型对象之间的关系。它的抽象性和广泛性使它能够触及并连接数学的几个不同分支：代数、几何、拓扑、分析等。

![图片](assets/48b1952dd7a0.png)

范畴论的一个中心主题是抽象，通过概括而不是单独关注它们来理解对象。与分类学类似，范畴论提供了一种将数学概念抽象和统一的方法. 其中最重要的米田引理(Yoneda Lemma)使我们能够通过一个对象与其他对象的关系正式定义该对象，这是范畴论所采取的以关系为中心的视角的核心.

#### 2.2.1 范畴的定义

一个范畴 $\mathcal{C}$ 由 a universe of `objects` $Ob(\mathcal{C})$, and `morphisms` $Mor(\mathcal{C})$ between them, 注意故意用universe而不是 a `set` of objects的原因是为了避免罗素悖论，同时我们给出一个`小范畴(small category)`的定义：

A category is `small` if it has a `small set` of `objects` and a `small set` of `morphisms`.

而这些对象(objects)和态射(morphisms)还需要满足如下条件

对于每个对象 $A$,存在唯一的恒等态射(identity morphism) $id_A: A\rightarrow A$

对于 $f: A \rightarrow B$ 和 $g: B \rightarrow C$，存在一个复合(Composition) $g \circ f: A \rightarrow C$

任意一个态射 $f: A \rightarrow B$, $id_B \circ f = f \circ id_A = f$

对于任意可复合 $f,g,h$, $ h \circ ( g \circ f) = (h \circ g)\circ f $

`Example.1` 例如以 “鱿，鲐，鲇，鲸，鲉，鲽” 作为对象，而带有鱼的偏旁部首作为关系构成态射:

对于 $f: 鱿 \rightarrow 鲐$ 鲐鲽 ,存在复合 $g \circ f: 鱿 \rightarrow 鲽$

任意一个态射，例如 $f: 鱿 \rightarrow 鲐$, $id_a: 鱿 \rightarrow 鱿$ , $id_b: 鲐 \rightarrow 鲐$ , 则 $id_B \circ f = f \circ id_A = f$

对于任何Composable $f,g,h$, $ h \circ ( g \circ f) = (h \circ g)\circ f $

因此，我们构造了一个带有鱼的偏旁部首作为关系的范畴，然后它的对象都是单个汉字可以构成一个集合，而所有的关系也可以构成一个集合，那么我们可以说这个鱼关系范畴是小的。

**思考题**: 范畴论中, 对象满足的态射条件是非常简单的. 我们以一个Layout作为对象, 那么如何定义态射, 使其满足范畴的定义呢?

#### 2.2.2 态射(Morphism)

对于一个态射 $f: X \rightarrow Y$,我们可以把它看作一个X到Y的箭头，并记 $dom f=X$ (domain)和 $cod f = Y$ (codomain).对于两个对象 $X,Y$,所有以X作为domain，Y作为codomain的态射构成 $Hom_C(X,Y)$

注意 $Hom_C(X,Y)$ 并不一定是一个集合，当对任意 $X,Y \in Obj(C)$,是一个集合时，我们称 $Hom_C(X,Y)$ is `locally small`

既然有态射的domain和codomain，那么如果多个指向一个，或者一个指向多个怎么定义，逆箭头是否存在？自己指向自己呢？，对于这些类型做了如下定义：

`同构(isomorphism)`:令 $f: X \rightarrow Y$，若存在态射 $g: Y \rightarrow X$ 使得 $ f \circ  g  = id_Y$ 和 $ g \circ  f  = id_X$ 成立，则称 $f$ 为一个同构态射，也称X与Y同构，记为 $X \cong Y$

如果一个范畴的所有态射都是同构的，我们将它称为一个`Groupoid`

`Example.1`例如Transformer中利用LoRA替代全连接层，如何保证两层之间的Morphism同构呢？

`满同态(epimorphism)`：$f: X \rightarrow Y$，如果对于所有的态射 $Y \rightarrow Z$ 的 $g_1,g_2$, $g_1 \circ f = g_2 \circ f \Rightarrow g_1 = g_2$ 成立。 类似于函数定义的满射(surjective)

`单同态(monomorphism)`:$f: X \rightarrow Y$，如果对于所有的态射 $Z \rightarrow Y$ 的 $g_1,g_2$, $f \circ g_1 = f \circ g_2 \Rightarrow g_1 = g_2$ 成立。类似于函数定义的单射(injective)

`自同态(endomorphism)`: $dom f = cod f = X$,即 $f: X \rightarrow X$

`自同构(automorphism)`:若一个自同态也是同构的，那么称之为自同构。

**问题** 一个有趣的话题，我们在谈论异构计算的时候，对应的同构表达是什么？各种异构硬件同构表达的IR层长成什么样？这个同构是在pytorch上模型结构的同构表达，还是张量计算这一层上，还是底层上？是否要兼容CUDA的重构？否则各个异构加速卡都有自己的框架，这样的异构是毫无意义的. 这个问题的答案对于张量运算又是什么?  是Triton? 是CuteDSL? 是Tilelang?

#### 2.2.3 图

在范畴论中, 通常通过图表的方式表示对象和态射, 态射通常在图中用箭头表示. 每个箭 $f$ 都具有以下属性:

源对象 (Source Object) 或 定义域 (Domain): 箭头出发的对象, 记作 $dom(f)$

目标对象 (Target Object) 或 上域 (Codomain): 箭头指向的对象, 记作 $cod(f)$

范畴论中的可复合性, 即对于 $f: A \rightarrow B$ 和 $g: B \rightarrow C$，存在一个复合(Composition) $g \circ f: A \rightarrow C$ 可以通过一个图表表示

$$
\require{AMScd}
\begin{CD}
A @>{f}>>B\\
@VV{h}V @VV{g}V \\
D @>{k}>> C
\end{CD}
$$

![图片](assets/20a5599e0192.png)

一个交换图 (Commutative Diagram) 是由对象和箭头组成的一个图, 它满足一个特定的属性: 从图中任意一个起始对象到任意一个结束对象, 所有路径的组合结果都是相等的. 换句话说, "殊途同归". 例如上图是一个交换图, 下图是另一个例子

### 2.3 Functor(函子)

#### 2.3.1 函子定义

设是两个范畴，一个函子(Functor) :

每个中对象,对应中有一个对象

每个中态射,对应中有一个态射

并且满足

对于中对象,

中任意态射,，有

`Example1` 在训练集中，训练数据和标签之间有一个态射f，我们期望机器学习模型能够在`模型范畴`构建态射，以很多语言模型为例，Tokenizer实际上就是一个函子。

训练数据标签训练数据标签

#### 2.3.2 共变/反变函子

`Covariant Functor`:共变函子，,

`Contravariant Functor`:反变函子，,

#### 2.3.3 Faithful/Full

`Faithful`: 对于每个 $A, B \in C$, $f,g \in Hom_C(A,B)$, $Ff=Fg \Rightarrow f=g$ 中文将Faithful翻译成忠实的,简单来说F诱导的映射是单射, 即 $Hom_D(A,B) \rightarrow Hom_D(F(A),F(B))$ 是单射

`Full`:对于每个 $ h \in Hom_D(FA,FB)$,存在 $f \in Hom_C(A,B)$,使得 $h=Ff$, 简单来说F诱导的映射是满射, 即 $Hom_D(A,B) \rightarrow Hom_D(F(A),F(B))$ 是满射

`Fully Faithful`：其实就是一个F诱导的映射 $Hom_D(A,B) \rightarrow Hom_D(F(A),F(B))$ 是双射,中文有个翻译叫完全忠实。

`Example1`:我们对基础模型的需求就是它和`世界范畴`的函子是完全忠实的，

#### 2.3.4 遗忘/自由函子

`forgetful functor`,即遗忘掉范畴中的一些结构，例如 $F: Grp \rightarrow Set$,即一个有复杂代数结构的群范畴到集合范畴的函子.

`free functor`,一个遗忘函子的反向，我们可以将其定义为自由函子(free functor)例如 $F: Set \rightarrow C$

#### 2.3.5 Hom函子

$h^A:Hom(A, –):C\rightarrow Set$ 这是一个共变函子(Covariant Functor),它包含

范畴C中每个元素X的态射 $Hom(A,X)$

对于每个态射 $f:X \rightarrow Y (X,Y \in C)$, $Hom(A,f):Hom(A,X) \rightarrow Hom(A,Y)$ 则可视为一系列态射 $f \circ g:A \rightarrow X \rightarrow Y$ 构成,其中 $g$ 为 $Hom(A,X)$ 中的每个态射.

$h_B:Hom(-,B):C\rightarrow Set$ 这是一个反变函子(contravariant Functor),它包含

范畴C中每个元素X的态射 $Hom(X,B)$

对于每个态射 $f:X \rightarrow Y (X,Y \in C)$, $Hom(f,B):Hom(Y,B) \rightarrow Hom(X,B)$ 则可视为一系列态射 $g \circ f:X \rightarrow Y \rightarrow A$ 构成,其中 $g$ 为 $Hom(X,A)$ 中的每个态射.

对 $A,A',B,B' \in C$, $f:B \rightarrow B'$, $h:A' \rightarrow A'$,

![图片](assets/7c79cfa5b8b3.png)

### 2.4 自然变换(Natural transformation)

#### 2.4.1 定义

设和是范畴，和是和之间的函子，一个从到的`自然变换`，对中每个对象,n能给出一个在D的对象间的态射,称为在X处的分量(component),使得对中每个态射都有:,用交换图表示为：

如果F和G是反变函子，则将图表中的水平箭头方向反转。若是到的自然变换，可记为或.

F和G之间的自然变换的集合被记为

#### 2.4.2 可表函子(Representable Functor)

我们可以看到，选择C中的每一个对象,我们可以获得一个从C到Set的函子, 这种指向Set的保持结构的态射，通常被成为一个表示(representable).形式化的定义如下：

一个共变(或反变)函子, C is locally small category, 如果存在一个对象,使得和自然同构(如果F为反变函子，则为),则称函子F是可以被对象A表示的.

`Example`对于一个机器学习任务，我们通常可以将其看为 $Learning = Representation + Evaluation + Optimization$,例如ChatGPT的训练过程来看,预训练就是一种对世界范畴构建可表函子的过程.

详细内容可以参考nLab[2]

#### 2.4.3 函子范畴(Functor Category)

设 $C$ 和 $D$ 是范畴， $[C,D]$ 的函子范畴的对象是 $F:C \rightarrow D$,态射是所有这些函子的自然变换，复合律是基于Vertical composition,即假设Functor $F,G,H: C\rightrightarrows D$, $\alpha : F \Rightarrow G, \beta:G \Rightarrow H$, vertical composition即 $\beta \cdot \alpha : F \Rightarrow H$. $[C,D]$ 的函子范畴记为 $D^C$

#### 2.4.4 Presheaf预层

函子范畴中最重要的一个例子就是预层(presheaf)范畴，记为， Presheaf是C上的一个函子, 上的所有presheaf构成的对象和presheaves之间的自然变换构成态射，这样的范畴被成为预层范畴。

`Example` 对于一个对象A，大模型的预训练过程实际上是通过尽量多的数据来构建A和其它对象的Attention的集合，实际上是 $h_A:Hom(-,A):C\rightarrow Set$,它是一个反变函子，也可记为 $h_A:C^{op}\rightarrow Set$,我们注意到 Presheaf是C上的一个函子 $F:C^{op} \rightarrow Set$。本质上大模型的预训练过程实际上就是需要构造一个预层范畴。

可能您读到这里感觉都是一些抽象的废话(Abstract nonsence),但这些内容都是为后面的米田引理做前置知识的铺垫

### 2.5 米田引理

米田(Yoneda)引理得名于日本数学家兼计算机科学家米田信夫(Nobuo Yoneda).大白话来说如果能够理解：“人的本质是一切社会关系的总和”，大概就清楚了它的核心了

#### 2.5.1 Yoneda Lemma

给定一个局部小范畴 $C$ 上的预层 $P$，对于C中的对象 $A$,有 $Nat(Hom(-,A),P)\simeq PA$

#### 2.5.2 Yoneda Embedding

对于一个局部小范畴 $C$，每个对象 $X$ 包含一个C上的预层：可表示的预层(representable presheaf) $h_x$,实际上也就构成了一个 $F:C \rightarrow [C^{op},Set]$ 的函子，这些函子构成预层范畴。Yoneda Lemma 这些函子是完全忠实(Fully faithful)的,即任何局部小范畴中的对象都可被对应的预层范畴中的元素表示

**问题** 这不正是我们对基础大模型泛化的要求么? 大模型的预训练的本质不就是构建预层范畴么?

另一方面

$$X\simeq Y \Rightarrow Hom(-,X) \simeq Hom(-,Y)$$

而 $F:C \rightarrow [C^{op},Set]$ 的函子完全忠实的，那么

$$Hom(-,X) \simeq Hom(-,Y) \Rightarrow  X\simeq Y $$

于是, $X\simeq Y$ 当且仅当它们对应的Hom函子同构。而这个推论来看，我们可以说：**"对象由它与其他对象之间的关系完全决定"**

### 2.6 泛构造 (Universal Construction)

Colfax的论文中还有一些Pushforward(推出)和pullback(拉回)相关的概念. 拉回和推出都属于范畴论中的泛构造 (Universal Construction).

正如前一节Yoneda Lemma所阐述的, 范畴论的核心思想不是去定义一个对象“是什么”, 而是通过它与其它所有相关对象的关系来唯一地(在同构意义下)刻画它. 这体现了范畴论的基本哲学：**对象是什么不重要, 重要的是它如何与其他对象相互作用.**

一个泛构造通常包含两个部分:

对象和一组态射: 存在一个特定的对象 $U$ 和一组从它出发 (或指向它) 的态射, 它们满足某个图表的交换性.

泛性质 (A Universal Property): 对于任何满足类似条件的其他对象，存在**唯一的态射**使得相应的图交换.

#### 2.6.1 拉回 (Pullback)

给定一个由两个态射 $f: X \rightarrow Z$ 和 $g: Y \rightarrow Z$ 构成的 **Span**, 即 $X \rightarrow Z \leftarrow Y$. 它们的 **拉回** 是一个对象 $P$ 和两个态射 $p_1: P \rightarrow X$ 和 $p_2: P \rightarrow Y$, 它们共同满足以下两个条件:

**交换性 (Commutativity)**: 下面的图表是 **交换的 (commutative)**, 意味着从 $P$ 到 $Z$ 的两条路径是等价的, 即 $f \circ p_1 = g \circ p_2$

**泛性质 (Universal Property)**: 对于`任何`其他满足类似条件的对象 $Q$ 和态射 $q_1: Q \rightarrow X$ 和 $q_2: Q \rightarrow Z$ (即 $f \circ q_1 = g \circ q_2$), 都`存在唯一`态射 $u: Q \rightarrow P$, 使得下图中的所有三角形都交换, 即: $p_1 \circ u = q_1$ 且 $p_2 \circ u = q_2$

$$
\require{AMScd}
\begin{CD}
P  @>{p_2}>> Y\\
@VV{p_1}V @VV{g}V \\
X @>{f}>> Z
\end{CD}
$$

![图片](assets/1ffa7e62e66f.png)

直观的从图上来看, 似乎是有一个箭头逐渐的把这些对象往拉回.

#### 2.6.2 推出 (Pushout)

**推出** 是拉回的**对偶 (dual)** 概念. 它回答了这样一个问题, 假设有两个态射 $f: Z \rightarrow X$ 和 $g: Z \rightarrow Y$ 都从同一个源对象 $Z$ 出发. 我们如何能找到一个“最好的”目标对象 $P$, 它既能被 $X$ 映射, 也能被 $Y$ 映射, 并且这种映射方式“尊重”了它们共同的源头 $Z$ ?

给定一个由两个态射 $f: Z \rightarrow X$ 和 $g: Z \rightarrow Y$ 构成的 **Co-span**, 即 $X \leftarrow Z \rightarrow Y$.

它们的 **推出** 是一个对象 $P$ 和两个态射 $i_1: X \rightarrow P$ 和 $i_2: X \rightarrow P$, 它们共同满足以下两个条件:

**交换性 (Commutativity)**: 下面的图是 **交换的**, 意味着从 $Z$ 到 $P$ 的两条路径是等价的: $i_1 \circ f = i_2 \circ g$

**泛性质 (Universal Property)**: 对于`任何`其他满足类似条件的对象 $Q$ 和态射 $j_1: X \rightarrow Q$, $j_2: Y \rightarrow Q$, 都`存在唯一`态射 $u: P \rightarrow Q$, 使得下图中的所有三角形都交换, 即: $u \circ i_1 = j_1$ 且 $u \circ i_2 = j_2$

$$
\require{AMScd}
\begin{CD}
P  @<{i_2}<< Y\\
@AA{i_1}A @AA{g}A \\
X @<{f}<< Z
\end{CD}
$$

![图片](assets/6f3802e324d0.png)

## 3. Layout代数

在纯粹数学和应用数学中, 有许多实例中我们都有一系列对象 (objects) 和它们之间的态射 (morphisms), 它们具有与集合和函数相同的形式化行为: 态射可以以一种可结合的方式进行组合, 并且对象拥有单位态射. 尽管集合间的函数是原型例子, 但范畴中的对象不一定是集合, 态射也不一定是函数.

Layout代数是一套形式化的数学规则和运算体系, 用于描述多维数据(如矩阵、张量)在计算机一维内存中的排列方式, 并对这些排列方式进行组合与变换. 它是NVIDIA CuTe库的核心思想, 旨在让程序员和编译器能够以一种声明式、可组合且数学上可靠的方式来处理复杂的数据布局. 可以把它想象成一套用于"内存排布"的"语法"和"运算法则".

从范畴论的观点来考察Layout的一系列运算法则和语法是非常巧妙和严谨的.

### 3.1 为什么Layout重要

**内存合并访问 (Coalesced Memory Access):** GPU 的 SIMT 模型中, 一个 Warp (通常是 32 个线程) 会同时执行相同的指令. 当这些线程访问全局内存时, 如果它们访问的地址是连续的或在某个对齐的块内, 这些访问就可以被硬件合并成一次或少数几次内存事务. 这极大地提高了内存带宽利用率. 一个好的布局可以确保一个 Warp 内的线程访问的是连续的物理内存, 从而实现合并访问. 糟糕的布局则会导致访问分散, 触发多次内存事务, 性能急剧下降.

**共享内存岸冲突 (Shared Memory Bank Conflict):** 共享内存被划分为多个 bank. 如果一个 Warp 内的多个线程同时访问同一个 bank (除了广播情况), 就会发生Bank Conflict, 导致访问串行化, 严重影响性能. 布局的设计必须考虑数据在共享内存中的排布, 避免或减少Bank Conflict.

**Tensor Cores:** 从 Volta 架构开始, NVIDIA GPU 引入了Tensor Core, 专门用于加速矩阵乘法累加 (MMA) 操作. Tensor Core操作的是特定尺寸和布局的小矩阵 (例如, `16x16x16` 的 FP16 矩阵). 为了高效使用Tensor Core, 程序员必须通过布局精确地控制数据分片 (tile), 将数据加载到寄存器/SMEM/TMEM中, 并排列成Tensor Core要求的格式. CuTe 的主要设计动机之一就是为了灵活、精确地描述这些复杂的数据分片和排列.

因此对于矩阵分块计算通常会抽象成如下流程:

![图片](assets/d636bef1f59f.png)

### 3.2 代数抽象

简单的行主序和列主序布局在高性能计算场景中是不足够的. 现代 GPU 算法, 特别是像 GEMM (通用矩阵乘法) 这样的计算密集型任务, 严重依赖于 data locality. 算法通常被设计为在高速缓存 (如 L1/L2 Cache) 或可编程缓存 (如SMEM/TMEM) 中对一小块数据 (称为 tile 或 block) 进行重复操作.

这就要求我们不仅要能表示整个张量的布局, 还要能方便地描述、切分、重组和操作这些数据块的布局. 这就是从一个静态的 "布局" 概念演进到一套动态的 "布局代数" 的根本原因. 因此通过代数运算(如 composition, logical_divide 等)从简单布局派生出任意复杂的、适应算法和硬件需求的新布局.

在1.3节我们已经介绍过了一些代数抽象, 例如传统的布局通常用一个 stride 元组来描述, 例如 $(d_M, d_N)$ 用于 $M\times N$ 矩阵. 对于数据加载, 由于GPU自身的内存层次化结构, 我们需要Layout允许**嵌套的, 层次化的**布局. 比如, 一个 $(4, 8)$ 的 tile 可以被看作是 $((2, 2), (2, 4))$ 的 `Block x Thread` 布局, 并且进一步还可以嵌套.

然后我们需要像乐高那样, 把这些Layout作为一个个积木, 然后定义一些“玩法”, 允许程序员像搭积木一样组合这些布局, 来描述复杂的数据移动和变换. 例如, 将一个Global Memory中的 tile 布局, "组合"上一个线程块内的分片布局, 就能得到每个线程应该读取的数据的最终布局.

范畴论是研究数学结构和它们之间保持结构的关系 (态射) 的学科. CuTe 布局的各种操作 (特别是Composition) 与范畴论中的**态射组合 (composition of morphisms)** 有着惊人的相似性. 将 CuTe 布局抽象为范畴论中的对象 (Objects) 和态射 (Morphisms), 可以:

**提供严谨性:** 将经验性的规则和算法用精确的数学语言定义, 保证其正确性和无歧义性.

**提供通用性:** 发现不同操作背后的统一结构. 例如前文展示的“除法”

**提供直观工具:** 利用范畴论中的箭头和图, 它将抽象的代数运算变成了直观的连线游戏, 极大地简化了推理和计算.

### 3.3 Tractable Layouts

虽然为任意布局定义和计算布局操作很困难, 但我们可以通过限制在**易处理的布局 (tractable layouts)** 内来开发一个直观的工作框架. 这包括了实践中遇到的几乎所有布局, 例如:

**行主序 (row-major)** 和 **列主序 (column-major)** 布局

**紧凑 (compact)** 布局, 将数据存储在连续的内存地址中

**投影 (projections)** 用于broadcast数据的多个副本

**扩张 (dilations)**, 用于实现带填充的LD/ST

那么什么是难以处理的Layout呢?
对于一个Layout来看, 我们可以通过Shape:Stride定义. Shape可以是任意形状任意维度构成的一个元组(Tuple). 是否容易处理取决于在Shape约束下的Stride如何定义. 例如一个 Layout: (4,8):(2,-1)数据在内存中的位置是不连续的, 甚至会出现overlap(两个逻辑地址访问同一个物理地址)或者访问地址越界的情况.

因此, 我们考虑到一个最简单的Flatten的Layout, 即在Shape和Stride中都没有嵌套, 只用一维的数组 $(x_1, \ldots, x_n)$ 表示, 对于一个Layout $L = S:D = (s_1, \ldots, s_m) : (d_1, \ldots, d_m) $ 为了便于描述, 我们再引入**Mode**概念

**Size:** 定义 $\text{size}(L) = s_1 \cdot s_2 \cdot \ldots \cdot s_m$, 它是Shape中各个dim的积.

**Length:** 定义 $L$ 的Length为 $\text{len}(L) = m$

**Mode:** 对于 $\forall k \in [1,m],\quad (s_k):(d_m) $ 是一个length=1的Layout,我们将其称为 $L$ 的Mode.

然后对于Shape和Stride做如下约束. 首先, 我们在整数对 $s : d$ 上定义一个序关系 $\preceq$ 如下:

$$s : d \preceq s' : d' \quad \text{ if and only if } \quad d < d' \: \text{ or } \: d = d' \text{ and } s \leq s' $$

注: 此处实际上是按照Stride从小到大对Layout的Mode进行排序.

**定义:** 我们说一个扁平布局(Flatten Layout)

$$L = (s_1, \ldots, s_m) : (d_1, \ldots, d_m) $$

是**易处理的 (tractable)**, 如果对于所有整数对 $1 \leq i, j \leq m$, 以下条件成立:

$$\text{if  } s_i : d_ i \preceq s_j : d_j \text{  and  } d_i, d_j \neq 0, \text{  then  } s_i d_i \text{  divides  } d_j $$

序关系 $\preceq$ 首先按 stride 排序, stride 相同再按 shape 排序. 这定义了一种从 "最快变化" (stride 最小) 到 "最慢变化" (stride 最大) 的维度排序. 条件 $s_i * d_i$ 整除 $d_j$ 的含义是: 如果维度 $i$ 比维度 $j$ 变化更快, 那么维度 $i$ 这一整块在内存中的跨度 ( $s_i * d_i$ ) 必须能够无缝地铺满构成维度 $j$ 的一个单位.

**Example:** 考虑一个行主序的 $(M, N)$ 矩阵, 其布局为 shape=(M, N), stride=(N, 1). 即Layout

$$L = S:D = (s_1, s_2) : (d_1, d_2) = (M, N):(N, 1)$$

它的两个Mode为 $s_1:d_1$ 即 $M:N$, $s_2:d_2$ 即 $N:1$

根据定义, $1 < N$, 所以 $N:1 \preceq  M:N$

此时 $i=2, j=1$. 我们需要检查 $s_2 * d_2$ 是否整除 $d_1$

$s_2 * d_2 = N * 1 = N$. $d_1 = N$. $N$ 整除 $N$

所以, 行主序布局是易处理的(tractable).

对于一个Layout是否为易处理的, 我们可以通过colfax开源的layout-categories工具tract进行分析, 安装如下

```
git clone https://github.com/ColfaxResearch/layout-categoriescd layout-categoriescd tractpip install .
```

然后我们可以在CuteDSL环境中进行测试

```
import cutlassimport cutlass.cute as cuteimport tract@cute.jitdef test_is_tractable():    A = cute.make_layout(shape=(2,2,2), stride=(1,2,4))    B = cute.make_layout(shape=(2,2,2), stride=(1,7,4))    A_is_tractable = tract.is_tractable(A)    B_is_tractable = tract.is_tractable(B)    print(f"A =", A)    print(f"A is tractable: {A_is_tractable}")    print(f"B =", B)    print(f"B is tractable: {B_is_tractable}")test_is_tractable()# output:A = (2,2,2):(1,2,4)A is tractable: TrueB = (2,2,2):(1,7,4)B is tractable: False
```

colfax的文章有几个图有点误导, 实质上, 我们只是对S:D按Mode 对Stride进行了升序排序, 这个重排序的过程是一个一一映射, 然后这个可以很容易的画成一个图

![图片](assets/c20dfe9441fa.png)

例如我们对于Layout= (2,3,2,4):(1,4,2,24)重排序的映射为

![图片](assets/2ff4a3eb8f10.png)

colfax的blog原始贴图和解释有一些错误在此, 其实并不是prefix products, 而是重排后能整除...

易处理" (Tractable) 的本质是对Layout的Shape和Stride进行一个简单的约束, 这个条件保证了布局具有良好的"分层"或"瓦片化"结构对应于GPU的内存层次结构. 它排除了那些在内存中交错、混乱的"病态"布局.

### 3.4 Tuple-范畴

我们对于3.3节中的对象和箭头表示的图来构建一个范畴, 即Tuple-范畴

`对象`: 是一个正整数元组 .

`态射`: $f : (s_1, \ldots, s_m) \to (t_1, \ldots, t_n)$ 由一个有限点集 (finite pointed sets) 之间的映射 $\alpha$ 指定

$$\alpha: \{ \ast, 1, \ldots, m \} \to \{ \ast, 1, \ldots, n\} $$

并满足以下条件:

$\alpha(*) = *$

如果 $\alpha(i) \neq * $ 且 $\alpha(i) = \alpha(i')$, 那么 $i = i'$

如果 $\alpha(i) = j \neq * $, 那么 $s_i = t_j$

我们称这样的态射 $f$ 位于 $\alpha$ 之上, 并称 $f$ 为一个元组态射 (tuple morphism).

对于条件1和2, 表示每个源维度要么被丢弃/广播(映射到 `*` ), 要么唯一地映射到一个目标维度. 不能多个源维度挤占同一个目标维度. 对于条件3( $s_i = t_j$ )表示映射前后, 维度的"大小"不变. 这确保了布局变换只是"重排"和"广播", 而不是改变张量本身的数据量.

**定义:** 如果 $f$ 是一个元组态射, 那么由 $f$ 编码的布局是

$$L_f = (s_1, \ldots, s_m) : (d_1, \ldots, d_m) $$

其 Shape 是 $f$ 的定义域, 其 Stride 由下式给出:

$$
d_i = \begin{cases} t_1 \cdots t_{j-1} & \text{if } \alpha(i) = j \\
0 & \text{if } \alpha(i) = * \end{cases}
$$

对于 $f:  (s_1, \ldots, s_m) \to (t_1, \ldots, t_n)$, Shape是一个定义域, 右侧 $(t_1, \ldots, t_n)$ 是一个中间表示(IR). 即 $\text{Shape} \rightarrow \text{IR}$. 实际上对于同一个Layout可以用多个IR表示. 例如一个Layout=(4,5):(1,64), IR表示可以为(4,16,5), (4,16,5,7), (4,2,8,5)等... 只需要能够满足Tuple范畴条件即可.

![图片](assets/0429e355f73b.png)

对于这些IR表示中, 我们希望获得一个最简的表示, 即没有多余条目(例如没有 $g$ 中的 7 ), 未被映射的条目合并(例如对比 $f,h$, $f$ 中的16是 $h$ 中(2,8)合并的). 对于这个最简的表示, 我们可以称为 $f:(4,5) \rightarrow (4,16,5)$ 具有**标准形式 (standard form)**.

进一步假设布局和态射是**非退化的 (non-degenerate)**, 这对应于如下条件

Stride 计算公式: $d_i = t_1 \cdots t_{j-1}$ 这个公式是(colexicographic, colex)编码的体现. 态射的目标 $(t_1,\ldots, t_n)$ 定义了一个线性的内存空间, 其大小为 $\prod t_k$, $t_j$ 元素 $t_1\cdot \ldots \cdot t_{j-1}$ 的偏移量是 $s_i$.

当源维度 $s_i$ 映射到 $t_j$ 时, $s_i$ 就继承了这个偏移量作为其 stride. 如果 $s_i$ 映射到 `*`, 它的 stride 就是 0, 这对应于广播 (broadcasting).

在高层次上, "non-degenerate" (非退化) 是一个约束条件, 用于消除布局表示中的冗余. 具体来说, 它处理的是大小为 1 的维度. 一个大小为 1 的维度在逻辑上是微不足道的, 因为它只包含一个元素(索引为 0), 沿着这个维度移动没有任何意义.

反例: L=(1,8):(8,1)中 $s_1 =1, d_1=8 \neq 0$ 这违反了非退化条件, 所以它是一个退化布局. 这个布局实际上描述的是一个 8 元素的向量. 逻辑坐标 $(0, j)$ (其中 j 从 0 到 7) 映射到的物理偏移是 $0\times 8+j \times 1=j$. 这个 $s_1=1$ 的维度是多余的, 它只是给一个简单的一维向量强加了一个二维的外壳, 因此可以退化为一个一维的布局.

**定理:** 非退化的易处理扁平布局(non-degenerate tractable flat layouts) 与 标准形式的非退化元组态射(non-degenerate tuple morphisms of standard form) 之间存在一一对应的关系.

**A. 非退化的易处理扁平布局 (Non-degenerate Tractable Flat Layout)**

这是一个布局 $L = (S: D)$ 同时满足以下三个条件:

**扁平 (Flat):** $S$ 和 $D$ 都是简单的整数元组, 形如 $(s_1, \ldots, s_m)$ 和 $(d_1, \ldots, d_m)$.

**易处理 (Tractable):** 对于所有 $1 \le i, j \le m$, 如果 $s_i:d_i \preceq s_j:d_j$ (即 $d_i < d_j$ 或 ( $d_i = d_j$ 且 $s_i \le s_j$ )) 且 $d_i, d_j \ne 0$, 那么 $s_i d_i$ 必须整除 $d_j$.

**非退化 (Non-degenerate):** 如果 $s_i = 1$, 那么 $d_i = 0$.

**B. 标准形式的非退化元组态射 (Non-degenerate Tuple Morphism of Standard Form)**

这是一个元组态射 $f: S \to T$ (由映射 $\alpha$ 定义) 同时满足以下三个条件:

**非退化 (Non-degenerate):** 如果 $s_i = 1$, 那么 $\alpha(i) = *$.

**标准形式 (Standard Form):** 这个条件比较复杂, 论文中给出了严谨定义, 我们可以将其直观地理解为两点:

**无冗余 1:** 余定义域 T 中不包含值为 `1` 的元素. 即 $t_k \neq 1$ 对所有 $k$ 成立. 因为任何 `1` 都可以被合并到其他元素中而不改变编码的布局.

**最大化合并:** 余定义域 T 中所有未被箭头指向的元素 (即不在 $\alpha$ 的像集中的元素) 都被合并成一个单一的元素 (如果存在多个的话). 这样做是为了保证表示的唯一性. 比如, 如果 $t_k$ 和 $t_l$ 都没有被箭头指向, 它们应该被合并成一个新的元素 $t_{new} = t_k * t_l$.

**元组态射 (Tuple Morphism):** 满足之前提到的基本规则.

这个定理是连接**应用世界 (布局)** 和 **数学世界 (态射)** 的关键桥梁.一一对应表示:

$L_{f_L} = L$: 从一个布局 $L$ 出发, 构造出态射 $f_L$, 再从 $f_L$ 编码回布局, 得到的一定是原来的 $L$.

$f_{L_f} = f$: 从一个标准形式的态射 $f$ 出发, 编码出布局 $L_f$, 再从 $L_f$ 构造回态射, 得到的一定是原来的 $f$.

这两个性质源于**标准形式**的唯一性和构造算法的确定性. 从 $L$ 构造 $f_L$ 的过程本质上是"分解" stride, 而从 $f_L$ 计算 $L$ 的过程是"合成" stride. 由于标准形式保证了分解和合成的唯一性, 这两个操作自然互为逆运算.

当我们需要**直观理解和高性能编程**时, 我们处理**布局 (Layout)**.

当我们需要**严谨的数学运算和算法设计**时, 我们可以切换到**态射 (Morphism)** 的视角, 利用其清晰的结构和组合规则.

这个定理是整个理论框架的基础, 后续所有关于布局操作 (组合, 补, 划分等) 的讨论, 都建立在这个坚实的一一对应关系之上.

### 3.5 Layout函数和Realization函子

一个布局 L 最重要的不变量是它的`布局函数 (layout function)`. 当 L 是易处理的时, 它的布局函数可以通过一个`实现函子 (realization functor)` 从范畴 `Tuple` 自然地产生.

#### 3.5.1 余字典序同构

定义: 如果 $S = (s_1, \ldots, s_m)$ 是一个大小为 M ( $M = \prod s_i$ ) 的正整数元组,

**余字典序同构 (colexicographic isomorphism)** 是函数:

$$\mathrm{colex}_S: [0, s_1) \times \cdots \times [0, s_m) \to [0, M) $$

定义为:

$$\mathrm{colex}_S(x_1, \ldots, x_m) = \sum_{i=1}^m x_i \cdot s_1 \cdots s_{i-1} $$

**逆余字典序同构 (inverse colexicographic isomorphism)** 是函数:

$$\mathrm{colex}_S^{-1}: [0, M) \to [0, s_1) \times \cdots \times [0, s_m) $$

定义为:

$$\mathrm{colex}_S^{-1}(x) = (x_1, \ldots, x_m) $$

$$x_i = \lfloor x / (s_1 \cdots s_{i-1} ) \rfloor \pmod{s_1 \cdots s_i} $$

其中:

对于函数 $colex_S$, 它的输入是一个多维逻辑坐标 $(x_1, \ldots, x_m)$, 其中 $x_i \in [0, s_i)$, 输出是一个一维线性索引 $k \in [0, M)$. 它定义了一种将多维坐标空间"铺平"成一维线段的标准方式. 对于Layout函数, 我们可以看作为三个步骤:

**逻辑坐标 -> 线性索引 (输入):** 用 $colex_S$ 将输入的逻辑坐标 $(x_1, \ldots, x_m)$ 转换为一个单一的线性索引 $k_S \in [0, M)$.

**线性索引 -> 线性索引 (核心映射):** 将输入空间的线性索引 $k_S$ 映射到输出空间的线性索引 $k_T$ ( $|f|$ )。

**线性索引 -> 物理偏移 (输出):** 输出的线性索引 $k_T$ 实际上就是我们想要的物理内存偏移.

#### 3.5.2 实现函子(Realization Functor)

当 L 是易处理的时, 我们可以通过一个从 **Tuple** 到**FinSet** 有限集范畴的**实现函子 (realization functor)** 来恢复其布局函数.

**定理:** 存在一个函子

我们称之为**实现 (realization)**, 它满足以下性质:

$$| \cdot | : \textbf{Tuple} \to \textbf{FinSet} $$

如果 S 是一个大小为 M 的元组, 那么 $|S| = [0, M)$

如果 S 和 T 是大小分别为 M 和 N 的元组, 且 $f: S \to T$ 是一个元组态射, 那么它的实现 $|f|: [0, M) \to  [0,N) \subset \mathbb{Z}$ 就是 $L_f$ 的布局函数.

特别地, 这个结果提供了一个简单的的证明, 即`元组态射`的组合(Composition)与`Layout`的组合(Composition)是兼容的.

首先来看如何构造, 对于一个输入索引 $k \in [0, M)$:

**Decompose:** 使用逆余字典序同构, 将线性索引 $k$ 分解为多维逻辑坐标 $(x_1, \ldots, x_m) = \mathrm{colex}_S^{-1}(k)$.

**Map:** 对每个坐标分量 $x_i$, 查找其在态射 `f` (即 $\alpha$ 映射) 中的去向.

如果 $\alpha(i) = j \neq *$, 那么这个坐标分量 $x_i$ 贡献到了目标空间的第 $j$ 个维度. 我们记 $y_j = x_i$.

如果 $\alpha(i) = * $, 那么这个坐标分量 $x_i$ 被丢弃 (它不贡献到目标坐标).

**Recompose:** 我们现在有了一组目标坐标分量 $(y_1, \ldots, y_n)$ (其中某些 `y` 可能未定义, 但这在 `colex` 中可以处理为 0). 使用目标空间的余字典序同构, 将这些分量重组为一个单一的输出索引:

$$|f|(k) = \mathrm{colex}_T(y_1, \ldots, y_n) = \sum_{j=1}^n y_j \cdot t_1 \cdots t_{j-1} $$

布局函数 $\Phi_L(x_1, \ldots, x_m)$ 的定义是 $\sum_i x_i d_i$.

将我们从态射得到的 $d_i$ 公式代入:

$$\Phi_{L_f}(x_1, \ldots, x_m) = \sum_{i \text{ s.t. } \alpha(i) \neq *} x_i \cdot (t_1 \cdots t_{\alpha(i)-1}) $$

令 $y_j = x_i$ (当 $\alpha(i)=j$ ), 上式就变成了:

$$\sum_{j \in \text{Im}(\alpha)} y_j \cdot (t_1 \cdots t_{j-1}) $$

这正是我们上面推导的 $colex_T$ 的结果, 所以, $|f|$ 的函数定义可以写成:

$$|f| = \mathrm{colex}_T \circ \alpha^* \circ \mathrm{colex}_S^{-1} $$

其中 $\alpha^*$ 是一个根据 $\alpha$ 重排坐标的函数.

这个函子的作用是连接两个范畴. $|f|$ 就是 $L_f$ 的布局函数, 即

直接计算法 (传统方法): `shape` + `stride` -> $\Phi_L(x) = \sum x_i d_i$.

函子实现法 (范畴论方法): `morphism f` -> $|f| = \mathrm{colex}_T \circ \alpha^* \circ \mathrm{colex}_S^{-1}$.

对于`组合的兼容性`, 即 $L_{g \circ f} = L_g \circ L_f$. 证明如下:

布局组合的定义是函数组合:

$$\Phi_{L_g \circ L_f} = \Phi_{L_g} \circ \Phi_{L_f}$$

根据本章定理:

$$\Phi_{L_g} = |g| \quad and \quad \Phi_{L_f} = |f|$$

所以,

$$\Phi_{L_g \circ L_f} = |g| \circ |f|$$

函子的基本性质是保持组合:

$$|g \circ f| = |g| \circ |f|$$

因此, 我们得到

$$\Phi_{L_g \circ L_f} = |g \circ f|$$

再次使用本章定理, 一个函数的布局函数是 $|g \circ f|$, 那么这个函数本身一定是 $L_{g \circ f}$

所以, $L_{g \circ f} = L_g \circ L_f$. 证明完毕.

## 4. Layout操作

这一章展示了如何将之前建立的范畴论框架 (态射、布局图) 应用于 CuTe 中最常见, 最核心的几种布局操作: **Coalesce (合并)**, **Complement (补)**, 和 **Composition (组合)**. 核心思想是为每一种布局操作, 都在态射的世界里找到一个对应的, 更简单直观的图形化操作, 并证明这两者是等价的.

### 4.1 合并 (Coalesce)

布局 $L$ 的合并操作是, 当 $d_{i+1} = s_i * d_i$ 时, 可以将

$$(..., s_i, s_{i+1}, ...):(..., d_i, d_{i+1}, ...)$$

合并为

$$(..., s_i * s_{i+1}, ...):(..., d_i, ...)$$

我们可以用 $\mathit{coal}(L)$ 表示, 例如对于如下一个Layout

$$L = (2,2,5,5,5):(1,2,8,40,200)$$

可以合并为

$$\mathit{coal}(L)=(4,125):(1:8)$$

在图中表示如下, 即同一个目标元组中相邻位置的 "平行" 箭头合并

![图片](assets/818f0ef7c7b8.png)

**定理:** 图的合并等价于布局的合并

$$L_{\mathit{coal}(f)} = \mathit{coal}(L_f)$$

 $d_{i+1} = s_i * d_i$ 这个条件是关键, 它意味着维度 $i$ 和 $i+1$ 在逻辑上相邻, 并且在物理内存中也是完全连续的. 这是一个优化操作, 用于简化布局的表示, 减少逻辑维度的数量, 使其更接近于物理内存的线性本质.

Cute-DSL中的Example如下:

```
import cutlassimport cutlass.cute as cute@cute.jitdef coalesce_example():    """    Demonstrates coalesce operation flattening and combining modes    """    layout = cute.make_layout(shape=(2,2,5,5,5), stride=(1,2,8,40,200))     result = cute.coalesce(layout)    print(">>> Original:", layout)    print(">>> Coalesced:", result)    coalesce_example()# output>>> Original: (2,2,5,5,5):(1,2,8,40,200)>>> Coalesced: (4,125):(1,8)
```

合并操作的本质是**识别并消除布局中的冗余逻辑维度**. 在 GPU 编程中, 我们希望逻辑布局尽可能地贴近物理内存. 例如, 一个 `4x8` 的 tile, 如果它是列主序存储的, 其 shape=(4, 8), stride=(1, 4). 满足合并条件 ( $4 = 4 * 1$ ). 合并后得到 `shape=(32), stride=(1)`. 这清晰地表明, 这 32 个元素在内存中是连续的, 非常适合使用 `memcpy` 或向量化加载指令(如 `ld.global.v4.b32`) 来进行高效访问.

 这个定理再次体现了范畴论框架的简洁. 它保证了我们在图上进行的直观的"合并箭头"操作, 其结果与在布局上进行的复杂的"检查stride并合并"操作是完全等价的. 这个图形操作比在 stride 元组上进行复杂的代数运算要直观得多.

### 4.2 补 (Complement)

*L* 是一个布局, *N* 是一个正整数, 那么 *comp(L, N)* 是一个经过排序和合并的布局, 其与 *L* 拼接后是紧凑的 (compact). 这意味着拼接后布局的布局函数在其像集上是同构的. 存在一个最小的整数 *N*, *L* 对其承认一个补, 在这种情况下, 我们写作 *comp(L) = comp(L, N)*. 例如, 如果

$$L = (2,2,2):(1,10,60) $$

那么

$$comp(L) = (5, 3) : (2, 20) $$

同样, 在范畴 **Tuple** 中也有补的类比. 我们可以通过包含那些**未被 *f* 击中**的条目来计算一个元组态射 *f* 的补 $f^c$. 例如:

![图片](assets/e2989331c279.png)

即在范畴 **Tuple** 中的补与布局的补是兼容的.

**定理:** 如果 *f* 是一个标准形式的单射元组态射, 那么

$$L_{f^c} = \mathit{comp}(L_f) $$

Cute-DSL中的例子

```
import cutlassimport cutlass.cute as cute@cute.jitdef comp_example():    """    Demonstrates complement operation     """    layout = cute.make_layout(shape=(2,2,2), stride=(1,10,60))     result = cute.complement(layout, cute.size(layout))    print(">>> Original:", layout)    print(">>> Complement:", result)    comp_example()# output>>> Original: (2,2,2):(1,10,60)>>> Complement: (5,3):(2,20)
```

补操作的本质是**描述一个子集之外的剩余部分**.想象一个大的数据张量, $L$ 描述了其中一个 tile 的布局. 那么 $comp(L)$ 就描述了 "所有其他数据" 组成的集合的布局. 这个操作对于实现 Partitioning 和 Tiling 至关重要. CuTe 中一个常见的模式是 make_tile(L, comp(L)), 它将一个大张量划分为 $L$ 描述的 tile 部分和 $comp(L)$ 描述的剩余部分. 程序员可以处理 $L$ 部分, 然后在 $comp(L)$ 上递归地进行下一次划分.

另外 $L$ 和 $comp(L)$ 拼接后是紧凑的. "紧凑"意味着布局函数是单射的, 即没有两个不同的逻辑坐标映射到同一个物理地址, 并且其像集(所有物理地址的集合)是连续的. 换句话说, $(L, comp(L))$ 完整地、无重叠地、无空洞地覆盖了整个数据张量.

另外,在态射的世界里, 求补操作变得直观. 给定态射 $f: S \to T$, 它描述了 $S$ 中的维度如何"挑选" $T$ 中的某些维度. 它的补 $f^c$ 的定义域就是 $T$ 中那些`未被挑选`的维度组成的元组, 而 $f^c$ 的态射则将这些维度"挑选"出来.

这使得程序员或编译器设计者可以从更高层次思考数据划分问题, 只需考虑"我需要这部分", 剩下的部分可以由系统通过 `comp` 操作自动、正确地计算出来.

### 4.3 复合 (Composition)

如果 A 和 B 是布局, 那么**复合 (composition)** 是一个布局 $B \circ A$, 使得对于任何 $x \in [0, \mathrm{size}(B \circ A))$, 我们有:

$$\Phi_{B \circ A}(x) = \Phi_B ( \Phi_A(x)) $$

还有其他属性可以唯一地刻画布局 $B \circ A$.

例如, 如果 A = (2, 2) : (5, 50) 且 B = (5, 2, 5, 2) : (1, 25, 5, 50), 那么 A 和 B 的复合是 (2, 2) : (25, 50).

如果 $f$ 和 $g$ 是元组态射, 且 `codomain(f) = domain(g)`, 那么我们可以复合 $f$ 和 $g$ 形成元组态射 $g \circ f$. 例如, 下图的态射是可复合的:

![图片](assets/fbe2005ee06f.png)

它们的复合如下图所示:

![图片](assets/65b26b8b0e7c.png)

从图上证明了在范畴 **Tuple** 中的复合与布局的复合是兼容的.

**定理:** 如果 $f$ 和 $g$ 是可复合的元组态射, 那么

$$L_{g \circ f} = L_g \circ L_f $$

复合是描述**层次化 (Hierarchical)** 内存访问模式, 它是将两个独立的布局映射串联起来, 形成一个更宏观的映射.

经典应用场景 (Global -> Shared -> Register):

**布局 A:** 将一个线程块 blockIdx 映射到它在全局内存中负责的那个大 tile 的基地址. `A: BlockCoord -> GlobalOffset`.

**布局 B:** 将一个线程 threadIdx 映射到它在共享内存中负责的那个小 tile 的位置. `B: ThreadCoord -> SharedOffset`.

**复合 $B \circ A$**: 描述了从 (blockIdx, threadIdx) 这样的 "全局线程ID" 直接到它应该访问的全局内存数据的映射.

在态射的世界里, 复合就是将布局图首尾相连, 然后消除中间部分.

#### 4.4 一些展开

对于4.3中的复合条件 `codomain(f) = domain(g)` 在现实世界中, 这个条件几乎总是**不满足**的. 比如布局A的输出是 `(128, 128)` 的 tile, 而布局B的输入是 `(32, 8)` 的线程网格. `(128, 128)` 和 `(32, 8)` 根本不匹配. 实际上我们是否有可能在 $f$ 上细化一个态射 $f'$ 以及在 $g$ 上细化一个态射 $g'$, 然后使得 $g'\circ f'$ 能够表示 $B \circ A$ ?

这个挑战正是下一章节 "Nested layouts and the composition algorithm" (嵌套布局与合成算法) 要解决的核心问题. 后续的 "相互细化 (mutual refinement)", "pullback", "pushforward" 等更高级的范畴论工具, 就是为了解决 `codomain(f) != domain(g)` 这个难题而引入的.

另一方面, 我们再来看Tiling中使用的逻辑除法, 逻辑除法是 `complementation` 和 `composition` 两种基础运算的结合.

## 5. 嵌套布局和复合算法

在前面的章节中, 我们为扁平布局 (flat layouts) 建立了一套理论, 并指出了其局限性: 即使两个布局 `A` 和 `B` 在逻辑上是可复合的, 它们对应的标准型态射 $f$ 和 $g$ 也很可能因为 codomain 和 domain 不匹配而无法在范畴中直接组合. 本章引入**嵌套布局 (Nested Layouts)** 来解决这个问题, 并基于此构建一个通用的**组合算法 (Composition Algorithm)**.

### 5.1 嵌套布局与嵌套元组态射

这部分内容首先将之前的所有理论从"扁平 (flat)"世界推广到"嵌套 (nested)"世界. 下图是一个例子:

![图片](assets/515d05110810.png)

我们首先定义一些术语

#### profile

Profile是一个嵌套元组, 其每个条目都是符号 $*$. 例如 $P* = (*, (*, *))$ 和 $Q* = ((*, *), *, (*, *))$ 都是Profile.

一个**嵌套元组 (Nested Tuple)** 由其**扁平化 (flattening)** (一个普通元组) 及其Profile $P$ 唯一确定. 在使用嵌套元组时, 写作 $S = (s_1, \ldots s_m)_P$ 会很方便. 例如, 如果 $S = ((2, 2), (5, 5))$, 我们可以写作 $S = (2, 2, 5, 5)_P$ 其中 $P* = ((*, *), (*, *))$

如果 $L = S:D$ 是一个布局, 那么 $S$ 和 $D$ 需要有相同的Profile, 所以我们可以将一个通用布局写作

$$L = (s_1, \ldots, s_m)_P : (d_1, \ldots, d_m)_P$$

我们称布局

$$L^\flat = (s_1, \ldots, s_m) : (d_1, \ldots, d_m)$$

为 $L$ 的**扁平化**. 关于扁平布局的大部分理论可以轻松地移植到嵌套情况.

**定义:** 如果一个布局 $L$ 的扁平化 $L^\flat$ 是易处理的, 我们就说 $L$ 是**易处理的 (tractable)**.

如果 L 是易处理的, 那么 L 可以被一个图表示. 例如:

![图片](assets/dfb883c4f1be.png)

这些图代表了范畴 **Nest** 中的态射.

**定义:** 令 **Nest** 表示这样一个范畴:

`对象`: 正整数的嵌套元组

`态射`: 其态射的定义与 **Tuple** 范畴中的元组态射相同(基于扁平化的索引)

**嵌套元组态射(Nested Tuple Morphism)**:

态射 $f : (s_1, \ldots, s_m)_P \to (t_1, \ldots, t_n)_Q$ 由一个有限点集 (finite pointed sets) 之间的映射 $\alpha$ 指定

$$\alpha: \{ \ast, 1, \ldots, m \} \to \{ \ast, 1, \ldots, n\} $$

并满足以下条件:

$\alpha(*) = *$

如果 $\alpha(i) \neq * $ 且 $\alpha(i) = \alpha(i')$, 那么 $i = i'$

如果 $\alpha(i) = j \neq * $, 那么 $s_i = t_j$

我们说这样的态射 $f$ 位于 $\alpha$ 之上，并将 $f$ 称为嵌套元组态射。

如果 $f$ 是嵌套元组态射，则由 $f$ 编码的布局是布局

$$L_f = (s_1, \ldots, s_m)_P : (d_1, \ldots, d_m)_P $$

其形状是 $f$ 的定义域, 其Stride由以下公式给出:

$$
d_i = \begin{cases} t_1 \cdots t_{j-1} & \text{if } \alpha(i) = j \\
0 & \text{if } \alpha(i) = \ast \end{cases}
$$

我们可以在嵌套情况下定义**标准型 (standard form)** 和**非退化性 (non-degeneracy)**, 并再次得到一个对应定理.

**定理:** 非退化的易处理布局( non-degenerate tractable layouts) 与 标准型的非退化嵌套元组态射(non-degenerate nested tuple morphisms of standard form) 之间存在一一对应关系.

![图片](assets/c1238557ec96.png)

我们可以通过**扁平化函子 (flattening functor)** 来比较 **Nest** 和 **Tuple** 这两个范畴:

$$(-)^\flat : \mathbf{Nest} \to \mathbf{Tuple}. $$

特别地, 我们可以将其与从 **Tuple** 到 **FinSet** 的实现函子进行后组合 (postcompose), 从而得到一个在 **Nest** 上定义的实现函子:

$$| \cdot | : \textbf{Nest} \to \textbf{FinSet} $$

它享有与之前相同的性质.

**定理:** 从 **Nest** 到 **FinSet** 的实现函子满足以下性质:

如果 $S$ 是一个大小为 $M$ 的嵌套元组, 那么 $|S| = [0, M)$.

如果 $S$ 和 $T$ 分别是大小为 $M$ 和 $N$ 的嵌套元组, 且 $f : S \to T $ 是一个元组态射, 那么其实现 $ |f| : [0, M) \to [0, N) \subset \mathbf{Z} $ 就是 $ L_f $ 的布局函数.

特别地, 这个定理可以轻松证明以下结果:

**定理:** 如果 $f$ 和 $g$ 是可复合的嵌套元组态射, 那么:

$$L_{g \circ f} = L_g \circ L_f $$

**Nest** 范畴支持许多重要布局操作的类似物, 例如合并 (coalesce), 互补 (complement), 逻辑除 (logical division), 和逻辑乘积 (logical product). 我们在下面的定理中总结了这些操作及其与相应布局操作的兼容性.

**定理:**

我们在嵌套元组态射上定义了一个**合并**操作 $coal(f)$, 它与布局的合并兼容, 即

$$L_{\mathit{coal}(f)} = \mathit{coal}(L_f) $$

我们在嵌套元组态射上定义了一个**互补**操作 $f^c$, 它与布局的互补兼容, 即如果 $f$ 是一个标准型的单射嵌套元组态射, 则

$$L_{f^c} = \mathit{comp}(L_f) $$

我们定义了嵌套元组态射的**可除性 (divisibility)** 概念, 以及当 $g$ 整除 $f$ 时的**逻辑除**操作 $ f \oslash g $. 此操作与布局的逻辑剖分兼容, 即

$$\mathit{coal}(L_{f \oslash g}) = \mathit{coal}(L_f \oslash L_g) $$

我们定义了嵌套元组态射的**乘积可容许性 (product admissibility)** 概念, 以及当 $f$ 和 $g$ 乘积可容许时的**逻辑乘积**操作 $ f \otimes g $. 此操作与布局的逻辑乘积兼容, 即

$$L_{f \otimes g} = L_f \otimes L_g $$

通过引入**嵌套**的概念, 极大地扩展了理论框架的表达能力, 为解决扁平布局复合的难题铺平了道路.

这一节通过引入嵌套结构, 将布局的"形状"从一个刚性的整数序列, 升级为了一个灵活的, 可重构的树状结构. 这为后续通过 `refinement`, `pullback` 等操作来动态匹配不同布局的接口提供了可能.

### 5.2 复合算法 (The composition algorithm)

现在我们已经将理论推广到嵌套情况, 我们可以解释我们的**复合算法**了. 该算法使用范畴论框架来计算易处理布局 `A` 和 `B` 的组合 $B \circ A$. 我们的算法中使用了一些我们尚未讨论的重要构造, 即**共同细化 (mutual refinements)**, **拉回 (pullbacks)**, 和**前推 (pushforwards)**.

假设我们想计算布局 $A = (6, 6):(1, 6)$ 和 $B = (12, 3, 6):(1, 72, 12)$ 的复合. 由于 $A$ 和 $B$ 都是易处理的, 我们可以用元组态射来表示它们:

![图片](assets/f7528db42c0e.png)

这些态射是不可复合的, 因为 $f$ 的codomain (6, 6) 不等于 $g$ 的 domain $(12, 3, 6)$. 这意味着我们不能直接使用态射 $f$ 和 $g$ 来计算复合 $B \circ A$.

然而, 我们可以通过找到 `(6, 6)` 和 `(12, 3, 6)` 的一个**共同细化 (mutual refinement)** 来继续我们的计算, 如下图所示:

![图片](assets/fce6385359e0.png)

直观上, 这样一个共同细化是一种规范, 说明了我们如何以一种兼容的方式分解 $f$ 的codomain和 $g$ 的domain. 我们可以使用我们的共同细化将 $f$ 和 $g$ 转换为可复合的态射 $f'$ 和 $g'$.

对于 $f$ 的情况, 我们的共同细化指示我们应该将第一个 `6` 分解为 `(2, 3)`, 并在 $f$ 的codomain中包含一个额外的 `6`:

![图片](assets/501c9dab5d17.png)

严格来说, 从 $f$ 构建 $f'$ 的过程是**拉回 (pullback)** 的一个实例.

对于 $g$ 的情况, 我们的共同细化指示我们应该将 `12` 分解为 `(6, 2)`:

![图片](assets/9e1fd5373b24.png)

严格来说, 从 $g$ 构建 $g'$ 的过程是**前推 (pushforward)** 的一个实例.

嵌套元组态射 $f'$ 和 $g'$ 是可复合的, 所以我们可以形成它们的复合:

![图片](assets/5cc78779baf5.png)

计算编码后的布局得到:

$$B \circ A = L_{g' \circ f'} = ((2, 3), 6) : ((6, 72), 1) $$

Cute-DSL的例子

```
import cutlassimport cutlass.cute as cute@cute.jitdef composition_example():    """    Demonstrates basic layout composition R = A ◦ B    """    A = cute.make_layout((6, 6), stride=(1,6))     B = cute.make_layout((12, 3, 6), stride=(1, 72, 12))    R = cute.composition(A, B)    # Print static and dynamic information    print(">>> Layout A:", A)    print(">>> Layout B:", B)     print(">>> Composition R = A ◦ B:", R)composition_example()# output>>> Layout A: (6,6):(1,6)>>> Layout B: (12,3,6):(1,72,12)>>> Composition R = A ◦ B: (12,3,6):(1,72,12)
```

我们已经通过一个例子演示了复合算法的实际操作. 需要强调的是, 由于**逻辑除**和**逻辑乘积**都是根据复合来定义的, 因此该算法也可用于计算这些操作.

### 5.3 逻辑除

例如我们需要对一个矩阵进行分块运算的时候, 我们可以定义一个Layout除法, 使得一个Layout被另一个Layout划分, 这样的函数可以作为Tiling或者Layout分区的基(Basis).

逻辑除定义如下:

对于Layout $A=S:D$ 令 $M = size(A)$, B为另一个Layout, 假设是可补(Comlementation)的 $\{B,M\}$, 且是可复合的(Composition) $\{S,B\}$, 则定义逻辑除法(Logical Divide)如下:

$$A \oslash B := A \circ (B, complement(B,M))$$

在cute-DSL中的函数定义也是如此

```
def logical_divide(layoutA, layoutB):## text ommited  return composition(layoutA, make_layout(layoutB, complement(layoutB, size(layoutA))))
```

我们以一个实际的矩阵分块, 从(1024,7168)按照Tile (32,32)进行切分执行逻辑除法, 如下所示:

```
@cute.jitdef logical_divide_example():    """    Demonstrates logical divide    """    # Define the original layout    layout = cute.make_layout((1024,7168), stride=(1,1024))          # Define the tiler    tiler = cute.make_layout((32,32), stride=(32,1))         # Apply logical divide    result = cute.logical_divide(layout, tiler=tiler)        # Print results    print(">>> Layout:", layout)    print(">>> Tiler :", tiler)    print(">>> Logical Divide Result:", result)logical_divide_example()# output>>> Layout: (1024,7168):(1,1024)>>> Tiler : (32,32):(32,1)>>> Logical Divide Result: ((32,32),7168):((32,1),1024)
```

### 5.4 小结

我们可以将 `f` 的 `codomain(f)` 和 `g` 的定义域 `domain(g)` 看作是两个软件模块的**接口**. 布局组合失败的根源在于接口不匹配. 这个算法的本质, 就是通过引入一个共同的, 更细粒度的"中间接口表示" (即共同细化), 然后将 `f` 和 `g` 都"适配"到这个新接口上, 从而使它们可以连接.

这种方法展示了如下优点:

**形式化与自动化:** 它将一个需要大量人工推导和下标计算的复杂过程, 变成了一套定义明确的, 基于图 (态射) 的变换规则. 这使得整个过程可以被计算机程序**自动化**. 这正是 CuTe 库实现 `compose` 函数的理论依据.

**正确性保证:** 由于整个过程建立在范畴论的严谨框架之上, 每一步操作 (pullback, pushforward, composition) 都有其数学上的正确性保证. 最终得到的组合布局在数学上被证明是正确的. 这消除了手动计算可能引入的各种错误.

**统一性:** 如文末所说, 这个算法的核心思想是通过接口细化和适配来实现组合, 同样适用于定义和计算更复杂的操作, 如逻辑剖分 (Tiling/Partitioning) 和逻辑乘积. 这表明该理论框架具有很好的统一性和表达能力.

## 6. 与算子理论的联系

这一章阐述了布局理论与**算子理论 (theory of operads)** 之间存在一些有趣的联系. 首先描述**Tuple**范畴如何自然地作为一个算子 (operad) 的**算子范畴 (categories of operators)** 的子范畴出现. 然后我们引入profile的算子 (operad of profiles), 并提出一个嵌套元组范畴的替代定义, 该定义将细化 (refinements) 构建为"backward"的态射. 这为组合算法中围绕细化所做的许多操作提供了背景和语境. 遵循惯例, 我们将算子与其算子范畴等同起来, 例如, 交换算子 (commutative operad) 就是有限点集 (finite pointed sets) 的范畴.

### 6.1 什么是算子

为了理解这一章, 我们需要对"算子"有一个直观的认识.

算子是一种代数结构, 它专门用来描述**带有多个输入的运算以及如何组合它们**. 一个典型的例子是函数组合: 假如有函数 $f(x, y)$ 和 $g(z)$, 我们可以把 $g$ 的输出作为 $f$ 的一个输入, 形成一个新的函数 $h(z, y) = f(g(z), y)$. 算子就是对这类"组合规则"的形式化和公理化.

一个最简单的算子包含:

对每个 $n \geq 0$, 有一个 $n$ 元操作的集合 $O(n)$.

一个组合规则 $\circ_i$ 允许将一个 $m$ 元操作插入到一个 $n$ 元操作的第 $i$ 个"输入槽"中, 得到一个 $(n+m-1)$ 元操作.

满足结合律和单位元等公理.

### 6.2 将布局理论嵌入算子理论

考虑在`整除关系`下由正整数构成的`偏序集`: $\mathbb Z_{ \gt 0}$ 当且仅当 $a \leq b$ ( $a$ 整除 $b$ ). 如同任何偏序集一样, 我们可以关联一个范畴, 其对象是集合中的元素, 且当且仅当 $a \leq b$ 时存在一个态射 $a \to b$. 为简便起见, 我们也用 $\mathbb Z_{ \gt 0}$ 表示这个范畴.

现在, 将这个范畴 $\mathbb Z_{ \gt 0}^{\otimes}$ 视为在乘法运算下的一个对称幺半范畴 (symmetric monoidal category, SMC), 并应用**算子神经 (operadic nerve)** 来产生算子, 它配备了一个到有限点集范畴的函子.

我们有一个wide subcategory(保留全部对象, 但仅保留部分态射) $E_{0}^{\otimes}$, 它是有限点集上那些在基点之外是单射的映射构成的, 这个 $E_{0}^{\otimes}$ 是单个一元操作的算子. 然后, 将 $\mathbb Z_{ \gt 0}^{\otimes}$ 拉回到 $E_{0}^{\otimes}$ 上, 其结果等同于排除了条件 2c(即如果 $\alpha(i) =j \neq * $ , 则 $s_i = t_j$ ) 的 **Tuple** 范畴的定义, 而施加条件 2c 则将 **Tuple** 定义为这个拉回的一个子范畴.

#### 算子神经 (operadic nerve)

它是类似于范畴论中的Nerve[3]构造, Nerve 用于将一个范畴转化为一个单纯集合 (Simplicial Set). 一个单纯集合 `X` 不是一个单独的集合, 而是一族集合 $\{ X_n | n \geq 0\}$ 以及它们之间特定的*面映射 (face maps)* 和*退化映射 (degeneracy maps)*.

$X_0$: 0-单纯形的集合 (可以想象成*顶点/点*).

$X_1$: 1-单纯形的集合 (可以想象成*有向边/线段*).

$X_2$: 2-单纯形的集合 (可以想象成*填充好的三角形*).

$X_3$: 3-单纯形的集合 (可以想象成*填充好的四面体*).

...

$X_3$: n-单纯形的集合.

Operadic Nerve 是一种数学上的构造 (construction), 它的核心作用是将一个代数结构 (在这里是“对称幺半范畴”) 转换成一个几何/组合结构(单纯集合). 可以把它想象成一座桥梁, 连接了两个看似不同的数学世界:

桥的一端: 是带有良好乘法/积运算的代数世界.

桥的另一端: 是描述复杂组合模式的几何/组合世界, 在这个世界里, 我们可以清晰地讨论如何将多个操作以树状结构进行组合.

在本文中, 作者使用算子神经, 将 "正整数与乘法" 这个简单的代数结构, 变成了能够描述 CuTe 布局中复杂的 "形状组合" 规则的算子.

算子 (Operad)处理的是*树状/多输入组合*. 一个算子操作有多个输入和一个输出, 就像一个函数 $f(x, y, z)$. 这些操作可以像树一样嵌套组合, 例如 $h(u, v) = f(g_1(u), g_3(v, u), z)$. *算子神经* 正是将一种特定的代数结构, 即*对称幺半范畴 (Symmetric Monoidal Category, SMC)* "几何化"成一个算子.

#### 对称幺半范畴 (SMC)

一个 SMC 是一个范畴 $C$, 同时配备了一个"积"运算 $\otimes$ , 它类似于乘法. 这个 $\otimes$ 运算满足:

*结合律:* $ (A \otimes B) \otimes C $ 与 $ A \otimes (B \otimes C )$ 是"相同"的(在范畴论意义下是自然同构的).

*交换律:* $A \otimes B$ 与 $B \otimes A$ 是"相同"的.

*单位元:* 存在一个对象 $I$, 使得 $A \otimes I$ 与 $A$ 是"相同"的.

对于Layout代数, 使用的 SMC 是 $(\mathbb Z, \times)$.

*对象:* 正整数.

*态射:* 如果 $a$ 整除 $b$, 就有一个态射 $a \rightarrow b$.

*积运算 $\otimes$:* 就是普通的整数乘法.

*单位元:* 整数 `1`.

算子神经作用于一个 SMC, 如 $(\mathbb Z_{\gt 0}, \times)$ 时, 它生成的算子中的 $n$ 元操作, 对应于用 $\otimes$ 运算将 $n$ 个对象组合起来的所有可能方式. 这些"组合方式"可以用`树`来可视化. SMC 中的 `n` 个对象 $A_1, \ldots , A_n$ ,用 $\otimes$ 运算将它们组合, 组合后的结果 $A_1 \otimes A_2 \otimes \ldots \otimes A_n$.

算子神经将这种 "组合树"的结构几何化 了. 在它生成的算子范畴中: *对象*是原 SMC 中的对象 (例如, 正整数). 一个 $n$ 元态射 (n-ary morphism) $A_1, \ldots , A_n \rightarrow B$ 对应于一个从 $A_1 \otimes A_2 \otimes \ldots \otimes A_n$ 到 $B$ 的态射. 在 $(\mathbb Z_{\gt 0}, \times)$ 的例子中, 这就是一个整除关系 $t_1 \times \ldots \times t_n$ 整除 $b$

进行这个构造的目的是为了*从最基本的数学原理出发, 构建出能够描述 CuTe 布局的范畴*.

*起点:* 作者从最简单的结构 $\mathbb Z^{\otimes}$ 出发. 这代表了布局 `shape` 维度可以被乘起来合并 (`coalesce`) 这一基本事实.

*应用算子神经:* 对 $\mathbb Z^{\otimes}$ 应用算子神经, 得到了算子 $\mathbb Z^{\otimes}$. 这个新生成的算子是一个非常丰富的结构, 它内在地包含了所有关于"将一串数字通过乘法组合起来"的规则.

$\mathbb Z^{\otimes}$ 中的一个`n`元操作, 就对应于一个 $n$ 维的`shape`元组 $(s_1, \ldots , s_n)$.

$\mathbb Z^{\otimes}$ 中的组合, 就对应于将 `shape` 元组进行嵌套和合并. 例如, 将 $(s_1,s_2)$ 和 $s_3$ 组合成 $(s_1 \times s_2,s_3)$.

*构造 Tuple范畴:* 这个结构过于庞大和通用. CuTe 的布局态射还有额外的约束, 比如"一个输入维度最多映射到一个输出维度" (单射性). 因此, 作者通过拉回 (pullback)的技巧, 从 $\mathbb Z^{\otimes}$ 中"裁剪"出了一个子结构, 这个子结构恰好就是前面定义的 *Tuple范畴*.

从这个角度来看, 我们又该如何整合 profiles 呢? profile本身构成一个(单色的, 对称的)算子, 其 $n$ 元操作的集合由长度为 $n$ 且对称群作用平凡的profile组成. 在算子神经下, 将此算子表示为 $P^{\otimes}$. 然后, 通过构造算子的拉回, 就可以考虑在任何对称幺半范畴 $C^{\otimes}$ 中带有标签的profile; 对于 $\mathbb Z_{\gt 0}^{\otimes}$, 我们将得到的拉回表示为 $P\mathbb Z_{\gt 0}^{\otimes}$. 于是, 考虑深度为1的profile, 我们同样得到 **Tuple** 是 $P\mathbb Z_{\gt 0}^{\otimes}$ 在 $E_{0}^{\otimes}$ 上的拉回的一个子范畴 (实际上, 也是 $P\mathbb Z_{\gt 0}^{\otimes}$ 自身的子范畴).

此外, 由于 $P\mathbb Z_{\gt 0}^{\otimes}$ 同时包含了元组态射和细化 (因为整数乘法是幺半积), 它是一个合适的范畴可以在其中进行更复杂的构造. 具体来说, 正如我们在组合算法中出现的图中所看到的, 将因式分解后面跟着元组态射的过程, 视为其本身就构成了某个范畴中的态射.

范畴论中有一个标准的构造可以为我们做到这一点; 即, 我们可以在 $P\mathbb Z_{\gt 0}^{\otimes}$ 中构造一个特定的Span范畴[4]. 在这个范畴中, 前向态射的类是wide subcategory **Tuple** 中的态射, 而后向态射的类 **Ref** 由那些在有限点集映射 $ \alpha: { \ast, 1, \ldots, m } \to { \ast, 1, \ldots, n} $ 上的cocartesian edges[5] 组成, 这些映射 $\alpha$ 满足:

$\alpha$ 是active的: 如果 $\alpha(i)=* $, 那么 $i = * $.

$\alpha$ 是满射的 (surjective).

当限制在 ${1, \ldots, m}$ 上时, $\alpha$ 是非递减的 (non-decreasing).

简单的解释如下:

active: 不丢弃任何维度. 每个输入维度都必须对输出维度有贡献.

surjective:  输出的每个维度都必须由至少一个输入维度构成. 不会产生"无源"的输出维度.

Non-decreasing: 保持维度的相对顺序, 避免交错.

在这里, 在这类映射上取余笛卡尔边的效果, 意味着我们考虑的是例如映射 $(a, b) \rightarrow c$ 其中 $ab = c$, 而不是 $ab$ 整除 $c$ 的一般情况 . 也就是说, 我们得到的精确是细化(refinement).

注意, 为了让Span构造是良定义(well-defined)的, 我们需要检查在 $P\mathbb Z_{\gt 0}^{\otimes}$ 中, **Tuple** 中的态射沿着 **Ref** 中的态射的拉回是可以形成的. 这一点可以被证明.

最后, 我们将得到的Span范畴表示为 **Span(Tuple, Ref)**. 这个范畴中的一个典型态射看起来像这样:

![图片](assets/da05b704bf82.png)

这里, 遵循Span图的标准记法, 我们现在将左边的箭头组画成了指向后方的箭头.

根据定义, **Span(Tuple, Ref)** 包含 $\text{Tuple}$ 和 $\text{Ref}^{\text{op}}$ (Ref 的反范畴) 作为子范畴. 然后我们可以将实现函子从 **Tuple** 扩展到 **FinSet**, 使其作用于整个 **Span(Tuple, Ref)** 范畴, 这样**细化 (refinements)** 就被映射到*逆余字典序同构 (inverse colexicographic isomorphisms)*.

从概念上讲, 这为嵌套元组的范畴提供了一个替代性的视角, 因为与 **Nest** 范畴相反, **Span(Tuple, Ref)** 范畴的**对象**是扁平元组, 但**态射**是嵌套的; 这与将一个布局看作是定义了一个映射, 其值域是其形状的深度1规约(reduction)的观点是一致的.

### 6.3 一些额外的分析

作者通过一系列精巧的数学构造, 将 `Tuple` 范畴嵌入到算子的世界中. 让我们一步步拆解这个过程:

首先是构造 **基础结构: $\mathbb Z_{ \gt 0}^{\otimes}$**:

作者的出发点是正整数集合, 配合"乘法"运算. 这构成了一个**幺半群 (monoid)**.通过 " 当且仅当 a 整除 b" 这个关系, 作者把它变成了一个**偏序集**, 进而是一个**范畴**. 将这个范畴看作一个**对称幺半范畴**, 这里的"幺半积"就是整数乘法 $\times$.

接下来**构造算子: 算子神经 (Operadic Nerve)**

这是一个标准的技术, 可以从任何对称幺半范畴 (如 $(\mathbb Z_{\gt 0}, \times)$ ) 构建出一个算子. 我们得到的算子 $\mathbb Z_{\gt 0}^{\otimes}$ 直观上可以这样理解:

一个 `n` 元操作就是 `n` 个正整数的元组 $(s_1, \ldots , s_n)$.

组合规则就是将一个元组插入到另一个元组中, 并在底层通过整数乘法来合并 `shape` 值. 这恰恰就是布局中"细化"或"合并"操作的雏形.

然后是**拉回 (Pullback) 构造**, 作者并未直接使用这个算子, 而是通过**拉回**操作来施加约束, 从而精确地构造出 `Tuple` 范畴.`Tuple` 范畴中的态射 $f$ 有一个关键属性: 非基点映射是**单射** (一个输入维度最多映射到一个输出维度). 作者找到了编码这个属性的算子 (只含一元操作的算子).

通过将 $\mathbb Z_{\gt 0}^{\otimes}$ 拉回到 $E_0^{\otimes}$ 上, 得到的新结构就自动满足了单射属性. 再加上 $s_i = t_j$ 这个条件, 就重构了 `Tuple` 范畴. 这一步的意义在于: 它说明 `Tuple` 范畴不是一个凭空捏造的结构, 而是由两个更基本的结构 ("乘法组合规则" 和 "单射映射规则") 通过标准的范畴论构造 (拉回) 自然产生的.

最后**引入Profile** , Profile本身也构成一个算子 $P^{\otimes}$, 它的操作就是"如何将 `n` 个东西组合成一个嵌套结构". 通过再次使用拉回, 将 $\mathbb Z_{\gt 0}^{\otimes}$ 和 $P^{\otimes}$ 结合起来, 得到 $P\mathbb Z_{\gt 0}^{\otimes}$. 这个新的算子同时编码了**数值的乘法组合**和**结构的嵌套组合**. `Nest` 范畴就生活在这个更丰富的世界里.

然后进一步引入**Span范畴**

在第五章, 为了复合 $f$ 和 $g$, 我们需要先找到一个"共同细化", 然后对 $f$ 做 `pullback` , 对 $g$ 做 `pushforward`. 这个过程看起来像是在范畴之外进行了一些"手工操作".

作者提出, 我们可以构造一个新的范畴 `Span(Tuple, Ref)`, 在这个范畴里, 态射本身就包含了"细化"的过程.

一个**Span** 是形如 $X \leftarrow S \rightarrow Y$ 的一对态射. 在 **Span(Tuple, Ref)** 中, 一个态射 $A \rightarrow B$ 被定义为一个Span, 其中:

 来自于 **Ref** 范畴, 代表**细化 (Refinement)** 操作 (例如, 将 `(36)` 细化为 `(6,6)`).

$\rightarrow$ 来自于 **Tuple** 范畴, 代表我们之前定义的**元组态射**.

直观上, 这个新范畴中的一个态射就代表了 "先对输入进行一次细化, 然后再进行一次元组映射" 这一整个过程. 复合算法中 $f$ 到 $f'$ 的变换, 就可以看作是在 `Span(Tuple, Ref)` 范畴中, 将 $f$ 与一个代表细化的态射进行组合.

通过 `Span` 范畴, 第五章中为了组合 $f$ 和 $g$ 而进行的"适配"操作, 现在变成了在新范畴中进行一次标准的态射组合.

所有操作都被内化到了一个统一的范畴框架内. **细化**操作在`实现函子`下对应于*逆余字典序同构*, 而**元组态射** 对应于*布局函数*. 这一对偶关系非常优雅, 揭示了数据布局背后深刻的数学对称性.

## 7. 结尾

基于范畴论视角下的Cute Layout代数更加直观, 可以通过图上的箭头完成一系列复杂的Stride运算. 同时将其嵌入到算子理论, 它将 CuTe 布局理论追溯到了算子理论这一更普适的数学分支, 赋予了其深刻的代数背景.

虽然这篇文章的数学语言非常抽象, 但它传达了一个重要的信息: 在设计用于处理复杂硬件的编程模型时, 寻找其背后正确的数学抽象是至关重要的. 一个看似复杂的工程问题 (如为不同的 Tensor Core 架构生成最优的数据分片和加载模式), 可能在一个合适的数学框架下, 会呈现出惊人的简洁性和规律性.

CuTe 的成功, 以及这篇文章为其提供的理论基础, 再次证明了从抽象代数到计算机体系结构的这条路径是多么富有成效. 这也为未来设计能够自动适应新硬件的编译器和库指明了方向: **去寻找并利用问题背后隐藏的数学结构**.

这也是我一直以来另一个工作的方式, 期望通过范畴论的视角来寻找大模型结构. 正如在[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)专题所讲:

这一次人工智能革命的数学基础是：范畴论/代数拓扑/代数几何这些二十世纪的数学第一登上商用计算的舞台。

正是大概十年前这个观点的引导下, 逐渐的完成了相关学科的学习, 未来大家一起努力吧.

参考资料

[1] 
Categorical Foundations for CuTe Layouts: *https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/*
[2] 
representable functor: *https://ncatlab.org/nlab/show/representable+functor*
[3] 
Nerve_(category_theory): *https://en.wikipedia.org/wiki/Nerve_(category_theory)*
[4] 
Span范畴: *https://ncatlab.org/nlab/show/span*
[5] 
cocartesian edges: *https://ncatlab.org/nlab/show/Cartesian+morphism*