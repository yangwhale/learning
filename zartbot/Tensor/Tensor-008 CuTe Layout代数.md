# Tensor-008 CuTe Layout代数

> 作者: zartbot  
> 日期: 2024年8月28日 20:57  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492220&idx=1&sn=4ec36b34df55ae6c0b643709da3316e1&chksm=f995f2becee27ba8a259e48c51c8197dd804a28932236c2ee1cf3bb11f697aef8dea2a414938#rd

---

CuTe Layout代数可能是大家对Cutlass最困扰的一块内容,但又是非常巧妙的一种工具, 接下来我们详细对这块内容进行分析, 本文目录如下.

```
 0. 为什么需要Layout代数  1. Layout 代数概述 1.1 Layout定义 1.2 Layout函数  2. 合并(Coalesce) 2.1 合并操作概述 2.2 合并的代数规则 2.3 按Mode合并 3. 复合(Composition) 3.1 可复合条件 3.1.1 左整除(Left divisible) 3.1.2 可复合条件 3.1.3 复合实例 3.1.4 按Mode复合 3.1.5 复合小结  4. 补集(Complementation) 4.1 补集定义 4.2 补集示例  5. 除法 5.1 除法定义 5.2 1-D逻辑除 5.3 2-D逻辑除 5.4 Zipped,Tiled,Flat Divides  6. 乘法 6.1 逻辑乘法 6.2 1D-逻辑乘法 6.3 2D-逻辑乘法 6.3.1 Blocked乘法 6.3.2 Raked乘法 6.3.3 Zipped /Tiled Product   
```

## 0. 为什么需要Layout代数

我们首先来看看在很多大模型的计算过程中对张量Layout进行变换的需求, 以Multi-Head-Attention为例, 它涉及到按照n_head拆分以及在相应的维度进行转置并构建back-to-back的GEMM乘法.

```
        # Pass through the pre-attention projection: b x lq x (n*dv)        # Separate different heads: b x lq x n x dv        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)        # Transpose for attention dot product: b x n x lq x dv        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
```

对于这些算子,我们期望有一种能够更容易构建的可复合(Composible)的方式来构建, 例如对于Q矩阵,本质上是的复合.

`转置`: 在Tensor-007中已经介绍了通过Stride可以改变其Layout形式构成.

`Multi-Head拆分`: 本质上是可兼容的(Compatible) Shape之间的映射.

通过对Shape和Stride的复合,我们可以构造成一个Layout对象, 然后对于其上的不同Layout变换操作构成一个封闭的运算. 然后当这些运算时可复合时, 我们就可以构建出一个Layout范畴. 然后我们可以在Layout上的定义Transpose, Concate等一系列函数.

资料参考:《A note on the algebra of CuTe Layouts》[1] 《CuTe Layout Algebra》[2]

## 1. Layout 代数概述

### 1.1 Layout定义

对于整数, Layout , 其中为一个整型元组(IntTuple)被称为Layout的Shape, 而被称为Layout的Stride.

`Size`: 定义,它是Shape中各个Dim的乘积.

`Length`: 定义L的Length为

`Mode` : 对于, 是一个length=1的Layout,我们将其称为的Mode.

对于和可以将其定义为`IntTuple`对象, 并在其之上定义`Group`和`Flatten`或`Coalesce`操作. 例如:

对于任意一个Layout  都可以分解(Decompose)成它自己的Mode, 我们可以定义  为自然数的一个子集, 其中. 那么可以存在如下同构:

即我们还可以构造一个`Coalesce`合并操作,但合并操作和Layout Stride有关, 后面第二章会详细解释.

对于给定的作为坐标(Coord)可以映射为如下形式

例如针对Shape  构成的映射如下

xh-D
xh-D`0``(0,(0,0))`
`9``(0,(1,1))``1``(1,(0,0))`
`10``(1,(1,1))``2``(2,(0,0))`
`11``(2,(1,1))``3``(0,(1,0))`
`12``(0,(0,2))``4``(1,(1,0))`
`13``(1,(0,2))``5``(2,(1,0))`
`14``(2,(0,2))``6``(0,(0,1))`
`15``(0,(1,2))``7``(1,(0,1))`
`16``(1,(1,2))``8``(2,(0,1))`
`17``(2,(1,2))`

对于任意两个Layout , 它们可以构造出一个Concatenation , 即对于元组我们都可以通过`flatten`展平然后连接后再进行排序和`group`操作构成一个新的Layout. 类似的, 定义多个Layout的Concate操作构成

### 1.2 Layout函数

对于上的Layout函数 可以定义为如下可复合的形式:

从另一个角度看, 由多个多重线性函数复合而成, 其中第一个箭头表示同一个Shape的不同Stride.

即坐标和Stride的内积构成index.

广义来看, 我们可以定, 将定义中的替换为, 则其复合性可以表示为

则

因此对于坐标x我们可以有多种维度的映射

![图片](assets/e2686568e2e2.png)

## 2. 合并(Coalesce)

### 2.1 合并操作概述

正如我们在第一章所述, 任何Layout都可以分解成它自己的Mode, 如下所示:

我们定义在Layout上的函数 即, 通过合并操作可以节省我们在高维矩阵上内存寻址的时间, 变成一个低维度的坐标映射.

例如针对一个高维度的矩阵, 合并实例如下

```
template<class T>void print_coalesce(T layout) {    printf("H-Layout  :");    print(layout);    printf("\nCoalesce-Layout :");    print(coalesce(layout));    printf("\n");}int main(){    Layout a0 = make_layout(Shape<_2,_4>{},Stride<_1,_2>{});    print_coalesce(a0);        auto s1 = Shape<_2,Shape<_3,_4>>();    auto s2 = Shape<_5,Shape<_6,_7>>();    auto s3 = make_shape(s1,s2);        Layout a_col = make_layout(s3, GenColMajor{});  //GenColMajor == LayoutLeft    print_coalesce(a_col);    Layout a_row = make_layout(s3, GenRowMajor{});  //GenRowMajor == LayoutRight    print_coalesce(a_row);}//outputH-Layout  :(_2,_4):(_1,_2)Coalesce-Layout :_8:_1H-Layout  :((_2,(_3,_4)),(_5,(_6,_7))):((_1,(_2,_6)),(_24,(_120,_720)))Coalesce-Layout :_5040:_1H-Layout  :((_2,(_3,_4)),(_5,(_6,_7))):((_2520,(_840,_210)),(_42,(_7,_1)))Coalesce-Layout :(_2,_3,_4,_5,_6,_7):(_2520,_840,_210,_42,_7,_1)
```

没有合并前, 我们需要以`(_2,_4)`二维坐标或者`((_2,(_3,_4)),(_5,(_6,_7)))`高维坐标进行寻址, 合并后即可以通过单个维度进行迭代. 但是并不是所有的Layout都可以合并到单个维度, 例如a_row

### 2.2 合并的代数规则

但是并不是所有的操作都可以合并到一维空间的,它和实际的Stride相关, 某些Stride情况下会有重复的地址访问, 因此不能合并. 我们只考虑两个Layout `s0:d0`和`s1:d1`的合并规则, 记合并算符为`s0:d0 ++ s1:d1`

对于任何Shape为`_1`的合并,满足`s0:d0  ++  _1:d1  =>  s0:d0`和`_1:d0  ++  s1:d1  =>  s1:d1`

如果第二个Layout的Stride `d1`是第一个Layout的Shape和Stride的乘积`d1 = s0 * d0`, 则可以合并成如下形式:`s0:d0  ++  s1:d1 = s0:d0  ++  s1:s0*d0  =>  s0*s1:d0`

其它Layout合并,则需要区分对待`s0:d0  ++  s1:d1  =>  (s0,s1):(d0,d1)`

我们注意到广义列优先(GenColMajor, 也被称为LayoutLeft)是CuTe Layout的默认Layout形式,

记 , 记,
其Layout Stride为 , 满足规则2 , 因此正如上例中的a_col Layout是可以递归合并到一个维度的.

### 2.3 按Mode合并

某些时候, 我们需要聚合一些维度, 保持其它Layout维度不变. 例如`((_2,(_3,_4)),(_5,(_6,_7)))`我们想聚合`(_2,(_3,_4))`, 可以利用step迭代器, 示例如下

```
    auto s1 = Shape<_2,Shape<_3,_4>>();    auto s2 = Shape<_5,Shape<_6,_7>>();    auto s3 = make_shape(s1,s2);        Layout a_col = make_layout(s3, GenColMajor{});      auto result = coalesce(a_col, Step<_1,_1>{});   //(_24,_210):(_1,_24)     //它等同于对layout<0>(a_col)和layout<1>(a_col)进行聚合后合并    auto same_r = make_layout(coalesce(layout<0>(a_col)),                          coalesce(layout<1>(a_col)));    //同理我们还可以构造其它Step聚合    auto b1 = coalesce(a_col,Step<_1,Step<_1,_1>>{}); //(_24,(_5,_42)):(_1,(_24,_120))
```

## 3. 复合(Composition)

接下来我们讨论Layout A和Layout B的复合操作, 例如我们需要对一块输入数据基于Thread Layout来分配变量, 即我们可以构建如下复合,然后通过Thread-Id索引来获取相应的数据, 如下图所示:

![图片](assets/6aa53eb6938d.png)

本质上**复合表示在Layout A中按照Layout B的规则选择一些坐标,构成新的Layout** 对于A和B的复合本质上是定义关联的Layout函数所满足的条件, 也就是说对于坐标c

其中B'为和B兼容的Shape. 为了简单起见, 我们考虑形状中不包含`_1`的情况.

### 3.1 可复合条件

#### 3.1.1 左整除(Left divisible)

令为正整数, 且给定一个分解. 将扩展到维度, 并且认为可以被任意正整数整除. 令

如果存在, 使得

(1) 

(2) 当(1)成立时, 令. 如果,另外要求

(3) 对于(2)当, 另外要求

则我们称可以被 `左整除`(left divisible), 满足如上(1)(2)条件,不需要满足(3),则称可以被`弱左整除`(weakly left divisible).

注意, 如果i存在则一定是唯一的. 我们记,  并称 i 为 division-index.

对于左整除, 我们赋予如下诱导分解:

(a) 当,  ,且满足, 对于 满足 

(b) 若, 令, 并且

#### 3.1.2 可复合条件

我首先来考虑在复合时, 第二个矩阵B只有1维的情况.

`定义3.1`令Layout A的Shape为, Layout , 如果满足如下条件,则{S,B}可复合:

(1) M可以被r左整除, 记

(2) 对于它的诱导分解可以被弱左整除.

可复合需要“沿着A的mode除以B”, 对于可复合性更准确的定义如下:

`定义3.2`:设Layout A的Shape为,Stride为任意元组.Layout 的Length=1. 令, , divide index .

当时,

如果,  ,否则 , 令

如果, 则 , 但,则

我们可以注意到,最终复合的矩阵. `定义3.2`满足时, .

对于Layout  即length=1的情况, 在Cutlass一个更加直观的解释, 即在A的Shape中按照`r`作为跨步取前`N`个元素. 也就是说对于A的前若干个Shape维度之积能够整除`r`, 而后继维度向前的累积能够整除`r`. 并且保证在跨步`rN`后也可以有类似的整除机制. 即对于A的Shape, 记, 则

并且对于要取前`N`个元素,且跨步为`r`,亦有

对于B的Stride r和A的Shape 是否满足条件在Cutlass中采用`shape_div`计算, 即从左到右, 对A的Shape的N个维度,逐渐的除以StrideB. 而对于Shape B的条件采用`shape_mod`计算(注:官方文档的shape_mod计算是存在一定问题的,后面详细解释)

另一个问题是, 对于动态类型的Layout Composition Cutlass出于性能考虑并没有进行可复合性检查, 因此会有计算错误发生将无法复合的Layout产生错误输出. 实际测试,我们可以通过如下方式静态定义, 并进行编译,在编译期进行检查.

```
    auto s2 = make_shape(Int<2>{}, Int<6>{}, Int<10>{},Int<14>{});    auto a2 = make_layout(s2, GenRowMajor{});    auto b2 = make_layout(make_shape(Int<60>{}), make_stride(Int<4>{}));    auto c2 = composition(a2, b2);    print(c2);
```

可以看到 , 则 , , . 而对于, , , 则亦满足条件. 输出Layout为((_3,_10,_2)):((_280,_14,_1)).

例如我们修改B的SHAPE为70, 虽然但是 复合在编译期检查失败

```
      static_assert(IntTupleA::value % IntTupleB::value == 0 || IntTupleB::value % IntTupleA::value == 0, "Static shape_div failure");      ^          detected during:            instantiation of "auto cute::shape_div(const IntTupleA &, const IntTupleB &) [with IntTupleA=cute::C<70>, IntTupleB=cute::C<3>]" at line 1034 of /opt/cutlass/include/cute/layout.hpp            instantiation of function "lambda [](const auto &, const auto &)->auto [with <auto-1>=cute::tuple<cute::tuple<cute::C<1>>, cute::C<70>>, <auto-2>=cute::C<3>]" at line 422 of /opt/cutlass/include/cute/algorithm/tuple_algorithms.hpp            instantiation of "decltype(auto) cute::detail::fold(T &&, V &&, F &&, cute::seq<I, Is...>) [with T=const cute::tuple<cute::_1, cute::C<3>, cute::C<10>> &, V=cute::tuple<cute::tuple<cute::C<1>>, cute::C<70>>, F=lambda [](const auto &, const auto &)->auto &, I=1, Is=<2>]" at line 424 of /opt/cutlass/include/cute/algorithm/tuple_algorithms.hpp            instantiation of "decltype(auto) cute::detail::fold(T &&, V &&, F &&, cute::seq<I, Is...>) [with T=const cute::tuple<cute::_1, cute::C<3>, cute::C<10>> &, V=cute::tuple<cute::tuple<>, cute::C<70>>, F=lambda [](const auto &, const auto &)->auto &, I=0, Is=<1, 2>]" at line 441 of /opt/cutlass/include/cute/algorithm/tuple_algorithms.hpp            instantiation of "auto cute::fold(T &&, V &&, F &&) [with T=const cute::tuple<cute::_1, cute::C<3>, cute::C<10>> &, V=cute::tuple<cute::tuple<>, cute::C<70>>, F=lambda [](const auto &, const auto &)->auto]" at line 1035 of /opt/cutlass/include/cute/layout.hpp            instantiation of "auto cute::detail::composition_impl(const LShape &, const LStride &, const RShape &, const RStride &) [with LShape=cute::tuple<cute::C<2>, cute::C<6>, cute::C<10>, cute::C<14>>, LStride=cute::tuple<cute::C<840>, cute::C<140>, cute::C<14>, cute::_1>, RShape=cute::C<70>, RStride=cute::C<4>]" at line 987 of /opt/cutlass/include/cute/layout.hpp
```

但是我们注意到Cutlass的检查方式是有更严格的, 例如在引用资料1中所述, 对于如下Composition, 满足定义3.1的条件C=(2,3):(9,5)但是Cutlass检查失败

```
    auto s2 = make_shape(Int<4>{}, Int<6>{}, Int<8>{},Int<10>{});    auto a2 = make_layout(s2, make_stride(Int<2>{}, Int<3>{}, Int<5>{},Int<7>{}));    auto b2 = make_layout(make_shape(Int<6>{}), make_stride(Int<12>{}));    auto c2 = composition(a2, b2);    print(c2);
```

我们再扩展到B的Layout为多个元组的情况, 对于多维坐标和相应的Stride构成的内积作为index, 即对于

对于坐标,

那么, 如果满足左分配率, 有 那么就能够退化到B为一维的情况, 即

下面我们来分析满足左分配率的充分条件

对于每个, 都要满足可复合条件.

B需要满足一些额外的条件. 下面进行详细分析.

对于 的像的凸闭包为, 排除0点情况它在A中的交集为

那么为了保证复合函数, 不光需要保证为单射, 还需要保证是不相交的. 对于函数的复合通过范畴论的交换图表示

![图片](assets/66a9bc20e04c.png)

但是我们考虑复合的情况, 注意到下半部通常是非交换的

![图片](assets/69ad68c23b8d.png)

对于如下一个因式分解

需要保证对于的像 在分割下是不相交的.

对于可加性, 对于两个不相交的的非零点 , ,
其坐标在 分割下,  , 需要保证加法不越界, 即 

**注: 原有的CuTe左分配率的检查仅要求B满足单射是有问题的.**

#### 3.1.3 复合实例

例如我们通过复合Layout构建的Reshape `20:2  o  (5,4):(4,1)`, Layout`20:2`即对于一个向量`[0,2,4,...,38]`,我们要将它Reshape成`(5,4)`的矩阵.首先按照左分配率转换为`(20:2 o 5:4, 20:2 o 4:1)`. 对于`20:2 o 5:4 => 5:2*4 => 5:8`, 对于`20:2 o 4:1 => 4:2`. 因此`(20:2 o 5:4, 20:2 o 4:1) => (5:8,4:2)`按照Concate元组规则`(5:8,4:2)=>(5,4):(8,2)`.
代码测试如下:

```
    auto sa = make_shape(Int<20>{});    auto a = make_layout(sa, Stride<_2>{});    auto sb = make_shape(Int<5>{}, Int<4>{});    auto b = make_layout(sb, make_stride(Int<4>{}, Int<1>{}));    Tensor tb =make_tensor(A, b);    print_tensor(tb);        auto c = composition(a, b);    Tensor tc =make_tensor(A, c);    print_tensor(tc);    //outputptr[32b](0x563fa0d065d0) o (_5,_4):(_4,_1):    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19ptr[32b](0x563fa0d065d0) o (_5,_4):(_8,_2):    0    2    4    6    8   10   12   14   16   18   20   22   24   26   28   30   32   34   36   38
```

#### 3.1.4 按Mode复合

CuTe支持通过取Layout的某些Mode进行复合操作, 例如仅需要对矩阵中某些维度的数据进行整形, 例如MultiHeadAttention中的按Head拆分等场景.

在Cutlass中, 将多个Layout组合构成的元组称为Tile, 通过`make_tile`函数构建, 并记为<Layout1,Layout2...,LayoutN>, 注意它和Concate的区别. 当composition的参数B是Tiler时, 则按照如下取mode的方式进行复合.等同于每个Mode复合Tiler里相应的Layout.

```
// (12,(4,8)):(59,(13,1))auto a = make_layout(make_shape (12,make_shape ( 4,8)),                     make_stride(59,make_stride(13,1)));// <3:4, 8:2>auto tiler = make_tile(Layout<_3,_4>{},  // Apply 3:4 to mode-0                       Layout<_8,_2>{}); // Apply 8:2 to mode-1// (_3,(2,4)):(236,(26,1))auto result = composition(a, tiler);// Identical toauto same_r = make_layout(composition(layout<0>(a), get<0>(tiler)),                          composition(layout<1>(a), get<1>(tiler)));
```

另外Cute也可以将Shape元组解释为Tiler, 默认将Stride赋值为1, 如下示例:

```
// (12,(4,8)):(59,(13,1))auto a = make_layout(make_shape (12,make_shape ( 4,8)),                     make_stride(59,make_stride(13,1)));// (8, 3)auto tiler = make_shape(Int<3>{}, Int<8>{});// Equivalent to <3:1, 8:1>// auto tiler = make_tile(Layout<_3,_1>{},  // Apply 3:1 to mode-0//                        Layout<_8,_1>{}); // Apply 8:1 to mode-1// (_3,(4,2)):(59,(13,1))auto result = composition(a, tiler);
```

#### 3.1.5 复合小结

复合运算的作用是巨大的, 对于Composition函数的参数B,可以为Layout, 也可以多个Layout构成的Tile以及多个Shape构成的Tile按Mode运算. 对于矩阵分块(MxNxK -> 3x5x8 sub block)或者对于一个8x16矩阵展平成1-D向量然后重新排列成32x4的块等操作都可以通过composition表达.后面我们即将看到相应的实例.

### 4. 补集(Complementation)

复合表示在Layout A中按照Layout B的规则`选择`一些坐标,构成新的Layout. 对于那些`没有被选择的坐标` 我们需要一种表示方法.例如下图所示:

![图片](assets/476b78b4eac6.png)

我们把Layout看作是坐标的一个函数, 则原始坐标为定义域(domain),而对应的像为陪域(codomain).

cosize可以视作陪域所占用的物理内存空间, 定义为

例如`(_5,_4):(_4,_2)`, 其size=5x4=20, .

```
    0    2    4    6    4    6    8   10    8   10   12   14   12   14   16   18   16   18   20   22
```

### 4.1 补集定义

在Cutelass的文档中, 对于一个Layout 相对于Shape 的补集 定义如下:

(1) 和的codomain是不相交的, 对于

(2) 是有序的, 即R的Stride 为正且递增, 使得是一个严格递增的函数.

(3) 在M下是有界的, 即R的size和cosize受size(M)限制.  , 

**可补条件**:

对于严格递增, 我们可以对进行置换reindex, 使得中Stride递增, 且. 对于正整数 满足如下条件,则是可补的.

(1) 

(2) 

当是可补的, 补集如下

### 4.2 补集示例

例如官方示例complement((2,2):(1,6), 24),  带入可得

```
#define MAXN  128*128int main(){    // initial memory with physical layout    int* A = (int*)malloc(MAXN * sizeof(int));    for(int i =0 ; i < MAXN ; i++){     A[i]=int(i);    }       auto sa = make_shape(Int<2>{},Int<2>{});    auto a = make_layout(sa, Stride<_1,_6>{});    Tensor ta =make_tensor(A, a);    print_tensor(ta);    auto c = complement(a, 24);    Tensor tc =make_tensor(A, c);    print_tensor(tc);   }\\outputptr[32b](0x562d2ed985d0) o (_2,_2):(_1,_6):    0    6    1    7ptr[32b](0x562d2ed985d0) o (_3,2):(_2,_12):    0   12    2   14    4   16
```

可视化如下图所示:

![图片](assets/b623450cc25a.png)

其实这里面就有了Layout除法的影子了. 除法是最常用的矩阵分块的算法, 但是GTC上的PPT确实没把事情讲清楚

![图片](assets/c78e88022bcf.png)

后面我们就详细阐述一下.

## 5. 除法

我们可以定义一个Layout除法, 使得一个Layout被另一个Layout划分, 这样的函数可以作为Tiling或者Layout分区的基(Basis).

### 5.1 除法定义

对于Layout  令,  B为另一个Layout, 假设是可补(Comlementation)且可复合(Composition), 则定义逻辑除法(Logical Divide)如下

逻辑除法具体代码实现实际上是一个复合函数

```
template <class LShape, class LStride,          class TShape, class TStride>auto logical_divide(Layout<LShape,LStride> const& layout,                    Layout<TShape,TStride> const& tiler){  return composition(layout, make_layout(tiler, complement(tiler, size(layout))));}
```

它构成两个mode,  本质上B将A划分为两个mode.

第一个mode    为B指向的所有元素

第二个mode   是B所有未指向的元素

我们可以将其理解为如果是一个Tiler, 则是Tile的Layout.

### 5.2 1-D逻辑除

考虑Layout A = (4,2,3):(2,1,8), Tiler B=4:2, 即在A内部按照Stride 2取4个元素.

如下图所示:

![图片](assets/5c0952c8a809.jpg)

相关的测试代码如下所示:

```
#include <getopt.h>#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define MAXN  128*128int main(){    // initial memory with physical layout    int* A = (int*)malloc(MAXN * sizeof(int));    for(int i =0 ; i < MAXN ; i++){     A[i]=int(i);    }       //layout-a    auto sa = make_shape(Int<4>{},Int<2>{},Int<3>{});    auto da = make_stride(Int<2>{},Int<1>{},Int<8>{});    auto a = make_layout(sa, da);    Tensor ta =make_tensor(A, a);    printf("\nLayout A: ");    print_tensor(ta);    //layout-a    auto sb = make_shape(Int<4>{});    auto db = make_stride(Int<2>{});    auto b = make_layout(sb, db);    Tensor tb =make_tensor(A, b);    printf("\nLayout B: ");    print_tensor(tb);    auto b_star = complement(b, size(a));    Tensor tb_star =make_tensor(A, b_star);    printf("\nLayout B*: ");    print_tensor(tb_star);    auto c1 = composition(a,b);    Tensor tc1 =make_tensor(A, c1);    auto c2 = composition(a,b_star);    Tensor tc2 =make_tensor(A, c2);    printf("\nLayout A o B: ");    print_tensor(tc1);    printf("\nLayout A o B*: ");    print_tensor(tc2);    auto d = logical_divide(a,b);    Tensor td =make_tensor(A, d);     printf("\nLayout A div B: ");    print_tensor(td);}
```

### 5.3 2-D逻辑除

由前一节Tiler的定义, 我们可以将其推广到高维空间. 我们可以使用类似组合的方式, 将logical_divide作用在不同的维度上. 例如一个2D Layout `A = (9,(4,8)):(59,(13,1))`,对于列(Mode-0)的Shape 9 希望按照`3:3`的方式切分, 而针对行(mode-1)按照 `(2,4):(1,8)`矩阵切分. 我们记为`B = <3:3, (2,4):(1,8)>` 对于by-mode, 我们需要采用make_tile函数构建,代码如下:

```
#include <getopt.h>#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define MAXN 128 * 128int main(){    // initial memory with physical layout    int *A = (int *)malloc(MAXN * sizeof(int));    for (int i = 0; i < MAXN; i++)    {        A[i] = int(i);    }    // A: shape is (9,32)    auto layout_a = make_layout(make_shape(Int<9>{}, make_shape(Int<4>{}, Int<8>{})),                                make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{})));    Tensor ta = make_tensor(A, layout_a);    printf("\nLayout Tensor A: ");    print_tensor(ta);    // B-Tile < 3:3, (2,4):(1:8) >    auto tiler = make_tile(Layout<_3, _3>{},     // Apply     3:3     to mode-0                           Layout<Shape<_2, _4>, // Apply (2,4):(1,8) to mode-1                                  Stride<_1, _8>>{});    // ((TileM,RestM), (TileN,RestN)) with shape ((3,3), (8,4))    auto ld = logical_divide(layout_a, tiler);    Tensor tld = make_tensor(A, ld);    printf("\nLayout Tensor Logical Divide: ");    print_tensor(tld);}
```

下图将A描述为二维布局，其中B所指向的元素以灰色突出显示。对B所描述的Tile< 3:3 , (2:4):(1:8)> 表示我们在列方向, 每隔3个取数,取3个. 而对于行方向, 连续取2个,然后再跳步8个在连续取两个. 在A中有12个这样的块，用每种颜色表示。
![图片](assets/a8546f1767e3.jpg)

最后构造的除法结果为

### 5.4 Zipped,Tiled,Flat Divides

当我们想通过上图中下半部那样取出同色的块时, 利用mode取值则出现了问题.

```
    // ((TileM,RestM), (TileN,RestN)) with shape ((3,3), (8,4))    auto ld = logical_divide(layout_a, tiler);    Tensor tld = make_tensor(A, ld);    print_tensor(tensor<0>(tld));//outputLayout Tensor Logical Divide(mode-0): (_3,_3):(_177,_59):    0   59  118  177  236  295  354  413  472
```

因此我们期望对除法结果的维度进行某种排序,适合我们进行分块取值. 例如我们需要将A矩阵按照threadIdx.x和threadIdx.y寻址, 得到相应的子块.
对于logical_divide 输出结果总是

而我们期望的输出为:

这样我们只需要取前几个Mode的值为相应的threadIdx作为坐标就能拿到相关的矩阵了. 因此针对这些对输出维度的排序需求, CuTe定义了多种除法

```
Layout Shape : (M, N, L, ...)Tiler Shape  : <TileM, TileN>logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

当使用zipped_divide时, 就能满足我们的需求了

```
    // ((TileM,TileN), (RestM,RestN)) with shape ((3,8), (3,4))    auto zd = zipped_divide(layout_a, tiler);    Tensor tzd = make_tensor(A, zd);    print_tensor(tensor<0>(tzd));//outputLayout Tensor Zipped Divide(mode-0): (_3,(_2,_4)):(_177,(_13,_2)):    0   13    2   15    4   17    6   19  177  190  179  192  181  194  183  196  354  367  356  369  358  371  360  373    
```

**注: 其实这里我们还可以把前面的置换mode相关的定义为一个Permutation Layout函数, 在参考资料1中有详细的阐述**

## 6. 乘法

它定义了一个Layout和另一个Layout得乘积, 大致的一个想法是, 对于Layout A, 其补集的元素按照LayoutB进行放置. 简而言之就是将Layout A的Tensor重复放置若干次, 然后按照Layout B得规则放置.

### 6.1 逻辑乘法

logical_product定义如下, 它由两个Mode组成, 第一个Mode就是Layout A, 第二个Mode为Layout B, 但是每个元素被Layout A中的“unique replication”替代. “unique replication”看上去像是A的`size(A)*cosize(B)`的补集. 记乘法为:

在CuTe中实现如下所示:

```
template <class LShape, class LStride,          class TShape, class TStride>auto logical_product(Layout<LShape,LStride> const& layout,                     Layout<TShape,TStride> const& tiler){  return make_layout(layout, composition(complement(layout, size(layout)*cosize(tiler)), tiler));}
```

### 6.2 1D-逻辑乘法

例如CuTe的一个1D示例
![图片](assets/13e5dbcafcde.png)

`A = (2,2):(4,1)` 对应的坐标映射为如上第一行, 即[0,4,1,5]

`B = 6:1`, 即[0,1,2,3,4,5]
对于逻辑乘法的结果, 首先复制A作为灰色的部分,再根据B的内容进行填充[2,6,3,7] ....
我们进一步将B构成一个2-D形状B=(4,2):(2:1)

![图片](assets/1d0be5f9e1c2.png)

另外官方PPT中的一个例子, A:(4,2):(1,16)  B:(2,2):(2:1) , size(A)=8,cosize(B)=4 ,但是画的图很难让人理解清楚.

![图片](assets/002e35b24032.png)

我们计算如下:

测试代码如下

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define MAXN 128 * 128int main(){    // initial memory with physical layout    int *A = (int *)malloc(MAXN * sizeof(int));    for (int i = 0; i < MAXN; i++)    {        A[i] = int(i);    }    auto layout_a = make_layout(make_shape(Int<4>{}, Int<2>{}),                                make_stride(Int<1>{}, Int<16>{}));    Tensor ta = make_tensor(A, layout_a);    printf("\nLayout Tensor A: ");    print_tensor(ta);    auto layout_b = make_layout(make_shape(Int<2>{}, Int<2>{}),                                make_stride(Int<2>{}, Int<1>{}));    Tensor tb = make_tensor(A, layout_b);    printf("\nLayout Tensor B: ");    print_tensor(tb);    Layout a_star = complement(layout_a, size(layout_a) * cosize(layout_b));    Tensor ta_star = make_tensor(A, a_star);    printf("\nLayout Tensor A* : ");    print_tensor(ta_star);    Layout a_star2 = composition(complement(layout_a, size(layout_a) * cosize(layout_b)), layout_b);    Tensor ta_star2 = make_tensor(A, a_star2);    printf("\nLayout Tensor A* o B: ");    print_tensor(ta_star2);    auto lp = logical_product(layout_a, layout_b);        Tensor tlp = make_tensor(A, lp);    printf("\nLayout Tensor Logical Product: ");    print_tensor(tlp);}//outputLayout Tensor A: (_4,_2):(_1,_16):    0   16    1   17    2   18    3   19Layout Tensor B: (_2,_2):(_2,_1):    0    1    2    3Layout Tensor A* : _4:_4:    0    4    8   12Layout Tensor A* o B:  (_2,_2):(_8,_4):    0    4    8   12Layout Tensor Logical Product:  ((_4,_2),(_2,_2)):((_1,_16),(_8,_4)):    0    8    4   12    1    9    5   13    2   10    6   14    3   11    7   15   16   24   20   28   17   25   21   29   18   26   22   30   19   27   23   31
```

### 6.3 2D-逻辑乘法

类似的我们可以构造B为一个Tiler, 例如<3:5,4:6>两个Layout来实现by-mode的logical_product.

![图片](assets/32638c5860d3.png)

但是这样的表达方式上Tiler B很不直观, 需要完全了解A的Shape和Stride. 因此我们期望一种更直观的 "Tile Layout A according to Layout B"表达方式.

本质上是和在乘法输出结果的分配上, 使得更直观的描述

乘法结果Shapelogical_((xa,ya),(xb,yb))blocked_((xa,xb),(ya,yb))raked_((xb,xa),(yb,ya))

#### 6.3.1 Blocked乘法

例如我们构造Tiler <3:5,4:6>, 本质的意图是数据按照3x4的矩阵去把A进行分块填充, 更复合直觉的B的描述为(3:4):(1:3). 对于这样基于分块描述的乘法, 被称为blocked_product.

![图片](assets/a310e23fd6a9.png)

#### 6.3.2 Raked乘法

对于输出的Shape,在各个Mode上交换排列

![图片](assets/712c75372eb8.png)

#### 6.3.3 Zipped /Tiled Product

针对Tile mode-based描述, 同样对输出的结果排列可得, 如下所示:

```
Layout Shape : (M, N, L, ...)Tiler Shape  : <TileM, TileN>logical_product : ((M,TileM), (N,TileN), L, ...)zipped_product  : ((M,N), (TileM,TileN,L,...))tiled_product   : ((M,N), TileM, TileN, L, ...)flat_product    : (M, N, TileM, TileN, L, ...)
```

大概关于Layout 代数的内容就介绍这么多, 我们下一期将开始介绍CuTe tensor相关的内容.

参考资料

[1] 
A note on the algebra of CuTe Layouts: https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/
[2] 
CuTe Layout Algebra: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md