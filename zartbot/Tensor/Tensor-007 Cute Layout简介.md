# Tensor-007 Cute Layout简介

> 作者: zartbot  
> 日期: 2024年8月24日 13:47  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491741&idx=1&sn=c1eed8d4c5d7c20bd3cd1ee660062d28&chksm=f995f05fcee279497c54345574ebaac047d552a5469d942445bfef0d378a509bf44b5efb27bc#rd

---

### TL;DR

针对不同的硬件平台架构, 在Cutlass 2.x中定义了多种Layout抽象, 在做矩阵分块计算/解决访问内存的Bank Conflict以及算子融合的过程涉及大量的内存访问地址映射转换等复杂的计算. 因此期望有一个相对通用的代数结构, 能够进行可组合的抽象(Composable Abstration), 这是CuTe产生的原因

![图片](assets/a045169282e2.jpg)

CuTe Layout从根本上来说, 是从坐标空间到内存地址索引空间的一种映射代数. 为高维数组访问提供了一个通用的抽象接口. 用户不需要考虑列优先或者行优先的内存布局, 不用为某个分块的实际内存地址进行复杂的偏移计算. 而最关键的是针对矩阵从GMEM->SMEM->RF多次分块的过程中, 支持层次化Tensor结构和Layout代数, 可以通过一系列组合操作实现基于线程Layout的数据划分.

```
1. CuTe Overview1.1 基本类型和概念1.1.1 Integers整型1.1.2 Tuple元组1.2 Shape, Stride and Layout1.3 层次化访问1.4 使用Layout1.4.1 Layout可视化1.4.2 向量Layout1.4.3 2D矩阵Layout2. Layout布局详解2.1 Layout兼容性2.2 Layout坐标3. Layout操作3.1 SubLayout3.2 Concate3.3 Group和Flatten
```

## 1. CuTe Overview

在CuTe Layout运算中,为了可组合性来泛化表示各种矩阵拆分和线程分配任务, 因此需要一个代数系统来保证计算的封闭性和可组合性.我们先介绍一些很枯燥的基本类型和运算, 在github上有一个cute layout[1]的文档, 本章基于该文档进行分析.

### 1.1 基本类型和概念

对于Layout代数的抽象如下所示:

![图片](assets/1b5ff33799f8.png)

通过整型构成元组, 表示张量形状(Shape)和跨步(Stride), 并通过Shape和Stride组合构成Layout, 而通过一个内存的base地址指针配合Layout定义张量. 对于不同的访问内存需求, 通过Layout函数转换, 并且保证计算的封闭性. 通过这样的代数组合,我们可以通过一个统一的方式针对不同的业务需求对矩阵如何映射到内存地址进行映射

![图片](assets/4bb27f6b77c1.png)

#### 1.1.1 Integers整型

CuTe中定义的整形分为两类:

**Dynamic**: 动态整型(在运行时赋值), 和`int`/`size_t`等普通整型相同.所有被`std::is_integral<T>`接受的类型都可以作为动态整型.

**Static**: 静态整型(编译时赋值), 对于这些静态整型, CuTe定义了一些别名, 例如`Int<1>`,`Int<2>`,`Int<3>`或者`_1`,`_2`,`_3`等, `_m1`,`_m2`等表示负数.

这些自定义整型基础上还定义了相关的表达式运算, 具体代码在`/include/cute/numeric/integral_constant.hpp` 例如下的复合运算

```
    //动态整型    auto dynamic_var = int{2};    dynamic_var = 4;        //静态整型    auto static_var = Int<3>{};      // static_var  -= 3 , compile error        //复合运算    auto var = Int<8>{} + max (_4{}, _3{}) - abs(_m4{}) * dynamic_var;
```

同时CuTe还提供了一些类型检测的函数, 例如is_intergal, is_static等, 常见的用法如下

```
CUTE_STATIC_ASSERT_V(is_static<decltype(shape<0>(gmem))>{});
```

#### 1.1.2 Tuple元组

和`std::tuple`类似, 包含多个元素的有序列表, 但它可以同时适用于CPU(host)和GPU(device)函数. CuTe还定义了IntTuple类型,并且可以作为`Shape` / `Stride` / `Step` / `Coord`等多种概念的容器. 通过`make_tuple`函数可以构造元组, 并且还可以递归构造.

```
//dynamic和static构成元组make_tuple(int{2}, Int<3>{})//嵌套元组make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{})
```

IntTuple上定义了一些函数,

`rank` : IntTuple中元素数量

`get<I>`: IntTuple中的第I个元素(I < rank)

`depth`: IntTuple的层次化结构

`size` : IntTuple中所有元素的乘积

下面有一些例子

```
#define PRINT_TUPLE(name, content)      \    print(name);                  \    print(" : ");                 \    print(content);               \    print(" rank: ");             \    print(cute::rank(content));   \    print(" depth: ");            \    print(cute::depth(content));  \    print(" size: ");             \    print(cute::size(content));   \    print("\n");    auto a =  make_tuple(uint16_t{42}, int{7});    PRINT_TUPLE("a",a);    auto b =  make_tuple(uint16_t{4}, int{8},Int<9>{} );    PRINT_TUPLE("b",b);    auto c = make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), b);    PRINT_TUPLE("c",c);    PRINT_TUPLE("c<2>",get<2>(c));    //outputa : (42,7) rank: _2 depth: _1 size: 294b : (4,8,_9) rank: _3 depth: _1 size: 288//由于元组有两层, depth=2, size = 42 * 1 * 3 * 4 * 8 * 9c : (42,(_1,3),(4,8,_9)) rank: _3 depth: _2 size: 36288//取出tuple的第三个元素, size = 4 * 8 * 9c<2> : (4,8,_9) rank: _3 depth: _1 size: 288
```

### 1.2 Shape, Stride and Layout

Shape和Stride都由IntTuple描述, 而它们组合在一起构成Layout对象, 从语义上讲, 我们就可以基于Stride实现在Shape内的任何坐标到内存地址索引的映射. 并且Layout还可以组合, 如下示例:

```
Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),                               LayoutLeft{});Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),                               LayoutRight{});//Shape/Stride嵌套Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),                           make_stride(4,make_stride(2,1)));Layout s2xh4_col = make_layout(shape(s2xh4),                               LayoutLeft{});                               Layout a = make_layout(make_shape(_6{},_2{}),make_stride(_1{},_7{}));Layout b = make_layout(make_shape(_3{},_2{}),make_stride(_2{},_3{}));Layout c = composition(a,b);                               
```

Layout构造通过make_layout(shape,stride)函数实现, 而Shape和Stride都可以通过相应的make_shape/make_stride以intTuple做参数实现.
当省略Stride参数时, 将会从Shape的参数重生成, 默认采用LayoutLeft方式, 即排除自身元素的Shape, 然后从左至右乘积. 例如下第3个维度, 排除自己4, 然后从左向右乘法2x3,stride为6.这也被称作`广义的列优先跨步生成`. 采用LayoutRight标签时, 则从右至左累乘, 统一昂以第三个维度,排除4, 右侧为5x6,stride为30. 这被称作`广义的列优先跨步生成`.

```
    Layout f_col = make_layout(make_shape(Int<2>{},3,4,5,6),                               LayoutLeft{});    Layout f_row = make_layout(make_shape(Int<2>{},3,4,5,6),                               LayoutRight{});//LayoutLeft:stride = (1, 2, 2*3, 2*3*4, 2*3*4*5)fcol : (_2,3,4,5,6):(_1,_2,6,24,120) Shape: (_2,3,4,5,6) Stride: (_1,_2,6,24,120) //LayoutRight: stride = (3*4*5*6,4*5*6,5*6,6,1)frow : (_2,3,4,5,6):(360,120,30,6,_1) Shape: (_2,3,4,5,6) Stride: (360,120,30,6,_1) 
```

Layout上也类似Tuple的定义了rank/get<I>/depth/size等函数, 同时还有取shape和stride的函数. 还有cosize代表其余域(codomain)上的size. 具体的codomain的含义,我们将在CuTe Layout代数章节详细阐述. 下面有一个例子

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>#include <cutlass/numeric_types.h>using namespace cute;#define PRINT_LAYOUT(name, content)      \    print(name);                  \    print(" : ");                 \    print(content);               \    print(" Shape: ");            \    print(cute::shape(content));  \    print(" Stride: ");           \    print(cute::stride(content)); \    print(" rank: ");             \    print(cute::rank(content));   \    print(" depth: ");            \    print(cute::depth(content));  \    print(" size: ");             \    print(cute::size(content));   \    print(" cosize: ");           \    print(cute::cosize(content)); \    print("\n");int main(){    Layout a = make_layout(make_shape(_6{}, _2{}), make_stride(_1{}, _7{}));    Layout b = make_layout(make_shape(_3{}, _2{}), make_stride(_2{}, _3{}));    Layout c = composition(a, b);    Layout d = complement(a, c);    Layout e = make_layout(a, c);    PRINT_LAYOUT("a", a);    PRINT_LAYOUT("b", b);    PRINT_LAYOUT("c", c);    PRINT_LAYOUT("c-get<1>", get<1>(c));    PRINT_LAYOUT("d", d);    PRINT_LAYOUT("e", e);}//outputa : (_6,_2):(_1,_7) Shape: (_6,_2) Stride: (_1,_7) rank: _2 depth: _1 size: _12 cosize: _13b : (_3,_2):(_2,_3) Shape: (_3,_2) Stride: (_2,_3) rank: _2 depth: _1 size: _6 cosize: _8c : (_3,_2):(_2,_3) Shape: (_3,_2) Stride: (_2,_3) rank: _2 depth: _1 size: _6 cosize: _8c-get<1> : _2:_3 Shape: _2 Stride: _3 rank: _1 depth: _0 size: _2 cosize: _4d : _1:_0 Shape: _1 Stride: _0 rank: _1 depth: _0 size: _1 cosize: _1e : ((_6,_2),(_3,_2)):((_1,_7),(_2,_3)) Shape: ((_6,_2),(_3,_2)) Stride: ((_1,_7),(_2,_3)) rank: _2 depth: _2 size: _72 cosize: _20
```

### 1.3 层次化访问

get/rank/depth/shape/size函数都可以通过模板进行层次化访问, 如下所示:

```
int main(){    //构造层次化Layout    auto s1 = make_shape(_1{}, _2{});    auto d1 = make_stride(_1{}, _2{});    auto s2 = make_shape(_2{}, _3{}, s1);    auto d2 = make_stride(_2{}, _3{}, d1);    auto s3 = make_shape(_3{}, _4{}, _5{}, s2);    auto d3 = make_stride(_3{}, _4{}, _5{}, d2);    auto s4 = make_shape(_4{}, _5{}, _6{}, s3);    auto d4 = make_stride(_4{}, _5{}, _6{}, d3);    auto s5 = make_shape(_5{}, _6{}, _7{},_8{}, s4);    auto d5 = make_stride(_5{}, _6{}, _7{},_8{}, d4);    Layout a = make_layout(s5, d5);    PRINT_LAYOUT("a", a);    PRINT_LAYOUT("a<4>", get<4>(a));    auto a43 = get<4,3>(a);    PRINT_LAYOUT("a<4,3>",a43 );    auto a433 = get<4,3,3>(a);    PRINT_LAYOUT("a<4,3,3>",a433 );    auto a4332 = get<4,3,3,2>(a);    PRINT_LAYOUT("a<4,3,3,2> ",a4332 );    //从a433中获取Tuple中的第三个元素    auto a4332_1 = get<2>(a433);    PRINT_LAYOUT("a<4,3,3,2>1",a4332_1 );}//Outputa : (_5,_6,_7,_8,(_4,_5,_6,(_3,_4,_5,(_2,_3,(_1,_2))))):(_5,_6,_7,_8,(_4,_5,_6,(_3,_4,_5,(_2,_3,(_1,_2))))) Shape: (_5,_6,_7,_8,(_4,_5,_6,(_3,_4,_5,(_2,_3,(_1,_2))))) Stride: (_5,_6,_7,_8,(_4,_5,_6,(_3,_4,_5,(_2,_3,(_1,_2))))) rank: _5 depth: _5 size: _145152000 cosize: _259a<4> : (_4,_5,_6,(_3,_4,_5,(_2,_3,(_1,_2)))):(_4,_5,_6,(_3,_4,_5,(_2,_3,(_1,_2)))) Shape: (_4,_5,_6,(_3,_4,_5,(_2,_3,(_1,_2)))) Stride: (_4,_5,_6,(_3,_4,_5,(_2,_3,(_1,_2)))) rank: _4 depth: _4 size: _86400 cosize: _111a<4,3> : (_3,_4,_5,(_2,_3,(_1,_2))):(_3,_4,_5,(_2,_3,(_1,_2))) Shape: (_3,_4,_5,(_2,_3,(_1,_2))) Stride: (_3,_4,_5,(_2,_3,(_1,_2))) rank: _4 depth: _3 size: _720 cosize: _49a<4,3,3> : (_2,_3,(_1,_2)):(_2,_3,(_1,_2)) Shape: (_2,_3,(_1,_2)) Stride: (_2,_3,(_1,_2)) rank: _3 depth: _2 size: _12 cosize: _11a<4,3,3,2>  : (_1,_2):(_1,_2) Shape: (_1,_2) Stride: (_1,_2) rank: _2 depth: _1 size: _2 cosize: _3a<4,3,3,2>1 : (_1,_2):(_1,_2) Shape: (_1,_2) Stride: (_1,_2) rank: _2 depth: _1 size: _2 cosize: _3
```

### 1.4 使用Layout

#### 1.4.1 Layout可视化

我们先从一个最简单的基于Layout的循环打印来看, 假设我们有一个2D张量, Shape为(M,N). 通过size<0>(layout)即可取到M的值, 循环如下:

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;template <class Shape, class Stride>void print2D(Layout<Shape, Stride> const &layout){    for (int m = 0; m < size<0>(layout); ++m)    {        for (int n = 0; n < size<1>(layout); ++n)        {            printf("%3d  ", layout(m, n));        }        printf("\n");    }}int main() {    Layout s46_col = make_layout(make_shape(Int<4>{}, 6), LayoutLeft{});    Layout s46_row = make_layout(make_shape(Int<4>{}, 6), LayoutRight{});    printf("2d-col-major layout\n");    print2D(s46_col);    printf("2d-row-major layout\n");    print2D(s46_row);}
```

输出的Layout结果如下所示:

```
2d-col-major layout  0    4    8   12   16   20    1    5    9   13   17   21    2    6   10   14   18   22    3    7   11   15   19   23  2d-row-major layout  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23
```

注: CuTe提供了内置的Layout打印函数, 调用和输出如下

```
print_layout(s46_col);//output(_4,6):(_1,_4)       0    1    2    3    4    5     +----+----+----+----+----+----+ 0  |  0 |  4 |  8 | 12 | 16 | 20 |    +----+----+----+----+----+----+ 1  |  1 |  5 |  9 | 13 | 17 | 21 |    +----+----+----+----+----+----+ 2  |  2 |  6 | 10 | 14 | 18 | 22 |    +----+----+----+----+----+----+ 3  |  3 |  7 | 11 | 15 | 19 | 23 |    +----+----+----+----+----+----+
```

并且还可以通过`print_latex(s46_col);`生成Latex并通过pdflatex转换成图像格式

```
 ./a.out > foo.tex pdflatex foo.tex 
```

![图片](assets/c12c3a0799c9.png)

#### 1.4.2 向量Layout

我们先从最基本的一维向量(rank==1的Layout)解释Shape和Stride. 假设我们以Shape=8为例,来观察Stride的影响,测试代码如下

```
#define MAXN 128*128#define PRINTTENSOR(name,  tensor) \    print(name);                          \    print("\nTensor : ");                 \    print_tensor(tensor);                 \    print("\n");                    int main(){    // initial memory with physical layout    int* A = (int*)malloc(MAXN * sizeof(int));    for(int i =0 ; i < MAXN ; i++){     A[i]=int(i);    }       auto shape_1d = make_shape(Int<8>{});    //Layout _8:_1    Tensor t_1d = make_tensor(A, make_layout(shape_1d, make_stride(_1{})));    PRINTTENSOR("1d layout",t_1d)        //Layout _8:_2    Tensor t_s2 = make_tensor(A,make_layout(shape_1d, make_stride(_2{})));    PRINTTENSOR("1d stride2",t_s2)        //Layout _8:_m1    Tensor t_s_m1 = make_tensor(A+7,make_layout(shape_1d, make_stride(_m1{})));    PRINTTENSOR("1d stride -1",t_s_m1)    //Layout _8:_m2    Tensor t_s_m2 = make_tensor(A+16,make_layout(shape_1d, make_stride(_m2{})));    PRINTTENSOR("1d stride -1",t_s_m2)}
```

可以看到Stride可以控制数据的步长, 当取之为负时还可以进行逆序输出, Layout结果如下:

```
Layout:  8:1Coord :  0  1  2  3  4  5  6  7Index :  0  1  2  3  4  5  6  7Layout:  8:2Coord :  0  1  2  3  4  5  6  7Index :  0  2  4  6  8 10 12 14Layout:  8:-1, BaseAddress A+7Coord :  0  1  2  3  4  5  6  7Index :  7  6  5  4  3  2  1  0Layout:  8:-2, BaseAddress A+16Coord :  0  1  2  3  4  5  6  7Index : 16 14 12 10  8  6  4  2
```

本质上输出的Index为坐标和Stride的内积, 我们将在后面的一节在2D情况下介绍.

#### 1.4.3 2D矩阵Layout

我们再来扩展到rank=2的Layout, 即2D矩阵,Stride在对应的Rank上定义相对的Stride

```
#define PRINTTENSOR(name,  tensor)  \    printf("\nTensor : %s :",name); \    print_tensor(tensor);           \    print("\n");      int main(){    // initial memory with physical layout    int* A = (int*)malloc(MAXN * sizeof(int));    for(int i =0 ; i < MAXN ; i++){     A[i]=int(i);    }           // 2D tensor    auto shape2d = make_shape(_4{},_3{});        //(_4,_8):(_1,_4)    Layout l1 = make_layout(shape2d, LayoutLeft{});    Tensor t1 = make_tensor(A, l1);    PRINTTENSOR("LayoutLeft",t1)    //(_4,_8):(_8,_1)    Layout l2 = make_layout(shape2d, LayoutRight{});    Tensor t2 = make_tensor(A, l2);    PRINTTENSOR("LayoutRight",t2)        //(_4,_8):(_3,_2)    Layout l3 = make_layout(shape2d, make_stride(_3{},_2{}));    Tensor t3 = make_tensor(A, l3);    PRINTTENSOR("(_4,_8):(_3,_2)",t3)}
```

如前面章节所述, LayoutLeft{}标签为`广义的列优先布局`,因此其产生的Stride为`(_1,_4)`即针对第一个维度每个元素的Stride为1(例如下面的每一列, 针对第二个维度每个元素的Stride为4(例如下第二行 1,5,9...)

```
Tensor : LayoutLeft :ptr[32b](0x5587c68585d0) o (_4,_8):(_1,_4):    0    4    8   12   16   20   24   28    1    5    9   13   17   21   25   29    2    6   10   14   18   22   26   30    3    7   11   15   19   23   27   31`` LayoutRight{}标签为`广义的行优先布局`,因此其产生的Stride为`(_8,_1)`即针对第一个维度每个元素的Stride为8, 第二个维度每个元素的Stride为1.```cTensor : LayoutRight :ptr[32b](0x5587c68585d0) o (_4,_8):(_8,_1):    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31
```

具体的编排顺序示意图如论文《Graphene: An IR for Optimized Tensor Computations on GPUs》[2], 它也借用了CuTe Layout的代数结构

![图片](assets/17b13caee0c5.png)

掌握的这种按照行列Stride定义的方式后, 我们再来构造第三种Stride为`(_3,_2)`的情况,即第一个维度(列)的跳步为3, 第二个维度的跳步为2. 通过Stride的方式即可索引到内存地址.

```
Tensor : (_4,_8):(_3,_2) :ptr[32b](0x56113764d5d0) o (_4,_8):(_3,_2):    0    2    4    6    8   10   12   14    3    5    7    9   11   13   15   17    6    8   10   12   14   16   18   20    9   11   13   15   17   19   21   23
```

## 2. Layout布局详解

在这一章中, 我们将引入坐标以及和Stride对应的内积表示构成Index. 同时对于同一个矩阵的不同Layout兼容性进行分析.

### 2.1 Layout兼容性

在pytorch中,对于一个tensor我们可以通过view函数来改变形状

![图片](assets/a64d6d856959.png)

在Cute Layout中, 我们类似的来定义Layout兼容性. 对于Layout A和Layout B, 如果它们的Shape满足如下情况,则称A和B是兼容的(Compatible)

A和B的size是相等的

A内所有坐标都是B内的有效坐标

注意到第二条规则, Shape兼容是一个弱偏序关系, 即满足自反性,反对称性和传递性, 我们用如下代码进行测试

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;template<class T1,class T2>void print_compatible(T1 l1, T2 l2) {    print(l1);    printf(" -> ");    print(l2);    printf(" is ");    if (is_compatible<decltype(l1),decltype(l2)>()) {        printf("compatible\n");    } else {        printf("NOT compatible\n");    }}int main(){        auto s1 = make_shape(_24{});        printf("reflexive\n");    print_compatible(s1,s1);    printf("\n\ntransitive\n");    auto s3 = make_shape(make_tuple(_4{},_6{}));        auto s5 = make_shape(make_tuple(make_tuple(_2{},_2{}),_6{}));    print_compatible(s1,s3);    print_compatible(s3,s5);    print_compatible(s1,s5);    printf("\n\nantisymetric\n");    auto s2 = make_shape(make_tuple(_24{}));    print_compatible(s1,s2);    print_compatible(s2,s1);    print_compatible(s1,s3);    print_compatible(s3,s1);    printf("\n\nothers\n");    auto s4 = make_shape(make_tuple(_2{},_3{}),_4{});    auto s6 = make_shape(make_tuple(_2{},_3{},_4{}));    print_compatible(s1,s4);    print_compatible(s1,s6);}
```

结果如下:

```
reflexive(_24) -> (_24) is compatibletransitive(_24) -> ((_4,_6)) is compatible((_4,_6)) -> (((_2,_2),_6)) is compatible(_24) -> (((_2,_2),_6)) is compatibleantisymetric(_24) -> ((_24)) is compatible((_24)) -> (_24) is NOT compatible(_24) -> ((_4,_6)) is compatible((_4,_6)) -> (_24) is NOT compatibleothers(_24) -> ((_2,_3),_4) is NOT compatible(_24) -> ((_2,_3,_4)) is compatible(_24) -> (((_2,_3),_4)) is compatible((_4,_6)) -> (((_2,_3),_4)) is NOT compatible
```

### 2.2 Layout坐标

每个Layout都可以接受多种坐标, 并且每个Layout都接受与其兼容的任何Shape的坐标, CuTe通过Colex Order(colexicographical Order)提供这些坐标集合之间的映射.

Lexicographical Order即字典序, 是指按照单词首字母顺序在字典中进行排序的方法, 它是一种从左至右读取的方法进行排序。在数学中可推广到有序符号序列，可视为完全有序集合的元素序列的一种排序方法。 而Co-Lexicographical Order则是按照字典顺序从右向左排序.

![图片](assets/1560038932c4.png)

我们以Shape(3,(2,3))为例来介绍Colex Order, 对于这个Shape, 它可以接受1D/2D和原生的层次化(x,(y,z))的h-D坐标, 其对应关系如下所示:

1-D2-DNatural(h-D)
1-D2-DNatural(h-D)`0``(0,0)``(0,(0,0))`
`9``(0,3)``(0,(1,1))``1``(1,0)``(1,(0,0))`
`10``(1,3)``(1,(1,1))``2``(2,0)``(2,(0,0))`
`11``(2,3)``(2,(1,1))``3``(0,1)``(0,(1,0))`
`12``(0,4)``(0,(0,2))``4``(1,1)``(1,(1,0))`
`13``(1,4)``(1,(0,2))``5``(2,1)``(2,(1,0))`
`14``(2,4)``(2,(0,2))``6``(0,2)``(0,(0,1))`
`15``(0,5)``(0,(1,2))``7``(1,2)``(1,(0,1))`
`16``(1,5)``(1,(1,2))``8``(2,2)``(2,(0,1))`
`17``(2,5)``(2,(1,2))`

这个Shape包含3x2x3=18个元素, 我们可以看到高维坐标的排序即是按照Colex Order 从右到左按照字典序排序的.CuTe提供坐标到Index以及Index到坐标的映射, 如下所示:

```
    auto shape = Shape<_3, Shape<_5, _4>>{};    printf("\nidx2crd 19 : ");     print(idx2crd(19, shape));     printf("\nidx2crd (1,5) : ");     print(idx2crd(make_coord(1, 5), shape));          printf("\nidx2crd (1,(1,2)) : ");     print(idx2crd(make_coord(1, make_coord(1, 2)), shape));       printf("\ncrd2idx (1,5) : ");
```

输出结果

```
idx2crd 19 : (1,(1,1))idx2crd (1,5) : (1,(0,1))idx2crd (1,(1,2)) : (1,(1,2))
```

### 2.3 Index映射

从坐标到index的映射是通过自然坐标和Layout的Stride内积来表示的

![图片](assets/c86e423d9eaa.png)

对于 Shape : `(_4,(_2,_4))`, Stride: `(_2,(_1,_8))` Layout如下所示, 对于坐标(i,(j,k))表示如下

```
 i      0    1    2    3    4    5    6    7   <==  1-D col coord |   (0,0)(1,0)(0,1)(1,1)(0,2)(1,2)(0,3)(1,3)  <==  2-D col coord (j,k) v  +----+----+----+----+----+----+----+----+ 0  |  0 |  1 |  8 |  9 | 16 | 17 | 24 | 25 |    +----+----+----+----+----+----+----+----+ 1  |  2 |  3 | 10 | 11 | 18 | 19 | 26 | 27 |    +----+----+----+----+----+----+----+----+ 2  |  4 |  5 | 12 | 13 | 20 | 21 | 28 | 29 |    +----+----+----+----+----+----+----+----+ 3  |  6 |  7 | 14 | 15 | 22 | 23 | 30 | 31 |    +----+----+----+----+----+----+----+----+
```

我们可以通过crd2idx函数获取index, 例如对于元素21的坐标为 (2,(1,2))的坐标对应的index

```
    auto shape = Shape<_4, Shape<_2, _4>>{};    auto stride = Stride<_2,Stride<_1,_8>>{};    auto l = make_layout(shape,stride);    print_layout(l);    printf("\ncrd2idx 22 : ");     print(crd2idx(22, shape, stride));     printf("\ncrd2idx (2,5) : ");     print(crd2idx(make_coord(2,5), shape, stride));     printf("\ncrd2idx (2,(1,2)) : ");     print(crd2idx(make_coord(2,make_coord(1,2)), shape, stride)); //outputcrd2idx 22 : 21crd2idx (2,5) : 21crd2idx (2,(1,2)) : 21
```

我们可以按照下图标记的灰色线来计算内存中的index

![图片](assets/7312f0022176.png)

## 3. Layout操作

### 3.1 SubLayout

我们可以在一个Layout中通过`layout<I...>`, `select<I...>`, `take<I...>`来抽取Sub-Layout

![图片](assets/8b44d8882886.png)

Layout处理如上图所示:

```
a :(_4,((_4,_5),(_6,_7))):(_1,((_4,_16),(_80,_480)))layout<0>(a) :_4:_1layout<1>(a) :((_4,_5),(_6,_7)):((_4,_16),(_80,_480))layout<1,0>(a) :(_4,_5):(_4,_16)layout<1,1>(a) :(_6,_7):(_80,_480)layout<1,1,0>(a) :_6:_80
```

另外我们也可以通过select函数来选择某几个维度

```
b :(_2,_3,_5,_7):(_1,_2,_6,_30)select<2>(b) :(_5):(_6)select<1,3>(b) :(_3,_7):(_2,_30)select<0,1,3>(b) :(_2,_3,_7):(_1,_2,_30)
```

CuTe还提供了`take<begin,end>`的方式来选择

```
take<1,3>(b) :(_3,_5):(_2,_6)take<1,4>(b) :(_3,_5,_7):(_2,_6,_30)
```

### 3.2 Concate

Layout可以通过make_layout来修改

```
Layout a = Layout<_3,_1>{};                     // 3:1Layout b = Layout<_4,_3>{};                     // 4:3//分别对Tuple中相同的rank构造IntTuple, 例如A<0>=_3,b<0>=_4, row<0>=(_3,_4)Layout row = make_layout(a, b);                 // (3,4):(1,3)Layout col = make_layout(b, a);                 // (4,3):(3,1)Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))Layout aa  = make_layout(a);                    // (3):(1)Layout aaa = make_layout(aa);                   // ((3)):((1))Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

CuTe还支持append, prepend,replace等方式来构建某个维度的IntTuple

```
ayout a = Layout<_3,_1>{};                     // 3:1Layout b = Layout<_4,_3>{};                     // 4:3//将同Rank的b的Int添加在a之后Layout ab = append(a, b);                       // (3,4):(1,3)Layout ba = prepend(a, b);                      // (4,3):(3,1)Layout c  = append(ab, ab);                     // (3,4,(3,4)):(1,3,(1,3))//替换某个Rank的IntLayout d  = replace<2>(c, b);                   // (3,4,4):(1,3,3)
```

### 3.3 Group和Flatten

Cute Layout还可以通过group<begin,end>将某些Int聚合成IntTuple,或者通过flatten函数展平

```
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

虽然Layout也支持切片(Slice), 但是应用中更多用到的还是在Tensor切片, 详细内容我们在以后的文章中介绍, 下一篇我们将来看看CuTe Layout代数相关的内容.

参考资料

[1] 
CuTe layout: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md
[2] 
Graphene: An IR for Optimized Tensor Computations on GPUs: https://dl.acm.org/doi/pdf/10.1145/3582016.3582018