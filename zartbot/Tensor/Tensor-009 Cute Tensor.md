# Tensor-009 Cute Tensor

> 作者: zartbot  
> 日期: 2024年9月9日 11:09  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492327&idx=1&sn=05b466701155b74279876130737c1b5c&chksm=f995f225cee27b3315a24862a1b95720929b25dceb197925c9f28207f6537bb3f46daef5b7f0#rd

---

前一篇我们详细介绍了CuTe Layout及其相关的代数, 这一篇我们将开始介绍Tensor. 简单来看Layout只是定义了元素编排和底层存储之间的位置关系, 但是并没有关联到真正的存储. 给一个相对狭义的Tensor定义, 它是通过物理存储空间和Layout定义的一种数据结构, 对外暴露出一种多维数组的形式,对内基于Layout进行索引.

对于一个Tensor由Layout和Engine两个模版参数表示, Layout已经在前一篇详细介绍了, 它是一个逻辑结构将坐标映射到偏移量(Offset), 而Engine是一个基于Offset和解引用的迭代器. 本文目录如下:

```
 1. 创建Tensor 1.1 张量所有权 1.1.1 Non-Owning Tensor 1.1.2 Owning Tensor 1.2 小结 2. 使用张量 2.1 基本操作 2.2 元素访问 2.3 张量切片(Slicing) 2.4 Flatten/Coalesce/Group_modes 3. Tiling & Partitioning 3.1 Tensor除法 3.2 Partitioning 3.2.1 Inner分区 3.2.2 Outer分区 3.2.3 Thread-Value分区 4. 张量算法 4.1 Fill 4.2 clear 4.3 Axpby 
```

## 1. 创建Tensor

### 1.1 张量所有权

在构建CuTe Tensor时根据对象的所有权分为`owning`和`non-owning`, Owning Tensor的行为类似于`std:array`. 拷贝时会执行deepcopy去复制每个元素, Tensor的析构函数会释放元素数组. Non-owning则类似于一个指针, 拷贝时不复制元素, 销毁时也不会释放数据数组. 这些需要在传递参数时留意.

#### 1.1.1 Non-Owning Tensor

Tensor通常是对现有内存的一个non-owning视图, 例如采用cudamalloc一段内存后, 然后根据内存指针作为参数, 并通过`make_tensor`函数创建, 通常我们可以创建host内存里的张量, 也可以在device上创建, 并可以通过内存空间来标记迭代器, 例如`make_gmem_ptr`指定全局内存(GMEM), 或者`make_smem_ptr`指定共享内存(SMEM), 如下所示:

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define MAXN 128 * 128#define PRINTTENSOR(name, tensor) \    printf("%20s : ", name);      \    print(tensor);                \    print("\n");__global__ void tensor_kernel(float *A){    // 不使用tag    Tensor tensor_8 = make_tensor(A, make_layout(Int<8>{})); // Construct with Layout    Tensor tensor_8s = make_tensor(A, Int<8>{});             // Construct with Shape    Tensor tensor_8d2 = make_tensor(A, 8, 2);                // Construct with Shape and Stride    PRINTTENSOR("tensor_8", tensor_8)    PRINTTENSOR("tensor_8s", tensor_8s)    PRINTTENSOR("tensor_8d2", tensor_8d2)    // 全局内存,使用make_gmem_ptr标记,支持动态和静态Layout    Tensor gmem_8s = make_tensor(make_gmem_ptr(A), Int<8>{});    Tensor gmem_8d = make_tensor(make_gmem_ptr(A), 8);    Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{}, 16));    Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape(8, Int<16>{}),                                     make_stride(Int<16>{}, Int<1>{}));    PRINTTENSOR("gmem_8s", gmem_8s)    PRINTTENSOR("gmem_8d", gmem_8d)    PRINTTENSOR("gmem_8sx16d", gmem_8sx16d)    PRINTTENSOR("gmem_8dx16s", gmem_8dx16s)    // 共享内存,使用make_smem_ptr标记,支持动态和静态Layout    Layout smem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}));    __shared__ float smem[decltype(cosize(smem_layout))::value]; // (static-only allocation)    Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);    Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), GenRowMajor{});    PRINTTENSOR("smem_4x8_col", smem_4x8_col)    PRINTTENSOR("smem_4x8_row", smem_4x8_row)}int main(){    // initial memory     float *A = (float *)malloc(MAXN * sizeof(float));    for (int i = 0; i < MAXN; i++)    {        A[i] = float(i);    }    // Untagged pointers    Tensor tensor_8 = make_tensor(A, make_layout(Int<8>{})); // Construct with Layout    Tensor tensor_8s = make_tensor(A, Int<8>{});             // Construct with Shape    Tensor tensor_8d2 = make_tensor(A, 8, 2);                // Construct with Shape and Stride    PRINTTENSOR("host_tensor_8", tensor_8)    PRINTTENSOR("host_tensor_8s", tensor_8s)    PRINTTENSOR("host_tensor_8d2", tensor_8d2)    printf("\n");    //分配显存    float *dA;    cudaMalloc(&dA, MAXN * sizeof(float));    cudaMemcpy(dA, A, MAXN * sizeof(float), cudaMemcpyHostToDevice);    //执行Kernel    tensor_kernel<<<1, 1>>>(dA);        cudaDeviceSynchronize();    free(A);    cudaFree(dA);}
```

输出结果如下, 可以看到它们关联的地址是相同的.

```
       host_tensor_8 : ptr[32b](0x55b68f9b2630) o _8:_1      host_tensor_8s : ptr[32b](0x55b68f9b2630) o _8:_1     host_tensor_8d2 : ptr[32b](0x55b68f9b2630) o 8:2            tensor_8 : ptr[32b](0x7f925aa00000) o _8:_1           tensor_8s : ptr[32b](0x7f925aa00000) o _8:_1          tensor_8d2 : ptr[32b](0x7f925aa00000) o 8:2                       gmem_8s : gmem_ptr[32b](0x7f925aa00000) o _8:_1             gmem_8d : gmem_ptr[32b](0x7f925aa00000) o 8:_1         gmem_8sx16d : gmem_ptr[32b](0x7f925aa00000) o (_8,16):(_1,_8)         gmem_8dx16s : gmem_ptr[32b](0x7f925aa00000) o (8,_16):(_16,_1)                 smem_4x8_col : smem_ptr[32b](0x7f9281000000) o (_4,_8):(_1,_4)        smem_4x8_row : smem_ptr[32b](0x7f9281000000) o (_4,_8):(_8,_1)
```

#### 1.1.2 Owning Tensor

通过`make_tensor<T>`创建, 但仅支持静态Layout

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define PRINTTENSOR(name, tensor) \    printf("%20s : ", name);      \    print(tensor);                \    print("\n");__global__ void tensor_kernel(){      // Register memory (static layouts only)    Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});    Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{},                                             LayoutRight{});    Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{},                                             Stride<_32, _2>{});    Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);    PRINTTENSOR("rmem_4x8_col", rmem_4x8_col)    PRINTTENSOR("rmem_4x8_row", rmem_4x8_row)    PRINTTENSOR("rmem_4x8_pad", rmem_4x8_pad)    PRINTTENSOR("rmem_4x8_like", rmem_4x8_like)}int main(){    // Register memory (static layouts only)    Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});    Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{},                                             LayoutRight{});    Tensor rmem_4x8_pad = make_tensor<float>(Shape<_4, _8>{},                                             Stride<_32, _2>{});    Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);    PRINTTENSOR("host_rmem_4x8_col", rmem_4x8_col)    PRINTTENSOR("host_rmem_4x8_row", rmem_4x8_row)    PRINTTENSOR("host_rmem_4x8_pad", rmem_4x8_pad)    PRINTTENSOR("host_rmem_4x8_like", rmem_4x8_like)    printf("\n");    tensor_kernel<<<1, 1>>>();    cudaDeviceSynchronize();}
```

可以看到每个Tensor都有唯一的地址

```
   host_rmem_4x8_col : ptr[32b](0x7ffeaccf0a70) o (_4,_8):(_1,_4)   host_rmem_4x8_row : ptr[32b](0x7ffeaccf0af0) o (_4,_8):(_8,_1)   host_rmem_4x8_pad : ptr[32b](0x7ffeaccf0bf0) o (_4,_8):(_32,_2)  host_rmem_4x8_like : ptr[32b](0x7ffeaccf0b70) o (_4,_8):(_8,_1)        rmem_4x8_col : ptr[32b](0x7faec3fff990) o (_4,_8):(_1,_4)        rmem_4x8_row : ptr[32b](0x7faec3fffa10) o (_4,_8):(_8,_1)        rmem_4x8_pad : ptr[32b](0x7faec3fffa90) o (_4,_8):(_32,_2)       rmem_4x8_like : ptr[32b](0x7faec3fffc50) o (_4,_8):(_8,_1)
```

### 1.2 小结

对于Dynamic Layout只能采用`non-owning`的方式创建Tensor. 而静态类型可以通过创建时`make_tensor`函数指定指针创建为`non-owning`类型, 同时对于指针支持指定特定内存(GMEM/SMEM)的tag. 如果不指定指针并采用静态类型可以创建`owning` Tensor.

## 2. 使用张量

### 2.1 基本操作

Tensor可以通过layout/shape/stride/size/rank/depth等函数获得其张量的维度等各种信息. 同时还可以通过data函数获得数据存储空间的基地址.示例如下:

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define MAXN 128 * 128#define PRINT(name, tensor)  \    printf("%20s : ", name); \    print(tensor);           \    print("\n");__global__ void tensor_kernel(float *A){    Tensor t = make_tensor(A, make_shape(_8{}, _4{}), GenColMajor{});    PRINT("tensor_8x4", t)    PRINT("Layout", t.layout())    PRINT("SHAPE", t.shape())    PRINT("STRIDE", t.stride())    PRINT("SIZE", t.size())    PRINT("Data", t.data())    PRINT("Rank", t.rank)    PRINT("Depth", depth(t))}int main(){    // initial memory    float *A = (float *)malloc(MAXN * sizeof(float));    for (int i = 0; i < MAXN; i++)    {        A[i] = float(i);    }    float *dA;    cudaMalloc(&dA, MAXN * sizeof(float));    cudaMemcpy(dA, A, MAXN * sizeof(float), cudaMemcpyHostToDevice);    tensor_kernel<<<1, 1>>>(dA);    cudaDeviceSynchronize();    free(A);    cudaFree(dA);}//output          tensor_8x4 : ptr[32b](0x7f3822a00000) o (_8,_4):(_1,_8)              Layout : (_8,_4):(_1,_8)               SHAPE : (_8,_4)              STRIDE : (_1,_8)                SIZE : _32                Data : ptr[32b](0x7f3822a00000)                Rank : 2               Depth : _1
```

其中Tensor也可以提供按Mode的层次化操作

`rank<I...>(Tensor)`: The rank of the I...th mode of the Tensor.

`depth<I...>(Tensor)`: The depth of the I...th mode of the Tensor.

`shape<I...>(Tensor)`: The shape of the I...th mode of the Tensor.

`size<I...>(Tensor)`: The size of the I...th mode of the Tensor.

`layout<I...>(Tensor)`: The layout of the I...th mode of the Tensor.

`tensor<I...>(Tensor)`: The subtensor corresponding to the the I...th mode of the Tensor.

### 2.2 元素访问

Tensor对象可以基于小括号和中括号运算符进行数据读写访问

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;int main(){    Tensor A = make_tensor<float>(Shape<Shape<_4, _5>, Int<13>>{},                                  Stride<Stride<_12, _1>, _64>{});    float *b_ptr = (float *)malloc(13 * 20 * sizeof(float));    Tensor B = make_tensor(b_ptr, make_shape(13, 20));    // Fill A via natural coordinates op[]    for (int m0 = 0; m0 < size<0, 0>(A); ++m0)        for (int m1 = 0; m1 < size<0, 1>(A); ++m1)            for (int n = 0; n < size<1>(A); ++n)                A[make_coord(make_coord(m0, m1), n)] = n + 2 * m0;    // Transpose A into B using variadic op()    for (int m = 0; m < size<0>(A); ++m)        for (int n = 0; n < size<1>(A); ++n)            B(n, m) = A(m, n);    // Copy B to A as if they are arrays    for (int i = 0; i < A.size(); ++i)        A[i] = B[i];        print_tensor(A);    print_tensor(B);    free(b_ptr);}
```

### 2.3 张量切片(Slicing**)

在访问Tensor时, 可以通过`_`传入坐标进行切片, 返回该Mode下的所有子张量, 除此之外, 类似于Layout,还可以通过`take<Begin,End>`来选择某几个维度的数据

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define MAXN 128 * 128int main(){    // initial memory    int *a_ptr = (int *)malloc(MAXN * sizeof(int));    for (int i = 0; i < MAXN; ++i)        a_ptr[i] = i;    //(_3,_4,_5):(_20,_5,_1)    Tensor A = make_tensor(a_ptr, make_shape(Int<3>{}, Int<4>{}, Int<5>{}),                           GenRowMajor{});    print_tensor(A);    Tensor A1 = A(_, _, 2);    print_tensor(A1);    //(_3,_4),(_2,_4,_2)):((_64,_16),(_8,_2,_1)    Tensor B = make_tensor(a_ptr, make_shape(make_shape(Int<3>{}, Int<4>{}),                                   make_shape(Int<2>{}, Int<4>{}, Int<2>{})),                                  GenRowMajor{});    print_tensor(B);    Tensor C = B(make_coord(_, _), make_coord(1, 2, 1));    print_tensor(C);    Tensor D = B(make_coord(1, _), make_coord(0, _, 1));    print_tensor(D);        Tensor E = take<0,1>(B);    print_tensor(E);    }
```

### 2.4 Flatten/Coalesce/Group_modes

和Layout类似, Tensor支持将层次化结构进行展平/合并和按照Mode聚合.

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define MAXN 128 * 128int main(){    // initial memory    int *a_ptr = (int *)malloc(MAXN * sizeof(int));    for (int i = 0; i < MAXN; ++i)        a_ptr[i] = i;    //(_3,_4),(_2,_4,_2)):((_64,_16),(_8,_2,_1)    Tensor B = make_tensor(a_ptr, make_shape(make_shape(Int<3>{}, Int<4>{}), make_shape(Int<2>{}, Int<4>{}, Int<2>{})),                           GenRowMajor{});    //flatten将层次化结构展平    //(_3,_4,_2,_4,_2):(_64,_16,_8,_2,_1)    Tensor C = flatten(B);    print_tensor(C);    // ((_3,_4),(_2,_4,_2)):((_1,_3),(_12,_24,_96))    Tensor D = make_tensor(a_ptr, make_shape(make_shape(Int<3>{}, Int<4>{}), make_shape(Int<2>{}, Int<4>{}, Int<2>{})),                           GenColMajor{});    print_tensor(D);        //按照层级合并连续的坐标, 由于是GenColMajor,则合并到1维.    //_192:_1    Tensor E = coalesce(D);    print_tensor(E);    //按照Mode Begin=1, End=4聚合, 即(_4,_2,_4)    //(_3,(_4,_2,_4),_2):(_64,(_16,_8,_2),_1):    Tensor F = group_modes<1,4>(C);    print_tensor(F);}
```

## 3. Tiling & Partitioning

### 3.1 Tensor除法

**Layout代数中的乘法没有在Tensor中实现**, 主要原因是乘法会导致cosize变化引起内存访问越界等安全问题. 而除法在前一篇Cute Layout代数已经解释过, 具体内容可以参考前一篇

```
   composition(Tensor, Tiler)logical_divide(Tensor, Tiler) zipped_divide(Tensor, Tiler)  tiled_divide(Tensor, Tiler)   flat_divide(Tensor, Tiler)
```

### 3.2 Partitioning

为了实现通用的张量分块, 我们可以通过composition或者Tiling并配合Slicing完成. 通常有三种非常有用的分区方法.例如我们对一个4x6的Tensor进行分区, Tiler为(_2,_3)

```
Tensor A = make_tensor(ptr, make_shape(4,6));  auto tiler = Shape<_2,_3>{};                   Tensor tiled_a = zipped_divide(A, tiler); //((_2,_3),(2,2)):((_1,4),(_2,12))     
```

![图片](assets/16e15733b36a.jpg)

#### 3.2.1 Inner分区

例如我们需要为每个ThreadGroup提供4x8的Tile, 则可以对zipped_divide的后一个mode(2,2)进行索引,如下所示:

```
    int blockIdx_x = 0;    int blockIdx_y = 1;    Tensor cta_a = tiled_a(make_coord(_, _), make_coord(blockIdx_x, blockIdx_y));    PRINT("CTA_A", cta_a)    Tensor local_tileA = local_tile(A, tiler, make_coord(0, 1));    PRINT("LOCAL_TILE", local_tileA)//output(blockIdx.x = 0, blockIdx.y =1)    CTA_A : ptr[32b](0x55aa80892600) o (_2,_3):(_1,4):   12   16   20   13   17   21LOCAL_TILE : ptr[32b](0x55aa80892600) o (_2,_3):(_1,4):   12   16   20   13   17   21
```

我们将其称为Inner分区，因为它保持了内部的“Tile”这个Mode。这种先应用Tiling然后通过对剩余Mode进行索引来切出该Tile的模式的方式很常见,并已经封装成函数`inner_partition(Tensor, Tiler, Coord)`. 我们经常会看到`local_tile(Tensor, Tiler, Coord)`，也是inner_partition的别名

#### 3.2.2 Outer分区

另一种做法是通过第一个Mode来索引数据.

```
    int threadIdx_x = 3;    Tensor thr_a = tiled_a(threadIdx_x, make_coord(_, _));    PRINT("THR_A", thr_a)    Tensor outer_partA = outer_partition(A, tiler, make_coord(1, 1));    PRINT("OUTER_PART", outer_partA)    Tensor local_partA = local_partition(A, make_layout(Shape<_2, _3>{}), 3);    PRINT("LOCAL_PART", local_partA)    //output   THR_A : ptr[32b](0x55aa808925e4) o (2,2):(_2,12):    5   17    7   19OUTER_PART : ptr[32b](0x55aa808925e4) o (2,2):(_2,12):    5   17    7   19LOCAL_PART : ptr[32b](0x55aa808925e4) o (2,2):(_2,12):    5   17    7   19
```

对于这种索引第一个Mode,保留了剩下的Mode的, 因此也被称为Outer分区, 并且可以通过`outer_partition(Tensor, Tiler, Coord)`或者`local_partition(Tensor, Layout, Idx)`进行索引.

#### 3.2.3 Thread-Value分区

Thread-Value Partition的目的是对于一个MxN的矩阵, 我们期望以(threadIdx, ValueIdx)的方式去进行分区, 让相应的Thread能够处理相应的Value. 如下所示:

![图片](assets/d98efac68be1.jpg)

对于任意的4x8 Layout, 通过和Thread-Value复合,即可得到Thread-Value Partition的矩阵, 通过TID和VID即可对相应的值进行处理

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;#define MAXN 128 * 128int main(){    // initial memory    int *hA = (int *)malloc(MAXN * sizeof(int));       for (int i = 0; i < MAXN; ++i)        hA[i] = i;      // 构造TV Layout, 共计8个线程, 每个线程4个值, 映射到MxN=4x8的空间    // (T8,V4) -> (M4,N8)    auto tv_layout = Layout<Shape<Shape<_2, _4>, Shape<_2, _2>>,                            Stride<Stride<_8, _1>, Stride<_4, _16>>>{}; // (8,4)    print_layout(tv_layout);    // 数据张量MxN=4x8    Tensor A = make_tensor(hA,make_shape(_4{},_8{}),GenColMajor{});    print_tensor(A);        // 复合数据张量和TV-Layout, 生成TV张量    Tensor tv = composition(A, tv_layout);         // 基于tv张量按照threadIdx取值计算    int tid = 1;    Tensor v = tv(tid, _); // (4)    print_tensor(v);}//output((_2,_4),(_2,_2)):((_8,_1),(_4,_16))       0    1    2    3     +----+----+----+----+ 0  |  0 |  4 | 16 | 20 |    +----+----+----+----+ 1  |  8 | 12 | 24 | 28 |     +----+----+----+----+ 2  |  1 |  5 | 17 | 21 |    +----+----+----+----+ 3  |  9 | 13 | 25 | 29 |    +----+----+----+----+ 4  |  2 |  6 | 18 | 22 |    +----+----+----+----+ 5  | 10 | 14 | 26 | 30 |    +----+----+----+----+ 6  |  3 |  7 | 19 | 23 |    +----+----+----+----+ 7  | 11 | 15 | 27 | 31 |    +----+----+----+----+//数据张量ptr[32b](0x55a151d3a5d0) o (_4,_8):(_1,_4):    0    4    8   12   16   20   24   28    1    5    9   13   17   21   25   29    2    6   10   14   18   22   26   30    3    7   11   15   19   23   27   31//threadIdx =1的值(V0~V3)ptr[32b](0x55a151d3a5f0) o ((_2,_2)):((_4,_16)):    8   12   24   28
```

通过这样的TV Partition, 原始数据映射到ThreadID,ValueID的关系如下所示:

![图片](assets/7f665f54d039.png)

## 4. 张量算法

在`include/cute/algorithm`目录中定义了一系列tensor相关的常见数值算法的接口和实现. 其中Copy和MMA我们将在后续的文章中单独介绍.

### 4.1 Fill

按照某个值进行张量填充.

```
    Tensor A = make_tensor<int>(make_shape(_4{},_8{}),GenColMajor{});    fill(A, 7);    print_tensor(A);//outputptr[32b](0x7fff0275ff70) o (_4,_8):(_1,_4):    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7    7
```

### 4.2 Clear

使用默认构造函数对张量中的元素赋值

```
    Tensor A = make_tensor<int>(make_shape(_4{},_8{}),GenColMajor{});    fill(A, 7);    clear(A);    print_tensor(A);//outputptr[32b](0x7ffe833948f0) o (_4,_8):(_1,_4):    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
```

### 4.3 axpby

axpby(alpha,A, beta,B)表示对两个张量进行线性运算 B = alpha * A + beta * B

```
#include <cuda.h>#include <stdlib.h>#include <cute/tensor.hpp>using namespace cute;int main(){      Tensor A = make_tensor<int>(make_shape(_4{},_8{}),GenColMajor{});    fill(A, 3);    Tensor B = make_tensor<int>(make_shape(_4{},_8{}),GenColMajor{});    fill(B, 2);    //B = 3 * A + 2 * B    axpby(3,A, 2, B);    print_tensor(B);}//outputptr[32b](0x7ffe83394970) o (_4,_8):(_1,_4):   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13   13
```