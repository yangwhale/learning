# CuteDSL-2: 基本操作

> 作者: zartbot  
> 日期: 2025年9月29日 23:19  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496212&idx=2&sn=72439a38516bc9754bb6fde9dca99084&chksm=f995e2d6cee26bc02faf18bae0e57d12ae1c711e894d5719b658cccf438f3b0a82da371e1734#rd

---

### TL;DR

这是Cute-DSL的第二篇, 主要对一些基本的操作进行介绍, 主要是逐个整理一下官方的notebook[1].

## 1. 基本操作

### 1.1 数据类型

CuteDSL基本上常见的整数和浮点类型都支持. 下面是一个示例

```
import osos.environ['CUTE_DSL_ARCH'] = 'sm_101a' #Thor在cuda 13.0改名为SM110, 但是cutedsl-4.2还是基于12.9, 因此需要设置一个环境变量import cutlassimport cutlass.cute as cute@cute.jitdef bar(va: cutlass.Constexpr[int], vb: cutlass.Float32):    print("va(static) =", va)                 cute.printf("c(dynamic) = {}", va)       print("vb(static) =", vb)                 cute.printf("c(dynamic) = {}", vb)       a = cutlass.BFloat16(3.14)    print("a(static) =", a)                 cute.printf("a(dynamic) = {}", a)       b = cutlass.Float4E2M1FN(5.99e-3)    print("b(static) =", b)                 cute.printf("b(dynamic) = {}", b)       ca = cutlass.Constexpr[cutlass.BFloat16]    ca = (7+ a + vb)    print("ca(static) =", ca)                 cute.printf("ca(dynamic) = {}", ca)  bar(9, 3.14)# outputva(static) = 9vb(static) = ?a(static) = ?b(static) = ?ca(static) = ?c(dynamic) = 9c(dynamic) = 3.140000a(dynamic) = 3.140000b(dynamic) = 0.005990ca(dynamic) = 13.280001
```

print vs cute.print
在cute-DSL中`print`可以打印在编译时的静态值, 例如上文中的va申明为constexpr. 而cute.print则可以在运行时打印.

运算符重载支持如下多种类型

Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`

Comparison: `<`, `<=`, `==`, `!=`, `>=`, `>`

Bitwise: `&`, `|`, `^`, <<, `>>`

Unary: `-` (negation), `~` (bitwise NOT)

```
@cute.jitdef operator_demo():    # Arithmetic operators    a = cutlass.Int32(10)    b = cutlass.Int32(3)    cute.printf("a: Int32({}), b: Int32({})", a, b)    x = cutlass.Float32(5.5)    cute.printf("x: Float32({})", x)    cute.printf("")    sum_result = a + b    cute.printf("a + b = {}", sum_result)    y = x * 2  # Multiplying with Python native type    cute.printf("x * 2 = {}", y)    # Mixed type arithmetic (Int32 + Float32) that integer is converted into float32    mixed_result = a + x    cute.printf("a + x = {} (Int32 + Float32 promotes to Float32)", mixed_result)    # Division with Int32 (note: integer division)    div_result = a / b    cute.printf("a / b = {}", div_result)    # Float division    float_div = x / cutlass.Float32(2.0)    cute.printf("x / 2.0 = {}", float_div)    # Comparison operators    is_greater = a > b    cute.printf("a > b = {}", is_greater)    # Bitwise operators    bit_and = a & b    cute.printf("a & b = {}", bit_and)    neg_a = -a    cute.printf("-a = {}", neg_a)    not_a = ~a    cute.printf("~a = {}", not_a)operator_demo()# outputa: Int32(10), b: Int32(3)x: Float32(5.500000)a + b = 13x * 2 = 11.000000a + x = 15.500000 (Int32 + Float32 promotes to Float32)a / b = 3.333333x / 2.0 = 2.750000a > b = 1a & b = 2-a = -10~a = -11
```

### 1.2 张量

然后是张量类型, 对于一个张量在CuTe中由Engine(E)和Layout(L)构成. 其中Engine主要是提供基于坐标offset后的元素访问, 而Layout则是定义各种坐标如何映射到实际的Offset. 我们可以将一个pytorch的张量在cuteDSL中采用不同的layout描述, 同时通过cute.print_tensor打印

```
import torchfrom cutlass.torch import dtype as torch_dtypeimport cutlassimport cutlass.cute as cuteimport cutlass.cute.runtime as cute_rta = torch.arange(1, 40, 1, dtype=torch_dtype(cutlass.Float32))ptr_a = cute_rt.make_ptr(cutlass.Float32, a.data_ptr())@cute.jitdef create_tensor_from_ptr(ptr: cute.Pointer):    layout1 = cute.make_layout((4, 5), stride=(5, 1))    tensor1 = cute.make_tensor(ptr, layout1)    cute.print_tensor(tensor1)        layout2 = cute.make_layout((2, 10), stride=(1, 2))    tensor2 = cute.make_tensor(ptr, layout2)    cute.print_tensor(tensor2)    cute.print_tensor(tensor2,verbose=True)create_tensor_from_ptr(ptr_a)# outputtensor(raw_ptr(0x000000002581d6c0: f32, generic, align<4>) o (4,5):(5,1), data=       [[ 1.000000,  2.000000,  3.000000,  4.000000,  5.000000, ],        [ 6.000000,  7.000000,  8.000000,  9.000000,  10.000000, ],        [ 11.000000,  12.000000,  13.000000,  14.000000,  15.000000, ],        [ 16.000000,  17.000000,  18.000000,  19.000000,  20.000000, ]])tensor(raw_ptr(0x000000002581d6c0: f32, generic, align<4>) o (2,10):(1,2), data=       [[ 1.000000,  3.000000,  5.000000, ...,  15.000000,  17.000000,  19.000000, ],        [ 2.000000,  4.000000,  6.000000, ...,  16.000000,  18.000000,  20.000000, ]])        # print时添加verbose=True参数可以打印具体的坐标对应的值.tensor(raw_ptr(0x000000002581d6c0: f32, generic, align<4>) o (2,10):(1,2), data= ( (0,0)= 1.000000 (1,0)= 2.000000 (0,1)= 3.000000 (1,1)= 4.000000 (0,2)= 5.000000 (1,2)= 6.000000 (0,3)= 7.000000 (1,3)= 8.000000 (0,4)= 9.000000 (1,4)= 10.000000 (0,5)= 11.000000 (1,5)= 12.000000 (0,6)= 13.000000 (1,6)= 14.000000 (0,7)= 15.000000 (1,7)= 16.000000 (0,8)= 17.000000 (1,8)= 18.000000 (0,9)= 19.000000 (1,9)= 20.000000)
```

另一个重要的功能是Cute-DSL支持DLPACK, 我们以pytorch tensor和numpy数组为例

```
import numpy as npimport torchimport cutlass.cute as cutefrom cutlass.cute.runtime import from_dlpack@cute.jitdef print_tensor_dlpack(src: cute.Tensor):    cute.print_tensor(src)a = torch.randn(4, 5)print_tensor_dlpack(from_dlpack(a))b = np.random.randn(4,5).astype(np.float32)print_tensor_dlpack(from_dlpack(b))# outputtensor(raw_ptr(0x00000000147c7900: f32, generic, align<4>) o (4,5):(5,1), data=       [[-0.976391,  1.239386, -0.485398, -0.854000,  0.391258, ],        [ 2.096474, -1.257927,  0.552251,  0.145064,  0.854345, ],        [-1.461262,  0.257915, -0.783814, -0.694687,  1.760151, ],        [-0.466722,  2.175940, -0.427469, -0.791567, -0.211634, ]])tensor(raw_ptr(0x0000000013fa3b20: f32, generic, align<4>) o (4,5):(5,1), data=       [[ 1.204166,  0.768091,  0.620764, -0.122040,  0.255480, ],        [-0.450180, -1.648787,  0.000650,  2.055988,  0.998210, ],        [ 0.964397, -0.495548, -0.586975,  1.266420, -1.793599, ],        [-0.285070, -0.390202,  0.702245,  0.701813,  0.045866, ]])
```

对于Tensor内element操作如下:

```
import torchimport cutlass.cute as cutefrom cutlass.cute.runtime import from_dlpack@cute.jitdef tensor_access_item(a: cute.Tensor):    # access data using linear index    cute.printf("a[2] = {} (equivalent to a[{}])", a[2],                cute.make_identity_tensor(a.layout.shape)[2])    cute.printf("a[9] = {} (equivalent to a[{}])", a[9],                cute.make_identity_tensor(a.layout.shape)[9])    # access data using n-d coordinates, following two are equivalent    cute.printf("a[2,0] = {}", a[2, 0])    cute.printf("a[2,4] = {}", a[2, 4])    cute.printf("a[(2,4)] = {}", a[2, 4])    # assign value to tensor@(2,4)    a[2,3] = 100.0    a[2,4] = 101.0    cute.printf("a[2,3] = {}", a[2,3])    cute.printf("a[(2,4)] = {}", a[(2,4)])# Create a tensor with sequential data using torchdata = torch.arange(0, 8*5, dtype=torch.float32).reshape(8, 5)tensor_access_item(from_dlpack(data))print(data)## outputa[2] = 10.000000 (equivalent to a[(2,0)])a[9] = 6.000000 (equivalent to a[(1,1)])a[2,0] = 10.000000a[2,4] = 14.000000a[(2,4)] = 14.000000a[2,3] = 100.000000a[(2,4)] = 101.000000tensor([[  0.,   1.,   2.,   3.,   4.],        [  5.,   6.,   7.,   8.,   9.],        [ 10.,  11.,  12., 100., 101.],        [ 15.,  16.,  17.,  18.,  19.],        [ 20.,  21.,  22.,  23.,  24.],        [ 25.,  26.,  27.,  28.,  29.],        [ 30.,  31.,  32.,  33.,  34.],        [ 35.,  36.,  37.,  38.,  39.]])
```

### 1.3 结构体

CuteDSL还可以构造一些结构体, 这些结构体在后续的矩阵乘法中将会被经常用到

![图片](assets/292c92c4d077.png)

```
import cutlassimport cutlass.cute as cute@cute.structclass complex:    real: cutlass.Float32    imag: cutlass.Float32@cute.structclass MyStorage:    x: cutlass.Float32    y: cutlass.Int32    nested: cute.struct.Align[complex, 16]    mem: cute.struct.Align[        cute.struct.MemRange[cutlass.Float32, 768], 1024    ]@cute.kerneldef kernel():    tidx, _, _ = cute.arch.thread_idx()    allocator = cutlass.utils.SmemAllocator()    s = allocator.allocate(MyStorage)    s.x = 7.5 + tidx    s.nested.real = 13    s.nested.imag = 13.7    cute.printf("struct: x={}  nested={}+{}i", s.x , s.nested.real , s.nested.imag)@cute.jitdef hello_world():    kernel().launch(        grid=(1, 1, 1),   # Single thread block        block=(8, 1, 1)  # One warp (32 threads) per thread block    )cutlass.cuda.initialize_cuda_context()hello_world()
```

## 2. 控制流

Tri Dao在QuACK项目中有一个文档

### 2.1 For循环

For循环分为如下三种:

`range`: python built-in

`cutlass.range` : 和内置的range一样, 但是支持unroll和pipeline控制

`cutlass.range_constexpr`: 在编译时展开循环.

```
import cutlassimport cutlass.cute as cute@cute.jitdef control_for_loop(upper_bound : cutlass.Int32):    n = 5    for i in range(upper_bound):        cute.printf("python built-in range: {}",i)    # 支持带参数展开     for i in cutlass.range(upper_bound, unroll=4):        cute.printf("cutlass range: {}",i)    # 使用常量表达式作为循环    for i in cutlass.range_constexpr(n):        cute.printf("cutlass range(const_expr): {}",i)control_for_loop(8)
```

在CuteDSL上还对Pipeline进行了一个小的优化, 传统的代码中, 通常需要手工去控制一个Prefetch Stage Loop

```
@cute.jitdef example():    ...    # build a circular buffer    buffer = ...    # prefetch loop    for i in range(prefetch_stages):        cute.copy(atom, gmem[i], buffer[i], ...)    # main loop    for i in range(bound):        if i + prefetch_stages < bound:            cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)        use(buffer[i % total_stages])    ...
```

而在CuteDSL可以直接在for loop中定义Prefetch stage:

```
@cute.jitdef example():    ...    # build a circular buffer    buffer = ...    for i in cutlass.range(bound, prefetch_stages=prefetch_stages):        # Compiler automatically handles the pipelining:        # - Generates prefetch loop for initial stages        # - In main loop, prefetches future data while using current data        cute.copy(atom, gmem[i], buffer[i % total_stages], ...)        use(buffer[i % total_stages])  # Uses data from previous iterations    ...
```

CuteDSL将自动生成Prefetch loop的代码. 后面在一些GEMM代码分析的时候会详细展开.

### 2.2 If-Else

基于Const-expr做branch condition时可以在编译期进行处理, 例如一个算子是否需要包含ReLU相关的代码

```
@cute.kerneldef gemm(..., do_relu: cutlass.Constexpr):    # main GEMM work    ...    if cutlass.const_expr(do_relu):    # compile-time guard        # ReLU code is emitted only when do_relu is True        ...
```

当调用`gemm(..., False)`时, ReLU相关代码在生成IR时就被省略了.

当然, 针对动态变量的分支判断也是支持的, 但const expr 无法接受动态变量作为参数输入

```
@cute.jitdef main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):    # ✅ This branch is Python branch, evaluated at compile time.    if cutlass.const_expr(const_var):        cute.printf("Const branch\\n")    else:        cute.printf("Const else\\n")    # ✅ This branch is dynamic branch, emitted IR branch.    if dynamic_var == 10:        cute.printf("Dynamic True\\n")    else:        cute.printf("Dynamic False\\n")    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.    if cutlass.const_expr(dynamic_var == 10):        cute.printf("Bound is 10\\n")
```

### 2.3 While循环

同样也支持const expr和dynamic value, 但是condition无法支持 dynamic value作为const expr的参数

```
@cute.jitdef main(dynamic_var: cutlass.Int32):    n = 0    # ✅ This is Python while loop, evaluated at compile time.    while cutlass.const_expr(n < 10):        cute.printf("Const branch\\n")        n += 1    # ✅ This is dynamic while loop, emitted IR while loop.    while dynamic_var == 10:        cute.printf("Dynamic True\\n")        n += 1    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.    while cutlass.const_expr(n < dynamic_var):        n += 1
```

### 2.4 动态控制流的约束

需要注意的是, 和Python不同, Cute-DSL无法支持`break`,`continue`,`pass`或者抛出异常这些在控制流内Early-exit的行为. 以及在一些control body内无法对变量类型进行改变等..

```
@cute.jitdef control_flow_negative_examples(predicate: cutlass.Boolean):    n = 10    # ❌ This loop is dynamic, early-exit isn't allowed.    for i in range(n):        if i == 5:            break         # Early-exit    if predicate:        val = 10        # ❌ return from control flow body is not allowed.        return        # ❌ Raising exception from control flow body is not allowed.        raise ValueError("This is not allowed")        # ❌ Using pass in control flow body is not allowed.        pass    # ❌ val is not available outside the dynamic if    cute.printf("%d\\n", val)    if predicate:        # ❌ Changing type of a variable in control flow body is not allowed.        n = 10.0
```

## 3. TensorSSA

`TensorSSA`: 是一个 Python 类，在 CuTe DSL 中以的Static Single Assignment形式表示的张量值. 在编译器设计中, 静态单赋值形式(SSA) 是一种IR, 其中每个变量只赋值一次. TensorSSA 将这个概念引入到张量操作中, 意味着 TensorSSA 对象是**不可变的 (immutable)**.当你对它进行操作时 (如 x > 0), 它不会改变自身, 而是返回一个新的, 代表操作结果的 TensorSSA 对象.

例如可以对TensorSSA进行简单的LD/ST和基于运算符重载的运算

```
import numpy as npimport cutlassimport cutlass.cute as cutefrom cutlass.cute.runtime import from_dlpack@cute.jitdef load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):    a_vec = a.load()    print(f"a_vec: {a_vec}")      b_vec = b.load()    print(f"b_vec: {b_vec}")      res.store(a_vec + b_vec)    cute.print_tensor(res)a = np.ones(20).reshape((5, 4)).astype(np.float32)b = np.ones(20).reshape((5, 4)).astype(np.float32)c = np.zeros(20).reshape((5, 4)).astype(np.float32)load_and_store(from_dlpack(c), from_dlpack(a), from_dlpack(b))#outputa_vec: tensor_value<vector<20xf32> o (5, 4)>b_vec: tensor_value<vector<20xf32> o (5, 4)>tensor(raw_ptr(0x000000001a9d18d0: f32, generic, align<4>) o (5,4):(4,1), data=       [[ 2.000000,  2.000000,  2.000000,  2.000000, ],        [ 2.000000,  2.000000,  2.000000,  2.000000, ],        [ 2.000000,  2.000000,  2.000000,  2.000000, ],        [ 2.000000,  2.000000,  2.000000,  2.000000, ],        [ 2.000000,  2.000000,  2.000000,  2.000000, ]])
```

另一方面, 对于加载到寄存器中的数据还可以进行各种计算/转换/切片等操作

下面是TensorSSA和标量乘加以及Reduce的一个例子.

```
@cute.jitdef apply_slice(res: cute.Tensor, src: cute.Tensor, a : cute.Float32, b: cute.Float32):    src_vec = src.load()    res_vec = a * src_vec + b        if cutlass.const_expr(isinstance(res_vec, cute.TensorSSA)):        res.store(res_vec)        cute.print_tensor(res)    else:        res[0] = res_vec        cute.print_tensor(res)        # 还可以针对TensorSSA做Reduce操作    res_reduction = res_vec.reduce(        cute.ReductionOp.ADD,        0.0,        reduction_profile=0    )    cute.printf("Reduction sum: {}", res_reduction)shape = (4,5)a = np.arange(np.prod(shape)).reshape(*shape).astype(np.float32)res = np.empty(shape, dtype=np.float32)apply_slice(from_dlpack(res), from_dlpack(a), 4, 5)# outputtensor(raw_ptr(0x000000001bd8d520: f32, generic, align<4>) o (4,5):(5,1), data=       [[ 5.000000,  9.000000,  13.000000,  17.000000,  21.000000, ],        [ 25.000000,  29.000000,  33.000000,  37.000000,  41.000000, ],        [ 45.000000,  49.000000,  53.000000,  57.000000,  61.000000, ],        [ 65.000000,  69.000000,  73.000000,  77.000000,  81.000000, ]])Reduction sum: 860.000000
```

下面是一个张量切片的例子.

```
@cute.jitdef apply_slice(src: cute.Tensor, dst: cute.Tensor, indices: cutlass.Constexpr):    """    Apply slice operation on the src tensor and store the result to the dst tensor.    :param src: The source tensor to be sliced.    :param dst: The destination tensor to store the result.    :param indices: The indices to slice the source tensor.    """    src_vec = src.load()    dst_vec = src_vec[indices]    print(f"{src_vec} -> {dst_vec}")    if cutlass.const_expr(isinstance(dst_vec, cute.TensorSSA)):        dst.store(dst_vec)        cute.print_tensor(dst)    else:        dst[0] = dst_vec        cute.print_tensor(dst)def slice_1():    src_shape = (4, 2, 3)    dst_shape = (4, 3)    indices = (None, 0, None)    a = np.arange(np.prod(src_shape)).reshape(*src_shape).astype(np.float32)    dst = np.random.randn(*dst_shape).astype(np.float32)    apply_slice(from_dlpack(a), from_dlpack(dst), indices)slice_1()#outputtensor_value<vector<24xf32> o (4, 2, 3)> -> tensor_value<vector<12xf32> o (4, 3)>tensor(raw_ptr(0x000000001ab1aa10: f32, generic, align<4>) o (4,3):(3,1), data=       [[ 0.000000,  1.000000,  2.000000, ],        [ 6.000000,  7.000000,  8.000000, ],        [ 12.000000,  13.000000,  14.000000, ],        [ 18.000000,  19.000000,  20.000000, ]])
```

另外官方还给出了一个bcast的例子

```
import cutlassimport cutlass.cute as cute@cute.jitdef broadcast_examples():    a = cute.make_fragment((1,3), dtype=cutlass.Float32)    a[0] = 0.0    a[1] = 1.0    a[2] = 2.0    a_val = a.load()    cute.print_tensor(a_val.broadcast_to((4, 3)))    # tensor(raw_ptr(0x00007ffe26625740: f32, rmem, align<32>) o (4,3):(1,4), data=    #    [[ 0.000000,  1.000000,  2.000000, ],    #     [ 0.000000,  1.000000,  2.000000, ],    #     [ 0.000000,  1.000000,  2.000000, ],    #     [ 0.000000,  1.000000,  2.000000, ]])    c = cute.make_fragment((4,1), dtype=cutlass.Float32)    c[0] = 0.0    c[1] = 1.0    c[2] = 2.0    c[3] = 3.0    cute.print_tensor(a.load() + c.load())    # tensor(raw_ptr(0x00007ffe26625780: f32, rmem, align<32>) o (4,3):(1,4), data=    #        [[ 0.000000,  1.000000,  2.000000, ],    #         [ 1.000000,  2.000000,  3.000000, ],    #         [ 2.000000,  3.000000,  4.000000, ],    #         [ 3.000000,  4.000000,  5.000000, ]])broadcast_examples()
```

参考资料

[1] 
cuteDSL notebook: *https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks*