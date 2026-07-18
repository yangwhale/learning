# CuteDSL-1:  Introduction

> 作者: zartbot  
> 日期: 2025年9月21日 01:28  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496107&idx=1&sn=79ad7de13f2b21d280421750eca0a566&chksm=f995e169cee2687f2b816b5f19ea415b1ce8c4b90f6ffe6c86c0e3cba27d2cf9e0e0f723fe57#rd

---

### TL;DR

Hotchip 2025上有几个AI Kernel Programming的Session, 其中Tri Dao谈到了CuteDSL. 一直没有时间补充这一块的内容. 最近花一些时间来梳理一下这块内容, 并且后续再对比分析一下Triton和Tilelang. 这一篇主要以基本的CuteDSL介绍和安装测试为主. 主要参考了GTC25的Session s74639 《Enable Tensor Core Programming in Python with CUTLASS 4.0》[1]

## 1. 为什么需要CuteDSL

### 1.1 Cutlass

前面在[《Tensor》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3557619493198151684&scene=173&subscene=&sessionid=svr_32119fe6ccb&enterid=1722676230&from_msgid=2247491424&from_itemidx=1&count=3&nolastread=1#wechat_redirect)这个专题中对Cutlass已经有了很多很详细的介绍.

![图片](assets/7111d95f2dcd.png)

大概的一个观点是, 像 PyTorch, TensorFlow 这样的高层框架通过编译器隐藏了大量细节, 在常见用例上表现出色. 但对于算法创新 (algorithmic innovations) 和需要精细控制的高级硬件特性 (如PDL - Programming Dependent Launch), 它们的抽象层次过高, 无法满足需求.

这张图Tri Dao也引用了, Triton这些DSL就像是坐电梯, 很多东西都自动优化了处理了, 开发者只关注算法的逻辑, tile-based编程模型开发速度很便捷, 就像电梯很快就能上行到顶楼. 但是呢为了追求极致的性能需要细粒度的优化需要暴露更多的底层抽象. 而PTX和CUDA则需要手工的一步步的去调整, 就像爬楼梯一步步的爬. 但是它拥有最细致的调整能力.  因此Nvidia针对Tensor的计算构建了一个基于C++的模版库cutlass, 并引入了一些Tensor Layout的算法, 并且还允许开发者细粒度的控制, 就有点像坐扶梯的感觉, 相对于裸CUDA/PTX编程便捷一些, 但同时“自己在扶梯上走几步”也是可行的.

Cutlass定义了一个很好的抽象结构

![图片](assets/00ab002b5e38.png)

### 1.2 Cutlass的问题及CuteDSL的原因

C++ 模板元编程是 CUTLASS 实现其 "zero-cost abstraction" (零成本抽象) 的基石. 它允许在编译期生成针对特定数据类型, 布局, 硬件架构高度特化的代码, 避免了运行时的判断和虚函数调用开销. 但Cutlass其实也有很多痛点, 模版库编译很慢, 同时出错了debug也很困难. 然后现代的一些DL相关的开发都在Python上, 反复写pybinding也挺烦人的.

![图片](assets/bd2e3d7bf609.png)

那么是否能在Python中构建一个DSL来支持呢? 深度学习研究和应用的主战场在 Python. 将底层性能库直接暴露给 Python 开发者, 可以极大地缩短 "从想法到高性能实现" 的路径. 研究人员可以快速验证新的算子融合, 混合精度策略或稀疏算法, 而无需跨语言开发的障碍. 这降低了性能优化的门槛, 有可能催生更多自下而上的创新.

这就是Cutlass-DSL, 初始的版本支持了CuTe的一些底层的支持. 后续还会继续添加.

![图片](assets/2ad22ed3a895.png)

好处是编译时间短了很多, debug也相对方便点, 但是性能没有损失

![图片](assets/8f3f616437a7.png)

### 1.3 CuteDSL quick start

官方的PPT似乎是早期的版本, 正式的cutlass-dsl 4.2 使用如下, 首先系统需要安装CUDA 12.9 以上的版本, Python版本也需要大于3.12, GPU支持Ada/Ampere/Hopper/blackwell, 线上随手开了一台A10的机器, 然后安装cutlass-dsl库即可

```
 pip install nvidia-cutlass-dsl   #当然我们还会继续安装torch jupyter进行后续的一些开发 pip install torch jupyter
```

一个简单的Hello world实例如下

```
import cutlassimport cutlass.cute as cute# Kernel函数定义@cute.kerneldef kernel():    # Get the x component of the thread index (y and z components are unused)    tidx, _, _ = cute.arch.thread_idx()    # Only the first thread (thread 0) prints the message    if tidx == 0:        cute.printf("Hello world")@cute.jitdef hello_world():    # Print hello world from host code    cute.printf("hello world")    # Launch kernel    kernel().launch(        grid=(1, 1, 1),   # Single thread block        block=(32, 1, 1)  # One warp (32 threads) per thread block    )# 运行前需要初始化cuda_contextcutlass.cuda.initialize_cuda_context()hello_world()
```

## 2. CuteDSL Infra

### 2.1 Soul of CuTe DSL

![图片](assets/1cc8ad8b6465.png)

这一页阐述了如下内容, 概括了Cutlass Python的战略意图, 它描述了以 MLIR 为编译器基石, 将 CUTLASS C++ 中被验证成功的 CuTe 硬件抽象模型引入 Python, 打造一个既能让广大 Python 开发者轻松上手, 又能让性能专家进行极限优化的新一代 GPU 编程范式. 其终极目标是在不牺牲性能控制的前提下, 将 GPU 高性能核函数的开发生产力提升一个数量级.

一个基于 Python 的编程语言, 用于通过 CuTe 语义来编程 Tensor Cores, 以达到最佳性能.

支持在 Python 中进行 Kernel (核函数) 编写, 而不仅仅是调用已有的 CUTLASS Kernel.

由 CuTe 的抽象能力所驱动和赋能.

能够轻松地与流行的 Python 框架 (如 PyTorch) 集成.

精确地对硬件进行建模, 以实现对性能的完全控制.

基于 MLIR 框架, 以便利用 MLIR 生态系统的强大能力.

CuTe 是实现"高层抽象"与"底层控制"统一的基石. GPU 编程中最困难, 最易出错的部分就是处理多维数据的内存布局, 线程映射和索引计算. CuTe 将这些复杂的细节抽象成一系列定义良好, 可组合的"布局"对象和代数运算. 开发者可以通过操作这些高层对象来描述复杂的并行模式, 而不必手动计算指针偏移和线程索引, 从而在保证性能的同时极大地提高了生产力.

**因此, 对于CuTe Layout代数的理解是能够很好的使用CuTeDSL的关键**

另一方面, 通过支持像 DLPack 协议, CUTLASS Python Kernel 可以无缝地接收 torch.Tensor 作为输入, 并将结果直接写入另一个 torch.Tensor. 这意味着研究人员和工程师可以轻松地将他们定制的高性能 Kernel 插入到现有的 PyTorch 模型中, 实现端到端的加速, 又避免了写一些复杂的pybinding code.

### 2.2 Cutlass Python架构

下面这张图展示了Cutlass Python的架构, 通过一张流程图展示了 CUTLASS Python 的整体架构, 描绘了从用户编写的 Python 代码到最终在 GPU 上执行的完整路径.

![图片](assets/7f2779a2cbbd.png)

整个流程是一个典型的现代编译器架构:

**前端** Python 代码 + DSL 编译器. 负责处理语言语法, 类型检查, 并将高级概念转换成 IR. 这使得语言本身可以快速迭代, 而不影响后端.

**中间** MLIR + CUTLASS 栈. 这是优化的核心. 在 MLIR 层面, 可以执行各种与硬件无关或与硬件相关的优化, 例如循环展开, 算子融合, 内存布局优化. "CUTLASS 栈" 的存在表明, 编译器不仅仅是从头生成所有代码, 还会智能地链接或内联 CUTLASS 库中已有的, 经过专家调优的recipes.

**后端** NVVM/LLVM -> PTX -> SASS. 这一部分利用了 NVIDIA 现有且成熟的 CUDA 编译器工具链, 保证了最终生成的代码能够充分利用硬件特性.

然后初始的CuTeDSL版本支持了DSL Compiler和CuteDSL,  其中“Coming Soon”的就是近期会发布的CuTile.

### 2.3 如何使用Python写Kernel

这一页展示了如何通过cuteDSL构建Kernel, 通过直观的对比, 揭示了 CUTLASS Python 在**编程模型和开发者体验**上的巨大飞跃. 左侧为传统的Cutlass代码, 右侧为CuteDSL. 通过使用 `@cute.kernel` 装饰器标记Kernel函数.  然后通过使用 `@cute.jit` 装饰器标记在CPU端调用时启动Kernel并进行编译.

![图片](assets/b8e96c967ba8.png)

另一个很明显的对比是Template vs Pythonic, 代码会变得更加清晰

**C++:** CUTLASS C++ 的强大之处在于其模板元编程, 它能在编译时生成高度特化的代码, 实现零成本抽象. 但其代价是极差的可读性和可维护性. 开发者面对的是一个包含十几个模板参数的声明, 任何一个参数的错误都可能导致成百上千行难以理解的编译错误. 这种 "编译时配置" 的方式心智负担极重.

**Python:** CUTLASS Python 将这一切都转化为了 Python 的函数参数和类型提示. 模板参数变成了函数参数, 并且拥有清晰的命名和类型 (`tma_atom_a: cute.CopyAtom`), 这大大提高了代码的可读性. 配置核函数的行为从 "填写复杂的模板列表" 变成了 "向函数传递参数", 这更符合常规的编程直觉.

另一方面是 C++ 中, 模板参数是纯粹的编译时概念, 而函数参数是运行时概念. 两者之间的交互和逻辑编写起来非常复杂.`@cute.jit` 和 `@cute.kernel` 的设计巧妙地管理了这一界限. `@cute.jit` 函数运行在主机端 (Python 解释器中), 它可以进行常规的 Python 计算, 准备参数. 当 `@cute.jit` 函数调用 `@cute.kernel` 函数时, DSL 编译器接管. 传递给 `@cute.kernel` 的参数, 根据其类型 (例如, `cutlass.Constexpr` vs. `cute.Tensor` 的动态形状), 会被 JIT 编译器智能地判断为编译时常量或运行时变量.

这种方式使得开发者可以用统一的函数调用语法来处理编译时和运行时的配置, 复杂性被 DSL 自身消化了.

另外, `epilogue_op` 被标记为 `Constexpr`, 编译器就会在代码生成时将这个 lambda 函数内联并优化, 实现零开销的算子融合. 而 `mA_mkl` 的具体数据指针和维度则是运行时信息.

### 2.4 Pytorch集成

这页通过一个简单的代码示例, 展示了 CUTLASS Python 如何与主流深度学习框架 PyTorch 进行无缝集成.

![图片](assets/6f82056d7a60.png)

**无缝地将 torch.tensor 作为输入:** 代码直接将一个 CUDA 上的 `torch.tensor` 对象 `A_tensor` 传递给了期望 `cute.Tensor` 类型参数的 `jit_func`. 这展示了一种隐式的, 自动的类型转换.

**通过显式调用进行更精细的控制:** 注释中的代码展示了另一种方式. 使用 `from_dlpack` 函数可以显式地将 `torch.tensor` 转换为一个 `cute.Tensor` 对象, 并且可以链式调用 `.mark_layout_dynamic()` 等方法来进一步控制其属性.

DLPack 是一个开放的, 用于在不同深度学习框架之间交换张量 (Tensor) 数据的内存中标准. 它定义了一个 C 数据结构, 该结构包含了描述一个张量所需的所有信息:

指向数据内存的指针

设备类型 (CPU, GPU 等) 和设备 ID

数据类型 (int32, float32 等)

维度数量 (ndim)

形状 (shape)

步长 (strides)

字节偏移(offset)

DLPack 允许多个框架共享同一个设备上的内存块, 而**无需任何数据拷贝**. PyTorch, TensorFlow, CuPy, JAX, MXNet 等主流框架都支持这个协议.

![图片](assets/5e60a1a2b6eb.png)

这里对比的是**测试和验证**一个自定义核函数结果的流程, 进一步强化了 CUTLASS Python 的优势. C++ 的流程则冗长得多. 开发者需要关心许多底层细节: 实例化模板, 显式同步 GPU, 手动触发数据回传, 调用特定的比较函数. 每一项都是一个潜在的出错点.  Python整个验证过程只需要两行核心代码. 逻辑清晰直观: 1) CPU计算参考值; 2) 比较. 这是因为开发者可以利用 PyTorch 生态中现成的, 高度封装的工具.

**Python:** 开发者可以在一个交互式的环境 (如 Jupyter Notebook) 中快速编写代码, 运行, 看到 `assert_close` 的结果, 如果出错, PyTorch 会给出详细的差异报告 (例如最大误差, 误差的位置等). 然后可以立即修改代码, 再次运行. 这个"编码-运行-调试"的循环非常快.

**C++:** 传统的 C++ 工作流通常是 "编码 -> 编译 -> 运行". 编译本身就很耗时 (尤其是对于 CUTLASS 这样重度使用模板的库). 如果测试失败, 开发者只能得到一个 "Failed" 的字符串, 然后需要添加更多的 `printf` 或者使用专门的 GPU 调试器 (如 `cuda-gdb`) 来定位问题, 整个过程要慢得多.

![图片](assets/4187b101b016.png)

`cute.Tensor` 的布局被视为其类型的一部分. 因为 `(3:1)` 和 `(5:1)` 是不同的布局, 所以它们被识别为不同的类型, 从而触发了两次独立的编译.

![图片](assets/4e99c1c5d537.png)

通过调用 `.mark_layout_dynamic(mode=[0])` 方法, 开发者告诉编译器, 张量的第0个模式 (mode, 即维度) 的大小是动态的, 不应被硬编码到核函数中. 两种不同大小的输入 `A_tensor` 和 `B_tensor` 复用了同一个已编译的核函数.

静态的编译, 编译器可以进行更精确的内存依赖分析, 边界检查等, 甚至在编译时就能发现一些 out-of-bounds 错误. 但是针对推理场景, 在处理动态输入 (如不同 batch size, 不同序列长度) 的场景下, 每遇到一个新的形状, 都会触发一次代价高昂的编译, 导致整体性能严重下降.

通过 `mark_layout_dynamic()`, 开发者从"隐式"的默认行为切换到了"显式"的控制. 开发者**主动告知**编译器: "这个维度的大小在运行时是可变的, 请不要将它硬编码."

下图展示了在LLM中如何通过自定义Kernel替换原来的nn.Linear的例子

![图片](assets/374f08c0c5c0.png)

然后在这个自定义的Linear类中, 可以实例化一个自定义的Kernel函数, 在前向传播过程中调用该核函数, 并利用了从 DLPack 隐式转换的特性.

![图片](assets/b2dc1a824c63.png)

MyCutlassLinear 模块将底层的 MyGemmKernel 封装了起来. 这是一个很好的软件工程实践. 模块的使用者 不需要知道内部是用了 CUTLASS, 还是 cuBLAS, 或者是其他任何实现. 它只需要调用 forward 方法即可. 另一方面Kernel本身也带了TMA/Tile大小等一系列调整参数, 这些参数直接映射到底层硬件的能力和性能调优的关键选项, 使得CuteDSL非常简便的让用户得到了更完整的硬件控制能力.

### 2.5 命令式风格进行元编程

接下来两页ppt展示了在Kernel函数中, 利用python直接进行分支跳转和循环的任务.

![图片](assets/6969e0c3a475.png)

CUTLASS Python 允许开发者用几乎完全相同的逻辑 `if tidx < cute.size(A): ...` 来实现同样的功能, 而无需关心底层的编译细节. meta-kernel这个词在这里的含义是, 你写的这份 Python 代码是一个模板, JIT 编译器可以根据你是否提供了动态布局, 来决定是生成一个包含动态 `if` 的通用核函数, 还是一个移除了 `if` (因为条件在编译时已知) 的特化核函数.

![图片](assets/ee23df283d4f.png)

然后扩展了动态控制流的概念, 从 `if` 语句推广到了 `for` 循环. `range` 的行为如上所述, 生成一个标准的动态循环. `range_dynamic(..., unroll=1)` 提供了一个额外的**unroll hint** 给编译器.

然后还有一些constexpr的能力

![图片](assets/9d5c539d4fde.png)

并且在CuteDSL中还可以通过kernel函数传参的方式,将TileCopy和SMEM Layout传入

![图片](assets/25d1ef3d545e.png)

在C++中, TiledMma, GmemTiledCopy, SmemLayoutAtom 等策略都是类型. 它们作为模板参数传入. 而python则是可以通过函数参数直接传递. 它将 C++ 中必须在编译时固化的核心算法策略 (如 TiledCopy, SMEM Layout, TiledMma), 变成了可以在运行时动态构建和传递的一等公民对象. 极大地增强了代码的灵活性, 可组合性和可读性, 同时又不以牺牲最终性能为代价.

### 2.6 DSL DataType

![图片](assets/cccbd4112c7c.png)

它覆盖了全面的数据类型支持能力, 发者不需要学习多套类型系统, 降低了心智负担.torch.dtype 和 numpy.dtype 的支持是实现与生态无缝集成的关键. 当一个 torch.Tensor 通过 DLPack 传递过来时, CUTLASS Python 不仅获取了数据指针和形状, 还会检查其 torch.dtype (例如 torch.float16), 并将其自动映射到内部对应的 cutlass.Float16 类型.

### 2.7 运算符重载

![图片](assets/440ab153734b.png)

允许开发者使用自然的 Python 运算符来操作 DSL 中的数据类型. 文中提到的`arith.muli(a, arith.constant(a.type, 4))`. 这种表示对于编译器来说非常清晰, 但对于人类来说则非常冗长和不直观. 因此提供了一个语法糖.对于提升代码的可读性, 减少心智负担, 加快开发和调试速度具有很大的价值.

另一方面是向量化的运算符重载能力.

![图片](assets/293ad0218dc9.png)

`TensorSSA`: 是指基于Tensor的Static Single Assignment. 在编译器设计中, 静态单赋值形式(SSA) 是一种IR, 其中每个变量只赋值一次. `TensorSSA` 将这个概念引入到张量操作中, 意味着 `TensorSSA` 对象是**不可变的 (immutable)**.当你对它进行操作时 (如 `x > 0`), 它不会改变自身, 而是返回一个新的, 代表操作结果的 `TensorSSA` 对象.

而Thread Local Data指的是, 对存储在寄存器中的数据进行建模. 每个线程都有一份自己的寄存器, 所以这些数据是线程私有的. 一个 `TensorSSA` 对象代表的不是一个标量, 而是一组由单个线程持有的, 存在于寄存器中的数据分片.

同时在右侧示例可以看到, 它还支持通过lambda函数调用.

`epilogue_op` 是一个**运行时传入的参数**. 这意味着用户可以在主机端**任意定义**他们想要的融合操作, 而无需修改 GEMM 核函数的内部代码.`lambda x: cute.where(x > 0, x, cute.full_like(x, 0))`被调用时, 它的输入 `x` 是一个 `TensorSSA` 对象, 代表了寄存器中的一批数据. 以下有几个例子:

**GeLU 融合:** epilogue_op=lambda x: 0.5 * x * (1 + cute.tanh( ... ))

**乘以一个标量 (alpha):** epilogue_op=lambda x: x * alpha

**加偏置 (bias) 并激活:** epilogue_op=lambda x: cute.relu(x + bias)

向量化的版本让编译器更容易识别出整体操作, 从而生成更优化的代码.

### 2.8 Cute结构体

![图片](assets/8cc9ce755d2e.png)

`@cute.struct` 是 CUTLASS Python 的一个非常重要的特性, 它将 GPU 编程中最繁琐, 最容易出错, 但又对性能至关重要的SMEM布局管理, 抽象成了一种类型安全, 声明式且高度可控的 Pythonic 方式. 它不仅模拟了 C 语言 `struct` 的精确控制能力, 更通过与 DSL 其他部分的集成和 Python 的动态特性, 提供了更强的灵活性和表达力, 大大降低了编写和维护高性能, 复杂 CUDA 核函数的难度.

![图片](assets/198a5a9bbe51.png)

这一页展示了 `@cute.struct` 如何让SMEM的管理变得更加结构化和面向对象.

传统方式是"先分配一块 Int64, 再分配一块 Int64, 再分配一块对齐到1024字节的张量..." 然后每个stage还需要做大量的指针运算, 很容易犯错. 而`@cute.struct` 装饰器会负责处理"如何"实现这个布局的复杂细节. 封装在 `SharedStorage` 类中的做法,使得复杂的SMEM管理可以被组织成一个清晰, 内聚且可复用的组件.

### 2.9 JIT缓存

在多次迭代的过程中, JIT都会执行编译导致kernel launch变慢

![图片](assets/340254d9cb1d.png)

然后cuteDSL提供了可以编译成cubin的KV Cache方式来处理这个问题

![图片](assets/5ed58c027b2e.png)

![图片](assets/f4524b35c93b.png)

![图片](assets/76b0b72af04f.png)

## 3. 基于CuTe的Python Kernel开发

### 3.1 介绍CuTe

![图片](assets/2e2a82c66fbf.png)

对于一个高性能的Kernel通常需要考虑不同代际的GPU架构, 并都保持保持峰值性能. 这部分内容详细展开可以参考

[《Tensor-007 Cute Layout简介》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491741&idx=1&sn=c1eed8d4c5d7c20bd3cd1ee660062d28&scene=21#wechat_redirect)

Cute Layout提供了一个单一的层级化的代数抽象来应对Layout.提供了一套操作这些对象的**代数运算**, 如 `composition`, `partition`, `tile`. 开发者可以用这些高阶函数来组合, 切分, 变换布局, 而无需关心底层的指针运算. 这就像用线性代数来操作矩阵, 而不是手动操作二维数组的每个元素.

CuTe 的核心是抽象的数学概念 (Layout 和代数), 而不是具体的硬件实现, 所以它的编程范式是持久的. 当新一代 GPU 出现时, 可能会有新的 `Atom Layout` (例如, 描述一个新 MMA 指令的寄存器布局), 但组合和操作这些 `Layout` 的代数和惯用法 (`make_tiled_mma`, `partition_A` 等) 保持不变. 这使得为新硬件编写代码的学习曲线大大降低.

而CuteDSL也依旧继承了这套Layout代数

![图片](assets/0cf1e5dfea54.png)

注: 后续有几页PPT讲述了一些example, 这里略去, 后面几篇再来完整的带着代码实现.

然后就是一些Performance的对比, 基本上和C++没有太大的差异.

![图片](assets/06c28de908f3.png)

## 4. DSL for GPU Kernel

然后另外一个视角来自于Tri Dao在Hotchip上的演讲《DSL for GPU Kernels & Automatic Kernel Authoring with LLMs》, 很喜欢下面这个图

![图片](assets/6d87211034f0.png)

前几天有一篇文章从硬件的角度介绍了一些设计上的TradeOff和DSE

[《从GPU缓存的视角看芯片设计和互连》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495963&idx=1&sn=00f05c90d7ec22f90911ac4618180c9a&scene=21#wechat_redirect)

而Tri Dao则是从软件和算法的角度来看, DSL和一个好的抽象能够加快算子的开发速度和降低开发难度. CuteDSL暴露了完整的芯片结构体系

![图片](assets/c64f11a1ddac.png)

其实他也提到其它DSL, 个人觉得Tilelang是一个同等的不错的选择, 因为Cutlass Layout代数还是太复杂了, TileLang没有底层的一些历史负担, 可能做的会更好.

![图片](assets/9eae3a23826b.png)

Tri Dao展示了几个例子, 例如Cute DSL中的一些async copy能力

![图片](assets/79636fa53064.png)

TensorSSA来做一些Reduction的操作

![图片](assets/87b646ff1b50.png)

以及一些warp level的reduction

![图片](assets/b82e1971a976.png)

然后是Thread Block的Reduction

![图片](assets/75a4ddb3f3cf.png)

通过Python `cute.jit`函数封装后, 都可以做到非常简单快速的操作.

![图片](assets/b43dcfcfb9c1.png)

并且整个Kernel性能基本上都能打满, 相对于Torch Compile和Triton都有很大的优势.

![图片](assets/54220ca206a2.png)

GEMM的算子也都比cublas快

![图片](assets/c55ee508944b.png)

![图片](assets/6ed3fe2e22ef.png)

算子融合上也比cublas+triton快不少

![图片](assets/c14aa2c70f2a.png)

另外还有一些FlashAttention-4的例子

![图片](assets/3f190aa57107.png)

然后Tri Dao也给出了一些Trade-off

![图片](assets/b67cf1eefa62.png)

CuteDSL可能唯一的缺点就是学习周期相对长一些, 希望我这个系列的文档在自己学习的时候也能够帮助社区的同学. 然后Tri Dao还给了一些建议

![图片](assets/5a171a41d65b.png)

## 5. 小结

本文算是介绍Cute-DSL的第一篇, 大概介绍了整个项目的由来和一些基本的概念. 后面再根据官方的一些notebook和Tri Dao的QuACK[2]来补一些实际的代码开发相关的内容吧.

![图片](assets/2cb6f8387723.png)

参考资料

[1] 
Enable Tensor Core Programming in Python with CUTLASS 4.0: *https://www.nvidia.com/en-us/on-demand/session/gtc25-s74639/*
[2] 
QuACK: *https://github.com/Dao-AILab/quack*