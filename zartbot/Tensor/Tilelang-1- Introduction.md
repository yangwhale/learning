# Tilelang-1: Introduction

> 作者: zartbot  
> 日期: 2025年10月2日 12:23  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496243&idx=1&sn=a5c8beb7e4872d13b9b2f495bacc2be1&chksm=f995e2f1cee26be783f4c879f8d1338391da97a685966b86aa0e32ca86efd3425c545d236ee7#rd

---

### TL;DR

其实很早就在关注Tilelang, 特别是在今年三月的时候, 看到一篇知乎上的文章《TileLang: 80行Python kernel代码实现FlashMLA 95%的性能 》[1]. 我想这个东西应该对所有的基模训练团队, 特别是一些算法团队有很大的帮助吧.  因为我们可以为自己的算法很快速的开发一个接近峰值性能的算子, 对于一些小规模的验证很大程度的提高了效率, 即便是最后大规模的训练也可以先训练起来. 等算子团队完善优化后再替换, 这样也可以节省不少时间. 然后九月初和薛老师一起也聊到这方面的工作, 后来就买了一块Jetson Thor, 它的SM和B200是相同的, 都有TMEM. 不过当时还在等Tilelang支持blackwell, 于是就先开始玩了一下CuteDSL. 最近发现Tilelang B卡已经支持了, 昨天很快的hack了一下, 发现在Thor上也能用了.

这是Tilelang整个系列学习笔记的第一篇, 基于Tilelang的论文《TileLang: A Composable Tiled Programming Model for AI Systems》[2]大概overview的介绍一下.

## 0. 为什么需要TileLang

大概的一个观点是, 像 PyTorch, TensorFlow 这样的高层框架通过编译器隐藏了大量细节, 但对于算法创新和需要精细控制的高级硬件特性, 它们的抽象层次过高, 无法满足需求.

![图片](assets/2ef1f45a33c3.png)

这张图Tri Dao也引用了, Triton这些DSL就像是坐电梯, 很多东西都自动优化了处理了, 开发者只关注算法的逻辑, tile-based编程模型开发速度很便捷, 就像电梯很快就能上行到顶楼. 但是呢为了追求极致的性能需要细粒度的优化需要暴露更多的底层抽象. 而PTX和CUDA则需要手工的一步步的去调整, 就像爬楼梯一步步的爬. 但是它拥有最细致的调整能力. 因此NV通过模版引入了Cutlass, 相对来说简化了不少, 但是Cutlass本身修改/编译/执行的速度还是很慢.

### 0.1 Triton的问题

现有方法(如Triton)虽然简化了编程, 但通常以牺牲底层控制为代价, 这限制了专家级开发者榨干硬件的极致性能.

![图片](assets/1336207cab44.png)

但是不可否认的是这种Pythonic的编程方式更容易让很多做算法的同学接受. 因此NV在逐渐的通过CuteDSL和CuTile补全这一块. 然后Tri Dao的QuACK似乎也在CuteDSL上提供一些原语.

### 0.2 Tilelang

TileLang会更干净一些, 同时对于不同的硬件平台支撑也好很多. 这次DeepSeek DSA发布后, 国产卡相对来说有一个统一的Tile based IR层.

![图片](assets/bdca362f7e0c.png)

TileLang的核心设计哲学是**将数据流(Dataflow)与调度策略(Scheduling)解耦**.  正如Tilelang论文所讲

*TileLang decouples scheduling space (thread binding, layout, tensorize and pipeline) from dataflow, and encapsulated them as a set of customization annotations and primitives.*

**数据流**: 开发者使用一系列高级的, 可组合的**Tile算子**(如 `T.gemm`, `T.copy`)来描述计算的核心逻辑, 即数据如何在不同层级的存储(全局内存, 共享内存, 寄存器)之间移动和被处理.

**调度**: 硬件相关的优化, 如线程绑定, 内存布局, Tensorization和Pipeline等, 被封装成一系列独立的注解和原语. 编译器会默认进行自动化优化, 但同时也允许专家开发者通过这些原语进行精细的手动调优.

通过这种方式, TileLang试图在**易用性**和**灵活性/高性能**之间取得更好的平衡. 论文通过在NVIDIA和AMD GPU上的大量实验证明, TileLang能够达到甚至超越当前最先进的专用库(如cuBLAS, FlashAttention-3)和编译器(如Triton)的性能, 同时代码实现更为简洁.

核心问题是, 尽管领域特定编译器(DSL)试图减轻编写高性能Kernel的负担, 但它们常常在易用性和表达能力方面存在差距.作者指出现有工具(如Triton)存在两个主要问题:

`Usability gaps` (易用性差距): 可能仍然不够简单, 对初学者有门槛.

`Expressiveness gaps` (表达能力差距): 这是更核心的批判. 现有工具为了追求易用性, 隐藏了过多的底层细节, 导致专家开发者无法实现某些高级的或非标准的优化技巧. 比如, 用户想用一个非常特殊的内存布局来配合某种数据类型, 但编译器不允许或不支持, 这就是表达能力不足.

关键问题是在`Scheduling space`怎么做. 描述如何在物理硬件上高效地执行这个数据流. 作者明确列出了调度的四个关键维度:

`thread binding`: 哪个线程负责哪部分数据.

`layout`: 数据在内存中如何排列.

`tensorize`: 如何使用硬件的专用矩阵/张量计算单元(如Tensor Core).

`pipeline`: 如何重叠数据移动和计算以隐藏延迟.

然后通过`customization annotations and primitives`实现数据流(Dataflow)与调度策略(Scheduling)解耦. 调度策略不是一个完全由编译器决定的黑盒. TileLang将它们暴露为用户可以使用的"开关"或"旋钮"(注解和原语).

## 1. Introduction

文章先介绍了过去几年硬件的一些发展和对应的专用硬件的Kernel演进, 例如FlashAttention这样的自定义Kernel已经出现, 用于优化注意力机制, 减少内存开销并提高处理吞吐量. 尽管如此, 在不断演进的加速器硬件上实现高效率, 仍然依赖于硬件感知设计和复杂调优的精妙结合, 这些挑战激发了人们对更具表达能力的领域特定编译器日益增长的兴趣.

### 1.1 高性能Kernel的几个挑战

深度学习Kernel通常被表示为数据流模式, 其中涉及在DRAM和SRAM之间移动数据分块(tiles), 并对这些分块执行一系列计算. 尽管这些模式表面上很清晰, 但构建高性能Kernel仍然充满挑战, 因为开发者必须手动解决几个关键的优化问题:

**线程绑定 (Thread Binding)**: 绑定是指将分块操作和数据映射到合适线程的过程. 在现代加速器架构中(如GPU), 这涉及到跨线程块(thread blocks), 线程束(warps)和单个线程仔细分配任务, 以最大化并行度并最小化负载不均衡. 一个最优的绑定策略能增强数据局部性, 减少线程同步和分歧带来的开销, 从而有助于提高计算吞吐量.

**内存布局 (Memory Layout)**: 内存布局优化需要系统地组织数据在物理内存中的排列, 以消除存储bank conflicts并确保高效的访问模式. 这个过程通常需要将数据的自然表示形式转换为与硬件内存子系统对齐的分块或块状格式. 这种重组有助于合并访问(coalesced accesses)和有效的缓存利用, 从而减少内存延迟并增强整体系统性能.

**硬件指令张量化 (Intrinsic Tensorization)**: 利用硬件固有函数(intrinsic functions)意味着直接使用为性能而优化的特定于目标的指令. 现代处理器和加速器提供了专门的操作——例如Tensor Core和Matrix Core, 它们可以同时执行多个算术运算, 同时还有像矢量拷贝(vector copy)和异步拷贝(asynchronous copy)这样的机制来更好地利用带宽. 使用这些固有指令需要对数据类型, 内存对齐和控制流进行精确管理, 以充分发掘硬件的计算能力, 从而在关键Kernel操作中带来显著的加速.

**流水线 (Pipeline)**: 流水线是一种通过重叠数据移动与计算来缓解内存访问延迟的技术. 通过并行调度数据传输和计算任务, 流水线确保处理单元保持活跃, 并最小化因内存延迟造成的空闲时间. 在先进的NVIDIA Hopper架构中, 张量内存加速器(Tensor Memory Accelerator, TMA)[10]可以通过为不同计算单元(如CUDA Cores和Tensor Cores)启用异步处理来促进此过程, 从而进一步增强并行性.

### 1.2 现有工具的局限性

尽管最近针对AI workload的一些DSL极大地简化了高性能Kernel的创建, 但它们仍然将大多数底层优化与Kernel实现交织在一起, 即使在数据流被明确暴露的情况下也是如此. 例如, **Triton**提供了直观的块级原语, 但将线程行为, 内存布局和地址空间注解隐藏在自动生成的策略之后. 这种抽象简化了编程, 但它阻碍了那些寻求榨取极致性能的经验丰富的开发者.

例如, 在实现带有量化权重的矩阵乘法时. 这类Kernel通常需要**内联汇编**来执行矢量化的数据类型转换, 以及需要与硬件特定内存缓冲区仔细对齐的自定义数据布局. 尽管Triton提供了像`tl.dot`这样的矢量化操作, 但将它们扩展到定制的用例, 例如通过PTX注册手工制作的高性能tile算子仍然很繁琐. 此外, 即使Triton暴露了一个用户友好的流水线控制(`num_stage`), 但它也**不允许用户定义一个完全自定义的流水线**. 因此, 领域专家在开发需要显式控制内存层次结构和其他细粒度优化的Kernel时受到了限制.

### 1.3 TileLang提出的原因

在保留Triton简洁性的同时提供更大灵活性的编程模型. TileLang旨在为用户提供对调度空间的细粒度控制, 以实现更高性能. 

实现这一点的关键在于**数据流与调度的解耦**: 用户只需专注于使用可组合的tile算子来定义数据流, 而编译器负责探索和应用调度策略. 当编译器的默认优化不尽如人意时, 用户可以在前端施加更精确的控制. 因此引入了一个可组合的分块编程抽象, 其中核心计算模式, 如GEMM, COPY, ATOMIC和REDUCE 都使用tile算子来表示. 这些算子独立于调度决策来定义Kernel的数据流. 与此同时提供了一套调度原语和注解来捕捉进一步的优化, 使用户可以选择依赖编译器生成的调度, 或手动微调性能关键的方面.

Triton的核心思想是让用户在块(block)的逻辑层面编程, 而编译器负责生成高效的线程级代码. 这在90%的情况下都很好. 但在剩下10%的极端场景, 这种"隐藏"就成了障碍.

"解耦数据流和调度"是TileLang的核心.

**数据流 (What to do)**: `T.copy`, `T.gemm` 等算子构成了计算的"语义图". 用户只需要像搭积木一样描述计算逻辑.

**调度 (How to do it)**: `T.Pipelined`, `T.Parallel`, `T.annotate_layout` 等原语则是对这个"语义图"的"渲染指令". 它们告诉编译器如何将逻辑图映射到物理硬件上.

这种解耦的好处是双重的:

**对于普通用户**: 可以忽略调度部分, 只关注数据流, 编译器提供一个"足够好"的默认调度, 大大降低了编程门槛.

**对于专家用户**: 当默认调度不满足性能需求时, 可以使用调度原语进行精确的手动干预. 这种干预是**结构化的**和**声明式的**, 比起直接写CUDA或PTX汇编, 仍然要高级和易于维护得多.

### 1.4 TileLang的实现

首先为了便于用户使用, 采用了Python实现了前端语言, 以支持灵活的编程风格和最少的类型注解. 此外为TileLang引入了一个编译器, 它能将用户定义的程序翻译成高度优化的底层代码, 以便在现代硬件上高效执行. 该编译器自动化了关键优化, 减少了性能调优所需的手动工作. 总的来说, Tilelang的贡献如下:

**Tile级编程语言 (Tile-Level Programming Language)**: 设计了一种分块级编程语言, 允许用户明确声明缓冲区在硬件内存层次结构中的位置. 通过利用一个**布局推导(Layout Inference)** 机制, 该系统在抽象掉高效并行化缓冲区操作复杂性的同时, 暴露了线程级控制接口, 使专家能够精确管理每个线程如何与缓冲区交互.

**带自动优化的编译器 (Compiler with Automated Optimization)**: 为TileLang提供了一个配套的编译器, 其中包含一系列自动化的passes. 这些pass包含的功能有: 通过布局推导机制实现自动并行化, 为Kernel库进行动态参数简化, 自动流水线派生, 以及为动态形状进行循环尾部切分优化. 这个编译器确保了TileLang程序既高效又易于编写.

**业界顶尖的性能 (State-of-the-Art Performance)**: 在真实AI Kernel上的经验评估表明, TileLang在NVIDIA和AMD GPU上均取得了与专业厂商库和其他基于DSL的方法(如Triton)相当, 有时甚至超越的性能.

## 2. A TileLang Example

### 2.1 设计原则和背景

现有的将调度与计算分离的机器学习编译器, 例如TVM, 要求用户明确地区分计算和调度. 此外, 用户必须手动注册新的张量指令并指定缓冲区布局以获得最佳性能. 然而, 编写和理解调度程序仍然具有挑战性.

尽管像Triton这样的现代框架允许用户专注于块级(tile-level)编程, 但它们的数据流表示通常不够清晰, 并且需要使用一些变通方法, 例如带掩码的条件加载或硬件特定功能, 如张量内存加速器(TMA).

虽然像ThunderKitten这样的框架将程序抽象为加载, 计算, 存储和同步操作的块粒度组合, 但它们的数据流仍然不够透明, 限制了用户应用进一步优化的能力.

此外, 随着基于Python的深度学习框架的广泛采用, 手动将模型翻译成C++进行优化是不切实际的. 因此, 在设计TileLang时, 作者强调三个关键原则:

`Pythonic设计 (Pythonic design)`: 与Python生态系统无缝集成, 提供熟悉的编码体验并降低学习曲线.

`以数据流为中心 (Dataflow-centric)`: 使用户能够主要关注数据流, 同时抽象掉底层的调度复杂性. 它将调度方面(例如线程绑定, 内存布局, 张量化和流水线)与数据流解耦, 将它们封装为一组可定制的注解和原语, 以增强可编程性和可维护性.

`可组合性 (Composability)`: 确保Kernel, 原语和调度策略可以无缝组合以构建复杂的设计.

### 2.2 GEMM示例代码讲解

以在TileLang中实现一个GEMM Kernel, 以说明其基本语法并展示它如何提高生产力. 如下所示

![图片](assets/da895062ef18.png)

首先定义了GEMM Kernel的输入和输出(第8行), 指定它们的形状和数据类型. 随后初始化Kernel上下文(第9-11行), 它决定了网格大小(grid size)和总线程数, 接着是Kernel主体(第12-27行), 其中包括片上内存分配和数据流管理. 由于TileLang是一种基于Python嵌入式编程语言, 它支持Python的所有命令式结构(例如, if-else, for, while), 关键区别在于用户必须为函数参数和变量声明提供**显式的类型注解**. 这个要求源于Python的动态类型特性, 它可能不天然适用于设备代码生成(例如CUDA/HIP), 因为静态数据类型对于确定精确的数据位宽至关重要. 在TileLang中, 类型注解明确定义了元素类型和张量形状, 确保了正确性和高效的代码生成.

此外, TileLang允许**显式内存分配**, 从而对数据放置和访问模式提供更强的控制. 在给定的实现中, TileLang使用`T.alloc_shared`将A和B的子矩阵存储在共享内存中, 而使用`T.alloc_fragment`在块级别将累加器分配在寄存器文件中. 此外, 使用流水线执行(`T.Pipelined`)可以使内存传输与计算重叠, 有效地隐藏内存延迟并提高整体吞吐量. `T.gemm`操作利用NVIDIA CUTLASS或手工编写的HIP代码来高效地执行块级矩阵计算. 通过自动化底层的调度和同步, TileLang允许开发者专注于算法设计而不是硬件特定的优化, 从而在保持计算效率的同时提高生产力.

**L1-L7**: 定义算法参数, 如矩阵维度, 分块大小, 流水线阶段数和线程数.

**L8**: `def Matmul(A: T.Tensor, B: T.Tensor, C: T.Tensor):` 定义Kernel函数. `T.Tensor`是TileLang的类型注解, 用于标记这是一个需要被编译器处理的张量对象.

**L9-L11**: `with T.Kernel(N // block_N, M // block_M, threads=threads) as (bx, by):`

`T.Kernel`是Kernel的入口.

`N // block_N`, `M // block_M`它决定了网格大小(grid size)和总线程数, 这意味着每个线程块负责计算输出C的一个`block_M x block_N`大小的子块.

`threads=threads`指定每个线程块包含128个线程.

`as (bx, by)`: 从Context返回`bx`和`by`, 它们分别代表当前线程块在网格中的x和y坐标, 类似于CUDA中的`blockIdx.x`和`blockIdx.y`.

**L13-L15**: `T.alloc_shared(...)`, `T.alloc_fragment(...)`. 这是显式的片上内存分配. `A_shared`和`B_shared`在共享内存中, `C_local`在寄存器文件中. 注意此时的视角是**块级**的.

**L18**: `T.clear(C_local)`. 将块级的累加器清零.

**L21**: `for k in T.Pipelined(K // block_K, num_stages):`. 这是数据流的核心循环. *   `T.Pipelined`是一个特殊的迭代器, 它告诉编译器这个循环体需要被流水线化. `K // block_K`是循环的总次数.

**L22-L23**: `T.copy(...)`. 描述数据流: 将全局内存`A`和`B`的相应分块拷贝到共享内存`A_shared`和`B_shared`. 索引计算`by * block_M`, `k * block_K`等利用了`by`和循环变量`k`来定位正确的数据分块.

**L24**: `T.gemm(A_shared, B_shared, C_local)`. 描述计算: 在共享内存中的两个分块上执行矩阵乘法, 结果累加到寄存器`C_local`中.

**L27**: `T.copy(C_local, C[...])`. 循环结束后, 将寄存器中的最终结果写回到全局内存`C`的相应位置.

**L31**: `tilelang.compile(...)`. 触发即时编译(JIT compilation).

### 2.3 编译流程

当调用`tilelang.compile`, Tilelang程序将降级成下图(b)中的IR, 随后被进一步生成Cuda Code

![图片](assets/c1ab5a00e402.png)

图(b)中的IR
`T.decl_buffer`: `alloc`操作变成了更底层的声明. 注意`A_shared`和`B_shared`被展平成了一维数组, 大小为`4096 = 128 * 32`. `C_local`也被降低为线程级的缓冲区, 大小为`128`.

`T.thread_binding`: `bx`, `by`, `tid`这些变量被显式地绑定到了硬件的`blockIdx.x`, `blockIdx.y`, `threadIdx.x`.

`T.unroll`: 循环被标记为需要展开.

**流水线展开**: `T.Pipelined`循环消失了, 取而代之的是显式的`cp_async`(异步拷贝), `cp_async_commit`(提交异步拷贝), `cp_async_wait`(等待异步拷贝完成)指令, 与`gemm_ss`(ss代表shared-shared输入)计算交错在一起. 这清晰地展示了软件流水线的具体实现.
图(c)中的Cuda代码
`tl::cp_async_gs`, `tl::cp_async_commit`, `tl::cp_async_wait`: 这些是TileLang提供的模板库函数, 封装了底层的PTX异步拷贝指令.

`tl::gemm_ss`: TileLang的GEMM模板库函数, 内部会调用CUTLASS.

### 2.4 一些分析

总体来看,当我们在写Tilelang程序的时候, 用户可以完全以Dataflow为中心. 首先为计算块分配暂存空间 (`A_shared`, `B_shared`, `C_local`), 然后初始化累加器(`T.clear`),  接下来循环K维度, 使用`T.copy`加载A, B的下一个分块到SMEM, 然后`T.gemm`计算并累加. 最终通过`T.copy`写回结果. 整个过程就像在画一个数据流图, 开发者完全不需要关心:

`T.copy`是如何被128个线程并行执行的? 内存访问是否合并了?

`T.gemm`是如何利用Tensor Core的? 数据是如何在共享内存和寄存器之间流动的?

`T.Pipelined`是如何将加载和计算重叠的? 异步拷贝指令和同步barrier应该插在哪里?

这些复杂的调度问题全部被抽象掉了, 交由编译器处理. 这与直接用CUDA编程形成了鲜明对比, 在CUDA中, 开发者必须手动处理上述所有问题. 同时和Triton做一些对比.

**显式内存声明**: TileLang的`T.alloc_shared`和`T.alloc_fragment`让数据的位置在代码中一目了然. Triton中, 指针操作隐式地处理加载, 而累加器通常就是普通的Python变量, 其在寄存器中的位置是隐含的. TileLang的显式声明为后续的布局推导和优化提供了更清晰的起点.

**原语的封装**: TileLang将`copy`, `gemm`, `clear`等操作封装成了高级API. Triton更多地使用指针运算和`tl.load`, `tl.store`, `tl.dot`等更低一级的原语. TileLang的API更高层, 更接近算法描述; Triton的原语更接近硬件操作.

**流水线表达**: TileLang的`T.Pipelined`是一个作用于`for`循环的迭代器封装, 语义上非常直观, "这个循环是流水线化的". Triton的`num_stages`是`@triton.jit`装饰器的一个参数, 作用于整个Kernel. TileLang的方式可能在表达嵌套循环或部分循环的流水线时更灵活.

**透明度**: Figure 1(b)展示的IR是TileLang设计的一大亮点. 它向用户(尤其是专家用户)展示了编译器是如何理解和转换高级代码的. 用户可以看到`T.Pipelined`被转换成了具体的`cp_async`操作顺序. 这种处理让开发者对编译器的行为有信心, 并在需要时能进行更有针对性的优化. Triton的编译过程则更像一个黑盒.

## 3. TIlelang 设计

本章详细介绍了 TileLang 的基础, 包括其基于Tile的编程模型, 以及 TileLang 如何系统地高效管理 Kernel 的开发. 本章的核心是阐述 TileLang 将 **数据流 (Dataflow)** 与其他 **调度空间 (Scheduling Space)** 分离的设计方法.

![图片](assets/13831e5a4036.png)

上图展示了 TileLang 的五阶段编译流水线.

**开发者**使用 TileLang 编写描述计算逻辑和数据访问模式的高级程序.

**解析器 (Parser)** 阶段将 TileLang 程序(Python 代码)解析为 Python AST (抽象语法树), 随之转换为 TileLang AST.

**IR 构建器 (IR Builder)** 阶段将 AST 转换为 TVM 的中间表示 (IR). 这样做可以复用 TVM 成熟的语法树结构和相关基础设施.

**优化 (Optimization)** 阶段对 IR 执行一系列图优化和调度变换, 以提升执行效率.

**代码生成 (Codegen)** 阶段将优化后的 IR 翻译为后端代码, 如 LLVM IR, CUDA C/C++ 或 HIP C/C++, 以支持不同的硬件平台.

下表展示了 TileLang 提供的一部分具有代表性的数据流算子和调度原语. Tile 语言拥抱一种 **以数据为中心的编程范式**, 其核心计算语义通过分块级算子(如 `T.copy`, `T.gemm`, `T.reduce`)来表达. 作为补充, TileLang 暴露了一系列调度原语, 允许开发者对性能关键方面(如并行度, 流水线和内存布局)进行微调.

![图片](assets/5995a151f2c2.png)

### 3.1 Tile-Based 编程模型

论文原文此处有个关于Figure.11的Typo, 实际的是在讲Figure.3. 它通过一个简单的GEMM示例, 展示了开发者如何使用高级结构 (如分块, 内存放置, 流水线和算子调用) 来精细控制数据移动和计算. 特别是, 该代码片段展示了多级分块如何利用不同的内存层级 (全局内存, 共享内存, 寄存器) 来优化带宽利用率和降低延迟. 总体而言, TileLang 的类 Python 语法允许开发者在一个用户友好的编程模型内对性能关键优化进行推理.

![图片](assets/1796e362aa75.png)

**Tile声明 (Tile declarations)**: 方法的核心是将 **Tile** 作为编程模型中的一等公民. 一个分块代表一部分有形状的数据, 它可以被一个 warp, 线程块或等效的并行单元拥有和操作. 在 Matmul 示例中, A 和 B 缓冲区在 Kernel 循环内部以分块的形式 (由 `block_M`, `block_N`, `block_K` 决定) 被读取. 通过 `T.Kernel`, TileLang 定义了执行上下文, 包括线程块索引 (`bx`, `by`) 和线程数量. 这些上下文可以帮助我们计算每个线程块的索引, 并使得 TileLang 更容易地自动推断和优化内存访问与计算. 此外, 这些上下文也允许用户手动控制线程块内每个独立线程的行为.

**显式硬件内存分配 (Explicit Hardware Memory Allocation)**: TileLang 的一个标志性特征是能够显式地将这些分块缓冲区放置在硬件内存层级中. TileLang 不依赖于编译器不透明的优化过程, 而是暴露了面向用户的内建函数 (intrinsics), 这些函数直接映射到物理内存空间或加速器特定的结构. 具体包括:

**T.alloc_shared**: 在一个高速的片上存储空间中分配内存, 这对应于 NVIDIA GPU 上的 SMEM. SMEM是缓存计算过程中间数据的理想选择, 因为它比全局内存快得多, 并允许同一线程块内的线程高效共享数据. 例如, 在矩阵乘法中, 矩阵分块可以被加载到SMEM中以减少全局内存带宽需求并提升性能.

**T.alloc_fragment**: 在 **fragment memory** 中分配累加器, 这对应于 NVIDIA GPU 上的寄存器文件 (register files, RF). 通过将输入和部分和保存在寄存器或硬件级缓存中, 延迟被进一步最小化. 注意, 在这个分块程序中, 每个分块分配了与共享内存一样的本地缓冲区, 这可能看起来违反直觉, 因为共享内存通常比寄存器文件更充裕但速度稍慢. 这是因为这里的分配指的是整个线程块的寄存器文件. TileLang 在编译期间使用 **布局推导过程 (Layout Inference Pass)** 来派生一个 `T.Fragment` 布局对象, 该对象决定了如何为每个线程分配相应的寄存器文件. 这个过程将在后续章节详细讨论.

全局内存与硬件特定内存之间的数据传输可以使用 `T.copy` 来管理. 此外, 硬件特定缓冲区可以使用 `T.clear` 或 `T.fill` 来初始化. 对于数据赋值, 也可以使用 `T.Parallel` 并行地执行操作, 如同在下图中展示的那样.

![图片](assets/dc5c07eef678.jpg)

### 3.2 Dataflow Centric Tile Op

TileLang 抽象了一系列 **分块算子 (Tile Operators)**, 使得开发者可以专注于数据流逻辑, 而无需管理每个分块操作的底层实现细节. 图 4 展示了一个分块算子的接口以及几个代表性示例, 包括 `GEMM`, `Copy`, 和 `Parallel`.

![图片](assets/72df0ce7cf19.png)

每个分块算子被要求实现两个关键接口: `Lower` 和 `InferLayout`.

**Lower接口**: 定义了如何将高级的分块算子降级 (lower) 为更低级的 IR, 例如线程绑定或向量化的内存访问. 比如, `Copy` 可以被降级为一个带有显式线程绑定和向量化加载/存储的循环.

**InferLayout接口**: 负责确定与该分块算子相关联的内存和循环布局. 这包括推断缓冲区布局 (例如, swizzled memory) 或循环级布局 (例如, 线程绑定). 举例来说, `T.gemm` 会对其共享内存输入应用swizzled布局, 并使用一个矩阵特定的布局来写回 MMA fragments. 类似地, `T.Parallel` 中的并行循环结构可以用线程级绑定和向量化访问模式来表示, 这两者都是通过布局推导得出的. 4.1 节将更详细地讨论布局组合及其在降级过程中的作用.

前列出了一部分 TileLang 算子, 以简化分块编程中的常见操作. 这些内置算子抽象了硬件内存访问和计算的底层细节, 允许开发者从数据流的角度专注于高级算法设计, 同时保持对性能关键方面的精细控制. 下面描述几个关键算子:

**copy**: `copy` 操作是 `T.Parallel` 加上内存拷贝的一个语法糖, 它允许从/向 `fragment` 作用域 (寄存器), `shared` 作用域 (静态共享内存), `shared.dyn` 作用域 (动态共享内存) 和 `global` 作用域 (全局内存) 之间进行拷贝.

**gemm**: 内置的 `T.gemm` 算子是为通用矩阵乘法高度优化的实现, 支持多种内存访问模式 (`ss`, `sr`, `rs`, `rr`), 其中 `r` 代表寄存器内存, `s` 代表SMEM. 该算子会根据 Kernel 配置自动选择最优实现. 对于 CUDA 后端, `T.gemm` 利用 NVIDIA 的 CUTLASS 库来高效利用 Tensor Cores 或 CUDA Cores; 对于 AMD GPU, 它则采用 Composable Kernel 和手写的 HIP 代码进行性能优化. 用户还可以通过在 Python 中注册自定义原语来扩展 `T.gemm`, 使其对特定用例更加灵活.

**reduce**: `T.reduce` 算子为跨维度聚合数据提供了一个灵活高效的归约机制. 它支持多种归约操作, 如 `sum`, `min`, `max`, `product` 等. 归约可以跨指定轴进行, 从而实现矩阵的行归约或列归约等操作. `T.reduce` 的实现利用了 warp 级和线程块级的并行性, 以在 CUDA 和 AMD 后端上都获得最佳性能.

**atomic**: `T.atomic` 算子为在并行上下文中安全更新共享或全局内存提供了原子操作. 常见的原子操作如 `add`, `min`, `max` 都被原生支持. `T.atomic` 在并发更新期间确保线程安全,它旨在利用 NVIDIA 和 AMD GPU 上的原生硬件原子指令，在确保并行执行正确性的同时，实现高性能。

### 3.3 调度注解与原语 (Schedule Annotations and Primitives)

虽然数据流模式构成了组织运算的基础, 但现代高性能计算要求对执行模式进行更精细的控制. 为了满足这一需求, TileLang 提供了一套全面的调度原语, 使开发者能够精确调整其应用程序的性能关键方面, 如前表所述:

**Pipelined**: `T.Pipelined` 原语允许对循环进行高效的流水线化执行, 通过重叠计算和内存操作来提升性能. 在图 1 中, 遍历 k (归约维度) 的循环被流水线化, `num_stages=3` 创建了一个 3 级流水线. 这个流水线允许数据传输, 计算和后续的数据准备Overlap, 有效地减少了内存瓶颈并提高了计算吞吐量. 从 `T.Pipelined` 降级到 CUDA 源码的详细设计将在 4.4 节讨论.

**Parallel**: `T.Parallel` 原语通过将迭代映射到线程来自动并行化循环. 在图 8 中, 拷贝数据到 `A_shared` 的操作使用了 `T.Parallel(8, 32)` 在 8 和 32 两个维度上进行并行化. 它不仅通过利用硬件并行性提升性能, 还自动将线程映射到迭代, 并支持向量化以进行进一步优化.

**annotate_layout**: `T.annotate_layout` 原语允许你使用用户定义的内存布局来为共享或全局内存指定内存布局优化. 默认情况下, TileLang 采用一种旨在最小化 NVIDIA 和 AMD GPU 上 bank conflicts 的优化内存布局.

**use_swizzle**: `T.use_swizzle` 原语通过启用交错内存访问 (swizzled memory access) 来改善 L2 缓存的局部性, 提升了光栅化过程中的数据复用. 这个原语在并行线程块处理分块数据时特别有效.

在这一章详细阐述了Tilelang的设计原则, 如何通过 **分离数据流与调度** 来构建其编程模型.

将复杂的 GPU 编程分解为两个可管理的部分:

**"做什么" (What to do)**: 通过 `T.gemm`, `T.copy` 等数据流算子描述计算逻辑. 这部分对领域专家 (如算法研究员) 非常直观.

**"怎么做" (How to do it)**: 通过 `T.Pipelined`, `T.Parallel`, `T.alloc_shared` 等调度原语和内存分配接口控制执行方式. 这部分赋予了性能工程师精细调优的能力. 这种分离使得不同角色的开发者可以专注于自己擅长的领域.

另一方面是 **可组合性和可扩展性**: `Tile Operator` 接口 (`Lower`, `InferLayout`) 的设计是其扩展性的基础. 用户可以定义自己的算子, 只要实现了这两个接口, 就能无缝地融入 TileLang 的编译和优化流程.

## 4. 调度设计与自动化 (Scheduling Design and Automation)

这章将讨论除数据流之外的四种调度空间及其在 TileLang 中的自动化设计. 其中一些相对独立 (如流水线和张量化), 而另一些则耦合更紧密 (如线程绑定和内存布局设计). 在接下来的章节中, 首先解释内存布局基础设施的设计, 接着是线程绑定. 然后将讨论张量化的自动化设计, 最后分享流水线的设计.

### 4.1  内存布局组合 (Memory Layout Composition)

在 TileLang 中支持使用高级接口 (如 `A[i, k]`) 对多维数组进行索引. 这种高级索引最终通过一系列软件和硬件抽象层被翻译成一个物理内存地址. 为了对这种索引翻译过程进行建模, 引入了关键的抽象 **Layout**, 它描述了数据在内存中是如何组织和映射的.

在物理地址层面, 一个布局可以表示为一个线性地址表达式, 形式为 , 其中  表示第  维的索引,  是该维度对整个线性内存地址贡献的步长 (stride). 给定一个布局 , TileLang 采用了一种受 TVM启发的设计, 引入了一个基于 `IterVar` (迭代变量) 的可组合, 可堆叠的布局函数抽象. 由于一个 `IterVar` 可以封装步长信息, 布局表达式可以被简化为关于 `IterVar` 的代数形式. 因此, 一个布局函数可以正式地表示为一个映射 , 其中  编码了从高级索引到内存地址的转换.

![图片](assets/52c85a17265f.png)

图 5(a) 展示了 TileLang 中 `Layout` 的定义. 其核心组件包括 `iter_vars` (迭代变量, 可以选择性地携带范围信息) 和一组 `forward_index` 表达式, 这些表达式基于这些迭代变量计算内存位置. 这些表达式共同定义了一个代数函数 .

如图 5(b) 所示, 这允许表达一个 2D 到 1D 的布局转换. 给定缓冲区的形状, `iter_vars` 被绑定到特定区域, 产生的表达式被传递给算术分析器以确定符号或常数边界. 这些边界被用来推断转换后缓冲区的形状, 并相应地调整缓冲区访问索引.

TileLang 也支持非双射 (non-bijective) 的布局转换. 例如, 图 5(c) 展示了如何使用布局来对缓冲区访问应用填充 (padding). 这些布局转换是可组合的, 并且 TileLang 包含几种内置的布局策略, 例如布局交错 (layout swizzling), 这通常用于缓解 GPU 上的共享内存 bank conflicts.

此外, TileLang 引入了 `Layout` 抽象的一个扩展, 称为 **Fragment**. 与标准布局不同, `Fragment` 布局总是产生一个形式为  的输出, 其中两个输出维度分别代表 **线程在寄存器文件中的位置** 和 **在本地寄存器文件中的索引**. 例如, 在图 1 的 Kernel 中, 在线程块级别分配了一个寄存器文件 `C_local`. 然而, 由于 GPU 寄存器文件必须在块内线程之间划分, `Fragment` 布局提供了对这种划分方案的精确描述.

![图片](assets/cbd31cb7bcee.jpg)

图 6(a) 展示了 `Fragment` 布局的定义, TileLang 提供了四个原语操作来帮助用户扩展现有的 `Fragment` 布局. 图 6(b) 显示了一个示例, 说明如何使用这些原语从用于 `m16k16` 矩阵fragment的 `mma_ldmatrix` 指令的基础布局中派生出一个完整的块级布局. 这里, `base_layout` 表示一个 warp 消耗一个 `m16k16` 矩阵的布局. 这个布局通过 `repeat` 原语扩展成一个 `warp_layout`, 允许单个 warp 消耗一个 `m32k16` 矩阵. 图 6(c) 可视化了这一转换. 然后, `warp_layout` 被进一步使用 `repeat_on_thread` 和 `replicate` 等原语扩展, 生成一个 `block_layout`, 它代表四个 warp 共同消耗一个 `m128k16` 矩阵.

### 4.2 线程绑定 (Thread Binding)

在 `Fragment` 布局的抽象之上, 出现的一个关键挑战是如何在执行期间将这些布局映射到线程上. 这就引出了 **线程绑定 (Thread Binding)** 问题, 它涉及到如何将块级寄存器文件分配给单个线程, 以及如何推断合适的 `Fragment` 布局. 此外, 它还要求确定循环应如何被正确地并行化以匹配布局约束.

虽然 4.1 节引入的 `Fragment` 布局有助于简化此过程, 但为任意计算表达式确定所有缓冲区的合适 `Fragment` 布局仍然很困难. 通过两个关键观察来指导这个过程:

由于多个分块算子通常共享相同的缓冲区, 它们各自的布局和线程绑定策略是相互依赖的.

不同算子对布局和线程绑定的要求严格程度不同. 例如, 在 GPU 上, `GEMM` 算子 (利用 Tensor Cores) 对布局和线程绑定都施加了严格的约束, 而逐元素算子通常允许更大的灵活性.

基于这些观察, 提出了一种基于 `Layout` 和 `Fragment` 对象的推导方案来优化缓冲区布局和线程绑定. 为了系统地管理缓冲区布局, Tilelang维护一个 `LayoutMap` 来记录所有缓冲区的布局信息. 并为分块算子的布局定义了一个 **分层优先级系统**, 其中更高的优先级水平表示更严格的布局要求和更大的性能影响. TileLang 以 **自顶向下** 的方式处理布局推导, 依次从最高到最低优先级水平推导布局. 在每个优先级水平, TileLang 试图为所有未确定的缓冲区推导布局, 直到无法取得更多进展, 然后再进入下一个较低的优先级水平.

![图片](assets/c00b1ebe132d.png)

如图 7 所示, 考虑一个场景, 矩阵 `C` 是 `GEMM` 操作的结果, 对应一个 `Fragment` 对象, 它需要在 `GEMM` 计算后加上偏置 `D`. 鉴于 `GEMM` 在推导过程中拥有最高优先级, 其线程绑定配置是预先确定的, 而 `D` 的线程绑定策略仍待确定. 输出矩阵 `C` 的维度是 4x4, 分布在 8 个线程上, 每个线程负责 2 个元素. 因此, 偏置缓冲区 `D` 的布局必须与此配置对齐. 由于张量 `C` 的每一行由 2 个线程处理, 这两个线程都需要访问 `D` 中相同的元素来进行加法操作. 因此, `D` 必须被复制以确保每个线程都能访问到相应的元素. `D` 的布局可以用相同的方法推导出来.

![图片](assets/edfa927fc638.jpg)

图 8 展示了线程绑定推导过程的一个示例. 具体来说, 图 8(a) 展示了一个用于拷贝数据的简单代码片段, 描述了一个子块从全局内存传输到共享内存的数据流. 适当的线程绑定和向量化访问可以充分利用 GPU 的并行性并利用高性能内存访问指令. 在图 8(b) 中, `T.copy` 操作被展开为多个循环. 在应用布局推导过程后, 如图 8(c) 所示, 程序经历了自动向量化和并行化. 最后, 在图 8(d) 所示的阶段, 应用了Layout Swizzling.

此处稍微展开分析一下这个看似简单的 `T.copy` 操作, 如何经过 TileLang 编译器的多个阶段, 逐步转换成高度优化的, 针对 GPU 硬件的并行内存访问代码.
block auto copy
图(a)描述了用户最原始的意图, 将全局内存 `A` 中的一个数据块拷贝到共享内存 `A_shared` 中. `A_shared` 的大小被声明为 `(8, 32)`. 整个 Kernel 使用了 32 个线程 (`threads=32`).在这个阶段, 用户完全不需要关心这个拷贝任务是如何在 32 个线程之间分配的, 也不需要关心每个线程具体拷贝哪个数据. 用户只描述了 "做什么" , 即 拷贝数据, 而不是"怎么做".
Desugaring
图(b)是一个去语法糖的阶段. 编译器将高级的 `T.copy(A, A_shared)` 操作 "展开"为其等价的低级形式. 这个低级形式是一个并行的循环 (`T.Parallel`).

**T.Parallel(8, 32)**: 这个原语表示一个两层的嵌套循环, 外层循环范围是 8 (`i`), 内层循环范围是 32 (`k`), 并且指示编译器将这两个循环的迭代并行化.

**数据流明确化**: 在这一步, 数据拷贝的逻辑变得更加明确: `A_shared` 中的每一个元素 `[i, k]` 都来自于全局内存 `A` 中一个相应的位置. 注意这里的 `by * BM + i` 和 `ko * BK + k` 是代表了从全局张量 `A` 中计算偏移量来定位源数据块.

**抽象层次**: 相比 (a), 这里的逻辑更具体了, 但仍然是 **逻辑并行 (Logical Parallelism)**. 代码还没有说明这个 `8x32` 的迭代空间是如何具体映射到 32 个物理线程上的. 这是一个待解决的 **线程绑定** 问题.
Layout Inference
图(c)实现了一个布局推导(Layout Inference)这是最核心的自动化调度步骤. 编译器现在需要解决如何将 (b) 中的 `8x32` 的并行任务分配给 32 个线程.编译器推导出了一个具体的 **线程绑定策略 (Thread Binding Strategy)**.

`tid = T.get_thread_env("threadIdx.x")`: 获取当前线程的 ID (范围 0-31).

任务划分: 编译器决定, 每个线程负责拷贝 `(8 * 32) / 32 = 8` 个元素.

向量化 (Vectorization): 编译器发现每个线程连续拷贝 8 个元素, 这非常适合使用向量化加载/存储指令来提升内存带宽利用率. 因此, 它引入了一个 `T.vectorized(8)` 循环.

索引计算: `A_shared` 的索引 `[tid // 4, tid % 4 * 8 + v % 8]`:

`tid // 4`: `tid` 的范围是 `0..31`, 所以 `tid // 4` 的范围是 `0..7`. 这对应了 `A_shared` 的第一个维度 (大小为 8), 也就是行索引 `i`. 这意味着每 4 个连续的线程 (`tid = 0,1,2,3` 或 `4,5,6,7` 等) 会被分配到同一行.

`tid % 4 * 8 + v % 8`: `tid % 4`: 这是一个线程在其 4 线程小组内的局部 ID (范围0..3). `tid % 4 * 8`: 这决定了每个线程负责的 8 个元素的起始列索引. 线程 0, 4, 8... 负责从列 0 开始; 线程 1, 5, 9... 负责从列 8 开始; 线程 2, 6, 10... 从列 16 开始; 线程 3, 7, 11... 从列 24 开始. `v % 8`: `v` 是向量化循环的索引, 范围 `0..7`. `v % 8` 就是 `v` 本身, 表示在 8 个元素内的偏移.

右侧的 "Data with thread binding" 图非常直观地展示了这个映射关系.

`tid` 0-3 (lane 0-3) 负责第一行 (`i=0`).

`tid` 4-7 (lane 0-3) 负责第二行 (`i=1`).

...

在第一行内, `tid=0` 负责 `k=0..7`, `tid=1` 负责 `k=8..15`, `tid=2` 负责 `k=16..23`, `tid=3` 负责 `k=24..31`.

这个阶段, 逻辑并行已经完全映射到了物理线程并行. 代码描述了每个线程 (`tid`) 在每个向量化步骤 (`v`) 中应该读写哪个内存地址.
Hardware-Specific Layout Swizzling
图(d)这是针对特定硬件的进一步优化.GPU 的共享内存被划分为多个 **banks**. 如果一个 warp (32个线程) 中的多个线程同时访问同一个 bank, 就会发生 **bank conflict (bank 冲突)**, 导致访存操作被串行化, 严重影响性能.

**Swizzling**. 这是一种内存布局变换技术, 它通过对地址进行一些位运算 (通常是异或 XOR), 来重新排列数据在共享内存中的物理存储位置, 从而打乱访问模式, 避免 bank conflict. 分解这个复杂的索引:

`tid % 4 * 8 + v % 8`: 这是 (c) 中原始的逻辑列索引, 我们称之为 `logical_k`.

`(logical_k // 8)`: 这代表 `logical_k` 所在的 8个元素的块索引.

`((tid // 4) % 8 // 2)`: `tid // 4` 是行索引 `i`. 这里是对行索引进行了一系列计算.

`^`: 异或操作. 编译器将 "块索引" 和 "行索引的一部分" 进行了异或.

`* 8 + logical_k % 8`: 将计算出的新的块索引转换回地址偏移, 并加上块内偏移.

这个看似复杂的地址计算, 其目的就是让原本在物理上连续的内存访问变得在 bank 上分散开. 例如, 原本 `tid=0` 和 `tid=4` 可能访问同一个 bank (如果 bank 数量是 4 的倍数), 经过 swizzling 后, 它们的访问目标 bank 就可能被错开了.

### 4.3 利用高性能硬件指令

现代硬件架构通常支持多种指令来实现相同的计算操作. 例如, 在 NVIDIA GPU 上, 一个 8 位乘加操作可以通过几种类型的指令实现: `IMAD` (标量), `DP4A` (向量), `MMA` (矩阵). 它们的吞吐量差异巨大.

![图片](assets/ae66d9e09d5c.png)

在 TileLang 中, 如图 9 所示, 有两种方法来调用硬件张量指令:

**C++ 源码注入** (图 9(a)): 通过 `T.import_source` 和 `T.call_extern` 手动封装和调用 C++ 模板化的指令.

**内联 PTX (`T.ptx`)** (图 9(b)): 直接在 Kernel 中嵌入 PTX 汇编指令.

然而, 根据输入形状和数据类型选择最合适的指令可能很具挑战性. 为了简化这一过程, TileLang 还支持与 **Tile Libraries** 集成, 如图 9(c) 所示. 像 NVIDIA 的 `cute` 或 AMD 的 `composable kernel (ck)` 这样的分块库, 提供了高级, 标准化的基于分块的 API (例如 `tl::gemm_ss`). 这些库抽象了硬件细节, 并允许底层实现自动选择给定输入配置的最有效指令. 在 TileLang 中, 开发者可以使用 `T.call_extern` 以一种直接且一致的方式调用这些库.

总而言之, TileLang 提供了两种互补的方法来利用高性能指令. 第一种利用Tile-Lib, 简化了集成并受益于厂商优化的性能. 但高级抽象可能限制底层控制. 此外, 由于大量使用模板, 编译可能会变得非常慢. 第二种方法是直接在 TileLang 内部通过 `T.gemm` 等算子实现指令逻辑. 这避免了布局注释限制并减少了编译时间, 但要求用户为每个目标硬件指令在 TileLang 中实现一个完整的指令集. 目前, TileLang 支持这两种方法, 默认使用基于分块库的方法, 以便快速支持新的硬件指令.

### 4.4 软件定义流水线

TileLang 采用一种自动化的软件流水线推导机制来分析计算块之间的依赖关系 (例如, `Copy` 和 `GEMM`), 并生成一个结构化的流水线调度, 以在保持正确执行顺序的同时最大化并行度. 具体来说, 该机制将 `Copy` 任务与其他计算密集型操作交错执行以减少空闲时间, 当检测到异步处理的机会时, 它会自动将这些任务映射到可用的硬件资源上进行并发执行. 因此, TileLang 只需向用户暴露一个单一的 `num_stages` 接口, 极大地简化了流程. 然而, 如果需要, 我们也允许用户显式地提供关于顺序和阶段的信息.

![图片](assets/699ba0218eac.png)

对于 **Ampere** 架构, TileLang 支持使用 `cp.async` 的异步内存拷贝操作. TileLang 通过分析循环结构并自动为符合条件的内存传输插入 `cp.async` 指令来整合此功能.

在 **Hopper** 架构中, 引入了两个新特性: 一个新的 TMA 单元, 专用于全局内存和共享内存之间的数据拷贝; PTX 指令集引入了新的 `wgmma` 指令, 它使一个 **warpgroup** (由四个 warp 组成) 能够执行矩阵乘法 (MMA) 操作, 以提高 TensorCore 利用率. 此外, `wgmma.mma_async` 指令是异步的. Hopper 架构的 Kernel 优化通常采用 **Warp Specialization**, 其中线程被分为生产者和消费者. 生产者线程使用 TMA 移动数据, 而消费者线程负责计算. 在 TileLang 中, 降级过程中自动执行 Warp Specialization 优化. 具体来说, TileLang 分析所有语句的缓冲区使用情况, 并确定它们的角色 (生产者或消费者), 然后根据 `threadIdx` 将它们划分到不同的执行路径中.

**AMD CDNA** 架构中也提供了异步拷贝指令和 DMA 支持, TileLang 通过 HIP 封装的 `Copy` 原语来利用这些支持.

## 5. 数值实验

### 5.1 测试环境

**硬件平台**: 实验使用了三款新的 GPU:

NVIDIA H100 (80 GB)

NVIDIA A100 (80 GB)

AMD Instinct MI300X (192 GB)

对于 NVIDIA H100, 使用 CUDA 12.4; 对于 MI300X, 使用 ROCm 6.1.0. 所有平台都在 Ubuntu 20.04 下运行.

**算子工作负载**: 在一系列经常出现在大规模深度学习流水线中的算子工作负载上评估 TileLang.

在 **NVIDIA H100** 上, 专注于多头注意力 (MHA), 线性注意力 (Linear Attention), 和通用矩阵乘法 (GEMM).

在 **NVIDIA A100** 上, 测量了Dequantized GEMM Kernel 的性能.

在 **AMD Instinct MI300X** 上, 对 GEMM 和 MHA 进行了基准测试, 以捕捉跨越不同 GPU 架构的代表性用例.

**基准 (Baselines)**: 为了评估 TileLang 的性能, 将其与几个在机器学习和 GPU 编程中广泛使用的最先进基准进行比较.

**FlashAttention-3**: 针对多头注意力进行优化, 使用了如 `tma` 和 `wgmma.mma_async` 等 CUDA 指令.

**Triton**: 一个用于高效 GPU Kernel 的开源框架, 支持 Nvidia 和 AMD GPU, 但需要手动优化.

**cuBLAS/rocBLAS**: NVIDIA 和 AMD 的高性能稠密线性代数库.

**PyTorch**: 具有手写的优化 Kernel (如 GEMM 和 FlashAttention-2), 但并非完全优化.

**BitsandBytes**: 专为支持如  等格式而设计, 并提供高效 Kernel.

**Marlin**: 针对  计算的高度优化 Kernel.

### 5.2 实验

![图片](assets/47426eef8a8f.png)

**Flash Attention 性能 (图 12)**: 与 FlashAttention-3, Triton, 和 PyTorch 相比, TileLang 分别取得了 *1.36×*, *1.41×*, 和 *1.70×* 的加速. 因为 FlashAttention-3 是一种手工打造的方法, 它不能有效地适应变化的工作负载大小. 特别是, 其固定的分块大小导致在较小序列长度上性能不佳. 对于较长的序列长度 (例如 8k), TileLang 的性能仍然接近 FlashAttention-3. PyTorch 使用了手写的 FlashAttention-2 Kernel, 导致性能低于 FlashAttention-3. 与这些基于手动模板的实现相比, TileLang 可以自动利用如 `cp.async.bulk` 和 `wgmma.mma_async` 等指令, 并且自动应用如 Warp Specialization 等优化. 值得注意的是, 在 H100 GPU 上, TileLang 能够表达出与 FlashAttention-3 使用的一样复杂的流水线调度方案.

**线性注意力性能 (图 12)**: 在线性注意力实验中, 使用了 Mamba-2 的 `chunk-scan` 和 `chunk-state` 函数. 与 Triton 相比, TileLang 平均取得了 *1.77×* 和 *2.10×* 的加速.

**MLA性能 (图 14)**: 图 14 展示了 MLA 的性能和相应 Kernel 实现的代码行数 (LOC).

![图片](assets/a6bdc7c66c6f.jpg)

在 **H100** 上, TileLang 相比 Torch 取得了 **1075.9×** 的加速, 显著优于 Triton 和 FlashInfer, 并达到了手写优化的 FlashMLA 实现性能的 **98%**. 此外, TileLang 仅需要约 **70 行** Python 代码, 展示了比其他基准好得多的易用性.

在 **MI300X** 上, TileLang 相比 Torch 取得了 **129.2×** 的加速, 并在性能和代码紧凑性上都超过了 Triton. 与手写的库 AITER 相比, TileLang 达到了其性能的 **95%**.

**矩阵乘法 (Matmul) 性能 (图 13)**: 图 13 展示了在 NVIDIA 和 AMD GPU 上的 GEMM 工作负载性能.

![图片](assets/50ba8369473b.png)

在 RTX 4090, A100, H100, 和 MI300X 上, TileLang 相较于厂商优化的库 (cuBLAS/rocBLAS) 分别取得了 *1.10×, 0.97×, 1.00×, 1.04×* 的加速比.

与 Triton 相比, TileLang 在相同 GPU 上分别交付了 *1.08×, 1.03×, 1.13×, 1.25×* 的加速.

对于矩阵乘法, TileLang 使用简单的语法就达到了与厂商优化库相当的性能.

**Dequantize Matmul性能 (图 15)**: BitBLAS 是一个用于混合精度计算的高性能库. 将其后端替换为 TileLang.

![图片](assets/e6ca9fc9ed35.png)

## 6. 如何使用最低成本学习B卡上Tilelang的开发

当然合法的境内渠道要使用B200/B300这些Blackwell架构的卡还是... 但不妨碍我们利用Jetson Thor这些车载芯片来做一些关于B卡的实验并了解Blackwell SM的架构. tilelang==0.1.6.post1并没有支持Blackwell, 因此可以参考官方文档《Tilelang Installation Guide》[3] 安装.

但是需要注意的是, Thor 在CUDA 12.9中叫sm_101, 因为一些不太好的事情, 在CUDA 13.0开始改为了sm_110, 从源码编译安装的方法如下

```
cd /optgit clone --recursive https://github.com/tile-ai/tilelangcd tilelang
```

为Thor做一点小修改

```
zartbot@zartbot-thor:/opt/tilelang$ git diffdiff --git a/src/target/utils.cc b/src/target/utils.ccindex 06ff20f..ca4f857 100644--- a/src/target/utils.cc+++ b/src/target/utils.cc@@ -57,7 +57,7 @@ bool TargetIsSm100(Target target) {   if (!TargetIsCuda(target))     return false;   int arch = GetArchInt(target);-  return arch >= 100 & arch <= 103;+  return arch >= 100 & arch <= 110; } bool TargetIsSM120(Target target) {
```

然后编译

```
mkdir buildcp 3rdparty/tvm/cmake/config.cmake buildcd build# echo "set(USE_LLVM ON)"  # set USE_LLVM to ON if using LLVMecho "set(USE_CUDA ON)" >> config.cmake # or echo "set(USE_ROCM ON)" >> config.cmake to enable ROCm runtimecmake..make -j 10
```

最后在~/.bashrc中添加

```
export PYTHONPATH=/opt/tilelang/:$PYTHONPATH
```

验证环境

```
zartbot@zartbot-thor:/opt/tilelang$ source ~/.bashrczartbot@zartbot-thor:/opt/tilelang$ python -c "import tilelang; print(tilelang.__version__)"0.1.6.post1+a35ac496
```

最后做一个简单的测试

```
import tilelangimport tilelang.language as Tdef matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):    @T.prim_func    def main(        A: T.Buffer((M, K), dtype),        B: T.Buffer((K, N), dtype),        C: T.Buffer((M, N), dtype),    ):        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):            A_shared = T.alloc_shared((block_M, block_K), dtype)            B_shared = T.alloc_shared((block_K, block_N), dtype)            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)            T.clear(C_local)            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):                T.copy(A[by * block_M, ko * block_K], A_shared)                for k, j in T.Parallel(block_K, block_N):                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]                T.gemm(A_shared, B_shared, C_local)            T.copy(C_local, C[by * block_M, bx * block_N])    return mainfunc = matmul(1024, 1024, 1024, 128, 128, 32)jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")import torcha = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)c = jit_kernel(a, b)ref_c = a @ btorch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)print("Kernel output matches PyTorch reference.")cuda_source = jit_kernel.get_kernel_source()print("Generated CUDA kernel:\n", cuda_source)profiler = jit_kernel.get_profiler()latency = profiler.do_bench()print(f"Latency: {latency} ms")
```

测试输出:

```
zartbot@zartbot-thor:~$ python3 foo.pyKernel output matches PyTorch reference.Generated CUDA kernel:#include <tl_templates/cuda/gemm.h>#include <tl_templates/cuda/copy.h>#include <tl_templates/cuda/reduce.h>#include <tl_templates/cuda/ldsm.h>#include <tl_templates/cuda/threadblock_swizzle.h>#include <tl_templates/cuda/debug.h>#ifdef ENABLE_BF16#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>#endifextern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, half_t* __restrict__ B, half_t* __restrict__ C);extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, half_t* __restrict__ B, half_t* __restrict__ C) {  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];  float C_local[128];  __shared__ uint64_t mbarrier_mem[6];  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);if (tl::tl_shuffle_elect<0>()) {    tl::prefetch_tma_descriptor(A_desc);    mbarrier[0].init(128);    mbarrier[1].init(128);    mbarrier[2].init(128);    mbarrier[3].init(128);    mbarrier[4].init(128);    mbarrier[5].init(128);  }  __syncthreads();if (128 <= ((int)threadIdx.x)) {    tl::warpgroup_reg_dealloc<24>();    for (int ko = 0; ko < 32; ++ko) {      mbarrier[((ko % 3) + 3)].wait((((ko % 6) / 3) ^ 1));      if (tl::tl_shuffle_elect<128>()) {        mbarrier[(ko % 3)].expect_transaction(8192);        tl::tma_load(A_desc, mbarrier[(ko % 3)], (&(((half_t*)buf_dyn_shmem)[((ko % 3) * 4096)])), (ko * 32), (((int)blockIdx.y) * 128));      }      #pragma unroll      for (int i = 0; i < 2; ++i) {        for (int vec = 0; vec < 2; ++vec) {          *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((ko % 3) * 4096) + (((((int)threadIdx.x) & 7) >> 2) * 2048)) + (i * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + vec) & 1) * 8)) + 11264)) = *(uint4*)(B + (((((((ko * 32768) + (i * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (vec * 8)) - 16384));        }      }      tl::fence_proxy_async();      tl::mbarrier_cp_async_arrive(mbarrier[(ko % 3)]);      mbarrier[(ko % 3)].arrive();    }  } else {    tl::warpgroup_reg_alloc<240>();    #pragma unroll    for (int i_1 = 0; i_1 < 64; ++i_1) {      *(float2*)(C_local + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);    }    tl::fence_proxy_async();    for (int ko_1 = 0; ko_1 < 32; ++ko_1) {      mbarrier[(ko_1 % 3)].wait(((ko_1 % 6) / 3));      tl::gemm_ss<128, 128, 32, 2, 2, 0, 0, 0, 32, 128, 0, 0>((&(((half_t*)buf_dyn_shmem)[((ko_1 % 3) * 4096)])), (&(((half_t*)buf_dyn_shmem)[(((ko_1 % 3) * 4096) + 12288)])), (&(C_local[0])));      mbarrier[((ko_1 % 3) + 3)].arrive();    }    #pragma unroll    for (int i_2 = 0; i_2 < 64; ++i_2) {      uint1 __1;      float2 v_ = *(float2*)(C_local + (i_2 * 2));      ((half2*)(&(__1.x)))->x = (half_t)(v_.x);      ((half2*)(&(__1.x)))->y = (half_t)(v_.y);      *(uint1*)(C + (((((((((((int)blockIdx.y) * 131072) + (((i_2 & 7) >> 1) * 32768)) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + ((i_2 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((i_2 >> 3) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;    }  }}#define ERROR_BUF_SIZE 1024static char error_buf[ERROR_BUF_SIZE];extern "C" const char* get_last_error() {    return error_buf;}extern "C" int init() {    error_buf[0] = '\0';    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);    if (result_main_kernel != CUDA_SUCCESS) {        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 49152, cudaGetErrorString(result_main_kernel));        return-1;    }    return0;}extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) { CUtensorMap A_desc; CUtensorMapDataType A_desc_type= (CUtensorMapDataType)6; cuuint32_t A_desc_tensorRank= 2; void *A_desc_globalAddress= A; cuuint64_t A_desc_globalDim[2]= {1024,1024}; cuuint64_t A_desc_globalStride[2]= {2,2048}; cuuint32_t A_desc_boxDim[2]= {32,128}; cuuint32_t A_desc_elementStrides[2]= {1,1}; CUtensorMapInterleave A_desc_interleave= (CUtensorMapInterleave)0; CUtensorMapSwizzle A_desc_swizzle= (CUtensorMapSwizzle)2; CUtensorMapL2promotion A_desc_l2Promotion= (CUtensorMapL2promotion)2; CUtensorMapFloatOOBfill A_desc_oobFill= (CUtensorMapFloatOOBfill)0; CUresult A_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(    &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);if (A_desc_result != CUDA_SUCCESS) {  std::stringstream ss;  ss << "Error: Failed to initialize the TMA descriptor A_desc";  snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());return-1; } main_kernel<<<dim3(8, 8, 1), dim3(256, 1, 1), 49152, stream>>>(A_desc, B, C); TILELANG_CHECK_LAST_ERROR("main_kernel");return0;}Latency: 0.1454080045223236 ms
```

参考资料

[1] 
TileLang: 80行Python kernel代码实现FlashMLA 95%的性能: *https://zhuanlan.zhihu.com/p/27965825936*
[2] 
TileLang: A Composable Tiled Programming Model for AI Systems: *https://arxiv.org/abs/2504.17577*
[3] 
Tilelang Installation Guide: *https://tilelang.com/get_started/Installation.html*