# CUDA-Next: 基于任务的张量计算的DSL?

> 作者: zartbot  
> 日期: 2025年4月26日 06:57  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494047&idx=1&sn=0fe4f9dbfc473692c733145385740c33&chksm=f995f95dcee2704b739a0b7b25635ba42edbc3e2b712530d3c069499b622cd91c63c7ab5340b#rd

---

### TL;DR

最近在学习TileLink以及ScaleUP/ScaleOut总线的协同设计, 于是就开始关注一些编译器相关的工作.  前几天爬Nvidia Research的网页, 发现一个很有趣的工作《Task-Based Tensor Computations on Modern GPUs》[1]. 这篇文章中了编程语言的顶会PLDI‘25. 其实最近一段时间都在思考一个问题, 一方面是从微架构而言, GPU内的DSA(Tensor Core/TMA/RayTracing)这些DSA越来越多, 似乎又进入到一个牧本周期... 周期性的变化带来的编程方式的挑战是什么? 大量的异步化编程肯定是一个难题. 另一方面是随着算力的需求, ScaleUP/ScaleOut带来的分布式计算, 对于在分布式系统上的编程语言适配的问题, 特别是分布式系统一致性的问题. 两者似乎可以协同做一些设计...

这篇文章提出了一种基于Tensor任务的名为Cypress的DSL, 有点意思... 虽然编译器底层还是产生CUDA代码, 但是值得我们关注的是, 既然DSL的出现并在牧本周期下DSA逐渐成为主流, 国产加速器bypass CUDA生态垄断的机会一定要抓住, 有可能又会像当年OpenGL那样出现一个百花齐放的年代.

### 1.为什么需要新的DSL

伴随着Volta引入Tensor Core,并且引入了每个线程独立的PC和cooperative_groups, 实际上就已经开始在破坏CUDA SIMT的bulk-sync结构了, 然后Ampere引入async.cp解决一些L1/RF的压力. 进一步在Hopper上引入TMA以及临时拼凑出来的一个WGMMA, 并实现了warp specialization. SIMT的bulk-sync的编程机制逐渐转向了Producer-Consumer的结构. 再到Blackwell为了进一步降低RF的负担引入了TMEM. 详细的NV-GPU几十年的发展可以看看下面这个专题

[《GPU架构演化史》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=2538479717163761664&scene=173&from_msgid=2247487954&from_itemidx=3&count=3&nolastread=1#wechat_redirect)

很有意思的一个话题是GPU作为一个异构加速器本身也开始变得更加异构了...而这些异构器件伴随着深度学习所采用的张量规模扩大, 异步访问提高整个系统吞吐的需求越来越高. 详细可以参考

[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect)

异步编程和DSA带来的复杂性也逐渐衍生出一些编程库, 例如CUTLASS/ThunderKitten, 但是需要程序员去管理通信和同步机制,cutlass有一些介绍在下面这个专题

[《Tensor》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3557619493198151684&scene=173&subscene=&sessionid=svr_32119fe6ccb&enterid=1722676230&from_msgid=2247491424&from_itemidx=1&count=3&nolastread=1#wechat_redirect)

手动管理数据移动和同步对于专家程序员来说也容易出错,可能导致Data Race, 或者一些细节上需要详细设计Overlap流程,否则会在性能上有所损失.

另一条路是依赖于Compiler去推测所有的细节, 例如Triton或者Triton-Distributed(TILELINK), 虽然这些方式能够通过一些自动化的启发式方式来减轻程序员的一些负担, 但是如何将计算和数据映射到物理机器上, 也是一个难题, 并且比起一些专家所写的Kernel也会产生一定的性能折扣.

论文中有一个例子, 可以看到伴随着Ampere/Hopper的新的DSA引入, 异步编程变得更加复杂,

![图片](assets/a8cf61f0b288.png)

而到了Blackwell还需要显式的去管理Tensor Memory, 基本上每一代都要重写Kernel.

因此, 工业界期望在纯手撸算子让程序员管理数据移动和同步的复杂性, 以及在纯自动的靠编译器启发的两种方式上取得某种平衡, 避免程序员感知复杂的数据移动和同步, 同时又有对计算如何分解如何映射到硬件上这些对性能有关键影响的地方获得足够的控制权. 这就是Cypress的初衷.

### 2.Cypress Overview

Cypress依赖于TensorCore/TMA这些异步的固定功能的DSA进行张量代数计算, 然后进行了两层抽象, 以Tensor作为DSL的一等公民, Cypress定义了一个顺序编程的DSL来包住TensorCore/TMA这些DSA异步调用的复杂性, 同时提供了一个compiler来生成GPU执行的代码, 并维持接近峰值的性能. 一个现代的GPU内存层次结构如下:

![图片](assets/959f78d0c216.png)

然后通过task定义一系列顺序编程的算子函数, 并对计算任务进行了如下抽象

Logical Description: 定义对计算的算子拆分和相应的计算可调优的一些参数.

Mapping Specification: 如何映射到计算单元,特别是内存层次结构上.

其实这样的拆分会有很大的好处, 计算本身的逻辑和机器的内存层次结构的分离, 使得未来新的DSA硬件, 或者进一步支持ScaleUP/ScaleOut的Distributed-GEMM计算时的算子适配成本会变得更小. 顺便补充一点, 在设计ScaleUP和ScaleOut的互联时, 硬件设计上也需要更好的让软件和编译器能够更好的处理, 因此内存层次抽象上还有很多工作要去做.

### 3.编程模型

#### 3.1 Logical Description

计算逻辑描述上定义了一个层次化的Task函数结构, 它定义了一些tunable的变量, 用于在不同硬件上构建调优的参数, 针对计算任务如何划分,  参数的读写属性以及处理的scope都进行了定义.

![图片](assets/370d09f65b82.png)

同时它定义了一个Task Variant Kind, 支持inner和leaf两种类型, inner task表示在其中可以支持一些有限的标量计算时, 还可以进一步调用sub-task. 而leaf没有sub-task, 主要是利用CUDA Core访问Tensor去做一些其它的运算. 然后针对Inner task, 它还定义了两种range调用sub-task的策略, srange为顺序执行, prange为并行执行.

对于张量拆分也定义了partition类型, 例如针对MMA的swizzle等..拆分后的一些新的子矩阵坐标映射一类的可以参考cutlass layout代数相关的内容.

[《Tensor-008 CuTe Layout代数》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492220&idx=1&sn=4ec36b34df55ae6c0b643709da3316e1&scene=21#wechat_redirect)

其实本质上是一个可组合算子的抽象过程, 那么当我们在构造一个Composable Disaggregation的分布式计算架构的同时, 也需要满足软硬件交互界面的可组合性. 例如最近大火的MCP,本质上是在更高的LLM层上定义的一个Monad.

[《Tensor-006 AI软硬件交互界面: 可组合的Kernel》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491708&idx=1&sn=1fd03181e44f573f6ec1d90d66d93a24&scene=21#wechat_redirect)

其实这也是我在很多年前设计NetDAM的时候强调的, 虽然支持内存Semi-Lattice语义有很大的代价, 但是访问内存的时候支持交换律能够out-of-order执行可以很大的提高并发能力, 另一方就是访问内存满足结合律使得编程上变得更加容易构建Composable的算子结构, 而幂等则是进一步提高整个系统的可靠性,在出现丢包和链路失效的场景下, replay 保证不会产生副作用.

#### 3.2 Mapping Specification

Cypress的另一个抽象组件就是Mapping Specification, 它规定了如何将程序绑定到特定机器的描述上, MappingSpec允许对性能敏感的决策进行控制, 而Cypress的编译器确保这些Mapping不会影响程序的正确性. MappingSpec通过确定在每个处理器Scope上采用哪一种Task, 以及每一个Task的内存存放位置来静态的实例化一个任务树. 每个实例都有一个名称, 并且通过名称来引用另一个实例

![图片](assets/a693fc12c0ec.png)

如图所示, 一个Instance申明了多个属性:

Instance在执行时所使用的Task Variant

Task Variant应该在哪个处理器(Scope)上执行

Task的每个张量参数应放置在哪个内存中, 并且每个张量可以放置在不同的内存中

绑定到task variant中可调参数(Tunable变量)的具体值

每个启动的Sub-Task应当调度到哪个Task映射实例上

MappingSpec还可以控制特定处理器相关的行为, 例如对某个特定的Task进行Warp-specialization或者pipeline执行时进行一些额外的控制以及对张量layout的一些控制(例如Swizzle).

另一方面, 程序员通常由于内存层次化结构, 在容量和性能的约束下, 期望某些Tensor在特定的区域无需完整的实例化,而是与Task-Tree中更下层的Task Instance共享, 因此在Mapping的时候可以使用None Memory属性, 例如在GEMM中的Accumulator, 通过更下层的task进行分区实例化, Compiler在对None Memory属性的张量在编译时需要对相应的约束有一定的检查机制.

#### 3.3 HopperGEMM的例子

如前图的一个例子, 它是一个基于Python的DSL, 该程序描述了GEMM计算任务的分解以及如何层次化的进行分块处理. 计算任务的入口为gemm_host, 在MappingSpec中也定义proc=Host在Host scope执行, Host GEMM利用U V两个可调整参数来拆分矩阵, 并定义了每个子Block, 然后通过调用prange并行化的拉起sub-task.

然后在gemm_block中添加了warp-specialization和Pipeline的编排.  然后接着进一步对矩阵分割构建gemm_warpgroup和gemm_warp/gemm_thread任务. 并且针对RF进行的一些拆分和约束. 例如warp group的pieces映射到一个WarpGroup中调用4个SM, 每个SM在gemm_warp中调用4个thread. 最后在Leaf task中, Cypress不会对它的内容进行分析和优化, 而是保证CUDA代码在线程级的灵活性.

#### 3.4 一些讨论

Logical Description和MappingSpec的分离, 使得程序员可以在对应用代码不做侵入修改的情况下, 对性能敏感的一些算子编排基于任务形式进行调整, 并通过MappingSpec抽象, 同时隐式的执行TMA等内存拷贝.

Cypress处于一个折中的位置, 它不像cutlass那样要求程序员手动管理算法和在硬件上的映射, 也不像Triton那样将很多对性能敏感的决策隐藏起来. Cypress 为程序员提供了对算法和映射决策的低级控制,同时自动化实现了策略,既保证了性能,也保证了正确性.

Cypress编程模型在源算法层面将数据分区以及它们的使用方法明确的表达出来, 然后通过编译器来保证程序执行的顺序语义和并发, 相对于直接底层编程的实现, 能够更好的保证程序执行的正确性.

### 4. 编译

Cypress的整个编译器架构和基于Event的IR如下所示:

![图片](assets/56fc07a0e954.png)

#### 4.1 IR

上图中展示了Cypress的IR的简化的语法, 该语法包含了Tensor之间的显式数据复制和任务调度的操作以及采用顺序和并行的for循环(prange/srange)操作对应的(for/pfor). IR中潜在的异步操作都会产生一个event, block拥有自己完整的事件机制.

Cypress IR 中最有趣的部分是事件的表示. 事件类型可以是单个单位,也可以是事件数组,其中数组中的每个维度都标注有特定的处理器类型. 事件数组由并行循环创建,每个数组元素对应于在特定处理器上一次迭代的完成事件. 可以通过索引事件数组来提取未来操作应该依赖的特定事件. 事件索引可以使用整数值,也可以使用广播操作符, 采用广播操作符很有意思, 代表索引在某个维度上所有事件的完成标志, 对被索引的处理器按照维度进行同步.

#### 4.2 编译器架构

Cypress通过Logical Description和MappingSpec描述表示程序, 并通过编译器产生CUDA代码, 编译器通过对IR的一系列传递来完成这一个转换. 通过依赖分析/向量化/复制消除捕获Task-Based表示的程序中的重要信息, 并将任务抽象层次向下转化, 接下来通过资源分配/Warp-Specialization执行优化, 最终降级为CUDA C++代码,并用系统特定的同步机制来替代事件, 并构造执行有效的CUDA程序所需要的其它转换.
4.2.1 依赖分析
它基于Task的Logical Description和MappingSpec描述进行语法转换, 将其转换为Cypress IR. 然后基于IR上的Event进行以来分析, 对于操作数据的权限, 例如Read-Only或者没有依赖来进行并行处理. 此外对于单个逻辑张量的分块并且由不同Task映射到不同内存时, 保证数据移动的一致性依赖.

依赖分析通过对实例化的Task Tree进行便利, 并从MappingSpec中的Task Variant入口开始遍历, 对于Task中的每个张量都维护一个event, 每遇到一个Task的启动点时, 利用被启动Task的数据权限, 将相应的event注册成为该Task的前置条件来保证顺序, 并通过任务完成时反馈更新相应的上级event完成事件.

例如,如果一个任务写入一个张量,那么后续读取该张量的任务会记录一个依赖于写入任务完成的事件｡依赖关系通过在 IR 中链式连接事件,在任务调用之间得以强制执行｡为了降低一个任务调用,编译器会查阅MappingSpec,从而确定被调用的Task Variant及每个张量参数应放置在哪个内存中｡对于被调用任务的所有张量参数,依赖分析采用了“Copyinput/CopyOut”的规范｡将一个任务调用降级到 Cypress IR 包含四个步骤:

为被调用Task的每个张量参数在映射指定的内存中创建一个新的分配

对于被调用Task所读取的每个张量参数,生成从现有张量分配向新分配的复制,并记录该复制的所有事件前置条件

将所有复制完成的事件记录为被调用任务的前置条件,然后递归遍历被调用任务所选定的任务变体以生成其 IR

对于被调用Task写入的每个张量参数,生成从新分配向调用者任务的现有分配的复制,并以被调用任务的完成事件作为前置条件

当一组任务并行执行时, 采用广播索引操作符来使后续操作以所有并行迭代完成为前提,  并采用CopyInput/CopyOut规范确保依赖分析在单个任务变体内, 虽然这样的方式引入了一些不必要的复制, 但简化了编译器的复杂性,并最终通过后续的Copy Elimination的方式保证这些不必要的复制被消除.
4.2.2 向量化
接下来,Cypress 执行一个向量化过程,将依赖分析生成的程序中的嵌套循环结构扁平化. 此阶段会移除 GPU 编程模型中隐含的嵌套循环,例如针对 warpgroups､warps 和线程的 pfor 循环. 向量化过程利用 IR 中可索引的事件数组, 在展开并行循环之后仍然保留迭代之间的依赖关系.

![图片](assets/4afb593c9bb2.png)

向量化的机制非常直接:从最深层的嵌套开始,将每个隐含的并行循环展开扁平化,并将迭代变量替换为一个能计算出处理器索引(例如 warp 或线程索引)的表达式. 在扁平化的隐含循环内创建的所有事件数组都将通过添加一个大小等于该循环范围的新维度而提升. 随后, 隐含循环内所有事件的使用者都将被重写,以用处理器索引对每个事件数组进行索引.  这样,展开打平的循环的独立迭代之间的点对点依赖关系得以保留,而在复制和依赖任务之前所需的同步则通过对事件的广播索引得以编码.
4.2.3 拷贝消除
第一阶段为了简化数据以来分析的复杂性, 引入了copyinput/copyoutput的机制, 然后在这个阶段对这些数据拷贝进行分析消除.

![图片](assets/6bd4c05b3dec.png)

`Spill Elimination`: 一个张量复制到其父张量的一个切片中, 然后再将该父张量切片复制回原张量, 这种方式可以消除拷贝

`Spill Hoisting`: 识别到一个循环内, 先从父张量复制到一个子张量, 然后再从子张量复制回父张量的情况, 这些复制操作可以被提升到循环的前导和后导部分.

然后基于这些拷贝消除后,对于数据依赖的事件还进行了一些处理, 确保依赖中的数据都已经完全被准备好.
4.2.4 资源分配
在拷贝消除阶段避免了复制或者中间张量之后, 剩余的张量必须要映射到相应物理内存上, 由于每个SM内部的SMEM受片上资源约束, 在内存资源和并行性之间进行trade-off.Cypress针对异步操作的环境需要构建一些特殊的策略, 对于一个逻辑张量由于异步操作, 需要决定在不同的时间复用同一块物理内存空间.

![图片](assets/c5291bf0ed99.png)

Cypress首先构造出所有共享内存的Tensor相互之间的影响的图, 然后添加辅助的边将图补全, 从而迫使所有的张量进入独立分配的方式. 然后迭代的构造一个在用户提供的内存上限内能够满足要求的方案, 通过不断的移除辅助的边, 直到完成.

由于时分复用同一个内存, 那么在编译器上必须额外的加入一些事件依赖, 以确保被分配到同一物理内存的逻辑张量的活跃周期不会重叠. 需要注意的是, 由于这些张量将被分配到同一内存上,Cypress 会在使用该物理内存的相邻逻辑张量之间, 即在前一个张量的最后读取者与下一个张量的第一个写入者之间插入事件依赖边, 从而避免写后读的风险.
4.2.5 Warp Specialization
它将计算划分到线程块的多个warp中,从而暴露出warp之间的并发性, 并允许将资源分布在多个warp之间. Cypress会将Task划分为多个计算warp和一个数据移动warp, 防止在不同的DSA之间(TC/TMA)交互时的干扰, 并使得计算warp能够获得更多的寄存器文件. Cypress将其视为一种图划分的算法, 对于IR中的依赖图必须在计算warp和数据移动warp之间进行划分并插入barrier

![图片](assets/d846b5eeff97.png)
4.2.6 CUDA C++代码生成
代码生成中其实就是基于前述图中将独立的计算任务分别构建相应的__device__ function 然后启动kernel. 很关键的一个任务是一些特定的依赖/异步事件的同步机制的实现.

### 5.评估

例如针对GEMM的处理, 特别是GEMM+Reduction的处理上, 可以看到Cypress的一些优势

![图片](assets/e72fdda640a6.png)

然后还有一个FlashAttn的评估

![图片](assets/729cfaee7643.png)

### 6. 相关的工作

主要分几块, 例如Sequoia基于任务的并行编程, 对于内存层次结构的限制, 这种模型无法描述现代 GPU 中多个层级的处理器能够访问多个内存的情况. 然后Cutlass这类的模版库, 用户需要自己处理显式通信/同步/warp specialization的细节. Cypress复用了Cutlass中的CuTe以及layout代数. 另一块是基于Tile的抽象, 例如Triton, 随着底层架构变得越来越复杂以及程序多样化, 完全依赖编译器来决定如何降级到线程块的程序表达会产生一些性能问题. 还有一些基于函数式编程和基于调度的DSL编译器...

### 7. 关于未来软硬件协同设计的想法

读完论文后, 我也在考虑一个问题, 例如Cypress over ScaleUP/ScaleOut的设计, 相应的MappingSpec怎么弄, 各种TP/SP/EP/DP的表达如何弄, 类似的Triton-Distributed(TileLink)的工作, 类似于DeepEP这样的Buffer的一层抽象如何和它们结合. 对于处理器在ScaleUP和ScaleOut上内存层次化结构的处理, 以及在更长远一点在推理系统中的KVCache等...

![图片](assets/bf6574760b16.png)

例如前几天讨论的腾讯的DeepEP优化下, 关于一些Atomic和Memory Fence在ScaleUP和ScaleOut协议上的系统设计... 很有趣的一个话题...

更广泛一点, 例如NSA或者Google Titan或者MOBA这类的新的Attn block对于内存访问的一些更多的需求, 该如何进一步去处理? 这两天读到了另一篇Google Research的论文, 还要进一步的去分析

![图片](assets/5bd0fa6d9295.png)

参考资料

[1] 
Task-Based Tensor Computations on Modern GPUs: *https://research.nvidia.com/publication/2025-06_task-based-tensor-computations-modern-gpus*