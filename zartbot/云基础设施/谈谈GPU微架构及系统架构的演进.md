# 谈谈GPU微架构及系统架构的演进

> 作者: zartbot  
> 日期: 2025年11月11日 00:17  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496776&idx=1&sn=3857e2d6c67c17d93b149729fb2d7cc4&chksm=f995e48acee26d9c1aa53505deab55149dfe1633f1076a91fb5205937935d40d04896e458f09#rd

---

### TL;DR

前面一篇文章[《Inside Nvidia GPU: 谈谈Blackwell的不足并预测一下Rubin的微架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21#wechat_redirect)谈到一些GPU微架构的演进, 其中的一些预测引起了一些讨论, 这一篇进一步对如下几个方面的问题进行更深入的探讨:

**算力及调度**: 当算力越来越强后, 很多Kernel的运行时间小于100us时, kernel launch的开销, 以及对于复杂计算任务的流水线编排调度成为一个需要解决的难题.

**内存访问** : 当前Nvidia针对内存容量不足的问题, 一方面是使用多Die拼接Scale-in的方式扩展带宽和容量, 另一方面是通过CPU-GPU unified Memory的方式扩展. 同时伴随着算法上Long Context的需求, Sparse/Linear Attn对于KVCache的访问还需要一些更特殊的硬件优化.

**互连架构** : 这也是一个充满争议的领域, 例如ScaleUP的规模需要多大, 延迟有什么约束. ScaleOut需要如何改进, 整个系统的可靠性如何取舍等.

## 1. 基于算法和应用的视角

没有实际的workload需求来分析微架构和系统架构属于空谈, 因此先来从应用的视角谈谈需求. 其实最近关于Attention block的算法路线争议也给未来的架构带来了很多变数, 并且伴随着MoE带来的大规模EP并行也给系统架构带来了很多挑战, 通常设计一块芯片需要一个架构师去预测未来3~5年的主流架构, 在算法快速变化的今天这使得架构师的压力会更大.

### 1.1 Overview

对于当前的基于Transformer架构的模型, 我一直有这样一个观点: **对于AGI的实质是, 我们需要通过一个模型构造一个自己能够产生代码运行的通用计算机架构, 即token as instruction.** 那么在这种观点下我们将模型的各个组件映射到一个通用的计算机体系结构可以得到如下一个图:

![图片](assets/5fc1866345c8.png)

在Google那篇《Titans: Learning to Memorize at Test Time》[1]论文的开头引用了一句话:

![图片](assets/46277ea5dcd0.png)

这句话是 18 世纪英国作家兼词典编纂家塞缪尔·约翰逊所说, 它的核心含义是, 有效的记忆力并非一种神秘的先天技能, 而是专注努力和正念的直接结果. 你不可能记住你没有首先注意的东西. 而换到今天, 似乎也有了类似的理解, 关于大模型的Attention的艺术, 实际上就是一个关于内存的艺术.

从计算机体系结构的视角来观察算法的演进是一个比较独特的视角. 对于MoE的演进从这个视角来看可以类比为很简单的内存分页的问题, 至于Fine-Grained MoE和最开始的MoE实际上有点类似于页表大小的区别了.这个话题我们稍后展开.

另一方面我们来看Attention block, 实际上它可以类比为一个控制单元, 通过Attention Score来生成访问内存的“指针”. 如果以传统的Attention为例, 我们可以把当作指令内存, 而原有的Token构成的Context当作一个栈的结构被存放在寄存器内. KV-Cache实质上就有一种很直观的D-Cache的概念.

更进一步, 最近对Attention的一些算法上的修改, 例如Linear Attn上构建出的一些状态矩阵的结构, Sparse Attention中一些indexer的结构, 以及UT/MoR这些在Attention上的递归结构, 实质上也可以看作是对Instruction Memory进行某种程度的修改.

### 1.2 Attention

关于Attention的一些演进, 前面一篇文章[《谈谈未来Attention算法的选择, Full, Sparse or Linear ?》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496753&idx=1&sn=b66ffd8d2e977cb4e7e27603ea9a9951&scene=21#wechat_redirect)已经详细的进行了一些分析. 总体来看Attention算法的复杂度还在进一步升高, 先说结论, 从算法的角度看有如下两个观点:

对于Sparse Attention带来的内存访问的变化, 硬件需要考虑访存效率上的优化.

对于Sparse和Linear两条路径的选择, 需要把Linear Attn **避免Softmax** 和 它等价于一个RNN模式引入的 **“状态矩阵”** 两者分开讨论. 从而在硬件设计上还是要为SM配置较高的SFU算力.

对于观点1, 以我们写程序的视角来看, 我们期望有一个非常大近似无限的内存(即对于大模型需要一个非常长的context), 但实际上只需要一个有限内存的栈即可, 任务分治下可能我们真的不需要很长的context. 例如我最近在用一些模型读一些数学书时, 通常我会用pdftk这样的工具截取文章的不同章节分开处理, 然后再拼接一些summary结果继续处理. 同时针对书籍本身的章节依赖结构来做一些手工的block选择.

另一方面在一些DeepResearch相关的任务上, 作为一个应用的开发者, 更希望的是对获取的不同的文档并行的进行Prefill后, 我可以按照Block-Level进行重组和拼接成一个新的context. 基于这两点, 从技术路线选择上来看, 个人比较倾向于Sparse Attention的路线. 即Context-Engineering和Sparse Attention 协同设计的方式, 避免拼接后的内容再做一次整体的Prefill.

结论
Sparse Attention是一种非常直观的选择. 实际上我们需要这样的按块处理的方式构建一个相对较小的栈的结构, 尽力提高Data Locality, 类似于传统计算机系统中充分利用Locality避免Register Spill/Cache Miss的行为.

对于观点1, 在[《大模型时代的数学基础(9)- SDPA和最优传输, 强化学习及信息几何的联系》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494688&idx=1&sn=3d589f6d4be56ee372d5db4f8631b0cc&scene=21#wechat_redirect)中介绍了一篇论文的观点: 注意力机制的前向计算过程, 即通过Softmax函数生成的注意力权重的过程, 完全等价于一个单边的熵最优输运(One-Sided Entropic Optimal Transport, EOT)问题的精确解. 另一方面,通过标准反向传播计算出的梯度, 在数学上等同于强化学习中的REINFORCE这样的策略梯度(Policy Gradient)算法. 意味着注意力机制的学习过程是一种理性的有明确目标的优化策略.这揭示了注意力学习的动态过程--它会“奖励”那些带来高于平均效用的键(Key), “惩罚”那些低于平均效用的键.

结论
从数学的角度来看, 我们不能过早地放弃Softmax, 特别是在硬件架构上. 因为SDPA本身的框架上是非常自洽的. 比起传统的观点认为Softmax是可微分的argmax, 在Optimal Transport的视角下从理论上推导出来SDPA完全等价于一个单边的熵最优输运(One-Sided Entropic Optimal Transport, EOT)问题的精确解.

另一方面对于Linear Attn等价的 RNN 模式引入的“状态矩阵”, 从某种意义上来看, 相当于一台计算机有了一个根据状态可动态调整Instruction Memory中指令的能力. 同时在UT,MoR这些Attn的工作中也在引入递归结构, 另一方面是《Titans: Learning to Memorize at Test Time》这篇论文中也在引入一个Test Time的Memory结构, 特别是对未来一些任务, 模型可能需要赋予在线学习能力时, 使用额外的算力来修改“instruction memory”是一条值得探索的路.

结论
Attention Block内会逐渐再演化出一个利用算力生成的Scratch Pad(我们可以将其称为“状态矩阵”,“短期记忆”等), 并且它还可以控制Attn自身的计算指令, 即在1.1节中的“Instruction Memory”会出现被额外的算力去修改的情况. 这也符合一个非常直观的观点: AGI的实质是通过一个模型构造一个自己能够产生代码运行的通用计算机架构.

总体来看, 整个Attention Block未来几年的演进会变得更加复杂, Sparse Attn会对SM有更高的内存访问需求, 另一方面多个算子融合对于计算资源的调度和流水线排布引入了新的复杂度. CUDA-Graph和PDL是否够用? 是否易用? 多个算子之间 Overlap 时的资源约束和隔离如何进行? 另一方面是关于存储的问题 大量的KVCache如何高效的缓存和快速获取? 这个话题会在下一章详细展开.

### 1.3 MoE

对于MoE的算法演进, 以前有一篇文章

[《详细谈谈DeepSeek MoE相关的技术发展》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493157&idx=1&sn=51c0e27a347dd3fe1ed868d87f667897&scene=21#wechat_redirect)

这个问题涉及到Fine-grain MoE到底会有多细的问题. 现阶段专家数目为256/384. 类比于传统计算机架构中的PageSize的取舍, 同等参数规模下, 如果专家数量较少, 相对于单个专家的参数更多, 算力的开销会非常大, 同时访存的开销也很大. 因此我们希望有一个Finegrain的MoE, 用更多的专家数, 一方面让这些小的“内存页”能够在训练中更充分的被训练.

但是真的会出现1024 Expert的情况么? 结论是比较困难的, 因为专家太稀疏会导致训练难以控制. 类比一个传统计算机架构中的例子是, 假设我们训练的信息是5KB, 系统采用4KB的页, 因此训练的数据信息将会跨越两个不同页, Expert Specialization实际上是受损的.

当然这里还有一个很有趣的想法,即构建一个2级的页表, 首先命中一个相对较大的页(L1 Expert), 在L1 Expert内部再进一步构建Expert Specialization即L2 MoE. 当然这一点也不一定有收益, 因为本来涉及到EP并行中的跨卡通信的Overhead以及更小的专家参数带来的GEMM效率相关的影响这些TradeOff下, 可能没有额外的收益.

结论
MoE中专家数量不会进一步上升, 可能很长一段时间停留在256/384专家数的规模.

## 2. 基于微架构的视角

在Blackwell中引入TMEM后, TensorCore无需占用RMEM, 并且UMMA相关的指令只需要一个线程issue, 从这个角度来看 TensorCore 已经是一个相对独立的组件了. 仅是涉及TMEM内存管理和拷贝的操作需要Warp Level issue. 其实从管理复杂度来看, 下图中的Sch warp/TMA Warp和TC Warp都可以做到单线程的处理, 仅有EpilogueWarp才需要传统CUDA中的SIMT处理逻辑.

![图片](assets/fd5d4d8d5456.png)

从一些调度的视角来看, 我认为需要在SM内配置一个超标量核来进一步增加调度算法的可编程灵活性, 以及其它CUDA Core无法有效处理的workload, 如下图所示:

![图片](assets/eee97241a6dc.png)

首先我们从几个方面来看现在的架构遇到的问题, 然后再来详细展开相关Trade-Off的分析.

### 2.1 从寄存器使用量的视角

对于Sch Warp/TMA Warp/UMMA Warp, 实际上单线程就可以执行, 因此整个Warp中还是有大量的寄存器资源是被浪费的. 虽然在Blackwell中降低了RMEM的占用, 但是伴随着Rubin未来继续加大TensorCore的运算规模, 实际上Epilogue阶段的寄存器压力还是非常大的.

同时针对寄存器在WarpSpecialization的情况下, 编译器如何分配也是一个非常复杂的问题, 其中还耦合了大量Warp调度相关的复杂性.

另外从应用的视角来看, Attn的算法复杂度还在继续增加, 不确定未来是否还有Epilogue阶段进行一些计算后, 结果从RMEM ST到TMEM触发另一个GEMM运算的case.  例如一些像 UT/MoR 一类的递归Transformer. 或者通过RMEM的一些计算结果动态生成一些TMA描述符的可能性, 例如Sparse Attn中一些算法的演进也需要更灵活的异步内存访问.

无论是TensorCore还是TMA, 实际上都可以采用单线程的方式进行issue. 如果在SM内构建一个超标量核用于处理MBarrier和动态的去Issue TMA和TensorCore的指令流. 一方面是可以节省不必要的寄存器开销, 另一方面将它们的指令和CUDA Core的指令流路径分离也能获取收益.

另一方面是TMA Descriptor动态生成的能力, 虽然当前是CPU来生成并存放到GPU, 然后在Kernel执行时进行Prefetch. 这种模式对于当前的workload来说 由于TMA Descr基本没有变化的需求时足够的. 但是未来的一些Sparse Attention可能还是需要更加动态的TMA加载能力.

### 2.2 从Warp调度的视角

今年三月有一篇文章《Analyzing Modern NVIDIA GPU cores》[2]对于Nvidia GPU中CUDA Core的一些微架构进行了一些分析. 其中对于Warp调度和指令Issue的逻辑进行了逆向分析

![图片](assets/a9ff17f5035b.png)

*当然这些分析是否准确对于非NV的同学, 还是值得进一步去逆向分析的.*

不过从调度器的角度来看, 硬件调度器在做指令发射的时候, 考虑到指令发射压力和硬件调度的效率, 通常都会采用非常简单的贪心和或者轮询策略. 例如论文中推测的一个调度策略, 针对Warp是否Ready的条件如下:

指令缓冲区中有有效的指令

warp的最老的指令不能与同一个warp中尚未完成的较旧的指令存在任何数据依赖的风险, 指令之间的数据依赖通过控制位处理.

对于固定延迟的指令, 一个warp只有在可以保证一旦发射就能获得执行所需的所有资源时.才会被认定为在给定的cycle内发射其最老指令的候选者. 这些资源包括执行单元(EU), EU有一个输入锁存器, 如果EU的宽度为半个warp, 则该锁存器占用两个Cycle, 如果宽度是一个完整的warp,则占用一个周期.

Warp调度器使用一种贪心策略, 如果满足前述Warp准备就绪条件, 就从同一个Warp中选择指令. 当转换到其它Warp时, 则选择满足前述条件的最年轻的Warp. 一些例子如下图所示:

![图片](assets/28e4e9624d2e.png)

当RMEM 压力加大时, 可能这样的调度策略并不一定是最优的, 最近和一些做算子的同学在聊天, 他们也希望有一定的控制Warp调度的能力. 实际上在Blackwell上也出现了相对简单的一个Cluster Launch Control(CLC)的方案来避免一些算子因为调度而产生的长尾, 但是可能我们还需要更灵活的调度方式.

另外从编译器的角度或许也是有一个额外的可编程的超标量处理器给出一些简单的调度平衡策略上的Hint也挺好的?

实质的问题是, 我们针对调度需要更多的复杂性, 实际上可以表述为需要额外的算力来进行处理. 通过将SM内构建一个超标量的核和相对独立的指令流和分支跳转能力, 并且控制Warp Scheduler可以带来更多的灵活编程和动态流水线排布, 甚至是一些启发式流水线排布的能力.

### 2.3 从Kernel Launch的视角

当算力变得越来越强的时候, 通常一个Kernel的运行时间小于100us时, Kernel Launch的代价就变得更大了. 虽然有一些Persistent Kernel /CUDA Graph / PDL的处理方式. 沿着这条路走下去, 比较极致的方法就是构建一个Mega Kernel, 例如Mirage[3]这样的项目. 但是Mirage这样的实现, Scheduler本身也会占用较多的SM资源.

从根本上来看, Kernel Launch的延迟来自于几方面, 首先是CPU侧 Runtime的参数打包, 用户态/内核态切换, 驱动命令生成等,  另一方面是CPU和GPU之间通信的延迟, 如下图所示, 通常需要数百个Cycle.

![图片](assets/80a91e6b982a.png)

最后是GPU硬件调度, 例如命令解析, 资源检查, 线程块分配, Warp创建与调度等.cudaGraphLaunch绕过了几乎所有的CPU端开销和通信开销, 直接在GPU上触发一个预先编译和优化的执行路径.

对于灵活的调度控制在GPU的SM内构建一个简单的超标量核, 不但可以拿到和CudaGraph相似的性能收益, 同时还可以增加Graph的动态性.

### 2.4 ScaleOut和ScaleUP通信融合

现阶段MoE模型所使用的DeepEP通信都会占用相当一部分的SM资源用于通信, 并且耦合了Warp调度的一些影响和复杂度, 同时由于ScaleUP内存语义和ScaleOut消息语义不同, 进一步占用了SM的计算资源. 因此在DeepSeek的一篇论文《Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures》[4]中的4.4节专门阐述了Scale-Up and Scale-Out Convergence. 其中有4点建议:

**Unified Network Adapter**: 设计NIC或者I/O Die能够统一的连接ScaleUP和ScaleOut网络, 这部分内容我们将在后面一章详细展开.

**Dedicated Communication Co-Processor**:引入专用协处理器或可编程组件（例如 I/O 芯片）来处理网络流量.该组件可将数据包处理从 GPU SM 卸载, 从而防止性能下降. 此外, 它还应包含硬件加速的内存复制功能, 以实现高效的缓冲区管理.

**Flexible Forwarding, Broadcast and Reduce Mechanisms**: 这部分也会在下一章互连结构中讨论.

**Hardware Synchronization Primitives**:提供细粒度的硬件同步指令, 以在硬件层面处理内存一致性问题或乱序数据包到达. 这将消除对软件同步机制（例如 RDMA 完成事件）的需求.

另外如何统一ScaleOut和ScaleUP实质是语义上的问题, 例如ScaleOut当前的RDMA语义有什么问题? WQE准备对CUDA Core资源的占用和单发射引入的延迟, 以及等待DATA写入GMEM的延迟和kick doorbell产生的延迟. 还有一个问题是针对接收端的Completion Queue的处理, 如果按照队列的方式读取, 由于CQ存在GMEM中, 从SM读取需要多次因此也会产生额外近1000个cycle的延迟.

如果在ScaleOut支持统一的Memory语义需要如何处理? 直接透传LD/ST还涉及大量的异常处理和内存模型的问题. 实质是需要在整个通信上引入一些条件执行和异常处理的能力.

实际上这些需求本质上指向了我们需要在SM内部实现一个超标量核. 在外部SM/GPC间的NOC实现一个独立的Co-Processor会带来一个较大的问题是, 对于SMEM上的MBarrier更新通常会经历几百个cycle的交互才能被这个独立的Co-Processor处理. 而在SM内部构建, 直接可以以非常低的延迟读取SMEM, 甚至是可以让TensorCore和TMA的MBarrier更新到完全紧耦合到这个超标量核的一片Private Memory上.

### 2.5 SPMD or MPMD

其实当这个SM内存在一个乱序的多发射的标量核后, 还有一个潜在的优势可以解锁MPMD. 这一点上价值是巨大的.

首先当前的NV GPU架构是一种SPMD (Single Program, Multiple Data - 单一程序, 多重数据)的处理方式, 所有并行执行单元 (例如, 线程, 进程, 核心) 运行完全相同的程序代码副本. 但是, 每个执行单元会根据自己唯一的ID (如线程ID, 进程排名) 来处理不同的数据子集或执行代码中的不同分支. 它是一个典型的**数据并行**模式.

实际上当单个GPU的算力越来越强的时候, 单一算子可能很难打满整个GPU, 例如在Rubin Ultra中四个Die时考虑访问内存的跨Die延迟影响等因素下要做一些亲和性调度或者特殊的Layout. 另一方面我们是否能够探索**任务并行**的方式? 即MPMD, 不同的并行执行单元可以运行不同的程序代码. 这允许在功能上进行划分, 让不同的执行单元扮演不同的角色, 专注于不同的任务.

实际上在当前的NV GPU架构下, 使用Green Context已经可以实现类似的功能, 并且在某些场景下也有了明确的性能收益. 而在Google的《PATHWAYS: ASYNCHRONOUS DISTRIBUTED DATAFLOW FOR ML》[5]也谈到了这一点.

![图片](assets/8c5fc141ccd5.png)

### 2.6 编程接口

针对带有超标量核的SM架构, 我会为它设计一片很小的Private SMEM(例如2KB~4KB)用于存放MBarrier, 这样异步程序架构就可以更容易的解耦合了, 不需要复杂的WarpSpecialization的处理, 就是一个TC Function, 一个TMA Function, 和原来的CUDA SIMT的kernel function做Epilogue即可. 然后会将Sch warp以及TMA/ TC的描述符准备相关的function放入这个标量核, 更进一步还可以和Warp Scheduler做更进一步的调度交互. 实质上还是类似于Helion/TVM/Tilelang的想法, 采用调度和算法分离的方法. 代码上可能更有利于编译器生成和优化.

另外针对IBGDA, GIDS(GPU Initial Direct Storage)这些, 部分文件系统的处理逻辑也可以放入到这个标量核内, 降低了通信SM的占用.

甚至是在这种情况下, 我们可以做一些更复杂的MPMD的编程和尝试. 特别是在Rubin Ultra上, 4个Die拼接时还有很多好玩的并行策略可以去做,例如用一些Green CTX配合CTA affinity做一些事情更好的利用片内的结构

![图片](assets/d9371b40c7c2.png)

### 2.7 PPA的分析

简单的来说, 用一个超标量核去换原来的CUDA Core做单线程的Issue, 并且还受到Warp Scheduler带来的一些调度延迟的影响, 从PPA的角度来看是更划算的.

然后从总体的SM面积增加一个标量核的面积几乎是可以忽略不计的, 其中I-Cache和D-Cache只需要L1即可, 大概32KB应该足够了,  整体来看在这个超标量核中运行的代码大概也就几千条指令的规模. 另一方面需要一个额外的Private SMEM大概4KB即可, 用于处理和TMA/TensorCore之间交互所用到的MBarrier以及RDMA一类的CQ和一些临时性的描述符.

## 3. 互连架构相关的视角

### 3.1 NOC互连

首先还是从NOC的视角来看. Nvidia TensorCore的设计其实挺值得考虑的, 相比之下很多DSA的架构喜欢更大的TensorCore, 例如Google在第一代TPU选择了256x256的脉动阵列, 而后面缩减到128x128. Nvidia则是用相对较小的TensorCore, 然后在Hopper使用了WGMMA, 整个SM 4个TensorCore构成一个更大的执行单元. 而在Blackwell中进一步引入了2SM, 未来Rubin可能还会进一步引入4SM.

但是这样的坏处是在数据路径上会增加复杂度. NV在这个地方做出了一些取舍, 通过提供硬件保证的并发执行(CGA)和低延迟的直接通信路径(DSMEM + GXBAR), 该设计从根本上解决了传统模型中跨SM协作的低效问题. 它使得GPU能够真正地像一个紧密耦合的多核处理器集群一样工作, 而不仅仅是多个独立处理单元的集合. SM-to-SM流量与L2/DRAM流量在GPC层面进行物理分离(GXBAR vs. MXBAR)是非常明智的选择.

另一方面在GPU NOC上, Nvidia避免了像Dojo/Tenstorrent一类的2D-Mesh的拓扑, 这类架构对于NOC的处理和编译器的优化会带来更多的复杂性. 而Nvidia的选择是尽力的去构建一个XBAR并在数据路径上增强组播能力:

![图片](assets/cf398a41e248.png)

### 3.2 ScaleUP互连

前段时间在[《谈谈ESUN, SUE和UALink》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496512&idx=2&sn=0c10cef05fb1cc4e175f326d62b266e3&scene=21#wechat_redirect)中已经详细探讨了该问题. 对于ScaleUP的规模分析如下:
从并行策略分析
首先从并行策略来看, PP并行和DP并行实际上当前已经可以被很好的Overlap了. 对于TP/SP/CP都是对张量进行切分后的分块运算, 对于整个分块的计算规模和XPU的算力可以得到一个计算延迟. 另一方面例如MHA按头的数量切分, TP的规模本身就会限制在64/128. 当矩阵被切的太小后, 又会对XPU算力带来影响, 导致小矩阵无法打满, 这些都是约束ScaleUP节点规模的因素.

另一方面是物理约束带来的, 大规模的组网也必然会带来的静态延迟, 例如一米光纤的传输延迟差不多4~5ns, 通常需要2层组网的大规模ScaleUP整体光路径延迟接近500ns, 另一方面由于需要穿越至少3跳交换机, 以及考虑到这样的系统需要更复杂的可靠传输协议, 整体延迟接近3~5us. 因此在这类并行中的集合通信开销也有非常显著的影响.

当然国内有一种说法, 由于国产芯片算力比较弱所以需要更大的TP域. 但是这种说法似乎也不太成立, 多Die的封装去ScaleUP单颗的算力 vs 单颗很弱的芯片通过大规模ScaleUP组网, 一定会选择前者. 而且现在国产XPU来看, 单颗性能已经可以做到和H100差不多的算力了, 因此这个观点也是不成立的.

接下来讨论EP并行, 一卡一专家的观点是无效的, 边际收益并不大. 对于单个卡而言, 在并发处理 请求Decode的过程中, 会产生  个不同的Token, 每个Token按照Top-K, 从  个专家中选择  个, 则一次剩下的未被访问的专家的概率为 , 累积  个Token, 未被取出的概率为 , 则需要访问内存的专家数为

假设一个极限的稀疏的情况, 专家数目为M=512, K=8, batchsize N=32时, 需要访问的专家数目为202个. 当专家数扩展到1024个更极端的情况下时, 需要访问的专家数为227个. 这样就带来了大量的内存访问瓶颈. 通过EP并行, 例如每卡8个专家, 访问专家参数的带宽已经下降到原来的3%. 此时EP的规模为128卡. 这就是EP收益的来源.

那么对于EP并行的边际收益来看, 如果超过交换机的Radix需要两层组网, 本身就带来更高的延迟, 累计将会影响到TPS. 同时再对比Token的累计访问内存带宽和参数的访问内存带宽, 可以得出并不需要一卡一专家的部署.

更进一步从部署的规模而言, 由于大模型在Serving过程中随着时间的变化有明显的峰谷效应, 集群应该根据请求有更好的弹性扩容能力. 单次扩容1024卡和单次扩容64/128卡相比, 整体的成本肯定是更细粒度的扩容更加经济并具有弹性.

结论:
从并行策略来分析, 满足交换机单层组网的Radix规模, 例如最大512卡即可.实际部署时可能考虑弹性缩扩容的需求, 还会进一步降低ScaleUP的规模. 您可以看到即便是Rubin Ultra的NVL576, Kyber机柜的背面可以看到, 单个ScaleUP域也只有144卡的规模. 并且NV还是在选择铜互连.

从系统可靠性分析
另一个决策点是ScaleUP的物理互连距离, 如果大于5m可能就需要光互连了. 光互连现阶段还是存在一些稳定性的隐患, 即便是单层组网, 也需要在可靠传输上做更多的工作. 当然这一点上并不是说铜比光好, 只是现阶段而言, 光传输的平均无故障时间(Mean Time Between Failures,MTBF)还没有达到和铜相同的数量级. 但是我们还是要对OIO/CPO/NPO/MicroLED等技术的演进持更加开放的心态, 如果这些问题解决了, 那么光互连也势在必行.

另一方面是从集群的视角, 对于ScaleUP域, 节点数量增加也会显著降低整个集群的MTBF. ScaleUP节点扩大一倍, 则平均无故障时间缩短一半. 因此我们也需要控制ScaleUP的规模, 或者通过可靠传输的设计来规避一些故障, 增加MTBF, 或者采用备份节点的方式.

然后就是从成本的视角考虑, 当某个XPU的互连出现故障后, 如果有相应的备份节点(例如UB-Mesh 64+1), 也可以显著的降低故障的影响. 但是同时也带来了资源的消耗, 有一张卡长期处理冷备份的状态, 整体解决方案的成本也会显著提高. 当ScaleUP集群故障时, 整个爆炸域的影响也是巨大的. ScaleUp规模越大, 受损的经济损失需要按照MTTR * ScaleUP节点数计算.

结论: 从系统可靠性的角度来看, 我们依然需要约束ScaleUP的规模.

另一方面, 我们也可以看看BRCM以太网交换机BU的GM Ram Velega先生的反馈, 他也认为构建单层的交换网络即可

![图片](assets/0a7c06f0faf4.png)

另一方面从延迟的视角分析.如果使用多层交换机组网构成一个超大规模的ScaleUP, 一方面是传输的延迟变得更加显著, 另一方面由于稳定性影响导致需要更复杂的可靠传输协议来解决问题, 例如拥塞控制, 多路径负载均衡, 基于Lossy的丢包重传等. 这些都会导致延迟显著增加.

例如以两层组网的ScaleUP为例, 传输延迟为3~5us 单个SM内的SMEM容量为256KB, 按照Ping-pong buffer来看, 以及矩阵乘法中其它矩阵的占用空间, 实际上用于传输的大概只有40KB, 则单个SM的峰值传输带宽为10GB/s. 虽然我们可以整体来看简单的做一个乘法(25GB/s * num_SM) 得到峰值的ScaleUP的峰值允许带宽, 但并不是所有的SM都在同一时刻并行的发出流量. 因此延迟将极大的约束整体的峰值ScaleUP带宽.
从ScaleUp和ScaleOut融合的角度分析
首先是内存模型的视角, 在[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect)已经有一些详细的分析. 另一方面是可靠传输选择的分析在[《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect)也有详细的分析.  然后从GPU微架构的视角也有一个分析[《从GPU缓存的视角看芯片设计和互连》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495963&idx=1&sn=00f05c90d7ec22f90911ac4618180c9a&scene=21#wechat_redirect).

其实这也是我比较喜欢的一种方式. 实质上就是把原SMEM---Local HBM---Remote-HBM的路径改为SMEM--- Remote Memory Pool的方式, 在这里适当的添加一些算力, 例如LayerNorm和集合通信的Reduce计算/Allgather的通信等.

![图片](assets/db1bdb883cc2.png)

这样的也就匹配了DeepSeek关于ScaleUP和ScaleOut融合的需求, 首先在第一层的交换网上采用Lossless的传输和现有的NVLink/UAL保持一致, 进一步的降低GPU之间的通信延迟. 然后有额外的一些I/O节点或者NIC芯片, 配置一部分很小的内存(甚至不需要外挂DDR, 内置的SRAM就够)用于处理一些集合通信相关的状态和一些跨域之间的通信, 甚至是接入到存储集群.

另一方面这些I/O Node配合第二章建议的SM中的超标量核会产生很多有趣的场景. 打个哑谜吧.

参考资料

[1] 
Titans: Learning to Memorize at Test Time: *https://arxiv.org/pdf/2501.00663v1*
[2] 
Analyzing Modern NVIDIA GPU cores: *https://arxiv.org/pdf/2503.20481*
[3] 
Mirage: *https://github.com/mirage-project/mirage*
[4] 
Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures: *https://arxiv.org/html/2505.09343v1*
[5] 
PATHWAYS: ASYNCHRONOUS DISTRIBUTED DATAFLOW FOR ML: *https://arxiv.org/pdf/2203.12533*