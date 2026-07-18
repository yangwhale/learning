# Tensor-103.3 Hopper Persistent Kernel

> 作者: zartbot  
> 日期: 2025年10月27日 13:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496608&idx=1&sn=1eac0c4b71e7c5251c1ca031cb8e744e&chksm=f995e362cee26a749bef66cca5397f410675dbc46b45851b335fd15f93b8ec8d1cc3ee7e7923#rd

---

### TL;DR

这一篇是对Hopper GEMM的补充, 从整个网格的视角还有两个优化. 在前一篇中介绍了CTA Swizzle来提升L2缓存的命中率. 而这一篇将主要介绍Persistent Kernel, 更好的将任务调度到各个CTA, 以充分利用计算资源并实现良好的负载均衡.在《CUTLASS Tutorial: Persistent Kernels and Stream-K》[1]有非常详细的介绍. 但是这一篇我们将使用CuteDSL来详细展开. 本文目录如下:

```
1. Tile & Wave Quantization问题1.1 Tile Quantization1.2 Wave Quantization2. Stream-K2.1 传统的解决量化效率低下方法2.2 Split-K分解算法2.3 Stream-K分解算法2.4 Hybrid Stream-K3. Persistent Kernel3.1 Host函数3.1.1 Tile调度抽象3.2 Kernel函数3.2.1 初始化3.2.2 Persistent Main Loop3.2.2.1 DMA Warp3.2.2.2 MMA Warp3.2.3 Epilogue
```

## 1. Tile & Wave Quantization问题

在《Matrix Multiplication Background User's Guide》[2] 中有这个问题的详细阐述.

特性

Tile Quantization

Wave Quantization

层面

微观 (单个线程块内部)

宏观 (整个GPU层面)

量化单元

Tile尺寸 (如 128x128)

Wave大小 (如 132个Tile)

浪费来源

线程块内部分线程的无效计算

整个SM在尾波期间的空闲

影响范围

主要影响边缘Tile

影响整个Kernel的执行时间

图表周期

性能随维度以Tile尺寸为周期波动

性能随总Tile数以Wave大小为周期波动

### 1.1 Tile Quantization

为了确保覆盖所有输出矩阵的元素, GPU启动的CTA时所使用的Tile数量会向上取整. 当矩阵的维度不能被CTA的Tile Size整除时, 就会发生Tile Quantization效应. 如下图所示

![图片](assets/c1f2f0532d75.png)

尽管算法上仅需要多出0.39%的运算, 但实际执行的算术操作数量却是左图的1.5倍. 这表明, 只有当输出矩阵的维度能够被Tile维度整除时, 才能达到最高的利用率.

为了进一步说明这个效应, 考虑一个GEMM例子: `M = 27648`, `K = 4096`, 并且库函数固定使用256x128的Tile. 当以 8 为步长将 N 从 136 增加到 256 时, 这个由Tensor Core加速的GEMM总是运行相同数量的 Tile 列, 即 N 维度总是被划分为2个Tile. 尽管Tile数量不变, 但这些Tile中包含有效数据的比例, 会随着N的增加而增加, 这也反映在下面的左图的GFLOPS**中. 注意, 吞吐量在`N = 128` (此时每行的一个Tile被有效数据完全填满)和`N = 136`(此时每行增加到第二个Tile, 但其中只有8/128 = 6.25%是有效数据)之间出现了显著的下降. 同时, 请注意只要Tile数量保持不变, 执行时间(duration)也是恒定的.

![图片](assets/988adf955014.png)

当一个维度逐渐增加时, 性能曲线会呈现锯齿状.

**下降:** 维度刚刚超过Tile尺寸的倍数, 性能骤降.

**上升:** 维度继续增加, 数据填充新Tile的"空白区域", 有效工作比例增加, 吞吐量线性回升.

**峰值:** 维度再次达到Tile尺寸的整数倍, 性能达到局部峰值.

通常为了解决这样的问题, 需要动态的根据原始矩阵的Shape选择稍微小一点的Tile Shape, 或者在算法上将矩阵的维度对齐到Tile的维度.

### 1.2 Wave Quantization

Nvidia GPU由多个SM构成, 每个SM有独立的SMEM/RMEM/TensorCore等硬件资源, 并且各自独立运行. 理想情况下, 负载能够均匀的分布到每个SM并行执行, 并且在整个Kernel执行期间所有的SM都保持忙碌状态. 但是如果出现负载不均衡的情况, 某些SM分配的任务较少提前完成, 这样就会导致资源空闲, 等待其它SM完成.

以GEMM为例, 通常每个SM会负责一个 bM x bN 的Tile运算作为一个工作单元(Work Unit). 这些工作单元会被分配给CTA, 每个CTA会在可用的SM上完成计算. 当工作单元的数量超过可用 SM 的数量时, 这些工作单元会分批次被处理, 每一批次称为一个波（wave): 即每个可用 SM 各自完成一个工作单元，便构成一个满波(full wave). 当总的Tile数量无法被GPU上SM的数量整除时, 需要额外的一个wave才能完成处理.

如图所示, 假设有10个工作单元, 仅有4个SM处理时, 最后一个wave只有一半的SM被占用.

![图片](assets/930e63a4e143.png)

考虑一个例子, `K=4096`, 但使用一个较小的 `M=2304`并改变 N 维度,  一块NVIDIA A100 GPU拥有108个SM. 在使用256x128的Tile的特定情况下, 每个SM可以执行一个线程块, 这意味着一个 wave 的大小是108个可以被同时执行的Tile. 因此, 当总的Tile数量是108的整数倍或刚好略低于该倍数时, GPU的利用率会最高.

在这个例子中, M维度 (`M=2304`) 将总是被划分为 `2304 / 256 = 9` 行Tile. 性能跳变点:

**N=1536:**`N维度Tile数 = 1536/128 = 12`. 总Tile数 = `9 * 12 = 108` (1个满波). 性能峰值.

**N=3072:**`N维度Tile数 = 3072/128 = 24`. 总Tile数 = `9 * 24 = 216` (2个满波). 性能峰值.

**N=4608:**`N维度Tile数 = 4608/128 = 36`. 总Tile数 = `9 * 36 = 324` (3个满波). 性能峰值.

**N=6144:**`N维度Tile数 = 6144/128 = 48`. 总Tile数 = `9 * 48 = 432` (4个满波). 性能峰值.

在每个峰值点之后, 只要 N 稍微增加 (例如从1536到1544), 就会引入一个利用率极低的尾波(tail wave), 导致性能骤降.

## 2. Stream-K

为了解决Wave quantization问题, 我们需要创建一个更好的分区和调度方案.
首先我们需要对《Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU》[3]算法展开分析.

这篇论文提出了一种名为 Stream-K 的新型并行分解策略, 用于在GPU上高效执行GEMM及相关计算. 与当前主流的基于 "Tile" 的数据并行分解方法不同, Stream-K采用了一种"以工作为中心"(work-centric)的思路. 它将整个矩阵乘法所需的全部乘加(Multiply-Accumulate, MAC)循环迭代次数作为总工作量, 然后将这个总工作量均匀地划分给GPU上的物理处理单元(SM).

### 2.1 传统的解决量化效率低下方法

正如前一章所介绍的, 传统的基于数据并行的处理方式将矩阵分块放置到多个处理器上并行运行的算法, 伴随着处理器核心数量增多, 这种数据并行分解的方式在应对各种矩阵Shape时, 由于任务块(输出Tile)的数量无法被处理器核心数整除, 最后一"波"计算任务将导致部分核心空闲, 从而拉低整体利用率, 即量化效率低下 (Quantization Inefficiency)问题.

当输出Tile的数量远超过SM数量时, 每个SM都是在Oversubscription的状态, 任务完成时就会迅速被新的任务填补, 这样保持了很高的利用率. 但是随着现代GPU SM数量越来越多, 即便有数百个Tile并行运算, 也只需要几个wave就能做完, 这样就大大增加了最后一个wave不满载而导致的性能影响. 考虑一个简化的问题, 如下左图所示, 假设我们输出Tile有9个, 整个仅有4个SM的情况下, 最后一个wave将浪费掉3/4的算力.

![图片](assets/a46bfed2e363.png)

有一种办法如上右图所示, 为了更高的SM利用率, 并降低负载不均衡的影响, 我们可以把Tile拆小, 然后构建多个wave来降低负载不均衡的影响. 即ensemble of tiling configurations方法.  我们可以把Tile的尺寸从左图的128x128减半为128x64. 总共的Tile数量为18个,按照5个wave在4个SM中运算, 因此理论的利用率上限为 18 / (5 wave * 4 SM) = 90%.

因此计算库当理想的分块因子无法很好地量化时, 会从备选的平铺方案中选择一个具有更小并发工作量的. 但是这样计算库的代码量会产生膨胀, 因为我们可能不得不为某个给定的API给每个GPU架构提供数十个预编译的内核特化版本. 并且为每个新GPU架构维护和调优一个庞大的内核集合是一项巨大的工程挑战.另一方面启发式搜索算法也有一些局限性, GEMM的性能空间是极其复杂和非凸的. 设计一个能够为任意(m, n, k)组合以及转置情况都选对内核的启发式规则几乎是不可能的任务.

然后作者在第二章Background中介绍了整个GEMM优化的演进过程, 大致可以分为五个阶段.

**图形API时代**: Larsen和McAllister的工作代表了在GPGPU**概念的黎明时期, 把矩阵数据存为纹理, 利用像素着色器中的纹理采样和颜色混合硬件(本质上是乘加操作)来模拟矩阵乘法. 巧妙地利用固定的图形渲染管线来完成计算任务.

**CUDA**+SMEM时代**: CUDA的诞生是一个分水岭. 可编程的SMEM是关键. 它是一种由程序员手动管理的, 速度远快于GMEM. 这催生了至今仍在使用的核心优化技术——两级分块: 第一级分块(Grid-level): 将输出矩阵C分解为Tiles, 每个Tile分配给一个CTA(线程块).第二级分块(CTA-level): 在一个CTA内部, 将计算一个Tile所需的输入矩阵A和B的子块(sub-tile)从慢速的全局内存加载到快速的共享内存中. CTA内的所有线程可以重复访问共享内存中的数据, 从而实现数据复用, 大大减少了对全局内存的访问带宽压力.

**应对多样性 - 内核集合的诞生(MAGMA)**: MAGMA团队的工作是解决"量化效率低下"问题的第一次系统性尝试. 他们认识到单一Tile尺寸无法适应所有GEMM形状. 他们的策略是: 首先创建一个带有可变参数(如Tile尺寸)的内核模板.预先生成数百个不同参数组合的内核, 对它们进行基准测试, 并从中挑选出少数几个(3-5个)在不同尺寸范围表现优异的内核, 组成一个"集合(ensemble)".在运行时, 使用非常简单的 if-else 逻辑, 根据矩阵的 m, n, k 尺寸来决定调用哪个内核. 这套"生成-筛选-选择"的流程奠定了现代数学库的基础.

**复杂化与智能化**: ISAAC: 代表了"智能化选择"的思路. 它试图用机器学习模型来取代MAGMA中的手写规则, 以期做出更精准的内核参数预测. cuBLAS: 代表了"工业级蛮力"的思路. 它预置了大量的"算法"(这些算法不仅是Tile尺寸不同, 还可能包含了不同的分解策略, 如数据并行, K轴切分等). 它依赖于NVIDIA工程师精心调优的内部启发式算法来做选择. 作者在这里尖锐地指出了这种方法的后果——"笛卡尔积"导致的代码膨胀(code bloat).

**编程模型与DSL**抽象**:这一段非常重要, 它将本文的工作与另一大类相关研究区分开来. CUTLASS, Triton, TVM等工具本身不是GEMM的分解策略, 而是用于实现这些策略的工具或语言. CUTLASS提供了一套C++模板化的"乐高积木", 比如用于线程/Warp/CTA级别的数据加载, 存储和计算的组件. 开发者可以用这些积木来**搭建(compose)** 自己的GEMM或类GEMM内核. Stream-K可以被看作是一种新颖的"搭建方案". Triton, Halide, TVM: 这些是更高层次的DSL(领域特定语言). 它们的核心思想是将`算法逻辑`与`调度策略`分离. 程序员用简单的语言描述"计算什么"(如点积), 然后在"调度"部分指定"如何计算"(如何分块, 如何并行). 编译器负责将这两者结合生成高效的GPU代码.

将Stream-K放在这个背景下, 我们可以更清晰地认识到它的贡献: 它不是一种新的编程语言或编译器, 而是一种新的, 通用的调度策略(a novel scheduling policy), 这种策略可以被各种工具(如CUTLASS)实现, 或者在未来被DSL(如Triton)的编译器自动生成.

### 2.2 Split-K分解算法

在论文的第三章中介绍了一些现有的工作分解策略 (Existing Work Decomposition Strategies), 从最早的六重循环到Tile-based分块等就不再赘述了. 主要来谈谈在 M 和 N 维度上划分之外, 其实还有另一个可以划分的维度：K 方向. 当 K 很大时, 在 K 方向上划分（Split-K）会非常有效；不过 bK 太小, 同样会带来算术强度和延迟隐藏方面的损失.

Split-K 调度方式会将一个 tile 沿 K 方向均匀地分成 s 份(s为切分因子). 如下图所示, 网格变为了一个二维结构[num_tiles, s]. 总CTA数量是 num_tiles × s. 一个Tile沿K轴的iters_per_tile次迭代被划分给了s个CTA. 每个CTA只负责其中一小段.

![图片](assets/3bbc0f5a4a04.png)

由于多个CTA协同计算一个Tile, 每个CTA都只得到一个"部分和(partial sum)". 因此必须有一个机制来Reduce这些部分和. 伪代码展示了一种常见的实现:

"从属"CTA (y ≠ 0) 将其计算出的部分和写入一个临时的全局内存数组 partials.

"主"CTA (y = 0) 负责最后的reduce工作. 它会等待其他所有协同CTA完成(通过检查flags标志位), 然后从partials中读取它们的结果, 累加到自己的结果上, 最后将最终值写入矩阵C.

但是我们仔细分析, 总任务数 num_tiles × s 仍然可能无法被SM数量整除. split-K只是增加了任务总数, 提高了"恰好整除"的概率, 但没有从根本上解决问题. reduce部分和需要额外的全局内存读写和同步. 临时存储partials的大小与输出Tile数量成正比, 通信和同步的次数也与总任务数成正比. 这意味着当问题规模(m, n)很大时, 这个开销会变得非常显著.

### 2.3 Stream-K分解算法

Stream-K策略为每个SM分配一个持久化的单一CTA, 每个CTA会被分配到一个“分数”数量的working tile. 拆分依旧是沿着 K 方向进行划分. Stream-K将GEMM的总MAC循环迭代工作量, 均匀地划分给一个大小恒为 g 的CTA网格. 每个CTA所负责的MAC循环迭代区间, 被连续地映射到GEMM形状的  的线性空间中, 并可能因此跨越输出Tile的边界.

整个算法如下所示:

![图片](assets/94e4b42e0be4.png)

如果一个给定CTA的起始/结束迭代与Tile边界不重合, 它必须将其部分结果与覆盖同一Tile的其他CTA的结果进行合并. 值得注意的是, Stream-K的通信, 同步和全局存储开销与问题规模无关, 而是与CTA的数量 g 成比例. 另一方面当输出Tile的数量大于CTA的数量时, 同步等待的开销可能微不足道. 在这种情况下, 每个输出Tile最多只被两个CTA覆盖, 并且**Tile处理的倾斜(tile-processing skew)** 确保了负责累加的CTA在需要其协作者的贡献时, 那些协作者早已完成并产出了结果.如下图所示:

![图片](assets/e19e7e548a5e.png)

想象一下CTA_0和CTA_1接力计算. CTA_0从迭代0开始, CTA_1从迭代iters_per_cta(如上图中的Tile-2)开始. 假设一条缝隙出现在Tile T 的中间. CTA_0计算T的前半部分, CTA_1计算后半部分. CTA_0是起始者, 需要等待CTA_1的结果. 但是, CTA_0做完T的前半部分后, 它会继续做Tile T+1, T+2... 直到它的任务结束. CTA_1也是如此.

由于CTA_0比CTA_1早开始了iters_per_cta次迭代, 当CTA_1完成对Tile T的贡献时, CTA_0很可能还在忙于计算它自己的后续任务(比如Tile T+k). 只有当CTA_0完成了它所有分配的迭代后, 它才会回头去真正地执行Wait和LoadPartials操作来完成Tile T的写回. 到那个时候, CTA_1的结果早就准备好了. 因此, Wait操作的实际等待时间非常短, 甚至为零. 这个由于任务启动时间不同而自然产生的"时间差"巧妙地隐藏了同步延迟.

### 2.4 Hybrid Stream-K

在论文中的第五章还展开了一下, 主要有两个核心主题:

Kernel配置: 如何选择最优的Tile Size和Grid Size.

Hybrid方案: 如何结合Stream-K和传统Data-parallel的优点.

Tile尺寸选择原则与硬件特性强相关, Stream-K解放了Tile尺寸选择. 因为不再需要用小Tile来做负载均衡, 所以可以大胆选择计算效率最高的Tile尺寸. 作者选择的策略是"能达到99%峰值性能的最小尺寸", 这是一个很好的权衡: 尺寸够大, 足以发挥硬件性能和隐藏延迟; 尺寸又不过分大, 以免在小矩阵问题上产生过多的浪费(padding).这是Stream-K工程优势的体现. 每个精度只需要确定一个"黄金尺寸", 而不是维护一个庞大的多尺寸列表.

而网格尺寸选择启发式算法基于一个简单的分析模型, 该模型在均分每个CTA的MAC循环迭代的同时, 最小化读, 写和累加部分和的成本.

基础Stream-K分解在某些情况下会表现出Tile处理倾斜(tile-processing skew), 这可能对缓存性能产生不利影响. 当输出Tile数 t 不是网格尺寸 g 的整数倍时, 每个CTA中第一个MAC循环迭代的起始k轴偏移量将会不同. 根据输入矩阵和分块因子的大小和形状, 这种倾斜可能会阻止这些数据片段在GPU的缓存结构中被跨CTA复用. 例如, 在图3(a)中, 四个CTA的初始k轴片段偏移量将分别为k=0, k=32, k=64和k=96. 此外, CTA之间这32个元素的倾斜将在整个GEMM计算期间持续存在.

![图片](assets/a22ca2085433.png)

我们可以采取措施来限制其持续时间, 方法是将Stream-K的迭代均衡应用于总迭代域中一个更小的, 与Tile对齐的区域, 以便剩余的Tile能够以完整的, 时间上对齐的波次来生产. 最简单的混合方案是图3(b)所示的"数据并行 + 单Tile Stream-K"调度. 不幸的是, 当多个CTA覆盖同一个Tile时, 这个策略隐藏同步延迟的能力很弱.

图3(c)所示的"双Tile Stream-K + 数据并行"混合调度来解决这些问题. 它少执行一波完整的数据并行, 以换取每个Stream-K CTA接收超过一个Tile但少于两个Tile的迭代工作量. 当总波数  时, 这提供了好得多的延迟隐藏, 并且每个负责累加的CTA将只需要从另外一个贡献CTA接收部分和. 而大部分工作(12个Tile)是数据并行的, 缓存友好. 只有一小部分(9个Tile)是Stream-K的, "倾斜"的持续时间被限制了.

## 3. Persistent Kernel

接下来我们结合CuteDSL中的Example[4]来进行详细分析.

### 3.1 Host函数

具体功能上和普通的Dense GEMM基本一致, 主要用于准备kernel所需要的参数

```
@cute.kerneldef kernel(    self,    tma_atom_a: cute.CopyAtom,      # A矩阵的TMA加载原子操作    mA_mkl: cute.Tensor,            # A矩阵的全局内存视图    tma_atom_b: cute.CopyAtom,      # B矩阵的TMA加载原子操作    mB_nkl: cute.Tensor,            # B矩阵的全局内存视图    tma_atom_c: cute.CopyAtom,      # C矩阵的TMA存储原子操作    mC_mnl: cute.Tensor,            # C矩阵的全局内存视图    tiled_mma: cute.TiledMma,       # WGMMA计算的配置对象    cta_layout_mnk: cute.Layout,    # Cluster内部CTA的布局    a_smem_layout_staged: cute.ComposedLayout, # A在SMEM中的分阶段布局    b_smem_layout_staged: cute.ComposedLayout, # B在SMEM中的分阶段布局    epi_smem_layout_staged: cute.ComposedLayout, # C在SMEM中的分阶段布局    tile_sched_params: utils.PersistentTileSchedulerParams, # 持久化调度器参数):
```

避免Register Spill, 对于较大的Tile Size依旧需要设置atom_layout_mnk. 另外使用了 cute.arch.warpgroup_reg_dealloc()为TMA Warp和MMA Warp配置不同的寄存器数量.

构造Tiled_MMA和Tiled_TMA的方法是相同的. 不同的是对于Epilogue Tile也使用了TMA.

smem_layout中, 由于EpiTile也需要存入SMEM, 因此增加了一个epi_smem_layout_staged

同样SharedStorage结构体也增加了EpiTile SMEM的分配.

主要的变化在计算所需要的grid和Tile调度抽象

```
    tile_sched_params, grid = self._compute_grid(        c,        self.tile_shape_mnk,        self.cluster_shape_mn,        self.swizzle_size,        self.raster_along_m,        max_active_clusters,    )            @staticmethod    def _compute_grid(        c: cute.Tensor,        tile_shape_mnk: tuple[int, int, int],        cluster_shape_mn: tuple[int, int],        swizzle_size: int,        raster_along_m: bool,        max_active_clusters: cutlass.Constexpr,    ) -> tuple[int, int, int]:            #根据C的Shape和tile_shape计算需要的CTA Shape        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))        gc = cute.zipped_divide(c, tiler=c_shape)        num_ctas_mnl = gc[(0, (None, None, None))].shape        cluster_shape_mnl = (*cluster_shape_mn, 1)                #Tile调度相关的参数        tile_sched_params = utils.PersistentTileSchedulerParams(            num_ctas_mnl,            cluster_shape_mnl,            swizzle_size,            raster_along_m,        )                #Tile调度抽象        grid = utils.StaticPersistentTileScheduler.get_grid_shape(            tile_sched_params, max_active_clusters        )        return tile_sched_params, grid
```

#### 3.1.1 Tile调度抽象

在cuteDSL utils中static_persistent_tile_scheduler.py[5]定义了一个Static Persistent Tile调度器.

代码主要由三个类构成: `WorkTileInfo`, `PersistentTileSchedulerParams`, 和 `StaticPersistentTileScheduler`. 它们分别定义了: 工作单元信息, 调度器配置参数, 以及调度器本身的行为逻辑. 其中`PersistentTileSchedulerParams`用于在Host端,  而`StaticPersistentTileScheduler`用于在Kernel内对Tile进行调度.
WorkTileInfo类
这是一个简单的数据结构, 用于封装单个WorkTile的信息.

`tile_idx: cute.Coord`: 一个 cute::Coord 对象, 代表了分块在整个问题空间中的多维坐标

`is_valid_tile: Boolean`: 检查调度器返回的最新Tile是否有效. 在所有任务完成后, 任何后续的调度请求都将返回一个无效的Tile.
PersistentTileSchedulerParams
它主要的功能是将计算任务的逻辑描述(例如,我需要计算100个128x256的矩阵乘法Tile) 映射到实际的物理执行配置(例如, 在GPU上启动 8x4x32 的CTA网格). 并且在映射过程中构造swizzle优化L2Cache命中率.  主要初始化参数如下:

**problem_shape_ntile_mnl:** 类型为cute.Shape, 定义了整个问题在 M, N, L维度上分别有多少个CTA来处理Tile

**cluster_shape_mnk:** 类型为cute.Shape, 定义了CGA的Shape, 即一个CGA分别在M, N, K维度包含了多少个CTA. 代码中仅支持Cluster在 M,N维度上组织 CTA, 而当 K!=1时会抛出异常.

**Swizzle_size:** 对CGA的layout进行交错排布, 来提高L2 Cache的命中率

**raster_along_m:** 在Swizzle_size > 1时生效, 值为 True 时代表主要沿着 M 维度进行光栅化处理.

raster_along_m: "沿M轴光栅化", 相当于*行主序 (Row-Major)*. 它会先遍历完一行中的所有 N 维度tile, 再换到下一行.

raster_along_n: "沿N轴光栅化", 相当于*列主序 (Column-Major)*. 它会先遍历完一列中的所有 M 维度tile, 再换到下一列.

具体的流程如下:

当无需Swizzle时, 即 swizzle_size == 1. 此时按照如下方式计算每个维度需要多少个CGA来覆盖整个问题空间, 即在 L(batchsize)维度保持不变, 仅对 M, N 维度按照 CGA Shape 来做ceil_div

```
        self.problem_layout_ncluster_mnl = cute.make_layout(            cute.ceil_div(                self.problem_shape_ntile_mnl, cluster_shape_mnk[:2], loc=loc, ip=ip            ),            loc=loc,            ip=ip,        )
```

当需要实现Swizzle优化时, 即swizzle_size > 1时, 此时有两种光栅化选择, 我们以 `raster_along_m=True` 为例进行分析, 此时则需要对 N 维度进行操作处理. 首先我们按照另一个维度(即 N )向上取整(round_up)到swizzle_size的倍数.

```
            problem_shape_ncluster_mnl = cute.round_up(                self.problem_layout_ncluster_mnl.shape,                (1, swizzle_size, 1) if raster_along_m else (swizzle_size, 1, 1),            )
```

然后在make_layout的时候, 在 N 维度上拆分为(swizzle_size, problem_shape_cluster_N // swizzle_size)

```
            if raster_along_m:                self.problem_layout_ncluster_mnl = cute.make_layout(                    (                        problem_shape_ncluster_mnl[0],                        (swizzle_size, problem_shape_ncluster_mnl[1] // swizzle_size),                        problem_shape_ncluster_mnl[2],                    ),                    stride=(                        swizzle_size,                        (1, swizzle_size * problem_shape_ncluster_mnl[0]),                        problem_shape_ncluster_mnl[0] * problem_shape_ncluster_mnl[1],                    ),                    loc=loc,                    ip=ip,                )
```

最后还提供了一个get_grid_shape方法, 用于launch kernel的时候, 计算网格维度(grid_dim).  输入参数为`max_active_clusters`, 这个值通常由设备属性(SM数量)和CGA Shape共同决定.例如H20有78个SM, CGA Shape为(2,1),此时max_active_clusters = 78/(2x1) = 39. 代码如下所示

```
hardware_info = cutlass.utils.HardwareInfo()cluster_shape_mn = (2,1)max_active_clusters = hardware_info.get_max_active_clusters(        cluster_shape_mn[0] * cluster_shape_mn[1])
```

然后具体的计算逻辑如下, 首先根据problem_layout_ncluster_mnl shape和CGA shape计算完成整个问题需要的CTA总数`num_ctas_in_problem`, 然后通过max_active_clusters和每个cluster内CTA的数量之积计算在一个wave中, GPU能同时运行的最大CTA数量, 即`num_ctas_per_wave`.

然后对`num_ctas_in_problem`和`num_ctas_per_wave`进行处理.

如果问题规模很小, `num_ctas_in_problem` < `num_ctas_per_wave`, 那么只启动问题所需的CTA数量, 避免资源浪费.

如果问题规模很大, `num_ctas_in_problem` > `num_ctas_per_wave`, 那么只启动硬件能高效容纳的最大数量的CTA (`num_ctas_per_wave`). 这些CTA会"持久"运行多个wave循环处理所有工作.

最后返回值`(*self.cluster_shape_mn, num_persistent_clusters)`很巧妙:

gridDim.x = cluster_shape_m

gridDim.y = cluster_shape_n

gridDim.z = num_persistent_clusters

`blockIdx.x` 和 `blockIdx.y` 在Kernel中用于定位一个CTA在它所属的2D Cluster中的位置, 而 `blockIdx.z` 则作为这个Cluster在整个持久化网格中的唯一ID, 直接用作静态调度的起始工作索引.
StaticPersistentTileScheduler
这个类是调度器的主体, 在内核执行期间, 为每个 CTA 计算当前应该处理哪个工作分块(Tile), 并在完成工作后推进到下一个分块. 它的内部成员变量如下:

变量

类型

描述

params

PersistentTileSchedulerParams

从主机端传入的配置参数.

num_persistent_clusters

Int32

启动的持久化集群总数, 即 gridDim.z.

_current_work_linear_idx

Int32

当前 CTA 所在CGA正在处理的Cluster的线性索引. 
初始值为 blockIdx.z.

cta_id_in_cluster

cute.Coord

当前 CTA 在其CGA内部的 2D 坐标. 
计算方式为 (blockIdx.x, blockIdx.y).

_num_tiles_executed

Int32

记录当前 CTA 已处理的Tile数量.

create(params, block_idx, grid_dim)
这是调度器的工厂方法, 它接收来自Host端传入的 `PersistentTileSchedulerParams`配置参数,  然后根据CUDA 内置的变量 `blockIdx` 和 `gridDim` 来初始化每个 CTA 的调度器状态.

**num_persistent_clusters** 直接从 grid_dim 计算得出.

**current_work_linear_idx** 初始化为 blockIdx.z, 每个 z 索引代表一个独立的持久化工作流.

**cta_id_in_cluster** 通过 blockIdx.x 和 blockIdx.y 得到, 标识了 CTA 在一个 Cluster 内的身份.
_get_current_work_for_linear_idx(...)
这是一个私有方法, 将返回一个WorkTileInfo对象.  首先检查当前的linear_idx是否超过了整个问题的边界. 然后根据params中的problem_layout_ncluster_mnl恢复成计算任务的多维(m,n,l)逻辑cluster坐标. 最后, 根据集群坐标和 CTA 在集群内的 ID, 计算出最终的分块坐标.
get_current_work()  和 initial_work_tile_info()
公开接口, 调用私有方法 `_get_current_work_for_linear_idx` 来获取当前应处理的分块信息.
advance_to_next_work()
更新状态以处理下一个工作单元. 这是静态调度的方法, 每个集群在完成自己的任务后, 都向前跳跃 `gridDim.z` (即 `num_persistent_clusters`) 个单位, 去领取下一个任务.

```
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):        self._current_work_linear_idx += Int32(advance_count) * Int32(            self.num_persistent_clusters        )        self._num_tiles_executed += Int32(1)
```

例如一个CGA只包含一个CTA, 调度顺序如下所示:

![图片](assets/810fd2419e62.png)

正如上图中右下的示例所示, 静态调度虽然非常简单有效, 但也有缺点, 例如某个SM被其它Grid占用后, 会导致长尾并消耗多个Wave完成计算.

### 3.2 Kernel函数

#### 3.2.1 初始化

首先还是获取线程ID和Warp ID, 并在Warp 0 预取TMA描述符

```
        tidx, _, _ = cute.arch.thread_idx()        warp_idx = cute.arch.warp_idx()        warp_idx = cute.arch.make_warp_uniform(warp_idx)        # Prefetch Tma desc        if warp_idx == 0:            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
```

然后计算CTA在CGA中的rank以及cluster_coord_mnk. 并且根据Cluster的Shape设置TMA组播属性, 并计算TMA拷贝所需要的tx-bytes大小

```
        cta_rank_in_cluster = cute.arch.make_warp_uniform(            cute.arch.block_idx_in_cluster()        )        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)        a_mcast_mask = cute.make_layout_image_mask(            cta_layout_mnk, cluster_coord_mnk, mode=1        )        b_mcast_mask = cute.make_layout_image_mask(            cta_layout_mnk, cluster_coord_mnk, mode=0        )        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))        tma_copy_bytes = cute.size_in_bytes(            self.a_dtype, a_smem_layout        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
```

然后是根据SharedStorage结构体分配内存

```
        # Alloc and init AB full/empty + ACC full mbar (pipeline)        smem = cutlass.utils.SmemAllocator()        storage = smem.allocate(self.shared_storage)        
```

紧接着对SharedStorage内的Mbarrier进行初始化, Producer TMA Warp的arrive_cnt为1, 而Consumer则需要考虑mcast_size / mma_warp_group数量, 最后创建PipelineTmaAsync 对象

```
        # mbar arrays        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()        # Threads/warps participating in this pipeline        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(            pipeline.Agent.Thread        )        # Each warp will constribute to the arrive count with the number of mcast size        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1        consumer_arrive_cnt = (            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group        )        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(            pipeline.Agent.Thread, consumer_arrive_cnt        )        mainloop_pipeline = pipeline.PipelineTmaAsync.create(            barrier_storage=mainloop_pipeline_array_ptr,            num_stages=self.ab_stage,            producer_group=mainloop_pipeline_producer_group,            consumer_group=mainloop_pipeline_consumer_group,            tx_count=tma_copy_bytes,            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),        )        # Cluster arrive after barrier init        if cute.size(self.cluster_shape_mn) > 1:            cute.arch.cluster_arrive_relaxed()
```

接下来, 对需要操作的Tensor进行初始化, 并完成Layout相关的处理和Fragment的分配

```
        # Generate smem tensor A/B        sA = storage.sA.get_tensor(            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner        )        sB = storage.sB.get_tensor(            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner        )        sC = storage.sC.get_tensor(            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner        )        # Local_tile partition global tensors        # (bM, bK, RestM, RestK, RestL)        gA_mkl = cute.local_tile(            mA_mkl,            cute.slice_(self.tile_shape_mnk, (None, 0, None)),            (None, None, None),        )        # (bN, bK, RestN, RestK, RestL)        gB_nkl = cute.local_tile(            mB_nkl,            cute.slice_(self.tile_shape_mnk, (0, None, None)),            (None, None, None),        )        # (bM, bN, RestM, RestN, RestL)        gC_mnl = cute.local_tile(            mC_mnl,            cute.slice_(self.tile_shape_mnk, (None, None, 0)),            (None, None, None),        )        # Partition shared tensor for TMA load A/B        # TMA load A partition_S/D        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)        a_cta_crd = cluster_coord_mnk[1]        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(            tma_atom_a,            a_cta_crd,            a_cta_layout,            cute.group_modes(sA, 0, 2),            cute.group_modes(gA_mkl, 0, 2),        )        # TMA load B partition_S/D        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)        b_cta_crd = cluster_coord_mnk[0]        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(            tma_atom_b,            b_cta_crd,            b_cta_layout,            cute.group_modes(sB, 0, 2),            cute.group_modes(gB_nkl, 0, 2),        )        # Partition global tensor for TiledMMA_A/B/C        warp_group_idx = cute.arch.make_warp_uniform(            tidx // self.num_threads_per_warp_group        )        mma_warp_group_thread_layout = cute.make_layout(            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group        )        thr_mma = tiled_mma.get_slice(            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)        )        # Make fragments        tCsA = thr_mma.partition_A(sA)        tCsB = thr_mma.partition_B(sB)        tCrA = tiled_mma.make_fragment_A(tCsA)        tCrB = tiled_mma.make_fragment_B(tCsB)        tCgC = thr_mma.partition_C(gC_mnl)        acc_shape = tCgC.shape[:3]        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)        k_tile_cnt = cute.size(gA_mkl, mode=[3])
```

然后再整个Cluster同步等待barrier完成初始化

```
        # Cluster wait for barrier init        if cute.size(self.cluster_shape_mn) > 1:            cute.arch.cluster_wait()        else:            cute.arch.sync_threads()
```

#### 3.2.2 Persistent Main Loop

这是Kernel最核心的部分, 实现了持久化调度和计算流水线. DMA和MMA Warp组在这里进入各自的并行循环. 首先根据warp_idx区分不同的warp功能, 并针对DMA warp group设置更少的寄存器数量(40个)

```
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups        if is_dma_warp_group:            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)
```
3.2.2.1 DMA Warp
然后在DMA Warp中, 首先通过StaticPersistentTileScheduler.create实例化调度器, 然后获取work_tile. 并且获取producer_mbarrier_state

```
        if warp_idx == self.load_warp_id:            tile_sched = utils.StaticPersistentTileScheduler.create(                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()            )            work_tile = tile_sched.initial_work_tile_info()            mainloop_producer_state = pipeline.make_pipeline_state(                pipeline.PipelineUserType.Producer, self.ab_stage            )
```

然后整个Producer的循环逻辑如下:

```
if warp_idx == self.load_warp_id:    while work_tile.is_valid_tile:        # ... 获取当前tile的坐标        for k_tile in range(k_tile_cnt):            # 1. 请求一个空的SMEM buffer            mainloop_pipeline.producer_acquire(mainloop_producer_state)            # ... 准备TMA加载的源(GMEM)和目标(SMEM)地址                        # 2. 异步TMA加载A和B, 并将完成事件绑定到mbarrier            cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=...)            cute.copy(tma_atom_b, tBgB_k, tBsB_pipe, tma_bar_ptr=...)                        # 3. 提交加载操作            mainloop_pipeline.producer_commit(mainloop_producer_state)            mainloop_producer_state.advance()                # 4. 处理下一个tile        tile_sched.advance_to_next_work()        work_tile = tile_sched.get_current_work()    mainloop_pipeline.producer_tail(...)
```
3.2.2.2 MMA Warp
对于MMA Warp 首先也是设置寄存器数量为232, 并且初始化StaticPersistentTileScheduler和 consumer_read/release Mbarrier

```
        if not is_dma_warp_group:            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)            tile_sched = utils.StaticPersistentTileScheduler.create(                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()            )            work_tile = tile_sched.initial_work_tile_info()            mainloop_consumer_read_state = pipeline.make_pipeline_state(                pipeline.PipelineUserType.Consumer, self.ab_stage            )            mainloop_consumer_release_state = pipeline.make_pipeline_state(                pipeline.PipelineUserType.Consumer, self.ab_stage            )
```

然后MMA Warp针对Epilogue 初始化从RMEM拷贝到SMEM st_matrix op 以及从SMEM拷贝GMEM的TMA op, 并对Layout进行处理, 同时分配对应的Fragment, 并且初始化TMAStore 所需要的Mbarrier.

```
            # Partition for epilogue            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(                self.c_layout,                elem_ty_d=self.c_dtype,                elem_ty_acc=self.acc_dtype,            )            copy_atom_C = cute.make_copy_atom(                cute.nvgpu.warp.StMatrix8x8x16bOp(                    self.c_layout.is_m_major_c(),                    4,                ),                self.c_dtype,            )            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)            tiled_copy_r2s = cute.make_tiled_copy_S(                copy_atom_r2s,                tiled_copy_C_Atom,            )            # (R2S, R2S_M, R2S_N, PIPE_D)            thr_copy_r2s = tiled_copy_r2s.get_slice(                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group            )            # (t)hread-partition for (r)egister to (s)mem copy (tRS_)            tRS_sD = thr_copy_r2s.partition_D(sC)            # (R2S, R2S_M, R2S_N)            tRS_rAcc = tiled_copy_r2s.retile(accumulators)            # Allocate D registers.            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))            tRS_rD_layout = cute.make_layout(rD_shape[:3])            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)            size_tRS_rD = cute.size(tRS_rD)            k_pipe_mmas = 1            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)            # prologue_mma_cnt指在主循环开始时, 先执行几轮只有`consumer_wait`和`gemm`的循环            # 目的是填满整个流水线. 只有当流水线满了之后, 才进入"等待-计算-释放"的稳态循环.            # Initialize tma store pipeline            tma_store_producer_group = pipeline.CooperativeGroup(                pipeline.Agent.Thread,                self.num_mma_threads,            )            tma_store_pipeline = pipeline.PipelineTmaStore.create(                num_stages=self.epi_stage,                producer_group=tma_store_producer_group,            )
```

然后也是通过一个循环, 从调度器获取work tile进行处理

```
if not is_dma_warp_group:    while work_tile.is_valid_tile:        # ... 初始化累加器为0        accumulators.fill(0.0)                # Prologue        for k_tile in range(prologue_mma_cnt):            # 1. 等待TMA完成            mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)                        # 2. 执行WGMMA            for k_block_idx in cutlass.range_constexpr(num_k_blocks):                cute.gemm(tiled_mma, accumulators, tCrA[...], tCrB[...], accumulators)                            # 3. 等待WGMMA计算完成            cute.nvgpu.warpgroup.wait_group()                         # 4. 仅advance read_state            mainloop_consumer_read_state.advance()                    # K维度上的计算循环        for k_tile in range(prologue_mma_cnt, k_tile_cnt):            # 1. 等待SMEM buffer被填满 (等待TMA加载完成)            mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)                        # 2. 执行WGMMA计算            for k_block_idx in cutlass.range_constexpr(num_k_blocks):                cute.gemm(tiled_mma, accumulators, tCrA[...], tCrB[...], accumulators)            cute.nvgpu.warpgroup.commit_group() # 提交一批WGMMA指令                        # 3. 等待WGMMA计算完成            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)                         # 4. 释放已使用的SMEM buffer            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)            # ... advance state ...        # ... 等待所有剩余的WGMMA完成        cute.nvgpu.warpgroup.wait_group(0)        # Epilogue: 写回结果        # ... (详细见下一节)        # 处理下一个tile        tile_sched.advance_to_next_work()        work_tile = tile_sched.get_current_work()
```

#### 3.2.3 Epilogue

Epilogue也在MMA Warp的循环内, 当一个(M,N) tile的所有K维度计算完成后, MMA Warp组进入Epilogue阶段, 将累加器中的结果写回全局内存.

```
# (在MMA Warp组的 while 循环内部, K维度循环之后)# Epilogue# ... (为TMA Store准备分区和布局)# 1. 遍历Epilogue Tile (将大C tile切成更小的块处理)for epi_idx in cutlass.range_constexpr(epi_tile_num):    # 2. 从累加器寄存器拷贝到另一组寄存器 (R2R)    for epi_v in cutlass.range_constexpr(size_tRS_rD):        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]    # 3. 类型转换 (如FP32 -> FP16)    acc_vec = tRS_rD.load()    tRS_rD_out.store(acc_vec.to(self.c_dtype))    # 4. 从寄存器拷贝到SMEM (R2S)    cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[...])    # 5. 同步, 确保所有MMA线程都完成了R2S    self.epilog_sync_barrier.arrive_and_wait()        # 6. fence    cute.arch.fence_proxy(        cute.arch.ProxyKind.async_shared,        space=cute.arch.SharedSpace.shared_cta,    )        # 7. 从SMEM拷贝到GMEM (S2G), 由一个专门的Warp执行    if warp_idx == self.epi_store_warp_id:        cute.copy(tma_atom_c, bSG_sD[...], bSG_gD[...])        # ... (TMA Store流水线操作)    # 8. 再次同步, 确保S2G的TMA发起后再进入下一轮epi_idx循环    self.epilog_sync_barrier.arrive_and_wait()
```

参考资料

[1] 
CUTLASS Tutorial: Persistent Kernels and Stream-K: *https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/*
[2] 
Matrix Multiplication Background User's Guide: *https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html*
[3] 
Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU: *https://arxiv.org/pdf/2301.03598*
[4] 
hopper dense gemm persistent: *https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm_persistent.py*
[5] 
static_persistent_tile_scheduler.py: *https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/static_persistent_tile_scheduler.py*