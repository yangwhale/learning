# 现代NVidia GPU架构

> 作者: zartbot  
> 日期: 2025年5月5日 02:39  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494136&idx=1&sn=958bc734efd43a59b03c04a2e65ec408&chksm=f995f93acee2702c39ae5cb992a5ed00ecf05077968d03a98e6feea142667b974e6650e1e9e5#rd

---

来自于今年三月的一篇论文《Analyzing Modern NVIDIA GPU cores》[1]和今年一月的一篇《Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis》[2], 趁着假期读了一下.

### 1. 早期的NV GPU架构

学术界通常研究的GPGPU架构通常依赖于GPGPU-Sim模拟器采用的微架构, 如下图所示:

![图片](assets/ddeb4b5b849d.png)

在GPU流水线中执行取指时, 通常采用一个RR轮询调度器选择一个warp, 而每个Warp有独立的指令缓冲区, 当下一条指令在L1 I-Cache中并且这个指令在warp指令缓冲区存在空槽时, 将其执行取指译码操作后放入缓冲区内. 在指令发射阶段, 采用一个GTO(Greedy To Oldest)的调度器选择一个warp, 这个warp不能存在一个等待的barrier, 并且它最老的指令与流水线中的其它in-flight指令没有数据依赖.以往的假设是存在两个记分牌来检查数据依赖. 然后是一个供数组件, RF有多个bank, 每个bank也有多个port, 同时还有一个Collector Unit, 当指令的所有源操作数都在CU内后, 转移到指令派发阶段. 指令会被派发到对应的EU. 最后结果会写回到寄存器.

这个微架构基于2006年发布的Tesla架构, 然后在Accel-Sim中增加了少量现代特性.  但是还缺乏很多现代的GPU的组件, 例如L0 I-Cache, Uniform RF, issue logic / RF / RF Cache等多个组件的变化也没有包含其中, 因此《Analyzing Modern NVIDIA GPU cores》通过一些逆向工程, 进行了分析.

### 2. 现代的NV GPU架构中控制位

现代的NV GPU架构中, 在ISA上包含由编译器控制的控制位(Ctonrol Bits)信息, 以保证程序的正确性. 它不会像以前的GPU那样在运行时追踪寄存器读写的数据依赖, 而是通过编译器来处理寄存器的数据依赖, 因此所有的汇编指令都包含了一些control bits用于恰当的管理数据依赖并提高性能和降低能耗.

SubCore能够每个cycle发射一条指令, 默认情况下, 发射调度器在相同warp最老的一条指令准备好时发射. 编译器采用控制位指示指令是否准备好发射.

#### 2.1 Stall Counter

每个warp有一个Stall counter,用于处理固定延迟的Producer-Consumer指令依赖, 然后预先计算好延迟并在编译器中将Stall Counter设置, 然后执行时每个Cycle控制该计数器减1, 直到为零表示Ready.

#### 2.2 Yield

用于指示硬件, 下一个cycle调度器无需发射相同warp的指令.

#### 2.3 Dependence Counter

对于访问内存和SFU这些指令执行时间是变长的, 编译期无法确定执行时间, 因此使用Stall Counter是有风险的, 然后在每个Warp定义了6个特殊的寄存器SBx, 每个计数器取值最高为63. 这些寄存器初始值为0, 为了处理Producer-Consumer依赖, Producer发射指令后增加特定寄存器, 而消费者指令在这个寄存器为零前被阻塞. 同时为了避免WAR风险, 在其源操作数获取后减少寄存器的值, 而不是在写回阶段减少.

在指令设计上, 有一些控制位指示在指令发射时最多增加两个寄存器Counter.  一个其中一个Counter在写回时减少用于处理RAW和WAW依赖, 另一个在寄存器读取时减少, 用于处理WAR依赖. 因此每条指令都有2个域, 每个域3bits来指示哪两个寄存器Counter. 同时每条指令都有一个6bits的掩码

考虑到一个指令可能有多个源操作数, Producer具有可变延迟, 筒一个Dependence Counter可以被所有这些生产者共用, 而不会丧失并行性. 但是需要注意的是, 该Counter累计只有6个, 当存在超过6条具有不同可变延迟的Producer指令时, 并行性可能会受到影响. 因此编译器需要在这里进行一些优化, 例如将更多的指令归入同一个Dependence Counter, 或者以不同的方式重排指令顺序.

另一个细节是, 当递增Dependence Counter后需要在下一个Cycle才能生效, 因此cunsumer的Stall counter被设置为2, 避免指令在下一个cycle被发射产生错误.

下图为一个可变延迟的Producer实例, 这段代码由3个LD和一个加法构成.

![图片](assets/6c187b77185e.png)

PC 0x80的加法指令在0x50/0x60之后存在RAW依赖, 因此发射0x50和0x60时会增加SB3计数器, 并在写回时减少. 另一方面0x60和0x70指令存在WAR依赖, 因此在发射时增加SB0寄存器的值, 并在读取各自寄存器的源操作数后减少. 最后加法的依赖计数器Mask指示SB0和SB3在发射前必须为0. 另外0x70指令还使用SB4来控制与未来指令的RAW/WAR风险.

另一种做法是使用DEPBAR.LE指令, 例如, DEPBAR.LE SB1, 0x3, {4,3,2} 要求依赖计数器 SB1 的值小于或等于 3 才能继续执行, 最后一个参数（[, {4,3,2}]）是可选的, 如果使用, 表示指导这些ID制定的依赖计数器的值等于零才发射. DEPBAR.LE还有一些很有用的用途, 例如有N个变长延迟的指令需要在写回时保序, 但是另一个consumer指令期望等待前M个执行完就可以被发射, 因此可以使用DEPBAR.LE N-M来执行. 另一个用途是复用同一个Counter来保护RAW/WAW和WAR风险. 如果一条指令对两种类型都依赖于相同的一个计数器, 由于WAR比RAW/WAW更早解决, 可以使用DEPBAR.LE SBx 0x1来等待直到WAR风险解决后, 允许warp继续执行. 稍后需要消费其结果的指令等到计数器为零后继续执行.

类似的AMD也有waitcnt指令, 在每个wavefront有三个或者四个计数器, 每个计数器用于特定类型的指令.

#### 2.4 寄存器缓存

此外, GPU还拥有一个RF缓存,用于节省功耗并减少读取RF的端口争用. 这种结构通过为每个源操作数添加一个Reuse控制位并通过软件进行管理, 该位指示硬件是否缓存寄存器中的内容.

### 3. GPU Core微架构

现代GPU Core的微架构如下图所示:

![图片](assets/275cdbff0295.png)

#### 3.1 发射调度器

Warp准备就绪条件:

指令缓冲区中有有效的指令

warp的最老的指令不能与同一个warp中尚未完成的较旧的指令存在任何数据依赖的风险, 指令之间的数据依赖通过控制位处理.

对于固定延迟的指令, 一个warp只有在可以保证一旦发射就能获得执行所需的所有资源时.才会被认定为在给定的cycle内发射其最老指令的候选者. 这些资源包括执行单元(EU), EU有一个输入锁存器, 如果EU的宽度为半个warp, 则该锁存器占用两个Cycle, 如果宽度是一个完整的warp,则占用一个周期.

在源操作数中有常量缓存的指令, 在发射时需要进行tag查询. 当指令的一个操作数在常量缓存而其它操作数没有在Cache中时, 调度器并不发射任何指令. 在等待4个cyecle这些缺失的操作数还未到达时, 调度器将切换到另外一个Warp.

另外关于寄存器文件中读取端口的可用性,发射调度器并不知道评估的指令是否有足够的端口在接下来的周期中无需停顿即可进行读取. 作者假设了一个模型, 在这个模型中固定延迟指令在发射阶段和读取源操作数阶段之间有两个中间阶段:

第一阶段为控制阶段, 适用于固定和可变延迟指令, 其职责是增加依赖计数器或在需要时读取时钟计数器的值. 通过实验证明, 这导致一条增加依赖计数器的指令和等待该依赖计数器归0的指令之间至少需要一个周期来使该增加可见, 因此两条连续的指令不能使用依赖计数器来避免数据依赖危害, 必须第一条指令设置Yield bit或者大1的Stall Counter.

第二阶段仅存在于固定延迟指令中. 在这个阶段,将检查寄存器文件读取端口的可用性,并且在保证指令可以在没有任何寄存器文件端口冲突的情况下继续进行之前, 指令将在此阶段停滞. 称这个阶段为Allocate.

可变延迟指令（例如, 内存指令）在经过控制阶段后直接被送入队列（不经过Allocate阶段）. 当保证这些指令不会有任何冲突时,允许它们进入寄存器文件读取流水线. 固定延迟指令在分配寄存器文件端口时被赋予优先权. 因为如上所述, 依赖关系由软件处理,它们需要在发射后固定的周期数内完成, 以保证代码的正确性.

调度策略:

为了了解发射调度器的策略, 作者测试了一些case, 在多个warp中, 记录每个cycle调度器选择哪个warp的信息. 这些信息时通过允许保存GPU当前CLOCK周期数的指令收集的. 由于硬件不允许连续发射两个指令, 因此在两者之间使用了控制数量的其它指令, 通常采用了NOP, 并且在测试中还改变了Yield和Stall Counter控制位的值. 在测试中得出一个结论:

Warp调度器使用一种贪心策略, 如果满足前述Warp准备就绪条件, 就从同一个Warp中选择指令. 当转换到其它Warp时, 则选择满足前述条件的最年轻的Warp. 一些例子如下图所示:

![图片](assets/eca22c60da94.png)

该图描述了在同一SubCore中执行四个 warp 时, 三种不同情况下的指令发射情况.每个 warp 执行由 32 个独立指令组成的相同代码, 每个周期可以发射一个指令.

在第一种情况(a)下,所有的Stall计数器/依赖掩码和Yield位都被设置为零, 调度器从最年轻的 warp,即 W3 开始发射指令,直到在 Icache 中出现缺失. 由于缺失, W3 没有有效的指令,因此调度器转而从 W2 发射指令.由于 W2 重用了 W3 带来的指令,它在 Icache 中命中,并且当它到达 W3 出现缺失的地方时, 缺失已经得到服务,所有剩余的指令都在 Icache 中找到,所以调度器贪婪地发射该 warp 直到结束. 之后, 调度器继续从 W3(最年轻的 warp)发射指令直到结束, 因为现在所有指令都在 Icache 中. 然后, 调度器转而从 W1 发射指令, 最后,它对 W0(最老的 warp)做同样的事情.

第二种情况(b)显示了每个warp的第二个指令将其Stall Counter设置为4时, 指令发射的时间线, 我们可以观察到调度器执行玩两个周期后从W3切换到W2, 再过两个周期切换到W1, 然后再过2个周期又切换回W3(此时W3的Stall计数器已经归零), 一旦W2,W3, W1完成后, 调度器开始从W0发射指令, 并且在W0第二个指令后,产生了一个4Cycle的空泡.

第三种情况(c), 每个warp的第二个指令设置Yield位, 我们可以看到, 在发射每个warp的第二个指令后, 调度器切换到其余Warp中最年轻的一个, 例如W3切换为W2, W2切换回W3.

另外作者还测试了一个场景, 其中设置了Yield, 并且没有更多可用的warp, 可以观察到调度器产生了一个Cycle的空泡. 作者将这种调度方式成为编译器引导的最年轻贪婪算法Compiler Guided Greedy Then Youngest (CGGTY). 编译器通过Stall Counter和Yield以及Dependence Counter来协助调度. 但是这种分析只测试了同一个CTA内部的warp行为, 目前尚未设计出一种可靠的方法来分析不同CTA的Warp间的交互.

#### 3.2 前端

根据NV的多个图表和文件来看, SM拥有4个不同的Sub-Core, warp轮询的均匀分布在Sub-Core之间, 即warp-ID%4. 每个Sub-Core有私有的L0指令缓存连接到L1Cache, L1Cache在所有四个SubCore内共享. 我们假设有一个仲裁器来处理不同SubCore的多个请求. 每个L0-ICache都有一个指令Prefetcher, 但是NV没有公开其具体设计, 作者怀疑是一个简单的Stream Buffer, 在发生缺失时, 预取连续的内存块, 缓冲区的大小为16. 但无法用实验确定指令fetch策略,但它必须要和Issue策略类似, 否则回经常发生在指令缓冲区内无法找到有效指令的情况.作者假设每个SubCore在每个Cycle可以获取并解码一个指令.

Fetch调度器试图从前一个Cycle(或者最近一个指令发射的Cycle)发射的同一个Warp中获取指令, 除非它检测到指令缓冲区已有的指令数加上在途Fetch的数量等于指令缓冲大小. 在这种情况下, 它将切换到其它指令缓冲区中有空闲条目的最年轻的Warp.

#### 3.3 RF

现代NV GPU拥有多种类型的寄存器:

`Regular`: 最近的架构每个SM拥有65536个32位寄存器, 用于存储线程操作的值. 这些寄存器按32个一组排列, 每组对英语一个warp中的32个线程. 共计2048个warp寄存器. 这些寄存器在subCore之间均匀分布, 每个SubCore中的寄存器被分为2-Bank. 特定的Warp时用的寄存器数量可以从1~256不等, 这在编译时决定, 每个Warp使用的寄存器越多, 则SM中并行运行的Warp更少.

`Uniform`: 每个Warp有64个私有的32位寄存器,用于存储所有线程共享的值

`Predicate`: 谓词寄存器, 用于分支, 每个warp有8个32位谓词寄存器, 每个bit由warp中的不同线程使用..

`Unifomr Predicate`: 统一的位次寄存器, 每个warp有8个1-bit寄存器, 存储所有线程在warp中共享的predicate.

`SB Register`: 每个warp 6个, 用于Dependence Counter

`B寄存器`: 每个 warp 至少有 16 个 B 寄存器, 用于管理控制流重新汇聚

`Special`: 用于存储特殊值的寄存器, 例如Thread-ID/Block-ID等.

作者通过实验证明了传统的操作数收集器不见了, 因为操作数收集器会引入从发射到回写之间时间的变化性, 使得NVIDIA ISA无法拥有固定延迟的指令. 作者通过一些实验确认了操作数收集器的缺失, 并观察到, 无论RF的端口冲突数量如何, Stall计数器和执行指令经过的时间都是保持不变的.

当一个LD和一个固定延迟指令在同一个cycle结束时, LD指令将会延迟一个cycle.另一方面当两个固定延迟指令之间存在冲突时, 它们都不会被延迟, 这意味着使用了结果队列来处理固定延迟指令, 这些指令的Consumer不会被延迟, 这意味着使用了旁路转发, 在将结果写入RF之前转发给了Consumer.

同时实验还显示, 每个RF的bank都有一个专用的1024 bits的写端口, 关于读取, 作者观察到每个Bank的位宽为1024bits. 但是作者无法通过逆向工程找到一个且当的读取策略适用于所有的案例. 观察到的空泡生成取决于指令类型和指令中的每个操作数的任务. 作者推测了一种近似的几乎符合测试例的方法. 即在固定延迟指令发射和操作数读取之间存在一个Control/Allocate两阶段方案. Allocate负责保留RF中的读端口, 并通过RF Cache来缓解读冲突.
3.3.1 RF Cache(RFC)
RFC由编译器控制, 仅由具有Regular Register File操作的指令使用, 经过实验显示, 它在每个SubCore的两个RF Bank中各有一个Entry, 每个Entry可以保存3个 1024-bits的值, 每个值对应指令中可能具有三个Regular寄存器操作数中的一个. 总体来看RFC容量为6个1024bits的操作数值. 注意, 有些指令需要两个连续寄存器的操作数(例如TensorCore), 这种情况下, 两个寄存器分别来自于不同的bank,并缓存在相应的Entry中, 编译器管理分配策略. 当一个指令被发射读取操作数时, 如果编译器为该操作数设置了Reuse bit, 则操作数将存储在RFC中.

#### 3.4 Memory Pipeline

现代NV GPU的内存流水线在每个SubCore都存在一些初始阶段的处理, 而在执行内存访问的最后阶段由四个SubCore共享, 因为L1DCache和SMEM是SM内共享的.

每个SubCore有独立的LD/ST队列, 为了发现队列大小和内存带宽, 作者做了一系列试验. 其中每个SubCore要么执行一个warp,要么处于空闲状态. 每个Warp执行一系列独立的LD/ST操作, 这些操作总是命中DCache或者SMEM, 并使用Regular寄存器. 下表显示了实验结果:

![图片](assets/07fbd5bfe937.png)

第一列显示指令编号, 接下来四列代表不同的活动Subcore数量时, 指令issue的cycle. 可以观察到在Ampere中, 每个SubCore可以在5个连续的内存指令中,每个周期发射一条指令, 在第六个内存指令发射时, 由于不同的Active SubCore的影响, 会导致若干个Cycle的延迟. 可以推断, 每个SubCore可以在不延迟的情况下最多缓冲五条连续的指令. 而从全局结构来看, 每2个cycle可以从任何一个subcore接收一个内存请求. 还可以推断出, 每个Subcore中进行地址计算的吞吐量为每个四个Cycle一条指令.

对于每个SubCore中的内存队列大小, 尽管每个SubCore可以缓冲5条连续的指令, 考虑指令在到达时Reserve Slot, 离开时Release Slot, 作者的估计队列大小为4.

从延迟来看, 作者在缓冲命中且执行单个线程时, 为每种指令测量了两种不同的延迟, 如下表所示:

![图片](assets/8bc66f73709b.png)

RAW/WAW延迟: 从发出LD指令时刻T1到可以发出消费该数据或者覆盖相同目标寄存器的指令的最早时刻T2之间经过的时间(T2-T1).

WAR延迟: 从发出LD/ST指令时刻T1, 到可以发出在LD/ST的源寄存器中写入指令的最早时刻T2之间经过的时间(T2-T1).

可以看到, 如果指令在加载GMEM使用Uniform寄存器,而不是采用Regular寄存器来计算它们的地址, 则内存访问会更快, 这种差异是因为地址计算更快(Uniform寄存器在warp内所有线程共享只计算一个, 而Regular每个线程需要计算一个可能不同的内存地址).

WAR延迟在Uniform和Regular寄存器时相同的, 而RAW/WAW, Uniform寄存器少一个Cycle. WAR延迟对于Uniform和Regular相同可以表明, 共享内存的地址计算是在共享结构中完成的, 而不是在每个SubCore内完成, 所以一旦读取源寄存器, WAR依赖就会被释放.

延迟也取决于读写值的大小, 对于WAR依赖, LD延迟不会改变, 因为源操作数仅用于地址计算. 而对于ST指令, 写入内存的值也需要从RF中读取的源操作数, WAR延迟会随着写入值的大小而增加. 对于RAW/WAW依赖(仅适用于LD), 随着读取值的加大也会增大, 因为从内存到RF传输了更多的数据. 作者测量这种传输的带宽为每个Cycle 512bits.

另外作者还观察到, 常量缓存的 WAR 延迟明显大于对全局内存的加载, 而 RAW/WAW 延迟则略低, 并且发现通过固定延迟指令完成的对常量内存的访问与LD常量指令使用的是不同的缓存级别. 访问常量地址空间的固定延迟指令使用 L0 FL（固定延迟）常量缓存, 而通过 LDC 指令进行的访问则使用 L0 VL（可变延迟）常量缓存.

最后还测量了LDGSTS指令, 直接从GMEM加载到SMEM并且bypass RF.

### 4. Hopper内存子系统

#### 4.1 延迟

采用传统的P-Chase benchmark测试的延迟:

![图片](assets/65365e21389a.png)

![图片](assets/5eecd8185152.png)

TMA延迟包括TMA指令issue, Mbarrier事务初始化和Mbarrier等待的过程, 可以观察到和常规内存访问相比, TMA的拐点更少, 这表明TMA操作受L2缓存影响, 另一方面TMA延迟比常规内存访问高170个cycle, 可以归因为TMA单元的开销和同步等待的时间.

另外针对Ampere和Hopper L2 分成两块, 因此对L2 Partition进行了延迟 测试

![图片](assets/ab056d916595.png)

#### 4.2 带宽

测试时, 对于L1缓存的访问, 采用了ld.ca加载到L1中, 并且采用了一个包含1024线程的block来重复访问SMEM.对于L2带宽, 采用ld.cg加载到L2, 使用的block数量为SM的两倍, 并用于加载GMEM. 为了减少内存指令, 采用float4读取, 每个线程读5次写1次, 时用的block数量为SM的4倍.

![图片](assets/531dfccfa6bf.png)

值得关注的是Hopper的L2Cache带宽比Ampere翻了一倍, 并且这些测试结果超过了平台理论峰值带宽的90%.

### 5. TMA

TMA的延迟测试前述章节已经有了, TMA的吞吐值得关注一下. 作者测试了无需Tensor描述符(Non-Tensor)和1D/2D/3D Tensor, 从GMEM加载4GB内存到SMEM中. block size选择了114/228/342/456对应于SM数量的1,2,3,4倍.  在Non-Tensor中评估了1KB/2KB/4KB/8KB/12KB/16KB加载, 而对于1D, TensorMap限制SMEM每个dim最多256个元素, 即便是使用int64类型, 最大的加载Size只有2KB, 因此测试了 0.5KB、0.75KB、1KB、1.25KB、1.5KB 和 2KB, 而在2D Tensor测试中, 使用FP32测试形状为 16×16、32×16、32×32、64×32、96×32、64×64的加载, 在3D Tensor中测试FP32  8×8×4、8×8×8、16×8×8、16×16×8、16×16×12、16×16×16加载的情况, 结果如下

![图片](assets/3910bfb4a782.png)

结果可以看到, 只有在LD Size大于2KB的时候, TMA才会有效的利用内存带宽. 另一方面TMA计算地址的组件有一些瓶颈, 例如在3D情况下, Block数量为SM数量4倍时, 吞吐率还有明显的下降.  另外, 通常需要放置Block数量超过SM数量2倍时才能打满内存带宽, 隐藏访问延迟.

另一方面作者对加载16KB数据采用不同形状的case也进行了测试, 试的配置是 16×16×16、64×64×1、256×16×1、16×256×1 和 4×4×256. 结果显示在下图中, 揭示了一个明显的趋势：在相同的加载大小下, 较大的 x 轴尺寸提供了更高的吞吐量, 而增加的线程块数量没有带来负面影响. 然而, 增加 y 轴和 z 轴尺寸显著降低了吞吐量, 这可能随着更多线程块的增加而进一步减少. 因此, 当使用 TMA 处理维数大于 3D 的张量时, 选择合适的参数以实现最佳性能非常重要.

### 6. DSMEM

直接访问SMEM大概要29个Cycle, 当采用DSMEM方式访问block local SMEM时, 即便不通过SM-to-SM网络, 延迟也要增加4个cycle. 而将clusterSize加到2, 即需要通过SM-to-SM访问DSMEM时, 延迟增加到181 Cycle, 但是还是比通过L2 Dcache少了32%.

DSMEM吞吐测试需要区分SM内和跨SM的情况, SM内的理论带宽为1755 MHz ×128 Bytes = 225 GB/s. 通过DSMEM接口访问block内的内存只有205GB/s,大概是理论带宽的80%, 而不使用DSMEM可以达到99.8%. 可以看到DSMEM接口访问还是有很大的overhead的.

对于跨SM访问, 采用了三种常见的模式

![图片](assets/253f9aa10c4a.png)

另外Hopper还提供多种调度模式:default, Spread, and LoadBalancing, 在Default和Spread模式下, 同一个cluster的块被分配到GPC内不同的SM上, 在LoadBalance模式下, Block可以被分配到相同的SM上, 测试带宽如下:

![图片](assets/454a536e5b14.png)

在测试结果中显示LB模式在大多数情况下优于其它两种测量, 可能是因为同SM的block可以不使用SM-to-SM网络访问内存所致. 另一方面随着cluster size的变化, 不同的访问模式出现了显著的吞吐率变化. 对于Ring和Pair模式差异较小, 而Bcast则出现了明显的下降.

另一方面作者通过调整clustersize, block大小和ILP来探索了各种参数对DSMEM吞吐的影响. 如图所示:

![图片](assets/c58862d87eb5.png)

当blocksize=64太小时, 未能充分利用带宽, 增加ILP可以提高SM-to-SM的利用率, 当blocksize足够大时, ILP并不提高DSMEM性能. 在clustersize=2时观察到的峰值为3.28TB/s, 随着clustersize增加到4, 带宽减少到2.78TB/s. 另外随着cluster中blocksize增加, 更多的block会争抢SM-to-SM带宽, 降低整体吞吐.

参考资料

[1] 
Analyzing Modern NVIDIA GPU cores: *https://arxiv.org/pdf/2503.20481v1*
[2] 
Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis: *https://arxiv.org/pdf/2501.12084*