# Inside Nvidia GPU: 谈谈Blackwell的不足并预测一下Rubin的微架构

> 作者: zartbot  
> 日期: 2025年11月4日 23:38  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&chksm=f995e4e6cee26df07bf7101b58cbdfdf80d577c67122304482e3e788edfa74a71135dbf77d36#rd

---

### TL;DR

花了几周的时间, 在CuteDSL上把Hopper和Blackwell一些GEMM运算梳理了一下. 看到了从Ampere到Hopper再到Blackwell的一些演进, 正好前一周周末参加华为的图灵计算峰会和廖博以及其它一些Ascend的同学聊了一下. 然后接着又有GTC老黄的Keynote现场展示了一下Vera Rubin的工程开发板以及BlueField4等.. 因此准备做一个全面的分析和新一代的微架构的预测(反正是猜的, 猜错了不要怪我...)

其实在渣B看来, Nvidia最大的壁垒并不是简单的一句话两句话能说清楚的. CUDA生态, 或者SIMT一类的争议很多. 但它真正的壁垒恰恰是在整个体系结构中的很多Dirty work处理干净了产生的, 再加上从算法到系统再到芯片全栈的能力. 这也是给国产芯片很多启发的地方, 特别是很多细微之处, 平衡好了易用性/可编程性 vs 性能.  另一个就是整个体系结构推出到市场的时机和营销都做的很好.  正所谓“领先一步是先驱, 领先半步才是神”

当然每一种架构都有它的取舍和不足, Nv也不是神, 接下来也会谈到Nv的很多问题, 例如Blackwell, Grace以及新发布的BlueField4等... 然后假设渣B是一个Vera Rubin的架构师, 会去做些什么来谈谈未来可演进的地方.

## 1. 谈谈从Volta到Blackwell的演进

从Volta开始引入Tensor Core, 其实Nvidia的传统定义的SIMT架构就已经开始被破坏了. 而真正完成整个架构的迁移, 或许要到Rubin这一代. 整个过程耗时十年, 既是硬件上的逐步迭代, 又是软件上的逐步创新.

### 1.1 TensorCore

从硬件上来看, 从最早的FMA指令, 到向量化的DP4A,再到Volta(SM70)的第一代TensorCore,然后Ampere/Hopper/Blackwell都在提高矩阵乘法的规模, 提高计算访存比,同时支持更低精度的数据格式.

![图片](assets/f391de11fbe2.png)

从数值精度的变化来看, 如下所示, 伴随着芯片面积的约束, 在Blackwell Ultra(B300)这一代已经在开始砍掉高精度计算的算力了.

Arch

FP64

FP16

INT8

INT4

FP8

MXFP

Volta

❌

✅ FP16

❌

❌

❌

❌

Turing

❌

✅ FP16

✅

✅

❌

❌

Ampere

✅

✅ FP16/BF16

✅

✅

❌

❌

Hopper

✅

✅ FP16/BF16

✅

❌

⚠️FP8/FP22

❌

Blackwell

✅

✅ FP16/BF16

✅

❌

✅

✅ MXFP(8/6/4)
NVFP4

Blackwell Ultra

⚠️砍算力

✅ FP16/BF16

⚠️砍算力

❌

✅

✅ MXFP(8/6/4)
NVFP4

预计Rubin这一代还会进一步翻倍TensorCore的规模, 预计256 x N x 256bits的规模, 另一方面猜测会出现进一步扩大Blackwell 2-CTA MMA到Rubin 4-CTA共同参与的MMA指令. 但是对于CGA内的调度还有更进一步的需求.

算力的提升带来的另一方面的问题就是供数路径的变化. 早期(Volta)的TensorCore开始复用CUDA Core寄存器. 然后伴随着Ampere TensorCore规模扩大, 考虑到寄存器的压力, 使用了cp.async bypass L1和RMEM的占用. 然后到Hopper引入TMA, 并且可以将操作数直接放入SMEM, 并且引入CGA和DSMEM, 但此时Accumulator的结果还在RMEM中, 便于后续的Epilogue操作, 但是它还需要采用waitgroup的barrier机制. 再到Blackwell引入TMEM, 使得整个TensorCore和CUDA Core之间基本上分离了, 同时也复用了TMA异步操作引入的Mbarrier机制. 如下图所示:

Arch

Matrix A

MatrixB

MatrixD

Volta
RFRFRF
Ampere
RFRFRF
Hopper
RF/SMEMSMEMRF
Blackwell
TMEM/SMEMSMEMTMEM

整个过程差不多耗时10年, 从Volta开始像是一个临时添加的TensorCore组件,再到Blackwell引入TMEM不依赖RMEM基本上完全异步化的分离, 每一步都做的挺稳的.

详细内容可以参考下面两个专题

[《Tensor》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3557619493198151684&scene=173&subscene=&sessionid=svr_32119fe6ccb&enterid=1722676230&from_msgid=2247491424&from_itemidx=1&count=3&nolastread=1#wechat_redirect)

[《GPU架构演化史》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=2538479717163761664&scene=173&from_msgid=2247487954&from_itemidx=3&count=3&nolastread=1#wechat_redirect)

### 1.2 异步化处理

另一方面是异步化的处理, 当Volta这一代为每个Thread引入独立的PC, 其实就标志着异步化的开始.

![图片](assets/eba135051f10.png)

就此Thread可以开始等到一些message进行异步处理, 相对于传统的对齐PC的架构打开了异步化编程的窗口

![图片](assets/71196df1952e.png)

比较好的地方是, Nv在软件上提供了Cooperative Group的抽象. 但是TensorCore还是需要整个Warp上同步的执行. 然后Ampere中引入cp.async开始, 实际上整个程序的供数路径已经就出现异步化了, 也就是Nv提到的Async Thread的概念.

![图片](assets/8c7add9833a7.png)

到了Hopper更进一步引入了MBarrier, 软件上围绕着MBarrier构建的异步流水线和Warp Specialization变得流行起来. 然后引入了Async Proxy, 并通过General Proxy和Async Proxy来区分不同的内存访问路径.对于Async Proxy的操作, 通常有一个memory barrier, general proxy的LD/ST可以wait这个barrier完成, 使得TMA这些异步操作也可以和原来的SIMT访问内存的LD/ST结合起来, 保证内存序的要求.

![图片](assets/3d825ace8050.png)

当然Hopper也有不完善的地方, WGMMA是一个临时的解决方案, 占用大量的RMEM同时又需要同步等待, 因此在Hopper发布的时候就明确告知SM_90a的WGMMA不会向后兼容. 这样有一个很大的缺点:

![图片](assets/3dcdd7488c2b.png)

到了Blackwell, TensorCore也变成了完全异步的操作, 并且也复用了MBarrier构建, 因此issue TMA和tcgen05.mma的指令都可以做到Thread level. 但是TMEM的内存分配和拷贝这些还是需要WarpLevel的处理. 另一方面就是引入了ClusterLaunchControl的机制, 有了一部分动态调度的能力.

![图片](assets/843e945ef6e1.png)

然后我们可以构建更复杂的WarpSpecialization处理模式

![图片](assets/99ae04d93e54.png)

详细内容可以参考

[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect)

### 1.3 CuTe Layout

这也是一个非常棒的软件的抽象, 特别是在Hopper和Blackwell上把Swizzle的复杂度也隐藏了. 另一方面从代数的角度解决了复杂的Tile/Partition边界计算, 使得代码变得更加直观, 当然对于非代数专业的学习CuTe还是会面临一个比较陡峭的学习曲线.  对于CuTe Layout代数下文开了个头

[《CuTe Layout代数-1: Overview》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496154&idx=1&sn=474a5450c46b86169095d84dd3cfd7dc&scene=21#wechat_redirect)

但是您需要注意的是, 在Blackwell dual die的架构, 甚至是Rubin Ultra 4 Die的架构上, 甚至是未来3D-DRAM的架构, 这套代数已经把很多问题简化了. 后面的章节我将详细阐述一下为什么.

当然这部分内容, 我后面几个月有时间了还会进一步更新.

## 2. 谈谈Blackwell的不足

前面说了一些好话, 这一章来说一些不足的地方, 主要目的是祛魅.

### 2.1 B200 SFU的问题

在疯狂的卷TensorCore性能的时候, 添加了大量的TMEM, 同时GPC的一些互联网络构成的DSMEM也占用了很多Die面积, L2取消Partition也会带来Die面积的占用, 因此单个Die上SM的数量降低到了80个. 但是很遗憾的是CUDA Core上配的SFU的性能并没有增强, 导致GEMM运算看上去强了很多, 但在Attention算Softmax的时候出现的瓶颈.

![图片](assets/b61ab6c2f060.png)

当然也有人说, 没啥问题啊, 反正可以用Linear Attention呀. 确实最近Attention的一些变化带来了一些争议. 一方面是Qwen-Next的GDN, Kimi Linear的KDA. 而另一方面minmax M2又放弃了Linear Attn.  另一条路径是Google/DeepMind MoR和传言GPT5中使用的Universal Transformer似乎还在增强Attn block的算力.

[《谈谈Transformer的一些演进: UT,MoD,MoR...》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494744&idx=1&sn=20f307c5e0fe7c5c5d62a46d81f48646&scene=21#wechat_redirect)

而DeepSeek-V3.2的DSA和以往的NSA则走向了Sparse Attn的路.

个人的观点是和DeepSeek一致的, Linear Attn并没有很好的解决内存访问的瓶颈. 计算本身是很容scale的, 而内存访问却是很难的, 因此选择Sparse Attn才是正确的路.

另一方面是前段时间读到的一篇文章, 从最优输运的角度来看SDPA. 即注意力机制的前向计算过程, 即通过Softmax函数生成的注意力权重的过程, 完全等价于一个单边的熵最优输运(One-Sided Entropic Optimal Transport, EOT)问题的精确解. 因此Softmax是无法避免的.

[《大模型时代的数学基础(9)- SDPA和最优传输, 强化学习及信息几何的联系》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494688&idx=1&sn=3d589f6d4be56ee372d5db4f8631b0cc&scene=21#wechat_redirect)

基于这个角度我个人的观点是, SFU的能力还是要跟TensorCore的算力匹配, 好在B300上把这个问题解决了, 代价是砍掉了很多其它的高精度运算的算力. 针对这个问题, 我一直觉得B200和GB200都不是一个很值得投资的平台.

### 2.2 Blackwell复杂的指令结构

其实从Hopper开始, 整个异步编程就变得很复杂了, 而Blackwell引入TMEM其实也增加了更多的复杂度. 例如整个TensorCore的tcgen05指令, 既有同步的, 也有异步的

tcgen05.* operation

同步指令
**.alloc****.dealloc****.relinquish_alloc_permit**

**.fence::*****.wait::*****.commit**

异步指令
**.mma****.cp****.shift****.ld****.st**

另一方面issue指令的粒度(Granularity)也是不同的, 有的是线程粒度的, 有的又是warp粒度的, 还需要考虑2SM的情况.

![图片](assets/0b77cab7a3c8.png)

一些同步没有做好很容易犯错, 不过Nvidia在这里又引入了很多Pipeline的抽象, 避免了不少错误. 并配合TMEM的内存管理分配机制, 通过alloc/dealloc降低了多线程并行情况下对TMEM的管理复杂度.

其实从管理复杂度来看, 下图中的Sch warp/TMA Warp和TC Warp都可以做到单线程的处理, 而只有EpilogueWarp才需要原来SIMT的那套东西, 搞明白了似乎也不复杂, 但是编程的时候总要心里惦记着...好在长时间在搞各种异步编程的事情, 处理起来也没什么太大的难点.

![图片](assets/8e1cfb106d1a.png)

### 2.3 CPU的问题

虽然在Hopper这一代就引入了NVLink C2C, Grace能够和Hopper或者Blackwell直接NVLink连接. 但是Grace的CPU也有不少的问题.  实际上伴随着Blackwell算力越来越强, 很多Kernel的运行时间降低到微秒级别, 这就产生了一个非常经典的 Killer Microsecond问题. 对于ns级别的问题, 同步等待就行了. 对于ms级别的时间, 上下文切换的代价也不大. 但是当到了 us 级别, 其实对于处理器而言已经有很大的挑战了. 虽然引入了很多异步的编程优化, 但是当前Grace这类的CPU还是面临很多瓶颈. 一方面是Kernel Launch的速度不够快, 虽然可以狡辩通过cuda-graph或者一些persistent kernel的方式来解决. 但并不是所有的workload都满足这个条件.

另一方面是一些Grace微架构的缺陷. 虽然Grace使用了ARM当时最强的Neoverse V2的Core, 它的设计上并没有采用V2所使用的2MB L2Cache, 而是裁剪到了1MB. 相比之下,同样使用V2 Core的AWS Graviton 4 采用了2MB L2Cache. 当前有些客户遇到了GB200上Grace L1 ICache Miss的问题很大程度上与这个有关.  我们看到Nvidia在关于Grace的宣传上, 基本上都在谈一些HPC相关的应用...

![图片](assets/1d76d3dbd18f.png)

另一个强调的点是更大的内存带宽和容量的平衡选择了LPDDR5x, 通过NVLink C2C进一步扩展Hopper和Blackwell的访存能力.

然后整个片上网络也是一个Mesh架构, 整个L3的访问延迟需要在NOC上经过多跳, 影响也挺大的.

![图片](assets/269133fb80d5.png)

另一方面是GB200搭配的CX7由于没有内置的PCIe Switch, ScaleOut的RDMA流量都会穿越整个Grace的NOC再通过NVLink C2C到达Blackwell.

![图片](assets/c7511cd44d7b.png)

这样会导致蛮多的问题, 例如这些流量穿越整个Grace NOC的时候, 对于L2 Cache较小的情况下, Cache miss的penalty会进一步增加.  Chips and cheese上有一个测试, Grace的延迟远高于X86, 同时对比Graviton 4 也高了不少, L2 Cache太小了, 而周边noise neighbor和NOC noise的影响太大的缘故.

![图片](assets/8a3abb66736a.png)

顺便吐个槽, 基于Grace的BlueField4 也有同样的问题....NV在这一块的能力真是不行, 另一方面CX8/CX9的设计上有重大的问题...

其实我们看最近Megatron有一个在GB200上训练DeepSeek-V3的报告《Optimizing DeepSeek-V3 Training Performance on NVIDIA GB200 NVL72》[1]也提到了CPU Overhead这个问题.

![图片](assets/3ae9889295a6.png)

AWS额外引入了一颗PCIe Switch解决了ScaleOut RDMA穿越Grace NOC的问题

![图片](assets/8657c6bed368.png)

而Meta则是将Grace和Blackwell的配比做成了1:1来缓解

![图片](assets/9e549a7a2e3c.png)

当然这部分的问题在GB300上解决了一些

![图片](assets/8554640e1384.png)

相比之下, 我们来看Intel GNR, Cache的处理上有SNC3的选择:

![图片](assets/944344a86ed5.png)

当然GNR也有一些NOC相关的问题, 导致内存速度上有影响, 就不展开多说了... 实质的问题是在核心数多到一定层度后, NOC的复杂度和性能影响会影响非常大, 特别是对这些带Cache Coherency的通用CPU. 即便是非Cache Coherecy的处理器, 例如渣B在思科的时候, 2004年做第一代QFP 40 Core 160线程, 到2008年做第二代 56 Core 224线程, 都挺好的. 再到第三代扩展到 224 Core 896线程的QFP3也遇到了大量的问题. 其实通用处理器单socket卷到几百个核后, 都会遇到类似的问题....具体细节就不展开了...

### 2.4 Blackwell memory

还有一个问题是Dual Die的架构, 跨Die的内存访问必然会带来更大的延迟. 这样会导致SM访问GMEM的效率产生问题, 当前CUDA 13.0 还不支持CTA的内存亲和性调度, 不过直觉是通过CuTe Layout可以做一些事情, 把内存bank错开, 当然未来CUDA是否会增加类似的CTA亲和性调度API, 我估计会...以前写过一篇文章

[《英伟达GB200架构解析4: BlackWell多die和Cache一致性相关的分析》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489759&idx=1&sn=2c55ec63d6deaeb39ff7f767896ba853&scene=21#wechat_redirect)

## 3. 预测一下Vera Rubin的架构

### 3.1 Vera CPU

从老黄GTC拿出来的开发板来看, 首先是Vera内存采用了8通道, 相对于Grace内存带宽应该翻倍了. CPU Core的数目增加到了88个, 然后PCIe支持了Gen6估计有80个Lane. Core的规格估计采用了ARM的 Neoverse V3.  V3可以支持最多3MB的L2Cache, 不过真不确定NV会怎么选择. 至少在Jetson Thor上引入的Neoverse V3-AE还是停留在1MB的L2Cache... 另外它也是一个多Die的结构, 把PCIe/Memory Controller放在了不同的Die上, 这个设计和Graviton 3/4一致了.

![图片](assets/c3ffa0d1320c.jpg)

但是CPU的Overhead估计还会存在, 根本的原因如下, 一个CPU Core从L1 Cache读取通常只需要3个cycle, 而当它需要穿越L3时, 又面临一个Mesh的NOC时, 通常需要120个cycle以上.

当然这个问题并不大, 或许有Intel X86的NVlink C2C的CPU出现后会缓解...

### 3.2 Rubin架构推测

其实有一些明牌的东西了, 例如TensorCore的规模还会进一步在M维度扩大一倍, 然后TMEM的容量估计也会增加来应对这个变化. 但是整个芯片面积已经捉襟见肘了, 正是这个面积的约束, Nvidia在Rubin上采用了独立的I/O Die

![图片](assets/36a33e9d6ec0.png)

然后推测局部的会采用4-SM的MMA来进一步复用数据, CGA的cluster shape会到4, 并沿用Blackwell的preference cluster shape的方式来调度打满整个GPC. 

对于SM的微架构, 我从Blackwell上的一些情况来看, 可能在SM内增加1个标量核会取得很多收益, 整个标量核加上Private SMEM 对于Die面积的占用也非常小.

![图片](assets/ef3a72f81046.png)

首先是TMA/MMA的描述符现在基本上都是Host CPU在生成, 虽然这些描述符基本上都是不大变的, 简单prefetch一下也不是不行? 但是Host CPU和Rubin之间虽然有NVLink C2C但是延迟还是很大, 一些控制信息传递也需要数百个cycle了, 因此集成一个标量核直接L1/SMEM上共享也不是不行? 可以进一步的释放RMEM的占用, 例如isuee TMA和UMMA以及调度cluster只需要一个线程即可, 没必要在CUDA Core上运行.  然后ICache也可以省掉不少, 转移很多复杂的控制指令到标量核上.

![图片](assets/11d80d7aa8cf.png)

然后我会为它设计一片很小的Private SMEM(例如2KB~4KB)用于存放MBarrier, 这样异步程序架构就可以更容易的解耦合了, 不需要复杂的WarpSpecialization的处理, 就是一个TC Function, 一个TMA Function, 和原来的CUDA SIMT的kernel function做Epilogue即可. 然后会将Sch warp以及TMA/ TC的描述符准备相关的function放入这个标量核, 更进一步还可以和Warp Scheduler做更进一步的调度交互. 实质上还是类似于Helion/TVM/Tilelang的想法, 采用调度和算法分离的方法. 代码上可能更有利于编译器生成和优化. 

另外针对IBGDA, GIDS(GPU Initial Direct Storage)这些, 部分文件系统的处理逻辑也可以放入到这个标量核内, 降低了通信SM的占用. 

甚至是在这种情况下, 我们可以做一些更复杂的MPMD的编程和尝试. 特别是在Rubin Ultra上, 4个Die拼接时还有很多好玩的并行策略可以去做,

![图片](assets/68cc8ee8233e.jpg)

例如用一些Green CTX配合CTA affinity做一些事情更好的利用片内的结构, 其实Google Pathways[2]这些MPMD的框架早就有了.

![图片](assets/b029710eb02a.png)

然后你会说, 呀这不是和华为的Ascend 910很像了么, 人家也有Scalar的AI CPU, SIMD的Vector Core以及Tensor Core?

![图片](assets/0141c5efd12f.png)

这也是我前文调侃的“领先一步是先驱, 领先半步才是神”. Nvidia的成功很大程度上在于对整个计算生态的缓慢牵引, 让客户一步步的从Volta Tensor开始, 花了整整十年才慢慢走到这一步.  做技术的人很容易用一个以终为始的思维方式去构建. 我也犯过很多类似的错误, 例如2018年在Cisco搞AI Infra, 在思科的网络设备上做一些神经网络的动态控制算法, 做一些边缘人工智能相关的硬件产品研发, 时至今日Cisco才发布边缘AI的产品... 这些都是我在8年前就全部搞完的东西...

![图片](assets/1a090d92e166.png)

当全世界的人认知在那里的时候, 很多超前的东西会变成布鲁诺那样被活活的烧死...例如换个话题RDMA的Lossy和Lossless...全世界都在Lossless的时候, 太超前也会出问题. 直到现在当你不得不要去考虑Scale Across的时候... 看看Google的Fellow怎么说, 他说Falcon是这十年内非常重要的一篇Paper.

![图片](assets/3ffe171083f7.png)

而我们eRDMA在3年前就是这样了...

[《谈谈Google Falcon的可靠传输论文并对比分析CIPU eRDMA》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495848&idx=2&sn=e55764ca731533c76e55ab4cb0bf25d4&scene=21#wechat_redirect)

下面这个图才是正路, 可惜这里面dirty work太多, 魔鬼般的细节, 一般人肯定学不会...

![图片](assets/f63856e77867.png)

不过在市场上被毒打后, 我现在也学乖了, 很多时候很多事情即便看明白了, 也会跟着市场的节奏慢慢来, 根据用户的心智和生态慢慢演进..

回到正题上, 其实GPU的微架构上还是有很多细微的差别的, 例如SIMD vs SIMT的一些争议, Task的调度框架, Memory Barrier的设计, 标量核的架构, 片内NOC的互连等等... 更重要的一个问题是在易用性和性能之间的一个平衡, 一个架构师需要保证的是尽量不让客户因为一些编程上的难度掉入一个性能雪崩的case里.

其实这里面有大量的dirty work, 一方面你需要对运行在上面的算法有足够深度的了解, 例如前文谈到的从Optimal Transport的角度来看SDPA中softmax就是一个最优解, 然后从计算效率的角度来看, Memory是很难Scale的, 因此不会去选择Linear Attn的路线, 更多的去选择Sparse Attn. 明白这个道理后, 你就不会去砍SFU的算力, 而难得Nvidia在B200上犯了个错, 所幸在B300上救回来了.

作为一个芯片架构师, 更多的是要去预测未来3~5年的workload, 当然这很难但人家NV有全栈的能力, 这就是人家的壁垒. 而我们或许只能靠“人工”智能, 前段时间开完云栖大会, 我还在调侃我自己

![图片](assets/332e79119203.png)

另外是半年前的一个预测, 有些已经提前发生了, 过两年我们再来回顾吧

[《民科: 预测一下未来五年大模型的架构?》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494117&idx=1&sn=a0f3f66faff51d407b6f52c02e2577c8&scene=21#wechat_redirect)

当然“人工”的代价是很大的, 兜兜转转花了差不多二十年的时间, 从算法到芯片基本上轮了一圈倒是有了很大的收获. 从芯片来看, 从网络做到计算, 互连这一块无论是协议设计还是芯片实现上, 我们都是非常领先的. 还记得几年前NV来跟我们说BlueField-4的Roadmap, RDMA相关的路标我们早就实现了. 下一代芯片碾压CX10也没什么问题, 大家请放心.

对于ScaleUP上, 基本上全球也找不出几个在这个领域的专家比我懂的多, 各种Trade-off早就分析清楚了.

[《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect)

算子这一块最近基本上也很快的把坑填平了, 虽然很多年前就在开始写CUDA, 最近买了一个Thor把Blackwell的微架构以及编程完全弄清楚了.可能欠缺的还有一些框架层的代码, 下个月训个小一点的模型补一下应该问题也不多了.

至于算法上从很多年前打OI比赛然后搞量化算法, 在思科自己训练模型做一些分布式强化学习相关的最优控制, 图算法来分析设备异常和加速分布式数据库搜索也搞了很多年了. 至于数学上或许也不算差, 毕竟还读了好几年的数学系学了几十门数学课, 基本上在2014年就看清楚了方向, 代数这一块还在努力的学习中...

[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

## 4. 一些建议

这是说给很多国产卡听的, 忠言逆耳别认为我是在攻击国内的厂商....

其实很多国内的芯片设计(以往主要是和华为数通的同学打交道)还有欠缺的地方. 以前数通产品的集采测试中, 我总能找到几个客户要用到的场景但是很容易陷入性能雪崩的case, 不过我这种人还是挺好的, 测试赢了也会直接给华为的同学反馈他们的不足, 其实这几年来看华为的数通产品基本上已经非常成熟了.

前两周也在华为的图灵技术峰会上和廖博以及Ascend的一些同学聊过这个事情, 很多时候魔鬼都藏在细节里, 例如Nvidia的SM互连特别是CGA内的实现是怎么做的, MBarrier为什么要那样设计, Async Proxy是怎么设计的, 它对异步内存访问的软件简化有什么帮助, 软件抽象上怎么包住这些复杂性的, CuTe Layout为什么要这么抽象? 甚至是在一些易用性的小细节里, 例如一个很小的细节在tcgen05中, 为什么他们会在pipeline.consumer.release的时候包住tcgen05.commit, 以及为什么要在TMEM上实现col based alloc/dealloc?  当然这些设计本身搞明白了以后是很简单的, 背后涉及到的取舍有很多, 易用性这一点不光是一个简单的生态问题, 更多的是知其然并知其所以然的体会.

这些事情没有捷径可走, 必须要踏踏实实的把每个细节做好. 所谓的弯道超车大概率是弯道翻车...其实在很多局部的领域我们已经开始领先了, 大家一起努力一步步脚踏实地的做吧. “中国人怎么不行啊？外国人能搞的，难道中国人不能搞？中国人比他们矮一截?“

参考资料

[1] 
Optimizing DeepSeek-V3 Training Performance on NVIDIA GB200 NVL72: *https://github.com/NVIDIA/Megatron-LM/blob/dev/docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-optimization.md*
[2] 
Pathways: *https://proceedings.mlsys.org/paper_files/paper/2022/file/37385144cac01dff38247ab11c119e3c-Paper.pdf*