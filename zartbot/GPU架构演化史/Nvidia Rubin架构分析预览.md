# Nvidia Rubin架构分析预览

> 作者: zartbot  
> 日期: 2026年7月23日 00:20  
> 原文: https://mp.weixin.qq.com/s/mrSaS-MzHgN9CjZnm7J7ig

---

**TL;DR**

Nvidia最近公布了一些Rubin架构相关的资料《Inside NVIDIA Rubin GPU Architecture: Powering the Era of Agentic AI》[1] 然后对应的PTX 9.4指令集也有一个Preview的文档《PTX 9.4》[2]. 因此做一些简单的分析. 从微架构编号来看, Rubin基本上延续了blackwell SM100 family的特征, 例如TensorCore依旧是tcgen05的架构. 但是在很多细节设计的地方做了很多提升.

回到去年的一篇文章《Inside Nvidia GPU: 谈谈Blackwell的不足并预测一下Rubin的微架构》, 这是一篇对Blackwell架构的批评的文章, 老黄还在公司内部转发过. 在Rubin发布后, 基本上一些细节上的不足都做了很好的补充. 例如SFU的性能, 新的Thread Block Level Dependecy Launch等,我将详细的进行一些分析.

为了方便外国人阅读, 弄了一个英文版: https://github.com/zartbot/blog/issues/11, 老黄这次你转不转? 这次没骂人在夸赞NV...

## 1. 芯片架构Overview

### 1.1 SM数量变化

Blackwell系列每颗芯片有80个SM, 累计为160个SM. 但是在B200时代, 因为SFU的瓶颈问题带来Attention/MoE Activation的计算过程中Epilogue过慢引起, 因此在B300的时代, 通过砍掉一些高精度的算力增加了2xSFU性能. 但是还有一些情况下SFU的算力还是不足的, 例如MoE中的Activation函数计算....

我们回顾一下Blackwell的芯片layout, 可以看到实际上空间已经很紧张了, 在B300上只能通过砍掉一些功能增加算力.

![图片](assets/fec9419f6b06.png)

而在Rubin时代, SM数量增加了50%, 它是怎么进一步从160个SM扩展到224个的呢? 做了哪些取舍呢?

![图片](assets/29321e21b812.png)

很明显的一个变化就是, 将PCIe/NVLink/NVLinkC2C构成了一些独立的I/O Die 通过NV-HBI合封, 使得主计算Die有了更大的面积用于放置更多的SM, I/O Die的移出估计能够提供额外20%左右的芯片面积用于放置SM.

![图片](assets/a4cdb44c9bff.png)

但是考虑到SFU的数量增长, 个人怀疑是不是还有一些其它的面积上的Trade-off, 例如单个SM内部的一些实现变化, L2Cache Size调整, 这个要等我玩上Rubin的机器实际测试以后才能知道.

### 1.2 TensorCore性能

**整体性能对比如下:**

| 项目 | sm_103 Blackwell Ultra | sm_107 Rubin |
| --- | --- | --- |
| SM | 160 | 224 |
| Tensor Core | 640 | 896 |
| 每 SM Tensor Core | 4 | 4 |
| dense NVFP4 | 15 PFLOPS | 35 PFLOPS |
| dense FP8 | 5 PFLOPS | 17.5 PFLOPS |
| HBM 带宽 | 8 TB/s | 22 TB/s |

前一节的介绍可知, Rubin 仅靠 Tensor Core 数量就有 896/640 = 1.4×. 但是官方整卡 dense NVFP4 峰值约为 35/15 = 2.33×, dense FP8 约为 17.5/5 = 3.5×. 我们注意到官方的blog显示, Rubin通过将Tensor Core 在 K 维度上可处理的数据量翻倍, 使每个时钟周期的 Tensor Core 吞吐量翻倍.

![图片](assets/f4b225dcd865.png)

![图片](assets/7f4037e0a4d5.png)

排除1.4x的SM数量带来的算力提升, 我们可以看到单个TensorCore的性能提升为NVFP4 2.33/1.4 = 1.66x, FP8 3.5/1.4 = 2.5x. 那么再进一步排除K维度2x, 可以推测出整个芯片的频率似乎提升了25%, 另一个佐证是BF16的K维度没变, 性能从B300的2.25PFLOPs 提升到了Rubin 4PFLOPs. 折算1.4x SM数量提升4/2.25/1.4 = 1.27x, 基本可以确定频率提升25%左右.

但是值得注意的是NVFP4的提升相对较少主要是B300 SM_103已经有dense K=96, 在Rubin上增加到128, K维度相对提升1.3x, 再加上SM数量和频率提升, 数据是能对齐的.

**实际 kernel 能提升多少?**

实际任务中的性能提升, 我们可能还需要考虑到其它子系统相关的约束, 例如 GEMM 已经是 HBM、通信或 epilogue bound, TMA/shared-memory pipeline 供数不足, 或者 K 很小的一些情况... 例如 K_total=160：K=64 需要覆盖到 192, 利用率 83.3%, K=128 需要覆盖到 256, 利用率只有 62.5%.

### 1.3 SFU性能

在去年的关于Blackwell架构分析文档中阐述了SFU性能的瓶颈, 即便是在B300系列中2x的SFU性能, 实际上还是在一些场景中会出现瓶颈, 例如MoE中Expert的SwiGLU激活计算时, 导致Epilogue无法很好的overlap GEMM的情况. 在Rubin中进一步翻倍了SFU的性能:

![图片](assets/94a21dd463af.png)

实质原因是, 在文章《大模型时代的数学基础(9)- SDPA和最优传输, 强化学习及信息几何的联系》 中讲过Attention的计算在Optimal Transport的视角下SDPA是最优的, 因此很大程度上Softmax的运算是无法避免的, MoE的Expert SwiGLU Activation倒是有办法做一些替代, 因此去年在和NV的一些交流中我的态度就是SFU一定不能砍. 而在Rubin这一代中, SFU的性能继续翻倍正好也匹配了TensorCore的性能, 但估计实际上还是会在Epilogue阶段有一些空泡出现.

### 1.4 Attention Sparse处理

因为SFU的一些瓶颈, Nvidia在blog中给出了一个新的做法. 虽然Sparse的GEMM过去几年一直在提, 但是基本上没啥用处... 这一次 Rubin 通过将激活稀疏 (activation sparsity) 与自适应压缩和改进的 softmax 吞吐相结合来加速注意力. Rubin 官方文章给出的 attention 路径非常具体, 简单来说, 就是在Attention计算中   还是以Dense方式执行计算, 生成中间注意力分数. 随后 Rubin 可以将这些中间数据从 Tensor Memory 加载为结构化 2:4 稀疏压缩形式, 同时生成非零值与高效利用这些值所需的元数据, 从而降低分数的写回成本与存储需求. 这使得后续注意力阶段可以在更少的数据上运算, 同时保持模型其余部分所期望的稠密输出格式.

![图片](assets/99ed86f564c9.png)

结合新的PTX 9.4指令集的文档, 我们来详细展开这个过程:

第一段GEMM QK 还是以 Dense 方式执行, 然后在 TMEM 中 Accumulator 保持 Dense 的 Score 矩阵. 然后使用tcgen05.ld.red.spcompress指令, 压缩逻辑检查全部 dense score 进行稀疏加载. 这种方式直接减少的是:

TMEM load 输出到普通寄存器的数据量.
后续保留值的 exponent 和 normalization 工作量.
完成 staging 后的 sparse P x V 乘加数量.

**tcgen05.ld.red.spcompress 到底做了什么**

该指令固定处理 f32,使用 b2 metadata,执行 2:4 选择,并支持 .x4 到 .x128.普通版本输出 cdata 和 mdata;.red 版本额外输出 redval.

这样来看 nonzero 加载到寄存器的数量会减半, 但寄存器 footprint 不是严格 50%. 短 vector 中 metadata 摊销更明显,长 vector 又会增加寄存器峰值和 live range..x128 不一定优于 .x32.

然后我们注意到.red 能在同一次 TMEM load 中返回局部 min 或 max. 对 softmax, 最自然的组合是 rowop::max:每 4 个 logits 保留数值最大的 2 个, 同时生成 fragment 的局部最大值.

但需要注意它仍然不是一条完整 softmax 指令.软件还必须完成:

跨 fragment,warp 或 CTA 合并 row max.
对保留值执行 ex2.approx((x - max) * log2(e)).
求 row sum 和 reciprocal.
在 online softmax 中维护跨 tile running max 和 running sum.
把 probability 转换并重排为 sparse MMA 可以消费的格式.

但是另一个问题值得我们注意, ld.spcompress 把 f32 cdata 和 b2 mdata 放进普通寄存器.而 tcgen05.mma.sp 要求 sparse A 来自 shared-memory descriptor 或 TMEM, metadata operand 也是 TMEM 地址.因此中间存在一个不能忽略的 staging gap:

```
f32 logits
  -> sparse softmax probability
  -> dtype conversion
  -> MMA A-layout repack
  -> metadata layout repack
  -> shared/TMEM store
  -> completion wait
  -> tcgen05.mma.sp
```

只有减少的 exponent,normalization 和 sparse PV 时间大于 conversion/staging 成本时, 端到端才会加速. 另一方面, 我们需要注意, 当 logits 很尖锐时,每组 top-2 可能已经覆盖绝大部分概率质量.当 logits 很平坦时,删掉 50% 元素会显著改变分布.因此性能测试必须同时报告Attention的精度影响.

**结论**

其实这里我们不能简单的说, 用了这种算法就能提高2x的Attention的性能, 可以看到Sparse在CUDA Core内也增加了一些运算量, 但很大程度上降低了SFU的瓶颈. 但最终收益有多大, 还需要取决于Attention的精度影响, 是否可用我觉得还是持怀疑态度的.

## 2. 软件功能

对比分析了一下PTX 9.3 和 PTX 9.4中关于Rubin SM_107增加的部分.

![图片](assets/5496a8e32aec.png)

![图片](assets/e199d6c7afef.png)

### 2.1 Kernel调度

在去年的文章中其实也分析到了Blackwell的调度会有很多问题, GPU 需要高效地从一个 kernel 过渡到下一个 kernel. 在推理中这一点尤为重要, 因为激活往往位于关键路径上: 一个 kernel 产生激活数据、写入内存, 下一个 kernel 消费这些数据以继续生成下一个 token.

传统的生产者-消费者执行模式会在 GPU 时间线上产生气泡. 生产者 kernel 可能提前完成部分 tile 或线程块的工作, 但消费者可能要等到更大范围的依赖关系解除后才能开始有效工作. Blackwell 的可编程依赖启动(programmatic dependent launch,PDL) 通过允许消费者 kernel 更早推进改善了这一点, 但依赖工作仍可能因等待所需激活数据就绪而停滞.

![图片](assets/a918309779b1.png)

这一点很好的解决了我对Blackwell架构缺陷的烦恼, 实际上可能这种方式会使得我们不需要再去特别的实现一些MegaKernel的框架了, 直接使用这种依赖即可.

**具体软件实现**

从PTX 9.4的文档中, 我并没有找到关于Grid Scheduling 或者 Tile publication相关的指令, 因此Nvidia可能这个功能的实现还是以暴露数据依赖, 而不暴露完整的调度器实现的. 因此我个人的一些猜测是后面一个做为Consumer的Kernel在调度器层面允许它同时Launch, 但仅在SM内占用少量的寄存器和ICache资源. 然后在Producer的那个Kernel有一个ThreadBlock的跨Kernel在SMEM内或者GMEM内共享的flag/barrier. 当前一个Thread Block完成时, 使用st.release语义释放这个barrier. 然后对于Consumer Kernel中的ThreadBlock可以先加载一小段Poll循环代码在ICache内, 就是一个很简单的polling, 通过ld.acquire读取这个flag/barrier, 如果没有ready那么就很简单的做一个nanosleep. 直到完成后, 一个分支跳转出Poll这个循环.

这是一个纯软件的实现方式, 应该在Blackwell上也可以做. 当然比较理想的情况是在硬件调度器上做这件事, 可以复用一些mbarrier的硬件能力, 避免Consumer这个ThreadBlock的polling

### 2.2 MoE相关的优化

#### 2.2.1 TMA

![图片](assets/ea60c55f7f12.png)

实际上对应的指令集应该是TMA copy中的override::global_address,.override_attribute指令, 这是因为MoE expert tensor 常有相同 layout,不同 base address.传统路径为每个 expert 维护一份 descriptor,或者在软件侧频繁 patch descriptor.Rubin 可以保留一份共享 tensorMap,在 TMA 指令运行时覆盖:

global base address.
global dimensions.
global strides.

router 只需给出当前 expert 的运行时字段,TMA 仍负责 tensor addressing,copy 和 mbarrier completion. 收益包括更少 descriptor storage, 更少 patch/proxy fence,更低 metadata 流量,以及更适合 persistent kernel 的动态专家调度.

#### 2.2.2 Applypriority

这是一个关于Cache生命周期管理的用法, 通常对于MoE Expert参数这类需要多个token反复读取的热点数据, 我们可以将其Cache Hint设置为evict_last, 然后在多个计算阶段中反复使用增加Cache命中率. 当结束时, 调用applypriority将其Cache Hint策略改为evict_normal. 在Rubin中增加了这个功能

![图片](assets/ab913ede1f92.png)

**大致的作用如下:**

![图片](assets/04819c0d46ec.png)

例如route先选择了Expert-3进行处理, 此时我们将Expert 3的权重使用evict_last策略, 它将同时处理多个token. 当路由切换到Expert 7的时候, 我们且Expert 3完成 Last-use, 此时我们可以设置applypriority(Expert 3 range)将其改为evict_normal然后让 Expert 7 获得更多有效 L2 capacity.

其实这种策略不光对于MoE Expert Weight, 对于多阶段的GEMM 或者 Fused Kernel, 甚至是Sliding-winddow Attention 或者 KV 的生命周期管理都是非常有用的. 一个常见的用法是 applypriority.async.bulk.tensor 使用 tensorMap 和 tensor coordinates 计算整个 tensor tile footprint. 典型组合是:

```
一份共享 tensorMap
    -> TMA runtime override 选择当前 expert base
    -> TMA 加载 expert tile
    -> Tensor Core 重复使用
    -> tensor applypriority 将相同 footprint 恢复为 normal
```

### 2.3 ScaleUP通信

当推理从单 GPU 扩展到整机架系统时, 通信成为关键性能路径的一部分. 当通信被直接融合进 GPU kernel 内部时, kernel 无需停下并把控制权交还 CPU; 它可以在计算仍在进行时, 直接通过 NVLink 向另一块 GPU 写数据或执行归约. 传统的 GPU 间通信除了搬运有效载荷数据外, 还需要协调与同步工作. 这些步骤会增加延迟并消耗互连带宽, 尤其是在分布式推理负载中通信频繁发生的情况下.

Rubin 为设备发起的 NVLink 通信引入了计数写入 (counted writes). 该能力让接收方 GPU 能更高效地跟踪传输完成情况, 从而简化 GPU 间数据传输的同步.

![图片](assets/839c0dadba5c.png)

通信对比图: Blackwell 使用内存屏障, 确认与原子标志, Rubin 在接收 GPU 加载数据前使用计数器更新

采用计数写入的 NVLink 融合通信为协调 GPU 间数据搬运提供了更低延迟的机制, 帮助计算持续推进, 而不是等待同步操作完成.

实际上在ScaleOut网络上, 我们大概在2023年就实现了这样的功能, 通过接收端更新counter来节省一个RTT. 也不是啥很难的技术, 又一次领先NVidia 3年, 然后具体的实现就是采用PTX指令fabric.try_put.async.counted::bytes ... 和fabric.try_red.async.counted::bytes .... PTX 的 fabric 指令可以由 GPU kernel 发起跨 GPU put 或 reduction.其中 counted form 让 remote counter 按 destination-accessed bytes 累加. receiver 可以轮询 counter,确认目标数据访问已经达到预期字节数,再消费数据.这能减少独立 memory barrier,acknowledgment 和 atomic flag 的协调.

另一个重要事实是:counted fabric syntax 在 PTX 9.3 已经存在,最低 target 是 sm_100+, 也就是说Blackwell也应该能够支持, 不知道啥时候能有卡去测试一下.

### 2.4 Tcgen05 LUT

**把低比特索引变成 Tensor Core 的 B operand**

![图片](assets/6d70e564a059.png)

tcgen05.mma.decompress::lut::b 很容易被名字误导.它不是可以查询任意函数的通用 LUT, 它做的是一种非常具体的 local codebook decompression:

B operand 保存 3-bit index.
每个局部 group 有 8 个 E4M3 候选值.
MMA 在内部用 index 选择候选值并恢复 B value.
B collector 可以让解码后的 B tile 在多次 MMA 中复用.

## 3. 其它问题

### 3.1 峰值功耗管理

AI 负载期间的功率需求可能剧烈波动, 产生的瞬态尖峰会闲置可用容量并降低工厂级吞吐. Vera Rubin 电源采用荷电状态 (SoC) 智能功率平滑技术来吸收这些波动: 相比上一代功率平滑技术, 平均功耗降低约 10%, 50 毫秒峰值功率降低约 20%. 通过提供更平稳的功率曲线, 系统可以降低持续最大功率需求, 惠及配套供电基础设施与电网, 并在相同 AI 工厂功耗预算内容纳更多算力.

在 AI 工厂层面, NVIDIA DSX MaxLPS 将这一方法扩展到跨 GPU, 跨机架, 45°C 液冷与各类负载, 而 DSX OS 则提供调度, 生命周期管理与健康自动化的运营层. 二者协同设计, 旨在回收闲置功率, 并在固定兆瓦包络内增加有效算力. DSX MaxLPS 可让运营方在相同功耗预算内, 以高能效工作点部署多达 40% 的额外 GPU, 且对负载性能影响极小.

![图片](assets/d3353d0ab428.png)

具体实现上, 其实我在少年时期就跟着父亲做各种高低压配电设备. 大致的方法很简单:

机架电容吸收亚秒级功率尖峰.
Rack power steering 在 CPU,GPU 和 NVLink switch 之间动态移动功率预算.
DSX MaxLPS 根据 workload,telemetry,冷却和全厂功率上限重新分配 GPU power limits 和任务资源.

这里需要再说一句, 其实Blackwell的功率控制做的非常差, 前段时间做B卡逆向分析的时候专门测试过, 例如对MegaMoE这种极致使用硬件的算子, 针对8卡B300, EP8, V4-Pro config: 384 experts, top-6, hidden=7168, intermediate=3072的配置下, 降频分析如下表所示:

![图片](assets/67ca148ed04e.png)

然后做了一些消融的功耗实验, 单个子系统都不足以触发 DVFS（例如纯 cuBLAS GEMM 峰值 1072W / 97% TDP 也不降频），但 MegaMoE 因为 TC + HBM + NVLink 三者同时饱和而越过阈值。

![图片](assets/a344193314ac.png)

不知道Rubin是否还有降频的现象, 总觉得这块Nvidia做的并不是很好.

### 3.2 NV-HBI Cross Die片上网络的问题

其实在Blackwell上, 过去有很多文章都在写Cross-Die的访问内存延迟不一致的问题.  但是这些文章似乎和具体的测试结果有差异. 很简单的一个反例是在Hopper上Cross L2 Partition也有很大的延迟差异, 那么Blackwell上考虑到每个Die有两个L2 Partition, 整个芯片两个Die, 那么是否延迟测试中会出现4个峰值呢?  然而并没有. 其实NV-HBI的延迟远不是你们所认为的那样差异大到几百个cycles, 相反它的延迟是很低的.

但是在Rubin上, HBM带宽达到了22TB/s, 实际上L2的容量以及NV-HBI的带宽, 我觉得如果不做一些亲和性的处理可能会导致整体来看HBM带宽打不满, 大概率只能发挥到60~70%的峰值HBM带宽.

## 4. 总结

总体来看, Rubin架构是一个在Blackwell基础上进行了很小改动的产品, 将I/O移除到专用的I/O Die上是一个很好的Trade-off, 这样Main Die的面积用于放置了更多的SM.然后伴随着 K 维度的增加和提升频率和加大HBM的带宽, 使得性能大幅度的提高. 在具体的runtime中也进行了多项精细的优化, 使得整个SM空泡降低. 特别是ThreadBlock Level的dependancy使得构建一些相对低延迟的Megakernel变得更加容易.

但是在功耗和散热约束下, 我对极致的打满整个芯片时的功耗墙是否存在还是持怀疑的态度, 对于Sparse的GEMM在Attention中是否能够使用也持怀疑态度.

更详细的分析, 不知道哪个金主爸爸给我一个 R 卡, 大概一周内可以用Agent 完成所有的逆向工程分析:)

## 参考资料

[1]

Inside NVIDIA Rubin GPU Architecture: Powering the Era of Agentic AI: https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/

[2]

PTX 9.4: https://docs.nvidia.com/cuda/developer-preview/13.4/pdf/ptx_isa_9.4.pdf
