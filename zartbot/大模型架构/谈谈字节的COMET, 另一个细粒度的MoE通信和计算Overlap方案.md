# 谈谈字节的COMET, 另一个细粒度的MoE通信和计算Overlap方案

> 作者: zartbot  
> 日期: 2025年3月6日 13:30  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493369&idx=1&sn=6adc84f2786147b8832cd8377ad24955&chksm=f995f63bcee27f2d40bfabf167517bb70e3f30a046e35ec2e20b287386ea7bedff6779a09e65#rd

---

### TL;DR

字节前几天发布了一篇论文《Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts》[1], 也是在通过通信和计算的Overlap解决MoE的计算效率问题, 最近做了一点分析, 记录如下:

首先是当前的训练框架下, 论文中说跨机通信会导致47%的时间开销在MoE通信上, 因此需要更佳细粒度的Overlap,Comet的主要工作是对通信和计算之间的共享Buffer(即文章中的Shared Tensor)进行分析, 并按照特定的维度分解来消除通信和计算之间粒度不匹配的问题, 并重新组织数据进行调度, 实现了细粒度的Overlap, 最终单个MoE层执行速度性能提升了1.96倍. E2E提升了1.71倍.

然后代码已经开源并整合到了Flux**框架[2]

### 1. 通信和计算Overlap的难题

论文中在2.2章节阐述了一下MoE模型进行细粒度Overlap的难点:

#### 1.1 通信和计算的粒度不匹配

如图所示, 在MoE中按照Token(1x Hidden-dim) 作为基本的移动单位. 而GEMM计算是以Tile进行处理的(图中紫色的block, 例如size为128x128), 因此这样的错配对于细粒度的Overlap带来了难题.

![图片](assets/3d5bbd343f4f.png)

这样的错配带来了复杂的数据依赖性, Token的具体路由决策由MoE Gating决定, 每一个Batch Expert需要处理的Token都对MoE disptach阶段的实际结果有数据依赖. 即进行GEMM的block Tile将会关联多个token, 这些是由Disptach的token_idx决定的. 然后作者直接得出了一个结论, 需要通过细粒度的通信, 让每一个计算的Tile通过UVA(Unified Virtual Address)读写, 并利用数据的重组和重新调度来有效的隐藏访问延迟并提高计算效率.

紧接着就谈到了细粒度的通信. Token粒度的通信和TileBased Computing进行Overlap挺难的, GPU之间的跨机远程I/O操作**延迟更高, 对远程的token进行大量的细粒度读写可能导致耗时过长阻塞后续计算, 使得计算资源利用率较低. 然后提到了对Hopper中的TMA**构建的异步pipeline延迟影响更大.

#### 1.2 不同的计算和通信负载

每个Expert接收的Token不同, 导致运算时的token输入的形状也有不同, 因此需要将计算和通信的负载动态的调整.

### 2. Comet详细介绍

如下图所示, Comet设计原则主要有两点:

Shared tensor based dependency resolving: 如下图中的(1)(2), 通过沿特定维度分解共享张量打破粗粒度的数据依赖, 然后重新调度计算来提高效率,确保有效的overlap

Adaptive workload assignment: 动态负载均衡, 平衡通信和计算核的开销

![图片](assets/d420fad383be.png)

#### 2.1 Shared Tensor Based Dependency Resolving

在dispatch阶段是以tensor维度独立操作的, 而在Combine阶段有reduce计算是在hidden-dim维度上独立操作的.

![图片](assets/9b5e21b27a98.png)

然后从消费者和生产者的视角来看待这个问题, 例如在Dispatch阶段, 通信算子是生产者, 而GEMM算子是消费者. 那么在调度的时候就尽量希望GEMM能够很快的拿到相应的Tile就进行计算. 然后在重排调度的时候, 优先调度local token进行计算.

![图片](assets/c2575cee9319.png)

然后在Combine Reduction阶段, 按照Hidden-dim的维度解耦调度

![图片](assets/f93af8990d98.png)

#### 2.2 Adaptive Workload Assignment

一种很直观简单的方法是对于GEMM, 在Cutlass的Prologue和Epilogue阶段融合通信, 然后所有的SM采用同构的计算架构. 但是一些不规则的通信延迟会导致计算和通信Overlap的不确定性.  另一方面是token level的细粒度的I/O可以在Hopper这样的架构下显著提升Kernel的计算效率. 因此字节在Comet上还是实现了通信和计算的Thread block level的隔离, 如下图所示:

![图片](assets/05cf03a6c499.png)

GEMM采用Cutlass实现, 并采用了Warp Specialization, Producer Warp采用TMA从GMEM拷贝内存到SMEM, Consumer Warp执行MMA计算, 并将结果写回到GMEM. 然后通信的warp再从GMEM读取进行后续的通信.

这是常规操作, 但总感觉这里很不舒服, 就像一个Linux的TCP协议栈那样, 多了一次拷贝, 所以后面写了一小段`HW resource restriction`.理论上可以将COMPUTE和COMM**放在同一个Thread Block内, 避免额外的GMEM access, 但是Thread Number的限制是的通信算子无法有效的利用资源,同时COMM和COMPUTE在一起还有互相的干扰.

既然分离了COMPUTE和COMM, 那么就要有一个根据负载自适应调整的能力. 主要就是针对不同的workload做了profiling

![图片](assets/816f66dd4dfb.png)

#### 2.3 实现

具体实现上, GEMM算子采用了CUTLASS, 通信上和DeepSeek DeepEP一样, 也采用了NVSHMEM库. 但是从测试的数据来看, 似乎版本都比较老, 估计是蛮早的工作了, 等着DeepEP开源以后才公布出来的? 从代码上来看, 在`/docs/moe_usage.md`中显示, EP并行采用的是Allgather + GEMM+ Reduce Scatter

Design Guide里面有一个图画的很清楚, 在 Dispatch阶段如下所示:

![图片](assets/2ce0c2c251c1.png)

在Combine阶段如下:

![图片](assets/00b1e00e0b29.png)

整体的一个图:

![图片](assets/497ed49cc193.png)

这个工作考虑到通信和计算的粒度不同和数据依赖做了细粒度的解耦和调度, 工作也非常细致.  但是这个工作还是在单机内的EP并行, 并没有像DeepSeek那样有跨机的IBGDA的处理, 不知道火山云的DeepSeek推理优化做了哪些工作. 

另一方面是, 渣B对于Flux代码不熟悉(没看过,懒, 没时间看), 最近工作也比较忙, 后面空一点再来详细读读代码吧. 

参考资料

[1] 
Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts: *https://arxiv.org/abs/2502.19811*
[2] 
flux: *https://github.com/bytedance/flux/*