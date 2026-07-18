# 谈谈DeepSeek-v3在算力约束下的出色工作

> 作者: zartbot  
> 日期: 2024年12月28日 14:52  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492969&idx=1&sn=0a4f24450171d6659fce9bab093a15d6&chksm=f995f5abcee27cbd0d14e0151c71acaf0cfee07329de2046ea4363a9d0e11558f5788dc4f84c#rd

---

寒冷的周末, 加完班挤点时间读个论文吧. Deepseek-v3仅用了2048块H800 GPU就超越了Llama 3 405B模型, 要知道Meta训练Llama3可是用了16384块H100, 而DSv3的训练成本非常低

![图片](assets/161c1c60b680.png)

在所有人追求更大规模集群的时候, Deepseek这样的工作只有一个词评价: Respect!

其实还有另一件事情让渣B内心深处与之共振了一下, 上周末12.20是我们量化基金算法十周年的纪念日. 十年前渣B和合伙人在张爱玲故居常德公寓的咖啡馆里, 突然想到了一个并行和近似计算的算法, 当天回去写了一下, 把算法的计算时间从10天缩短到了3分钟, 使得模型预测做到了近实时性上线的能力, 于是我们就把这一天当作了基金的纪念日, 当然渣B是一个非常佛系的人, 离梁总那样带出如此出色的幻方和DeepSeek的成就差太远了.

但是看到Deepseek FP8 Training, Block-Wise Quantization, MoE的ALF负载均衡, 以及MTP, 还有在集群通信上针对AlltoAll和PP并行的优化, **一系列手段, 特别是对Hopper的架构挖的很深, Infra团队出活非常细致. 作为量化同行和Infra同行, 对这些算法和算力协同的系统性优化所取得的成就感到敬佩.** 另外在3.5. Suggestions on Hardware Design这一章中, 对GPU和通信硬件的设计做出了建议, 这一部分跟我正在做的一些工作完全重合.

因为工作上还有其它很多重要的事情, 时间有限,本文仅做一些大概的分析,而DS团队在实现这一系列通信计算Overlap,负载均衡, 同时兼顾并行分布式推理的大量工作, 后面有机会分析时将详细叙述.

这一篇主要涉及AIInfra这一块, 对于PostTrain和模型结构这些后面再仔细做一个分析.

## 1. DeepSeek-V3概述

DeepSeek-V3是一个671B的MoE模型, 每个Token激活参数为37B, 采用了MLA和DeepSeekMoE架构, 在大多数模型还在维持Transformer架构时, DeepSeek直接对架构进行了两个非常重要的创新, 并且通过v2充分验证了MLA和MoE的性能, 非常出色的工作. 一些Benchmark如下, 突然有点心疼Meta的几个亿美金...

![图片](assets/db747458043a.png)

在同类产品中基本上做到了领先, 特别是在Code和math上.

![图片](assets/cd479e9d7139.png)

### 1.1 模型结构

Deepseek-v3模型结构如下:

![图片](assets/ba6aa3992296.png)

关于MLA和DeepSeekMoE在DeepSeek-v2发布时已经进行过分析:

[《继续谈谈MLA以及DeepSeek-MoE和SnowFlake Dense-MoE》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489919&idx=1&sn=e0f253eef5637a364defc1ce2051d713&scene=21#wechat_redirect)

模型的Hidden Dim为7168,  attention heads: 128, 模型层数为61层, 比DSv2多了一层, DSv2的第一层为FFN, 而DS v3的前三层都为Dense MLP. MoE层采用了1个共享专家和256个路由专家, 每个Token激活8个专家, 并确保Token仅被路由到4个节点.

新的工作是Auxiliary-Loss-Free Load Balancing和Multi-Token Prediction, MTP的工作使得模型内嵌了一些推测解码能力.

![图片](assets/c33d55a6fdc1.png)

具体来说就是通过额外的几个MTP模块来顺序的预测K个额外的Token, 这些模块非常简单. 需要注意的是这个可能是对未来模型有非常重要影响的一个功能, 通过MTP增加了数据的使用效率.

**MTP让我想到了Zen5的2-Ahead Branch Predictor** 非常有趣的工作, 其实对于o3这样的模型, 本质上是token as an intruction.

原来GPT是一个顺序执行结果predic next token 类似于 pc++, 然后在栈上(historical tokens  as stack)操作. 顺序预测下一个token

o1/o3 Large Reasoning Model 无论是MoE或者是强化学习一类的PRM, 实质上是在Token Predict上做了Divergence,  例如跳转/循环/回溯 等,  PRM可以看作是一个CPU分支预测器.  从体系架构上渐渐的可以让大模型做到类似于图灵完备的处理能力.

基于这个观点, 那么当前的GPU的TensorCore/Cuda Core实际上就构成了一个执行引擎,  外面还需要一系列控制, 分支预测,  译码器, LSU来配合, 对于基础设施带来的演进还是有很多有趣的话题可以去探索的

当然还有post training中使用DeepSeek-R1也是非常赞的工作. 这些内容后面有空伴随着o3的LRM分析再一起来做.

### 1.2 训练并行策略

很早就在关注DS的模型框架, 他们并没有使用Megatron这些现有的框架, 而是自己从零开始打造的HAI-LLM, 对于模型层数为61层, 而且前三层为Dense MLP, 从训练的并行策略来看采用PP=16, EP=64放置在8个节点上, DP采用了ZeRO-1 Offload. 然后通过一系列内存优化, **没有使用代价很大的TP并行!**,
这也是针对H800被砍了NVLINK带宽的优化, DS这帮厨子干的非常巧妙!

在PP通信上, 设计了DualPipe算法, 与现有的PP方法相比，DualPipe产生的管道气泡较少。更重要的是，它在前向和后向过程中重叠计算和通信阶段，从而解决了跨节点专家并行引入的通信开销大的挑战.
然后针对EP的跨节点all-to-all通信也做了非常细致的优化.

### 1.3 并行推理策略

这也是一项非常关键的工作, 使得模型进入分布式推理的时代了. 首先是针对SLO使用了PD分离的策略.

Prefilling最小部署单元为4台机器32卡, 在Attention计算部分采用了TP4并结合序列并行(SP), 并同时和DP8相结合. 较小的TP可以获得更小的TP通信开销.

针对前面三层的Dense MLP采用了TP1的方式来进行运算, 目的也是降低TP通信开销. 在MoE层, EP=32,即让它在四个H800节点(32卡)之间同时采用ScaleOut和ScaleUp对AlltoAll通信优化. 而针对EP中的AlltoAll专家通信通信也进行了非常细致的调优.

然后有一个非常关键的创新是为了保证MoE部分不同专家之间的负载均衡, 采用了Redundant Experts策略,即复制高负载专家并在多个位置部署这些冗余专家。高负载专家是根据在线部署期间收集的统计数据检测出来的，并定期调整（例如每10分钟）。确定冗余专家集合后，根据观察到的负载，在节点内的GPU之间仔细重新安排专家，尽量平衡GPU间的负载而不增加跨节点Alltoall通信的开销.

在Prefill阶段每个卡多配置了一个冗余专家, 总共9个专家提供服务.另一方面为了隐藏A2A和TP的通信开销, 同时处理2个计算量相似的microbatch负载, 将一个微批次的注意力和MoE计算与另一个微批次的分发和合并操作重叠进行. 这种做法比Snowflake的MoE策略看上去更加简单有效,SnowFlake是通过将Attention和MoE并行连入网络的方式进行的.

然后还在探索Prefill阶段提供更多的冗余专家进行更多的动态路由和全局调度的工作.

在Decoding阶段, 每个token在路由的时候会选择9个专家, 其中共享专家被视为一个总是被选中的高负载专家. Decoder集群部署的最小规模为40个节点,共计320张卡. Attention计算采用了TP4+SP, 并且配合DP80, MoE部分采用了EP320, 对于MoE部分，每个GPU仅托管一个专家，且有64个GPU专门负责托管冗余专家和共享专家。分发和合并部分的A2A通信通过IB上的直接点对点传输来实现低延迟, 并且采用了IBGDA进一步降低延迟.但是GDA也有它内在缺点, 例如GPU准备WQE和敲Doorbell带来的影响, 虽然bypass了CPU降低了延迟, 但是对GPU的开销和通信效率上的影响还是很大的.

![图片](assets/387c08c11681.png)

类似于Prefill阶段也有一个scheduler动态监控负载情况. 然后全局负载均衡使用的optimal routing算法要和dispatch/combine kernel融合, 这里应该有一些很有趣的工作可以再细致的挖掘一下.例如文章提到的在Decode阶段隐藏A2A开销, 类似的做两个microbatch来overlap? 但是Decode阶段的attention计算消耗的时间更长.

这样的并行策略哟一个优势, 每个专家的批次大小相对较小（通常不超过256个token），瓶颈在于内存访问而非计算。由于MoE部分只需加载一个专家的参数，内存访问开销极小，因此使用较少的SM不会显著影响性能. 为了避免对Attention计算的干扰, 对dispatch/MoE/Combine Kernel进行了融合.

## 2. 训练用到的并行技术

### 2.1 DualPipe Overlap计算和通信

DSv3由于专家数量非常多, 必然会存在跨节点的专家并行, 另一方面很多人为了避免跨节点之间的A2A通信效率的问题, 在单机内做EP完全是胡搞, 你这么瞎搞的EP8还不如直接一个DenseMLP.但是正如论文说的, 跨节点EP导致计算与通信比率低至大约1:1，效率不高.

于是DS采用了DualPipe的方式, 不光有效的Overlap了FW/Backward的计算和通信, 还减少了PP中的气泡,非常优雅的解决方案.

![图片](assets/f2f3d501e7a2.png)

他们通过将独立的forward/backward chunk配对的方式进行overlap, 将每个chunk分为四个组件:
Attention, A2A dispatch, MLP和A2A combine. 对于att和MLP更进一步分为输入反向传播和权重反向传播两个部分.在这种重叠策略下，我们可以确保所有全对全和PP通信在执行过程中都能被完全隐藏。基于高效的重叠策略，完整的DualPipe调度如图

![图片](assets/488ebaf73455.png)
它采用双向管道调度，从管道的两端同时馈入微批次，并且大部分通信可以完全重叠。这种重叠还确保了随着模型规模的进一步扩大，只要保持恒定的计算与通信比率，仍然可以在节点之间使用细粒度的专家，同时实现几乎为零的A2A通信开销。

### 2.2 高效的跨节点A2A通信

为了确保DualPipe具有足够的计算性能, DS针对跨节点的A2A通信开发了专用的Kernel,可以节省用于通信的SM数量, 同时又将集群网络拓扑和MoE Gating算法协同进行了设计.

在H800上,DS的报告是按照单向带宽计算的, NVLink提供的带宽是160GB/s, 节点之间互联的IB带宽是50GB/s(400Gbps每卡). 考虑到带宽差距为3.2倍, 将每个Token最多分发到4个节点减少IB流量. 对于每个token，当其路由决策确定后，它将首先通过IB传输到目标节点上具有相同节点内Rank的GPU, 再通过NVLINK转发到目标GPU, 其实就是Nvidia的A2A PTX的优化. 这样IB和NVLINK通信重叠, 然后每个token平均选择每个节点3.2个专家, 因此不会产生额外的NVLINK的开销, 通过这个策略, 仅使用20个SM就可以充分使用IB和NVLINK的带宽.

在通信Kernel中, 将20个SM划分为10个channel, 在dispatch过程中分为1. IB Send, 2. IBtoNVLINK转发, 3. NVLINK接收, 这些任务都在不同的warp处理, 并且采用了Hopper的Warpspecialization的功能, 并且分配给每个通信任务的warp数量根据SM上的实际工作负载动态调整, 同样Combine也是类似的过程: (1) NVLink sending,(2) NVLink-to-IB forwarding and accumulation, and (3) IB receiving and accumulation，也由动态调整的warp处理。

另外一个非常细致的工作时, 自动调整通信块大小, 并通过PTX指令显著减少L2缓存对其它SM的干扰. 其实就是在LD/ST过程中使用cs(cache streaming)策略, 因为这些数据仅在通信时访问一次, 标记在L2 Cache中尽快的被evict.

![图片](assets/62a4c32714b7.jpg)

### 2.3 内存优化

主要是Activation重计算, 然后在CPU存储模型参数的指数移动平均值,并异步更新, 用于提前估计模型的性能.然后将模型较浅的几层和Embedding层与模型的最后一层(Output)放在同一个PP Rank中, 主要是模型采用了MTP, 可以共享.

### 2.4 FP8训练

这是非常棒的一项工作. 主要是Tile/Block-based 的细粒度量化训练策略以及混合精度训练. 并且对于1T token的训练对比了FP16和FP8, 相对损失误差始终保持在0.25%以下.

#### 2.4.1 混合精度框架

对于大部分计算密集型任务采用FP8精度计算, 这些GEMM操作接受FP8张量作为输入，并产生BF16或FP32格式的输出。如图6所示，与线性算子相关的所有三个GEMM操作，即前向传播（Fprop）、激活反向传播（Dgrad）和权重反向传播（Wgrad），均在FP8中执行。这一设计理论上比原始的BF16方法快一倍。此外，FP8 Wgrad GEMM允许激活函数以FP8格式存储用于反向传播，这大大减少了内存消耗。

![图片](assets/543485594430.png)
一些成本较低的运算和误差影响较大的计算还采用更高的精度, 例如Emb, Output Head, MoE Gating, Norm, attention operator. 同时为了保持数值稳定性, Optimizer/Grad/Master weight等还是维持FP16/FP32. 当然这些通过DP中的Zero-1 Sharding还是可以在多个GPU之间分担的.

#### 2.4.2 提高精度

引入几种提高精度的策略, 重点在于对量化和乘法的改进.

细粒度量化: 对于激活函数以1x128的Tile为基础进行分组和缩放,  对于权重, 以128x128 Block为基础进行分组和缩放, 这种方法确保了量化过程能够通过根据较小的元素组调整比例更好地适应异常值.
其中一个关键的修改为引入了沿GEMM操作的内部维度的分组缩放因子, 并且配合了FP32的累加策略消除误差, 非常巧妙的一个做法.

提高累加的精度: 低精度GEMM操作常常面临下溢问题，其准确性在很大程度上依赖于高精度累加，并且观察到，在NVIDIA H800 GPU上的FP8 GEMM累加精度仅限于保持大约14位，这比FP32累加精度显著降低。当内部维度K较大时, 这个问题会变得更加明显，这是大规模模型训练中的典型情况，其中批量大小和模型宽度都会增加。以两个随机矩阵的GEMM操作为例，当K = 4096时，在初步测试中，Tensor Core中的有限累加精度导致的最大相对误差接近2%,

![图片](assets/e402e7d7b146.png)
DS做了一个修改, 在TensorCore上执行矩阵MMA时，中间结果使用有限的位宽进行累加。一旦达到一个𝑁𝐶间隔，这些部分结果将被复制到CUDA Cores上的FP32寄存器中，在那里进行FP32累加。并且通过细粒度量化沿内部维度K应用每组缩放因子。这些缩放因子可以在CUDA Cores上高效地作为反量化过程的一部分进行乘法运算，几乎不增加额外的计算成本。但是这样的做法降低了WGMMA的执行效率, 但是Hopper本身就有Warp Specialization的能力, 当一组Warp在执行精度提升操作时, 另一组执行MMA. 并且可以重叠.

FP尾数优于指数, 对于FP8有E4M3和E5M2两种表示, 对所有张量都采用了E5M2, 并且由于Tile/Block-wise 量化, 有效地在这些分组元素之间共享指数位，从而减轻了有限动态范围的影响。

#### 2.4.3 低精度存储和通信

Activation和Optimizer state进一步压缩成低精度, 从而节省内存使用,避免TP并行带来的开销.

采用BF16保存AdamW优化器中的Moments, 但是主权重和梯度仍然保持FP32.

低精度激活函数: Wgrad操作是在FP8中执行的。为了减少内存消耗，自然选择是以FP8格式缓存激活函数以供线性算子的反向传播使用。但是，对于某些运算符采取了特别考虑，以便进行低成本高精度训练：注意力运算符之后的线性输入。这些激活函数也在注意力运算符的反向传播中使用，因此对精度敏感。因此为这些激活函数专门采用了定制的E5M6数据格式。此外，在反向传播过程中，这些激活函数将从1x128量化Tile转换为128x1 Tile。为了避免引入额外的量化误差，所有的缩放因子都是整数次幂的2。另一方面在MoE中的SwiGLU运算符输入。为了进一步降低内存成本，缓存SwiGLU运算符的输入并在反向传播时重新计算其输出。这些激活函数也以FP8格式存储，并使用细粒度量化方法，在内存效率和计算准确性之间取得平衡。

低精度通信: 通信带宽是MoE模型训练中的关键瓶颈。为了缓解这一挑战，在MoE up-projection前将激活函数量化为FP8，然后应用Dispatch组件，这与MoE up-projection中的FP8前向传播兼容。类似于注意力运算符后的线性输入，此激活函数的缩放因子也是2的整数次幂。类似的策略应用于MoE下投影前的激活函数梯度。对于前向和后向Combine组件，保留BF16精度

## 3. 对硬件设计的建议

这一部分非常有趣, 他们的这些观点和渣B现在正在做的一些工作基本上是重合的.

### 3.1 网络硬件

当前H800的132个SM中被分配了20个SM用于通信, 限制计算吞吐量。此外，使用SMs进行通信会导致显著的效率低下，因为TensorCore完全未被充分利用。

因此希望硬件供应商能够开发对通信和集合通信Offload的专用网络处理器和协处理器, 例如AWS Trainium上很早就有Collective Engine. 另一方面是为了减少应用程序编程的复杂性，希望这种硬件能够从计算单元的角度统一ScaleOut和ScaleUp网络。通过这种统一接口, 计算单元可以通过提交基于简单原语的通信请求.

例如渣B在推测Rubin架构时也提到了这个问题

[《推测一下Nvidia Rubin的288卡系统架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492912&idx=1&sn=7a6dddb36e0644182a57ecf0f48354ab&scene=21#wechat_redirect)

ScaleUP和ScaleOut语义的融合是一个非常重要的工作, 准确的来说在ScaleOut使用RDMA就是一个错误, 并且想简单的在ScaleUP使用RDMA也是一个错误.

[《HotChip2024后记: 谈谈加速器互联及ScaleUP为什么不能用RDMA》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492300&idx=1&sn=8a239883c831233e7e06659ec3425ea2&scene=21#wechat_redirect)

![图片](assets/61998a0338bc.png)

### 3.2 计算部件

在FP8 GEMM中, 对于TensorCore采用更高精度的Accumulator, 支持Tile/Block based的量化, 使Tensor Cores能够接收缩放因子并实现带有组缩放的MMA来支持细粒度的量化.避免数据的移动.
另一方面支持Online的量化能力, 例如当前需要从HBM读取128个BF16 Activation然后进行量化, 并以FP8写入HBM, 然后再次读出来进行MMA.

然后DS的建议是FP8转换和TMA同时实施, 直接进行量化以便Activation在从GMEM到SMEM直接完成量化, 避免多次内存读写. 同时还建议加入warp level的转换指令, 进一步促进NormLayer和FP8转换融合.

或者，可以采用近内存计算方法，将计算逻辑放置在靠近HBM的位置。在这种情况下，BF16元素可以在从HBM读入GPU时直接转换为FP8，减少大约50%的片外内存访问.

最后还建议了一个Transpose GEMM的操作, 因为在FP的过程中, Activation Tile被量化并存储为1x128的向量, 然后在BP时需要读取矩阵, 反量化, 转置并重新量化成128x1的向量, 希望这些密集的呢次访问操作降低指令issue数量和HBM带宽占用.

这一块脑补了一下, 实现应该很简单, 在L2Cache和TMA上改一下即可,并不是很复杂.