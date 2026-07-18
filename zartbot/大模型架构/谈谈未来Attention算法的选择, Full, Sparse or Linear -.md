# 谈谈未来Attention算法的选择, Full, Sparse or Linear ?

> 作者: zartbot  
> 日期: 2025年11月8日 12:24  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496753&idx=1&sn=b66ffd8d2e977cb4e7e27603ea9a9951&chksm=f995e4f3cee26de5d015e5822bed23c67fe583b9c9aeecf02031208e7f7017e80593179273e0#rd

---

### TL;DR

前几天当芯片哥搬砖的时候, 有一个观点似乎引起了争议. 今天变身为算法哥, 来详细阐述一下.

最近好像有一个很有趣的问题, 关于Attention的一些变化产生了不少争议, 大概分为几派:

首先是以DeepSeek为代表的**Sparse Attention**, 包括 NSA 以及后来在DeepSeek-V3.2上的DSA, 当然还有Kimi以前的一个关于MoBA的工作. 另一方面GPT-OSS中的一些滑动窗口的注意力机制和Attention Sink也可算到这里面. 然后是以Qwen-Next和Kimi-KDA为代表的**Linear Attention**, 特别的来说KDA使用了Linear和MLA Hybrid的方式结论性能比MLA还好, 挺值得关注的. 当然这一块还有似乎最近讨论比较激烈的RWKV. 接着是延续**Full Attention**的一派, 例如MiniMax从线性注意力机制回到Full Attention, 为此还专门发了一篇文 《为什么MiniMax M2是一个Full Attention模型？》[1]. 争锋相对的理由如下:

“一个现实的工业系统中，Efficient Attention想要打败Full Attention还有些距离. 在目前算力bound的前提下，还没有哪个模型真的把Softmax Attention打到能力上限。因此，从实际应用的角度来看，目前大家做Linear Attention或者Sparse Attention都是奔着省算力去的“

当然我们还不得不去看国外的一些情况, 以Google MoR为代表以及传言中GPT-5使用的UniversalTransformer这一派, 进一步的在提高Arithmetic intensity. 具体内容可以参考[《谈谈Transformer的一些演进: UT,MoD,MoR...》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494744&idx=1&sn=20f307c5e0fe7c5c5d62a46d81f48646&scene=21#wechat_redirect)它的视角是通过Recursive的方法, 再加上自适应的停时使得不同的Token有不同的算术密度来降低整体的算力,  这样的取舍来看, 虽然每层的Attn算力消耗是增加了, 但是可以降低模型的层数从而减少更多的MoE的内存访问的代价, 整体的TPS反而提高了.

最后还有一条从表示论视角出发的路径, 即DeepSeek-OCR和智谱Glyph把信息以图片的形式表示, 来构造一个Context压缩表示的方法, 从范畴论的视角看Word Embedding+Attention是一个一阶范畴, 那么这些构造图片的方式该是属于一个高阶范畴.

好像各种观点都有道理, 下面展开详细进行一些分析. 首先会有一个综述展开介绍各种方法在做什么, 熟悉这部分内容的可以直接跳过. 在第二章会进行详细的分析, 例如KDA为什么比MLA好, Sparse 和 Linear的一些区别.

## 1. 各种Attention的区别

先做一个综述来看看每一种技术路线做了什么取舍, 首先我们引入一些记号:

实质来看Attn是构建一个的态射, 各种算法是在这个态射的计算上

### 1.1 Full Attention

Full Attention公式如下所示:

简单来说, 它是一个**All-to-all**的注意力机制. 即序列中的每一个 token 的 Query 都需要与序列中**所有**的 token 的 key 进行交互并计算一个注意力分数. 这是一种无损的, 最全面的信息交互方式. 对于一个长度为 n 的序列, 其时间复杂度和空间复杂度均为. 而如今很多任务(Agent/RL)序列长度非常长时带来了极大的开销, 这是业界对它进行优化的初衷.

### 1.2 Sparse Attention

一个很自然的想法就是, 真实的情况并不是需要每个Token都需要去计算All-to-all的注意力机制, 因为从语言本身来看都有相应的语法结构, 修饰/指代等带来的注意力机制上应该是相对稀疏的. 另一方面, softmax本身相当于是给定了一个概率的budget给所有token分配, 这样过多的token参与到概率的分配上从这一点来看似乎也不对? 另一方面从范畴论的态射来看, 直觉上也不应该是一个满射.

那么, 自然而然的产生 **"Important-to-all"** or **"Local-to-all"** 的注意力机制. 认为完整的  注意力矩阵是稀疏的, 即每个 token 实际上只需要关注一小部分 "重要" 的其他 token. 它试图通过某种策略**预先剪枝**, 只计算注意力矩阵的一个子集.

例如早期的Transformer-XL / ReFormer / LongFormer/ Routing Transformer / Big Bird等, 在文章[《大模型时代的数学基础(4)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&scene=21#wechat_redirect)的2.2节有一些介绍. 最近的NSA/DSA相关的介绍[《谈谈DeepSeek Native Sparse Attention》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493234&idx=1&sn=cdca1661864f5ebf21c37e26fc51be10&scene=21#wechat_redirect),[《学习一下DeepSeek-V3.2》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496212&idx=1&sn=3ff9767a1b93ed8a495d2be614146f2d&scene=21#wechat_redirect)本质来讲主要分为两类:

**固定模式**: 如Sliding Window只关注邻近的 token; Dilated Window 跳跃式关注; BigBird 中的全局 token + 随机 token.

![图片](assets/db702544ee99.png)

**可学习/动态模式**: 如 Reformer 中的 LSH ; 近期的 MoBA, DSA 等通过学习一个索引器或路由网络来动态选择要交互的 token 或块.

而 NSA 则是进一步的融合压缩/block/sliding window 并通过一个索引器选择. 然后基于Chunk的Top-K路由机制对硬件也非常友好.

![图片](assets/033ec286c13e.png)

直观来看, 一方面通过Compression的方式实际上是得到了一个overall的压缩状态, 然后通过Top-K获取了更重要的一些Chunk, Sliding window又进一步加强了近期的Context注意力. 整体的Softmax概率对其它token的注意力分配变成了有选择性的处理, 也就是说从Full Attention的 **all-to-all** 变成了 **Important-to-all**

### 1.3 Linear Attention

对于GPU而言softmax中的指数计算的开销确实很高, 最开始的思路就是希望用一些更小计算量的方式去近似Softmax的效果. 即构造一个函数  满足

这也就是Linear Attention的核心思想**Kernelization & Associativity**. 通过核函数(Kernel)方法, 改变注意力分数的计算方式, 从而利用矩阵乘法的**结合律**来改变计算顺序. 它避免了显式计算  的  矩阵.线性注意力计算如下所示:

如果省略归一化项, 就变成了 , 利用结合律: , 其中 . 计算顺序变为先算  (一个小的固定大小矩阵), 再与  相乘. 让我们把线性注意力的公式重写一下:

在 Softmax 中, 我们必须先计算  得到一个  的大矩阵, 然后再乘以 .

在线性注意力中, 我们可以**先计算**, 然后再让  与其相乘.

这个  就是关键, 让我们定义一个**状态矩阵 (State Matrix)**:

 是一个列向量, 维度是 ,  是一个行向量, 维度是 . 所以  结果是一个  的矩阵.因此, 状态  是一个  的矩阵. 它的**大小是固定的**, 与序列长度  无关.

现在, 让我们看看这个状态是如何随时间演化的:

我们发现:

这种计算等价于一个 RNN 过程, 维护一个固定大小的"状态"(State), 不断用新的 (key, value) 对来更新这个状态.总结一下, 线性注意力的 RNN 计算模式如下:

初始化:  (一个  的零矩阵), 循环 (for i = 1 to N):

**状态更新 (State Update)**:

**输出计算 (Output Calculation)**: (这里转置了一下, 结果是  的列向量, 更符合习惯)

这就是线性注意力的 RNN 模式, 它的价值非常大:

** 解码复杂度**: 在解码阶段, 当我们生成第  个 token 时, 我们手里已经有了 . 我们只需要计算当前步的 , 做一次简单的矩阵加法和乘法来得到 , 然后再做一次矩阵-向量乘法得到 . 所有这些操作的计算量和内存访问量都是**常数**, 与序列长度  无关.

**等价性**: RNN 模式和并行模式 (如 ) 在数学上是**完全等价的**. 它们只是同一计算任务的两种不同分解方式.

**并行模式**: 适合在 Prefill/训练阶段, 利用 GPU 的并行能力一次性处理整个序列.

**RNN 模式**: 适合在 Decode 阶段, 以极高的效率进行自回归生成.

然后我们注意到 RNN 模式上来看, 实际上是需要去**理解并利用数据(状态 S)中的时间依赖性**, 换一个视角来看和数学中的时间序列分析的核心目标是类似的. 我们期望这个状态序列在某种意义上是具有**平稳性 (Stationarity)**, 即一个时间序列是平稳的, 意味着其统计特性 (如均值, 方差, 自相关性) 不随时间推移而改变. *   **均值恒定**: 序列没有明显的上升或下降趋势. *   **方差恒定**: 序列的波动幅度不随时间变化. *   **自相关性恒定**: 序列在任意两个时间点  和  的相关性只依赖于时间差 , 而不依赖于具体的时间点 .

在传统的时间序列分析上, 构建了一系列模型

模型

全称

核心思想

处理对象

关键特征

AR(p)

Autoregressive

当前值是过去值的函数

平稳序列

捕捉"惯性", PACF 在 p 阶后截尾

MA(q)

Moving Average

当前值是过去误差的函数

平稳序列

捕捉"冲击滞后", ACF 在 q 阶后截尾

ARMA(p,q)

Autoregressive Moving Average

结合 AR 和 MA

平稳序列

更节约参数, ACF/PACF 均拖尾

ARIMA(p,d,q)

Autoregressive Integrated MA

对差分后的序列使用 ARMA

有趋势的非平稳序列

"I"代表差分, 将非平稳转为平稳

GARCH(m,s)

Generalized Autoregressive Conditional Heteroskedasticity

波动率是历史波动和历史误差的函数

波动性时变的序列 (通常是均值模型的残差)

捕捉"波动聚集", 对误差的方差进行建模

**演进关系**:

**AR + MA = ARMA** (组合拳, 更灵活)

**ARMA + Differencing = ARIMA** (处理趋势, 适用范围更广)

**ARMA/ARIMA (均值模型) + GARCH (方差模型) = ARMA-GARCH** (一个完整的模型, 同时对序列的期望值和波动性进行建模, 在金融领域极其常用)

实际上, 我们在Linear Attention训练和推理过程中, 某种意义上来看也是需要一个状态矩阵的平稳性来防止梯度炸掉. 因此类似的在Linear Attention的算法中引入了各种变种. 通常他们会描述成一种线性时不变(LTI)系统.

很简单的一个做法就是引入了**衰减/遗忘门 (Decay/Forget Gate)**. 让我们看看这如何改变 RNN 的形式.以一个带标量衰减  的线性注意力为例:

我们再次使用结合律:

这次我们定义状态  为:

现在来推导它的循环形式:

然后在这个基础上, 我们进一步去构建一个更复杂的状态转移方程, 例如DeltaNet的RNN形式:

它将 DeltaNet 的状态转移函数分解为两个部分, 修正项 它将前一个状态矩阵  投影到一个新的空间.  为了更好地学习新的映射 , 模型需要**"忘记"或"削弱"**  中与  相关的旧信息. 在遇到新的输入  时, DeltaNet 的目标是最小化重构误差 .  直观的说, 这个修正项的作用就是: 将  中沿着  方向的分量"减去"一部分. 想象一下, 它在  所代表的"知识空间"中, 沿着  的方向"凿"掉了一块, 为即将到来的新知识"腾出空间". 然后就是一个更新项, 这部分与最基础的线性注意力形式相同.

它是一种**基于快速权重和在线学习理论的矩阵状态 RNN**. 它通过一种新颖的"修正-更新"机制, 实现了对记忆的动态内容寻址修改, 而非传统 RNN 的全局门控衰减. 这种设计赋予了它独特的记忆管理能力.

而Gated DeltaNet则在DeltaNet基础上增加了遗忘项, 首先, 我们回顾 DeltaNet 的状态更新公式:

修正项更新项

DeltaNet 的"修正"是被动的, 是由新的输入触发的. 想象一下一个旧的记忆, 比如 , 它被编码在了状态  中.只要后续没有新的输入  与  "冲突" (即, 在  的方向上有分量), 那么关于  的记忆就会**无限期地保留**在状态  中, 即使它早已变得无关紧要.它缺乏一种**主动的, 全局性的遗忘机制**来清除那些随着时间流逝而变得过时的信息.

为了解决 DeltaNet 的问题, Gated DeltaNet 引入了一个简单的**遗忘门**.

遗忘修正更新项

 的存在, 使得记忆有了"半衰期". 即使一个记忆不再被新的输入"修正", 它也会因为在每一步都被乘以一个小于 1 的 , 随着时间的推移而**自然地, 逐渐地淡出**.

然后更进一步的是Kimi的KDA, 它在GDN的基础上将**标量 (Scalar)** 遗忘门变为了一个**对角矩阵 (Diagonal Matrix)** 遗忘门, 实现**逐通道独立**的细粒度遗忘.

对角矩阵

标量  会被**广播**并乘以状态矩阵  的**每一个元素**. 所有历史记忆, 无论它们存储在状态矩阵的哪个"通道"(维度)里, 都会以**完全相同**的速率  进行衰减. 而KDA对角矩阵  与状态矩阵  的乘法, 相当于对  的**每一行**乘以一个**不同**的衰减因子.状态矩阵的每个维度/通道 (对应  个维度) 都有了自己**独立**的遗忘率. 模型可以学会:

让某些通道的  接近 1, 用于存储**长期, 稳定**的信息.

让另一些通道的  接近 0, 用于存储**短期, 瞬时**的信息.

KDA 中的状态转移可以被看作是更通用的 DPLR (对角加低秩) 转移的一种约束形式:

对比通用的DPLR:

**通用 DPLR (GLA) 的路线**: 追求极致的**理论表达能力**. 为了让最通用的公式能在并行算法中稳定运行, 不得不引入复杂的, 对硬件不友好的计算技巧 (对数域, 二次分块). 这是一种"理论优先, 工程妥协"的路线.

**KDA 的路线**: 追求**理论表达与工程效率的最佳平衡**. 它认识到, 并不需要完全通用的 DPLR. 通过将  与  绑定, 它保留了 DPLR 的核心能力 (细粒度衰减 + 内容修正), 但极大地简化了其代数结构. 这种简化的结构恰好可以被映射到一系列硬件原生的, 高效的操作上, 从而能够充分压榨 GPU 的性能. 这是一种典型的**算法-硬件协同设计 (Hardware-Software Co-design)** 思想.

### 1.4 UT,MoR路线

其实在Linear Attention中逐渐产生的 RNN 这样的递归结构是很有价值的. 另一条路线就是类似于UT, MoR的方式, 通过对Attention做递归的方式. 以UT的编码器为例, 编码器的目标是将输入序列 (长度为m, 维度为) 的初始词嵌入矩阵通过步迭代, 得到成最终的表征矩阵  .

在每个递归步骤 , 对所有 个位置的表征 进行并行更新, 得到. 其更新公式如下:

是上一步的表征. 一个关键的创新点是加入了 , 即位置和时间步编码 (Position and Time-step Embeddings). 另一方面经过自注意力处理后的输出  会通过一个在所有位置和所有时间步都共享参数的转换函数.

这个 Transition() 函数可以是一个 position-wise 的前馈网络 (Feed-Forward Network), 也可以是一个深度可分离卷积 (depth-wise separable convolution). 这种参数共享机制是实现`递归归纳偏置`的核心.

而MoR就是将这两个方向结合起来: 通过递归 (Recursion), 即反复使用同一组网络层, 实现参数共享.  通过路由器 (Router), 动态决定每个 token 需要经历多少次递归实现自适应计算.
![图片](assets/9ed6d86341cc.png)

这样的做法有几个优势:

`计算效率`: 注意力机制的计算复杂度是序列长度的二次方. MoR 只对在特定递归深度下仍然"活跃"的 token 进行计算, 大大减少了总计算量 (FLOPs).

`内存效率`: 传统Transformer需要对所有的token缓存KV, 非常消耗内存.  MoR 只缓存活跃 token 的 KV 对 (称为递归级 KV 缓存), 显著降低了内存占用和 I/O 开销

`性能`: 实验证明, MoR 在同等计算预算下, 性能超越了传统的 Transformer 和普通的递归 Transformer

实质上这些路线是在Softmax Attention的基础上引入递归的机制, 特别来说MoR是Sparse Attention和递归的一个结合体. 递归归纳偏置是指模型在处理序列数据时, 内在倾向于使用一种固定的, 循环的状态更新机制来压缩和传递信息.

## 2. 一些分析

### 2.1 从芯片和体系结构的视角

前几日随手写了一段, 其实表达的意思是: 作为一个芯片架构师不能去赌哪一条路线, 例如赌Linear方向去把SFU中的指数运算算力和TensorCore的算力比值做小(例如B200), 一般来说这样的赌博(Trade-off)十赌九输. 然后另一个观点是标准的Attention在数学上从Optimal Transport的视角来看, Softmax函数生成注意力权重的过程完全等价于一个单边熵OT问题的精确解.

然后紧接着的一个观点是在计算机体系结构中, 堆算力是一件比较容易的事情, 很容易Scale. 而内存访问带宽和容量是非常难Scale的. 因此不同的体系结构的取舍实际上都是在内存访问上进行优化, 进一步的去提升时间和空间的Data Locality.

![图片](assets/dfdc3999cb28.png)

从Linear Attention的角度来看是最节省内存带宽的, Decoding阶段只需要一个的状态矩阵即可.  对于Sparse则是一个恒定的常数 k 个Token的KV. 区别如下图所示:
![图片](assets/c2bb566761ce.png)

其实个人的观点是在打满访问内存带宽的时候, 需要进一步的去提高Arithmetic intensity, 对等的来说就是在针对Compute-Bound的计算时, 进一步增加一些内存访问. 也就是说  的状态矩阵 vs 一个相对大一些又不至于变成Memory-Bound的“状态矩阵”或者KV Cache, 我倾向于选择后者, 使得硬件的能力被充分的使用.

增强Arithmetic intensity一方面是如果硬件有足够的Softmax算力(例如改进后的B300 SFU), 似乎没必要去做Linear Attention省掉Softmax. 当然这只是对Native Linear Attention而言, 对于 GDN 和 KDA 实际上还加入了状态矩阵的遗忘和修正, 这些工作的收益本身是和 native Linear Attention正交的优化, 因为在Sparse Attention上也可以叠加一些递归的操作, 例如Google MoR同样可以做一些递归的操作.

### 2.2 从算法的视角为什么KDA优于MLA?

首先表明观点, 避免Kimi的同学误解. KDA是一个非常不错的工作. 而且我自己也是一个Kimi的重度用户, 在很多Agent任务上一直选择K2, 而最近刚发布的Kimi K2 Thinking虽然还没来得及去测试, 但总体看大家的评价都非常高.

对于传统的观点, 例如MiniMax谈到的: “Linear Attention或者Sparse Attention都是奔着省算力去的“, 即传统的Linear和Sparse Attn模型本身的性能上还是存在调点的情况.  而Kimi KDA结果是比MLA更好, 具体收益在哪?

对于Full Attention而言, 它的出发点是一个更加精确的模式匹配过程, 侧重于对信息的查找和恢复的过程. 而对于KDA一类的算法更多的是有一种“递归归纳偏置”的结构. 模型更被鼓励用于泛化, 抽象, 压缩信息, 通过Hybrid的方式将两者结合获得更好的性能.

另一方面有一个怀疑点, 这样的算法是否在50T token的训练一个更大规模的模型下还成立? 是否因为模型较小, softmax attention并没有得到充分的训练? 所以在这个阶段KDA相对于MLA有更大的优势? 这个是一个值得探讨的话题.

但总体来看KDA还是一个非常好的工作.

### 2.3 从应用的视角分析

前几个月一直在写一些Agent相关的代码, 实际上一开始也是需要很长的context, 但很长的Context很容易让模型执行的过程中出现一些幻觉和错误, 最终还是将任务拆分成多个sub-agent的方式在处理, 同时对于长Context的依赖降低了. 并且任务本身也因为多个subagent并行执行, 执行速度(Decoding速度)上的需求也小了一些.

其实这就和我们写程序一样的视角, 我们期望有一个非常大近似无限的内存(即对于大模型需要一个非常长的context), 但实际上只需要一个有限内存的栈即可, 任务分治下可能我们真的不需要很长的context. 例如我最近在用一些模型读一些数学书时, 通常我会用pdftk这样的工具截取文章的不同章节分开处理, 然后再拼接一些summary结果继续处理. 同时针对书籍本身的章节依赖结构来做一些手工的block选择.

另一方面在一些DeepResearch相关的任务上, 作为一个应用的开发者, 更希望的是对获取的不同的文档并行的进行Prefill后, 我可以按照Block-Level进行重组和拼接成一个新的context. 基于这两点, 从技术路线选择上来看, 个人比较倾向于Sparse Attention的路线. 即Context-Engineering和Sparse Attention 协同设计的方式, 避免拼接后的内容再做一次整体的Prefill.

对于Decoding的速度来看, 前面也阐述过, 一些subagent的并行处理对单个用户感知来看, 但用户的TPOT由于任务并行而提高了很多倍, 例如我在处理 N 只股票的一些量化分析时, 我可以同时向平台并行Query N个任务, 并不会在意处理的速度. 而且对于很多Browser-Use/Computer-Use的场景下, 更重要的是每次操作的精准性, 避免整个任务因为错误的操作而Rollback. 它们的交互延迟通常也是几十个毫秒的级别, 因此可能也不需要极致低的Action反馈.

其实还有一个是从商业的视角考虑. 假设我用Linear Attention Decoding的速度是Full Attention的数倍, 但是模型效果上导致单Token的定价是其它模型的20%~50%, 整体来看整个GPU服务器做MaaS的营收实际上是差异不大的, 并不能单纯的去从TPS的角度比较经营的收益. 另一方面, 我们还需要考虑整个任务的token开销, 例如针对同一个任务Linear Attn产生的Token数量多于Full Attn, 后续的任务累积起来的开销也会变大.

## 3. 一些结论和个人的观点

首先个人的观点是不能简单的把Sparse和Linear两种路线对立起来看. 特别是 GDN 和 KDA 在“状态矩阵”引入了遗忘和修正项后, 特别来说 KDA还引入了一个对角矩阵的按通道的不同遗忘权重, 实际上是很有价值的. 另一方面Sparse Attn中按Block的处理对于很多Agent任务拼接context也可以节省掉大量的Prefill的算力, 这方面也是相对Linear有很多优势的.

至于是否省掉Softmax来作为评判Sparse和Linear的观点是相对片面的, 特别来说在[《大模型时代的数学基础(9)- SDPA和最优传输, 强化学习及信息几何的联系》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494688&idx=1&sn=3d589f6d4be56ee372d5db4f8631b0cc&scene=21#wechat_redirect)中介绍了一篇论文的观点: 注意力机制的前向计算过程, 即通过Softmax函数生成的注意力权重的过程, 完全等价于一个单边的熵最优输运(One-Sided Entropic Optimal Transport, EOT)问题的精确解. 另一方面,通过标准反向传播计算出的梯度, 在数学上等同于强化学习中的REINFORCE这样的策略梯度(Policy Gradient)算法. 意味着注意力机制的学习过程是一种理性的有明确目标的优化策略.这揭示了注意力学习的动态过程--它会“奖励”那些带来高于平均效用的键(Key), “惩罚”那些低于平均效用的键.

在Sparse Attn的路线上引入类似于 KDA 的“状态矩阵”, 增加一些递归结构, 例如UT/MoR, 甚至是Google Titan 《Titans: Learning to Memorize at Test Time》[2]这样的方法, 引入Memory Context达到类似于 KDA 这样算法的“状态矩阵”的效果, 即Memory as Context(MAC).

![图片](assets/ba2362d64f15.png)

正是基于这个视角, 前段时间有一个[《谈谈 Hierarchical Sparse Attention (HSA)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496275&idx=1&sn=5f8a8d8efff22033d3f2aed8a5844e53&scene=21#wechat_redirect)的脑洞...

另一方面从压缩的视角来看, DeepSeek-OCR和智谱Glyph也是一个针对长Context很不错的压缩处理方式, DS-OCR 的贡献是纵向的, 深入的. 它深入到模型架构的底层, 设计了一个在物理层面就更适合处理高分辨率图像并进行高效压缩的编码器. 这是一种更根本的, 从第一性原理出发的优化. Glyph 的贡献是横向的, 应用驱动的. 它接受现有 VLM 的能力作为前提, 探索如何通过最优化输入 (渲染方式) 来最大化其潜能. 这种方法更灵活, 更具普适性, 理论上可以应用于任何强大的 VLM.

DS-OCR 在其讨论部分明确提出了将光学压缩与人类记忆的遗忘机制进行类比, 通过逐步降低历史图像的分辨率来模拟记忆的模糊化. Glyph 虽然没有直接强调 "遗忘", 但其对不同分辨率下性能的实验, 以及对极端压缩的探索, 实际上也为这种层级化, 有损的上下文管理提供了实验基础.

最后还有一个未来演进中非常值得思考的一个问题, 那就是分布式的在线强化学习的能力. Attn的结构如何与这些任务相匹配...这个问题等以后慢慢有空想清楚点了再来谈吧...

参考资料

[1] 
为什么MiniMax M2是一个Full Attention模型: *https://mp.weixin.qq.com/s/pujTJMUYMp0nVbt7uAb4vg*
[2] 
Titans: Learning to Memorize at Test Time: *https://arxiv.org/abs/2501.00663*