# 从GTC25谈谈大模型架构的演进

> 作者: zartbot  
> 日期: 2025年3月23日 11:07  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493596&idx=1&sn=ad651efb4add04122cdb789111fcff74&chksm=f995f71ecee27e08fd7b30f437fc3b9925e66b94942aa0680e2f85fe8d79a6454dcb2b773a6a#rd

---

对于一个做芯片的算法工程师, 很自然的会从上而下去考虑这个问题, 毕竟一颗芯片出来的时候要几年的时间, 还要再继续用几年,对于模型架构的演进势必要做出更多的预测和估计.

由于中美的算力的差距, 因此在模型架构的演进上也会产生一些分裂, 其实我以前的一些语境表述也有相对含糊的地方, 没有严格区分这两种不同的情况下模型发展的路径. 例如很多时候在谈论稀疏的MoE和Transformer时, 更多的是考虑到国产算力可能有很长一段时间会有差距时, 如何通过一些计算/网络/存储的trade-off构建, 这些取舍下必然会走向一条稀疏的路, 但是稀疏本身也有代价, 在两年多前就在文章[《大模型时代的数学基础(2)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488528&idx=1&sn=fa49e334201e738e7ddb4258030798b3&scene=21#wechat_redirect)基于范畴论的视角做出了一个判断

例如在降低Transformer的计算量时，稀疏Transformer或者MoE是否会破坏态射结构? Transformer算子的可组合性如何设计?通过这样顶层的抽象视角会得出不少有价值的答案。当然还有很多范畴论的内容，例如limit/colimit，以及相应约束下的强化学习和基于Hom函子去构造数据，最终来提高大模型的逻辑推理能力，范畴论视角下函数式编程和大模型的融合，这些都是非常值得我们去深思的问题，或许这也部分回答了OpenAI Q* 的一些解法，我们拭目以待...

预训练模型本身是一个presheaf, 而稀疏的Attention和稀疏的MoE本质上在训练的过程中会破坏其中的一些态射, 但过于Dense似乎又会有更多的错误的态射会被保留, 最终Post-Training的一些RL又能某种程度上强化一些态射的概率.

那么一个比较严谨的叙事是, 首先在不考虑国产算力的约束和一些先进芯片可获得性的约束下, 从第一性原理模型该如何设计, 然后再是针对国产算力的一些取舍.

### 1. GTC25发布的一些硬件趋势

#### 1.1 Jensen Math Translation

老黄的数学需要做一个详细的翻译才行, 否则很多数据无法对齐, 一方面是单向带宽/双向带宽混用, 另一方面是算力上以前很多都是以稀疏的算力计算的, 当然比AWS那种bit/byte大小B混用还是好很多.

![图片](assets/a12c8a274d0b.jpg)

Blackwell Ultra单机柜有72颗B300的芯片,单芯片FP4 Dense算力为B200的1.5x即每芯片15000TFLOPS, 累计72卡1.1EFlops, NVLINK互联带宽为双向1800GB/s, 累计72卡的总带宽为64TB/s, ScaleOut网络累计有72个800Gbps的CX8,按照双向计算是200GB/s,累计为14.4TB/s. 内存容量为单芯片288GB HBM3e,单芯片8颗HBM累计带宽8TB/s.

![图片](assets/3034ca9b4e22.jpg)

Rubin只有整机柜的数据,从物理尺寸上来看,复用了GB300-NVL72的机柜结构, 从散热和单板的密度上来看, 似乎无法真正放下4颗Rubin的Chip, 这里的Chip是以单颗2-Reticle-Size的芯片为单位.我们注意到它和B300的对比

![图片](assets/b5967a9000ae.png)

Serdes在NVLINK5上已经是224G了, NVLink6上短时间448G肯定是无法突破的, 整个Rubin在I/O的地方DieSize占比更大,似乎能够引出更多的NvLink Port. ScaleOut上PCIe Gen7标准还不成熟, 因此估计还是单芯片2个PCIe Gen6x16连接到CX9. CX9估计还是两个独立的800Gbps的pipeline.

内存容量上还是维持在288GB, 升级到HBM4可以使得整个芯片的内存带宽提升到13TB/s.

计算核部分, 单芯片实际上应该包含了4个Rubin的计算Die, 相对于B300平均每个Die的内存容量和带宽反而是下降的, FP4的性能提升估计应该是砍掉很多FP64的单元获得的

而对于Rubin Ultra, 实际上我们看到它是两个Rubin合封在一起的, RubinNVL144的NVlink机柜带宽为260TB/s, 而Rubin Ultra-NVL576为1.5PB/s, 按照NVL144x4倍为1PB/s, 因此单芯片NVLink7的性能相对于NVLink6提升了1.5倍.

![图片](assets/1bb070f637a1.jpg)

内存由于采用了HBM4e, 累计到了1TB, 按照Jensen计数法折算出来, 单个为256GB, 带宽按照4.6PB/s / 576, 单个为8TB/s, PCIe带宽依旧是PCIe Gen6x16.

最终Rubin按照Jensen计数法对齐Blackwell的性能指标如下, 浮点算力(单位TFLOP/s), 内存容量按GB计算, 内存带宽按GB/s计算, 互联带宽按照单向计算(单位GB/s):

FP4

HBM

Mem BW

NVLink

PCIe

B300

15000

288

8000

900

100

Rubin*

25600

144

6500

900

100

RubinUltra*

26000

256

8000

1350

100

Noramlized by Jensen Counting Method

#### 1.2 算力的瓶颈

首先, 我们可以看到一直到2028年的Rubin, Nvidia在数制上都会长期维持在FP4, 暂时不会有一些log8数制出现, 受制于芯片DieSize的影响, 只能逐渐的砍掉FP64的算力来做取舍, 为了能够更好的Overlap TensorCore的计算, 在Attention中的Softmax一类的SFU的算力还要进一步增强. 也就是说单颗芯片的算力上限存在很大的约束了.

从Noramlized的性能来看, 即便是到Rubin Ultra性能增长还是有限的, 毕竟内存墙还在那里.

#### 1.3 互联带宽

然后从互联带宽上来看, Normalized的带宽提升也是有限的, 单芯片的带宽并没有翻倍的增加, 特别是ScaleOut上三代都相同.没有新的材料新的方法这一点上其实也很难演进了. 所以我对单纯的做大NVLink的带宽和规模这件事情上是存在疑虑的, 至少短期内没有新的技术突破, 那么很大程度上就要考了从算法上绕开.

![图片](assets/a91c64add9d7.jpg)

CPO可能是一个趋势, 但是那么多根光纤的可靠性我还是存疑的,

![图片](assets/315cc77f17c2.jpg)

或许密集的波分复用才是一条正路.

![图片](assets/e568300d9f3c.png)

### 2. 模型架构演进

老黄的演讲和放出来的胶片上有一些差异, 对比下面两图, 注意到纵坐标,不同的SeqLen的性能影响差距10倍.

![图片](assets/3223b01f943d.png)

![图片](assets/ae59498a4a4e.png)

实际上这也是我要去做ShallowSim的原因, 几个典型平台下不同SeqLen和不同EP策略的性能如下

![图片](assets/0ae646a3c0bf.png)

另一方面随着SeqLen增长, 主要的影响在Attn的计算上, 相反通信上的budget更多:

![图片](assets/52b9298af98f.png)

也就是说假设一个较长的Context时, 我们可以提供更加Dense的MoE, 其实在另外一篇两年前的文章里[《大模型时代的数学基础(5)-谈谈MoE和Mixtral 8x7B》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488705&idx=1&sn=f7d81af3260550ed231471b97a1f6260&scene=21#wechat_redirect)中就谈到, 过于稀疏的Attn和MoE都会对模型产生影响.

从代数上看，Experts越多则前面路由所造成的负载不平衡情况越弱，但是多个处在不同位置的Token集中在一个专家的概率也会低很多，这样会影响模型的泛化能力。个人还是建议像GPT-4/Mixtral那样选择少量的Experts,同时提高Top-K获得更高的Density，不要过分追求稀疏化才能有更好的表现。

本质上对于Reasoning模型, 不光要考虑预训练模型的训练效率, 更多的会因为后续的RL和实际业务上线的ROI考量上, 需要更快的推理能力. 模型的深度将极大的影响整个计算过程.

我们用ShallowSim来看模型深度的影响

```
args = sb.ModelArgs()c = sb.Config()# generate datadfs = []for nlayers in trange(30,256,2):    args.n_layers = nlayers    df = sb.decode_time_with_ep_list(args,gpu_all_decode,c,fp8_combine=True)    df['index_value'] = nlayers    df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)    df_o.drop_duplicates(subset=['GPU','BatchSize','EP'], keep='first', inplace=True)    dfs.append(df_o)df = pd.concat(dfs)    df.reset_index(inplace=True,drop=True)df.to_csv('perf_vs_n_layers.csv')
```

![图片](assets/68cd58daa947.png)

模型深度对于推理性能的影响非常大, 而特别的注意到H20在模型深度<40层时, 性能还非常不错. 前一篇文章[《从DeepSeek MoE专家负载均衡谈起》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493405&idx=1&sn=3768b760428245600f77e2b636c8c4d6&scene=21#wechat_redirect)也谈到过一个问题, DeepSeek-R1其实在模型层数 > 40层时是欠训练的

![图片](assets/4661a906ae0a.png)

那么我们是否有一个办法, 将模型的层数控制到40附近来考虑呢?

#### 2.1 从计算复杂度和内存的视角

Reasoning模型先打草搞Thinking,然后再回答已经成为主流, 经过这两个月的教育, 好像普通模型不Thinking一下被动消费一下Tokens都不好意思说自己的答案很完善. 因此Test-time的long seq似乎成为了刚需. 但是思考的过程为何要那么多的草稿纸呢?

本质上无论是DeepSeek NSA, 还是Google Titans都在尝试着解决这个问题

![图片](assets/d9f0a114be87.png)

![图片](assets/98bf33f62b6a.png)

其实更多的从计算机体系结构的视角来看待, 如同以前的一篇[《谈谈大模型架构的演进之路, The Art of memory.》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493032&idx=1&sn=206eed2e4127b9971a1e0c380f70b082&scene=21#wechat_redirect) 本质上维持在4000个tokens左右的context, 通过block置换无论是在国外还是国内算力受限场景下都是一条必然的路径

![图片](assets/db111651a43d.png)

不过在上图的基础上, 是否可以通过Attention中增加一些访问RAG的能力, 通过向量数据库的抽取来降低Attn的计算复杂度? 相当于是在Attn计算的过程中, 产生一个基于attn score的向量地址, 然后通过在RAG中提取多个token, 将计算任务变成访存任务. 从这个视角上提高TPS.

这种做法是相对自然的, 例如我们做题的时候, 总归会在脑子里回忆哪些类似的题目可以借鉴参考的, 这样的做法就避免了token by token的generation,而是直接查询出多个tokens补进来. 这样的做法相对于已有的模型通过MoE提取内存的做法或许更加有效, 但是这种场景中需要有大量的RAG数据的KVCache存储和动态加载的过程.

另一方面这样的做法比起token by token的generation稳定性更高, 不过对于预训练过程还需要很多调整. 有一个比较大的优势是可以把Grace Blackwell和Vera Rubin的CPU和GPU之间的高带宽和CPU的算力协同用起来.

#### 2.2 通信带宽的约束视角

对于MoE更多的需要考虑通信带宽的约束, 我们也可以做一个仿真, 在256个路由专家中选择更多的activated_expert

```
# generate datadfs = []for active_expert in trange(8,128,2):    args.n_activated_experts = active_expert    df = sb.decode_time_with_ep_list(args,gpu_all_decode,c,fp8_combine=True)    df['index_value'] = active_expert    df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)    df_o.drop_duplicates(subset=['GPU','BatchSize','EP'], keep='first', inplace=True)    dfs.append(df_o)df = pd.concat(dfs)    df.reset_index(inplace=True,drop=True)df.to_csv('perf_vs_n_active_experts.csv')
```

可以看到TPS的影响也挺大的, GB300-NVL72这些系统在EP72并行时, 无需走ScaleOut网络, 性能尚可, 而其它的无一例外的下降

![图片](assets/b95d913b6606.png)

但是我们仔细来看, 主要是因为alltoall通信无法被overlap带来的影响, 但是也有一个很有趣的结论, 对于H20这些算力阉割卡是有足够的time budget来增加n_activated_expert的.

![图片](assets/9ea61513a311.png)

既然是通信量的约束, 那么自然又想到两个优化方式:

Latent MoE, 在MoE计算时, 首先将Token从dim=7168投射到一个更小的Latent Space, 例如2048, 然后实际的通信带宽的约束就会降低很多.

MoE Group函数的重新设计, 增加n_routed_expert的数量, 降低每个Expert的权重数, 采用多级的Gating, 使得dispatch时降低发出的份数, 例如激活专家为32个时, 是否可以选择先构建一个仅选8个Group的方式, 维持dispatch乘数为8的时候, 然后再到每张卡通过一个二级的Gating选择4个专家进行GroupGEMM计算?

另一个问题也是之前谈到的, 模型深度越深,需要的n_routed_expert数量可能会更多的, 模型本身需要构建成一种PR-MoE的金字塔结构.

#### 2.3 关于训练数据

我们是否可以先按照30层进行预训练, 例如先训练5T tokens, 然后再逐渐冻结前30层然后加深模型的方式进行训练, 使得更深层次的专家更加具有专业性. 同时训练语料的顺序编排,就像我们学习知识也是从容易到难的一个过程, 然后模型的后面层能够吸收到更多的更难的知识. 这也是一个值得去探讨的地方.

### 3. 总结

从GTC25的发布来看, 其实按照Jensen Counting Method进行折算, 还是可以清晰的看到内存墙/通信墙的存在, 对于超大规模ScaleUP的可靠性,其实还有很多的疑虑, 那一张Keynote现场未讲,而事后ppt放出来的一页, 即便是用上最顶级的硬件也无法应对seqlen导致的10倍的性能差异. 数值精度上长期停留在FP4, 算力降精度的路似乎也走到了头. 在这些约束下, 虽然光互连可能会带来一些突破, 但可靠性到真的大规模商用可能还有不少路要走.

因此在这些约束下, 模型架构本身还需要进一步的调整, 利用shallowSim做了一些简单的仿真, 并在文章中提出了几个路径

模型的深度和推理速度的tradeoff, 在MoE模型中可能适当的降低模型的层数才是更好的获取TPS的方法

对于Reasoning Model, 引入一系列RAG填入attn的context中, 类似于内存换页的做法, 通过NSA/Titan维持在seqlen=4000的规模, 并且降低的反复token by token generate的时间和算力消耗是有必要的.

MoE本身的激活专家数和稀疏性的问题需要考虑, 主要是在一些通信的约束下. 当然有一个很有趣的结论,当前H20这些阉割的算力平台反而有更大的Overlap的空间承载更多的activated expert, 结合GB300-NVL72训练和一些低算力卡做推理, 激活专家数可能可以加到32左右.

MoE的专家在模型深度变深后还有一些欠训练的状况, 是否需要金字塔结构的PR_MOE增加模型后面层的专家数, 以及相应的构建一个渐进式的训练方式.

其实最后还有一些观点是, 例如当前的671B的DeepSeek-V3/R1,是否能砍一半的层蒸馏出一个更小的模型用于快速推理, 牺牲一些模型的性能, 保证业务可用性也是一个值得探索的话题.

对于RL相关的问题, 或许后面空一点再来单独列一个话题.