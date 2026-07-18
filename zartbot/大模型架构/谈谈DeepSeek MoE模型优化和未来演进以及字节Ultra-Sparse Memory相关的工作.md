# 谈谈DeepSeek MoE模型优化和未来演进以及字节Ultra-Sparse Memory相关的工作

> 作者: zartbot  
> 日期: 2025年2月15日 03:49  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493218&idx=1&sn=f394f39a4346fd09a19008a53d0a8022&chksm=f995f6a0cee27fb678cc9ed19c9687593843c568a5ce685e5c94f3cd0e8cec62ce136df12c08#rd

---

大概一个月前发了一篇[《一个关于MoE的猜想》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493019&idx=1&sn=25a87af51b9077d50b40685d6987ca71&scene=21#wechat_redirect),  最近优化DeepSeek MoE推理的时候, 发现这是一个值得去解决的问题, 从算法上, 这是DeepSeek-V3/R1 MoE的一个不足之处, 值得进一步去分析和提高效率. 后文顺便也分析了一下字节的Ultra-sparse Memory Network的工作.

### 1. DeepSeek-MoE Profiling

其实任何算法的效率提升, 无非就是在计算/通信/存储上取得平衡. 我们以DeepSeek MoE为例. 它的一个Expert结构如下:

```
import torchfrom torch import nnimport torch.nn.functional as Fclass Expert(nn.Module):    def __init__(self, dim: int, inter_dim: int):        super().__init__()        self.w1 = nn.Linear(dim, inter_dim)        self.w2 = nn.Linear(inter_dim, dim)        self.w3 = nn.Linear(dim, inter_dim)    def forward(self, x: torch.Tensor) -> torch.Tensor:        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

按照DeepSeek-R1定义的dim = 7168, moe-inter-dim=2048, 以及论文中提到的对256个Tokens Batch处理, 进行算力评估

```
dim = 7168inter_dim = 2048tokens = 256e = Expert(dim, inter_dim)from ptflops import get_model_complexity_infoinput_tokens = (1,tokens,dim)flops, params = get_model_complexity_info(e, input_tokens, as_strings=True,print_per_layer_stat=True)Expert(  44.05 M, 100.000% Params, 11.28 GMac, 99.995% MACs,   (w1): Linear(14.68 M, 33.329% Params, 3.76 GMac, 33.328% MACs, in_features=7168, out_features=2048, bias=True)  (w2): Linear(14.69 M, 33.341% Params, 3.76 GMac, 33.340% MACs, in_features=2048, out_features=7168, bias=True)  (w3): Linear(14.68 M, 33.329% Params, 3.76 GMac, 33.328% MACs, in_features=7168, out_features=2048, bias=True))
```

单个专家要加载44.05MB的参数, 整个模型的专家参数为(256+1) x 60层 x 44.05M = 663B, 几乎占到了98%的参数量. 对于训练而言, 每个Batch大, 加载数据和加载参数的比值影响比较小. 但是对于推理单个Batch, 加载44.05MB参数仅处理256个Token数据为256 x 7168B = 1.8MB数据,开销非常大, 而实际运算为11.28GMAC, 开销非常小, 可以说细粒度MoE的算法和FP8本质上先解决了国内算力受限的问题.

但是访问内存和通信的问题还需要进一步解决, 因此DeepSeek-V3才提及了需要做Prefill Decode分离,以及在Decoding集群需要按照EP320并行, 甚至是从袁进辉老师的一段话可以知道, 梁总为啥要推荐性能最好需要80台, 通过EP并行获得更好的Data Locality.

![图片](assets/176b605c3019.png)

### 2. MoE演进

因此, 渣B一直在提一个问题就是, 能否进一步优化来降低MoE阶段的数据/参数访存比? 所以才会提到2层Gate Function的做法

![图片](assets/5d5f85ef07d3.png)

所以期望的方式是构造2级的Routing Gate, 使得本来Attention里面携带的信息通过两个Gate找到矩阵中(x,y)对应的某个Expert,或者多个expert. 例如我们继续按照Hidden-dim切分成16片, 然后也实现一个256选8的MoE算法, 这样每个专家的inter-dim维度应该是可以继续降低的.同时整个模型的参数量还可以进一步提升.

其实某种意义上来看, 这就是最早来自于PKM《Large Memory Layers with Product Keys》[1]相关的工作, 也是字节Ultra-Sparse Memory Network一开始讲的一个图

![图片](assets/0a17982af7a0.png)

其实你会看到这本质上和渣B提到的通过两个Gate找到矩阵中(x,y)对应的某个Expert是一致的. 而字节论文的描述里, 谈到的其实更适合于用渣B另外一篇[《谈谈大模型架构的演进之路, The Art of memory.》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493032&idx=1&sn=206eed2e4127b9971a1e0c380f70b082&scene=21#wechat_redirect)中的描述

![图片](assets/dc1966e3c349.png)

通过这样的视角来看, 将MoE作为内存层, 然后两级页表的方式作为Gating函数, 从这个视角上看, 大模型本身就可以构造一个`自己能够产生代码运行的通用计算机架构`.

### 3.字节Ultra-Sparse Memory Network

原始的PKM(Product Key-Based Memory)算法如下: 它将内存层抽象成为两段Key  和Value .  是一个查询向量. 同样通过一个非线性激活算子计算获得一个分数, 然后乘以V的到结果

当Memory size加大时, 采用稀疏的方式获取Row和Column的topM score计算

然后再通过row/col score计算grid score, 并得到输出

然后PKM也采用了类似的Multihead机制, 有多个Key来检索共享的Value.

字节UltraSparse Memory(USM)的工作对PKM有几个改进:

移除了Ouput时的Softmax操作

对Query和Key做了LayerNorm来提高训练的稳定性.

采用了逐渐衰减的学习率.

相对于PKM采用线性层产生Query, 字节在该线性层前面添加了一个causal depthwise的卷积来增强Query信号.

进一步采用类似于Group Query Attention的机制, 共享Key来降低计算复杂度.

减半, 然后翻倍了Value的数量.

相对于PKM每一层添加内存层, 在USM架构上,采用了多个Transformer block中间插入Memory层的做法.

![图片](assets/bd4f91665a57.png)

主要原因是, 随着内存的增加, Gating Score函数很难查询到正确的Value, 然后大规模训练的时候存在通信和计算负载不均衡的问题. 但是另一方面PRM还有一个问题, 对于 Row和Col各自取topM,然后再相加取TopM获得grid score的做法无法满足对内存访问的多样性. 因此期望能够将Row和Col协同在一起运算, 即Tucker Decomposed Query-Key Retrieval (TDQKR). 如下所示

其中 为一些可学习的参数构成tucker core, 但是效率不高, 因此又引入了对 C的SVD分解

![图片](assets/532f9ea9f94c.png)

然后通过特征值约束构建了一个辅助损失函数

另一方面的技术是Implicit Value Expansion(IVE), 主要是大量的内存访问时, 训练期间维护大的内存表开销也很大. 本质上就是一个虚拟内存, 个人觉得这里的做法挺复杂的, 还有优化的空间.

![图片](assets/f45f5982497f.png)

最终如文章开头所说, 相对于MoE推理的延迟降低, 并且访问内存的带宽加大, Validation Loss近似.

![图片](assets/d96105e9c043.png)

### 4. 进一步的分析

字节的论文里只有一个小模型的测试

![图片](assets/9e596318dec6.png)

从消融实验的结果来看, 似乎这些工程上的优化并没有很大程度的改善Loss

![图片](assets/1230d364f0bb.png)

另一方面这些复杂的工作, 个人猜测对于超过600B的模型训练是否收敛, 以及训练效率上的损失有多大还值得进行进一步的分析. 另一个观点是这个方案整体太复杂了, 特别是对未来的一些Post-Train的RL工作流的影响可能会非常大.

其实DeepSeek的AuxLoss-Free, 通过Gating函数增加Bias的方法来做路由选择, 然后在hidden-dim维度上再做一次Gating可能是一个更好的改进方式.  然后结合访问内存, 做一些基于Expert indecs的Prefetch, 然后是否要基于ExpertGroup做一些多级页表, 这个涉及到算法和GPU微架构以及训练框架的协同, 这一点也是非常重要的.

### 5. 总结

算法的角度如何匹配硬件架构, 训练和推理时的访问内存的Trade-Off, DeepSeek MoE第一步解决了对大算力的依赖, 未来模型算法和基础设施的整合, 进一步解除对网络和内存访问的依赖, 是一条我们更需要解决的路.

但是, 一定要保证算法本身足够简单才能Scale, 也对未来的一些Reasoning的RL工作流有更好的算法稳定性. 接下来几天如果工作稍微空闲一点, 还会基于范畴论的视角来分析一下DeepSeek-R1相关的强化学习的工作,  个人的观点是还要进一步结合和通盘考虑来做Infra的优化, 这是一个算法和Infra紧密协同的过程, 不能单独的为了MFU或者解除Memory Bound优化, 训练和推理的性能也要协同, 有时候还得牺牲一些训练的MFU来追求更高的推理效率, 毕竟推理的计算规模未来会远超过训练.

更进一步, 从DeepSeek的MoE也大概看出了Nvidia GB200 NVL72的设计初衷, 为什么要加大内存的容量和带宽, 为什么要更高密度的NVL互联, 但是我并不觉得这是一条对的路径, 算法上解除对大带宽网络的需求和延迟约束才是真正正确的一条路. 

参考资料

[1]
Large Memory Layers with Product Keys: https://arxiv.org/pdf/1907.05242