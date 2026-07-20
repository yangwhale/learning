# 继续谈谈MLA以及DeepSeek-MoE和SnowFlake Dense-MoE

> 作者: zartbot  
> 日期: 2024年5月11日 17:43  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489919&idx=1&sn=e0f253eef5637a364defc1ce2051d713&chksm=f99609bdcee180ab09a8ccd5dcaa6aa1333d062afa376436204f8da2fc7bbb1c5dbbda83f8bd#rd

---

智谱也1元一百万个Token了，大模型Token汇率单位都从K到M了，我给这些大模型厂商出一招学PDD来一个9.9元1B Tokens,然后像PDD那样好友砍一刀，顺便把社交网络的邻接图都给补上了，获客成本又低多好呀？

今天继续关注DeepSeekV2模型，前一部分再来谈谈MLA，然后第二部分谈谈MoE并结合Snowflake Dense-MoE来探讨一下未来的一些模型结构

## 1. 再来谈谈MLA

前一篇文章从代数的视角来看待了DeepSeekV2的MLA结构，理论上配合大规模词表在训练时，对于Attention计算时投射到一个低维空间压缩KV存储使用量是可行的。单也有一些疑问，既然可以在推理阶段被其它权重吸收，那么为什么要在训练阶段拆开呢？训练时的算力消耗是否更大了？另一方面共享的压缩KV缓存是否会像MQA那样产生一些问题？省了那么多显存的利用，缺点是什么？于是继续进行了一些分析。

DeepSeekV2的一些超参数如下：

| 参数名称 | 参数符号 | DeepSeekV2 | DeepSeek-67B |
| --- | --- | --- | --- |
| 模型层数 |  | 60 | 95 |
| hidden_size |  | 5120 | 8192 |
| Head数量 |  | 128 | 64/GQA=8 |
| Head维度 |  | 128 | 128 |
| KV压缩维度 |  | 512 | - |
| Q压缩维度 |  | 1536 | - |
| ROPE维度 |  | 64 | - |

我们可以注意到相对于DeepSeek-67B，DeepSeekV2模型的层数更浅，Hidden_Size也更低，同时输入到压缩KV的维度为Head维度的4倍，而 $d_c = 512 \ll d_h n_h = 128 \times 128$

采用MHA时

<!-- 公式存疑,需核对: 原文此处为公式图片, 微信渲染为空白 -->

采用MLA时

<!-- 公式存疑,需核对: 原文此处为公式图片, 微信渲染为空白 -->

训练时额外的计算量和实际参数量

```python
W_q = np.random.randn(n_h,d_h,d)
W_uq,W_dq = np.random.randn(n_h,d_h,d_c_prime),np.random.randn(d_c_prime,d)

path_info = oe.contract_path('ijk,kh->ijh',W_uq,W_dq,optimize='optimal')
print(path_info[1])
```

![图片](assets/9661837e2004.png)

对于KV采用MLA时

$$W^K = W^{UK} \cdot W^{DKV}, \quad W^V = W^{UV} \cdot W^{DKV}$$

训练时额外的计算量和实际参数量

```python
W_k = np.random.randn(n_h,d_h,d)
W_uk,W_dkv = np.random.randn(n_h,d_h,d_c),np.random.randn(d_c,d)

path_info = oe.contract_path('ijk,kh->ijh',W_uk,W_dkv,optimize='optimal')
print(path_info[1])

print("MHA参数量:",W_k.size*2 )
print("MLA参数量:",W_uk.size*2+W_dkv.size)
print("MLA参数量:",(W_uk.size*2+W_dkv.size)/(W_k.size*2))
```

![图片](assets/ecec4836ae5d.png)

我们可以看到在MLA的参数量Q大概为MHA的40%，而KV的参数量仅为MHA的11%，简单的从参数数量作为信息量的视角来衡量，Q的压缩比并不高，而KV相对较高。但另一方面又通过MoE补足的模型整体的参数量。

整个训练FP阶段增加的FLOPS还是很显著的，但在推理时期可以和结合，和结合，实际上并无额外的算力开销。

那么有一个问题，为什么不在训练阶段直接结合和呢？

1. 从梯度更新的角度来看，这样的过程使得优化更加简单

$$\nabla(\varphi\psi) = \psi\nabla(\varphi) + \varphi\nabla(\psi)$$

2. 从Projection的角度来看，KV共享 $W^{DKV}$ 某种意义上对于空间构成了一种约束，Weight Tying 使得模型能够更好的收敛，并且提高其泛化能力，还可以提高模型的稳定性。

3. 在推理需要量化时，也可以BF16把 $W^{UK}, W^{UV}$ 结合计算完后再量化，相应的精度损失应该也可控。

这种做法实际上还使得每个Head输入的维度为MHA的4倍，进一步带来了一些优势。

总体来说MLA实际上是对于Attention稀疏性通过矩阵分解的方式来降低维度，虽然增加了一些训练阶段的FLOP开销，但是对于推理阶段并没有影响，同时Weight-Tying和梯度更新时带来的优势及每个Head输入维度的增加，还可以增加模型的稳定性,并且极大的降低了KV Cache的开销，解决了推理阶段的Memory bound问题。这是一项值得respect的工作，特别是在一众模型很多年都不敢在Attention上动刀时，DeepSeek团队的勇气值得尊敬，而且MLA的处理确实非常的优雅，因此给国内其它一众团队带来了更多的反思。

## 2. 谈谈MoE

对于MoE，我在两年前刚开始设计RDMA拥塞控制协议时就认为它会成为大模型的标配，而对于all-to-all集合通信带来的incast优化也被做为最高优先级的设计目标。下面这篇文章做了一些专门的介绍

《大模型时代的数学基础(5)》

对于MoE模型，主要存在两大难题：

Dynamic Routing and Load imbalance

Tradeoff between model quality(trim token) and hardware efficiency(zero padding)

从代数上看，Experts越多则前面路由所造成的负载不平衡情况越弱，但是多个处在不同位置的Token集中在一个专家的概率也会低很多，这样会影响模型的泛化能力。个人还是建议像GPT-4/Mixtral那样选择少量的Experts,同时提高Top-K获得更高的Density，不要过分追求稀疏化才能有更好的表现。

DeepSeekMoE的工作主要是构建了

从代数的角度来看,MoE计算实际上是对Token进行一次置换群的操作，构成

$$P \circ T \circ concat(Experts) \circ P^{-1}$$

P为一个进行Token位置置换的稀疏矩阵，实际上也构成了代数上的一个置换群的结构

而我们再来看Monarch矩阵，两者代数结构上是相通的，Monarch矩阵定义如下

$$M = PLP^T R$$

其中 $P, P^T$ 是Permutation矩阵，$L, R$ 是Block Diagonal矩阵：

![图片](assets/93c02a91f825.png)

而在MoE中，$P^{-1}$ 是需要对Token进行还原，保证原有的Token顺序输出到下一层。

![图片](assets/2764160f7c3a.png)

对于MoE实现的本质问题是，基于Permutation矩阵后构建的稀疏矩阵乘法如何进行并行

Tutel需要维持每个Expert Capacity相同，采用自适应的方式来处理。MegaBlocks不用EC约束，而直接构造Block based 稀疏矩阵乘法来处理。Google Pathsways则是在模型框架上采用MPMD来构建异步化的Dataflow处理负载不均衡的问题。

DeepSeek-MoE采用了Shared Expert + Routing Expert的策略

![图片](assets/11fd4934b58b.png)

某种程度上也保持了一些Dense度和模型的稳定性，可以通过一些实验数据看到Shared Expert Isolation带来的收益，另一方面细粒度的专家切分(Fine-Grained expert Segmentation)也可以带来收益的提升

![图片](assets/a88db6fef334.png)

在DeepSeekV2中选择了2个Shared Expert以及在160个Routing Experts中激活了4个Expert。

然后针对通信和路由不平衡的问题，增加了Expert-Level/Device-Level及Communication Balance的Loss，以及针对Compute Bound做了一些Token Dropping策略。同时针对All-to-All的incast影响，在Top-K路由函数时，对Expert所在的Device进行了约束，即Device-Limited Routing。

还有一个细节论文没提出，我在查看模型结构时，发现DeepSeekV2第一层没有使用MoE，而是采用了MLP,这也是一个非常巧妙的做法，在靠前的层采用更加Dense的MLP，等Attention经过几层更加稀疏了再加载MoE。从范畴论的角度来看，这种做法是很值得推广的。

```
DeepseekV2ForCausalLM(  (model): DeepseekV2Model(    (embed_tokens): Embedding(102400, 5120)    (layers): ModuleList(      (0): DeepseekV2DecoderLayer(        (self_attn): DeepseekV2Attention(          (q_a_proj): Linear(in_features=5120, out_features=1536, bias=False)          (q_a_layernorm): DeepseekV2RMSNorm()          (q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)          (kv_a_proj_with_mqa): Linear(in_features=5120, out_features=576, bias=False)          (kv_a_layernorm): DeepseekV2RMSNorm()          (kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)          (o_proj): Linear(in_features=16384, out_features=5120, bias=False)          (rotary_emb): DeepseekV2YarnRotaryEmbedding()        )        (mlp): DeepseekV2MLP(          (gate_proj): Linear(in_features=5120, out_features=12288, bias=False)          (up_proj): Linear(in_features=5120, out_features=12288, bias=False)          (down_proj): Linear(in_features=12288, out_features=5120, bias=False)          (act_fn): SiLU()        )        (input_layernorm): DeepseekV2RMSNorm()        (post_attention_layernorm): DeepseekV2RMSNorm()      )      (1-59): 59 x DeepseekV2DecoderLayer(        (self_attn): DeepseekV2Attention(          (q_a_proj): Linear(in_features=5120, out_features=1536, bias=False)          (q_a_layernorm): DeepseekV2RMSNorm()          (q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)          (kv_a_proj_with_mqa): Linear(in_features=5120, out_features=576, bias=False)          (kv_a_layernorm): DeepseekV2RMSNorm()          (kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)          (o_proj): Linear(in_features=16384, out_features=5120, bias=False)          (rotary_emb): DeepseekV2YarnRotaryEmbedding()        )        (mlp): DeepseekV2MoE(          (experts): ModuleList(            (0-159): 160 x DeepseekV2MLP(              (gate_proj): Linear(in_features=5120, out_features=1536, bias=False)              (up_proj): Linear(in_features=5120, out_features=1536, bias=False)              (down_proj): Linear(in_features=1536, out_features=5120, bias=False)              (act_fn): SiLU()            )          )          (gate): MoEGate()          (shared_experts): DeepseekV2MLP(            (gate_proj): Linear(in_features=5120, out_features=3072, bias=False)            (up_proj): Linear(in_features=5120, out_features=3072, bias=False)            (down_proj): Linear(in_features=3072, out_features=5120, bias=False)            (act_fn): SiLU()          )        )        (input_layernorm): DeepseekV2RMSNorm()        (post_attention_layernorm): DeepseekV2RMSNorm()      )    )    (norm): DeepseekV2RMSNorm()  )  (lm_head): Linear(in_features=5120, out_features=102400, bias=False))
```

当然DeepSeek-MoE由于受到一些All-to-All通信长尾的影响，做了一些约束，并且针对Imbalance的情况进行了处理，但其实还有更好的MoE处理方式，接下来我们来看看SnowFlake Dense-MoE

## 3. SnowFlake Dense-MoE

我们注意到Attention计算是一个Compute-Bound算子，而在MoE中是一个带同步阻塞的通信密集型算子，Infiniband和当前的RoCE针对All-to-All incast的拥塞控制都有一些缺陷。 而SnowFlake做了一件非常棒的事情，利用Attention计算隐藏MoE延迟，即Dense-MoE。

SnowFlake 关于Arctic模型训练的内容都放在了《SNOWFLAKE ARCTIC COOKBOOK》[1]，它背后的模型训练有几位来自微软DeepSpeed团队的Leader，所以你可以看到他们对通信的优化也非常的优雅。

第一篇文章中针对MoE的一些超参数进行了分析，Top-K=2更有效，然后每一层放置MoE也更有效，最关键的一个做法是构建了Dense-MoE，将MoE和Transformer模型并联的方式

![图片](assets/c40c4d68e1c8.png)

这样方式来看，一部分数据通过Attention走到了原有的一个Dense-FFN层，这里可以看作是DeepSeek-MoE中的Shared Experts，每个都要走，另一方面，然后SnowFlake采用了128选2的方式选择专家进行路由，直接从Attention之前拉出一路数据，这里可以等效的人为是bypass了Attention，有一些类似于Mixture-of-Depth的原理，或者是相当于从上一层的输出直接进行MoE，而这些长尾延迟被主Attention的Dense路径隐藏了。

SnowFlake还采用了MoE-Gating-Fusion和MoE-Gather kernel的方式进行优化

![图片](assets/ee60e44e14e6.png)

然后还通过一个令牌桶的算法来保证专家负载均衡

![图片](assets/88690d14b751.png)

另外对于专家的分布也采用Zero-Stage-2的方式进行Sharding

![图片](assets/85d92c4dbe71.png)

## 4. 展望

MLA+Dense-MoE或许是未来的一个很好的方向，MLA的算力消耗覆盖了Dense-MoE的延迟，而Dense-MoE Sharding Zero-Stage-2的方式又可以保持相对通信平衡的时候又降低对网络全互联的带宽压力。至于路由平衡，Dense-MoE采用了通信网络上常见的根据Compute Capacity提供令牌并随机Mask丢弃的做法，同时还避免了MegaBlock这样的稀疏矩阵乘法带来的效率下降。

然后这样的方式和Google Mixture-of-Depths也有异曲同工之妙，相当于是Dense路径上的MLP相当于Shared，而路由到Route-MoE的时候隐含了MoD Bypass了Attention的计算。

参考资料

[1] 
SNOWFLAKE ARCTIC COOKBOOK: https://www.snowflake.com/en/data-cloud/arctic/cookbook/