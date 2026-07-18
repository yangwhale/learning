# 大模型时代的数学基础(5)-谈谈MoE和Mixtral 8x7B

> 作者: zartbot  
> 日期: 2023年12月12日 10:21  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488705&idx=1&sn=f7d81af3260550ed231471b97a1f6260&chksm=f9960403cee18d15b1f80a4b9a92a928cfb9c1db68c3050560d49809d09c8f61ffe65646c792#rd

---

### TL;DR

今天来专门谈谈MoE,Mixtral 8x7B大概是这几天的大热门吧？32K token的Context Length，类似于13B Dense模型的推理速度，性能普遍超越LlaMA2 70B

![图片](assets/e64b40914481.png)

GPT-4已经在使用MoE，而Mixtral引爆的关键点是，它算是第一个开源的MoE大模型，有8个Experts，并且Top-K=2，不知道未来MoE这条路能火多久，或许它将和多模态一起成为2024年的主要卷点，但这周肯定能火，而我们关注的是它可能给AI Infra带来了蛮多的新的需求和挑战。

从通信上看MoE原语Router/Switch Transformer，还有Top-K的Multicast。但是从代数的角度来看,MoE计算实际上是对Token进行一次置换群的操作，构成

P为一个进行Token位置置换的稀疏矩阵，如何从代数编码上让这些计算获得更好的性能是本文要探讨的,当然也会顺带提一下SPMD在MoE上带来的困难以及Tutel/DeepSpeed-MoE和Mixtral所使用的MegaBlcoks的算法，还有Google Pathways的工作如何应对稀疏矩阵的计算和进行二次调度。

除了模型架构以外，还有一些问题就是

Expert_Capacity如何设置?

MoE是不是Transformer每层MLP都要替换? 还是把多个模型作为Expert?

专家要多少个合适?

专家参数大小该如何考虑?

非对称的专家是否可以考虑，例如我们把Tokens通过路由器映射到一个线性空间，然后通过BVH**对空间划分降低复杂度

### 1. MoE概述

MoE模型的开山之作来自于1991年《Adaptive Mixtures of Experts》[1]大概的思路是一个模型处理不同领域多种数据时的收敛和泛化能力都会比较差，然后通过门控网络选择不同的模型子块来解决问题。

把MoE再次拿出来用的是2017年的一篇文章《Outrageously large neural networks: The sparsely-gated mixture-of-experts layer》[2] 在2016年Transformer出来之前，RNN/LSTM这样的模型深度无法做的太深，模型的参数量决定了整个神经网络能吸收的信息量，如何修改网络架构来获得更多的参数？于是论文将MoE模型引入到了RNN的网络中

![图片](assets/2834ad109c34.png)

它把一个大的FF层拆分成多个子层，我们把拆分后的子层称为Expert，训练数据通过Gating Network路由，可以选择K个子层(Expert)，这样的模型架构被称为Mixture-of-Experts(MoE)。到了大模型时代来看，引入MoE去解决的问题很清楚，主要是针对Transformer的FFN层带来的资源限制，其计算复杂度为,因此这样的方法也用在了Transformer类的模型上，《GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding》[3]

![图片](assets/2f49ed11b3ad.png)

基于All-to-all的集合通信机制出现在多机多卡的分布式训练中，这一点和很多读者初略的直观印象有所不同。可能我们会直观的认为，哪个Token通过Router决定了那个Expert，那么就直接通过NVLINK内存拷贝或者RDMA发送过去就好了，为什么需要AlltoAll的通信范式呢？实际上这来自于GPU SIMT进行细颗粒度的稀疏计算性能较差以及当前模型训练过程中的SPMD机制带来的同步。

#### 1.1 MoE的路由和负载均衡机制

MoE的整个计算过程如下图所示：

![图片](assets/8201c7e81733.png)

`Routing`: 也被称为Gating Network Token通过和Router Weights矩阵相乘得到一个路由分数矩阵，然后Softmax来决定该Token需要发送给哪些Expert

![图片](assets/624a20e8b60d.png)

需要注意的是Gating network很容易倾向于选择某几个效果好的专家，而抛弃其它的不使用，这样会导致某几个专家的单个Batch**实际处理的token数量增多，显存需求也增大，并且这种负载不均衡会在训练过程中进一步被强化，通常会在计算总体Loss的时候进行一个Loss的权重约束

一个Token可以选择发送给多个Experts，具体取决于Top-k参数，也可以通过一些hash函数或者强化学习等方法训练出来路由决策。动态路由决策的关键是如何实现负载均衡？即每个Expert在每次Batch的运算量是近似的，显存消耗也是近似的。

`Permutation`: 由前一步Router会产生一个Expert Indices，此时GPU会根据这个决策矩阵构建本地的置换Token位置的后的临时矩阵，然后通过All-to-All通信发送给不同的Experts所在的GPU进行运算,此时需要注意一个问题是，每个Expert单个Batch能够处理多少个Token，因此引入了一个Capacity_Factor的超参数

![图片](assets/3632299d6523.png)

Permutation根据Routing决策逐渐将Token放到对应的Expert上，Expert根据Capacity_Factor的超参数定义相应的Slot用于存放Token，如果Capacity_Factor太小，当slot被放满后，多余的token将被丢弃，导致模型的性能受损，而如果Capacity_Factor太大会导致Slot中有zero padding value 构成的unused slot，造成算力资源浪费。

`Computation`: 主要是Permutation完成重排序的Token数据输入到每个Experts的MLP中进行计算。

`Un-Permutation`：实际上是Permutation的逆运算，把Token从专家返回给原来的节点，然后继续向下游处理。

#### 1.2 MoE实现的难题

主要存在两大难题：

Dynamic Routing and Load imbalance

Tradeoff between model quality(trim token) and hardware efficiency(zero padding)

从代数上看，Experts越多则前面路由所造成的负载不平衡情况越弱，但是多个处在不同位置的Token集中在一个专家的概率也会低很多，这样会影响模型的泛化能力。个人还是建议像GPT-4/Mixtral那样选择少量的Experts,同时提高Top-K获得更高的Density，不要过分追求稀疏化才能有更好的表现。而计算负载度和不均衡的问题MegaBlocks的处理还是值得参考的。

#### 1.3 从置换群看MoE

从代数的角度来看,MoE计算实际上是对Token进行一次置换群的操作，构成

P为一个进行Token位置置换的稀疏矩阵，实际上也构成了代数上的一个置换群的结构，而我们再来看Monarch矩阵，两者代数结构上是相通的，Monarch矩阵定义如下

其中是Permutation矩阵，是Block Diagonal矩阵：

![图片](assets/28796334213b.png)

而在MoE中，是需要对Token进行还原，保证原有的Token顺序输出到下一层。

![图片](assets/95ca74c40947.png)

对于MoE实现的本质问题是，基于Permutation矩阵后构建的稀疏矩阵乘法如何进行并行

Tutel需要维持每个Expert Capacity相同，采用自适应的方式来处理。MegaBlocks不用Expert Capacity约束，而直接构造Block based 稀疏矩阵乘法来处理。Google Pathsways则是在模型框架上采用MPMD来构建异步化的Dataflow处理负载不均衡的问题。

### 2. MoE实现

MoE实现对于当前的框架来看是非常有挑战的，稀疏性的结构虽然可以带来更好的性能，但同时针对GPU这些细颗粒度的稀疏计算对GPU这些芯片的算力并没有有效的发挥。我们在这里先简略探讨一下.

#### 2.1 Mixtral 8x7B架构

我们通过HuggingFace下载了Mixtral 8x7B模型，模型架构如下：

```
from transformers import AutoModelForCausalLM, AutoTokenizermodel = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")print(model)MixtralForCausalLM(  (model): MixtralModel(    (embed_tokens): Embedding(32000, 4096)    (layers): ModuleList(      (0-31): 32 x MixtralDecoderLayer(        (self_attn): MixtralAttention(          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)          (rotary_emb): MixtralRotaryEmbedding()        )        (block_sparse_moe): MixtralSparseMoeBlock(          (gate): Linear(in_features=4096, out_features=8, bias=False)          (experts): ModuleList(            (0-7): 8 x MixtralBLockSparseTop2MLP(              (w1): Linear(in_features=4096, out_features=14336, bias=False)              (w2): Linear(in_features=14336, out_features=4096, bias=False)              (w3): Linear(in_features=4096, out_features=14336, bias=False)              (act_fn): SiLU()            )          )        )        (input_layernorm): MixtralRMSNorm()        (post_attention_layernorm): MixtralRMSNorm()      )    )    (norm): MixtralRMSNorm()  )  (lm_head): Linear(in_features=4096, out_features=32000, bias=False))
```

可以看到词汇表是32000，维度(dim)4096，总共有32层，隐藏层维度(hidden_dim) 14336,有8个 Expert，每个Token选择2个Expert，采用了MegaBlocks的block_sparse_mode,同时也使用了RMSNorm和RotaryEmbedding， 手上也没有好的GPU用4xA10-24GB加载了一下

```
from transformers import AutoTokenizerimport transformersimport torchmodel = "mistralai/Mixtral-8x7B-Instruct-v0.1"tokenizer = AutoTokenizer.from_pretrained(model)pipeline = transformers.pipeline(    "text-generation",    device_map="auto",    model=model,    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},)
```

推理时的显存消耗：
![图片](assets/6901f69b563a.png)

```
messages = [{"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."}]prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)print(outputs[0]["generated_text"])
```

![图片](assets/92e53235ff5b.png)

#### 2.2 MegaBlocks MoE实现

Mixtral基于Megablocks实现的MoE，Megablocks来自于论文《MegaBlocks: Efficient Sparse Training with Mixture-of-Experts》[4]

首先我们从业务需求上来看，减少Experts数量，增加TopK的数量，并且增大每个专家的参数规模是不能损失模型性能的一个更好的实践。那么带来的代价就是Token负载不均衡的问题以及Token由于Expert Capacity导致drop性能损失的问题。因此对于MoE模型来说，打破原来动态路由负载均衡的限制是必须要去做的了，MegaBlocks采用基于Block的稀疏矩阵乘法的方式来构建，当然像Google还可以用Pathways来处理，后面会单独介绍。如下图所示：

![图片](assets/ad4ee1984e69.png)

通过对矩阵采用BCSR+BCOO的编码，并将转置的信息也编码在其中，并配合高效的稀疏计算核来实现：

![图片](assets/127a24547072.png)

#### 2.3 Tutel MoE

来自于微软的工作《Tutel: Adaptive Mixture-of-Experts at Scale》[5]

相对于MegaBlock压根就不管Expert Capacity(EC)的事情，直接稀疏矩阵乘法去干。而Tutel则考虑采用自适应的并行策略来处理，其核心机制是自适应并行性切换，能够在每次迭代时动态切换并行策略，而不会引入任何额外的切换开销。另外针对All-to-All通信Tutel对NCCL有一些优化，当然这些都是工程实践上的问题，和代数结构无关就不多展开了。

#### 2.4 Janus

《Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models》[6] 在SIGCOMM，而不是MLSys里面谈MoE有点意思，大致的思路就是以前的工作是以Expert为中心，现在以Token为中心，通过参数服务器把Experts的参数射向Token，然后本地计算

![图片](assets/d8eaee677c82.png)

### 3.MoE演进

#### 3.1 Google Pathways

《Pathways: Asynchronous Distributed Dataflow for ML》[7] Pathways是一个大规模的加速器(TP)编排框架，用于探索新的系统和ML研究思路，相对于现在的基于MPI**构建的SPMD(Single Program multiple Data)模型(注：即每个节点都执行相同的代码程序), 程序通常在多机之间执行需要有多个控制平面进行周期性的同步，节点失效和资源细粒度的调度编排相对困难，无法实现很高的资源利用率，同时又针对当前的模型框架过度专用化的设计，无法适应未来MoE等稀疏模型研发。
3.1.1 Single Controller vs MultiController
这是Pathways框架的关键，我们需要区分SPMD/MPMD/Single-Controller/Multi-Controller的概念，以及业务对稀疏性和灵活编程的需求，这样才能了解Pathways的价值。
3.1.2 TF1 Single-Controller
TF1(Tensorflow v1)通过Python构造一个静态的计算图，然后通过Session.run的方式交给runtinme， runtime系统就是一个single-controller(Master)执行计算图的编译和切割，然后把每个子图发送到对应的Worker执行：

![图片](assets/3dc528430ee2.png)

但是这样的框架有一定问题是worker和controller之间的通信带来的设备上的大量空闲，所以最简单的数据并行实现上TF都和其它框架有明显的劣势。
3.1.3 Multi-Controller
业界逐渐开始引入MPI的方式，例如Uber Horovod和后来的Pytorch等用来解决单个controller的问题，每个host都是一个控制进程，且运行相同的代码，通过MPI进行分布式协同，这样就成为了SPMD(Single Program Multi Data)的运行方式

![图片](assets/315caaa7c483.png)

每个host只处理自己rank相关的数据，每个控制器通过PCIe向加速器发送任务指令避免了single-controller通过DCN**的缺陷， 因此只要集群内设备间集合通信完成足够迅速，整个加速器基本上就能完全跑满。
3.1.4 MPMD
由于模型规模的扩大，张量并行和数据并行(DP+TP)还可以继续在MPI的框架下工作，但是扩展到流水线并行，则设备同步执行压力和慢节点影响就变得更加明显，且不同的设备处在不同的流水线的阶段，执行的代码也有不同，无法完全做到同步。另一方面MoE和其它稀疏算法会导致每个设备算子的运行时间会不同，强制SPMD对齐反而会带来效率的影响。同时整个集群同步执行带来的电源功耗同时达到影响。动态路由，异步执行的MPMD成为一条值得探索的技术路线

![图片](assets/a4f9a09683a8.png)

如图所示，每个worker工作流是不同的，worker 2产生的中间结果要发送给Worker3， worker2和worker1中间会进行一次集合通信同步，然后worker3的结果要发送给worker1等。而在编排调度时需要保证Send/Recv的顺序。
3.1.5 综上所述
虽然Multi-Controller模式性能较好，但对于Pipeline和计算稀疏性等问题处理能力较差，同时由于独占资源和同步处理机制，对于节点失效处理较为困难，通常采用checkpoint的机制整个任务停止后重新加载。SingleController和GPU/TPU加速器通过DCN通信，延迟大于SPMD模式下的本地PCIe通信，因此无法达到很高的利用率，节点间的协同和消息传递容易导致计算资源的浪费。但是Single-Controller也有其优势，编程灵活性更高同时对资源的虚拟化能力更强。

Pathways采用的方法是既采用了Single-Controller 的灵活可编程能力和资源虚拟化能力，又兼顾了Multi-Controller的高性能，最后构建了基于Dataflow模型的分片并行处理和基于Gang-Scheduler的异步调度机制。

![图片](assets/3e7ea6386dcb.png)

#### 3.2 模型架构演进

从路由机制上，也有蛮多的研究
![图片](assets/14c2255c9078.png)

另外还有基于时间域逐渐演进Experts的Lifelong Language Pretraining

![图片](assets/25f0ad4a8aed.png)

除了模型架构以外，还有一些问题就是Expert_Capacity如何设置? MoE是不是Transformer每层MLP都要替换? 专家要多少个合适? 专家参数大小该如何考虑? 如何利用MoE更好的适配低算力低带宽的设备，例如4090或者更低端的卡？如何通过代数编码上和软硬件协同的体系结构上来设计使得MoE训练推理更加有效？

#### 参考

[1]
Adaptive Mixtures of Experts: https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf,
[2]
Outrageously large neural networks: The sparsely-gated mixture-of-experts layer: https://arxiv.org/pdf/1701.06538.pdf,
[3]
GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding: https://arxiv.org/pdf/2006.16668.pdf,
[4]
MegaBlocks: Efficient Sparse Training with Mixture-of-Experts: https://arxiv.org/abs/2211.15841,
[5]
Tutel: Adaptive Mixture-of-Experts at Scale: https://arxiv.org/abs/2206.03382,
[6]
Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models: https://dl.acm.org/doi/pdf/10.1145/3603269.3604869,
[7]
Pathways: Asynchronous Distributed Dataflow for ML: https://arxiv.org/abs/2203.12533,