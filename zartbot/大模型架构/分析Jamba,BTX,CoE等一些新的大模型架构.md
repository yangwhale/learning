# 分析Jamba,BTX,CoE等一些新的大模型架构

> 作者: zartbot  
> 日期: 2024年4月2日 05:09  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489217&idx=1&sn=b4f787fc485f4dfa51eb180ccc352264&chksm=f9960603cee18f15fb227d26c4b872443ee6f41772d32277fabdbe09b56073f0afcc6c12cb51#rd

---

时刻跟踪大模型演进的趋势对于AI基础设施的架构师而言是一门必修课，因为涉及到通信范式算力规划等一系列问题， 作为体系结构相关的架构师，通常需要在算法和算力间有足够深的洞察，从而去寻求更加简单的通用解法。因此新开一个专题，不定期的探讨一些有价值的模型。

正如我在整个《大模型时代的数学基础》系列文章中反复阐述的一个观点，大模型的Composable能力是非常重要的，这将极大的泛化模型的部署，也是通向AGI的必经之路。

[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

本文主要分析几个模型：

基于SSM的Mamba**结构和Transformer结构融合解决长序列问题

Branch-Train-Mix解决MoE训练的复杂度

参数空间和数据流空间融合

SambaNova的Composable of Experts(CoE**)

## 1. Mamba+Transformer混合支持长序列的Jamba

Kimi前段时间在国内卷了一下长文本大模型，似乎热度已经开始退去了，对于模型的Sequence Length，Transformer的时间复杂度为，因此工业界探索了多种改进方法，例如Mamba。详细的内容在几个月前有一篇文章总结过

[《大模型时代的数学基础(4)》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&chksm=f996046acee18d7c687403c557a6e30155ba0c04cca7e897de3126e88a5d3ba3a2c2fe0507bd&scene=21#wechat_redirect)

而最近的一个工作是AI21Labs的《Jamba:A Hybrid Transformer-Mamba Language Model》[1]

通过组合Mamba/MoE和Transformer层，构建模型

![图片](assets/55fe7c332a28.png)

优势是对256K token的KV cache用量非常小

![图片](assets/46bf25ac7ab5.png)

推理速度也不错

![图片](assets/06b11c32754c.png)

模型性能也凑合

![图片](assets/a6abd38f815d.png)

## 2. 分支训练混合Branch-Train-Mix

这是Meta的一个工作，在2022年有一个版本BTM《Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models》[2]大概的思路就是现在大模型需要在多个领域刷分，然后呢相互之间的参数会有影响，作者提出了一种并行训练策略，不同的训练子集去独立训练专家大模型(ELM). 从而消除了当前训练LLM所需的大量多节点同步, 每个ELM专长于不同的文本领域，如科学或法律文本。这些ELM可以被添加或移除以更新数据覆盖范围，通过集成以泛化到新领域，或通过平均以高效推断为目的回缩到单个LM。实验表明，当控制训练成本时，与GPT风格的Transformer LM相比，BTM在领域内和领域外的困惑度方面有所改善。

![图片](assets/5c4771a9dca1.png)

BTM以无同步方式进行LLMs的极度并行训练，以提高预训练的吞吐量。该方法首先创建种子LLM的多个副本，然后分别在不同数据子集上单独训练每个副本。这产生了多个独立的LLM，它们之间不共享任何参数，且每个LLM都是其自身数据分布（如知识领域、语言甚至模态）的专家。测试时，将输入提示分类到一个或多个领域中，然后由相应的专家模型组合形成最终输出，预测下一个Token。

虽然这种方法使训练更高效，但其主要缺点是缺乏统一的单一模型，无法进行进一步的监督微调（SFT）或基于人类反馈的强化学习微调（RLHF），这两种方法都能进一步提升性能，并且是构建对齐LLMs的关键步骤。

当然另一种方法就是完全同步训练的MoE，虽然这种方法被广泛采用，但全连接all-to-all的通信成本非常高，特别是对很多基础设施都需要针对All-to-All优化bisection bandwidth，建设成本很高。

因此出现了一个新的工作BTX《Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM》[3] 通过结合MoE和Branch-Train-Merge的各自优势，并同时缓解各自的劣势。

算法是通过类似Branch-Train-Merge方法的方式分别训练多个专家LLM，然后使用MoE架构将这些专家整合到单一模型中。具体来说，将所有专家LLM的前馈子层合并到每一层的单个MoE模块中，路由器网络在每个令牌处选择使用哪个前馈专家。我们通过简单地平均其权重来合并专家LLM的其他模块，包括自注意力层。然后继续训练，对所有组合数据进行MoE微调，以便路由器能够学会混合专家前馈模块。

![图片](assets/c6608418e975.png)

相比于MoE，BTX的主要优势在于专家训练是极度并行且异步的，降低了通信成本并提高了训练吞吐量。与Branch-Train-Merge相比，最终的BTX模型是一个统一的神经网络，可以像任何其他标准LLM一样进行微调或使用。与种子模型相比，最终的BTX模型的推理FLOPs**不会显著增加，尽管参数数量大得多，但由于其稀疏激活，仍具有较高的计算效率。

最后作者使用Llama-2 7B作为种子模型，针对数学、代码和维基百科等领域数据的不同子集训练专家LLM。将原始Llama-2 7B权重作为第四个专家添加后，对组合的MoE模型进行相对较短时间的微调，与预训练过程相比。由此得到的BTX模型在各种领域的任务上都显著超越了种子模型，特别是在数学和代码相关任务上缩小了与专门模型的差距，同时在专门模型因灾难性遗忘而在原生能力上表现不佳的地方保持了性能。

![图片](assets/ae8edc746f86.png)

BTX在所有任务上均优于BTM，显示了通过MoE微调学习路由的好处。与纯粹的MoE训练（如稀疏升级）相比，BTX具有更高的计算效率、更高的训练吞吐量以及在不同领域任务间更均衡的性能。

![图片](assets/58dad57978b2.png)

## 3. 模型融合

这里有一篇日本学者的文章《Evolutionary Optimization of Model Merging Recipes》[4],很有趣的一个研究，日本在整个大模型领域并没有出现一个Foundation Model，由于其本土能源等一系列限制似乎也没有大规模训练集群的建设。 因此他们的研究更多的是一种进化的观点，通过自动发现多种开源模型的有效组合，无需大量额外训练数据或计算资源即去利用它们的集体智慧。一种模型合并的方法是加权平均多个基于基础模型的微调子模型，但是这种线性变化效果并不好，泛化能力影响也很大。

实际上我们需要的一种统一的框架，能够从一组基础模型中自动生成合并模型，并确保所生成的合并模型性能超越集合中任意单一模型。这篇文章核心是应用进化算法，用以优化模型合并过程中涉及的复杂细节。首先将合并过程细分为两个相互独立、正交的配置空间，并分别分析它们各自的影响。在此分析基础上，我们进一步引入一个紧密结合这些空间的连贯框架。

![图片](assets/a9111fa2402e.png)

参数空间(Parameter Space,PS)合并

将多个基础模型的权重整合为具有相同神经网络架构的统一实体，同时超越单个模型的性能。利用任务向量分析来理解每个模型的优势，依据它们所针对或擅长的具体任务。具体来说，我们通过引入DARE来增强TIES-Merging，从而实现更细致的逐层(Embedding/Transformer etc...)合并,并且为每一层设定稀疏化和权重混合的合并配置参数。然后，针对选定任务务特定指标，使用进化算法来优化这些配置。

数据流空间(Data Flow Space,DFS)合并

论文《Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space》[5]通过逆向工程Transformer模型构建块之一的前馈网络（FFN）层的操作,将token表示视为词汇表上不断变化的分布，并将每个FFN层的输出视为对该分布的加性(Additive)更新

![图片](assets/715eeefc45d2.png)

另一篇论文《Locating and Editing Factual Associations in GPT》[6]中阐述了一些因果路径依赖的特征

![图片](assets/740e575b0381.png)

还有一篇论文《Analyzing Transformers in Embedding Space》[7]分析了Embedding空间的特征

![图片](assets/1084d0c6f4d2.png)

DFS**中的模型合并保持每一层的原始权重不变。相反，它优化了Token在网络中穿行时遵循的推理路径。例如，在模型A的第i层之后，一个Token可能会被导向模型B的第j层。

## 4. SambaNova CoE

HotChip33上有一个Session介绍了SambaNova的架构[8],它是一个Dataflow处理器，片上网络架构如下：

![图片](assets/c52fab96935b.png)

PCU**是计算单元：

![图片](assets/264cbacc466b.png)

PMU是存储单元：

![图片](assets/b37a13528162.png)

通过数据流DataFlow编程， 由编译器放置算子到片上

![图片](assets/fd52a78453c6.png)

整个数据流过程：

![图片](assets/352d84ef93c5.png)

正是这样的数据流架构，对多个算子在集群内组合成为可能

![图片](assets/f3e8c74b6ea4.png)

这也是SambaNova构建Composable of Experts的原因，整体性能来看是比较出色的

![图片](assets/e0cb07800e75.png)

同时推理性能也能到330token/s，结合本文第三章模型融合的分析来看，Dataflow的编程模式和DFS融合方法可以很好的结合起来。

## 5. 总结

看到SambaNova的CoE模型，正如我在《大模型时代的数学基础》开篇所讲这一次人工智能革命的数学基础是：范畴论/代数拓扑/代数几何这些二十世纪的数学第一登上商用计算的舞台。 而CoE正是代表了范畴论中很重要的一个Composition的视角。而Jamba通过SSM/Mamba和Transformer算子的Composition优化了长序列问题，而BTX也是一个Composition。

建议您在关注芯片架构/互联架构这些影响算力的因素时，同时也关注一下算法和底层的数学基础

[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

[《GB200架构解析》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3394087933455859721#wechat_redirect)

参考资料

[1] 
Jamba:A Hybrid Transformer-Mamba Language Model: https://arxiv.org/pdf/2403.19887.pdf
[2] 
Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models: https://arxiv.org/pdf/2208.03306.pdf
[3] 
Branch-Train-MiX:Mixing Expert LLMs into a Mixture-of-Experts LLM: https://arxiv.org/abs/2403.07816
[4] 
Evolutionary Optimization of Model Merging Recipes: https://arxiv.org/pdf/2403.13187.pdf
[5] 
Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space: https://arxiv.org/pdf/2203.14680.pdf
[6] 
Locating and Editing Factual Associations in GPT: https://arxiv.org/pdf/2202.05262.pdf
[7] 
Analyzing Transformers in Embedding Space: https://arxiv.org/pdf/2209.02535.pdf
[8] 
SambaNova SN10RDU: https://hc33.hotchips.org/assets/program/conference/day2/SambaNova%20HotChips%202021%20Aug%2023%20v1.pdf