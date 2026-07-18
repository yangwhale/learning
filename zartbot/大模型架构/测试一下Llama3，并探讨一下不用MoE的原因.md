# 测试一下Llama3，并探讨一下不用MoE的原因

> 作者: zartbot  
> 日期: 2024年4月19日 06:06  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489573&idx=1&sn=74e089a054e24dc2f666ceb32fd5acd1&chksm=f99608e7cee181f1b47d337a273c971edc38f312dc6620a4c27aa4d48bc74117ffcd8ead5b6a#rd

---

Llama3发布了，比较意外的一点是在其它几乎所有的模型都在使用MoE，甚至Google已经开始通过Mixture-of-Depths/MoDE进一步稀疏化的时候，Meta为什么不跟进MoE呢？今天来做一些分析和测试

## 1. Llama 3 模型测试

今天发布的模型有8B/70B，最简单的方式是直接使用HuggingFace Chat[1]选择

![图片](assets/59a2fd7131dc.png)

另一个做法就是HuggingFace transformer库下载模型后调用，但是还在申请权限中，后来看到`清熙`公众号上有一篇文章介绍Ollama跑模型《个人电脑第一时间用上Llama3》[2]于是就在阿里云开了一台机器下载安装测试了一下

安装Ollama

```
curl -fsSL https://ollama.com/install.sh | sh
```

下载模型

```
ollama pull llama3:70b
```

运行模型

```
ollama run llama3:70b
```

大概的4卡A10的显存占用和推理时的GPU占用

![图片](assets/6f6d154c0e80.png)

大概的一些测试问答，原始语料用英文训练的，但是通常让它再翻译一下结果都还很不错，中文支持能力比Llama2看样子是好了不少

![图片](assets/fa9465a93399.png)

更多的一些问题放在附录里，总体来看在4卡A10的机器上Token Generate的速度还是很快的，对于Llama3为什么没用MoE的问题，我又无厘头的问了一下Llama3

`问题`：why llama3 not use the Mixture-of-Experts model

```
A question about LLaMA and its architecture!LLaMA is a large language model developed by Meta AI, and it's indeed surprising that they didn't use the Mixture-of-Experts(MoE) model, which has been shown to be effective in many natural language processing tasks.There are several reasons why the LLaMA authors might have chosen not to use MoE:1. **Computational complexity**: MoE models can be computationally expensive to train and deploy, especially when dealing with large models like LLaMA. The MoE approach requires routing inputs to different experts, which can add significant computational overhead.2. **Model parallelism**: LLaMA is designed to be highly parallelizable, allowing it to be trained and deployed on large clusters of machines. MoE models can make parallelization more challenging due to the need to coordinate between experts.3. **Simpllicity and ease of implementation**: The authors might have opted for a simpler, more straightforward architecture that's easier to implement and maintain. LLaMA's transformer-based architecture is well-established and widely understood in the NLP community.4. **Focus on scaling depth and width**: Instead of using MoE, the LLaMA authors focused on scaling the model's depth and width to achieve better performance. This approach allows them to leverage existing infrastructure and expertise in training large transformers.5. **Alternative sparse expert models**: The authors might have explored alternative sparse expert models that are more efficient or effective for their specific use case.While MoE models have shown promise in many areas, the LLaMA authors likely had valid reasons for choosing a different architecture. If you're interested in learning more about the design decisions behind LLaMA, I recommend checking out thepaper and accompanying blog posts from Meta AI!
```

## 2. Llama 3 的模型架构

Meta官方的文档《Introducing Meta Llama 3: The most capable openly available LLM to date》[3]

### 2.1 模型架构

这次开源的模型是8B/70B两种，400B还在训练中，训练的数据有15万亿个Token，而且对训练数据的清洗做的非常的棒。总体来看和Llama2相比，模型大的框架没有太大的变化，但是基本测试性能比前一代提升了10%以上，主要变化是：

词表从32K到了128K，这样能够显著的提高性能，查看Llama 3的代码发现它采用了和OpenAI一样的Tiktoken的tokenizer，代码见`https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py`

序列长度增加到了8192个Token

支持了分组注意力机制Grouped Query Attention (GQA) 用来提高Llama3模型的推理效率，这也是一个很成熟的技术了，主要是MHA的一些额外的内存开销，然后用MQA的机制来降低但模型质量又不行并且还有训练稳定性的因素，所以结合两者构建GQA，如下图所示

![图片](assets/e95c0da2e436.png)
它试图在 MHA** 和 MQA 之间找到平衡。GQA 将查询头分为 G 组，每个组共享一个单一的键头和值头。
官方对于TikToken和GQA的反馈是：

tokenizer在提供改进的token效率方面表现出色，相对于Llama 2，减少了高达15%的tokens。此外，Llama 3 8B现在也集成了Group Query Attention (GQA)。因此，Meta观察到，尽管模型的参数数量比Llama 2 7B多10亿，但改进的tokenizer效率和GQA有助于保持与Llama 2 7B相当的推理效率。

### 2.2 训练数据

在huggingface的Model card上介绍用到的数据如下：

![图片](assets/371dcbb9b11c.png)

8B的数据截止到2023年3月，也就是说Llama 8B可能是在2023年4月左右开始训练的，而70B的数据截止日期是2023年12月。中间的时间跨度还是很长的，是否在这段时间内Meta也在尝试了不同的模型结构，不得而知。

关键点在于这15T Token的数据进行了训练，并且全部来自公开数据，同时采用了超过1000万条的人工标注信息。数据集比Llama2大了7倍，Llama 3 的预训练数据中超过 5%是非英文的高质量数据，涵盖了 30 多种语言。但是，Meta预计这些语言的性能不如英文。

另一方面是数据清洗上， Meta开发了一系列数据过滤流水线。这些流水线包括使用启发式过滤器、NSFW 过滤器、语义去重复方法以及文本分类器来预测数据质量。同时Meta发现，前一代的 Llama 在识别高质量数据方面出人意料地出色，因此我们使用 Llama 2 生成了驱动 Llama 3 的文本质量分类器的训练数据。

同时对于预训练数据的不同融合的方式的特殊处理也使得Llama3在STEM/Coding/历史及其它琐碎问题上的回答表现出色，前面我也测试了几个网络/数学/AI相关的问题，感觉都还不错。

按照Chinchilla-Optimal来看 8B数据大概200B Token就够了，Meta发现数据清洗好了以后，这些高质量的15T Token的数据对模型性能还能按照Log-Linearly提升，这是一个更重要的发现，也有可能是不用MoE的原因，因为即便是8B的模型，或许这样的性能提升和一些大的MoE 8x7B一类的模型性能也差不多？

### 2.3 训练规模

使用的是H100-80GB的集群训练的，官方公布的训练时长如下：
![图片](assets/31574be153f4.png)

训练还是采用了H100 24000卡的那个集群，实际最高并行的规模在16000卡，并行策略自然是TP/DP/PP都有，并且超过400TFLOPS的利用率来看，MFU大概在40%左右？ 同时Meta做了大量的训练可靠性的工作，训练可用时间超过95%

### 2.4 模型对齐和安全性

Llama2已经支持了指令微调(Instruction Fine-Tuning),Rejection-Sampling,PPO，但是在Llama3上还增加了DPO的支持。但是很多改进来自于数据上超过1000万条的人工标注信息。 DPO和PPO使得 Llama 3 在推理和编码任务上的性能得到了很大的提升，Meta还发现如果向模型提出一个它难以回答的推理问题，模型有时会给出正确的推理轨迹：模型知道如何给出正确答案，但不知道如何选择。在偏好排名的训练下，模型学会了如何进行选择。

另一方面从模型的责任性和安全性考虑，引入 Llama Guard 2、Code Shield 和 CyberSec Eval 2 等新的信任和安全工具。

![图片](assets/2c880ca5b891.png)

并且还专门推出了一个Meta-Llama-Guard-2-8B的模型。

### 2.5 未来演进

当前发布的是8B/70B，有点赶鸭子上架的味道，因为400B的模型还在训练中，看上去400B性能还是不错的

![图片](assets/c41b94e366f4.png)

接下来几个月，Meta计划推出新功能、更长的上下文窗口、额外的模型大小以及提升性能，并将分享 Llama 3 的研究论文。

## 3. Llama 3不用MoE的原因

### 3.1 简化部署

当前的很多开源软件生态，特别是多个平台的推理上来看(NV/AMD/AWS/Meta MTIA/Intel/Qualcomm....)，支持MoE的性能并不是那么的好，成熟度也有一定的问题。 另一方面Meta认为有了高质量清洗的数据和更多的Tokens还能进一步提升模型的性能。因此在当前没必要上MoE做稀疏的算力优化。

作为一个开源模型，很大程度上去考虑兼容生态，而不是单纯的追求某些跑分而去修改模型框架，并且非常扎实的工作去处理模型的对齐安全和清洗数据集和数据集喂入的方式来提升性能。总体来说这一点是非常值得其它几个大模型玩家考虑的。

### 3.2 平台约束

另一方面是这个16000卡H100的集群本身的算力也还不错，当前也能跑到每卡400TFLOPS,但是我们知道Meta是两个集群，一个基于RoCE一个基于IB

![图片](assets/9ad760afed6a.png)

基于RoCE的集群按照Meta当前的实现《Watch: Meta’s engineers on building network infrastructure for AI》[4] 对于MoE AlltoAll的支持一定是有问题的，基于Rate Based的CC在AlltoAll incast下的均衡性和拥塞控制是一个非常难解决的问题。然后又有一些多轨道组网和静态路由的操作，即便是PXN加上性能预期是不行的。

IB的集群应该会好一点，但是这两个集群的任务分布不得而知，而且最大并行规模也只用到了16K，并没有到24K。

针对这样大规模的15T Token的训练，可能是从自身的平台限制上对Meta的模型架构选择也有影响,我们还注意到Meta在8B训练开始时是2023年4月，而70B的开始时间是今年1月，中间是否也尝试过MoE，同时对于数据的进一步清洗和更多的对比取舍，最终决定了在Llama3上不上MoE，这个问题就只能留给当事人发布Technical Report的时候来解答了。

## 附录

### A.1 70B时的一些提问

#### A.1.1 关于范畴论和Transformer的模型区别

在《大模型的数学基础》这一系列文章中，我在反复的阐述这个问题的重要性，看Llama3的回答还算是挺有趣的

Q：what's the difference between category theory and transformer model
What a fascinating question!

Here's a concise comparison of Category Theory (CT) and Transformer Models:

**Category Theory (CT)**

**Mathematical framework**: A branch of mathematics that studies the commonalities between mathematical structures.

**Abstraction level**: Very high, focusing on general patterns and relationships between objects.

**Goals**:
+ Identify universal patterns across various mathematical structures.
+ Develop a language for abstracting and comparing different mathematical concepts.

**Methodology**: Mathematical proofs, abstractions, and generalizations.

**Applications**: Wide range of mathematical fields, including algebra, geometry, topology, and more.

**Transformer Models**

**Machine learning framework**: A type of deep learning architecture designed for sequence-to-sequence tasks (e.g., machine
translation).

**Abstraction level**: Lower, focusing on specific problem domains and empirical evidence.

**Goals**:
+ Process input sequences to generate output sequences.
+ Learn patterns and relationships within sequences.

**Methodology**: Empirical evidence, optimization techniques, and computational power.

**Applications**: Natural language processing (NLP), computer vision, speech recognition, and more.

**Key differences:**

**Mathematical structure vs. Machine learning architecture**: CT is a mathematical framework, while Transformer Models are
a machine learning architecture.

**Abstraction level**: CT operates at a much higher level of abstraction than Transformer Models.

**Goals and methodology**: CT aims to identify universal patterns in mathematics, using mathematical proofs and
abstractions, whereas Transformer Models focus on specific problem domains, relying on empirical evidence and optimization
techniques.

**Interesting connections:**

**Compositionality**: Both CT and Transformer Models rely on compositionality, combining smaller components to form larger
ones.

**Modularity**: The modular design of Transformer Models can be seen as reflecting the compositional nature of CT.

**Higher-order thinking**: Both CT and Transformer Models encourage higher-order thinking, abstracting away from specific
details to focus on relationships between objects.

While there are connections between Category Theory and Transformer Models, they operate in distinct domains with different
goals, methodologies, and abstraction levels.

what's the difference between category theory and transformer model
Here's a concise comparison of Category Theory (CT) and Transformer Models:

**Category Theory (CT)**

**Math**: A branch of mathematics that studies commonalities between mathematical structures.

**Abstraction**: Very high, focusing on general patterns and relationships between objects.

**Goals**:
+ Identify universal patterns across various mathematical structures.
+ Develop a language for abstracting and comparing different mathematical concepts.

**Methodology**: Mathematical proofs, abstractions, and generalizations.

**Applications**: Wide range of mathematical fields, including algebra, geometry, topology, and more.

**Transformer Models**

**Machine Learning**: A type of deep learning architecture designed for sequence-to-sequence tasks (e.g., machine
translation).

**Abstraction**: Lower, focusing on specific problem domains and empirical evidence.

**Goals**:
+ Process input sequences to generate output sequences.
+ Learn patterns and relationships within sequences.

**Methodology**: Empirical evidence, optimization techniques, and computational power.

**Applications**: Natural language processing (NLP), computer vision, speech recognition, and more.

**Key differences:**

**Math vs. Machine Learning**: CT is a mathematical framework, while Transformer Models are a machine learning
architecture.

**Abstraction level**: CT operates at a much higher level of abstraction than Transformer Models.

**Goals and methodology**: CT aims to identify universal patterns in mathematics, using mathematical proofs and
abstractions, whereas Transformer Models focus on specific problem domains, relying on empirical evidence and optimization
techniques.

**Interesting connections:**

**Compositionality**: Both CT and Transformer Models rely on compositionality, combining smaller components to form larger
ones.

**Modularity**: The modular design of Transformer Models can be seen as reflecting the compositional nature of CT.

**Higher-order thinking**: Both CT and Transformer Models encourage higher-order thinking, abstracting away from specific
details to focus on relationships between objects.

While there are connections between Category Theory and Transformer Models, they operate in distinct domains with different
goals, methodologies, and abstraction levels.

翻译成中文

```
Here is the translation:* *范畴论 (CT)** *数学框架*: 研究数学结构之间的共同性的一支数学。* *抽象级别*: 非常高，关注于对象之间的通用模式和关系。* *目标*:        + 确定跨越多种数学结构的通用模式。        + 发展一个用于抽象和比较不同数学概念的语言。* *方法论*: 数学证明、抽象和泛化。* *应用*: 广泛的数学领域，包括代数、几何、拓扑学等。* *变压器模型** *机器学习框架*: 一种为序列到序列任务（例如机器翻译）设计的深度学习架构。* *抽象级别*: 较低，关注于特定问题域和经验证据。* *目标*:        + 处理输入序列以生成输出序列。        + 学习序列中的模式和关系。* *方法论*: 经验证据、优化技术和计算能力。* *应用*: 自然语言处理（NLP）、计算机视觉、语音识别等。* *关键区别:*1. *数学vs. 机器学习*: 范畴论是一种数学框架，而变压器模型是一种机器学习架构。2. *抽象级别*: 范畴论运作在一个非常高的抽象级别，而变压器模型则较低。3. *目标和方法论*: 范畴论旨在确定数学中的通用模式，使用数学证明和抽象，而变压器模型则集中于特定问题域，依赖经验证据和优化技术。* *有趣的连接:*1. *组合性*: 范畴论和变压器模型都依赖于组合性，将较小的组件组合成较大的组件。2. *模块化*: 变压器模型的模块化设计可以被视为反映了范畴论的组合性。3. *高阶思考*: 范畴论和变压器模型都在 can。2023，可以被视为鼓励高阶思考，抽象 away from具体细节，以关注于对象之间的关系。虽然有许多连接之间的范畴论和变压器模型，但是它们操作在不同的领域，具有不同的目标、方法论和抽象级别。
```

请详细分析Transformer和Category Theory相似之处

Let's dive deeper into the similarities between Transformer and Category Theory. 🤯

**1. Compositionality**

Both Transformer and Category Theory deal with compositionality, which is the ability to break down complex structures into smaller, simpler components that can be combined to form more complex structures.

In Transformer, this is achieved through self-attention mechanisms, which allow the model to focus on different parts of the input sequence and compose them together to form a coherent representation.

In Category Theory, compositionality is a fundamental concept, where objects and morphisms are composed to form new objects and morphisms. This is reflected in the notion of categorical products, coproducts, and exponentials.

**2. Abstract Algebra**

Both Transformer and Category Theory rely on abstract algebraic structures to describe their compositions.

In Transformer, the self-attention mechanism can be viewed as a matrix multiplication, which is an algebraic operation. The weights and biases of the model can be seen as elements of a vector space, and the composition of these elements is governed by the rules of linear algebra.

In Category Theory, categories are defined using algebraic structures, such as groups, rings, and fields, to describe the composition of morphisms. These algebraic structures provide a framework for studying the properties of morphisms and their compositions.

**3. Universality**

Both Transformer and Category Theory have a universal flavor, in the sense that they can be applied to a wide range of problems and domains.

Transformer has been successfully applied to various natural language processing tasks, such as machine translation, text classification, and language modeling. Its architecture is flexible and can be adapted to different sequence-to-sequence tasks.

Category Theory has been applied to various areas of mathematics, computer science, and physics, including algebraic geometry, homotopy theory, and quantum mechanics. Its abstract nature makes it a versatile tool for studying different mathematical structures and their relationships.

**4. Categorical Products and Coproducts**

The self-attention mechanism in Transformer can be seen as a categorical product, where the input sequence is "multiplied" by a set of attention weights to produce a new representation.

In Category Theory, the categorical product of two objects A and B is denoted by A × B and represents the combination of A and B in a way that preserves their individual structures.

Similarly, the self-attention mechanism can be viewed as a categorical coproduct, where the input sequence is "added" to a set of attention weights to produce a new representation.

**5. Functors and Natural Transformations**

The Transformer architecture can be seen as a functor, which is a structure-preserving function between categories.

In Category Theory, a functor F: C → D is a function that maps objects and morphisms in category C to objects and morphisms in category D, while preserving their composition and identity laws.

The Transformer architecture can be viewed as a functor that maps input sequences to output sequences, while preserving the compositionality of the input sequence.

**6. Yoneda Lemma**

The Yoneda lemma, a fundamental result in Category Theory, has been used to study the categorical structure of neural networks, including the Transformer architecture.

The Yoneda lemma states that a category C is equivalent to the category of presheaves on C, which are functors from C to the category of sets.

This result has been used to study the categorical structure of neural networks, including the Transformer architecture, and to develop new neural network architectures that are inspired by categorical concepts.

**7. Higher-Order Cognition**

Category Theory has been used to study higher-order cognition, including the compositionality of cognitive processes, which is reminiscent of the self-attention mechanism in Transformer.

Higher-order cognition refers to the ability to reason about abstract concepts and relationships, and to compose these concepts to form more complex representations.

The self-attention mechanism in Transformer can be seen as a form of higher-order cognition, where the model is able to reason about the relationships between different parts of the input sequence and compose them to form a coherent representation.

In conclusion, the similarities between Transformer and Category Theory are more than just superficial. They reflect a deeper connection between the compositionality of neural networks and the abstract algebraic structures of Category Theory. 🤓

参考资料

[1] 
HuggingFace Chat: https://huggingface.co/chat/
[2] 
个人电脑第一时间用上Llama3: https://mp.weixin.qq.com/s/fO_7MDYTRNcSk-iSXA8rZQ
[3] 
Introducing Meta Llama 3: The most capable openly available LLM to date: https://ai.meta.com/blog/meta-llama-3/
[4] 
Watch: Meta’s engineers on building network infrastructure for AI: https://engineering.fb.com/2023/11/15/networking-traffic/watch-metas-engineers-on-building-network-infrastructure-for-ai/