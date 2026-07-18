# 大模型时代的数学基础(6)-从word2vec谈谈表示论，组合性，幺半范畴和Dataflow Optics

> 作者: zartbot  
> 日期: 2023年12月21日 15:03  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488775&idx=1&sn=1793eb897beb71ce4a64c9ab44beee6b&chksm=f99605c5cee18cd3481913d17122bb9da63f6385901c9842e8173186e98040d7f91620a91f95#rd

---

### TL;DR

前几天NeurIPS2023的`Test of Time`时间检验奖办法给了word2vec《Distributed Representations of Words and Phrases and their Compositionality》[1]我们注意到文章中的两个词`Representation`和`Compositionality`，那么今天就来稍微展开从数学上讲讲这两个话题。

### 1. Representation,表示论

Word2Vec的第一篇论文应该是《Efficient Estimation of Word Representations in Vector Space》，它属于数学中抽象代数的一个分支：表示论(Representation Theory).表示论主要的方法是将抽象代数结构中的元素“表示”成向量空间上的线性变换，并研究这些代数结构上的模，藉以研究结构的性质。表示论的妙用在于能将抽象代数问题转为较容易解决的线性代数问题。

#### 1.1 群和群表示

我们来回顾一下前几节关于群的定义，对于一个二元运算, 满足如下条件的则构成一个代数结构被称为群(Group).

运算是封闭的, , 则

运算满足结合律,

包含单位元，即

G中任意元素都有逆,即 

通常我们也把这样的二元运算并不局限于常见的加法等，而是更加广泛的抽象为一类操作，例如物理中引入对称性。 如果把这些群的操作写成一个个矩阵，我们就可以把这种作用称为该群的一个表示。

一个群在域上的向量空间的表示是一个从到的群同态:

稍为给非数学系的同学扩展一下，是上的一般线性群(General Linear Group)，V是一个n维向量空间，则它是一个nxn的可逆矩阵，并且由矩阵乘法构成一个群。
对于一个有限群的有限维表示,由Maschke定理，如果有非平凡的子表示,则有表示的直和分解,由于是有限维，则存在如下分解

其中每个子表示都没有非平凡子表示。

`不可约表示`:如果的有限维表示没有非平凡的子表示，则称为的不可约表示。

接下来，我们就要探讨一系列问题：

什么样的表示是不可约的？

如何判别两个不可约表示是否等价？

这里就要介绍一下Schur引理（Schur's lemma）：

设和为的有限维不可约表示，表示到的线性变换的集合

若不等价，则

若是代数闭域，则

舒尔引理是群与代数的表示论中一个初等但非常有用的命题。若, 那么是可逆的或

看到这里，您会想到什么？

对于一个有限群表示可以通过直和分解成多个不可约表示，那么我们就可以将每个不可约表示定义为一个Expert，通过Schur引理，使得模型通过路由和多次通过Experts MLP + Attention 是否可以构造出这样的表示？

![图片](assets/be737f52811a.png)

#### 1.2 Word2Vec

例如Word2Vec论文中"Additive Compositionality" 有一些论述，模型中学到的词和短语表现出一种线性结构，这使得使用简单的向量运算进行精确的类比推理成为可能，同时一些简单的元素级别的向量表示加法可以有意义地组合单词。

![图片](assets/7d3860615ca5.png)

例如我们定义一种`加法`能够表示`北京 + 中国 = 东京 + 日本`，当然这只是一个更加容易理解的例子，而不是严格的群的定义，Word2vec实际上就是构建了这样的二元运算群到线性空间的映射，使得它们在线性空间的运算和群的操作同构。那么对于大语言模型，我们首先将词映射到一个向量空间,另一方面是通过Attention这一系列映射构建词和词之间的关系。

![图片](assets/c7e66edd7fe5.png)

#### 1.3 多模态和向量数据库

从更广义的角度来看，文档/视频/用户行为似乎都可以映射到线性空间，然后通过线性代数的方式获得其相似性或者其它运算的能力。那么也因此诞生了很多向量数据库，例如Pinecone由AWS SageMaker的创建者Edo Liberty成立，他在自家的博客中这样写道：
“机器学习将一切都表示为向量，包括文档、视频、用户行为等等。这些表示使得不同事物可根据相似性或相关性，就能够准确检索、搜索、排名和分类。在很多场景中，如产品推荐、语义搜索、图像搜索、异常检测、欺诈检测、人脸识别等都有应用。”

![图片](assets/0d862a3f7393.png)

当视频/音频都可以用向量表示后，那么多模态也就逐渐出现了,然后Transformer的Encoder也罢，decoder也罢，从这里再对应1.1小节的不可约表示和Schur引理，你应该能够悟出点什么了？

![图片](assets/950c29309854.png)

### 2. Compositionality

范畴论本身是一个很大很泛的框架，对于大模型本身的可能没有太多直接的影响。但是很多时候我们在设计一个新的算法时，特别是一些代数结构的时候，通常需要考虑BP怎么实现? 自动微分怎么搞?当然很多人会说：“链式法则啊，又有什么难的？”

而另一方面，在我们构建一个更大的场景时，如何自动的将LLM/LVM和Function Call一起像乐高那样组合成一个更大的系统？Agent之间的接口是什么？如何能够让整个模型Lifetime Learning？

Compositionality是我们值得去关注的一个话题，也是介绍范畴论时很重要的一部分。当然它也是函数式编程中非常重要的一个话题，当然这里还会顺便引出Monad，Lense和Optics的概念，然后后面的章节再慢慢展开谈一下Ray的Task和Actor的设计模式，以及gradient-checkpoint这样的时空折中方案的原理。

这里先从积范畴开始讲起...

#### 2.1 Product & Coproduct
2.1.1 Product
范畴和的product category  定义为：

对象:由一对构成，其中

态射:构成一对态射,其中为中的态射, 为中的态射。

Composition: 

Identities: 

你可能会直观的想到这不就是集合上定义的笛卡尔积(Cartesian product)么？对于集合范畴来看确实是这样的，而范畴论会将这些概念进一步推广到其它范畴，例如对于群/环/模这样的代数结果中的直积，以群的直积为例，即给定两个群G和H，这两个群的积可以表示为：，而里面的元素配对运算，各群内的运算法则独立运行.

实际上对于一个Product ,我们可以构建两个投射(Projection).

编程上，我们通常用如下方式表示,例如

```
template<class A, class B> Afst(pair<A, B> const & p) {    return p.first;}
```

注：这部分内容也可以参考《Category Theory for Programmers:Products-and-coproducts》[2]

事实上可能会有多个积符合这样的投射，例如下图所示：
![图片](assets/7d9096d28ece.png)

同样我们可以用来表示态射, 根据这个交换图，态射的复合如下

当然还会有其它的态射可以构建出.并且也存在一个态射. 这样就会产生一个锥(cone)的结构，然后有一系列态射

也就是说 都有到的箭头，

![图片](assets/42e7d9d15e8b.png)

结合前面我们讲到过的始对象和终对象，锥范畴中的终对象称为极限 (limit)
2.1.2 Coproduct
Product可以看作是一个对象向另外两个对象的投射(Projection),那么是否会存在一个对偶的CoProduct呢？实际上就是两个对象注入(Injection)到一个对象

![图片](assets/8af7c8af70af.png)

同样的也会存在一系列态射

这样就构成了一个余锥的结构，同样余锥范畴中的始对象称为余极限 (colimit)。

![图片](assets/8cbfe0764e3d.png)

#### 2.2 Injection or Projection

正好前几天看到一个微软的论文《Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs》[3] 在探讨通过FineTune还是Retrieval将各种各样的知识射入到大模型，那么从范畴论的角度来看，这些射入的知识是否能够构成一个余锥呢？

而另一方面针对大语言模型生成的数据，是否能看做是一个Projection呢？那么对于的锥的结构呢？ 从这个角度来看，是否能够构建出一种复合，即利用小模型生成数据Projection，然后再Injection到大模型，并同时通过锥的结构进行约束呢？

![图片](assets/c0d86f57a292.png)

其实也就是从范畴这样的方法论的角度构造出了OpenAI最近提出的《Weak-to-Strong Generalization》的方法，是不是很有趣？或者把一个大模型Projection到多个小模型决策，然后再Injection回来，这不就是MoE么？

无论是哪种方法，背后最根本的是Compositionality.而这些能力实际上来自哪呢？

#### 2.3 DataFlow

事实上，一系列算子的复合最终构成了一个数据流(DataFlow)模型，正如论文《One Model To Learn Them All》[4]都是一系列算子的复合

![图片](assets/e05822b01cfa.png)

所以您也会看到Google在其下一代的训练框架上《Pathways: Asynchronous Distributed Dataflow for ML》[5] ，对于数据流编程模型来看，实际上程序建模为数据在运算（operation）之间流动的有向图。

我们再用LangChain举一个例子，实际上大模型本身也构成一个算子

```
model = "meta-llama/Llama-2-13b-chat-hf"tokenizer = AutoTokenizer.from_pretrained(model)hf_pipe = transformers.pipeline(    "text-generation",    model=model,    torch_dtype=torch.float16,    device_map="auto",)llm = HuggingFacePipeline(pipeline=hf_pipe)
```

然后我们再把Prompt做成一个算子

```
template = """<s>[INST] <<SYS>>你现在是一个资深的{role}<</SYS>>Question: {question}Answer: Let's think step by step.[/INST]"""prompt = PromptTemplate.from_template(template)
```

然后就可以复合成一个Chain

```
chain = prompt | llm
```

最后就可以复合成一个带Prompt的Query

```
question = "请比较Semi-Lattice和Semi-Group的区别?"role = "数学家"print(chain.invoke({"role": role, "question": question}))
```

#### 2.4 monoidal category

假如我们有两个函数和，我们期望有一个`大模型`能够泛化成为一个`计算容器`同时支持这两个函数，即

如果，，，都属于同一个范畴,那么我们就可以定一个bifunctor 使得

这样两个函数就组合起来了,从数学上来看，Monoidal Category（幺半范畴)实际上是这样一个范畴

被赋予了一个二元函子(bifunctor),被称为monoidal product或者tensor product

被赋予了一个单位对象，被称为unit object

有三个自然同构(natural isomorphism)映射

并且满足以下条件：

![图片](assets/52061336bf30.png)

![图片](assets/f02a2114269c.png)

再回到前面，我们期望大模型作为一个计算容器同时支持两种函数

也就是说这个复合的输入输出都需要在一个范畴，实际上我们就可以通过对原来的数据类型加上一个盒子来实现，构建原始数据类型到盒子的自然变换

![图片](assets/8a1f4d74de31.png)

因此我们从幺半范畴的角度引入了函数式编程的单子(monad)的概念：

在函数式编程中，单子（monad）是一种抽象，它允许以泛型方式构造程序。支持它的语言可以使用单子来抽象出程序逻辑需要的样板代码。为了达成这个目标，单子提供它们自己的数据类型（每种类型的单子都有特定的类型），它表示一种特殊形式计算，与之在一起的有两个过程，一个过程用来包装单子内“任何”基本类型的值（产生单子值），另一个过程用来复合那些输出单子值的函数（叫做单子函数）

对大模型而言， 通常我们也需要构造类似的Function Call，并且还要和向量数据库等一起复合使用，大模型和函数计算的融合会是一件非常有趣的事情，但是整个数据流框架来看需要更加全盘的考虑，否则又会出现大模型适配一类的繁杂的人肉工程。

#### 2.5 Optics

haskell Optics[6]库有这样一句定义：

An optic is a first-class, `composable` notion of `substructure`.

`Optics.Iso:isomorphisms`首先是类型之间的同构，简单的来说就是前面一节讲的盒子，原始的可能有不同的数据类型，我们可以理解为不同的范畴，然后我们需要构建一个并存在逆箭头，因此就构成了一个同构，A是一个`substructure`, 例如我们定义一个年龄类型和一个整形是同构的，并且定义了view和review函数用于类型之间的转换

```
       coerced :: Iso' Age Intview   coerced :: Age -> Intreview coerced :: Int -> Age
```

`Optics.Lens: generalised fields`是一个积范畴(product category), 它由一个`view`函数构成projection，即 ,同时还有一个`set`(也称为update)函数来更新，即

```
     _1 :: Lens' (X,Y) X     _2 :: Lens' (X,Y) Yview _1 :: (X,Y) -> Xset  _2 :: Y -> (X,Y) -> (X,Y)
```

这里不是为了完整的介绍函数式编程中的Optics，我们只是通过它介绍Lens范畴。

`Prism`

![图片](assets/609b0ca623e9.png)

它涉及关于和的态射, 其中, 

`Traversal`涉及遍历执行

![图片](assets/ba793682a55a.png)

更详细的内容可以参考其它资料，例如Profunctor Optics: The Categorical View[7]
2.5.1 Lens范畴
更一般来看，范畴定义如下：

对象： 在中的一对(a pair of)对象 

态射:  存在一对方法,使得， 都是中的态射

由于是一个范畴，因此还有复合：

![图片](assets/ed67c8f70b93.png)

则

另一方面我们如何构造

考虑如下变换：

所以：

思考题 看到上图你会想到什么？view是一个forward propagation，set是一个backward propagation是不是一下就明白了，同时连为什么要保留activation都搞清楚了？

实际上求导就是一个Lens，而链式法则就是Lens的复合。另一方面深度神经网络中的优化器也是一个Lens，下图来自于论文《Categorical Foundations of Gradient-Based Learning
》[8] 感兴趣可以展开读一下

![图片](assets/6ecabb91fbb6.png)

事实上还有更多的结构都可以成为Lens，例如状态方程等，假设状态为, 输入输出为我们可以通过获得output，并且可以通过,这也是非attention建模中很多利用状态空间模型(SSM)的原因
![图片](assets/d7def0cc2cd3.png)

2.5.2 Optics范畴
范畴定义如下：

对象： 在中的一对(a pair of)对象 

态射:  存在一个三元组构成的态射,其中为范畴中的对象,和都是中的态射

实际上我们可以把Optics看作一个带内部状态的结构

![图片](assets/72d958b03038.png)

而和是同构的

![图片](assets/09f3a74e0740.png)

这里你会想到什么？“gradient/activation checkpointing”这样的技术来降低中间存储

![图片](assets/e82014c42de5.png)

本质上这就是和同构带来的时空折中，例如两个Lens复合后的set函数(和深度学习中的backward propagation过程相似)

对于中间的可以本地存储为，这样就变成了一个,也是大多数模型框架的做法，当显存不够的时候，也可以从换成，由前面几层的Activation使用view函数进行FP计算而得，但是又会产生瓶颈

![图片](assets/09dda52700ec.png)

是不是很有趣呢？然后再来一个例子Ray的Task和Actor的抽象？

更多的内容可以参考《Space-time tradeoffs of lenses and optics via higher category theory 》[9]

### 3. 小结

今天展开讲了一下`Representation`和`Compositionality`,通过不可约表示和Schur引理看到MoE似乎是一条路，而从Compositionality来看，我们引入了一些基本的幺半范畴和Monad的介绍，而对于深度神经网络训练中的FP和BP实际上是一个Lens，DNN中的层和其它函数式编程代码如何复合，特别是后面大模型Agent和其它工具链的整合，例如低代码场景，在计算平台设计上如何构建像乐高那样容易搭建的环境，以及多模态数据如何通过表示论和复合能力更好的融合，都是值得我们探索的方向。

#### 参考

[1] 
Distributed Representations of Words and Phrases and their Compositionality: https://arxiv.org/pdf/1310.4546.pdf,
[2] 
Category Theory for Programmers:Products-and-coproducts: https://bartoszmilewski.com/2015/01/07/products-and-coproducts/,
[3] 
Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs: https://arxiv.org/abs/2312.05934,
[4] 
One Model To Learn Them All: https://arxiv.org/abs/1706.05137,
[5] 
Pathways: Asynchronous Distributed Dataflow for ML: https://arxiv.org/abs/2203.12533,
[6] 
haskell Optics document: https://hackage.haskell.org/package/optics-0.4.2.1/docs/Optics.html,
[7] 
Profunctor Optics: The Categorical View: https://golem.ph.utexas.edu/category/2020/01/profunctor_optics_the_categori.html,
[8] 
Categorical Foundations of Gradient-Based Learning: https://arxiv.org/pdf/2103.01931.pdf,
[9] 
Space-time tradeoffs of lenses and optics via higher category theory: https://arxiv.org/pdf/2209.09351.pdf,