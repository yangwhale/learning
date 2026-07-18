# AI需要新的数学语言?

> 作者: zartbot  
> 日期: 2026年3月16日 11:51  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497848&idx=1&sn=57a70f7e56b49eeb2d3ca2d071289b5a&chksm=f995e8bacee261acb9e195b317af6cbb82b67b8542b7b4dffc28e8490d2b556902dc43fe7b4f#rd

---

### TL;DR

昨天看到统计之都的一篇文章《统计学最高荣誉回归华人！苏炜杰：AI需要一门新的数学语言》[1]. 引起了一些共鸣, 但是其中有一个观点: *“现有的数学语言，或许从一开始就不适合描述AI”* 并不是完全的认同, 可能加一个约束“现有的(概率统计)的数学语言”比较适合. 模仿GPT的那味写了一段话:

先抛结论: 很多20世纪的数学都没用上, 就要新的数学语言? 我们不是需要一门新的数学语言,而是我们需要在简单的概率统计工具之外, 利用 范畴论 / 代数拓扑 / 代数几何这样的工具来拓展整个计算任务, 这个观点只是专注于统计学的数学家们自身工具缺乏带来的限制, 以及当前的计算机体系结构又把它拉回到传统的概率统计框架中来追求并行计算效率, 这才是最大的矛盾和约束.

## 1. Overview

我们真的需要全新的数学语言吗? 很多强大的, 现成的20世纪数学工具(如范畴论, 代数拓扑等)在计算机科学中远未被充分利用. 因此我一直以来有一个观点: **这一次人工智能革命的数学基础是：范畴论**/代数拓扑/代数几何这些二十世纪的数学第一登上商用计算的舞台.**

又想起几年前知乎上 @黑暗里谁还不睡 的一段话:

现在AI领域的现状就是这样，数学上的high level intuition和ml theory的一堆assumption是割裂的，based on ml theory的formulation和跑出来的玄学实验结果又是割裂的，在苛刻条件下的实验结果和现实中的task上的performance还是割裂的，仅有的几个performance比较好的任务和工业界要求的变现能力强的落地场景仍旧是割裂的，但凡有人可以补上这几环中的任何一环足以拯救这个行业，但现在来看，真的很难乐观

实质性的问题是当代数学已经是一个非常庞大的学科. 但是通常工科背景的同学可能大家就学过几门 高等数学, 复变函数, 微分方程, 线性代数, 概率与数理统计, 随机过程. 例如数值分析与科学计算这些对AI很重要的基础课很多高校也并没有太重视, 导致在一些算法收敛性的问题上, 还是以炼丹为主.

当代数是一个很大的世界, 但我们似乎只在上面用了一小块. 甚至在一些场景下还有一些概念上的问题. 例如DeepSeek发布mHC后, 数学圈的朋友们普遍反馈是在某种程度上拔高概念. 工业界(专指AI相关)对于流形的理解大多停留在一个高维数据投影到低维子空间的概念上, 如下图所示:

![图片](assets/b823c3ef185b.png)

然而, 在mHC上用到了切空间**, 黎曼度量, 测地线距离, 曲率这些概念了么? 并没有, 所以描述成流形约束其实就是在概念上拔高, 更准确的说法或许是“基于双随机矩阵约束的HC”.

概观整个数学的全貌, 在18/19世纪数学经历了爆炸性的发展, 形成了我们今天熟悉的三大分支:分析/代数/几何. 但是进入20世纪后, 数学家们发现, 许多最深刻, 最困难的问题, 无法在单一学科的框架内解决. 例如代数拓扑/代数几何/微分几何/解析数论的出现, 逐渐的融合了这几个方向.  当一个几何问题用代数语言重新表述后, 可能会变得异常简单, 反之亦然. 就像掌握多门外语的人能更深刻地理解不同文化一样.

![图片](assets/a217cc448f85.png)

因此当前的问题不在于创造"新的数学语言", 而在于扩展我们的数学工具箱, 多用一些20世纪出现的数学语言. 我们应该超越目前主流依赖的"微积分和概率统计"框架, 引入更擅长描述结构, 关系, 和不变性的现代数学分支.

但是这条路是很难的.

例如算子哥写 cutlass 通常会涉及到 layout 代数(可以参考[《CuTe Layout代数-1: Overview》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496154&idx=1&sn=474a5450c46b86169095d84dd3cfd7dc&scene=21#wechat_redirect)), 它涉及一些范畴论和数论的知识, 但是工科背景的同学通常在这一块上理解相对会困难一些.

另一方面基于范畴论的一些态射交换图的概念和实际的运算也很难结合起来, 或许只能做为一个模型算法的宏观分析来使用. 例如对Sparse Attn和Linear Attn的一些争议, 很有趣的是最近看到知乎 @刀刀宁 的一篇文章《笔记：简单图解一下线性注意力机制》[2] 有一段关于Linear Attn的话挺有意思的:

而 linear attention 因为每次都在更新 SSM ，所有信息都保留在了 SSM 里，SSM 大小不变，叠加进去的具体次数的信息因为加法操作后失去了 query 的指向标签，在运行时是无法再将具体哪一次的信息单独抽取出来的。

同时，既不能强调什么，也不能丢弃什么，也就是说重点也不能突出，非重点也不会忘记，就好像一锅粥一样。左脑都是面粉，右脑都是水，脑子一动就都是浆糊。没法精炼提取太多有价值的信息出来。

对于Sparse和Linear的争议, 如果使用最优传输理论**的视角来看, 有一些很不错的答案, 而本质上Optimal Transport这个数学分支它也涵盖了测度论 / 泛函分析 / PDE /  微分几何 / 概率论 / 组合优化与计算数学 等多方面的内容. 新的工具的引入带来了新的视角.

但是一些新的概念引入下, 如何进行高效的计算还受限于当前的计算机体系结构的约束. 例如拓扑数据分析(TDA), 在文章[《谈谈Attention SInk及未来Attention算法设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497613&idx=1&sn=b7dfc41a83978a789ac582c8034faac8&scene=21#wechat_redirect)中的附录中介绍了一些持久化同调(Persistent Homology,PH)和Betti数来量化注意力网络的拓扑结构. 但是persistence diagram, VR复形和Ripser算法这些的计算复杂度还是非常高的. 并且很多算法本身是很难并行执行的, 另外很多算法本身是否在低精度运算下成立也是一个值得探讨的问题.

下面我们来谈谈几个方向上的进展.

## 2. 信息几何**

自香农创建信息论, 通常以欧式几何, 赋范空间和线性代数这三大传统数学理论作为基础. 但是近代涌现的复杂网络, 非线性控制, 高维信号分析等给传统的数学理论提出了新的挑战. 信息几何也是一门很新的学科, 它将微分几何,概率论与信息论结合而构成的心的学科. 它主要使用黎曼几何的工具, 假设流形是光滑的, 并赋予其Fisher信息度量. 但是对于奇异点, 它将其视为困难的特殊情况, 需要奇异学习理论 (Singular Learning Theory, SLT)等高级理论来处理.

例如我们可以看看这篇文章[《大模型时代的数学基础(9)- SDPA和最优传输, 强化学习及信息几何的联系》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494688&idx=1&sn=3d589f6d4be56ee372d5db4f8631b0cc&scene=21#wechat_redirect)

另外我们在前一个章节中描述的, 对于mHC实际上并没有用到流形/切空间/黎曼度量**等概念. 而基于信息几何在当前AI中的应用也没有太大的进展, 因为Fisher信息矩阵及其逆计算成本是很高的, 通常我们只能用来对一些训练好的模型采样做一些分析. 实际的训练过程中似乎也没有很好的能够用上的手段, 而对于SLT我们将在下一章中介绍.

## 3. 代数几何

其实最早在2009年由渡边澄夫 (Sumio Watanabe) 撰写《Algebraic Geometry and Statistical Learning Theory》开始就在想用代数几何的视角(特别是奇点理论)来改进统计学习理论. 他利用了代数几何中的Resolution of Singularities的概念, 将崎岖不平的奇异空间"拉平", 变成一个由简单坐标轴交叉构成的空间. 在这个新的, 被"整理好"的空间里, 模型的学习过程变得可以分析和预测.

另外还有一篇论文《Recent Advances in Algebraic Geometry and Bayesian Statistics》[3]也是关于代数几何在统计机器学习中的一些应用.

近期的一篇论文是《Algebra Unveils Deep Learning An Invitation to Neuroalgebraic Geometry》[4], 核心是想利用如dimension, degree, sigularities, parameterization and fibers等这些代数几何工具来研究神经流形(Neuromanifold, 即使用多项式或ReLU激活函数的神经网络, 其所能表示的函数空间)性质. 例如模型的表达能力(能表示多复杂的函数)和样本复杂度(需要多少数据才能学好)并非凭空而来, 而是由其神经流形的几何形状(具体由维数和度量化)所决定的. 这在模型的表达能力和学习效率之间建立了一个根本性的权衡. 另一方面 奇异点在优化过程中扮演“吸引子”的角色, 导致梯度下降趋向于这些点, 从而产生隐式偏置. 对于许多神经网络, 奇异点恰好对应于更简单的子网络.

另一个相关的方向是我最近也在看的涉及Grassmann manifold的内容, 比如这篇论文《A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects》[5]. 因为前段时间在思考一些Sparse Attention以及Hierarchy Sparse Attention以及Attn Sink时想到的一些算法来解决Full Attn 复杂度的问题.

具体可以参考[《谈谈Attention SInk及未来Attention算法设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497613&idx=1&sn=b7dfc41a83978a789ac582c8034faac8&scene=21#wechat_redirect)中第五章, 利用 ASink作为几何锚点的性质. 通常我们可以将其看作为一个高维流形下的某个低维度流形. 例如一本书的章节1可以在基于Anchor1的子空间内计算, 章节2 在Anchor2的子空间内计算. 这样就避免了大量的长序列的章节1中的token和章节2中的token之间的无效的Attention计算, 章节的关系, 通过可学习Anchor来定义相对的参考系, 如下图所示:

![图片](assets/08438a013081.png)

然后逐渐的在找一些工具, 也就是Grassmann manifold相关的内容. Grassmann manifold  是  中所有  维线性子空间的集合.然而, 它不是一个简单的欧几里得空间, 而是一个弯曲的流形, 这使得在其上进行微积分和优化变得复杂. 但是也是值得去探索的, 或许它是Sparse Attention中一个很好的方向, 毕竟的复杂度还是很值得去努力试一试的. 这部分内容可能接下来一两个月有些结果了再单独写一篇.

## 4. 代数拓扑

在分析AttnSink那篇文章也谈论到了这个话题, 拓扑数据分析 (TDA) 与 持久同调 (Persistent Homology)相当于使用代数拓扑的工具对模型做了一个“CT”, 例如现在模型的结构是3:1的Linear:Full构成的Hybrid Attn, 是否可以通过对每一层的拓扑结构的分析来指导模型结构的设计. 例如前几层是一些连续的Linear Attn变换, 而后面几层伴随着拓扑结构的变化采用Full/Sparse Attn?

另一方面是Anthropic最近的一篇论文《When Models Manipulate Manifolds: The Geometry of a Counting Task》[6], 从代数拓扑的角度看, 这篇论文的贡献是实验性地证明了深度学习模型能够学习到数据背后潜在的拓扑和代数结构, 并将这些结构用于计算. 这超越了单纯的模式识别, 触及了某种形式的"抽象推理". 我们能否通过在模型设计中显式地引入拓扑约束, 来引导模型更快更好地学习这些抽象结构? 比如, 设计一种损失函数, 惩罚那些拓扑结构与目标概念不符的表示. 这也是很值得去研究的一个方向.

## 5. 范畴论

把Attention看作是一个态射, 然后以范畴论的语言来解释一些问题, 同时是否可以通过一些Nerve构造来设计模型的结构? 这也是一个很值得讨论的问题. 另一方面在下面这个专题中, 还有一些和范畴深度学习相关的内容.

[《大模型的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

## 6. 其它

例如《Fiber Bundle Networks: A Geometric Machine Learning Paradigm》[7]利用纤维丛的思想来做一些分类任务, 也是挺有趣的观点.另外还有一些关于TOPOS理论如何应用到大模型的探讨, 比较有代表新的就是菲尔兹奖得主Laurent Lafforgue加入华为后, 有几个session在讲相关的内容, 但似乎都是很早期的工作. 用黑话说就是, 通过这些数学知识在现在的大模型训练和推理中找不到抓手. 

另一方面是我最近在详细的学习整个大模型后训练的RL相关的算法, 对于各种reward的设计, 采样等, 总觉得又回到了那个炼丹的年代, 找不出任何说服自己的理论依据. 

但是这些新的数学工具是否真的有用, 我个人的观点是不学一定不会知道, 无知的情况下可能会胡乱造一些所谓的新的数学语言, 然后发现知识在做一些可能几十年前人家早就发现的东西. 

当然另一方面还有一个问题是, 对于数学专业的同学而言, 代码能力相对较强的人并不多, 对于计算机体系结构的理解可能也有一些专业上的欠缺. 这也是一个亟待解决的问题. 算法和芯片的协同设计是一个更值得探讨的话题.

Anyway...

![图片](assets/2e054e633d12.jpg)

嗯, 人蠢(我说我自己)就得多读书. 多学一点, 万一发现点什么呢?

参考资料

[1] 
统计学最高荣誉回归华人！苏炜杰：AI需要一门新的数学语言: *https://mp.weixin.qq.com/s/ztNscRRHFzmblqbBxmMfbQ*
[2] 
笔记：简单图解一下线性注意力机制: *https://zhuanlan.zhihu.com/p/718156896*
[3] 
Recent Advances in Algebraic Geometry and Bayesian Statistics: *https://arxiv.org/abs/2211.10049*
[4] 
Algebra Unveils Deep Learning An Invitation to Neuroalgebraic Geometry: *https://arxiv.org/pdf/2501.18915*
[5] 
A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects: *https://arxiv.org/pdf/2011.13699*
[6] 
When Models Manipulate Manifolds: The Geometry of a Counting Task: *https://arxiv.org/pdf/2601.04480*
[7] 
Fiber Bundle Networks: A Geometric Machine Learning Paradigm: *https://arxiv.org/abs/2512.01151*