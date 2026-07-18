# 谈谈大模型架构的演进之路, The Art of memory.

> 作者: zartbot  
> 日期: 2025年1月23日 00:43  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493032&idx=1&sn=206eed2e4127b9971a1e0c380f70b082&chksm=f995f56acee27c7c757d4e4c95e8bcd5ebb2e1b326e4717191cf87a4e18dbdc819c052b09d1b#rd

---

### TL;DR

看到最近大模型几个小龙们动作频频, 大概也是年关难过吧. 开复老师非常实在的捅破了窗户纸. 而接下来受到DeepSeekv3的影响, Minimax和阶跃都公开了自己的模型架构. 它们都是非常新颖的工作, Minimax通过Linear Attention降低了长文本的计算复杂度和空间复杂度. 而阶跃提出了广义多头注意力（GMHA）的概念框架来理解不同的MHA变种, 并将Fully Parameterized Bilinear Attention (FPBA)做为容量上限, 非常不错的视角. 然后找到一种矩阵分解方案最大限度的减少参数和KV Cache用量, 并使得模型容量尽可能接近FPBA. 基于这些分析, 提出了MFA和MFA-KR, 并且显著增加了Attention Head的数量和维度,同时 以及采用 single value KV.

本质上这些工作和DeepSeek的MLA都是对Attention的改造要么是在降低空间复杂度, 要么是在降低时间复杂度, 但是似乎对它的改造引出另一个视角.

如果说最早的token by token的大模型推理是一个顺序纸带的图灵机. 而o1这类的出现本质更像是一个比较完备的图灵机了, 但似乎缺少一些纸带回退和擦除的能力. 这些回退和擦除或许是推理阶段节省复杂度的一个好方法. 毕竟我们解决很多问题的时候并不需要一张无限大的草稿纸, 大模型亦然.

就像前段时间说的, 本文就是在试图通过大模型架构构造一个自己能够产生代码运行的通用计算机架构, token as instruction. 当脑子里补出这图的时候, 就豁然开朗了. 大模型从自回归可能真的要走向自生成Instruction的路了.... 那么构造一个大模型的冯诺伊曼架构大概就如下了.

![图片](assets/b1b230bb488e.png)

### 谈谈Attention block的演进

其实去年就在构思如何修改Attention block, 例如[《谈谈DeepMind会做算法导论的TransNAR并引出基于SAE-GNN的可组合Transformer猜想》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490297&idx=1&sn=7d758e84bdce7ae4f20f031f4ac3f221&scene=21#wechat_redirect) 提到的 GNN-SAE Adapter

![图片](assets/0c9ac37a6925.png)

大概的原理就是从范畴论的视角来看, 计算Attention的时候需要对其做一些约束避免错误的输入. 其实一些PostTraining的工作本质也是在更新Attention中的 WQ/WK/WV的参数进行约束. 另一方面类比计算机体系结构, `幻觉的本质其实和通用计算机访问内存越界是类似的, 类似的PostTraining实际上是在做一些越界访问的保护, 大模型安全和计算机体系结构中的安全很类似`.

最近在看一篇Google Research的论文, 更加加深了这种直觉.

![图片](assets/a92893f252ab.png)

https://arxiv.org/pdf/2501.00663v1

GNN-SAE Adapter其实本质上和Google这篇论文的MAG(Memory as Gate) Architecture是一致的. 都需要一定程度的Test Time的verification.

![图片](assets/19954d83630e.png)

开篇引用的一句话突然激发了很多灵感.

![图片](assets/e87f9a7ec5db.png)

其实针对Attention计算O(N^2d)的复杂度所面临的问题类似于当年没有虚拟内存的时代, 而各种Linear Attention, MFA, MLA, GQA,MQA实质是一种内存压缩机制去在一个更小的LatentSpace处理.

### 从计算机体系结构的视角看大模型

诚然, 我们在做很多尝试, 通过KV Head的构造来修改Attention block降低训练和推理时的内存开销. 但是很多压缩到Latent Space本身就是有损的, 所以这也是我一开始就在怀疑DeepSeek MLA的作用(当然这些担忧现在看起来是没有意义了).

这些工作实质是在一个固定的物理内存上进行处理, 放不下的时候, 去压缩数据结构. 再加上《Space-time tradeoffs of lenses and optics via higher category theory 》这篇文章通过高阶范畴中的一些关于Optics和Lens的时空折中描述, 也是一个很值得去分析的东西.

但是从计算机体系架构的视角, 我们真的需要把所有的Context都在内存里放着, 供Attention这样的Control Unit全量去访问么? 其实一个真正的Attention应该在这个基础上引入页表的机制. 就像我们读书或者读论文, 某一段某一章节引用另一章节. 整个计算引擎只需要它程序指向的那几个页表.

基于这个思路, 我们就可以构造一个虚拟内存空间, 将真正所需要的那些页来进行Attention计算. MoE或许就是一个最早期的页表内存的实现. 而前段时间提出的MoE Group本质上就是想要把它扩展成一个多级页表.

### Tokens as Instruction

其实这种自回归的生成式AI模型, 本质上就是在传统计算机体系结构的基础上进一步扩展了可以自主生成程序段代码的能力. 那么Prompt的实质或许就是一个大模型最早期的代码段(.text). 后续生成的结构中逐渐出现堆和栈的结构, 而o1和DeepSeek-R1一类的模型进一步的扩展了堆栈的处理, 通过RL来控制删除堆栈内的信息, 然后继续运算.

这个视角下, 我们有必要重新审视是否要构造一套大模型的ISA来将这些特殊的Tokens作为代码段使用? Maybe...

那么下一波大模型能够探索的方向, 或者说Test Time ScalingLaw的实质就是对产生的Token序列(Context)做一些特殊的修改和Page置换, 然后让它走新的代码段执行.

这个时候就需要一个自我验证纠错的机制了, 而Google Titans的论文提供了一个最早期的例子

![图片](assets/5d797eaca9c1.png)

![图片](assets/374b2cdc447d.png)

通过TestTime Learning并根据已有的Context Retrieve “代码段”(Instruction Tokens)

![图片](assets/a97ab0250a20.png)

同时又根据Attention的输出去更新一些控制指令内存. 这种方式或许才是真正的去省KVCache内存消耗的一条正路. 当然我可能还会觉得应该在这里面再加一些控制逻辑, 而这些东西应该涉及到高阶范畴/代数拓扑/代数几何上的一些态射约束

### Speculation Decode as Branch Predictor

另一方面就是类似投机解码, 或者DeepSeek的MTP, 从计算机体系结构上来看, 实际上它也构成了一个take N的Branch Predictor.

### MoE Group as multilevel page table

从范畴论的角度讲, MoE的Block Diagonal矩阵其实本质上是破坏了这样的结构, 使得一些态射被忽视了. 那么这个时候引入多级页表呢?

![图片](assets/6d74bd283433.png)

或许这才是真正的“Page” Attention吧?

### Put all togather

其实正如本文开头所述, 这些东西很自洽的构成了一个比较完备的通用处理器架构. 只要你能够有效的在Test Time去把内存(Context)换页, 让模型能够按照这个Context生成新的指令, 整个计算架构就逐渐趋向完备了.

当然这条路只是在架构上我自己说服了自己, 通向AGI的路可能有一些光亮了... 当然还有很多训练和推理工程上需要实现的大量工作要去做, 或许这是我们Deepseek这类公司从追赶OAI到超越OAI的一条路吧...