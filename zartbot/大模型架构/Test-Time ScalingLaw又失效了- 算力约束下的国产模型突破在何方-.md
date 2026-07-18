# Test-Time ScalingLaw又失效了? 算力约束下的国产模型突破在何方?

> 作者: zartbot  
> 日期: 2025年4月27日 05:04  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494066&idx=1&sn=1a33e8a3625d4c280aa563c94a5080dc&chksm=f995f970cee270661d919ffa3fc26a30bf669f22c3aa2279ad4e882593e965ccfd514ef8c487#rd

---

最近有一篇论文挺火的《Rethinking Reflection in Pre-training》[1] 阐述了一些在预训练阶段的反思推理的问题.  然后姚顺雨的《The Second Half》以及Sutton**的《The Era of Experience》也在阐述一些RL和现实经验结合的问题. 然后还有清华的一篇论文《Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?》[2]直接质疑RL.对于它能够提升推理上限进行了一些批判, 通过做了一个大规模的pass@k分析, 认为被RL训练出来的那些好的结果, 都是底座模型早就能生成的, 只是光使用底座模型挑中的概率低, RL提升了这些答案的概率. 另外还爬到一篇Stanford的论文《Cognitive Behaviors that Enable Self-Improving Reasoners》[3]也指向了基座模型的能力.

其实简单的来说, 例如你有无限只猴子, 经过无限长的时间, 一定能够输出正确答案. 然后把无限只猴子里通过强化学习, 通过Reward把差的饿死能够最终获得一个数量有限的猴子集群产生高质量答案么? 或许经过无限长时间的繁衍, 假设猴子之间没有通信, 能高概率产生高质量答案么? 那么继续假设猴子之间有通信猴子可以记忆并复述呢? **而人的出现是通信(语言)和记忆, 以及在漫长的岁月里夹杂着对工具的制造和使用而逐渐演化的, 当然还夹杂着人类自身的反思**

o1刚出来的时候写了一篇文章, 现在读起来恍如隔世的感觉...不过还是有不少东西猜对了.

[《致敬小镇做题家的OpenAI o1》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492389&idx=1&sn=4824511f530df08502bdb9997f7170f8&scene=21#wechat_redirect)

然后最近看着一些测试结果...

![图片](assets/46640f9cc90f.png)

多模态模型在IQ测试上的领先, 突然想到一句话: “Algebra is but written geometry and geometry is but figured algebra”, 而代数的本质不严谨的说就是一堆字符串的处理... 而很多时候把问题转换成图的视角又会变得清晰, 所谓一图省千言. 偶然又搜到一个关于这句话马毅老师的一个推

![图片](assets/045ac476b748.png)

很有趣的一个观点: 统计就是一个Sampled的几何. 其实本质的问题涉及到一个蛮新的领域《信息几何**》, 有一本蛮不错的书, 只是最近工作太忙, 刚开头就烂尾了, 五一假期捡起来...

![图片](assets/e2e02bc788be.jpg)

我一直以来认为在没有新的数学工具引入下的大模型训练/推理的天花板就在那里, 因此对于渣B而言还是要像人类演化的过程那样, 学会用更多的工具, 很长一段时间在做量化交易算法时, 都深感数学工具不足,对于很多特征没有一个很好的工具/理论/算法去描述...所以现在也完全没有心思去卷算法, 还是安静的学一点数学吧.

[《大模型的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

隐隐约约有一个直觉:  或许人类自然语言本身就不是解决这些问题的好工具...

**国内模型在算力的约束下, 未来路在何方?** 从年初春节的惊喜, 到现在OAI/Gemini**/Claude还在一路狂奔, 差距又拉开了, 我们少了什么? 算力的禁售越来越严重, 而国产算力的问题还需要很长的时间去慢慢解决...那么算法和对应的基础设施重构, 或许是当下能够做的事情了. 当然伴随着Agentic/MCP**的繁荣, 应用的增多, 国内还有一个优势, 模型对于工具使用的能力在大量程序员的努力下可能会卷的比国外稍微快一些, 说不定又是一个逼到绝路的柳暗花明...

去年底的时候MCP发布时还随手写过一篇

[《抄袭与创新》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492672&idx=1&sn=12878081fe367c69ea858ee2413db139&scene=21#wechat_redirect)

文中有这样一段话

本质上对现有的基于Transformer的自回归模型的天花板的观点在于其数学工具的局限性. 而我一直以来有一个暴论:这一次人工智能革命的数学基础是：范畴论/代数拓扑/代数几何这些二十世纪的数学第一登上商用计算的舞台。

当然这条路非常艰难, 例如大模型怎么和真实物理世界的定律结合, 举个稍微简单的例子, 我们能否使用纤维丛理论来描述一些结构? 当然对于预训练构成的attention网络, 是否能用纤维丛来分析它们相互之间的关系和映射,构成一个更高维度的代数结构, 并在此之上构建自约束的训练?

刚看到DeepSeek-R1的工作时, 脑子里就有一个直观的想法, Pre-Train构建了一个预层范畴, 而在RL上强化了一些态射... 但现实世界里数据的分是幂律分布的, 在Pre-train的数据信息压缩的过程中, 其实很多低概率的内容很难被模型捕获到, 然后国内算力的稀缺, 在稀疏的MoE模型下, 这些问题还会进一步加重...

在这种情况下的底座模型, 就像一个钢铁直男(pass@1)那样, 面对老婆/女朋友不停的追问: “知不知道哪里错了?" 但反过来, 你如果有某种类似于MCP/Agent2Agent连接了她的闺蜜, 或许问题好处理一些了? 其实本质上我们需要的是在Transformer的基础上建立一套高阶的代数结构**, 构建出一套Composable Transformer的架构, 其实这也是真实世界里团队协作的例子...

也就是说退而求其次, 我们搞一堆32B/70B的模型, 大概能力像接受完九年义务教育的人. 然后再来单独的训练,或者在部署Agent中吸取大量的经验, 并动态完成模型-模型间的交互和传导. 或许这才是一条更自然泛化的路.

其实我在很多年前的毕业论文里就完成过类似的算法, 构造过10000个agent, 每个Agent有一些自己的逻辑, 只是那个年代计算规模很小, 单个Agent参数比较少, 主要是单一任务做仿真的股票交易, 在300ETF里面选择股票并构建order book, 然后构造一个随机图, 相互之间有一些随机跟风行为等影响交易状态的随机矩阵来做一些状态迁移概率的模拟. 最后构建了一个虚拟的撮合系统, 完成仿真交易. 包括后面做了一个分布式强化学习优化网络的算法, 一直以来都有这样一个研究路径....

其实同理一万个各自学习经验操作工具的32B模型  vs 一个1T参数的大模型, 其实这就是Sutton的《The Era of Experience》, 那么模型结构上来看, 需要继续为Composable的模型间通信和“经验”的提升做出更多的处理, 例如不同的小模型之间通信的机制是JSON还是直接Hidden State, 是否允许层间垮模型通信? 比起MoE这样比较规则的结构变得更加复杂的图结构.

另一方面的问题是这些32B的小模型如何学习经验并构建自我反思? Google的《It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization》[4]是一个很不错的值得借鉴的工作.

![图片](assets/23922a5a2ffe.png)

还有一个值得关注的地方就是Antropic和Google一直在SAE上做一些工作, 国内一直没有看到有这方面的工作在开展, 其实它是在Attn之上的一个更高阶的代数结构, 或许这也是和国外差距拉大的一个算法相关的因素...

参考资料

[1] 
Rethinking Reflection in Pre-training: *https://arxiv.org/abs/2504.04022*
[2] 
Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?: *https://arxiv.org/abs/2504.13837*
[3] 
Cognitive Behaviors that Enable Self-Improving Reasoners: *https://arxiv.org/html/2503.01307v1*
[4] 
It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization: *https://arxiv.org/html/2504.13173v1*