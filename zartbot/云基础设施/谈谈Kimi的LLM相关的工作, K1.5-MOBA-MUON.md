# 谈谈Kimi的LLM相关的工作, K1.5/MOBA/MUON

> 作者: zartbot  
> 日期: 2025年2月23日 03:44  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493252&idx=1&sn=693a78f3ba99657a35b6c43102121730&chksm=f995f646cee27f50b21033f4c1631931b969f1df4dd8c9fee379ed536adc3b1be17b0467d2e9#rd

---

Kimi的很多工作也是很领先的, 但是K1.5和DeepSeek R1撞车, 然后MOBA又和DeepSeek NSA撞车, 社区的关注度少了一些. 但是Kimi最近几个小时开源了一个基于Muon的优化器的工作[1]进一步优化了训练效率

![图片](assets/e0cd36055be0.png)

Muon是Keller Jordan在去年的一个工作, 在去年十月渣B就在内部建议过Muon的工作, 好像没啥人关注

![图片](assets/6b60147bb43f.png)

今年年初在[《Pretrain ScalingLaw真的终结了么?》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493002&idx=1&sn=736a3bd3e03c8a34a9831d1d3324cd1e&scene=21#wechat_redirect) 以及后续的文章也谈到了Muon优化器的工作.

Kimi Muon的作者苏剑林老师在去年也对Moun进行了分析《Muon优化器赏析：从向量到矩阵的本质跨越》[2], 接下来几周时间, 渣B会对K1.5, MOBA以及Muon的工作进行一些分析, 例如MOBA的上下文长度可以扩展到10M.

其实很多时候, 研究的品味和追求真理从第一性原理出发来看待大模型的研发路径是非常重要的. 前段时间听到张一鸣准备创建Seed Edge做一些Foundation的研究, 最近组织结构调整似乎也完成了, 在前谷歌DeepMind副总裁吴永辉博士加入了字节并负责Seed的研发将会进一步加速这个过程.

其实从第一性原理来讲, 原有的Transformer模型Attention block和MLP block的平方复杂度的问题是肯定很难scale的, 一开始其实渣B就在质疑ScalingLaw, 针对这两个block的稀疏化工作一直都是研究的方向, 因此几年前就一直反复在提国内大模型厂商不要简单的跟随国外的技术路线, 现在DeepSeek-V3的MoE, NSA和MOBA就是最好的证明. 还有就是直接根据一些数学逻辑的Rulebased方式, 例如过去两年一直在探索基于高阶范畴的Rule来进行ORM based RL, deepseek R1这些工作也完成了落地. 个人觉得DeepSeek-R1的数据集生成可能采用了类似的做法, 因为我前段时间在复现R1的时候, 发现数据集的构造可能比RL的工作流更加重要.

另一方面是在AI Infra中, 通过FP8以及未来FP4的进行优化,但是需要在数值稳定性上做大量的工作.

前段时间的一篇文章也做了一些反思

[《从Deepseek R1和NSA算法谈谈个人的一些反思》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493241&idx=1&sn=539a79b731ce275ef7eb79503036a23c&scene=21#wechat_redirect)

更进一步,当前很多大模型厂商和Infra的协同还是有不少问题的, 例如DeepSeek的MoE推理上采用的IBGDA. 其实这是一个在高频交易行业很普遍的一个技术. 当我们收到一个行情报文后, 需要以最低的延迟让GPU进行推理,并反馈相应的决策快速产生订单. 前几年就和某著名高频交易商Opxxxx的一个朋友深入探讨过, 所以这一次看到DeepSeek用IBGDA做alltoall降低延迟一点都不意外.

但是开源的Sglang的MoE MP并行框架里, 并没有采用标准的alltoall通信, 而是前期MLA有一个allgather, 然后通过在Local Expert对tokens进行permutation重排序计算后,再通过allreduce对MoE的多个专家的输出加总求和, 好处是通过简单的动态添加local expert list就可以很快的实现redudancy expert和dynmaic loadbalance, 但是坏处是通信量会远大于标准的alltoall的实现方式.

其实当EP并行规模进一步加大时, 直接在通信上的alltoall需要有大量的优化, 例如RDMA的拥塞控制上针对incast的设计以及网络拓扑结构的设计, 所以两年前就在针对MoE模型进行一些通信的设计, 我一直强调的在RDMA上128-to-1的incast控制如何保证每个流之间的误差小于100kbps,降低长尾. 而且你会看到DeepSeek-V3报告中对硬件演进的建议,实际上在2021年做NetDAM的时候就已经预测到通信本身对GPU SM的占用, 需要在硬件上进行协同设计, 后来在eRDMA的通信设计上也解决一部分alltoall拥塞控制的问题.

当前的另一个问题是, 基本上所有的Infra特别是芯片团队不懂大模型算法, 而算法团队也没有很多的硬件能力, 导致一些trade-off可能无法实施, 当硬件和算法完全的协同在一起时, 将会出现显著的改变, 训练和推理基本上还可以提高十倍以上的效率, 所以这也是渣B过去几年很长一段时间在做的工作去拉通算法和底层硬件团队.

这一篇是关于Kimi介绍的专题的开篇, 看到国内越来越多的优秀的企业,优秀的算法, 优秀的人, 想到钱老的一句话:“怎么不行？外国人能搞的，难道中国人不能搞？中国人比他们矮一截？”

大家都一起加油干吧!

参考资料

[1] 
Kimi Moonlight Muon优化器: *https://github.com/MoonshotAI/Moonlight/*
[2] 
Muon优化器赏析：从向量到矩阵的本质跨越: *https://spaces.ac.cn/archives/10592*