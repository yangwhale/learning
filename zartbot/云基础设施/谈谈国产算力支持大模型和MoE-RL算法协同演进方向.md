# 谈谈国产算力支持大模型和MoE/RL算法协同演进方向

> 作者: zartbot  
> 日期: 2025年2月12日 19:32  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493182&idx=3&sn=8c7d5569d7b0b558507ad8fceebd3b1d&chksm=f995f6fccee27fea0e3c7db4a0f63ab5992d18cd1361fd03ece9810f26072651b5d5e983f5a0#rd

---

### TL;DR

DeepSeek-v3/R1这一波最大的受益者或许是前些年建设了大量国产算力的一些机构, 终于把这些卡在推理上用起来了. 然后另一方面做一个比较极限的推演, 如果接下来国内公司无法获得更好的卡, 例如连H20都被禁的情况下, 如何在国产算力上完成超过1T参数的模型训练并支持低成本的推理? 同时兼顾一些RL后训练任务?

昨天和同事以及一些做芯片的同行聊了一下, 大概想到了几个路径, 就此分享一下. 前提假设是如果我们只有算力在100TFOPS左右的卡, 没有太多的高速互联能力, 同时显存带宽受限时该怎么办?

MoE的设计

MTP和Temporal-Difference RL

RL PostTrain任务和推理集群混布构建持续学习

### 1. MOE的设计

模型专家数
(激活)共享专家expert_grpgrp_limitdimMOE inter_dim16B64(6)2--20481408236B160(6)28351201536671B256(8)18471682048

当前DeepSeek-v3几个模型MoE相关的参数如上表所示, 当前的MoE实现还是受到很大的通信上的约束, 一方面为了训练的时候需要数据尽量激活多个专家所以会有一些loadbalance函数, 但是通信的约束还是需要在一个较小的范围. 同时受到Nvidia IB本身的一些缺陷, 例如组网和路由导致的PXN等额外的内存拷贝开销, AlltoAll通信上GDA-KI的开销等, 还有IB/RoCEv2 本身在IOPS很高的情况下incast的问题(当然这些问题也好解决,就是工业界现在的水货太多了搞不定而已). 这些都是导致`n_expert_groups`和`n_limited_groups`的约束. 

继续推导下一代模型, 例如hidden-state带来的dim进一步扩大, 然后专家数进一步扩大, 例如构建`n_routed_experts=2048/4096`, 然后`activated_experts = 16/32`会如何? 如何分组, 如何限制通信域? 进一步拷问, 如果这个时候用一些国产卡替代Experts的运算又该如何?

当然我们要考虑国产卡的显存带宽和容量的影响以及自身浮点算力的影响, 我们来假设一个稍微极致一点的情况, 假设我们构建一个集群, 用少量的H800**配合一大堆国产卡做训练?

![图片](assets/17db95047aed.png)

此时Gating函数的一些设计上就很有趣了, 当然可以延续Expert Group的分组, 在Gating函数输出<grp_id, expert_id> , 同样去部分的构造locality.  也就是前段时间提到的一个做法[《一个关于MoE的猜想》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493019&idx=1&sn=25a87af51b9077d50b40685d6987ca71&scene=21#wechat_redirect)

![图片](assets/9197f8b5074f.png)

当然最优的情况下是无所谓group的情况, 全局的专家负载均衡, 但是这个现在几乎所有的RoCEv2或者IB的网络在这样的通信模式下还是存在不少问题的,例如某些卡QP数量多了以后ManyToOne或者AlltoAll的长尾大的比计算时间还长.

进一步这些国产卡的算力相对较低时, 针对MLP的大矩阵乘法还是有点吃力的, 那么应该如何优化? 同时伴随着多模态和Reasoning模型Context越来越长, 如何在这种模式下能够更好的隐藏通信延迟? 那么是否可以对Hidden-state dim拆分, 构造成多个小的MLP然后做完了再concat?  然后查阅了一下资料, 微软韦福如老师他们做MultiHead MoE[1]已经蛮久了.

![图片](assets/04d1102d6b8e.png)

特别来说MH-MoE对于多模态的场景下, 或许还有更多意想不到的优势, 例如MH-MOE另一篇论文[2]中的例子

![图片](assets/1480f4efcbde.png)

结论: 当我们把一些通信的效率问题解决了(当然解决这些问题压根不需要什么ScaleUP网络, 只要抑制好长尾和Overlap好通信延迟, 事实上已经解决了XD). 结合原来假设的MoEG做两级Gating的想法, 整个Token做一个Gating, 然后dispatch到一个Group, 然后再Group内再做一次MultiHead MoE的Gating, data locality/Hierarchy都有了保障.

### 2. MTP和Temporal-Difference RL

其实第一次看到MTP的时候, 恰逢那几天正在研究AMD Zen5架构的Branch Predictor, 它采用的 2-Ahead Branch Predictor Unit 也是快30年前的东西了. 从体系结构的视角来看Speculated Decode和Branch Prediction本质上是相通的.

![图片](assets/d43688901407.png)

仔细看了一下, 其实MTP的工作应该算在《Better & Faster Large Language Models via Multi-token Prediction》[3]

昨天在讨论另一个RL相关的问题的时候, 突然觉得如果把Post Training的RL工作流和MTP结合起来,构建一个Reward Model**,  MTP的输出作为如何? 有了和, 很自然就套上一个TD(Temporal-Difference)的强化学习流程了, 然后通过一个RL的流程来提高一些MTP准确度?

另一方面直觉上这样的处理方式还可以进一步加快RL的迭代时间.

### 3. RL PostTrain任务和推理集群混布构建持续学习

在DAU**超过2000万以后, 虽然现在多个平台还是有卡顿的现象, 但至少峰谷效应会越来越明显. 有一个很朴素的想法就是在推理谷时段混布一些RL任务, 然后构建部分的异步的参数更新服务, 根据当日的一些推理结果日志进行持续性的RL迭代. 其实这也是当下很多推荐系统的样子, 相信PostTrain的一些工作流应该会持续集成进去. 这个应该就是一个纯工程上的问题了, 或许用大量的国产卡来把RL的任务跑起来, 也是一个不错的选择?

参考资料

[1]
MH-MoE: Multi-Head Mixture-of-Experts: https://arxiv.org/abs/2411.16205
[2]
MH-MoE2: https://arxiv.org/pdf/2404.15045
[3]
Better & Faster Large Language Models via Multi-token Prediction: https://arxiv.org/pdf/2404.19737