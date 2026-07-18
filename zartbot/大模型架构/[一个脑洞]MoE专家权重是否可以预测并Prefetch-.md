# [一个脑洞]MoE专家权重是否可以预测并Prefetch?

> 作者: zartbot  
> 日期: 2025年2月27日 16:34  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493299&idx=1&sn=fdf9d0928ccca8638535ab94db22378f&chksm=f995f671cee27f673d8d44ee5ec43e7089ea97eaf852cc2dacbd8ee09afd7f38cf0d3be7c823#rd

---

大概两周前, 夏Fellow在某个群里说了一句关于Expert权重是否可以Prefetch**的话题, 今天晚上走路下班回家的路上在考虑通用处理器Branch Prediction的一个问题时, 伴随着前几天分析DeepEP和今天开源的Expert Loadbalance及一些训练和Prefill的trace, 突然有一个脑洞, 是否能在Gate函数上做一点手脚, 让它输出下一层的topk_idx的一个预测, 然后在下一层计算Attention的时候, 提前做一些下一层的Expert权重的Prefetch?

`业务价值`: 对于一些云提供的API服务,通过EP并行的方式已经可以获得很好的吞吐率**了, 但是对于一些私有化部署,还是要维持在两机四机的模式下, 在这种方式下,内存带宽受限时, 吞吐和EP320这样的部署差距接近几十倍, 如果通过Prefetch专家权重或许能够提高性能, 使得私有化部署的成本差异降低.

对于一个已经训练好的模型, 例如DeepSeek-R1, 可能需要一个额外的很轻量的神经网络来预测. 首先对于R1这样的模型前三层为MLP, 大概率估计从第四层开始可能已经有比较明确的稀疏的expert_weight分布了, 然后DeepSeek-V3/R1的MoEGate激活函数采用了Sigmoid, 似乎这样的信号会更加明确一些.

大致想了一下,最简单的算法就是拿下一层的`group_idx`和`topk_idx`作为Label, 然后基于attention-score的输入和MoE Gate后的Score作为输入, 构造一个很简单的MLP网络去预测. 同时损失函数中添加一下针对`topk_idx`和`group_idx`错配的一个辅助损失项. 但是似乎觉得这样的预测准确率估计不会太高? 特别是在DeepSeek这样的细粒度MoE的解决方法上预测的难度感觉还是有点大的, 加上Group约束可能会好一些?

因此又想到一个强化学习的算法, 因为专家的选择和预测可能跟输入的话题有关, 然后前面若干层的已经选择的专家`topk_idx`,`group_idx`序列可能会隐藏着更多的信息, 因此针对MoE的第t层的AttentionScore和GatingScore以及GroupScore以及前面若干层的`topk_idx`,`group_idx`作为一个state , 然后预测t+1层的MoE Gating的`topk_idx`和`group_idx`作为一个action , 最后对整个58层的`topk_idx`, `group_idx`预测准确率作为一个Reward函数来计算每一层的预测的收益, 并且对`topk_idx`和`group_idx`错配构成一个惩罚项. 或者直接计算预测的下一层的`moe_gate_score`和真实score的KL散度?

然后我们通过对一些广泛使用的测试集的数据来进行训练, 这样构成一个小的预测器网络, 然后在每一层MoE计算结束的时候执行预测器操作,并在计算attention时异步的对专家权重数据进行一部分prefetch.

更进一步, 是否可以根据这些预测构建一个Mixture-of-Depths: Dynamically allocating compute in transformer-based language models[1]的算法? 当然这个可能就要改动模型本身了.

![图片](assets/d16007fc89cd.png)

其实建议下一代模型在训练时的MoE Gating函数输出中增加一路next_layer_topk_idx的输出, 或许是解决当前DeepSeek-V3/R1私有化部署性能较差的一个折中的做法.

另外腾讯今天发布了`快思考模型混元Turbo S`, 它使用了Mamba**+Transformer+MoE的混合结构, 测试了一下效果还是不错的, 特别是token产生的速度感觉非常快, 但是想到Jim Keller**的一个演讲中的漫画...

![图片](assets/2eec9d8dd2ae.png)

嘿嘿嘿~~

BTW, 似乎最近几天微信总收到一些抄袭投诉通知, 觉得挺烦的, 渣B都是自己动手写的呀, 该引用的都注明了的, 后来仔细看了一下是别人抄袭渣B被其它人举报了, 要我处理. 大概说一下吧, 反正别人能拿渣B写的字赚点钱也挺好的, 知识本身没有太大的价值, 学会知识才有价值嘛, 这一点渣B并不介意, 抄就抄了吧,确实工作忙懒得处理这些小事, 做好自己不干这些不道德的事情就好. 

参考资料

[1] 
Mixture-of-Depths: Dynamically allocating compute in transformer-based language models: *https://arxiv.org/pdf/2404.02258.pdf*