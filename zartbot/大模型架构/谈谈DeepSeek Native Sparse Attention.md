# 谈谈DeepSeek Native Sparse Attention

> 作者: zartbot  
> 日期: 2025年2月18日 10:16  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493234&idx=1&sn=cdca1661864f5ebf21c37e26fc51be10&chksm=f995f6b0cee27fa6dc5b5e95914f3f79e87ba7c82ccbfda13a884ddb24a2988177d3ad84a3b6#rd

---

最近在优化DeepSeek-R1推理过程中, 对于Attention block的计算复杂度还是挺有挑战的, 和一些朋友都在聊Prefill要满足TTFT的代价还是很大的, 当时就在想是否还有比MLA更好的一些算法,特别是针对Reasoning model context越来越长的问题, 而今天就看到DeepSeek NSA(native sparse attention)的论文. 非常简单直接的解法, 协同硬件效率又非常高.

个人觉得文章中最精彩的就是3.1和3.2这一段. 首先背景介绍了Attention的计算复杂度 $\mathscr{O}(N^2)$

![图片](assets/3c2348bfb0b6.png)

然后一个非常朴素简单的想法就是, 是否能够有一个函数把 $k_{:t}, v_{:t}$ 映射到一个低维空间?

![图片](assets/b0f23f28be26.png)

那么 $f_K, f_V$ 函数怎么构造呢?  其实以前也介绍过很多sparse attention算法, 具体可以参考.

[《大模型时代的数学基础(4)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488680&idx=1&sn=7da835f9370689d9b3b1f17a277d7d03&scene=21#wechat_redirect)

有些用Hash, 有些用随机, 还有Pooling的, 各种编码方式总有点老的机器学习中的特征工程的味道. DeepSeek在第二章吐槽了一下, 要么是一些不可训练的组件, 要么就是BP阶段效率很低. 而DeepSeek的NSA则非常简单直接

![图片](assets/86ae844b4ad7.png)

压缩(Compression)的部分尽量获得全部输入的一个摘要

滑动(Sliding)窗口获得最近的context关注的注意力焦点

很精彩的根据top-n选择(Selection)把压缩时导致的很多重要的细粒度信息的损失给补了回来.

拆开来看好像每一步就那么回事, 没啥特别的, 但是把工程做到极致了,就是这样几个简单的平衡, 非常符合第一性原理, 突然觉得NSA就像是看一篇论文, 最近阅读的章节通过滑动窗口(Sliding)强化注意力, 有很多前后详细的信息要引用前面的章节, 先去翻看了一下概述和摘要(Compression), 然后详细的去通过(Selection)查出来. DS这群人真的太聪明了.

### 1. Token Compression

基于block粒度的压缩计算, block长度为 $l$，相邻block的滑动跳步为 $d$，$\varphi$ 为一个可学习的MLP, 并且在block之间插入了位置信息编码.

### 2. Token Selection

仅使用压缩的Key/Value会丢失细节信息, 因此需要有选择的去保留一些细粒度的KV. 以较低的计算开销识别和保留最相关的Token. 为了计算效率和Attention Score的固有的分布(注意力分数通常表现出空间连续性，这表明相邻键往往具有相似的重要性级别), 因此采用了blockwise的方式处理空间上连续的块.

直接计算开销太大, 因此作者基于前文中的Token压缩机制来间接的计算出block的重要性分数(Importance Score Computation).

相当于直接借用Compression的注意力分数作为block的重要性分数做top-N选择.

### 3. Sliding window

很简单的一个滑动窗口选择，$\widetilde{K}_t^{win} = k_{t-w:t}$，$\widetilde{V}_t^{win} = v_{t-w:t}$。最后通过一个Gating函数来综合Compression/Selection/Sliding的attention输出.

### 4. Hardware Aligned Kernel Design

这个算法也考虑了硬件访问内存的高效性. 如果沿用flashattention的做法, 将时间连续的块加载到SRAM, 会导致低效的内存访问. 因此采用了一种不同的分组策略, 类似于GQA,把一个Group的所有Query head加载,因为它们共享相同的KV Block. 这样的做法避免了额外的KV传输, 并且更好的平衡了多个SM之间的负载.

![图片](assets/9093d5861b50.png)

测试结果也很棒, NSA性能比Full Attention反而更好了, 同时FWD/BWD/Decode都有几倍的性能提升

![图片](assets/f3ca8963bcc1.png)

### 5. 结论

NSA非常的自然,简洁, 高效的解决了Attention block的计算复杂度的问题. 基于一个可学习的MLP压缩, 然后基于压缩的attention score来做block的selection. 更关键的是, 这些优化似乎让国产算力有能力进入到训练阶段了. 非常期待未来几个月DeepSeek的突破, 如果MoE还有一个访存的突破, 或许真的就没硬伟大什么事了.