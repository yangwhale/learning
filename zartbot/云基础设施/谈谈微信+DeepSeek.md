# 谈谈微信+DeepSeek

> 作者: zartbot  
> 日期: 2025年2月16日 06:00  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493223&idx=1&sn=3aa971cacb4784b58482452eba491236&chksm=f995f6a5cee27fb304a0e829f614745f0c3783e45f4d7d62480ae62781c31084c905bec50ba2#rd

---

注: 本文仅代表个人观点, 和任职的公司无关.

ima.copilot刚出来的时候, 用了一段时间. 除了混元自身模型有一些差距以外, 对整个公众号的内容生态上的支持是非常好的. 前段时间也测试了一下公众号后台用LLM自动回复,除了业务逻辑上有些交互的问题(例如正常的留言交流和搜索整理公众号内容的区分),其实整理的内容还是可读性很高的, 当然也有因为基础模型的问题导致的 最近微信开始灰度DeepSeek了, 在2C的市场上将会迎来更多的变数, 特别是对字节豆包的生态上的影响. 毕竟腾讯微信的DAU是接近10亿级的.

另一方面最近百度和火山的表态(参考快科技的报道[1])

百度智能云事业群总裁沈抖在全员会上表示，国内大模型去年的“恶意”价格战，导致行业整体创收相较于国外差了多个数量级。

字节跳动旗下火山引擎总裁谭待通过朋友圈回应称，大模型降价是通过技术创新实现的，大家`应像DeepSeek一样聚焦基本功，少无端猜测，归因外部。` 谭待指出，火山引擎的豆包1.5Pro模型的预训练成本和推理成本均低于DeepSeek V3，更是远低于国内其他模型，在当前价格下有非常不错的毛利。他进一步解释：“国内外的厂商都在依靠技术创新，降低模型价格。我们也只是实现了Gemini 2.0 Flash的价格水平而已，这个价格完全是依赖技术进步可以做到的。”

其实很多时候成本的估计的分歧, 本质是技术上的差距. 例如尤洋老师估计的数据, 和DeepSeek-V3的论文实现的PD分离+EP并行性能差距超过10倍以上. 本质上的成本差异就是拿一些开源社区简单的TP/PP并行的结果来估计是有很大的差距的. 特别是Google Gemini 2.0 Flash的价格水平来看, 技术上还有更多的优化空间. 例如昨天提到的一篇文章对MOE的性能分析.

[《谈谈DeepSeek MoE模型优化和未来演进以及字节Ultra-Sparse Memory相关的工作》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493218&idx=1&sn=f394f39a4346fd09a19008a53d0a8022&scene=21#wechat_redirect)

这里简单的做一个Roofline的分析, 从算力上讲DeepSeek-V3/R1模型的算力需求是相对较小的, 瓶颈主要是在访存和All2All的通信上以及如何解决推理时的专家负载均衡上. 例如华为昇腾提到的:“通过EP混合并行算法, 通信优化性能提升30%+, 访存性能提升20%+, 从而降低专家不均衡度, 推理吞吐性能提升20%~35%” 另一方面从袁进辉老师的一段话可以知道, 梁总为啥要推荐性能最好需要80台, 主要是通过EP并行获得更好的Data Locality.

从Network-Bound上来看, 我们以单个Token 7168B来计算, 400Gbps网络机间互联网络即50GB/s, 简单的做一个上限估计. 模型需要传输60层, 每个Token需要8个Routed Expert和一个Shared Expert, 即单个Token需要 7168 x 9 x60 ~ 4MB的数据, 再算上Attention block的通信量, 即一秒单卡能够产生的Token为超过6000个, 再加上一些通信损耗和SLA的延迟保障约束, 按照只有30%~50%的折扣估计, 基本上单卡在并行策略恰当时能够做到1800~3000个tokens.

从Memory-Bound上来看, 虽然按照DSv3的论文Decoding阶段需要256个token作为一个batch, 其数据量为1.8MB, 但是单个Expert的参数数据量为44MB. 因此如果尽量的把专家打散, 然后保证L2 Cache的hit-rate时, 内存带宽的效率会高出很多倍.

而单机/双机/四机的PP/TP并行则很难获得这样的Data Locality的优势, 这也就是为什么梁总建议40台或者80台做更大规模EP并行的原因.

另一方面我们还需要考虑到DeepSeek-V3/R1对MTP的支持, 例如Sglang最近实现MTP后, 性能几乎又翻倍了. 所以在MTP支持的情况下, 单卡的TPS还可以接近翻倍.

考虑到一些额外的开销, 我们按照单卡的性能下限2000TPS计算, 单机8卡就是差不多16000TPS. 按照每个用户20TPS的速率, 大概单台H20可以承担800个用户, 考虑PD分离中的Prefill节点的另一些开销, 估计单机H20做到600个用户技术上是可行的.

那么紧接着针对微信10亿DAU, 早上7~10点基本上是各种资讯类信息的公众号消息推送, 下午大多是一些广告/电商, 晚上内容还会更丰富, 基本上一天内都可以维持在一个相对较高的水平. 按照单用户微信使用时长60分钟来估计, 大概并发活跃用户数以4000万估计, 按照单机800个用户, 大概需要5万台也就是说40万卡. 但实际上如果进一步放宽到10Tokens/s和考虑到一些泊松到达和用户使用频率的情况, 大概10万卡~20万卡即可, 也就是公众号`共识粉碎机`在[《微信+Deepseek：2C应用的转折点》](https://mp.weixin.qq.com/s?__biz=MzI2MTM2MTgxNQ==&mid=2247488575&idx=1&sn=0fd261a2b9c593ccf44b832153788d3e&scene=21#wechat_redirect)提到的

我们在之前就已经从供应链看到了腾讯加单了10-20万张H20，现在看微信版Deepseek就是明确的用途。

再来说点题外话, 最近一段时间除了调优推理以外, 还在做一些R1的复现的工作. 整个强化学习的工作流才是DeepSeek的主线, 通过强化学习来实现AGI/ASI是目的, 至于MLA/MoE/MTP/FP8等是在实现这条路上的手段, 包括DeepSeek的app本身, 我相信有传言说梁总还不太想要这几千万的DAU, 这是一个非常真实可信的想法. 特别是渣B最近在做R1复现的时候, 对推理性能的瓶颈又伴随着推理优化的工作, 对这一点理解更深刻了.

其实谈起强化学习这个话题, 似乎有说不完的故事. 从差不多快25年前搞OI竞赛的时候写了大量的动态规划算法, 最后拿奖保送上海西南某技校, 再到差不多20年前毕业论文是基于元胞细胞机和复杂网络在博弈论视角下对金融资产价格进行定价,通过Multi-Agent的仿真交易实现.在7~8年前Cisco基于强化学习模型和SegmentRouting构建的SDN和SWAN网络拿了CEO大奖, 并发布了Cisco Predicatable Network.  再到两年前用了一个非常简单的动态规划算法设计eRDMA的拥塞控制算法. 春节期间重新整理了一遍强化学习的算法, 而最近一周虽然忙着DS-R1推理调优, 也穿插着做一些复现R1的模型训练工作, 然后训练的时候发现推理的效率太低, 又要动手去优化trl和vllm, 这个链条就完全理解了.

很多时候, 大家应像DeepSeek一样聚焦基本功，少无端猜测，归因外部。这也是渣B还在继续做一些关于MoE算法和基础设施协同优化的研究. 例如前面所述和MoE相关的内容

[《谈谈DeepSeek MoE模型优化和未来演进以及字节Ultra-Sparse Memory相关的工作》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493218&idx=1&sn=f394f39a4346fd09a19008a53d0a8022&scene=21#wechat_redirect)

和数学基础算法相关的, 是否存在一些非线性空间的高性能学习算法? 代数结构上的一些约束是否可以构成RL的一些Reward特征? 通过RL修改每一层attention-score中softmax的temprature是否也是一条路?

[《大模型时代的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)

还有就是算法和基础设施协同相关的

[《谈谈AISys架构师的基本素养》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493090&idx=1&sn=f23cac95c9e8e7486dbffa8f6aa99208&scene=21#wechat_redirect) ,[《GPU架构演化史》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3596388845010403333#wechat_redirect)

当然还有更多对国产算力的支持, 例如芯片的ScaleUP和ScaleOut互联和Tensor运算的分析等

[《AI加速器互联》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3596388845010403333#wechat_redirect) ,  [《Tensor运算》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3557619493198151684&scene=173&subscene=&sessionid=svr_32119fe6ccb&enterid=1722676230&from_msgid=2247491424&from_itemidx=1&count=3&nolastread=1#wechat_redirect)

在这个时代, 需要一群安心做技术的人, 大家加油.

参考资料

[1] 
https://finance.sina.com.cn/tech/roll/2025-02-14/doc-inekivrz1692453.shtml: https://finance.sina.com.cn/tech/roll/2025-02-14/doc-inekivrz1692453.shtml