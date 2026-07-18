# 再来谈谈AI落地的事情

> 作者: zartbot  
> 日期: 2024年8月2日 01:34  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491072&idx=1&sn=2d27261232f41a8be8234c2a8caa4058&chksm=f9960ec2cee187d452f8d6cbfc8f340919281446f873a5dd92cc20dd8092f51b7db3e8dd7e8d#rd

---

Meta 2024财年第二季度未经审计财报：营收为390.71亿美元，同比增长22%, 看样子还是应验了以前的两篇文章, 还是直接能撸钱的业务搞AI落地比较容易, 所以这个市场上过的比较滋润的也就Meta, 还有国内的DeepSeek**了, MLA的架构带来的短期ROI和开源模型中能力都是不错的. 

[《谈谈AI落地难的问题》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488792&idx=1&sn=d41f2715b27c182eb8f8bc48547f7273&chksm=f99605dacee18cccc1468bbee9c3b8a9269e769320d5847c5ceb7ca8d2a44ce86c6e3a66d3c6&scene=21#wechat_redirect)

[《谈谈AI落地容易的业务-搜广推》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488831&idx=1&sn=192ac23cf674db05d80576c6eac2200f&chksm=f99605fdcee18ceb926a9f59682c7203cc589305be08d9e3cf289da2f489ab50826654e66b88&scene=21#wechat_redirect)

而另一方面微软的财报引起了市场的一些负面反馈, 被迫又要抛出 AI货币化的言论.  国内其实也逐渐开始了一些基于算力证券化/结构化的建设, 前段时间有一个一直做非标和不良处置的好朋友在跟我聊算力,  看到小红书/咸鱼上都在零星的卖一些H卡, 估计过不了多久算力不良资产处置...

IaaS**其实本身就是一个资产证券化的过程, 但是机构投资的抵押物要求又被迫让一些大模型公司开始自建集群. 

[《从金融的视角谈云计算》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488712&idx=1&sn=c3ae48e1c0b3fe9bebfdf2b6d8d5bc0f&chksm=f996040acee18d1c86d93b9a44ad9de972f57e0c99086e04a7d20a342f9175ade81fb8e1b51d&scene=21#wechat_redirect)

做大规模, 提升弹性才是算力证券化的关键.

另外沐神讲了一个八卦[1]

![图片](assets/e9b0f90204e6.png)

![图片](assets/53101902095e.png)

从Llama3.1的Technical Report来看, 大概2个2.4万卡的集群, IB的被用来训练较小的几个模型去了. 而RoCE**的这套训练15.6T Token时ContextLength只有8K, 早期也只是在8000卡上训练, 中间才慢慢换成16K卡.

![图片](assets/243e6788b56b.png)

也就是说MoE的模型可能在中间某个时间被停掉的, 停掉的原因是因为模型架构太Fancy了导致梯度不收敛呢, 还是训练的MFU上不去,例如导致整个15.6T的训练时长要长到100天呢? 至少有一些公开的信息是, Meta解决不了Alltoall的问题,去年还在Call for Action.

当然还有一些原因, 正如我以前在[《大模型的数学基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3210156532718403586#wechat_redirect)中谈到的

[《大模型时代的数学基础(2)》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488528&idx=1&sn=fa49e334201e738e7ddb4258030798b3&chksm=f99604d2cee18dc45a78ee39db2f1c493b4e3f4fae6c3a8ef0b04d1aff8590b8a2b259827f74&scene=21#wechat_redirect)

从representable presheaf的视角来看, Dense的Foundation Model**效果肯定会好, 但是推理的ROI可能就非常差了. 当然用405B来做一些蒸馏和后续更大模型的训练前的数据清洗还是非常有价值的. 等过段时间有空再来更新吧..

参考资料

[1]
MoE没训练出来: https://www.bilibili.com/video/BV1WM4m1y7Uh/