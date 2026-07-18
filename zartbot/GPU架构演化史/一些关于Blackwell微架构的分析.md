# 一些关于Blackwell微架构的分析

> 作者: zartbot  
> 日期: 2025年1月25日 11:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493041&idx=1&sn=97009afc89e8a031d7bfd63a2cc9a56f&chksm=f995f573cee27c65066a29ff5403ad79beefe3a64c0c7ab59822e610270699aca7ea6136fc06#rd

---

Blackwell Architecture的文档一直没有官宣更新出来,  前几天看到CUDA 12.8 Release Note中添加了一些关于Blackwell的内容, 于是做了一些分析, Blackwell Compute Capability为10.x, Hopper为9.0, 后面的CC 12.0是RTX5090的Core?

![图片](assets/1d6db0c40e1b.png)

可以发现SM的Shared Memory为228KB和Hopper没有变化.warp调度这些也没变化,然后从另一个Blackwell Tuning Guide[1]可以看到L2Cache从Hopper的60MB增加到了126MB, 然后同样可以对Kernel的L2 Cache persistency做一些配置

```
cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structurestream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointerstream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratiostream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hitstream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.//Set the attributes to a CUDA stream of type cudaStream_tcudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

其实这里面对于Infra的优化还有很多可以挖掘的地方, 特别是PP/DP/TP/EP/CP这些多个并行在一起的时候, 只是现阶段好像还没有太多的人关注到这一块的优化, 多个Kernel一起争抢L2Cache的容量和带宽时, 做一些取舍还是很有必要的. 虽然看上去B200的L2Cache比Hopper的60MB翻倍了, 但实际上B200是两个Die合封的, 所以容量上算单个Die也就增加了3MB, 推测SM数量从Hopper的144增加到160, 反而平均每个SM的L2Cache的量变少了

总体来看吧, 微架构的变化和Hopper比起来似乎并不大, 或许是CUDA 12.8只是兼容Hopper这一代的, 而Blackwell专有的功能或许要在CUDA 13.0中才会出来? 但总体来说, 我对Blackwell的微架构的变化是挺负面的评价, 或许很多精力花在MCM上了?

[《英伟达GB200架构解析4: BlackWell多die和Cache一致性相关的分析》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489759&idx=1&sn=2c55ec63d6deaeb39ff7f767896ba853&scene=21#wechat_redirect)

另外, 看到晚点的一个介绍字节Seed Edge的Post

![图片](assets/209f9ab01e32.png)

大概比较懂的就是第三题, 然后第一题很有趣, 有些工作已经在展开了, 第二题还需要大量的TOPOS相关的理论来支撑, 第四,五题可能要从前几日的文章来看待, 和第三题紧耦合.例如Attention Block中的MAG/MAC, GNN adapter, 然后MoE block中的MoE Group把Gate Function做二级页表索引

[《谈谈大模型架构的演进之路, The Art of memory.》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493032&idx=1&sn=206eed2e4127b9971a1e0c380f70b082&scene=21#wechat_redirect)

参考资料

[1]
Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#nvidia-blackwell-tuning