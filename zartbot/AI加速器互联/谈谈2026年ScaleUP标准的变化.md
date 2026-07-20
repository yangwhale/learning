# 谈谈2026年ScaleUP标准的变化

> 作者: zartbot  
> 日期: 2026年6月26日 00:40  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498909&idx=1&sn=592553ef04245e01b9002d45f5e5aa76&chksm=f995ec5fcee265496b59e11481e59e5397ca6026e8752fda2e1a56629d1114cf796d6a9aed9f#rd

---

### TL;DR

这几天在参加ODCC的夏季会议, 发现产业界有些有趣的变化. 特别是在ScaleUP互联上.  然后被大家吐槽说最近更新少了, 那么就补一篇吧...

### 1. 说个比较有趣的事情

昨天云豹讲了两页ppt

![图片](assets/cd1b02746b4b.jpg)

![图片](assets/6a070424795f.jpg)

差不多5~6年前NetDAM的工作被人捡起来了, 挺高兴的. 其实我一直讲过, 手上做过的很多工作都是领先工业界五六年的存在. 特别是这几天还在和很多厂商讨论RDMA多路径的算法和一些跨AZ跨Region的需求. 3年前我们就完全做完了上生产了, 而且最近eRDMA规模疯狂增长的同时, 只有一两个同学偶尔支持一下客户, 因为几乎没有遇到客户的工单.

例如某客户用来做自驾的模型训练, 最近很长一段时间, 没有发生像RoCE什么的网络中断/拥塞的投诉.. 然而Mellanox(Nvidia)似乎到现在为止, 还在不停的兜兜转转... 即便是OAI设计的什么MRC也是一坨....

[“漫”谈RDMA现代化](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497377&idx=1&sn=a8645893c61e582d9cb8613b70e65506&chksm=f995e663cee26f75ab3cbfef2b940ce0db5890106367ab4f6869983f9a5a383004078380d7fb&scene=21&cur_album_id=3398249338911260673&search_click_id=#wechat_redirect)

当然还有很多人莫名其妙的坚持PFC/DCQCN... 笑而不语.... 这些东西一开始就是个错误... 不知道网络圈是否也有酥神?

### 2. 谈谈ScaleUP标准的变化

其实这也是最近一直被问到的, 怎么看SUE和UALink... 首先来说一下结论:

UAlink的生态在逐渐变好, 参与的厂家也越来越多.

整个生态争议的结论会在大概今年10月会有比较明确的结果.  看两个事情, 一个是AWS的下一代Trainium 对比NVLink Fusion和UALink 是否有明显的差异.  另一个事情是AMD本身的UALink over Ethernet和 Pure UALink的性能差异.

然后这里稍微展开一下细节, SUE和UALink最大的区别是什么:

![图片](assets/2503c255ff50.png)

**SUE: Per-Destination Queue + WRR**

- 按 `{destination XPU, Virtual Channel}` 分离队列
- 同一队列内的命令被机会性打包为单个 SUE PDU（最大 4096B）
- Weighted Round-Robin 跨 VC 调度 + arrival-order within VC
- 关键约束: 每个 PDU 只发往单一目的 XPU
- Work-conserving: 不会为打包延迟发送

SUE通常会在发送端的I/O Die上根据不同目的地的报文缓冲到不同队列中, 然后一个轮询调度器去发送报文到交换机. 交换机根据不同的dst 地址转发. 并在交换机出口队列上有一个缓冲, 最后再传输到目标GPU. 当出现Incast的情况时, 这个交换机的Egress队列很容易积压.

![图片](assets/46871cd96b0f.png)

**UALink: Unified Buffer + DL Flit Aggregation**

- UPLI 各通道信号打包为 64B TL Flit（不区分目的地）
- DL 层收集多个 TL Flit 组装为 640B DL Flit
- 核心特性: DL Flit 内部的 TL Flit 可以有不同目的地
- 多级 Credit-based flow control（UPLI / TL / DL 各层独立）
- 链路始终以满 Flit (640B) 传输 — 最大化带宽利用率

UALink产生的TL Flit 在发送端不区分目的GPU, 直接到一个独立的缓冲区, 当buffer内凑满一个DL Flit后, 直接转发到交换机, 交换机会对DL Flit内部的多个TL Flit逐个进行解析和转发到目标GPU的交换机端口的Egress缓冲区, Egress缓冲区内凑满一个DL Flit后,就会转发给目标GPU, 长尾延迟会好很多.

![图片](assets/c143ad1af7af.png)

`核心差异一句话`:SUE 交换机看到的是完整以太网帧（每帧只发往一个目的地），做标准 L2 转发；UALink 交换机需要拆解 DL Flit、逐个解析每个 TL Flit 的目的地、分别路由、在出口重新聚合 ,  本质上是 Flit-level Crossbar Switch。

实质上交换机实现并没有多大难度, 只是大量做网络的人既没做过芯片, 协议也就设计了一点皮毛就张嘴胡说八道什么拆包组包难...

incast场景下, UALink会比SUE好很多.

![图片](assets/753dba2bffa6.png)

关于这些争议以前都写的很清楚了

[《谈谈ESUN, SUE和UALink》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496512&idx=2&sn=0c10cef05fb1cc4e175f326d62b266e3&scene=21#wechat_redirect)

以太网ScaleUP的始作俑者我承认是我, 大概2020年~2021年的时候伴随着NetDAM的项目就在弄, 思科比较愚蠢看不懂它的价值, 我就滚蛋离开了, 否则现在思科就不会傻傻的只能卖卖交换机芯片了. 差不多到了23~24年的时候在和BRCM一起谈这个项目的时候, 我明确的给出了需要类似于Nvidia FinePack的需求, 我的意思是需要像UALink那样pack多个TL

![图片](assets/1aadaf481746.png)

但是BRCM理解的是当年论文后面一个图, 所以导致了今天SUE pack上的各种问题...

![图片](assets/3b2eee874c3f.png)

当然, 回到2020~2023年的时间点, 那时候能提出来ScaleUP的, 估计也就只有华为UB和我了,  当时很朴素的想法就是, 专用的ScaleUP机柜总线交换芯片太贵. 拿个以太网凑合并承认它的不足, 例如延迟/长尾的代价. 总体来说就是很简单的一个想法用内存语义,对处理器更加友好.

一晃这么多年过去了, 既然BRCM要搞专用芯片方案, 那么整个故事就变了... 我大概率估计工业界会得到一个结果, 基于Ethernet(BRCM TF1)的Ualink over Ethernet的性能会和纯 UALink的差异非常大.

当然BRCM也会说(狡辩), xxxxxx DSA都没问题啊. 仔细展开谈谈这些问题.

通常DSA架构的 TensorCore 都比较大, 例如TPU是256x256 , LPU是320x320. 然后片上通常有一大块Buffer. 从Little’s Law来看, 整个传输过程中大Buffer能够很好的处理长尾的影响, 另一方面较大的Tensor Core, operand的 Tile size都比较大, 因此也不需要考虑小的Flit.

但是当你的GPU微架构是GPGPU的时候, 也就是说, Tensor Core 规模通常是16x16, 然后有较小的SM上的 SMEM , 每个SM的SMEM容量非常有限, 那么考虑Little’s Law 就对小size低延迟低长尾的需求非常依赖了.

简单一句话总结, 如果你的GPU微架构类似于GPGPU, 并且希望做一些CUDA 兼容的市场. 很有可能UALink是你唯一的选择. 如果你是DSA的架构, 两个都可以选, 但为啥不选个更好的呢?

反正今年我看到很多家都在做一些聪明的选择了, 那您呢?