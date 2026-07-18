# 谈谈搜广推场景的RDMA

> 作者: zartbot  
> 日期: 2025年8月11日 10:45  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494628&idx=1&sn=54fdc65bb011eae0daf62b0a5e31a546&chksm=f995fb26cee27230b5db980a8135151a7290e6badbe3247e1fe021a32ec1a607695d8c1d18bc#rd

---

### TL;DR

前段时间看到快手一个很不错的RDMA相关的工作: 快手DHPS：国内首个实现基于RDMA 通信的可负载均衡高性能服务架构！[1] 周末和快手的同学简单聊了一下, 今天稍微展开详细分析一下, 并介绍一下我们在CIPU eRDMA设计上的一些思考.

### 1. 搜广推业务概述

关于推荐系统的介绍可以参考以前的一篇文章[《谈谈AI落地容易的业务-搜广推》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488831&idx=1&sn=192ac23cf674db05d80576c6eac2200f&scene=21#wechat_redirect)以及 《小红书训推异构引擎的设计与应用》[2].

Embedding Table巨大的存储需求使得这些系统通常采用CPU和GPU异构部署的方式构建. 正如快手DHPS公众号文章提出的那样, 随着模型参数量剧增, 特别是模型需要捕获用户更长期更丰富的行为, 同时物料Embedding也快速增长时, Embedding Table这些存储节点和GPU推理服务器之间的通信还在使用TCP导致CPU占用率占用高、延迟高、吞吐低等劣势, 严重制约了服务响应时间,限制了模型预估机器的ScaleOut规模.

![原图来自于快手DHPS](assets/e381328dbfa5.png)
原图来自于快手DHPS
### 2. 搜广推中的RDMA

快手DHPS主要工作是通过端网协同的方式, 在搜广推场景中使用了RDMA能力, 并开发了一个高性能的RDMA网络通信库opt-rdma. 具体细节做了很多不错的工作.

#### 2.1 搜广推RDMA部署难点

通常CPU集群和GPU集群部署时, 通常放在不同的Pod, 异构节点之间的通信会存在大量约束

![原图来自于快手DHPS](assets/1c44bb3133e3.png)
原图来自于快手DHPS
首先对于通用计算节点, 部分老的服务器并没有RDMA网络, 构建独立的RDMA网络成本也很高. 因此通常需要实现超大规模RDMA和TCP混跑的能力, 两种流量的拥塞控制算法有着很大的区别, 混跑时的公平调度也是非常困难的.

另一方面是由于部署时, 通常不同类型的服务器放置在不同的Pod中, 通常数据需要跨越Pod甚至跨越AZ传输.

当需要大规模组网时, 并且需要和TCP混跑, 给RDMA的拥塞控制带来了极大的挑战.

#### 2.2 快手DHPS

首先快手通过在MLNX网卡上使用自研的拥塞控制算法, 利用 RTT + ECN + TX_Event这些精细化的信号构建了Rate-Based结合Window的拥塞控制.  然后实现了Lossy RoCE的支持, 避免了PFC带来的影响. 同时一定程度上支持了多路径转发. 最后构建了TCP和RDMA双栈混跑的能力. 总体来看整个系统架构如下:

![原图来自于快手DHPS](assets/b6e1e916a245.png)
原图来自于快手DHPS
根据快手的消息来看, 跨越4层网络后的带宽利用率为90%, 很不错的工作. POD内P99延迟比DCQCN好30%, POD间比TCP好30%. 同时整个协议栈也保持了兼容性.

### 3. 谈谈基于eRDMA的方案

#### 3.1 设计原则

当我们需要在云上构建类似的RDMA网络时, 面临的难度比快手的场景更大.  首先在云网络中有大量的租户, 各种业务的流量混跑在一起, 对于开展RDMA业务带来了极大的挑战. 传统的基于Lossless网络无法满足云的大规模部署需求, PFC在Overlay网络中也无法构建, 即便是更改一些协议也会导致多个租户之间的流量干扰影响性能.

另一方面是, 对于云服务提供商期望的是用户自服务的方式, 前几天看到一篇刘博说云的公众号 [《五个人教一个人点鼠标：云服务的荒诞现实》](https://mp.weixin.qq.com/s?__biz=Mzg2NzE4MDExNw==&mid=2247485237&idx=1&sn=e187bd5ff5223455bf0547ed29f6a038&scene=21#wechat_redirect)很清楚的阐述了这一点. 优秀的产品设计应该遵循"Don't make me think"原则, 因此我们在云上提供RDMA时,希望把所有的拥塞控制相关的问题全部处理干净, 用户无需感知. 也无需像DCQCN那样进行相对耗时的调整.

最后, 我们在提供这些RDMA服务时, 期望能够和用户生态相兼容, 显然就是线下的基于MLNX的RDMA应用能够不修改任何代码直接在线上环境运行. 那么我们仅能选择采用RDMA RC Verbs兼容的接口.  另一方面我们还期望能够对原有的TCP应用进行加速, 并且也需要维持原有的软件生态不改变, 因此我们在eRDMA的基础上提供了SMC-R和NetACC的技术.  最后我们的技术选择如下:

![图片](assets/5f7f418f2692.png)

我们选择了在VPC上支持RDMA, 并提供原生的多路径能力, 兼容标准的RC Verbs接口. 同时考虑到运维的复杂性, 我们并没有采用端网融合的技术,而是简单的在端侧构建拥塞控制和可靠传输机制, 降低对已有的DCN的影响. 同时我们在流量公平调度上, 采用了TCP和RDMA相对独立的令牌桶, 使得eRDMA和TCP公平共享带宽. 最后在系统层面, 我们还实现了RDMA的虚拟化, 并支持RDMA热升级和热迁移的能力, 提高了系统的稳定性.

#### 3.2 谈谈RDMA和TCP混跑的难题

对于这些推荐系统通常客户已经发展了很多年, 并且期望能够以应用不感知, 同时以平滑升级的方式升级.  这是这几点业务上的需求, 使得我们需要TCP和RDMA混跑. 相反如果采用独立组网, 带来的网络成本上升还不如多购买一些CPU服务器, 因此我们需要一种零成本的方式来实现性能30%的提升.

当混跑时, 我们发现在RDMA重载流量时, CX6 TCP和RDMA业务流量的平均延迟和长尾延迟都有数倍的上升, 显然这种情况下部署RDMA是得不偿失的. 因此我们在eRDMA的拥塞控制上做了很多特殊的设计.

![RDMA重载背景流时TCP延迟影响](assets/8dc9eb9a1244.png)
RDMA重载背景流时TCP延迟影响
同样的在TCP重载的情况下我们也可以评估RDMA的平均延迟和长尾, eRDMA几乎没有显著的延迟影响

![TCP重载背景流时RDMA延迟影响](assets/77a1ac5cc463.png)
TCP重载背景流时RDMA延迟影响
其实还有一个值得关注的点, 当RDMA和TCP采用不同QP和不同Flow数量混跑时, 如何保证带宽公平性. 这里商卡也需要很多复杂的调整. 而eRDMA设计时就是按照业务类别均分. 当只存在TCP或者RDMA时, 单个业务能够打满带宽. 而两者共存争抢带宽时, 我们保证两者之间与QP/TCP-Flow数量无关.  对比如下:

![图片](assets/df277432a500.png)

正是因为我们解决了eRDMA和TCP混跑的难题, 特别是无需用户任何配置, 完全自服务的方式解决干净后. 用户就可以大量开展相关的工作了.

例如很多搜广推业务在使用bRPC, 可以参考如下文章《基于eRDMA部署高网络性能的bRPC应用》[3]

#### 3.3 基于eRDMA的TCP透明加速

此时有些保守的用户还会问, 能不能不改应用, 在应用无感知的情况下获得eRDMA的高质量网络性能?  我们和操作系统团队一起借助于IBM在大型机种的SMC-R技术构建了一个透明转换的传输方式

![图片](assets/a846d7d8c0a9.png)

执行以下命令, 打开net namespace范围全局替换开关net.smc.tcp2smc. 此后新创建的TCP socket将被转变为SMC socket,而存量的TCP socket不受影响. 同时对于不支持SMC-R的节点, 也可以自动fallback到TCP上

```
sudo sysctl net.smc.tcp2smc=1
```

更多详细的内容可以参考《共享内存通信（SMC）使用说明》[4]

#### 3.4 业务收益

例如我们使用Nvidia开源的WideAndDeep[5]进行测试, ps采用 ecs.c8i.24xlarge 第8代CPU实力, worker: ecs.ebmgn8is.32xlarge 即采用8卡L20.

当使用SMC-R技术, 在用户完全无感知的情况下, 多种PS/Worker数量配比下, 整体吞吐提升接近30%, 延迟下降30%.

### 4. 小结

本文分析了快手DHPS相关的工作, 通过在PCC上的一些开发, 快手实现了Rate Based + 近似Window的拥塞控制机制, 并且在搜广推上获得了性能收益. 同时构建了大规模的跨AZ的lossy组网架构, 并完成了一些TCP和RDMA混跑时的负载均衡相关的工作.

在第三章, 我们详细介绍了eRDMA的一些设计原则, 以及如何实现用户无配置且应用无感知的情况下, 借助CIPU eRDMA和SMC-R技术, 在用户完全无感知的情况下, 获得搜广推系统30%的性能提升, 同时延迟也下降近30%

阿里云eRDMA技术从第八代通用计算实例开始, 实现了全地域(Region)全AZ的部署, 用户无需任何额外的费用即可使用eRDMA技术构建高性能网络, 对于搜广推/数据库/大数据等数据密集型应用的客户, 还不赶紧来尝试一下?

参考资料

[1] 
快手DHPS：国内首个实现基于RDMA 通信的可负载均衡高性能服务架构！: *https://mp.weixin.qq.com/s/rrwzlQIG0XbMvApHZilAkg*
[2] 
小红书训推异构引擎的设计与应用: *https://zhuanlan.zhihu.com/p/714266999*
[3] 
基于eRDMA部署高网络性能的bRPC应用: *https://help.aliyun.com/zh/ecs/use-cases/deploy-brpc-applications-with-high-network-performance-based-on-erdma*
[4] 
共享内存通信（SMC）使用说明: *https://help.aliyun.com/zh/ecs/user-guide/smc-instructions*
[5] 
WideAndDeep: *https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep*