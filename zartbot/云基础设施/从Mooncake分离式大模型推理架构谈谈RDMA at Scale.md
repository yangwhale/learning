# 从Mooncake分离式大模型推理架构谈谈RDMA at Scale

> 作者: zartbot  
> 日期: 2024年12月2日 23:37  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492691&idx=1&sn=584daa6901215ec87af037e997f8421e&chksm=f995f491cee27d8791a50a8bbf1e834d20e65f210c6b466fba9fdac07f1c643cd043cbc92e89#rd

---

### TL;DR

最近Mooncacke KV-Cache为中心的分离式大模型推理架构开源(github.com/kvcache-ai/Mooncake)了, 看了一下README意外的发现支持eRDMA, 谈谈个人的感想, 所有观点仅代表个人与作者任职的公司无关.

![图片](assets/2571f99632be.png)

性能测试的Demo动画也是用eRDMA跑的

![图片](assets/a9517fcc814b.png)

**问了一下eRDMA研发的同学们, 大家都说没有支持过他们, 其实这样一个能够自服务的产品就算成功了.** 

支持标准的RDMA Verbs RC生态, 同时又解决了大规模组网的问题, 用户不需要考虑繁琐的PFC/ECN参数配置, 这样就构建了一个自服务的RDMA网络, 并具备大规模部署的能力(这才是真正的RDMA At Scale). 同时它基于VPC FrontEnd网络, 这样不同种类的GPU(H20/L20)和CPU实例都可以通过RDMA进行大规模组网, 并充分满足客户多种模型对多种算力的需求.

![图片](assets/99d7db8596f7.jpg)

在全球所有CSP中, 这是独一份. AWS SRD并不兼容标准的RC生态, 而Google的Falcon还没有完全落地,当前还有很多实例采用的是GPUDirect-TCPX来支持. 而微软还在采用大量的Infiniband部署.

阿里云eRDMA在新的H系列/L系列GPU实例上以及所有第八代通用计算的CPU实例上都可以免费开启, 并且做到了全地域全可用区的供给, 除GPU实例外,线上在HPC/数据库/大数据/消息中间件/Redis/容器网络等多个场景也有大规模部署.

### RDMA at Scale的问题

对于客户而言, 线下有大量的机房采用Nvidia(Mellanox) CX系列RDMA网卡, 几乎所有的RDMA生态都是基于标准的RDMA Verbs RC语义构建的. 当然当它构建在商用以太网(RDMA over Converged Ethernet ,RoCE) 上时, 存在大量的问题, 如下图所示不支持多路径和乱序提交, Go-back-N效率差, DCQCN难以调整, 很难做到大规模组网, 这也是UltraEthernet出现并需要解决的.

![图片](assets/717cb8bf4fbc.png)

### 为什么一定要兼容RDMA Verbs RC生态

AWS SRD是最早在FrontEnd VPC网络构建RDMA技术的CSP, 但是很遗憾它为了解决VPC网络中的拥塞和抖动以及多路径和乱序提交的问题, 没有采用标准的RDMA Verbs RC语义, 而是构建了一个Scalable RD语义.

当然对于GPU互联的场景,它可以采用在NCCL上构建插件的方式, 但对于更广泛的生态, 特别是一些老旧的HPC应用, 商业数据库应用, 高性能并行文件系统等已经实现了标准的RC语义的应用生态, 需要大量的适配工作, 因此需要大量的人力来进行沟通, 客户通常也需要消耗大量的学习成本.

因此对于CSP而言,需要一种兼容RDMA Verbs RC生态, 它可以使得线下机房的代码一行不改即可在云端部署,并且不需要复杂的DCQCN参数调整,也无需考虑PFC风暴带来的影响, 完全能够做到超大规模的部署.

阿里云eRDMA就是这样一种技术, 在最新的H系列和L系列的GPU上,以及所有的第八代通用计算CPU实例都可以支持, eRDMA技术基于阿里云CIPU构建,无需额外配置专用RDMA网卡即可实现, 因此也做到全地域全科用去的部署. 并且在eRDMA基础上还有SMC-R的生态, 可以在用户不改任何代码的基础上对TCP Socket通信的应用进行加速.

前几个月在[《谈谈大模型推理KVCache加速和内存池化》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490427&idx=1&sn=759b9751469885ac0122943ea61ef2c4&scene=21#wechat_redirect)也谈到了类似的问题, 如果CPU和GPU实例都可以通过FrontEnd VPC通信, 那么很多CPU的内存就可以被用作KVCache存储了, 同时针对一些小规模的模型Decode阶段用一些较便宜的CPU/L20/A10推理也成为降本的一种手段.

### RDMA At Scale over VPC的难点

为什么eRDMA是全世界独一份做到RC兼容又大规模部署在VPC网络中的RDMA技术呢?

其实它的背后隐藏着很多工业界的难题, 在大规模的物理组网中通常采用CLOS架构, 由TOR/Leaf/Spine交换机组成, 数据包需要经过多个交换机多条链路转发, 由于转发路径的Hash冲突导致了局部的拥塞

![图片](assets/ca6040191994.jpg)

另一方面, RDMA需要和VPC的大量TCP流量混跑, TCP流量的微突发和大象流的干扰也使得RDMA通信容易受损. 如果将RDMA放入更高优先级队列又会导致影响正常的VPC中其它租户的流量, DCQCN的ECN信号也因为流量混跑和微突发导致错误的信号, 甚至因此产生PFC阻塞和死锁的现象. 在超大规模部署中,靠调整交换机ECN门限几乎是不可能实现的. 而标准RoCE导致的丢包Go-Back-N重传又极大的影响了传输效率.

![图片](assets/09b184e2221f.png)

工业界在以太网上支撑RDMA业务走了大量的弯路, 可以参考

[《RDMA这十年的反思1：从协议演进的视角》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489240&idx=1&sn=53c7512d8551a44834bd405fd38b15dd&scene=21#wechat_redirect)

[《RDMA这十年的反思2：从应用和芯片架构的视角》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489285&idx=1&sn=3a53f4d177aca0a2a052450fd1a58fe2&scene=21#wechat_redirect)

AWS为了将RDMA运行在VPC网络放弃了RC语义兼容, 采用自定义的SRD实现, 具体的背景如下

[《RDMA这十年的反思3：AWS HPC为什么不用Infiniband》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489300&idx=1&sn=3ad44db6269dcf74c885d33f04885019&scene=21#wechat_redirect)

Google Falcon通过Swift拥塞控制算法进一步演进, 但还没有完全落地, H100部分实例还靠GPU-Direct-TCPX撑着,集合通信Fabric的利用率只能做到60%左右,即便是新的H100实例也开始采用CX7部署, 却无法用在通用计算实例上. 微软还在继续使用Infiniband构建HPC/GPU实例. 即便是Nvidia(Mellanox)的Spectrum-X方案还只能在Lossless的网络中开启AdaptiveRouting.

[《谈谈英伟达的SpectrumX以太网RDMA方案》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489018&idx=1&sn=25b3df2a17d49681edc0e621049b058f&scene=21#wechat_redirect)

如何在保证RDMA RC语义兼容的情况下,并且同时能够在VPC网络中应对多种突发流量拥塞/Hash极化带来的非均衡负载, 同时又不会因为过大的流量影响到其它VPC上租户的流量, 并且能够在GPU集合通信时能够达到97%以上的交换网利用率, 这是一个非常难的课题.

### 为什么eRDMA可以做到

eRDMA采用了2002年iWARP的Direct-Data-Placement技术实现了乱序提交和保序完成的能力

![图片](assets/1172871cad55.png)

这样数据包经过多个路径乱序到达的问题就可以解决了, 因此就是实现了97%的Fabric利用率

![图片](assets/d66c4f6f2879.png)

而时至今日, Nvidia(Mellanox)由于RoCE协议定义的缺陷, 仅能在WRITE操作上实现DDP的能力.

另一方面在拥塞控制上, eRDMA采用了类似于Swift的基于RTT的细粒度滑动窗口控制协议,并支持SACK等选择重传技术. 这样就避免了重传效率,同时也压低了交换机Buffer的使用率, 我们也可以看到Google Falcon继承了Swift的拥塞控制机制

![图片](assets/566ae8f46044.png)

而Nvidia(Mellanox)最近的NVCC拥塞控制算法也在朝着RTT Based的方式演进.

整个eRDMA方案无论是拥塞控制还是多路径转发都不需要借助交换机的辅助(ECN/PFC/INT/PacketSpray/VOQ等都不需要), 因此降低了大量的运维负担,完全做到了用户自服务开启的能力.

### 写在最后

在今年云栖大会发布的CIPU2.0支持了单卡400G转发能力,同时支持VPC流量的安全加密能力, RDMA数据通信将变得更加安全, GPU服务器推理效率和数据访问效率也更加高效.

![图片](assets/e347fd29e054.png)

除了GPU场景外, 当前eRDMA已经在HPC/数据库/大数据/消息中间件/容器网络等多个场景进行了大规模部署, 而Redis等业务也提供了原生的RDMA支持能力, 这些现代化的云基础设施将极大的改进用户用云的效率, 同时用户使用eRDMA无需支付任何额外的费用, 开箱即用. **它是全世界独一份做到RC兼容又大规模部署在VPC网络中同时还能支持数据加密的RDMA技术.**

欢迎各位来使用, 当前H系列和L系列GPU以及所有8代CPU通用计算实例都可以在全地域全可用区免费开启eRDMA能力, 加速您的各种通信传输效率. 甚至是您的一些传统的TCP Socket业务, 也可以通过基于eRDMA的SMC-R技术, 无需修改任何代码直接享用高效高质量的数据传输服务.

更多的内容可以参考阿里云官网eRDMA介绍[1] 以及今年英伟达GTC上的介绍基于阿里云eRDMA的训练实例大幅提升多机训练性能[2]

参考资料

[1]
eRDMA产品介绍: https://help.aliyun.com/zh/ecs/user-guide/elastic-rdma-erdma/
[2]
基于阿里云eRDMA的训练实例大幅提升多机训练性能: https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52281/