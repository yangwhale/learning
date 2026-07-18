# 再来谈谈以太网GPU互联：一块价值10亿美金的胶布

> 作者: zartbot  
> 日期: 2024年4月29日 12:27  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489846&idx=1&sn=c61d8c584032732ed20592c05d8a6862&chksm=f99609f4cee180e2ecda3d6156f5d966290f99e58eefcaccbf8f7280ba891c667a727cb0a28c#rd

---

Jensen讲B200这些东西花了100亿美金，而Jim Keller说Ultra Ethernet花个10亿就能搞定。反复的品味一下这块价值10亿美金的胶布， 但是这块胶布要用好确实不容易，至少Ultra Ethernet还搞不定。

![图片](assets/33f6c58f5d76.png)

另一方面最近一段时间准备再一些Intel Gaudi3的分析的，发现公众号`IT奶爸`已经有几篇很好的文章了，那么就把链接放在下面

[《不是NV用不起，而是以太网更有性价比 ——Gaudi 3 技术白皮书解读-1》](http://mp.weixin.qq.com/s?__biz=MzIzMzIyMjIyMw==&mid=2650169229&idx=1&sn=95a649b277f8615ffd31d1c576cdd2fe&chksm=f08a06adc7fd8fbb4ebe591d8470bbe0775715d2a0551b66cb4a0cc4eee29ae0136a4da17f75&scene=21#wechat_redirect)

[《不是NV用不起，而是以太网更有性价比 ——Gaudi 3 技术白皮书解读-2》](http://mp.weixin.qq.com/s?__biz=MzIzMzIyMjIyMw==&mid=2650169258&idx=1&sn=835618af1c69b032a22c7bf67b12ca2d&chksm=f08a068ac7fd8f9ce52b00d2939257e42ba654aba0ef70dce56c1cb86dadb0a9387b9ffb3a6f&scene=21#wechat_redirect)

[《不是NV用不起，而是以太网更有性价比 ——Gaudi 3 技术白皮书解读-3》](http://mp.weixin.qq.com/s?__biz=MzIzMzIyMjIyMw==&mid=2650169275&idx=1&sn=684a05879038d469646f393d0c3bee7d&chksm=f08a069bc7fd8f8d26e9ab85615340a4c793a5af03daad6ad9998f47eec707a80647623e502b&scene=21#wechat_redirect)

而我的视角来看，其实并不只是以太网的性价比，而是Gaudi误打误撞走出了一条很优雅的路，只要稍加修改就能释放出巨大的能量来。我们来看看Jim Keller的Tenstorrent Blackhole 12x400G以太网接口，或者微软的Maia 100, 其实就差一层窗户纸。

本质上的灵魂拷问是：为什么要设计成GPU之间非阻塞的拓扑？延迟敏感么？针对Allreduce这些流量可以做到带收敛比性能无影响，而关键是MoE这些模型引入的AlltoAll通信

![图片](assets/ab21372dac87.png)

MoE要玩好也不简单，例如Meta去年OCP Global Summit也谈到AlltoAll这个问题，Burst难控制。但是MoE的通信是否可以优化呢？**本篇后面会谈到Snowflake Arctic的一些相关的工作再配合软硬件协同的设计。**

## 1. 10亿美金的以太网胶布不好贴

对于以太网互联，为什么这块胶布值10亿，因为以太网有蛮多的问题需要去修复，当然不是以太网自身的问题，而来自于RoCE协议实现的问题。我们来举个例子，Llama 3不用MoE当时推测了两个原因：一个是算力本来就是足够的，Dense就好，对于推理框架而言依赖更少生态兼容更好。然后是数据质量的提升还有ScalingLaw可以继续挖掘，并且15T的数据涨点不少。当然还有一个问题是猜测他们在MoE训练上遇到了一些问题，从8B到70B模型之间的间隔时间长达几个月可以看出一些端倪，另外还有2023年10月中旬Meta在OCP熵的一个Session《Analysis of network traffic patterns for collective communication》，针对AlltoAll做了一个测试，总共4台8卡机器，其中三台24个网卡打另外一卡

![图片](assets/909015d1de1d.png)

理论上应该是每个卡均分1/24带宽，但实际上如下：

![图片](assets/a9ce2ee98c12.png)

理论上每个流都应该很均衡，实际上抖得帕金森

![图片](assets/0249a9d40e74.png)

交换机buffer也是一个问题，ECN的水线怎么搞呢？

![图片](assets/da1bac7c4f43.png)

而这些microburst确实让人头疼

![图片](assets/c4561ebf897b.png)

对于Meta也只能发个胶片讲Call to Action，其实仔细看看Google Swift的论文不就清楚了？

![图片](assets/86840d9f129f.png)

这些利用交换机显式反馈对Large Incast是无效的， 那么结论就很简单了不用PFC, DCQCN换Swift不就行了么? 那不就是Google Falcon么？但这事又没那么简单，多路径PacketSpray和Swift协同，以及SACK怎么处理？然后在这基础上集合通信卸载(Collective Offload)这些怎么实施，也是一个难题，所以这块胶布还是值10亿美金的，但是UltraEthernet似乎从现在的工作组情况来看很难给出一个很好的答案。

## 2. 胶带的四种用法

孔乙己说到: “胶带有四种用法，你知道么？”

“不就是die2die，ScaleUp，ScaleOut，FrontEnd么？”

英伟达对于需要三套网络松耦合的机制做了很多解释，个人觉得这很苍白

![图片](assets/4fa726af8f4c.png)

来捅一捅它的痛处，很简单的一个逻辑，因为有三种不同的协议，所有需要三个不同的网络形态。特别是针对FrontEnd和ScaleOut网络的解释，为什么GB200需要接两个CX8的基础上还要再一个BF3？裸金属的是一部分，而更重要的是需要一个以太网和外界连接...你可以看到AWS无论是GH200还是GB200，都是直接Nitro合并了ScaleOut和FrontEnd网络。另一方面还存在一个悖论是，英伟达有不少的论文在讨论GPU Direct Storage，但是在FrontEnd和ScaleOut之间又生硬的把存储放到了FrontEnd，问题在哪呢？

## 3. 谈一谈以太网万能胶带

### 3.1 ScaleOut和FrontEnd

这两种胶带不同之处英伟达讲了很多，本质上是Infiniband和以太网协议不同，但是加上Spectrum-X的以太网ScaleOut后又要谈论一下东西南北流量的区分，还把Lossless拿出来讲。但是AWS不管是H100还是GB200，依旧直接Nitro EFA搞定。

AWS AI infrastructure and services already have security features in place to give customers control over their data and ensure that it is not shared with third-party model providers. The combination of the AWS Nitro System and the NVIDIA GB200 takes AI security even further by preventing unauthorized individuals from accessing model weights. The GB200 allows inline encryption of the NVLink connections between GPUs, and encrypts data transfers, while EFA encrypts data across servers for distributed training and inference. **The GB200 will also benefit from the AWS Nitro System, which offloads I/O for functions from the host CPU/GPU to specialized AWS hardware to deliver more consistent performance**, while its enhanced security protects customer code and data during processing—on both the customer side and AWS side. This capability—available only on AWS—has been independently verified by NCC Group, a leading cybersecurity firm.

AWS and NVIDIA Extend Collaboration to Advance Generative AI Innovation[1]

### 3.2 ScaleUp和ScaleOut

第一个例子是Google的TPU，当然传输协议物理层上是不是以太网不知道，它是一个很灵活的通过OCS把ScaleOut和ScaleUP连接起来的工程实现

《大规模弹性部署：Google如何管理TPUv4集群》

![图片](assets/ec9c13564fd8.png)

另一个可以用胶布的英特尔的Gaudi3，只是他们自己还没意识到这个问题，都是以太网，用21个ScaleUP和3个ScaleOut

![图片](assets/25cf85bd9553.png)

为什么不直接搞个交换机把24个口全部同构起来统一ScaleUP和ScaleOut呢？或者像Microsft的Maia用的如下拓扑互联呢？

![图片](assets/11cb777fefb7.png)

### 3.3 D2D + ScaleUP + ScaleOut

能够把D2D + ScaleUP + ScaleOut这三张胶布混一起的大概只有Jim Keller了

![图片](assets/35998ab7e6df.png)

![图片](assets/a52ff6b79728.png)

![图片](assets/afa3f3edd8d6.png)

当然还有一个Dojo

![图片](assets/8890a32d29b4.png)

### 3.4 英伟达 Simba

英伟达也干过一颗Simba，很多场景并不需要一颗大的MCM-GPU和过多的Cache结构

![图片](assets/ebd43457895e.png)

其架构如下：

![图片](assets/a20e7a759f6d.png)

![图片](assets/ce1a7e29659b.png)

### 4. 胶带的绑法

胶带绑的好是艺术，绑的不好就成了事故，最终看的还是MFU。

首先我们来看各种并行策略的通信原语，流水线并行就是一些标准的Send/Recv，对于张量并行和数据并行这类的Reduce-Scatter/Allgather/Allreduce语义通信Torus-Ring其实也能较好的满足需求，唯一困难的是MoE出现带来的AlltoAll通信。

MoE 架构最初的承诺是在不增加推理和训练的计算成本的情况下提高模型质量。但是MoE并没有被工业界详细的研究，例如每个专家的规模，应该要多少个专家，TopK选择多少，MoE的层数等，是否Dense和MoE交替或者并行出现？Snowflake做了一些测试最终产生了一种Dense-MoE的模型

![图片](assets/a17008d0027a.png)

详细内容可以参考文章《Snowflake Arctic Cookbook Series: Building an Efficient Training System for Arctic》[2]

128个专家Top2 并且和Transformer架构并行，由于SnowFlake团队有不少人来自于微软的DeepSpeed团队，所以自然而然的采用了DeepSpeed Zero Stage-2和专家并行的方式，如果 GPU 总数为 N，专家并行度为 E，则使用 ZeRO Stage 2 将稠密参数缩放到所有 N 个设备，同时将专家分布在 E 个专家中，每个专家Scale到其余 N/E GPU 并使用 ZeRO Stage-2。下图显示了处理并行性的这三个维度以及并行工作负载所需的通信集合的方式。

![图片](assets/3a7339d92e0e.png)

每个设备包含两组参数：密集（蓝色）和稀疏（黄色）。最核心的一个问题是，MoE的通信由于和标准Dense Transformer在一起而被很好的Overlap了。

关于Snowflake Arctic的内容我们将在另外一篇单独叙述。

实际上我们可以看到，在胶带绑法(互联拓扑)中，并不一定需要完全的无阻塞互联，通过模型架构的设计，算子的放置也可以解决一些问题。包括Experts也可以做一些数据并行的Sharding来分散AlltoAll的通信压力。

另一方面是不是可以先接一些Torus Ring做小集群，例如4x4x4 然后全部接到交换机上FatTree，但是FatTree上又带收敛比这样的混合拓扑？那么对于这张胶布而言就需要更多的功能了。

正如我在[《算力受限下的大模型发展和AI基础设施建设》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489813&idx=1&sn=d39b0334306d3b220eca935e9b694e84&chksm=f99609d7cee180c162633581494fa07c8dc92bd79df9f50508c7d7d28ddef3e002832c81a248&scene=21#wechat_redirect) 一文讲的，这是一个系统工程。

参考资料

[1]
AWS and NVIDIA Extend Collaboration to Advance Generative AI Innovation: https://press.aboutamazon.com/2024/3/aws-and-nvidia-extend-collaboration-to-advance-generative-ai-innovation
[2]
Snowflake Arctic Cookbook Series: Building an Efficient Training System for Arctic: https://medium.com/snowflake/snowflake-arctic-cookbook-series-building-an-efficient-training-system-for-arctic-6658b9bdfcae