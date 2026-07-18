# AWS Re:invent GenAI路上快速的追赶者, 详细谈谈Trainium2/3架构

> 作者: zartbot  
> 日期: 2024年12月4日 16:34  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492863&idx=1&sn=16930a20429695f8a2c9182a83e97739&chksm=f995f43dcee27d2b4a766686146491e3b12a08ca420bcb084d1a5cdc394ec7e13e42b9e54e4f#rd

---

### 先谈谈大模型发布和GenAI场景

其实最近两年一直有一种声音, AWS在GenAI上落后了, 但是昨晚AWS Re:invent的第二场发布会, 似乎有些后来居上的感觉. 除了和Antropic深度合作外, 它一口气发布了Amazon Nova系列6个大模型, 其中包括Micro、Lite、Pro、Premier四个规模的语言模型, 一个canvas图像生成模型还有一个Reel视频生成模型. 同时还预告了明年会有语音-to-语音和Any-to-Any的两个模型在任何模态内转化.

![图片](assets/e85922ac78d4.png)

![图片](assets/7e883a0cfdb7.jpg)

另一方面, 苹果站台, 说在AWS上推理负载实现了40%效率提升, 并且预估在Trn2上预训练会得到50%的效率提升. 并且伴随着Apple Intelligence的逐步上线, 苹果的Adapter模型功能也会越来越多, 对于端侧推理, 一个较好的小规模的基础模型配备多个任务的Adapater或许是在内存/功耗等约束下最优的一种解法.

![图片](assets/ce4322cfde78.png)

另外JPMC也为AWS站台, 谈论了很多业务场景中使用GenAI的方法. 最后AWS CEO Andy也介绍了GenAI在Amazon AI电商中的应用场景, 例如改善客服、卖家详情页创建、库存管理、机器人、Alexa、Amazon Lens、线上购物衣服尺码匹配、Prime Video等多个场景落地.

然后在整个Bedrock平台上也做了很多端到端的覆盖, 从延迟优化到新发布的模型蒸馏, 然后Fine-Tune工具链, RAG, 自动推理检查, Agent编排和多Agent协作等, 非常全面.

![图片](assets/45273f3cc274.jpg)

特别是已经在AWS内部广泛使用的自动推理检查也增加到了整个Bedrock平台上

![图片](assets/40acef45f812.png)

![图片](assets/84f9ed130f49.png)

当然,还有计算/存储/数据库的很多发布, 例如发布了Blackwell平台的P6

![图片](assets/6a79cbda154e.png)

存储在S3上支持了TableStore, 同时基于TableStore构建了S3 Metadata服务便于检索和其它GenAI数据处理. 然后数据库也做到多主多地域(MultiRegion-Active-Active)的部署

![图片](assets/b406c1789db1.png)

![图片](assets/151930645f6f.png)

### Trainium 3预告

在Trainium2 GA的时候, 预告了Trainium3, 3nm的工艺, 性能翻倍, 能效提升40%, 首批实例预计将在2025年底上市.

![图片](assets/f90cae2b9257.png)

估计Trn3同时ScaleUP域提升一倍到单机支持32卡, 而TrnUltraServer到128卡规模, 拓扑结构上可能会伴随着UALink升级到带交换机的架构. 而当前Trainium2还是一个HyperCube的拓扑.

### Trainium 2架构详解

![图片](assets/59169aa8b31b.png)

SemiAnalysis有一个Trainium2的详细分析[1]

#### Spec Overview

Trn2每个芯片大概500w, 仅需要风冷散热, 单芯片的BF16浮点算力为650TFLOPS,大约为H100的60%左右, 内存为96GB HBM3e, 内存带宽和H100持平

![图片](assets/a74865ea3a9b.png)

然后整个ComputeTray采用无线缆设计, 由8个Nitro卡和2颗Trainium 2芯片构成, 结构简单

![图片](assets/c1e6b4e5d751.jpg)

每个Trainium2芯片配置了4个200G的Nitro5 DPU, 通过EFAv3构建800Gbps单芯片的ScaleOut网络, 单个Trn2实例支持16个Trainium2芯片, Trn2-Ultra则是通过把4个Trn2组合构成的, 从AWS的Datasheet来看, Trn2u拆分出来的也可以支持3.2Tbps网络的, 每个ComputeTray也就400Gbps, 每卡也就200Gbps.

针对Trn-2 Ultra SemiAnalysis对ScaleOut的分析单卡仅200Gbps的带宽数据能对齐, 但是有待考证, 是否ScaleUP的NeuronLink占用了ScaleOut PCIe带宽? 后面我们在互联架构的章节详细阐述.

![图片](assets/7ff9ad16004c.png)

对比TPUv6e和GB200以及H100的指标如下

![图片](assets/d0bb4abd460e.png)

其实从模型的实际MFU来看, H100综合MFU大概也就只能到50%左右 ,所以Trainium2稍低的算力配置应该是恰当的, 从Roofline来看和Google TPUv6e和NV GB200都很有竞争力

![图片](assets/085dd887915f.png)

至少从整体架构而言, ScaleUP不需要复杂的224G Serdes的铜连接器和背板, 也不需要液冷, 感觉整体的可运维能力和可靠性比NV好了很多.

从封装来看, 昨天也Peter也讲清楚了Reticle Size的约束, 采用了CoWoS封装

![图片](assets/76813e7b028c.jpg)

![图片](assets/5fce9df6d60c.png)

#### CPU/GPU/NPU微架构的区别

![图片](assets/d06e3583185f.png)

对于CPU而言, 有大量的分支预测/预取解码/ROB等单元, Cache层次化结构也占用了芯片的大量面积, 它用于提高指令的执行效率, 针对的是大量复杂逻辑的快速执行

![图片](assets/afa943605de9.png)

GPU主要是针对数据并行执行, 控制单元较小, 执行单元众多, 同时有大量的寄存器文件用于在多个执行线程上隐藏延迟

![图片](assets/6dc5364f8022.png)

虽然CUDA SIMT是一个非常优雅的抽象, 但自从进入大模型阶段, 大量的GEMM(矩阵乘矩阵)运算, 并为了应对Google TPU竞争, Nvidia引入了TensorCore. 它破坏了SIMT的结构, 以至于现在要通过各种内联汇编去控制内存访问实现预取, 执行WMMA/WGMMA也要考虑大量的异步抽线, 算子开发难度也越来越高.

而Peter点中了一个NV的死穴, 即便是有Distributed-SMEM, 但是很多CUDA运算Kernel之间的结果还是需要存储到HBM上,再由下一个Kernel加载运算. SM之间的网络无法进行直接的LD/ST协同.

因此AWS采用了脉动阵列的方式来构建NPU, 和Google TPU一致了. 这样也降低了HBM的带宽需求.

![图片](assets/ae32df70e713.png)

#### NeuronCore微架构

使用脉动阵列的优势在AWS第一代推理卡Inferentia就介绍过

![图片](assets/b4d2340d94e9.png)

另外为什么使用片上缓存

![图片](assets/a14887f45859.png)

在NeuronCore-v2第二代上增加GPSIMD引擎, 可以通过C++编写更多的自定义算子

![图片](assets/eb3a2dec2fcf.png)

同时还提供了集合通信卸载的引擎, 并支持4根NeuronLink-v2.

Tensor Engine采用128x128的脉动阵列, on-Chip SRAM分为SBUF和PartialSum(PSUM). SBUF可以用于异步通信的过程中隐藏网络的抖动和延迟, 使得脉动阵列流水线处于满载状态.

![图片](assets/e95bd2500f54.png)
同时对于输出的结果在PSUM中,也可以不存入HBM直接通过NeuronLink发送到其它Trn芯片的SBUF中.

另一方面向量引擎可以在进行脉动阵列计算的同时执行BatchNorm/LayerNorm/Softmax等计算, 调度器可以并行调度各种运算引擎提升效率.

![图片](assets/98edfdb9ac72.png)

而标量引擎可以支持element-wise的操作,例如Exp或者SiLU这样的函数, 或者在矩阵计算结果中添加bias.

Trainium 2和Trn1一样, 也内置了集合通信加速引擎, 大概率这就是一个TMA类似的组件, 允许通信和计算的重叠. 而在NV和AMD上, 集合通信需要XCCL等占用SM/CU的计算资源.  Trainium2推测的芯片架构如下:

![图片](assets/c65f9efdb5d3.png)

### 服务器架构

采用一个2U x86的CPU机头配合 8个2U的Trn2 ComputeTray构成16卡的trn2服务器

![图片](assets/712aa56612ac.png)
然后CPU HeadTray通过PCIe AEC铜缆链接到8个ComputeTray上

![图片](assets/4e378e38f4b6.png)

整个机器的PCIe拓扑如下:

![图片](assets/79324410b1f8.png)

CPU机头也配置了两块Nitro, 一块走EBS一块走前端网络, 并且放置了多个NVMe本地盘

![图片](assets/f0a47830da9d.png)

### ComputeTray

SemiAnalysis分析的结构如下:

![图片](assets/3d01c23bb0ba.png)

Nitro网卡一共8块,放置如下:
![图片](assets/0704b82f3fc7.png)

而在Trn2Ultra中, SemiAnalsys的分析是错误的, 可能他们从一些非官方渠道知道了Trn2-Ultra的ScaleOut规格, 推测互联如下

![图片](assets/136ae49aee6e.png)

系统可以根据售卖策略单独卖一个trn2ultra.x48large, 也能支持3.2Tbps的带宽, 折合出来每卡也是200Gbps. 但是从trn2-Ultra的发布会实物来看但实际上Nitro卡还是8块满配的.

### 机柜结构

Trn2单个机柜可以放置两台Trn2,如下所示:

![图片](assets/1b2e51301846.png)

双机柜可以通过AEC线缆构建4机并行的Trn2-Ultra

![图片](assets/2d1c16e0b537.png)

功耗分析上, 单机柜26kw左右也挺好的, 供电也不需要太大的改造

![图片](assets/e60e635b2032.png)

### Project Rainer

Antropic联合创始人配合AWS发布了Rainer项目集群, 累计40万卡Trn2. 该园区已完成一期建设，目前有7栋建筑，每栋IT电力65MW，总计455MW 。印第安纳州 AWS 园区的第二阶段将再增加 9 座 65 兆瓦的建筑，总发电量为 1,040 兆瓦。

![图片](assets/51daeca9c178.png)

### ScaleUP链接

SemiAnalysis的推测是NeuronLink-v3是基于PCIe 5.0实现的, 配合着AWS进入UALink还带着AsteraLabs, 估计也是有这个目的. 从图上看是一个采用了OSFP-XD的AEC线缆.然后配合官方宣称的2TB/s 1us NeuronLink-v3

![图片](assets/5dd6bfd84fd2.jpg)

按照我以往的经验, AWS从来对Bytes和bits混着分不开的尿性, 这个应该是一个Gen5 x16, 然后算双向2Tbps? 然后在trn2中采用HyperCube的拓扑

![图片](assets/923a36fe19c2.png)

逻辑拓扑如下所示:

![图片](assets/845c094ad2de.png)

它和4x4的Torus 同构

![图片](assets/b43472e0af5c.png)

而针对Trn2-Ultra 64卡, 链接构成4x4x4的Torus结构

![图片](assets/548640d45483.png)

Trn2-Ultra机柜互联线如下:

![图片](assets/38bb7eec8e00.png)

未来是否会使用Optic, SemiAnalysis做了一些展望, 我们可以参考以前整理的一篇文章

[《大规模弹性部署：Google如何管理TPUv4集群》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489642&idx=1&sn=db30c4606db2f181f8f602c8e71abf91&scene=21#wechat_redirect)

如果NueronLink-V3也基于PCIe, 其实对PCIe协议魔改一下,把BDF做成一个维度路由的地址即可实现3D-Torus的路由.

### ScaleOut连接

按照NueronLink-V3构建3D-Torus拓扑, 每个可以拆分出两个Link, 累计8个Link.然后在Trn2中2D-Torus两个维度需要4个Link, 剩下4个可以做ScaleOut, 但是和官方发布的16卡累计3.2Tbps的拓扑对不上. 因此具体的拓扑情况还是要购买实例再来分析一下. 另外NueronLink和Nitro EFA之间线路切换也是一个值得探讨的问题.

按照10p10u网络部署, 拓扑还是基于FatTree的, 并没有多轨道部署, ToR采用12.8T, Leaf/Spine都还在25.6T上并没有使用51.2T的TH5. 布线方式, 然后还有SIDR路由协议等昨天已经详细叙述过了.然后机头还有2张Nitro负责EBS和ENA的流量. 当然值得关注的是ScaleOut的Nitro本身是否接入到VPC中.

反正它EFA做的不行, Nitro带宽停留在200Gbps, 多路径能力虽好但又不支持标准的RCverbs, 就不分析了.真正ScaleOut怎么做看下面这篇

[《从Mooncake分离式大模型推理架构谈谈RDMA at Scale》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492691&idx=1&sn=584daa6901215ec87af037e997f8421e&scene=21#wechat_redirect)

但是基于PCIe的ScaleUP和ScaleOut有一个好处, 可以根据工作负载动态配比拆分切割实例售卖, 对于云的弹性来说是非常重要的.

### 软件框架

![图片](assets/5aaf40ac6f4a.jpg)

由于是脉动阵列, 然后复用了Google TPU的JAX/XLA生态, 这样支持Pytorch也容易了

![图片](assets/6298ef137f3f.jpg)

但是这次发布提供了NKI的API, 做了一个基于Tensor的抽象层.

![图片](assets/2321435a42be.png)

![图片](assets/524c37763596.png)

然后分布式调试工具上和NV Nsight Compute类似的也有

![图片](assets/2d11334ef696.png)

而集合通信还是要基于EFA SRD去搞, 反正和NCCL差不多就不多说啥了.

### 集群管理和负载调度

这些AWS都做的蛮成熟的, 异步的检查点机制, 类似NV DCGM的自动化的故障检测机制等. 然后K8S工作编排等基本上都是标配.

### 芯片供电

昨天发布的, 采用了Vertical Power Delivery的方式, 背部供电稳定性好很多

![图片](assets/105a7610384d.png)

![图片](assets/5f56045aac8d.png)

![图片](assets/948e8d0cf3f5.png)

### BOM分析

SemiAnalysis做了一个BOM分析, 真便宜呀~

![图片](assets/f7c3a696b14e.png)

![图片](assets/f6ee277eaf9a.png)

然后按照4000卡集群做了一个网络成本分析

![图片](assets/1b8340a46de2.png)

![图片](assets/e8617e7daeda.png)

总体成本分析比H100好了很多

![图片](assets/7269a4d4b816.png)

性价比分析

![图片](assets/e055115bf8c7.png)

### Trainium3的猜测

可能和Rubin会比较类似, 更大的支持4x以上Reticle的封装, 4个Die互联. 然后ScaleUP会引入交换机, 然后估计也会把它NueronLink的方案推到UALink去搞. 然后Nitro 5还停留在200Gbps上,太丢人, 估计明年底会和Trainium 3一起出来一张新的卡?

另一方面最近听说NV的Rubin也要加速到明年年底, 这样GB200还有存在的必要么, GB300估计会更加短命...

### 编程框架

自从NV引入TensorCore后, 整个CUDA的可编程能力在肉眼可见的下降, 例如TMA这些, 直到生命周期末期了, 才有FlashAttention3/ TK这些项目用上, 并且还很大程度需要NV原厂的人指导. Hopper的WGMMA临时搞了一个SM90a的补丁, 并且不向后兼容. 很多场景中要想打满TensorCore都还有很多复杂的预取/Warp Divergence的工作要做.

TensorCore的出现把NV SIMT优雅的抽象都破坏了.  以GEMM作为最小单元的编程似乎总觉得有哪儿不对. 另一方面正是因为SIMT和SM Warp的抽象, 使得中间数据无法从片上网络直接访问必须要落到HBM中, Distribute-SMEM也仅是一个很小的范围, Warp调度也有很多难题要处理.

而AWS/Google都在脉动阵列上走了很远了, 但是我也不确定这条路是否是对的. NKI进一步做了一个Tensor抽象的NKI... 但是我觉得这也是很难搞的, 毕竟脉动阵列还要考虑流水线Stall/空泡带来的效率影响, 虽然有一个SBUF可以缓冲一下.

另外的框架就是Tenstorrent的RISC-V相关的实现, 也类似于一个脉动阵列.

但总体来看降低HBM的内存读写开销都是值得的, 特别值得国内HBM被禁的GPU厂商. 然后可编程框架上, 国内常年ToB的2B面向标书的架构被迫让很多家走到DSA的路上去, 摊手....

参考资料

[1]
Trainium2的详细分析: https://semianalysis.com/2024/12/03/amazons-ai-self-sufficiency-trainium2-architecture-networking/#aws-trainium1-inferentia2-genai-weakness