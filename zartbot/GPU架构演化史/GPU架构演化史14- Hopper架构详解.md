# GPU架构演化史14: Hopper架构详解

> 作者: zartbot  
> 日期: 2022年9月7日 16:00  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488380&idx=1&sn=bf83d9150f629adbd46016c0a1ba7062&chksm=f99603becee18aa841afc71cabbfc2e5a003aad98ccc89f5dd2188bec30ff83f109e7a53dc22#rd

---

2017年发布的Transformer模型解决了RNN的难以训练的问题后，在NLP领域独占鳌头, 而它在2020年下半年又火了起来，ViT的发布使得它在图像和视频识别领域也大放异彩. 但是另一个问题来了,Transformer自身设计大尺寸矩阵的乘法, 在TPU这样的脉动阵列上处理性能好于Tensor Core, 而同时Biren / Tenstorrent/ Tesla Dojo / Cerebras / GraphCore等一系列厂家对nVidia在深度学习硬件加速这个市场带来了很大的压力，此刻nVidia又该怎么办呢？

当然作为一个架构师最容易犯的错误就是针对某个特殊场景去做优化，例如前些年很多AI芯片厂商拼命优化CNN，Relu、sigmoid都要全部优化掉，跑分是高了，编程灵活性就下降了，而另一方面nVidia还有大量的图像类的业务，Omniverse这些元宇宙的生意不可能不去考虑， 因此在架构设计上，它们会有它们自己的权衡，所以从某个一个领域来看，你会觉得为什么有些明显提升性能的事情，别人不去做。读懂他们的妥协也很重要。而另一方面收购了Mellanox，A100都还用了IB，而现在全部用NVLink Switch代替了，NvLink也升级到了56G Serdes，而国内还有一大群在RoCE和IB上发力的厂家，不想想为什么？

本文前面会介绍一些Hooper的架构，当然后面还会继续谈谈Grace Hopper以及NVLink Switch， 但是从架构的细节上来看，似乎堆料已经走到了尽头了，接下来又要有一些奇技淫巧了，而SM内各种专用的电路也越来越多...体系架构该如何走也是一个很值得探讨的话题。

### GH100

新特性总结如下:

![图片](assets/51def191d794.png)

采用TSMC 4N工艺，频率提升了不少，满配的GH100有8个GPC，每个GPC有9个TPC，每个TPC内2个SM，累计144个SM，支持第四代NVLink和PCIe Gen5，支持6个HBM3 Stacks，DRAM带宽增加到3TB/s, L2Cache增加到了60MB，基于SXM5的H100砍掉了6个TPC，只有66个TPC累计132个SM， L2Cache也降为50MB，而PCIe板则降为7~8个GPC，57个TPC，累计114个SM

![图片](assets/6a25b4d8931d.png)

### SM

SM内含128个FP32 CUDA Core，同样分为4组，  L1D/SharedMem也扩大到了256KB

![图片](assets/215503c80a35.png)

比较有趣的改动有几方面,第四代的TensorCore , 从A100的4x8x8 到了H100的4x8x16

![图片](assets/160fd418f348.png)

它无论如何也要保留一个4的维度，为什么？因为后面的图形卡Ada Lovelace也会要用到这个Tensor Core呀，另外它们在WMMA去load数据的时候，针对乘法的block也有一些访存的优化。

![图片](assets/cdf0bf81cdc7.png)

总体来看就是乘法器个数翻倍了,然后有更宽的LD/ST Path, 但是需要您注意到的是，它并不完全为了追求矩阵乘法的1000TFLOPS而存在的，另一方面除了AI训练，他们对于超算和渲染集群还是有很大的基本盘的，因此它还需要兼顾传统的图形渲染业务和超算的业务，因此整个SM中除Tensor Core外，FP32和FP64的性能也有了2~3倍的提升，因此HPC的业务基本上这一代性能也提升了3~6倍，有些时候这些顶级架构师的取舍不要轻易的去想他有什么失误的，或者你为什么能够轻易的弯道超车的，新增了FP8的支持就很有趣：

![图片](assets/ba38adc94d5f.png)

你可能会觉得FP8这玩意尾数就2、3能做点什么，有些无厘头的感觉，但事实上你再去看Transformer本身的值分布，同时再看一下< Attention is all you need >论文中Label Smoothing那一段 Epsilon_ls = 0.1

![图片](assets/f93f7575f906.png)

#### Transformer Engine

然后紧接着就出现了所谓的针对Transformer加速的硬件逻辑了，实际上就是根据Label输出的值域动态调整浮点精度

![图片](assets/b0b5f1cd1311.png)

其实正如我昨天谈到的从范畴论视角来看待transformer模型，它已经成为一个新的模型范式，相对于CNN这种更深多达100层的网络，Transformer更加扁平化，对于乘法矩阵的规模也大了很多，从数值计算的角度和计算中的值分布来看，这种牺牲精度的做法是值得的，而nVidia在这个地方也是深思熟虑过后给出的取舍，非常出色的工作。

### TMA

当然Tensor Core的规模加大了，LD/ST Path也需要一系列的优化嘛，于是Tensor Memory Accelerator就登场了,当然这玩意最早做的还是华为昇腾，nVidia现在才出来不一定说先做的就一定好，后做的就是抄，其实每个架构师做什么不做什么都要看周边器件的情况的，等下讲到的Thread Block和分布式的共享内存Bill Lynch在华为的时候10年前也提过，而现在很多AI加速器做片上Mesh网络，思科在2004年就做过了一直商用了15年了。很多东西，先做后做只是看你要解决的问题规模是否值得去这么做。

![图片](assets/0b79f876202f.png)

其实就是在A100中已经实现了一些异步的内存拷贝，但是这些都还要每个Thread来产生相应element的访存地址请求，然后Warp Level去合并产生LDST指令，TMA节省了这一步，并且通过一个异步的信号来通知数据传输的状态，支持1D~5D tensor的拷贝

![图片](assets/57328a9b6d64.png)

整个异步访存流程如下：

![图片](assets/5efb606b8b91.png)

看到这里，想到去年一整年折腾NetDAM的场景，同样的问题，TMA解决的是GPU Global Memory拷贝到Shared memory的问题，而我是解决Host之间的Remote Memory拷贝的问题，也有大量的矩阵搬运、Prefetch的工作来隐藏更大的访问延迟，同时还有一系列的转换逻辑甚至像GPU还需要在Warp level做Reduce，扩展到多机集群那么就是在Host Level做Reduce，RDMA在这种场景下就很难适合了，虽然nVidia有收购自Mellanox的SHARP，但是个人口味上不喜欢...当然nVidia自己也不大喜欢，后面讲NVLink Switch的时候再细说

### DPX

关于动态规划类算法的优化的指令集，现在还没什么资料，要等到年底CUDA12正式发布估计才有详细的细节

![图片](assets/31bea4988fa8.png)

考虑DP，一方面来自于图深度学习的研究，另一方面是超算中也有大量的应用，同时针对自动驾驶等业务都需要这类算法的加速，从体系架构上来看，如果我来做可能就是在以前Dynamic Parallism实现递归的时候，由于有足够的Cache和片上内存空间和足够多的核心满足一些问题的计算规模，再加上后面讲到的Thread Block和Distributed Shared Memory，那么对于递归的Kernel<<<x,y,z>>>(n-1)结果做一些动态Cache，同时Thread之间的协作使得n覆盖在一个较小的范围内。等到年底再来看这块吧，不过反正也禁运了，想玩也玩不了.

### Thread Block

在Hopper以前，CUDA处理问题的规模采用Grid、Block两级调度，Block映射到SM上，但是随着协作组Cooperative Groups的出现和异步编程的支持，多个Kernel之间以生产者、消费者的方式通信， SM到SM之间的通信带宽需求也在增加

![图片](assets/bf993c6b8305.png)

而在Hopper上新增了一个Distribute Shared Memory (DSMEM)的概念，在一个GPC内部的SM有了专用的通信带宽，因此CUDA上增加了一级调度层次：

![图片](assets/e28b82527e36.png)

同一个Thread Block内采用PGAS构成分布式共享内存(DSMEM),在一个GPC内部实现多个SM的LD/ST，Atomic，reduce和异步DMA操作都变得非常的简洁，

![图片](assets/f1c065791b8e.png)

而我在做NetDAM的时候则是将它扩展到更广的异构空间去实现分布式内存共享

![图片](assets/df8317af6f5c.png)

nVidia在这个地方设计使用了一个Transaction Counter的方式来构建异步操作：

![图片](assets/ff11f0971616.png)

做的也非常的聪明呀

![图片](assets/9aa9f4161fb2.png)

就像我自己也是采用类似的方式，在NetDAM上开辟一块Atomic计数器来构建一个操作是否完成的通知，NetDAM-Seq可以为Transaction Counter提供每秒5.92亿次的操作能力

另外说一句，这些SM-to-SM network 和分布式内存，本质上属于一个新的Mesh的片上网络，也为Grace CPU构建SCF起了很大的作用，点到为止了

![图片](assets/5c0c1c88ab18.png)

而且正是这样的片上网络使得Grace和Hopper的组合变得非常有趣了

![图片](assets/04912048e46f.png)

和Apple M1过分强调并行的片间互联不同的是，NvLink使用了PAM4 Serdes，它的取舍其实在NvSwitch上，而且封装的灵活性更好，2xGrace，1xGrace+1xHopper，说不定哪天还会出一个1xGrace+2xHopper的玩意.

![图片](assets/d29a3287a246.png)

同时再想想fugaku的TofuD，有太多可以玩的东西了

![图片](assets/3543bd0d2f75.png)

### NVLink 4.0

从HotChip34上看到的NvSwitch的介绍，您可以发现，SM2SM的网络一部分直接连在了NvLink接口上,而其它的NvLink连接到了片上的L2Cache

![图片](assets/991a2c2f992d.png)

然后这次使用了多根50Gbps-PAM4，提供了片间900GB/s的吞吐：

![图片](assets/e68437a10a56.png)

虽然有112Gbps PAM4的Serdes可以用，但是Nvidia这个时候还是选择50G和多根link，一方面是总带宽900GB/s够用了，同时用相对低速的还可以获得更加灵活的拓扑连接，并且为大模型训练时参数同步的AllReduce类业务做更好的优化，想想看交换机要做SHARP呀

![图片](assets/12ff750f4b6b.png)

另一个问题来了,那么是什么动机让他们连RDMA和Infiniband都不用了呢?

![图片](assets/6060c49866d4.png)

其实这一点我是认同的， 毕竟我在做NetDAM的时候就明确发现根源就在DMA上.我们来看NVSwitch的架构：

![图片](assets/caae6b9b4387.png)

看看SHARP Scratch SRAM+ SHARP ALU，看看PortLogic和NetDAM，你就会清楚了，只是我们考虑到更加通用的场景，连XPU端选择了CXL，连网络侧选择了向下兼容Ethernet，当然以后换成CXL3.0未尝不可

![图片](assets/79ade5f7557b.png)

再想想我当初怎么说RDMA的，看到一群做RDMA的DPU厂商...

![图片](assets/dfc0606baee7.png)

当然NvLink在上面还有一个问题就是解决RDMA编址的问题，它有了一套自己的寻址机制，这个值得我们在设计CXL3.0的时候去仔细思考

![图片](assets/f1cfd3c3f3a7.png)

Link TLB又勾起了我另一个回忆

![图片](assets/09099ead364c.png)

而另一个问题是Multicast,同样我在做NetDAM的时候也用了

![图片](assets/4156f0f4a167.png)

### CNX

顺便说说DPU领域，这一代nVidia也开始将Hopper和CX7弄在一起，试图在边缘网络中提供算力

![图片](assets/36ea4f67aa83.png)

所以你也会看到这一代的Hopper MIG支持了一些可信计算的能力

![图片](assets/c158f9180e46.png)

这一点我是挺喜欢的，对于联邦学习这样的场景也很有趣

![图片](assets/69b92ca91404.png)

4年前在思科领导过一个分布式AI计算的研究任务,最终获得全公司的一个大奖，也是中国团队第一次问鼎该奖，

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

有基于LSTM的DNS恶意域名探测

![图片](assets/5fcee01d28a4.png)

有基于强化学习的全网路径优化

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

有基于决策树模型的网路故障分段检测，和NLP的交互

![图片](assets/57771c2b5fb2.png)

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

各方面反馈很好，但是落地的时候我们发现设备端的AI处理能力太弱了，同时对多模型任务推理的性能也不太好，所以被迫在模型上做一些修改，同时还等待新的芯片到位。

![图片](assets/53b63ab868ed.png)

nVidia Morpheus框架发布出来以后，他们也找过我，但是有一些竞争对手(主要是IB Switch vs 思科的交换机)和成本的关系，没有接这个活

>[**nVidia Morpheus：浅谈AI在网络中的应用**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485688&idx=1&sn=ccdfaa21419557f0bb16aaed399c505d&chksm=f996183acee1912c14a596247590cc1b4f72ece3ddc401859cf64274002e1ddd9ffc58e68a57&scene=21#wechat_redirect)<

而且对于AI如何应用到网络安全，特别是数百万每秒的数据包处理时还要做推理，是否需要复杂的模型，同时GPU本身利用大量的延迟隐藏和并行调度机制对于数据包转发的影响也非常大，当然在PaloAlto那里也有一些落地的场景，过分关注这一点推DOCA就有一点... 话题扯远了...有空单独说吧

### 结论

Hopper被禁运了，当然对于我们而言更要去看它的内部的取舍，它的集群的设计，对于单个指标的极限追求可能需要停一停了，你会看到Hopper内部也逐渐开始在堆料的路上走到头了，开始在各个细节的地方去优化了，而这一个系列的文章也到了尾声。后续或许会写一个单独的总结。

而在我自己专长的领域，今年HC34 NvSwitch的发布，似乎有找到几分和NetDAM知音的感觉，特别的喜欢. 对于nVidia敬佩的是它一路走过来的危机意识和不断地颠覆自己的决心，也看到了大量的聪明的工程师的努力，真是一家值得敬佩的公司。