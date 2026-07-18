# [Sigcomm论文解析] Llama 3训练RoCE网络

> 作者: zartbot  
> 日期: 2024年8月5日 11:05  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491483&idx=1&sn=13ad0defd3c71bb733de86a5b3246107&chksm=f9960f59cee1864fb5a6bcc16d5e05d842bca9f7dd7ad41c5dd35eb82102f581d41a0fb34dfc#rd

---

这周悉尼正在开的Sigcomm上, Meta有一篇论文《RDMA over Ethernet for Distributed AI Training at Meta Scale》[1]详细介绍了它的物理网络部署和相关的拥塞控制机制. 本文来对它进行一些详细的解读.

### TL;DR

#### 0.1 先谈谈学术界和工业界的差距

Sigcomm虽然也号称顶会了, 但是`工业界和学术界的差距是非常大`的, 即便是工业界用网络的客户和设备网络研发的技术差距也是非常巨大的, 例如长期针对网络协议和网络芯片架构进行协同设计的架构师和网络运维相关的工程师的差距, 例如渣这种在思科搞过最核心的转发芯片相关工作的来看, 真正顶级的工作拿来发论文? 你搞笑么? 有些东西连专利都不会写的.

举个不太涉密的例子吧, Google今年Sigcomm的《A Decentralized SDN Architecture for the WAN》[2]. 实际上这些工作还远没达到我2018年做的Nimble Network和后面Ruta Disaggregation Routing System的水平, 周末有空我来对这个事情做个详细解读.

所以很多时候看这些顶会的论文也要辩证的看, 毕竟Meta这群人也就是有什么用什么的能力, 真要去做芯片和协议的协同设计的能力还差的太远, 开头做个总结吧

#### 0.2 渣的评价

`好的方面`:
这些针对可靠性/宕机迁移/降低爆炸域的设计, 针对用户可编排性的对称拓扑设计都是值得我们学习的

恢复到标准的Spine-Leaf对称拓扑, 接入TOR交换机到网卡这一段就不需要采用光模块直接采用铜的DAC, 省了不少成本, 例如单个400G光模块大概$1000美金. 同时铜缆也提高了稳定性.

爆炸域缩小, 如果某个GPU故障, 或者TOR交换机故障, 仅需要很快的迁移到另一个机柜上就行了, 爆炸域16卡, 单个Pod有3072卡, 而Llama 3 405B训练规模为16384卡, 从并行策略看正好每个Pod 2048卡, 因此每个Pod其实可以分为2048+ 1024, 剩下的1024卡还可以跑一些别的业务, 例如论文中写的一些256卡左右的推荐系统任务.  当故障发生时, 很快的杀掉小的任务来替换保证2048卡的训练即可.

而多轨道的部署, 如果TOR故障或者某个GPU服务器故障将会影响到千卡规模.

多轨道的非对等连接, 有些大厂采用在TOR上做DP, 更上层交换机做PP的方式. 在CP并行引入后,事实上这样的拓扑会带来更多的问题. 同时MoE的Bisection带宽也无法保证.

Spine采用了框式交换机, 更深的buffer可以吸收集合通信的burst, 同时基于VOQ的Linecard+Fabric方式扩大了交换机的Radix, 可以支持最大576个400G端口互联. 而交换机模块化的设计, 留了64个口用于端口故障时的更换, Fabric也有冗余.

Spine整个交换机故障也无所谓,整个集群的的Spine交换机按照1:1.125荣誉, down两台都没关系.

集合通信可以在有收敛比的网络中很好的运行.

通过在NCCL上实现了一个类似于Window Based拥塞控制机制.

`待完善的`:

![图片](assets/3dfc751b06af.png)

多路径的问题转移到了框式交换机上, 网络成本上升了30%

静态延迟高达22us, alltoall的通信问题实际上是没有解决的, 特别是小消息size的情况下还需要Rail-Based AlltoAll来提升消息size

我们还是需要更仔细的从根源上去解决RoCE的多路径的问题, 但是Meta当前是没有解决的.

单口400G,单上连带来的可靠性问题, Meta需要考虑双上联提高稳定性的情况下, 增加两个Hash冲突点如何避免冲突

不需要DCQCN本质上是在转移矛盾, 把本来该在网卡Buffer上解决的问题转移到CTSW交换机上.

整个工作充满了各种Hack和各种琐碎的配置, 并不能很好的做到开箱即用.

FlowLet相关的工作还在探索.

故障收敛速度的问题没有提及, 只是在RTSW到CTSW上增加了12.5%的链路做冗余.

本质上对meta这篇论文的评价是负面的, 因为他们还没有触及核心. 所以还是继续提出一个已经解决的问题, 给所有的厂商一个很简单的考题:

采用Spine-Leaf拓扑, 不用任何框式交换机, 不需要DeepBuffer. 如何不利用交换机任何Hash函数信息, 不需要交换机任何特殊配置, 不启用ECN和PFC. 通过网卡算法自动打散流量,并维持交换网97.5%以上的利用率, 对于交换机的buffer需求为队列深度低于3us. 并能够针对128:1的时候incast时最大流和最小流量之间的带宽差异小于100Kbps, 同时针对任何网络线缆故障, 通信中断无感知, 模型训练收敛时间小于100ms.

以下是对Meta这篇Sigcomm论文的详细解读.

## 1. Introduction

Meta自己解读论文的视频[3]如下:

      
     
       
         
           
             
                                

                 
                   
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
   
 

就像英伟达的首席科学家Bill Dally讲的其实网络就是`拓扑`,`路由`,`流控`这三块东西, 从Meta论文的Abstration来看, 对于这个RoCE训练网络能够调整的大概就几点:

1.`网络拓扑`: 例如是否需要Multi-Rail多轨道的优化等, 这一篇论文Meta放弃了多轨道部署,而选择使用了Spine-Leaf的全对称部署模式

2.`路由`: 针对训练网络的负载不均衡和突发进行的一些优化, 例如Ehanced-ECMP

3.`传输和流控`: 主要是抛弃了DCQCN, 所以不要说我一直怼DCQCN和PFC, 这是工业界的趋势,至于PFC的问题Meta其实是靠CTSW上的Jericho offchip HBM buffer规避的, 后面会详细介绍.

当然论文后面还有一些Meta的运维经验的介绍, 针对RDMA还是多看看下面这几篇仔细想想, 有句话叫能力决定认知的天花板.

[《RDMA这十年的反思1：从协议演进的视角》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489240&idx=1&sn=53c7512d8551a44834bd405fd38b15dd&chksm=f996061acee18f0c26fb6d3f745dfa717a1f9b41a5f63de139e72acbc00968f4a197a16dd272&scene=21#wechat_redirect)

[《RDMA这十年的反思2：从应用和芯片架构的视角》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489285&idx=1&sn=3a53f4d177aca0a2a052450fd1a58fe2&chksm=f99607c7cee18ed14e672132ba249a61c8d968d4eeec4a5d086a3529daffcd41097704d7e7e6&scene=21#wechat_redirect)

### 1.1 RDMA Verbs生态的重要性

Meta论文第一章Introduction对workload进行了分析. 主要讲述了在训练时FastSocket这类的没法做GPU Direct导致性能下降, 而EFA和IB以及专用的NVLink和Google ICI的专有性质限制了部署灵活性. 而Meta采用了RoCE保证生态兼容, 同时利用以太网的基础设施, 基于CLOS构建网络运维容易. 同时整个技术栈相对开发不会存在Vendor Lockin的局面.

**渣注**:

传输协议保证对RDMA RC Verbs语义的兼容非常重要.  针对RDMA over Ethernet, 虽然像AWS的SRD和UEC正在搞的Libfabric可以解决RDMA RC语义的一些缺陷, 实现Out-Of-Order Packet Delivery, 但是生态兼容上会出现问题需要软件协同更多的改造. 如何保证兼容RDMA RC语义, 同时实现多路径转发和拥塞控制能力才是工业界需要考虑的方向. 现在能够实现的只有两家, 其中Mellanox还因为RoCE协议的缺陷无法实现SEND/RECV语义的DDP, 很有趣的是本来Direct Data Placement 2002年就有的技术, Mellanox还非得自己造个名字叫Direct Packet Placement (DPP).

### 1.2 Meta的运维经验

紧接着Meta的作者谈了一些经验:

`需要专用的Backend网络`: 针对大规模训练,带宽的需求和通用计算网络是割裂的, 同时针对拓扑和路由还有很多特殊的需求. Job manager也需要进行拓扑感知的调度.

`路由机制`:早期实现基于集中式的SDN控制器的流量工程来实现负载均衡, 而如今在搞的一些ECMP增强来辅助.

`拥塞控制和集合通信`: 写了一大段主要是讲集合通信库上做Reciever-Driven的改造和抛弃DCQCN.

同时很有趣的一点, Meta在说以前RoCE相关的研究主要是在存储网络, 而如今在AI分布式训练上和以往有很多不同, 并且需要充分了解Workload并精心设计拓扑/路由/传输/运维流程.

**渣注**:

其实选择了Reciever-Driven的方式, 并且抛弃了DCQCN. 本质上是把拥塞控制的问题转移到了集合通信库上. 但是需要集合通信库和底层网络做更多的配合和他调整来协同优化.

实际上这样的做法并不干净. 更干净的做法是在网卡上直接实施基于Window Based的拥塞控制和SACK, 例如Falcon和xxx(你猜~ 你猜~). 当然英伟达在未来的CX系列也会走到这条路上来. 这是必然趋势.

## 2. Background

这一章在讲一些Workload相关的内容. Meta这套集群设计时不光是要跑生成式的大模型, 还有一些小规模的推荐系统Ranking相关的业务混跑

![图片](assets/b4067a9ac16f.png)

主要的集合通信源语是AlltoAll, AllReduce, AllGather和ReduceScatter

![图片](assets/463acdc3c5db.png)

AllReduce/AllGather/ReduceScatter由于在DP并行时可以比较容易的进行Overlap, 因此网络有一定的收敛比是可行的(例如Llama3训练时CTSW到ATSW的收敛比为1:7), 而针对AlltoAll, 例如LLM的MoE模型, 或者在几百卡的规模的推荐系统模型的Embedding交换, 都需要大量的AlltoAll通信能力. 而AlltoAll需要很大的Bisection带宽. 而针对集合通信的逻辑拓扑也有对拥塞控制和Hash冲突有影响, 例如Allreduce的Ring和Tree算法.

![图片](assets/fe6df7f6ff4a.png)

论文中的2.3 Training Workload分析只有小于128卡的一些任务, 因此Llama 3这些LLM训练任务并没有包含其中. 所以配的几个workload的图看看就好. 然后2.4 Challenges:

模型训练在快速演变: 单任务万卡, 带宽400Gbps更高这些已知的问题描述了一下

流量模式: 主要是低熵, 通过UDP报文五元组Hash打不散.

不同级别的网络拥塞: 主要是一些Fullmesh和层次化集合通信的流量产生的拥塞模式不同. 对于交换机buffer的需求:

![图片](assets/85880e456061.png)

协同调优的需求: 主要是NCCL和RoCE都无法做到开箱即用的提供最优性能, 开发环境和生产环境的差异等, 而低熵又带来了Hash冲突和负载不均衡的问题, AlltoAll也产生了大量的Micro Burst.

**渣注**:这一段分析中规中矩吧, 但是提出的挑战还是存在一些问题的, 毕竟Meta这种厂商也没法自己做芯片和设计协议的,所以需要协同调优.至于RoCE的Low Entropy大概他们的最大能力也就是去改改博通交换机的一个SDK接口, 把Dest QPN放入Hash, 还美其名曰Enhanced-ECMP Protocol? 我觉得这群人是不是对协议这个词有误解吧?

## 3. Hardware

训练节点硬件采用Grand Teton平台

![图片](assets/df0baa7f812b.png)

可以看到由CPU框/网络交换框和GPU框构成, 网卡为8块 400G的RDMA NIC

![图片](assets/b52c3f94724e.png)

网络上还是区分了前端和后端网络

![图片](assets/2262ceb2fe68.png)

对于一个训练Only的集群没啥问题, 但是对于一个训推一体的集群来看, 例如PD分离的一些推理业务来看, 以及更大规模模型的checkpoint读写, 多模态模型的训练, 还有推荐系统中的稀疏特征和PS的需求, 这样的分离网络会遇到问题么? 而另一方面我们看到AWS和Google都是Backend和FrontEnd的融合方案.

训练网络拓扑如下图所示:

![图片](assets/3a03cee36242.png)

每个机柜放置两台H100服务器, 每个服务器8个400Gbps单口的网卡, 然后两台服务器连接到同一个TOR Switch(RTSW)上,采用DAC线缆.它并没有使用2x200G得冗余保护机制, 因此在故障中断中网络的占比较高.

![图片](assets/b90d7f3e5e5e.png)

整个拓扑如上图所示, 一个机柜两台服务器16卡H100, ToR交换机为基于TH3 12.8T或者Cisco Silicon One 12.8T的交换机.下行16个400G端口, 上行16个400G端口连接Cluster交换机(CTSW), 每个Pod有192个机柜,按照1:1无收敛的方式连接到Cluster交换机.

Cluster交换机采用了Arista 7800的框式交换机架构, 线卡采用BRCM Jericho-3系列芯片, 片上Buffer更大,同时携带了HBM作为Off-chip buffer, 更容易吸收集合通信带来的突发流量, 如下图所示

![图片](assets/573368bd433c.png)

在一个Pod(AI Zone)内有16 * 192 = 3072个H100, 它们之间互联的通信距离为最大经过3跳交换机. 选择框式交换机还有一个好处是,它可以单个交换机提供更大规模Radix互联, 例如Meta的拓扑需要192+28个400G接口在单台交换机上. 总吞吐需求要86Tbps, 已经超过了最大的单颗交换机芯片51.2Tbps极限. 而基于Jericho 3+ Ramon的可以将单个交换机的密度通过多级交换+VOQ的方式进一步扩展到576个400G端口.

![图片](assets/48fd93505ac7.png)
即Jericho3将数据切分成信元, 然后均匀的分担到多颗Ramon Fabric卡上, 最后再汇聚到出方向的Jericho线卡上. 中间采用了Virtual Output Queue的调度算法.

![图片](assets/a037d68064b4.png)
依照当前Meta的部署, 选择288 x 400G 端口的平台即可, 这样还有一些可运维的好处. 端口实际上只使用了220个, 还有64个剩余. 这样对于Cluster交换端口坏或者线卡坏都可以很快的进行维修替换, 甚至是Fabric转发引擎故障也可以快速替换.

然后另一方面多路径Hash冲突的问题, 由于Cluster交换机内部可以做信元切分和VOQ转发, 并且Jericho的深buffer特性也可以很好的吸收集合通信带来的burst. 因此只需要在TOR交换机上考虑解决Hash冲突即可.

当然这样的方式也带来一个问题, 整体网络建设的成本会上升30%以上. 另一方面GPU上联仅有1个400G端口, 增加了故障中断的概率,但也避免了两个口带来的Hash冲突.

最后24K卡由 8个3072卡的Pod互联构成, 但是Pod之间的带宽有1:7的收敛比. 这会拖慢集合通信的性能, 但是Meta将DP并行通过FSDP overlap后, 隐藏了延迟, 因此在这里构建一些带收敛比的网络也未尝不可. Inter-Pod的交换机(ATSW) 采用Minipack 2, 基于BRCM Tomhawk 4 或者Tomhawk 5构建.

![图片](assets/f4aa61dab03d.png)

**渣注:**  其实早一些的集群建设, Meta都在采用多轨道的部署方式《Optimized Network Architectures for Large Language Model Training with Billions of Parameters》[4], 如下图所示

![图片](assets/777a864bb8bb.png)

通过相同Rank的GPU接到同一个Rail的交换机上, 基本上国内的几个公司的拓扑也是采用这种方式. 通过这样的方式可以避免一些Hash冲突带来的影响, 尽量将流量调度到Rail交换机上. 然后对于上到Spine交换机的流量,通过一定的编排和指定NCCL通信库选择源UDP端口的方式来错开,无论是Meta还是国内基本上采用这样的方式.

这样的拓扑可以构建千卡规模的完全无冲突的互联, 但是这样的拓扑也会带来一系列的问题. 一方面是任何一个卡有了故障, 必须要在同一个Rail交换机下找到一台备用机器. 而任何一台Rail交换机故障都会导致整个千卡集群规模的受损.并且为了满足布线的需求, GPU网卡到Rail交换机之间只能使用AOC光缆, 这样不但增加了成本,同时还增加了故障率. 非对等的拓扑结构也给并行策略编排和调度带来了极大的挑战.

`我一直是反对通过拓扑的方式去解决Hash冲突的问题`, 一定要用更加对等的拓扑来做, 并且通过网卡的算法去解决Hash冲突问题, 并且做到对交换机Hash函数不感知. 很高兴看到Meta在新的H100集群做出了改变, 回到了原有的Spine-Leaf的架构.

## 4. Routing

主要还是解决三个问题: 低熵, 突发, 大象流.

### 4.1 Path Pinning

Meta在4.1.1坦诚的表示了当前ECMP带来的Hash冲突和性能影响, 然后做了一个非常Hack的做法, 就是针对RTSW(ToR SW)进行切片, 将数据包路由到特定路径. 单个训练任务或者网络中没有故障的时候, 这玩意工作的挺好的. `However, this was seldom true.` 碎片化的作业分配带来的上行链路带宽不均匀以及hash碰撞, 故障受影响时的rebalance带来的问题等. 这玩意就完全不工作了, 性能下降达30%.

然后4.1.3里面还有一个扯淡的做法, Hash冲突解不了就扩大ToR的上行带宽为下行带宽的两倍, 还好诚实的提了一嘴这玩意很费钱, 但为啥不提他们家的CTSW很贵很花钱呢?Arista的框式交换机,Jericho带HBM多少钱来着?

### 4.2 Enhanced ECMP

一方面是通过 通过`NCCL_IB_QPS_PER_CONNECTION=16`来将流量打散到多个QP, 但是这样会导致集合通信的性能有所下降. 然后就是改交换机, 通过UDF能力把RoCE Header中的Dest QP也加进去算Hash. 然后贴了一个图

![图片](assets/c47f3557f4f2.png)

他们还算比较坦诚, QP多了会导致消息更小,性能下降. 然后需要做一些Trade-Off.

**渣注** : 当前我们能做到的QPs=1 or 2就能完全的跨越CTSW保证> 97.5%的利用率, 而且不需要什么加大RTSW带宽, 不需要CTSW搞什么DeepBuffer交换机, DCQCN和PFC什么都不需要, 也不需要交换机做什么PacketSpray. 还能做到完全RDMA Verbs RC兼容. 真想吐槽一句:** 学渣文具多.**

### 4.3  Centrailzed TE

集中式的流量工程

![图片](assets/d2153a8c04ee.png)

控制平面Collector做拓扑发现, 然后有一个内部的链路状态路由协议Open/R来构建拓扑. 然后还收集来自网络设备的Telemetry, 然后CSPF计算并更新交换机配置. 数据面采用AC精确匹配<source port, destination prefix>做Policy Based Routing.

正好这几天Google Sigcomm的论文在写Decentralized SDN,把TE放在设备上

![图片](assets/2e7d15130ac6.png)

还有渣6年前搞的Nimble Network

![图片](assets/5153b9f071c6.png)

Meta还是自己好好学习一下吧, 好在Meta还是比较诚恳的谈到了TE的问题. 一方面规模大了计算的复杂性和开销, 另一方面在故障发生的时候这玩意的可用性的问题.

另外论文有一个图可以看到, Meta Enhanced-ECMP 现在的AlltoAll大概只能做到带宽的80%左右的利用率

![图片](assets/e863036a6226.png)

**渣注**: 我们可以做到97.5%的利用率, 并且128卡打1卡的incast做到流量最大最小差控制在100Kbps内, 不需要任何额外的TE引擎, 不需要动任何交换机配置, 给个指标让Meta追吧.

### 4.5 Future Direction: Flowlet switching

Flowlet这段完全属于扯淡, 对于Out-Of-Order Packet这些的认知, 好好看看什么叫Out-of-Order Packet Delivery,In-Order Completion吧

![图片](assets/e6ed64cedfb8.png)

其实这段本质就在谈博通的DLB解决方案, FlowLet降低到单个Packet就是Packet Spray了.

### 4.6 Discussion

由于ECMP的效率低下，Hash冲突导致了链路利用率低，进而引发了性能的不一致性。为了缓解这个问题，采取了临时的网络过度建设措施, 大概就是说搞了一堆操作, 包括静态路由、动态路由和流量工程，问题还是解决不了, 还需要研究.

## 5. 传输层

这一章谈了一下像“AlltoAll”这样的集体通信模式仍可能造成瞬间缓存堆积和微突发, 所以流控和拥塞控制的需求很强.

### 5.1 实现

没有芯片开发能力,只能调整一些配置了, 然后就是运维快速识别并隔离不健康的网元和链路. 没啥特别的东西.

### 5.2 拥塞控制

Meta发现200Gbps有用的DCQCN+PFC在400Gbps下性能并不好, 400G是个分水岭呀, 几年前我就说过DCQCN搞不好的,可惜很多人没有远见. ECN水线低了吧可以避免PFC传播,但是又显著影响集合通信. 看看博通在UEC写的吧

![图片](assets/7103ffb2f9cd.png)

DCQCN hard to tune, 目标是要Configuration-Free的拥塞控制. 然后Meta的解决办法有两个

利用集合通信库控制inflight流量总量

通过接收端发送`Cleat-to-Send(CTS)`消息给Sender , 并且给CTS消息高优先级队列的方式来控制inflight.

![图片](assets/303e49744d42.png)

但是GPU通信Kernel的Channel数和buffersize都有影响, 实际上调整还是有难度的.也需要充分的针对不同的作业进行调整.  然后做了一个16:1的incast测试, 并且比较了Perftest和Gather集合通信, Perftest没有这样的流控机制, 而Gather在NCCL上做了CTS控制.

![图片](assets/3e2274083624.png)

**渣注:** 从本质来看, 为啥不在RoCE网卡上直接做Window Based Congestion Control呢? 这还是Mellanox网卡做的不行, RoCE协议瞎搞带来的后果, 而我们不管是perftest还是其他的,在128:1 incast下都没它遇到的那些问题...

### 5.3 discussion

拥塞控制一直是RDMA网络研究的重点, DCQCN在以存储为中心的网络中被视为“金标准”。然而，Meta在分布式AI训练工作负载方面的经验提供了对定制拥塞控制算法的不同视角。尽管关闭了DCQCN，并且有多个RTSW向具有大缓冲区的CTSW发送PFC，但并未遇到生产环境下的AI训练流量导致CTSW持续向RTSW发送PFC的情况。然后就准备开始评估是否有可能完全不使用传输级别的拥塞控制来进行操作。

**渣注** 画个图来说, 你把中间交换机多加了一堆Buffer也好意思说不需要做拥塞控制?

![图片](assets/027f9bde7df1.png)

实际上每个网卡的片上buffer配合一个超简单的window based拥塞控制就能搞定的事情.

## 6. EXPERIENCES

### 6.1 通信库和网络共同调整

作者画了一个图, 看上去好像做的很多蛮出色的工作把性能提上去了

![图片](assets/32f4103403d9.png)

但是问题就是中间的CTSW采用VOQ和DeepBuffer的架构,  空载延迟就22us, 这样对于很多小消息size的集合通信性能影响就特别明显了, 所以只能降低NCCL Channel数来增大消息大小. 而对小消息只能通过不同的协议和LL128/LL这样的方式或者Tree-Based集合通信算法去补..然后还提了一嘴用PXN(Rail-based AlltoAll). 然后又继续折腾CTSW的信用分配机制,还改了网卡PCIe的credits和relax ordering, 拓扑感知的Rank分配等. 处处充满hack,到处打补丁.

**渣注** 我就想问一句,Mellanox能不能好好做一个RoCE网卡开箱即用,交换机毛都不用配,啥DCQCN/PFC都不开, 也不需要DeepBuffer,随便找点TD4/TH4,也不用管什么flowlet或者packetSpray, 就开箱即用, 让人家开发NCCL通信库的人专心做自己的事情?

### 6.2 Impact of Routing and Topology

![图片](assets/41da35fc4fe8.png)

第一阶段RTSW-CTSW采用1:1收敛比, 然后抖得一塌糊涂. 第二阶段采用1:2的收敛比, RTSW上行到CTSW带宽是下行到GPU带宽的2倍, 抖动🤭. 然后第三阶段开了流量工程, 最后在第四阶段把收敛比降低到1:1.125, 即RTSW下行16个400G接两个GPU主机, 上行18个400G接CTSW.用来应对链路故障.

也就是说实际的Fabric带宽利用率80%左右, 还要NCCL各种魔改.

### 6.3 可观测工具

主要是一些计数器采集, 例如NIC感知的乱序包, 交换机/超时/buffer的计数器, PFC Watchdog这些工作, 然后还有一些ping的连通性检查等. 然后一些线上故障, 例如CTSW交换机软件故障带来的性能退化. 以及一些配置迁移导致的SRAM buffer太小出现的问题等.

参考资料

[1]
RDMA over Ethernet for Distributed AI Training at Meta Scale: https://dl.acm.org/doi/10.1145/3651890.3672233
[2]
A Decentralized SDN Architecture for the WAN: https://research.google/pubs/a-decentralized-sdn-architecture-for-the-wan/
[3]
Video for RDMA over Ethernet for Distributed AI Training at Meta Scale: https://www.youtube.com/watch?v=wLW3UzUw5rY
[4]
Optimized Network Architectures for Large Language Model Training with Billions of Parameters: https://arxiv.org/pdf/2307.12169v2