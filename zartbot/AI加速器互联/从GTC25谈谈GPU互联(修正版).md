# 从GTC25谈谈GPU互联(修正版)

> 作者: zartbot  
> 日期: 2025年3月31日 06:05  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493730&idx=1&sn=689efb4b7eb0c13dcb27fdd75b341f40&chksm=f995f8a0cee271b6d193f707e5f029dfdb9ef1738a89d261e091cc4cdc45ef095e2236c007c1#rd

---

注: 昨天的版本被Jensen计数法搞混了,算错了芯片的数量, 实际上每个机框有18个ComputeTray,每个Tray有两颗Rubin Ultra的芯片, 然后整个机柜有18x4=72个ComputeTray,累计144颗Rubin Ultra芯片, 互联的NVSwtich的分析上做一些修正.

### TL;DR

GTC25上有一些Rubin互联相关的信息, 特别是老黄讲的什么“先要ScaleUP然后再ScaleOut”的观点,个人是不认同的, 这几天又看到华为UB-Mesh的论文, 再有一篇港科大的FuseLink的论文, 基本上和我一直以来认为ScaleUP和ScaleOut需要融合的观点是一致的, 因此做了一些分析.

## 1. Rubin NVL576

### 1.1 Rubin机柜结构

整个Rubin的机柜结构被称为Kyber, 由单个的计算柜配置一个Kyber SideCar的机柜构成

![图片](assets/f2052999f925.jpg)

计算柜内由4个NVL144的计算机框构成, 每个计算框包含18个ComputeTray.

![图片](assets/8a9aac9c64f2.jpg)

每个计算机框内部采用中背板(Mid-Plane)结构,计算板和交换板正交布局

![图片](assets/b4356af497f6.jpg)

MidPlane放大看连接器如下图, 我们从中背板结构可以看到单个Rubin Ultra 接插件有72个插头, 单个插头2个pin,如下所示, 累计带宽为3600GB/s, 那么每个pin就要50GB/s, 也就是说在NVLink Gen6/Gen7上会有448G Serdes

![图片](assets/99de3f64f75e.jpg)

其实这样的结构对于连接器的稳定性和插损的要求会非常高, 因此可以看到无论是计算板还是交换板都有一个巨大的Locker把手,当然个人不太喜欢这样的中背板结构, 直接去掉正交连接到交换板不行么? 答案是不行的, 因为和NVSwitch芯片的Radix约束相关.

### 1.2 Rubin计算板

单个ComputeTray的结构来看更加紧凑, 各个模块如下图所示

![图片](assets/12baafce3cc4.jpg)

侧面图如下, 前面板上部为BF3, 中间为连接ScaleOut网络的RDMA光模块接口, 下面为4个NVMe盘

![图片](assets/74cda019b08a.jpg)

在整个系统内显示有576个CX9 NIC, 但很明显在ComputeTray上是无法放下8颗CX9的芯片的

![图片](assets/3cca77c0d9b6.png)

然后互联带宽来看,累计ScaleOut带宽为115.2TB/s, 按照Jensen计数法应该是双向带宽, 那么折合出来单个CX9 NIC的带宽为800Gbps, 因此推测出CX9其实就是很简单的双Die胶水封一块构成的1.6Tbps

![图片](assets/45d29a3cfff3.png)

### 1.3 SwitchTray

按照老黄的路标图显示,交换机下方的带宽数值应该为NVLink的速率, 而不是交换芯片的容量. 另外预计NVSwitch Gen6和Gen7的区别是224G Serdes和448G Serdes的区别, 因为Rubin还需要兼容原来的Blackwell的NVL72机架结构.

![图片](assets/9f33aabc6ae5.png)

如果按照老黄在GB-NVL72的计算方法, 整个系统标注的Switch数量为18个, 带宽计算130TB/s也是按照双向计算的, 实际交换芯片容量为28.8Tbps, Radix=72 NVLink Port.

![图片](assets/53e9fe96ece5.png)

一个很简单的Switch芯片路径的推算, 在Rubin这一代, 继续使用224G Serdes ,容量扩容到57.6T, 端口数增加到144.  然后Rubin Ultra这一代, 可能存在两个变数, 一个是容量维持不变, 但是Serdes换成448G Serdes 

Rubin Ultra整个系统显示有144个NVLinkSwitch芯片, 应该是指的交换芯片的数量,容量为1500PB/s, 但是前一页显示的是1.5PB/s,估计是一个Typo, 另一方面按照Rubin-NVL144为260TB/s, 理论上NVL576的带宽为1040TB/s, 实际容量还提高了50%, 以1.5PB/s折算,1500x1024/144x8/2,单颗交换芯片的容量为42.6Tbps, 具体交换芯片容量和端口数就有点疑问了.

![图片](assets/6e9259cb0955.png)

如果按照448G Serdes来看, 单颗NVSwitch芯片容量为57.6Tbps,累计提供144个port.整个42U机柜背面放置144颗Nvswitch交换芯片, 那么估计背部需要36个SwitchTray, 每个Tray内放置4块芯片.

![图片](assets/611ab3af0a82.jpg)

单个RubinUltra芯片应该是两边各有一个3.6TB/s的NVLINK接口

![图片](assets/9917d414ba23.png)

累计单个ComputeTray需要提供4个3.6TB/s的接口, 总共72个ComputeTray, 那么总吞吐也就1PB/s, 没有到1.5PB/s, 真心请教一下多的500TB/s是怎么来的? 唯一的猜测是Rubin Ultra中间的C2C也算了3.6T, 这样单个芯片为3.6TBx3= 10.8TB/s, 累计144个Rubin Ultra提供1.55PB/s的互联带宽咯?

然后NVSwitch芯片本身提供144个Port, 每个连接一个Rubin Ultra芯片, Radix也是满足的.

### 1.4 为什么要MidPlane

很简单的一个原因, 整个机柜最底下的一个NVL144的框和最顶上的框要互相通信, 而正交的交换板必定连不到那么远.

## 2. Huawei UB-Mesh

华为前几天发布了一篇论文《UB-Mesh: a Hierarchically Localized nD-FullMesh Datacenter Network Architecture》[1] 介绍了他们的ScaleUP+ScaleOut的统一互联

![图片](assets/56ccb6864046.png)

### 2.1 互联需求及设计原则

文章中提出了4个需求:

R1:大规模, 支持10万卡规模的互联

R2:高带宽需求

R3:性价比

R4:高可用性

然后基于这些定了3个设计原则:

P1: 流量模型驱动的网络拓扑设计

P2: 拓扑感知的计算和通信

P3: 容错和自愈能力

### 2.2 递归的Fullmesh

基于这些需求和原则设计了UB-Mesh系统.通过递归的Fullmesh构建

![图片](assets/c1438552b03e.png)

然后每张卡可以在多个维度灵活分配连接

![图片](assets/37a5e23f993a.png)

互联方式上, 在不同维度采用了不同的介质

![图片](assets/a5f3da8de252.png)

主要通过UBIO连接CPU和GPU,同时还提供了不同Radix的交换机芯片

![图片](assets/ad4687d8b54d.png)

相对于传统的两套网ScaleUP和ScaleOut部署, UB实现了统一的组网

![图片](assets/f67fc13f1643.png)

整个组网架构如下:

![图片](assets/20da9a2eeb17.png)

论文提供了一些实物的照片, 其中还设计了专用的64+1的冗余备份NPU.

![图片](assets/3bd341cb53c5.png)

然后为了应对流量的动态调度和冗余自愈能力, 构建了源路由的能力

![图片](assets/69d37bfc0f44.png)

然后针对拓扑结构, 构建了层次化的集合通信优化

![图片](assets/bc22776aded1.png)

## 3. FuseLink

这是来自于港科大的一篇论文, 但是还没有Publish出来, 主要想法就是ScaleOut的网络多网卡融合

![图片](assets/63475dd8473e.png)

本质上和我推测NV会把CX9/CX10加入到ScaleUP domain形成两网融合的想法是一致的, 其实在内部两年前我就在提ScaleUP和ScaleOut融合的事情.

![图片](assets/659046a0afe8.png)

可惜NV还是没意识到这个问题, 受制于PCIe Gen6的标准, 从Blackwell到Rubin Ultra整个ScaleOut的带宽是没有扩容的, 过分的去卷NVLink的带宽, 实际上是没有太大意义的.

## 4. 谈谈个人的观点

从个人的观点来看, 华为UB和港科大FuseLink统一ScaleOut+ScaleUP是完全正确的一条路, 但是还是希望能够实现GPU之间的对称拓扑, 华为的UBMesh或者是类似于Tenstorrent/TPU这样的xD-Mesh和XD-Torus总觉得算子编排上和资源调度上还存照很多难题.

同时个人希望能够基于标准的以太网来实现, 薅以太网交换芯片的羊毛. 前段时间和BRCM交换机芯片的CTO聊过, 似乎BRCM在以太网ScaleUP上还是存在不少问题的, 交换机本身没啥问题,更多的是在和NPU互联的I/O Die上还缺少很多经验.

其实对于ScaleUP和ScaleOut融合的需求, 从应用上来看, 计算未来会以Tile/Token为单位进行处理, 以64x64 Tile, FP4计算, 单个消息为2KB, 以FP8计算为4KB. 如果按照token调度, 以DSv3为例, hidden_dim=7168,那么消息大小为7KB. 因此无需过分的优化小size的传输. 但是, 从计算芯片侧来看, 内存语义是需要维持的, 当然这里面有些话涉密就不用多说了.

另一方面ScaleUP的规模来看, TP并行拆分后GEMM size变小带来的效率问题, 实际上加上通信本身的延迟, 我个人不觉得会出现TP=64这样的需求, EP并行这些本身也可以很好的通过micro batch overlap, 所以整个ScaleUP域的规模,可能256就够了.

另一方面从芯片的角度来看, 由于应用的消息size决定传输延迟, 再考虑一些队列延迟的情况, 无需过分追求超低延迟的ScaleUP互联. 至于INCA/SHARP这些技术, 说实话没有太大的收益, 并且对于EP并行, 交换机无法维持很大的Context, 交换机难以实现EP的dispatch / reduction offload, congestion control也是一个难题.

其实为了考虑成本, 做一个256卡的ScaleUP,在其之上连接一些NIC芯片再走一些带收敛比的ScaleOut可能是更好的一条路.

关于硬件互联的内容, 大概就这样吧, 另外最近还在考虑基于NSA/Titan一类的模型, KVCache如何处理的问题, 话说这两天看到一篇左鹏飞老师的论文《Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation》[2], 其实现阶段如何把xPyD做好就行了,可能更远期的方向是Attention和MoE的AE分离的问题.

然后关于MoE模型, 有一篇文章《Dense与MoE大模型架构后续发展解读》 挺有意义的, 说实话过分稀疏的MoE是有问题的, 这条路发展到一定的程度也会存在瓶颈, 特别是在推理系统的复杂度和平均的TPS上还存在很多约束, 未来怎么走,可能还需要一些别的方法.

最后一个准备做的工作是整个MaaS推理平台的一些最优调度的问题, 后面可能有空会基于ShallowSim来配合ILP做一些这类的调度算法的研究...

话说从上到下全栈的来看待这一系列问题, 讲真不是想去抢谁的活,或者要去卷谁. 我一直以来都是那种“不争”之人. 只是简单的把自己认为正确的和错误的观点讲出来, “水善利万物而不争，处众人之所恶，故几于道。”, 水因善下终归海，山不争高自成峰. 也希望工业界/学术界少一些带着屁股的争论和各种因为利益和营销而故意的曲解.

参考资料

[1] 
UB-Mesh: a Hierarchically Localized nD-FullMesh Datacenter Network Architecture: *https://arxiv.org/pdf/2503.20377v1*
[2] 
Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation: *https://arxiv.org/abs/2503.20552*