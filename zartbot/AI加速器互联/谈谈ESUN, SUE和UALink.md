# 谈谈ESUN, SUE和UALink

> 作者: zartbot  
> 日期: 2025年10月18日 00:13  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496512&idx=2&sn=0c10cef05fb1cc4e175f326d62b266e3&chksm=f995e382cee26a949fa1bf12144854e1e50c0cff5e8c579564e923c6d52fa8fe985deda44d46#rd

---

### TL;DR

其实不想写这个话题的, 因为前面很多年的文章累积起来已经把道理讲清楚了, 但是公司内外很多人都来问, 还是写一下吧. 主要原因是发现很多人的解读是有问题的, 特别是把ESUN和SUE混为一谈, 然后来分析和UALink之间的关系. 其实ESUN和SUE是两个不同的Group, 数据链路层和传输层的关系. ESUN达成共识并不代表SUE也达成共识了, 而我更希望看到的是传输层百花齐放.

从2020年我在Cisco做NetDAM以Memory Interface(memif)和以太网传输开始, 已经过去5年了, 实质就是通过以太网承载内存语义. 个人而言, 我是坚定的选择基于Ethernet ScaleUP的, 当然再加一个约束, 这里的Ethernet指的物理层和链路层. ESUN的出现也证明了这条路的正确性. 而且过去几年一直在和BRCM进行深度的沟通和合作.

![图片](assets/2e4df8c46a86.png)

### 1. 什么是ESUN

ESUN是一个数据链路层的协议, 使用了以太网的物理层, 数据链路层并采用了以太网头. 在这一点上让众多厂家达成了共识, 但是很奇怪里面为什么没有Intel?

![图片](assets/5fa11f032c4e.png)

然后, 我们需要定义什么是以太网头. 比较准确的定义是, 在不修改前导帧长度, 不修改以太网头报文长度, 并且EtherType字段的Offset和长度不修改的情况下, 通过扩展EthType, 支持更灵活的编码以适应ScaleUP网络需求的Header. 例如SUE定义的AFH

![图片](assets/3a871951df85.png)

这样有个好处就是针对原有的以太网交换机的PortLogic只需要针对不同的EthType执行不同的Lookup即可, 这样交换机既可以作为普通的以太网交换机使用, 又可以支持ESUN这类的针对ScaleUP网络. 这是使用以太网ScaleUP的根本逻辑:

使用以太网ScaleUP的根本逻辑和取舍
利用以太网大量供货来摊销专用硬件(以太网芯片/连接器/光模块/线缆等)的成本, 但同时接受以太网Header过长导致的一些不足.

如果使用以太网的PHY, 但是不使用标准的以太网头, 做双模或者专有的交换机实际上就成了专用硬件了, 这个成本逻辑就不成立了. 所以我对UALink的建议也是最好采用基于以太网头的DataLink层, 但保留其TL FLIT的设计.

当ESUN在数据链路层达成一致时, 剩下来就需要讨论传输层如何实现了

### 2. ScaleUP需要什么样的传输层协议

实质的问题是, 我们需要如下这个决策流程:

![图片](assets/0f94f1330435.png)

核心的问题是第一个决策点, ScaleUP的规模需要多大?

#### 2.1 ScaleUP规模分析
从并行策略来分析
首先从并行策略来看, PP并行和DP并行实际上当前已经可以被很好的Overlap了. 对于TP/SP/CP都是对张量进行切分后的分块运算, 对于整个分块的计算规模和XPU的算力可以得到一个计算延迟. 另一方面例如MHA按头的数量切分, TP的规模本身就会限制在64/128. 当矩阵被切的太小后, 又会对XPU算力带来影响, 导致小矩阵无法打满, 这些都是约束ScaleUP节点规模的因素.

另一方面是物理约束带来的, 大规模的组网也必然会带来的静态延迟, 例如一米光纤的传输延迟差不多4~5ns, 通常需要2层组网的大规模ScaleUP整体光路径延迟接近500ns, 另一方面由于需要穿越至少3跳交换机, 以及考虑到这样的系统需要更复杂的可靠传输协议, 整体延迟接近3~5us. 因此在这类并行中的集合通信开销也有非常显著的影响.

当然国内有一种说法, 由于国产芯片算力比较弱所以需要更大的TP域. 但是这种说法似乎也不太成立, 多Die的封装去ScaleUP单颗的算力 vs 单颗很弱的芯片通过大规模ScaleUP组网, 一定会选择前者. 而且现在国产XPU来看, 单颗性能已经可以做到和H100差不多的算力了, 因此这个观点也是不成立的.

剩下来就是需要讨论EP并行, 即某些观点宣称一卡一专家的部署方式. 首先我们需要明白EP的收益和边际收益. 对于单个卡而言, 在并发处理 请求Decode的过程中, 会产生  个不同的Token, 每个Token按照Top-K, 从  个专家中选择  个, 则一次剩下的未被访问的专家的概率为 , 累积  个Token, 未被取出的概率为 , 则需要访问内存的专家数为

假设一个极限的稀疏的情况, 专家数目为M=512, K=8, batchsize N=32时, 需要访问的专家数目为202个. 当专家数扩展到1024个更极端的情况下时, 需要访问的专家数为227个. 这样就带来了大量的内存访问瓶颈. 通过EP并行, 例如每卡8个专家, 访问专家参数的带宽已经下降到原来的3%. 此时EP的规模为128卡. 这就是EP收益的来源.

那么对于EP并行的边际收益来看, 如果超过交换机的Radix需要两层组网, 本身就带来更高的延迟, 累计将会影响到TPS. 同时再对比Token的累计访问内存带宽和参数的访问内存带宽, 可以得出并不需要一卡一专家的部署.

更进一步从部署的规模而言, 由于大模型在Serving过程中随着时间的变化有明显的峰谷效应, 集群应该根据请求有更好的弹性扩容能力. 单次扩容1024卡和单次扩容64/128卡相比, 整体的成本肯定是更细粒度的扩容更加经济并具有弹性.

结论: 从并行策略来分析, 满足交换机单层组网的Radix规模, 例如最大512卡即可.实际部署时可能考虑弹性缩扩容的需求, 还会进一步降低ScaleUP的规模. 您可以看到即便是Rubin Ultra的NVL576, Kyber机柜的背面可以看到, 单个ScaleUP域也只有144卡的规模. 并且NV还是在选择铜互连.

从系统可靠性分析
另一个决策点是ScaleUP的物理互连距离, 如果大于5m可能就需要光互连了. 光互连现阶段还是存在一些稳定性的隐患, 即便是单层组网, 也需要在可靠传输上做更多的工作. 当然这一点上并不是说铜比光好, 只是现阶段而言, 光传输的平均无故障时间(Mean Time Between Failures,MTBF)还没有达到和铜相同的数量级. 但是我们还是要对OIO/CPO/NPO/MicroLED等技术的演进持更加开放的心态, 如果这些问题解决了, 那么光互连也势在必行.

另一方面是从集群的视角, 对于ScaleUP域, 节点数量增加也会显著降低整个集群的MTBF. ScaleUP节点扩大一倍, 则平均无故障时间缩短一半. 因此我们也需要控制ScaleUP的规模, 或者通过可靠传输的设计来规避一些故障, 增加MTBF, 或者采用备份节点的方式.

然后就是从成本的视角考虑, 当某个XPU的互连出现故障后, 如果有相应的备份节点(例如UB-Mesh 64+1), 也可以显著的降低故障的影响. 但是同时也带来了资源的消耗, 有一张卡长期处理冷备份的状态, 整体解决方案的成本也会显著提高. 当ScaleUP集群故障时, 整个爆炸域的影响也是巨大的. ScaleUp规模越大, 受损的经济损失需要按照MTTR * ScaleUP节点数计算.

结论: 从系统可靠性的角度来看, 我们依然需要约束ScaleUP的规模.

另一方面, 我们也可以看看BRCM以太网交换机BU的GM Ram Velega先生的反馈, 他也认为构建单层的交换网络即可

![图片](assets/af1cdc2ba819.png)

#### 2.2 从延迟的视角分析

如果使用多层交换机组网构成一个超大规模的ScaleUP, 一方面是传输的延迟变得更加显著, 另一方面由于稳定性影响导致需要更复杂的可靠传输协议来解决问题, 例如拥塞控制, 多路径负载均衡, 基于Lossy的丢包重传等. 这些都会导致延迟显著增加.

例如以两层组网的ScaleUP为例, 传输延迟为3~5us 单个SM内的SMEM容量为256KB, 按照Ping-pong buffer来看, 以及矩阵乘法中其它矩阵的占用空间, 实际上用于传输的大概只有40KB, 则单个SM的峰值传输带宽为10GB/s. 虽然我们可以整体来看简单的做一个乘法(25GB/s * num_SM) 得到峰值的ScaleUP的峰值允许带宽, 但并不是所有的SM都在同一时刻并行的发出流量. 因此延迟将极大的约束整体的峰值ScaleUP带宽.

当然网党有一个延迟不重要的说法, 即基于内存语义先存到本地HBM, 然后消息语义再拷贝到远端. 这样就可以利用本地的SMEM->GMEM低延迟路径, 降低对SMEM的BDP约束. 但这种做法也存在明显的缺点, 首先存到HBM中也需要接近1us的延迟. 然后再写描述符通知ScaleUP I/O发送, ScaleUP I/O读取并转发累计也需要2us的延迟, 再加上可靠传输协议1us的开销和网络上3us的开销, 并且由于先存放到Local HBM,再从Local HBM读取进一步带来了HBM带宽的占用,影响了算力以及需要考虑Memory带宽拥塞的时候带来的延迟增加.

结论: 在ScaleUP中需要根据SMEM的BDP去进一步降低延迟, 提升峰值传输带宽.

#### 2.3 从Die面积分析 SUE vs UALink

SUE和UALink在传输协议上, 特别是在如何Packing Packet上存在差异. 主要原因是以太网头的开销还是相对较大的, 因此需要打包多个数据报文一起传输来摊销Header的开销.
2.3.1 SUE
SUE的Packing策略其实是我在2024年初给BRCM提供的, 当时的一个前提是利用通用的Shared Buffer Switch不做修改, 本身参考了Nvidia在2023年的一篇论文《FinePack: Transparently Improving the Efficiency of Fine-Grained Transfers in Multi-GPU Systems》论文提出的方法. Pack多个LD/ST到一个packet里面

![图片](assets/e7285de0f96b.png)

然后针对不同目的GPU的数据包构建独立的队列, 然后调度打包传输. 这就是BRCM SUE的Packing方案.

![图片](assets/47fc76bfa9ad.png)

但在一个带宽极大的ScaleUP上来看, 通常也需要很大的SRAM空间构建Buffer队列. 初步从国产GPU的几个厂家那里得到的信息是SUT的Die面积占用接近200mm^2. 而UALink大概只占用50~60mm^2, 对于单个XPU Die通常受限于光罩面积, 大约面积在800mm^2. 这样就会导致算力的损失接近20%.

因此BRCM也提出了SUE-Lite的方案, 将传输层交给XPU厂家自己实现.
2.3.2 UALink
UALink和SUE的区别是在于Packing机制上. UALink 的TL Flit不区分不同的目的XPU, 而是只要打满640B就可以封装成DL发出. 这样整个传输的逻辑变得很简单, 因此Die面积的占用也小了很多. 缺点是需要交换机做更多的处理. 相当于是将在GPU上的Die面积占用转移到了交换机芯片上.

在网党一直有一个“Smart Edge, Dumb Core”的原则, 这样的转移似乎某种意义上影响了交换机的吞吐, 传统交换机只根据Ethernet Header一次查表就可以完成转发, 但针对UALink则需要对Payload进行更多的解析, 因此会影响到整体的转发效率, 使得ScaleOut交换机的规模做不大.

![图片](assets/dbc2ab1afad5.png)
2.3.3 交换机架构
然而这里有一个细节, BRCM的Tomhawk Ultra证明了51.2Tbps 64B linerate是可以达到的.

![图片](assets/0221c46b68c7.png)

整体延迟也有确定性的250ns

![图片](assets/159879bc6e07.png)

Tomhawk Ultra和原来的Tomhawk实质的差异是, 前者采用了PortBased Buffer,并使用cut-through的机制进行快速转发. 而后者是Shared Buffer需要store-forward.
Shared Buffer Switch
通常它会将收到的数据包存入一个集中式的缓冲区, 然后取出Packet Header,解析并查表然后通知出接口转发. 如果需要对Payload进行解析和处理, 对于这块Buffer而言, 需要更高的操作速率, 因为它是一个多端口的SRAM构成的MMU, 多端口的SRAM带来的开销是非常大的, 同时解析和多次查表也需要更高的访问速率. 因此工业界的认知是针对这类交换机处理Payload会对整块芯片的吞吐带来极大的影响.
PortBased Buffer Switch
Tomhawk Ultra为了满足64B小包LineRate转发, 采用了PortBased Buffer, 这样避免了大规模多端口SRAM集中式buffer的占用和带宽/操作速率的开销, 并且通过Cut-Through的方式进行转发降低了延迟. 它的芯片如下图所示,

![图片](assets/db17974f6214.png)

它是一个扁长的Die结构, 实际上它的结构整体有32个独立的PortLogic构成,每个PortLogic上有独立的buffer, 用于缓冲和链路层重传(LLR).

![图片](assets/660bb4a3db5c.png)

实际上对于一个UALink的DL FLIT, 通过CutThrough的方式, 一边按照64B TL FLIT逐个处理Payload中的地址lookup, 然后dispatch到Egress的PortLogic, 而Egress PortLogic累计收到640B后就可以打包成一个DL FLIT发送给GPU.

实际上交换芯片来看, 增加这些处理的代价并不大, 并且PortLogic上的Buffer还可以复用, 整体要支持UALink那样的解析TL FLIT转发对于芯片面积增加是不显著的, 同时带宽演进上也是有保证的.

### 3. 结论

对于可靠传输如何构建, 其实以前一篇文章

[《谈谈RDMA和ScaleUP的可靠传输》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495506&idx=1&sn=385c2b750379214ea1deefaf7587837b&scene=21#wechat_redirect)

已经讲的很清楚了. 通过如上多个维度分析, ScaleUP设计上采用一层网络(这也是BRCM GM Ram V的观点), 并且简化可靠传输的复杂性降低对XPU DieSize的侵占. 并且采用Local Bus(Lossless, Credit Based, GBN)的方式而不是Network的方式(Full functional Transport, eg, TCP) 是更加恰当的.

当然针对更大带宽的组网, 例如我们谈到消息语义由于一些控制处理带来的Overhead是否可以在ScaleOut网络上启用内存语义来避免? 或者更直接的一个问题, ScaleUP和ScaleOut到底在定义什么? 具有内存语义的ScaleOut并带有完善的可靠传输机制算不算ScaleUP? 这也是我一直在提的Memif, 也是DeepSeek在论文中多次提到的, 需要统一ScaleUP和ScaleOut语义的根本原因.

![图片](assets/f65bac93fd29.png)

当下的很多争议, 其实来自于对ScaleUp和ScaleOut定义是什么都是相对模糊的, 只是因为历史上NVLink vs RDMA而产生的一个比较模糊的区分. 另一方面简单的把ESUN达成的一致和SUE vs UALink混淆在一次也犯了一个逻辑上的错误. 正确的描述应该是ESUN是一个数据链路层标准, 而传输层标准可以选择SUE也可以选择UAL,或者其它传输层协议.

ESUN在数据链路层能达成一致也是一个令人激动的消息, 还记得去年和BRCM CSG(以太网交换机Group)一起吃饭的时候, 他们的CTO Mohan和Tomhawk Ultra的架构师Surendra一起还在开玩笑: “凡事和以太网对着干的, 基本上都死了”. Ethernet win, 但是不代表UALink failed. 我更期望看到的是在统一的数据链路层上有多种传输协议的支持, 针对不同的XPU有不同的设计方案也挺好的.

正如我以前的一个观点, UALink稍微改改DL FLIT的header变为以太网兼容的格式, 同时维持TL Flit的方式Packing和转发, 可能在工业界来说生态和架构都是比较好的一个选择. 而不是在这里谈SUE vs UALink谁赢了的问题.