# NetDAM拥塞控制

> 作者: zartbot  
> 日期: 2021年10月17日 16:03  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486752&idx=1&sn=70fdeee15208755ddba5c0fff8daf5bc&chksm=f9961de2cee194f4b4f263e7fe96079d502c4afd85acb25100301d2cd4ce8e5f57b989f9f4e5#rd

---

拥塞控制是一个研究了数十年的大问题，例如<TCP/IP Illustrated vol.1>花了整整三章。但是有些奇怪的事情似乎把整个节奏带偏了，那就是RoCE的无损网络。本文先介绍(吐槽)一下无损网络，然后介绍一下HPCC和NDP的工作，紧接着大概说下Google Swift引出终端拥塞的问题，最后介绍NetDAM拥塞控制实现。

### 无损的由来

开玩笑讲一句话：银行都不刚兑了, 网络还在搞无损?其实很多时候应用层消息有幂等保护就不需要网络层可靠了，所以NetDAM的可靠传输是可选的，特别是在互联网API-Gateway的场景，复杂的事情要丢给终端：）

宏观来看，由于RoCE替代TCP在数据中心很多`高吞吐` `低延迟`的场景下用的非常棒，唯一的问题是要采用PFC**构建无损网络，因为go-back-N的策略对性能影响太大。而从根本上讲，go-back-N是因为PCIe总线上带来的，由于PCIe总线设计为一个短距离总线，通常所谓的`丢包`大概率是因为干扰带来的CRC错误而产生：

![图片](assets/153bde0384cc.png)

而当把RDMA加载到以太网上时， 无损网络的概念就出来了，紧接着各种拥塞控制算法出现，甚至还有AI调参的，强化学习调队列的，还通过更加激进的测量(Telemetry)并采用MIMD**的方式调整。

那么把DMA**隔离开，网卡上有内存直接写就不需要无损了呀，CXL的出现也使得访存容易了

### 丢包的必然

丢包本质上说是一种传输上的`损失`，而拥塞控制算法本质上是为了降低`损失率`,是不是联想到了金融风控？例如一个金融产品，当它能够刚性兑付(承诺保本)的时候，是不是同样也是无损的？但是这样无损的承诺给承载这个产品的机构带来了多大的风险？银行打破刚性兑付已经好多年了，也就是说早就没有`无损保本`的说法了。

而风险或者损失的本质就是`不确定性`, 所以我们必须要接受一个现实， 丢包是必然发生的，光模块的问题、硬件故障、路径中断，各种误码...一系列的问题

### 拥塞控制的核心

既然`丢包`损失可以看作是由`不确定性`引起的，不确定性主要是在延迟上。从概率的角度来看，一系列拥塞算法本质就是在大量的随机变量中估计出一个期望，例如早期的TCP采用滑动平均RTT，后期同时测量`均值`和`方差`，并用方差修正RTO。

构建`确定性网络`和`准确测量`成了对冲丢包风险的常见思路。
确定性网络
主要是以DetNet实现现IP网络从“尽力而为（best-effort）”到“准时、准确、快速”， 控制并降低端到端时延的技术。通常需要在分布式系统中考虑严格的时钟同步和控制，这样很有可能会降低整个网络的使用率，当然对于相对空载的网络是可行的，或者在一些工业控制、音视频环境下是必须的，但在超大规模数据中心必然会带来极大的成本。

倒是觉得阿里提出的`可预期网络`的文案上还是比较务实的(仅是文案上去掉了确定性，具体实现没看过不评价技术，早上我还开玩笑说，既然期望有了，方差、偏度呢？)。

另一种确定性实践是P4实现的`NDP`，也就是说真的丢包的时候不是把报文全部丢弃，而是Trim掉Payload把包头一些信息留下来然后由收端通告丢包，但个人直觉上总觉得不对，一个快递公司准备弄丢快件时，先把上面的快递单撕下来然后加急发给客户... 正如NDP自己说的，在完全对等的CLOS**拓扑中它基本可以接近最优，但是在其他非对称情况下就有问题了，所以这样的拥塞控制协议是完全硬件和拓扑依赖的,而且还要完全依赖智能网卡来配合。
准确测量
这一类算法的代表性作品有几种，一种是阿里提出的依赖于软硬件结构而实现的高精度测量拥塞控制`HPCC`，通过交换机和网卡逐包逐跳提供链路状态，然后供应用快速调整速率，当然这样的代价非常大，所以后面一年SIGCOMM又提出了基于概率的HPCC(PINT).

![图片](assets/51635225d4ab.png)

薛定谔的延迟 :事实上延迟或者拥塞(队列深度)测量,其观测的状态和收到观测值的状态是不一致的，而且主动测量也会影响其观测状态。而对测量值的计算处理又会带来额外的CPU消耗。而和交换机紧耦合的协同对于部署和维护都很麻烦。

这些工作引用Google在Swift论文的评价，总结的非常好：

*Protocols such as DCTCP,PFC, DCQCN and HPCC use explicit feedback from switches to keep network queues short and RPC completion times
low. They can provide good performance, but they do not help under large incasts and IOPS-intensive workloads. In particular, congestion build-up on hosts is a practical concern that is not addressed. Tight coordination with switches also complicates deployability and maintainability.*

所以很简单的一个观点就是，尽量不要对网络硬件有过多的依赖，而且针对主机的延迟和拥塞需要考虑。所以Google采用了在传输层内添加4Byte的方法将远端主机的情况和经过的跳数回报到本地

![图片](assets/db58e93add74.png)

然后针对Fabric拥塞和主机拥塞维护两个窗口，Target Delay的计算也非常有效，整个工作非常扎实。

其实很多年前，思科就做过网络延迟和主机延迟解耦的工作即Cisco AVC中的ART(Application Response Time)，通过网络设备记录时间戳来获得最终服务器的响应延迟：

![图片](assets/374b4ff4934f.jpg)

很简单的一个假设，就是服务器端和客户端在握手阶段回复SYN-ACK和ACK时是确定性的秒回，而针对应用的Data回复会有延迟，即主机响应时间，只不过是Google的SWIFT把这个时间直接带在了传输层包里。通过AVC技术，并借助AI进行分析，我们就可以很容易的观测出主机和网络的使用情况了，下图是我们在2018年的工作：

![图片](assets/d307464c1161.png)

当然这个工具主要的目的是主机团队要网络团队背锅时，帮助网络团队甩锅用的:)

### NetDAM拥塞控制

接下来我们来看看NetDAM的实现方式，和Google类似，解耦主机和网络的拥塞。
主机侧延迟隔离
首先来看主机侧，传统的网卡Buffer满了就会带来丢包，而处理器的工作负载也决定了响应的时间，另一方面PCIe DMA也会带来长尾延迟，这些其实都是影响主机侧延迟测量的重要因素。但是我们换个视角就容易理解了。

本质上来讲，当一台主机响应远端RPC时，都需要将请求放置到内存中，那么为什么不直接malloc一块NetDAM上的内存让远端写呢？例如NetDAM可以根据应用的需求，为单个Socket malloc 128KB的空间构成一个无锁队列，并针对socket构建一个socketMem的虚拟地址空间，然后远端的主机直接往这个空间写就行了：
![图片](assets/26f469b3c63b.png)
而每次WRITE完，直接由NetDAM交换源目的MAC、IP地址和源目的端口号，把Instruction字段改成ACK，然后Trim掉其他DATA发出给客户端即可，这样一来整个延迟是非常固定的，根据我们测试，平均响应`618ns`，抖动`39ns`，主机侧的抖动几乎可以忽略不记。而报文中最多带几个bits记录socketMem的使用率即可。

我们采用这样的设计主要还有另一个考虑，从数学上看`传输协议算子`如果满足`交换律`和`结合律`, 那么并行就非常容易实现了。如果是TCP，一定需要保序在buffer中重排序，本质上是TCP包与包之间是有依赖的，因此TCP_Send是非交换的。与此同时多路径(Multi-Path)实现也一直有问题，无法充分利用数据中心多路径带宽资源。

而NetDAM每个报文的请求是独立的内存地址，并且保证了Send空间没有Overlap，因此是满足交换律的，所以可以支持乱序写入，并且写操作还是`幂等`的，随意重传都无所谓（Sequence和address解耦了)，最终写入的数据就是一整块内存，也满足结合律，连LRO(Large Receive Offload),GRO(Generic Receive Offload)都省了，正是由于满足交换律，多路径支持也可以做了呀，看看当年flowlet搞得多烦人，而现在多简单，另外这个东西对于Serverless也很有用，明天说

而主机的处理能力，通过主机在SocketMem中Poll的数据即可观测到，关键是将SocketMem映射到用户态原生绕开kernel了呀，直接内存访问把整个延迟问题转换成了缓存深度的问题，而最好的情况是将缓存的深度保持为如下概率密度函数的分布，不用太深也不用太浅，类似于金融风控中的VaR来基于缓存深度阈值构建多bits的ECN就好。远端可以根据ECN采取不同的算法，当然MIMD在这种视角下是没有价值的：
![图片](assets/fca2440961d4.png)
简单来说就是在buffer深度浅的时候，拼命加杠杆(MIAD, multiplicative increase and additive decrease)，而在深度适中的时候平衡，而在极端情况下拼命减杠杆（AIMD，Additive increase and multiplicative decrease）

总结：通过确定性的响应延迟并采用共享SocketMem内存的方式通信，使得传输可以支持乱序并以SocketMem的空间使用率作为主机拥塞的指标。隔离了延迟测不准的大量可能性(CPU Hog/PCIe DMA/Memory contention...)而且基于BDP(Bandwidth-Delay-Product)来计算，内存空间并不大。

### Ruta Fabric延迟控制

认同Google的观点，从运维角度来看，没有必要将主机和交换机耦合协作，所以整个NetDAM设计都是完全和交换机无关的，除非是前文讲到的那种内存池业务利用交换机做MMU，当然这也是把它当成业务网关了。

数据中心交换机网络也可以做一些事情的，例如提供网络状况和拓扑信息供主机调度。例如下视频的自动驾驶网络

      
     
       
         
           
             
                                

                 
                   
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
   
 

NetDAM在Fabric通信上使用了Ruta解决方案，这是我们在2020年的一项工作，即Fabric可以通过主动测量延迟和丢包数据并将链路状态和节点信息放置于ETCD中：

![图片](assets/4ba0be5e7fb6.png)

例如使用Ruta构建的一个数据中心Spine-Leaf架构：

![图片](assets/d841cb96c2ad.png)

产生的拓扑和延迟信息如下：
![图片](assets/8518a338da8b.png)

而任何终端可以通过Subscribe ETCD相应的topic获得实时的链路信息，并根据这些信息，在传输层编码然后自主绕开拥塞路径

![图片](assets/261434ac0249.png)

具体技术细节可以参考如下文章：

[Ruta协议实战和详解](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485222&idx=2&sn=1802d8e50970ad9a8ffc2cfd0a82af34&chksm=f99617e4cee19ef238c6012e690d5238923c8e9380e32adb565957bb497540fa31bf7263e867&scene=21#wechat_redirect)

[Internet 的性能测量](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485629&idx=1&sn=753fbb174f3dffd7c05474d608d24952&chksm=f996187fcee19169d18aac2a4389e3ddafdc35bd991cbab76fae1430b10ea717f48fe3aba584&scene=21#wechat_redirect)

当然也有很多已有的工作，例如思科提出的SRv6或者华为正在推进的增强版IPv6+都是非常棒的技术，但是从硬件实现上考虑，和通信效率上考虑有些得不偿失，当然数据中心如果9000B MTU也无所谓，而SRv6的另一个问题是Option Header，用户态访问非常困难，仔细想想为什么Google要把SWIFT的Header插在那里，而不是Option Header？当然也有绕过去的办法，利用首跳的交换机copy出来，例如在传输层放置BSID然后网络团队自己拷贝。但最终不会比应用层自己做多路径有效。

![图片](assets/16303515aefe.png)

### 大象流处理

针对大象流的通信是一个非常难的话题，大象流的持续时间长，带宽占用大，而且按照传统的TCP的方式无法多路径打散，但是既然它持续时间长，那么就延迟不敏感，稍微增加一些store-load的延迟只要能够保证Pipeline能让收端带宽跑满即可，基于这种思路那么就可以玩Mempool了，例如昨天写道的Mempool架构，我们可以通过交换机虚拟一个IP地址，然后根据UDP内不同的NetDAM内存地址让交换机做MMU去路由：
![图片](assets/b2757ca67baf.png)

如果我们采用Interleaving的编址方式，连续的地址写入就变成per-packet load-balance：

![图片](assets/019f07aa27b1.png)

而且我们还可以使用Cisco贡献出来的memif来抽象管理这些内存，进行无锁通信。而这个实现变得非常简单，因为应用可感知大象流，通过allocate memory pool的地址空间作为Memif即可完成这个通信，并不需要其他特殊的操作。

想起一句话: Don't communicate by sharing memory; share memory by communicating. 但是很多人理解有问题，本质上是利用对内存的串行化访问避免加锁的意思，而NetDAM正是这样，通信和内存有很多辩证的关系，**通信本身是一个数据流，而内存只是其中某个时间点的切片而已**：）

### 结论

Google Swift的工作非常扎实，希望大家能够去看看。通过解耦网络和主机的拥塞情况控制非常认同，NetDAM通过主机侧解耦计算域与I/O域内存的方式，使得响应延迟具有了确定性(抖动仅39ns)，并且通过无锁共享内存的方式来实现通信，构建SocketMem空间，使得收发算子满足结合律和交换律从而多路径并行也可实现了，这样就降低了交换网Incast的可能性，同时在交换域使用Ruta协议实现了轻量级的主机可控多路径控制，使得延迟的期望得到了有效控制。最后针对大批量长连接大象流通信还可以借助interleaving in-network-storage的方式，使用内存池来打散和吸收大象流，进一步提高了Fabric的利用率.

想到乔布斯爷爷的一段话，分享一下，共勉:

*"Design is a funny word. Some people think design means how it looks. But of course, if you dig deeper, it's really how it works. The design of the Mac wasn't what it looked like, although that was part of it. Primarily, it was how it worked. To design something really well, you have to get it. You have to really grok what it's all about. It takes a passionate commitment to really thoroughly understand something, chew it up, not just quickly swallow it. Most people don't take the time to do that."*