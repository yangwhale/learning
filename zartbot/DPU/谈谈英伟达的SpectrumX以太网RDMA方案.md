# 谈谈英伟达的SpectrumX以太网RDMA方案

> 作者: zartbot  
> 日期: 2024年3月18日 16:06  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489018&idx=1&sn=25b3df2a17d49681edc0e621049b058f&chksm=f9960538cee18c2ed59729db4a0194fd54b6b99670af13bec8a8eb82b250ccb6b0aa5f07caa7#rd

---

英伟达对于Infiniband和Ethernet的Spectrum-X定位还是非常清楚的， AI Cloud和AI Factory的最大区别在于多租户规模和训练任务的大小。

![图片](assets/4aa379c448f2.png)

因此最近读了一下下面三篇文档，记了一些笔记，分享出来。不想看这么多的直接跳到最后看结论。

Networking for the Era of AI: The Network Defines the Data Center[1]

Next-Generation Networking for the Next Wave of AI[2]

NVIDIA Spectrum-X Network Platform Architecture: The First Ethernet Network Designed to Accelerate AI Workloads[3]

## 1. 关于刚性兑付**(Lossless)以太网

这是我经常拿RoCE Lossless开玩笑的一个梗，银行都不搞刚性兑付了，怎么网络还搞？

其实一直有个问题，在多租户环境中，如何基于Lossy Ethernet做到租户隔离，多任务性能隔离？似乎Spectrum-X的解决方案利用SP4交换机+BF3做到了部分，但是很遗憾的是交换网开启了PFC，并且Mellanox强调了Lossless Ethernet的重要性：

![图片](assets/fd2edd67749b.png)

但是在2018年IRN论文中《Revisiting Network Support for RDMA》描述的是：

![图片](assets/49f81ce13017.png)

这一点是一个很矛盾的地方，当然我能理解Nvidia在市场策略上希望和Infiniband尽量统一技术栈，但是对于云VPC网络而言，如果Nvidia希望Spectrum-X方案用于AI Cloud的多租户环境，那么势必就要接受Lossy Ethernet，并且其组网规模相对于PFC based Lossless会大很多，并且不会产生PFC导致的拥塞树。

事实上，Mellanox(Nvidia)关于PFC这事的摇摆，本质上是在多路径环境下，当发生丢包了以后，是CC降速还是选择更改路径，这两套算法之间的矛盾，最终选择了打开PFC刚性兑付降低CC难度来实现。

## 2. 关于多路径和乱序提交

这也是英伟达重要阐述的，即便是在1:1无阻塞的交换网中大象流也会导致长尾，Direct Data Placement(DDP)和Dynamic Load Balancing是正解

![图片](assets/fac3067363ad.png)

这一点上基本上工业界已经达成共识。

## 3. 关于拥塞控制

英伟达的观点是ECN并不适合生成式AI任务，因为这些数据量太Bursty导致丢包的问题。

![图片](assets/0aaca2706305.png)

![图片](assets/3020406b1049.png)

关于Busty Traffic有一个图

![图片](assets/c7ccb865fb6f.png)

然后英伟达还写了Deepbuffer交换机无法解决这个问题，解法是需要交换机SP4和网卡BF3进行端网协同。并且英伟达花了很长的篇幅写Deep Buffer这个事情， 背后在diss啥，其实大家都清楚

![图片](assets/99cff6db2332.png)

这一点上关于ECN的结论是正确的， 关于Shallow buffer的结论也是正确的，单是拿延迟去说Deep Buffer交换机不好可能有些不是那么的公允了，并且指向性到某家的某个芯片还是太明显了。事实上最好的解法就是Window Based CC来替代Rate+ECN based CC，这一点相信Nvidia自己也清楚了。毕竟Google Falcon和GPUDirectTCP-X都在那里放着，Swift based CC对交换机buffer的使用，特别是针对incast这样的环境，谁心里都有数。

## 4. 拥塞控制需要算力

英伟达有一段描述：

![图片](assets/7465d0e9f145.png)

![图片](assets/febae985a527.png)

这一段话证明了传统的SmartNIC并不适合AI workload，更多的需要DPU来支撑，并且对于CC算法有足够的可编程能力来适应不同的拓扑，这里是特别需要算力的地方。

## 5. 拓扑亲和性调度

如果将Job放置在同一个机柜性能肯定比放在不同机柜好
![图片](assets/18a22b838d70.png)

![图片](assets/11a973956e71.png)

如上图左端所示，通过对Job节点的调度基本上可以获得15%以上的收益。但是通过使用多路径的Spectrum-X解决了调度亲和性的难题，为了实现这个功能，需要BF3的算力支持，一旦支持了，收益是非常明显的，扩大了调度域。

## 6. 多租户性能隔离

一个问题是Performance Isolation, 有Noise neighbor，然后多个job有干扰什么的，还有需要安全隔离等问题

![图片](assets/5aaa5119bbd7.png)

但是在这个问题上，Mellanox似乎有些认知不足呀？现在哪个交换机不都是Shared Buffer的，值得拿出来说么？

![图片](assets/20215cec9804.png)

另一方面明确了VPC网络实现上的问题

![图片](assets/f605e6e0e2a0.png)

这里明确了VXLAN on Switch 存在ACL/安全组等高级功能缺失，特别是在扩展到万卡集群时。多租户隔离一个是数据面安全隔离，另一个是不同job之间的隔离，防止Noise Neighbor。本质上我们来看是否能够通过拥塞控制算法在端侧做完，尽量保证所有的端对交换机buffer的最小占用，并且在subRTT内相互快速避让，这个事情就解决了。

## 7.网络失效恢复

大模型训练对Link Failure十分敏感，影响很大

![图片](assets/a9420a50b728.png)

英伟达的解法是在卷交换机上做快速重路由和负载均衡。

其实这也是对多路径和拥塞控制算法考验的一个要点，如果不要交换机，直接在端侧能做到毫秒级收敛并在剩余链路中达到负载均衡的多路径算法该怎么做？都可以仔细思考一下背后的难度。能做到这点才是真正的挑战。

## 8.对于workload的定义

前面说了这么多问题，来说点好的，英伟达对问题的定义是非常精准的

![图片](assets/425b2b5d61a2.png)

## 9. IB和Ethernet的差异化

英伟达的角度来看，IB有更大的Scale，更适合单租户的场景和少量任务超大规模Foundation Model的训练场景。并且还有In-network-Computing(SHARP)的支持能力。这些都是它定义的AI工厂的业务逻辑。

而针对AI Cloud， 基于以太网，VPC多租户的安全隔离和性能隔离，通过AR提高Fabric利用率是关键，这些关键因素决定了在以太网产品线上需要有算力支撑，因此必须要走BlueField的产品线，同时以太网本身的Lossy特性需要通过Lossless和端网协同的技术，要求BF3和SP4交换机配合进行实时的Metering

![图片](assets/72502c81004e.png)

当然除此之外还有一些FECN BECN的东西，这些几十年前Frame-Relay/ATM上就有，在以太网上搞一套并不难。

![图片](assets/1491b03a0f35.png)

事实上基于以太网VPC并池的逻辑，Google GPUDirectTCP-X可以很容易的拿未售卖的2Core实例来构建一个VM做Reduction Server 实现类似于SHARP的功能。另一方面性能隔离BF3和SP4配合的工作可以通过Swift测量RTT配合Window Based congestion controll很好的解决掉， 这样做好了Lossless PFC带来的Scaling的问题也可以解决。因此可以实现一套AI Cloud scaleout to AI Factory的能力。

## 10. 结论

本文通过英伟达的白皮书，分析了其在多路径算法和拥塞控制上面临的两难问题，最终为了简化问题，选择了使用PFC Lossless 来构建BF3+SP4的AI Cloud方案

英伟达认为ECN这些东西不行，需要通过BF3 Flow metering和SP4 realtime detection来做，事实上per-packet RTT 并且通过Swift Window based CC可以很好的解决这个问题。

英伟达认为AI cloud的多租户环境不仅要安全隔离还要性能隔离，前者在交换机上跑ACL/VXLAN针对万卡集群是不能Scale的，因此这些工作需要在端侧BF3上做。

解决拥塞控制的难题需要算力提供，BF3上有相应的计算资源来做这个事情构建Programmable CC，未来产品定义中很有可能CX7/CX8这些就专门做IB了，针对以太网Lossy环境，还是需要DPU算力走BF3/BF4路线。

英伟达认为SHARP和Scale是IB相对于Ethernet方案的优势点，其实这些问题在以太网上解决了多路径和拥塞控制算法都可以做的很干净。

AI网络对于链路失效和快速收敛要求很高。单个迭代内快速收敛，最好是ms级别收敛是一门艺术，很多人在做多路径算法时，对这个问题的处理并不干净。

亲和性调度的收益在多路径环境中会变小，多路径可以增加调度域。

最后，英伟达说这些性能可以提升1.7x，是真实的，当然不需要这些网卡和交换机，甚至普通的上一代的交换网都能做到，端口利用率还会高于英伟达的这套方案。

![图片](assets/5173cfe183fa.png)

多读读文档，多想想背后的逻辑，多动手做点有趣的事情吧～ 接下来就要分析B100/B200/GB200了

![图片](assets/189e95598943.png)

![图片](assets/e1840e120140.png)

很多人对GB200和GH200针对推荐系统场景下的一些问题处理嗤之以鼻。事实上这是一个非常好的平台和大模型真正落地能产生收益的地方。具体可以读读下面这篇文章

[》谈谈AI落地容易的业务-搜广推](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488831&idx=1&sn=192ac23cf674db05d80576c6eac2200f&chksm=f99605fdcee18ceb926a9f59682c7203cc589305be08d9e3cf289da2f489ab50826654e66b88&scene=21#wechat_redirect)《

参考资料

[1]
Networking for the Era of AI: The Network Defines the Data Center: https://nvdam.widen.net/s/bvpmlkbgzt/networking-overall-whitepaper-networking-for-ai-2911204
[2]
Next-Generation Networking for the Next Wave of AI: https://resources.nvidia.com/en-us-accelerated-networking-resource-library/next-generation-netw
[3]
Spectrum-X Network Platform Architecture: https://nvdam.widen.net/s/h6klwtqv5z/nvidia-spectrum-x-whitepaper-2959968