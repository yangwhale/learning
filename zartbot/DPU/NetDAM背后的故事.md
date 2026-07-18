# NetDAM背后的故事

> 作者: zartbot  
> 日期: 2021年10月13日 16:06  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486663&idx=1&sn=0c50fdb95195c821d625acbb91b16672&chksm=f9961c05cee1951302b5830ebcdf3b448b946a179597bcaf8bb940a0b9b6994472bfb6ab7d23#rd

---

其实任何研究或者创新都有大量的取舍和思索并没有放置在台面上，至少论文不会写出来

NetDAM研发始于思科去年投资了第四范式后，我们作为思科工程团队的代表跟第四范式团队探讨合作创新时，针对MPI Allreduce聊到了GenZ、CCIX、CXL等技术，当然也聊到了RoCE，一些以前的经历让我们敏锐的发觉了PCIe本身才是RoCE最大的软肋，大量的DMA带来了内存带宽管理的难度，所以目的很简单就是隔离IO域和计算域的内存。而I/O域的内存很多都是一次使用，基于MPI模型下的内存并不太需要一致性维护，但是I/O域的内存怎么放置呢？

NanoPU犯的错，大概15年前某芯片就犯过，当年的FMN和NanoPU也没啥区别，不知道现在是否明白当年怎么错了的么？

![图片](assets/b591cfa71540.png)

而为什么另一个思科的QuantumFlow处理活了15年，整个平台卖了几十亿美金，至今到单集群4块QFP-3，每颗处理器224个核，每个核4个线程，总计3，584个线程。核的充分使用和调度的最核心问题就是，我们在处理器和外部I/O之间添加了可编程控制的内存器件GPM和相应的dispatcher和gather,下面蓝色的那一块东西

![图片](assets/75ee8f3e63be.png)

QFP这个架构诞生于2004年，在I/O设备上放置可编程控制并可以直接访问的内存对于提高性能有非常大的帮助，这也是当年那么多网络处理器竞争，思科能完全保持领先的原因之一，当然另一个最大的原因就是对软件编程的友好性。有些厂家虽然也能做到数千核的产品，但是编程有大量的限制，指令条数约束，访存约束，而思科的处理基本上没有啥约束，这也是后面虚拟化时代和低端平台能够很快迁移到Intel X86的原因。

所以在整个设计的过程中，我们必须要保持一个最基本的原则：

Software when you can, hardware when you must. Whenever possible, compute, networking, and storage functions should be done in software where reasonable performance can be attained.If you have to accelerate something, use the most generic and malleable compute engine or network ASIC that does the trick. This might mean sticking with a CPU or a GPU for certain functions, or even using an FPGA

但是有些人读完评价说这个项目志向远大，把Xeon弄没了，其实是完全错误的。`比RoCE快20%`就可以让所有做DPU的厂家自己去找差距了，特别是那只用TCP的Fungible~嘻嘻~

更详细的解释如下： 

第一，这个项目是用于隔离I/O和内部CPU计算对主存的争抢而存在的，从而在I/O域获得确定性时延来降低拥塞控制复杂度。并且在编程方式上，即便是没有NetDAM这样的硬件，软件上也是通常使用多核共享内存，核间MPI的方式,并没有用专用指令集取代通用CPU的意思.而且有些任务offload要200ns 因此在进入和离开cpu时构建一些指令做加工是有必要的

第二，相反这个项目后期的目的是会进一步的优化和CPU之间的PCIe、CXL总线，并扩展一些Remote Memory Prefetch RPC指令，使得CPU能够在主机内更加容易获得所要的信息，但与此同时避免I/O对CPU的Cache和主内存的干扰。我们下一个项目就是要做X86多核的一些调度算法。

第三，将独立的I/O内存映射到虚拟机的User-space中，并通过思科开源的MemIf实现Node.JS/ Golang 等多种语言仅数行代码微小修改或者不修改的情况下获得每秒千万级消息响应的能力，而同样这些东西也可以通过DPDK在没有NetDAM的环境中用CPU核来软件模拟。

另一个关于incast的处理，的确是借鉴了很多年前Juniper M系列路由器的处理方式,它会将接收到的报文切分为J-Cell每个64B然后分布式的方式丢到各个线卡上,然后最终目的接口的线卡从每个卡上取出来.其实思科也是采用同样的方式构建interleaving的memory pool实现的多芯片访问而避免incast情况，但是这些东西必须挡在I/O和处理器之间。

NetDAM针对于MPI Reduce的处理方式在我今年看到HotChip33 Cerebras的时候，会心一笑：

![图片](assets/bca85b94928c.png)

![图片](assets/86cc1eef8f46.png)

同样看到Jim Keller所在的TensTorrent的架构时，这才是世界最顶尖的设计方式：

![图片](assets/87817ad7e231.png)

其实现阶段，除了真的在数学上算法上或者物理材料上颠覆式的创新。在材料固定的方式下，大多数的情况都是大同小异，很多人容易因为自己的一知半解而忽视掉很多精妙的细节和取舍。很多人单纯的以为，不就是把内存怼到网卡上么，不就是想拿ALU替代CPU么？其实背后只是在尽量不动软硬件的情况下取舍，各让一步海阔天空。

在整个开发的过程中，我们经常讨论的一些问题都在开玩笑，这些东西都是20年前就做出来的了，为啥还不知道，特别是在去年读到NanoPU的时候，由于100G的timing constraint和片上网络约束，我们当时就断定直接怼处理器上去不行的，至少可执行的cycle最多200个左右, 另一方面的佐证就是Intel OPA的失败,Xeon Phi+OPA就是类似的问题。最根本的原因是，现代社会知识体系结构太复杂，很多人只是把一个微小的领域挖的很深已经人到中年了，自己顺手的技术自然会想着去拓展它。

例如做网络的人总想着最好SRv6把汇编指令都支持，以后不需要CPU了。而且很多指令的处理上和主机内通信的方式格格不入，这是导致SRv6推进缓慢的最大的原因。而另一方面做主机的人不想求网络，总觉得自己把CXL、PCIe拉出来几个机器连好就行，GenZ和最近基于CXL的内存机框也就是这样的思路。最终这些会因为差分信号的传输长度和硬件上的时序约束使得它只可能发生在1~2个机柜内。

因此作为架构师一定要尽量多的扩展自己的知识广度，当然这样端到端的理解对于很多人来说时间和精力都是难题，这些东西要么就是学的快、要么就是学的早，时域拉长或者频率提高的问题...

当然我们是因为国内某交易所的低延迟交易项目，自然要从硬件上一个一个扣，自然会从物理层上看802.3看起， 例如在接收的PHY上少一个寄存器，觉得AXI延迟很大还改协议少一个信号，至于CPU Prefetch这些东西都懒得放台面说...真的高手过招哪个不是一点点的扣?

最近看到两弹一星时期的计算机指令手稿， 我们已远离那个奋斗的年代太远了...各自做了一点微小的工作就沾沾自喜了... 我们这一代人啊，缺少了太多的勇气,即便是历史给我们了太多的机遇，我们依旧保守的走着教条主义...

![图片](assets/b190037c0d3e.png)

![图片](assets/ed059e9b065c.png)

![图片](assets/03fe93afa00b.png)