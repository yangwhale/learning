# SDM:软件定义的内存

> 作者: zartbot  
> 日期: 2021年11月5日 02:54  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486882&idx=1&sn=6fc254fe22ab74b00ca50bccd6a5aac1&chksm=f9961d60cee19476e2f8b9e6b05ea5be3a7107d41a4ebae4ae1c6f8fe787657a56b113ee936c#rd

---

这也不是我创造的什么新词，只是刚好和最近NetDAM的工作有关，又看到SDC21的一个ppt来给大家分享一些看法.另外昨天拜访了某个跟脸书竞争元宇宙的大厂，发现Ruta的很多架构设计已经被他们用起来了，非常开心~ 

说到SDM要不要做个软件仿真的NetDAM呢~~

![图片](assets/5c979071b4dc.png)

SDM**简单的说就是提供一个软件的抽象层在应用和底层内存资源之间实现动态的内存控制和SLA保障，技术架构上来如下:

![图片](assets/0daa8ecfc43a.png)

而我们在NetDAM中进一步的把这个内存抽象层扩展到了网络甚至直接为一些DSA提供AXI的接口，例如一些Computional Storage的需求:

![图片](assets/5063d72c712d.png)

NetDAM的论文可以访问: http://arxiv.org/abs/2110.14902

SDM主要的使用场景如下:

![图片](assets/69428492cee2.png)

In Memory Databases
在分布式事务中通常需要一个Sequencer做逻辑时钟，然后我们在NetDAM上实现了Fetch-Add-1的原子操作，即任何一个UDP请求可以Fetch一个uint64的整数，然后NetDAM会利用自身的SRAM block fetch**后原子操作自增1来构建这样一个高速的Sequencer. 当然也有很多现成的FPGA based K-V store的工作，NetDAM也可以把这些工作整合起来构建一个超大规模的分布式存内数据库,期待和某大厂的合作~
ML/DL/Inference
这也是我们一开始做NetDAM的主要目的，第一个就是解决超大规模分布式机器学习中的参数同步MPI-Allreduce场景而实现的，而另一方面在NetDAM上配置了大量的ALU**资源也非常适合直接拿NetDAM来做模型推理，顺便吐个槽，樱桃、Marvell和nVidia都在DPU上加了ML引擎，但是基本上没有能够商用的用例，需要技术扶贫联系我~
Multi-tenant Memory
这也是一个很好玩的场景， 那就是Serverless，内存的多租户隔离和相应的寻址保护，光靠CXL Fabric或者RDMA**是搞不定的，同时你要考虑到普通用户怎么使用，所以我们一开始就基于UDP/Ethernet实现，好的技术是能让大家都没有门槛的玩。
In-Mem Analytics & HPC
其实这两个场景和第二个很类似，就不多做阐述了.

针对未来更多更复杂的计算场景，通信的本质其实就是在共享内存，所以在这些计算系统中隐藏通信层，在软件层用户态上提供内存抽象和管理变成了解决冯诺依曼**架构瓶颈的一条路，而SDM必定会在这条路上写下浓墨重彩的一笔

![图片](assets/aa24e4676fe2.png)

最近做了一些测试，利用stream顺序内存copy，在Intel Xeon 6230R x2 + DDR4-2933 x 12上，峰值带宽两个Numa节点加起来能到150GB/s，但是考虑到未来400Gbps主机网卡，PCIe只能接在一个Numa上，这样400Gbps光DMA写入就占用了 (400G/8) / (150G/2) 66%的内存带宽，而在阿里云上测试了一下AMD Milan 7T83，考虑到Milan有8个Channel应该更好，但也只能跑到100GB/s， 也就是说现在的CPU的内存子系统已经完全无法支撑RDMA的发展了，所以NetDAM这样的技术必定会成为未来的一个刚需.

还是那句话，早上车早富裕~，然后才能带领大家共同富裕~ 有些上Ruta的公司先富起来了：）