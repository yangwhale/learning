# MISD是DPU的必然出路

> 作者: zartbot  
> 日期: 2022年8月23日 16:00  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488032&idx=1&sn=6ed8672fa9f58fe1f777ac87b8baa369&chksm=f99602e2cee18bf42989ff663a130424a2ea685f5b1b51ba065a706577bd1c920e2ade604a7e#rd

---

最近写了一系列读书笔记，从应用侧看3D渲染算法本身的需求，然后从芯片侧，追溯到1980年SGI的首款图形ASIC**，再探索到现代固化流水线，然后多级流水线可编程，再到GPGPU的整个发展路径，把自己带入到当年的架构师决策者的视角，看懂了很多问题，也就是“Boil things down to their fundamental truth and reason up from there”，

[GPU架构演化史1: 3D渲染算法概述](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487954&idx=3&sn=f494b3d84bc8a0b0ac912e0940938f18&chksm=f9960110cee1880664e00f3b5dad57eef629e98793ecd71f2adf3a3b39f3e2b3d9281d764ad1&scene=21#wechat_redirect)

[GPU架构演化史2: 1980-1993 SGI时代](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487954&idx=2&sn=038983b8328c6e6c56fe25188b16b640&chksm=f9960110cee188065d7c7c7cba3ae149e20f8d54f7c8f2a38a7363e22ade1b4f3b69be9e64a2&scene=21#wechat_redirect)

[GPU架构演化史3: 1994-2000 群魔乱舞](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487954&idx=1&sn=a7a8a61b1d1fd179dc525d251bdfdbd1&chksm=f9960110cee18806664f3728109483eee36e2d2f12b2b8c3f0f2624fdf211a7eafb2ef7701eb&scene=21#wechat_redirect)

[GPU架构演化史4: 2000-2006 AN争霸可编程](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487976&idx=1&sn=d258826f5829b2225b93183248f3f893&chksm=f996012acee1883cd8a41eb8d57d6b6fee57357d5ffd184883a89a5b4524c796aec27e7840dc&scene=21#wechat_redirect)

[GPU架构演化史5: 2006-2010 统一着色器带来初代CUDA](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488000&idx=1&sn=95ff38f9e2d1a6f4df3e2adc223bd8af&chksm=f99602c2cee18bd4cb7781cc1e3af0f1fc1aad3ad455920cd7c993515e7d2baa304bb71355b7&scene=21#wechat_redirect)

[GPU架构演化史6: Telsa CUDA架构详解](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247488023&idx=1&sn=2a3d6f808b9d52396edb230ca8ce6e09&chksm=f99602d5cee18bc357812b1f682bb3f45c404da328688d825071de81ed8b69d7c92e1e48e592&scene=21#wechat_redirect)

当然这个笔记还会继续记录下去，有机会跟大家再分享出来Fermi**到Hopper整个过程中又遇到了什么难题，做出了什么取舍。

有人问我为什么要看GPU，一方面是好奇，而更重要的是如果说一个做DPU**的人，不懂上层应用，不懂网络，不知其它两个Slot的区别和第三Slot的差异，这个问题是很难讲清楚的，Jensen Huang似乎在讲DPU的时候都没讲清楚，为什么数据中心需要第三个Slot? 有什么CPU不能做的，有什么GPU不能做的，每个Slot处理的任务在新时代又什么划分依据？DPU做了什么CPU和GPU都不能做的事情?

![图片](assets/f47b80ae0975.jpg)

而上一个回答这个问题的是Intel Larrabee，它认为能找出一个既能又能的东西，结局大家都懂

![图片](assets/d7675e2e8894.png)

然后Intel又开始搞Tofino，包括思科也在弄Silicon One，或许又是一个既要又要的东西，结果是什么不清楚，可编程网关这个市场本来就是高Margin和低销量的场景才能养活团队的，点到为止..

回顾当年nVidia搞CUDA的时候和CPU区分的可明白了:

![图片](assets/ed3cfa2f3115.png)

所以CPU是MIMD的，GPU是SIMD的，也就明白为啥Linus要吐槽AVX512这种东西在CPU core内就是垃圾了，从体系架构上来看，DPU是需要MISD的，所以像我在Ruta+NetDAM里面要将Data和多个Instruction紧耦合，并且支持多指令栈，但基本上不太需要有Branching的地方。

而另一方面，CPU是task并行，要求低延迟复杂逻辑，数据吞吐量并不大，而GPU是延迟可以容忍给Warp调度同时又有高并行和高吞吐。从延迟和吞吐的角度来看， **DPU必然是需要一个极低延迟又要有极高吞吐，然后还要有类似的完全灵活可编程的能力，例如一个常见的例子就是inline crypto with some branch decision for scheduler and others, GPU做不了，CPU做的慢，是不是?**

问题的关键是怎么填坑和拉高整个曲线，坑形成的原因，都是值得各位仔细回味的：

![图片](assets/6e4c907dbc59.png)

而另一方又有一个痛苦的事情，TCP要求保序从而使得DPU可并行度降低，利用Flow并行，大象流或者大量老鼠流都会弄死你。So？

看到nvidia SIMT和初代CUDA的Telsa，当然这也让我想到同一年思科做QuantumFlow处理器的场景，仔细对比了两者的区别，一方面是上层API，另一方面是线程调度机制，其实我们从90年代末期用各种微码NP，包括后来用Intel IXP，最后自己弄完QFP后想的很清楚，市场的需求在哪，应用的需求在哪，架构该如何取舍。

这个问题的答案还和历史背景和相应的应用相关，如果说到当下，那么FPGA成了必然，因为无论是阿里还是AWS都要宣称自己的延迟和支持的PPS作为DPU关键指标，这样也应证了我前面的话，体系结构上MISD是必然，至于Nitro会逐渐的在延迟上吃亏，包括BlueField亦然，且会因为吞吐也出现落差，而至于那些微码引擎，你看有几个商用的还在做微码编程NP的公司？

至于成本的问题，做成ASIC是一个代数问题，涉及编译和编码，一个崭新的体系结构

而同样遗憾的是，nvdia在转向CUDA前也有Cg、HLSL GLSL这些domain specific的语言， 不就和我们现在某些人张口闭口就要P4的论调一样么？当然还有一波拿SRv6做VLIW**的，该醒醒了...

而未来，至于MISD该如何设计，硬件架构，Core的架构，调度、I/O笑而不语，避而不谈.

最后突然想起新的国标红绿灯，真觉得交通部应该让做网络的人给你设计，路口一定搞八车道，根据车牌来做DSCP，说不定再红绿灯前面再搞一个红绿灯做双速率桶的流控，然后还要留一个专用通道做确定性行驶。拥塞发生了，先把乘客和货物丢了，然后自动驾驶开回起点通知人没了，然后一开头就每过一个红绿灯换个车牌，行驶状态通过车联网往下游传，到末端后采用自动驾驶和确定性行驶告知头端发车速率。

有些时候吧，真的没必要把事情搞复杂了，简单的无歧义的三个灯比什么都好用。