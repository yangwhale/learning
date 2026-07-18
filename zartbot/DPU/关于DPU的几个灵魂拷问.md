# 关于DPU的几个灵魂拷问

> 作者: zartbot  
> 日期: 2021年9月25日 16:45  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486464&idx=1&sn=530855cac1c93332c9c6da77c02e9d2b&chksm=f9961cc2cee195d497faac20157ffa370a5b4709fc4bc2e9ef6757200702ffc488e2c88f5926#rd

---

### 1. 是什么撑起了DPU的估值?

从市场来看,一线公有云除了Google没有DPU都有自己的团队,包括国内的阿里神龙. 二线公有云会用一些,但量有多大,一万片?唯一的机会是腾讯、电信、头条， 其它云一坨屎。。。

### 2. DPU企业网市场

估值中很多都提到了企业市场，企业网是否真的有Datacenter Tax， 如果只有数量有限的主机，DC Tax和对应的管理成本开销比起来，根本没必要上DPU。相反，某司专注在CWOM、Intersight、AppDynamics以及最近收购的一家K8S profiling公司，在应用层挖掘DC Tax更加有效。Pensando很想卖企业里，但是做不成的原因就在此，

### 3. 轻视软件的DPU

几乎所有的厂商，除了Nv以外，都在说自己DPU多么的牛逼，硬件有什么功能，谁想过软件该怎么做？都是去offload OVS？按照一个芯片的开发团队分配来看，软件SDK团队比硬件更加重要，最近几天被某厂的IPsec offload折磨的半死。。。深刻体会到硬件和软件巨大的鸿沟

想起一句话：

Software when you can, hardware when you must. Whenever possible, compute, networking, and storage functions should be done in software where reasonable performance can be attained.

Xilinx 这种逐渐硬化的路其实是非常正确的， 软硬件融合必定会加入FPGA

If you have to accelerate something, use the most `generic` and `malleable compute engine` or network ASIC that does the trick. This might mean sticking with a CPU or a GPU for certain functions, or even using an FPGA.

### 4. 最佳的DPU实践

Marvell算一个，毕竟思科的中低端产品线都是用它家的，而且Cn10k系列....然后另一个是Xilinx的新的带HBM的versal，加解密和PCIe已经硬化了，所以用起来非常爽啊，400Gbps的主机网络不是梦~至于其它几家， 都是...Fungible架构有问题，用MIPS导致工具链就有问题，各种offload engine如何访问也有难度，软件协议栈上的问题大大的有， 然后Pensando，P4和ARM Core的协调，到底什么feature要用p4写什么用ARM写， 软件上处理非常困难。

领头的两家都这样了，国内这些要号称的对标他们的尴尬不？还没事自己折腾个峰会来玩。。。自娱自乐啊？

### 结论

当一个行业最开始创业的两家公司三年都没挣到钱，你们还前仆后继的往坑里跳，这是干嘛呢？当然我也承认我也在做一些DPU的项目， 例如基于Xilinx FPGA做针对AI训练的MPI优化，这个东西的结果远超ROCE,大概国庆以后把几个bug修掉测试完了就可以发布。然后另外一个就是基于Marvell CN10X的100Gbps以下的项目，以及基于Xilinx Versal 400Gbps以上的项目， 具体是什么就不多说了。。。

结论二：你们啊。。。忽悠投资机构的技术还需要提高啊。。口号喊得特别响，得有真本事啊，你们想想看，多少风投找到我来给他们审项目，你们觉得你们能忽悠机构么？机构的人比你们还懂技术，哇哈哈哈哈哈~~~

例如，有些人当年搞732 就没干过某大厂，现在觉得自己牛逼了？还不是从Arm8升级到Arm9把自己的研发周期拖后了2年， 然后还有很多厂的BP抄哥哥我的公众号几个意思？给你们几个DPU的参数，自己找差距

![图片](assets/07283948e213.png)

![图片](assets/eacf6a00a282.png)

某司DPU从2000年开始规划，最终整个产品线买了10B USD，里面有各种特殊的技术， 只是因为某司NDA的要求，无法透露，当然我们也在玩更先进的东西，技术太先进无法展示。。

国内的这些厂啊，真希望你们能够弯道超车好好的踏踏实实的做技术啊，我也是在某大厂磨砺了快15年了才学会了很多东西，你们啊，少点口号，少点浮夸，踏踏实实多做点事情不行么！ 

一天到晚讲赛道和风口的人是最没价值的， 短视而又肤浅.. 卧薪尝胆四个字，好好琢磨一下吧.....