# 使用Agent做Nvidia GPU芯片微架构分析

> 作者: zartbot  
> 日期: 2026年5月28日 23:18  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498845&idx=1&sn=df702e2854383184d47307fd39a5f309&chksm=f995ec9fcee2658920b772d802b56b42d0c5b7b89ad157e3a2618d606f7d40f307ba7b73e27f#rd

---

### TL;DR

最近玩 Agent, 正好手里有几台 RTX 5000 pro 的机器, 就拿来做了一个微架构分析的尝试. 分析报告有两份, 都是Agent生成的, 一篇是正常的微架构介绍, 另一篇是偏论文格式的阐述. 大家都知道, 做这些活很累, 但是有了Agent, 从写 skill 到完成所有的报告大概也就3天不到, 整个测试流程估计就花了6~10个小时.

感觉老黄以后出一颗芯片, 我就可以拿到手以后很快的扒干净了... 嘿嘿...  其实这个工作是受到 Ligeng 的启发, 昨天他也在 SGLang Office-Hour讲了一下《SGLang Office Hour on Agent Loops》[1] 去使用 Agent Loop 做算子优化的工作《Kernel Design Agents》[2]. 或许OPC**的时代会很快到来, 而且大企业内部也会出现一些相对较小的精英团队.... 但是我觉得吧, 老板更容易考察我是否摸鱼了... 例如哪天我的token花销少于4个亿, 一定会怀疑我摸鱼的...

当然这只是一个调试Agent的版本, 里面估计还有不少错误, 正在开发第二代的 Agent 来解决, 于是这一代就不修了...反正全部是 Agent 在扯的, 错了不要怪我...
微架构解析
**https://zartbot.github.io/micro_arch/nvidia/sm_120/microarchitecture.html**

![图片](assets/3477aad49a7e.png)
论文格式阐述
**https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html**

![图片](assets/d640002c6f30.png)

### 1. 谈谈大概的原理

其实Nvidia的最大优势是 CUDA Programming Guide, PTX document 和 CUDA Binary Utility做的很好, 这样就可以爬下来对每个指令进行分析.  然后呢还有一些技巧, 例如 CuAssembler 不支持新的SM_120, 不妨碍我用 Agent 一个个的自己校验, 支持SM120的可以在 **https://github.com/zartbot/CuAssembler** 获得. 然后就是让 Agent 一个个的测咯...比较难的是如何根据ptx文档找到所有指令和让 Agent 自己设计case, 然后就是一些让 agent 构建反汇编以及自动用 ncu profile , 最难的是harness的几个skill. 当然有 humanize** 这样的很棒的项目, 不过暂时还由于一些原因没用上...

### 2. 谈谈感想

我觉得Agent时代吧, 和我当年刚上互联网的时候差不多的感觉, 大概96年/97年的时候, 当时经常通宵上网, 而现在经常通宵并发好几个任务. 客户问到的问题, 基本上随手丢给 Agent 去先搞一下. 开源代码的调研阅读也是... 上周还随手写了一个 [《GFD》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498446&idx=1&sn=0a9a70a346ff1ba03348ff07d11829e2&scene=21#wechat_redirect)在Host-to-Device的拷贝速度上暴锤Nvidia自己的Cudamemcpy, 还顺便发现了一个cudamemcpBatchAsync的潜在问题... 另外还有一些内部的工作, 例如KVCache的整体架构设计, 新的Attention架构设计, 顺带还可以让 Agent 给我制作教材学习, 例如《从高中数学到代数几何》[3].

基本上觉得个人的工作效率提高了 5~10 倍, 因为通常是好几个任务同时并发干...

谈谈开发过程中的一些感想吧, 首先第一条就是不要把人的主观经验带入, 模型已经掌握了比人更多的知识了. 举个例子, SM_120 的 GPU 支持 MXFP4 的一些 PTX 指令挺烦的, 需要 sm_120a 的架构, 然后Agent一直搞不对, 于是我强加一些限制, 遇到 unsupported 的时候, 去改改 compute_arch 试试. 但是似乎模型并没有那么听话, 在后面的case 无论遇到什么编译错误, 都莫名其妙的去切 compute arch, 然后过一会儿就自己报错 case failed掉了.

然后针对测试有1000个case, 过滤failed也挺烦人的, 我这么懒的人可以懒到让 Agent 自己帮我写skill来处理, 反而比我自己写的好得多.

### 3. 谈谈 Agent 时代的个人技能

可能以前比较熟悉某几个开源库的代码就能找到一份不错的工作了, 而在 Agent 时代, 这些或许会被很快的替代. 例如写算子的岗位, 我估计年底就差不多可以大部分交给机器了. 或者更直白一点, 对于比较垂直的知识体系结构(换句话说, 知识面不够广但对某个特定领域专研比较深)的人, 或许被替代的概率会显著增高. 特别是哪些曾经靠类似于刷题(现在叫蒸馏)的做题家们, 在当下会越来越难. 毕竟去熟悉一套开源代码的时间已经大幅度降低了. 而人更多的应该关注在基础知识的层面有更广的涉猎. 例如最近在做的一些工作是代数几何这些算法如何用到大模型上... 其实大模型你让它找亚线性**的Sparse Attention算法, 我估计大概率它不会想到这一点. 而我无非是能够把几个领域串接起来

![图片](assets/b6de53a04cce.jpg)

当然还有一些本职工作是设计AI Infra的架构的...对GPU微架构的理解, 对传输协议的理解, 对各个器件的理解, 就很容易的做出来一个系统方案了...

![图片](assets/4879cc032120.png)

大概就这么多吧, 本篇是灌水... 因为很多具体的skill确实被管了, 不让说(怕被你们蒸馏, 笑~)....能说的等有一些结果或者一些事情公开了再放出来吧...

话说, 前面这么搞还只是RTX Pro, 要是哪天让我这种人碰到B卡, 老黄是不是又要注意到我了....

参考资料

[1] 
SGLang Office Hour on Agent Loops: *https://www.linkedin.com/events/7465511510824554496/*
[2] 
Kernel Design Agents: *https://github.com/mit-han-lab/kernel-design-agents*
[3] 
从高中数学到代数几何: *https://zartbot.github.io/alegbra_geometry_tutorials/*