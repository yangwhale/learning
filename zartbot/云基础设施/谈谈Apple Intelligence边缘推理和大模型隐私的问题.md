# 谈谈Apple Intelligence边缘推理和大模型隐私的问题

> 作者: zartbot  
> 日期: 2024年6月13日 23:34  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247490256&idx=1&sn=f5333e54e3e95f79877cdee7d6ad90c3&chksm=f9960a12cee18304d84d21a17cd7196c9ffe2ac46338a2f109feb5ee37d7bc9177ba7a63eba3#rd

---

对于Apple WDCC的更新虽然不是很惊艳, 也没有什么新奇的故事. 但是至少AI的故事让还在服役的iPhone7 ~ iPhone14都有了换机的冲动, 所以市值上涨也不足为奇了. 当然我们关注的更多的是背后的技术, 例如`边缘推理是如何实现的, 用户隐私怎么保护的`. 其实这个问题非常关键, 你看Musk对当前的边缘推理和Private Cloud Compute并没有反对, 只是反对OAI**在OS Level调用API. 另一个层面来看,`本质上Apple已经构建好了一个LLM OS的雏形`.

![图片](assets/12a4f3988496.png)

那么我们就来详细探索一下Apple边缘推理的实现,然后我在五六年前也因为GDPR**等合规需求,当时在Cisco也实现了一套边缘推理框架, 因此后面会给一点我个人的经验和建议, 特别是云端推理如何使用公有云的问题以及端侧模型个性化的问题, 最后留了一个彩蛋~嘻嘻~

```
1. Apple 边缘推理框架1.1 端侧模型1.2 服务器端模型1.3 模型优化1.4 个性化FineTune2. 端侧推理优化2.1 Flash & DRAM2.2 减少数据传输量2.3 提高传输吞吐量3. 模型安全和私有云计算(PCC)3.1 硬件架构和运行环境3.2 数据安全4.未来展望
```

## 1. Apple 边缘推理框架

### 1.1 端侧模型

《OpenELM: An Efficient Language Model Family with Open Training and Inference Framework》[1]这篇论文来看, 基本上就是一个很普通的LLM, Pre-Norm,Rope,SwiGLU, GQA,flashattention这些都有, 然后有一个小的创新点是Layer-wised Scaling:

![图片](assets/42e842ed5ebc.png)

这样缩放看似很巧妙,但实际上这样的结构会导致的问题是压缩更厉害,可以参考Transformer need glasses, 相当于在较前面的layer态射的能力较弱, 而运算中一些数值误差在后级放大.

另一个模型是跟UI相关的多模态《Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs》[2]让大模型理解屏幕上UI的组建和相应的用户操作顺序.

![图片](assets/761bc0abd789.png)

Ferret-UI 基于Ferret[3],Ferret是一个用于理解图像内任何形状和粒度内的细粒度的空间理解, 并准确的建立开放的词汇描述, 很有可能IPad新的计算器也使用了这个模型

![图片](assets/a63a7ef12879.png)

而Ferret-UI则是把一些UI操作的序列采用低分辩率图像用于快速推理,最后产生按键相关的行为

![图片](assets/3654916b922c.png)
例如查找Widget/icon/List等元素

![图片](assets/16523fe9b06b.png)

相当于原来大家用起来很不方便的“捷径(Shortcuts)”应用通过多模态大模型处理了, 这也是新版本Siri**的核心

![图片](assets/8a91b9e92ab3.png)

### 1.2 服务器模型

然后从Apple最近的一篇文章《Introducing Apple’s On-Device and Server Foundation Models》[4]来看端侧模型词表是49K,而服务器端是100K

然后充分强调了全链路的隐私和模型安全的问题

![图片](assets/cc4260bf6c0e.png)

框架是一个基于JAX/XLA的自研框架,估计是为了和各种算力匹配,以及和自己的服务器芯片匹配, 预训练基本上常见的DP/TP/SP,还有FSDP这些优化都有,就不展开了. 后面的对齐任务就大概讲了一下.

### 1.3 模型优化

主要是针对推理的, 使用了混合2-bit/4-bit以及LoRA的策略.然后比较有趣的事他们开发的延迟和功耗评估分析工具Talaria[5]

基本任务评估看上去还行?

![图片](assets/cacd547437dc.png)

但是Server模型就差一些了, 也难怪要用OAI来补位
![图片](assets/3131dd231b60.png)

![图片](assets/54e59ad02fc4.png)

### 1.4 个性化FineTune

这是一个很有意思的功能

![图片](assets/bbdc9a1ca571.png)

针对日常的用户行为构建了一个小的神经网络可以进行微调

通过仅微调适配器层，基本预训练模型的原始参数保持不变，保留模型的一般知识，同时定制适配器层以支持特定任务。

## 2. 端侧推理优化

端侧推理优化来自于论文《LLM in a flash: Efficient Large Language Model Inference with Limited Memory》[6]

### 2.1 Flash & DRAM

端侧的统一内存架构如下:

![图片](assets/272ffee18cf3.png)

Flash的容量大带宽低, 片上GPU和CPU互联带宽也不是很高, DRAM**的大小也只有10GB. 因此模型无法全部加载到内存中,还要考虑一些功耗的问题.

### 2.2 减少数据传输量

首先是将Attention和Embedding的参数常驻内存中, 这就是Selective Persistence Strategy, 然后MLP层的处理来自于论文《Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time》[7]的思路,进行稀疏性预测剪枝

![图片](assets/6213bf6795af.png)

然后就是滑动窗口的方法

![图片](assets/769a8aef1505.png)

### 2.3 提高传输吞吐量

存储Layout上进行优化,提高读取吞吐

![图片](assets/e68cb0ba3ad2.png)

然后就是一些主动的内存管理

![图片](assets/a0164810c3f8.png)

## 3. 模型安全和私有云计算(PCC**)

### 3.1 硬件架构和运行环境

PCC采用了Apple自定义的芯片, 联系到训练用的JAX/XLA框架来看, 应该是整个训推一体到端侧完全统一了软件栈. 然后和iPhone一样支持Secure Enclave 和 Secure Boot这些安全技术, 同时针对大模型推理工作对操作系统也做了增强

### 3.2 数据安全

采用了完全无状态的计算模式, 当然不要因为一个无状态又和Serverless容器啥的扯上了, 本质上是数据处理完了就删除. 然后苹果对于如何运营一个Cloud写了很多, 例如如何避免特权, 避免各种日志, 避免用户数据泄漏等, 发布每个生产 PCC 版本的二进制映像的同时，为了进一步帮助研究，还会定期发布安全关键的 PCC 源代码的子集. 还有相应的安全赏金计划.

## 4. 未来展望

虽然模型本身和相应的功能不咋地,但是苹果整套端云协同的方案是值得学习的, 同时针对每个用户的一些习惯构建的adpater小型的可以FineTune的神经网络也是值得关注的.

这也是我做前面一篇《谈谈大模型的可解释性》里提到的一个思路, 要维持主模型的数据通路上不修改参数, FineTune需要在旁路上进行, 当时是想从一些用户喜好上对于SAE的一些特征进行强化来处理的, 当然苹果的一些操作的任务可能还有一些不同, 例如如何让大模型记住一个“捷径”而不用让终端用户去复杂配置这些的行为FineTune要和Ferret-UI结合还是有些工作的, 大概想到的一种算法是一些用户行为时间序列的拟合, 通过旁路的神经网络来修改反馈给主网络的Attention?

另一个值得考虑的问题是, 苹果这套系统要在国内运营, 前期似乎有些传闻出来会用某家的大模型, 但是针对PCC这些,苹果在国内是否能够放置相应的服务器呢?

然后我们再切过来,对于安卓生态怎么处理呢? 可能这是国内一众手机厂商正在努力追赶的一个赛道吧? 这里公有云应该可以承接一些针对每个客户的端侧模型FineTune的需求,以及更大规模Server端模型的推理需求.

最后留个视频, 听听TTS的声音~

      
     
       
         
           
             
                                

                 
                   
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
   
 

参考资料

[1]
OpenELM: An Efficient Language Model Family with Open Training and Inference Framework: https://machinelearning.apple.com/research/openelm
[2]
Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs: https://arxiv.org/abs/2404.05719
[3]
Ferret: https://machinelearning.apple.com/research/ferret
[4]
Introducing Apple’s On-Device and Server Foundation Models: https://machinelearning.apple.com/research/introducing-apple-foundation-models
[5]
Talaria: https://machinelearning.apple.com/research/talaria
[6]
LLM in a flash: Efficient Large Language Model Inference with Limited Memory: https://arxiv.org/abs/2312.11514
[7]
Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time: https://arxiv.org/pdf/2310.17157