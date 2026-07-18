# 谈谈Kimi K2, 第一个开源的1T参数模型

> 作者: zartbot  
> 日期: 2025年7月11日 23:15  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247494374&idx=1&sn=6ab47abdf6256c1484a06025bfea910c&chksm=f995fa24cee273328712fdcef965968ba4f57ba4a7c55c80db63bf060237be5e9937062faa24#rd

---

### TL;DR

最近一直在做一些Agent相关的开发, 测试了若干模型效果都很一般, 感觉就像写了一个遍地都是野指针的程序, 运行时各种出错, 然后不停的修bug. 正好昨天下午一个朋友跟我说, 赶紧试一下马上要发布的Kimi K2, 于是把手上正在开发的几个任务换成K2测试了一下, 结果还挺不错的, 发了一个朋友圈, 然后章老师让我锐评一下, 这不就熬夜交作业了么....

晚上Kimi将它开源了, 但是正式的Technical Report还没出来. 先根据已知的一些信息做点分析吧. 从Kimi的一个页面Kimi K2: Open Agentic Intelligence[1]来看, Agentic/Coding的能力是这个模型非常重要的一环, 一些测试结果也非常不错, 基本上和Anthropic在同一水平线上了....

![图片](assets/db9a06043462.png)

特别的来说, Kimi K2还兼容了Anthropic的API, 因此可以做到Claude Code采用K2的平替...

![图片](assets/6f6efddd861f.png)

### 1. 模型结构

Huggingface上的开源连接里有它的一些信息config.json[2], hidden_dim和DSv3一样是7168, 然后从模型的这些参数来看使用了MLA, MoE结构也类似于DSv3, 只是专家数增加到了384, 激活专家还是8个, 但是专家并没有像DSv3那样分Group. 然后模型层数也是61层, 仅第一层为Dense.

```
  "hidden_size": 7168,  "initializer_range": 0.02,  "intermediate_size": 18432,  "kv_lora_rank": 512,  "max_position_embeddings": 131072,  "model_type": "kimi_k2",  "moe_intermediate_size": 2048,  "moe_layer_freq": 1,  "n_group": 1,  "n_routed_experts": 384,  "n_shared_experts": 1,  "norm_topk_prob": true,  "num_attention_heads": 64,  "num_experts_per_tok": 8,  "num_hidden_layers": 61,  "num_key_value_heads": 64,  "num_nextn_predict_layers": 0,  "pretraining_tp": 1,  "q_lora_rank": 1536,  "qk_nope_head_dim": 128,  "qk_rope_head_dim": 64,
```

#### 1.1 关于MLA

苏剑林老师在最近两个月写了两篇关于MLA的文章《Transformer升级之路：20、MLA好在哪里?（上）》, 《Transformer升级之路：21、MLA好在哪里?（下）》[3]关于为什么选择MLA, 其中有一段话

![图片](assets/82673beccf23.png)

但是这个模型上没有使用MOBA, 但是我还是挺希望能够有更长的context能力. 昨天晚上测试agent browser-use的时候, 在让K2分析Grok4的一些性能时, 就频繁的遇到context打满128K的情况. 根据这段时间写Agent的一些经验, 感觉Context长度要接近500K才够用, 当然后面我也会做一些Context Engineering的工作.

#### 1.2 关于MoE

和DeepSeek-V3的差别是没有选择使用MoE Group, 然后Expert的数量增加到了384. 在DeepSeek-V3采用了专家分组, 但是和Device-Limited Routing不同,主要是用于NVLINK和IB的带宽3.2x配比上. 当然Kimi可以用合规的说法使用了更大显存更大NVLink带宽的卡来训练, 例如H20-3E 141GB的显存容量和900GB的NVLink. 例如采用EP96或者EP48来训练? 不确定他们是否加了一些device-limit相关的辅助损失函数?

这个疑问要等到technical report出来.

### 2. 一些训练相关的手段

#### 2.1 MuonClip优化器

其实在几个月前就有一篇论文《MUON IS SCALABLE FOR LLM TRAINING》[4], 当时Kimi就以DeepSeek-V3-Small的结构做了一些关于Muon优化器的一些探索. 这个工作出来的时候, 还写了一篇文章

[《谈谈Kimi的LLM相关的工作, K1.5/MOBA/MUON》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493252&idx=1&sn=693a78f3ba99657a35b6c43102121730&scene=21#wechat_redirect)

其实去年10月我也在内部给一些同事建议过使用MUON, 但是好像没人关注

![图片](assets/50485e531f9a.png)

而这一次在Kimi进一步在MUON上做了一些优化构成了MUONClip工作(大概率是苏老师的杰作吧?), 也算是第一次证明了它能够在1T这样的大模型上使用, 并且效果优秀

![图片](assets/fb97daf6f09f.png)

#### 2.2 Agentic

Kimi讲得很清楚

![图片](assets/b97b83566e02.png)

Kimi采用了大量的Agentic合成数据和通用的强化学习手段来实现.

![图片](assets/fa284c579ffc.png)

当然还有一些self-judge相关的内容, 并且采用了可验证reward的on-policy rollout来更新Critic, 另外还有一篇论文最近也在看Bridging Offline and Online Reinforcement Learning for LLMs[5]

这也是一个很有意思的话题, 等官方的Technical Report出来后再详细学习

### 3. 一些测试任务

官方的文章:Kimi K2 发布并开源，擅长代码与 Agentic 任务[6]里面已经有一些Demo了, 然后我顺手也把正在开发的一些基于Google ADK的agent和MCP小工具做了一些适配.

例如让K2根据公司公告信息进行分析评分(股票代码是随手想到的, 没有别的意思.....)

![图片](assets/4bd3166dba00.jpg)

然后就是Browser-Use:

![图片](assets/f69eb0759fc0.jpg)

可以看到它先搜CIPU搜出来是什么 游泳池泵的CIPU, 然后自我更正为阿里云CIPU.... 同时搜索BlueField3也是找到了官方的Datasheet的PDF进行的分析...

如果我Prompt不指定用Google搜索, 而直接说阿里云CIPU, K2会非常智能的去阿里云官网上查询CIPU的信息, 而如果让他搜索股票, 它也会去东财查询, 在这些细节任务上, 感觉Kimi用心做了很多合成数据... 另一个案例是让它分析淘宝闪购和美团京东的竞争, 整体执行也非常顺畅, 结果也基本能用

![图片](assets/036c41548540.jpg)

网站上还有一些其他的Benchmark

![图片](assets/da7779b380b4.jpg)

### 4. 一些总结

整体来看, 代码能力还没有测试, 但Agent的能力上是让我觉得非常满意的, 过去的一周一直在调各种模型, 公司的网络连接OpenRouter或者其它家海外模型可靠性有点问题,任务一直中断. 而Kimi K2使用下来, 由于训练过程中使用了大量的合成数据, 效果非常不错. 当然也有一些还需要未来改进的地方, 那就是context长度还有点不够

例如有一个任务在分析Grok4性能时, 网站数据特别多

![图片](assets/94d21b18ebb6.png)

然后Kimi就很容易超过128K Context出现错误

![图片](assets/8c918146f255.png)

这些是未来值得改进的地方. 例如看到MCP工具获取大量的数据后, 能否进行滑动窗口那样的分段进行处理? 我会在Context Engineering中进行一些分析, 看能不能绕开Context长度的限制.

就用几个月前《谈谈Kimi的LLM相关的工作, K1.5/MOBA/MUON》的开头和结尾做个总结吧:

![图片](assets/fb4df4735562.png)

![图片](assets/239da6322c28.png)

参考资料

[1] 
Kimi K2: Open Agentic Intelligence: *https://moonshotai.github.io/Kimi-K2/*
[2] 
k2 config.json: *https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json*
[3] 
Transformer升级之路：21、MLA好在哪里?（下: *https://spaces.ac.cn/archives/11111*
[4] 
MUON IS SCALABLE FOR LLM TRAINING: *https://github.com/MoonshotAI/Moonlight/blob/master/Moonlight.pdf*
[5] 
Bridging Offline and Online Reinforcement Learning for LLMs: *https://arxiv.org/html/2506.21495v1*
[6] 
Kimi K2 发布并开源，擅长代码与 Agentic 任务: *https://mp.weixin.qq.com/s/2RPmHf_8KqIjXbY5jLdztQ*