# 详细谈谈DSpark投机解码的原理

> 作者: zartbot  
> 日期: 2026年7月4日 08:25  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247499222&idx=1&sn=f925cf29f57dec4fbc45bc675834e925&chksm=f995ed14cee26402ad8f899f420c00297f6f37b163589af1d73a302a6b318cc51a67a84dbc80#rd

---

`渣b又一夜返贫, 回到没卡没token的状态, 那就继续读点论文搞点算法吧`

### TL;DR

DSpark[1]是DeepSeek刚开源的一个投机解码**(Speculative Decoding)框架. 论文的核心目标是解决当前投机解码技术中的两个主要瓶颈:**草稿质量衰减**和**验证效率低下**. 简单来说, 当前的draft model虽然能快速生成一长串候选token, 但由于缺乏上下文依赖, 导致草稿越往后, 质量越差, 接受率迅速下降. 同时, 无论草稿质量如何, 盲目地将整个长草稿交给目标模型进行验证, 会在大概率被拒绝的token上浪费宝贵的计算资源.

针对这两个问题, DSpark提出了两个互补的创新机制:

**半自回归生成 (Semi-Autoregressive Generation):** 为了解决草稿质量问题, DSpark设计了一种混合架构. 它保留了一个计算量大的**并行主干网络**, 用于一次性快速生成所有草稿token的"骨架"; 然后在此基础上增加了一个轻量级的**串行模块**(Sequential Head), 在生成草稿时按顺序注入局部依赖信息. 这种设计既保留了并行生成的高速度, 又通过引入token间的关联性显著改善了草稿后半部分的质量, 减缓了接受率的衰减.

**置信度调度验证 (Confidence-Scheduled Verification):** 为了解决验证效率问题, DSpark引入了一个动态调度机制. 它训练了一个**置信度头(Confidence Head)** 来预测每个草稿词元被接受的概率. 然后, 一个**硬件感知调度器(Hardware-Aware Scheduler)** 会结合这些置信度分数和系统当前的实时负载(如GPU吞吐量)来动态决定每个请求应该验证多长的草稿. 当系统负载低时, 可以验证更长的草稿以追求更高的单用户速度; 当系统负载高时, 则会自动剪裁掉那些低置信度的草稿后缀, 避免在"垃圾"词元上浪费算力, 从而保证整个系统的吞吐量稳定.

接下来我们先从投机解码的原理谈起, 为什么一个草稿模型投机解码的结果和目标模型推理的结果是等价的. 然后第一章对一些典型的投机解码模型的演进做一个回顾. 再详细分析DSpark, 本文目录如下

```
0. 什么是投机解码?0.1 从一个通俗的故事讲起0.2 投机解码的数学原理0.3 代码实现0.4 核心参数三元组1. Draft Model算法介绍1.1 自回归投机解码1.1.1 Speculative Decoding1.1.2 Speculative Sampling1.1.3 BiLD1.1.4 Self-Speculative1.2 树状投机解码1.2.1 SpecInfer1.3 多头/特征层1.3.1 Medusa1.3.2 Lookahead1.3.3 EAGLE1.3.3.1 EAGLE-11.3.3.2 EAGLE-21.3.3.3 EAGLE-31.4 MTP训练集成1.4.1 Meta MTP1.4.2 DeepSeek-V3 MTP1.5 Dflash1.5.1 关键创新1.5.2 详细的推理过程1.5.3 详细的训练过程1.5.4 加速比分析与性能评估1.6 DDTree1.7 JetSpec1.7.1 构建树形因果性1.7.2 推理流程1.7.3 JetSpec训练2. DSpark2.1 半自回归生成 (Semi-Autoregressive Generation)2.2 置信度调度验证 (Confidence-Scheduled Verification)2.3 训练2.4 实验2.4.1 为什么并行生成能超越自回归?2.4.2 一点点自回归大有裨益2.4.3 置信度头的作用 2.5 在线服务2.5.1 可扩展和灵活的训练2.5.2 实践中的硬件感知前缀调度器2.5.3 高吞吐和低延迟推理2.5.4 线上用户流量下的性能 3. 结语3.1 投机解码的性能模型3.2 投机解码的数学模型3.3 调度策略设计考量附录 A: 投机解码发展史 — 时间维度的全景分析附录 B: DSpark训练代码解析
```

## 0. 什么是投机解码?

### 0.1 从一个通俗的故事讲起

想象一下你正在和一位学识渊博但打字很慢的专家(比如一位老教授)在线聊天. 这位老教授就是我们的**目标模型 (Target Model,)**, 比如GPT-4. 它的特点是:

**权威**: 每个字都深思熟虑, 绝对正确.

**缓慢**: 生成每个字都需要消耗大量时间.

你问了一个问题, 然后眼巴巴地看着他一个字一个字地往外蹦. 为了加速这个过程, 我们给他配一个反应飞快的博士生助手. 这个助手就是**草稿模型 (Draft Model,)**, 比如一个规模小得多的语言模型. 它的特点是:

**敏捷**: 打字飞快, 能瞬间生成一大段话.

**不靠谱**: 知识储备不如教授, 经常会猜错或说错话.

![图片](assets/e1a51ed229a6.jpg)

*投机解码(Speculative Decoding)就是这套"教授+学生"的协同工作流程:*

**教授起草**: 教授 () 先写下第一个字, 比如"今天天气...".

**学生抢答 (投机)**: 博士生 () 看到开头, 立刻"投机性地"猜出后面可能的内容, "唰唰唰"地生成一串草稿, 比如"**真不错, 阳光明媚, 万里无云**".

**教授批改 (验证)**: 教授 () 拿到这份草稿. 他不用自己一个字一个字地想了, 只需要批改就行. 他把学生的草稿和自己脑子里想说的话, 在一次思考中(一次前向传播)进行并行对比.

他看了看"真", 心想: "嗯, 我也想说'真', 通过."

他看了看"不错", 心想: "嗯, 我也想说'不错', 通过."

他看了看"阳光明媚", 心想: "不对, 我想说的是'有点阴沉'".**在这里, 出现了第一个不一致.**

**接受并修正**:

教授会接受学生猜对的所有部分, 即"**真不错**".

在第一个猜错的地方, 他会亲自写下正确的词, 即"**有点阴沉**".

对于学生草稿中剩下的部分("阳光明媚, 万里无云"), 则全部丢弃.

**完成一轮**: 这一轮解码, 系统最终输出了"**真不错 有点阴沉**"这三个词. 但请注意, 教授只亲自"思考"了一次(用于并行验证和生成修正词), 却得到了多个词的输出. 相比于他自己一个字一个字地思考三次, 速度大大提升了.

**循环往复**: 接下来, 学生会根据新的结尾"…有点阴沉", 再次进行抢答, 开始新一轮的投机.

**核心思想**: 用一个廉价、快速的草稿模型的大量猜测, 来换取昂贵、缓慢的目标模型少量但高效的验证. 只要草稿模型猜得越准, 整体的加速效果就越好.

### 0.2 投机解码的数学原理

具体算法参考论文《Accelerating Large Language Model Decoding with Speculative Sampling》[2]
基本设定
*目标模型*: 在给定前文 $x_{<t}$ 的情况下, 对下一个 token 的输出概率分布为 $p_{\text{target}}(x_t\mid x_{<t})$. 我们简写为 $p(x)$.

*草稿模型*: 在同样的前文下, 输出概率分布为 $q_{\text{draft}}(x_t\mid x_{<t})$. 我们简写为 $q(x)$.

我们的目标是, 以某种方式从 $p(x)$ 中采样一个 token, 但要尽量避免直接计算和调用昂贵的 $p(x)$. 这背后的数学原理是

**拒绝采样(Rejection Sampling)**
假设我们想从一个目标分布 $p(x)$ 中采样, 但直接采样很困难. 不过, 我们可以很容易地从另一个提议分布 $q(x)$ 中采样, 并且存在一个常数 $M$, 使得对于所有的 $x$, 都有 $p(x)\le M\,q(x)$. 拒绝采样的步骤如下:

![图片](assets/b882a22d0ae6.png)

1. 从提议分布 $q(x)$ 中采样一个候选值 $x$.

2. 从均匀分布 $u\sim U(0,1)$ 中采样一个随机数.

3. 如果满足如下条件, 则接受样本 $x$:

$$u \le \frac{p(x)}{M\,q(x)}$$

否则, 拒绝, 回到第1步.

可以证明, 通过这个过程接受的样本, 其分布*恰好是*目标分布 $p(x)$.

在投机解码中, 我们将这个思想应用到序列生成上. 假设草稿模型生成了 $\gamma$ 个 token 的序列 $x_1,\dots,x_\gamma$. 目标模型需要对它们进行验证. 验证过程从左到右 (对于 $i=1,\dots,\gamma$):

![图片](assets/1708dfc03492.jpg)

草稿模型在第 $i$ 步生成了 token $x_i$, 其概率为 $q(x_i)$.

目标模型在第 $i$ 步如果自己生成, 会以概率

$$p(x_i)$$

生成 token.

我们进行一次"抛硬币"实验. 以概率

$$\min\left(1,\ \frac{p(x_i)}{q(x_i)}\right)$$

接受 token $x_i$.

如果*接受*, 我们就继续验证下一个 token.

如果*拒绝*, 那么验证过程立刻停止. 我们不仅丢弃, 还丢弃它之后的所有草稿 token.
**为什么这个接受概率是 $\min\left(1,\frac{p(x)}{q(x)}\right)$?**
标准的拒绝采样**前提条件**是找到常数 $M$, 使得对所有 $x$ 都有:

$$p(x)\le M\,q(x)$$

即 $M\,q(x)$ 是 $p(x)$ 的一个**上包络**. 关键在于: **在候选点 $x$ 处, 接受的概率与 $\frac{p(x)}{M\,q(x)}$ 成正比**.

![图片](assets/ffc92bf1cc43.jpg)

$M=\max_x \frac{p(x)}{q(x)}$, 所以 **q 越接近 p, M 越小, 效率越高**

这可以看作是拒绝采样的一个变种, 在论文中普遍也被称为*投机采样(Speculative Sampling)*. 在实践中, 我们通常假设草稿模型的分布不会与目标模型差得太离谱. 如果碰巧在某个点 $p(x_i)\ge q(x_i)$, 我们就直接以概率1接受它, 这种情况非常有利. 如果 $p(x_i)<q(x_i)$, 那么接受的概率就是比率 $\frac{p(x_i)}{q(x_i)}$. 这两种情况合并起来, 就是 $\min\left(1,\frac{p(x_i)}{q(x_i)}\right)$.

**恢复目标模型分布证明**
给定离散分布 $p$, $q$ 和一个从 $q$ 中采样的草稿样本 $\tilde{x}$, 令 $X$ 为最终得到的样本. $X=x$ 为真的情况有两种: 我们要么采样得到 $\tilde{x}=x$ 并接受它, 要么在 $\tilde{x}$ (无论其值为何)被拒绝后重新采样得到它. 因此:

$$\mathbb{P}(X=x)=\mathbb{P}(\tilde{x}=x)\mathbb{P}(\tilde{x}\text{ accepted}\mid\tilde{x}=x)+\mathbb{P}(\tilde{x}\text{ rejected})\mathbb{P}(X=x\mid\tilde{x}\text{ rejected})$$

对于第一项, 我们应用接受规则:

$$\mathbb{P}(\tilde{x}=x)\mathbb{P}(\tilde{x}\text{ accepted}\mid\tilde{x}=x)=q(x)\min\left(1,\frac{p(x)}{q(x)}\right)=\min(p(x),q(x))$$

对于第二项的条件部分, 我们应用重采样规则:

$$\mathbb{P}(X=x\mid\tilde{x}\text{ rejected})=(q(x)-p(x))_+$$

其中 $(\cdot)_+$ 表示:

$$(f(x))_+=\frac{\max(0,f(x))}{\sum_{x'}\max(0,f(x'))}$$

最后, 我们计算拒绝的概率:

$$
\begin{aligned}\mathbb{P}(\tilde{x}\text{ rejected})&=1-\mathbb{P}(\tilde{x}\text{ accepted})\\
&=1-\sum_{x'}\mathbb{P}(X=x',\tilde{x}\text{ accepted})\\
&=1-\sum_{x'}\min(p(x'),q(x'))\\
&=\sum_{x'}q(x')-\sum_{x'}\min(p(x'),q(x'))\\
&=\sum_{x'}\max(0,q(x')-p(x'))\end{aligned}
$$

这等于 $(q(x)-p(x))_+$ 的分母, 所以:

$$\mathbb{P}(\tilde{x}\text{ rejected})\mathbb{P}(X=x\mid\tilde{x}\text{ rejected})=\max(0,q(x)-p(x))$$

因此:

$$\mathbb{P}(X=x)=\min(p(x),q(x))+\max(0,q(x)-p(x))=p(x)$$

我们已经恢复了期望的目标分布.
**渣注**
看上去证明很复杂, 我来说人话, 大致的证明思路是: 任何一个token $x$ 最终被我们选中, 只有两条路可以走. 我们把这两条路的概率加起来, 看看最终是不是等于"教授"本来想选 $x$ 的概率.

**路径一: "英雄所见略同" (草稿被接受)**

首先, 学生写 $x$ 的概率是 $q(x)$.

然后, 教授认可的概率是 $\min\left(1,\frac{p(x)}{q(x)}\right)$.

两者相乘, 这一路径的总概率是 $q(x)\cdot\min\left(1,\frac{p(x)}{q(x)}\right)$, 化简后就是 $\min(p(x),q(x))$. 这代表了学生和教授的 "共识" 部分.

**情况**: 学生的草稿刚好写了 $x$, 而且教授也认可了.

**概率计算**: $\min(p(x),q(x))$

**路径二: "另请高明" (草稿被拒绝, 重新采样)**

首先, 我们需要知道 "发生拒绝" 这件事本身的总概率. 证明中通过一系列计算得出, 这个总概率等于所有 $\sum_{x'}\max(0,q(x')-p(x'))$. 这部分代表了教授的想法超出学生认知的所有 "概率溢出" 的总量.

然后, 在发生拒绝后, 教授会从一个 "修正列表" $(q(x)-p(x))_+$ 中挑选. $x$ 被挑中的条件概率就是 $\dfrac{\max(0,p(x)-q(x))}{\sum_{x'}\max(0,p(x')-q(x'))}$

两者相乘, `(拒绝总概率) * (拒绝后选中x的条件概率)`, 分子分母正好抵消, 最终这条路径的总概率就是 $\max(0,p(x)-q(x))$. 这精确地代表了 $x$ 这个 token 上, 教授比学生更偏好的那部分 "独特想法".

**情况**: 学生写的任何草稿被教授拒绝了, 然后在补救环节, 教授亲自挑选出了 $x$.

**概率计算**: $\max(0,p(x)-q(x))$

**最终汇总**:

$$\min(p(x),q(x))+\max(0,p(x)-q(x))$$

这个表达式恒等于 $p(x)$!

我们把两条路径的概率加起来:`路径一概率 + 路径二概率`

**结论**: 账平了! 无论通过哪条路径, 最终选中 $x$ 的总概率不多不少, 正好就是教授最初想选 $x$ 的概率. 证明完毕.

### 0.3 代码实现

例如在DSpark代码中`deepspec/eval/base_evaluator.py`中有一个常见的验证函数`verify_draft_tokens`, 它使用 target model 对 draft model 提出的候选 token 进行验证, 算法流程:

将 draft tokens 序列一次性送入 target model 做并行前向传播, 得到每个位置的概率分布.

对每个 draft token 位置, 计算接受概率`min(1, p_target(x)/q_draft(x))`,通过随机数与该概率比较决定是否接受.

使用 cumprod 实现前缀截断语义: 一旦某个位置被拒绝, 后续所有位置均被拒绝.

若存在被拒绝的 token, 从残差分布`norm(max(0, p_target - q_draft))` 中采样新 token;若全部接受, 则从 target model 最后位置的分布中采样一个 bonus token.

拼接被接受的 draft tokens + 新采样的 token 作为最终输出.

首先是一些合法性检查, 需要保证验证输入`verify_input_ids`的第一个token必须是当前已接受的 token. 然后是第一步

```
    # ==================== Step 1: Target Model 并行前向传播 ====================    # verify_input_ids 的结构: [当前token, draft_token_1, ..., draft_token_k]    # 长度为 draft_token_count + 1（包含当前 token 本身）    draft_token_count = int(proposal.draft_token_count)    verify_length = draft_token_count + 1  # +1 是因为包含了当前已接受的 token    verify_position_ids = position_ids[:, start : start + verify_length]    # 将候选序列一次性送入 target model    target_output = target_model(        input_ids=proposal.verify_input_ids,        position_ids=verify_position_ids,        past_key_values=past_key_values_target,        use_cache=True,        output_hidden_states=True,    )    ...    # 将 logits 转换为概率分布 p_target(x), shape: [B, verify_length, V]    target_probs = logits_to_probs(target_output.logits, float(temperature))
```

第二步就是最关键的投机采样了

```
    # ==================== Step 2: Speculative Sampling ====================    accept_prefix_mask = None    if draft_token_count > 0:        assert proposal.draft_probs is not None        # proposed_tokens: draft model 提出的 token 序列, shape: [B, draft_token_count]        # verify_input_ids = [current_token, t1, t2, ..., tk], 取 [1:] 得到 draft tokens        proposed_tokens = proposal.verify_input_ids[:, 1:]        # 从 target_probs 中取出对应 draft token 位置的概率        # target_probs[:, :-1, :] 对应位置 0..k-1 预测位置 1..k 的概率        # 即 p_target(t_i | prefix) 对每个 draft token t_i        selected_target_probs = gather_token_probs(            target_probs[:, :-1, :],            proposed_tokens,        )        # 从 draft_probs 中取出对应 draft token 的概率 q_draft(t_i)        # clamp_min 防止除零        selected_draft_probs = gather_token_probs(            proposal.draft_probs,            proposed_tokens,        ).clamp_min(1e-8)        # 计算接受概率: accept_prob = min(1, p_target(x) / q_draft(x))        # shape: [B, draft_token_count]        accept_prob = torch.clamp(            selected_target_probs / selected_draft_probs,            max=1.0,        )        # 对每个位置生成均匀随机数 u ~ U(0,1), 若 u < accept_prob 则接受        accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)        # cumprod 实现前缀截断：一旦某位置被拒绝(=0), 后续所有位置都变为 0        # 这保证了 speculative decoding 的「前缀接受」语义        accept_prefix_mask = accept_mask.cumprod(dim=1)        # 被接受的 draft token 总数 = prefix mask 中 1 的个数        accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())    else:        # 没有 draft token 时, 直接跳过验证        accepted_draft_tokens = 0
```

我们注意到整个投机采样过程并不要求Draft Model是什么, 但是我们还是期望尽量让Draft Model输出的分布和Target Model接近. 接下来我们就进入下一个环节, 来介绍一些Draft Model的算法.

### 0.4 核心参数三元组

Leviathan et al. 的另一重大贡献是建立了完整的理论分析框架, 揭示投机解码的性能被三个核心参数完全支配:

**接受率** $\alpha$ 衡量"草稿有多靠谱". 形式上 $\alpha=1-D_{TV}(p,q)$, 即目标分布 $p$ 与草稿分布 $q$ 之间全变差距离(Total Variation Distance)的互补. 直觉上, $\alpha$ 等于两个分布的"重叠面积", $\alpha$ 越大说明草稿越接近目标.

**成本比** $c$ 衡量"起草有多便宜". $c=T_{\text{draft}}/T_{\text{target}}$, 即草稿模型单步推理时间与目标模型的比值. 理想情况下 $c\to0$, 但能保证合理接受率的草稿模型通常不会太小, 实践中 $c$ 在0.05-0.2之间.

**草稿长度** $\gamma$ 衡量"每轮猜多少个". $\gamma$ 太小则投机收益有限; $\gamma$ 太大则后面token的接受概率指数下降(连续 $\gamma$ 次全部猜对的概率为 $\alpha^\gamma$), 反而浪费起草成本.
**加速比的精确公式**
三者共同决定了端到端加速比:

$$S=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$$

**分子** $\frac{1-\alpha^{\gamma+1}}{1-\alpha}=\sum_{i=0}^{\gamma}\alpha^i$ 是每轮期望接受的token数. 这是一个截断几何级数, 完全由草稿质量 $\alpha$ 决定.

**分母** $\gamma c+1$ 是每轮的时间成本, $\gamma c$ 是起草 $\gamma$ 个token的相对耗时, 1 是验证一次的耗时.

## 1. Draft Model算法介绍

投机解码的Draft Model算法来看, 用AI整理了一个大概的时间线, 给大家阐述这个方向的算法是如何演变的.

![图片](assets/90b540e80846.png)

| 时间 | 论文 | 作者/机构 | 核心贡献 | 方法类别 |
|---|---|---|---|---|
| 2022.11 | Speculative Decoding (Leviathan) | Leviathan et al. (Google) | 奠基性 draft-then-verify 理论框架 | 独立草稿模型 |
| 2023.05 | SpecInfer | Miao et al. (CMU/北大) | 多SSM + 树状推测 + 树验证 | 独立草稿模型+树验证 |
| 2023.05 | BiLD | Kim et al. (UC Berkeley) | 置信度驱动 Fallback/Rollback + 确定性验证 | 独立草稿模型 |
| 2023.09 | Self-Speculative Decoding | Zhang et al. (浙大/UC Irvine) | 跳层自投机, 无需辅助模型 | 自投机 |
| 2023.10 | Online Speculative Decoding | Liu et al. (UC Berkeley) | 在线知识蒸馏持续更新草稿模型 | 蒸馏增强 |
| 2023.10 | DistillSpec | Zhou et al. (Google) | 离线KD对齐Draft与Target分布 | 蒸馏增强 |
| 2023.11 | REST | He et al. (北大/Princeton) | 数据库检索替代Draft模型 | 检索式 |
| 2023.11 | Lookahead Decoding | Fu et al. (UCSD/Google) | Jacobi迭代 + n-gram提取 | Jacobi方法 |
| 2023.11 | Ouroboros | Zhao et al. (清华) | 短语级并行起草 + 短语复用 | Jacobi方法 |
| 2024.01 | EAGLE | Li et al. (北大/微软) | 特征层预测 + 树链条件化 | 多头/特征预测 |
| 2024.02 | Sequoia | Chen et al. (CMU/Together) | DP最优树结构 + 无放回采样验证 | 树状最优验证 |
| 2024.02 | Speculative Streaming | Bhendawade et al. (Apple) | 流融合并生成: 投机与验证合一 | 自投机 |
| 2024.02 | SPACE | Yi et al. (云天励飞) | SAR-SFT + 自验证并行 | 自投机 |
| 2024.04 | ConsistentEE | Zeng et al. (华南理工) | Early Exit + RL难度引导退出 | Early Exit |
| 2024.04 | Multi-Token Prediction (Meta FAIR) | Gloeckle et al. (Meta FAIR) | 多token预测训练内置 + 自投机加速 | 多头/特征预测 |
| 2024.12 | DeepSeek-V3 MTP | DeepSeek-AI | 顺序因果链MTP + 训练增强/推理复用 | 多头/特征预测 |
| 2025 | DFlash (ICML 2026) | Kim et al. (UC San Diego) | 并行扩散起草 + KV注入条件化, O(1)起草延迟 | 扩散适配器 |
| 2026.03 | SSD / Saguaro | Kumar et al. (Stanford/Princeton) | 异步并行调度: 投机缓存消除起草等待 | 异步系统调度 |
| 2026.04 | DDTree | Ringel et al. (Technion) | 扩散marginal构建最优化草稿树 + 树验证 | 扩散+树验证 |
| 2026.06 | JetSpec | Hu et al. (UC San Diego) | 因果并行树掩码, 解决因果-效率困境 | 因果并行起草 |
| 2026.06 | DSpark | Cheng et al. (PKU/DeepSeek-AI) | 半自回归生成+置信度调度验证, 已部署V4 | 半自回归+调度验证 |

但是后面的内容, 篇幅有限, 还是古法挑选几个比较重要的draft model算法写一下吧...

![图片](assets/d1ce25e1f160.png)

### 1.1 自回归投机解码

#### 1.1.1 Speculative Decoding

从2022年底的论文《Fast Inference from Transformers via Speculative Decoding》[3]开始, 整个Speculative Decoding的算法, 都采用一个参数量远小于目标模型的独立自回归模型** 串行生成 个候选token. 它在`T5-XXL (11B)`上使用`T5-small (77M)`作为草稿模型. 比较形象的说法就是*"找个便宜的小弟代写, 错了我来改"*

#### 1.1.2 Speculative Sampling

在2023年《Accelerating Large Language Model Decoding with Speculative Sampling》[4]中提到的DraftModel源于当时的算力问题, Chinchilla是DeepMind**的一个700亿参数的语言模型, 但是由于单卡放不下, 采用了16卡并行推理的策略. 然后他们采用了和Target模型同样的分布式部署, 并构建了一个宽而浅的Draft Model

![图片](assets/518703025569.png)

虽然Hidden-dim Draft Model和Target Model接近, 但是模型深度很浅只有8层, 因此draft-model的延迟会低很多倍. 比较形象的说法是*"小弟要和我用同一间办公室, 身材匹配工位"*.

#### 1.1.3 BiLD

来自2023年的《Speculative Decoding with Big Little Decoder》[5]. 大致的观点是放弃了Speculative Sampling的概率无损的要求, Draft Model(小模型)基于`自身置信度` 决定何时"求助"Target Model(大模型).形式化的描述如下:

在第 $n$ 次解码迭代中, 小模型和大模型各自以部分生成的文本为输入, 生成在整个词汇表上的概率分布 $p_S(y\mid y_{<n})$ 和 $p_L(y\mid y_{<n})$. 然后, 下一个token $y_{S,n}$ 和 $y_{L,n}$ 分别从这些分布中采样:

$$y_{S,n}\sim p_S(y\mid y_{<n})\quad\text{and}\quad y_{L,n}\sim p_L(y\mid y_{<n})$$

根据策略函数 $\pi(y_{<n})$ 的返回值(布尔值), 决定使用哪个模型的预测:

$$
y_n=\begin{cases}y_{S,n},&\text{if }\pi(y_{<n})=0\\
y_{L,n},&\text{if }\pi(y_{<n})=1\end{cases}
$$

目标是设计一个轻量级策略, 使得在最小端到端延迟下获得高质量文本生成, 仅在必要时调用大模型.

Fallback策略: 小模型知道何时停止预测
策略的第一个原则是小模型应能判断何时将控制权交给大模型. 每当小模型对其预测缺乏信心时, 应让大模型接管. 作者发现一个基于最大预测概率的简单策略就足够有效, 即 $\max_y p_S(y\mid y_{<n})$. 如果最大预测概率低于某个阈值 $\alpha_{FB}$, 则认为小模型的预测不够自信, 回退到大模型生成下一个token.

**Fallback策略**: 如果 $\max_y p_S(y\mid y_{<n})<\alpha_{FB}$, 则回退到大模型, 令 $y_n=y_{L,n}$.

![图片](assets/d57e69971eee.png)
Rollback策略: 大模型知道何时回滚预测
Fallback策略虽然允许大模型在小模型不自信时接管, 但小模型仍可能对其错误预测过度自信. 此外, 早期解码步骤的单个错误预测可能导致灾难性后果, 影响所有后续token预测. 在BiLD框架中, 大模型的验证**不产生额外开销**. 当大模型接收小模型生成的token进行非自回归预测时, 它也为所有之前的解码步骤生成了自己的预测. 给定部分生成文本, 大模型对所有先前步骤生成, 这可用于验证小模型之前的预测.

对于某个距离度量 $d(\cdot,\cdot)$, 找到满足以下条件的最小解码步骤 $m$:

$$d(p_S(y\mid y_{<m}),\ p_L(y\mid y_{<m}))>\alpha_{RB}$$

如果这样的 $m$ 存在, 则认为小模型在位置 $m$ 的预测不准确, 回滚从 $m$ 到 $n$ 的所有预测(因为它们都依赖于错误的预测), 并用大模型的预测替换.

**Rollback策略**: 如果存在最小的 $m$ 使得 $d(p_S(y\mid y_{<m}),p_L(y\mid y_{<m}))>\alpha_{RB}$, 则回滚预测并令 $y_m=y_{L,m}$.

总体来看, 和标准的投机解码“严格要求无损”相比, 它允许微小质量损失(<0.5 BLEU). 然后首次引入*动态起草长度*概念, 后续DSpark的置信度调度即源于此思想.

#### 1.1.4 Self-Speculative

论文《Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding》[6], 采用原始模型“跳层”的方式做Drafting, 然后完整的原始模型做Verfication的算法.

但是存在一些挑战, 确定跳哪些层以及跳多少层. 跳太多层会严重损害 draft 质量, 导致接受率低, 增加总推理时间. 跳太少层保证了高接受率, 但限制了最大加速上限.实际上转换成了一个优化问题:

**目标函数**: 接收层选择向量 $z\in\{0,1\}^L$ 作为输入, 返回在开发集上每个 verified token 的平均推理时间 $f(z)$. 这是一个综合度量, 包含了 drafting 和 verifying 两个阶段.

**优化问题**:

$$z^*=\arg\min_z f(z),\quad\text{s.t. }z\in\{0,1\}^L$$

其中 $L$ 是总层数.

然后采用贝叶斯优化算法迭代地基于代理模型 (高斯过程) 和采集函数选择新的输入 $z$ 进行评估. 采集函数平衡了探索 (在不确定区域测试) 和利用 (在预期良好区域测试). 优化持续直到达到预定迭代次数. 搜索得到的 $z^*$ 在模型级别固定, 不再更新. 当目标任务差异很大时, 可以进行任务级优化. 此外, 跳层方案可以与**量化**和**稀疏化**结合, 进一步降低资源消耗.

然后还增加了一个自适应 Draft 退出机制.因为在投机解码中, 如果一个 draft token 被拒绝, 所有后续 draft token 都被丢弃. Draft 退出机制防止将计算资源浪费在不太可能被接受的 draft token 上.

**基本机制**: 比较每个 draft token 的预测概率与阈值. 若, 表明置信度低, 立即停止 drafting.

**自适应阈值更新**: 静态阈值无法准确反映实际接受率. 为此引入动态更新规则:

$$AR=\beta_1 AR+(1-\beta_1)\hat{AR}$$
$$
\gamma=\begin{cases}\gamma-\varepsilon,&\text{if }AR<\gamma\\
\gamma+\varepsilon,&\text{otherwise}\end{cases}
$$

其中:

$\gamma$ 是目标接受率

$\varepsilon$ 是更新步长

$\beta_1$ 是平滑因子, 缓解 $AR$ 和 $\hat{AR}$ 的波动

$\hat{AR}$ 是第 $t$ 次 drafting 阶段的接受率

当 $AR<\gamma$ 时, $\gamma\leftarrow\gamma-\varepsilon$

每次验证阶段后更新. 该更新规则确保接受率维持在目标接受率 $\gamma$ 附近.

### 1.2 树状投机解码

#### 1.2.1 SpecInfer

SpecInfer 不再局限于单一候选序列, 而是同时考虑多条推测候选, 并将它们组织为一棵**token tree**. 树的每个节点代表一个候选token序列. SpecInfer使用**树并行解码机制**, 在单次LLM解码步骤中并行验证token tree中所有候选序列的正确性.

![图片](assets/ad9d4d81f076.png)

SpecInfer的Draft模型没有从头训练一个全新模型, 而是:

直接使用同族预训练小模型, 这些模型使用相同的词表和大量几乎相同的训练数据, 天然具有一定的分布对齐.

Boosting微调, 类似集成学习中的 Boosting——后一个模型专门学习前面模型"做错的题". 这样多个 SSM(Small Speculative Model)组合起来的覆盖率远超单个模型.

然后在Speculative Inference阶段, 先为每个SSM生成一颗输出树, 即在每个token 取topk=5种可能生成的树, 然后将其合并.

![图片](assets/43daa79dd709.png)

然后针对Tree-based结构, 通过修改mask的矩阵方法, 使得模型可以一次验证多个 sequence.

![图片](assets/433652a8d60d.png)

之后作者参考 Speculative Sampling, 提出 Multi-Step Speculative Sampling 的算法, 并且证明了和直接从target model采样的等价性. 并且采样通过率更高:

![图片](assets/c4700db24748.png)

### 1.3 多头/特征层

#### 1.3.1 Medusa

论文《Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads》[7]和SpecInfer类似, 采用Tree Mask的方式, 但是它没有使用独立的小模型, 而是采用在原模型的最后一层的Embedding 后增加多个Medusa Head的方式.

![图片](assets/13cf351e975a.png)
K个MEDUSA头各自输出top-预测, 然后组合各个头预测构建成一颗树, 并通过tree attention mask在一次前向传播中验证所有候选. 然后验证时, 采用Typical Acceptance的方法. 拒绝采样虽然保证了分布一致性, 但在实践中过于保守. Typical Acceptance通过引入基于熵的自适应阈值, 实现了一个优雅的折中:

低熵情况(模型很确定): 阈值高, 只接受高概率token → 类似贪心

高熵情况(模型不确定): 阈值低, 接受更多token → 更宽松

但是它也存在一些问题, 当batchSize增大后, 系统转向Compute-bound可能会导致加速比消失. 但是MEDUSA开创了利用模型自身进行投机预测的研究方向, 后续EAGLE等工作延续此思路

#### 1.3.2 Lookahead

《Break the Sequential Dependency of LLM Inference Using Lookahead Decoding》[8]将自回归解码与**雅可比迭代法**联系起来, 这是一个非常巧妙的数学抽象.

传统的自回归解码可以看作一个函数, 我们一步步求解. 这本质上是一个串行过程.`LOOKAHEAD DECODING`的观点是, 我们可以把整个生成序列看作一个大型非线性方程组的解.

雅可比迭代法的思想是, 对于一个方程组, 我们把它分解为, 然后迭代求解. 关键在于,的每个分量可以**独立地**根据的所有分量来计算.

将这个思想推广到LLM解码, 就得到了**雅可比解码**: 在第`t`次迭代, 我们使用第`t-1`次迭代得到的整个序列来并行计算一个新的序列中的每一个token.

其实这种观点相当于转变为求解一个**不动点方程**:, 其中是整个目标序列,是一个复杂的算子, 其第个分量是.雅可比迭代法正是求解这类不动点问题的标准方法之一:.

但是这个算子的性质很差. 它不是一个**压缩映射**. 在度量空间中, 压缩映射保证了从任何点开始迭代都会收敛到唯一的不动点. 而这里的算子显然不满足这个条件, 它的迭代轨迹可能非常混乱, 甚至形成循环, 无法稳定收敛.

因此作者也指出"位置错误"和"迭代替换"问题, 但是作者在混乱的轨迹中, 发现隐藏着一些局部稳定的, 正确的片段(n-gram).

#### 1.3.3 EAGLE

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）系列有三篇

《EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty》[9]

《EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees》[10]

《EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test》[11]
1.3.3.1 EAGLE-1
EAGLE-1 的核心思想是: 在特征层(而非 token 层)进行自回归预测, 并通过输入"提前一步的 token"来消除采样不确定性.
两个关键洞察
*洞察 1:* LLM 的倒数第二层隐藏状态(即 LM Head 前的特征)是高维连续向量, 比离散 token 更具规律性, 预测起来更容易. 在特征层做自回归再通过 LM Head 转换为 token, 效果优于直接预测 token(加速比从 1.5x 提升到 1.9x).

*洞察 2:* 文本生成中的采样过程引入了随机性, 同一个特征 后面可能采样出 "am" 也可能是 "always", 导致后续特征完全不同. 草稿模型不知道目标模型选了哪个分支, 预测就会模糊. EAGLE 的解法是: 把采样结果(token)作为额外输入告诉草稿模型. 这一步将加速比从 1.9x 提升到 2.8x.

![图片](assets/b6550e2dec39.png)
Draft Model架构
和传统的投机解码方案有一个对比:

![图片](assets/f86ab3b9660f.png)

推测性采样和 Lookahead 基于 token 预测 token. Medusa 使用目标LLM的特征 独立地预测 和. EAGLE 则使用特征序列和一个提前了一个时间步的 token 序列 来预测. 从 中, 采样得到. 随后, 将 和 拼接到输入序列中, 以预测下一个特征 并采样下一个 token.

整个EAGLE-1 Draft Model的算法结构如下:

![图片](assets/5b4ad733cd30.png)

Draft Model由三个模块组成:

Embedding 层(冻结, 复用目标 LLM)

LM Head(冻结, 复用目标 LLM)

Autoregression Head(唯一需要训练的部分) = FC 层 + 单层 Transformer Decoder

推理时的一步操作: 输入特征序列 和提前一步的 token 序列, 拼接后通过 FC 降维(2×hidden_dim → hidden_dim), 再经 Decoder 层预测下一个特征, 最后用 LM Head 将其转为概率分布并采样得到草稿 token. 预测的特征和采样的 token 被追加到输入序列中, 继续下一步预测.

然后它也采用了*树形草稿 + Tree Attention*的方式, 在每个位置生成多个候选 token, 形成树状结构(例如通过 3 次前向传播生成含 10+ token 的草稿树). 验证时, 和SpecInfer一样, 目标 LLM 利用 tree attention 通过单次前向传播并行验证整棵树, 使用修改后的注意力掩码确保各分支间不可见. 在每个节点递归应用 speculative sampling 接受/拒绝算法, 保证输出分布与目标模型严格一致(贪心和采样设置均无损).
1.3.3.2 EAGLE-2
EAGLE 采用静态草稿树：在所有上下文中使用完全相同形状的树结构, 隐含假设"token 接受率仅取决于其位置", 这样对于简单任务浪费了候选资源, 对于复杂任务又候选不足. EAGLE-2 旨在通过引入一个可动态调整的草稿树来改进这一点. 如下图所示:

![图片](assets/5d42aabf8e75.png)

查询为"10+2"时, EAGLE-1 和 EAGLE-2 的行为是相同的, 但是在接收到 "10+2=" 这个前缀后, EAGLE-1 依旧包含静态的TOP-K个分支. 而EAGLE-2按照置信度, 大概率下一个token为1进行了动态调整.



从算法来看, 主要包含Expand和Rerank两个阶段
**Expand阶段**
一个 token $t_i$ 的全局接受率是从根节点到 $t_i$ 路径上所有 token 接受率的乘积. 因此将其定义为价值 $V_i$:

$$V_i=\prod_{j\in\text{Path}(root,t_i)}p_j\approx\prod_{j\in\text{Path}(root,t_i)}c_j$$

其中, $\text{Path}(root,t_i)$ 表示在草稿树中从根节点到节点 $t_i$ 的路径, $p_j$ 表示节点 $t_j$ 的接受率, 而 $c_j$ 表示来自草稿模型的 $t_j$ 的置信度分数. 置信度分数与接受率强烈正相关. 因此利用这种关系来近似这个价值. 从价值更高的 token 开始的分支更有可能被接受. 因此, 选择最后一层中价值最高的 top-k 个节点作为草稿模型的输入, 并根据输出扩展草稿树.

![图片](assets/661f6596183a.png)
**Rerank阶段**
Expand阶段的目的是加深草稿树. 由于接受率介于 0 和 1 之间, 一个更深的 token 的价值会更低. 一些没有被扩展的浅层节点可能比被扩展的深层节点具有更高的价值. 因此, 不直接使用 Expand 阶段选择的 token 作为草稿, 而是对所有草稿 token 进行重排, 并选择价值最高的 top-m 个 token. 一个节点的价值总是小于或等于其父节点的价值. 对于价值相同的节点, 优先选择更浅的节点. 这确保了重排后选择的 top-m 个 token 仍然构成一个连通的树.

之后, 将选定的 token 展平 (flatten) 为一个一维序列, 作为验证阶段的输入. 为了确保与原始自回归解码的一致性, 还需要调整注意力掩码. 在原始自回归解码中, 每个 token 可以看到所有在它之前的 token, 从而形成一个下三角的注意力矩阵. 当使用草稿树时, 来自不同分支的 token 不应该互相可见. 因此, 注意力掩码必须根据树结构进行调整, 以确保每个 token 只能看到它的祖先节点.

![图片](assets/1cab0b34338b.png)
1.3.3.3 EAGLE-3
EAGLE-3在输入特征上采用了低、中、高层的特征序列(下图中的)融合处理, 将维度为的向量拼接成一个维度为的向量, 然后通过一个全连接(FC)层将其降维至维, 得到一个融合了不同层级信息的特征.

顶层特征主要编码"当前 token 的 logits 信息"(对应下一个 token),而低/中层特征包含更广泛的语义、句法和上下文信息, 对预测更远的 token 更有帮助. 对于 LM Head 权重矩阵为满秩的情况, 顶层特征与下一 token logits 之间是一一对应的, 这意味着仅从顶层特征出发预测"下下个 token"本质上很困难. 多层融合打破了这一信息瓶颈

如图所示:

![图片](assets/60d1618ede07.png)

当前的目标是在前缀"How can I"的基础上生成一个草稿token序列. 如果只输入和, 草稿模型将无法获知随机采样的过程. 因此, 类似于EAGLE 引入采样得到的token "I"的嵌入(embedding). 这个拼接后的向量会通过一个FC层降维至维, 并随后输入到一个单层decoder中, 产生输出. 最后将输入到LM head并采样得到草稿token "do".

需要注意的一个区别是: EAGLE草稿模型的输入是, 或者至少近似是, 目标模型的顶层特征. 相比之下, EAGLE-3草稿模型的输入可能包含来自目标模型的特征, 也可能包含来自草稿模型自身的输出. 因此需要训练草稿模型以适应不同的输入.

因此有一个`Training-Time Test`的改进. EAGLE-3中草稿模型的核心是一个Transformer decoder层. 除了self-attention 操作外, 其他组件不与上下文交互, 因此在训练或测试时无需进一步修改. 唯一需要轻微修改的组件是Attn.

如图所示, 原始的训练数据是一个长度为3的序列 "How can I", 在上下文中具有正常的顺序依赖关系. 因此, 注意力掩码是一个标准的下三角矩阵. 这三个位置的输出是 "are", "we", "do", 它们与 "how", "can", "I" 具有树状的上下文关系. 结果是, 当输入 "are", "we", "do" 被送入第2步时, 注意力掩码需要相应地进行调整, 如下图右上角所示. 所有的注意力掩码都是对角矩阵, 除非当原始训练数据被用作 key时. 在这种情况下使用矩阵乘法会导致巨大的计算浪费, 所以可以使用向量点积来只计算相应位置的注意力分数.

![图片](assets/46dbdbc0e657.png)

在EAGLE-3中, 作者认为发现了Scaling Law, 即为草稿模型提供的训练数据越多, LLM的推理加速比就越高, 并且这种性能提升呈现出持续、可预测的增长趋势. 然后稍微懒了一下, 用模型总结了一个表

![图片](assets/f77c02b920fc.png)

### 1.4 MTP训练集成

#### 1.4.1 Meta MTP

Meta的论文《Better & Faster Large Language Models via Multi-token Prediction》[12] 提出了一个问题:**为什么不在预训练阶段就赋予模型投机能力？** 通过添加 $n$ 个独立输出头同时预测未来 $n$ 个 token, 多 token 预测不仅提升了模型在下游任务(尤其是代码生成)上的质量, 还让这些输出头在推理时直接充当投机解码的草稿, 实现了训练增强与推理加速的统一.

![图片](assets/283c3f40c6a9.png)

#### 1.4.2 DeepSeek-V3 MTP

Meta MTP使用**并行**的独立输出头预测 $n$ 个额外token; DeepSeek-V3则是**顺序**预测, 且保持每个预测深度的完整因果链.

![图片](assets/e3ced35a50f8.png)

**MTP模块结构**: 第 $k$ 个MTP模块包含:

共享嵌入层 Emb(·)

共享输出头 OutHead(·)

Transformer块 TRM(·)

投影矩阵 $M_k\in\mathbb{R}^{d\times2d}$

处理流程:

$$\mathbf{h}_i^{\prime k}=M_k[\text{RMSNorm}(\mathbf{h}_i^{k-1});\text{RMSNorm}(\text{Emb}(t_{i+k}))]$$
$$\mathbf{h}_{1:T-k}^{k}=\text{TRM}_k(\mathbf{h}_{1:T-k}^{\prime k})$$
$$P_{i+k+1}^{k}=\text{OutHead}(\mathbf{h}_i^{k})$$

**MTP训练目标**: 各深度交叉熵损失的加权平均:

$$\mathcal{L}_{\text{MTP}}=\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{\text{MTP}}^{k}$$

其中 $\lambda$ 为权重因子 (前10T tokens设为0.3, 后4.8T设为0.1).

DeepSeek-V3的MTP在每个深度都保持了完整的因果链, 而非像 Meta MTP 那样用独立头并行预测. 这意味着第 $k$ 层的预测依赖于第 $k-1$ 层的输出, 形成了自然的信息传递链路. DeepSeek-V3 MTP标志着投机解码从"后训练附加"到"预训练内置"的范式转变. 它证明了: 将多步预测能力融入预训练, 既能提升模型本身的质量(训练信号密集化), 又能获得推理加速的"免费午餐".

### 1.5 Dflash

前四个阶段的所有方法, 无论多么精巧, 都受制于同一个根本性约束:**自回归串行起草**. 即使EAGLE的草稿模型仅有1层, 生成 $\gamma$ 个token仍需要7次串行前向传播. 起草延迟 $T_{\text{draft}}=\gamma\cdot t_{\text{step}}$ 随草稿长度线性增长, 这迫使设计者在两个方向做出妥协: 要么限制草稿长度(通常 $\gamma\sim8$), 要么限制模型深度(通常1层). 无论怎么选择, $\gamma$ 的线性瓶颈都像一堵墙, 堵死了投机解码进一步突破的空间.

#### 1.5.1 关键创新

论文《DFlash: Block Diffusion for Flash Speculative Decoding》[13]的突破是: 为什么草稿模型必须是自回归的? 如果我们将它重新定位为一个**并行的扩散适配器**, 借助于一些关于Diffusion LLM的范式, 一次性填充所有位置, 不就能彻底摆脱串行的链锁吗?
创新一: 并行扩散起草


个

即一个锚点token(上一轮验证确认的最后一个token)加上 $\gamma$ 个[MASK]占位符. 块内所有位置使用**双向注意力**(而非因果注意力), 每个位置都能看到其他所有位置, 这正是扩散模型的天然特性. 单次前向传播后, 模型一次性输出所有 $\gamma$ 个位置的预测.

关键指标:**延迟与 $\gamma$ 无关**. 生成16个token和生成4个token的延迟几乎相同. 这是因为现代GPU的并行计算能力可以轻松处理多个位置的同时计算, 真正的瓶颈在于权重加载(访存受限)而非计算本身.

![图片](assets/773e3bd16054.png)
创新二: KV注入条件化
并行扩散解决了"速度"问题, 但如何保证"质量"? 一个纯粹独立的扩散模型缺乏目标模型的指导, 就像一个"闭门造车"的助教. DFlash的第二个关键创新是: 让草稿模型在**每一层**都能直接"读取"目标模型的信息.

具体机制: 从Target模型的等距分布的5层中提取隐藏状态, 融合为上下文特征:

$$H_{ctx}=\text{RMSNorm}(W_c[H^{(l_1)};H^{(l_2)};H^{(l_3)};H^{(l_4)};H^{(l_5)}])$$

然后在草稿模型的**每一层**, 将融合特征拼接到Key和Value中:

$$K_c=[W_K^c H_c;\ W_K^c H_d],\qquad V_c=[W_V^c H_c;\ W_V^c H_d]$$

这里 $H_d$ 是草稿token自身的表示.

#### 1.5.2 详细的推理过程

在DFlash中, 给定一个输入提示, 目标模型首先执行一次标准的Prefill, 以生成第一个token. 在此过程中, 从浅层到深层均匀采样的一组固定层中提取隐藏表示. 这些隐藏状态被拼接起来, 并通过一个轻量级的投影层, 以将跨层信息融合成一个紧凑的目标上下文特征, 该特征随后用于对草稿模型进行条件化.

![图片](assets/8a8d93a2d4eb.jpg)

DFlash的第一个核心机制是从目标模型中提取丰富的上下文特征. 论文指出, LLM不同层的隐藏状态包含了不同粒度的语义信息——浅层偏向语法/表面特征, 深层偏向语义/推理信息. DFlash从均匀分布的多个层中提取隐藏状态, 融合为一个紧凑的条件向量. 层选择的设计如下:

跳过第0层(太接近原始embedding, 语义信息不够丰富)

跳过最后3层(这些层过度拟合于下一token预测任务, 多样性不足)

均匀采样确保覆盖从浅层语法特征到深层语义特征的完整谱系

```
def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):    """    计算需要从目标模型中提取隐藏状态的层索引.     核心思想: 从目标模型的多个层中"均匀采样", 获取从浅到深的完整语义信息.     参数:        num_target_layers: 目标模型的总层数(如Qwen3-8B有32层)        num_draft_layers:  草稿模型的层数(如5层), 决定提取几层特征    返回:        层索引列表, 如 [1, 8, 15, 22, 28](对于32层目标模型、5层草稿模型)    """    if num_draft_layers == 1:        # 只有1层草稿模型时, 取目标模型的中间层        return [num_target_layers // 2]    # 起始层索引(跳过第0层, 因为它接近原始嵌入, 信息量有限)    start = 1    # 结束层索引(避免最后3层, 这些层可能过于专注于下一token预测)    end = num_target_layers - 3    span = end - start    # 在 [start, end] 区间内均匀分布 num_draft_layers 个采样点    return [        int(round(start + (i * span) / (num_draft_layers - 1)))        for i in range(num_draft_layers)    ]
```

然后特征抽取如下:

```
def extract_context_feature(    hidden_states: list[torch.Tensor],  # 目标模型所有层的隐藏状态列表    layer_ids: Optional[list[int]],     # 需要提取的层索引) -> torch.Tensor:    """    从目标模型的指定层中提取并拼接隐藏状态, 形成上下文特征.     原理: 将多层隐藏状态在最后一个维度上拼接.     例如, 每层隐藏维度为 D=2048, 提取5层后拼接为 5D=10240 维的特征向量.     后续会通过投影层压缩回 D 维.     注意 offset=1 的设计: hidden_states[0] 是嵌入层输出,     hidden_states[1] 才是第0层Transformer的输出, 所以需要 +1 偏移.     """    offset = 1  # 跳过嵌入层, hidden_states[0]是embed输出    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]    # 在隐藏维度(dim=-1)上拼接, 形成 [batch, seq_len, num_layers * hidden_size] 的特征    return torch.cat(selected_states, dim=-1)
```

**数学表达**: 设目标模型有 $L$ 层, 选取层 $l_1,\dots,l_5$, 则上下文特征为:

$$H_{ctx}=[H^{(l_1)};H^{(l_2)};\dots;H^{(l_5)}]\in\mathbb{R}^{B\times5D}$$

随后通过投影层压缩并归一化:

对应的代码在`DFlashDraftModel.forward()` 中:

```
# DFlashDraftModel.forward() 中的特征投影target_hidden = self.hidden_norm(self.fc(target_hidden))# self.fc: Linear(5 * hidden_size → hidden_size), 将5D压缩为D# self.hidden_norm: RMSNorm, 稳定训练和推理
```

然后是KV注入机制, DFlash将目标特征注入到每一层的Key和Value中. 每一层都能直接"看到"目标模型的指导信号

```
class Qwen3DFlashAttention(nn.Module):    def __init__(self, config: Qwen3Config, layer_idx: int):        super().__init__()        self.config = config        self.layer_idx = layer_idx        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads        self.scaling = self.head_dim**-0.5        self.attention_dropout = config.attention_dropout        self.is_causal = False  # 关键！扩散起草使用双向注意力, 非因果        # Q/K/V/O 投影层——与标准Transformer相同        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)        # QK归一化(Qwen3特有)        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)    def forward(        self,        hidden_states: torch.Tensor,    # 草稿模型的当前隐藏状态 [B, q_len, D]        target_hidden: torch.Tensor,    # 注入的目标模型上下文特征 [B, ctx_len, D]        position_embeddings: tuple[torch.Tensor, torch.Tensor],        attention_mask: Optional[torch.Tensor],        past_key_values: Optional[Cache] = None,        cache_position: Optional[torch.LongTensor] = None,        **kwargs: Unpack[FlashAttentionKwargs],    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:        bsz, q_len = hidden_states.shape[:-1]        ctx_len = target_hidden.shape[1]        # ═══════════════════════════════════════════════════════════════        # 步骤1: Query仅来自草稿模型的隐藏状态        # ═══════════════════════════════════════════════════════════════        q = self.q_proj(hidden_states)        q = q.view(bsz, q_len, -1, self.head_dim)        q = self.q_norm(q).transpose(1, 2)  # [B, num_heads, q_len, head_dim]        # ═══════════════════════════════════════════════════════════════        # 步骤2: Key和Value来自两个来源的拼接        #   - k_ctx/v_ctx: 来自目标模型的上下文特征(注入部分)        #   - k_noise/v_noise: 来自草稿模型自身的隐藏状态        # ═══════════════════════════════════════════════════════════════        k_ctx = self.k_proj(target_hidden)    # 目标特征 → Key空间        k_noise = self.k_proj(hidden_states)  # 草稿自身 → Key空间        v_ctx = self.v_proj(target_hidden)    # 目标特征 → Value空间        v_noise = self.v_proj(hidden_states)  # 草稿自身 → Value空间        # ═══════════════════════════════════════════════════════════════        # 步骤3: 在序列维度上拼接, 这就是"KV注入"的核心操作！        # Key: [目标特征的Key | 草稿自身的Key]        # Value: [目标特征的Value | 草稿自身的Value]        # 这等效于扩展了注意力机制的"知识库"        # ═══════════════════════════════════════════════════════════════        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)        k = self.k_norm(k).transpose(1, 2)  # [B, num_kv_heads, ctx_len+q_len, head_dim]        v = v.transpose(1, 2)        # ═══════════════════════════════════════════════════════════════        # 步骤4: 应用旋转位置编码 (RoPE)        # 注意: ctx部分的位置编码对应上下文位置, noise部分对应当前位置        # ═══════════════════════════════════════════════════════════════        cos, sin = position_embeddings        q, k = apply_rotary_pos_emb(q, k, cos, sin)        # ═══════════════════════════════════════════════════════════════        # 步骤5: KV缓存更新(支持多轮起草复用)        # ═══════════════════════════════════════════════════════════════        if past_key_values is not None:            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)        # ═══════════════════════════════════════════════════════════════        # 步骤6: 执行注意力计算        # Q: [B, heads, q_len, d] × K^T: [B, heads, d, ctx_len+q_len]        # 每个草稿位置都能"看到"目标模型的上下文 + 块内其他位置        # ═══════════════════════════════════════════════════════════════        attn_fn: Callable = eager_attention_forward        if self.config._attn_implementation != "eager":            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]        attn_output, attn_weights = attn_fn(            self, q, k, v, attention_mask,            dropout=0.0 if not self.training else self.attention_dropout,            scaling=self.scaling,            sliding_window=self.sliding_window,            **kwargs,        )        attn_output = attn_output.reshape(bsz, q_len, -1)        attn_output = self.o_proj(attn_output)        return attn_output, attn_weights
```

原本注意力机制的Key-Value空间维度为 $N_d$ (草稿序列长度), KV注入后扩展为 $N_c+N_d$ (目标特征长度 + 草稿序列长度):

$$K_c=[W_K^c H_c;\ W_K^c H_d],\qquad V_c=[W_V^c H_c;\ W_V^c H_d]$$

注意力计算变为:

$$\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK_c^\top}{\sqrt{d}}\right)V$$

这等效于**在每个草稿位置执行一次信息检索**, 从目标模型的"知识库"中检索与当前查询最相关的信息.

我们知道接受概率与目标分布和草稿分布的总变差(TV)距离直接相关:

$$P_{\text{accept}}=\sum_x\min(p_t(x),p_d(x))=1-\frac{1}{2}\|p_t-p_d\|_1$$

KV注入通过在信息几何意义上将草稿分布 $q$ "拉向"目标分布 $p$, 减小两者在统计流形上的距离, 从而提升接受率.

接下来是并行扩散草稿阶段(Diffusion Drafting). 传统自回归起草需要 $\gamma$ 次串行前向传播(每次生成1个token), 而DFlash只需1次前向传播即可生成 $\gamma$ 个token:

```
自回归起草 (EAGLE-3):  t=1: [anchor] → token₁  t=2: [anchor, token₁] → token₂  t=3: [anchor, token₁, token₂] → token₃  ...共需 γ 次前向传播块扩散起草 (DFlash):  t=1: [anchor, MASK, MASK, ..., MASK] → token₁, token₂, ..., token_γ  ...仅需 1 次前向传播.
```

DFlash使用特殊的MASK token作为占位符, 表示"待预测位置". 起草时:

第1个位置放入**锚点token**(上一轮验证产生的bonus token)

后续 $\gamma$ 个位置全部填充MASK token

另外注意`Qwen3DFlashAttention` 中`self.is_causal = False` 的设置, 块内所有位置使用**双向注意力**, 每个位置都能看到块内其他位置. 这与自回归的因果掩码根本不同, 允许块内信息的双向流动, 提升预测质量.

```
# === 起草阶段 ===# block_output_ids 初始为 [anchor_token, MASK, MASK, ..., MASK]block_output_ids = output_ids[:, start : start + block_size].clone()block_position_ids = position_ids[:, start : start + block_size]if block_size > 1:    # 将 block_output_ids(含MASK token)转换为嵌入    noise_embedding = target.model.embed_tokens(block_output_ids)    # 草稿模型一次前向传播: 输入是噪声嵌入, 条件是目标模型特征    draft_logits = target.lm_head(model(        target_hidden=target_hidden,         # 目标模型上下文特征(KV注入的条件)        noise_embedding=noise_embedding,     # [anchor_emb, MASK_emb, ..., MASK_emb]        position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],        past_key_values=past_key_values_draft,        use_cache=True,        is_causal=False,                     # 块内双向注意力！    )[:, 1 - block_size :, :])  # 取后 block_size-1 个位置的输出    # 裁剪草稿KV缓存(防止累积无效缓存)    past_key_values_draft.crop(start)    # 采样得到草稿token, 填入 block_output_ids 的第2~最后位置    block_output_ids[:, 1:] = sample(draft_logits)
```

关键细节解析:

`target.lm_head(...)`: 草稿模型**共享**目标模型的LM head, 不需要自己的词汇表投影

`[:, 1 - block_size :, :]`: 只取最后 $\gamma$ 个位置的输出(第1个位置是锚点, 不需要预测)

`is_causal=False`: 块内双向注意力, 允许后面的位置参考前面的MASK信息

最后是验证阶段, 使用目标模型对整个草稿块进行并行验证, 然后通过拒绝采样确定接受长度.

```
# === 验证阶段 ===def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:    if temperature < 1e-5:        return torch.argmax(logits, dim=-1)    bsz, seq_len, vocab_size = logits.shape    logits = logits.view(-1, vocab_size) / temperature    probs = torch.softmax(logits, dim=-1)    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)    # 目标模型并行处理整个草稿块output = target(    block_output_ids,                    # [anchor, draft₁, draft₂, ..., draft_{γ-1}]    position_ids=block_position_ids,    past_key_values=past_key_values_target,    use_cache=True,    output_hidden_states=block_size > 1, # 同时提取隐藏状态供下一轮使用)# 目标模型在每个位置的预测(posterior)posterior = sample(output.logits, temperature)# ═══════════════════════════════════════════════════════════════# 接受长度计算——这行代码是整个验证逻辑的精华！# ═══════════════════════════════════════════════════════════════acceptance_length = (    block_output_ids[:, 1:] == posterior[:, :-1]  # 逐位置比较: 草稿token vs 目标模型预测).cumprod(dim=1).sum(dim=1)[0].item()            # cumprod: 遇到第一个不匹配就截断
```
与标准 Speculative Sampling 的区别
![图片](assets/ce882b7a31f5.png)

根本原因: 扩散模型没有逐位置自回归概率.

标准 Speculative Sampling 要求草稿模型提供每个位置的条件概率: $q(x_i\mid x_{<i})$. 但 DFlash 的草稿模型是**块扩散模型**, 它通过单次前向传播**同时生成**所有位置的 token, 产出的是一个联合分布 $q(x_1,\dots,x_\gamma)$, 无法自然分解为逐位置的条件概率. 因此 DFlash 采用了更简洁的**"采样后比较"** 策略:

```
# 草稿: 扩散模型采样一个完整块draft_tokens = sample(draft_logits)# 验证: 目标模型独立采样, 逐位置比对posterior = sample(target_logits, temperature)accept = (draft == target_prediction)
```

当 temperature > 0 时, DFlash 的接受率**理论上低于** 标准 SS:

**标准 SS 期望接受率**:

$$\mathbb{E}[\alpha_{SS}]=\sum_x\min(p(x),q(x))$$

**DFlash 期望接受率**:

$$\mathbb{E}[\alpha_{DFlash}]=\sum_x p(x)q(x)$$

后者是两个分布的内积, 始终 ≤ 前者. 但 DFlash 通过更高质量的草稿(KV注入让 q ≈ p)来弥补这一差距, 当 q 足够接近 p 时, 两种方法的接受率趋于一致.

#### 1.5.3 详细的训练过程
随机锚点采样
训练时不是将序列均匀切块, 而是**随机选择锚点位置**来构建训练块:

```
训练序列: [t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈, t₉, t₁₀, ...]随机锚点位置: 3, 7构建的训练块:  块1: [t₃, MASK, MASK, MASK] → 目标: [t₄, t₅, t₆]  块2: [t₇, MASK, MASK, MASK] → 目标: [t₈, t₉, t₁₀]
```

设计动机:

**匹配推理分布**: 推理时锚点位置取决于前一轮验证结果, 本质上是随机的

**数据增强**: 同一序列每个epoch选不同锚点, 等效于生成大量不同训练样本

**鲁棒性**: 模型学会在任何位置都能作为起点进行并行预测
稀疏注意力掩码
训练时多个块被拼接成单个序列, 使用稀疏注意力掩码联合处理:

```
块内: 双向注意力(所有位置互相可见)块间: 完全隔离(不同块互不干扰)KV注入: 每个块的token都能注意到对应的目标上下文特征
```

这种设计允许高效的批量训练, GPU可以一次处理多个块的梯度计算.
位置加权损失
在投机解码中, 前面位置的错误代价远大于后面位置, 位置 $i$ 的拒绝会导致所有后续token被丢弃. DFlash使用指数衰减的损失权重来反映这种不对称性:

$$w_i=\exp\left(-\frac{i}{\tau}\right),\quad i=1,\dots,\gamma$$

其中 $\gamma$ 是块大小, $\tau$ 控制衰减速率.

理论依据是如果每个位置的条件接受概率为, 则从位置1存活到位置 的概率为, 近似为指数衰减. 给前面位置更大的损失权重, 本质上是在优化**预期接受长度**.
参数共享与冻结策略
设计思路:

**共享嵌入层**: 草稿模型不需要学习词汇表表示, 直接复用目标模型的

**共享LM头**: 确保草稿模型的输出空间与目标模型完全对齐

**只训练Transformer层**: 大幅减少可训练参数量, 草稿模型真正成为轻量级"扩散适配器"

```
# DFlashDraftModel 中的参数共享设计self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)# ↑ 只有这些是可训练参数(加上Transformer层)# 嵌入层和LM头共享自目标模型(冻结)# 在推理时:noise_embedding = target.model.embed_tokens(block_output_ids)  # 共享嵌入draft_logits = target.lm_head(model(...))                       # 共享LM头
```

#### 1.5.4 加速比分析与性能评估

投机解码的平均每token延迟为:

$$L=\frac{T_{\text{draft}}+T_{\text{verify}}}{\tau}$$

其中:

$T_{\text{draft}}$: 起草延迟

$T_{\text{verify}}$: 验证延迟(≈目标模型一次前向传播)

$\tau$: 预期接受token数

加速比为:

$$S=\frac{\tau\cdot T_{\text{verify}}}{T_{\text{draft}}+T_{\text{verify}}}$$

**自回归起草** (EAGLE-3): $T_{\text{draft}}=O(\gamma)$, $\tau$ 因模型容量限制快速饱和

**扩散起草** (DFlash): $T_{\text{draft}}=O(1)$, $\tau$ 随模型深度有效增长

DFlash的关键优势: 由于 $T_{\text{draft}}$ 对 $\gamma$ 不敏感, 可以同时增大 $\gamma$ (更长草稿)和增加模型深度(更高接受率), 在"起草质量-起草延迟"的帕累托前沿上完全占优.

DFlash证明了: 在投机解码这个受限问题域中,**改变计算范式(串行→并行)比在同一范式内优化(更好的自回归)能带来数量级的提升**. 然而, 范式革命也留下了新的挑战: "并行性 vs 因果性"这一新矛盾, 以及"高并发下无差别验证浪费batch容量"的系统问题, 将在接下来的最后一个阶段被逐一解决.

### 1.6 DDTree

DFlash的块扩散模型一次前向传播产出的完整logits分布, 但vanilla DFlash仅从每个位置采样一个token构成线性序列:

```
# dflash.py L70-79: vanilla DFlash的起草——只用了argmax/sample的结果draft_logits = target.lm_head(model(    target_hidden=target_hidden,    noise_embedding=noise_embedding,    ...)[:, -block_size + 1 :, :])past_key_values_draft.crop(start)block_output_ids[:, 1:] = sample(draft_logits)  # ← 仅采样一条路径!
```

DDTree不丢弃这些logits, 而是从中构建一棵**草稿树**, 在固定节点预算B下选择最可能被目标模型接受的多条路径:

```
# ddtree.py L363-373: DDTree的起草——保留完整draft_logits用于建树draft_logits = target.lm_head(model(    target_hidden=target_hidden,    noise_embedding=noise_embedding,    ...)[:, -draft_horizon:, :])past_key_values_draft.crop(start)# 注意: 不在这里采样! draft_logits直接传给build_ddtree_tree
```

DDTree 推理流程如下:
Stage.1 Drafting 阶段
起草阶段与vanilla DFlash完全相同, 调用DFlash草稿模型一次, 获得`draft_horizon` 个位置的logits:

```
# ddtree.py L360-373block_output_ids = output_ids[:, start : start + block_size].clone()root_token = block_output_ids[:, :1]noise_embedding = target.model.embed_tokens(block_output_ids)draft_logits = target.lm_head(model(    target_hidden=target_hidden,    noise_embedding=noise_embedding,    position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],    past_key_values=past_key_values_draft,    use_cache=True,    is_causal=False,)[:, -draft_horizon:, :])  # 取最后draft_horizon个位置的logits
```
Stage.2 最优树的构建
DDTree要解决的问题: 给定扩散模型产出的 L 个位置的边缘分布, 在节点预算 B 下构建一棵树使得期望接受长度最大., 对应论文中的Algorithm-1 算法实现如下:

![图片](assets/a2a4b0a9bc40.png)
Proposition 1
将期望接受长度分解为前缀概率之和:

$$\mathbb{E}_{Y\sim q}[\alpha_T(Y)]=\sum_{v\in\mathcal{T}}q(u_v\mid b)=\sum_{v\in\mathcal{T}}\prod_{i=1}^{|v|}q_i(u_{v,i}\mid b)$$

**直觉解释**: 接受长度 =(每个深度有节点命中的概率). 而"有节点命中"的概率就是该深度所有节点的前缀概率之和. 因此最大化目标等价于:*选B个概率最大的前缀*.
Proposition 2
证明贪心=最优: 由于因子化分布的乘积结构, 祖先概率 > 后代概率 ($q(u_{v,1},\dots,u_{v,i-1})\ge q(u_{v,1},\dots,u_{v,i})$), 所以按概率排序选top-B时, 祖先一定先于后代被选中, 前缀闭合性自动满足.
1. Top-K logits提取与log概率计算
```
# ddtree.py L102-114topk = min(budget, draft_logits.shape[-1])   # K = min(B, |V|)depth_limit = int(draft_logits.shape[0])     # L = 15# GPU上做top-K排序, 然后传回CPU做heap操作logits = draft_logits.float()top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)log_z = torch.logsumexp(logits, dim=-1, keepdim=True)  # log partition functiontop_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
```

对应论文 Lemma 1证明最优树只需考虑每个位置的top-K token. 这里直接用`torch.topk`实现, 然后通过`logsumexp`归一化为log概率.
2. Best-First Heap搜索 (Algorithm 1核心)
```
# ddtree.py L117-148: 论文Algorithm 1的精确实现top_log_probs_np = top_log_probs_cpu.numpy()top_token_ids_np = top_token_ids_cpu.numpy()# 初始化: 放入秩元组(1) — 第一个位置的最高概率tokenfirst_logw = float(top_log_probs_np[0, 0])heap = [(-first_logw, (0,), 0, 1, 0, first_logw)]#        ↑neg_score  ↑ranks ↑parent ↑depth ↑rank ↑logwnode_count = 0while heap and node_count < budget:    _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)    # 将弹出的节点加入树    token_id = int(top_token_ids_np[depth - 1, rank])    current_index = node_count + 1    node_token_ids_np[node_count] = token_id    node_depths_np[node_count] = depth    parents_np[current_index] = parent_index    child_maps[parent_index][token_id] = current_index    node_count += 1    # 推入下一个兄弟: 同位置的下一个概率token    if rank + 1 < topk:        sibling_logw = logw - top_log_probs_np[depth-1, rank] + top_log_probs_np[depth-1, rank+1]        heapq.heappush(heap, (-sibling_logw, ..., parent_index, depth, rank+1, sibling_logw))    # 推入第一个子节点: 下一个位置的最高概率token    if depth < depth_limit:        child_logw = logw + top_log_probs_np[depth, 0]        heapq.heappush(heap, (-child_logw, ..., current_index, depth+1, 0, child_logw))
```

对应论文Proposition 3: 此算法保证返回目标 $\mathbb{E}_{Y\sim q}[\alpha_T(Y)]$ 下的精确最优树, 复杂度 $O(B\log B)$.

为什么只需两种展开操作 ?
在因子化分布下, 前缀概率是各位置概率的乘积. 从任何前缀出发, 只能通过两种方式到达"相邻的较低概率前缀":

**兄弟**: 将当前位置的token替换为概率次高的token (同深度探索)

**子节点**: 在末尾追加下一位置的最高概率token (加深探索)

这两种操作恰好覆盖了所有可能的概率递减方向, 保证了heap按概率递减顺序弹出所有前缀. 具体来说:

使用**max-heap** (Python的heapq是min-heap, 所以取负数)

每次弹出当前概率最高的前缀 (对数空间中得分最高)

弹出后生成两个候选: 兄弟(score减少) 和 子节点(score减少)

由于因子化分布的乘积结构,**前缀闭合性自动满足** (Proposition 2)

构建可见性矩阵 (Ancestor-Only Mask)
```
# ddtree.py L151-159: 构建树注意力的visibility矩阵current_length = 1 + node_count  # 根节点(bonus token) + B个树节点visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)visibility_np[0, 0] = True  # 根节点只能看到自己for index in range(1, current_length):    parent_index = int(parents_np[index])    # 关键: 每个节点继承其父节点的可见性, 再加上自己    visibility_np[index, :index] = visibility_np[parent_index, :index]    visibility_np[index, index] = True
```

`visibility[i][j] = True` 表示节点 在 attention 中可以看到节点. 每个节点只能看到自己和所有祖先, 这就是 tree attention mask.
Stage 3: 树编译
将树结构编译为目标模型可消费的张量:

```
# ddtree.py L169-209def compile_ddtree_tree(root_token_id, start, node_token_ids, node_depths,                        visibility_cpu, past_length, ...):    current_length = 1 + int(node_token_ids.numel())  # bonus + B个节点    # 1. 构建input_ids: [bonus_token, node_1, node_2, ..., node_B]    verify_input_ids[0, 0] = root_token_id    verify_input_ids[0, 1:current_length].copy_(node_token_ids)    # 2. 构建position_ids: 按树深度分配位置编码    verify_position_ids[0, 0] = start           # bonus token的绝对位置    verify_position_ids[0, 1:current_length].copy_(node_depths)    verify_position_ids[0, 1:current_length].add_(start)  # 转为绝对位置    # 3. 构建attention_mask: tree block用finfo.min填充, 可见位置填0    tree_block = attention_mask_buffer[..., :current_length, past_length:past_length+current_length]    tree_block.fill_(torch.finfo(dtype).min)     # 默认全部遮蔽    tree_block.masked_fill_(visibility, 0)       # 可见位置设为0(不遮蔽)
```

**关键设计**:

同深度节点共享position_id (因为它们代表"同一个未来位置的不同候选")

attention_mask =`finfo.min`表示遮蔽,`0`表示可见

树内节点通过visibility矩阵互相可见, 同时都能attend to过去的KV cache
Stage.4 目标模型验证 (Tree Attention)
**核心**: 目标模型在**一次前向传播**中同时为所有 B+1 个位置产出 logits. 每个节点的logits表示"如果上下文是`[context + 该节点的祖先路径]`, 目标模型预测的下一个token分布".

```
# ddtree.py L408-416: 目标模型一次前向传播验证整棵树output = target(    verify_input_ids,            # [1, 1+B] — bonus token + B个树节点    position_ids=verify_position_ids,  # 按树深度分配    attention_mask=verify_attention_mask,  # ancestor-only mask    past_key_values=past_key_values_target,    use_cache=True,    output_hidden_states=True,)
```
Stage 5: 验证与KV Cache压缩
从根开始, 目标模型在每个位置选择一个token. 如果该token恰好是树中某个子节点, 则继续 walk; 否则停止. 最终`next_token`成为下一轮的bonus token.

![图片](assets/fd6e5ec6155b.png)

```
# ddtree.py L212-223: 沿树找最长匹配路径def follow_verified_tree(child_maps, posterior):    posterior_tokens = posterior[0].tolist()  # 目标模型在每个位置的预测    accepted_indices = [0]  # 从根节点(bonus token)开始    current_index = 0    next_token = int(posterior_tokens[current_index])    # 核心循环: 目标模型选的token是否在当前节点的子节点中?    while next_token in child_maps[current_index]:        current_index = child_maps[current_index][next_token]        accepted_indices.append(current_index)        next_token = int(posterior_tokens[current_index])    return accepted_indices, next_token  # 接受的路径 + 下一轮bonus token
```

![图片](assets/586d0bea6d21.png)

验证后, KV cache中包含了所有B+1个节点的KV, 但只有被接受路径上的KV需要保留:

```
# ddtree.py L419-429: 提交阶段posterior = sample(output.logits, temperature)accepted_indices, next_token = follow_verified_tree(child_maps, posterior)# 仅保留被接受路径的KV cachecompact_dynamic_cache(past_key_values_target, start, accepted_indices)# 仅保留被接受路径的target_hidden特征(供下一轮起草)target_hidden = extract_context_feature(    output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)
```

压缩操作的具体实现: 将被接受节点的KV紧凑排列到cache前部:

```
# ddtree.py L226-242: in-place KV cache压缩def _compact_appended_window(cache_tensor, past_length, keep_current_indices):    current_length = cache_tensor.shape[-2] - past_length    # 从新增的window中, 仅保留keep_current_indices指定的位置    kept_tail = cache_tensor.narrow(-2, past_length, current_length) \                            .index_select(-2, keep_current_indices)    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)
```

### 1.7 JetSpec

#### 1.7.1 构建树形因果性

DFlash虽然打破了串行瓶颈, 但也暴露了新的根本矛盾: 并行生成牺牲了因果性, 导致接受率沿草稿块快速衰减. JetSpec的改进是解决投机解码领域长期存在的*因果性-效率困境 (causality-efficiency dilemma)*:

**自回归起草器**(如EAGLE): 路径条件化, 高接受率, 但起草代价随树深度线性增长

**块扩散起草器**(如DFlash): 单次前向传递, 极低代价, 但分支无关预测导致不一致

JetSpec 通过**树形因果注意力掩码**在单次前向传递中同时实现并行预测和因果条件化

![图片](assets/6a63a6ded3f0.png)

JetSpec 的核心机制是: 在一次前向传递中并行预测整棵候选树的所有节点, 同时让每个节点条件化于其祖先节点的信息. **分支条件概率分解**:

$$q_{\text{tree}}(y_{t,b}\mid x)\propto\prod_{i=1}^{k} r_i(y_{t,i}\mid x)$$

其中 $\pi(v)$ 为从根到节点 $v$ 的路径, $h_v^*$ 为融合隐状态特征. 这个分解**镜像了**目标模型的自回归分解:

$$p(y_{t,b}\mid x)=\prod_{i=1}^{k}p(y_{t,i}\mid x,y_{<i})$$

为什么因果性至关重要?
没有因果条件化时(如DFlash), 树构造按照伪分布排序:

$$q_{\text{PAR}}(y_{t,b}\mid x)\propto\prod_{i=1}^{k}r_i(y_{t,i}\mid x)$$

其中 $r_i$ 为位置 $i$ 处的分支无关分布. 这意味着单独合理但相互不一致的 token 组合会被优先选择. 例如前缀为 "We" 时, 扩散头可能选出 " given told that", "given" 和 "told" 单独看合理, 但不能相邻出现.

掩码定义
对于两个树节点 $u$ 和 $v$, 掩码定义为:

$$
M_{v,u}=\begin{cases}0,&\text{if }u\in\text{Anc}(v)\cup\{v\}\\
-\infty,&\text{otherwise}\end{cases}
$$

语义:

节点**可以关注**自身和所有祖先 → 对应

节点**不能关注**其他任何节点(后代、兄弟分支) → 对应

前缀 tokens 对所有树节点可见
注意力计算
节点 $v$ 的掩码注意力计算:

$$\text{Attn}(Q_v,K,V)=\text{softmax}\left(\frac{Q_vK^\top}{\sqrt{d}}+M_v\right)V$$

代码实现: 祖先矩阵构建
掩码的核心是祖先关系矩阵. 在代码中, 通过`_build_ancestor_matrix_np` 构建:

```
# jetspec/tree/_core/ancestor.pydef _build_ancestor_matrix_np(parents: list[int], num_nodes: int) -> np.ndarray:    """Dense bool ancestor matrix in parent-before-child order."""    anc_np = np.eye(num_nodes, dtype=np.bool_)    for i in range(1, num_nodes):        p = parents[i]        if 0 <= p < num_nodes:            anc_np[i] |= anc_np[p]    return anc_np
```

该算法通过传递闭包构建祖先关系: 每个节点继承其父节点的所有祖先, 形成 的布尔矩阵, 其中`anc_np[i][j] = True` 表示节点 是节点 的祖先(或自身).
代码实现: Triton Tree Attention Kernel
树形注意力的 Triton kernel 实现了掩码逻辑:

前缀位置(`is_prefix`)对所有树节点全部可见

树内位置通过查询祖先矩阵(`Ancestor`)决定是否可以 attend

非祖先关系的位置被设为`-inf`, softmax 后权重为 0

```
# jetspec/core/tree_attention_kernel.py - Triton kernel 核心逻辑@triton.jitdef _tree_attn_fwd(Q, K, V, Out, Ancestor, sm_scale, prefix_len, N, KV_LEN, ...):    # ...    for start_n in range(0, KV_LEN, BLOCK_N):        offs_n = start_n + tl.arange(0, BLOCK_N)        # 计算 QK^T        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale        # 关键掩码逻辑: prefix 位置全部可见, tree 节点只看祖先        is_prefix = offs_n[None, :] < prefix_len        in_tree = offs_n[None, :] >= prefix_len        tree_kv = tl.maximum(offs_n[None, :] - prefix_len, 0)        anc = tl.load(            Ancestor + offs_m[:, None] * N + tree_kv,            mask=mask_m[:, None] & in_tree & mask_n[None, :],            other=0,        )        # 只有 prefix 或祖先关系才允许 attend        attend = is_prefix | (in_tree & (anc != 0))        qk = tl.where(attend & mask_n[None, :] & mask_m[:, None], qk, float("-inf"))
```

#### 1.7.2 推理流程

整个端到端推理流程如下:

![图片](assets/83c4ba35076b.jpg)
草稿头架构
JetSpec 的草稿头是一个**5层 Qwen3-style 解码器**:

```
# jetspec/models/draft_head.py - DFlashDraftModel.forward()def forward(self, position_ids, attention_mask=None, noise_embedding=None,            target_hidden=None, past_key_values=None, use_cache=False, **kwargs):    hidden_states = noise_embedding    # 步骤1: 融合目标模型多层隐状态    target_hidden = self.hidden_norm(self.fc(self.hidden_dim_adapter(target_hidden)))    position_embeddings = self.rotary_emb(hidden_states, position_ids)    # 步骤2: 5层解码器处理, 每层注入target特征作为KV    for layer in self.layers:        hidden_states = layer(            hidden_states=hidden_states,            target_hidden=target_hidden,  # 注入融合特征            attention_mask=attention_mask,            position_ids=position_ids,            past_key_value=past_key_values,            use_cache=use_cache,            position_embeddings=position_embeddings,            **kwargs,        )    return self.norm(hidden_states)  # caller applies target lm_head
```
特征融合机制
从目标模型的层 {1, 9, 17, 25, 33}(共36层中选5层)提取隐状态:

```
# jetspec/models/draft_head.py - extract_context_feature()def extract_context_feature(hidden_states, layer_ids: list[int]) -> torch.Tensor:    """沿通道维度拼接选定目标层的隐状态.    offset=1 因为HF返回embedding输出在index 0. """    offset = 1    selected_states = []    for layer_id in layer_ids:        selected_states.append(hidden_states[layer_id + offset])    return torch.cat(selected_states, dim=-1)  # (B, T, 5*4096)
```
注意力层中的 KV 注入
每个草稿层的注意力中,**target_hidden 作为上下文 KV 注入**:

```
# jetspec/models/draft_head.py - Qwen3DFlashAttention.forward()def forward(self, hidden_states, target_hidden, position_embeddings, attention_mask, ...):    bsz, q_len = hidden_states.shape[:-1]    ctx_len = target_hidden.shape[1]    q = self.q_proj(hidden_states)  # Query 来自草稿token    # Key/Value 同时来自 target_hidden 和 draft hidden_states    k_ctx = self.k_proj(target_hidden)   # 目标特征的 Key    k_noise = self.k_proj(hidden_states)  # 草稿token的 Key    v_ctx = self.v_proj(target_hidden)    v_noise = self.v_proj(hidden_states)    # 拼接: [context_KV | draft_KV]    k = torch.cat([k_ctx, k_noise], dim=1)    v = torch.cat([v_ctx, v_noise], dim=1)
```
块因果掩码构建
所有树节点的 Q/K/V 在同一个 batch 中并行计算, 但通过因果掩码确保每个节点只能看到其祖先. GPU 上的矩阵乘法是并行的(效率), 掩码确保信息流是因果的(质量).

```
# jetspec/models/draft_head.py - _build_dflash_causal_attention_mask()def _build_dflash_causal_attention_mask(*, query, key, cached_kv_len, ctx_len):    q_len = query.shape[-2]    kv_len = key.shape[-2]    key_positions = torch.arange(kv_len, device=query.device)    query_positions = cached_kv_len + ctx_len + torch.arange(q_len, device=query.device)    # 因果性: 只能关注位置 <= 自身的位置    can_attend = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)    mask = torch.zeros((1, 1, q_len, kv_len), dtype=query.dtype, device=query.device)    return mask.masked_fill(can_attend.logical_not().unsqueeze(0).unsqueeze(0),                           torch.finfo(query.dtype).min)
```
树构建算法 (Best-First Tree Building)
下图展示了 Best-First Heap 扩展构建候选树的完整过程, 以及目标模型沿 child_maps 的贪心验证

![图片](assets/57147a56d81e.png)

JetSpec 使用**最佳优先搜索 (Best-First Search)** 构建候选树:

```
# jetspec/tree/baselines/accum_logp.py - _build_from_topk()def _build_from_topk(root_token, topk_tokens_cpu, topk_logprobs_cpu, budget, device):    tokens_list = [root_token]    parents_list = [-1]    depths_list = [0]    cum_lp_list = [0.0]    num_nodes = 1    # 最大堆, 按累积log概率排序(取负数用于最小堆)    counter = 0    heap = [(0.0, counter, 0)]  # (neg_cum_logprob, order, node_idx)    while heap and num_nodes < budget:        neg_cum_lp, _, node_idx = heapq.heappop(heap)  # 弹出最高分节点        d = depths_list[node_idx]        if d >= D:            continue  # 已达最大深度        children_to_add = min(k, budget - num_nodes)        for j in range(children_to_add):            child_token = topk_tokens_cpu[d][j]            child_cum_lp = -neg_cum_lp + topk_logprobs_cpu[d][j]            tokens_list.append(child_token)            parents_list.append(node_idx)            depths_list.append(d + 1)            cum_lp_list.append(child_cum_lp)            counter += 1            heapq.heappush(heap, (-child_cum_lp, counter, num_nodes))            num_nodes += 1
```

使用累积草稿对数概率作为评分:

$$s(\pi(v))=\sum_{u\in\pi(v)}\log q(y_u\mid x,h_u^*,\pi_{<u})$$

工作流程总结
**初始化**: 优先队列放入根节点, 评分为 0

**循环扩展**: 弹出最高评分(累积 log-prob 最高)的可扩展节点

**子节点生成**: 在该节点深度取 Top-W 候选 token, 计算子节点评分 = 父节点评分 + 当前位置 log-prob

**终止条件**: 节点数达到预算 或无可扩展节点

**输出**: 构建包含 token_ids, parent_indices, depth, ancestor 等的`DraftTree` 结构

Tree Verification
接受规则基于投机解码的拒绝采样:
$$\alpha_t = \min\left(1, \frac{p(y_t \mid x, y_{<t})}{q(y_t \mid x, y_{<t})}\right)$$

代码实现如下

```
# jetspec/tree/_core/accept.py - tree_accept()def tree_accept(tree: DraftTree, target_logits, temperature=0.0):    """找到最长被接受的根到叶路径. """    logits = target_logits.reshape(-1, target_logits.shape[-1])    posterior = _sample_greedy(logits, temperature)  # 目标模型的 argmax    posterior_tokens = [int(token) for token in posterior.reshape(-1).tolist()]    # 构建子节点映射    if tree.child_maps is None:        tree.child_maps = _build_child_maps_cpu(token_ids, parent_indices, tree.num_nodes)    # 从根开始贪心遍历: 沿目标模型预测走    accepted_path = [0]    current = 0    while True:        next_token = posterior_tokens[current]  # 目标模型在当前节点的预测        child_idx = tree.child_maps[current].get(next_token)  # 找匹配的子节点        if child_idx is None:            break  # 无匹配子节点, 停止        accepted_path.append(child_idx)        current = child_idx    acceptance_length = len(accepted_path) - 1    correction_token = posterior_tokens[current]  # 修正token    return accepted_path, acceptance_length, correction_token
```

在GPU上通过 Pointer Jumping 加速:

```
# jetspec/tree/_core/accept.py - gpu_tree_accept()def gpu_tree_accept(tree_tokens, greedy_targets, parent_indices, depths, max_depth=None):    """向量化贪心树接受, 完全在GPU上执行."""    # ═══════════════════════════════════════════════════════    # 步骤1: 计算每个节点的局部匹配 (全部并行)    # ═══════════════════════════════════════════════════════    match[0] = True  # 根节点总是匹配    match[1:] = (valid_parent[1:] & ~overwritten[1:]                 & (tree_tokens[1:] == greedy_targets[safe_parents[1:]]))    # 含义: 节点i匹配 ⟺ 目标模型在i的父节点处预测的token == 节点i的token    # ═══════════════════════════════════════════════════════    # 步骤2: Pointer Jumping — O(log D) 传播前缀匹配信息    # ═══════════════════════════════════════════════════════    prefix_match = match.clone()     # 初始: 每个节点只知道自己是否局部匹配    jump = safe_parents.clone()      # 初始: 每个节点的跳跃目标 = 父节点    for _ in range(max(1, max_depth.bit_length())):        # 核心操作1: 合并——我匹配 且 我跳到的祖先也匹配 → 整段路径匹配        prefix_match = prefix_match & prefix_match[jump]        # 核心操作2: 倍增——下次跳到更远的祖先        jump = jump[jump]    # ═══════════════════════════════════════════════════════    # 步骤3: 选择最深的完全匹配节点    # ═══════════════════════════════════════════════════════    score = torch.where(prefix_match, depths, neg_ones)    best_node = torch.argmax(score)  # 深度最大且前缀完全匹配的节点
```

![图片](assets/1c3a4788e182.jpg)

渣注
个人觉得, 对于JetSpec的因果性, 可能更恰当的说法是**“位置感知的受限注意力模式"** 的工程设计.

![图片](assets/c29f1a38c2d8.jpg)

因果掩码的真正作用不是"实现因果性", 而是**防止不同分支之间的信息污染**(cross-branch contamination)+ 提供**有序的位置归纳偏置**. 这两点虽然有用, 但与"因果条件化"是完全不同的概念.

## 2. DSpark

令 $\tau$ 表示每个周期接受的 token 数量, 令 $T_{\text{draft}}$ 和 $T_{\text{verify}}$ 分别表示起草和验证过程的 Wall clock 时间. 平均每个生成 token 的延迟为:

$$L=\frac{T_{\text{draft}}+T_{\text{verify}}}{\tau}$$

因此, 提升加速比可以归结为三个杠杆: 降低 $T_{\text{draft}}$ (更快地起草), 提高 $\tau$ (更好地起草), 或者降低有效的 $T_{\text{verify}}$ (更智能地验证). 作者明确指出了三个努力方向:

**让学生写得更快** (降低 $T_{\text{draft}}$).

**让学生猜得更准** (提高 $\tau$).

**让教授查得更巧** (降低 $T_{\text{verify}}$).

但是已有的方法的一些限制:

![图片](assets/26654163b1d0.png)

| 特征 | 自回归起草模型 (Eagle3) | 并行起草模型 (DFlash) |
|---|---|---|
| 工作方式 | 逐 token 串行生成 | 一次前向传播生成所有位置 |
| 草稿质量 $\tau$ | 高(有上下文依赖) | 低(独立生成, 后缀衰减) |
| 起草速度 $T_{\text{draft}}$ | 慢($O(\gamma)$) | 快($\approx$ 常数) |
| 核心瓶颈 | 起草速度 | 质量受限 |

DSpark 的解法是:**融合两者优势, 同时引入系统级调度来避免资源浪费**.

### 2.1 半自回归生成 (Semi-Autoregressive Generation)

DFlash一个并行起草模型在一次前向传播中产生所有 $\gamma$ 个草稿的logits, 因此每个预测都不能以块内其他位置已采样的 token 为条件. 当上下文允许多种合理的续写时, 例如"of course"和"no problem", 并行起草模型可能会产生不连贯的组合, 如"of problem"或"no course", 因为每个位置都是对所有可能的前置 token 进行边缘化, 而不是以实际采样的那一个为条件. 因此, 接受率会沿着草稿块迅速衰减, 浪费了起草和验证的计算资源. 因此, 作者采用一种半自回归结构, 将草稿生成分为两个阶段:

![图片](assets/3939c058c923.jpg)
并行阶段 (Parallel stage)
一个并行主干网络(在我们的实例中是DFlash)在整个块上运行一次前向传播, 产生隐藏状态 $H$ 和基础logits. 作者只对原始的DFlash主干做了一个微小的修改: 不输入一个锚点 token 加上 $\gamma$ 个掩码 token 并只预测掩码位置, 而是将锚点本身视为第一个预测位置, 因此 $\gamma$ 个输入 token (锚点 + $\gamma-1$ 个掩码)产生 $\gamma$ 个草稿logits. 这在保持相似草稿质量的同时, 减少了起草的计算量.

![图片](assets/465d70c99178.png)
串行阶段 (Sequential stage)
串行阶段为基础logits补充一个依赖于前缀的转移偏差, 允许每个草稿位置以块内先前已采样的 token 为条件. 串行阶段并非定义一个全局归一化的能量模型, 而是通过自回归分解的方式导出一个因果的块分布:

$$P(X\mid x_0)=\prod_{k=1}^{\gamma}p_k(x_k\mid x_0,x_{<k}),\quad p_k(x_k\mid x_0,x_{<k})=\frac{\exp(U_k(v)+B_k(x_0,x_{<k},v))}{\sum_{v'}\exp(U_k(v')+B_k(x_0,x_{<k},v'))}$$

$x_0$ 表示上一轮验证周期中目标模型生成的锚点 token

$U_k$ 是并行主干在位置 $k$ 产生的基础logit向量

$V$ 是词汇表.

在推理时, 串行模块根据 $B_k(x_0,x_{<k},v)$ 从左到右进行采样. 由于这个采样过程本质上是串行的, 该模块的计算必须是轻量级的 ($T_{\text{sequential}}\ll T_{\text{parallel}}$), 以便总的起草延迟仍然由并行阶段主导. 我们在下面描述该串行模块的两种实现.
马尔可夫头 (Markov head)
最简单的实现将 $B_k$ 的依赖限制在紧邻的前一个 token $x_{k-1}$ 上, 将其简化为一阶转移. 原则上这是一个完整的 $|V|\times|V|$ 矩阵; 用一个低秩分解来近似它: $B=W_2 W_1$, 其中 $W_1\in\mathbb{R}^{|V|\times r}$ 且 $W_2\in\mathbb{R}^{r\times|V|}$. 给定前一个 token $x_{k-1}$, 位置 $k$ 的转移偏差为:

$$B(x_{k-1},\cdot)=W_2[W_1[x_{k-1}]]\in\mathbb{R}^{|V|}$$

其中:

$W_1\in\mathbb{R}^{|V|\times r}$ 作为一个嵌入查找表

$W_2\in\mathbb{R}^{r\times|V|}$ 作为一个logit投影

低秩分解 (默认) 使得存储和每步计算都保持很小, 即使对于大词汇表, 串行循环也能保持高效. 回到前面的例子: 一旦位置1采样了"of", 马尔可夫头就会在位置2提升"course"的概率并抑制"problem"的概率, 从而减轻了跨模态冲突.**代码实现** (`deepspec/modeling/dspark/markov_head.py`):

```
class VanillaMarkov(nn.Module):    def __init__(self, *, vocab_size: int, markov_rank: int):        super().__init__()        self.vocab_size = int(vocab_size)        self.markov_rank = int(markov_rank)        # W1: 嵌入查找表 [vocab_size, r]        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)        # W2: logit 投影 [r, vocab_size]        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)    def compute_step_bias(self, token_ids, hidden_states):        """计算单步转移偏差 B(x_{k-1}, ·)"""        # 查找前一个 token 的马尔可夫嵌入        prev_emb = self.markov_w1(token_ids.long())  # [batch, r]        # 投影到词汇表空间        return self.markov_w2(prev_emb)  # [batch, vocab_size]    def apply_step_logits(self, logits, *, token_ids, hidden_states):        """将转移偏差加到基础 logits 上"""        return logits + self.compute_step_bias(token_ids, hidden_states)
```

代码中还发现门控马尔可夫头的实现, 在 VanillaMarkov 的基础上引入门控机制,利用主干隐藏状态:

```
class GatedMarkovHead(VanillaMarkov):    def __init__(self, *, vocab_size, markov_rank, hidden_size):        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)        # 门控投影: 结合隐藏状态和前一token嵌入        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)    def compute_step_bias(self, token_ids, hidden_states):        prev_embeddings = self.get_prev_embeddings(token_ids)        # 计算门控值        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)        gate = torch.sigmoid(self.gate_proj(gate_inputs))        # 门控后的偏差        return self.project_bias(gate * prev_embeddings)
```

**串行采样过程**:

```
def sample_block_tokens(self, base_logits, *, first_prev_token_ids, ...):    """从左到右串行采样整个草稿块"""    sampled_tokens = []    corrected_logits = []    prev_token_ids = first_prev_token_ids.long()    for step_idx in range(proposal_len):        # 应用马尔可夫偏差修正当前位置的 logits        step_logits = self.apply_step_logits(            base_logits[:, step_idx, :],            token_ids=prev_token_ids,            hidden_states=...,        )        corrected_logits.append(step_logits)        # 采样当前位置的 token        next_token_ids = sample_tokens(step_logits, temperature=temperature)        sampled_tokens.append(next_token_ids)        # 将采样结果传递给下一步        prev_token_ids = next_token_ids    return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)
```
RNN头 (RNN head)
马尔可夫头在一步之后就没有记忆了, 位置 $k$ 无法访问 $x_{k-2}$ 之前的 token . RNN头通过维持一个循环状态 $s_k\in\mathbb{R}^r$ 来放宽这个限制, 该状态累积了块内的完整前缀历史. 在每一步, 该模块将当前状态 $s_{k-1}$, 前一个 token 的嵌入 $W_1[x_{k-1}]$, 以及主干网络的隐藏状态 $h_k$ 拼接成一个输入向量, 然后应用一次门控更新:

$$s_k=g\odot s_{k-1}+(1-g)\odot\tanh(W_c z_k)$$
$$B_k(x_{k-1},\cdot)=W_2^\top\tanh(W_o z_k)$$

其中 $W_c,W_o,W_g\in\mathbb{R}^{(2r+d)\times r}$ 由一个单一的线性投影联合参数化, 该投影被分割为门、候选和输出三个部分. 状态 $s_0$ 初始化为零.

**代码实现**:

```
class RNNHead(VanillaMarkov):    def __init__(self, *, vocab_size, markov_rank, hidden_size):        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)        # 联合投影: [s_{k-1}; W1[x_{k-1}]; h_k] -> [gate; candidate; output]        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)    def _rnn_step(self, state, prev_embeddings, hidden_states):        """单步 RNN 更新"""        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)        proj = self.joint_proj(z)        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)        gate = torch.sigmoid(gate_raw)        candidate = torch.tanh(candidate_raw)        # GRU-like 状态更新        new_state = gate * state + (1.0 - gate) * candidate        # 计算 logit 偏差        bias = self.project_bias(torch.tanh(output_raw))        return new_state, bias
```

最后代码提供了一个工厂方法, 根据配置构建对应的马尔可夫头:

```
def build_markov_head(config) -> nn.Module | None:    markov_rank = int(config.markov_rank)    if markov_rank == 0:        return None    markov_head_type = str(config.markov_head_type).lower()    if markov_head_type == "vanilla":        return VanillaMarkov(vocab_size=config.vocab_size, markov_rank=markov_rank)    if markov_head_type == "gated":        return GatedMarkovHead(...)    if markov_head_type == "rnn":        return RNNHead(...)
```

渣注
我们可以将词汇表的logits空间看作一个高维向量空间. 并行主干在每个位置 生成一个点. 串行阶段引入了一个**离散时间动力系统**.

马尔可夫头定义了一个映射, 使得新的logits.

RNN头则定义了一个更复杂的系统, 其中状态演化, 并且.

从微分几何的角度, 如果我们将概率分布的单纯形(simplex)视为一个流形, 那么并行主干生成了一系列"孤立"的基准点. 串行头则是在这个流形上施加了一个依赖于路径历史的**向量场(vector field)**, 将每个基准点沿着该向量场进行微小的"平移"(通过), 使其移动到流形上一个更好的位置(更接近目标分布).

而低秩分解 $B=W_2 W_1$ 是一个经典的数学技巧. 它假设完整的转移矩阵 $B\in\mathbb{R}^{|V|\times|V|}$ 是低秩的, 即 token 间的转移关系可以由一个更小的"概念"空间(维度为 $r$)来描述. $W_1$ 将一个 token 映射到这个概念空间, $W_2$ 再从这个概念空间映射回完整的logits空间. 这本质上是一种压缩表示, 大大降低了模型的参数量和计算复杂度.

### 2.2 置信度调度验证 (Confidence-Scheduled Verification)

半自回归架构使 DSpark 能高效生成大的草稿块,但**不加选择地验证整个草稿块会降低系统吞吐量**:

**数据维度**:代码等结构化任务接受率高,开放式聊天接受率低

**系统维度**:高并发下,验证被拒绝的 token 会占用宝贵的批处理容量

![图片](assets/3e48b99b6a7e.png)

因此, 要完全释放大草稿块的潜力, 需要一个统一的机制, 将目标模型的计算资源仅导向那些具有正预期回报的 token . DSpark通过耦合一个预测前缀存活概率的**置信度头**, 和一个根据当前系统负载动态确定最优验证长度的**硬件感知前缀调度器**来实现这一点.
置信度头 (Confidence Head)
置信度头为每个草稿位置 $k$ 输出一个标量估计值 $c_k\in(0,1)$. 至关重要的是, $c_k$ 建模的是在块内所有先前 token 都已被接受的条件下, 位置 $k$ 的草稿 token 能够通过目标模型验证的**条件概率**. 该架构采用一个轻量级的线性投影后接一个sigmoid函数:

$$c_k=\sigma(w^\top[h_k;W_1[x_{k-1}]])$$

其中 $h_k$ 是主干网络的Hidden State, $W_1[x_{k-1}]$ 是来自前一个草稿 token 的马尔可夫嵌入. 使用每一步的解析接受率 $a_k^*$ 来监督. 这个率由草稿分布 $p_k^d$ 和目标分布 $p_k^t$ 之间的总变差距离决定:

$$a_k^*=1-\frac{1}{2}\|p_k^d-p_k^t\|_1$$

代码实现:

```
class AcceptRatePredictor(nn.Module):    """置信度头:预测每个位置的条件接受概率"""    def __init__(self, input_dim: int):        super().__init__()        self.proj = nn.Linear(int(input_dim), 1)    def forward(self, features):        return self.proj(features).squeeze(-1)
```

在模型中的使用 :

```
# 置信度头初始化if self.enable_confidence_head:    input_dim = int(config.hidden_size)    if self.confidence_head_with_markov:        input_dim += config.markov_rank  # 加入马尔可夫嵌入特征    self.confidence_head = AcceptRatePredictor(input_dim=input_dim)# 推理时预测置信度def predict_confidence_step(self, hidden_states, prev_token_ids=None):    if self.confidence_head_with_markov:        prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids)        features = torch.cat([hidden_states, prev_embeddings], dim=-1)        return self.confidence_head(features).float()    return self.confidence_head(hidden_states).float()
```

**后处理校准 (Post-hoc Calibration).** 与基于阈值的验证启发式方法(这些方法只要求置信度分数能正确排序草稿 token 的质量)不同, 硬件感知调度方法精确地需要累积接受概率的**绝对值**来计算预期的接受长度. 由于神经网络的置信度估计常常过于自信, 直接使用原始置信度分数会扭曲吞吐量估计, 导致次优的调度.

为了解决这个问题, 引入了**序列化温度缩放 (Sequential Temperature Scaling, STS)**. 因为每个 $c_k$ 建模的是一个条件概率, 链式法则决定了一个草稿前缀被接受的联合概率可以分解为累积乘积. 使用一个留出的验证集, STS从左到右依次校准这个联合概率. 具体来说, 在每个位置, 作者进行一个简单的1D网格搜索来找到最优的温度标量, 以最小化累积乘积的期望校准误差(ECE), 同时保持所有先前位置已校准的分数固定. 关键的是, 温度缩放是一个保序变换: 它在纠正预测概率以匹配经验接受率的同时, 不会扰乱置信度头学到的相对草稿 token 排名.

说人话就是, 在学生写完每个草稿 token 后, 预测这个 token 被教授接受的概率. 比如, 对"of course", 它可能会预测出很高的置信度. 但是作者发现, 神经网络天生"迷之自信", 预测的概率总偏高. 所以他们用了一个叫"序列化温度缩放"的技术给它"降降温", 让它预测的概率更接近真实情况. 这对于后面的精确计算至关重要.

硬件感知前缀调度器 (Hardware-Aware Prefix Scheduler)
先前的方法通常对置信度分数应用一个静态阈值来决定验证长度. 虽然在孤立的、单请求的假设下有效, 但静态阈值在动态的服务系统中可能是次优的, 因为在这些系统中, 验证一个草稿 token 的效用在很大程度上取决于当前的系统负载.

为了解决这个问题, 我们将验证长度选择问题形式化为一个**全局吞吐量最大化问题 (算法1)**. 考虑一个包含 $r$ 个活动请求的批次. 对于请求 $i$, 令 $c_{i,1},\dots,c_{i,\gamma}$ 为每个位置的置信度估计, 令 $\ell_i$ 表示被调度的验证长度. 由于投机解码只动态地接受连续前缀的草稿 token , 一个在位置 $j$ 的 token 的存活概率是累积乘积 $a_{i,j}=\prod_{k\le j}c_{i,k}$.

![图片](assets/e3aa6f86b26b.png)

在单次验证步骤中, 发送到目标模型的总批次大小(以 token 计)为 $B=\sum_{i=1}^{r}(1+\ell_i)$, 预期的成功接受 token 数为 $\sum_{i=1}^{r}\sum_{j=1}^{\ell_i}a_{i,j}$. 令 $\text{SPS}(B)$ 表示引擎对于给定前向传播批次大小 $B$ 的吞吐量, 以每秒步数(steps per second)计. 关键是, 这个容量曲线在引擎初始化时分析一次, 并存储为一个轻量级的成本表. 调度器旨在通过动态选择验证长度 $\ell_i$ 来最大化预期的系统范围 token 吞吐量.

虽然寻找 的全局最大值看起来像一个组合搜索, 但目标函数的结构允许一个高效的贪心解法. 因为 相对于 是单调不增的 (即), 将请求 的验证长度从 扩展到 所带来的预期接受 token 数的边际增益恰好是. 这种单调性确保了按 全局排序候选 token 自然地尊重了块内的前缀依赖关系. 因此, 如果总的验证批次大小 是固定的, 最优的分配 将通过从所有 的全局池中贪心地选择具有最高存活概率的草稿 token 来确定.

基于这一洞见, 可以在这条贪心准入路径上评估优化过程. 首先全局地按存活概率降序排列所有有效的前缀扩展. 为了动态确定最优的目标批次大小, 作者从这个排序池中增量地准入 token , 通过从预先分析的成本表中进行 查找来更新预期吞吐量.

无损投机解码严格要求**非预见性 (non-anticipating property)**: 准入决策不能依赖于未来的候选 token . 因为置信度头依赖于先前采样 token 的马尔可夫特征, 计算下一个存活概率 明确需要实例化的候选项. 因此, 一个回顾性的全局搜索会无意中将 泄露到第 步的准入决策中, 引入选择偏差

为了强制严格的因果性, 调度器(算法1)采用了一种**提前停止机制**. 通过在吞吐量下降时()立即中断贪心搜索, 截断决策仅依赖于截至该精确步骤已处理的前缀. 这将准入事件与未来的 token 隔离开来, 确保了对目标分布的精确恢复. 注意, 这种步进式的提前停止仅在目标函数 是单峰的情况下才能产生全局最大吞吐量, 这隐含地假设了硬件容量曲线是平滑衰减的.

渣注
这是整篇文章中最关键的一部分, 传统的投机解码通常只会展示单batch下的吞吐, 并没有系统性的思考. **硬件感知调度器**解决了一个非常实际的问题: "现在有一堆用户的请求等着处理, 我该让教授为每个请求验证多长的草稿, 才能让整个系统的总产出(每秒生成的总 token )最高?"

它的决策过程是这样的(如算法1所示):

**收集信息**: 收集所有等待处理的请求的草稿, 以及每个草稿 token 的(校准后)存活概率.

**全局排序**: 把所有请求的所有草稿 token , 按照"存活概率"从高到低排一个大队.

**模拟"加人"**:

从最基本的验证开始(每个请求只验证一个 token ). 计算一下此时的系统总吞吐量.

然后, 按照排序, 把队伍里存活概率最高的那个 token "加"到验证任务里. 此时, 验证的总 token 数`B` 增加1.

查一下"硬件性能表"(`SPS(B)`), 看看验证`B`个 token 时系统每秒能跑几轮.

重新计算新的总吞吐量.

**寻找峰值**: 如果新的吞吐量比之前高, 就继续"加人". 如果发现吞吐量开始下降了, 说明"人加得太多, 系统开始拥堵了", 就立刻停止, 采用上一步那个能达到最高吞吐量的验证方案.

这个调度器的精妙之处在于, 它把"验证多长"这个抽象问题, 转化成了一个**基于真实硬件性能和概率估计的、可计算的优化问题**. 它不再是拍脑袋设一个固定阈值, 而是动态地为整个系统寻找当前负载下的"最佳工作点".
**调度器的目标函数**
这是一个在组合优化中常见的目标形式. 它不是简单地最大化收益($\sum a_{i,j}$)或最小化成本(与 $B$ 成正比), 而是最大化它们的比率, 即效率 $\dfrac{\text{SPS}(B)\cdot\sum_{i,j}a_{i,j}}{1}$.
**贪心算法的正确性**
论文指出, "单调性确保了按 $a_{i,j}$ 全局排序...是有效的". 我们可以更深入地分析这一点. 每次贪心选择, 都是选择使"边际收益/边际成本"最大的那一项. 这里的边际收益是 $a_{i,j}$, 边际成本是使总批次大小B增加1所带来的吞吐量下降 (即 $\text{SPS}(B)-\text{SPS}(B+1)$). 算法1的简化贪心策略之所以有效, 是因为它假设了所有边际成本(增加一个token)是相同的, 于是问题简化为最大化边际收益, 即贪心地选择 $a_{i,j}$ 最大的项.
**因果性与鞅理论**
"非预见性"原则在数学上与**鞅(Martingale)** 的概念密切相关. 一个随机过程是鞅, 如果其在未来的期望值等于现在的观测值. 投机解码的无损性要求最终采样的分布不受未来信息的影响. 如果调度决策依赖了 $x_{i,j+1}$, 那么对 $x_{i,j}$ 的采样过程就不是一个纯粹的、基于 $x_{<j}$ 的过程, 其条件期望会发生改变, 从而破坏了鞅的性质, 导致分布偏差. 算法1中的`break`确保了决策只依赖于"过滤(filtration)"中到当前为止的信息, 维持了过程的正确性.

附录有一个因果性反例详解, 解释为什么不能去掉`break`.

### 2.3 训练

在训练期间, 作者从每个目标序列中随机采样多个锚点位置, 以形成 $\gamma$ token 块作为训练数据. 目标模型在整个训练过程中保持冻结; 草稿模型共享其嵌入层和语言建模头并保持它们冻结, 只更新主干起草模型、串行模块和置信度头.

训练目标由三项组成: 一个交叉熵损失, 一个分布匹配损失, 和一个置信度损失. 这三者都通过 $w_k=\exp(-(k-1)/\gamma)$ 进行位置加权, 这强调了在基于前缀的验证下对预期接受长度贡献更多的块中较早的位置.

交叉熵损失 $L_{ce}$ 训练起草模型预测正确的下一个 token :

$$L_{ce}=-\sum_k w_k\log p_k^d(x_k^*)$$

其中 $x_k^*$ 是真实 token , $p_k^d$ 是草稿分布. 分布匹配损失 $L_{l1}$ 惩罚草稿分布和目标分布之间的总变差距离:

$$L_{l1}=\sum_k w_k\|p_k^d-p_k^t\|_1$$

由于总变差距离是接受率的直接代理: 每步接受概率等于 $1-\frac{1}{2}\|p_k^d-p_k^t\|_1$, 最小化 $L_{l1}$ 直接最大化了预期的接受率.

置信度损失 $L_{conf}$ 是一个二元交叉熵, 训练置信度头预测来自公式的软接受标签:

$$L_{conf}=-\sum_k w_k[a_k^*\log c_k+(1-a_k^*)\log(1-c_k)]$$

总的目标函数是这三项的加权组合 (默认权重):

$$L=\alpha_{ce}L_{ce}+\alpha_{l1}L_{l1}+\alpha_{conf}L_{conf}$$

### 2.4 实验

#### 2.4.1 为什么并行生成能超越自回归?

并行起草模型(DFlash)和半自回归起草模型(DSpark)通常能产生比完全自回归的起草模型(Eagle3)更长的接受长度. 这一发现与标准的预期相悖, 即逐步的自回归会比并行模型产生更高质量的序列.作者进行了详细的分析.

![图片](assets/6c229248284d.png)

**位置1的容量优势.** 在第一个草稿位置, 两种架构都仅根据目标上下文来预测下一个 token. 这里的性能差异完全源于**架构容量**: 像Eagle3这样的自回归模型由于其 的延迟, 被限制在浅层网络, 而 的并行起草模型可以负担得起更深的网络. 这种结构上的差距在位置1产生了显著的准确率优势, DFlash的起点明显高于Eagle3 (例如, 在数学上是0.88对0.81, 在聊天上是0.72对0.53). 由于投机解码作为一个严格的前缀匹配生存过程运行, 第一个 token 具有最高的杠杆作用, 这里的拒绝会立即导致整个块失效. 因此, 这种初始的容量优势不成比例地提升了最终的接受长度, 解释了为什么并行起草模型尽管在后续位置接受率快速衰减, 但在全球范围内最终仍能胜过自回归模型.

**后续位置独立性的局限.** 检查曲线的尾部(位置2到7)暴露了独立并行生成的内在局限性. 随着早期的 token 锁定了一个特定的语义路径, 后续的 token 自然变得更可预测. 像Eagle3这样的自回归模型有效地利用了这种条件确定性, 在块的更深处保持甚至增加了条件接受率(例如, 在聊天上从0.53到0.74). 相比之下, DFlash遭受了快速的接受率衰减, 在代码上从0.87降至0.78, 在聊天上从0.72降至0.63. 因为每个并行位置都是对所有可能的先前 token 进行边缘化, 而不是以一个确切采样的前缀为条件, 模型频繁地提出不一致的后缀组合——这种模式被称为**多模态冲突**.

**通过半自回归缓解后缀衰减.** 前述分析突出了一个清晰的架构目标: 将并行主干的高容量用于初始 token , 并将自回归模型的依赖建模用于后续 token . 这直接激发了DSpark的半自回归设计. 如图所示, DSpark继承了深度并行起草模型的高初始接受率(例如, 在数学上从0.93开始). 同时, 其轻量级的串行头缓解了并行生成典型的快速接受率衰减. 通过解决这一权衡, DSpark在整个草稿块中保持了高而稳定的条件接受率.

#### 2.4.2 一点点自回归大有裨益

![图片](assets/bdab3ebf68a9.png)

实验证明, 哪怕只给DSpark一个很浅的串行头(2层), 它的表现也超过了更深的全并行模型(5层DFlash). 这说明半自回归这种设计"性价比"极高. 加上那个小小的串行头后, 总的写稿时间只增加了1%左右, 几乎可以忽略不计, 但换来的却是高达30%的草稿质量提升.

#### 2.4.3 置信度头的作用

虽然DSpark在长草稿块上能维持高接受率, 但验证整个提议仍然是低效的. 由于固有领域差异, 开放式聊天中的尾部 token 仍然面临高拒绝风险, 使得盲目验证成为对目标计算资源的浪费.

**诊断: 静态阈值扫描 (Diagnostic: Static Threshold Sweep).** 下图绘制了在不同置信度阈值下的平均每步 token 数(柱状图)和总体验收率(折线图). 随着阈值增加, 接受率稳步上升, 因为估计器过滤掉了那些最终会被拒绝的 token (带阴影的柱条). 这表明置信度头能够识别价值较低的后缀 token , 并且这种修剪在聊天工作负载上最为显著, 因为在聊天中, 更高熵的 token 分布限制了固定长度验证的效率. 在"Chat"子图中, 提高阈值显著减少了被拒绝的 token , 将接受率从45.7%提高到95.7%. 相比之下, 结构化任务(数学和代码)经历了较温和的修剪并保留了更多的草稿 token , 接受率分别从76.9%提高到92.5%和67.6%到92.0%.

![图片](assets/da0df7d13ceb.png)

**从静态阈值到校准调度 (From Static Thresholds to Calibrated Scheduling).** 虽然对诊断有用, 但静态阈值在动态服务环境中是次优的, 因为它忽略了系统负载: 在低并发下验证低置信度 token 只产生极小的机会成本, 但在高并发下则浪费了关键的批处理容量. 这种对负载的依赖性激发了硬件感知前缀调度器. 正如论文3.2节所阐述的, 要最大化系统级吞吐量, 需要置信度模型同时展现出**强大的预测区分能力(strong predictive discrimination)** 和**精确的校准(precise calibration)**, 以准确估计累积存活概率.

置信度头 是对转移概率 的一个估计.下图的可靠性图是在**概率空间**中进行的分析. x轴是模型预测的概率, y轴是经验观察到的频率. "完美校准"对应于对角线.

![图片](assets/0822d03478d8.png)

ECE (Expected Calibration Error) 是一个度量预测分布与经验分布之间差异的指标, 可以看作是两条曲线之间的加权L1距离..

STS (Sequential Temperature Scaling): 温度缩放 是一个在logit空间进行的线性变换, 对应于在概率单纯形上施加的一个非线性变换. 它在保持序关系的同时, 将概率分布"推向"或"拉离"均匀分布, 从而修正其"自信程度". STS的序列化特性, 确保了对联合概率 的校准, 而非仅仅对单个 的校准.

上图的表明, 虽然原始模型实现了强大的区分能力(ROC-AUC从0.81到0.90), 但它**过于自信(overly confident)** (ECE 3%-8%). 应用后处理的STS缓解了这种过度自信, 将平均ECE降低到约1%, 产生了可靠的存活估计.

### 2.5 在线服务

#### 2.5.1 可扩展和灵活的训练

DSpark 草稿模型与 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 的预览版本协同部署. 将最大草稿块大小设置为, 并使用马尔可夫头进行序列建模. 此外, 置信度头与草稿模型一同进行端到端训练, 并随后通过 STS 进行校准, 以提供可靠的调度信号.

训练草稿模型需要目标模型的输出分布作为监督信号. 在完整的文档上下文中评估两个模型会产生巨大的内存占用和工作节点间的通信开销. 为了解决这些瓶颈, 作者在内部训练框架(HAI-LLM)中实现了两个系统级优化:

**隐层状态通信 (Hidden state communication)**: 在并行工作节点之间传输目标模型的全词汇表 logits ($O(|V|)$) 会造成严重的带宽瓶颈. 作为替代, 临时缓存目标模型前向传播的激活值, 并且只通信语言建模(LM)头之前的隐层状态. LM 头的投影计算随后只在草稿模型所在的工作节点上, 针对被采样出的目标位置局部执行. 这将每个 token 的通信复杂度降低到 $O(d)$, 其中 $d$ 是隐层维度.

**基于锚点的序列打包 (Anchor-bounded sequence packing)**: 为了将草稿模型的计算成本与目标模型的上下文长度解耦, 从训练序列中采样固定数量的草稿锚点, 并将这些孤立的预测块打包成密集的训练批次. 作者们通过 token 级别的注意力索引而非标准的二维掩码来管理这种打包. 这在多个独立的序列和锚点之间维持了精确的因果掩码, 避免了与标准填充相关的计算和内存开销.

渣注
**隐层状态通信**: 这是一个非常聪明且实用的优化. 与其传递最终的10万维 logits, 不如传递生成 logits 前的最后一步, 也就是几千维的隐层状态`hidden state`. 这个`hidden state` 包含了所有必要信息. 学生拿到`hidden state` 后, 自己在本地用 LM head (一个矩阵乘法) 算出最终的 logits. 这大大减少了网络传输的数据量, 从 降到了, 其中.

**基于锚点的序列打包**: 这个技巧解耦了老师和学生的计算负担. 老师可以读一篇长篇小说, 但学生只需要关注老师读到某几个关键"锚点"位置时, 后面该怎么续写5个词. 训练框架将这些分散在各处的"续写任务"打包在一起, 形成一个密集的批次, 让 GPU 高效处理. 避免了大量的无用计算和内存浪费(padding).

具体训练代码详细分析参考附录.

#### 2.5.2 实践中的硬件感知前缀调度器

算法1提供了一个理论上可靠且无损的调度机制. 然而, 将这个算法直接部署到生产环境中, 会暴露出现实世界基础设施的两个根本性冲突.

该算法假设了一个平滑、单峰的性能容量曲线, 而真实的硬件容量 本质上是离散的, 呈现出锯齿状、阶梯式的下降趋势.

该算法要求对每一步的动态草稿 token 进行调度, 这与连续的 CUDA graph 回放和零开销调度 (ZOS, Zero-Overhead Scheduling)存在冲突.

为了在系统兼容性、吞吐量和算法正确性之间进行权衡, 作者将调度器调整为**异步操作**. 由于 ZOS 要求下一步的批处理大小在当前步骤完成之前就必须知道, 同步调度将不可避免地导致 GPU 流水线停顿. 作为替代, 作者使用**两步之前**的置信度头输出来**近似**下一个验证容量.

具体机制上, 当前步骤的候选 token 仍然严格按照它们实际的、最新的累积置信度分数进行排序; 而来自两步前的历史预测,**仅仅**用于决定动态截断的长度 (即批处理容量限制). 这有效地将准入过程转变为一个动态的 top- 选择. 虽然近似容量 引入了轻微的时间偏差, 但选择机制在根本上是保序的: 最有信心的草稿 token 总是被优先验证. 这种适配完全隐藏了调度延迟, 并确保了与 ZOS 的无缝集成.

基于这个异步流水线, 解决了硬件利用率的瓶颈. 为了防止调度器被锯齿状的 SPS 悬崖困在局部最优解中, 作者**移除了 early-stopping 的 break 条件**, 从而实现无约束的全局搜索. 通常情况下, 这种回顾性搜索会泄露未来的 token 信息并违反无损保证 (附录 A). 然而, 由 ZOS 驱动的适配方案天然地防止了这个问题. 因为无约束搜索只评估来自两步之前的历史预测, 所以准入决策与当前 token 的实现是隔离的. 截断长度内在地只依赖于两步之前可用的信息. 因此,**异步设计形成了一道因果屏障**, 在跨越硬件悬崖以最大化物理吞吐量的同时, 保持了对目标分布的精确恢复.

渣注
这个细节做的非常扎实. 算法1假设增加一个 token 的验证, 系统吞吐量的变化是平滑可预测的. 但现实是, GPU 性能曲线是"阶梯状"的, 增加一个 token 可能会让 batch size 刚好跨过某个硬件处理的"坎", 导致性能急剧下降. 算法1的贪心策略很容易陷入这种"局部最优"的陷阱. 另一方面算法1需要知道当前步所有候选 token 的信息才能做决策. 但像 ZOS 这样的现代推理引擎为了极致的效率, 采用了"流水线"作业, 在当前步还没算完的时候, 就必须知道下一步要处理多少任务 (batch size). 这就产生了时序冲突, 强行等待会导致 GPU 空闲, 流水线断裂.

他们引入了一个**两步延迟的异步机制**.
决策什么 (Which ones)?
用最新的信息. 在当前步, 依然用实时计算出的置信度分数来对所有候选 token 进行排序, 保证最好的 token 排在最前面.
决策多少 (How many)?
用旧的信息. 使用**两步前** 的数据来预测一个大概的"容量限制". 比如, 两步前系统看起来很闲, 就预测一个大的; 两步前系统很忙, 就预测一个小. 然后, 从当前排好序的候选 token 里, 无脑取出前 个.

这个设计带来了两个巨大的好处:

**解决了时序冲突**: 决定"How many" () 的计算可以在两个周期前就完成, 完美配合 ZOS 流水线, 没有任何等待和停顿.

**意外地保证了算法无损性**: 附录A中证明, 如果你回头看当前步的结果来决定是否截断, 会引入偏差, 破坏原始分布. 但 DSpark 的异步设计, 决定截断长度 时, 看的是两步前的数据, 与当前步生成的具体 token 内容完全无关. 这就像一道"因果屏障", 把决策和结果隔离开了. 这使得作者可以大胆地去掉`early-stopping`, 进行全局搜索来找到真正的吞吐量最大点, 而不必担心破坏数学上的正确性.

#### 2.5.3 高吞吐和低延迟推理

在解码期间, 生产服务系统必须同时优化两个相互竞争的目标:`单个请求的延迟`和`聚合的吞吐量`. 前者决定了单个用户的服务质量, 这个因素在基于 Agent 的工作负载中日益关键. 而后者决定了可以同时服务的用户总数. 因为推测解码不可避免地会产生浪费的验证计算, 它内在地就在这个权衡中导航, 用额外的系统计算换取更快的单请求生成速度.

然而, 在作者的部署环境中, 每一步处理的请求数量经常受到资源限制(例如, 每个请求固定的 KV-cache 容量)和可用用户流量池(例如, RL 长尾负载)的约束. 因此,**有效的批处理大小持续地远低于 GPU 的计算饱和阈值**. 在这种机制下, 传统的权衡被简化了: 给定一个固定的并发限制,**最大化每 GPU 的总 token 吞吐量**和**最大化每个用户的生成速度 (tok/s/user)** 变成了高度相关的目标, 而非相互竞争.

为了达到这个最大吞吐量, 异步调度器主动地将空闲计算资源导向最有希望的草稿 token. 然而, 执行这种动态路由在物理执行层引入了一个严峻的挑战: 推理框架必须在一个批次内高效地支持可变长度的查询. 标准的解码核为固定查询长度进行了高度优化; 简单地处理可变长度的验证前缀会导致严重的 GPU 利用率不足, 这是由于填充和不均匀的工作负载分布.

作者通过**将物理执行与逻辑序列追踪解耦**来解决这个问题. 在计算 kernel 中, 来自不同请求的所有 token 被**展平**并作为独立的元素被同等处理. 复杂的序列内依赖关系则严格地通过一个集成到稀疏注意力实现中的**标记张量 (marker tensor)** 来传达. 具体到 DeepSeek-V4 架构上, 只有 index-attention 和 compress kernel需要修改来支持这种可变长度路由, 从而使得动态调度器能够无缝运作, 而不引入底层执行开销.

解决方案: 解耦物理与逻辑
**逻辑上**: 每个请求还是一个独立的序列.

**物理上**: 把所有请求的所有待处理 token (包括原始 token 和草稿 token) 全部"倒"在一起, 形成一个巨大的一维"token 池".

**连接**: 使用一个额外的"标记张量"来记录每个 token 属于哪个原始请求的哪个位置.

**计算**: GPU 对这个"token 池"进行统一、密集的计算. 当需要计算注意力等依赖上下文的操作时, 再通过"标记张量"查找到正确的依赖关系.

#### 2.5.4 线上用户流量下的性能

在 DeepSeek-V4-Flash (预览版) 和 DeepSeek-V4-Pro (预览版) 的生产服务引擎中, 对 DSpark-5 (配置最大草稿长度) 与 MTP-1 基线进行了评估. MTP-1 代表了以前的生产设置, 在 DeepSeek-V4 预览版发布两周后被 DSpark 取代. 这个单 token 设置在历史上一直维持在生产中, 是因为部署一个静态的多 token 草稿模型 (例如 MTP-3/5) 在高并发下会由于过度的验证开销而严格地降低聚合吞吐量. 因此, 将 DSpark 与这个已建立的基线进行比较, 直接展示了其在动态服务环境中安全解锁更长草稿块性能潜力的能力. 在所有图中, 散点代表直接从真实用户流量中采样的原始遥测数据, 捕捉了复杂的、真实世界的请求分布, 而实线代表拟合的性能前沿.

![图片](assets/9d00bcad7f58.png)

**服务帕累托前沿 (The Serving Pareto Frontier)**. 上图阐释了聚合系统吞吐量和单用户生成速度(交互性)之间的权衡. 为了量化 DSpark 在实际部署约束下的行为, 作者在几个交互性 SLA 锚点上评估系统. 这里, SLA 指定了系统必须保证的最低单用户生成速度(单位: tok/s/user).

对于 V4-Flash 引擎, 在 80 和 120 tok/s/user 的 SLA 锚点上评估系统. 在中等的 80 tok/s/user SLA 下, DSpark 比 MTP-1 基线提升了 51% 的聚合吞吐量. 更严格的 120 tok/s/user SLA 代表了一个性质上不同的区间: 在此约束下, 单 token 的 MTP-1 基线接近其操作边界, 只能维持非常小的并发批次. 因此, 此时的相对吞吐量比率在数值上很大, DSpark 实现了名义上 661% 的更高聚合吞吐量. 因此, 作者主要将这个高 SLA 点解释为 DSpark 扩展了可行的交互性前沿的证据, 而非在一个充分利用的基线上具有代表性的倍数级加速. 在匹配的实际吞吐量水平上(这提供了更稳定的比较), DSpark 将单用户的生成速度加快了 60% 到 85%.

V4-Pro 的部署呈现了相同的模式. 在中等的 35 tok/s/user SLA 下, DSpark 提升了 52% 的聚合吞吐量. 在更严格的 50 tok/s/user SLA下, MTP-1 再次进入低并发区间, 使得 DSpark 获得了名义上 406% 的相对吞吐量优势. 与 V4-Flash 一样, 我们将此点视为一个迹象, 表明 DSpark 在基线无法有效支持的交互性目标下, 仍能维持有用的吞吐量. 在匹配的系统容量下, DSpark 提供了 57% 到 78% 更快的单用户生成速度. 总体而言, 这些结果表明 DSpark 将观测到的吞吐量-交互性前沿向外推移: 它在中等 SLA 区域提升吞吐量, 更重要的是, 在严格的交互性约束下保持了非退化的服务能力.

**负载下的吞吐量动态 (Throughput Dynamics under Load)**. 下图通过绘制聚合吞吐量(顶行)和动态验证预算(底行)随系统并发度的变化, 分析了驱动这些增益的潜在机制.

![图片](assets/81142c1439aa.png)

生产部署典型的中等并发区域(V4-Flash 少于200个并发请求, V4-Pro 少于150个), 硬件感知调度器通过分配更长的验证预算来利用可用的目标计算能力, 将 MTP-1 的静态2个 token 扩展到大约每个请求 4-6 个 token. 这种扩展的验证在每次前向传播中产生更多的接受 token, 直接贡献了在帕累托前沿上观察到的吞吐量增益.

随着系统并发度扩展和目标容量饱和, 调度器会动态地限制这个预算. 平均验证长度随着负载平滑地减少, 确保了低置信度的草稿 token 在消耗关键的批处理容量之前就被剪枝. 这种负载感知行为稳定了生产部署: DSpark 在轻度流量下最大化了空闲计算的效用, 同时在重度流量下有效地保留了关键的批处理容量.

**局限性 (Limitations)**. 尽管前缀调度器最小化了浪费的目标模型验证, DSpark 仍然需要通过并行主干网络生成初始的-token 块, 产生固定的草稿端成本. 对于那些接受率天生就很低的复杂查询, 这种预先的草稿计算是无法挽回的. 未来的优化可以引入草稿模型内部的、基于难度的提前退出机制, 使得这类请求可以绕过全块生成.

## 3. 结语

### 3.1 投机解码的性能模型

投机解码的**第一性原理**可以用一个公式精确描述:

$$L=\frac{T_{\text{draft}}+T_{\text{verify}}}{\tau}$$

其中:

$L$: 平均每个生成 token 的墙上时钟延迟(wall-clock latency)

$T_{\text{draft}}$: 草稿模型生成 $\gamma$ 个候选 token 所需的时间

$T_{\text{verify}}$: 目标模型在单次前向传播中并行验证所有候选的时间

$\tau$: 每个解码周期实际被接受的 token 数量(期望值)

相比标准自回归解码的单步延迟 $T_{\text{verify}}$ (即目标模型生成一个 token 的时间), 加速比为:

$$S=\frac{\tau\cdot T_{\text{verify}}}{T_{\text{draft}}+T_{\text{verify}}}$$

假设每个位置 $i$ 的条件接受概率为 $\alpha_i$ (即在前 $i-1$ 个 token 均被接受的条件下, 第 $i$ 个 token 被接受的概率), 则:

$$\tau=1+\sum_{i=1}^{\gamma}\prod_{j=1}^{i}\alpha_j$$

在 i.i.d. 简化假设下(各位置接受率独立同分布为 $\alpha$), 这简化为一个截断几何级数:

$$\tau=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

关键性质:

当 $\alpha\to1$ (草稿几乎完美匹配目标)时, $\tau\to\gamma+1$, 即所有候选均被接受

当 $\alpha\to0$ 时, $\tau\to1$, 退化为标准自回归(每周期仅获得一个 bonus token)

$\tau$ 对 $\alpha$ 极其敏感: $\alpha$ 从 0.7 提升到 0.9, 当 $\gamma=10$ 时, $\tau$ 从 2.94 跃升到 5.22

从性能公式出发, 有且仅有三条路径:

![图片](assets/cba1ed1cd45a.png)

杠杆1和杠杆2之间存在根本性的架构权衡:

**自回归起草**: $T_{\text{draft}}=O(\gamma)$ (线性增长), 但 $\alpha$ 高(序列连贯)

**并行起草**: $T_{\text{draft}}=O(1)$ (与 $\gamma$ 无关), 但 $\alpha$ 低(后缀衰减)

**半自回归(DSpark)**: $T_{\text{draft}}\approx O(1)$ (接近并行), $\alpha$ 介于两者之间

杠杆3独立于前两者, 但在高并发场景下成为**决定性因素**: 即使 $\tau$ 很高, 不加选择地验证长草稿块仍会因为占用系统批处理容量而降低全局吞吐量.

另一方面, 投机解码的加速效果对 batch size 敏感:

**Batch=1(延迟优先)**: 目标模型严重 memory-bound, GPU 算力大量闲置. 投机解码通过将 $\gamma+1$ 个 token 打包为一次前向传播, 将算术强度从 $O(1)$ 提升到 $O(\gamma)$, 趋向 compute-bound, 加速最显著.

**大Batch(吞吐优先)**: GPU 已趋于 compute-bound, 额外验证 token 不再"免费". 此时每增加一个验证 token 的边际成本为正, 投机解码的"免费验证"假设失效. 这正是 DSpark 置信度调度的核心动机.

### 3.2 投机解码的数学模型

投机解码的接受率 $\alpha$ 与草稿分布 $q$ 和目标分布 $p$ 之间存在精确的数学关系:

$$\alpha=1-D_{TV}(p,q)$$

其中全变差距离(Total Variation Distance)定义为:

$$D_{TV}(p,q)=\frac{1}{2}\sum_x|p(x)-q(x)|$$

**几何直觉**: $\alpha$ 等于两个分布在概率空间中的"重叠面积". 当 $p=q$ 时, $\alpha=1$, 完美接受; 当 $p$ 和 $q$ 的支撑集完全不重叠时, $\alpha=0$.

从最优传输(OT)的视角, 我们可以将"分布对齐"问题形式化为:

$$W_1(p,q)=\inf_{\gamma\in\Pi(p,q)}\mathbb{E}_{(x,y)\sim\gamma}[d(x,y)]$$

其中 $W_1$ 是 Wasserstein-1 距离, $\Pi(p,q)$ 是所有边际为 $p$ 和 $q$ 的联合分布.

**与投机解码的联系**: 虽然投机解码直接使用 TV 距离(而非 Wasserstein 距离)来定义接受率, 但 OT 视角提供了更深层的理解:

**TV 距离是 OT 的特殊情况**: 当度量空间为离散的 $d(x,y)=\mathbb{1}[x\ne y]$ 时, $W_1(p,q)=D_{TV}(p,q)$.

**分布对齐的"运输代价"**: 草稿模型训练的本质是**最小化将 $q$ 运输到 $p$ 的代价**. 在投机解码中, 这等价于最大化接受率:

$$\max_q\alpha=\max_q(1-D_{TV}(p,q))$$

**逐位置的链式分解**: 在位置 $k$, 条件接受率为:

$$\alpha_k=1-D_{TV}(p_k,q_k)=\sum_x\min(p_k(x),q_k(x))$$

DSpark 论文(公式8)正是用此作为置信度头的监督信号:

$$a_k^*=1-\frac{1}{2}\|p_k^d-p_k^t\|_1$$

在实际系统中, 我们面临一个**算力受限的最优传输问题**:

$$\max_{q\in\mathcal{Q}_C}\ \mathbb{E}[1-D_{TV}(p,q)]$$

其中 $\mathcal{Q}_C$ 是在计算预算 $C$ 下可实现的草稿模型集合. 这不是一个标准的 OT 问题, 因为:

**模型容量约束**: 草稿模型的参数量/层数直接决定了其表达能力上界

**延迟约束**: $T_{\text{draft}}$ 必须远小于 $T_{\text{verify}}$, 否则投机解码无收益

**逐位置非独立**: 位置 $k$ 的接受率依赖于前缀路径

DSpark 的训练损失函数直接反映了 OT 视角下的对齐目标:

$$L=\alpha_{ce}L_{ce}+\alpha_{l1}L_{l1}+\alpha_{conf}L_{conf}$$

其中:

$L_{l1}$ 直接最小化全变差距离(即最大化接受率)

$L_{ce}$ 辅助对齐 mode(确保 argmax 正确)

权重 $\alpha_{l1}>\alpha_{ce}$ 表明分布对齐比点预测更重要

这揭示了投机解码训练的**第一性原理**: 草稿模型的训练目标不应是"预测下一个正确 token", 而应是"使整个输出分布尽可能接近目标模型"(分布匹配目标).

另外在DSpark 的实验揭示了一个关键现象: 并行起草模型的接受率沿位置快速衰减. 从 OT 视角, 这是因为**条件分布的运输代价随深度增加**. DSpark 半自回归的 OT 解释为: 串行头注入的转移偏差 $B_k$ 相当于在 OT 问题中引入了一个**条件运输映射**, 将并行主干输出的边缘分布 $q_k$ 沿着实际采样路径进行"投影", 从而减少与目标条件分布 $p_k$ 的运输距离.

### 3.3 调度策略设计考量

调度策略回答的核心问题是: **给定一批草稿 token, 应该提交多少给目标模型验证?** 形式化为优化问题:

$$\max_{\{\ell_i\}}\ \text{SPS}(B)\cdot\sum_{i=1}^{r}\sum_{j=1}^{\ell_i}a_{i,j},\quad B=\sum_{i=1}^{r}(1+\ell_i)$$

其中:

$r$: 活动请求数

$\ell_i$: 请求 $i$ 的验证长度

$\sum_{i,j}a_{i,j}$: 预期接受 token 总数

$B$: 总验证 batch 大小

$\text{SPS}(B)$: 硬件在 batch $B$ 下的每秒步数(throughput profile)

置信度调度(DSpark)核心思想: 让草稿模型自身评估每个位置的接受概率, 基于此动态决定验证长度.
置信度头设计

输入: 并行主干隐藏状态 $h_k$ + 前一 token 的马尔可夫嵌入 $W_1[x_{k-1}]$

输出: 位置 $k$ 在所有前序 token 被接受条件下的条件接受概率 $c_k$

监督信号: $a_k^*=1-\frac{1}{2}\|p_k^d-p_k^t\|_1$ (解析接受率)
**前缀存活概率**
由链式法则:

$$a_{i,j}=\prod_{k=1}^{j}c_{i,k}$$

$a_{i,j}$ 是草稿前缀 $(x_{i,1},\dots,x_{i,j})$ 被完整接受的联合概率. DSpark 的动态调度: 将所有请求的候选 token 按 $a_{i,j}$ 全局排序, 贪心准入直到系统吞吐量不再增加:

$$\text{Throughput}(B)=\text{SPS}(B)\cdot\sum_{i,j}a_{i,j}$$

$\sum_{i,j}a_{i,j}$ 随准入 token 增加而增加(收益)

$\text{SPS}(B)$ 随 $B$ 增加而下降(成本)

最优点即为"边际收益 = 边际成本"

在高并发下, 每个验证 slot 具有**机会成本**: 它本可以用于服务另一个活动请求. 设系统有 $r$ 个活动请求, 为请求 $i$ 多验证一个 token 的边际收益和成本分别为:

预期增加的接受数 $\Delta_{\text{gain}}=a_{i,\ell_i+1}$

吞吐量下降 $\Delta_{\text{cost}}=\text{SPS}(B)-\text{SPS}(B+1)$

DSpark 的贪心算法本质上在做: 当 $a_{i,\ell_i+1}\cdot\text{SPS}(B+1)<\text{当前吞吐量}$ 时停止准入.

调度策略的理论极限来看, 对于一个接受率为 $\alpha$ 的理想系统, 无论调度策略如何, 加速比的理论上界为:

$$S_{\max}=\frac{1}{1-\alpha}$$

当 $\gamma\to\infty$

**系统约束下的实际上界**: 在并发度 $r$ 和 GPU 容量 $C$ 的约束下:

$$S_{\text{actual}}=\min\left(\frac{1}{1-\alpha},\ \frac{C}{r}\right)$$

当 $r$ 较大时, 第二项成为瓶颈, 这正是调度策略需要解决的问题.

参考资料

[1]
DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation:*https://github.com/deepseek-ai/DeepSpec*
[2]
Accelerating Large Language Model Decoding with Speculative Sampling:*https://arxiv.org/abs/2302.013180*
[3]
Fast Inference from Transformers via Speculative Decoding:*https://arxiv.org/abs/2211.17192*
[4]
Accelerating Large Language Model Decoding with Speculative Sampling:*https://arxiv.org/abs/2302.01318*
[5]
Speculative Decoding with Big Little Decoder:*https://arxiv.org/abs/2302.07863*
[6]
Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding:*https://arxiv.org/pdf/2309.08168*
[7]
Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads:*https://arxiv.org/abs/2401.10774*
[8]
Break the Sequential Dependency of LLM Inference Using Lookahead Decoding:*https://arxiv.org/abs/2402.02057*
[9]
EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty:*https://arxiv.org/abs/2401.15077*
[10]
EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees:*https://arxiv.org/abs/2406.16858*
[11]
EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test:*https://arxiv.org/abs/2503.01840v3*
[12]
Better & Faster Large Language Models via Multi-token Prediction:*https://arxiv.org/pdf/2404.19737*
[13]
DFlash: Block Diffusion for Flash Speculative Decoding:*https://arxiv.org/abs/2602.06036*

### 附录 A: 投机解码发展史 — 时间维度的全景分析

`这段有点懒了, 是AI生成的`

投机解码从2022年末的理论萌芽到2026年中大规模生产部署, 仅用了不到四年. 这四年可划分为六个清晰的演进阶段, 每个阶段都由一组核心矛盾的发现与解决驱动.

#### 第一阶段: 奠基(2022.11)— 理论与框架的建立

投机解码的诞生源于一个优雅的观察:**自回归解码的瓶颈在于串行性而非计算量**. Leviathan et al. (Google, 2022.11) 在 ICML 2023 上发表的奠基性工作首次提出了完整的 draft-then-verify 框架: 用一个轻量草稿模型串行生成 个候选 token, 再由目标大模型在单次前向传播中并行验证, 通过拒绝采样严格保持目标分布.

该框架引入了三个支配投机解码全部后续研究的核心参数:

(接受率): 由 draft 与 target 分布的全变差距离决定,

(成本比): 草稿模型与目标模型的单步推理时间比

(草稿长度): 每轮投机的候选 token 数

加速比公式 精确刻画了三者间的权衡关系, 为后续所有优化提供了统一的理论语言.

**这一阶段的根本矛盾**: 草稿质量与草稿成本不可兼得. 独立小模型自回归起草虽理论简洁, 但 的串行瓶颈从一开始就埋下了后续范式革命的种子.

#### 第二阶段: 探索与分化(2023.05–2023.12)— 多路径并行与草稿来源多样化

2023年是投机解码的"寒武纪大爆发". 研究者从两个方向突破 Leviathan 框架的局限.

**方向一: 验证侧的树状并行化**. SpecInfer (CMU/北大, 2023.05) 率先将树状推测引入验证阶段——多个 SSM 各自生成候选链, 组织为树结构后由目标模型通过 Tree Attention 单次并行验证. 这打破了"一轮只能验证一条链"的假设, 验证效率大幅提升. 同年 BiLD (UC Berkeley, 2023.05) 从另一角度优化验证: 引入置信度驱动的动态 Fallback/Rollback 机制和确定性 argmax 验证, 避免不必要的拒绝采样开销.

**方向二: 摆脱独立草稿模型**. 2023年下半年见证了草稿来源的急剧多样化:

**自投机**(2023.09): Self-Speculative Decoding (浙大) 发现目标模型自身层间存在冗余——跳过若干层生成的"早退"表征即可作为草稿,**零额外模型、零额外内存**

**蒸馏增强**(2023.10): Online SD (UC Berkeley) 和 DistillSpec (Google) 分别从在线和离线角度通过知识蒸馏提升草稿与目标的分布对齐度

**检索替代**(2023.11): REST (北大/Princeton) 激进地提出**用数据库检索完全替代参数化草稿模型**, 免训练、免维护

**Jacobi 范式**(2023.11): Lookahead Decoding (UCSD) 和 Ouroboros (清华) 将自回归解码重新诠释为非线性方程组的 Jacobi 迭代求解——利用迭代轨迹中的 n-gram 作为免费草稿

**特征层预测**(2023.11): PaSS (EPFL/Apple) 提出 Look-ahead 嵌入, 在单模型内实现并行猜测

**级联加速**(2023.12): Cascade SD (UIUC) 递归地用更小模型为草稿模型生成草稿, 形成"模型的模型"的套娃结构

**这一阶段的根本矛盾**: 草稿来源的多样性带来了"选择困难"——不同方法在质量、成本、通用性上各有优劣, 缺乏统一的比较框架. 这直接催生了下一阶段的系统性综述和头部收敛.

#### 第三阶段: 系统化与质量突破(2024.01–2024.04)— 多头预测、最优树与综述

2024年初出现了三个里程碑式的进展, 标志着投机解码从"百花齐放"走向"精耕细作".

**Medusa (Princeton/CMU, 2024.01)** 开创了"模型自身多头预测"范式: 在目标模型最后隐藏层上附加 个轻量级解码头(每个仅单层 FFN + 残差), 各头独立并行预测未来不同位置的 token. 配合 Tree Attention 和 Typical Acceptance, Medusa 以极低的参数代价(~6%)实现了 2.2-2.8× 加速. 更重要的是, 它**证明了模型自身蕴含的预测能力足以替代独立草稿模型**, 开启了"自投机"研究方向.

**EAGLE (北大/微软, 2024.01)** 将投机解码的质量推向新的高度. 核心洞察是:**token 级别的不确定性源于特征级别的不确定性**. 与其在 token 空间建模, 不如在目标模型的隐藏特征空间中进行自回归——将顶层特征与 token 嵌入拼接后输入单层 Transformer, 以因果链方式逐步预测. 这种特征空间自回归的建模能力远超独立位置预测, 接受率大幅领先 Medusa. EAGLE 奠定了后续 EAGLE-2(动态 draft tree)、EAGLE-3(Training-Time Test)的演进基础, 成为自回归草稿范式的标杆.

**Sequoia (CMU/Together, 2024.02)** 在验证侧实现了理论完备的突破. 通过将树结构构造形式化为动态规划问题, Sequoia 证明了在给定节点预算下存在唯一的最优树拓扑；其无放回采样验证算法更是**唯一同时满足最优传输性质和覆盖性质**的验证策略, 跨温度(0-1.0)全范围保持优势. 在 Offloading 场景下达到 9.5× 加速, 为投机解码的树验证设立了理论标准.

同期,**Speculative Streaming (Apple, 2024.02)** 和**SPACE (云天励飞, 2024.02)** 从架构融合角度探索了投机与验证的边界消解——两者都在单次前向传播中同时完成生成和验证, 彻底消除串行等待.

此外,**Survey of Speculative Decoding (港理工/北大, 2024.01)** 提供了首个系统性综述, 建立了统一分类学, 为领域提供了清晰的研究地图. 而**ConsistentEE (华南理工, 2024.04)** 的 Early Exit 方法虽非严格投机解码, 但其 RL 难度引导的退出策略为后续动态调度提供了重要参照.

**这一阶段的根本矛盾**: 起草质量已大幅提升, 但自回归起草器仍受 的串行约束——即使单步极快(1层), 生成 个 token 仍需 7 次串行前向传播. 这成为限制草稿块规模的根本瓶颈.

#### 第四阶段: 训练集成与范式分裂(2024.04–2024.12)— 从后训练到预训练

**Meta MTP (Meta FAIR, 2024.04)** 提出了一个具有深远影响的问题:**为什么不在预训练阶段就赋予模型投机能力？** 通过添加 个独立输出头同时预测未来 个 token, 多 token 预测不仅提升了模型在下游任务(尤其是代码生成)上的质量, 还让这些输出头在推理时直接充当投机解码的草稿——实现了训练增强与推理加速的统一.

关键发现是"Usefulness only at scale"——多 token 预测的收益随模型规模增长, 小模型反而略差. 这解释了该方法此前被忽视的原因, 也为后续大规模实践提供了指导.

**DeepSeek-V3 MTP (DeepSeek-AI, 2024.12)** 在 Meta MTP 的基础上做出关键改进: 将并行独立头改为**顺序因果链**(每步以前一步输出为条件), 更贴近 EAGLE 的特征自回归理念. 更重要的是, DeepSeek-V3 将 MTP 定位为"训练为主、推理复用"——训练时密集化信号提升数据效率, 推理时 MTP 模块直接作为草稿模型. 这是投机解码从"后训练附加"到"预训练内置"的里程碑.

**这一阶段的根本矛盾**: 训练集成带来了完美的分布对齐(联合训练), 但 D=1 的保守设置限制了单轮加速潜力——如何在不大幅增加训练成本的前提下支持更长草稿块？

#### 第五阶段: 范式革命(2025)— 从串行到并行的

**DFlash (UC San Diego, 2025, ICML 2026)** 完成了投机解码最根本的范式突破:**起草延迟从 降为**.

这一突破建立在两个关键创新之上:

**并行扩散起草**: 将草稿模型重新定位为"目标模型的扩散适配器"——输入锚点 token + 个 MASK token, 块内所有位置双向注意力, 单次前向传播一次性输出 个草稿 token

**KV 注入条件化**: 在草稿模型的**每一层**将目标模型的多层隐藏状态拼接到 K 和 V 中, 使每层都能直接访问目标信息——解决了 EAGLE 输入端融合的"信息逐层衰减"问题

 延迟意味着草稿模型可以突破 1 层的自回归瓶颈, 使用更深的架构(5 层)和更大的块(). 在 Qwen3-8B 上达到 4.86× 加速, 是 EAGLE-3(2.02×)的约 2.5 倍.

**这一阶段的根本矛盾**: 并行生成带来了 的速度, 但位置独立预测导致的"多模态碰撞"使得接受率沿块快速衰减——**并行性牺牲了因果性**. 此外, 长块中低置信度后缀的无差别验证在高并发下严重浪费 batch 容量.

#### 第六阶段: 融合与落地(2026.01–2026.06)— 因果性回归、系统调度与生产部署

2026年是投机解码的"集大成"之年. 四条线索交织推进, 共同将领域推向成熟.

**线索一: 因果性与并行性的统一**. JetSpec (UC San Diego, 2026.06) 直面 DFlash 的核心矛盾——通过设计**树形因果注意力掩码**, 在单次前向传播中同时实现了并行预测和因果条件化: 每个树节点可关注其祖先但不能关注兄弟分支. 这打破了"并行 = 牺牲因果性"的固有假设, 在 Qwen3-8B 上达到 9.64× 加速(比 DFlash+DDTree 的最强基线高 10%), 证明了**因果性和并行性可以兼得**.

**线索二: 扩散最优树的构建**. DDTree (Technion, 2026.04) 在 DFlash 的基础上解决了"并行生成的丰富位置分布信息被浪费"的问题——利用 DFlash 天然产生的逐位置边际分布, 通过 Best-First Heap 算法构建最优草稿树, 在 Tree Attention 下实现单次并行验证. 它将 DFlash 的 5.56× 提升至 7.52×(+35%), 证明了**扩散模型与树验证天然互补**.

**线索三: 系统级调度优化**. SSD/Saguaro (Stanford/Princeton, 2026.03) 将优化视角从算法层提升到系统层——核心观察是: 验证结果的可能空间虽大, 但概率集中在少数结果上. 通过在验证进行时预测可能结果并预先准备投机序列(投机缓存), 缓存命中时完全消除起草等待. SSD 理论上严格优于标准 SD(Corollary 8), 在实践中相对最强 SD 基线平均加速 30%. 它证明了**系统级调度优化与算法优化完全正交**——可与任意起草策略组合.

**线索四: 半自回归与自适应调度的统一**. DSpark (北大/DeepSeek-AI, 2026.06) 是这一阶段的集大成者. 它同时解决了第五阶段遗留的两个核心矛盾:

**生成质量**: 半自回归架构——并行骨干保持 起草延迟, 轻量 Markov/RNN 串行头注入块内 token 间依赖, 缓解 suffix decay

**系统效率**: 置信度调度验证——置信度头估计每位置存活概率, 硬件感知前缀调度器将验证长度选择形式化为全局吞吐量最大化问题, 根据实时系统负载动态裁剪验证前缀

更重要的是, DSpark**首次将并行/半自回归投机解码部署到大规模生产环境**(DeepSeek-V4), 在真实用户流量下验证了算法的实用性——匹配吞吐量下用户生成速度提升 60-85%, 严格 SLA 下维持基线无法达到的吞吐水平.

#### 演进主线总结

回顾四年发展史, 投机解码的演进沿四条相互缠绕的主线展开:

![图片](assets/f99532e27277.png)

**四条主线**:

**草稿来源**: 独立模型 → 模型自身(跳层/多头/特征) → 检索/Jacobi → 训练内置 → 扩散适配器 → 半自回归

**验证方式**: 单序列拒绝采样 → 树验证 → 最优树(DP) → 扩散树 → 因果树 → 硬件感知自适应调度

**起草延迟**: 串行 → 并行 → 并行+因果 → 并行+依赖+自适应

**系统集成**: 离线 benchmark → 训练-推理统一 → 异步调度 → 生产流量部署

### 附录 B: DSpark训练代码解析

训练时的`forward` 方法(以`Qwen3DSparkModel.forward`为例)是 DSpark 最核心的逻辑, 它在一次调用中完成了从锚点采样到损失所需全部输出的计算. 下面逐步详解:
步骤 0: 输入说明
```
def forward(    self,    input_ids: torch.Tensor,            # [bsz, seq_len] 完整训练序列的 token ids    target_hidden_states: torch.Tensor,  # [bsz, seq_len, m*d] 目标模型多层隐藏状态拼接    loss_mask: torch.Tensor,             # [bsz, seq_len] 哪些位置参与损失计算    target_last_hidden_states: Optional[torch.Tensor] = None,  # [bsz, seq_len, d] 目标模型最后一层hidden) -> DSparkForwardOutput:
```

`input_ids`: 由目标模型生成的完整回复序列(用作 ground truth)

`target_hidden_states`: 目标模型在各`target_layer_ids` 处的隐藏状态拼接(用于 KV 注入)

`target_last_hidden_states`: 目标模型 LM 头前一层的输出(用于计算目标分布, 作为 损失的监督信号)
步骤 1: 锚点采样
```
anchor_positions, block_keep_mask = sample_anchor_positions(    seq_len=seq_len, loss_mask=loss_mask,    num_anchors=self.num_anchors,  # 默认 512    device=device,)
```

从序列中随机选取`num_anchors` 个有效锚点. 每个锚点代表一个-token 草稿块的起始位置.

采样逻辑(见`common.py` 的`sample_anchor_positions`):

筛选有效候选: 位置 有效当且仅当`loss_mask[i] > 0.5` 且`loss_mask[i+1] > 0.5`(确保锚点及其第一个预测目标都在有效范围内)

对有效位置赋随机值, 无效位置赋 2.0(排到最后)

按随机值排序, 取前`num_anchors` 个

最终排序使锚点按位置递增(便于后续 attention mask 构建)

`block_keep_mask`标记哪些块是真实采样到的(vs 填充的 dummy 块)
步骤 2: 创建噪声嵌入(草稿输入)
```
noise_embedding = create_noise_embed(    self.embed_tokens, input_ids, anchor_positions, block_keep_mask,    mask_token_id=self.mask_token_id, block_size=self.block_size,)
```

对于每个锚点块, 构造输入 token 序列:`[anchor_token, MASK, MASK, ..., MASK]`, 具体逻辑(见`common.py` 的`create_noise_embed`):

```
# 初始化: 所有位置填充 mask_token_idnoise_ids = torch.full((bsz, num_blocks * block_size), mask_token_id)# 每个块的第一个位置替换为锚点 tokennoise_ids[batch_idx, block_starts] = anchor_tokens  # anchor_tokens = input_ids[anchor_positions]# 嵌入查找return embed_tokens(noise_ids)  # → [bsz, num_blocks * block_size, hidden_dim]
```

这与论文描述一致: "将一个锚点 token 的嵌入加上 个掩码 token 的嵌入作为输入". 但 DSpark 做了一个微小改动, 锚点本身也作为第一个预测位置, 因此 个输入产生 个草稿 logits.
步骤 3: 构建位置编码
```
context_position_ids = torch.arange(seq_len)           # [0, 1, ..., seq_len-1]draft_position_ids = create_position_ids(anchor_positions, self.block_size)# 对每个锚点: [anchor_pos, anchor_pos+1, ..., anchor_pos+block_size-1]full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
```

草稿 token 的 position_id 与它们在原始序列中的实际位置对齐, 确保旋转位置编码(RoPE)正确.
步骤 4: 构建注意力掩码
```
dspark_attn_mask = create_dspark_attention_mask(    anchor_positions=anchor_positions, block_keep_mask=block_keep_mask,    seq_len=seq_len, block_size=self.block_size, device=device,)
```

这个掩码是 DSpark 训练的关键设计. 它控制每个草稿 token 能"看到"什么:

```
def dspark_mask_mod(b, h, q_idx, kv_idx):    q_block_id = q_idx // block_size    anchor_pos = anchor_positions[b, q_block_id]    # 规则1: 可以注意到锚点之前的上下文(因果)    mask_context = (kv_idx < seq_len) & (kv_idx < anchor_pos)    # 规则2: 可以注意到同一块内的所有草稿 token(双向)    mask_draft = (kv_idx >= seq_len) & (q_block_id == kv_block_id)    # 规则3: 只有有效块才参与计算    return (mask_context | mask_draft) & block_keep_mask[b, q_block_id]
```

注意:**块内是双向注意而非因果** , 这正是并行起草的核心. 所有 个位置互相可见, 使得并行主干在一次前向传播中就能产生所有位置的隐藏状态.
步骤 5: 并行主干前向传播
```
output_hidden = self._forward_backbone(    position_ids=full_position_ids,    noise_embedding=noise_embedding,        # 草稿输入    target_hidden_states=target_hidden_states,  # 目标模型特征(KV注入)    attention_mask=dspark_attn_mask,)
```

`_forward_backbone` 的内部逻辑:

```
def _forward_backbone(self, *, position_ids, noise_embedding, target_hidden_states, ...):    hidden_states = noise_embedding    # KV注入: 将目标模型多层隐藏状态投影到草稿模型维度    target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))    # 计算旋转位置编码    position_embeddings = self.rotary_emb(hidden_states, position_ids)    # 逐层通过 DSpark Decoder Layer    for layer in self.layers:  # 5层        hidden_states = layer(            hidden_states=hidden_states,            target_hidden_states=target_hidden_states,  # 每层都注入            attention_mask=attention_mask,            position_embeddings=position_embeddings,        )    return self.norm(hidden_states)
```

每一层的`Qwen3DSparkAttention` 都将`target_hidden_states`(来自目标模型)拼接到 Key/Value 中:

```
k = torch.cat([k_proj(target_hidden_states), k_proj(hidden_states)], dim=1)  # [context; draft]v = torch.cat([v_proj(target_hidden_states), v_proj(hidden_states)], dim=1)
```

即KV 注入:.
步骤 6: 提取目标 token 与构建 prev_token_ids(用于串行头)
```
# 每个锚点的预测目标: anchor_pos+1, anchor_pos+2, ..., anchor_pos+block_sizelabel_indices = anchor_positions.unsqueeze(-1) + torch.arange(1, block_size+1)target_ids = torch.gather(input_ids, ...)  # ground truth token ids# 构建 teacher-forcing 的前一 token 序列:# prev_token_ids[k] = 位置 k 的 "前一个 token"# 对位置0: 前一个是锚点自身# 对位置k>0: 前一个是 ground truth target_ids[k-1]prev_token_ids = torch.cat(    [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)
```

这里使用**teacher-forcing**: 串行头在训练时看到的 "前一个 token" 是 ground truth, 而非模型自己的采样结果. 这确保了梯度稳定和高效并行训练.
步骤 7: 计算基础 logits 并应用马尔可夫头
```
# 基础 logits: 并行主干的输出通过 LM 头draft_logits = self.compute_logits(output_hidden).reshape(    bsz, num_blocks, self.block_size, -1)  # [bsz, num_blocks, block_size, vocab_size]# 应用马尔可夫头: 加上转移偏差 B_kif self.markov_head is not None:    draft_logits = self.markov_head.apply_block_logits(        draft_logits,        token_ids=prev_token_ids,      # teacher-forcing 的前一 token        hidden_states=output_hidden_4d, # 主干隐藏状态(RNN头需要)    )
```

这实现了论文公式(4):

训练时,`apply_block_logits` 对 RNN 头会展开整个 block_size 的循环(teacher-forced):

```
# RNNHead.apply_block_logits:state = torch.zeros(...)  # s_0 初始化为零for k in range(block_size):    prev_emb = self.get_prev_embeddings(token_ids[..., k])  # W1[x_{k-1}]    h_k = hidden_states[..., k, :]                          # 主干隐藏状态    state, bias = self._rnn_step(state, prev_emb, h_k)      # GRU更新+输出偏差    output_logits.append(base_logits[..., k, :] + bias)
```
步骤 8: 对齐目标模型 logits(用于 损失)
```
if target_last_hidden_states is not None:    # 目标模型在位置 anchor_pos, anchor_pos+1, ..., anchor_pos+block_size-1 的预测    target_pred_indices = (safe_label_indices - 1).clamp(min=0)    aligned_target_hidden = torch.gather(target_last_hidden_states, ...)    aligned_target_logits = self.compute_logits(aligned_target_hidden)
```

这一步从目标模型的最后一层隐藏状态中提取与草稿块对齐的位置, 然后通过**共享的 LM 头**计算目标分布. 用于计算总变差距离:.
步骤 9: 构建 eval_mask
```
eval_mask = build_eval_mask(    seq_len=seq_len, loss_mask=loss_mask,    label_indices=label_indices, safe_label_indices=safe_label_indices,    block_keep_mask=block_keep_mask,)
```

`eval_mask` 决定哪些位置参与损失计算. 关键操作:

```
eval_mask = target_valid & (target_loss_mask > 0.5) & block_keep_mask# 累积乘积: 一旦某位置无效, 之后全部屏蔽return eval_mask.to(torch.int32).cumprod(dim=-1).bool()
```

`cumprod` 确保了**连续前缀语义**: 与投机解码推理时一致 — 第一个拒绝点之后的所有位置不再计算损失.
步骤 10: 计算置信度预测
```
if self.confidence_head is not None:    if self.confidence_head_with_markov:        # 输入 = [主干隐藏状态; 马尔可夫嵌入]        prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids)        confidence_features = torch.cat([output_hidden_4d, prev_embeddings], dim=-1)        confidence_pred = self.confidence_head(confidence_features).float()    else:        confidence_pred = self.confidence_head(output_hidden_4d).float()
```

这实现了论文公式(7):

置信度头的输入是主干隐藏状态 和前一 token 的马尔可夫嵌入 的拼接, 通过一个线性层+sigmoid 输出条件接受概率估计.
步骤 11: 返回输出
```
return DSparkForwardOutput(    draft_logits=draft_logits,           # [bsz, num_blocks, block_size, vocab] 草稿分布    target_ids=target_ids,               # [bsz, num_blocks, block_size] ground truth    eval_mask=eval_mask,                 # [bsz, num_blocks, block_size] 有效前缀掩码    block_keep_mask=block_keep_mask,     # [bsz, num_blocks] 有效块掩码    confidence_pred=confidence_pred,     # [bsz, num_blocks, block_size] 置信度logits    aligned_target_logits=aligned_target_logits,  # [bsz, num_blocks, block_size, vocab] 目标分布)
```

这些输出随后被传入`compute_dspark_loss()` 计算三部分损失:

`draft_logits` +`target_ids` →

`draft_logits` +`aligned_target_logits` →

`confidence_pred` + 由`aligned_target_logits` 计算的接受率 →

损失代码实现 (`loss.py`):

```
def compute_dspark_loss(*, outputs, loss_decay_gamma, ce_loss_alpha,                        l1_loss_alpha, confidence_head_alpha):    # 位置衰减权重    positions = torch.arange(block_size, device=device)    decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))    loss_weight_mask = eval_mask.float() * decay_weights    # 交叉熵损失    loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")    ce_loss = (loss_per_token * flat_weights).sum() / flat_weights.sum()    # L1分布匹配损失(总变差距离)    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)    l1_dist = (draft_probs - target_probs).abs().sum(dim=-1)    l1_loss = (l1_dist * loss_weight_mask).sum() / loss_weight_mask.sum()    # 置信度损失    confidence_targets = accept_rate_3d.detach()  # 解析接受率作为监督    confidence_loss = F.binary_cross_entropy_with_logits(        outputs.confidence_pred.float(), confidence_targets, reduction="none"    )    # 加权组合    total_loss = ce_loss_alpha * ce_loss + l1_loss_alpha * l1_loss                 + confidence_head_alpha * confidence_loss    return total_loss
```