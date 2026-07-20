# DeepSeek-V4详细分析(2): MegaMoE

> 作者: zartbot  
> 日期: 2026年5月14日 10:57  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498412&idx=1&sn=103dd28729b39dd6313eb664bfca3a80&chksm=f995ea6ecee26378d1d7995fb1adca18617d26614b409d60f3b97c271e465b950c6266d807b0#rd

---

### TL;DR

前一篇分析了DeepSeek-V4的算法和模型结构[《DeepSeek-V4详细分析(1): 算法和模型结构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498131&idx=1&sn=f9405408342c0d97cd6255a72cdde2e7&scene=21#wechat_redirect), 接下来我们分析技术报告的第三章和基础设施相关的内容, 篇幅比较长因此也会拆成几篇, 这一篇专门介绍MegaMoE. MegaMoE通过细致的Overlap 通信和计算的延迟, 整体性能提升了1.5x~1.9x, 下面是DeepSeek-V4-Pro在不同batchSize下的测试结果:

![图片](assets/c16eadfb1e69.png)

本文目录如下

```
1. Overview
1.1 通信延迟隐藏分析
1.2 细粒度EP方案
2. Legacy EP实现
2.1 EP的计算和通信流
2.2 Legacy EP实现
3. MegaMoE实现
3.1 整体架构
3.2 Scheduler
3.2.1 启发式配置
3.2.1.1 Block大小选择
3.2.1.2 Pool容量
3.2.1.3 Expert Wave粒度
3.2.1.4 SMEM分布及流水线深度估计
3.2.2 详细的调度流程
3.3 Buffer Layout
3.3.1 Workspace
3.3.2 Buffer
4. 详细的代码分析
4.1 Dispatch Warp
4.1.1 统计本 SM 发往 expert i 的 token 数
4.1.2 本地计数 → 全局偏移
4.1.3 写远端src_token_topk_idx
4.1.4 SM0聚合
4.1.5 NVLink Barrier
4.1.6 Pull token
4.1.6.1 初始化
4.1.6.2 主循环
4.1.6.3 Min-Peeling 算法
4.1.6.4 Pull token
4.1.6.5 Pull尾处理
4.2 TMA Producer A Warp
4.2.1 Scheduler
4.2.2 Tensor_map 处理
4.2.3 等待数据到达机制
4.2.3.1 L1等待机制
4.2.3.2 L2等待机制
4.2.4 TMA加载数据
4.3 TMA Producer B Warp
4.4 MMA Warp
4.4.1 2-CTA UMMA
4.4.2 AB Swap
4.4.4 MMA Config
4.4.5 Per-lane描述符
4.4.6 MMA指令 shape 静态检查
4.4.7 动态更新 UMMA_N 值
4.4.9 Empty Barrier Arrive
4.4.8 TMEM 双缓冲
4.4.10 K 循环
4.5 Epilogue Warp
4.5.1 初始化阶段
4.5.2 和Dispatch Warp同步
4.5.3 Block循环
4.5.4 L1 Epilogue
4.5.4.1 任务如何切片的
4.5.4.2 加载 topk_weight 权重
4.5.4.3 TMEM 加载
4.5.4.4 TMEM 释放
4.5.4.5 SwiGLU激活计算
4.5.4.6 Amax 归约
4.5.4.7 FP8量化
4.5.4.9 写入L2 input SF
4.5.4.9 TMA store L1 output
4.5.5 L2 Epilogue
4.5.5.1 TMEM读取
4.5.5.2 转换为BF16并保存到SMEM
4.5.5.3 NVLink远端写
4.5.6 Combine 阶段
4.5.6.1 预处理及同步
4.5.6.2 工作切分
4.5.6.3 SMEM分配
4.5.6.4 主循环流程
4.5.6.5 Chunk 内主循环
4.5.6.6 move_mask_and_load
4.5.6.7 累加循环
4.5.6.8 Cast BF16 + Store
5. 一些分析讨论
```

## 1. Overview

MoE可以通过专家并行(EP)进行加速. 然而, EP需要复杂的节点间通信, 并对互连带宽和延迟提出了巨大要求. 为了缓解EP中的通信瓶颈, 并在较低的互连带宽要求下实现更高的端到端性能, DeepSeek团队提出了一种细粒度的EP方案, 该方案将通信和计算融合成一个单一的流水线化核(kernel), 以实现通信与计算的重叠. 关于细粒度的Overlap, 字节也有COMMET这样的工作, 具体可以参考[《谈谈字节的COMET, 另一个细粒度的MoE通信和计算Overlap方案》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493369&idx=1&sn=6adc84f2786147b8832cd8377ad24955&scene=21#wechat_redirect)

### 1.1 通信延迟隐藏分析

在MoE层中, 通信延迟可以被有效地隐藏在计算之下. 如图所示, 在DeepSeek-V4系列中, 每个MoE层主要可以分解为四个阶段: 两个通信密集型阶段, 分发(Dispatch)和合并(Combine), 以及两个计算密集型阶段, 线性层-1(Linear-1)和线性层-2(Linear-2).

![图片](assets/7012c218d452.png)

性能分析显示, 在单个MoE层内, 通信的总时间少于计算的总时间. 因此, 将通信和计算融合成一个统一的流水线后, 计算仍然是主要瓶颈, 这意味着系统可以容忍较低的互连带宽而不会降低端到端性能.

### 1.2 细粒度EP方案

为了进一步降低互连带宽要求并扩大重叠带来的好处, 作者引入了一种更细粒度的专家划分方案. 受许多相关工作(FlashMoE, COMMET)的启发, 作者将专家分批调度, 称之为“波次(waves)”. 每个波次由一小部分专家组成. 一旦一个波次内的所有专家完成了它们的通信, 计算就可以立即开始, 无需等待其他专家. 在稳定状态下, 当前波次的计算, 下一个波次的token传输, 以及已完成专家的结果发送都同时进行, 如上图所示. 这在专家之间形成了一个细粒度的流水线, 使计算和通信在整个波次处理过程中保持连续. 基于波次的调度提升了在极端情况下的性能, 例如强化学习(RL)的rollout过程, 该过程通常会遇到长尾小批量(long-tail small batches)问题.

作者在NVIDIA GPU和HUAWEI Ascend NPU平台上都验证了这种细粒度的EP方案. 与非融合基线相比, 它在通用推理工作负载上实现了1.50 ~ 1.73倍的加速, 在延迟敏感场景(如RL rollout和高速智能体服务)中最高可达1.96倍. 作者已经开源了名为MegaMoE的基于CUDA的MegaKernel实现, 作为DeepGEMM的一个组件, 具体代码可以参考DeepGEMM的`pr304`和`pr316`.

## 2. Legacy EP实现

### 2.1 EP的计算和通信流

关于DeepSeek在 MoE的路由算法可以参考[《详细谈谈DeepSeek MoE相关的技术发展》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493182&idx=2&sn=7a6017161753ae1f984bc85e98d00987&scene=21&poc_token=HC7j_2mjWDkQc9siuHwjaH8FcJ5qoJeKh1pw6q3t#wechat_redirect), EP通信中的 dispatch 和 combine 涉及跨节点 NVLink 或者 RDMA的通信, 传统的做法是将通信和计算分为独立的Kernel串行执行, 因此NVLink带宽利用率,同时SM也在等待通信时计算利用率低. 关于DeepEPv2的实现和使用NCCL Gin的backend可以参考下面的文章:

[NCCL Gin & Symmetric Memory](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498168&idx=1&sn=adfe6ba01ff8cdbe20cdf5aeb655d0cb&token=724990387&lang=zh_CN&scene=21#wechat_redirect)

[DeepEPv2分析(1)](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498240&idx=2&sn=ebdf052f7c54fd655ea040de7d228f3c&scene=21#wechat_redirect)

[DeepEPv2分析(2)-EP Overview](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498240&idx=1&sn=7ad4bc64bf889e6f58dfd49bfe1133fe&scene=21#wechat_redirect)

[DeepEPv2分析(3)-EP Direct Dispatch/Combine Kernel](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498255&idx=1&sn=3d0b93a65cb5aad476b611e36b5b512a&scene=21#wechat_redirect)

[DeepEPv2分析(4)-EP Hybrid Dispatch/Combine Kernel](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498255&idx=2&sn=4911355e9cabc2f7303e148d247da2ea&scene=21#wechat_redirect)

下面我们再分析Legacy EP之前补充一下Expert FFN的运算

Expert FFN运算
这里我们补充一下 Expert 中的 FFN 运算, 现代 LLM 模型基本上都选择 Noam Shazeer 在 2020 年的论文`《GLU Variants Improve Transformer》`中的SwiGLU方法. 它采用 Swish (也称为 SiLU, Sigmoid-weighted Linear Unit)激活函数.

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

其中 $\beta$ 是一个可学习的或固定的超参数(通常为1). 当 $\beta=1$ 时, 公式就是:

$$\text{Swish}(x) = \frac{x}{1 + e^{-x}}$$

结合 GLU 的思想和 Swish 激活函数, **SwiGLU** 的运算过程如下:

输入一个向量 $x$ ,并同时送入**两个不同**的线性层(或者一个大的线性层然后切分成两半).

第一个线性变换(也被称为 up-projection): $x_1 = xW_1 + b_1$

第二个线性变换(也被称为 gate-projection: $x_2 = xW_2 + b_2$

然后将第二个线性变换的结果 $x_2$ 通过 **Swish** 激活函数, 得到"门控值".

最后将第一个线性变换的结果 $x_1$ 与门控值 $\text{gate}$ 逐元素相乘.

$$\text{output} = x_1 \otimes \text{gate}$$

SwiGLU 的最终公式:

$$\text{SwiGLU}(x) = (xW_1 + b_1) \otimes \text{Swish}(xW_2 + b_2)$$

使用 SwiGLU 的 FFN计算如下:

$$\text{FFN}_{\text{SwiGLU}}(x) = \text{Linear}_{\text{down}}(\text{SwiGLU}(x))$$

在具体计算层面, 如下代码所示:

```
w1 = Linear(dim, inter_dim, dtype=dtype)  # 门控投影（gate）w2 = Linear(dim, inter_dim, dtype=dtype)  # 上投影（up）w3 = Linear(inter_dim, dim, dtype=dtype)  # 下投影swiglu_limit = swiglu_limit  # 激活值裁剪阈值gate = self.w1(x).float()  # 门控分支up = self.w2(x).float()  # 上投影分支if self.swiglu_limit > 0:    # 裁剪激活值以防止数值爆炸    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)    gate = torch.clamp(gate, max=self.swiglu_limit)x = F.silu(gate) * up  # SwiGLU: SiLU(gate) * upif weights is not None:    x = weights * x  # 乘以路由权重return self.w3(x.to(dtype))  # 下投影回原始维度
```

通常在计算的时候, 我们会把 gate 和 up 权重拼接在一起, 也就是DeepSeek文中所讲的`L1`, 最后的down-projection为第二个线性层`L2`, 如下图所示:

![图片](assets/bc50768ac960.png)

⚠️ **注意**: gate和up也可以采用Interleave的方式存储, 并配合Swap AB输出后就可以直接进行 swiGLU 的运算. MegaMoE采用了interleave的方式. 我们将在后续的章节详细进行分析.

### 2.2 Legacy EP实现

在此之前EP并行模式下MoE 层的执行包含五个串行步骤：`dispatch → linear1 → SwiGLU → linear2 → combine` , 我们可以看到`run_baseline()`函数的流程如下:
阶段 1: [EP Dispatch] → 跨 rank 分发 token
将本 rank 的 token（`x` 是 `(fp8_data, sf)` 元组）按 `topk_idx` 路由, 通过 NVLink all-to-all 送到负责对应专家的目标 rank. 其中`ep_buffer`是通过DeepEP库创建的

```
recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(    x, topk_idx=topk_idx, topk_weights=topk_weights,    num_experts=num_experts, expert_alignment=alignment,    do_cpu_sync=False, do_handle_copy=False,    do_expand=True, use_tma_aligned_col_major_sf=True)
```
阶段 2: [L1 分组 GEMM] → 输入投影(gate + up)
对接收到的 token 按专家分组做矩阵乘法, 计算 `l1_y = recv_x @ l1_weights^T`.

```
n = recv_x[0].size(0)l1_y = torch.empty((n, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(    recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert,    use_psum_layout=True, recipe=(1, 1, 32))
```
阶段 3: [SwiGLU 激活 + TopK 加权 + 量化]
这是一个Fused Kernel(由 `tilelang` 提供), 在一次 kernel 内完成四件事: SwiGLU 激活  → Clamp 截断 → 乘以 TopK 权重(后续 combine 只需做纯加法) → 量化回 FP8

```
l1_y = tilelang_ops.swiglu_apply_weight_to_fp8(    x=l1_y,                                                  # [gate | up] 拼接输入    topk_weights=recv_topk_weights,                          # 每 token 的路由权重    avail_tokens=handle.psum_num_recv_tokens_per_expert[-1], # 实际有效 token 数    num_per_channels=32,                                     # SF(scale-factor) 分组粒度（=32）    use_col_major_scales=True,                               # SF 是否列主序    round_scale=True,                                        # True 时把 SF 向上取整到最接近的 2 的幂    ue8m0_scale=True,                                        # SF 是否 UE8M0 格式     output_bf16=False,                                       # 是否同时输出 BF16    clamp_value=args.activation_clamp,                       # clamp阈值    fast_math=bool(args.fast_math)                           # fast-math)
```
阶段 4: [L2 分组 GEMM] → 输出投影
计算 `l2_y = l1_y @ l2_weights^T`, 将中间激活投影回 `hidden` 维

```
l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(    l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert,    use_psum_layout=True, recipe=(1, 1, 32))
```
阶段 5: [EP Combine] → 聚合回原 token
将分散在各 rank 的加权结果按 `handle` 中记录的源信息反向 all-to-all 送回原 token 所在 rank, 并把同一个原 token 的 k 个专家结果进行reduce

```
return ep_buffer.combine(l2_y, handle=handle)[0]
```

## 3. MegaMoE实现

### 3.1 整体架构

MegaMoE 将 EP Dispatch、Linear1 (Gate/Up)、SwiGLU、Linear2 (Down)、EP Combine 五个操作融合到单个 CUDA Kernel 中，并通过通信和计算Overlap实现更好的性能. 其中还使用了FP8 x FP4 混合精度的GEMM.
函数输入输出Layout
每个 Rank 的输入:

![图片](assets/cc2676e61fe7.png)

经过预变换的专家权重:

![图片](assets/66840761f2a0.png)

输出:

![图片](assets/bb60393ea8d3.png)
Overlap的方式和Warp功能划分
论文中的图如下:

![图片](assets/b3f472cf7abc.png)

那么什么是Expert Wave呢? 首先在Dispatch阶段, 原始的输入是按照每个Rank生成的token放置在`input_buffer`中, 并有额外的`input_tok_idx_buffer`用于专家索引. 我们以下面一个Rank=2,总共6个专家平均每个Rank 3个专家, topk=4的简单例子来说明.

![图片](assets/36b3b27767c4.png)

通过dispatch的处理后, 逻辑上`L1 pool`会以本地的Expert作为主序排列, 然后将token放入相应的slot内. 因此整个流水线我们就可以按照这个维度拆分成多个wave. 例如:

Expert Wave 1 处理Expert 0 相关的MoE计算

Expert Wave 2 处理Expert 1 相关的MoE计算

Expert Wave 3 处理Expert 2 相关的MoE计算

这样的方式就可以在多个Expert Wave中进行Overlap, 隐藏通信所需要的时间. 当然这只是一个很简单的示例, 实际的处理中按照block进行调度我们将在后续的章节详细分析. 接下来, 我们把单个wave拆开, 单个Expert wave 计算流程分为 5 个阶段:

阶段 0 — EP Dispatch: 专家 token 统计 → 全局聚合 → NVLink Pull → TMA Store

阶段 1 — Linear1+SwiGLU: Wave-based 调度 → Swap AB UMMA → UTCCP 转置 → TMEM_LOAD interleaved Gate/Up → SwiGLU → FP8 Cast

阶段 2 — Linear2: L2 arrival mask 自旋等待 → Down 投影 GEMM

阶段 3 — L2 Epilogue : TMEM→BF16→NVLink 远程写入 Combine Buffer

阶段 4 — Combine: 双缓冲 Top-k 加载 → Float 累加 → TMA Store 最终输出

我们需要把这几个阶段完全揉到一个persistent Kernel内, 那么就需要通过WarpSpecialization的方式分工, 并配合细粒度的Barrier机制来实现计算和通信的重叠. 参考源文件: `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`, 具体分工如下:

![图片](assets/2ee99f521a73.png)

基于Warp展开成详细的流程图如下所示:

![图片](assets/d8d57676f6ba.png)

这里标注了一些不同 warp 之间是如何通过一些 barrier 和 counter 协同进行工作, 这里我们做一个简单的介绍, 后续的章节中详细展开介绍.
Dispatch Warp
Dispatch Warp承担整个 MoE 的 all-to-all dispatch: 统计 expert 命中、上报 `expert_send_count`、写远端 topk 索引、通过 NVLink + TMA 把 token 主体 / SF / weight 拉回 L1 pool, 最后清理 workspace 并与 epilogue 协同释放资源. Dispatch warp 在 persistent kernel 中经历 6 个阶段, 由 intra-SM named barrier、 grid_sync、nvlink_barrier 三级屏障串接, 整个处理流程如下:

![图片](assets/7b36f84847d1.png)

1️⃣ **统计本 SM 发往 expert i 的 token 数**:  具体做法是并行遍历所有 token, 每个SM负责一部分, 它在smem中用 atomicAdd 统计每个 expert 在本 SM 的 token 数.

2️⃣ **全局send_count**: 这一步很巧妙, 用 atomicAdd 向全局 workspace 写入 expert 发送计数`expert_send_count[i]` . 注意全局的这个计数器相当于一个发号器的作用. 例如当前值为`A1`, SM 1 发送atomicAdd后, 计数器更新为`A1+sm1_count` ,同时返回旧值`A1`, 此时SM 1就知道在`[A1, A1+sm0_count)`这段区间作为接收数据的段, `A1`作为在远端expert slot上的偏移起始点. 后续 SM2 发送atomicAdd将会返回`A1+sm1_count`, 并以此作为SM1在远端expert slot上的偏移起始点.

3️⃣ **写远端topk idx**: 然后根据前一步的起始偏移就可以将本地token的`topk_idx`写入到远端. 为什么第三步要将topk idx写到 **远端**? 因为发送方 rank 知道自己往哪个 expert 发了什么 token, 但 expert 归属的 **接收方 rank** 才是后续 pull 的执行主体, 它需要直接读本地 workspace 就能查到每个 slot 的源 index, 避免在 pull 时再做一次 NVLink 反查.

4️⃣ **SM0聚合**: 当进行完`grid_sync`后, 第二步中的`expert_send_count[i]`已经是本 rank **聚合后**的终值(kNumSMs 个贡献者已全部加入). 跨 rank 写只需用SM0写 1 次, 它将更新接收端的`recv_count`, 这是一个`[rank, expert_idx]`的二维数组. 然后它还做了一次atomicAdd更新远端的`expert_recv_count_sum`, 它代表所有 rank 发来的该 expert 的汇总 token 数. 在后续的scheduler中会根据`fetch_expert_recv_count` 函数对每个 expert 执行 `ld_volatile` 自旋循环, 当到齐后触发block分发调度并触发后续的GEMM运算.

5️⃣ **拉取token**: 将其它 rank 命中本地 expert 的 token 主体(FP8 权重 + scale factor + topk 权重)拉回到本地的**L1 token pool**, 同时设置`l1_arrive_cnt`触发TMA-Producer A消费.

6️⃣ **清理workspace**: 这是在整个完成阶段进行的内存清理工作.
TMA-Producer A Warp
它在 MoE kernel 的 GEMM 阶段担任 **「加载激活(activation)及其 scale factor 的生产者」** , 它由调度器驱动`scheduler.for_each_block`

![图片](assets/e7f742ba9b11.png)

它将同时处理L1 GEMM和L2 GEMM的activation数据加载:

![图片](assets/85ce6bb35a5e.png)

L1阶段通过等待Dispatch Warp更新的`l1_arrive_cnt`触发. 而L2阶段则通过L1 Epilogue warp更新的`l2_arrive_mask`触发. 触发后它将配合TMA-Producer B和MMA Warp在K方向上循环并完成GEMM运算的数据加载工作. 然后它将通过`full_barriers[stage_idx]`通知MMA Warp已经加载完成. 并且MMA Warp也可通过`empty_barriers[stage_idx]`通知TMA-Producer A继续进行下一轮加载.
TMA-Producer B Warp
TMA Load B warp 是 **MoE kernel 的 GEMM 权重生产者**, 它无需等待, 直接通过TMA在K维度加载权重. 同时通过`full_barriers[stage_idx]`和`empty_barriers[stage_idx]`与TMA-Producer A 和 MMA Warp协同在K维度推进, 直到完成计算后, 切换到 L2继续进行权重加载.
MMA Warp
MMA Issue warp 是整个 kernel 的 **计算心脏**: 它从 smem 消费 A/B 数据与 SF, 把 SF 通过 UTCCP 搬到 TMEM, 发射 SM100 **2-CTA UMMA** 指令, 让两个 CTA 共同完成一个 GEMM tile 的计算, 累加结果直接留在 TMEM 中供 epilogue warp 读取. 这里有一些Swap AB的技巧, 我们将在后续的章节中详细展开.

然后结果在TMEM accumulator 中采用双缓冲机制, 并通过`tmem_full_barriers`和`tmem_empty_barriers`和 Epilogue Warp进行协同.
Epilogue Warp
它也是一个处理非常复杂的Warp, 覆盖三个阶段:
L1 Epilogue
主要包含 SwiGLU + FP8 量化 + TMA store + UE8M0 SF 写入等几步. L1 = 第一阶段 GEMM（gate+up projection）的结果后处理. 它首先加载 top-k 权重到寄存器缓存, 然后TMEM_LOAD gate/up interleaved的值, 并进行SwiGLU: silu(gate) × up × weight计算. 然后在warp内进行 amax 处理. 然后per-lane 计算 amax, warp reduce + cross-warp reduce, 然后量化为 FP8 E4M3, 按 UE8M0 方案存储 SF. 最后通过 TMA store 写到 `tensor_map_l1_output`(即 `l2_token_buffer` 的 GMEM 视图), 然后通过 `red_or_rel_gpu(l2_arrival_mask)` 原子位运算通告「此 N 子块已就绪」供TMA-Producer-A进行L2的block加载.
L2 Epilogue
主要包含TMEM → BF16 → STSM → NVLink 远端写入这几步. 2 = 第二阶段 GEMM（down projection）结果后处理.首先从 TMEM 读累加器获得L2 GEMM完成的数据, 然后转 BF16 格式, 并 通过 STSM 写到 `smem_cd_l2`, 接着按 `token_src_metadata` 确定每行的远端 rank/token/topk 位置, 并通过 NVLink(`sym_buffer.map`)直接写入远端 `combine_token_buffer[topk_idx][token_idx]`.
Combine 阶段
Combine 阶段目标：每个本 rank 发出的 token 有 `kNumTopk` 份结果(由远端不同 expert 计算), 现在要把它们归约成一份, 写回用户的 `y`. 首先它需要一个`nvlink_barrier`等所有 rank 完成 L2 写入, 然后通知允许 Dispatch 开始清理. 接着对每个 token 读取 top-k slot 索引, 然后按chunk进行 FP32累加, 最后将结果Cast回BF16格式存入SMEM, 并调用TMA Store 最终输出 y.

整个5类warp在单个Expert Wave中的工作流程我们进行了简略的介绍, 但是还缺少一些细节, 例如Expert Wave是如何切分的, block任务是如何分配的, 因此下面我们先从调度器开始分析, 然后分析相关的控制计数器和用于Warp间协同的Buffer.

### 3.3 Scheduler

Scheduler是整个算法中非常关键的一环, 代码在`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`中. 整个调度分为三层状态机, 从外到内:

Wave: expert 按 kNumExpertsPerWave 打包

Block: expert, m_block, n_block

Phase: Linear1 → Linear2

在执行GEMM的时候, 首先将 `kNumExpertPerWave` 个专家的处理打包到一个Wave, 对于每个Wave内的每个专家会按照`BLOCK_M`的大小对齐, 然后执行分块矩阵乘法, 由于MoE FFN计算有两次GEMM, Phase 负责切换 Linear1↔Linear2, wave 内需要全部 L1 block 做完,再统一切 L2. 对于单个Phase内, Block 负责 SM 间 block-cyclic 分派.

#### 3.2.1 启发式配置

在介绍调度器之前, 对于每个Wave要处理的数据量, Block的大小等参数需要进行一个启发式配置. 源码路径: `csrc/jit_kernels/heuristics/mega_moe.hpp`, 根据运行时输入(`num_ranks`/`num_experts`/`hidden`/…)推导出一套满足 **正确性 + 共享内存上限 + SM 利用率** 三重约束的 `MegaMoEConfig`, 交给下游 `sm100_fp8_fp4_mega_moe.hpp` 做 JIT 源码模板实例化. 它对外提供的API如下:

![图片](assets/0375a36743dc.png)

整个调用链路如下:

```
apis/mega.hpp      │ (Python 层传入 shape / rank 等信息)      ▼get_mega_moe_config()   ← 顶层  ├── get_block_config_for_mega_moe()           ← 选 BLOCK_M 等  ├── SM100ArchSpec::get_sf_uttcp_aligned_block_sizes()  ← 选 SF 块  ├── layout::get_num_max_pool_tokens()         ← 选 Pool 容量  ├── get_num_experts_per_wave_for_mega_moe()   ← 选 wave 粒度  └── get_pipeline_config_for_mega_moe()        ← 解出 num_stages + smem      │ (返回 MegaMoEConfig)      ▼impls/sm100_fp8_fp4_mega_moe.hpp              ← JIT 代码生成
```

内部的一些字段定义如下:

![图片](assets/180881524ea9.png)
3.2.1.1 Block大小选择
对于选择 BLOCK_M 参数的函数为:

```
static std::tuple<int, int, int, int> get_block_config_for_mega_moe(    const int& num_ranks, const int& num_experts,    const int& num_max_tokens_per_rank, const int& num_topk,    const int& num_tokens);
```

它的输入参数为:

![图片](assets/1f5f1bb739ef.png)

返回 `std::tuple<cluster_size, block_m, store_block_m, num_epilogue_warpgroups * 128>`:

`cluster_size`: 始终为 2(2-CTA cluster)

`block_m`: 从 `{16, 32, 64, 96, 128, 192}` 中选一个

`store_block_m`: epilogue TMA store 的 M 粒度

最后一元是 `num_epilogue_warpgroups * 128`, 即 `num_epilogue_threads`

核心算法是按 token-per-expert 档位分派

```
float num_expected_tokens_per_expert =    static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
```

公式解释如下:

本 rank 每次收到的 token 总数上界 ≈ `num_tokens * num_ranks`(全部 rank 都把自己的 token 发给当前 rank)

每个 token 触发 `num_topk` 次 expert 路由

路由平均分布到 `num_experts` 个 expert 上

得到每个 expert 期望被分到的 token 数

由于采用2-CTA进行GEMM, `cluster_size` 恒为 2, 其它值按 6 档阶梯选配置:

![图片](assets/7cb44ce54325.png)

设计要点:

**`store_block_m ≤ block_m`**: 让 epilogue 可以多次 TMA store, 把 M 方向与 TMEM 读重叠

**小 BLOCK_M 用 2 个 warpgroup(256 epilogue 线程)**: epilogue 工作量"薄 tile 多、每 tile 轻", 用两组 warp 并行 pipeline 多次 store

**BLOCK_M=64 档位 1 个 warpgroup**: tile 刚好, 单组足够掩盖延迟, 节省 smem 里的 dispatch/combine barrier 数
3.2.1.2 Pool容量
每个 rank 在 dispatch 阶段会收到来自所有 rank 的 token. 这些 token 按专家分组存放在一块连续的共享 pool 中(所有本地专家共用一段 buffer), 每个专家占用一段连续区间. 这样 MMA warp 可以像处理普通 GEMM 一样按 `BLOCK_M` 切块扫描. 因此 pool 容量需满足两个条件:

容得下**最坏情况**下真实接收的所有 token；

每个专家的起始地址都要对齐到 `BLOCK_M`(否则 TMA / UMMA 寻址会错), 对齐要对**所有候选 BLOCK_M** 都成立.

完整的计算公式如下:

$$N_{\text{pool}} = \text{align}\Bigl( \underbrace{R \cdot T \cdot \min(k, E_r)}_{\text{极端真实 token 数}} + \underbrace{E_r \cdot (192 - 1)}_{\text{所有候选 BLOCK\_M 下的填充上限}},\ \ 384 \Bigr)$$

其中:

$R$ = num_ranks

$T$ = num_max_tokens_per_rank

$k$ = num_topk

$E_r$ = num_experts_per_rank

接下来详细解释一下, 本 rank 可能收到的 token 来源只有"全网每个 rank 各自的输入 token". 最坏情况下**每个 rank 都把所有 token 路由给本 rank 的专家**, 因此上界是:

$$N_{\text{recv\_max}} = num\_ranks \times num\_max\_tokens\_per\_rank$$

这是 token 数(未乘 top-k)的上界. 一个 token 会被复制成 `num_topk` 份发给不同专家. 但是:

如果 `num_topk ≤ num_experts_per_rank`: 最坏情况下这 top-k 份**全部**落在本 rank 的专家里 → `num_topk`.

如果 `num_topk > num_experts_per_rank`: 由于一个 token 的 top-k 不会选到重复专家(不同 expert_idx 不同), 最多只能选到本 rank 上 `num_experts_per_rank` 个专家, 也就是每个专家最多收 1 份 → `num_experts_per_rank` 封顶.

所以取 `min(num_topk, num_experts_per_rank)` 作为单 token 在本 rank 上的最大副本数.

```
const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
```

把两者相乘:

$$N_{\text{token\_upper}} = \bigl(num\_ranks \times num\_max\_tokens\_per\_rank\bigr) \times \min(num\_topk, num\_experts\_per\_rank)$$

这表示"所有 rank 的所有 token 都打到本 rank, 并且每个 token 在本 rank 上尽量复制最多份"的极端情况下总 token 数. 这是 pool 里真实数据的上限.

然后考虑到对齐`BLOCK_M`因此增加`+ num_experts_per_rank * (kMaxCandidateBlockM - 1)`.

为什么需要填充？
pool 里每个专家的起始位置要被对齐到 `BLOCK_M`(为了 TMA / UMMA 寻址、SF 排布). 一般做法是: 把每个专家末尾填充到下一个 `BLOCK_M` 边界.

单个专家最坏情况下会多填 `BLOCK_M - 1` 个位置(真实 token 数恰好比 `BLOCK_M` 的整数倍多 1).

本 rank 有 `num_experts_per_rank` 个专家, 所以总填充最坏为: `num_experts_per_rank × (BLOCK_M - 1)`.
为什么用 `kMaxCandidateBlockM`？
`BLOCK_M` 是 JIT 运行时才确定的(从候选 `{8,16,32,64,96,128,192}` 中选). 这里 pool 大小必须一次性预留, 不能因 `BLOCK_M` 不同而变化. 因此取所有候选里最大的 192, 保证对任何选择都足够:

$$P_{\text{padding}} = num\_experts\_per\_rank \times (192 - 1)$$

这是一个"覆盖所有可能 BLOCK_M"的保守上界.

最后, 需要把前面算出的"真实上限 + 填充上限"再向上对齐到 `kLCMCandidateBlockM = 384`.

```
return math::constexpr_align(    num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (kMaxCandidateBlockM - 1),    static_cast<T>(kLCMCandidateBlockM));
```

`kLCMCandidateBlockM = 384` 是所有候选 `BLOCK_M` 的最小公倍数, pool 总容量能被任何候选 `BLOCK_M` 整除. 这样无论 heuristic 最终选哪个 `BLOCK_M`, pool 的 block 数都是整数, 不会有"切出一个不完整 block"的麻烦.

举个例子,设 `num_ranks=8, num_max_tokens_per_rank=256, num_topk=8, num_experts_per_rank=32`:

![图片](assets/0397c6b7608e.png)
3.2.1.3 Expert Wave粒度
这一部分决定在一个 wave(调度波次)里, 该让每个 rank 并发处理多少个本地专家, 才能把所有 SM 都吃满, 同时又不让负载不均匀的问题被放大?

首先先估算"每专家期望 token 数"

```
float expected_tokens_per_expert =    static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
```

`num_tokens * num_topk`: 本 rank 一共要发给本地专家多少个 token(一个 token 被复制 top-k 份), 除以 `num_experts_per_rank`: 假设路由完全均匀时每个本地专家分到的 token 数. 这是"纸面均值", 后面所有负载估算都以它为基础.

稀疏极端情况, 当平均下来每个专家不到 1 个 token(例如 token 很少、专家很多的极稀疏场景), 再分 wave 调度反而浪费. 直接**把所有本地专家塞进一个 wave** 一次性算完, 避免空转.

```
if (expected_tokens_per_expert < 1) {    return num_experts_per_rank;}
```

然而实际路由并不均匀: 一些热专家拿到的 token 远超均值, 冷专家却很少. 如果只按"均值"算出 block 数, 热专家会拖尾；因此在后面要把**目标工作量放大 2 倍**, 即`kImbalanceFactor = 2`(相当于把冷专家算少了一半的容量补回来), 留出冗余让热专家也能吃满 SM.

接着按均匀路由估算单个专家的 L1 block 数.

```
const int num_m_blocks = ceil_div(    static_cast<int>(std::ceil(expected_tokens_per_expert)), block_m);const int num_n_blocks = (2 * intermediate_hidden) / block_n;   // L1 的 N 维是 2I(gate||up)const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;
```

`num_m_blocks`: M 方向按 `block_m` 切分, 得到每个专家要算多少个 M-block.

`num_n_blocks`: Linear1 的输出宽度是 `2 * intermediate_hidden`(gate 和 up 拼接), 按 `block_n` 切分.

相乘得到**一个专家在 L1 阶段需要计算的 (m_block × n_block) 总数**.

这个量代表"一个专家能喂饱多少个 SM", 然后就可以通过以下公式求到wave内专家数的下界, 计算的目的是: 如果取太大会导致 wave 数变少、调度粒度太粗；太小又会让部分 SM 饿着. 所以取"刚好够吃满 SM"的值.

$$num\_experts\_per\_wave = \left\lceil \frac{imbalance\_factor \cdot num\_sms}{num\_l1\_blocks\_per\_expert} \right\rceil$$

随后用 `num_experts_per_rank` 封顶, 避免超过本 rank 实际拥有的专家总数.

```
num_experts_per_wave = std::min(num_experts_per_wave, num_experts_per_rank);
```

最后向上取整到 `num_experts_per_rank` 的因子

```
while (num_experts_per_wave < num_experts_per_rank and       num_experts_per_rank % num_experts_per_wave != 0)    ++num_experts_per_wave;
```

调度器对 wave 的要求是每个 wave 处理相同数量的专家(否则最后一个 wave 会是不规则的尾巴, 会触发调度器里的静态断言). 因此不断 `+1`, 直到它能整除 `num_experts_per_rank`. 例如 `num_experts_per_rank = 16`:

若公式算出 5 → 向上调到 8 (因为 16%5≠0、16%6≠0、16%7≠0、16%8=0).

若公式算出 3 → 向上调到 4.

若公式算出 9 → 向上调到 16.

假设 `num_tokens=1024`, `num_topk=8`, `num_experts_per_rank=32`, `intermediate_hidden=2048`, `block_m=128`, `block_n=128`, `num_sms=148`:

`expected_tokens_per_expert = 1024*8/32 = 256`(≥1, 进入正常分支)

`num_m_blocks = ceil(256/128) = 2`

`num_n_blocks = 2*2048/128 = 32`

`num_l1_blocks_per_expert = 2*32 = 64`

`num_experts_per_wave = ceil(2*148/64) = ceil(4.625) = 5`

`min(5, 32) = 5`

向上取整到因子: 5→8 (32 的因子中最近的 ≥5 者)

**最终 `num_experts_per_wave = 8`**, 即 32 个本地专家分 4 个 wave, 每 wave 8 个专家, 并发约 `8×64=512` 个 L1 block, 对 148 个 SM 来说有 ~3.5 倍冗余, 足够吸收路由不均.
3.2.1.4 SMEM分布及流水线深度估计
这个函数要回答的问题是: 在 SM 的共享内存上限内, 能开多少级流水(`num_stages`), 才能把 K 方向的 TMA load 与 MMA 计算最大程度地重叠？它利用"(总 smem − 固定开销) / 单级流水开销"向下取整. 在SMEM中的内存分配如下:

![图片](assets/8a132b30d1b1.png)
固定开销部分
首先是`Dispatch`区

```
smem_expert_count_size = align(num_experts * 4, 1024)smem_send_buffers_size = align( Buffer(Data(hidden), num_dispatch_warps, 1).bytes , 1024 )smem_dispatch_size     = smem_expert_count_size + smem_send_buffers_size
```

`expert_count`: 每个全局专家一个 `uint32` 计数器(dispatch 阶段统计本 SM 要发出的 token 数). 对齐到 1 KB.

`send_buffers`: 每个 dispatch warp 持有一个 `hidden` 大小的缓冲区, 用于暂存要跨 rank 发送的 token(通过 `layout::Buffer(Data(hidden), num_dispatch_warps, 1)` 计算字节数). 对齐到 1 KB.

然后是C/D输出区

```
smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStagessmem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * sizeof(bf16)smem_cd    = max(smem_cd_l1, smem_cd_l2)
```

L1 和 L2 的 epilogue **复用同一段 smem**, 所以取两者 max:

L1(Linear1 输出, SwiGLU 后): 数据类型是 FP8(1 byte)；SwiGLU 把 gate×up 合并为一半宽度 → `block_n/2`；需要 2 个 TMA store 缓冲做双缓冲重叠.

大小 = warpgroup 数 × `store_block_m` × `block_n/2` × 1 byte × 2 stage

L2(Linear2 输出): 数据类型是 BF16(2 byte), 只有 1 份(直接经 NVLink 写远端).

大小 = warpgroup 数 × `store_block_m` × `block_n` × 2 byte

接着是Amax 归约缓冲, L1 epilogue 做 FP8 量化时要跨 warp 求 amax(绝对值最大), 这块 smem 用于 warp 之间交换中间结果. 每个 store tile 的每一行对每个 epilogue warp 各 4 字节.

```
smem_amax_reduction = store_block_m * num_epilogue_warps * sizeof(float)
```

最后是barrier区

```
smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2) * 8
```

mbarrier 每个占 8 byte, 分三类:

![图片](assets/757d367f61d3.png)

还有一个TMEM 分配器返回的指针(4 byte)需要挂在 smem 上供所有 epilogue warp 读取.
单级流水开销
每一个流水级都要复制一份"A 片 + B 片 + SFA + SFB + 2 个 barrier"——**这是流水能并发 N 级所付出的成本**. 关键点:

**A tile**: 因为 2-CTA multicast, 单 CTA 只需 `load_block_m × block_k = (BLOCK_M/2) × BLOCK_K`. 数据类型 FP8(1 byte)所以没额外乘 size.

**B tile**: `block_n × block_k`.

**SFA/SFB**: 按 UTCCP 要求, 每 128 元素共享 1 个 scale(UE8M0 打包成 uint32 存), 所以按 `sf_block_m/sf_block_n` 算, 每个槽 4 byte.

**2 × 8 byte**: 每个 stage 的 `full_barrier` 与 `empty_barrier`(producer/consumer 双 barrier 协议).

```
smem_sfa_per_stage = sf_block_m * 4smem_sfb_per_stage = sf_block_n * 4smem_per_stage     = load_block_m * block_k    // A tile(multicast 过后只占一半)                   + block_n      * block_k    // B tile(两 CTA 各完整 BLOCK_N)                   + smem_sfa_per_stage        // SFA(UTCCP 对齐, 每 128 元素 1 组 SF, 4 B)                   + smem_sfb_per_stage        // SFB                   + 2 * 8                      // 本 stage 的 full + empty barrier
```
计算 `num_stages`
通过SMEM容量减去固定开销得到能分给流水缓冲的"可用 smem". 除以单级开销, 向下取整得到最大可容纳的级数. 最后断言`num_stages >= 2`至少要双缓冲才能形成 load↔compute 的并发, 否则退化为串行.

```
const int num_stages = (smem_capacity - smem_fixed) / smem_per_stage;DG_HOST_ASSERT(num_stages >= 2);
```

设 SM100 常见情况: `smem_capacity = 232 KB`, `num_experts=256, num_dispatch_warps=4, num_epilogue_warps=8, block_m=128, block_n=128, block_k=128, store_block_m=32, sf_block_m=128, sf_block_n=128, hidden=7168`.

先算固定区(粗略):

| 项 | 近似值 |
|---|---|
| smem_expert_count | align(256·4, 1024) = 1024 B |
| smem_send_buffers | align(4·hidden·1B, 1024) ≈ 29 KB |
| smem_dispatch | ≈ 30 KB |
| smem_cd_l1 | 2·32·64·2 = 8 KB |
| smem_cd_l2 | 2·32·128·2 = 16 KB |
| smem_cd = max | 16 KB |
| smem_barriers | (4 + 4 + 16)·8 = 192 B |
| smem_amax | 32·8·4 = 1 KB |
| smem_tmem_ptr | 4 B |
| smem_fixed | ≈ 47 KB |

再算单级:

| 项 | 值 |
|---|---|
| A tile | 64·128 = 8 KB |
| B tile | 128·128 = 16 KB |
| SFA | 128·4 = 512 B |
| SFB | 128·4 = 512 B |
| 2 barrier | 16 B |
| smem_per_stage | ≈ 25 KB |

最后: `num_stages = (232 - 47) / 25 = 185 / 25 = 7`, 即开 **7 级流水**, 总占用约 `47 + 7×25 = 222 KB`, 刚好落在 232 KB 的预算内, 预留约 10 KB 对齐冗余.

#### 3.2.2 详细的调度流程

wave 起始 expert = `align_down(cur_expert, kNumExpertsPerWave)`, wave 结束 expert = `get_wave_expert_end_idx()`.假设一个 expert 的 token 在 M 维按 `BLOCK_M=16` 切；不足一块的尾部 pad 到 16 行但只有 `valid_m` 个有效.

```
uint32_t get_current_num_m_blocks() const {    return math::ceil_div(current_num_tokens, BLOCK_M);   // ceil(num_tokens / 16)}
```

Block 总数公式, 其中 `kNumL1BlockNs = L1_SHAPE_N / BLOCK_N`、`kNumL2BlockNs = L2_SHAPE_N / BLOCK_N`.

| 阶段 | 每 expert 的 block 数 |
|---|---|
| L1 (Linear1) | `num_m_blocks × kNumL1BlockNs` |
| L2 (Linear2) | `num_m_blocks × kNumL2BlockNs` |

那么对于一个 wave 包含 `W = kNumExpertsPerWave` 个 expert; 每个 expert 的 L1 block 是一张 `num_m_blocks(e) × kNumL1BlockNs` 的二维表. 调度时把整个 wave 的所有 expert 按 expert 顺序头尾拼接成一维地址:

![图片](assets/2f335653c1f9.png)

整体的调度流程如下所示:

![图片](assets/08d9397a4eae.png)

具体来看, BlockPhase结构如下:

```
// Computation phase for the current blockenum class BlockPhase {    None = 0,      // 全部任务已处理完, 外层循环应当退出    Linear1 = 1,   // 当前任务属于 MoE 第 1 层线性变换(通常是 gate/up projection)    Linear2 = 2    // 当前任务属于 MoE 第 2 层线性变换(通常是 down projection)};
```

`get_next_block`的状态机如下所示, 其中`block_idx += kNumSMs` 让本 SM 每次跨 `kNumSMs` 步取下一个块, 天然实现 **SM 间条带化(block-cyclic)分配**:

| block_idx | 对应 SM |
|---|---|
| 0, kNumSMs, 2·kNumSMs, ... | SM 0 |
| 1, 1+kNumSMs, ... | SM 1 |
| ... | ... |

```
    // Core state machine: assigns the next block    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {        while (true) {            // 终止条件: 所有本地 expert 都处理完.             if (current_local_expert_idx >= kNumExpertsPerRank)                break;            if (next_phase == BlockPhase::Linear1) {                if (fetch_next_l1_block()) {                    // 命中 L1 块: 由 m_block_idx 反推 n_block_idx(N 维展平).                     n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;                    // 跳到本 SM 的下一个候选块(block-cyclic 步长 = kNumSMs).                     block_idx += kNumSMs;                    return {BlockPhase::Linear1,                            current_local_expert_idx, m_block_idx, n_block_idx};                } else {                    // 当前 wave 的 L1 已全部分配, 切换到 L2.                     next_phase = BlockPhase::Linear2;                    // 关键回退: 把 expert 重置到当前 wave 的起点, 重新扫一遍发 L2 块.                     // 这里用 align<..., false>(向下对齐), 配合 "-1" 抵消 fetch 循环                    // 退出时 current_local_expert_idx 已越过 wave 尾的情况.                     set_expert_idx(math::align<uint32_t, false>(                        current_local_expert_idx - 1, kNumExpertsPerWave));                }            } else {                if (fetch_next_l2_block()) {                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;                    block_idx += kNumSMs;                    return {BlockPhase::Linear2,                            current_local_expert_idx, m_block_idx, n_block_idx};                } else {                    // 当前 wave 的 L2 也发完, 下一 wave 从 Linear1 继续.                     // 注意: 此时 current_local_expert_idx 由 fetch_next_l2_block 的 advance_expert_idx                    // 推进到了"越过当前 wave 尾"的位置, 正好是下一 wave 的起点.                     next_phase = BlockPhase::Linear1;                }            }        }        // 全部处理完, 返回 None 让外层循环退出.         return {BlockPhase::None, 0, 0, 0};    }
```

然后这些调度器模块以 `for_each_block(func)` 单一接口暴露给 kernel, 每个Warp通过调度器获取任务, 降低上层代码复杂度.

### 3.3 Buffer Layout

在介绍详细的执行流程前, 我们首先来看一下它的Buffer是怎么在多个GPU上进行划分的, MegaMoE使用了NVLink Symmetric Memory, 整个Layout分为Workspace和Buffer两块, 具体的代码在`csrc/apis/mega.hpp`和`deep_gemm/include/deep_gemm/layout/mega_moe.cuh`中定义.

![图片](assets/211afa58563a.png)

#### 3.3.1 Workspace

`Layout::Workspace` 对象定义了 MegaMoE kernel 中所有跨 warp / 跨 CTA / 跨 rank 的控制面数据在一整块对称内存(symmetric buffer, 通过 NVLink 连接的多 GPU 共享地址空间)上的内存结构布局. 它从 `sym_buffer.get_base_ptr()` 起按段切分, 它主要包含以下几类:

![图片](assets/45b81ab24483.png)

![图片](assets/c12ca996e72f.png)

每段由 `get_*_ptr(indices)` 计算偏移, 每个 getter 都调用上一段的"尾部指针"作为自身的起点. 例如 `get_expert_recv_count_ptr` 基于 `get_expert_send_count_ptr(num_experts)` 推算, `get_l2_arrival_mask_ptr` 基于 `get_l1_arrival_count_ptr(align(...))` 推算. 这种设计使得修改一段的大小时**只需改一处**, 整条链自动重新对齐. 另外所有 getter 都是纯指针运算, 在编译期完全展开为常数偏移或简单的 `base + const * indices`. Device 端调用等同于直接指针访问.

为了更好的便于理解后续在多个warp之间的状态同步, 我们来对这些barrier和计数器进行详细的分析.

![图片](assets/a5653c792e42.png)

首先是系统level的两个API

`get_num_bytes()`
它是是 host 端确定总分配量的唯一入口.
`get_end_ptr()`
host 链式分配  `advance_ptr(base, get_num_bytes())`

用于同步Barrier

`get_grid_sync_count_ptr<kIndex>()`
`kIndex` 是编译期模板参数, 取值范围 `0..3`, 选择一个独立的 grid sync 计数器槽位. 它指向一个 `uint32_t`, 该 32 位数字被分为两部分使用: 低位累积到达的 SM 数, 最高位 `0x80000000` 作为"完成标志". grid sync的实现在`/deep_gemm/include/deep_gemm/comm/barrier.cuh`, 工作流程如下:

每个 SM 的 thread 0 **atomic add**: SM 0 写入 `0x80000000 - (kNumSMs - 1)`, 其他 SM 写入 `1`

所有 SM 的 thread 0 自旋 `ld_acq` 等待最高位翻转, 即 `(new ^ old) & 0x80000000 != 0`

**调用时机**: Dispatch 在完成计数和源索引写入后调用 `grid_sync<kDispatchGridSyncIndex = 0>`. Epilogue 在 NVLink 写回完成后调用 `grid_sync<kEpilogueGridSyncIndex = 1>`. 两个通道完全解耦, dispatch 的 grid sync 不会阻塞 epilogue 的 grid sync.

**与其他组件的交互**: grid sync 是 NVLink barrier 的前置/后置条件 ,`nvlink_barrier` 函数可以选择性地在前面和后面各做一次 grid sync(由 `sync_prologue` / `sync_epilogue` 参数控制), 保证跨 rank 操作前所有 SM 已对齐.
get_nvl_barrier_counter_ptr()/get_nvl_barrier_signal_ptr(phase)
指向一个 32 位整数, 其低 2 位编码当前 NVLink barrier 的**相位**(bit 0)和**信号符号**(bit 1), 高 30 位记录 arrive 计数. 在`deep_gemm/include/deep_gemm/comm/barrier.cuh`中`nvlink_barrier`使用,  仅 **SM 0** 操作这个计数器. `status & 1` 提取相位信息；`status >> 1` 提取符号；每次 barrier 完成后 SM 0 的 thread 0 执行 `red_add(counter_ptr, 1)` 翻转状态.

**调用时机**: 在 dispatch 拉取 token 前(`kBeforeDispatchPullBarrierTag`)、combine 归约前(`kBeforeCombineReduceBarrierTag`)、workspace 清理后(`kAfterWorkspaceCleanBarrierTag`)各调用一次 nvlink barrier, 对应 3 次跨 rank 同步点. NVL barrier counter 在每次调用间被自动翻转为下一个相位.

Expert收发计数器

`get_expert_send_count_ptr(expert_idx)`
`expert_idx` 是全局 expert 索引(0 到 `num_experts - 1`, 跨所有 rank). 它指向一个 `uint64_t`, 其编码非常精巧: **高 32 位**记录 SM 的 commit 计数(每个 SM +1), **低 32 位**累加该 expert 收到的 token 数. 两条信息通过一次 `atomic_add` 同时更新: `send_value = (1ull << 32) | smem_expert_count[i]`.

![图片](assets/d61582b6bca8.png)

**调用时机**:

**Dispatch 写入**: 每个 dispatch thread 将它在 shared memory 中统计的该 expert 的本地 token 数原子性地累加到全局 workspace, 同时高 32 位 +1 表示"这个 SM 已经报告完毕". 返回值是原子操作前的旧值, 低 32 位被用作该 SM 在 expert 的 source indices 数组中的**起始偏移**.

**SM0 读取**: 在 grid sync 之后, SM 0 读取每个 expert 的 send count, 将低 32 位(总 token 数)分发到对应 rank 的 `expert_recv_count` 中. 这里 `expert_status & 0xffffffff` 提取 token 数, `expert_status >> 32` 隐含的 SM 计数在上层调度器 `fetch_expert_recv_count` 中被用来判断数据是否就绪.

**清理阶段清零**: SM 0 在 combine 完成后将 send count 清零, 准备下一轮 kernel 调用.

**与其他组件的交互**: `get_expert_recv_count_ptr` 和 `get_expert_recv_count_sum_ptr` 的基址都是通过调用 `get_expert_send_count_ptr` 来计算的, 体现了链式偏移设计.
`get_expert_recv_count_ptr(rank_idx, expert_idx)`
`rank_idx` 是源 rank 索引(0 到 `num_ranks - 1`), `expert_idx` 是**本 rank 的本地 expert 索引**(0 到 `num_experts_per_rank - 1`). 注意与 `send_count` 的索引空间不同, 这里是 per-rank 的本地空间. 它向一个 `uint64_t`, 存储来自特定 rank 的特定 expert 的接收 token 计数.

**调用时机**:

**SM0 分发**: SM 0 在 grid sync 后, 将 `get_expert_send_count_ptr` 读到的值通过 `sym_buffer.map` 写入**对应 rank 的 recv count 槽**. 这里 `sym_buffer.map` 把本地指针映射为远端 rank 的对称地址, 实现了"本 rank 的 send count 就是目标 rank 的 recv count 来源".

**Dispatch 拉取阶段读取**: dispatch warp 在拉取 token 前, 读取所有 rank 对当前 expert 的 recv count, 存入寄存器数组 `stored_rank_count`. 这些值被 min-peeling 算法用于确定每个 token 来自哪个 rank.

**清理阶段清零**: 在每轮结束后清零, 准备下一轮.
`get_expert_recv_count_sum_ptr(expert_idx)`
`expert_idx` 是本 rank 的本地 expert 索引. 指向一个 `uint64_t`, 存储**所有 rank 发来的该 expert 的汇总 token 数**(= `get_num_tokens` 的运行时值). 同样高 32 位是 SM 计数(`kNumSMs * kNumRanks`), 低 32 位是实际 token 数.

**调用时机**:

**调度器自旋等待**: `fetch_expert_recv_count` 函数对每个 expert 执行 `ld_volatile` 自旋循环, 直到高 32 位达到 `kNumSMs * kNumRanks`(即所有 SM 和所有 rank 都完成了 atomic_add). 这是所有 GEMM warp 进入主循环的Barrier, 调度器必须等所有 expert 的 token 计数就绪才能开始分配 block.

**SM0 远端聚合**: SM 0 在分发 recv count 的同时, 通过 `atomic_add_sys` 把所有 rank 的 send count 聚合到 sum 中.

**清理阶段读取和清零**: 清理 workspace 时先读出 token 数以确定需要清理多少个 block, 然后清零.

L1/L2到达计数器

`get_l1_arrival_count_ptr(pool_block_idx)`
`pool_block_idx` 是 pool 中的**全局 block 索引**(从 0 到该 expert 所占 block 数 - 1). 由 `expert_pool_block_offset + token_idx_in_expert / BLOCK_M` 计算得出(`expert_pool_block_offset` 是当前 expert 之前所有 expert 占用的 block 总数). 它指向一个 `uint32_t`, 作为 counter. 初始值 0, 目标值 = `valid_m`(该 block 实际包含的有效 token 行数, ≤ BLOCK_M).

**调用时机**:

**Dispatch 写入**: dispatch warp 在将一个 token 的 FP8 数据 TMA-store 到 L1 pool 后, 执行 `ptx::red_add_rel(ptr, 1)` 原子递增该 counter. 这是 dispatch 到 TMA-A 的生产者-消费者信号.

**TMA-A warp 自旋等待**: L1 阶段的 TMA-A warp 在处理每个 pool block 之前, 执行 `while (ptx::ld_acq(ptr) != expected)` 自旋, 其中`expected = get_valid_m<false>()`. 只有当该 block 的所有 token 都已被 dispatch 拉取并存储后, TMA-A 才能安全地加载该 block.

**清理阶段清零**: 每个 expert 处理完后清零.

**设计要点**: L1 arrival 使用**计数**而非位图, 因为每个 pool block 内包含多个 token(最多 BLOCK_M 个), dispatch warp 逐个 token 递增, 只需要一个整型 counter 即可. `red_add_rel` 的 `.rel` 语义保证了 release 顺序, dispatch 对 token data 的写入一定在 counter 递增之前对 TMA-A visible.
`get_l2_arrival_mask_ptr(pool_block_idx)`
与 L1 arrival 相同, `pool_block_idx` 是 pool 中的 block 索引. 指向一个 `uint64_t`, 作为bitmap, 每个 bit 代表一个 N-block 是否已经完成 L1 SwiGLU + store. 目标值 = `(1ull << (2 * num_k_blocks)) - 1`, 即低 `2 * num_k_blocks` 位全部置 1.

这里有一个细节, 为什么是 `2 * num_k_blocks`? : 由于SwiGLU 将 gate/up 对合并为单个输出, N 维度减半(`L1_OUT_BLOCK_N = BLOCK_N / 2`), 因此 L2 阶段需要的 N-block 数量是 L1 阶段的 2 倍. TMA-A warp 必须等待这 2 × 数量的 L1 块全部完成.

**调用时机**:

**L1 Epilogue 写入**: 每个 epilogue warpgroup 在完成一个 N-block 的 SwiGLU + TMA store 后, 通过 `ptx::red_or_rel_gpu(ptr, 1ull << n_block_idx)` 原子性地将对应 bit 置位.

**TMA-A warp 自旋等待**: L2 阶段的 TMA-A warp 在处理 pool block 前自旋 `while (ptx::ld_acq_gpu(ptr) != expected)`. L1 和 L2 阶段使用**不同的等待原语**: L1 用 `ld_acq`(SM 内 + L1 cache), L2 用 `ld_acq_gpu`(GPU 全局 scope), 因为 L2 arrival 的写入者可能在不同的 TPC 上.

**清理阶段清零**: 每个 expert 处理完后清零.

**设计要点**: L2 arrival 使用**bitmap**而非计数, 因为每个 pool block 的 N 维度被固定划分为 `num_k_blocks`(或 2×)个独立处理的子块, 每个子块由一个独立的 epilogue warpgroup 处理. bitmap允许**并行且无序**的完成通知, epilogue warpgroup 不需要协调顺序, 只需要 atomically OR 自己的 bit. 这与 L1 arrival 的计数器形成了对比: L1 的 token 是一个接一个由 dispatch 写入的(串行), 所以用计数器；L2 的 N-block 是并行的, 所以用bitmap.

Token 索引和Metadata

`get_src_token_topk_idx_ptr(expert_idx, rank_idx, token_idx)`
采用三维索引`expert_idx`(本地 expert)、`rank_idx`(源 rank)、`token_idx`(在该 rank 该 expert 内的 token 序号). 指向一个 `uint32_t`, 存储的是 `token_topk_idx` , 即该 token 在**源 rank** 的全局 topk 索引数组中的位置(= `token_idx * kNumTopk + topk_idx`).

**调用时机**:

**远端 Dispatch 写入**: 在 dispatch 阶段, 每个 SM 在完成本地 expert 计数后, 为每个 token 计算它应该被发送到哪个 rank 的哪个 expert, 然后通过 `*sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx` 将**跨 rank 写入**对方的 workspace. 这里 `sym_buffer.map` 把本地指针映射为远端 rank 的对称地址.

**本地 Dispatch 拉取时读取**: dispatch warp 在拉取 token 时, 读取这个索引来获取**源 token 的 topk 槽号**和**源 token 索引**: `src_token_idx = src_token_topk_idx / kNumTopk`, `src_topk_idx = src_token_topk_idx % kNumTopk`. 这两个值随后被用于从源 rank 拉取 token 数据和 topk weight.

**与其他组件的交互**: 这是 dispatch 阶段跨 rank 通信的核心索引结构. 一个 rank 的 dispatch warp 写入远端 rank 的 workspace, 另一个 rank 的 dispatch warp 后续读取它, 通过 NVLink 的对称内存模型完成零拷贝的数据交换, 无需显式的 send/recv.
`get_token_src_metadata_ptr(pool_token_idx)`
`pool_token_idx` 是 **L2 pool 中的全局 token 索引**(跨所有 expert 的 pool), 由 `pool_block_idx * BLOCK_M + token_idx_in_block` 计算得出. 它指向一个 `TokenSrcMetadata` 结构体(12 字节), 包含三个 `uint32_t` 字段:

```
struct TokenSrcMetadata {    uint32_t rank_idx;   // 源 rank: 该 token 来自哪个 GPU    uint32_t token_idx;  // 源 token: 在源 rank 内的 token 序号    uint32_t topk_idx;   // topk 槽: 该 token 的第几个 top-k 选择};
```

**调用时机**:

**Dispatch 写入**: dispatch warp 在将一个 token 存入 L1 pool 后, 把该 token 的源信息写入 metadata. 这三个值在 GEMM 计算期间被"冷藏", 直到 epilogue 阶段才被重新读取.

**L2 Epilogue Combine读取**: L2 epilogue warp 在完成 BF16 转换后, 根据 `m_idx + m_idx_in_block`(pool token 索引)读取 metadata, 获取三个目标路由信息. 然后用 `combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx)` 定位到远端 combine buffer 的正确位置, 通过 `*sym_buffer.map(dst_ptr, dst_rank_idx) = packed` 写入结果.

**与其他组件的交互**: `TokenSrcMetadata` 是 dispatch 阶段和 combine 阶段之间的唯一信息桥梁. dispatch 阶段只知道"这个 pool token 来自远程 rank X 的第 Y 个 token 的第 Z 个 topk 选择", 而 combine 阶段需要源路由, 把计算结果精确地放回原始发送方的正确 topk 槽. 没有 metadata, combine 阶段无法知道计算结果应该写入哪里.

#### 3.3.2 Buffer

在`Layout::workspace`区域之后紧接着的就是一个buffer缓冲区, 每段由 `layout::Buffer(data_layout, outer, inner, base_ptr)` 生成, `get_end_ptr()` 串到下一段起点. l1_output TMA 描述符在逻辑上把 L1 kernel 的 FP8 输出映射到 l2_token_buffer 同一物理区(不同 swizzle). 首先它通过如下`Data`结构体描述一个记录

```
struct Data {    uint32_t num_bytes;            // 每个逻辑元素占用的字节数    bool require_tma_alignment;    // 是否要求 16 字节 TMA 对齐    void* base;                    // 运行时基地址（nullptr = 仅用于尺寸计算）};
```

`Data` 描述单个 token（或单个 slot）的存储布局, 例如:

![图片](assets/39d9236ba75f.png)

整个Buffer的10段布局如下图所示:
![图片](assets/73ae1589aea9.png)

Input Buffer Pool

[1] input_token_buffer
Shape:`[num_tokens, hidden]` Dtype: `FP8 E4M3`

用途: 输入token的激活

写: 通过Host写入, `buffer.x[:num_tokens].copy_(x_fp8)`

读: dispatch pull 阶段, 通过 sym_buffer.map() 跨 rank 访问

```
ptx::tma_load_1d(pull_buffer, sym_buffer.map(    input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),    dst_rank_idx), pull_mbarrier, kHidden);
```
[2] input_sf_buffer
Shape:`[num_tokens, hidden/128]` Dtype: `UE8M0`

用途:输入token的Scale Factor(K-major)

写: 通过Host写入, `buffer.x_sf[:num_tokens].copy_(x_sf)`

读: dispatch pull 阶段, 普通 LD/ST (非 TMA)

```
const auto remote_sf_ptr = sym_buffer.map(    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),    current_rank_in_expert_idx);
```

⚠️注意: 此 SF 为 **K-major** 布局(按 hidden 维度排列), 而 `l1_sf_buffer` 和 `l2_sf_buffer` 为 **M-major**. 转换在 dispatch pull 写入 `l1_sf_buffer` 时通过 `transform_sf_token_idx` 完成.
[3] input_topk_idx_buffer
Shape:`[num_tokens, num_topk]` Dtype: `int64`

用途:Expert 路由索引, 用于确定每个 token 被路由到哪些 expert。包含值为 `-1` 表示该 slot 未使用（masked）

写: Host 写入,`buffer.x_sf[:num_tokens].copy_(x_sf)`

读: dispatch warp 阶段, 通过 __ldg() 读取

```
  __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx)
```
[4] input_topk_weights_buffer
Shape:`[num_tokens, num_topk]` Dtype: `float32`

用途:Top-k 权重

写: Host 写入,`buffer.topk_weights[:num_tokens].copy_(topk_weights)`

读: dispatch pull 阶段, 与 token 一起拉取到本地, 存入`[7]l1_topk_weights_buffer`

```
sym_buffer.map(   input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,   dst_rank_idx)
```

L1 Pool(Linear1 输入/输出缓冲区)

[5] l1_token_buffer
Shape:`[max_pool_tokens][kHidden]` Dtype: `FP8`

用途: 存储所有被 dispatch 到本 rank 的 token. **这是整个 buffer 中最大的段**.

写: dispatch pull warp, 通过TMA执行

```
ptx::tma_store_1d(    l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
```

读: TMA-Producer A warp (via TMA descriptor → tensor_map_l1_acts)

```
tma::copy<BLOCK_K, LOAD_BLOCK_M, swizzle>(    &tensor_map_l1_acts, full_barriers[stage], smem_a[stage],    k_idx, m_idx, 2);
```

数据交互方式: dispatch pull → **l1_token_buffer** → GEMM TMA-Producer A load → MMA
[6] l1_sf_buffer
Shape:`[max_pool_tokens][kHidden/128]` Dtype: `UE8M0`

用途:L1 Pool Scale Factor(M-major)

写: dispatch pull warp (普通 LD/ST)

```
local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];//然后通过 transform_sf_token_idx 做 UTCCP 4×32 转置地址映射
```

读:  TMA-Producer A warp (via TMA descriptor → tensor_map_l1_acts_sf)

```
tma::copy<SF_BLOCK_M, 1, 0>(   &tensor_map_l1_acts_sf, full_barriers[stage], smem_sfa[stage],   sfa_m_idx, sfa_k_idx, 2);// 加载后通过 UTCCP 拷贝到 TMEM 的 SFA 列
```
[7] l1_topk_weights_buffer
Shape:`[max_pool_tokens]` Dtype: `float`

用途: Pool Token 的 Top-k Weight

写: dispatch pull warp

```
*l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
```

读: L1 epilogue warp (SwiGLU 阶段), L1 epilogue 在做 SwiGLU 激活时, 需要乘上每个 token 的 top-k weight: `silu(gate) × up × weight`.

```
stored_cached_weight = *l1_topk_weights_buffer    .get_data_buffer(m_idx + ...).get_base_ptr<float>();
```

L2 Pool (Linear2 输入/输出缓冲区，复用 L1 输出)

[8] l2_token_buffer
Shape:`[max_pool_tokens][L1_OUT_BLOCK_N]` Dtype: `BF16`

用途:  L2 Pool Token(FP8 Intermediate), 其中`l2_token_buffer` 与 `l1_token_buffer` 使用**同一个物理 tensor**(l2_acts view), 但 shape 和 stride 不同.

写:   L1 epilogue (TMA store)

```
SM90_TMA_STORE_2D::copy(&tensor_map_l1_output, smem_cd, out_n_idx, m_idx);// tensor_map_l1_output 指向 l2_acts (即 l2_token_buffer), N 维度 = intermediate_hidden// 但 swizzle mode 是 halved (64 vs 128), 因为 post-SwiGLU 的 N 是 BLOCK_N/2
```

读:   TMA-Producer-A warp (L2 阶段)

```
tma::copy<BLOCK_K, LOAD_BLOCK_M, swizzle>(    &tensor_map_l2_acts, full_barriers[stage], smem_a[stage], k_idx, m_idx, 2);
```

数据交互方式:  L1 MMA → L1 epilogue → TMA store → **l2_token_buffer** → L2 TMA-A load → L2 MMA
[9] l2_sf_buffer
Shape:`[max_pool_tokens][L2_SHAPE_K/32]` Dtype: `UE8M0`

用途: 2 Pool Scale Factor(M-major)

写: L1 epilogue warp (SwiGLU 量化后)

```
    sf_base_ptr[sf_addr] = (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);    → 将 float SF 转换为 UE8M0 格式, 写入 M-major 布局
```

读: TMA-A warp (L2 阶段)

```
tma::copy<SF_BLOCK_M, 1, 0>(    &tensor_map_l2_acts_sf, full_barriers[stage], smem_sfa[stage],    sfa_m_idx, sfa_k_idx, 2);
```

Combine Buffer

combine_token_buffer
Shape:`[kNumTopk][num_tokens][kHidden]` Dtype: `BF16`

用途:EP Combine 输出缓冲区

写:L2 epilogue warp (BF16 write-back via NVLink)

```
//获取 (dst_rank, dst_token, dst_topk)src_metadata = get_token_src_metadata_ptr(pool_token_idx);const auto dst_token = combine_token_buffer    .get_rank_buffer(dst_topk_idx)     // 按 topk slot 选择 rank 子区    .get_data_buffer(dst_token_idx);    // 按 token 索引定位*sym_buffer.map(dst_ptr, dst_rank_idx) = packed;  // 跨 rank NVLink 写
```

读: Combine 阶段 (本 rank 本地读), TMA load 各 topk slot 的 BF16 token → float 累加 → TMA store 到 y

```
combine_token_buffer.get_rank_buffer(slot_idx)                    .get_data_buffer(token_idx).get_base_ptr();
```

数据交互方式:

L2 MMA → L2 epilogue → NVLink write → **combine_token_buffer** (远程 rank 的 buffer)

NVLink write (远程 rank 的 L2 epilogue)→本 rank 的**combine_token_buffer** → Combine warp TMA load → float reduce → TMA store → output y

## 4. 详细的代码分析

接下来我们对计算中的5个warp进行详细的分析.

`sm100_fp8_fp4_mega_moe_impl` 是一个 **持久化(persistent)融合(fused) kernel**, 在单个 grid launch 内依次完成: Dispatch → Linear1 GEMM(+SwiGLU) → Linear2 GEMM → Combine. 整个 grid 的 warp 按 `warp_idx` 被划分为 5 种角色. 这里有一个细节, MegaMoE针对不同的warp分配了不同的寄存器

![图片](assets/80e50049bd8d.png)

为什么 epilogue 用 208 寄存器？
SwiGLU 需要同时处理 2 组 gate/up 对(8 个 float)

Amax reduction 需要缓存

Combine 阶段的 topk 累加需要大量临时寄存器

Register spilling 代价极高，宁愿多分配

每个warp的功能已经在3.1节中详细介绍了, 这里从代码层面详细展开并分析其中的一些细节.

### 4.1 Dispatch Warp

![图片](assets/071f970c6b43.png)

#### 4.1.1 统计本 SM 发往 expert i 的 token 数

原始的输入为 N 个 token, 每个 token 有自己的`topk_idx`. 对于Dispatch Warp需要并行的处理它们, 并统计出每个expert有多少个token.  这里采用了一个函数

```
// 第 366–383 行const auto read_topk_idx = [&](const auto& process) {    #pragma unroll    for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;         i < num_tokens;         i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {        int expert_idx = -1;        if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {            expert_idx = static_cast<int>(                __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));            if (expert_idx >= 0)                process(i * kNumTopk + lane_idx, expert_idx);        }        __syncwarp();    }};
```

它按照 `(sm_idx, warp_idx)` 组成的全局 warp id 从输入 token 序列中以 `kNumTokensPerWarp` 为步长跳跃地取走 token, 保证全 grid 中所有 dispatch warp 共同均匀覆盖 `num_tokens × kNumTopk` 个条目. warp 内 32 个 lane 被划分成 `kNumTokensPerWarp` 组, 每组 `kNumTopk` 个 lane, lane 内用 `lane_idx / kNumTopk` 解出 token offset, 用 `lane_idx % kNumTopk` 解出 topk 槽位. 并且它通过`__ldg` 从 global 读取 topk 索引, 减少冲突.

然后`smem_expert_count` 是共享内存上的 `kNumExperts` 长数组, 对于每个线程直接atomicAdd即可

```
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {   atomicAdd_block(smem_expert_count + expert_idx, 1);});ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
```

渣注
这里有一个细节, 它采用了两级进行统计的方式, 首先在SM内部进行统计, 这样atomicAdd会比直接global mem Atomic 快很多. 做完了以后后面一步再算全局整个Rank的.

然后barrier这里使用了 `kDispatchBarrierIdx = 0`, 这是 **intra-SM** 的硬件 barrier 索引(`bar.sync` 的 name 字段). Dispatch warp 在 intra-SM 内共用同一个 named barrier.

#### 4.1.2 本地计数 → 全局偏移

用 atomicAdd 向全局 workspace 写入 expert 发送计数`expert_send_count[i]` .

```
#pragma unrollfor (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {    const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);    smem_expert_count[i] = static_cast<uint32_t>(        ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));}ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
```

这里的64bit值也是一个技巧, 高32为表示已上报的 SM 计数器, 低32位为第一步计算的本 rank 在该 expert 上累计的 token 数`smem_expert_count[i]`. `send_value = (1ull << 32) | local_count`: 一个 64-bit 加法 **同时** 把本 CTA 的 token 计数加到低 32 位、把「SM 计数」加到高 32 位.

`ptx::atomic_add(...)` 返回 **加法前** 的旧值, 其低 32 位就是「本 SM 在该 expert 上的全局起始偏移」, 被写回 `smem_expert_count[i]` 覆盖原值, 作为下一步 dst slot 的基址.  具体来说, 全局的这个计数器相当于一个发号器的作用. 例如当前值为`A1`, SM 1 发送atomicAdd后, 计数器更新为`A1+sm1_count` ,同时返回旧值`A1`, 此时SM 1就知道在`[A1, A1+sm0_count)`这段区间作为接收数据的段, `A1`作为在远端expert slot上的偏移起始点. 后续 SM2 发送atomicAdd将会返回`A1+sm1_count`, 并以此作为SM1在远端expert slot上的偏移起始点.

#### 4.1.3 写远端src_token_topk_idx

根据前一步的起始偏移就可以将本地token的`topk_idx`写入到远端. 为什么要将topk idx写到 **远端**? 因为发送方 rank 知道自己往哪个 expert 发了什么 token, 但 expert 归属的 **接收方 rank** 才是后续 pull 的执行主体, 它需要直接读本地 workspace 就能查到每个 slot 的源 index, 避免在 pull 时再做一次 NVLink 反查.

```
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {    const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;    const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);    const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(        expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);    *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;});
```

其中`dst_rank_idx = expert_idx / kNumExpertsPerRank`: 目标 expert 所在的 rank. `dst_slot_idx = atomicAdd_block(...)`表示在 **本 SM 的偏移(刚写回的 smem_expert_count)** 基础上再递增, 得到 `[SM 偏移, SM 偏移 + 本 SM 计数)` 区间内唯一的 slot；

`workspace.get_src_token_topk_idx_ptr(local_expert, src_rank, slot)`: 在 **目标 rank** 的 workspace 上定位槽位. 布局为 `[local_expert][src_rank][slot]`；其中 `src_rank = sym_buffer.rank_idx`, 即 **本 rank 自己的编号**. `sym_buffer.map(dst_ptr, dst_rank_idx)` 把本地 workspace 指针重映射到 `dst_rank_idx` 这一远端 rank 的对应地址(symmetric buffer 假设所有 rank 共享同样的虚拟布局). 写入通过 NVLink 自然走远端；

写入值 `token_topk_idx = token_idx * kNumTopk + topk_idx` 是 **源端** 的全局定位信息, 接收端后续用它回查 token 主体.

至此, 本 rank 告诉所有目标 rank「你要处理我的哪些 token」.

#### 4.1.4 SM0聚合

首先会进行一个`grid_sync`, 完成后, 第二步中的`expert_send_count[i]`已经是本 rank **聚合后**的终值(kNumSMs 个贡献者已全部加入). 跨 rank 写只需用SM0写 1 次, 它将更新接收端的`expert_recv_count`, 这是一个`[rank, expert_idx]`的二维数组. 它是 **接收端** 视角的「从 src_rank 收到的 token 数」, 用于后续 round-robin.

然后它还做了一次atomicAdd更新远端的`expert_recv_count_sum[local_expert]`, 是接收端的总计数器: 高 32 位累计 `kNumSMs`(每个发送 rank 贡献 `kNumSMs`), 低 32 位累计 token 数. 当高 32 位 == `kNumSMs * kNumRanks` 时, 说明所有 rank 所有 SM 都已完成上报, 此时低 32 位即全局最终 token 数. 它代表所有 rank 发来的该 expert 的汇总 token 数. 在后续的scheduler中会根据`fetch_expert_recv_count` 函数对每个 expert 执行 `ld_volatile` 自旋循环, 当到齐后触发block分发调度并触发后续的GEMM运算.

```
if (sm_idx == 0) {    #pragma unroll    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {        const auto dst_rank_idx = i / kNumExpertsPerRank;        const auto dst_local_expert_idx = i % kNumExpertsPerRank;        const auto expert_status = *workspace.get_expert_send_count_ptr(i);        // (1) 告诉远端: 我这个 rank 给你的 local_expert 发了多少 token        *sym_buffer.map(            workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),            dst_rank_idx) = expert_status & 0xffffffff;        // (2) 聚合到远端的 sum 计数器: 低 32 位 = token 数累加；高 32 位 = 完成 SM 数累加        // atomic_add_sys(.sys scope)确保跨 rank 的一致性, 而普通 atomic_add_rel 只保证同设备内.         ptx::atomic_add_sys(            sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),            expert_status);    }}ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
```

#### 4.1.5 NVLink Barrier

然后进行NVLink Barrier,确保 **所有 rank** 的 `expert_recv_count[*][*]` 和 `src_token_topk_idx` 都已被各自 SM 0 写入完毕, 后续拉取 token 时读到的一定是最终值.

```
comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,                     kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(    workspace, sym_buffer, sm_idx, thread_idx,    [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },    /* sync_prologue = */ false,                 // 上一步已 grid_sync 过    /* sync_epilogue = */ true                   // 结束后再 grid_sync);
```

后面还有一个与 Epilogue Warp的屏障, 目的是 **阻止 epilogue 阶段的 NVLink barrier 与当前 pull barrier 相互影响**

```
ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
```

#### 4.1.6 Pull token
4.1.6.1 初始化
将其它 rank 命中本地 expert 的 token 主体(FP8 权重 + scale factor + topk 权重)拉回到本地的**L1 token pool**, 同时设置`l1_arrive_cnt`触发TMA-Producer A消费.

首先它有一个上下文初始化的过程:

```
// Pull token data and SF from remote ranks into local L1 bufferuint32_t pull_mbarrier_phase = 0;// `pull_buffer`: 每个 warp 在 smem 的私有 1-token 暂存区, 作为 TMA load 的目标const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);// `pull_mbarrier`: 每 warp 一个 transaction mbarrier, 用于 TMA load 完成信号const auto pull_mbarrier = dispatch_barriers[warp_idx];// Cache expert token counts in registers (same pattern as scheduler)scheduler.fetch_expert_recv_count();// Per-rank counts for current expert (re-loaded when expert changes)// 当 `kNumRanks > 32` 时每个 lane 需要管理多个 rank 的计数；否则 `=1`；constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);int current_expert_idx = -1;// 当前 expert 上, lane `lane_idx` 对第 `i*32+lane_idx` 号远端 rank 发来的 token 数的本地缓存uint32_t stored_rank_count[kNumRanksPerLane] = {};// 当前 expert 在全局 token 序列中的 `[start, end)` 区间uint32_t expert_start_idx = 0, expert_end_idx = 0;// 当前 expert 在 L1 token pool 中的起始 block 偏移(按 `BLOCK_M` 对齐的 block 计)uint32_t expert_pool_block_offset = 0;// 跨 grid 的 dispatch warp 总数, 作为 token-level 分片步长constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
```

其中`scheduler.fetch_expert_recv_count()` 等待 `expert_recv_count_sum` 的高 32 位 == `kNumSMs * kNumRanks`, 同时把最终 token 数缓存到 `stored_num_tokens_per_expert[i]`；每个 lane 管理 `expert_idx = i*32 + lane_idx` 的 expert, 这一缓存随后也给 `fetch_next_l1_block`、`get_pool_block_offset` 使用.
4.1.6.2 主循环
接下来进入主循环, 以token级进行分片处理, 每个 dispatch warp 以 `kNumGlobalWarps` 为步长、`global_warp_id` 为起点遍历本 rank 合并后的 token 序列(所有 local expert 首尾相连), 并且在内部推进expert指针

```
for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {    int old_expert_idx = current_expert_idx;    while (token_idx >= expert_end_idx) {        if (++ current_expert_idx >= kNumExpertsPerRank)            break;        // Update pool block offset for the new expert        expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);        // Move start and end to the next expert        expert_start_idx = expert_end_idx;        expert_end_idx += scheduler.get_num_tokens(current_expert_idx);    }     // Finish all tokens    if (current_expert_idx >= kNumExpertsPerRank) break;        // 接下来的处理..... }
```

整个`token_idx` 单调增长, while 从上次 expert 继续向前扫, 切换时增量累加 `expert_pool_block_offset`, 我们以一个简单的例子来描述`BLOCK_M = 8`的情况下, token如下所示:

![图片](assets/99f5984bf126.png)

可以看到上图的表格, 这是一次单调推进: token_idx 顺序增长, 所以每次 while 只需从上次 expert 继续往前扫. 专家切换时, 用前一个专家的 `(end-start)` 算出它占了多少 m-block, 累加到 `expert_pool_block_offset`, 然后把 `(start, end)` 滚到下一个专家的区间.

接下来它会在Expert切换时, 触发 per-rank 计数重载, lane 从 `workspace.get_expert_recv_count_ptr(j, cur)` 取来它负责的那些 rank 的计数器, 更新per-rank的计数器`expert_recv_count[src_rank][local_expert]`. 例如在处理 Expert 0 的 10 个token时, 如上图所示这 10 个 token 可能来自不同的rank, 下面这段代码就会读取时每个 lane 拿对应 rank 的值到寄存器`stored_rank_count[i]`, 为接下来的 min-peeling 提供数据.

```
if (old_expert_idx != current_expert_idx) {    old_expert_idx = current_expert_idx;    #pragma unroll    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {        const uint32_t j = i * 32 + lane_idx;        stored_rank_count[i] = j < kNumRanks ?            static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;    }}
```
4.1.6.3 Min-Peeling 算法
这一部分实质是在处理某个Expert的token按照哪种顺序通过NVLink拉取到本地, 并且尽量考虑在整个拉取的过程中负载均衡. 前一步获取了单个 Expert 内token来自于哪些rank的计数器`stored_rank_count[i]`. 然后这一步就是通过轮询的方式将不同rank的token对应到这个专家的相应的slot, 具体计算流程示例如下图所示

![图片](assets/46dac9b9937d.png)

前一步生成的`stored_rank_count`作为每个 token 迭代的初始值, 拷贝到工作副本 `remaining[]`. 然后以 min-peeling 方式在 rank 间 round-robin 选取当前 token 来源.  例如第一轮的时候, 首先找到还`remaining[]`为空的rank作为`active_rank`, 当前轮每个rank都是active的, 因此`num_active_ranks = 4`, 然后选取所有`active_rank`中最小的长度`minlength = 1`(Rank3最小长度), 并通过`num_active_ranks x min_length`得到本轮需要拉取的token数, 然后进行拉取. 接下来按照同样的算法迭代, 直到完成. 下面我们来看详细的代码实现.

初始化阶段如下, 它会拷贝`stored_rank_count`到`remaining[]`中, 并且把正在处理的token_idx转换为在当前Expert的相对坐标`slot_idx`.

```
uint32_t remaining[kNumRanksPerLane];for (uint32_t i = 0; i < kNumRanksPerLane; ++i)    remaining[i] = stored_rank_count[i];   // 创建副本uint32_t offset              = 0;uint32_t token_idx_in_expert = token_idx - expert_start_idx; // 在本专家的相对坐标uint32_t slot_idx            = token_idx_in_expert;   uint32_t token_idx_in_rank;
```

然后是一个循环, 在循环内, 首先找到所有活跃 rank 中 remaining 的最小值 length 以及统计active_ranks. 由于是并行的每个lane处理`kNumRanksPerLane`, 它的做法是先在 lane 内进行统计, 然后再采用warp level的reduce `__reduce_min_sync`/`__reduce_add_sync`计算.

```
        // 先 lane 内聚合活跃 rank 的个数与最小值，再全 warp reduce        uint32_t num_actives_in_lane = 0;        uint32_t min_in_lane = 0xffffffff;#pragma unroll        for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {          num_actives_in_lane += remaining[i] > 0;          if (remaining[i] > 0)            min_in_lane = cute::min(min_in_lane, remaining[i]);        }                //warp reduce        const uint32_t num_active_ranks =            __reduce_add_sync(0xffffffff, num_actives_in_lane);        const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);                // 本轮 token 数 = length × num_active_ranks        const uint32_t num_round_tokens = length * num_active_ranks;
```

然后根据 `num_round_tokens` 处理命中的token, 即`slot_idx < num_round_tokens`. 待处理的数据其实构成一个`[length][num_active_ranks]`的二维网格, 如下所示:

![图片](assets/24f6d78af2f5.png)

图中 `num_active_ranks = 4`, `length = 2`. 构成一个`length`行 x `active_ranks`列的二维结构, `slot_idx`按照行优先排序.

```
        if (slot_idx < num_round_tokens) {  // 本轮命中          const uint32_t slot_idx_in_round = slot_idx % num_active_ranks; // 本轮内第几个 slot          uint32_t num_seen_ranks = 0;          current_rank_in_expert_idx = 0;#pragma unroll          for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {            const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);            const uint32_t num_active_lanes = __popc(mask);            if (slot_idx_in_round >= num_seen_ranks and                slot_idx_in_round < num_seen_ranks + num_active_lanes)              current_rank_in_expert_idx =                  i * 32 +                  __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);            num_seen_ranks += num_active_lanes;          }          token_idx_in_rank = offset + (slot_idx / num_active_ranks);          break;        }
```

ballot + fns
这里采用了一个CUDA并行计算中一个比较常见`__ballot_sync` 和 `__fns`处理方式. 它们是 warp 级的原语, 用于高效的线程间通信和位操作.

` __ballot_sync(unsigned mask, int predicate)`可以用于“投票”的原语. 在 Warp 内执行条件测试, 如果 predicate 为真(非 0), 对应线程设置结果整数中的相应位. 它映射生成一个 32 位的掩码, 最高对应 32 个线程的投票结果. 代码中通过`remaining[i] > 0`作为predicate 进行投票, 最终获取的mask就是哪些`rank`是active的. 然后通过`__popc(mask)`即可获得活跃 rank 数.

`__fns(unsigned mask, unsigned base, k )` fns即find-n-th-set, 通过这个函数在找出 32 位整数中第 n 位为 1 的位置. 通过这样的方式可以找到第 `(slot - num_seen) + 1` 个置位的 bit 位置, 得到组内 rank 偏移, 再 `+ i*32` 得到全局 rank 编号.

这一轮做完了, 则更新减掉本轮消耗的 slot, 累加 rank 内偏移, 并给所有 rank 减去 length 为下一轮做准备

```
        slot_idx -= num_round_tokens;        offset += length;#pragma unroll        for (uint32_t i = 0; i < kNumRanksPerLane; ++i)          remaining[i] -= cute::min(remaining[i], length);      }
```

整个算法实现来看, 它在 warp 内 32 个 lane 完全协同完成一次查询. 处理复杂度 `O(kNumRanks)`(在最坏情况下要 peel 掉 `kNumRanks-1` 次), 通常 peel 次数远小于 rank 数. 最后它很简单的实现了**load balancing**: 即使某个 rank 发出大量 token、另一些 rank 发得少, 也会按 round-robin 交错. 这对后续的 NVLink 带宽利用很关键, 同一时间所有 dispatch warp 更可能命中 **不同** 远端 rank, 避免单一 rank 通信瓶颈.
4.1.6.4 Pull token
这一步将其它 rank 命中本地 expert 的 token 主体(FP8 权重 + scale factor + topk 权重)拉回到本地的**L1 token pool**, 同时设置`l1_arrive_cnt`触发TMA-Producer A消费.

首先它将读取 `src_token_topk_idx`:

```
// Read source token-topk index (written by remote dispatch via NVLink)const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;
```

`get_src_token_topk_idx_ptr` 采用三维索引`expert_idx`(本地 expert)、`rank_idx`(源 rank)、`token_idx`(在该 rank 该 expert 内的 token 序号). 指向一个 uint32_t, 存储的是 `token_topk_idx` , 即该 token 在源 rank 的全局 topk 索引数组中的位置. 值是 dispatch push 阶段由远端 rank 写入的 `token_topk_idx = src_token_idx * kNumTopk + src_topk_idx`:

`src_token_idx`: 在 **远端 rank 的输入 X** 中的行号

`src_topk_idx`: 该 token 选中本专家时所占用的 topk 槽位(用于后面 combine 时回填到正确的 topk 行)

这个函数调用的参数为:

| index | 含义 |
|---|---|
| `current_expert_idx` | 本 rank 上的本地专家序号(0 ~ kNumExpertsPerRank-1) |
| `current_rank_in_expert_idx` | 该专家本轮 token 来源的"远端 rank 序号", 由 min-peeling round-robin 选出 |
| `token_idx_in_rank` | 该专家收到的、来自这个远端 rank 的第几个 token(rank 内偏移) |

TMA拉取Token
然后反解出来这两个值, 用于TMA, 其中目的地址为`smem_send_buffers`, 它是 dispatch 区的 smem, layout 为 `[kNumDispatchWarps][1 token][fp8_token_layout]`, 每 warp 一行；`get_data_buffer(0)` 取本 warp 这行的起点, 长度恰为 `kHidden` 字节(一行 token 的 FP8 数据).

源张量 `input_token_buffer` 是 `[num_max_tokens_per_rank][kHidden]` FP8 行主序张量. 通过`src_token_idx`寻址.

```
const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);// TMA load token from remote rank into shared memoryif (cute::elect_one_sync()) {    ptx::tma_load_1d(        pull_buffer.get_base_ptr(),  //dst        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),                       current_rank_in_expert_idx), //src        pull_mbarrier, kHidden);}__syncwarp();
```
直接LD ScalingFactor
SF 的总字节数 `kHidden / 32 = kNumSFUint32 × 4` 很小, 直接用 warp 并行 `LDG/STG` 拷贝.并且**与 TMA 加载 token 并行**: token 还在走 TMA 管道时, 32 个 lane 已经把 SF 从远端搬到本地 `l1_sf_buffer`.

```
constexpr uint32_t kNumSFUint32 = kHidden / 128;DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");const auto remote_sf_ptr = sym_buffer.map(    input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),    current_rank_in_expert_idx);const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();const auto sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M +    transform_sf_token_idx(token_idx_in_expert);#pragma unrollfor (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {    const uint32_t j = i * 32 + lane_idx;    if (j < kNumSFUint32)        local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];}__syncwarp();
```

UTCCP 4×32 转置
需要注意此处调用了`transform_sf_token_idx`, SF 在 pool 中不是按 token 平铺, 而是为了 SM100 UTCCP 指令的内存布局重排, 目标地址 `local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx]`: SF 以 `[sf_channel, pool_token]` 布局存储(方便后续 GEMM warp 按 `BLOCK_M × SF_CH` Tile 加载).

topk_weight拷贝
接着它还会拉取相应的`topk_weight`

```
    const auto weight = *sym_buffer.map(        input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,        current_rank_in_expert_idx);    *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
```
4.1.6.5 Pull尾处理
在尾处理阶段, 它会等待pull的TMA处理完成后, 再通过TMA把拉回的token数据存入本地的`l1_token_buffer`. 并写入token元数据.

```
    // 等 TMA load 完成    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);    ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);    // TMA store 到本地 l1_token_buffer    ptx::tma_store_1d(        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());    // 写源元数据, 用于 combine 阶段把结果送回源 rank    *workspace.get_token_src_metadata_ptr(pool_token_idx) =        {current_rank_in_expert_idx, src_token_idx, src_topk_idx};    // 等 TMA store 完成    cute::tma_store_arrive();    ptx::tma_store_wait<0>();
```

最后还有最关键的一步, 通过更新`l1_arrive_count`来通知GEMM 的 TMA-Producer A是否完成数据拉取进行后续的GEMM计算流水线

```
    ptx::red_add_rel(        workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
```

![图片](assets/5f5e651d08e6.png)

### 4.2 TMA Producer A Warp

它在 MegaMoE kernel 的 GEMM 阶段担任 **「加载激活(activation)及其 scale factor 的生产者」** , 它由调度器驱动`scheduler.for_each_block`

![图片](assets/527d5160021d.png)

#### 4.2.1 Scheduler

整个代码由调度器`scheduler.for_each_block`驱动, 同一 scheduler 驱动 TMA-A / TMA-B / MMA / Epilogue 四个 warp 共享迭代空间, 代码如下

```
  template <typename Func>  CUTLASS_DEVICE void for_each_block(Func&& func) {    // 等待所有专家计数器完成汇总    fetch_expert_recv_count();    // 从第 0 个专家开始遍历（同时初始化 current_num_tokens 和 pool offset）    set_expert_idx(0);    // 遍历当前 SM 分到的所有块    while (true) {      // 拆包返回的 tuple 到同名局部变量（CUTE_TIE_DECL 是结构化绑定宏）      CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx,                    m_block_idx, n_block_idx);      if (block_phase == BlockPhase::None) break;      // 回调签名：(BlockPhase, expert_idx, num_k_blocks, m_block_idx,      // n_block_idx)      func(block_phase, current_local_expert_idx,           block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,           m_block_idx, n_block_idx);    }  }};
```

`fetch_expert_recv_count()`自旋等待 workspace 中每个专家的 **recv count 汇总值**就绪, 它检测计数器高32位到达 `kNumSMs * kNumRanks`. 证明所有的 Rank 的所有的 SM 都完成 Dispatch 的计数后, 再把低 32 位 token 数写到寄存器数组 `stored_num_tokens_per_expert[i]`.这一步是 **dispatch → compute** 之间的全局同步点: 只有所有专家的 token 数都确定了, 才能开始划分 M 维的 block.

然后它把 expert idx 重置到第 0 个本地专家开始遍历. 然后在`while(true)`循环中反复调用 `get_next_block()`, 它返回 `(phase, expert, m_block, n_block)` 四元组. 调度顺序如下:

```
Linear1 命中 → 返回 L1 块；block_idx += kNumSMs 滚动到下一全局块Linear1 穷尽 → next_phase = Linear2, expert 回倒到当前 wave 起点Linear2 命中 → 返回 L2 块Linear2 穷尽 → next_phase = Linear1, 进入下一 wave全部结束     → 返回 BlockPhase::None, 跳出循环
```

TMA-Producer A 只负责一件事: 把 **激活 A + SFA** 通过 TMA 拉到 smem 的 A 流水线 stage. 它用 `for_each_block` 把这件事对每个 block 重复一遍, 因此整个执行逻辑被包在这个函数内

```
scheduler.for_each_block([&](const sched::BlockPhase& block_phase,                             const uint32_t& local_expert_idx,                             const uint32_t& num_k_blocks,                             const uint32_t& m_block_idx, const uint32_t& n_block_idx) {        /* ... TMA Load A 的处理 ... */});                             
```

这个函数回调参数解释如下:

![图片](assets/30f56d14fbd5.png)

整个Warp基于只有 `(expert, phase, m_block, n_block)` 在 scheduler 内部推进.

#### 4.2.2 Tensor_map 处理

对于TMA descriptor 已经在kernel启动的初始化阶段完成了, 这里有由于 Expert FFN 的两个 GEMM 数据源完全不同:
![图片](assets/80a4dac41f7f.png)

因此它会根据返回的`Phase`决定使用 L1 或者 L2 的 `tensor_map`

```
const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2                                  ? &tensor_map_l2_acts                                  : &tensor_map_l1_acts;const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2                                    ? &tensor_map_l2_acts_sf                                    : &tensor_map_l1_acts_sf;
```

然后它会计算`pool_block_idx`, `scheduler.get_current_pool_block_offset()` 返回当前 expert 在 pool 中的起始 block 偏移,  加上 `m_block_idx` 就得到本 expert 的第 `m_block_idx` 个 M 方向 tile 在共享 pool中的全局 block 编号.

```
// Compute pool block offset for this expertconst uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
```

#### 4.2.3 等待数据到达机制

L1阶段通过等待Dispatch Warp更新的`l1_arrive_cnt`触发. 而L2阶段则通过L1 Epilogue warp更新的`l2_arrive_mask`触发.

```
if (block_phase == sched::BlockPhase::Linear1) {  // L1：等该 m 块的 token 全部到齐 (l1_arrival_count == valid_m)  const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);  const auto expected = scheduler.template get_valid_m<false>();  while (ptx::ld_acq(ptr) != expected);} else {  // L2：等 l2_arrival_mask 低 2 * num_k_blocks 位全为 1  DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");  const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);  const uint64_t expected = ((1ull << num_k_blocks) << num_k_blocks) - 1;  while (ptx::ld_acq_gpu(ptr) != expected);}
```

为什么要两次左移
当 `num_k_blocks == 32` 时, `uint64_t expected = (1ull << (2 * num_k_blocks)) - 1;`会导致`1ull << 64 → UNDEFINED BEHAVIOR`. 而分两次左移,  每次 ≤ 32, 都在 [0, 63] 合法范围内.

```
uint64_t expected = ((1ull << num_k_blocks) << num_k_blocks) - 1;// num_k_blocks == 32://   (1ull << 32) = 0x100000000//   << 32         = 0 (高位溢出)//   - 1           = 0xFFFFFFFFFFFFFFFF
```

4.2.3.1 L1等待机制
`get_l1_arrival_count_ptr(pool_block_idx)` 指向 `workspace.l1_arrival_count[pool_block_idx]`, 这是一个 `uint32_t`, dispatch warp 每成功拉入一个 token 就 `red_add_rel(count_ptr, 1)`, 语义是**release**. 本处 `ptx::ld_acq` 为**acquire**语义, 与之配对形成 release/acquire 同步, 即加载到 `expected` 时, 之前 dispatch warp 对 `l1_token_buffer[pool_block_idx * BLOCK_M .. ]` 的 TMA store 对当前 warp 可见.

`expected = scheduler.get_valid_m<false>()`是指这个block的实际token数. 例如下图所示, 虽然dispatch warp会更新`l1_arrive_count`, 但是block需要按照专家对齐, 因此会有一些padding的场景, 使得实际block的token数会小于BLOCK_M, 例如下图中的 block 1 和 block 2.

![图片](assets/839c64907a75.png)

对于 Linear1, A 的 tile 是 `BLOCK_M × BLOCK_K` 的 activation；UMMA 一次消费整个 `BLOCK_M × BLOCK_K` 的数据, 不能容忍行级别的未初始化. 因此 producer 必须等整个block 填满, 即`l1_arrive_count == expected` 才能发起 TMA.
4.2.3.2 L2等待机制
首先有一个细节, 在处理L1 GEMM的时候, 采用了`gate`和`up` interleave的方式, 然后在`L1 Epilogue`中进行SwiGLU计算后, block_N 的 size 会减半, 如下图所示:

![图片](assets/45d6f1e69fe0.png)

在L2等待过程中有一个断言`DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");`, 其实在`heuristics`中也固定了`BLOCK_K = BLOCK_N = 128`. 这样的做法是要考虑在L2 GEMM的过程中的 K 维度要对齐, 这样处理后每 2 个相邻的 L1 N 子块(L1_OUT_BLOCK_N = BLOCK_N / 2 = 64)拼成 1 个 L2 K 子块(BLOCK_K = 128)的数据, 并且很容易通过`l2_arrive_mask`这个 bitmap 表示.

![图片](assets/a96ac8837d8e.png)

L2等待机制在`PR304`和`PR316`之间有一个修改. 最早的`PR304`中采用如下方式, 它通过两个连续的bit(`3ull`)按需等待2个`L1_Output`数据块的方式, 以在 L1 计算和 L2 计算之间实现重叠.

```
// pr304if (block_phase == sched::BlockPhase::Linear2) {     // The L1 output's block N is halved into `BLOCK_K / 2`, so we have to wait 2 L1 blocks' arrival     DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");     const uint64_t needed = 3ull << (k_block_idx * 2);     if ((cached_l2_arrival_mask & needed) != needed) {         const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);         do {             cached_l2_arrival_mask = ptx::ld_acq_gpu(ptr);         } while ((cached_l2_arrival_mask & needed) != needed);     } }
```

但是在`PR316`中的注释显示: `num_experts_per_wave`足够大时, 能确保在 L2 开始时 L1 已完成计算时, 这么做反而会带来负优化, 因此在`PR316`中将它移除, 但未来, 如果`num_experts_per_rank` 较小导致 `num_experts_per_wave` 不够大时, 可能会重新引入. `PR316`的等待机制为:

```
// pr316const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);// NOTES: Equivalent to `(1ull << (2 * num_k_blocks)) - 1`, but split// into two shifts to avoid undefined behavior when `num_k_blocks == 32`const uint64_t expected = ((1ull << num_k_blocks) << num_k_blocks) - 1;while (ptx::ld_acq_gpu(ptr) != expected);
```

#### 4.2.4 TMA加载数据

流水线主循环如下:

```
for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
```

其中 `advance_pipeline`: 同时递增 `k_block_idx` 并翻转流水线的 `stage_idx`(`0..kNumStages-1` 轮转)和 `phase`(当 stage 回到 0 时翻转), 而`stage_idx` 和 `phase` 是 kernel 级别的共享变量, TMA Producer A warp、TMA Producer B warp、MMA Issue warp 必须保持同步推进, 他们三者遍历 block 的顺序完全一致(都通过同一个 `scheduler.for_each_block`), 且每个 block 内循环 `num_k_blocks` 次, 配对关系严格成立.

然后生产者(Producer A/B warp)和消费者(MMA)之间的同步通过`empty_barriers[i]`和`full_barriers[i]`进行. 这些barrier数组中每个元素分别代表不同的stage.

首先需要等待消费者释放

```
empty_barriers[stage_idx]->wait(phase ^ 1);
```

`empty_barriers[i]` 在初始化时 `arrival_count = 1`当 MMA issue warp 发射完对应 stage 的 UMMA(第 821 行)后通过 `umma_arrive_multicast_2x1SM` arrive 这个 barrier. 而`phase ^ 1`表示每个 stage 在 `phase=0`、`phase=1` 间来回翻转, 因此 wait 的相位要与上一次 arrive 的相位相反.

例如在`kNumStages = 3`时, 如果`num_k_blocks = 6`,stage和phase变化如下所示:

![图片](assets/c104567eaab4.png)

然后TMA的坐标计算如下:

```
//token pool 是按 `BLOCK_M` 对齐、紧密拼接的；`m_idx` 直接对应 tensor map 的外维(M 轴)偏移. uint32_t m_idx = pool_block_idx * BLOCK_M;//K 轴每次推进 `BLOCK_K = 128` 元素. uint32_t k_idx = k_block_idx * BLOCK_K;//SFA 在 pool 中按 `SF_BLOCK_M` 对齐, 与 UTCCP 4×32 布局匹配uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;//每 `kGranK * 4 = 128` 个 K 元素共用一个 SF, 所以 SFA 的 K 轴步进就是 `k_block_idx`uint32_t sfa_k_idx = k_block_idx;// Add 2 CTA offsets for non-leader CTAif (not is_leader_cta)    m_idx += scheduler.template get_valid_m<true>() / 2;
```

2-CTA Multicast 的 M 偏移
这里有一个细节, kernel 使用 2-CTA cluster GEMM, 这是一个Blackwell上的新特性, 如下图所示:

![图片](assets/bfe7580e1c89.png)

leader CTA 和 non-leader CTA各加载 `LOAD_BLOCK_M = BLOCK_M / 2` 行的 activation.

然后issue TMA, 如下代码所示, 这里它将通过`full_barriers[stage_idx]`通知MMA Warp已经加载完成.

```
if (cute::elect_one_sync()) {  tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(      tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],      k_idx, m_idx, 2);  tma::copy<SF_BLOCK_M, 1, 0>(      tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],      sfa_m_idx, sfa_k_idx, 2);  if (is_leader_cta) {    full_barriers[stage_idx]->arrive_and_expect_tx(        SMEM_A_SIZE_PER_STAGE * 2 + SF_BLOCK_M * sizeof(uint32_t) * 2);  } else {    full_barriers[stage_idx]->arrive(0u);  }
```

### 4.3 TMA Producer B Warp

TMA Producer B warp 是 **MegaMoE kernel 的 GEMM 权重生产者**, 与 TMA Producer A warp 构成一对对称的双生产者, 它和 TMA Producer A Warp的区别如下:

![图片](assets/5634a0bf91bb.png)

同样它以调度器`scheduler.for_each_block`驱动

```
// 第 735–738 行scheduler.for_each_block([&](const sched::BlockPhase& block_phase,                             const uint32_t& local_expert_idx,                             const uint32_t& num_k_blocks,                             const uint32_t& m_block_idx, const uint32_t& n_block_idx) {   /*--- TMA Producer B Warp 处理逻辑---*/                             }
```

两个 TMA warp 都调用 **同一个调度器** 的 `for_each_block`, 参数签名完全一致. 由于 scheduler 内部状态(`stage_idx`、`phase`、`block_idx` 等)都是基于lambda 外的共享变量(`uint32_t stage_idx = 0, phase = 0;`), 且 `advance_pipeline` 也是共享 lambda, 两个 warp 遍历 block 的顺序与节拍完全同步.并且由于分支分隔、scheduler 等幂的迭代顺序、以及 `stage_idx/phase` 共享、`empty_barriers` 消费者同步, 两个 warp 在每个 `(block, k_block)` 迭代上天然对齐.

同样TMA Producer B由于L1/L2权重参数的shape不同需要针对L1和L2阶段使用不同的Tensor_map

![图片](assets/b05d0198364e.png)

```
const auto tensor_map_b_ptr =    block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;const auto tensor_map_sfb_ptr =    block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;//在 TMA Producer B中增加了 `shape_n` 本地变量, 因为 B warp 要用 `shape_n` 计算 expert 批次的 N 偏移const auto shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;// shape_sfb_k = ceil_div(shape_k, 128) // 表示 SFB 在 K 轴上的 scale 行数(每 128 个 K 元素共用一个 FP8 E8M0 scale)// 稍后用于计算 `sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx`, 定位当前 expert 的 SFB tile. const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);
```

和 Producer A的另一个区别是, 它无需等待直接进入下面的 K 循环. 这是因为权重是在kernel launch 前就已常驻 GMEM 的静态张量, 没有 producer-consumer 关系

```
// 直接进入 K 循环, 无任何预先等待for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
```

它和 Producer A Warp一样, 共用`empty_barrier[i]`和`full_barrier[i]`与 MMA Warp 进行多 stage 的交互, 开始阶段等待消费者释放

```
// Wait consumer releaseempty_barriers[stage_idx]->wait(phase ^ 1);
```

A warp 和 B warp **都 wait 同一个 barrier**这是安全的, `wait` 是只读操作, 两个 warp 会同时看到 phase 翻转, 然后各自重新填充该 stage 的 smem.
B/SFB 的 TMA 坐标计算
这是 B warp 与 A warp 差异最显著的一段.

```
// Compute weight offsetuint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;uint32_t k_idx = k_block_idx * BLOCK_K;uint32_t sfb_n_idx = n_block_idx * BLOCK_N;uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx;
```

首先对于`n_idx`的计算,`local_expert_idx * shape_n`: 跳到当前 expert 的权重切片起点(`shape_n` 是该 expert 权重在 N 方向的总行数)；`+ n_block_idx * BLOCK_N`: 再跳到该 expert 内的第 `n_block_idx` 个 N tile；这就是 TMA 描述符 `tensor_map_l{1,2}_weights` 的外维(N 轴)偏移.  而`k_idx`和Producer A Warp完全相同, 每次 K 推进 `BLOCK_K = 128`.

`sfb_n_idx`**不** 包含 `local_expert_idx * shape_n`, 因为 SFB 张量的布局是 `[kNumExperts, shape_sfb_k, shape_n]`, expert 批次轴是 **K 轴侧**, 所以 SFB 的 N 偏移只包含 `n_block_idx * BLOCK_N`, 而 expert 偏移在 K 方向计算. 计算`sfb_k_idx`时, `local_expert_idx * shape_sfb_k`: 当前 expert 的 SFB 切片起点(在 `shape_sfb_k` 行内部的每一行都属于同一个 expert)；`+ k_block_idx`: 选中该 expert 的第 `k_block_idx` 个 SF 行；

关于SF的转置问题和A对比如下:

![图片](assets/f0de9412a093.png)

Producer A 和 Producer B的对比如下:

![图片](assets/7d712febacc1.png)

然后直接issue TMA, 此时采用multicast在两个CTA同时加载

```
// TMA copy weights with SFif (cute::elect_one_sync()) {    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(        tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 2);    tma::copy<BLOCK_N, 1, 0>(        tensor_map_sfb_ptr, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);    if (is_leader_cta) {        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t) * 2);    } else {        full_barriers[stage_idx]->arrive(0u);    }}__syncwarp();
```

最终也是通过`full_barriers[stage_idx]`通知MMA Warp数据已经完成加载.

### 4.4 MMA Warp

MMA Issue warp 是整个 kernel 的 **计算心脏**: 它从 smem 消费 A/B 数据与 SF, 把 SF 通过 UTCCP 搬到 TMEM, 发射 SM100 **2-CTA UMMA block-scale FP8×FP4** 指令, 让两个 CTA 共同完成一个 GEMM tile 的计算, 累加结果直接留在 TMEM 中供 epilogue warp 读取.

![图片](assets/aeb49cbb0a84.png)

这里注意一下, DeepSeek-V4的 MoE FFN 采用了 w4a8 的方式, 即`weight`为FP4 , 而`activation`为FP8. `weight`采用FP4有两个好处, 第一是模型中大量参数为专家权重, 采用FP4后可以在B系列的8卡服务器上进行单机推理. 另一方面是针对Decoding阶段Memory-Bound的特点, FP4降低了内存带宽压力, 提升了性能.

#### 4.4.1 2-CTA UMMA

SM100 的 **2-CTA UMMA 指令**(`SM100_MMA_MXF8F6F4_2x1SM_SS`)的语义:

**指令在 cluster 范围内协作**: `tcgen05.mma.cta_group::2` 指令由 leader CTA 发起, 硬件自动协调 non-leader CTA 的数据；

**累加器分布**: TMEM 累加结果的一半存在 leader CTA 的 TMEM, 一半存在 non-leader CTA 的 TMEM(对应 M 方向的上/下半区)；

**单点发射**: 如果两个 CTA 都发射, 会导致指令重复执行, 结果错误；

**对称的 warp 占位**: non-leader CTA 上 `warp_idx == kNumDispatchWarps + 2` 的 warp 虽然走进 `else if` 分支, 但直接跳过 `if (is_leader_cta)` 后的所有代码——它只保留了 `warpgroup_reg_dealloc` 的 register 配额调整, 同组 warpgroup 的线程数仍为 128.

![图片](assets/0a800bfd6426.png)

#### 4.4.2 AB Swap

这是一个在TRT-LLM `PR4430` 和 DeepGEMM `PR192`的个优化, 在 MMA 中使用了AB Swap, 即 `activation` 作为 MMA 的B矩阵, 而 `weight` 作为A矩阵.  我们注意到Blackwell(SM100)的MMA指令, 在M维度是固定的128, 2-CTA合并的情况下为256. 考虑MoE中的`activation`和`weight`

![图片](assets/a7eaf95e3635.png)

因此将通常GEMM所用的 A(`activation`)和B(`weight`)两个操作数互换, 即AB Swap的方法, 让`weight`去对齐 M 的 128/256 维度约束, 而 `activation`作为 B 操作数, 然后根据指令去约束`BLOCK_M`对齐. 这种优化特别是对Decoding阶段的小batch-size是非常有效的, 此时BLOCK_M 通常较小.
![图片](assets/e38b229cbb04.png)

另一方面是2-CTA的设计, Blackwell的2-CTA MMA设计如下左图所示, 硬件约定是：A 操作数两 CTA 各持一半 M、B 操作数两 CTA 各持完整 N

![图片](assets/47ecc112b558.png)

通过AB Swap后, 在TMA Producer B中, 两个CTA同时加载相同的`weight`, 并自然的利用了TMA Multicast的能力

```
tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(    tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx],    k_idx, n_idx, 2);                // ★ num_tma_multicast = 2tma::copy<BLOCK_N, 1, 0>(    tensor_map_sfb_ptr, full_barriers[stage_idx], smem_sfb[stage_idx],    sfb_n_idx, sfb_k_idx, 2);        // ★ SFB 也是 num_tma_multicast = 2
```

如下图所示:

![图片](assets/d74d462fd123.png)

而`activation`在两 CTA 在 M 方向各算半区, 需要不同的 M 行, 所以用"同一条 2-CTA TMA 但不同坐标"的方式分开读, 即每个SM读`BLOCK_M/2`.

#### 4.4.4 MMA Config

正如前文所述, 这里对`activation`(a_dtype_t=FP8)和`weight`(b_dtype_t = FP4)进行了定义, 然后在MMA configs中定义了一些约束

```
 // Data types  // NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)  // 数据类型：激活 FP8(e4m3)；权重 FP4(e2m1)，smem 中以 8bit 形式解包  using a_dtype_t = cutlass::float_e4m3_t;  using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;  // MMA configs  // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major  // MMA 配置：此处固定交换 A/B（A 为权重维，B 为激活维）；2-CTA UMMA；K-major  //   LAYOUT_AD_M=128       : 单 CTA TMEM 布局高度  //   UMMA_M = 256          : 2-CTA multicast 后有效 MMA 高度  //   UMMA_N = BLOCK_M      : 交换 A/B 后的 N 维，与专家 token 分块 M 相等  //   UMMA_K = 32           : 单次 MMA 的 K 宽  //   LOAD_BLOCK_M = BLOCK_M/2 : A 上 multicast 后单 CTA 只读到一半  constexpr uint32_t LAYOUT_AD_M = 128;  constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;  constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB  constexpr uint32_t UMMA_K = 32;  constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A  constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;  // BLOCK_M 必须要能够整除16  DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");  // 由于AB Swap BLOCK_N 必须要等于 LAYOUT_AD_M  DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");  DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
```

#### 4.4.5 UMMA指令构造

SM100 UMMA 的指令发射依赖三类描述符:

**指令描述符**(shape/dtype/swizzle)

**smem 描述符**(A、B 的内存布局)

**SF 描述符**(UTCCP 搬运 scale factor 用)

指令描述符`instr_desc`构造如下:

```
auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<    b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,    UMMA_M, UMMA_N,    cute::UMMA::Major::K, cute::UMMA::Major::K>();
```

注意这里为了AB Swap, 改变了`a_dtype_t`和`b_dtype_t`的顺序, 因此模板参数展开:

![图片](assets/e362dcb8fbef.png)

`instr_desc` 是一个 `cute::UMMA::InstrDescriptorBlockScaled`(带 SF 支持的变体), 存储 kind/block-scaled flag、shape、dtype、swizzle 等 **静态** 字段. 它会在循环中 **动态** 更新 `n_dim_` 和 `a_sf_id_/b_sf_id_`.

SF描述符`sf_desc`构造如下:

```
auto sf_desc = mma::sm100::make_sf_desc(nullptr);//引用自`mma/sm100.cuh`cute::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr) {    // NOTES: the UTCCP layout is K-major by default    // Atom size: 8 x 128 bits    // {SBO, LBO} means the byte stride between atoms on {MN, K}    // Since the UTCCP we used is 128b-wide (only 1 atom on K), so LBO can be zero    return make_smem_desc(cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);}
```

`sf_desc` 是为 **UTCCP(tensor memory copy)** 指令准备的 smem 描述符, `smem_ptr` 传入 `nullptr`, 稍后在每次发射 UTCCP 前通过 `replace_smem_desc_addr` 更新实际地址. 关于SBO和LBO解释如下:

SBO (stride byte outer) = `8 * 16 = 128`: atom 之间在 MN 方向跨 128 字节；

LBO (stride byte inner) = 0: UTCCP 一次搬一个 128-bit atom, K 方向无跨步；

SMEM描述符:

```
DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
```

其中`DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages")`这个检查的原因是下面要用 lane_idx 作为 stage 索引参与预计算, 32 lane 最多覆盖 32 stage.

`make_umma_desc` 构造一个 **cutlass UMMA smem 描述符**, 包含 start address、layout type、stride 等. 基址先填 `smem_a[0]` / `smem_b[0]`(第 0 个 stage), 后面用 lane-level 偏移生成各 stage 的版本.

模板参数: `K-major` + `LOAD_BLOCK_{M,N}` + `BLOCK_K` + `swizzle mode` 必须与 TMA load 的 box 参数严格匹配, 否则 UMMA 读到错位数据.

#### 4.4.5 Per-lane描述符

接下来有一个优化, `a_desc` / `b_desc` 是 `cute::UMMA::SmemDescriptor`(64 位), 拆成两半:

![图片](assets/08a86b3de3be.png)

不同 stage 的 smem 基址相差 `SMEM_{A,B}_SIZE_PER_STAGE` 字节, 所以它们的 `desc.lo` 相差常量 `SMEM_{A,B}_SIZE_PER_STAGE / 16`(因为地址字段已经 `>>4` 编码). 下面这两行代码的目的是把 `kNumStages` 个 stage 的 desc low half 预计算在 **32 个 lane 的寄存器中**, 每个 lane 存一个 stage 的 lo 值, 后续循环中用 `ptx::exchange` 从对应 lane shuffle 过来, 避免循环内重复计算.  这也是为什么`kNumStages <= 32 `的原因.

```
uint32_t a_desc_lo = lane_idx < kNumStages    ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16    : 0u;uint32_t b_desc_lo = lane_idx < kNumStages    ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16    : 0u;
```

如果不采用这种方式, 循环内需要进行 `a_desc.lo = base + stage_idx * stride` 的乘加. 而通过这种做法在 K 循环的过程中仅需要一条指令即可

```
const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
```

`ptx::exchange(reg, src_lane)` 对应 PTX `shfl.idx.b32`——从 lane `stage_idx` 读取其 `a_desc_lo` 寄存器的值

对比两种实现:

![图片](assets/47acea400094.png)

K 循环是 **kernel 最热的路径**(每个 block 跑 `num_k_blocks × BLOCK_K/UMMA_K` 条 UMMA), 省下的每一条指令都会累积, 而且 `shfl` 不占用整数 ALU, 能更好地与 `tcgen05` 发射、`barrier wait` 等其他指令并行.

#### 4.4.6 MMA指令 shape 静态检查

```
DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or                 (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or                 (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),                 "Invalid MMA instruction shape");
```

这是 SM100 `tcgen05.mma.cta_group::2.kind::mxf8f6f4` 指令硬件约束, 本 kernel 中 `UMMA_M = 256`(`LAYOUT_AD_M * 2`), `UMMA_N = BLOCK_M`, 满足第三行约束.

| UMMA_M | UMMA_N 约束 |
|---|---|
| 64 | 8 的倍数, [8, 256] |
| 128 | 16 的倍数, [16, 256] |
| 256 | 16 的倍数, [16, 256] |

#### 4.4.7 Persistent Block 迭代

这里依旧是用了`scheduler.for_each_block`和TMA Producer A/B一致.

```
// Persistently schedule over blocksuint32_t current_iter_idx = 0;scheduler.for_each_block([&](const sched::BlockPhase& block_phase,                             const uint32_t& local_expert_idx,                             const uint32_t& num_k_blocks,                             const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
```

这些参数仅是用了 `num_k_blocks`驱动内层 K 循环.

![图片](assets/f9e72e3712e3.png)

但是增加了一个`current_iter_idx`的全局 GEMM 迭代计数, 随着block递增, 并且不重置. 它有两个用途, 用于映射到 **epilogue stage 索引**`accum_stage_idx = current_iter_idx % kNumEpilogueStages` 以及 accumulator 的 **phase bit**`accum_phase = (current_iter_idx / kNumEpilogueStages) & 1`.

#### 4.4.7 动态更新 UMMA_N 值

由于 AB swap, UMMA_N 对应 activation 的 M 维. 对于 expert 的最后一个 tile, 可能 token 数不足 `BLOCK_M`.

![图片](assets/d20c3a82e4f2.png)

此时让 UMMA 只计算 `valid_m` 行, 剩余行会被 TMA 的 OOB 填零忽略.

```
// Dynamic update of UMMA N based on effective Mmma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());// ref `mma/sm100.cuh`  void update_instr_desc_with_umma_n(cute::UMMA::InstrDescriptorBlockScaled& desc, const uint32_t& umma_n) {      desc.n_dim_ = umma_n >> 3;  // UMMA_N 以 8 为单位编码  }
```

#### 4.4.8 TMEM 双缓冲

在执行 MMA 之前, 有一个TMEM 空 barrier 等待, 这里`kNumEpilogueStages = 2`表示 TMEM 上有 **两套 accumulator 槽**, 在两个 block 之间切换, `accum_stage_idx` 在 `0/1` 间循环, `accum_phase` 每完成 `kNumEpilogueStages` 次迭代翻转一次.

![图片](assets/eec55d83f322.png)

```
// Wait tensor memory empty barrier arrivalconst auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);//此barrier之前的所有普通线程同步操作必须对当前线程可见, 然后才允许发射 tcgen05 指令(UTCCP、UMMA)ptx::tcgen05_after_thread_sync();
```

对于`tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads)`初始化的到达计数为**2 × 整个 epilogue warpgroup 的线程数**, 双 CTA 的 epilogue 都要 arrive(CTA×2), 每个 CTA 内所有 epilogue 线程都 arrive. Epilogue warp 完成从 TMEM 读出累加结果后会集体 arrive 这个 barrier, 释放 accumulator 槽, 同时 MMA warp 必须 **等所有 epilogue 线程都读完才能覆写**.

#### 4.4.9 Empty Barrier Arrive

这一步构成了一个双向的同步机制, `empty_barrier[i]`用于通知TMA Producer A/B warp, 而`tmem_full_barriers[k]`用于通知 Epilogue Warp

![图片](assets/fa8ba8e153d8.png)

```
// Empty barrier arrivalauto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {    auto umma_arrive = [](const uint64_t* barrier) {        constexpr uint16_t kCTAMask = (1 << 2) - 1;        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);    };    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting    if (do_tmem_full_arrive)        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));    __syncwarp();};
```

`umma_arrive_multicast_2x1SM` 实际上是CUTLASS封装的 PTX 指令`tcgen05.commit.cta_group::2.mbarrier::arrive::one.multicast::cluster.shared::cluster`. 作用是 MMA 完成后, 自动对 **cluster 内所有 CTA 的同一barrier** 做 arrive, 这是 2-CTA UMMA 与 smem barrier 协同的官方方法. `kCTAMask = (1 << 2) - 1 = 0b11`: bitmap 标识cluster 内 CTA 0 和 CTA 1 都要收到 arrive.

另一个细节的问题是为什么要`multicast`释放 `empty_barriers`? `empty_barriers[stage_idx]` 位于各 CTA 的 smem, TMA Load A/B warp 在 **各自的 CTA** 上 wait, leader CTA 的 MMA warp 需要 **同时** 通知两个 CTA 的 TMA warps “smem stage 已空, 可以覆写”. 用 2x1SM multicast 原语一次 arrive 两个 CTA 的 barrier, 比两次普通 arrive 更高效.

然后它还会等待`tmem_full_barriers[i]->init(1)`, 它的到达计数 = 1, **只需 leader CTA 的 MMA warp 一次 arrive**.

`do_tmem_full_arrive` 参数在最后一个 K tile 才为 true(`k_block_idx == num_k_blocks - 1`), 也就是说只有一整个 block 的 GEMM 完成后 才通知 epilogue accumulator 已就绪. 注意虽然此 barrier 不需要 multicast, 但为了代码复用仍用 `umma_arrive` 路径(`kCTAMask` 冗余但无害).

#### 4.4.10 K 循环

主循环如下所示, 它是用了参数传入的`num_k_blocks`作为循环控制.

```
// Launch MMAs#pragma unroll 2for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
```

注意 `for` 循环的 `advance_pipeline(k_block_idx)` 调用同步更新 `k_block_idx`、`stage_idx`、`phase`, 下次迭代会看到新的 `stage_idx` 指向下一个 smem stage.

循环内部, 首先它需要等待TMA Producer A/B完成

```
// Wait TMA load completionfull_barriers[stage_idx]->wait(phase);ptx::tcgen05_after_thread_sync();
```

到达计数 4(2 CTA × 2 warps)+ 所有 expect_tx 字节到齐后, barrier 翻转 phase. 注意这里 wait 的 phase 是 `phase`(而非 TMA warp 的 `phase ^ 1`), 即 MMA warp 与 TMA warp 对 `phase` 的使用方式相反, 因为一个是 consumer、一个是 producer.

然后是提取 4.4.5节预计算好的描述符 `.lo` 字段, 本行从 lane `stage_idx` 取值, 得到 stage `stage_idx` 对应的 desc .lo 基地址

```
const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
```

然后发射`UTCCP`指令和`UMMA`指令, 在blackwell中这些指令只需要一个线程发射, 因此

```
if (cute::elect_one_sync()) {
```
UTCCP
UTCCP 是 SM100 上的 TMEM 加载指令(实际名称 `tcgen05.cp`), 专门把 smem 的 SF 数据高效搬到 TMEM. MXFP8/FP4 UMMA 指令要求 SF 必须驻留在 TMEM. 就此整个TMEM布局如下:
![图片](assets/06dd5186665c.png)

```
using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;#pragma unrollfor (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {    auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);}#pragma unrollfor (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i) {  auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;  mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);  cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);}
```

我们首先来看循环切分, - `kNumUTCCPAlignedElems = 128`, 每个 UTCCP instruction 覆盖 128 个 M 元素；

`SF_BLOCK_M / 128` 次循环: 把整个 `SF_BLOCK_M` 个 M 元素的 SF 分批搬到 TMEM；

每次搬到 TMEM 列 `kTmemStartColOfSFA + i * 4`, 每个 UTCCP 实例占用 4 个 TMEM 列(对应 `128 / 32 = 4`).

`SM100_UTCCP_4x32dp128bit_2cta` 指令表示:

`4x32`: 4 行 × 32 列(4 个 UMMA_K group × 32 个 M 元素)

`dp128bit`: 每次 data path 128 bit

`2cta`: 2-CTA 协作(与 UMMA 2-CTA 一致)

然后`replace_smem_desc_addr`只替换 smem 描述符的 **起始地址字段**, 保留其它字段(layout、SBO/LBO 等), 相比重新构造整个描述符快得多.

```
void replace_smem_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr) {    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);}
```
UMMA
外层 K 循环每次推进 `BLOCK_K = 128` 元素, 发一次 TMA + UTCCP, 内层 K 循环每次推进 `UMMA_K = 32` 元素, 发一条 UMMA 指令. 这样一个 TMA tile 对应 `BLOCK_K / UMMA_K = 128 / 32 = 4` 条 UMMA, 充分利用 TMEM 中的数据.

![图片](assets/a2f057754bdd.png)

```
// Issue UMMA#pragma unrollfor (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {    const auto runtime_instr_desc =        mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);    a_desc.lo = mma::sm100::advance_umma_desc_lo<        cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);    b_desc.lo = mma::sm100::advance_umma_desc_lo<        cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(        b_desc, a_desc, accum_stage_idx * UMMA_N,        k_block_idx > 0 or k > 0, runtime_instr_desc,        kTmemStartColOfSFB, kTmemStartColOfSFA);}
```

在循环中采用runtime指令描述符, 用于修改当前 UMMA 调用设置 SF 索引: `a_sf_id_ = b_sf_id_ = k`, 表示用 TMEM 中的第 k 个 SF atom. 正好每 UMMA_K = 32 个 K 元素对应 1 个 SF 槽.

```
const auto runtime_instr_desc =    mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);// ref mma/sm100.cuhuint64_t make_runtime_instr_desc_with_sf_id(    cute::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;}
```

然后推进 desc.lo: K 方向

```
a_desc.lo = mma::sm100::advance_umma_desc_lo<    cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);b_desc.lo = mma::sm100::advance_umma_desc_lo<    cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);// ref mma/sm100.cuh// - `base` = stage 基址, 从per-lane预计算的值拿到// - `offset = 0`: MN 方向无偏移(单条 UMMA 覆盖完整 LOAD*BLOCK*{M,N})// - `k_idx = k * UMMA_K`: K 方向推进 `k * 32` 元素uint32_t advance_umma_desc_lo(const uint32_t& base, const uint32_t& offset, const uint32_t& k_idx) {    return base + (((offset + k_idx * get_umma_desc_stride_k<...>()) * sizeof(dtype_t)) >> 4u);}
```

然后 PTX 指令发射, 注意AB Swap使得`b_desc`和 `a_desc`以及`kTmemStartColOfSFB`和`kTmemStartColOfSFA`进行了交换.

```
ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(    b_desc, a_desc, accum_stage_idx * UMMA_N,    k_block_idx > 0 or k > 0, runtime_instr_desc,    kTmemStartColOfSFB, kTmemStartColOfSFA);
```

对应 PTX: `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%tmem_c], %desc_a, %desc_b, %desc_hi, [%tmem_sfa], [%tmem_sfb], p`(其中 `p = (scale_c != 0)`).

整个参数解析如下:

![图片](assets/d68a9f326882.png)

第一个 UMMA(`k_block_idx == 0 && k == 0`): `scale_c = false`, tcgen05 指令用 `scale::0`, 表示 **覆写** 累加器(初始化), 后续所有 UMMA: `scale_c = true`, 用 `scale::1`, 表示累加, 这比先把 accumulator 清零再累加省一次遍历, 第一条 UMMA 自动完成清零.

最后warp同步并在每 K tile 结束都调用一次用于释放 `empty_barriers[stage_idx]`, 通知 TMA warp 可以覆写 smem stage通知 TMA Producer A/B 可以覆写SMEM

```
}__syncwarp();// Commit to the mbarrier object// No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
```

最后是退出前的尾部barrier, 设置`tmem_empty_barriers`等待Epilogue完成另一个TMEM缓冲区的计算再执行下一次MMA.

```
// To safely deconstruct barriers, we need another round of waitsif (current_iter_idx > 0) {    const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;    tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages]->wait(accum_phase_idx);}
```

#### 4.4.11 双流水线机制

最后再来强调一下它是如何与 Producer A/B Warp 和 Epilogue Warp进行流水线交互的. MMA warp 同时管理 **两套相互正交** 的流水线:
![图片](assets/f0b475ec5090.png)

GEMM smem 流水线(`stage_idx`, `phase`)
深度: `kNumStages`(2/3/4, JIT 可调)；

控制变量: `stage_idx`, `phase`(与 TMA warp 共享命名, 通过 `advance_pipeline` 同步推进)；

Barrier:

`full_barriers[stage_idx]`(producer ← TMA warps, consumer ← MMA warp)；

`empty_barriers[stage_idx]`(producer ← MMA warp, consumer ← TMA warps)；

生命周期: 每个 K tile(`BLOCK_K = 128` 元素)；

目的: 在 TMA load 与 UMMA 计算之间做 `kNumStages` 级重叠.
TMEM accumulator 流水线(`accum_stage_idx`, `accum_phase`)
深度: `kNumEpilogueStages = 2`(第 185 行)；

控制变量: `accum_stage_idx`, `accum_phase`(MMA 与 epilogue 共享)；

Barrier:

`tmem_full_barriers[accum_stage_idx]`(producer ← MMA warp, consumer ← epilogue)

`tmem_empty_barriers[accum_stage_idx]`(producer ← epilogue, consumer ← MMA warp)；

生命周期: 每个 GEMM block(`num_k_blocks` 个 K tile)；

目的: 让 MMA 计算 block N+1 时, epilogue 可以并行处理 block N 的 accumulator(SwiGLU/量化/TMA store).

### 4.5 Epilogue Warp

它也是一个处理非常复杂的Warp, 覆盖三个阶段: `L1 Epilogue`, `L2 Epilogue` 和 `Combine`

![图片](assets/da48e6266694.png)

![图片](assets/5805480b65c2.png)

#### 4.5.1 初始化阶段

一些关键常量如下, 它表示 2 个 epilogue warpgroup 各自负责 `BLOCK_M / 2` 行, 4 个 warp/wg 在 BN 方向再切 4 份,  M 方向进一步切为 `STORE_BLOCK_M`, 再切为 `ATOM_M = 8`.

```
constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;   // 每个 wg 负责的 M 行数constexpr uint32_t ATOM_M = 8;                                       // 最小存储粒度的 M 行数constexpr uint32_t kNumBankGroupBytes = 16u;                         // swizzle 粒度 = 16 字节constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;       // 一个 store block 包含几个 atom
```

切分示意图如下:

![图片](assets/ab3142fbe204.png)

因此 Warp 内存在多级 ID 分解

```
const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);  // 0..kNumEpilogueWarps-1const auto epilogue_wg_idx = epilogue_warp_idx / 4;                                        // warpgroup 编号const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;                        // warpgroup 内线程号const auto warp_idx_in_wg = epilogue_warp_idx % 4;                                         // warp 在 wg 内的编号
```

#### 4.5.2 和Dispatch Warp同步

在启动前和Dispatch有一个同步, 使用`unaligned`的原因是dispatch(通常 1 个 warpgroup,128 线程)与 epilogue(2 个 warpgroup,256 线程)线程数不同 且 warp 边界不对齐,只能使用 `bar.sync` 的 `unaligned` 变体

```
ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
```

这里稍微展开一下 `Dispatch` 和 `Epilogue` 的整个同步机制. 它们之间的同步机制比较复杂,如下图所示

![图片](assets/b96d28ee7bd3.png)

**第一次同步**: 主要是解决SMEM所有权切换. dispatch 的 `smem_send_buffers`(pull TMA 1D 落地区)与 epilogue combine 阶段的 `combine_load_buffer` / `combine_store_buffer` 复用同一段 smem.  在Dispatch warp pull阶段会使用 smem. 两者同步后在 Dispatch pull阶段和 GEMM 每个block 的 Epilogue阶段(使用`smem_cd`)对于 smem 使用互不干扰.

![图片](assets/01e1b8561640.jpg)

**第二次同步**: 主要是解决workspace的所有权切换. Epilogue Warp的combine阶段需要**复用 smem**(`smem_buffer` 到 `barrier_start_ptr` 区间)做 combine chunk 缓冲. 此时 Dispatch Warp 需要开始清理 Workspace中的字段, 它需要等最后的Combine 阶段开始后才能进行清理工作, 并且和 Combine的计算overlap.

#### 4.5.3 Block循环

接下来Epilogue Warp同样会进入`scheduler.for_each_block`循环, 在这个循环内会有两个分支来处理 L1 阶段或者 L2 阶段的 Epilogue计算.

```
uint32_t current_iter_idx = 0;scheduler.for_each_block([&](const sched::BlockPhase& block_phase,                             const uint32_t& local_expert_idx,                             const uint32_t& num_k_blocks,                             const uint32_t& m_block_idx,                             const uint32_t& n_block_idx) {    // Wait UMMA arrival    const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;    const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;    tmem_full_barriers[accum_stage_idx]->wait(accum_phase);    // 保证普通线程同步结果对后续 tcgen05 指令(`SM100_TMEM_LOAD_*`)可见.    ptx::tcgen05_after_thread_sync();        // Offset计算    const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);    const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;    uint32_t m_idx = pool_block_idx * BLOCK_M;    uint32_t n_idx = n_block_idx * BLOCK_N;            ....        if (block_phase == sched::BlockPhase::Linear1) {        /*----  L1 Epilogue Phase ---*/    } else {        /*----  L2 Epilogue Phase ---*/    }}/*--- Combine ------*/
```

在循环中, 首先等待 L1/L2 MMA Warp 执行完MMA, 并标记`tmem_full_barriers[accum_stage_idx]` arrive. 其中 `tmem_full_barriers[i]->init(1)`, 只需 leader CTA 的 MMA warp arrive 一次. 然后所有 CTA 的 epilogue warp 都 wait 这同一个 barrier: 因为 2-CTA UMMA 在两个 CTA 的 TMEM 上都写了结果；leader arrive 一次后, 硬件保证两侧 TMEM 都已就绪.

`valid_m`采用`ptx::exchange(..., 0)`: 从 lane 0 读取值广播到全 warp——显式告诉编译器"这个值全 warp 一致, 不会产生 divergence", 然后根据`pool_block_offset`和scheduler输入的 `m_block_idx`/`n_block_idx`计算出实际的这个 block 在全局坐标系的起点.

#### 4.5.4 L1 Epilogue

L1 = 第一阶段 GEMM(gate+up projection)的结果后处理. 主要包含:

加载 top-k 权重, 用于在 SwiGLU中加权

从 TMEM 中读 Accumulator 的结果

计算SwiGLU并乘以 topk权重  (silu(gate) * up) * weight

per-lane  amax, 然后 warp reduce + cross-warp reduce

量化为 FP8 E4M3, 按 UE8M0 方案存储 SF

通过 TMA store 写到 `tensor_map_l1_output`(即 `l2_token_buffer` 的 GMEM 视图)

通过 `red_or_rel_gpu(l2_arrival_mask)` 原子位运算通告"此 N 子块已就绪". 供TMA-Producer-A进行L2的block加载.
4.5.4.1 任务如何切片的
由于 `Epilogue Warp` 有 8 个 warp, 对于L1 MMA产生的`BLOCK_M × BLOCK_N`的块需要在这 8 个 Warp进行切分, 切分方式我们在4.5.1节已经介绍. 这里详细展开一下, 首先 4 个 Warp 会构成一个 `WarpGroup`, 因此整个 Epilogue Warp 通常有 2 个 WarpGroup(WG), 它们在`M`维度拆分, 每个 WG 负责`WG_BLOCK_M = BLOCK_M / 2` 的数据. 然后一个 WG 内的 4 个 Warp 在 N 方向平铺, 一个 Warp 负责 `WG_BLOCK_M × (BLOCK_N / 4)`的数据, 如下所示:

![图片](assets/e3b1f4704eee.png)

然后一个 Warp 内在 M 维度继续划分成 `WG_BLOCK_M / STOCK_BLOCK_M` 个 `STOCK_BLOCK_M  × (BLOCK_N / 4)` 的块, `STOCK_BLOCK_M`由启发式调度器根据`BLOCK_M`的size决定.

![图片](assets/a4d44fdb1324.png)
外层循环: 基于 Store Tile 迭代
代码中首先是一个外层基于`STORE_BLOCK_M`的循环, 循环中`s` 是 `Store Tile` 索引.

```
float stored_cached_weight = 0;#pragma unrollfor (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {    // 判断如果 store tile 的起始行如果超过 valid_m, 直接退出.    // 这里epilogue_wg_idx是 WG的编号    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {        ptx::tcgen05_before_thread_sync();        tmem_empty_barriers[accum_stage_idx]->arrive(0u);        break;    }
```

这里有一个优化, 对于 M 轴方向超过 valid_m 的直接跳过. 但是仍需通知 `tmem_empty_barriers` 释放 TMEM accumulator 槽, 否则 MMA warp 会永久 stall. 注意`tmem_empty_barriers[accum_stage_idx]->init(2 * kNumEpilogueThreads)`: 到达计数 = 2 × 全部 epilogue 线程; 每个 CTA 的每个 epilogue 线程都要 arrive 一次, 所以这里用 `arrive(0u)`(单 lane 贡献 1).
内部循环:  基于 ATOM_M 迭代
在 Store Tile 内部继续按照 M 维度拆分成`kNumAtomsPerStore =  STORE_BLOCK_M / ATOM_M`个 M 维度为`ATOM_M = 8`的 ATOM Tile, 大小为`ATOM_M  × (BLOCK_N / 4)`如下图所示:

![图片](assets/48414d8bc6e8.png)

```
#pragma unroll// 遇到一个 store tile, 按 ATOM_M 展开; 暂存 SwiGLU 结果 + amax// 交给后续量化for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {    const uint32_t j = s * kNumAtomsPerStore + i;   // Load weights from global into register cache per 32 tokens    DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");            /*------ ATOM_M Tile -----*/    }
```

这里还有一个判断: 一个 warp 32 lane, 如果 `j * ATOM_M % 32 == 0`(即每 4 atom), 就把接下来 32 个 M 行的 top-k weight 全部加载到 `stored_cached_weight`(每 lane 1 个), 中间 3 个 atom 直接从寄存器 shuffle 取, 这样每 4 atom 只做 1 次 GMEM 加载.
4.5.4.2 加载 topk_weight 权重
这里把`topk_weight`加权直接吸收到SwiGLU计算的过程中, 它由 Dispatch Warp 提前发送到了接收端`l1_topk_weights_buffer`, 加载代码如下:

```
    //判断每 4 个 ATOM_M 且 WG_BLOCK_M能整除32 , 并且不越界时, 加载到寄存器中.    if ((j * ATOM_M) % 32 == 0 and (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {        stored_cached_weight = *l1_topk_weights_buffer            .get_data_buffer(m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)            .get_base_ptr<float>();    }    // `ptx::exchange(cached, src_lane)`: 从 src_lane 读 weight    const float2 weights = {        ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),        ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)    };
```
4.5.4.3 TMEM 加载
TMEM地址计算中, `accum_stage_idx * UMMA_N`: 选中当前 accumulator 槽(第 0 或第 UMMA_N 列). `epilogue_wg_idx * WG_BLOCK_M`: warpgroup 偏移, `j * ATOM_M`: atom 偏移. 因为 kernel 用 AB-swap, UMMA_N 对应 M 方向, 所以这里的 column offset 代表 **M 方向偏移**. 加载方式如下:

```
// Load from TMEMuint32_t tmem_addr = accum_stage_idx * UMMA_N // 选stage                   + epilogue_wg_idx * WG_BLOCK_M // WarpGroup选半                   + j * ATOM_M; //ATOM_M 偏移                   uint32_t values[ATOM_M];cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,                                       values[0], values[1], values[2], values[3]);cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,                                       values[4], values[5], values[6], values[7]);//TMEM load 是异步的, 需要barrier确保后续普通寄存器运算能看到 load 结果cutlass::arch::fence_view_async_tmem_load();
```

TMEM加载使用`SM100_TMEM_LOAD_16dp256b1x` 指令

`16dp256b1x` = 16 data path × 256 bit × 1 atom；

每次调用加载 4 个 uint32(每线程)到寄存器
为什么要两次TMEM 加载
我们此时来看看它在切分成ATOM后是如何消费 TMEM Accumulator 的. TMEM本身是一个2D的memory寻址架构,每个CTA包含512列和128行, 每个Cell是32bit. 每个Lane 2KB, 地址采用32bits Lane<31:16> Column<15:0>的方式.

![图片](assets/d5b02783c752.png)

由于AB Swap 在 UMMA 中定义TMEM 按照 `[0, UMMA_N)=BLOCK_M`列存放, 而行数占用为`BLOCK_N = 128`行.

在进行 WarpGroup 切分时, 每个 WarpGroup处理 **WG_BLOCK_M = 64** 列, **BLOCK_N** 行

继续在 N 维度切分为 4 个 Warp, 每个 Warp 处理 **WG_BLOCK_M = 64** 列, **BLOCK_N / 4 = 32** 行

继续在 M 维度切分成 **Store Tile** 和 **ATOM Tile**

**ATOM Tile**需要处理 TMEM  **ATOM_M = 8** 列, **BLOCK_N / 4 = 32** 行, 其中结果按照`gate` 和 `up` 在行间交替存储.

![图片](assets/7a5e5de552be.png)

`SM100_TMEM_LOAD_16dp256b1x`处理 `256-bits x 16` 行, 恰好为 FP32 时`ATOM_M = 8` 列, 为了读满 32 行, 每16行的偏移为`0x0010.0000`因此使用 `tmem_addr` 和 `tmem_addr | 0x00100000` 加载两次凑满 32 行

在PTX文档中的`Matrix fragments for shape .16x256b` 如下图所示:

![图片](assets/1987e6f1f4f8.png)

我们以`Thread 0` 为例, 第一次加载 `v[0]`,`v[2]`则构成一对(gate, up). 而`Thread 4`为该 token 在 N 方向的下一对值
4.5.4.4 TMEM 释放
`j` 是 wg 内的全局 atom 索引——当 `j == WG_BLOCK_M/ATOM_M - 1` 时表示本 wg 的 **最后一个 atom**, 此刻 TMEM 已完整读出, 可以通知 MMA warp 覆写这个 accumulator 槽.

```
// Signal tensor memory consumed on the last atomif (j == WG_BLOCK_M / ATOM_M - 1) {    ptx::tcgen05_before_thread_sync();    tmem_empty_barriers[accum_stage_idx]->arrive(0u);}
```
4.5.4.5 SwiGLU激活计算
SwiGLU 公式如下, 同时这里吸收了`topk_weight`的计算

$$SwiGLU(x, W_g, W_u) = silu(x W_g) \cdot (x W_u) \cdot top\_k\_weight$$

其中 `silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)`.

```
// Apply SwiGLU: silu(gate) * up// Gate/up pairs: (0, 2), (1, 3), (4, 6), (5, 7)auto fp32_values = reinterpret_cast<float*>(values);#pragma unrollfor (uint32_t k = 0; k < 2; ++ k) {    auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));    auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));    // Clamp    if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {        bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});        bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});        bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});    }    // SwiGLU    auto gate = __bfloat1622float2(bf16_gate);    auto neg_gate_exp = make_float2(        kFastMath ? __expf(-gate.x) : expf(-gate.x),        kFastMath ? __expf(-gate.y) : expf(-gate.y));    const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);    if constexpr (kFastMath) {        gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});    } else {        gate = {gate.x / denom.x, gate.y / denom.y};    }    const auto up = __bfloat1622float2(bf16_up);    swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);}
```

首先TMEM 读出的是 FP32, 采用 `__float22bfloat162_rn`截断为 BF16 降低后续运算精度需求. 然后有一个**Clamp**计算, 它是可选: 若 `kActivationClamp` 不是 inf, 把 gate 限制在 $[-\infty, +clamp]$, up 限制在 $[-clamp, +clamp]$, 论文中选择的的是 `10`.

然后针对 swiGLU , 指数运算`exp(-gate)`考虑到SFU在CUDA Core中的算力开销, 这里可以选择使用`fastmath`的方式做近似计算, 然后对一组`(gate.x, gate.y)`计算**exp(-gate) + 1 → 分母**. 同样也可以采用 `fast_rcp` 近似计算把除法变乘法.

最后**silu(gate) × up × weight**: 两次 `__fmul2_rn` 完成.  接下来写入`swiglu_values[i * 2 + k]`(`float2`): 每个 atom 产生 2 个 `float2`(累计 4 个 fp32 值).

注意为什么有 `k = 0/1` 的两轮:

`values[0..3]` 是第 0 组 gate/up, `values[4..7]` 是第 1 组

每个 8 元素 = 2 组 (gate.x, gate.y, up.x, up.y) 因此 `k` 循环 2 次
4.5.4.6 Amax 归约 + FP8量化
实质是一个 **三层 max 规约**:寄存器内（4 lane）→ smem 共享数组 → 跨 warp 配对（warp ⊕ 1）.
第一层: warp 内 4-lane 规约
代码如下:

```
// Amax reductionamax_values[i].x = math::warp_reduce<4, true>(    cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),    math::ReduceMax<float>());amax_values[i].y = math::warp_reduce<4, true>(    cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),    math::ReduceMax<float>());
```

这里展开对`math::warp_reduce<4, true>`进行一个解释. 对于一个ATOM Tile内, TMEM加载指令中, 不同的列代表不同的 token. 这样同一个token 在`N`维度的线程跳步为`4`, 如下图: `T0`, `T4`, `T8`, `T12`, `T16`, `T20`, `T24`, `T26`, `T28`为一个 token 在 hidden-dim 维度的值.

![图片](assets/abf31c3e27b4.png)

warp_reduce实现如下, 这里我们同时做一个简单的测试

```
template <typename T> struct ReduceMax { __device__ T operator()(T a, T b) const { return a > b ? a : b; } };template <uint32_t kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>__device__  T warp_reduce(T value, Op op) {    constexpr uint32_t mask = 0xffffffff;    if constexpr (kIntergroupReduce) {        if constexpr (kNumLanesPerGroup <=  1) value = op(value, __shfl_xor_sync(mask, value,  1));        if constexpr (kNumLanesPerGroup <=  2) value = op(value, __shfl_xor_sync(mask, value,  2));        if constexpr (kNumLanesPerGroup <=  4) value = op(value, __shfl_xor_sync(mask, value,  4));        if constexpr (kNumLanesPerGroup <=  8) value = op(value, __shfl_xor_sync(mask, value,  8));        if constexpr (kNumLanesPerGroup <= 16) value = op(value, __shfl_xor_sync(mask, value, 16));    } else {        if constexpr (kNumLanesPerGroup >= 32) value = op(value, __shfl_xor_sync(mask, value, 16));        if constexpr (kNumLanesPerGroup >= 16) value = op(value, __shfl_xor_sync(mask, value,  8));        if constexpr (kNumLanesPerGroup >=  8) value = op(value, __shfl_xor_sync(mask, value,  4));        if constexpr (kNumLanesPerGroup >=  4) value = op(value, __shfl_xor_sync(mask, value,  2));        if constexpr (kNumLanesPerGroup >=  2) value = op(value, __shfl_xor_sync(mask, value,  1));    }    return value;}__global__ void verifyWarpReduceMaxKernel(const float* input, float* output) {        int tid = blockIdx.x * blockDim.x + threadIdx.x;        float my_val  = tid * 1.0;        float warp_max = warp_reduce<4,true>(my_val, ReduceMax<float>());        printf(" tid %d , %f \n", tid, warp_max);        output[tid] = warp_max;}
```

输出结果可以看到,  `tid = [0,4,8,12,16,20,24,28]`的max值都为28. 另外三组同理.. 这样就在 token 的 hidden 维度实现了求最大值.
第二层 : 写Smem
`smem_amax_reduction`也是以`float2`为单位进行存储的, `epilogue_warp_idx`为在一个WarpGroup内的 warp 编号, 计算时由于`float2`只需要 `* (STORE_BLOCK_M / 2)` 再加上 atom tile 的 offset, 最后在包含 atom tile内的index, 由于前一步采用了broadcast模式, 例如`tid = [0,4,8,12,16,20,24,28]`都为最大值, 因此只需要处理前面 4 个lane .

```
if (lane_idx < 4)    smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];__syncwarp();
```

然后等待 TMA store 释放 smem.

```
// Wait shared memory release from previous TMA store// And fence `smem_amax_reduction`const uint32_t tma_stage_idx = s % kNumTMAStoreStages;ptx::tma_store_wait<kNumTMAStoreStages - 1>();ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
```
第三层: 跨 Warp 配对取 max(`warp ⊕ 1`)
SwiGLU 把 `gate * up` 后, **两个相邻 warp 的输出共享同一个 BF16 swizzle ATOM**(即 BF16 视图下两 warp 的 8+8=16 行属于同一个 stmatrix 单元), 它们必须用**同一个缩放因子**, 否则反量化时无法还原. 所以这里通过异或 1(pair: warp0↔warp1, warp2↔warp3)取到对方写入的半 amax, 并取 `max`.

同前面第二层, 每个 warp 在自己的槽位 `epilogue_warp_idx*(STORE_BLOCK_M/2) + i*(ATOM_M/2) + [0..3]` 写 4 个 float2.

```
#pragma unrollfor (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {    // Reduce amax    // `^ 1` 把当前 warp 与"邻居 warp"配对.    const float2 wp_amax =        smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];    amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);    amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);
```
4.5.4.7 FP8量化1. 求UE8M0缩放因子
![图片](assets/8d93caff94aa.png)

如下所示:

```
   // Calculate SF    float2 sf, sf_inv;    math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);/*---------ref: get_e4m3_sf_and_sf_inv-----------*/template <bool kUseUE8M0 = true>CUTLASS_DEVICE void get_e4m3_sf_and_sf_inv(const float2& amax, float2& sf, float2& sf_inv) {    DG_STATIC_ASSERT(kUseUE8M0, "Must use UE8M0");    const float2 finfo_factor = {1.0 / 448.0, 1.0 / 448.0};  // E4M3 max=448    const auto scaled = __fmul2_rn(amax, finfo_factor);      // amax / 448    const auto exp_x = fast_log2_ceil(scaled.x);             // 向上取整的 log2    const auto exp_y = fast_log2_ceil(scaled.y);    sf.x = fast_pow2(exp_x), sf_inv.x = fast_pow2(-exp_x);   // 2^e —— 反量化用, 2^-e —— 量化用    sf.y = fast_pow2(exp_y), sf_inv.y = fast_pow2(-exp_y);}
```

其中 UE8M0 = 只保留 FP32 的 8 位指数, 无尾数无符号**. 所以 SF 必须是 2 的整数幂, 这里用 `log2_ceil` 保证 `amax * sf_inv ≤ 448`, 即量化后绝对落在 E4M3 可表示范围内.
2. cast 到 FP8 E4M3
然后将值 cast 到 FP8 E4M3, 先用 `sf_inv` 把 4 个 float 缩放进 E4M3 范围, 然后`__nv_fp8x4_e4m3` 一次把 4 个 float 打包成 32 位寄存器(4 字节 = 4 个 E4M3). 这正是后面 STSM 要的 32-bit 源数据.

![图片](assets/2fe933add37b.png)

```
    // Cast    const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);    const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);    const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));
```
3. STSM 写入共享内存
从`STORE_BLOCK`的角度来看, 布局如下:

![图片](assets/098b80c376a0.png)

存储的STSM指令采用了`stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [addr], {reg};` 其中 `m16n8.x1.trans`: 一个 warp 协作把 16×8 的 8-bit 矩阵**转置**写到 smem, 每 lane 提供 32 bit; 这里实际上一次写入 16 行 × 8 列的 FP8(128 字节).

在smem定位时, 首先根据 Warpgroup id 和 `store tile` 内的第 `i` 个ATOM Tile 定位.

```
    // STSM    uint32_t row = lane_idx;       // 0..31    uint32_t col = warp_idx_in_wg; // 0..3, 对应 4 个 16B 段    const auto smem_ptr = smem_cd[tma_stage_idx]         + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N  // 定位是哪个 WarpGroup        + i * ATOM_M * L1_OUT_BLOCK_N                       // 定位是哪个 ATOM        + row * L1_OUT_BLOCK_N                              // 行偏移, 每行(BLOCK_N / 2 = 64)个字节        + (col ^ (row / 2)) * kNumBankGroupBytes;           // 列偏移 + swizzle    ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);
```

XOR Swizzle 表示同一行内 4 个 16B bank-group 的物理位置随行号偏移, 如下图所示:

![图片](assets/df1873e3dc5c.png)
4.5.4.8 写入L2 input SF
```
// - 两 warp 配对后 SF 已同步, 只让偶数 warp 写；// - 每 4 lane 共享 1 个 SF(4 lane 内已 broadcast)——所以 `lane < 4` 每 lane 写 2 行(sf.x + sf.y).if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {    const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2;    const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;    const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);    const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();    const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;    __builtin_assume(token_base_idx < BLOCK_M);    const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M        + m_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;    const auto sf_addr = k_uint_idx * mn_stride + sf_pool_token_idx * sizeof(uint32_t) + byte_idx;        sf_base_ptr[sf_addr] =        (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);        // fp32 的 bit pattern 中, bit 23–30 是 exponent(8 bit)        // 右移 23 位得到 bits = sign(1) + exp(8) = 9 bit, 但 sign 恒为 0,         // 所以 `>> 23` 得到 `exp` 的 8-bit 值；        // 写入 `uint8_t`: 自动截断只保留低 8 位 = UE8M0 byte.     sf_base_ptr[sf_addr + 4 * sizeof(uint32_t)] =        (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);}
```
4.5.4.9 TMA store L1 output
最后issue TMA:

```
if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {    uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;    cute::tma_store_fence();    cute::SM90_TMA_STORE_2D::copy(        &tensor_map_l1_output,        smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,        out_n_idx,        m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);    cute::tma_store_arrive();}__syncwarp();
```

然后通过`l2_arrive_mask`通知L2

```
// Notify L2ptx::tma_store_wait<0>();ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {    DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");    ptx::red_or_rel_gpu(        workspace.get_l2_arrival_mask_ptr(pool_block_idx),        1ull << n_block_idx    );}__syncwarp();
```

#### 4.5.5 L2 Epilogue

L2 = 第二阶段 GEMM(down projection)结果后处理. 关键任务:

从 TMEM 读累加器

转 BF16

通过 STSM 写到 `smem_cd_l2`

按 `token_src_metadata` 确定每行的远端 rank/token/topk 位置

通过 NVLink(`sym_buffer.map`)直接写入远端 `combine_token_buffer[topk_idx][token_idx]`
4.5.5.1 TMEM读取
首先是任务切分和 L1 Epilogue相同, TMEM读取也基本一致:

```
#pragma unrollfor (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {        ptx::tcgen05_before_thread_sync();        tmem_empty_barriers[accum_stage_idx]->arrive(0u);        break;    }    #pragma unroll    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;        uint32_t values[ATOM_M];        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,                                               values[0], values[1], values[2], values[3]);        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,                                               values[4], values[5], values[6], values[7]);        cutlass::arch::fence_view_async_tmem_load();
```

然后这里会等待并释放 TMEM.

```
// Wait shared memory release from previous NVLink storeif (i == 0 and s > 0)    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);// Signal tensor memory consumedif (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {    ptx::tcgen05_before_thread_sync();    tmem_empty_barriers[accum_stage_idx]->arrive(0u);}
```
4.5.5.2 转换为BF16并保存到SMEM
```
// Store into shared memoryuint32_t row = lane_idx % 8;uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;const auto smem_ptr = smem_cd_l2 +    epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +    (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +    i * ATOM_M * kSwizzleCDMode +    row * (kNumBankGroupBytes * 8) +    (col ^ row) * kNumBankGroupBytes;ptx::SM90_U32x4_STSM_T<uint32_t>::copy(    math::cast_into_bf16_and_pack(values[0], values[1]),    math::cast_into_bf16_and_pack(values[2], values[3]),    math::cast_into_bf16_and_pack(values[4], values[5]),    math::cast_into_bf16_and_pack(values[6], values[7]),    smem_ptr);
```

然后等待, 并重新计算 `row_in_atom` 和 `bank_group_idx` 因为 NVLink 写入时的布局与 STSM 时不同(一warp 对应一行).

```
// Wait shared memory readyptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);// Write into remote buffersconst uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;const uint32_t bank_group_idx = lane_idx % 8;
```
4.5.5.3 NVLink 远端写
每个 epilogue warpgroup（4 个 warp = 128 线程）负责写 `STORE_BLOCK_M` 行. 本阶段的分配方式和 STSM 存储阶段不同: 每 warp 独占若干整行（one warp per row）, 每行被 16 个 lane 分成 16 个 float4（即 16 × 8 = 128 BF16 = BLOCK_N）.

![图片](assets/82e284524ccb.png)

`row_in_store` 和`m_idx_in_block`计算如下 :

```
#pragma unroll//每 warp 负责 `kNumRowsPerWarp = STORE_BLOCK_M / 8` 行for (uint32_t j = 0; j < kNumRowsPerWarp; ++j) {  const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;  const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;
```

然后排除 padding 行的处理, 提前退出, 对于合法的token寻找 dispatch阶段写入的源路由信息

```
  // 跳过属于 padding 的行（超出本专家实际 token 数）  if (m_idx_in_block >= valid_m) break;// 读取 dispatch 阶段写入的回写元数据：目标 rank / token / topk 槽位  const auto src_metadata =      *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);  const uint32_t dst_rank_idx = src_metadata.rank_idx;  const uint32_t dst_token_idx = src_metadata.token_idx;  const uint32_t dst_topk_idx = src_metadata.topk_idx;
```

对于smem的指针计算本质是一个 四项加法： 基地址 + 本 wg 大块 + 行内前/后半 atom + 行号 + XOR 后的 16 B 槽.

![图片](assets/f618c35c2c4e.png)

```
const auto smem_ptr =    smem_cd_l2                                            // ① warpgroup 的 smem 基地址  + epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * 2         // ② 本 wg 的 STORE_BLOCK_M × BLOCK_N 大块（×2 因 BF16）  + (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode  // ③ 行内前半 / 后半 atom（两个 128 B 段）  + row_in_store * kSwizzleCDMode                         // ④ 第几行 × 128 B  + (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;  // ⑤ XOR 反查后的 16 B 槽（= 1 个 float4）//从 smem 读出 16B packed float4const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));
```

涉及到的常量与中间量：

`bank_group_idx = lane_idx % 8` 8 个 bank group 一组循环（每个 bg = 4 banks × 4 B = 16 B）

`row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M`， `ATOM_M = 8`

`row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16` 行在整个 STORE_BLOCK_M 内的编号

`kSwizzleCDMode = 128 B`: 一个 swizzle atom 的字节数（= 64 BF16）

`kNumBankGroupBytes = 16 B` : 一个 float4 的大小（= 8 BF16）

`combine_token_buffer`的结构和寻址如下:

![图片](assets/7b7cb9c0208c.png)

```
  // 通过 sym_buffer.map 将本 rank 的指针映射为远端 rank 的 NVLink  // 地址 最终写入到 dst_rank 的  // combine_token_buffer[dst_topk][dst_token] 对应输出  const auto dst_token = combine_token_buffer      .get_rank_buffer(dst_topk_idx)  // 1. topk 维度      .get_data_buffer(dst_token_idx); // 2. 该 topk 的 token 槽    //3. 选择 hidden 切片  const auto dst_ptr = math::advance_ptr<float4>(dst_token.get_base_ptr(),      n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) +      (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));  *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;}
```

最后的 epilogue 全局同步

```
  // 确保下一轮 epilogue 在此之前不会读写 smem  ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
```

#### 4.5.6 Combine 阶段

他是Mega-MoE persistent kernel 的最末段. 把 L2 Epilogue 阶段各个rank写回到 `combine_token_buffer` 中的「带权 BF16 部分和」 按 token 做 top-k reduction, 最终写出全局输出 `y[num_tokens, hidden]`.
![图片](assets/5cfb0c891650.png)

4.5.6.1 预处理及同步
首先释放TMEM, 并进行 NVLink:Barrier, 等待所有的 rank 把数据写到 `combine_token_buffer`内, 然后再和dispatch warp又一次同步, 这是因为 Combine阶段需要占用 SMEM, 同步完成SMEM管理权交接, dispatch阶段可以安全的清理 workspace. 整个 smem 复用如下所示:

![图片](assets/b0a246009a16.png)

```
    // 释放 TMEM：两个 CTA 必须由相同逻辑 warp id 调用    if (epilogue_warp_idx == 0) Allocator().free(0, kNumTmemCols);    // NVLink barrier (grid sync + cross-rank signal + grid sync): ~4 us    // Combine 前的全局同步：跨 SM grid sync + 跨 rank NVLink barrier + 再一次    // grid sync，确保所有 rank 的 L2 输出已写入各 combine buffer    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,                         kEpilogueGridSyncIndex,                         kBeforeCombineReduceBarrierTag>(        workspace, sym_buffer, sm_idx, epilogue_thread_idx, [&]() {          ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);        });    // 与 dispatch warp 同步：保证 dispatch warp 可以安全地清理 workspace    ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads,                        kDispatchWithEpilogueBarrierIdx);
```
4.5.6.2 工作切分
整个token推进的循环如下所示:

```
for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;     token_idx < num_tokens;     token_idx += kNumSMs * kNumEpilogueWarps) { ... }
```

可以看到步进为`kNumSMs * kNumEpilogueWarps`, 也就是说整体会以一个warp一个token的方式进行处理. 然后在warp内部进一步沿 hidden 维切 1 或 2 个 chunk, 然后 32 个 lane 并行处理. 如下图所示:

![图片](assets/a3d56294bab6.png)

Chunk切分相关的变量如下, 整体在SMEM中分为 3 个Slot, 两个读一个写.  然后对于一个token 切分成 1 或 2 个chunk受到 SMEM容量的约束. 并且只允许切分成 1 或 2 个, 避免切分出太小的chunk. 最后根据 hidden 维度的信息, 计算出每个lane处理的数据量.

```
constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);constexpr uint32_t kNumChunkSlots = 3;                  // 2 load + 1 storeconstexpr uint32_t kNumMaxRegistersForBuffer = 128;constexpr uint32_t kNumChunks =    (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE     and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;constexpr uint32_t kNumChunkBytes  = kNumHiddenBytes / kNumChunks;constexpr uint32_t kNumChunkUint4  = kNumChunkBytes / sizeof(uint4);constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;   // 每个 lane 负责的 uint4 数
```

然后在编译期间还有一些静态检查: `kHidden % kNumChunks == 0`（可整除）, `3 * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks ≤ SMEM_BEFORE_BARRIER_SIZE`（smem 装得下）, `kNumChunkBytes % 16 == 0`（满足 TMA 1D 16B 对齐）,  `kNumChunkUint4 % 32 == 0`（每 lane 至少一个 uint4）, `kNumTopk ≤ 32`（一个 warp 即可承接全部 top-k 槽）.

运行期再做一次 smem 越界检查:

```
DG_DEVICE_ASSERT(    kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <=    static_cast<uint32_t>(reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));
```
4.5.6.3 内存分布SMEM分布
每个 epilogue warp 分得 3 个 chunk 槽：0/1 是 load 双缓，2 是 store , 多个 warp 通过 (warp_idx + slot * kNumEpilogueWarps)的索引交错排列以充分利用 smem.

![图片](assets/7bbf9f9f3e65.png)

这种**外层是 slot、内层是 warp**的 stripe pattern 让相邻 warp 的同 slot 块在物理 smem 中**紧邻**, 可最大化 bank 利用率，避免单 warp 局部热点.

```
const auto combine_load_buffer =    utils::PatternVisitor([&](const uint32_t& i) {      return math::advance_ptr<uint4>(          smem_buffer,          (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);    });    const auto combine_store_buffer = math::advance_ptr<uint4>(    smem_buffer,    (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);
```

然后每个 warp 拥有 2 个 mbarrier(对应 load slot 0/1). 在 kernel 初始化阶段：`combine_barriers[i]->init(1)`（L461），arrival count = 1(一次 TMA 完成 = 一次 arrive), 总数 = `kNumEpilogueWarps * 2`, 与 stripe 中 load slot 数量匹配.

```
    auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {      return combine_barriers[i + epilogue_warp_idx * 2];    });
```
寄存器布局
![图片](assets/7bfb95ed3dbb.png)
Combine_token_buffer 结构
Combine Epilogue需要处理的数据源油L2 epilogue 阶段经 `sym_buffer.map(dst_ptr, dst_rank_idx)` 写到**目标 rank** 的同一 buffer, 本阶段读取本 rank 自己的副本, 它的它的维度: `(kNumTopk, kNumMaxTokensPerRank)`, 每槽一个 BF16 token (`hidden` 维). 物理位置在 `sym_buffer` 内(symmetric memory), 所有 rank 同 offset.

```
const auto combine_token_buffer =    layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank,                   l2_sf_buffer.get_end_ptr());
```

![图片](assets/c7302073c2e1.png)
4.5.6.4 主循环流程
整个循环如, 按 warp 粒度遍历所有 token, 步长 = kNumSMs * kNumEpilogueWarps. 具体的工作切分细节参考 4.5.6.2 节.

```
for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;     token_idx < num_tokens;     token_idx += kNumSMs * kNumEpilogueWarps) { ... }
```
Top-k 槽读取
每个 lane 读一个 topk 槽(存储的是目标 rank id，-1 代表未使用).

```
const int stored_topk_slot_idx =    lane_idx < kNumTopk        ? static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>()                                 + token_idx * kNumTopk + lane_idx))        : -1;const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);
```

`__ballot_sync` 把 32 个 lane 的 valid 位编织成 `total_mask`,  后续在 chunk 循环里反复消费这份 mask.
4.5.6.5 Chunk 内主循环
Chunk循环如下所示:

```
// 遍历 chunk：hidden 被平分为 kNumChunks 个 chunkfor (uint32_t chunk = 0; chunk < kNumChunks; ++chunk) {    const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;        /*----1.Load----------*/    /*----2.Accumlate-----*/    /*----3.TMA store-----*/
```

每个 chunk 都执行三段: 预取 → 累加 → 写出, 构成 ping-pong 流水.

![图片](assets/b24ae651a6c5.png)
4.5.6.6 move_mask_and_load
主要作用是, 使用 `__ffs(mask) - 1` 按位顺序遍历 valid top-k 槽位, 从mask中移除选择rank, 然后在 warp 内挑选一个 lane 发 TMA, 一次 TMA 1D load 拉回 `kNumChunkBytes` 字节到 `combine_load_buffer[i]`, `mbarrier_arrive_and_set_tx` 把期望传输字节数注册到 mbarrier, 让消费者侧通过 phase 等待真正完成, `__syncwarp()` 保证 mask 修改在 warp 内可见, 避免后续判定竞争.

```
const auto move_mask_and_load = [&](const uint32_t& i) {    if (mask) {        const uint32_t slot_idx = __ffs(mask) - 1;          // 取最低 bit 对应 rank        mask ^= 1 << slot_idx;                              // 从 mask 移除        if (cute::elect_one_sync()) {                       // 选 1 个 lane 发起            const auto src_ptr = math::advance_ptr<uint8_t>(                combine_token_buffer.get_rank_buffer(slot_idx)                                    .get_data_buffer(token_idx)                                    .get_base_ptr(),                chunk_byte_offset);            ptx::tma_load_1d(combine_load_buffer[i], src_ptr,                             combine_load_barriers[i], kNumChunkBytes);            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i],                                            kNumChunkBytes);        }        __syncwarp();        return true;    }    return false;};
```
4.5.6.7 累加循环
整个数据流程如下图所示, 采用Ping-Pong预取的方式在 accum 当前 stage 时, 先把下一 stage 的 TMA 发出去.

![图片](assets/01583c288c27.png)

具体代码如下:

```
bool do_reduce = move_mask_and_load(load_stage_idx);   // 启动第一次 loadfloat2 reduced[...] = {};while (do_reduce) {    // 预取下一个 top-k 到另一个 stage    do_reduce = move_mask_and_load(load_stage_idx ^ 1);    // 等当前 stage TMA 完成    combine_load_barriers[load_stage_idx]->wait(combine_phase);    #pragma unroll    for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {        // 第 `lane_idx` 个 lane 在 chunk 内负责索引 `j*32 + lane_idx` 的所有 uint4；        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];                // 使用 `ptx::accumulate(float2, bf16x2)` 把 BF16 累加到 FP32 寄存器, 避免精度损失        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);        #pragma unroll        for (uint32_t l = 0; l < kNumElemsPerUint4; ++l)            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);    }    combine_phase ^= load_stage_idx;     // stage 翻转一整圈才翻 phase    load_stage_idx ^= 1;}
```
4.5.6.8 Cast BF16 + Store
累加器的精度为FP32, 但是`combine`的输出为BF16, 因此这里有一个cast, 将 float32 累加结果转回 BF16, 按 lane 写入 combine_store_buffer.

```
#pragma unrollfor (uint32_t j = 0; j < kNumUint4PerLane; ++j) {    uint4 casted;    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);    #pragma unroll    for (uint32_t l = 0; l < kNumElemsPerUint4; ++l)        casted_bf16[l] = __float22bfloat162_rn(reduced[j*4 + l]);    if (j == 0) {                  // 仅在第一次写之前等        ptx::tma_store_wait<0>();  // 等上一轮 TMA store 完成        __syncwarp();    }    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,                   casted.x, casted.y, casted.z, casted.w);}__syncwarp();
```

最后写完 smem → fence → 启动 TMA, 将本 chunk 作为一个 1D TMA store 写入 `y[token_idx][chunk]`

```
if (cute::elect_one_sync()) {    cute::tma_store_fence();    ptx::tma_store_1d(        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes                             + chunk_byte_offset),        combine_store_buffer, kNumChunkBytes);    cute::tma_store_arrive();}__syncwarp();
```

## 5. 一些分析讨论

首先我们来看DeepSeek在论文中的一些分析和建议:

观察与建议
我们分享在 Kernel 开发过程中的观察和经验, 并向硬件供应商提出一些建议, 希望有助于高效的硬件设计并实现更好的软硬件协同设计:

**计算-通信比.** 完全的通信-计算重叠取决于计算-通信比, 而不仅仅是带宽本身. 记峰值计算吞吐量为 $C$, 互连带宽为 $B$, 当 $C/B \le V_{comp} / V_{comm}$ 时, 通信可以被完全隐藏, 其中 $V_{comp}$ 表示计算量, $V_{comm}$ 表示通信量. 对于DeepSeek-V4-Pro, 每个token-专家对需要 $6hd$ 的FLOPs( SwiGLU 的 Gate, up/down projection), 但只需要 $3h$ 字节的通信(FP8分发 + BF16合并), 这可以简化为:

$$\frac{C}{B} \le 2d = 6144 \text{ FLOPs/Byte}$$

也就是说, 每GBps的互连带宽足以隐藏6.1 TFLOP/s计算量所需的通信. 一旦带宽满足这个阈值, 它就不再是瓶颈, 将额外的芯片面积用于进一步增加带宽会带来递减的回报. 作者鼓励未来的硬件设计瞄准这样的平衡点, 而不是无条件地扩展带宽.

**功率预算.** 极致的核融合会同时将计算, 内存和网络推向高负载, 使得功率限制(power throttling)成为一个关键的性能限制因素. 我们建议未来的硬件设计为这种完全并发的工作负载提供足够的功率余量.

**通信原语.** 作者采用一种基于“拉取(pull)”的方法, 每个GPU主动从远程GPU读取数据, 避免了细粒度“推送(push)”所需的高通知延迟. 未来具有更低延迟跨GPU信令的硬件将使推送变得可行, 并能实现更自然的通信模式.

**激活函数.** 作者建议将SwiGLU替换为一个不涉及指数或除法运算的低成本逐元素激活函数. 这直接减轻了GEMM后的处理负担, 并且在相同的参数预算下, 去掉门控投影可以扩大中间维度 $d$, 进一步放宽对带宽的要求.

渣注
**有些话涉密不能说, 这里说点简单的.**

首先对于**计算-通信比**, 这一段写的挺好的. 确实需要某种意义上的平衡, 并且通信和计算都打满的时候, 还有一些`功耗 / NOC拥塞干扰 / 通信带来的 CacheMiss` 等一系列的问题要处理. 顺带吐个槽, 那些年天天想拿 RoCE 做 ETH-ScaleUP 的出来讲讲 ? 其实还有一些潜在的问题, 例如整个代码中对于通信同步, 各种barrier, L2 Epilogue阶段为什么没有用TMA直接用的per lane `st.global`. 其实这些都是较大影响的地方.

这些问题其实又耦合到**通信原语**这个问题上, 简单来说基于`消息语义`还是`内存语义`, 或者某种混合的语义. 例如基于内存语义在需要发送token的时候, 还是需要昂贵的atomicAdd去拿到相应的slot.. 这里点到为止.

最后关于**激活函数**, 其实也是Blackwell系列芯片更大的面积用于TensorCore和TMEM后导致的SM数量相对较少, SFU的性能在B200上也没跟上, 因此对数和除法这些实现上还用了fastmath.

但是根本原因还是大量的通信控制类的代码在多个warp之间耦合在一起, 相互的等待和warp调度本质上在Blackwell这样的微架构是有些缺陷的. 参考以前的文章[《Inside Nvidia GPU: 谈谈Blackwell的不足并预测一下Rubin的微架构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496740&idx=1&sn=c9403138fa59d126fe6cfda19d9b2f76&scene=21#wechat_redirect)

最后, 整个代码花了一周多的时间认真看完分析完, 收获很大. 突然有种以前某项目的错觉, 既要低延迟又要高吞吐, 最后在各个地方一条条的指令扣, 内存也要非常细致的排布. 不过MegaMoE这些代码真的是一个非常不错的艺术品, 值得认真学习.