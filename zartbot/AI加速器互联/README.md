# DeepEPv2 系列读书笔记

> 结合 Google Cloud AI Infra (TPU/GPU) 视角的深度阅读笔记。
> 不是原文摘要，而是提炼核心设计思想 + 大量类比 + 与我们 TPU/GPU 工作的关联。

**原文**:
- [第 1 篇：ElasticBuffer + Barrier + PP + Engram + AGRS](DeepEPv2分析(1).md)
- [第 2 篇：EP Dispatch/Combine 总览](DeepEPv2分析(2)-EP Overview.md)
- [第 3 篇：EP Direct Dispatch/Combine Kernel 实现](DeepEPv2分析(3)-EP Direct Dispatch-Combine Kernel.md)
- [第 4 篇：EP Hybrid Dispatch/Combine Kernel 实现](DeepEPv2分析(4)-EP Hybrid Dispatch Combine Kernel.md)

---

## 一、DeepEPv2 是什么？解决什么问题？

MoE (Mixture of Experts) 模型的核心挑战：**token 要跨 GPU 搬家**。

一个 token 经过 router 后被分配给若干 expert（比如 Qwen3.5 是 512 选 10），但这些 expert 分布在不同 GPU 上。所以需要两步通信：

1. **Dispatch**: 把 token 从"我在哪里"发到"expert 在哪里"（散射）
2. **Combine**: expert 算完后，把结果从"expert 在哪里"发回"我原来在哪里"（聚合）

DeepEPv2 就是 DeepSeek 开源的、专门做这两步通信的库。V2 相比 V1 的核心变化是**底层从 NVSHMEM 换成了 NCCL Gin**，同时扩展支持了 PP、Engram、AGRS 等通信原语。

> **类比**: 想象一个快递分拣中心。每个 GPU 是一个分拣站，token 是包裹，expert 是目的地仓库。Dispatch 就是按地址把包裹发出去，combine 就是把处理完的回执寄回发件人。DeepEPv2 就是这个分拣中心的调度系统。

---

## 二、地基：Symmetric Memory（对称内存）

所有设计的基础是一块**跨所有 GPU 共享的内存**。

每个 GPU 分配一块同样大小、同样布局的内存，通过 NVLink 的 LSA (Local Shared Address) 机制，任何 GPU 都能**直接用本地指针读写远端 GPU 的对应地址**。

```
GPU 0 的 buffer:  [workspace | region A | region B | ...]
GPU 1 的 buffer:  [workspace | region A | region B | ...]
GPU 7 的 buffer:  [workspace | region A | region B | ...]
                   ↑ 完全相同的布局，任意 GPU 可以直接寻址任意 GPU 的任意位置
```

> **类比**: 8 个人坐在一张圆桌上，桌面被划成 8 块完全相同的区域。每个人可以直接伸手在别人的区域写字，不需要传纸条、不需要打招呼。NVLink 就是"伸手可达"的物理距离。
>
> **TPU 对比**: TPU 的 ICI (Inter-Chip Interconnect) 天然就是这种模型——chip 之间直接高速互联，集合通信由 XLA compiler 自动编排。GPU 需要手动做 `ncclMemAlloc` + `ncclCommWindowRegister` 来建立这个共享空间。TPU 的优势是 compiler 保证正确性；GPU 的优势是开发者可以做任意不规则的访问模式（比如 Engram 的随机 gather）。

初始化还有个细节值得注意：RDMA 的 QP (Queue Pair) 数量。如果网卡支持 fast-RDMA-atomic（CX7 以上），需要 65 个 QP（64 数据 + 1 notify）；不支持的话需要 129 个（数据 QP 翻倍）。

> **例子**: 以一个 8 机 × 8 卡的 64 GPU 集群为例（Qwen3.5 典型部署），每个 GPU 节点内 8 卡走 NVLink（ScaleUP），节点间走 RDMA（ScaleOut）。Symmetric Memory 覆盖 NVLink 域的 8 卡，RDMA 域用 Gin 的 signal API 做跨节点同步。

---

## 三、Barrier：怎么做分布式同步？

有了共享内存，下一个问题是：怎么确保"我写完了，你可以读了"？这就是 barrier。

### ScaleUP Barrier（NVLink 域，同一台机器内）

用一个 **phase-sign 乒乓协议**：

```
第 1 次 barrier: phase=0, sign=+1 → 每个 rank 给所有 peer 的 signal 槽加 1 → 等信号值 == 8（8 个 rank 都加了）
第 2 次 barrier: phase=1, sign=-1 → 每个 rank 给所有 peer 的 signal 槽减 1 → 等信号值 == 0
第 3 次 barrier: phase=0, sign=+1 → 又回到加 1...
```

> **类比**: 地铁的进出闸机计数器。进站 +1，出站 -1，来回交替。不需要每次把计数器归零，数字在 0 和 N 之间乒乓。传统方案是"写标记 → 等标记 → 清零标记"三步，乒乓方案省了清零这步。
>
> 内存语义用 `release/acquire`：写用 `red_add_rel_sys`（release 语义原子加），读用 `ld_acquire_sys`（acquire 语义读）。这保证 barrier 前的所有写操作对 barrier 后的读操作可见。

### ScaleOut Barrier（RDMA 域，跨机器）

走 NCCL Gin 的 signal API。每个 rank 给所有远端 rank 发 signal（原子 +1），然后忙等本地 signal 达到目标值。

关键设计：**QP 0 专门做 notify**，不跟数据传输的 QP 混用。

> **类比**: 数据 QP 是高速公路车道，notify QP 是紧急车道。你不能让救护车（同步信号）跟大卡车（数据包）挤在一起。

### Hybrid Barrier（混合，最常见的实际场景）

用 **2 个 SM 并行跑**：SM0 做 ScaleUP barrier（NVLink），SM1 做 ScaleOut barrier（RDMA）。两者走不同物理通道，互不干扰。

> **具体例子**: 8 机 × 8 卡集群，rank 0 要跟所有 63 个 rank 同步。SM0 用 NVLink 跟本机其他 7 个 rank 同步（~1μs），SM1 用 RDMA 跟其他 7 台机器同步（~5-10μs）。两者并行，总时间取较慢的那个（RDMA），而不是串行加起来。

---

## 四、PP / Engram / AGRS：三种通信原语

在 Symmetric Memory + Barrier 这套基础设施之上，DeepEPv2 构建了三种通信原语。

### PP（Pipeline Parallel）

环形拓扑，每个 rank 只跟前后 neighbor 通信。Buffer 分四段：recv-from-next、recv-from-prev、send-to-prev、send-to-next，每段内是环形队列。

Send 流程：
1. 等 slot 空出来（反向流控）
2. TMA 引擎异步拷贝 tensor → send buffer（2 级 mbarrier 流水线）
3. Grid sync 确保数据可见
4. RDMA put 到对端 recv buffer + signal

> **类比**: 工厂传送带上的托盘。发货前要确认托盘是空的（流控），然后机械臂把货放上去（TMA），确认放稳了（sync），最后按按钮传走（RDMA put）。对面收到后按另一个按钮说"我拿走了，托盘空了"（release signal）。

### Engram（分布式 KV 存储）

每个 rank 存一部分 entries，fetch 时可以跨 rank 读取任意 entry。32 个 QP 并行发 RDMA get 请求。

> **例子**: 假设 64 个 rank 各存 1000 个 expert state。Rank 5 需要读 rank 37 的第 42 号 entry。Engram 会通过 RDMA get 直接去 rank 37 的 symmetric buffer 里读，32 个 QP 可以同时给不同 rank 发请求。
>
> **TPU 场景**: 类似于跨 host 的 KV cache 共享——推理时一个 host 需要读另一个 host 上缓存的 KV 对。TPU 目前没有这么灵活的不规则 gather 机制。

### AGRS（All-Gather Reduce-Scatter，零 SM）

这是最优雅的设计。**限定在 NVLink 域内**，用 `cudaMemcpyBatchAsync` 一次 driver 调用下发所有 rank 对的拷贝，纯走 NVLink DMA copy engine，**不占用任何 SM**。

```
传统 AllGather (NCCL):  SM 跑 kernel 驱动通信 → 占 SM → 跟计算抢资源
AGRS:                   DMA copy engine 自动搬 → 0 SM → 跟计算完全 overlap
```

同步用 `cuStreamBatchMemOp`（CUDA driver 级的 write/wait），Session 机制用单调递增 ID 做隔离。

> **类比**: 传统方式是让员工（SM）亲自去送文件。AGRS 是把文件放在传送带（DMA engine）上自动送，员工腾出来干正事（计算）。
>
> **TPU 对比**: TPU 的 AllGather 走 ICI 硬件 collective，天然不占 MXU 计算资源——理念跟 AGRS 完全一致。区别是 AGRS 是软件层面在 NVLink 上做到的，TPU 是硬件原生支持。
>
> **限制**: AGRS 只在 NVLink 域内有效（因为依赖 `cudaMemcpyBatchAsync` 的 P2P 语义）。跨节点的 AllGather 还是得用 NCCL。

---

## 五、EP Dispatch/Combine：核心通信流程

### Buffer Layout：时分复用

Dispatch 和 combine 共用一块 buffer，`size = max(dispatch_size, combine_size)`。

> **类比**: 一个会议室，上午用来开 dispatch 会议（散射），下午用来开 combine 会议（聚合）。只需要按最大的会议人数配椅子就行。

Dispatch token 携带完整信息（hidden + scale factor + topk_idx + topk_weight + 源地址 + 链表指针），combine token 只带 hidden + topk_weight。

> **为什么 combine 更轻？** 因为 dispatch 是"散射"——源 rank 要告诉目标 rank"我是谁、我从哪来、我的权重是多少"。Combine 是"聚合"——所有路由信息已经在 EPHandle 里存好了，不需要再传。

### Direct vs Hybrid：单层 vs 两层互联

| | Direct 模式 | Hybrid 模式 |
|---|---|---|
| 适用场景 | 单机 8 卡，或多机但禁用 hybrid | 多机（NVLink + RDMA） |
| Buffer 分区 | send + recv 两段 | scaleup_recv + scaleout_send + scaleout_recv 三段 |
| SM 内 warp 角色 | 4 Notify + N Dispatch | 4 Notify + N Scaleout + N Forward |
| 典型例子 | 1×DGX H100 做 8-expert MoE | 8×DGX H100 做 512-expert MoE |

> **容易踩的坑**: `allow_hybrid_mode=false` 时，物理上有多机 RDMA 也会走 Direct 模式，把所有 rank 折成一个 flat 域。这时 RDMA 被当成"慢 NVLink"来用。
>
> **TPU 例子**: Qwen3.5-397B 在 TPU v7x-8 上用 DP=8+EP=8。4 个 chip / 8 个 device 全在 ICI 域内，没有 DCN 跨节点通信。这相当于 DeepEP 的 Direct 模式。如果扩展到多 host（比如 4×v7x-8），就需要 Hybrid 模式——ICI 做 ScaleUP，DCN 做 ScaleOut。
>
> **GPU 例子**: DeepSeek V3 训练用 2048 张 H800。每台 8 卡走 NVLink（ScaleUP），256 台之间走 RDMA（ScaleOut）。这是典型的 Hybrid 模式。每个 SM 里 Scaleout Warp 负责 RDMA 发数据到远端节点，Forward Warp 负责把从 RDMA 收到的数据通过 NVLink 转发到同节点其他 GPU。

### Forward Warp：两级互联的代价

Hybrid 模式独有的 Forward Warp 是理解两级互联开销的关键。

```
GPU 0 (机器 A) 要发 token 给 GPU 3 (机器 B):

Dispatch:
  GPU 0 → [Scaleout Warp: RDMA put] → GPU 0 (机器 B)
  GPU 0 (机器 B) → [Forward Warp: NVLink write] → GPU 3 (机器 B)

                  跨节点 RDMA              节点内 NVLink
```

> **类比**: 你寄国际快递。先从你的城市发到对方国家的海关口岸（RDMA），然后海关把包裹转给当地快递（NVLink）送到收件人手上。Forward Warp 就是那个"海关转快递"的环节。
>
> **TPU 对比**: TPU 的 DCN → ICI 转发也有类似的两步。差异是 TPU 的 ICI 转发是 hardware 自动做的，而 GPU 需要 Forward Warp 用 SM 资源手动搬。

### 三阶段 Kernel 流水线

Dispatch 和 combine 各自由 2-3 个 kernel 组成：

**Dispatch:**
1. **Deterministic Prologue**（可选）: 预算 per-rank/per-expert 计数
2. **Main Dispatch**: 核心通信，读 token → 写远端 buffer
3. **Copy Epilogue**: 从 buffer 拷贝到用户输出 tensor（PDL 依赖）

**Combine:**
1. **Main Combine**: 读 expert 结果 → 写回源 rank 的 reduce buffer
2. **Reduce Epilogue**: 做加权求和 + 加 bias（PDL 依赖）

PDL (Programmatic Dependent Launch) 是 Hopper 以上的特性：epilogue 不需要等主 kernel 全部完成，只要主 kernel 的最后几个 SM 跑完就能启动。

> **例子**: 假设主 dispatch kernel 用 108 个 SM，epilogue 用 132 个 SM（全量）。传统方式是 108 个 SM 全跑完 → grid sync → 再启动 epilogue。PDL 方式是 108 个 SM 中最后的几个一完成，epilogue 就开始跑，跟前面还在收尾的 SM 重叠执行。

### EPHandle：Dispatch 和 Combine 的桥梁

EPHandle 保存了 dispatch 产生的 7 个 metadata tensor，combine 用它来反向路由。

```
dispatch(tokens, topk_idx, topk_weights)
    → recv_tokens, handle

expert_compute(recv_tokens)
    → expert_output

combine(expert_output, handle)      ← handle 告诉 combine 结果该发回哪里
    → combined_output
```

**缓存复用**：如果路由不变，handle 可以直接传给下一次 dispatch。

> **例子**: RL 训练的 rollout replay——同一批 token 用相同的路由重新跑一遍 forward。第一次 dispatch 花 100μs 做 layout 计算 + CPU 同步，第二次传入 handle 后这些全跳过，notify warp 降为 0，可能省到 30μs。
>
> **推理场景**: Continuous batching 中如果连续几个 decode step 的 router 输出稳定（同一批 token 路由到同一批 expert），可以复用 handle 省掉 CPU 侧开销。这对小 batch（比如 batch_size=1 的实时推理）收益巨大，因为 CPU layout 计算的固定开销占比很高。

### 通信-计算 Overlap

三个 event 控制点形成流水线：

```
Step 1: dispatch_A → compute_A (expert GEMM) → combine_A
Step 2:              dispatch_B → compute_B   → combine_B
                     ↑ 跟 combine_A 重叠     ↑ 跟 dispatch_C 重叠
```

- `previous_event`: 主通信开始前等计算完成
- `previous_event_before_epilogue`: epilogue 开始前等另一个分支
- 返回 `event`: 标记通信完成

> **TPU 对比**: TPU 用 XLA 的 async collective（`start_all_gather` / `done_all_gather`），compiler 自动在计算和通信之间插入 overlap。GPU 需要开发者手动编排这条 event 链。自由度更高，但出错概率也更高。
>
> **例子**: DeepSeek V3 的 MoE 层。Attention 算完 → dispatch 发到各 expert → expert FFN 计算 → combine 发回来 → 加残差。通过 overlap，下一层的 attention 可以跟本层 combine 的尾巴重叠。

### allow_multiple_reduction：精度 vs 内存的权衡

Combine 阶段有个关键开关：

- `true`（默认）: 边收边 reduce，每个 rank 只保留 `min(num_ranks, num_topk)` 份副本
- `false`: 保留所有 (token × topk) 副本，最后一次性求和。精度更高但内存开销翻 num_topk 倍

> **具体数字**: Qwen3.5，512 expert top-10。如果 `false`，combine buffer 比 `true` 大 10 倍。在 H100 80GB 上可能直接 OOM。所以大 MoE 几乎必须用 `true`。

---

## 六、全局架构图：从 token 到 expert 再回来

```
Token (rank 0)
    │
    ▼ Router: topk_idx=[expert_42, expert_137, ...]
    │
    ▼ Dispatch ─────────────────────────────────┐
    │   Direct: NVLink 直写远端 buffer           │
    │   Hybrid: Scaleout Warp (RDMA) + Forward Warp (NVLink) │
    │                                            │
    ▼                                            ▼
Expert 42 (rank 5)                    Expert 137 (rank 17)
    │ FFN compute                         │ FFN compute
    ▼                                     ▼
    │                                     │
    ▼ Combine ──────────────────────────┐ │
    │   反向路由（EPHandle）             │ │
    │   加权求和 + bias                  │ │
    ▼                                    ▼ ▼
Token (rank 0) = Σ weight_i × expert_i(token)
```

整个过程的内存复用：
- 同一块 Symmetric Memory buffer，dispatch 时用 dispatch layout，combine 时用 combine layout
- Workspace 里的计数器、signal、链表指针等固定不变
- EPHandle 串联两个阶段，也可以缓存给下一次

---

## 七、Kernel 内部：Dispatch/Combine 的精密机械

> 第 1-2 篇讲了 DeepEPv2 "做什么"，这一节讲"怎么做"——把 dispatch 和 combine 的 kernel 实现拆开来看。
> **原文**: [第 3 篇：EP Direct Dispatch/Combine Kernel](DeepEPv2分析(3)-EP Direct Dispatch-Combine Kernel.md)

### 7.1 Warp 分工：流水线上的角色分配

一个 Direct Dispatch kernel 启动后，每个 SM 里的 warp 被分成 **4 种角色**，各司其职。

<details>
<summary>🔧 SVG 架构图：Dispatch Kernel Warp 分工</summary>

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 480" font-family="system-ui, sans-serif">
  <!-- Background -->
  <rect width="800" height="480" fill="#FAFAFA" rx="12"/>
  
  <!-- Title -->
  <text x="400" y="35" text-anchor="middle" font-size="18" font-weight="bold" fill="#1A237E">Direct Dispatch Kernel — Warp 角色分工</text>
  
  <!-- SM Box -->
  <rect x="30" y="55" width="740" height="400" fill="white" stroke="#E0E0E0" stroke-width="2" rx="10"/>
  <text x="50" y="80" font-size="14" font-weight="bold" fill="#424242">SM (1 of 108)</text>
  
  <!-- Notify Warps -->
  <rect x="50" y="95" width="160" height="130" fill="#E3F2FD" stroke="#1565C0" stroke-width="1.5" rx="8"/>
  <text x="130" y="118" text-anchor="middle" font-size="13" font-weight="bold" fill="#1565C0">Notify Warps</text>
  <text x="130" y="138" text-anchor="middle" font-size="11" fill="#37474F">Warp 0-3 (固定 4 个)</text>
  <line x1="60" y1="148" x2="200" y2="148" stroke="#90CAF9" stroke-width="1"/>
  <text x="130" y="166" text-anchor="middle" font-size="10" fill="#546E7A">① 统计 per-expert token 数</text>
  <text x="130" y="182" text-anchor="middle" font-size="10" fill="#546E7A">② 通知远端 rank</text>
  <text x="130" y="198" text-anchor="middle" font-size="10" fill="#546E7A">③ 等远端 count 回复</text>
  <text x="130" y="214" text-anchor="middle" font-size="10" fill="#546E7A">④ CPU 同步 (可跳过)</text>

  <!-- Dispatch Warps -->
  <rect x="230" y="95" width="160" height="130" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5" rx="8"/>
  <text x="310" y="118" text-anchor="middle" font-size="13" font-weight="bold" fill="#2E7D32">Dispatch Warps</text>
  <text x="310" y="138" text-anchor="middle" font-size="11" fill="#37474F">Warp 4-N (可变)</text>
  <line x1="240" y1="148" x2="380" y2="148" stroke="#A5D6A7" stroke-width="1"/>
  <text x="310" y="166" text-anchor="middle" font-size="10" fill="#546E7A">① TMA load hidden+SF</text>
  <text x="310" y="182" text-anchor="middle" font-size="10" fill="#546E7A">② 读 topk → 写 metadata</text>
  <text x="310" y="198" text-anchor="middle" font-size="10" fill="#546E7A">③ Dedup + slot 分配</text>
  <text x="310" y="214" text-anchor="middle" font-size="10" fill="#546E7A">④ TMA store NVLink/RDMA</text>

  <!-- Forward Warps (Hybrid only) -->
  <rect x="410" y="95" width="160" height="130" fill="#FFF3E0" stroke="#E65100" stroke-width="1.5" rx="8" stroke-dasharray="6,3"/>
  <text x="490" y="118" text-anchor="middle" font-size="13" font-weight="bold" fill="#E65100">Forward Warps</text>
  <text x="490" y="138" text-anchor="middle" font-size="11" fill="#37474F">仅 Hybrid 模式</text>
  <line x1="420" y1="148" x2="560" y2="148" stroke="#FFCC80" stroke-width="1"/>
  <text x="490" y="166" text-anchor="middle" font-size="10" fill="#546E7A">接收 RDMA 数据</text>
  <text x="490" y="182" text-anchor="middle" font-size="10" fill="#546E7A">↓</text>
  <text x="490" y="198" text-anchor="middle" font-size="10" fill="#546E7A">NVLink 转发到</text>
  <text x="490" y="214" text-anchor="middle" font-size="10" fill="#546E7A">本节点其他 GPU</text>

  <!-- Scaleout Warps (Hybrid only) -->
  <rect x="590" y="95" width="160" height="130" fill="#FCE4EC" stroke="#C62828" stroke-width="1.5" rx="8" stroke-dasharray="6,3"/>
  <text x="670" y="118" text-anchor="middle" font-size="13" font-weight="bold" fill="#C62828">Scaleout Warps</text>
  <text x="670" y="138" text-anchor="middle" font-size="11" fill="#37474F">仅 Hybrid 模式</text>
  <line x1="600" y1="148" x2="740" y2="148" stroke="#EF9A9A" stroke-width="1"/>
  <text x="670" y="166" text-anchor="middle" font-size="10" fill="#546E7A">RDMA put 到远端节点</text>
  <text x="670" y="182" text-anchor="middle" font-size="10" fill="#546E7A">↓</text>
  <text x="670" y="198" text-anchor="middle" font-size="10" fill="#546E7A">跨机器搬运数据</text>
  <text x="670" y="214" text-anchor="middle" font-size="10" fill="#546E7A">+ signal 通知到达</text>

  <!-- Flow arrows -->
  <defs>
    <marker id="arrowD" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#757575"/>
    </marker>
  </defs>
  
  <!-- Timeline -->
  <rect x="50" y="245" width="700" height="55" fill="#F5F5F5" stroke="#BDBDBD" stroke-width="1" rx="6"/>
  <text x="60" y="265" font-size="12" font-weight="bold" fill="#424242">执行时序 →</text>
  
  <rect x="70" y="272" width="100" height="20" fill="#E3F2FD" stroke="#1565C0" rx="3"/>
  <text x="120" y="287" text-anchor="middle" font-size="9" fill="#1565C0">Notify 先行</text>
  
  <rect x="180" y="272" width="30" height="20" fill="#EEEEEE" stroke="#9E9E9E" rx="3"/>
  <text x="195" y="287" text-anchor="middle" font-size="9" fill="#757575">等</text>
  
  <rect x="220" y="272" width="200" height="20" fill="#E8F5E9" stroke="#2E7D32" rx="3"/>
  <text x="320" y="287" text-anchor="middle" font-size="9" fill="#2E7D32">Dispatch 主循环 (per-token)</text>
  
  <rect x="430" y="272" width="120" height="20" fill="#FFF3E0" stroke="#E65100" rx="3"/>
  <text x="490" y="287" text-anchor="middle" font-size="9" fill="#E65100">Forward 转发</text>
  
  <rect x="560" y="272" width="120" height="20" fill="#FCE4EC" stroke="#C62828" rx="3"/>
  <text x="620" y="287" text-anchor="middle" font-size="9" fill="#C62828">Scaleout RDMA</text>
  
  <!-- Analogy box -->
  <rect x="50" y="315" width="700" height="55" fill="#FFFDE7" stroke="#F9A825" stroke-width="1" rx="6"/>
  <text x="65" y="335" font-size="12" fill="#F57F17">💡 类比：餐厅后厨</text>
  <text x="65" y="352" font-size="11" fill="#795548">Notify = 前台点单员（统计每桌要几道菜）→ Dispatch = 备料厨师（把食材送到各灶台）</text>
  <text x="65" y="366" font-size="11" fill="#795548">Forward = 传菜员（在楼层之间传菜）→ Scaleout = 外卖骑手（送到其他门店）</text>

  <!-- TPU comparison -->
  <rect x="50" y="385" width="700" height="60" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1" rx="6"/>
  <text x="65" y="405" font-size="12" fill="#283593">🔗 TPU 对比</text>
  <text x="65" y="422" font-size="11" fill="#37474F">TPU 没有"warp 分工"概念 — XLA compiler 把通信编译成 ICI/DCN 硬件指令序列。</text>
  <text x="65" y="438" font-size="11" fill="#37474F">GPU 的 warp specialization 相当于让程序员手动做 XLA 做的事，换来对不规则通信的完全控制权。</text>
</svg>
```

</details>

> **类比**: 想象一个大餐厅的后厨。Notify Warp 是前台点单员——先统计每桌要几道菜，把汇总表传到后厨。Dispatch Warp 是备料厨师——按单子把食材分配到各个灶台。Forward Warp 是楼层传菜员——只在有多层楼的大餐厅（Hybrid 模式）才需要，负责把一楼接到的菜搬到二楼。Scaleout Warp 是外卖骑手——负责送到其他分店（跨机器 RDMA）。
>
> **TPU 对比**: TPU 没有"warp 分工"的概念。XLA compiler 直接把通信编译成 ICI/DCN 硬件指令序列，不需要手动分配 SM 资源。GPU 的 warp specialization 相当于让程序员手动做 XLA 编译器做的事——代价是代码复杂度，收益是对不规则通信模式的完全控制权。

### 7.2 Dispatch 主 kernel：16 步精密编舞

Dispatch 的核心逻辑可以拆成三个阶段：**准备 → 搬运 → 触发收尾**。

**阶段一：Notify Warp 先行侦察**

1. 每个 warp 遍历自己负责的 token，读 `topk_idx`，给对应 expert 的计数器做 `atomicAdd`
2. **跨 SM 归约**（grid reduce）：用 NVLink 原子加把所有 SM 的计数合并到 SM 0
3. Notify warp 把 per-rank 计数发给远端 rank（NVLink 或 RDMA signal）
4. 等远端回复计数，算出 per-expert 的前缀和（prefix sum）→ 这就是每个 expert 在 buffer 里的起始槽位

> **类比**: 双十一前，物流系统先做"预分拣"——统计每个分拣站要处理多少包裹，预留好货架空间。不能等包裹到了再临时找位置，否则挤成一团。
>
> **具体数字**: 108 个 SM 各统计一部分 token，最后合并到 SM 0。如果每 token topk=10，1024 个 token 就是 10240 次 atomicAdd 分散在 108 个 SM 上。Grid reduce 用 NVLink 原子加，~1μs 搞定。

**阶段二：Dispatch Warp 搬运**

5. **Dual psum**（双前缀和）：每个 SM 用自己的局部 `psum_expert` 做 `atomicAdd` 抢槽位。"Dual"是指有 deterministic 和 runtime 两套 psum，后面讲
6. **TMA load**：把 hidden 向量 + scale factor 从用户 tensor 异步加载到 SMEM
7. **读 topk + 写 metadata**：从 `topk_idx`/`topk_weight` 读路由信息，写入 buffer 的 metadata 区
8. **Dedup + slot 分配**：同一个 token 被多个 expert 选中时去重，用 `atomicAdd(psum_expert + e, 1)` 抢互斥 slot
9. **等 TMA 完成**：`mbarrier_wait` 等 hidden 数据到 SMEM
10. **TMA store**：写到目标位置（NVLink 直达远端 / 或 RDMA staging buffer）
11. **RDMA put**（如果需要跨节点）：等 staging 落盘后发 RDMA

> **一个巧妙的 trick — `encode_decode_positive`**: 写 hidden 的时候，在 scale factor 的**符号位**做手脚。写入时把正数变负数（flip sign bit），收端轮询这个值——看到负数就知道数据已到达，再 flip 回来恢复原值。这样用一个 bit 实现了"ready flag"，不需要额外的同步信号。
>
> **类比**: 你在停车场等朋友。约定好：车灯灭=没到，车灯亮=到了。朋友不需要打电话告诉你"我到了"，你扫一眼车灯就知道。DeepEPv2 把"车灯"嵌入了数据本身（scale factor 的符号位），省了单独的信号通道。
>
> **TPU 对比**: TPU 的 XLA 不需要这种 trick——ICI 的 send/recv 有硬件级的完成通知。GPU 这样做是因为 NVLink 的 `st.global` 没有自带的 completion signal，只能用数据本身携带"就绪"信息。

**阶段三：触发 Epilogue**

12. 所有 warp 完成后，主 kernel 末尾调用 `cudaTriggerProgrammaticLaunchCompletion`——这是 PDL 机制，直接唤醒预排好的 epilogue kernel，**不经过 CPU**

> **例子**: 主 kernel 108 SM 跑完最后一个 token 的瞬间，epilogue kernel（132 SM）立刻启动。如果用传统方式，需要主 kernel 结束 → CPU 收到完成回调 → CPU 启动 epilogue kernel，中间可能有 5-20μs 的 CPU 调度延迟。PDL 把这段省掉了。

### 7.3 Dispatch Epilogue：从 buffer 到用户 tensor

Epilogue 的任务是把 dispatch 主 kernel 写入 buffer 的数据，**搬到用户期望的输出 tensor**（`recv_x`）。

为什么不在主 kernel 里直接写 `recv_x`？因为主 kernel 不知道数据什么时候全部到齐。Buffer 的每个 slot 由不同 rank 写入，到达时间不确定。Epilogue 等 PDL 触发后，保证所有 slot 都已写入，再统一搬运。

两种模式的搬运方式不同：

| | 非 Expand 模式 | Expand 模式 |
|---|---|---|
| 输出布局 | 一行 = 一个 token | 一行 = 一个 (token, expert) 对 |
| 槽位数 | ≤ `num_tokens_per_rank` | ≤ `num_tokens_per_rank × topk` |
| 去向索引 | `atomicAdd(psum)` 抢 slot | 从 metadata 读 `dst_tensor_idx` |
| 典型场景 | 标准 MoE | 需要精确追踪每份 topk 副本 |

> **类比**: 非 expand 就像快递站按收件人地址分拣——一个包裹对应一个地址。Expand 模式就像按"收件人 × 商品"分拣——一个包裹里的每件商品单独编号入库。

Epilogue 还维护了一个**链表结构**：同一个 token 被多个 expert 选中时，第一份的 metadata 里存着指向第二份的指针，第二份指向第三份……combine 阶段顺着链表就能找到所有副本。

### 7.4 Deterministic Prologue：可复现性的代价

这是一个可选步骤。为什么需要它？

```
普通模式:  atomicAdd(psum + expert_id, 1)  →  返回的 old_value 就是 slot 编号
问题:      多个 SM 并行 atomicAdd，谁先谁后不确定 → 每次运行 slot 分配不同
后果:      浮点加法不满足结合律！ a+(b+c) ≠ (a+b)+c
           → reduce 结果微小偏差 → 调试噩梦
```

Deterministic prologue 用 **4 轮协作式前缀和** 提前算好每个 SM 在每个 expert 上的起始 slot：

1. 每个 SM 独立统计自己负责的 token 中 per-expert 计数
2. SM 0 收集所有 SM 的局部计数
3. SM 0 做 exclusive prefix sum（跨 SM 维度）
4. 广播回各 SM

这样每个 SM 的 slot 分配完全确定，跟调度顺序无关。

> **类比**: 考试的时候，老师可以让学生自己找座位（`atomicAdd` 抢座，快的先坐），也可以按学号提前排好座位表（deterministic prologue）。前者效率高但每次座位不同，后者多花一步排座但结果可复现。
>
> **代价**: 需要一次 grid sync（所有 SM 同步），加上 SM 0 的 prefix sum 计算。在 108 SM 的 H100 上大约增加 2-5μs。对于需要可复现的训练场景（比如调试 loss spike），这个代价完全值得。

### 7.5 Combine 主 kernel：三条回家的路

Combine 是 dispatch 的逆过程——把 expert 算完的结果发回源 rank。但逆过程**不是对称的**，因为 dispatch 是"一对多散射"，combine 是"多对一聚合"，需要做 **reduce**（加权求和）。

Combine 主 kernel 的核心是一个**三分支决策**，根据 expand 模式和 `allow_multiple_reduction` 的组合，选择不同的 reduce 策略：

<details>
<summary>🔧 SVG 架构图：Combine 三分支决策树</summary>

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 520" font-family="system-ui, sans-serif">
  <!-- Background -->
  <rect width="800" height="520" fill="#FAFAFA" rx="12"/>
  
  <!-- Title -->
  <text x="400" y="35" text-anchor="middle" font-size="18" font-weight="bold" fill="#1A237E">Combine 主 Kernel — 三分支决策</text>
  
  <!-- Decision node -->
  <polygon points="400,60 550,110 400,160 250,110" fill="#E8EAF6" stroke="#3F51B5" stroke-width="2"/>
  <text x="400" y="107" text-anchor="middle" font-size="12" font-weight="bold" fill="#1A237E">expand 模式?</text>
  <text x="400" y="123" text-anchor="middle" font-size="12" font-weight="bold" fill="#1A237E">multiple_reduction?</text>
  
  <!-- Branch labels -->
  <text x="260" y="170" text-anchor="middle" font-size="11" fill="#757575">非 expand</text>
  <text x="260" y="184" text-anchor="middle" font-size="11" fill="#757575">或 expand 仅 1 份有效</text>
  <text x="400" y="170" text-anchor="middle" font-size="11" fill="#757575">expand +</text>
  <text x="400" y="184" text-anchor="middle" font-size="11" fill="#757575">multi_reduce</text>
  <text x="560" y="170" text-anchor="middle" font-size="11" fill="#757575">expand +</text>
  <text x="560" y="184" text-anchor="middle" font-size="11" fill="#757575">非 multi_reduce</text>
  
  <!-- Lines from decision -->
  <line x1="325" y1="135" x2="170" y2="200" stroke="#3F51B5" stroke-width="1.5"/>
  <line x1="400" y1="160" x2="400" y2="200" stroke="#3F51B5" stroke-width="1.5"/>
  <line x1="475" y1="135" x2="630" y2="200" stroke="#3F51B5" stroke-width="1.5"/>
  
  <!-- Branch A -->
  <rect x="50" y="200" width="240" height="180" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5" rx="8"/>
  <text x="170" y="225" text-anchor="middle" font-size="14" font-weight="bold" fill="#2E7D32">分支 A：直通</text>
  <text x="170" y="245" text-anchor="middle" font-size="11" fill="#546E7A">no_local_reduce</text>
  <line x1="60" y1="255" x2="280" y2="255" stroke="#A5D6A7" stroke-width="1"/>
  <text x="170" y="275" text-anchor="middle" font-size="11" fill="#37474F">TMA load 源 token</text>
  <text x="170" y="292" text-anchor="middle" font-size="11" fill="#37474F">↓</text>
  <text x="170" y="309" text-anchor="middle" font-size="11" fill="#37474F">TMA store 到远端 buffer</text>
  <text x="170" y="326" text-anchor="middle" font-size="11" fill="#37474F">(NVLink 直达 / RDMA staging)</text>
  <line x1="60" y1="340" x2="280" y2="340" stroke="#A5D6A7" stroke-width="1"/>
  <text x="170" y="358" text-anchor="middle" font-size="10" fill="#795548">💡 最简单：只搬不算</text>
  <text x="170" y="374" text-anchor="middle" font-size="10" fill="#795548">类似快递原件退回</text>
  
  <!-- Branch B -->
  <rect x="310" y="200" width="200" height="180" fill="#FFF3E0" stroke="#E65100" stroke-width="1.5" rx="8"/>
  <text x="410" y="225" text-anchor="middle" font-size="14" font-weight="bold" fill="#E65100">分支 B：先合再发</text>
  <text x="410" y="245" text-anchor="middle" font-size="11" fill="#546E7A">local reduce</text>
  <line x1="320" y1="255" x2="500" y2="255" stroke="#FFCC80" stroke-width="1"/>
  <text x="410" y="275" text-anchor="middle" font-size="11" fill="#37474F">N 份 topk 副本</text>
  <text x="410" y="292" text-anchor="middle" font-size="11" fill="#37474F">↓ SMEM BF16 累加</text>
  <text x="410" y="309" text-anchor="middle" font-size="11" fill="#37474F">1 份 reduce 结果</text>
  <text x="410" y="326" text-anchor="middle" font-size="11" fill="#37474F">↓ TMA store 到远端</text>
  <line x1="320" y1="340" x2="500" y2="340" stroke="#FFCC80" stroke-width="1"/>
  <text x="410" y="358" text-anchor="middle" font-size="10" fill="#795548">💡 省带宽：N 份变 1 份</text>
  <text x="410" y="374" text-anchor="middle" font-size="10" fill="#795548">类似先在本地合账再转</text>
  
  <!-- Branch C -->
  <rect x="530" y="200" width="230" height="180" fill="#FCE4EC" stroke="#C62828" stroke-width="1.5" rx="8"/>
  <text x="645" y="225" text-anchor="middle" font-size="14" font-weight="bold" fill="#C62828">分支 C：逐份单发</text>
  <text x="645" y="245" text-anchor="middle" font-size="11" fill="#546E7A">expanded send</text>
  <line x1="540" y1="255" x2="750" y2="255" stroke="#EF9A9A" stroke-width="1"/>
  <text x="645" y="275" text-anchor="middle" font-size="11" fill="#37474F">每个有效 topk 副本</text>
  <text x="645" y="292" text-anchor="middle" font-size="11" fill="#37474F">↓ 独立 TMA load → store</text>
  <text x="645" y="309" text-anchor="middle" font-size="11" fill="#37474F">各自发送到远端</text>
  <text x="645" y="326" text-anchor="middle" font-size="11" fill="#37474F">(NVLink / RDMA 逐条发)</text>
  <line x1="540" y1="340" x2="750" y2="340" stroke="#EF9A9A" stroke-width="1"/>
  <text x="645" y="358" text-anchor="middle" font-size="10" fill="#795548">💡 最精确：不丢精度</text>
  <text x="645" y="374" text-anchor="middle" font-size="10" fill="#795548">类似每份快递单独寄</text>
  
  <!-- Bottom summary -->
  <rect x="50" y="400" width="710" height="50" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1" rx="6"/>
  <text x="65" y="420" font-size="12" fill="#283593">🔗 TPU 对比: XLA 的 reduce-scatter 编译时就确定策略 (hierarchical / ring)，</text>
  <text x="65" y="438" font-size="12" fill="#283593">     不像 GPU 需要 runtime 在三条路径间动态选择。TPU 的确定性换来简单性。</text>
  
  <!-- Arrows -->
  <rect x="50" y="460" width="710" height="45" fill="#FFFDE7" stroke="#F9A825" stroke-width="1" rx="6"/>
  <text x="65" y="478" font-size="11" fill="#F57F17">💡 选择逻辑：A 最快（0 计算），B 最省带宽（N→1 压缩），C 最精确（无 reduce 误差）。</text>
  <text x="65" y="494" font-size="11" fill="#F57F17">    Qwen3.5 (top-10) 典型用 B：同一 token 在本 rank 有多份 topk 副本时，先本地合再发。</text>
</svg>
```

</details>

**分支 A（直通）**: 源 tensor 里只有 1 份有效数据（非 expand 模式，或 expand 但只有 1 个 topk 副本落在本 rank）。直接 TMA load → TMA store，不做任何计算。

> **类比**: 快递原件退回——包裹完好无损，直接贴个退回地址就寄出去。

**分支 B（先合再发）**: 本 rank 持有同一 token 的多份 topk 副本（expand + `allow_multiple_reduction`）。先在 SMEM 做 BF16 向量化累加，reduce 成 1 份，再 TMA store 发走。

> **类比**: 你有 3 张银行卡要转账到同一个人。与其转 3 笔，不如先在本地合账成 1 笔再转——省了 2 次跨行手续费（网络带宽）。
>
> **实现细节**: `combine_reduce` 用 `nv_bfloat16` 的 `hadd`（hardware add）做累加，展开因子（unroll factor）由编译期根据 hidden 大小和寄存器压力自动选择（最大 4）。`__popc(reduce_valid_mask)` 算出本 rank 持有几份副本——如果 8 个 rank 平分 topk=10，每 rank 约 1-2 份。

**分支 C（逐份单发）**: expand 但不允许 multiple reduction。每份 topk 副本单独发送，保留最高精度。

> **为什么有人不想 reduce？** 浮点加法不满足结合律。边收边 reduce 的结果可能跟最后一次性 reduce 略有不同。对于某些需要数值可复现的训练场景，宁可多花带宽也要精度一致。

**NVLink vs RDMA 双路径**: 三个分支最终都需要把数据发到源 rank。如果源 rank 在 NVLink 域内，用 `get_sym_ptr` 直接远端写入（零拷贝）。否则先 TMA store 到本地 staging buffer，再走 RDMA put。

> **具体例子**: 8 机 × 8 卡集群。Rank 5（机器 A）的 expert 算完了 rank 37（机器 E）的 token。`gin.is_nvlink_accessible(37)` 返回 false → 走 RDMA 路径：先写到本地 `send_buffer[37][token_idx]` → `gin.put` → RDMA 引擎搬到机器 E 的 `recv_buffer[5][token_idx]`。如果目标是 rank 6（同机器 A），直接 NVLink 写到 rank 6 的 recv_buffer，省了 staging 拷贝。

**Combine 结尾**: 用传统 `gpu_barrier`（不是 PDL），确保所有 RDMA/NVLink 写入完成后才进入 epilogue。

> **为什么不像 dispatch 用 PDL？** Dispatch → epilogue 是同一 rank 上的接力（主 kernel 写 buffer，epilogue 读 buffer），PDL 天然适合。Combine 涉及跨 rank 数据传输——必须等**所有** rank 都发完才能开始 reduce，这需要全局 barrier 而非单机的 PDL 触发。

### 7.6 Combine Epilogue：最终的加权求和

Combine 主 kernel 把各 rank 的结果写入 `recv_buffer` 后，epilogue 负责最后一步：**把 N 份结果 reduce 成 1 份**，写入用户的 `combined_x` tensor。

<details>
<summary>🔧 SVG 架构图：Dispatch-Combine 完整流水线</summary>

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 500" font-family="system-ui, sans-serif">
  <!-- Background -->
  <rect width="850" height="500" fill="#FAFAFA" rx="12"/>
  
  <!-- Title -->
  <text x="425" y="30" text-anchor="middle" font-size="17" font-weight="bold" fill="#1A237E">Dispatch → Expert → Combine 完整 Kernel 流水线</text>
  
  <!-- Defs -->
  <defs>
    <marker id="arrowP" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#546E7A"/>
    </marker>
    <marker id="arrowPDL" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#D32F2F"/>
    </marker>
    <marker id="arrowBar" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#1565C0"/>
    </marker>
  </defs>
  
  <!-- ===== DISPATCH PHASE ===== -->
  <text x="30" y="60" font-size="13" font-weight="bold" fill="#2E7D32">DISPATCH 阶段</text>
  
  <!-- Deterministic Prologue -->
  <rect x="30" y="70" width="150" height="55" fill="#C8E6C9" stroke="#2E7D32" stroke-width="1.5" rx="6" stroke-dasharray="5,3"/>
  <text x="105" y="92" text-anchor="middle" font-size="11" font-weight="bold" fill="#1B5E20">Deterministic</text>
  <text x="105" y="107" text-anchor="middle" font-size="11" font-weight="bold" fill="#1B5E20">Prologue (可选)</text>
  <text x="105" y="122" text-anchor="middle" font-size="9" fill="#546E7A">4 轮 grid sync</text>
  
  <!-- Arrow -->
  <line x1="180" y1="97" x2="198" y2="97" stroke="#546E7A" stroke-width="1.5" marker-end="url(#arrowP)"/>
  
  <!-- Main Dispatch -->
  <rect x="200" y="70" width="180" height="55" fill="#E8F5E9" stroke="#2E7D32" stroke-width="2" rx="6"/>
  <text x="290" y="92" text-anchor="middle" font-size="12" font-weight="bold" fill="#1B5E20">Main Dispatch</text>
  <text x="290" y="107" text-anchor="middle" font-size="10" fill="#546E7A">Notify → Dispatch → RDMA</text>
  <text x="290" y="120" text-anchor="middle" font-size="9" fill="#757575">108 SM / 16 步</text>
  
  <!-- PDL Arrow -->
  <line x1="380" y1="97" x2="420" y2="97" stroke="#D32F2F" stroke-width="2" marker-end="url(#arrowPDL)" stroke-dasharray="4,2"/>
  <text x="400" y="88" text-anchor="middle" font-size="9" font-weight="bold" fill="#D32F2F">PDL</text>
  
  <!-- Dispatch Epilogue -->
  <rect x="422" y="70" width="160" height="55" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5" rx="6"/>
  <text x="502" y="92" text-anchor="middle" font-size="11" font-weight="bold" fill="#1B5E20">Dispatch Epilogue</text>
  <text x="502" y="107" text-anchor="middle" font-size="10" fill="#546E7A">buffer → recv_x</text>
  <text x="502" y="120" text-anchor="middle" font-size="9" fill="#757575">132 SM / 链表遍历</text>
  
  <!-- ===== EXPERT COMPUTE ===== -->
  <line x1="582" y1="97" x2="608" y2="97" stroke="#546E7A" stroke-width="1.5" marker-end="url(#arrowP)"/>
  <rect x="610" y="65" width="210" height="65" fill="#F3E5F5" stroke="#7B1FA2" stroke-width="2" rx="6"/>
  <text x="715" y="87" text-anchor="middle" font-size="12" font-weight="bold" fill="#4A148C">Expert Compute</text>
  <text x="715" y="104" text-anchor="middle" font-size="10" fill="#546E7A">FFN / Attention / MoE gate</text>
  <text x="715" y="120" text-anchor="middle" font-size="9" fill="#757575">用户 kernel，不属于 DeepEP</text>
  
  <!-- ===== COMBINE PHASE ===== -->
  <text x="30" y="170" font-size="13" font-weight="bold" fill="#1565C0">COMBINE 阶段</text>
  
  <!-- Main Combine -->
  <rect x="30" y="180" width="250" height="55" fill="#E3F2FD" stroke="#1565C0" stroke-width="2" rx="6"/>
  <text x="155" y="202" text-anchor="middle" font-size="12" font-weight="bold" fill="#0D47A1">Main Combine</text>
  <text x="155" y="217" text-anchor="middle" font-size="10" fill="#546E7A">三分支 (A 直通 / B 合发 / C 逐发)</text>
  <text x="155" y="230" text-anchor="middle" font-size="9" fill="#757575">NVLink bypass / RDMA put</text>
  
  <!-- Barrier Arrow -->
  <line x1="280" y1="207" x2="330" y2="207" stroke="#1565C0" stroke-width="2" marker-end="url(#arrowBar)"/>
  <text x="305" y="198" text-anchor="middle" font-size="9" font-weight="bold" fill="#1565C0">gpu</text>
  <text x="305" y="222" text-anchor="middle" font-size="9" font-weight="bold" fill="#1565C0">barrier</text>
  
  <!-- Combine Epilogue -->
  <rect x="332" y="180" width="250" height="55" fill="#E3F2FD" stroke="#1565C0" stroke-width="1.5" rx="6"/>
  <text x="457" y="202" text-anchor="middle" font-size="11" font-weight="bold" fill="#0D47A1">Combine Epilogue</text>
  <text x="457" y="217" text-anchor="middle" font-size="10" fill="#546E7A">去重 → reduce → TMA store</text>
  <text x="457" y="230" text-anchor="middle" font-size="9" fill="#757575">+ bias 融合 + topk_weights</text>
  
  <!-- Output -->
  <line x1="582" y1="207" x2="608" y2="207" stroke="#546E7A" stroke-width="1.5" marker-end="url(#arrowP)"/>
  <rect x="610" y="185" width="210" height="45" fill="#E0F2F1" stroke="#00695C" stroke-width="1.5" rx="6"/>
  <text x="715" y="207" text-anchor="middle" font-size="12" font-weight="bold" fill="#004D40">combined_x</text>
  <text x="715" y="222" text-anchor="middle" font-size="10" fill="#546E7A">+ combined_topk_weights</text>
  
  <!-- ===== KEY DIFFERENCES ===== -->
  <text x="30" y="275" font-size="13" font-weight="bold" fill="#424242">Dispatch vs Combine 关键差异</text>
  
  <!-- Comparison table as boxes -->
  <rect x="30" y="285" width="390" height="90" fill="white" stroke="#E0E0E0" stroke-width="1" rx="6"/>
  <text x="225" y="305" text-anchor="middle" font-size="12" font-weight="bold" fill="#2E7D32">Dispatch (散射)</text>
  <text x="50" y="325" font-size="10" fill="#37474F">• Kernel 结束: PDL 触发 epilogue (无全局 barrier)</text>
  <text x="50" y="342" font-size="10" fill="#37474F">• Epilogue: buffer → 用户 recv_x (纯搬运)</text>
  <text x="50" y="359" font-size="10" fill="#37474F">• Warp: Notify + Dispatch + Forward + Scaleout</text>
  <text x="50" y="369" font-size="10" fill="#37474F">• Barrier tag: kDispatchTag (不与 Combine 冲突)</text>

  <rect x="440" y="285" width="380" height="90" fill="white" stroke="#E0E0E0" stroke-width="1" rx="6"/>
  <text x="630" y="305" text-anchor="middle" font-size="12" font-weight="bold" fill="#1565C0">Combine (聚合)</text>
  <text x="460" y="325" font-size="10" fill="#37474F">• Kernel 结束: gpu_barrier (需全局同步)</text>
  <text x="460" y="342" font-size="10" fill="#37474F">• Epilogue: 去重 + reduce + bias → combined_x</text>
  <text x="460" y="359" font-size="10" fill="#37474F">• Warp: 全部 warp 均匀分 token (无角色区分)</text>
  <text x="460" y="369" font-size="10" fill="#37474F">• Barrier tag: kCombineTag0/1 (两轮 barrier)</text>

  <!-- ===== TPU COMPARISON ===== -->
  <rect x="30" y="395" width="790" height="90" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1" rx="6"/>
  <text x="50" y="418" font-size="13" font-weight="bold" fill="#283593">TPU 对比：XLA 怎么做同样的事？</text>
  <text x="50" y="438" font-size="11" fill="#37474F">TPU 的 dispatch/combine 被编译成 ICI all-to-all + reduce-scatter 硬件指令。没有 warp 分工、没有 TMA、</text>
  <text x="50" y="455" font-size="11" fill="#37474F">没有手动 barrier tag 管理。XLA 保证 correctness，GPU warp specialization 保证 performance flexibility。</text>
  <text x="50" y="475" font-size="11" fill="#37474F">代价对比：GPU 需要 ~2000 行 CUDA 实现 dispatch+combine；TPU 只需 jax.lax.all_to_all() 一行调用 + XLA 编译。</text>
</svg>
```

</details>

Epilogue 的执行流程：

1. **PDL 同步**：`cudaGridDependencySynchronize()` 等 combine 主 kernel 完成
2. **读 topk_idx → 映射 dst_rank_idx**：每个 lane 处理一个 topk 副本
3. **去重**：根据配置决定去重策略：
   - Expand + 不 reduce：不去重（每份 topk 独立）
   - Hybrid + 非 expand：按 expert 所属 rank 去重
   - 其他：按 dst_rank_idx 去重
4. **`combine_reduce`**：把所有有效副本累加到 SMEM，**同时融合 bias**（bias_0 + bias_1 在 reduce 循环内一并加，省一个 round-trip）
5. **TMA store** 到 `combined_x[token_idx]`
6. **写 topk_weights**：从 comm_buffer 拷贝到 `combined_topk_weights`

> **Bias 融合是个聪明设计**: 传统做法是 reduce hidden → 加 bias → 写出，两次 HBM 读写。DeepEPv2 把 bias 加法嵌入 reduce 的内层循环——每次从 `x[slot]` 读一个向量时，顺手加上 `bias[token_idx]` 的对应值。少了一次完整的 tensor 遍历。
>
> **TPU 对比**: XLA 的 `reduce_scatter` 后如果跟着 `bias_add`，编译器会自动做类似的 fusion（算子融合）。GPU 需要在 CUDA kernel 里手动实现，但换来了完全控制哪些操作融合、哪些不融合的灵活度。

### 7.7 小结：kernel 级设计哲学

| 设计选择 | 为什么这样做 | TPU 对应 |
|---|---|---|
| Warp specialization | 一个 kernel 内多种角色并行 | XLA 编译期静态调度 |
| TMA (Tensor Memory Accelerator) | 异步搬运 hidden 向量，不占 warp 执行单元 | DMA engine (VPU 管理) |
| PDL 触发 epilogue | 省 CPU 调度延迟 (~5-20μs) | XLA pipelining |
| encode_decode_positive trick | 用数据符号位做 ready flag | ICI 硬件 completion signal |
| 三分支 combine | 精度/带宽/复杂度三方权衡 | XLA 编译时固定策略 |
| Bias 融合 reduce | 省一次 HBM round-trip | XLA 算子融合 |
| Deterministic prologue | 训练可复现性 | XLA 编译期确定性 |
| gpu_barrier (combine) vs PDL (dispatch) | Combine 需全局同步，dispatch 不需要 | 统一用 ICI barrier |

---

## 八、Hybrid Kernel：两层互联下的 dispatch/combine

> 第 3 篇讲的是 **Direct** 模式——所有 rank 折成一个 flat NVLink 域。但真实的大 MoE（512 expert、几百张卡）跑不进单机，必须跨节点。这一节讲 **Hybrid** 模式：NVLink（机内）+ RDMA（机间）两层互联怎么协同。
> **原文**: [第 4 篇：EP Hybrid Dispatch/Combine Kernel](DeepEPv2分析(4)-EP Hybrid Dispatch Combine Kernel.md)

Direct 和 Hybrid 的本质区别，是**世界被拆成了两维**。

### 8.1 两层拓扑：world = scaleout × scaleup

Direct 模式里，rank 是一维的——64 个 rank 排成一条线，谁跟谁都能直接 NVLink 寻址。Hybrid 模式里，rank 变成**二维坐标**：

```
rank_idx = scaleout_rank_idx × kNumScaleupRanks + scaleup_rank_idx
           └── 你在哪台机器 ──┘                    └── 机器内第几张卡 ─┘
```

`scaleup` 维度走 **NVLink**（机内 8 卡直连，零拷贝对称寻址），`scaleout` 维度走 **RDMA**（机间跨网卡）。两个维度用不同的 NCCL team tag 区分：

- `ncclTeamTagLsa` = NVLink / scaleup（LSA = Local Shared Address，机内对称 VA）
- `ncclTeamTagRail` = RDMA / scaleout（Rail = 网络轨道）

<details>
<summary>🔧 SVG 架构图：两层拓扑 + rank 编址</summary>

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 470" font-family="system-ui, sans-serif">
  <rect width="820" height="470" fill="#FAFAFA" rx="12"/>
  <text x="410" y="34" text-anchor="middle" font-size="18" font-weight="bold" fill="#1A237E">Hybrid 拓扑：world = scaleout × scaleup</text>
  <text x="410" y="56" text-anchor="middle" font-size="12" fill="#757575">rank_idx = scaleout_rank_idx × kNumScaleupRanks + scaleup_rank_idx</text>

  <!-- Machine A -->
  <rect x="40" y="80" width="330" height="180" fill="#E3F2FD" stroke="#1565C0" stroke-width="2" rx="10"/>
  <text x="205" y="104" text-anchor="middle" font-size="13" font-weight="bold" fill="#0D47A1">机器 0 (scaleout_rank=0)</text>
  <!-- GPUs in A -->
  <rect x="60" y="120" width="60" height="55" fill="white" stroke="#42A5F5" rx="6"/>
  <text x="90" y="143" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565C0">rank 0</text>
  <text x="90" y="160" text-anchor="middle" font-size="9" fill="#757575">su=0</text>
  <rect x="130" y="120" width="60" height="55" fill="white" stroke="#42A5F5" rx="6"/>
  <text x="160" y="143" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565C0">rank 1</text>
  <text x="160" y="160" text-anchor="middle" font-size="9" fill="#757575">su=1</text>
  <rect x="200" y="120" width="60" height="55" fill="white" stroke="#42A5F5" rx="6"/>
  <text x="230" y="143" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565C0">rank 2</text>
  <text x="230" y="160" text-anchor="middle" font-size="9" fill="#757575">su=2</text>
  <rect x="270" y="120" width="80" height="55" fill="white" stroke="#42A5F5" rx="6"/>
  <text x="310" y="143" text-anchor="middle" font-size="11" font-weight="bold" fill="#1565C0">rank 3</text>
  <text x="310" y="160" text-anchor="middle" font-size="9" fill="#757575">su=3</text>
  <!-- NVLink line -->
  <line x1="70" y1="195" x2="340" y2="195" stroke="#2E7D32" stroke-width="3"/>
  <text x="205" y="215" text-anchor="middle" font-size="11" font-weight="bold" fill="#2E7D32">NVLink (scaleup / ncclTeamTagLsa)</text>
  <text x="205" y="232" text-anchor="middle" font-size="9" fill="#546E7A">机内零拷贝对称寻址 · ~1μs</text>

  <!-- Machine B -->
  <rect x="450" y="80" width="330" height="180" fill="#FCE4EC" stroke="#C62828" stroke-width="2" rx="10"/>
  <text x="615" y="104" text-anchor="middle" font-size="13" font-weight="bold" fill="#B71C1C">机器 1 (scaleout_rank=1)</text>
  <rect x="470" y="120" width="60" height="55" fill="white" stroke="#EF5350" rx="6"/>
  <text x="500" y="143" text-anchor="middle" font-size="11" font-weight="bold" fill="#C62828">rank 4</text>
  <text x="500" y="160" text-anchor="middle" font-size="9" fill="#757575">su=0</text>
  <rect x="540" y="120" width="60" height="55" fill="white" stroke="#EF5350" rx="6"/>
  <text x="570" y="143" text-anchor="middle" font-size="11" font-weight="bold" fill="#C62828">rank 5</text>
  <text x="570" y="160" text-anchor="middle" font-size="9" fill="#757575">su=1</text>
  <rect x="610" y="120" width="60" height="55" fill="white" stroke="#EF5350" rx="6"/>
  <text x="640" y="143" text-anchor="middle" font-size="11" font-weight="bold" fill="#C62828">rank 6</text>
  <text x="640" y="160" text-anchor="middle" font-size="9" fill="#757575">su=2</text>
  <rect x="680" y="120" width="80" height="55" fill="white" stroke="#EF5350" rx="6"/>
  <text x="720" y="143" text-anchor="middle" font-size="11" font-weight="bold" fill="#C62828">rank 7</text>
  <text x="720" y="160" text-anchor="middle" font-size="9" fill="#757575">su=3</text>
  <line x1="480" y1="195" x2="750" y2="195" stroke="#2E7D32" stroke-width="3"/>
  <text x="615" y="215" text-anchor="middle" font-size="11" font-weight="bold" fill="#2E7D32">NVLink (scaleup / ncclTeamTagLsa)</text>
  <text x="615" y="232" text-anchor="middle" font-size="9" fill="#546E7A">机内零拷贝对称寻址 · ~1μs</text>

  <!-- RDMA between machines -->
  <line x1="370" y1="150" x2="450" y2="150" stroke="#E65100" stroke-width="4" stroke-dasharray="8,4"/>
  <text x="410" y="140" text-anchor="middle" font-size="10" font-weight="bold" fill="#E65100">RDMA</text>
  <text x="410" y="175" text-anchor="middle" font-size="9" fill="#E65100">scaleout</text>

  <!-- Address formula box -->
  <rect x="40" y="280" width="740" height="70" fill="#FFFDE7" stroke="#F9A825" stroke-width="1" rx="6"/>
  <text x="55" y="302" font-size="12" fill="#F57F17">💡 编址例子 (kNumScaleupRanks=4):</text>
  <text x="55" y="322" font-size="11" fill="#795548">rank 6 = scaleout_rank 1 × 4 + scaleup_rank 2 → 机器 1 的第 3 张卡</text>
  <text x="55" y="340" font-size="11" fill="#795548">要发给 rank 6 → 先 RDMA 跨到机器 1（scaleout 维），再 NVLink 送到 su=2（scaleup 维）= 两跳</text>

  <!-- TPU comparison -->
  <rect x="40" y="360" width="740" height="90" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1" rx="6"/>
  <text x="55" y="382" font-size="12" font-weight="bold" fill="#283593">🔗 TPU 对比</text>
  <text x="55" y="402" font-size="11" fill="#37474F">TPU 的两维拓扑天然存在：ICI（芯片内 mesh，对应 scaleup）+ DCN（跨 pod slice，对应 scaleout）。</text>
  <text x="55" y="420" font-size="11" fill="#37474F">但 TPU 的 rank 编址由 XLA + mesh topology 自动管理，程序员写 jax.sharding.Mesh 声明维度即可，</text>
  <text x="55" y="438" font-size="11" fill="#37474F">不用手算 rank_idx。GPU 这里的二维编址 = 手动实现 TPU mesh 的 device_id → (ici, dcn) 映射。</text>
</svg>
```

</details>

> **类比**: 想象一栋楼里的公司。`scaleup` 是同一层楼的工位——你伸手就能把文件递给邻座（NVLink 零拷贝）。`scaleout` 是不同楼层——得走楼梯／坐电梯（RDMA）。给二楼 3 号工位送文件，你要先坐电梯上二楼（跨 scaleout），再在二楼走到 3 号位（走 scaleup）。这就是**两跳**。
>
> **TPU 对比**: TPU 的 ICI（芯片间 mesh）+ DCN（跨 slice）就是天然的两维拓扑。区别在于 TPU 用 `jax.sharding.Mesh(devices, axis_names=('dcn','ici'))` 声明两维，rank → 坐标的映射由 XLA 自动算。GPU 这里的 `rank_idx = scaleout × N + scaleup` 是手动实现同一件事。

### 8.2 Hybrid Dispatch：三角色接力，两跳到位

Direct dispatch 是 4 Notify + N Dispatch 两类角色，一跳（NVLink 直写）到位。Hybrid dispatch 多了一维，warp 角色也变成三类，数据流变成两跳：

| | Direct 模式 | Hybrid 模式 |
|---|---|---|
| Warp 角色 | Notify + Dispatch | Notify + **Scaleout-Send** + **Forward** |
| Buffer 分段 | send + recv | **scaleup** + **scaleout_send** + **scaleout_recv** |
| 数据流 | 1 跳（NVLink 直写） | 2 跳（RDMA → recv → NVLink → scaleup） |
| rank 空间 | 一维 flat | 二维 scaleout × scaleup |

数据的两跳旅程（源 rank 要把 token 发给远机的某张卡）：

```
第 1 跳 (RDMA, 跨机):   源 rank ──Scaleout-Send warp: gin.put<Rail>──► 目标机的 scaleout_recv_buffer
第 2 跳 (NVLink, 机内): 目标机的中转 rank ──Forward warp: NVLink 写──► 目标 scaleup peer 的 scaleup_buffer
```

<details>
<summary>🔧 SVG 架构图：Hybrid Dispatch 两跳数据流</summary>

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 500" font-family="system-ui, sans-serif">
  <rect width="840" height="500" fill="#FAFAFA" rx="12"/>
  <text x="420" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#1A237E">Hybrid Dispatch — 两跳数据流</text>
  <text x="420" y="54" text-anchor="middle" font-size="12" fill="#757575">源 rank (机器0) → 目标 rank (机器1 的 scaleup peer)</text>

  <defs>
    <marker id="hdArrow" markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L9,3.5 L0,7" fill="#E65100"/>
    </marker>
    <marker id="hdArrowN" markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L9,3.5 L0,7" fill="#2E7D32"/>
    </marker>
  </defs>

  <!-- Source machine -->
  <rect x="30" y="75" width="230" height="230" fill="#E3F2FD" stroke="#1565C0" stroke-width="2" rx="10"/>
  <text x="145" y="98" text-anchor="middle" font-size="13" font-weight="bold" fill="#0D47A1">机器 0 — 源 rank</text>

  <rect x="50" y="115" width="190" height="45" fill="#E8F5E9" stroke="#2E7D32" rx="6"/>
  <text x="145" y="133" text-anchor="middle" font-size="11" font-weight="bold" fill="#2E7D32">① Notify warp</text>
  <text x="145" y="150" text-anchor="middle" font-size="9" fill="#546E7A">算 dst_scaleup_slot + 通知</text>

  <rect x="50" y="170" width="190" height="45" fill="#FCE4EC" stroke="#C62828" rx="6"/>
  <text x="145" y="188" text-anchor="middle" font-size="11" font-weight="bold" fill="#C62828">② Scaleout-Send warp</text>
  <text x="145" y="205" text-anchor="middle" font-size="9" fill="#546E7A">gin.put&lt;Rail&gt; RDMA 发出</text>

  <rect x="50" y="225" width="190" height="60" fill="white" stroke="#90A4AE" rx="6"/>
  <text x="145" y="245" text-anchor="middle" font-size="10" font-weight="bold" fill="#455A64">scaleout_send_buffer</text>
  <text x="145" y="262" text-anchor="middle" font-size="9" fill="#757575">RDMA staging (打包待发)</text>
  <text x="145" y="277" text-anchor="middle" font-size="9" fill="#757575">kNumTopk 份副本</text>

  <!-- Target machine -->
  <rect x="450" y="75" width="360" height="300" fill="#FCE4EC" stroke="#C62828" stroke-width="2" rx="10"/>
  <text x="630" y="98" text-anchor="middle" font-size="13" font-weight="bold" fill="#B71C1C">机器 1 — 目标机</text>

  <!-- relay rank -->
  <rect x="470" y="115" width="320" height="120" fill="#FFF8E1" stroke="#F9A825" stroke-width="1.5" rx="8"/>
  <text x="630" y="135" text-anchor="middle" font-size="11" font-weight="bold" fill="#F57F17">中转 rank (scaleout peer)</text>
  <rect x="485" y="145" width="290" height="40" fill="white" stroke="#90A4AE" rx="6"/>
  <text x="630" y="163" text-anchor="middle" font-size="10" font-weight="bold" fill="#455A64">scaleout_recv_buffer</text>
  <text x="630" y="178" text-anchor="middle" font-size="9" fill="#757575">RDMA 落地点（第 1 跳终点）</text>
  <rect x="485" y="192" width="290" height="38" fill="#FFF3E0" stroke="#E65100" rx="6"/>
  <text x="630" y="210" text-anchor="middle" font-size="11" font-weight="bold" fill="#E65100">③ Forward warp</text>
  <text x="630" y="224" text-anchor="middle" font-size="9" fill="#546E7A">查 topk → 链表 → NVLink 落地（第 2 跳）</text>

  <!-- scaleup peer -->
  <rect x="470" y="255" width="320" height="100" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5" rx="8"/>
  <text x="630" y="275" text-anchor="middle" font-size="11" font-weight="bold" fill="#1B5E20">目标 scaleup peer (同机器 1)</text>
  <rect x="485" y="285" width="290" height="55" fill="white" stroke="#66BB6A" rx="6"/>
  <text x="630" y="305" text-anchor="middle" font-size="10" font-weight="bold" fill="#2E7D32">scaleup_buffer</text>
  <text x="630" y="322" text-anchor="middle" font-size="9" fill="#757575">expert 最终读取的输入 token</text>
  <text x="630" y="336" text-anchor="middle" font-size="9" fill="#757575">(NVLink 零拷贝写入 get_sym_ptr&lt;Lsa&gt;)</text>

  <!-- Hop 1: RDMA -->
  <line x1="240" y1="192" x2="470" y2="165" stroke="#E65100" stroke-width="3" stroke-dasharray="8,4" marker-end="url(#hdArrow)"/>
  <text x="355" y="150" text-anchor="middle" font-size="11" font-weight="bold" fill="#E65100">第 1 跳: RDMA 跨机</text>
  <text x="355" y="166" text-anchor="middle" font-size="9" fill="#E65100">~5-10μs</text>

  <!-- Hop 2: NVLink -->
  <line x1="630" y1="230" x2="630" y2="283" stroke="#2E7D32" stroke-width="3" marker-end="url(#hdArrowN)"/>
  <text x="700" y="258" text-anchor="middle" font-size="11" font-weight="bold" fill="#2E7D32">第 2 跳: NVLink</text>
  <text x="700" y="273" text-anchor="middle" font-size="9" fill="#2E7D32">~1μs 机内</text>

  <!-- TPU comparison -->
  <rect x="30" y="395" width="780" height="90" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1" rx="6"/>
  <text x="45" y="417" font-size="12" font-weight="bold" fill="#283593">🔗 TPU 对比：DCN → ICI 两跳</text>
  <text x="45" y="437" font-size="11" fill="#37474F">TPU 的跨 slice all-to-all 也是两跳：DCN 把数据搬到目标 slice 的某个芯片，再走 ICI 分发到目标芯片。</text>
  <text x="45" y="455" font-size="11" fill="#37474F">差异是 TPU 的"中转 + 转发"由 XLA collective 编译器生成，没有显式的 "Forward warp"。</text>
  <text x="45" y="473" font-size="11" fill="#37474F">GPU 的 Forward warp = 手动实现 TPU 编译器自动做的 DCN→ICI relay，代价是 SM 占用，收益是灵活度。</text>
</svg>
```

</details>

> **类比**: 国际转运仓。你（源 rank）把一批包裹交给国际物流（RDMA），它先运到对方国家的**中转仓**（scaleout_recv_buffer）。中转仓的分拣员（Forward warp）拆开一看地址，再用**同城快递**（NVLink）送到最终收件人（scaleup peer）。三个角色：你的打包员（Scaleout-Send）、国际物流、当地分拣员。
>
> **为什么要中转仓？** 因为 RDMA 只能点到点发到"某台机器的某块内存"，不能直接跳到那台机器里的另一张卡。跨机后必须落地一次，再靠机内 NVLink 二次分发。TPU 的 DCN → ICI 也是同理——DCN 落到目标 slice 的入口芯片，再 ICI mesh 内转发。

### 8.3 Notify warp 的四阶段（A→D）

Hybrid 的 Notify 比 Direct 复杂，因为要同时统计两个维度的计数。它分四个阶段：

- **阶段 A**：本地统计——遍历自己的 token，按 (scaleout_rank, scaleup_rank) 二维累加计数
- **阶段 B**：scaleout 维交换——通过 RDMA signal 把 per-scaleout 计数发给各远机
- **阶段 C**：scaleup 维交换——通过 NVLink 把 per-scaleup 计数在机内交换
- **阶段 D**：算两级前缀和——先算 scaleout 偏移，再算机内 scaleup 偏移，拼出每个 token 的最终落点 `dst_scaleup_slot`

> **类比**: 快递预分拣的两级版本。先统计"每个城市要发多少件"（scaleout），再统计"每个城市里每个小区要多少件"（scaleup），最后拼出每件包裹的货架编号（省级仓 + 小区网点）。一维预分拣变两级预分拣。

### 8.4 Forward warp：round-robin 轮询多个链表

Forward warp 是 Hybrid dispatch 的灵魂——它要盯着从 RDMA 收到的数据，一到就转发到 NVLink。难点在于**它同时服务多个 scaleup peer**，不能死盯一个。

DeepEPv2 的做法是**多 peer 链表 + round-robin 派发**：

1. 每个 channel 为 `kNumScaleupRanks` 个 peer 各维护一条链表（`channel_linked_list`）
2. 每轮从所有链表各读一个节点，用一个 `wip_mask`（bitmask）标记"本轮哪些 peer 有新数据"
3. 内层 while 从 `dst_scaleup_rank_idx + 1` 开始 round-robin 找下一个有效 peer，`ptx::ffs`（find first set）挑出来派发
4. 派发完这轮，链表游标 `stored_ll_idx` 前进，继续下一轮

关键是那句 round-robin：`start = (dst + 1) % N`，避免每次都从 peer 0 开始扫——否则永远优先服务 peer 0，其他 peer 饿死。

> **类比**: 一个理货员管 8 个货架，每个货架有一队待处理的货。他不能盯着 1 号货架清完再看 2 号（后面的会积压），而是**转圈巡查**：这轮处理到 3 号，下轮就从 4 号开始扫，扫一圈回来。`wip_mask` 是"哪些货架这轮有货"的速查表，`ffs` 是"从当前位置往后第一个有货的货架"。
>
> **链表索引变换的巧思**: 链表节点存的是 `token_idx`，但要还原成源全局坐标 `src_global_token_idx = (scaleout_rank × scaleup_ranks + scaleup_rank) × max_tok + token`——就是 §8.1 那个二维编址的逆运算。Forward warp 读出 token 后，靠这个公式反推它从哪台机器的哪张卡来。

### 8.5 Hybrid Combine：反着走的两跳

Combine 是 dispatch 的逆过程，所以两跳方向也反过来：dispatch 是 **RDMA 下行 → NVLink 落地**，combine 是 **NVLink 上行 → RDMA 下行回源**。

combine 用三段 buffer + 两类 warp 角色：

| Buffer 段 | 作用 |
|---|---|
| `scaleup_buffer` | expert 结果先在机内 NVLink 汇聚到中转 rank |
| `scaleout_send_buffer` | 中转 rank 打包待 RDMA 发回源机 |
| `scaleout_recv_buffer` | 源机接收 RDMA 回来的结果 |

| Warp 角色 | 职责 |
|---|---|
| **Scale-up warp** | 按链表把本 rank 的 expert 结果 NVLink 写到中转 peer 的 scaleup_buffer（第 1 跳，上行）|
| **Forward warp** | 从 scaleup_buffer 读出、（可选）本地 reduce、RDMA 发回源 scaleout peer（第 2 跳，下行）|

<details>
<summary>🔧 SVG 架构图：Hybrid Combine 反向流 + compute-comm overlap</summary>

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 540" font-family="system-ui, sans-serif">
  <rect width="840" height="540" fill="#FAFAFA" rx="12"/>
  <text x="420" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#1A237E">Hybrid Combine — 反向两跳 + 通信计算重叠</text>
  <text x="420" y="54" text-anchor="middle" font-size="12" fill="#757575">NVLink 上行汇聚 → RDMA 下行回源</text>

  <defs>
    <marker id="hcN" markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L9,3.5 L0,7" fill="#2E7D32"/>
    </marker>
    <marker id="hcR" markerWidth="9" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L9,3.5 L0,7" fill="#E65100"/>
    </marker>
  </defs>

  <!-- Target machine (expert side) -->
  <rect x="30" y="75" width="360" height="240" fill="#FCE4EC" stroke="#C62828" stroke-width="2" rx="10"/>
  <text x="210" y="98" text-anchor="middle" font-size="13" font-weight="bold" fill="#B71C1C">机器 1 — expert 算完的一侧</text>

  <rect x="50" y="115" width="150" height="80" fill="#E8F5E9" stroke="#2E7D32" rx="6"/>
  <text x="125" y="135" text-anchor="middle" font-size="11" font-weight="bold" fill="#1B5E20">expert peer</text>
  <text x="125" y="153" text-anchor="middle" font-size="9" fill="#546E7A">Scale-up warp</text>
  <text x="125" y="168" text-anchor="middle" font-size="9" fill="#546E7A">链表派发</text>
  <text x="125" y="183" text-anchor="middle" font-size="9" fill="#546E7A">st_release_sys tail</text>

  <rect x="230" y="115" width="140" height="80" fill="#FFF8E1" stroke="#F9A825" stroke-width="1.5" rx="6"/>
  <text x="300" y="135" text-anchor="middle" font-size="11" font-weight="bold" fill="#F57F17">中转 rank</text>
  <text x="300" y="153" text-anchor="middle" font-size="9" fill="#546E7A">scaleup_buffer</text>
  <text x="300" y="168" text-anchor="middle" font-size="9" fill="#546E7A">↓ Forward warp</text>
  <text x="300" y="183" text-anchor="middle" font-size="9" fill="#546E7A">reduce + 打包</text>

  <!-- NVLink hop up -->
  <line x1="200" y1="155" x2="230" y2="155" stroke="#2E7D32" stroke-width="3" marker-end="url(#hcN)"/>
  <text x="215" y="145" text-anchor="middle" font-size="9" font-weight="bold" fill="#2E7D32">NVLink</text>

  <rect x="50" y="210" width="320" height="90" fill="#FFFDE7" stroke="#F9A825" stroke-width="1" rx="6"/>
  <text x="65" y="230" font-size="11" font-weight="bold" fill="#F57F17">⚡ compute-comm overlap 核心</text>
  <text x="65" y="248" font-size="10" fill="#795548">combine_reduce 的"等 SMEM release"回调里</text>
  <text x="65" y="264" font-size="10" fill="#795548">插入 flush_last_tma_and_issue_rdma()</text>
  <text x="65" y="280" font-size="10" fill="#795548">→ 本轮 reduce 计算 与 上轮 RDMA 发送 重叠</text>
  <text x="65" y="295" font-size="10" fill="#795548">双缓冲：last_send_buffer 记住上一份待发</text>

  <!-- Source machine -->
  <rect x="450" y="75" width="360" height="240" fill="#E3F2FD" stroke="#1565C0" stroke-width="2" rx="10"/>
  <text x="630" y="98" text-anchor="middle" font-size="13" font-weight="bold" fill="#0D47A1">机器 0 — token 的源头</text>
  <rect x="470" y="120" width="320" height="55" fill="white" stroke="#90A4AE" rx="6"/>
  <text x="630" y="140" text-anchor="middle" font-size="10" font-weight="bold" fill="#455A64">scaleout_recv_buffer</text>
  <text x="630" y="157" text-anchor="middle" font-size="9" fill="#757575">RDMA 回来的 expert 结果落地</text>
  <text x="630" y="170" text-anchor="middle" font-size="9" fill="#757575">scaleout_signaled_tail = RDMA 完成数</text>
  <rect x="470" y="190" width="320" height="55" fill="#E0F2F1" stroke="#00695C" rx="6"/>
  <text x="630" y="212" text-anchor="middle" font-size="11" font-weight="bold" fill="#004D40">Combine Epilogue</text>
  <text x="630" y="229" text-anchor="middle" font-size="9" fill="#546E7A">最终加权求和 + bias → combined_x</text>

  <!-- RDMA hop down -->
  <line x1="370" y1="250" x2="470" y2="215" stroke="#E65100" stroke-width="3" stroke-dasharray="8,4" marker-end="url(#hcR)"/>
  <text x="420" y="248" text-anchor="middle" font-size="10" font-weight="bold" fill="#E65100">RDMA 回源</text>

  <!-- Sync counters -->
  <rect x="30" y="330" width="780" height="60" fill="#F3E5F5" stroke="#7B1FA2" stroke-width="1" rx="6"/>
  <text x="45" y="352" font-size="12" font-weight="bold" fill="#4A148C">两个 tail 计数器 = 生产者-消费者同步</text>
  <text x="45" y="372" font-size="11" fill="#37474F">channel_scaleup_tail = NVLink 送达数（Scale-up warp 生产 → Forward warp 消费，每 3 轮批量推 st_release_sys）</text>
  <text x="45" y="386" font-size="10" fill="#757575">scaleout_signaled_tail = RDMA 完成数（Forward warp red_add → 源机确认全部到齐）</text>

  <!-- TPU comparison -->
  <rect x="30" y="400" width="780" height="90" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1" rx="6"/>
  <text x="45" y="422" font-size="12" font-weight="bold" fill="#283593">🔗 TPU 对比</text>
  <text x="45" y="442" font-size="11" fill="#37474F">TPU 的 combine = reduce-scatter，跨 slice 用 hierarchical（先 ICI 机内 reduce，再 DCN 跨机），</text>
  <text x="45" y="460" font-size="11" fill="#37474F">和这里"NVLink 先汇聚→RDMA 回源"完全同构。compute-comm overlap 在 TPU 由 XLA latency-hiding scheduler 自动做，</text>
  <text x="45" y="478" font-size="11" fill="#37474F">GPU 靠手动把 flush_last 塞进 reduce 回调——同一个思想（算的时候顺手把上一份发出去），一个自动一个手动。</text>
</svg>
```

</details>

> **类比**: 报销流程反着走。dispatch 是"公司总部把预算下发到各分部各员工"（下行两跳）。combine 是"各员工的报销单先在分部汇总（NVLink 上行到中转 rank），分部再统一寄回总部财务（RDMA 下行回源）"。中转 rank 就是分部财务，先把本部门的账合了再往上寄，省得每个员工单独寄一次。

### 8.6 生产者-消费者：channel_scaleup_tail + 批量推送

Scale-up warp 和 Forward warp 之间靠 `channel_scaleup_tail` 做同步——这是个经典的**生产者-消费者队列**：

- **生产者** = 本 rank 的 Scale-up warp，把 token NVLink 写到 peer 的 scaleup_buffer 后，tail += 1
- **消费者** = 远端 peer 的 Forward warp，用 `stored_num_tokens_recv < cached_tail` 判断"有没有新数据"

tail 是个 **cumulative count**（累计值），不是 flag。Forward warp 缓存上次读到的 tail，只要本地消费数 < 缓存 tail，就知道有活干。

一个重要优化：`st_release_sys` 走 system scope fence，**开销大**。所以 Scale-up warp 不是每发一个 token 就推一次 tail，而是攒够 `kNumScaleupUpdateInterval=3` 个才批量推一次。

> **类比**: 餐厅后厨的取餐叫号。厨师（生产者）做好一道菜就把号码牌往前推（tail++），服务员（消费者）盯着号码牌，号大了就来端。但"推号码牌"这个动作本身有成本（要广播到全场），所以厨师攒够 3 道菜再推一次号，而不是每道都吼一嗓子。
>
> **寄存器再分配的细节**: Scale-up warp 只做"索引 + TMA store"，寄存器压力小（分 40 个）；Forward warp 要做多副本 reduce，压力大（分 216 个）。DeepEPv2 用 `warpgroup_reg_dealloc<40>` / `warpgroup_reg_alloc<216>` 在两组 warp 间**重新分配寄存器**，让 reduce 侧吞吐更高。这是 Hopper warp-group 级的资源调度，TPU 上没有对应物——TPU 的 VMEM/寄存器分配由 XLA 编译期静态决定。

### 8.7 compute-comm overlap：flush_last 双缓冲

这是整个 Hybrid Combine 隐藏 RDMA 延迟的关键，也是最精妙的一处。

Forward warp 每处理一个 token 要做两件事：**本地 reduce**（算，占 SM）+ **RDMA 发回源**（发，占网卡）。如果串行做，网卡发数据时 SM 干等。

DeepEPv2 的解法是**双缓冲 + 回调注入**：

```
combine_reduce(..., 回调 = flush_last_tma_and_issue_rdma)
                          ↑ 在"等 SMEM buffer release"的空隙里，
                            触发上一轮 token 的 RDMA put
```

`combine_reduce` 内部有个"等 SMEM release"的等待点（等上一份数据被消费才能复用 buffer）。DeepEPv2 把 `flush_last_tma_and_issue_rdma()` 塞进这个等待回调——**这一轮在 reduce 计算的时候，上一轮的 RDMA 正在网卡上飞**。用 `last_send_token_buffer_ptr` 等变量记住上一份待发数据，形成一条 reduce ‖ RDMA 的流水线。

> **类比**: 洗衣店的洗+烘流水线。你不会等第一桶衣服烘干了才去洗第二桶——而是第二桶在洗的时候（reduce），第一桶正在烘（RDMA）。`flush_last` 就是"启动上一桶的烘干机"这个动作，恰好塞在"往洗衣机放第二桶"的空隙里。
>
> **TPU 对比**: TPU 的 XLA latency-hiding scheduler 会自动分析数据依赖，把 collective 和 compute 重叠——你写朴素的 `reduce_scatter` 后接矩阵乘，编译器自己插好 overlap。GPU 这里是手动把通信塞进计算的等待缝隙，思想完全一致，一个编译器自动、一个程序员手动。这正是 §五讲的 "XLA 自动 vs GPU 手动 event 链" 的 kernel 内部版本。

### 8.8 小结：Hybrid 比 Direct 多付出了什么

| 维度 | Direct | Hybrid | 多出的代价 |
|---|---|---|---|
| 拓扑 | 一维 flat | 二维 scaleout × scaleup | 手动 rank 编址 + 逆变换 |
| 数据流 | 1 跳 NVLink | 2 跳 RDMA→NVLink | 中转仓落地 + 二次分发 |
| Warp 角色 | Notify + Dispatch | + Scaleout-Send + Forward | Forward warp 吃 SM 资源 |
| 同步 | 单层 barrier | scaleup_tail + scaleout_signaled_tail 双计数 | 两级生产者-消费者 |
| Combine reduce | 就地 | NVLink 上行汇聚 + RDMA 下行 | hierarchical reduce |
| overlap | event 链 | flush_last 塞进 reduce 回调 | 手动双缓冲 |

> **一句话**: Hybrid 就是把 Direct 的每个环节都"竖切一刀"分成机内（NVLink）+ 机间（RDMA）两层，中间加一个 Forward warp 做中转。多出来的所有复杂度，本质都是"跨机这一跳不能直达目标卡，必须落地再转发"逼出来的。这跟 TPU 的 DCN → ICI hierarchical collective 是同一个物理约束下的两种工程答案——TPU 交给编译器，GPU 交给 warp。

---

## 九、与 TPU/GPU 工作的关联速查

| DeepEP 设计 | GPU 上的体现 | TPU 对应 | 对我们的启发 |
|---|---|---|---|
| Symmetric Memory | NVLink LSA + ncclMemAlloc | ICI 直连 | TPU compiler 自动处理，GPU 需要手动管理 |
| ScaleUP Barrier | NVLink 原子操作 | ICI barrier | 都是 ~1μs 级 |
| ScaleOut Barrier | RDMA Gin signal | DCN AllReduce | DCN 延迟 ~5-10μs，是主要瓶颈 |
| Hybrid 2-SM | SM0 NVLink + SM1 RDMA 并行 | ICI + DCN 并行 | 两层通信必须并行化 |
| AGRS 零 SM | DMA copy engine | ICI hardware collective | 理念相同：通信不占计算资源 |
| Forward Warp | RDMA → NVLink 中转 | DCN → ICI 中转 | 两级互联的固有代价 |
| Direct vs Hybrid | 单机 vs 多机 | 单 host vs 多 host | 相同的 sharding 选择空间 |
| EPHandle 缓存 | 推理 continuous batching | prefill/decode 路由复用 | 路由稳定时省 CPU 开销 |
| PDL epilogue | Hopper kernel 依赖 | XLA pipelining | 更细粒度的异步控制 |
| Event overlap | 手动 event 链 | XLA async collective | GPU 手动 vs TPU 自动 |
| Multiple reduction | 大 MoE combine 内存 | 大 MoE reduce 精度 | 512+ expert 必须边收边 reduce |
| Warp specialization | 4 种 warp 角色并行 | XLA 静态编译 | GPU 手动分工 vs TPU 编译器自动 |
| TMA 异步搬运 | Tensor Memory Accelerator | VPU DMA engine | 专用硬件搬数据，释放计算单元 |
| encode_decode_positive | 符号位当 ready flag | ICI 完成信号 | 没有硬件信号时的软件 trick |
| Bias 融合 reduce | 嵌入 reduce 内循环 | XLA 算子融合 | 手动 vs 自动 fusion |
| Deterministic prologue | 4 轮 grid sync 前缀和 | XLA 编译期确定性 | 可复现性的工程代价 |
| 二维 rank 编址 | scaleout × scaleup 手算 | jax.sharding.Mesh 声明 | GPU 手动映射 vs TPU 声明式 |
| 两跳数据流 | RDMA→recv→NVLink→scaleup | DCN→入口芯片→ICI→目标 | 跨机不能直达目标卡的固有约束 |
| Forward warp (Hybrid) | RDMA 落地后 NVLink 转发 | DCN→ICI relay | GPU 吃 SM 做中转 vs TPU 编译器生成 |
| ncclTeamTag 分维 | Lsa(NVLink)/Rail(RDMA) | ICI/DCN axis | 两维通信显式区分 |
| channel_scaleup_tail | 生产者-消费者累计计数 | XLA 数据依赖调度 | 每 3 轮批量推省 fence 开销 |
| warpgroup 寄存器再分配 | dealloc 40 / alloc 216 | VMEM 编译期静态分配 | Hopper 动态资源调度，TPU 无对应 |
| flush_last 双缓冲 overlap | reduce 回调塞 RDMA | XLA latency-hiding | kernel 内 compute-comm 重叠，手动 vs 自动 |
| Hierarchical combine | NVLink 汇聚+RDMA 回源 | ICI reduce+DCN reduce-scatter | 两层 reduce 同构 |
