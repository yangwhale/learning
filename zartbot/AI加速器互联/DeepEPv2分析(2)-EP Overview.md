# DeepEPv2分析(2)-EP Overview

> 作者: zartbot  
> 日期: 2026年5月4日 11:27  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498240&idx=1&sn=7ad4bc64bf889e6f58dfd49bfe1133fe&chksm=f995eac2cee263d4dbe35c2c4517a71a07d724def54cb77ceded843e696e11d5ed26d2dadecd#rd

---

### TL;DR

这是DeepEPv2分析的第二篇, 详细分析EP并行的实现. 然后第三篇将介绍Direct Dispatch/Combine kernel, 第四篇介绍Hybrid Dispatch/combine kernel本文目录如下:

```
1. 初始化及BufferLayout1.1 TokenLayout1.2 BufferLayout1.3 dispatch buffer1.4 combine buffer2. Dispatch/Combine overview2.1 launch流程2.1.1 dispatch 参数解析2.1.2 dispatch launch流程2.1.3 combine 参数解析2.1.4 combine launch流程2.2 EPHandle2.3 Overlap和Event控制
```

## 1. 初始化及BufferLayout

初始化配置如下:

```
deep_ep.ElasticBuffer(    group,                                      # 来自 Pytorch 的 proccessGroup, 用于获得 ncclComm_t handle    num_max_tokens_per_rank=args.num_tokens,    # 每 rank token 上界    hidden=args.hidden,                         # Hidden dim    deterministic=args.deterministic,           # 用于RL等场景的确定性推理    allow_hybrid_mode=args.allow_hybrid_mode,   # 支持 RDMA+NVLink     allow_multiple_reduction=args.allow_multiple_reduction,    prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),    sl_idx=args.sl_idx,                         # RDMA 服务等级    num_allocated_qps=max(args.num_allocated_qps, args.num_qps),    explicitly_destroy=True,                    # 手动 destroy    num_gpu_timeout_secs=args.num_gpu_timeout_secs,    num_cpu_timeout_secs=args.num_cpu_timeout_secs)
```

`num_bytes` 参数未显式指定时调 `_C.calculate_elastic_buffer_size(...)` 由 C++ 根据 `(num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch, allow_hybrid_mode, allow_multiple_reduction)` 求最坏情形的 `buffer` 段字节数.

```
static int64_t calculate_buffer_size(    const int64_t& nccl_comm, const int& num_max_tokens_per_rank,    const int& hidden, int num_topk,    const bool& use_fp8_dispatch, const bool& allow_hybrid_mode,    const bool& allow_multiple_reduction){    EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);    EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);     // (1)    num_topk = (num_topk == 0 ? 32 : num_topk);                                // (2)    // (3) Topology discovery    auto [num_rdma_ranks,      num_nvl_ranks]      = nccl::get_physical_domain_size(nccl_comm);    auto [num_scaleout_ranks,  num_scaleup_ranks]  = nccl::get_logical_domain_size(nccl_comm, allow_hybrid_mode);    const auto is_scaleup_nvlink = (num_scaleup_ranks == num_nvl_ranks);    // (4) Dispatch path    const auto elem_size    = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);    const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0;    const auto num_dispatch_bytes = get_dispatch_buffer_size(        num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,        num_scaleout_ranks, num_scaleup_ranks, is_scaleup_nvlink);    // (5) Combine path    const auto num_combine_bytes = get_combine_buffer_size(        num_max_tokens_per_rank, hidden, num_topk,        num_scaleout_ranks, num_scaleup_ranks, is_scaleup_nvlink,        allow_multiple_reduction);    // (6) Return the max (同一块 buffer 在不同阶段复用)    return std::max(num_dispatch_bytes, num_combine_bytes);}
```

### 1.1 TokenLayout

在计算`get_dispatch_buffer_size`和`get_combine_buffer_size`之前, 首先我们需要分析Token在Buffer中的Layout, 在`deep_ep/include/deep_ep/common/layout.cuh`中定义, 它的结构包括一个token hidden-dim, 如果是FP8还需要ScalingFactor(sf), 最后还包含一个metadata, 如果需要使用TMA Mbarrier,  `<kWithMBarrier=true>` 在每 token 末尾追加 `ptx::mbarrier`,用于 TMA shared-memory 版本的 arrival 同步.

```
    __forceinline__ __device__ __host__    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,                const int& num_topk, const bool& with_metadata, void* base = nullptr) :        num_hidden_bytes(num_hidden_bytes),        num_sf_bytes(num_sf_bytes),        // Metadata includes: top-k indices, weight and source rank/token index        with_metadata(with_metadata),        num_topk(num_topk),        num_metadata_bytes(num_topk * (sizeof(int) + sizeof(float)) +     // topk_idx + topk_weight                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0)),   // src_global_idx + linked_list        base(base) {        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);    }
```

单个token的总字节由`get_num_bytes`计算

```
//    aligned(num_hidden_bytes, 16)      ← hidden data (bf16 或 fp8)//  + aligned(num_sf_bytes,     16)      ← FP8 scale factor(bf16 时为 0)//  + aligned(num_metadata_bytes,16)     ← topk_idx + topk_weights [+ src_rank/token_idx + linked_list]//  + aligned(kWithMBarrier ? sizeof(mbarrier) : 0, 16)    template <bool kWithMBarrier, typename dtype_t = int>    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {        const auto num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +                               math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +                               math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes) +                               math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, ptx::kNumTMAAlignBytes);        return static_cast<dtype_t>(num_bytes);    }
```

然后定义了一些getter函数:

![图片](assets/b01754b6712d.png)

### 1.2 BufferLayout

BufferLayout中会依赖WorkspaceLayout中的一些区域

字段

用途

访问函数
`notify_reduction_workspace`
Direct/Hybrid notify 阶段跨 SM 做 64-bit red_add 汇总 `(sm_count << 32| value)`
`get_notify_reduction_workspace_ptr()``scaleout_rank_expert_count`
Hybrid: SM0 对所有 SM 规约后写入, 再经 RDMA put 到对端
`get_scaleout_rank_expert_count_ptr<local>()``scaleout_rank_count`
 / `scaleout_expert_count`

Hybrid: scale-out 的 per-rank / per-expert 计数(local=本地发送; remote=接收侧)
`get_scaleout_rank_count_ptr<local>(rank_idx, idx)``scaleup_rank_count`
 / `scaleup_expert_count`

Hybrid: scale-up LSA 层的计数
`get_scaleup_rank_count_ptr<local>(idx)``scaleup_rank_expert_count`
最终本 rank 本地的 scaleup_rank + per-expert 计数
`get_scaleup_rank_expert_count_ptr<local>()``scaleup_atomic_sender_counter`
Hybrid dispatch: 非 cached 路径下 forward warp 给 scale-up 分 slot 的 atomic 计数器
`get_scaleup_atomic_sender_counter()``scaleout_channel_signaled_tail`
Hybrid dispatch: scaleout warp → forward warp 发出 `<finish, tail>` 信号；combine 阶段复用做全局 all-to-all 完成信号
`get_scaleout_channel_signaled_tail_ptr(channel, rank)``channel_scaleup_tail`
Hybrid dispatch forward warp → combine: per-channel × per-scaleup-rank 推进的 tail(用于线性链表收尾)；combine 阶段 scaleup warp 增发
`get_channel_scaleup_tail_ptr(channel, rank)`
Barrier signals
`comm::gpu_barrier`
 每个 tag 使用一组 signal 条目

—

然后在Buffer段以 TokenLayout 为单元做二维(rank, token)寻址: 总容量 = num_ranks × num_max_tokens_per_rank × per_token_bytes, 并且也定义了一些getter函数:

`get_rank_buffer(rank_idx)` —— 得到第 r 个 peer 的整片 `[num_max_tokens]`;

`get_token_buffer(token_idx)` —— 得到某 token 的 TokenLayout 视图(可寻址各字段);

`get_channel_buffer<kNumTokensPerChannel>(channel_idx)` —— 把 rank 方向切成 channel(per-SM 的 token 窗口)

对于Buffer分区, 在创建时采用`num_buffer_bytes = max(dispatch_sz, combine_sz)`, 因此整个Buffer区域会进行时分复用, T1时刻 dispatch执行时采用dispatch layout, T2时刻Combine执行时采用 combine layout

### 1.3 dispatch buffer

这个时候需要考虑仅有ScaleUP NVLink的情况(Direct模式)和ScaleUP NVLink + ScaleOut RDMA(Hybrid模式)两种情况.记:

T = per_token_bytes(dispatch_token, kWithMBarrier=false)

N = num_max_tokens_per_rank

模式

条件

容量计算

Direct (NVLink Only)
`num_scaleout==1 ∧ is_scaleup_nvlink``num_ranks·N·T`
Direct (没有NVLink)
`num_scaleout==1 ∧ ¬is_scaleup_nvlink``(num_ranks + 1)·N·T`
Hybrid (NVLink+RDMA)
`num_scaleout > 1``T·[num_scaleup·num_scaleout·N + N + num_scaleout·(N + kMaxCh)]`

```
static int64_t get_dispatch_buffer_size(    int num_max_tokens_per_rank, int hidden,    int num_sf_packs, int num_topk, int elem_size,    int num_scaleout_ranks, int num_scaleup_ranks,    bool is_scaleup_nvlink){    const auto num_ranks    = num_scaleup_ranks * num_scaleout_ranks;        // 总 rank 数    const auto token_layout = get_dispatch_token_layout(                     // (kernels/elastic/dispatch.hpp#L209)        hidden, elem_size, num_sf_packs, num_topk);    //   = TokenLayout(hidden*elem_size, num_sf_packs*sf_pack_t,    //                 num_topk, with_metadata=TRUE)    //     ← dispatch token 必带 (topk_idx, topk_weight, src_global_idx, linked_list)    if (num_scaleout_ranks == 1) {            // ─── Direct 分支 ───        send_buffer_layout = BufferLayout<false>(            token_layout,            is_scaleup_nvlink ? 0 : 1,         // 0 = 零拷贝;否则本地 staging 一份            num_max_tokens_per_rank);        recv_buffer_layout = BufferLayout<false>(            token_layout,            num_ranks,                         // 为每个 peer 分一行            num_max_tokens_per_rank);        return send.get_num_bytes() + recv.get_num_bytes();    } else {                                  // ─── Hybrid 分支 ───        scaleup_recv = BufferLayout<false>(            token_layout,            num_scaleup_ranks,            num_scaleout_ranks * num_max_tokens_per_rank);          scaleout_send = BufferLayout<false>(            token_layout, 1, num_max_tokens_per_rank);        scaleout_recv = BufferLayout<false>(            token_layout, num_scaleout_ranks,            num_max_tokens_per_rank + kNumMaxChannels);     // +kNumMaxChannels 冗余        return scaleup_recv + scaleout_send + scaleout_recv;    }}
```

Direct 和 Hybrid 模式的详细区别
仔细查看Direct模式的代码, 你会发现虽然它的执行判断条件是`num_scaleout==1`, 但是它依旧有RDMA的通信能力, 这是什么原因呢? 既然scaleout的rank只有一个, 那为何还需要使用ScaleOut RDMA通信?

我们查询代码可以看到, 用户在构造 `ElasticBuffer` 时传入 `allow_hybrid_mode`. C++在后端把物理拓扑`(num_rdma_ranks, num_nvl_ranks)`** 映射为逻辑域 `(num_scaleout_ranks, num_scaleup_ranks)`时有一个特殊的处理

```
// csrc/kernels/backend/nccl.cu  L58-L62std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm,                                             const bool& allow_hybrid_mode) {    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);    return {allow_hybrid_mode ? num_rdma_ranks : 1,            allow_hybrid_mode ? num_nvl_ranks  : num_rdma_ranks * num_nvl_ranks};}
```

当`allow_hybrid_mode == false`时, num_scaleout_ranks=1, 但是num_scaleup_ranks = num_rdma_ranks * num_nvl_ranks, 此时依旧会使用 direct 模式.

总结如下:

场景
`allow_hybrid_mode`
物理拓扑
`num_scaleout_ranks`
分支

单机 8 卡 NVLink

True / False
`num_rdma=1, num_nvl=8``1`**Direct**
多机，但用户禁用 hybrid (RDMA only)
**False**`num_rdma=N, num_nvl=8``1`**Direct**
多机正常 hybrid（典型多机 MoE）

True
`num_rdma=N>1, num_nvl=8``N`**Hybrid**

Direct模式 Buffer Layout如下, recv 用对称内存(peer 通过 `get_sym_ptr(recv_buffer.token_ptr, dst_rank)` 直写此 rank 的 `recv[rank][slot]`).

![图片](assets/1d20d70f47e6.png)

Hybrid模式 Buffer Layout如下:

![图片](assets/aa7aa9d6efc8.png)

### 1.4 combine buffer

Combine阶段的TokenLayout和dispatch有一些差异, 首先Combine是BF16, 因此不需要ScalingFactor, 另外它也不需要其它的metadata, 仅需要topk_weight. 另外还有一个影响因素`allow_multiple_reduction=false`时, 需要推迟到 epilogue 做单次规约以保最高精度,代价是 send 端要保留每 (token, topk) 副本.

```
static int64_t get_combine_buffer_size(    int num_max_tokens_per_rank, int hidden, int num_topk,    int num_scaleout_ranks, int num_scaleup_ranks,    bool is_scaleup_nvlink, bool allow_multiple_reduction){    const auto num_ranks    = num_scaleup_ranks * num_scaleout_ranks;    const auto token_layout = get_combine_token_layout(                  // (combine.hpp#L109)        hidden, sizeof(nv_bfloat16), num_topk);    //   = TokenLayout(hidden*bf16, 0, num_topk, with_metadata=FALSE)    //     ← combine 反向只传 hidden + topk_weight,不需要 src 索引,无 FP8 SF    if (num_scaleout_ranks == 1) {           // ─── Direct combine ───        num_tokens_in_layout = allow_multiple_reduction                               ? min(num_ranks, num_topk)                               : num_topk;                                       send = BufferLayout<false>(            token_layout,            is_scaleup_nvlink ? 0 : num_ranks,                                       num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));            //  单次规约 ⇒ 最坏 do_expand,每 rank 放 topk 份副本        recv = BufferLayout<false>(            token_layout, num_tokens_in_layout, num_max_tokens_per_rank);        return send + recv;    } else {                                 // ─── Hybrid combine ───        n_up  = allow_multiple_reduction ? min(num_scaleup_ranks,  num_topk) : num_topk;        n_out = allow_multiple_reduction ? min(num_scaleout_ranks, num_topk) : num_topk;        scaleup_recv  = BufferLayout<false>(token_layout, n_up,                            num_scaleout_ranks * num_max_tokens_per_rank);        scaleout_recv = BufferLayout<false>(token_layout, n_out,                            num_max_tokens_per_rank);        scaleout_send = BufferLayout<false>(token_layout,                            allow_multiple_reduction ? 1 : num_topk,                            num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));        return scaleup_recv + scaleout_send + scaleout_recv;    }}
```

记:

T' = per_token_bytes(combine_token): 它包含BF16 token hidden-state 和 topk-weight.

M  =  allow_multiple_reduction ? 1 : num_topk

Direct模式 Buffer Layout如下:

![图片](assets/2dfbcc4abca5.png)

Hybrid模式 Buffer Layout如下:

![图片](assets/cfa935d60b14.png)

## 2. Dispatch/Combine overview

### 2.1 launch流程

在python接口`elastic.py`中调用`ElasticBuffer.dispatch()/combine()`. 我们首先来分析它们的输入参数和返回结果, 然后再进一步分析它的运行过程.

#### 2.1.1 dispatch 参数解析

dispatch 函数的参数部分如下:

```
    def dispatch(        self,        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],        topk_idx: Optional[torch.Tensor] = None,        topk_weights: Optional[torch.Tensor] = None,        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,        num_experts: Optional[int] = None,        num_max_tokens_per_rank: Optional[int] = None,        expert_alignment: Optional[int] = None,        num_sms: int = 0,        num_qps: int = 0,        previous_event: Optional[EventHandle] = None,        previous_event_before_epilogue: Optional[EventHandle] = None,        async_with_compute_stream: bool = False,        allocate_on_comm_stream: bool = False,        handle: Optional[EPHandle] = None,        do_handle_copy: bool = True,        do_cpu_sync: Optional[bool] = None,        do_expand: bool = False,        use_tma_aligned_col_major_sf: bool = False,    ) -> Tuple[        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],        Optional[torch.Tensor],        Optional[torch.Tensor],        EPHandle,        EventOverlap,    ]:
```

由于输入输出的参数数量很庞大, 我们将首先对这些参数解析
输入张量类参数

`x : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`
待分发的 token 特征张量, 是 dispatch 的主数据载荷, 支持两种形态:

**BF16 模式**: `torch.Tensor`, 形状 `[num_tokens, hidden]`, dtype = `torch.bfloat16`.

**FP8 模式**: `(fp8_tensor, scale_factors)` 二元组, 其中 `fp8_tensor` 为 `[num_tokens, hidden]`、`torch.float8_e4m3fn`, `scale_factors` 为对应的缩放因子.
`topk_idx: Optional[torch.Tensor] = None`
每个 token 选中的专家索引, 形状 `[num_tokens, num_topk]`, dtype 通常为 `torch.int64`(即 `deep_ep.topk_idx_t`), 值为 `-1` 表示未选择任何专家.
`topk_weights: Optional[torch.Tensor] = None`
每个 token 分发到各选中专家的权重, 形状 `[num_tokens, num_topk]`, dtype = `torch.float32`. combine 阶段会以此做加权归并.
`cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None`
形状 `[num_local_experts]`、dtype = `torch.int` 的累计计数张量, 用于在线 EP 负载均衡监控. dispatch kernel 会把每个本地 expert 实际收到的 token 数累加进去, 供外部采样使用.

layout/对齐配置类参数

`num_experts: Optional[int] = None`
集群中的专家总数. 若传入 `handle` 则会从 `handle.num_experts` 推导, 且若显式传值必须与 handle 一致.
`num_max_tokens_per_rank: Optional[int] = None`
每个 rank 在本次 dispatch 中最多发送的 token 数(**所有 rank 必须一致**), 用于上界分配接收缓冲区.
`expert_alignment: Optional[int] = None`
每个本地 expert 接收到的 token 数量向上对齐到该值(便于后续 GEMM 的 tile 对齐), 默认 `1`(无对齐). 若传入 `handle`, 该值必须与 handle 一致.

并行资源配置类参数

`num_sms: int = 0`
dispatch kernel 使用的 SM 数量. `0` 表示通过`get_theoretical_num_sms`基于 `num_experts` 和 `num_topk` 的带宽模型估算
`num_qps: int = 0`
RDMA QP(Queue Pair)数量, 直接决定跨节点并发链路数. `0` 表示自动推导

事件 / 流调度类参数

`previous_event: Optional[EventHandle] = None`
kernel 启动前需要等待的 CUDA event. 用于与计算流 overlap 的依赖链. 若此项被设置, 则 `allocate_on_comm_stream` 必须为 `True`(避免张量所属流与 event 所在流冲突).
`previous_event_before_epilogue: Optional[EventHandle] = None`
kernel 的 copy epilogue(后处理阶段)启动前需等待的 event. 粒度比 `previous_event` 更细, 表示主通信阶段可以先开始, 只在 epilogue 之前再同步一次, 以获取更激进的 overlap.
`async_with_compute_stream: bool = False`
若为 `True`, 当前(计算)流不等待通信 kernel 完成, dispatch 立即返回, 调用方需自己消费返回的 `event` 做显式同步. 这是实现通信-计算重叠的关键开关.
`allocate_on_comm_stream: bool = False`
控制本次 dispatch 分配的所有输出张量是否挂在通信流上. 开启后与 `previous_event` 配合, 可避免 allocator 在错误流上释放内存.

布局模式类参数

`do_expand: bool = False`
是否使用**扩展布局**(expanding layout): 每个 token 为其选中的每个 top-k expert 各占一个独立 slot(一 token 占 `num_topk` 行). 该值同时被写入 `EPHandle.do_expand`, 决定 `psum_num_recv_tokens_per_expert` 的语义.

`False`(默认)= 紧凑布局

`True` = 扩展布局, 下游 GEMM 不再需要 scatter.
`use_tma_aligned_col_major_sf: bool = False`
FP8 模式下, 是否将 scale-factor 按 TMA 对齐的列主序(col-major)布局输出, 避免后续额外的转置/重排. 仅当 `x` 是 FP8 元组时才有意义.

其它参数

`handle: Optional[EPHandle] = None`
可选的缓存通信句柄, 来自上一次 dispatch 的返回值. 传入后会复用其中的 7 个 metadata 张量与 layout 参数, **跳过 CPU 侧 layout 重算**, 显著降低小 batch 场景的开销. 使用约束:

`topk_idx`、`topk_weights` 必须为 `None`；

`do_cpu_sync` 必须为 `None` 或 `False`(缓存路径没有新鲜计数)；

显式传入的 `num_experts / expert_alignment / num_max_tokens_per_rank` 必须与 handle 完全一致.
`do_handle_copy: bool = True`
返回的 handle 中是否 clone 一份 `topk_idx`. 默认 `True`, 防止用户后续 in-place 修改影响 combine 阶段的反向路由. 设为 `False` 可节省内存, 但需要保证后续不改 `topk_idx`.
`do_cpu_sync: Optional[bool] = None`
是否与 CPU 做一次同步以拿到精确的接收 token 计数(而不是上界).

`None` 的默认行为: 未用 handle 时为 `True`(需要精确值)；用 handle 时强制 `False`.

返回值为元组 `(recv_x, recv_topk_idx, recv_topk_weights, handle, event)`:

`recv_x`: 按接收侧 layout 排好的 token(与输入同类型, FP8 时为 `(tensor, sf)` 元组)；

`recv_topk_idx`: 接收侧 expert 索引；

`recv_topk_weights`: 接收侧权重, 若入参 `topk_weights` 为空则返回 `None`；

`handle`: 新的 `EPHandle`(或入参传入的那个), 供 combine / 下次 dispatch 复用；

`event`: 封装为 `EventOverlap`, 仅在 `async_with_compute_stream=True` 时有实际等待语义.

#### 2.1.2 dispatch launch流程

dispatch主要执行如下几个步骤:

自动推导 num_sms / num_qps (若未给定)

FP8 模式解包 (x, scale_factors)

若传入 cached handle，则复用 num_experts/expert_alignment/num_max_tokens 与 7 个内部 metadata tensor (跳过 layout 计算)

调 C++ runtime.dispatch 启动 dispatch kernel + epilogue

用返回值构造新 EPHandle (若用户未提供)

包装 event 为 EventOverlap 以支持 overlap 语义

根据输入参数的选择, dispatch存在三种模式

变体

Python 侧标志

Kernel 模板标志

用途

普通 dispatch
`do_expand=False``kDoExpand=False`
, `kReuseSlotIndices=False`

每个 (token, rank) 一个 slot, 向下兼容

Expanded dispatch
`do_expand=True`
, 首次
`kDoExpand=True`
, `kReuseSlotIndices=False`

每个 (token, expert) 一个 slot

Cached dispatch

传入已有 `handle`
`kReuseSlotIndices=True`
复用 slot 索引, 跳过 layout 计算

`ElasticBuffer::dispatch()` 在 C++ 侧的执行流程分为以下阶段:
Phase A: 输入校验
判断 `cached_mode = cached_num_recv_tokens.has_value()`

校验 `x` 的形状 `[num_tokens, hidden]`、连续性、16B 对齐

校验 SF(scale factors)的形状和类型

校验 `topk_idx`、`topk_weights` 的形状

提取 `cumulative_local_expert_recv_stats` 指针
Phase B: 流控制 Prologue
```
const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);
```

若 `allocate_on_comm_stream`, 切换当前 stream 为 `comm_stream`(后续所有 tensor 分配在 comm_stream 上)

若有 `previous_event`, `comm_stream` 等待该事件；否则 `comm_stream` 等待 `compute_stream`
Phase C: 张量分配

张量

形状

用途
`psum_num_recv_tokens_per_expert``[num_local_experts+1]`
每个本地 expert 的接收 token 前缀和
`psum_num_recv_tokens_per_scaleup_rank``[num_scaleup_ranks]`
每个 scaleup rank 的接收 token 前缀和
`dst_buffer_slot_idx`
Direct: `[num_tokens, num_topk]`; 
Hybrid: `[num_channels, num_scaleout_ranks, max_tokens_per_channel, num_topk]`

目标 buffer 槽位索引
`deterministic_rank_count_buffer``[num_sms, num_scaleup_ranks]`
仅确定性模式使用的计数缓冲区
`token_metadata_at_forward``[num_channels, max_forwarded_tokens, 2+2*num_topk]`
Hybrid 模式的 forward 元数据
`channel_linked_list``[num_channels, max_forwarded_tokens+1, num_scaleup_ranks]`
Hybrid 模式的通道链表
`copied_topk_idx``[num_tokens, num_topk]`
topk_idx 的克隆(防止用户修改)

针对Hybrid 模式 Channel 配置:

```
num_channels_per_sm = min((smem - notify_smem) / dispatch_token_layout_bytes, 32 - kNumNotifyWarps);num_channels_per_sm = min(smem / combine_token_layout_bytes, num_channels_per_sm);num_channels_per_sm = min(num_channels_per_sm / 2, kNumMaxChannelsPerSM);num_channels = num_sms * num_channels_per_sm;
```
Phase E: 主 Dispatch Kernel Launch
```
launch_dispatch(x.data_ptr(), sf_ptr, topk_idx.data_ptr<topk_idx_t>(), ...)
```
Phase F: CPU 同步获取接收计数 (L1008-1076)
三种模式:

模式

条件

行为

cached_mode
`cached_num_recv_tokens != None`
直接复用缓存值, 无同步

do_cpu_sync
`!cached && do_cpu_sync`
忙等待 host workspace 中的 encoded 计数值, 超时抛异常

worst-case
`!cached && !do_cpu_sync`
按最大可能值分配: `num_ranks × num_max_tokens_per_rank`

CPU 同步使用 `encode_decode_positive()` 编码(区分"未写入"与"写入了 0"), 通过内存映射的 `host_workspace` 实现 GPU→CPU 的零拷贝通知.
Phase G: 输出张量分配

张量

形状
`recv_x``[num_allocated_tokens, hidden]``recv_sf``[num_allocated_tokens, num_sf_packs]`
(可选)
`recv_topk_idx``[num_allocated_tokens, num_topk]`
(非 expand 模式)
`recv_topk_weights``[num_allocated_tokens, num_topk]`
(可选)
`recv_src_metadata``[num_recv_tokens, num_topk+2]`
(源 token 索引 + 槽位)

Phase H: Copy Epilogue Kernel Launch
等待可选的 `previous_event_before_epilogue`, 然后启动 Copy Epilogue 内核.

```
stream_control_before_epilogue(previous_event_before_epilogue);launch_dispatch_copy_epilogue(buffer, workspace, ..., comm_stream);
```
Phase I: 流控制 Epilogue
若 `async_with_compute_stream`: 在 comm_stream 上记录 event 并返回, compute_stream 不等待 ,  否则: compute_stream 等待 comm_stream 完成

```
const auto event = stream_control_epilogue({...tensors...}, compute_stream, allocate_on_comm_stream, async_with_compute_stream);
```

实际的CUDA Kernel由三个部分组成
Kernel 1: Deterministic Prologue(可选)

属性

值

Runtime 类
`DispatchPrologueRuntime`
内核模板
`dispatch_deterministic_prologue_impl<grid_dim, 8, num_ranks, max_tokens, num_experts, num_topk>`
Grid
`(num_sms,)`
Block
`256`
 (8 warps × 32)

共享内存
`(2 × 8 + 1) × num_ranks × 4`
 字节

职责

确定性计算 per-rank/per-expert token 计数和目标槽位

cooperative
`true`
PDL

无

Kernel 2: Main Dispatch
Direct 模式 (`num_scaleout_ranks == 1`), 用于仅有NVLink 没有RDMA ScaleOut的场景

属性

值

Runtime 类
`DispatchRuntime`
内核模板
`dispatch_impl<is_nvlink, do_cpu_sync, reuse_slot, grid_dim, notify_warps,`
` dispatch_warps, num_ranks, hidden_bytes,`
` sf_packs, max_tokens, experts, topk, alignment, qps, timeout>`
Grid
`(num_sms,)`
Notify Warps
`4`
 (固定, cached 模式为0)

Dispatch Warps
`min((smem - notify_smem) / token_layout_bytes, 32 - notify_warps, ceil_div(512, num_sms))`
共享内存

设备最大动态共享内存

cooperative
`true`
cluster_dim
`2 - (num_sms % 2)`
(与计算 kernel 交错)

Hybrid 模式 (`num_scaleout_ranks > 1`), 用于包含NVLink ScaleUP和RDMA ScaleOut的场景

属性

值

内核模板
`hybrid_dispatch_impl<do_cpu_sync, reuse_slot, grid_dim, notify_warps,`
`scaleout_warps, forward_warps, scaleout_ranks, scaleup_ranks, hidden_bytes,`
`sf_packs, max_tokens, experts, topk, alignment, qps, timeout>`
Scaleout Warps
`num_channels_per_sm`
Forward Warps
`num_channels_per_sm`
总 Warps
`notify_warps + scaleout_warps + forward_warps`
Warp 角色分工

Warp 类型

Direct 模式

Hybrid 模式

Notify Warps (4个)

写入 rank/expert 的完成标记

同左

Dispatch Warps

读取输入 token → 写入远端 buffer

N/A

Scaleout Warps

N/A

RDMA 发送到远端 scaleout peer

Forward Warps

N/A

NVLink 转发到 scaleup peer

Launch 参数传递
```
// Direct 模式jit::launch_kernel(kernel, config,    args.x, args.sf, args.topk_idx, args.topk_weights,    args.copied_topk_idx,    args.cumulative_local_expert_recv_stats,    args.psum_num_recv_tokens_per_scaleup_rank,    args.psum_num_recv_tokens_per_expert,    args.dst_buffer_slot_idx,    args.num_tokens,    args.sf_token_stride, args.sf_hidden_stride,    args.nccl_dev_comm, args.nccl_window,    args.buffer,    args.workspace, args.mapped_host_workspace,    args.scaleup_rank_idx);
```
Kernel 3: Copy Epilogue

属性

值

Runtime 类
`DispatchCopyEpilogueRuntime`
内核模板
`dispatch_copy_epilogue_impl<do_expand, cached_mode, grid_dim, channels, num_warps, scaleout_ranks, scaleup_ranks, hidden_bytes, sf_packs, max_tokens, experts, topk>`
Grid
`(全部 SM,)`
 — 注意用的是设备全部 SM

Warps
`min(smem / token_layout_bytes, 32)`
cooperative
`false`
PDL
`true`
(Programmatic Dependent Launch, 依赖于主 kernel 完成)

职责

从 buffer 复制 recv_x、recv_sf、recv_topk_idx、recv_topk_weights、recv_src_metadata 到用户输出 tensor

#### 2.1.3 combine 参数解析

combine函数参数部分如下所示:

```
    def combine(        self,        x: torch.Tensor,        handle: EPHandle,        topk_weights: Optional[torch.Tensor] = None,        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,        num_sms: int = 0,        num_qps: int = 0,        previous_event: EventHandle = None,        previous_event_before_epilogue: Optional[EventHandle] = None,        async_with_compute_stream: bool = False,        allocate_on_comm_stream: bool = False,    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
```
输入数据类参数

`x: torch.Tensor`
待 reduce 回源 rank 的 token 特征张量.

形状: `[num_tokens, hidden]`, dtype = `torch.bfloat16`.

这里的 `num_tokens` 是 dispatch 后在本 rank 上承载的 token 数(即 dispatch 返回的 `recv_x` 的第一维), 并且已经过了 expert 的前向计算.

combine 的任务: 把这些 token 按 `handle` 中记录的反向路由路径发回原始发送 rank, 并在落地时做 top-k 加权累加.

注意: combine 仅支持 BF16, 不支持 FP8.
`handle: EPHandle`
**必传**的通信句柄, 来自 dispatch的返回值, 是 combine 的"反向路由表". 它需要从 handle 中取出以下字段直接传给 C++ runtime:

`recv_src_metadata`: 每个接收 slot 对应的源 token 索引 + 源缓冲区槽位 → 反向发送地址；

`topk_idx`: dispatch 时克隆的 top-k 索引, 决定权重如何回乘；

`psum_num_recv_tokens_per_scaleup_rank`: 按 scaleup rank 的前缀和, 划分每段 token 归属；

`token_metadata_at_forward` / `channel_linked_list`: hybrid 模式下的逐 channel 转发元数据；

`num_experts`、`num_max_tokens_per_rank`、`do_expand`: layout 参数, 必须与 dispatch 完全一致.
`topk_weights: Optional[torch.Tensor] = None`
token 的 top-k 专家权重, 形状 `[num_tokens, num_topk]`, dtype = `torch.float32`. reduce 时将各路副本按此权重做**加权求和**

若为 `None`: C++ 侧等同于所有权重为 1.

注意expand 模式下不使用, 因为 expand 模式下每个 token 已被展开成 `num_topk` 行, 权重已在下游 GEMM 中吸收, 此处再乘会重复.
`bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None`
输出的偏置项(residual/bias), 会在 reduce 结果之上再加一次. 支持 **0、1 或 2** 个 bias, 形状: `[num_combined_tokens, hidden]`, dtype = `torch.bfloat16`.

`None` → 无 bias；

单个 `torch.Tensor` → 1 个 bias；

`(b0, b1)` 二元组 → 2 个 bias(典型用于同时加残差连接和专家输出偏置).

并行资源配置类参数

`num_sms: int = 0`
combine kernel 使用的 SM 数量.

`0` 的默认行为(1125 行): 直接复用 `handle.num_sms`(即 dispatch 时用的 SM 数).

dispatch 与 combine 的 channel/block 划分必须一致, 否则 buffer slot 对应关系会错位.
`num_qps: int = 0`
RDMA QP(Queue Pair)数量, 控制跨节点并发链路数.

`0` 表示自动推导: 基于`self.get_theoretical_num_qps(num_sms)`计算

与 dispatch 不同, combine 这里 QP 数不默认复用 handle, 而是按当前 `num_sms` 重新估算——因为 QP 数仅影响带宽上限, 不影响 layout 对应关系.

事件 / 流调度类参数(通信-计算 overlap)

`previous_event: EventHandle = None`
combine kernel **启动前**要等待的 CUDA event.

典型用法: 把 expert 前向的计算流 event 传进来, 让 combine 在 GEMM 完成后再读 `x`.

约束: 若设置了 `previous_event`, 则 `allocate_on_comm_stream` 必须同时为 `True`, 否则新分配张量与 event 所属流不一致会引发 allocator 误释放.
`previous_event_before_epilogue: Optional[EventHandle] = None`
combine 的"reduce epilogue"阶段启动前要等待的 event. 粒度比 `previous_event` 更细:

主通信部分(RDMA write / NVLink send)可以提前启动；

只在最终 reduce + bias add 的 epilogue 之前再等一次 event, 用于 **最大化 overlap**, 典型用于等待另一个旁路分支(如 residual 的准备)完成.
`async_with_compute_stream: bool = False`
是否让 combine 与调用者所在的计算流**异步执行**. 这是"通信-计算重叠"的主开关.

`True`: 当前流不等待 combine 完成, 立即返回, 调用方必须自行用返回的 `event` 做同步；

`False`(默认): 退化为串行语义, 当前流会隐式等待 combine 结束.
`allocate_on_comm_stream: bool = False`
控制本次 combine 分配的输出张量(`combined_x`、`combined_topk_weights`)的 ownership 是否归通信流.

开启后张量生命周期由通信流管理, 避免在错误流上释放；

与 `previous_event` 强绑定.

最终的返回值: `Tuple[combined_x, combined_topk_weights, event]`:

`combined_x`: 最终 token, 形状 `[num_combined_tokens, hidden]`、`torch.bfloat16`. 这里 `num_combined_tokens` 等于当初 dispatch 之前该 rank 本地持有的 token 数(即回到原点).

`combined_topk_weights`: top-k 权重, 形状 `[num_combined_tokens, num_topk]`、`torch.float32`. 仅当入参 `topk_weights` 非空时有效.

`event`: 被包装为 `EventOverlap`, 仅在 `async_with_compute_stream=True` 时才有显式同步语义；否则视作 no-op.

#### 2.1.4 combine launch流程

SM/QP 推导, num_sms由于包含dispatch传来的EPHandle, 直接使用,  num_qps根据带宽估算

Bias unpack, 支持 0/1/2 个 bias tensor, 用于在 reduce 结果上叠加最终偏置.

调用C++ Runtime

返回

`ElasticBuffer::combine()` 在 C++ 侧的执行流程分为以下阶段:
Phase A: 输入校验
校验 `x` 为 `[num_tokens, hidden]`、BF16、16B 对齐

校验 `combined_topk_idx` 形状和类型

校验 `src_metadata` 形状 `[num_reduced_tokens, num_topk+2]`

校验可选的 `topk_weights`、`bias_0`、`bias_1`
Phase B: 流控制 Prologue
与 dispatch 相同, 切换 stream 和等待事件.
Phase C: Hybrid Metadata 校验
若 `num_scaleout_ranks > 1`, 校验 `token_metadata_at_forward` 和 `channel_linked_list` 的形状一致性.
Phase D: Main Combine Kernel Launch
```
const auto reduce_buffer = launch_combine(x.data_ptr(), ...);
```

→ 返回 `reduce_buffer` 指针, 指向待 reduce 的 buffer 区域.
Phase E: 输出张量分配

张量

形状
`combined_x``[num_combined_tokens, hidden]`
 BF16
`combined_topk_weights``[num_combined_tokens, num_topk]`
 FP32(可选)

Phase F: Reduce Epilogue Kernel Launch
```
stream_control_before_epilogue(previous_event_before_epilogue);launch_combine_reduce_epilogue(combined_x.data_ptr(), ..., reduce_buffer, bias_0, bias_1, ...);
```
Phase G: 流控制 Epilogue
同 dispatch 的流控制逻辑.

实际的 Combine 执行由两个 CUDA Kernel 组成:
Kernel 1: Main Combine
Direct 模式 (`num_scaleout_ranks == 1`):

属性

值

Runtime 类
`CombineRuntime`
内核模板
`combine_impl<is_nvlink, use_expand, allow_multi_reduce, grid_dim,`
` num_warps, num_ranks, hidden, max_tokens, experts, topk, qps, timeout>`
Grid
`(num_sms,)`
Warps
`min(smem / token_layout_bytes, 32)`
cooperative
`true`
cluster_dim
`2 - (num_sms % 2)`
职责

读取本地 expert 处理结果 → 写入远端 rank 的 reduce buffer

Hybrid 模式 (`num_scaleout_ranks > 1`):

属性

值

内核模板
`hybrid_combine_impl<use_expand, allow_multi_reduce, grid_dim,`
` scaleup_warps, forward_warps, scaleout_ranks, scaleup_ranks, hidden,`
` max_tokens, experts, topk, qps, timeout>`
Scaleup Warps
`num_channels / num_sms`
Forward Warps
`num_channels / num_sms`
总 Warps
`scaleup_warps + forward_warps`

返回值(`reduce_buffer` 指针):

Direct 模式: 直接返回 `buffer` 起始地址

Hybrid 模式: 跳过 scaleup buffer 区域, 返回偏移后的指针

```
// Hybrid 模式需要跳过 scaleup bufferconst auto scaleup_buffer = layout::BufferLayout<false>(    token_layout,    is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,    num_scaleout_ranks * num_max_tokens_per_rank, buffer);return scaleup_buffer.get_buffer_end_ptr();
```
Kernel 2: Reduce Epilogue

属性

值

Runtime 类
`CombineReduceEpilogueRuntime`
内核模板
`combine_reduce_epilogue_impl<use_expand, allow_multi_reduce,`
` grid_dim, num_warps, scaleout_ranks, scaleup_ranks,`
` hidden, max_tokens, experts, topk>`
Grid
`(全部 SM,)`
Warps
`min(smem / token_layout_bytes, 32)`
(上限 1024 线程避免退化)

cooperative
`false`
PDL
`true`
职责

从 reduce_buffer 读取累加结果 + 可选 bias, 写入最终 `combined_x` 和 `combined_topk_weights`

### 2.2 EPHandle

Combine阶段必须使用dispatch阶段传入的EPHandle, 它保存 dispatch 操作生成的路由元数据. 它有两个用途:

**传递给 combine**: 提供反向路由信息, 让 combine 知道如何将 expert 处理结果送回原始 rank

**缓存复用给下一次 dispatch**: 跳过 layout 重算和 CPU 同步, 节省 CPU 开销

分组

字段

作用

布局参数
`do_expand`
, `num_experts`, `expert_alignment`, `num_max_tokens_per_rank`

控制 token 在 buffer 中的排布方式

SM 配置
`num_sms`
dispatch 使用的 SM 数, combine 默认复用以保证 channel 数一致

路由索引
`topk_idx``[N, topk]`
每个 token 选中的 expert 索引(克隆后防篡改)

计数前缀和
`psum_num_recv_tokens_per_scaleup_rank``[num_scaleup_ranks]`
每个 scaleup rank 接收 token 的累计值

`psum_num_recv_tokens_per_expert``[num_local_experts]`
每个本地 expert 接收 token 的对齐前缀和

`num_recv_tokens_per_expert_list`
 (CPU list)

CPU 侧精确的 per-expert token 计数

反向路由
`recv_src_metadata``[M, topk+2]`
每个接收 token 的来源 rank 和源 index

`dst_buffer_slot_idx`
接收 token 在对称 buffer 中的槽位索引

Hybrid 模式
`token_metadata_at_forward`
, `channel_linked_list`

多节点混合模式下 forward warp 和 channel 链表

推断值
`num_recv_tokens`
= `recv_src_metadata.shape[0]`, 本 rank 实际接收的 token 总数

Step 1: dispatch 构造 EPHandle
dispatch 的 C++ kernel 返回 12 个 tensor + 1 个 event, Python 侧用其中的值构造 EPHandle:

```
handle = EPHandle(    do_expand, num_experts, expert_alignment, num_max_tokens_per_rank, num_sms,    cloned_topk_idx if do_handle_copy else topk_idx,      num_recv_tokens_per_expert_list,                    psum_num_recv_tokens_per_scaleup_rank,             psum_num_recv_tokens_per_expert,                    recv_src_metadata,                                  dst_buffer_slot_idx,                                  token_metadata_at_forward,                             channel_linked_list,                                )
```
Step 2: combine 消费 EPHandle
combine 从 handle 中读取 8 个字段传给 C++ runtime, 其中最关键的是 `recv_src_metadata`——它记录了每个接收 token 的来源 `(src_rank, src_token_idx)`, combine kernel 据此将处理结果写回正确的远端 rank.

```
self.runtime.combine(    x, topk_weights, bias_0, bias_1,    handle.recv_src_metadata,                        # 反向路由: 谁发给了我    handle.topk_idx,                                 # 加权规约的索引    handle.psum_num_recv_tokens_per_scaleup_rank,    # per-rank 前缀和    handle.token_metadata_at_forward,                # Hybrid 转发元数据    handle.channel_linked_list,                      # Hybrid 链表    handle.num_experts,                              # 专家数    handle.num_max_tokens_per_rank,                  # 缓冲区尺寸参数    num_sms,                                         # 默认用 handle.num_sms    ...,    handle.do_expand,                                # 布局模式)
```
缓存复用路径
当用户将上一次的 EPHandle 作为 `handle` 参数传入下一次 dispatch 时:

```
# 第二次 dispatch 复用第一次的 handlerecv_x2, _, _, handle2, _ = buf.dispatch(x2, handle=handle1)
```

此时 dispatch 走 cached 路径:

跳过新路由: `topk_idx` 和 `topk_weights` 必须为 `None`, 直接复用 `handle.topk_idx`

禁止 CPU 同步: `do_cpu_sync` 强制为 `False`(缓存路径无法获取 fresh count)

参数一致性校验: `num_experts / expert_alignment / num_max_tokens_per_rank` 必须与 handle 完全一致

_unpack_handle() 将 handle 拆成 7 个缓存值传给 C++:

C++ 侧 `cached_mode = true`, **主 dispatch kernel 中 notify warps 设为 0**(无需通知计数), `reuse_slot_indices = true`(复用目标槽位)

第4步unpack_handle缓存值

C++ 侧效果
`num_recv_tokens`
直接复用, 不需要 CPU 等待计数
`num_recv_tokens_per_expert_list`
跳过 expert 计数统计
`psum_*`
 张量

跳过前缀和计算
`dst_buffer_slot_idx`
跳过槽位分配

Hybrid 元数据

跳过 forward metadata 和链表构建

这种模式可以用在MoE模型训练的backward阶段, 似乎也可以构造这样的Handle来处理RL场景下的Rollout Routing Replay

### 2.3 Overlap 和 Event控制

无 Overlap 模式 (`async_with_compute_stream = false`)如下

```
Compute Stream:  ──────[等待comm完成]──────────────────>                        ↑Comm Stream:     ──[等待compute]─[dispatch]─[epilogue]──>
```

有 Overlap 模式 (`async_with_compute_stream = true`)下, event包含如下几个:

previous_event:                等待计算完成后再启动 dispatch

previous_event_before_epilogue: 等待另一个计算完成后再启动 epilogue

event (返回):                  标记通信完成, 供下一个计算等待

```
Compute Stream:  ──────[计算任务]───────────[等待event]──>                   ↑                           ↑                   │ previous_event            │ event                   ↓                           │Comm Stream:     ──[等待event]─[dispatch]──[epilogue]────>                                    ↑                                    │ previous_event_before_epilogue
```

这使得 dispatch/compute/combine 三者可以形成流水线:

```
Compute Stream:  ──[compute_A]────────────[compute_B]────────────[compute_C]──>                        ↓                      ↓                      ↓Comm Stream:     ────[dispatch_A]──[combine_A]─[dispatch_B]──[combine_B]──────>
```