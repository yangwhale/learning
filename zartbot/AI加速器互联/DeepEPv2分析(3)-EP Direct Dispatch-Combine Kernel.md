# DeepEPv2分析(3)-EP Direct Dispatch/Combine Kernel

> 作者: zartbot  
> 日期: 2026年5月5日 02:11  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498255&idx=1&sn=3d0b93a65cb5aad476b611e36b5b512a&chksm=f995eacdcee263db197eec00bf072ed7b4d38494fa0c84346a0b46e4a1f2e8141487ee5ef330#rd

---

### TL;DR

这一篇继续分析Direct Dispatch/Combine Kernel的详细实现, 本文目录如下:

```
1. Dispatch Kernel详细分析1.1 Direct Dispatch1.2 Dispatch Copy Epilogue1.3 Dispatch Deterministic Prologue2. Combine Kernel详细分析2.1 Direct Combine2.2 Combine Epilogue
```

## 1. Dispatch Kernel详细分析

### 1.1 Direct Dispatch

详细代码在`deep_ep/include/deep_ep/impls/dispatch.cuh` 文件头的大段注释给出了完整顶层设计:

**模板开关**`kIsScaleupNVLink`: 决定 team 是 `Lsa`(Symmetric Memory LSA Window, NVLink 零拷贝)还是 `World`(RDMA put 语义)；

**Warp 分工**: 前 `kNumNotifyWarps` 条为 **notify warp**(管计数), 其余为 **dispatch warp**(搬数据)；

**Buffer 布局**: `recv_buffer[kNumRanks][kNumMaxTokensPerRank]` 对称暴露给 peer；`send_buffer[1][kNumMaxTokensPerRank]` 仅非 NVLink 做 staging；

**单 token 7 步数据流**: TMA load hidden → cp.async load sf → 写 metadata → 去重分槽 → mbarrier wait → TMA store → RDMA put(可选).

整个数据流程如下图所示:

![图片](assets/eee8b0ee2892.png)
模板参数
```
template <bool kIsScaleupNVLink,      // 【核心开关】所有 peer 是否都可 NVLink 对称访问          bool kDoCPUSync,            // 是否把 rank/expert count 镜像到 host_workspace (供 CPU 轮询)          bool kReuseSlotIndices,     // 是否复用 deterministic prologue 预算好的 slot 号          int kNumSMs,                // 参与本 kernel 的 SM 数          int kNumNotifyWarps,        // 每 block 内 notify warp 数 (warpgroup 倍数)          int kNumDispatchWarps,      // 每 block 内 dispatch warp 数          int kNumRanks,              // 参与 dispatch 的 rank 总数          int kNumHiddenBytes,        // 单 token hidden 字节数          int kNumSFPacks,            // FP8 scale factor pack 数；0 表示无 SF          int kNumMaxTokensPerRank,   // 每 rank 最大接收 token 数          int kNumExperts,            // 全局 expert 总数          int kNumTopk,               // 每 token 选几个 expert          int kExpertAlignment,       // expand 模式下 per-expert block 对齐粒度 (FP8 常用 128)          int kNumQPs,                // RDMA QP 数          int64_t kNumTimeoutCycles,  // 通信超时阈值 (GPU clock cycles)          int kNumNotifyThreads   = kNumNotifyWarps   * 32,          int kNumDispatchThreads = kNumDispatchWarps * 32,          int kNumThreads         = kNumNotifyThreads + kNumDispatchThreads,          typenameteam_t = std::conditional_t<kIsScaleupNVLink,                                               ncclTeamTagLsa,      // NVLink -> LSA team                                               ncclTeamTagWorld>>   // 否则 -> World RDMA team
```
函数签名与参数语义
```
__global__ void __launch_bounds__(kNumThreads, 1)dispatch_impl(    void*        x,                                       // 【输入】用户 token 张量 [num_tokens, hidden]    sf_pack_t*   sf,                                      // 【输入】FP8 scale factor(可 nullptr)    topk_idx_t*  topk_idx,                                // 【输入】router 输出 [num_tokens, topk]    float*       topk_weights,                            // 【输入】可选权重 [num_tokens, topk]    topk_idx_t*  copied_topk_idx,                         // 【输出】topk_idx 的 GPU 侧拷贝(可选)    int*         cumulative_local_expert_recv_stats,      // 【输出】本 rank 累计接收统计    int*         psum_num_recv_tokens_per_scaleup_rank,   // 【输出】inclusive psum [kNumRanks]    int*         psum_num_recv_tokens_per_expert,         // 【输出】exclusive+align psum [kNumExpertsPerRank]    int*         dst_buffer_slot_idx,                     // 【输入/输出】per-token 的目的 slot, deterministic 时复用    constint    num_tokens,                              // 【输入】本 rank 要发送的 token 数    constint    sf_token_stride,    constint    sf_hidden_stride,                        // 【输入】SF 张量的两维 stride    const ncclDevComm_t nccl_dev_comm,                    // 【输入】NCCL device 通信域    const ncclWindow_t  nccl_window,                      // 【输入】对称窗口 (LSA 或 World)    void*        buffer,                                  // 【输入/输出】对称通信缓冲 (recv_buffer + send_buffer)    void*        workspace,                               // 【输入/输出】per-rank 元数据工作区    void*        mapped_host_workspace,                   // 【输入/输出】host 可读镜像 (kDoCPUSync 时使用)    constint    rank_idx) {                              // 【输入】本 rank 全局编号
```

具体的执行流程如下:
1. 公共初始化: 索引 / workspace / SMEM / Gin handle
```
    // kNumExperts 必须能被 kNumRanks 整除(每 rank 均匀承担 expert)    constexprint kNumExpertsPerRank = kNumExperts / kNumRanks;    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");    // Utils    constauto sm_idx     = static_cast<int>(blockIdx.x),               thread_idx = static_cast<int>(threadIdx.x);    constauto warp_idx   = ptx::get_warp_idx(),   // block 内 warp 编号               lane_idx   = ptx::get_lane_idx();   // warp 内 lane 编号    // Workspaces    // workspace_layout 指向 GPU 侧 workspace, host_workspace_layout 指向被 map 的 host 镜像    // 第 2 个参数 1 表示 scaleout_rank 维度为 1(Direct Dispatch 的定义)    constauto workspace_layout      = layout::WorkspaceLayout(workspace,             1, kNumRanks, kNumExperts);    constauto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, 1, kNumRanks, kNumExperts);    // SMEM 基址必须 TMA 对齐；注意全 kernel 不使用 static shared memory    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];        // notify 区 SMEM: 存放 rank_count[kNumRanks] + expert_count[kNumExperts], 按 notify 线程数对齐    constexprint kNumSmemBytesForNotify =        kNumNotifyThreads > 0            ? math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int)            : 0;    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");    // Named barrier indices    //   index 1 用于 notify warp 内部的 named_barrier；index 0 留给硬件缺省    constexprint kNotifyBarrierIndex = 1;    // Gin handle    // 根据 (sm, warp) 映射到具体的 QP 和共享模式；notify warp 与 dispatch warp 分开编址    constauto [qp_idx, sharing_mode] =        comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(            sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);    constauto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);    // Barrier without TMA store flush, without prologue grid sync    // 起始 barrier: tag0 = "进入 dispatch"；保证所有 rank 都到齐才开始动；    //   参数 kFlushTMA=false, kPrologueGridSync=false, kEpilogue=true    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks, kNumSMs, kNumThreads, kNumQPs,                      kNumTimeoutCycles, comm::kDispatchTag0,                      /*flush_tma=*/false, /*prologue_sync=*/false, /*epilogue=*/true>(        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);
```
2. Warp Specialization: Notify vs Dispatch
```
    // Different warp roles    if (warp_idx < kNumNotifyWarps) {        // ================== Notify warp: 管理 rank/expert 计数 ==================        ...    } else {        // ================== Dispatch warp: 搬运 token 数据 ==================        ...    }
```
3. Notify warp: SMEM 计数器清零 + 本 SM 统计
Notify warp 共 `kNumNotifyWarps` 个, 目标是算出"所有 rank 加起来, 每个 rank 将收到多少 token、每个 expert 将收到多少 token". 每个 notify warp 用 `(warp_idx * kNumSMs + sm_idx)` 交错领取 token:

每个 lane 读一个 topk 项, `atomicAdd_block(expert_count + e, 1)`.

rank 维度需要**去重**(同一 token 可能路由到同一 rank 的多个 expert), 用 `ptx::deduplicate(dst_rank, lane)` 保证每个 token 对每个 rank 只加一次.

```
        // Assign shared memory        // SMEM 按 int 计算；rank_expert_count = [rank_count | expert_count] 连续布局        constexprint kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);        constauto rank_expert_count = math::advance_ptr<int>(smem, 0);        // Clean initial counts        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`        int *rank_count   = rank_expert_count,            *expert_count = rank_expert_count + kNumRanks;#pragma unroll        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++i)            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;        // 保证所有 notify 线程清零完成后, 再开始计数        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);        // Atomic add on shared memory        // lane 编号 = 第几个 topk 副本        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");        constauto global_warp_idx = warp_idx * kNumSMs + sm_idx;        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {            // 每 lane 读该 token 的第 lane 个 topk 专家 id(-1 表示该副本无效)            constauto dst_expert_idx =                lane_idx < kNumTopk ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;                       // expert 计数: 无需去重(DeepSeek router 保证同 token 不重复选同 expert)            if (dst_expert_idx >= 0)                atomicAdd_block(expert_count + dst_expert_idx, 1);            // rank 计数: 需要去重, 因为多个 topk 副本可能命中同一 rank 的不同 expert            //   ptx::deduplicate 用 warp-wide shuffle 判重, 仅保留最小有效 lane            constauto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)                atomicAdd_block(rank_count + dst_rank_idx, 1);        }        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
```
4. Notify warp: 跨 SM 全 grid reduce
这里有一个技巧, 一个 `int64` 同时编码两个量:

```
counter = (1ll << 32) | rank_expert_count[i];  // 高32位=1(SM到达计数)低32位=本SM 计数ptx::red_add(notify_reduction_workspace + i, counter);
```

所有 SM 用 `red.add.release.sys` 原子加到同一位置, 最后低 32 位是所有 SM 的 per-rank/per-expert 总数, 高 32 位是已到达的 SM 数.

```
// Do full-grid reduction// ★ 跨 SM reduce: 每个 SM 把自己的本地 count 用 red_add 累加到 gmem 的 reduction workspace//   同时高 32 位 +1 计数 "已到达 SM 数"；SM 0 会轮询这个 high-word 达到 kNumSMs 才继续#pragma unroll        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {            // 低 32 位 = count, 高 32 位 = arrived SM 数(+1 表示自己已到)            constint64_t counter = (1ll << 32ll) | rank_expert_count[i];            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);        }        // ★ 只让 SM 0 完成后续的 peer 通告 / 等待 / psum 工作, 避免重复        if (sm_idx == 0) {// Reduce all SM's count, Wait all SMs' arrival// ★ 超时轮询: status >> 32 == kNumSMs 表示所有 SM 都 red_add 完成#pragma unroll            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {                comm::timeout_while<kNumTimeoutCycles>(true, [=](constbool& is_last_check) {                    constauto status = ptx::ld_volatile<int64_t>(                        workspace_layout.get_notify_reduction_workspace_ptr() + i);                    if ((status >> 32) == kNumSMs) {                        // 低 32 位是真正的 count；encode_decode_positive 做非零编码保证后续可检测 ready                        constauto encoded = math::encode_decode_positive(static_cast<int>(status & 0xffffffffll));                        rank_expert_count[i] = encoded;                        // RDMA 场景还要把 encoded count 写到 scaleup_rank_expert_count 工作区供 put<World> 使用                        ifconstexpr (not kIsScaleupNVLink)                            workspace_layout.get_scaleup_rank_expert_count_ptr<true>()[i] = encoded;                        // Clean for the next usage                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;                        returntrue;                    }                    if (is_last_check) {                        printf("DeepEP notify (GPU reduction) timeout, rank: %d/%d, "                               "thread: %d, status: %d | %d, expected: %d\n",                               rank_idx, kNumRanks, thread_idx,                               static_cast<int>(status >> 32),                               static_cast<int>(status & 0xffffffff),                               kNumSMs);                    }                    returnfalse;                });            }            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
```
5. Notify warp: 向所有 peer 通告 rank/expert count
```
            // TODO: for further optimization, we can fuse rank and expert counters            // Issue scaleup rank count writes to peers            // ★ 第一部分: rank count —— 对每个 peer i, 把 "我给 i 发了 rank_count[i] 个 token"            //   put 到 peer 的 scaleup_rank_count[rank_idx] 位置            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {                constauto dst_rank_counter =                    workspace_layout.get_scaleup_rank_count_ptr<false>() + rank_idx;                gin.put_value<team_t>(                    dst_rank_counter,                    static_cast<int64_t>(rank_count[i]),                    i,                                          // 目的 peer                    ncclGinOptFlagsAggregateRequests);          // 聚合请求, 提升吞吐            }            __syncwarp();            // Issue scaleup expert count writes to peers            if constexpr (kIsScaleupNVLink) {                // NVLink per-element copy                // We don't use TMA as the dtype of shared memory and global is different                // 【NVLink 分支】: 逐 int put, 因为 SMEM 是 int 而 GMEM 是 int64                for (int i = thread_idx; i < kNumExperts; i += kNumNotifyThreads) {                    // peer i 的本地 expert 起点位置 = kNumExpertsPerRank * rank_idx                    constauto idx = kNumExpertsPerRank * rank_idx + (i % kNumExpertsPerRank);                    gin.put_value<team_t>(                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,                        static_cast<int64_t>(expert_count[i]),                        i / kNumExpertsPerRank);                // 目的 peer = expert / per_rank                }            } else {                // RDMA bulk copy                // 【RDMA 分支】: 对每个 peer i, 批量 put 其全部 kNumExpertsPerRank 个 int64 count                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {                    constauto src_ptr = workspace_layout.get_scaleup_expert_count_ptr<true>()                                         + kNumExpertsPerRank * i;                    constauto dst_ptr = workspace_layout.get_scaleup_expert_count_ptr<false>()                                         + kNumExpertsPerRank * rank_idx;                    gin.put<team_t>(dst_ptr, src_ptr, kNumExpertsPerRank * sizeof(int64_t), i);                }            }            // This is necessary, as the waited results will rewrite the shared memory            // ★ 必需的 barrier: 下一段会等 peer 写过来的结果, 而存储区域复用 SMEM            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
```
6. Notify warp: 等待 peer count 到达 + 解码
注意这里有一个技巧, encode_decode_positive的计算公式为

`encode_decode_positive` 同时是**编码**函数和**解码**函数, 如下所示:

原始语义

取值域

经过 f 后

判定

未到达状态（初值或被清零）
`value = 0``f(0) = -1``-1 < 0`
 → **not ready**

已到达，count = 0
`v = 0``f(0) = -1`
写到 gmem 的是 -1

已到达，count = k > 0
`v = k``f(k) = -(k+1)`
写到 gmem 的是 -(k+1)

发送端写入的所有 encoded 值 : **严格为负**

未到达时 gmem 保持 `0`: **严格非负**;

符号位天然成为 ready 标志. 因此 `is_decoded_positive_ready` 用非负性判断“是否已成功解码回原始值”,

```
            // Wait for rank and expert count            // ★ 对称等待: 每个 slot 轮询 encode_decode_positive 的 ready 标志            constauto start_clock = clock64();            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {                comm::timeout_while<kNumTimeoutCycles>(                    [=](constbool& is_last_check) {                        // NOTES: the global memory type has 64 bits                        constauto count = static_cast<int>(ptx::ld_volatile<int64_t>(                            workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + i));                        constauto decoded = math::encode_decode_positive(count);                        if (math::is_decoded_positive_ready(decoded)) {                            // 清零供下次使用；把解码后的 count 搬到 SMEM                            workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] = 0;                            rank_expert_count[i] = decoded;                            returntrue;                        }                        if (is_last_check)                            printf("DeepEP notify timeout, rank: %d, thread: %d, count: %d\n",                                   rank_idx, i, decoded);                        returnfalse;                    },                    start_clock);            }            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);
```
7. Notify warp: 本 rank expert count 聚合 + `expert_alignment` 对齐
```
            // Reduce expert count and add stats            // ★ 核心 —— 对 "本 rank 的每个 local expert i" 累加所有 peer 给过来的计数            //   然后 math::align(sum, kExpertAlignment)            //   这是 expand 模式 per-expert block 对齐的源头(alignment padding 在 epilogue 处体现)            for (int i = thread_idx; i < kNumExpertsPerRank; i += kNumNotifyThreads) {                int sum = 0;#pragma unroll                for (int j = 0; j < kNumRanks; ++j)                    sum += expert_count[j * kNumExpertsPerRank + i];                expert_count[i] = math::align(sum, kExpertAlignment);                // Update statistics counters                // 外部统计累计值(可选)                if (cumulative_local_expert_recv_stats != nullptr)                    atomicAdd(cumulative_local_expert_recv_stats + i, sum);            }            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);            // Write host workspace            // 【CPU sync 分支】把 rank_count[:] + expert_count[:kNumExpertsPerRank] 镜像到 host workspace            //  CPU 侧轮询这里获得精确 num_recv_tokens            if constexpr (kDoCPUSync) {                for (int i = thread_idx; i < kNumRanks + kNumExpertsPerRank; i += kNumNotifyThreads) {                    host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] =                        math::encode_decode_positive(rank_expert_count[i]);                }                __syncwarp();            }
```
8.  Notify warp: 双 psum(rank inclusive / expert exclusive)
```
            // Do prefix sum by the warps            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use            // ★ 自实现 warp 级前缀和；is_exclusive=0 → inclusive, is_exclusive=1 → exclusive            constauto do_psum = [=](constint* count, int* out, constint n, constint is_exclusive) {                int psum = 0;#pragma unroll                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++i) {                    constauto idx = i * 32 + lane_idx;                    constauto mem_idx = idx - is_exclusive;                    constauto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;                    // warp_inclusive_sum = __shfl_up 的 log2(32) 步合并                    constauto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);                    // Store into global memory                    if (idx < n + is_exclusive)                        out[idx] = sum;                    // Update `psum` by using the last lane's value                    // 下一 32 元素块的起点 = 当前块 lane 31 的值                    psum = ptx::exchange(sum, 31);                }            };            if (warp_idx == 0) {                // Inclusive prefix sum —— 供 epilogue 的 scaleup_rank 段定位                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumRanks, 0);            } elseif (warp_idx == 1) {                // Exclusive prefix sum for later expanding                // ★ expand 模式的起点: epilogue 里 atomicAdd(psum_expert + e, 1) 从此处开始抢槽                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);            }        }    } else {        // ================== Dispatch warp: 处理 token 数据搬运 ==================
```
9. Dispatch warp: Buffer layout + mbarrier 初始化
```
        // 本 dispatch warp 在 dispatch warp 群内的编号(0..kNumDispatchWarps-1)        constint dispatch_warp_idx = warp_idx - kNumNotifyWarps;        // Buffer layouts        // TokenLayout: hidden + sf + metadata(with_metadata=true) ← dispatch 必带 src_global_idx 等        constauto token_layout = layout::TokenLayout(            kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);        // smem 上的 per-warp staging: 带 mbarrier, 跳过前面 notify 占用的 SMEM        constauto tma_buffer =            layout::BufferLayout<true>(                token_layout,                kNumDispatchWarps,                1,                math::advance_ptr<int>(smem, kNumSmemBytesForNotify))                .get_rank_buffer(dispatch_warp_idx)                .get_token_buffer(0);        // gmem 对称 recv 区: [kNumRanks, kNumMaxTokensPerRank]        //   peer 会写到 recv_buffer[rank_idx], 所以本 rank 先切出该段备用        auto recv_buffer = layout::BufferLayout<false>(            token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);        // gmem 本地 send 区 (RDMA staging, NVLink 下不用)        auto send_buffer = layout::BufferLayout<false>(            token_layout, 1, kNumMaxTokensPerRank, recv_buffer.get_buffer_end_ptr());        // 本 rank 视角: peer 会写到 recv_buffer[rank_idx]; 此 slice 用于后续 get_sym_ptr 翻译        recv_buffer = recv_buffer.get_rank_buffer(rank_idx);        // Init TMA        // 初始化 smem mbarrier, 用于等待 TMA load(hidden) + cp.async(sf) 全部完成        ptx::arrival_phase phase = 0;        constauto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();        if (ptx::elect_one_sync())            ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);        __syncwarp();
```
10. Dispatch warp: 主循环 + 步骤 1-2(TMA load hidden + cp.async load SF)
```
        // Iterate all tokens        // warp 间按 (warp * kNumSMs + sm) 交织分布 token, 提高并行度        constauto token_start  = dispatch_warp_idx * kNumSMs + sm_idx;        constauto token_stride = kNumDispatchWarps * kNumSMs;        for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {            constauto token_i64_idx = static_cast<int64_t>(token_idx);            // Wait TMA store arrivals            // 等待上一轮 TMA store 全部落地, 才能安全复用 tma_buffer            ptx::tma_store_wait();            __syncwarp();            // Issue data TMA            // ---- 步骤1: TMA load hidden 到 smem ----            // TMA 是 warp 级操作, 只需一条 lane 发起            if (ptx::elect_one_sync()) {                ptx::tma_load_1d(                    tma_buffer.get_hidden_ptr(),                    math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),                    mbarrier_ptr,                    kNumHiddenBytes);            }            __syncwarp();            // Issue SF TMA or cp.async            // ---- 步骤2: 若有 FP8 scale-factor, 用 cp.async 载入 smem ----            if constexpr (kNumSFPacks > 0) {                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");                constauto gmem_src_ptr = math::advance_ptr<sf_pack_t>(                    sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));                constauto smem_dst_ptr = tma_buffer.get_sf_ptr();                constexprauto kNumFullIters = kNumSFPacks / 32;#pragma unroll                for (int k = 0; k < kNumFullIters; ++k) {                    // 32 lane 合作 cp.async, 每 lane 搬一个 pack                    ptx::cp_async_ca(                        gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,                        smem_dst_ptr + k * 32 + lane_idx);                }                // 余数轮                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {                    ptx::cp_async_ca(                        gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,                        smem_dst_ptr + kNumFullIters * 32 + lane_idx);                }                // cp.async 通过 mbarrier_arrive 告知 mbarrier 完成事件                //   → hidden 的 TMA load 与 SF 的 cp.async 会同一个 mbarrier 上等待                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);                __syncwarp();            }
```
11. Dispatch warp: 步骤 3a/3b(读 topk + 写 src_global_idx)
```
            // Load top-k indices and weights            // ---- 步骤3a: 读 topk_idx 并解算目的 rank_idx ----            // lane_idx = 第几个 topk 副本；stored_dst_rank_idx = 该副本对应的目的 rank            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");            int stored_dst_rank_idx = -1;            if (lane_idx < kNumTopk) {                constauto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);                constauto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);                // expert_id / per_rank = rank_id                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;                // 把 expert_id / weight 写入 SMEM slot(供远端 epilogue 解包)                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;                if (topk_weights != nullptr)                    tma_buffer.get_topk_weights_ptr()[lane_idx] =                        __ldg(topk_weights + token_idx * kNumTopk + lane_idx);                // 可选: 在 GPU 上拷一份 topk_idx, 供反向 combine 使用                if (copied_topk_idx != nullptr)                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;            }            __syncwarp();            // Add source metadata (rank index and token index)            // Please ensure no TMA buffer shared memory writes after this part            // ---- 步骤3b: 写 src_token_global_idx (供 combine 反向路由) ----            //   rank_idx * kNumMaxTokensPerRank + token_idx —— 全局唯一的源 token 标识            if (ptx::elect_one_sync())                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;            // fence: 确保 metadata 写入对后续 TMA store 可见            ptx::tma_store_fence();            __syncwarp();
```
12.Dispatch warp: 步骤 4(去重 + slot 分配)
```
            // Deduplicate ranks and assign slots            // ---- 步骤4: 目标 rank 去重 & 分 slot ----            int stored_dst_slot_idx = -1;            if constexpr (kReuseSlotIndices) {                // 【Deterministic 模式】复用 prologue 预算好的 slot                //   prologue 已通过前缀和确定每个 (src_token, topk) 在 peer recv 的唯一位置                //   这里只需按 lane 读取, 并减去 rank_idx 基址还原为 peer 内 slot                if (lane_idx < kNumTopk)                    stored_dst_slot_idx = __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + lane_idx);                stored_dst_slot_idx = stored_dst_slot_idx >= 0                    ? (stored_dst_slot_idx - rank_idx * kNumMaxTokensPerRank)                    : -1;            } else {                // 【运行时 atomic 模式】per-dst-rank 原子计数器                //   ptx::deduplicate: 同一 (token, rank) 对只保留一条有效 lane(最小 lane)                //   原子分槽的计数器放在 workspace 的 scaleup_atomic_sender_counter                if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)                    stored_dst_slot_idx = atomicAdd(                        workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);                // 把 slot 写回 dst_buffer_slot_idx, 供 combine 反向查找                //   value = -1 或 (rank_idx * kNumMaxTokensPerRank + slot)                if (lane_idx < kNumTopk) {                    constauto value = stored_dst_slot_idx >= 0                        ? rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx                        : -1;                    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;                }            }            __syncwarp();
```
13. Dispatch warp: 步骤 5(等 TMA load + cp.async 完成)
```
            // Wait TMA load arrival            // NOTES: this arrive must be after the `ptx::cp_async_mbarrier_arrive`            // ---- 步骤5: mbarrier 等 hidden + sf 全部到位 ----            //   mbarrier_arrive_and_set_tx(bytes) 设置 TMA 事务字节数(hidden 部分)            //   wait_and_flip_phase 阻塞直到:             //     ① 字节数达到 kNumHiddenBytes  ② cp.async_mbarrier_arrive 也已计入            if (ptx::elect_one_sync()) {                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);            }            __syncwarp();
```
14. Dispatch warp: 步骤 6a/6b(TMA store)
```
            // TMA store to send buffer            // ---- 步骤6a: 非 NVLink 下 TMA store 到本地 send_buffer (RDMA staging) ----            //   send_buffer 是本 rank 私有的一份镜像；下面的 gin.put 会以它为源            auto send_buffer_ptr = send_buffer.get_token_buffer(token_idx).get_base_ptr();            if constexpr (not kIsScaleupNVLink) {                if (ptx::elect_one_sync())                    ptx::tma_store_1d(                        send_buffer_ptr,                        tma_buffer.get_base_ptr(),                        tma_buffer.get_num_bytes<false>());                ptx::tma_store_commit();                __syncwarp();            }            // Issue TMA NVLink stores            // ---- 步骤6b: NVLink 下零拷贝: TMA store 到 get_sym_ptr(peer recv[slot]) ----            //   ★ 关键: get_sym_ptr 把本地地址翻译成对应 peer 的对称地址            //   可 NVLink 访问的 peer 走这条路径；不可访问的 peer 在步骤 7 走 RDMA            //   每条有效 topk lane 独立发一次 TMA store → 一个 warp 可能发 kNumTopk 次            EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid top-k selection");            constauto dst_ptr = stored_dst_slot_idx >= 0                ? gin.get_sym_ptr<team_t>(                      recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),                      stored_dst_rank_idx)                : nullptr;            if (dst_ptr != nullptr)                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());            ptx::tma_store_commit();            __syncwarp();
```
15. Dispatch warp: 步骤 7(RDMA put, 仅非 NVLink)
```
            // Issue RDMA put            // ---- 步骤7: 非 NVLink 下用 gin.put<World> 把 send_buffer 内容发到 peer recv ----            if constexpr (not kIsScaleupNVLink) {                // Wait the send buffer store to arrive                // ★ tma_store_wait<1> = 等待 "最多 1 条" 未完成的 store(即步骤 6a 那条)                //   保证 send_buffer 内容已落盘, 才能作为 RDMA 的源                ptx::tma_store_wait<1>();                __syncwarp();                // NOTES: we should skip the NVLink accessible ranks                // 对可 NVLink 访问的 peer, 已在步骤6b 直写; 此处仅对无法直访的 peer 发 RDMA                //   dst_ptr == nullptr 意味着 get_sym_ptr 返回空(peer 不在 LSA 域)                if (stored_dst_slot_idx >= 0and dst_ptr == nullptr) {                    gin.put<team_t>(                        recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),                        send_buffer_ptr,                        tma_buffer.get_num_bytes<false>(),                        stored_dst_rank_idx);                }                __syncwarp();            }        }  // end of for-loop over tokens    }  // end of dispatch warp branch
```
16. Kernel 结尾: 收尾 barrier + 触发 PDL + 清理
```
    // Barrier to ensure data arrival    // 结尾 barrier: tag1 = "dispatch 完成"    //   kFlushTMA=true  → 落地所有未完成的 TMA store    //   kPrologueGridSync=true → 开启与下一 kernel 的 grid 依赖    //   kEpilogue=false → 不是尾段 barrier    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks, kNumSMs, kNumThreads, kNumQPs,                      kNumTimeoutCycles, comm::kDispatchTag1,                      /*flush_tma=*/true, /*prologue_sync=*/true, /*epilogue=*/false>(        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);    // Trigger the copy epilogue kernel    // ★ PDL 触发点: 允许后继 dispatch_copy_epilogue_impl grid 开始真正执行    //   epilogue kernel 里的 cudaGridDependencySynchronize() 正是等待这个信号    cudaTriggerProgrammaticLaunchCompletion();    // Clean atomic counters    // 运行时 atomic 分槽模式下, 清零 scaleup_atomic_sender_counter 供下次 dispatch 使用    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");    if (not kReuseSlotIndices and sm_idx == 0and thread_idx < kNumRanks)        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;}}  // namespace deep_ep::elastic
```

### 1.2 Dispatch Copy Epilogue

Dispatch 主 kernel 把远端 token 扔到对称内存 `scaleup_buffer` 后, 该 kernel 负责把每个 slot 里的内容拆解、分发并最终落到用户可见的 `recv_x / recv_sf / recv_topk_idx / recv_topk_weights / recv_src_metadata`.  两段处理流程之间通过PDL(Programmatic Dependent Launch)串在一起

![图片](assets/c4e0ed99f7da.png)

需要注意的是Expand模式发生在这个位置. 两种模式在 `recv_x` 上的布局

非 expand 模式: `recv_x[i] = 第 i 个接收 token` , 按 rank-channel 顺序原位堆叠, 最终形状为`[num_recv, hidden]`

expand模式:  按照Expert分块, 并按照 expert_alignment 进行padding, 最终形状为`[num_expanded, hidden]`

我们以一个 2 ranks x 2 local-expert/rank, topk=2的例子来说明, 假设本 rank 是 rank 0(持有 expert 0/1), 接收到 2 个 token:

`tok_a.topk_idx = [0, 3]` : 仅命中本 rank expert 0 , 另一个在其它rank

`tok_b.topk_idx = [0, 1]` : 命中本 rank expert 0 + expert 1

模式
`recv_x`
 行数

布局

非 expand

2
`recv_x[0] = tok_a; recv_topk_idx[0] = [0, -1]`

`recv_x[1] = tok_b; recv_topk_idx[1] = [0, 1]`

Expand (Alignment=1)

3
`recv_x[0] = tok_a (→expert 0)`

`recv_x[1] = tok_b (→expert 0)`
`recv_x[2] = tok_b (→expert 1)`

Expand (Alignment=4)

8
`recv_x[0..1] = tok_a/tok_b 属 expert 0；recv_x[2..3] padding`

`recv_x[4] = tok_b 属 expert 1；recv_x[5..7] padding`

Expand模式对于后续的GroupGEMM有好处, 可以直接在recv_x中切片即可, 同时由于添加了expert_alignment padding处理起来也很简单.

Kernel详细的执行流程如下
1. 模板参数: 编译期定制 kernel 行为
```
template <bool kDoExpand,                  //【核心开关】true = expand 模式, 一行一个 (token, expert) 副本          bool kCachedMode,                // 是否是缓存 dispatch(略过链表构建)          // NOTES: this channel concept only applies for scale-out ranks          int kNumSMs,                     // 参与 kernel 的 SM 数          int kNumChannels,                // 每 rank 的通信通道数(仅 scaleout 生效)          int kNumWarps,                   // 每 block 的 warp 数          int kNumScaleoutRanks,           // scaleout 域 rank 数(通常走 RDMA)          int kNumScaleupRanks,            // scaleup 域 rank 数(通常走 NVLink)          int kNumHiddenBytes,             // 单 token hidden 维度的字节数(已含 dtype)          int kNumSFPacks,                 // FP8 scale factor pack 数；0 表示无 SF          int kNumMaxTokensPerRank,        // 每 rank 最大 token 容量(worst case 的占位)          int kNumExperts,                 // 全局 expert 总数          int kNumTopk,                    // 每 token 选几个 expert          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,          int kNumThreads = kNumWarps * 32,          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),          bool kDoCreateLinkedList = (kNumScaleoutRanks > 1andnot kCachedMode)>          //                         ↑ 仅 hybrid 非 cached 模式需要构建 per-channel 链表
```
2. 函数签名与参数
```
__global__ void __launch_bounds__(kNumThreads, 1)dispatch_copy_epilogue_impl(    void* buffer,                                   // 【输入】对称通信缓冲, 主 kernel 填充；本 kernel 只读    void* workspace,                                // 【输入/输出】per-rank 元数据工作区(链表 tail 指针等)    int*  psum_num_recv_tokens_per_scaleup_rank,    // 【输入】inclusive psum, 用于定位每段 scaleup rank 边界    int*  psum_num_recv_tokens_per_expert,          // 【输入/输出】exclusive+align psum                                                    //    expand 模式下本 kernel 会 atomicAdd 在此计数    void* recv_x,                                   // 【输出】用户可见 token 张量 [num_recv|num_expanded, hidden]    sf_pack_t*   recv_sf,                           // 【输出】FP8 scale factor(strided 布局)    topk_idx_t*  recv_topk_idx,                     // 【输出-非 Expand 专用】[num_recv, topk] 命中 local expert id    float*       recv_topk_weights,                 // 【输出】非 Expand: [num_recv, topk]；Expand: [num_expanded]    int*         recv_src_metadata,                 // 【输出】[num_recv, 2+topk], 供 combine 反向路由    int*         channel_linked_list,               // 【输出】hybrid 非 cached: per-channel 链表体    int          num_recv_tokens,                   // 【输入】CPU-sync 下是精确值；否则为 worst-case 哨兵    constint    recv_sf_token_stride,              // 【输入】recv_sf 在 token 维的 stride(packs 数)    constint    recv_sf_hidden_stride,             // 【输入】recv_sf 在 hidden 维的 stride    constint    scaleout_rank_idx,                 // 【输入】本 rank 的 scaleout 编号    constint    scaleup_rank_idx) {                // 【输入】本 rank 的 scaleup 编号
```
3. 步骤 0: 线程/warp 索引与本 rank 的 expert 区间
```
    // Utils    constauto sm_idx     = static_cast<int>(blockIdx.x),               thread_idx = static_cast<int>(threadIdx.x);    constauto warp_idx   = ptx::get_warp_idx(),     // block 内 warp 编号 (0..kNumWarps-1)               lane_idx   = ptx::get_lane_idx();     // warp 内 lane 编号 (0..31)    // 跨 SM 先展开 warp 再跨 block —— 同一 warp_idx 分散到不同 SM,     // 提高 L2 局部性；尾部 "不满一波" 时均匀落到各 SM    constauto global_warp_idx = warp_idx * kNumSMs + sm_idx;    // For top-k index transformations    // 每个 rank 分到的 local expert 数(assumes 整除)    constexprint kNumExpertsPerRank = kNumExperts / kNumRanks;    // 本 rank 在全局 expert 线性空间里的起止位置    constauto rank_idx         = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;    constauto expert_start_idx = kNumExpertsPerRank * rank_idx,               expert_end_idx   = kNumExpertsPerRank * (rank_idx + 1);    // 含义: 后续只接受 dst_expert_idx ∈ [expert_start_idx, expert_end_idx)
```
4. 步骤 1: 三个 Layout 对象 + TMA mbarrier 初始化
```
    // Buffer layouts    // SMEM 基址；TMA 要求 128B 对齐    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];    // 单 token 的 slot 结构(hidden + sf + topk_idx + topk_weights + src_global_idx + linked_list_idx)    // 第 4 个实参 true = with_metadata, 意味着 slot 里包含 topk_idx / topk_weights / 元数据    constauto token_layout  = layout::TokenLayout(        kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, /*with_metadata=*/true);    // 第一层 buffer: SMEM 里的 per-warp tma 缓冲区    //   模板参数 true 表示 slot 尾部带 mbarrier(由 TMA 事务计数器使用)    //   get_rank_buffer(warp_idx) 把每个 warp 隔到不同 slot；.get_token_buffer(0) 取第 0 条    constauto tma_buffer = layout::BufferLayout<true>(        token_layout, kNumWarps, /*num_token=*/1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);    // 第二层 buffer: GMEM 里对称通信缓冲    //   shape = [kNumScaleupRanks][kNumScaleoutRanks * kNumMaxTokensPerRank] × TokenSlot    //   第一维按发送方 scaleup 切段；第二维才是接收 token 序号    constauto scaleup_buffer =        layout::BufferLayout<false>(            token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);    // Init TMA    // 每个 warp 在自己的 SMEM slot 里初始化一个 mbarrier；count=1 意味着一次到达即可翻转 phase    ptx::arrival_phase phase = 0;    constauto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();    if (ptx::elect_one_sync())                             // 仅一条 lane 执行, 避免重复初始化        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);    __syncwarp();                                          // fence 对全 warp 可见
```
5. 步骤 2: PDL 跨 grid 同步 + num_recv_tokens 动态读取
```
    // PDL Barrier. 等主 dispatch kernel grid 全部落盘；    //   不能用 __ldg(read-only cache 在 PDL 下可能读旧值)    cudaGridDependencySynchronize();    // For no CPU sync case, the number of received tokens should be read from the GPU tensor    // 非 CPU-sync: host 侧把 num_recv_tokens 设成 worst-case 哨兵 (=kNumMaxTokensPerRank  * kNumRanks)    //   这里从 GPU 侧精确 psum 的最后一项重读真实值, 后续循环据此截断    if (num_recv_tokens == kNumMaxTokensPerRank * kNumRanks)        num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks - 1];
```
6. 步骤 3: 主循环 — 动态维护 `current_rank_idx`
```
    // Current rank indices should be maintained    // 每个 warp 在整个 recv_buffer 上跳步处理 token, 需维护当前所属的 scaleup rank 段    int current_rank_idx = -1, stored_psum_num_recv_tokens;    // stored_* 是 warp 内每 lane 的寄存器缓存    int current_rank_start = 0, current_rank_end = 0;          // 当前段在全局接收序列里的 [start, end)#pragma unroll    for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {        // Calculate token index in the buffer        // ---- 步骤 2: 根据 psum 定位 i 属于哪一段 scaleup rank, 并算出 buffer_token 地址 ----        while (i >= current_rank_end) {                         // 循环推进 rank 段, 直到覆盖当前 i            current_rank_idx += 1;            EP_DEVICE_ASSERT(current_rank_idx < kNumScaleupRanks);            // 【优化】每 32 次 rank 切换只发一次 gmem 读:             //   - lane 0 (stored_lane_idx == 0 时) 32 lane 协同一次性读入 32 个 psum            //   - 后续 31 次从寄存器通过 warp shuffle (ptx::exchange) 取            constauto stored_lane_idx = current_rank_idx % 32;            if (stored_lane_idx == 0and current_rank_idx + lane_idx < kNumScaleupRanks)                stored_psum_num_recv_tokens =                    psum_num_recv_tokens_per_scaleup_rank[current_rank_idx + lane_idx];            current_rank_start = current_rank_end;            // ptx::exchange(reg, src_lane) = __shfl_sync: 把 src_lane 的 reg 广播给全 warp            current_rank_end = ptx::exchange(stored_psum_num_recv_tokens, stored_lane_idx);        }        // 把全局接收序号 i 换算成当前 scaleup_rank 段内的本地 slot 索引 → 得到对应 slot 的基址        constauto buffer_token =            scaleup_buffer.get_rank_buffer(current_rank_idx).get_token_buffer(i - current_rank_start);
```
7. 步骤 4: SMEM 流水线握手 + TMA load
```
        // Wait buffer releases        // ★ 等待上一轮 tma_store 把 SMEM slot 的内容 flush 到 gmem,         //   然后本轮才能复用这条 SMEM slot(整个 kernel 只用 1 条 SMEM slot / warp)        ptx::tma_store_wait();        __syncwarp();        // Issue TMA loads        // Including all stuffs: data, SF, top-k metadata        // ---- 步骤 4: TMA 一次把整 token (hidden+sf+topk+metadata) 搬到 SMEM ----        //   TMA 是 warp 级操作, 只能一条 lane 发；        //   mbarrier_arrive_and_set_tx 设置本次事务字节数(到达 = 字节数满足)        if (ptx::elect_one_sync()) {            ptx::tma_load_1d(tma_buffer.get_base_ptr(),                             buffer_token.get_base_ptr(),                             mbarrier_ptr,                             tma_buffer.get_num_bytes<false>());            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, tma_buffer.get_num_bytes<false>());        }        __syncwarp();
```
8. 步骤 5: 利用 TMA 延迟并行读 topk_idx + 命中过滤
```
        // Load target expert indices separately to tolerate TMA load latency        // ---- 步骤 5a: 并行 gmem 直读 topk_idx 以掩盖 TMA load 的延迟 ----        // lane 编号 = 第几个 topk 副本        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");        int dst_expert_idx = -1;        if (lane_idx < kNumTopk)            // 此处 gmem 读可以与上方 TMA load 完全并行, 形成"计算隐藏数据"            dst_expert_idx = buffer_token.get_topk_idx_ptr()[lane_idx];        __syncwarp();        // Validate target expert indices and store for non-expand mode        // ---- 步骤 5b: 过滤命中本 rank expert 的 topk 副本 ----        constauto in_range = expert_start_idx <= dst_expert_idx and dst_expert_idx < expert_end_idx;        // ptx::gather(pred) → 32 位 bitmask；get_master_lane_idx → 最小为 1 的 lane 号        // 【用途】选出"本 token 的主副本 lane"(metadata 字段 1 会引用)        constauto master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(in_range));        // 把全局 expert_id 转成 local expert_id；未命中者置 -1        dst_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;        // 断言: 同一 token 不会把同一个 local expert 选两次(DeepSeek router 的要求)        EP_DEVICE_ASSERT(ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1);        //【非 Expand】所有 lane 都写 recv_topk_idx[i, lane], 未命中 lane 写 -1        //   combine 反向根据 -1 跳过该 expert；        //【Expand】不写(位置信息由 metadata 的 field 2..2+topk 替代)        if (not kDoExpand and lane_idx < kNumTopk)            recv_topk_idx[i * kNumTopk + lane_idx] = static_cast<topk_idx_t>(dst_expert_idx);        __syncwarp();
```
9. 步骤 6: 计算 `dst_tensor_idx`(普通模式和Expand模式的不同处理逻辑)
```
        // Calculate target indices in the tensor        // ---- 步骤 6: 计算 dst_tensor_idx (并根据expand模式处理) ----        int dst_tensor_idx = -1;        //【非 Expand】: elect_one 选一条 lane, dst_tensor_idx = i → 整个 warp 后续只发 1 次 store        if (not kDoExpand and ptx::elect_one_sync()) {            dst_tensor_idx = i;        //【Expand】: 每条 in_range lane 独立 atomicAdd 本 expert 槽的计数器        //   atomicAdd 返回的旧值即 "第几个落到该 expert 的 token"        //   起点 = psum_num_recv_tokens_per_expert[e](已由主 kernel 做 align+exclusive psum)        //   所以多条 lane 将产生多个不同的 dst_tensor_idx, 对应 "一 (token, expert) 一行"        } else if (kDoExpand and dst_expert_idx >= 0) {            dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);        }        __syncwarp();
```
10. 步骤 7: 等 TMA load 完成
```
        // Wait for TMA arrival        // ---- 步骤 7: mbarrier 阻塞直到 TMA 事务字节数满足；        //   然后翻转 phase (mbarrier 是双 phase 复用的, 避免每轮重新 init) ----        if (ptx::elect_one_sync())            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);        __syncwarp();        // 执行到此处: SMEM tma_buffer 完整可用        // 注意: 步骤 5a / 5b / 6 都在 wait 之前(它们只需 gmem 的 topk 字段 + 寄存器运算)        //      利用 TMA 载入的延迟窗口
```
11. 步骤 8a: Hybrid 链表前驱指针(仅 hybrid 非 cached)
```
        // Maintain linked list        // ---- 步骤 8a: 写前驱节点的 next 指针, 把本 token 串进 per-channel 链表 ----        //   slot 里的 linked_list_idx 字段由主 dispatch kernel 填入 "本 token 的前驱下标"        //   这里令 channel_linked_list[prev] = i, 形成 prev → i → ... → -1 的链        if constexpr (kDoCreateLinkedList) {            if (ptx::elect_one_sync())                channel_linked_list[tma_buffer.get_linked_list_idx_ptr()[master_src_topk_idx]] = i;            __syncwarp();        }
```
12. 步骤 8b: TMA store hidden
```
        // Issue TMA stores for data        // ---- 步骤 8b: TMA store hidden 到 recv_x[dst_tensor_idx] ----        //【非 Expand】: elect_one  → 1 次 store → recv_x[i] 写入一行        //【Expand】  : 每条有效 lane (dst_tensor_idx >= 0) → N 次 store → recv_x 多行同 hidden        //             这就是 "一行一个 (token, expert) 副本" 的物理实现        if (kDoExpand ? (dst_tensor_idx >= 0) : ptx::elect_one_sync()) {            ptx::tma_store_1d(                math::advance_ptr(recv_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),                tma_buffer.get_hidden_ptr(),                kNumHiddenBytes);            ptx::tma_store_commit();  // 只把请求送入异步通道, 不等完成        }        __syncwarp();
```
13. 步骤 8c: Scale Factor 散写
```
        // Store SF        // ---- 步骤 8c: scale-factor 按 stride 散列写到 recv_sf ----        //   SF 在 recv_sf 的布局是 [num_tokens, hidden/pack] 且带 stride, 不适合 TMA store        //   所以用 "warp 协同读 SMEM + 逐个有效 lane 按 stride 写" 的方案        if constexpr (kNumSFPacks > 0) {            constexprauto kNumFullIters = kNumSFPacks / 32;                       // 完整 32 lane 轮次            constbool do_last_iter = (kNumSFPacks % 32 != 0) and                  // 余数轮                                      (kNumFullIters * 32 + lane_idx < kNumSFPacks);            EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");            // --- 阶段 1: 32 个 lane 合作把 SMEM 里的 SF packs 读到各自寄存器 ---            constauto smem_src_ptr = tma_buffer.get_sf_ptr();            sf_pack_t reg_src[kNumFullIters + 1];#pragma unroll            for (int k = 0; k < kNumFullIters; ++k)                reg_src[k] = smem_src_ptr[k * 32 + lane_idx];            if (do_last_iter)                reg_src[kNumFullIters] = smem_src_ptr[kNumFullIters * 32 + lane_idx];            // Prepare strides            constauto recv_sf_token_stride_i64  = static_cast<int64_t>(recv_sf_token_stride);            constauto recv_sf_hidden_stride_i64 = static_cast<int64_t>(recv_sf_hidden_stride);            // --- 阶段 2: 对每个有效 dst_tensor_idx 各写一份 SF ---            //【非 Expand】mask=1, 只写 lane 0 对应的那 1 份；            //【Expand】  mask=ptx::gather(dst_tensor_idx >= 0), 多个有效 lane 各写 1 份            auto mask = kDoExpand ? ptx::gather(dst_tensor_idx >= 0) : 1;            while (mask) {                constint valid_lane_idx = __ffs(mask) - 1;                         // 最低位 1 的 lane                // 把 valid_lane_idx 的 dst_tensor_idx 广播给全 warp, 再全 warp 并发写 hidden 维                constauto gmem_dst = math::advance_ptr<sf_pack_t>(                    recv_sf,                    ptx::exchange(dst_tensor_idx, valid_lane_idx)                        * (recv_sf_token_stride_i64 * sizeof(sf_pack_t)));#pragma unroll                for (int k = 0; k < kNumFullIters; ++k)                    gmem_dst[(k * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[k];                if (do_last_iter)                    gmem_dst[(kNumFullIters * 32 + lane_idx) * recv_sf_hidden_stride_i64] =                        reg_src[kNumFullIters];                mask ^= 1 << valid_lane_idx;                                        // 清除已处理 lane            }        }
```
14. 步骤 8d: Top-k weights
```
        // Store the top-k weights        // ---- 步骤 8d: 写 recv_topk_weights ----        //【Expand】: 每个有效 topk 副本各写 1 个标量到 dst_tensor_idx(一维扁平)        //    → combine 反向直接按 slot 取值, 无需乘 topk 偏移        if (kDoExpand and recv_topk_weights != nullptr and dst_tensor_idx >= 0) {            recv_topk_weights[dst_tensor_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];        //【非 Expand】: 所有 topk 权重连续写到 i*num_topk 起始位置 (二维)        } else if (not kDoExpand and recv_topk_weights != nullptr and lane_idx < kNumTopk) {            // For backward, weights are optional            recv_topk_weights[i * kNumTopk + lane_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];        }        __syncwarp();
```
15. 步骤 8e: 源元数据(combine 反向路由)
```
        // Write source token index        // And:        //   - Non-hybrid mode: the source scaleup peer rank index and master top-k lane index        //   - Hybrid mode: the slot index and master top-k lane index        // ---- 步骤 8e: 写 recv_src_metadata ----        //   字段 0: src_global_idx (源 token 在发送端用户 tensor 中的行号)        //   字段 1: 非 hybrid → current_rank_idx*num_topk + master_topk        //          hybrid     → slot_idx*num_topk + master_topk(slot_idx = i - current_rank_start)        //   字段 2..2+topk: expand 下每个 topk 副本对应的 dst_tensor_idx        constexprint kMetadataStride = 2 + kNumTopk;        if (ptx::elect_one_sync()) {            // 字段 0: 源 token 全局行号(来自 TMA 搬进 SMEM 的 metadata)            recv_src_metadata[i * kMetadataStride + 0] = *tma_buffer.get_src_token_global_idx_ptr();            // 字段 1: 根据是否 hybrid 选择打包方式            if constexpr (kNumScaleoutRanks == 1) {                // 非 hybrid: peer_rank_idx * topk + master_topk                recv_src_metadata[i * kMetadataStride + 1] =                    current_rank_idx * kNumTopk + master_src_topk_idx;            } else {                // hybrid: 用段内 slot 下标(作为 combine 定位远端 slot 的依据)                recv_src_metadata[i * kMetadataStride + 1] =                    (i - current_rank_start) * kNumTopk + master_src_topk_idx;            }        }        __syncwarp();        // Write reduction source indices        //【Expand】field 2..2+topk: 每条 topk lane 写自己的 dst_tensor_idx        //    → combine 反向按 stored_topk_slot_idx 一次取回 N 行 reduce        //【非 Expand】不写(combine 通过 recv_topk_idx + 几何位置 i 反推)        if (kDoExpand and lane_idx < kNumTopk)            recv_src_metadata[i * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;        __syncwarp();    }  // end of for-loop over tokens
```
16. 步骤 9: 尾部链表封口 + workspace 清零
```
    // Maintain linked list's ending    // Or you can understand it as writing the tail at once    // ---- 步骤 9: 写 per-channel 链表 -1 结束符, 并清零 tail 指针 ----    //   为下一次 combine 阶段的遍历/写入做准备    if constexpr (kDoCreateLinkedList) {        // 每 lane 处理几个 scaleup_rank(kNumScaleupRanks 可能 > 32)        constexprint kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32);        constauto workspace_layout =            layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);        // 每个 warp 负责一组 channel        for (int i = global_warp_idx; i < kNumChannels; i += kNumSMs * kNumWarps) {#pragma unroll            for (int j = 0; j < kNumScaleupRanksPerLane; ++j) {                // 确保 (j,lane) 对应的 k = j*32 + lane_idx 落在 [0, kNumScaleupRanks)                if (constauto k = j * 32 + lane_idx;                    j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks) {                    // 取出本 (channel, scaleup_rank) 链表尾部节点下标, 写 -1 作终止符                    channel_linked_list[*workspace_layout.get_channel_scaleup_tail_ptr(i, k)] = -1;                    // 清零 tail 指针, 供 combine 阶段重新累计使用                    *workspace_layout.get_channel_scaleup_tail_ptr(i, k) = 0;                }            }            __syncwarp();        }    }}  // end of dispatch_copy_epilogue_impl}  // namespace deep_ep::elastic
```

### 1.3 Dispatch Deterministic Prologue

普通 dispatch 在 Notify warp 中使用 `atomicAdd(psum_num_recv_tokens_per_rank, 1)` 动态抢 slot. 两次运行的 slot 分配顺序可能不同(例如 SM 调度时序抖动、RDMA 到达顺序变化), 导致 recv_buffer 内 token 排列不同.

非确定性会带来一些问题,例如在TML的文章《Defeating Nondeterminism in LLM Inference》中有解释, 实际的问题是**浮点数的加法不满足结合律**. 例如

```
(0.1 + 1e20) - 1e20>>> 00.1 + (1e20 - 1e20)>>> 0.1
```

**Deterministic 模式解法**: 在主 dispatch 之前先跑一个 **prologue kernel**, **按确定顺序** 预分配每个 (token, topk) 的 slot 号, 写入 `dst_buffer_slot_idx` 表；主 dispatch kernel 切换到 `kReuseSlotIndices=true`, 直接查表写, 不再调用 atomicAdd, 从而满足顺序一致性. 两种模式对比如下:

维度

普通 dispatch (Notify warp)

Deterministic Prologue

slot 分配时机

运行时(主 kernel 内)

预先(前置 kernel)

分配原语
`atomicAdd(psum_expert + e, 1)`
4 阶段 grid-sync prefix sum

结果存放
`psum_num_recv_tokens_per_*``dst_buffer_slot_idx[num_tokens, kNumTopk]`
主 dispatch 切换
`kReuseSlotIndices=false``kReuseSlotIndices=true`
两次运行一致性

否

是

性能开销

0

一次额外 kernel + grid-sync

整个执行流程如下所示:

```
输入:  - topk_idx        : [num_tokens, kNumTopk] 每个 token 选中的专家索引  - rank_count_buffer : [kNumSMs, kNumScaleupRanks] 用于 SM 间做 prefix sum输出:  - dst_buffer_slot_idx : [num_tokens, kNumTopk]                          每个 (token, topk) 要写入到 "目标 rank 的 recv 行" 中的 slot 号                          无效 topk (-1) 对应 -1                          有效值 = scaleup_rank_idx * kNumMaxTokensPerRank + local_slot整体执行流程 (4 阶段 cooperative_groups grid sync):  阶段1: 每个 warp 扫自己负责的 token 区间, 按 rank 聚合计数到 warp 私有 smem  阶段2: block 级别 reduce 得到 block 对每个 rank 的总数, 写入全局 rank_count_buffer  阶段3: grid.sync() 后, 每个 SM 计算自身之前所有 SM 的 prefix sum  阶段4: 本 SM 内再累加本 block 先前 warp 的 sum, 二次扫描 token 写出 slot 号
```

具体的代码篇幅有限(其实是偷懒了...)不写了

## 2. Combine Kernel详细分析

### 2.1 Direct Combine

Direct Combine 作用是把各 rank 上 MoE 专家计算的输出(每个源 token 的 topk 副本)**送回源 rank**, 由后续 `combine_reduce_epilogue` 做加权规约, 还原出源 tensor 形状的 `combined_x`.

函数的输入为:

`x`(本 rank expert FFN 输出)

`topk_weights`(非 expand 模式)

`src_metadata`(dispatch 写入的路由表)

`psum_num_recv_tokens_per_scaleup_rank`

输出为在远端 `recv_buffer[kNumTokensInLayout, kNumMaxTokensPerRank]` 对应槽位写满数据 + topk_weights, 它的执行阶段有三大执行分支, 主要是针对 dispatch 阶段是否为 expand 模式`kUseExpandedLayout`和是否允许Combine阶段在发送端先做一次Reduce`kAllowMultipleReduction`所决定的. 传输时可以Direct Dispatch一样, 支持NVLink和RDMA传输.
函数签名
```
__global__ void __launch_bounds__(kNumThreads, 1) combine_impl(    nv_bfloat16* x,                                   // 本 rank 的 combine-pre 数据 (dispatch 之后执行Expert FFN的结果)    float* topk_weights,                              // 非 expand 模式下的 topk 权重 (可空)    int* src_metadata,                                // dispatch 写下的 src_token/src_rank/topk 索引表    int* psum_num_recv_tokens_per_scaleup_rank,       // CPU 不 sync 时的前缀和 fallback    const ncclDevComm_t nccl_dev_comm,    const ncclWindow_t nccl_window,    void* buffer,                                     // 连续的 recv_buffer+send_buffer    void* workspace,                                  // barrier workspace    const int rank_idx,    int num_reduced_tokens)                          
```
1. 公共初始化
首先对warp_idx的处理 `(ptx::get_warp_idx() + rank_idx) % kNumWarps`, 不同 rank 起始 warp 错开, 减小对远端同一槽位的并发冲突.

```
const auto sm_idx       = static_cast<int>(blockIdx.x);constauto thread_idx   = static_cast<int>(threadIdx.x);constauto warp_idx     = (ptx::get_warp_idx() + rank_idx) % kNumWarps;  // rank-rotated 减少热点constauto lane_idx     = ptx::get_lane_idx();constauto global_warp_idx = warp_idx * kNumSMs + sm_idx;//kDoExpandedSend: 仅当 `expand + 不允许本地 reduce` 时为 true, 对应分支"每份 topk 单独发送"语义constexprbool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;// CPU no_sync 时回退用 device 侧前缀和if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)    num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumRanks - 1);
```
2. Buffer布局
![图片](assets/fc1263ac9f13.png)

`kUseRankLayout=true` 时 `kNumTokensInLayout=kNumRanks`: 远端按 rank 分槽, 本 rank 写 `recv_buffer[rank_idx][src_token]`

`kUseRankLayout=false + expand` 时 `kNumTokensInLayout=kNumTopk`: 远端按 topk 分槽, 本 rank 写 `recv_buffer[src_topk_idx][src_token]`

```
// SMEM (TMA 中转 + mbarrier)extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];constauto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);constauto tma_buffer = layout::BufferLayout<true>(    token_layout, kNumWarps, 1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);// 全局内存 bufferconstauto recv_buffer = layout::BufferLayout<false>(    token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, buffer);constauto send_buffer = layout::BufferLayout<false>(    token_layout, kNumRanks, kNumMaxTokensPerRank * (kDoExpandedSend ? kNumTopk : 1),    recv_buffer.get_buffer_end_ptr());// TMA mbarrierptx::arrival_phase phase = 0;constauto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();if (ptx::elect_one_sync())    ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);__syncwarp();// Expand 模式不允许传 topk_weightsif constexpr (kUseExpandedLayout)    EP_DEVICE_ASSERT(topk_weights == nullptr);
```
3.  Gin 初始化与起始 barrier
```
// 每 warp 当一个 channelconstauto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumWarps>(sm_idx, warp_idx);constauto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);// 确保远端 recv_buffer 可写之前做一次全网 barrierconstauto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);comm::gpu_barrier<kIsScaleupNVLink,                  /*kNumScaleoutRanks=*/1,                  kNumRanks,                  kNumSMs,                  kNumThreads,                  kNumQPs,                  kNumTimeoutCycles,                  comm::kCombineTag0, //`kCombineTag0` vs dispatch 使用的不同 tag, 避免同一 workspace 复用时串扰                  /*kWaitRecv=*/false,                  /*kSignalRecv=*/false,                  /*kSignalSend=*/true>(gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);
```
4. 主循环: token 切分
总 token 数均匀分给 `kNumSMs × kNumWarps` 个 warp(每 SM `kNumWarps` 个, 每个 warp 一个 channel)

```
int num_tokens_per_warp = math::ceil_div(num_reduced_tokens, kNumSMs * kNumWarps);const int token_start_idx = num_tokens_per_warp * global_warp_idx;const int token_end_idx   = min(token_start_idx + num_tokens_per_warp, num_reduced_tokens);for (int i = token_start_idx; i < token_end_idx; ++i) {    ...}
```

在主循环内部分为如下几步:
步骤 1: 从 src_metadata 还原三元组
读取 dispatch 写下的 src_metadata, 还原 src_rank / src_token / src_topk, 三元组 `(src_rank_idx, src_token_idx, src_topk_idx)` 唯一定位一个 dispatch 副本, 即本 combine 副本要发回的目标位置.`% kNumMaxTokensPerRank`: dispatch 时原始编码可能带高位 flag, 此处去掉

```
// The master slot index during dispatchconstexpr int kMetadataStride = 2 + kNumTopk;const int src_token_idx     = __ldg(src_metadata + i * kMetadataStride) % kNumMaxTokensPerRank;const int src_rank_topk_idx = __ldg(src_metadata + i * kMetadataStride + 1);const int src_rank_idx      = src_rank_topk_idx / kNumTopk;const int src_topk_idx      = src_rank_topk_idx % kNumTopk;
```
步骤 2: master_token_buffer 地址构造
根据 NVLink 可达性构造 master_token_buffer 的写入地址, 如果nvlink可以访问,则走nvlink_bypass路径, 否则放入send_buffer staging 走RDMA通信.两种路径对照:

路径

master_token_buffer 指向

后续写入

NVLink bypass

远端 `recv_buffer[rank_idx or src_topk_idx][src_token_idx]`

TMA store 直达

RDMA

本地 `send_buffer[src_rank_idx][src_token_idx]`

TMA store→staging→gin.put

```
// Directly to the remote or via RDMAconstbool nvlink_bypass = gin.is_nvlink_accessible<team_t>(src_rank_idx);layout::TokenLayout master_token_buffer = [=]() {    // NVLink bypass    if (nvlink_bypass) {        auto token_buffer = recv_buffer            .get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx)            .get_token_buffer(src_token_idx);        token_buffer.set_base_ptr(            gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx));        return token_buffer;    }    // Use RDMA    return send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx);}();
```
步骤 3: expand 模式读 topk_slot_idx
expand 模式下, 每 lane 从 src_metadata 读出自己 topk 的 dst_tensor_idx

```
// Read source indices for expand modeint stored_topk_slot_idx = -1;if constexpr (kUseExpandedLayout) {    if (lane_idx < kNumTopk)        stored_topk_slot_idx = __ldg(src_metadata + i * kMetadataStride + (2 + lane_idx));    __syncwarp();}
```

其中每个 lane(0..kNumTopk-1)负责读取一个 topk 副本在 **本 rank x 张量** 的偏移(即 dispatch 时 `atomicAdd(psum_expert + e, 1)` 抢到的那个 slot),  `stored_topk_slot_idx = -1` 表示这个 topk 副本 **不落在本 rank**(dispatch 时未分配到本 rank)
步骤 4: 三分支决策
```
auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);auto no_local_reduce   = not kUseExpandedLayout                      or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
```

具体决策逻辑:

条件

分支

语义
`!kUseExpandedLayout`
A

非 expand: 源是整张 x[i], 只需 load→store 一次
`kAllowMultipleReduction`
 且 `__popc(mask)==1`

A

允许 reduce 但只有 1 份有效副本, 等效不需要 reduce
`kUseExpandedLayout + kAllowMultipleReduction + __popc > 1`
B

本地 reduce 多份 topk 副本后一次发送
`kUseExpandedLayout + !kAllowMultipleReduction`
C

每个有效副本各发一份(expanded send)

`reduce_valid_mask = ptx::gather(...)`: warp 内收集每 lane 的"是否是本 rank 的 topk 副本"为 32-bit mask

`__popc(reduce_valid_mask)` = 本 rank 拥有的该 token 的 topk 副本数
分支 A: no_local_reduce(TMA load→store)
```
if (no_local_reduce) {    // ---- 分支 A: 无需本地 reduce, TMA load 源 token -> TMA store 到 master buffer ----    int token_idx_in_tensor = i;    if constexpr (kUseExpandedLayout)        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));    // No reduce    if (ptx::elect_one_sync()) {        constauto load_ptr = math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);        ptx::tma_store_wait();        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);        ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);        ptx::tma_store_commit();    }    __syncwarp();}
```
分支 B: local reduce(expand + multi_reduce)
本地Reduce多份 topk 副本后一次发送, 其中:

`compute_topk_slots`: 把有效 `stored_topk_slot_idx` 按 lane 顺序紧凑到数组前 `__popc(mask)` 位(见 `combine_utils.cuh`)

`combine_reduce`: 用 BF16 hadd / float2 向量化累加多份 `x[slot]` 到 `tma_buffer`

模板参数 `math::constexpr_ceil_div(kNumTopk, kNumRanks)` = "每 rank 最多持有的同一 token 的副本数"

`get_max_unroll_factor<kHiddenVec, 4>()`: 编译期选择展开因子(最大 4), 匹配寄存器压力

`tma_store_fence`: 确保 SMEM reduce 结果对 TMA 可见, 再发 store

```
} else ifconstexpr (kAllowMultipleReduction) {    // ---- 分支 B: 本地 reduce (expand + multi_reduce) ----    // Sort valid top-k indices to front    int topk_slot_idx[kNumTopk];    compute_topk_slots(topk_slot_idx, reduce_valid_mask,        [=](constint& idx) { return ptx::exchange(stored_topk_slot_idx, idx); });    // Reduce into shared memory    constexprint kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();    combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(        lane_idx,        topk_slot_idx,        static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),        /* Get source base */        [=](constint& slot_idx) {            return math::advance_ptr<combine_vec_t>(x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));        },        /* Wait buffer release */        [=]() {            ptx::tma_store_wait();            __syncwarp();        });    ptx::tma_store_fence();    __syncwarp();    // Issue TMA stores    if (ptx::elect_one_sync()) {        ptx::tma_store_1d(master_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);        ptx::tma_store_commit();    }    __syncwarp();}
```
分支 C: expanded send(每副本单独发)
具体执行一下操作:

遍历 `k = 0..kNumTopk-1`: `ptx::exchange(stored_topk_slot_idx, k)` 让 warp 读取第 k 个 lane 的 slot_idx

有效副本条件 `slot_idx >= 0`: 跳过不属于本 rank 的 topk 副本

目标槽固定为 `recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx)`: expand 模式下 `kUseRankLayout=false`, 远端按 topk 维度索引

NVLink bypass: 直接 `tma_store_1d(get_sym_ptr<team_t>(...), ...)` 原位写远端

RDMA: 先 TMA store 到 `send_buffer[src_rank][src_token * kNumTopk + k]` staging(与分支 A/B 的 send_buffer 偏移不同: 每 topk 副本一个槽), 再 `gin.put` 提交 RDMA, 这里 **send_buffer 需要 kNumMaxTokens × kNumTopk 容量**

`tma_store_wait` + `gin.put`: 必须确保 staging 数据已落盘才发 RDMA(RDMA 是 DMA, 源端读内存)

```
} else {// ---- 分支 C: expanded send (expand + 非 multi_reduce) ----#pragma unroll    for (int k = 0; k < kNumTopk; ++k) {        constauto slot_idx = ptx::exchange(stored_topk_slot_idx, k);        if (slot_idx >= 0) {            constauto src_token_ptr = math::advance_ptr<int4>(x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));            constauto token_buffer  = recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx);            if (ptx::elect_one_sync()) {                // Load                ptx::tma_store_wait();                ptx::tma_load_1d(tma_buffer.get_base_ptr(), src_token_ptr, mbarrier_ptr, kNumHiddenBytes);                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);                if (nvlink_bypass) {                    // Write into the same position                    ptx::tma_store_1d(gin.get_sym_ptr<team_t>(token_buffer.get_base_ptr(), src_rank_idx),                                      tma_buffer.get_base_ptr(),                                      kNumHiddenBytes);                    ptx::tma_store_commit();                } else {                    // Write to the RDMA send buffer                    constauto send_token_buffer =                        send_buffer.get_rank_buffer(src_rank_idx).get_token_buffer(src_token_idx * kNumTopk + k);                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(), kNumHiddenBytes);                    ptx::tma_store_commit();                    ptx::tma_store_wait();                    // Issue RDMA                    gin.put<team_t>(token_buffer.get_base_ptr(), send_token_buffer.get_base_ptr(), kNumHiddenBytes, src_rank_idx);                }            }            __syncwarp();        }    }}
```
步骤 5: 写 topk_weights
仅非 expand + 非空: expand 模式禁止 发送 topk_weight , warp 分工为: `lane_idx < kNumTopk` 每 lane 写一份, `master_token_buffer.get_topk_weights_ptr()` 是 token 末尾的权重字段, 最终远端 epilogue 会读这个 weight 做加权 reduce

```
// Write topk weights// ---- 步骤 5: 非 expand 时, 把 topk_weights 写到 master_token_buffer 的 weight 字段 ----if (not kUseExpandedLayout and topk_weights != nullptr and lane_idx < kNumTopk) {    const float value = __ldg(topk_weights + (i * kNumTopk + lane_idx));    master_token_buffer.get_topk_weights_ptr()[lane_idx] = value;}__syncwarp();
```
步骤 6: 非 expand + 非 bypass 的 RDMA put
```
// Wait send buffer's TMA store and issue RDMA send// NOTES: `kDoExpandedSend` mode has already issued// ---- 步骤 6: 非 expand + 非 bypass 时, 由 elect_one 等 TMA 落盘, 发起 RDMA put ----if (not kDoExpandedSend and not nvlink_bypass and ptx::elect_one_sync()) {    ptx::tma_store_wait();    const auto dst_ptr =        recv_buffer.get_rank_buffer(kUseRankLayout ? rank_idx : src_topk_idx)                   .get_token_buffer(src_token_idx).get_base_ptr();    gin.put<team_t>(dst_ptr, master_token_buffer.get_base_ptr(),                    master_token_buffer.get_num_bytes<false>(), src_rank_idx);}
```
收尾 gpu_barrier
wait + signal recv, 确保所有 rank 都完成所有 RDMA/NVLink 写后, 再进入 epilogue. 对比  dispatch 结尾用 `cudaTriggerProgrammaticLaunchCompletion`(PDL)直接触发 epilogue kernel, 而 combine 用传统 gpu_barrier + 后续 launch epilogue

```
// Final barrier to ensure data arrivalcomm::gpu_barrier<kIsScaleupNVLink,                  1, kNumRanks, kNumSMs, kNumThreads, kNumQPs,                  kNumTimeoutCycles,                  comm::kCombineTag1,                  /*kWaitRecv=*/true,                  /*kSignalRecv=*/true,                  /*kSignalSend=*/false>(    gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);}
```

### 2.2 Combine Epilogue

combine 主 kernel 之后的尾段规约, 将 recv_buffer 中的多个副本在 hidden 维上 reduce 成最终 `combined_x`, 并把 `topk_weight` 拷回 `combined_topk_weights`. 同时需要注意, 如果有bias在reduce阶段嵌入.

它包含两种 layout:

comm_token_layout   : dispatch/combine 通信 slot, 含 topk_weight 字段

output_token_layout : 纯 hidden, 对应最终 combined_x

去重策略 (与 kUseExpandedLayout / kAllowMultipleReduction / hybrid 组合):

expand + 非 multi_reduce                    : 每个 topk 副本都是独立的, 不去重

hybrid + 非 expand + 非 multi_reduce        : 按 expert 所属 rank 去重 (kNumExpertsPerRank)

其他                                         : 按 dst_rank_idx 去重

条件

should_deduplicate

deduplicate_key

含义

expand + 非 multi_reduce

false

-

每 topk 副本独立, 不去重

hybrid + 非 expand + 非 multi_reduce

true
`dst_expert_idx / kNumExpertsPerRank`
按 expert 所属 rank 去重

其他

true
`stored_dst_rank_idx`
按 dst_rank_idx 去重

执行流程 (per-warp / per-token):

cudaGridDependencySynchronize 等 combine 主 kernel 落盘 (PDL);

读 combined_topk_idx -> 计算 dst_rank_idx, 根据策略决定去重键;

compute_topk_slots: 把有效副本紧凑到 topk_slot_idx 数组前部;

combine_reduce: 把 N 份 hidden (+ 可选 bias_0/bias_1) 在 SMEM 上 reduce;

TMA store 到 combined_x[token_idx];

读每个 topk 副本对应的 topk_weight 写到 combined_topk_weights.
函数签名
```
__global__ void __launch_bounds__(kNumThreads, 1) combine_reduce_epilogue_impl(    nv_bfloat16* combined_x,              // 最终用户输出 (num_tokens × hidden)    float*       combined_topk_weights,   // 最终 topk 权重 (可空)    topk_idx_t*  combined_topk_idx,       // 原始 topk 专家 id (每 token k 个)    void*        recv_buffer,             // combine 主 kernel 产出的 comm_buffer    void*        bias_0,                  // 可选 bias 张量 (num_tokens × hidden)    void*        bias_1,                  // 可选 bias 张量 (num_tokens × hidden)    const int    num_combined_tokens,     // 本 rank 要 combine 回的 token 数    const int    scaleout_rank_idx,    const int    scaleup_rank_idx)
```
Buffer 布局
```
// SMEM TMA 中转extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];// comm_buffer: combine 主 kernel 输出的通信 slot 布局(含 topk_weight 字段)constauto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);constauto comm_buffer       = layout::BufferLayout<false>(    comm_token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, recv_buffer);// output_buffer: 纯 hidden 的用户输出布局constauto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);constauto output_buffer       = layout::BufferLayout<false>(    output_token_layout, 1, num_combined_tokens, combined_x);// SMEM tma_buffer (每 warp 一份纯 hidden)constauto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)                         .get_rank_buffer(warp_idx).get_token_buffer(0);// Bias layout (与 output 同)constauto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);constauto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);
```

两种 TokenLayout 对照:

Layout

Hidden bytes

SF bytes

Topk bytes

用途
`comm_token_layout``kNumHiddenBytes`
0
`kNumTopk * 4 (f32)`
recv_buffer 每槽
`output_token_layout``kNumHiddenBytes`
0

0

combined_x / bias

`kUseRankLayout=true` 时 `kNumTokensInLayout=kNumRanks`: comm_buffer 第一维按 rank

`kUseRankLayout=false` 时 `kNumTokensInLayout=kNumTopk`: comm_buffer 第一维按 topk

然后Kernel会在这里进行一个PDL同步,  combine 主 kernel 结束后自动触发本 kernel, 只有 `cudaGridDependencySynchronize()` 之后 combine 的 writes 才对本 kernel 可见
主循环: token 切分
每 warp 独立负责一批 token(跨步 = `kNumWarps × kNumSMs`), 由于 `global_warp_idx = warp_idx * kNumSMs + sm_idx`, 连续 token 落在不同 SM, 负载均衡更好

```
for (int token_idx = global_warp_idx;     token_idx < num_combined_tokens;     token_idx += kNumWarps * kNumSMs) {    ...}
```

主循环内包含如下几步
步骤 1: 读 topk_idx → dst_rank_idx
```
// Preprocess all indices// ---- 步骤 1: 读 topk_idx, 映射成 dst_rank_idx (依赖是否 hybrid) ----int stored_dst_rank_idx = -1, stored_dst_expert_idx = -1;//每 lane 一个 topk 副本: `kNumTopk ≤ 32` 保证一 warp 能覆盖EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");if (lane_idx < kNumTopk) {    stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);    stored_dst_rank_idx = stored_dst_expert_idx >= 0        ? stored_dst_expert_idx / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : kNumExpertsPerScaleout)        : -1;}__syncwarp();
```

expert → rank 映射:

Direct(`kNumScaleoutRanks==1`): `dst_rank = dst_expert / kNumExpertsPerRank`(单维 rank 空间)

Hybrid(`kNumScaleoutRanks>1`): `dst_rank = dst_expert / kNumExpertsPerScaleout`(按 scaleout 维聚合, 因为 combine 的 comm_buffer 只按 scaleout 维展开)
步骤 2: 去重策略 + 紧凑 slot
三种去重策略:

分支

场景
`should_deduplicate``deduplicate_key``expand + !multi_reduce`
combine.cuh 分支 C(每 topk 独立送 comm_buffer[k])
`false`
—
`hybrid + !expand + !multi_reduce`
hybrid_combine.cuh forward 的 per-副本转发
`true``dst_expert / kNumExpertsPerRank`
其他(Direct 非 expand / multi_reduce 任意)

本 rank 每 rank 只产出一份
`true``stored_dst_rank_idx`

`compute_topk_slots` 的 slot 编码

`kUseRankLayout=true`(rank-based): slot 紧凑成每 lane 存 `stored_dst_rank_idx`(通过 `ptx::exchange`), 对应 `comm_buffer.get_rank_buffer(dst_rank)`

`kUseRankLayout=false`(topk-based): slot 直接用 lane 序号 `idx`, 对应 `comm_buffer.get_rank_buffer(topk_idx)`(即 combine 主 kernel 写入的 `get_rank_buffer(k)`)

```
// Sort valid top-k indices to front// ---- 步骤 2: 根据 (hybrid / expand / multi_reduce) 组合决定去重策略, 紧凑到 topk_slot_idx ----constauto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {    ifconstexpr (kUseExpandedLayout andnot kAllowMultipleReduction) {        // Activations are never reduced before        return {false, 0};    } elseifconstexpr (kNumScaleoutRanks != 1andnot kUseExpandedLayout andnot kAllowMultipleReduction) {        // Hybrid mode without expanded layout and multiple reduction. Should deduplicate on a per-rank basis        return {true, stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / kNumExpertsPerRank : -1};    } else {        // Should deduplicate on a per-rank (for non-hybrid mode) or a per-scale-rank (for hybrid mode) basis        return {true, stored_dst_rank_idx};    }}();auto reduce_valid_mask = should_deduplicate    ? ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0)    : ptx::gather(stored_dst_rank_idx >= 0);int topk_slot_idx[kNumTokensInLayout];compute_topk_slots(topk_slot_idx, reduce_valid_mask, [=](constint& idx) {    return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;});
```
步骤 3: combine_reduce(含 bias)
combine_reduce 五个参数作用

```
 combine_reduce(     lane_idx,                  // lane 分工     topk_slot_idx,             // 紧凑后的有效 slot 数组(前 popc 位有效)     dst (SMEM),                // 累加目标: SMEM tma_buffer     get_src_base(slot_idx),    // 源 hidden 基址 getter (comm_buffer[slot][token])     wait_buffer_release(),     // 让 TMA store 完成(SMEM 复用前等上次落盘)     bias_0 (可空),             // 偏置 0     bias_1 (可空));            // 偏置 1
```

`wait_buffer_release` : 在每次 reduce 开始前调用, 确保上一轮 TMA store 已落盘、SMEM tma_buffer 可复用

bias 的融合累加: bias 不单独做一次 load/store, 而是直接嵌入 reduce 累加, 省一个 round-trip

```
// Iterate over per-hidden-chunk stage// ---- 步骤 3: combine_reduce 把所有有效副本累加 (+ bias_0/bias_1) 到 SMEM tma_buffer ----usingcombine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;constexprint kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);constexprint kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout>(    lane_idx,    topk_slot_idx,    static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),    /* Get source base */    [=](constint& slot_idx) {        returnstatic_cast<combine_vec_t*>(            comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr());    },    /* Wait buffer release */    [=]() {        ptx::tma_store_wait();        __syncwarp();    },    /* Bias 0 */ bias_0 == nullptr ? nullptr        : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr()),    /* Bias 1 */ bias_1 == nullptr ? nullptr        : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr()));//确保 SMEM reduce 结果对后续 TMA store 可见(SMEM 的跨线程写要 fence 才能被 TMA 看到).         ptx::tma_store_fence();__syncwarp();
```
步骤 4: TMA store 到 combined_x
```
// Issue TMA copy// ---- 步骤 4: TMA store SMEM 规约结果到 combined_x[token_idx] ----if (ptx::elect_one_sync()) {    ptx::tma_store_1d(output_buffer.get_token_buffer(token_idx).get_base_ptr(),                      tma_buffer.get_base_ptr(),                      kNumHiddenBytes);    ptx::tma_store_commit();}__syncwarp();
```
步骤 5: 写 topk_weights
两种 layout 下的 slot 选择

配置
`comm_buffer`
 第一维索引

为什么这样选
`kUseRankLayout=true``stored_dst_rank_idx`
每 rank 一槽, 直接用 lane 本身的 dst_rank
`kUseRankLayout=false``master_lane_idx`
每 topk 一槽, 但同 dst_rank 的多 lane 要取同一 master

```
// Write top-k weights// ---- 步骤 5: 把每个 topk 副本的 weight 从 comm_buffer 拷贝到 combined_topk_weights ----//   rank 布局: 直接用 stored_dst_rank_idx; topk 布局: 用 master_lane (同一 dst_rank 的第一 lane)if (combined_topk_weights != nullptr) {    constauto master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));    if (lane_idx < kNumTopk) {        float value = 0;        if (stored_dst_rank_idx >= 0) {            constauto dst_ptr = comm_buffer                .get_rank_buffer(kUseRankLayout ? stored_dst_rank_idx : master_lane_idx)                .get_token_buffer(token_idx)                .get_topk_weights_ptr() + lane_idx;            value = *dst_ptr;        }        combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;    }    __syncwarp();}
```