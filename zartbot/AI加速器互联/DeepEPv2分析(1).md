# DeepEPv2分析(1)

> 作者: zartbot  
> 日期: 2026年5月4日 11:27  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498240&idx=2&sn=ebdf052f7c54fd655ea040de7d228f3c&chksm=f995eac2cee263d435661b9ce091c52bbea6f7f7824cd2c4a64b7f294b48ebbe0de3ae6e853c#rd

---

### TL;DR

以前对老版本的DeepEP有过一个分析[《分析一下EP并行和DeepSeek开源的DeepEP代码》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493292&idx=1&sn=7af7db0f3d78f0fb52dc847934c7800e&scene=21#wechat_redirect), 而这次DeepEP v2 更新了蛮多内容. 从通信上扩展支持了Engram, PP, CP以及AGRS(All-Gather和ReduceScatter), 所以从名字上也改为了`DeepEveryParallel`. V1采用了NVSHMEM, 而在V2中采用了NCCL GIN的后端, 显著降低 SM 占用并扩展可扩展性, 同时引入了ElasticBuffer, 当前支持纯GPU后端, 未来会计划支持CPU和混合的后端.

由于篇幅太长, 分为两篇, 第一篇介绍ElasticBuffer结构和内存布局以及其它非EP的实现, 第二篇将专门详细介绍EP并行中的dispatch/combine实现, 本文目录如下:

```
1. ElasticBuffer1.1 初始化Buffer1.2 内存布局2. Barrier实现2.1 函数调用2.2 Kernel主体2.3 ScaleUP barrier2.4 ScaleOut barrier2.5 Hybrid barrier3. PP并行通信实现3.1 初始化配置3.2 函数调用3.3 Buffer分配3.4 Send流程4. Engram4.1 Buffer分配4.2 engram write4.3 engram fetch5. AGRS5.1 Buffer分配5.2 Session based上下文管理5.3 all-gather流程
```

## 1. ElasticBuffer

### 1.1 初始化Buffer

在初始化Buffer的过程中, DeepEPv2统一使用了Symmetric Memory和NCCL Gin的backend, 整个初始化的调用过程如下所示.  NCCL Gin 的内容可以参考[《NCCL Gin & Symmetric Memory》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498168&idx=1&sn=adfe6ba01ff8cdbe20cdf5aeb655d0cb&scene=21#wechat_redirect).

具体来说, 首先通过`get_nccl_comm_handle(group)`拿到 NCCL comm 裸指针, 然后调用`calculate_elastic_buffer_size`估计 buffer 所需要的字节数. 然后在 C 的backend `ElasticBuffer`中调用`NCCLSymmetricMemoryContext`构建对称内存. 它会通过`ncclCommQueryProperties`检查 Gin 的可用性, 然后根据实际的物理拓扑选择是否支持多平面和多轨道组网.

关于RDMA, 它还会计算`num_allocated_qps`, 首先需要一个独立的notify QP, 然后如果是Hybrid模式并支持fast-RDMA-atomic, 则需要64channel QP共计 65个QP, 如果不支持fast-RDMA-atomic, 则需要翻倍的QP数目, 总计QP数目为64channel * 2 + 1 Notify = 129. 查看了一下代码`check_fast_rdma_atomic_support`中仅CX7(MT4131)或者更新的网卡支持, 吐个槽Mellanox的atomic实现也是有一些问题的, 后面抽空看看CX7改了什么. 然后根据这些结果创建devComm:

```
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;if (num_ranks > 1 and !EP_DISABLE_GIN) {    reqs.ginContextCount    = num_allocated_qps;          // QP 池大小    reqs.ginExclusiveContexts = true;                     // QP 独占    reqs.ginQueueDepth      = 1024;    reqs.ginTrafficClass    = sl_idx;                     // RDMA Service Level    reqs.ginSignalCount     = num_ranks + 2 * 2;          // 自定义 barrier 用的额外信号    reqs.ginConnectionType  = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL;}ncclDevCommCreate(comm, &reqs, &dev_comm);
```

接下来会根据LSA的Rank和Size推导scaleup和scaleout的rank分布

```
// Physicalnum_nvl_ranks = dev_comm.lsaSize;nvl_rank_idx = dev_comm.lsaRank;num_rdma_ranks = num_ranks / num_nvl_ranks;rdma_rank_idx = rank_idx / num_nvl_ranks;EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0               and nvl_rank_idx == rank_idx % num_nvl_ranks);EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);// Logicalif (allow_hybrid_mode) {    num_scaleout_ranks = num_rdma_ranks;   scaleout_rank_idx = rdma_rank_idx;    num_scaleup_ranks  = num_nvl_ranks;    scaleup_rank_idx  = nvl_rank_idx;} else {    num_scaleout_ranks = 1;                scaleout_rank_idx = 0;    num_scaleup_ranks  = num_ranks;        scaleup_rank_idx  = rank_idx;}is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;
```

然后, 对称内存的分配和注册CommWindow直接调用的NCCL, 如下所示:

```
NCCL_CHECK(ncclMemAlloc(&raw_window_ptr, size));                          // 分配物理内存NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, size, &window,   // 注册为跨 rank 窗口                                  NCCL_WIN_DEFAULT));
```

然后处理NVLink peer的指针

```
ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr);    // 本 rank 在窗口上的有效地址nvl_window_ptrs.resize(num_nvl_ranks);for (int i = 0; i < num_nvl_ranks; ++i)    ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]);          // 所有 NVLink peer 的同偏移地址
```

`mapped_window_ptr`: 本 rank 后续所有操作的"基址".

`nvl_window_ptrs[i]`: 第 `i` 个 NVL peer 在本地虚拟地址空间中可直接访问的对应地址；后续 `get_sym_ptr(local, peer)` 只需用 `peer_base + (local - mapped_window_ptr)` 即可.

最后得到所有的字段

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `comm` | `ncclComm_t` | host 侧 NCCL comm |
| `dev_comm` | `ncclDevComm` | device 侧设备 comm(含 GIN context 池) |
| `rank_idx / num_ranks` | `int` | 全局坐标 |
| `nvl_rank_idx / num_nvl_ranks` | `int` | NVLink 物理域坐标(来自 `dev_comm.lsaRank/Size`) |
| `rdma_rank_idx / num_rdma_ranks` | `int` | RDMA 物理域坐标 |
| `scaleout_rank_idx / num_scaleout_ranks` | `int` | 逻辑 scaleout 域 |
| `scaleup_rank_idx / num_scaleup_ranks` | `int` | 逻辑 scaleup 域 |
| `is_scaleup_nvlink` | `bool` | scaleup 是否完全等同于 NVLink 域 |
| `num_allocated_qps` | `int` | 预留 QP 数 |
| `raw_window_ptr` | `void*` | `ncclMemAlloc` 返回的原始地址 |
| `window` | `ncclWindow_t` | 注册的窗口句柄 |
| `mapped_window_ptr` | `void*` | 本 rank 访问窗口的基址 |
| `nvl_window_ptrs` | `vector<void*>` | 所有 NVL peer 的同偏移基址(`get_sym_ptr` 用) |

整个过程如下:

![图片](assets/71e582511b62.png)

### 1.2 内存布局

在内存布局中切分workspace和buffer

```
workspace = this->nccl_context->mapped_window_ptr;workspace_layout_wo_expert = std::make_shared<layout::WorkspaceLayout>(    workspace, nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks, /*num_experts=*/0);buffer = static_cast<uint8_t*>(workspace) + layout::WorkspaceLayout::get_num_bytes();
```

![图片](assets/97f107cd7d08.png)

`workspace_layout_wo_expert`: "without expert 信息"的 layout(`num_experts=0`). 这是因为构造期还不知道 MoE 的 expert 数；需要 expert 相关偏移的 API(如 dispatch)会在运行时临时构造带 expert 的 `WorkspaceLayout`. 但所有"非 expert 维"的区域(barrier、rank count、channel metadata、PP、AGRS signals)此时已全部可用. 它包含的11个Region如下:

| # | 偏移起点 | 区域 | 字节 |
| --- | --- | --- | --- |
| 1 | 0 | NVL barrier counter + signals | `16` |
| 2 | 16 | notify reduction workspace | `(1024+2048)*8 = 24576` |
| 3 | ... | scaleup rank count send/recv | `1024*8*2 = 16384` |
| 4 | ... | scaleup expert count send/recv | `2048*8*2 = 32768` |
| 5 | ... | scaleup atomic sender counter | `1024*4 = 4096` |
| 6 | ... | scaleout rank count send/recv | `1024*4*2 = 8192` |
| 7 | ... | scaleout expert count send/recv | `2048*4*2 = 16384` |
| 8 | ... | scaleout channel metadata | `1024*1280*8 ≈ 10 MB` |
| 9 | ... | channel scaleup tail | `1024*1280*4 ≈ 5 MB` |
| 10 | ... | PP send/recv counts | `2*2*8 = 32` |
| 11 | ... | AGRS signals | `(32+1)*1024*4 = 135168` |
| — | — | align to 32 B | — |

另外针对WorkspaceLayout, 他还会在host侧创建一个映射

```
        // Allocate host workspaces        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
```

] 主 `buffer` 由多种原语共享, 按功能划分出**可动态调整的子区域**(切分时机取决于当前运行的 API):

![图片](assets/6b281b97aca3.png)

## 2. Barrier实现

### 2.1 函数调用

Python的入口为`ElasticBuffer.barrier`

```
def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:    self.runtime.barrier(use_comm_stream, with_cpu_sync)
```

函数有两个开关:

`use_comm_stream`: `True` 时 barrier kernel 跑在 `comm_stream`, 否则跑在当前 compute stream；

`with_cpu_sync`: `True` 时 barrier 前后各加一次 `cudaDeviceSynchronize()`

具体实现如下, 这一层只管流级协同, 真正的跨 rank 同步全在 kernel 内部完成.

```
void barrier(const bool& use_comm_stream, const bool& with_cpu_sync) const {    constauto compute_stream = at::cuda::getCurrentCUDAStream();    constauto stream = use_comm_stream ? comm_stream : compute_stream;    // 让 comm_stream 等 compute_stream 已入队的工作    if (use_comm_stream)        stream_wait(comm_stream, compute_stream);    // CPU sync    if (with_cpu_sync)        cudaDeviceSynchronize();    // launch barrier kernel    launch_barrier(nccl_context->dev_comm, nccl_context->window,                   workspace,                   nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,                   nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,                   num_gpu_timeout_cycles,                   nccl_context->is_scaleup_nvlink,                   stream);    // CPU sync    if (with_cpu_sync)        cudaDeviceSynchronize();    // compute_stream 反向等 comm_stream    if (use_comm_stream)        stream_wait(compute_stream, comm_stream);}
```

### 2.2 Kernel主体

然后在launch barrier kenerl时, 需要判断SM数和线程数:

**1 或 2 个 SM**: 纯 scaleup(单节点)用 1 SM 即可；hybrid(多节点)必须用 **2 SM 才能并行跑 scaleup + scaleout barrier**；

**512 线程/block**: 满足 `kNumRanks <= kNumThreads`(每个 rank 一个线程来写 signal)；

**cooperative=true**: kernel 会用 `this_grid().sync()` 跨 SM 同步, 所以要以 cooperative launch；

**JIT 编译**: 所有维度 (`num_scaleout_ranks/num_scaleup_ranks/is_scaleup_nvlink/num_timeout_cycles`) 都是模板参数, 同一 (shape, topology) 只编一次.

```
constexpr auto kNumThreads = 512;constauto num_sms = num_scaleout_ranks > 1 ? 2 : 1;   // hybrid 才需要 2 SMconst BarrierRuntime::Args args = {    .is_scaleup_nvlink = is_scaleup_nvlink,    .num_scaleout_ranks = num_scaleout_ranks,    .num_scaleup_ranks = num_scaleup_ranks,    .num_timeout_cycles = num_timeout_cycles,    .nccl_dev_comm = nccl_dev_comm,    .nccl_window = nccl_window,    .workspace = workspace,    .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,    // cluster_dim=1, cooperative=true(需要 grid.sync())    .launch_args = jit::LaunchArgs(num_sms, kNumThreads, 0, 1, true)};constauto code = BarrierRuntime::generate(args);constauto runtime = jit::compiler->build("barrier", code);BarrierRuntime::launch(runtime, args, stream);
```

具体的实现在`/deep_ep/include/deep_ep/impls/barrier.cuh`中

```
template <bool kIsScaleupNVLink, int kNumSMs, int kNumThreads,          int kNumScaleoutRanks, int kNumScaleupRanks, int64_t kNumTimeoutCycles>__global__ void __launch_bounds__(kNumThreads, 1)barrier_impl(...) {    constauto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, 0);    constauto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);    comm::gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,                      kNumSMs, kNumThreads,                      comm::kFlushAllAllocatedQPs,       // flush 所有 QP                      kNumTimeoutCycles,                      comm::kKernelBarrierTag,           // tag=1                      false, false, false>(              // kFlushStores/kSyncAtStart/kSyncAtEnd 都关            gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);}
```

然后它会调用`comm::gpu_barrier`, 注意三个关键模板参数:

`kFlushStores = false`: barrier 本身不写业务数据, 无需做 `tma_store_commit/wait`；

`kSyncAtStart = false`: kernel 入口不需要 `grid.sync()`(没有待发布的 store)；

`kSyncAtEnd = false`: 出口也不再强制 grid 同步(CPU 侧 `cudaDeviceSynchronize` 或 stream event 会接管).

```
do_scaleout &= kNumScaleoutRanks > 1;   // 多节点才需要do_scaleup  &= kNumScaleupRanks  > 1;   // 单 GPU 不需要if (do_scaleup && do_scaleout) {    // Hybrid: 两个 SM 并行做    if (sm_idx == 0)  scaleup_barrier_wo_local_sync(...);   // SM0 跑 scaleup    else              scaleout_barrier_wo_local_sync(...);  // SM1 跑 scaleout} else if (do_scaleup)  {  // 单节点 / 纯 NVLink 域    scaleup_barrier_wo_local_sync(...);} else if (do_scaleout) {  // 纯 RDMA    scaleout_barrier_wo_local_sync(...);}
```

我们可以看到:

scaleup barrier 用 **workspace 里的对称 signal 变量**(NVLink LD/ST)或 **GIN 同 team 的 signal**；

scaleout barrier 用 **GIN 跨节点的 team signal**；

两者走**不同通道(NVLink memory vs RDMA QP)**, 互不干扰, 物理并行是最大化吞吐的关键.

### 2.3 ScaleUP barrier

它包含两种实现:

```
template <bool kIsScaleupNVLink, ...>void scaleup_barrier_wo_local_sync(...) {    if constexpr (kIsScaleupNVLink)        nvlink_barrier_wo_local_sync<kNumScaleupRanks, ...>(...);   // 使用NVLink LD/ST    else        gin_barrier_wo_local_sync<..., ncclTeamTagWorld, ...>(...); // 使用GIN}
```

下面我们先来看 `nvlink_barrier_wo_local_sync`,它采用workspaceLayout中预留的字段:

```
// 开头 8B: phase counter(u64)unsigned long long* get_nvl_barrier_counter_ptr() { return workspace; }// 紧接两个 4B signal 槽: phase=0 和 phase=1 各一个int* get_nvl_barrier_signal_ptr(int phase) { return workspace + (2+phase)*sizeof(int); }
```

算法流程如下所示:

```
// 只用 1 个 SMif (kNumSMs > 1 && sm_idx > 0) return;// 读取状态: 低位是 phase (0/1), 次位是 signconstint status = (*counter_ptr) & 3;constint phase  = status & 1;constint sign   = status >> 1;// 每个线程负责一个 rank: 向对应 peer 的 signal 槽做 atomic add/subif (thread_idx < kNumRanks) {    auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(        workspace.get_nvl_barrier_signal_ptr(phase), thread_idx);    ptx::red_add_rel_sys(dst_ptr, sign ? -1 : +1);  // release 语义原子加}__syncthreads();// thread_idx=0 把 counter +1(phase、sign 会翻转: 0→1→2→3→0)if (thread_idx == 0)    atomicAdd(counter_ptr, 1);// 忙等直到本 rank 的 signal 槽达到目标constauto target = sign ? 0 : kNumRanks;timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=]{    auto signal = ptx::ld_acquire_sys<int>(workspace.get_nvl_barrier_signal_ptr(phase));    return signal == target;  // 到达即通过});
```

首先它使用了Symmetric Memory LSA进行寻址, `gin.get_sym_ptr<ncclTeamTagLsa>(local_ptr, peer_rank)` 把本地 signal 地址翻译成"peer 在 LSA 域中对应偏移的 virtual addr"并可以通过NVLink直接进行LD/ST. N 个线程并行给 N 个 peer 的**同一 signal 槽**做 atomic add, 每个 peer 都会看到自己的槽被加 `kNumRanks` 次.  然后设计了双Phase的方式, 避免 barrier 连续调用时的 ABA: 上次 barrier 用 phase 0、这次用 phase 1.

然后关于sign反转, 上次用 `+1` 这次用 `-1`；不需要清零槽位即可复用.  具体来说:

`sign=0` 时加 1、等 `== kNumRanks`

`sign=1` 时加 -1、等 `== 0`.

槽位值在 `[0, kNumRanks]` 之间乒乓震荡, 下次 barrier 刚好从当前值开始反向计数, 零成本复位. 内存语义上, 采用了**release/acquire 语义**: `red_add_rel_sys` 是 system-scoped release 原子, `ld_acquire_sys` 是 system-scoped acquire load, 组合等价于 happens-before 传递, 保证 barrier 前的 store 对 barrier 后的 load 可见.

另外, 它还可以选择在ScaleUP域内使用 `gin_barrier_wo_local_sync` GIN的方式进行barrier, 这主要是`is_scaleup_nvlink=false`只有RDMA的场景, 例如一些RTX 6000pro这类的PCIe卡, 这种场景相对少见, 具体的GIN的调用和ScaleOut barrier相同, 我们在下一节中进行解释.

### 2.4 ScaleOut barrier

scaleout 走 NCCL GIN(GPU Initiated Networking)的 signal API, 完成跨节点 RDMA 同步. `gin_barrier_wo_local_sync`首先会有一个flush阶段,

```
if constexpr (kFlushStores) {    for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps) {        ncclGin(dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA)            .flush(ncclCoopWarp());   // 每个 warp 负责一批 QP 的 DB flush    }    (gridDim.x > 1) ? this_grid().sync() : __syncthreads();}
```

`kNumQPs == kFlushAllAllocatedQPs(-1)` → 运行时取 `nccl_dev_comm.ginContextCount`, flush 所有 QP； `ncclGin.flush(ncclCoopWarp())` 让 warp 协作组把**之前所有 posted work(put/atomic)落到对端**, 然后全 grid 同步确保**所有 SM 都 flush 完**后, 再进 signal 阶段.

需要注意的是 barrier 本身 `kFlushStores=false`, 这段会被跳过, 而在EP并行中 dispatch/combine 结尾调 `gpu_barrier` 时才需要.

然后在Signal阶段的的处理方式如下, 它只需要在SM0处理:

```
if (sm_idx == 0) {    // team: world(full互连的scaleout)或 rail(多轨道的模式)    constauto team = is_world ? ncclTeamWorld(dev_comm) : ncclTeamRail(dev_comm);    const ncclGin gin(dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);  // QP 0 专用    // (a) 发: 每个线程给 team 里 rank i 发 signal, signal_id = 本 rank 的 rank_idx    for (int i = thread_idx; i < kNumRanks; i += kNumThreads)        gin.signal(team, i, ncclGin_SignalInc{static_cast<ncclGinSignal_t>(rank_idx)});    // (b) 等: 本 rank 检查 team 里每个 rank i 发来的 signal    for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {        constauto shadow_ptr = gin.getSignalShadowPtr(i);        constauto target = ++(*shadow_ptr);  // shadow 本地累计, 每次 barrier +1        constauto gdaki = static_cast<ncclGinGdakiGPUContext*>(gin._ginHandle) + gin.contextId;        constauto signal_ptr = reinterpret_cast<uint64_t*>(            __ldg(&gdaki->signals_table.buffer)) + i;        timeout_while<kNumTimeoutCycles>([=](bool is_last_check) {            auto signal = ptx::ld_acquire_sys<uint64_t>(signal_ptr);            return signal >= target;   // ≥ shadow 就放行        });    }}
```

team选择上和实际的RDMA物理连接拓扑相关

`ncclTeamTagWorld`用于标准的RDMA互连拓扑 → scaleout 内所有 rank

`ncclTeamTagRail`用于多轨道部署的方式 → scaleout 同 rail 内所有 rank

发送时使用QP0这个独立的用于notify的QP发送, 避免和其它data QP混用导致的阻塞和延迟. 另外, barrier需要对所有rank进行发送, 并使用多个线程并行发送的方式.

在接收上设计了一个Shadow Counter, `getSignalShadowPtr(i)` 是本地影子计数, 每次 barrier 先 `++` 再比较, 执行时采用了`ld_acquire_sys`, 保证 signal 达到目标后, signal 之前的 remote RDMA put 也都已全局可见.

### 2.5 Hybrid barrier

针对常见的集群, 既有ScaleUP的连接又有ScaleOut连接, 采用两个SM并行发送的方式.

```
if (do_scaleup && do_scaleout) {    if (sm_idx == 0) {        // SM0: scaleup barrier(NVLink 对称内存 phase+sign 协议)        scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, ...>(            gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);        if constexpr (kFlushStores)            this_grid().sync();  // 与 SM1 的 flush→signal 之间的 grid 同步对齐    } else {        // SM1: scaleout barrier(GIN rail team signal)        scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs-1, ...>(            gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx-1, thread_idx);    }}
```

## 3. PP并行通信实现

### 3.1 初始化配置

首先可以通过`get_pp_buffer_size_hint`估算buffer大小, 两个 `*2` 分别对应 **send/recv 双缓冲** 和 **prev/next 两个方向**, 因此每 rank 共有 `4 × inflight` 个 slot.

```
@staticmethoddef get_pp_buffer_size_hint(    num_max_tensor_bytes: int, num_max_inflight_tensors: int) -> int:    # Align with `LDG.256`    num_max_tensor_bytes = align(num_max_tensor_bytes, 32)    # PP 环形通信，每 rank 的 buffer 需要 4 组：    #          (send, recv) × (prev, next) × num_max_inflight_tensors × 对齐后 bytes    return num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2
```

然后在`pp_set_config`中配置, 调用参数包括(num_max_tensor_bytes, num_max_inflight_tensors). 在设置时会同时计算prev/next的rank index

```
void pp_set_config(const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors) {    // Flush previous operations    barrier(false, true);    EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);    EP_HOST_ASSERT(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2 <= num_buffer_bytes);    this->prev_rank_idx = (nccl_context->rank_idx + nccl_context->num_ranks - 1) % nccl_context->num_ranks;    this->next_rank_idx = (nccl_context->rank_idx + 1) % nccl_context->num_ranks;    this->num_max_pp_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, 32);    this->num_max_pp_inflight_tensors = num_max_inflight_tensors;}
```

`barrier(false, true)`: `use_comm_stream=false`(在当前计算流上做 barrier)、`with_cpu_sync=true`(barrier 前后各插 `cudaDeviceSynchronize`), **彻底排空之前 inflight 的所有 kernel 与 RDMA, 把 send/recv 计数器和 Gin signal 推到一致状态**. 因此 `pp_set_config` 必须在首次 PP 通信之前调用, 并且每次改参数都会付出全局barrier的代价.

### 3.2 函数调用

然后完成初始化后, 就可以调用send/recv了

```
def pp_send(self, t: torch.Tensor, dst_rank_idx: int, num_sms: int = 0) -> None:    self.runtime.pp_send(t, dst_rank_idx, num_sms)def pp_recv(self, t: torch.Tensor, src_rank_idx: int, num_sms: int = 0) -> None:    self.runtime.pp_recv(t, src_rank_idx, num_sms)
```

执行前会先做三个检查, 三条断言分别保证: 已调用过 `pp_set_config`、张量合法且不超限、对端必须是环上相邻 rank. `pp_recv` 的处理完全对称, 下面以send为例.

```
void pp_send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {    EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0and num_max_pp_inflight_tensors > 0);    EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);    EP_HOST_ASSERT(dst_rank_idx == prev_rank_idx or dst_rank_idx == next_rank_idx);    launch_pp_send(        nccl_context->dev_comm, nccl_context->window,        x.data_ptr(), x.nbytes(),        buffer, workspace,        nccl_context->rank_idx, dst_rank_idx, nccl_context->num_ranks,        num_max_pp_tensor_bytes,        num_max_pp_inflight_tensors,        // num_sms == 0 时默认使用 device_runtime->get_num_sms()全量 SM        num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,        num_gpu_timeout_cycles,        jit::device_runtime->get_num_smem_bytes(),        at::cuda::getCurrentCUDAStream()    );}
```

最终通过 `launch_pp_send` 走 JIT 编译, 并以`(num_sms, num_ranks, smem_bytes, timeout_cycles)` 四个常量模板实例化kernel. launch_args`LaunchArgs(num_sms, 32, smem_bytes, 1, true)` 表示 num_sms 个 block × 32 线程(即 1 个 warp)、动态共享内存 = smem_bytes、cooperative=true.

```
static std::string generate_impl(const Args& args) {    return fmt::format(R"(#include <deep_ep/impls/pp_send_recv.cuh>using namespace deep_ep::elastic;static void __instantiate_kernel() {{    auto ptr = reinterpret_cast<void*>(&pp_send_impl<{}, {}, {}, {}>);}})", args.launch_args.grid_dim.first,    args.num_ranks,    args.num_smem_bytes,    args.num_timeout_cycles);}// ...const PPSendRuntime::Args args = {    // ...    .launch_args = jit::LaunchArgs(num_sms, 32, num_smem_bytes, 1, true)};constauto code = PPSendRuntime::generate(args);constauto runtime = jit::compiler->build("pp_send", code);PPSendRuntime::launch(runtime, args, stream);
```

### 3.3 Buffer分配

首先通过下面的函数返回 `(local_idx_in_dst, dst_idx_in_local)`:

当 `dst == next` → `(0, 1)`: 即本 rank 在对端视角里是 prev(用 slot 0 接收)；对端在本 rank 视角里是 next(占本地 slot 1).

当 `dst == prev` → `(1, 0)`: 对称情形.

```
template <int kNumRanks>__device__ __forceinline__ std::pair<int, int> get_buffer_offset(    const int& src_rank_idx, const int& dst_rank_idx) {    const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;    return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);}
```

这个返回值直接驱动 `buffer` 的 4 段分段:

| 段号 = 偏移系数 | 表达式 | 含义 |
| --- | --- | --- |
| 0 | `(local_idx_in_dst + 0) * inflight`(next 视角 0) | 本 rank 从 **next** 接收的 recv 区 |
| 1 | `(local_idx_in_dst + 0) * inflight`(prev 视角 1) | 本 rank 从 **prev** 接收的 recv 区 |
| 2 | `(dst_idx_in_local + 2) * inflight`(prev 视角 0+2) | 本 rank 发往 **prev** 的 send uffer |
| 3 | `(dst_idx_in_local + 2) * inflight`(next 视角 1+2) | 本 rank 发往 **next** 的 send buffer |

然后还有两个counter存放在workspaceLayout内

```
__forceinline__ __device__ __host__ int64_t* get_pp_send_count_ptr(const int& offset) const {    const auto base_ptr = math::advance_ptr<int64_t>(        get_channel_scaleup_tail_ptr(0, 0),        kNumMaxRanks * kNumMaxChannels * sizeof(int));    return base_ptr + offset;}__forceinline__ __device__ __host__ int64_t* get_pp_recv_count_ptr(const int& offset) const {    const auto base_ptr = math::advance_ptr<int64_t>(        get_pp_send_count_ptr(0), 2 * sizeof(int64_t));    return base_ptr + offset;}
```

即各自 2 个 int64: `offset=0/1` 对应 prev/next 的已发/已收数量. NCCL Gin 信号则按两组划分:

`signal = kNumRanks + offset`: **data-ready**(发端 `gin.put` 携带的 SignalInc, 触发后收端可消费)；

`signal = kNumRanks + offset + 2`: **slot-release**(收端消费完 `gin.signal` 告诉发端这个 slot 空了).

整体收发交互流程如下:

![图片](assets/7226d11ac261.png)

### 3.4 Send流程

具体实现在`deep_ep/include/deep_ep/impls/pp_send_recv.cuh`中:

```
template <int kNumSMs, int kNumRanks, int kNumSmemBytes, int64_t kNumTimeoutCycles>//__launch_bounds__(32, 1) 表示每 block 32 线程、1 个 warp, 对应 launch_args 的 32__global__ void __launch_bounds__(32, 1)pp_send_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,             void* x, constint64_t num_x_bytes,             void* buffer, void* workspace,             constint rank_idx, constint dst_rank_idx,             constint64_t num_max_tensor_bytes,             constint num_max_inflight_tensors) {    constauto sm_idx = static_cast<int>(blockIdx.x);    constauto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);    constauto [local_idx_in_dst, dst_idx_in_local] = get_buffer_offset<kNumRanks>(rank_idx, dst_rank_idx);    // Gin handle    constauto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);    // Buffer offsets    constauto send_count_ptr = workspace_layout.get_pp_send_count_ptr(dst_idx_in_local);    constauto send_count = __ldg(send_count_ptr);    constauto slot_idx = send_count % num_max_inflight_tensors;    auto send_buffer_ptr = math::advance_ptr(        buffer, ((dst_idx_in_local + 2) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);    auto recv_buffer_ptr = math::advance_ptr(        buffer, ((local_idx_in_dst + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);    // Wait buffer slot release and do TMA    if (ptx::elect_one_sync()) {        check_signal<kNumTimeoutCycles>(            gin,            static_cast<ncclGinSignal_t>(kNumRanks + dst_idx_in_local + 2),            send_count - num_max_inflight_tensors + 1,            []() { printf("DeepEP PP send timeout, recv buffer is full"); }        );        tma_copy<kNumSMs, kNumSmemBytes>(x, send_buffer_ptr, num_x_bytes, sm_idx);    }    cooperative_groups::this_grid().sync();    // Issue RDMA put    if (sm_idx == 0and ptx::elect_one_sync()) {        gin.put<ncclTeamTagWorld>(            recv_buffer_ptr,            send_buffer_ptr,            num_x_bytes, dst_rank_idx,            0,            ncclGin_SignalInc(static_cast<ncclGinSignal_t>(local_idx_in_dst + kNumRanks)));        *send_count_ptr += 1;    }}
```

整个流程如下:

定位 slot: `__ldg(send_count_ptr)` 读本地已发出计数(每 put 完 +1), `slot_idx = send_count % inflight` 得到环形队列写入位置.

两侧指针:

`send_buffer_ptr` 指向本地 send 暂存段(段 2 或 3), 由 `dst_idx_in_local + 2` 选出；

`recv_buffer_ptr` 指向对端 recv 段(段 0 或 1), 由 `local_idx_in_dst + 0` 选出. 两者的 `slot_idx` 相同, 保证跨 rank 对齐.

反向流控(等 slot 腾出): 用 `ptx::elect_one_sync()` 选出 warp 内唯一一条线程调用 `check_signal`, 具体流程如下:

参数 `signal_idx = kNumRanks + dst_idx_in_local + 2`(release-signal)

参数 `target = send_count - inflight + 1`, 只有当对端至少已 release 过 `send_count - inflight + 1` 次, 这个 slot 才不再被对端占用；

超时则打印 "DeepEP PP send timeout, recv buffer is full".

```
   template <int64_t kNumTimeoutCycles, typenametimeout_print_t>   __device__ __forceinline__ void check_signal(       const handle::NCCLGin& gin,       const ncclGinSignal_t& signal_idx,       const int64_t& target,       const timeout_print_t& timeout_print) {       constauto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin.gin._ginHandle) + gin.gin.contextId;       constauto signal_ptr = reinterpret_cast<int64_t*>(           __ldg(reinterpret_cast<int64_t*>(&gdaki->signals_table.buffer))) + signal_idx;       comm::timeout_while<kNumTimeoutCycles>([=](constbool& is_last_check) {           constauto signal = ptx::ld_acquire_sys<int64_t>(signal_ptr);           if (signal >= target)               returntrue;           if (is_last_check)               timeout_print();           returnfalse;       });   }
```

TMA 本地拷贝: `tma_copy`用 `kNumStages=2` 的 mbarrier 流水分段把用户 tensor 搬到 `send_buffer_ptr`. 工作划分: `num_tma_blocks = num_bytes / kNumTMAAlignBytes`, 按 SM 分片 `num_tma_blocks_per_sm = ceil_div(num_tma_blocks, kNumSMs)`, 每个 SM 独立流水式处理自己那段.

```
   for (int64_t iter_idx = 0; iter_idx < num_iterations; ++ iter_idx) {       constauto stage_idx = static_cast<int>(iter_idx % kNumStages);       constauto [store_offset, num_store_bytes] = get_iter_info(iter_idx);       if (iter_idx < kNumStages) {                       // 填满流水: 前 kNumStages 次仅发 load           ptx::tma_load_1d(tma_buffers + stage_idx * kNumTMABytesPerStage,                            math::advance_ptr(src_ptr, store_offset),                            mbarriers + stage_idx, num_store_bytes);           ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_store_bytes);       }       ptx::mbarrier_wait_and_flip_phase(mbarriers + stage_idx, phases[stage_idx]);       ptx::tma_store_1d(math::advance_ptr(dst_ptr, store_offset),                         tma_buffers + stage_idx * kNumTMABytesPerStage,                         num_store_bytes);       ptx::tma_store_commit();       constauto next_iter_idx = iter_idx + kNumStages;       if (next_iter_idx < num_iterations) {              // 预取: 释放 stage 后立刻发下一个 load           ptx::tma_store_wait<kNumStages - 1>();           constauto [load_offset, num_load_bytes] = get_iter_info(next_iter_idx);           ptx::tma_load_1d(tma_buffers + stage_idx * kNumTMABytesPerStage,                            math::advance_ptr(src_ptr, load_offset),                            mbarriers + stage_idx, num_load_bytes);           ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_load_bytes);       }   }   ptx::tma_store_wait();
```

barrier: 由于TMA是异步搬运的, 因此需要`cooperative_groups::this_grid().sync()` 等所有 SM 的 TMA store 全部可见, 再发起 RDMA put, 避免出现数据还没搬完就被 put 读走.

RDMA PUT + signal:

`sm_idx == 0` 的唯一线程调用 `gin.put(recv_buffer_ptr, send_buffer_ptr, num_x_bytes, dst_rank_idx, 0, SignalInc(local_idx_in_dst + kNumRanks))`.

NCCL Gin 会把本地 send 段写入对端 recv 段, 完成后原子把 data-ready 信号 +1(`signal_idx = kNumRanks + local_idx_in_dst`, 收端据此阻塞等). 最后 `*send_count_ptr += 1` 推进本地计数.

```
    // Issue RDMA put    if (sm_idx == 0 and ptx::elect_one_sync()) {        gin.put<ncclTeamTagWorld>(            recv_buffer_ptr,            send_buffer_ptr,            num_x_bytes, dst_rank_idx,            0,            // TODO: is this signal highly optimized?            ncclGin_SignalInc(static_cast<ncclGinSignal_t>(local_idx_in_dst + kNumRanks)));        *send_count_ptr += 1;    }
```

### 3.5 Recv流程

接收端执行过程如下, 同样接收后进行TMA copy

```
// 模板常量: kNumSMs=grid 中 block 数(每 block 1 warp)；kNumRanks=通信域 rank 数；//           kNumSmemBytes=TMA 流水可用的动态共享内存；kNumTimeoutCycles=自旋超时周期上限. // __launch_bounds__(32, 1): 每 block 32 线程 = 1 个 warp, 最少驻留 1 个 block, 与 32 线程 warp-scoped PTX 对齐. template <int kNumSMs, int kNumRanks, int kNumSmemBytes, int64_t kNumTimeoutCycles>__global__ void __launch_bounds__(32, 1)pp_recv_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,  // NCCL Gin 设备侧通信器 + 对称窗口(用于读 signals_table)             void* x, int64_t num_x_bytes,                                        // 用户接收张量指针与字节数(≤ num_max_tensor_bytes)             void* buffer, void* workspace,                                       // buffer: 对称 4 段 slot 区；workspace: 存 pp_recv_count 等计数器             constint rank_idx, constint src_rank_idx,                          // 本 rank 索引 / 数据来源 rank 索引(必为 prev 或 next)             constint64_t num_max_tensor_bytes,                                  // 单 slot 对齐后字节数(pp_set_config 里 align(bytes,32))             constint num_max_inflight_tensors) {                                // 环形队列深度(信用额度)    constauto sm_idx = static_cast<int>(blockIdx.x);    // WorkspaceLayout 计算 workspace 内各计数器的偏移(这里只用到 pp_recv_count 两个 int64 槽)    constauto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);        // 以「src 视角调用 get_buffer_offset」解出本地 recv 段号(src_idx_in_local)和发端那边我的段号(local_idx_in_src)    //   · src == next  → (0, 1): 收自 next → 用本地段 0；我在 src 那边占段 1    //   · src == prev  → (1, 0): 收自 prev → 用本地段 1；我在 src 那边占段 0    constauto [src_idx_in_local, local_idx_in_src] = get_buffer_offset<kNumRanks>(src_rank_idx, rank_idx);    // 构造 NCCL Gin 句柄, CTA 级资源共享；0 号 context 即默认 QP    constauto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);    // 读本地「已从 src 收到的数量」. 每成功 recv 一次 +1(见函数末尾), 决定槽位与期望 signal    constauto recv_count_ptr = workspace_layout.get_pp_recv_count_ptr(src_idx_in_local);    constauto recv_count = __ldg(recv_count_ptr);    // 环形队列取模定位本次 slot；与发端同一公式(send_count % inflight)严格对齐    constauto slot_idx = recv_count % num_max_inflight_tensors;    // 本地 recv 段起址: 段号 = src_idx_in_local + 0(段 0 或 1 专用于接收)    // 偏移 = (段号 * inflight + slot_idx) * 对齐后字节数    constauto recv_buffer_ptr = math::advance_ptr(        buffer, ((src_idx_in_local + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);    // elect_one_sync: 在 warp 的 32 线程中选出唯一一条「当选线程」执行后续单线程逻辑    // 等待 data-ready 信号 + 发起 TMA 的拷贝都由这条线程驱动(TMA 描述符一条线程发即可)    if (ptx::elect_one_sync()) {        // 等待发端对本 slot 的 put 完成:         //   signal_idx = kNumRanks + src_idx_in_local(data-ready 信号组)        //   target     = recv_count + 1(发端每 put 一次会把该 signal +1)        // 自旋 ld_acquire_sys 轮询, 达到 kNumTimeoutCycles 未满足则调用 timeout_print 并继续退出检查        check_signal<kNumTimeoutCycles>(            gin,            static_cast<ncclGinSignal_t>(src_idx_in_local + kNumRanks),            recv_count + 1,            []() { printf("DeepEP PP recv timeout, recv buffer is empty\n"); }        );        // TMA 1D 拷贝: 把对称 buffer 里的 recv 段搬到用户 tensor x        //   内部按 kNumTMAAlignBytes 切片, 跨 kNumSMs 个 SM 平均分片；        //   每 SM 2 级 mbarrier 流水(load → wait → store → commit → 预取下一块), 全量吞吐压满 TMA 引擎        tma_copy<kNumSMs, kNumSmemBytes>(recv_buffer_ptr, x, num_x_bytes, sm_idx);    }        // Grid-level 栅栏: 等全部 kNumSMs 个 block 的 TMA store 都完成并对全局可见    // 必须放在「发 release signal」之前, 否则发端可能在数据还没搬完时就重用该 slot    cooperative_groups::this_grid().sync();    // 只由 0 号 block 的 leader 线程负责通知发端「这个 slot 已消费, 可回收」    if (sm_idx == 0and ptx::elect_one_sync()) {        // 向 src_rank_idx 发 release-signal 原子自增:         //   signal_idx = kNumRanks + local_idx_in_src + 2(+2 区别于 data-ready 段)        // 发端的 pp_send_impl 内 check_signal 正是等这个计数 ≥ send_count - inflight + 1        gin.signal<ncclTeamTagWorld>(            src_rank_idx, ncclGin_SignalInc(static_cast<ncclGinSignal_t>(kNumRanks + local_idx_in_src + 2))        );        // 本地计数推进；下一次 pp_recv 会以新的 recv_count 算槽位与期望 signal        *recv_count_ptr += 1;    }}
```

## 4. Engram

### 4.1 Buffer分配

`get_engram_storage_size_hint`用于计算Buffer大小, 通过累计条目数`num_engram_entries`,以及输入token hidden-dim 的大小 `engram_hidden`决定,

```
num_sf_packs = ceil_div(hidden, 32) if dtype.itemsize <= 1 else 0  # FP8 才预留 scalenum_bytes_per_entry = align(hidden * dtype.itemsize + num_sf_packs * 4, 32)return num_bytes_per_entry * (num_entries + num_max_tokens_per_rank)#                              ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^#                              storage 区     + 预留 fetch 接收区
```

buffer分配分为两段

![图片](assets/eda01731fa59.png)

storage段根据数值类型, 如果支持FP8则需要考虑保存Scale

recv 段为 buffer 剩余部分, 必须满足 `num_tokens * hidden * 2 ≤ num_engram_recv_bytes`

| 字段 | 含义 |
| --- | --- |
| `num_buffer_bytes` | buffer 段总字节数 |
| `buffer` | buffer 段起点(对称地址, 本地+所有 peer 均同 offset) |
| `num_engram_entries`, `engram_hidden` | 本 rank 的 storage 条目数与 hidden |
| `num_engram_storage_bytes` | 写入的 storage 占用字节(对齐到 32B) |
| `num_engram_recv_bytes` | `num_buffer_bytes - num_engram_storage_bytes`, 供 fetch 写入 |

### 4.2 engram write

engram write很简单, 就是一个异步的拷贝, 执行分为4步

barrier(false,true) : 即use_comm_stream=false, with_cpu_sync=true, 使用compute_stream , 同时会在barrier kernel前后插入`cudaDeviceSynchronize`, 确保前一次 `engram_fetch` 的远端 RDMA get 都完成, 避免 peer 正在读取旧 storage 时被覆盖.

校验storage

```
   EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16);   EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());   num_engram_entries = num_entries;   engram_hidden     = hidden;   EP_HOST_ASSERT(storage.nbytes() <= num_buffer_bytes);
```

执行异步拷贝,由于 `buffer` 已通过 `ncclCommWindowRegister` 注册为对称内存, 其它 rank 可直接通过 `gin.get` 读取此地址.

```
   cudaMemcpyAsync(buffer, storage.data_ptr(), storage.nbytes(),                   cudaMemcpyDeviceToDevice, compute_stream);   num_engram_storage_bytes = align<int64_t>(storage.nbytes(), 32);   num_engram_recv_bytes    = num_buffer_bytes - num_engram_storage_bytes;
```

再进行一次barrier, 保证对称窗口内容刷新并被所有 peer 观测到.

### 4.3 engram fetch

首先也是校验参数:

```
   EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);   EP_HOST_ASSERT(num_tokens * engram_hidden * sizeof(nv_bfloat16)                  <= num_engram_recv_bytes);  // 接收区够大   if (num_qps == 0) num_qps = nccl_context->num_allocated_qps;   EP_HOST_ASSERT(num_engram_entries > 0);   // 必须先 engram_write 过
```

然后构造 fetched tensor 视图

```
   const auto fetched = torch::from_blob(       math::advance_ptr(buffer, num_engram_storage_bytes),  // recv 段起点       {num_tokens, engram_hidden},       torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
```

接着分配 last_gin_requests, 大小为 `num_ranks × num_qps`: 每个 (QP, peer) 对记录最后一次聚合 issue 的请求 handle, 用于 wait.

```
   const auto last_gin_requests = torch::empty(       {nccl_context->num_ranks * num_qps, sizeof(ncclGinRequest_t)}, ...);
```

然后issue kernel `launch_engram_fetch`, JIT 生成并构建 `engram_fetch_impl` ,在 `current CUDA stream` 上发射 `grid = num_qps, threads = 1024`. `engram_fetch_impl`的模版参数为:  `<kNumQPs, kNumEntriesPerRank, kHidden, kNumRanks, kNumThreads=1024, kNumWarps = kNumThreads / 32>`, launch 为 `grid=kNumQPs, block=kNumThreads`, 即每个 block 对应一个 QP, 共计32 个 block. 具体执行流程如下:

初始化

```
const auto qp_idx         = blockIdx.x;const auto warp_idx       = ptx::get_warp_idx();const auto global_warp_idx = qp_idx * kNumWarps + warp_idx;const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx,                                 NCCL_GIN_RESOURCE_SHARING_CTA);__shared__ bool sent_to_rank[kNumRanks];      // 记录本 QP 向哪些 peer 发过 get
```

发送读请求采用leader线程issue, `indices[i]` 是全局条目号(范围 `[0, num_ranks * num_entries)`), 通过整除/取模拆为 `(src_rank, src_entry)`.

```
if (ptx::elect_one_sync()) {    for (int i = global_warp_idx; i < num_tokens; i += kNumQPs * kNumWarps) {        constauto global_idx    = __ldg(indices + i);        constauto src_rank_idx  = global_idx / kNumEntriesPerRank;        constauto src_entry_idx = global_idx % kNumEntriesPerRank;        // src: peer buffer 的 storage 段第 src_entry_idx 条        // dst: 本 rank buffer 的 recv 段第 i 个槽        gin.get<team_t>(            advance_ptr(storage, src_entry_idx * kNumHiddenBytes),            advance_ptr(fetched, i             * kNumHiddenBytes),            kNumHiddenBytes, src_rank_idx,            ncclGinOptFlagsAggregateRequests);   // 聚合不立即敲 DB        sent_to_rank[src_rank_idx] = true;    }}__syncthreads();
```

由于是读请求, 接下来针对每个 peer 执行 `flush_async` , 它会提交此前所有 aggregate read 到网卡, 并把"最后一个请求"的 handle 写入 `last_gin_requests`. , 触发聚合 DB ring

```
if (ptx::elect_one_sync()) {    for (int i = warp_idx; i < kNumRanks; i += kNumWarps) {        auto* request_ptr = last_gin_requests + qp_idx * kNumRanks + i;        if (sent_to_rank[i]) {            gin.flush_async<team_t>(i, request_ptr);   // 返回可等待的 request        } else {            *reinterpret_cast<int4*>(request_ptr) = make_int4(0, 0, 0, 0);        }    }}
```

最后是等待读回数据:

```
for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {    auto v4 = __ldg(reinterpret_cast<int4*>(last_gin_requests + qp_idx*kNumRanks + i));    if (v4.x|v4.y|v4.z|v4.w) {                 // 该 peer 有实请求        auto req = *reinterpret_cast<ncclGinRequest_t*>(&v4);        gin.wait(req);                         // 阻塞直到完成    }}
```

## 5. AGRS

AGRS 的命名虽然叫 "all-gather reduce-scatter", 但它把自己约束在 NVLink 域内, 以换取 DMA-only + 0-SM 的极致性能, 针对跨节点包含RDMA的通信则可以直接调用NCCL的all-gather.

### 5.1 Buffer分配

首先通过`get_agrs_buffer_size_hint`计算buffer大小, 并返回`num_max_session_bytes`, 具体计算方法为"每 rank 本地 session 总字节 × num_ranks × num_max_inflight_agrs" 的总和.

```
num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_buffer_size_hint(    group, num_bytes_per_tensor * group.size() * num_max_inflight_agrs)buffer = deep_ep.ElasticBuffer(group, explicitly_destroy=True, num_bytes=num_max_session_bytes)buffer.agrs_set_config(num_max_session_bytes, num_max_inflight_agrs)
```

然后通过`deep_ep.ElasticBuffer`创建buffer, 在workspaceLayout中包含:

AGRS recv signals(kNumMaxInflightAGRS × kNumMaxRanks= 32 × 1024 个)

AGRS session signals (kNumMaxRanks 个) 设备侧通过如下函数获取指针

```
// AGRS recv signal: 每次 all_gather 独占一个 slot, 用于 slot 级同步__forceinline__ __device__ __host__ int* get_agrs_recv_signal_ptr(const int& slot, const int& rank_idx) const {    const auto base_ptr = math::advance_ptr<int>(        get_pp_recv_count_ptr(0), 2 * sizeof(int64_t));    return base_ptr + slot * kNumMaxRanks + rank_idx;}// AGRS session signal: 每个 rank 一个 int, 用于 session 级同步__forceinline__ __device__ __host__ int* get_agrs_session_signal_ptr(const int& rank_idx) const {    const auto base_ptr = math::advance_ptr<int>(        get_agrs_recv_signal_ptr(0, 0), kNumMaxInflightAGRS * kNumMaxRanks * sizeof(int));    return base_ptr + rank_idx;}
```

在buffer段分配AGRS所用的buffer(= num_max_session_bytes), 具体Layout如下, Session buffer 由 `agrs_buffer_offset`(字节偏移游标)和 `agrs_buffer_slot_idx`(slot 计数器)协同推进.

![图片](assets/fb7d8f700010.png)

接着`agrs_set_config` 的配置流程

```
barrier(true, true);  // 把之前所有 op flush 掉EP_HOST_ASSERT(num_max_session_bytes > 0 and new_num_max_agrs_per_session > 0);EP_HOST_ASSERT(num_max_session_bytes <= num_buffer_bytes);// 每个 session 最多 32 次 all-gather, 决定了 workspace 中 AGRS 信号槽数.EP_HOST_ASSERT(new_num_max_agrs_per_session <= layout::WorkspaceLayout::kNumMaxInflightAGRS); // AGRS 仅限单节点 NVLinkEP_HOST_ASSERT(nccl_context->num_nvl_ranks == nccl_context->num_ranks); this->num_max_agrs_session_bytes = math::align<int64_t>(num_max_session_bytes, 32); // LDG.256 对齐this->num_max_agrs_per_session = new_num_max_agrs_per_session;
```

### 5.2 Session based上下文管理

`create_agrs_session` 在Host侧做状态变更, 不下发任何 GPU op. `agrs_session_idx`**从未归零**, 单调自增 —— 这让 recv signal 的"值"本身就能区分不同 session 的残留写入. 并且不在这里做 barrier: 依赖上一次 `destroy_agrs_session` 写下的 session 完成信号来保证 peer 的 buffer 已经被消费完.

```
void create_agrs_session() {    EP_HOST_ASSERT(not agrs_in_session);  // 不允许嵌套    agrs_in_session = true;    agrs_buffer_offset = 0;   // 从 buffer 起点重新开始切分    agrs_buffer_slot_idx = 0; // slot 计数清零    agrs_session_idx += 1;    // 全局单调递增 session id}
```

`destroy_agrs_session` 流程如下:

```
void destroy_agrs_session() {    EP_HOST_ASSERT(agrs_in_session);    agrs_in_session = false;    // 1. 让 comm stream 等 compute stream —— 确保用户侧 consume 完毕    stream_wait(comm_stream, at::cuda::getCurrentCUDAStream());    // 2. 对 N-1 个 peer 做 "写对方 signal + 等对方 signal" 的 batch op    std::vector<void*> write_ptrs(num_ranks-1), wait_ptrs(num_ranks-1);    for (int i = 0; i < num_ranks - 1; ++i) {        int dst = (rank_idx + i + 1) % num_ranks;        // 写到对方 workspace 的 "我(本 rank)的 session signal"        write_ptrs[i] = get_sym_ptr(session_signal_ptr(rank_idx), dst);        // 等待本地 workspace 上对方的 session signal 到达        wait_ptrs[i]  = session_signal_ptr(dst);    }    cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, agrs_session_idx);}
```

最后一行使用 `cuStreamBatchMemOp` 把一组操作作为一个驱动级命令提交:

```
void batched_write_and_wait(CUstream stream, const std::vector<void*>& write_ptrs, const std::vector<void*>& wait_ptrs, const int& value) {    std::vector<CUstreamBatchMemOpParams> ops(write_ptrs.size() + wait_ptrs.size());    for (int i = 0; i < write_ptrs.size(); ++ i)       ops[i] = create_mem_op(write_ptrs[i], value, CU_STREAM_MEM_OP_WRITE_VALUE_32);    for (int i = 0; i < wait_ptrs.size(); ++ i)       ops[write_ptrs.size() + i] = create_mem_op(wait_ptrs[i], value, CU_STREAM_MEM_OP_WAIT_VALUE_32, CU_STREAM_WAIT_VALUE_GEQ);    CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp(stream, ops.size(), ops.data(), 0));}
```

`CU_STREAM_MEM_OP_WRITE_VALUE_32`: 把 `agrs_session_idx` 写到 peer 的 session signal 位置.

`CU_STREAM_MEM_OP_WAIT_VALUE_32 + WAIT_VALUE_GEQ`: 在 comm stream 上自旋等待本地 signal ≥ `agrs_session_idx`(GEQ 允许后来更大的 session id 提前到达).

destroy 相当于一个 session 级 barrier:

告诉所有 peer "我已经不再读 session X 的 buffer"

等待所有 peer 也告诉我同样的消息

自此 session X 占用的 buffer 区对所有 rank 都是可重用的. 这是 `create_agrs_session` 不需要做 barrier 的原因: 上一轮 destroy 已经提供了"写后可回收"的保证(`agrs_buffer_offset=0` 才能安全复用空间).

### 5.3 all-gather流程

以 test_agrs.py 中的典型调用为例:

```
with buffer.agrs_new_session():    out_tensors, handle = do_all_gather(buffer, is_inplace, is_batched, tensors)    for h in handle: h()
```

AGRS 的 all-gather 阶段；底层通过 NCCL 对称内存 LSA 指针,  每 rank 将本地 slot 直接 put 到 peer 的对应槽. 另外可选`agrs_get_inplace_tensor`, 返回的 tensor 指针就落在 session buffer 中本 rank 的接收槽, 用户在此写入数据后, 后续 all_gather 时 src==dst 会被跳过拷贝.

```
for (const auto& num_bytes: num_bytes_list) {    out.push_back(torch::from_blob(        buffer + offset + num_bytes * rank_idx,  // 本 rank 的槽位        {num_bytes}, uint8));    offset += num_bytes * num_ranks;  // 预留全 rank 槽位}
```

C++ `all_gather` 实现如下:
1. 规划 slot 偏移 + 判断 inplace
```
        for (int i = 0; i < num_tensors; ++i) {            constauto& x = tensors[i];            // 输入必须 contiguous、CUDA 上, nbytes 按 32B 对齐（LDG.256 要求）            EP_HOST_ASSERT(x.is_contiguous());            EP_HOST_ASSERT(x.is_cuda() and x.nbytes() % 32 == 0);            // inplace 检测 —— 若 x 本身就是从 agrs_get_inplace_tensor 切出来的            //          （指针落在 session buffer 范围内）, 那么"本 rank→本 rank"的拷贝可以省掉            constauto x_offset = math::ptr_diff(x.data_ptr(), buffer);            constbool is_inplace = 0 <= x_offset and x_offset < num_max_agrs_session_bytes;            offset[i] = agrs_buffer_offset;            // 每 tensor 要发 num_ranks 份（到每个 peer），inplace 时省一份            num_copies += nccl_context->num_ranks - is_inplace;            // 推进游标, 为这个 tensor 预留 num_ranks 份槽位            agrs_buffer_offset += x.nbytes() * nccl_context->num_ranks;            // 判断inplace tensor 必须精确落在"本 rank 的槽位"偏移上, 否则逻辑错位            EP_HOST_ASSERT(not is_inplace or x.data_ptr() == math::advance_ptr(buffer, offset[i] + x.nbytes() * nccl_context->rank_idx));        }        // session 容量上限校验 —— 超了说明配置太小或忘了 flush（destroy_session）        EP_HOST_ASSERT(agrs_buffer_offset <= num_max_agrs_session_bytes and agrs_buffer_slot_idx < num_max_agrs_per_session and                       "Not enough session buffer size. Did you forget to flush session?");
```
2. Stream 同步 + 构建 batched copy 参数
```
        // ========== 阶段 B: Stream 同步 + 构建 batched copy 参数 ==========        // Wait compute stream        // comm_stream 等 compute_stream, 确保用户在 compute_stream 上产生 x 的写入可见        constauto compute_stream = at::cuda::getCurrentCUDAStream();        stream_wait(comm_stream, compute_stream);        // 发送数据到所有rank        // 构建 (src, dst, size) 三元组列表, 一次 cudaMemcpyBatchAsync 发完.         //   · src: 本 rank 的原始数据(或 inplace 的本 rank 槽位)        //   · dst: 对端 buffer 中"本 rank 槽位"的 NVLink P2P 地址(通过 get_sym_ptr 解析)        //   · 外层遍历 peer 时用 (rank_idx + i) 偏移, 错峰 peer 接收顺序, 避免全员同时出口冲突        std::vector<size_t> sizes(num_copies);        std::vector<void*> dst_ptrs(num_copies), src_ptrs(num_copies);        int count = 0;        for (int i = 0; i < nccl_context->num_ranks; ++i) {            for (int j = 0; j < num_tensors; ++j) {                constauto& x = tensors[j];                constauto dst_rank_idx = (nccl_context->rank_idx + i) % nccl_context->num_ranks;                void* src_ptr = x.data_ptr();                // get_sym_ptr 把本地地址翻译为对端 NVLink P2P 地址                //  (本地 offset 不变, 只换基址为 nvl_window_ptrs[dst])                void* dst_ptr =                    nccl_context->get_sym_ptr(math::advance_ptr(buffer, offset[j] + x.nbytes() * nccl_context->rank_idx), dst_rank_idx);                // src == dst 发生在 inplace 且 dst_rank_idx == rank_idx 时, 跳过空拷贝                if (src_ptr != dst_ptr) {                    src_ptrs[count] = src_ptr;                    dst_ptrs[count] = dst_ptr;                    sizes[count] = x.nbytes();                    count += 1;                }            }        }        //  SrcAccessOrderStream —— 告诉 runtime 源按 stream 顺序访问, 避免一些保守的同步障碍.        //  PreferOverlapWithCompute —— 倾向走独立 copy engine, 与 compute kernel overlap        cudaMemcpyAttributes attrs = {.srcAccessOrder = cudaMemcpySrcAccessOrderStream, .flags = cudaMemcpyFlagPreferOverlapWithCompute};                // 一次 driver 调用下发所有拷贝到 comm_stream；0 SM 占用, 纯 NVLink DMA 并发        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, comm_stream));
```
3.Slot 级信号交换（等所有 peer 的数据到位）
```
        // Wait for data from other ranks        // 快照 session id 作为信号值；slot_idx 是本次 all_gather 独占的信号槽位        constint current_session = agrs_session_idx;        constint slot_idx = agrs_buffer_slot_idx;        agrs_buffer_slot_idx += 1;        // 对每个 peer(不含自己)构造 (write, wait) 对        // write: 向 peer 的 recv_signal[slot, 本 rank] 写入 current_session        // 含义: "我已完成到 slot_idx 的所有前序拷贝"        // wait: 自旋等本地 recv_signal[slot, peer] >= current_session        // 含义: "对端已完成它到 slot_idx 的拷贝, 我可以读它写到我槽位的数据了"        // stream 内顺序保证: cudaMemcpyBatchAsync 的拷贝必先完成再做 signal write,         // 所以 peer 一旦看到 signal, 数据一定已经可见(无需额外 fence).         std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);        for (int i = 0; i < nccl_context->num_ranks - 1; ++i) {            constauto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;            // 写地址需要翻译到 peer 的对称内存(get_sym_ptr)；            //          等地址就是本地 workspace(无需翻译)            write_ptrs[i] = nccl_context->get_sym_ptr(                workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, nccl_context->rank_idx), dst_rank_idx);            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, dst_rank_idx);        }        // batched_write_and_wait 内部使用 CU_STREAM_MEM_OP_WAIT_VALUE_32 + WAIT_VALUE_GEQ,         //  "≥ current_session" 的语义保证跨 session 残留信号不会误触发        //  (agrs_session_idx 单调递增, 老 session 的 signal 值一定更小).         //  自己不需要 write/wait 自己: comm_stream 的严格串行已经保证 in-stream 顺序.         cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, current_session);
```
4. 构造零拷贝输出视图 + 返回异步 wait handle
```
        // Build output tensors eagerly        // 每个输出在原 shape 前面插入 num_ranks 维度, 用 torch::from_blob 直接指向        //          session buffer 中的 offset[i](连续覆盖 num_ranks 个槽位), 无分配、无拷贝        std::vector<torch::Tensor> out(num_tensors);        for (int i = 0; i < num_tensors; ++i) {            auto shape = tensors[i].sizes().vec();            shape.insert(shape.begin(), nccl_context->num_ranks);            out[i] = torch::from_blob(math::advance_ptr(buffer, offset[i]), shape, tensors[i].options());        }        // Return tensors and a handle to wait for data arrival        // event 到 comm_stream, 作为"数据完全就绪"的时间点标志        constauto event = EventHandle(comm_stream);        // 返回一个闭包 handle, 由调用者在消费 out 之前调用 h():         // stream_wait(compute_stream, event): 让 compute_stream 等 comm_stream 追平        // 两个断言: 必须在原 compute_stream 上调用, 且 session 未被重开        // (跨 session 的 handle 无效, 因为 buffer 早已被下轮覆盖)        auto handle = [=, this]() {            EP_HOST_ASSERT(compute_stream == at::cuda::getCurrentCUDAStream());            EP_HOST_ASSERT(agrs_in_session and current_session == this->agrs_session_idx);            stream_wait(compute_stream, event);        };        return {std::move(out), std::move(handle)};
```

整个调用的总时序如下图所示:

![图片](assets/47ba567b55bc.png)