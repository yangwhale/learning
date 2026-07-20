# 分析一下EP并行和DeepSeek开源的DeepEP代码

> 作者: zartbot
> 日期: 2025年2月26日 23:37 浙江
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493292&idx=1&sn=7af7db0f3d78f0fb52dc847934c7800e&chksm=f995f66ecee27f785788acda0075ba451a92619d66587b2d317e00b8ca23d20c691ee762ca23#rd

---

被约了几个团队的人追着要清B来分析一下DeepEP的工作，公司内外的团队都有…简单的一句话说，非常棒的工作。但是还有一些硬件上的缺陷，在DeepSeek-V3团队的论文中提出的建议要结合在一起看就会更清楚了。我们还是由浅入深来读读EP并行，并进一步分析一下这份出色的工作。顺便展开讨论一下ScaleUP和ScaleOut网络遇到的难题和新的需求，DeepSeek采用了IB的技术，虽然在github上说理论上兼容RoCE，但是还有很多细节的地方需要探讨。

**本文目录如下**

1. EP并行概念
2. SGlang EP并行实现
3. DeepEP
   - 3.1 用于通信的Buffer
   - 3.2 用于训练和Prefill的高吞吐Kernel
     - 3.2.1 Dispatch通信流程
       - 3.2.1.1 Notify_Dispatch
       - 3.2.1.2 Intranode::dispatch
       - 3.2.1.3 Internode::dispatch
         - 3.2.1.3.1 kRDMASender
         - 3.2.1.3.2 kRDMASenderCoordinator
         - 3.2.1.3.3 kRDMAAndNVLForwarder
         - 3.2.1.3.4 kForwarderCoordinator
         - 3.2.1.3.5 kNVLReceivers
     - 3.2.2 Combine流程
       - 3.2.2.1 Intranode_Combine
       - 3.2.2.2 Internode_Combine
         - 3.2.2.2.1 kNVLSender
         - 3.2.2.2.2 kNVLAndRDMAForwarder
         - 3.2.2.2.3 kRDMAReceiver
         - 3.2.2.2.4 kCoordinator
   - 3.2 用于Decoding的低延迟Kernel
     - 3.2.1 LowLatency Layout
     - 3.2.2 低延迟Dispatch
       - 3.2.2.1 SEND PHASE
       - 3.2.2.2 RECV PHASE
     - 3.2.3 低延迟Combine
       - 3.2.3.1 SEND PHASE
       - 3.2.3.2 RECV PHASE
   - 3.3 其它细节
     - 3.3.1 文档针对行为的PTX指令
     - 3.3.2 Memory Order
     - 3.3.3 nvshmem需要的修改
4. RoCE上运行DeepEP的挑战
   - 4.1 DeepSeek用到的IB网络技术
   - 4.2 RoCE使用DeepEP的问题
     - 4.2.1 Multi-Rail和Rail-Only拓扑的问题
     - 4.2.2 incast
     - 4.2.3 RC兼容
     - 4.2.4 In Network Computing如何做？
   - 4.3 真正的适合EP的RDMA over Ethernet
5. 关于DeepSeek-V3论文的建议
6. 关于MoE模型的演进

---

# 1. EP并行概念

DeepSeek MoE的原理和演进在以前的一篇文章详细分析过

> [详细读读DeepSeek MoE相关的技术发展](#)

Expert Parallelism(EP并行)如下图所示:

![EP并行示意图：GPU1/GPU2 各含 MHA/Norm/Gating/Expert1~4，通过 all2all 通信互联，右侧扩展到 GPUs 3,4](assets/epcode-01.png)

在DeepSeek-R1推理过程中，为什么要实现EP并行? 从系统的角度来看，单个专家的参数数据量为44MB,具体计算如下所示:

```python
dim = 7168
inter_dim = 2048
tokens = 256
e = Expert(dim, inter_dim)

from nnflops.import get_model_complexity_info
```

```
Expert:
44.05 M, 100.000% Params, 22.19 GMac, 99.995% MACs
    (w1): Linear(14.68 M, 33.32% Params, 3.76 GMac, 33.328% MACs, in_features=7168, out_features=2048, ...
    (w2): Linear(14.68 M, 33.24% Params, 3.76 GMac, 33.328% MACs, in_features=2048, out_features=7168, ...
    (w3): Linear(14.68 M, 33.34% Params, 3.76 GMac, 33.328% MACs, in_features=7168, out_features=2048, ...
```

如果一次加载44MB的专家参数权重，而仅处理几个token是很不划算的，同时多个专家放置在一台机器上，很容易压之间负载不平衡，并且完全占满显存带宽。整个系统的瓶颈在显存带宽上。

那么一个很朴素的想法就是通过多台机器并行，把专家放置在多个机器上，一方面等待的时候可以隐存的开销，另一方面把一个token发送到多个机器上的data locality。然后用更大的batch(在这个256,序列代码为512)进行一次权重加载能够被更多的token,合乎摊薄提升了。

但是难题就是要更大量的all2all的跨机通信，此时跨机网络通信的问题不解决好，EP并行的性能就会转移到网络。这就是DeepEP的工作的重要性了。在读DeepEP之前，我们先来看看开源的EP并行实现。

# 2. SGlang EP并行实现

在RDMA网络中，大规模组网下Alltoall将不同的通信效率都是一个很大的问题，因此开源的软件有Alltoall通过alltoreduce的方式来实现。规避Alltoall,因此在开源社区如所用上DeepEP的工作。还需要进一步分析。这里对SGlang EP并行做一个简单的介绍。先启用官方在`\python\sglang\srt\layers\moe\ep_moe\layer.py`中的EPMoE类, Sglang通过将ep_size和ep_size做同, 然后它会根据自己的tp_rank来判断如何加载本专家。

```python
class EPMoE(torch.nn.Module):
    def __init__(...):
        ...
        self.num_experts = num_experts
        assert self.num_experts % self.tp_size == 0
        ...
        self.num_experts_per_partition = self.num_experts // self.tp_size
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
```

然后在该类的forward函数中来看详细的处理过程

```python
def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
    ...
    # 选择专家, 计算Gating得出top_idx以及权重, 从而实现将不同的token分发到不同的专家
    topk_weights, topk_ids = select_experts(...)

    # 重排序数据的预处理过程
    pre_reorder_triton_kernel[(hidden_states.shape[0],)]()

    # 由子当前rank获取需要处理的segment
    seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
    weight_indices_cur_rank = torch.arange(
        0,
        self.num_experts_per_partition,
        device=hidden_states.device,
        dtype=torch.int64,
    )

    # 第一次GroupGemm做gateup projection矩阵运算
    gateup_output = self.grouped_gemm_runner(
        a=gateup_input,
        b=self.w13_weight,
        c=gateup_output,
        ...)

    # Expert 的激活函数处理
    if self.activation == "silu":
        silu_and_mul_triton_kernel[(gateup_output.shape[0],)]()

    # Expert的第二次down projection GroupGEMM处理
    down_output = self.grouped_gemm_runner(
        a=down_input,
        b=self.w2_weight,
        c=down_output,
        ...)

    # Post重排
    post_reorder_triton_kernel[(hidden_states.size(0),)](
        down_output,
        output,
        src2dst, ...)
```

您肯定很好奇，实际的函数中并没有看到alltoall通信? 它在MLA阶段会执行DP并行，有一次allgather。然后EP并行时候，每个卡都持有本地专家进行运算，等完成Post_reorder后，其实同一个token需要经过Expert处理后的数据，都已经分散在不同卡上，此时进行一次allreduce即可。

这种宏观上的allgather+allreduce替代alltoall的做法虽然相对于alltoall直接dispatch/combine大了很多，但有时候还是会选择，因为工业界一直对alltoall通信的效率耿耿于怀，特别是在各种RoCE环境中，由于incast等问题带来的长尾延迟。实际通信效率还是有很多种以AG+AR好。本文后面的第五章会针对这一议题讨论这个话题。

当然刚注意到SGLang已经在开始整合DeepEP了~

![SGLang 团队 zhyncs42 推特截图：祝贺 DeepSeek 开源 DeepEP，SGLang 已在与核心成员讨论后开始适配 DeepEP，下方引用 DeepSeek 官方 Day 2 OpenSourceWeek 推文](assets/epcode-02.png)

> **zhyncs** @zhyncs42
> Liyue before the release and is **now adapting to DeepEP. Stay tuned!**
>
> **DeepSeek** @deepseek_ai · 12小时
> Day 2 of #OpenSourceWeek: DeepEP
> Excited to introduce DeepEP — the first open-source EP communication library for MoE model training and inference.
> …
> 下午6:40 · 2025年2月25日 · **991** 查看

但是SGlang在RoCE上还有很多网络的调整需要处理，后面详细介绍。

# 3. DeepEP

DeepEP是DeepSeek开源周第二天发布的一个项目，专注于EP并行和Alltoall通信的效率，在Prefill和Training阶段提供高吞吐的能力，同时在decoding阶段也提供了低延迟的传输。同时还支持一些直接的控制overlap计算和通信。

![DeepEP 发布截图：DeepSeek 官方推文 Day 2 of #OpenSourceWeek: DeepEP，首个开源 EP 通信库，列出高效 all-to-all、NVLink/RDMA 支持、高吞吐/低延迟 kernel 等特性](assets/epcode-03.png)

- Efficient and optimized all-to-all communication
- Both intranode and internode support with NVLink and RDMA
- High-throughput kernels for training and inference prefilling
- Low-latency kernels for inference decoding
- Native FP8 dispatch support
- Flexible GPU resource control for computation-communication overlapping

通信的核心就是是对Buffer的高效管理和使用，并且在延迟和吞吐上做TradeOff。下面我们分为三个小章节来讨论。

## 3.1 用于通信的Buffer

DeepEP对外的API处理原理是，先通过`get_buffer`函数来申请一段buffer,可以通过`Buffer.set_num_sms(sm_num)`来设定需要用到的SM

```python
_buffer: Optional[Buffer] = None
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer
    # 计算NVLink和RDMA需要多少字节
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
    # 分配buffer
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.nu
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer
```

在`\deep_ep\utils.py`中定义了一些buffer的config

```python
# Intranode
if num_ranks <= 8:
    return Config(Buffer.num_sms, 6, 256, 6, 128)

# Internode
config_map = {
    16: Config(Buffer.num_sms, 16, 288, 20, 128),
    24: Config(Buffer.num_sms, 8, 288, 32, 128),
    32: Config(Buffer.num_sms, 8, 288, 32, 128),
    64: Config(Buffer.num_sms, 20, 288, 28, 128),
    128: Config(Buffer.num_sms, 20, 560, 32, 128),
    144: Config(Buffer.num_sms, 32, 720, 12, 128),
    160: Config(Buffer.num_sms, 28, 720, 12, 128),
}
```

后面四个参数定义在 `csrc\config.hpp` 中定义

```cpp
struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;
};
```

这些config在`tests`目录下用于测试和benchmark，很显然对不同的组网和不同的性能需求要重新做performance tuning。但是这些值是基于H800和其实为了兼容不同的nvshmem网络配置比如不同的NVLink带宽，还需要一定的余量。未来还有更好针对不同的值来定义并修改config_file。

Buffer类型在 `csrc\kernel\buffer.cuh` 中定义了三类，通过这些结构体，可以灵活地管理GPU在内存中区，支持多种内存管理和访问模式。

- SymBuffer: Decouple模式存储送在 send_buffer和 recv_buffer,是解耦成对调用 buffer
- csrc\deep_ep.cpp 中定义了Buffer类，构造函数 `Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode)` 初始化缓冲区，然后定义了一系列的方法

关于对内核的初始化(Normal Kernel [intranode/internode]_dispatch/combine)

- 通用于训练和Decoding的低延迟(low_latency_[dispatch/combine]
- 用于topk_idx和num_experts产生的layout的获取dispatch_layout
- 定义了一个FiFo Slots的结构用于校验和结果处理的 move_fifo_slots()

**Task Fifo Size如下所示:**

```cpp
// Task fifo memory
int64_t fifo_bytes = sizeof(int) * NUM_MAX_FIFO_SLOTS;
int64_t buffer_ptr_bytes = sizeof(void*) * NUM_MAX_NVL_PEERS;
int64_t task_ptr_bytes = sizeof(int*) * NUM_MAX_NVL_PEERS;
```

在Host做还有一些和为Counter用于记录本地专家相关的信息

```cpp
// Host-side MoE info
volatile int* moe_recv_counter = nullptr;
int* moe_recv_counter_mapped = nullptr;

// Host-side expert-level MoE info
volatile int* moe_recv_expert_counter = nullptr;
int* moe_recv_expert_counter_mapped = nullptr;

// Host-side RDMA level MoE info
volatile int* moe_recv_rdma_counter = nullptr;
int* moe_recv_rdma_counter_mapped = nullptr;
```

## 3.2 用于训练和Prefill的高吞吐Kernel

训练和Prefill的batch对较大对延迟还不是很敏感，因此主要是最大化的利用带宽，README中的测试数据显示基本本上和主达到了跑满的效果:

| Type | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|---|---|---|---|---|
| Intranode | 8 | 153 GB/s (NVLink) | 8 | 158 GB/s (NVLink) |
| Internode | 16 | 43 GB/s (RDMA) | 16 | 43 GB/s (RDMA) |
| Internode | 32 | 44 GB/s (RDMA) | 32 | 47 GB/s (RDMA) |
| Internode | 64 | 46 GB/s (RDMA) | 64 | 45 GB/s (RDMA) |

看了DeepEP的实现，更加能够理解MoE Group的这个方法。从理论文之的3.2可以知道，它在训练阶段采用带宽的方式并行，正好在台H800的机器，然后从8个Group并将GroupScore选择4台进行

另一方面由于有其它上的专家会选择本地跨机，统计上每个专家的概率和平均反应到相对均衡的效果。因此在EP并行上采用了2层拓扑，对于单个GPU,首先根据Group idx直接将数据发送到Group idx对应的remote卡上和同Rank的GPU,然后再通过NVLink来传输。在DeepEP代码中通过 `intranode(NVLink)` 和 `internode(RDMA)` 来区分这两种通信技术。

在调用后dispatch之前通过 `get_dispatch_layout` 获得需要发送的Token,哪些要通过RDMA,以及token是否需要传输到某个Rank

```python
# Calculate layout before actual dispatch
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_ev
    _buffer.get_dispatch_layout(topk_idx, num_experts,
                                 previous_event=previous_event, async_finish=True,
                                 allocate_on_comm_stream=True)
```

其中由topk_idx为MoEGating函数产生的 `[num_tokens, num_topk]` 数组。通过dispatch_layout输出的结果为5个

- num_tokens_per_rank: `[num_ranks]` 的数组，每个rank需要发送到的token数目
- num_tokens_per_rdma_rank: `[num_rdma_ranks]`, 在每个RDMA Rank上要发送到的token数目
- num_tokens_per_expert: `[num_experts]` 每个专家的token数目
- is_token_in_rank: `[num_tokens, num_ranks]` 二维数组，标识每个token是否在某个rank来传送

### 3.2.1 Dispatch通信流程

dispatch具体的通信流程如下图所示:

![Dispatch 通信流程图：CPU 侧 Launch notify/dispatch/computation/combine 时序，GPU 侧 Notify tensor size ASAP、Dispatch 拆分 IB chunk/NVL chunk、Computation kernels 与 Combine，底部 Reuse layout information](assets/epcode-04.png)

详细代码为 `csrc\deep_ep.cpp` 中，以 `intranode_dispatch` 为例

```cpp
Buffer::intranode_dispatch(
    const torch::Tensor& x, //Tensor数据x
    const std::optional<torch::Tensor>& x_scales,//用于FP8缩放的值
    const std::optional<torch::Tensor>& topk_idx, // MoE Gating产生的专家index
    const std::optional<torch::Tensor>& topk_weights, //MoE Gating产生的weight
    const std::optional<torch::Tensor>& num_tokens_per_rank, //每个Rank的token数
    const torch::Tensor& is_token_in_rank, //标记token是否从某个rank发送
    const std::optional<torch::Tensor>& num_tokens_per_expert, //基于专家的token数统计
    int cached_num_recv_tokens, //需要缓存的token数
    const std::optional<torch::Tensor>& cached_rank_prefix_matrix, //[num_ranks,num_ranks]矩阵
    const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
    int expert_alignment,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async, bool allocate_on_comm_stream)
```

当 `cached_rank_prefix_matrix` 参数有值时，`cached_mode=True`, 然后根据Config中之义的BSM数量定义Channel，一个channel使用两图个block，偶数block用于发送，奇数用于接收，因此Channel数为 config.num_sms的一半。然后配置好这几个前处理数据，如果 `allocate_on_comm_stream=True` 则会创建comm_stream,并等待previous stream或者参数中的previous_event完成

```cpp
auto compute_stream = at::cuda::getCurrentCUDAStream();
if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() and async);
    at::cuda::setCurrentCUDAStream(comm_stream);
}

// Wait previous tasks to be finished
if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
} else {
    stream_wait(comm_stream, compute_stream);
}
```

#### 3.2.1.1 Notify_Dispatch

然后通过 notify_dispatch 函数来launch kernel执行 notify_dispatch, 通过多个节点交换需要发送信息回到，并统计好需要发送的Token数目以及统计好prefix_matrix

```cpp
// 初始化前变量为-1
*moe_recv_counter = -1; *moe_recv_rdma_counter = -1;
for (int i = 0; i < num_local_experts; ++i)
    moe_recv_expert_counter[i] = -1;

// 调用notify_dispatch
intranode::notify_dispatch(
    num_tokens_per_rank->data_ptr<int>(), // 每个Rank需要发送的token数目
    moe_recv_counter_mapped,
    num_ranks,
    num_tokens_per_rdma_rank->data_ptr<int>(), //dispatch_layout的结果，记每个RDMA Rank上需要发送的token数目
    moe_recv_rdma_counter_mapped,
    num_tokens_per_expert->data_ptr<int>(), //dispatch_layout的结果，基于专家的token数目
    moe_recv_expert_counter_mapped, //基于本地专家的token数目
    num_experts,
    is_token_in_rank.data_ptr<bool>(), //[num_tokens, num_ranks] 标记token需不要发送到某rank
    channel_prefix_matrix.data_ptr<int>(), //token偏移Channel数
    hidden_int4, num_scales, num_topk, expert_alignment,
    rdma_channel_prefix_matrix.data_ptr<int>(), //[num_rdma_ranks, num_channel]
    recv_rdma_rank_prefix_sum.data_ptr<int>(), //[num_rdma_ranks]
    ...)
```

当 sm_id == 0 时: 主线全负责同步至器，然后将每个rank和Expert的token数据量通过nvshmem_int_put_nbi 发送到RDMA对端 ranks。然后将计算好的token的数量发送到本rank, 使用 buffer_ptrs 和 nvl_send_buffer, 接下来计算各个rank所对应的前缀和，更新 recv_rdma_rank_prefix_sum 和 rank_prefix_sum。最后更新会先更新 moe_recv_expert_counter_mapped，最终再让全部 nvshmem_barrier(with_same_gpu_idx) 和 barrier_device 保完成后再各 sm_id == 0 时才算Tcshannel相关的处理，计算每个channel需要发送到的Token的数量以及rdma_channel_prefix_matrix 的前缀和，确保后续任务的调度和分发

在CPU上会根据notify的计算结果分配显存给Tensor

```cpp
// allocate new Tensor
auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>();
auto recv_src_meta = std::optional<torch::Tensor>();
auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
auto send_rdma_head = std::optional<torch::Tensor>();
auto send_nvl_head = std::optional<torch::Tensor>();

if (not cached_mode) {
    recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte)
    recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).d
    recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device...);
    send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device...);
    send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device...
}
```

最后再调用 dispatch 函数执行分发，dispatch函数在RDMA和NVLINK 各阶分别在 `csrc\kernels\internode.cu` 和 `csrc\kernels\intranode.cu` 主要目的就是根据token, topk_idx和weight放入缓冲区。

#### 3.2.1.2 Intranode::dispatch

`intranode::dispatch` 在NVLINK上执行发送，首先它会判断 `const bool is_sender = sm_id % 2 == 0`; 判断是sender还是receiver, 然后计算好每个核处理要的channel和rank。

**1. 发送逻辑:**

- 初始化状态变量，包括状态信息范围、队列起始和结束的编码等
- 等待发送通道可用数量
- 从头逐个发送数据缓冲区数据(包括metadata、源数据、topk等重要数据)
- 更新发送指针，标识数据已发送完成从而促成后续处理逻辑

**2. 接收逻辑:**

- 初始化接收变量，计算最大有效范围
- 等待接收通道可用数据
- 从头逐个接收数据缓冲区数据到目标位置，包括token、源数据、topk数等重要数据
- 更新接收指针，同步数据接收完成状态
- 处理数据方式来确保正确性，打完成后释放通道资源占用

#### 3.2.1.3 Internode::dispatch

`internode::dispatch` 涉及RDMA上的操作，相对更加复杂。它分配了多种WarpRole

```cpp
enum class WarpRole {
    kRDMASender,           //从发存本地缓冲读RDMA发送RDMA区域进行
    kRDMASenderCoordinator,//与前面协同发送区域进行同步
    kRDMAAndNVLForwarder,  //RDMA接收后经过本地的NVLink进行转发
    kForwarderCoordinator, //与转发缓冲进行协同同步
    kNVLReceivers          //从本地NVL缓冲区接收数据并写入最终缓冲区
};
```

##### 3.2.1.3.1 kRDMASender

首先通过 `get_channel_task_range` 获取任务范围，得到起始和结束的token 索引号 `token_start_idx` 和 `token_end_idx`, 然后通过 `rdma_channel` 指针计算好为对应share memory, 并将发送这个第一个token的idx指向 `token_start_idx`。

然后将这个channel需要发送的token数量以负数 `-value-1` 的方式记录到其它节点，然后回同步等待。

```cpp
for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMARanks) {
    // 计算本地rank的send_buffer的偏移量
    nvshmemx_int_put_nbi_warp(rdma_channel_meta.recv_buffer(rdma_rank), rdma_channel_meta.send_buffer(d
        translate(dst_rdma_rank%kLowLatencyMode) ? dst_rdma_rank%kNumRDMARanks : nvl_rank), 0);
}
nvshmem_fence();
sync_rdma_sender_smem();
```

然后开始逐代处理Token, 等待到列同头buffer中，获取锁并更新是指针，然后复制数据到发送内存中，并广播这个数据到其他本地流(如 x_、x_scales、topk_idx 和 topk_weights),复制的代码如下:

```cpp
// Copy `x` into symmetric send buffer
auto st_broadcast = *(reinterpret_cast<const int4*>(&value));
#pragma unroll
for (int j = 0; j < num_topk_ranks; ++j)
    ld_nc_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + i, value);
```

```cpp
#pragma unroll
for (int i = 0; i < num_topk_ranks; ++i)
    dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;

// Copy source metadata into symmetric send buffer
if (lane_id < num_topk_ranks) {
    st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
    #pragma unroll
    for (int i = 0; i < num_topk_ranks; ++i)
        dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;
}

// Copy `x_scales` into symmetric send buffer
#pragma unroll
for (int i = lane_id; i < num_scales; i += 32) {
    auto value = ld_nc_global(x_scales + token_idx * num_scales + i);
    #pragma unroll
    for (int j = 0; j < num_topk_ranks; ++j)
        st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
}
...
```

需要注意的是这段代码很巧妙,在store的时候采用了指令 `st.global.L1::no_allocate` 避免L1的 allocation。虽然注释说测试2也没看有allocate,但是我个人觉得这样一条指令会直接写通到HBM应该是不行的，应该是从L2Cache上直接优化处理。

```cpp
// `st.global.L1::no_allocate` will be translated into `ST.E.NA [addr16]` in SASS,
// which does not have cache allocation, and `CONSTANT` memory does not have coherence control,
// so we have to control them by queue semantics
#ifdef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif
```

最后再完成一些收尾工作，然后更新原指针针来释放锁等，同步。

##### 3.2.1.3.2 kRDMASenderCoordinator

这段代码主要实现了RDMA消费者和NVL生产者的通信逻辑，具体包括以下几个模块

**1. 初始化目标rank和专家范围:**
- 计算目标rank(dst_nvl_rank)范围
- 确定每个rank处理的专家范围(sm_expert_begin和 sm_expert_end)

**2. 使用循环不断处理RDMA通信的元数据从缓冲读取元数据:**
- 通过 `ld_volatile_global` 读取(rdma_channel_metadata)所存放的元数据信息
- 无数据处理时(如 meta_0, meta_1, meta_2, meta_3), 通过记录rank的数据量和其他信息处理

**3. 然后使用 `__syncwarp` 同步所有线程，确保所有线程一致地执行同步操作**

**4. 检查提供好的token数量再遍历发送NVL头部的位置**

**5. 调用 `sync_forwarder_smem()` 确保共享内存已被覆盖，为后续操作作准备**

**6. 转发NVL缓冲中区中的token到NVL缓冲区：**
- 判断收所到底是否某个RDMA rank
- 检查目标RDMA区各是否已可用，检测某个NVL缓冲区状态，并更新和处理数据
- 更新最尾指针针以准备同步

##### 3.2.1.3.3 kRDMAAndNVLForwarder

负责协调多个针发送warp，确保数据的同步和头尾指针的正确更新，使用共享内存进行进行状态返回它个循环一个个死循环，在循环中当每个从底retired的Channel全部标记 min_head 状态。在这个死循环 底次每个Channel都校平retired状态，并推出循环。另外，更新话端头尾指针针发送数据。最后有一个nanosleep允许其他warp有机会执行

##### 3.2.1.3.4 kForwarderCoordinator

负责协调多个针发送warp，确保数据的同步和头尾指针的正确更新，使用共享内存进行状态返回它个循环一个个死循环，在循环中当每个从底retired的Channel全部标记 min_head 状态。在这个死循环 底次每个Channel都校平retired状态，并推出循环。另外，更新话端头尾指针针发送数据。最后有一个nanosleep允许其他warp有机会执行

##### 3.2.1.3.5 kNVLReceivers

首先从从barrier中获取 src_nvl_rank ,并计算总的偏移量 total_offset,然后通过过循环检查每个lane,进行处理 num_tokens_to_recv第号token拷贝复制到最终缓冲区，包括 data、source meta、fp8 scale、topk_idx 和 topk_weights等。

最后dispatch函数复通过recv的过程

### 3.2.2 Combine流程

然后向Combine阶段的代码也类似，但是它可以复用dispatch_layout的信息，在 deep_ep/buffer.py 中定义的combine函数如下

```python
def combine(self, x: torch.Tensor,
            handle: Tuple,
            topk_weights: Optional[torch.Tensor] = None,
            config: Optional[Config] = None,
            previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
            allocate_on_comm_stream: bool = False) -> \
    Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
```

其中x为 `[num_tokens, hidden]` 的BF16数组, topk_weights 为 `[num_tokens, num_topk]` 的数组，记录MoE Gate函数得到的权重信息, handle 通信所对应的handler, 可以从dispatch函数获得, handle包含了如下信息

```cpp
// intranode handle
handle = (rank_prefix_matrix,
          channel_prefix_matrix,
          recv_channel_prefix_matrix,
          recv_src_idx, is_token_in_rank, send_head)

// internode handle
handle = (is_token_in_rank,
          rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
          recv_rdma_channel_prefix_matrix,
          recv_rdma_rank_prefix_sum,
          recv_gbl_channel_prefix_matrix,
          recv_gbl_rank_prefix_sum,
          recv_src_meta, send_rdma_head, send_nvl_head)
```

函数返回的结果包括 combined_x, combined_topk_weights 和 event 。

#### 3.2.2.1 Intranode_Combine

同样它会根据SM数量分配接收所有的SM, Channel数为SM数量的一半，一个Channel对应两个SM各负责其一，调整为发送为接收SM, 通过 `is_sender = sm_id % 2 == 0` 判断分工。因为在Combine阶段还有reduce的操作，为了保证store的时候对齐相加，因此按照int4来划分workload的方式

```cpp
constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
auto x_int4 = reinterpret_cast<const int4*>(x);
auto recv_x_int4 = reinterpret_cast<int4*>(recv_x);
```

它将根据每个rank的send_buffer来进行reduce接收操作。首先初始化用于接收的共享变量

```cpp
// Shared head, tail and retired flags for receiver warps
__shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
__shared__ volatile int warp_channel_tail_idx[num_recv_warps][kNumRanks];
__shared__ volatile bool warp_retired[num_recv_warps];
```

对于threadIdx < 32的第一个warp, 负责更新头尾头的头尾状态，并检查是否否所有的接收warp都已经处于 retired状态。如果都已经是retire状态就退出循环，否则根据其它warp的最小头 min_head 来更新

```cpp
// Reduce data
#pragma unroll
for (int i = recv_lane_id; i < hidden_int4; i += 32) {
    // Read from sources
    int4 recv_value_int4[kNumRanks];
    #pragma unroll
    for (int j = 0; j < num_topk_ranks; ++j)
        recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j])
    ...
    // Reduce all the dtypes
    for (int j = 0; j < num_topk_ranks; ++j) {
        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
        #pragma unroll
        for (int k = 0; k < kDtypePerInt4; ++k)
            values[k] += static_cast<float>(recv_value_dtypes[k]);
    }
    // Cast back to `dtype_t` and write
    int4 out_int4;
    auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
    #pragma unroll
    for (int j = 0; j < kDtypePerInt4; ++j)
        out_dtypes[j] = static_cast<dtype_t>(values[j]);
    recv_x_int4[token_idx * hidden_int4 + i] = out_int4;
}
```

#### 3.2.2.2 Internode_Combine

`internode::combine` 涉及RDMA和NVLINK操作更加复杂一些, Reduce操作定义了一个独立的函数 combine_token 计算方式和前一节完似, 然后Internode_combine warp分为如下4种WarpRole

```cpp
enum class WarpRole {
    kNVLSender,           //NVLINK上发送
    kNVLAndRDMAForwarder, //NVLink上和RDMA转发
    kRDMAReceiver,        //RDMA接收
    kCoordinator          //协调器
};
```

它将一个Channel解耦到2个SM

```cpp
const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
auto role_meta = [=]() -> std::pair<WarpRole, int> {
    auto warp_id = thread_id / 32;
    if (not is_rdma_receiver_sm) {
        if (warp_id < NUM_MAX_NVL_PEERS) {
            auto shuffled_warp_id = warp_id;
            shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
            return {WarpRole::kNVLSender, shuffled_warp_id};
        } else if (warp_id < NUM_MAX_NVL_PEERS + NumForwarders) {
            auto shuffled_warp_id = warp_id - NUM_MAX_NVL_PEERS;
            shuffled_warp_id = (shuffled_warp_id + channel_id) % NumForwarders;
            return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
        } else {
            return {WarpRole::kCoordinator, 0};
        }
    } else {
        if (warp_id < NUM_MAX_NVL_PEERS + NumForwarders) {
            return {WarpRole::kRDMAReceiver, warp_id};
        } else {
            return {WarpRole::kCoordinator, 0};
```

##### 3.2.2.2.1 kNVLSender

这段代码实现了NVLink通信中的数据发送逻辑，主要功能包括

- 初始化NVLink Channel和缓冲区
- 获取每个RDMA Channel的任务范围
- 送代获选数据，逐列所有有效数据
- 每次发送完毕再更新头尾指针并标记状态和缓冲区

##### 3.2.2.2.2 kNVLAndRDMAForwarder

主要看是将NVL上收到的数据从RDMA转发。首先调整好的指针位置，然后清除共享内存并等待同步。然后在从本地要发送RDMA之前，会调用 combine_token 进行一次reduce操作, z最后向再从RDMA接收发送数据

##### 3.2.2.2.3 kRDMAReceiver

使用 `get_channel_task_range` 函数获取当前Channel的任务范围 (`token_start_idx` 和 `token_end_idx`) 在循环中处理, 从底次会合作层 rdma_channel_data.recv_buffer(src_rdma_rank)加载数据, 然后执行reduce操作 `combine_token`。

##### 3.2.2.2.4 kCoordinator

同步共享状态，并更新RDMA rank和NVL rank的min_head

## 3.2 用于Decoding的低延迟Kernel

Decoding阶段为了降低延迟只使用了RDMA进行点到点通信。测试性能如下

| Dispatch #EP | Latency | RDMA bandwidth | Combine #EP | Latency | RDMA bandwidth |
|---|---|---|---|---|---|
| 8 | 163 us | 46 GB/s | 8 | 318 us | 46 GB/s |
| 16 | 173 us | 43 GB/s | 16 | 329 us | 44 GB/s |
| 32 | 182 us | 41 GB/s | 32 | 350 us | 41 GB/s |
| 64 | 186 us | 40 GB/s | 64 | 353 us | 41 GB/s |
| 128 | 192 us | 39 GB/s | 128 | 369 us | 39 GB/s |
| 256 | 194 us | 39 GB/s | 256 | 360 us | 40 GB/s |

这里的延迟应该指的实产网络环境使用报128个token作为一个batch(论文之256),8个routed expert,然后采用FP8 dispatch和BF16进行combine的处理方式的整体延迟，等实真果卡上也是超打准。

需要注意的是，在Decoding阶段占对达通信高吞极针对于降低小安全外突表现最好

在Decoding阶段采用了7降低延迟这个原因是使用CUDA Graph可以显性地调整, 论文文过到的两个microbatch overlap也有了对应的实现建议

![双 microbatch overlap 示意图：上半为传统需占用通信 SM 的 overlap（两条 stream 交错 Attention/Dispatch/MoE/Combine），下半为 DeepEP 不占用通信 SM 的 overlap，Dispatch/Combine 用后台 RDMA，把更多 SM 留给计算](assets/epcode-05.png)

通过一个钩子函数(hook),RDMA网络流量在后台进行，不会占用任何计算资源，只使用了本地SMs 的计算部分。Overlap的部分可以规避很麻烦，执行Decoding阶段EP并行时，通过get_buffer函数取获取缓冲区，并利用 low_latency_combine 方法法进行allotoall处理。

### 3.2.1 LowLatency Layout

在 `csrc\config.hpp` 中定义了一个个 LowLatencyBuffer 结构，dispatch和combine都分离了发送和接收Buffer, 添加了一些count和cnt原子操作的token_counter,然后combine阶段有一个 recv_flag Buffer

```cpp
struct LowLatencyBuffer {
    int num_clean_int = 0;

    void* dispatch_rdma_send_buffer = nullptr;
    void* dispatch_rdma_recv_data_buffer = nullptr;
    int* dispatch_rdma_recv_count_buffer = nullptr;

    void* combine_rdma_send_buffer = nullptr;
    void* combine_rdma_recv_data_buffer = nullptr;
    int* combine_rdma_recv_flag_buffer = nullptr;

    std::pair<void*, int> clean_meta() {
        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer);
        return {dispatch_rdma_recv_count_buffer, num_clean_int};
    }
};
```

LowLatencyLayout 采用了两个对称的缓冲区，交替使用减少待时时间，分为三组buffer: send、recv、signaling。

```cpp
size_t total_bytes = 0;
LowLatencyBuffer buffers[2];

template<typename dtype_t = void*> dtype_t* advance(const void* ptr, size_t count) {
    return reinterpret_cast<dtype_t*>(reinterpret_cast<uint8_t*>(ptr) + count);
}

LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank, int hidden, int num_r
    const int num_scales = hidden / 128;
    const int num_local_experts = num_experts / num_ranks;

    // Dispatch and combine layout:
    //  - 2 symmetric odd/even send buffers
    //  - 2 symmetric odd/even receive buffers
    //  - 2 symmetric odd/even signaling buffers
    ...
}
```

然后传输消息的的Size为:

```cpp
// Message size
EP_HOST_ASSERT(num_scales * sizeof(float) <= hidden);
size_t num_bytes_per_dispatch_msg = hidden + num_scales * sizeof(float) + sizeof(int4);
size_t num_bytes_per_combine_msg = hidden * sizeof(nv_bfloat16);
```

send、recv、signaling 缓存区的定义如下

```cpp
// Send buffer
size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_m
size_t combine_send_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_pe
size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);
total_bytes += send_buffer_bytes * 2;

// Symmetric receive buffers
// TODO: optimize memory usages
size_t dispatch_recv_data_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_b
size_t combine_recv_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_pe
size_t recv_buffer_bytes = std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);
EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);
total_bytes += recv_buffer_bytes * 2;

// Symmetric signaling buffers
size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);
size_t dispatch_recv_atomic_token_counter_bytes = num_local_experts * sizeof(int);
size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;
size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes + dispatch_recv_atomi
                                         combine_recv_flag_buffer_bytes);
total_bytes += signaling_buffer_bytes * 2;
```

### 3.2.2 低延迟Dispatch

调用的函数如下，注释有有一个非常关键的信息 `compatible with CUDA graph` 这也是为什么这份代码使用IBGDA的原因. 传统的RDMA需要以完成发送这, 并使用WRITE_WITH_IMM消息在这里主要就是为了完成传成CPU, CPU再进一步launch kernel, 这样的方式打断了CUDA graph使得延迟增大, 另一个关键还是 `double-batch overlapping`, 然后可以通过第三方开源的DeepGEMM实现MoE的专家知识计算.

```python
def low_latency_dispatch(self, hidden_states: torch.Tensor, topk_idx: torch.Tensor, num_max_dispatch_tokens_
    global _buffer

    # Do RDMA dispatch, using CUDA graph can restore some buffer status once you call `handle`.
    recv_hidden_states, recv_expert_count, handle, event, hook = \
        _buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
                                     async_finish=False, return_recv_hook=True)

    # NOTES: the actual tensor will not be received only if you call `hook()`,
    # it is useful for double-batch overlapping, but **without any SM occupation**
    # If you don't want to overlap, please set `return_recv_hook=False`.
    # Later, you can use our GEMM library in the meantime with a hidden dimension specified by `hidden_states`
    return recv_hidden_states, recv_expert_count, handle, event, hook
```

`num_max_dispatch_tokens_per_rank` 在一次batch的 `num_tokens=128`. 然后再调用 `csrc\deep_ep.cpp` 中的 Buffer::low_latency_dispatch 函数, 该函数首先分配 LowLatencyLayout, 然后按获得前面所示task完成后, 分配好packed tensor

```python
# Allocate packed tensors
packed_recv_x = torch.empty((num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, h
packed_recv_src_info = torch.empty((num_local_experts, num_ranks * num_max_dispatch_tokens_per_
packed_recv_layout_range = torch.empty((num_local_experts, num_ranks), dtype=torch.int64,...)
packed_recv_count = torch.from_blob(buffer.dispatch_rdma_atomic_token_counter, (num_local_exper
                                    (num_local_experts,), dtype=torch.int32).device...)
```

另外还分配了FP8精粒度量化时使用的Scale,并且考虑到TMA加载的优化需要保证num_token除除4

```python
# Allocate column-majored scales
EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number
packed_recv_x_scales = torch.empty((num_local_experts, num_ranks * num_max_dispatch_tokens_per_
packed_recv_x_scales = torch.transpose(packed_recv_x_scales, 1, 2);
```

Fine-grain quantization阶段的Scale-Factor如图所示，

![DeepSeek-V3 Figure 7：(a) 细粒度量化，Input/Weight 各配 Scaling Factor 送入 Tensor Core WGMMA；(b) 提高累加精度，FP8 GEMM 每 N_C=128 元素提升到 CUDA Core 做 FP32 高精度累加](assets/epcode-06.png)

**Figure 7 | (a)** We propose a fine-grained quantization method to mitigate quantization errors caused by feature outliers; for illustration simplicity, only Fprop is illustrated. **(b)** In conjunction with our quantization strategy, we improve the FP8 GEMM precision by promoting to CUDA Cores at an interval of N_C = 128 elements MMA for the high-precision accumulation.

然后就是调用 `csrc\kernels\internode_ll.cu` 的dispatch kernel。它通过一个 `phase` 变量判判是 `LOW_LATENCY_SEND_PHASE` 还是 `LOW_LATENCY_RECV_PHASE`。

#### 3.2.2.1 SEND PHASE

SEND阶段的Warp分为两类, 第一类先FP8转换和发送TopK token, 第二类为最后一个warp于读取topk_idx并给于per-expert信息.FP8转换的算法如下, 转换后将其写入到发送Buffer中。

```cpp
// FP8 cast
#pragma unroll
for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
    // Read and calculate local amax
    auto int4_value = __ldg(x_int4 + i);
    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
    float fp32_values[kNumElemsPerRead];
    float amax = kFP8Margin, scale, scale_inv;
    #pragma unroll
    for (int j = 0; j < kNumElemsPerRead; ++j) {
        fp32_values[j] = static_cast<float>(bf16_values[j]);
        amax = fmaxf(amax, fabsf(fp32_values[j]));
    }

    // Reduce amax and scale
    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
    amax = half_warp_reduce_max(amax); scale = kFP8Amax / amax;  scale_inv = amax / kFP8Amax;
    if (lane_id == 0) rdma_x_scales[i / kNumPerChannels / 128] = scale_inv;

    // Cast into send buffer
    int2 int2_value;
    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
    #pragma unroll
    for (int j = 0; j < kNumElemsPerRead; j += 2) {
        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
    }
```

然后通过调用IBGDA进行发送到指定的slot, 这样可以实现在AdaptiveRouting开启时发送不用保存, 发送完成后更新数据计数器

```cpp
// Issue IBGDA sends
if (dst_expert_idx >= 0) {
    int slot_idx = lane_id % 32 == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
    slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
    constexpr int dst_expert_idx_num_local_experts;
    constexpr int src_expert_local_idx = dst_expert_idx % num_local_experts;
    constexpr int src_ptr = reinterpret_cast<uint64_t>(rdma_x_int2);
    constexpr int dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
        dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
        rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
        slot_idx * num_bytes_per_msg;
    if (dst_rank != rank) {
        nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx,
    } else {
        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
        constexpr int dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
        constexpr int src_int4_ptr = reinterpret_cast<int4*>(src_ptr);
        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_n
    }
    // Increase counter after finishing
    __syncwarp();
    lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
}
```

最后一个Warp用于分配任务给不同的SM并处理Expert任务分发和同步。

#### 3.2.2.2 RECV PHASE

接收采用了sub warp空转轮空管理，首先会根据 responsible_expert_idx 计算 src_rank 和本地专家 local_expert_idx 然后通过 nvshmemi_ibgda_poll_recv(src_rank, local_expert_idx) 进行polling,然后再对Token和相应的scale数据获取source info。

### 3.2.3 低延迟Combine

同样也分为SEND和RECV两个阶段, 调用方式如下

```python
def low_latency_combine(self, hidden_states: torch.Tensor,
                        topk_idx: torch.Tensor, topk_weights: torch.Tensor, handle: Tuple):
    global _buffer

    # Do RDMA combine, compatible with CUDA graph (but you may restore some buffer status once you call `handl
    combined_hidden_states, event_overlap, hook = \
        _buffer.low_latency_combine(hidden_states, topk_idx, topk_weights, handle, ...)
```

#### 3.2.3.1 SEND PHASE

有一个注释的小错误, 实际上是执行了BF16的和IBGDA发送,而没有FP8 Cast:)

```cpp
// FP8 cast and issue IBGDA sends
if (responsible_expert_idx < num_experts) {
    constexpr int rank = responsible_expert_idx / num_local_experts;
    constexpr int local_expert_idx = responsible_expert_idx % num_local_experts;
    constexpr int global_expert_idx = rank * num_local_experts + local_expert_idx;
    constexpr int layout = _ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
    constexpr int local_x = reinterpret_cast<int4*>(rdma_recv_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
    constexpr int local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_pe
    constexpr int rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_x) +
        local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

    // Unpack layout
    int offset, num_tokens_to_send;
    unpack2(layout, num_tokens_to_send, offset);

    // Issue IBGDA send
    for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += n
        constexpr int rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_by
        constexpr int rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_x_vec_row + 4);

        // Copy directly to local rank, or copy to buffer and issue RDMA
        auto src_idx = _ldg(reinterpret_cast<int*>(dst_ptr));
        constexpr int buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
        if (dst_rank == rank) {
            constexpr int dst_ptr = reinterpret_cast<uint64_t>(local_x) + ...;
            UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_...
        } else {
            constexpr int dst_ptr = reinterpret_cast<uint64_t>(buf_ptr);
            UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, buf_int4, x_int4, ld_nc_global, st_...
            nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, hidden * sizeof(nv_bfloat16), dst_rank, l
        }
    }
}

// Put finishing flag
EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(kNumWarpsPerGroup * 32));
if (sub_warp_id == 1 and lane_id == 0) {
    while (ld_acquire_global(atomic_clear_flag) == 0);
    if (dst_rank != rank) {
        nvshmemi_ibgda_rma_p(rdma_recv_flag + global_expert_idx, 1, dst_rank, local_expert_idx, 0);
    } else {
        atomic_add_release_global(atomic_clear_flag, -1);
    }
    __syncwarp();
}
```

#### 3.2.3.2 RECV PHASE

也是通过Polling的方式获取 `nvshmemi_ibgda_poll_recv(src_rank, src_expert_idx)`; 然后执行reduce操作。

## 3.3 其它细节

### 3.3.1 文档针对行为的PTX指令

在 utils.cuh 中描述如下

```cpp
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA [addr].CONSTANT` in SASS,
// which does not have cache allocation, and `CONSTANT` memory does not have coherence control,
// so we have to control them by queue semantics
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t *ptr) {
    auto ret = ld_nc_global(const_cast<dtype_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}
```

ld.global.nc指令用于从Global Memory中加载数据到寄存器中不使用一致性缓存, 这样可以降低对L1 Cache的使用提高吞吐并提高性能。其实PTX文档中定义了这样的行为

> **ld.global.nc**
> Load a register variable from global state space via non-coherent cache.
>
> **Syntax**
> ```
> ld.global{.cop}.nc{.level::cache_hint}{.level::prefetch_size}.type  d, [a]{, cache-policy};
> ld.global{.cop}.nc{.level::cache_hint}{.level::prefetch_size}.vec.type  d, [a]{, cache-policy};
> ...
> ```
> ```
> .cop                   = { .ca, .cg, .cs };            // cache operation
> .level::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,
>                              .L1::evict_first, .L1::evict_last, .L1::no_allocate};
> .level::cache_hint     = { .L2::cache_hint };
> .level::prefetch_size  = { .L2::64B, .L2::128B, .L2::256B };
> .cop                   = { .ca, .cg, .cs };
> .type                  = { .b8, .b16, .b32, .b64, .b128,
>                            .u8, .u16, .u32, .u64,
>                            .s8, .s16, .s32, .s64,
>                            .f32, .f64 };
> ```

![PTX 文档中 ld.global.nc 指令截图：Load a register variable from global state space via non-coherent cache，以及 .cop / .level::eviction_priority / .level::prefetch_size 等修饰符语法](assets/epcode-07.png)

这些config的 `tests` 目前对用于测试和benchmark, 很显然对不同的组网和不同的性能需求要重新做performance tuning。既在L2Cache做了prefetch,同时又是使用了L1Cache的占用,而直接将数据保存入寄存器的。

### 3.3.2 Memory Order

在 utils.cuh 中还定义了大量的PTX操作, 特别是LD-ST都采用了 acquire/relaxed 并且在程序中大量使用, 进一步提高的处理效率。

### 3.3.3 nvshmem需的修改

在 third_party\nvshmem.patch 中对nvshmem库进行了一些修改, 主要有几点

1. 分离了SEND和RECV的CQ,并调整了相应的结构体
2. 修改了接收队列的CQ,同步消费者和收
3. 增加了polling的机制,减少接收者的等待
4. 修改了QP创建的顺序,确保不会创建到自身的连接

# 4. RoCE上运行DeepEP的挑战

## 4.1 DeepSeek用到的IB网络技术

DeepSeek的组网结构从论文 [Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning](https://arxiv.org/abs/2408.14158) [1] 可以看到用的是FatTree结构的网络

![Fire-Flyer FatTree 拓扑：Zone A 800 节点，20 台 spine 交换机 + 40 台 leaf 交换机（均为 40-port sw），GPU 节点与 storage 节点接入，下方 Zone B 同样 800 节点](assets/epcode-08.png)

其两层 FatTree 的三维立体结构示意如下：

![两层 FatTree 三维立体结构示意图：多个平面的 Spine sw / Leaf sw 与底部 GPU 服务器堆叠展开，展示 Fire-Flyer 集群的整体拓扑（zartbot 绘制）](assets/epcode-09.png)

但是这边这一些信息分析，在训练和Prefill阶段。基于nvshmem库使用的是RDMA_WRITE消息实产的内存语义，并没有采用WRITE_WITH_IMM因此这样了因此以我们了对adaptive-routing将会导致接收缓乱序，因此在README中做了一个 `(support may be added soon)`。而在Decoding阶段使用的低延迟kernel上是要支持adaptive-routing的，同时IBGDA直接对GPU发送队列, 而绕过了Normal Kernel通过NVLINK转发。

基于这两个信息我们知道，DeepSeek还是应该采用的FatTree组网的架构, 在训练和Prefill上是采用OpenSM配置了一些静态的路由避免乱序发生冲突路径, 而在Decoding的交换机上开启了Adaptive Routing.

## 4.2 RoCE使用DeepEP的问题

虽然论文提到了一句话理论上可以兼容RoCE, 但是实际上会遇到不少问题, 这是我们做维修需要的地方.

> DeepEP is fully tested with InfiniBand networks. However, it is **theoretically compatible** with RDMA over Converged Ethernet (RoCE) as well.

关于在RoCE中使用的一些个专题, 可以访问后面下连接, 最近近这十年, Mellanox在RoCE上确实有不少的问题。后面分几点从头详细分析。

> [RDMA…]

### 4.2.1 Multi-Rail和Rail-Only拓扑的问题

另一个由头出于有着以上的专家负载均衡, 统计上每个token选择本专家的概率率反应到相对均衡的效果。因此EP并行时候, 对于单个GPU, 首先根据Group idx直接将数据发送到Group idx对应的remote卡上和同Rank的GPU,然后再通过NVLink来传输。这正对应了 Rail-optimized 的组网结构:

![Rail-optimized 拓扑：顶层 Spine Switches，中间 Rail 1~K Switches，底部多个 High-bandwidth Domain（HBI Domain 1~M），每个域内 GPU 1~8 同 Rail 直连到对应 Rail Switch](assets/epcode-10.png)

主要问题是在RoCE网络中，由于路由的Hash冲突的问题是非常严重的，而针对老的Dense模型设计的，由于并行策略只有TP/DP/PP因此从流量上分析跨越Rail的流量很少

![GPT-1T MegatronLM 流量矩阵：(a) GPU 1 到 48 单个 pipeline stage，可见密集的 TP traffic 与稀疏的 DP traffic；(b) GPU 1 到 192 四个 pipeline stage，跨 stage 的 PP traffic 稀疏，右侧色标从 300 GB 到 1 KB，区分 Same Rail / Same HBD](assets/epcode-11.png)

然后Meta还提出了一种Rail-Only的做法

![Meta Rail-Only 拓扑：去掉了 spine 层，只保留 Rail 1~K Interconnect 直连各 High-bandwidth Domain（HBI 1~M），域内 GPU 1~8 通过 High-Bandwidth Interconnect 互联，跨域仅走同 Rail](assets/epcode-12.png)

但是这样在Decoding阶段就要通过NVLINK转发, 可以看一下DeepEP的Normal Kernel就用的这种发送方式, 而Decoding的TPOT性能是受到影响的。如果直接在这种的RoCE网络上使用DeepEP的IBGDA 2P通信, 网络中的Hash冲突就成立的问题.

理论其实一直存在, 在网络值这样的对约十不平衡, 甚至一个rDMA的疑难杂症就是所有报文全部选取一个Hash值发送出去而不能利用其它路径的问题.

## 4.3 真正的适合EP的RDMA over Ethernet

当然另一种潜在的解法, 例如使用Spectrum-X的解决方案是在这个网上开启Adaptive-Routing功能, 具体方案可以参考

> [浅谈英伟达的Spectrum-X以太网RDMA方案](#)

当然Spectrum-X案也有一些缺陷, 有一些则是知道Nvidia也在改善它, 例如比其它Lossless等...

另一种选择是像Meta在Llama3论文中讲的那样

> [Sigcomm论文解析 | Llama 3训练RoCE网络](#)

但是采用在Jericho/Ramon的那模式交换机, 但是很复杂, 这样的方案实际会增加相对的延迟和增加对的网络成本, 例如这些成本我个人一直在思考如何能不能干掉IB？

![Jericho/Ramon 深缓存交换机方案示意：NIC 经 RTSW 用 PFC 逐跳反压连到中间的 CTSW（内含 4 块 Jericho HBM 深缓存），再到对端 RTSW/NIC，代价是成本 +30% 且引入 22us 静态延迟](assets/epcode-13.png)

# 5. 关于DeepSeek-V3论文的建议

DeepSeek-V3 论文 3.5 节对 AI 硬件厂商提出的硬件设计建议原文截图如下：

![DeepSeek-V3 论文 3.5 Suggestions on Hardware Design 截图：指出当前 all-to-all 通信占用 H800 上 132 个 SM 中的 20 个（Communication Hardware 一节），列出 SM 承担的 forwarding/transporting/reduce/管理内存布局四类任务，呼吁厂商用类似 NVIDIA SHARP 的协处理器 offload 通信并统一 IB 与 NVLink 域](assets/epcode-14.png)

## 3.5 Suggestions on Hardware Design

Based on our implementation of the all-to-all communication and FP8 training scheme, we propose the following suggestions on chip design to AI hardware vendors.

### 3.5.1 Communication Hardware

In DeepSeek-V3, we implement the overlap between computation and communication to hide the communication latency during computation. This significantly reduces the dependence on communication bandwidth compared to serial computation and communication. However, the current communication implementation relies on expensive SMs (e.g., we allocate 20 out of the 132 SMs available in the H800 GPU for this purpose), which will limit the computational throughput. Moreover, using SMs for communication results in significant inefficiencies, as tensor cores remain entirely -utilized.

Currently, the SMs primarily perform the following tasks for all-to-all communication:

- **Forwarding** data between the IB (InfiniBand) and NVLink domain, while aggregating IB traffic destined for multiple GPUs within the same node from a single GPU.
- **Transporting** data between RDMA buffers (registered GPU memory regions) and input/output buffers.
- **Executing** reduce operations for **all-to-all combine**.
- **Managing** fine-grained memory layout during chunked data transferring to multiple experts across the IB and NVLink domain.

We aspire to see future vendors developing hardware that offloads these communication tasks from the valuable computation unit SM, serving as a GPU co-processor or a network co-processor like NVIDIA SHARP Graham et al. [2016]. Furthermore, to reduce application programming complexity, we aim for this hardware to unify the IB (scale-out) and NVLink (scale-up) networks from the perspective of the computation units. With this unified interface, computation units can easily accomplish operations such as **read**, **write**, **multicast**, and **reduce** across the entire IB-NVLink-unified domain via submitting communication requests based on simple primitives.

当然要DeepEP的代码, 也就意识到刚才为什么么会说了需要是同一种统一的通信语义了, 有一个类似ScaleUP和ScaleOut语义了。针对的的第二年就是上以太优化力面去年大概花了半年的时间在部完通我们已经完成了, 这些的内容能在DeepSeek这样很好的团队认可到.

# 6. 关于MoE模型的演进

其实这是一个更大的话题, 当前的对MoE的内存需求和需求和显存的需求是显示比较大的. 但是其实有很多方法能来优化模型加速的对内存的影响能比较更, 我们之前是一ScaleUP和ScaleOut语义, 我在之前是一个很大的话题...

进一步对MoE稀疏化，降低访问内存带宽

![MoE 稀疏化演进示意图：底部 Token1~4 先经 Gate 路由到 Expert Group 1/2/N，组内再经二级 Gate 选出具体 Expert（如 Expert 1-1/1-5/1-M、Expert 2-1/2-7/2-M），层级化稀疏路由以降低访存带宽](assets/epcode-15.png)

一个新的时代来DeepSeek开启了，如手上有一篇文章 《[FlashMLA性能简案](#)》 [3] 作者提到有一段话:

> 最近由出末来的Sod的东西, 相是在硬以为都完的东西了, 相比之前x1B+t差不上的性能提, 现在换成相较低较的hidow, 不过infra方面和其它有比列上单进降到更好的位置. 未来这个方面还有很多好玩玩的东西, 现现在做infra我我心里在不上得那么恐. 另一方面从算法之的层面来看, 现在, 就是在前沿的一定是占据了模型+infra+芯片三位一体来紧密配合的优势, 估计像清B这样的全机工程师真的很活不了, 大家也一起加油~

是的, 就是在前沿的一定是占据了模型+infra+芯片三位一体来紧密配合的优势, 估计像清B这样的全机工程师真的很活不了, 大家也一起加油~

另外弄言的是提醒一下, 基础设施施是更加更加不要要用DeepSeek-R1的模型带需求要来分析, 更多的是需要有更多是必须的更好前沿的模型的变化. 更下的想法会会进一步解决基础设施的的各种痛点, 后面几天大概想空一点后续继续分析一下FlashMLA和DeepGEMM吧~

---

## 参考资料

[1] Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning: https://arxiv.org/abs/2408.14158

[2] https://arxiv.org/pdf/2307.12169 · https://arxiv.org/pdf/2307.12169

[3] FlashMLA性能简案: https://zhuanlan.zhihu.com/p/26113545571

<!-- 说明: 本文正文由微信公众号原文(mid=2247493292)截图逐段重建。原文正文全部为图片公式与代码块, 转换器丢失。表格 2 张已重建; FP8 精度公式说明(Figure 7)与 ld.global.nc PTX 语法已转写。大段 C++/CUDA/Python 代码块从截图转录, 变量名/长行处可能有个别 OCR 误差, 以 GitHub deepseek-ai/DeepEP 源码为准。若干正文段落原文本身为手写口语化叙述, 截图分辨率下个别字词存疑, 已尽力转写。 -->
<!-- 公式存疑,需核对: 正文多处叙述性中文因截图分辨率存在个别字词不确定; 代码块内 constexpr/reinterpret_cast 等长行按截图可见部分转录, 精确类型签名以 DeepEP 源码为准 -->
