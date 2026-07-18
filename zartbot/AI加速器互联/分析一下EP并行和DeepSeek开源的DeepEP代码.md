# 分析一下EP并行和DeepSeek开源的DeepEP代码

> 作者: zartbot  
> 日期: 2025年2月26日 23:37  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493292&idx=1&sn=7af7db0f3d78f0fb52dc847934c7800e&chksm=f995f66ecee27f785788acda0075ba451a92619d66587b2d317e00b8ca23d20c691ee762ca23#rd

---

被好几个团队的人追着要渣B来分析一下DeepEP的工作, 公司内外的团队都有...简单的一句话说, 非常棒的工作,很多细节都值得学习.  但是还有一些硬件上的缺陷, 在DeepSeek-V3的论文中提出的建议要结合在一起看就会更清楚了. 我们还是由浅入深来谈谈EP并行, 并进一步分析一下这份出色的工作. 顺便展开讨论一下ScaleUP和ScaleOut网络遇到的难题和新的需求, DeepSeek采用了IB的技术, 虽然在github上说理论上兼容RoCE, 但是还有很多细节的地方需要探讨.

本文目录如下

```
1. EP并行概念2. SGlang EP并行实现3. DeepEP3.1 用于通信的Buffer3.2 用于训练和Prefill的高吞吐Kernel3.2.1 Dispatch通信流程3.2.1.1 Notify_Dispatch3.2.1.2 Intranode::dispatch3.2.1.3 Internode::dispatch3.2.1.3.1 kRDMASender3.2.1.3.2 kRDMASenderCoordinator3.2.1.3.3  kRDMAAndNVLForwarder3.2.1.3.4 kForwarderCoordinator3.2.1.3.5 kNVLReceivers3.2.2 Combine流程3.2.2.1 Intranode_Combine3.2.2.2 Internode_Combine3.2.2.2.1 kNVLSender3.2.2.2.2 kNVLAndRDMAForwarder3.2.2.2.3 kRDMAReceiver3.2.2.2.4 kCoordinator3.2 用于Decoding的低延迟Kernel3.2.1 LowLatency Layout3.2.2 低延迟Dispatch3.2.2.1 SEND PHASE3.2.2.2 RECV PHASE3.2.3 低延迟Combine3.2.3.1 SEND PHASE3.2.3.2 RECV PHASE3.3 其它细节3.3.1 文档外行为的PTX指令3.3.2 Memory Order3.3.3 nvshmem库的修改4. RoCE上运行DeepEP的挑战4.1 DeepSeek用到的IB网络技术4.2 RoCE使用DeepEP的问题4.2.1 Multi-Rail and Rail-Only拓扑的问题4.2.2 incast4.2.3 RC兼容4.2.4 In Network Computing如何做?4.3 真正的适合EP的RDMA over Ethernet5. 关于DeepSeek-V3论文的建议6. 关于MoE模型的演进
```

## 1. EP并行概念

DeepSeek MoE的原理和演进在以前的一篇文章详细分析过

[《详细谈谈DeepSeek MoE相关的技术发展》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493157&idx=1&sn=51c0e27a347dd3fe1ed868d87f667897&scene=21#wechat_redirect)

Expert Parallelism(EP并行)如下图所示:

![图片](assets/09d531d34708.png)

在DeepSeek-R1推理过程中, 为什么要实现EP并行? 从系统的角度来看, 单个专家的参数数据量为44MB,具体计算如下所示:

```
dim = 7168inter_dim = 2048tokens = 256e = Expert(dim, inter_dim)from ptflops import get_model_complexity_infoinput_tokens = (1,tokens,dim)flops, params = get_model_complexity_info(e, input_tokens, as_strings=True,print_per_layer_stat=True)Expert(  44.05 M, 100.000% Params, 11.28 GMac, 99.995% MACs,   (w1): Linear(14.68 M, 33.329% Params, 3.76 GMac, 33.328% MACs, in_features=7168, out_features=2048, bias=True)  (w2): Linear(14.69 M, 33.341% Params, 3.76 GMac, 33.340% MACs, in_features=2048, out_features=7168, bias=True)  (w3): Linear(14.68 M, 33.329% Params, 3.76 GMac, 33.328% MACs, in_features=7168, out_features=2048, bias=True)
```

如果一次加载44MB的专家参数权重, 而仅处理几个token是很不划算的, 同时多个专家放置在一台机器上, 很容相互之间加载干扰, 并且完全占满显存带宽. 整个系统的瓶颈在显存带宽上.

那么一个很朴素的想法就是通过多台机器并行, 把专家放置在多个机器上, 一方面等效的增加了显存的带宽, 另一方面也增加了专家参数权重的data locality. 然后用更大的batch(论文为256,开源代码为128)使得一次权重加载能够处理更多的token, 吞吐率就提升了.

但是难题就是需要大量的all2all的跨机通信, 此时如果网络通信的问题不解决好, EP并行的瓶颈就会转移到网络上, 这就是DeepEP工作的重要性了. 在谈DeepEP之前, 我们先来看看开源的EP并行实现.

## 2. SGlang EP并行实现

在RDMA网络中, 大规模组网下AlltoAll的通信效率都是一个非常大的问题, 因此开源推理软件如Sglang通过allreduce的方式来实现, 规避AlltoAll, 因此在开源社区如何用上DeepEP的工作, 还需要进一步分析.  这里对SGlang EP并行做一个简单的介绍, 来看一下`\python\sglang\srt\layers\moe\ep_moe\layer.py`中的EPMoE类, Sglang默认tp_size和ep_size相同, 然后它会根据自己的tp_rank来初始化和加载本地专家.

```
class EPMoE(torch.nn.Module):  def __init__():              self.num_experts = num_experts        assert self.num_experts % self.tp_size == 0                ## 根据节点的Rank计算出EP并行时, 该节点需要放置的专家.        self.num_experts_per_partition = self.num_experts // self.tp_size        self.start_expert_id = self.tp_rank * self.num_experts_per_partition        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
```

然后在该类的Foward函数中有详细的处理过程

```
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):     ## 初始化GroupGemm          ## 选择专家, 并根据输出的topk_ids进行重排序, 便于单个专家权重加载的时候复用内存处理多个token      topk_weights, topk_ids = select_experts()      reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(            topk_ids, self.num_experts      )            # 重排序数据的预处理过程      pre_reorder_triton_kernel[(hidden_states.shape[0],)]()            # 基于当前rank获取需要处理的segment      seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]      weight_indices_cur_rank = torch.arange(            0,            self.num_experts_per_partition,            device=hidden_states.device,            dtype=torch.int64,        )                   # 第一次GroupGemm做个Expert的Gate和Up-projection矩阵运算        gateup_output = self.grouped_gemm_runner(            a=gateup_input,            b=self.w13_weight,            c=gateup_output,            ....)                    # Expert 的激活函数处理         if self.activation == "silu":            silu_and_mul_triton_kernel[(gateup_output.shape[0],)]()                    # Expert的 down projection GroupGEMM处理         down_output = self.grouped_gemm_runner(            a=down_input,            b=self.w2_weight,            c=down_output,)                # Post重排        post_reorder_triton_kernel[(hidden_states.size(0),)](            down_output,            output,            src2dst,...)
```

您肯定很好奇, 实际的函数中并没有看到alltoall通信? 它在MLA阶段会执行DP并行, 有一次allgather. 然后EP并行时, 每个卡都将Token和本地专家进行运算, 等完成Post_reorder后, 其实同一个token需要经过Expert处理后的数据, 都已经分散在不同卡上, 此时进行一次allreduce即可.

这种宏观上的allgather+allreduce 替代alltoall的做法通信量相对于alltoall直接dispatch/combine大了很多, 但有时候还是会选择, 因为工业界一直对alltoall通信的效率解决的并不好, 特别是在各种RoCE环境中, 由于incast等问题带来的长尾延迟, 实际通信效率还没有直接做AG+AR好. 本文后面的章节会详细讨论这个话题.

当然刚注意到SGLang已经在开始整合DeepEP了~

![图片](assets/3f6fbf51fee5.png)

但是SGlang在RoCE上还有很多网络的调整需要处理, 后面详细介绍.

## 3. DeepEP

DeepEP是DeepSeek开源周第二天发布的一个项目, 解决了EP并行时AlltoAll通信的效率, 在Prefill和Training阶段提供高吞吐的能力, 同时在decoding阶段提供低延迟的传输. 同时还支持一些灵活的控制overlap计算和通信.

![图片](assets/484f785e5865.png)

通信库的核心就是在对Buffer的高效管理和使用, 并且在延迟和吞吐之间进行TradeOff. 下面我们分为三个小章节来介绍.

### 3.1 用于通信的Buffer

DeepEP对外的工作原理是, 先通过`get_buffer`函数申请一段buffer, 并且可以通过`Buffer.set_num_sms(sm_num)`来设置需要用到的SM.

```
from deep_ep import Buffer, EventOverlap_buffer: Optional[Buffer] = Nonedef get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:    # 计算NVLink和RDMA需要多少Bytes    num_nvl_bytes, num_rdma_bytes = 0, 0    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)    # 为Buffer分配内存    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)    return _buffer
```

在`\deep_ep\buffer.py`中定义了一些buffer的 config

```
 # Intranodeif num_ranks <= 8:     return Config(Buffer.num_sms, 6, 256, 6, 128)# Internode config_map = {    16: Config(Buffer.num_sms, 16, 288, 20, 128),    24: Config(Buffer.num_sms, 8, 288, 32, 128),    32: Config(Buffer.num_sms, 8, 288, 32, 128),    64: Config(Buffer.num_sms, 20, 288, 28, 128),    128: Config(Buffer.num_sms, 20, 560, 32, 128),    144: Config(Buffer.num_sms, 32, 720, 12, 128),    160: Config(Buffer.num_sms, 28, 720, 12, 128),  }
```

后面四个参数定义在`csrc\config.hpp`中定义

```
  struct Config {    int num_sms;    int num_max_nvl_chunked_send_tokens;    int num_max_nvl_chunked_recv_tokens;    int num_max_rdma_chunked_send_tokens;    int num_max_rdma_chunked_recv_tokens; }
```

这些config在`tests\`目录下的几个测试例中看到, 应该是专门做过performance tuning, 但是在H20/H100等其它实例上, 不同的RDMA网络配比和不同的NVLink带宽, 还需要一定的修改. 未来或许需要针对不同的设备定义单独的config_file.

Buffer类型在`csrc\kernel\buffer.cuh`中定义了三类, 通过这些结构体，可以灵活地管理 GPU 内存缓冲区，支持多种内存管理和访问模式

`Buffer`  管理单个缓冲区，支持基本的内存分配和访问

`AsymBuffer`管理多个缓冲区，支持多个Rank

`SymBuffer` Decouple模式分别设置`send_buffer`和`recv_buffer`, 非解偶模式调用`buffer`

在`csrc\deep_ep.cpp`中定义了Buffer类, 构造函数 `Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode)` 初始化缓冲区然后定义了一系列方法

用于训练和Prefill的高吞吐Normal Kernel `[intranode/internode]_[dispatch/combine]`

用于推理Decoding的低延迟Kernel `low_latency_[dispatch/combine]`

基于topk_idx和num_experts产生dispatch layout的函数`get_dispatch_layout`

定义了一个任务fifo队列和移动fifo slot的私有方法`move_fifo_slots()`

Task Fifo Size如下所示:

```
    // Task fifo memory    int64_t fifo_bytes = sizeof(int) * NUM_MAX_FIFO_SLOTS;    int64_t buffer_ptr_bytes = sizeof(void*) * NUM_MAX_NVL_PEERS;    int64_t task_ptr_bytes = sizeof(int*) * NUM_MAX_NVL_PEERS;
```

在Host侧还有一些私有Counter用于记录MoE专家相关的信息

```
    // Host-side MoE info    volatile int* moe_recv_counter = nullptr;    int* moe_recv_counter_mapped = nullptr;    // Host-side expert-level MoE info    volatile int* moe_recv_expert_counter = nullptr;    int* moe_recv_expert_counter_mapped = nullptr;    // Host-side RDMA-level MoE info    volatile int* moe_recv_rdma_counter = nullptr;    int* moe_recv_rdma_counter_mapped = nullptr;
```

### 3.2 用于训练和Prefill的高吞吐Kernel

训练和Prefill阶段batch相对较大对延迟不是很敏感, 因此主要是最大化的利用带宽, README中的测试数据显示基本上带宽已经打满

Type

Dispatch #EP

Bottleneck bandwidth

Combine #EP

Bottleneck bandwidth

Intranode

8

153 GB/s (NVLink)

8

158 GB/s (NVLink)

Internode

16

43 GB/s (RDMA)

16

43 GB/s (RDMA)

Internode

32

44 GB/s (RDMA)

32

47 GB/s (RDMA)

Internode

64

46 GB/s (RDMA)

64

45 GB/s (RDMA)

看了DeepEP的通信库, 更加能够理解MoE Group设计的方法, 从原始论文的3.2可以知道, 它在训练阶段采用EP=64的方式并行, 正好8台H800的机器, 然后MoE分为8个Group并根据GroupScore选择4台进行通信, 这也就是论文所述的TOken的分发限制在4个节点.

另一方面由于有算法上的专家负载均衡, 统计上每个Token选择专家的概率应该是相对均衡的.  因此从通信上来看, 对于单个GPU, 首先根据Group idx直接将数据发送到该Group idx对应的remote节点相同Rank的GPU, 然后再通过机内NVLink通信. 在DeepEP代码中通过`intranode(NVLink)`和`internode(RDMA)`来区分这两种通信模式.

在调用dispatch之间通过`get_dispatch_layout`获得需要发送的Token,哪些要通过RDMA,以及token是否需要传输到某个Rank

```
 # Calculate layout before actual dispatch    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \        _buffer.get_dispatch_layout(topk_idx, num_experts,                                    previous_event=previous_event, async_finish=True,                                    allocate_on_comm_stream=previous_event is not None)
```

其中topk_idx为MoEGating函数产生的`[num_tokens, num_topk]`数组, 通过dispatch_layout输出的结果为

num_tokens_per_rank: `[num_ranks]`的数组, 每个Rank需要发送的token数量

num_tokens_per_rdma_rank: `[num_rdma_ranks]`的数组, 在每个RDMA Rank上需要发送的token数量

num_tokens_per_expert: `[num_experts]`数组, 基于每个专家粒度的token数量统计.

is_token_in_rank: `[num_tokens, num_ranks]`二维数组, 数据类型为bool型, 标记token是否从某个rank发送.

#### 3.2.1 Dispatch通信流程

dispatch具体的通信流程如下图所示:

![图片](assets/136a8a99a805.png)

详细代码在`csrc\deep_ep.cpp`中, 以`intranode_dispatch`为例

```
Buffer::intranode_dispatch(  const torch::Tensor& x,//Tensor数据xconststd::optional<torch::Tensor>& x_scales, //用于FP8缩放的值conststd::optional<torch::Tensor>& topk_idx,// MoE Gating产生的专家indexconststd::optional<torch::Tensor>& topk_weights, //MoE Gating产生的weightconststd::optional<torch::Tensor>& num_tokens_per_rank, //每个Rank的token数const torch::Tensor& is_token_in_rank, //标记token是否从某个rank发送conststd::optional<torch::Tensor>& num_tokens_per_expert, //基于专家的token数统计int cached_num_recv_tokens, //需要缓存的token数conststd::optional<torch::Tensor>& cached_rank_prefix_matrix,  //[num_ranks,num_ranks]矩阵conststd::optional<torch::Tensor>& cached_channel_prefix_matrix,//[num_ranks, num_channels]的矩阵int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream //是否需要异步执行和分配通信的cudastream) {
```

当`cached_rank_prefix_matrix`参数有值时, `cached_mode=True`. 然后根据Config中定义的SM数量定义Channel, 一个channel使用两个block, 偶数block用于发送, 奇数用于接收, 因此Channel数为config.num_sms的一半, 然后会对输入参数进行一系列validation.  如果`allocate_on_comm_stream==True`则会创建comm_stream, 并等待compute stream或者参数中的previous_event完成

```
    auto compute_stream = at::cuda::getCurrentCUDAStream();    if (allocate_on_comm_stream) {        EP_HOST_ASSERT(previous_event.has_value() and async);        at::cuda::setCurrentCUDAStream(comm_stream);    }    // Wait previous tasks to be finished    if (previous_event.has_value()) {        stream_wait(comm_stream, previous_event.value());    } else {        stream_wait(comm_stream, compute_stream);    }
```
3.2.1.1 Notify_Dispatch
然后通过`notify_dispatch`函数launch kernel执行`notify_dispatch`, 通过多个节点交换将发送信息同步,并统计需要接收的Token数目以及统计几个prefix_matrix

```
//初始化接收counter*moe_recv_counter = -1, *moe_recv_rdma_counter = -1;for (int i = 0; i < num_local_experts; ++ i)    moe_recv_expert_counter[i] = -1;//调用notify_dispathinternode::notify_dispatch(  num_tokens_per_rank->data_ptr<int>(), //每个Rank需要发送的token数量  moe_recv_counter_mapped,   num_ranks,  num_tokens_per_rdma_rank->data_ptr<int>(),  //dispatch_layout的结果,在每个RDMA Rank上需要发送的token数量  moe_recv_rdma_counter_mapped,  num_tokens_per_expert->data_ptr<int>(), //dispatch_layout的结果,基于专家的token数统计  moe_recv_expert_counter_mapped,   num_experts,  is_token_in_rank.data_ptr<bool>(), //[num_tokens, num_ranks]二维数组, 数据类型为bool型, 标记token是否从某个rank发送.  num_tokens, num_channels, //token数和Channel数  hidden_int4, num_scales, num_topk, expert_alignment,  rdma_channel_prefix_matrix.data_ptr<int>(), //{num_rdma_ranks, num_channels}  recv_rdma_rank_prefix_sum.data_ptr<int>(), //{num_rdma_ranks}  gbl_channel_prefix_matrix.data_ptr<int>(), //{num_ranks, num_channels}  recv_gbl_rank_prefix_sum.data_ptr<int>(), //{num_ranks}  rdma_buffer_ptr,   config.num_max_rdma_chunked_recv_tokens,  buffer_ptrs_gpu,   config.num_max_nvl_chunked_recv_tokens,  task_fifo_ptrs_gpu,   head, rank, comm_stream,  config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),  num_nvl_bytes, low_latency_mode);
```

当`sm_id == 0`时：首先全局同步和清理缓冲区,然后将每个rank和expert的token数量通过`nvshmem_int_put_nbi`发送到RDMA ranks. 然后将计算好的token数量发送到同一节点内的其他ranks，使用`buffer_ptrs`和`nvl_send_buffer`. 接下来计算各个rank和expert的token数量的前缀和，更新`recv_rdma_rank_prefix_sum`和 `recv_gbl_rank_prefix_sum`. 最后更新全局计数器`moe_recv_counter_mapped`和`moe_recv_expert_counter_mapped`, 最终再次使用`nvshmem_barrier_with_same_gpu_idx`和`barrier_device`确保所有线程完成任务 当`sm_id != 0`时,计算与channel相关的元数据，使用 warp_reduce_sum 进行warp级别的归约操作，计算每个channel的token数量同时计算`rdma_channel_prefix_matrix`和`gbl_channel_prefix_matrix`的前缀和，确保后续任务的调度和分配

在CPU上会根据notify的结果分配新的Tensor

```
    // Allocate new tensors    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();    auto recv_src_meta = std::optional<torch::Tensor>();    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();    auto send_rdma_head = std::optional<torch::Tensor>();    auto send_nvl_head = std::optional<torch::Tensor>();    if (not cached_mode) {        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));    }
```

最后调用`dispatch`函数执行分发, dispatch函数在RDMA和NVLINK的实现分别在`csrc\kernels\internode.cu`和`csrc\kernels\intranode.cu` 主要目的就是将token, topk_idx和weight放入缓冲区.
3.2.1.2 Intranode::dispatch
`intranode::dispatch`在NVLINK上执行收发, 首先它会根据`const bool is_sender = sm_id % 2 == 0;`判断是sender还是reciver. 然后计算分配每个线程需要处理的channel和rank.

1. 发送逻辑：

初始化发送参数，包括计算任务范围、队列起始和结束偏移量

检查目标队列是否为空闲状态

分批发送数据，包括token、topk索引、权重等，并更新尾指针

如果超时未完成发送，打印错误信息并终止程序

2. 接收逻辑：

初始化接收参数，计算偏移量和需要接收的数据量

等待发送者写入数据

从发送缓冲区复制数据到目标缓冲区，包括token、源索引、topk索引和权重等

更新头指针，继续处理剩余数据直到全部接收完毕

如果超时未完成接收，打印错误信息并终止程序
3.2.1.3 Internode::dispatch
`internode::dispatch`涉及RDMA上的操作, 相对更加复杂, 它分配了多种WarpRole

```
    enum class WarpRole {        kRDMASender, //负责将本地数据通过RDMA发送到远程节点        kRDMASenderCoordinator, //管理RDMA发送的进度与同步        kRDMAAndNVLForwarder, //将RDMA接收的数据转发到本地NVLink缓冲区        kForwarderCoordinator, //全局协调转发任务        kNVLReceivers //从NVLink缓冲区读取数据并写入最终接收数组    };
```
3.2.1.3.1 kRDMASender
首先通过`get_channel_task_range`获取任务范围, 得到起始和结束的token索引`token_start_idx`, `token_end_idx`. 然后通过将`rdma_send_channel`的tail指针设置为0清除share memory.并将发送的下一个token的idx指向`token_start_idx`.

然后将这个Channel需要发送的token数量以负数`-value-1`的方式发送给其它节点, 然后同步等待.

```
for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {    // 根据不同的lane_id设置send_buffer的内容    nvshmemx_int_put_nbi_warp(rdma_channel_meta.recv_buffer(rdma_rank), rdma_channel_meta.send_buffer(dst_rdma_rank), NUM_MAX_NVL_PEERS * 2 + 2,                              translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank));}nvshmem_fence();sync_rdma_sender_smem();
```

然后开始迭代处理Token, 将token拷贝到buffer中, 获取锁并更新尾指针. 然后复制数据到缓冲区, 并广播元数据和其他相关信息（如`x`、`x_scales`、`topk_idx`和`topk_weights`）,复制的代码如下

```
// Copy `x` into symmetric send buffer            auto st_broadcast = [=](constint key, const int4& value) {                #pragma unroll                for (int j = 0; j < num_topk_ranks; ++ j)                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);            };            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);            #pragma unroll            for (int i = 0; i < num_topk_ranks; ++ i)                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;            // Copy source metadata into symmetric send buffer            if (lane_id < num_topk_ranks)                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);            #pragma unroll            for (int i = 0; i < num_topk_ranks; ++ i)                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;            // Copy `x_scales` into symmetric send buffer            #pragma unroll            for (int i = lane_id; i < num_scales; i += 32) {                auto value = ld_nc_global(x_scales + token_idx * num_scales + i);                #pragma unroll                for (int j = 0; j < num_topk_ranks; ++ j)                    st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);            }            #pragma unroll            for (int i = 0; i < num_topk_ranks; ++ i)                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;            // Copy `topk_idx` and `topk_weights` into symmetric send buffer            #pragma unroll            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {                auto rank_idx = i / num_topk, copy_idx = i % num_topk;                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);            }
```

需要注意的是这段代码很巧妙, 在store的时候采用了指令`st.global.L1::no_allocate`避免L1的allocation, 虽然注释说猜测L2也没有allocate, 但是我个人觉得这样一条指令直接写透到HBM应该是不行的, 应该是会在L2Cache上放置然后write back.

```
// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS,// which does not have cache allocation (obviously in L1, I guess not in L2 too)#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS#define ST_NA_FUNC "st.global.L1::no_allocate"#else#define ST_NA_FUNC "st.global"#endif
```

最后完成一些收尾工作, 例如更新尾指针并释放锁等. 然后同步.
3.2.1.3.2 kRDMASenderCoordinator
这段代码实现了RDMA发送协调者的逻辑，主要功能包括：

检查和同步共享内存

计算每个RDMA rank需要发送的token数量

迭代所有RDMA rank，检查是否有未处理的token，并发送RDMA请求

更新发送状态并确保数据一致性
3.2.1.3.3  kRDMAAndNVLForwarder
这段代码主要实现了RDMA消费者和NVL生产者的通信逻辑，具体包括以下几个模块：

1.初始化目标rank和专家范围：

计算目标NVL rank (dst_nvl_rank) 和 RDMA rank (dst_rank)

确定每个rank负责的专家范围 (dst_rank_expert_begin 和 dst_rank_expert_end)

等待RDMA通道元数据到达：

2.使用循环不断检查RDMA通道的元数据是否准备好

通过`ld_volatile_global`读取rdma_channel_meta的接收buffer读取元数据

元数据包含四个值 (meta_0, meta_1, meta_2, meta_3)，用于确定接收token的数量和其他信息,

如果元数据满足条件，则进行处理并通知NVL ranks

3.然后使用`__syncwarp()`同步所有线程，确保所有线程一致地执行后续操作

4.根据接收到的token数量调整发送NVL头部的位置

5.调用 `sync_forwarder_smem()` 确保共享内存已被清理，为后续操作做准备

6.转发RDMA缓冲区中的token到NVL缓冲区：

使用轮询方式选择下一个源RDMA rank

检查目标队列是否为空或有足够的空间

将RDMA缓冲区中的token复制到NVL缓冲区，并更新相关指针

处理超时检查和队列满的情况

7.更新头尾指针并同步线程：

更新RDMA和NVL通道的头尾指针

再次同步线程并标记任务完成
3.2.1.3.4 kForwarderCoordinator
负责协调各个转发器warp，确保数据的同步和头尾指针的正确更新，使用共享内存进行通信和状态跟踪它内部有一个死循环, 在循环中找到最小的没有retired的Channel的头指针`min_head`, 如果所有值都是最大值,则判断所有的Channel都处于retired状态,并推出循环. 否则, 更新远端头指针并发送数据.最后有一个nanosleep允许其他warp有机会执行.
3.2.1.3.5 kNVLReceivers
首先从从barrier结果中获取`src_nvl_rank`，并计算总的偏移量 `total_offset`, 然后通过循环检查每个lane, 并计算`num_tokens_to_recv` 然后通过一个条件为`num_tokens_to_recv>0`的循环拷贝搬迁数据,  包括`data`,`source meta`,`fp8 scale`,`topk_idx`和`topk_weights`等.

最后dispatch函数返回recv的数据

```
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event
```

#### 3.2.2 Combine流程

然后Combine阶段的代码也类似, 但是它可以复用dispatch_layout的信息, 在`deep_ep/buffer.py`中定义的combine函数如下:

```
    def combine(self, x: torch.Tensor, handle: Tuple,                topk_weights: Optional[torch.Tensor] = None,                config: Optional[Config] = None,                previous_event: Optional[EventOverlap] = None, async_finish: bool = False,                allocate_on_comm_stream: bool = False) -> \            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
```

其中 x为`[num_tokens, hidden]`的BF16数组, topk_weights: `[num_tokens, num_topk]`的数组, 记录MoE Gate函数得到的权重信息. `handle`通信用的handler, 可以从dispatch函数获得, handle包含了如下信息

```
# intranode handlehandle = (rank_prefix_matrix,      channel_prefix_matrix,      recv_channel_prefix_matrix,      recv_src_idx, is_token_in_rank, send_head)# internode handlehandle = (is_token_in_rank,    rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,    recv_rdma_channel_prefix_matrix,     recv_rdma_rank_prefix_sum,     recv_gbl_channel_prefix_matrix,     recv_gbl_rank_prefix_sum,    recv_src_meta, send_rdma_head, send_nvl_head)
```

函数返回的结果包含`combined_x`,`combined_topk_weights`和`event`.
3.2.2.1 Intranode_Combine
同样也是根据SM数量分配接收和发送的SM, Channel数为SM数量的一半, 一个Channel分别由两个SM负责收发, 偶数为发送SM,奇数为接收SM, 通过`is_sender = sm_id % 2 == 0`判断, 因为在Combine阶段还有reduce的操作, 为了保证store的时候尽量高效, 使用了按照int4来切分workload的方式

```
    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);    int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);    auto x_int4 = reinterpret_cast<const int4*>(x);    auto recv_int4 = reinterpret_cast<int4*>(recv_x);
```
发送SM的功能如下
1.计算每个线程所属的warp和rank, 并确定其在rank中的位置

`num_send_warps`：发送线程的warp数量, `num_send_warps_per_rank`：每个Rank负责的warp数量,`num_threads_per_rank`：每个秩的线程数量, `send_thread_id`、`send_lane_id`、`send_rank_id` 和 `send_warp_id_in_rank` 分别表示当前线程ID、线程在warp中的位置、线程所属的rank以及线程在rank内的warp ID.

2. 根据Rank ID计算缓冲区的指针, 用于后续数据访问

`ptr`：指向当前秩的缓冲区起始地址, `channel_head_idx`、`channel_tail_idx`、`channel_x_buffers`、`channel_src_idx_buffers` 和 `channel_topk_weights_buffers` 分别表示Channel的头部索引、尾部索引、数据缓冲区、源索引缓冲区和topk权重缓冲区.

3.根据prefix matrix计算每个线程需要处理的token范围

`rank_offset` 和 `num_rank_tokens`：计算rank的偏移量和token数量, `channel_offset` 和 `num_channel_tokens`：计算channel的偏移量和token数量, `token_start_idx` 和 `token_end_idx`：确定当前线程要处理的token范围.

4. 基于`token_start_idx` 和 `token_end_idx`按照chunk迭代发送数据

首先获取一个empty slot, 然后使用`UNROLLED_WARP_COPY` 宏复制数据到目标缓冲区, 再发送`source index`和`topk weights` 到 `channel_src_idx_buffers` 和 `channel_topk_weights_buffers`

最后更新尾指针 `channel_tail_idx`
接收SM的功能如下
其中一个warp用于队列更新, 其他的用于reduce接收的数据. 首先初始化用于接收的共享变量

```
        // Shared head, tail and retired flags for receiver warps        __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];//每个warp的channel头索引        __shared__ volatile int channel_tail_idx[kNumRanks];  //每个Rank的channel尾索引        __shared__ volatile bool warp_retired[num_recv_warps]; //每个接收 warp 是否已Retried的标志
```

对于threadid < 32的第一个warp, 负责更新队列头尾的idx, 并检查是否所有的接收warp已经处于retired状态, 如果都已经是retire状态退出循环. 否则根据其它warp的最小头索引`min_head`更新.

其他的warp则进行接收数据的reduce操作,

```
 // Reduce data#pragma unrollfor (int i = recv_lane_id; i < hidden_int4; i += 32) {     // Read buffers     int4 recv_value_int4[kNumRanks];     #pragma unroll     for (int j = 0; j < num_topk_ranks; ++ j)         recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);     // Reduce all-to-all results     float values[kDtypePerInt4] = {0};     #pragma unroll     for (int j = 0; j < num_topk_ranks; ++ j) {         auto recv_value_dtypes = reinterpret_cast<constdtype_t*>(&recv_value_int4[j]);         #pragma unroll         for (int k = 0; k < kDtypePerInt4; ++ k)             values[k] += static_cast<float>(recv_value_dtypes[k]);     }     // Cast back to `dtype_t` and write     int4 out_int4;     auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);     #pragma unroll     for (int j = 0; j < kDtypePerInt4; ++ j)         out_dtypes[j] = static_cast<dtype_t>(values[j]);     recv_int4[token_idx * hidden_int4 + i] = out_int4; }
```
3.2.2.2 Internode_Combine
`internode::combine`涉及RDMA和NVLINK操作更加复杂一些, Reduce操作定义了一个独立的函数`combine_token`计算方式和前一节类似, 然后Internode_combine warp分为如下4种WarpRole

```
    enum class WarpRole {        kNVLSender, //从NVLINK上发送        kNVLAndRDMAForwarder,//NVLink到RDMA转发        kRDMAReceiver, //RDMA接收        kCoordinator //协调器    };
```

它将一个Channel 解耦到2个SM

```
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;    auto role_meta = [=]() -> std::pair<WarpRole, int> {        auto warp_id = thread_id / 32;        if (not is_rdma_receiver_sm) {            if (warp_id < NUM_MAX_NVL_PEERS) {                auto shuffled_warp_id = warp_id;                shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;                return {WarpRole::kNVLSender, shuffled_warp_id};            } elseif (warp_id < NUM_MAX_NVL_PEERS + kNumForwarders) {                auto shuffled_warp_id = warp_id - NUM_MAX_NVL_PEERS;                shuffled_warp_id = (shuffled_warp_id + channel_id) % kNumForwarders;                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};            } else {                return {WarpRole::kCoordinator, 0};            }        } else {            if (warp_id < NUM_MAX_NVL_PEERS + kNumForwarders) {                return {WarpRole::kRDMAReceiver, warp_id};            } else {                return {WarpRole::kCoordinator, 0};            }        }    }();
```
3.2.2.2.1 kNVLSender
这段代码实现了NVLink通信中的数据发送逻辑。主要功能包括：

初始化NVLink Channel和缓冲区

获取每个RDMA Channel的任务范围

迭代发送数据块，直到所有任务完成

每次发送时检查是否有空闲槽位，并同步发送数据到目标缓冲区
3.2.2.2.2 kNVLAndRDMAForwarder
主要是将NVLink上收到的数据从RDMA转发. 首先调整NVL Buffer的指针位置, 然后清除共享内存并同步. 然后在NVLink向RDMA转发之前, 会调用`combine_token`进行一次reduce操作. z最后再从RDMA上发送数据.
3.2.2.2.3 kRDMAReceiver
使用`get_channel_task_range`函数获取当前Channel的任务范围（`token_start_idx`和`token_end_idx`）在循环中迭代,从rdma_channel_data.recv_buffer(src_rdma_rank)加载数据, 然后执行reduce操作`combine_token`
3.2.2.2.4 kCoordinator
同步共享内存状态, 并更新RDMA rank和NVL rank的min_head

### 3.2 用于Decoding的低延迟Kernel

Decoding阶段为了降低延迟只使用了RDMA进行点到点通信, 测试性能如下

Dispatch #EP

Latency

RDMA bandwidth

Combine #EP

Latency

RDMA bandwidth

8

163 us

46 GB/s

8

318 us

46 GB/s

16

173 us

43 GB/s

16

329 us

44 GB/s

32

182 us

41 GB/s

32

350 us

41 GB/s

64

186 us

40 GB/s

64

353 us

41 GB/s

128

192 us

39 GB/s

128

369 us

39 GB/s

256

194 us

39 GB/s

256

360 us

40 GB/s

这里的延迟应该是指的生产网络环境中按照128个token作为一个batch(论文是256), 8个routed expert, 然后采用FP8 dispatch和BF16进行combine的处理方式的整体延迟, 带宽基本上也能打满.

需要注意的是, 在Decoding阶段点到点通信高负载时由于路由冲突会带来拥塞, 需要打开IB的自适应路由的功能, 另一方面如果decode阶段的流量相对轻载时, Adaptive routing可能会导致一些额外的延迟, 此时可以采用静态路由的方式, 这些内容我们将在下一章节进行详细分析.

在Decoding阶段采用了IBGDA的原因是避免CPU参与, 使得CUDA Graph可以直接调度, 论文中提到的两个microbatch overlap也有了对应的实现建议

![图片](assets/ea1f0ebc6e16.png)

通过一个钩子函数(hook), RDMA网络流量在后台进行，不会占用任何 GPU SMs 的计算部分.Overlap的部分可以根据负载调整. 执行Decoding阶段EP并行时, 首先也要通过get_buffer函数获取缓冲区, 并利用`low_latency_dispatch`和`low_latency_combine`方法进行alltoall处理.

#### 3.2.1 LowLatency Layout

在`csrc\config.hpp`中定义了一个`LowLatencyBuffer`结构, dispatch和combine都分离了发送和接收Buffer, 添加了接收count和rdma原子操作的token_counter.然后combine阶段有一个`recv_flag` buffer

```
struct LowLatencyBuffer {    int num_clean_int = 0;    void* dispatch_rdma_send_buffer = nullptr;    void* dispatch_rdma_recv_data_buffer = nullptr;    int* dispatch_rdma_recv_count_buffer = nullptr;    int* dispatch_rdma_atomic_token_counter = nullptr;    void* combine_rdma_send_buffer = nullptr;    void* combine_rdma_recv_data_buffer = nullptr;    int* combine_rdma_recv_flag_buffer = nullptr;    std::pair<int*, int> clean_meta() {        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer);        return {dispatch_rdma_recv_count_buffer, num_clean_int};    }};
```

LowLatencyLayout 采用了两个对称的缓冲区, 交替使用减少等待时间, 分为三组buffer:`send`,`recv`,`signaling`,

```
struct LowLatencyLayout {    size_t total_bytes = 0;    LowLatencyBuffer buffers[2];    template <typenameout_ptr_t = void*, typenamecount_ptr_t = uint8_t*, typenamein_ptr_t = void*>    out_ptr_t advance(constin_ptr_t& ptr, size_t count) {        returnreinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);    }    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {        constint num_scales = hidden / 128;        constint num_local_experts = num_experts / num_ranks;        // Dispatch and combine layout:        //  - 2 symmetric odd/even send buffer        //  - 2 symmetric odd/even receive buffers        //  - 2 symmetric odd/even signaling buffers
```

然后传输消息的Size为:

```
        // Message sizes        EP_HOST_ASSERT(num_scales * sizeof(float) <= hidden);        size_t num_bytes_per_dispatch_msg = hidden + num_scales * sizeof(float) + sizeof(int4);        size_t num_bytes_per_combine_msg = sizeof(int4) + hidden * sizeof(nv_bfloat16);
```

`send`,`recv`,`signaling`缓冲区的定义如下

```
        // Send buffer        size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;        size_t combine_send_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;        size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);        EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);        total_bytes += send_buffer_bytes * 2;        // Symmetric receive buffers        // TODO: optimize memory usages        size_t dispatch_recv_data_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;        size_t combine_recv_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;        size_t recv_buffer_bytes = std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);        EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);        total_bytes += recv_buffer_bytes * 2;        // Symmetric signaling buffers        size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);        size_t dispatch_recv_atomic_token_counter_bytes = num_local_experts * sizeof(int);        size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;        size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes + dispatch_recv_atomic_token_counter_bytes,                                                 combine_recv_flag_buffer_bytes);        total_bytes += signaling_buffer_bytes * 2;
```

#### 3.2.2 低延迟Dispatch

调用的函数如下, 注释中有一个非常关键的信息`compatible with CUDA graph` 这也是为什么使用IBGDA的原因, 传统的RDMA需要以消息粒度发送, 并使用WRITE_WITH_IMM消息,将立即数作为CQE传递给CPU, CPU再进一步launch kernel, 这样的方式打断了CUDA graph使得延迟增大. 另一个关键点是采用了`double-batch overlapping`的方式, 然后可以通过第三天开源的DeepGEMM进行MoE的专家矩阵计算.

```
def low_latency_dispatch(hidden_states: torch.Tensor, topk_idx: torch.Tensor, num_max_dispatch_tokens_per_rank: int, num_experts: int):    global _buffer    # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)    recv_hidden_states, recv_expert_count, handle, event, hook = \        _buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,                                     async_finish=False, return_recv_hook=True)    # NOTES: the actual tensor will not be received only if you call `hook()`,    # it is useful for double-batch overlapping, but **without any SM occupation**    # If you don't want to overlap, please set `return_recv_hook=False`    # Later, you can use our GEMM library to do the computation with this specific format    return recv_hidden_states, recv_expert_count, handle, event, hook
```

`num_max_dispatch_tokens_per_rank`为一次batch的`num_tokens=128`. 然后将调用`csrc\deep_ep.cpp`中的`Buffer::low_latency_dispatch`函数. 该函数首先分配`LowLatencyLayout`, 然后等待前面的task完成后, 分配packed tensor

```
    // Allocate packed tensors    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden}, x.options().dtype(torch::kFloat8_e4m3fn));    auto packed_recv_src_info = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));    auto packed_recv_count = torch::from_blob(buffer.dispatch_rdma_atomic_token_counter,                                              {num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));
```

另外还分配了FP8细粒度量化时使用的scale, 并且考虑到TMA加载的优化需要保证num_token能整除4

```
    // Allocate column-majored scales    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");    auto packed_recv_x_scales = torch::empty({num_local_experts, num_scales, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kFloat32).device(torch::kCUDA));    packed_recv_x_scales = torch::transpose(packed_recv_x_scales, 1, 2);
```

Fine-graine quantization所使用的Scale-Factor如图所示:

![图片](assets/30431bfd2370.png)

然后就是调用`csrc\kernels\internode_ll.cu`的dispatch kernel. 它通过一个`phase`变量判断是`LOW_LATENCY_SEND_PHASE`还是`LOW_LATENCY_RECV_PHASE`.
3.2.2.1 SEND PHASE
SEND阶段的Warp分为两类, 第一类执行FP8转换和发送TopK token, 第二类为最后一个warp用于读取topk_idx并统计per-expert信息.FP8转换的算法如下, 转换后将其写入到发送Buffer中.

```
// FP8 cast#pragma unrollfor (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {    // Read and calculate local amax    auto int4_value = __ldg(x_int4 + i);    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);    float fp32_values[kNumElemsPerRead];    float amax = kFP8Margin, scale, scale_inv;    #pragma unroll    for (int j = 0; j < kNumElemsPerRead; ++ j) {        fp32_values[j] = static_cast<float>(bf16_values[j]);        amax = fmaxf(amax, fabsf(fp32_values[j]));    }    // Reduce amax and scale    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");    amax = half_warp_reduce_max(amax), scale = kFP8Amax / amax, scale_inv = amax * kFP8AmaxInv;    if (lane_id == 0or lane_id == 16)        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;    // Cast into send buffer    int2 int2_value;    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);    #pragma unroll    for (int j = 0; j < kNumElemsPerRead; j += 2) {        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);    }    rdma_x_int2[i] = int2_value;}            
```

然后通过调用IBGDA进行发送到指定的slot, 这样可以实现在AdaptiveRouting开启时发送不用保序, 发送完成后更新原子计数器

```
// Issue IBGDA sendsif (dst_expert_idx >= 0) {    int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;    slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);    constauto dst_rank = dst_expert_idx / num_local_experts;    constauto dst_expert_local_idx = dst_expert_idx % num_local_experts;    constauto src_ptr = reinterpret_cast<uint64_t>(rdma_x_int2);    constauto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +                         dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +                         rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +                         slot_idx * num_bytes_per_msg;    if (dst_rank != rank) {        nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);    } else {        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls        constauto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);        constauto* dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);    }    // Increase counter after finishing    __syncwarp();    lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;}
```

最后一个Warp用于分配任务给不同的SM并处理Expert任务分发和同步.
3.2.2.2  RECV PHASE
接收采用两个sub warp交替处理, 首先根据`responsible_expert_idx` 计算`src_rank`和本地专家索引`local_expert_idx`通过`nvshmemi_ibgda_poll_recv(src_rank, local_expert_idx)`进行polling.然后拷贝Token和相应的scale数据和source info.

#### 3.2.3 低延迟Combine

同样也分为SEND和RECV两个PHASE, 调用方式如下

```
def low_latency_combine(hidden_states: torch.Tensor,                        topk_idx: torch.Tensor, topk_weights: torch.Tensor, handle: Tuple):    global _buffer    # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)    combined_hidden_states, event_overlap, hook = \        _buffer.low_latency_combine(hidden_states, topk_idx, topk_weights, handle,                                    async_finish=False, return_recv_hook=True)    # NOTES: the same behavior as described in the dispatch kernel    return combined_hidden_states, event_overlap, hook
```
3.2.3.1 SEND PHASE
有一个注释的小错误, 实际上是执行了BF16的IBGDA发送,而没有FP8 Cast:)

```
    // FP8 cast and issue IBGDA sends    if (responsible_expert_idx < num_experts) {        constauto dst_rank = responsible_expert_idx / num_local_experts;        constauto local_expert_idx = responsible_expert_idx % num_local_experts;        constauto global_expert_idx = rank * num_local_experts + local_expert_idx;        constauto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);        constauto local_x = reinterpret_cast<const int4*>(x) +                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;        constauto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;        constauto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_x) +                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;        // Unpack layout        int offset, num_tokens_to_send;        unpack2(layout, num_tokens_to_send, offset);        // Issue IBGDA send        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += kNumWarpsPerGroup) {            constauto x_int4 = local_x + token_idx * hidden_bf16_int4;            constauto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);            constauto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row + 4);            // Copy directly to local rank, or copy to buffer and issue RDMA            auto src_idx = __ldg(local_src_info + token_idx);            constauto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);            constauto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot + sizeof(int4);            if (dst_rank == rank) {                constauto dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);            } else {                constauto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);                nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, hidden * sizeof(nv_bfloat16), dst_rank, local_expert_idx, lane_id, token_idx - offset);            }        }        // Put finishing flag        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(kNumWarpsPerGroup * 32));        if (sub_warp_id == 1and lane_id == 0) {            while (ld_acquire_global(atomic_clean_flag) == 0);            if (dst_rank != rank) {                nvshmemi_ibgda_rma_p(rdma_recv_flag + global_expert_idx, 1, dst_rank, local_expert_idx, 0);            } else {                st_na_release(rdma_recv_flag + global_expert_idx, 1);            }            atomic_add_release_global(atomic_clean_flag, -1);        }        __syncwarp();    }
```
3.2.3.2 RECV PHASE
也是通过Polling的方式获取` nvshmemi_ibgda_poll_recv(src_rank, src_expert_idx);`然后执行reduce操作.

### 3.3 其它细节

#### 3.3.1 文档外行为的PTX指令

在`utils.cuh`中描述如下

```
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"#else#define LD_NC_FUNC "ld.volatile.global"#endif// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS,// which does not have cache allocation, and `CONSTANT` memory does not have coherence control,// so we have to control them by queue semanticstemplate <typenamedtype_t>__device__  __forceinline__ dtype_t ld_nc_global(const dtype_t *ptr) {    auto ret = ld_nc_global(reinterpret_cast<consttypename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));    return *reinterpret_cast<dtype_t*>(&ret);}
```

ld.global.nc指令用于从Global Memory中加载数据到寄存器中,不使用一致性缓存, 这样可以降低对L1 Cache的使用提高缓存命中率从而提高性能. 其实PTX文档中定义了这样的行为

![图片](assets/d2bfff7b2fde.png)

很聪明的做法, 只有性能优化到极致的团队才会注意到这些细节, 既在L2Cache做了prefetch,同时又避免了L1Cache的占用,直接将数据存入寄存器文件.

#### 3.3.2 Memory Order

在`utils.cuh`中还定义了大量的PTX操作, 特别是LD/ST都采用了`acquire/relaxed`并且在程序中大量使用, 进一步提高的处理效率.

#### 3.3.3 nvshmem库的修改

在`third_party/nvshmeme.patch`中对nvshmem库进行了一些修改, 主要有几点

分离了SEND和RECV的CQ,并调整了相应的结构体

增加了接收队列的消费者索引

关闭了polling的超时检查

改变了QP创建的顺序, 确保不会创建到自身的连接

## 4. RoCE上运行DeepEP的挑战

### 4.1 DeepSeek用到的IB网络技术

DeepSeek的组网结构从论文《Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning 》[1]可以了解到采用标准的FatTree结构的IB网络

![图片](assets/f4a0f79cdc82.png)

虽然论文尾部展望未来的时候提到了多平面的架构

![图片](assets/bdeffc6c035a.png)

但是这次的一些信息分析, 在训练和Prefill阶段, 基于nvshmem库使用的是RDMA_WRITE消息实习的内存语义, 并没有采用WRITE_WITH_IMM触发CQE进行处理. 因此当打开adpative-routing将会导致接收端乱序, 从而可能导致数据损坏和死锁的情况. 不过README中也提了一句`(support may be added soon)`. 而在Decoding阶段使用的低延迟kernel上是支持adaptive-routing的, 并且IBGDA直接对目标GPU发送,而避免了Normal Kernel通过NVLINK转发.

基于这两个信息判断, **DeepSeek还是应该采用的FatTree组网的架构**, 在训练和Prefill上基于OpenSM配置了一些静态的路由规则避免路由冲突导致的性能下降,而在Decoding的交换机上开启了Adaptive Routing.

### 4.2 RoCE使用DeepEP的问题

虽然论文提到一句理论上也可以兼容RoCE, 但是实际上会遇到不少问题, 这是我们值得重视的.

DeepEP is fully tested with InfiniBand networks. However, it is `theoretically compatible` with RDMA over Converged Ethernet (RoCE) as well.

关于RoCE的各种问题写了一个专题, 可以访问如下连接, 最近这十年, Mellanox在RoCE上确实有不少的问题,   后面分几点来详细分析

[《RDMA》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3398249338911260673#wechat_redirect)

#### 4.2.1 Multi-Rail and Rail-Only拓扑的问题

Meta有一篇论文《Rail-only: A Low-Cost High-Performance Network for Training LLMs with Trillion Parameters》[2]提到了一种优化

![图片](assets/afc5d1fd6c20.png)

主要问题是在RoCE网络中, 由于路由的ECMP Hash冲突的问题是非常严重的, 而对于老的Dense模型设计时, 由于并行策略只有TP/DP/PP因此从流量上分析跨越Rail的流量很少

![图片](assets/c96f77a8fa82.png)

甚至Meta还提出了一种Rail-Only的做法

![图片](assets/393e396c8ff5.png)

但是这样在Decoding阶段就要将流量从NVLINK转发, 可以看一下DeepEP的Normal Kernel就明白这样的方式将增加很大的延迟, 使得decoding的TPOT性能受到影响. 如果直接在这样的RoCE网络上使用DeepEP IBGDA P2P通信, 网络中的Hash冲突是难以控制的.

渣B其实一直在谈, 在网络建设的过程中,不要为了一些局部的优化或者规避一些问题采用非对称拓扑, 要直接面对这些问题并保留对称的拓扑.

当然有一种潜在的解法, 例如使用Spectrum-X的解决方案在以太网上开启Adaptive-Routing功能, 具体可以参考

[《谈谈英伟达的SpectrumX以太网RDMA方案》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489018&idx=1&sn=25b3df2a17d49681edc0e621049b058f&scene=21#wechat_redirect)

当然Spectrum-X方案也有一些缺陷, 有一些消息知道Nvidia也在改善它, 例如去掉Lossless等..

另一种选择是像Meta在Llama3论文中讲的那样

[《[Sigcomm论文解析] Llama 3训练RoCE网络》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247491483&idx=1&sn=13ad0defd3c71bb733de86a5b3246107&scene=21#wechat_redirect)

但是采用基于Jericho/Ramon的框式交换机, 但是很抱歉, 这样的方案还会增加额外的延迟和增加30%的网络成本, 倒不如直接上IB了.

![图片](assets/fdd9fa9cfc52.png)

而基于BRCM的交换机, 大概想了一下, 似乎可以做一个NV不官方支持的方案, 将EP创建的QP设置为不同的DSCP配置到不同的交换机队列中, 在该队列中打开DLB功能.

#### 4.2.2 incast

另一个问题是RoCE网络中的incast, 当构建EP320并行的时候, 有大量的QP在发送数据到某一个专家所在的GPU时, 所以渣B一直反复在提拥塞控制一定要考虑解决好incast的问题, 特别是在2023年设计eRDMA拥塞控制算法时,因为早在2022年渣B就判断MoE必然会成为LLM的主流选择, 因此必须要解决AlltoAll的incast问题, 因此在2023年就完成了设计和实现, 并且实现了128-to-1时, 每个QP的流量差额小于100Kbps的高精度的负载均衡.

#### 4.2.3 RC兼容

RoCE的解决方案中, 以UEC为代表, 或者AWS的SRD这样的技术, 这些技术抛弃了对Reliable Connection的兼容, 因此这些平台上适配基于IBGDA的DeepEP工作量会非常大.

#### 4.2.4 In Network Computing如何做?

DeepEP的工作中也没有使用SHARP, 连NVLINK上的NVLS也没有采用, 其实在EP并行上这些基于交换机的In-network computing技术受到context约束几乎完全无法实现.

### 4.3 真正的适合EP的RDMA over Ethernet

在两个月前写了一篇文章

[《从Mooncake分离式大模型推理架构谈谈RDMA at Scale》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492691&idx=1&sn=584daa6901215ec87af037e997f8421e&scene=21#wechat_redirect)

中间就提到了`为什么一定要兼容RDMA Verbs RC生态`,另一方面针对以太网承载RDMA如何构建多路径, 如何避免PFC构建大规模组网. 其实这些技术都已经完全做到了, 等过几个月专利对外公布的时候大家就明白了.

## 5. 关于DeepSeek-V3论文的建议

![图片](assets/300cb2826194.png)

当您看完DeepEP的代码,您就明白为什么这么痛了需要提一些建议了, 为什么要统一ScaleUP和ScaleOut语义了, 针对EP的第二轮通信上的优化方案去年大概花了半年的时间全部设计完成了, 这些内容还是留给和DeepSeek这样懂的团队讨论吧.

## 6. 关于MoE模型的演进

其实这是一个更大的话题, 当前的MoE对内存带宽的需求和网络带宽的需求是巨大的, 这也导致很多私有化的部署性能比较差, 那么是否能够在算法上做进一步的协同呢? 例如前段时间谈到的

[《谈谈DeepSeek MoE模型优化和未来演进以及字节Ultra-Sparse Memory相关的工作》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493218&idx=1&sn=f394f39a4346fd09a19008a53d0a8022&scene=21#wechat_redirect)

进一步对MoE稀疏化,降低访问内存带宽

![图片](assets/9acb82a15478.png)

一个新的时代被DeepSeek开启了, 知乎上有一篇文章《FlashMLA性能简测》[3]作者提到一段话:

最近放出来的infra的东西，都是在我认知射程之内的东西了，相比之前v3和r1使不上劲的感觉，现在倒是能很快follow。不过infra只有和顶尖的芯片和顶尖的模型配合才有价值，单独卷infra是没有意义的，所以即使在射程之内了，另外两项缺憾却难以补足。一个想法，在排除少数极强个体之外，未来能走在前沿的，一定是团队配合，这就首先排除了很多高校，因为高校博士生大多数都是单打独斗，形不成合力；其次，`能走在前沿的一定是占据了模型+infra+芯片三位一体紧密配合的优势`，这又排除了很多大厂，因为尾大不掉和部门墙天然走向了另一条道路。

是的, 能走在前沿的一定是占据了模型+infra+芯片三位一体紧密配合的优势, 估计像渣B这样的全栈工程师真的有活干了, 大家也一起加油~

另外善意的提醒一下, 基础设施建设不要简单的以DeepSeek-R1的模型推理需求来分析, 更多的是需要有更长远的眼光去判断模型的变化, 新的模型会进一步解决基础设施的各种痛点, 后面几天稍微空一点再继续分析一下FlashMLA和DeepGEMM吧~~

参考资料

[1] 
Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning: *https://arxiv.org/abs/2408.14158*
[2] 
https://arxiv.org/pdf/2307.12169: *https://arxiv.org/pdf/2307.12169*
[3] 
FlashMLA性能简测: *https://zhuanlan.zhihu.com/p/26113545571*