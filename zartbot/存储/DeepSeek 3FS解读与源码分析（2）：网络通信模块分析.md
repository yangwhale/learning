# DeepSeek 3FS解读与源码分析（2）：网络通信模块分析

> 作者: zartbot  
> 日期: 2025年3月2日 12:18  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493353&idx=1&sn=80ad7afcd4360fd389d841ad05222238&chksm=f995f62bcee27f3dc8847baacef3a3fa7c00db3a62f7fada7fe714e811d5c5d9dbbaf8a9e313#rd

---

在 2025 年 2 月 28 日，DeepSeek 重磅开源了Fire-Flyer 文件系统 3FS。本文结合 DeepSeek 发表的技术报告和开源的 3FS 代码分析 3FS 中的网络通信模块。

3FS 集群由 180 个存储节点组成，每个存储节点配备 2×200 Gbps InfiniBand 网卡和 16 个 14TiB NVMe SSD。180 节点聚合读取吞吐量达到约 6.6 TiB/s。

3FS 主要有 4 大核心：Cluster Manager，Metadata Service，Storage Service 和 Client，各个模块之间都使用 RDMA** 网络连接。

通过研究代码，我们认为 3FS 网络通信模块设计优势包括并不限于：

• 设计细节层面追求极致性能，包括 I/O 链路全栈引入 folly 协程将 I/O bounded 操作异步化，数据处理使用 C++20 新特性等

• 设计上考虑了通用性，例如 Client 和 Server 独立成型，不对某种应用场景有强依赖和强定制化，不依附于特定的 RPC 框架，不依赖元数据中心等

• 在工程实现上充分挖掘了多网卡并行的能力，并且在流控实现，自研消息编解码 Serde 服务等细节上有工业级打磨

### 通信核心类分析

3FS 网络通信相关的功能实现代码在 src/common/net 目录下，其他使用到通信功能组件的模块分布在其他目录，例如 src/core/storage 等。

3FS 网络通信主要支持 RDMA 和 TCP 传输。本文主要关注在 RDMA 链路设计上。3FS 使用 IB，代码逻辑上也支持 RoCEv2。

前面提到 RDMA 会在 3FS 的各个模块中运行。其核心设计思想是通信模块抽象出来了一个 Listener，它配合 IOWorker，EventLoop，IBSocket 类等共同组成一个强大的 RDMA网络子系统。这个子系统能部署在各个功能子模块中，这点和 BeeGFS 非常类似。

#### IBSocket 类

IBSocket 主要负责 Socket 相关逻辑处理。它包含若干内部类（结构）定义和一些和 RDMA 相关的成员。IBSocket 建立过程在下文 Listener 和 IOWorker 流程中详细描述。

![图片](assets/3b1dc707c609.png)

其中较为重要的几个方法是：rdmaRead 和 rdmaWrite。

![图片](assets/0521fefc30c2.png)

其中 rdmaWrite() / rdmaRead() 是 rdma() 的封装，组建一个 RDMAReqBatch 并 post 出去。这里会通过 IBSocket 的 协程方法 rdmaBatch() 将读写请求发送出去。在下文 RDMA I/O 路径中会做进一步分析。

![图片](assets/65b49a3c0519.png)

#### RDMA 设备管理

IBDevice 类负责 RDMA 设备管理。打开 RDMA 设备的核心工作包括：查询设备，打开设备，分配 Protection Domain，查询设备属性等。

#### RDMA 连接建立阶段

![图片](assets/8afd327af9a8.png)
客户端主动建连的过程在 IBSocket::connect 中，调用 qpCreate，然后通过 verbs 创建 qpair。之后通过协程调用 IBConnect<>::connect() 建立连接。

![图片](assets/fbd3eb987c3e.png)

之后客户端等待服务器回应qpair信息，最后修改 qpair 状态到 READY。
服务器 IBSocket::accept 在收到一个连接请求，会通过 verbs 创建 qpair。之后 init qpair，并在 qpReadyToRecv 中修改 qpair 的状态到 READY。

![图片](assets/8d65f3078266.png)

这里有一个优化细节，在 IBSocket::qpCreate 中设置属性 attr.sq_sig_all = 0。它能避免对每个 WR 都产生一个 CQE** 带来的性能损失，仅仅在设置了 IBV_SEND_SIGNAL 的 WR 上才会产生一个 CQE。这样的优化可以减少 WC 的个数，真正发挥 batch 的优势。

![图片](assets/3e98afac9648.png)

#### Listener 类

Listener 类中 setup() 会 loop 所有网卡，将它们加入到 addressList_ 中，并且对每个 address ，从链接池中拿一个EventBase线程，并且创建一个 socket。这个 socket 使用 folly 的 blockingWait 创建。最后将这个 socket 加入到 Listener 的 serverSockets_ 这个vector中。至此，每个网卡建立了一个 listening socket。

![图片](assets/4486686623de.png)

之后 Listener::acceptRDMA 中每得到一个连接，并将IBSocket加入到 IOWorker 中。

#### IOWorker 类

IOWorker 负责处理所有 I/O tasks。IOWorker 整体结构如下：

![图片](assets/7241a1f4bdf1.png)

每个 ServiceGroup*有一个 Listener 和一个 IOWorker。每个 Listener又有一个 **IOWorker**。在创建 ServiceGroup 实例的时候会把 Listener 也一起创建出来。

之前提到每当 Listener 收到一个 RDMA 连接请求，往 IOWorker 加入一个 **IBSocket**。IOWorker 会根据请求类型创建一个 Transport，同时在 EventLoopPool 中创建一个 transport 的 event loop，并且将这个 transport 本身也加入到 transport pool中。

![图片](assets/b67bfefe7242.png)

#### EventLoop 类

EventLoop 类提供了一个 main loop，当 fd 上的 I/O ready 的时候用来通知 EventHandler callback对象。

EventLoop 类中包含一个 EventHandlerWrapper 的 list，之所以用 Wrapper 的原因代码中提到因为 folly 库有内存泄漏。这不是我们关心的重点。

EventLoop::add 中为感兴趣的 events 向 epoll 监听注册 handler。这里的一个优雅手法是如果注册不成功就从 wrapperList_ 回滚。

![图片](assets/606fbc870410.png)

EventLoop::loop() 就是一个 while(1) 循环，等到 epoll_wait 中 poll 到消息，则调用 handle_events 逻辑。个人认为这里对 wrapper->handler 加锁是不得已而为之。RDMA的链路上如果能做到完全无锁能更好地提升性能。

![图片](assets/7fefaf82bb27.png)

### 网络资源管理模块分析

RDMA 的内存需要提前分配并且注册。3FS 的 RDMA 内存管理模式和 Mooncake TransferEngine 等其他开源实现稍有不同。

3FS 的 RDMA buffer 资源不需要中心化管理，只需要在读写的时候做协商。这样让模块更具灵活性，不依赖第三方中心元数据。

Mooncake TransferEngine 为大模型推理使用，它的内存分配在读写之前需要上层应用（例如vllm）将 source 和 target 的内存提前注册好，通知元数据中心 MetaService。后续在读写的时候就直接能够通过元数据查询到 src 和 target 的RDMA地址，避免使用双边的 verbs 通信来进行协商。

#### 内存池化 RDMABufPool

3FS 中设计了 RDMABufPool 来管理 RDMA 内存。Client 和 Server 都会使用这个 RDMABufPool。

RDMABufPool 分配逻辑返回一个 RDMABuf，并封装在一个 CoTask 中以便异步执行。这里使用非阻塞的 Buf 分配，也能进一步提升性能。

![图片](assets/944ccb31795a.png)

#### 传输池化 TransportPool

每个 IOWorker 中有一个 TransportPool，它会对 address 做 shard 分配以提升性能。

![图片](assets/49926537439a.png)

### RDMA I/O路径流程分析

RDMA I/O 路径阶段是整个通信系统的核心部分。它分为几个阶段：

#### 数据发送准备阶段

核心的数据传输模块在 IBSocket 中封装。它使用 folly 协程来工作。

IBSocket::rdmaBatch() 是消息发送准备的核心逻辑。这个函数实现批量 RDMA 读/写操作，通过协程 (CoTryTask) 进行 异步操作，确保高效处理 RDMA 请求，同时记录延迟信息。

![图片](assets/dc1011a8d56e.png)

rdmaBatch 返回值 CoTryTask 是一个 folly coroutine task。

![图片](assets/e93c2d3ab33d.png)

在 rdmaBatch 中 wrsPerPost 记录了连接配置中每个 post 能发送多少个 rdma_wr。

读写链路上使用单边操作 RDMA_WRITE 和 RDMA_READ 保证高效率。

rdmaBatch 过程中使用了C++20 std::span 来提供一个轻量级视图容器以非拥有的方式访问一段连续存储的元素，而不需要拷贝RDMAReq list数据。

在处理 RDMA 发送请求的时候，使用了 vector 来装载所有 RDMAPostCtx。这个 RDMAPostCtx 存储了若干 RDMAReq 和 RDMABuf。

![图片](assets/48e396c30be4.png)

如果只有一个 batch，则使用 co_await rdmaPost(posts[0]) 进行RDMA传输。如果有多个 batch，则创建多个 rdmaPost任务。并放入到 task vector 中，在后续以 collectAllRange 的方式并行执行 rdmaPost()。

![图片](assets/3ed33456104d.png)

#### 数据接收阶段

我们知道对于单边请求 READ/WRITE 只需要在发送端使用 ibv_poll_cq 轮询 complete queue 中的结果。这部分逻辑封装在 IBSocket::cqPoll() 中。同时在 IBSocket::drain() 中也会在关闭 socket 的时候 poll 这个 IBSocket 上的 cq。

![图片](assets/7f2e50c2b334.png)

处理 event 响应的逻辑和 EventHandler** 类有关：
Transport，IBDevice(包括两个内部类)，IBSocket，IBSocketManager 都继承自这个 EventLoop::EventHandler 类。EventLoop::loop() 中会负责在 handler->handleEvent() 中处理 event 事件。

![图片](assets/661126220311.png)

#### 数据发送接收 verbs 封装

数据发送接收阶段其实包括多个类型数据的处理。它利用了一个 WRId 来描述 WorkReq 的类型。各个 WR 定义如下：

• IBSocket::check()，它通过 IBV_WR_RDMA_WRITE 发送一个空 RDMA 消息到对端检查连接是否正常。

• IBSocket::closeGracefully()，它通过发送 IBV_WR_RDMA_WRITE_WITH_IMM 到对端做优雅关闭一个 RDMA 连接。立即数带上的是 ImmData::close() 标记。

• IBSocket::postSend()，它也是通过发送一个带立即数的写请求 IBV_WR_RDMA_WRITE_WITH_IMM 到对端，通知对方消息已经发送完毕。立即数带上的是 ImmData::ack(buf_ack_batch)。

• IBSocket::postRecv(idx)，作用是处理双边请求 SEND/RECV 的过程中接受方的 RECV 逻辑，或者处理 WRITE_WITH_IMM 在接受方的 RECV CQE。

• IBSocket::rdmaPostWR()，作用通过发送 IBV_WR_RDMA_READ/WRITE 的单边请求来发送数据，它可以是一个batch操作，其中包括若干个RDMARequest，并带上 rkey 给对端网络。
在 RDMA batch 过程中，会在最后一个 WR 塞一个 IBV_SEND_SIGNALED flag。这样的好处是减少 CQE 的产生和对 CQE 的读取次数。

![图片](assets/acc795e11936.png)

RDMAPostCtx 中记录一组 reqs，每个 req 和一个 RDMABuf 相对应。如果是 WRITE 请求，数据从 localBuf[idx] 中取，如果是 READ 请求，数据读到 localBufs[idx] 中来。

![图片](assets/f5357dd7defc.png)

3FS 在 cqPoll 收到错误的 wc，或者在 post_send 返回错误的时候只标记 IBSocket 状态为 Error，没有错误重试逻辑。感觉这里可以做进一步优化提升鲁棒性。

#### 写数据I/O流程

写流程使用 CRAQ 链式复制。从代码阅读上看，链式写的过程中第一跳 Client 到 ChunkServer 使用的是 **ReliableUpdate** 模块，ChunkServer(TargetA) 到 ChunkServer(TargetB) 之间使用的是 **ReliableForwarding** 模块。后者在内部也会调用 ReliableUpdate 模块。RDMA 写并没有使用 write batch，背后的考量可能是为了尽快 forward 到链式写连路上。

ReliableUpdate 是从 src 到 target 发送一个 update 请求，之后 target 向 src 发起一个反向的 RDMA READ。在这个过程中一个 target 同时可能会保存两个 chunk版本：committed 版本和 pending 版本。其中 committed 版本是之前的旧版本，pending 版本是指 committed版本 叠加上 update 发过来的新修改的数据版本。假如自己是最后一跳（如下图Target C）或者等到下一跳返回了 ack 才将 pending 版本覆盖 committed 版本，整体相当于是一个 in-place 的。Read-Modify-Write操作。

[感想] 这里感觉修改 Chunk 级别的粒度有点大，在中间状态下会有 2 倍的空间放大。假如是只改动 Chunk 中的一小部分数据，也需要将整个 committed version Chunk 数据从磁盘读出来，再叠加 update，写一个pending version 的全量 Chunk 上面去。链式写速度受磁盘写盘速度限制，如果能在本地读写 NVMe 的时候将 libaio 替换成用户态存储栈 SPDK，能进一步大幅度提升 I/O 性能，甚至可以考虑引入 NVMeoF，将数据直接通过 RDMA 写到远端 NVMe 上。个人大胆猜测没有这样设计可能有以下几个方面原因：

1. 目前 3FS 主要为训练 ckpt 场景服务着重提升写吞吐，对延迟没有太大要求。但是 3FS 同时也为推理的 KVCache 提供 SSD offload，对延迟也是有一定需要的。

2. NVMeoF 的设计更适合于 1-to-N 的星形写，而对类似 HDFS 的 chain-replicate 或者 CRAQ 这种链式复制的架构不是很友好。而虽然 SPDK NVMe 层采用 polling mode driver 模式能带来高性能，但是对容器部署不友好带来的隐形成本，以及 RTC** 模式与 folly 协程的协同带来的性能抵消可能是另一个考虑的因素。

![图片](assets/f02e4af22756.png)

从实现细节看，StorageOperator::write 中会通过 ReliableUpdate的update 方法做数据传输。

![图片](assets/6137e3af1275.png)

![图片](assets/c530bf2abeea.png)

ReliableForwarding 中从磁盘上读数据是在 AioReadWorker 中提交一个任务。

![图片](assets/1ac3f1d67a82.png)

在 doForward 中向 AIOReadWorker 提交一个 batch 请求。

![图片](assets/8b015db90056.png)

#### 读数据I/O流程
在读链路上为了速度更快，可以做 batch 读。整个读入口是在 StorageOperator::batchRead，目前理解 3FS 的客户端读本质上也同样是从服务器端发起 RDMA write 请求将数据写到 Client端，所以在读操作中会生成一个 writeBatch。把 batchRead 中的 batch 请求添加到**BatchReadJob**中，然后把它发给**AioReadWorker**。

![图片](assets/1cf9e162b947.png)

![图片](assets/ad67887a6c17.png)

![图片](assets/6d13957d5bb9.png)

Storage 中的 AioReadWorker读取的目的 buffer 是一段 RDMA buffer。

对其中的 job.readIO() 会读取 SERDE 服务编解码之后的一段 RDMA buffer，再加上 offset+length+key 编码组合。这部分逻辑相当于对数据和位置信息做了 RPC 编码。

![图片](assets/3f61c0fe5b60.png)

### Folly 协程库在 3FS 中使用浅析

从源代码中可以看到，3FS 中基本上全栈都使用了 folly 协程库。我们通过一个 RDMA 请求发送过程中 RDMAPostCtx 的协程交互流程来分析一下 folly coroutine 的作用。

在 IBSocket::rdmaBatch 中会将所有的 RDMA 请求按照 max_rdma_wr_per_post 来分 batch，每个 batch 组织一个 RDMAPostCtx。

![图片](assets/3a016e84d071.png)

RDMAPostCtx 中有一个 folly::coro::Baton 同步原语用来同步协程的切换和切回。当 rdmaPostWR 发送请求之后，这个协程会将自己挂起，等待处理 complete queue 的协程设置 baton.post 后的唤醒。

![图片](assets/15cdd984b39e.png)

全链路使用无栈协程的手法对开发工程师对整体链路的把控要求非常高，一旦有任何一环涉及到 I/O bound 的操作就需要考虑整条链路使用协程，并且对生产级代码的调试和错误排查提出了非常高的要求。3FS 为我们展示了一套端到端高性能异步化协程处理的工业级产品。

### 总结

本文基于 3FS 网络通信模块的机制原理和源代码进行了初步分析。整体来说 3FS 的实现涉及到非常多的细节优化，通信模块和存储模块的设计环环相扣，交相呼应，展示了极高水平的存储架构设计。

3FS 的开源为蚂蚁集团甚至整个 AI 业界设计大模型高性能存储提供了优秀范本，值得我们反复推敲，深入学习。