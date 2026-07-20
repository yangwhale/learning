# NCCL Gin & Symmetric Memory

> 作者: zartbot  
> 日期: 2026年5月1日 06:29  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498168&idx=1&sn=adfe6ba01ff8cdbe20cdf5aeb655d0cb&chksm=f995e97acee2606c472f860415f481d0b3762a0c656a1a6e390d27943de51ee27e42c49a2a51#rd

---

### TL;DR

这是分析DeepSeekv4的番外篇, 由于DeepEPv2调用了NCCL Gin backend, 同时Symmetric memory对于实现计算和通信融合也是一个非常重要的概念, 因此我们先来展开介绍一下这方面的内容, 我们首先参考论文《GPU-Initiated Networking for NCCL》[1]来了解为什么有NCCL Gin以及Symmetric Memory相关的概念. 总体来说, Symmetric Memory使得并行计算的算子对于分布式系统的peer内存访问抽象到一个flatten VA空间, 算法上会更容易处理地址. 但是具体的通信时, 还是需要不同的语义, 针对NVLink/PCIe P2P直接的内存语义即可, 而针对RDMA ScaleOut网络, 则需要使用Gin(GPU-Initiated Networking)在设备侧发起基于消息语义的网络通信.

本文目录如下:

- 1. Symmetric Memory
  - 1.1 什么是Symmetric Memory
  - 1.2 Symmetric Memory实现
    - 1.2.1 统一抽象
      - 1.2.1.1 内存管理
      - 1.2.1.2 通信窗口注册
      - 1.2.1.3 Communicator
      - 1.2.1.4 对端通知机制
    - 1.2.2 LSA
      - 1.2.2.1 LSA 数据交互
      - 1.2.2.2 LSA 同步机制
      - 1.2.2.3 低延迟AlltoAll
    - 1.2.3 Gin数据交互
      - 1.2.3.1 连接建立
      - 1.2.3.2 窗口注册
      - 1.2.3.3 Gin 通信流程
      - 1.2.3.4 Gin 数据同步
- 2. Symmetric Memory完整示例
  - 2.1 initial
  - 2.2 分配对称内存与注册窗口
  - 2.3 创建 Device Communicator
  - 2.4 启动 Kernel 与验证
  - 2.5 资源释放

## 1. Symmetric Memory

### 1.1 什么是Symmetric Memory

传统的GPU通信遵循主机发起模型, CPU负责编排所有通信操作. 无论是 NCCL 还是 MPI 都基于消息传递(Message Passing)的松耦合的跨节点通信模式, 这种方式需要显式的`Host-Device`同步, 并且每次通信都要调用单独的Kernel启动, 但是对需要计算与通信紧密集成的应用(例如EP并行, JAX/Triton Kernel中编译器生成的通讯)需要使用设备侧直接发起的通信来消除由基于CUDA的主机侧同步带来的CPU协调开销. NVSHMEM库已经成功展示了GPUDirect Async Kernel-Initiated(GDA-KI)能力的可行性和性能影响, 提供了设备原语来实现AI工作负载的通信与计算融合.

另一方面的变化是伴随着 NVL72 这样的超节点出现, 传统的NCCL通信模式在NVLink上延迟和开销都很大, 并且伴随着DeepEP的出现和发展, 通信库的生态上也出现了一些多样化的解决方案. 维护多个通信库也是一件比较麻烦的事情, 然后NCCL在v2.28开始支持了Device API,  让GPU线程可以在 Kernel 里直接发起网络通信, 也就是GPU Initial Networking(Gin).

当然这件事情还有另外两个推动的因素, DeepSeek曾经在论文中阐述 ScaleUP 和 ScaleOut 不同的语义带来的复杂性, 期望能够尽量统一. 还有就是Pytorch的视角, 希望将大量的GPU集群, 统一成一个拥有海量内存和计算能力的巨形GPU来编程. 由于DeepEP所使用的NVSHMEM,使得我们把目光投向了一个在传统HPC领域常见的基于OpenSHMEM实现PGAS(partitioned global address space). 大致意思是在一个分布式系统中, 各个执行单元能够共同访问一块global Address Space, 同时这块global address又被partion成多个块分给每个执行单元.

![图片](assets/da69f879f3e3.png)

### 1.2 Symmetric Memory实现

下面我们来展开介绍一下这种编程模型, 第一个问题什么是Symmetric Memory, 为什么是对称的? 在这种编程模型下, 每个GPU都在相同的虚拟地址空间内分配内存, 然后任何GPU可以使用该虚拟地址空间, 根据其它GPU的Rank进行offset计算访问其他GPU分配的地址空间. 这样每个参与其中的进程都被映射成一个**逻辑上统一但物理上分区**的全局虚拟地址空间, 并且它具有**对称性**, 简单来说:

在所有参与的进程中, Symmetric Memory的虚拟地址空间大小和布局是一致的, 为开发者提供了相对统一的编程视角.

在语法上, 对于任意进程 $P_x$ 而言, 访问其本地显存或者另一个远程进程 $P_y$ 的显存地址, 无论那个进程 $P_y$ 是通过 ScaleUP NVLink连接还是通过ScaleOut RDMA连接, 在语法上能够保证完全相同, 屏蔽底层的复杂性.

在具体的底层实现来看, 它还是分为了两类:

**LSA (Load Store Access)** 用于同节点内其它Peer的直接内存语义访问.

**GIN (GPU-Initiated Networking)** 用于不同节点之间的远端Peer访问.

#### 1.2.1 统一抽象
1.2.1.1 内存管理
我们首先来看内存管理, 在传统的CUDA编程中我们通常使用`cudaMalloc`, `cudaFree`来分配内存, 为了支持Symmetric Memory 我们需要使用`cuMem`系列API来实现, 通过它将物理内存分配和虚拟地址空间指定解耦. 在NCCL对它进行了封装, 可以参考`src/allocator.cc`中的`ncclMemAlloc`函数.

```cpp
  void *d_sendbuff;
  void *d_recvbuff;
  NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
  NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));
```

`ncclMemAlloc`具体执行步骤如下:

使用`ncclCuMemEnable`检查打开GPU VMM.

通过`cuMemCreate(&handle, handleSize, &memprop, 0)`分配物理内存

使用`cuMemAddressReserve((CUdeviceptr*)ptr, handleSize, ...)` 预留虚拟地址空间

使用`cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0)` 建立虚拟地址到物理地址的映射.

最后调用`cuMemSetAccess((CUdeviceptr)*ptr, handleSize, &accessDesc, 1)`允许读写访问

需要注意的是, 对于NVLink互连的场景, 只有这样产生的物理段才能被 `cuMemExportToShareableHandle` 交换给其它 rank, 再在对端 `cuMemImportFromShareableHandle` + `cuMemMap` 形成跨 rank 同地址的"flat VA". 而用 `cudaMalloc` 出来的 buffer 没有可导出的 memHandle. 跨节点地址分配和VA映射如下图所示, 图片来自蚂蚁的Amem[2]
![图片](assets/4fdebaa43124.png)

具体来看, 在后续的`ncclDevrInitOnce`中进行对称内存子系统的一次性初始化:

```cpp
if (devr->bigSize != 0) return ncclSuccess;       // 幂等

lsaSize = computeLsaSize(comm);                    // 基于 rankToNode 用 gcd 计算
devr->lsaSize     = lsaSize;
devr->lsaSelf     = comm->rank % lsaSize;
devr->lsaRankList = malloc(lsaSize * sizeof(int)); // lsaRankList[i] = rank + (i - lsaSelf)
devr->nLsaTeams   = comm->nRanks / lsaSize;

// 1) 取 cuMem 的推荐 granularity(通常 2MB / GPU 独立分配单元)
memProp.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
cuMemGetAllocationGranularity(&devr->granularity, &memProp,
                              CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

// 2) 确定 bigSize: NCCL_WIN_STRIDE, 默认取所有 rank GPU totalGlobalMem 的最大值
devr->bigSize = ncclParamWinStride();
if (-devr->bigSize <= 1) {
  devr->bigSize = 1;
  for (r=0; r<nRanks; ++r)
    devr->bigSize = max(devr->bigSize, peerInfo[r].totalGlobalMem);
}
devr->bigSize = alignUp(devr->bigSize, 1ULL<<32);  // 对齐到 4GB

ncclSpaceConstruct(&devr->bigSpace);  // VA 分配器初始化
ncclShadowPoolConstruct(&devr->shadows);
```

其中一些概念解释如下:

`bigSize`:  每个 rank 在"全局统一对称视图"中占据的虚拟地址窗口大小(例如 128GB / 4GB 对齐)

`bigSpace`: 一个 `ncclSpace` 线性整数分配器, 只分配**相对 offset**(并不分配物理 VA)

`lsaFlatBase`:把 `lsaSize × bigSize` 段**连续**虚拟地址保留下来, 每个 rank 占 `bigSize` 一段

`granularity`:`cuMem` 物理分配粒度, 所有 window 大小必须对齐到该值

`bigSpace` 此时尚未保留任何真实 VA；`lsaFlatBase` 也尚未创建. 然后通过`ncclSpaceAlloc`在Symmetric Memory VA空间`bigSpace`进行分配. 分配后用一个纯 CPU 端的 `ncclSpace` 来"记账", 输出的 `bigOffset` 再用于驱动 `cuMemMap`/`cuMemAddressReserve`.

```cpp
struct ncclSpace {
  int count;        // cuts[] 中有效元素个数
  int capacity;     // cuts[] 分配容量(动态倍增)
  int64_t* cuts;    // 严格递增的"切点"数组
};
```

具体的分配把整数线段 `[0, limit)` 看作被一串"切点"分成若干**交替的满/空段**, 满/空判定`isFull(i) = (i % 2 != count % 2)`:

若 `count` 偶数 → 偶数下标段空、奇数下标段满.

若 `count` 奇数 → 偶数下标段满、奇数下标段空.

最后一段(下标等于 `count`)必空, 这一性质使得分配器可以始终在末尾追加.

以 `bigSize = 128GB, granularity = 2MB`、前两次分配分别 4MB、8MB 为例:

初始
`count=0, cuts=[]`, 段: `[0, 128GB)` 空
alloc(4MB)
`i = 0`, `lo=0, hi=128GB`, `off=0`, `off+size=4MB ≤ hi` 命中

`i == 0` 走慢路径 `insertSegment(0, 0, 4MB)`

插入 cuts=[0, 4MB], 融合: 前导 0 被丢弃 → `cuts=[4MB]`, `count=1`

段: `[0,4MB)` 满 | `[4MB,128GB)` 空

返回 `bigOffset = 0`
alloc(8MB)
`i = count%2 = 1`, `lo=cuts[0]=4MB, hi=128GB`, `off=4MB`(已对齐 2MB)

命中, 走快路径: `cuts[i-1] = cuts[0] = 4MB + 8MB = 12MB`

`cuts=[12MB]`, `count=1`

段: `[0,12MB)` 满 | `[12MB,128GB)` 空

返回 `bigOffset = 4MB`
free(offset=0, size=4MB):
找满段 `i=0`: `lo=0, hi=12MB`

满足 `lo==offset && offset+size!=hi` → 快路径 `cuts[i-1]`

但 `i==0` 没有 `cuts[-1]`, 代码里通过 `i!=0` 条件兜底 → 走 `insertSegment(0, 0, 4MB)`

`cuts=[0,4MB,12MB]`→融合前导 0→`cuts=[4MB,12MB]`, `count=2`

段: `[0,4MB)` 空 | `[4MB,12MB)` 满 | `[12MB,128GB)` 空

称性保证
所有 rank 在注册流中按相同顺序调用本函数, `bigSpace` 是确定性演化的 → 各 rank 得到相同的 `bigOffset`.

在`symMemoryObtain` 调用时, 把一组本 rank 已有的物理分配句柄 `memHandles[numSegments]` 转换成一个**跨 LSA Team(甚至跨节点 via GIN/RMA)的对称内存对象**`ncclDevrMemory*`,具体功能如下:

在 `devr->bigSpace` 中分到一个 **`bigOffset`**(所有 rank 一致).

把 LSA Team内所有 peer 的同名物理内存挂到统一的 **flat VA**: `lsaFlatBase + r*bigSize + bigOffset`.

完成 NVLS multicast / GIN / RMA Proxy 的注册.

执行过程如下:

```cpp
symMemoryObtain
 ├── 阶段1: 去重查找 (primaryAddr + size + numSegments + handles 完全一致)
 ├── 阶段2: 构造 ncclDevrMemory 骨架 + 填 segmentSizes
 ├── 阶段3: 全局 allgather numSegments/hasSysmem → 聚合 max/global 标志 + lsaNumSegments
 ├── 阶段4: ncclSpaceAlloc 切 bigOffset
 ├── 阶段5: symMemoryMapLsaTeam 建立 lsaFlatBase VA + 映射各 peer 物理段
 ├── 阶段6: primaryAddr 回填(无 VA 调用者)
 ├── 阶段7: symBindTeamMemory 绑定 NVLS multicast
 ├── 阶段8: symMemoryRegisterGin / symMemoryRegisterRma 跨节点注册
 └── 阶段9: 挂入 devr->memHead, refCount=1, 返回
```

针对NVLink的情况会通过如下方式导出本 rank 的句柄

```cpp
for (int seg = 0; seg < numSegments; seg++) {
  symLsaMessage* msg = messages + devr->lsaSelf * maxSegments + seg;
  symMemoryExportSegmentHandle(comm, msg, mem->memHandles[seg], segmentSizes[seg]);
}
```

具体执行如下: 从 `memHandle` 反查 `CUmemAllocationProp` → 填 `msg->type`(DEVICE/HOST_NUMA).

`msg->segmentSize = segmentSize`.

**POSIX fd 模式**: 直接把 `memHandle` 本身填入(接收端通过 proxy 拿 fd).

**FABRIC 模式**: 调用 `cuMemExportToShareableHandle(&msg->fabricHandle, memHandle, ncclCuMemHandleType, 0)` 导出 fabric handle.

然后通过 LSA Team的Allgather 同步, 其它节点拿到后把每个 peer 的物理段映射到对应 VA

```cpp
for (int r = 0; r < devr->lsaSize; r++) {
  symMemoryImportAndMapSegmentsForRank(comm, r, messages, maxSegments,
                                       mem->lsaNumSegments[r],
                                       mem->memHandles, mem->bigOffset);
}
```

针对跨节点的处理, 采用symMemoryRegisterGin(GDAKI/RDMA)或 symMemoryRegisterRma(CPU proxy)构建.

最后构建的flat VA 的空间布局

```cpp
[ lsaFlatBase ]
|  rank 0 bigSize  |  rank 1 bigSize  |  ...  | rank (lsaSize-1) bigSize |
         ↑                    ↑
  bigOffset + seg0      bigOffset + seg0
  (rank 0 物理段0)      (rank 1 物理段0)
```

设备端内核访问 peer `x` 的同名内存只需简单计算: `lsaFlatBase + x * bigSize + bigOffset + 用户 offset`.
1.2.1.2 通信窗口注册
`ncclCommWindowRegister` 是 NCCL **对称内存 (symmetric memory)** 模型的用户入口, 它把一段 CUDA 虚拟内存 `(buff, size)` 注册为所有 rank 拥有"**同一逻辑视图**"的 window, 返回设备端可直接使用的 `ncclWindow_t`.

```cpp
  ncclWindow_t send_win;
  NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win,
                                   NCCL_WIN_COLL_SYMMETRIC));
```

为什么需要进行通信窗口注册, 主要有以下几点:

**对称VA统一视图**: 原始 `cudaMalloc / cuMemCreate` 给每个 rank 的 VA 是**互不关联**的, peer rank 的 `buff` 在本 rank 根本访问不到. 注册之后, NCCL 建立了 lsaFlatBase 统一 VA 布局, **所有 rank 同名 window 都映射到**`flatBase + r*bigSize + bigOffset`, 设备端内核可以用 `lsaFlatBase + peer*stride + offset` 的纯算术直接访问任意 peer.

**物理句柄跨 rank 交换**: `cuMemMap` peer 内存需要先拿到 peer 的 `CUmemGenericAllocationHandle`(POSIX fd 或 FABRIC handle). 这个 export/proxy/fd/import/map 的过程必须由 NCCL 以 bootstrap 集合操作完成, **用户不可能自己调**.

**多个后端注册聚合**: 由于通信语义不同在ScaleUP(NVLink)和ScaleOut(RDMA)上需要处理并注册, 主要包括以下几类

**LSA Team内**: `cuMemMap` + `cuMemSetAccess` → GPU 原生 P2P/NVLS 访问

**NVLS multicast**: `cuMulticastBindAddr` → `ld/st.multimem` 可用

**GIN (GDAKI/RDMA)**: `ncclGinRegister` → RDMA 远端访问(跨节点)

**RMA Proxy**: `ncclRmaProxyRegister` → 跨节点 CPU proxy 通道

**本地注册句柄**: `ncclCommRegister` → 用于 collectives 的零拷贝识别

最后生成设备端可见、由 device kernel 直接消费的 `ncclWindow_vidmem`(含 flat 基址、stride4G、mcOffset4K、GIN 句柄数组等), 这是 `ncclDevice_*` API 能访问内存的唯一通路. 我们来分析一下`ncclWindow_t`的结构, 它在 `nccl.h` 中被 typedef 为 `ncclWindow_vidmem*`.

```cpp
struct ncclWindow_vidmem {
  void* winHost;  
  char* lsaFlatBase; 
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
  uint32_t ginOffset4K;
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclSegmentWindow* ginMultiSegmentWins; // multi-segment: pointer to accommodate variable num segments
  int numSegments;
};
```

| 字段 | 含义 | 设备端用途 |
|---|---|---|
| `winHost` | host 端 `ncclDevrWindow*` | host 侧反查(调试/销毁) |
| `lsaFlatBase` | LSA Team rank 0 的 window 首地址 | `lsaFlatBase + peer*stride` = peer 的 window 基址 |
| `lsaRank / worldRank` | 本 rank | 内核知晓自身位置 |
| `stride4G` | `bigSize >> 32` | `add4G(base, peer*stride4G)` 获得 peer 基址 |
| `mcOffset4K` | `bigOffset >> 12` | 多播 VA `mcBasePtr + mcOffset4K*4K` |
| `ginOffset4K` | `memOffset >> 12` | window 在 GIN mem 内的 4KB 偏移 |
| `ginWins[4]` | 每个 GIN 连接的远端句柄 | GIN put/get 目标 |
| `ginMultiSegmentWins` | 多段 elastic buffer 的每段 `ncclSegmentWindow` | 多段 GIN 路径 |
| `numSegments` | GIN 段数 | 决定走单段还是多段代码 |

注册通过在所有rank进行allgather来交互`ncclDevrRegTask`来交互, 然后逐个 task 调 `ncclDevrWindowRegisterInGroup`进行注册, 最后使用`symWindowCreate`创建Winddow对象. 它会分配 host 侧 `ncclDevrWindow`

```cpp
struct ncclDevrWindow* win = malloc(sizeof(*win));
memset(win, 0, sizeof(*win));
win->memory         = mem;
win->size           = userSize;
win->bigOffset      = mem->bigOffset + memOffset;   // window 在 flat 中的偏移 = mem 偏移 + window 在 mem 内偏移
win->winFlags       = winFlags;
win->localRegHandle = localReg;
if (userPtr == nullptr) {
  win->userPtr = userPtr = (char*)devr->lsaFlatBase
                         + devr->lsaSelf*devr->bigSize + mem->bigOffset;
} else {
  win->userPtr = userPtr;
}
```

**注意**: `win->bigOffset` 和 `mem->bigOffset` 不同 , 同一块 `mem` 可能被注册出多个 window(如用户对同一 buffer 的不同子区间), 每个 window 有自己的 `memOffset`.

然后再分配设备侧使用的window结构体`ncclWindow_vidmem`:

```cpp
struct ncclWindow_vidmem* winDev;       // 设备端地址
struct ncclWindow_vidmem* winDevHost;   // host shadow
ncclShadowPoolAlloc(&devr->shadows, &winDev, &winDevHost, stream);
win->vidmem = winDev;

winDevHost->lsaFlatBase = (char*)devr->lsaFlatBase + win->bigOffset;  // ★ peer rank 0 的 window 基址
winDevHost->mcOffset4K  = win->bigOffset >> 12;           // NVLS 多播偏移(4KB 单位)
winDevHost->stride4G    = devr->bigSize >> 32;            // 相邻 rank 基址步长(4GB 单位)
winDevHost->lsaRank     = devr->lsaSelf;
winDevHost->worldRank   = comm->rank;
winDevHost->winHost     = (void*)win;                     // 反查 host 端 ncclDevrWindow
winDevHost->ginOffset4K = memOffset >> 12;                // window 在 mem 内的 GIN 偏移
winDevHost->numSegments = mem->numGinSegments;
for (i=0; i<NCCL_GIN_MAX_CONNECTIONS; i++)
  winDevHost->ginWins[i] = mem->ginDevWins[i];            // GIN 设备句柄数组

allocAndPopulateSegmentWindows(...);                       // 多段 GIN 时填 ginMultiSegmentWins
winDevHost->ginMultiSegmentWins = segmentWindowsDev;

cudaMemcpyAsync(winDev, winDevHost, sizeof(ncclWindow_vidmem), H2D, stream);  // ★ 同步到设备
```
1.2.1.3 Communicator
具体的通信, 通过定义`ncclSymkDevComm`来统一抽象.

```cpp
struct ncclSymkDevComm {
  struct ncclDevComm devComm;          // 基础设备通信器(全局信息)
  struct ncclLLA2AHandle lsaLLA2A;     // LSA: 节点内低延迟 A2A 句柄
  struct ncclGinOutboxHandle ginOutbox;// GIN: 跨节点发送 mailbox
  ncclGinCounter_t ginCounterPerBlock; // GIN: 每 block 的完成计数器
  struct ncclGinInboxA2AHandle ginInboxRail; // GIN: 跨节点接收 mailbox(沿 rail)
  struct ncclGinSyncHandle ginSyncHandle;    // GIN: 跨节点 signal 集合
};
```

`ncclSymkDevComm` 把 LSA 和 GIN 的 Handler **并列存放**,它们共享同一个嵌入的 `ncclDevComm`(即同一套对称内存空间、同一组资源 ID 分配表).

`Host侧`: 统一通过 `ncclDevCommRequirements` + `ncclDevResourceRequirements` 链表描述资源需求,由 ncclDevrCommCreateInternal一次性分配和映射

`Device侧`: 通过模板化的 `Session` 对象(LLA2A / LsaBarrier / GinOutbox / GinInboxA2A)暴露原语,Kernel按需组合使用.

Host端调用`ncclSymkInitOnce`统一分配

所有后端的缓冲区、signal、counter 都以 `ncclDevResourceRequirements` 为统一单元, 挂在同一链表里

运行时根据 `lsaRanks < nRanks`**动态决定**是否要 GIN 资源

每类资源在分配后把 handle/id 回填到 `kcomm` 内对应字段, 从而设备端直接通过 `args->kcomm` 一次拿到全部句柄

Device端在kernel启动的时候, 通过传入`ncclSymkDevWorkArgs` 获得所有的句柄

```cpp
struct ncclSymkArgsHandler {
  ncclDevComm       const& comm;
  ncclLLA2AHandle   const& lsaLLA2A;
  ncclGinOutboxHandle    const& ginOutbox;
  ncclGinInboxA2AHandle  const& ginInboxRail;
  ncclGinCounter_t         ginCounterPerBlock;
  ncclGinSyncHandle const& ginSyncHandle;
  ...
};
```

需要注意的是, 数据发送的API并没有统一, Kernel会用三种标签 `ncclTeamTagLsa / ncclTeamTagRail / ncclTeamTagWorld` 和四类 Session 类(`ncclLsaBarrierSession`、`ncclLLA2ASession`、`ncclGinOutboxSession`、`ncclGinInboxA2ASession`)按需组合. 其中`ncclTeamTagRail`是针对常见的RDMA ScaleOut多轨道(Multi-Rail)组网而设计的, 我们后面将详细展开.

渣注
Symmetric Memory & Gin 只是针对较复杂的通信交互做了一个更简单的封装, 通过一个统一的对称内存的地址空间来方便并行计算算子的开发, 但是实际上底层的通信执行还是有很多复杂的操作和数据同步. 例如Gin依旧需要构造RDMA WQE(GDA-KI模式)或者GPU-Function Descriptor,GFD(Proxy模式).

另一方面数据同步模式也相对复杂, 在GPU内部TMA这些操作的Async Proxy和直接的LD/ST, 再加上RDMA的一些同步, 整体处理会非常复杂. 这是未来硬件设计上需要优化的地方, 例如将ScaleOut NIC接入到ScaleUP domain, 在软件交互层面避免LSA/Gin两种语义, 这部分内容的讨论未来会单独写一篇文章来阐述. 例如将TMA Descriptor和GFD统一, 降低编程难度.

| 机制 | 适用 | 实现 | 触发方 |
|---|---|---|---|
| Flag/Epoch (LLA2A) | 节点内小消息 | 16B slot 中 8B flag 编码 valid + epoch | 写入端 send |
| LSA Barrier (原子计数) | 节点内 CTA 级同步 | 共享 buffer 上 atomic/`multimem.red` | `arrive()` / `wait()` |
| GIN Signal (`ncclGin_SignalInc/Add/Set`) | 跨节点通知 | 由 put 的远端动作在对端原子加 | put 附带 |
| GIN Counter (`ncclGin_CounterInc`) | 本地源消费完成 | put 的本地动作在本地原子加 | put 附带 |

#### 1.2.2 LSA
1.2.2.1 LSA 数据交互
LSA支持直接的LD/ST和基于NVLink的多播处理, 它只需要根据如下函数计算目的地址即可

```cpp
// 本地 big-VA 起点
ncclGetLocalPointer(w, offset):
    return lsaFlatBase + lsaRank * stride4G + offset

// LSA 团队内第 peer 个成员的 big-VA
ncclGetLsaPointer(w, offset, peer):
    return lsaFlatBase + peer * stride4G + offset      // stride4G 以 4GB 为单位

// 多播写地址(一次写广播所有 rank)
ncclGetMultimemPointer(w, offset, mm):
    return mm.mcBasePtr + (mcOffset4K << 12) + offset
```

然后通信分为简单的LD/ST, 以及基于NVLink Multicast的 STMC/LDMC,其中:

LDMC(`multimem.ld_reduce`): 单指令向 NVLS Switch 下发,Switch 聚合所有 peer 的对应地址数据后返回 reduced 值

STMC(`multimem.st`): 单指令将值写到所有 peer 的同一地址

```cpp
for (int peer=0; peer<nRanks; peer++) {
    float* sendPtr = (float*)ncclGetLsaPointer(sendwin, sendoffset, peer);
    v += sendPtr[offset];           // 直接读对端显存
}
for (int peer=0; peer<nRanks; peer++) {
    float* recvPtr = (float*)ncclGetLsaPointer(recvwin, recvoffset, peer);
    recvPtr[offset] = v;            // 直接写对端显存
}
//STMC / LDMC
tmp[u] = applyLoadMultimem<Red, BytePerPack>(red, inputUptr + cursor);   // 一次拉取+硬件 reduce
multimem_st_global(outputUptr + cursor, tmp[u]);                         // 一次写广播全 LSA
```
1.2.2.2 LSA 同步机制
关于Nvidia GPU的内存模型可以参考[《谈谈GPU的内存模型及互联网络设计》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493955&idx=1&sn=0e880f3d509f0b494287cb552cbdb236&scene=21#wechat_redirect).

LSA同步机制通过`ncclLsaBarrierSession` 实现,  关于LSA的同步的数据结构在`src/include/nccl_device/impl/lsa_barrier__types.h`中定义, `ncclLsaBarrierSession_internal` 是 LSA Barrier 的**内部状态载体**,包含一次 barrier 会话所需的全部运行时信息. 设备端的 `ncclLsaBarrierSession` 继承自它,把"状态字段 + 地址计算方法 + 等待逻辑"组织在同一个类里.

```cpp
struct ncclLsaBarrierHandle {
    ncclDevResourceHandle_t bufHandle;
    int nBarriers;          // 通常 = ncclSymkMaxBlocks(64),每个 block 一个 barrier
};

template<typename Coop>
struct ncclLsaBarrierSession_internal {
    Coop               coop;        // CTA/warp 协作域(thread_rank / sync)
    ncclDevComm const& comm;        // 全局通信域,含 LSA 资源池指针
    ncclTeam           team;        // LSA team坐标 {nRanks, rank, stride}
    ncclLsaBarrierHandle handle;    // 资源句柄 {bufHandle, nBarriers}
    int                index;       // 本会话在 nBarriers 中的索引(通常 = blockIdx.x)
    bool               multimem;    // 后端模式: NVLS multimem or unicast
    ncclMultimemHandle mmHandle;    // Multimem 虚拟基址
    uint32_t           epoch;       // 当前 epoch 计数器

    uint32_t* mcInbox(bool multimem) { ... }
    uint32_t* ucInbox(int owner, int peer) { ... }

    template<bool EnableTimeout>
    ncclResult_t waitInternal(Coop, cuda::memory_order, uint64_t timeoutCycles);
};
```

其中nBarriers 通常 = ncclSymkMaxBlocks(64),每个 block 一个 barrier, 每个 CTA 根据其blockIdx, 让 nBarriers 个 block 共享同一块 state 数组但互不串扰. state数组按照以下方式有4段用途:

```cpp
state[0 .. nBarriers)                     : 本地 MC epoch 缓存
state[nBarriers .. 2*nBarriers)           : 本地 UC epoch 缓存
state[2*nBarriers + index]                : mcInbox (多播聚合计数)
state[3*nBarriers + index*nRanks + peer]  : ucInbox (单播 fanout)
```

前两个state用于持久化epoch, 后面mcInbox和ucInbox用于同步, 具体寻址方式如下:

```cpp
// 多播的计数器(一个 barrier 一个槽)
uint32_t* mcInbox(bool multimem) {
    state = multimem ? ResourceBufferMultimemPointer(...)
                     : ResourceBufferLocalPointer(...);
    return state + 2*nBarriers + index;
}

// 单播 fanout 格(矩阵: [index][peer])
uint32_t* ucInbox(int owner, int peer) {
    state = ResourceBufferPeerPointer(team, owner);   // owner 的内存
    return state + 3*nBarriers + index*team.nRanks + peer;
}
```

`mcInbox` 用 **multicast 写 + unicast 读**:arrive 时用 `multimem.red.add` 一条 PTX 累加到所有 rank 的同一位置, wait 时本地 `ld.acquire.sys` 读自己那份.

`ucInbox` 是**每对 (owner, peer) 一格**:peer 在 owner 的内存里留言"我到了".

为什么有Multicast和Unicast模式区别
在一些新的支持NVLink Switch Sharp能力的平台, 可以通过组播的方式向NVLink Switch发送一份数据, 让交换机复制到其它所有GPU, 因此对于较新的平台(Hopper以后)通常使用这种方式, 通过multimem写入.

而对于一些老的平台, 或者类似于RTX 6000pro这些没有NVLink ScaleUP的平台, 则需要通过unicast模式进行同步.

Epoch持久化
每个 barrier 独立持久化自己的 epoch(slot 0..nBarriers 给 MC,slot nBarriers..2*nBarriers 给 UC),让同一份 barrier 资源可被连续多个 kernel 接力使用. 具体实现时,通过构造函数`ncclLsaBarrierSession`读回上次的epoch, 然后同步完成后通过析构函数`~ncclLsaBarrierSession`写回 epoch.
arrive
在同步时, 通过arrive通知对端“我到了”, 如果支持Multimem(NVLS)的平台, 采用PTX指令直接操作

```cpp
if (multimem) {
    if (coop.thread_rank() == 0) {
        uint32_t* inbox = mcInbox(/*multimem=*/true);   // 多播地址
        if (release != relaxed) {
            asm("multimem.red.release.sys.add.u32 [%0],1;" :: "l"(inbox) : "memory");
        } else {
            asm("multimem.red.relaxed.sys.add.u32 [%0],1;" :: "l"(inbox) : "memory");
        }
    }
}
```

此时只需要leader线程进行处理即可, PTX指令解释如下:

`multimem.red.*.sys.add.u32`: NVLS Switch 把"+1"硬件多播到团队所有 rank 的同一物理 slot,并在每份上做原子加

`.release.sys`: 可选 release 语义 —— 保证 arrive 之前的所有 `.sys`-范围 store 对其它 GPU 可见

`.relaxed`: kernel 内部同步(后面还有别的 fence),省掉多余 release 性能更高

如果不支持Multimem, 采用unicast路径

```cpp
if (team.nRanks > 1) {
    // 显式 fence, 因为后面的 store 是 `relaxed`,必须先加 release fence 保证数据可见性
    cuda::atomic_thread_fence(releaseOrderOf(order));   
}
//所有线程并行写,`nRanks-1` 个目标分摊到 `coop.size()` 个线程,负载均衡
for (int i = coop.thread_rank(); i < team.nRanks-1; i += coop.size()) {
    int peer = i + (team.rank <= i ? 1 : 0);            // 跳过自己
    cuda::atomic_ref<uint32_t> inbox(*ucInbox(peer, team.rank));
    inbox.store(epoch+1, cuda::memory_order_relaxed);   // 写对端的mailbox
}
```
wait
首先我们来看Multimem模式, NVLS 一次 arrive 使本 slot 在每个 rank 看到的值都 `+1`. 因为有 nRanks 个 rank 同时 arrive,所以总增量为 `nRanks`. 因此等到的条件为 `count >= epoch + nRanks` , 单轮需要消耗`nRanks`.

```cpp
//只需要leader线程处理
if (coop.thread_rank() == 0) {
    cuda::atomic_ref<uint32_t> inbox(*mcInbox());  // ★ unicast 读
    while (true) {
        uint32_t got = inbox.load(acquireOrderOf(order));
        // 处理epoch绕回的情况
        if (got - (epoch + nRanks) <= uint32_t(-1)>>1) break;        
        //可以在wait的时候基于cycle数来估计timeout
        if (EnableTimeout && clock64() - startCycle >= timeoutCycles) { ret=ncclTimeout; goto exit; }
        else if (testAbort(abortFlag, steps)) break;
    }
    epoch += nRanks;   // 本轮 barrier 消耗 nRanks 次
}
```

对于unicast模式, 由于arrive: `ucInbox(peer, self).store(epoch+1)` 写对端mailbox仅epoch+1, 因此 wait读本地mailbox `ucInbox(self, peer).load() >= epoch+1`作为判断条件. 同时每个线程盯一个 peer 的mailbox,`nRanks-1` 个 peer 均分到 `coop.size()` 个线程,**并行等待**. 最后 `epoch += 1` 推进到下一轮.

```cpp
for (int i = coop.thread_rank(); i < nRanks-1; i += coop.size()) {
    int peer = i + (team.rank <= i ? 1 : 0);
    cuda::atomic_ref<uint32_t> inbox(*ucInbox(team.rank, peer));   // ★ 读本地ucInbox
    while (true) {
        uint32_t got = inbox.load(acquireOrderOf(order));
        if (got - (epoch + 1) <= uint32_t(-1)>>1) break;
        ...abort/timeout 检查...
    }
}
epoch += 1;
```
1.2.2.3 低延迟AlltoAll
LLA2A 是 NCCL 在 LSA 对称内存之上实现的小消息高吞吐 all-to-all 原语. 它抛弃了独立的 flag 数组, 把 **8B 数据 + 双 epoch 标志打包进一个 16B 事务**, 利用 NVLink 对 16B store 的原子性, 以一次原子 store 同时完成"送数据 + 通知完成". 具体代码可以参考`src/nccl_device/ll_a2a.cc`.它的结构定义在`src/include/nccl_device/impl//ll_a2a__types.h`:

```cpp
truct ncclLLA2AHandle {
  ncclDevResourceHandle_t bufHandle;   // 对称资源缓冲区句柄
  uint32_t nSlots;                     // 每个 block 每 epoch 的槽位数
};

#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclLLA2ASession_internal {
  Coop coop;                      // 协作组(warp / block / thread 等抽象)  
  ncclDevComm const& comm;        // 设备端Communicator引用     
  ncclTeam team;                  // 参与此次 A2A 的 rank 集合
  ncclLLA2AHandle handle;         // 资源句柄(含 nSlots)   
  int block;                      // 当前 session 使用的第几个块(对应一个 CUDA block 或并发单位) 
  int pitch;                      // 多 uint4 组成 T 时, T 内相邻 uint4 在 slot 维度上的跨距(单位 uint4)
  bool multimem;                  // 是否用 multicast 一次写多 peer 
  ncclMultimemHandle mmHandle;    // multimem 句柄     
  uint32_t epoch; 
  uint32_t slotsOffset;           // 当前 epoch 在本块内的槽位起始偏移(单位 uint4)  

  NCCL_DEVICE_INLINE uint32_t calcSlotOffset() const {
    return block*(1 + 2*handle.nSlots) + 1 + (epoch & 1)*handle.nSlots;
    // `block*(1 + 2*nSlots)`: 跳到当前 block 的起始
    // `+1`: 跳过 block 头部保存的"上次 epoch"元数据
    //`+ (epoch & 1)*nSlots`: 双缓冲选择, 奇偶 epoch 使用不同半区
  }
};
```

具体内存布局如下:

```cpp
outReq->bufferSize  = nBlocks * (1 + 2*nSlots) * 16;  // 16 = sizeof(uint4)
outReq->bufferAlign = 16;
```

以 `uint4`(16B)为最小单位, 单块个block如下:

![图片](assets/ac37ed38cc6b.png)
生命周期
它使用了一个`ncclLLA2ASession`的方式来管理生命周期. 代码在`src/include/nccl_device/impl/ll_a2a__funcs.h`中. 其中针对header字段(uint4), 仅使用第一段来持久化epoch.具体执行时, 依旧是构造Session, 执行send/recv/bcast等操作, 然后析构Session.

构造函数如下所示:

```cpp
uint4* line = (uint4*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
line += block*(1 + 2*handle.nSlots);   // 定位到 header 行
this->epoch = line->x + 2;             // ★ 读回持久化 epoch + 2
this->slotsOffset = this->calcSlotOffset();
```

析构函数, 仅 leader 参与并把当前 epoch (减 2 用以给下次 +2 留出 gap)写回 header,然后同步.

```cpp
if (coop.thread_rank() == 0) line->x = this->epoch - 2;
this->coop.sync();
```

为什么epoch + 2
析构时写的是 `epoch - 2`,跨 session 残留的 slot 可能携带旧 epoch 值 `last_epoch` 或 `last_epoch+1`,新 session 从 `last_epoch + 2` 起才能保证读到的 tag 不可能等于上次遗留值,避免误判.

大消息拆分
当 `sizeof(T) > 8` 时, `divUp(sizeof(T), 8)` 个 uint4 组成一条逻辑槽, 第 `v` 个 uint4 存放在:

```
addr = buf + slotsOffset + elt + v * pitch
```

通过 `divUp(sizeof(T),8)` + `pitch` 步长,`T` 可以是任意大小,每 8B数据一个 16B slot, pitch 让多个 slot 可能交错布局以减少 bank 冲突.
Send操作
Send函数如下所示:

```cpp
void send(int peer, int elt, T data) {
    union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
    tmp = data;

    // 1. 取 peer 的 slot 地址(通过 LSA 对称 VA)
    uint4* buf = ncclGetResourceBufferPeerPointer(comm, handle.bufHandle, team, peer);
    buf += this->slotsOffset + elt;

    // 2. 对每 8B 数据做一次 16B 原子 store
    for (int u=0; u < divUp(sizeof(T),8); u++) {
        asm volatile("st.relaxed.sys.v4.u32 [%0],{%1,%3,%2,%3};" ::
            "l"(buf + u*this->pitch),
            "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
            : "memory");
    }
}
```

具体在操作时, 采用`uint4={data_lo, epoch, data_hi, epoch}`的方式进行处理, `.sys`作用于为整个系统可以跨NVLink, 然后`.relaxed` 语义:无 fence 性能最优.

```cpp
st.relaxed.sys.v4.u32 [addr], {u32[u][0], epoch, u32[u][1], epoch};
                              ^^^^^^^^^^  ^^^^^  ^^^^^^^^^^  ^^^^^
                                 x         y        z         w
```
bcast操作
除此之外, 它还提供了广播能力, 可以利用NVLS进行加速.

```cpp
uint4* bufmc = ncclGetResourceBufferMultimemPointer(comm, handle.bufHandle, mmHandle);
bufmc += this->slotsOffset + elt;
asm volatile("st.relaxed.sys.v4.u32 [%0],{%1,%3,%2,%3};" ...);
```

针对非NVLS的环境, 可以使用Unicast unroll的方式,

```cpp
int r = this->team.rank;       // 从自己开始
for (; dr+8 <= nRanks; dr += 8) {
    #pragma unroll
    // 采用unroll8, 让 LDG/STG 流水线填满,NVLink 典型 latency 下 8 路并发覆盖掉单次延迟.
    for (int ur=0; ur < 8; ur++) {
        uint4* buf = ncclGetResourceBufferPeerPointer(comm, handle.bufHandle, team, r);
        buf += slotsOffset + elt;
        // 与 send 相同的 PTX 写入
        r = (r+1) % team.nRanks;
    }
}
```
Recv操作
Recv操作涉及几种模式, 共同点是接收端软件做了一个保障 `y == epoch && w == epoch` 就能判定数据已经完全可见.

recvUnrolled操作如下:

```cpp
void recvUnrolled(int eltStart, int eltCount, int eltStride, T (&elts)[MaxEltCount]) {
    uint4* buf = ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);   // ★ 本地 slot
    buf += this->slotsOffset + eltStart;

    uint4 tmp[MaxEltCount][divUp(sizeof(T),8)];

    // ——— 外层轮询循环 ———
    while (!testAbort(comm.abortFlag, steps)) {
        // ——— 内层 unroll: 一次读所有 MaxEltCount 个 slot ———
        for (int u=0; u < MaxEltCount; u++) {
            if (u < MinEltCount || u < eltCount) {
                asm volatile("ld.acquire.sys.v4.u32 {%0,%1,%2,%3},[%4];"
                    : "=r"(tmp[u][v].x), "=r"(tmp[u][v].y),
                      "=r"(tmp[u][v].z), "=r"(tmp[u][v].w)
                    : "l"(buf + u*eltStride + v*pitch) : "memory");
            }
        }

        // ——— 检查所有 slot 的双 tag ———
        bool okAll = true;
        for (int u=0; u < MaxEltCount; u++)
            for (int v=0; v < divUp(sizeof(T),8); v++)
                if (u < MinEltCount || u < eltCount) {
                    okAll &= (tmp[u][v].y == epoch && tmp[u][v].w == epoch);
                }
        if (okAll) break;   // 全部满足才退出
    }

    // ——— 解包提取数据位(discard tag 位) ———
    for (int u=0; u < MaxEltCount; u++) {
        union { T val; uint32_t u32[...][2]; };
        u32[v][0] = tmp[u][v].x;   // data_lo
        u32[v][1] = tmp[u][v].z;   // data_hi
        elts[u] = val;
    }
}
```

`ld.acquire.sys.v4.u32`中, `.acquire`确保了看到 epoch tag 后,所有"tag 写入之前"的内存操作都可见. 然后`.v4.u32`为 16B 原子 load,保证 `{x,y,z,w}` 四个字段来自同一次 store,不会拼凑. 然后读取采用并行操作, 先把 MaxEltCount 个 slot 一把读进 `tmp` 寄存器,再 AND 所有 tag. 即使只有一个 slot 的 tag 不对,也要**整批重读**. 这样让 LDG 保持流水, 这样比一个个读一个个判断快很多.  然后批量进行判断, 完全正确后才退出循环, 进行批量解包的处理.

另一方面, 它还支持在接收侧进行reduce的操作.

```cpp
auto recvReduce(eltStart, eltCount, eltStride, eltToAcc, reduce) -> Acc {
    Acc acc;
    int i = 0;
    // ——— 主循环: 每次读 Unroll 个 slot ———
    for (; i+Unroll <= eltCount; i += Unroll) {
        Elt got[Unroll];
        recvUnrolled</*Min=*/Unroll>(eltStart + i*eltStride, Unroll, eltStride, got);
        Acc acc0 = eltToAcc(got[0]);
        acc = i==0 ? acc0 : reduce(acc, acc0);
        for (int j=1; j < Unroll; j++) acc = reduce(acc, eltToAcc(got[j]));
    }
    // ——— 尾巴: 不足 Unroll 个的收尾 ———
    if (i < eltCount) { ... recvUnrolled</*Min=*/1> ... }
    return acc;
}
```
bcast + recvReduce构成allreduce
针对低延迟的allreduce就可以通过如下方法构建

```cpp
lla2a.bcast(/*slot=*/nIterPacks*rank + t, inp);   // 生产
AccPack out = lla2a.template recvReduce</*Unroll=*/8, Pack>(
    /*slotStart=*/t, /*slotCount=*/nRanks, /*slotStride=*/nIterPacks,
    /*eltToAcc=*/[&](Pack x) { return applyCast<T,Acc>(x); },
    /*reduce=*/  [&](AccPack a, AccPack b) { return applyReduce(red, a, b); });
```
endEpoch
endEpoch用于结束时处理, 并考虑epoch绕回的处理情况, 快到溢出边界时,如果 peer 还未消费的残留 slot 带有 `epoch = 0xFFFFFFFE`,下一轮 wrap 后 epoch=2,我方发出 `epoch=2`,peer 的轮询可能被残留 `0xFFFFFFFE` 误判(其实不会,但清零更保险). 这里清零保证无论下一个 epoch 值是多少,都不会撞残留.

```cpp
void endEpoch(Coop) {
    // ——— (1) 近溢出时清零全部 slot ———
    if (__builtin_expect(this->epoch >= -2u, false)) {
        this->coop.sync();
        uint4* buf = ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
        buf += this->slotsOffset;
        for (int i=coop.thread_rank(); i < nSlots; i += coop.size())
            buf[i] = uint4{0,0,0,0};
    }

    // ——— (2) 全员同步,保证本轮读写彻底完成 ———
    this->coop.sync();

    // ——— (3) epoch 推进: 正常 +1, 溢出时 +3 跳过 0,1 ———
    this->epoch += (this->epoch == -1u) ? 3 : 1;

    // ——— (4) 切换双缓冲 ———
    this->slotsOffset = this->calcSlotOffset();
}
```

一个完整的数据交互流程如下图所示:

![图片](assets/0ddd62a8caa9.png)

这种设计, 通过数据上保持16B(uint4)原子写, 并在数据结构上拆分支持双epoch的校验方式, 在无需 barrier 的情况下, 完成了低延迟的异步A2A. 但是它占用了2倍的带宽, 因此仅针对 NCCL 小消息低延迟需求的场景下使用.

#### 1.2.3 Gin数据交互

跨节点必须走 GIN session进行RDMA通信, GIN 是 NCCL 提供的"GPU 主动发起的网络通信"子系统. 其核心思想是让 CUDA kernel 在设备端调用 API(put / signal / wait / flush), 直接向远端 GPU 的对称内存窗口写数据, 而不再借助主机端的 ncclSend/Recv 调度. 它有两种后端:

GDAKI: GPU直接驱动网卡, 写wqe和敲doorbell

PROXY: GPU和CPU之间建立一个环形队列, GPU设备侧kernel向队列写入128B的GFD(GPU-Functional Descriptor)提交任务, CPU通过代理线程轮询并调用IBRC接口进行数据发送. 两种路径数据交互方式如下:

![图片](assets/27d124b9fc84.png)

为什么要两种模式
由于GDA-KI直接操作网卡, 实际上在工程上,不同网卡的wqe数据结构会因为不同的硬件结构有所不同, 这里针对的是Mellanox系列网卡而实现的. 为了保证第三方网卡的兼容性, 额外支持了Proxy模式, GPU还是可以在device侧发送工作请求到CPU内存侧, 然后通过CPU polling的方式再通过标准的IBRC verbs接口进行数据处理.

当然本质上这两种方式都不是最优的, GDA-KI利用CUDA Core构造wqe也需要大量的指令, 对于GDA-KI处理CQ队列也是一个非常消耗资源的事情, 而proxy模式占用了CPU的资源. 我一直在说RDMA并不是一个对GPU友好的接口, 未来需要更多的改进.

1.2.3.1 连接建立
RDMA连接建立在`nnclCommInitRank`中处理, 在连接建立时, 针对Gin它会调用`ncclGinInit()`初始化, 并调用`setLocalGinType()`选择后端决定[Proxy | GDAKI]. 然后调用`ncclGinConnectOnce()`建立所有 peer 的 connection, 最后调用`ncclGinDevCommSetup()` 为每条连接 createContext 并灌给 ncclDevComm.

需要注意的是, 它这里还有一个优化. 针对Multi-Rail的多轨道部署, 它会判断是否使用Rail模式, 即只与"同节点内相同编号 GPU"建连. 当然个人一直对Multi-Rail的拓扑持保留意见, 如果多路径做好了, QP规模不成问题, 还搞这个干嘛...

在`src/include/nccl_device/impl/comm__types.h`中定义了`ncclDeComm`. 与 GIN 有关的字段:

```cpp
int                 ginConnectionCount;        // 几条 NIC/连接
ncclNetDeviceType   ginNetDeviceTypes[4];      // 每连接后端类型
void*               ginHandles[4];             // createContext 返回的 devHandle
int                 ginSignalCount;
int                 ginCounterCount;
uint64_t*           ginSignalShadows;          // 本地 shadow, 紧跟 resource buffer
int                 ginContextCount;
bool                ginIsRailed;
```

设备端 `ncclGinInitCommon()` 在 CTA 里根据 `contextIndex` 取出对应的 backend、handle、rank 数等, 构造出轻量 `ncclGinCtx`.
1.2.3.2 窗口注册
窗口注册 `ncclGinRegister()`, 它会对每条连接依次:

```cpp
for (int d = 0; d < ginCommCount; ++d)
    ginBackend->regMrSym(
        ginComm[d], hostAddr, size, type, dmabufFd,
        /*out*/ &mhandle[d],          // host 侧, 用于 deregMr
        /*out*/ &ginDevWins[d]);      // device 侧, 写入 ncclWindow_vidmem::ginWins[d]
```

`regMrSym` 与常规 `regMr` 不同, 它要求**所有 peer 的对称地址在 VA 上一一对应**(通过 CU_MEM_ADDRESS_RESERVE 保留同一片 VA range 后 map 本地物理页实现), 因此单次注册即可让任意 peer 通过 `(dstOff, dstHandle)` 读写对方.
1.2.3.3 Gin 通信流程
Gin通信所使用的语义如下:

| 方法 | 语义 |
|---|---|
| `put(team, peer, dstWin, dstOff, srcWin, srcOff, bytes,`<br>`RemoteAction, LocalAction, Coop, desc, givenRelease,`<br>`requiredRelease, optFlags, SegType)` | RDMA Write；可附带 signal/counter；多段时自动切块 |
| `put(team, peer, ncclSymPtr dst, ncclSymPtr src, nElts, …)` | 对称指针重载 |
| `putValue(team, peer, dst, value, …)` | 立即数写入(≤8B), `T` 由 size 分派 `uint8/16/32/64` |
| `get(team, peer, remoteWnd, remoteOff, localWnd, localOff, bytes, …)` | RDMA Read |
| `signal(team, peer, RemoteAction, …)` | 纯信令, 无数据 |
| `flushAsync(team, peer, outRequest, Coop, optFlags)`<br>`wait(req, Coop, desc, ord)` | 返回 request 对象, 显式等待 |
| `flush(Coop, ord)` | 同步等待本 ctx 所有已发请求落地 |

以最常见的put为例

```cpp
template <typename Coop,
          typename RemoteAction,   // ncclGin_None / SignalInc / SignalAdd / VASignalInc
          typename LocalAction,    // ncclGin_None / CounterInc
          typename DescriptorSmem, // 可选 smem 描述符
          typename SegmentType,    // DevSegmentOnly / Mixed
          cuda::thread_scope givenRelease,
          cuda::thread_scope requiredRelease,
          uint32_t optFlags>
__device__ void put(
    ncclTeam team, int peer,          // 对端坐标
    ncclWindow_t dstWin, size_t dstOff,  // 远端目标
    ncclWindow_t srcWin, size_t srcOff,  // 本地源
    size_t bytes,
    RemoteAction remote,              // 如 SignalInc{signalPtr}
    LocalAction  local,               // 如 CounterInc{counterId}
    Coop coop);
```

在其内部会调用`teamRankToGinRank`, 这一步把"某个 team(world/lsa/rail)中的 rank 编号"换算成 **GIN plugin 实际看到的连接下标**. GIN 只关心"peer 在我的 QP 表里是第几个".

```cpp
// gin__funcs.h
teamRankToGinRank(comm, team, peer) {
    int worldPeer = ncclTeamRankToWorld(comm, team, peer);
    if (comm.ginIsRailed) {
        // RAIL 模式:只有同 rail 能互通,映射到 rail 内序号
        return worldPeer / comm.lsaSize;
    } else {
        return worldPeer;  // FULL 模式
    }
}
```

然后计算本地和远端地址

```cpp
size_t dst_ginAddr = 4096 * size_t(dst->ginOffset4K) + dstOff;
size_t src_ginAddr = 4096 * size_t(src->ginOffset4K) + srcOff;
ncclGinWindow_t dstGinWin = dst->ginWins[connectionId];   // = mhandle 指针
ncclGinWindow_t srcGinWin = src->ginWins[connectionId];
```

具体执行时会有两种backend, 一种是基于GDA-KI的backend, GPU自己构建RDMA wqe. 另一种是GPU和CPU之间构建一个GPU-Functional Descriptor,GFD描述符队列, 由CPU构建wqe.

GDA-KI路径执行如下:

```cpp
// ——— 1. 取上下文 ———
ncclGinGdakiGPUContext* gdaki = &((ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;  // ★ peer 索引 QP

// ——— 2. 构造 RDMA 三元组 ———
ncclGinGdakiMemHandle* dstMh = (ncclGinGdakiMemHandle*)dstWin;
ncclGinGdakiMemHandle* srcMh = (ncclGinGdakiMemHandle*)srcWin;

doca_gpu_dev_verbs_addr raddr, laddr;
raddr.addr = dstOff;
raddr.key  = loadConst(loadConst(&dstMh->rkeys) + peer);  // ★ 查 rkey 表
laddr.addr = srcOff;
laddr.key  = loadConst(&srcMh->lkey);

// ——— 3. 额外 signal/counter(可选) ———
if (hasSignal) {
    sig_raddr.addr = signalOffset;
    sig_raddr.key  = signalKey;
}
if (hasCounter) {
    companion_qp = loadConst(&gdaki->companion_gdqp) + peer;  // ★ 专用 QP
    counter_raddr.addr = sizeof(uint64_t) * (counterId + gdaki->counters_table.offset);
    counter_raddr.key  = loadConst(&gdaki->counters_table.rkeys) + ctx.rank;
}

// ——— 4. Fence 升级(若需要 system scope) ———
if (required == thread_scope_system && given > required)
    doca_gpu_dev_verbs_fence_release<SYS>();

// ——— 5. 发射(单次融合调用) ———
if (hasSignal && hasCounter) {
    doca_gpu_dev_verbs_put_signal_counter<...>(
        qp, raddr, laddr, bytes,
        sig_raddr, sig_laddr, signalOpArg,
        companion_qp, counter_raddr, counter_laddr, 1, codeOpt);
} else if (hasSignal) {
    doca_gpu_dev_verbs_put_signal<...>(qp, raddr, laddr, bytes, sig_raddr, sig_laddr, ...);
} else if (hasCounter) {
    doca_gpu_dev_verbs_put_counter<...>(...);
} else {
    doca_gpu_dev_verbs_put<...>(qp, raddr, laddr, bytes, codeOpt);
}
```

DOCA 函数内部做了什么:

在 QP 的 SQ ring 上分配一条 WQE 槽位(原子 `fetch_add(producer_index)`)

按 IB/RoCE 协议填充 WQE:opcode=RDMA_WRITE、remote_addr/rkey/lkey/sgl

如果有 signal:追加一条 RDMA_ATOMIC_FETCH_ADD WQE

如果有 counter:在 companion QP 上追加一条 loopback WQE(NIC 发回本机,更新 counter)

Doorbell:`ST.RELEASE [db_addr], producer_index` 一条 PTX 写到 NIC BAR,触发 NIC 读 WQE

如果是Proxy路径, 首先在设备端写GFD队列

```cpp
postGfd(coop, proxyCtx, gfd, pe) {
    cuda::atomic_ref<uint32_t, thread_scope_system> pi(proxyCtx->pis[pe]);
    cuda::atomic_ref<uint32_t, thread_scope_system> ci(proxyCtx->cis[pe]);

    if (coop.thread_rank() == 0) {
        // 1. 原子申请槽位
        uint32_t idx = pi.fetch_add(1, memory_order_relaxed);

        // 2. 等 CPU 消费(防止 overflow)
        while (queueSize <= idx - ci.load()) {}

        // 3. 16B × 8 = 128B write-through 写到 pinned 队列
        idx &= queueSize - 1;
        #pragma unroll
        for (uint8_t i = 0; i < sizeof(ncclGinProxyGfd_t) / sizeof(uint4); i++) {
            __stwt((uint4*)&q[idx] + i, ((uint4*)gfd)[i]);  // write-through
        }
    }
}
```

GFD(Generic Fast Descriptor) 128B 格式包含:

op 类型(`PUT`/`PUT_SIGNAL`/`GET`/`FLUSH`)

`srcOff / dstOff / size / peerRank`

`srcMhandle / dstMhandle` 指针(Host 侧可解)

signal 偏移/值、counter ID

然后 Host 端 Proxy Progress 线程通过Polling处理GFD和轮询RDMA CQ

```cpp
ncclGinProxyProgress() {
    for (each ginContext) {
        for (each peer rank) {
            // 1. 完成先前 posted 的请求
            proxyGinPollCompletions(state);   // 检查 state->request 完成,原子++ counter

            // 2. 读 GFD 队列
            if (proxyGinPollGfd(ctx, peer, &gfd)) {
                proxyGinProcessGfd(&gfd);     // 转发到 plugin
            }

            // 3. 推动 backend 进度(IB CQ 轮询)
            ginBackend->ginProgress(ctx);
        }
    }
}
```

在proxyGinProcessGfd中根据op类型进行处理, 例如put时会调用plugin `iput`来构建WQE

```cpp
void* srcPtr = (void*)(srcMrHandle->base_vas[self] + srcOff);
void* dstPtr = (void*)(dstMrHandle->base_vas[rank] + dstOff);  // ★ 绝对 VA
uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
uint32_t rkey = dstMrHandle->rkeys[rank];

ibv_send_wr wr = {};
wr.opcode              = IBV_WR_RDMA_WRITE;
wr.send_flags          = IBV_SEND_SIGNALED;
wr.wr.rdma.remote_addr = (uint64_t)dstPtr;
wr.wr.rdma.rkey        = rkey;
sge.addr               = (uintptr_t)srcPtr;
sge.lkey               = lkey;
sge.length             = size;

wrap_ibv_post_send(qp->qp, &wr, &bad_wr);  // 真正下发
```

Host 在下一轮 `ginProgress` 的 CQ poll 中,发现 WC 完成:

原子 `ci.fetch_add(1)` —— 设备端 `waitCounter` 能看到

若 GFD 带 localCounter,额外原子加对应 counter

| 机制 | 检测点 | 典型调用 |
|---|---|---|
| **Signal** (远端) | 对端看到 | `gin.waitSignal(signalPtr, expected)` |
| **Counter** (本地) | 本端看到 | `gin.waitCounter(counterId, expected)` |
| **Flush ticket** | 本端 | `auto t = gin.flushAsync(); gin.wait(t);` |
1.2.3.4 Gin 数据同步
GIN 在 flat VA 之外提供跨节点 mailbox 和信号机制:

| 结构 | 作用 |
|---|---|
| `ncclGinOutboxHandle` | 跨节点发送 mailbox(`bufHandle` + `counter0` + `size_log2`) |
| `ncclGinInboxA2AHandle` | 跨节点接收 All-to-All mailbox(`bufHandle` + `signals` + `size_log2`) |
| `ncclGinSyncHandle` | Rail barrier 信号集 |
| `ncclGinBarrierHandle` | 跨节点 barrier 句柄(`signal0` + 关联的 `ncclGin` 上下文) |

## 2. Symmetric Memory完整示例

基于Symmetric Memory的NCCL编程可以参看NCCL代码中过的`docs/examples/06_device_api/03_alltoall_hybrid/main.cu`. 该 Kernel 实现了**混合通信的 AlltoAll**: 对同节点(LSA 范围内)的 peer 使用**内存直接读写(LSA Store)**, 对跨节点的 remote peer 使用**GIN put(GPU-Initiated Networking)**.

### 2.1 initial

首先还是调用`ncclCommInitRank`进行初始化, 它会进行

**拓扑发现**, 包含探测 NVLink/NVSwitch/PCIe/IB 拓扑，计算 GPU-GPU/NIC 路径

**图计算**, 生成 ring、tree、CollNet、NVLS 四种通信图，确定 channel 数等

**传输建连**,为每个 peer pair 选最优传输（NVLink P2P / NVLS / IB / TCP），分配 buffer、交换 IPC handle / QP信息.

**架构能力标记**, 自动探测并设置 symmetricSupport、nvlsSupport、ginEnabled、collnetEnable 等, 这些影响后续集合通信的 algo 选择.

其中是否能够启用Symmetric Memory需要进行如下判断

```cpp
symmetricSupport = isAllCudaP2p          // 节点内全 P2P 可达
                && NCCL_WIN_ENABLE       // 环境变量开启 Window API
                && ncclCuMemEnable()     // CUDA VMM 可用
                && (sm70+，支持 multimem....)
```

### 2.2 分配对称内存与注册窗口

这是 Device API 的关键前置条件: **必须使用 `ncclMemAlloc` 分配**, 因为它使用 `cuMemCreate/cuMemMap` 基于 VMM 分配可对称映射的显存段.

```cpp
NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));
NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win, NCCL_WIN_COLL_SYMMETRIC));
```

`NCCL_WIN_COLL_SYMMETRIC` 告诉 NCCL 在所有 rank 上把这块显存映射到**统一的虚拟地址结构**, 窗口内部同时包含:

`lsaFlatBase + lsaRank*stride4G`: 本节点内任意 LSA 对端的可访问指针(LSA 路径用)；

`ginWins[i]`: 每条 GIN 连接对应的 RDMA 注册信息(GIN 路径用).

### 2.3 创建 Device Communicator

```cpp
ncclDevComm devComm;
ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.barrierCount       = NCCL_DEVICE_CTA_COUNT;        // 16
reqs.ginSignalCount     = NCCL_DEVICE_CTA_COUNT;        // 16
reqs.ginConnectionType  = NCCL_GIN_CONNECTION_FULL;     // 全连接 GIN
NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
```

| 字段 | 作用 |
|---|---|
| `barrierCount` | 为 `ncclBarrierSession`(LSA + GIN 混合 barrier)分配多少个槽位. 每个 CTA 需要 1 个. |
| `ginSignalCount` | 预分配的 GIN signal 数. 每个 CTA 使用一个 signal, 用于统计收到多少远端 put. |
| `ginConnectionType` | `FULL`: 每个 rank 到所有其他 rank 都建 QP；`RAIL`: 按 rail 建；`NONE`: 不启用. |
| `ginContextCount` | 建议的 GIN context 数量(默认 4). 每个 context 拥有一组 QP, 可独立并发. |

### 2.4 启动 Kernel 与验证

```cpp
HybridAlltoAllKernel<float>
    <<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA, 0, stream>>>(
        send_win, 0, recv_win, 0, count, devComm);
CUDACHECK(cudaStreamSynchronize(stream));
```

以启动配置: `<<<16, 512>>>`, 即 16 个 CTA, 每个 CTA 512 线程, 共 8192 个线程为例, 具体执行Kernel的代码如下:
阶段 0: 初始化与信号值快照
每个 CTA 用自己的 `blockIdx.x` 作为 signalIndex(因此 `reqs.ginSignalCount = 16`), 后面 `waitSignal` 的阈值计算基于这个 baseline.

```cpp
int ginContext = 0;
unsigned int signalIndex = blockIdx.x;      // 每个 CTA 拥有独立的 signal slot
ncclGin gin { devComm, ginContext };
uint64_t signalValue = gin.readSignal(signalIndex);  // 记录进入 kernel 时的信号基线
```
阶段 1: acquire 屏障(前置同步)
使用 **world team + GIN** 的 barrier(因此 `reqs.barrierCount = 16`, 每个 CTA 一个 barrier 槽位). `memory_order_acquire` 语义: 保证**所有 rank 都进入 kernel 并准备好接收**之后, 本 rank 才开始发送. 避免写入到对方的 recv buffer 时对方尚未启动.

```cpp
ncclBarrierSession<ncclCoopCta> bar { ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x };
bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
```
阶段 2: 对 Remote Peer 执行 GIN Put
World rank 布局: `[0 ... startLsa-1 | startLsa ... startLsa+lsaSize-1 | startLsa+lsaSize ... nRanks-1]`,, 中间段是本节点的 LSA team.

然后采用多 CTA 协作: `tid = threadIdx.x + blockIdx.x * blockDim.x`, `nthreads = 8192`；所有 CTA 的所有线程**联合分片**遍历 remote rank 列表(stride 为 8192). 每个远程 put 只会由某一个 CTA 的某个线程发起.  对端通知机制采用`ncclGin_SignalInc{signalIndex}` 告诉网络层: put 完成后在**对端的 signalIndex 槽位**做 +1. 这里 `signalIndex = blockIdx.x`, 意味着来自不同发送 CTA 的 put 会落在对端**不同的 signal 槽位**上.

```cpp
// 向 LSA 区段下方的 remote rank 发送
for (int r = tid; r < startLsa; r += nthreads) {
    gin.put(world, r,
        recvwin, recvoffset + world.rank * size,   // 目标: 对端 recvbuf 的第 world.rank 槽
        sendwin, sendoffset + r * size,            // 源: 本地 sendbuf 中给 r 的那一段
        size, ncclGin_SignalInc{signalIndex});     // 到达后对端对应 signalIndex++
}
// 向 LSA 区段上方的 remote rank 发送
for (int r = startLsa + lsaSize + tid; r < world.nRanks; r += nthreads) {
    gin.put(...);
}
```
阶段 3: 对 Local Peer 执行 LSA Store
通过 `ncclGetLsaPointer` 取得**同节点其他 rank 的 recvbuf 虚拟地址**(NVLink/P2P 可直达). 外层按 `offset` 分片给全部 8192 线程, 内层串行枚举 LSA 团队中每个 peer.  等价于: 把本地 sendbuf 中 `[wr*count, wr*count+count)` 段直接写入 peer `lp` 的 recvbuf 的 `[rank*count, rank*count+count)` 槽位. 这部分可见性由最后的 release barrier 保证.

```cpp
T* sendLocal = (T*)ncclGetLocalPointer(sendwin, sendoffset);  // 本地 sendbuf
for (size_t offset = tid; offset < count; offset += nthreads) {
    for (int lp = 0; lp < lsa.nRanks; lp++) {
        int wr = startLsa + lp;
        T* recvPtr = (T*)ncclGetLsaPointer(recvwin, recvoffset, lp);  // 对端 recvbuf 指针
        recvPtr[world.rank * count + offset] = sendLocal[wr * count + offset];
    }
}
```
阶段 4: 接收端等待 GIN put 全部到达
发送方使用 `ncclGin_SignalInc{blockIdx.x}`, 对端到达后在**发送方 blockIdx.x** 对应的槽位加 1. 因此**每个 remote peer 会给我贡献一个自增**, 但这些自增落在**哪个槽位取决于发送方 CTA id**.

由于本 rank 同一个槽位只被 “ 以同样 blockIdx.x 发送 put 且其 put 目标正好是本 rank” 的远端 peer 增量. 代码中通过 `receivingCta = (world.rank % nthreads) / blockDim.x` 挑唯一一个 CTA负责等待, 阈值是 `signalValue + numRemotePeers`, 表示该槽位要累计到 numRemotePeers 次到达.

```cpp
int numRemotePeers = world.nRanks - lsa.nRanks;
int receivingCta = (world.rank % nthreads) / blockDim.x;
if (blockIdx.x == receivingCta)
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + numRemotePeers);
```
阶段 5: flush 网络路径
确保本 CTA 发起的所有 GIN 操作完全完成(网卡侧已排空), 保证后续 release barrier 前网络侧无未决请求.

```
gin.flush(ncclCoopCta());
```
阶段 6: release 屏障(后置同步)
`memory_order_release`: 将本 rank 完成的 **LSA 写入** 发布给所有 peer. 这一步同时同步了 **LSA 参与者和 GIN 参与者**——所以主机端 `cudaStreamSynchronize` 之后, recvbuf 的全部内容(无论来自 LSA 还是 GIN)都已就绪.

```
bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
```

### 2.5 资源释放

```cpp
// ncclDevCommDestroy 先释放 barrier/signal/QP
NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
// ncclCommWindowDeregister 拆 VA 映射
NCCLCHECK(ncclCommWindowDeregister(comm, send_win));
NCCLCHECK(ncclCommWindowDeregister(comm, recv_win));
// ncclMemFree 归还 VMM 段
NCCLCHECK(ncclMemFree(d_sendbuff));
NCCLCHECK(ncclMemFree(d_recvbuff));
// 再做标准 Finalize + Destroy
NCCLCHECK(ncclCommFinalize(comm));
NCCLCHECK(ncclCommDestroy(comm));
```

参考资料

[1] 
GPU-Initiated Networking for NCCL: *https://arxiv.org/abs/2511.15076*
[2] 
Amem: *https://github.com/inclusionAI/asystem-amem*