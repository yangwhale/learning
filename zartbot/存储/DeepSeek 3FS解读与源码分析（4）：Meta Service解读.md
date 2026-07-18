# DeepSeek 3FS解读与源码分析（4）：Meta Service解读

> 作者: zartbot  
> 日期: 2025年3月17日 01:34  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493477&idx=1&sn=9c6967159f75b07ed319ca0ddb84b919&chksm=f995f7a7cee27eb134cdfbd198fb25586cbbb501afa7c69877e7ef1ceea057a5ce2920d8994b#rd

---

### 引言

在当今数字化时代，随着数据量的爆炸性增长，分布式文件系统已成为处理大规模数据存储和访问的核心技术之一。DeepSeek 开源的 3FS（Fire-Fly File System）作为一款高性能、高可用的分布式文件系统，凭借其创新的设计和强大的功能，吸引了众多开发者的关注。其中，Meta Service 作为 3FS 的核心组件之一，承担着元数据管理的关键职责，是整个文件系统高效运行的基石。

在本文中，我们将深入探讨 3FS 中 Meta Service 的架构设计、关键特性以及其实现机制。

### 整体架构

3FS 的整体架构由 Cluster Manager、Client、Meta Service 和 Storage Service 四部分组成。其中，Meta Service 负责存储和管理文件系统的元数据（如下图），采用存算分离的设计理念，将元数据持久化存储于 FoundationDB 中，同时利用 FoundationDB 提供的事务机制实现文件系统目录树语义。这种设计不仅提高了系统的可扩展性和可靠性，还通过无状态的 Meta Service 节点实现了横向扩展能力。

将元数据构建在高性能KV存储系统，构建易扩展的大规模文件系统元数据服务已成为共识，比如百度CFS、HopsFS、Microsoft ADLS等。可以看出3FS的元数据架构与此类架构一致，并在此基础之上其将文件系统元数据分解为 inode 和目录条目（directory entries）两种核心结构，然后通过事务化的键值存储实现高效的Posix操作。

![图片](assets/a377b16e9712.png)

如上图3FS元数据服务整体架构，3FS侧Meta请求会按照多种随机策略将请求转发到不同的MetaServer上，MetaServer再将Posix语义转换为KV事务请求下发到FoundationDB。整个请求路径上端到端没有任何缓存而是采用及其简单的事务配合以coroutine调度来满足高吞吐，与此同时MetaServer内部内置了多个组件配合保证Posix语义在事务场景下的高效运转：

• **MetaOperation**: 具体Posix请求的解析和处理

• **Forward & BatchOp**: 将部分高频可合并的写请求转发到对应的节点并Batch执行

• **InodeAllocator**: InodeID分配器

• **Session**: 在分布式场景下，维护文件open状态

• **ChainAllocator**: 文件数据Layout所依赖Chain分配器

这些组件共同协作从而实现了MetaServer高效的实现Posix语义，下文会从事务及语义两个角度分别详细介绍以上组件的作用。

### 元数据模型

由于3FS选择了键值存储作为元数据存储底座，那么在理解整体架构前首先让我们得了解下其如何组织这些元数据的。

为了满足复杂的Posix语义，3FS采用了Inode和Dentry分离的设计方式，两种结构的主键分别采用不同的前缀从而模拟出两种不同的数据模型，而每个文件或者目录目录会抽象为两条数据，比如下图中的/home/file1在KV视图中对应了第二条Dentry和第三条Inode。

采用这种数据模型的目的便是是为了支持目录树层级结构上以下几种读写模式：

• 两种点查询模式：路径查找+Inode属性查找，两种查找都可以转换为对应的Prefix Key上的点读，具体参考下面例子：

• 下图路径/home/file1的Path Resolve流程，既是在DentryTable依次查找出(DENT,0,"home")和（DENT,2,"file1"）

• /home/file1的Getattr流程，既是在Inode Table出查找(INOD,3)

• 一种范围查询模式：Readdir查询可以转换为基于DENT+父目录InodeID的范围查询，具体参考下面例子：

• /home目录下的list既在Dentry Table中以DENT+2为Prefix进行范围查询

• 写事务：一条或多条记录的写事务，具体参考下文目录树管理

![图片](assets/36d16d03e59d.png)

此外由于FoundationDB是基于Range Based来实现负载均衡, 这套主键编码格式可以保证3FS在元数据物理层面的Locality。因此以上数据模型还有如下优势：

• 可以让同一目录下的DENT聚集在一起从而高效的支持list请求

• DENT承载的信息较少，这样可以用较少的分片区管理整颗目录树的DENT，从而更好的支持一些Path Resolve场景

这里需要额外提到的是：MetaServer由于是无状态服务，除了Dentry和Inode外的相关信息都会以不同的Prefix编码存储到KV层，比如USER和CHAIN信息，这里不一一罗列。

### 事务模型

在KV的put/get以及RangeQuery基础上，3FS为所有的元数据服务提供了两种Transaction模型抽象：

• **IReadOnlyTransaction**: 类似一种SI隔离级别的只读事务模型，为Get和RangeQuery提供快照语义，用来支持大多数只读Operation(如Stat、list等)，

• **IReadWriteTransaction**: 类似一种SSI隔离级别的读写事务模型，提供了AddReadConflict和AddReadConflictRange接口，从而保证上层复杂Posix语义（如remove、rename）的安全性

此外3FS对事务模型做了一层抽象并于底层KV Engine解耦，所以3FS可以接入除FoudationDB外其他引擎，比如项目中同时实现了一种简易的支持MVCC的MemKV引擎。

#### 事务优化

事务是所有3FS内所有Posix请求执行的内核，所以高效的事务对于其元数据服务性能至关重要，为了让事务的处理在Meta服务中更加高效，让3FS在AI场景下有更好的效果，3FS在很多环节对事务也做了优化：
异步化+协程化
首先在FDB的IO路径上采用完全异步化实现，其将FDBFuture与3FS的元数据服务基于folly coroutine框架结合，将所有事务异步读写CallBack转换成coroutine里的Task，从而提升MetaService整体的吞吐和CPU利用率，与此同时简化了编程的心智负担，具体实现如下：

```
static void coroCallback(FDBFuture *, void *para) {auto baton = static_cast<folly::coro::Baton *>(para);  baton->post();}template <classT, classV>Task<T> Result<T, V>::toTask(FDBFuture *f) {  T result;  result.future_.reset(f);  folly::coro::Baton baton;  result.error_ = fdb_future_set_callback(f, coroCallback, &baton);if (result.error()) {    co_return result;  }  std::atomic_bool cancel = false;auto token = co_await folly::coro::co_current_cancellation_token;folly::CancellationCallback cb(token, [&]() {    cancel = true;    fdb_future_cancel(f);  });co_await baton;if (cancel.load()) {    throw folly::OperationCancelled();  }  result.error_ = fdb_future_get_error(f);if (result.error()) {    co_return result;  }  result.extractValue();co_return result;}
```
Batch处理
然后在MetaOperator层，3FS对一些高频且可合并的写事务做了**Forward**和**Batch**处理:

• 将不同Inode上的写请求的处理按照InodeID转发到对应的Node上(**这里依赖Distributer组件**)

• 同一个Meta Node下按照InodeID的Hash散到不同的BatchOP上，同一个Hash分片上的BatchOP采用队列组织，每完成一个Batch事务会唤醒下一个

• 同一个Inode上的写请求在BatchOP采用一个共同的Transaction并在本地做合并，比如多个Setattr可以在内存里Apply完最后一把提交

目前支持Batch化处理的请求包括Sync、SetAttr、Create、Close。基于以上优化不仅仅可以提升系统整体的吞吐，而且可以极大的降低FoundationDB侧事务冲突的概率。
InodeID小端序
由于FoundationDB采用了一种RangeBased的KV层的数据散列策略，尽管这样可以很好的保证数据的Locality，但是对于文件系统不同目录下的Inode还是期望能够均匀的散到KV层的不同分片上。3FS的InodeID是一个uint64值，如果InodeID作为主键采用递增的方式分配出去，必然会导致系统中产生热点。为此3FS采用了小端序的方式来对分配出来的InodeID进行编码.

此外因为InodeID需要保证全局唯一，所以在分配InodeID为了避免单点瓶颈，InodeID将uint64的值区间拆分成两个区间：

• 高52位：从全局ID分配器（DB）申请获取

• 低12位：本地负责分配

同时本地InodeID分配器也采用了类似滑动窗口的实现，在本地InodeID不足的情况下提前去DB分配高位InodeID。
缓存
尽管3FS整体架构充满了端到端无缓存的设计理念，但是仔细阅读代码还是发些一些缓存复用的设计。

在FoundationDB的事务处理模型中，每个读取操作需通过协调节点获取全局读版本号（GRV）以建立一致性视图。该过程因涉及跨节点通信，可能引入毫秒级延迟，成为元数据读取的性能瓶颈。

同时在一些批量获取元数据请求场景下某些数据可能会成为热点数据被频繁访问（比如root），这些重复的访问在同一上下文里可以被折叠和优化。为此3FS元数据加速系统采用双层缓存策略实现性能优化：

• GRV动态缓存层：这里依赖FoudantionDB内置的GRV缓存池，支持时间窗口内的版本号复用，但是这里会牺牲掉一些读一致性。

• BatchContext: 相同键的请求自动复用正在进行的加载操作，同时结合了协程和线程安全设计，优化高频元数据访问 Inode、DirEntry 场景的性能。

### Posix实现

Posix标准规范了文件系统在目录树和文件数据两个层面的操作接口行为，不同的文件系统的实现上也存在比较大的区别，由于遵循无状态无缓存的设计原则，3FS完全构建在KV引擎的事务之上，所以放弃了对Posix兼容度而采用简化的实现，下面分别从目录树和文件Layout两个维度介绍3FS对于Posix的实现。

#### 目录树管理

目录树管理涵盖很多操作，但是比较有代表的文件的创建删除以及Rename，3FS路径上也非常简洁既完全采用事务，此外为了避免到不同MetaServer之间的并发事务破坏Posix语义，每个写请求的事务内会额外指定Conflict Range。

**Operation****Transaction****Conflict Range**
create

1.检查Entry是否已经存在，存在则执行OpenExist
2.Allocate Chain来构建文件Layout
3.分配InodeId
4.写入Dentry和Inode以及Session(写打开)

父目录Inode
待创建的Dentry
避免孤儿节点

remove

1.判断删除对象类型，如果是文件将nlinks减1，如果是目录判断是否是递归删除
2.删除Dentry
3.非空目录递归删除和nlinks为0的文件会创建GCEntry

只有空目录的删除场景下会将对应Inode以及Dentry加入到Conflict Range避免孤儿节点

rename

1.检查各种边际条件：如目标目录是否存在且非空等
2.如果src和dst都是目录判断是否成环
3.删除Src Entry和Old Dst Entry
4.创建新的Dst Entry

Src和Dst的Parent Inode
以及Dentry
避免成环

这里需要注意的是3FS没有额外维护父目录的mtime的变更语义。

#### 文件Layout

3FS将文件按照固定大小切到不同的Chunk，这些Chunk同时会按照Stripe Size散在一堆Chain集合中。

因此对于Meta Service，每个文件的Layout只需要维护如下结构：

```
struct Layout {  // ChainTableId：存储链对应TableID  // stripe：条带参数，影响数据在Chain上的分布  // ChunkSize: 固定Chunk大小  // Chains: Chain的集合，目前支持Empty、ChainRange、ChainList三种};
```

以上Layout元数据信息都是**静态信息，在文件创建时生成**，这样在IO路径不需要额外访问元数据服务，只需要根据读写的Range计算出ChunkID并根据Layout算出对应的Chain即可。

这种Layout组织方式虽然解放了元数据服务层，但是完全静态的路由信息也带来弊端，如获取文件真实长度或者对文件做Trucate操作时，必须向Layout中的所有Chain发送请求，而对于一些小文件（小于Stripe Size）来说这无疑会增加不必要的网络层的开销，为了解决这一问题，3FS 引入了 **Dynamic Stripe Size **机制。该机制的核心在于动态维护一个“可能存放有 chunk 的 chain 数量”，类似于 C++ 中 `std::vector` 的扩容策略。具体来说，当文件大小逐渐增加时，chain 的数量会以每次翻倍的方式动态扩展（例如，从 1 扩展到 2、4、8……），直到达到 stripe size（200）为止。
Session
3FS为每个写打开的文件描述符（FD）维护了Session，这里的Session在文件写打开元数据处理过程会一并写到KV中作持久化，并在文件Close阶段删除，为什么需要引入Session机制？本质上是为了解决多个客户端同时写删一个文件引入的一些一致性问题：

• 在多个Client在向文件写入数据，如果文件被某个Client删除导致无效的写入而产生垃圾数据

• 并发的文件写入和Truncate操作，导致文件精确长度需要去收集所有Chain上的最后一个Chunk来获取，这样性能损耗非常大，因此需要一些Hint机制来保证文件长度的最终一致性

对于前者文件会让存在Session的Inode保活并让删除操作Delay到所有Session都被删除掉。

对于后者3FS会为每个Session周期（每隔5s）的汇报当前文件的最大长度，如果当前文件长度超过了元数据服务维护的长度且没有并发的Truncate请求的情况下，则更新文件长度，这里的具体实现时基于如下结构。

```
// src/fbs/meta/Schema.hstruct VersionedLength {  uint64_t length;   // 文件长度  uint64_t truncateVer;  // 截断操作版本号};
```

当然Client可能因为种种原因丢失Session，所以这里每个MetaServer会维护Session Mananger来周期回收掉这些Dead Session。

#### GC

为了让用户在删除文件尤其是递归删除目录路径上能够快速返回，3FS在元数据服务层设计了一个轻巧的回收站，所有的文件删除或者递归删除都会生成相应的GCEntry并交给后台任务执行。

![图片](assets/1d5da611f462.png)

在该回收站设计下，与Root目录树（InodeID为0）相对，在逻辑层元数据服务虚拟出另外一颗GC目录树（InodeID为1），该目录树为不同NodeID划分到多个专属的GC目录，目录名编码格式为GC-InodeID.x；而被删除的文件和目录会被相应的Node挪到自己回收站目录下，其命名规则为Type-Time-InodeID：

• 其中Type分为目录以及文件，而文件又会按照大小分为大中小三个子Type，这样做是为了GC的过程中区分优先级

• Time为删除的时间，这里应该是为了实现TTL，优先获取删除时间长的文件或者目录

• InodeID用于查询Inode信息

### 总结

3FS 元数据服务采用存算分离的架构，其核心数据一致性机制完全依托底层分布式事务型KV存储实现。服务层通过多维度优化手段显著提升元数据处理效率，包括但不限于：批量请求合并处理、基于协程的异步加载机制，以及在Posix实现上也是采用极简的设计：端到端没有任何缓存，请求事务化。其整体设计对分布式存储系统的元数据管理具有重要参考价值。

但从另一面可以看到3FS并未针对事务执行效率进行深度优化，尤其是写请求性能上限有限，所以在海量小文件场景不友好，但这恰恰不是3FS重点考虑的场景。