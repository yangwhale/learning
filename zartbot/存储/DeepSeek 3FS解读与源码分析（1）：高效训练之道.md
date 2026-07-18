# DeepSeek 3FS解读与源码分析（1）：高效训练之道

> 作者: zartbot  
> 日期: 2025年3月2日 00:56  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493351&idx=1&sn=cafc860454a35b6df86ce781722b9f02&chksm=f995f625cee27f338ec97df6677252decb8612ad3fad8de8b3d912f5297f292b9f737aaabc8b#rd

---

2025年2月 DeepSeek AI 项目的陆续开源引发了业界的强烈关注，每个开源项目都干货十足，也慢慢地揭开了 DeepSeek 高效训练背后的秘密。近期随着 3FS 的开源，再次引爆了存储业界。蚂蚁存储团队也在第一时间对 3FS 做了初步的调研，并结合团队自身在 NLP、多模态等多个场景万卡规模训练的支持经验，从设计文档和源码，深入对 3FS 在文件系统和 AI workload 方面做一系列的解读。如有错误欢迎指正。

## 3FS Design Doc

https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md

3FS 满足 AI 处理过程中的大部分场景：
**Training data preprocessing**
数据集的预处理需求。
**Dataset loading**
训练过程中的数据集读取需求 。
**Checkpointing**
训练过程中，**高并发 **checkpoint 文件的写入。
**KVCache for Inference**
为 KVCache 提供了比 DRAM** 更加经济的替代方案，提供更低的成本和**更大的容量（回答更smart）**

Embedding vector search

提供向量搜索功能的持久化。

### 系统架构

**整体架构**

![画板](assets/38eaa9243ed1.jpg)
画板
**核心组件**

• Cluster Manager：用于管理整个集群的状态和 FO ，其它组件通过 Heartbeat 和 Cluster Manager 交互，Cluster Manager：本身通过多副本的选主机制来保证自身的可靠性，使用 ZK 或者 ETCD。

• Metadata Service：存储文件元数据，3FS 的元数据使用独立/外置的 FundationDB 集群来存储，FoundationDB** 本身是一个完备的分布式 KV DB，Metadata 的数据可靠性由 FoundationDB 来保证。

• Storage Service：存储数据，数据同步协议是 CRAQ，Storage Service 自身来保障数据的高可用性。

• Client：对性能不敏感的应用使用 Fuse 客户端，对性能要求高的应用使用 native 客户端。

**外部依赖**

• ClickHouse：用于存储服务产生的 Metrics。

• FoundationDB：用于元数据服务存储文件元数据。

• Zookeeper/etcd：用于 ClusterManager 实现多副本选主。

### MetaService 元数据服务 

#### 方案

类似于 ADLS，3FS 使用分布式的、支持 SSI 隔离级别事务(和 2PL 实现的一样，都是可串行化的隔离级别)的数据库 FoundationDB。基于串行化隔离级别事务实现会大幅降低文件系统目录树的设计实现成本，所有的复杂性都交由数据库解决，因为串行化隔离级别事务是 ACID 完备的，特别是 Consistency，不存在任何外部一致性问题，**语义上等价于所有事务是一个一个按序执行的**，**而每一个目录树操作都基于事务做，自然就是等价于每个目录树操作都是等价于串行化运行的**，自然不存在任何一致性问题。因此没有基于 SSI 隔离级别实现时的异常，比如下面的 rename 成环异常：

![图片](assets/8d54df0e2250.png)

借助于 SSI 事务，状态都在数据库维护，因此 3FS 的元数据服务 MetaService 可以做到无状态，所有的 MetaService 在目录树结构的层面都相当于一个可有可无的 proxy。这里不得不提到，3FS 完全可以去掉 MetaService 这一层，让元数据请求路径更短，逻辑都在 client 中处理即可。当然保留 MetaService 的好处是如果元数据处理逻辑有 bug 需要修改可以不依赖于 client 升级 MetaService 即可，另一方面也能够避免 client 如果过多会造成数据库 session 过多可能有负面影响。

#### 实现

元操作利用 FoundationDB 的 SSI 事务：

• 只读事务用于元数据查询：`fstat`、`lookup`、`listdir`等。

• 读写事务用于元数据更新：`create`、`link`、`unlink`、`rename`等。

所有的一致性复杂性都交给 FoundationDB 解决，当比如 rename 成环问题出现，事务层就会冲突导致有一个事务被取消，之后 3FS 元数据服务会自动重试事务。小团队设计存储和数据库系统基本会采用这种设计模式。

思考

可以认为3FS在元数据服务架构属于HopsFS、Tectonic以及百度CFS**这一流派，考虑到Posix在文件系统层级结构中存在事务属性：比如rename成环问题，以及create file与unlink并发可能出现孤儿节点等问题，在这类系统出现之前很多元数据服务都是作为专有系统来做，但这样就很难解决跨分片事务问题，所以这类系统的思路是将问题下沉到分布式数据库，然后将元数据服务层实现为无状态，这样兼具了横向扩展同时保证了事务属性。

这里强调了3FS之所以选择FoundationDB是因为其支持SSI的**事务隔离级同时又支持简单高效的KV接口**。同时又介绍了在FDB上怎样构建自己的schema。比如将Inode和Directory作为两种独立的record分别组织在kv里。

从3FS源码可以看到在MetaService上很多设计和百度CFS是高度契合的，尤其是FoundationDB选型上可以看出这个团队在这块确实做了大量的工作和经验积累的，HopsFS这类系统对于底层数据库的选型及使用过程中会发现过重的事务处理和SQL解析没有必要，而一个支持强一致的分布式键值存储更加合适，这也是FoundationDB的定位。但是FoundationDB的乐观事务模型在事务冲突场景下性能会衰退的比较厉害，所以这里可以看到3FS在schema上有针对Inode和Dentry作了分离的设计，这也是百度CFS在schema上最关键的关键，至于这里是否是做了类似于百度CFS中针对事务冲突的优化还有待进一步在代码里确认，但是可以Design强调了不同的type采用的不同Prefix，这里**其实意图很明显了**：

让同一目录下的Dentry聚集在一起，这样减少readdir的交互。

让文件系统中Dentry聚集，这里**大概率（待求证）**是为了将文件系统里的Dentry集合在少数甚至一个分片上（相比于Inode的属性集合，Dentry承载的信息更加简单），从而可以避免rename跨分片事务，这里设计确实很妙

此外这里很多细节也是拉满，确实是在很多点上做的很细致：

Inode才用little-endian：这里是为了将业务层的元数据请求均匀的散列的到不同分片上，因为这里inode id作为kv系统里的key，如果采用自增的方式必然会导致热点。

文件Inode属性上存在parent id：这里解释是为了在rename场景下判断是否会成环，实际上通用的文件系统上也需要支持lookup。

在数据库选型上采用了外部数据库FoundationDB，我想这里DeepSeek团队应该是出于一些方便开源的考虑，换一个角度存储自带元数据底座也有一些其他优势：

高度定制化的元数据接口让性能更好：比如百度 Tafdb实现了高度定制1pc原语而非简单的kv接口
可以定制一些目录树友好相关的均衡策略：可以针对数据对数据湖做自适应的分裂策略

Chunk** 位置
和 GFS** 一样，3FS 在元数据中不维护 chunk list 这样的结构信息，chunk 的 id 和位置都是可计算的。

• **Chunk id**：Chunk 是定长的，因此 chunk 在 inode 中的位置(chunk_index)可以通过 offset 除以 chunk_size 得到，chunk id 由 ino 和 chunk_index 构成，所以 chunk_id = {ino}{chunk_index}

• **Chunk 的位置路由**：每个 inode 在创建之后都会分配 chain_table 和 shuffle_seed，shuffle_seed 和 chain_table 都是常量，因此想要获取一个 chunk 的位置只需要根据 chunk_index, shuffle_seed 和 chain_table 即可计算出具体属于哪一个 chain。而全局 chain 的位置信息维护在 MetaService 中，client 会缓存这个信息。
数据结构
**Metadata****Key****Value****Description****dentry**
DENT{parent_ino}{name}

{parent_ino}{ino}{chain_table}{chunk_size}{stripe_size}{owner}{permission}{atime}{mtime}{ctime}

1. DENT 和 INOD 前缀的目的是为了做数据隔离，FoundationDB 是一个大的全剧有序的 KV，没有 table 的概念，如果不加前缀隔离则有可能 scan 某个目录下的子时扫出来不属于自己子的其他 inode
2. ino(inode number) 是小端编码存储的，这样能保证低位在低地址，而因为 ino 是从小打到分配，如果不这样做那么数据前缀都相同(比如一开始都是 000....n)没法做数据分区和打散
3. inode 不维护 chunk-list，因为 chunk 定长，因此 chunk index 可以根据 offset 除以 chunk_size 得到，这个设计和 GFS 是一样的
**inode**
INOD{ino}
{file_length}{chunk_size}{selected_range_in_chain_table}{shuffle_seed}
{owner}{permission}{atime}{mtime}{ctime}
{target_path: just for symbol link}

动态文件属性
• **延迟回收处于写打开状态时被删除的文件**：Posix 规范要求已打开的文件即便被 unlink 也可以正常读写。3FS 没有实现这么强的语义，只是保证了写打开的文件会被延迟回收，目的是避免产生垃圾数据。3FS 为每个写打开的文件维护状态，MetaService 会延迟删除一个写打开的文件。分析具体的实现是维护了类似 delay_unlink 前缀作为 key 的 inode 信息表，里面 value 部分也维护了打开该文件的 clients，只有当 clients 为 0 了才会回收。如果 client 挂了，MetaService 会检测到并在 session 到期之后直接释放 client session。而文档中提到对于读的量太大，没有做类似的行为，因此分析读打开的文件被删除之后访问会报错，比如 chunk not found，转为 errno 就是 EIO。

• **文件的长度更新**：3FS 对于文件长度更新是 lazy 的，客户端会定期（默认每 5 秒）向元数据服务报告每个以写模式打开的文件的最大写入位置。如果此位置超过了 inode 中存储的长度并且没有并发的截断操作，则该位置会被采纳为新的文件长度。

![图片](assets/adcaad59a006.png)

设计文档中对于 inode 长度的更新有不太理解的地方（如果有理解错误请指出），既然客户端维护了最大有效长度，在 sync/close 时就可以作为 length hint 传递给 MetaService，进而 MetaService 就没必要再去 query 所有的 chunks 获取指定 inode 的最后一个 chunk 的 id 和长度了。目前分析是因为当 client sync/close 之前如果 session 断开了，MetaService 可以在回收 session 之后通过这种方式来得到未提交的 chunk，进而避免降低垃圾 chunk 数据的存在(已写入但是长度还没有提交到元数据中)。

• 如何处理并行写入的同时发生 truncate 的情况？最简单的方式是增加一个 truncate_version，update length 请求必须满足 inode truncate_version == req.truncate_version 才允许。当然这么实现比较简单且有针对性只能应对这一种场景，最优雅的还是 GPFS 的分布式锁方案。

• 为了避免并发写入过程更新相同文件的元数据，导致 FoundationDB 由于事务冲突出现性能劣化，3FS 在客户度中通过 rendezvous hash 保证相同 inode ID 的请求会打到固定的元数据节点上，并在元数据节点上提前对这些请求进行合并和处理。

#### 分析

整体方案和微软的 ADLS 是一致的，都是利用串行化隔离级别事务解决文件系统目录树的一致性问题。文件系统本身没有像字节 DanceNN v3，百度 CFS 那样基于低隔离级别事务并在应用层做拆锁(CFS 虽然抽象了一些 CAS 原语但是这些原语也是需要锁来实现的)，和 HopsFS 也不一样，HopsFS 也是仅依赖 RC 隔离级别事务，需要 App 层面维护锁。

• **优点**

• 规模上可以一定程度的 scale-out。之所以说是一定程度上是因为 SSI 的事务实现依赖于一个全局的单点 TSO，这是系统的瓶颈所在。

• 开发设计简单。基于串行化隔离级别事务，所有复杂性都在数据库。

• 潜在问题

• 性能一般：但是对于 AI 场景，可以利用 client 缓存缓解，因为 AI 场景写少读多且 IO Pattern 简单，一般 Close-to-Open 语义即可满足需求。

• 两阶段提交：写在数据库未经优化时需要两阶段提交事务，开销比较大。

• 读生写：SSI 隔离级别的实现每一个读请求都需要有一条 SIREAD 记录产生写流量，读性能一般(不排除 FoundationDB 有优化)。没有 SIREAD 就没办法完整追踪事务的依赖关系导致事务之间成环了也检测不出来。

• 如果维护目录项 mtime，当一个目录的子大量变动时事务冲突会比较多，SSI 这种 OCC 的实现面对这种情况表现不佳，远不如 2PL 的实现性能好，因此需要上层来解决冲突并做 batch 优化，类似于 DanceNN v3 做的那样。

### Chunk Storage Service 存储子系统  

和 Ceph 类似，3FS 的数据也是条带化的。3FS 系统创建之后会维护一系列基于 CRAQ 一致性协议维护的 chains。一个文件在创建之后会根据配置信息创建指定个数的条带，条带和 chain 是一一对应的，即 inode 中维护的 chain table。文件的每一个 chunk 数据块根据 round-robin 之类的策略分配所属的 chain(条带)。

相比于 HDFS，3FS 的数据块写入是基于固定复制组(CRAQ chain)的，而 HDFS 的数据块写入是面向全集群的，每一个数据块多副本 pipeline 都是“临时的”，故障重试并不依赖于复制组的状态恢复，可用性比 CRAQ 更好。

#### CRAQ 而不是星型写

![image.png](assets/1ca70d2a15e8.jpg)

写入：头节点接收来自客户端的写入操作。此时数据被标记为脏数据。然后，头节点通过后继节点传播写入操作。尾节点收到数据后，会将其标记为干净数据，最终尾节点会通过链向后发送确认信息。

读取：读取可以在链中的任何节点上进行。每个节点都保留一份相同的数据副本。如果脏节点收到读取请求，则该节点会联系尾节点以检查其状态。作为回报，尾节点会发回其状态，这有助于保持强一致性，因为读取操作相对于尾节点是序列化的。

备注：对读友好，写延迟大（不过AI 场景如果多客户端并发写以及可见性要求低的话，客户端缓存数据，保持 Single Writer Multiple Reader的语义也是还好）。

CRAQ 理论见论文：https://www.usenix.org/legacy/event/usenix09/tech/full_papers/terrace/terrace.pdf

思考

• **分层下故障恢复时职责清晰确定**：集群管理器只需要对 chain 的状态进行管理，故障副本的恢复由 CRAQ chain 自行处理

• **客户端带宽低**：相比 GPFS 和盘古之类系统的星型写，写时客户端出口带宽只有一份这是优势。但如果考虑到直写就明显没有优势了(EC副本的情况下是: 1/1.3)。具体可参见下图:

![image.png](assets/00d1b26f8e1c.jpg)

•潜在的问题

• **可用性低**：相比 GPFS 的 quorum 星型写，可用性更低。3FS 类似于 Ceph，为了维护 chain 的一致性状态，因此相当于多了一层状态管理，当写入由副本故障时需要需要等待相应副本恢复或者被判定 offline，在此期间会 stop the world，Ceph 这个设计就很让业务头痛，fuse 进程有时候长时间处于 D 状态无法处理。当然 3FS 可能对此优化，要再具体分析代码才能知道。

• **EC 困难**：链式复制相比于星型写很难做在线 EC。如果需要 EC 只能是离线 EC，这会导致后台的读写放大以及元数据管理的麻烦。

#### 数据分布

类似于预先定义好的复制组（CopySet）。user、specify chain table、chunksize、stripe size （Per Directory based Policy），选择策略基于如上 Table 的 Round Robin 

![图片](assets/46e642b29493.png)

这一段简而言之就是 chain 的副本构成没有任何 node 约束。也就是说对于上表，A-E 是节点，每一个 chain 的副本都是可以分布在 A-E 这些任意 node 上，而不是某些 chain 只能 A-C，某些 chain 只能 D-E。当然，3FS 提到了 node 也是分 group 的，这样做的好处是在线和离线任务能够很简单的做隔离避免干扰，进而某个 group 的 chain 的副本只能在对应 group 的 node 上。
关于 CopySet ，FaceBook 有一篇专门的文章对 CopySet Replication 详细分析 ：Copysets: Reducing the Frequency of Data Loss in Cloud Storage | USENIX。

通过 CopySet 约束副本的放置，可有效降低在换盘情况下的丢数据概率，同样通过预定义的 CopySet + 放置算法可以使得写入带宽能够相对很好的打散到整个集群。 文中也提到可以在 Directory 等层面设置对应的 CopySet，对于集群扩容，以及在离线的负载打散层面 都是比较容易和运维的，运维结果的可预期性比较高。
### 故障处理

3FS 为每一个 chain 都维护了版本号，注意此版本号不同于 chain 副本中维护的 chunk 版本号。

• **Chunk 版本号**在 CRAQ 协议内部使用，用于决议一个 chunk 的 commited 版本

• **Chain 版本号**用于故障处理，对外可见，最重要的是对 client 可见

整体基于 lease 机制。每个数据节点，MetaService 都和 ClusterManager 维护 lease。

每个 chain 副本(target)都会在维护：

**本地状态**：

![图片](assets/e20a8092f9ce.png)

**全局状态**：

![图片](assets/611631641fd1.png)

ClusterManager 可以根据状态管理集群，这块比较常规不多做介绍了。

#### MetaService

MetaService 无状态比较简单，一旦 lease 到期则从集群中移除掉，lease 到期或者是因为自身故障 crash，或者因为网络隔离被 ClusterManager 移除。Client 把请求重试发送到其他 MetaSerivce 上，也会更新新的拓扑。

#### 数据节点

每个数据节点管理着多个 chains 的副本(3FS 称为 target)，每个副本含有多个 chunk 对象副本。当一个数据节点 lease 到期或者检测到硬件损坏主动下线，集群管理者就会升级 chain 的版本，修改 chain 的故障节点到 chain 的 end(在 TAIL 之后)，并广播消息。

这段逻辑的处理类似于 GPFS ，GPFS粒度更细只是受影响的 chunk 才不可读(当然如果彻底 down 掉新的空副本加入则没有历史数据则完全不可读)。故障副本恢复后加入 chain 后进入 syncing 状态，此时副本接受数据写入，但是不可读。在恢复期间，先副本扫描本地所有 chunks 作为待恢复列表，恢复时每一个 chunk 都需要加锁以免出现客户端写入了新的数据 chunk 成功之后被后台拉取的旧 chunk 数据冲掉导致不一致。当所有 chunks 恢复完成后该副本即可加入集群。

#### 一定程度的 Fencing

每一个 chain 都有版本，当成员变更时 chain 的版本会提升。如果此时客户端的写入请求附带的是旧的 chain 版本则拒绝请求。否则如果不拒绝则存在这种可能：

1. Client1 写入 chunk1 数据请求 R1 到 HEAD，HEAD 还没来得及处理时

2. HEAD 隔离，client1 crash

3. Chain 成员变更

4. 随后 HEAD 恢复并重新加入集群

5. Client2 写入 chunk1 数据成功

6. Client1 在 HEAD 中未处理的 R1 得到处理，导致 Client2 的数据被非预期覆盖

如果有 chain 版本管理，在上述第 3 和 4 步 chain 的版本会提升，随后在第 6 步请求 R1 会被拒绝。之所以说这是一定程度的 fencing 是因为如果 chain 没有变化，client1 写入 R1 在 HEAD 中未处理完时 client1 crash，新的 client2 后来接管任务后成功写入数据，随后 client1 的 R1 得到执行(requst 的版本chain 当前版本一致)就会非预期地覆盖 client2 写入的数据。当然这个也很容易解决，比如在 chain 执行请求之前加锁，在锁内判断 client1 的 lease 有效性即可。

GPFS 解决这个问题是另外一个思路，整个处理流程无需加锁，通过全局维护 client lease 解决，当检测到 client offline 后，全局管理器会广播到数据处理节点，当且仅当所有数据处理节点都恢复收到后(条件是一切关于这个 client 的请求都结束且 client 从有效列表中移除)才会对 client 持有的文件锁释放掉让其他 client 可以写该文件。

#### 分析

3FS 在故障处理方面比较完善，也算是业界系统常用的处理手段，但是单从文档上看还是有可能在极端情况下出现外部一致性被破坏的情况但是概率极低。

文件系统接口

设计文档表达了观点，对象存储接口虽然更加 popular，更cloud native，但文件口有目录树，更易和单机系统各种文件格式接入，还有软硬链接等，所以做了这个选择。

#### 使用模式

应用程序有两种模式使用 3FS：

• **Posix API 文件接口**：依托于传统的文件系统接口而不是对象存储接口，主要原因是：

• **有目录原子操作的需求**，对象存储没有这个能力。比如对于目录的 rename，存在大量小文件目录的递归删除(类似于 HDFS recursive O(1) 的方式删除一个目录而不需要像 Posix API 一样遍历出来挨个删除)

• **软连接和硬链接**：应用程序利用这样的能力实现快照功能。不同于 HDFS 中快照的实现方式，3FS 的快照需要应用程序配合实现。应用程序先利用软硬链接的功能创建快照，增量的部分创建新文件写进去，而 HDFS 的快照增量变更的部分是文件系统做的，应用程序无需自管理。

• **普适的接口**：Posix 接口广为人知使用起来没有学习成本，而且许多框架的数据集是 CSV/Parquet 格式基于文件系统接口很自然直接。

• **Native Client API**：有些应用程序对性能有极致的要求会使用原生 client。

#### Fuse

3FS 基于用户态 libfuse 而不是 kernel fs 实现的文件系统，他们的考虑是内核模块开发比用户空间系统编程要复杂得多，调试错误非常困难，可能会导致生产环境中的灾难性故障。

• **实现**：元数据操作即控制面依然走的 libfuse，数据面走的是共享内存。共享内存用于在应用程序和 3fs native client 之间通信和数据传输。

• **lov**：InfiniBand(IB) 共享内存，数据面的数据走这个通道，可以做到数据完全 zero-copy。用户**写**数据直接写到 IB 共享内存中，之后提交请求到 lor 的 ring buffer，fuse 中的 native client 会处理对应的写请求，走 IB 直接把数据发送到对端的 IB 内存中。用户读取的数据也类似，先发送读取请求到 lor，fuse native client 收到请求后直接把远端数据写到 lov 后应用程序直接使用 lov 中的数据即可。

• **lor**: 自管理的基于共享内存的 ring buffer，数据流的控制信息走 lor 通道。比如写数据在数据写入到 lov 之后需要提交到 ring buffer 一个写请求，fuse native client 收到请求后直接通过 IB 把 lov 的数据发送出去；读数据需要先提交到 ring buffer 一个读请求，fs native client 收到请求之后通过 IB 把直接数据写入到 lov 中。

• **背景**：3FS 这么做的原因是标准的 libfuse 控制流和数据流都需要走内核，需要占用内存带宽也会增加端到端时延。而且 libfuse 性能也不够好，内核是维护单队列和用户态 fuse 进程交互，单文件也不支持并发写入。

• **为什么控制路径走 libfuse**：3FS 数据流基于共享内存管理，但是控制流依然走标准的 fuse 路径，比如 open，close，stat 而并没有全部走 ring buffer + native client。估计有以下几点考虑：

1. **开发和管理简单**。少一些接口的开发量，而且基于 fuse 也方便在客户机中做 mountpoint 管理。

2. **需要一个守护进程维护基本的状态信息**：集群拓扑，数据节点地址信息，客户端为了实现并行写单文件需要的 session 管理都需要一个常驻的守护进程来管理而不是一个 lib 受限于应用程序生命周期，使用 libfuse 很自然和方便。

3. **需要共享的缓存能力**：元数据需要做缓存。

1. 3FS 文件系统元数据是基于 SSI(快照可串行化，等价于两阶段锁实现的可串行化) 隔离级别事务实现的，SSI 隔离级别读请求也会产生 disk 写 IO，性能不好，所以依然要依赖于客户端的元数据缓存。

2. 基于 libfuse 实现的统一 daemon 进程可以在多客户端访问时做到缓存共享，而不是以 lib 的方式集成进应用程序时缓存各自一份。

3. 有了 fuse daemon 程序的缓存，在模型重启时有更好的冷启动性能。

## 训练相关Workload解读

## 在官方开源的介绍里提及了 3FS 对多种 AI workload 的高性能支持，但是开源项目里没有体现相关的实现。在重新阅读了 DeepSeek 的论文后发现这部分关键内容在论文里。训练优化需要依赖多系统的 co-design，文件系统只是训练提效其中一环，这部分和蚂蚁当前在MOE场景里的观察结果基本一致。

### Checkpointing

![图片](assets/8a8599e0bbc7.png)

![图片](assets/8dbd1076b821.png)

**高性能checkpointing的关键是模型文件切分和保存流程的优化**

从论文中的信息看，3FS 主要提供高性能 IO，单节点吞吐可达到 10GB/s；同时 CKPT 的写入频率达到了5分钟。高效的 CKPT 保存主要依赖以下几个方面：

• 参数和优化器文件被切分为多个小的 chunk，减少了每个 rank 的写出数据量，以及 3D 并行下 DP Group 额外的数据合并开销；同时可以使用 batch API 和并行化等手段，最大程度地利用存储写入吞吐。

• 参数和优化器文件会从 GPU 异步的传输到 CPU ，减少对训练的阻塞。

• 数据已经同步到 CPU 后，后续的计算（index 等元数据处理）也可以异步化，进一步减少对训练的阻塞。

## 总结

### 并行文件文件系统

3FS 是一个面向 AI 特定场景的、读优化的存储系统。并和训练框架、网络、数据格式等一起做了很多细致的co-design工作；同时其也是一个面向写少读多场景优化的存储系统。

**亮点**：

1. 数据链路基于 InfiniBand 或者 RDMA 做到 zero-copy，读吞吐和时延极好。

2. 元数据基于数据库实现可以支撑超大规模，个人判断 200 亿问题不大。

3. 支持文件并发写、覆盖写，某些场景这些能力有意义降低使用负担。

### AI Co-Design

回顾团队支持大模型训练的经验来看，传统的高性能并行文件系统只关注训练数据集、模型加载和 CKPT 写入这两个阶段的 IO 性能，但在复杂和大规模的训练场景下，瓶颈往往不在训练这两个末端阶段。通常存储需要和训练框架、网络一起 co-design 才能最大限度的发挥大模型存储的优势和价值。因此 DeepSeek 最大的优势还是各 AI 系统的有机组合，以实现极高的训练效率。

接下来的文章会继续分析 3FS 这个优秀项目更多源码细节。