# 谈谈字节的veRoCE, 但为啥不叫VeiWARP呢?

> 作者: zartbot  
> 日期: 2025年12月21日 10:29  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247496944&idx=1&sn=1d1a0859bc50a49bdff6a85f54c0b4a4&chksm=f995e432cee26d242e7204a358a6a13b7e1ee02325554c0e75a7ca3d7303ea0dea009f6c502b#rd

---

`本文仅代表个人观点, 与作者所任职的机构无关`

### TL;DR

前几天字节发布了自研的veRoCE协议, 仔细看了一下Spec, 相对于RoCE放弃了Lossless, 支持了基于SACK的重传, 然后为了多路径支持了iWARP DDP, 并且把iWARP的DDP Header中的MSN / Offset 通过MSNETH / POETH实现. 开个玩笑, 要是在加上Window Based CC那不就成了veiWarp了么? 言归正传, veRoCE这项工作本身是很不错的, 这是对RoCE现代化的一个很好的改造. 当然对于这些改良以前几年的文章已经分析的很透彻了, 并且我们已经线上运行好多年了.

例如在以前的一个文章中[《RDMA这十年的反思1：从协议演进的视角》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489240&idx=1&sn=53c7512d8551a44834bd405fd38b15dd&scene=21#wechat_redirect)已经很清楚的阐述了

最早谈到多路径能力居然是在2007年的一篇论文《Analyzing the Impact of Supporting Out-of-Order Communication on In-order Performance with iWARP》文章开篇就写到：Due to the growing need to tolerate network faults and congestion in high-end computing systems,supporting multiple network communication paths is becoming increasingly important. 也就是当今AI训练网络中非常重要的拥塞和链路失效的问题。解决办法就是采用Weak Ordering的Direct Data Placement实现.

20年前就已经有答案的问题, 却出现在当今的RDMA现代化改造似乎有些讽刺的意味, 也就是我一直在讲的, **RoCE这十年一开始就走错了...** 事实上正确的选择如下:

![图片](assets/6d05f776ccc0.png)

对于RoCE的缺陷, 字节的PR[《火山引擎 Force 大会发布 veRoCE 传输协议！》](https://mp.weixin.qq.com/s?__biz=MzkwNTIwNzc3OQ==&mid=2247494639&idx=1&sn=42bfedbb009944e56b8e987dfdf3463c&scene=21#wechat_redirect)中提到

在典型测试场景中，veRoCE 为大模型训练带来显著收益：LLM 训练速度相较于 RoCEv2 提升约 11.2%；AlltoAll 通信吞吐提升约 48.4%；在 2% 丢包率下，veRoCE 的有效吞吐仍能达到网卡带宽的约 95.7% 左右，而 RoCEv2 在这一场景下因为丢包过多而通信中断。

也就是说以往的字节高网因为使用传统RoCEv2进行LLM训练, 由于RoCE的缺陷使得整体算力浪费了11.2%? MoE这些需要的AlltoAll性能只有67%的带宽利用率? 另外工业界对于Lossy的测试, 无论是Google Falcon或者是CIPU eRDMA都遵循极端的5%丢包率下实现90% Goodput的能力, 仅2%丢包率的测试是不足的, 并且即便是2%丢包率, 重传2%带宽占用, 理论上起码要到97%以上才算打满, 这里还有一些错误重传的损失没解决

下面来详细分析一下veRoCE的Spec

## 1. Overview

这一章开头就讲:

IB 和 RoCEv2 都需要一个无损网络才能有效运行, 这使得 RDMA 传输协议在处理丢包和乱序报文到达时效率低下. 此外, 无损网络架构容易出现拥塞扩散和死锁, 这严重限制了网络的规模. 随着 AI 网络将扩展到支持数十万, 甚至数百万的 GPU 或 AI 加速器, 这些限制变得更加突出. 在这样的规模下, 由多路径传输和有损网络引起的丢包和乱序到达是必须被解决的, 不可避免的"事实".

这是对的, 不知道还在坚持Lossless的那群人在想什么? “是男人就要硬刚Lossy”, 很朴素的一个观点就是传统以太网的设计哲学是"尽力而为"(Best-Effort), 可靠性由端点(如 TCP)保证. 很高兴看到字节在这上面的转变.

然后值得称赞的是维持了RDMA Verbs语义, 不像AWS SRD那样为了多路径搞的整个生态都不兼容.

另一个现代化改造是多路径的支持能力, 相对于NV和BRCM倾向于在交换机上做PacketSpray, veRoCE也选择支持了端侧多路径的能力. 最终通过“多路径传输, 乱序交付, 高效的丢包检测, 硬件友好的选择性重传, 以及灵活的拥塞控制等特性“实现了RDMA现代化改造.

其实我们仔细看一下RoCE和iWARP的区别, 抛开TCP和UDP不谈. 主要的区别是: Lossless vs Lossy, 字节已经做出了选择, 然后为了实现Lossy, 采用了SACK.  然后多路径转发的能力, 引入了iWARP的DDP. 实质上如果再加上滑动窗口的拥塞控制. 本质上和iWARP的区别只剩下UDP/TCP这个协议号了. 同时延迟测量也采用了Swift那样的方式.  当然除此之外还有一个很细微的区别就是PSN vs SeqNum, 当然这里面还有很多取舍, 考虑一下ScaleAcross的场景?

## 2. veRoCE的报文格式

Spec的第三章介绍了报文格式

### 2.1 BTH

报文格式和IB Spec一致的, 主要修改是增加了SACK/ACK_Rsp/SACK_Rsp. 然后还增加了RTT Probe和一个SlowPath的OpCode

![图片](assets/d80a6dafdad3.png)

### 2.2 DDP Header

为了支持DDP的功能, veRoCE增加了MSN Extended Transport Header (MSNETH)和Packet Offset Extended Transport Header (POETH).

其实以前也讲过, 在RoCE的报文格式中, 因为传输层要求严格保序, 报文设计上相对简单.就是一个简单的First/Middle/Last.

![图片](assets/c7d4ce940a87.png)

中间的报文并不携带操作远端内存的地址. 因此乱序后将无法知道远端该如何写入.  当然Nvidia也做了一个简单的处理, 是否能够通过拆分成多个独立的WRITE报文, 每个都携带地址, 这样就可以乱序发送了呢? 但是对于一个消息而言, 我们还需要通知对方是否完成了, 因此发送端还需要等前面独立拆分的WRITE报文都确认接收后, 再发送一个WRITE_With_Imm消息确认. 或者通过一个ATOMIC消息更新接受端的Fence flag, 这样实际上还是增加了一个Round-Trip-time. 这种做法对于WRITE可以实现, 但是对于SEND/RECV是有缺陷的.因为RECV端的buffer缓冲区的排布并不是绝对的物理地址. 所以在CX7中, Out-of-order的处理只支持WRITE语义.

当然网络这个圈子也有很懂这一块的人, 例如2002年发布的iWARP, 其中就提到了Direct Data Placement(DDP)的技术.

![图片](assets/a0272b8d2bd3.png)

DDP在每个数据分段中间添加了一个Message Sequence Number(MSN)和Message Offset(MO), 通过MSN和MO其实就非常容易的在多种语义上支持乱序接收, 并且可以在接受端判断消息是否完成, 然后完成Imm提交或者执行Fence flag更新, 这样相对于RoCE现有的实现还节省了一个Round-Trip-Time.

其实这里实现上还是有一些差异的, 字节采用的是PacketOrder(PO), 而不是DDP中的Offset, 因此地址计算DestAdd= BaseAddr + PO x Payload Size Per Pkt.

## 3. Reliable Connection服务

在设计新的协议时, 维持Reliable Connection服务接口的兼容性是非常值得称赞的. AWS EFA的问题就是在解决多路径转发时,放弃了对RC的兼容, 而采用了SRD的方式, 这样带来了大量的兼容性的问题.

![图片](assets/9a2a259b7b4e.png)

### 3.1 PSN

每个报文都在其 BTH 中携带一个 PSN 进行传输. PSN 用于识别丢失或乱序的报文, 并且对于可靠连接服务, 用于将一个确认报文关联到一个给定的语义报文.

然后veRoCE定义了两个PSN空间, 一个 QP 同时维护两套独立的 PSN 序列号:

SQ PSN 空间: 用于自己发起的 Send/Write/Read Request.

Response PSN 空间: 用于自己为收到的 Read/Atomic Request 生成的响应.

这个设计也命中了RoCE的一个缺陷. 标准的RoCE协议实际上是无法支持Window Based拥塞控制的.

数据包和ack报文单独发送,数据包里面没有携带ack信息

Read resp报文是用ack报文封装,实质是数据包,而read resp并没有对应的ack;

也就是说

由于响应报文不使用 PSN, 它们无法被接收端(即原始的 Requester)用常规的 ACK/NAK 机制来确认.

对 Read Request 报文的确认是隐式的, 通过 Read Response 报文中携带的 PSN 来实现.

对 Read Response 报文的可靠性保证几乎完全依赖于请求端的超时重传 (RTO).

如果采用window based拥塞控制,需要ack来驱动,read场景就会导致read resp直接发不出去了, 不改RoCE协议, 解法只能变成使用定时器加token/credit,又变成rate based了.

因此veRoCE定义了两个PSN空间, 理论上也可以解决这个问题.

然后我们来讨论一下PSN编码, 理论上PSN的长度很大(24bits), 但接收端 PSN bitmap长度限制了整个乱序区的大小, 如果一个报文的 PSN 等于或大于 aPSN + bitmap_length + 1, 接收端无法在其 PSN bitmap中记录该报文的到达, 因此该报文会被接收端丢弃.

这其实这里我们来看一个潜在的问题, 考虑ScaleAcross的场景, 例如跨AZ通常几十公里,假设延迟为500us, 硬件的bitmap为128bit, 那么实际上的inflight大小为8KB * 128 = 1MB, 则单个QP的最大理论带宽仅为16Gbps. 如果需要做到400Gbps, 那么需要3200bits... 另外在跨AZ不同光纤物理走向不同时, 还会出现不同路径延迟的差异, 处理起来更难.

相反, iWARP采用SeqNum表示可以更有效的表示, 具体怎么做就不说了.

### 3.2 MSN

MSN 将可靠性从报文(Packet)粒度提升到了消息粒度.aMSN 的前进是发送端硬件可以安全地生成 CQE, 通知应用操作完成的唯一可靠信号, 这样就可以很高效的实现Write-with-Imm的处理.

### 3.3 ACK

这部分是全章的重中之重, 定义了 veRoCE 的核心交互逻辑. ACK报文的格式描述如下:

![图片](assets/183378150575.png)

#### 3.3.1 ACK/SACK

对于每个Packet都需要ACK, 然后ACK也是可以聚合的, 即允许单个 ACK 作为对一个或多个先前语义报文的确认. 当接收端生成一个 ACK/SACK 时, 其 BTH.PSN 字段用 aPSN (确认 PSN)填充. 当发送端收到一个 ack_pkt 时, 它应将其 aPSN 前进到该 ack_pkt 的 BTH.PSN. 对于 Read, Atomic 和 Send-with-Invalidation, ACK 报文确认该报文已被收到.

一个常见的ACK流程如下:

![图片](assets/7d7ada79767a.png)

SACK 报文用于选择性地确认在接收端乱序到达的语义报文, SACK的聚合和具体实现相关. 考虑到SACK有128bit的bitmap, 可以使用一种Lazy SACK的处理方式, 不为每个乱序报文都生成一个 SACK, 而是当接收端的乱序度(OOOD) 超过一个阈值时, 接收端才发送一个 SACK. 这也意味着当 OOOD 低于阈值时, 接收端继续返回 ACK. OOOD 被定义为最高收到的 PSN (hPSN) 与 aPSN 之间的差值. 用于发送 SACK 报文的 OOOD 阈值可以被认为是多路径所引起的最大乱序程度.

SACK示例如下:

![图片](assets/6158a49735e8.png)

这里也引入了一个问题, 固定的OOOD阈值会带来一些复杂度, 特别是在基于RateBased流控机制下, 后面在拥塞控制的部分详细展开.

#### 3.3.2 NAK

veRoCE NAK 协议与 IB 规范保持一致. NAK 码 b'00000' (PSN 序列错误)被重用为Packet Drop NAK. 对于 Packet Drop NAK, NAK 报文应在错误被检测到后立即传输.

实际上在这里是针对交换机支持Packet Trimming(NDP)机制的一个响应, 这是一个显示的丢包响应信号.

#### 3.3.3 丢包检测

采用SACK和RTO两种方法:

SACK 报文: SACK 中的bitmap是哪个 PSN 收到哪个没收到的直接指示. 可以应用启发式方法来推断丢包. 使用"Lazy SACK"时, 发送端收到 SACK 就可以假定 aPSN + 1 已经丢失, 并随后启动对 aPSN + 1 的重传, 同时附带重传靠近它的、bitmap中指示为未收到的若干个 PSN.

传输定时器: 如果一个包在定时器超时(RTO)前未被确认, 将触发重传. 一旦 RTO 重传被触发, 发送端应停止由 SACK 触发的快速重传, 直到 aPSN+1 被确认为止.

#### 3.3.4 快速的选择性重传

连续的 SACK 报文可能有重叠的bitmap. 如果每个 SACK 报文都简单地触发对 [aPSN + 1, aPSN+N] 范围内未收到报文的重传, 将会有大量不必要的重传, 因为连续的 SACK 报文很可能包含重叠的bitmap. 下图提供了一个例子. 报文 2 被重传了两次, 因为两个 SACK 的bitmap都指示报文 2 尚未收到.

![图片](assets/14b5484c6876.png)

veRoCE 推荐一种快速选择性重传机制, 以缓解不必要的重传. 发送端或接收端维护一个变量 RxtPSN, 它记录了来自上一个 SACK bitmap的最高有效 PSN, 并且在每次收到或传输 SACK 时进行更新. 对于每个收到的或传输的新 SACK, 只有 PSN > RxtPSN 的bitmap条目才应被考虑用于选择性重传. 为了避免由丢失的重传报文引起的 RTO, 发送端或接收端还应记录 RxtPSN 最后一次更新的时间. 当 RxtPSN 在一段时间内没有更新时, RxtPSN 应被重置为 aPSN, 以允许进行第二次选择性重传. 该时间阈值的典型值为网络 RTT.

下图描绘了当 RxtPSN 在发送端维护时, 快速选择性重传是如何工作的.

![图片](assets/7284ea2cf03c.png)

注意 veRoCE 并不强制要求此设计, 因为 RxtPSN 也可以在接收端维护. 在发送端收到第一个 SACK 后, 它解析bitmap, 重传报文 [5, 6, 8] 并将 RxtPSN 更新为 8. 当第二个 SACK 到达时, 重传的报文 [5, 6, 8] 仍在网络中传输. 在 RxtPSN 的帮助下, 发送端只重传了 PSN 大于 RxtPSN 的报文 10 和 11.

如果重传的报文再次丢失, RxtPSN 将会长时间停滞, 导致 RxtPSN 被重置. 下图描绘了这个过程.

![图片](assets/c203b1dac0fd.png)

图中的第一个 SACK 触发了报文 [5, 6, 8] 的重传, 但报文 [5, 6] 再次丢失了. 第二个 SACK 只重传了报文 10, 11, 因为 RxtPSN 已经前进到 8. 当第三个 SACK 到达时, RxtPSN 已经停滞了足够长的时间, 从而被重置为 aPSN (即 4), 于是报文 [5, 6] 被第二次重传.

其实这一部分的异常处理实现还是有一些问题的, 尝试在纯硬件ASIC上构建SACK是一件非常难的事情, Google Falcon和Intel一起搞IPU重新流片了3~4次. 当然veRoCE这样的处理方式还是有一些缺陷的, 就不展开指出了.

### 3.4 WRITE和SEND实现

实际上是在解释DDP的工作原理, RDMA Write 操作的一个示例如下:

![图片](assets/22ca38d8ed38.png)

请求端向发送队列(Send Queue)提交一个 Write WQE. QP 根据消息大小和 MTU 创建了 4 个 write 请求报文 (PSN = 100-103).

请求端按顺序将报文传输给响应端. 报文可能被网络重排序. 例如, 报文 100 和报文 101 在响应端乱序到达. 响应端直接将数据放置到主机内存而无需对它们进行重排序.

响应端向请求端返回 ack_pkts 以确认 Write 请求的到达, 并且响应端可以执行 ACK 聚合以减少 ack_pkts 的数量. 例如, 最后 4 个报文的 ACK 被聚合了. 在该消息的最后一个散乱报文(straggler packet)被响应端处理后, 响应端回复一个 AETH.MSN 设置为 1 的 ACK. 一个 Write 消息的散乱报文可以通过响应端的 PSN bitmap来识别.

在收到带有新 MSN 值(即本例中的 1)的 ack_pkt 后, 请求端完成该 write 消息.

对于 Write-with-ImmDt 和 Send 消息, 由最后一个报文携带的立即数需要被复制到 CQE 中. 在乱序到达的情况下, 最后一个报文可能在 CQE 生成之前到达, 立即数需要被缓存在响应端.

![图片](assets/2fb3e1a9bee3.png)

在 MSN 为 2 的 Write 请求完全收到后, 响应端才为 MSN 为 3 的 Write-with-ImmDt 请求发布 CQE.

### 3.5 READ实现

Read 请求仅占用一个 PSN. 后续的请求会顺序使用下一个 PSN, 不会跳过任何 PSN. 在收到 Read 请求后, 响应端会缓冲它们, 直到所有在前的请求都被处理完毕. 对于一个 Read Response, 其 AETH.MSN 填充的是相应 Read Request 的 MSN.

RDMA Read 操作的一个示例如下:

![图片](assets/3b824393240d.png)

请求端向发送队列(Send Queue)提交一个 Read Request WQE, 随后一个 Read Request 报文被传输到响应端. 该 Read Request 仅占用一个 PSN.

在收到 Read Request 后, 响应端应向请求端回复一个 ACK (也受 ACK 聚合影响). 这个 ACK **不**表示 read 请求的完成, 而是确认 Read Request 已被响应端收到.

响应端缓冲该 read 请求. 当所有在前的请求都处理完毕后, 响应端开始为该 Read Request 生成一个 Read Response 消息.

响应端按照被缓冲的 Read Request 的指示发送出 Read Response 报文. 对于一个 Read Request, 响应端可能根据消息大小和 MTU 生成多个 Read Response 报文. 来自 Read Response 的报文和来自 SQ 的报文处于**独立的 PSN 空间**.

请求端收到 Read Response 报文, 并回复 ack_pkts (即 ACK_Rsp/SACK_Rsp) 以确认它们的到达. 当收到的 Read Response 报文填补了 PSN 范围中的一些"空洞"时, PSN 和 MSN 可能得以推进, 并通过确认报文返回给响应端.

响应端收到这些 ack_pkts. 当 ack_pkts 中的 MSN (指 Response MSN)相比上次收到的 MSN 有所前进时, 响应端**释放**相应数量的 Read Request.

### 3.6 ATOMIC实现

原子操作使请求端能够在响应端的一个指定地址上执行一个 64 位的操作. 这些操作原子性地读取, 修改和写入目标地址, 并保证对该地址的其他 QP 的操作不会在读取和写入之间发生. veRoCE 支持 IB 规范(章节 9.4.5)中定义的 FetchAdd 和 CmpSwap 原子操作. 原子操作请求仅占用一个 PSN. 在收到原子操作请求后, 响应端会**缓冲**它们, 直到所有在前的请求都被处理完毕. 响应端可以回复一个 ACK 或 SACK 来表明收到了原子操作请求. 然而, 这**不能**被用作请求端操作的完成信号. 原子操作只有在收到 **AtomicACK** 时才能被认为是完成的.

原子操作的一个示例如下:

![图片](assets/3d9631cf56c7.png)

请求端向发送队列提交一个原子操作请求 WQE, 此时还有一个在途的 Write 请求 WQE. 随后一个原子操作请求报文被传输到响应端. 该原子操作请求仅占用一个 PSN.

在收到原子操作请求后, 响应端应向请求端回复一个 ACK (也受 ACK 聚合影响). 这个 ACK **不**表示原子操作的完成, 而是确认原子操作请求已被响应端收到.

响应端缓冲该原子操作请求. 当所有在前的请求都处理完毕后, 响应端执行该原子操作, 并为该原子操作生成一个 **AtomicAck 消息**.

响应端发送出 AtomicAck 消息. 每个 AtomicAck 消息占用**一个 PSN** 和**一个 MSN**.

请求端收到 AtomicAck 报文, 并回复 `ACK_Rsp` 以确认它们的到达. 当收到的 AtomicAck 报文填补了 PSN 范围中的一些"空洞"时, PSN 和 MSN 可能得以推进, 并通过确认报文返回给响应端.

响应端收到这些 ack_pkts. 当 ack_pkts 中的 MSN 相比上次收到的 MSN 有所前进时, 响应端**释放**相应数量的原子操作请求.

我们要理解原子操作的本质. 像 `FetchAdd`或 `Compare-and-Swap`这样的操作, 都包含两个步骤:

**Read**: 从远端内存地址读取一个原始值.

**Write**: 根据读取到的值和一个输入值进行计算, 然后将新值写回同一个内存地址.

关键在于, 这两个步骤必须是**原子**的, 中间不能被其他任何操作打断.  如果 `AtomicAck` 丢失了怎么办?

在 RoCEv2 模型下, Requester 只能等待 RTO 超时, 然后重传整个 `Atomic Request`. 这会导致远端的原子操作被**错误地执行两次**, 完全违背了"原子性"的初衷.

在 veRoCE 模型下, 由于 `AtomicAck` 有自己的 PSN, 如果它丢失了, Requester 会发现 Response PSN 空间出现"空洞", 或者最终由 Requester 侧的定时器超时(注意, 这里是等待响应的定时器, 而非 RTO). Requester 就可以向 Responder 发送一个 `NAK_Rsp` 或者通过其他机制(如超时后发送一个查询状态的特殊包), 请求 Responder 仅重传那个丢失的 `AtomicAck`, 而**不是重新执行原子操作**. Responder 在执行完原子操作后, 应该缓存住结果(`AtomicAck` 的内容), 直到它被 Requester 确认为止.

### 3.7 Operation Order

实际上是在解释DDP的Weak Ordering. 相对于IB定义的Strong Ordering, 有一个修改: 一个 Send 或 RDMA Write 不会必然在其后续的 Send 或 RDMA Write 请求之前完成. 如果两个 Send 或 RDMA Write 写入相同的目标内存, 来自第二个操作的数据可能会被第一个操作的数据覆盖. 这个修改是为了加速 DDP 过程并避免在硬件中缓冲数据载荷.

## 4. 拥塞控制

### 4.1 拥塞通知

每个接收端 QP 维护一个或多个拥塞信号上下文(Congestion Signal Contexts, CSC)来记录拥塞相关信息. 有两种方法将拥塞信号回传给发送端:

**带内(Inband)**: ack_pkts 可以携带拥塞信号, 如 ECN (使用 BTH 头中的 BECN 字段), 这可以被基于窗口的拥塞控制算法利用.

**带外(Out-of-band)**: 拥塞信号也可以使用独立的 CNP (Congestion Notification Packets)返回给发送端, 这使得基于速率的拥塞控制算法成为可能.

CNP 的生成速率是实现相关的. 例如, 即使接收端 QP 要求 NIC 硬件为每个 ECN 标记的报文都生成一个 CNP, 硬件也可能以类似于 ACK 聚合的方式来聚合 CNP. 一旦一个 CNP 被发送, 拥塞信号上下文应被重置.

### 4.2 RTT 探测

RTT 是使用独立的 RTT 探测报文来测量的. 具体算法和Google Falcon和CIPU eRDMA是一样的, 一个 RTT 探测请求返回 4 个时间戳:

![图片](assets/89997bda1d30.png)

Tx timestamp 1: RTT 请求报文在发送端的发送时间. 该时间戳由请求中的 RTTReqETH 携带, 并由 RTT 响应中的 RTTRspETH 回显.

Rx timestamp 1: RTT 请求报文在接收端的接收时间. 该时间戳由 RTT 响应中的 RTTRspETH 携带.

Tx timestamp 2: RTT 响应报文在接收端的发送时间. 该时间戳由 RTT 响应中的 RTTRspETH 携带.

Rx timestamp 2: RTT 响应报文在发送端的接收时间. 该时间戳由发送端自己标注.

这 4 个时间戳被传递给 CC (拥塞控制)模块. 如何使用这些时间戳由 CC 算法决定. 三种可能的方式包括: a.  网络往返延迟可以由  计算得出. b.  主机处理延迟可以由  计算得出. c.  如果请求端和响应端时间同步, 网络单向延迟可以由  和  计算得出.

一个 RTT 请求的 UDP 源端口字段根据它正在探测的路径来设置, 而 RTT 响应的源端口与其对应的 RTT 请求保持相同.

其实这一点veRoCE还是做的有问题的, 在《谈谈Google Falcon的可靠传输论文并对比分析CIPU eRDMA》也写到过: **RoCE 没有将拥塞控制与数据路径集成.** 它的拥塞控制是作为一个附加组件实现的, 依赖于带外探测(out-of-band probes)来收集拥塞信号. 这种分离使其拥塞响应迟缓.

事实上像Google Falcon和CIPU eRDMA都采用带内的方式, 有更大的优势. 具体能解决什么问题涉密就不多说了.

### 4.3 拥塞控制模式

一个采用多路径的可靠连接, 其拥塞控制机制可以工作在两种模式下:  **连接级(connection-wise)** 和 **路径级(path-wise)**. 在任一模式下, 发送端 QP 维护一个或多个拥塞控制上下文(Congestion Control Context, CCC)来控制发送速率. 注意对于单路径的可靠连接, 无需区分连接级和路径级模式, 它们是相同的.

![图片](assets/798513d262bf.png)

其实这里还有很多技巧似乎字节的同学没有掌握到, 例如path-wise的方法会导致整个网卡的QP规模受限.

### 4.4 慢路径检测

随着时间的推移, 路径质量可能会下降(例如, 由于链路错误), 使一条路径变成"慢路径"(高延迟/丢包). 慢路径检测是一个可选但推荐的增强功能, 以提高网络效率. 发送端可以使用以下机制来识别并从慢路径迁移流量:

**大的 PSN 差异 (在接收端)**: 如果一个新语义报文的 PSN 与 hPSN (已收到的最高 PSN)之间的差异超过一个阈值, 那么该报文被认为是一个慢包. 接收端随后向发送端发送一个慢包信号(Slow-Packet Signal, 一个 Opcode 为 `bin10000100` 的报文). 如果发送端在一段时间内收到针对某条路径的多个慢包信号, 该路径就可能被标记为慢路径.

慢路径检测算法可以是实现相关的. 其他指标, 例如, 延迟的 ACK, RTT 探测/响应丢失, 也可以被用来识别慢路径.

事实上在一些ScaleAcross的场景, 物理路径上延迟就有很大的差异, 简单剔除慢路径将会导致高延迟路径实际带宽无法使用的问题.  其实多路径转发吧, 里面需要很多巧妙的设计, 但是抱歉涉密不能讲.

## 5. 总结

### 5.1 RDMA现代化目标分析

其实我一直给过一个RDMA现代化的目标:

集合通信能够保证95%以上的Fabric利用率

丢包率5%的时候仍然能够保证90%的Goodput

无需任何交换机的高级特性, 网卡实现多路径和拥塞控制

超大规模(128K QPs)并支持所有QP开启多路径转发能力.

兼容RDMA RC Verbs, 线下RDMA应用无需修改代码即可直接运行.

Incast 128打1这样的场景, 每个QP之间的带宽差额最大100Kbps.

完全OS无感知的热迁移

完善的RDMA虚拟化支持

总体来看veRoCE是一个不错的工作, 按照传统RoCE的based line性能, 集合通信在一个标准的FatTree网络中因为Hash冲突等原因, 实际利用率为60%左右. 理论上多路径会带来1.6x的性能提升, 即第一个目标的需求. 从目前字节的数据来看, 提升为1.48x, 也就是说实际上的Fabric利用率在89%左右, 应该还有几个点的提升空间.

第二个目标是5%丢包率下90%的Goodput, 这是对SACK实现的一个更严格的考核, 字节公开的数据是否可以补充5%丢包率的测试情况?

第三个目标是无需使用交换机任何高级特性, 这一点应该是veRoCE可以做到, 当然部分场景还是需要NDP和交换机增加额外的熵才能Hash. 另外多路径算法上仅是剔除慢路径也是一个问题.

第四个目标, 如果使用Path-wise CC是无法实现的, 或者会导致多路径算法的子路径变少.

第五个目标, 这是veRoCE做的好的, 坚持RC Verbs兼容才是正路

第六个目标, 总体来看字节的PR AlltoAll性能提升看上去还是不错的, 但是是否能够达到如此严苛的标准, 难说. 这一条对EP并行的AlltoAll是非常关键的.

最后两个, 属于实现上的问题...

### 5.2 Lossless or Lossy

选择Lossy, 这是veRoCE做的非常好的一点. 但实际上当你选择Lossy和支持多路径的时候, 就不可避免的将整个协议按照iWARP的方式改造, 对UDP传输做加法, 从TCP中学来SACK, 延迟测量来自于Swift.多路径又要不可避免的抄iWARP的DDP. 还记得某人莫名其妙的在几个月前某个会上大放厥词: iWARP已死, 结果实际上都要在iWARP那里抄作业.

实质上的问题就是, 早期RDMA主要用于HPC场景, 带宽需求很小, 相反延迟需求更高, 因此基本上不怎么丢包, 所以一个强Lossless假设和简单的可靠传输就可以解决问题. 但是到了AI时代这是完全不同的情况, 极大带宽的带宽需求要求必须放弃Lossless和Strong Order, 通过iWARP DDP这样的技术实现weak order和多路径转发充分利用带宽才能获得收益.

当然这件事本身是很难的, 对于NV(Mellanox)在已有的硬件架构上推导重来更是尴尬. 也就导致了他们持续性的坚持走Lossless的路.

### 5.3 多路径和拥塞控制

veRoCE还是受制于原有的一些硬件实现约束上, 和RoCE一样, 没有将拥塞控制与数据路径集成. 它的拥塞控制是作为一个附加组件实现的, 依赖于带外探测(out-of-band probes)来收集拥塞信号. 这种分离使其拥塞响应迟缓.

从ZTR-CC开始, Nvidia就在尝试用带外信号来探测拥塞, 但是信号处理相对缓慢, 所以拥塞控制并不好. 然后近期的CX8通过PSA以及Spectrum-X的把Telemetry处理能力提高了1000x, 但事实上更频繁的带外周期行probe不光和数据路径分离, 还会很多的资源.

至于多路径的算法上, 似乎规范上并没有公布太多, 但实现上还有大量的细节和技巧, 特别是多路径和拥塞控制的融合上, 涉密就不展开了.

### 5.4 结语

其实历史的选择就是这么简单, 但是裹挟了大量的市场营销和偏见.. 很高兴看到veRoCE选择了一条正确的路.

![图片](assets/7751268a9901.png)

其实要实现这些, 最大的难点还是在软硬件协同上. 很多协议设计者并没有处理高性能Stateful L4的经验和能力, 有一些号称xxxx的, 实际上并没有自己动手做过一块芯片, 自然有很多东西无法去考虑. 同理veRoCE也受制于其承载的卡的硬件提供商的限制, 协议上还有一些不完善的地方.

毕竟学术界和工业界, 特别是搞了十多年的Stateful L4处理的Service Routing的人比起来, 可能我们认为的常识性的芯片设计,对于学术界的人来看难上登天.

当然总体来说, veRoCE还是一个很不错的工作, RC兼容下实现Lossy和多路径基本上完成了RDMA现代化改造. 但还有一些路要走, 正确的选择如下:

![图片](assets/288f7677c856.png)

建议多读读下面这几篇文章吧:

[《谈谈Google Falcon的可靠传输论文并对比分析CIPU eRDMA》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247495848&idx=2&sn=e55764ca731533c76e55ab4cb0bf25d4&scene=21#wechat_redirect)

[《RDMA这十年的反思1：从协议演进的视角》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489240&idx=1&sn=53c7512d8551a44834bd405fd38b15dd&scene=21#wechat_redirect)

[《RDMA这十年的反思2：从应用和芯片架构的视角》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247489285&idx=1&sn=3a53f4d177aca0a2a052450fd1a58fe2&scene=21#wechat_redirect)

[《RDMA》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=3398249338911260673#wechat_redirect)