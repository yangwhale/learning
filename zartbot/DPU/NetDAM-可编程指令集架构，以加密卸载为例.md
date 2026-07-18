# NetDAM:可编程指令集架构，以加密卸载为例

> 作者: zartbot  
> 日期: 2021年10月15日 16:00  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486684&idx=1&sn=204f2cd884317a21bfaaa4d021f27b9d&chksm=f9961c1ecee1950837175c56dcc5095a7bfca5996174c997e70e7fd6fd2c97b9a39e20b24ff5#rd

---

如果单纯的从把以太网和内存结合起来，其实也有很多类似的工作了。例如我们过去十多年一直在做的思科QuantumFlow处理器平台就是这样，利用独立的网络I/O内存GPM来隔离流表等存储域，同时针对LLC的优化及配置相应的调度器，以及多颗芯片实现channel interleaving等，具体公开的内容可以参考2008年的HotChip

同样的工作有朋友也指出了2019年的NetDIMM, 讲到的我们2004年做QFP的时候就解决了，工业界有太多的秘密不愿意说的也不能说的。我们用片内SRAM所以布线很容易，而NetDIMM的问题在布线上, 一块芯片要接两个大IO的内存并行总线加上网络的信号线，成本会非常高, NUMA的问题也不好解决. 另一方面新的Intel和AMD处理器已经4000多根pin了，而内存、UPI和I/O都要抢带宽，有些时候并不是樱桃不给啥留带宽，这个问题还涉及到片上网络的问题和Cache放置的问题，以后有空慢慢讲。现阶段唯一的办法就是利用并行转串行的方式，通过在CPU上使用Serdes降低布线难度和提升I/O和复用I/O，这也是CXL、CCIX、OpenCAPI出现的原因，但是串行链路将产生更高的延迟，而延迟的坑一定要填，填坑的做法就是存内计算，而存内计算的计算指令以何种方法呈现到用户态，能够更加方便的让软件调用，架构设计要考虑的东西多着呢，从软到硬都有微妙的平衡。

下面要来看的是软件的取舍，如何让处理器内的虚拟化和容器的用户态访问外部协处理器和资源并降低主存消耗。

NetDAM架构最大的不同是，从软件和硬件可编程的友好性出发，针对容器或者虚机内用户态需要调用DPU资源的情况，例如数据搜索的正则表达式加速、数据的压缩解压缩、加密解密等，这些东西对于用户态最容易的实现就是写一段类似于“指令”的结构体,然后要么发送到UDP socket，要么直接到NetDAM的request queue去.

```
struct netDAM_instruction {  uint8_t inst;  uint16_t seq;  uint64_t addr;}
```

另一个现状是，最近的一些研究出现了专门的CIM、PIM的内存芯片,例如Samsung最早在ISSCC上叫Function-In-Memory,后来在HotChip上改名叫Processing-In-Mem,

![图片](assets/3d83b9566a3b.png)

本质上就是把一个小型的ALU通过HBM堆叠封装夹在基板和HBM存储之间，并可以有额外的命令总线输入命令，基本的指令都齐全了，但是如何控制他们需要一个用户态的接口来承载。

![图片](assets/32bf9dc05909.png)

内存芯片本身能够接收一些指令集了， 那么我们这里又采用了类似于RISC-V的处理方式，NetDAM只保留部分内存`READ`/`WRITE`/`ATOMIC`的操作作为基本子集，然后用户可以定义其它的指令，或者简单来说就是一个RPC丢给NetDAM处理就好，这样的处理方式对用户态和proggraming language agnostic是很有好处的，例如Node.JS和Python一类的语言都可以很好的调度DPU资源做加速处理。

而NetDIMM的问题就是和当年OPA没有太大的区别，直接A怼到B上面去，没有从体系结构上想一些微妙的平衡。而内存本来就是并行总线，布线困难的基础上还要再加上网络相关的信号线，特别是有多网口的情况下，对于主板来说是非常困难的...但是OpenCAPI的内存接口生态又不是很好，只有IBM自己玩，不知道AMD跟进以后会不会好一点，但是大哥樱桃在搞CXL啊，为了生态不得不低头。体系架构的创新需要兼顾的东西太多~

### 什么需要Offload？

其实过去几年我们自己也在讨论，很多东西可以offload到可编程交换机上，那么什么值得offload，什么值得自己做，什么能够方便Offload？最近在给甲方爸爸开发一个IPSec的东西，加解密让CPU做肯定不切实际，同样的计算密集型应用还有压缩、搜索、Hash等... 而很多DPU的场景都提供了相应的卸载引擎，如何从软件层面更容易地去调度DPU上的这些加速器呢？而且每次Offload都会带来额外的内存拷贝，得不偿失，而且offload回来的200ns以上的时延，这些时间还不如CPU自己多花几个cycle做了，以QAT卡加速为例：

![图片](assets/40da19707c65.png)

首先报文DMA到主存后,然后CPU读出来,解析报文SPI和获取IV,这个时候通常由于UDP-IPSec和IP-IPSec需要判断不同的offset,然后去查询SA数据库,例如DPDK是这样干的:

```
struct rte_ipsec_sadv4_key {    uint32_t spi;    uint32_t dip;    uint32_t sip;};int i;union rte_ipsec_sad_key keys[BURST_SZ];const union rte_ipsec_sad_key *keys_p[BURST_SZ];void *vals[BURST_SZ];for (i = 0; i < BURST_SZ_MAX; i++) {    keys[i].v4.spi = esp_hdr[i]->spi;    keys[i].v4.dip = ipv4_hdr[i]->dst_addr;    keys[i].v4.sip = ipv4_hdr[i]->src_addr;    keys_p[i] = &keys[i];}rte_ipsec_sad_lookup(sad, keys_p, vals, BURST_SZ);
```

完了以后，要根据查询到的sa信息，构建解密的Crypto Operation,如果加密设备里面有session表，则放置session信息，否则需要告知解密算法`xform`，然后还要根据不同的加解密算法告知解密字段的起始位置和长度，以及Digest的内容

```
struct rte_crypto_sym_op {    struct rte_mbuf *m_src;    struct rte_mbuf *m_dst;    union {        struct rte_cryptodev_sym_session *session;        /**< Handle for the initialised session context */        struct rte_crypto_sym_xform *xform;        /**< Session-less API Crypto operation parameters */    };    union {        struct {            struct {                uint32_t offset;                uint32_t length;            } data; /**< Data offsets and length for AEAD */            struct {                uint8_t *data;                rte_iova_t phys_addr;            } digest; /**< Digest parameters */            struct {                uint8_t *data;                rte_iova_t phys_addr;            } aad;            /**< Additional authentication parameters */        } aead;...        };    };};
```

而这一系列的操作带来的时延，可能对于一些小于128B的报文根本就不值得，直接CPU解密了就好，这也是DPDK后来要自己弄一个crypto scheduler的原因。

### NetDAM Offload框架

而不同的厂商有不同的APi，看着DPDK里面Crypto Device Driver或者REGEX Device Driver就烦，经常要搞好多复杂东西.`通用的API`并且能让`userspace`使用的API就是我们在NetDAM设计`可扩展指令集`的初衷， 因为原有的设计在整个报文的处理过程中有多次的主存读写操作，而对于CPU真正重要的或许只是解密后的明文，当然有很多芯片都开始支持in-line crypto了，但是需要您记住的是in-line crypto通常会在加密芯片内放置一个很小的SRAM保存大概1000~10k个session， 像Intel QAT这样的芯片居然连session-less都不支持，这样面对互联网SideCar的一些场景，需要Offload数百万个TLS会话或者IPSec会话就麻烦了，为什么要这么大规模，TLS1.3 带来的影响了解下

另一个问题是，加解密协议非常繁多，不同的算法有不同的报文格式，不同的协议也有不同的session-key位置，全部固化在硬件电路里没有太大的必要，因此通用的做法是一开始可能需要根据报文的不同字段，拿一些bytes来看，或者通过主CPU下放一些SIMD的并行parser逻辑(类似于VPP那样向量化的同时处理32个packet)，然后NetDAM执行类似于P4那样的decode，并返回PHV给CPU，然后CPU去查询,或者P4查询NetDAM自己的DRAM都行，如下图所示：

![图片](assets/4a8ac4873917.png)

```
struct netDAM_instruction ndam_parser;ndam_parser.instr = PKT_PARSER;ndam_parser.data = ipsec.p4;ndam_device_enqueue_burst(ndam_parser,num_pkts);...ndam_device_dequue_burst(pkts[]);ipsec_sa_lookup(pkts[],sa_info[]);...ndam_prepare_cop(ndam_cop[],sa_info[],pkts[]);ndam_device_enqueue_burst(ndam_cop[]);...ndam_device_dequeue_burst(clear_text_pkts[]);do_packet_processing();
```

于是所有的解密操作的内存读写都在一个单独的I/O相关的NetDAM内存解决了。本质上是一个CSP模型，利用通信的方式来共享内存，并且隐藏延迟，可以注意到这个时候的CPU可以是主CPU或者DPU上的ARM处理器，甚至是一个虚拟机里的用户态进程,或者是一个远程的普通Host的UDP Socket带来的RPC, 本质上都是对NetDAM DRAM的访问控制指令，所以明白ALU在其中的作用了吧：）看下图就会更明白，三个方向都可以注入指令。

![图片](assets/84f4163d35b7.png)

加解密只是其中一个例子，您还可以换成Hash，这样对于很多区块链应用就可以加速了，或者换成压缩指令对存储就可以加速了，或者换成正则表达式，对安全和数据库业务就可以加速了，而神奇的是所有的加速指令都可以在虚拟化的环境中通过应用Userspace来调用，对于很多公有云来说，这是一个增值业务点，毕竟大量的应用本身需要加解密的支持，而基于VFIO的crypto有各种各样自身硬件的限制，虚拟化数量、QueuePair数量、硬件API的问题、VFIO数量，资源分配的问题...

统一抽象成MemoryAcces并开放给Userspace不更好么？后面大家是想听AI-MemoryPool应用呢，还是想听公有云的Serverless应用呢？当然都要咯~ 一天发一个~

预告：先发内存池化吧，很多AI的哥们等着用呢...

![图片](assets/38caabeefccd.png)

![图片](assets/a64ad60afa0d.png)