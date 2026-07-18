# AIOps实战2.1 Log4j漏洞后的异常主机探测

> 作者: zartbot  
> 日期: 2021年12月23日 08:32  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487271&idx=1&sn=ba3a05f6779b7ac58003810b89fb4c1e&chksm=f9961fe5cee196f36007531fc44f297ab391eb8e080ad8fa63011d546498b7117b5bb3f12753#rd

---

Log4j漏洞发生后，虽然很多客户都逐渐打了相应的补丁，但是大家还继续处于恐慌之中，有点类似于打了新冠疫苗，但是还不清楚自己是不是密接，或者是否处于被感染的潜伏期，而且有些客户又因为年终各种流程上的难处，于是渣给大家带来两种检测方法。一种是基于Netflow的全网关联的处理方法，即思科Secure Network Analytics方案(以前的Stealthwatch)

而本文的重点是基于核心交换机直接镜像. 然后使用渣为大家提供的一个基于DPDK的布隆过滤器`zbf` (性能`60Mpps`基本上满足很多客户大型数据中心骨干的镜像检测）和一些开源基于python的数据分析工具.

代码在:** github.com/zartbot/zbf**
木马主机特征
任何一种病毒如果需要传播都需要广泛接触其它易感群体，计算机病毒和木马也不例外。将任何一台主机和其它主机之间的通信记录下来便成了在严重漏洞爆发后的常见`流调`机制.但是在一个超大规模的数据中心内，伴随着数千台应用服务器又该如何实施呢？简单的使用镜像数据包需要大量的存储肯定不行，而通过Netflow一类的记录流日志算是一种比较可行的做法，如下图所示，某个数据中心在漏洞产生这段时间内有80亿条数据，而通常网络运维的团队也不可能有很多的服务器通过大数据集群来处理这些日志：

![图片](assets/e9a3dd945708.png)

Secure Network Analtyics
前面介绍过一些关于Netflow的概念，即把网络中的数据包通过流进行聚合，然后产生日志。下图是渣通过github.com/zartbot/goflow 的netflow解析器采集并送入ElasticSearch的，可以观察到从11.30日起，整个数据中心流日志就开始迅速增加了

![图片](assets/9199365fd15f.png)

当然思科有更加成熟的商业解决方案，以前叫作StealthWatch，现在改名为Secure Network Analtyics(SNA),它可以通过关联多个数据源实现端到端的安全诊断：

![图片](assets/6b4061b56c1b.png)

通过对全路径，包括终端设备的用户id、进程名称到云端VPC流表日志， 都可以通过netflow日志导出，SNA可以将其进行整个路径的关联分析，并通过一些AI模型来检测异常行为：

![图片](assets/6ed145c2249d.png)

具体的算法涉及一些商业机密就不多谈了，可以联系您的销售代表。

DPDK Based BloomFilter
而今天的重点是这几天给客户做的一个基于DPDK的布隆过滤器，毕竟年末了各个公司预算可能用的差不多了，又摊上这么大一个事情心里又不安，然后通常网络也因为国产化和异构的原因支持Netflow有限，那么最简单的处理方式，交换机镜像，然后通过DPDK来分析一下~

另外一个原因是，很多交换机采用1000：1以上的采样处理，而很多病毒采用低剂量几个小时扫描几个主机的低频方式来躲过检测，因此需要一个长期的通信基线监控来避免这样的潜伏期漏检

`布隆过滤器`（英语：Bloom Filter）是1970年由布隆提出的。它实际上是一个很长的二进制向量和一系列随机映射函数。布隆过滤器可以用于检索一个元素是否在一个集合中。它的优点是`空间效率`和`查询时间`都远远超过一般的算法，缺点是有一定的`误识别率`和`删除困难`。

本质上我们需要应对数据中心每秒数千万个数据包，然后把访问过的源目的地址记录下来， 而在这种场景下使用BloomFilter就非常高效了，zbf基于DPDK实现， 软件架构如下：

![图片](assets/3bd13ecd2105.png)

首先报文进入后采用了RSS分配到多核，BloomFilter最大的问题在于冷启动时，所有的flow都是新的需要添加，所以将报文解析(Decode)提取源目的地址和BloomFilter查询添加的进程分开了。具体软件架构设计的原理参见最后一节。下面我们来看如何使用

```
git clone https://github.com/zartbot/zbfcd zbf/dpdkmake clean;make#telemetry_reciever已经有一个编译好的版本，如果可以执行就不用编译了，因为编译可能需要您下载geoip的库cd ../telemetry_recievergo build
```

使用时可以支持如下参数:

```
zbf [EAL options] -- <Parameters> -f --first_lcore         First lcore used for forwarding thread -n --core_num            Number of lcore used for forwarding -r --export_rate         Number of flow info export per millisecond. -d --dip                 Exporter Destination IP address. -D --dport               Exporter Destination UDP port.
```

例如使用8个service-core-pairs(8 decode, 8 bloomfilter)输出速率为每毫秒100个UDP包，输出到192.168.99.101：12345的接收器

```
sudo ./zbf/dpdk/build/zbf -a 0000:0e:00.1 -- --first_lcore 24 --core_num 8  --export_rate 100 --dip 192.168.99.101 --dport 12345
```

遥测数据接收的Golang程序您可以为自己添加逻辑， 数据库源文件由于版权原因没有添加，您可以通过如下地址下载：https://dev.maxmind.com/geoip/geolite2-free-geolocation-data
下载后保存为city.mmdb和asn.mmdb,目录结构如下所示:

```
zartbot@zartbotWS:/opt/ruta/zbf/telemetry_reciever$ tree.├── asn.mmdb├── city.mmdb├── geoipmap│   └── geoipmap.go├── main.go└── telemetry_reciever
```

执行telemetry_reciever 就会根据dpdk进程export的信息打印在终端上,格式为源IP、源国家、源ASN、目的IP、目的国家、目的ASN

```
./telemetry_reciever > ../log_analysis/aaa.csv10.75.x.z|Local Private Network|0::LAN|131.6.116.170|United States|385::AFCONC-BLOCK1-AS
```

然后这些数据你可以自己添加到elasticsearch中，也可以像我这样就直接当成csv在python里面处理。
数据处理
由于处理结果都在log_analysis/aaa.csv中，我们可以打开一个jupyter-notebook来分析

```
import pandas as pdimport numpy as npimport matplotlibimport matplotlib.pyplot as plt%matplotlib inline  from IPython.core.display import display, HTMLdisplay(HTML("<style>.container { width:100% !important; }</style>"))
```

通过pandas读取数据

```
df = pd.read_csv("aaa.csv",delimiter="|",names=['src','src_country','src_as','dst','dst_country','dst_as'])
```

构建国家和运营商词典

```
country_dict = dict(zip(df['src'],df['src_country']))dst_country_dict = dict(zip(df['dst'],df['dst_country']))country_dict.update(dst_country_dict)as_dict = dict(zip(df['src'],df['src_as']))dst_as_dict = dict(zip(df['dst'],df['dst_as']))as_dict.update(dst_as_dict)
```

利用network-x构建图

```
import networkx as nxG = nx.Graph()for index, row in df.iterrows():    G.add_edge(row['src'],row['dst'])
```

查看度分布就可以找到一些奇怪的主机了

```
sorted(G.degree, key=lambda x: x[1], reverse=True)[('10.74.x.238', 156664), ('10.75.a.251', 155209), ('10.7f.b.228', 154932), ('10.1f4.cc.161', 154819), ('10.124.dd.199', 154703), ('10.75.1ee.232', 154116), ('10.75.1aa.125', 141842), ('10.7f.2c.155', 515), ('10.74.4c.164', 239), ('10.7a.58.242', 222), ('10.7e.28.190', 202), ('10.1f4.22.22', 199),
```

通常在这个排行榜中，除非是真的负责公有云互联网接入的服务器节点，一般内部节点度数量超过DNS服务器的度数量的都可以判定为异常，然后我们对异常节点进行统计

```
df[df['src']=='10.74.x.238'].groupby([df['dst_country']]).count().sort_values('src',ascending=False).head(20)
```

可以大致检查一下国家分布，看看是不是有一些您公司尚未开展业务的国家或者一些容易产生攻击源的国家

![图片](assets/a4ba59a4e5b3.png)

当然牵扯到图了，就有很多图算法可以使用了，例如计算介数中心性，计算特征根等计算方法，这个以后慢慢谈.

```
cr = nx.betweenness_centrality(G)
```

最后还准备了一个小彩蛋， 利用BFS产生某个server的通信拓扑

```
./layout.py <serverip> <depth> > result.jsonpython3 -m http.server
```

然后访问http 8000端口，可以看到一个3d.html点开，它就会自动根据result.json 渲染一个3D拓扑了,当然您也可以改为DFS去找路径或者其它方式渲染，这个对微服务非常有用，您可以通过对比不同时间基线看到异常行为。

![图片](assets/050d1a7501d1.png)

zbf原理
Decode函数也非常简单，Batch收到报文后，直接提取源目的IP, 当然只考虑了IPv4和带Vlan的情况，vxlan的数据中心大家可以稍微加一点代码也可以解析到overlay层，为了节省内存，处理完了以后直接就把原始数据包free了，只保留flow_key

```
static inline voidfetch_flowkey(struct rte_mbuf *pkt, struct flow_key *key){    struct rte_ether_hdr *eth_hdr;    struct rte_ipv4_hdr *ipv4_hdr;    eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);    key->ip_src = 0;    key->ip_dst = 0;    if (likely(eth_hdr->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)))    {        ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));        key->ip_src = rte_be_to_cpu_32(ipv4_hdr->src_addr);        key->ip_dst = rte_be_to_cpu_32(ipv4_hdr->dst_addr);        rte_pktmbuf_free(pkt);        return;    }    if (likely(eth_hdr->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN)))    {        struct rte_vlan_hdr *vlan_hdr;        vlan_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_vlan_hdr *, sizeof(struct rte_ether_hdr));        //printf("vlan id : %d\n", rte_be_to_cpu_16(vlan_hdr->vlan_tci));        if (likely(vlan_hdr->eth_proto == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)))        {            ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr) + sizeof(struct rte_vlan_hdr));            key->ip_src = rte_be_to_cpu_32(ipv4_hdr->src_addr);            key->ip_dst = rte_be_to_cpu_32(ipv4_hdr->dst_addr);            rte_pktmbuf_free(pkt);            return;        }        rte_pktmbuf_free(pkt);        return;    }}
```

然后数据自然需要prefetch，并且将最终整个batch的结果构成一个结构体送到下游Bloomfilter

```
int lcore_decode(struct decode_lcore_params *p){    printf("Core %u doing packet RX.\n", rte_lcore_id());    while (1)    {        struct rte_mbuf *pkts[BURST_SIZE];        const uint16_t nb_rx = rte_eth_rx_burst(ETH_PORT_ID, p->rx_qid, pkts,                                                BURST_SIZE);        if (unlikely(nb_rx == 0))        {            continue;        }        struct telemetry *t = malloc(sizeof(struct telemetry));        t->num = nb_rx;        int i;        /* Prefetch first packets */        for (i = 0; i < PREFETCH_OFFSET && i < nb_rx; i++)        {            rte_prefetch0(rte_pktmbuf_mtod(pkts[i], void *));        }        for (i = 0; i < (nb_rx - PREFETCH_OFFSET); i++)        {            rte_prefetch0(rte_pktmbuf_mtod(pkts[i + PREFETCH_OFFSET], void *));            fetch_flowkey(pkts[i], &t->key[i]);        }        /* Process left packets */        for (; i < nb_rx; i++)        {            fetch_flowkey(pkts[i], &t->key[i]);        }        rte_ring_sp_enqueue(p->telemetry_ring, (void *)t);    }    return 0;}
```

Bloom过滤器的初始化在main函数中:

```
    struct rte_member_setsum *setsum_vbf;    static struct rte_member_parameters params = {        .num_keys = MAX_ENTRIES, /* Total hash table entries. */        .key_len = 8,            /* Length of hash key. */        /* num_set and false_positive_rate only relevant to vBF */        .num_set = 32,        .false_positive_rate = 0.001,        .prim_hash_seed = 1,        .sec_hash_seed = 11,        .socket_id = 1 /* NUMA Socket ID for memory. */    };    params.name = "vbf_name";    params.type = RTE_MEMBER_TYPE_VBF;        setsum_vbf = rte_member_create(&params);    if (setsum_vbf == NULL)    {        printf("Creation of setsum_vbf fail\n");        return -1;    }
```

然后BloomFilter从上游decoder消费telemtry结构体，内含若干对flow pair，查找和找不到添加并导出日志的逻辑也非常简单

```
int lcore_service(struct service_lcore_params *p){    uint16_t ret_vbf,set_vbf;    printf("Core %u doing bloom filter checking.\n", rte_lcore_id());    while (1)    {        struct telemetry *t[BURST_SIZE];        const uint16_t nb_rx = rte_ring_dequeue_burst(p->telemetry_ring,                                                      (void *)t, BURST_SIZE, NULL);        if (unlikely(nb_rx == 0))        {            continue;        }        /* check flow*/        for (int i = 0; i < nb_rx; i++)        {            uint16_t *sets = (uint16_t *)malloc(sizeof(uint16_t) * t[i]->num);            for (int j = 0; j < t[i]->num; j++)            {                ret_vbf = rte_member_lookup(p->setsum, &t[i]->key[j], &set_vbf);                if (unlikely(set_vbf == 0 || ret_vbf == 0))                {                    rte_member_add(p->setsum, t[i]->key, p->rx_qid + 1);                    struct flow_info *f = malloc(sizeof(struct flow_info));                    f->ip_src = rte_cpu_to_be_32(t[i]->key[j].ip_src);                    f->ip_dst = rte_cpu_to_be_32(t[i]->key[j].ip_dst);                    rte_ring_enqueue(p->export_ring,(void *)f);                }            }            free(t[i]);        }    }    return 0;}
```

大家可以看到第一级decode完到第二级做bloomfilter的时候， packet per seconds就下降为原来的1/BURST_SIZE(32),然后BloomFilter过滤后再输出到Exporter时，初期几乎每个都要输出，因此ring buffer size开的特别大，主要是应对刚开始冷启动的时候， 而exporter考虑到golang的UDP接收能力和整个工作过程中就冷启动时有大峰值的情况，因此写了一个最简单的token based 流控就能搞定。另外为什么解析过滤用C、DPDK来写，而采集分析用Go呢？由于C的geoip还需要装相应的lib库比较麻烦，而且我们通常需要多点采集汇总分析的处理方式，所以中间对于export用了一个udp socket。

至于Go的代码，封装好了一个geoip的库，主要是处理掉港澳台被放入Country一栏的问题，和整理好结构，然后主程序非常简单的一个udp接收和parse

```
func main() {        src := "0.0.0.0:12345"        listener, err := net.ListenPacket("udp", src)        if err != nil {                logrus.Fatal(err)        }        defer listener.Close()        g, err := geoipmap.NewGeoIPCollector("./city.mmdb", "./asn.mmdb", 31.123, 111.11)        if err != nil {                logrus.Fatal(err)        }        for {                buf := make([]byte, 2048)                n, addr, err := listener.ReadFrom(buf)                if err != nil {                        continue                }                go serve(g, addr, buf[:n])        }}func serve(g *geoipmap.GeoIPCollector, addr net.Addr, buf []byte) {        for i := 0; i < len(buf); i += 8 {                src := net.IP(buf[i : i+4])                dst := net.IP(buf[i+4 : i+8])                s := g.Lookup(src)                d := g.Lookup(dst)                fmt.Printf("%s|%s|%s|%s|%s|%s\n", src, s.Country, s.ASN, dst, d.Country, d.ASN)        }}
```