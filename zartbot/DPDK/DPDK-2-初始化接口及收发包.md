# DPDK-2:初始化接口及收发包

> 作者: zartbot  
> 日期: 2021年5月20日 11:32  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485852&idx=1&sn=7f3ee47d5c47d6afe97ac186eecb0eae&chksm=f996195ecee19048fb75576b721bae976b5239cd1956e4b5353dd760be343eacc08e43ac7185#rd

---

❝
第二篇包括基本的接口初始化及基本的收发包等内容
❞
### 端口配置 

CCNA教程通常也是从接口IP地址配置开始的，那么我们也从如何配置接口开始讲述.第一个程序很简单，我们看看DPDK支持多少个接口并把MAC地址存下来.

首先需要初始化EAL

```
  int ret = rte_eal_init(argc, argv);  if (ret < 0)      rte_exit(EXIT_FAILURE, "initlize fail!");
```

然后通过`rte_eth_dev_count_avail()`函数获取系统的接口数目:

```
  int nb_ports;  nb_ports = rte_eth_dev_count_avail();  printf("number of available port: %d\n", nb_ports);
```

然后我们可以通过`rte_eth_dev_info_get`如下方式获取device info：

```
    struct rte_eth_dev_info dev_info;    for (int portid = 0; portid < nb_ports; ++portid)    {        ret = rte_eth_dev_info_get(portid, &dev_info);        if (ret < 0)            rte_exit(EXIT_FAILURE, "Cannot get device info: err=%d, port=%d\n", ret, portid);        printf("port: %d Driver:%s\n", portid, dev_info.driver_name);    }
```

最后可以根据`rte_eth_macaddr_get`函数获取接口MAC地址,并放在ports_eth_addr数组中.

```
    static struct rte_ether_addr ports_eth_addr[MAX_PORTS];    for (int portid = 0; portid < nb_ports; ++portid)    {        ret = rte_eth_macaddr_get(portid, &ports_eth_addr[portid]);        if (ret < 0)            rte_exit(EXIT_FAILURE, "Cannot get MAC address: err=%d, port=%d\n", ret, portid);        char mac[18];        rte_ether_format_addr(&mac[0], 18, &ports_eth_addr[portid]);        printf("port: %d->MAC-> %s\n", portid, mac);    }
```

最后整个文件如下，我们将其保存为`main.c`

```
#include <stdint.h>#include <inttypes.h>#include <rte_eal.h>#include <rte_ethdev.h>#include <rte_cycles.h>#include <rte_lcore.h>#include <rte_mbuf.h>#include <rte_ether.h>#include <rte_ip.h>#include <rte_udp.h>#include <pthread.h>#include <string.h>#define MAX_PORTS 16int main(int argc, char *argv[]){    int ret = rte_eal_init(argc, argv);    if (ret < 0)        rte_exit(EXIT_FAILURE, "initlize fail!");    printf("\n\n\n*****************************************\n");    int nb_ports;    nb_ports = rte_eth_dev_count_avail();    printf("number of available port: %d\n", nb_ports);    struct rte_eth_dev_info dev_info;    for (int portid = 0; portid < nb_ports; ++portid)    {        ret = rte_eth_dev_info_get(portid, &dev_info);        if (ret < 0)            rte_exit(EXIT_FAILURE, "Cannot get device info: err=%d, port=%d\n", ret, portid);        printf("port: %d Driver:%s\n", portid, dev_info.driver_name);    }    /* ethernet addresses of ports */    static struct rte_ether_addr ports_eth_addr[MAX_PORTS];    for (int portid = 0; portid < nb_ports; ++portid)    {        ret = rte_eth_macaddr_get(portid, &ports_eth_addr[portid]);        if (ret < 0)            rte_exit(EXIT_FAILURE, "Cannot get MAC address: err=%d, port=%d\n", ret, portid);        char mac[18];        rte_ether_format_addr(&mac[0], 18, &ports_eth_addr[portid]);        printf("port: %d->MAC-> %s\n", portid, mac);    }    return 0;}
```

在同一个目录下建立一个`Makefile`的文件，这个文件可以在dpdk example中随便抄一个，不熟悉C编程的读者请注意它的缩进必须要用Tab.

```
# SPDX-License-Identifier: BSD-3-Clause# Copyright(c) 2010-2014 Intel Corporation# binary nameAPP = portinit# all source are stored in SRCS-ySRCS-y := main.c# Build using pkg-config variables if possibleifneq ($(shell pkg-config --exists libdpdk && echo 0),0)$(error "no installation of DPDK found")endifall: shared.PHONY: shared staticshared: build/$(APP)-shared ln -sf $(APP)-shared build/$(APP)static: build/$(APP)-static ln -sf $(APP)-static build/$(APP)PKGCONF ?= pkg-configPC_FILE := $(shell $(PKGCONF) --path libdpdk 2>/dev/null)CFLAGS += -O3 $(shell $(PKGCONF) --cflags libdpdk)LDFLAGS_SHARED = $(shell $(PKGCONF) --libs libdpdk)LDFLAGS_STATIC = $(shell $(PKGCONF) --static --libs libdpdk)ifeq ($(MAKECMDGOALS),static)# check for broken pkg-configifeq ($(shell echo $(LDFLAGS_STATIC) | grep 'whole-archive.*l:lib.*no-whole-archive'),)$(warning "pkg-config output list does not contain drivers between 'whole-archive'/'no-whole-archive' flags.")$(error "Cannot generate statically-linked binaries with this version of pkg-config")endifendifCFLAGS += -DALLOW_EXPERIMENTAL_APIbuild/$(APP)-shared: $(SRCS-y) Makefile $(PC_FILE) | build $(CC) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS) $(LDFLAGS_SHARED)build/$(APP)-static: $(SRCS-y) Makefile $(PC_FILE) | build $(CC) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS) $(LDFLAGS_STATIC)build: @mkdir -p $@.PHONY: cleanclean: rm -f build/$(APP) build/$(APP)-static build/$(APP)-shared test -d build && rmdir -p build || true
```

然后我们用`make`编译,输入的执行文件会在新创建的`build`目录下，执行即可:

```
zartbot@zartbotWS:~/learn/dpdk/01_port_init$ sudo ./build/portinitEAL: Detected 96 lcore(s)EAL: Detected 2 NUMA nodesEAL: Detected shared linkage of DPDKEAL: Multi-process socket /var/run/dpdk/rte/mp_socketEAL: Selected IOVA mode 'VA'EAL: No available 1048576 kB hugepages reportedEAL: Probing VFIO support...EAL: VFIO support initializedEAL:   using IOMMU type 1 (Type 1)EAL: Probe PCI driver: net_i40e (8086:1572) device: 0000:5e:00.0 (socket 0)EAL: Probe PCI driver: net_i40e (8086:1572) device: 0000:5e:00.1 (socket 0)EAL: Probe PCI driver: net_i40e (8086:1572) device: 0000:5e:00.2 (socket 0)EAL: Probe PCI driver: net_i40e (8086:1572) device: 0000:5e:00.3 (socket 0)EAL: No legacy callbacks, legacy socket not created*****************************************number of available port: 4port: 0 Driver:net_i40eport: 1 Driver:net_i40eport: 2 Driver:net_i40eport: 3 Driver:net_i40eport: 0->MAC-> 3C:FD:FE:A9:A8:88port: 1->MAC-> 3C:FD:FE:A9:A8:89port: 2->MAC-> 3C:FD:FE:A9:A8:8Aport: 3->MAC-> 3C:FD:FE:A9:A8:8B
```

### 更复杂的接口初始化 

在需要用DPDK收发包时，通常我们需要做更复杂的接口初始化操作，因此我们通常会专门写一个`port_init`函数,这个函数的参数为`portid`和相关的`mbuf_pool`：

```
static inline intport_init(uint16_t port, struct rte_mempool *mbuf_pool){...}
```

首先我们需要定义一些常量， 主要是`RX_RING` / `TX_RING`的大小， MBUF的大小和Cache_size等

```
#define RX_RING_SIZE 1024#define TX_RING_SIZE 1024#define NUM_MBUFS 8191#define MBUF_CACHE_SIZE 250#define BURST_SIZE 32
```

接着定义一个default config的结构体

```
static const struct rte_eth_conf port_conf_default = {    .rxmode = {        .max_rx_pkt_len = RTE_ETHER_MAX_LEN,    },};
```

接下来就是整个port_init函数了：

```
static inline intport_init(uint16_t port, struct rte_mempool *mbuf_pool){    struct rte_eth_conf port_conf = port_conf_default;    const uint16_t rx_rings = 1, tx_rings = 1;    uint16_t nb_rxd = RX_RING_SIZE;    uint16_t nb_txd = TX_RING_SIZE;    int retval;    uint16_t q;    struct rte_eth_dev_info dev_info;    struct rte_eth_txconf txconf;        //查看这个接口是否为valid，非法则返回-1    if (!rte_eth_dev_is_valid_port(port))        return -1;            //获取接口信息    retval = rte_eth_dev_info_get(port, &dev_info);    if (retval != 0)    {        printf("Error during getting device (port %u) info: %s\n",               port, strerror(-retval));        return retval;    }    printf("\n\ninitializing port %d...\n",port);    //查看接口硬件Offload的能力是否支持，如果支持打开该功能    if (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_CHECKSUM)    {        printf("port[%u] support RX cheksum offload.\n", port);        port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_CHECKSUM;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)    {        printf("port[%u] support TX mbuf fast free offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM)    {        printf("port[%u] support TX IPv4 checksum offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_IPV4_CKSUM;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_UDP_CKSUM)    {        printf("port[%u] support TX UDP checksum offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_UDP_CKSUM;    }    //配置接口    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);    if (retval != 0)        return retval;    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);    if (retval != 0)        return retval;    //分配RX队列    for (q = 0; q < rx_rings; q++)    {        retval = rte_eth_rx_queue_setup(port, q, nb_rxd,                                        rte_eth_dev_socket_id(port), NULL, mbuf_pool);        if (retval < 0)            return retval;    }    txconf = dev_info.default_txconf;    txconf.offloads = port_conf.txmode.offloads;        //分配TX队列    for (q = 0; q < tx_rings; q++)    {        retval = rte_eth_tx_queue_setup(port, q, nb_txd,                                        rte_eth_dev_socket_id(port), &txconf);        if (retval < 0)            return retval;    }    //使能接口    retval = rte_eth_dev_start(port);    if (retval < 0)        return retval;    //获取接口MAC地址    struct rte_ether_addr addr;    retval = rte_eth_macaddr_get(port, &addr);    if (retval != 0)        return retval;    printf("Port[%u] MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8           " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",           port,           addr.addr_bytes[0], addr.addr_bytes[1],           addr.addr_bytes[2], addr.addr_bytes[3],           addr.addr_bytes[4], addr.addr_bytes[5]);    //打开混杂模式    retval = rte_eth_promiscuous_enable(port);    if (retval != 0)        return retval;    return 0;}
```

然后main函数就很简单的了, 初始化EAL，然后创建Mbuf pool

```
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports,                                        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());    if (mbuf_pool == NULL)        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");
```

然后初始化接口时，可以使用一个宏`RTE_ETH_FOREACH_DEV(portid)`将每个接口使能.

```
int main(int argc, char *argv[]){    struct rte_mempool *mbuf_pool;    unsigned nb_ports;    uint16_t portid;    int ret = rte_eal_init(argc, argv);    if (ret < 0)        rte_exit(EXIT_FAILURE, "initlize fail!");    printf("\n\n\n*****************************************\n");    nb_ports = rte_eth_dev_count_avail();    printf("number of available port: %d\n", nb_ports);    /* Creates a new mempool in memory to hold the mbufs. */    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports,                                        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());    if (mbuf_pool == NULL)        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");    /* Initialize all ports. */    RTE_ETH_FOREACH_DEV(portid)    if (port_init(portid, mbuf_pool) != 0)        rte_exit(EXIT_FAILURE, "Cannot init port %" PRIu16 "\n",                 portid);    return 0;}
```

### 收包程序 

收包主要是采用`rte_eth_rx_burst`函数，首先需要从`mbuf_pool`中分配一些空间,采用dpdk自带的`rte_pktmbuf_alloc`函数

```
    struct rte_mbuf *rx_pkt[BURST_SIZE];    for (int i = 0; i < BURST_SIZE; i++)    {        rx_pkt[i] = rte_pktmbuf_alloc(mbuf_pool);    }
```

然后一个for循环，不停的接收就行了,接收时`rte_eth_rx_burst(1, 0, rx_pkt, BURST_SIZE)`中第一个参数为portid，第二个为队列id，由于我们这个示例每个接口只有一个队列，同时我们采用loopback cable把port0和port1对连的，因此主要就是一个port0发，port1收的场景,因此接收的portid=1.
收到报文后可以通过`rte_pktbuf_mtod`函数去解析报文，然后可以通过结构体内变量赋值的方式修改值，更具体的示例我们在发包函数里讲， 这里只有一个简单的parse源MAC的地址的场景

```
    for (;;)    {                   uint16_t nb_rx = rte_eth_rx_burst(1, 0, rx_pkt, BURST_SIZE);        if (nb_rx == 0)        {            continue;        }        struct rte_ether_hdr *eth_hdr;        for (int i = 0; i < nb_rx; i++)        {            eth_hdr = rte_pktmbuf_mtod(rx_pkt[i], struct rte_ether_hdr *);            printf("Recv Pkt[%d] from MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8                   " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 " \n",i,                   eth_hdr->s_addr.addr_bytes[0], eth_hdr->s_addr.addr_bytes[1],                   eth_hdr->s_addr.addr_bytes[2], eth_hdr->s_addr.addr_bytes[3],                   eth_hdr->s_addr.addr_bytes[4], eth_hdr->s_addr.addr_bytes[5]);            rte_pktmbuf_free(rx_pkt[i]);        }    }  
```

### 发包程序 

本次发包程序的示例是以UDP发包为主，因此我们需要逐层初始化报文,然后报文的发送可以Burst的方式一次发送32个，我们也用这种方式来处理，报文发送的函数如下`rte_eth_tx_burst`.

首先我们初始化源目的MAC地址、IP地址和UDP Payload里面的内容，我们以SRoU header的一部分为例,相关的结构体定义如下：

```
    struct rte_ether_hdr *eth_hdr;    struct rte_ipv4_hdr *ipv4_hdr;    struct rte_udp_hdr *udp_hdr;            //Defined header in UDP    struct SRoU    {        uint8_t magic_num;        uint8_t srou_length;        uint8_t flags;        uint8_t next_protcol;        uint64_t pad;    };
```

然后我们分别来初始化每一层

```
    //init mac    struct rte_ether_addr s_addr = {{0x14, 0x02, 0xEC, 0x89, 0x8D, 0x24}};    struct rte_ether_addr d_addr = {{0x3c, 0xfd, 0xfe, 0xa9, 0xa8, 0x89}};    //init IP header    rte_be32_t s_ip_addr = string_to_ip("1.0.0.253");    rte_be32_t d_ip_addr = string_to_ip("1.0.0.1");    uint16_t ether_type = rte_cpu_to_be_16(0x0800);    //init udp payload    struct SRoU obj = {        .magic_num = 1,        .srou_length = 4,        .flags = 0xFF,        .next_protcol = 0,    };
```

初始化IP地址时有一个函数是从string转换为be32值

```
rte_be32_t string_to_ip(char *s){    unsigned char a[4];    int rc = sscanf(s, "%hhd.%hhd.%hhd.%hhd", a + 0, a + 1, a + 2, a + 3);    if (rc != 4)    {        fprintf(stderr, "bad source IP address format. Use like: 1.2.3.4\n");        exit(1);    }    return (rte_be32_t)(a[3]) << 24 |           (rte_be32_t)(a[2]) << 16 |           (rte_be32_t)(a[1]) << 8 |           (rte_be32_t)(a[0]);}
```

接下来我们来从mbuf中分配空间并初始化每个报文,注意其中`rte_pktmbuf_mtod_offset`函数的用法，大量的报文修改都采用这种方式.

```
    struct SRoU *msg;    struct rte_mbuf *pkt[BURST_SIZE];    for (int i = 0; i < BURST_SIZE; i++)    {        //分配空间        pkt[i] = rte_pktmbuf_alloc(mbuf_pool);                //利用rte_pktmbuf_mtod函数修改二层头，        eth_hdr = rte_pktmbuf_mtod(pkt[i], struct rte_ether_hdr *);        eth_hdr->d_addr = d_addr;                //这里我们根据burst循环改改源MAC地址玩~        struct rte_ether_addr s_addr = {{0x14, 0x02, 0xEC, 0x89, 0x8D, i}};        eth_hdr->s_addr = s_addr;        eth_hdr->ether_type = ether_type;                //然后利用rte_pktmbuf_mtod_offset函数， 移到IPv4头开始的地方，并定义结构体        ipv4_hdr = rte_pktmbuf_mtod_offset(pkt[i], struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));        ipv4_hdr->version_ihl = 0x45;        ipv4_hdr->next_proto_id = 0x11;        ipv4_hdr->src_addr = s_ip_addr;        ipv4_hdr->dst_addr = d_ip_addr;        ipv4_hdr->time_to_live = 0x40;                //修改UDP头，注意大端小端转换的rte_cpu_to_be_16函数        udp_hdr = rte_pktmbuf_mtod_offset(pkt[i], struct rte_udp_hdr *, sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));        udp_hdr->dgram_len = rte_cpu_to_be_16(sizeof(struct SRoU) + sizeof(struct rte_udp_hdr));        udp_hdr->src_port = rte_cpu_to_be_16(1234);        udp_hdr->dst_port = rte_cpu_to_be_16(6666);                msg = (struct SRoU *)(rte_pktmbuf_mtod(pkt[i], char *) + sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr));        *msg = obj;        int pkt_size = sizeof(struct SRoU) + sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr);                //最后是采用HW Offload的方式去计算Checksum        pkt[i]->l2_len = sizeof(struct rte_ether_hdr);        pkt[i]->l3_len = sizeof(struct rte_ipv4_hdr);        pkt[i]->l4_len = sizeof(struct rte_udp_hdr);        pkt[i]->ol_flags |= PKT_TX_IPV4 | PKT_TX_IP_CKSUM | PKT_TX_UDP_CKSUM;                ipv4_hdr->total_length = rte_cpu_to_be_16(sizeof(struct SRoU) + sizeof(struct rte_udp_hdr) + sizeof(struct rte_ipv4_hdr));        ipv4_hdr->hdr_checksum = 0;        udp_hdr->dgram_cksum = rte_ipv4_phdr_cksum(ipv4_hdr, pkt[i]->ol_flags);                //定义报文长度        pkt[i]->data_len = pkt_size;        pkt[i]->pkt_len = pkt_size;    }
```

然后我们采用每两秒发送一次的方式

```
    for(;;) {    uint16_t nb_tx = rte_eth_tx_burst(0, 0, pkt, BURST_SIZE);    printf("successful send %d pkts\n", nb_tx);    sleep(2);    }    for (int i = 0; i < BURST_SIZE; i++)    {        rte_pktmbuf_free(pkt[i]);    }
```

执行时，我们希望收发并行执行，因此我们可以将发包函数封装好, 并在main函数中调用

```
static int lcore_send(struct rte_mempool *mbuf_pool) {...}int main(){    rte_eal_remote_launch((lcore_function_t *)lcore_send,mbuf_pool,1);}
```

完成后的整个程序如下

```
#include <stdint.h>#include <unistd.h>#include <inttypes.h>#include <rte_eal.h>#include <rte_ethdev.h>#include <rte_cycles.h>#include <rte_lcore.h>#include <rte_mbuf.h>#include <rte_ether.h>#include <rte_ip.h>#include <rte_udp.h>#include <pthread.h>#include <string.h>#define MAX_PORTS 16#define RX_RING_SIZE 1024#define TX_RING_SIZE 1024#define NUM_MBUFS 8191#define MBUF_CACHE_SIZE 250#define BURST_SIZE 32static const struct rte_eth_conf port_conf_default = {    .rxmode = {        .max_rx_pkt_len = RTE_ETHER_MAX_LEN,    },};static inline intport_init(uint16_t port, struct rte_mempool *mbuf_pool){    struct rte_eth_conf port_conf = port_conf_default;    const uint16_t rx_rings = 1, tx_rings = 1;    uint16_t nb_rxd = RX_RING_SIZE;    uint16_t nb_txd = TX_RING_SIZE;    int retval;    uint16_t q;    struct rte_eth_dev_info dev_info;    struct rte_eth_txconf txconf;    if (!rte_eth_dev_is_valid_port(port))        return -1;    retval = rte_eth_dev_info_get(port, &dev_info);    if (retval != 0)    {        printf("Error during getting device (port %u) info: %s\n",               port, strerror(-retval));        return retval;    }    printf("\n\ninitializing port %d...\n", port);    if (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_CHECKSUM)    {        printf("port[%u] support RX cheksum offload.\n", port);        port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_CHECKSUM;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)    {        printf("port[%u] support TX mbuf fast free offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MT_LOCKFREE)    {        printf("port[%u] support TX MT lock free offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MT_LOCKFREE;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM)    {        printf("port[%u] support TX IPv4 checksum offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_IPV4_CKSUM;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_UDP_CKSUM)    {        printf("port[%u] support TX UDP checksum offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_UDP_CKSUM;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_TCP_CKSUM)    {        printf("port[%u] support TX TCP checksum offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_TCP_CKSUM;    }    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_SCTP_CKSUM)    {        printf("port[%u] support TX SCTP checksum offload.\n", port);        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_SCTP_CKSUM;    }    /* Configure the Ethernet device. */    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);    if (retval != 0)        return retval;    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);    if (retval != 0)        return retval;    /* Allocate and set up 1 RX queue per Ethernet port. */    for (q = 0; q < rx_rings; q++)    {        retval = rte_eth_rx_queue_setup(port, q, nb_rxd,                                        rte_eth_dev_socket_id(port), NULL, mbuf_pool);        if (retval < 0)            return retval;    }    txconf = dev_info.default_txconf;    txconf.offloads = port_conf.txmode.offloads;    /* Allocate and set up 1 TX queue per Ethernet port. */    for (q = 0; q < tx_rings; q++)    {        retval = rte_eth_tx_queue_setup(port, q, nb_txd,                                        rte_eth_dev_socket_id(port), &txconf);        if (retval < 0)            return retval;    }    /* Start the Ethernet port. */    retval = rte_eth_dev_start(port);    if (retval < 0)        return retval;    struct rte_eth_link link;    do    {        retval = rte_eth_link_get_nowait(port, &link);        if (retval < 0)        {            printf("Failed link get (port %u): %s\n",                   port, rte_strerror(-retval));            return retval;        }        else if (link.link_status)            break;        printf("Waiting for Link up on port %" PRIu16 "\n", port);        sleep(1);    } while (!link.link_status);    /* Display the port MAC address. */    struct rte_ether_addr addr;    retval = rte_eth_macaddr_get(port, &addr);    if (retval != 0)        return retval;    printf("Port[%u] MAC: %02" PRIx8 ":%02" PRIx8 ":%02" PRIx8           ":%02" PRIx8 ":%02" PRIx8 ":%02" PRIx8 "\n",           port,           addr.addr_bytes[0], addr.addr_bytes[1],           addr.addr_bytes[2], addr.addr_bytes[3],           addr.addr_bytes[4], addr.addr_bytes[5]);    /* Enable RX in promiscuous mode for the Ethernet device. */    retval = rte_eth_promiscuous_enable(port);    if (retval != 0)        return retval;            return 0;}rte_be32_t string_to_ip(char *s){    unsigned char a[4];    int rc = sscanf(s, "%hhd.%hhd.%hhd.%hhd", a + 0, a + 1, a + 2, a + 3);    if (rc != 4)    {        fprintf(stderr, "bad source IP address format. Use like: 1.1.1.1\n");        exit(1);    }    return (rte_be32_t)(a[3]) << 24 |           (rte_be32_t)(a[2]) << 16 |           (rte_be32_t)(a[1]) << 8 |           (rte_be32_t)(a[0]);}static intlcore_send(struct rte_mempool *mbuf_pool) {    struct rte_ether_hdr *eth_hdr;    struct rte_ipv4_hdr *ipv4_hdr;    struct rte_udp_hdr *udp_hdr;    //Defined header in UDP    struct SRoU    {        uint8_t magic_num;        uint8_t srou_length;        uint8_t flags;        uint8_t next_protcol;        uint64_t pad;    };    //init mac    struct rte_ether_addr s_addr = {{0x14, 0x02, 0xEC, 0x89, 0x8D, 0x24}};    struct rte_ether_addr d_addr = {{0x3c, 0xfd, 0xfe, 0xa9, 0xa8, 0x89}};    //init IP header    rte_be32_t s_ip_addr = string_to_ip("1.0.0.253");    rte_be32_t d_ip_addr = string_to_ip("1.0.0.1");    uint16_t ether_type = rte_cpu_to_be_16(0x0800);    //init udp payload    struct SRoU obj = {        .magic_num = 1,        .srou_length = 4,        .flags = 0xFF,        .next_protcol = 0,    };    struct SRoU *msg;    struct rte_mbuf *pkt[BURST_SIZE];    for (int i = 0; i < BURST_SIZE; i++)    {        pkt[i] = rte_pktmbuf_alloc(mbuf_pool);        eth_hdr = rte_pktmbuf_mtod(pkt[i], struct rte_ether_hdr *);        eth_hdr->d_addr = d_addr;         struct rte_ether_addr s_addr = {{0x14, 0x02, 0xEC, 0x89, 0x8D, i}};        eth_hdr->s_addr = s_addr;        eth_hdr->ether_type = ether_type;        ipv4_hdr = rte_pktmbuf_mtod_offset(pkt[i], struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));        ipv4_hdr->version_ihl = 0x45;        ipv4_hdr->next_proto_id = 0x11;        ipv4_hdr->src_addr = s_ip_addr;        ipv4_hdr->dst_addr = d_ip_addr;        ipv4_hdr->time_to_live = 0x40;        udp_hdr = rte_pktmbuf_mtod_offset(pkt[i], struct rte_udp_hdr *, sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));        udp_hdr->dgram_len = rte_cpu_to_be_16(sizeof(struct SRoU) + sizeof(struct rte_udp_hdr));        udp_hdr->src_port = rte_cpu_to_be_16(1234);        udp_hdr->dst_port = rte_cpu_to_be_16(6666);        ipv4_hdr->total_length = rte_cpu_to_be_16(sizeof(struct SRoU) + sizeof(struct rte_udp_hdr) + sizeof(struct rte_ipv4_hdr));        msg = (struct SRoU *)(rte_pktmbuf_mtod(pkt[i], char *) + sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr));        *msg = obj;        int pkt_size = sizeof(struct SRoU) + sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr);        pkt[i]->l2_len = sizeof(struct rte_ether_hdr);        pkt[i]->l3_len = sizeof(struct rte_ipv4_hdr);        pkt[i]->l4_len = sizeof(struct rte_udp_hdr);        pkt[i]->ol_flags |= PKT_TX_IPV4 | PKT_TX_IP_CKSUM | PKT_TX_UDP_CKSUM;        ipv4_hdr->hdr_checksum = 0;        udp_hdr->dgram_cksum = rte_ipv4_phdr_cksum(ipv4_hdr, pkt[i]->ol_flags);        pkt[i]->data_len = pkt_size;        pkt[i]->pkt_len = pkt_size;    }    for(;;) {    uint16_t nb_tx = rte_eth_tx_burst(0, 0, pkt, BURST_SIZE);    printf("successful send %d pkts\n", nb_tx);    sleep(2);    }    for (int i = 0; i < BURST_SIZE; i++)    {        rte_pktmbuf_free(pkt[i]);    }}int main(int argc, char *argv[]){    struct rte_mempool *mbuf_pool;    unsigned nb_ports;    uint16_t portid;    int ret = rte_eal_init(argc, argv);    if (ret < 0)        rte_exit(EXIT_FAILURE, "initlize fail!");    printf("\n\n\n*****************************************\n");    nb_ports = rte_eth_dev_count_avail();    printf("number of available port: %d\n", nb_ports);    /* Creates a new mempool in memory to hold the mbufs. */    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports,                                        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());    if (mbuf_pool == NULL)        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");    /* Initialize all ports. */    RTE_ETH_FOREACH_DEV(portid)    if (port_init(portid, mbuf_pool) != 0)        rte_exit(EXIT_FAILURE, "Cannot init port %" PRIu16 "\n",                 portid);    /* start packet send function on lcore-1 */    rte_eal_remote_launch((lcore_function_t *)lcore_send,mbuf_pool,1);    struct rte_mbuf *rx_pkt[BURST_SIZE];    for (int i = 0; i < BURST_SIZE; i++)    {        rx_pkt[i] = rte_pktmbuf_alloc(mbuf_pool);    }    for (;;)    {        uint16_t nb_rx = rte_eth_rx_burst(1, 0, rx_pkt, BURST_SIZE);        if (nb_rx == 0)        {            continue;        }        struct rte_ether_hdr *eth_hdr;        for (int i = 0; i < nb_rx; i++)        {            eth_hdr = rte_pktmbuf_mtod(rx_pkt[i], struct rte_ether_hdr *);            printf("Recv Pkt[%d] from MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8                   " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 " \n",i,                   eth_hdr->s_addr.addr_bytes[0], eth_hdr->s_addr.addr_bytes[1],                   eth_hdr->s_addr.addr_bytes[2], eth_hdr->s_addr.addr_bytes[3],                   eth_hdr->s_addr.addr_bytes[4], eth_hdr->s_addr.addr_bytes[5]);            rte_pktmbuf_free(rx_pkt[i]);        }    }    return 0;}
```

最后编译执行

```
zartbot@zartbotWS:~/learn/dpdk/01_port_init$ makecc -O3 -include rte_config.h -march=native -I/usr/local/include -I/usr/include/libnl3 -DALLOW_EXPERIMENTAL_API main.c -o build/portinit-shared  -L/usr/local/lib/x86_64-linux-gnu -Wl,--as-needed -lrte_node -lrte_graph -lrte_bpf -lrte_flow_classify -lrte_pipeline -lrte_table -lrte_port -lrte_fib -lrte_ipsec -lrte_vhost -lrte_stack -lrte_security -lrte_sched -lrte_reorder -lrte_rib -lrte_regexdev -lrte_rawdev -lrte_pdump -lrte_power -lrte_member -lrte_lpm -lrte_latencystats -lrte_kni -lrte_jobstats -lrte_ip_frag -lrte_gso -lrte_gro -lrte_eventdev -lrte_efd -lrte_distributor -lrte_cryptodev -lrte_compressdev -lrte_cfgfile -lrte_bitratestats -lrte_bbdev -lrte_acl -lrte_timer -lrte_hash -lrte_metrics -lrte_cmdline -lrte_pci -lrte_ethdev -lrte_meter -lrte_net -lrte_mbuf -lrte_mempool -lrte_rcu -lrte_ring -lrte_eal -lrte_telemetry -lrte_kvargs -lbsdln -sf portinit-shared build/portinitzartbot@zartbotWS:~/learn/dpdk/01_port_init$ sudo ./build/portinit[sudo] password for zartbot:EAL: Detected 96 lcore(s)EAL: Detected 2 NUMA nodesEAL: Detected shared linkage of DPDKEAL: Multi-process socket /var/run/dpdk/rte/mp_socketEAL: Selected IOVA mode 'VA'EAL: No available 1048576 kB hugepages reportedEAL: Probing VFIO support...EAL: VFIO support initializedEAL:   using IOMMU type 1 (Type 1)EAL: Probe PCI driver: net_i40e (8086:1572) device: 0000:5e:00.0 (socket 0)EAL: Probe PCI driver: net_i40e (8086:1572) device: 0000:5e:00.1 (socket 0)EAL: Probe PCI driver: net_i40e (8086:1572) device: 0000:5e:00.2 (socket 0)EAL: No legacy callbacks, legacy socket not created*****************************************number of available port: 3initializing port 0...port[0] support RX cheksum offload.port[0] support TX mbuf fast free offload.port[0] support TX IPv4 checksum offload.port[0] support TX UDP checksum offload.port[0] support TX TCP checksum offload.port[0] support TX SCTP checksum offload.Port[0] MAC: 3c:fd:fe:a9:a8:88initializing port 1...port[1] support RX cheksum offload.port[1] support TX mbuf fast free offload.port[1] support TX IPv4 checksum offload.port[1] support TX UDP checksum offload.port[1] support TX TCP checksum offload.port[1] support TX SCTP checksum offload.Port[1] MAC: 3c:fd:fe:a9:a8:89initializing port 2...port[2] support RX cheksum offload.port[2] support TX mbuf fast free offload.port[2] support TX IPv4 checksum offload.port[2] support TX UDP checksum offload.port[2] support TX TCP checksum offload.port[2] support TX SCTP checksum offload.Port[2] MAC: 3c:fd:fe:a9:a8:8asuccessful send 32 pktsRecv Pkt[0] from MAC: 14 02 ec 89 8d 00Recv Pkt[1] from MAC: 14 02 ec 89 8d 01Recv Pkt[2] from MAC: 14 02 ec 89 8d 02Recv Pkt[3] from MAC: 14 02 ec 89 8d 03Recv Pkt[4] from MAC: 14 02 ec 89 8d 04Recv Pkt[5] from MAC: 14 02 ec 89 8d 05Recv Pkt[6] from MAC: 14 02 ec 89 8d 06Recv Pkt[7] from MAC: 14 02 ec 89 8d 07Recv Pkt[8] from MAC: 14 02 ec 89 8d 08Recv Pkt[9] from MAC: 14 02 ec 89 8d 09Recv Pkt[10] from MAC: 14 02 ec 89 8d 0aRecv Pkt[11] from MAC: 14 02 ec 89 8d 0bRecv Pkt[12] from MAC: 14 02 ec 89 8d 0cRecv Pkt[13] from MAC: 14 02 ec 89 8d 0dRecv Pkt[14] from MAC: 14 02 ec 89 8d 0eRecv Pkt[15] from MAC: 14 02 ec 89 8d 0fRecv Pkt[16] from MAC: 14 02 ec 89 8d 10Recv Pkt[17] from MAC: 14 02 ec 89 8d 11Recv Pkt[18] from MAC: 14 02 ec 89 8d 12Recv Pkt[19] from MAC: 14 02 ec 89 8d 13Recv Pkt[20] from MAC: 14 02 ec 89 8d 14Recv Pkt[21] from MAC: 14 02 ec 89 8d 15Recv Pkt[22] from MAC: 14 02 ec 89 8d 16Recv Pkt[23] from MAC: 14 02 ec 89 8d 17Recv Pkt[24] from MAC: 14 02 ec 89 8d 18Recv Pkt[25] from MAC: 14 02 ec 89 8d 19Recv Pkt[26] from MAC: 14 02 ec 89 8d 1aRecv Pkt[27] from MAC: 14 02 ec 89 8d 1bRecv Pkt[28] from MAC: 14 02 ec 89 8d 1cRecv Pkt[29] from MAC: 14 02 ec 89 8d 1dRecv Pkt[30] from MAC: 14 02 ec 89 8d 1eRecv Pkt[31] from MAC: 14 02 ec 89 8d 1fsuccessful send 32 pktsRecv Pkt[0] from MAC: 14 02 ec 89 8d 00Recv Pkt[1] from MAC: 14 02 ec 89 8d 01Recv Pkt[2] from MAC: 14 02 ec 89 8d 02Recv Pkt[3] from MAC: 14 02 ec 89 8d 03Recv Pkt[0] from MAC: 14 02 ec 89 8d 04Recv Pkt[1] from MAC: 14 02 ec 89 8d 05Recv Pkt[2] from MAC: 14 02 ec 89 8d 06Recv Pkt[3] from MAC: 14 02 ec 89 8d 07Recv Pkt[4] from MAC: 14 02 ec 89 8d 08Recv Pkt[5] from MAC: 14 02 ec 89 8d 09Recv Pkt[6] from MAC: 14 02 ec 89 8d 0aRecv Pkt[7] from MAC: 14 02 ec 89 8d 0bRecv Pkt[8] from MAC: 14 02 ec 89 8d 0cRecv Pkt[9] from MAC: 14 02 ec 89 8d 0dRecv Pkt[10] from MAC: 14 02 ec 89 8d 0eRecv Pkt[11] from MAC: 14 02 ec 89 8d 0fRecv Pkt[12] from MAC: 14 02 ec 89 8d 10Recv Pkt[13] from MAC: 14 02 ec 89 8d 11Recv Pkt[14] from MAC: 14 02 ec 89 8d 12Recv Pkt[15] from MAC: 14 02 ec 89 8d 13Recv Pkt[16] from MAC: 14 02 ec 89 8d 14Recv Pkt[17] from MAC: 14 02 ec 89 8d 15Recv Pkt[18] from MAC: 14 02 ec 89 8d 16Recv Pkt[19] from MAC: 14 02 ec 89 8d 17Recv Pkt[20] from MAC: 14 02 ec 89 8d 18Recv Pkt[21] from MAC: 14 02 ec 89 8d 19Recv Pkt[22] from MAC: 14 02 ec 89 8d 1aRecv Pkt[23] from MAC: 14 02 ec 89 8d 1bRecv Pkt[24] from MAC: 14 02 ec 89 8d 1cRecv Pkt[25] from MAC: 14 02 ec 89 8d 1dRecv Pkt[26] from MAC: 14 02 ec 89 8d 1eRecv Pkt[27] from MAC: 14 02 ec 89 8d 1f
```