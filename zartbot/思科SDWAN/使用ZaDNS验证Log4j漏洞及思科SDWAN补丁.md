# 使用ZaDNS验证Log4j漏洞及思科SDWAN补丁

> 作者: zartbot  
> 日期: 2021年12月20日 09:12  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487259&idx=1&sn=bad4cec5e9ba86beea9e3c212216ea3d&chksm=f9961fd9cee196cfe161b54cbf16da060a04a7fe72927d45e742c1a89d1a7fbf3f213f5312e5#rd

---

最近log4j的漏洞就不多说背景了，此文是帮助思科Viptela SDWAN客户复现漏洞和升级软件的教程，受影响的仅有vManage，因此升级vManage即可。而渣平时常常使用的自制DNS(ZaDNS)也为其它中招主机故障溯源提供了帮助。
Log4j漏洞触发
JNDI (Java Naming and Directory Interface) 是 Java 提供的一系列通用的接口服务封装，用户可以通过 JNDI 来访问不同协议下的多种资源， 而最容易触发进行bug验证的就是DNS协议了。国内普遍采用DNSlog.cn一类的服务来验证漏洞，但是这个服务本身很不安全的，有泄露信息的风险，因此这里用到了zaDNS来实现一个**私有的DNS**日志服务器验证内网漏洞。
ZaDNS
ZaDNS(github.com/zartbot/zadns)是一个基于golang开发的智能DNS路由系统，可以根据DNS域名选择特定服务器，也可以在众多服务器中选择最低延迟服务器响应，并集成了Tensorflow实现基于深度神经网络的DGA域名过滤等功能。并且还支持DNS请求日志的本地记录。

》[**SDWAN的智能DNS**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486086&idx=1&sn=231a54ea9c884da6ff558ae981c61952&chksm=f9961a44cee19352ce1900524c720eb404814c37d7cc6d5a8b65027b8cbb293250dd3886171e&scene=21#wechat_redirect)《

》[**支持AI的ZaDNS服务器**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486069&idx=1&sn=8c49e9e57d26a1225eb1363475f2b610&chksm=f9961ab7cee193a165175997e239fa76fc90c0e5474d87c14326f47c1cc6fda2d4e5791b6e5c&scene=21#wechat_redirect)《

故障复现
我们为您提供了基于Windows、MAC、Linux等多种版本的zadns，直接下载就可以使用，可以通过配置，将vManage的DNS服务器指向zadns：

```
vmanage# conf tEntering configuration mode terminalvmanage(config)# vpn 0vmanage(config-vpn-0)# no dns <existing-dns-server>vmanage(config-vpn-0)# dns 192.168.99.2 primary  <-zadns server ipvmanage(config-vpn-0)# commitCommit complete.
```

然后在zadns服务器的log目录下查看:

```
tail -f log/dns.log
```

在vmanage web登录界面，用户名那一栏填入:

```
${jndi:dns://testlog4j}
```

登录密码随便输入，如下图示

![图片](assets/9edc83a647d7.png)

观察ZaDNS服务器日志：

![图片](assets/b84a22cb07a4.jpg)

如果出现相应的testlog4j query则证明您的vmanage存在漏洞。

vManage升级流程
首先在思科官网下载vmanage的升级补丁包

![图片](assets/1c7b499f08e2.png)

使用管理员密码登录后， 点击`Maintenance`->`Software Repository` 

->`Software Image` ->`Add new Software` -> `vManage`

![图片](assets/684ae945fbd0.png)

然后将下载好的tar包拖动到页面上传.

![图片](assets/d585888094b0.png)

上传时间可能有些长,上传完毕后点击`Maintenance`->`Software Upgrade` ->`vManage` ->`Upgrade`，在弹出窗口中选择新的软件版本

![图片](assets/9f3a89f5f298.png)

等待安装完成

![图片](assets/008adc3ace2f.png)

安装完成后，点击`activate`并选择新的版本，vManage将会激活新版本并重启

![图片](assets/b21b1423a052.png)

重启完成后按照前述方法验证是否漏洞依旧存在， 完成后选择Set DefaultVersion,并改回到生产用的DNS服务器地址即可。

![图片](assets/1e77a59df2bc.png)

总结

平时对于服务器的DNS请求和内网发起的SYN连接通过BloomFilter过滤保存是一个良好的安全日志习惯， 渣平时都通过zaDNS留存终端和服务器的DNS请求， 而基于BloomFilter的访问日志是通过DPDK镜像流量实现的，一个非常简单的refcode, 通过这样的方式我们可以比较简便的获取日常的访问行为，为日后出现漏洞回溯并评估影响范围提供依据，当然还可以通过GeoIP2这些库关联地域等信息， 具体的内容等以后有空了慢慢更新吧...

```
#define NIPQUAD(addr)                \    ((unsigned char *)&addr)[0],     \        ((unsigned char *)&addr)[1], \        ((unsigned char *)&addr)[2], \        ((unsigned char *)&addr)[3]int lcore_recv_pkt(struct rx_params *rx){    const int socket_id = rte_socket_id();    printf("Core %u doing RX dequeue.\n", rte_lcore_id());    struct rte_member_setsum *setsum_vbf;    static struct rte_member_parameters params = {        .num_keys = MAX_ENTRIES, /* Total hash table entries. */        .key_len = 8,           /* Length of hash key. */        /* num_set and false_positive_rate only relevant to vBF */        .num_set = 16,        .false_positive_rate = 0.03,        .prim_hash_seed = 1,        .sec_hash_seed = 11,        .socket_id = 0 /* NUMA Socket ID for memory. */    };    params.name = "test_member_vbf";    params.type = RTE_MEMBER_TYPE_VBF;    setsum_vbf = rte_member_create(&params);    if (setsum_vbf == NULL)    {        printf("Creation of setsum_vbf fail\n");        return -1;    }    while (1)    {        struct rte_mbuf *bufs[BURST_SIZE];        const uint16_t nb_rx = rte_ring_dequeue_burst(rx->ring,                                                      (void *)bufs, BURST_SIZE, NULL);        if (unlikely(nb_rx == 0))        {            continue;        }        pkt_cnt += nb_rx;        struct flow_key *key;        for (int i = 0; i < nb_rx; i++)        {            struct rte_ether_hdr *eth_hdr;            struct rte_ipv4_hdr *ipv4_hdr;            uint16_t src_port, dst_port, set_vbf;            int src_key, dst_key, ret_vbf;            eth_hdr = rte_pktmbuf_mtod(bufs[i], struct rte_ether_hdr *);            if (likely(!(bufs[i]->packet_type & (RTE_PTYPE_L4_TCP | RTE_PTYPE_L4_UDP))))            {                rte_pktmbuf_free(bufs[i]);                continue;            }            //offset 4B for vlan packet            size_t eth_size = sizeof(struct rte_ether_hdr);            if (eth_hdr->ether_type == 0x81)            {                eth_size += 4;            }            ipv4_hdr = rte_pktmbuf_mtod_offset(bufs[i], struct rte_ipv4_hdr *, eth_size);            /* check flow */            if (likely(bufs[i]->packet_type & (RTE_PTYPE_L4_TCP | RTE_PTYPE_L4_UDP)))            {                key = malloc(sizeof(struct flow_key));                key->ip_src = rte_be_to_cpu_32(ipv4_hdr->src_addr);                key->ip_dst = rte_be_to_cpu_32(ipv4_hdr->dst_addr);                ret_vbf = rte_member_lookup(setsum_vbf, key, &set_vbf);                if (unlikely(set_vbf == 0 || ret_vbf == 0))                {                    rte_member_add(setsum_vbf, key, (uint16_t)src_key % 16 + 1);                    printf("%d.%d.%d.%d--->%d.%d.%d.%d\n", NIPQUAD(ipv4_hdr->src_addr),NIPQUAD(ipv4_hdr->dst_addr));                }                free(key);            }            rte_pktmbuf_free(bufs[i]);        }    }    return 0;}
```