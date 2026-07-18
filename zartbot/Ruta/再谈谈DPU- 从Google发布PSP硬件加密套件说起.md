# 再谈谈DPU: 从Google发布PSP硬件加密套件说起

> 作者: zartbot  
> 日期: 2022年5月25日 23:04  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487789&idx=1&sn=781039fc4959611d6387cb5a1cd842a9&chksm=f99601efcee188f93f31ef640d4916b14d9f298ddfc84b345011e0b22a878d948947b5d7eaf2#rd

---

Google搞QUIC,加上他们内部的Snap以及如今开源的PSP和Orion分布式SDN控制器.这些东西串接起来读者就会逐渐的明白渣搞Ruta和NetDAM的原因,当然还有AWS Aurora和一些Stateless的计算的融合

这应该是Intel IPU中给Google定制的几个IP之一,正如Google自己所述[1],云计算中有大量的安全隔离的需求,基于per-connection的,整个加解密消耗了大约0.7%的处理器资源以及相关的内存空间,所以最终需要将其Offload到网卡上,而相应的加解密协议就是 今天开源的PSP (a recursive acronym for PSP Security Protocol).

加解密主要面临的挑战是需要支持百万级的会话和100K级的新建链接, 这对于IPSec来说是一件比较困难的事情,主要是SADB的硬件卸载需要大量片上SRAM**,或者使用Cache, 例如根据Google的计算,假设一条SA需要占用256B,如果需要支持10M的会话,需要5GB (256B x 2 x 10M)的存储,而很多crypto offload引擎对于SA的数量支持非常有限,通常只有10K左右.

IPsec本身的IKE**会话建立也是一个非常消耗资源的过程,针对大规模的部署和网络带来的抖动,单机支持容量通常非常小. 思科的GetVPN,后面的Viptela SDWAN和渣的Ruta基本上都是从通信层来解决IKE建立连接的过程.

所以Google PSP也类似的支持Stateless或者Stateful的做法

![图片](assets/122c1ef4f993.jpg)

Stateless和Ruta类似的就是把SPI带到明文中,然后由中间的DPU或者其他网络设备offload加密, 然后封装格式和Ruta/NetDAM也类似,在UDP后面加一个特殊的头来做

![图片](assets/de7152048329.jpg)

结构体如下,很有趣的是仿照了IPv6中的NextHeader的定义

```
struct psp_hdr {  uint8_t next_hdr;  uint8_t hdr_ext_len;  uint8_t crypt_off;  uint8_t s_d_ver_v_1;  uint32_t spi;  uint64_t iv;};
```

看到这里有没有想起来去年渣在愚人节给大家开玩笑的IPv6-,即在IPv4上通过UDP+NextHDR灵活定义扩展取代IPv6+ :)

[**IPv6- : 基于IPv5的48bits寻址互联网协议**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485564&idx=1&sn=0e40eebc00311795c4de65909a2ec220&chksm=f99618becee191a8b8e1579062e95da872737d70d73954260b74ddacf9bbb600d064e072cb34&scene=21#wechat_redirect)

Offload的另一个需求其实Google没讲清楚,那就是必须要支持Inline Crypto,否则数据包密文先DMA一次,然后再DMA back到crypto engine,再DMA明文回去,压力太大了.
![图片](assets/40da19707c65.png)

所以说,你们要明白NetDAM的另一个设计,也需要别的公司大佬开源类似的东西后才懂下面这图的含义:
![图片](assets/4a8ac4873917.png)

正如Google前面所述,SADB需要大量的内存,网卡没有,网卡可不可以有呢? NetDAM和Ruta都是看似倒行逆施的东西, 在背后藏着的深意或许要等很多人再一次碰壁后才会明白.

有人问渣最近这半年在干嘛, 特别是基于NetDAM和Ruta之后. 疫情隔离在家讲真没法去实验室捣腾硬件的玩意,所以更多的在软件上和编程语言上花功夫. 一方面是读了一些代数的书,在看Rust**和Scala思考未来几年异构计算的同构表达的问题, 这是一个非常有趣的事情,虽然Intel也在搞OneAPI,但是C++本身的复杂性和安全性都出现了问题,而Rust+WebAssembly似乎是一个更安全高效的解决方案,例如RisingWave[2], Materialize[3],rust-flink等项目的出现, 以及Flink本身的一些Stateful function支持其他语言的特性. 另一方面是思科Viptela SDWAN的数据湖项目,从年初的架构和MVP到后来和整个团队一起做出原形,完成了一系列性能和功能的验证,也非常有趣.

当然在这个过程中又在考虑一个存算分离的问题,即关系型数据库的Serverless实现, 阿里云昨天有一篇文章挺有趣的走进RDS｜说说关系型数据库与Serverless[4] 正如文章所属:

“State-heavy applications will remain as BaaS”是目前对于数据库的一个基本认知，但这与数据库本身是否具备一定程度的Serveless能力其实是两回事, 数据库做Serverless有若干难点，总结如下：

Serverless没有内置的持久化存储，需要依赖远端存储，这就会导致在延时上较高；

客户端是基于连接的方式访问数据库，在客户端往往会维护连接池的方式供应用访问，而函数计算往往具备飘忽不定的网络地址，与数据库传统的IP+User+password鉴权的方式迥异；

很多高性能的数据库使用共享内存技术，而FAAS本身不具备共享内存的能力，会使得计算和数据库之前的资源动态扩展能力不一致

所以一方面读范畴论的书是从代数上去看如何解决关系型数据关系的问题,这个领域有个教授David I. Spivak也在做. 而另一方面是从底层上思考, 当你看到AWS Aurora以ACU的方式去统一底层的资源，不再对上层暴露底层具体的机型和代数。1ACU“相当于”2GiB的RAM，统一对底层资源做了标准化和规范化的处理。这与Serverless理念中资源的解耦、以及对底层资源的屏蔽一致,再回头来看NetDAM的Stateless

![图片](assets/dbecc4a70b99.png)

以数据为中心的角度来看,网卡上赋能内存就会做出很多很有趣的事情出来了, 当然这种想法也是我最早在几年前实现Nimble的时候想到的, 当年是因为Flink的各种问题,特别是面临分布式边缘侧计算的问题和SDWAN融合, Flink的内存管理又要在JVM上绕一圈感觉非常恶心,所以逐渐才有了NetDAM和Ruta的想法.

而最近网络的圈子里,似乎各种假大空的IPv6+算力网络, 有些笑不出来,又要面对同行的询问,只好简单友善的回复一句: 建议您们多去看看应用,当一个沦为管道的1D线性生物想通过算力扩张成2D的平面生活, 最好的方式是再加一个应用的维度,从3D生态体系去降维考虑问题,甚至带上历史,从4D的时空去思考成败.

#### 参考资料

[1]
Announcing PSPs cryptographic hardware offload at scale is now open source: *https://cloud.google.com/blog/products/identity-security/announcing-psp-security-protocol-is-now-open-source*
[2]
risingWave: *https://github.com/singularity-data/risingwave*
[3]
Materialize: *https://github.com/MaterializeInc/materialize*
[4]
走进RDS｜说说关系型数据库与Serverless: *https://mp.weixin.qq.com/s/wXD5Jf0UXkDW43HJh1HzTA*