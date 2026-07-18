# SDWAN的智能DNS

> 作者: zartbot  
> 日期: 2021年8月9日 16:12  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486086&idx=1&sn=231a54ea9c884da6ff558ae981c61952&chksm=f9961a44cee19352ce1900524c720eb404814c37d7cc6d5a8b65027b8cbb293250dd3886171e#rd

---

本文主要讲述两个话题: 基于SDWAN的CDN识别和多路径优化DNS及基于DoT(DoH)的零信任网络架构实现

### SDWAN智能DNS

最近处理一些客户的案例,发现在应用进行动态选路的时候,DNS通常并没有识别SaaS**的CDN能力,最后流量调度南辕北辙. 所以最近基于ZaDNS项目写了一些DNS多出口并行解析和CDN发现的功能.

项目地址: github.com/zartbot/zadns

很多CDN业务的DNS会根据请求的源地址不同响应不同的业务地址，一个SDWAN Fabric通常跨越多地并且有多个运营商出口，将DNS解析结构收集归纳，并通过反查AS号来识别CDN或者使用GeoIP查询来计算物理距离。然后将最优的结果反馈给终端并且控制SDWAN转发平面的路径配置。

例如访问思科可以得到如下列表
IPASNCityCountryLat/LongDistance(km)96.16.179.237GTT Communications Inc.San JoseUnited States37.33,-121.889977.2972.163.4.161CISCOSYSTEMSRichardsonUnited States32.94,-96.7011860.22104.111.198.247AKAMAI-ASHong KongChina22.29,114.151199.81104.95.63.78AKAMAI-ASDallasUnited States32.77,-96.8011870.65104.76.12.36AKAMAI-ASOsakaJapan34.68,135.511390.96
或者某网站：
IPASNCityCountryLat/LongDistance(km)157.240.11.35FACEBOOKLos AngelesUnited States34.05,-118.2410464.99157.240.218.35FACEBOOKSingaporeSingapore1.30,103.853778.04179.60.194.35FACEBOOKKuala LumpurMalaysia3.16,101.703714.4831.13.93.35FACEBOOKDallasUnited States32.77,-96.8011870.65
SDWAN应该内置一个DNS服务器**来实现这些功能，并且和SDWAN控制器和本地网关联动实现动态的基于性能的选路，例如对每个路径进行HTTPS Ping或者TCP ping 80、443、UDP-QUIC等计算RTT，然后和网关间的SLA整合一起计算出最优路径来。ZaDNS在Cisco SDWAN路由器上可以直接通过Container**安装运行，然后配合API控制vSmart或者本地Local Policy即可完成路径选择

### 基于DNS的ZTNA

很多云原生业务都有DNS实现业务发现和注册，这些在K8s里面已经很普遍了，而针对广域网侧或者用户端侧，例如SDWAN、SDA，我们的实现是LISP**动态加载Overlay地址进转发表实现的，或者在SDWAN路由器上构建防火墙实现广域网的安全隔离。为了将整个链路拉通对齐，传统的零信任网络架构需要一个控制器，那么不干脆把这些东西全部整合进DNS里面。但是DNS无法对终端进行鉴权，还好DNS over TLS和DNS over HTTPS的出现可以解决这些问题了、这样我们就可以通过DNS知道客户端需要访问某个服务，然后自动的通过DNS鉴权客户后修改网关的ACL和路由表实现动态的策略通行。

![图片](assets/275d5891ec46.png)

这个功能架构设计已经完成，以后会在ZaDNS中实现。