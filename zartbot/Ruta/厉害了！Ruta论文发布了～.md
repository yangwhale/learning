# 厉害了！Ruta论文发布了～

> 作者: zartbot  
> 日期: 2021年12月17日 01:18  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487244&idx=1&sn=3eafe98c20993918c78fed01675569c1&chksm=f9961fcecee196d82c18aff831293253cef5dc988b4ae709dcb28084d04452c8a5898926eab3#rd

---

国内有一个习惯，没事就先搞个行业峰会颁个奖互相吹捧一下，然后估值就可以升天了。这次再来补充一个问题：在云原生环境下，SDWAN控制器的分区容错性和可靠性如何保证？Yang Model的配置和流表及相应的配置依赖所需要的顺序一致性如何保证？集中式算路面临的复杂度如何考虑？云网络VPC路由和SDWAN路由联动如何考虑？如何保障移动端实现多路径可靠的流量工程？如何保障SDWAN不利用专线的情况下实现Internet全球200ms零丢包的访问能力？

有些答案在论文里 <Ruta: Dis-aggregated routing system over multi-cloud>，下载地址：

https://arxiv.org/abs/2112.08686

![图片](assets/34d903c00ed5.jpg)
​

具体要再翻译成中文我真是懒得没空了...正如论文里分析的那样， 过去十年在SDN**的历程里，我们发明了太多的Overlay，而每个Overlay都在解决自己的一些问题，从终端的location-agnostic、到user-centric，然后到数据中的application-centric，再到SDWAN的transport-centric，再到K8S cloud-agnostic. 每个人都在摸象, 而通盘能够从园区网到广域网，再到数据中心和公有云并且完善的帮助国内各种企业建设过广域网的人，几乎不存在。而又能深刻理解每种网络设计原理的凤毛菱角。然后能够从客户架构做到路由协议设计再到数据面编码再到最终路由器软硬件体系架构和研发端到端拉通的还剩几个？

在整体架构完成设计，论文预印初稿已发之际，渣来回顾一下Ruta整个的研发历程.

SDWAN最早的雏形大概是在2011年思科企业网路由器BU的全球TME group meeting上（渣当然参与其中），主要是看到当时internet线路的价格远低于专线，然后整个公司下一代产品研发该往哪走？最后大家提出一种利用IPSec VPN构建基于Internet的广域网，然后选择了非常成熟的DMVPN方案，一方面是当年就有现网部署而且规模特别大，例如某国内快消企业和某经济型酒店当年都基于ADSL部署了几千个分支机构的规模。然后我们在上面叠加了Performance Routing(PfR)和一些广域网优化**的功能(WaaS)然后构造成了第一代的SDWAN架构，只是当时整个工业界连SDWAN这个名字都没有，我们将其称为Intelligence WAN。

这种简单的产品包装也带来了一个大问题，没有集中式的控制器，所以后面才慢慢补上了APIC-EM这样的控制器节点。但是整个IOS操作系统CLI**配置也没有完善的Yang model支持，而且本身配置也是on-the-fly change没有commit机制，所以整个项目风险重重，渣当时虽然可以很容易忽悠客户上iwan然后做业绩升职加薪的，但是内心还是一切为客户服务，反过来阻止了很多客户升级到iwan，同时渣也开始使用OpenWRT**网关和MQTT自研一个轻量级的SDWAN解决方案， 也就是同时渣的老板离开思科去创立了大地云网，而渣的一位好朋友也同时创立的大河云联，这也就是国内SDWAN的最早的企业了，渣没去是因为渣自己当时都没想明白一开始拷问各位的问题....与此同时，思科开始内部自研下一代IWAN，跟你们这些厂家想的一样，控制器算天算地算空气，最终项目取消，600M买了Viptela...

另一个拷问灵魂的问题来了， 为啥viptela要把控制器拆成那么多个网元？关于这一段历史和思科Viptela SDWAN设计的初衷，去年五月发过一篇

》[**企业应该如何上云？**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247483819&idx=1&sn=fa03e123df0cb72c967274d0d53799f9&chksm=f9961169cee1987fe76c583a7a5ddb24ffe6799b42a43af4e0c674e642cd58b11592fbedf744&scene=21#wechat_redirect)《

其中也介绍了思科SDWAN设计的思路，去中心化是最关键的一步。我一直调侃就是这个去中心化几个字花了思科6亿美金的学费，可惜现在工业界其他人还是搞不懂。。。

回到Ruta的研发上来， 2020年因为疫情的影响，大量的企业可能需要弹性将算力迁移到公有云上，于是渣和同事们很早就开始在阿里云和腾讯云上开始预研和测试思科的SDWAN解决方案，大概去年五月的时候完成了整个测试，并分析了数据

[》**国内共有云互通测试**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247483848&idx=1&sn=92ea64e1f707a27cb480118063b4b14f&chksm=f996110acee1981c69dd8a36b0c7c026ac077568a2ac839208b4094c47afe539d02c2bf620db&scene=21#wechat_redirect)《

》[**公有云互通测试报告(2)**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247483870&idx=1&sn=2bea5f422ef5bfeac20bccdace1f1130&chksm=f996111ccee1980ad8738bc6794fd94881912762789bbc0697be95f07cd4a61013f39af46395&scene=21#wechat_redirect)《

然后非常敏锐的发现，通过在Internet上实现Segment Routing可以非常容易的绕开拥塞点， 进一步考虑到QUIC这样的协议进行整合， 于是就有了：

》[**QUIC-SR：关于NewIP的答案**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247483970&idx=1&sn=d6617d2bdc7ef57eefa2c53d72b97aa4&chksm=f9961280cee19b96819f132d645bc4b91796dd32abe6a8261edb2103dbfb5940142fb6b85d25&scene=21#wechat_redirect)《

然后很快的在六月实现了整个Dataplane，并且发布出来了第一个RFC-draft: deraft-zartbot-sr-udp-00, 工程样品测试如下：

》[**下一代数据中心架构-3：从视频会议应用谈起**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484016&idx=1&sn=60d8909e513fe34e1c813d5588458b9e&chksm=f99612b2cee19ba4bbf71c276e8ff5c442fb501e07f97fe7703cc1fbf05a51dfaa41909d17df&scene=21#wechat_redirect)《

测试结果很漂亮，但是似乎有些SDWAN的难题还是需要从控制平面解决，一方面是linkstate的测量、另一方面是overlay路由的承载，于是七月初正式开始了ruta的架构设计，主要是ETCD的Key分配的问题， 论文里面写的很清楚了，然后七月底开始代码搬砖的工作：

![图片](assets/d55adc3c3894.jpg)

最终三周9000行代码搞定...

》[**重新发明一次路由器？**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484048&idx=1&sn=2c1904f2b627e1787d707f672c279d72&chksm=f9961252cee19b4428d7bb90348213c1ca9db29253e0f75acd4c2e9babe00fe54ac0f848e400&scene=21#wechat_redirect)《

然后也第一次用Ruta连通了腾讯云和阿里云10个VPC，并且把它们打通成一个大的***Transparency VPC.这个概念后面重点讲，事关各个公有云同学明年的3.75~***

》[**SRoU多云链接**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484066&idx=1&sn=6a9dc062e99e2b1ee3c1314b8866c253&chksm=f9961260cee19b76e11f63cc342100cd7bdc7c1c9ffaf6f7e792c022411d49e1e8f72702853f&scene=21#wechat_redirect)《

紧接着做了控制面的RFC-draft发布和相应的架构介绍:

》[**Ruta？智能插座？重新发明路由器！**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484099&idx=1&sn=8d335fa402c8bee2844d200588885bab&chksm=f9961201cee19b17a9c9e96a54e37754296754231451a2e243be47438f021584652752e326b8&scene=21#wechat_redirect)《

此时觉得渣的效率如何？然后Ruta就开始自举式的发展了，八月底采集了两周的数据开始做分析

》[**Ruta：不用花10个亿也能做千眼**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484142&idx=1&sn=2a6533102a81ef169124b57a02b4e83d&chksm=f996122ccee19b3accdb3c4100d0e8b532f482254436f1574ff53b7c3b97e12c1dea7d2fb0f4&scene=21#wechat_redirect)《

一个更宏伟的想法产生了，直接全球部署实现比东风快递还快的200ms可靠可达~但是有些东西很反人类，说早了人家听不懂，于是渣就去玩了一会儿别的事情，顺便去考了一下马老师说的老年人的健康证FRM...当然甲方爸爸的活也干了不少，国内某个大运营商的SDWAN集群配置和帮助一些外企实现Office 365的加速...以及思科SDWAN在阿里云的最终落地

》[**意图网络的语言学思考**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247483987&idx=1&sn=45e60a5c81fd6fb47498c53ad6116d62&chksm=f9961291cee19b872d7b445b3df7d0e9914ea0ef08bb37a902ec7c5b4e471b108993c4a49e02&scene=21#wechat_redirect)《

》[**思科SDWAN探秘(7)--基于意图的策略**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485126&idx=1&sn=df929df111ffcc61f237fe6bfc99887e&chksm=f9961604cee19f12fd1041d7e05af849ab3febbda7ffa20f0ceb8196e05a326fe6efc69ed521&scene=21#wechat_redirect)《

最终完成了200ms零丢包可达的超级快递工程

》[**200ms零丢包全球可达**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485381&idx=1&sn=e9f681e0e5c921568fedcb2455321ee1&chksm=f9961707cee19e11a2c6ad12d06efe6a8b5c3e6ec8cc93c3d8062d661272dc77275c89104222&scene=21#wechat_redirect)《

然后再进一步完善了体系结构设计:

》[**包处理的艺术(1)-从大自然中学习**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484506&idx=1&sn=e616536ffd9aa34dfe5db9308be90d5c&chksm=f9961498cee19d8e417048e43b5a6131cd2d65c80460c45d4fd4285517455c62b9371bb5602a&scene=21#wechat_redirect)《

》[**包处理的艺术(2)---如何设计协议**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484550&idx=1&sn=0627d92a24590316a573af70f24cb3f0&chksm=f9961444cee19d5251efcac850ee9e3339090fc99cc454c496750b197a514bab7e0f22043003&scene=21#wechat_redirect)《

》**[包处理的艺术(3)-RTC vs Pipeline](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485339&idx=1&sn=62439ad879f26f18c4434e1e51a0bdc3&chksm=f9961759cee19e4f776bf5719ce4c0635b09bbbebdcf7e37a682972e836d6c22e23bd64e2982&scene=21#wechat_redirect)**《

》**[包处理的艺术(4)-低延迟智能网卡设计](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485370&idx=1&sn=3b5590ccf58909f2d390df00bfb5d853&chksm=f9961778cee19e6e9b29c6898f2618e067422c69d8a1865ed67c798d764d97ef44bec91196cf&scene=21#wechat_redirect)**《

》[**云网融合的探索**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485478&idx=1&sn=668636e9656df16efc076bb1d49742a1&chksm=f99618e4cee191f29790c83418e8b7e7c3b03acc77d446464f38fd393c59d77d14f3d36cc923&scene=21#wechat_redirect)《

Transparency VPC
当然也发现一些在云端VPC的难题，混合云SDWAN域路由和VPC路由互通的问题，AWS需要我们SDWAN路由器跟它的网关建IPSec、Azure需要我们跟他们网关启用BGP，然后阿里还要改VPC的静态路由，还被一个客户逼着去写了一个BGP 侦听协议栈去触发底层静态路由的修改。混合云部署真的很烦人。。。

当然那么多VPC的路由对云提供商本身也很难受， 这不AWS又出事咯，一个流量工程就挂一大片，细颗粒度流量工程又做不了，然后阿里Sailfish的论文里，为了解决路由的问题这不也只能把barefoot pipeline折叠么，损失一半的性能？那么能不能在云上做类似的MPLS P、PE的业务直接分离把云提供商的基础架构变成P，然后边缘的智能网卡变成PE呢？或者云主机一个轻型网络协议栈变成PE呢？这样云网络的网关就不需要承载大量的路由了啊。

其实这就是渣前几天口误本来想说Transit VPC，错误的把正在写论文的Transparency VPC写出来了。利用在VPC之间实现Segment Routing over UDP，妥妥的在公有云私有云和园区网之间建立了一个统一的抽象层，而这一层居然还可以直通到用户的userspace，两端应用改个通信库，妥妥的完成，于是就有了Transparency VPC的概念。

研发大概就是这么一个过程，而今年一年基本上花精力在落地上，一方面给某个视频大厂的同学一起上了SRoU，另一方面也写了不少文章布道Ruta，紧接着还疯狂的输出了搞了一块DPU：NetDAM，直接目标就是在超算领域吊打某螺丝，当然NetDAM也借助了Ruta实现了各种风骚的拓扑结构下的MPI Allreduce。而通过一系列工程实践，Ruta该盖棺定论，该落地了~

正如论文结尾，It took a village to make Ruta possible，感谢Feng Cai, Yanhuan Mao, Yinghao Li, Bin Shi,Yijen Wang, Xing Jiang , Yin Wang , Sam Gao 等多位同事，好友和领导们的支持。据说下周还有一个新东西马上要开始测试，跟数据库相关。。。另外据说接下来思科还有一个大东西要发布~ 大动静~