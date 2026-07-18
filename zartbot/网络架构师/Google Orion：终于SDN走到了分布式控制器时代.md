# Google Orion：终于SDN走到了分布式控制器时代

> 作者: zartbot  
> 日期: 2021年4月19日 23:10  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485752&idx=1&sn=61a3d9efc021f9e7f51ed468c5285f9d&chksm=f99619facee190ec56441939b689d4e38136affd9976ec087de5ed1d4779cdf40fda888c235f#rd

---

❝
今年春天是收获的季节, `网络法外狂徒扎波特`两个几年前预研的项目都有了顶级公司的跟进, 一个是AIOps的Nimble Engine，nVidia Morpheus开始做同样的事情了。另一件事是NSDI'21上Google发布的Orion，跟Ruta项目惊人的一致，SDN的终结一定是分布式的控制器。但是下一步是什么? Morpheus+Orion实现完全L5级的自动驾驶网络? 为啥不是带Nimble Engine的Ruta呢?
❞
接下来带着大家一起来读读Google在nsdi'21的这篇论文[1]

### 分布式控制器 

其实关于控制器怎么设计，我在包处理的艺术[2]中就做过详细的阐述过
❝
基于CAP理论，控制面需要保证一致性，因此只能CA或者CP二者选其一，但是本来数据面就有可能连不上控制面，因此Availability就可以不用考虑了，CP是最佳的选择。而数据面则需要保证尽力而为的转发，但基于目的地址的转发会在一致性上出现难点，因此采用SegmentRouting的思路，保证数据面BASE就好，最后是同构的方式来扩大整个集群的规模。
❞
另一篇文章则是分析Google分布式控制器实现后，去年还是发生了一次事故，本质就是数据面没有实现BASE导致[3] ,当然Google自己也有自己的难处，大量基于OpenFlow**的底层网络架构并不是一两年之内能够升级完的，所以Orion还是有Openflow的agent，即便是OF已经被全世界抛弃了。

其实几年前我在思科内部推动分布式控制器架构的时候，也遇到了好几Fellow的反对，观点出奇的一致:"只有集中式控制器才有统一的视角去调度资源，分布式控制器就是扯淡"

`感谢Google`的Orion为`分布式控制器`立下了一个标杆。

所以我当年做的图不知道有人现在明白了么？Orion+Morpheus，将控制逻辑和智能全下放~

![图片](assets/32734be9c8fd.png)

论文第一章的话值得大家回味:
❝
`Logically centralized` control require fundamentally high performance for updates, `in-memory` representation of state, and `appropriate consistency` levels among `loosely coordinating` micro-service SDN applications
❞
也就是说控制器本身应当是逻辑上集中式的，但是为了高性能需要实现基于内存的状态呈现，适当的一致性并配合相对松散的微服务协同，看见没有，本质上我用ETCD实现Ruta的控制面然后把很多路由计算和策略控制的任务推向网络边缘节点就是这个目的。而且充分的利用了ETCD来构建一个一致性的Pub-Sub服务，并且又借用了Lease机制实现了相对松散的失效控制。
❝
The decoupling of control from hardware elements breaks fate sharing in ways that make corner-case failure handling more complex.
❞
也就是说其实控制面某种意义上应该和转发面有一种耦合机制才能简化对失效事件的处理，有些时候也需要带外(out-of-band)的传输机制，所以在Ruta设计机制中也很充分的利用了ETCD的gRPC proxy机制提供带外和带内混合传输的方式.
❝
Managing the tension between centralization and fault isolation must be balanced carefully.
❞
其实这个问题是在说同构，不知道Google自己想明白没有，本质上当你有了同构以后才能实现这样的失效隔离.
❝
In a global network setting, we must integrate existing routing protocols, primarily BGP, into Orion to allow interoperation with non-SDN peer networks.
❞
所以看见没有，Ruta的路由前缀设计第一条原则就是要兼容BGP的EVPN机制.

### Orion设计原则 

论文的第三章讲述了一些设计原则, 文章中有一个图，关于Intent和Ground Truth的链式反应

![图片](assets/73d43e72b1e9.png)

然后利用Pub-Sub来共享Intent和Ground Truth，例如我在Ruta设计中也是，Intent来自于任何控制器或者人的意图，然后发布到ETCD中，可以是路由、也可以是策略、也可以是Identity、但是这个地方并没有看的很透，因为下面这句话讲的真的太好了,关于Intent和Ground Truth**的:

`Policy` comes primarily out of `business decisions`, and business decisions should be `close` to the `business`, not the topology. Hence, policy, or least some element of policy, is often best done when `centralized`.

`Topology and reachability`, however, are `grounded` in what should be the only source of truth about the state of the network, the network itself. Therefore, it makes sense that decisions related to the topology and reachability, from detection to reaction, should be kept close to the network itself; hence, topology and reachability decisions should trend toward being `decentralized`

基于意图网络的描述很有趣，但是最终这件事情还是要从人的意图触发, 大家可以参考意图网络基于语言学的思考[4]

但是后面的`Continuous Reconciliation`把事情搞复杂了:

![图片](assets/e64e855b3339.png)

复杂的原因就来自于Openflow，每个设备都有一些状态，而这些状态某种程度上必须做出取舍携带在包里随路转发，也就是我一直讲的数据面需要实现基于SR的BASE(Basiclly Available，Soft state，Eventually consistent).所以这也是去年Google云导致故障的最根本原因.

然后就是一个`无脑`的问题，控制器断开连接,看见没有，又回到了我说的控制器需要保持CP的设计，因为Availability无法满足会导致Headless

![图片](assets/9e23bb1a295a.png)

这种情况下，Google就没有经验了，本质上你要看网络是什么？云原生路由架构探索[5]写的非常清楚,网络要做成一种资源供应用调用,所以看到Google架构里有Orion App直接操作NIB和我想用K8S集群直接操作ETCD有异曲同工之妙。

![图片](assets/8df1460ef84f.png)

但是Google似乎还没有在这块抽象的很好，毕竟也不是专门做网络的情有可原,这个流程太长了，真的...

![图片](assets/fea33751a33d.png)

这个地方插入一个RE是一个错误的决策，当然这也是数据面OpenFlow导致的必须要做一些计算，而我在Ruta中汲取了思科做SDWAN的教训，直接采用链路状态[6]分布式计算的方式，把计算放在了头端，而利用源路由[7]的能力解决了中间有状态的部署，是不是和Clarence当年说RSVP-TE很类似:)

当然为什么要这样弄，因为可以通过Nimble engine[8]来构造一个搞不死的网络法外狂徒Zartbot~

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

Ruta相关资料[9]:

https://tools.ietf.org/html/draft-zartbot-sr-udp-00

https://tools.ietf.org/html/draft-zartbot-srou-control-00

#### Reference

[1]
Orion: Google's SDN Control Plane: https://www.usenix.org/conference/nsdi21/presentation/ferguson
[2]
包处理的艺术(2)如何设计协议: https://mp.weixin.qq.com/s/m0QCtKskB7WZQrVRZyx_Fg
[3]
谷歌云故障分析及反思: https://mp.weixin.qq.com/s/_QqBXn8hB3TyYqkEfIp0aA
[4]
意图网络的语言学思考: https://mp.weixin.qq.com/s/7yydq3QK4VgP-RTOEznyJw
[5]
云原生路由架构探索: https://mp.weixin.qq.com/s/NcKyIJ-8dpfAtkKJ-kUsdA
[6]
Internet的性能测量: https://mp.weixin.qq.com/s/Wehkmy28yQdVR3lvzceyUQ
[7]
Ruta数据面转发: https://mp.weixin.qq.com/s/IlzogM0jUBWRXwLgmzoM5Q
[8]
nVidia Morpheus：浅谈AI在网络中的应用: https://mp.weixin.qq.com/s/Uci3n2NrpmN7115TKI7kgQ
[9]
Ruta: https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxNzQ5MTExNw==&action=getalbum&album_id=1364600286966415361#wechat_redirect

**最后一句话，各个国内企业要抄Google Orion的，还不如找我一起玩Ruta，已经有大厂一起玩了~，另外P4的交换机已经下单，到货了我就Porting，eBPF或者VPP的软件平面也很容易实现，我已经给出Reference设计，控制面协议开放的RFC，ETCD上搞点Lease、Watch特别容易吧~**