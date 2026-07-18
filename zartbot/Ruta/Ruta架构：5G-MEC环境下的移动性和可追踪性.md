# Ruta架构：5G/MEC环境下的移动性和可追踪性

> 作者: zartbot  
> 日期: 2020年9月20日 14:10  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484313&idx=1&sn=ba4f29c5004c057b1b3cefdfb007df1f&chksm=f996135bcee19a4dbb621c28865d8f6faee5a8efa835fe6105cd22c797fda56be211921c1fd3#rd

---

大概2011年的时候，某司无线网络事业部（WNBU）要来中国成立研发部门，被老板安排去建立这个团队。一方面是进入某司前就有很丰富的无线部署经验，因为2004年Intel发布迅驰架构时为了促进WLAN发展给了某校一堆思科的胖AP，于是跟着老师一起开始了RF测量选点和部署的任务。另一方面，非常感激那位老板那些年的培养，他管的产品线都给我安排机会去轮岗做项目，从IPRAN 到 Starent的PGW + SP-WiFi解决方案。还主动让我挂着TME的头衔深入到销售一线做架构和方案设计以及根据市场的反馈回来做新产品的规划和研发。现在回想起来真是感激不尽啊~

记得某一年好多城市出现地铁因为干扰而停运的事故，其实就是这些城市地铁没有采用GSM-R，而是采用了基于WLAN**的解决方案。正是因为一些项目经历和现网故障，所以对如何基于IP网络实现高移动性的探索一直没有停止过， 过去几年也一直和某司IOT WLAN团队在探讨如何保障高速移动环境下的IP通信能力，最终的结论就是需要覆盖范围周边做随路同步（利用Segment Routing)，而设备的IP包还需要一些流的特征（类似于TEID），因此在Ruta的设计中针对5G/MEC环境的需求设计了FlowID+ Segment Routing的处理方式。

![图片](assets/6e377d5d1f02.png)

**可追踪性**（FlowID）：**

FlowID的设计来自于做一些web应用时，用户登录地址端口也经常变化，服务器地址也因为负载均衡的需求经常在变。Json Web Token(JWT)机制便是一个很好的处理方式，另一方面基于Google Dapper论文的APM系统也需要Tracing ID做标记，而APM和NPM的结合最头疼的便是NPM可能无法解密Tracing ID,或者需要parse**整个应用层协议栈找到Tracing ID。所以一直在尝试着做两者的结合，再加上以往做核心网的经历，GTP协议中也有TEID，而移动网中的大量策略也基于TEID。同时还受到QUIC Connection-ID的影响

![图片](assets/342e7cd8a2df.jpg)

还有一个原因是基于意图网络的思考，IP地址编址通常基于地理位置可以逐渐汇聚降低路由表的容量。而安全策略则是基于终端编址，传统的SGT**/EPG等Group Tag无法做到1-1 Mapping，也就是说它们是非满秩的不存在逆矩阵。而Connection-ID/TEID/TracingID都是满秩的。所以下一代路由协议标签设计，所以报文标签的抽像规则应该如下

![图片](assets/0dce8464ea70.png)

Policy Label必须要成为一个满秩映射，否则源目的地址变更后无法识别流信息，基于上述原因，在Ruta头部设计的过程中添加了FlowID的定义。

![图片](assets/501edcbe9a78.jpg)

但是为啥为变长的，主要是节约编码空间，类似于QUIC Long/Short header定义， 如果中继设备**有流表缓冲的情况下，仅在地址变化时补充源目的ConnID/TEID即可。

**移动性（SR over UDP）：**

下面来说一下5G/MEC环境下uLLC场景的潜在部署算法，这个需求来自高德导航的一位朋友，主要是车联网场景下的随路计算。通常随着车辆高速移动，移动方向车辆自己是清楚的，所以它也可以query的方式获取下一个UPF或者MEC的地址。但是如果从局方来看，车辆运动轨迹预测，潜在UPF分配无法实施。而随路计算通常需要将计算的context从一个MEC迁移到另一个MEC。且终端还存在非MEC区域漫游到MEC区域，和五元组变化等问题。如果按照传统的TCP会话处理，UPF上可能还需要进行TCP shadow proxy等处理。一个比较聪明的做法，我们可以通过SR的方式同步计算context到MEC周围的节点，这样计算的context是完全同步的，备份的问题也解决了，当然这里面还有几篇论文要发，就不说太详细了。

![图片](assets/990cb702c0df.png)