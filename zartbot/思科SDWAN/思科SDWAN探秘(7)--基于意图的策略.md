# 思科SDWAN探秘(7)--基于意图的策略

> 作者: zartbot  
> 日期: 2021年1月29日 12:30  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485126&idx=1&sn=df929df111ffcc61f237fe6bfc99887e&chksm=f9961604cee19f12fd1041d7e05af849ab3febbda7ffa20f0ceb8196e05a326fe6efc69ed521#rd

---

**题记：SDWAN提供了很多的工具，用不用的好却是一门艺术**

**下面是一个真实案例来给大家介绍一下思科意图网络如何将意图扩展到多域环境,实现SDA、SDWAN和ACI以及云网络的基于意图的多域融合**

SDWAN的策略一直是非常难的一个话题，最近解一个客户的bug借助以前关于意图网络声明式策略框架《[**意图网络的语言学思考**](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485111&idx=2&sn=b515657c21488f98eee7e57b65b9a67b&chksm=f9961675cee19f63ff747dd5f1f8f855837e56cf96181a5a2b2a268077be10b71d4d047c41a7&scene=21#wechat_redirect)》的研究，把这事想明白了，顺便通过这个案例来给大家做一个SDWAN策略的串烧~一个案例基本上把很多的策略都用全了。

**网络是由多个域组建而成的，数据中心有BGP-EVPN或者ACI的解决方案，甚至是在云端，我们的策略是以应用为中心。而在园区网的策略通常是UCI，即以用户为中心的策略。对于SDWAN的策略，就成了下面这样...**

![图片](assets/3549c6031372.png)

但是我们仔细看看用户对于策略的意图，实际上是很简单的，本质上和安全相关的是一些访问控制的策略。

![图片](assets/9c79f61dd106.png)

但是另一方面流量调度策略需要多个Segment，如下图所示：

![图片](assets/61ab5f372cf2.png)

但是像下面这种解决方案，您用起来肯定心惊胆战吧：）

![图片](assets/45462869c64b.png)

所以说，我一直都在针对报文如何标记做阐述，一定要把多租户、安全、流量工程这三个标签分离才行。

![图片](assets/a762ef3b9ebe.png)

但是现阶段很多SDWAN仅支持一层标签，怎么办呢？那么就把三个东西映射在一个标签上，然后在不同的策略点实施策略就好：）

我们来看一个需求，很多企业有业务出海多云连接的需求，通常会安装如下拓扑构建SDWAN网络。

![图片](assets/7a42c06d1c0b.png)

业务需求也非常明确，国内的Internet线路用于和国内的PoP点传输公有云的业务，例如Office365.而原有的MPLS网络由于等保合规等要求，承载私网业务，并且要隔离。当然同时还伴生着某些用户能够访问应用，某些不能访问的需求。

**我们利用一个特殊的VPN-ID来代表用户和应用之间的Contract，然后针对这个VPN做路由泄露，将UserVPN和应用Vxlan中的路由泄露进这个VPN，自然就把它们连通了，这样相对于ACL访问控制的方案更加直观，能访问就有路由。**

![图片](assets/ee54835a9b5a.png)

**另一方面，我们针对这个VPN做基于VPN的流量工程或者FEC、QoS，或者Service-Chaining到firewall，这样就完成了整个基于意图的策略控制。**

**
**

那么接下来咱们进入实战环节，用户侧路由器为ZartbotLTE和ISR-Home，中间的PoP点为Cedge-MPLS-PE，应用侧为两台C8KV_DMZ_1/2

![图片](assets/a1138ba3a08a.png)

首先，我们定义一些以后要用到的list，如前文所述，我们需要定义一些站点组

```
site-list O365Service   site-id 101site-list UserSite   site-id 201   site-id 202
```

然后是PoP点的资源

```
  tloc-list O365_ServicePoP   tloc 102.0.0.1 color biz-internet encap ipsec
```

然后我们定义承载Office365的应用VPN列表，您可以认为这是应用端的EPG对应的，可以根据Application组来划分。

```
  vpn-list Office365   vpn 365
```

以及用户侧可以访问O365资源的VPN列表，例如对应SDA中的SGT

```
  vpn-list O365_ALLOWED_VPN_LIST   vpn 200   vpn 201
```

最后肯定是要匹配Office365的应用列表咯：

```
app-list Office365app excel_onlineapp grooveapp hockeyappapp live_groupsapp live_hotmailapp live_meshapp livemail_mobileapp lyncapp lync_onlineapp microsoftapp ms-live-accountsapp ms-lyncapp ms-lync-audioapp ms-lync-controlapp ms-lync-videoapp ms-office-365app ms-office-web-appsapp ms-servicesapp ms-teamsapp ms-teams-audioapp ms-teams-mediaapp ms-teams-videoapp ms-updateapp ms_communicatorapp ms_onenoteapp ms_plannerapp ms_swayapp ms_translatorapp office365app office_docsapp onedriveapp outlookapp outlook-web-serviceapp owaapp powerpoint_onlineapp share-pointapp sharepointapp sharepoint_adminapp sharepoint_blogapp sharepoint_calendarapp sharepoint_documentapp sharepoint_onlineapp skydriveapp skydrive_loginapp skypeapp windows_marketplaceapp windowsliveapp word_onlineapp yammer
```

然后我们在应用侧节点，也就是Site101中添加两个控制策略：

```
apply-policy site-list O365Service  control-policy O365_To_USER_IN in  control-policy O365_TO_USER_OUT out
```

in和out的方向需要注意，下面这张图很清楚：

![图片](assets/a38919e62b26.png)

in代表的是路由进入到vsmart控制器的时候需要做的策略，也就是说，当应用侧需要往控制器发布路由的时候，我们期望它将路由信息通告到能够访问的业务侧VPN中：

```
 control-policy O365_To_USER_IN  sequence 10   match route    vpn 365   !   action accept    export-to     vpn-list O365_ALLOWED_VPN_LIST    !   !  !  default-action accept !
```

out代表的是控制器公告路由给其它节点，我们定义了这些需要访问O365业务的Site-list中节点看到的路由下一跳资源都会被指向O365_ServicePoP的TLOC中，由此可以约束流量仅走Internet的color

```
control-policy O365_TO_USER_OUT  sequence 10   match route    site-list UserSite    vpn       365   !   action accept    set     tloc-list O365_ServicePoP    !   !  !  default-action accept !
```

针对用户侧节点的策略有2个，但有一个是Datapolicy

```
 site-list UserSite  control-policy USER_TO_O365_IN in  control-policy USER_TO_O365_OUT out  data-policy O365_Optimize from-service
```

用户侧通告给控制器的路由策略如下, 将自己的客户端相应VPN并且走Internet的路由通告给控制器时导出到Contract定义的VPN365中，这样就实现了应用侧和用户侧通过VPN365互通。

```
 control-policy USER_TO_O365_IN  sequence 10   match route    color    public-internet    vpn-list O365_ALLOWED_VPN_LIST   !   action accept    export-to     vpn 365    !   !  !  default-action accept
```

最后定义流量如何进入这个Contract就是用了DataPolicy，匹配应用，导入流量到VPN365.当然我们还做了一些针对DNS的特殊的处理，例如Office365的DNS全部走内网的服务器，而互联网应用的其它DNS则本地BreakOut出去，这样的做法可以防止一些隐私的泄露。最后很简单的将匹配到的Office365流量列表(app-list）设置下一跳TLOC资源和VPN到365 就搞定了。

```
data-policy O365_Optimize  vpn-list O365_ALLOWED_VPN_LIST   sequence 5    match     dns-app-list Office365    !    action accept     count o365dns    !   !   sequence 6    match     dns request    !    action accept     count        normal     nat use-vpn 0     redirect-dns 192.168.1.1    !   !   sequence 10    match     app-list Office365    !    action accept     count o365_opt     set      vpn       365      tloc-list O365_ServicePoP     !    !   !   default-action accept  ! ! 
```

最后当然是验证策略呢~这么多跳，都还加密的验证起来不费力么？而且哪个策略管用哪个不管用你看的清么？**思科中国研发中心的一群大神同事们做了一个非常牛逼的软件功能，在控制器上可以轻松实现多跳关联和转发日志解析**，在vManage中点“Monitor-》Network Wide Path Insight”，选择你所在的SiteID和VPN点击Start就好

![图片](assets/650de113bd44.png)

然后下面的Flow列表中过滤搜索你感兴趣的应用，就能看到转发的详情了：

![图片](assets/655b0552600e.png)