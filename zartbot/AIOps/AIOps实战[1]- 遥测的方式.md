# AIOps实战[1]: 遥测的方式

> 作者: zartbot  
> 日期: 2021年12月7日 22:19  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247487217&idx=1&sn=e5e1c47003e6fd964a11fdb2f9bba360&chksm=f9961e33cee1972517ecfdbcc22d1765ab566130dce7031013fff96842dbb6badd500ee47f2a#rd

---

对于遥测这个词，一开始的印象就是长征火箭发射时那些清脆而响亮的声音，“东风，光学，雷达，USB，跟踪正常，遥测信号正常”，正是这一声声正常，让人感到心安。而渣对于遥测的理解也从中学到了太多的东西.英文Telemetry一词前缀是Tele，来自希腊副词tele，原意为afar (遥远地）针对远距离测量自然有很多需要取舍的地方. 例如飞机在运行状态时对外发送ADS-B信号，而更多的数据记录在自身的黑匣子里.
如何降低遥测采集数据量
既然是遥测，测量通信量应该尽可能的小，甚至到最简单的情况就是一个比特报告正常就好。只是我们很多做网络的人忘记了遥测最根本的艺术，总是想尽可能地采集更多的信息，最后可能遥测通信量还大于业务传输数据量,例如很多逐跳的PostCard Telemetry。以前听说某个云一个集群如果需要采集分析交换机的运行数据需要一个500台服务器的集群，一下子不知道该说什么好...

所以诞生了很多降低通信量的协议和算法。首先是简单的包采样，例如很多核心路由器按照10000：1的比例采样分析流量，但是这样很容易漏掉一些通信。另一种做法就是降低输出的数据量大小，例如通过Netflow、IPFIX、SFLOW这样的技术产生数据。当然还有思科ASR1000那种完全支持1：1采样的Netflow技术。

而另一方面是一些交换机支持的Hardware-telemetry，最出名的自然是P4-INT，当然innovium这些也支持类似的IPT的技术，思科也有自己的FT和FTE，甚至还有很多年前的ERSPAN，而这些技术在国内基本上就没有好好用起来过，基本上整个遥测场景下都没有很好的采集软件和分析软件，这是渣写这个AIOps实战的用意，希望通过告诉大家一些方法来赋能整个行业，若是能够多个厂家一起玩个开源的Sonic-Telemetry就更好了。某些云虽然有，但是整个方式方法上还是太笨重了...

至于控制面的采集已经有很多成熟的方案了

![图片](assets/d9dc0cea656b.png)

Netflow
Netflow是一个非常老的协议了，它在`RFC3954`中定义， 后来为了扩展一些可变长度字段的遥测升级成了`IPFIX`(RFC7011). 但是很多人没能明白它的精髓，因为大多数厂商的实现都是简单的捕获一些五元组的信息和传了多少Bytes等.实际上它是一套压缩的通信协议标准，您可以理解成为它是一个对网络设备友好的Protobuf的编码方式，相对于PB，netflow协议可以自我产生Template和紧凑的编码使得占用极小带宽资源的情况下可以灵活的传输各种遥测数据。

![图片](assets/0fc64e890557.png)

思科在实现Netflow协议栈的过程中，可以传递DPI的结果，NAT的转换映射日志、防火墙日志、IPS丢包日志、ETA等安全事件，甚至在一些anyconnect终端上还可以监控host的运行程序情况。所以思科又把这样的Netflow称作Flexible Netflow(FNF).例如下面这条日志：

```
CONNECTION IPV4 INITIATOR ADDRESS:          10.aa.bb.1ccCONNECTION INITIATOR PORT:                  52584CONNECTION IPV4 RESPONDER ADDRESS:          1aa.bb.1cc.1ddCONNECTION RESPONDER PORT:                  9080TIMESTAMP MONITOR START:                    14:30:00.000FLOW OBSPOINT ID:                           4294967301IP PROTOCOL:                                6APPLICATION NAME:                           layer7 vmware-vsphereconnection to server resp delay sum:        498connection server resp counter:             3connection histogram late:                  0connection client retries counter:          0connection application delay sum:           4connection client server resp delay sum:    498connection total transaction duration sum:  502connection transaction counter:             3connection client retrans octets:           0connection responder retrans octets:        0connection responder retrans packets:       0connection ll to cnd sum:                   0connection ll cnd samples counter:          2connection ll to snd sum:                   165connection ll snd samples counter:          1connection ll csnd sum:                     495connection ll csnd samples counter:         3ip vrf id input:                            0          (DEFAULT)flow direction:                             Inputtimestamp abs first:                        14:51:22.330timestamp abs last:                         14:51:22.664connection initiator:                       Reverse initiatorconnection count new:                       1connection sum duration:                    1tcp window size sum:                        3135connection server packets counter:          13connection client packets counter:          16connection server network bytes counter:    5550connection client network bytes counter:    2636application ssl common-name:                172.2c.15a.18eip dscp:                                    0x00
```

其实任何一台新的思科基于IOS-XE的路由器都可以打开这样的功能，配置也非常简单

```
Router(config)#performance monitor context foo profile application-performanceRouter(config-perf-mon)# mode optimizedRouter(config-perf-mon)# exporter destination 192.168.99.101 source TenGigabitEthernet0/0/0.43 Router(config-perf-mon)# traffic-monitor allRouter(config)# interface Gi0/1/0Router(config-if)# ip nbar protocol-discoveryRouter(config-if)# performance monitor context foo
```

而采集和分析则需要思科的 StealthWatch、DNA-C或者vManage，或者第三方的LiveAction软件， 而渣在学习Netflow协议时和平时工作时需要给客户解释我们FNF的编码和原始数据导出格式时，做了一个能够灵活支持FNF各个字段的采集器

github.com/zartbot/goflow

这是一个受到cloudflare/goflow启发，然后fork过来发现一些问题后大改了一堆代码而构建的一个基于Go的流量采集器，严格按照RFC3954和RFC7011实现的，针对思科Flexible的理念，所有的字段采用了一个csv格式的InformationElementDB来构建，而不像cloudflare那样hardcode实现的。内部有个一example文件夹，编译后直接就可以使用。当然这个项目只是一个用来作为市场演示和导出数据敏感度分析用的小工作，真正的生产环境还是建议您采用思科和其他第三方合作伙伴提供的软件。

当然还有一些开源软件，只是支持dump的字段数量不够多..

https://www.elastic.co/guide/en/beats/filebeat/7.15/filebeat-module-netflow.html

如果您没有思科的路由器环境，也可以采用NTOP或者以前渣提供的zSpan这样的工具，收包后自己parse相关的信息然后产生特定的UDP telemetry报文来构建自己的遥测数据源。

例如思科开源出来的另一个安全分析工具:

https://github.com/cisco/mercury

数据采集的方式方法很多，在此就不多做举例了，最关键的还是接下来的环节。
如何构建一个好的采集器
在遥测过程中由于涉及到大量的数据处理，特别是相关性分析，有点类似于数据库的join操作，例如某个流源地址的DNS域名是啥，它属于哪个国家，是否在风控黑名单里，这些数据都需要即时的关联查询好了dump出来，例如在渣的goflow中就采用了多个日志更新工具关联信息:

```
func GeoRecordMap(d *datarecord.DataFrame, g *geoipmap.GeoIPCollector) { d.TypeAssertion() createAt := time.Unix(int64(d.ExportTime), 0) for _, value := range d.Record {  value["CreateAt"] = createAt  value["AgentID"] = d.AgentID  value["Type"] = d.Type  //Optional Template Correlation  optiontemplatemap.UpdateInterfaceMap(value, d.AgentID)  optiontemplatemap.UpdateAppMap(value, d.AgentID)  optiontemplatemap.UpdateCiscoVarString(value, d.AgentID)  optiontemplatemap.UpdateC3PLMap(value, d.AgentID)  optiontemplatemap.UpdateDropCauseMap(value, d.AgentID)  optiontemplatemap.UpdateViptelaTLOCMap(value, d.AgentID)  optiontemplatemap.UpdateFWEventMap(value, d.AgentID)  optiontemplatemap.UpdateFWZonePairMap(value, d.AgentID)  optiontemplatemap.UpdateFWClassMapMap(value, d.AgentID)  //Additional Service  flowinfo.UpdateFlowInfo(value, true)  identity.AddressLookUP(value, true)  geoipmap.UpdateGeoLocationInfo(value, g)  ciscoavc.ExtendField(value)  ciscoeta.ExtendField(value) }}
```

然后你的采集器需要构建好一个Streaming data的接口给后续的分析设备处理，例如使用MQTT、Kafka等，然后接下来您需要一个高性能的数据库落盘或者直接通过文件存储的方式落盘也行。

第一章大概就这么多了，接下来第二章我们将详细介绍一下AIOps中最关键的一点，数据抽取清洗和特征工程的实现。