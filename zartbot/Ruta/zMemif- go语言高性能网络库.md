# zMemif: go语言高性能网络库

> 作者: zartbot  
> 日期: 2021年10月30日 14:10  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486848&idx=1&sn=6822302c918e0bc60eb3763973c40116&chksm=f9961d42cee19454e2fc9018cef2510cf4bd2c49dfc3155204476b5a1aa67e2a28d2725f9ca1#rd

---

使用场景：高性能音视频传输(RTN)、网络遥测数据采集，SRv6或者Ruta等流量调度及其它测试场景...实测性能:收包`20Mpps`，发包`12Mpps`.

地址: github.com/zartbot/zmemif

### 简介

开发zMemif的主要动机是go有很高的处理能力，但是内置的udp库的确有些寒酸， 纯c开发效率又有些低，虽然可以用nff-go来实现go和dpdk的融合，但是cgo编译的确有点烦人，而且这个项目似乎也死了。然后考虑到容器的场景和手上`netDAM`及`ruta`两个项目的需求，使用无锁内存队列来在go和dpdk之间共享是一个不错的选择。

思科开源了memif的库，但是需要和vpp配合使用，对于很多互联网企业VPP**应用部署太麻烦，而且很多功能其实并不需要。同时netDAM也需要提供用户态的无锁内存队列功能，考虑到生态的兼容性，而且dpdk已经支持了memif的PMD**，因此还是选择了memif的数据结构， go的库来自于vpp/gomemif库，并做了一些修改，原来的代码组织结构不太好，同时Interface的定义和go的interface会让人混淆，因此将Interface改为了port，同时发现在dpdk中已经把master/slave政治正确到server/client了，于是也就顺手改成了同样的名称保证一致。

### 系统架构

如下图所示，主要是在收发包路径上提供一条基于共享内存访问的路径memif来承载UDP业务并通过memif和DPDK**绕开Kernel，为Golang提供原生的高性能包处理能力。而考虑到云端虚机等场景下除了业务的socket以外还需要一些管理的SSH或者以太网本身的ARP**等二层协议的支持，于是在dpdk侧创建了memif和vhost_user两个接口。

![图片](assets/f7c8a36406a1.png)

### Memif原理

memif通过一个UnixSocket来通信并交互共享内存区域，Server端会发送HELLO消息，客户端响应INIT，并且使用ADD_REGION消息来共享内存区域，然后通过ADD_RING消息共享size、offset、interrupt等信息给Server，最后通过CONNECT、CONNECTED消息确认连接。

![图片](assets/764ccd72b485.png)

这样的共享有一个好处，作为client端可以直接在用户态纯go(native-go)编程获得极高性能的收发包能力，而把复杂的内容通过共享内存交给DPDK处理，实现了基础架构和业务逻辑的很好的分离。

![图片](assets/b87a46e48f6e.png)

当然这种结构我们在以后会通过netDAM替代DPDK的Server端，直接为Go提供原生的Memif支持。同时这样的处理方式还可以为Serverless平台减轻网络栈的压力，直接通过共享内存交互event和数据。这就是以后容器和虚机、Serverless 统一和网络的交互方式，具体在一些特定场景下的应用以后慢慢说：）

![图片](assets/874411321832.png)

### zMemif使用
server侧(DPDK)
直接编译好了执行就行，然后默认会创建一个rutasys0的vhost-user接口，您可以把它当成一个普通网口配置IP地址和外界通信。同时系统会默认创建一个`/tmp/memif.sock`文件用于客户端go程序访问并建立UNIX-Socket。
client侧(native go)
首先肯定是创建UNIX Socket咯

```
socketName := flag.String("socket", "", "control socket filename")ctrlSock, err := zmemif.NewSocket("foo", *socketName)if err != nil { logrus.Fatal("create socket failed: %v", err)}
```

然后创建memif接口配置:

```
cfg := &zmemif.PortCfg{  Id:       0,  Name:     "memif_c0",  IsServer: false,  MemoryConfig: zmemif.MemoryConfig{   NumQueuePairs: 1,  },  ConnectedFunc: Connected, }
```

注意通常在userspace侧使用client模式，queue pair需要少于8个，如果有多核RSS的需求，请创建多个interface，具体可以参考`example/dpdk_co_worker`目录，请注意里面有一个ConnectedFunc的Callback函数，主要用于实现业务逻辑,例如一个简单的Echo，通过port.GetRXQueue/port.GetTXQueue函数获取queuepair，然后调用q.WritePacket/q.ReadPacket收发包即可。

```
func packetprocessing(p *zmemif.Port) { p.Wg.Add(1) defer p.Wg.Done() pkt := make([]byte, 2048) rxq0, err := p.GetRxQueue(0) if err != nil {  logrus.Fatal("Get RX-Queue failed.") } txq0, err := p.GetTxQueue(0) if err != nil {  logrus.Fatal("Get TX-Queue failed.") } //Server simply echo result to client for {  pktLen, err := rxq0.ReadPacket(pkt)  if err != nil {   logrus.Warn("recv error:", err)   continue  }  if pktLen > 0 {   txq0.WritePacket(pkt[:pktLen])  } }}func Connected(p *zmemif.Port) error { fmt.Println("Connected: ", p.GetName()) go packetprocessing(p) return nil}
```

接口结构体里面还新增了一个ExtendData接口，您可以将一些和这个Memif相关的数据结构放置其中，例如example/bw_test/sender/foo.go 中的包计数器

```
type PortStats struct { PacketCnt *uint64}cfg := &zmemif.PortCfg{   Id:       ifindex,   Name:     ifName,   IsServer: serverMode,   MemoryConfig: zmemif.MemoryConfig{    NumQueuePairs: queueNum,   },   ConnectedFunc: Connected,   ExtendData: &PortStats{    PacketCnt: &pktCnt,   },  }func sendpkt(p *zmemif.Port, qid int) { p.Wg.Add(1) defer p.Wg.Done() data := p.ExtendData.(*PortStats) txq, err := p.GetTxQueue(qid) if err != nil {  logrus.Fatal("Get TX-Queue failed.") } //Client simply send result and calculate the RTT for {  select {  case <-p.QuitChan: // channel closed   return  default:   sendpkt := make([]byte, 64)   s := txq.WritePacket(sendpkt)   if s > 0 {    atomic.AddUint64(data.PacketCnt, 1)   }  } }}    
```

创建完接口配置后，直接使用newport函数创建接口:

```
port, err := zmemif.NewPort(ctrlSock, cfg, nil) if err != nil {  logrus.Fatal(err) }
```

最后调用unix-socket startpolling函数使能接口

```
ctrlSock.StartPolling()
```