# Ruta数据面转发行为详解

> 作者: zartbot  
> 日期: 2021年4月9日 13:08  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485614&idx=1&sn=e77bb75bf2b9f82254ba13d818623981&chksm=f996186ccee1917a9653576874467a317e67cff5fb985d75b3a35dccb5f3fe43354f17341cc7#rd

---

❝
在云网络中，Overlay可以采用各种自定义的编码方式满足公有云的各种业务诉求。但是也有一些方法论的东西，例如可编程能力，源路由能力(SR)，快速收敛、流可视化(FlowID)、带内遥测(INT)。SRoU是针对这一系列问题构建的数据面工程实践
❞
本文采用基于Golang的伪代码的方式讲述Ruta的转发面行为，为SRoU RFC-Draft提供一个参考

当然针对不同的平台有各种工程实践，例如对Go比较熟悉的可以采用nff-go，或者eBPF、或者VPP的方式，甚至P4实现然后编译到Barefoot或者Xilinx网卡上。

### SRoU Header 

SRoU Header定义如下,这是最早第一版Draft的时候随手编码的，分为Common Header和SR Header两块

```
Common Header  0                   1                   2                   3  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ | Magic Number  |  SRoU Length  | Flow ID Length| Protocol-ID   | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                                                               | |                 Flow ID( Variable length)                     | |                                                               | |                                                               | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                        Source Address                         | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |      Source Port              | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+SR Header +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ | Segment Type  |  SR Hdr Len   | Last Entry    | Segments Left | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                                                               | |            Segment List[0] (length based on segment type)     | |                                                               | |                                                               | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                                                               | |                                                               |                               ... |                                                               | |                                                               | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                                                               | |            Segment List[N] (length based on segment type)     | |                                                               | |                                                               | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

下一阶段草案可能的改进:

`Cache Line`对齐的:可能考虑到一些软件平台的问题，将一些没有对齐的地方增加一些Padding，或者增加一些新的Instruction字段

`FlowID`:可能会变成定长的64bits，对很多硬件平台会更加友好，但是这个FlowID如何编码是一个值得研究的问题，特别是如何跟微分段整合。

`Source Address`: 主要用于IPv4场景中的NAT穿越，和云VPC上的一些源地址验证，SRv6的处理仅修改IP包头的目的地，源地址未变，但是uRPF检验会失效，因此也有将原有源地址Cache在SRoU Header内部的需求，只是它的位置是否应该排到SRH以后，还在做一些性能的对比测试。

### SRoU Header Encode 

通常我会构造一个结构体来保存SRoU的信息：

```
type SRoUHeader struct { Conn         net.PacketConn //发送报文的源Socket，用于 RemoteAddr   net.Addr       //Remote UDP address EVPNHdr      []byte         //预封装好的带EVPN业务的SRoU头 SRoUHdr      []byte         //预封装好的SRoU头 SRStartLoc   int            //SRH的起始位置 EVPNStartLoc int            //EVPN模式下的，VPNID填入操作时的起始位置}
```

Encode函数如下：

```
func EncodeSRHdr(path string, flowid []byte) (*SRoUHeader, error) {
```

path string来自于计算引擎，其格式为

Color|IP:port,Color|IP:port,Color|IP:port...，

接下来我们将其解析为sid list数组，并更新转发的目的地址RemoteAddr字段

```
  sidlist := strings.Split(path, ",") if len(sidlist) < 2 {  return nil, fmt.Errorf("Invalid Path length: %d , Path: %s", len(sidlist), path) } for k, v := range sidlist {  sid := strings.Split(v, "|")  if len(sid) != 2 {   return nil, fmt.Errorf("Invalid SID: %s", v)  } else {   sidlist[k] = sid[1]  } }    remoteAddr, err := net.ResolveUDPAddr("udp", sidlist[1]) if err != nil {  return nil, fmt.Errorf("Invalid SID: %s", sidlist[1]) } result := &SRoUHeader{  RemoteAddr: remoteAddr, }
```

然后构造Common Header

```
var commonHdr bytes.BufferprotoID := uint8(0x1) //IPv4srcAddr := StrToByte(sidlist[0])if len(srcAddr) == 18 { protoID = 0x2 //IPv6}flowidLen := uint8(0)//SRoU Common Headerif flowid != nil { flowidLen = uint8(len(flowid)) commonHdr.Write([]byte{SROU_MAGIC_NUM, 0x0, uint8(flowidLen), protoID}) commonHdr.Write(flowid) commonHdr.Write(srcAddr)} else { commonHdr.Write([]byte{SROU_MAGIC_NUM, 0x0, uint8(flowidLen), protoID}) commonHdr.Write(srcAddr)}
```

更新SRoU Header的Start Location和根据SIDList编码SRH

```
result.SRStartLoc = commonHdr.Len()segLength := uint8(len(sidlist) - 1) //sidlist[0] is source addrvar pathsid bytes.Bufferfor i := len(sidlist) - 1; i > 0; i-- { pathsid.Write(StrToByte(sidlist[i]))}var SRoUHdr bytes.BufferSRoUHdr.Write(commonHdr.Bytes())SRoUHdr.Write([]byte{protoID, uint8(pathsid.Len()), segLength - 1, segLength - 1})SRoUHdr.Write(pathsid.Bytes())result.SRoUHdr = SRoUHdr.Bytes()result.SRoUHdr[1] = uint8(len(result.SRoUHdr))
```

针对EVPN场景，需要填充一个255..<END.X>的SID

```
var EVPNHdr bytes.Buffer EVPNHdr.Write(commonHdr.Bytes()) if protoID == 0x1 {  result.EVPNStartLoc = commonHdr.Len() + 5  EVPNHdr.Write([]byte{protoID, uint8(pathsid.Len() + 6), segLength, segLength})  EVPNHdr.Write([]byte{0xFF, 0x0, 0x0, 0x0, 0x0, 0x0}) } else {  //TODO: IPv6 mode EVPN header  EVPNHdr.Write([]byte{protoID, uint8(pathsid.Len()), segLength, segLength}) } EVPNHdr.Write(pathsid.Bytes())  result.EVPNHdr = EVPNHdr.Bytes() result.EVPNHdr[1] = uint8(EVPNHdr.Len()) return result, nil  
```

### SRLOC Route 

如上的函数在路由更新的时候执行，通常控制面会下发一条路由定义源节点和目的路径,转发平面有一个带timeout功能的sync.Map用于存放路由信息和预封装好的路径信息，FlowID这里随手写了一个DATA用于draft

```
func (f *FIBMgr) UpdateSRLOCRoute(key, value string, dp *dplane.DataPlane) error {path := strings.Split(value, ">>")srcnode := path[0]fwdsock, valid := dp.SocketMap[srcnode]if !valid { return fmt.Errorf("invalid path calculation, source not found")}h, err := lib.EncodeSRHdr(path[1], []byte("DATA"))if err != nil { return err}h.Conn = fwdsock.Connf.SRLOCRoute.Store(key, h, time.Now())return nil}
```

### 转发面函数 

当从UDP Socket接收到一个报文后，采用如下函数处理验证和Parse Common Header

```
func forwarder(d *dplane.DataPlane, fm *forward.FIBMgr, f *dplane.FWDSocket, addr net.Addr, buf []byte) { //packet payload less than min size   if len(buf) < 10 {  return } //validate magic number if buf[0] != SROU_MAGIC_NUM {  return } //invalidate length SRoULen := uint8(buf[1]) if len(buf) < int(SRoULen) {  return }    //TODO: Add algorithm to validate flowid //buf[2]=flowID Length srcAddrLoc := uint8(buf[2]) + 4    //validate srcAddrLoc if srcAddrLoc > SRoULen {  return } af := uint8(buf[3])
```

当af=0时，为OAM消息处理，其逻辑主要为STUN处理和链路状态Probe的处理:

```
//OAM Message proccessing if af == 0 {  now := time.Now()  rxTS := make([]byte, 8)  binary.BigEndian.PutUint64(rxTS, uint64(now.UnixNano()))  oamType := buf[srcAddrLoc]  //LinkState Packet Recieved.  if oamType == 0 {   lsType := buf[srcAddrLoc+1]   //Recieve PM_REQ message   if lsType == 0 {    sseq := binary.BigEndian.Uint32(buf[srcAddrLoc+4 : srcAddrLoc+8])    lseq := uint32(0)    //Check local database counter    lseqT, exist := f.PerfMeasureTable.Load(addr.String())    if !exist {     f.PerfMeasureTable.Store(addr.String(), lseq, now)    } else {     lseq = lseqT.(uint32)    }    if int(lseq)-int(sseq) > 50 || sseq == 0 {     lseq = 0    }    tmp := bytes.NewBuffer(buf)    tmp.Write(rxTS)                              //recieved TS    tmp.Write(buf[srcAddrLoc+4 : srcAddrLoc+16]) //copy sendSeq/TS    pkt := tmp.Bytes()    pkt[1] = uint8(len(pkt))     //update SRoU length    pkt[srcAddrLoc+1] = uint8(1) //modify pmtype    binary.BigEndian.PutUint32(pkt[srcAddrLoc+4:srcAddrLoc+8], lseq+1)    now = time.Now()    binary.BigEndian.PutUint64(pkt[srcAddrLoc+8:srcAddrLoc+16], uint64(now.UnixNano()))    _, err := f.Conn.WriteTo(pkt, addr)    if err == nil {     f.PerfMeasureTable.Store(addr.String(), lseq+1, now)    }   }   //Recieve PM_RESP Message   if lsType == 1 {    t4 := uint64(now.UnixNano())    rseq := binary.BigEndian.Uint32(buf[srcAddrLoc+4 : srcAddrLoc+8])    t3 := binary.BigEndian.Uint64(buf[srcAddrLoc+8 : srcAddrLoc+16])    t2 := binary.BigEndian.Uint64(buf[srcAddrLoc+16 : srcAddrLoc+24])    sseq := binary.BigEndian.Uint32(buf[srcAddrLoc+24 : srcAddrLoc+28])    t1 := binary.BigEndian.Uint64(buf[srcAddrLoc+28 : srcAddrLoc+36])    result := &pm.PM_RESP{     RSeq: rseq,     SSeq: sseq,     T1:   t1,     T2:   t2,     T3:   t3,     T4:   t4,    }    sess, valid := f.PMSession[addr.String()]    if valid {     if sess != nil && sess.PMChan != nil {      sess.PMChan <- result     }    }   }  }                             //Send STUN packet to control plane.  if oamType == 2 || oamType == 4 {   pkt := &fman.Packet{    PktType:    fman.PKT_STUN,    SockID:     f.SockID,    RemoteAddr: addr,    Length:     len(buf),    Content:    buf,   }   f.PktChanToCP <- pkt   return  }  //drop other OAM msg type  return }
```

SRoU的Relay转发很简单

```
 //SRoU forwarding var segLen uint8 = 6 if af == 2 {  segLen = 18 } //srh start pointer srhLoc := srcAddrLoc + segLen //Normal forwarding case segmentLeft := uint8(buf[srhLoc+3]) lastEntry := buf[srhLoc+2] if segmentLeft == lastEntry {  //NAT ALG applied on 1st hop  segment := lib.StrToByte(addr.String())  copy(buf[srcAddrLoc:srhLoc], segment) } //reduce segmentleft buf[srhLoc+3]-- if segmentLeft > 1 {  start := srhLoc + 4 + (segmentLeft-1)*segLen  dst := buf[start : start+segLen]  addr1 := lib.ByteToNetAddr(dst)  f.Conn.WriteTo(buf, addr1) }
```

当SegmentLeft==1时可能需要触发EVPN的查询：

```
 //segmentLeft == 1, last sid is EVPN action table if segmentLeft == 1 {  start := srhLoc + 4  if buf[start] == 0xFF {   vnid := binary.BigEndian.Uint32([]byte{0x0, buf[start+1], buf[start+2], buf[start+3]})   endX := binary.BigEndian.Uint16(buf[start+4 : start+6])   if DEBUG {    logrus.Info("EVPN Forwarding:", vnid, "|", endX)    fmt.Println(hex.Dump(buf))   }   if endX == 0x0017 {    //End.DT2U, check mac table    data := buf[SRoULen:]    //    DataPtr := (unsafe.Pointer)(&data[0])    PayloadLength := uint(len(data))    eth, err := ethernet.Decode(DataPtr, PayloadLength)    if err != nil {     logrus.Info("Decode Ethernet Layer error:", err)     return    }    fib, valid := fm.FIBMap[vnid]    if !valid {     return    }    adj := fib.MACLookup(eth.DstAddr)    if adj != nil {     if len(adj.LocalPort) == 0 {      return     }     //TODO: Hash later     port := adj.LocalPort[0]     port.Handle.WritePacketData(data)    }   }   if endX == 0x0013 {    data := buf[SRoULen:]    DataPtr := (unsafe.Pointer)(&data[0])    PayloadLength := uint(len(data))    layerIPv4, err := ipv4.Decode(DataPtr, PayloadLength)    if err != nil {     logrus.Info("Decode IPv4 Layer error:", err)     return    }    fib, valid := fm.FIBMap[vnid]    if !valid {     return    }    dstadjT, err := fib.L3.FindIP(layerIPv4.DstAddr)    if err != nil {     return    }    if dstadjT == nil {     return    }    adj := dstadjT.(*forward.Adjacency)    switch adj.Type {    case forward.ADJ_LOCAL_RECIEVED:     if layerIPv4.NextProto == types.PROTOCOL_ICMP {      response := forward.ICMPResponseDataIPv4(data)      adjT, err := fib.L3.FindIP(layerIPv4.SrcAddr)      if err != nil {       return      }      if adjT == nil {       return      }      adj := adjT.(*forward.Adjacency)      if adj.Type != forward.ADJ_REMOTE_SRLOC {       return      }      if len(adj.SRLOC) == 0 {       return      }      srloc := adj.SRLOC[0]      srouRouteT, ok := fm.SRLOCRoute.Load(srloc)      if !ok {       return      }      srouRoute := srouRouteT.(*lib.SRoUHeader)      var buf bytes.Buffer      buf.Write(srouRoute.EVPNHdr)      buf.Write(response)      outputData := buf.Bytes()      vnidByte := lib.VNIDToByte(vnid)      outputData[srouRoute.EVPNStartLoc] = vnidByte[0]      outputData[srouRoute.EVPNStartLoc+1] = vnidByte[1]      outputData[srouRoute.EVPNStartLoc+2] = vnidByte[2]      //End.DT4: 0x0013      outputData[srouRoute.EVPNStartLoc+3] = 0x00      outputData[srouRoute.EVPNStartLoc+4] = 0x13      srouRoute.Conn.WriteTo(outputData, srouRoute.RemoteAddr)      return     }    case forward.ADJ_LOCAL_CONNECTED:     dmac := fib.ARPLookup(layerIPv4.DstAddr)     if dmac == nil {      //TODO: Add ARP Request support later      return     }     smac := adj.LocalPort[0].MAC     buf := bytes.NewBuffer(dmac)     buf.Write(smac)     buf.Write([]byte{0x08, 0x00})     buf.Write(data)     adj.LocalPort[0].Handle.WritePacketData(buf.Bytes())    case forward.ADJ_LOCAL_ATTACHED:     dmac := fib.ARPLookup(adj.DstIP[0])     smac := adj.LocalPort[0].MAC     if dmac == nil {      //TODO: Add ARP Request support later      return     }          buf := bytes.NewBuffer(dmac)     buf.Write(smac)     buf.Write([]byte{0x08, 0x00})     buf.Write(data)     adj.LocalPort[0].Handle.WritePacketData(buf.Bytes())    }   }  } } //segmentLeft == 0 case is used for SRoU interworking with native udp socket
```

关于上文中的Adjacency Type定义如下,这部分主要是和EVPN interworking时用的

```
 ADJ_LOCAL_CONNECTED AdjacencyTYPE = iota //Type:0 Local Switching ADJ_LOCAL_RECIEVED                      //Type:1 Local Interface like Anycast gateway ADJ_LOCAL_ATTACHED                      //Type:2 Local L3 Routed(Local RIB/Redistributed/Static) ADJ_REMOTE_SRLOC                        //Type:3 Packet require to send to remote by SRoU-DP 
```

现阶段的转发逻辑，收到一个SRoU报文

```
  ---------SRoU Recieved ----->  End.DT2U:	  L2FIB lookup---> Adj.Type0 ---> fwd  End.DT4	  L3FIB LPM lookup ---> ADJ_LOCAL_RECIEVED ---> Handle ICMP only				ADJ_LOCAL_CONNECTED ---> Attached Route, Trigger MAC lookup				ADJ_LOCAL_ATTACHED  ---> Nexthop IP in SRLOC[Local Static Route Configured]				ADJ_REMOTE_SRLOC    ---> Wrong, Source should use SRoU direct send to remote SRLOC, drop
```

如果时从Tunnel收到的报文，同End.DT4做L3FIB的LPM查询转发,如果本地接口为一个二层口，则需要按照如下流程转发

```
L2FIB Lookup----DMAC =-------------------> ADJ_LOCAL_CONNECTED:  Local Switch			   ADJ_LOCAL_RECIEVED:  anycast gateway			                    ----->L3FIB Lookup------>						 ADJ_LOCAL_CONNECTED ------>Route to another port						 ADJ_LOCAL_RECIEVED------> Another Anycast GW ? [Support in next phase]						 ADJ_REMOTE_SRLOC ------> Remote SRLOC: encap send ...						 ADJ_LOCAL_ATTACHED ------> Another Local port
```