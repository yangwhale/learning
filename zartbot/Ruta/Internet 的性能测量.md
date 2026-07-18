# Internet 的性能测量

> 作者: zartbot  
> 日期: 2021年4月10日 13:35  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485629&idx=1&sn=753fbb174f3dffd7c05474d608d24952&chksm=f996187fcee19169d18aac2a4389e3ddafdc35bd991cbab76fae1430b10ea717f48fe3aba584#rd

---

❝
很多人测量延迟只会Ping,包括某些大厂做流量工程的时候,实在是看不下去了，在此继续技术扶贫一下,让你们明白你们既不懂通信，又不懂代码，也不懂算法，还不懂数据分析。发出来的目的就是让你们别一天到晚吹牛x窝里横.
❞
今天提前几个小时做完一个好玩的东西，某个新算法效果不错,很多人以为哥是玩Marketing的，大错特错...

![图片](assets/542b99a87f76.png)

另外，又提前几天预测了A股的风险，这次又没失手，嘻嘻，好吧回到正题...

延迟测量有一个标准协议，RFC5357：A Two-Way Active Measurement Protocol (TWAMP).但是这玩意设计的太复杂，而且很多时候也不适合测试Internet的性能，因为互联网有它的特殊性，而某些搞SDWAN的大厂连测都测不准，还谈什么优化呢？

### 延迟测量理论 

我们可以简单的构造如下报文:

```
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                       Sequence Number                         | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                         TimeStamp                             | |                                                               | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                     Recieved TimeStamp                        | |                                                               | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                    Sender Sequence Number                     | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ |                       Sender TimeStamp                        | |                                                               | +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

在Ruta中还有一个Common header来定义,即Protocol-ID字段为0时代表Ruta SRoU Payload为一个OAM消息,紧接着有一个字段定义是PerformanceMeasurement Request/Response(PM_Req/PM_Resp).

```
  oamType := buf[srcAddrLoc]  //LinkState Packet Recieved.  if oamType == 0 {   lsType := buf[srcAddrLoc+1]   //Recieve PM_REQ message   if lsType == 0 {   ...   //Recieve PM_RESP Message   if lsType == 1 {   ...
```

### PM_Req报文封装 

```
func ReqPktEncode(flowid []byte, seq uint32) []byte { buf := bytes.NewBuffer([]byte{0, 0, 0, 0}) //SRoU common header flowidLen, _ := buf.Write(flowid)   //PM packet space buf.Write(make([]byte, 16)) //4B + 4B Seq + 8B TS var startLoc int = 8 + flowidLen pkt := buf.Bytes() pkt[1] = uint8(len(pkt)) pkt[2] = uint8(flowidLen) binary.BigEndian.PutUint32(pkt[startLoc:startLoc+4], seq) binary.BigEndian.PutUint64(pkt[startLoc+4:startLoc+12], uint64(time.Now().UnixNano())) return pkt}
```

发送PM_Req时采用一个Goroutine的函数,其中Describe库是用于做统计描述的一个库，稍后讲.另外Prob的Internval参数是艺术，不告诉你们

```
//SendPMReq is used to send PM_Req Messagefunc (s *Session) SendPMReq(Interval time.Duration) { atomic.StoreUint32(s.CurrentSeq, 0) atomic.StoreUint32(s.RecvCnt, 0) s.Lock.Lock() s.RTT = describe.New() s.S2D = describe.New() s.D2S = describe.New() s.SSeq = 0 s.RSeq = 0 s.Lock.Unlock() for i := uint32(0); i < PROBE_NUM; i++ {  en := ReqPktEncode([]byte("TWAMP"), i)  s.Conn.WriteTo(en, s.raddr)  atomic.AddUint32(s.CurrentSeq, 1)  time.Sleep(Interval)  //TODO: Add random seed later        //process stop signal, just use atomic counter :)  if atomic.LoadUint32(s.sig) == 1 {   break  } }}
```

### 响应PM_Req 

主要是通过数据面直接响应,记录接收到的Timestamp，并更新本地的counter

```
 //Recieve PM_REQ message   if lsType == 0 {    sseq := binary.BigEndian.Uint32(buf[srcAddrLoc+4 : srcAddrLoc+8])    lseq := uint32(0)    //Check local database counter    lseqT, exist := f.PerfMeasureTable.Load(addr.String())    if !exist {     f.PerfMeasureTable.Store(addr.String(), lseq, now)    } else {     lseq = lseqT.(uint32)    }    if int(lseq)-int(sseq) > 50 || sseq == 0 {     lseq = 0    }    tmp := bytes.NewBuffer(buf)    tmp.Write(rxTS)                              //recieved TS    tmp.Write(buf[srcAddrLoc+4 : srcAddrLoc+16]) //copy sendSeq/TS    pkt := tmp.Bytes()    pkt[1] = uint8(len(pkt))     //update SRoU length    pkt[srcAddrLoc+1] = uint8(1) //modify pmtype    binary.BigEndian.PutUint32(pkt[srcAddrLoc+4:srcAddrLoc+8], lseq+1)    now = time.Now()    binary.BigEndian.PutUint64(pkt[srcAddrLoc+8:srcAddrLoc+16], uint64(now.UnixNano()))    _, err := f.Conn.WriteTo(pkt, addr)    if err == nil {     f.PerfMeasureTable.Store(addr.String(), lseq+1, now)    }   }
```

#PM_Resp处理

当收到远端响应的PM_Resp时，做如下处理,类似于TWAMP有四个时间戳，T1~T4，然后还有发送seq和接收seq，整体一起传到控制面

```
  //Recieve PM_RESP Message   if lsType == 1 {    t4 := uint64(now.UnixNano())    rseq := binary.BigEndian.Uint32(buf[srcAddrLoc+4 : srcAddrLoc+8])    t3 := binary.BigEndian.Uint64(buf[srcAddrLoc+8 : srcAddrLoc+16])    t2 := binary.BigEndian.Uint64(buf[srcAddrLoc+16 : srcAddrLoc+24])    sseq := binary.BigEndian.Uint32(buf[srcAddrLoc+24 : srcAddrLoc+28])    t1 := binary.BigEndian.Uint64(buf[srcAddrLoc+28 : srcAddrLoc+36])    result := &pm.PM_RESP{     RSeq: rseq,     SSeq: sseq,     T1:   t1,     T2:   t2,     T3:   t3,     T4:   t4,    }    sess, valid := f.PMSession[addr.String()]    if valid {     if sess != nil && sess.PMChan != nil {      sess.PMChan <- result     }
```

控制面采用如下方法计算：

```
func (s *Session) RecvChan() { for {  select {  case <-s.StopChan:   //logrus.Info("Stop Signaling recieve")   atomic.StoreUint32(s.sig, 1)   break  case msg := <-s.PMChan:   if msg != nil {    atomic.AddUint32(s.RecvCnt, 1)    rtt := math.Max(0, float64(msg.T4-msg.T1-(msg.T3-msg.T2)))    s2d := math.Max(0, float64(msg.T2-msg.T1))    d2s := math.Max(0, float64(msg.T4-msg.T3))    //ignore time out-of-sync case    if s2d > rtt {     s2d = 0    }    //ignore time out-of-sync case    if d2s > rtt {     d2s = 0    }    s.Lock.Lock()    if msg.RSeq > s.RSeq {     s.RSeq = msg.RSeq    }    if msg.SSeq > s.SSeq {     s.SSeq = msg.SSeq    }    s.RTT.Append(rtt, 2) //2：二阶矩，仅统计标准差，不统计偏度和峰度    s.S2D.Append(s2d, 2)    s.D2S.Append(d2s, 2)    s.Lock.Unlock()   }  } }}
```

日志汇总,主要是双向丢包率的统计和异常case的过滤，这里吐槽一下公有云，几乎每家NTP都是一坨屎...

```
if atomic.LoadUint32(s.CurrentSeq) == PROBE_NUM {  totalLoss := (float64(atomic.LoadUint32(s.CurrentSeq)) - float64(atomic.LoadUint32(s.RecvCnt))) * 100 / float64(PROBE_NUM)  s2dLoss := (float64(s.SSeq+1) - float64(s.RSeq)) * 100 / float64(PROBE_NUM)  d2sLoss := (float64(s.RSeq) - float64(atomic.LoadUint32(s.RecvCnt))) * 100 / float64(PROBE_NUM)  if d2sLoss < 0 || d2sLoss > 100 {   d2sLoss = 0  }  if s2dLoss < 0 || d2sLoss > 100 {   s2dLoss = 0  }  if totalLoss < 0 || d2sLoss > 100 {   totalLoss = 0  }  upstate := uint8(1)  if atomic.LoadUint32(s.RecvCnt) == 0 {   upstate = 0   totalLoss = 100   s2dLoss = 100   d2sLoss = 100  }  stats := &fman.PM_STATS{   SRC:        s.LocalSRLOC.String(),   DST:        s.RemoteSRLOC.String(),   LinkState:  upstate,   RTT_Avg:    s.RTT.Mean,   RTT_Min:    s.RTT.Min,   RTT_Max:    s.RTT.Max,   RTT_Jitter: s.RTT.Std(),   S2D_Avg:    s.S2D.Mean,   S2D_Min:    s.S2D.Min,   S2D_Max:    s.S2D.Max,   S2D_Jitter: s.S2D.Std(),   D2S_Avg:    s.D2S.Mean,   D2S_Min:    s.D2S.Min,   D2S_Max:    s.D2S.Max,   D2S_Jitter: s.D2S.Std(),   Loss:       totalLoss,   S2D_Loss:   s2dLoss,   D2S_Loss:   d2sLoss,   TimeStamp:  uint64(time.Now().UnixNano()),   S2DBW:      math.Min(s.LocalSRLOC.TXBW, s.RemoteSRLOC.RXBW),   D2SBW:      math.Min(s.LocalSRLOC.RXBW, s.RemoteSRLOC.TXBW),  }  sjson, err := json.Marshal(stats)  if err != nil {   logrus.Warn("JSON Marshal Error:", err, "["+stats.Sprint()+"]")   return  }  msg := &fman.Message{   Type:    fman.MSG_PERF_MEASURE,   Action:  fman.MSG_ACTION_RESPONSE,   Key:     fmt.Sprintf("%s->%s", s.LocalSRLOC.EncodeSRLOC(), s.RemoteSRLOC.EncodeSRLOC()),   Content: string(sjson),  }  s.CtrlChanToRP <- msg //最后会发送上报到ETCD }
```

### 统计库 

反正很多人也看不懂，懒得解释了,自己抄作业就好了。

```
package describeimport ( "math" "sync")var ONLINE_STATS_SYNC_POOL *sync.Pool//Item is used to store stats valuetype Item struct { N    float64 Min  float64 Max  float64 Mean float64 M2   float64 M3   float64 M4   float64}type Result struct { Count float64 Min   float64 Max   float64 Mean  float64 Std   float64 Skew  float64 Kurt  float64}func init() { ONLINE_STATS_SYNC_POOL = &sync.Pool{New: func() interface{} {  return &Item{   N:    0,   Min:  0,   Max:  0,   Mean: 0,   M2:   0,   M3:   0,   M4:   0,  } }}}func New() *Item { return ONLINE_STATS_SYNC_POOL.Get().(*Item)}//Append is used to store new data for stats, highOrder is in range 2~4 for X^2~4 statsfunc (i *Item) Append(x float64, highOrder uint8) { n1 := i.N if n1 == 0 {  i.Min = x  i.Max = x } if x < i.Min {  i.Min = x } if x > i.Max {  i.Max = x } i.N = i.N + 1 delta := x - i.Mean deltaN := delta / i.N deltaN2 := deltaN * deltaN term1 := delta * deltaN * n1 i.Mean = i.Mean + deltaN switch highOrder { case 4:  i.M4 = i.M4 + term1*deltaN2*(i.N*i.N-3*i.N+3) + 6*deltaN2*i.M2 - 4*deltaN*i.M3  i.M3 = i.M3 + term1*deltaN*(i.N-2) - 3*deltaN*i.M2  i.M2 = i.M2 + term1 case 3:  i.M3 = i.M3 + term1*deltaN*(i.N-2) - 3*deltaN*i.M2  i.M2 = i.M2 + term1 case 2:  i.M2 = i.M2 + term1 default: }}//Len is return data countfunc (i *Item) Len() float64 { return i.N}//Sum is return data sumfunc (i *Item) Sum() float64 { return i.Mean * i.N}//Variance is return the variancefunc (i *Item) Variance() float64 { if i.N < 2 {  return float64(0.0) } else {  return i.M2 / i.N }}//Std is return the stdfunc (i *Item) Std() float64 { if i.N < 2 {  return float64(0.0) } else {  return math.Sqrt(i.M2 / i.N) }}//Skewness is return skewness...func (i *Item) Skewness() float64 { if i.M2 < 1e-14 || i.N <= 3 {  return float64(0.0) } else {  return math.Sqrt(i.N) * i.M3 / i.M2 / math.Sqrt(i.M2) }}//Kurtosis is return kurtosisfunc (i *Item) Kurtosis() float64 { if i.M2 < 1e-14 || i.N <= 4 {  return float64(0.0) } else {  return (i.N*i.M4)/(i.M2*i.M2) - 3 }}
```

### 分析 

然后从ETCD Watch这些Key转存为csv,再换到Python/Jupyter里做进一步的分析，可视化工具就比较多了，我通常的做法是很多时序线图或者Stack bar这样的图采用Bokeh,而heatmap类的直接就pandas sytle染色了，当然matplotlib有些时候也会用一点，主要是比较懒的时候。通常我也喜欢把Jupyter的屏幕宽度调到100%屏幕宽度，这样干活舒服点。

```
import pandas as pdimport numpy as npimport matplotlibimport matplotlib.pyplot as plt%matplotlib inline  from IPython.core.display import display, HTMLdisplay(HTML("<style>.container { width:100% !important; }</style>"))
```

然后数据基本上就是一个CSV格式的，pandas read一下就好了，但我比较喜欢把它转存为pickle，因为下次再用的时候加载快，而延迟相关的数据我们都采用ns uint64记录的，所以算出来的delta值也是ns，这里统一改为ms

```
df = pd.read_pickle("./srou.dat")#convert ns delta to msfor item in ['rtt_avg', 'rtt_min', 'rtt_max', 'jitter','s2d_avg', 's2d_min', 's2d_max', 's2d_jitter','d2s_avg', 'd2s_min', 'd2s_max', 'd2s_jitter']:    df[item]= df[item]/1000000
```

这样处理还有一些问题，过滤处理起来很麻烦，例如要选择某个时间段或者某个源/目的节点的信息，干脆外面包一层把一些常见的过滤条件都包起来弄成几个常用的函数就好了，其实本来应该弄一个新的class去加一些方法的，还是一个字：懒，够用就行了。

```
def filter_src(df,src):    return df[df['src'].str.contains(src)]def filter_dst(df,dst):    return df[df['dst'].str.contains(dst)]def filter_time(df,starttime,endtime):    return df[(df['datetime']>starttime )& (df['datetime']<endtime )]def filter_link(df,src,dst):    return filter_dst(filter_src(df,src),dst)def filter_link_time(df,src,dst,starttime,endtime):    return filter_link(filter_time(df,starttime,endtime),src,dst)def filter_src_time(df,src,starttime,endtime):    return filter_src(filter_time(df,starttime,endtime),src)def filter_dst_time(df,dst,starttime,endtime):    return filter_dst(filter_time(df,starttime,endtime),dst)
```

画一些timeseries stacked bar的函数如下：

```
import bokeh.plottingimport bokeh.modelsimport bokeh.layoutsimport bokeh.palettesfrom bokeh.resources import INLINEimport bokeh.iobokeh.io.output_notebook(INLINE)def bokeh_datetime_vbar_stack_chart(df,stacked_value,stack_name,title,x_axis='datetime',width=1900,height=600):    lines_counter = len (stacked_value)if(lines_counter <=10):        color_list=bokeh.palettes.Category10[10]else:        color_list=bokeh.palettes.Category20[20]    fig =  bokeh.plotting.figure(x_axis_type="datetime", plot_width=width,plot_height=height, title=title)#,toolbar_location=None, tools="hover")    fig.vbar_stack(stacked_value, x=x_axis, width=0.5, color=color_list[0:lines_counter], source=df,legend_label=stack_name)    fig.y_range.start = 0    #fig.x_range.range_padding = 0.1    #fig.xgrid.grid_line_color = None    fig.axis.minor_tick_line_color = None    fig.outline_line_color = None    fig.legend.location = "top_left"    fig.legend.orientation = "horizontal"    return fig    def describe(df):    data= df.groupby('datetime').mean().reset_index()    chart2 =bokeh_datetime_vbar_stack_chart(data,['d2s_avg','s2d_avg'],['d2s','s2d'],title="Latency",height=400)    chart3 =bokeh_datetime_vbar_stack_chart(data,['d2s_jitter','s2d_jitter'],['d2s','s2d'],title="Jitter",height=400)    chart4 =bokeh_datetime_vbar_stack_chart(data,['d2s_loss','s2d_loss'],['d2s','s2d'],title="Loss",height=400)    bokeh.plotting.show(bokeh.layouts.column([chart2,chart3,chart4]))
```

我们来看第一组数据，所有数据的平均值：

![图片](assets/e4beb1a51c12.png)

很多时候我们的直觉又让我们开始犯错误了，平均值看，丢包率10%左右，还有明显的周期性，很多人或许就满足了，然后这一步就停止继续工作了，例如thousandeye只按照天平均，魔鬼都在细节中~

例如我们看从腾讯东京到阿里杭州这样的链路

```
describe(filter_link(data,"TE_Tokoyo","ALI_Hangzhou"))
```

https://mmbiz.qpic.cn/mmbiz_png/9v5mpBibQrkgocdbL1ouic5pVLhgibDiaiaEJoMdz4a2xbnwgcaqbk7UMGOKBqeT5KUEHjEpJTTwNlS1pmJkfU6fFuw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1

可以注意到在繁忙时段，延迟是稳定的，延迟下降基本上都是每天半夜队列空了导致的，抖动基本上出现在半夜空闲时段，也就是队列不满不空的中间态，而丢包率每天的峰值会达到40%左右，并且具有很明显的时间周期特征，丢弃也是单方向的，主要是入境流量的丢弃。

所以前面求overall平均看到的丢包率10%就有很大的偏差了吧，这也就是很多人处理数据不细心导致的。

既然闲时和忙时丢包率差距这么大，又具有周期性，我们能干点什么吗？这就是Ruta选路的精髓了。首先我们根据时间轴来分片取样,利用pandas的Pivot_table

```
import seaborn as snsdef pivot_table(df,value):    t2=df.groupby([df['dst'],data['src']]).agg(['mean','median','std'])    t2.columns =[ "_".join(x) for x in t2.columns.ravel()]    t3=t2.iloc[:,0:].reset_index()    return t3.pivot_table(index=['src'],columns='dst')[value]def pivot_loss_table(df,value,cmap):    t4= pivot_table(df,value)    return t4.fillna(0).style.background_gradient(cmap=cmap).format("{:3.2f}%")    def pivot_delta_table(df1,df2,value,cmap):    t1= pivot_table(df1,value)    t2= pivot_table(df2,value)    return (t1-t2).fillna(0).style.background_gradient(cmap=cmap,axis=1).format("{:3.2f}%")
```

例如我们选择如下两个时间段进行比对：

```
busynight=filter_time(df,"2020-08-19 21:00:00",'2020-08-19 22:00:00')deepnight=filter_time(df,"2020-08-20 03:00:00",'2020-08-20 04:00:00')pivot_loss_table(busynight,'s2d_loss_mean',sns.light_palette("orange",as_cmap=True))
```

忙时链路丢包率：

![图片](assets/fb7b5b024016.png)

可以看到的是国内的节点即便是在忙时，出境方向上的流量丢包率并不大，而入境方向上普遍存在丢包，你可以明显的看到竖着的几列(阿里杭州，阿里呼和浩特，腾讯北京，腾讯重庆，腾讯广州），同时你也需要注意到，阿里成都和阿里呼和浩特，腾讯广州重庆北京到腾讯旧金山和腾讯新加坡节点，都有一些质量非常好的少丢包或者零丢包线路。

空闲时段丢包率如下，基本上你可以额看到半夜时分，阿里伦敦/阿里硅谷/腾讯多伦多到腾讯重庆腾讯广州的链路还是维持在较高的丢包水平，其它链路几乎都变得非常空闲。

![图片](assets/1dabcd756cdd.png)

具体的差值我们也可以看一下：

```
pivot_delta_table(busynight,deepnight,'s2d_loss_mean',sns.light_palette("green",as_cmap=True))
```

![图片](assets/4c93c9c5170c.png)

可以注意到入境流量在深夜都有很大的提升。然后我们来看看延迟，延迟的函数类似，就是包一层样式，把丢包的%改为ms单位

```
def pivot_latency_table(df,value,cmap):    t4= pivot_table(df,value)    return t4.fillna(0).style.background_gradient(cmap=cmap).format("{:3.2f}ms")def pivot_delta_latency_table(df1,df2,value,cmap):    t1= pivot_table(df1,value)    t2= pivot_table(df2,value)    return (t2-t1).fillna(0).style.background_gradient(cmap=cmap,axis=1).format("{:3.2f}ms")pivot_latency_table(busynight,'s2d_avg_mean',sns.light_palette("navy",as_cmap=True))    
```

![图片](assets/faf7c82e2e55.png)

而深夜的闲时链路也还好:

![图片](assets/e66e097e986e.png)

接下来还是看看delta,变化最大的是各个云到成都的这条线路：

![图片](assets/0cfee141f204.png)

### BGP分析 

那么我们来看看成都这个地址段47.108.0.0具体的BGP路由表其实可以通过RouteViews project (www.routeviews.org)去查看下载BGP路由表，然后通过bgpdump查看即可：

```
kevin@PeentosM:~/srou/rib$ ./bgpdump -m rib3.bz2 | grep "|47.108.0.0/"2020-08-25 09:15:06 [info] logging to syslogTABLE_DUMP2|1597996804|B|195.208.112.161|3277|47.108.0.0/15|3277 3267 1299 4837 37963|IGP|195.208.112.161|0|0|3277:39710|NAG||TABLE_DUMP2|1597996804|B|208.51.134.255|3549|47.108.0.0/15|3549 3356 4134 58461 37963|IGP|208.51.134.255|0|0|3356:2 3356:22 3356:100 3356:123 3356:500 3356:2106 3549:2018 3549:30840|NAG||TABLE_DUMP2|1597996804|B|162.251.163.2|53767|47.108.0.0/15|53767 3257 4837 37963|IGP|162.251.163.2|0|0|3257:6520 3257:8044 3257:30084 3257:50002 3257:51200 3257:51201 53767:5000|NAG||....
```

然后我们把这个CSV导入到Jupyter中

```
bgproute = pd.read_csv("./bgp.csv",delimiter="|")
```

![图片](assets/94b5ad1d7eeb.png)

利用NetworkX这个库分析，画图，就可以呈现ThousandEyes的BGP连接图了,是不是非常容易呀~

```
import networkx as nxG =nx.Graph()for item in bgproute['ASPATH']:    itemlist=str(item).split(" ")    for i in range(0,len(itemlist)-1):        G.add_edge(itemlist[i],itemlist[i+1])plt.figure(figsize=(20,13))nx.draw(G,with_labels=True, node_size=3000,node_color="navy",font_color="white",node_shape="h",font_weight="bold")plt.show()
```

![图片](assets/c5daec085dcb.png)

然后光看这些数不预测可不行，例如云资源都要花钱，但是很多时候如果有可行解的情况下做SRoU，没有可行解的时候再调度云资源付费加速不更好么？所以肯定就存在一个预测的处理，当然很多人会想到时间序列分析来拟合咯...例如通过statsmodel的内置的seasonal decompose：

```
tsa_data = filter_link(data,'TE_Tokoyo','ALI_Hangzhou')from statsmodels.tsa.seasonal import seasonal_decomposedecomposition = seasonal_decompose(tsa_data['s2d_loss'],freq=288)tsa_data['loss_seasonal']= decomposition.seasonaltsa_data['loss_trend'] = decomposition.trendtsa_data['loss_residual'] = decomposition.residchart2 =bokeh_multi_line_chart(tsa_data,['loss_seasonal'],['周期性'],title='周期')chart3 =bokeh_multi_line_chart(tsa_data,['loss_trend'],['延迟'],title='趋势')chart4 =bokeh_multi_line_chart(tsa_data,['loss_residual'],['残差'],title='残差')show_column([chart2,chart3,chart4])
```

![图片](assets/4accbe59763c.png)

趋势项基本上稳定在22附近，周末会有所下降，周期性也很明显的刻画了忙时和闲时的情况，残差相对较小，基本上可以接受这个数据，然后我们继续用facebook的prophet：

```
from fbprophet import Prophettsa_data['ds'] = tsa_data['datetime']tsa_data['y'] = tsa_data['s2d_loss']m = Prophet(changepoint_prior_scale=0.01).fit(tsa_data)future = m.make_future_dataframe(periods=96,freq='H')fcst = m.predict(future)fig = m.plot(fcst)
```

![图片](assets/4ca7fe444cf3.png)

当然原始数据刚好只有几天，结尾值又恰逢周末丢包变少，所以最终prophet预测出一个缓慢下降的趋势。这个错误是原始数据测量周期太短导致的。