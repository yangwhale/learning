# Ruta for Telemetry

> 作者: zartbot  
> 日期: 2020年9月20日 14:10  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247484313&idx=5&sn=c58cc0796e2f63de7ee47c12f9c5358e&chksm=f996135bcee19a4dcfbdd4a63f47ad65d3f183a45150cccb74a8d0a8cab508212c138bd664d9#rd

---

本来只是给Ruta的去中心化协议做的linkstate database，轻松的做出了Thousandeye的效果，这可是某司花了10亿美金收购的...最近它家出了份云的报告：<ThousandEyes-Cloud-Performance-Benchmark-2019-2020-Edition>.它说可以到处放Agent来进行全网监控：

![图片](assets/5b1c3c5c8f76.jpg)

而Ruta比它更厉害的地方在于，除了测量以外，它还可以互相同步链路质量，并直接就可以选路优化路径转发了。

![图片](assets/2ffd0e87a8ac.jpg)

**1. Ruta如何进行性能监测**

**1.1 Probe算法**

Ruta通过ETCD发现邻居节点后就会根据节点类型启用一个和TWAMP算法类似的Probe机制进行链路状态测量。这里又有一个反直觉的做法了，最开始写的时候，我直接就随手写了一个for循环丢100个包出去，后来发现每次测量误差都很大，然后又想多测几次求平均，后来想想还要packet pacing，模拟G.711或者G.729的ptime，越写越复杂，实际上这样高频的数据包发送会话多了也扛不住的，毕竟fullmesh的时候是O(n^2)，而且瞬时值误差很大，可能需要1000个包才能测试的相对准确，而我采用的做法是一秒钟发一次包，累计发100次来记录一个2分钟左右的时间，这种Trade-off一方面使得设备可以同时probe多点性能也不会下降，更重要的是从概率上来讲更好的拟合了线路的真实情况，误差相对较小了。因为一方面链路的的变化真的没那么快不需要很频繁的测量，另一方面随机抽样做好了会让算法的scale高很多。

**1.2 Ruta测试站点**

这次测试，我们共计设置了21个站点覆盖全球：

![图片](assets/83e7605d332d.jpg)

当然这些地址你们也别去攻击了我都释放了，腾讯云十个阿里云十个，还有一个是某个公司的香港的DMZ地址就不给了，TE前缀代表腾讯，ALI前缀代表阿里：

```
TE_SanJose|172.26.0.13|49.51.46.221TE_Tokoyo|10.203.0.9|124.156.229.92TE_Virginia|10.200.0.7|170.106.35.153TE_Guangzhou|172.16.0.8|106.55.162.180TE_Chongqing|172.30.0.11|139.186.129.7TE_Beijing|172.21.0.13|140.143.152.117TE_Toronto|172.18.0.12|49.51.93.39TE_Singapore|172.22.0.9|101.32.126.199TE_Moscow|10.202.0.17|162.62.11.162TE_Frankfurt|172.28.0.16|49.51.161.106ALI_Hangzhou|172.16.185.191|121.196.9.35ALI_Tokoyo|172.21.84.134|8.209.210.90ALI_Huhehaote|172.24.100.101|39.104.15.129ALI_London|172.21.14.26|8.208.26.141ALI_Dubai|172.31.11.156|47.91.105.39ALI_Chengdu|172.18.112.96|47.108.204.7ALI_Huhehaote|172.24.100.101|39.104.15.129ALI_Sydney|172.20.99.140|47.74.87.50ALI_Virginia|172.17.73.119|47.252.76.121ALI_SanJose|172.29.252.46|47.251.46.133
```

**1.3 Ruta测量方法**

由于Ruta基于一个分布式一致性数据库，因此我们通过一个节点watch “/states/linkstates"这个前缀，并将Watch的KV pairs带上时间戳转存到本地文件获取的数据，测试时间是从2020年8月19日1:00AM~2020年8月23日 15:00PM，然后通过一小段代码把这些数据存入到csv中。

**2. Ruta LSDB分析**

**2.1 数据处理的小技巧**

在做inline的路由决策或者其它操作前，我通常喜欢把历史上一段时间的数据导出来分析一下，也就是ThousandEyes前文那个报告里的部分内容，有些时候我比较懒不想自己画图那么就导入到ElasticSearch，利用Kibana做一些基础的数据呈现：

![图片](assets/0e5d9790b172.png)

比较有用的数据，那么再换到Python/Jupyter里做进一步的分析，可视化工具就比较多了，我通常的做法是很多时序线图或者Stack bar这样的图采用Bokeh,而heatmap类的直接就pandas sytle染色了，当然matplotlib有些时候也会用一点，主要是比较懒的时候。通常我也喜欢把Jupyter的屏幕宽度调到100%屏幕宽度，这样干活舒服点。

```
import pandas as pdimport numpy as npimport matplotlibimport matplotlib.pyplot as plt%matplotlib inline  from IPython.core.display import display, HTMLdisplay(HTML("<style>.container { width:100% !important; }</style>"))
```

然后数据基本上就是一个CSV格式的，pandas read一下就好了，但我比较喜欢把它转存为pickle，因为下次再用的时候加载快，而延迟相关的数据我们都采用ns uint64记录的，所以算出来的delta值也是ns，这里统一改为ms

```
df = pd.read_pickle("./srou.dat")#convert ns delta to msfor item in ['rtt_avg', 'rtt_min', 'rtt_max', 'jitter','s2d_avg', 's2d_min', 's2d_max', 's2d_jitter','d2s_avg', 'd2s_min', 'd2s_max', 'd2s_jitter']:    df[item]= df[item]/1000000
```

这样处理还有一些问题，过滤处理起来很麻烦，例如要选择某个时间段或者某个源/目的节点的信息，干脆外面包一层把一些常见的过滤条件都包起来弄成几个常用的函数就好了，其实本来应该弄一个新的class去加一些方法的，还是一个字：懒，够用就行了。

```
def filter_src(df,src):    return df[df['src'].str.contains(src)]def filter_dst(df,dst):    return df[df['dst'].str.contains(dst)]def filter_time(df,starttime,endtime):    return df[(df['datetime']>starttime )& (df['datetime']<endtime )]def filter_link(df,src,dst):    return filter_dst(filter_src(df,src),dst)def filter_link_time(df,src,dst,starttime,endtime):    return filter_link(filter_time(df,starttime,endtime),src,dst)def filter_src_time(df,src,starttime,endtime):    return filter_src(filter_time(df,starttime,endtime),src)def filter_dst_time(df,dst,starttime,endtime):    return filter_dst(filter_time(df,starttime,endtime),dst)
```

画一些timeseries stacked bar的函数如下：

```
import bokeh.plottingimport bokeh.modelsimport bokeh.layoutsimport bokeh.palettesfrom bokeh.resources import INLINEimport bokeh.iobokeh.io.output_notebook(INLINE)def bokeh_datetime_vbar_stack_chart(df,stacked_value,stack_name,title,x_axis='datetime',width=1900,height=600):    lines_counter = len (stacked_value)if(lines_counter <=10):        color_list=bokeh.palettes.Category10[10]else:        color_list=bokeh.palettes.Category20[20]    fig =  bokeh.plotting.figure(x_axis_type="datetime", plot_width=width,plot_height=height, title=title)#,toolbar_location=None, tools="hover")    fig.vbar_stack(stacked_value, x=x_axis, width=0.5, color=color_list[0:lines_counter], source=df,legend_label=stack_name)    fig.y_range.start = 0    #fig.x_range.range_padding = 0.1    #fig.xgrid.grid_line_color = None    fig.axis.minor_tick_line_color = None    fig.outline_line_color = None    fig.legend.location = "top_left"    fig.legend.orientation = "horizontal"    return fig    def describe(df):    data= df.groupby('datetime').mean().reset_index()    chart2 =bokeh_datetime_vbar_stack_chart(data,['d2s_avg','s2d_avg'],['d2s','s2d'],title="Latency",height=400)    chart3 =bokeh_datetime_vbar_stack_chart(data,['d2s_jitter','s2d_jitter'],['d2s','s2d'],title="Jitter",height=400)    chart4 =bokeh_datetime_vbar_stack_chart(data,['d2s_loss','s2d_loss'],['d2s','s2d'],title="Loss",height=400)    bokeh.plotting.show(bokeh.layouts.column([chart2,chart3,chart4]))
```

我们来看第一组数据，所有数据的平均值：

![图片](assets/1ff8bbb39e79.png)

很多时候我们的直觉又让我们开始犯错误了，平均值看，丢包率10%左右，还有明显的周期性，很多人或许就满足了，然后这一步就停止继续工作了，例如thousandeye只按照天平均，魔鬼都在细节中~

例如我们看从腾讯东京到阿里杭州这样的链路

```
describe(filter_link(data,"TE_Tokoyo","ALI_Hangzhou"))
```

‍

![图片](assets/c4ac2c0209b2.png)

可以注意到在繁忙时段，延迟是稳定的，延迟下降基本上都是每天半夜队列空了导致的，抖动基本上出现在半夜空闲时段，也就是队列不满不空的中间态，而丢包率每天的峰值会达到40%左右，并且具有很明显的时间周期特征，丢弃也是单方向的，主要是入境流量的丢弃。

所以前面求overall平均看到的丢包率10%就有很大的偏差了吧，这也就是很多人处理数据不细心导致的。

既然闲时和忙时丢包率差距这么大，又具有周期性，我们能干点什么吗？这就是Ruta选路的精髓了。首先我们根据时间轴来分片取样,利用pandas的Pivot_table

```
import seaborn as snsdef pivot_table(df,value):    t2=df.groupby([df['dst'],data['src']]).agg(['mean','median','std'])    t2.columns =[ "_".join(x) for x in t2.columns.ravel()]    t3=t2.iloc[:,0:].reset_index()    return t3.pivot_table(index=['src'],columns='dst')[value]def pivot_loss_table(df,value,cmap):    t4= pivot_table(df,value)    return t4.fillna(0).style.background_gradient(cmap=cmap).format("{:3.2f}%")    def pivot_delta_table(df1,df2,value,cmap):    t1= pivot_table(df1,value)    t2= pivot_table(df2,value)    return (t1-t2).fillna(0).style.background_gradient(cmap=cmap,axis=1).format("{:3.2f}%")
```

例如我们选择如下两个时间段进行比对：

```
busynight=filter_time(df,"2020-08-19 21:00:00",'2020-08-19 22:00:00')deepnight=filter_time(df,"2020-08-20 03:00:00",'2020-08-20 04:00:00')pivot_loss_table(busynight,'s2d_loss_mean',sns.light_palette("orange",as_cmap=True))
```

忙时链路丢包率：

![图片](assets/95842a9fcde0.png)

可以看到的是国内的节点即便是在忙时，出境方向上的流量丢包率并不大，而入境方向上普遍存在丢包，你可以明显的看到竖着的几列(阿里杭州，阿里呼和浩特，腾讯北京，腾讯重庆，腾讯广州），同时你也需要注意到，阿里成都和阿里呼和浩特，腾讯广州重庆北京到腾讯旧金山和腾讯新加坡节点，都有一些质量非常好的少丢包或者零丢包线路。

空闲时段丢包率如下，基本上你可以额看到半夜时分，阿里伦敦/阿里硅谷/腾讯多伦多到腾讯重庆腾讯广州的链路还是维持在较高的丢包水平，其它链路几乎都变得非常空闲。

![图片](assets/12194d6476bd.png)

具体的差值我们也可以看一下：

```
pivot_delta_table(busynight,deepnight,'s2d_loss_mean',sns.light_palette("green",as_cmap=True))
```

![图片](assets/d14dc144769f.png)

可以注意到入境流量在深夜都有很大的提升。然后我们来看看延迟，延迟的函数类似，就是包一层样式，把丢包的%改为ms单位

```
def pivot_latency_table(df,value,cmap):    t4= pivot_table(df,value)    return t4.fillna(0).style.background_gradient(cmap=cmap).format("{:3.2f}ms")def pivot_delta_latency_table(df1,df2,value,cmap):    t1= pivot_table(df1,value)    t2= pivot_table(df2,value)    return (t2-t1).fillna(0).style.background_gradient(cmap=cmap,axis=1).format("{:3.2f}ms")pivot_latency_table(busynight,'s2d_avg_mean',sns.light_palette("navy",as_cmap=True))    
```

![图片](assets/680e60784feb.png)

而深夜的闲时链路也还好:

![图片](assets/9da07d4a79cb.png)

接下来还是看看delta,变化最大的是各个云到成都的这条线路：

![图片](assets/3a67ac5e0ccc.png)

那么我们来看看成都这个地址段47.108.0.0具体的BGP路由表其实可以通过RouteViews project (www.routeviews.org)去查看下载BGP路由表，然后通过bgpdump查看即可：

```
kevin@PeentosM:~/srou/rib$ ./bgpdump -m rib3.bz2 | grep "|47.108.0.0/"2020-08-25 09:15:06 [info] logging to syslogTABLE_DUMP2|1597996804|B|195.208.112.161|3277|47.108.0.0/15|3277 3267 1299 4837 37963|IGP|195.208.112.161|0|0|3277:39710|NAG||TABLE_DUMP2|1597996804|B|208.51.134.255|3549|47.108.0.0/15|3549 3356 4134 58461 37963|IGP|208.51.134.255|0|0|3356:2 3356:22 3356:100 3356:123 3356:500 3356:2106 3549:2018 3549:30840|NAG||TABLE_DUMP2|1597996804|B|162.251.163.2|53767|47.108.0.0/15|53767 3257 4837 37963|IGP|162.251.163.2|0|0|3257:6520 3257:8044 3257:30084 3257:50002 3257:51200 3257:51201 53767:5000|NAG||....
```

‍

然后我们把这个CSV导入到Jupyter中

```
bgproute = pd.read_csv("./bgp.csv",delimiter="|")
```

![图片](assets/69a257629f81.png)

利用NetworkX这个库分析，画图，就可以呈现ThousandEyes的BGP连接图了,是不是非常容易呀~

```
import networkx as nxG =nx.Graph()for item in bgproute['ASPATH']:    itemlist=str(item).split(" ")    for i in range(0,len(itemlist)-1):        G.add_edge(itemlist[i],itemlist[i+1])plt.figure(figsize=(20,13))nx.draw(G,with_labels=True, node_size=3000,node_color="navy",font_color="white",node_shape="h",font_weight="bold")plt.show()
```

![图片](assets/a3284c84f098.png)

然后光看这些数不预测可不行，例如云资源都要花钱，但是很多时候如果有可行解的情况下做SRoU，没有可行解的时候再调度云资源付费加速不更好么？所以肯定就存在一个预测的处理，当然很多人会想到时间序列分析来拟合咯...以前写过一篇:

**[> AIOps系列（1）：时间序列分析的方法](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247483881&idx=1&sn=f372faf5e0115dcf54fc40d106520db3&chksm=f996112bcee1983df0f6762d0b20d08bd4a200defc21c154fa344e38349b1c2b9cf086c65ab9&scene=21#wechat_redirect) <**

这里还是稍微再做一次,例如通过statsmodel的内置的seasonal decompose：

```
tsa_data = filter_link(data,'TE_Tokoyo','ALI_Hangzhou')from statsmodels.tsa.seasonal import seasonal_decomposedecomposition = seasonal_decompose(tsa_data['s2d_loss'],freq=288)tsa_data['loss_seasonal']= decomposition.seasonaltsa_data['loss_trend'] = decomposition.trendtsa_data['loss_residual'] = decomposition.residchart2 =bokeh_multi_line_chart(tsa_data,['loss_seasonal'],['周期性'],title='周期')chart3 =bokeh_multi_line_chart(tsa_data,['loss_trend'],['延迟'],title='趋势')chart4 =bokeh_multi_line_chart(tsa_data,['loss_residual'],['残差'],title='残差')show_column([chart2,chart3,chart4])
```

![图片](assets/423b83d0bde6.png)

趋势项基本上稳定在22附近，周末会有所下降，周期性也很明显的刻画了忙时和闲时的情况，残差相对较小，基本上可以接受这个数据，然后我们继续用facebook的prophet：

```
from fbprophet import Prophettsa_data['ds'] = tsa_data['datetime']tsa_data['y'] = tsa_data['s2d_loss']m = Prophet(changepoint_prior_scale=0.01).fit(tsa_data)future = m.make_future_dataframe(periods=96,freq='H')fcst = m.predict(future)fig = m.plot(fcst)
```

![图片](assets/23f045054ac8.png)

当然原始数据刚好只有几天，结尾值又恰逢周末丢包变少，所以最终prophet预测出一个缓慢下降的趋势。这个错误是原始数据测量周期太短导致的。

**3. Ruta SRoU可行性**

其实这次测量并不是要故意看看哪家云好哪家云坏，各位云厂商的人别紧张，我没偏好的，我只是有代表性的随机选了几个region的云来验证SRoU的可行性，结论是在很多时候可以通过SRoU构建一些跨运营商的多归属节点，例如我家里放一个同时接电信和联通和移动的Fabric节点，然后这样子就可以跨越BGP ASPATH的一些限制选路了，也不一定是要ASPATH最短的路径，因为具体的延迟Ruta自己也可以测量了。

例如GCP没有欧洲到印度的链路，那么我可以通过很多运营商直接构建多跳中继，说不定还可以走莫斯科和一些东欧国家到印度。

![图片](assets/f3f6a2e01c8d.png)

**Ruta SRoU 也是在互联网上构建一条新的丝绸之路，一带一路伟大事业的互联网构建，**

**
**

**另外有人对networkX画拓扑感兴趣，其实还有很好玩的股票分析，哈哈哈~~**

![图片](assets/f1d5cbcef75a.png)