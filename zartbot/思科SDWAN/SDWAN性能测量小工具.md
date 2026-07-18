# SDWAN性能测量小工具

> 作者: zartbot  
> 日期: 2021年4月17日 13:07  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485705&idx=1&sn=f9c119bebaf26b24b19aae8efdb01fdf&chksm=f99619cbcee190ddc9de1e5463f317b36e3f60455c3809e372fea39d8e869ba0ed8515bfcfbb#rd

---

❝
通常我们在部署SDWAN时，最大的困难是验证策略是否生效，以及SDWAN一些优化功能是否真的有用。
❞
因此今天花了半天的时间开发了一个跨平台的小工具分享给大家,也就600多行code的小东西，而这个测试工具接下来也会整合进Ruta中，来构建一个专门的多云性能测试框架。

https://github.com/zartbot/sdwan-perf

### 常规测试 

通常对于SDWAN测试的做法是自己去装`iperf`等工具然后构建`Server`/`Client`,或者利用厂商自带的功能

![图片](assets/b007d8f42578.png)

### 延迟测量需求 

但是在很多场合可能还需要测试百分位点的延迟，例如我最近给一个券商测试的时候对比一些功能,主要是为跨市场套利时的交易数据做广域网延迟保障,需要如下的测试方式,具体功能以后跟某客户一起发布
功能交易笔数平均延迟抖动P99Max传输速率默认4907045.81ms55.27ms117ms1222ms2.4MbpsFEC-14959945.13ms44.56ms117ms1112ms3.5MbpsFEC-26154537.42ms43.46ms64ms1162ms7.5MbpsCiscoX6671335.16ms10.44ms63ms116ms11.2Mbps
### 策略验收需求 

另一方面是很多时候SDWAN都是要远程开局的，本地基本上都是一些完全不懂网络的现场工作人员，这种情况下需要使用轻量级的工具进行测试和验收。这些验收可能需要测试一些第三方网站，并且还要验证访问路径和可达性。

### SDWAN-Perf用法 

给大家准备的这个小工具基于fasthttp，资源消耗很小，托Golang的福可以很容易支持Linux、Windows、MAC以及X86、ARM、MIPS等多种平台，您可以在自己的终端上安装，也可以直接在路由器上使用它，例如我最近经常喜欢使用Cisco IOS XE自带的Container**,把Container接口放入到Service VPN中，然后可以在Hub路由器上执行SDWAN-perf的server模式,这个时候我通常喜欢多开几个端口listen，用于测试QoS和不同的policy的情况。

```
./sdwan-perf_linux -role=server -port=8000,8001,8002,8003
```

然后客户端随便找个windows、MAC执行客户端：

```
./sdwan-perf_linux -role=client  -duration=100 -server=10.199.0.2 -port=8001 -size=1000000 -num=100
```

执行过程中就会看到如下的统计结果了

```
   SDWAN Performance Test Report+--------------+---------------------------+--------------------------+|    Stats     |        Latency(ms)        |  Bandwidth(Per Session)  |+--------------+---------------------------+--------------------------+| mean         |               166.82ms    |                75.35Mbps || Jitter       |               770.53ms    |                          ||              |                           |                          || Min          |                87.00ms    |                 0.80Mbps || p25          |               102.00ms    |                72.74Mbps || p75          |               110.00ms    |                78.44Mbps || p90          |               113.00ms    |                81.64Mbps || p95          |               116.00ms    |                83.34Mbps || p99          |               136.00ms    |                86.03Mbps || Max          |              9978.00ms    |                91.97Mbps |+--------------+---------------------------+--------------------------+| Count: 16102 | Error: 312 | Timeout: 300 | Total-BW:    7534.52Mbps |+--------------+---------------------------+--------------------------+
```

当然您也可以用来测试其它网站:

```
 ./sdwan-perf_linux -role=client -url=https://www.google.com -num=1 
```

这个软件只有几兆，很容易传输，server、client模式都是同一个binary执行文件，反正我用了它以后就把iperf丢了...

它还有些其它选项，例如客户端并发数、客户端pipeline request数，每次请求的response size、timeout选择，server传输完了是否fin等等...

```
Usage of ./sdwan-perf_linux:  -duration int     Test Duration (default 60)  -fin     server mode close connection after send response  -num int     Num of clients (default 10)  -port string     Server Port (default "8000")  -reqs int     Pipeline reqs per client (default 10)  -role string     Role: client|server (default "client")  -server string     Server IP address (default "127.0.0.1")  -size int     bandwidth test block size (default 1)  -timeout int     client timeout seconds (default 10)  -url string     Testing URL
```

自己拿去用拿去改吧... 以后别再用ping或者iperf 测速了....