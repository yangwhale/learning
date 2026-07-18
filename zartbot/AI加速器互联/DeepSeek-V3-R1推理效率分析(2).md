# DeepSeek-V3/R1推理效率分析(2)

> 作者: zartbot  
> 日期: 2025年3月21日 01:05  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493558&idx=1&sn=f2239d459c383d25d403d0bd155496b7&chksm=f995f774cee27e62d05d6c33c7a75db99e0e28621fa07215c7a06587df513f5fd6f5aa007a57#rd

---

本文由于计算量非常大,步骤也很多难免有错误之处,欢迎大家指正.并且每个计算时使用的函数都已列出,可供大家自行修改. 本文仅代表个人观点, 与任职的机构无关.

有些人总有取名字的癖好,例如我自己搞的Nimble/Ruta,现在越来越简单了,把这个模拟仿真的项目取名叫ShallowSim吧, Deep不敢当, 另外每次`import shallowsim as sb`的时候还有一些快感...可以访问`github.com/zartbot/shallowsim`获取.shallowsim的主要用途如下:

用于对模型架构的研究, 特别是在新的GB200/GB300架构下, 如何利用好这些新的硬件优势设计模型架构, 以及如何同时又能兼顾到国内的一些推理硬件

通过对模型的架构估计, 来分析ScaleUP和ScaleOut的需求, 是否真的如老黄所说的先要ScaleUP再ScaleOut, 以及在国内各种工艺受限的情况下如何规避的研究.

基于这两个目的, 渣B会适当的高估计算Kernel的效率,然后比较充分的暴露出互联的一些问题, 然后再想办法去看如何从模型结构和算法上规避. 对于整个计算过程, 前面已经有两篇文章有一些详细的介绍了

[《DeepSeek-V3/R1推理效率分析(v0.17)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493473&idx=2&sn=893f29e472c7d241bda9c040468ade2b&scene=21#wechat_redirect)

[《DeepSeek-V3/R1推理效率分析-Blackwell性能估计(V0.4)》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493504&idx=1&sn=45d4ceb21a51b013d0acc83a79d495a4&scene=21#wechat_redirect)

这个系列文章后续还会增加对Expert负载不均衡的一些估计, 以及未来模型架构演进的一些探讨.

### 1. 性能校准

和@Han Shen讨论了一下, 以前建模过程中的一些误差:

#### 1.1 MLA的性能估计

MLA的计算过程并没有考虑到Softmax无法overlap的情况, 后来关注到FlashMLA的benchmark为580TFLOPS, 在H800上打不满的主要问题是否就是Attention中的Softmax的计算无法Overlap?

![图片](assets/a340b4da0b4e.png)

根据SemiAnalysis的分析, 在B300上提到的Attention指令2.5x的加速,应该是指的MUFU.EX2指令的优化,使得WGMMA能够更好的和softmax计算overlap起来? 渣B在建模的时候一开始就是考虑到完全能够overlap的,因此就没有计算这部分耗时, 现在基于FlashMLA在H800上的效率对计算进行了等比例的修正, 后续针对B300的估计直接将这部分的折扣系数提高即可.

然后针对TP并行, 针对每个计算kernel估计了一个计算下界, NVLink上的Allreduce增加了静态延迟估计.

### 2. MoE GEMM性能估计

原始的估计是基于batchsize拍脑袋估的一个值, 后来参考了@Han Shen在H800和H20上的实测结果, 按照m_per_group进行估计,并且留了gemm_group_num的接口用于后续根据不同的GPU架构进行仿真, 而对于H20 @Han Shen的测试结果发现了很有意思的一个点, 它的groupGEMM很容易打满. 因此单独针对H20进行了一些修正.

### 3. Prefill性能

```
import os,syssys.path.insert(1, os.path.join(os.getcwd()  , '..'))import shallowsim as sbargs = sb.ModelArgs()gpu_blackwell = sb.get_gpu_info('./device/gpu_info.csv', print_console=True ) seq_len = 4383kv_cache_rate = 0.563decode_len = 1210bs_list =[ 16, 32, 64, 128, 256, 512]eplist = [ 8 , 16, 36, 72, 144, 320]detail,summary = sb.prefill_time(args,gpu_blackwell,seq_len, kv_cache_rate, tp=4, dp=8)
```

![图片](assets/47c9261a1476.png)

统计Prefill性能如下:

```
tp=4_ , ttft_sum = sb.prefill_time(args,gpu_blackwell,seq_len, kv_cache_rate, tp=tp, dp=8, print_console=False)print(ttft_sum.apply(lambda x: seq_len/tp * (1000/ x)).loc['Sum'].to_markdown(floatfmt=".1f"))
```

GPU

Sum

DGX-B300

21998.5

DGX-B200

17607.5

GB200-NVL72

19473.2

GB300-NVL72

26824.2

H200

6703.7

H800

6222.4

H20

1143.2

H20-3E

1144.8

### 4. Decoding性能

```
import os,sysimport warningssys.path.insert(1, os.path.join(os.getcwd()  , '..'))warnings.filterwarnings('ignore', category=DeprecationWarning)warnings.filterwarnings('ignore', category=FutureWarning)import shallowsim as sbargs = sb.ModelArgs()c = sb.Config()gpu_blackwell_decode = sb.get_gpu_info('./device/gpu_info.csv',                                     decoding_mode=True, print_console=True)                                     
```

例如您需要估计某个卡ScaleOut不同的带宽配比, 例如H20采用1.6Tbps时,改一下参数即可

```
gpu_blackwell_decode['H20'].pcie_bw = 25 #change bw to 1.6Tbps..
```

#### 4.1 GPU性能天梯榜

然后仿真结果, 并对dataframe进行分组排序.

```
dfs = sb.decode_time_with_ep_list(args,gpu_blackwell_decode,c,print_console=False,fp8_combine=True)dfs_o = dfs.groupby(['GPU','BatchSize'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)dfs_o.style.bar(subset=['TPS','Total'],color='#6495ED').applymap(sb.color_positive_red, subset=['Delta']).background_gradient(subset=['Comm_Impact'],cmap=sb.cm).format(precision=3) 
```

这个就是不同的平台基于每卡总的Token/s的天梯榜了. 详细大图可以参考`https://github.com/zartbot/shallowsim/blob/main/figures/performance.png`
![图片](assets/cf07c001959f.png)

可以看到GB300-NVL72的性能还是不错的, 不过只能在EP=72的时候, 更大的EP要经过ScaleOut网络,性能会受损, 另外它必须要在更大的Batchsize下才有性能收益. 如果要维持40~50TPS, DGX-B300也是很不错的, 特别的来说TP=1, EP320在BatchSize=256的情况下, 也能到11270 Tokens/s, 这就很有趣了, 那么买320块DGX-WorkStation理论上也能到这样的性能?

![图片](assets/57e223943fdc.png)

这样就有一个灵魂拷问了, 先ScaleUp再ScaleOut是否成立呢? 然后GB200和GB300之间大概还有1.3倍左右的性能差距. 对于H800 EP320的性能还是有一些优势的, 例如按照论文所讲的TP4+EP320的部署, 单个Query在BatchSize=64时TPS可以做到48 tokens/s, 累计单机3000Tokens/s, 但是这个估计会偏高一些. 而对于H20按照DeepGEMM修正后由于较小的m_per_group就能打满, 反而性能不错, 没有受到算力阉割的影响, 性能基本接近H800的80%,和@Han Shen的结果是比较相似了,但是最佳部署还是要EP320.

#### 4.2 基于GPU的性能分析

我们还可以通过如下方法分析GPU在不同并行策略下的性能

```
gpu = 'GB300-NVL72'tps_limit = 20tdf = sb.df_filter(dfs,gpu,tps_limit=tps_limit)sb.df_sort(tdf,value='Total',ascending=False).style.bar(subset=['TPS','Total'],color='#6495ED').applymap(sb.color_positive_red, subset=['Delta']).background_gradient(subset=['Comm_Impact'],cmap=sb.cm).format(precision=3) 
```

![图片](assets/3503b72f0775.png)

可以看到性能最好的是TP=4, EP=72/36, BatchSize=256的场景. 而TP=1, EP=320时完全不使用NVLink, 性能也还不错.

对于H800的性能如下:

```
gpu = 'H800'tps_limit = 20tdf = sb.df_filter(dfs,gpu,tps_limit=tps_limit)sb.df_sort(tdf,value='Total',ascending=False).style.bar(subset=['TPS','Total'],color='#6495ED').applymap(sb.color_positive_red, subset=['Delta']).background_gradient(subset=['Comm_Impact'],cmap=sb.cm).format(precision=3) 
```

![图片](assets/b6993aa40103.png)

TP=1,EP=144,BatchSize=128的性能相对于官方的数据还是高了很多的, 猜测了几点未来可以校准的方向:

专家负载不均衡和一些长尾导致的性能损失, 后面会适当的再通过Kingman公式引入网络和计算的变异系数来估计一个等待时间.

在线服务Token长度的不均衡影响, 峰谷的影响等..

对于H20的分析如下, TP=8, EP=320, 并且采用更小的BatchSize反而性能更好, 主要得益于GroupGEMM能够很快打满.

![图片](assets/0c22d31bd2aa.png)

H20-3E的分析如下:

![图片](assets/555dac5f9a22.png)

#### 4.3 统计不同Seq_Len

有了这个仿真平台, 我们就可以估计不同的Seq_Len的影响, Github已经上传了这个csv,只需要加载就可以分析, 实际的计算过程如下:

```
dfs = []for seq_len in range(1024,16384,32):    c.seq_len = seq_len    df = sb.decode_time_with_ep_list(args,gpu_all_decode,c,fp8_combine=True)    df['Seq_len'] = seq_len    df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)    df_o.drop_duplicates(subset=['GPU','BatchSize','EP'], keep='first', inplace=True)    dfs.append(df_o)df = pd.concat(dfs)    df.reset_index(inplace=True,drop=True)df.to_csv('perf_vs_seq_len.csv')
```

加载csv并画图

```
df = pd.read_csv('perf_vs_seq_len.csv')df['BatchSize']= df['BatchSize'].astype(int).astype(str)df1 = df[df['EP'] == 144].reset_index(drop=True)sb.draw(df1, gpu_all_decode, 'Total','Token per second')
```

Blackwell相对于Hopper大概有3.5倍的提升

![图片](assets/a9bd00dfa870.png)

估计单个Query TPS的性能:

![图片](assets/28bfc4a22924.png)