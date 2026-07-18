# DeepSeek-V3/R1推理效率分析-Blackwell性能估计(V0.4)

> 作者: zartbot  
> 日期: 2025年3月17日 16:06  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493504&idx=1&sn=45d4ceb21a51b013d0acc83a79d495a4&chksm=f995f742cee27e54f0d1fab3e9f8881667109aefb6260fefa10c3520a09899d05a722d417f48#rd

---

其实DeepSeek-V3/R1效率的分析, 主要工作是分析B200这些NVL实际的价值和业务收益, 前一篇的分析主要是在H800和H20上去对齐现有的数据, 给整个计算构建一个相对接近真实值的一个参考模型. 另一方面也可以补齐对于模型架构和基础设施结合设计的一环.

本文由于计算量非常大,步骤也很多难免有错误之处,欢迎大家指正.并且每个计算时使用的函数都已列出,可供大家自行修改. 本文仅代表个人观点, 与任职的机构无关.

ver

0.1

initial version

0.2

更新GPU性能数据处理函数

0.21

基于GPU内存约束过滤decoding阶段的batchsize

0.3

处理Decoding阶段SparseMLA TP并行时数据没有按照TP策略更新的问题

0.4

增加参数加载时间,网络静态延迟等估计

注: 这个估计结果对于B200 GEMM采用FP4计算, 并且在Dispatch阶段采用FP4, 而Combine阶段考虑精度影响依旧采用FP16, 当然是否可以在网络上传输Combine时采用FP8,然后在Combine-Reduce的时候按照BF16计算,这个问题不得而知, 在Combine采用FP16时, GB200-NVL72还是可以Overlap, 但DGX-B200存在影响, 采用FP8 Combine时,则可以Overlap. 另外我们做了一个假设, 当DGX-B200有CX9(1.6Tbps)网卡时,FP16 Combine通信可以完全Overlap.

### TL;DR

直接贴一下汇总结果, 平均每卡Prefill阶段的每秒token数

GPU

Sum

DGX-B200

21574.6

GB200-NVL72

25560.2

H200

8423.6

H800

7677.5

H20

1162.9

Decoding阶段,如果采用FP16 Combine, 可以看到在ScaleOut网络中, DGX-B200即便是每卡配置800Gbps带宽,还是存在大量的无法Overlap的情况

![图片](assets/cb5d43e4b5dc.png)

即便是采用FP8 Combine也会大量无法Overlap, ScaleOut网络在DeepSeek-V3这样的模型下无法支撑B200这样的算力.

![图片](assets/c8d4acd83bc2.png)

每卡的每秒token数, 蓝线为BatchSize=32, 绿线为BatchSize=128

![图片](assets/af6ab4e98162.png)

每个Session的TPS:

![图片](assets/29930982d894.png)

当然这个结果只是针对DeepSeek-V3模型, 可能以偏概全, 并且并行策略上可能还有一些调优的空间, 例如MoE Group在这些新硬件上的设计, n_activate_expert的选择等... 或者专家本身调大一些, 然后激活专家少一点, 或者路由规则/负载均衡规则上做一些处理, 后面的文章可能会从这个方面来分析.

本文目录如下:

```
1. B200的性能参数2. MLA性能分析2.1 MLA计算复杂度2.2 MLA GEMM_FP4和TP分析3. Prefill阶段分析3.1 MLA计算耗时3.2 DenseMLP耗时3.3 MoE Expert计算耗时3.4 AlltoAll通信耗时3.5 汇总统计4. Decoding阶段分析4.1 最大BatchSize估计4.2 MLA计算耗时4.3 DenseMLP耗时4.4 MoE计算耗时4.5 AlltoAll通信耗时4.6 总耗时分析5. 结论
```

## 1. B200的性能参数

按照8卡的DGX-B200和GB200-NVL72评估, 具体的参数如下, 虽然当前售卖的DGX-B200采用CX7 400Gbps网卡, 但是考虑到FP4_GEMM的性能以及MLA的计算时间减少会导致Combine耗时有显著影响无法Overlap, 因此估计时还是按照CX8 800Gbps网卡计算.

gpu_type

sm

comm_sm

fp16

fp8

fp4

mem

mem_bw

nvlink_bw

pcie_bw

gpu_per_node

DGX-B200

160

20

2250

4500

9000

180

8000

900

100

8

GB200-NVL72

160

20

2500

5000

10000

192

8000

900

100

72

H200

132

20

989

1979

0

141

4800

450

50

8

H800

132

20

989

1979

0

80

4000

200

50

8

H20

78

10

118.4

236.8

0

96

4000

450

50

8

H20-3E

78

10

118.4

236.8

0

141

4800

450

50

8

具体建模的函数如下所示:

```
class GPU_perf():    def __init__(self,gpu_type, sm,comm_sm, gpu_per_node,fp16_flops,fp8_flops,fp4_flops,mem,mem_bw, nvlink_bw,pcie_bw, discount_rate):        self.gpu_type = gpu_type        self.sm = sm        self.gpu_per_node = gpu_per_node        self.comm_sm = comm_sm        self.fp16_flops = fp16_flops        self.fp8_flops = fp8_flops        self.fp4_flops = fp4_flops        self.mem = mem        self.mem_bw = mem_bw        self.nvlink_bw = nvlink_bw        self.pcie_bw = pcie_bw        self.discount_rate = discount_rate    def get_fp16_flops(self):        return self.fp16_flops * self.discount_rate  * ( self.sm  - self.comm_sm) / self.sm    def get_fp8_flops(self):        return self.fp8_flops *  self.discount_rate * ( self.sm  - self.comm_sm) / self.sm    def get_fp4_flops(self):        return self.fp4_flops *  self.discount_rate * ( self.sm  - self.comm_sm) / self.sm    def get_mem_bw(self):        return self.mem_bw *  self.discount_rate    def get_nvlink_bw(self):        return self.nvlink_bw *  self.discount_rate    def get_pcie_bw(self):        return self.pcie_bw *  self.discount_ratedef get_gpu_info(filename, discount_rate= 0.85, device_list=[],decoding_mode=False):    gpu_dict = {}    df = pd.read_csv(filename)    print(df.set_index('gpu_type').to_markdown())    if decoding_mode:        df['comm_sm'] = 0    for _, c in df.iterrows():        key = c['gpu_type']        gpu = GPU_perf(            gpu_type= c['gpu_type'],            sm = c['sm'], comm_sm = c['comm_sm'],            fp16_flops= c['fp16'],             fp8_flops=c['fp8'],            fp4_flops=c['fp4'],            mem = c['mem'],            mem_bw = c['mem_bw'],            nvlink_bw = c['nvlink_bw'],            pcie_bw = c['pcie_bw'],            gpu_per_node = c['gpu_per_node'],        discount_rate = discount_rate)        if (len(device_list)==0) | (key in device_list):            gpu_dict[key] = gpu    return gpu_dict
```

相关的GPU参数如下:

```
gpu_blackwell = get_gpu_info('gpu_info.csv', device_list=['DGX-B200','GB200-NVL72','H200','H800','H20'] )   
```

## 2. MLA性能分析

### 2.1 MLA计算复杂度

MLA计算复杂度评估函数定义如下:
![图片](assets/55172acf6cdf.png)

```
#非吸收的版本def mla_flops(q_len, kv_len, args:ModelArgs, kv_cache_rate):    #calculate MACs and estimate Flops approx. 2xMAC.    q_down_proj = q_len * args.dim * args.q_lora_rank #wq_a    q_up_proj = q_len * args.q_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim) #wq_b    kv_down_proj = kv_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a    k_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim #w_uk    v_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.v_head_dim #w_uv    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)    gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj        #把它看成一个标准的args.n_heads的MHA    mha = args.n_heads * ( q_len * args.qk_rope_head_dim * kv_len #QK_score_rope                          + q_len * args.qk_nope_head_dim * kv_len #QK_score_nope                          + q_len * kv_len * args.v_head_dim) #ScoreV    wo = q_len * args.n_heads * args.v_head_dim * args.dim #wo    attn_sum = mha + wo    #return flops by 2* Sum(MACs)    GEMM_FP8_FLOPS = gemm_sum * 2/1e9    ATTN_FP16_FLOPS =  attn_sum * 2/1e9        return GEMM_FP8_FLOPS+ATTN_FP16_FLOPS, GEMM_FP8_FLOPS,ATTN_FP16_FLOPS
```

矩阵吸收的版本

![图片](assets/9b5cca312b21.png)

```
#矩阵吸收的版本def mla_matabsob_flops(q_len, kv_len, args:ModelArgs, kv_cache_rate=0):    #calculate MACs and estimate Flops approx. 2xMAC.    q_down_proj = q_len * args.dim * args.q_lora_rank #wq_a    q_rope_up_proj = q_len * args.q_lora_rank * args.n_heads * args.qk_rope_head_dim #wq_b_rope    q_absorb = q_len * args.n_heads * ( args.q_lora_rank * args.qk_nope_head_dim  #wq_b                                       + args.qk_nope_head_dim * args.kv_lora_rank ) #w_uk        kv_down_proj = kv_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a    kv_down_proj = kv_down_proj * (1 - kv_cache_rate) #KV-Cache命中率修正    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj         #把它看成一个标准的args.n_heads的MQA    mqa = args.n_heads * ( q_len * args.qk_rope_head_dim * kv_len #Score_rope                          + q_len * args.kv_lora_rank * kv_len #Score_nope                          + q_len * kv_len * args.kv_lora_rank) #Score V        attn_up_proj = q_len * args.n_heads * args.v_head_dim * args.kv_lora_rank    o_proj = q_len * args.n_heads * args.v_head_dim * args.dim    attn_sum = mqa + attn_up_proj + o_proj        #return flops by 2* Sum(MACs)    gemm_sum =  gemm_sum * 2/1e9    attn_sum = attn_sum * 2/1e9        return gemm_sum + attn_sum, gemm_sum,attn_sum 
```

参数量计算, 用于后续计算参数加载时间

```
def mla_mem(args:ModelArgs):    q_down_proj =  args.dim * args.q_lora_rank #wq_a    q_up_proj = args.q_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim) #wq_b    kv_down_proj = args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a    k_up_proj =  args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim #w_uk    v_up_proj =  args.kv_lora_rank * args.n_heads * args.v_head_dim #w_uv    wo = args.n_heads * args.v_head_dim * args.dim #wo    return (q_down_proj + q_up_proj + k_up_proj + kv_down_proj + v_up_proj + wo)/1024/1024
```

### 2.2 MLA GEMM_FP4和TP分析

MLA计算时间的分析如下:

```
def mla_elapse_time(args:ModelArgs,                     gpu:GPU_perf,                     seq_len,                     kv_cache_rate,                    tp=[2,4,8,16,32],                    decoding_mode=True,                    batchsize = 1,                    enable_gemm_fp4=True,                    min_ar_time = 0.015, #Allreduce的静态延迟                    is_print=False):    if decoding_mode:        #Decoding时计算为qlen=1, kv_cache_rate = 1        _ , gemm_flops, attn_fp16_flops = mla_matabsob_flops(1,seq_len,args, 1)        gemm_flops *= batchsize        attn_fp16_flops *= batchsize    else:        #prefill阶段使用非吸收的版本        _ , gemm_flops, attn_fp16_flops = mla_flops(seq_len,seq_len,args, kv_cache_rate)    gemm_fp8_t = gemm_flops / gpu.get_fp8_flops()     attn_fp16_t = attn_fp16_flops / gpu.get_fp16_flops()     #load weight    load_t = mla_mem(args) / gpu.get_mem_bw()        total = gemm_fp8_t + attn_fp16_t + load_t        if  enable_gemm_fp4 :        if  gpu.get_fp4_flops() == 0:            if is_print :                            print('[%8s]This GPU does not support FP4' % gpu.gpu_type)        else:            gemm_fp4_t = gemm_flops / gpu.get_fp4_flops()            total = gemm_fp4_t + attn_fp16_t    ar_len = batchsize if decoding_mode else seq_len     all_reduce_comm_size = ar_len * args.dim * 2 /1024/1024#fp16 take 2Bytes    all_reduce_t = all_reduce_comm_size / gpu.get_nvlink_bw() +min_ar_time    tp_time = {}    for v in tp:        if v == 1 :            tp_time[v] =total         else:            tp_time[v] =total / v + all_reduce_t    if is_print :        if  enable_gemm_fp4 & (gpu.get_fp4_flops() != 0):            print("[%8s]GEMM_FP4 Elapsed time(ms): %.3f" % (gpu.gpu_type,gemm_fp4_t))        print("[%8s]GEMM_FP8 Elapsed time(ms): %.3f" %  (gpu.gpu_type,gemm_fp8_t))        print("[%8s]ATTN_FP16 Elapsed time(ms): %.3f" % (gpu.gpu_type, attn_fp16_t))        print("[%8s]Total Elapsed time(ms):%.3f" % (gpu.gpu_type, total))        print("[%8s]AR Elapsed time(ms):%.3f"  % (gpu.gpu_type,all_reduce_t))        for v in tp:            print("[%8s]TP[%2d] Elapsed time(ms):%.3f" % (gpu.gpu_type,v,tp_time[v]))    return total, tp_time
```

其中针对Allreduce通信做了下界估计, 并考虑了参数加载的时间,

对于Blackwell架构新增的FP4计算, 我们进行一下评估, 在Prefill阶段FP4 GEMM的收益相对于Attn计算是可以忽略不计的, 另一方面, TP并行来看, NVL72并没有太大的优势, 因为Allreduce本身的耗时已经很长了, TP=8即可.

```
#Prefill阶段mla_elapse_time(args,                  gpu_blackwell['GB200-NVL72'],                  seq_len,kv_cache_rate=0,                  decoding_mode=False,                  enable_gemm_fp4=True,                  is_print=True)[GB200-NVL72]GEMM_FP4 Elapsed time(ms): 0.082[GB200-NVL72]GEMM_FP8 Elapsed time(ms): 0.164[GB200-NVL72]ATTN_FP16 Elapsed time(ms): 1.400[GB200-NVL72]Total Elapsed time(ms):1.482[GB200-NVL72]AR Elapsed time(ms):0.093[GB200-NVL72]TP[ 2] Elapsed time(ms):0.834[GB200-NVL72]TP[ 4] Elapsed time(ms):0.464[GB200-NVL72]TP[ 8] Elapsed time(ms):0.279[GB200-NVL72]TP[16] Elapsed time(ms):0.186[GB200-NVL72]TP[32] Elapsed time(ms):0.140
```

在Decoding阶段, FP4也没有太明显的收益

```
#Decode阶段mla_elapse_time(args,                  gb200_nvl72,                  seq_len,kv_cache_rate=0,                  decoding_mode=True,                  batchsize=128,                  enable_gemm_fp4=True,                  is_print=True)[GB200-NVL72]GEMM_FP4 Elapsed time(ms): 0.002[GB200-NVL72]GEMM_FP8 Elapsed time(ms): 0.004[GB200-NVL72]ATTN_FP16 Elapsed time(ms): 0.101[GB200-NVL72]Total Elapsed time(ms):0.103[GB200-NVL72]AR Elapsed time(ms):0.017[GB200-NVL72]TP[ 2] Elapsed time(ms):0.069[GB200-NVL72]TP[ 4] Elapsed time(ms):0.043[GB200-NVL72]TP[ 8] Elapsed time(ms):0.030[GB200-NVL72]TP[16] Elapsed time(ms):0.024[GB200-NVL72]TP[32] Elapsed time(ms):0.021
```

当BatchSize加到512时,结果如下所示:

```
_=mla_elapse_time(args,                  gpu_blackwell['GB200-NVL72'],                  seq_len,kv_cache_rate=0,                  decoding_mode=True,                  batchsize=512,                  enable_gemm_fp4=True,                  is_print=True)                  [GB200-NVL72]GEMM_FP4 Elapsed time(ms): 0.008[GB200-NVL72]GEMM_FP8 Elapsed time(ms): 0.016[GB200-NVL72]ATTN_FP16 Elapsed time(ms): 0.405[GB200-NVL72]Total Elapsed time(ms):0.413[GB200-NVL72]AR Elapsed time(ms):0.024[GB200-NVL72]TP[ 2] Elapsed time(ms):0.231[GB200-NVL72]TP[ 4] Elapsed time(ms):0.127[GB200-NVL72]TP[ 8] Elapsed time(ms):0.076[GB200-NVL72]TP[16] Elapsed time(ms):0.050[GB200-NVL72]TP[32] Elapsed time(ms):0.037
```

## 3. Prefill阶段分析

### 3.1 MLA计算耗时

MLA计算耗时的函数如下:

```
def prefill_mla(args:ModelArgs, gpu_dict, seq_len, kv_cache_rate,is_print=False):    df = pd.DataFrame(columns=['GPU','TP1','TP4','TP8'])    for key in gpu_dict.keys():        tp1,tp_list = mla_elapse_time(args,gpu_dict[key],                                     seq_len, kv_cache_rate,                                     tp=[4,8],                                     decoding_mode=False,                                     enable_gemm_fp4=True,                                     is_print=is_print)        df.loc[len(df)]=[gpu_dict[key].gpu_type,tp1] + list(tp_list.values())    if is_print:        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))    return dfseq_len = 4383kv_cache_rate = 0.563prefill_mla(args,gpu_blackwell,seq_len,kv_cache_rate,is_print=True)
```

MLA耗时结果如下所示(单位ms):

GPU

TP1

TP4

TP8

DGX-B200

1.644

0.504

0.299

GB200-NVL72

1.479

0.463

0.278

H200

4.107

1.198

0.685

H800

4.116

1.396

0.882

H20

33.087

8.443

4.308

### 3.2 DenseMLP耗时

DenseMLP耗时如下, 由于Nvidia公布了在B200上推理,FP4的精度和FP8基本一致, 因此计算时都采用FP4-GEMM

```
def densmlp_flops(args:ModelArgs, seq_len):    return3 * seq_len * args.dim * args.inter_dim *2/1e9def densmlp_mem(args:ModelArgs):    return3 * args.dim * args.inter_dim /1024/1024def _prefill_dense_mlp(args:ModelArgs,gpu:GPU_perf, seq_len,is_print=False):    gemm_flops = densmlp_flops(args, seq_len)    if gpu.get_fp4_flops()!=0:        gemm_time = gemm_flops / gpu.get_fp4_flops()    else:        gemm_time = gemm_flops / gpu.get_fp8_flops()    load_time = densmlp_mem(args) / gpu.get_mem_bw()    gemm_time = gemm_time + load_time    if is_print:        print("[%8s]Elapsed time(ms): %.3f" % (gpu.gpu_type, gemm_time))    return gemm_timedef prefill_dense_mlp(args:ModelArgs, gpu_dict,seq_len,is_print=False):    df = pd.DataFrame(columns=['GPU','DenseMLP'])    for key in gpu_dict.keys():        t = _prefill_dense_mlp(args,gpu_dict[key], seq_len,is_print=is_print)        df.loc[len(df)]=[gpu_dict[key].gpu_type,t]    if is_print:        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))    return dfprefill_dense_mlp(args,gpu_blackwell,seq_len,is_print=True)
```

计算结果如下所示:

GPU

DenseMLP

DGX-B200

0.575

GB200-NVL72

0.523

H200

2.527

H800

2.546

H20

19.912

### 3.3 MoE Expert计算耗时

这里, Prefill节点规模考虑对齐H800的部署, 4台32卡DGX-B200. 而针对GB200-NVL72 , 我们需要单独的计算一下DP=9 TP=8来评估一下NVL72的收益.

计算方式如下所示, 当GPU支持FP4时,按照FP4计算

```
def moe_expert_flops(args:ModelArgs, seq_len):    return3 * seq_len * args.dim * args.moe_inter_dim *2/1e9def moe_expert_mem(args:ModelArgs):    return3 * args.dim * args.moe_inter_dim / 1024 / 1024def _prefill_moe(args:ModelArgs,gpu:GPU_perf, seq_len, tp, dp):    load_time = moe_expert_mem(args) / gpu.get_mem_bw()    gemm_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops()!=0else gpu.get_fp8_flops()    num_device = tp * dp    num_shared_token = dp * seq_len / num_device    shared_flops = moe_expert_flops(args, num_shared_token)    shared_time = shared_flops / gemm_flops + load_time    num_routed_token = seq_len * dp * args.n_activated_experts / num_device    routed_flops = moe_expert_flops(args, num_routed_token)    expert_num = math.ceil(args.n_routed_experts) / dp / tp    routed_time = routed_flops / gemm_flops +load_time * expert_num     return shared_time, routed_timedef prefill_moe(args:ModelArgs, gpu_dict, seq_len,                 tp_list=[4,8],                 dp_list=[4,8,9],                is_print=False):    df = pd.DataFrame(columns=['GPU','TP','DP','Shared Expert','Routed Expert'])    for key in gpu_dict.keys():        for tp in tp_list:            for dp in dp_list:                s, r = _prefill_moe(args,gpu_dict[key], seq_len,tp,dp)                df.loc[len(df)]=[gpu_dict[key].gpu_type,tp,dp, s,r]    if is_print:        df['TP'] = df['TP'].astype(int).astype(str)        df['DP'] = df['DP'].astype(int).astype(str)        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))    return df
```

具体性能如下所示, 可以看到在Routed Expert计算上, FP4 GEMM相对于Hopper架构有显著的收益:

GPU

TP

DP

Shared Expert

Routed Expert

DGX-B200

4

4

0.021

0.214

DGX-B200

4

8

0.021

0.165

DGX-B200

4

9

0.021

0.159

DGX-B200

8

4

0.013

0.107

DGX-B200

8

8

0.013

0.082

DGX-B200

8

9

0.013

0.080

GB200-NVL72

4

4

0.019

0.203

GB200-NVL72

4

8

0.019

0.153

GB200-NVL72

4

9

0.019

0.148

GB200-NVL72

8

4

0.013

0.101

GB200-NVL72

8

8

0.013

0.077

GB200-NVL72

8

9

0.013

0.074

H200

4

4

0.078

0.706

H200

4

8

0.078

0.623

H200

4

9

0.078

0.614

H200

8

4

0.044

0.353

H200

8

8

0.044

0.312

H200

8

9

0.044

0.307

H800

4

4

0.080

0.739

H800

4

8

0.080

0.640

H800

4

9

0.080

0.629

H800

8

4

0.046

0.369

H800

8

8

0.046

0.320

H800

8

9

0.046

0.314

H20

4

4

0.562

4.598

H20

4

8

0.562

4.499

H20

4

9

0.562

4.488

H20

8

4

0.287

2.299

H20

8

8

0.287

2.249

H20

8

9

0.287

2.244

### 3.4 AlltoAll通信耗时

DeepSeek-V3设计的MoE Group是按照NVLINK:RDMA带宽3:1进行的设计, 最多路由到4个节点, 对于H200, B200来看, 通信存在PCIe带宽约束, 因此计算时依旧采用4机32卡的方式, 并且NVLINK带宽比H800高, 能满足要求

而GB200-NVL72则可以完全按照NVLINK带宽估计, 直观来看72卡配置时放置32个冗余专家, 平均每卡4个专家, 因此只有1/64的概率留在本地. 也就是说在dispatch过程中需要接近8份的数据传输到网络上, 有一个取舍就是每卡放置一个Group的专家, 这样每张卡就只需要最多dispatch 4份数据, GB200显存是能够承受的. 但是实际上我们考虑到Combine的时间和(MLA+Shared Expert) overlap以及Dispatch时间和Routed Expert Overlap即可, 因此还是按照每卡4个专家计算

```
def _prefill_alltoall(args:ModelArgs, gpu, seq_len, tp,static_latency=0.05):    if gpu.gpu_per_node ==8:        dp = gpu.gpu_per_node/tp        dispatch_node = 4        dispatch_size = (dispatch_node - 1) *  dp * seq_len * args.n_activated_experts / gpu.gpu_per_node * args.dim /1024/1024        comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node    else:        #NVL72        expert_num = math.ceil(args.n_routed_experts / gpu.gpu_per_node)         dispatch_prob = (args.n_routed_experts - expert_num ) / args.n_routed_experts         dispatch_size = dispatch_prob * args.n_activated_experts * seq_len/tp * args.dim /1024/1024        comm_bw = gpu.get_nvlink_bw()             combine_size = 2 * dispatch_size  #fp16      if gpu.get_fp4_flops != 0:        dispatch_size  = dispatch_size /2    dispatch_time = dispatch_size / comm_bw + static_latency    combine_time = combine_size / comm_bw + static_latency    return dispatch_time, combine_timedef prefill_alltoall(args:ModelArgs, gpu_dict, seq_len, is_print=False):    df = pd.DataFrame(columns=['GPU','TP','Dispatch','Combine'])    for tp in [4,8]:        for key in gpu_dict.keys():            dispatch_time, combine_time = _prefill_alltoall(args, gpu_dict[key],seq_len, tp)            df.loc[len(df)]=[key,tp,dispatch_time,combine_time]    if is_print:        df['TP'] = df['TP'].astype(int).astype(str)        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))    return df    prefill_alltoall(args,gpu_blackwell,seq_len,is_print=True)
```

计算结果统计如下(单位ms):

GPU

TP

Dispatch

Combine

DGX-B200

4

0.182

0.579

GB200-NVL72

4

0.089

0.204

H200

4

0.314

1.107

H800

4

0.314

1.107

H20

4

0.314

1.107

DGX-B200

8

0.116

0.314

GB200-NVL72

8

0.069

0.127

H200

8

0.182

0.579

H800

8

0.182

0.579

H20

8

0.182

0.579

### 3.5 汇总统计

部署时按照两个microbatch进行Overlap

![图片](assets/26434bf14f22.png)

汇总时,考虑非Overlap进行修正, 计算函数如下:

```
def _prefill_time(args:ModelArgs, gpu, seq_len, kv_cache_rate, tp , dp):    dense_mla,tp_mla = mla_elapse_time(args,gpu,                                     seq_len, kv_cache_rate,                                     tp=[tp],                                     decoding_mode=False,                                     enable_gemm_fp4=True)    dense_mlp = _prefill_dense_mlp(args,gpu,seq_len)    shared, routed = _prefill_moe(args,gpu, seq_len, tp, dp)    dispatch, combine = _prefill_alltoall(args, gpu, seq_len, tp)    return dense_mla, dense_mlp, tp_mla[tp], shared, combine, routed, dispatchdef prefill_time(args:ModelArgs, gpu_dict, seq_len, kv_cache_rate, tp , dp, is_print=False):    df = pd.DataFrame(columns=['GPU','MLA','DenseMLP','TP_MLA','Shared Expert','Combine','Overlap1','Routed Expert','Dispatch','Overlap2'])    df2 = pd.DataFrame(columns=['GPU','Compute','Comm','Sum'])    n_sparse_layers = args.n_layers - args.n_dense_layers    df.loc[len(df)]= ['Layers', args.n_dense_layers, args.n_dense_layers,  #MLA+ DenseMLP                       n_sparse_layers, n_sparse_layers, n_sparse_layers,n_sparse_layers,                       n_sparse_layers, n_sparse_layers, n_sparse_layers]    for key in gpu_dict.keys():        dense_mla, dense_mlp, tp_mla,shared, combine, routed, dispatch = _prefill_time(args, gpu_dict[key], seq_len, kv_cache_rate , tp , dp)        overlap1 = combine - (tp_mla + shared)        overlap2 =  dispatch - routed        df.loc[len(df)]= [ key, dense_mla,dense_mlp, tp_mla,shared, combine, overlap1, routed, dispatch,overlap2]         comp_time = args.n_dense_layers * (dense_mla + dense_mlp) + n_sparse_layers * (tp_mla + shared +routed)        comm_time = n_sparse_layers * (combine + dispatch)        sum_time = comp_time        if overlap1 >0:            sum_time += overlap1 * n_sparse_layers        if overlap2 >0:            sum_time += overlap2 * n_sparse_layers        df2.loc[len(df2)]= [ key, comp_time,comm_time,sum_time]    df = df.set_index('GPU').T    df2 = df2.set_index('GPU').T    if is_print :        df['Layers'] = df['Layers'].astype(int).astype(str)        print(df.to_markdown(floatfmt=".3f"))        print('-----------SUM-------------')        print(df2.to_markdown(floatfmt=".3f"))      return df,df2
```

TP=4时,分层的结果如下所示(单位ms), DGX-B200此时配置CX8-800Gbps网络还有部分无法Overlap的时间.

```
prefill_time(args,gpu_blackwell,seq_len, kv_cache_rate, tp=4, dp=8, is_print=True)
```

Layers

DGX-B200

GB200-NVL72

H200

H800

H20

MLA

3

1.644

1.479

4.107

4.116

33.087

DenseMLP

3

0.575

0.523

2.527

2.546

19.912

TP_MLA

58

0.504

0.463

1.198

1.396

8.443

Shared Expert

58

0.021

0.019

0.078

0.080

0.562

Combine

58

0.579

0.204

1.107

1.107

1.107

Overlap1

58

0.054

-0.278

-0.169

-0.369

-7.898

Routed Expert

58

0.165

0.153

0.623

0.640

4.499

Dispatch

58

0.182

0.089

0.314

0.314

0.314

Overlap2

58

0.017

-0.065

-0.309

-0.325

-4.185

累计的TTFT时间如下所示:

DGX-B200

GB200-NVL72

H200

H800

H20

Compute

46.654

42.869

130.081

142.723

942.273

Comm

44.134

16.981

82.467

82.467

82.467

Sum

50.789

42.869

130.081

142.723

942.273

TP=8时, 分层的结果如下所示(单位ms):

Layers

DGX-B200

GB200-NVL72

H200

H800

H20

MLA

3

1.644

1.479

4.107

4.116

33.087

DenseMLP

3

0.575

0.523

2.527

2.546

19.912

TP_MLA

58

0.299

0.278

0.685

0.882

4.308

Shared Expert

58

0.013

0.013

0.044

0.046

0.287

Combine

58

0.314

0.127

0.579

0.579

0.579

Overlap1

58

0.002

-0.164

-0.150

-0.349

-4.016

Routed Expert

58

0.107

0.101

0.353

0.369

2.299

Dispatch

58

0.116

0.069

0.182

0.182

0.182

Overlap2

58

0.009

-0.032

-0.171

-0.187

-2.117

累计的TTFT时间如下所示:

DGX-B200

GB200-NVL72

H200

H800

H20

Compute

30.974

28.757

82.657

95.235

558.837

Comm

24.967

11.390

44.134

44.134

44.134

Sum

31.622

28.757

82.657

95.235

558.837

综合分析, TP=8时的性能并没有比TP=4快一倍, 因此最优策略还是采用TP=4进行Prefill累计吞吐更高. 另外Prefill时GB200-NVL72也没有比DGX-B200有更大的优势. 但是Blackwell浮点性能翻倍和FP4 GEMM相对于H200还是可以获得3~4倍的性能提升.平均每卡输出Token

```
tp=4_ , ttft_sum = prefill_time(args,gpu_blackwell,seq_len, kv_cache_rate, tp=tp, dp=8, is_print=False)print(tp4_sum.apply(lambda x: seq_len/tp * (1000/ x)).loc['Sum'].to_markdown(floatfmt=".1f"))
```

GPU

Sum

DGX-B200

21574.6

GB200-NVL72

25560.2

H200

8423.6

H800

7677.5

H20

1162.9

平均每卡的KVCache带宽(GB/s)如下:

```
print(tp4_sum.apply(lambda x: seq_len/tp * (1000/ x) * (args.kv_lora_rank + args.qk_rope_head_dim)/1024/1024).loc['Sum'].to_markdown(floatfmt=".1f"))
```

GPU

Sum

DGX-B200

11.9

GB200-NVL72

14.0

H200

4.6

H800

4.2

H20

0.6

如果存储独立组网, GB200的Grace配置400Gbps带宽即可, 但对于DGX-B200可能需要在CPU节点配置1.2Tbps的存储网络, 并且还需要考虑CPU的转发性能.

## 4. Decoding阶段分析

### 4.1 最大BatchSize估计

Decoding阶段我们同时引入H20进行对比分析:

```
## Decoding阶段无需通信SMgpu_blackwell_decode = get_gpu_info('gpu_info.csv',                                     device_list=['DGX-B200','GB200-NVL72','H200','H20'],                                    decoding_mode=True)  
```

最大BatchSize估计如下, decode_len=1210:

```
def _decoding_batchsize(args:ModelArgs, gpu:GPU_perf, seq_len,decode_len,tp, expert_num):    mem_util_rate = 0.9#torch/activation等其它开销的折扣    mla = 187.17#MLA的参数(单位M)    expert_mem = 44.05#expert的参数(单位M)    others_parameter = 2.91#其它参数2.91GB    kv_cache = (seq_len+decode_len) * (args.kv_lora_rank + args.qk_rope_head_dim) *args.n_layers *tp    mem = gpu.mem * mem_util_rate - others_parameter - mla * args.n_layers/tp/1024    mem -= expert_mem *(args.n_layers - args.n_dense_layers) * expert_num /1024    return mem * 1024 * 1024 * 1024 / kv_cachedef decode_batchsize(args:ModelArgs, gpu_dict, seq_len,decode_len, tp):    df = pd.DataFrame(columns=['GPU','EP320','EP144','EP72','EP34'])    for key in gpu_dict.keys():        item = key        value = [item]        for exp_num in [2,3,5,9]:            bs = _decoding_batchsize(args, gpu_dict[key], seq_len,decode_len, tp,exp_num)            value.append(bs)        df.loc[len(df)]= value    print(df.set_index('GPU').to_markdown(floatfmt=".0f"))      return dfdecode_len = 1210df = decode_batchsize(args,gpu_blackwell_decode, seq_len,decode_len, tp=1)
```

TP=1时

GPU

EP320

EP144

EP72

EP34

DGX-B200

781

767

740

686

GB200-NVL72

840

826

799

745

H200

589

576

548

494

H20

368

354

327

273

TP=8时

GPU

EP320

EP144

EP72

EP34

DGX-B200

104

103

99

92

GB200-NVL72

112

110

107

100

H200

80

79

75

68

H20

53

51

48

41

注: 对于Sglang  DP MLA的分析后续会补上.

### 4.2 MLA计算耗时

Decode阶段不需要通信SM, 因此定义GPU如下

```
gpu_blackwell_decode = get_gpu_info('gpu_info.csv',                                     device_list=['DGX-B200','GB200-NVL72','H200','H20'],                                    decoding_mode=True) 
```

计算方式如下所示:

```
def decode_mla(args:ModelArgs, gpu_dict,bs_list, seq_len,decode_len,expert_num=2, is_print=False):    df = pd.DataFrame(columns=['GPU','BatchSize','TP','LoadKV','DenseMLA','SparseMLA'])    tp_list = [1,4,8]    for key in gpu_dict.keys():        for bs in bs_list:            kv_cache = seq_len * (args.kv_lora_rank + args.qk_rope_head_dim)  * bs             load_kv_time = kv_cache /1024/1024/1024 / gpu_dict[key].get_mem_bw() *1000            dense_mla,sparse_mla = mla_elapse_time(args,gpu_dict[key],                                     seq_len, kv_cache_rate=1,                                     tp=tp_list,                                     batchsize=bs,                                     decoding_mode=True,                                     enable_gemm_fp4=True)            for tp_num in tp_list:                max_bs = _decoding_batchsize(args, gpu_dict[key], seq_len, decode_len,expert_num=expert_num,tp=tp_num)                if bs > max_bs:                    continue                else:                    df.loc[len(df)]=[gpu_dict[key].gpu_type,bs,tp_num,load_kv_time,dense_mla,sparse_mla[tp_num]]    if is_print:        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))    return df    bs_list =[32, 64, 128, 256, 512]decode_mla(args, gpu_blackwell_decode, bs_list,seq_len,decode_len, is_print=True)
```

计算结果如下(单位ms):

GPU

BS

TP

LoadKV

DenseMLA

SparseMLA

DGX-B200

32

1

0.011

0.025

0.025

DGX-B200

32

4

0.011

0.025

0.022

DGX-B200

32

8

0.011

0.025

0.019

DGX-B200

64

1

0.022

0.050

0.050

DGX-B200

64

4

0.022

0.050

0.029

DGX-B200

64

8

0.022

0.050

0.022

DGX-B200

128

1

0.044

0.100

0.100

DGX-B200

128

4

0.044

0.100

0.042

DGX-B200

256

1

0.089

0.201

0.201

DGX-B200

512

1

0.177

0.402

0.402

GB200-NVL72

32

1

0.011

0.023

0.023

GB200-NVL72

32

4

0.011

0.023

0.021

GB200-NVL72

32

8

0.011

0.023

0.018

GB200-NVL72

64

1

0.022

0.045

0.045

GB200-NVL72

64

4

0.022

0.045

0.027

GB200-NVL72

64

8

0.022

0.045

0.022

GB200-NVL72

128

1

0.044

0.090

0.090

GB200-NVL72

128

4

0.044

0.090

0.040

GB200-NVL72

256

1

0.089

0.181

0.181

GB200-NVL72

512

1

0.177

0.362

0.362

H200

32

1

0.018

0.102

0.102

H200

32

4

0.018

0.102

0.042

H200

32

8

0.018

0.102

0.029

H200

64

1

0.037

0.160

0.160

H200

64

4

0.037

0.160

0.057

H200

64

8

0.037

0.160

0.037

H200

128

1

0.074

0.277

0.277

H200

128

4

0.074

0.277

0.089

H200

256

1

0.148

0.510

0.510

H200

512

1

0.295

0.975

0.975

H20

32

1

0.022

0.539

0.539

H20

32

4

0.022

0.539

0.151

H20

32

8

0.022

0.539

0.083

H20

64

1

0.044

1.025

1.025

H20

64

4

0.044

1.025

0.274

H20

128

1

0.089

1.998

1.998

H20

256

1

0.177

3.943

3.943

可以看到, 此时Load KVCache的影响变得非常显著. 因此在并行策略上需要进行取舍.

### 4.3 DenseMLP耗时

计算函数如下所示:

```
def decode_dense_mlp(args:ModelArgs, gpu_dict,bs_list,seq_len, decode_len,expert_num=2, is_print=False):    tp_list = [1,4,8] #only used for calc max batchsize    df = pd.DataFrame(columns=['GPU','BatchSize','TP','DenseMLP'])    for key in gpu_dict.keys():        for bs in bs_list:                      t = _prefill_dense_mlp(args,gpu_dict[key], bs)            for tp_num in tp_list:                max_bs = _decoding_batchsize(args, gpu_dict[key], seq_len, decode_len,expert_num=expert_num,tp=tp_num)                if bs > max_bs:                    continue                else:                    df.loc[len(df)]=[gpu_dict[key].gpu_type,bs,tp_num, t]    if is_print:            df['BatchSize'] = df['BatchSize'].astype(int).astype(str)        print(df[df['TP']==1][['GPU','BatchSize','DenseMLP']].set_index('GPU').to_markdown(floatfmt=".3f"))    return df    decode_dense_mlp(args, gpu_blackwell_decode, bs_list,seq_len,decode_len, is_print=True)
```

计算结果如下(单位ms):

GPU

BatchSize

DenseMLP

DGX-B200

32

0.059

DGX-B200

64

0.062

DGX-B200

128

0.069

DGX-B200

256

0.082

DGX-B200

512

0.109

GB200-NVL72

32

0.059

GB200-NVL72

64

0.062

GB200-NVL72

128

0.068

GB200-NVL72

256

0.079

GB200-NVL72

512

0.103

H200

32

0.108

H200

64

0.123

H200

128

0.153

H200

256

0.213

H200

512

0.334

H20

32

0.237

H20

64

0.363

H20

128

0.615

H20

256

1.119

### 4.4 MoE计算耗时

根据不同的GPU类型计算耗时, 其实这个和EP策略无关, 因为任何一个token都要dispatch 8份发到其它节点, 因此简化计算流程, 同时还需要考虑到GroupGEMM和相对较小的batchsize无法打满的影响, 这里按照DeepGEMM的性能, 定义了一个和batchsize相关的性能折算估计系数, 这里后面还是需要实际在机器上跑一下,只是拍脑袋预估.在Blackwell上相对影响较小.

```
def _decode_moe_expert(args:ModelArgs,gpu:GPU_perf,bs,expert_num=2):    load_time = moe_expert_mem(args) / gpu.get_mem_bw()    gpu_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops()!=0else gpu.get_fp8_flops()    #实际上在gpu.get_flops函数中已经有一个折扣.此时在这个基础上叠加一些折扣系数    if bs <= 32 :        gpu_flops *= 0.2    elif bs <= 64 :        gpu_flops *= 0.3    elif bs <= 128:        gpu_flops *= 0.5    else:        gpu_flops *= 0.7            shared_flops = moe_expert_flops(args, bs)    shared_time = shared_flops / gpu_flops + load_time    num_routed_token = bs * args.n_activated_experts    routed_flops = moe_expert_flops(args, num_routed_token)    routed_time = routed_flops / gpu_flops +load_time *(expert_num -1)    return shared_time, routed_timedef decode_moe_expert(args:ModelArgs,gpu_dict, bs_list,seq_len, decode_len,expert_num=2, is_print=False):    tp_list = [1,4,8] #only used for calc max batchsize        df = pd.DataFrame(columns=['GPU','BatchSize','TP','SharedExpert','RoutedExpert'])    for gpu_key in gpu_dict.keys():        for bs in bs_list:             s, r = _decode_moe_expert(args,gpu_dict[gpu_key], bs,expert_num= expert_num)            for tp_num in tp_list:                max_bs = _decoding_batchsize(args, gpu_dict[gpu_key], seq_len, decode_len,expert_num=expert_num,tp=tp_num)                if bs > max_bs:                    continue                else:                    df.loc[len(df)]=[gpu_dict[gpu_key].gpu_type,str(bs),tp_num, s,r]    if is_print:            df['BatchSize'] = df['BatchSize'].astype(int).astype(str)          print(df[df['TP']==1][['GPU','BatchSize','SharedExpert','RoutedExpert']].set_index('GPU').to_markdown(floatfmt=".3f"))    return dfdecode_moe_expert(args, gpu_blackwell_decode, bs_list,seq_len,decode_len, is_print=True)
```

计算结果如下(单位ms):

GPU

BatchSize

SharedExpert

RoutedExpert

DGX-B200

32

0.008

0.021

DGX-B200

64

0.009

0.026

DGX-B200

128

0.009

0.030

DGX-B200

256

0.010

0.040

DGX-B200

512

0.015

0.074

GB200-NVL72

32

0.008

0.019

GB200-NVL72

64

0.008

0.024

GB200-NVL72

128

0.009

0.027

GB200-NVL72

256

0.010

0.036

GB200-NVL72

512

0.014

0.067

H200

32

0.019

0.077

H200

64

0.021

0.100

H200

128

0.024

0.118

H200

256

0.029

0.163

H200

512

0.049

0.317

H20

32

0.082

0.572

H20

64

0.106

0.759

H20

128

0.124

0.909

H20

256

0.172

1.293

### 4.5 AlltoAll通信耗时

对于DGX-B200, AlltoAll继续采用IBGDA的方式, 直接通过RDMA传输, 因此计算时仅需要考虑GPU的PCIe带宽, 而对于GB200-NVL72按照EP72的方式仅在机内NVLINK通信, 计算函数如下所示:

```
def _moe_a2a(args:ModelArgs,gpu:GPU_perf,bs,fp8_combine=False ,static_latency=0.005):    dispatch_size = bs * args.dim * args.n_activated_experts /1024/1024    if fp8_combine & (gpu.get_fp4_flops()!=0): #支持FP4GPU才能开启FP8 Combine         combine_size = dispatch_size    else:        combine_size = dispatch_size * 2#FP16    comm_bw = gpu.get_pcie_bw() if gpu.gpu_per_node == 8else gpu.get_nvlink_bw()        dispatch_t = dispatch_size / comm_bw + static_latency    combine_t = combine_size / comm_bw +static_latency    return dispatch_t, combine_tdef decode_a2a(args:ModelArgs, gpu_dict,  bs_list, seq_len, decode_len,expert_num=2, is_print=False,fp8_combine=False):    tp_list = [1,4,8] #only used for calc max batchsize        df = pd.DataFrame(columns=['GPU','BatchSize','TP','Dispatch','Combine'])    for key in gpu_dict.keys():        for bs in bs_list:             dispatch_time, combine_time = _moe_a2a(args, gpu_dict[key],bs,fp8_combine=fp8_combine)            for tp_num in tp_list:                max_bs = _decoding_batchsize(args, gpu_dict[key], seq_len, decode_len,expert_num=expert_num,tp=tp_num)                if bs > max_bs:                    continue                else:                    df.loc[len(df)]=[gpu_dict[key].gpu_type,bs,tp_num,dispatch_time,combine_time]    if is_print:            df['BatchSize'] = df['BatchSize'].astype(int).astype(str)          print(df[df['TP']==1][['GPU','BatchSize','Dispatch','Combine']].set_index('GPU').to_markdown(floatfmt=".3f"))    return dfdecode_a2a(args,gpu_blackwell,bs_list,seq_len, decode_len,is_print=True, fp8_combine=False)
```

FP16 Combine的计算结果如下(单位ms):

GPU

BatchSize

Dispatch

Combine

DGX-B200

32

0.026

0.046

DGX-B200

64

0.046

0.087

DGX-B200

128

0.087

0.170

DGX-B200

256

0.170

0.334

DGX-B200

512

0.334

0.664

GB200-NVL72

32

0.007

0.010

GB200-NVL72

64

0.010

0.014

GB200-NVL72

128

0.014

0.023

GB200-NVL72

256

0.023

0.042

GB200-NVL72

512

0.042

0.078

H200

32

0.046

0.087

H200

64

0.087

0.170

H200

128

0.170

0.334

H200

256

0.334

0.664

H200

512

0.664

1.323

H800

32

0.046

0.087

H800

64

0.087

0.170

H800

128

0.170

0.334

H800

256

0.334

0.664

H20

32

0.046

0.087

H20

64

0.087

0.170

H20

128

0.170

0.334

H20

256

0.334

0.664

FP8 Combine的计算结果如下(单位ms):

GPU

BatchSize

Dispatch

Combine

DGX-B200

32

0.026

0.026

DGX-B200

64

0.046

0.046

DGX-B200

128

0.087

0.087

DGX-B200

256

0.170

0.170

DGX-B200

512

0.334

0.334

GB200-NVL72

32

0.007

0.007

GB200-NVL72

64

0.010

0.010

GB200-NVL72

128

0.014

0.014

GB200-NVL72

256

0.023

0.023

GB200-NVL72

512

0.042

0.042

H200

32

0.046

0.087

H200

64

0.087

0.170

H200

128

0.170

0.334

H200

256

0.334

0.664

H200

512

0.664

1.323

H20

32

0.046

0.087

H20

64

0.087

0.170

H20

128

0.170

0.334

H20

256

0.334

0.664

### 4.6 总耗时分析

汇总数据表如下所示:

```
from functools import reducedef _decode_time(args:ModelArgs, gpu, bs_list,seq_len, decode_len,expert_num=2,is_print=False,fp8_combine=False):    mla = decode_mla(args,gpu, bs_list,seq_len, decode_len,expert_num=expert_num)    dense_mlp = decode_dense_mlp(args,gpu, bs_list,seq_len, decode_len,expert_num=expert_num)    moe = decode_moe_expert(args,gpu, bs_list,seq_len, decode_len,expert_num=expert_num)    a2a = decode_a2a(args,gpu, bs_list,seq_len, decode_len,expert_num=expert_num, fp8_combine=fp8_combine)    dfs = [ mla, dense_mlp, moe, a2a]     for decode_df in dfs:        decode_df['BatchSize'] = decode_df['BatchSize'].astype(int).astype(str)     df = reduce(lambda left, right: pd.merge(left,right, on=['GPU','BatchSize','TP'], how='left'), dfs)    if is_print:            print(df.set_index('GPU').to_markdown(floatfmt=".3f"))    return df    dfs = _decode_time(args,gpu_blackwell_decode,bs_list,seq_len,decode_len,is_print=True)dfs.set_index('GPU').style.format(precision=3)   
```

![图片](assets/c6db530064c0.png)

Decoding阶段Overlap如下所示:

![图片](assets/634a0c35b67c.png)

我们针对模型结构和最优TP策略进行修正,并计算TPOT,如下所示:

```
def decode_time(args:ModelArgs, gpu_dict,bs_list,seq_len, decode_len,expert_num=2,tps_limit=0, is_print=False,fp8_combine=False):    df =  _decode_time(args,gpu_dict,bs_list,seq_len, decode_len,expert_num=2,fp8_combine=fp8_combine)    def overlap_adjust(r):        if r['Delta'] > 0:            return r['TPOT_O']+ r['Delta']  * (args.n_layers - args.n_dense_layers)        else:            return r['TPOT_O']    # 修正TP执行时间, 按照加载FP8的KV计算    df['DenseMLA'] = df['DenseMLA'] + df['LoadKV']    df['SparseMLA'] = df['SparseMLA'] + df['LoadKV']    df['COMP_SUM'] = df['SparseMLA'] +df['SharedExpert']+ df['RoutedExpert']    df['COMM_SUM'] = df['Dispatch'] + df['Combine']    df['Delta'] = df['COMM_SUM'] - df['SparseMLA'] - df['SharedExpert']    df['TPOT_O'] = (df['DenseMLA'] + df['DenseMLP']) * args.n_dense_layers     df['TPOT_O'] += (df['SparseMLA'] + df['SharedExpert'] + df['RoutedExpert']) * (args.n_layers - args.n_dense_layers)        df['TPOT'] = df.apply(lambda row:  overlap_adjust(row),axis=1)    df = df[['GPU','TP','BatchSize','DenseMLA','DenseMLP','SparseMLA','Combine','SharedExpert','RoutedExpert','Dispatch','COMP_SUM','COMM_SUM', 'Delta','TPOT','TPOT_O']]    df['TPS'] = 1000 / df['TPOT']    df['TPS_O'] = 1000 / df['TPOT_O']        df['Total'] =  df['TPS'] * df['BatchSize'].astype(int)    df['Total_O'] =  df['TPS_O'] * df['BatchSize'].astype(int)    df['Comm_Impact'] = (df['Total_O'] - df['Total'] )/ df['Total_O']    df = df[df['TPS']>tps_limit]    if is_print:        print(df.set_index('GPU').T.to_markdown(floatfmt=".3f"))    return df    dfs = decode_time(args,gpu_blackwell_decode,bs_list,seq_len,decode_len,is_print=True)cm = sns.light_palette("red", as_cmap=True)dfs.style.bar(subset=['TPS','Total'],color='#6495ED').background_gradient(subset=['Comm_Impact'],cmap=cm).format(precision=3)  
```

![图片](assets/96d59d2f2b79.png)

可以看到在ScaleOut网络中, DGX-B200即便是每卡配置800Gbps带宽,还是存在大量的无法Overlap的情况, 性能影响33~54%,  同样H200也会因为无法Overlap产生影响, 而GB200-NVL72则可以完全Overlap.

如果采用FP8 Combine, DGX-B200依旧也无法Overlap.

```
gpu_blackwell_decode2 = get_gpu_info('gpu_info.csv',                                     device_list=['DGX-B200','GB200-NVL72'],                                    decoding_mode=True)  dfs = decode_time(args,gpu_blackwell_decode,bs_list,seq_len,decode_len,is_print=False,fp8_combine=True)dfs.style.bar(subset=['TPS','Total'],color='#6495ED').background_gradient(subset=['Comm_Impact'],cmap=cm).format(precision=3)  
```

![图片](assets/d2876e0a5173.png)

## 5. 结论

### 5.1  DGX-B200 ScaleOut带宽

假设我们有一个DGX-B200 PCIe-Gen7支持CX9 1.6T的网卡, 那么性能可以基本追平GB200-NVL72, 而当前配置CX7的DGX-GB200在DeepSeek-V3模型下性能损失接近60%和H200没有差异. 即便采用FP8 Combine, 性能损失也接近50%

```
gpu_blackwell_decode3 = get_gpu_info('gpu_info.csv',                                     device_list=['DGX-B200','GB200-NVL72'],                                    decoding_mode=True)  b200_cx7 = copy.deepcopy(gpu_blackwell['DGX-B200'])b200_cx7.gpu_type = b200_cx7.gpu_type + '-CX7'b200_cx7.pcie_bw = 50gpu_blackwell_decode3[b200_cx7.gpu_type] = b200_cx7b200_cx9 = copy.deepcopy(gpu_blackwell['DGX-B200'])b200_cx9.gpu_type = b200_cx9.gpu_type + '-CX9'b200_cx9.pcie_bw = 200gpu_blackwell_decode3[b200_cx9.gpu_type] = b200_cx9dfs = decode_time(args,gpu_blackwell_decode3,bs_list,seq_len,decode_len,is_print=False,fp8_combine=False)dfs.style.bar(subset=['TPS','Total'],color='#6495ED').background_gradient(subset=['Comm_Impact'],cmap=cm).format(precision=3)  
```

采用FP16 Combine的数据

![图片](assets/c34198740c75.png)

采用FP8 Combine的数据

![图片](assets/0131dec7edb2.png)

### 5.2 所有卡汇总结果

```
gpu_all_decode = get_gpu_info('gpu_info.csv', decoding_mode=True)  bs_list_new = [32,64,128,256,512]dfs = decode_time(args,gpu_all_decode,bs_list,seq_len,decode_len,is_print=False,fp8_combine=True)dfs.style.bar(subset=['TPS','Total'],color='#6495ED').background_gradient(subset=['Comm_Impact'],cmap=cm).format(precision=3) 
```

![图片](assets/f2526c1bc331.png)

排序整理最佳策略后:

```
dfs_o = dfs.groupby(['GPU','BatchSize'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)dfs_o.style.bar(subset=['TPS','Total'],color='#6495ED').background_gradient(subset=['Comm_Impact'],cmap=cm).format(precision=3) 
```

![图片](assets/0246e244109d.png)

当前B200主要瓶颈是ScaleOut网络的带宽, 假设存在支持CX9的B200, 性能如下:

```
dfs = decode_time(args,gpu_blackwell_decode3,bs_list,seq_len,decode_len,is_print=False,fp8_combine=True)dfs_o = dfs.groupby(['GPU','BatchSize'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)dfs_o.style.bar(subset=['TPS','Total'],color='#6495ED').background_gradient(subset=['Comm_Impact'],cmap=cm).format(precision=3) 
```

![图片](assets/6ad1b50973fb.png)

### 5.3 针对不同Seq_len分析

考虑到Reasoning的长seq_len影响, 统计如下

```
gpu_all_decode = get_gpu_info('gpu_info.csv', decoding_mode=True)  bs_list_new = [32,64,128,256,512]dfs = []for seq_len in range(1024,16384,16):    df = decode_time(args,gpu_all_decode,bs_list,seq_len,decode_len,is_print=False,fp8_combine=True)    df['Seq_len'] = seq_len    df_o = df.groupby(['GPU','BatchSize'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)    df_o.drop_duplicates(subset=['GPU','BatchSize'], keep='first', inplace=True)    dfs.append(df_o)df = pd.concat(dfs)     
```

绘图的函数

```
import numpy as npimport matplotlib.pyplot as pltimport seaborn as snsdef df_filter(df, gpu, bs,value_list):    df1 =  df[ (df['GPU']==gpu) & (df['BatchSize']==str(bs))][value_list]    return df1def draw(df,gpu_dict,val,val_unit_name):    num_gpu = len(gpu_dict)    height= 4 * num_gpu    fig, axs = plt.subplots(nrows=num_gpu, ncols=1, figsize=(9, height))    ax12 = axs[0]    ax22 = axs[1]    #fig.suptitle(title, y=0.97,fontsize='large')    value_list=[val,'Seq_len']    cnt = 0    for key in gpu_dict.keys():        axt = axs[cnt]        df1 = df_filter(df,key,32,value_list)        df2 = df_filter(df,key,128,value_list)            sns.lineplot(x='Seq_len',y=val,data=df1,color="deepskyblue",ax=axt)        sns.lineplot(x='Seq_len',y=val,data=df2,color="#698339",ax=axt)        axt.set_ylabel(val_unit_name)        axt.set_xlabel(key)        cnt+=1        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)    #plt.savefig(title.replace(' ','_')+'.png',bbox_inches='tight', pad_inches=0.05)    plt.show()
```

每卡的每秒token数, 蓝线为BatchSize=32, 绿线为BatchSize=128

```
draw(df,gpu_all_decode,'Total','tokens per second')
```

![图片](assets/703fb315772e.png)

每个Session的TPS:

```
draw(df,gpu_all_decode,'TPS','tokens per second')
```

![图片](assets/d6ccd32d4f71.png)