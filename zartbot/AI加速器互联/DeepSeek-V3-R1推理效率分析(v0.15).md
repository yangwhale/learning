# DeepSeek-V3/R1推理效率分析(v0.15)

> 作者: zartbot  
> 日期: 2025年3月15日 07:30  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493447&idx=1&sn=7988d642c96c9bbf72d81d85fb40de44&chksm=f995f785cee27e93c421cc310539d718979464517c0f0524335da86b5c6899b016281582dc2f#rd

---

本文由于计算量非常大,步骤也很多难免有错误之处,欢迎大家指正.并且每个计算时使用的函数都已列出,可供大家自行修改. 本文仅代表个人观点, 与任职的机构无关.

Ver

0.1

initial version

0.11

修正prefill_mla_elapse_time函数的typo

0.12

根据留言@mp的指正, 修正MLA matsorb的计算, W_uv和W_o不吸收更快

0.15

修正H20 NVLINK和Memory带宽, 考虑batchsize小时, GEMM的性能损耗

TODO-list:

基于EP并行策略的修正, 考虑内存带宽的影响, 后面单独再分析一篇

每当看到代码里有Low Latency的字眼时, 网党们就激动的不行要去降低静态延迟扩大带宽, 上大规模ScaleUP网络, 但是事实上是否是这样呢? 本文作为第一篇分析一下H800和H20在DeepSeek-R1 EP并行的推理性能峰值, 后续将继续分析B200-NVL72这样的实例, 看看ScaleUP网络是否有优势.

### TL;DR

H800和H20分析结果如下所示, 基本上H800的数据能够和DeepSeek官方数据对齐.

Prefill阶段

H800

H20

Sum(Overlap)

42277.8

7959.7

Sum

29284.1

7346.0

Decoding阶段, 这个版本对于GEMM在BatchSize较小的情况下进行了修正.

seq_len =4383

H800_TP1

H800_TP1

H800_TP1

H800_TP4

H20_TP4

H20_TP4

H20_3e_TP8

H20_3e_TP8

H20_3e_TP8

BatchSize

32.000

64.000

128.000

256.000

32.000

64.000

32.000

64.000

128.000

DenseMLA

0.098

0.196

0.392

0.785

0.530

1.060

0.526

1.052

2.105

DenseMLP

0.019

0.038

0.075

0.151

0.126

0.252

0.126

0.252

0.504

SparseMLA

0.098

0.196

0.392

0.681

0.306

0.455

0.239

0.320

0.484

Combine

0.082

0.165

0.329

0.659

0.082

0.165

0.082

0.165

0.329

SharedExpert

0.010

0.014

0.017

0.024

0.070

0.093

0.070

0.093

0.112

RoutedExpert

0.084

0.112

0.134

0.191

0.560

0.747

0.560

0.747

0.896

Dispatch

0.041

0.082

0.165

0.329

0.041

0.082

0.041

0.082

0.165

COMP_SUM

0.192

0.322

0.543

0.897

0.936

1.295

0.869

1.161

1.493

COMM_SUM

0.124

0.247

0.494

0.988

0.124

0.247

0.124

0.247

0.494

TPOT(Overlap)

11.505

19.366

32.901

54.822

56.248

79.043

52.342

71.232

94.391

TPOT

18.670

33.695

61.560

112.140

63.412

93.372

59.507

85.561

123.050

TPS_O

86.918

51.637

30.395

18.241

17.779

12.651

19.105

14.039

10.594

TPS

53.562

29.678

16.244

8.917

15.770

10.710

16.805

11.688

8.127

Total_O

2781.372

3304.787

3890.496

4669.646

568.913

809.685

611.366

898.479

1356.054

Total

1713.996

1899.377

2079.289

2282.865

504.634

685.427

537.756

748.005

1040.225

H20峰值Tokens/s在放宽到TPS=10时, 可以做到809Tokens/s, H20-3E可以做到845Tokens/s, 但如果要满足20TPS的需求, H20需要更小的BatchSize或者在线上环境中, 考虑到一些短的seq_len请求, 也可以满足需求, 例如seqlen=2000时的性能:

seq_len =2000

H800_TP1

H800_TP1

H800_TP1

H800_TP4

H20_TP4

H20_TP4

H20_TP4

H20_TP4

H20_3e_TP8

H20_3e_TP8

H20_3e_TP8

H20_3e_TP8

BatchSize

32.000

64.000

128.000

256.000

32.000

64.000

128.000

256.000

32.000

64.000

128.000

256.000

DenseMLA

0.098

0.196

0.392

0.785

0.530

1.060

2.119

4.239

0.526

1.052

2.105

4.209

DenseMLP

0.019

0.038

0.075

0.151

0.126

0.252

0.504

1.008

0.126

0.252

0.504

1.008

SparseMLA

0.098

0.196

0.392

0.681

0.306

0.455

0.753

1.349

0.239

0.320

0.484

0.812

Combine

0.082

0.165

0.329

0.659

0.082

0.165

0.329

0.659

0.082

0.165

0.329

0.659

SharedExpert

0.010

0.014

0.017

0.024

0.070

0.093

0.112

0.160

0.070

0.093

0.112

0.160

RoutedExpert

0.084

0.112

0.134

0.191

0.560

0.747

0.896

1.280

0.560

0.747

0.896

1.280

Dispatch

0.041

0.082

0.165

0.329

0.041

0.082

0.165

0.329

0.041

0.082

0.165

0.329

COMP_SUM

0.192

0.322

0.543

0.897

0.936

1.295

1.761

2.789

0.869

1.161

1.493

2.252

COMM_SUM

0.124

0.247

0.494

0.988

0.124

0.247

0.494

0.988

0.124

0.247

0.494

0.988

TPOT(Overlap)

11.505

19.366

32.901

54.822

56.248

79.043

110.015

177.527

52.342

71.232

94.391

146.281

TPOT

18.670

33.695

61.560

112.140

63.412

93.372

138.673

234.845

59.507

85.561

123.050

203.598

TPS_O

86.918

51.637

30.395

18.241

17.779

12.651

9.090

5.633

19.105

14.039

10.594

6.836

TPS

53.562

29.678

16.244

8.917

15.770

10.710

7.211

4.258

16.805

11.688

8.127

4.912

Total_O

2781.372

3304.787

3890.496

4669.646

568.913

809.685

1163.482

1442.035

611.366

898.479

1356.054

1750.059

Total

1713.996

1899.377

2079.289

2282.865

504.634

685.427

923.032

1090.083

537.756

748.005

1040.225

1257.377

本文目录如下:

```
1. DeepSeek-V3/R1模型架构及计算复杂度分析1.1 MLA计算复杂度1.1.1 标准模式1.1.2 矩阵吸收模式1.2 DenseMLP计算复杂度1.3 MoE Expert计算复杂度1.4 数据汇总2. Prefill阶段2.1 MLA计算耗时2.2 DenseMLP计算耗时2.3 MoE计算耗时2.4 AlltoAll通信耗时2.5 总耗时2.6 Overlap分析2.7 KVCache计算3. Decoding阶段3.1 EP策略分析3.2 Memory利用率分析3.3 MLA耗时3.4 DenseMLP耗时3.5 AlltoAll通信耗时3.6 总耗时3.7 Overlap分析4. 小结
```

### 1. DeepSeek-V3/R1模型架构及计算复杂度分析

DeepSeek-V3/R1模型架构如下

![图片](assets/c926154683c6.png)

模型的参数定义如下

```
class ModelArgs:    max_batch_size: int = 8    max_seq_len: int = 4096 * 4    vocab_size: int = 129280    dim: int = 7168    inter_dim: int = 18432    moe_inter_dim: int = 2048    n_layers: int = 61    n_dense_layers: int = 3    n_heads: int = 128    # moe    n_routed_experts: int = 256    n_shared_experts: int = 1    n_activated_experts: int = 8    n_expert_groups: int = 8    n_limited_groups: int = 4    route_scale: float = 2.5    # mla    q_lora_rank: int = 1536    kv_lora_rank: int = 512    qk_nope_head_dim: int = 128    qk_rope_head_dim: int = 64    v_head_dim: int = 128
```

虽然各个模块的浮点运算量, 参数量虽然一种很简便的办法, 使用`ptflops`库中的`get_model_complexity_info`直接处理block得出, 但是该库对于一些相对复杂的运算还是有一些错误, 本文都进行了手工校正

#### 1.1 MLA计算复杂度
1.1.1 标准实现
MLA模块代码来自DeepSeek-V3 Github示例, 它是一个标准的MLA实现

```
class MLA(nn.Module):    def __init__(self, args: ModelArgs):        super().__init__()        self.dim = args.dim #隐藏层维度        self.n_heads = args.n_heads         self.n_local_heads = args.n_heads // world_size        self.q_lora_rank = args.q_lora_rank #q的低秩压缩的维度        self.kv_lora_rank = args.kv_lora_rank #kv的低秩压缩的维度        self.qk_nope_head_dim = args.qk_nope_head_dim #qk不带旋转位置编码的头的维度        self.qk_rope_head_dim = args.qk_rope_head_dim #qk旋转位置编码的头的维度        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim        self.v_head_dim = args.v_head_dim #v的多头注意力中头的维度                self.wq_a = nn.Linear(self.dim, self.q_lora_rank)        #q的down-projection矩阵                self.q_norm = nn.RMSNorm(self.q_lora_rank)                self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)        #q的up-projection矩阵                self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)        # wkv_a为K和V的down-projection矩阵        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)                self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))        # wkv_b为K和V的up-projection矩阵                self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim) #output权重矩阵        self.softmax_scale = self.qk_head_dim ** -0.5#计算1/sqrt(d_k)        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)            def forward(self, x: torch.Tensor):        bsz, seqlen, _ = x.size()        start_pos = 1        end_pos = start_pos + seqlen        # ---- 计算q--------        q = self.wq_b(self.q_norm(self.wq_a(x)))        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) #分离nope,rope        q_pe = apply_rotary_emb(q_pe, freqs_cis) #执行RoPE计算                # ----计算KV----------        kv = self.wkv_a(x)        #KV-Cache大小为wkv_a outputdim(self.kv_lora_rank + self.qk_rope_head_dim)        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) #分离KV和K位置编码        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) #执行RoPE计算                # -----处理KV u-pprojection矩阵        wkv_b = self.wkv_b.weight         wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)                # q中不需要位置编码的先和K的不需要位置编码的权重相乘        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)#保存KV Cache        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2) #保存K的位置编码Cache(pe cache)                # 计算QK^T/sqrt(d_k)        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +                  torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)                # 计算V        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])                x = self.wo(x.flatten(2)) #wo权重, 从n_head * v_head_dim -> dim        return x
```

为了便于理解具体的计算流程, 我们将代码执行流程图如下图所示:

![图片](assets/e05a6c182be2.png)

从图上可知, 单个Token的KVCache用量从`forward`函数中的`kv = self.wkv_a(x)`中得知, 维度为kv_lora_rank(512)+ qk_rope_head_dim(64) 为 576.

分析计算复杂度如下:

```
args = ModelArgs()m = MLA(args)num_tokens = 1mla_flops, mla_params = get_model_complexity_info(m, (num_tokens,args.dim),as_strings=True,print_per_layer_stat=True)##输出结果如下MLA(187.17 M, 99.999% Params, 170.36 MMac, 100.000% MACs,   (wq_a): Linear(11.01 M, 5.883% Params, 11.01 MMac, 6.464% MACs, in_features=7168, out_features=1536, bias=True)  (q_norm): RMSNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (1536,), eps=None, elementwise_affine=True)  (wq_b): Linear(37.77 M, 20.181% Params, 37.77 MMac, 22.172% MACs, in_features=1536, out_features=24576, bias=True)  (wkv_a): Linear(4.13 M, 2.206% Params, 4.13 MMac, 2.424% MACs, in_features=7168, out_features=576, bias=True)  (kv_norm): RMSNorm(0, 0.000% Params, 0.0 Mac, 0.000% MACs, (512,), eps=None, elementwise_affine=True)  (wkv_b): Linear(16.81 M, 8.981% Params, 0.0 Mac, 0.000% MACs, in_features=512, out_features=32768, bias=True)  (wo): Linear(117.45 M, 62.748% Params, 117.45 MMac, 68.940% MACs, in_features=16384, out_features=7168, bias=True))
```

即单个MLA block有187.17M个参数, 参数数量没啥问题.

但是单个Token的计算复杂度为 170.36M Mac这个值实际上是有错误的, wkv_b由于split为w_uk和w_uv, 算力消耗没有计算, 因此我们定义了一个手工计算的函数

```
def mla_flops(q_len, kv_len, args:ModelArgs, kv_cache_rate=0):    #calculate MACs and estimate Flops approx. 2xMAC.    q_down_proj = q_len * args.dim * args.q_lora_rank #wq_a    q_up_proj = q_len * args.q_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim) #wq_b    kv_down_proj = kv_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a    k_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim #w_uk    v_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.v_head_dim #w_uv    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)    gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj        #把它看成一个标准的args.n_heads的MHA    mha = args.n_heads * ( q_len * args.qk_rope_head_dim * kv_len #QK_score_rope                          + q_len * args.qk_nope_head_dim * kv_len #QK_score_nope                          + q_len * kv_len * args.v_head_dim) #ScoreV    wo = q_len * args.n_heads * args.v_head_dim * args.dim #wo    attn_sum = mha + wo        #return flops by 2* Sum(MACs)    GEMM_FP8_FLOPS = gemm_sum * 2/1e9    ATTN_FP16_FLOPS =  attn_sum * 2/1e9        return GEMM_FP8_FLOPS+ATTN_FP16_FLOPS, GEMM_FP8_FLOPS,ATTN_FP16_FLOPS
```

单个token的实际运算复杂度为:

```
mla_flops(1,1,args,0)(0.37429248000000004, 0.139329536, 0.234962944)
```
1.1.2 矩阵吸收模式
这里还需要再提一点, 在DeepSeek-V2的论文中提到

Fortunately, due to the associative law of matrix multiplication, we can absorb 𝑊_𝑈𝐾 into 𝑊𝑈𝑄, and 𝑊_𝑈𝑉 into 𝑊𝑂

WU_Q其实就是上文代码中的`wq_b`. 在上图中第(3)步前可以将`w_uk`先和`wq_b`相乘. 以及在第(7)步中可以`w_uv`和`wo`相乘.如下图所示:

![图片](assets/708ba44dd658.png)

**wq_b_nope**为[`q_lora_rank`(1536), `n_head`(128) x `qk_nope_head_dim`(128)]矩阵

**w_uk**为[`kv_lora_rank`(512) , `n_head`(128) x `qk_nope_head_dim`(128)]矩阵

矩阵吸收以后的**q_absorb**为['q_lora_rank'(1536),`h_head`(128)x `kv_lora_rank`(512)].

同理对wo吸收wu_v分析如下:

**wo**为[`n_head`(128) x `v_head_dim`(128), `dim`(7168)]矩阵

**w_uv**为[`kv_lora_rank`(512) , `n_head`(128) x `v_head_dim`(128)]矩阵

矩阵吸收以后的**o_absorb**为[`dim`(7168),`h_head`(128)x `kv_lora_rank`(512)].

对于算力消耗定义一个函数如下:

```
def mla_matabsob_flops(q_len, kv_len, args:ModelArgs, kv_cache_rate=0):    #calculate MACs and estimate Flops approx. 2xMAC.    q_down_proj = q_len * args.dim * args.q_lora_rank #wq_a    q_rope_up_proj = q_len * args.q_lora_rank * args.n_heads * args.qk_rope_head_dim #wq_b_rope    q_absorb = q_len * args.n_heads * args.q_lora_rank * args.kv_lora_rank         kv_down_proj = kv_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a    kv_down_proj = kv_down_proj * (1 - kv_cache_rate) #KV-Cache命中率修正    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj         #把它看成一个标准的args.n_heads的MQA    mqa = args.n_heads * ( q_len * args.qk_rope_head_dim * kv_len #Score_rope                          + q_len * args.kv_lora_rank * kv_len #Score_nope                          + q_len * kv_len * args.kv_lora_rank) #Score V    o_absorb = q_len * args.n_heads * args.kv_lora_rank * args.dim     attn_sum = mqa + o_absorb        #return flops by 2* Sum(MACs)    gemm_sum =  gemm_sum * 2/1e9    attn_sum = attn_sum * 2/1e9        return gemm_sum + attn_sum, gemm_sum,attn_sum
```

对于单个Token的实际运算复杂度为:

```
mla_matabsob_flops(1,1,args,0)(1.196572672, 0.256770048, 0.939802624)
```

相对于非吸收的复杂度为`mla_matabsob_flops(1,1,args,0)[0] / mla_flops(1,1,args,0)[0]`, 运算复杂度反而增加了3.197倍.

经@mp指正, 在`wo`和`w_uv`计算时, FlashInfer的实现是吸收成`o_absorb`但是这个矩阵维度为[`dim`(7168),`h_head`(128)x `kv_lora_rank`(512)], 而Sglang的`sglang/python/sglang/srt/models/deepseek_v2.py`中的forward_absorb函数是采用不吸收的方式.

```
#计算MQAattn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)#计算W_UVattn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)#计算W_Ooutput, _ = self.o_proj(attn_output)
```

此时不吸收W_uv,则是两个乘法,运算复杂度会降低, 即如下图所示:

![图片](assets/274bac7115af.png)

此时的算力消耗为:

```
def mla_matabsob_flops_v2(q_len, kv_len, args:ModelArgs, kv_cache_rate=0):    #calculate MACs and estimate Flops approx. 2xMAC.    q_down_proj = q_len * args.dim * args.q_lora_rank #wq_a    q_rope_up_proj = q_len * args.q_lora_rank * args.n_heads * args.qk_rope_head_dim #wq_b_rope    q_absorb = q_len * args.n_heads * args.q_lora_rank * args.kv_lora_rank         kv_down_proj = kv_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a    kv_down_proj = kv_down_proj * (1 - kv_cache_rate) #KV-Cache命中率修正    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj         #把它看成一个标准的args.n_heads的MQA    mqa = args.n_heads * ( q_len * args.qk_rope_head_dim * kv_len #Score_rope                          + q_len * args.kv_lora_rank * kv_len #Score_nope                          + q_len * kv_len * args.kv_lora_rank) #Score V        attn_up_proj = q_len * args.n_heads * args.v_head_dim * args.kv_lora_rank    o_proj = q_len * args.n_heads * args.v_head_dim * args.dim    attn_sum = mqa + attn_up_proj + o_proj        #return flops by 2* Sum(MACs)    gemm_sum =  gemm_sum * 2/1e9    attn_sum = attn_sum * 2/1e9        return gemm_sum + attn_sum, gemm_sum,attn_sum
```

对比性能差异, Decoding阶段性能会快29%.

```
seq_len = 4383mla_matabsob_flops_v2(1,seq_len,args,1)[0] / mla_matabsob_flops(1,seq_len,args,1)[0] 0.7144392373982301
```

对于吸收后的模型参数估计如下:

```
def mla_matabsob_mem(args:ModelArgs):    q_down_proj = args.dim * args.q_lora_rank #wq_a    q_rope_up_proj =  args.q_lora_rank * args.n_heads * args.qk_rope_head_dim #wq_b_rope    q_absorb = args.n_heads * args.q_lora_rank * args.kv_lora_rank     kv_down_proj =  args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a    attn_up_proj = args.n_heads * args.v_head_dim * args.kv_lora_rank    o_proj = args.n_heads * args.v_head_dim * args.dim    return q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj + attn_up_proj + o_projmla_matabsob_mem(args)/1e6254.214144
```

参数数量为254.21M, 参数规模也增加了1.35倍.

但是, MLA_Absorb在Decoding阶段会有额外的收益, 有官方的数据《DeepSeek-V3 / R1 推理系统概览》[1],平均每输出一个 token 的 KVCache 长度是4989, 以此计算两者有着显著的差异.

```
#Prefillmla_matabsob_flops_v2(4989,4989,args,0)[0] / mla_flops(4989,4989,args,0)[0]2.424#Decoding时qlen=1,KVcache不需要计算kv_cache_rate=1mla_matabsob_flops(1,4989,args,1)[0] / mla_flops(1,4989,args,1)[0]0.0112
```

结论: 在Prefill阶段采用非吸收的版本, 在Decoding采用矩阵吸收的版本.

#### 1.2 DenseMLP计算复杂度

在模型的前三层采用Dense MLP, 其计算复杂度如下

```
class DenseMLP(nn.Module):    def __init__(self, dim: int, inter_dim: int):        super().__init__()        self.w1 = nn.Linear(dim, inter_dim, dtype=torch.bfloat16)        self.w2 = nn.Linear(inter_dim, dim, dtype=torch.bfloat16)        self.w3 = nn.Linear(dim, inter_dim, dtype=torch.bfloat16)    def forward(self, x: torch.Tensor) -> torch.Tensor:        return self.w2(F.silu(self.w1(x)) * self.w3(x))args = ModelArgs()#dim=7168,inter_dim=18432d = DenseMLP(args.dim, args.inter_dim)num_tokens = 1mlp_flops, mlp_params = get_model_complexity_info(d, (1,num_tokens,args.dim),as_strings=True,print_per_layer_stat=True)##输出结果如下:DenseMLP(396.41 M, 100.000% Params, 396.41 MMac, 99.995% MACs,   (w1): Linear(132.14 M, 33.334% Params, 132.14 MMac, 33.333% MACs, in_features=7168, out_features=18432, bias=True)  (w2): Linear(132.13 M, 33.331% Params, 132.13 MMac, 33.330% MACs, in_features=18432, out_features=7168, bias=True)  (w3): Linear(132.14 M, 33.334% Params, 132.14 MMac, 33.333% MACs, in_features=7168, out_features=18432, bias=True))
```

单个MLP block有396.41M个参数, 单个Token的计算复杂度为 396.41M Mac~792.82MFLOPS.定义DenseMLP计算复杂度函数如下:

```
def densmlp_flops(args:ModelArgs, seq_len):    return 3 * seq_len * args.dim * args.inter_dim *2 /1e9
```

#### 1.3 MoE Expert计算复杂度

在模型的后58层采用了MoE,  其计算复杂度如下

```
class Expert(nn.Module):    def __init__(self, dim: int, inter_dim: int):        super().__init__()        self.w1 = nn.Linear(dim, inter_dim, dtype=torch.bfloat16)        self.w2 = nn.Linear(inter_dim, dim, dtype=torch.bfloat16)        self.w3 = nn.Linear(dim, inter_dim, dtype=torch.bfloat16)    def forward(self, x: torch.Tensor) -> torch.Tensor:        return self.w2(F.silu(self.w1(x)) * self.w3(x))args = ModelArgs()num_tokens = 1#dim=7168,moe_inter_dim=2048e = Expert(args.dim, args.moe_inter_dim)        moe_flops, moe_params = get_model_complexity_info(e, (1,num_tokens,args.dim),as_strings=True,print_per_layer_stat=True)##输出结果如下:Expert(44.05 M, 100.000% Params, 44.05 MMac, 99.995% MACs,   (w1): Linear(14.68 M, 33.329% Params, 14.68 MMac, 33.328% MACs, in_features=7168, out_features=2048, bias=True)  (w2): Linear(14.69 M, 33.341% Params, 14.69 MMac, 33.340% MACs, in_features=2048, out_features=7168, bias=True)  (w3): Linear(14.68 M, 33.329% Params, 14.68 MMac, 33.328% MACs, in_features=7168, out_features=2048, bias=True))
```

单个MoE Expert有44.05M个参数, 单个Token的计算复杂度为 44.05M Mac~88.1MFLOPS, 定义MoE Expert计算复杂度函数如下:

```
def moe_expert_flops(args:ModelArgs, seq_len):    return 3 * seq_len * args.dim * args.moe_inter_dim *2/1e9
```

#### 1.4 数据汇总

模型整体的参数分布如表所示, 另外模型还有MoE Gating函数, 参数量为`dim x n_routed_expert + n_routed_expert(bias)=1.83M`, 以及Embedding和Output层的参数`vocab_size x dim=926.67M`,

Block

单层参数量

层数

累计参数

MLA

187.17M

61

11.41B

DenseMLP

396.41M

3

1.19B

Expert

44.05Mx(256_routed+1_shared)

58

656.6B

Gate

1.83M

58

106.14M

Embedding

926.67M

1

926.67M

Output

926.67M

1

926.67M

SUM

-

-

671.16B

对于不同block的算力消耗统计如下

Block

参数量

运算复杂度(FLops)

KVCache用量

MLA

187.17M

374.29M

576B(FP8)

MLA_absorb

254.21M

508.7M

576(FP8)

DenseMLP

396.41M

792.82M

-

Expert

44.05 M

488.1M

-

实际计算时会按照具体的Prefill和Decode以及KVCache命中率进行评估.

**KVCache用量** :单个Token 的KVCache需要累积61层,实际消耗按照FP16保存KVCache为 2x 576x 61 =68.62KB. 按照FP8保存为34.31KB.

H20/H800算力指标如下表所示

GPU类型

SM

FP16算力

FP8算力

显存大小

显存带宽

NVLINK带宽

PCIe带宽

H800

132

989.5

1979

80GB

3350

200

50

H20

78

148

296

96GB

4000

450

50

注: 算力单位为TFLOPS,带宽单位为GB/s

为了便于后续计算, 定义GPU性能函数如下所示, GPU性能估计按照峰值的85%估计. H800需要24个通信SM. 这里考虑到H20浮点算力比较弱, H20估计需要10个通信SM,

```
class GPU_perf():    def __init__(self,sm,comm_sm, fp16_flops,fp8_flops,mem,mem_bw, nvlink_bw,pcie_bw, discount_rate):        self.sm = sm        self.comm_sm = comm_sm #用于通信的SM数量        self.fp16_flops = fp16_flops        self.fp8_flops = fp8_flops        self.mem = mem        self.mem_bw = mem_bw        self.nvlink_bw = nvlink_bw        self.pcie_bw = pcie_bw        self.discount_rate = discount_rate #整体性能按峰值性能折扣        #TODO: 可以分离网络性能折扣和算力性能折扣    def get_fp16_flops(self):        return self.fp16_flops * self.discount_rate  * ( self.sm  - self.comm_sm) / self.sm    def get_fp8_flops(self):        return self.fp8_flops *  self.discount_rate * ( self.sm  - self.comm_sm) / self.sm    def get_mem_bw(self):        return self.mem_bw *  self.discount_rate    def get_nvlink_bw(self):        return self.nvlink_bw *  self.discount_rate    def get_pcie_bw(self):        return self.pcie_bw *  self.discount_rateh800 = GPU_perf( sm = 132 ,comm_sm = 20,                  fp16_flops = 791.6, fp8_flops = 1583.2,                  mem = 80,mem_bw = 4000,                 nvlink_bw = 200,pcie_bw = 50,                 discount_rate = 0.85)h20 = GPU_perf( sm = 78 ,comm_sm = 10,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 96,mem_bw = 4000,                 nvlink_bw = 450,pcie_bw = 50,                 discount_rate = 0.85)gpu = dict({'H800': h800, 'H20': h20})
```

### 2. Prefill阶段

从DeepSeek官方的报告可知, Prefill：路由专家 EP32、MLA 和共享专家 DP32，一个部署单元是 4 节点，32 个冗余路由专家，每张卡 9 个路由专家和 1 个共享专家. 另一方面Attention计算并行策略参考论文中的描述

The minimum deployment unit of prefilling stage consists of 4 nodes with 32 GPUs. The attention part employs 4-way Tensor Parallelism (TP4) with Sequence Parallelism (SP), combined with 8-way Data Parallelism (DP8). For the MoE part, we use 32-way Expert Parallelism (EP32)

以Attention的视角来看, 推理请求在API Server通过负载均衡器以DP=8分配到不同的Prefill节点的DP组内, 一个DP组内有4张H800构成一个TP+SP的组进行MLA计算. 然后以MoE的视角来看, 32个GPU组成一个EP32的Group, 每一层256个Expert平均每张卡8个Routed Expert,然后每一张卡还有一个Shared Expert, 并根据论文再承载一个Redundant Expert, 累计10个Expert.

#### 2.1 MLA计算耗时

按照文章《DeepSeek V3/R1 推理效率分析（2）: DeepSeek 满血版逆向工程分析》[2]中提到的知乎作者`@天阿西吧`提到的Prefill和Decoding长度分析:

假设P代表sequence的平均输入长度，D代表sequence的平均输出长度，那对于每一个输出token的平均KVcache的长度约等于P+D/2=4989; 再加上P/D=608B/168B；P的取值大概为4383，D的取值大概为1210

以平均Prefill `seq_len`为4383计算, KVCache命中率按照官方的 56.3%计算, GPU性能估计按照峰值的85%估计.

定义计算函数如下, 并考虑TP并行的情况, 我们假设在`seq_len`中有56.3%的长度是可以从KVCache中提取的, 那么Prefill的时候就需要计算(1-kv_cache_rate)的token.

```
def prefill_mla_elapse_time(args:ModelArgs, gpu:GPU_perf, seq_len, kv_cache_rate):    _ , gemm_fp8_flops, attn_fp16_flops = mla_flops(seq_len,seq_len,args, kv_cache_rate)    gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops()    print("GEMM_FP8 Elapsed time(ms): %.3f" % gemm_fp8_time)    attn_fp16_time = attn_fp16_flops / gpu.get_fp16_flops()    print("ATTN_FP16 Elapsed time(ms): %.3f" % attn_fp16_time)    total_time = gemm_fp8_time + attn_fp16_time    print("Total Elapsed time(ms):%.3f" % total_time)    all_reduce_comm_size = seq_len * args.dim * 2 /1024/1024#fp16 take 2Bytes    ar_elapsed_time = all_reduce_comm_size / gpu.get_nvlink_bw()    print("AR Elapsed time(ms):%.3f" % ar_elapsed_time)    tp4_time = total_time/4 + ar_elapsed_time    print("TP4 Elapsed time(ms):%.3f" % tp4_time)    tp8_time = total_time/8 + ar_elapsed_time    print("TP8 Elapsed time(ms):%.3f" % tp8_time)    return total_time, tp4_time,tp8_timedef prefill_mla(args:ModelArgs, gpu_dict, seq_len, kv_cache_rate):    df = pd.DataFrame(columns=['GPU','TP1','TP4','TP8'])    for key in gpu_dict.keys():        print('------------ %s --------------' % key)        tp1,tp4,tp8 = prefill_mla_elapse_time(args,gpu_dict[key], seq_len, kv_cache_rate)        df.loc[len(df)]=[key,tp1,tp4,tp8]    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
```

H800需要20个通信SM. 这里考虑到H20浮点算力比较弱, H20估计需要10个通信SM, 同时计算了TP=4和TP=8两种情况, 并在后面针对两种并行策略的吞吐进行分析, 当TP=8时, DP组将会变成4个.使用TP并行时, Allreduce通信量为`seq_len` x `dim` x `2Bytes`(BF16)

```
seq_len = 4383kv_cache_rate = 0.563prefill_mla(args,gpu,seq_len,kv_cache_rate)------------ H800 --------------GEMM_FP8 Elapsed time(ms): 0.517ATTN_FP16 Elapsed time(ms): 4.560Total Elapsed time(ms):5.077AR Elapsed time(ms):0.352TP4 Elapsed time(ms):1.622TP8 Elapsed time(ms):0.987------------ H20 --------------GEMM_FP8 Elapsed time(ms): 3.364ATTN_FP16 Elapsed time(ms): 29.671Total Elapsed time(ms):33.035AR Elapsed time(ms):0.157TP4 Elapsed time(ms):8.415TP8 Elapsed time(ms):4.286
```

统计MLA中GPU的计算时间(单位ms)为

GPU

TP1

TP4

TP8

H800

5.077

1.622

0.987

H20

33.035

8.415

4.286

#### 2.2 DenseMLP计算耗时

DenseMLP运算量统计如下:

```
def densmlp_flops(args:ModelArgs, seq_len):    return3 * seq_len * args.dim * args.inter_dim *2/1e9    def dense_mlp_elapse_time(args:ModelArgs,gpu:GPU_perf, seq_len):    gemm_fp8_flops = densmlp_flops(args, seq_len)    gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops()    print("Elapsed time(ms): %.3f" % gemm_fp8_time)    return gemm_fp8_timedef prefill_dense_mlp(args:ModelArgs, gpu_dict, seq_len):    df = pd.DataFrame(columns=['GPU','DenseMLP耗时'])    for key in gpu_dict.keys():        print('------------ %s --------------' % key)        t = dense_mlp_elapse_time(args,gpu_dict[key], seq_len)        df.loc[len(df)]=[key,t]    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
```

实际运算的长度为

```
q_len = seq_len *( 1- kv_cache_rate)------------ H800 --------------Elapsed time(ms): 3.156------------ H20 --------------Elapsed time(ms): 19.801
```

DenseMLP累计耗时(单位ms):

GPU

DenseMLP耗时

H800

3.156

H20

19.801

#### 2.3 MoE计算耗时

TP=4时, DP=8, 那么相当于MLA同时产生了8组`seq_len`的token, 平均每卡Shared Expert计算的token数为 `seq_len`* `dp_group` / `num_gpu`

对于Routed Expert, 当topk=8时总共需要处理的Routed Expert的计算量为`seq_len` * `dp_group` * `topk`, 然后平均分摊到32卡上, 每卡的Routed Expert计算量为`seq_len` * `dp_group` * `topk` / `num_gpu`

```
def moe_expert_flops(args:ModelArgs, seq_len):    return3 * seq_len * args.dim * args.moe_inter_dim *2/1e9def moe_expert_elapse_time(args:ModelArgs,gpu:GPU_perf, seq_len, tp, dp):    num_device = tp * dp    num_shared_token = dp * seq_len / num_device    shared_flops = moe_expert_flops(args, num_shared_token)    shared_time = shared_flops / gpu.get_fp8_flops()    print("Shared Expert Elapsed time(ms): %.3f" % shared_time)    num_routed_token = seq_len * dp * args.n_activated_experts / num_device    routed_flops = moe_expert_flops(args, num_routed_token)    routed_time = routed_flops / gpu.get_fp8_flops()    print("Routed Expert Elapsed time(ms): %.3f" % routed_time)    return shared_time, routed_timedef prefill_moe(args:ModelArgs, gpu_dict, seq_len, tp, dp ):    df = pd.DataFrame(columns=['GPU','Shared Expert','Routed Expert'])    for key in gpu_dict.keys():        print('------------ %s --------------' % key)        s, r = moe_expert_elapse_time(args,gpu_dict[key], seq_len,tp,dp)        df.loc[len(df)]=[key,s,r]    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
```

TP=4时, DP = 8, GPU耗时如下q

```
prefill_moe(args,gpu, seq_len, tp=4,dp=8)------------ H800 --------------Shared Expert Elapsed time(ms): 0.088Routed Expert Elapsed time(ms): 0.701------------ H20 --------------Shared Expert Elapsed time(ms): 0.550Routed Expert Elapsed time(ms): 4.400
```

MoE Expert计算耗时(单位ms):

GPU

Shared Expert

Routed Expert

H800

0.088

0.701

H20

0.550

4.400

TP=8时, DP = 4, GPU耗时如下

```
prefill_moe(args,gpu, seq_len, tp=8,dp=4)------------ H800 --------------Shared Expert Elapsed time(ms): 0.044Routed Expert Elapsed time(ms): 0.351------------ H20 --------------Shared Expert Elapsed time(ms): 0.275Routed Expert Elapsed time(ms): 2.200
```

MoE Expert计算耗时(单位ms):

GPU

Shared Expert

Routed Expert

H800

0.044

0.351

H20

0.275

2.200

#### 2.4 AlltoAll通信耗时

DeepSeek-V3设计了MoE Group的概念, 用于平衡NVLINK和IB的带宽. 一个Token通过MoE Gating函数, 一个token最多仅会分发到4个节点上. 按照EP并行专家负载完全均衡的情况下考虑, 在RDMA上的跨机通信为 3 * token数. Dispatch通信量为: TP=4:  每节点有2个DP组,累计需要发送 2 * 3 * `seq_len`  * `dim` . TP=8:  每节点有1个DP组,累计需要发送 3 * `seq_len`  * `dim` .

Combine阶段由于数据为FP16, 通信量翻倍, H800和H20 ScaleOut带宽相同, 按照DeepEP可以打满45GB/s, 但同时兼顾总带宽利用率80%~40GB/s计算, 总带宽为 40GB/s * 8 = 320GB/s, 通信耗时为:

```
def prefill_alltoall_time(args:ModelArgs, gpu, seq_len, dispatch_node, tp):    ##通信量估计    gpu_per_node = 8    dp = gpu_per_node/tp    dispatch_size = (dispatch_node - 1) * dp * seq_len * args.dim /1024/1024    combine_size = 2 * dispatch_size  #fp16      comm_bw = gpu.get_pcie_bw() * gpu_per_node    dispatch_time = dispatch_size / comm_bw    combine_time = combine_size / comm_bw    return dispatch_time, combine_timedef prefill_alltoall(args:ModelArgs, gpu_dict, seq_len, dispatch_node, tp):    df = pd.DataFrame(columns=['GPU','Dispatch','Combine'])    for key in gpu_dict.keys():        print('------------ %s --------------' % key)        dispatch_time, combine_time = prefill_alltoall_time(args, gpu_dict[key],seq_len, dispatch_node, tp)        print("Dispatch Elapsed time(ms): %.3f" % dispatch_time)        print("Combine Elapsed time(ms): %.3f" % combine_time)              df.loc[len(df)]=[key,dispatch_time,combine_time]    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
```

在TP=4时, 单个节点有2个DP组, 计算结果如下(单位ms):

```
prefill_alltoall(args,gpu,seq_len,dispatch_node=4,tp=4)------------ H800 --------------Dispatch Elapsed time(ms): 0.529Combine Elapsed time(ms): 1.057------------ H20 --------------Dispatch Elapsed time(ms): 0.529Combine Elapsed time(ms): 1.057
```

GPU

Dispatch

Combine

H800

0.529

1.057

H20

0.529

1.057

在TP=8时, 单个节点只有1个DP组, 计算结果如下(单位ms):

```
prefill_alltoall(args,gpu,seq_len,dispatch_node=4,tp=8)------------ H800 --------------Dispatch Elapsed time(ms): 0.264Combine Elapsed time(ms): 0.529------------ H20 --------------Dispatch Elapsed time(ms): 0.264Combine Elapsed time(ms): 0.529
```

GPU

Dispatch

Combine

H800

0.264

0.529

H20

0.264

0.529

#### 2.5 总耗时

累计耗时, 非Overlap计算

3x(MLA_tp1 + DenseMLP) + 58x(MLA_tpN + Shared Expert + Routed Expert +Dispatch + Combine)

完全Overlap计算

3x(MLA_tp1 + DenseMLP) + 58x(MLA_tpN + Shared Expert + Routed Expert)

定义计算函数如下所示:

```
def prefill_time(args:ModelArgs, gpu, seq_len, kv_cache_rate, tp , dp):    dispatch_node = 4    gpu_per_node = 8    num_device  =  tp * dp    dense_mla,tp4_mla,tp8_mla = prefill_mla_elapse_time(args, gpu,  seq_len, kv_cache_rate)     tp_mla = tp4_mla if tp == 4else tp8_mla    dense_mlp = dense_mlp_elapse_time(args, gpu, seq_len)    shared, routed = moe_expert_elapse_time(args, gpu, seq_len, tp, dp)    dispatch, combine = prefill_alltoall_time(args, gpu, seq_len, dispatch_node, tp)    return dense_mla, dense_mlp, tp_mla, shared, routed, dispatch, combine    def prefill_time_sum(args:ModelArgs, gpu_dict, seq_len, kv_cache_rate, tp , dp):    df = pd.DataFrame(columns=['MLA','DenseMLP','TP_MLA','Shared Expert','Routed Expert','Dispatch','Combine','GPU'])    df2 = pd.DataFrame(columns=['Sum(Overlap)','Sum','GPU'])    n_sparse_layers = args.n_layers - args.n_dense_layers    df.loc[len(df)]= [ args.n_dense_layers, args.n_dense_layers,  #MLA+ DenseMLP                       n_sparse_layers, n_sparse_layers, n_sparse_layers, #SparseLayer MLA + MoE                       n_sparse_layers, n_sparse_layers, 'Layers'] #Dispatch & Combine Layers    for key in gpu_dict.keys():        t  = list(prefill_time(args, gpu_dict[key], seq_len, kv_cache_rate , tp , dp))        t.append(key)        df.loc[len(df)]= t        sum_overlap = args.n_dense_layers * (t[0] + t[1]) + n_sparse_layers * ( t[2] + t[3] + t[4])         sum_non_overlap = sum_overlap + n_sparse_layers * ( t[5] + t[6]) #alltoall        df2.loc[len(df2)]= [ sum_overlap, sum_non_overlap, key]    df = df.set_index('GPU').T    df['Layers'] = df['Layers'].astype(int).astype(str)    print(df.to_markdown(floatfmt=".3f"))      print('-----------SUM-------------')    df2 = df2.set_index('GPU').T    print(df2.to_markdown(floatfmt=".3f"))          return df,df2
```

TP=4时, DP=8, 耗时分析如下(单位ms):

```
tp4_detail,tp4_sum = prefill_time_sum(args, gpu, seq_len, kv_cache_rate,tp=4 , dp=8)
```

Layers

H800

H20

MLA

3

5.077

33.035

DenseMLP

3

3.043

19.801

TP_MLA

58

1.622

8.415

Shared Expert

58

0.085

0.550

Routed Expert

58

0.676

4.400

Dispatch

58

0.529

0.529

Combine

58

1.057

1.057

累计时间分析(单位ms):

H800

H20

Sum(Overlap)

162.539

933.703

Sum

254.540

1025.704

TP=8时, DP=4, 耗时分析如下(单位ms):

```
tp8_detail,tp8_sum  = prefill_time_sum(args, gpu, seq_len, kv_cache_rate,tp=8 , dp=4)
```

Layers

H800

H20

MLA

3

5.077

33.035

DenseMLP

3

3.043

19.801

TP_MLA

58

0.987

4.286

Shared Expert

58

0.042

0.275

Routed Expert

58

0.338

2.200

Dispatch

58

0.264

0.264

Combine

58

0.529

0.529

累计时间分析(单位ms):

H800

H20

Sum(Overlap)

103.672

550.648

Sum

149.672

596.648

由于累计为DP组seq_len的推理, 平均1s单机能够处理的Token为 `DP * seq_len * (1000ms / 计算时间)/节点数`, 计算如下

官方的TP=4的部署方式:

```
dp = 8num_node = 4print(tp4_sum.apply(lambda x: dp * seq_len * (1000/ x)/num_node).to_markdown(floatfmt=".1f"))
```

H800

H20

Sum(Overlap)

53931.6

9388.4

Sum

34438.6

8546.3

而TP=8的部署方式的吞吐:

H800

H20

Sum(Overlap)

42277.8

7959.7

Sum

29284.1

7346.0

可以看到DeepSeek官方选择的TP=4的配置是吞吐更优的选择, 另外官方的数据为单机73.7K tokens/s(含缓存命中), 折算出来非命中需要计算的平均token/s为 32207, 考虑到每天的峰谷效应, 该值符合预期.

另一方面考虑到H20对于TTFT首Token延迟的影响, TP=4已经超过1s, 可以采用TP=8的策略降低首Token延迟.

#### 2.6 Overlap分析

在官方部署方案中, 可以按照如下方式分两个Micro-batch进行Overlap
![图片](assets/bbbf3d74a567.png)
基于此对官方的prefill.json trace标注如下, 实际上的Trace还是有一些没有Overlap的:
![图片](assets/32826cb04d24.png)

实际计算TP=4时, Prefill的计算耗时如下, 可以看到通信是可以被计算Overlap的.

Layers

H800

H20

TP_MLA

58

1.622

8.415

Shared Expert

58

0.085

0.550

Combine

58

1.057

1.057

-

-

-

-

Routed Expert

58

0.676

4.400

Dispatch

58

0.529

0.529

特别的来看, H20中还可以降低RDMA ScaleOut的带宽, 做了一些初步的估计

```
h20_32 = GPU_perf( sm = 78 ,comm_sm = 10,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 96,mem_bw = 4000,                 nvlink_bw = 450,pcie_bw = 50,                 discount_rate = 0.85)h20_16 = GPU_perf( sm = 78 ,comm_sm = 10,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 96,mem_bw = 4000,                 nvlink_bw = 450,pcie_bw = 25,                 discount_rate = 0.85)h20_8 = GPU_perf( sm = 78 ,comm_sm = 10,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 96,mem_bw = 4000,                 nvlink_bw = 450,pcie_bw = 12.5,                 discount_rate = 0.85)gpu_h20 = dict({ 'H20-3.2T': h20_32, 'H20-1.6T': h20_16 , 'H20-800G': h20_8})tp4_detail,tp4_sum = prefill_time_sum(args, gpu_h20, seq_len, kv_cache_rate,tp=4 , dp=8)
```

Layers

H20-3.2T

H20-1.6T

H20-800G

TP_MLA

58

8.415

8.415

8.415

Shared Expert

58

0.550

0.550

0.550

Dispatch

58

0.529

1.057

2.115

-

-

-

-

-

Routed Expert

58

4.400

4.400

4.400

Combine

58

1.057

2.115

4.230

可以看到在Prefill阶段, 如果平均`seq_len`足够长时, 800G也能很好的Overlap

H20-3.2T

H20-1.6T

H20-800G

Sum(Overlap)

933.703

933.703

933.703

Sum

1025.704

1117.705

1301.706

注:如果大量的Prefill长度在1000~2000左右依旧需要1.6Tbps~3.2Tbps RDMA.

#### 2.7 KVCache计算

对于Token/s我们还可以折算出传输KVCache的总量:

```
dp = 8num_node = 4tp4_detail,tp4_sum = prefill_time_sum(args, gpu, seq_len, kv_cache_rate,tp=4 , dp=8)kvcache_fp8 = tp4_sum.apply(lambda x: dp * seq_len * (1000/ x)/num_node * (args.kv_lora_rank + args.qk_rope_head_dim)/1024/1024)kvcache_fp16 = kvcache_fp8 *2kvcache=kvcache_fp8.join(kvcache_fp16, lsuffix='(FP8)',rsuffix='(FP16)')print(kvcache.to_markdown(floatfmt=".1f"))
```

H800(FP8)

H20(FP8)

H800(FP16)

H20(FP16)

Sum(Overlap)

29.6

5.2

59.3

10.3

Sum

18.9

4.7

37.8

9.4

注: 这里没有考虑KVCache命中率, 考虑后应该带宽折算为读写两个方向.

对于H800, 如果KV-Cache采用FP16存储,则已经超过了连接CPU的那张400Gbps(50GB/s)网卡带宽, 需要采用GPU直连的RDMA Scaleout网络进行传输. 正好前几天和夏Core谈到这个问题, 实际上存储接入GPU互联的ScaleOut网络, 平均分摊到每张网卡上,适当的编排通信算子, 对于EP并行影响是几乎可以忽略的...

### 3. Decoding阶段

Decoding集群采用18台部署, 路由专家EP144, MLA和共享专家DP144. 32个冗余路由专家, 每张卡2个路由专家和1个共享专家.而论文的做法是40台部署EP320, 每张卡1个专家, TP=4, DP=80, Decode阶段不需要独立的通信SM, 因此对GPU性能数据建模如下:

```
h800 = GPU_perf( sm = 132 ,comm_sm = 0,                  fp16_flops = 791.6, fp8_flops = 1583.2,                  mem = 80,mem_bw = 4000,                 nvlink_bw = 200,pcie_bw = 50,                 discount_rate = 0.85)h20 = GPU_perf( sm = 78 ,comm_sm = 0,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 96,mem_bw = 4000,                 nvlink_bw = 450,pcie_bw = 50,                 discount_rate = 0.85)h20_3e = GPU_perf( sm = 78 ,comm_sm = 0,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 141,mem_bw = 4800,                 nvlink_bw = 450,pcie_bw = 50,                 discount_rate = 0.85)gpu_decode = dict({'H800': h800, 'H20': h20,'H20_3e': h20_3e})gpu_decode2 = dict({'H800': h800, 'H20': h20}) 
```

#### 3.1 EP策略分析

我们需要根据EP并行策略来分析, 假设集群的并行策略为为Decoding集群的总的卡数, 为冗余专家数.平均每卡的路由专家数为满足

由于线上环境中还是有大量的Expert负载不均衡的情况, 需要保证有足够多的冗余专家数量用于EPLB调度, 例如我们定义冗余专家数量不能少于16, 可以构常见的几种EP并行策略如下:

冗余专家

每卡专家

EP34

16

8

EP72

32

4

EP144

32

2

EP320

64

1

对于不同并行策略需要通信和处理的Token数按如下方式计算

```
class MoE_EP():    def __init__(self,args:ModelArgs,ep_num, redundant_exp):        self.ep_num = ep_num        self.redundant_exp = redundant_exp        self.dispatch_num = args.n_activated_experts        self.n_routed_experts = args.n_routed_experts        self.expert_num = (args.n_routed_experts + redundant_exp) / self.ep_num    def expert_per_gpu(self):        return self.expert_num            def total_tokens(self,bs):        return bs * self.ep_num    def comm_tokens(self, bs):        #平均每个token有self.expert_num / self.n_routed_experts概率本地处理         return bs * self.dispatch_num *(1- self.expert_num / self.n_routed_experts)            def compute_tokens(self, bs):        #总token数为bs * dispatch_num * ep_num, 平摊到每张卡/ep_num        return bs * self.dispatch_num  ep_dict = { 'EP34': MoE_EP(args, 34,16),            'EP72' :MoE_EP(args, 72,32),            'EP144' :MoE_EP(args, 144,32),            'EP320' :MoE_EP(args, 320,64)}
```

#### 3.2 Memory利用率分析

我们先以Memory容量进行分析, 得出各种并行场景下的最大BatchSize. 对于模型的参数参考1.4章节, 因为Decoding阶段要采用matabsorb的MLA, 因此除去MLA和Expert的参数为`671.16B - MLA(187.17M)* 61 - Expert(44.05M)* (256-Routed+1-Shared) * 58`= 3.13B参数. 折算成实际显存消耗为`3.13 *(1000/1024)^3` = 2.91GB

BatchSize计算如下,考虑Decoding的长度为1210, 根据前一节EP策略分析估计专家数

```
def _decoding_batchsize(args:ModelArgs, gpu:GPU_perf, seq_len,decode_len,tp, expert_num, absorb=True, kvcache_fp16=False):    mem_util_rate = 0.9#torch/activation等其它开销的折扣    mla = 254.21if absorb else187.17#MLA的参数(单位M)    expert_mem = 44.05#expert的参数(单位M)    others_parameter = 2.91#其它参数2.91GB    kv_cache = (seq_len+decode_len) * (args.kv_lora_rank + args.qk_rope_head_dim) *args.n_layers *tp    if kvcache_fp16 :        kv_cache *=2    mem = gpu.mem * mem_util_rate - others_parameter - mla * args.n_layers/tp/1024    mem -= expert_mem *(args.n_layers - args.n_dense_layers) * expert_num /1024    return mem * 1024 * 1024 * 1024 / kv_cachedef decode_batchsize(args:ModelArgs, gpu_dict, seq_len,decode_len, tp):    df = pd.DataFrame(columns=['GPU','EP320','EP144','EP72','EP34'])    for fp16_kvcache in range(0,2):        for key in gpu_dict.keys():            for absorb in range(0,2):                item = key                if bool(fp16_kvcache):                    item +='_FP16'                else:                    item +='_FP8'                if bool(absorb):                    item +='_Absorb'                                    value = [item]                for exp_num in [2,3,5,9]:                    bs = _decoding_batchsize(args, gpu_dict[key], seq_len,decode_len, tp,exp_num, bool(absorb),bool(fp16_kvcache))                    value.append(bs)                df.loc[len(df)]= value    print(df.set_index('GPU').to_markdown(floatfmt=".0f"))      return dfdecode_len = 1210df = decode_batchsize(args,gpu_decode, seq_len,decode_len, tp=1)
```

不同的并行策略结果可以承载的BatchSize如下所示:

GPU

EP320

EP144

EP72

EP34

H800_FP8

289

276

248

194

H800_FP8_Absorb

267

254

227

172

H20_FP8

368

354

327

273

H20_FP8_Absorb

346

333

305

251

H20_3e_FP8

589

576

548

494

H20_3e_FP8_Absorb

567

554

527

472

H800_FP16

145

138

124

97

H800_FP16_Absorb

134

127

113

86

H20_FP16

184

177

164

136

H20_FP16_Absorb

173

166

153

125

H20_3e_FP16

295

288

274

247

H20_3e_FP16_Absorb

284

277

263

236

TP=4时, 能够承受的BatchSize

GPU

EP320

EP144

EP72

EP34

H800_FP8

84

80

74

60

H800_FP8_Absorb

82

79

72

59

H20_FP8

103

100

93

80

H20_FP8_Absorb

102

99

92

78

H20_3e_FP8

159

155

149

135

H20_3e_FP8_Absorb

157

154

147

134

H800_FP16

42

40

37

30

H800_FP16_Absorb

41

39

36

29

H20_FP16

52

50

47

40

H20_FP16_Absorb

51

49

46

39

H20_3e_FP16

79

78

74

67

H20_3e_FP16_Absorb

79

77

74

67

TP=8时, 能够承受的BatchSize

GPU

EP320

EP144

EP72

EP34

H800_FP8

43

41

38

31

H800_FP8_Absorb

42

41

37

31

H20_FP8

53

51

48

41

H20_FP8_Absorb

52

51

47

40

H20_3e_FP8

80

79

75

68

H20_3e_FP8_Absorb

80

78

75

68

H800_FP16

21

21

19

15

H800_FP16_Absorb

21

20

19

15

H20_FP16

26

25

24

20

H20_FP16_Absorb

26

25

24

20

H20_3e_FP16

40

39

38

34

H20_3e_FP16_Absorb

40

39

37

34

结论: 从内存容量的角度看, 更大的显存更容易放下足够的BatchSize, 由于Decoding的算力影响, 需要考虑MLA矩阵吸收后的带宽占用, 但整个EP并行策略需要保证每个卡的路由专家数不超过8. 另外,对于H800如果在MLA矩阵吸收模式下运行, 还需要保证KVCache按照FP8存储才能满足batchsize=128的需求.

#### 3.3 MLA耗时

Decoding阶段我们采用带矩阵吸收的MLA计算方式, 由于计算延迟较低, 我们还需要考虑加载KVCache的时间, 计算方式如下

```
bs_list =[32, 64, 128, 256]def decode_mla_elapse_time(args:ModelArgs, gpu:GPU_perf, seq_len, bs, absorb=True):    mla_flops_func = mla_matabsob_flops_v2 if absorb else mla_flops    #Decoding时计算为qlen=1, kv_cache_rate = 1    _ , gemm_fp8_flops, attn_fp16_flops = mla_flops_func(1,seq_len,args, 1)        gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops() * bs    print("GEMM_FP8 Elapsed time(ms): %.3f" % gemm_fp8_time)    attn_fp16_time = attn_fp16_flops / gpu.get_fp16_flops() *bs    print("ATTN_FP16 Elapsed time(ms): %.3f" % attn_fp16_time)     total_time = gemm_fp8_time + attn_fp16_time    print("Total Elapsed time(ms):%.3f" % total_time)    all_reduce_comm_size = seq_len * args.dim * 2 /1024/1024#fp16 take 2Bytes    ar_elapsed_time = all_reduce_comm_size / gpu.get_nvlink_bw()    print("AR Elapsed time(ms):%.3f" % ar_elapsed_time)    tp4_time = total_time/4 + ar_elapsed_time    print("TP4 Elapsed time(ms):%.3f" % tp4_time)    tp8_time = total_time/8 + ar_elapsed_time    print("TP8 Elapsed time(ms):%.3f" % tp8_time)    return total_time, tp4_time, tp8_timedef decode_kvcache_load_time(args:ModelArgs, gpu:GPU_perf, seq_len, bs):    kv_cache = seq_len * (args.kv_lora_rank + args.qk_rope_head_dim)  * bs     load_kv_time = kv_cache /1024/1024/1024 / gpu.get_mem_bw() *1000    return load_kv_time     def decode_mla(args:ModelArgs, gpu_dict, seq_len,absorb=True):    df = pd.DataFrame(columns=['GPU','BatchSize','TP1','TP4','TP8','LoadKV_FP8','LoadKV_FP16'])    for key in gpu_dict.keys():        for bs in bs_list:              tp1, tp4,tp8 = decode_mla_elapse_time(args,gpu_dict[key], seq_len, bs,absorb)             kv = decode_kvcache_load_time(args,gpu_dict[key], seq_len, bs)             df.loc[len(df)]= [key, bs,tp1,tp4,tp8,kv, kv*2]             df['BatchSize'] = df['BatchSize'].astype(int).astype(str)    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))      return dfdecode_mla(args,gpu_decode,seq_len)
```

计算结果(单位ms):

GPU

BatchSize

TP1

TP4

TP8

LoadKV_FP8

LoadKV_FP16

H800

32

0.076

0.371

0.362

0.022

0.044

H800

64

0.152

0.390

0.371

0.044

0.089

H800

128

0.304

0.428

0.390

0.089

0.177

H800

256

0.607

0.504

0.428

0.177

0.354

H20

32

0.508

0.284

0.220

0.022

0.044

H20

64

1.015

0.411

0.284

0.044

0.089

H20

128

2.031

0.664

0.411

0.089

0.177

H20

256

4.062

1.172

0.664

0.177

0.354

如果不采用矩阵吸收模式, 计算结果如下(单位ms)

GPU

BatchSize

TP1

TP4

TP8

LoadKV_FP8

LoadKV_FP16

H800

32

3.528

1.234

0.793

0.022

0.044

H800

64

7.055

2.116

1.234

0.044

0.089

H800

128

14.111

3.880

2.116

0.089

0.177

H800

256

28.222

7.408

3.880

0.177

0.354

H20

32

23.586

6.053

3.105

0.022

0.044

H20

64

47.172

11.950

6.053

0.044

0.089

H20

128

94.343

23.742

11.950

0.089

0.177

H20

256

188.686

47.328

23.742

0.177

0.354

因此无论是H800还是H20在Decoding阶段都需要采用矩阵吸收的MLA计算模式, 对于H800 MLA计算时, BatchSize=128时, TP=4的并行策略和TP1耗时相近, BatchSize=64的时候还更快, 因此在EP144的部署中没有使用TP并行,而在EP320的部署中, 如果BatchSize=256时, 使用TP=4并行有收益.

而针对H20的最佳实践是, MLA的计算必须要使用TP并行. 但是TP并行还会导致额外的KVCache开销, 因此需要核算显存利用率, 对于H20, 虽然TP=8可以显著降低运算延迟, 但是也会使得最大BatchSize受到约束, 因此H20最优的策略为TP=4

#### 3.4 DenseMLP耗时

计算方法如下所示, 主要考虑不同的BatchSize下的计算延迟:

```
def decode_dense_mlp(args:ModelArgs, gpu_dict):    df = pd.DataFrame(columns=['GPU','BatchSize','DenseMLP'])    for key in gpu_dict.keys():        for bs in bs_list:             t = dense_mlp_elapse_time(args,gpu_dict[key], bs)            df.loc[len(df)]=[key,bs,t]    df['BatchSize'] = df['BatchSize'].astype(int).astype(str)            print(df.set_index('GPU').to_markdown(floatfmt=".3f"))    return df    decode_dense_mlp(args,gpu_decode)
```

计算耗时如下:

GPU

BatchSize

DenseMLP

H800

32

0.019

H800

64

0.038

H800

128

0.075

H800

256

0.151

H20

32

0.126

H20

64

0.252

H20

128

0.504

H20

256

1.008

#### 3.5 MoE耗时计算

根据不同的GPU类型计算耗时, 其实这个和EP策略无关, 因为任何一个token都要dispatch 8份发到其它节点, 因此简化计算流程, 同时还需要考虑到GroupGEMM和相对较小的batchsize无法打满的影响, 这里按照DeepGEMM的性能, 定义了一个和batchsize相关的性能折算估计系数, 这里后面还是需要实际在机器上跑一下,只是拍脑袋预估.

注: ver 0.1的时候只采用了一个固定的折扣, 因为是要评估网络延迟对Overlap的影响, 这里根据建议做了一些折扣, 考虑到基础的GPU性能已经对峰值FLOPS做了0.85的折扣, 实际上的折扣率为0.85 * group_gemm_discount_rate.

```
def _moe_expert_time(args:ModelArgs,gpu:GPU_perf,bs):    if bs <= 32 :        group_gemm_discount_rate = 0.2    elif bs <= 64 :        group_gemm_discount_rate = 0.3    elif bs <= 128:        group_gemm_discount_rate = 0.5    else:        group_gemm_discount_rate = 0.7    shared_flops = moe_expert_flops(args, bs)    shared_time = shared_flops / gpu.get_fp8_flops() / group_gemm_discount_rate    num_routed_token = bs * args.n_activated_experts    routed_flops = moe_expert_flops(args, num_routed_token)    routed_time = routed_flops / gpu.get_fp8_flops() / group_gemm_discount_rate    return shared_time, routed_timedef moe_expert_time(args:ModelArgs,gpu_dict):    df = pd.DataFrame(columns=['GPU','BatchSize','SharedExpert','RoutedExpert'])    for gpu_key in gpu_dict.keys():        for bs in bs_list:             s, r = _moe_expert_time(args,gpu_dict[gpu_key], bs)            df.loc[len(df)]=[gpu_key,str(bs),s,r]    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))            return dfmoe_expert_time(args,gpu_decode)
```

各种组合的结果如下所示(单位ms):

GPU

BatchSize

SharedExpert

RoutedExpert

H800

32

0.010

0.084

H800

64

0.014

0.112

H800

128

0.017

0.134

H800

256

0.024

0.191

H20

32

0.070

0.560

H20

64

0.093

0.747

H20

128

0.112

0.896

H20

256

0.160

1.280

#### 3.5 AlltoAll通信耗时

AlltoAll由于采用IBGDA的方式, 直接通过RDMA传输, 因此计算时仅需要考虑GPU的PCIe带宽, 计算函数如下所示:

```
def _moe_a2a(args:ModelArgs,gpu:GPU_perf,bs):    dispatch_size = bs * args.dim * args.n_activated_experts /1024/1024    combine_size = dispatch_size * 2#FP16    dispatch_t = dispatch_size / gpu.get_pcie_bw()    combine_t = combine_size / gpu.get_pcie_bw()    return dispatch_t, combine_tdef decode_a2a(args:ModelArgs, gpu_dict):    df = pd.DataFrame(columns=['GPU','BatchSize','Dispatch','Combine'])    for key in gpu_dict.keys():        for bs in [64, 128, 256]:             dispatch_time, combine_time = _moe_a2a(args, gpu_dict[key],bs)            df.loc[len(df)]=[key,str(bs),dispatch_time,combine_time]    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))decode_a2a(args,gpu_decode)
```

Dispatch和Combine的计算结果为(单位ms):

GPU

BatchSize

Dispatch

Combine

H800

32

0.041

0.082

H800

64

0.082

0.165

H800

128

0.165

0.329

H800

256

0.329

0.659

H20

32

0.041

0.082

H20

64

0.082

0.165

H20

128

0.165

0.329

H20

256

0.329

0.659

#### 3.6 总耗时

统计总耗时表的函数如下, 由于H20和H20-3e在加载KV-Cache时有很小的性能差距, 接下来的计算仅计算H800和H20

```
from functools import reducedef _decoding_time(args:ModelArgs, gpu:GPU_perf,seq_len):    mla = decode_mla(args,gpu,seq_len)    dense_mlp = decode_dense_mlp(args,gpu)     moe = moe_expert_time(args,gpu)    a2a = decode_a2a(args,gpu)    dfs = [ mla, dense_mlp, moe, a2a]    df = reduce(lambda left, right: pd.merge(left,right, on=['GPU','BatchSize'], how='left'), dfs)    print(df.set_index('GPU').T.to_markdown(floatfmt=".3f"))    return df    dfs = _decoding_time(args,gpu_decode2,seq_len)
```

统计结果如下所示(单位ms):

H800

H800

H800

H800

H20

H20

H20

H20

BatchSize

32.000

64.000

128.000

256.000

32.000

64.000

128.000

256.000

TP1

0.076

0.152

0.304

0.607

0.508

1.015

2.031

4.062

TP4

0.371

0.390

0.428

0.504

0.284

0.411

0.664

1.172

TP8

0.362

0.371

0.390

0.428

0.220

0.284

0.411

0.664

LoadKV_FP8

0.022

0.044

0.089

0.177

0.022

0.044

0.089

0.177

LoadKV_FP16

0.044

0.089

0.177

0.354

0.044

0.089

0.177

0.354

DenseMLP

0.019

0.038

0.075

0.151

0.126

0.252

0.504

1.008

SharedExpert

0.010

0.014

0.017

0.024

0.070

0.093

0.112

0.160

RoutedExpert

0.084

0.112

0.134

0.191

0.560

0.747

0.896

1.280

Dispatch

0.041

0.082

0.165

0.329

0.041

0.082

0.165

0.329

Combine

0.082

0.165

0.329

0.659

0.082

0.165

0.329

0.659

我们针对模型结构和最优TP策略进行修正,并计算TPOT,如下所示:

```
def decoding_time(args:ModelArgs, gpu_dict,seq_len):    df = _decoding_time(args,gpu_dict,seq_len)    def mla_tp(r):        if r['TP1'] > r['TP4']:            if r['GPU'].find('H20_3e')!=-1:                return'TP8'            else:                return'TP4'        else:            return'TP1'                def mla_tp2(r):        tp = r['MLA_TP']        return r[tp]    #使用最佳的TP策略估计    df['MLA_TP'] = df.apply(lambda row:  mla_tp(row),axis=1)    df['SparseMLA'] = df.apply(lambda row:  mla_tp2(row),axis=1)        # 修正TP执行时间, 按照加载FP8的KV计算    df['DenseMLA'] = df['TP1'] + df['LoadKV_FP8']    df['SparseMLA'] = df['SparseMLA'] + df['LoadKV_FP8']    df['COMP_SUM'] = df['SparseMLA'] +df['SharedExpert']+ df['RoutedExpert']    df['COMM_SUM'] = df['Dispatch'] + df['Combine']    df['TPOT(Overlap)'] = (df['DenseMLA'] + df['DenseMLP']) * args.n_dense_layers     df['TPOT(Overlap)'] += (df['SparseMLA'] + df['SharedExpert'] + df['RoutedExpert']) * (args.n_layers - args.n_dense_layers)    df['TPOT'] = df['TPOT(Overlap)'] + (df['Dispatch'] + df['Combine']) * (args.n_layers - args.n_dense_layers)    df['GPU'] = df['GPU']+ "_" + df['MLA_TP']     df = df[['GPU','BatchSize','DenseMLA','DenseMLP','SparseMLA','Combine','SharedExpert','RoutedExpert','Dispatch','COMP_SUM','COMM_SUM', 'TPOT(Overlap)','TPOT']]    df['TPS_O'] = 1000 / df['TPOT(Overlap)']    df['TPS'] = 1000 / df['TPOT']    df['Total_O'] =  df['TPS_O'] * df['BatchSize'].astype(int)    df['Total'] =  df['TPS'] * df['BatchSize'].astype(int)    print(df.set_index('GPU').T.to_markdown(floatfmt=".3f"))    return df        dfs= decoding_time(args,gpu_decode,seq_len)
```

统计结果如下所示(单位ms)

H800_TP1

H800_TP1

H800_TP1

H800_TP4

H20_TP4

H20_TP4

H20_TP4

H20_TP4

H20_3e_TP8

H20_3e_TP8

H20_3e_TP8

H20_3e_TP8

BatchSize

32.000

64.000

128.000

256.000

32.000

64.000

128.000

256.000

32.000

64.000

128.000

256.000

DenseMLA

0.098

0.196

0.392

0.785

0.530

1.060

2.119

4.239

0.526

1.052

2.105

4.209

DenseMLP

0.019

0.038

0.075

0.151

0.126

0.252

0.504

1.008

0.126

0.252

0.504

1.008

SparseMLA

0.098

0.196

0.392

0.681

0.306

0.455

0.753

1.349

0.239

0.320

0.484

0.812

Combine

0.082

0.165

0.329

0.659

0.082

0.165

0.329

0.659

0.082

0.165

0.329

0.659

SharedExpert

0.010

0.014

0.017

0.024

0.070

0.093

0.112

0.160

0.070

0.093

0.112

0.160

RoutedExpert

0.084

0.112

0.134

0.191

0.560

0.747

0.896

1.280

0.560

0.747

0.896

1.280

Dispatch

0.041

0.082

0.165

0.329

0.041

0.082

0.165

0.329

0.041

0.082

0.165

0.329

COMP_SUM

0.192

0.322

0.543

0.897

0.936

1.295

1.761

2.789

0.869

1.161

1.493

2.252

COMM_SUM

0.124

0.247

0.494

0.988

0.124

0.247

0.494

0.988

0.124

0.247

0.494

0.988

TPOT(Overlap)

11.505

19.366

32.901

54.822

56.248

79.043

110.015

177.527

52.342

71.232

94.391

146.281

TPOT

18.670

33.695

61.560

112.140

63.412

93.372

138.673

234.845

59.507

85.561

123.050

203.598

TPS_O

86.918

51.637

30.395

18.241

17.779

12.651

9.090

5.633

19.105

14.039

10.594

6.836

TPS

53.562

29.678

16.244

8.917

15.770

10.710

7.211

4.258

16.805

11.688

8.127

4.912

Total_O

2781.372

3304.787

3890.496

4669.646

568.913

809.685

1163.482

1442.035

611.366

898.479

1356.054

1750.059

Total

1713.996

1899.377

2079.289

2282.865

504.634

685.427

923.032

1090.083

537.756

748.005

1040.225

1257.377

只有H800满足TPS> 20的条件, 稍微放宽一点, 以TPS>10条件进行过滤:

```
print(dfs[dfs['TPS_O']>10].set_index('GPU').T.to_markdown(floatfmt=".3f"))
```

H800_TP1

H800_TP1

H800_TP1

H800_TP4

H20_TP4

H20_TP4

H20_3e_TP8

H20_3e_TP8

H20_3e_TP8

BatchSize

32.000

64.000

128.000

256.000

32.000

64.000

32.000

64.000

128.000

DenseMLA

0.098

0.196

0.392

0.785

0.530

1.060

0.526

1.052

2.105

DenseMLP

0.019

0.038

0.075

0.151

0.126

0.252

0.126

0.252

0.504

SparseMLA

0.098

0.196

0.392

0.681

0.306

0.455

0.239

0.320

0.484

Combine

0.082

0.165

0.329

0.659

0.082

0.165

0.082

0.165

0.329

SharedExpert

0.010

0.014

0.017

0.024

0.070

0.093

0.070

0.093

0.112

RoutedExpert

0.084

0.112

0.134

0.191

0.560

0.747

0.560

0.747

0.896

Dispatch

0.041

0.082

0.165

0.329

0.041

0.082

0.041

0.082

0.165

COMP_SUM

0.192

0.322

0.543

0.897

0.936

1.295

0.869

1.161

1.493

COMM_SUM

0.124

0.247

0.494

0.988

0.124

0.247

0.124

0.247

0.494

TPOT(Overlap)

11.505

19.366

32.901

54.822

56.248

79.043

52.342

71.232

94.391

TPOT

18.670

33.695

61.560

112.140

63.412

93.372

59.507

85.561

123.050

TPS_O

86.918

51.637

30.395

18.241

17.779

12.651

19.105

14.039

10.594

TPS

53.562

29.678

16.244

8.917

15.770

10.710

16.805

11.688

8.127

Total_O

2781.372

3304.787

3890.496

4669.646

568.913

809.685

611.366

898.479

1356.054

Total

1713.996

1899.377

2079.289

2282.865

504.634

685.427

537.756

748.005

1040.225

满足用户TPS>10,则H800需要BatchSize<=128, 此时H800峰值每卡每秒可以产生3130个token, 考虑实际的峰谷效应和一些专家负载不均衡带来的不可Overlap的延迟, 线上环境平均每卡1850个token/s.这里也可以看到, 在EP144时, batchsize=128无需选择TP=4执行MLA, 而在官方DeepSeek-V3论文中,BatchSize=256时则使用TP=4时是最佳的, 但是实际上这个虽然可能是一个更高吞吐的方案, 但是TPS不满足20, 因此官方还是选择了EP144.

H20需要维持BatchSize<=32才能满足15TPS以上的需求. 此时H20性能为533个Token. H20_3e由于更大的显存,可以采用TP=8并行在BatchSize<=64时维持TPS> 15, 此时的吞吐为586个Token.

#### 3.7 Overlap分析

DeepSeek的官方trace中并没有decode的相关的内容, 只是有一个Overlap的图.

![图片](assets/b45560dfaaac.png)

从前一节的汇总表可以看到, Combine是显著小于Attention的, 因此官方对于Attention进行了拆分来Overlap.我们按照以下评估TimeBudget

```
dfo=dfs[dfs['TPS_O']>10]dfo['TimeBudget'] =  dfo['SparseMLA'] + dfo['SharedExpert'] - (dfo['Dispatch']+dfo['Combine'])print(dfo[['GPU','BatchSize','TimeBudget']].set_index('GPU').to_markdown(floatfmt=".3f"))
```

GPU

BatchSize

TimeBudget

H800_TP1

32

-0.015

H800_TP1

64

-0.037

H800_TP1

128

-0.085

H800_TP4

256

-0.283

H20_TP4

32

0.252

H20_TP4

64

0.301

H20_3e_TP8

32

0.185

H20_3e_TP8

64

0.167

H20_3e_TP8

128

0.102

注:其实对于H800的通信是有些轻微的无法Overlap的, 当然这里还没有计算MoE Gating这些的开销, 这也是为什么需要使用IBGDA的原因. EP320的方式部署BatchSize到256时,无法Overlap的量更大,

对于H20我们注意到还有很大的时间预算, 那么是否可以通过使用1.6T/800G的实例呢?计算如下

```
h20_32 = GPU_perf( sm = 78 ,comm_sm = 0,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 96,mem_bw = 4000,                 nvlink_bw = 450,pcie_bw = 50,                 discount_rate = 0.85)h20_16 = GPU_perf( sm = 78 ,comm_sm = 0,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 96,mem_bw = 4000,                 nvlink_bw = 450,pcie_bw = 25,                 discount_rate = 0.85)h20_8 = GPU_perf( sm = 78 ,comm_sm = 0,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 96,mem_bw = 4000,                 nvlink_bw = 450,pcie_bw = 12.5,                 discount_rate = 0.85)h20_3e_32 = GPU_perf( sm = 78 ,comm_sm = 0,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 141,mem_bw = 4800,                 nvlink_bw = 450,pcie_bw = 50,                 discount_rate = 0.85)h20_3e_16 = GPU_perf( sm = 78 ,comm_sm = 0,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 141,mem_bw = 4800,                 nvlink_bw = 450,pcie_bw = 25,                 discount_rate = 0.85)h20_3e_8 = GPU_perf( sm = 78 ,comm_sm = 0,                  fp16_flops = 118.4, fp8_flops = 236.8,                  mem = 141,mem_bw = 4800,                 nvlink_bw = 450,pcie_bw = 12.5,                 discount_rate = 0.85)gpu_decode_h20 = dict({'H20-3.2T': h20_32,'H20-1.6T': h20_16,'H20-800G': h20_8,                       'H20_3e-3.2T': h20_3e_32,'H20_3e-1.6T': h20_3e_16,'H20_3e-800G': h20_3e_8,})dfs= decoding_time(args,gpu_decode_h20,seq_len)dfo=dfs[dfs['TPS_O']>10]dfo['TimeBudget'] =  dfo['SparseMLA'] + dfo['SharedExpert'] - (dfo['Dispatch']+dfo['Combine'])dfo['E2E_impact'] = dfo['TimeBudget'] * (args.n_layers - args.n_dense_layers)print(dfo[['GPU','BatchSize','TimeBudget','E2E_impact']].set_index('GPU').to_markdown(floatfmt=".3f"))
```

GPU

BatchSize

TimeBudget

E2E_impact

H20-3.2T_TP4

32

0.252

14.628

H20-3.2T_TP4

64

0.301

17.462

H20-1.6T_TP4

32

0.129

7.463

H20-1.6T_TP4

64

0.054

3.132

H20-800G_TP4

32

-0.118

-6.866

H20-800G_TP4

64

-0.440

-25.526

H20_3e-3.2T_TP8

32

0.185

10.733

H20_3e-3.2T_TP8

64

0.167

9.672

H20_3e-3.2T_TP8

128

0.102

5.927

H20_3e-1.6T_TP8

32

0.062

3.568

H20_3e-1.6T_TP8

64

-0.080

-4.657

H20_3e-1.6T_TP8

128

-0.392

-22.732

H20_3e-800G_TP8

32

-0.186

-10.761

H20_3e-800G_TP8

64

-0.574

-33.316

H20_3e-800G_TP8

128

-1.380

-80.050

结论为800G实例无法满足需求, 1.6T实例仍然有很大的通信余量. 但是对于H20_3e的实例还是需要配置3.2T的网络, 以保证更大的BatchSize下的性能, 但是这里又存在一个成本核算的问题.另外从timebudget的角度来看, 网卡等静态延迟的影响可以忽略不计.

### 4. 小结

本文通过对计算量/内存带宽/网络带宽等几方面的约束,详细的逆向分析了DeepSeek-R1在H800和H20上的性能. H800最佳部署即为官方的EP144方案, 分析数据和官方数据基本一致. 另外对于H800的Overlap时间预算来分析, 必须要使用IBGDA来降低延迟.

而H20部署中, 我们发现由于算力的约束使得MLA计算缓慢需要通过TP并行加速, 但是TP过大时又会因为大量的KVCache的占用导致batchsize受限. 此时H20-3E(141GB)的版本显示出了额外的性能收益. 另外我们还对H20的互联带宽进行了评估, 在EP并行实现恰当时, 1.6Tbps带宽即可满足需求, 但是当BatchSize=64时, 依旧需要3.2T的网络.

另一方面关于TPS, 由于采用了DeepSeek线上的平均Seq-len = 4383估计, 如果平均Seq-len=2000时, 性能数据如下所示:

H800_TP1

H800_TP1

H800_TP1

H800_TP4

H20_TP4

H20_TP4

H20_TP4

H20_TP4

H20_3e_TP8

H20_3e_TP8

H20_3e_TP8

H20_3e_TP8

BatchSize

32.000

64.000

128.000

256.000

32.000

64.000

128.000

256.000

32.000

64.000

128.000

256.000

DenseMLA

0.098

0.196

0.392

0.785

0.530

1.060

2.119

4.239

0.526

1.052

2.105

4.209

DenseMLP

0.019

0.038

0.075

0.151

0.126

0.252

0.504

1.008

0.126

0.252

0.504

1.008

SparseMLA

0.098

0.196

0.392

0.681

0.306

0.455

0.753

1.349

0.239

0.320

0.484

0.812

Combine

0.082

0.165

0.329

0.659

0.082

0.165

0.329

0.659

0.082

0.165

0.329

0.659

SharedExpert

0.010

0.014

0.017

0.024

0.070

0.093

0.112

0.160

0.070

0.093

0.112

0.160

RoutedExpert

0.084

0.112

0.134

0.191

0.560

0.747

0.896

1.280

0.560

0.747

0.896

1.280

Dispatch

0.041

0.082

0.165

0.329

0.041

0.082

0.165

0.329

0.041

0.082

0.165

0.329

COMP_SUM

0.192

0.322

0.543

0.897

0.936

1.295

1.761

2.789

0.869

1.161

1.493

2.252

COMM_SUM

0.124

0.247

0.494

0.988

0.124

0.247

0.494

0.988

0.124

0.247

0.494

0.988

TPOT(Overlap)

11.505

19.366

32.901

54.822

56.248

79.043

110.015

177.527

52.342

71.232

94.391

146.281

TPOT

18.670

33.695

61.560

112.140

63.412

93.372

138.673

234.845

59.507

85.561

123.050

203.598

TPS_O

86.918

51.637

30.395

18.241

17.779

12.651

9.090

5.633

19.105

14.039

10.594

6.836

TPS

53.562

29.678

16.244

8.917

15.770

10.710

7.211

4.258

16.805

11.688

8.127

4.912

Total_O

2781.372

3304.787

3890.496

4669.646

568.913

809.685

1163.482

1442.035

611.366

898.479

1356.054

1750.059

Total

1713.996

1899.377

2079.289

2282.865

504.634

685.427

923.032

1090.083

537.756

748.005

1040.225

1257.377

当然这一版本还没有考虑折算单卡专家数量不一致导致的GEMM访问内存效率问题,

参考资料

[1] 
DeepSeek-V3 / R1 推理系统概览: *https://zhuanlan.zhihu.com/p/27181462601*
[2] 
DeepSeek V3/R1 推理效率分析（2）: DeepSeek 满血版逆向工程分析: *https://zhuanlan.zhihu.com/p/29841050824*