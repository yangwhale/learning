# MoE 环游记 · 深度解读

> 苏剑林（Jianlin Su）"MoE 环游记"系列全文精读与深度解析

## 关于本系列

苏剑林是 RoPE（Rotary Position Embedding）的发明者，现任月之暗面（Moonshot AI）研究员。他在个人博客"科学空间"（kexue.fm）上发表的"MoE 环游记"系列，是中文社区对 Mixture-of-Experts 架构最系统、最深入的技术分析。

该系列从 2025 年 2 月启动，至 2026 年 6 月已连载 9 篇正篇 + 1 篇番外，横跨一年半。其中第 6 篇提出的 **Quantile Balancing** 算法被 Kimi K3（2.8T 参数）直接采用，成为其 MoE 训练的核心负载均衡方案。

## 系列文章索引

| # | 标题 | 日期 | 核心主题 | 解读文档 |
|---|------|------|----------|----------|
| 1 | [从几何意义出发](https://kexue.fm/archives/10699) | 2025-02-08 | 从 Dense 模型的最佳逼近推导 MoE | [01-geometric-interpretation.md](01-geometric-interpretation.md) |
| 2 | [不患寡而患不均](https://kexue.fm/archives/10735) | 2025-02 | Aux Loss 与负载均衡的 STE 推导 | [02-aux-loss.md](02-aux-loss.md) |
| 3 | [换个思路来分配](https://kexue.fm/archives/10757) | 2025-03-05 | DeepSeek Loss-Free 方案解读 | [03-loss-free.md](03-loss-free.md) |
| 4 | [难处应当多投入](https://kexue.fm/archives/10815) | 2025-03-28 | 动态激活：难 token 多分配资源 | [04-dynamic-activation.md](04-dynamic-activation.md) |
| 5 | [均匀分布的反思](https://kexue.fm/archives/10945) | 2025-05-16 | Shared Expert 与 Fine-Grained Expert | [05-rethinking-uniform.md](05-rethinking-uniform.md) |
| 6 | [最优分配促均衡](https://kexue.fm/archives/11619) | 2026-02-22 | **Quantile Balancing 提出** | [06-quantile-balancing.md](06-quantile-balancing.md) |
| 7 | [动态激活极简解](https://kexue.fm/archives/11626) | 2026-02-23 | QB 动态版：一步 quantile 求解 | 待写 |
| 番外 | [DeepSeek V4 的 tid2eid](https://kexue.fm/archives/11750) | 2026-05-15 | Hash Routing 映射表生成 | 待写 |
| 8 | [强制序列级均衡](https://kexue.fm/archives/11760) | 2026-05-22 | 从全局均衡到序列级均衡 | 待写 |
| 9 | [门控归一化之争](https://kexue.fm/archives/11782) | 2026-06-17 | Softmax vs Sigmoid，归一化时机 | 待写 |

## 系列演进脉络

```
第一阶段：基础建设（2025.02 - 2025.03）
  #1 几何直觉 → #2 Aux Loss → #3 Loss-Free
  从"MoE 是什么"到"怎么让它均衡"

第二阶段：深度探索（2025.03 - 2025.05）
  #4 动态激活 → #5 均匀性反思
  质疑均匀分布的最优性，引入 Shared/Fine-Grained Expert

  ～ 沉寂 9 个月 ～

第三阶段：理论突破（2026.02）
  #6 Quantile Balancing → #7 动态版 QB
  线性规划对偶推导，一步 quantile 即最优解

第四阶段：工程落地（2026.05 - 2026.06）
  番外 Hash Routing → #8 序列级均衡 → #9 门控之争
  DeepSeek V4 实践 + K3 大规模训练经验
```

## 三代负载均衡方案对比

| 维度 | Aux Loss (#2) | Loss-Free (#3) | Quantile Balancing (#6) |
|------|---------------|-----------------|-------------------------|
| 核心思路 | 加惩罚项推负载分布趋于均匀 | 不改打分，加 bias 调排序 | 线性规划对偶，取分位数 |
| 超参数 | 惩罚系数（难调） | bias 学习率 γ（与激活函数耦合） | 无 |
| 均衡速度 | 慢 | 中等 | 快（尤其擅长极端情况） |
| 对模型梯度的干扰 | 有（STE 次优梯度） | 无 | 无 |
| 适用激活函数 | 任意（但 γ 需重调） | 与 Sigmoid 耦合 | 任意 |
| 提出者 | GShard (Google, 2020) | DeepSeek (2024) | 苏剑林 (Moonshot, 2026) |
| 采用模型 | 几乎所有早期 MoE | DeepSeek V3, Kimi K2 | **Kimi K3** |

## 阅读建议

- 如果只想了解 K3 用了什么：直接读 #6 和 #7
- 如果想系统理解 MoE 负载均衡演进：按 #2 → #3 → #6 的顺序
- 如果想从头建立 MoE 直觉：从 #1 开始按顺序读
- 每篇解读文档独立成文，但会标注与前后文的关联
