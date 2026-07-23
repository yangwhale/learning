# DeepSeek 论文合集（第一篇 → 最新）

DeepSeek-AI 官方论文全收藏，共 **23 篇**，从 2024-01 的 DeepSeek LLM 到 2026-01 的 DeepSeek-OCR 2。PDF 存放在 [`pdfs/`](./pdfs/)，均从 arXiv 下载，文件名按 `年月_名称_arxivID.pdf` 命名（按文件名排序即时间线）。

## 系列速览

| 系列 | 论文 |
|---|---|
| **主线 LLM** | DeepSeek LLM → DeepSeekMoE → V2 → V3 → R1 → V3.2 |
| **代码** | DeepSeek-Coder → Coder-V2 |
| **数学 / 定理证明** | DeepSeekMath → Prover → Prover-V1.5 → Prover-V2 → Math-V2 |
| **多模态** | DeepSeek-VL → Janus → JanusFlow → VL2 → Janus-Pro |
| **架构 / 系统** | ESFT · Fire-Flyer AI-HPC · NSA |
| **OCR / 光学压缩** | DeepSeek-OCR → OCR 2 |

## 时间线（23 篇）

| 日期 | 论文 | 系列 | arXiv | 一句话 |
|---|---|---|---|---|
| 2024-01 | DeepSeek LLM | LLM | [2401.02954](https://arxiv.org/abs/2401.02954) | scaling law 指导的开源 LLM 奠基之作 |
| 2024-01 | DeepSeekMoE | LLM | [2401.06066](https://arxiv.org/abs/2401.06066) | 细粒度专家 + 共享专家隔离的 MoE 架构 |
| 2024-01 | DeepSeek-Coder | 代码 | [2401.14196](https://arxiv.org/abs/2401.14196) | 仓库级预训练 + FIM 的代码 LLM |
| 2024-02 | DeepSeekMath | 数学 | [2402.03300](https://arxiv.org/abs/2402.03300) | 数学推理，提出 **GRPO**（后续 R1 基石）|
| 2024-03 | DeepSeek-VL | 多模态 | [2403.05525](https://arxiv.org/abs/2403.05525) | 第一代视觉语言模型 |
| 2024-05 | DeepSeek-V2 | LLM | [2405.04434](https://arxiv.org/abs/2405.04434) | 提出 **MLA**（多头潜在注意力）+ DeepSeekMoE，经济高效 |
| 2024-05 | DeepSeek-Prover | 数学 | [2405.14333](https://arxiv.org/abs/2405.14333) | Lean 定理证明 + 大规模合成数据 |
| 2024-06 | DeepSeek-Coder-V2 | 代码 | [2406.11931](https://arxiv.org/abs/2406.11931) | MoE 代码模型，逼近闭源 |
| 2024-07 | ESFT | 架构 | [2407.01906](https://arxiv.org/abs/2407.01906) | 专家专精微调，MoE 参数高效微调 |
| 2024-08 | DeepSeek-Prover-V1.5 | 数学 | [2408.08152](https://arxiv.org/abs/2408.08152) | 证明助手反馈 + RL + MCTS |
| 2024-08 | Fire-Flyer AI-HPC | 系统 | [2408.14158](https://arxiv.org/abs/2408.14158) | 自建万卡集群软硬件协同设计 |
| 2024-10 | Janus | 多模态 | [2410.13848](https://arxiv.org/abs/2410.13848) | 解耦视觉编码的统一多模态理解 + 生成 |
| 2024-11 | JanusFlow | 多模态 | [2411.07975](https://arxiv.org/abs/2411.07975) | 自回归 + rectified flow 统一多模态 |
| 2024-12 | DeepSeek-VL2 | 多模态 | [2412.10302](https://arxiv.org/abs/2412.10302) | MoE 视觉语言模型 |
| 2024-12 | **DeepSeek-V3** | LLM | [2412.19437](https://arxiv.org/abs/2412.19437) | 671B MoE：MLA + MTP + FP8 训练 + aux-loss-free 负载均衡 |
| 2025-01 | **DeepSeek-R1** | LLM | [2501.12948](https://arxiv.org/abs/2501.12948) | 纯 RL 激励推理（R1-Zero）+ 冷启动 SFT |
| 2025-01 | Janus-Pro | 多模态 | [2501.17811](https://arxiv.org/abs/2501.17811) | Janus 升级：数据 + 规模优化 |
| 2025-02 | NSA | 架构 | [2502.11089](https://arxiv.org/abs/2502.11089) | 原生稀疏注意力，硬件对齐 + 可训练 |
| 2025-04 | DeepSeek-Prover-V2 | 数学 | [2504.21801](https://arxiv.org/abs/2504.21801) | 形式化数学推理 RL |
| 2025-10 | DeepSeek-OCR | OCR | [2510.18234](https://arxiv.org/abs/2510.18234) | 上下文光学压缩（视觉 token 压文本）|
| 2025-11 | DeepSeekMath-V2 | 数学 | [2511.22570](https://arxiv.org/abs/2511.22570) | 自验证数学推理 |
| 2025-12 | **DeepSeek-V3.2** | LLM | [2512.02556](https://arxiv.org/abs/2512.02556) | **DSA** 稀疏注意力，效率 + 推理 + agent |
| 2026-01 | DeepSeek-OCR 2 | OCR | [2601.20552](https://arxiv.org/abs/2601.20552) | Visual Causal Flow |

> 一句话摘要为个人理解概述，非论文原文。论文版权归 DeepSeek-AI 及各作者所有，PDF 仅作个人学习收藏，引用请以 arXiv 原文为准。
