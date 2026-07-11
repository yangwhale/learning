# Transformer Circuits Study Notes

[transformer-circuits.pub](https://transformer-circuits.pub/) — Anthropic Mechanistic Interpretability Team

## What is this?

Anthropic's research platform dedicated to reverse-engineering transformer internals. The team treats language models like biological specimens — dissecting attention heads, tracing information flow through residual streams, and extracting interpretable features from production-scale models.

## Repository Structure

```
transformer-circuits/
├── README.md              # This file — index & reading guide
├── papers/                # Paper summaries & key takeaways
│   ├── 2021-mathematical-framework.md
│   ├── 2022-toy-models-superposition.md
│   ├── 2022-induction-heads.md
│   ├── 2023-towards-monosemanticity.md
│   ├── 2024-scaling-monosemanticity.md
│   ├── 2025-biology-of-llm.md
│   ├── 2025-circuit-tracing.md
│   └── 2026-global-workspace.md
├── concepts/              # Core concepts & mental models
│   └── glossary.md
└── notes/                 # Our learning notes & insights
    └── .gitkeep
```

## Recommended Reading Order

### Main Quest (read in order)

| # | Paper | Year | One-line Summary |
|---|-------|------|-----------------|
| 1 | [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | Residual stream as communication bus; attention heads as independent circuits |
| 2 | [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 2022 | Induction heads are the primary mechanism behind in-context learning |
| 3 | [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) | 2022 | Why neurons encode multiple unrelated concepts (feature > neuron) |
| 4 | [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | 2023 | Sparse Autoencoders extract interpretable features from a 1-layer transformer |
| 5 | [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) | 2024 | Million-scale features from Claude 3 Sonnet, including safety-relevant ones |
| 6 | [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | 2025 | Full-scale mechanistic investigation of Claude 3.5 Haiku |
| 7 | [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | 2025 | Method to trace step-by-step computation graphs in language models |
| 8 | [Verbalizable Representations Form a Global Workspace](https://transformer-circuits.pub/2026/global-workspace/index.html) | 2026 | Claude maintains a privileged set of representations it can self-report on |

### Side Quests (read as needed)

| Paper | Year | Topic |
|-------|------|-------|
| [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html) | 2022 | Activation function that increases interpretable neurons |
| [Privileged Bases in the Transformer Residual Stream](https://transformer-circuits.pub/2023/privileged-bases/index.html) | 2023 | Adam optimizer creates privileged coordinate directions |
| [Superposition, Memorization, and Double Descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html) | 2023 | Mechanistic understanding of overfitting |
| [Sparse Crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html) | 2024 | Cross-layer features and model diffing |
| [Natural Language Autoencoders](https://transformer-circuits.pub/2026/natural-language-autoencoders/index.html) | 2026 | Train Claude to translate its internal state into natural language |
| [Emotion Concepts](https://transformer-circuits.pub/2026/emotion-features/index.html) | 2026 | Emotion representations in Claude that causally influence outputs |

## Key Concepts

- **Residual Stream**: The central communication bus in transformers. All layers read from and write to it.
- **Superposition**: Models store more features than they have neurons by encoding features in overlapping directions.
- **Induction Heads**: Attention heads that implement pattern completion (A B ... A → B). Primary driver of in-context learning.
- **Sparse Autoencoder (SAE)**: Tool to decompose superposed activations into interpretable monosemantic features.
- **Circuit**: A subgraph of the model's computation that implements a specific behavior.
- **Feature**: A direction in activation space that corresponds to a human-interpretable concept.
- **Polysemanticity**: One neuron responding to multiple unrelated concepts (the problem SAEs solve).
- **Transcoders**: An alternative to SAEs that maps MLP inputs to outputs, decomposing computation rather than representation.
- **Attribution Graph**: A directed graph showing how features interact across layers to produce a specific output.

## Why This Matters for Infra

Understanding model internals informs infrastructure decisions:

- **Quantization**: Superposition theory explains why some layers tolerate aggressive quantization while others don't — layers with higher superposition density are more sensitive to precision loss.
- **KV Cache**: Induction heads explain why KV cache is critical for quality — without cached keys, the model loses its primary pattern-matching mechanism.
- **Pipeline Parallelism**: The residual stream as communication bus means pipeline cut points interrupt information flow — optimal splits should minimize cross-boundary feature dependencies.
- **Pruning**: SAE-extracted features can identify which attention heads and MLP neurons are safety-critical vs. expendable.
- **Sparse Attention**: Understanding which heads do what (induction, retrieval, positional) guides which heads can use sparse patterns without quality loss.
