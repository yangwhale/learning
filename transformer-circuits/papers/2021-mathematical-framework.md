# A Mathematical Framework for Transformer Circuits (2021)

- **URL**: https://transformer-circuits.pub/2021/framework/index.html
- **Authors**: Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, et al.
- **Local**: [HTML](../pdfs/2021-mathematical-framework.html) (no arXiv version)
- **Key Idea**: Treat transformers as computational graphs of interpretable components

## The Problem

We have this thing called a transformer, it works incredibly well, but we have zero idea what's happening inside. It's like a precision instrument with no labels — you have to figure out what every part does yourself. This paper draws the first parts diagram.

## Core Insights

### 1. Residual Stream = Shared Bus

The traditional view: information flows layer by layer, each layer processes and passes forward.

This paper's view: **wrong**. The residual stream is a shared blackboard. Every attention head and MLP reads from the blackboard, does some computation, and **adds** the result back. Not overwrites — adds.

Why this matters: Layer 10 can directly read what Layer 1 wrote, because it's all on the same blackboard. The skip connection isn't an engineering trick — it IS the core communication mechanism. This explains why transformers are so good at long-range information transfer.

```
Traditional view:  Input → Layer1 → Layer2 → ... → Output  (pipeline)

Correct view:      ┌─────── Residual Stream (shared blackboard) ──────┐
                   │  ↑write  ↑write    ↑write   ↑write              │
                   │  ↓read   ↓read     ↓read    ↓read               │
                   │ Attn1   MLP1      Attn2    MLP2     → Output    │
                   └──────────────────────────────────────────────────┘
```

### 2. Attention Heads are Independent

Multiple attention heads within the same layer operate **completely in parallel**. They each independently read from the residual stream, compute, and add back. Their contributions are additive.

Implication: you can analyze each head in isolation. You don't need to understand all 32 heads to understand one of them. This massively simplifies the analysis.

### 3. QK and OV Circuit Separation (the most elegant insight)

Each attention head contains two logically independent sub-circuits:

- **QK circuit** (W_Q^T · W_K): decides **WHERE** to attend. "Which other position should I look at?"
- **OV circuit** (W_O · W_V): decides **WHAT** to move. "Once I'm looking there, what information do I bring back?"

**Analogy**: QK is your eyes (deciding where to look), OV is your hands (deciding what to grab). Eyes and hands can be analyzed separately — where you look is independent of what you pick up.

These can be analyzed as low-rank matrices, which makes the math tractable.

### 4. Virtual Weights

When a Layer 2 attention head reads the output of Layer 1's MLP, you can multiply their weight matrices together to get an **effective weight matrix**. This "virtual weight" describes the cross-layer end-to-end computation.

You don't have to analyze each layer separately — you can directly study the combined effect. This is crucial for understanding multi-layer circuits like induction heads.

### 5. Progressive Analysis: 0-Layer → 1-Layer → 2-Layer

The paper builds understanding from simple to complex:

| Layers | Capability | Mechanism |
|--------|-----------|-----------|
| 0-layer | Unigram statistics only | Just embedding → unembedding, learns word frequencies |
| 1-layer | Bigrams + skip-trigrams | Attention copies previous token info to current position |
| 2-layer | **Induction heads emerge** | Two heads compose: head1 shifts info backward, head2 uses it for pattern matching |

The 2-layer result is where the magic happens — this is where in-context learning first appears, and it's the seed for the Induction Heads paper.

## Infrastructure Implications

- **Quantization**: The residual stream framework tells you that quantizing an intermediate layer's precision affects ALL subsequent layers, not just the next one. Information propagates through the shared bus, not a pipeline.
- **Pipeline Parallelism**: Choosing where to cut is choosing where to sever the shared bus. You need to consider which cross-layer communications you're interrupting, not just load balancing.
- **KV Cache**: The QK/OV separation explains why KV cache compression affects WHERE the model attends (QK) independently of WHAT information it retrieves (OV). You can potentially compress them with different strategies.
- **Pruning Heads**: Since heads are independent and additive, removing a head is a clean subtraction from the residual stream. The effect is predictable and localized.

## Reading Notes

- Dense but foundational. Read sections 1-4 carefully, skim section 5 (specific circuits in small models).
- The QK/OV separation is used constantly in every subsequent paper.
- "Feature > Neuron" is already implicit here: the residual stream directions that matter aren't aligned with neuron axes.
- This paper's mathematical language becomes the lingua franca of the entire field.
