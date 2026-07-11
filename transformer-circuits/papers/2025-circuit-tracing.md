# Circuit Tracing: Revealing Computational Graphs in Language Models (2025)

- **URL**: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- **Authors**: Emmanuel Ameisen, Jack Lindsey, Adly Templeton, Tom Conerly, Chris Olah, Joshua Batson, et al.
- **Local**: [HTML](../pdfs/2025-circuit-tracing.html) (no arXiv version)
- **Key Idea**: A scalable method to trace complete computation graphs — from features to full circuits

## The Gap This Paper Fills

By 2024, we could extract millions of interpretable features from a model. But features in isolation aren't enough. Knowing "there's a Golden Gate Bridge feature" doesn't tell you:
- How does information flow from input → this feature → output?
- Which other features feed into it? Which features does it influence?
- When the model does multi-step reasoning, what's the step-by-step circuit?

We need the **wiring diagram**, not just the parts list.

## The Method: Replacement Model + Autodiff

### Step 1: Replace every component with its interpretable decomposition

```
Original model:  Embed → Attn1 → MLP1 → Attn2 → MLP2 → ... → Unembed

Replacement:     Embed → SAE(Attn1) → TC(MLP1) → SAE(Attn2) → TC(MLP2) → ... → Unembed
                          ↑ features    ↑ features   ↑ features   ↑ features
```

- **Attention layers** → replaced by SAE decomposition (features of attention output)
- **MLP layers** → replaced by **Transcoder** decomposition (features of MLP computation)

### Step 2: Differentiate through the replacement model

Once every component is expressed in terms of features, compute gradients to get feature-to-feature causal influence:

```
Feature A (layer 3) ──[gradient = 0.7]──→ Feature B (layer 5) ──[gradient = 0.4]──→ Output logit

This means: A causally influences B with strength 0.7, and B influences the output with strength 0.4.
```

### Step 3: Threshold and visualize

Only keep edges above a causal influence threshold. The result: a sparse directed graph — the **attribution graph**.

## Why Transcoders, Not SAEs, for MLPs

This is a key technical distinction:

| Tool | What it does | Analogy |
|---|---|---|
| SAE on MLP | Encode-decode MLP activations. Captures what the representation LOOKS LIKE. | Photographing factory products |
| Transcoder on MLP | Map MLP input → MLP output. Captures what the MLP COMPUTES. | Cameras on the production line |

For building causal graphs, you need to know what each component DOES (computation), not just what it OUTPUTS (representation). Transcoders decompose the computation itself into interpretable steps.

**Analogy expanded**: If you want to draw the circuit diagram of a factory, product photos tell you what comes out at each stage, but production line footage tells you what each machine does to its input. The circuit diagram needs the machine operations, not the product snapshots.

## Faithfulness: Does the Replacement Model Actually Work?

The replacement model (with SAEs/transcoders swapped in) must behave similarly to the original. If swapping in decompositions breaks behavior, the attribution graph is meaningless.

They measure this by running both models on the same inputs and comparing:
- Output probability distributions
- Specific behavioral tests (reasoning, factual recall, etc.)

The replacement model achieves high faithfulness on most behaviors — the decomposition preserves the important computations.

## Scalability

This works on **Claude 3.5 Haiku** (~8B parameters). Key engineering that makes it tractable:

1. **Sparse activations**: Most features are inactive at any time (thanks, superposition!), so the graph is manageable
2. **Thresholding**: Only trace features above an activation threshold — ignore weak signals
3. **Automatic differentiation**: Use standard ML framework autodiff, not brute-force ablation experiments (which would require O(features²) forward passes)

## Interactive Explorer

The paper includes an interactive visualization tool at the companion page. You can:
- Click on any feature node to see its upstream (what influences it) and downstream (what it influences)
- Trace paths from input tokens to output predictions
- Zoom into specific computation steps

This is the "oscilloscope" for language models.

## Infrastructure Implications

- **Mechanism-based debugging**: Model behaving wrong? Open the attribution graph, find where information flow breaks. Like using an oscilloscope to debug a circuit instead of guessing from outputs.
- **Targeted fine-tuning**: If a specific circuit is misbehaving, you can identify exactly which features/weights are involved and fine-tune only those. Much more surgical than full-model fine-tuning.
- **Compute cost of tracing**: Building attribution graphs requires running SAEs + transcoders + backward pass. Not cheap for production, but valuable for offline analysis and debugging.
- **Model surgery**: With the full circuit diagram, you could theoretically remove or modify specific circuits (e.g., remove a biased reasoning pathway) without retraining.

## Reading Notes

- This is the METHODS paper. Read alongside "Biology of a Large Language Model" for the applications.
- The SAE vs. transcoder distinction matters a lot — if you only remember one thing, remember that transcoders decompose computation, SAEs decompose representation.
- The interactive visualization at https://transformer-circuits.pub/2025/attribution-graphs/index.html is worth exploring hands-on.
- This paper enables everything in the Biology paper. Without the tracing method, the detailed case studies wouldn't be possible.
