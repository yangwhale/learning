# A Mathematical Framework for Transformer Circuits (2021)

- **URL**: https://transformer-circuits.pub/2021/framework/index.html
- **Authors**: Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, et al.
- **Key Idea**: Treat transformers as computational graphs of interpretable components

## Core Contributions

1. **Residual Stream = Communication Bus**: Information flows through a shared d_model vector. Attention heads and MLPs read from it and write to it additively. This means any later layer can access any earlier layer's output — the model is NOT a strict pipeline.

2. **Attention Heads are Independent**: Each attention head operates independently (parallel, not sequential). The combined effect is additive on the residual stream. This means we can analyze each head in isolation.

3. **QK and OV Circuits**: Each attention head has two logically separate circuits:
   - QK circuit (W_Q^T W_K): determines WHERE to attend
   - OV circuit (W_O W_V): determines WHAT information to move
   These can be analyzed as low-rank matrices.

4. **Virtual Weights**: When an attention head in layer L reads the output of an MLP in layer L-1, the effective weight matrix is a composition (product) of both layers' weights. These "virtual weights" can be analyzed directly to understand cross-layer interactions.

5. **Zero-Layer, One-Layer, Two-Layer Analysis**: Progressively analyze:
   - 0-layer: just token embeddings → unigram statistics
   - 1-layer: attention heads do bigram + skip-trigram statistics
   - 2-layer: induction heads emerge (the key finding)

## Takeaways

- This paper sets the mathematical language for everything that follows.
- The residual stream framing is essential — without it, you can't meaningfully decompose model behavior.
- "Feature > Neuron" is already implicit here: the residual stream directions that matter aren't aligned with neuron axes.

## Reading Notes

- Dense but foundational. Read sections 1-4 carefully, skim section 5 (specific circuits in small models).
- The "QK/OV separation" insight is used constantly in later papers.
